# %%
# %%capture
# aaa
import time
import random
import torch
import optuna
import os
import re
import gc
import shap
import logging
import optuna

import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from pandas.api.types import is_integer_dtype as is_integer
from pandas.api.types import is_float_dtype as is_float
from google.cloud import bigquery
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import auc, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, make_scorer, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# %% [markdown]
# ### Callable Functions

# %%
def calculate_metrics(baseline):
    # Probabilities of the positive class
    y_pred_test = baseline.predict_proba(X_test)[:, 1]
    y_pred_val = baseline.predict_proba(X_val)[:, 1]

    # Resetting the index of y_test for alignment
    y_test_reset = y_test.reset_index(drop=True)

    # ROC AUC Score
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    # Sorting indices by predicted probabilities in descending order
    idx = np.argsort(y_pred_test)[::-1]

    # Function to calculate metrics for a given cutoff percentile
    def calculate_percentile_metrics(y_test_reset, idx, percentile):
        cutoff = int(percentile * len(y_test_reset))  # Number of data points in the top percentile
        top_indices = idx[:cutoff]  # Indices of the top percentile predictions

        # Binary predictions for the top percentile
        predictions_binary = np.zeros(len(y_test_reset), dtype=int)
        predictions_binary[top_indices] = 1

        # Confusion matrix calculation
        tn, fp, fn, tp = confusion_matrix(y_test_reset, predictions_binary, labels=[0, 1]).ravel()

        # Performance metrics
        ppv = 100 * (tp / (tp + fp)) if (tp + fp) > 0 else 0  # Positive Predictive Value
        sensitivity = 100 * (tp / (tp + fn)) if (tp + fn) > 0 else 0  # Sensitivity or Recall

        # Lift calculation
        actual_positives_top_perc = y_test_reset[top_indices].sum()
        expected_positives = y_test_reset.sum() * percentile
        lift = actual_positives_top_perc / expected_positives if expected_positives > 0 else 0

        return lift, ppv, sensitivity

    lift_1_perc, ppv_1_perc, sensitivity_1_perc = calculate_percentile_metrics(y_test_reset, idx, 0.01)
    lift_10_perc, ppv_10_perc, sensitivity_10_perc = calculate_percentile_metrics(y_test_reset, idx, 0.10)

    return roc_auc_test, lift_1_perc, lift_10_perc, ppv_1_perc, sensitivity_1_perc, ppv_10_perc, sensitivity_10_perc

# %%
def check_label_distribution(y):
    # Count the number of occurrences of each label 
    count_0 = (y == 0).sum()
    count_1 = (y == 1).sum()
    ratio = count_1 / count_0 if count_0 != 0 else np.inf
    print(count_0, count_1, ratio)
    return count_0, count_1, ratio

# %%
def plot_lift_curve(y_val, y_pred, step=0.01):
    
    #Define an auxiliar dataframe to plot the curve
    aux_lift = pd.DataFrame()
    #Create a real and predicted column for our new DataFrame and assign values
    aux_lift['real'] = y_val
    aux_lift['predicted'] = y_pred
    #Order the values for the predicted probability column:
    aux_lift.sort_values('predicted',ascending=False,inplace=True)
    
    #Create the values that will go into the X axis of our plot
    x_val = np.arange(step,1+step,step)
    #Calculate the ratio of ones in our data
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    #Create an empty vector with the values that will go on the Y axis our our plot
    y_v = []
    
    #Calculate for each x value its correspondent y value
    for x in x_val:
        num_data = int(np.ceil(x*len(aux_lift))) #The ceil function returns the closest integer bigger than our number 
        data_here = aux_lift.iloc[:num_data,:]   # ie. np.ceil(1.4) = 2
        ratio_ones_here = data_here['real'].sum()/len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)
           
   #Plot the figure
    fig, axis = plt.subplots()
    fig.figsize = (40,40)
    axis.plot(x_val, y_v, 'g-', linewidth = 3, markersize = 5)
    axis.plot(x_val, np.ones(len(x_val)), 'k-')
    axis.set_xlabel('Proportion of sample')
    axis.set_ylabel('Lift')
    plt.title('Lift Curve')
    plt.show()

# %%
def lift_chart(X,actual_target,model):
    avg_tgt = actual_target.sum()/len(actual_target)
    df_data = X.copy()
    X_data = df_data.copy()
    df_data['Actual'] = actual_target
    df_data['Predict']= model.predict(X_data)
    y_Prob = pd.DataFrame(model.predict_proba(X_data))
    df_data['Prob_1']=list(y_Prob[1])
    df_data.sort_values(by=['Prob_1'],ascending=False,inplace=True)
    df_data.reset_index(drop=True,inplace=True)
    df_data['Percentile']=pd.qcut(df_data.index,100,labels=False)
    output_df = pd.DataFrame()
    grouped = df_data.groupby('Percentile',as_index=False)
    output_df['Max_Scr']=grouped.max().Prob_1
    output_df['Min_Scr']=grouped.min().Prob_1
    output_df['Actual']=grouped.sum().Actual
    output_df['Total']=grouped.count().Actual
    output_df["Population_perc"] = (output_df["Total"]/len(actual_target))*100
    output_df['Per_Events'] = (output_df['Actual']/output_df['Actual'].sum())*100
    output_df['Gain_Percentage'] = output_df.Per_Events.cumsum()
    output_df["Cumulative_Population"] = output_df.Population_perc.cumsum()
    output_df["Lift"] = output_df["Gain_Percentage"]/output_df["Cumulative_Population"]
    return output_df

# %%
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

client = bigquery.Client()

# %% [markdown]
# ### Parameter setting for script

# %%
target_names = ['without outcome', 'with outcome', ] # used to pass labels to confusion matrices in metrics section

threshold = 0.25 # custom decision point threshold if you don't want to go with default 0.5

model_name = "IP_all_Medicaid_XGBoost" # Model name for use in plots

one_hot_backup = "preprocessed_data_name_here.feather" #back up computationally intensive step so it only needs to be completed once

outcome_var = "HCC_flag" #name the variable that we are training to predict so we can pull this column as our y automatically anywhere pertinent

random_st = 53 # set a seed for replicability

feature_vars = 4:260 # column indices for the features you want to be used for creating the test/train/val X matrices

indexing_var = 'asdb_member_key' #name of the column to index on (i.e. individual_id)

#vars for undersampling - can add or remove values based on your sample's specific ratio of rare outcome
ratios = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # Undersampling ratios
seeds = [53]  # Random seeds for reproducibility

save_model_nm = 'Model_specific_name.cbm' #name you want the model to be saved under

# %% [markdown]
# ### Load Data

# %%
# Get outcome
sql = """
SELECT

FROM 
    ``
WHERE 1 = 1
"""
outcome = client.query(sql).to_dataframe() 

outcome.shape

# %%
outcome.head()

# %%
# Get features
sql = """
SELECT

FROM
    ``
WHERE 1 = 1
"""
features = client.query(sql).to_dataframe() 
features.shape

# %%
features.head()

# %%
# If you need to rename your ID columns for one or both tables, do so here
outcome = outcome.rename(columns = {"emr_id": "individual_id"})

# %%
#set null embeddings to 0 (standard protocol)
emb_pattern = r'emb[0-255]+'
emb_col = [col for col in df.columns if re.match(emb_pattern ,col)]
df[emb_col] = df[emb_col].fillna(0)
for c in df.columns: 
    dt = df[c].dtype
    if is_integer(dt) or is_float(dt):
        df[c] = df[c].fillna(0) 
        # print("Floatint:", dt)
    else:
        try:
            df[c] = df[c].fillna('')
        except:
            print("ERROR - DATE VARIABLE FOUND", dt)

# %% [markdown]
# ### One hot encode (if needed)

# %%
#  0 := categorical, 1 := continuous, 2 := binary
# example variables
nem_to_type = {
    'narc':2, 
    'urbsubr':0, 
    'gender':0, # TODO: IF = M, 1 ELIF = F 0 ELSE NULL 
    'tenure_yr1':1,
}

# %%
index_to_feature = dict(enumerate(features.columns))
feature_to_index = {value: key for key, value in index_to_feature.items()}
categorical_features = [feature for feature in nem_to_type if nem_to_type[feature] == 0]
categorical_indices = [feature_to_index[feature] for feature in categorical_features if feature in feature_to_index]
len(categorical_features)

# %%
categorical_features.remove('index_dt')
features[categorical_features] = features[categorical_features].astype(str)
features['gender'] = features['gender'].map({'M': 1, 'F': 0}).fillna(-1)  
categorical_features.remove('gender')

# %%
# ONE HOT ENCODING
from tqdm.notebook import tqdm
from sklearn.preprocessing import OneHotEncoder
min_occurrence = 2000

encoder = OneHotEncoder(sparse_output=False)

for feature in tqdm(categorical_features):
    counts = features[feature].value_counts()
    categories_to_keep = counts[counts >= min_occurrence].index
   
    filtered_df = features[features[feature].isin(categories_to_keep)]
   
    if not filtered_df.empty:
        encoded_data = encoder.fit_transform(filtered_df[[feature]])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([feature]))
       
        features = features.drop(feature, axis=1)
        features = pd.concat([features, encoded_df], axis=1)

# %%
#Set index to member ID for easy down-stream tracking
features.set_index(indexing_var, inplace = True)
outcome.set_index(indexing_var, inplace = True)
df = outcome.merge(features, on=indexing_var, how='left')

# %%
df.to_feather(one_hot_backup)

# %% [markdown]
# ### Create test/train/validation split

# %%
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, feature_vars], df.loc[:, outcome_var], test_size = 0.2, random_state = random_st, stratify = df.loc[:, outcome_var])
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = random_st, stratify = y_test)

# %%
X_train.shape

# %%
y_train = y_train.astype('int')
y_val = y_val.astype('int')
y_test = y_test.astype('int')

# %% [markdown]
# ### Baseline untuned model

# %%
baseline = xgb.XGBClassifier(n_estimators = 2, max_depth = 2, learning_rate = 1, objective='binary:logistic')
baseline.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = 0)

# %%
roc, lift_1_perc, lift_10_perc, ppv_1_perc, sensitivity_1_perc, ppv_10_perc, sensitivity_10_perc = calculate_metrics(baseline)
print("ROC: ",roc)
print("1% Lift: ",lift_1_perc)
print("1% PPV: ",ppv_1_perc)
print("1% Sensitivity: ",sensitivity_1_perc)
print("10% Lift: ",lift_10_perc)
print("10% PPV: ",ppv_10_perc)
print("10% Sensitivity: ",sensitivity_10_perc)

# %%
explainer = shap.TreeExplainer(baseline)
explanation = explainer(X_test)

pred = baseline.predict(X_test)

shap_values = explanation.values
# make sure the SHAP values add up to marginal predictions
np.abs(shap_values.sum(axis=1) + explanation.base_values - pred).max()
shap.plots.beeswarm(explanation)

# %% [markdown]
# ### Undersampling optimization

# %%
results = []  # Store results

for r in ratios:
    for s in seeds:
        # Setup undersampler
        undersample = RandomUnderSampler(sampling_strategy=r, random_state=s)
        X_train_u, y_train_u = undersample.fit_resample(X_train, y_train)

        # Initialize and train XGBoost model
        xgb_model = XGBClassifier(seed = random_st, n_jobs = 15, verbosity = 0, enable_categorical = True)
        xgb_model.fit(X_train_u, y_train_u, eval_set = [(X_val, y_val)], verbose = False)
        print("done fit")

        # Predict probabilities
        y_pred_val = xgb_model.predict_proba(X_val)[:, 1]
        y_pred_test = xgb_model.predict_proba(X_test)[:, 1]

        # Calculate lift for top 1% as an example of model evaluation
        idx = np.argsort(y_pred_test)[::-1]
        top_1_percent = int(0.01 * len(y_test))  # 1% of test data size

        # Find actual positives in top 1 percent
        predicted_top_1_percent = y_test.iloc[idx][:top_1_percent]
        actual_positives_top_1_percent = predicted_top_1_percent.sum()

        # Calculate expected positives
        expected_positives = y_test.sum() * 0.01

        # Calculate lift
        lift = actual_positives_top_1_percent / expected_positives

        # Append results
        results.append((r, s, lift))
        print("one iteration", r, s, lift)

# %%
print(results)

# %%
# No undersampling
xgb_model = XGBClassifier(random_state = random_st, n_jobs = 15, verbosity = 0, enable_categorical = True)
xgb_model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose=False)

print("done fit")

# Prediction and calculate lift
y_pred_test = xgb_model.predict_proba(X_test)[:, 1]
y_test_reset = y_test.reset_index(drop = True)
idx = np.argsort(y_pred_test)[::-1]
top_1_percent = int(0.01 * len(y_test_reset))

predicted_top_1_percent = y_test_reset.iloc[idx][:top_1_percent]
actual_positives_top_1_percent = predicted_top_1_percent.sum()
expected_positives = y_test_reset.sum() * 0.01
lift = actual_positives_top_1_percent / expected_positives

results.append((0, 0, lift))
print("one iteration, no sample")

# No undersampling but with class weights
weights = np.where(y_train == 0, 1, len(y_train) / (2 * np.sum(y_train == 1)))

xgb_model = XGBClassifier(random_state=53, n_jobs=15, verbosity=0, enable_categorical=True)
xgb_model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)

print("done fit")

# Prediction and calculate lift
y_pred_test = xgb_model.predict_proba(X_test)[:, 1]
y_test_reset = y_test.reset_index(drop = True)
idx = np.argsort(y_pred_test)[::-1] 
top_1_percent = int(0.01 * len(y_test_reset))

predicted_top_1_percent = y_test_reset.iloc[idx][:top_1_percent]
actual_positives_top_1_percent = predicted_top_1_percent.sum()
expected_positives = y_test_reset.sum() * 0.01
lift = actual_positives_top_1_percent / expected_positives

results.append((999, 999, lift))
print("one iteration, class weights")

# %%
results_df = pd.DataFrame(results, columns = ['Ratio', 'Seed', '1% Lift'])

# %%
print(results)

# %%
sns.barplot(data = results_df, x = 'Ratio', y = '1% Lift')
plt.title('1% Lift Across Different Ratios and Seeds')
plt.show()

results_df.to_csv('lift_results.csv', index=False)

# %%
print("Training Set:")
check_label_distribution(y_train)

print("\nValidation Set:")
check_label_distribution(y_val)

print("\nTest Set:")
check_label_distribution(y_test)

# %% [markdown]
# ### Recursive Feature Elimination

# %%
sampling_strat = 0.2 #pick best sample weight from above

undersample = RandomUnderSampler(sampling_strategy = sampling_strat, random_state = random_st)  
steps = [
    ('u', undersample)
]
pipeline = Pipeline(steps = steps)
X_train, y_train = pipeline.fit_resample(X_train, y_train)

# %%
print("Training Set after undersampling:")
check_label_distribution(y_train)

# y_train.value_counts()
# X_train.value_counts()

# %%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%
import xgboost
print(xgboost.__version__)

# %%
# Check if GPU is available
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if not gpus:
    raise SystemError("No GPUs found. Please ensure your environment has a GPU available.")

# Initialize the GPU-enabled XGBoost classifier
xgb_model_rfe = XGBClassifier(
    random_state = rand_st,
    n_jobs = -1,
    verbosity = 0,
    use_label_encoder = False,
    tree_method = "hist", 
    device = "cuda"
)

# Create RFECV with GPU-enabled XGBoost
rfecv = RFECV(
    estimator = xgb_model_rfe,
    step = 5,
    cv = StratifiedKFold(3),
    scoring = "roc_auc",
    min_features_to_select = 1,
    n_jobs = -1
)

# Fit RFECV and manage memory
rfecv.fit(X_train, y_train)

# Print the optimal number of features
print(f"Optimal number of features: {rfecv.n_features_}")

# Transform datasets to include only the selected features
X_train_selected = rfecv.transform(X_train)
X_val_selected = rfecv.transform(X_val)
X_test_selected = rfecv.transform(X_test)

# Clear memory after fitting
gc.collect()

# %%
n_features = list(range(1, len(rfecv.cv_results_['mean_test_score']) + 1))
mean_test_scores = rfecv.cv_results_['mean_test_score']
cv_results = pd.DataFrame({
    'n_features': n_features,
    'mean_test_score': mean_test_scores
})

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.plot(cv_results["n_features"], cv_results["mean_test_score"])
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()


# %%
X_train_df = pd.DataFrame(X_train, columns = df.columns)
selected_features = X_train.columns[rfecv.support_]
print(selected_features)
len(selected_features)
with open('xgboost_selected_features.txt', 'w') as file:
    # Join the list elements into a single string with a newline character
    data_to_write = '\n'.join(selected_features)
     
    # Write the data to the file
    file.write(data_to_write)

# %%
import pandas as pd
# open file in read mode
f = open('xgboost_selected_features.txt', 'r')
selected_features = [x.strip() for x in f.readlines()]
print(len(selected_features))
f.close()

X_train_df = pd.DataFrame(X_train, columns = df.columns)

X_train = X_train_df[selected_features]

X_val_df = pd.DataFrame(X_val, columns = df.columns)

X_val = X_val_df[selected_features]

X_test_df = pd.DataFrame(X_test, columns = df.columns)

X_test = X_test_df[selected_features]

# X_test

# %% [markdown]
# ### Optuna hyperparameter tuning

# %%
logging.basicConfig(filename = 'optuna_xgboost.log', level = logging.INFO)

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 7000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 53, 
        'tree_method' : "hist", 
        'device' : "cuda",
        'n_jobs': -1
    }
   
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = 0)
   
    y_pred_test = model.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, y_pred_test)
   
    logging.info(f"Trial {trial.number} AUC: {roc_auc_test} Params: {params}")
   
    study_results = study.trials_dataframe(attrs = ('number', 'value', 'params', 'state'))  # Saving the results to a CSV file
    study_results.to_csv('optuna_results.csv', index = False)
   
    return roc_auc_test

study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 75)  # trials must be at least 50
print("Number of finished trials: ", len(study.trials))

best_trial = study.best_trial
print("Best trial:")
print(" AUC:", best_trial.value)
print(" Params:", best_trial.params)

# Log the best result
logging.info(f"Best AUC: {best_trial.value} with params: {best_trial.params}")
study_results = study.trials_dataframe(attrs = ('number', 'value', 'params', 'state'))  # Saving the results to a CSV file
study_results.to_csv('optuna_results.csv', index = False)

# %%
print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
trial_with_highest_accuracy = max(study.best_trials, key = lambda t: t.values[0])
print(f"Trial with highest accuracy: ")
print(f"\tnumber: {trial_with_highest_accuracy.number}")
print(f"\tparams: {trial_with_highest_accuracy.params}")
print(f"\tvalues: {trial_with_highest_accuracy.values}")
optuna.visualization.plot_param_importances(
    study, target = lambda t: t.values[0], target_name = "auc"
)

# %% [markdown]
# ### Build optimal model

# %%
# Optimal modelimport xgboost as xgb
import xgboost as xgb
params = {
    'random_state': 53, 
    'tree_method' : "hist", 
    'device' : "cuda",
    'n_jobs': -1,
    'learning_rate': 0.007665879565991145, 
    'n_estimators': 4295,
    'max_depth': 5, 
    'subsample': 0.6078798211042953, 
    'colsample_bytree': 0.8320241653395914
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

# %%
model.save_model(save_model_nm)

# %%
# For 1% lift:
y_pred_val = model.predict_proba(X_val)[:, 1]
y_pred_test = model.predict_proba(X_test)[:, 1]
y_test_reset = y_test.reset_index(drop=True)
idx = np.argsort(y_pred_test)[::-1]
top_1_percent = int(0.01 * len(y_test_reset))  # 1% of test data size
predicted_top_1_percent = y_test_reset.iloc[idx][:top_1_percent]
actual_positives_top_1_percent = predicted_top_1_percent.sum()
expected_positives = y_test_reset.sum() * 0.01
lift_1_perc = actual_positives_top_1_percent / expected_positives
print(lift_1_perc)

# %%
explainer = shap.TreeExplainer(model)
explanation = explainer(X_test)

pred = model.predict(X_test)

shap_values = explanation.values
# make sure the SHAP values add up to marginal predictions
np.abs(shap_values.sum(axis = 1) + explanation.base_values - pred).max()
shap.plots.beeswarm(explanation)

# %% [markdown]
# ### Model metric checks

# %%
roc, lift_1_perc, lift_10_perc, ppv_1_perc, sensitivity_1_perc, ppv_10_perc, sensitivity_10_perc = calculate_metrics(model)
print("ROC: ",roc)
print("1% Lift: ",lift_1_perc)
print("1% PPV: ",ppv_1_perc)
print("1% Sensitivity: ",sensitivity_1_perc)
print("10% Lift: ",lift_10_perc)
print("10% PPV: ",ppv_10_perc)
print("10% Sensitivity: ",sensitivity_10_perc)

# %%
# Other metrics

y_pred_test_class = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_test_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

class_names = [0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%
print(classification_report(y_test, y_pred_test_class, target_names=target_names))

# %%
# Other metrics
y_pred_val = model.predict_proba(X_test)[:, 1]
y_pred_test_class = (y_pred_val >= threshold).astype(int)
#y_pred_test_class = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_test_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%
print(classification_report(y_test, y_pred_test_class, target_names=target_names))

# %%
y_pred_test = model.predict_proba(X_test)[:, 1]
plot_lift_curve(y_test, y_pred_test, step=0.01)

# %%
display = PrecisionRecallDisplay.from_estimator(
    model, X_test, y_test, name=model_name, plot_chance_level=True
)
_ = display.ax_.set_title("2-class Precision-Recall curve")

# %%
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_test)
# Use AUC function to calculate the area under the curve of precision recall curve
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)

# %%
lift_chart= lift_chart(X_test, y_test, model)
lift_chart.to_csv('lift_chart_xgboost.csv')
lift_chart

# %%
feature_importances = model.get_booster().get_score(importance_type = 'weight') 
importance_df = pd.DataFrame(feature_importances.items(), columns = ['Feature', 'Importance'])
importance_df = importance_df.sort_values(by = 'Importance', ascending=False)

# Display the feature importance
print(importance_df)

# importance_df = pd.DataFrame({ 'Feature': X_train.columns, 'Importance': feature_importances }).sort_values(by='Importance', ascending=False) # Display the DataFrame 
# pd.set_option("display.max_rows", None)
importance_df.to_csv('importance_df_xgboost.csv')
# importance_df

# %%


# %%


# %%



