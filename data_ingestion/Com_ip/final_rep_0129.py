# %%
%%capture
!pip install xgboost;
!pip install catboost;
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import time
from google.cloud import bigquery
import catboost
from catboost import CatBoostClassifier,Pool, metrics, cv
from sklearn.metrics import f1_score

client = bigquery.Client()

from sklearn.metrics import make_scorer,roc_auc_score
import numpy as np
def lift_at_1_percent(y_true, y_scores):
    top_1_percent = max(1, int(0.01 * len(y_true)))
    top_indices = np.argsort(y_scores)[-top_1_percent:]
    actual_positives = y_true.iloc[top_indices].sum()
    expected_positives = y_true.sum() * 0.01
    lift = actual_positives / expected_positives if expected_positives != 0 else 0
    return lift

def true_positives_at_1_percent(y_true, y_scores):
    top_1_percent = max(1, int(0.01 * len(y_true)))
    top_indices = np.argsort(y_scores)[-top_1_percent:]
    actual_positives = y_true.iloc[top_indices].sum()
    return actual_positives

def num_samples_at_1_percent(y_true, y_scores):
    top_1_percent = max(1, int(0.01 * len(y_true)))
    return top_1_percent

def f1_score_top_1_percent(y_true, y_pred_proba):
    top_1_percent = int(0.01 * len(y_true))
    top_indices = np.argsort(y_pred_proba)[-top_1_percent:]
    y_true_top = y_true.iloc[top_indices]
    y_pred_top = np.ones(top_1_percent)
    return f1_score(y_true_top, y_pred_top)


# Make scorers
scorers = {
    'lift@1%': make_scorer(lift_at_1_percent, needs_proba=True),
    'f1@1%': make_scorer(f1_score_top_1_percent, needs_proba=True),
    'true_positives@1%': make_scorer(true_positives_at_1_percent, needs_proba=True),
    'num_samples@1%': make_scorer(num_samples_at_1_percent, needs_proba=True),
    'AUC':'roc_auc',
    'weighted AUC':make_scorer(roc_auc_score, needs_proba = True, multi_class = 'ovr', average = 'weighted') 
}

# %%
import random

# Step 1: Set a seed
random.seed(100)

# %%
import pandas as pd
feature_importance_df = pd.read_csv("catboost_rfs_435.csv")
names_to_select_round2 = feature_importance_df["Name"].to_list()


exludel = ["clm_allowed_amt_1yr",
"clm_allowed_amt_2yr",
"clm_allowed_amt_3mo",
"clm_allowed_amt_6mo",
"clm_paid_amt_1yr",
"clm_paid_amt_2yr",
"clm_paid_amt_3mo",
"clm_paid_amt_6mo",
"clm_par_allowed_amt_1yr",
"clm_par_allowed_amt_2yr",
"clm_par_allowed_amt_3mo",
"clm_par_allowed_amt_6mo",
"clm_par_paid_amt_1yr",
"clm_par_paid_amt_2yr",
"clm_par_paid_amt_3mo",
"clm_par_paid_amt_6mo",
"clm_srv_copay_amt_1yr",
"clm_srv_copay_amt_3mo",
"clm_srv_copay_amt_6mo",
"covid_19",
"hpd_major_flag",
"chronic",
"txt_member",
"txt_referral",
"txt_1yr_outreach",
"talked"
]

names_to_select_round3 = sorted(list(set(names_to_select_round2) - set(exludel)))
len(names_to_select_round3)


# %%
sql = "Select individual_id,ip6," + ",".join(names_to_select_round3) + f""" From `anbc-hcb-dev.clin_analytics_hcb_dev.yc_a565095_cp_ip_neg_10_trs_3` where exclude_ip = 0 and include_post_6_status = 1 and ind_id_last_digit between 0 and 7 order by individual_id;"""
df = client.query(sql).to_dataframe()
df = df.drop_duplicates('individual_id', keep='last')  
df = df.drop(columns = ["individual_id"])
df = df.sample(frac = 1,random_state =100).reset_index(drop= True)

string_columns = df.select_dtypes(include =['object']).columns
print(string_columns, len(string_columns))
df[string_columns] = df[string_columns].fillna('missing')
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

print(string_columns)
X = df.drop(columns = ["ip6"])
print(X.shape)
y = df["ip6"].astype(int)

# %%
sql = "Select individual_id,ip6," + ",".join(names_to_select_round3) + f""" From `anbc-hcb-dev.clin_analytics_hcb_dev.yc_a565095_cp_ip_combine_janefewer_3` where exclude_ip = 0 and  ind_id_last_digit = 8;"""
df = client.query(sql).to_dataframe()
df = df.drop_duplicates('individual_id', keep='last')  
df = df.drop(columns = ["individual_id"])
string_columns = df.select_dtypes(include =['object']).columns
print(string_columns, len(string_columns))
df[string_columns] = df[string_columns].fillna('missing')
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

X_val = df.drop(columns = ["ip6"])
y_val = df["ip6"].astype(int)


# %%
sql = "Select individual_id, ip6," + ",".join(names_to_select_round3) + f""" From `anbc-hcb-dev.clin_analytics_hcb_dev.yc_a565095_cp_ip_combine_janefewer_3` where exclude_ip = 0 and  ind_id_last_digit = 9;"""
df = client.query(sql).to_dataframe()
df = df.drop_duplicates('individual_id', keep='last')  
df = df.drop(columns = ["individual_id"])

string_columns = df.select_dtypes(include =['object']).columns
print(string_columns, len(string_columns))
df[string_columns] = df[string_columns].fillna('missing')
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

X_test = df.drop(columns = ["ip6"])
print(X_test.shape)
y_test = df["ip6"].astype(int)


# %%
sql = "Select individual_id, ip6," + ",".join(names_to_select_round3) + f""" From `anbc-hcb-dev.clin_analytics_hcb_dev.yc_a565095_cp_ip_oot_3` where exclude_ip = 0;"""
df = client.query(sql).to_dataframe()
df = df.drop_duplicates('individual_id', keep='last')  
df = df.drop(columns = ["individual_id"])
df.shape
string_columns = df.select_dtypes(include =['object']).columns
print(string_columns, len(string_columns))
df[string_columns] = df[string_columns].fillna('missing')
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

X_test_oot = df.drop(columns = ["ip6"])
print(X_test.shape)
y_test_oot = df["ip6"].astype(int)


# %%
string_columns = df.select_dtypes(include =['object']).columns
valid_pool = Pool(X_val,y_val,cat_features = string_columns.to_list())
test_pool = Pool(X_test,y_test,cat_features = string_columns.to_list())
test_pool_oot = Pool(X_test_oot,y_test_oot,cat_features = string_columns.to_list())


# %%
string_columns = X.select_dtypes(include='object').columns.tolist()
catboost_model = CatBoostClassifier(random_seed = 100,thread_count= 15)
catboost_model.fit(X,y, eval_set=valid_pool, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)
catboost_model.save_model(f"catboost_nofs_ip6.cbm", format = "cbm" )


# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)

# %%
f_df = pd.read_csv(f"catboost_rfs_251.csv")
names_to_select_temp = f_df["Name"].to_list()
print(len(names_to_select_temp))
start = time.time()
X_s = X[names_to_select_temp]
string_columns = X_s.select_dtypes(include =['object']).columns
X_val_s = X_val[names_to_select_temp]
X_test_s = X_test[names_to_select_temp]
X_test_oot_s = X_test_oot[names_to_select_temp]
valid_pool_s = Pool(X_val_s,y_val,cat_features = string_columns.to_list())
test_pool_s = Pool(X_test_s,y_test,cat_features = string_columns.to_list())
test_pool_oot_s = Pool(X_test_oot_s,y_test_oot,cat_features = string_columns.to_list())
string_columns = X_s.select_dtypes(include='object').columns.tolist()

# %%

# Instantiate the CatBoost model
catboost_model = CatBoostClassifier(random_seed = 100,thread_count= 15)
catboost_model.fit(X_s,y, eval_set=valid_pool_s, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)
catboost_model.save_model(f"catboost_fs_ip6.cbm", format = "cbm" )

# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool_s)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool_s)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot_s)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)

#print(lift2_score)
#print(lift2_score_2)
#print(lift2_score_3)

end = time.time()
print(f"time:{end-start}")


# %%
param =  {'iterations': 3072, 'depth': 6, 'learning_rate': 0.0153400505076495,
          'random_strength': 7, 'l2_leaf_reg': 6.67489746621405, 'border_count': 206,
          'min_data_in_leaf': 24, 'grow_policy': 'Lossguide', 'od_wait': 59, 
          'bootstrap_type': 'Bernoulli', 'subsample': 0.674856992438242,
          'max_leaves': 41,
          'random_seed':100,
          'leaf_estimation_iterations': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'thread_count':15}

# Instantiate the CatBoost model
catboost_model = CatBoostClassifier(**param)
catboost_model.fit(X_s,y, eval_set=valid_pool_s, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)

catboost_model.save_model(f"catboost_fs_finetuned_ip6.cbm", format = "cbm" )

feature_importance = catboost_model.get_feature_importance().tolist()
feature_importance_df  = pd.DataFrame({"Name":X_s.columns.tolist(),"Importance": feature_importance}).sort_values("Importance",ascending = False)
#feature_importance_df.to_csv(f"Imp_fs_finetuned_ip6.csv",index =False)

# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool_s)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool_s)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot_s)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)


# %%
param =  {'iterations': 2658, 'depth': 5, 'learning_rate': 0.0222043591525155,
          'random_strength': 5, 'l2_leaf_reg': 4.29771760732085, 'border_count': 138,
          'min_data_in_leaf': 33, 'grow_policy': 'Lossguide', 'od_wait': 84, 
          'bootstrap_type': 'Bernoulli', 'subsample': 0.663063278012767,
          'max_leaves': 48,
          'random_seed':100,
          'leaf_estimation_iterations': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'thread_count':15}

# Instantiate the CatBoost model
catboost_model = CatBoostClassifier(**param)
catboost_model.fit(X_s,y, eval_set=valid_pool_s, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)

catboost_model.save_model(f"catboost_fs_finetuned_ip6.cbm", format = "cbm" )

feature_importance = catboost_model.get_feature_importance().tolist()
feature_importance_df  = pd.DataFrame({"Name":X_s.columns.tolist(),"Importance": feature_importance}).sort_values("Importance",ascending = False)
#feature_importance_df.to_csv(f"Imp_fs_finetuned_ip6.csv",index =False)

# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool_s)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool_s)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot_s)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)


# %%
param =  {'iterations': 2409, 'depth': 9, 'learning_rate': 0.020251978641620625, 
          'random_strength': 5, 'l2_leaf_reg': 1.214779369151024,
          'border_count': 188, 'min_data_in_leaf': 35, 'grow_policy': 'Lossguide',
          'od_wait': 56, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.4935981781887141, 
          'max_leaves': 39,
                    'random_seed':100,
         'leaf_estimation_iterations': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'thread_count':15}


# Instantiate the CatBoost model
catboost_model = CatBoostClassifier(**param)
catboost_model.fit(X_s,y, eval_set=valid_pool_s, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)

#catboost_model.save_model(f"catboost_fs_finetuned_ip6.cbm", format = "cbm" )

feature_importance = catboost_model.get_feature_importance().tolist()
feature_importance_df  = pd.DataFrame({"Name":X_s.columns.tolist(),"Importance": feature_importance}).sort_values("Importance",ascending = False)
#feature_importance_df.to_csv(f"Imp_fs_finetuned_ip6.csv",index =False)

# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool_s)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool_s)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot_s)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)


# %%
param =  {'iterations': 2409, 'depth': 9, 'learning_rate': 0.020251978641620625, 
          'random_strength': 5, 'l2_leaf_reg': 1.214779369151024,
          'border_count': 188, 'min_data_in_leaf': 35, 'grow_policy': 'Lossguide',
          'od_wait': 56, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.4935981781887141, 
          'max_leaves': 39,
                    'random_seed':100,
         'leaf_estimation_iterations': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'thread_count':15}


# Instantiate the CatBoost model
catboost_model = CatBoostClassifier(**param)
catboost_model.fit(X_s,y, eval_set=valid_pool_s, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)

#catboost_model.save_model(f"catboost_fs_finetuned_ip6.cbm", format = "cbm" )

feature_importance = catboost_model.get_feature_importance().tolist()
feature_importance_df  = pd.DataFrame({"Name":X_s.columns.tolist(),"Importance": feature_importance}).sort_values("Importance",ascending = False)
#feature_importance_df.to_csv(f"Imp_fs_finetuned_ip6.csv",index =False)

# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool_s)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool_s)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot_s)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)


# %%
param =  {'iterations': 2436, 'depth': 7, 'learning_rate': 0.026766501358942353, 
          'random_strength': 3, 'l2_leaf_reg': 2.949072748915259, 'border_count': 136, 
          'min_data_in_leaf': 30, 'grow_policy': 'SymmetricTree', 'od_wait': 84,
          'bootstrap_type': 'Bernoulli', 'subsample': 0.7901138550649578,
         'random_seed':100,
         'leaf_estimation_iterations': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'thread_count':15}


# Instantiate the CatBoost model
catboost_model = CatBoostClassifier(**param)
catboost_model.fit(X_s,y, eval_set=valid_pool_s, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)

#catboost_model.save_model(f"catboost_fs_finetuned_ip6.cbm", format = "cbm" )

feature_importance = catboost_model.get_feature_importance().tolist()
feature_importance_df  = pd.DataFrame({"Name":X_s.columns.tolist(),"Importance": feature_importance}).sort_values("Importance",ascending = False)
#feature_importance_df.to_csv(f"Imp_fs_finetuned_ip6.csv",index =False)

# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool_s)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool_s)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot_s)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)


# %%
param =  {'iterations': 3072, 'depth': 6, 'learning_rate': 0.0153400505076495,
          'random_strength': 7, 'l2_leaf_reg': 6.67489746621405, 'border_count': 206,
          'min_data_in_leaf': 24, 'grow_policy': 'Lossguide', 'od_wait': 59, 
          'bootstrap_type': 'Bernoulli', 'subsample': 0.674856992438242,
          'max_leaves': 41,
          'random_seed':100,
          'leaf_estimation_iterations': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'thread_count':15}

# Instantiate the CatBoost model
catboost_model = CatBoostClassifier(**param)
catboost_model.fit(X_s,y, eval_set=valid_pool_s, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)

#catboost_model.save_model(f"catboost_fs_finetuned_ip6.cbm", format = "cbm" )

feature_importance = catboost_model.get_feature_importance().tolist()
feature_importance_df  = pd.DataFrame({"Name":X_s.columns.tolist(),"Importance": feature_importance}).sort_values("Importance",ascending = False)
#feature_importance_df.to_csv(f"Imp_fs_finetuned_ip6.csv",index =False)

# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool_s)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool_s)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot_s)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)


# %%


param =  {'iterations': 2849, 'depth': 7, 'learning_rate': 0.0123887300085087,
          'random_strength': 7, 'l2_leaf_reg': 5.92399168175751, 'border_count': 115,
          'min_data_in_leaf': 39, 'grow_policy': 'Depthwise', 'od_wait': 72, 
          'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.778552303264736,
          'random_seed':100,
          'leaf_estimation_iterations': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'thread_count':15}

# Instantiate the CatBoost model
catboost_model = CatBoostClassifier(**param)
catboost_model.fit(X_s,y, eval_set=valid_pool_s, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)

#catboost_model.save_model(f"catboost_fs_finetuned_ip6.cbm", format = "cbm" )

feature_importance = catboost_model.get_feature_importance().tolist()
feature_importance_df  = pd.DataFrame({"Name":X_s.columns.tolist(),"Importance": feature_importance}).sort_values("Importance",ascending = False)
#feature_importance_df.to_csv(f"Imp_fs_finetuned_ip6.csv",index =False)

# Calculate the lift score
y_pred = catboost_model.predict_proba(valid_pool_s)[:,1]
lift_score = lift_at_1_percent(y_val, y_pred)
#lift2_score = lift_at_2_percent(y_val, y_pred)

y_pred_2 = catboost_model.predict_proba(test_pool_s)[:,1]
lift_score_2 = lift_at_1_percent(y_test, y_pred_2)
#lift2_score_2 = lift_at_2_percent(y_test, y_pred_2)

y_pred_3 = catboost_model.predict_proba(test_pool_oot_s)[:,1]
lift_score_3 = lift_at_1_percent(y_test_oot, y_pred_3)
#lift2_score_3 = lift_at_2_percent(y_test_oot, y_pred_3)

print(lift_score)
print(lift_score_2)
print(lift_score_3)


# %%
24.424408307840928
24.860808520939237
25.47818166763269


