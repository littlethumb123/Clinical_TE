# %% [markdown]
# #run this command as needed to re-establish connection to Big Query
# gcloud auth login --update-adc

# %% [markdown]
# ### Set up and getting data

# %%
!pip install xgboost
!pip install imbalanced-learn
!pip install catboost
!pip install optuna
!pip install torch
import pandas as pd
import numpy as np
import time
from google.cloud import bigquery
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
import random
import torch
import optuna
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
random.seed(35)
np.random.seed(35)

client = bigquery.Client()

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
# Get features
sql = """
SELECT
    f.* EXCEPT (asdb_plan_key, post_mnths, first_prv_dt, last_prv_dt, index_dt)
    , e.* EXCEPT (individual_id)
FROM 
    `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_non_embedding_features` AS f
LEFT JOIN
    `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_embeddings` AS e
        ON f.asdb_member_key = e.individual_id
WHERE 1=1
    AND NOT asdb_plan_key IN (33, 54)
    AND post_mnths >= 6
    
"""
df_og = client.query(sql).to_dataframe() 

df_og.shape
#2,542,308 members who qualify with 564 features we are exploring

# %%
# Get labels (acute IP) stay (0=no, 1=yes) in the 6 months post-index date
sql = """
SELECT
    asdb_member_key
    , acute_ip_flag
FROM 
    `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip` AS o
WHERE 1=1 
  AND o.asdb_member_key IN (SELECT asdb_member_key FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_non_embedding_features` WHERE 1=1 AND NOT asdb_plan_key IN (33, 54) AND post_mnths >= 6)"""
outcome_og = client.query(sql).to_dataframe() 
outcome_og.shape

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# #### Fill in zeros

# %%
from pandas.api.types import is_integer_dtype as is_integer
from pandas.api.types import is_float_dtype as is_float
import re

emb_pattern = r'emb[0-255]+'
emb_col = [col for col in df_og.columns if re.match(emb_pattern ,col)]
df_og[emb_col] = df_og[emb_col].fillna(0)
for c in df_og.columns: 
    dt = df_og[c].dtype
    if is_integer(dt) or is_float(dt):
        df_og[c]=df_og[c].fillna(0) 
        # print("Floatint:", dt)
    else:
        try:
            df_og[c]= df_og[c].fillna('')
        except:
            print("ERROR - DATE VARIABLE FOUND", dt)

# %%
df = df_og
outcome = outcome_og

print(df.shape)
print(outcome.shape)

# %% [markdown]
# #### One Hot Encode Categorical

# %%
#  0 := categorical, 1 := continuous, 2 := binary
nem_to_type = {
    'narc':2, 
    # 'otc_fills_yr2': 1,
    # 'otc_fills_yr1':1,  
    'COP':2, 
    'sleep_apnea':2, 
    'spinal_inj':2,
    'back':2,
    'substance':2,
    'ALC':2,
    'bipolar':2,
    'psychoses':2,
    'EDO':2, 
    'SCA':2, 
    'DIA':2, 
    'DEP':2, 
    'abdominal_pain':2, 
    'OST':2, 
    'AID':2, 
    'IDA':2, 
    'ANX':2, 
    'DEM':2, 
    'CYS':2, 
    'autoimmune':2, 
    'MOH':2, 
    'HEM':2, 
    'esrd':2, 
    'HepC':2, 
    'HYP':2, 
    'HYC':2, 
    'immune':2, 
    'intel_dsblty':2, 
    'meta_cancer':2, 
    'liver_dis':2, 
    'MSS':2, 
    'OBE':2, 
    'oud':2, 
    'liver_other':2, 
    'paralysis':2, 
    'PAR':2, 
    'PUD':0, 
    'hmd':2, 
    'PVD':2, 
    'CRO':2, 
    'AST':2, 
    'EPL':2, 
    'low_med_sev_ed_flag_yr2':2, 
    'med_high_sev_ed_flag_yr2':2, 
    'high_sev_ed_flag_yr2':2, 
    'acute_ip_flag_yr1':2, 
    'CHO':2,
    'burns':2, 
    'acute_ip_flag_yr2':2, 
    'cad':2, 
    'Cancer':2, 
    'ed_flag_yr2':2, 
    'high_sev_ed_flag_yr1':2, 
    'med_sev_ed_flag_yr2':2, 
    'AUT':2, 
    'med_high_sev_ed_flag_yr1':2, 
    'low_med_sev_ed_flag_yr1':2, 
    'low_sev_ed_flag_yr1':2, 
    'CBD':2, 
    'CHF':2, 
    'CRF':2, 
    'VNA':2, 
    'CHD':2, 
    'ed_flag_yr1':2, 
    'med_sev_ed_flag_yr1':2, 
    'low_sev_ed_flag_yr2':2, 
    'urbsubr':0, 
    'gender':0, # TODO: IF = M, 1 ELIF = F 0 ELSE NULL 
    'cms_prost_cancer_scrn':0, 
    'cms_hpv_scrn':0, 
    'cms_cvd_scrn':0, 
    'cms_lung_cancer_scrn':0, 
    'cms_pelvic':0,
    'coa_population_group':0, 
    'cms_pap':0, 
    'cms_t2d_scrn':0, 
    'cms_bone_scrn':0, 
    'cms_alc_scrn':0,
    'cms_ibt_cvd':0, 
    'cms_col_scrn':0,
    'index_dt':0, 
    'cms_ibt_obese':0, 
    'cms_flu_vax':0, 
    'cms_pneum_vax':0, 
    'cms_dep_scrn':0,
    'tenure_yr2':1, # either 
    'tenure_yr1':1, # either
    'cms_hepb_vax':0,
    'low_sev_ed_visits_yr2':0, # 7/2/24 removed
    'coa_population_category':0, 
    'cms_mam_scrn':0, # 7/2/24 removed
    'low_sev_ed_visits_yr1':1,
    'sum_acute_ip_admits_yr2':1,
    'sum_acute_ip_admits_yr1':1,
    'cms_tobacco':0,# 7/2/24 removed
    'low_med_sev_ed_visits_yr2':1, 
    'cms_t2d_train':0, # 7/2/24 removed
    'sum_preventable_yr2':1, 
    'cms_hepb_scrn':0, ###
    'sum_preventable_yr1':1, 
    'major_chronic_cnt':1, 
    'low_med_sev_ed_visits_yr1':1,
    'sum_ob':1, 
    'sum_unnecessary_yr2':1, 
    'sum_chol_lab':1, 
    'cms_nutrition':0,  # 7/2/24 removed
    'coe_anesth_clm_yr2':1, 
    'sum_a1c_lab':1, 
    'sum_unnecessary_yr1':1, 
    'sum_avoidable_yr2':1, 
    'gpi2_cnt_yr1':1, 
    'high_sev_ed_visits_yr2':1,
    'gpi2_cnt_yr2':1, 
    'ms_brand_fills_yr2':1, 
    'coe_anesth_clm_yr1':1, 
    'med_sev_ed_visits_yr2':1, 
    'cms_sti_scrn':0, 
    'med_high_sev_ed_visits_yr2':1, 
    'high_sev_ed_visits_yr1':1, 
    'ms_brand_fills_yr1':1, 
    'sum_avoidable_yr1':1, 
    'inhaled_steroid_scripts_yr2':1, 
    'med_high_sev_ed_visits_yr1':1, 
    'med_sev_ed_visits_yr1':1, 
    'inhaled_steroid_scripts_yr1':1, 
    'obs_clm_yr2':1,
    'gpi4_cnt_yr2':1,
    'gpi4_cnt_yr1':1,
    'obs_clm_yr1':1, 
    'sum_dme':1, 
    'antianginal_agent_scripts_yr2':1, 
    'mail_order_fills_yr2':1, 
    'branded_generic_fills_yr2':1, 
    'gpi_cnt_yr1':1,
    'gpi_cnt_yr2':1,
    'adi_score':1, 
    'sdi_score':1, 
    'antianginal_agent_scripts_yr1':1, 
    'ethnicity_code':0, 
    'sum_ed_visits_yr2':1,
    'mail_order_fills_yr1':1,
    'uc_clm_yr2':1, 
    'calcium_channel_blk_scripts_yr2':1,
    'antianxiety_scripts_yr2':1, 
    'beta_blocker_scripts_yr2':1, 
    'sum_ed_visits_yr1':1, 
    'branded_generic_fills_yr1':1,
    'coe_maternity_clm_yr2':1,
    'sum_chemo':1, 
    'agenbr':1, 
    'uc_clm_yr1':1,
    'calcium_channel_blk_scripts_yr1':1,
    'coe_maternity_clm_yr1':1,
    'diuretic_scripts_yr2':1,
    'primarylanguage_desc':0, # TODO: Check counts, limit to other if <2000 ppl
    'beta_blocker_scripts_yr1':1, 
    'antianxiety_scripts_yr1':1,
    'antihypertensive_scripts_yr2':1,
    'ndc_cnt_yr1':1, 
    'ndc_cnt_yr2':1, 
    'lipid_lowering_scripts_yr2':1,
    'diuretic_scripts_yr1':1, 
    'coe_surg_clm_yr2':1, 
    'sum_acute_calc_los_yr1':1, 
    'sum_acute_calc_los_yr2':1,
    'sum_pcp':1, 
    'lipid_lowering_scripts_yr1':1, 
    'antihypertensive_scripts_yr1':1,
    'coe_surg_clm_yr1':1,
    'coe_mrx_clm_yr2':1,
    'antidepressant_scripts_yr2':1,
    'coe_mrx_clm_yr1':1, 
    'antipsychotic_scripts_yr2':1, 
    'anticonvulsant_scripts_yr2':1, 
    'antidiabetic_scripts_yr2':1, 
    'antidepressant_scripts_yr1':1, 
    'coe_radio_clm_yr2':1,
    'anticonvulsant_scripts_yr1':1, 
    'coe_radio_clm_yr1':1, 
    'emis_mrx_clm_yr2':1, 
    'coe_ltc_community_clm_yr1':1, 
    'emis_community_clm_yr1':1,
    'coe_ltc_community_clm_yr2':1, 
    'emis_community_clm_yr2':1, 
    'emis_radio_clm_yr1':1, 
    'emis_radio_clm_yr2':1, 
    'antipsychotic_scripts_yr1':1, 
    'antidiabetic_scripts_yr1':1, 
    'emis_mrx_clm_yr1':1,
    'coe_ip_hos_clm_yr2':1,
    'coe_ip_hos_clm_yr1':1, 
    'coe_ip_non_hos_clm_yr2':1,
    'emis_pcp_clm_yr1':1,
    'emis_pcp_clm_yr2':1,
    'coe_phy_clm_yr1':1, 
    'coe_ip_non_hos_clm_yr1':1,
    'inhaled_steroid_days_supply_yr2':1, 
    'coe_phy_clm_yr2':1, 
    'emis_ip_clm_yr2':1,
    'ss_brand_fills_yr2':1, 
    'emis_ip_clm_yr1':1, 
    'inhaled_steroid_days_supply_yr1':1, 
    'coe_eval_clm_yr2':1, 
    'coe_ltc_ins_clm_yr2':1, 
    'emis_ins_clm_yr2':1, 
    'sum_spec':1, 
    'coe_eval_clm_yr1':1, 
    'ss_brand_fills_yr1':1,
    'retail_fills_yr2':1, 
    'retail_fills_yr1':1, 
    'emis_ins_clm_yr1':1, 
    'coe_ltc_ins_clm_yr1':1, 
    'emis_spec_clm_yr2':1,
    'emis_spec_clm_yr1':1,
    'emis_ed_clm_yr2':1,
    'coe_lab_clm_yr2':1, 
    'generic_fills_yr2':1,
    'emis_ed_clm_yr1':1,
    'last_prv_dt':1, 
    'first_prv_dt':1,
    'coe_lab_clm_yr1':1,
    'maint_drug_fills_yr2':1,
    'emis_lab_clm_yr2':1,
    'formulary_fills_yr2':1,
    'emis_lab_clm_yr1':1, 
    'generic_fills_yr1':1, 
    'coe_op_hos_clm_yr2':1, 
    'formulary_fills_yr1':1,
    'emis_misc_clm_yr2':1,
    'maint_drug_fills_yr1':1,
    'coe_op_hos_clm_yr1':1, 
    'rx_claim_cnt_yr2':1, 
    'antianginal_agent_days_supply_yr2':1,
    'coe_mh_clm_yr2':1,
    'coe_mh_clm_yr1':1,
    'emis_misc_clm_yr1':1, 
    'coe_ltc_home_clm_yr2':1,
    'emis_home_clm_yr2':1,
    'antianginal_agent_days_supply_yr1':1, 
    'ltc_clm_yr2':1,
    'coe_ltc_home_clm_yr1':1,
    'emis_home_clm_yr1':1, 
    'emis_hh_clm_yr2':1, 
    'emis_hh_clm_yr1':1, 
    'ltc_clm_yr1':1, 
    'rx_claim_cnt_yr1':1, 
    'calcium_channel_blk_days_supply_yr2':1, 
    'sum_op_visits_yr2':1, 
    'sum_op_visits_yr1':1, 
    'coe_op_non_hos_clm_yr2':1,
    'emis_ambul_clm_yr2':1, 
    'water_quality':1, 
    'coe_op_non_hos_clm_yr1':1, 
    'calcium_channel_blk_days_supply_yr1':1, 
    'beta_blocker_days_supply_yr2':1,
    'emis_ambul_clm_yr1':1,
    'beta_blocker_days_supply_yr1':1, 
    'emis_mh_clm_yr2':1, 
    'emis_mh_clm_yr1':1, 
    'diuretic_days_supply_yr2':1, 
    'antianxiety_days_supply_yr2':1, 
    'lipid_lowering_days_supply_yr2':1, 
    'coe_other_clm_yr2':1, 
    'coe_other_clm_yr1':1,
    'antianxiety_days_supply_yr1':1, 
    'antihypertensive_days_supply_yr2':1,
    'diuretic_days_supply_yr1':1, 
    'lipid_lowering_days_supply_yr1':1, 
    'antihypertensive_days_supply_yr1':1, 
    'antipsychotic_days_supply_yr2':1,
    'antidepressant_days_supply_yr2':1,
    'antipsychotic_days_supply_yr1':1,
    'antidepressant_days_supply_yr1':1, 
    'anticonvulsant_days_supply_yr2':1, 
    'anticonvulsant_days_supply_yr1':1,
    'antidiabetic_days_supply_yr2':1, 
    'antidiabetic_days_supply_yr1':1, 
    'income_inequality':1, 
    'svi_score':1, 
    'days_supply_sum_yr2':1, 
    'zip_weight_avg_medinc':1,
    'days_supply_sum_yr1':1,
    'food_access':1, 
    'citizenship_index':1,
    'acs_social_risk_score':1, 
    'housing_desert':1, 
    'unemployment_index':1,
    'health_habits':1,
    'natural_disaster':1,
    'proactive_health':1,
    'housing_ownership':1, 
    'health_infra':1,
    'language_score':1, 
    'racial_diversity':1, 
    'housing_quality':1,
    'income_index':1,
    'education_index':1,
    'transport_access':1, 
    'disability_score':1,
    'technology_access':1,
    'poverty_score':1, 
    'social_isolation':1,
    'health_access':1,
    'csdi_social_risk_score':1,
     'asdb_member_key':1,
}

# %%
index_to_feature = dict(enumerate(df.columns))
feature_to_index = {value: key for key, value in index_to_feature.items()}
categorical_features = [feature for feature in nem_to_type if nem_to_type[feature] == 0]
categorical_indices = [feature_to_index[feature] for feature in categorical_features if feature in feature_to_index]
len(categorical_features)

# %%
categorical_features.remove('index_dt')

df[categorical_features] = df[categorical_features].astype(str)
df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(-1)  # Assuming NaN or other values to be marked as -1

categorical_features.remove('gender')

from sklearn.preprocessing import OneHotEncoder
# Minimum occurrence threshold
min_occurrence = 2000

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Process each categorical feature
for feature in categorical_features:
    # Filter categories based on value count threshold
    counts = df[feature].value_counts()
    categories_to_keep = counts[counts >= min_occurrence].index
   
    # Filter the DataFrame to include only the categories to keep
    filtered_df = df[df[feature].isin(categories_to_keep)]
   
    # Apply one-hot encoding to the filtered data
    if not filtered_df.empty:
        encoded_data = encoder.fit_transform(filtered_df[[feature]])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([feature]))
       
        # Drop the original column and concatenate the new encoded columns
        df = df.drop(feature, axis=1)
        df = pd.concat([df, encoded_df], axis=1)

# %%
df.shape

# %%
df = df.set_index('asdb_member_key')
outcome = outcome.set_index('asdb_member_key')
#merge outcome and features on asdb_member_key so we don't have label/predictors from different members
merged = df.merge(outcome, on='asdb_member_key', how='left')

# assert merged.shape[1] == 561, "The number of columns in the DataFrame does not match expected feature + label count" # Splitting the data into train and test sets X_train, X_test, y_train, y_test = train_test_split( merged.iloc[:, :560], # Selecting first 560 columns as features, adjust if column count changes merged['acute_ip_flag'], # Using direct column selection for clarity test_size=0.2, random_state=100 )


# %%
X_train, X_test, y_train, y_test = train_test_split(merged.iloc[:, 1:560], merged.loc[:, 'acute_ip_flag'], test_size=0.2, random_state=35, stratify=merged.loc[:, 'acute_ip_flag'])
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=35, stratify=y_test)

# 80:10:10 split

# %%
y_train = y_train.astype('int')
y_val = y_val.astype('int')
y_test = y_test.astype('int')


# %%
string_columns = X_train.select_dtypes(include='object').columns.tolist() # CatBoost handles categorical features natively, thus we need to specify categorical columns

# %% [markdown]
# ### Undersampling Tests (only run for finding optimal downsample ratio)

# %%
ratios = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # FILL IN RATIOS TO TEST
seeds = [53]  # Random seeds for reproducibility
results = []  # Store results

for r in ratios:
    for s in seeds:
        # Setup undersampler
        undersample = RandomUnderSampler(sampling_strategy=r, random_state=s)
        X_train_u, y_train_u = undersample.fit_resample(X_train, y_train)

        # Initialize and train XGBoost model
        xgb_model = XGBClassifier(seed=53, n_jobs=15, verbosity=0, enable_categorical=True)
        xgb_model.fit(X_train_u, y_train_u, eval_set=[(X_val, y_val)], verbose=False)
        print("done fit")

        # Predict probabilities
        y_pred_val = xgb_model.predict_proba(X_val)[:, 1]
        y_pred_test = xgb_model.predict_proba(X_test)[:, 1]

        idx = np.argsort(y_pred_test)[::-1]
        top_1_percent = int(0.01 * len(y_test))  # 1% of test data size

        predicted_top_1_percent = y_test.iloc[idx][:top_1_percent]
        actual_positives_top_1_percent = predicted_top_1_percent.sum()

        expected_positives = y_test.sum() * 0.01

        lift = actual_positives_top_1_percent / expected_positives

        results.append((r, s, lift))
        print("one iteration", r, s, lift)

# %%
print(results)

# %%
# No undersampling test
xgb_model = XGBClassifier(random_state=53, n_jobs=15, verbosity=0, enable_categorical=True)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

print("done fit")

# Prediction and calculate lift
y_pred_test = xgb_model.predict_proba(X_test)[:, 1]
y_test_reset = y_test.reset_index(drop=True)
idx = np.argsort(y_pred_test)[::-1]
top_1_percent = int(0.01 * len(y_test_reset))

predicted_top_1_percent = y_test_reset.iloc[idx][:top_1_percent]
actual_positives_top_1_percent = predicted_top_1_percent.sum()
expected_positives = y_test_reset.sum() * 0.01
lift = actual_positives_top_1_percent / expected_positives

results.append((0, 0, lift))
print("one iteration, no sample")

# No undersampling but with class weights test
weights = np.where(y_train == 0, 1, len(y_train) / (2 * np.sum(y_train == 1)))

xgb_model = XGBClassifier(random_state=53, n_jobs=15, verbosity=0, enable_categorical=True)
xgb_model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)

print("done fit")

# Prediction and calculate lift
y_pred_test = xgb_model.predict_proba(X_test)[:, 1]
y_test_reset = y_test.reset_index(drop=True)
idx = np.argsort(y_pred_test)[::-1]
top_1_percent = int(0.01 * len(y_test_reset))

predicted_top_1_percent = y_test_reset.iloc[idx][:top_1_percent]
actual_positives_top_1_percent = predicted_top_1_percent.sum()
expected_positives = y_test_reset.sum() * 0.01
lift = actual_positives_top_1_percent / expected_positives

results.append((999, 999, lift))
print("one iteration, class weights")

# %%
results_df = pd.DataFrame(results, columns=['Ratio', 'Seed', '1% Lift'])

# %%
print(results)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data=results_df, x='Ratio', y='1% Lift')
plt.title('1% Lift Across Different Ratios and Seeds')
plt.show()

results_df.to_csv('lift_results.csv', index=False)


# %%
def check_label_distribution(y):
    # Count the number of occurrences of each label 
    count_0 = (y == 0).sum()
    count_1 = (y == 1).sum()
    ratio = count_1 / count_0 if count_0 != 0 else np.inf
    print(count_0, count_1, ratio)
    return count_0, count_1, ratio

print("Training Set:")
check_label_distribution(y_train)

print("\nValidation Set:")
check_label_distribution(y_val)

print("\nTest Set:")
check_label_distribution(y_test)


# %% [markdown]
# ### RFE Tests (only run for generating list of selected features)

# %%
undersample = RandomUnderSampler(sampling_strategy=0.03, random_state=53)  # FILL IN THE OPTIMAL RATIO FOUND IN THE SECTION ABOVE
steps = [
    ('u', undersample)
]
pipeline = Pipeline(steps=steps)
X_train, y_train = pipeline.fit_resample(X_train, y_train)

# %%
def check_label_distribution(y):
    # Count the number of occurrences of each label 
    count_0 = (y == 0).sum()
    count_1 = (y == 1).sum()
    ratio = count_1 / count_0 if count_0 != 0 else np.inf
    print(count_0, count_1, ratio)
    return count_0, count_1, ratio

print("Training Set:")
check_label_distribution(y_train)

print("\nValidation Set:")
check_label_distribution(y_val)

print("\nTest Set:")
check_label_distribution(y_test)

# y_train.value_counts()
# X_train.value_counts()


# %%
pip install tensorflow

# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# %%
import xgboost
print(xgboost.__version__)

# %%
import gc
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Check if GPU is available
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if not gpus:
    raise SystemError("No GPUs found. Please ensure your environment has a GPU available.")

# Initialize the GPU-enabled XGBoost classifier
xgb_model_rfe = XGBClassifier(
    random_state=53,
    n_jobs=-1,
    verbosity=0,
    use_label_encoder=False,
    tree_method = "hist", 
    device = "cuda"
)

rfecv = RFECV(
    estimator=xgb_model_rfe,
    step=10,
    cv=StratifiedKFold(3),
    scoring="roc_auc",
    min_features_to_select=1,
    n_jobs=-1
)

rfecv.fit(X_train, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")

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


# TODO: GET LIST OF FEATURES

# %%
# Write selected features to file to save time (no need to run RFE again)

X_train_df = pd.DataFrame(X_train, columns = df.columns)
selected_features = X_train.columns[rfecv.support_]
print(selected_features)
len(selected_features)
with open('xgboost_selected_features.txt', 'w') as file:
    # Join the list elements into a single string with a newline character
    data_to_write = '\n'.join(selected_features)
     
    # Write the data to the file
    file.write(data_to_write)

# %% [markdown]
# ### Optuna Hyperparameterization (only run for determining optimal hyperparameters)

# %%
import pandas as pd

# Get the selected features from RFE

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

# %%
X_train.shape

# %%
import logging
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


logging.basicConfig(filename='optuna_xgboost.log', level=logging.INFO)

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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
   
    y_pred_test = model.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, y_pred_test)
   
    logging.info(f"Trial {trial.number} AUC: {roc_auc_test} Params: {params}")
   
    study_results = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))  # Saving the results to a CSV file
    study_results.to_csv('optuna_results.csv', index=False)
   
    return roc_auc_test

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=75)  # trials must be at least 50
print("Number of finished trials: ", len(study.trials))

best_trial = study.best_trial
print("Best trial:")
print(" AUC:", best_trial.value)
print(" Params:", best_trial.params)

# Log the best result
logging.info(f"Best AUC: {best_trial.value} with params: {best_trial.params}")
study_results = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))  # Saving the results to a CSV file
study_results.to_csv('optuna_results.csv', index=False)

# %%
print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
print(f"Trial with highest accuracy: ")
print(f"\tnumber: {trial_with_highest_accuracy.number}")
print(f"\tparams: {trial_with_highest_accuracy.params}")
print(f"\tvalues: {trial_with_highest_accuracy.values}")
optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="auc"
)

# %% [markdown]
# ### Building Optimal Model

# %%
import xgboost as xgb

# Optimal model hyperparameters
params = {
    'random_state': 53, 
    'tree_method' : "hist", 
    'device' : "cuda",
    'n_jobs': -1,
    'learning_rate': 0.009719586136010807, 
    'n_estimators': 1253,
    'max_depth': 8, 
    'subsample': 0.9206518647230545, 
    'colsample_bytree': 0.672444373451293
}
 
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

# %% [markdown]
# #### Metrics for Optimal Model

# %%
from sklearn.metrics import confusion_matrix

def calculate_metrics(model):
    # Probabilities of the positive class
    y_pred_test = model.predict_proba(X_test)[:, 1]
    y_pred_val = model.predict_proba(X_val)[:, 1]

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
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt


y_pred_test_class = model.predict(X_test)

# # Uncomment for applying threshold, can always tune after model run
# threshold = 0.25
# y_pred_test_class = (y_pred_test >= threshold).astype(int)

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
from sklearn.metrics import classification_report
target_names = ['without IP', 'with IP']
print(classification_report(y_test, y_pred_test_class, target_names=target_names))

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
y_pred_test = model.predict_proba(X_test)[:, 1]
plot_lift_curve(y_test, y_pred_test, step=0.01)


# %%
from sklearn.metrics import PrecisionRecallDisplay

display = PrecisionRecallDisplay.from_estimator(
    model, X_test, y_test, name="KMDO_logreg", plot_chance_level=True
)
_ = display.ax_.set_title("2-class Precision-Recall curve")

# %%


# %%
### PASTED OVER FROM GITHUB
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
lift_chart= lift_chart(X_test, y_test, model)
lift_chart.to_csv('lift_chart_xgboost.csv')
lift_chart

# %%
model.save_model('xgboost_model.cbm')

# %%
feature_importances = model.get_booster().get_score(importance_type='weight') 
importance_df = pd.DataFrame(feature_importances.items(), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(importance_df)
importance_df.to_csv('importance_df_xgboost.csv')

# %%
import matplotlib.pyplot as plt

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



