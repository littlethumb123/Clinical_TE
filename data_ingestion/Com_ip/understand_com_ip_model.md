# Comprehensive Technical Report: Commercial Inpatient Admission Prediction Model

Based on my analysis of `final_rep_0129.py` and the supporting SQL files, here is a detailed technical breakdown of the existing pipeline.

---

## 1. Problem Definition

| Aspect | Detail |
|--------|--------|
| **Prediction Task** | Binary classification: Will a commercial member have an acute inpatient admission in the next 6 months? |
| **Target Variable** | `ip6` (integer: 0 or 1) |
| **Prediction Horizon** | 180 days (6 months) from the index date |

The outcome `ip6` is derived from claims data where **maternity, trauma, transplant, and non-impactible admissions are excluded** (per the SQL logic in `commercial_ip_outcome_generation.sql`).

---

## 2. Data Pipeline Architecture

### 2.1 Data Source
Data is queried directly from **Google BigQuery** tables using the `google.cloud.bigquery` client:

```21:26:data_ingestion/Com_ip/final_rep_0129.py
def lift_at_1_percent(y_true, y_scores):
    top_1_percent = max(1, int(0.01 * len(y_true)))
    top_indices = np.argsort(y_scores)[-top_1_percent:]
    actual_positives = y_true.iloc[top_indices].sum()
    expected_positives = y_true.sum() * 0.01
    lift = actual_positives / expected_positives if expected_positives != 0 else 0
```

### 2.2 Data Splitting Strategy

The data is split **deterministically by member ID** using the last digit of `individual_id` (`ind_id_last_digit`):

| Split | `ind_id_last_digit` | Approx. % | Source Table |
|-------|---------------------|-----------|--------------|
| **Train** | 0–7 | 80% | `yc_a565095_cp_ip_neg_10_trs_3` |
| **Validation** | 8 | 10% | `yc_a565095_cp_ip_combine_janefewer_3` |
| **Test** | 9 | 10% | `yc_a565095_cp_ip_combine_janefewer_3` |
| **Out-of-Time (OOT)** | All | Temporal | `yc_a565095_cp_ip_oot_3` |

**Key Observation**: This approach prevents **member-level leakage** (same member appearing in train and test), which is critical since members may have multiple rows over time.

```102:106:data_ingestion/Com_ip/final_rep_0129.py
sql = "Select individual_id,ip6," + ",".join(names_to_select_round3) + f""" From `anbc-hcb-dev.clin_analytics_hcb_dev.yc_a565095_cp_ip_neg_10_trs_3` where exclude_ip = 0 and include_post_6_status = 1 and ind_id_last_digit between 0 and 7 order by individual_id;"""
df = client.query(sql).to_dataframe()
df = df.drop_duplicates('individual_id', keep='last')  
df = df.drop(columns = ["individual_id"])
df = df.sample(frac = 1,random_state =100).reset_index(drop= True)
```

Additional filters applied during training data selection:
- `exclude_ip = 0`: Excludes members with prior IP exclusions
- `include_post_6_status = 1`: Ensures members have valid enrollment during the outcome window

---

## 3. Feature Engineering & Selection

### 3.1 Initial Feature Set
The pipeline starts with a pre-computed feature importance file (`catboost_rfs_435.csv`) containing **435 features**:

```64:66:data_ingestion/Com_ip/final_rep_0129.py
import pandas as pd
feature_importance_df = pd.read_csv("catboost_rfs_435.csv")
names_to_select_round2 = feature_importance_df["Name"].to_list()
```

### 3.2 Feature Exclusion (Leakage Prevention)
A hardcoded exclusion list removes **26 features** that are either:
- **Potential leakage** (future-looking or cost-related that might encode outcome information)
- **Operationally unavailable** at prediction time

```69:95:data_ingestion/Com_ip/final_rep_0129.py
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
```

**Excluded categories:**
- **Cost features** (`clm_allowed_amt_*`, `clm_paid_amt_*`) – potentially correlated with outcome
- **Outreach/intervention flags** (`txt_member`, `txt_referral`, `talked`) – operationally unavailable or confounded
- **Aggregate flags** (`chronic`, `hpd_major_flag`) – possibly redundant or leaky

### 3.3 Final Feature Set
After exclusion, the remaining features are sorted:

```97:98:data_ingestion/Com_ip/final_rep_0129.py
names_to_select_round3 = sorted(list(set(names_to_select_round2) - set(exludel)))
len(names_to_select_round3)
```

A further refined subset (**251 features**) is loaded for optimized experiments:

```200:202:data_ingestion/Com_ip/final_rep_0129.py
f_df = pd.read_csv(f"catboost_rfs_251.csv")
names_to_select_temp = f_df["Name"].to_list()
print(len(names_to_select_temp))
```

### 3.4 Feature Types (from SQL analysis)
Based on the SQL generation code, features fall into these categories:

| Category | Count (Approx.) | Examples |
|----------|-----------------|----------|
| **Demographics** | ~14 | `age`, `gender_cd`, `product_ln_cd` |
| **HPD Chronic Conditions** | ~94 | Binary flags for chronic diseases |
| **MDC/Utilization** | ~332 | Major Diagnostic Category, case mix |
| **Lab Results** | ~72 | Lab test indicators |
| **Membership/Geographic** | ~19 | Region, plan type |
| **Cost/Utilization PMPM** | ~68 | Per-member-per-month normalized metrics |

---

## 4. Data Preprocessing

### 4.1 Deduplication
Each dataset is deduplicated to **one row per member**, keeping the last occurrence:

```104:104:data_ingestion/Com_ip/final_rep_0129.py
df = df.drop_duplicates('individual_id', keep='last')  
```

### 4.2 Missing Value Handling
Simple imputation strategy:

```108:112:data_ingestion/Com_ip/final_rep_0129.py
string_columns = df.select_dtypes(include =['object']).columns
print(string_columns, len(string_columns))
df[string_columns] = df[string_columns].fillna('missing')
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)
```

| Type | Imputation |
|------|------------|
| Categorical (object) | `'missing'` string |
| Numeric | `0` |

**Note**: This imputation is applied identically to train, validation, test, and OOT sets. While consistent, filling numerics with 0 may introduce signal (e.g., 0 visits vs. missing visits are treated the same).

### 4.3 Shuffling
Training data is shuffled with a fixed seed for reproducibility:

```106:106:data_ingestion/Com_ip/final_rep_0129.py
df = df.sample(frac = 1,random_state =100).reset_index(drop= True)
```

---

## 5. Model Architecture & Training

### 5.1 Model Selection
**CatBoostClassifier** is used exclusively throughout the pipeline:

```177:178:data_ingestion/Com_ip/final_rep_0129.py
catboost_model = CatBoostClassifier(random_seed = 100,thread_count= 15)
catboost_model.fit(X,y, eval_set=valid_pool, verbose=0, plot=False,  cat_features=string_columns,  use_best_model=True)
```

### 5.2 Categorical Feature Handling
CatBoost handles categorical features natively via **Ordered Target Statistics** encoding. Features are passed using the `cat_features` parameter:

```169:172:data_ingestion/Com_ip/final_rep_0129.py
string_columns = df.select_dtypes(include =['object']).columns
valid_pool = Pool(X_val,y_val,cat_features = string_columns.to_list())
test_pool = Pool(X_test,y_test,cat_features = string_columns.to_list())
test_pool_oot = Pool(X_test_oot,y_test_oot,cat_features = string_columns.to_list())
```

### 5.3 Hyperparameter Configurations
The script tests **multiple hyperparameter configurations** manually (not automated search):

**Configuration 1** (Default CatBoost):
```python
CatBoostClassifier(random_seed=100, thread_count=15)
```

**Configuration 2** (Tuned - Lossguide):

```247:257:data_ingestion/Com_ip/final_rep_0129.py
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
```

**Configuration 3** (Bayesian Bootstrap):

```329:339:data_ingestion/Com_ip/final_rep_0129.py
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
```

**Configuration 4** (SymmetricTree):

```413:422:data_ingestion/Com_ip/final_rep_0129.py
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
```

### 5.4 Key Training Parameters

| Parameter | Range Tested | Purpose |
|-----------|--------------|---------|
| `iterations` | 2409–3072 | Number of boosting rounds |
| `depth` | 5–9 | Tree depth |
| `learning_rate` | 0.012–0.027 | Step size |
| `grow_policy` | Lossguide, Depthwise, SymmetricTree | Tree construction strategy |
| `bootstrap_type` | Bernoulli, Bayesian | Sampling method |
| `od_wait` | 56–84 | Early stopping patience |
| `l2_leaf_reg` | 1.2–6.7 | L2 regularization |

### 5.5 Early Stopping
All configurations use **early stopping** based on validation AUC:

```python
'od_type': 'Iter',  # Overfitting detector type
'od_wait': 59,       # Wait N iterations without improvement
'use_best_model': True  # Restore best iteration after training
```

---

## 6. Evaluation Framework

### 6.1 Custom Scoring Functions
The pipeline defines **business-oriented metrics** focused on the **top 1% of predictions**:

```21:44:data_ingestion/Com_ip/final_rep_0129.py
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
```

### 6.2 Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Lift @ 1%** | `(TP in top 1%) / (Expected TP if random)` | How much better than random at identifying high-risk members |
| **F1 @ 1%** | F1 score treating top 1% as predicted positives | Precision-recall balance in top bucket |
| **TP @ 1%** | Count of actual positives in top 1% | Absolute capture |
| **AUC** | Standard ROC AUC | Global discrimination ability |

### 6.3 Evaluation Protocol
Models are evaluated on **three holdout sets**:

```183:197:data_ingestion/Com_ip/final_rep_0129.py
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
```

| Dataset | Purpose |
|---------|---------|
| **Validation** | Early stopping & model selection |
| **Test** | In-time generalization |
| **OOT** | Temporal generalization (different time period) |

### 6.4 Reported Results
The final reported Lift @ 1% values appear at the end of the script:

```537:539:data_ingestion/Com_ip/final_rep_0129.py
24.424408307840928
24.860808520939237
25.47818166763269
```

This indicates ~**24–25x lift** at the top 1%, meaning the model identifies positives ~25 times more effectively than random selection.

---

## 7. Model Persistence

Trained models are saved in CatBoost's native format:

```179:179:data_ingestion/Com_ip/final_rep_0129.py
catboost_model.save_model(f"catboost_nofs_ip6.cbm", format = "cbm" )
```

| Saved Model | Description |
|-------------|-------------|
| `catboost_nofs_ip6.cbm` | No feature selection (all 435 features) |
| `catboost_fs_ip6.cbm` | With feature selection (251 features) |
| `catboost_fs_finetuned_ip6.cbm` | Feature selection + tuned hyperparameters |

---

## 8. Summary of Strengths & Observations

### Strengths ✓
1. **Member-level split** prevents data leakage across train/test
2. **Out-of-time validation** tests temporal stability
3. **Business-aligned metrics** (Lift @ 1%) match operational use case
4. **Native categorical handling** via CatBoost avoids encoding pitfalls
5. **Early stopping** prevents overfitting
6. **Exclusion of cost features** reduces potential leakage

### Potential Concerns ⚠️
1. **No pipeline encapsulation**: Preprocessing (imputation) is applied separately to each split, not via a fitted pipeline object. This is acceptable only if imputation is a static rule (0 for numeric, 'missing' for categorical).
2. **Manual hyperparameter search**: Configurations appear hand-tuned rather than systematically optimized.
3. **No calibration analysis**: Only discrimination metrics are reported; probability calibration is not verified.
4. **No confidence intervals**: Lift values are point estimates without bootstrap CIs.
5. **No baseline comparison**: No dummy/logistic regression baseline is shown for context.

---

This report is based entirely on the code in `final_rep_0129.py` and the referenced SQL files. No assumptions were made beyond what is directly observable in the implementation.