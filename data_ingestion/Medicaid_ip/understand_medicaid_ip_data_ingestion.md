I'll analyze this ML project systematically by examining the README and all the training scripts to understand every aspect of the pipeline.

Let me first explore the DNN_models folder and other subdirectories to get a complete picture:

[7 tools called]

Now let me check the optuna results and feature importance files to understand the final model configurations:

[2 tools called]

Now I have a comprehensive understanding of this ML project. Let me provide a detailed analysis of the entire pipeline framework.

---

# Comprehensive ML Pipeline Analysis: Medicaid IP Model Refresh

## 1. Project Overview

**Project Goal**: Build predictive models to identify Medicaid members at risk for **acute inpatient (IP) hospitalization** within 6 months of an index date. This is a **binary classification** problem used for care management intervention targeting.

**Project Owner**: Eric Ma (Intern) - Medicaid IP Model Refresh

---

## 2. Data Sources & Tables

### 2.1 BigQuery Tables Used

| Table | Purpose |
|-------|---------|
| `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_non_embedding_features` | Non-embedding features (demographics, claims, utilization, conditions) |
| `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_embeddings` | Pre-computed embeddings (emb0-emb255) - 256 dimensions |
| `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip` | Target variable (`acute_ip_flag`) |

### 2.2 Data Filtering Criteria
```sql
WHERE NOT asdb_plan_key IN (33, 54)  -- Exclude specific plan keys
  AND post_mnths >= 6                 -- Require 6 months post-observation
```

### 2.3 Population Size
- **~2,542,308 members** after filtering
- **564 total features** initially (before feature engineering)

---

## 3. Feature Categories

The codebase defines features using a type mapping (`nem_to_type`):

| Type Code | Meaning | Count | Examples |
|-----------|---------|-------|----------|
| 0 | Categorical | ~30 | `urbsubr`, `gender`, `cms_*_scrn`, `coa_population_group`, `primarylanguage_desc` |
| 1 | Continuous | ~200+ | All `*_yr1`, `*_yr2` utilization/claims, scores, days supply |
| 2 | Binary | ~60 | Condition flags (CHF, DIA, DEP, etc.), ED severity flags |

### 3.1 Feature Groups

**A. Demographic Features:**
- `agenbr` (age), `gender`, `ethnicity_code`, `primarylanguage_desc`
- `tenure_yr1`, `tenure_yr2`
- `urbsubr` (urban/suburban/rural)

**B. Social Determinants of Health (SDOH) Scores:**
- `adi_score`, `sdi_score`, `svi_score`
- `acs_social_risk_score`, `csdi_social_risk_score`
- Individual indices: `citizenship_index`, `education_index`, `food_access`, `health_access`, `housing_desert`, `income_index`, `poverty_score`, etc.

**C. Utilization Features (Year 1 & Year 2):**
- ED visits: `sum_ed_visits_yr*`, severity levels (`low_sev_ed_*`, `med_sev_ed_*`, `high_sev_ed_*`)
- IP admits: `sum_acute_ip_admits_yr*`, `sum_acute_calc_los_yr*`
- OP visits: `sum_op_visits_yr*`
- Care setting claims: `emis_*_clm_yr*`, `coe_*_clm_yr*`

**D. Pharmacy Features (Year 1 & Year 2):**
- Counts: `rx_claim_cnt_yr*`, `ndc_cnt_yr*`, `gpi_cnt_yr*`
- Fill types: `retail_fills_yr*`, `mail_order_fills_yr*`, `generic_fills_yr*`
- Drug-specific: `antidiabetic_scripts_yr*`, `antipsychotic_days_supply_yr*`, etc.

**E. Chronic Condition Flags (Binary):**
- 60+ conditions: CHF, DIA, CRF, DEP, ANX, OBE, CHD, DEM, etc.
- `major_chronic_cnt` (aggregate count)

**F. Pre-computed Embeddings:**
- `emb0` through `emb255` (256-dimensional vector from transformer/representation learning)

---

## 4. Data Preprocessing Pipeline

### 4.1 Missing Value Handling

```python
# Step 1: Fill embedding columns with 0
emb_pattern = r'emb[0-255]+'
emb_col = [col for col in df.columns if re.match(emb_pattern, col)]
df[emb_col] = df[emb_col].fillna(0)

# Step 2: Fill numeric columns with 0
for c in df.columns:
    if is_integer(df[c].dtype) or is_float(df[c].dtype):
        df[c] = df[c].fillna(0)
    else:
        df[c] = df[c].fillna('')  # String columns get empty string
```

### 4.2 Categorical Encoding

**Gender Encoding:**
```python
df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(-1)
```

**One-Hot Encoding (with frequency threshold):**
```python
min_occurrence = 2000  # Only encode categories with 2000+ occurrences

encoder = OneHotEncoder(sparse_output=False)
for feature in categorical_features:
    counts = df[feature].value_counts()
    categories_to_keep = counts[counts >= min_occurrence].index
    # One-hot encode only frequent categories
```

**Note**: CatBoost handles categorical features natively via `cat_features` parameter, avoiding explicit one-hot encoding.

### 4.3 Data Merging

```python
df = df.set_index('asdb_member_key')
outcome = outcome.set_index('asdb_member_key')
merged = df.merge(outcome, on='asdb_member_key', how='left')
```

---

## 5. Train/Validation/Test Split

### 5.1 Split Strategy

```python
# 80% Train, 10% Validation, 10% Test (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    merged.iloc[:, 1:560], 
    merged['acute_ip_flag'], 
    test_size=0.2, 
    random_state=35, 
    stratify=merged['acute_ip_flag']
)

X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, 
    test_size=0.5, 
    random_state=35, 
    stratify=y_test
)
```

### 5.2 Label Conversion
```python
y_train = y_train.astype('int')
y_val = y_val.astype('int')
y_test = y_test.astype('int')
```

---

## 6. Class Imbalance Handling

The target variable has severe class imbalance (typical healthcare problem). Multiple strategies are tested:

### 6.1 Random Undersampling (Primary Method)

```python
from imblearn.under_sampling import RandomUnderSampler

ratios = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # Tested ratios

undersample = RandomUnderSampler(sampling_strategy=ratio, random_state=53)
X_train_u, y_train_u = undersample.fit_resample(X_train, y_train)
```

**Findings from experiments:**
- **XGBoost optimal ratio**: 0.03 (3% minority-to-majority)
- **CatBoost optimal ratio**: 0.2 (20% minority-to-majority)

### 6.2 Class Weights (Alternative)

**XGBoost:**
```python
weights = np.where(y_train == 0, 1, len(y_train) / (2 * np.sum(y_train == 1)))
model.fit(X_train, y_train, sample_weight=weights, ...)
```

**CatBoost:**
```python
class_weights = {
    0: 1,
    1: (len(y_train) / (2 * y_train.sum()))
}
CatBoostClassifier(class_weights=class_weights, ...)
```

### 6.3 No Rebalancing (Baseline Comparison)
Both models are also tested without rebalancing to establish baseline performance.

---

## 7. Feature Selection

### 7.1 Recursive Feature Elimination with Cross-Validation (RFECV)

```python
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

xgb_model_rfe = XGBClassifier(
    random_state=53,
    n_jobs=-1,
    tree_method="hist",
    device="cuda"  # GPU acceleration
)

rfecv = RFECV(
    estimator=xgb_model_rfe,
    step=10,              # Remove 10 features per iteration
    cv=StratifiedKFold(3), # 3-fold stratified CV
    scoring="roc_auc",     # AUC as selection criterion
    min_features_to_select=1,
    n_jobs=-1
)

rfecv.fit(X_train, y_train)
selected_features = X_train.columns[rfecv.support_]
```

### 7.2 Feature Selection Results

| Model | Selected Features | Notes |
|-------|------------------|-------|
| CatBoost | 499 features | Includes all 256 embeddings |
| XGBoost (BH variant) | 68 features | Only 13 embeddings selected |

The selected features are saved to text files for reuse:
- `catboost_selected_features.txt`
- `xgboost_selected_features.txt`

---

## 8. Hyperparameter Tuning

### 8.1 Optuna Bayesian Optimization

```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 7000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=75)  # Minimum 50-75 trials
```

### 8.2 Hyperparameter Search Spaces

**XGBoost:**
| Parameter | Range |
|-----------|-------|
| `learning_rate` | 0.001 - 0.3 (log scale) |
| `n_estimators` | 100 - 7000 |
| `max_depth` | 4 - 12 |
| `subsample` | 0.5 - 1.0 |
| `colsample_bytree` | 0.5 - 1.0 |

**CatBoost:**
| Parameter | Range |
|-----------|-------|
| `learning_rate` | 0.001 - 0.3 (log scale) |
| `iterations` | 100 - 7000 |
| `depth` | 4 - 12 |
| `l2_leaf_reg` | 0.01 - 10 (log scale) |

### 8.3 Best Found Hyperparameters

**XGBoost (from optuna_results.csv):**
```python
params = {
    'learning_rate': 0.009719586136010807,
    'n_estimators': 1253,
    'max_depth': 8,
    'subsample': 0.9206518647230545,
    'colsample_bytree': 0.672444373451293,
    'tree_method': "hist",
    'device': "cuda"
}
# Best AUC: ~0.8685
```

**CatBoost (from optuna_results_catboost.csv):**
```python
params = {
    'learning_rate': 0.015742881221129403,
    'iterations': 2665,
    'l2_leaf_reg': 0.222046549398224,
    'depth': 7,
    'verbose': 0
}
# Best AUC: ~0.8737
```

---

## 9. Model Training

### 9.1 XGBoost Final Training

```python
import xgboost as xgb

model = xgb.XGBClassifier(**optimal_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=0
)
model.save_model('xgboost_model.cbm')
```

### 9.2 CatBoost Final Training

```python
from catboost import CatBoostClassifier, Pool

valid_pool = Pool(X_val, y_val, cat_features=string_columns)

catboost_model = CatBoostClassifier(**optimal_params)
catboost_model.fit(
    X_train, y_train,
    eval_set=valid_pool,
    cat_features=string_columns,
    use_best_model=True,
    verbose=0
)
catboost_model.save_model('catboost_model.cbm', format='cbm')
```

---

## 10. Model Evaluation

### 10.1 Primary Metrics

The project uses healthcare-specific metrics focused on **top percentile performance**:

| Metric | Description | Formula |
|--------|-------------|---------|
| **ROC-AUC** | Overall discrimination | Standard AUC |
| **Lift@1%** | How many times better than random in top 1% | (Actual positives in top 1%) / (Expected positives × 0.01) |
| **Lift@10%** | Same for top 10% | (Actual positives in top 10%) / (Expected positives × 0.10) |
| **PPV@1%** | Precision in top 1% | TP / (TP + FP) for top 1% |
| **Sensitivity@1%** | Recall in top 1% | TP / (TP + FN) for top 1% |

### 10.2 Custom Metric Functions

```python
def lift_at_1_percent(y_true, y_scores):
    top_1_percent = max(1, int(0.01 * len(y_true)))
    top_indices = np.argsort(y_scores)[-top_1_percent:]
    actual_positives = y_true.iloc[top_indices].sum()
    expected_positives = y_true.sum() * 0.01
    return actual_positives / expected_positives if expected_positives != 0 else 0
```

### 10.3 Comprehensive Evaluation Function

```python
def calculate_metrics(model):
    y_pred_test = model.predict_proba(X_test)[:, 1]
    y_test_reset = y_test.reset_index(drop=True)
    
    # ROC-AUC
    roc_auc_test = roc_auc_score(y_test, y_pred_test)
    
    # Sort by predicted probability
    idx = np.argsort(y_pred_test)[::-1]
    
    def calculate_percentile_metrics(y_test_reset, idx, percentile):
        cutoff = int(percentile * len(y_test_reset))
        top_indices = idx[:cutoff]
        
        predictions_binary = np.zeros(len(y_test_reset), dtype=int)
        predictions_binary[top_indices] = 1
        
        tn, fp, fn, tp = confusion_matrix(y_test_reset, predictions_binary).ravel()
        
        ppv = 100 * (tp / (tp + fp)) if (tp + fp) > 0 else 0
        sensitivity = 100 * (tp / (tp + fn)) if (tp + fn) > 0 else 0
        
        actual_positives = y_test_reset[top_indices].sum()
        expected = y_test_reset.sum() * percentile
        lift = actual_positives / expected if expected > 0 else 0
        
        return lift, ppv, sensitivity
    
    return metrics_for_1_and_10_percent
```

### 10.4 Expected Model Performance

Based on the Optuna results:
- **ROC-AUC**: ~0.87
- **1% Lift**: ~19-20x
- **10% Lift**: ~6-7x

---

## 11. Visualization & Reporting

### 11.1 Generated Artifacts

| File | Content |
|------|---------|
| `lift_chart_*.csv` | Decile-level lift table |
| `importance_df_*.csv` | Feature importance rankings |
| `optuna_results*.csv` | Hyperparameter trial results |
| `optuna_*.log` | Detailed optimization logs |
| `RFE_CV_Plot.png` | Feature selection curve |
| `*_selected_features.txt` | Final feature list |
| `*_model.cbm` | Serialized model |

### 11.2 Visualization Code

```python
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')

# Lift Curve
def plot_lift_curve(y_val, y_pred, step=0.01):
    aux_lift = pd.DataFrame({'real': y_val, 'predicted': y_pred})
    aux_lift.sort_values('predicted', ascending=False, inplace=True)
    # ... calculate cumulative lift at each percentile

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(model, X_test, y_test, plot_chance_level=True)

# Lift Chart (Decile Table)
def lift_chart(X, actual_target, model):
    # Creates percentile-based lift table with cumulative metrics
```

---

## 12. File Comparison Summary

| Aspect | `xgboost_ELP_edits.py` | `xgboost_results/xgboost.py` | `catboost_results/catboost.py` | `general_train_pipeline.py` |
|--------|------------------------|------------------------------|--------------------------------|----------------------------|
| **Model** | XGBoost | XGBoost | CatBoost | XGBoost (Template) |
| **One-Hot Encoding** | Yes (min 2000) | Yes (min 2000) | No (native categorical) | Yes (min 2000) |
| **Optimal Undersample** | 0.03 | 0.03 | 0.2 | Configurable |
| **Feature Selection** | RFECV | RFECV | RFECV | RFECV |
| **Hyperparameter Tuning** | Optuna | Optuna | Optuna | Optuna |
| **GPU Acceleration** | Yes (CUDA) | Yes (CUDA) | No | Yes (CUDA) |
| **SHAP Explanations** | No | No | No | Yes |
| **Code Organization** | Exploratory | Cleaner | Clean | Modular/Template |
| **Parameterization** | Hardcoded | Hardcoded | Hardcoded | Config section |

---

## 13. Pipeline Framework Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEDICAID IP PREDICTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────┐
    ▼                                                           ▼
┌─────────────────────┐                           ┌─────────────────────┐
│  DATA INGESTION     │                           │  DATA INGESTION     │
│  - Non-embeddings   │                           │  - Embeddings       │
│  - Filters applied  │                           │  - 256 dimensions   │
└─────────────────────┘                           └─────────────────────┘
                    │                                       │
                    └───────────────┬───────────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │     PREPROCESSING               │
                    │  - Missing value imputation     │
                    │  - Gender encoding (M/F→1/0)    │
                    │  - One-hot (min 2000 freq)      │
                    │  - Merge features + outcomes    │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │     TRAIN/VAL/TEST SPLIT        │
                    │  - 80% / 10% / 10%              │
                    │  - Stratified by outcome        │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │     CLASS IMBALANCE HANDLING    │
                    │  - Test multiple ratios         │
                    │  - RandomUnderSampler           │
                    │  - Compare with class weights   │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │     FEATURE SELECTION (RFECV)   │
                    │  - Step size: 10 features       │
                    │  - 3-fold Stratified CV         │
                    │  - Scoring: ROC-AUC             │
                    │  - Save selected features       │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │     XGBOOST       │           │     CATBOOST      │
        │  - GPU: CUDA      │           │  - Native cat     │
        │  - Optuna: 75     │           │  - Optuna: 75     │
        │  - AUC optimized  │           │  - AUC optimized  │
        └───────────────────┘           └───────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │     EVALUATION                  │
                    │  - ROC-AUC                      │
                    │  - Lift@1%, Lift@10%            │
                    │  - PPV@1%, Sensitivity@1%       │
                    │  - Confusion Matrix             │
                    │  - Precision-Recall Curve       │
                    │  - Feature Importance           │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │     ARTIFACTS                   │
                    │  - Model file (.cbm)            │
                    │  - Selected features (.txt)     │
                    │  - Lift chart (.csv)            │
                    │  - Feature importance (.csv)    │
                    │  - Optuna results (.csv/.log)   │
                    └─────────────────────────────────┘
```

---

## 14. Key Insights & Recommendations for Replication

### 14.1 Critical Steps to Preserve
1. **Data filtering**: Must exclude plan keys 33, 54 and require `post_mnths >= 6`
2. **Missing value strategy**: Embeddings and numeric → 0, strings → empty string
3. **Categorical encoding threshold**: 2000 minimum occurrences
4. **Stratified splitting**: Essential for class imbalance
5. **Undersampling before RFECV**: Run feature selection on balanced data

### 14.2 Model-Specific Notes
- **XGBoost**: Benefits from lower undersampling ratio (0.03), requires one-hot encoding
- **CatBoost**: Handles categoricals natively, better with higher ratio (0.2)

### 14.3 Variants in Repository
The `xgboost_results/` folder shows multiple experiment variants:
- `BH/` - Behavioral Health focus
- `FE_only/` - Feature Engineering only (no embeddings?)
- `TE_only/` - Transformer Embeddings only
- `No_BH/` - Without Behavioral Health features

This indicates the team tested different feature subsets to understand contribution of embeddings vs. hand-crafted features.