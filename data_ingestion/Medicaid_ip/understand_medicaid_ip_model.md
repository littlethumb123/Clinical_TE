# Medicaid IP Model: Complete Pipeline Documentation

This document provides a comprehensive guide to the Medicaid IP (Inpatient Hospitalization) prediction model, including the downstream evaluation pipeline for testing new transformer embeddings.

---

## 1. Overview

**Objective**: Predict which Medicaid members are at risk for acute inpatient hospitalization within 6 months of an index date.

**Use Case**: Care management intervention targeting - identifying high-risk members for proactive outreach.

**Original Developer**: Eric Ma (Intern) - Medicaid IP Model Refresh

---

## 2. Data Sources

### 2.1 BigQuery Tables

| Table | Purpose |
|-------|---------|
| `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_non_embedding_features` | Non-embedding features (demographics, claims, utilization, conditions) |
| `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_embeddings` | Pre-computed embeddings (emb0-emb255) - 256 dimensions |
| `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip` | Target variable (`acute_ip_flag`) |

### 2.2 Data Filtering

```sql
WHERE NOT asdb_plan_key IN (33, 54)  -- Exclude specific plan keys
  AND post_mnths >= 6                 -- Require 6 months post-observation
```

### 2.3 Population Size

- **~2,542,308 members** after filtering
- **~564 total features** initially (before feature selection)
- **~2-3% positive rate** (acute IP admission)

---

## 3. Feature Categories

### 3.1 Selected Features (499 from RFECV)

The CatBoost model uses 499 features selected via Recursive Feature Elimination with Cross-Validation (RFECV):

| Category | Count | Examples |
|----------|-------|----------|
| Embeddings | 256 | `emb0` through `emb255` |
| ED Utilization | 17 | `sum_ed_visits_yr*`, `high_sev_ed_visits_yr*` |
| IP/OP Visits | 6 | `sum_acute_ip_admits_yr*`, `sum_op_visits_yr*` |
| Claims by Setting | 60+ | `emis_*_clm_yr*`, `coe_*_clm_yr*` |
| Chronic Conditions | 34 | `CHF`, `DIA`, `CRF`, `DEP`, `psychoses` |
| Pharmacy | 60+ | `rx_claim_cnt_yr*`, `antipsychotic_days_supply_yr*` |
| Demographics | 7 | `agenbr`, `gender`, `ethnicity_code` |
| SDOH Scores | 28 | `sdi_score`, `adi_score`, `food_access` |

### 3.2 Categorical Features

These require special handling (CatBoost handles natively):

- `coa_population_group`
- `gender` (encoded: M→1, F→0, other→-1)
- `ethnicity_code`
- `primarylanguage_desc`
- `urbsubr`
- CMS screening flags (`cms_*_scrn`)

---

## 4. Preprocessing Pipeline

### 4.1 Missing Value Handling

```python
# Embeddings: Fill with 0
df[emb_cols] = df[emb_cols].fillna(0)

# Numeric columns: Fill with 0
df[numeric_cols] = df[numeric_cols].fillna(0)

# String columns: Fill with empty string
df[string_cols] = df[string_cols].fillna('')
```

### 4.2 Gender Encoding

```python
df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(-1)
```

### 4.3 One-Hot Encoding (XGBoost only)

```python
# Only encode categories with 2000+ occurrences
min_occurrence = 2000
# CatBoost handles categoricals natively - no one-hot needed
```

---

## 5. Train/Validation/Test Split

### 5.1 Split Strategy

```python
# 80% Train, 10% Validation, 10% Test (Stratified by target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=35, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=35, stratify=y_test
)
```

---

## 6. Class Imbalance Handling

### 6.1 Undersampling (Primary Method)

```python
from imblearn.under_sampling import RandomUnderSampler

# CatBoost optimal ratio: 0.2 (5:1 negative-to-positive)
undersample = RandomUnderSampler(sampling_strategy=0.2, random_state=53)
X_train_u, y_train_u = undersample.fit_resample(X_train, y_train)
```

### 6.2 Optimal Ratios by Model

| Model | Optimal Ratio | Meaning |
|-------|---------------|---------|
| CatBoost | 0.2 | 5:1 negative-to-positive |
| XGBoost | 0.03 | 33:1 negative-to-positive |

---

## 7. Feature Selection (RFECV)

### 7.1 Configuration

```python
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

rfecv = RFECV(
    estimator=CatBoostClassifier(random_seed=53, thread_count=15),
    step=10,              # Remove 10 features per iteration
    cv=StratifiedKFold(3), # 3-fold stratified CV
    scoring="roc_auc",     # AUC as selection criterion
    min_features_to_select=1
)
```

### 7.2 Results

| Model | Selected Features | Note |
|-------|------------------|------|
| CatBoost | 499 features | All 256 embeddings included |
| XGBoost (BH) | 68 features | Only 13 embeddings |

---

## 8. Hyperparameter Tuning (Optuna)

### 8.1 CatBoost Best Parameters

```python
# From Optuna optimization (75 trials, AUC = 0.8737)
CATBOOST_TUNED_PARAMS = {
    'learning_rate': 0.015742881221129403,
    'iterations': 2665,
    'l2_leaf_reg': 0.222046549398224,
    'depth': 7,
    'verbose': 0,
    'use_best_model': True,
}
```

### 8.2 XGBoost Best Parameters

```python
# From Optuna optimization (AUC = 0.8685)
XGBOOST_TUNED_PARAMS = {
    'learning_rate': 0.009719586136010807,
    'n_estimators': 1253,
    'max_depth': 8,
    'subsample': 0.9206518647230545,
    'colsample_bytree': 0.672444373451293,
    'tree_method': 'hist',
    'device': 'cuda'
}
```

---

## 9. Evaluation Metrics

### 9.1 Primary Metrics

| Metric | Description | Business Use |
|--------|-------------|--------------|
| **ROC-AUC** | Overall discrimination | Model quality |
| **Lift@1%** | Times better than random in top 1% | Prioritization |
| **Lift@10%** | Times better than random in top 10% | Outreach capacity |
| **PPV@1%** | Precision in top 1% | Intervention efficiency |
| **Sensitivity@1%** | Recall in top 1% | Coverage at top |

### 9.2 Metric Calculations

```python
def lift_at_percentage(y_true, y_prob, pct):
    """Lift = precision@k / baseline_prevalence"""
    k = int(len(y_true) * pct)
    top_k = np.argsort(y_prob)[::-1][:k]
    precision_at_k = y_true[top_k].mean()
    baseline = y_true.mean()
    return precision_at_k / baseline
```

### 9.3 Expected Performance

| Metric | CatBoost | XGBoost |
|--------|----------|---------|
| ROC-AUC | ~0.87 | ~0.87 |
| Lift@1% | ~19-20x | ~19-20x |
| Lift@10% | ~6-7x | ~6-7x |

---

## 10. Downstream Evaluation Pipeline

### 10.1 Purpose

The downstream evaluation pipeline (`moe_flashattn_3_medicaid_downstream.py`) enables testing new transformer embeddings on the Medicaid IP task using:

- **Same preprocessing** as original model
- **Same selected features** (or embeddings only)
- **Same tuned hyperparameters**
- **Same evaluation metrics**

### 10.2 Feature Set Options

| Feature Set | Features | Use Case |
|-------------|----------|----------|
| `embedding_only` | 256 embeddings | Pure embedding performance |
| `tabular_only` | ~243 tabular | Baseline (no embeddings) |
| `hybrid` | ~499 total | Full model comparison |

### 10.3 Usage Example

```python
from moe_flashattn_3_medicaid_downstream import (
    run_medicaid_ip_evaluation,
    evaluate_multiple_embeddings
)

# Single evaluation
result = run_medicaid_ip_evaluation(
    embedding_path="path/to/new_embeddings.npz",
    feature_set='hybrid',
    sample_frac=0.1,  # 10% sample for testing
)
print(f"Test AUC: {result['test_auc_roc']:.4f}")
print(f"Lift@1%: {result['test_lift_1pct']:.2f}x")

# Multiple embeddings comparison
embedding_paths = {
    'exp1_baseline': 'path/to/exp1_embeddings.npz',
    'exp2_flash': 'path/to/exp2_embeddings.npz',
    'exp6_moe': 'path/to/exp6_embeddings.npz',
}
results_df = evaluate_multiple_embeddings(
    embedding_paths,
    feature_sets=['embedding_only', 'hybrid']
)
```

### 10.4 Key Functions

| Function | Purpose |
|----------|---------|
| `load_medicaid_data_from_bigquery()` | Load raw data from BigQuery |
| `preprocess_features()` | Apply original preprocessing |
| `prepare_medicaid_evaluation_data()` | Full data preparation |
| `run_medicaid_ip_evaluation()` | End-to-end evaluation |
| `evaluate_multiple_embeddings()` | Compare multiple experiments |
| `compare_embedding_effects()` | Quantify embedding contribution |

---

## 11. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEDICAID IP EVALUATION PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────┐
    ▼                                                           ▼
┌─────────────────────┐                           ┌─────────────────────┐
│  BigQuery Tables    │                           │  New Embeddings     │
│  - Features         │                           │  - NPZ file         │
│  - Outcomes         │                           │  - From transformer │
└─────────────────────┘                           └─────────────────────┘
                    │                                       │
                    └───────────────┬───────────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │     PREPROCESSING               │
                    │  - Missing value imputation     │
                    │  - Gender encoding              │
                    │  - Merge features + embeddings  │
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
                    │     CLASS IMBALANCE             │
                    │  - Undersample training (0.2)   │
                    │  - Keep val/test as-is          │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │     CATBOOST (Tuned Params)     │
                    │  - lr: 0.0157                   │
                    │  - iterations: 2665             │
                    │  - depth: 7                     │
                    │  - Auto class weights           │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │     EVALUATION                  │
                    │  - ROC-AUC                      │
                    │  - Lift@1%, Lift@10%            │
                    │  - PPV@1%, Sensitivity@1%       │
                    │  - Brier Score                  │
                    └─────────────────────────────────┘
```

---

## 12. File References

| File | Purpose |
|------|---------|
| `dev/moe/moe_flashattn_3_medicaid_downstream.py` | Downstream evaluation pipeline |
| `data_ingestion/Medicaid_ip/dev_archive/descriptive_analyses/catboost.py` | Original CatBoost training |
| `data_ingestion/Medicaid_ip/dev_archive/descriptive_analyses/xgboost.py` | Original XGBoost training |
| `data_ingestion/Medicaid_ip/dev_archive/descriptive_analyses/catboost_selected_features.txt` | 499 selected features |
| `data_ingestion/Medicaid_ip/dev_archive/descriptive_analyses/optuna_results_catboost.csv` | Hyperparameter tuning results |

---

## 13. Replication Checklist

To replicate the original Medicaid IP model exactly:

- [x] **Data filtering**: Exclude plan keys 33, 54; require `post_mnths >= 6`
- [x] **Missing values**: Embeddings/numeric → 0; strings → empty
- [x] **Gender encoding**: M→1, F→0, other→-1
- [x] **Split**: 80/10/10 stratified with `random_state=35`
- [x] **Undersampling**: Ratio 0.2 (CatBoost) with `random_state=53`
- [x] **Feature selection**: Use RFECV-selected 499 features
- [x] **Hyperparameters**: Use Optuna-tuned values
- [x] **Metrics**: ROC-AUC, Lift@1%, Lift@10%, PPV, Sensitivity





