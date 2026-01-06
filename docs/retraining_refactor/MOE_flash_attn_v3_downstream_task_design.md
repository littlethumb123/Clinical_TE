I'll analyze all the components systematically and create a comprehensive report and plan. Let me first verify some additional details about the data structures.

[3 tools called]

Now let me check the exact feature set used in the previous model:

[1 tool called]

Now I have all the information needed. Let me compile the comprehensive technical report and plan.

---

# Technical Report: Downstream Evaluation of Pretrained Transformer Embeddings for Commercial IP Prediction

## Executive Summary

This report outlines a rigorous plan to evaluate three pretrained transformer embeddings for predicting 6-month acute inpatient (IP) admission risk in commercial members. The evaluation will compare:
1. **Embedding-only** features (256-dimensional transformer representations)
2. **Tabular-only** features (~409 hand-crafted features)
3. **Hybrid** features (embeddings + tabular)

---

## Part 1: Understanding Embedding Generation

### 1.1 Three Pretrained Transformer Models

Based on my analysis of `moe_flashattn_3_downstream.py` (lines 151-167):

| Experiment | Model Class | Architecture Details | Mixed Precision |
|------------|-------------|---------------------|-----------------|
| `exp1_dense_baseline` | `BaselineTransformer` | Standard PyTorch TransformerEncoder, 16 heads, GELU, FP32 | ❌ No |
| `exp2b_flash_learned_pool` | `FlashAttentionTransformer` | Flash Attention + Learned Attention Pooling + SwiGLU, 8 heads, FP16 | ✅ Yes |
| `exp6_auxiliary_free_v3` | `FlashMoETransformer` | Flash Attention + MoE (8 experts, top-2) + DeepSeek bias correction, FP16 | ✅ Yes |

### 1.2 Embedding Extraction Process

From `moe_flashattn_3_core.py` (EmbeddingExtractor class, lines 2635-2761):

**Extraction Layer:**
- **BaselineTransformer**: Hooks on `transformer_encoder_dy` output (the temporal encoder)
- **FlashAttentionTransformer/FlashMoETransformer**: Hooks on input to `model.norm` (after all temporal + MoE layers, before final LayerNorm + decoder)

**Patient-Level Embedding:**
```python
# From get_patient_embedding() at lines 2721-2745
def get_patient_embedding(self, dt_cnt: List[int]) -> torch.Tensor:
    embeddings = self.get_embeddings()  # [batch, len_dy, embedding_size]
    for i, valid_days in enumerate(dt_cnt):
        if valid_days > 0:
            # Get embedding at LAST VALID DAY
            patient_embeddings.append(embeddings[i, valid_days - 1, :])
```

**Rationale for using last-day embedding:**
- The model uses **causal attention** (future days cannot attend to past days)
- The embedding at the last valid day contains the **cumulative representation** of the entire patient history
- This is analogous to the [CLS] token representation in BERT-style models

**Embedding Dimension:** 256 (for all three experiments, from `config.embedding_size`)

### 1.3 Storage Format

From `save_embeddings()` function (lines 805-862 in `moe_flashattn_3_downstream.py`):

**Location:** `embedding_output/exp_round5_3lobs_pretrain_multi_gpu_test_v2/{exp_name}/`

**Files Saved:**
1. **NPZ file** (`embeddings_{exp_name}_{timestamp}.npz`):
   - `embeddings`: np.ndarray [num_members, 256] (float32)
   - `individual_ids`: np.array of strings
   - `index_dts`: np.array of strings
   - `embedding_dim`: 256
   - `num_members`: count
   - `model_path`, `model_type`, `use_mixed_precision`: metadata

2. **CSV file** (`embeddings_{exp_name}_{timestamp}_ids.csv`):
   - `individual_id`, `index_dt`, `embedding_idx` (for easy lookup)

### 1.4 Data Used for Embedding Generation

From `commercial_heldout_transformer_input_4_te_experiment_round_5.sql`:

```sql
-- Source: Full transformer training data
FROM `a834793_Combined_All_LOB_o3_train_ending` AS full_data

-- Filter: Only Commercial LOB members that exist in features table
INNER JOIN `a834793_Commercial_final_dataset_4_te_experiment` AS features
    ON full_data.individual_id = features.individual_id

WHERE full_data.lob = 'Commercial'
-- CRITICAL: Exclude members used in pretraining (prevents leakage)
AND NOT EXISTS (
    SELECT 1 FROM `a834793_Combined_All_LOB_o3_train_10pct_sample` AS pretrain
    WHERE pretrain.individual_id = full_data.individual_id
)
```

**Key Insight:** The heldout transformer input table contains **only members NOT used in pretraining**, ensuring no leakage from pretraining to downstream evaluation.

### 1.5 Sampling Strategy in Python

From `moe_flashattn_3_downstream.py` (lines 977-985):

```python
df_cm['index_dt'] = pd.to_datetime(df_cm['index_dt'])
df_cm_b4_oct = df_cm[df_cm['index_dt'] <= pd.to_datetime("2023-10-16")]
df_cm_after_oct = df_cm[df_cm['index_dt'] > pd.to_datetime("2023-10-16")]
df_cm_b4_oct_sample = df_cm_b4_oct.sample(frac=0.3, random_state=42)
df_cm_after_oct_sample = df_cm_after_oct.sample(frac=0.3, random_state=42)
df_cm_sample = pd.concat([df_cm_b4_oct_sample, df_cm_after_oct])  # ← NOTE: Uses full after_oct!
```

**Observation:** The final sample includes:
- **30% of members with index_dt ≤ 2023-10-16** (for in-time train/val/test)
- **100% of members with index_dt > 2023-10-16** (for out-of-time validation)

This appears intentional: preserve all OOT data while reducing in-time data for efficiency.

---

## Part 2: Previous Downstream Classification Task

### 2.1 Task Definition

From `understand_com_ip_model.md`:

| Aspect | Detail |
|--------|--------|
| **Task** | Binary classification: 6-month acute IP admission prediction |
| **Target** | `ip6` (1 = had acute IP admission, 0 = no admission) |
| **Exclusions** | Maternity, trauma, transplant, non-impactible admissions excluded |
| **Horizon** | 180 days from index date |

### 2.2 Previous Data Split Strategy

From `final_rep_0129.py` (lines 102-153):

| Split | Criterion | Approx. % | Purpose |
|-------|-----------|-----------|---------|
| **Train** | `ind_id_last_digit` 0-7 | 80% | Model training |
| **Validation** | `ind_id_last_digit` 8 | 10% | Early stopping & model selection |
| **Test** | `ind_id_last_digit` 9 | 10% | In-time generalization |
| **OOT** | Separate table (temporal) | Variable | Temporal generalization |

**Additional Filters Applied:**
- `exclude_ip = 0`: No prior IP exclusions
- `include_post_6_status = 1`: Valid enrollment during outcome window

**Member Deduplication:**
```python
df = df.drop_duplicates('individual_id', keep='last')
```
Keeps one row per member (last occurrence).

### 2.3 Previous Feature Set

**Initial features:** 435 (from `catboost_rfs_435.csv`)

**Excluded features (26):** Cost amounts, outreach flags, aggregate flags
```python
exludel = ["clm_allowed_amt_1yr", ..., "talked"]  # 26 features
```

**Final features:** ~409 after exclusions, further reduced to 251 in optimized runs

**Feature Categories (from SQL analysis):**
| Category | Count | Examples |
|----------|-------|----------|
| Demographics | ~14 | age, gender_cd, product_ln_cd |
| HPD Chronic Conditions | ~94 | Binary flags for chronic diseases |
| MDC/Utilization | ~332 | Major Diagnostic Category, case mix |
| Lab Results | ~72 | Lab test indicators |
| Membership/Geographic | ~19 | Region, plan type |
| Cost/Utilization PMPM | ~68 | Per-member-per-month metrics |

### 2.4 Previous Model and Evaluation

**Model:** CatBoostClassifier with native categorical handling

**Hyperparameters (best configuration from code):**
```python
param = {
    'iterations': 2436, 'depth': 7, 'learning_rate': 0.027,
    'grow_policy': 'SymmetricTree', 'bootstrap_type': 'Bernoulli',
    'subsample': 0.79, 'od_wait': 84, 'use_best_model': True
}
```

**Evaluation Metrics:**
- **Lift @ 1%**: ~24-25x (main business metric)
- **F1 @ 1%**: Precision-recall balance in top bucket
- **AUC-ROC**: Global discrimination

**Reported Results:** Lift @ 1% of ~24-25x across validation, test, and OOT sets.

---

## Part 3: Data Join Strategy

### 3.1 Key Tables and Join Keys

**Table A: Transformer Embeddings** (from NPZ/CSV files)
- Keys: `individual_id`, `index_dt`
- Columns: `embedding_0` ... `embedding_255` (256 columns)
- Source: Sampled from `a964286_commercial_heldout_transformer_input_4_te_experiment_round_5`

**Table B: Features + Outcomes** (from BigQuery)
- Table: `a834793_Commercial_final_dataset_4_te_experiment`
- Keys: `individual_id`, `index_dt`
- Columns:
  - 14 demographics columns
  - ~599 feature columns
  - 3 outcome columns (`ip6`, `sum_ip6_admits`, `sum_ip6_los`)
  - 3 continuity flags (`mon_3_include`, `mon_6_include`, `mon_12_include`)
  - `ind_id_last_digit` (for splits)

### 3.2 Join Logic

From `commercial_ip_outcome_generation.sql` (lines 3525-3530), the expected join is:

```python
merged_df = embeddings_df.merge(
    features_df, 
    on=['individual_id', 'index_dt'],  # Both keys required
    how='inner'
)
```

**Critical Observation:** 
- Both tables are keyed by `(individual_id, index_dt)` - a **composite key**
- The embedding table was generated from members in the features table (via INNER JOIN in SQL)
- Therefore, the join should be a **1:1 match** with no missing rows (if sampling is consistent)

### 3.3 Potential Data Integrity Issues

1. **Sampling Mismatch:** Embeddings were generated for a 30% sample + OOT. Must filter features table to same members.

2. **Date Format:** Embeddings have `index_dt` as string (from `astype(str)`). Features table has DATE type. Must ensure format alignment.

3. **Member Deduplication:** Previous model used `drop_duplicates('individual_id', keep='last')`. For fair comparison, apply same logic.

### 3.4 Proposed Join Procedure

```
Step 1: Load embeddings from NPZ
        → Extract: embeddings, individual_ids, index_dts

Step 2: Create embedding DataFrame
        → Columns: individual_id, index_dt, embedding_0...embedding_255

Step 3: Load features from BigQuery
        → Query: SELECT * FROM a834793_Commercial_final_dataset_4_te_experiment
        → Filter: WHERE individual_id IN (embedding_individual_ids)

Step 4: Join on (individual_id, index_dt)
        → Verify: All embedding rows matched (or log unmatched)

Step 5: Apply eligibility filters
        → WHERE exclude_ip IS NULL OR exclude_ip = 0 (if column exists)
        → WHERE mon_6_include = 1 (ensure valid outcome window)

Step 6: Deduplicate (if needed)
        → drop_duplicates('individual_id', keep='last')
```

---

## Part 4: Downstream Evaluation Plan

### 4.1 Three Feature Sets

| Feature Set | Columns | Description |
|-------------|---------|-------------|
| **Embedding-only** | 256 | Transformer embeddings (embedding_0...embedding_255) |
| **Tabular-only** | ~409 | Hand-crafted features (excluding leakage columns) |
| **Hybrid** | ~665 | Concatenation of embeddings + tabular features |

### 4.2 Model Suite

Following the downstream classification rules:

| Model | Purpose | Embedding-only | Tabular-only | Hybrid |
|-------|---------|----------------|--------------|--------|
| **Dummy (stratified)** | Baseline reference | ✅ | ✅ | ✅ |
| **Logistic Regression** | Linear probe (standard for embeddings) | ✅ | ✅ | ✅ |
| **XGBoost** | Strong non-linear baseline | ✅ | ✅ | ✅ |
| **CatBoost** | Native categorical handling (for tabular) | ❌ | ✅ | ✅ |

### 4.3 Data Split Strategy

**Option A: Replicate Previous Split (Recommended for Fair Comparison)**
- Use `ind_id_last_digit` for member-level splits:
  - Train: digits 0-7 (80%)
  - Val: digit 8 (10%)
  - Test: digit 9 (10%)
  - OOT: index_dt > 2023-10-16

**Option B: Temporal Split Only**
- Train: index_dt ≤ 2023-07-01
- Val: 2023-07-01 < index_dt ≤ 2023-10-16
- Test (OOT): index_dt > 2023-10-16

**Recommended:** Option A (matches previous evaluation for fair comparison)

### 4.4 Preprocessing Pipeline

**For Embedding-only:**
1. StandardScaler (fit on train only)
2. Optional: PCA for dimensionality reduction (experimental)

**For Tabular-only:**
1. Numeric: fillna(0) + StandardScaler (for LR only)
2. Categorical: fillna('missing') + pass to CatBoost native OR one-hot encode

**For Hybrid:**
1. Embeddings: StandardScaler
2. Tabular: Same as tabular-only
3. Concatenate after preprocessing

**Critical:** All preprocessing fit on **training data only**, then applied to val/test/OOT.

### 4.5 Evaluation Metrics

**Primary Metrics (Business-Aligned):**
- Lift @ 1% (main metric from previous model)
- Lift @ 10%
- True Positives @ 1%
- F1 @ 1%

**Standard Metrics:**
- AUC-ROC
- AUC-PR (Average Precision)
- Brier Score (calibration)

**Uncertainty Quantification:**
- 95% Bootstrap CIs for all metrics
- DeLong test for AUC comparisons

### 4.6 Results Table Structure

| Feature Set | Model | Train AUC | Val AUC | Test AUC | OOT AUC | Lift@1% (Test) | Lift@1% (OOT) |
|-------------|-------|-----------|---------|----------|---------|----------------|---------------|
| Embedding-only (exp1) | LR | | | | | | |
| Embedding-only (exp1) | XGB | | | | | | |
| Embedding-only (exp2b) | LR | | | | | | |
| ... | ... | | | | | | |
| Tabular-only | CatBoost | | | | | | |
| Hybrid (exp6+tab) | XGB | | | | | | |
| ... | ... | | | | | | |

---

## Part 5: Implementation Plan (Not to Execute Yet)

### Phase 1: Data Preparation

**Step 1.1: Load and Validate Embeddings**
```
For each exp_name in [exp1_dense_baseline, exp2b_flash_learned_pool, exp6_auxiliary_free_v3]:
    - Load NPZ file from embedding_output/{exp_name}/
    - Validate: shape should be (N, 256)
    - Extract: embeddings, individual_ids, index_dts
    - Create DataFrame with individual_id, index_dt, embedding_0...embedding_255
```

**Step 1.2: Load Features + Outcomes from BigQuery**
```
Query:
SELECT 
    individual_id, 
    index_dt,
    ind_id_last_digit,  -- For splits
    ip6,                 -- Target
    mon_6_include,       -- Eligibility filter
    [all ~409 feature columns]
FROM `a834793_Commercial_final_dataset_4_te_experiment`
WHERE individual_id IN (
    SELECT DISTINCT individual_id FROM embedding_individual_ids
)
```

**Step 1.3: Join and Validate**
```
- Merge on (individual_id, index_dt)
- Assert: all embedding rows matched (inner join should be 100%)
- Apply filters: mon_6_include = 1
- Log: row counts before/after filtering
```

**Step 1.4: Create Splits**
```
- Train: ind_id_last_digit IN (0,1,2,3,4,5,6,7)
- Val: ind_id_last_digit = 8
- Test: ind_id_last_digit = 9 AND index_dt <= '2023-10-16'
- OOT: index_dt > '2023-10-16'
```

### Phase 2: Feature Set Construction

**Step 2.1: Define Feature Columns**
```
EXCLUDE_COLUMNS = [
    'individual_id', 'member_id', 'index_dt', 'birth_dt', 'feature_end_dt',
    'ip6', 'sum_ip6_admits', 'sum_ip6_los', 'sum_ip6_acu_days',  # Outcomes
    'mon_3_include', 'mon_6_include', 'mon_12_include',  # Flags
    'ind_id_last_digit',  # Split key
    # Plus 26 leakage columns from previous model
    'clm_allowed_amt_1yr', ..., 'talked'
]

TABULAR_FEATURES = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
EMBEDDING_FEATURES = [f'embedding_{i}' for i in range(256)]
```

**Step 2.2: Create Feature Matrices**
```
X_tab = df[TABULAR_FEATURES]
X_emb = df[EMBEDDING_FEATURES]
X_hybrid = pd.concat([X_tab, X_emb], axis=1)
y = df['ip6'].astype(int)
```

### Phase 3: Model Training

**Step 3.1: Create Preprocessing Pipelines**
```
# For Logistic Regression (needs scaling)
lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# For XGBoost (handles missing, no scaling needed)
xgb_pipeline = Pipeline([
    ('model', XGBClassifier(scale_pos_weight=neg/pos, eval_metric='auc'))
])

# For CatBoost (native categorical)
catboost_params = {
    'iterations': 2500, 'depth': 7, 'learning_rate': 0.025,
    'grow_policy': 'SymmetricTree', 'od_wait': 80,
    'cat_features': cat_column_indices
}
```

**Step 3.2: Train Each Model**
```
For each (feature_set, model) combination:
    - Fit on train (using val for early stopping if applicable)
    - Predict probabilities on val, test, OOT
    - Compute all metrics
    - Store results
```

### Phase 4: Evaluation and Comparison

**Step 4.1: Compute Metrics**
```
metrics_to_compute = {
    'auc_roc': roc_auc_score,
    'auc_pr': average_precision_score,
    'brier': brier_score_loss,
    'lift_1pct': lift_at_percentage(y, y_prob, 0.01),
    'lift_10pct': lift_at_percentage(y, y_prob, 0.10),
    'tp_at_1pct': true_positives_at_percentage(y, y_prob, 0.01),
    'f1_at_1pct': f1_at_percentage(y, y_prob, 0.01),
}
```

**Step 4.2: Bootstrap Confidence Intervals**
```
For each metric:
    - 1000 bootstrap samples
    - Compute 2.5% and 97.5% percentiles
    - Report: metric (95% CI: lower, upper)
```

**Step 4.3: Statistical Comparisons**
```
- DeLong test: Compare AUC between embedding-only vs tabular-only
- McNemar test: Compare error rates at fixed threshold
```

### Phase 5: Deliverables

1. **Saved Artifacts:**
   - Merged datasets (parquet)
   - Train/val/test/OOT split indices
   - Trained model checkpoints
   - Predictions (y_true, y_prob for each split)

2. **Results Table:** Comprehensive comparison across all feature sets and models

3. **Visualizations:**
   - ROC curves (overlay by feature set)
   - PR curves
   - Calibration plots
   - Lift curves

4. **Report:**
   - Key findings: Does embedding add value over tabular?
   - Where do embeddings help most? (subgroup analysis)
   - Recommendations for production

---

## Part 6: Potential Issues and Mitigations

### 6.1 Data Integrity Concerns

| Issue | Risk Level | Mitigation |
|-------|------------|------------|
| Sampling mismatch between embeddings and features | High | Verify 100% join rate; log unmatched rows |
| Date format misalignment | Medium | Standardize to YYYY-MM-DD string before join |
| Duplicate member rows | Medium | Apply same deduplication as previous model |

### 6.2 Leakage Risks

| Issue | Risk Level | Mitigation |
|-------|------------|------------|
| Pretraining leakage | Low (already handled) | SQL excludes pretrained members |
| Split leakage (member in train+test) | Low | Using ind_id_last_digit ensures member-level split |
| Temporal leakage | Low | Embeddings use only lookback data; OOT uses later dates |
| Pipeline leakage | Medium | Must fit scalers/encoders on train only |

### 6.3 Comparability Concerns

| Issue | Risk Level | Mitigation |
|-------|------------|------------|
| Different cohorts (sampling) | High | Use same eligibility filters as previous model |
| Different features (new table) | Medium | Verify overlap with previous 409 features |
| Different hyperparameters | Medium | Use same CatBoost config for fair comparison |

---

## Summary

This plan provides a rigorous framework for evaluating transformer embeddings against traditional tabular features for Commercial IP prediction. Key design decisions:

1. **Join on (individual_id, index_dt)** - Composite key ensures exact member-timepoint matching
2. **Replicate previous splits** using `ind_id_last_digit` for fair comparison
3. **Three feature sets** (embedding, tabular, hybrid) × multiple models
4. **Business-aligned metrics** (Lift@1%) as primary evaluation criterion
5. **Bootstrap CIs and statistical tests** for robust comparisons

**Next step:** Await your confirmation to proceed with implementation.

# Downstream Evaluation Implementation_v1 (legacy)

## Overview

This implementation follows directly from `moe_flashattn_3_downstream.py` (after embedding generation). We'll:
1. Load saved embeddings
2. Load tabular features + outcomes from BigQuery
3. Join on `(individual_id, index_dt)`
4. Create 3 feature sets: Embedding-only, Tabular-only, Hybrid
5. Train and evaluate models
6. Compare results

---

## Step 1: Additional Imports (Add after line 68 in your notebook)

Copy this cell after your existing imports:

```python
# ============================================================================
# ADDITIONAL IMPORTS FOR DOWNSTREAM EVALUATION
# ============================================================================
# These are needed beyond what's already imported in moe_flashattn_3_downstream.py

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

# Reuse metrics from moe_flashattn_3_core.py
# These are already available from the import at line 28
# lift_at_percentage, precision_at_percentage, etc. are in DownstreamConfig scope
```

---

## Step 2: Define Constants and Paths

```python
# ============================================================================
# CONSTANTS AND PATHS
# ============================================================================

# Embedding output directory (where embeddings were saved)
EMBEDDING_OUTPUT_DIR = "embedding_output/exp_round5_3lobs_pretrain_multi_gpu_test_v2"

# Experiment names (same as MODEL_PATHS keys)
EXPERIMENT_NAMES = [
    'exp1_dense_baseline',
    'exp2b_flash_learned_pool', 
    'exp6_auxiliary_free_v3'
]

# BigQuery project and dataset
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"

# Features table (contains tabular features + outcomes)
FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_Commercial_final_dataset_4_te_experiment"

# Temporal split date (from the embedding generation sampling)
OOT_CUTOFF_DATE = "2023-10-16"

# Target column
TARGET_COLUMN = "ip6"

# Columns to exclude from features (outcomes, keys, flags, leakage)
EXCLUDE_COLUMNS = [
    # Keys and identifiers
    'individual_id', 'member_id', 'index_dt', 'birth_dt', 'feature_end_dt',
    
    # Outcome columns (target and related)
    'ip6', 'sum_ip6_admits', 'sum_ip6_los', 'sum_ip6_acu_days',
    
    # Eligibility/continuity flags
    'mon_3_include', 'mon_6_include', 'mon_12_include',
    'exclude_ip', 'include_post_6_status',
    
    # Split key
    'ind_id_last_digit',
    
    # Leakage columns (cost amounts, outreach flags from previous model)
    'clm_allowed_amt_1yr', 'clm_allowed_amt_2yr', 'clm_allowed_amt_3mo', 'clm_allowed_amt_6mo',
    'clm_paid_amt_1yr', 'clm_paid_amt_2yr', 'clm_paid_amt_3mo', 'clm_paid_amt_6mo',
    'clm_par_allowed_amt_1yr', 'clm_par_allowed_amt_2yr', 'clm_par_allowed_amt_3mo', 'clm_par_allowed_amt_6mo',
    'clm_par_paid_amt_1yr', 'clm_par_paid_amt_2yr', 'clm_par_paid_amt_3mo', 'clm_par_paid_amt_6mo',
    'clm_srv_copay_amt_1yr', 'clm_srv_copay_amt_3mo', 'clm_srv_copay_amt_6mo',
    'covid_19', 'hpd_major_flag', 'chronic',
    'txt_member', 'txt_referral', 'txt_1yr_outreach', 'talked'
]

print(f"Embedding directory: {EMBEDDING_OUTPUT_DIR}")
print(f"Features table: {FEATURES_TABLE}")
print(f"OOT cutoff date: {OOT_CUTOFF_DATE}")
print(f"Target column: {TARGET_COLUMN}")
print(f"Excluded columns: {len(EXCLUDE_COLUMNS)}")
```

---

## Step 3: Load Saved Embeddings

```python
# ============================================================================
# STEP 3: LOAD SAVED EMBEDDINGS
# ============================================================================
# Load embeddings from NPZ files saved during embedding generation

import glob

def load_embeddings_from_dir(exp_name: str, base_dir: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load embeddings from the NPZ file for a given experiment.
    
    Args:
        exp_name: Experiment name (e.g., 'exp1_dense_baseline')
        base_dir: Base directory containing experiment subdirectories
        
    Returns:
        embeddings: numpy array [num_members, 256]
        individual_ids: list of member IDs
        index_dts: list of index dates
    """
    exp_dir = os.path.join(base_dir, exp_name)
    
    # Find the NPZ file (there should be one)
    npz_files = glob.glob(os.path.join(exp_dir, "embeddings_*.npz"))
    
    if not npz_files:
        raise FileNotFoundError(f"No NPZ file found in {exp_dir}")
    
    # Use the most recent one if multiple exist
    npz_path = sorted(npz_files)[-1]
    print(f"Loading: {npz_path}")
    
    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)
    
    embeddings = data['embeddings']
    individual_ids = data['individual_ids'].tolist()
    index_dts = data['index_dts'].tolist()
    
    print(f"  Shape: {embeddings.shape}")
    print(f"  Members: {len(individual_ids):,}")
    
    return embeddings, individual_ids, index_dts


# Load all embeddings
print("="*70)
print("LOADING SAVED EMBEDDINGS")
print("="*70)

embeddings_dict = {}
for exp_name in EXPERIMENT_NAMES:
    print(f"\n--- {exp_name} ---")
    embeddings, ids, dts = load_embeddings_from_dir(exp_name, EMBEDDING_OUTPUT_DIR)
    embeddings_dict[exp_name] = {
        'embeddings': embeddings,
        'individual_ids': ids,
        'index_dts': dts
    }

# Verify all experiments have the same members (they should)
ref_ids = set(embeddings_dict[EXPERIMENT_NAMES[0]]['individual_ids'])
for exp_name in EXPERIMENT_NAMES[1:]:
    other_ids = set(embeddings_dict[exp_name]['individual_ids'])
    if ref_ids != other_ids:
        print(f"⚠️ Warning: {exp_name} has different members!")
        print(f"   Overlap: {len(ref_ids & other_ids):,}")
    else:
        print(f"✅ {exp_name} has same members as reference")
```

---

## Step 4: Create Embedding DataFrames

```python
# ============================================================================
# STEP 4: CREATE EMBEDDING DATAFRAMES
# ============================================================================
# Convert embeddings to DataFrames with proper column names

def create_embedding_df(exp_name: str, emb_data: dict) -> pd.DataFrame:
    """
    Create a DataFrame from embedding data.
    
    Columns: individual_id, index_dt, embedding_0...embedding_255
    """
    embeddings = emb_data['embeddings']
    individual_ids = emb_data['individual_ids']
    index_dts = emb_data['index_dts']
    embedding_dim = embeddings.shape[1]
    
    # Create DataFrame with ID columns
    df = pd.DataFrame({
        'individual_id': individual_ids,
        'index_dt': index_dts
    })
    
    # Add embedding columns
    embedding_cols = [f'embedding_{i}' for i in range(embedding_dim)]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    
    # Combine
    df = pd.concat([df, embedding_df], axis=1)
    
    # Add experiment name for tracking
    df['_exp_name'] = exp_name
    
    return df


# Create DataFrames for each experiment
embedding_dfs = {}
for exp_name in EXPERIMENT_NAMES:
    embedding_dfs[exp_name] = create_embedding_df(exp_name, embeddings_dict[exp_name])
    print(f"{exp_name}: {embedding_dfs[exp_name].shape}")

# Use first experiment's embedding df as reference for joining
# (All should have same members)
emb_df_ref = embedding_dfs[EXPERIMENT_NAMES[0]].copy()
print(f"\nReference embedding DataFrame shape: {emb_df_ref.shape}")
print(f"Columns: {emb_df_ref.columns.tolist()[:5]} ... (256 embedding columns)")
```

---

## Step 5: Load Features + Outcomes from BigQuery

```python
# ============================================================================
# STEP 5: LOAD FEATURES + OUTCOMES FROM BIGQUERY
# ============================================================================
# Load the tabular features table and filter to embedding members only

# Get the unique individual_ids from embeddings
embedding_member_ids = emb_df_ref['individual_id'].unique().tolist()
print(f"Embedding members: {len(embedding_member_ids):,}")

# Create a temp table or use IN clause (for smaller datasets)
# For efficiency, we'll query with a filter
# Note: For very large member lists, consider creating a temp table

# Query features table
print(f"\nLoading features from: {FEATURES_TABLE}")
print("This may take a few minutes...")

# Build the query - select all columns, filter to embedding members
features_sql = f"""
SELECT *
FROM `{FEATURES_TABLE}`
WHERE individual_id IN UNNEST(@member_ids)
"""

# Use parameterized query for safety
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter("member_ids", "STRING", embedding_member_ids)
    ]
)

# Execute query
df_features = client.query(features_sql, job_config=job_config).to_dataframe()

print(f"\nFeatures loaded: {df_features.shape}")
print(f"Columns: {len(df_features.columns)}")

# Check for required columns
required_cols = ['individual_id', 'index_dt', TARGET_COLUMN, 'ind_id_last_digit']
missing_cols = [c for c in required_cols if c not in df_features.columns]
if missing_cols:
    print(f"⚠️ Missing required columns: {missing_cols}")
else:
    print("✅ All required columns present")

# Show sample
print(f"\nSample of features DataFrame:")
print(df_features[['individual_id', 'index_dt', TARGET_COLUMN, 'ind_id_last_digit']].head())
```

---

## Step 6: Join Embeddings with Features

```python
# ============================================================================
# STEP 6: JOIN EMBEDDINGS WITH FEATURES
# ============================================================================
# Join on (individual_id, index_dt) - composite key

# Ensure date formats match
emb_df_ref['index_dt'] = pd.to_datetime(emb_df_ref['index_dt']).dt.strftime('%Y-%m-%d')
df_features['index_dt'] = pd.to_datetime(df_features['index_dt']).dt.strftime('%Y-%m-%d')

# Perform the join
print("Joining embeddings with features on (individual_id, index_dt)...")

# Keep only embedding columns (not the tracking column) for join
embedding_cols = [c for c in emb_df_ref.columns if c.startswith('embedding_')]
emb_for_join = emb_df_ref[['individual_id', 'index_dt'] + embedding_cols].copy()

df_merged = df_features.merge(
    emb_for_join,
    on=['individual_id', 'index_dt'],
    how='inner'  # Only keep matched rows
)

print(f"\nJoin results:")
print(f"  Features rows:   {len(df_features):,}")
print(f"  Embeddings rows: {len(emb_for_join):,}")
print(f"  Merged rows:     {len(df_merged):,}")
print(f"  Match rate:      {len(df_merged)/len(emb_for_join)*100:.1f}%")

# Check for duplicates
dup_count = df_merged.duplicated(subset=['individual_id', 'index_dt']).sum()
if dup_count > 0:
    print(f"⚠️ Found {dup_count} duplicate (individual_id, index_dt) pairs")
    # Deduplicate keeping last (same as previous model)
    df_merged = df_merged.drop_duplicates(subset=['individual_id', 'index_dt'], keep='last')
    print(f"  After dedup: {len(df_merged):,}")
else:
    print("✅ No duplicates found")
```

---

## Step 7: Apply Eligibility Filters

```python
# ============================================================================
# STEP 7: APPLY ELIGIBILITY FILTERS
# ============================================================================
# Same filters as the previous model to ensure fair comparison

print("Applying eligibility filters...")
print(f"Before filtering: {len(df_merged):,} rows")

# Filter 1: mon_6_include = 1 (valid 6-month outcome window)
if 'mon_6_include' in df_merged.columns:
    df_merged = df_merged[df_merged['mon_6_include'] == 1]
    print(f"  After mon_6_include=1: {len(df_merged):,}")

# Filter 2: exclude_ip = 0 or NULL (no prior IP exclusion)
if 'exclude_ip' in df_merged.columns:
    df_merged = df_merged[(df_merged['exclude_ip'] == 0) | (df_merged['exclude_ip'].isna())]
    print(f"  After exclude_ip=0: {len(df_merged):,}")

# Filter 3: include_post_6_status = 1 (valid enrollment during outcome window)
if 'include_post_6_status' in df_merged.columns:
    df_merged = df_merged[df_merged['include_post_6_status'] == 1]
    print(f"  After include_post_6_status=1: {len(df_merged):,}")

print(f"\nAfter all filters: {len(df_merged):,} rows")

# Check target distribution
print(f"\nTarget distribution ({TARGET_COLUMN}):")
print(df_merged[TARGET_COLUMN].value_counts())
print(f"Prevalence: {df_merged[TARGET_COLUMN].mean()*100:.2f}%")
```

---

## Step 8: Create Data Splits

```python
# ============================================================================
# STEP 8: CREATE DATA SPLITS
# ============================================================================
# Replicate the previous model's split strategy using ind_id_last_digit

# Convert index_dt to datetime for OOT split
df_merged['index_dt_parsed'] = pd.to_datetime(df_merged['index_dt'])

# Create split masks
train_mask = df_merged['ind_id_last_digit'].isin([0, 1, 2, 3, 4, 5, 6, 7])
val_mask = df_merged['ind_id_last_digit'] == 8
test_mask = (df_merged['ind_id_last_digit'] == 9) & (df_merged['index_dt_parsed'] <= OOT_CUTOFF_DATE)
oot_mask = df_merged['index_dt_parsed'] > OOT_CUTOFF_DATE

# Create split DataFrames
df_train = df_merged[train_mask].copy()
df_val = df_merged[val_mask].copy()
df_test = df_merged[test_mask].copy()
df_oot = df_merged[oot_mask].copy()

# Print split statistics
print("="*70)
print("DATA SPLITS")
print("="*70)
splits = {
    'Train (digit 0-7)': df_train,
    'Validation (digit 8)': df_val,
    'Test (digit 9, in-time)': df_test,
    f'OOT (after {OOT_CUTOFF_DATE})': df_oot
}

for split_name, split_df in splits.items():
    if len(split_df) > 0:
        prevalence = split_df[TARGET_COLUMN].mean() * 100
        positives = split_df[TARGET_COLUMN].sum()
        print(f"{split_name}:")
        print(f"  Rows: {len(split_df):,} | Positives: {positives:,} | Prevalence: {prevalence:.2f}%")
    else:
        print(f"{split_name}: EMPTY")

# Verify no member overlap between splits
train_ids = set(df_train['individual_id'])
test_ids = set(df_test['individual_id'])
val_ids = set(df_val['individual_id'])

overlap_train_test = train_ids & test_ids
overlap_train_val = train_ids & val_ids
if overlap_train_test or overlap_train_val:
    print(f"\n⚠️ Member overlap detected!")
else:
    print("\n✅ No member overlap between train/val/test")
```

---

## Step 9: Identify Feature Columns

```python
# ============================================================================
# STEP 9: IDENTIFY FEATURE COLUMNS
# ============================================================================
# Separate tabular features from embeddings, excluding non-feature columns

# Get all columns
all_columns = df_merged.columns.tolist()

# Embedding columns (256 features)
EMBEDDING_FEATURES = [c for c in all_columns if c.startswith('embedding_')]
print(f"Embedding features: {len(EMBEDDING_FEATURES)}")

# Tabular columns (all others except excluded)
all_excluded = EXCLUDE_COLUMNS + EMBEDDING_FEATURES + ['index_dt_parsed', '_exp_name']
TABULAR_FEATURES = [c for c in all_columns if c not in all_excluded and c != TARGET_COLUMN]
print(f"Tabular features: {len(TABULAR_FEATURES)}")

# Identify categorical vs numeric columns
df_tab = df_train[TABULAR_FEATURES]
CATEGORICAL_FEATURES = df_tab.select_dtypes(include=['object', 'category']).columns.tolist()
NUMERIC_FEATURES = df_tab.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical features: {len(CATEGORICAL_FEATURES)}")
print(f"Numeric features: {len(NUMERIC_FEATURES)}")

# Print some examples
print(f"\nSample categorical: {CATEGORICAL_FEATURES[:5]}")
print(f"Sample numeric: {NUMERIC_FEATURES[:5]}")
```

---

## Step 10: Prepare Feature Matrices

```python
# ============================================================================
# STEP 10: PREPARE FEATURE MATRICES
# ============================================================================
# Create X and y for each split and each feature set

def prepare_features(df: pd.DataFrame, feature_cols: List[str], target_col: str):
    """
    Prepare feature matrix X and target y from DataFrame.
    Handles missing values in a basic way.
    """
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()
    
    # Handle missing values
    # For numeric: fill with 0
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    # For categorical: fill with 'missing'
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    X[cat_cols] = X[cat_cols].fillna('missing')
    
    return X, y


# Prepare for each split
print("Preparing feature matrices...")

# Train
X_train_emb, y_train = prepare_features(df_train, EMBEDDING_FEATURES, TARGET_COLUMN)
X_train_tab, _ = prepare_features(df_train, TABULAR_FEATURES, TARGET_COLUMN)
X_train_hybrid = pd.concat([X_train_tab, X_train_emb], axis=1)

# Validation
X_val_emb, y_val = prepare_features(df_val, EMBEDDING_FEATURES, TARGET_COLUMN)
X_val_tab, _ = prepare_features(df_val, TABULAR_FEATURES, TARGET_COLUMN)
X_val_hybrid = pd.concat([X_val_tab, X_val_emb], axis=1)

# Test
X_test_emb, y_test = prepare_features(df_test, EMBEDDING_FEATURES, TARGET_COLUMN)
X_test_tab, _ = prepare_features(df_test, TABULAR_FEATURES, TARGET_COLUMN)
X_test_hybrid = pd.concat([X_test_tab, X_test_emb], axis=1)

# OOT
X_oot_emb, y_oot = prepare_features(df_oot, EMBEDDING_FEATURES, TARGET_COLUMN)
X_oot_tab, _ = prepare_features(df_oot, TABULAR_FEATURES, TARGET_COLUMN)
X_oot_hybrid = pd.concat([X_oot_tab, X_oot_emb], axis=1)

print(f"\nFeature matrix shapes:")
print(f"  Train - Emb: {X_train_emb.shape}, Tab: {X_train_tab.shape}, Hybrid: {X_train_hybrid.shape}")
print(f"  Val   - Emb: {X_val_emb.shape}, Tab: {X_val_tab.shape}, Hybrid: {X_val_hybrid.shape}")
print(f"  Test  - Emb: {X_test_emb.shape}, Tab: {X_test_tab.shape}, Hybrid: {X_test_hybrid.shape}")
print(f"  OOT   - Emb: {X_oot_emb.shape}, Tab: {X_oot_tab.shape}, Hybrid: {X_oot_hybrid.shape}")
```

---

## Step 11: Define Evaluation Metrics

```python
# ============================================================================
# STEP 11: DEFINE EVALUATION METRICS
# ============================================================================
# Metrics for comparing models - reuse from moe_flashattn_3_core.py where possible

def lift_at_percentage(y_true, y_prob, pct: float) -> float:
    """Calculate lift at top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    indices = np.argsort(y_prob)[::-1][:k]
    
    if isinstance(y_true, pd.Series):
        precision_at_k = y_true.iloc[indices].mean()
        baseline = y_true.mean()
    else:
        precision_at_k = np.mean(np.array(y_true)[indices])
        baseline = np.mean(y_true)
    
    return precision_at_k / baseline if baseline > 0 else 0.0


def true_positives_at_percentage(y_true, y_prob, pct: float) -> int:
    """Count true positives in top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    indices = np.argsort(y_prob)[::-1][:k]
    
    if isinstance(y_true, pd.Series):
        return int(y_true.iloc[indices].sum())
    else:
        return int(np.sum(np.array(y_true)[indices]))


def precision_at_percentage(y_true, y_prob, pct: float) -> float:
    """Calculate precision at top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    indices = np.argsort(y_prob)[::-1][:k]
    
    if isinstance(y_true, pd.Series):
        return float(y_true.iloc[indices].mean())
    else:
        return float(np.mean(np.array(y_true)[indices]))


def compute_all_metrics(y_true, y_prob, split_name: str = "") -> Dict:
    """
    Compute all evaluation metrics for a prediction.
    
    Returns dict with all metrics.
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Binary predictions at 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        f'{split_name}_auc_roc': roc_auc_score(y_true, y_prob),
        f'{split_name}_auc_pr': average_precision_score(y_true, y_prob),
        f'{split_name}_brier': brier_score_loss(y_true, y_prob),
        f'{split_name}_lift_1pct': lift_at_percentage(y_true, y_prob, 0.01),
        f'{split_name}_lift_10pct': lift_at_percentage(y_true, y_prob, 0.10),
        f'{split_name}_tp_1pct': true_positives_at_percentage(y_true, y_prob, 0.01),
        f'{split_name}_precision_1pct': precision_at_percentage(y_true, y_prob, 0.01),
        f'{split_name}_n_samples': len(y_true),
        f'{split_name}_n_positives': int(y_true.sum()),
        f'{split_name}_prevalence': float(y_true.mean()),
    }
    
    return metrics


print("Evaluation metrics defined:")
print("  - AUC-ROC, AUC-PR (discrimination)")
print("  - Brier score (calibration)")
print("  - Lift@1%, Lift@10% (business metric)")
print("  - TP@1%, Precision@1% (top-bucket performance)")
```

---

## Step 12: Train and Evaluate Models - Embedding Only

```python
# ============================================================================
# STEP 12A: TRAIN MODELS - EMBEDDING ONLY (LOGISTIC REGRESSION)
# ============================================================================
# Logistic regression is the standard "linear probe" for embedding evaluation

print("="*70)
print("TRAINING: EMBEDDING-ONLY MODELS")
print("="*70)

# Store results
results = []

# For each embedding experiment
for exp_name in EXPERIMENT_NAMES:
    print(f"\n--- {exp_name} ---")
    
    # Load embeddings for this experiment
    emb_data = embeddings_dict[exp_name]
    
    # Create embedding DataFrame and join with merged data
    exp_emb_df = create_embedding_df(exp_name, emb_data)
    exp_emb_df['index_dt'] = pd.to_datetime(exp_emb_df['index_dt']).dt.strftime('%Y-%m-%d')
    
    # Get embeddings for each split by individual_id + index_dt
    # Use the already-created splits but replace embedding columns
    
    # Simpler approach: use the merged df and just swap embeddings
    # Since all experiments should have same members, we can use position-based alignment
    
    # Actually, we need to join again to ensure alignment
    exp_emb_cols = [c for c in exp_emb_df.columns if c.startswith('embedding_')]
    
    # Join with train/val/test/oot
    X_train_exp = df_train.merge(
        exp_emb_df[['individual_id', 'index_dt'] + exp_emb_cols],
        on=['individual_id', 'index_dt'],
        how='inner'
    )[exp_emb_cols]
    
    X_val_exp = df_val.merge(
        exp_emb_df[['individual_id', 'index_dt'] + exp_emb_cols],
        on=['individual_id', 'index_dt'],
        how='inner'
    )[exp_emb_cols]
    
    X_test_exp = df_test.merge(
        exp_emb_df[['individual_id', 'index_dt'] + exp_emb_cols],
        on=['individual_id', 'index_dt'],
        how='inner'
    )[exp_emb_cols]
    
    X_oot_exp = df_oot.merge(
        exp_emb_df[['individual_id', 'index_dt'] + exp_emb_cols],
        on=['individual_id', 'index_dt'],
        how='inner'
    )[exp_emb_cols]
    
    # ========================================================================
    # LOGISTIC REGRESSION (with scaling)
    # ========================================================================
    print("  Training Logistic Regression...")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_exp)
    X_val_scaled = scaler.transform(X_val_exp)
    X_test_scaled = scaler.transform(X_test_exp)
    X_oot_scaled = scaler.transform(X_oot_exp) if len(X_oot_exp) > 0 else None
    
    # Train logistic regression
    lr_model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_prob_val = lr_model.predict_proba(X_val_scaled)[:, 1]
    y_prob_test = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_oot = lr_model.predict_proba(X_oot_scaled)[:, 1] if X_oot_scaled is not None else None
    
    # Compute metrics
    result = {
        'exp_name': exp_name,
        'feature_set': 'embedding_only',
        'model': 'LogisticRegression',
    }
    result.update(compute_all_metrics(y_val, y_prob_val, 'val'))
    result.update(compute_all_metrics(y_test, y_prob_test, 'test'))
    if y_prob_oot is not None and len(y_oot) > 0:
        result.update(compute_all_metrics(y_oot, y_prob_oot, 'oot'))
    
    results.append(result)
    
    print(f"    Val AUC: {result['val_auc_roc']:.4f}, Lift@1%: {result['val_lift_1pct']:.2f}")
    print(f"    Test AUC: {result['test_auc_roc']:.4f}, Lift@1%: {result['test_lift_1pct']:.2f}")
    if 'oot_auc_roc' in result:
        print(f"    OOT AUC: {result['oot_auc_roc']:.4f}, Lift@1%: {result['oot_lift_1pct']:.2f}")

print("\n✅ Embedding-only (LR) training complete")
```

---

## Step 13: Train and Evaluate Models - Tabular Only

```python
# ============================================================================
# STEP 13: TRAIN MODELS - TABULAR ONLY
# ============================================================================
# CatBoost with native categorical handling (same as previous model)

print("="*70)
print("TRAINING: TABULAR-ONLY MODELS")
print("="*70)

# Identify categorical columns for CatBoost
cat_cols = X_train_tab.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical columns for CatBoost: {len(cat_cols)}")

# Create CatBoost pools
train_pool = Pool(X_train_tab, y_train, cat_features=cat_cols)
val_pool = Pool(X_val_tab, y_val, cat_features=cat_cols)
test_pool = Pool(X_test_tab, y_test, cat_features=cat_cols)
oot_pool = Pool(X_oot_tab, y_oot, cat_features=cat_cols) if len(X_oot_tab) > 0 else None

# Train CatBoost (similar config to previous model)
print("\nTraining CatBoost (Tabular-only)...")

catboost_model = CatBoostClassifier(
    iterations=2500,
    depth=7,
    learning_rate=0.025,
    grow_policy='SymmetricTree',
    bootstrap_type='Bernoulli',
    subsample=0.8,
    od_wait=80,
    use_best_model=True,
    random_seed=42,
    thread_count=-1,
    verbose=100  # Print every 100 iterations
)

catboost_model.fit(
    train_pool,
    eval_set=val_pool,
    plot=False
)

# Predictions
y_prob_val_tab = catboost_model.predict_proba(val_pool)[:, 1]
y_prob_test_tab = catboost_model.predict_proba(test_pool)[:, 1]
y_prob_oot_tab = catboost_model.predict_proba(oot_pool)[:, 1] if oot_pool is not None else None

# Compute metrics
result_tab = {
    'exp_name': 'tabular',
    'feature_set': 'tabular_only',
    'model': 'CatBoost',
}
result_tab.update(compute_all_metrics(y_val, y_prob_val_tab, 'val'))
result_tab.update(compute_all_metrics(y_test, y_prob_test_tab, 'test'))
if y_prob_oot_tab is not None and len(y_oot) > 0:
    result_tab.update(compute_all_metrics(y_oot, y_prob_oot_tab, 'oot'))

results.append(result_tab)

print(f"\nTabular CatBoost Results:")
print(f"  Val AUC: {result_tab['val_auc_roc']:.4f}, Lift@1%: {result_tab['val_lift_1pct']:.2f}")
print(f"  Test AUC: {result_tab['test_auc_roc']:.4f}, Lift@1%: {result_tab['test_lift_1pct']:.2f}")
if 'oot_auc_roc' in result_tab:
    print(f"  OOT AUC: {result_tab['oot_auc_roc']:.4f}, Lift@1%: {result_tab['oot_lift_1pct']:.2f}")

print("\n✅ Tabular-only training complete")
```

---

## Step 14: Train and Evaluate Models - Hybrid

```python
# ============================================================================
# STEP 14: TRAIN MODELS - HYBRID (EMBEDDINGS + TABULAR)
# ============================================================================
# Combine embeddings with tabular features

print("="*70)
print("TRAINING: HYBRID MODELS (Embedding + Tabular)")
print("="*70)

for exp_name in EXPERIMENT_NAMES:
    print(f"\n--- {exp_name} (Hybrid) ---")
    
    # Get embeddings for this experiment
    emb_data = embeddings_dict[exp_name]
    exp_emb_df = create_embedding_df(exp_name, emb_data)
    exp_emb_df['index_dt'] = pd.to_datetime(exp_emb_df['index_dt']).dt.strftime('%Y-%m-%d')
    exp_emb_cols = [c for c in exp_emb_df.columns if c.startswith('embedding_')]
    
    # Create hybrid feature sets for each split
    # Join embeddings with tabular
    
    def create_hybrid_split(split_df, split_tab_X, emb_df, emb_cols):
        """Join tabular features with embeddings for a split."""
        # Get embeddings matching this split
        matched_emb = split_df.merge(
            emb_df[['individual_id', 'index_dt'] + emb_cols],
            on=['individual_id', 'index_dt'],
            how='inner'
        )[emb_cols]
        
        # Reset indices to align
        split_tab_reset = split_tab_X.reset_index(drop=True)
        matched_emb_reset = matched_emb.reset_index(drop=True)
        
        # Combine
        hybrid = pd.concat([split_tab_reset, matched_emb_reset], axis=1)
        return hybrid
    
    X_train_hybrid_exp = create_hybrid_split(df_train, X_train_tab, exp_emb_df, exp_emb_cols)
    X_val_hybrid_exp = create_hybrid_split(df_val, X_val_tab, exp_emb_df, exp_emb_cols)
    X_test_hybrid_exp = create_hybrid_split(df_test, X_test_tab, exp_emb_df, exp_emb_cols)
    X_oot_hybrid_exp = create_hybrid_split(df_oot, X_oot_tab, exp_emb_df, exp_emb_cols) if len(df_oot) > 0 else None
    
    # For hybrid, we use XGBoost (handles mixed numeric/categorical after encoding)
    # Embedding columns are numeric, tabular may have categoricals
    
    # One-hot encode categoricals for XGBoost
    cat_cols = X_train_tab.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Simple approach: use CatBoost which handles this natively
    # All embedding columns are numeric, CatBoost will auto-detect
    
    print("  Training CatBoost (Hybrid)...")
    
    # Create pools - only tabular columns need cat_features specification
    # Embedding columns are float, so not categoricals
    cat_cols_in_hybrid = [c for c in cat_cols if c in X_train_hybrid_exp.columns]
    
    train_pool_h = Pool(X_train_hybrid_exp, y_train, cat_features=cat_cols_in_hybrid)
    val_pool_h = Pool(X_val_hybrid_exp, y_val, cat_features=cat_cols_in_hybrid)
    test_pool_h = Pool(X_test_hybrid_exp, y_test, cat_features=cat_cols_in_hybrid)
    oot_pool_h = Pool(X_oot_hybrid_exp, y_oot, cat_features=cat_cols_in_hybrid) if X_oot_hybrid_exp is not None else None
    
    # Train
    catboost_hybrid = CatBoostClassifier(
        iterations=2500,
        depth=7,
        learning_rate=0.025,
        grow_policy='SymmetricTree',
        bootstrap_type='Bernoulli',
        subsample=0.8,
        od_wait=80,
        use_best_model=True,
        random_seed=42,
        thread_count=-1,
        verbose=0  # Silent for brevity
    )
    
    catboost_hybrid.fit(train_pool_h, eval_set=val_pool_h, plot=False)
    
    # Predictions
    y_prob_val_h = catboost_hybrid.predict_proba(val_pool_h)[:, 1]
    y_prob_test_h = catboost_hybrid.predict_proba(test_pool_h)[:, 1]
    y_prob_oot_h = catboost_hybrid.predict_proba(oot_pool_h)[:, 1] if oot_pool_h is not None else None
    
    # Compute metrics
    result_h = {
        'exp_name': exp_name,
        'feature_set': 'hybrid',
        'model': 'CatBoost',
    }
    result_h.update(compute_all_metrics(y_val, y_prob_val_h, 'val'))
    result_h.update(compute_all_metrics(y_test, y_prob_test_h, 'test'))
    if y_prob_oot_h is not None and len(y_oot) > 0:
        result_h.update(compute_all_metrics(y_oot, y_prob_oot_h, 'oot'))
    
    results.append(result_h)
    
    print(f"    Val AUC: {result_h['val_auc_roc']:.4f}, Lift@1%: {result_h['val_lift_1pct']:.2f}")
    print(f"    Test AUC: {result_h['test_auc_roc']:.4f}, Lift@1%: {result_h['test_lift_1pct']:.2f}")
    if 'oot_auc_roc' in result_h:
        print(f"    OOT AUC: {result_h['oot_auc_roc']:.4f}, Lift@1%: {result_h['oot_lift_1pct']:.2f}")

print("\n✅ Hybrid training complete")
```

---

## Step 15: Create Results Summary Table

```python
# ============================================================================
# STEP 15: CREATE RESULTS SUMMARY TABLE
# ============================================================================

print("="*70)
print("RESULTS SUMMARY")
print("="*70)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Select key columns for display
display_cols = [
    'exp_name', 'feature_set', 'model',
    'val_auc_roc', 'val_lift_1pct', 
    'test_auc_roc', 'test_lift_1pct',
    'oot_auc_roc', 'oot_lift_1pct'
]

# Filter to existing columns
display_cols = [c for c in display_cols if c in results_df.columns]
results_display = results_df[display_cols].copy()

# Format numeric columns
for col in results_display.columns:
    if 'auc' in col:
        results_display[col] = results_display[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    elif 'lift' in col:
        results_display[col] = results_display[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

print("\n" + results_display.to_string(index=False))

# Save results
output_results_path = os.path.join(EMBEDDING_OUTPUT_DIR, "downstream_evaluation_results.csv")
results_df.to_csv(output_results_path, index=False)
print(f"\n✅ Results saved to: {output_results_path}")
```

---

## Step 16: Key Findings Summary

```python
# ============================================================================
# STEP 16: KEY FINDINGS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# Find best embedding-only model
emb_results = results_df[results_df['feature_set'] == 'embedding_only']
best_emb = emb_results.loc[emb_results['test_lift_1pct'].idxmax()]

# Find tabular-only model
tab_results = results_df[results_df['feature_set'] == 'tabular_only']
if len(tab_results) > 0:
    tab_result = tab_results.iloc[0]
else:
    tab_result = None

# Find best hybrid model
hybrid_results = results_df[results_df['feature_set'] == 'hybrid']
if len(hybrid_results) > 0:
    best_hybrid = hybrid_results.loc[hybrid_results['test_lift_1pct'].idxmax()]
else:
    best_hybrid = None

print("\n1. BEST EMBEDDING-ONLY MODEL:")
print(f"   Experiment: {best_emb['exp_name']}")
print(f"   Test AUC: {best_emb['test_auc_roc']:.4f}")
print(f"   Test Lift@1%: {best_emb['test_lift_1pct']:.2f}")

if tab_result is not None:
    print("\n2. TABULAR-ONLY (CatBoost) BASELINE:")
    print(f"   Test AUC: {tab_result['test_auc_roc']:.4f}")
    print(f"   Test Lift@1%: {tab_result['test_lift_1pct']:.2f}")

if best_hybrid is not None:
    print("\n3. BEST HYBRID MODEL:")
    print(f"   Experiment: {best_hybrid['exp_name']}")
    print(f"   Test AUC: {best_hybrid['test_auc_roc']:.4f}")
    print(f"   Test Lift@1%: {best_hybrid['test_lift_1pct']:.2f}")

# Compare: Does embedding add value over tabular?
if tab_result is not None and best_hybrid is not None:
    tab_lift = tab_result['test_lift_1pct']
    hybrid_lift = best_hybrid['test_lift_1pct']
    improvement = (hybrid_lift - tab_lift) / tab_lift * 100
    
    print(f"\n4. EMBEDDING VALUE-ADD:")
    print(f"   Tabular Lift@1%: {tab_lift:.2f}")
    print(f"   Best Hybrid Lift@1%: {hybrid_lift:.2f}")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print("   ✅ Embeddings ADD value over tabular features")
    else:
        print("   ⚠️ Embeddings do NOT add significant value")

print("\n" + "="*70)
```

---

## Summary of What This Implementation Does

1. **Step 1-2**: Additional imports and constants
2. **Step 3-4**: Load saved NPZ embeddings from each experiment
3. **Step 5-6**: Load tabular features from BigQuery and join with embeddings
4. **Step 7**: Apply eligibility filters (same as previous model)
5. **Step 8**: Create train/val/test/OOT splits using `ind_id_last_digit`
6. **Step 9-10**: Prepare feature matrices for each feature set
7. **Step 11**: Define evaluation metrics (Lift@1%, AUC, etc.)
8. **Step 12**: Train Logistic Regression on embedding-only features (for each experiment)
9. **Step 13**: Train CatBoost on tabular-only features (baseline)
10. **Step 14**: Train CatBoost on hybrid features (embedding + tabular)
11. **Step 15-16**: Summarize results and compare

The code is designed to be **copied cell-by-cell** into your Jupyter notebook, continuing from where `moe_flashattn_3_downstream.py` left off after embedding generation.


# Downstream Evaluation Implementation_v2
I'll design a modular, optimized version of the downstream evaluation code. The key improvements:

1. **Clear function separation** - each function does one thing well
2. **Single main interface function** with the inputs you specified
3. **Reusable metric computation** that returns a standardized dictionary
4. **Efficient data handling** - avoid redundant operations

Here's the optimized, modularized code:

```python
"""
Downstream Evaluation Module for Transformer Embeddings
========================================================
Modular, efficient code for evaluating pretrained embeddings on downstream tasks.

Interface:
    evaluate_downstream_task(
        embedding_location_path: str,
        ml_model_object,
        df_features: pd.DataFrame,
        exp_name: str,
        feature_set: str = 'embedding_only',
        apply_scaling: bool = False,
        cat_features: Optional[List[str]] = None
    ) -> Dict

Example:
    from sklearn.linear_model import LogisticRegression
    
    # Load features externally
    df_features = client.query("SELECT * FROM ...").to_dataframe()
    
    # Configure model externally
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    
    # Evaluate
    results = evaluate_downstream_task(
        embedding_location_path="embedding_output/exp1_dense_baseline/embeddings_*.npz",
        ml_model_object=lr_model,
        df_features=df_features,
        exp_name="exp1_dense_baseline",
        feature_set='embedding_only',
        apply_scaling=True
    )
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

TARGET_COLUMN = "ip6"
OOT_CUTOFF_DATE = "2023-10-16"

# Columns to exclude from tabular features
EXCLUDE_COLUMNS = frozenset([
    # Keys and identifiers
    'individual_id', 'member_id', 'index_dt', 'birth_dt', 'feature_end_dt',
    # Outcomes
    'ip6', 'sum_ip6_admits', 'sum_ip6_los', 'sum_ip6_acu_days',
    # Eligibility flags
    'mon_3_include', 'mon_6_include', 'mon_12_include',
    'exclude_ip', 'include_post_6_status',
    # Split key
    'ind_id_last_digit',
    # Leakage columns
    'clm_allowed_amt_1yr', 'clm_allowed_amt_2yr', 'clm_allowed_amt_3mo', 'clm_allowed_amt_6mo',
    'clm_paid_amt_1yr', 'clm_paid_amt_2yr', 'clm_paid_amt_3mo', 'clm_paid_amt_6mo',
    'clm_par_allowed_amt_1yr', 'clm_par_allowed_amt_2yr', 'clm_par_allowed_amt_3mo', 'clm_par_allowed_amt_6mo',
    'clm_par_paid_amt_1yr', 'clm_par_paid_amt_2yr', 'clm_par_paid_amt_3mo', 'clm_par_paid_amt_6mo',
    'clm_srv_copay_amt_1yr', 'clm_srv_copay_amt_3mo', 'clm_srv_copay_amt_6mo',
    'covid_19', 'hpd_major_flag', 'chronic',
    'txt_member', 'txt_referral', 'txt_1yr_outreach', 'talked'
])


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def lift_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """Calculate lift at top percentile. Lift = precision@k / baseline_prevalence."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    precision_at_k = y_true[top_k_indices].mean()
    baseline = y_true.mean()
    return precision_at_k / baseline if baseline > 0 else 0.0


def true_positives_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> int:
    """Count true positives in top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    return int(y_true[top_k_indices].sum())


def precision_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """Calculate precision at top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_indices].mean())


def compute_split_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics for a single split.
    
    Returns:
        Dict with metric names as keys (without split prefix)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    return {
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'lift_1pct': lift_at_percentage(y_true, y_prob, 0.01),
        'lift_5pct': lift_at_percentage(y_true, y_prob, 0.05),
        'lift_10pct': lift_at_percentage(y_true, y_prob, 0.10),
        'tp_1pct': true_positives_at_percentage(y_true, y_prob, 0.01),
        'precision_1pct': precision_at_percentage(y_true, y_prob, 0.01),
        'n_samples': len(y_true),
        'n_positives': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
    }


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_embeddings(embedding_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from NPZ file.
    
    Args:
        embedding_path: Path to NPZ file or directory containing NPZ files
        
    Returns:
        Tuple of (embeddings, individual_ids, index_dts)
    """
    # Handle directory or file path
    if os.path.isdir(embedding_path):
        npz_files = glob.glob(os.path.join(embedding_path, "embeddings_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files found in {embedding_path}")
        npz_path = sorted(npz_files)[-1]  # Use most recent
    elif '*' in embedding_path:
        npz_files = glob.glob(embedding_path)
        if not npz_files:
            raise FileNotFoundError(f"No files matching {embedding_path}")
        npz_path = sorted(npz_files)[-1]
    else:
        npz_path = embedding_path
    
    data = np.load(npz_path, allow_pickle=True)
    
    return (
        data['embeddings'],
        data['individual_ids'],
        data['index_dts']
    )


def create_embedding_dataframe(
    embeddings: np.ndarray,
    individual_ids: np.ndarray,
    index_dts: np.ndarray
) -> pd.DataFrame:
    """Convert embedding arrays to DataFrame with proper column names."""
    embedding_dim = embeddings.shape[1]
    embedding_cols = [f'embedding_{i}' for i in range(embedding_dim)]
    
    df = pd.DataFrame({
        'individual_id': individual_ids,
        'index_dt': pd.to_datetime(index_dts).strftime('%Y-%m-%d')
    })
    
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    return pd.concat([df, embedding_df], axis=1)


# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def join_embeddings_with_features(
    emb_df: pd.DataFrame,
    df_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Join embeddings with features on (individual_id, index_dt).
    Applies eligibility filters automatically.
    """
    # Standardize date format
    df_features = df_features.copy()
    df_features['index_dt'] = pd.to_datetime(df_features['index_dt']).dt.strftime('%Y-%m-%d')
    
    # Inner join
    df_merged = df_features.merge(
        emb_df,
        on=['individual_id', 'index_dt'],
        how='inner'
    )
    
    # Apply eligibility filters
    if 'mon_6_include' in df_merged.columns:
        df_merged = df_merged[df_merged['mon_6_include'] == 1]
    if 'exclude_ip' in df_merged.columns:
        df_merged = df_merged[(df_merged['exclude_ip'] == 0) | (df_merged['exclude_ip'].isna())]
    if 'include_post_6_status' in df_merged.columns:
        df_merged = df_merged[df_merged['include_post_6_status'] == 1]
    
    # Remove duplicates
    df_merged = df_merged.drop_duplicates(
        subset=['individual_id', 'index_dt'], 
        keep='last'
    )
    
    return df_merged


def create_data_splits(
    df: pd.DataFrame,
    oot_cutoff_date: str = OOT_CUTOFF_DATE
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test/OOT splits based on ind_id_last_digit and date.
    
    Split logic:
    - Train: digit 0-7 AND date <= cutoff (no temporal leakage)
    - Val: digit 8 AND date <= cutoff
    - Test: digit 9 AND date <= cutoff
    - OOT: date > cutoff (all digits - tests temporal generalization)
    
    Returns:
        Dict with keys 'train', 'val', 'test', 'oot'
    """
    df = df.copy()
    df['_index_dt_parsed'] = pd.to_datetime(df['index_dt'])
    
    oot_cutoff = pd.to_datetime(oot_cutoff_date)
    
    splits = {
        'train': df[(df['ind_id_last_digit'].isin([0,1,2,3,4,5,6,7])) & (df['_index_dt_parsed'] <= oot_cutoff)],
        'val': df[(df['ind_id_last_digit'] == 8) & (df['_index_dt_parsed'] <= oot_cutoff)],
        'test': df[(df['ind_id_last_digit'] == 9) & (df['_index_dt_parsed'] <= oot_cutoff)],
        'oot': df[df['_index_dt_parsed'] > oot_cutoff]
    }
    
    # Remove temp column
    for key in splits:
        splits[key] = splits[key].drop(columns=['_index_dt_parsed'])
    
    return splits


def downsample_negatives(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = 10.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Downsample negative class to achieve target ratio.
    
    This replicates the 10:1 negative sampling strategy used in the previous model
    (table name contained 'neg_10' indicating 10:1 negative sampling).
    
    Args:
        X: Feature matrix
        y: Target series (0/1 binary)
        ratio: Target ratio of negatives to positives (default 10.0 = 10:1)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_resampled, y_resampled)
    
    Example:
        # Original: 100 positives, 10000 negatives (100:1 ratio)
        # After downsample with ratio=10: 100 positives, 1000 negatives (10:1 ratio)
    """
    np.random.seed(random_state)
    
    # Separate positive and negative indices
    pos_mask = y == 1
    neg_mask = y == 0
    
    n_positives = pos_mask.sum()
    n_negatives = neg_mask.sum()
    
    # Calculate target number of negatives
    target_negatives = int(n_positives * ratio)
    
    if target_negatives >= n_negatives:
        # No downsampling needed - already at or below target ratio
        print(f"No downsampling needed: {n_negatives} negatives / {n_positives} positives = {n_negatives/n_positives:.1f}:1")
        return X, y
    
    # Randomly sample negatives
    neg_indices = y[neg_mask].index.tolist()
    sampled_neg_indices = np.random.choice(neg_indices, size=target_negatives, replace=False)
    
    # Combine with all positives
    pos_indices = y[pos_mask].index.tolist()
    selected_indices = pos_indices + list(sampled_neg_indices)
    
    # Shuffle
    np.random.shuffle(selected_indices)
    
    X_resampled = X.loc[selected_indices].reset_index(drop=True)
    y_resampled = y.loc[selected_indices].reset_index(drop=True)
    
    print(f"Downsampled: {n_negatives} -> {target_negatives} negatives "
          f"(ratio: {n_negatives/n_positives:.1f}:1 -> {target_negatives/n_positives:.1f}:1)")
    
    return X_resampled, y_resampled


def identify_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify embedding and tabular feature columns.
    
    Returns:
        Tuple of (embedding_features, tabular_features)
    """
    all_cols = set(df.columns)
    
    embedding_features = sorted([c for c in all_cols if c.startswith('embedding_')])
    
    excluded = EXCLUDE_COLUMNS | set(embedding_features) | {'_exp_name', 'index_dt_parsed', '_index_dt_parsed'}
    tabular_features = sorted([
        c for c in all_cols 
        if c not in excluded and c != TARGET_COLUMN
    ])
    
    return embedding_features, tabular_features


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target y, handling missing values.
    """
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int)
    
    # Fill missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    X[cat_cols] = X[cat_cols].fillna('missing')
    
    return X, y


# =============================================================================
# DATA PREPARATION FUNCTION (DECOUPLED - PREPARE ONCE, USE MANY TIMES)
# =============================================================================

@dataclass
class PreparedData:
    """Container for prepared evaluation data. Prepare once, evaluate many models."""
    X_splits: Dict[str, pd.DataFrame]
    y_splits: Dict[str, pd.Series]
    feature_cols: List[str]
    embedding_features: List[str]
    tabular_features: List[str]
    cat_feature_indices: List[int]  # Column indices for CatBoost
    feature_set: str
    embedding_path: str
    downsampled: bool = False  # Whether training data was downsampled


def prepare_evaluation_data(
    df_features: pd.DataFrame,
    embedding_location_path: str = "",
    feature_set: str = 'embedding_only',
    oot_cutoff_date: str = OOT_CUTOFF_DATE,
    downsample_ratio: Optional[float] = None,
    random_state: int = 42
) -> PreparedData:
    """
    Prepare data once for multiple model evaluations.
    
    This function decouples data preparation from model training, allowing
    the same prepared data to be used across multiple models efficiently.
    
    Args:
        df_features: DataFrame with features and outcomes (from BigQuery)
        embedding_location_path: Path to NPZ file or directory (not needed for tabular_only)
        feature_set: One of 'embedding_only', 'tabular_only', 'hybrid'
        oot_cutoff_date: Date string for OOT split cutoff
        downsample_ratio: If provided, downsample training negatives to this ratio 
                          (e.g., 10.0 for 10:1 negative:positive ratio). 
                          Only applied to training set. Set to 10.0 to match previous model.
        random_state: Random seed for downsampling reproducibility
    
    Returns:
        PreparedData object containing X_splits, y_splits, and metadata
    """
    valid_feature_sets = {'embedding_only', 'tabular_only', 'hybrid'}
    if feature_set not in valid_feature_sets:
        raise ValueError(f"feature_set must be one of {valid_feature_sets}")
    
    # Load embeddings and merge (skip for tabular_only)
    if feature_set != 'tabular_only':
        embeddings, individual_ids, index_dts = load_embeddings(embedding_location_path)
        emb_df = create_embedding_dataframe(embeddings, individual_ids, index_dts)
        df_merged = join_embeddings_with_features(emb_df, df_features)
    else:
        # For tabular-only, just apply filters directly
        df_merged = df_features.copy()
        df_merged['index_dt'] = pd.to_datetime(df_merged['index_dt']).dt.strftime('%Y-%m-%d')
        if 'mon_6_include' in df_merged.columns:
            df_merged = df_merged[df_merged['mon_6_include'] == 1]
        if 'exclude_ip' in df_merged.columns:
            df_merged = df_merged[(df_merged['exclude_ip'] == 0) | (df_merged['exclude_ip'].isna())]
        if 'include_post_6_status' in df_merged.columns:
            df_merged = df_merged[df_merged['include_post_6_status'] == 1]
        df_merged = df_merged.drop_duplicates(subset=['individual_id', 'index_dt'], keep='last')
    
    # Create splits
    splits = create_data_splits(df_merged, oot_cutoff_date)
    
    # Identify feature columns
    embedding_features, tabular_features = identify_feature_columns(df_merged)
    
    # Select features based on feature_set
    if feature_set == 'embedding_only':
        feature_cols = embedding_features
    elif feature_set == 'tabular_only':
        feature_cols = tabular_features
    else:  # hybrid
        feature_cols = tabular_features + embedding_features
    
    # Prepare feature matrices for each split
    X_splits, y_splits = {}, {}
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            X_splits[split_name], y_splits[split_name] = prepare_features(split_df, feature_cols)
    
    # Apply downsampling to training set only (if requested)
    downsampled = False
    if downsample_ratio is not None and 'train' in X_splits:
        print(f"Applying {downsample_ratio}:1 negative downsampling to training set...")
        X_splits['train'], y_splits['train'] = downsample_negatives(
            X_splits['train'], 
            y_splits['train'], 
            ratio=downsample_ratio,
            random_state=random_state
        )
        downsampled = True
    
    # Pre-compute categorical column indices for CatBoost (only for tabular/hybrid)
    cat_feature_indices = []
    if feature_set != 'embedding_only' and 'train' in X_splits:
        cat_cols = X_splits['train'].select_dtypes(include=['object', 'category']).columns
        cat_feature_indices = [X_splits['train'].columns.get_loc(c) for c in cat_cols]
    
    return PreparedData(
        X_splits=X_splits,
        y_splits=y_splits,
        feature_cols=feature_cols,
        embedding_features=embedding_features,
        tabular_features=tabular_features,
        cat_feature_indices=cat_feature_indices,
        feature_set=feature_set,
        embedding_path=embedding_location_path,
        downsampled=downsampled
    )


# =============================================================================
# MODEL EVALUATION FUNCTION (USES PREPARED DATA)
# =============================================================================

def evaluate_model_on_splits(
    model,
    X_splits: Dict[str, pd.DataFrame],
    y_splits: Dict[str, pd.Series],
    apply_scaling: bool = False,
    cat_feature_indices: Optional[List[int]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Train model on train split, evaluate on all splits.
    
    Args:
        model: sklearn-compatible model with fit() and predict_proba()
        X_splits: Dict with 'train', 'val', 'test', 'oot' DataFrames
        y_splits: Dict with corresponding target Series
        apply_scaling: Whether to apply StandardScaler
        cat_feature_indices: List of categorical column indices (for CatBoost)
    
    Returns:
        Dict with split names as keys, metrics dict as values
    """
    # Clone model to avoid modifying original
    model = clone(model)
    
    X_train, y_train = X_splits['train'], y_splits['train']
    
    # Handle scaling
    scaler = None
    if apply_scaling:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
    else:
        X_train_processed = X_train
    
    # Handle CatBoost-specific training
    model_type = type(model).__name__
    
    if model_type == 'CatBoostClassifier' and cat_feature_indices:
        from catboost import Pool
        
        train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
        val_pool = Pool(X_splits['val'], y_splits['val'], cat_features=cat_feature_indices)
        
        model.fit(train_pool, eval_set=val_pool, verbose=0)
    else:
        model.fit(X_train_processed, y_train)
    
    # Evaluate on all splits
    results = {}
    for split_name in ['val', 'test', 'oot']:
        X_split = X_splits.get(split_name)
        y_split = y_splits.get(split_name)
        
        if X_split is None or len(X_split) == 0:
            continue
        
        # Apply same preprocessing
        if apply_scaling and scaler is not None:
            X_processed = scaler.transform(X_split)
        else:
            X_processed = X_split
        
        # Predict
        if model_type == 'CatBoostClassifier' and cat_feature_indices:
            from catboost import Pool
            pool = Pool(X_split, cat_features=cat_feature_indices)
            y_prob = model.predict_proba(pool)[:, 1]
        else:
            y_prob = model.predict_proba(X_processed)[:, 1]
        
        # Compute metrics
        results[split_name] = compute_split_metrics(np.array(y_split), y_prob)
    
    return results


def evaluate_with_prepared_data(
    prepared_data: PreparedData,
    ml_model_object: Any,
    exp_name: str,
    apply_scaling: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a model using pre-prepared data.
    
    This is the efficient way to evaluate multiple models on the same dataset.
    Call prepare_evaluation_data() once, then call this function for each model.
    
    Args:
        prepared_data: PreparedData object from prepare_evaluation_data()
        ml_model_object: Pre-configured sklearn-compatible model
        exp_name: Experiment name for result identification
        apply_scaling: Whether to apply StandardScaler (True for LR, False for tree-based)
    
    Returns:
        Dict with exp_name, model_type, feature_set, and all metrics
    """
    # Determine if we should use categorical features
    use_cat_features = (
        prepared_data.feature_set != 'embedding_only' and 
        len(prepared_data.cat_feature_indices) > 0
    )
    
    # Evaluate model
    split_results = evaluate_model_on_splits(
        model=ml_model_object,
        X_splits=prepared_data.X_splits,
        y_splits=prepared_data.y_splits,
        apply_scaling=apply_scaling,
        cat_feature_indices=prepared_data.cat_feature_indices if use_cat_features else None
    )
    
    # Build output dictionary
    output = {
        'exp_name': exp_name,
        'model_type': type(ml_model_object).__name__,
        'feature_set': prepared_data.feature_set,
        'n_features': len(prepared_data.feature_cols),
    }
    
    # Flatten split results with prefixes
    for split_name, metrics in split_results.items():
        for metric_name, value in metrics.items():
            output[f'{split_name}_{metric_name}'] = value
    
    return output


# =============================================================================
# MAIN INTERFACE FUNCTION (BACKWARD COMPATIBLE)
# =============================================================================

def evaluate_downstream_task(
    embedding_location_path: str,
    ml_model_object: Any,
    df_features: pd.DataFrame,
    exp_name: str,
    feature_set: str = 'embedding_only',
    apply_scaling: bool = False,
    downsample_ratio: Optional[float] = None,
    cat_features: Optional[List[str]] = None  # Kept for API compatibility, but now auto-detected
) -> Dict[str, Any]:
    """
    Main interface function for downstream evaluation (backward compatible).
    
    For evaluating multiple models on the same data, use prepare_evaluation_data()
    and evaluate_with_prepared_data() instead for better efficiency.
    
    Args:
        embedding_location_path: Path to NPZ file or directory containing embeddings
        ml_model_object: Pre-configured sklearn-compatible model with fit()/predict_proba()
        df_features: DataFrame with features and outcomes (from BigQuery)
        exp_name: Experiment name (e.g., 'exp1_dense_baseline')
        feature_set: One of 'embedding_only', 'tabular_only', 'hybrid'
        apply_scaling: Whether to apply StandardScaler (True for LR, False for tree-based)
        downsample_ratio: If provided, downsample training negatives to this ratio.
                          Use 10.0 to match previous model's 10:1 negative sampling.
        cat_features: Deprecated - categorical features are now auto-detected
    
    Returns:
        Dict with exp_name, model_type, feature_set, and all metrics
    """
    # Prepare data
    prepared_data = prepare_evaluation_data(
        df_features=df_features,
        embedding_location_path=embedding_location_path,
        feature_set=feature_set,
        downsample_ratio=downsample_ratio
    )
    
    # Evaluate using prepared data
    return evaluate_with_prepared_data(
        prepared_data=prepared_data,
        ml_model_object=ml_model_object,
        exp_name=exp_name,
        apply_scaling=apply_scaling
    )


# =============================================================================
# BATCH EVALUATION HELPER (OPTIMIZED - PREPARES DATA ONCE PER GROUP)
# =============================================================================

def evaluate_all_experiments(
    experiment_configs: List[Dict],
    df_features: pd.DataFrame,
    downsample_ratio: Optional[float] = None
) -> pd.DataFrame:
    """
    Evaluate multiple experiments efficiently by grouping by data requirements.
    
    Data is prepared once per unique (embedding_path, feature_set, downsample_ratio) 
    combination, then reused for all models in that group.
    
    Args:
        experiment_configs: List of dicts, each with:
            - embedding_location_path: str
            - ml_model_object: model
            - exp_name: str
            - feature_set: str (optional, default 'embedding_only')
            - apply_scaling: bool (optional, default False)
            - downsample_ratio: float (optional, overrides global downsample_ratio)
        df_features: DataFrame with features and outcomes
        downsample_ratio: Global downsample ratio for all experiments.
                          Use 10.0 to match previous model's 10:1 negative sampling.
                          Can be overridden per-experiment in config.
    
    Returns:
        DataFrame with one row per experiment, all metrics as columns
    """
    from collections import defaultdict
    
    # Group configs by (embedding_path, feature_set, downsample_ratio) to avoid redundant data preparation
    groups = defaultdict(list)
    for config in experiment_configs:
        embedding_path = config.get('embedding_location_path', '')
        feature_set = config.get('feature_set', 'embedding_only')
        # Per-experiment downsample_ratio overrides global
        ds_ratio = config.get('downsample_ratio', downsample_ratio)
        # Use tuple as key for grouping (include downsample_ratio since it affects data)
        key = (embedding_path, feature_set, ds_ratio)
        groups[key].append(config)
    
    results = []
    prepared_cache = {}  # Cache prepared data for reuse
    
    for (embedding_path, feature_set, ds_ratio), group_configs in groups.items():
        # Prepare data once for this group
        cache_key = (embedding_path, feature_set, ds_ratio)
        if cache_key not in prepared_cache:
            ds_str = f", downsample={ds_ratio}:1" if ds_ratio else ""
            print(f"Preparing data for: feature_set={feature_set}{ds_str}, path={embedding_path[:50] if embedding_path else 'N/A'}...")
            prepared_cache[cache_key] = prepare_evaluation_data(
                df_features=df_features,
                embedding_location_path=embedding_path,
                feature_set=feature_set,
                downsample_ratio=ds_ratio
            )
        
        prepared_data = prepared_cache[cache_key]
        
        # Evaluate each model in this group using prepared data
        for config in group_configs:
            model = config['ml_model_object']
            exp_name = config['exp_name']
            apply_scaling = config.get('apply_scaling', False)
            
            print(f"  Evaluating: {exp_name} ({type(model).__name__})")
            
            result = evaluate_with_prepared_data(
                prepared_data=prepared_data,
                ml_model_object=model,
                exp_name=exp_name,
                apply_scaling=apply_scaling
            )
            results.append(result)
    
    return pd.DataFrame(results)
```

---

## Usage Example

Here's how to use this modular code in your Jupyter notebook:

```python
# ============================================================================
# EXAMPLE USAGE
# ============================================================================

from google.cloud import bigquery
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

# 1. Load features separately (as per your requirement)
client = bigquery.Client()
df_features = client.query("""
    SELECT * 
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_ip_heldout_transformer_matched_final_dataset_4_te_experiment_round5_downstream`
""").to_dataframe()

print(f"Features loaded: {df_features.shape}")

# 2. Define embedding paths
EMBEDDING_BASE = "embedding_output/exp_round5_3lobs_pretrain_multi_gpu_test_v2"

# ============================================================================
# OPTION A: EFFICIENT - Prepare data once, evaluate multiple models
# ============================================================================
# This is the recommended approach when evaluating multiple models on the same data

# Step 1: Prepare data ONCE per (embedding_path, feature_set) combination
# Use downsample_ratio=10.0 to match previous model's 10:1 negative sampling
prepared_emb = prepare_evaluation_data(
    df_features=df_features,
    embedding_location_path=f"{EMBEDDING_BASE}/exp1_dense_baseline",
    feature_set='embedding_only',
    downsample_ratio=10.0  # Match previous model's 10:1 negative sampling
)

prepared_tabular = prepare_evaluation_data(
    df_features=df_features,
    embedding_location_path="",  # Not needed for tabular_only
    feature_set='tabular_only',
    downsample_ratio=10.0  # Match previous model's 10:1 negative sampling
)

print(f"Embedding data prepared: {len(prepared_emb.feature_cols)} features, downsampled={prepared_emb.downsampled}")
print(f"Tabular data prepared: {len(prepared_tabular.feature_cols)} features, downsampled={prepared_tabular.downsampled}")

# Step 2: Evaluate MULTIPLE models using prepared data (fast - no data reload)
# Logistic Regression on embeddings
result_lr = evaluate_with_prepared_data(
    prepared_data=prepared_emb,
    ml_model_object=LogisticRegression(max_iter=1000, class_weight='balanced'),
    exp_name="exp1_lr",
    apply_scaling=True  # LR needs scaling
)

# XGBoost on same embeddings (reuses prepared_emb - no re-preparation!)
from xgboost import XGBClassifier
result_xgb = evaluate_with_prepared_data(
    prepared_data=prepared_emb,
    ml_model_object=XGBClassifier(n_estimators=500, max_depth=6, random_state=42),
    exp_name="exp1_xgb",
    apply_scaling=False  # Tree models don't need scaling
)

# CatBoost on tabular features
result_cat = evaluate_with_prepared_data(
    prepared_data=prepared_tabular,
    ml_model_object=CatBoostClassifier(iterations=2500, depth=7, verbose=0),
    exp_name="tabular_catboost",
    apply_scaling=False
)

print(f"\nLR Test Lift@1%: {result_lr['test_lift_1pct']:.2f}")
print(f"XGB Test Lift@1%: {result_xgb['test_lift_1pct']:.2f}")
print(f"CatBoost Test Lift@1%: {result_cat['test_lift_1pct']:.2f}")

# ============================================================================
# OPTION B: BATCH - Auto-groups by (embedding_path, feature_set) for efficiency
# ============================================================================
# evaluate_all_experiments automatically groups configs to minimize data preparation

experiment_configs = [
    # These 3 share (exp1 path, embedding_only) - data prepared ONCE
    {
        'embedding_location_path': f"{EMBEDDING_BASE}/exp1_dense_baseline",
        'ml_model_object': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'exp_name': "exp1_lr",
        'feature_set': 'embedding_only',
        'apply_scaling': True
    },
    {
        'embedding_location_path': f"{EMBEDDING_BASE}/exp1_dense_baseline",
        'ml_model_object': XGBClassifier(n_estimators=500, max_depth=6),
        'exp_name': "exp1_xgb",
        'feature_set': 'embedding_only',
        'apply_scaling': False
    },
    # Different embedding path - new data preparation
    {
        'embedding_location_path': f"{EMBEDDING_BASE}/exp6_auxiliary_free_v3",
        'ml_model_object': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'exp_name': "exp6_lr",
        'feature_set': 'embedding_only',
        'apply_scaling': True
    },
    # Tabular-only - separate data preparation (no embeddings)
    {
        'embedding_location_path': "",
        'ml_model_object': CatBoostClassifier(iterations=2500, depth=7, verbose=0),
        'exp_name': "tabular_baseline",
        'feature_set': 'tabular_only',
        'apply_scaling': False
    },
    # Hybrid - uses exp6 embeddings + tabular
    {
        'embedding_location_path': f"{EMBEDDING_BASE}/exp6_auxiliary_free_v3",
        'ml_model_object': CatBoostClassifier(iterations=2500, depth=7, verbose=0),
        'exp_name': "exp6_hybrid",
        'feature_set': 'hybrid',
        'apply_scaling': False
    },
]

# This will prepare data 4 times (4 unique embedding_path + feature_set combos)
# then evaluate 5 models - much more efficient than preparing 5 times!
# Use downsample_ratio=10.0 to match previous model's 10:1 negative sampling
results_df = evaluate_all_experiments(experiment_configs, df_features, downsample_ratio=10.0)
print(results_df[['exp_name', 'feature_set', 'model_type', 'test_auc_roc', 'test_lift_1pct']])

# ============================================================================
# OPTION C: SIMPLE - Single evaluation (backward compatible)
# ============================================================================
# For single evaluations, the original API still works

result = evaluate_downstream_task(
    embedding_location_path=f"{EMBEDDING_BASE}/exp1_dense_baseline",
    ml_model_object=LogisticRegression(max_iter=1000, class_weight='balanced'),
    df_features=df_features,
    exp_name="exp1_dense_baseline",
    feature_set='embedding_only',
    apply_scaling=True,
    downsample_ratio=10.0  # Match previous model's 10:1 negative sampling
)
print(f"\nTest AUC: {result['test_auc_roc']:.4f}, Lift@1%: {result['test_lift_1pct']:.2f}")
```

---

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Data/Model separation** | `prepare_evaluation_data()` + `evaluate_with_prepared_data()` | Prepare once, evaluate many models - major efficiency gain |
| **PreparedData dataclass** | Container for X_splits, y_splits, metadata | Clean interface, IDE autocomplete, type hints |
| **Auto-grouping in batch** | Groups by (embedding_path, feature_set, downsample_ratio) | Automatic optimization without user effort |
| **Negative downsampling** | `downsample_ratio` parameter (e.g., 10.0 for 10:1) | Match previous model's 10:1 neg sampling strategy |
| **Backward compatible API** | `evaluate_downstream_task()` unchanged | Existing code continues to work |
| **Cat feature auto-detection** | Pre-compute indices in `prepare_evaluation_data()` | Avoid redundant detection per model |
| **Model cloning** | `clone(model)` | Prevents modifying user's model object |
| **Scaling** | Optional parameter | LR needs it, tree-based models don't |
| **Split logic** | Date filter on train/val/test, temporal OOT | Prevents temporal leakage while testing temporal generalization |

---

## Output Structure

The returned dictionary has this structure:

```python
{
    'exp_name': 'exp1_dense_baseline',
    'model_type': 'LogisticRegression',
    'feature_set': 'embedding_only',
    'n_features': 256,
    
    # Validation metrics
    'val_auc_roc': 0.7523,
    'val_auc_pr': 0.1234,
    'val_brier': 0.0456,
    'val_lift_1pct': 18.5,
    'val_lift_5pct': 8.2,
    'val_lift_10pct': 5.1,
    'val_tp_1pct': 145,
    'val_precision_1pct': 0.45,
    'val_n_samples': 12000,
    'val_n_positives': 320,
    'val_prevalence': 0.0267,
    
    # Test metrics (same structure)
    'test_auc_roc': 0.7489,
    ...
    
    # OOT metrics (same structure)
    'oot_auc_roc': 0.7312,
    ...
}
```

This structure aligns with the original modeling pipeline metrics while being easy to convert to a DataFrame for comparison.