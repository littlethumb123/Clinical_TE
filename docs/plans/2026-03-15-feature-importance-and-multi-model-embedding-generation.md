# Feature Importance (SHAP) + Multi-Model Embedding Generation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared, model-agnostic SHAP feature importance module that quantifies the proportion of embedding features in the top-N important features for both Commercial and Medicaid, and generate + upload embeddings to GCP for 3 new model checkpoints across both LOBs.

**Architecture:** A single `compute_shap_feature_importance()` function (placed once in the file, before both Commercial and Medicaid sections) that accepts any fitted sklearn-compatible model, the PreparedData/MedicaidPreparedData, and returns a summary DataFrame + embedding proportion analysis at top-10/20/50 cutoffs. New embedding generation cells for Commercial and Medicaid use the existing `generate_embeddings` → `save_embeddings_to_bigquery` / `save_medicaid_embeddings_to_bigquery` pipeline with updated MODEL_PATHS and table naming conventions consistent with existing patterns.

**Tech Stack:** Python, SHAP, scikit-learn, CatBoost, XGBoost, pandas, numpy, BigQuery

---

## Context for Implementer

### File Under Modification
`dev/downstream/moe_flashattn_3_lob3_downstream_running.py` — a 4247-line notebook-style Python script with two major sections:
- **Lines 1–2535**: Commercial IP downstream (data loading, embedding generation, model evaluation)
- **Lines 2544–4247**: Medicaid IP downstream (separate data loading, preprocessing, evaluation)

### Key Data Structures
- **Commercial**: `PreparedData` dataclass (line 1751) with `X_splits`, `y_splits`, `feature_cols`, `embedding_features`, `tabular_features`
- **Medicaid**: `MedicaidPreparedData` dataclass (line 3312) with `X_train`, `X_val`, `X_test`, `y_train`, etc.

### Key Functions Already Existing
- `evaluate_model_on_splits()` (Commercial: line 1916, Medicaid: line 3533) — trains and evaluates a model
- `generate_embeddings()` (line 516) — unified embedding generation for all LOBs
- `save_embeddings_to_bigquery()` (line 977) — saves to GCP
- `save_medicaid_embeddings_to_bigquery()` (line 3917) — thin Medicaid wrapper
- `load_model_from_checkpoint()` (line 278) — loads any checkpoint

### Table Naming Conventions
- Commercial: `a964286_te4exp_3lob_exp_round{N}_{dim}emb_{safe_exp_name}_commercial_all_sample_embedding`
- Medicare: `a964286_te4exp_3lob_exp_round{N}_v2_{safe_exp_name}_medicare_all_sample_embedding`
- Medicaid: `a964286_te4exp_{safe_exp_name}_medicaid_heldout_embedding`

### Three Model Checkpoints (User-Specified)
1. **Round 10 (256d)**: `logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool/saved_models/exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt`
2. **Round 7 (512d)**: `logs/exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim/exp2b_flash_learned_poolexp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final.pt`
3. **Round 9 (256d)**: `logs/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim/exp2b_flash_learned_pool_v2/saved_models/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim_exp2b_flash_learned_pool_bs128_ep1_d256_20260310_123547_final.pt`

---

## Task 1: Add Shared SHAP Feature Importance Module

**Files:**
- Modify: `dev/downstream/moe_flashattn_3_lob3_downstream_running.py` — insert a new cell after the existing imports/utility functions (around line 1388, before the Commercial METRIC FUNCTIONS section)

### Step 1: Add SHAP import and the shared feature importance function

Insert a new notebook cell around line 1388 (just before `# =============================================================================` / `# METRIC FUNCTIONS`). This function must work for **both** Commercial `PreparedData` and Medicaid `MedicaidPreparedData` by duck-typing on available attributes.

```python
# In[ ]:


# =============================================================================
# SHARED FEATURE IMPORTANCE MODULE (SHAP — Model-Agnostic)
# =============================================================================
# Used by both Commercial and Medicaid sections.
# Goal: quantify what proportion of top-N important features are embeddings.

import shap

def compute_shap_feature_importance(
    fitted_model,
    X_eval: pd.DataFrame,
    feature_cols: List[str],
    embedding_features: List[str],
    top_k_list: List[int] = [10, 20, 50],
    max_samples: int = 2000,
    random_state: int = 42,
    model_name: str = "",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute SHAP feature importance and analyze embedding feature proportions.

    Works with any model that has a predict_proba method (LogisticRegression,
    CatBoost, XGBoost, LightGBM, etc.).

    Args:
        fitted_model: A TRAINED model with predict_proba().
        X_eval: Evaluation DataFrame (use val or test split, NOT train).
        feature_cols: Ordered list of feature column names matching X_eval columns.
        embedding_features: List of embedding column names (subset of feature_cols).
        top_k_list: List of top-K cutoffs for proportion analysis (default [10, 20, 50]).
        max_samples: Cap on background/eval samples for SHAP speed (default 2000).
        random_state: Seed for sampling reproducibility.
        model_name: Label for output (e.g. "CatBoost_hybrid").
        verbose: Print progress.

    Returns:
        shap_summary_df: DataFrame with columns [feature, mean_abs_shap, rank, is_embedding]
                         sorted by mean_abs_shap descending.
        proportion_df:   DataFrame with columns [model_name, top_k, n_embedding_in_top_k,
                         proportion_embedding, n_tabular_in_top_k, proportion_tabular]
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"SHAP FEATURE IMPORTANCE: {model_name or type(fitted_model).__name__}")
        print(f"{'='*70}")

    X_sample = X_eval
    if len(X_eval) > max_samples:
        X_sample = X_eval.sample(n=max_samples, random_state=random_state)
        if verbose:
            print(f"  Sampled {max_samples} rows from {len(X_eval)} for SHAP computation")

    model_type = type(fitted_model).__name__

    if model_type in ('CatBoostClassifier',):
        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type in ('XGBClassifier', 'LGBMClassifier'):
        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type == 'LogisticRegression':
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.LinearExplainer(fitted_model, background)
        shap_values = explainer.shap_values(X_sample)
    else:
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.KernelExplainer(
            fitted_model.predict_proba, background
        )
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    embedding_set = set(embedding_features)
    shap_summary_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap,
        'is_embedding': [f in embedding_set for f in feature_cols],
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_summary_df['rank'] = range(1, len(shap_summary_df) + 1)

    proportion_rows = []
    for k in top_k_list:
        k_actual = min(k, len(shap_summary_df))
        top_k_df = shap_summary_df.head(k_actual)
        n_emb = int(top_k_df['is_embedding'].sum())
        n_tab = k_actual - n_emb
        proportion_rows.append({
            'model_name': model_name or model_type,
            'top_k': k,
            'n_embedding_in_top_k': n_emb,
            'proportion_embedding': round(n_emb / k_actual, 4),
            'n_tabular_in_top_k': n_tab,
            'proportion_tabular': round(n_tab / k_actual, 4),
        })

    proportion_df = pd.DataFrame(proportion_rows)

    if verbose:
        print(f"\n  Top 20 features by mean |SHAP|:")
        print(shap_summary_df[['rank', 'feature', 'mean_abs_shap', 'is_embedding']].head(20).to_string(index=False))
        print(f"\n  Embedding Proportion Analysis:")
        print(proportion_df.to_string(index=False))

    return shap_summary_df, proportion_df


def run_shap_for_all_feature_sets(
    fitted_models: Dict[str, Any],
    prepared_data_dict: Dict[str, Any],
    top_k_list: List[int] = [10, 20, 50],
    max_samples: int = 2000,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run SHAP analysis across multiple (model, feature_set) combos.

    Args:
        fitted_models: Dict mapping label -> fitted model (must already be trained).
        prepared_data_dict: Dict mapping same labels -> prepared data objects
                            (PreparedData or MedicaidPreparedData).
        top_k_list: Top-K cutoffs.
        max_samples: SHAP sample cap.
        verbose: Print progress.

    Returns:
        all_shap_df: Concatenated SHAP summaries with 'experiment' column.
        all_proportion_df: Concatenated proportion summaries.
    """
    all_shap = []
    all_proportion = []

    for label, model in fitted_models.items():
        pd_obj = prepared_data_dict[label]

        if hasattr(pd_obj, 'X_splits'):
            X_eval = pd_obj.X_splits.get('test', pd_obj.X_splits.get('val'))
        elif hasattr(pd_obj, 'X_test'):
            X_eval = pd_obj.X_test
        else:
            raise ValueError(f"Cannot find evaluation data in {type(pd_obj)}")

        feature_cols = pd_obj.feature_cols
        embedding_features = pd_obj.embedding_features

        shap_df, prop_df = compute_shap_feature_importance(
            fitted_model=model,
            X_eval=X_eval,
            feature_cols=feature_cols,
            embedding_features=embedding_features,
            top_k_list=top_k_list,
            max_samples=max_samples,
            model_name=label,
            verbose=verbose,
        )
        shap_df['experiment'] = label
        all_shap.append(shap_df)
        all_proportion.append(prop_df)

    return pd.concat(all_shap, ignore_index=True), pd.concat(all_proportion, ignore_index=True)
```

### Step 2: Verify the SHAP import doesn't break existing code

Run (in the terminal):
```bash
cd dev/downstream && python -c "import shap; print('SHAP version:', shap.__version__)"
```
Expected: Prints SHAP version. If missing, run `pip install shap`.

### Step 3: Commit

```bash
git add dev/downstream/moe_flashattn_3_lob3_downstream_running.py
git commit -m "feat: add shared model-agnostic SHAP feature importance module"
```

---

## Task 2: Add SHAP Evaluation Cells for Commercial Section

**Files:**
- Modify: `dev/downstream/moe_flashattn_3_lob3_downstream_running.py` — insert new cells after the Commercial evaluation results (around line 2344, after the excel export cells)

### Step 1: Add a cell that trains models and runs SHAP for Commercial

Insert after the commercial evaluation export cells (around line 2344):

```python
# In[ ]:


# =============================================================================
# COMMERCIAL: SHAP Feature Importance Analysis
# =============================================================================
# Demonstrates the additional value of embeddings via SHAP
# Uses the hybrid feature set to see embedding vs tabular importance

# Prepare hybrid data (reuse existing if available, or re-prepare)
embedding_path_shap = 'edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round5_v2_exp2b_flash_learned_pool_asym_focalloss_densesampler_commercial_all_sample_embedding'

prepared_hybrid_commercial = prepare_evaluation_data(
    df_features=df_ip_features,
    embedding_location_path=embedding_path_shap,
    feature_set='hybrid',
    downsample_ratio=10.0
)

# Train CatBoost on hybrid and capture the fitted model
from sklearn.base import clone as sk_clone

catboost_shap = sk_clone(catboost_model)
cat_indices = prepared_hybrid_commercial.cat_feature_indices if prepared_hybrid_commercial.cat_feature_indices else []
from catboost import Pool
train_pool_shap = Pool(
    prepared_hybrid_commercial.X_splits['train'],
    prepared_hybrid_commercial.y_splits['train'],
    cat_features=cat_indices,
)
val_pool_shap = Pool(
    prepared_hybrid_commercial.X_splits['val'],
    prepared_hybrid_commercial.y_splits['val'],
    cat_features=cat_indices,
)
catboost_shap.fit(train_pool_shap, eval_set=val_pool_shap, verbose=0)
print("Commercial CatBoost (hybrid) trained for SHAP analysis")


# In[ ]:


# Run SHAP
commercial_shap_df, commercial_proportion_df = compute_shap_feature_importance(
    fitted_model=catboost_shap,
    X_eval=prepared_hybrid_commercial.X_splits['test'],
    feature_cols=prepared_hybrid_commercial.feature_cols,
    embedding_features=prepared_hybrid_commercial.embedding_features,
    top_k_list=[10, 20, 50],
    max_samples=2000,
    model_name="commercial_catboost_hybrid",
    verbose=True,
)

# Save results
commercial_shap_df.to_excel("experiment_logs/commercial_shap_feature_importance.xlsx", index=False)
commercial_proportion_df.to_excel("experiment_logs/commercial_shap_embedding_proportions.xlsx", index=False)
print("\nSHAP results saved to experiment_logs/")
```

### Step 2: Commit

```bash
git add dev/downstream/moe_flashattn_3_lob3_downstream_running.py
git commit -m "feat: add SHAP evaluation cells for commercial downstream"
```

---

## Task 3: Add SHAP Evaluation Cells for Medicaid Section

**Files:**
- Modify: `dev/downstream/moe_flashattn_3_lob3_downstream_running.py` — insert new cells after the Medicaid evaluation summary (around line 4247, at the end of the file)

### Step 1: Add cells for Medicaid SHAP analysis

Append at the end of the file:

```python
# In[ ]:


# =============================================================================
# MEDICAID: SHAP Feature Importance Analysis
# =============================================================================
# Uses the hybrid feature set to quantify embedding vs tabular importance.
# Reuses the last merged df_merged and prepared_data from the Medicaid evaluation loop above.

# Re-prepare hybrid data for SHAP (or reuse if last iteration was hybrid)
prepared_medicaid_hybrid = prepare_medicaid_evaluation_data(
    df=df_merged,
    feature_set='hybrid',
    apply_downsampling=True,
    downsample_ratio=CATBOOST_UNDERSAMPLE_RATIO,
    split_random_state=RANDOM_STATE,
    undersample_random_state=UNDERSAMPLE_RANDOM_STATE,
    verbose=True,
)

# Train CatBoost for SHAP
catboost_medicaid_shap = CatBoostClassifier(**CATBOOST_TUNED_PARAMS)
train_pool_md_shap = Pool(
    prepared_medicaid_hybrid.X_train,
    prepared_medicaid_hybrid.y_train,
    cat_features=prepared_medicaid_hybrid.cat_feature_indices if prepared_medicaid_hybrid.cat_feature_indices else None,
)
val_pool_md_shap = Pool(
    prepared_medicaid_hybrid.X_val,
    prepared_medicaid_hybrid.y_val,
    cat_features=prepared_medicaid_hybrid.cat_feature_indices if prepared_medicaid_hybrid.cat_feature_indices else None,
)
catboost_medicaid_shap.fit(train_pool_md_shap, eval_set=val_pool_md_shap, verbose=0)
print("Medicaid CatBoost (hybrid) trained for SHAP analysis")


# In[ ]:


# Run SHAP
medicaid_shap_df, medicaid_proportion_df = compute_shap_feature_importance(
    fitted_model=catboost_medicaid_shap,
    X_eval=prepared_medicaid_hybrid.X_test,
    feature_cols=prepared_medicaid_hybrid.feature_cols,
    embedding_features=prepared_medicaid_hybrid.embedding_features,
    top_k_list=[10, 20, 50],
    max_samples=2000,
    model_name="medicaid_catboost_hybrid",
    verbose=True,
)

# Save results
medicaid_shap_df.to_excel("experiment_logs/medicaid_shap_feature_importance.xlsx", index=False)
medicaid_proportion_df.to_excel("experiment_logs/medicaid_shap_embedding_proportions.xlsx", index=False)
print("\nMedicaid SHAP results saved to experiment_logs/")
```

### Step 2: Commit

```bash
git add dev/downstream/moe_flashattn_3_lob3_downstream_running.py
git commit -m "feat: add SHAP evaluation cells for medicaid downstream"
```

---

## Task 4: Add Commercial Embedding Generation for 3 New Models

**Files:**
- Modify: `dev/downstream/moe_flashattn_3_lob3_downstream_running.py` — insert new cells after the existing commercial embedding generation loop (around line 1274, after the current `results` loop)

### Step 1: Add new MODEL_PATHS and embedding generation loop for Commercial

Insert a new cell after line ~1277 (`embeddings.shape`):

```python
# In[ ]:


# =============================================================================
# COMMERCIAL: Embedding Generation for Round 10, 9, and 7 Models
# =============================================================================
# Three new checkpoints to evaluate

MODEL_PATHS_NEW = {
    'exp_round10_formal_exp2b_flash_learned_pool_d256':
        'logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool/saved_models/'
        'exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt',

    'exp_round7_exp2b_flash_learned_pool_d512':
        'logs/exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim/'
        'exp2b_flash_learned_poolexp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final/'
        'saved_models/'
        'exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final.pt',

    'exp_round9_decoupled_exp2b_flash_learned_pool_v2_d256':
        'logs/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim/exp2b_flash_learned_pool_v2/saved_models/'
        'exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim_exp2b_flash_learned_pool_bs128_ep1_d256_20260310_123547_final.pt',
}

results_new = {}
batch_size = 64
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"
LOB = 'commercial'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for exp_name, model_path in tqdm(MODEL_PATHS_NEW.items(), desc="Commercial embedding generation"):
    cleanup_gpu_memory(verbose=False)
    model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
        model_path=model_path,
        device=device,
        verbose=True
    )

    inference_start_time = time.time()
    embeddings, individual_ids, index_dts = generate_embeddings(
        model=model,
        config=config,
        data=df_cm_sample,
        device=device,
        id_column='individual_id',
        lob_value=None,
        desc_prefix='Commercial',
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        verbose=True,
        multi_gpu=True,
        moe_config=moe_config,
    )
    inference_duration = time.time() - inference_start_time
    print(f"Inference duration for {exp_name}: {round(inference_duration/3600, 2):.2f} hr")

    safe_exp_name = exp_name.replace('-', '_').replace('.', '_')
    table_name = f"a964286_te4exp_3lob_{safe_exp_name}_{LOB}_all_sample_embedding"
    bq_table_path = save_embeddings_to_bigquery(
        embeddings=embeddings,
        individual_ids=individual_ids,
        index_dts=index_dts,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_name=table_name,
        exp_name=exp_name,
        model_type=model_type,
        if_exists="replace"
    )
    results_new[exp_name] = {
        'bq_table_path': bq_table_path,
        'embedding_shape': embeddings.shape,
        'model_type': model_type,
        'model_path': model_path,
        'inference_duration_hr': round(inference_duration / 3600, 2),
        'status': 'success'
    }

    del model
    del embeddings
    torch.cuda.empty_cache()

print("\n=== Commercial Embedding Generation Summary ===")
for exp_name, result in results_new.items():
    print(f"  {exp_name}: {result['embedding_shape']} -> {result['bq_table_path']}")
```

### Step 2: Commit

```bash
git add dev/downstream/moe_flashattn_3_lob3_downstream_running.py
git commit -m "feat: add commercial embedding generation for round 10/9/7 models"
```

---

## Task 5: Add Medicare Embedding Generation for 3 New Models

**Files:**
- Modify: `dev/downstream/moe_flashattn_3_lob3_downstream_running.py` — insert new cells in the Medicare section (around line 2535, after the existing Medicare embedding generation loop)

### Step 1: Add new MODEL_PATHS and embedding generation loop for Medicare

Insert after line ~2535 (after the existing Medicare embedding loop):

```python
# In[ ]:


# =============================================================================
# MEDICARE: Embedding Generation for Round 10, 9, and 7 Models
# =============================================================================

MODEL_PATHS_NEW_MEDICARE = {
    'exp_round10_formal_exp2b_flash_learned_pool_d256':
        'logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool/saved_models/'
        'exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt',

    'exp_round7_exp2b_flash_learned_pool_d512':
        'logs/exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim/'
        'exp2b_flash_learned_poolexp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final/'
        'saved_models/'
        'exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final.pt',

    'exp_round9_decoupled_exp2b_flash_learned_pool_v2_d256':
        'logs/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim/exp2b_flash_learned_pool_v2/saved_models/'
        'exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim_exp2b_flash_learned_pool_bs128_ep1_d256_20260310_123547_final.pt',
}

results_new_medicare = {}
batch_size = 64
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"
LOB = 'medicare'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for exp_name, model_path in tqdm(MODEL_PATHS_NEW_MEDICARE.items(), desc="Medicare embedding generation"):
    cleanup_gpu_memory(verbose=False)
    model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
        model_path=model_path,
        device=device,
        verbose=True
    )

    inference_start_time = time.time()
    embeddings, individual_ids, index_dts = generate_embeddings(
        model=model,
        config=config,
        data=df_me_sample,
        device=device,
        id_column='individual_id',
        lob_value=None,
        desc_prefix='Medicare',
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        verbose=True,
        multi_gpu=True,
        moe_config=moe_config,
    )
    inference_duration = time.time() - inference_start_time
    print(f"Inference duration for {exp_name}: {round(inference_duration/3600, 2):.2f} hr")

    safe_exp_name = exp_name.replace('-', '_').replace('.', '_')
    table_name = f"a964286_te4exp_3lob_{safe_exp_name}_{LOB}_all_sample_embedding"
    bq_table_path = save_embeddings_to_bigquery(
        embeddings=embeddings,
        individual_ids=individual_ids,
        index_dts=index_dts,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_name=table_name,
        exp_name=exp_name,
        model_type=model_type,
        if_exists="replace"
    )
    results_new_medicare[exp_name] = {
        'bq_table_path': bq_table_path,
        'embedding_shape': embeddings.shape,
        'model_type': model_type,
        'model_path': model_path,
        'inference_duration_hr': round(inference_duration / 3600, 2),
        'status': 'success'
    }

    del model
    del embeddings
    torch.cuda.empty_cache()

print("\n=== Medicare Embedding Generation Summary ===")
for exp_name, result in results_new_medicare.items():
    print(f"  {exp_name}: {result['embedding_shape']} -> {result['bq_table_path']}")
```

### Step 2: Commit

```bash
git add dev/downstream/moe_flashattn_3_lob3_downstream_running.py
git commit -m "feat: add medicare embedding generation for round 10/9/7 models"
```

---

## Task 6: Add Medicaid Embedding Generation for 3 New Models

**Files:**
- Modify: `dev/downstream/moe_flashattn_3_lob3_downstream_running.py` — insert new cells in the Medicaid embedding generation section (around line 4036, after the existing Medicaid embedding generation loop)

### Step 1: Add new MODEL_PATHS and embedding generation loop for Medicaid

Insert after line ~4036 (after the existing Medicaid embedding generation loop):

```python
# In[ ]:


# =============================================================================
# MEDICAID: Embedding Generation for Round 10, 9, and 7 Models
# =============================================================================

MODEL_PATHS_NEW_MEDICAID = {
    'exp_round10_formal_exp2b_flash_learned_pool_d256':
        'logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool/saved_models/'
        'exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt',

    'exp_round7_exp2b_flash_learned_pool_d512':
        'logs/exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim/'
        'exp2b_flash_learned_poolexp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final/'
        'saved_models/'
        'exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final.pt',

    'exp_round9_decoupled_exp2b_flash_learned_pool_v2_d256':
        'logs/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim/exp2b_flash_learned_pool_v2/saved_models/'
        'exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim_exp2b_flash_learned_pool_bs128_ep1_d256_20260310_123547_final.pt',
}

results_new_medicaid = {}
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for exp_name, model_path in tqdm(MODEL_PATHS_NEW_MEDICAID.items(), desc="Medicaid embedding generation"):
    cleanup_gpu_memory(verbose=False)
    model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
        model_path=model_path,
        device=device,
        verbose=True
    )

    inference_start_time = time.time()
    embeddings, member_keys, index_dts = generate_embeddings(
        model=model,
        config=config,
        data=df_te_input,
        device=device,
        id_column='asdb_member_key',
        lob_value='Medicaid',
        desc_prefix='Medicaid',
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        verbose=True,
        multi_gpu=True,
        moe_config=moe_config,
    )
    inference_duration = time.time() - inference_start_time
    print(f"Inference duration for {exp_name}: {round(inference_duration/3600, 2):.2f} hr")

    safe_exp_name = exp_name.replace('-', '_').replace('.', '_')
    table_name = f"a964286_te4exp_{safe_exp_name}_medicaid_heldout_embedding"
    bq_table_path = save_medicaid_embeddings_to_bigquery(
        embeddings=embeddings,
        member_keys=member_keys,
        index_dts=index_dts,
        table_name=table_name,
        exp_name=exp_name,
        model_type=model_type,
    )
    results_new_medicaid[exp_name] = {
        'bq_table_path': bq_table_path,
        'embedding_shape': embeddings.shape,
        'model_type': model_type,
        'model_path': model_path,
        'inference_duration_hr': round(inference_duration / 3600, 2),
        'status': 'success'
    }

    del model
    del embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n=== Medicaid Embedding Generation Summary ===")
for exp_name, result in results_new_medicaid.items():
    print(f"  {exp_name}: {result['embedding_shape']} -> {result['bq_table_path']}")
```

### Step 2: Commit

```bash
git add dev/downstream/moe_flashattn_3_lob3_downstream_running.py
git commit -m "feat: add medicaid embedding generation for round 10/9/7 models"
```

---

## Task 7: Final Verification and Cleanup Commit

### Step 1: Run a quick syntax check on the full file

```bash
cd dev/downstream && python -c "
import py_compile
py_compile.compile('moe_flashattn_3_lob3_downstream_running.py', doraise=True)
print('Syntax OK')
"
```
Expected: `Syntax OK`

### Step 2: Check linter for introduced errors

Run the ReadLints tool on `dev/downstream/moe_flashattn_3_lob3_downstream_running.py`.
Fix any issues introduced by the changes.

### Step 3: Final commit (if any linter fixes)

```bash
git add dev/downstream/moe_flashattn_3_lob3_downstream_running.py
git commit -m "fix: address linter warnings in downstream pipeline"
```

---

## Summary of GCP Table Names Created

### Commercial (3 tables)
| Experiment | Table Name |
|---|---|
| Round 10 (256d) | `a964286_te4exp_3lob_exp_round10_formal_exp2b_flash_learned_pool_d256_commercial_all_sample_embedding` |
| Round 7 (512d) | `a964286_te4exp_3lob_exp_round7_exp2b_flash_learned_pool_d512_commercial_all_sample_embedding` |
| Round 9 (256d) | `a964286_te4exp_3lob_exp_round9_decoupled_exp2b_flash_learned_pool_v2_d256_commercial_all_sample_embedding` |

### Medicare (3 tables)
| Experiment | Table Name |
|---|---|
| Round 10 (256d) | `a964286_te4exp_3lob_exp_round10_formal_exp2b_flash_learned_pool_d256_medicare_all_sample_embedding` |
| Round 7 (512d) | `a964286_te4exp_3lob_exp_round7_exp2b_flash_learned_pool_d512_medicare_all_sample_embedding` |
| Round 9 (256d) | `a964286_te4exp_3lob_exp_round9_decoupled_exp2b_flash_learned_pool_v2_d256_medicare_all_sample_embedding` |

### Medicaid (3 tables)
| Experiment | Table Name |
|---|---|
| Round 10 (256d) | `a964286_te4exp_exp_round10_formal_exp2b_flash_learned_pool_d256_medicaid_heldout_embedding` |
| Round 7 (512d) | `a964286_te4exp_exp_round7_exp2b_flash_learned_pool_d512_medicaid_heldout_embedding` |
| Round 9 (256d) | `a964286_te4exp_exp_round9_decoupled_exp2b_flash_learned_pool_v2_d256_medicaid_heldout_embedding` |

---

## Key Design Decisions

1. **Single SHAP module, no code duplication**: `compute_shap_feature_importance()` is placed once at ~line 1388, before both Commercial and Medicaid sections. It accepts any sklearn-compatible fitted model and duck-types on `PreparedData` vs `MedicaidPreparedData`.

2. **Model-agnostic via SHAP backend selection**: Uses `TreeExplainer` for tree-based models (CatBoost, XGBoost, LightGBM) for speed, `LinearExplainer` for LogReg, and falls back to `KernelExplainer` for unknown models.

3. **Consistent table naming**: Follows existing conventions — Commercial/Medicare use `a964286_te4exp_3lob_{safe_exp_name}_{lob}_all_sample_embedding`, Medicaid uses `a964286_te4exp_{safe_exp_name}_medicaid_heldout_embedding`.

4. **Medicare uses `individual_id` + `lob_value=None`**: Matches existing Medicare section (line 2484) where data already has lob column.

5. **Medicaid uses `asdb_member_key` + `lob_value='Medicaid'`**: Matches existing Medicaid section (line 3997-3998).

6. **Round 7 model path note**: The user-provided path has `exp2b_flash_learned_poolexp_round7...` (no separator between `pool` and `exp_round7`). This appears intentional based on the directory naming and is preserved exactly as provided.
