# Raw Code vs Transformer Embedding Downstream Comparison

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Quantify the information value of Transformer Embeddings (TE) as dimension reduction by comparing downstream IP prediction performance across raw codes, PCA, T-SNE, feature-selected codes, and TE — with SHAP-based frequency-tier importance analysis.

**Architecture:** A Jupyter notebook (`dev/downstream/raw_code_vs_te_comparison.ipynb`) handles data extraction from BigQuery, construction of the raw-code frequency feature table, dimension reduction (PCA, T-SNE, feature selection), SHAP frequency-tier analysis, and result visualization. The existing downstream pipeline (`moe_flashattn_3_lob3_downstream_running.py`) is reused for CatBoost model training/evaluation via function imports. The notebook produces feature tables (DataFrames/NPZ files) that plug into the existing `prepare_evaluation_data()` → `evaluate_with_prepared_data()` pipeline.

**Tech Stack:** Python, Jupyter, BigQuery (google-cloud-bigquery), pandas, numpy, scipy (sparse), scikit-learn (PCA, T-SNE, SelectKBest/mutual_info_classif), CatBoost, SHAP, matplotlib/seaborn

---

## Data Flow Architecture

```
BigQuery Table (a834793_Combined_All_LOB_o3_train_ending)
  ↓ [Query: Commercial members only, JOIN with downstream features table]
  ↓
Raw cd sequences (STRING: "123,456*789,101*...")
  ↓ [Parse: flatten all codes across all days per member, count frequencies]
  ↓
Sparse Code-Frequency Matrix (~84k columns, one per unique code)
  ↓ [Save as intermediate artifact]
  ↓
  ├─→ [1] Raw Codes (full ~84k features) → CatBoost → metrics
  ├─→ [2a] PCA(256) on raw codes → CatBoost → metrics  
  ├─→ [2b] T-SNE(256) on PCA-reduced codes → CatBoost → metrics
  ├─→ [2c] SelectKBest(256) on raw codes → CatBoost → metrics
  └─→ [3] TE embeddings (256d, from existing pipeline) → CatBoost → metrics
  
All 5 feature sets → Same CatBoost config, same splits, same metrics
  ↓
Comparison Table + SHAP Frequency-Tier Analysis
```

## Key Design Decisions

1. **Commercial LOB only** — matches existing downstream evaluation scope (IP prediction on `a964286_commercial_ip_heldout_transformer_matched_final_dataset_4_te_experiment_round5_downstream`)
2. **Member-level code frequency** — for each member, flatten all days in `cd`, count occurrences of each code → sparse vector of ~84k dimensions
3. **Same data splits** — use `create_data_splits()` with same OOT cutoff (2023-10-16) and `ind_id_last_digit` logic
4. **Same CatBoost hyperparams** — reuse `catboost_model` config from downstream script (iterations=2500, depth=7, lr=0.025, balanced weights)
5. **Same metrics** — `compute_split_metrics()` (AUC-ROC, AUC-PR, Brier, lift@1/5/10%, TP@1%, precision@1%)
6. **256 dimensions for all reduced representations** — matches TE embedding dimension for fair comparison
7. **T-SNE(256)** — T-SNE doesn't scale to 256d natively; we'll use PCA(500) → T-SNE(3) for visualization but PCA(256) as the primary linear reduction; replace T-SNE with UMAP(256) as a nonlinear alternative that handles higher dimensions
8. **Sparse storage** — raw code matrix is ~84k×N; use scipy.sparse CSR format to avoid memory issues
9. **SHAP on raw-code model** — TreeSHAP on CatBoost is fast; group SHAP values by frequency tier for aggregate importance

## Frequency Tier Definitions

**Exactly matches `compute_stratified_metrics()` in `moe_flashattn_3.py` (line 7796).**

Tiers are defined by **percentiles of non-zero code occurrence counts** (total occurrences across all training samples), NOT by fixed member-prevalence thresholds:

```python
freq_percentiles = np.percentile(code_frequencies[code_frequencies > 0], [20, 50, 80])
```

| Tier | Definition | Approx % of Codes |
|------|-----------|-------------------|
| Common | freq > 80th percentile | ~20% |
| Medium | 50th-80th percentile | ~30% |
| Rare | 20th-50th percentile | ~30% |
| Tail | 0th-20th percentile (freq > 0) | ~20% |

Exact percentile thresholds are data-driven and reported at runtime. The same 4 tier names (`common`, `medium`, `rare`, `tail`) are used throughout to match training evaluation metrics (`common_top10_acc`, `medium_top10_acc`, `rare_top10_acc`, `tail_top10_acc`).

---

### Task 1: Notebook Scaffolding and BigQuery Data Loading

**Files:**
- Create: `dev/downstream/raw_code_vs_te_comparison.ipynb`

**Step 1: Create notebook with imports and constants**

```python
# Cell 1: Imports
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from google.cloud import bigquery
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import umap
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Reuse downstream pipeline functions
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
from moe_flashattn_3_lob3_downstream_running import (
    compute_split_metrics,
    create_data_splits,
    identify_feature_columns,
    downsample_negatives,
    prepare_features,
    OOT_CUTOFF_DATE,
    TARGET_COLUMN,
    EXCLUDE_COLUMNS,
)
from catboost import CatBoostClassifier, Pool
```

```python
# Cell 2: Constants
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"

# Source tables
RAW_TE_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_Combined_All_LOB_o3_train_ending"
FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_commercial_ip_heldout_transformer_matched_final_dataset_4_te_experiment_round5_downstream"
W2IND_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_member_w2ind"

# TE embedding table (best R6 model for comparison)
TE_EMBEDDING_TABLE = "edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round5_v2_exp2b_flash_learned_pool_asym_focalloss_densesampler_commercial_all_sample_embedding"

TARGET_DIM = 256  # Match TE embedding dimension
RANDOM_STATE = 42
NEGATIVE_DOWNSAMPLE_RATIO = 10

# CatBoost config matching downstream pipeline
CATBOOST_PARAMS = {
    'iterations': 2500,
    'depth': 7,
    'learning_rate': 0.025,
    'grow_policy': 'SymmetricTree',
    'auto_class_weights': 'Balanced',
    'od_wait': 80,
    'use_best_model': True,
    'random_seed': RANDOM_STATE,
    'verbose': 0,
}

# Output paths
OUTPUT_DIR = "experiment_logs/raw_code_vs_te_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

**Step 2: Load downstream features table (reuse existing)**

```python
# Cell 3: Load downstream features table from BigQuery
client = bigquery.Client()
print("Loading downstream features table...")
features_sql = f"SELECT * FROM `{FEATURES_TABLE}`"
df_features = client.query(features_sql).to_dataframe()
print(f"Features loaded: {len(df_features):,} rows, {len(df_features.columns)} columns")
print(f"Target prevalence: {df_features[TARGET_COLUMN].mean()*100:.2f}%")
```

**Step 3: Load raw cd sequences for matched Commercial members**

```python
# Cell 4: Load raw code sequences for members in the downstream features table
# Only load Commercial members that exist in the features table
member_ids = df_features['individual_id'].unique().tolist()
print(f"Downstream members to match: {len(member_ids):,}")

# Query raw sequences — filter to Commercial LOB and members in downstream table
raw_seq_sql = f"""
SELECT 
    individual_id,
    cd,
    dt_cnt
FROM `{RAW_TE_TABLE}`
WHERE lob = 'Commercial'
"""
print("Loading raw code sequences from BigQuery...")
df_raw = client.query(raw_seq_sql).to_dataframe()
print(f"Raw sequences loaded: {len(df_raw):,} Commercial members")

# Filter to members present in downstream features table
df_raw['individual_id'] = df_raw['individual_id'].astype(str)
df_features['individual_id'] = df_features['individual_id'].astype(str)
df_raw_matched = df_raw[df_raw['individual_id'].isin(set(df_features['individual_id']))]
print(f"Matched members: {len(df_raw_matched):,} / {len(member_ids):,}")
```

**Step 4: Run notebook cells 1-4 to verify data loads**

Run: Execute cells 1-4 in Jupyter
Expected: Feature table loads (~7M rows), raw sequences load and match to downstream members

**Step 5: Commit**

```bash
git add dev/downstream/raw_code_vs_te_comparison.ipynb docs/plans/2026-03-13-raw-code-vs-te-downstream-comparison.md
git commit -m "feat: scaffold raw code vs TE comparison notebook with BigQuery data loading"
```

---

### Task 2: Parse Raw Codes into Sparse Frequency Matrix

**Files:**
- Modify: `dev/downstream/raw_code_vs_te_comparison.ipynb` (add cells)

**Step 1: Build code-frequency vector per member**

```python
# Cell 5: Parse cd sequences into per-member code frequency counts
def parse_cd_to_code_frequencies(cd_string: str) -> Counter:
    """
    Parse a cd string like "123,456*789,101*..." into a Counter of code frequencies.
    * separates days, , separates codes within a day.
    Returns Counter mapping code_index -> total_count_across_all_days.
    """
    if not cd_string or pd.isna(cd_string):
        return Counter()
    
    freq = Counter()
    for day_str in cd_string.split('*'):
        if not day_str:
            continue
        for code_str in day_str.split(','):
            try:
                code = int(code_str)
                if code > 0:  # Skip padding (0)
                    freq[code] += 1
            except ValueError:
                continue
    return freq

# Parse all members
print("Parsing code sequences into frequency vectors...")
member_freqs = []
member_ids_ordered = []

for _, row in tqdm(df_raw_matched.iterrows(), total=len(df_raw_matched)):
    freq = parse_cd_to_code_frequencies(row['cd'])
    member_freqs.append(freq)
    member_ids_ordered.append(row['individual_id'])

print(f"Parsed {len(member_freqs):,} members")
```

**Step 2: Build sparse matrix from frequency counters**

```python
# Cell 6: Build sparse code-frequency matrix
# Determine vocabulary size (max code index)
all_codes = set()
for freq in member_freqs:
    all_codes.update(freq.keys())
max_code = max(all_codes)
vocab_size = max_code + 1
print(f"Unique codes observed: {len(all_codes):,}")
print(f"Vocabulary size (max index + 1): {vocab_size:,}")

# Build sparse CSR matrix: rows = members, cols = code indices, values = frequency
from scipy.sparse import lil_matrix

print("Building sparse frequency matrix...")
code_matrix = lil_matrix((len(member_freqs), vocab_size), dtype=np.float32)

for i, freq in tqdm(enumerate(member_freqs), total=len(member_freqs)):
    for code, count in freq.items():
        code_matrix[i, code] = count

code_matrix_csr = code_matrix.tocsr()
print(f"Sparse matrix shape: {code_matrix_csr.shape}")
print(f"Non-zero elements: {code_matrix_csr.nnz:,}")
print(f"Sparsity: {1 - code_matrix_csr.nnz / (code_matrix_csr.shape[0] * code_matrix_csr.shape[1]):.6f}")
print(f"Memory (CSR): {(code_matrix_csr.data.nbytes + code_matrix_csr.indices.nbytes + code_matrix_csr.indptr.nbytes) / 1e6:.1f} MB")

# Save for reuse
sparse.save_npz(f"{OUTPUT_DIR}/code_frequency_matrix.npz", code_matrix_csr)
np.save(f"{OUTPUT_DIR}/member_ids_ordered.npy", np.array(member_ids_ordered))
print(f"Saved to {OUTPUT_DIR}/")
```

**Step 3: Compute code frequency distribution and define tiers**

```python
# Cell 7: Analyze code frequency distribution and define frequency tiers
# Code frequency = number of members who have this code at least once
code_member_counts = np.array((code_matrix_csr > 0).sum(axis=0)).flatten()
n_members = code_matrix_csr.shape[0]

# Only consider codes that appear at least once
active_codes = np.where(code_member_counts > 0)[0]
active_counts = code_member_counts[active_codes]
active_fracs = active_counts / n_members

print(f"Total active codes: {len(active_codes):,} / {vocab_size:,}")
print(f"\nCode frequency distribution (member prevalence):")
for pct in [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]:
    n_above = (active_fracs >= pct).sum()
    print(f"  ≥{pct*100:.1f}% of members: {n_above:,} codes ({n_above/len(active_codes)*100:.1f}%)")

# Define frequency tiers based on member prevalence
# Tier boundaries (fraction of members)
TIER_BOUNDARIES = {
    'frequent': 0.10,   # ≥10% of members
    'medium': 0.01,     # ≥1% and <10%
    'rare': 0.005,      # ≥0.5% and <1%
    'tail': 0.0,        # >0 and <0.5%
}

# Assign tiers
code_tiers = {}
for code_idx in active_codes:
    frac = code_member_counts[code_idx] / n_members
    if frac >= TIER_BOUNDARIES['frequent']:
        code_tiers[code_idx] = 'frequent'
    elif frac >= TIER_BOUNDARIES['medium']:
        code_tiers[code_idx] = 'medium'
    elif frac >= TIER_BOUNDARIES['rare']:
        code_tiers[code_idx] = 'rare'
    else:
        code_tiers[code_idx] = 'tail'

tier_counts = Counter(code_tiers.values())
print(f"\nFrequency tier assignment:")
for tier in ['frequent', 'medium', 'rare', 'tail']:
    n = tier_counts.get(tier, 0)
    print(f"  {tier}: {n:,} codes ({n/len(active_codes)*100:.1f}%)")

# Save tier assignments
import json
tier_data = {
    'tier_boundaries': TIER_BOUNDARIES,
    'tier_counts': dict(tier_counts),
    'code_tiers': {str(k): v for k, v in code_tiers.items()},
    'n_members': n_members,
    'n_active_codes': len(active_codes),
}
with open(f"{OUTPUT_DIR}/code_frequency_tiers.json", 'w') as f:
    json.dump(tier_data, f, indent=2)
print(f"Saved tier assignments to {OUTPUT_DIR}/code_frequency_tiers.json")
```

**Step 4: Run cells 5-7 to verify parsing and distribution**

Run: Execute cells 5-7
Expected: Sparse matrix built, tier distribution printed, files saved

**Step 5: Commit**

```bash
git add dev/downstream/raw_code_vs_te_comparison.ipynb
git commit -m "feat: parse raw codes into sparse frequency matrix with tier classification"
```

---

### Task 3: Build Feature Tables for All 5 Representation Types

**Files:**
- Modify: `dev/downstream/raw_code_vs_te_comparison.ipynb` (add cells)

**Step 1: Build raw-code feature table joined with downstream features**

```python
# Cell 8: Build feature DataFrames for each representation type
# Join code frequency matrix with downstream features table by individual_id

# Create DataFrame from sparse matrix with code column names
print("Creating raw code feature DataFrame...")
code_col_names = [f'code_{i}' for i in range(vocab_size)]

# We'll work with member_ids_ordered which aligns with code_matrix_csr rows
df_code_ids = pd.DataFrame({'individual_id': member_ids_ordered})

# Merge with features table to get target variable and split columns
df_base = df_features[['individual_id', 'index_dt', TARGET_COLUMN, 'ind_id_last_digit']].copy()
df_base['individual_id'] = df_base['individual_id'].astype(str)
df_code_ids['individual_id'] = df_code_ids['individual_id'].astype(str)

# Add row index to track which row in code_matrix each member maps to
df_code_ids['_sparse_row_idx'] = range(len(df_code_ids))

df_merged_base = df_base.merge(df_code_ids, on='individual_id', how='inner')
df_merged_base = df_merged_base.drop_duplicates(subset=['individual_id', 'index_dt'], keep='last')
print(f"Merged base: {len(df_merged_base):,} rows")

# Extract the corresponding rows from sparse matrix
sparse_row_indices = df_merged_base['_sparse_row_idx'].values
code_matrix_matched = code_matrix_csr[sparse_row_indices]
print(f"Matched code matrix: {code_matrix_matched.shape}")
```

**Step 2: Apply PCA(256) dimension reduction**

```python
# Cell 9: PCA reduction to 256 dimensions
print("Applying PCA(256) to raw code matrix...")
# StandardScaler on sparse matrix (center=False for sparse compatibility)
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
code_matrix_scaled = scaler.fit_transform(code_matrix_matched)

pca = PCA(n_components=TARGET_DIM, random_state=RANDOM_STATE)
# PCA requires dense input — use sparse-to-dense batch approach if memory allows
# For large datasets, use TruncatedSVD instead (works on sparse)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=TARGET_DIM, random_state=RANDOM_STATE)
pca_features = svd.fit_transform(code_matrix_scaled)
print(f"PCA features shape: {pca_features.shape}")
print(f"Explained variance ratio (cumulative): {svd.explained_variance_ratio_.sum():.4f}")

np.save(f"{OUTPUT_DIR}/pca_256_features.npy", pca_features)
```

**Step 3: Apply UMAP(256) nonlinear reduction**

```python
# Cell 10: UMAP reduction to 256 dimensions (nonlinear alternative to T-SNE)
# T-SNE doesn't work well beyond 3d; UMAP handles higher dimensions
print("Applying UMAP(256) to raw code matrix...")
print("(This may take 10-30 minutes for large datasets)")

reducer = umap.UMAP(
    n_components=TARGET_DIM,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=RANDOM_STATE,
    verbose=True
)
# UMAP can handle sparse input directly
umap_features = reducer.fit_transform(code_matrix_matched)
print(f"UMAP features shape: {umap_features.shape}")

np.save(f"{OUTPUT_DIR}/umap_256_features.npy", umap_features)
```

**Step 4: Apply feature selection (SelectKBest with mutual information)**

```python
# Cell 11: Feature selection — top 256 codes by mutual information with target
print("Applying SelectKBest(mutual_info, k=256) to raw code matrix...")

# Need target variable aligned with code matrix rows
y_for_selection = df_merged_base[TARGET_COLUMN].values

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=TARGET_DIM
)
selected_features = selector.fit_transform(code_matrix_matched.toarray(), y_for_selection)
print(f"Selected features shape: {selected_features.shape}")

# Record which codes were selected
selected_mask = selector.get_support()
selected_code_indices = np.where(selected_mask)[0]
selected_scores = selector.scores_[selected_mask]
print(f"Selected {len(selected_code_indices)} codes")
print(f"MI score range: [{selected_scores.min():.6f}, {selected_scores.max():.6f}]")

# Save selected code indices and their tiers
selected_tiers = [code_tiers.get(idx, 'unknown') for idx in selected_code_indices]
tier_breakdown = Counter(selected_tiers)
print(f"Selected codes by tier: {dict(tier_breakdown)}")

np.save(f"{OUTPUT_DIR}/selected_256_features.npy", selected_features)
np.save(f"{OUTPUT_DIR}/selected_code_indices.npy", selected_code_indices)
```

**Step 5: Load TE embeddings for the same members**

```python
# Cell 12: Load TE embeddings from BigQuery
print("Loading TE embeddings from BigQuery...")
te_emb_sql = f"SELECT * FROM `{TE_EMBEDDING_TABLE}`"
df_te_emb = client.query(te_emb_sql).to_dataframe()
print(f"TE embeddings loaded: {len(df_te_emb):,} rows")

# Extract embedding columns
te_emb_cols = sorted([c for c in df_te_emb.columns if c.startswith('embedding_')])
print(f"TE embedding dimensions: {len(te_emb_cols)}")

# Merge with our base DataFrame to align members
df_te_emb['individual_id'] = df_te_emb['individual_id'].astype(str)
df_te_merged = df_merged_base[['individual_id', 'index_dt']].merge(
    df_te_emb[['individual_id', 'index_dt'] + te_emb_cols],
    on=['individual_id', 'index_dt'],
    how='inner'
)
print(f"TE embeddings matched: {len(df_te_merged):,} / {len(df_merged_base):,}")

te_features = df_te_merged[te_emb_cols].values
print(f"TE features shape: {te_features.shape}")

np.save(f"{OUTPUT_DIR}/te_256_features.npy", te_features)
```

**Step 6: Run cells 8-12 to build all feature representations**

Run: Execute cells 8-12
Expected: 5 feature arrays saved, all with 256 columns (except raw codes)

**Step 7: Commit**

```bash
git add dev/downstream/raw_code_vs_te_comparison.ipynb
git commit -m "feat: build PCA, UMAP, feature-selected, and TE feature representations"
```

---

### Task 4: Downstream Evaluation — Run CatBoost on All 5 Feature Sets

**Files:**
- Modify: `dev/downstream/raw_code_vs_te_comparison.ipynb` (add cells)

**Step 1: Define evaluation function that wraps existing pipeline**

```python
# Cell 13: Evaluation wrapper using existing downstream pipeline functions
def evaluate_feature_set(
    feature_matrix: np.ndarray,
    feature_names: list,
    df_base: pd.DataFrame,
    feature_set_name: str,
    catboost_params: dict = CATBOOST_PARAMS,
    downsample_ratio: int = NEGATIVE_DOWNSAMPLE_RATIO,
) -> dict:
    """
    Evaluate a feature matrix using the same pipeline as TE downstream evaluation.
    
    Args:
        feature_matrix: numpy array (n_members, n_features)
        feature_names: column names for the features
        df_base: DataFrame with individual_id, index_dt, TARGET_COLUMN, ind_id_last_digit
        feature_set_name: name for logging
        
    Returns:
        Dict with split metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {feature_set_name} ({feature_matrix.shape[1]} features)")
    print(f"{'='*70}")
    
    # Build DataFrame with features
    df_eval = df_base[['individual_id', 'index_dt', TARGET_COLUMN, 'ind_id_last_digit']].copy()
    
    # Align: feature_matrix rows correspond to df_base rows
    assert len(feature_matrix) == len(df_eval), \
        f"Row mismatch: {len(feature_matrix)} features vs {len(df_eval)} base rows"
    
    for i, name in enumerate(feature_names):
        df_eval[name] = feature_matrix[:, i].astype(np.float32)
    
    # Create splits using existing function
    splits = create_data_splits(df_eval)
    
    # Prepare features
    X_splits, y_splits = {}, {}
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            X_splits[split_name] = split_df[feature_names].copy()
            y_splits[split_name] = split_df[TARGET_COLUMN].astype(int)
    
    # Downsample training set
    if downsample_ratio and 'train' in X_splits:
        X_splits['train'], y_splits['train'] = downsample_negatives(
            X_splits['train'], y_splits['train'],
            ratio=downsample_ratio, random_state=RANDOM_STATE
        )
    
    # Train CatBoost
    from sklearn.base import clone
    model = CatBoostClassifier(**catboost_params)
    
    train_pool = Pool(X_splits['train'], y_splits['train'])
    val_pool = Pool(X_splits['val'], y_splits['val'])
    model.fit(train_pool, eval_set=val_pool, verbose=0)
    
    # Evaluate on all splits
    results = {'feature_set': feature_set_name, 'n_features': feature_matrix.shape[1]}
    for split_name in ['val', 'test', 'oot', 'oot_strict']:
        if split_name not in X_splits:
            continue
        y_prob = model.predict_proba(X_splits[split_name])[:, 1]
        metrics = compute_split_metrics(
            np.array(y_splits[split_name]), y_prob
        )
        for metric_name, value in metrics.items():
            results[f'{split_name}_{metric_name}'] = value
        print(f"  {split_name}: AUC={metrics['auc_roc']:.4f}, "
              f"Lift@1%={metrics['lift_1pct']:.2f}, "
              f"Lift@5%={metrics['lift_5pct']:.2f}")
    
    results['model'] = model  # Keep for SHAP analysis
    results['X_test'] = X_splits.get('test')
    results['y_test'] = y_splits.get('test')
    
    return results
```

**Step 2: Run evaluation on all 5 feature sets**

```python
# Cell 14: Evaluate all 5 feature representations
all_results = {}

# 1. Raw codes (full sparse → dense for the matched subset)
# For raw codes with ~84k features, CatBoost can handle it but may be slow.
# Use only active codes (codes that appear at least once in the dataset)
active_mask = np.array(code_matrix_matched.sum(axis=0)).flatten() > 0
code_matrix_active = code_matrix_matched[:, active_mask].toarray()
active_code_names = [f'code_{i}' for i in np.where(active_mask)[0]]
print(f"Active code features: {code_matrix_active.shape[1]:,}")

all_results['raw_codes'] = evaluate_feature_set(
    code_matrix_active, active_code_names, df_merged_base,
    f"Raw Codes ({code_matrix_active.shape[1]:,} features)"
)

# 2. PCA(256)
pca_names = [f'pca_{i}' for i in range(TARGET_DIM)]
all_results['pca_256'] = evaluate_feature_set(
    pca_features, pca_names, df_merged_base,
    "PCA(256)"
)

# 3. UMAP(256) 
umap_names = [f'umap_{i}' for i in range(TARGET_DIM)]
all_results['umap_256'] = evaluate_feature_set(
    umap_features, umap_names, df_merged_base,
    "UMAP(256)"
)

# 4. SelectKBest(256)
selected_names = [f'selected_code_{idx}' for idx in selected_code_indices]
all_results['selected_256'] = evaluate_feature_set(
    selected_features, selected_names, df_merged_base,
    "SelectKBest(256, MI)"
)

# 5. TE Embeddings (256)
# Need to align TE features with df_merged_base
# Some members may not have TE embeddings (they weren't in the 30% sample)
# Use only the intersection
te_member_set = set(df_te_merged['individual_id'].values)
te_mask = df_merged_base['individual_id'].isin(te_member_set)
df_te_base = df_merged_base[te_mask].reset_index(drop=True)
te_features_aligned = te_features[:len(df_te_base)]  # Already aligned from merge

te_names = [f'te_emb_{i}' for i in range(te_features.shape[1])]
all_results['te_embedding'] = evaluate_feature_set(
    te_features_aligned, te_names, df_te_base,
    "TE Embedding(256)"
)
```

**Step 3: Create comparison summary table**

```python
# Cell 15: Build comparison summary
comparison_rows = []
for name, result in all_results.items():
    row = {
        'representation': result['feature_set'],
        'n_features': result['n_features'],
    }
    for split in ['val', 'test', 'oot', 'oot_strict']:
        for metric in ['auc_roc', 'auc_pr', 'brier', 'lift_1pct', 'lift_5pct', 'lift_10pct', 'tp_1pct', 'precision_1pct']:
            key = f'{split}_{metric}'
            if key in result:
                row[key] = result[key]
    comparison_rows.append(row)

df_comparison = pd.DataFrame(comparison_rows)
print("\n" + "="*70)
print("DOWNSTREAM COMPARISON: Raw Codes vs Dimension Reduction vs TE")
print("="*70)

# Display key metrics for oot_strict (primary evaluation split)
display_cols = ['representation', 'n_features', 
                'oot_strict_auc_roc', 'oot_strict_auc_pr', 'oot_strict_brier',
                'oot_strict_lift_1pct', 'oot_strict_lift_5pct', 'oot_strict_lift_10pct']
print(df_comparison[display_cols].to_string(index=False))

# Save
df_comparison.to_csv(f"{OUTPUT_DIR}/comparison_results.csv", index=False)
df_comparison.to_excel(f"{OUTPUT_DIR}/comparison_results.xlsx", index=False)
print(f"\nResults saved to {OUTPUT_DIR}/")
```

**Step 4: Run cells 13-15**

Run: Execute cells 13-15
Expected: All 5 feature sets evaluated, comparison table printed and saved

**Step 5: Commit**

```bash
git add dev/downstream/raw_code_vs_te_comparison.ipynb
git commit -m "feat: downstream CatBoost evaluation across all 5 representation types"
```

---

### Task 5: SHAP Frequency-Tier Importance Analysis

**Files:**
- Modify: `dev/downstream/raw_code_vs_te_comparison.ipynb` (add cells)

**Step 1: Compute SHAP values for the raw-code model**

```python
# Cell 16: SHAP analysis on the raw-code CatBoost model
raw_code_model = all_results['raw_codes']['model']
X_test_raw = all_results['raw_codes']['X_test']

print("Computing SHAP values for raw-code model (TreeSHAP)...")
print(f"Test set size: {len(X_test_raw):,} rows, {X_test_raw.shape[1]:,} features")

# TreeSHAP is fast for CatBoost
explainer = shap.TreeExplainer(raw_code_model)

# Use a subsample for SHAP if test set is very large
SHAP_SAMPLE_SIZE = min(5000, len(X_test_raw))
np.random.seed(RANDOM_STATE)
shap_sample_idx = np.random.choice(len(X_test_raw), SHAP_SAMPLE_SIZE, replace=False)
X_shap_sample = X_test_raw.iloc[shap_sample_idx]

shap_values = explainer.shap_values(X_shap_sample)
# For binary classification, shap_values may be a list [class0, class1]
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Use positive class
print(f"SHAP values shape: {shap_values.shape}")

# Save SHAP values
np.save(f"{OUTPUT_DIR}/shap_values_raw_codes.npy", shap_values)
```

**Step 2: Map SHAP values to frequency tiers and compute tier-level importance**

```python
# Cell 17: Aggregate SHAP values by frequency tier
# Map feature names back to code indices
feature_to_code_idx = {}
for fname in active_code_names:
    code_idx = int(fname.split('_')[1])
    feature_to_code_idx[fname] = code_idx

# Compute per-feature mean |SHAP|
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Group by tier
tier_shap = {tier: [] for tier in ['frequent', 'medium', 'rare', 'tail']}
tier_feature_count = {tier: 0 for tier in ['frequent', 'medium', 'rare', 'tail']}

for feat_idx, fname in enumerate(active_code_names):
    code_idx = feature_to_code_idx[fname]
    tier = code_tiers.get(code_idx, 'tail')
    tier_shap[tier].append(mean_abs_shap[feat_idx])
    tier_feature_count[tier] += 1

# Multiple aggregate metrics per tier
print("\n" + "="*70)
print("SHAP IMPORTANCE BY FREQUENCY TIER")
print("="*70)

tier_metrics = {}
for tier in ['frequent', 'medium', 'rare', 'tail']:
    values = tier_shap[tier]
    if not values:
        continue
    values = np.array(values)
    n_codes = tier_feature_count[tier]
    
    metrics = {
        'n_codes': n_codes,
        'total_shap': values.sum(),
        'mean_shap': values.mean(),
        'median_shap': np.median(values),
        'max_shap': values.max(),
        'std_shap': values.std(),
        'pct_of_total_importance': values.sum() / mean_abs_shap.sum() * 100,
        'pct_of_codes': n_codes / len(active_code_names) * 100,
        'importance_concentration': (values.sum() / mean_abs_shap.sum() * 100) / (n_codes / len(active_code_names) * 100),
    }
    tier_metrics[tier] = metrics
    
    print(f"\n{tier.upper()} tier:")
    print(f"  Codes: {n_codes:,} ({metrics['pct_of_codes']:.1f}% of all codes)")
    print(f"  Total |SHAP|: {metrics['total_shap']:.4f} ({metrics['pct_of_total_importance']:.1f}% of total)")
    print(f"  Mean |SHAP|: {metrics['mean_shap']:.6f}")
    print(f"  Median |SHAP|: {metrics['median_shap']:.6f}")
    print(f"  Max |SHAP|: {metrics['max_shap']:.6f}")
    print(f"  Importance concentration: {metrics['importance_concentration']:.2f}x")
    print(f"    (ratio of %importance to %codes — >1 means overrepresented)")

df_tier_importance = pd.DataFrame(tier_metrics).T
df_tier_importance.to_csv(f"{OUTPUT_DIR}/tier_importance_summary.csv")
```

**Step 3: Compute additional tier importance metrics**

```python
# Cell 18: Additional tier analysis — cumulative importance and per-tier ablation
# 1. Cumulative importance: if we only keep codes from tiers X and above, 
#    how much of total importance do we capture?
print("\n" + "="*70)
print("CUMULATIVE IMPORTANCE BY TIER")
print("="*70)

cumulative = 0
total_importance = mean_abs_shap.sum()
for tier in ['frequent', 'medium', 'rare', 'tail']:
    tier_total = sum(tier_shap[tier]) if tier_shap[tier] else 0
    cumulative += tier_total
    print(f"  + {tier}: cumulative = {cumulative/total_importance*100:.1f}% "
          f"(added {tier_total/total_importance*100:.1f}%)")

# 2. Per-tier top-10 codes
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT CODES PER TIER")
print("="*70)

for tier in ['frequent', 'medium', 'rare', 'tail']:
    tier_indices = [i for i, fname in enumerate(active_code_names) 
                    if code_tiers.get(feature_to_code_idx[fname], 'tail') == tier]
    if not tier_indices:
        continue
    tier_shap_vals = mean_abs_shap[tier_indices]
    top10_within = np.argsort(tier_shap_vals)[::-1][:10]
    
    print(f"\n{tier.upper()} tier top 10:")
    for rank, idx_within in enumerate(top10_within):
        global_idx = tier_indices[idx_within]
        code_idx = feature_to_code_idx[active_code_names[global_idx]]
        member_pct = code_member_counts[code_idx] / n_members * 100
        print(f"  {rank+1}. code_{code_idx}: |SHAP|={mean_abs_shap[global_idx]:.6f}, "
              f"member prevalence={member_pct:.2f}%")

# 3. Importance-weighted tier distribution
# What fraction of the top-K most important features come from each tier?
print("\n" + "="*70)
print("TIER COMPOSITION OF TOP-K FEATURES BY IMPORTANCE")
print("="*70)

sorted_feat_indices = np.argsort(mean_abs_shap)[::-1]
for K in [50, 100, 256, 500, 1000]:
    top_k_indices = sorted_feat_indices[:K]
    top_k_tiers = [code_tiers.get(feature_to_code_idx[active_code_names[i]], 'tail') 
                   for i in top_k_indices]
    tier_dist = Counter(top_k_tiers)
    dist_str = ", ".join(f"{t}={tier_dist.get(t,0)}" for t in ['frequent','medium','rare','tail'])
    print(f"  Top {K:>4}: {dist_str}")
```

**Step 4: Run cells 16-18**

Run: Execute cells 16-18
Expected: SHAP values computed, tier importance printed, CSVs saved

**Step 5: Commit**

```bash
git add dev/downstream/raw_code_vs_te_comparison.ipynb
git commit -m "feat: SHAP frequency-tier importance analysis for raw-code model"
```

---

### Task 6: Visualization and Summary Report

**Files:**
- Modify: `dev/downstream/raw_code_vs_te_comparison.ipynb` (add cells)

**Step 1: Create comparison bar chart**

```python
# Cell 19: Visualization — comparison across representations
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Prepare data for plotting
rep_names = df_comparison['representation'].values
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

# Plot 1: AUC-ROC across splits
ax = axes[0]
for i, split in enumerate(['test', 'oot_strict']):
    col = f'{split}_auc_roc'
    vals = df_comparison[col].values
    x = np.arange(len(rep_names)) + i*0.35
    ax.bar(x, vals, 0.35, label=split, alpha=0.8)
ax.set_xticks(np.arange(len(rep_names)) + 0.175)
ax.set_xticklabels([n.split('(')[0].strip() for n in rep_names], rotation=45, ha='right')
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC: Test vs OOT-Strict')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Lift@1% for oot_strict
ax = axes[1]
lift_vals = df_comparison['oot_strict_lift_1pct'].values
ax.barh(rep_names, lift_vals, color=colors[:len(rep_names)])
ax.set_xlabel('Lift @ 1%')
ax.set_title('Lift@1% (OOT-Strict)')
ax.grid(axis='x', alpha=0.3)

# Plot 3: Feature count vs AUC (efficiency)
ax = axes[2]
for i, (name, row) in enumerate(df_comparison.iterrows()):
    ax.scatter(row['n_features'], row['oot_strict_auc_roc'], 
              s=150, c=colors[i], label=row['representation'], zorder=5)
ax.set_xlabel('Number of Features')
ax.set_ylabel('AUC-ROC (OOT-Strict)')
ax.set_title('Dimensionality vs Performance')
ax.set_xscale('log')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/representation_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved to {OUTPUT_DIR}/representation_comparison.png")
```

**Step 2: Create SHAP tier importance visualization**

```python
# Cell 20: SHAP tier importance visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

tier_colors = {'frequent': '#2196F3', 'medium': '#4CAF50', 'rare': '#FF9800', 'tail': '#F44336'}
tiers = ['frequent', 'medium', 'rare', 'tail']

# Plot 1: Pie chart — % of total SHAP importance per tier
ax = axes[0, 0]
importance_pcts = [tier_metrics[t]['pct_of_total_importance'] for t in tiers if t in tier_metrics]
tier_labels = [f"{t}\n({tier_metrics[t]['n_codes']:,} codes)" for t in tiers if t in tier_metrics]
ax.pie(importance_pcts, labels=tier_labels, colors=[tier_colors[t] for t in tiers if t in tier_metrics],
       autopct='%1.1f%%', startangle=90)
ax.set_title('Share of Total SHAP Importance by Tier')

# Plot 2: Bar chart — mean |SHAP| per tier
ax = axes[0, 1]
means = [tier_metrics[t]['mean_shap'] for t in tiers if t in tier_metrics]
ax.bar(tiers, means, color=[tier_colors[t] for t in tiers])
ax.set_ylabel('Mean |SHAP|')
ax.set_title('Mean Feature Importance per Code Tier')
ax.grid(axis='y', alpha=0.3)

# Plot 3: Importance concentration ratio
ax = axes[1, 0]
concentrations = [tier_metrics[t]['importance_concentration'] for t in tiers if t in tier_metrics]
ax.bar(tiers, concentrations, color=[tier_colors[t] for t in tiers])
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Proportional (1.0)')
ax.set_ylabel('Importance Concentration Ratio')
ax.set_title('Importance Concentration\n(>1 = overrepresented, <1 = underrepresented)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Tier composition of top-K features
ax = axes[1, 1]
K_values = [50, 100, 256, 500, 1000]
tier_fracs = {t: [] for t in tiers}
for K in K_values:
    top_k = sorted_feat_indices[:K]
    top_k_tiers_list = [code_tiers.get(feature_to_code_idx[active_code_names[i]], 'tail') 
                        for i in top_k]
    tier_ct = Counter(top_k_tiers_list)
    for t in tiers:
        tier_fracs[t].append(tier_ct.get(t, 0) / K * 100)

bottom = np.zeros(len(K_values))
for t in tiers:
    ax.bar([str(k) for k in K_values], tier_fracs[t], bottom=bottom, 
           label=t, color=tier_colors[t])
    bottom += np.array(tier_fracs[t])
ax.set_xlabel('Top-K Features')
ax.set_ylabel('% from each tier')
ax.set_title('Tier Composition of Top-K Most Important Features')
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_tier_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved to {OUTPUT_DIR}/shap_tier_analysis.png")
```

**Step 3: Create summary markdown cell**

```python
# Cell 21: Summary report
print("="*70)
print("EXPERIMENT SUMMARY: Raw Code vs TE Downstream Comparison")
print("="*70)

print(f"""
METHODOLOGY:
- Task: Commercial IP (inpatient) 6-month prediction
- Downstream model: CatBoost (iterations=2500, depth=7, balanced weights)
- Splits: Train (digits 0-7), Val (8), Test (9), OOT (>2023-10-16), OOT_strict (OOT + digit 9)
- Downsample: 10:1 negative-to-positive on training set
- Primary metric: AUC-ROC on OOT_strict split

REPRESENTATIONS COMPARED:
1. Raw Codes: ~{code_matrix_active.shape[1]:,} binary/frequency features (all active codes)
2. PCA(256): TruncatedSVD on raw codes, {svd.explained_variance_ratio_.sum()*100:.1f}% variance explained
3. UMAP(256): Nonlinear reduction via UMAP (cosine distance)
4. SelectKBest(256): Top 256 codes by mutual information with target
5. TE Embedding(256): Transformer pretrained encoder output (R6 best model)

KEY FINDINGS:
""")

# Print the comparison table
print(df_comparison[display_cols].to_string(index=False))

print(f"""
FREQUENCY TIER IMPORTANCE (SHAP on raw-code model):
""")
print(df_tier_importance.to_string())

print(f"""
IMPLICATIONS FOR TE:
- Compare raw codes vs TE: does TE add value beyond bag-of-codes?
- Compare PCA/UMAP vs TE: does TE capture more than linear/nonlinear reduction?
- Compare SelectKBest vs TE: is the TE's "selection" better than MI-based selection?
- Tier analysis: are rare/tail codes naturally important for IP prediction?
  If yes → TE's failure to capture rare codes explains its limited downstream value
  If no → the information is in frequent codes and TE's redundancy with tabular is expected
""")
```

**Step 4: Run cells 19-21**

Run: Execute cells 19-21
Expected: Visualizations generated, summary printed

**Step 5: Final commit**

```bash
git add dev/downstream/raw_code_vs_te_comparison.ipynb
git commit -m "feat: visualization and summary report for raw code vs TE comparison"
```

---

## Output Artifacts Summary

| Artifact | Path | Description |
|----------|------|-------------|
| **Jupyter Notebook** | `dev/downstream/raw_code_vs_te_comparison.ipynb` | Complete analysis notebook |
| **Sparse Code Matrix** | `experiment_logs/raw_code_vs_te_comparison/code_frequency_matrix.npz` | Reusable sparse frequency matrix |
| **Frequency Tiers** | `experiment_logs/raw_code_vs_te_comparison/code_frequency_tiers.json` | Code-to-tier mapping |
| **Reduced Features** | `experiment_logs/raw_code_vs_te_comparison/{pca,umap,selected,te}_256_features.npy` | All 256d representations |
| **Comparison Results** | `experiment_logs/raw_code_vs_te_comparison/comparison_results.{csv,xlsx}` | Performance comparison table |
| **SHAP Values** | `experiment_logs/raw_code_vs_te_comparison/shap_values_raw_codes.npy` | Raw SHAP values |
| **Tier Importance** | `experiment_logs/raw_code_vs_te_comparison/tier_importance_summary.csv` | Tier-level SHAP aggregate |
| **Figures** | `experiment_logs/raw_code_vs_te_comparison/*.png` | Comparison and SHAP plots |

## Data Flow Between Notebook and Existing Pipeline

```
NOTEBOOK outputs → EXISTING PIPELINE inputs:
  - Not needed: notebook self-contains evaluation via imported functions
  - Notebook imports: compute_split_metrics, create_data_splits, downsample_negatives
  - Notebook imports: CatBoostClassifier with same CATBOOST_PARAMS

EXISTING PIPELINE outputs → NOTEBOOK inputs:
  - TE embeddings from BigQuery (TE_EMBEDDING_TABLE)
  - Downstream features from BigQuery (FEATURES_TABLE)
  - Same evaluation methodology (splits, metrics, CatBoost config)
```

## Interpretation Guide

After running this experiment, the key comparisons are:

1. **Raw Codes vs TE**: If raw codes ≥ TE → TE fails to add value beyond bag-of-codes (confirms V1's tabular redundancy hypothesis)
2. **PCA(256) vs TE**: If PCA ≈ TE → TE is essentially doing linear compression (the "expensive PCA" critique)
3. **UMAP(256) vs TE**: If UMAP > TE → TE fails to capture nonlinear code interactions
4. **SelectKBest(256) vs TE**: If SelectKBest ≈ TE → the 256d bottleneck selects the same information regardless of method
5. **SHAP tier analysis**: If rare/tail codes have high importance → underrepresentation in TE is a real loss; if low importance → frequent-code dominance in TE is rational behavior
