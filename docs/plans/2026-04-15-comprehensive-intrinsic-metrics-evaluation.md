# Comprehensive Intrinsic Metrics Evaluation Notebook

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Jupyter notebook that evaluates the trained exp2b model on the validation set, computing micro/macro recall@10/@20, micro/macro precision@10/@20, plus all metrics broken down by code type (ICD-10, CPT, GPI, etc.) — without retraining.

**Architecture:** Load the trained checkpoint via `load_model_from_checkpoint()`, reconstruct the exact validation split from the training BigQuery table using the same `train_test_split` parameters, then run a streaming evaluation loop that tracks both sample-level and per-code statistics. Per-code accumulators (~6297 ints × 5 arrays ≈ 126KB) enable macro averaging and code-type-specific breakdowns with negligible memory overhead.

**Tech Stack:** PyTorch, BigQuery (google-cloud-bigquery), numpy, pandas, moe_flashattn_4_core.py (reuse model classes, data utilities, StreamingMetrics)

---

## Prerequisites

1. **Run environment:** GCP Vertex AI notebook instance (same as training) with GPU access
2. **Working directory:** `dev/moe/` (so `moe_flashattn_4_core.py` is importable)
3. **Trained model:** `logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool_formal/saved_models/exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt`
4. **BigQuery tables:**
   - Training data: `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
   - w2ind_target: `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`
5. **CRITICAL:** The BigQuery training table must contain the same data as when the model was trained. If the table was updated after training, the validation split will not reproduce correctly.

## Metrics Definitions (Mathematical Precision)

All metrics are computed over flattened (sample, valid-day) pairs. Let N = total valid-day pairs.

### Sample-Level Metrics (from existing StreamingMetrics)

| Metric | Formula | Notes |
|--------|---------|-------|
| Recall@K (Hit Rate) | Σᵢ 𝟙(\|trueᵢ ∩ topKᵢ\| > 0) / N | Binary: any hit = success |
| Precision@K | (1/N) Σᵢ (\|trueᵢ ∩ topKᵢ\| / K) | = Micro Precision@K |
| NDCG@K | See DCG/IDCG formula | Position-discounted ranking quality |
| MRR | (1/N) Σᵢ 1/rankᵢ | Reciprocal rank of first hit |
| Positive Brier | mean((p_c - 1)² for all positive labels) | Calibration on positives only |

### New Global Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Micro Recall@K** | Σᵢ\|trueᵢ ∩ topKᵢ\| / Σᵢ\|trueᵢ\| | What fraction of ALL true codes are captured |
| **Micro Precision@K** | Σᵢ\|trueᵢ ∩ topKᵢ\| / (N × K) | What fraction of predictions are correct |
| **Macro Recall@K** | (1/\|C_active\|) Σ_c (hits_c / true_c) | Per-code recall, averaged |
| **Macro Precision@K** | (1/\|C_pred\|) Σ_c (hits_c / pred_c) | Per-code precision, averaged |

Where for each code c:
- `true_c` = count of (sample,day) pairs where c ∈ ground_truth
- `pred_c@K` = count of (sample,day) pairs where c ∈ topK_predictions
- `hits_c@K` = count of (sample,day) pairs where c ∈ ground_truth AND c ∈ topK_predictions
- `C_active` = {c : true_c > 0} (codes that appear in ground truth)
- `C_pred` = {c : pred_c@K > 0} (codes that appear in predictions)

### Code-Type-Specific Metrics

Same micro/macro formulas but restricted to codes belonging to each type:

| Code Type | Prefix Pattern | Example |
|-----------|---------------|---------|
| ICD-10 Diagnosis | `icd9_dx_cd*` | icd9_dx_cdG24 |
| Procedure Groups | `prcdr_group_*` | prcdr_group_992 |
| GPI Medications | `gpi*` | gpi22 |
| Provider Taxonomy | `provider_taxonomy_cd*` | provider_taxonomy_cd207Q |
| Revenue Codes | `revenue_cd*` | revenue_cd025 |
| DRG Codes | `drg_cd*` | drg_cd470 |
| Days Count | `days_cnt*` | days_cnt_5 |
| Place of Service | `hcfa_plc_srv_cd*` | hcfa_plc_srv_cd21 |

---

## Task 1: Create Notebook File with Environment Setup

**Files:**
- Create: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Create the notebook with markdown title cell**

```python
# Cell 0 (markdown):
"""
# Comprehensive Intrinsic Metrics Evaluation
## exp2b_flash_learned_pool — Validation Set Analysis

**Purpose:** Evaluate trained model on validation set without retraining.
Compute micro/macro recall and precision @10/@20, plus breakdowns by code type.

**Model:** `exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt`

### Metrics Computed
1. **Global metrics** — Micro/Macro Recall@10, @20; Micro/Macro Precision@10, @20; NDCG; MRR; Positive Brier
2. **Code-type metrics** — All above broken down by: ICD-10, Procedures, GPI, Provider, Revenue, DRG, Days, Place of Service

### Table of Contents
1. Environment Setup
2. Configuration
3. Load Trained Model
4. Load Validation Data
5. Load w2ind_target & Build Code Type Mapping
6. Comprehensive Evaluation
7. Global Metrics Results
8. Code-Type-Specific Results
9. Visualization
"""
```

**Step 2: Create the imports cell**

```python
# Cell 1 (code):
import sys
import os
import gc
import time
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from google.cloud import bigquery

warnings.filterwarnings("ignore")

MODULE_DIR = os.getcwd()
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from moe_flashattn_4_core import (
    BaseConfig,
    FlashAttentionConfig,
    MoEConfig,
    ClinicalDatasetLazy,
    create_collate_fn,
    FlashAttentionTransformer,
    DataParallelWrapper,
    StreamingMetrics,
    prepare_data_once,
    cleanup_gpu_memory,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {props.total_memory / 1e9:.1f} GB")
```

**Step 3: Verify the notebook runs the imports cell without error**

Run the notebook on the GCP Vertex instance. Expected: No import errors, GPU detected.

---

## Task 2: Configuration Cell

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Add configuration markdown and code cell**

```python
# Cell 2 (markdown):
"""
## 2. Configuration

### Key Parameters
- Model checkpoint path on GCP Vertex
- Training data table (to reconstruct validation split)
- w2ind_target table (for code type classification)
- K values for top-K metrics
- Train/val split ratio and seed (MUST match training run)
"""
```

```python
# Cell 3 (code):
# ============================================================================
# CONFIGURATION
# ============================================================================

TRAINED_MODEL_PATH = (
    "logs/exp_round10_3lobs_formal_training/"
    "exp2b_flash_learned_pool_formal/saved_models/"
    "exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt"
)

TRAINING_DATA_TABLE = (
    "edp-prod-storage.edp_ent_sdoheir_cns."
    "a834793_Combined_All_LOB_o3_train_ending"
)

W2IND_TARGET_TABLE = (
    "edp-prod-storage.edp_ent_sdoheir_cns."
    "a834793_member_w2ind_target"
)

# CRITICAL: Must match the training run exactly
TRAIN_RATIO = 0.99    # 99% train / 1% validation (formal training)
RANDOM_SEED = 42

# Top-K values for metrics
K_VALUES = (1, 5, 10, 20, 50)
PRIMARY_K_VALUES = (10, 20)  # For detailed reporting

# Evaluation
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 4

# Macro metrics: minimum sample count per code for inclusion
MACRO_MIN_COUNT = 5

print(f"Model: {TRAINED_MODEL_PATH}")
print(f"Training data: {TRAINING_DATA_TABLE}")
print(f"w2ind_target: {W2IND_TARGET_TABLE}")
print(f"Train/Val split: {TRAIN_RATIO}/{1-TRAIN_RATIO:.2f} (seed={RANDOM_SEED})")
print(f"K values: {K_VALUES}")
print(f"Primary K values for detailed report: {PRIMARY_K_VALUES}")
assert os.path.exists(TRAINED_MODEL_PATH), f"Model not found: {TRAINED_MODEL_PATH}"
```

---

## Task 3: Load Trained Model from Checkpoint

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Add the `load_model_from_checkpoint` function**

Copy the `load_model_from_checkpoint()` function directly from `exp_round10_training_inference_handoff.ipynb` cell (execution_count=3, id=16aa4c93). This is the same function — do NOT modify it.

```python
# Cell 4 (markdown):
"""
## 3. Load Trained Model

Load the trained FlashAttentionTransformer from checkpoint.
The checkpoint contains model_state_dict, config, and model_type.
"""
```

```python
# Cell 5 (code):
# Paste load_model_from_checkpoint() from exp_round10_training_inference_handoff.ipynb
# (the full function ~140 lines, copy verbatim)
```

**Step 2: Load the model**

```python
# Cell 6 (code):
cleanup_gpu_memory(verbose=False)

model, config, moe_config_loaded, use_mixed_precision, model_type = load_model_from_checkpoint(
    model_path=TRAINED_MODEL_PATH,
    device=device,
    verbose=True,
)

print(f"\nConfig summary:")
print(f"  target_cd_cnt: {config.target_cd_cnt}")
print(f"  len_dy: {config.len_dy}")
print(f"  len_cd: {config.len_cd}")
print(f"  cd_cnt: {config.cd_cnt}")
print(f"  embedding_size: {config.embedding_size}")
```

Expected output: Model type FlashAttentionTransformer, 25.3M params, target_cd_cnt=6297.

---

## Task 4: Load Validation Data

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Load full training data from BigQuery**

```python
# Cell 7 (markdown):
"""
## 4. Load Validation Data

Reconstruct the exact validation split used during training:
1. Load full training table from BigQuery
2. Deduplicate (single-record members only)
3. Stratified split with same TRAIN_RATIO and RANDOM_SEED
4. Keep only val_df; free training data from memory
"""
```

```python
# Cell 8 (code):
client = bigquery.Client()

training_sql = f"""
SELECT *
FROM `{TRAINING_DATA_TABLE}`
"""

print("Loading training data from BigQuery (full dataset)...")
print(f"Table: {TRAINING_DATA_TABLE}")
start_time = time.time()

input_data = client.query(training_sql).to_dataframe()

elapsed = time.time() - start_time
print(f"Loaded {len(input_data):,} rows in {elapsed:.1f}s")
print(f"Columns: {list(input_data.columns)}")
print(f"Memory: {input_data.memory_usage(deep=True).sum() / 1e9:.2f} GB")
```

**Step 2: Deduplicate and split**

```python
# Cell 9 (code):
# Deduplicate: keep only members with exactly 1 record
member_counts = input_data.groupby("individual_id").size()
single_record_members = member_counts[member_counts == 1].index
df_unique = input_data[input_data["individual_id"].isin(single_record_members)].copy()

del input_data
gc.collect()

print(f"Unique members (single record): {len(df_unique):,}")
print(f"LOB distribution:\n{df_unique['lob'].value_counts()}")

# Stratified train/val split — MUST match training run
train_df, val_df = train_test_split(
    df_unique,
    train_size=TRAIN_RATIO,
    stratify=df_unique["lob"],
    random_state=RANDOM_SEED,
)

# Free training data immediately — we only need val_df
del train_df, df_unique
gc.collect()

print(f"\nValidation set: {len(val_df):,} members")
print(f"Val LOB distribution:\n{val_df['lob'].value_counts()}")
```

**Step 3: Create validation DataLoader**

```python
# Cell 10 (code):
val_dataset = ClinicalDatasetLazy(val_df, config)
val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=create_collate_fn(config),
    pin_memory=True,
    drop_last=False,
)

print(f"Validation DataLoader: {len(val_loader)} batches "
      f"({len(val_dataset)} samples, batch_size={EVAL_BATCH_SIZE})")
```

---

## Task 5: Load w2ind_target and Build Code Type Mapping

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Load w2ind_target from BigQuery**

```python
# Cell 11 (markdown):
"""
## 5. Code Type Classification

Load w2ind_target to map each target vocabulary index to its code string,
then classify by code type using prefix patterns.

Code types (from `create_w2ind_target_from_w2ind.sql`):
| Type | Prefix | Example |
|------|--------|---------|
| ICD-10 Diagnosis | `icd9_dx_cd` | icd9_dx_cdG24 |
| Procedure Groups | `prcdr_group_` | prcdr_group_992 |
| GPI Medications | `gpi` | gpi22 |
| Provider Taxonomy | `provider_taxonomy_cd` | provider_taxonomy_cd207Q |
| Revenue Codes | `revenue_cd` | revenue_cd025 |
| DRG Codes | `drg_cd` | drg_cd470 |
| Days Count | `days_cnt` | days_cnt_5 |
| Place of Service | `hcfa_plc_srv_cd` | hcfa_plc_srv_cd21 |
"""
```

```python
# Cell 12 (code):
w2ind_target_sql = f"""
SELECT cd, ind
FROM `{W2IND_TARGET_TABLE}`
WHERE cd IS NOT NULL AND cd != ''
ORDER BY ind
"""

print("Loading w2ind_target from BigQuery...")
w2ind_target_df = client.query(w2ind_target_sql).to_dataframe()
print(f"Loaded {len(w2ind_target_df):,} target codes")

# Build index → code string mapping
idx_to_code = {}
for _, row in w2ind_target_df.iterrows():
    idx_to_code[int(row['ind'])] = str(row['cd'])

print(f"Index range: 1 to {max(idx_to_code.keys())}")
print(f"Sample mappings: {dict(list(idx_to_code.items())[:5])}")
```

**Step 2: Build code type classification**

```python
# Cell 13 (code):
def classify_code_type(code_str: str) -> str:
    """
    Classify a target code string into its code type category.
    Mirrors the grouping logic in create_w2ind_target_from_w2ind.sql.
    """
    if code_str.startswith('icd9_dx_cd'):
        return 'ICD-10 Diagnosis'
    elif code_str.startswith('prcdr_group_'):
        return 'Procedure Groups'
    elif code_str.startswith('gpi'):
        return 'GPI Medications'
    elif code_str.startswith('provider_taxonomy_cd'):
        return 'Provider Taxonomy'
    elif code_str.startswith('revenue_cd'):
        return 'Revenue Codes'
    elif code_str.startswith('drg_cd'):
        return 'DRG Codes'
    elif code_str.startswith('days_cnt'):
        return 'Days Count'
    elif code_str.startswith('hcfa_plc_srv_cd'):
        return 'Place of Service'
    else:
        return 'Other'

# Build index → code_type mapping
idx_to_type = {}
for idx, code_str in idx_to_code.items():
    idx_to_type[idx] = classify_code_type(code_str)

# Build code_type → set of indices mapping
type_to_indices = defaultdict(set)
for idx, code_type in idx_to_type.items():
    type_to_indices[code_type].add(idx)

# Summary
print("Code Type Distribution in Target Vocabulary:")
print("-" * 50)
for code_type in sorted(type_to_indices.keys()):
    indices = type_to_indices[code_type]
    print(f"  {code_type:25s}: {len(indices):,} codes")
print(f"  {'TOTAL':25s}: {len(idx_to_type):,} codes")

# Create a numpy array for fast lookup: idx → type_id
CODE_TYPES = sorted(type_to_indices.keys())
type_to_id = {t: i for i, t in enumerate(CODE_TYPES)}
idx_type_array = np.full(config.target_cd_cnt, -1, dtype=np.int32)
for idx, code_type in idx_to_type.items():
    if idx < config.target_cd_cnt:
        idx_type_array[idx] = type_to_id[code_type]

print(f"\nType ID mapping: {type_to_id}")
```

---

## Task 6: Comprehensive Evaluation Loop

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Define the per-code accumulator class**

This is the core new component. It wraps `StreamingMetrics` and adds per-code tracking arrays.

```python
# Cell 14 (markdown):
"""
## 6. Comprehensive Evaluation

### Approach
1. Reuse `StreamingMetrics` for sample-level metrics (recall@K, precision@K, NDCG, MRR, Brier)
2. Add per-code accumulators (numpy arrays, ~126KB total) for macro metrics:
   - `code_true_count[c]`: times code c appears in ground truth
   - `code_topk_count[k][c]`: times code c appears in top-K predictions
   - `code_hit_count[k][c]`: times code c appears in BOTH ground truth AND top-K
3. After evaluation, derive macro and code-type metrics from per-code arrays
"""
```

```python
# Cell 15 (code):
class PerCodeAccumulator:
    """
    Memory-efficient per-code tracking for macro metrics and code-type breakdowns.
    
    Maintains 3 types of counters as numpy arrays (vocab_size ≈ 6297):
    - code_true_count: ground truth frequency per code
    - code_topk_count: top-K prediction frequency per code (one array per K)
    - code_hit_count: hit frequency per code (one array per K)
    
    Total memory: ~5 arrays × 6297 × 8 bytes ≈ 252KB
    """
    
    def __init__(self, vocab_size: int, k_values: Tuple[int, ...]):
        self.vocab_size = vocab_size
        self.k_values = k_values
        self._max_k = max(k_values)
        
        self.code_true_count = np.zeros(vocab_size, dtype=np.int64)
        self.code_topk_count = {k: np.zeros(vocab_size, dtype=np.int64) for k in k_values}
        self.code_hit_count = {k: np.zeros(vocab_size, dtype=np.int64) for k in k_values}
        self.total_samples = 0
    
    def update(
        self,
        predictions: torch.Tensor,   # [batch, vocab_size] logits
        targets: List[List[int]],     # target code lists per sample
    ) -> None:
        """Update per-code counters from a batch (vectorized on GPU)."""
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Build target boolean tensor [batch, vocab_size]
        target_tensor = torch.zeros(
            batch_size, self.vocab_size, dtype=torch.bool, device=device
        )
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, target_codes in enumerate(targets):
            valid_codes = [c for c in target_codes if 0 < c < self.vocab_size]
            if valid_codes:
                target_tensor[i, valid_codes] = True
                valid_mask[i] = True
        
        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            return
        
        self.total_samples += num_valid
        
        # Update code_true_count: sum targets over batch
        code_true_batch = target_tensor[valid_mask].sum(dim=0).cpu().numpy()
        self.code_true_count += code_true_batch
        
        # Get top-K indices (compute once for max K)
        with torch.no_grad():
            _, top_k_indices = torch.topk(predictions, self._max_k, dim=-1)
        
        for k in self.k_values:
            top_k = top_k_indices[:, :k]  # [batch, k]
            
            # Build top-K boolean tensor [batch, vocab_size]
            topk_tensor = torch.zeros(
                batch_size, self.vocab_size, dtype=torch.bool, device=device
            )
            topk_tensor.scatter_(1, top_k, True)
            
            # code_topk_count: how many valid samples have this code in top-K
            topk_valid = topk_tensor[valid_mask].sum(dim=0).cpu().numpy()
            self.code_topk_count[k] += topk_valid
            
            # code_hit_count: intersection of targets and top-K
            hits = (target_tensor & topk_tensor)[valid_mask].sum(dim=0).cpu().numpy()
            self.code_hit_count[k] += hits
        
        # Cleanup
        del target_tensor, topk_tensor, top_k_indices
    
    def compute_macro_metrics(
        self, min_count: int = 5
    ) -> Dict[str, float]:
        """
        Compute macro-averaged recall and precision at each K.
        
        Args:
            min_count: Minimum ground-truth count for a code to be included
                       in macro averaging (avoids noise from ultra-rare codes).
        """
        metrics = {}
        
        for k in self.k_values:
            # Macro Recall@K
            active_mask = self.code_true_count >= min_count
            if active_mask.sum() > 0:
                per_code_recall = np.divide(
                    self.code_hit_count[k],
                    self.code_true_count,
                    out=np.zeros_like(self.code_hit_count[k], dtype=np.float64),
                    where=active_mask,
                )
                macro_recall = per_code_recall[active_mask].mean()
                metrics[f'macro_recall@{k}'] = float(macro_recall)
                metrics[f'macro_recall@{k}_num_codes'] = int(active_mask.sum())
            else:
                metrics[f'macro_recall@{k}'] = 0.0
                metrics[f'macro_recall@{k}_num_codes'] = 0
            
            # Macro Precision@K
            pred_mask = self.code_topk_count[k] >= min_count
            if pred_mask.sum() > 0:
                per_code_precision = np.divide(
                    self.code_hit_count[k],
                    self.code_topk_count[k],
                    out=np.zeros_like(self.code_hit_count[k], dtype=np.float64),
                    where=pred_mask,
                )
                macro_precision = per_code_precision[pred_mask].mean()
                metrics[f'macro_precision@{k}'] = float(macro_precision)
                metrics[f'macro_precision@{k}_num_codes'] = int(pred_mask.sum())
            else:
                metrics[f'macro_precision@{k}'] = 0.0
                metrics[f'macro_precision@{k}_num_codes'] = 0
        
        return metrics
    
    def compute_code_type_metrics(
        self,
        idx_type_array: np.ndarray,
        type_to_id: Dict[str, int],
        code_types: List[str],
        min_count: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics broken down by code type.
        
        Returns:
            Dict[code_type_name → Dict[metric_name → value]]
        """
        results = {}
        
        for code_type in code_types:
            type_id = type_to_id[code_type]
            type_mask = idx_type_array == type_id
            num_codes_in_type = type_mask.sum()
            
            if num_codes_in_type == 0:
                continue
            
            type_metrics = {
                'num_codes': int(num_codes_in_type),
                'total_true_occurrences': int(self.code_true_count[type_mask].sum()),
            }
            
            for k in self.k_values:
                type_true = self.code_true_count[type_mask]
                type_topk = self.code_topk_count[k][type_mask]
                type_hits = self.code_hit_count[k][type_mask]
                
                # Micro Recall: total hits of this type / total true of this type
                total_true = type_true.sum()
                total_hits = type_hits.sum()
                total_topk = type_topk.sum()
                
                type_metrics[f'micro_recall@{k}'] = (
                    float(total_hits / total_true) if total_true > 0 else 0.0
                )
                type_metrics[f'micro_precision@{k}'] = (
                    float(total_hits / total_topk) if total_topk > 0 else 0.0
                )
                
                # Macro Recall: average per-code recall within this type
                active = type_true >= min_count
                if active.sum() > 0:
                    per_code_recall = np.divide(
                        type_hits, type_true,
                        out=np.zeros_like(type_hits, dtype=np.float64),
                        where=active,
                    )
                    type_metrics[f'macro_recall@{k}'] = float(per_code_recall[active].mean())
                    type_metrics[f'macro_recall@{k}_num_codes'] = int(active.sum())
                else:
                    type_metrics[f'macro_recall@{k}'] = 0.0
                    type_metrics[f'macro_recall@{k}_num_codes'] = 0
                
                # Macro Precision: average per-code precision within this type
                pred_active = type_topk >= min_count
                if pred_active.sum() > 0:
                    per_code_prec = np.divide(
                        type_hits, type_topk,
                        out=np.zeros_like(type_hits, dtype=np.float64),
                        where=pred_active,
                    )
                    type_metrics[f'macro_precision@{k}'] = float(per_code_prec[pred_active].mean())
                    type_metrics[f'macro_precision@{k}_num_codes'] = int(pred_active.sum())
                else:
                    type_metrics[f'macro_precision@{k}'] = 0.0
                    type_metrics[f'macro_precision@{k}_num_codes'] = 0
            
            results[code_type] = type_metrics
        
        return results
```

**Step 2: Define the comprehensive evaluation function**

```python
# Cell 16 (code):
from contextlib import nullcontext

def _model_has_moe(model):
    """Check if model has MoE layers."""
    if isinstance(model, nn.DataParallel):
        model = model.module
    if isinstance(model, DataParallelWrapper):
        model = model.model
    return hasattr(model, 'temporal_layers') and any(
        hasattr(layer, 'moe') for layer in model.temporal_layers
    )

def evaluate_comprehensive(
    model: nn.Module,
    dataloader: DataLoader,
    config: BaseConfig,
    device: torch.device,
    k_values: Tuple[int, ...] = (1, 5, 10, 20, 50),
    use_mixed_precision: bool = True,
    idx_type_array: np.ndarray = None,
    type_to_id: Dict[str, int] = None,
    code_types: List[str] = None,
    macro_min_count: int = 5,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation with sample-level, macro, and code-type metrics.
    
    Returns dict with keys:
        'sample_level': StreamingMetrics results
        'macro': Macro recall/precision results
        'code_type': Per-code-type breakdown
        'per_code': Raw per-code accumulators (for further analysis)
    """
    model.eval()
    num_batches = len(dataloader)
    
    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and
        isinstance(model.module, DataParallelWrapper)
    )
    
    # Standard sample-level streaming metrics
    streaming = StreamingMetrics(
        k_values=k_values,
        compute_mrr=True,
        compute_brier=True,
        vocab_size=config.target_cd_cnt,
    )
    
    # Per-code accumulator for macro metrics
    per_code = PerCodeAccumulator(
        vocab_size=config.target_cd_cnt,
        k_values=k_values,
    )
    
    # Loss criterion (for val_loss computation)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Evaluating {num_batches} batches...")
    eval_start = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # === Forward pass ===
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            lob = batch['lob'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),
                codes,
            ], dim=-1)
            
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if use_mixed_precision else nullcontext()
            )
            
            with autocast_ctx:
                if is_wrapped:
                    targets_mh = batch['target_multihot'].to(device, non_blocking=True)
                    dt_cnt_tensor = (
                        dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor)
                        else torch.tensor(dt_cnt, device=device)
                    )
                    result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                    if isinstance(result, tuple):
                        loss_val, extras = result
                        output = extras.get('predictions')
                    else:
                        loss_val = result
                        output = None
                    loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
                else:
                    if _model_has_moe(model):
                        output, _ = model(x, return_moe_losses=False)
                    else:
                        output = model(x)
                    # Compute loss manually
                    from moe_flashattn_4_core import compute_loss
                    loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
            
            if output is None:
                streaming.update_loss(loss)
                continue
            
            # === Flatten over valid days ===
            batch_size = output.shape[0]
            actual_len_dy = output.shape[1]
            
            dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
            y_flat = [item for sublist in y for item in sublist]
            
            output_flat = output.view(batch_size * actual_len_dy, config.target_cd_cnt)
            
            valid_outputs = []
            valid_targets = []
            
            for j in range(batch_size):
                valid_days = min(int(dt_cnt_values[j]), actual_len_dy)
                if valid_days <= 0:
                    continue
                
                out_start = actual_len_dy * j
                out_end = out_start + valid_days
                valid_outputs.append(output_flat[out_start:out_end])
                
                y_start = config.len_dy * j
                y_end = y_start + valid_days
                valid_targets.extend(y_flat[y_start:y_end])
            
            if valid_outputs:
                predictions_flat = torch.cat(valid_outputs)
                
                # Update sample-level metrics
                streaming.update_loss(loss)
                streaming.update(predictions_flat, valid_targets)
                
                # Update per-code metrics
                per_code.update(predictions_flat, valid_targets)
                
                del predictions_flat, valid_outputs, valid_targets
            
            del output
            if batch_idx % 500 == 0:
                gc.collect()
    
    eval_time = time.time() - eval_start
    print(f"\nEvaluation completed in {eval_time:.1f}s ({eval_time/60:.1f} min)")
    
    # === Compute all metrics ===
    sample_metrics = streaming.compute()
    macro_metrics = per_code.compute_macro_metrics(min_count=macro_min_count)
    
    code_type_metrics = {}
    if idx_type_array is not None and type_to_id is not None and code_types is not None:
        code_type_metrics = per_code.compute_code_type_metrics(
            idx_type_array, type_to_id, code_types, min_count=1,
        )
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'sample_level': sample_metrics,
        'macro': macro_metrics,
        'code_type': code_type_metrics,
        'per_code': per_code,
        'eval_time_sec': eval_time,
    }
```

**Step 3: Run the evaluation**

```python
# Cell 17 (code):
results = evaluate_comprehensive(
    model=model,
    dataloader=val_loader,
    config=config,
    device=device,
    k_values=K_VALUES,
    use_mixed_precision=use_mixed_precision,
    idx_type_array=idx_type_array,
    type_to_id=type_to_id,
    code_types=CODE_TYPES,
    macro_min_count=MACRO_MIN_COUNT,
)

print("Evaluation complete!")
print(f"  Total valid (sample, day) pairs evaluated: {results['sample_level'].get('num_samples', 'N/A'):,}")
print(f"  Evaluation time: {results['eval_time_sec']:.1f}s")
```

---

## Task 7: Display Global Metrics

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Display sample-level metrics**

```python
# Cell 18 (markdown):
"""
## 7. Global Metrics Results

### 7.1 Sample-Level Metrics (from StreamingMetrics)
"""
```

```python
# Cell 19 (code):
sl = results['sample_level']

print("=" * 70)
print("SAMPLE-LEVEL METRICS")
print("=" * 70)
print(f"\n  Val Loss:        {sl.get('val_loss', 0):.6f}")
print(f"  MRR:             {sl.get('mrr', 0):.4f}")
print(f"  Positive Brier:  {sl.get('positive_brier', 0):.4f}")

print(f"\n  {'K':>4s}  {'Recall@K':>10s}  {'MicroRecall@K':>14s}  {'Precision@K':>12s}  {'NDCG@K':>8s}")
print(f"  {'-'*4}  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*8}")
for k in K_VALUES:
    print(f"  {k:4d}  {sl.get(f'recall@{k}', 0):10.4f}  "
          f"{sl.get(f'micro_recall@{k}', 0):14.4f}  "
          f"{sl.get(f'precision@{k}', 0):12.4f}  "
          f"{sl.get(f'ndcg@{k}', 0):8.4f}")
```

**Step 2: Display macro metrics**

```python
# Cell 20 (code):
ma = results['macro']

print("=" * 70)
print("MACRO METRICS (per-code averaged, min_count >= {})".format(MACRO_MIN_COUNT))
print("=" * 70)
print(f"\n  {'K':>4s}  {'MacroRecall@K':>14s}  {'#Codes(R)':>10s}  {'MacroPrecision@K':>17s}  {'#Codes(P)':>10s}")
print(f"  {'-'*4}  {'-'*14}  {'-'*10}  {'-'*17}  {'-'*10}")
for k in K_VALUES:
    print(f"  {k:4d}  {ma.get(f'macro_recall@{k}', 0):14.4f}  "
          f"{ma.get(f'macro_recall@{k}_num_codes', 0):10,}  "
          f"{ma.get(f'macro_precision@{k}', 0):17.4f}  "
          f"{ma.get(f'macro_precision@{k}_num_codes', 0):10,}")
```

**Step 3: Display focused comparison table for K=10,20**

```python
# Cell 21 (code):
print("=" * 70)
print("PRIMARY METRICS SUMMARY (K=10, K=20)")
print("=" * 70)

for k in PRIMARY_K_VALUES:
    print(f"\n  === K = {k} ===")
    print(f"  Micro Recall@{k}:     {sl.get(f'micro_recall@{k}', 0):.4f}")
    print(f"  Macro Recall@{k}:     {ma.get(f'macro_recall@{k}', 0):.4f}")
    print(f"  Micro Precision@{k}:  {sl.get(f'precision@{k}', 0):.4f}")
    print(f"  Macro Precision@{k}:  {ma.get(f'macro_precision@{k}', 0):.4f}")
    print(f"  NDCG@{k}:             {sl.get(f'ndcg@{k}', 0):.4f}")
```

---

## Task 8: Display Code-Type-Specific Metrics

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Create summary table per code type**

```python
# Cell 22 (markdown):
"""
## 8. Code-Type-Specific Metrics

Breakdown of all metrics by code type. This reveals which clinical domains
the model predicts best and worst.
"""
```

```python
# Cell 23 (code):
ct = results['code_type']

for k in PRIMARY_K_VALUES:
    print(f"\n{'=' * 100}")
    print(f"CODE-TYPE BREAKDOWN — K = {k}")
    print(f"{'=' * 100}")
    
    header = (f"  {'Code Type':25s}  {'#Codes':>7s}  {'#TrueOcc':>9s}  "
              f"{'µRecall':>8s}  {'MRecall':>8s}  "
              f"{'µPrec':>8s}  {'MPrec':>8s}")
    print(header)
    print(f"  {'-' * 93}")
    
    for code_type in CODE_TYPES:
        if code_type not in ct:
            continue
        m = ct[code_type]
        print(f"  {code_type:25s}  "
              f"{m.get('num_codes', 0):7,}  "
              f"{m.get('total_true_occurrences', 0):9,}  "
              f"{m.get(f'micro_recall@{k}', 0):8.4f}  "
              f"{m.get(f'macro_recall@{k}', 0):8.4f}  "
              f"{m.get(f'micro_precision@{k}', 0):8.4f}  "
              f"{m.get(f'macro_precision@{k}', 0):8.4f}")
```

**Step 2: Create a pandas DataFrame for detailed per-type analysis**

```python
# Cell 24 (code):
rows = []
for code_type in CODE_TYPES:
    if code_type not in ct:
        continue
    m = ct[code_type]
    for k in K_VALUES:
        rows.append({
            'Code Type': code_type,
            'K': k,
            'Num Codes': m.get('num_codes', 0),
            'Total True Occurrences': m.get('total_true_occurrences', 0),
            'Micro Recall@K': m.get(f'micro_recall@{k}', 0),
            'Macro Recall@K': m.get(f'macro_recall@{k}', 0),
            'Micro Precision@K': m.get(f'micro_precision@{k}', 0),
            'Macro Precision@K': m.get(f'macro_precision@{k}', 0),
            'Macro Recall Num Codes': m.get(f'macro_recall@{k}_num_codes', 0),
            'Macro Precision Num Codes': m.get(f'macro_precision@{k}_num_codes', 0),
        })

metrics_df = pd.DataFrame(rows)
print("Full Code-Type Metrics DataFrame:")
display(metrics_df.to_string(index=False))

# Pivot for K=10 and K=20 comparison
for k in PRIMARY_K_VALUES:
    subset = metrics_df[metrics_df['K'] == k].copy()
    subset = subset.sort_values('Total True Occurrences', ascending=False)
    print(f"\n=== K={k} (sorted by true occurrences) ===")
    display(subset[['Code Type', 'Num Codes', 'Total True Occurrences',
                     'Micro Recall@K', 'Macro Recall@K',
                     'Micro Precision@K', 'Macro Precision@K']].to_string(index=False))
```

---

## Task 9: Visualization

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Add visualization imports and code**

```python
# Cell 25 (markdown):
"""
## 9. Visualization

Bar charts comparing micro vs macro recall and precision across code types.
"""
```

```python
# Cell 26 (code):
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (14, 6)
matplotlib.rcParams['font.size'] = 11

for k in PRIMARY_K_VALUES:
    subset = metrics_df[metrics_df['K'] == k].copy()
    subset = subset.sort_values('Total True Occurrences', ascending=False)
    types = subset['Code Type'].values
    x = np.arange(len(types))
    width = 0.2
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Recall subplot
    ax = axes[0]
    ax.bar(x - width/2, subset['Micro Recall@K'].values, width, label='Micro', color='steelblue')
    ax.bar(x + width/2, subset['Macro Recall@K'].values, width, label='Macro', color='coral')
    ax.set_xlabel('Code Type')
    ax.set_ylabel(f'Recall@{k}')
    ax.set_title(f'Recall@{k} by Code Type')
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Precision subplot
    ax = axes[1]
    ax.bar(x - width/2, subset['Micro Precision@K'].values, width, label='Micro', color='steelblue')
    ax.bar(x + width/2, subset['Macro Precision@K'].values, width, label='Macro', color='coral')
    ax.set_xlabel('Code Type')
    ax.set_ylabel(f'Precision@{k}')
    ax.set_title(f'Precision@{k} by Code Type')
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Micro vs Macro Metrics by Code Type — K={k}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

**Step 2: Add micro-macro gap analysis**

```python
# Cell 27 (code):
print("=" * 70)
print("MICRO-MACRO GAP ANALYSIS")
print("=" * 70)
print("\nA large gap between micro and macro indicates the model performs well")
print("on frequent codes but poorly on rare ones within that type.\n")

for k in PRIMARY_K_VALUES:
    subset = metrics_df[metrics_df['K'] == k].copy()
    subset = subset.sort_values('Total True Occurrences', ascending=False)
    
    print(f"  === K = {k} ===")
    print(f"  {'Code Type':25s}  {'µ-M Recall Gap':>15s}  {'µ-M Prec Gap':>13s}  {'#True':>9s}")
    print(f"  {'-' * 67}")
    for _, row in subset.iterrows():
        recall_gap = row['Micro Recall@K'] - row['Macro Recall@K']
        prec_gap = row['Micro Precision@K'] - row['Macro Precision@K']
        print(f"  {row['Code Type']:25s}  {recall_gap:+15.4f}  {prec_gap:+13.4f}  {row['Total True Occurrences']:9,}")
    print()
```

---

## Task 10: Save Results to JSON

**Files:**
- Modify: `dev/moe/exp_round10_comprehensive_intrinsic_eval.ipynb`

**Step 1: Export all metrics to JSON for reproducibility**

```python
# Cell 28 (markdown):
"""
## 10. Save Results
"""
```

```python
# Cell 29 (code):
output_dir = Path("logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool_formal/eval_metrics")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"comprehensive_intrinsic_metrics_{timestamp}.json"

export_data = {
    'model_path': TRAINED_MODEL_PATH,
    'training_data_table': TRAINING_DATA_TABLE,
    'train_ratio': TRAIN_RATIO,
    'random_seed': RANDOM_SEED,
    'eval_batch_size': EVAL_BATCH_SIZE,
    'k_values': list(K_VALUES),
    'macro_min_count': MACRO_MIN_COUNT,
    'num_val_members': len(val_df),
    'total_valid_day_pairs': results['sample_level'].get('num_samples', 0),
    'eval_time_sec': results['eval_time_sec'],
    'sample_level_metrics': {k: float(v) for k, v in results['sample_level'].items()
                             if isinstance(v, (int, float))},
    'macro_metrics': {k: float(v) if isinstance(v, float) else int(v)
                      for k, v in results['macro'].items()},
    'code_type_metrics': results['code_type'],
}

with open(output_path, 'w') as f:
    json.dump(export_data, f, indent=2, default=str)

print(f"Results saved to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
```

---

## Correctness Verification Checklist

Before running the notebook, verify these properties:

1. **No retraining**: The model is loaded in `model.eval()` mode, `torch.no_grad()` wraps the entire evaluation. No optimizer, no backward pass.

2. **Exact validation split**: Same table, same deduplication, same `TRAIN_RATIO`, same `RANDOM_SEED` as training → deterministic split via `sklearn.train_test_split`.

3. **Metric definitions match literature**:
   - Micro Recall@K = total_hits / total_true_labels (pooled across all samples)
   - Macro Recall@K = mean of per-code recall (only codes with true_count ≥ min_count)
   - Micro Precision@K = total_hits / (N × K) (equivalent to sample-averaged precision)
   - Macro Precision@K = mean of per-code precision (only codes with pred_count ≥ min_count)

4. **Valid-day flattening**: Same logic as `_update_streaming_metrics()` — only process `dt_cnt` valid days per member, skip padding.

5. **Code index 0 excluded**: All loops filter `0 < c < vocab_size`.

6. **Memory efficiency**: Per-code accumulators are numpy int64 arrays of size 6297 (~50KB each). No full prediction storage.

7. **Code type classification**: Prefix matching mirrors the SQL logic in `create_w2ind_target_from_w2ind.sql`.

8. **GPU/CPU alignment**: Predictions stay on GPU for `torch.topk` and boolean scatter ops, then `.cpu().numpy()` for accumulation.

---

## Known Considerations

1. **TRAIN_RATIO uncertainty**: The notebook code shows `0.9` but was likely modified to `0.99` for the formal 11M run. If results seem off, try both values.

2. **Macro precision denominator**: Codes that never appear in top-K predictions have `pred_c@K = 0` and are excluded from macro precision averaging. This is mathematically correct but means macro precision uses a different code set than macro recall.

3. **Code-type "Days Count" and "Place of Service"**: These are temporal/categorical features, not clinical events. Their prediction metrics may have different interpretation than diagnoses/procedures.

4. **min_count threshold**: Setting `MACRO_MIN_COUNT=5` excludes ultra-rare codes from macro averages. This is a tradeoff: higher threshold = more stable estimates but fewer codes included. Set to 1 for full coverage.
