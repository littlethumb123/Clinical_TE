# Experiment Round 10: Training & Inference Notebook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a self-contained Jupyter notebook for `exp2b_flash_learned_pool` (experiment round 10) that covers end-to-end pre-training on the 11M dataset and embedding generation for inference — suitable for handoff to another data scientist with zero codebase context.

**Architecture:** The notebook imports all model architecture, data parsing, and training orchestration from two existing core modules (`moe_flashattn_4_core.py` for model/config/data classes and `moe_flashattn_4.py` for training functions). Inference uses the `EmbeddingExtractor` hook-based approach from the downstream notebooks. No new modules or functionality are created — everything is derived directly from the existing codebase.

**Tech Stack:** PyTorch, pandas, numpy, BigQuery (google-cloud-bigquery), sklearn (for train/test split), tqdm

---

## Dependency Map

| What | Source Module | Key Symbols |
|------|--------------|-------------|
| **Config classes** | `moe_flashattn_4_core.py` | `BaseConfig`, `FlashAttentionConfig`, `MoEConfig`, `OptimizeConfig` |
| **Experiment configs** | `moe_flashattn_4_core.py` | `get_experiment_configs()` |
| **Data parsing** | `moe_flashattn_4_core.py` | `conv_cd`, `conv_age_gender`, `conv_lob`, `conv_target`, `ClinicalDataset`, `create_collate_fn` |
| **Model classes** | `moe_flashattn_4_core.py` | `FlashAttentionTransformer`, `FlashMoETransformer`, `BaselineTransformer`, `DataParallelWrapper` |
| **Embedding extraction** | `moe_flashattn_4_core.py` | `EmbeddingExtractor` |
| **GPU cleanup** | `moe_flashattn_4_core.py` | `cleanup_gpu_memory` |
| **Training dataset (lazy)** | `moe_flashattn_4.py` | `ClinicalDatasetLazy` |
| **Training pipeline** | `moe_flashattn_4.py` | `run_single_experiment`, `prepare_data_once`, `setup_experiment_logging` |

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| **Training data (11M)** | BigQuery: `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending` | Full 3-LOB training dataset (~11M members) |
| **Commercial heldout** | BigQuery: `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5` | Commercial LOB data for embedding generation |

## Required DataFrame Columns

| Column | Type | Description |
|--------|------|-------------|
| `individual_id` | str | Unique member identifier |
| `index_dt` | str/date | Index date |
| `age_in_months` | str | Day-separated age values (`"val1*val2*..."`) |
| `gender_cd` | str | Day-separated gender codes (`"val1*val2*..."`) |
| `cd` | str | Day-and-code-separated diagnosis codes (`"c1,c2*c3,c4*..."`) |
| `lob` | str | Line of business (`"Commercial"`, `"Medicare"`, `"Medicaid"`) |
| `dt_cnt` | int | Count of valid days for this member |
| `target` | str | Day-and-code-separated target codes (training only) |

---

### Task 1: Create notebook skeleton with markdown sections and imports

**Files:**
- Create: `dev/moe/exp_round10_training_inference.ipynb`

**Step 1: Create the notebook with header, TOC, and import cells**

Create a new Jupyter notebook with these cells in order:

**Cell 0 (markdown):**
```markdown
# Experiment Round 10: Training & Inference Pipeline
## exp2b_flash_learned_pool — Full 11M 3-LOB Pre-training + Embedding Generation

**Experiment:** `exp2b_flash_learned_pool` (Flash Attention + Learned Attention Pooling, no MoE)  
**Dataset:** ~11M members across Commercial, Medicare, Medicaid  
**Embedding dim:** 256  
**Architecture:** FlashAttentionTransformer with Learned Attention Pooling, SwiGLU, RoPE

### Table of Contents
1. **Environment Setup** — Imports, device config, path setup  
2. **Configuration** — Experiment parameters, optimization config  
3. **Data Loading** — BigQuery data ingestion, train/val split  
4. **Training** — `run_single_experiment` execution  
5. **Inference** — Load trained model, generate embeddings  
6. **Save Embeddings** — Export to BigQuery

### How to Use
1. Run cells 1-3 (setup, config, data) — ~15 min for BigQuery load
2. Run cell 4 (training) — GPU time depends on hardware (~hours on 4x T4)
3. Run cells 5-6 (inference) — ~30 min for embedding generation
```

**Cell 1 (markdown):**
```markdown
## 1. Environment Setup
```

**Cell 2 (code) — System path and imports:**
```python
import sys
import os

# Add the dev/moe directory to Python path so we can import the core modules
MODULE_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), ".")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

# Standard library
import gc
import time
import json
import copy
import threading
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

# Third-party
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from google.cloud import bigquery
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

# Core module imports — model architecture, configs, data utilities
from moe_flashattn_4_core import (
    # Configs
    BaseConfig,
    FlashAttentionConfig,
    MoEConfig,
    OptimizeConfig,
    # Experiment configs
    get_experiment_configs,
    # Data parsing utilities
    conv_cd,
    conv_age_gender,
    conv_lob,
    conv_target,
    ClinicalDataset,
    create_collate_fn,
    # Model architecture
    FlashAttentionTransformer,
    FlashMoETransformer,
    BaselineTransformer,
    DataParallelWrapper,
    # Embedding extraction
    EmbeddingExtractor,
    # Utilities
    cleanup_gpu_memory,
)

# Training pipeline imports — training orchestration (defined in moe_flashattn_4.py)
from moe_flashattn_4 import (
    run_single_experiment,
    prepare_data_once,
    setup_experiment_logging,
    ClinicalDatasetLazy,
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB")
```

**Step 2: Verify the imports resolve**

Run: `python -c "import sys; sys.path.insert(0, 'dev/moe'); from moe_flashattn_4_core import BaseConfig; print('OK')"` from workspace root.
Expected: `OK`

**Step 3: Commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "feat: scaffold exp round 10 training+inference notebook with imports"
```

---

### Task 2: Add configuration cells

**Files:**
- Modify: `dev/moe/exp_round10_training_inference.ipynb`

**Step 1: Add configuration markdown + code cells**

**Cell 3 (markdown):**
```markdown
## 2. Configuration

### Experiment Parameters
- **Experiment:** `exp2b_flash_learned_pool` — FlashAttention + Learned Attention Pooling
- **Embedding size:** 256
- **Epochs:** 1 (single pass over 11M; sufficient for convergence at this scale)
- **Experiment round:** `exp_round10_3lobs_formal_training`

### Optimizer Configuration
- **Scheduler:** Linear warmup (15%) + plateau (45%) + decay
- **Loss:** BCE with log-scaled pos_weight (max=200)
- **Optimizer:** AdamW (default from BaseConfig: lr=2e-4, weight_decay=0.01)
```

**Cell 4 (code) — Experiment config:**
```python
# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Experiment selection — exp2b uses FlashAttention + Learned Attention Pooling (no MoE)
EXP_NAME = "exp2b_flash_learned_pool"
EXPERIMENT_ROUND = "exp_round10_3lobs_formal_training"
EMBEDDING_SIZE = 256
EPOCHS = 1

# Get predefined model architecture config (moe_config, use_learnt_att_pool)
all_configs = get_experiment_configs()
moe_config, use_learnt_att_pool = all_configs[EXP_NAME]

print(f"Experiment: {EXP_NAME}")
print(f"Round: {EXPERIMENT_ROUND}")
print(f"Embedding size: {EMBEDDING_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"MoE config: {moe_config}")
print(f"Learned attention pooling: {use_learnt_att_pool}")
```

**Cell 5 (code) — Optimization config:**
```python
# ============================================================================
# OPTIMIZATION CONFIGURATION
# ============================================================================

optimize_config = OptimizeConfig(
    # Scheduler: linear warmup → plateau → decay
    scheduler_type="linear",
    warmup_pct=0.15,               # First 15% of steps: linear warmup
    plateau_pct=0.45,              # Next 45% at peak LR (total 60% before decay)
    min_lr_ratio=0.2,             # Decay to 20% of peak LR

    # Loss function: BCE with frequency-based pos_weight
    use_pos_weight=True,
    pos_weight_method="log_scaled",
    pos_weight_max=200,            # Cap weight at 200 for rare codes

    # No focal loss
    use_focal_loss=False,
)

print(f"Scheduler: {optimize_config.scheduler_type}")
print(f"Warmup: {optimize_config.warmup_pct*100:.0f}% | Plateau: {optimize_config.plateau_pct*100:.0f}%")
print(f"Loss: BCE with pos_weight (method={optimize_config.pos_weight_method}, max={optimize_config.pos_weight_max})")
```

**Step 2: Verify the cells parse correctly (no syntax errors)**

Visual inspection only — these are pure config.

**Step 3: Commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "feat: add experiment and optimization configuration cells"
```

---

### Task 3: Add data loading and preparation cells

**Files:**
- Modify: `dev/moe/exp_round10_training_inference.ipynb`

**Step 1: Add data loading cells**

**Cell 6 (markdown):**
```markdown
## 3. Data Loading & Preparation

### Data Source
- **BigQuery table:** `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
- Contains ~11M members across Commercial, Medicare, Medicaid
- Each row = one member's longitudinal clinical record

### Pipeline
1. Load from BigQuery → `input_data` DataFrame
2. Deduplicate: keep only members with exactly 1 record → `df_unique`
3. Stratified train/val split (99/1) by LOB → `train_df`, `val_df`
4. Lazy dataset creation (memory-efficient) → `data_prepared_11M`

### Required Columns
| Column | Format | Example |
|--------|--------|---------|
| `individual_id` | string | `"MBR_001"` |
| `age_in_months` | `"val*val*..."` | `"360*361*362"` |
| `gender_cd` | `"val*val*..."` | `"1*1*1"` |
| `cd` | `"c1,c2*c3*..."` | `"100,200*300"` |
| `lob` | string | `"Commercial"` |
| `dt_cnt` | int | `45` |
| `target` | `"c1,c2*c3*..."` | `"50,60*70"` |
```

**Cell 7 (code) — Load training data from BigQuery:**
```python
# ============================================================================
# STEP 1: LOAD TRAINING DATA FROM BIGQUERY
# ============================================================================

client = bigquery.Client()

training_sql = """
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
"""

print("Loading training data from BigQuery...")
print(f"Table: a834793_Combined_All_LOB_o3_train_ending")
start_time = time.time()

input_data = client.query(training_sql).to_dataframe()

print(f"Loaded {len(input_data):,} rows in {time.time() - start_time:.1f}s")
print(f"Columns: {list(input_data.columns)}")
print(f"Memory usage: {input_data.memory_usage(deep=True).sum() / 1e9:.2f} GB")
```

**Cell 8 (code) — Deduplicate and split:**
```python
# ============================================================================
# STEP 2: DEDUPLICATE — keep only members with exactly 1 record
# ============================================================================

member_counts = input_data.groupby("individual_id").size()
single_record_members = member_counts[member_counts == 1].index
df_unique = input_data[input_data["individual_id"].isin(single_record_members)].copy()

del input_data
gc.collect()

print(f"Unique members (single record): {len(df_unique):,}")
print(f"LOB distribution:\n{df_unique['lob'].value_counts()}")

# ============================================================================
# STEP 3: STRATIFIED TRAIN/VAL SPLIT
# ============================================================================

TRAIN_RATIO = 0.99
RANDOM_SEED = 42

train_df, val_df = train_test_split(
    df_unique,
    train_size=TRAIN_RATIO,
    stratify=df_unique["lob"],
    random_state=RANDOM_SEED,
)

print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,}")
print(f"Train LOB distribution:\n{train_df['lob'].value_counts()}")
print(f"Val LOB distribution:\n{val_df['lob'].value_counts()}")
```

**Cell 9 (code) — Prepare lazy datasets:**
```python
# ============================================================================
# STEP 4: CREATE LAZY DATASETS + COMPUTE CODE FREQUENCIES
# ============================================================================
# Uses ClinicalDatasetLazy for memory efficiency (~130 GB vs ~888 GB for eager)

data_prepared_11M = prepare_data_once(
    train_data=train_df,
    val_data=val_df,
    device=device,
    use_lazy=True,
)

gc.collect()
print(f"\n{data_prepared_11M}")
```

**Step 2: Commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "feat: add BigQuery data loading and lazy dataset preparation cells"
```

---

### Task 4: Add training execution cell

**Files:**
- Modify: `dev/moe/exp_round10_training_inference.ipynb`

**Step 1: Add training cells**

**Cell 10 (markdown):**
```markdown
## 4. Training

Runs `run_single_experiment` which handles:
1. Model creation (FlashAttentionTransformer with learned pooling)
2. Loss function setup (BCE + pos_weight)
3. DataParallel wrapping for multi-GPU
4. Optimizer + scheduler creation
5. Training loop (train → evaluate → checkpoint per epoch)
6. Comprehensive final evaluation
7. Model saving to `logs/{EXPERIMENT_ROUND}/{EXP_NAME}/saved_models/`

**Expected outputs:**
- Training logs in `logs/{EXPERIMENT_ROUND}/{EXP_NAME}/`
- Saved model `.pt` file with checkpoint containing `model_state_dict`, `config`, `moe_config`
- Final results JSON with intrinsic metrics
```

**Cell 11 (code) — Run training:**
```python
# ============================================================================
# RUN TRAINING
# ============================================================================

cleanup_gpu_memory(verbose=True)
torch.cuda.empty_cache()

exp2b_baseline_results_11M = run_single_experiment(
    exp_name=EXP_NAME,
    moe_config=moe_config,
    use_learnt_att_pool=use_learnt_att_pool,
    prepared_data=data_prepared_11M,
    train_data=train_df,
    val_data=val_df,
    device=device,
    epochs=EPOCHS,
    experiment_round=EXPERIMENT_ROUND,
    embedding_size=EMBEDDING_SIZE,
    log_dir="logs",
    save_model=True,
    optimize_config=optimize_config,
)
```

**Cell 12 (code) — Display results:**
```python
# ============================================================================
# TRAINING RESULTS SUMMARY
# ============================================================================

print(f"{'=' * 70}")
print(f"TRAINING COMPLETE: {EXP_NAME}")
print(f"{'=' * 70}")
print(f"Model path: {exp2b_baseline_results_11M.get('model_path', 'N/A')}")
print(f"Model name: {exp2b_baseline_results_11M.get('model_name', 'N/A')}")
print(f"\nKey Metrics:")
print(f"  Final Val Loss: {exp2b_baseline_results_11M.get('best_val_loss', 'N/A'):.4f}")
print(f"  Recall@10: {exp2b_baseline_results_11M.get('recall@10', 'N/A')}")
print(f"  NDCG@20: {exp2b_baseline_results_11M.get('ndcg@20', 'N/A')}")
print(f"  Training Time: {exp2b_baseline_results_11M.get('training_time', 'N/A'):.1f}s")

# Save the model path for inference (next section)
TRAINED_MODEL_PATH = exp2b_baseline_results_11M.get("model_path")
print(f"\nModel saved to: {TRAINED_MODEL_PATH}")
print("Use this path in the Inference section below.")
```

**Step 2: Commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "feat: add training execution and results summary cells"
```

---

### Task 5: Add inference pipeline — model loading and embedding generation functions

**Files:**
- Modify: `dev/moe/exp_round10_training_inference.ipynb`

**Step 1: Add inference section with model loading and embedding generation**

**Cell 13 (markdown):**
```markdown
## 5. Inference — Embedding Generation

### Overview
After training, we load the saved model checkpoint and generate member-level embeddings.

**Embedding extraction method:**
- Uses `EmbeddingExtractor` (forward hook on `model.norm` input) to capture the final temporal representation
- Patient embedding = representation at the **last valid day** (`dt_cnt - 1`)
- Output: `[num_members, 256]` float32 array

### Functions (derived from `moe_flashattn_3_lob3_downstream_running.ipynb`)
- `load_model_from_checkpoint()` — Reconstructs model architecture from checkpoint metadata
- `LazyClinicalDatasetInference` — Memory-efficient dataset for inference (no target required)
- `generate_embeddings()` — Main entry point for batch embedding generation
- `_generate_embeddings_single_gpu()` — Optimized single-GPU path
- `_generate_embeddings_multi_gpu()` — Parallel multi-GPU path with ThreadPoolExecutor
```

**Cell 14 (code) — Model loading function:**
```python
# ============================================================================
# MODEL LOADING FROM CHECKPOINT
# ============================================================================
# Source: dev/downstream/moe_flashattn_3_lob3_downstream_running.ipynb

def load_model_from_checkpoint(
    model_path: str,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, BaseConfig, Optional[MoEConfig], bool, str]:
    """
    Load a pretrained model from a checkpoint file.

    The checkpoint .pt file contains:
        'model_state_dict', 'model_name', 'model_type', 'embedding_size',
        'nlayers', 'checkpoint_dir', 'timestamp', 'config', 'moe_config'

    Args:
        model_path: Path to the .pt checkpoint file
        device: Torch device to load the model onto
        verbose: Whether to print loading details

    Returns:
        Tuple of (model, config, moe_config, use_mixed_precision, model_type)
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Loading model from: {model_path}")

    checkpoint_data = torch.load(model_path, map_location=device, weights_only=False)

    model_type = checkpoint_data.get("model_type", "Unknown")
    config_dict = checkpoint_data.get("config", {})
    moe_config_dict = checkpoint_data.get("moe_config", None)
    state_dict = checkpoint_data["model_state_dict"]

    if verbose:
        print(f"  Model type: {model_type}")
        print(f"  Embedding size: {config_dict.get('embedding_size', 256)}")
        print(f"  N layers: {config_dict.get('nlayers', 6)}")
        print(f"  Use learned attention pooling: {config_dict.get('use_learnt_att_pool', False)}")

    use_learnt_att_pool_inferred = "daily_pooling.query" in state_dict

    # Infer d_ff from expert weight shapes for MoE models
    inferred_d_ff = None
    if "FlashMoE" in model_type:
        for key in state_dict.keys():
            if "experts.0.ffn.w_gate.weight" in key:
                weight_shape = state_dict[key].shape
                d_ff_adjusted = weight_shape[0]
                inferred_d_ff = (d_ff_adjusted * 3 + 1) // 2
                if verbose:
                    print(
                        f"Inferred d_ff from expert weights: {inferred_d_ff} "
                        f"(d_ff_adjusted={d_ff_adjusted})"
                    )
                break
        if inferred_d_ff is None:
            inferred_d_ff = config_dict.get("nhid", 512)
            if verbose:
                print(f"Using nhid as d_ff fallback: {inferred_d_ff}")

    # Reconstruct config and model based on model type
    moe_config = None

    if "FlashMoE" in model_type:
        config = FlashAttentionConfig(
            embedding_size=config_dict.get("embedding_size", 256),
            nhid=config_dict.get("nhid", 512),
            nhead=config_dict.get("nhead", 8),
            nlayers=config_dict.get("nlayers", 6),
            dropout=config_dict.get("dropout", 0.1),
            use_learnt_att_pool=use_learnt_att_pool_inferred,
            use_swiglu=config_dict.get("use_swiglu", True),
            use_rope=config_dict.get("use_rope", True),
            use_flash=config_dict.get("use_flash", True),
        )
        d_ff_to_use = inferred_d_ff or config_dict.get("nhid", 512)

        if moe_config_dict:
            if verbose and moe_config_dict.get("d_ff") != d_ff_to_use:
                print(
                    f"  Correcting d_ff: checkpoint has {moe_config_dict.get('d_ff')}, "
                    f"using {d_ff_to_use}"
                )
            moe_config = MoEConfig(
                d_model=moe_config_dict.get("d_model", config.embedding_size),
                d_ff=d_ff_to_use,
                num_experts=moe_config_dict.get("num_experts", 8),
                num_shared_experts=moe_config_dict.get("num_shared_experts", 1),
                top_k=moe_config_dict.get("top_k", 2),
                expert_dropout=moe_config_dict.get("expert_dropout", 0.1),
                load_balance_strategy=moe_config_dict.get("load_balance_strategy", "deepseek"),
                aux_loss_weight=moe_config_dict.get("aux_loss_weight", 0.001),
                use_moe_from_layer=moe_config_dict.get("use_moe_from_layer", 2),
                use_swiglu_experts=moe_config_dict.get("use_swiglu_experts", True),
                router_warmup_steps=moe_config_dict.get("router_warmup_steps", 0),
                z_loss_weight=moe_config_dict.get("z_loss_weight", 0.005),
                bias_lr=moe_config_dict.get("bias_lr", 1e-3),
                bias_momentum=moe_config_dict.get("bias_momentum", 0.6),
            )
        else:
            moe_config = MoEConfig(d_model=config.embedding_size, d_ff=config.nhid)

        model = FlashMoETransformer(config, moe_config)
        use_mixed_precision = True

    elif "FlashAttention" in model_type:
        config = FlashAttentionConfig(
            embedding_size=config_dict.get("embedding_size", 256),
            nhid=config_dict.get("nhid", 512),
            nhead=config_dict.get("nhead", 8),
            nlayers=config_dict.get("nlayers", 6),
            dropout=config_dict.get("dropout", 0.1),
            use_learnt_att_pool=use_learnt_att_pool_inferred,
            use_swiglu=config_dict.get("use_swiglu", True),
            use_rope=config_dict.get("use_rope", True),
            use_flash=config_dict.get("use_flash", True),
        )
        model = FlashAttentionTransformer(config)
        use_mixed_precision = True

    else:
        config = BaseConfig(
            embedding_size=config_dict.get("embedding_size", 256),
            nhid=config_dict.get("nhid", 512),
            nlayers=config_dict.get("nlayers", 6),
            dropout=config_dict.get("dropout", 0.1),
        )
        model = BaselineTransformer(config)
        use_mixed_precision = False

    model.load_state_dict(checkpoint_data["model_state_dict"])
    model = model.to(device)
    model.eval()

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Mixed precision: {use_mixed_precision}")
        print(f"  Device: {device}")
        print(f"{'=' * 70}\n")

    return model, config, moe_config, use_mixed_precision, model_type
```

**Cell 15 (code) — Inference dataset class:**
```python
# ============================================================================
# LAZY CLINICAL DATASET FOR INFERENCE
# ============================================================================
# Source: dev/downstream/moe_flashattn_3_lob3_downstream_running.ipynb
# Same as training ClinicalDatasetLazy but with optional target column

class LazyClinicalDatasetInference(Dataset):
    """
    Memory-efficient dataset for inference. Parses data on-the-fly.

    Identical interface to ClinicalDatasetLazy, but target column is optional.
    When target is absent, returns dummy targets (required by collate_fn).
    """

    def __init__(self, df: pd.DataFrame, config: BaseConfig):
        self.config = config
        self.df = df.reset_index(drop=True)

        self.age_strs = self.df["age_in_months"].tolist()
        self.gender_strs = self.df["gender_cd"].tolist()
        self.cd_strs = self.df["cd"].tolist()
        self.lob_strs = self.df["lob"].tolist()
        self.dt_cnt = self.df["dt_cnt"].tolist()
        self.target_strs = (
            self.df["target"].tolist() if "target" in self.df.columns else None
        )

        print(f"LazyClinicalDatasetInference: {len(self.df):,} samples (lazy loading)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        config = self.config

        age = torch.tensor(
            conv_age_gender(self.age_strs[idx], config.len_dy), dtype=torch.long
        )
        gender = torch.tensor(
            conv_age_gender(self.gender_strs[idx], config.len_dy, max_val=3),
            dtype=torch.long,
        )
        lob = torch.tensor(
            conv_lob(self.lob_strs[idx], config.len_dy), dtype=torch.long
        )
        codes = torch.tensor(
            conv_cd(self.cd_strs[idx], config.len_dy, config.len_cd), dtype=torch.long
        )

        if self.target_strs is not None:
            target_list = conv_target(
                self.target_strs[idx], config.len_dy, config.target_cd_cnt
            )
        else:
            target_list = [[0] for _ in range(config.len_dy)]

        return {
            "age": age,
            "gender": gender,
            "lob": lob,
            "codes": codes,
            "dt_cnt": self.dt_cnt[idx],
            "target": target_list,
        }
```

**Cell 16 (code) — Embedding generation functions:**
```python
# ============================================================================
# EMBEDDING GENERATION FUNCTIONS
# ============================================================================
# Source: dev/downstream/moe_flashattn_3_lob3_downstream_running.ipynb


def generate_embeddings(
    model: torch.nn.Module,
    config: BaseConfig,
    data: pd.DataFrame,
    device: torch.device,
    id_column: str = "individual_id",
    lob_value: Optional[str] = None,
    desc_prefix: str = "",
    batch_size: int = 64,
    num_workers: int = 4,
    use_mixed_precision: bool = True,
    verbose: bool = True,
    multi_gpu: bool = False,
    moe_config: Optional[MoEConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Generate member-level embeddings from a trained model.

    Extracts the final temporal representation (before decoder) at each
    member's last valid day. Works for all model types and LOBs.

    Args:
        model: Trained model in eval mode
        config: Model configuration
        data: DataFrame with clinical input columns
        device: Primary CUDA device
        id_column: Column name for member IDs (default: 'individual_id')
        lob_value: If specified, auto-adds 'lob' column when missing
        desc_prefix: Label for progress bar (e.g., 'Commercial')
        batch_size: Batch size per GPU
        num_workers: DataLoader worker count
        use_mixed_precision: Enable FP16 autocast
        verbose: Print progress
        multi_gpu: Enable multi-GPU parallel processing
        moe_config: MoE config (needed for multi-GPU with MoE models)

    Returns:
        embeddings: np.ndarray [num_members, embedding_size]
        member_ids: List of member ID strings
        index_dts: List of index date strings
    """
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size

    has_moe = hasattr(model, "forward") and "return_moe_losses" in model.forward.__code__.co_varnames
    n_gpus = torch.cuda.device_count() if multi_gpu else 1

    desc = f"{desc_prefix} " if desc_prefix else ""
    desc += "Embedding Generation"

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"{desc.upper()}")
        print(f"{'=' * 70}")
        print(f"Samples: {n_samples:,} | Batch: {batch_size} | GPUs: {n_gpus}")
        print(f"Workers: {num_workers} | Mixed precision: {use_mixed_precision}")
        print(f"ID column: {id_column}")

    if lob_value and "lob" not in data.columns:
        data = data.copy()
        data["lob"] = lob_value
        if verbose:
            print(f"  Added 'lob'='{lob_value}' column")

    embeddings_output = torch.empty(
        (n_samples, embedding_dim), dtype=torch.float32, pin_memory=True
    )

    if id_column in data.columns:
        member_ids = data[id_column].astype(str).tolist()
    else:
        member_ids = data["individual_id"].astype(str).tolist()
        if verbose:
            print(f"  Warning: '{id_column}' not found, using 'individual_id'")

    index_dts = data["index_dt"].astype(str).tolist()
    pbar_desc = f"Generating {desc_prefix} embeddings" if desc_prefix else "Generating embeddings"

    if n_gpus > 1 and multi_gpu:
        return _generate_embeddings_multi_gpu(
            model=model, config=config, data=data,
            embeddings_output=embeddings_output,
            member_ids=member_ids, index_dts=index_dts,
            n_gpus=n_gpus, batch_size=batch_size, num_workers=num_workers,
            use_mixed_precision=use_mixed_precision, has_moe=has_moe,
            moe_config=moe_config, verbose=verbose, start_time=start_time,
            pbar_desc=pbar_desc,
        )
    else:
        return _generate_embeddings_single_gpu(
            model=model, config=config, data=data, device=device,
            embeddings_output=embeddings_output,
            member_ids=member_ids, index_dts=index_dts,
            batch_size=batch_size, num_workers=num_workers,
            use_mixed_precision=use_mixed_precision, has_moe=has_moe,
            verbose=verbose, start_time=start_time, pbar_desc=pbar_desc,
        )


def _generate_embeddings_single_gpu(
    model, config, data, device, embeddings_output,
    member_ids, index_dts, batch_size, num_workers,
    use_mixed_precision, has_moe, verbose, start_time,
    pbar_desc="Generating embeddings",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Single-GPU optimized embedding extraction."""
    n_samples = len(data)
    model.eval()

    dataset = LazyClinicalDatasetInference(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    current_idx = 0
    pbar = tqdm(dataloader, desc=pbar_desc, disable=not verbose)

    with torch.inference_mode():
        with EmbeddingExtractor(model) as extractor:
            for batch in pbar:
                batch_size_actual = batch["age"].shape[0]
                batch_start = current_idx
                batch_end = batch_start + batch_size_actual

                x = torch.cat(
                    [
                        batch["age"].unsqueeze(-1),
                        batch["gender"].unsqueeze(-1),
                        batch["lob"].unsqueeze(-1),
                        batch["codes"],
                    ],
                    dim=-1,
                ).to(device, non_blocking=True)

                dt_cnt = batch["dt_cnt"]

                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if has_moe:
                            _ = model(x, return_moe_losses=False)
                        else:
                            _ = model(x)
                else:
                    if has_moe:
                        _ = model(x, return_moe_losses=False)
                    else:
                        _ = model(x)

                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)

                embeddings_output[batch_start:batch_end].copy_(
                    patient_embs.float(), non_blocking=True
                )
                current_idx = batch_end

                elapsed = time.time() - start_time
                speed = batch_end / elapsed
                eta = (n_samples - batch_end) / speed if speed > 0 else 0
                pbar.set_postfix({"speed": f"{speed:.0f}/s", "ETA": f"{eta:.0f}s"})

    if device.type == "cuda":
        torch.cuda.synchronize()

    embeddings = embeddings_output.numpy()
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nComplete! Time: {elapsed:.1f}s | Speed: {n_samples / elapsed:,.0f} samples/s")
        print(f"   Output: {embeddings.shape}")

    return embeddings, member_ids, index_dts


def _generate_embeddings_multi_gpu(
    model, config, data, embeddings_output, member_ids, index_dts,
    n_gpus, batch_size, num_workers, use_mixed_precision, has_moe,
    moe_config, verbose, start_time,
    pbar_desc="Multi-GPU",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Multi-GPU parallel embedding extraction using ThreadPoolExecutor."""
    n_samples = len(data)

    if verbose:
        print(f"Multi-GPU mode: {n_gpus} GPUs")

    models = []
    for gpu_id in range(n_gpus):
        if verbose:
            print(f"  Cloning model to GPU {gpu_id}...")
        with torch.cuda.device(gpu_id):
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to(f"cuda:{gpu_id}")
            model_copy.eval()
            models.append(model_copy)

    chunk_size = (n_samples + n_gpus - 1) // n_gpus
    data_chunks = []
    start_indices = []

    for i in range(n_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        data_chunks.append(data.iloc[start_idx:end_idx].reset_index(drop=True))
        start_indices.append(start_idx)
        if verbose:
            print(f"  GPU {i}: samples {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,})")

    progress_lock = threading.Lock()
    total_processed = [0]
    errors = []

    def process_chunk(gpu_id: int, data_chunk: pd.DataFrame, start_idx: int):
        if len(data_chunk) == 0:
            return
        try:
            gpu_device = torch.device(f"cuda:{gpu_id}")
            gpu_model = models[gpu_id]

            dataset = LazyClinicalDatasetInference(data_chunk, config)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=create_collate_fn(config),
                num_workers=max(1, num_workers // n_gpus),
                pin_memory=True,
            )

            local_idx = start_idx
            with torch.inference_mode():
                with EmbeddingExtractor(gpu_model) as extractor:
                    for batch in dataloader:
                        batch_size_actual = batch["age"].shape[0]
                        x = torch.cat(
                            [
                                batch["age"].unsqueeze(-1),
                                batch["gender"].unsqueeze(-1),
                                batch["lob"].unsqueeze(-1),
                                batch["codes"],
                            ],
                            dim=-1,
                        ).to(gpu_device, non_blocking=True)

                        dt_cnt = batch["dt_cnt"]
                        if use_mixed_precision:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                if has_moe:
                                    _ = gpu_model(x, return_moe_losses=False)
                                else:
                                    _ = gpu_model(x)
                        else:
                            if has_moe:
                                _ = gpu_model(x, return_moe_losses=False)
                            else:
                                _ = gpu_model(x)

                        dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                        patient_embs = extractor.get_patient_embedding(dt_cnt_list)

                        embeddings_output[local_idx : local_idx + batch_size_actual].copy_(
                            patient_embs.float(), non_blocking=True
                        )
                        local_idx += batch_size_actual

                        with progress_lock:
                            total_processed[0] += batch_size_actual

            torch.cuda.synchronize(gpu_device)
        except Exception as e:
            errors.append((gpu_id, str(e)))

    if verbose:
        pbar = tqdm(total=n_samples, desc=f"{pbar_desc} ({n_gpus} GPUs)")

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(process_chunk, gpu_id, data_chunks[gpu_id], start_indices[gpu_id])
            for gpu_id in range(n_gpus)
        ]
        last_count = 0
        while not all(f.done() for f in futures):
            with progress_lock:
                current = total_processed[0]
            if verbose:
                pbar.update(current - last_count)
            last_count = current
            time.sleep(0.1)

        if verbose:
            pbar.update(n_samples - last_count)
            pbar.close()

        for f in futures:
            f.result()

    if errors:
        raise RuntimeError(f"GPU errors: {errors}")

    for m in models:
        del m
    torch.cuda.empty_cache()

    embeddings = embeddings_output.numpy()
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nComplete! Time: {elapsed:.1f}s | Speed: {n_samples / elapsed:,.0f} samples/s")
        print(f"   Effective: {n_samples / elapsed * n_gpus:,.0f} samples/s across {n_gpus} GPUs")
        print(f"   Output: {embeddings.shape}")

    return embeddings, member_ids, index_dts
```

**Step 2: Commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "feat: add inference pipeline — model loading and embedding generation"
```

---

### Task 6: Add inference execution cells (commercial embedding generation)

**Files:**
- Modify: `dev/moe/exp_round10_training_inference.ipynb`

**Step 1: Add inference execution cells**

**Cell 17 (markdown):**
```markdown
### 5.1 Configure Model Path

Set the path to the trained model. If you just ran training above, `TRAINED_MODEL_PATH` is already set. Otherwise, paste the model path below.
```

**Cell 18 (code) — Model path config:**
```python
# ============================================================================
# MODEL PATH CONFIGURATION
# ============================================================================
# Option 1: Use the path from training above (if you just trained)
# TRAINED_MODEL_PATH is already set from the training results

# Option 2: Set manually if loading a previously trained model
# TRAINED_MODEL_PATH = (
#     "logs/exp_round10_3lobs_formal_training/"
#     "exp2b_flash_learned_pool/saved_models/"
#     "<model_filename>.pt"
# )

print(f"Model path: {TRAINED_MODEL_PATH}")
assert os.path.exists(TRAINED_MODEL_PATH), f"Model file not found: {TRAINED_MODEL_PATH}"
```

**Cell 19 (code) — Load model for inference:**
```python
# ============================================================================
# LOAD MODEL FOR INFERENCE
# ============================================================================

cleanup_gpu_memory(verbose=False)

model, config, moe_config_loaded, use_mixed_precision, model_type = load_model_from_checkpoint(
    model_path=TRAINED_MODEL_PATH,
    device=device,
    verbose=True,
)
```

**Cell 20 (markdown):**
```markdown
### 5.2 Load Inference Data

Load the heldout commercial dataset for embedding generation.

**Data source:** `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5`

**Sampling strategy:**
- Before 2023-10-16: 30% sample (in-time)
- After 2023-10-16: 100% (out-of-time validation)
```

**Cell 21 (code) — Load commercial data:**
```python
# ============================================================================
# LOAD COMMERCIAL HELDOUT DATA FOR INFERENCE
# ============================================================================

client = bigquery.Client()

commercial_sql = """
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5`
"""

print("Loading commercial heldout data from BigQuery...")
df_cm = client.query(commercial_sql).to_dataframe()
print(f"Loaded {len(df_cm):,} rows")

# Sample: 30% before cutoff date, 100% after (for OOT validation)
OOT_CUTOFF = "2023-10-16"
df_cm["index_dt"] = pd.to_datetime(df_cm["index_dt"])

df_cm_before = df_cm[df_cm["index_dt"] <= pd.to_datetime(OOT_CUTOFF)]
df_cm_after = df_cm[df_cm["index_dt"] > pd.to_datetime(OOT_CUTOFF)]

df_cm_before_sample = df_cm_before.sample(frac=0.3, random_state=42)

df_cm_sample = pd.concat([df_cm_before_sample, df_cm_after])

print(f"\nSampling summary:")
print(f"  Before {OOT_CUTOFF}: {len(df_cm_before):,} → {len(df_cm_before_sample):,} (30% sample)")
print(f"  After {OOT_CUTOFF}: {len(df_cm_after):,} (100%, OOT)")
print(f"  Total inference set: {len(df_cm_sample):,}")
```

**Cell 22 (code) — Generate embeddings:**
```python
# ============================================================================
# GENERATE EMBEDDINGS
# ============================================================================

INFERENCE_BATCH_SIZE = 64

embeddings, individual_ids, index_dts = generate_embeddings(
    model=model,
    config=config,
    data=df_cm_sample,
    device=device,
    id_column="individual_id",
    lob_value=None,
    desc_prefix="Commercial",
    batch_size=INFERENCE_BATCH_SIZE,
    use_mixed_precision=use_mixed_precision,
    verbose=True,
    multi_gpu=True,
    moe_config=moe_config_loaded,
)

print(f"\nEmbedding matrix shape: {embeddings.shape}")
print(f"  Members: {embeddings.shape[0]:,}")
print(f"  Dimensions: {embeddings.shape[1]}")
print(f"  dtype: {embeddings.dtype}")
```

**Step 2: Commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "feat: add inference execution cells for commercial embedding generation"
```

---

### Task 7: Add embedding export to BigQuery

**Files:**
- Modify: `dev/moe/exp_round10_training_inference.ipynb`

**Step 1: Add BigQuery export cells**

**Cell 23 (markdown):**
```markdown
## 6. Save Embeddings to BigQuery

Exports the generated embeddings to BigQuery with member IDs and metadata. Large uploads are automatically chunked to avoid BigQuery memory limits.
```

**Cell 24 (code) — BigQuery export function:**
```python
# ============================================================================
# SAVE EMBEDDINGS TO BIGQUERY
# ============================================================================
# Source: dev/downstream/moe_flashattn_3_lob3_downstream_running.ipynb

def save_embeddings_to_bigquery(
    embeddings: np.ndarray,
    individual_ids: list,
    index_dts: list,
    project_id: str,
    dataset_id: str,
    table_name: str,
    exp_name: str = "",
    model_type: str = "",
    if_exists: str = "replace",
    max_bytes_per_chunk: int = 500_000_000,
) -> str:
    """
    Save embeddings to BigQuery with automatic chunking for large uploads.

    Creates a table with columns:
        individual_id, index_dt, embedding_0..embedding_{dim-1}, exp_name, model_type

    Args:
        embeddings: numpy array [num_members, embedding_dim]
        individual_ids: List of member IDs
        index_dts: List of index dates
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_name: Target table name
        exp_name: Experiment name metadata
        model_type: Model type metadata
        if_exists: 'replace', 'append', or 'fail'
        max_bytes_per_chunk: Max bytes per upload chunk (~500 MB default)

    Returns:
        Full BigQuery table path
    """
    from google.cloud import bigquery as bq

    n_total = len(individual_ids)
    embedding_dim = embeddings.shape[1]
    full_table_id = f"{project_id}.{dataset_id}.{table_name}"

    bytes_per_row = embedding_dim * 4 + 200
    estimated_total_bytes = bytes_per_row * n_total
    chunk_size = max(1, max_bytes_per_chunk // bytes_per_row)
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    print(f"Writing {n_total:,} rows to BigQuery: {full_table_id}")
    print(f"  Columns: {embedding_dim + 4} (embedding_dim={embedding_dim})")
    print(f"  Estimated payload: {estimated_total_bytes / 1e9:.2f} GB")
    if n_chunks > 1:
        print(f"  Chunking into {n_chunks} uploads of ~{chunk_size:,} rows each")

    bq_client = bq.Client()

    first_disposition = {
        "replace": bq.WriteDisposition.WRITE_TRUNCATE,
        "append": bq.WriteDisposition.WRITE_APPEND,
        "fail": bq.WriteDisposition.WRITE_EMPTY,
    }[if_exists]

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_total)

        df_chunk = pd.DataFrame(
            {"individual_id": individual_ids[start:end], "index_dt": index_dts[start:end]}
        )
        for i in range(embedding_dim):
            df_chunk[f"embedding_{i}"] = embeddings[start:end, i].astype(np.float32)
        df_chunk["exp_name"] = exp_name
        df_chunk["model_type"] = model_type

        write_disp = first_disposition if chunk_idx == 0 else bq.WriteDisposition.WRITE_APPEND
        job_config = bq.LoadJobConfig(write_disposition=write_disp)
        job = bq_client.load_table_from_dataframe(df_chunk, full_table_id, job_config=job_config)
        job.result()

        print(f"  Chunk {chunk_idx + 1}/{n_chunks}: rows [{start:,} - {end:,}) uploaded")
        del df_chunk

    table = bq_client.get_table(full_table_id)
    print(f"Loaded {table.num_rows:,} rows to {full_table_id}")
    return full_table_id
```

**Cell 25 (code) — Execute export:**
```python
# ============================================================================
# EXPORT EMBEDDINGS
# ============================================================================

PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"
TABLE_NAME = f"a964286_exp_round10_exp2b_commercial_embeddings"

full_table_path = save_embeddings_to_bigquery(
    embeddings=embeddings,
    individual_ids=individual_ids,
    index_dts=index_dts,
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_name=TABLE_NAME,
    exp_name=EXP_NAME,
    model_type=model_type,
    if_exists="replace",
)

print(f"\nEmbeddings saved to: {full_table_path}")
```

**Cell 26 (code) — Cleanup:**
```python
# ============================================================================
# CLEANUP
# ============================================================================

del model, embeddings, df_cm, df_cm_sample
gc.collect()
cleanup_gpu_memory(verbose=True)
print("Done. All resources released.")
```

**Step 2: Commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "feat: add BigQuery embedding export and cleanup cells"
```

---

### Task 8: Add unit tests for training and inference integration

**Files:**
- Modify: `dev/moe/exp_round10_training_inference.ipynb` (add test cells at the end)

**Step 1: Add unit test section**

**Cell 27 (markdown):**
```markdown
## 7. Unit Tests

Validates the training and inference pipeline with synthetic data.
These tests run locally without BigQuery access.

**Tests:**
1. `test_config_setup` — Config classes instantiate correctly
2. `test_lazy_dataset` — LazyClinicalDatasetInference parses data correctly
3. `test_collate_fn` — Collate function produces correct batch shapes
4. `test_model_forward_pass` — Model forward pass produces correct output shape
5. `test_embedding_extractor` — EmbeddingExtractor captures correct embeddings
6. `test_generate_embeddings_e2e` — End-to-end embedding generation on synthetic data
7. `test_load_model_roundtrip` — Save → load → verify model weights match
```

**Cell 28 (code) — Synthetic data factory:**
```python
# ============================================================================
# TEST UTILITIES — SYNTHETIC DATA FACTORY
# ============================================================================

def create_synthetic_data(n_samples: int = 20, n_days: int = 5) -> pd.DataFrame:
    """
    Create synthetic clinical data matching the expected DataFrame schema.
    All string fields use the same '*'-separated day format as production data.
    """
    np.random.seed(42)
    rows = []
    for i in range(n_samples):
        dt_cnt = np.random.randint(1, n_days + 1)
        age_vals = [str(np.random.randint(120, 960)) for _ in range(dt_cnt)]
        gender_vals = [str(np.random.randint(1, 3)) for _ in range(dt_cnt)]
        lob_choices = ["Commercial", "Medicare", "Medicaid"]
        lob_val = np.random.choice(lob_choices)

        cd_days = []
        for _ in range(dt_cnt):
            n_codes = np.random.randint(1, 10)
            day_codes = [str(np.random.randint(1, 1000)) for _ in range(n_codes)]
            cd_days.append(",".join(day_codes))

        target_days = []
        for _ in range(dt_cnt):
            n_targets = np.random.randint(0, 4)
            if n_targets > 0:
                day_targets = [str(np.random.randint(1, 500)) for _ in range(n_targets)]
                target_days.append(",".join(day_targets))
            else:
                target_days.append("")

        rows.append({
            "individual_id": f"MBR_{i:04d}",
            "index_dt": f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
            "age_in_months": "*".join(age_vals),
            "gender_cd": "*".join(gender_vals),
            "cd": "*".join(cd_days),
            "lob": lob_val,
            "dt_cnt": dt_cnt,
            "target": "*".join(target_days),
        })

    return pd.DataFrame(rows)


# Quick validation
_test_df = create_synthetic_data(5)
print("Synthetic data schema:")
print(_test_df.dtypes)
print(f"\nSample row:\n{_test_df.iloc[0].to_dict()}")
del _test_df
```

**Cell 29 (code) — Unit tests:**
```python
# ============================================================================
# UNIT TESTS
# ============================================================================

def test_config_setup():
    """Test that configuration classes instantiate with correct defaults."""
    base = BaseConfig()
    assert base.len_dy == 200
    assert base.len_cd == 80
    assert base.embedding_size == 256
    assert base.target_cd_cnt == 6297

    flash = FlashAttentionConfig()
    assert flash.use_flash is True
    assert flash.use_rope is True
    assert flash.use_swiglu is True

    opt = OptimizeConfig(scheduler_type="linear", pos_weight_max=200)
    assert opt.scheduler_type == "linear"
    assert opt.pos_weight_max == 200

    configs = get_experiment_configs()
    assert "exp2b_flash_learned_pool" in configs
    moe_cfg, use_pool = configs["exp2b_flash_learned_pool"]
    assert moe_cfg is None
    assert use_pool is True

    print("test_config_setup PASSED")


def test_lazy_dataset():
    """Test LazyClinicalDatasetInference creates correct tensors."""
    config = BaseConfig(len_dy=10, len_cd=5, target_cd_cnt=500)
    df = create_synthetic_data(n_samples=8, n_days=5)

    dataset = LazyClinicalDatasetInference(df, config)
    assert len(dataset) == 8

    item = dataset[0]
    assert item["age"].shape == (10,)
    assert item["gender"].shape == (10,)
    assert item["lob"].shape == (10,)
    assert item["codes"].shape == (10, 5)
    assert isinstance(item["dt_cnt"], (int, np.integer))

    # Test without target column
    df_no_target = df.drop(columns=["target"])
    dataset_no_target = LazyClinicalDatasetInference(df_no_target, config)
    item_no_target = dataset_no_target[0]
    assert len(item_no_target["target"]) == 10

    print("test_lazy_dataset PASSED")


def test_collate_fn():
    """Test collate function produces correct batch shapes."""
    config = BaseConfig(len_dy=10, len_cd=5, target_cd_cnt=500)
    df = create_synthetic_data(n_samples=8, n_days=5)

    dataset = LazyClinicalDatasetInference(df, config)
    collate_fn = create_collate_fn(config)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(loader))
    assert batch["age"].shape == (4, 10), f"age shape: {batch['age'].shape}"
    assert batch["gender"].shape == (4, 10)
    assert batch["lob"].shape == (4, 10)
    assert batch["codes"].shape == (4, 10, 5)
    assert batch["dt_cnt"].shape == (4,)
    assert batch["target_multihot"].shape == (4, 10, 500)

    print("test_collate_fn PASSED")


def test_model_forward_pass():
    """Test model forward pass with small config."""
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FlashAttentionConfig(
        len_dy=10, len_cd=5, embedding_size=32, nhid=64, nhead=4, nlayers=2,
        target_cd_cnt=500, cd_cnt=1000,
        use_learnt_att_pool=True, use_swiglu=True, use_rope=True, use_flash=True,
    )
    model = FlashAttentionTransformer(config).to(test_device)
    model.eval()

    batch_size = 4
    x = torch.randint(0, 100, (batch_size, 10, 5 + 3)).float().to(test_device)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (batch_size, 10, 500), f"Output shape: {output.shape}"
    print(f"test_model_forward_pass PASSED — output shape: {output.shape}")

    del model
    if test_device.type == "cuda":
        torch.cuda.empty_cache()


def test_embedding_extractor():
    """Test EmbeddingExtractor captures embeddings at correct dimensions."""
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FlashAttentionConfig(
        len_dy=10, len_cd=5, embedding_size=32, nhid=64, nhead=4, nlayers=2,
        target_cd_cnt=500, cd_cnt=1000,
        use_learnt_att_pool=True, use_swiglu=True, use_rope=True, use_flash=True,
    )
    model = FlashAttentionTransformer(config).to(test_device)
    model.eval()

    batch_size = 4
    x = torch.randint(0, 100, (batch_size, 10, 5 + 3)).float().to(test_device)
    dt_cnt = [3, 5, 2, 7]

    with torch.no_grad():
        with EmbeddingExtractor(model) as extractor:
            _ = model(x)
            patient_embs = extractor.get_patient_embedding(dt_cnt)

    assert patient_embs.shape == (4, 32), f"Embedding shape: {patient_embs.shape}"

    for i, cnt in enumerate(dt_cnt):
        raw = extractor.get_embeddings()
        expected_day_idx = min(cnt - 1, 9)
        assert expected_day_idx >= 0

    print(f"test_embedding_extractor PASSED — embedding shape: {patient_embs.shape}")

    del model
    if test_device.type == "cuda":
        torch.cuda.empty_cache()


def test_generate_embeddings_e2e():
    """End-to-end test: synthetic data → generate_embeddings → verify output."""
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FlashAttentionConfig(
        len_dy=10, len_cd=5, embedding_size=32, nhid=64, nhead=4, nlayers=2,
        target_cd_cnt=500, cd_cnt=1000,
        use_learnt_att_pool=True, use_swiglu=True, use_rope=True, use_flash=True,
    )
    model = FlashAttentionTransformer(config).to(test_device)
    model.eval()

    df = create_synthetic_data(n_samples=12, n_days=5)

    embeddings, member_ids, index_dts = generate_embeddings(
        model=model,
        config=config,
        data=df,
        device=test_device,
        id_column="individual_id",
        batch_size=4,
        num_workers=0,
        use_mixed_precision=False,
        verbose=False,
        multi_gpu=False,
    )

    assert embeddings.shape == (12, 32), f"Embedding shape: {embeddings.shape}"
    assert len(member_ids) == 12
    assert len(index_dts) == 12
    assert not np.isnan(embeddings).any(), "NaN in embeddings"
    assert not np.isinf(embeddings).any(), "Inf in embeddings"

    print(f"test_generate_embeddings_e2e PASSED — {embeddings.shape}")

    del model
    if test_device.type == "cuda":
        torch.cuda.empty_cache()


def test_load_model_roundtrip():
    """Test save → load → verify model weights match."""
    import tempfile

    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FlashAttentionConfig(
        len_dy=10, len_cd=5, embedding_size=32, nhid=64, nhead=4, nlayers=2,
        target_cd_cnt=500, cd_cnt=1000,
        use_learnt_att_pool=True, use_swiglu=True, use_rope=True, use_flash=True,
    )
    model = FlashAttentionTransformer(config).to(test_device)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": "FlashAttentionTransformer",
                "config": {
                    "embedding_size": 32, "nhid": 64, "nhead": 4, "nlayers": 2,
                    "dropout": 0.1, "use_learnt_att_pool": True,
                    "use_swiglu": True, "use_rope": True, "use_flash": True,
                    "len_dy": 10, "len_cd": 5, "target_cd_cnt": 500, "cd_cnt": 1000,
                },
                "moe_config": None,
            },
            save_path,
        )

        loaded_model, loaded_config, _, use_mp, model_type = load_model_from_checkpoint(
            save_path, test_device, verbose=False
        )

        assert model_type == "FlashAttentionTransformer"
        assert loaded_config.embedding_size == 32
        assert use_mp is True

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.state_dict().items(), loaded_model.state_dict().items()
        ):
            assert n1 == n2, f"Key mismatch: {n1} vs {n2}"
            assert torch.equal(p1.cpu(), p2.cpu()), f"Weight mismatch for {n1}"

    print("test_load_model_roundtrip PASSED")

    del model, loaded_model
    if test_device.type == "cuda":
        torch.cuda.empty_cache()
```

**Cell 30 (code) — Run all tests:**
```python
# ============================================================================
# RUN ALL TESTS
# ============================================================================

print("=" * 70)
print("RUNNING UNIT TESTS")
print("=" * 70)

tests = [
    test_config_setup,
    test_lazy_dataset,
    test_collate_fn,
    test_model_forward_pass,
    test_embedding_extractor,
    test_generate_embeddings_e2e,
    test_load_model_roundtrip,
]

passed = 0
failed = 0
for test_fn in tests:
    try:
        test_fn()
        passed += 1
    except Exception as e:
        print(f"  FAILED: {test_fn.__name__} — {e}")
        import traceback
        traceback.print_exc()
        failed += 1

print(f"\n{'=' * 70}")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
print(f"{'=' * 70}")

assert failed == 0, f"{failed} test(s) failed!"
```

**Step 2: Verify tests pass by running the test runner cell**

Expected: All 7 tests pass.

**Step 3: Commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "feat: add unit tests for training and inference integration"
```

---

### Task 9: Critical review and final verification

**Files:**
- Modify: `dev/moe/exp_round10_training_inference.ipynb` (fix any issues found)

**Step 1: Review checklist**

Verify each of these criteria:

1. **Import completeness** — Every symbol used in the notebook is either imported or defined locally
2. **No new functionality** — All code is derived from existing modules; no novel algorithms
3. **Config correctness** — `EXP_NAME`, `EXPERIMENT_ROUND`, `EMBEDDING_SIZE`, `EPOCHS`, `optimize_config` match the original notebook values
4. **Data pipeline** — BigQuery SQL matches original, dedup logic matches, split ratio (0.99) and seed (42) match
5. **Training call** — `run_single_experiment` arguments match the original call exactly
6. **Inference pipeline** — `load_model_from_checkpoint` handles all 3 model types (Baseline, FlashAttention, FlashMoE)
7. **Embedding extraction** — Uses `EmbeddingExtractor` hook on `model.norm` input, gets last valid day via `dt_cnt - 1`
8. **Multi-GPU support** — Both training (via DataParallel in `run_single_experiment`) and inference (`_generate_embeddings_multi_gpu`) work
9. **Memory efficiency** — Lazy datasets used, proper cleanup with `gc.collect()` and `torch.cuda.empty_cache()`
10. **No hardcoded paths** — Model path flows from training output to inference input

**Step 2: Fix any issues found in review**

Apply fixes to the notebook.

**Step 3: Final commit**

```bash
git add dev/moe/exp_round10_training_inference.ipynb
git commit -m "chore: critical review and final verification of training+inference notebook"
```

---

## Execution Order

| Task | Description | Estimated Time |
|------|------------|---------------|
| 1 | Notebook skeleton + imports | 3 min |
| 2 | Configuration cells | 2 min |
| 3 | Data loading cells | 3 min |
| 4 | Training execution cells | 2 min |
| 5 | Inference functions (model load + embedding gen) | 5 min |
| 6 | Inference execution (commercial) | 3 min |
| 7 | BigQuery export | 3 min |
| 8 | Unit tests | 5 min |
| 9 | Critical review | 5 min |
