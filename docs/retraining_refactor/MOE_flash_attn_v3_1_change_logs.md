### Mixture-of-Experts (MoE) Experimentation Framework for Hierarchical Clinical Transformer

#### Version History

##### Version 1 & 2 (`moe_flashattn_2.py`)
- Initial implementation of 5-experiment MoE ablation study
- Base framework with Flash Attention and MoE integration
- Single GPU training support
- Pre-training focused evaluation metrics

##### Version 3.0 (`moe_flashattn_3.py` - Initial Release)
- **Distributed Data Parallel (DDP) support** for multi-GPU training
- **Line of Business (LOB) feature** added as input embedding
- **Medicaid IP Risk downstream task evaluation** using linear probe methodology
- **Refactored experiment running** with modular helper functions
- **Model saving/loading utilities** for inference and deployment

##### Version 3.1 (`moe_flashattn_3.py` - Current)
**New Features:**
- **Multi-LOB downstream evaluation** - Run downstream evaluation across Commercial, Medicare, Medicaid in one pipeline
- **XGBoost/LightGBM probe classifiers** - Gradient boosting alternatives to logistic regression
- **Probability calibration** - Isotonic regression calibration for better probability estimates
- **Top-percentile metrics** - Lift@K%, Precision@K%, Recall@K%, F1@K%, etc.
- **Standalone downstream evaluation** - Run downstream eval from saved model without retraining
- **DataParallelWrapper** - Efficient multi-GPU training with distributed loss computation

**GPU & Memory Optimizations:**
- **DataLoader workers limit** - Reduced from 32 to 4-8 to prevent kernel death from memory exhaustion
- **Gradient accumulation configuration** - Proper `accumulation_steps` parameter with default=1 for DataParallel
- **Learning rate scaling** - Linear scaling with `num_gpus` (was sqrt) for proper multi-GPU convergence
- **Scheduler T_max fix** - Correct total steps calculation for CosineAnnealingLR
- **Enhanced collate function** - Pre-computes multi-hot targets as tensors for DataParallel efficiency
- **Persistent workers** - `persistent_workers=True` for reduced DataLoader overhead
- **Memory leak prevention** - Explicit tensor deletion and gc.collect() in training loop

**Intrinsic Metrics Enhancements:**
- **MRR bug fix** - Uses best-ranked true code (was: arbitrary first code in list)
- **Micro-Recall@K** - Per-code hit rate: `sum(hits) / sum(true_labels)` across all samples
- **NDCG@K** - Normalized Discounted Cumulative Gain for ranking quality
- **Positive-Only Brier** - Calibration metric focused on positive labels (not dominated by TNs)
- **Macro AUROC/AUPRC** - Threshold-agnostic discriminative metrics
- **Renamed top_K_acc → recall@K** - Consistent naming throughout codebase
- **Removed mAP@K** - Problematic metric that excluded samples with no hits

---

#### Experiment Overview

| Experiment | Model | Head Config | Activation | Load Balance | Precision | Daily Encoder |
|------------|-------|-------------|------------|--------------|-----------|---------------|
| **Exp 1: Dense Baseline** | BaselineTransformer | nhead=16, head_dim=16 | GELU only | N/A | FP32 | Standard transformer |
| **Exp 2: Dense Flash** | FlashAttentionTransformer | nhead=8, head_dim=32 | SwiGLU | N/A | FP16 | Flash Attention |
| **Exp 3: Standard Top-K MoE** | FlashMoETransformer | nhead=8, head_dim=32 | SwiGLU + GELU experts | Switch | FP16 | Flash Attention |
| **Exp 4: Shared Expert MoE** | FlashMoETransformer | nhead=8, head_dim=32 | SwiGLU + GELU experts | Switch | FP16 | Flash Attention |
| **Exp 5: Fine-Grained MoE** | FlashMoETransformer | nhead=8, head_dim=32 | SwiGLU + GELU experts | Switch | FP16 | Flash Attention |
| **Exp 6: Auxiliary-Free MoE** | FlashMoETransformer | nhead=8, head_dim=32 | SwiGLU + GELU experts | DeepSeek | FP16 | Flash Attention |

#### Experiment Variants

| Experiment Name | Type | Key Features | Rationale|
|----------------|------|--------------|-----------------|
| **Baselines** ||||
| `exp1_dense_baseline` | Dense | Standard Transformer, FP32 | Reference baseline |
| `exp2_dense_flash` | Dense | Flash Attention, Max-Pool | Flash attention baseline |
| `exp2b_flash_learned_pool` | Dense | Flash Attention, Learned Pooling | Best dense model |
| **Standard MoE** ||||
| `exp3_standard_moe` | MoE | 8 experts, top-2, GELU | Basic MoE test |
| `exp3a_moe_swiglu` | MoE | 8 experts, top-2, SwiGLU | SwiGLU vs GELU |
| `exp3b_moe_swiglu_learned_pool` | MoE | + Learned pooling | Best MoE variant |
| `exp3c_moe_swiglu_learned_pool_layer4` | MoE | MoE from layer 4 | Later MoE layers |
| `exp3d_moe_swiglu_learned_pool_layer4_aux001` | MoE | aux_loss=0.001, layer 4 | Lower aux loss |
| `exp3e_moe_swiglu_learned_pool_layer2_aux001` | MoE | aux_loss=0.001, layer 2 | Lower aux loss |
| **Shared Expert MoE** ||||
| `exp4_shared_expert` | MoE | 1 shared + 7 routed | Shared expert test |
| **Fine-grained MoE** ||||
| `exp5_fine_grained` | MoE | 16 experts, top-5, smaller | Fine-grained routing |
| **Auxiliary-free MoE (DeepSeek)** ||||
| `exp6_auxiliary_free` | MoE | DeepSeek balancing, no aux loss | Aux-free MoE |
| `exp6a_auxiliary_free_layer4` | MoE | DeepSeek, MoE from layer 4 | Later DeepSeek |
| `exp6b_auxiliary_free_no-share-exp` | MoE | DeepSeek, no shared experts | Pure DeepSeek |

---

## Version 3.0 Modifications (Initial Release)

### 1. Line of Business (LOB) Feature Support

**Configuration** (`BaseConfig`):
```python
lob_vocab: int = 4  # LOB categories (0=padding, 1=Commercial, 2=Medicare, 3=Medicaid)
```

**Data Processing**:
- New `conv_lob()` function maps LOB strings to indices
- `ClinicalDataset` now processes LOB column alongside age, gender, codes
- Input tensor dimension: **83** (age, gender, lob, 80 codes) vs **82** in v2

**Model Architecture**:
- All models (`BaselineTransformer`, `FlashAttentionTransformer`, `FlashMoETransformer`) now include:
  ```python
  self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)
  ```
- LOB embedding added to combined representation:
  ```python
  cd = cd_res + cd + gender_cd + age_in_months + lob_emb
  ```

---

### 2a. Data Parallelism (Active for Experimentation)

Automatically enabled when `torch.cuda.device_count() > 1`:

```python
num_gpus = torch.cuda.device_count()
use_data_parallel = num_gpus > 1

if use_data_parallel:
    model = nn.DataParallel(model)
```

**Features**:
| Feature | Implementation |
|---------|----------------|
| Auto-detection | Enables when multiple GPUs available |
| Batch scaling | Effective batch = `batch_size * num_gpus` |
| Learning rate scaling | Square root scaling: `lr * sqrt(num_gpus)` |
| Checkpoint handling | Unwraps `model.module` for compatible saves |

**Scaling Example** (4 GPUs):
- Per-GPU batch size: 32
- Effective batch size: 128
- Base LR: 1e-4 → Scaled LR: 2e-4

**Helper Functions**:
| Function | Purpose |
|----------|---------|
| `save_checkpoint_multigpu()` | Save checkpoint compatible with DataParallel wrapper |
| `load_checkpoint_multigpu()` | Load checkpoint into wrapped or unwrapped model |
| `monitor_gpu_memory_usage()` | Track per-GPU memory for DataParallel |

---

### 2b. Core DDP Functions (Infrastructure Ready, Not Actively Used)

| Function | Description |
|----------|-------------|
| `setup_ddp()` | Initialize DDP, returns (local_rank, world_size, is_main) |
| `cleanup_ddp()` | Clean up distributed process group |
| `is_dist_initialized()` | Check if DDP is initialized |
| `get_world_size()` | Get number of processes (1 if single GPU) |
| `get_rank()` | Get current process rank |
| `is_main_process()` | Check if rank 0 |
| `reduce_tensor()` | Reduce tensor across all processes |
| `sync_metrics()` | Synchronize metrics across processes |

**Utility Functions**:
| Function | Description |
|----------|-------------|
| `print_rank()` | Print message with rank prefix |
| `print_main()` | Print only on main process |
| `barrier_with_timeout()` | Barrier with timeout for hang detection |

**Training Integration**:
- `train_epoch()` now accepts `is_main` and `use_ddp` parameters
- `run_single_experiment()` accepts `local_rank` and `world_size`
- DataLoader worker count scales with world size

---

### 3. Downstream Task Evaluation (Initial - Medicaid IP Risk)

**Configuration** (`DownstreamConfig` v3.0):
```python
@dataclass
class DownstreamConfig:
    task_name: str = "medicaid_ip_risk"
    test_size: float = 0.2          # 20% for test
    val_size: float = 0.1           # 10% for validation
    random_state: int = 42
    n_cv_folds: int = 5             # Cross-validation folds
    max_iter: int = 1000            # Max LogReg iterations
    class_weight: str = 'balanced'  # Handle class imbalance
```

**DownstreamEvaluator Class (v3.0)**:

| Method | Description |
|--------|-------------|
| `extract_embeddings()` | Extract member-level embeddings from trained transformer |
| `prepare_downstream_data()` | Join features with outcomes, create stratified splits |
| `train_linear_probe()` | Train logistic regression on frozen embeddings |
| `evaluate_probe()` | Compute comprehensive metrics on a data split |
| `evaluate()` | Full pipeline: extract → prepare → train → evaluate |

**Evaluation Metrics (v3.0)**:
- Standard: Accuracy, AUC-ROC, AUC-PR, F1, Precision, Recall, Brier Score
- Dataset stats: Prevalence, sample counts

---

### 4. Model Saving/Loading Utilities (v3.0)

**Functions**:
| Function | Purpose |
|----------|---------|
| `generate_model_name()` | Standardized naming: `{round}_{exp}_bs{batch}_ep{epochs}_d{embedding}_{timestamp}` |
| `save_trained_model()` | Lightweight save for inference (state dict + config + results) |
| `load_trained_model()` | Load model for inference or downstream evaluation |
| `run_downstream_evaluation()` | Convenience wrapper for downstream evaluation |

---

### 5. Refactored Experiment Running

**Helper Functions**:
| Function | Purpose |
|----------|---------|
| `_setup_experiment_directories()` | Set up logging and checkpoint directories |
| `_create_model()` | Create model based on experiment type |
| `_create_dataloaders()` | Create train/val dataloaders with optional bucketing |
| `_resume_from_checkpoint()` | Resume training from checkpoint |
| `_build_epoch_metrics()` | Build comprehensive epoch metrics dictionary |
| `_build_final_results()` | Build final experiment results dictionary |
| `_model_has_moe()` | Check if model has MoE layers |

---

## Version 3.1 Modifications (Current Enhancements)

### 6. Enhanced Downstream Evaluation Configuration

**Updated `DownstreamConfig` (v3.1)**:
```python
@dataclass
class DownstreamConfig:
    task_name: str = "medicaid_ip_risk"
    test_size: float = 0.1                    # CHANGED: 10% for test (was 20%)
    val_size: float = 0.1                     # 10% for validation
    random_state: int = 42
    percentiles: List[float] = [0.1, 0.01]    # NEW: Top-K percentile thresholds
    n_cv_folds: int = 5
    max_iter: int = 1000
    class_weight: str = 'balanced'
    model_type: str = 'xgboost'               # NEW: 'logistic', 'xgboost', 'lightgbm'
    calibrate_proba: bool = True              # NEW: Probability calibration
    lob_name: Optional[str] = None            # NEW: LOB-specific evaluation
    outcome_column: str = 'acute_ip_flag'     # NEW: Configurable outcome column
```

**New Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `percentiles` | List[float] | Top-K thresholds for percentile metrics (e.g., [0.1, 0.01] = 10%, 1%) |
| `model_type` | str | Classifier type: 'logistic', 'xgboost', 'lightgbm' |
| `calibrate_proba` | bool | Whether to calibrate probabilities using isotonic regression |
| `lob_name` | str | LOB name for multi-LOB evaluation |
| `outcome_column` | str | Name of outcome column in outcomes_df |

---

### 7. Gradient Boosting Probe Classifiers

**New Methods in `DownstreamEvaluator`**:

```python
def train_xgboost_probe(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    downstream_config: DownstreamConfig
) -> Tuple[xgb.XGBClassifier, StandardScaler, CalibratedClassifierCV]:
    """Train XGBoost classifier with class imbalance handling and calibration."""

def train_lightgbm_probe(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    downstream_config: DownstreamConfig
) -> Tuple[lgb.LGBMClassifier, StandardScaler, CalibratedClassifierCV]:
    """Train LightGBM classifier with class imbalance handling and calibration."""
```

**Features**:
- Automatic `scale_pos_weight` computation for class imbalance
- Isotonic regression probability calibration using validation set
- Default hyperparameters optimized for embedding-based classification

---

### 8. Top-Percentile Metrics

**New Metrics in `evaluate_probe()`**:

| Metric | Description | Formula |
|--------|-------------|---------|
| `lift_{K}pct` | Lift at top K% | (TP@K% / N@K%) / prevalence |
| `true_positives_{K}pct` | Count of TPs in top K% | TP@K% |
| `n_samples_{K}pct` | Count of samples in top K% | N@K% |
| `precision_{K}pct` | Precision at top K% | TP@K% / N@K% |
| `recall_{K}pct` | Recall at top K% | TP@K% / total_positives |
| `f1_{K}pct` | F1 score at top K% | 2 * (P@K * R@K) / (P@K + R@K) |
| `roc_auc_{K}pct` | AUC-ROC at top K% | Binary classification AUC |
| `pr_auc_{K}pct` | PR-AUC at top K% | Precision-Recall AUC |
| `specificity_{K}pct` | Specificity at top K% | TN / (TN + FP) |

**Utility Functions** (from `utils.metrics`):
```python
from utils.metrics import (
    lift_at_percentage, 
    true_positives_at_percentage, 
    num_samples_at_percentage,
    precision_at_percentage, 
    recall_at_percentage, 
    f1_at_percentage, 
    pr_auc_at_percentage, 
    roc_auc_at_percentage,
    specificity_at_percentage
)
```

---

### 9. Multi-LOB Downstream Evaluation

**New Classes and Functions**:

```python
@dataclass
class LOBData:
    """Container for LOB-specific data."""
    lob_name: str
    features_df: pd.DataFrame
    outcomes_df: pd.DataFrame
    downstream_config: Optional[DownstreamConfig] = None

def run_multi_lob_downstream_evaluation(
    model_path: str,
    lob_data_list: List[LOBData],
    device: torch.device,
    base_downstream_config: Optional[DownstreamConfig] = None,
    log_dir: str = "logs/downstream",
) -> Dict[str, Dict[str, Any]]:
    """Run downstream evaluation for multiple LOBs using a single pretrained model."""
```

**Usage Example**:
```python
lob_data_list = [
    LOBData(lob_name='commercial', features_df=com_features, outcomes_df=com_outcomes),
    LOBData(lob_name='medicare', features_df=mcr_features, outcomes_df=mcr_outcomes),
    LOBData(lob_name='medicaid', features_df=mcd_features, outcomes_df=mcd_outcomes),
]

results = run_multi_lob_downstream_evaluation(
    model_path='logs/round5/exp3_moe/saved_models/model_final.pt',
    lob_data_list=lob_data_list,
    device=device,
    base_downstream_config=DownstreamConfig(model_type='xgboost')
)
```

---

### 10. Standalone Downstream Evaluation from Saved Model

**New Function**:
```python
def run_downstream_evaluation_from_saved_model(
    model_path: str,
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    device: torch.device,
    downstream_config: Optional[DownstreamConfig] = None,
    log_dir: str = "logs/downstream",
) -> Dict[str, Any]:
    """
    Run downstream evaluation using a pretrained model loaded from disk.
    
    This is the standalone downstream evaluation pipeline that can be run
    independently of pretraining.
    """
```

**Features**:
- Automatically reconstructs model architecture from saved checkpoint
- Supports FlashAttentionTransformer and FlashMoETransformer
- Saves results to JSON for comparison

---

### 11. DataParallelWrapper Class

**Purpose**: Efficient multi-GPU training with distributed loss computation.

```python
class DataParallelWrapper(nn.Module):
    """
    Wrapper that integrates loss computation into the forward pass.
    
    Standard DataParallel gathers outputs to GPU 0, then loss runs on GPU 0 only.
    This wrapper computes loss on EACH GPU, then DataParallel averages the losses.
    
    RESULT:
    - GPU 0 no longer bottlenecked by loss computation
    - All GPUs contribute equally to training
    - ~3-4x speedup with 4 GPUs
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: 'BaseConfig', 
        criterion: nn.Module,
        moe_config: Optional['MoEConfig'] = None
    ):
        ...
    
    def forward(
        self, 
        x: torch.Tensor,           # [batch, len_dy, features]
        dt_cnt: torch.Tensor,      # [batch] - valid days per sample
        targets: torch.Tensor,     # [batch, len_dy, target_cd_cnt] multi-hot
        return_predictions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass with integrated loss computation."""
        ...
```

**Benefits**:
| Aspect | Without Wrapper | With DataParallelWrapper |
|--------|-----------------|--------------------------|
| Loss computation | GPU 0 only | All GPUs |
| GPU 0 utilization | High (bottleneck) | Balanced |
| Throughput | Limited by GPU 0 | Near-linear scaling |

---

### 12. Enhanced Collate Function

**Version 3.1 Changes**:
```python
def clinical_collate_fn(batch: List[Dict], config: 'BaseConfig') -> Dict[str, Any]:
    """
    Enhanced collate function that pre-computes multi-hot targets.
    
    Returns:
        - age, gender, lob, codes: Stacked tensors
        - dt_cnt: torch.Tensor (was: List[int])          # CHANGED
        - target_multihot: [batch, len_dy, target_cd_cnt] # NEW
        - target: List of nested lists (backward compat)
    """
```

**New Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `target_multihot` | torch.Tensor | Pre-computed multi-hot targets [batch, len_dy, vocab_size] |
| `dt_cnt` | torch.Tensor | Valid day counts (was Python list) |

**Factory Function**:
```python
def create_collate_fn(config: 'BaseConfig'):
    """Factory to create collate function with config bound."""
    return partial(clinical_collate_fn, config=config)
```

---

### 13. Enhanced Model Saving

**Updated `save_trained_model()` (v3.1)**:
- Saves MoE configuration for proper model reconstruction
- Better handling of DataParallel and DataParallelWrapper wrapped models
- Includes additional metadata (embedding_size, nlayers) for easier reconstruction

```python
save_dict = {
    'model_state_dict': actual_model.state_dict(),
    'model_name': model_name,
    'model_type': type(actual_model).__name__,
    'embedding_size': config.embedding_size,        # NEW
    'nlayers': config.nlayers,                      # NEW
    'checkpoint_dir': checkpoint_dir,
    'timestamp': datetime.now().isoformat(),
    'config': {...},
    'moe_config': vars(moe_config) if moe_config else None,  # NEW
}
```

---

### 14. GPU & Memory Optimizations

#### 14a. DataLoader Workers Fix

**Problem**: Default worker calculation caused kernel death from memory exhaustion.

```python
# BEFORE (problematic):
n_workers = max(1, os.cpu_count() // max(world_size, 1) // 2)
# = 64 // 1 // 2 = 32 workers on 64-core machine

# AFTER (fixed):
if torch.cuda.device_count() > 1:
    n_workers = min(4, os.cpu_count() // 4)  # Max 4 for multi-GPU
else:
    n_workers = min(8, os.cpu_count() // 2)  # Max 8 for single GPU
```

**Impact**:
| Metric | Before | After |
|--------|--------|-------|
| Worker memory | 32 × 2-4GB = 64-128GB | 4 × 2-4GB = 8-16GB |
| Prefetch queue | 32GB+ | 4GB |
| Kernel stability | Frequent crashes | Stable |

#### 14b. Gradient Accumulation Configuration

**Problem**: Default `accumulation_steps=4` made training 4x slower without user awareness.

```python
# BEFORE: accumulation_steps not passed, defaulted to 4
train_metrics = train_epoch(model=model, dataloader=train_loader, ...)

# AFTER: Explicit parameter, defaults to 1 for DataParallel
train_metrics = train_epoch(
    model=model,
    dataloader=train_loader,
    ...,
    accumulation_steps=1  # No accumulation with DataParallel
)
```

**Effective Batch Size Calculation**:
| Setup | batch_size | accumulation | GPUs | Effective |
|-------|------------|--------------|------|-----------|
| Single GPU baseline | 32 | 1 | 1 | 32 |
| Multi-GPU (before) | 128 | 4 | 4 | 512 |
| Multi-GPU (after) | 128 | 1 | 4 | 128 |

#### 14c. Learning Rate Scaling

**Problem**: sqrt scaling was too conservative for linear batch scaling.

```python
# BEFORE: sqrt scaling
scaled_lr = base_lr * math.sqrt(num_gpus)  # 1e-4 * sqrt(4) = 2e-4

# AFTER: Linear scaling (industry standard for DataParallel)
if use_data_parallel:
    scaled_lr = base_lr * num_gpus  # 1e-4 * 4 = 4e-4
```

**Reference**: Linear scaling rule from "Accurate, Large Minibatch SGD" (Goyal et al., 2017)

#### 14d. Scheduler Configuration Fix

**Problem**: `T_max=epochs` with per-step scheduler calls caused cosine cycle to complete in one step.

```python
# BEFORE (broken):
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# With epochs=1, T_max=1, cosine completes after 1 step!

# AFTER (correct):
total_steps = len(train_loader) // accumulation_steps * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

#### 14e. Enhanced Collate Function for DataParallel

**Purpose**: Pre-compute multi-hot targets as tensors for efficient GPU scatter.

```python
def clinical_collate_fn_v2(batch: List[Dict], config: 'BaseConfig') -> Dict[str, Any]:
    """
    CRITICAL FOR DATAPARALLEL:
    - All outputs must be tensors (not Python lists)
    - Targets pre-computed to avoid GPU 0 bottleneck
    - dt_cnt as tensor for GPU scatter
    """
    # Pre-compute multi-hot targets: [batch, len_dy, target_cd_cnt]
    targets_multihot = torch.zeros(batch_size, len_dy, target_cd_cnt, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        for day_idx, day_codes in enumerate(item['target']):
            for code_idx in day_codes:
                if 0 <= code_idx < target_cd_cnt:
                    targets_multihot[i, day_idx, code_idx] = 1.0
    
    return {
        'age': ages,
        'gender': genders,
        'lob': lobs,
        'codes': codes,
        'dt_cnt': dt_cnts,              # Tensor (was list)
        'target_multihot': targets_multihot,  # NEW
        'target': targets_list           # Kept for metrics
    }
```

#### 14f. Memory Management in Training Loop

**Additions**:
```python
# Explicit tensor cleanup
del x
if 'output' in dir() and output is not None:
    del output
del total_loss

# Periodic garbage collection
if batch_idx % 100 == 0:
    gc.collect()
    
# GPU memory tracking
if is_main and device.type == 'cuda' and batch_idx % 1000 == 0:
    for gpu_id in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        print(f'    GPU {gpu_id}: {allocated:.2f}GB / {peak:.2f}GB peak')
```

---

### 15. Intrinsic Metrics Enhancements

#### 15a. MRR Bug Fix

**Problem**: MRR computed rank of arbitrary first code in list, not best-ranked true code.

```python
# BEFORE (buggy):
for i, target_codes in enumerate(targets):
    true_codes = [c for c in target_codes if c != 0]
    if len(true_codes) > 0:
        first_true = true_codes[0]  # ARBITRARY first code!
        rank = (sorted_indices[i] == first_true).nonzero()
        reciprocal_ranks.append(1.0 / (rank.item() + 1))

# AFTER (correct):
for i, target_codes in enumerate(targets):
    true_codes = [c for c in target_codes if c != 0]
    if len(true_codes) > 0:
        # Find rank of BEST-RANKED true code
        best_rank = float('inf')
        for code in true_codes:
            rank_tensor = (sorted_indices[i] == code).nonzero(as_tuple=True)[0]
            if len(rank_tensor) > 0 and rank_tensor.item() < best_rank:
                best_rank = rank_tensor.item()
        
        if best_rank < float('inf'):
            reciprocal_ranks.append(1.0 / (best_rank + 1))
```

#### 15b. New Metric Functions

**Micro-Recall@K** (per-code hit rate):
```python
def compute_micro_recall_at_k(predictions, targets, k_values=[5, 10, 20, 50]):
    """
    Micro-averaged Recall@K: Total hits / Total true labels across all samples.
    
    Unlike sample-level Recall@K (binary hit/miss per sample), this measures
    what fraction of ALL true codes are captured in top-K.
    """
    for k in k_values:
        top_k_preds = sorted_indices[:, :k]
        total_hits = 0
        total_true = 0
        
        for i, target_codes in enumerate(targets):
            true_codes = set(c for c in target_codes if c != 0)
            total_true += len(true_codes)
            pred_set = set(top_k_preds[i].tolist())
            total_hits += len(true_codes & pred_set)
        
        metrics[f'micro_recall@{k}'] = total_hits / total_true
```

**NDCG@K** (ranking quality):
```python
def compute_ndcg_at_k(predictions, targets, k_values=[10, 20, 50]):
    """
    Normalized Discounted Cumulative Gain @ K.
    
    Accounts for:
    1. Position-based discounting (earlier = better)
    2. Relevance scores (binary in our case)
    3. Normalized by ideal ranking
    """
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    
    for k in k_values:
        for i, target_codes in enumerate(targets):
            true_codes = set(c for c in target_codes if c != 0)
            top_k_preds = sorted_indices[i, :k].tolist()
            
            dcg = sum(discounts[rank] for rank, pred in enumerate(top_k_preds) 
                      if pred in true_codes)
            idcg = sum(discounts[:min(len(true_codes), k)])
            ndcg = dcg / idcg if idcg > 0 else 0.0
```

**Positive-Only Brier Score**:
```python
def compute_positive_brier_score(predictions, targets, vocab_size):
    """
    Brier score computed ONLY on positive labels.
    
    Standard Brier is dominated by true negatives (~99.7% of entries).
    This variant measures calibration specifically for positive predictions.
    """
    probs = torch.sigmoid(predictions)
    positive_probs = []
    
    for i, target_codes in enumerate(targets):
        for code in target_codes:
            if 0 < code < vocab_size:
                positive_probs.append(probs[i, code].item())
    
    # For positive labels, target=1, so Brier = (prob - 1)^2
    positive_brier = np.mean((np.array(positive_probs) - 1.0) ** 2)
```

**Macro AUROC/AUPRC**:
```python
def compute_auroc_auprc(predictions, targets, vocab_size, num_codes_to_sample=500):
    """
    Macro-averaged AUROC and AUPRC across codes.
    
    Due to computational cost, samples a subset of codes:
    - All codes that appear in targets
    - Random sample of additional codes
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # ... (sampling and computation logic)
    
    return {
        'macro_auroc': np.mean(aurocs),
        'macro_auprc': np.mean(auprcs),
        'num_codes_evaluated': len(aurocs)
    }
```

#### 15c. Metric Naming Consistency

**Renamed `top_K_acc` → `recall@K`** throughout codebase:

| Location | Before | After |
|----------|--------|-------|
| `evaluate()` returns | `top_1_acc`, `top_5_acc`, ... | `recall@1`, `recall@5`, ... |
| `_build_epoch_metrics()` | `eval_in_train_top_10_acc` | `eval_in_train_recall@10` |
| `_build_final_results()` | `final_top_10_acc` | `final_val_recall@10` |
| Logging statements | `Top-10: {val_metrics['top_10_acc']}` | `Recall@10: {val_metrics['recall@10']}` |

#### 15d. Removed Problematic Metrics

| Metric | Issue | Action |
|--------|-------|--------|
| `mAP@K` | Excluded samples with no hits, biasing upward | **Removed** |
| `brier_score` | Dominated by true negatives (~99.7%) | **Replaced** with `positive_brier` |
| `ece` | Same TN dominance issue | **Removed** |

#### 15e. Updated Training Print Statement

```python
# BEFORE:
print(f"    Loss: {loss:.4f} | R@10: {recall@10:.3f} | R@20: {recall@20:.3f} | "
      f"P@10: {precision@10:.3f} | P@20: {precision@20:.3f} | "
      f"mAP20: {mAP@20:.3f} | mAP50: {mAP@50:.3f} | Brier: {brier_score:.4f}")

# AFTER:
print(f"    Loss: {loss:.4f} | R@10: {recall@10:.3f} | R@20: {recall@20:.3f} | "
      f"μR@10: {micro_recall@10:.3f} | P@10: {precision@10:.3f} | "
      f"NDCG@20: {ndcg@20:.3f} | PosBrier: {positive_brier:.4f}")
```

#### 15f. Metrics Summary Table

| Metric | What It Measures | Clinical Interpretation |
|--------|------------------|------------------------|
| **Recall@K** | At least one true code in top-K? | Alert relevance |
| **Micro-Recall@K** | Fraction of ALL true codes captured | Per-code coverage |
| **Precision@K** | Hits / K predictions | Prediction accuracy |
| **NDCG@K** | Ranking quality with position discount | Correct codes ranked higher? |
| **MRR** | Rank of best true code | How quickly do we find relevance? |
| **Positive Brier** | Calibration on positive labels | Confidence when correct |
| **Macro AUROC** | Per-code discrimination ability | Model distinguishes per code? |
| **Macro AUPRC** | PR-AUC averaged across codes | Handles class imbalance |

---

## Directory Structure (Complete)

```
logs/{experiment_round}/{exp_name}/
├── checkpoints/                    # Training resume (save_checkpoint)
│   ├── checkpoint_latest.pt
│   ├── checkpoint_best.pt
│   └── checkpoint_epoch{N}.pt
├── saved_models/                   # Inference (save_trained_model)
│   ├── {model_name}_final.pt
│   ├── {model_name}_config.json
│   └── {model_name}_results.json
├── downstream/                     # NEW: Downstream results
│   ├── {lob_name}/
│   │   └── downstream_{task}_{lob}_{timestamp}.json
│   └── multi_lob_summary_{timestamp}.json
├── epoch_metrics.json
├── batch_metrics.json
├── config.json
├── final_results.json
└── {exp_name}.log
```

---

## Test Functions

| Test Function | Coverage |
|---------------|----------|
| `test_conv_lob()` | LOB string conversion |
| `test_clinical_dataset_with_lob()` | Dataset with LOB support |
| `test_model_forward_with_lob()` | Model forward pass with LOB input |
| `test_ddp_initialization()` | DDP initialization on all GPUs |
| `test_model_saving_and_loading()` | Model save/load cycle |
| `test_downstream_evaluator_with_real_data()` | Downstream evaluation pipeline |
| `test_run_single_experiment_with_downstream()` | End-to-end with downstream |
| `test_embedding_extractor()` | Embedding extraction from all model types |

---

## Data Requirements

| Version | Training Data | Validation Data |
|---------|--------------|-----------------|
| v2 | `sample_data/mdcd_train_1m.feather` | `sample_data/mdcd_val_10k.feather` |
| v3 | `sample_data/extrinsic_mdcd_ip/te_pretrain_train.feather` | `sample_data/extrinsic_mdcd_ip/te_pretrain_val_mdcd_ip_probe.feather` |

**Required Columns** (v3.1):
- Pre-training: `age_in_months`, `gender_cd`, `cd`, `target`, `dt_cnt`, `lob`
- Downstream: `individual_id`, `index_dt`, `{outcome_column}` (e.g., `acute_ip_flag`)

---

## Dependencies Added in v3.1

```python
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from utils.metrics import (
    lift_at_percentage, 
    true_positives_at_percentage,
    precision_at_percentage,
    recall_at_percentage,
    f1_at_percentage,
    # ... etc
)
```
