# Logging System Architecture & Version Control
- **Original Analysis**: Dec 28, 2025
- **Last Updated**: Dec 29, 2025
- **File**: `dev/moe/moe_flashattn_3.py` (V3.1)

---

## Version History

| Version | Date | Summary |
|---------|------|---------|
| V3.0 | Dec 28, 2025 | Initial analysis - identified `batch_metrics.json` never populated, `config.json` incomplete |
| V3.1 | Dec 29, 2025 | **FIXED**: Batch metrics logging integrated, config logging complete, train_epoch accepts loggers |

---

## Executive Summary

### Current Status (V3.1)

| # | Component | Location | Status | V3.1 Aligned? |
|---|-----------|----------|--------|---------------|
| 1 | `batch_metrics.json` | Line 6245 | ✅ **FIXED** | ✅ Yes |
| 2 | `config.json` | Line 6250 | ✅ **FIXED** | ✅ Yes |
| 3 | `epoch_metrics.json` | Line 6241 | ✅ Working | ✅ Yes |
| 4 | `final_results.json` | Line 6256 | ✅ Working | ✅ Yes |
| 5 | `loss_trajectory_epoch*.json` | Line 5033-5040 | ✅ Working | ✅ Yes |
| 6 | `training.log` | Line 331 | ✅ **ENHANCED** | ✅ Yes |
| 7 | `saved_models/` folder | Line 10294-10396 | ✅ Working | ✅ Yes |
| 8 | `checkpoints/` folder | Line 6024-6115 | ✅ Working | ✅ Yes |

---

## Changes from V3.0 to V3.1

### Change 1: Fixed `batch_metrics.json` - Now Populated

**Before (V3.0) - BROKEN:**
```python
# In train_epoch():
# batch_metrics_buffer populated but NEVER written to MetricsLogger
batch_metrics_buffer.append(batch_metrics)
# ... batch_metrics.json remained EMPTY
```

**After (V3.1) - FIXED:**

#### Step 1: `train_epoch()` Signature Updated (Lines 4741-4763)
```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_mixed_precision: bool = False,
    moe_config: Optional[MoEConfig] = None,
    epoch: int = 1,
    use_bucketing: bool = False,
    log_interval: int = 500,
    global_step: int = 0, 
    loss_tracker: Optional[LossTracker] = None,
    is_main: bool = True,
    use_ddp: bool = False,
    accumulation_steps: int = 1,
    track_gpu_memory: bool = True,
    metrics_logger: Optional['MetricsLogger'] = None,  # NEW: Batch-metrics logger
    logger: Optional[logging.Logger] = None,           # NEW: General training logger
) -> Dict[str, float]:
```

#### Step 2: Batch Entry Construction (Lines 5000-5004)
```python
# Inside train_epoch, every log_interval batches:
batch_entry = {
    'global_step': global_step,
    'loss': pred_loss_scalar,
    **batch_metrics  # recall@10, precision@10, micro_recall@10, ndcg@20, etc.
}
```

#### Step 3: MoE Metrics Added to Batch Entry (Lines 5044-5053)
```python
# If MoE model, add routing health metrics:
if moe_losses and 'expert_usage' in moe_losses:
    batch_entry.update({
        'moe_cv': moe_batch_metrics['expert_load_cv'],
        'moe_collapsed': moe_batch_metrics['num_collapsed_experts'],
        'moe_gini': moe_batch_metrics['expert_gini'],
        'router_gradnorm_mean': latest_router.get('router_grad_norm_mean', 0),
        'router_weight_std': latest_router.get('router_weight_std', 0),
        'router_grad_exploding': latest_router.get('router_grad_exploding', 0),
        'router_grad_vanishing': latest_router.get('router_grad_vanishing', 0)
    })
```

#### Step 4: Write to MetricsLogger (Lines 5059-5060)
```python
if metrics_logger:
    metrics_logger.log_batch(epoch=epoch, batch=batch_idx, metrics=batch_entry)
```

**Resulting `batch_metrics.json` Structure:**
```json
[
  {
    "epoch": 1,
    "batch": 500,
    "global_step": 500,
    "loss": 0.4523,
    "recall@5": 0.342,
    "recall@10": 0.456,
    "recall@20": 0.567,
    "precision@10": 0.045,
    "micro_recall@10": 0.234,
    "micro_recall@20": 0.345,
    "ndcg@20": 0.312,
    "positive_brier": 0.0234,
    "mrr": 0.289,
    "moe_cv": 0.234,
    "moe_collapsed": 0,
    "moe_gini": 0.123,
    "router_gradnorm_mean": 0.0456,
    "router_weight_std": 0.0123,
    "router_grad_exploding": 0,
    "router_grad_vanishing": 0
  },
  ...
]
```

---

### Change 2: Fixed `config.json` - Now Complete

**Before (V3.0) - INCOMPLETE:**
```python
# metrics_logger.log_config() was never called
# OR called with minimal data
# config.json was empty or missing OptimizeConfig/MoEConfig
```

**After (V3.1) - FIXED:**

#### Config Logging Call (Lines 10941-10953)
```python
metrics_logger.log_config({
    'experiment': exp_name,
    'embedding_size': eff_d_model,
    'nhid': dims['nhid'],
    'nhead': dims['nhead'],
    'batch_size': config.batch_size,
    'use_mixed_precision': use_mixed_precision,
    'use_bucketing': use_bucketing,
    'use_learnt_att_pool': use_learnt_att_pool,
    'optimize_config': vars(optimize_config) if optimize_config else None,
    'moe_config': vars(moe_config) if moe_config else None
})
```

**Resulting `config.json` Structure:**
```json
{
  "experiment": "exp1_moe_top2",
  "embedding_size": 256,
  "nhid": 512,
  "nhead": 8,
  "batch_size": 32,
  "use_mixed_precision": true,
  "use_bucketing": true,
  "use_learnt_att_pool": true,
  "optimize_config": {
    "scheduler_type": "cosine_warmup",
    "warmup_pct": 0.1,
    "min_lr_ratio": 0.1,
    "use_pos_weight": true,
    "pos_weight_method": "tiered",
    "use_focal_loss": false,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25
  },
  "moe_config": {
    "num_experts": 8,
    "num_shared_experts": 2,
    "top_k": 2,
    "use_moe_from_layer": 3,
    "aux_loss_weight": 0.01,
    "router_jitter": 0.1,
    "capacity_factor": 1.25
  }
}
```

---

### Change 3: Enhanced `training.log` - Batch-Level Debug Output

**Before (V3.0):**
```
# training.log only had INFO+ level messages
# Batch-level metrics only printed to console, not logged to file
```

**After (V3.1):**

#### Batch Metrics Logged at DEBUG Level (Lines 4995-4997)
```python
# In train_epoch, every log_interval batches:
batch_log_msg = (
    f"    Loss: {pred_loss_scalar:.4f} | "
    f"R@10: {batch_metrics['recall@10']:.3f} | "
    f"R@20: {batch_metrics['recall@20']:.3f} | "
    f"μR@10: {batch_metrics['micro_recall@10']:.3f} | "
    f"P@10: {batch_metrics['precision@10']:.3f} | "
    f"NDCG@20: {batch_metrics['ndcg@20']:.3f} | "
    f"PosBrier: {batch_metrics['positive_brier']:.4f}"
)
print(batch_log_msg)  # Console output

if logger:
    logger.debug(batch_log_msg)  # Also to training.log
```

#### MoE Metrics Also Logged (Lines 5041-5042)
```python
if logger:
    logger.debug(moe_log_msg)  # MoE routing health to training.log
```

**Example `training.log` Output:**
```
2025-12-29 10:15:23 - exp1_moe - INFO - --- Epoch 1/3 ---
2025-12-29 10:15:45 - exp1_moe - DEBUG -     Loss: 0.4523 | R@10: 0.456 | R@20: 0.567 | μR@10: 0.234 | P@10: 0.045 | NDCG@20: 0.312 | PosBrier: 0.0234
2025-12-29 10:15:45 - exp1_moe - DEBUG -     MoE: CV=0.234 | Collapsed=0 | Gini=0.123 | Router: GradNorm=0.0456 | WeightStd=0.0123
2025-12-29 10:16:07 - exp1_moe - DEBUG -     Loss: 0.4012 | R@10: 0.489 | ...
```

---

### Change 4: `train_epoch()` Call Site Updated

**Before (V3.0):**
```python
# In run_single_experiment():
train_metrics = train_epoch(
    model=model,
    dataloader=train_loader,
    ...
    accumulation_steps=1
    # NO metrics_logger or logger passed!
)
```

**After (V3.1) - Lines 11030-11034:**
```python
train_metrics = train_epoch(
    model=model,
    dataloader=train_loader,
    ...
    accumulation_steps=1,
    metrics_logger=metrics_logger,  # NEW: Enable batch_metrics.json
    logger=logger                    # NEW: Enable training.log debug output
)
```

---

## Logging Flow Diagram (V3.1)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LOGGING FLOW (V3.1)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXPERIMENT START (run_single_experiment):                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ metrics_logger.log_config({                                              │ │
│  │   experiment, embedding_size, nhid, nhead, batch_size,                   │ │
│  │   optimize_config, moe_config, ...                                       │ │
│  │ })                                                                       │ │
│  │  → config.json                                                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  DURING TRAINING (train_epoch):                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Every log_interval batches:                                              │ │
│  │                                                                          │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │ │
│  │  │ compute_batch_metrics_lightweight()                                │  │ │
│  │  │  → batch_metrics (recall@K, precision@K, micro_recall@K, ndcg@K,  │  │ │
│  │  │                   mrr, positive_brier)                             │  │ │
│  │  └────────────────────────────────────────────────────────────────────┘  │ │
│  │             │                                                            │ │
│  │             ├─── print(batch_log_msg)  → Console                         │ │
│  │             ├─── logger.debug(batch_log_msg)  → training.log             │ │
│  │             │                                                            │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │ │
│  │  │ compute_moe_batch_metrics() + compute_router_gradient_metrics()   │  │ │
│  │  │  → moe_cv, moe_collapsed, moe_gini, router_gradnorm_mean, etc.    │  │ │
│  │  └────────────────────────────────────────────────────────────────────┘  │ │
│  │             │                                                            │ │
│  │             ├─── print(moe_log_msg)  → Console                           │ │
│  │             ├─── logger.debug(moe_log_msg)  → training.log               │ │
│  │             │                                                            │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │ │
│  │  │ batch_entry = {global_step, loss, **batch_metrics, **moe_metrics} │  │ │
│  │  │ metrics_logger.log_batch(epoch, batch, batch_entry)               │  │ │
│  │  └────────────────────────────────────────────────────────────────────┘  │ │
│  │             │                                                            │ │
│  │             └─── batch_metrics.json (appended)                           │ │
│  │                                                                          │ │
│  │  loss_tracker.log_loss(step, loss)  → In-memory trajectory               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  AT EPOCH END (run_single_experiment):                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ epoch_metrics = _build_epoch_metrics(...)                                │ │
│  │ metrics_logger.log_epoch(epoch, epoch_metrics)  → epoch_metrics.json     │ │
│  │ loss_tracker.save_trajectory(...)  → loss_trajectory_epoch*.json         │ │
│  │ save_checkpoint(...)  → checkpoints/                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  AFTER ALL EPOCHS (run_single_experiment):                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ comprehensive_evaluation(...)  → evaluation results                      │ │
│  │ results = _build_final_results(...)                                      │ │
│  │ metrics_logger.save_final_results(results)  → final_results.json         │ │
│  │ save_trained_model(...)  → saved_models/                                 │ │
│  │ metrics_logger.save()  → Writes all JSON files                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Analysis (V3.1)

### 1. `batch_metrics.json` ✅ **FIXED**

**Location**: `MetricsLogger.save()` at line 6245

**What it logs** (per `log_interval` batches):
- `epoch`: Current epoch number
- `batch`: Batch index within epoch
- `global_step`: Cumulative step count
- `loss`: BCE/Focal loss value
- `recall@5/10/20`: Standard recall metrics
- `precision@10`: Precision at K
- `micro_recall@10/20`: Per-code micro recall
- `ndcg@20`: Normalized DCG
- `mrr`: Mean Reciprocal Rank
- `positive_brier`: Calibration metric
- `moe_cv`: Expert load coefficient of variation (MoE only)
- `moe_collapsed`: Number of collapsed experts (MoE only)
- `moe_gini`: Expert usage Gini coefficient (MoE only)
- `router_gradnorm_mean`: Router gradient norm (MoE only)
- `router_weight_std`: Router weight std (MoE only)
- `router_grad_exploding`: Gradient explosion flag (MoE only)
- `router_grad_vanishing`: Gradient vanishing flag (MoE only)

**Use Cases**:
- Real-time training monitoring
- Learning rate warmup validation
- Early detection of training instabilities
- MoE routing health analysis
- Debugging model collapse

---

### 2. `config.json` ✅ **FIXED**

**Location**: `MetricsLogger.save()` at line 6250

**What it logs**:
- `experiment`: Experiment name
- `embedding_size`: Model dimension (d_model)
- `nhid`: FFN hidden dimension
- `nhead`: Number of attention heads
- `batch_size`: Effective batch size
- `use_mixed_precision`: FP16/BF16 flag
- `use_bucketing`: Bucketing sampler flag
- `use_learnt_att_pool`: Learned pooling flag
- `optimize_config`: Full optimizer configuration
  - `scheduler_type`: LR scheduler type
  - `warmup_pct`: Warmup percentage
  - `min_lr_ratio`: Minimum LR ratio
  - `use_pos_weight`: Positive weight flag
  - `pos_weight_method`: Weight method (tiered/log_scaled/ENS)
  - `use_focal_loss`: Focal loss flag
  - `focal_gamma`: Focal loss gamma
  - `focal_alpha`: Focal loss alpha
- `moe_config`: Full MoE configuration
  - `num_experts`: Total experts
  - `num_shared_experts`: Shared experts
  - `top_k`: Top-K routing
  - `use_moe_from_layer`: MoE layer start
  - `aux_loss_weight`: Auxiliary loss weight
  - `router_jitter`: Router jitter noise
  - `capacity_factor`: Capacity factor

**Use Cases**:
- Experiment reproducibility
- Hyperparameter tracking
- Configuration comparison across runs

---

### 3. `epoch_metrics.json` ✅ **Working**

**Location**: `MetricsLogger.log_epoch()` at line 6206

**What it logs** (per epoch):
- Training trajectory: `train_loss`, `train_loss_mean`, `train_loss_first`, `train_loss_last`, `train_loss_std`, `train_loss_improvement`
- Train evaluation: `eval_in_train_loss_final`, `eval_in_train_recall@1/5/10/20`, `eval_in_train_micro_recall@10`, `eval_in_train_ndcg@20`
- Validation: `final_val_loss`, `final_val_recall@1/5/10/20`, `final_val_micro_recall@10/20`, `final_val_ndcg@10/20`, `final_val_mrr`, `final_val_positive_brier`
- `generalization_gap`: Train vs Val loss difference
- Embedding quality: `embedding_std_mean`, `nn_target_overlap` (if computed)

---

### 4. `final_results.json` ✅ **Working**

**Location**: `MetricsLogger.save_final_results()` at line 6253

**What it logs**:
- Experiment metadata: `experiment`, `parameters`, `use_learned_pooling`, `use_bucketing`
- Final training metrics: `train_loss_mean`, `train_loss_learned`, `train_loss_final`
- Final validation metrics: `val_loss_final`, `generalization_gap`
- All recall/precision metrics at various K values
- Comprehensive evaluation results (performance, efficiency, resources, MoE)
- Full epoch history (`all_epochs`)
- Total training time

---

### 5. `loss_trajectory_epoch*.json` ✅ **Working**

**Location**: `LossTracker.save_trajectory()` at lines 5033-5040

**What it logs**:
- `steps`: List of step numbers
- `losses`: List of loss values per step
- `epoch_summaries`: Aggregated epoch statistics

---

### 6. `training.log` ✅ **ENHANCED**

**Location**: `setup_experiment_logging()` at lines 294-347

**Configuration**:
- File handler at DEBUG level (captures all batch details)
- Console handler at INFO level (summary only)
- Append mode on resume (`file_mode = 'a' if resume else 'w'`)

**What it logs**:
- `INFO`: Epoch starts, evaluation results, checkpoint saves, training completion
- `DEBUG`: **NEW** Batch-level metrics, MoE routing health, learning rate
- `WARNING`: Expert collapse, gradient issues, memory warnings
- `ERROR`: Training failures, OOM recovery

---

### 7. `saved_models/` folder ✅ **Working**

**Location**: `save_trained_model()` at lines 10294-10396

**Contents**:
- `{model_name}_final.pt`: Model state dict + metadata
- `{model_name}_config.json`: Full BaseConfig as JSON
- `{model_name}_results.json`: Experiment results
- `{model_name}_best.pt`: Copy if best model (optional)

---

### 8. `checkpoints/` folder ✅ **Working**

**Location**: `save_checkpoint()` at lines 6024-6115

**Contents**:
- `checkpoint_latest.pt`: For training resume
- `checkpoint_best.pt`: Best validation loss checkpoint
- `checkpoint_epoch{N}.pt`: Per-epoch rolling checkpoints

**Checkpoint Contents**:
- `epoch`, `global_step`
- `model_state_dict`, `optimizer_state_dict`
- `scaler_state_dict`, `scheduler_state_dict`
- `best_val_loss`, `metrics`
- `moe_config` (if applicable)

---

## MetricsLogger Class Reference

**Location**: Lines 6156-6278

```python
class MetricsLogger:
    """JSON-based metrics logger for structured experiment tracking."""
    
    def __init__(self, exp_name: str, log_dir: str = "logs", resume: bool = False):
        """Initialize logger with optional resume support."""
    
    def init_resume(self):
        """Load existing epoch/batch metrics for training resume."""
    
    def log_config(self, config: Dict):
        """Log experiment configuration."""
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics."""
    
    def log_batch(self, epoch: int, batch: int, metrics: Dict[str, float]):
        """Log batch-level metrics (for real-time monitoring)."""
    
    @staticmethod
    def convert_to_serializable(obj):
        """Convert numpy/torch types to JSON-serializable Python types."""
    
    def save(self):
        """Save all metrics to JSON files (epoch, batch, config)."""
    
    def save_final_results(self, results: Dict):
        """Save complete experiment results to JSON."""
    
    def get_summary(self) -> Dict:
        """Get summary statistics (best val loss, final metrics, etc.)."""
```

---

## Output Directory Structure

```
logs/
└── {exp_name}/
    ├── config.json                    # Experiment configuration (V3.1 FIXED)
    ├── epoch_metrics.json             # Per-epoch metrics
    ├── batch_metrics.json             # Per-batch metrics (V3.1 FIXED)
    ├── final_results.json             # Comprehensive evaluation results
    ├── loss_trajectory_epoch0.json    # Per-batch loss (epoch 0)
    ├── loss_trajectory_epoch1.json    # Per-batch loss (epoch 1)
    ├── training.log                   # Human-readable log (V3.1 ENHANCED)
    ├── checkpoints/
    │   ├── checkpoint_latest.pt
    │   ├── checkpoint_best.pt
    │   └── checkpoint_epoch{N}.pt
    └── saved_models/
        ├── {model_name}_final.pt
        ├── {model_name}_config.json
        └── {model_name}_results.json
```

---

## Changelog

### V3.1 (Dec 29, 2025)
- **FIXED**: `batch_metrics.json` now populated via `metrics_logger.log_batch()` in `train_epoch()`
- **FIXED**: `config.json` now includes `optimize_config` and `moe_config`
- **ADDED**: `metrics_logger` and `logger` parameters to `train_epoch()` signature
- **ADDED**: Batch-level metrics logged to `training.log` at DEBUG level
- **ADDED**: MoE metrics (cv, collapsed, gini, router health) to batch_entry
- **ADDED**: Router gradient monitoring (exploding/vanishing detection)
- **ENHANCED**: `train_epoch()` call site passes both loggers

### V3.0 (Dec 28, 2025)
- Initial logging system analysis
- Identified `batch_metrics.json` never populated (log_batch() not called)
- Identified `config.json` incomplete (log_config() called with minimal data)
- Documented all 8 logging components

---

## Summary: What Was Fixed

| Issue | Root Cause | Fix Applied |
|-------|------------|-------------|
| `batch_metrics.json` empty | `log_batch()` never called | Added call in `train_epoch()` at line 5060 |
| `config.json` incomplete | `log_config()` called with minimal data | Expanded config dict at lines 10941-10953 |
| Batch metrics not in `training.log` | Only `print()`, no `logger.debug()` | Added logger.debug() at lines 4996-4997 |
| MoE metrics not persisted | Only in-memory buffers | Added to batch_entry at lines 5044-5053 |
| `train_epoch()` isolated from loggers | No logger parameters | Added `metrics_logger` and `logger` params |

---

## Alignment with V3.1 Changes

| Change | Logging Aligned? |
|--------|-----------------|
| Metric renaming (`top_K_acc` → `recall@K`) | ✅ Yes |
| New metrics (micro_recall, NDCG, positive_brier) | ✅ Yes |
| MRR bug fix | ✅ Yes |
| OptimizeConfig logging | ✅ **FIXED** |
| MoEConfig logging | ✅ **FIXED** |
| Batch metrics persistence | ✅ **FIXED** |
| MoE routing health metrics | ✅ **FIXED** |
| Router gradient monitoring | ✅ **FIXED** |
