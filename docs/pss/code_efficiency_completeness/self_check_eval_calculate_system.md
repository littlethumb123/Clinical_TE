# Evaluation System Architecture & Version Control
- **Original Analysis**: Dec 28, 2025
- **Last Updated**: Dec 29, 2025
- **File**: `dev/moe/moe_flashattn_3.py` (V3.1)

---

## Version History

| Version | Date | Summary |
|---------|------|---------|
| V3.0 | Dec 28, 2025 | Initial analysis - identified duplications and optimization opportunities |
| V3.1 | Dec 29, 2025 | **IMPLEMENTED**: Removed `evaluate(train_loader)`, merged `evaluate(val_loader)` into `comprehensive_evaluation()` on final epoch, added proper logging integration |

---

## Current Evaluation Flow (V3.1 - OPTIMIZED)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OPTIMIZED EVALUATION FLOW (V3.1)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DURING TRAINING (train_epoch):                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Every log_interval batches (~100):                                       │ │
│  │  → compute_batch_metrics_lightweight()                                   │ │
│  │    • recall@1/5/10/20                                                    │ │
│  │    • precision@5/10/20/50                                                │ │
│  │    • micro_recall@10/20                                                  │ │
│  │    • ndcg@10/20                                                          │ │
│  │    • mrr, positive_brier                                                 │ │
│  │                                                                          │ │
│  │  → Saved to:                                                             │ │
│  │    • batch_metrics_buffer (in memory) - for epoch averaging              │ │
│  │    • batch_metrics.json (via metrics_logger.log_batch())                 │ │
│  │    • training.log (via logger.debug())                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  AT EPOCH END (run_single_experiment):                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ ✅ OPTIMIZED: train_eval_metrics derived from batch averages             │ │
│  │    • NO separate evaluate(train_loader) call                             │ │
│  │    • Uses train_metrics from train_epoch() directly                      │ │
│  │    • Keys: val_loss (=train_loss), recall@K, micro_recall@K, ndcg@K, etc│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ NON-FINAL EPOCHS (epoch < epochs - 1):                                   │ │
│  │  → evaluate(val_loader) → val_metrics                                    │ │
│  │    • Lightweight validation on max_batches samples                       │ │
│  │    • Uses StreamingMetrics                                               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ FINAL EPOCH (epoch == epochs - 1):                                       │ │
│  │  ✅ OPTIMIZED: comprehensive_evaluation() directly                       │ │
│  │  • Skips separate evaluate(val_loader)                                   │ │
│  │  • val_metrics = comprehensive_result['performance']                     │ │
│  │  • Result cached in final_comprehensive_evaluation                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ (Optional) compute_embedding_quality_epoch()                             │ │
│  │  • Embedding std, NN overlap                                             │ │
│  │  • Only on non-final epochs when check_embeddings_every is set           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  AFTER ALL EPOCHS (run_single_experiment):                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ ✅ OPTIMIZED: Reuses cached comprehensive_evaluation                     │ │
│  │  if final_comprehensive_evaluation:                                      │ │
│  │      evaluation = final_comprehensive_evaluation  # No re-computation    │ │
│  │  else:                                                                   │ │
│  │      evaluation = comprehensive_evaluation(...)  # Edge case only        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Changes from V3.0 to V3.1

### Change 1: Removed `evaluate(train_loader)` 

**Before (V3.0):**
```python
# Lines ~11007-11017 (OLD)
logger.info("  Evaluating on training subset...")
train_eval_metrics = evaluate(
    model=model,
    dataloader=train_loader,  # ← REDUNDANT! Same data as batch_metrics
    criterion=criterion,
    ...
)
```

**After (V3.1):**
```python
# Lines ~11037-11061 (NEW)
logger.info("  Using batch-averaged training metrics (no re-evaluation)")
train_eval_metrics = {
    # Use train_loss as "val_loss" for the expected key format
    'val_loss': train_metrics.get('train_loss', 0.0),
    
    # Recall metrics (from batch sampling)
    'recall@1': train_metrics.get('train_recall@1', 0.0),
    'recall@5': train_metrics.get('train_recall@5', 0.0),
    'recall@10': train_metrics.get('train_recall@10', 0.0),
    'recall@20': train_metrics.get('train_recall@20', 0.0),
    
    # Micro-recall (from batch sampling)
    'micro_recall@10': train_metrics.get('train_micro_recall@10', 0.0),
    'micro_recall@20': train_metrics.get('train_micro_recall@20', 0.0),
    
    # NDCG (from batch sampling)
    'ndcg@10': train_metrics.get('train_ndcg@10', 0.0),
    'ndcg@20': train_metrics.get('train_ndcg@20', 0.0),
    
    # Other metrics
    'mrr': train_metrics.get('train_mrr', 0.0),
    'positive_brier': train_metrics.get('train_positive_brier', 0.0),
}
```

**Rationale:**
- The batch metrics computed during training (every `log_interval` batches) already provide training performance data
- Eliminates ~10-15% compute overhead of a separate forward pass
- Metrics are slightly noisier (training mode vs eval mode) but acceptable for monitoring

**Documentation Update in `_build_epoch_metrics()`:**
```python
"""
Args:
    train_eval_metrics (from train_metrics):
        - Derived from train_metrics batch averages
        - NOT a separate forward pass in eval mode
        - Slightly noisier than true eval-mode metrics
        - Adequate for epoch-level monitoring
        - For final comparison, use comprehensive_evaluation()
"""
```

---

### Change 2: Merged `evaluate(val_loader)` into `comprehensive_evaluation()` on Final Epoch

**Before (V3.0):**
```python
# At every epoch end:
val_metrics = evaluate(model, val_loader, ...)

# After all epochs:
comprehensive_evaluation(...)  # ← REDUNDANT StreamingMetrics computation
```

**After (V3.1):**
```python
# Lines ~11063-11101 (NEW)
if epoch == epochs - 1:
    # FINAL EPOCH: Run comprehensive_evaluation directly
    logger.info("  Final epoch: Running comprehensive evaluation...")
    
    comprehensive_result = comprehensive_evaluation(
        model=model,
        val_dataloader=val_loader,
        config=config,
        device=device,
        training_time_sec=current_time,
        epoch_history=epoch_history,
        code_frequencies=code_frequencies,
        moe_config=moe_config,
        use_mixed_precision=use_mixed_precision
    )
    
    # Extract val_metrics from comprehensive_evaluation
    val_metrics = comprehensive_result['performance']
    
    # Cache result to skip re-running after loop
    final_comprehensive_evaluation = comprehensive_result
    
else:
    # NON-FINAL EPOCHS: Use lightweight evaluate()
    logger.info("  Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader, ...)
```

**Rationale:**
- `comprehensive_evaluation()` already computes all StreamingMetrics that `evaluate()` computes
- Plus additional detailed metrics (stratified, AUROC, efficiency, resources, cost)
- Eliminates ~50% redundant compute on final epoch
- Single-epoch training (epochs=1) now correctly runs comprehensive evaluation

---

### Change 3: Added Caching for Post-Loop Reuse

**Before (V3.0):**
```python
# After training loop, always ran:
evaluation = comprehensive_evaluation(...)  # Even if just ran on final epoch
```

**After (V3.1):**
```python
# Lines ~11155-11171 (NEW)
if final_comprehensive_evaluation:
    logger.info("  Using cached comprehensive evaluation from final epoch")
    evaluation = final_comprehensive_evaluation  # No re-computation
else:
    # Fallback: run comprehensive evaluation (edge case: epochs=0)
    logger.info("  Running comprehensive evaluation...")
    evaluation = comprehensive_evaluation(...)
```

**Rationale:**
- Prevents double-computation when comprehensive_evaluation was already run on final epoch
- `final_comprehensive_evaluation` initialized as `None` before loop
- Set to result on final epoch, reused after loop

---

### Change 4: Logging Integration in `train_epoch()`

**Before (V3.0):**
- Batch metrics printed to console via `print()` statements
- `batch_metrics.json` never populated (metrics_logger.log_batch() not called)
- `training.log` did not contain batch-level details

**After (V3.1):**
```python
# train_epoch() signature updated to accept:
def train_epoch(
    ...
    metrics_logger: Optional['MetricsLogger'] = None,
    logger: Optional[logging.Logger] = None
):
```

```python
# Inside train_epoch, at each log_interval:
if logger:
    logger.debug(f"  Batch {batch_idx}/{total_batches} | Loss: {avg_loss:.4f} | ...")

if metrics_logger:
    batch_entry = {
        'epoch': epoch,
        'batch': batch_idx,
        'global_step': global_step,
        'loss': avg_loss,
        'lr': current_lr,
        # ... all batch metrics ...
    }
    metrics_logger.log_batch(batch_entry)
```

**Rationale:**
- `batch_metrics.json` now properly populated for post-training analysis
- `training.log` captures detailed batch-level metrics at DEBUG level
- Console output preserved via separate print statements for real-time monitoring

---

## Metrics Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           METRICS FLOW (V3.1)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  compute_batch_metrics_lightweight()  ──→  batch_metrics_buffer (memory)     │
│          │                                         │                         │
│          │                                         ▼                         │
│          │                            train_metrics (epoch avg)              │
│          │                                         │                         │
│          ▼                                         ▼                         │
│  metrics_logger.log_batch()           train_eval_metrics (derived)           │
│          │                                         │                         │
│          ▼                                         ▼                         │
│  batch_metrics.json                   _build_epoch_metrics()                 │
│                                                    │                         │
│                                                    ▼                         │
│                                        epoch_metrics → epoch_history         │
│                                                    │                         │
│                                                    ▼                         │
│                                        metrics_logger.log_epoch()            │
│                                                    │                         │
│                                                    ▼                         │
│                                        epoch_metrics.json                    │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  evaluate() / comprehensive_evaluation()  ──→  val_metrics                   │
│          │                                         │                         │
│          │                                         ▼                         │
│          │                            _build_epoch_metrics()                 │
│          │                                                                   │
│          ▼                                                                   │
│  StreamingMetrics (for basic metrics)                                        │
│          │                                                                   │
│          ▼                                                                   │
│  (comprehensive_evaluation only):                                            │
│    • compute_primary_task_metrics()                                          │
│    • compute_loss_metrics()                                                  │
│    • compute_stratified_metrics()                                            │
│    • compute_auroc_auprc()                                                   │
│    • compute_training_time_metrics()                                         │
│    • compute_memory_metrics()                                                │
│    • compute_flops_metrics()                                                 │
│    • compute_cost_metrics()                                                  │
│    • compute_moe_performance_metrics()                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Files Summary

| File | Source | Contents |
|------|--------|----------|
| `batch_metrics.json` | `metrics_logger.log_batch()` | Per-batch metrics (every log_interval) |
| `epoch_metrics.json` | `metrics_logger.log_epoch()` | Per-epoch aggregated metrics |
| `loss_trajectory_epoch*.json` | `loss_tracker.save_trajectory()` | Per-batch loss values |
| `config.json` | `metrics_logger.log_config()` | Full config including OptimizeConfig, MoEConfig |
| `final_results.json` | `metrics_logger.save_final_results()` | Comprehensive evaluation + experiment summary |
| `training.log` | `logging.FileHandler` | All INFO+ messages; DEBUG has batch details |

---

## Key Design Decisions

### 1. Training Eval vs. Validation Eval

| Aspect | Training Eval (`train_eval_metrics`) | Validation Eval (`val_metrics`) |
|--------|--------------------------------------|----------------------------------|
| **Source** | Derived from batch_metrics_buffer averages | `evaluate()` or `comprehensive_evaluation()` |
| **Mode** | Training mode (dropout active) | Eval mode (deterministic) |
| **Data** | Sampled batches (every log_interval) | All validation batches |
| **Overhead** | Zero (reuses existing computations) | Forward pass required |
| **Use Case** | Epoch-level monitoring | Model selection, final comparison |

### 2. Final Epoch Handling

```
if epoch == epochs - 1:
    # Run comprehensive_evaluation() directly
    # - Computes val_metrics (StreamingMetrics)
    # - Computes detailed metrics (stratified, AUROC, etc.)
    # - Computes efficiency metrics
    # - Computes resource metrics
    # - Caches result for post-loop reuse
else:
    # Use lightweight evaluate()
    # - Only computes StreamingMetrics
    # - Faster for non-final epochs
```

### 3. Single-Epoch Training Support

For `epochs=1`, the flow correctly handles:
1. Epoch 0 is the final epoch (0 == 1 - 1)
2. `comprehensive_evaluation()` runs directly
3. Result cached in `final_comprehensive_evaluation`
4. Post-loop reuses cached result (no re-computation)

---

## Performance Impact

| Optimization | Compute Savings |
|--------------|-----------------|
| Remove `evaluate(train_loader)` | ~10-15% per epoch |
| Merge `evaluate(val_loader)` into `comprehensive_evaluation()` | ~50% on final epoch |
| Cache `comprehensive_evaluation` result | ~100% post-loop savings |
| **Total (single epoch training)** | **~60-65% evaluation overhead reduction** |

---

## Future Considerations

### Not Implemented (Deferred)

1. **Mid-Epoch Validation**: For longer training (5+ epochs), consider validation every N steps
   - Current: Validation only at epoch end
   - Trade-off: ~10-20% extra compute vs. earlier detection of issues

2. **Validation Loss Trajectory**: Per-batch validation loss plotting
   - Current: Only training loss trajectory saved
   - Recommendation: Epoch-level validation is sufficient for 1-3 epoch training

### Potential Improvements

1. **Streaming Detailed Metrics**: Currently, detailed metrics require collecting 1000 samples
   - Could implement streaming versions for memory efficiency on large datasets

2. **Configurable Validation Frequency**: Add `validation_interval` parameter to `run_single_experiment()`
   - Allow N-step validation instead of epoch-only

---

## Code References

| Function | Lines | Purpose |
|----------|-------|---------|
| `train_epoch()` | ~4741-5139 | Main training loop with batch metrics |
| `_build_epoch_metrics()` | ~10630-10710 | Aggregates train/val metrics into epoch dict |
| `evaluate()` | ~10520-10610 | Lightweight validation (StreamingMetrics) |
| `comprehensive_evaluation()` | ~7799-8041 | Full evaluation with detailed metrics |
| `run_single_experiment()` | ~10767-11234 | Orchestrates training and evaluation flow |
| `MetricsLogger` | ~6589-6711 | JSON-based logging for all metrics |
| `StreamingMetrics` | ~8049-8400 | Memory-efficient metrics computation |

---

## Changelog

### V3.1 (Dec 29, 2025)
- **REMOVED**: `evaluate(train_loader)` call at epoch end
- **ADDED**: Inline `train_eval_metrics` derivation from batch averages
- **ADDED**: Conditional evaluation logic for final vs. non-final epochs
- **ADDED**: `final_comprehensive_evaluation` caching mechanism
- **ADDED**: Logging integration in `train_epoch()` (metrics_logger, logger params)
- **UPDATED**: `_build_epoch_metrics()` docstring to clarify metric sources
- **UPDATED**: `comprehensive_evaluation()` to ensure all required keys for `_build_epoch_metrics()`

### V3.0 (Dec 28, 2025)
- Initial evaluation system analysis
- Identified duplications: `evaluate(train_loader)` vs batch_metrics, `evaluate(val_loader)` vs `comprehensive_evaluation()`
- Documented optimization opportunities



## Evaluation Sampling Behavior Analysis

### Summary Table

| Function | What Gets Sampled? | Full Dataset Possible? | Memory Impact |
|----------|-------------------|------------------------|---------------|
| `evaluate()` | Batches (via `max_batches`) | ✅ Yes (set `max_batches=None`) | ⚠️ Low - uses StreamingMetrics |
| `comprehensive_evaluation()` | Detailed metrics only (1000 samples) | ✅ Core metrics already use full dataset | ⚠️ Medium |
| `compute_batch_metrics_lightweight()` | N/A (single batch) | N/A | ✅ Very Low |
| `StreamingMetrics` | None (accumulates counters) | ✅ Always full | ✅ Very Low |

---

## Detailed Analysis

### 1. `evaluate()` (Lines 5264-5355)

**Sampling Control**: `max_batches` parameter

```python
def evaluate(
    ...
    max_batches: Optional[int] = None,  # ← KEY PARAMETER
):
    batches_to_process = min(num_batches, max_batches) if max_batches else num_batches
```

**Current Usage**:

```4:10:dev/moe/moe_flashattn_3.py
# In run_single_experiment():
eval_max_batches: int = 200,  # Default: 200 batches
...
val_metrics = evaluate(..., max_batches=eval_max_batches)
```

**Memory Design**: 
- ✅ Uses `StreamingMetrics` - **NEVER accumulates predictions**
- ✅ Only scalar counters kept in memory
- ✅ Safe for full dataset evaluation

**Can evaluate full dataset?**: **YES**
- Set `max_batches=None` or remove the parameter
- Memory is **not a concern** because StreamingMetrics only keeps counters

**Why 200 batches default?**
- Historical - was added before StreamingMetrics existed
- Now that StreamingMetrics exists, this limit is **no longer necessary** for memory reasons
- Only reason to keep: **speed** (faster epoch-end evaluation)

---

### 2. `comprehensive_evaluation()` (Lines 7799-8041)

**Two-Tier Sampling Design**:

```python
def comprehensive_evaluation(
    ...
    max_samples_for_detailed_metrics = 1000  # ← ONLY affects detailed metrics
):
```

**What Uses Full Dataset** (via StreamingMetrics):
- ✅ `recall@K`, `micro_recall@K`, `precision@K`, `ndcg@K`
- ✅ `mrr`, `positive_brier`
- ✅ `val_loss`

**What Uses Sampled Data** (1000 samples):
- ⚠️ `compute_primary_task_metrics()` - (redundant, same as StreamingMetrics)
- ⚠️ `compute_loss_metrics()` - BCE loss, per-class variance
- ⚠️ `compute_stratified_metrics()` - Tail/common/rare code performance
- ⚠️ `compute_auroc_auprc()` - AUC-ROC, AUC-PR

**Sampling Logic** (Lines 7931-7944):

```python
# Sample for detailed metrics for memory efficiency
if samples_collected < max_samples_for_detailed_metrics:
    remaining = max_samples_for_detailed_metrics - samples_collected
    to_take = min(len(batch_preds), remaining)
    
    sampled_predictions.append(batch_preds[:to_take].cpu())  # ← Accumulates on CPU
    sampled_targets.extend(valid_targets[:to_take])
    sampled_multihot.append(multihot.cpu())  # ← Accumulates on CPU
    samples_collected += to_take
```

**Memory Bottleneck**:
```python
# These lists grow with samples:
sampled_predictions = []  # [num_samples, vocab_size] tensors
sampled_multihot = []     # [num_samples, vocab_size] tensors
```

For 1000 samples with `vocab_size=6297`:
- `sampled_predictions`: 1000 × 6297 × 4 bytes = **~25 MB**
- `sampled_multihot`: 1000 × 6297 × 1 byte = **~6 MB**
- Total: **~31 MB** - very manageable

**Can increase to full dataset?**: **PARTIALLY**
- StreamingMetrics already uses full dataset
- Detailed metrics (stratified, AUROC) **would require full accumulation**
- Full val set (~100K samples): ~3.1 GB for predictions alone
- **NOT recommended** - would cause CPU OOM

---

### 3. `compute_batch_metrics_lightweight()` (Lines 6286-6460)

**No Sampling** - processes single batch only

- Called during training every `log_interval` batches
- Computes metrics on current batch only
- Memory: O(batch_size × vocab_size) = ~1-2 MB per batch

---

### 4. `StreamingMetrics` (Lines 8085-8403)

**Key Design Principle**: **NEVER accumulates predictions**

```python
class StreamingMetrics:
    """
    Key design principles:
    1. never accumulate full predictions (prevents CPU OOM)
    2. Compute metrics incrementally per-batch
    3. Use only scalar counters, not tensors
    """
```

**State Tracked** (all scalars):

```python
@dataclass
class StreamingMetricsState:
    total_loss: float = 0.0
    num_batches: int = 0
    num_samples: int = 0
    
    # Per-K counters (not per-sample!)
    recall_hits: Dict[int, int]      # e.g., {5: 1234, 10: 2345, 20: 3456}
    recall_total: Dict[int, int]
    micro_recall_hits: Dict[int, int]
    micro_recall_true: Dict[int, int]
    precision_sum: Dict[int, float]
    ndcg_sum: Dict[int, float]
    mrr_sum: float
    positive_brier_sum: float
```

**Memory**: O(1) - constant regardless of dataset size

---

## Recommendations

### Option A: Full Dataset Evaluation (Recommended)

For `evaluate()`, **remove the `max_batches` limit**:

```python
# In run_single_experiment(), change:
eval_max_batches: int = 200,

# To:
eval_max_batches: Optional[int] = None,
```

**Impact**:
- ✅ Full validation set evaluated
- ✅ No memory increase (StreamingMetrics is O(1))
- ⚠️ Slower epoch-end evaluation (e.g., 200 batches → 500+ batches)

### Option B: Keep Partial, Document Clearly

If speed is critical:
- Keep `eval_max_batches=200` for non-final epochs
- Final epoch already uses `comprehensive_evaluation()` which uses full dataset for core metrics

### Option C: Increase `max_samples_for_detailed_metrics` (Careful)

For `comprehensive_evaluation()`:

```python
# Current:
max_samples_for_detailed_metrics = 1000

# Could increase to:
max_samples_for_detailed_metrics = 5000  # ~150 MB CPU memory
# OR
max_samples_for_detailed_metrics = 10000  # ~300 MB CPU memory
```

**Limits**:
- Full val set (100K samples): **~3 GB** - not recommended
- 10K samples is a reasonable maximum

---

## Architectural Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION SAMPLING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ StreamingMetrics (O(1) memory)                                          │ │
│  │  ✅ FULL DATASET - No sampling                                          │ │
│  │  • recall@K, micro_recall@K, precision@K, ndcg@K                        │ │
│  │  • mrr, positive_brier, val_loss                                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│              Used by:        │                                               │
│              ├── evaluate()  ↓  (max_batches controls SPEED, not memory)     │
│              └── comprehensive_evaluation()                                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Detailed Metrics (O(n) memory - SAMPLED)                                │ │
│  │  ⚠️ SAMPLED to 1000 samples                                             │ │
│  │  • compute_stratified_metrics() - rare/common/tail code performance     │ │
│  │  • compute_auroc_auprc() - per-code AUC                                 │ │
│  │  • compute_loss_metrics() - per-class variance                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│              Used by:        │                                               │
│              └── comprehensive_evaluation() only                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Bug/Issue Found

In `comprehensive_evaluation()`, there's an **undefined variable bug** at lines 7981-7991:

```python
if sampled_predictions:
    performance_metrics.update(
        compute_primary_task_metrics(all_predictions, ...)  # ← all_predictions is UNDEFINED
    )
    performance_metrics.update(
        compute_loss_metrics(all_predictions, all_targets_multihot, ...)  # ← both UNDEFINED
    )
```

**Should be**:

```python
if sampled_predictions:
    all_predictions = torch.cat(sampled_predictions)
    all_targets_multihot = torch.cat(sampled_multihot)
    performance_metrics.update(
        compute_primary_task_metrics(all_predictions, sampled_targets, config.target_cd_cnt)
    )
    ...
```

This would cause a `NameError` at runtime. The code must have worked before, so this might be a recent regression.

---

Would you like me to provide the specific code changes to:
1. Enable full-dataset evaluation in `evaluate()`?
2. Fix the undefined variable bug in `comprehensive_evaluation()`?
3. Increase `max_samples_for_detailed_metrics` with proper memory guards?

You'll need to switch to **Agent mode** for me to apply these changes.