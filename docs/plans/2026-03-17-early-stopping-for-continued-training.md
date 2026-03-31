# Early Stopping for Continued Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a composite sub-epoch early stopping strategy to `continue_training_from_checkpoint()` that monitors NDCG@20 at configurable optimizer-step intervals, with warmup protection, train-loss slope detection, and best-model checkpointing.

**Architecture:** A new `EarlyStoppingMonitor` dataclass + class is added to the logging infrastructure cell. The `continue_training_from_checkpoint()` function gains an `EarlyStoppingConfig` parameter and integrates the monitor into its inner training loop via a callback pattern at each optimizer step. Sub-epoch validation uses the existing `evaluate()` function with its `max_batches` parameter for partial validation.

**Tech Stack:** Python 3.10, PyTorch, NumPy, dataclasses. All code lives in `dev/legacy/legacy_full_training.ipynb`.

---

## System Context

Before implementing, read and understand these parts of the notebook:

| Component | Notebook Location (approximate raw lines) | Purpose |
|---|---|---|
| `LossTracker` | lines 325–384 | Batch-level loss tracking, epoch summaries, existing `should_stop_early()` |
| `StreamingMetrics` | lines 557–708 | Computes recall@K, NDCG@K, precision@K, MRR, Brier |
| `evaluate()` | lines 1564–1599 | Full/partial validation — note `max_batches` parameter |
| `train_epoch()` | lines 1296–1538 | Training loop with accumulation, metrics logging |
| `save_checkpoint_local()` | lines 1760–1775 | Checkpoint saving |
| `continue_training_from_checkpoint()` | lines 5543–5708 | **Target function to modify** |
| Execution cell | lines 5717–5728 | Calls `continue_training_from_checkpoint()` |

### Key Constants

- `MICRO_BATCH_SIZE = 32` (per-forward batch)
- `ACCUMULATION_STEPS = 16` (optimizer step every 16 micro-batches)
- `LOG_INTERVAL = 100` (batch metrics logging frequency)
- `GRADIENT_CLIP = 0.25`
- Epoch = 49,349 batches = 3,084 optimizer steps
- Val set = 175,466 samples = 5,484 batches (at micro_batch_size=32)
- Full val takes ~1 hour; 20% val (~1,097 batches) takes ~12 minutes

### Epoch 1 Baseline Metrics (from `exp_round5/exp1_dbcheck`)

| Metric | Value |
|---|---|
| train_loss (mean) | 0.1911 |
| val_loss | 0.0304 |
| val R@10 | 0.573 |
| val uR@10 | 0.299 |
| val NDCG@20 | 0.281 |
| val MRR | 0.486 |
| epoch time | ~7.5 hours |
| optimizer steps | 3,085 |

---

## Task 1: Add `EarlyStoppingConfig` dataclass

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — the cell containing `LossTracker` (cell with `class LossTracker:`)

**Step 1: Write the `EarlyStoppingConfig` dataclass**

Add this **above** the `LossTracker` class in the same cell:

```python
@dataclass
class EarlyStoppingConfig:
    """Configuration for sub-epoch early stopping with clinically-tailored metrics."""
    enabled: bool = True
    primary_metric: str = 'ndcg@20'
    mode: str = 'max'                      # 'max' for NDCG/recall; 'min' for loss
    patience: int = 5                      # validation checks without improvement before stopping
    min_delta: float = 0.001               # minimum improvement to count as "better"
    warmup_steps: int = 1000               # skip early stopping during initial optimizer steps
    val_check_interval: int = 500          # run validation every N optimizer steps
    val_fraction: float = 0.2              # fraction of val set for sub-epoch checks (1.0 at epoch end)
    train_loss_slope_window: int = 500     # optimizer steps window for plateau detection
    train_loss_slope_threshold: float = 1e-5  # minimum loss decrease rate per step
    restore_best: bool = True              # restore best checkpoint when stopping

    def __post_init__(self):
        assert self.mode in ('min', 'max'), f"mode must be 'min' or 'max', got '{self.mode}'"
        assert self.patience >= 1, "patience must be >= 1"
        assert 0.0 < self.val_fraction <= 1.0, "val_fraction must be in (0, 1]"
        assert self.val_check_interval >= 1, "val_check_interval must be >= 1"
```

**Step 2: Verify the cell runs without error**

Run the cell in the notebook. Expected: no output beyond the existing `"Logging infrastructure loaded."` print.

**Step 3: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: add EarlyStoppingConfig dataclass for sub-epoch early stopping"
```

---

## Task 2: Add `EarlyStoppingMonitor` class

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — same cell as Task 1, after `EarlyStoppingConfig`

**Step 1: Write the `EarlyStoppingMonitor` class**

Add this **after** the `EarlyStoppingConfig` dataclass and **before** the `LossTracker` class:

```python
class EarlyStoppingMonitor:
    """Sub-epoch early stopping monitor with clinically-tailored metric tracking.

    Integrates with the existing training loop to provide:
    - Configurable sub-epoch validation at optimizer-step intervals
    - Primary metric tracking (NDCG@20 by default) with patience
    - Warmup period to protect CosineAnnealingLR exploration phase
    - Train loss slope detection as a secondary signal
    - Best-model checkpoint tracking independent of stopping
    """

    def __init__(self, config: EarlyStoppingConfig, logger=None):
        self.config = config
        self.logger = logger

        self._best_metric = float('-inf') if config.mode == 'max' else float('inf')
        self._best_step = 0
        self._best_checkpoint_path = None
        self._checks_without_improvement = 0
        self._total_checks = 0
        self._should_stop = False

        self._val_history = []     # list of (global_step, metric_value, full_metrics_dict)
        self._train_loss_buffer = []  # list of (global_step, smoothed_loss)

    def _is_improvement(self, current: float) -> bool:
        if self.config.mode == 'max':
            return current > self._best_metric + self.config.min_delta
        return current < self._best_metric - self.config.min_delta

    def should_validate(self, global_step: int) -> bool:
        """Check if we should run validation at this optimizer step."""
        if not self.config.enabled:
            return False
        return global_step > 0 and global_step % self.config.val_check_interval == 0

    def record_validation(self, global_step: int, metrics: dict) -> dict:
        """Record a validation result and return a status dict.

        Returns dict with keys: improved, should_stop, metric_value,
        best_metric, checks_without_improvement, in_warmup.
        """
        metric_value = metrics.get(self.config.primary_metric, None)
        if metric_value is None:
            available = [k for k in metrics if 'ndcg' in k or 'recall' in k or 'loss' in k]
            raise KeyError(
                f"Primary metric '{self.config.primary_metric}' not in validation results. "
                f"Available metric-like keys: {available}"
            )

        self._val_history.append((global_step, metric_value, metrics))
        self._total_checks += 1

        in_warmup = global_step < self.config.warmup_steps
        improved = self._is_improvement(metric_value)

        if improved:
            self._best_metric = metric_value
            self._best_step = global_step
            self._checks_without_improvement = 0
        else:
            if not in_warmup:
                self._checks_without_improvement += 1

        if not in_warmup and self._checks_without_improvement >= self.config.patience:
            self._should_stop = True

        status = {
            'improved': improved,
            'should_stop': self._should_stop,
            'metric_value': metric_value,
            'best_metric': self._best_metric,
            'best_step': self._best_step,
            'checks_without_improvement': self._checks_without_improvement,
            'in_warmup': in_warmup,
            'total_checks': self._total_checks,
        }

        if self.logger:
            phase = "WARMUP" if in_warmup else "ACTIVE"
            marker = " ***NEW BEST***" if improved else ""
            self.logger.info(
                f"[EarlyStop|{phase}] step={global_step} "
                f"{self.config.primary_metric}={metric_value:.4f} "
                f"best={self._best_metric:.4f}@step{self._best_step} "
                f"patience={self._checks_without_improvement}/{self.config.patience}"
                f"{marker}"
            )

        return status

    def record_train_loss(self, global_step: int, smoothed_loss: float):
        """Record smoothed training loss for slope detection."""
        self._train_loss_buffer.append((global_step, smoothed_loss))

    def detect_train_loss_plateau(self) -> bool:
        """Check if training loss has plateaued using the configured window.

        Returns True if the loss slope over the last `train_loss_slope_window`
        optimizer steps is below the threshold.
        """
        window = self.config.train_loss_slope_window
        if len(self._train_loss_buffer) < window:
            return False

        recent = self._train_loss_buffer[-window:]
        first_loss = recent[0][1]
        last_loss = recent[-1][1]
        step_span = recent[-1][0] - recent[0][0]
        if step_span == 0:
            return False

        slope = (first_loss - last_loss) / step_span
        return slope < self.config.train_loss_slope_threshold

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def best_metric(self) -> float:
        return self._best_metric

    @property
    def best_step(self) -> int:
        return self._best_step

    @property
    def best_checkpoint_path(self) -> str:
        return self._best_checkpoint_path

    @best_checkpoint_path.setter
    def best_checkpoint_path(self, path: str):
        self._best_checkpoint_path = path

    def get_summary(self) -> dict:
        """Return a summary dict for logging/serialization."""
        return {
            'enabled': self.config.enabled,
            'primary_metric': self.config.primary_metric,
            'best_metric': self._best_metric,
            'best_step': self._best_step,
            'total_checks': self._total_checks,
            'stopped_early': self._should_stop,
            'checks_without_improvement': self._checks_without_improvement,
            'val_history': [
                {'step': s, self.config.primary_metric: v}
                for s, v, _ in self._val_history
            ],
        }
```

**Step 2: Verify the cell runs without error**

Run the cell. Expected: `"Logging infrastructure loaded."` with no errors.

**Step 3: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: add EarlyStoppingMonitor class with sub-epoch validation and metric tracking"
```

---

## Task 3: Modify `train_epoch()` to support a step-level callback

The existing `train_epoch()` function processes all batches in a tight loop with no hook for mid-epoch validation. We need to add an **optional callback** parameter that fires at each optimizer step, allowing the outer `continue_training_from_checkpoint()` to run validation mid-epoch.

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — the cell containing `def train_epoch()`

**Step 1: Add the `on_optimizer_step` callback parameter**

In the `train_epoch()` function signature, add one parameter after `scaler`:

```python
def train_epoch(
    model, dataloader, optimizer, criterion,
    epoch: int = 0,
    log_interval: int = 100,
    global_step: int = 0,
    loss_tracker: Optional[LossTracker] = None,
    metrics_logger: Optional[MetricsLogger] = None,
    logger: Optional[logging.Logger] = None,
    gradient_tier_analyzer: Optional[GradientTierAnalyzer] = None,
    accumulation_steps: int = 1,
    track_gpu_memory: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    on_optimizer_step: Optional[Callable] = None,
) -> Dict[str, Any]:
```

**Step 2: Call the callback after each optimizer step**

Inside the training loop, there are two places where optimizer steps happen:

1. **Inside the main loop** — where `accumulation_counter == accumulation_steps` (around the `optimizer.step()` block inside the `for batch_idx` loop). After the `global_step += 1` line, add:

```python
                global_step += 1
                # --- Sub-epoch callback (e.g., for early stopping validation) ---
                if on_optimizer_step is not None:
                    stop_signal = on_optimizer_step(
                        global_step=global_step,
                        model=model,
                        loss_tracker=loss_tracker,
                        epoch=epoch,
                        batch_idx=batch_idx,
                    )
                    if stop_signal:
                        # Early stop requested mid-epoch
                        epoch_time = time.time() - epoch_start_time
                        avg_loss = total_loss / max(batch_idx + 1, 1)
                        epoch_metrics = {
                            'train_loss': avg_loss,
                            'aux_loss': 0.0,
                            'global_step': global_step,
                            'num_batches': batch_idx + 1,
                            'epoch_time_s': epoch_time,
                            'data_load_time_s': data_load_time,
                            'forward_time_s': forward_time,
                            'backward_time_s': backward_time,
                            'optimizer_time_s': optimizer_time,
                            'samples_processed': samples_processed,
                            'throughput_samples_per_sec': samples_processed / epoch_time if epoch_time > 0 else 0,
                            'throughput_batches_per_sec': (batch_idx + 1) / epoch_time if epoch_time > 0 else 0,
                            'early_stopped': True,
                            'stopped_at_batch': batch_idx,
                        }
                        if torch.cuda.is_available():
                            epoch_metrics['gpu_memory_peak_gib'] = torch.cuda.max_memory_allocated() / 1024**3
                            epoch_metrics['gpu_memory_allocated_gib'] = torch.cuda.memory_allocated() / 1024**3
                        loss_summary = loss_tracker.get_epoch_summary()
                        epoch_metrics.update(loss_summary)
                        if batch_metrics_buffer:
                            for key in batch_metrics_buffer[0].keys():
                                epoch_metrics[f'train_{key}'] = np.mean([m[key] for m in batch_metrics_buffer])
                        if gradient_tier_analyzer is not None:
                            tier_epoch = gradient_tier_analyzer.aggregate_epoch()
                            epoch_metrics.update(tier_epoch)
                        print(f'  EARLY STOP at batch {batch_idx}/{num_batches}. '
                              f'Avg loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s')
                        return epoch_metrics
```

2. **After the main loop** (the trailing `if accumulation_counter > 0` block) — add the same callback call after `global_step += 1` there too, but the stop signal won't matter since the epoch is ending anyway. Just add:

```python
        global_step += 1
        if on_optimizer_step is not None:
            on_optimizer_step(
                global_step=global_step,
                model=model,
                loss_tracker=loss_tracker,
                epoch=epoch,
                batch_idx=num_batches - 1,
            )
```

**Step 3: Verify existing training loop cell still works**

The original training loop (Section 6) does NOT pass `on_optimizer_step`, so it defaults to `None` and the new code is never executed — backward compatible.

**Step 4: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: add on_optimizer_step callback to train_epoch for sub-epoch early stopping"
```

---

## Task 4: Rewrite `continue_training_from_checkpoint()` with early stopping integration

This is the main integration task. The function gains the `EarlyStoppingConfig` parameter and creates the callback that connects everything.

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — the cell containing `def continue_training_from_checkpoint()`

**Step 1: Update the function signature**

```python
def continue_training_from_checkpoint(
    checkpoint_path: str,
    train_loader,
    val_loader,
    additional_epochs: int = 5,
    experiment_round: str = None,
    exp_name: str = EXP_NAME,
    early_stopping: EarlyStoppingConfig = None,
):
```

**Step 2: Add early stopping initialization after the logging setup block (after `cont_gradient_tier_analyzer` setup)**

```python
    # --- 4b. Early stopping ---
    es_monitor = None
    if early_stopping and early_stopping.enabled:
        es_monitor = EarlyStoppingMonitor(config=early_stopping, logger=cont_logger)

        max_val_batches = None
        if early_stopping.val_fraction < 1.0:
            total_val_batches = len(val_loader)
            max_val_batches = max(1, int(total_val_batches * early_stopping.val_fraction))

        print(f"\nEarly Stopping Configuration:")
        print(f"  Primary metric: {early_stopping.primary_metric} (mode={early_stopping.mode})")
        print(f"  Patience: {early_stopping.patience} checks")
        print(f"  Warmup: {early_stopping.warmup_steps} optimizer steps")
        print(f"  Validation interval: every {early_stopping.val_check_interval} optimizer steps")
        print(f"  Validation fraction: {early_stopping.val_fraction:.0%} "
              f"({max_val_batches or len(val_loader)} batches)")
        print(f"  Train loss plateau window: {early_stopping.train_loss_slope_window} steps")
        print(f"  Restore best: {early_stopping.restore_best}")
```

**Step 3: Define the optimizer-step callback inside the function (before the training loop)**

```python
    # --- Sub-epoch validation callback ---
    def _on_optimizer_step(global_step, model, loss_tracker, epoch, batch_idx):
        """Called after every optimizer step. Returns True to stop training."""
        if es_monitor is None:
            return False

        # Record train loss for plateau detection
        recent = loss_tracker.get_recent_losses(n=100)
        if recent:
            smoothed = sum(recent) / len(recent)
            es_monitor.record_train_loss(global_step, smoothed)

        if not es_monitor.should_validate(global_step):
            return False

        # Run partial validation
        print(f"\n    --- Sub-epoch validation at step {global_step} (batch {batch_idx}) ---")
        val_metrics = evaluate(
            model, val_loader, cont_criterion,
            max_batches=max_val_batches,
            verbose=False,
            use_amp=True,
        )

        status = es_monitor.record_validation(global_step, val_metrics)

        print(f"    {early_stopping.primary_metric}={status['metric_value']:.4f} | "
              f"best={status['best_metric']:.4f}@step{status['best_step']} | "
              f"patience={status['checks_without_improvement']}/{early_stopping.patience} | "
              f"{'WARMUP' if status['in_warmup'] else 'ACTIVE'}")

        # Save best checkpoint
        if status['improved']:
            best_path = os.path.join(cont_checkpoint_dir, 'checkpoint_best_es.pt')
            save_checkpoint_local(
                model, cont_optimizer, cont_scheduler, epoch, val_metrics.get('val_loss', 0),
                best_path,
            )
            es_monitor.best_checkpoint_path = best_path
            print(f"    *** New best checkpoint saved: {best_path}")

        # Train loss plateau warning
        if es_monitor.detect_train_loss_plateau():
            print(f"    WARNING: Train loss plateau detected (slope < {early_stopping.train_loss_slope_threshold})")

        if status['should_stop']:
            print(f"\n    EARLY STOPPING TRIGGERED at step {global_step}")
            print(f"    Best {early_stopping.primary_metric}: {status['best_metric']:.4f} at step {status['best_step']}")
            return True

        return False
```

**Step 4: Pass the callback into `train_epoch()` inside the epoch loop**

Replace the existing `train_epoch()` call in the epoch loop:

```python
        train_metrics = train_epoch(
            cont_model, train_loader, cont_optimizer, cont_criterion,
            epoch=epoch, log_interval=LOG_INTERVAL,
            global_step=cont_global_step,
            loss_tracker=cont_loss_tracker,
            metrics_logger=cont_metrics_logger,
            logger=cont_logger,
            gradient_tier_analyzer=cont_gradient_tier_analyzer,
            accumulation_steps=ACCUMULATION_STEPS,
            track_gpu_memory=False,
            scaler=cont_scaler,
            on_optimizer_step=_on_optimizer_step if es_monitor else None,
        )
```

**Step 5: Add early-stop break logic after `train_epoch()` returns**

After `train_loss = train_metrics['train_loss']`, add:

```python
        # Check if train_epoch stopped mid-epoch due to early stopping
        if train_metrics.get('early_stopped', False):
            print(f"\n  Training stopped early at epoch {epoch + 1}, batch {train_metrics.get('stopped_at_batch', '?')}")
            # Still run a full validation for the final record
            val_metrics = evaluate(cont_model, val_loader, cont_criterion, verbose=True, use_amp=True)
            val_loss = val_metrics['val_loss']
            epoch_time = time.time() - epoch_start
            epoch_entry = {**train_metrics, 'lr': cont_optimizer.param_groups[0]['lr'],
                           'epoch_time_s': epoch_time}
            for k, v in val_metrics.items():
                epoch_entry[k] = v
            cont_history.append(epoch_entry)
            cont_metrics_logger.log_epoch(epoch + 1, epoch_entry)
            cont_metrics_logger.save()
            break
```

**Step 6: After the epoch loop ends, add best-model restoration logic**

Before the `total_time = time.time() - start_time` line:

```python
    # --- Restore best checkpoint if early stopped ---
    if es_monitor and es_monitor.should_stop and early_stopping.restore_best:
        best_path = es_monitor.best_checkpoint_path
        if best_path and os.path.exists(best_path):
            print(f"\nRestoring best checkpoint from {best_path}")
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            if parallel and isinstance(cont_model, nn.DataParallel):
                cont_model.module.load_state_dict(best_ckpt['model'])
            else:
                cont_model.load_state_dict(best_ckpt['model'])
            print(f"Restored best model (step {es_monitor.best_step}, "
                  f"{early_stopping.primary_metric}={es_monitor.best_metric:.4f})")
```

**Step 7: Add early stopping summary to the final print block**

After the training history table, add:

```python
    if es_monitor:
        es_summary = es_monitor.get_summary()
        print(f"\nEarly Stopping Summary:")
        print(f"  Stopped early: {es_summary['stopped_early']}")
        print(f"  Total validation checks: {es_summary['total_checks']}")
        print(f"  Best {early_stopping.primary_metric}: {es_summary['best_metric']:.4f} at step {es_summary['best_step']}")
        # Save summary to log dir
        es_summary_path = os.path.join(cont_log_dir, exp_name, 'early_stopping_summary.json')
        with open(es_summary_path, 'w') as f:
            json.dump(es_summary, f, indent=2)
        print(f"  Summary saved to: {es_summary_path}")
```

**Step 8: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: integrate sub-epoch early stopping into continue_training_from_checkpoint"
```

---

## Task 5: Update the execution cell with early stopping configuration

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — the cell that calls `continue_training_from_checkpoint()`

**Step 1: Update the execution cell**

Replace the entire execution cell with:

```python
# --- Execute: continue training with early stopping ---
CONTINUE_CHECKPOINT = 'logs/legacy_2026-03-16_06-21-09/legacy_replication/checkpoints/checkpoint_best.pt'
ADDITIONAL_EPOCHS = 5

# Early stopping: monitor NDCG@20 every 500 optimizer steps
# with warmup of 1000 steps (protects CosineAnnealingLR exploration phase)
es_config = EarlyStoppingConfig(
    enabled=True,
    primary_metric='ndcg@20',
    mode='max',
    patience=5,
    min_delta=0.001,
    warmup_steps=1000,
    val_check_interval=500,
    val_fraction=0.2,
    train_loss_slope_window=500,
    train_loss_slope_threshold=1e-5,
    restore_best=True,
)

cont_model, cont_optimizer, cont_scheduler, cont_history = continue_training_from_checkpoint(
    checkpoint_path=CONTINUE_CHECKPOINT,
    train_loader=train_loader,
    val_loader=val_loader,
    additional_epochs=ADDITIONAL_EPOCHS,
    experiment_round=EXPERIMENT_ROUND,
    early_stopping=es_config,
)
```

**Step 2: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: configure early stopping for continued training execution"
```

---

## Task 6: Add `Callable` import if missing

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — the imports cell

**Step 1: Check if `Callable` is already imported**

Search the notebook for `from typing import` — verify `Callable` is in the import list. If not, add it.

The existing import line likely reads:
```python
from typing import Dict, List, Optional, Tuple, Any
```

Update to:
```python
from typing import Callable, Dict, List, Optional, Tuple, Any
```

**Step 2: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "fix: add Callable to typing imports for train_epoch callback"
```

---

## Task 7: Verify end-to-end correctness (dry run)

This task ensures the implementation is syntactically correct and the callback wiring is sound.

**Step 1: Verify all cells parse without error**

In the notebook, run all cells from the imports through the `continue_training_from_checkpoint()` definition (do NOT execute the actual training call). This validates:
- `EarlyStoppingConfig` and `EarlyStoppingMonitor` are defined
- `train_epoch()` accepts `on_optimizer_step` parameter
- `continue_training_from_checkpoint()` accepts `early_stopping` parameter
- The callback closure correctly references `es_monitor`, `cont_criterion`, `val_loader`, etc.

**Step 2: Verify backward compatibility**

The original training loop (Section 6) does NOT pass `on_optimizer_step` or `early_stopping` — confirm these default to `None` and no code paths are affected.

**Step 3: Commit all remaining changes**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "chore: verify early stopping integration compiles and is backward-compatible"
```

---

## Design Rationale Reference

### Why NDCG@20 (not val_loss)?

BCE loss across 6,297 targets is dominated by negative predictions (~99.5% of targets per sample). NDCG@20 directly measures whether the model ranks correct clinical codes at the top — the actual clinical value (supported by arXiv:2601.15546, Naik et al., 2026).

### Why val_check_interval=500?

500 optimizer steps × 16 micro-batches × ~0.53s/batch ≈ 1.2 hours between checks. This gives 6 validation checkpoints per epoch (vs. 1 at epoch level), catching saturation within ~1 hour.

### Why val_fraction=0.2 for sub-epoch?

Full validation = 5,484 batches ≈ 1 hour. 20% = 1,097 batches ≈ 12 minutes — a reasonable overhead for early detection. Full validation still runs at epoch boundaries.

### Why warmup_steps=1000?

~1000 optimizer steps ≈ 2.4 hours into continued training. This protects the CosineAnnealingLR high-LR exploration phase where metrics may temporarily stagnate or regress.

### Why patience=5?

5 checks × 500 steps/check = 2,500 steps without improvement. At 6 checks/epoch, this means early stopping won't fire until at least ~40% into an epoch of stagnation, balancing responsiveness with noise tolerance.

---

## Risk Analysis

| Risk | Mitigation |
|---|---|
| Partial val may have high variance | `val_fraction=0.2` gives 35K samples — sufficient for stable NDCG@20 |
| Warmup too short | 1000 steps covers first ~32% of an epoch; adjustable |
| Callback overhead per step | Only `should_validate()` check runs at each step (cheap boolean) |
| Model in eval mode during training | `evaluate()` handles `model.eval()` / `model.train()` internally; verified |
| CosineAnnealingLR interaction | Scheduler `.step()` is called at epoch boundaries, not affected by mid-epoch stop |

---

Plan complete and saved to `docs/plans/2026-03-17-early-stopping-for-continued-training.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
