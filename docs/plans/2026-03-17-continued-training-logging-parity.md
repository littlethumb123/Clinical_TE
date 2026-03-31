# Continued Training Logging Parity — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `continue_training_from_checkpoint()` so that continued training produces *exactly* the same logging artifacts as the first-round training loop (Section 6), writing into the **same** log directory as the original run.

**Architecture:** The continued training function currently creates its own separate `cont_log_dir` and skips 4 critical logging calls that the first-round loop makes. We modify the function to: (a) accept the original checkpoint's log directory so artifacts land beside the first-round files, (b) add the missing `log_config`, `save_trajectory`, `save_final_results`, and `logger.info` epoch-summary calls, and (c) update the execution cell to pass the correct original log path. Checkpoints go into the same `checkpoints/` folder, updating `checkpoint_best.pt` and `checkpoint_latest.pt`.

**Tech Stack:** Python 3.10, PyTorch, JSON, dataclasses. All code lives in `dev/legacy/legacy_full_training.ipynb`.

---

## System Context — What First-Round Training Produces

The first-round training loop (Section 6, inline in the notebook) writes these artifacts to `logs/{EXPERIMENT_ROUND}/legacy_replication/`:

| Artifact | Written By | When |
|---|---|---|
| `config.json` | `metrics_logger.log_config(config_dict)` → `metrics_logger.save()` | Once at setup, saved each epoch |
| `epoch_metrics.json` | `metrics_logger.log_epoch(epoch+1, epoch_entry)` → `metrics_logger.save()` | Each epoch |
| `batch_metrics.json` | `metrics_logger.log_batch(epoch, batch_idx, batch_entry)` → `metrics_logger.save()` | Each epoch (already works via `train_epoch`) |
| `final_results.json` | `metrics_logger.save_final_results(final_results)` | Once after all epochs |
| `loss_trajectory_epoch{N}.json` | `loss_tracker.save_trajectory(filepath)` | Each epoch |
| `training.log` | `setup_experiment_logging()` + `logger.info()` / `logger.debug()` | Continuous |
| `checkpoints/checkpoint_best.pt` | `save_checkpoint_local()` when val_loss improves | Conditional |
| `checkpoints/checkpoint_epoch{N}.pt` | `save_checkpoint_local()` | Each epoch |
| `checkpoints/checkpoint_latest.pt` | `save_checkpoint_local()` | Each epoch |

### Epoch entry schema (first-round)

```python
epoch_entry = {
    **train_metrics,     # train_loss, aux_loss, global_step, num_batches, epoch_time_s,
                         # data/forward/backward/optimizer time, throughput, GPU memory,
                         # train_loss_mean/std/min/max/first/last/improvement/smoothed,
                         # train_recall@K, train_precision@K, train_ndcg@20, train_positive_brier,
                         # gradient tier summary
    'lr': optimizer.param_groups[0]['lr'],
    'epoch_time_s': epoch_time,      # wall clock for train + val
    'val_time_s': val_time,          # validation wall clock
    'train_time_s': train_metrics.get('epoch_time_s', 0),  # training wall clock
    # val metrics merged in:
    'val_loss': ..., 'recall@10': ..., 'micro_recall@10': ..., 'ndcg@20': ..., 'mrr': ..., etc.
}
```

### Final results schema (first-round)

```python
final_results = {
    'config': config_dict,
    'best_val_loss': best_val_loss,
    'total_time_s': total_time,
    'training_history': training_history,   # list of epoch_entry dicts
    'gradient_tier_diagnosis': gradient_tier_analyzer.get_diagnosis(),
    'efficiency': compute_training_time_metrics(...),
    'cost': compute_cost_metrics(...),
    'gpu_memory_peak_gib': ...,
}
```

---

## Logging Gap Analysis: `continue_training_from_checkpoint()` vs First-Round

| What | First-Round | `continue_training_from_checkpoint()` | Gap? |
|---|---|---|---|
| **`metrics_logger.log_config()`** | Yes (config_dict with model/data/optimizer info) | **NO** | **MISSING** |
| **`metrics_logger.save_final_results()`** | Yes (config + best_val + history + efficiency + cost + gradient) | **NO** | **MISSING** |
| **`loss_tracker.save_trajectory()`** | Yes (per epoch → `loss_trajectory_epoch{N}.json`) | **NO** | **MISSING** |
| **`logger.info()` epoch summary** | Yes (`Epoch N: train_loss=..., val_loss=..., ...`) | **NO** | **MISSING** |
| **`epoch_entry` includes `val_time_s` and `train_time_s`** | Yes | **NO** | **MISSING** |
| **`logger.info()` config at start** | Yes (`Config: {json}`) | **NO** | **MISSING** |
| **`setup_experiment_logging()` with `resume=True`** | N/A (first run) | Calls it but default `resume=False` used | **BUG** |
| **Log dir = same as original** | `logs/{EXPERIMENT_ROUND}/...` | Creates own `cont_log_dir` (depends on `experiment_round` param) | **PATH MISMATCH** — need to accept original `log_dir` directly |
| **Checkpoint path = same folder** | `logs/.../checkpoints/` | Own `cont_checkpoint_dir` | **PATH MISMATCH** |
| **`MetricsLogger` with `resume=True`** | N/A | Yes, but only if log_dir matches original | Works *if* paths align |
| **Timing accumulators** (`total_data_load_time`, etc.) | Yes — feeds `compute_training_time_metrics` | **NO** | **MISSING** |
| **`metrics_logger.log_batch()`** | Yes (inside `train_epoch`) | Yes (inside `train_epoch`) | OK |
| **`metrics_logger.log_epoch()`** | Yes | Yes | OK |
| **`metrics_logger.save()`** | Yes (each epoch) | Yes (each epoch) | OK |
| **`save_checkpoint_local()`** | best + epoch + latest | best + epoch + latest | OK |

---

## Task 1: Add `log_dir` parameter and fix path construction ⭐ LOGGING FIX

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — cell 53 (`continue_training_from_checkpoint`)

**Why:** The function currently constructs its own `cont_log_dir` from `experiment_round`, which may not match the original log directory. The user wants artifacts to go into the *same* folder as the first-round run (e.g., `logs/legacy_2026-03-16_06-21-09`). Adding a `log_dir` parameter gives direct control.

**Step 1: Update the function signature**

Change:
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

To:
```python
def continue_training_from_checkpoint(
    checkpoint_path: str,
    train_loader,
    val_loader,
    additional_epochs: int = 5,
    experiment_round: str = None,
    exp_name: str = EXP_NAME,
    early_stopping: EarlyStoppingConfig = None,
    log_dir: str = None,
):
```

**Step 2: Update `cont_log_dir` construction**

Replace:
```python
    # --- 4. Setup logging ---
    if experiment_round:
        cont_log_dir = os.path.join('logs', experiment_round)
    else:
        cont_log_dir = os.path.join('logs', f"legacy_continued_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
```

With:
```python
    # --- 4. Setup logging (reuse original log dir when provided) ---
    if log_dir:
        cont_log_dir = log_dir
    elif experiment_round:
        cont_log_dir = os.path.join('logs', experiment_round)
    else:
        cont_log_dir = os.path.join('logs', f"legacy_continued_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
```

**Step 3: Fix `setup_experiment_logging` to pass `resume=True`**

Replace:
```python
    cont_logger = setup_experiment_logging(exp_name, cont_log_dir)
```

With:
```python
    cont_logger = setup_experiment_logging(exp_name, cont_log_dir, resume=True)
```

This appends to the existing `training.log` instead of overwriting it.

**Step 4: Verify**

The `cont_checkpoint_dir` already derives from `cont_log_dir`:
```python
cont_checkpoint_dir = os.path.join(cont_log_dir, exp_name, 'checkpoints')
```
So if `log_dir` is `'logs/legacy_2026-03-16_06-21-09'`, checkpoints go to `logs/legacy_2026-03-16_06-21-09/legacy_replication/checkpoints/` — same folder as the first run.

**Step 5: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: add log_dir parameter to continue_training for artifact co-location"
```

---

## Task 2: Log continued training config ⭐ LOGGING FIX

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — cell 53 (`continue_training_from_checkpoint`)

**Why:** First-round calls `metrics_logger.log_config(config_dict)` and `logger.info(f"Config: {json.dumps(config_dict, indent=2)}")`. Continued training does neither.

**Step 1: Add config logging after the logging setup block**

After the line `cont_loss_tracker = LossTracker(window_size=100)`, add:

```python
    # --- 4a. Log continued training config ---
    cont_config_dict = {
        'mode': 'continued_training',
        'checkpoint_path': checkpoint_path,
        'start_epoch': start_epoch,
        'additional_epochs': additional_epochs,
        'total_epochs': total_epochs,
        'model_type': 'legacy_replication',
        'batch_size': batch_size,
        'micro_batch_size': MICRO_BATCH_SIZE,
        'accumulation_steps': ACCUMULATION_STEPS,
        'effective_batch_size': MICRO_BATCH_SIZE * ACCUMULATION_STEPS,
        'embedding_size': embedding_size,
        'nhid': nhid, 'nhead': nhead, 'nlayers': nlayers, 'ndropout': ndropout,
        'cd_cnt': cd_cnt, 'target_cd_cnt': target_cd_cnt, 'lob_vocab': lob_vocab,
        'len_dy': len_dy, 'len_cd': len_cd,
        'optimizer': 'SGD', 'learning_rate': LEARNING_RATE, 'momentum': 0.9,
        'scheduler': 'CosineAnnealingLR', 'T_max': total_epochs,
        'loss': 'BCEWithLogitsLoss', 'gradient_clip': GRADIENT_CLIP,
        'parallel': parallel, 'num_gpus': torch.cuda.device_count(),
        'mixed_precision': True,
        'prev_val_loss': prev_val_loss,
    }
    if early_stopping and early_stopping.enabled:
        cont_config_dict['early_stopping'] = {
            'primary_metric': early_stopping.primary_metric,
            'mode': early_stopping.mode,
            'patience': early_stopping.patience,
            'min_delta': early_stopping.min_delta,
            'warmup_steps': early_stopping.warmup_steps,
            'val_check_interval': early_stopping.val_check_interval,
            'val_fraction': early_stopping.val_fraction,
        }
    cont_metrics_logger.log_config(cont_config_dict)
    cont_logger.info(f"Continued Training Config: {json.dumps(cont_config_dict, indent=2)}")
```

**Step 2: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: log config at continued training start (parity with first-round)"
```

---

## Task 3: Add `val_time_s` / `train_time_s` to epoch entry and `logger.info` per epoch ⭐ LOGGING FIX

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — cell 53 (`continue_training_from_checkpoint`)

**Why:** First-round records `val_time_s` and `train_time_s` in `epoch_entry` and logs `logger.info(f"Epoch {epoch+1}: train_loss=..., val_loss=..., R@10=..., throughput=..., time=...")` after each epoch. Continued training does neither.

**Step 1: Add `val_start` timing around validation**

In the normal (non-early-stop) path, wrap `evaluate()`:

Replace:
```python
        val_metrics = evaluate(cont_model, val_loader, cont_criterion, verbose=True, use_amp=True)
        val_loss = val_metrics['val_loss']
```

With:
```python
        val_start = time.time()
        val_metrics = evaluate(cont_model, val_loader, cont_criterion, verbose=True, use_amp=True)
        val_time = time.time() - val_start
        val_loss = val_metrics['val_loss']
```

**Step 2: Add `val_time_s` and `train_time_s` to epoch_entry**

Replace (in the normal path):
```python
        epoch_time = time.time() - epoch_start
        epoch_entry = {**train_metrics, 'lr': cont_optimizer.param_groups[0]['lr'],
                       'epoch_time_s': epoch_time}
```

With:
```python
        epoch_time = time.time() - epoch_start
        epoch_entry = {**train_metrics, 'lr': cont_optimizer.param_groups[0]['lr'],
                       'epoch_time_s': epoch_time,
                       'val_time_s': val_time,
                       'train_time_s': train_metrics.get('epoch_time_s', 0)}
```

**Step 3: Do the same for the early-stop break path**

In the `if train_metrics.get('early_stopped', False):` block:

Replace:
```python
            val_metrics = evaluate(cont_model, val_loader, cont_criterion, verbose=True, use_amp=True)
            val_loss = val_metrics['val_loss']
            epoch_time = time.time() - epoch_start
            epoch_entry = {**train_metrics, 'lr': cont_optimizer.param_groups[0]['lr'],
                           'epoch_time_s': epoch_time}
```

With:
```python
            val_start = time.time()
            val_metrics = evaluate(cont_model, val_loader, cont_criterion, verbose=True, use_amp=True)
            val_time = time.time() - val_start
            val_loss = val_metrics['val_loss']
            epoch_time = time.time() - epoch_start
            epoch_entry = {**train_metrics, 'lr': cont_optimizer.param_groups[0]['lr'],
                           'epoch_time_s': epoch_time,
                           'val_time_s': val_time,
                           'train_time_s': train_metrics.get('epoch_time_s', 0)}
```

**Step 4: Add `logger.info()` epoch summary after `scheduler.step()` in the normal path**

After `cont_scheduler.step()`, add:

```python
        cont_logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                         f"R@10={val_metrics.get('recall@10', 0):.3f}, "
                         f"throughput={train_metrics.get('throughput_samples_per_sec', 0):.1f} samples/s, "
                         f"time={epoch_time:.0f}s")
```

And similarly in the early-stop break path, before the `break`:

```python
            cont_logger.info(f"Epoch {epoch+1} (EARLY STOPPED): train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                             f"R@10={val_metrics.get('recall@10', 0):.3f}, "
                             f"stopped_at_batch={train_metrics.get('stopped_at_batch', '?')}")
```

**Step 5: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: add val_time_s/train_time_s and epoch logger.info to continued training"
```

---

## Task 4: Save loss trajectory per epoch ⭐ LOGGING FIX

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — cell 53 (`continue_training_from_checkpoint`)

**Why:** First-round calls `loss_tracker.save_trajectory(filepath=os.path.join(effective_log_dir, EXP_NAME, f'loss_trajectory_epoch{epoch}.json'))` after each epoch. Continued training never calls it.

**Step 1: Add trajectory save in the normal epoch path**

After `cont_metrics_logger.log_epoch(epoch + 1, epoch_entry)`, add:

```python
        cont_loss_tracker.save_trajectory(
            filepath=os.path.join(cont_log_dir, exp_name, f'loss_trajectory_epoch{epoch}.json')
        )
```

**Step 2: Add trajectory save in the early-stop break path**

Before the `break` in the early-stop block, add:

```python
            cont_loss_tracker.save_trajectory(
                filepath=os.path.join(cont_log_dir, exp_name, f'loss_trajectory_epoch{epoch}.json')
            )
```

**Step 3: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: save loss trajectory per epoch in continued training"
```

---

## Task 5: Save final results after continued training completes ⭐ LOGGING FIX

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — cell 53 (`continue_training_from_checkpoint`)

**Why:** First-round calls `metrics_logger.save_final_results(final_results)` with config, best_val_loss, training_history, efficiency, cost, gradient tier diagnosis, and GPU peak memory. Continued training does none of this.

**Step 1: Add timing accumulators before the epoch loop**

After `start_time = time.time()`, add:

```python
    total_data_load_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
```

**Step 2: Accumulate timing in the epoch loop**

After `train_loss = train_metrics['train_loss']`, add:

```python
        total_data_load_time += train_metrics.get('data_load_time_s', 0)
        total_forward_time += train_metrics.get('forward_time_s', 0)
        total_backward_time += train_metrics.get('backward_time_s', 0)
```

(Do this in both the normal path and the early-stop path — in the early-stop path, add it before the val/break block.)

**Step 3: Add `save_final_results` after the early-stopping summary block, before the `return`**

After the early stopping summary block (the `if es_monitor:` block) and before `return cont_model, ...`, add:

```python
    # --- Save final results (parity with first-round training) ---
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    actual_epochs = len(cont_history)
    num_samples_total = sum(h.get('samples_processed', 0) for h in cont_history)
    num_tokens_total = num_samples_total * len_dy

    time_metrics = compute_training_time_metrics(
        total_train_time=total_time,
        num_epochs=actual_epochs,
        num_samples=num_samples_total,
        num_tokens=num_tokens_total,
        batch_size=MICRO_BATCH_SIZE * ACCUMULATION_STEPS,
        data_load_time=total_data_load_time,
        forward_time=total_forward_time,
        backward_time=total_backward_time,
    )

    cost_metrics = compute_cost_metrics(
        training_time_sec=total_time,
        num_epochs=actual_epochs,
        gpu_type="T4",
        num_gpus=num_gpus,
    )

    cont_final_results = {
        'config': cont_config_dict,
        'best_val_loss': cont_best_val_loss,
        'total_time_s': total_time,
        'training_history': cont_history,
        'gradient_tier_diagnosis': cont_gradient_tier_analyzer.get_diagnosis() if cont_gradient_tier_analyzer else {},
        'efficiency': time_metrics,
        'cost': cost_metrics,
    }
    if torch.cuda.is_available():
        cont_final_results['gpu_memory_peak_gib'] = torch.cuda.max_memory_allocated() / 1024**3
    if es_monitor:
        cont_final_results['early_stopping_summary'] = es_monitor.get_summary()

    cont_metrics_logger.save_final_results(cont_final_results)
    cont_metrics_logger.save()
```

**Note:** `cont_config_dict` comes from Task 2. `cont_final_results` mirrors the first-round `final_results` schema exactly.

**Step 4: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: save final_results.json after continued training (parity with first-round)"
```

---

## Task 6: Update the execution cell ⭐ LOGGING FIX

**Files:**
- Modify: `dev/legacy/legacy_full_training.ipynb` — cell 54 (execution cell)

**Why:** The execution cell needs to pass `log_dir` pointing to the original run's log directory so all artifacts co-locate.

**Step 1: Update the execution cell**

Replace the current execution cell with:

```python
# --- Execute: continue training with early stopping ---
CONTINUE_CHECKPOINT = 'logs/legacy_2026-03-16_06-21-09/legacy_replication/checkpoints/checkpoint_best.pt'
ADDITIONAL_EPOCHS = 5

# Log dir: reuse the ORIGINAL run's log directory so all artifacts co-locate
# This is the parent of the exp_name folder (i.e., the directory containing legacy_replication/)
CONTINUE_LOG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CONTINUE_CHECKPOINT)))
# Result: 'logs/legacy_2026-03-16_06-21-09'

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
    log_dir=CONTINUE_LOG_DIR,
)
```

**Step 2: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "feat: execution cell passes log_dir for artifact co-location"
```

---

## Task 7: Verify end-to-end artifact layout

**Step 1: Mentally trace the artifact layout**

Given `CONTINUE_CHECKPOINT = 'logs/legacy_2026-03-16_06-21-09/legacy_replication/checkpoints/checkpoint_best.pt'`:
- `CONTINUE_LOG_DIR` = `'logs/legacy_2026-03-16_06-21-09'`
- `cont_log_dir` = `'logs/legacy_2026-03-16_06-21-09'`
- `cont_checkpoint_dir` = `'logs/legacy_2026-03-16_06-21-09/legacy_replication/checkpoints'`
- `MetricsLogger` log_path = `'logs/legacy_2026-03-16_06-21-09/legacy_replication'`

Expected artifacts after continued training (epoch 1 = first-round, epoch 2+ = continued):

```
logs/legacy_2026-03-16_06-21-09/legacy_replication/
├── config.json                    # Updated with continued training config
├── epoch_metrics.json             # Epochs 1 + 2,3,4,5,6 (appended via resume=True)
├── batch_metrics.json             # All batches (appended via resume=True)
├── final_results.json             # Updated with continued training results
├── loss_trajectory_epoch0.json    # From first round
├── loss_trajectory_epoch1.json    # From continued training
├── loss_trajectory_epoch2.json    # From continued training
├── ...
├── training.log                   # Appended (resume=True → file mode 'a')
├── early_stopping_summary.json    # From continued training
└── checkpoints/
    ├── checkpoint_best.pt         # Updated if continued training improves
    ├── checkpoint_best_es.pt      # Early stopping best
    ├── checkpoint_epoch0.pt       # From first round
    ├── checkpoint_epoch1.pt       # From continued training
    ├── checkpoint_epoch2.pt       # ...
    └── checkpoint_latest.pt       # Always latest
```

**Step 2: Verify all cells compile**

```bash
python3 -c "
import json
with open('dev/legacy/legacy_full_training.ipynb') as f:
    nb = json.load(f)
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    if 'def continue_training_from_checkpoint' in src:
        compile(src, '<continue_cell>', 'exec')
        print('continue_training cell: OK')
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    if 'CONTINUE_LOG_DIR' in src and 'continue_training_from_checkpoint' in src:
        compile(src, '<execution_cell>', 'exec')
        print('execution cell: OK')
print('All compiled successfully.')
"
```

**Step 3: Commit**

```bash
git add dev/legacy/legacy_full_training.ipynb
git commit -m "chore: verify continued training logging parity compiles"
```

---

## Complete List of Logging Gaps Fixed

| Gap | Fixed In |
|---|---|
| `metrics_logger.log_config()` not called | Task 2 |
| `logger.info(f"Config: ...")` not called | Task 2 |
| `epoch_entry` missing `val_time_s` / `train_time_s` | Task 3 |
| `logger.info(f"Epoch N: ...")` not called | Task 3 |
| `loss_tracker.save_trajectory()` not called | Task 4 |
| `metrics_logger.save_final_results()` not called | Task 5 |
| `compute_training_time_metrics()` / `compute_cost_metrics()` not called | Task 5 |
| Timing accumulators not maintained | Task 5 |
| `setup_experiment_logging()` missing `resume=True` → overwrites training.log | Task 1 |
| Artifacts written to separate directory, not alongside first-round | Task 1, Task 6 |
| `MetricsLogger` potentially starts empty instead of appending | Task 1 (correct log_dir + resume=True) |

---

## Risk Analysis

| Risk | Mitigation |
|---|---|
| `MetricsLogger.save()` overwrites `config.json` with continued config | Acceptable — continued config is a superset; first-round config is in `final_results.json` from round 1 |
| `final_results.json` overwritten | The continued training writes its own `final_results.json`; first-round's is already there. `save_final_results` overwrites. Consider: the continued training `final_results` captures the *full* continued history, which is the latest state. |
| `checkpoint_best.pt` overwritten even if continued training is worse | Only overwritten when `val_loss < cont_best_val_loss` (which starts at `prev_val_loss` from checkpoint). So it only replaces if actually better. |
| `training.log` grows large | Acceptable; append mode is the standard pattern for continued training. |
