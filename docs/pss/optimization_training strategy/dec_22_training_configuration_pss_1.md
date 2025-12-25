# Comprehensive Analysis: Kernel Death & Slow Training

## Issue 1: DataLoader Workers Killing the Kernel 🔴 CRITICAL

**The Problem:**
```
Using DataLoader with 32 workers.
```

**Root Cause (line 8976):**
```python
n_workers = max(1, os.cpu_count() // max(world_size, 1) // 2)
# = 64 // 1 // 2 = 32 workers
```

**Why it kills the kernel:**
- 32 worker processes × ~2-4GB RAM each = 64-128GB just for workers
- Each worker prefetches batches with multi-hot tensors: 128 × 200 × 6297 × 4 bytes = 646MB per batch
- With `prefetch_factor=2` (default) × 32 workers = 32GB+ in prefetch queue
- Your 416GB RAM is shared with model + other processes → eventual memory exhaustion → kernel kill

**Fix:**
```python
def _create_dataloaders(...):
    # CRITICAL: Limit workers to prevent memory exhaustion
    # With 4 GPUs and large batches, 4-8 workers is plenty
    if torch.cuda.device_count() > 1:
        n_workers = min(4, os.cpu_count() // 4)  # Max 4 workers for multi-GPU
    else:
        n_workers = min(8, os.cpu_count() // 2)
    
    # Also limit prefetch to reduce memory
    prefetch_factor = 2 if n_workers > 0 else None
```

---

## Issue 2: Gradient Accumulation Not Configured Properly 🔴 CRITICAL

**The Problem:**
`train_epoch()` is called WITHOUT `accumulation_steps` parameter (line 9324-9342), so it defaults to 4.

**Impact:**
- Every 4 batches = 1 optimizer step
- With batch_size=128, effective batch = 128 × 4 = **512 samples per step**
- Only 12335/4 = **3084 optimizer steps** per epoch (not 12335!)
- Learning is 4x slower than you expect

**Your single-GPU baseline likely used accumulation_steps=1**, which is why multi-GPU feels slower.

**Fix - Option A (Disable accumulation):**
```python
train_metrics = train_epoch(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    config=config,
    device=device,
    scaler=scaler,
    use_mixed_precision=use_mixed_precision,
    moe_config=moe_config,
    epoch=epoch,
    use_bucketing=use_bucketing,
    log_interval=log_metrics_every,
    global_step=global_step,
    loss_tracker=loss_tracker,
    is_main=is_main,
    use_ddp=use_ddp,
    accumulation_steps=1  # ← ADD THIS: No accumulation with DataParallel
)
```

---

## Issue 3: Learning Rate Scaling Mismatch 🟡 HIGH

**The Problem:**
```python
# Line 9246
scaled_lr = base_lr * math.sqrt(num_gpus)  # 1e-4 * sqrt(4) = 2e-4
```

With effective batch = 512 (128 batch × 4 accumulation), this LR is **too conservative**.

**Industry Practice:**
- Linear scaling: `LR = base_lr × (effective_batch / reference_batch)`
- For reference batch=32 and effective batch=512: `LR = 1e-4 × (512/32) = 1.6e-3`
- But that's aggressive; sqrt scaling: `LR = 1e-4 × sqrt(512/32) = 4e-4`

**Current LR = 2e-4 is too low** for the effective batch size, slowing convergence.

**Fix:**
```python
# If using accumulation_steps=1 (recommended):
if use_data_parallel:
    # Linear scaling with warmup is standard for DataParallel
    scaled_lr = base_lr * num_gpus  # 1e-4 * 4 = 4e-4
```

---

## Issue 4: Scheduler Misconfiguration 🟡 HIGH

**The Problem:**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

`T_max=epochs` means the scheduler expects `epochs` total calls to `.step()`. But you're calling it once per optimizer step (every batch or every 4 batches with accumulation).

**With 12335 batches and accumulation=4:**
- Scheduler gets 3084 steps per epoch
- But T_max=1 (epochs=1), so cosine cycle completes in 1 step!

**Fix:**
```python
# Calculate total optimizer steps
total_steps = len(train_loader) // accumulation_steps * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

Or use per-epoch stepping:
```python
# In training loop, step scheduler ONCE per epoch, not per batch
# Move scheduler.step() outside the batch loop
```

---

## Issue 5: GPU Memory Imbalance Analysis 🟡 MEDIUM

**Your logs show:**
```
After forward:   GPU 0: 2.71GB    GPUs 1-3: 1.53GB each
After backward:  GPU 0: 1.22GB    GPUs 1-3: 0.02GB each
```

**This is actually expected DataParallel behavior:**
1. GPU 0 holds: input data (before scatter) + gathered outputs + loss computation
2. GPUs 1-3: Only hold scattered inputs + activations during forward
3. After backward: Gradients are all-reduced to GPU 0, then GPUs 1-3 release memory

**The real issue:** GPU 0 has ~1.2GB more than others BEFORE forward (after data to GPU). This is because:
```python
x = x.cuda(non_blocking=True)  # Data goes to GPU 0 first
```
DataParallel then scatters to other GPUs during forward.

**This is normal** - but inefficient for memory. The solution would be to use DistributedDataParallel (DDP) instead, where each GPU loads its own data shard.

---

## Issue 6: Batch Size Already Scaled → Double Counting 🔴 CRITICAL

**The Problem:**
```python
# Line 9262
config.batch_size = effective_batch_size  # 32 * 4 = 128

# Then at line 9287
train_loader, val_loader = _create_dataloaders(train_data, val_data, config, ...)
```

The DataLoader is created with `batch_size=128`. DataParallel will scatter this to 4 GPUs, giving **32 samples per GPU** - which is correct.

**BUT** if you also have `accumulation_steps=4`, you're accumulating 4 batches of 128 = **512 samples per optimizer step**.

**Your single-GPU training probably used batch_size=32, no accumulation = 32 samples per step**.

Current setup: 512 samples/step = **16x larger effective batch** with **2x learning rate**. This is why convergence is slower!

---

## Issue 7: Memory Leak Suspicion 🟡 MEDIUM

**Observation:**
```
GPU 0: 1.22GB → 2.71GB → 1.22GB   (batch 2)
GPU 0: 1.32GB → 3.17GB → 1.71GB   (batch 100)  ← Growing!
GPU 0: 1.41GB → 2.90GB → 1.41GB   (batch 50)   
GPU 0: 1.10GB / 5.83GB peak       (batch 2000)
```

The baseline memory fluctuates (1.22GB → 1.41GB → 1.32GB → 1.10GB), which is normal with bucketing (different sequence lengths).

**Peak growing** (3.17GB → 5.83GB) could indicate:
1. Python object accumulation (lists growing)
2. Gradient history not being cleared
3. Logging tensors without `.item()` or `.detach()`

---

# Summary: Recommended Fixes

## Immediate Fixes (Apply Now)

### 1. Reduce DataLoader workers
```python
# In _create_dataloaders:
n_workers = min(4, os.cpu_count() // 4)  # Was: 32
```

### 2. Disable gradient accumulation for DataParallel
```python
# In run_single_experiment call to train_epoch:
accumulation_steps=1
```

### 3. Fix learning rate for multi-GPU
```python
# In run_single_experiment:
if use_data_parallel:
    scaled_lr = base_lr * num_gpus  # Linear scaling: 4e-4 for 4 GPUs
```

### 4. Fix scheduler
```python
# Use epoch-based stepping instead of step-based:
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# Then move scheduler.step() to END of epoch (after train_epoch returns)
```

---

## Complete Fixed Code for `run_single_experiment`

Would you like me to provide the complete corrected version of the key functions? You can switch to Agent mode and I can apply these changes directly, or I can output them as code blocks for you to copy.

**Summary of expected improvements:**
- No more kernel deaths (4 workers vs 32)
- 4x faster training (no unnecessary accumulation)
- Better convergence (proper LR scaling)
- Correct scheduler behavior