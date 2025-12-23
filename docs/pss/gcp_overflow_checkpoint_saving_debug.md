### **GCP bash command**
Here are the bash commands to inspect your Vertex AI Workbench instance's memory and storage:

#### Memory Status (RAM)

```bash
# Quick overview of memory usage
free -h

# Detailed memory info
cat /proc/meminfo

# Real-time memory and CPU monitoring
top -o %MEM
# or use htop for a nicer interface (if installed)
htop
```

#### Storage Capacity (Disk Usage)

```bash
# Overall disk usage summary
df -h

# Show disk usage of current directory and subdirectories
du -sh *

# Find largest directories from root (top 20)
du -h / 2>/dev/null | sort -rh | head -20

# Check specific directory (e.g., home)
du -sh /home/*

# Find largest files (top 20 files over 100MB)
find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null | sort -k5 -rh | head -20
```

#### What's Taking Up the Most Memory (Processes)

```bash
# Top 10 memory-consuming processes
ps aux --sort=-%mem | head -11

# More detailed view with memory in MB
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -20

# Check GPU memory (if using GPU instance)
nvidia-smi

# Memory usage by user
ps -eo user,%mem --sort=-%mem | awk '{arr[$1]+=$2} END {for (i in arr) print i, arr[i]"%"}' | sort -k2 -rh
```

#### Combined Quick Diagnostic Script

```bash
echo "=== MEMORY STATUS ==="
free -h
echo ""
echo "=== DISK USAGE ==="
df -h
echo ""
echo "=== TOP 10 MEMORY PROCESSES ==="
ps aux --sort=-%mem | head -11
echo ""
echo "=== LARGEST DIRECTORIES IN HOME ==="
du -sh /home/* 2>/dev/null | sort -rh | head -10
echo ""
echo "=== GPU STATUS (if available) ==="
nvidia-smi 2>/dev/null || echo "No GPU or nvidia-smi not available"
```

#### Checking Python/Jupyter Specific Memory

```bash
# Find large Jupyter checkpoints
find /home -name ".ipynb_checkpoints" -exec du -sh {} \; 2>/dev/null

# Find large conda/pip cache
du -sh ~/.cache/pip 2>/dev/null
du -sh ~/.conda 2>/dev/null
du -sh /opt/conda 2>/dev/null

# List Python processes and their memory
ps aux | grep -E 'python|jupyter' | grep -v grep
```

These commands will help you identify memory bottlenecks and storage hogs on your Vertex AI Workbench instance.


### 📂 **What Gets Saved (Every Single Epoch)**

Looking at lines 4272-4281 of your `save_checkpoint` function:

```4272:4301:/Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/dev/moe/moe_flashattn_3.py
    # Build checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  # ⚠️ THIS IS 2x MODEL SIZE!
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,  # Growing epoch history
        'timestamp': time.time()
    }
    
    # ⚠️ SAVES 3 FILES EVERY EPOCH:
    torch.save(checkpoint, latest_path)      # checkpoint_latest.pt
    torch.save(checkpoint, epoch_path)       # checkpoint_epoch{N}.pt  
    if is_best:
        torch.save(checkpoint, best_path)    # checkpoint_best.pt
```

---

### 📊 **Why Each Checkpoint Is So Large**

| Component | Size Calculation | Estimated Size |
|-----------|------------------|----------------|
| **Input Embeddings** | 75,516 × 256 dim × 4 bytes | ~77 MB |
| **Input Embeddings** (512 dim) | 75,516 × 512 dim × 4 bytes | ~154 MB |
| **Output Layer** | 256/512 × 6,297 × 4 bytes | ~6-13 MB |
| **Transformer Layers** | 6 layers × ~2-8M params each | ~50-200 MB |
| **Age/Gender/LoB Embeddings** | (1440 + 4 + 4) × 256 × 4 | ~1.5 MB |
| **⚠️ Optimizer States (AdamW)** | **2× all above** (momentum + variance) | **~270-700 MB** |

**Total per checkpoint:**
- **256 dim model**: ~400-500 MB each
- **512 dim model** (exp_round3): ~800 MB - 1 GB each

---

### 🔥 **The Storage Explosion**

Your current strategy saves **every epoch checkpoint**:

```
For 10 epochs with 512 dim model:
├── checkpoint_latest.pt    (~800 MB)
├── checkpoint_best.pt      (~800 MB)
├── checkpoint_epoch0.pt    (~800 MB)
├── checkpoint_epoch1.pt    (~800 MB)
├── ...
└── checkpoint_epoch9.pt    (~800 MB)

Total: ~9.6 GB per experiment! 🚨
```

Your folder sizes confirm this:
- `exp_round3` (512 dim): 3.6 GB per experiment × ~5 experiments = **~17 GB**
- `exp_round1/2` (256 dim): ~400MB × epochs × experiments = **12-13 GB**

---

### ✅ **Solution: Smarter Checkpoint Strategy**

Here's an optimized version you can use. Since you're in **ask mode**, I'll provide the code for you to copy:

```python
def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    metrics: Dict,
    is_best: bool = False,
    keep_last_n: int = 2,  # 👈 NEW: Only keep last N epoch checkpoints
    save_optimizer: bool = True  # 👈 NEW: Option to skip optimizer for final save
):
    """
    Save checkpoint with automatic cleanup of old checkpoints.
    
    Space-saving features:
    - Only keeps last N epoch checkpoints (default: 2)
    - Optional: skip optimizer state for lightweight saves
    - Cleans up before saving to prevent disk full errors
    """
    import glob
    import shutil
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Unwrap DataParallel if needed
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Build checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': actual_model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'timestamp': time.time()
    }
    
    # Only include optimizer if requested (saves ~50% space when False)
    if save_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # ============================================================
    # CLEANUP OLD CHECKPOINTS BEFORE SAVING (prevents disk full!)
    # ============================================================
    existing_checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch*.pt')),
        key=lambda x: int(x.split('epoch')[-1].replace('.pt', ''))
    )
    
    # Remove old checkpoints, keeping only last (keep_last_n - 1) to make room for new one
    while len(existing_checkpoints) >= keep_last_n:
        oldest = existing_checkpoints.pop(0)
        if os.path.exists(oldest):
            os.remove(oldest)
            print(f"🗑️ Removed old checkpoint: {os.path.basename(oldest)}")
    
    # Save latest (for resume) - always overwritten
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    print(f"💾 Saved: checkpoint_latest.pt")
    
    # Save epoch checkpoint
    epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, epoch_path)
    
    # Save best (when val loss improves)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        shutil.copy(epoch_path, best_path)  # Copy instead of re-saving
        print(f"✅ New best checkpoint saved!")
    
    return latest_path
```

---

### 🎯 **Quick Cleanup Commands (Run Now)**

```bash
# Check current checkpoint sizes
du -sh logs/*/checkpoints/

# Remove all epoch checkpoints, keep only latest and best
find logs -name "checkpoint_epoch*.pt" -delete

# Or keep only the last 2 epochs per experiment
for dir in logs/*/checkpoints/; do
    ls -t "$dir"/checkpoint_epoch*.pt 2>/dev/null | tail -n +3 | xargs rm -f
done
```

---

### 📋 **Summary of Space-Saving Options**

| Strategy | Space Saved | Impact on Recovery |
|----------|-------------|-------------------|
| Keep only last 2 epoch checkpoints | **~80%** | Can resume from recent epochs only |
| Remove optimizer from final save | **~50%** | Can't resume training, only inference |
| Use FP16 for model weights | **~50%** | Minimal precision loss |
| Compress checkpoints (gzip) | **~30-40%** | Slower save/load |
| Save every N epochs (not every 1) | **~50-90%** | Less granular recovery |

---

### 🔧 **Recommended Update to `run_single_experiment`**

Update the `save_checkpoint` call around line 8074 to use the new parameters:

```python
# In run_single_experiment, around line 8074
save_checkpoint(
    checkpoint_dir=checkpoint_dir,
    epoch=epoch,
    global_step=global_step,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    metrics=epoch_history,
    is_best=is_best,
    keep_last_n=2  # 👈 ADD THIS: Only keep last 2 epoch checkpoints
)
```

This will automatically clean up old checkpoints before each save, preventing disk full errors and keeping your storage manageable. Each experiment will only use ~1.6-2 GB instead of ~3.6 GB.]

## More about the checkpoint saving strategy and safe cleanup

---

### 📋 **Checkpoint vs Saved Model: Two Different Things**

Your code has **two separate saving mechanisms**:

| File | Purpose | Contains | Size | When Needed |
|------|---------|----------|------|-------------|
| **`checkpoints/checkpoint_*.pt`** | Resume interrupted training | Model + Optimizer + Scheduler + Scaler + Metrics | ~500-800 MB each | **Only during training** |
| **`saved_models/{model}_final.pt`** | Inference & evaluation | Model weights only | ~150-300 MB | **After training** |

---

### 🔄 **How Training Resume Works**

When you call `run_single_experiment(resume_from="path/to/checkpoint_latest.pt")`:

```7718:7760:/Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/dev/moe/moe_flashattn_3.py
def _resume_from_checkpoint(...):
    """
    Resume training from checkpoint.
    """
    checkpoint = torch.load(resume_path, map_location=device)
    
    # 1. Restore model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Restore optimizer state (momentum buffers, etc.)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 3. Restore scheduler (learning rate position)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 4. Restore scaler (mixed precision state)
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # 5. Resume from next epoch
    start_epoch = checkpoint['epoch'] + 1
```

The **optimizer state is critical for resume** because AdamW stores momentum buffers. If you lose these, the optimizer "forgets" the training history and behaves as if starting fresh (bad for convergence).

---

### ✅ **YES, It's Safe to Delete Checkpoints After Training Completes**

Looking at lines 8123-8144, your code already saves a **lightweight final model**:

```8123:8144:/Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/dev/moe/moe_flashattn_3.py
    if save_model:
        model_name = generate_model_name(...)
        model_path = save_trained_model(
            model=model,
            config=config,
            model_name=model_name,
            save_dir=model_save_dir,
            exp_results=results,
            ...
        )
```

This `save_trained_model()` saves only what you need for inference:
- ✅ Model weights (`model_state_dict`)
- ✅ Config (JSON)
- ✅ Results (JSON)
- ❌ NO optimizer states

**After training completes:**
- `saved_models/` → **KEEP** (needed for inference, evaluation)
- `checkpoints/` → **SAFE TO DELETE** (only needed for resume)

---

### 🗑️ **Terminal Commands to Clean Up Completed Experiments**

```bash
# List all checkpoint directories with sizes
du -sh logs/*/exp*/checkpoints/

# Remove all checkpoint directories (keep saved_models)
find logs -type d -name "checkpoints" -exec rm -rf {} + 2>/dev/null

# Or be more surgical - remove only epoch checkpoints, keep latest/best
find logs -name "checkpoint_epoch*.pt" -delete

# Verify what remains
find logs -name "*.pt" | head -20
```

---

### 🤖 **Auto-Delete Checkpoints After Training Completes**

Add this function and call it at the end of `run_single_experiment`:

```python
def cleanup_checkpoints_after_training(
    checkpoint_dir: str,
    keep_best: bool = True,
    logger: Optional[logging.Logger] = None
):
    """
    Clean up checkpoint files after training is complete.
    
    Called after save_trained_model() to reclaim disk space.
    
    Args:
        checkpoint_dir: Path to checkpoints directory
        keep_best: If True, keeps checkpoint_best.pt for reference
        logger: Logger for output
    """
    import glob
    import shutil
    
    if not os.path.exists(checkpoint_dir):
        return
    
    files_removed = 0
    bytes_freed = 0
    
    # Remove all epoch checkpoints
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch*.pt')):
        bytes_freed += os.path.getsize(f)
        os.remove(f)
        files_removed += 1
    
    # Remove latest checkpoint
    latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    if os.path.exists(latest):
        bytes_freed += os.path.getsize(latest)
        os.remove(latest)
        files_removed += 1
    
    # Optionally remove best checkpoint
    if not keep_best:
        best = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        if os.path.exists(best):
            bytes_freed += os.path.getsize(best)
            os.remove(best)
            files_removed += 1
    
    # Remove empty directory
    remaining = os.listdir(checkpoint_dir)
    if not remaining:
        os.rmdir(checkpoint_dir)
    
    gb_freed = bytes_freed / (1024 ** 3)
    msg = f"🗑️ Cleaned up {files_removed} checkpoint files, freed {gb_freed:.2f} GB"
    if logger:
        logger.info(msg)
    else:
        print(msg)
```

**Then add this call at the end of `run_single_experiment` (around line 8180):**

```python
# ... existing code around line 8177 ...
results_path = metrics_logger.save_final_results(results)
logger.info(f"Complete results saved to {results_path}")

# ============================================================
# 11. CLEANUP CHECKPOINTS (training complete, no longer needed)
# ============================================================
if save_model:  # Only cleanup if we successfully saved the final model
    cleanup_checkpoints_after_training(
        checkpoint_dir=checkpoint_dir,
        keep_best=False,  # Set to True if you want to keep checkpoint_best.pt
        logger=logger
    )

metrics_logger.save()
# ... rest of existing code ...
```

---

### 📊 **Space Savings Summary**

| Experiment Round | Current Size | After Cleanup | Savings |
|------------------|--------------|---------------|---------|
| exp_round3 (512 dim) | 17 GB | ~1.5 GB | **~15.5 GB (91%)** |
| exp_round1 | 13 GB | ~1.0 GB | **~12 GB (92%)** |
| exp_round2 | 12 GB | ~1.0 GB | **~11 GB (92%)** |
| **Total** | **~52 GB** | **~5 GB** | **~47 GB freed** |

---

### 🚀 **Quick Cleanup Script (Run Now)**

```bash
#!/bin/bash
# Save as cleanup_checkpoints.sh

echo "=== Before cleanup ==="
du -sh logs/

echo ""
echo "=== Removing checkpoint directories ==="
find logs -type d -name "checkpoints" -exec rm -rf {} + 2>/dev/null

echo ""
echo "=== After cleanup ==="
du -sh logs/

echo ""
echo "=== Remaining model files ==="
find logs -name "*_final.pt" | wc -l
echo "model files preserved"
```

**Bottom Line:** Checkpoints are training-time insurance. Once training finishes successfully and you have `saved_models/*_final.pt`, the checkpoints serve no purpose and can be safely deleted.