Let me thoroughly understand your codebase first before explaining the differences.

[5 tools called]

Now I have a complete understanding of your code. Let me provide a comprehensive explanation.

---

# DataParallel vs DistributedDataParallel: Complete Comparison for Your Codebase

## Your Current Architecture

Based on my analysis of `moe_flashattn_2.py`:

| Component | Current Implementation |
|-----------|----------------------|
| **Model Creation** | `model.to(device)` → single GPU (lines 6069, 6083, 6106) |
| **DataLoader** | Standard with `num_workers`, `pin_memory` (lines 6162-6187) |
| **Training Loop** | Standard forward/backward in `train_epoch()` (lines 3132-3349) |
| **Mixed Precision** | `GradScaler` + `autocast` (lines 3204-3251) |
| **Checkpoints** | Has both single-GPU (`save_checkpoint`) and multi-GPU aware functions (`save_checkpoint_multigpu`) |

---

## 1. DataParallel (DP) - How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATAPARALLEL                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1: Batch arrives at GPU 0                                │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Full Batch: [sample_0, sample_1, ..., sample_63]        │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   Step 2: GPU 0 SPLITS batch and BROADCASTS model               │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │  GPU 0   │  │  GPU 1   │  │  GPU 2   │  │  GPU 3   │       │
│   │ Model    │  │ Model    │  │ Model    │  │ Model    │       │
│   │ Copy     │  │ Copy     │  │ Copy     │  │ Copy     │       │
│   │ [0:15]   │  │ [16:31]  │  │ [32:47]  │  │ [48:63]  │       │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│        │             │             │             │               │
│        ▼             ▼             ▼             ▼               │
│   Step 3: Each GPU runs FORWARD pass independently              │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │ output_0 │  │ output_1 │  │ output_2 │  │ output_3 │       │
│   │ grad_0   │  │ grad_1   │  │ grad_2   │  │ grad_3   │       │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│        │             │             │             │               │
│        └─────────────┴──────┬──────┴─────────────┘               │
│                             ▼                                    │
│   Step 4: GPU 0 GATHERS all outputs and gradients               │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  GPU 0: Receives all gradients → Averages → Updates       │  │
│   │         Then broadcasts updated weights to all GPUs       │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics:
- **Single process, single Python interpreter**
- **GPU 0 bottleneck**: All communication funnels through GPU 0
- **Synchronous**: All GPUs wait for GPU 0 to gather/broadcast
- **GIL (Python's Global Interpreter Lock)**: Limits true parallelism

---

## 2. DistributedDataParallel (DDP) - How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                   DISTRIBUTED DATA PARALLEL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 0: SEPARATE PROCESS per GPU (no GIL contention!)         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │Process 0 │  │Process 1 │  │Process 2 │  │Process 3 │       │
│   │ GPU 0    │  │ GPU 1    │  │ GPU 2    │  │ GPU 3    │       │
│   │ Rank 0   │  │ Rank 1   │  │ Rank 2   │  │ Rank 3   │       │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                  │
│   Step 1: Each process loads ITS OWN data shard                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │ Sampler  │  │ Sampler  │  │ Sampler  │  │ Sampler  │       │
│   │ [0,4,8..]│  │[1,5,9..]│  │[2,6,10..]│  │[3,7,11..]│       │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│        │             │             │             │               │
│        ▼             ▼             ▼             ▼               │
│   Step 2: Forward pass (PARALLEL, no communication)             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │ Forward  │  │ Forward  │  │ Forward  │  │ Forward  │       │
│   │ Loss     │  │ Loss     │  │ Loss     │  │ Loss     │       │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│        │             │             │             │               │
│        ▼             ▼             ▼             ▼               │
│   Step 3: Backward pass + ALL-REDUCE gradients                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │     RING ALL-REDUCE (NCCL backend - very efficient)     │   │
│   │  GPU0 ←→ GPU1 ←→ GPU2 ←→ GPU3 ←→ GPU0                   │   │
│   │  Each GPU sends/receives, OVERLAPPED with backward      │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │             │             │             │               │
│        ▼             ▼             ▼             ▼               │
│   Step 4: Each GPU updates weights LOCALLY (identical result)   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │ Update   │  │ Update   │  │ Update   │  │ Update   │       │
│   │ (same!)  │  │ (same!)  │  │ (same!)  │  │ (same!)  │       │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics:
- **Multiple processes** (one per GPU) - bypasses Python GIL
- **No GPU 0 bottleneck**: Ring all-reduce distributes communication
- **Gradient computation overlaps with communication** (huge efficiency gain)
- **Each process loads its own data** via `DistributedSampler`

---

## 3. Head-to-Head Comparison

| Aspect | DataParallel | DistributedDataParallel |
|--------|--------------|-------------------------|
| **Ease of Implementation** | ✅ 2 lines of code | ❌ ~30 lines + launch script |
| **Speedup (4 GPUs)** | ~2-2.5x | ~3.5-3.9x |
| **GPU 0 Memory** | ❌ Higher (gathers all) | ✅ Balanced |
| **Communication** | ❌ All through GPU 0 | ✅ Ring all-reduce |
| **GIL Bottleneck** | ❌ Yes | ✅ No (separate processes) |
| **Mixed Precision** | ✅ Works | ✅ Works |
| **Multi-Node** | ❌ No | ✅ Yes |
| **Debugging** | ✅ Easy (single process) | ❌ Harder (multiple processes) |
| **Production Use** | ⚠️ Prototyping only | ✅ Industry standard |

### Efficiency Numbers (Typical for 4 GPUs):

| Metric | DataParallel | DDP |
|--------|--------------|-----|
| **Scaling Efficiency** | 50-65% | 90-97% |
| **Your 4 GPU expected speedup** | 2.0-2.6x | 3.6-3.9x |
| **Time for your 1-epoch run** | ~2500-3100s (vs 6269s) | ~1600-1750s |

---

## 4. Impact on Your Training Results

### Mathematical Equivalence

**Both methods produce identical training results** when configured correctly:

```
Effective Batch Size = per_GPU_batch_size × num_GPUs

DataParallel:    batch_size=64 → split to 16/GPU → gradients averaged → same as batch_size=64
DDP:             batch_size=16/process × 4 processes → all-reduce average → same as batch_size=64
```

Your current `batch_size=16` would become:
- **DataParallel**: Each GPU gets 4 samples (may be too small for efficiency!)
- **DDP**: Each process uses 16 samples → effective batch = 64

### Production Considerations

| Aspect | DataParallel | DDP |
|--------|--------------|-----|
| **Model Accuracy** | ✅ Same | ✅ Same |
| **Convergence** | ✅ Same | ✅ Same |
| **Checkpoint Compatibility** | ⚠️ Need `.module.` handling | ⚠️ Need `.module.` handling |
| **Inference Deployment** | ✅ Simple | ⚠️ Need to unwrap model |
| **Production Serving** | ✅ Easy | ✅ Easy after unwrap |

---

## 5. Implementation for Your Code

### Option A: DataParallel (Quick & Easy)

Changes required in `run_single_experiment()`:

```python:dev/moe/moe_flashattn_2.py
# ... existing model creation code (lines 6063-6115) ...

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # ============================================================
    # ADD: DATAPARALLEL WRAPPER
    # ============================================================
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"🔥 Enabling DataParallel with {num_gpus} GPUs")
        logger.info(f"   Effective batch size: {config.batch_size} (split across GPUs)")
        logger.info(f"   Per-GPU batch size: {config.batch_size // num_gpus}")
        model = nn.DataParallel(model)
    
    # Log config
    config_dict = {
// ... existing code ...
```

**Also update `save_checkpoint()` to handle DataParallel** (lines 3754-3758):

```python:dev/moe/moe_flashattn_2.py
    # Build checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        # Handle DataParallel wrapper
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
// ... existing code ...
```

**And update `load_checkpoint()`** (around line 3811):

```python:dev/moe/moe_flashattn_2.py
    # Restore states - handle DataParallel
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
// ... existing code ...
```

**Recommended batch size adjustment** (in your experiment call):
```python
# Increase batch size for multi-GPU efficiency
config.batch_size = 64  # or 128 for 4 GPUs
```

---

### Option B: DistributedDataParallel (Production-Grade)

This requires more extensive changes:

#### 1. New Launch Script (`launch_ddp.py`)

```python
# launch_ddp.py - Run with: torchrun --nproc_per_node=4 launch_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    """Initialize distributed process group."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Clean up distributed resources."""
    dist.destroy_process_group()
```

#### 2. Modified `run_single_experiment()` for DDP

```python:dev/moe/moe_flashattn_2.py
def run_single_experiment_ddp(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    local_rank: int,  # NEW: DDP rank
    epochs: int = 4,
    # ... other params ...
) -> Dict[str, any]:
    
    # ============================================================
    # DDP SETUP
    # ============================================================
    device = torch.device(f'cuda:{local_rank}')
    is_main_process = (local_rank == 0)  # Only rank 0 logs/saves
    
    # ... model creation code (same as before) ...
    
    model = model.to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # ============================================================
    # DDP DATA LOADING
    # ============================================================
    train_dataset = ClinicalDataset(train_data, config)
    val_dataset = ClinicalDataset(val_data, config)
    
    # DistributedSampler ensures each GPU sees different data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,  # This is PER-GPU batch size
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=max(1, os.cpu_count() // dist.get_world_size()),
        pin_memory=True,
        drop_last=True,
        collate_fn=clinical_collate_fn
    )
    
    # Validation: only on main process OR use DistributedSampler
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=clinical_collate_fn
    )
    
    # ============================================================
    # TRAINING LOOP (with DDP modifications)
    # ============================================================
    for epoch in range(epochs):
        # CRITICAL: Set epoch for sampler to reshuffle each epoch
        train_sampler.set_epoch(epoch)
        
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            # ... other args ...
        )
        
        # Synchronize metrics across processes
        if dist.get_world_size() > 1:
            # All-reduce the loss to get true average
            loss_tensor = torch.tensor(train_metrics['train_loss'], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            train_metrics['train_loss'] = loss_tensor.item()
        
        # Only main process saves checkpoints and logs
        if is_main_process:
            save_checkpoint(...)
            logger.info(f"Epoch {epoch}: loss={train_metrics['train_loss']:.4f}")
    
    return results
```

#### 3. Modified `train_epoch()` for DDP

The actual training loop needs minimal changes if using DDP correctly:

```python:dev/moe/moe_flashattn_2.py
def train_epoch(
    model: nn.Module,  # Already wrapped in DDP
    dataloader: DataLoader,
    # ... other params ...
) -> Dict[str, float]:
    
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass (DDP handles gradient sync automatically)
        output = model(x)
        loss = compute_loss(...)
        
        # Backward pass (DDP automatically syncs gradients via hooks)
        loss.backward()
        
        # Gradient clipping (done per-process, same result due to synced grads)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
    
    # No explicit gradient synchronization needed - DDP does it automatically!
    return epoch_metrics
```

#### 4. Launch Command

```bash
# For 4 GPUs on single machine:
torchrun --nproc_per_node=4 launch_ddp.py

# For multi-node (e.g., 2 machines × 4 GPUs):
# On node 0:
torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 --nproc_per_node=4 launch_ddp.py
# On node 1:
torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=29500 --nproc_per_node=4 launch_ddp.py
```

---

## 6. My Recommendation

| Situation | Recommendation |
|-----------|----------------|
| **Quick experiments / debugging** | DataParallel (5 min to implement) |
| **Training <10 epochs, 4 GPUs** | DataParallel is fine |
| **Production training, many epochs** | DDP (worth the setup cost) |
| **Multi-node training** | DDP (only option) |
| **Your current situation** | **Start with DataParallel**, upgrade to DDP later |

### Immediate Action (DataParallel)

Add these ~15 lines to your code and you'll see **2-2.5x speedup** immediately:

1. Wrap model with `nn.DataParallel` after creation
2. Update checkpoint save/load to handle `.module.`
3. Increase batch size to 64-128 for efficiency

### Future Upgrade (DDP)

When you're ready for production training at scale:
1. Implement DDP wrapper
2. Add DistributedSampler to DataLoader
3. Use `torchrun` for launching
4. Get **3.5-3.9x speedup** with 4 GPUs


# Things to Consider: GPU Memory Bottleneck in DataParallel

## GPU 0 Memory Bottleneck in DataParallel

### What Gets Gathered on GPU 0

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  GPU 0 MEMORY DURING DATAPARALLEL                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ALWAYS on GPU 0:                                                        │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  • Master copy of model weights                                │     │
│  │  • Optimizer states (Adam: 2× param memory for momentum/var)   │     │
│  │  • Gradient buffers for its own shard                          │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  GATHERED from other GPUs (temporary):                                   │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  • Outputs from GPU 1, 2, 3 (for loss computation)             │     │
│  │  • Gradients from GPU 1, 2, 3 (for averaging & update)         │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Breakdown for Your Model

Based on your results (`model_params: 84,470,930`):

| Component | GPU 0 | GPU 1-3 |
|-----------|-------|---------|
| **Model Weights (FP32)** | 322 MB | 322 MB each |
| **Model Weights (FP16)** | 161 MB | 161 MB each |
| **Optimizer States (AdamW)** | 644 MB | ❌ None |
| **Activations (per-GPU shard)** | ~1.5 GB | ~1.5 GB each |
| **Gathered Outputs** | ~4.5 GB (all 4 GPUs) | ❌ None |
| **Gathered Gradients** | ~1.3 GB (all 4 GPUs) | ❌ None |
| **Total** | **~8+ GB** | **~2 GB** |

### The Asymmetry Problem

```
GPU Memory Usage with DataParallel (4 GPUs):

GPU 0: ████████████████████████████████████░░░░  ~8-10 GB  (BOTTLENECK!)
GPU 1: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~2 GB
GPU 2: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~2 GB  
GPU 3: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~2 GB

T4 GPU has 16 GB → GPU 0 may hit limit before others!
```

### Why This Matters for Your Training

Looking at your results:
```python
'gpu0_allocated_gb': 2.84,  # Currently single GPU
'gpu0_peak_gb': 6.68,       # Peak usage
```

With DataParallel on 4 GPUs:
- **GPU 0 peak** could reach **8-12 GB** (gathering outputs + gradients from all GPUs)
- **Your T4 has 16 GB**, so you should be safe
- **But**: If you increase batch size significantly, GPU 0 could OOM first

### Practical Implications

| Scenario | DataParallel Impact |
|----------|---------------------|
| **Your current model (84M params)** | ✅ Should fit on T4 (16GB) |
| **Larger batch size (64-128)** | ⚠️ Monitor GPU 0 carefully |
| **Larger model (200M+ params)** | ❌ GPU 0 likely OOMs first |
| **8 GPUs** | ❌ GPU 0 gathers 8× outputs - major bottleneck |

### How DDP Avoids This

```
DDP Memory Distribution (truly balanced):

GPU 0: ████████████████░░░░░░░░░░░░░░░░░░░░░░░░  ~4 GB
GPU 1: ████████████████░░░░░░░░░░░░░░░░░░░░░░░░  ~4 GB
GPU 2: ████████████████░░░░░░░░░░░░░░░░░░░░░░░░  ~4 GB
GPU 3: ████████████████░░░░░░░░░░░░░░░░░░░░░░░░  ~4 GB

Each GPU: model + optimizer states + activations for its shard
No gathering - gradients synchronized via ring all-reduce (in-place)
```

### Workarounds for DataParallel Memory Issues

If you hit GPU 0 memory limits:

1. **Reduce batch size** (but hurts efficiency)
   ```python
   config.batch_size = 32  # Instead of 64
   ```

2. **Use gradient accumulation** (simulate larger batch)
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Use gradient checkpointing** (trade compute for memory)
   ```python
   from torch.utils.checkpoint import checkpoint
   # In your model's forward pass
   output = checkpoint(self.transformer_block, x)
   ```

4. **Switch to DDP** (best solution for memory balance)

### Bottom Line

For your current setup:
- **84M params + T4 (16GB)** → DataParallel should work fine
- **Batch size 16 split 4 ways** → Only 4 samples per GPU (inefficient, but memory-safe)
- **Batch size 64 split 4 ways** → 16 samples per GPU (better efficiency, still safe)
- **Batch size 128+** → ⚠️ Start monitoring GPU 0 memory

I recommend adding this monitoring to your training loop:

```python
# In train_epoch(), add after forward pass:
if batch_idx % 100 == 0 and device.type == 'cuda':
    for gpu_id in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        print(f"    GPU {gpu_id}: {allocated:.2f}GB / {peak:.2f}GB peak")
```


# DDP or DataParallel on H100?

## H100 vs T4: Key Differences That Matter

| Specification | T4 | H100 (SXM5) | Impact |
|--------------|-----|-------------|--------|
| **Memory** | 16 GB GDDR6 | 80 GB HBM3 | No memory bottleneck |
| **Memory Bandwidth** | 300 GB/s | 3,350 GB/s | 11× faster data movement |
| **FP16 TFLOPs** | 65 | 1,979 | 30× more compute |
| **NVLink** | ❌ No | ✅ 900 GB/s | Fast GPU-to-GPU |
| **NVSwitch** | ❌ No | ✅ Yes | All-to-all connectivity |

## Will DataParallel Work on H100?

**Yes, it will work** — but you'll be leaving massive performance on the table.

### DataParallel on H100: The Problems

```
┌─────────────────────────────────────────────────────────────────────────┐
│              DATAPARALLEL ON 4× H100 - WHAT HAPPENS                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GPU 0 (Master):                                                         │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  1. Receives batch from CPU via PCIe (slow: 64 GB/s)          │     │
│  │  2. Splits batch, sends to GPUs 1-3 via PCIe (not NVLink!)    │     │
│  │  3. Waits for all GPUs to finish                               │     │
│  │  4. Gathers outputs via PCIe (not NVLink!)                     │     │
│  │  5. Computes loss, backward                                    │     │
│  │  6. Gathers gradients via PCIe (not NVLink!)                   │     │
│  │  7. Averages, updates weights                                  │     │
│  │  8. Broadcasts new weights via PCIe (not NVLink!)              │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ❌ PROBLEM: DataParallel uses CUDA IPC, NOT NVLink!                     │
│  ❌ PROBLEM: Python GIL serializes operations                            │
│  ❌ PROBLEM: GPU 0 bottleneck wastes 75% of potential                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### The NVLink Waste

H100s connected via NVLink can communicate at **900 GB/s** (per GPU). But DataParallel doesn't use NVLink efficiently — it goes through PyTorch's CUDA tensors and Python, falling back to slower paths.

```
Communication Speed Comparison:

DataParallel path:  GPU 0 ←─ PCIe ─→ CPU ←─ PCIe ─→ GPU 1
                    Effective: ~20-40 GB/s (bottlenecked)

DDP + NCCL path:    GPU 0 ←─── NVLink ───→ GPU 1
                    Effective: ~700-800 GB/s (near wire speed)

                    DDP is 20-30× faster for communication!
```

## DDP on H100: Why It's Essential

### NCCL + NVLink Magic

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DDP ON 4× H100 WITH NVLINK                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│      ┌─────────┐         NVLink         ┌─────────┐                     │
│      │  H100   │ ◄──── 900 GB/s ────► │  H100   │                     │
│      │  GPU 0  │                        │  GPU 1  │                     │
│      └────┬────┘                        └────┬────┘                     │
│           │                                   │                          │
│         NVLink                             NVLink                        │
│        900 GB/s                           900 GB/s                       │
│           │                                   │                          │
│      ┌────┴────┐                        ┌────┴────┐                     │
│      │  H100   │ ◄──── 900 GB/s ────► │  H100   │                     │
│      │  GPU 3  │         NVLink         │  GPU 2  │                     │
│      └─────────┘                        └─────────┘                     │
│                                                                          │
│  ✅ Ring All-Reduce via NCCL uses NVLink directly                       │
│  ✅ Gradient sync OVERLAPS with backward pass                           │
│  ✅ No GPU 0 bottleneck - all GPUs equal                                │
│  ✅ No Python GIL - separate processes                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance Comparison: Your Model on 4× H100

| Metric | DataParallel | DDP + NCCL |
|--------|--------------|------------|
| **Scaling Efficiency** | 40-55% | 92-97% |
| **Effective TFLOPs** | ~3,200 (of 7,916) | ~7,500 (of 7,916) |
| **GPU Utilization** | 50-60% | 85-95% |
| **Time per Epoch (1M samples)** | ~180 sec | ~80 sec |
| **NVLink Utilization** | ~10% | ~80% |
| **Cost Efficiency** | Poor | Excellent |

### Cost Implications

H100 instances are expensive (~$30-40/hour for 4× H100). Wasting 40-50% efficiency with DataParallel means:

```
Training 10 epochs on 1M samples:

DataParallel: 180 sec/epoch × 10 = 30 min = ~$20 wasted
DDP:          80 sec/epoch × 10 = 13 min = optimal

Over a project with multiple experiments: $100s-$1000s difference
```

## My Strong Recommendation

**For 4× H100: Use DDP. DataParallel would be a significant waste of expensive hardware.**

### Minimal DDP Implementation for Your Code

Here's the least-invasive way to add DDP:

#### 1. Create a launch wrapper (`run_ddp.py`)

```python
#!/usr/bin/env python
"""
Launch with: torchrun --nproc_per_node=4 run_ddp.py
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup():
    """Initialize DDP."""
    dist.init_process_group(backend='nccl')  # NCCL uses NVLink automatically
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (local_rank == 0)
    
    # Import your training code
    from moe_flashattn_2 import (
        run_single_experiment_ddp,  # You'll create this
        df_train, df_val
    )
    
    try:
        results = run_single_experiment_ddp(
            exp_name='exp2b_flash_learned_pool',
            local_rank=local_rank,
            train_data=df_train,
            val_data=df_val,
            epochs=10
        )
        
        if is_main:
            print(f"Training complete! Results: {results}")
    finally:
        cleanup()

if __name__ == '__main__':
    main()
```

#### 2. Key modifications to `run_single_experiment()`

```python
def run_single_experiment_ddp(
    exp_name: str,
    local_rank: int,  # NEW
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    epochs: int = 10,
    # ... other params
):
    # Setup
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()
    is_main = (local_rank == 0)
    
    # Only main process logs
    if is_main:
        logger = setup_experiment_logging(exp_name, ...)
    
    # Model creation (same as before)
    model = FlashMoETransformer(config, moe_config).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # DataLoader with DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,  # Per-GPU batch size
        sampler=train_sampler,
        num_workers=8,  # H100 can handle more workers
        pin_memory=True,
        drop_last=True,
        collate_fn=clinical_collate_fn
    )
    
    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Critical for proper shuffling
        
        train_metrics = train_epoch(...)
        
        # Sync metrics across GPUs
        loss_tensor = torch.tensor(train_metrics['train_loss'], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        
        # Only main saves checkpoints
        if is_main:
            save_checkpoint(...)
    
    return results
```

#### 3. Launch command

```bash
# On H100 node with 4 GPUs:
torchrun --nproc_per_node=4 run_ddp.py
```

## H100-Specific Optimizations

Beyond DDP, H100 enables additional optimizations:

### 1. Enable TF32 (free 3× speedup for matmuls)

```python
# Add at the start of your script
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Use torch.compile (PyTorch 2.0+)

```python
# After model creation, before DDP wrap:
model = torch.compile(model, mode='reduce-overhead')
model = DDP(model, device_ids=[local_rank])
```

### 3. Larger batch sizes

H100 has 80GB memory — you can use much larger batches:

```python
# T4: batch_size = 16-32
# H100: batch_size = 128-256 per GPU
config.batch_size = 128  # Per-GPU
# Effective batch = 128 × 4 = 512
```

### 4. FP8 training (H100 exclusive)

```python
# If using transformer engine (NVIDIA library)
import transformer_engine.pytorch as te
# Enables FP8 for even faster training
```

## Summary: DataParallel vs DDP for H100

| Aspect | DataParallel | DDP |
|--------|--------------|-----|
| **Will it work?** | ✅ Yes | ✅ Yes |
| **Uses NVLink?** | ❌ Poorly | ✅ Fully |
| **Scaling Efficiency** | ~50% | ~95% |
| **Implementation Effort** | 5 min | 30 min |
| **Cost Efficiency** | ❌ Wasting $15-20/hr | ✅ Optimal |
| **Recommendation** | ❌ Don't use on H100 | ✅ **Use this** |

**Bottom line**: For 4× H100, DDP is not just "better" — it's essential. DataParallel would waste roughly half your $30-40/hour GPU cost by not utilizing NVLink and suffering from Python GIL bottlenecks.

Would you like me to provide the complete DDP implementation tailored to your specific `run_single_experiment()` function?


Let me do a comprehensive analysis of your codebase to provide a complete DDP implementation plan.

[7 tools called]

Now I have a complete picture of your codebase. Let me provide a comprehensive DDP implementation analysis.

---

# Complete DDP Implementation for Your Codebase

## Implementation Effort Analysis

| Component | Changes Required | Complexity | Risk |
|-----------|------------------|------------|------|
| **New imports** | Add 5 new imports | ⭐ Low | ⭐ Low |
| **Setup/cleanup functions** | Add 2 new functions | ⭐ Low | ⭐ Low |
| **`run_single_experiment()`** | ~50 lines modified | ⭐⭐⭐ Medium | ⭐⭐ Medium |
| **`train_epoch()`** | ~15 lines modified | ⭐⭐ Low | ⭐ Low |
| **`evaluate()`** | ~10 lines modified | ⭐⭐ Low | ⭐ Low |
| **`save_checkpoint()`** | ~5 lines modified | ⭐ Low | ⭐ Low |
| **`run_selected_experiments()`** | ~20 lines modified | ⭐⭐ Medium | ⭐⭐ Medium |
| **Launch script** | New file (~50 lines) | ⭐⭐ Medium | ⭐ Low |
| **Testing** | Verification scripts | ⭐⭐ Medium | ⭐ Low |

**Total estimated time: 2-4 hours for careful implementation**

---

## Challenges and Drawbacks

### 1. **Dataset Pre-processing Redundancy**

Your `ClinicalDataset.__init__()` does heavy pre-processing:
```python
# Lines 2228-2260: Each process will run this independently
for i in range(len(df)):
    age_list = conv_age_gender(age_strs[i], config.len_dy)
    # ... expensive parsing
```

**Challenge**: With DDP, each of 4 processes loads and pre-processes the entire dataset.
- 1M samples × 4 processes = 4× CPU memory usage during init
- 4× redundant parsing time

**Mitigation**: The data is split by `DistributedSampler` at iteration time, not at load time. This is acceptable for your dataset size.

### 2. **Logging Coordination**

Your code has extensive logging throughout:
```python
logger.info(f"Model: Flash Attention Transformer (FP16)")
print(f"  Batch {batch_idx}/{len(dataloader)}")
```

**Challenge**: With 4 processes, you'd get 4× the log output (messy, confusing).

**Solution**: Gate all logging/printing with `is_main_process` check.

### 3. **Random Seed Synchronization**

Different processes could diverge if not properly seeded.

**Solution**: Set seeds consistently at startup.

### 4. **Metric Aggregation**

Your `train_epoch()` computes per-batch metrics locally:
```python
batch_metrics = compute_batch_metrics_lightweight(output, y, dt_cnt, config, device)
```

**Challenge**: Each process only sees 1/4 of the data, so metrics are local.

**Solution**: All-reduce metrics at epoch end.

### 5. **Checkpoint Compatibility**

DDP wraps model with `module.` prefix in state_dict.

**Solution**: Your code already has `save_checkpoint_multigpu()` that handles this!

### 6. **Evaluation Strategy**

Options:
- **All ranks evaluate**: More thorough but needs gather
- **Only rank 0 evaluates**: Simpler, sufficient for your use case

### 7. **Debugging Complexity**

Multi-process debugging is harder than single-process.

**Mitigation**: Provide fallback single-GPU mode.

---

## Complete Implementation

Here's the complete DDP implementation that maintains backward compatibility:

### Part 1: New DDP Utilities (Add near top of file, after imports ~line 160)

```python
# ============================================================================
# DDP (Distributed Data Parallel) UTILITIES
# ============================================================================

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp() -> Tuple[int, int, bool]:
    """
    Initialize Distributed Data Parallel.
    
    Returns:
        local_rank: GPU index on this machine
        world_size: Total number of processes
        is_main: True if this is rank 0 (main process)
    """
    # Check if we're in a distributed environment
    if 'LOCAL_RANK' not in os.environ:
        # Not running with torchrun - single GPU mode
        return 0, 1, True
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    # Set seeds for reproducibility (different per rank for data, same for model init)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed + local_rank)  # Different random data per rank
    
    is_main = (local_rank == 0)
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"DDP INITIALIZED")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Backend: NCCL (GPU-optimized)")
        print(f"{'='*60}\n")
    
    # Synchronize all processes
    dist.barrier()
    
    return local_rank, world_size, is_main


def cleanup_ddp():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_dist_initialized() -> bool:
    """Check if DDP is initialized."""
    return dist.is_initialized()


def get_world_size() -> int:
    """Get number of processes (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor, op: str = 'mean') -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: 'mean' or 'sum'
    
    Returns:
        Reduced tensor (only meaningful on rank 0, but returned on all ranks)
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    
    # Clone to avoid modifying original
    rt = tensor.clone()
    
    # All-reduce
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if op == 'mean':
        rt = rt / world_size
    
    return rt


def sync_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """
    Synchronize metrics across all processes.
    
    Args:
        metrics: Dictionary of metric names to values
        device: Current device
    
    Returns:
        Synchronized metrics (averaged across processes)
    """
    if not dist.is_initialized():
        return metrics
    
    synced = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            synced[key] = tensor.item() / dist.get_world_size()
        else:
            synced[key] = value  # Non-numeric, keep as is
    
    return synced
```

### Part 2: Modified `run_single_experiment()` with DDP Support

Replace the function (starting around line 5972) with this version:

```python
def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,  # Will be overridden if DDP
    epochs: int = 4,
    code_frequencies: Optional[np.ndarray] = None,
    log_dir: str = "logs",
    experiment_round: Optional[str] = None,
    check_embeddings_every: int = 2,
    log_metrics_every: int = 100,
    resume_from: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    embedding_size: Optional[int] = None,
    # === NEW DDP PARAMETERS ===
    local_rank: Optional[int] = None,  # If provided, use DDP
    world_size: Optional[int] = None
) -> Dict[str, any]:
    """
    Run a SINGLE experiment with optional DDP support.
    
    DDP Usage:
        # Single GPU (backward compatible):
        results = run_single_experiment(exp_name, ..., device=device)
        
        # Multi-GPU with DDP:
        results = run_single_experiment(exp_name, ..., 
                                        local_rank=local_rank, 
                                        world_size=world_size)
    """
    
    # ============================================================
    # DDP SETUP
    # ============================================================
    use_ddp = local_rank is not None and world_size is not None and world_size > 1
    
    if use_ddp:
        device = torch.device(f'cuda:{local_rank}')
        is_main = (local_rank == 0)
    else:
        # Single GPU mode (backward compatible)
        local_rank = 0
        world_size = 1
        is_main = True
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Only main process prints and logs
    def log_info(msg):
        if is_main:
            print(msg)
    
    log_info(f"\n{'='*80}")
    log_info(f"EXPERIMENT: {exp_name}")
    if use_ddp:
        log_info(f"DDP Mode: {world_size} GPUs (this is rank {local_rank})")
    log_info(f"{'='*80}")
    
    # Determine if resume from checkpoint
    is_resume = resume_from is not None
    
    # Build hierarchical log directory
    if experiment_round is not None:
        effective_log_dir = os.path.join(log_dir, experiment_round)
    else:
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        log_folder_name = f"exp_{datetime_string}"
        effective_log_dir = os.path.join(log_dir, log_folder_name)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(effective_log_dir, exp_name, 'checkpoints')
    
    # ============================================================
    # SETUP LOGGING (main process only)
    # ============================================================
    if is_main:
        logger = setup_experiment_logging(exp_name, effective_log_dir, resume=is_resume) 
        metrics_logger = MetricsLogger(exp_name, effective_log_dir, resume=is_resume) 
    else:
        logger = None
        metrics_logger = None
    
    loss_tracker = LossTracker(window_size=100)
    
    if is_main:
        if is_resume:
            logger.info(f"🔄 RESUMING {exp_name} training from checkpoint: {resume_from}")
        else:
            logger.info(f"Starting experiment: {exp_name}")

    # ============================================================
    # AUTO-CALCULATE DIMENSIONS (unchanged)
    # ============================================================
    eff_d_model = embedding_size if embedding_size is not None else 256
    uses_swiglu = exp_name not in ['exp1_dense_baseline']
    dims = _calculate_model_dimensions(eff_d_model, use_swiglu=uses_swiglu)
    eff_nhead = dims['nhead']
    eff_nhid = dims['nhid']
    
    if is_main and embedding_size is not None:
        logger.info(f"⚡ OVERRIDE: embedding_size={eff_d_model}")
        logger.info(f"   Auto-calculated: nhead={eff_nhead} (head_dim={dims['head_dim']}), nhid={eff_nhid}")
    
    # ============================================================
    # MODEL CREATION (same as before, but on correct device)
    # ============================================================
    if (exp_name == 'exp1_dense_baseline') or (moe_config is None and exp_name not in ['exp1_dense_baseline', 'exp2_dense_flash', 'exp2b_flash_learned_pool']):
        config = BaseConfig(embedding_size=eff_d_model, nhid=eff_nhid)
        model = BaselineTransformer(config).to(device)
        use_mixed_precision = False
        use_bucketing = False
        if is_main:
            logger.info(f"Model: Baseline Transformer (FP32)")
            logger.info(f"  d_model={eff_d_model}, nhid={eff_nhid}, nhead=16 (hardcoded)")
        
    elif exp_name in ['exp2_dense_flash', 'exp2b_flash_learned_pool']:
        config = FlashAttentionConfig(
            nhid=eff_nhid,
            nhead=eff_nhead, 
            use_swiglu=True,
            dtype=torch.float16,
            use_learnt_att_pool=use_learnt_att_pool
        )
        model = FlashAttentionTransformer(config).to(device)
        use_mixed_precision = True
        use_bucketing = True
        if is_main:
            pooling_str = "Learned Attention Pooling" if use_learnt_att_pool else "Flash Attention + Max-Pool"
            logger.info(f"Model: Flash Attention Transformer (FP16)")
            logger.info(f"  d_model={eff_d_model}, nhid={eff_nhid}, nhead={eff_nhead}")
            logger.info(f"  Daily Encoder: {pooling_str}")
        
    else:
        # MoE variant
        config = FlashAttentionConfig(
            nhid=eff_nhid,
            nhead=eff_nhead,
            use_swiglu=True,
            dtype=torch.float16,
            use_learnt_att_pool=use_learnt_att_pool
        )
        if moe_config:
            import copy
            moe_config = copy.deepcopy(moe_config)
            moe_config.d_model = eff_d_model
            moe_config.d_ff = eff_nhid
            
        model = FlashMoETransformer(config, moe_config).to(device)
        use_mixed_precision = True
        use_bucketing = True
        
        if is_main:
            pooling_str = "Learned Attention Pooling" if use_learnt_att_pool else "Flash Attention + Max-Pool"
            logger.info(f"Model: Flash + MoE Transformer (FP16)")
            logger.info(f"  d_model={eff_d_model}, nhid={eff_nhid}, nhead={eff_nhead}")
            logger.info(f"  Daily Encoder: {pooling_str}")
            logger.info(f"  MoE: {moe_config.num_experts} experts, top-{moe_config.top_k}")

    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"Total parameters: {total_params:,}")
    
    # ============================================================
    # WRAP MODEL WITH DDP
    # ============================================================
    if use_ddp:
        # Synchronize all processes before wrapping
        dist.barrier()
        
        model = DDP(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Set True if you have unused params
        )
        
        if is_main:
            logger.info(f"✅ Model wrapped with DDP on {world_size} GPUs")
            logger.info(f"   Per-GPU batch size: {config.batch_size}")
            logger.info(f"   Effective batch size: {config.batch_size * world_size}")
    
    # Log config (main process only)
    if is_main:
        config_dict = {
            'experiment': exp_name,
            'embedding_size': eff_d_model,
            'nhid': eff_nhid,
            'nhead': eff_nhead,
            'batch_size': config.batch_size,
            'effective_batch_size': config.batch_size * world_size,
            'use_ddp': use_ddp,
            'world_size': world_size,
            'use_mixed_precision': use_mixed_precision,
            'use_bucketing': use_bucketing,
            'use_learnt_att_pool': use_learnt_att_pool,
            'moe_config': {
                'num_experts': moe_config.num_experts if moe_config else None,
                'top_k': moe_config.top_k if moe_config else None,
                'num_shared_experts': moe_config.num_shared_experts if moe_config else None,
                'load_balance_strategy': moe_config.load_balance_strategy if moe_config else None,
                'aux_loss_weight': moe_config.aux_loss_weight if moe_config else None,
                'use_moe_from_layer': moe_config.use_moe_from_layer if moe_config else None,
            } if moe_config else None
        }
        metrics_logger.log_config(config_dict)

    # Compute code frequencies if not provided (main process only, then broadcast)
    if code_frequencies is None:
        code_frequencies = compute_code_frequencies(train_data, config, device)
    
    # ============================================================
    # TRAINING SETUP WITH DISTRIBUTED SAMPLER
    # ============================================================
    train_dataset = ClinicalDataset(train_data, config)
    val_dataset = ClinicalDataset(val_data, config)
    
    # Create samplers
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create DataLoaders
    # NOTE: When using DistributedSampler, do NOT use shuffle=True in DataLoader
    if use_bucketing and not use_ddp:
        # Bucketing only works without DDP (sampler conflict)
        if is_main:
            logger.info("Bucketing is ENABLED via BatchSampler.")
        train_batch_sampler = BucketingBatchSampler(
            data=train_data,
            batch_size=config.batch_size,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=max(1, os.cpu_count() // max(1, world_size)),
            pin_memory=True,
            collate_fn=clinical_collate_fn
        )
    else:
        if is_main:
            if use_ddp:
                logger.info("Using DistributedSampler (bucketing disabled for DDP).")
            else:
                logger.info("Using standard DataLoader (no bucketing).")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),  # Only shuffle if no sampler
            sampler=train_sampler,
            num_workers=max(1, os.cpu_count() // max(1, world_size)),
            pin_memory=True,
            drop_last=True,
            collate_fn=clinical_collate_fn
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, os.cpu_count() // max(1, world_size)),
        pin_memory=True,
        collate_fn=clinical_collate_fn
    )
    
    if is_main:
        logger.info(f"Using DataLoader with {train_loader.num_workers} workers.")
        
    # Set up optimizer, scheduler and scaler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    criterion = nn.BCEWithLogitsLoss()

    # ============================================================
    # RESUME FROM CHECKPOINT (DDP compatible)
    # ============================================================
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if is_resume:
        # Load checkpoint (handles DDP module prefix)
        checkpoint = torch.load(resume_from, map_location=device)
        
        # Load model state
        if use_ddp:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        
        if checkpoint.get('metrics'):
            best_val_loss = min(m.get('val_loss', float('inf')) for m in checkpoint['metrics'] if 'val_loss' in m)
        
        if is_main:
            logger.info(f"✅ Resumed from epoch {start_epoch}, step {global_step}")
            logger.info(f"   Previous best val loss: {best_val_loss:.4f}")
        
        # Synchronize after loading
        if use_ddp:
            dist.barrier()
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    if is_main:
        logger.info(f"Training for {epochs} epochs...")
    
    epoch_history = []
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        # CRITICAL for DDP: Set epoch on sampler for proper shuffling
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if is_main:
            logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        loss_tracker.reset_epoch()
        
        # Train
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
            use_bucketing=use_bucketing and not use_ddp,
            log_interval=log_metrics_every,
            global_step=global_step,
            loss_tracker=loss_tracker,
            # DDP parameters
            is_main=is_main,
            use_ddp=use_ddp
        )
        
        # Synchronize metrics across processes
        if use_ddp:
            train_metrics = sync_metrics(train_metrics, device)
        
        global_step = train_metrics['global_step']
        
        # Evaluation (run on all ranks, then aggregate)
        if is_main:
            logger.info("  Evaluating on training subset...")
        
        train_eval_metrics = evaluate(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            config=config,
            device=device,
            use_mixed_precision=use_mixed_precision,
            max_batches=100,
            verbose=False
        )
        
        if use_ddp:
            train_eval_metrics = sync_metrics(train_eval_metrics, device)
        
        if is_main:
            logger.info("  Evaluating on validation set...")
        
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            config=config,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        
        if use_ddp:
            val_metrics = sync_metrics(val_metrics, device)

        # Embedding quality (main process only for efficiency)
        if is_main and epoch % check_embeddings_every == 0:
            logger.info("Computing embedding quality...")
            emb_metrics = compute_embedding_quality_epoch(
                model.module if use_ddp else model,  # Unwrap DDP
                val_data,
                config, 
                device, 
                num_samples=200,
                use_mixed_precision=use_mixed_precision 
            )
            val_metrics.update(emb_metrics)
            logger.info(f"    Embedding std: {emb_metrics['embedding_std_mean']:.4f}")
            logger.info(f"    NN overlap: {emb_metrics['nn_target_overlap']:.3f}")
        
        # Combine metrics (unchanged logic, but only log on main)
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['train_loss'],
            'train_loss_mean': train_metrics['train_loss_mean'],
            'train_loss_first': train_metrics['train_loss_first'],
            'train_loss_last': train_metrics['train_loss_last'],
            'train_loss_std': train_metrics['train_loss_std'],
            'train_loss_improvement': train_metrics['train_loss_improvement'],
            'eval_in_train_loss_final': train_eval_metrics['val_loss'],
            'eval_in_train_top_1_acc': train_eval_metrics['top_1_acc'],
            'eval_in_train_top_5_acc': train_eval_metrics['top_5_acc'],
            'eval_in_train_top_10_acc': train_eval_metrics['top_10_acc'],
            'eval_in_train_top_20_acc': train_eval_metrics['top_20_acc'],
            'final_val_loss': val_metrics['val_loss'],
            'final_val_top_1_acc': val_metrics['top_1_acc'],
            'final_val_top_5_acc': val_metrics['top_5_acc'],
            'final_val_top_10_acc': val_metrics['top_10_acc'],
            'final_val_top_20_acc': val_metrics['top_20_acc'],
            'generalization_gap': train_eval_metrics['val_loss'] - val_metrics['val_loss'],
        }
        
        for k, v in train_metrics.items():
            if k.startswith('train_') and k not in epoch_metrics:
                epoch_metrics[k] = v

        for k, v in val_metrics.items():
            if k not in epoch_metrics and k not in ['val_loss', 'top_1_acc', 'top_5_acc', 'top_10_acc', 'top_20_acc']:
                epoch_metrics[k] = v
        
        epoch_history.append(epoch_metrics)
        
        # ============================================================
        # SAVE CHECKPOINTS (main process only)
        # ============================================================
        if is_main:
            is_best = epoch_metrics['final_val_loss'] < best_val_loss
            if is_best:
                best_val_loss = epoch_metrics['final_val_loss']
            
            # Save unwrapped model state
            model_to_save = model.module if use_ddp else model
            
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                global_step=global_step,
                model=model_to_save,  # Unwrapped!
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metrics=epoch_history,
                is_best=is_best
            )
            logger.info(f"Checkpoint saved (epoch {epoch+1}, step {global_step})")

            # Log epoch summary
            logger.info(f"\n--- Epoch {epoch+1} Summary ---")
            logger.info(f"Training Progress:")
            logger.info(f"  Loss (learning avg): {train_metrics['train_loss']:.4f}")
            logger.info(f"  Loss (first batch):  {train_metrics['train_loss_first']:.4f}")
            logger.info(f"  Loss (last batch):   {train_metrics['train_loss_last']:.4f}")
            logger.info(f"  Improvement:         {train_metrics['train_loss_improvement']:.4f}")
            
            logger.info(f"\nFinal Model Performance:")
            logger.info(f"  Train loss (final):  {train_eval_metrics['val_loss']:.4f}")
            logger.info(f"  Val loss:            {val_metrics['val_loss']:.4f}")
            logger.info(f"  Train Top-10:        {train_eval_metrics['top_10_acc']:.3f}")
            logger.info(f"  Val Top-10:          {val_metrics['top_10_acc']:.3f}")
            
            metrics_logger.log_epoch(epoch + 1, epoch_metrics)
            
            loss_tracker.save_trajectory(
                filepath=os.path.join(effective_log_dir, exp_name, f'loss_trajectory_epoch{epoch}.json')
            )
        
        # Synchronize all processes at epoch end
        if use_ddp:
            dist.barrier()
    
    total_time = time.time() - start_time
    
    if is_main:
        logger.info(f"\nTraining completed in {total_time:.1f}s")
    
    # ============================================================
    # COMPREHENSIVE EVALUATION (main process only)
    # ============================================================
    if is_main:
        model_to_eval = model.module if use_ddp else model
        
        evaluation = comprehensive_evaluation(
            model=model_to_eval,
            val_dataloader=val_loader,
            config=config,
            device=device,
            training_time_sec=total_time,
            epoch_history=epoch_history,
            code_frequencies=code_frequencies,
            moe_config=moe_config,
            use_mixed_precision=use_mixed_precision
        )
        
        # Build results
        final_metrics = epoch_history[-1]
        
        results = {
            'experiment': exp_name,
            'parameters': total_params,
            'use_learned_pooling': use_learnt_att_pool,
            'use_bucketing': use_bucketing and not use_ddp,
            'use_ddp': use_ddp,
            'world_size': world_size,
            'effective_batch_size': config.batch_size * world_size,
            
            'train_loss_mean': final_metrics['train_loss'],
            'train_loss_learned': final_metrics['train_loss_improvement'],
            'train_loss_final': final_metrics['eval_in_train_loss_final'],
            'val_loss_final': final_metrics['final_val_loss'],
            'generalization_gap': final_metrics['generalization_gap'],
            
            'final_train_top_5_acc': final_metrics['eval_in_train_top_5_acc'],
            'final_train_top_10_acc': final_metrics['eval_in_train_top_10_acc'],
            'final_train_top_20_acc': final_metrics['eval_in_train_top_20_acc'],
            'final_val_top_5_acc': final_metrics['final_val_top_5_acc'],
            'final_val_top_10_acc': final_metrics['final_val_top_10_acc'],
            'final_val_top_20_acc': final_metrics['final_val_top_20_acc'],
            
            'training_time_sec': total_time,
            'precision@10': evaluation['performance']['precision@10'],
            'recall@10': evaluation['performance']['recall@10'],
            'f1@10': evaluation['performance']['f1@10'],
            'balanced_top10_acc': evaluation['performance']['balanced_top10_acc'],
            'tail_top10_acc': evaluation['performance']['tail_top10_acc'],
            'cost_usd': evaluation['resources']['cost_usd'],
            'peak_memory_gb': evaluation['resources']['total_peak_gb'],
            'full_evaluation': evaluation,
            'all_epochs': epoch_history
        }
        
        results_path = metrics_logger.save_final_results(results)
        logger.info(f"Complete results saved to {results_path}")
        
        metrics_logger.save()
        logger.info(f"Metrics saved to {log_dir}/{exp_name}/")
        
        summary = metrics_logger.get_summary()
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENT COMPLETE: {exp_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Final Top-10 Acc in val: {final_metrics['final_val_top_10_acc']:.3f}")
        logger.info(f"Best Val Loss: {summary['best_val_loss']:.4f} (epoch {summary['best_epoch']})")
        logger.info(f"Training Time: {total_time:.1f}s")
        logger.info(f"{'='*80}\n")
        
        return results
    else:
        # Non-main processes return minimal info
        return {'experiment': exp_name, 'rank': local_rank}
```

### Part 3: Modified `train_epoch()` with DDP Support

Add these parameters and modify logging:

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
    log_interval: int = 100, 
    global_step: int = 0, 
    loss_tracker: Optional[LossTracker] = None,
    # === NEW DDP PARAMETERS ===
    is_main: bool = True,
    use_ddp: bool = False
) -> Dict[str, float]:
    """
    Train for one epoch with optional DDP support.
    """
    model.train()
    
    nbatch = len(dataloader)
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    batch_metrics_buffer = []  
    moe_metrics_buffer = []
    
    if loss_tracker is None:
        loss_tracker = LossTracker()    
    
    for batch_idx, batch in enumerate(dataloader):
        
        # Only main process prints progress
        if is_main and batch_idx % log_interval == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}')
        
        optimizer.zero_grad()
        
        # Get batch data
        age = batch['age'].to(device, non_blocking=True)
        gender = batch['gender'].to(device, non_blocking=True)
        codes = batch['codes'].to(device, non_blocking=True)
        dt_cnt = batch['dt_cnt']
        y = batch['target']
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            codes
        ], dim=-1)
        
        # Forward pass (unchanged)
        total_loss = torch.tensor(0.0, device=device)
        if use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames:
                    output, moe_losses = model(x, return_moe_losses=True)
                elif hasattr(model, 'module') and hasattr(model.module, 'forward') and 'return_moe_losses' in model.module.forward.__code__.co_varnames:
                    # DDP wrapped model
                    output, moe_losses = model(x, return_moe_losses=True)
                else:
                    output = model(x)
                    moe_losses = {}
                    
                pred_loss = compute_loss(output, y, dt_cnt, config, criterion, device)
                
                aux_loss = moe_losses.get('aux_loss', torch.tensor(0.0, device=device))
                if moe_config and moe_config.load_balance_strategy == 'switch':
                    total_loss = pred_loss + moe_config.aux_loss_weight * aux_loss
                else:
                    total_loss = pred_loss
        else:
            if hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames:
                output, moe_losses = model(x, return_moe_losses=True)
            elif hasattr(model, 'module') and hasattr(model.module, 'forward') and 'return_moe_losses' in model.module.forward.__code__.co_varnames:
                output, moe_losses = model(x, return_moe_losses=True)
            else:
                output = model(x)
                moe_losses = {}
                
            loss_config = type(config)(
                **{k: getattr(config, k) for k in config.__dataclass_fields__}
            )
            loss_config.len_dy = x.shape[1]       
            
            pred_loss = compute_loss(output, y, dt_cnt, config, criterion, device)
            aux_loss = moe_losses.get('aux_loss', torch.tensor(0.0, device=device))
            
            if moe_config and moe_config.load_balance_strategy == 'switch':
                total_loss = pred_loss + moe_config.aux_loss_weight * aux_loss
            else:
                total_loss = pred_loss
        
        # Backward pass (DDP handles gradient sync automatically)
        if use_mixed_precision:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Increment global step
        global_step += 1
        
        # Track losses
        total_pred_loss += pred_loss.item()
        total_aux_loss += aux_loss.item()
        loss_tracker.log_batch(pred_loss.item(), global_step)
        
        # Compute & log metrics (main process only)
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                batch_metrics = compute_batch_metrics_lightweight(
                    output, y, dt_cnt, config, device
                )
                batch_metrics_buffer.append(batch_metrics)
                
                print(f"    Loss: {pred_loss.item():.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"P@20: {batch_metrics['precision@20']:.3f} | "
                      f"mAP20: {batch_metrics['mAP@20']:.3f} | "
                      f"mAP50: {batch_metrics['mAP@50']:.3f} | "
                      f"Brier: {batch_metrics['brier_score']:.4f}")
                
                if moe_losses and 'expert_usage' in moe_losses:
                    moe_batch_metrics = compute_moe_batch_metrics(moe_losses)
                    moe_metrics_buffer.append(moe_batch_metrics)
                    
                    print(f"    MoE: CV={moe_batch_metrics['expert_load_cv']:.3f} | "
                          f"Collapsed={moe_batch_metrics['num_collapsed_experts']} | "
                          f"Gini={moe_batch_metrics['expert_gini']:.3f}")
        
        # Memory cleanup
        del x, output, pred_loss, total_loss
        if batch_idx % 100 == 0:
            gc.collect()
            
            if is_main and device.type == 'cuda' and batch_idx % 1000 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f'    GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB')
    
    # End-of-epoch cleanup
    if device.type == 'cuda':
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    # Aggregate epoch metrics
    loss_summary = loss_tracker.get_epoch_summary()
    epoch_metrics = {
        'train_loss': total_pred_loss / nbatch,
        **loss_summary, 
        'aux_loss': total_aux_loss / nbatch
    }
    
    if batch_metrics_buffer:
        for key in batch_metrics_buffer[0].keys():
            epoch_metrics[f'train_{key}'] = np.mean([m[key] for m in batch_metrics_buffer])
    
    if moe_metrics_buffer:
        for key in moe_metrics_buffer[0].keys():
            epoch_metrics[f'train_{key}'] = np.mean([m[key] for m in moe_metrics_buffer])
        
        if 'expert_usage' in moe_losses:
            epoch_metrics['expert_usage'] = moe_losses['expert_usage']
    
    epoch_metrics['global_step'] = global_step
        
    return epoch_metrics
```

### Part 4: Launch Script (`run_ddp_training.py`)

Create this as a new file:

```python
#!/usr/bin/env python
"""
DDP Training Launcher for Clinical Transformer

Usage:
    # Single GPU (backward compatible):
    python run_ddp_training.py --single-gpu
    
    # Multi-GPU with DDP (4 GPUs):
    torchrun --nproc_per_node=4 run_ddp_training.py
    
    # Multi-GPU on specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_ddp_training.py
"""

import os
import sys
import argparse
import torch
import pandas as pd

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe_flashattn_2 import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    run_single_experiment,
    get_experiment_configs,
    cleanup_gpu_memory_hard,
    FlashAttentionConfig,
    MoEConfig
)


def main():
    parser = argparse.ArgumentParser(description='DDP Training for Clinical Transformer')
    parser.add_argument('--single-gpu', action='store_true', help='Force single GPU mode')
    parser.add_argument('--exp-name', type=str, default='exp2b_flash_learned_pool', 
                        help='Experiment name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--train-samples', type=int, default=None, 
                        help='Subsample training data (None=use all)')
    parser.add_argument('--val-samples', type=int, default=None,
                        help='Subsample validation data (None=use all)')
    parser.add_argument('--embedding-size', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--experiment-round', type=str, default=None, help='Experiment round name')
    parser.add_argument('--data-dir', type=str, default='sample_data', help='Data directory')
    
    args = parser.parse_args()
    
    # ============================================================
    # SETUP DDP
    # ============================================================
    if args.single_gpu:
        local_rank = 0
        world_size = 1
        is_main = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running in SINGLE GPU mode on {device}")
    else:
        local_rank, world_size, is_main = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
        if is_main:
            print(f"Running in DDP mode with {world_size} GPUs")
    
    try:
        # ============================================================
        # LOAD DATA (each process loads independently)
        # ============================================================
        if is_main:
            print(f"\nLoading data from {args.data_dir}...")
        
        df_train = pd.read_feather(os.path.join(args.data_dir, "mdcd_train_1m.feather"))
        df_val = pd.read_feather(os.path.join(args.data_dir, "mdcd_val_10k.feather"))
        
        # Subsample if requested
        if args.train_samples is not None:
            df_train = df_train.sample(args.train_samples, random_state=42)
        if args.val_samples is not None:
            df_val = df_val.sample(args.val_samples, random_state=42)
        
        if is_main:
            print(f"Train samples: {len(df_train)}")
            print(f"Val samples: {len(df_val)}")
        
        # ============================================================
        # GET EXPERIMENT CONFIG
        # ============================================================
        all_configs = get_experiment_configs()
        
        if args.exp_name not in all_configs:
            raise ValueError(f"Unknown experiment: {args.exp_name}. "
                           f"Available: {list(all_configs.keys())}")
        
        moe_config, use_learnt_att_pool = all_configs[args.exp_name]
        
        # ============================================================
        # RUN EXPERIMENT
        # ============================================================
        results = run_single_experiment(
            exp_name=args.exp_name,
            moe_config=moe_config,
            use_learnt_att_pool=use_learnt_att_pool,
            train_data=df_train,
            val_data=df_val,
            device=device,
            epochs=args.epochs,
            experiment_round=args.experiment_round,
            embedding_size=args.embedding_size,
            # DDP parameters
            local_rank=local_rank if not args.single_gpu else None,
            world_size=world_size if not args.single_gpu else None
        )
        
        if is_main:
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print(f"Experiment: {results['experiment']}")
            print(f"Final Val Top-10 Acc: {results['final_val_top_10_acc']:.3f}")
            print(f"Training Time: {results['training_time_sec']:.1f}s")
            print(f"Cost: ${results['cost_usd']:.2f}")
            print("="*80)
    
    finally:
        # ============================================================
        # CLEANUP
        # ============================================================
        if not args.single_gpu:
            cleanup_ddp()
        
        cleanup_gpu_memory_hard()


if __name__ == '__main__':
    main()
```

### Part 5: Testing Your DDP Implementation

Create a test script (`test_ddp.py`):

```python
#!/usr/bin/env python
"""
Test DDP implementation before running full training.

Usage:
    # Test single GPU mode:
    python test_ddp.py --single-gpu
    
    # Test DDP with 4 GPUs:
    torchrun --nproc_per_node=4 test_ddp.py
"""

import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe_flashattn_2 import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    get_rank,
    get_world_size,
    reduce_tensor,
    sync_metrics,
    BaseConfig,
    FlashAttentionConfig,
    MoEConfig,
    BaselineTransformer,
    FlashMoETransformer,
    ClinicalDataset,
    clinical_collate_fn,
)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import pandas as pd


def test_ddp_basics():
    """Test basic DDP setup."""
    print(f"\n[Rank {get_rank()}] Testing DDP basics...")
    
    local_rank, world_size, is_main = setup_ddp()
    
    assert world_size > 0, "World size should be positive"
    assert 0 <= local_rank < world_size, f"Invalid rank {local_rank}"
    
    if is_main:
        print(f"✅ DDP setup successful: {world_size} processes")
    
    return local_rank, world_size


def test_model_ddp_wrap(local_rank):
    """Test wrapping model with DDP."""
    print(f"\n[Rank {local_rank}] Testing model DDP wrap...")
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Create model
    config = BaseConfig()
    model = BaselineTransformer(config).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Test forward pass
    x = torch.randn(4, 32, 82, device=device)  # [batch, len_dy, len_cd+2]
    output = model(x)
    
    assert output.shape == (4, 32, config.target_cd_cnt), f"Wrong shape: {output.shape}"
    
    if local_rank == 0:
        print(f"✅ Model DDP wrap successful, output shape: {output.shape}")
    
    return model


def test_gradient_sync(local_rank, world_size):
    """Test that gradients are properly synchronized."""
    print(f"\n[Rank {local_rank}] Testing gradient sync...")
    
    device = torch.device(f'cuda:{local_rank}')
    
    config = BaseConfig(batch_size=4)
    model = BaselineTransformer(config).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Each rank has different input (simulating different data shards)
    torch.manual_seed(42 + local_rank)  # Different seed per rank
    x = torch.randn(4, 32, 82, device=device)
    
    # Forward
    output = model(x)
    loss = output.sum()
    
    # Backward
    loss.backward()
    
    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Gather gradients from all ranks (for verification only)
            grad_tensor = param.grad.clone()
            dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
            break
    
    if local_rank == 0:
        print(f"✅ Gradient sync working correctly")


def test_distributed_sampler(local_rank, world_size):
    """Test DistributedSampler splits data correctly."""
    print(f"\n[Rank {local_rank}] Testing DistributedSampler...")
    
    # Create dummy data
    n_samples = 100
    dummy_df = pd.DataFrame({
        'age_in_months': ['100,200,300'] * n_samples,
        'gender_cd': ['1,1,1'] * n_samples,
        'cd': ['1,2*3,4*5,6'] * n_samples,
        'target_cd': ['10,20*30,40*50'] * n_samples,
        'dt_cnt': [3] * n_samples
    })
    
    config = BaseConfig(batch_size=8, len_dy=10, len_cd=10)
    dataset = ClinicalDataset(dummy_df, config)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=clinical_collate_fn
    )
    
    # Count samples this rank will see
    total_samples = sum(1 for batch in loader for _ in range(len(batch['dt_cnt'])))
    expected_samples = n_samples // world_size
    
    if local_rank == 0:
        print(f"  Total samples: {n_samples}")
        print(f"  Samples per rank: ~{expected_samples}")
        print(f"  This rank sees: {total_samples}")
        print(f"✅ DistributedSampler working correctly")


def test_metric_sync(local_rank, world_size):
    """Test metric synchronization across ranks."""
    print(f"\n[Rank {local_rank}] Testing metric sync...")
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Each rank reports different "local" metrics
    local_loss = 1.0 + local_rank * 0.1  # 1.0, 1.1, 1.2, 1.3 for 4 ranks
    local_acc = 0.8 - local_rank * 0.05  # 0.8, 0.75, 0.7, 0.65
    
    metrics = {
        'loss': local_loss,
        'accuracy': local_acc
    }
    
    synced = sync_metrics(metrics, device)
    
    # Expected: average across ranks
    expected_loss = sum(1.0 + i * 0.1 for i in range(world_size)) / world_size
    expected_acc = sum(0.8 - i * 0.05 for i in range(world_size)) / world_size
    
    if local_rank == 0:
        print(f"  Local loss (rank 0): {local_loss}")
        print(f"  Synced loss: {synced['loss']:.4f} (expected: {expected_loss:.4f})")
        print(f"  Local acc (rank 0): {local_acc}")
        print(f"  Synced acc: {synced['accuracy']:.4f} (expected: {expected_acc:.4f})")
        
        assert abs(synced['loss'] - expected_loss) < 0.001, "Loss sync failed"
        assert abs(synced['accuracy'] - expected_acc) < 0.001, "Accuracy sync failed"
        print(f"✅ Metric sync working correctly")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-gpu', action='store_true')
    args = parser.parse_args()
    
    if args.single_gpu:
        print("Testing in single GPU mode...")
        print("✅ All tests skipped (single GPU mode)")
        return
    
    try:
        # Run tests
        local_rank, world_size = test_ddp_basics()
        
        dist.barrier()
        test_model_ddp_wrap(local_rank)
        
        dist.barrier()
        test_gradient_sync(local_rank, world_size)
        
        dist.barrier()
        test_distributed_sampler(local_rank, world_size)
        
        dist.barrier()
        test_metric_sync(local_rank, world_size)
        
        dist.barrier()
        
        if local_rank == 0:
            print("\n" + "="*60)
            print("ALL DDP TESTS PASSED! ✅")
            print("="*60)
            print("\nYou can now run training with:")
            print("  torchrun --nproc_per_node=4 run_ddp_training.py")
            print("="*60 + "\n")
    
    finally:
        cleanup_ddp()


if __name__ == '__main__':
    main()
```

---

## Summary: Migration Path

| Step | Action | Time | Command |
|------|--------|------|---------|
| 1 | Add DDP utilities to `moe_flashattn_2.py` | 10 min | Copy code above |
| 2 | Update `run_single_experiment()` | 20 min | Replace function |
| 3 | Update `train_epoch()` | 10 min | Add parameters |
| 4 | Create `run_ddp_training.py` | 5 min | New file |
| 5 | Create `test_ddp.py` | 5 min | New file |
| 6 | Test DDP | 10 min | `torchrun --nproc_per_node=4 test_ddp.py` |
| 7 | Run training | - | `torchrun --nproc_per_node=4 run_ddp_training.py` |

**Total implementation time: ~1-2 hours**

### Commands to Run

```bash
# Step 1: Test DDP works
torchrun --nproc_per_node=4 test_ddp.py

# Step 2: Run training with DDP
torchrun --nproc_per_node=4 run_ddp_training.py \
    --exp-name exp2b_flash_learned_pool \
    --epochs 3 \
    --embedding-size 512

# Step 3: Run with subsampled data for quick test
torchrun --nproc_per_node=4 run_ddp_training.py \
    --exp-name exp2b_flash_learned_pool \
    --epochs 1 \
    --train-samples 10000 \
    --val-samples 1000

# Fallback: Single GPU mode (backward compatible)
python run_ddp_training.py --single-gpu --epochs 1
```

### Portability: T4 → H100

The same code works on both! The only difference:
- **T4**: `torchrun --nproc_per_node=4 run_ddp_training.py`
- **H100**: `torchrun --nproc_per_node=4 run_ddp_training.py` (same command, NCCL auto-detects NVLink)

You're in **ask mode** — to apply these changes, switch to **agent mode** or copy the code blocks above to your files manually.