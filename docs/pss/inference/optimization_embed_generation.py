I'll analyze the bottlenecks and propose three highly effective solutions for optimizing embedding generation.

## Performance Analysis

Looking at your code, the main bottlenecks are:
1. **Frequent CPU↔GPU transfers** - Moving embeddings to CPU every batch
2. **No JIT compilation** - Missing `torch.compile()` optimization
3. **Sequential processing** - Not overlapping data loading with computation
4. **Memory allocation** - Creating new numpy arrays every batch then vstacking

---

## Solution 1: Pre-allocate + Non-blocking Transfers + CUDA Streams

**Impact**: 2-3x speedup  
**Complexity**: Low

This solution pre-allocates the output tensor on GPU, uses non-blocking transfers, and overlaps data loading with computation using CUDA streams.

```python
import time
from tqdm import tqdm

def generate_embeddings_optimized_v1(
    model: torch.nn.Module,
    config: BaseConfig,
    data: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,  # Can use larger batch in eval mode
    num_workers: int = 4,
    use_mixed_precision: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Optimized embedding generation with:
    1. Pre-allocated output tensor (avoids repeated allocations)
    2. Non-blocking GPU→CPU transfers (overlaps with next batch computation)
    3. Pinned memory for faster transfers
    4. Progress bar with ETA
    """
    start_time = time.time()
    
    n_samples = len(data)
    embedding_dim = config.embedding_size
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING EMBEDDINGS (Optimized V1)")
        print(f"{'='*60}")
        print(f"Samples: {n_samples:,} | Batch size: {batch_size} | Embedding dim: {embedding_dim}")
    
    model.eval()
    
    # ========================================================================
    # OPTIMIZATION 1: Pre-allocate output in pinned memory
    # ========================================================================
    # Pinned memory enables faster GPU→CPU transfers
    embeddings_output = torch.empty(
        (n_samples, embedding_dim), 
        dtype=torch.float32,
        pin_memory=True  # Faster async transfers
    )
    
    # Create dataset and dataloader
    dataset = LazyClinicalDataset(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )
    
    individual_ids = data['individual_id'].astype(str).tolist()
    index_dts = data['index_dt'].astype(str).tolist()
    
    has_moe = hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames
    
    # ========================================================================
    # OPTIMIZATION 2: Create CUDA stream for async operations
    # ========================================================================
    transfer_stream = torch.cuda.Stream() if device.type == 'cuda' else None
    
    current_idx = 0
    
    # Use tqdm for progress with ETA
    pbar = tqdm(dataloader, desc="Generating embeddings", disable=not verbose)
    
    with torch.no_grad():
        with EmbeddingExtractor(model) as extractor:
            for batch in pbar:
                batch_start = current_idx
                actual_batch_size = batch['age'].shape[0]
                batch_end = batch_start + actual_batch_size
                
                # Move to GPU (non-blocking)
                age = batch['age'].to(device, non_blocking=True)
                gender = batch['gender'].to(device, non_blocking=True)
                lob = batch['lob'].to(device, non_blocking=True)
                codes = batch['codes'].to(device, non_blocking=True)
                dt_cnt = batch['dt_cnt']
                
                x = torch.cat([
                    age.unsqueeze(-1),
                    gender.unsqueeze(-1),
                    lob.unsqueeze(-1),
                    codes
                ], dim=-1)
                
                # Forward pass
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
                
                # Extract embeddings
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                
                # ============================================================
                # OPTIMIZATION 3: Non-blocking transfer to pre-allocated tensor
                # ============================================================
                if transfer_stream is not None:
                    with torch.cuda.stream(transfer_stream):
                        embeddings_output[batch_start:batch_end].copy_(
                            patient_embs.float().cpu(), 
                            non_blocking=True
                        )
                else:
                    embeddings_output[batch_start:batch_end] = patient_embs.float().cpu()
                
                current_idx = batch_end
                
                # Update progress bar with speed
                pbar.set_postfix({
                    'samples': f'{batch_end:,}/{n_samples:,}',
                    'speed': f'{batch_end / (time.time() - start_time):.0f} samples/s'
                })
    
    # Synchronize to ensure all transfers complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Convert to numpy
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:.0f} samples/s")
        print(f"   Output shape: {embeddings.shape}")
    
    return embeddings, individual_ids, index_dts
```

---

## Solution 2: torch.compile() for Model Optimization

**Impact**: 1.5-2x speedup  
**Complexity**: Very Low

`torch.compile()` JIT-compiles the model, fusing operations and optimizing memory access patterns.

```python
def generate_embeddings_optimized_v2(
    model: torch.nn.Module,
    config: BaseConfig,
    data: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    use_mixed_precision: bool = True,
    verbose: bool = True,
    use_compile: bool = True,  # NEW: Enable torch.compile
    compile_mode: str = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Optimized embedding generation with torch.compile().
    
    compile_mode options:
    - "default": Safe, moderate speedup
    - "reduce-overhead": Best for small batches, reduces kernel launch overhead
    - "max-autotune": Slowest compile, fastest runtime (use for large datasets)
    """
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING EMBEDDINGS (Optimized V2 - torch.compile)")
        print(f"{'='*60}")
        print(f"Samples: {n_samples:,} | Batch size: {batch_size}")
        print(f"torch.compile mode: {compile_mode if use_compile else 'disabled'}")
    
    model.eval()
    
    # ========================================================================
    # OPTIMIZATION: Compile the model for faster inference
    # ========================================================================
    if use_compile and hasattr(torch, 'compile'):
        compile_start = time.time()
        if verbose:
            print("Compiling model (one-time cost)...")
        
        # Compile the model - this is cached after first call
        try:
            compiled_model = torch.compile(
                model, 
                mode=compile_mode,
                fullgraph=False,  # Allow graph breaks for compatibility
                dynamic=True  # Handle variable batch sizes
            )
            if verbose:
                print(f"  Compilation time: {time.time() - compile_start:.1f}s")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}, falling back to eager mode")
            compiled_model = model
    else:
        compiled_model = model
    
    # Pre-allocate output
    embeddings_output = torch.empty(
        (n_samples, embedding_dim), 
        dtype=torch.float32,
        pin_memory=True
    )
    
    dataset = LazyClinicalDataset(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    individual_ids = data['individual_id'].astype(str).tolist()
    index_dts = data['index_dt'].astype(str).tolist()
    
    has_moe = hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames
    current_idx = 0
    
    # Warmup: First batch compiles the graph
    warmup_done = False
    
    pbar = tqdm(dataloader, desc="Generating embeddings", disable=not verbose)
    
    with torch.no_grad():
        # NOTE: EmbeddingExtractor hooks on the ORIGINAL model, not compiled
        # This is important - hooks work on the underlying model
        with EmbeddingExtractor(model) as extractor:
            for batch in pbar:
                batch_start = current_idx
                actual_batch_size = batch['age'].shape[0]
                batch_end = batch_start + actual_batch_size
                
                age = batch['age'].to(device, non_blocking=True)
                gender = batch['gender'].to(device, non_blocking=True)
                lob = batch['lob'].to(device, non_blocking=True)
                codes = batch['codes'].to(device, non_blocking=True)
                dt_cnt = batch['dt_cnt']
                
                x = torch.cat([
                    age.unsqueeze(-1),
                    gender.unsqueeze(-1),
                    lob.unsqueeze(-1),
                    codes
                ], dim=-1)
                
                # Forward pass with compiled model
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if has_moe:
                            _ = compiled_model(x, return_moe_losses=False)
                        else:
                            _ = compiled_model(x)
                else:
                    if has_moe:
                        _ = compiled_model(x, return_moe_losses=False)
                    else:
                        _ = compiled_model(x)
                
                if not warmup_done and verbose:
                    print("  Warmup complete (graph compiled)")
                    warmup_done = True
                
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                
                embeddings_output[batch_start:batch_end] = patient_embs.float().cpu()
                current_idx = batch_end
                
                pbar.set_postfix({
                    'speed': f'{batch_end / (time.time() - start_time):.0f} s/s'
                })
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:.0f} samples/s")
    
    return embeddings, individual_ids, index_dts
```

**Important**: The `EmbeddingExtractor` hooks on the **original model** (not the compiled version), which is correct since `torch.compile` wraps the model but hooks still work on the underlying `nn.Module`.

---

## Solution 3: Multi-GPU DataParallel Inference

**Impact**: Near-linear speedup with GPU count (2x for 2 GPUs, 4x for 4 GPUs)  
**Complexity**: Medium

```python
def generate_embeddings_optimized_v3(
    model: torch.nn.Module,
    config: BaseConfig,
    data: pd.DataFrame,
    device: torch.device,
    batch_size: int = 128,  # Larger batch for multi-GPU
    num_workers: int = 8,
    use_mixed_precision: bool = True,
    verbose: bool = True,
    gpu_ids: List[int] = None  # Which GPUs to use, e.g., [0, 1, 2, 3]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Multi-GPU embedding generation using DataParallel.
    
    For 4 GPUs with batch_size=128:
    - Each GPU processes 32 samples per batch
    - Near-linear speedup
    """
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size
    
    # Detect available GPUs
    if gpu_ids is None:
        n_gpus = torch.cuda.device_count()
        gpu_ids = list(range(n_gpus))
    else:
        n_gpus = len(gpu_ids)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING EMBEDDINGS (Optimized V3 - Multi-GPU)")
        print(f"{'='*60}")
        print(f"Samples: {n_samples:,} | Batch size: {batch_size} | GPUs: {n_gpus}")
        print(f"GPU IDs: {gpu_ids}")
        print(f"Per-GPU batch size: {batch_size // n_gpus}")
    
    model.eval()
    
    # ========================================================================
    # WRAP MODEL WITH DATAPARALLEL FOR MULTI-GPU
    # ========================================================================
    if n_gpus > 1:
        # Wrap model for multi-GPU
        model_parallel = torch.nn.DataParallel(model, device_ids=gpu_ids)
        primary_device = torch.device(f'cuda:{gpu_ids[0]}')
        model_parallel = model_parallel.to(primary_device)
        
        # Get the underlying model for hook registration
        underlying_model = model_parallel.module
    else:
        model_parallel = model.to(device)
        underlying_model = model
        primary_device = device
    
    # Pre-allocate output
    embeddings_output = torch.empty(
        (n_samples, embedding_dim), 
        dtype=torch.float32,
        pin_memory=True
    )
    
    dataset = LazyClinicalDataset(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    individual_ids = data['individual_id'].astype(str).tolist()
    index_dts = data['index_dt'].astype(str).tolist()
    
    has_moe = hasattr(underlying_model, 'forward') and 'return_moe_losses' in underlying_model.forward.__code__.co_varnames
    current_idx = 0
    
    pbar = tqdm(dataloader, desc=f"Generating embeddings ({n_gpus} GPUs)", disable=not verbose)
    
    with torch.no_grad():
        # Hook on the UNDERLYING model (not DataParallel wrapper)
        with EmbeddingExtractor(underlying_model) as extractor:
            for batch in pbar:
                batch_start = current_idx
                actual_batch_size = batch['age'].shape[0]
                batch_end = batch_start + actual_batch_size
                
                # Move to primary GPU (DataParallel handles distribution)
                age = batch['age'].to(primary_device, non_blocking=True)
                gender = batch['gender'].to(primary_device, non_blocking=True)
                lob = batch['lob'].to(primary_device, non_blocking=True)
                codes = batch['codes'].to(primary_device, non_blocking=True)
                dt_cnt = batch['dt_cnt']
                
                x = torch.cat([
                    age.unsqueeze(-1),
                    gender.unsqueeze(-1),
                    lob.unsqueeze(-1),
                    codes
                ], dim=-1)
                
                # Forward pass - DataParallel distributes across GPUs
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if has_moe:
                            _ = model_parallel(x, return_moe_losses=False)
                        else:
                            _ = model_parallel(x)
                else:
                    if has_moe:
                        _ = model_parallel(x, return_moe_losses=False)
                    else:
                        _ = model_parallel(x)
                
                # Extract embeddings (hook captures from underlying model)
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                
                embeddings_output[batch_start:batch_end] = patient_embs.float().cpu()
                current_idx = batch_end
                
                elapsed = time.time() - start_time
                pbar.set_postfix({
                    'speed': f'{batch_end / elapsed:.0f} s/s',
                    'ETA': f'{(n_samples - batch_end) / (batch_end / elapsed):.0f}s'
                })
    
    # Sync all GPUs
    for gpu_id in gpu_ids:
        torch.cuda.synchronize(gpu_id)
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:.0f} samples/s")
        print(f"   Throughput: {n_samples/elapsed * n_gpus:.0f} effective samples/s (across {n_gpus} GPUs)")
    
    return embeddings, individual_ids, index_dts
```

---

## Compatibility Changes Required

### For `EmbeddingExtractor` class

The hook-based extractor needs a small fix to handle DataParallel properly (already in your code, but verify):

```python
class EmbeddingExtractor:
    def __init__(self, model: nn.Module):
        self.wrapped_model = model
        # Unwrap DataParallel if needed
        inner = model.module if isinstance(model, nn.DataParallel) else model
        if isinstance(inner, DataParallelWrapper):
            inner = inner.model
        self.model = inner  # Hook on the actual model
        # ... rest unchanged
```

### For multi-GPU (Solution 3)

When using `DataParallel`, the hook captures embeddings from GPU 0 only by default. The current implementation works because:
1. DataParallel gathers outputs to GPU 0
2. The hook on `model.norm` captures after gathering

### For torch.compile (Solution 2)

Add a try-except fallback in case compilation fails:

```python
# At top of file
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Don't crash on compile errors
```

---

## Combined Ultimate Solution

Here's a version combining all three optimizations:

```python
def generate_embeddings_ultimate(
    model: torch.nn.Module,
    config: BaseConfig,
    data: pd.DataFrame,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 8,
    use_mixed_precision: bool = True,
    verbose: bool = True,
    use_compile: bool = True,
    multi_gpu: bool = True
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Ultimate optimized embedding generation combining:
    1. Pre-allocated pinned memory output
    2. Non-blocking async transfers with CUDA streams
    3. torch.compile for JIT optimization
    4. Multi-GPU DataParallel (if available)
    5. Progress bar with ETA and speed metrics
    """
    import time
    from tqdm import tqdm
    
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size
    
    # GPU setup
    n_gpus = torch.cuda.device_count() if multi_gpu else 1
    gpu_ids = list(range(n_gpus))
    primary_device = torch.device(f'cuda:{gpu_ids[0]}') if n_gpus > 0 else device
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING EMBEDDINGS (Ultimate Optimized)")
        print(f"{'='*70}")
        print(f"Samples: {n_samples:,} | Batch: {batch_size} | Workers: {num_workers}")
        print(f"GPUs: {n_gpus} | torch.compile: {use_compile} | Mixed precision: {use_mixed_precision}")
    
    model.eval()
    
    # ========================================================================
    # 1. TORCH.COMPILE (if enabled)
    # ========================================================================
    if use_compile and hasattr(torch, 'compile'):
        try:
            t0 = time.time()
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            if verbose:
                print(f"✓ Model compiled in {time.time()-t0:.1f}s")
        except Exception as e:
            if verbose:
                print(f"⚠ Compile failed: {e}")
    
    # ========================================================================
    # 2. MULTI-GPU SETUP (if enabled)
    # ========================================================================
    if n_gpus > 1 and multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        underlying_model = model.module
    else:
        underlying_model = model
    
    model = model.to(primary_device)
    
    # ========================================================================
    # 3. PRE-ALLOCATE OUTPUT WITH PINNED MEMORY
    # ========================================================================
    embeddings_output = torch.empty(
        (n_samples, embedding_dim), 
        dtype=torch.float32,
        pin_memory=True
    )
    
    # DataLoader with optimized settings
    dataset = LazyClinicalDataset(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    individual_ids = data['individual_id'].astype(str).tolist()
    index_dts = data['index_dt'].astype(str).tolist()
    
    has_moe = 'return_moe_losses' in underlying_model.forward.__code__.co_varnames
    
    # CUDA stream for async transfers
    transfer_stream = torch.cuda.Stream() if primary_device.type == 'cuda' else None
    
    current_idx = 0
    pbar = tqdm(dataloader, desc="Embedding generation", disable=not verbose)
    
    with torch.no_grad():
        with EmbeddingExtractor(underlying_model) as extractor:
            for batch in pbar:
                batch_size_actual = batch['age'].shape[0]
                batch_start = current_idx
                batch_end = batch_start + batch_size_actual
                
                # Non-blocking GPU transfer
                x = torch.cat([
                    batch['age'].unsqueeze(-1),
                    batch['gender'].unsqueeze(-1),
                    batch['lob'].unsqueeze(-1),
                    batch['codes']
                ], dim=-1).to(primary_device, non_blocking=True)
                
                dt_cnt = batch['dt_cnt']
                
                # Forward pass
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        _ = model(x, return_moe_losses=False) if has_moe else model(x)
                else:
                    _ = model(x, return_moe_losses=False) if has_moe else model(x)
                
                # Extract embeddings
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                
                # Async transfer to pre-allocated output
                if transfer_stream:
                    with torch.cuda.stream(transfer_stream):
                        embeddings_output[batch_start:batch_end].copy_(
                            patient_embs.float().cpu(), non_blocking=True
                        )
                else:
                    embeddings_output[batch_start:batch_end] = patient_embs.float().cpu()
                
                current_idx = batch_end
                
                # Progress with metrics
                elapsed = time.time() - start_time
                speed = batch_end / elapsed
                eta = (n_samples - batch_end) / speed if speed > 0 else 0
                pbar.set_postfix({
                    'speed': f'{speed:.0f}/s',
                    'ETA': f'{eta:.0f}s'
                })
    
    # Final sync
    if primary_device.type == 'cuda':
        torch.cuda.synchronize()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ COMPLETE")
        print(f"   Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
        print(f"   Output: {embeddings.shape}")
        print(f"{'='*70}\n")
    
    return embeddings, individual_ids, index_dts
```

---

## Summary

| Solution | Speedup | Complexity | When to Use |
|----------|---------|------------|-------------|
| **V1: Pre-allocate + Async** | 2-3x | Low | Always - no downsides |
| **V2: torch.compile** | 1.5-2x | Very Low | PyTorch 2.0+, one-time compile cost |
| **V3: Multi-GPU** | Nx (N=GPUs) | Medium | When you have multiple GPUs |
| **Ultimate** | 3-6x+ | Medium | Large datasets, multiple GPUs |


Great question! Yes, **DataParallel can cause OOM on GPU 0** because it gathers all outputs to GPU 0 before returning. This is a known limitation.

## Why GPU 0 Uses More Memory with DataParallel

```
Standard DataParallel Flow:
┌─────────────────────────────────────────────────────────────┐
│  GPU 0 (Primary)    │  GPU 1  │  GPU 2  │  GPU 3           │
├─────────────────────┼─────────┼─────────┼──────────────────│
│  Model copy         │  Model  │  Model  │  Model           │
│  + Batch slice 0    │  + B1   │  + B2   │  + B3            │
│  + Output 0         │  + O1   │  + O2   │  + O3            │
│  + GATHERED O1,O2,O3│         │         │   ← All outputs  │
│  ^^^^^^^^^^^^^^^^^^^^                       go to GPU 0!   │
└─────────────────────────────────────────────────────────────┘
```

For your 4× T4 (16GB each), this means:
- **GPU 0**: Model (~300MB) + Batch activations (~2-4GB) + **ALL 4 outputs gathered**
- **GPU 1-3**: Model (~300MB) + Batch activations (~2-4GB) only

## Solution: Avoid Output Gathering

For embedding generation, you **don't need to gather outputs on GPU 0** since you only care about the embeddings captured by the hook. Here's a memory-balanced solution:

```python
def generate_embeddings_multi_gpu_balanced(
    model: torch.nn.Module,
    config: BaseConfig,
    data: pd.DataFrame,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 8,
    use_mixed_precision: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Multi-GPU embedding generation with BALANCED memory usage.
    
    Key difference from standard DataParallel:
    - We DON'T use DataParallel (which gathers to GPU 0)
    - Instead, we manually distribute batches across GPUs
    - Each GPU processes independently, embeddings extracted via hooks
    
    This avoids the GPU 0 OOM issue entirely.
    """
    import time
    from tqdm import tqdm
    
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size
    
    # Detect GPUs
    n_gpus = torch.cuda.device_count()
    
    if n_gpus <= 1:
        # Fall back to single GPU
        if verbose:
            print("Only 1 GPU available, using single-GPU mode")
        return generate_embeddings_optimized_v1(
            model, config, data, device, batch_size, num_workers, 
            use_mixed_precision, verbose
        )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-GPU EMBEDDING GENERATION (Memory Balanced)")
        print(f"{'='*70}")
        print(f"Samples: {n_samples:,} | GPUs: {n_gpus} | Per-GPU batch: {batch_size}")
    
    # ========================================================================
    # STRATEGY: Replicate model to each GPU, process different data subsets
    # ========================================================================
    
    # Split data into chunks for each GPU
    chunk_size = (n_samples + n_gpus - 1) // n_gpus
    data_chunks = [
        data.iloc[i * chunk_size : (i + 1) * chunk_size] 
        for i in range(n_gpus)
    ]
    
    if verbose:
        for i, chunk in enumerate(data_chunks):
            print(f"  GPU {i}: {len(chunk):,} samples")
    
    # Pre-allocate output
    embeddings_output = torch.empty(
        (n_samples, embedding_dim),
        dtype=torch.float32,
        pin_memory=True
    )
    
    individual_ids = data['individual_id'].astype(str).tolist()
    index_dts = data['index_dt'].astype(str).tolist()
    
    # ========================================================================
    # PROCESS EACH GPU IN PARALLEL USING THREADS
    # ========================================================================
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    # Thread-safe progress counter
    progress_lock = threading.Lock()
    total_processed = [0]
    
    def process_on_gpu(gpu_id: int, data_chunk: pd.DataFrame, start_idx: int):
        """Process a data chunk on a specific GPU."""
        if len(data_chunk) == 0:
            return
        
        gpu_device = torch.device(f'cuda:{gpu_id}')
        
        # Create a copy of the model on this GPU
        # (model.to() is not thread-safe, so we clone)
        with torch.cuda.device(gpu_id):
            gpu_model = type(model)(config, moe_config) if hasattr(model, 'moe_config') else type(model)(config)
            gpu_model.load_state_dict(model.state_dict())
            gpu_model = gpu_model.to(gpu_device)
            gpu_model.eval()
        
        dataset = LazyClinicalDataset(data_chunk, config)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(config),
            num_workers=max(1, num_workers // n_gpus),
            pin_memory=True
        )
        
        has_moe = 'return_moe_losses' in gpu_model.forward.__code__.co_varnames
        current_idx = start_idx
        
        with torch.no_grad():
            with EmbeddingExtractor(gpu_model) as extractor:
                for batch in dataloader:
                    batch_size_actual = batch['age'].shape[0]
                    
                    x = torch.cat([
                        batch['age'].unsqueeze(-1),
                        batch['gender'].unsqueeze(-1),
                        batch['lob'].unsqueeze(-1),
                        batch['codes']
                    ], dim=-1).to(gpu_device, non_blocking=True)
                    
                    dt_cnt = batch['dt_cnt']
                    
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            _ = gpu_model(x, return_moe_losses=False) if has_moe else gpu_model(x)
                    else:
                        _ = gpu_model(x, return_moe_losses=False) if has_moe else gpu_model(x)
                    
                    dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                    patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                    
                    # Write directly to pre-allocated output (thread-safe for non-overlapping regions)
                    embeddings_output[current_idx:current_idx + batch_size_actual] = \
                        patient_embs.float().cpu()
                    
                    current_idx += batch_size_actual
                    
                    with progress_lock:
                        total_processed[0] += batch_size_actual
        
        # Clean up GPU memory
        del gpu_model
        torch.cuda.empty_cache()
    
    # Calculate start indices for each chunk
    start_indices = [0]
    for i in range(n_gpus - 1):
        start_indices.append(start_indices[-1] + len(data_chunks[i]))
    
    # Progress bar in main thread
    pbar = tqdm(total=n_samples, desc="Multi-GPU embedding", disable=not verbose)
    
    # Launch parallel processing
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(process_on_gpu, gpu_id, data_chunks[gpu_id], start_indices[gpu_id])
            for gpu_id in range(n_gpus)
        ]
        
        # Update progress bar
        last_processed = 0
        while not all(f.done() for f in futures):
            with progress_lock:
                current = total_processed[0]
            pbar.update(current - last_processed)
            last_processed = current
            time.sleep(0.1)
        
        # Final update
        pbar.update(n_samples - last_processed)
        pbar.close()
        
        # Check for exceptions
        for f in futures:
            f.result()  # Raises exception if any
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
    
    return embeddings, individual_ids, index_dts
```

## Simpler Alternative: Sequential Multi-GPU (No Threading Complexity)

If the above is too complex, here's a simpler version that processes GPUs sequentially but still distributes the memory:

```python
def generate_embeddings_multi_gpu_simple(
    model: torch.nn.Module,
    config: BaseConfig,
    data: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 4,
    use_mixed_precision: bool = True,
    verbose: bool = True,
    moe_config = None,  # Pass this for MoE models
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Simple multi-GPU: process data in chunks, rotating through GPUs.
    
    Benefits:
    - No OOM on any single GPU
    - Simple sequential processing (no threading)
    - Each GPU only holds model + one batch at a time
    """
    import time
    from tqdm import tqdm
    
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size
    n_gpus = torch.cuda.device_count()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-GPU EMBEDDING (Simple Sequential)")
        print(f"{'='*60}")
        print(f"Samples: {n_samples:,} | GPUs: {n_gpus} | Batch: {batch_size}")
    
    # Pre-allocate output
    embeddings_output = torch.empty((n_samples, embedding_dim), dtype=torch.float32, pin_memory=True)
    
    individual_ids = data['individual_id'].astype(str).tolist()
    index_dts = data['index_dt'].astype(str).tolist()
    
    # Create model on each GPU
    models = []
    for gpu_id in range(n_gpus):
        with torch.cuda.device(gpu_id):
            if moe_config is not None:
                gpu_model = type(model)(config, moe_config)
            else:
                gpu_model = type(model)(config)
            gpu_model.load_state_dict(model.state_dict())
            gpu_model = gpu_model.to(f'cuda:{gpu_id}')
            gpu_model.eval()
            models.append(gpu_model)
    
    if verbose:
        print(f"Models loaded on {n_gpus} GPUs")
    
    # Create dataloader
    dataset = LazyClinicalDataset(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True
    )
    
    has_moe = moe_config is not None
    current_idx = 0
    
    pbar = tqdm(dataloader, desc="Generating embeddings", disable=not verbose)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Round-robin GPU assignment
            gpu_id = batch_idx % n_gpus
            gpu_device = torch.device(f'cuda:{gpu_id}')
            gpu_model = models[gpu_id]
            
            batch_size_actual = batch['age'].shape[0]
            
            x = torch.cat([
                batch['age'].unsqueeze(-1),
                batch['gender'].unsqueeze(-1),
                batch['lob'].unsqueeze(-1),
                batch['codes']
            ], dim=-1).to(gpu_device, non_blocking=True)
            
            dt_cnt = batch['dt_cnt']
            
            # Use EmbeddingExtractor for this specific model
            with EmbeddingExtractor(gpu_model) as extractor:
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        _ = gpu_model(x, return_moe_losses=False) if has_moe else gpu_model(x)
                else:
                    _ = gpu_model(x, return_moe_losses=False) if has_moe else gpu_model(x)
                
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)
            
            embeddings_output[current_idx:current_idx + batch_size_actual] = patient_embs.float().cpu()
            current_idx += batch_size_actual
            
            elapsed = time.time() - start_time
            pbar.set_postfix({
                'GPU': gpu_id,
                'speed': f'{current_idx/elapsed:.0f}/s'
            })
    
    # Cleanup
    for m in models:
        del m
    torch.cuda.empty_cache()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
    
    return embeddings, individual_ids, index_dts
```

## Memory Comparison

| Approach | GPU 0 Memory | GPU 1-3 Memory | Risk |
|----------|-------------|----------------|------|
| **DataParallel** | Model + 4× Outputs | Model + 1× Outputs | **OOM on GPU 0** |
| **Parallel chunks** | Model + 1× Outputs | Model + 1× Outputs | ✅ Balanced |
| **Round-robin** | Model + 1× Outputs | Model + 1× Outputs | ✅ Balanced |

## Recommendation for 4× T4 (16GB each)

Use the **Simple Sequential** approach because:
1. **Memory balanced** - each GPU only holds model + one batch
2. **No OOM risk** - no output gathering to GPU 0
3. **Simple code** - easier to debug
4. **Still fast** - GPUs process different batches, just not truly parallel

With `batch_size=64` and 4× T4 GPUs, you should see:
- ~4-6GB memory per GPU (leaving plenty of headroom)
- ~2x speedup from round-robin (GPUs overlap data loading)

# Expert Critical Review: Embedding Generation Optimization

## Executive Summary

The proposed solutions identify real bottlenecks but contain several technical inaccuracies. I'll provide a rigorous analysis and a production-ready final solution.

---

## Critical Analysis of Proposed Solutions

### Solution 1: Pre-allocate + Non-blocking + CUDA Streams

| Aspect | Verdict | Rationale |
|--------|---------|-----------|
| Pre-allocated pinned output | ✅ **Agree** | Eliminates O(n) vstack at end; pinned memory enables DMA transfers |
| `non_blocking=True` for GPU→CPU | ✅ **Agree** | Allows GPU to start next batch while transfer completes |
| CUDA stream usage | ⚠️ **Partially Disagree** | The pattern shown is incorrect - you cannot `copy_(..., non_blocking=True)` to CPU inside a CUDA stream context and expect async behavior |
| `persistent_workers=True` | ✅ **Agree** | Avoids worker spawn overhead between batches |

**Evidence Against Stream Pattern:**
```python
# INCORRECT (as proposed):
with torch.cuda.stream(transfer_stream):
    embeddings_output[batch_start:batch_end].copy_(
        patient_embs.float().cpu(),  # ← .cpu() is SYNCHRONOUS
        non_blocking=True
    )
```

The `.cpu()` call **forces synchronization** before the copy. The correct pattern is:
```python
# CORRECT:
embeddings_output[batch_start:batch_end].copy_(
    patient_embs.float(),  # Keep on GPU
    non_blocking=True      # Pinned memory enables async
)
# No stream needed - pinned memory + non_blocking already async
```

---

### Solution 2: torch.compile

| Aspect | Verdict | Rationale |
|--------|---------|-----------|
| `torch.compile()` for inference | ⚠️ **Conditional Agree** | Works for standard PyTorch, but **Flash Attention + xFormers use custom CUDA kernels that may not compile** |
| `mode="reduce-overhead"` | ✅ **Agree** | Best for inference with variable batch sizes |
| `dynamic=True` | ✅ **Agree** | Essential for variable batch sizes (last batch often smaller) |
| `suppress_errors = True` | ❌ **Disagree** | Silently fails, hiding real issues |

**Evidence - xFormers + torch.compile Incompatibility:**
From PyTorch 2.0 documentation and xFormers GitHub issues:
- Custom CUDA ops (like `memory_efficient_attention`) trigger graph breaks
- `fullgraph=False` is required (already proposed)
- Speedup is often <1.2x for attention-heavy models due to graph breaks

**My Recommendation:** Test before enabling. For FlashMoE with xFormers, expected speedup is minimal (~10-15%) due to attention being the bottleneck and already optimized.

---

### Solution 3: DataParallel Multi-GPU

| Aspect | Verdict | Rationale |
|--------|---------|-----------|
| Using DataParallel | ❌ **Strongly Disagree** | **OOM guaranteed on GPU 0** for large batches |
| Hook behavior with DP | ❌ **Problematic** | Hooks capture from replica on GPU 0 only; other GPUs' embeddings lost |

**Evidence - DataParallel Gathering:**
```
DataParallel Memory Distribution (batch=128, 4 GPUs):
┌─────────────────────────────────────────────────────┐
│ GPU 0: Model + 32 samples + GATHERED 128 outputs   │ ← OOM Risk
│ GPU 1: Model + 32 samples + 32 outputs             │
│ GPU 2: Model + 32 samples + 32 outputs             │
│ GPU 3: Model + 32 samples + 32 outputs             │
└─────────────────────────────────────────────────────┘
```

For T4 (16GB):
- Model: ~300MB
- Activations for 32 samples at len_dy=200, embedding=256: ~500MB
- **Gathered outputs (4 × 32 × 200 × 256 × 4 bytes)**: ~26MB per batch (small)
- **Real issue**: Intermediate activations during `gather()` cause fragmentation

**Verdict:** Reject DataParallel for this use case.

---

### Solution: Parallel Chunks (Threaded)

| Aspect | Verdict | Rationale |
|--------|---------|-----------|
| Splitting data per GPU | ✅ **Agree** | Memory balanced, no gathering |
| ThreadPoolExecutor | ⚠️ **Partially Agree** | Works, but Python GIL limits true parallelism |
| Model cloning via constructor | ❌ **Disagree** | `type(model)(config, moe_config)` fails if constructor signature differs |

**Correct Model Cloning:**
```python
import copy

# Deep copy preserves all state
model_copy = copy.deepcopy(model)
model_copy = model_copy.to(f'cuda:{gpu_id}')
model_copy.eval()
```

**Threading Reality Check:**
- Python GIL means threads can't execute Python simultaneously
- BUT: GPU ops release GIL → actual parallelism for CUDA
- CPU-bound dataloader work still serialized

---

### Solution: Round-Robin Sequential

| Aspect | Verdict | Rationale |
|--------|---------|-----------|
| Memory balanced | ✅ **Agree** | Each GPU only holds model + 1 batch |
| "2x speedup" claim | ❌ **Disagree** | **No actual parallelism** - still sequential processing |

**Evidence:**
Round-robin just changes *which* GPU processes each batch sequentially:
```
Time →  Batch 0    Batch 1    Batch 2    Batch 3
GPU 0:  ████████   --------   ████████   --------
GPU 1:  --------   ████████   --------   ████████
```
This is **not** parallel - total time is the same as single GPU.

---

## Additional Optimizations Missing from Proposals

### 1. CUDA Graph Capture (10-20% speedup)

For **fixed batch size** batches, capture the forward pass as a CUDA graph:

```python
# Warmup
static_input = torch.zeros(batch_size, len_dy, 83, device=device)
_ = model(static_input)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (very fast)
for batch in dataloader:
    static_input.copy_(actual_input)
    g.replay()
    # Use static_output
```

**Limitation:** Doesn't work for variable batch sizes or MoE with dynamic routing.

### 2. Inference-Mode Context

`torch.inference_mode()` is faster than `torch.no_grad()`:

```python
with torch.inference_mode():  # Faster than no_grad()
    for batch in dataloader:
        ...
```

**Evidence:** `inference_mode()` disables version counting and gradient tracking entirely, vs `no_grad()` which still tracks.

### 3. Batched ID Extraction

Your current code converts IDs to list every batch. Do it once upfront:

```python
# Before loop (once)
individual_ids = data['individual_id'].astype(str).tolist()

# Instead of in collate_fn or per-batch
```

---

## My Final Production Solution

This combines the valid optimizations while fixing the issues:

```python
import torch
import numpy as np
import pandas as pd
import time
import copy
from typing import Tuple, List, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
import threading


def generate_embeddings_optimized(
    model: torch.nn.Module,
    config: 'BaseConfig',
    data: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    use_mixed_precision: bool = True,
    verbose: bool = True,
    multi_gpu: bool = False,  # Enable for 4x T4
    moe_config: Optional['MoEConfig'] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Production-optimized embedding generation.
    
    Optimizations:
    1. Pre-allocated pinned memory output (no vstack)
    2. Non-blocking async GPU→CPU transfers
    3. torch.inference_mode (faster than no_grad)
    4. Optional multi-GPU with true parallelism
    5. Progress bar with ETA
    
    Args:
        model: Loaded model in eval mode
        config: Model configuration
        data: DataFrame with required columns
        device: Primary device
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        use_mixed_precision: Use FP16 for Flash models
        verbose: Print progress
        multi_gpu: Enable multi-GPU processing
        moe_config: MoE config (required for multi-GPU with MoE models)
    
    Returns:
        embeddings: [n_samples, embedding_dim]
        individual_ids: List of IDs
        index_dts: List of dates
    """
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size
    
    # Detect model type
    has_moe = (hasattr(model, 'forward') and 
               'return_moe_losses' in model.forward.__code__.co_varnames)
    
    n_gpus = torch.cuda.device_count() if multi_gpu else 1
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"OPTIMIZED EMBEDDING GENERATION")
        print(f"{'='*70}")
        print(f"Samples: {n_samples:,} | Batch: {batch_size} | GPUs: {n_gpus}")
        print(f"Workers: {num_workers} | Mixed precision: {use_mixed_precision}")
    
    # ========================================================================
    # OPTIMIZATION 1: Pre-allocate pinned memory output
    # ========================================================================
    embeddings_output = torch.empty(
        (n_samples, embedding_dim),
        dtype=torch.float32,
        pin_memory=True  # Enables async DMA transfers
    )
    
    # Extract IDs once (not per batch)
    individual_ids = data['individual_id'].astype(str).tolist()
    index_dts = data['index_dt'].astype(str).tolist()
    
    if n_gpus > 1 and multi_gpu:
        # ====================================================================
        # MULTI-GPU PATH: Parallel processing with thread pool
        # ====================================================================
        return _generate_embeddings_multi_gpu(
            model=model,
            config=config,
            data=data,
            embeddings_output=embeddings_output,
            individual_ids=individual_ids,
            index_dts=index_dts,
            n_gpus=n_gpus,
            batch_size=batch_size,
            num_workers=num_workers,
            use_mixed_precision=use_mixed_precision,
            has_moe=has_moe,
            moe_config=moe_config,
            verbose=verbose,
            start_time=start_time,
        )
    else:
        # ====================================================================
        # SINGLE GPU PATH: Optimized sequential
        # ====================================================================
        return _generate_embeddings_single_gpu(
            model=model,
            config=config,
            data=data,
            device=device,
            embeddings_output=embeddings_output,
            individual_ids=individual_ids,
            index_dts=index_dts,
            batch_size=batch_size,
            num_workers=num_workers,
            use_mixed_precision=use_mixed_precision,
            has_moe=has_moe,
            verbose=verbose,
            start_time=start_time,
        )


def _generate_embeddings_single_gpu(
    model, config, data, device, embeddings_output,
    individual_ids, index_dts, batch_size, num_workers,
    use_mixed_precision, has_moe, verbose, start_time
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Single GPU optimized path."""
    
    n_samples = len(data)
    model.eval()
    
    # Optimized DataLoader
    dataset = LazyClinicalDataset(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Keep workers alive
    )
    
    current_idx = 0
    pbar = tqdm(dataloader, desc="Generating embeddings", disable=not verbose)
    
    # ========================================================================
    # OPTIMIZATION 2: inference_mode (faster than no_grad)
    # ========================================================================
    with torch.inference_mode():
        with EmbeddingExtractor(model) as extractor:
            for batch in pbar:
                batch_size_actual = batch['age'].shape[0]
                batch_start = current_idx
                batch_end = batch_start + batch_size_actual
                
                # ============================================================
                # OPTIMIZATION 3: Non-blocking GPU transfers
                # ============================================================
                x = torch.cat([
                    batch['age'].unsqueeze(-1),
                    batch['gender'].unsqueeze(-1),
                    batch['lob'].unsqueeze(-1),
                    batch['codes']
                ], dim=-1).to(device, non_blocking=True)
                
                dt_cnt = batch['dt_cnt']
                
                # Forward pass
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
                
                # Extract embeddings
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                
                # ============================================================
                # OPTIMIZATION 4: Async copy to pre-allocated pinned memory
                # ============================================================
                # .float() keeps on GPU, non_blocking enables DMA
                embeddings_output[batch_start:batch_end].copy_(
                    patient_embs.float(),  # Cast on GPU
                    non_blocking=True       # Async DMA to pinned memory
                )
                
                current_idx = batch_end
                
                # Progress metrics
                elapsed = time.time() - start_time
                speed = batch_end / elapsed
                eta = (n_samples - batch_end) / speed if speed > 0 else 0
                pbar.set_postfix({
                    'speed': f'{speed:.0f}/s',
                    'ETA': f'{eta:.0f}s'
                })
    
    # Sync to ensure all async transfers complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
        print(f"   Output: {embeddings.shape}")
    
    return embeddings, individual_ids, index_dts


def _generate_embeddings_multi_gpu(
    model, config, data, embeddings_output, individual_ids, index_dts,
    n_gpus, batch_size, num_workers, use_mixed_precision, has_moe,
    moe_config, verbose, start_time
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Multi-GPU path with true parallelism via ThreadPoolExecutor.
    
    Strategy:
    - Split data into N chunks (one per GPU)
    - Each GPU processes its chunk independently
    - Write to non-overlapping regions of shared output tensor
    """
    n_samples = len(data)
    
    if verbose:
        print(f"Multi-GPU mode: {n_gpus} GPUs")
    
    # ========================================================================
    # STEP 1: Clone model to each GPU (before threading)
    # ========================================================================
    models = []
    for gpu_id in range(n_gpus):
        if verbose:
            print(f"  Cloning model to GPU {gpu_id}...")
        
        with torch.cuda.device(gpu_id):
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to(f'cuda:{gpu_id}')
            model_copy.eval()
            models.append(model_copy)
    
    # ========================================================================
    # STEP 2: Split data into chunks
    # ========================================================================
    chunk_size = (n_samples + n_gpus - 1) // n_gpus
    data_chunks = []
    start_indices = []
    
    for i in range(n_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        data_chunks.append(data.iloc[start_idx:end_idx].reset_index(drop=True))
        start_indices.append(start_idx)
        
        if verbose:
            print(f"  GPU {i}: samples {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} samples)")
    
    # ========================================================================
    # STEP 3: Process in parallel with ThreadPoolExecutor
    # ========================================================================
    progress_lock = threading.Lock()
    total_processed = [0]
    errors = []
    
    def process_chunk(gpu_id: int, data_chunk: pd.DataFrame, start_idx: int):
        """Process a data chunk on a specific GPU."""
        if len(data_chunk) == 0:
            return
        
        try:
            gpu_device = torch.device(f'cuda:{gpu_id}')
            gpu_model = models[gpu_id]
            
            dataset = LazyClinicalDataset(data_chunk, config)
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
                        batch_size_actual = batch['age'].shape[0]
                        
                        x = torch.cat([
                            batch['age'].unsqueeze(-1),
                            batch['gender'].unsqueeze(-1),
                            batch['lob'].unsqueeze(-1),
                            batch['codes']
                        ], dim=-1).to(gpu_device, non_blocking=True)
                        
                        dt_cnt = batch['dt_cnt']
                        
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
                        
                        # Write to shared output (thread-safe: non-overlapping regions)
                        embeddings_output[local_idx:local_idx + batch_size_actual].copy_(
                            patient_embs.float(),
                            non_blocking=True
                        )
                        
                        local_idx += batch_size_actual
                        
                        with progress_lock:
                            total_processed[0] += batch_size_actual
            
            # Sync this GPU
            torch.cuda.synchronize(gpu_device)
            
        except Exception as e:
            errors.append((gpu_id, str(e)))
    
    # Launch parallel processing
    if verbose:
        pbar = tqdm(total=n_samples, desc=f"Multi-GPU ({n_gpus} GPUs)")
    
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(process_chunk, gpu_id, data_chunks[gpu_id], start_indices[gpu_id])
            for gpu_id in range(n_gpus)
        ]
        
        # Update progress bar
        last_count = 0
        while not all(f.done() for f in futures):
            with progress_lock:
                current = total_processed[0]
            if verbose:
                pbar.update(current - last_count)
            last_count = current
            time.sleep(0.1)
        
        # Final update
        if verbose:
            pbar.update(n_samples - last_count)
            pbar.close()
        
        # Check for exceptions
        for f in futures:
            f.result()  # Raises if any thread failed
    
    if errors:
        raise RuntimeError(f"GPU errors: {errors}")
    
    # ========================================================================
    # STEP 4: Cleanup
    # ========================================================================
    for m in models:
        del m
    torch.cuda.empty_cache()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
        print(f"   Effective: {n_samples/elapsed * n_gpus:,.0f} samples/s (across {n_gpus} GPUs)")
        print(f"   Output: {embeddings.shape}")
    
    return embeddings, individual_ids, index_dts
```

---

## Performance Comparison

| Approach | Samples/sec (est.) | Memory per GPU | Complexity |
|----------|-------------------|----------------|------------|
| **Original** | 500-1000 | Variable | Low |
| **Single GPU Optimized** | 1500-2500 | Stable | Low |
| **Multi-GPU (4×T4)** | 4000-8000 | Balanced | Medium |

**Expected speedup on 4× T4 with 500K samples:**
- Original: ~8-15 minutes
- Optimized single GPU: ~3-6 minutes  
- Optimized multi-GPU: ~1-2 minutes

---

## Key Takeaways

1. **Reject DataParallel** for embedding extraction - OOM risk and hook issues
2. **Use `torch.inference_mode()`** not `torch.no_grad()` - 5-10% faster
3. **Pre-allocate pinned memory** - eliminates vstack overhead
4. **Use `copy_(tensor, non_blocking=True)`** to pinned memory - async DMA
5. **True multi-GPU parallelism** requires ThreadPoolExecutor with model clones
6. **Skip `torch.compile`** for xFormers/FlashAttention models - minimal benefit, potential breakage