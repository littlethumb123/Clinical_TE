# Gpu memory optimization for transformer pretraining
_Exported on 12/25/2025 at 10:15:52 PST from Cursor (2.2.43)_

---

**User**

You are an expert in GPU memory management during transfomrer pretraining; you are familiar iwth all industry level practice strategies to keep the training from OOM errors; 
Inspect and review the implemneation closely, identify opporutnity to optimize the GPU memory amangemtn during the training. I always comes across OOM errors, especially when the batch_size go up to 64. @dev/moe/moe_flashattn_3.py  DO NOT HALLUCINATE. Your impelmenation should be very clear and detailed and indicate where to add/change what data. provide complete implemantion/modificaiton as much as possible so that I can easily modify teh code myself manually

---

**Cursor**

I'll analyze the implementation to identify GPU memory optimization opportunities. Let me first examine the key areas: training loop, model architecture, and data handling.

Let me look at more areas including gradient checkpointing, data loading, and the model's memory-intensive components:

Let me search for gradient checkpointing usage and examine the MoE layer implementation:

Let me check where the checkpoint import is used and look at some additional memory-related patterns:

Now I have a complete picture of the implementation. I've identified **11 major GPU memory optimization opportunities**. Here's my comprehensive analysis and detailed implementation recommendations:

---

## GPU Memory Optimization Analysis for `moe_flashattn_3.py`

### Summary of Issues Found

| Issue | Severity | Memory Saved (Est.) |
|-------|----------|---------------------|
| 1. Gradient Checkpointing not used | HIGH | 40-60% activation memory |
| 2. Gradient Accumulation not implemented | HIGH | Up to 4x batch size support |
| 3. Multi-hot targets as float32 | MEDIUM | ~25% for targets |
| 4. Memory leak in cleanup | LOW | Variable |
| 5. Missing in-place operations | MEDIUM | 10-20% |
| 6. Optimizer state optimization | MEDIUM | 15-30% |
| 7. Activation offloading missing | MEDIUM | Variable |
| 8. Code embedding intermediate tensor | HIGH | ~3.3GB |
| 9. Target tensor not pinned properly | LOW | Latency improvement |
| 10. No automatic batch size finding | MEDIUM | Prevents OOM |
| 11. Missing model sharding for MoE | MEDIUM | Expert memory |

---

## Detailed Implementation Changes

### **1. Implement Gradient Checkpointing (HIGH PRIORITY)**

The import exists at line 394 but isn't used. Gradient checkpointing trades compute for memory by not storing intermediate activations.

**Add to `FlashAttentionConfig` (around line 578):**

```python
@dataclass
class FlashAttentionConfig(BaseConfig):
    # ... existing fields ...
    use_flash: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    use_prenorm: bool = True
    dtype: torch.dtype = torch.float16
    use_learnt_att_pool: bool = False
    nhead: int = 8
    
    # NEW: Gradient checkpointing settings
    use_gradient_checkpointing: bool = True  # Enable by default for batch_size >= 32
    checkpoint_every_n_layers: int = 2  # Checkpoint every 2 layers (balance speed/memory)
```

**Modify `FlashMoETransformer.forward()` (around line 2746) - add checkpointing wrapper:**

```python
from torch.utils.checkpoint import checkpoint

class FlashMoETransformer(nn.Module):
    def __init__(self, config: FlashAttentionConfig, moe_config: Optional[MoEConfig] = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        self.use_moe_from_layer = moe_config.use_moe_from_layer if moe_config else 999
        
        # NEW: Store checkpointing config
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        self.checkpoint_every_n_layers = getattr(config, 'checkpoint_every_n_layers', 2)
        
        # ... rest of __init__ unchanged ...

    def _checkpointed_layer_forward(
        self, 
        layer_dict: nn.ModuleDict, 
        cd_input: torch.Tensor, 
        layer_idx: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Wrapper for gradient checkpointing.
        Must be a separate function that doesn't capture state (for checkpoint compatibility).
        """
        def layer_fn(x):
            residual = x
            x_norm = layer_dict['norm1'](x)
            x_attn = layer_dict['attention'](x_norm, is_causal=True)
            x = residual + x_attn
            
            residual = x
            x_norm = layer_dict['norm2'](x)
            
            if isinstance(layer_dict['ffn'], MoELayer):
                x_ffn, moe_losses = layer_dict['ffn'](x_norm, train=self.training)
                # Note: moe_losses returned separately since checkpoint doesn't handle dicts
                return residual + x_ffn
            else:
                x_ffn = layer_dict['ffn'](x_norm)
                return residual + x_ffn
        
        return layer_fn

    def forward(self, 
                x: torch.Tensor, 
                return_moe_losses: bool = True
               ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with optional gradient checkpointing."""
        gpu_batchsize = x.shape[0]
        actual_len_dy = x.shape[1]  
        actual_len_cd = x.shape[2] - 3
        device = x.device
        
        # ... existing embedding code through line 2802 ...
        
        # Temporal encoding with Flash + MoE + GRADIENT CHECKPOINTING
        total_aux_loss = torch.tensor(0.0, device=device)
        expert_usage_list = []
        
        for i, layer in enumerate(self.temporal_layers):
            # Determine if this layer should be checkpointed
            should_checkpoint = (
                self.training and 
                self.use_gradient_checkpointing and
                (i % self.checkpoint_every_n_layers == 0)
            )
            
            if should_checkpoint and not isinstance(layer['ffn'], MoELayer):
                # Use gradient checkpointing for non-MoE layers
                # (MoE layers have auxiliary losses that complicate checkpointing)
                def create_custom_forward(layer_module):
                    def custom_forward(x):
                        residual = x
                        x_norm = layer_module['norm1'](x)
                        x_attn = layer_module['attention'](x_norm, is_causal=True)
                        x = residual + x_attn
                        
                        residual = x
                        x_norm = layer_module['norm2'](x)
                        x_ffn = layer_module['ffn'](x_norm)
                        return residual + x_ffn
                    return custom_forward
                
                cd = checkpoint(
                    create_custom_forward(layer),
                    cd,
                    use_reentrant=False  # Recommended for newer PyTorch
                )
            else:
                # Standard forward (MoE layers or non-checkpointed)
                residual = cd
                cd_norm = layer['norm1'](cd)
                cd_attn = layer['attention'](cd_norm, is_causal=True)
                cd = residual + cd_attn
                
                residual = cd
                cd_norm = layer['norm2'](cd)
                
                if isinstance(layer['ffn'], MoELayer):
                    cd_ffn, moe_losses = layer['ffn'](cd_norm, train=self.training)
                    if self.training and return_moe_losses:
                        total_aux_loss = total_aux_loss + moe_losses['aux_loss']
                        if 'expert_usage' in moe_losses:
                            expert_usage_list.append(moe_losses['expert_usage'])
                else:
                    cd_ffn = layer['ffn'](cd_norm)
                
                cd = residual + cd_ffn
        
        # ... rest unchanged ...
```

---

### **2. Implement Gradient Accumulation (HIGH PRIORITY)**

The parameter exists but logic is missing. This allows simulating larger batches with less memory.

**Modify `train_epoch()` function (around line 4200):**

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
    is_main: bool = True,
    use_ddp: bool = False,
    accumulation_steps: int = 4  # Actually use this now!
) -> Dict[str, float]:
    """
    Train for one epoch with gradient accumulation support.
    
    For batch_size=64 with OOM, use:
    - actual batch_size=16 
    - accumulation_steps=4
    - Effective batch_size = 16 * 4 = 64
    """
    model.train()
    
    nbatch = len(dataloader)
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    batch_metrics_buffer = []  
    moe_metrics_buffer = []
    
    if loss_tracker is None:
        loss_tracker = LossTracker()
    
    # Track accumulated loss for proper averaging
    accumulated_loss = 0.0
    accumulation_counter = 0
    
    for batch_idx, batch in enumerate(dataloader):
        
        if is_main and batch_idx % log_interval == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}')
        
        # GPU utilization check (unchanged)
        if batch_idx == 0 and is_main:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"\n🔍 GPU UTILIZATION CHECK (Batch 0):")
                for gpu_id in range(num_gpus):
                    mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"   GPU {gpu_id}: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        
        # Only zero grad at accumulation boundaries
        if accumulation_counter == 0:
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves memory!
        
        # Get batch data (unchanged)
        age = batch['age']
        gender = batch['gender']
        lob = batch['lob']
        codes = batch['codes']
        dt_cnt = batch['dt_cnt']
        targets_mh = batch['target_multihot']
        y = batch['target']
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        x = x.cuda(non_blocking=True)
        dt_cnt = dt_cnt.cuda(non_blocking=True)
        targets_mh = targets_mh.cuda(non_blocking=True)
        
        # Forward pass
        need_predictions = is_main and (batch_idx % log_interval == 0)
        
        if use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
                if isinstance(result, tuple):
                    total_loss, extras = result
                    output = extras.get('predictions', None) if need_predictions else None
                    moe_losses = extras.get('moe_losses', {})
                    pred_loss = extras.get('pred_loss', total_loss)
                    aux_loss = extras.get('aux_loss', torch.tensor(0.0))
                else:
                    total_loss = result
                    pred_loss = total_loss
                    aux_loss = torch.tensor(0.0, device=device)
                    output = None
                    moe_losses = {}
        else:
            result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
            if isinstance(result, tuple):
                total_loss, extras = result
                output = extras.get('predictions', None) if need_predictions else None
                moe_losses = extras.get('moe_losses', {})
                pred_loss = extras.get('pred_loss', total_loss)
                aux_loss = extras.get('aux_loss', torch.tensor(0.0))
            else:
                total_loss = result
                pred_loss = total_loss
                aux_loss = torch.tensor(0.0, device=device)
                output = None
                moe_losses = {}

        # Handle multi-element tensors from DataParallel
        if total_loss.numel() > 1:
            total_loss = total_loss.mean()
        if pred_loss.numel() > 1:
            pred_loss = pred_loss.mean()
        if aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        scaled_loss = total_loss / accumulation_steps
        
        # Backward pass
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        # Optimizer step at accumulation boundaries
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
        
        # Track losses
        pred_loss_scalar = pred_loss.item()
        aux_loss_scalar = aux_loss.item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
        
        # Logging (unchanged except for metrics)
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                batch_metrics = compute_batch_metrics_lightweight(
                    output, y, 
                    dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt, 
                    config, device
                )
                batch_metrics_buffer.append(batch_metrics)
                
                print(f"    Loss: {pred_loss_scalar:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f}")
                
                if moe_losses and 'expert_usage' in moe_losses:
                    moe_batch_metrics = compute_moe_batch_metrics(moe_losses)
                    moe_metrics_buffer.append(moe_batch_metrics)
        
        # FIXED: Proper cleanup (was buggy: del x, creates tuple)
        del x, targets_mh, dt_cnt
        if output is not None:
            del output
        del pred_loss, total_loss, scaled_loss
        
        # Periodic cleanup
        if batch_idx % 100 == 0:
            gc.collect()
            
            if is_main and device.type == 'cuda' and batch_idx % 500 == 0:
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
                    print(f'    GPU {gpu_id}: {allocated:.2f}GB / {peak:.2f}GB peak')
    
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        global_step += 1
    
    # End-of-epoch cleanup
    if device.type == 'cuda':
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    # ... rest of function unchanged ...
```

---

### **3. Optimize Multi-hot Target Tensor Creation (MEDIUM PRIORITY)**

Change from float32 to float16 and use sparse representation for large vocabularies.

**Modify `clinical_collate_fn()` (around line 3096):**

```python
def clinical_collate_fn(batch: List[Dict], config: 'BaseConfig') -> Dict[str, Any]:
    """
    Custom collate function with memory-optimized target tensors.
    """
    batch_size = len(batch)
    len_dy = config.len_dy
    target_cd_cnt = config.target_cd_cnt
    
    # Extract each field
    ages = torch.stack([item['age'] for item in batch])
    genders = torch.stack([item['gender'] for item in batch])
    lobs = torch.stack([item['lob'] for item in batch])
    codes = torch.stack([item['codes'] for item in batch])
    dt_cnts = torch.tensor([item['dt_cnt'] for item in batch], dtype=torch.long)
    
    # OPTIMIZED: Use float16 for multi-hot targets (saves 50% memory)
    # For batch_size=64, len_dy=200, target_cd_cnt=6297:
    # float32: 64 * 200 * 6297 * 4 = 322 MB
    # float16: 64 * 200 * 6297 * 2 = 161 MB
    targets_multihot = torch.zeros(
        batch_size, len_dy, target_cd_cnt, 
        dtype=torch.float16  # Changed from float32
    )
    
    for i, item in enumerate(batch):
        target_list = item['target']
        for day_idx, day_codes in enumerate(target_list):
            if day_idx < len_dy and day_codes:
                for code_idx in day_codes:
                    if 0 <= code_idx < target_cd_cnt:
                        targets_multihot[i, day_idx, code_idx] = 1.0
    
    targets_list = [item['target'] for item in batch]
    
    return {
        'age': ages,
        'gender': genders,
        'lob': lobs,
        'codes': codes,
        'dt_cnt': dt_cnts,
        'target_multihot': targets_multihot,
        'target': targets_list
    }
```

**Also update `DataParallelWrapper.forward()` to handle float16 targets (around line 1107):**

```python
def forward(
    self, 
    x: torch.Tensor,
    dt_cnt: torch.Tensor,
    targets: torch.Tensor,
    return_predictions: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    # ... existing code ...
    
    # Ensure targets match output dtype for loss computation
    # BCEWithLogitsLoss can work with float16 in autocast context
    if targets.dtype != output_flat.dtype:
        targets_flat = targets.view(batch_size * actual_len_dy, self.target_cd_cnt).to(output_flat.dtype)
    else:
        targets_flat = targets.view(batch_size * actual_len_dy, self.target_cd_cnt)
    
    # ... rest unchanged ...
```

---

### **4. Add Automatic Mixed Precision for BCEWithLogitsLoss**

Ensure loss computation uses proper AMP context:

**Modify `DataParallelWrapper.forward()` (around line 1125):**

```python
# Compute loss only on valid positions
if valid_mask.any():
    valid_output = output_flat[valid_mask]
    valid_targets = targets_flat[valid_mask]
    
    # Ensure consistent dtype for loss (important for mixed precision)
    if valid_targets.dtype != valid_output.dtype:
        valid_targets = valid_targets.to(valid_output.dtype)
    
    pred_loss = self.criterion(valid_output, valid_targets)
else:
    pred_loss = torch.tensor(0.0, device=device, requires_grad=True)
```

---

### **5. Optimize Code Embedding Memory (HIGH PRIORITY)**

The code embedding creates a huge intermediate tensor. Optimize with chunked processing:

**Add helper method to model classes (add after line 2720 in `FlashMoETransformer`):**

```python
def _embed_codes_memory_efficient(
    self, 
    x: torch.Tensor, 
    gpu_batchsize: int, 
    actual_len_dy: int, 
    actual_len_cd: int
) -> torch.Tensor:
    """
    Memory-efficient code embedding that processes in chunks.
    
    Original approach creates: [batch, len_dy, len_cd, embedding_size]
    For batch=64, len_dy=200, len_cd=80, emb=256 = 3.3GB
    
    Chunked approach processes subsets and aggregates, reducing peak memory.
    """
    device = x.device
    code_indices = x[:, :, 3:].long()  # [batch, len_dy, len_cd]
    
    # For smaller batches, use standard approach
    if gpu_batchsize * actual_len_dy * actual_len_cd <= 100000:
        cd = self.embedding_cd(code_indices)  # [batch, len_dy, len_cd, emb]
        return cd.sum(dim=-2)  # [batch, len_dy, emb]
    
    # For larger batches, process day-by-day
    cd_res = torch.zeros(
        gpu_batchsize, actual_len_dy, self.config.embedding_size,
        dtype=x.dtype, device=device
    )
    
    # Process in chunks of days to limit memory
    chunk_size = max(1, 50000 // (gpu_batchsize * actual_len_cd))
    
    for start_day in range(0, actual_len_dy, chunk_size):
        end_day = min(start_day + chunk_size, actual_len_dy)
        
        # Get embeddings for this chunk
        chunk_codes = code_indices[:, start_day:end_day, :]  # [batch, chunk_days, len_cd]
        chunk_emb = self.embedding_cd(chunk_codes)  # [batch, chunk_days, len_cd, emb]
        
        # Sum over codes within each day
        cd_res[:, start_day:end_day, :] = chunk_emb.sum(dim=-2)
        
        # Immediately free chunk memory
        del chunk_emb
    
    return cd_res
```

**Update `forward()` method to use this (around line 2766):**

```python
def forward(self, x: torch.Tensor, return_moe_losses: bool = True) -> Tuple[torch.Tensor, Dict]:
    gpu_batchsize = x.shape[0]
    actual_len_dy = x.shape[1]  
    actual_len_cd = x.shape[2] - 3
    device = x.device
    
    # Extract demographics embeddings
    age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
    gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
    lob_emb = self.embedding_lob(x[:, :, 2].long())
    
    # OPTIMIZED: Memory-efficient code embedding
    cd_res = self._embed_codes_memory_efficient(
        x, gpu_batchsize, actual_len_dy, actual_len_cd
    )
    
    # Still need per-code embeddings for daily encoder
    cd = self.embedding_cd(x[:, :, 3:].long())
    
    # ... rest of forward unchanged ...
```

---

### **6. Add 8-bit Optimizer for Memory Savings**

Replace AdamW with 8-bit version for 50-75% optimizer state memory savings:

**Add at imports (around line 388):**

```python
# Try to import bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("bitsandbytes not available. Install with: pip install bitsandbytes")
```

**Modify optimizer creation in `run_single_experiment()` (around line 9184):**

```python
# 4. OPTIMIZER SETUP with memory optimization
if HAS_BITSANDBYTES and torch.cuda.is_available():
    # Use 8-bit Adam for ~75% memory savings on optimizer states
    # For a 100M parameter model: 
    #   - Standard AdamW: ~1.2GB optimizer states
    #   - 8-bit Adam: ~300MB optimizer states
    logger.info("Using 8-bit AdamW optimizer (bitsandbytes)")
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
else:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
```

---

### **7. Add Memory-Efficient DataLoader Configuration**

**Modify `_create_dataloaders()` (around line 8854):**

```python
def _create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: BaseConfig,
    use_bucketing: bool,
    world_size: int = 1,
    logger: Optional[logging.Logger] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create memory-optimized dataloaders."""
    train_dataset = ClinicalDataset(train_data, config)
    val_dataset = ClinicalDataset(val_data, config)
    
    # Optimize worker count based on batch size and memory
    # Fewer workers = less memory for prefetched batches
    if config.batch_size >= 64:
        n_workers = min(2, os.cpu_count() // 2)  # Limit workers for large batches
    else:
        n_workers = max(1, os.cpu_count() // max(world_size, 1) // 2)
    
    collate_fn = create_collate_fn(config)
    
    # Common DataLoader kwargs with memory optimizations
    common_kwargs = {
        'num_workers': n_workers,
        'pin_memory': True,
        'collate_fn': collate_fn,
        'prefetch_factor': 2 if n_workers > 0 else None,  # Limit prefetch queue
        'persistent_workers': n_workers > 0
    }
    
    if use_bucketing:
        # ... bucketing code unchanged ...
        pass
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **{k: v for k, v in common_kwargs.items() if k != 'persistent_workers'}
    )
    
    if logger:
        logger.info(f"DataLoader: {n_workers} workers, prefetch_factor=2")
    
    return train_loader, val_loader
```

---

### **8. Add Memory Profiling Utility**

**Add this utility function after line 9717:**

```python
def find_optimal_batch_size(
    model: nn.Module,
    config: BaseConfig,
    device: torch.device,
    min_batch: int = 8,
    max_batch: int = 128,
    use_mixed_precision: bool = True
) -> int:
    """
    Binary search to find maximum batch size that doesn't OOM.
    
    Usage:
        optimal_batch = find_optimal_batch_size(model, config, device)
        config.batch_size = optimal_batch
    """
    model.eval()
    
    def can_run_batch(batch_size: int) -> bool:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            # Create dummy inputs
            x = torch.randint(
                0, config.cd_cnt,
                (batch_size, config.len_dy, config.len_cd + 3),
                device=device
            )
            dt_cnt = torch.full((batch_size,), config.len_dy, device=device)
            targets = torch.zeros(
                batch_size, config.len_dy, config.target_cd_cnt,
                dtype=torch.float16 if use_mixed_precision else torch.float32,
                device=device
            )
            
            # Run forward + backward
            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if hasattr(model, 'module'):
                        out = model.module.model(x)
                    else:
                        out = model(x)
            else:
                out = model(x)
            
            if isinstance(out, tuple):
                out = out[0]
            
            loss = out.sum()
            loss.backward()
            
            del x, dt_cnt, targets, out, loss
            torch.cuda.empty_cache()
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            raise e
    
    # Binary search
    low, high = min_batch, max_batch
    best = min_batch
    
    print(f"Finding optimal batch size between {min_batch} and {max_batch}...")
    
    while low <= high:
        mid = (low + high) // 2
        print(f"  Trying batch_size={mid}...", end=" ")
        
        if can_run_batch(mid):
            print("✓")
            best = mid
            low = mid + 1
        else:
            print("✗ OOM")
            high = mid - 1
    
    # Apply safety margin (90% of max)
    safe_batch = int(best * 0.9)
    print(f"Optimal batch size: {best}, using safe value: {safe_batch}")
    
    return safe_batch
```

---

### **9. Enable PyTorch Memory Optimizations**

**Add at the start of training (in `run_single_experiment()` around line 9070):**

```python
def run_single_experiment(...):
    # ============================================================
    # 0. MEMORY OPTIMIZATION FLAGS
    # ============================================================
    if torch.cuda.is_available():
        # Enable TF32 for faster computation on Ampere+ GPUs (A100, H100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmark for optimized convolutions
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocation strategy
        # 'expandable_segments' reduces fragmentation
        if hasattr(torch.cuda.memory, '_set_allocator_settings'):
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        
        # Pre-allocate memory pool
        torch.cuda.empty_cache()
        
        # Log GPU memory configuration
        if is_main:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB")
    
    # ... rest of function ...
```

---

### **10. Summary: Complete Memory Optimization Configuration**

Here's how to use all these optimizations together. Add this configuration helper:

```python
@dataclass 
class MemoryOptimizationConfig:
    """Configuration for memory optimizations."""
    # Gradient checkpointing
    use_gradient_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    
    # Gradient accumulation
    accumulation_steps: int = 4  # Effective batch = batch_size * accumulation_steps
    
    # Mixed precision
    use_mixed_precision: bool = True
    targets_dtype: torch.dtype = torch.float16
    
    # 8-bit optimizer
    use_8bit_optimizer: bool = True
    
    # Memory-efficient embedding
    use_chunked_embedding: bool = True
    embedding_chunk_size: int = 50000
    
    # DataLoader
    num_workers: int = 2
    prefetch_factor: int = 2


def get_memory_optimized_config(
    base_batch_size: int = 64,
    available_memory_gb: float = 16.0  # T4 has 16GB
) -> Tuple[int, MemoryOptimizationConfig]:
    """
    Get optimized configuration based on target batch size and available memory.
    
    Returns:
        actual_batch_size: The per-GPU batch size to use
        mem_config: Memory optimization settings
    """
    mem_config = MemoryOptimizationConfig()
    
    # Estimate memory requirements (rough heuristics)
    # Base memory per sample ≈ 50MB for this model
    base_memory_per_sample = 0.05  # GB
    
    # With gradient checkpointing, reduce activation memory by ~60%
    if mem_config.use_gradient_checkpointing:
        base_memory_per_sample *= 0.4
    
    # With mixed precision, reduce by another ~40%
    if mem_config.use_mixed_precision:
        base_memory_per_sample *= 0.6
    
    # Calculate max batch size that fits in memory
    # Reserve 2GB for model weights and overhead
    usable_memory = available_memory_gb - 2.0
    max_batch_size = int(usable_memory / base_memory_per_sample)
    
    if base_batch_size <= max_batch_size:
        actual_batch_size = base_batch_size
        mem_config.accumulation_steps = 1
    else:
        # Need gradient accumulation
        actual_batch_size = max(8, max_batch_size // 2)  # Use half for safety margin
        mem_config.accumulation_steps = (base_batch_size + actual_batch_size - 1) // actual_batch_size
    
    print(f"Memory optimization config:")
    print(f"  Target batch: {base_batch_size}")
    print(f"  Actual batch: {actual_batch_size}")
    print(f"  Accumulation steps: {mem_config.accumulation_steps}")
    print(f"  Effective batch: {actual_batch_size * mem_config.accumulation_steps}")
    print(f"  Gradient checkpointing: {mem_config.use_gradient_checkpointing}")
    print(f"  Mixed precision: {mem_config.use_mixed_precision}")
    print(f"  8-bit optimizer: {mem_config.use_8bit_optimizer}")
    
    return actual_batch_size, mem_config
```

---

## Quick Fixes Summary (Immediate Impact)

For immediate OOM relief with batch_size=64, apply these changes in order of priority:

1. **Enable gradient accumulation** - Use batch_size=16 with accumulation_steps=4
2. **Enable gradient checkpointing** - ~50% activation memory reduction  
3. **Change targets to float16** - ~50% target tensor memory reduction
4. **Fix the `del x,` bug** at line 4387 → `del x`
5. **Install and use bitsandbytes** for 8-bit optimizer

With these changes, you should be able to run batch_size=64 effectively (16 actual × 4 accumulation) without OOM errors.

---

Would you like me to provide any of these sections in more detail, or would you prefer to switch to Agent mode so I can apply these changes directly to your codebase?

```python
@dataclass
class FlashAttentionConfig(BaseConfig):
    # ... existing fields ...
    use_flash: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    use_prenorm: bool = True
    dtype: torch.dtype = torch.float16
    use_learnt_att_pool: bool = False
    nhead: int = 8
    
    # NEW: Gradient checkpointing settings
    use_gradient_checkpointing: bool = True  # Enable by default for batch_size >= 32
    checkpoint_every_n_layers: int = 2  # Checkpoint every 2 layers (balance speed/memory)
```

```python
from torch.utils.checkpoint import checkpoint

class FlashMoETransformer(nn.Module):
    def __init__(self, config: FlashAttentionConfig, moe_config: Optional[MoEConfig] = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        self.use_moe_from_layer = moe_config.use_moe_from_layer if moe_config else 999
        
        # NEW: Store checkpointing config
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        self.checkpoint_every_n_layers = getattr(config, 'checkpoint_every_n_layers', 2)
        
        # ... rest of __init__ unchanged ...

    def _checkpointed_layer_forward(
        self, 
        layer_dict: nn.ModuleDict, 
        cd_input: torch.Tensor, 
        layer_idx: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Wrapper for gradient checkpointing.
        Must be a separate function that doesn't capture state (for checkpoint compatibility).
        """
        def layer_fn(x):
            residual = x
            x_norm = layer_dict['norm1'](x)
            x_attn = layer_dict['attention'](x_norm, is_causal=True)
            x = residual + x_attn
            
            residual = x
            x_norm = layer_dict['norm2'](x)
            
            if isinstance(layer_dict['ffn'], MoELayer):
                x_ffn, moe_losses = layer_dict['ffn'](x_norm, train=self.training)
                # Note: moe_losses returned separately since checkpoint doesn't handle dicts
                return residual + x_ffn
            else:
                x_ffn = layer_dict['ffn'](x_norm)
                return residual + x_ffn
        
        return layer_fn

    def forward(self, 
                x: torch.Tensor, 
                return_moe_losses: bool = True
               ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with optional gradient checkpointing."""
        gpu_batchsize = x.shape[0]
        actual_len_dy = x.shape[1]  
        actual_len_cd = x.shape[2] - 3
        device = x.device
        
        # ... existing embedding code through line 2802 ...
        
        # Temporal encoding with Flash + MoE + GRADIENT CHECKPOINTING
        total_aux_loss = torch.tensor(0.0, device=device)
        expert_usage_list = []
        
        for i, layer in enumerate(self.temporal_layers):
            # Determine if this layer should be checkpointed
            should_checkpoint = (
                self.training and 
                self.use_gradient_checkpointing and
                (i % self.checkpoint_every_n_layers == 0)
            )
            
            if should_checkpoint and not isinstance(layer['ffn'], MoELayer):
                # Use gradient checkpointing for non-MoE layers
                # (MoE layers have auxiliary losses that complicate checkpointing)
                def create_custom_forward(layer_module):
                    def custom_forward(x):
                        residual = x
                        x_norm = layer_module['norm1'](x)
                        x_attn = layer_module['attention'](x_norm, is_causal=True)
                        x = residual + x_attn
                        
                        residual = x
                        x_norm = layer_module['norm2'](x)
                        x_ffn = layer_module['ffn'](x_norm)
                        return residual + x_ffn
                    return custom_forward
                
                cd = checkpoint(
                    create_custom_forward(layer),
                    cd,
                    use_reentrant=False  # Recommended for newer PyTorch
                )
            else:
                # Standard forward (MoE layers or non-checkpointed)
                residual = cd
                cd_norm = layer['norm1'](cd)
                cd_attn = layer['attention'](cd_norm, is_causal=True)
                cd = residual + cd_attn
                
                residual = cd
                cd_norm = layer['norm2'](cd)
                
                if isinstance(layer['ffn'], MoELayer):
                    cd_ffn, moe_losses = layer['ffn'](cd_norm, train=self.training)
                    if self.training and return_moe_losses:
                        total_aux_loss = total_aux_loss + moe_losses['aux_loss']
                        if 'expert_usage' in moe_losses:
                            expert_usage_list.append(moe_losses['expert_usage'])
                else:
                    cd_ffn = layer['ffn'](cd_norm)
                
                cd = residual + cd_ffn
        
        # ... rest unchanged ...
```

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
    is_main: bool = True,
    use_ddp: bool = False,
    accumulation_steps: int = 4  # Actually use this now!
) -> Dict[str, float]:
    """
    Train for one epoch with gradient accumulation support.
    
    For batch_size=64 with OOM, use:
    - actual batch_size=16 
    - accumulation_steps=4
    - Effective batch_size = 16 * 4 = 64
    """
    model.train()
    
    nbatch = len(dataloader)
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    batch_metrics_buffer = []  
    moe_metrics_buffer = []
    
    if loss_tracker is None:
        loss_tracker = LossTracker()
    
    # Track accumulated loss for proper averaging
    accumulated_loss = 0.0
    accumulation_counter = 0
    
    for batch_idx, batch in enumerate(dataloader):
        
        if is_main and batch_idx % log_interval == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}')
        
        # GPU utilization check (unchanged)
        if batch_idx == 0 and is_main:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"\n🔍 GPU UTILIZATION CHECK (Batch 0):")
                for gpu_id in range(num_gpus):
                    mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"   GPU {gpu_id}: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        
        # Only zero grad at accumulation boundaries
        if accumulation_counter == 0:
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves memory!
        
        # Get batch data (unchanged)
        age = batch['age']
        gender = batch['gender']
        lob = batch['lob']
        codes = batch['codes']
        dt_cnt = batch['dt_cnt']
        targets_mh = batch['target_multihot']
        y = batch['target']
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        x = x.cuda(non_blocking=True)
        dt_cnt = dt_cnt.cuda(non_blocking=True)
        targets_mh = targets_mh.cuda(non_blocking=True)
        
        # Forward pass
        need_predictions = is_main and (batch_idx % log_interval == 0)
        
        if use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
                if isinstance(result, tuple):
                    total_loss, extras = result
                    output = extras.get('predictions', None) if need_predictions else None
                    moe_losses = extras.get('moe_losses', {})
                    pred_loss = extras.get('pred_loss', total_loss)
                    aux_loss = extras.get('aux_loss', torch.tensor(0.0))
                else:
                    total_loss = result
                    pred_loss = total_loss
                    aux_loss = torch.tensor(0.0, device=device)
                    output = None
                    moe_losses = {}
        else:
            result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
            if isinstance(result, tuple):
                total_loss, extras = result
                output = extras.get('predictions', None) if need_predictions else None
                moe_losses = extras.get('moe_losses', {})
                pred_loss = extras.get('pred_loss', total_loss)
                aux_loss = extras.get('aux_loss', torch.tensor(0.0))
            else:
                total_loss = result
                pred_loss = total_loss
                aux_loss = torch.tensor(0.0, device=device)
                output = None
                moe_losses = {}

        # Handle multi-element tensors from DataParallel
        if total_loss.numel() > 1:
            total_loss = total_loss.mean()
        if pred_loss.numel() > 1:
            pred_loss = pred_loss.mean()
        if aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        scaled_loss = total_loss / accumulation_steps
        
        # Backward pass
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        # Optimizer step at accumulation boundaries
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
        
        # Track losses
        pred_loss_scalar = pred_loss.item()
        aux_loss_scalar = aux_loss.item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
        
        # Logging (unchanged except for metrics)
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                batch_metrics = compute_batch_metrics_lightweight(
                    output, y, 
                    dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt, 
                    config, device
                )
                batch_metrics_buffer.append(batch_metrics)
                
                print(f"    Loss: {pred_loss_scalar:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f}")
                
                if moe_losses and 'expert_usage' in moe_losses:
                    moe_batch_metrics = compute_moe_batch_metrics(moe_losses)
                    moe_metrics_buffer.append(moe_batch_metrics)
        
        # FIXED: Proper cleanup (was buggy: del x, creates tuple)
        del x, targets_mh, dt_cnt
        if output is not None:
            del output
        del pred_loss, total_loss, scaled_loss
        
        # Periodic cleanup
        if batch_idx % 100 == 0:
            gc.collect()
            
            if is_main and device.type == 'cuda' and batch_idx % 500 == 0:
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
                    print(f'    GPU {gpu_id}: {allocated:.2f}GB / {peak:.2f}GB peak')
    
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        global_step += 1
    
    # End-of-epoch cleanup
    if device.type == 'cuda':
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    # ... rest of function unchanged ...
```

```python
def clinical_collate_fn(batch: List[Dict], config: 'BaseConfig') -> Dict[str, Any]:
    """
    Custom collate function with memory-optimized target tensors.
    """
    batch_size = len(batch)
    len_dy = config.len_dy
    target_cd_cnt = config.target_cd_cnt
    
    # Extract each field
    ages = torch.stack([item['age'] for item in batch])
    genders = torch.stack([item['gender'] for item in batch])
    lobs = torch.stack([item['lob'] for item in batch])
    codes = torch.stack([item['codes'] for item in batch])
    dt_cnts = torch.tensor([item['dt_cnt'] for item in batch], dtype=torch.long)
    
    # OPTIMIZED: Use float16 for multi-hot targets (saves 50% memory)
    # For batch_size=64, len_dy=200, target_cd_cnt=6297:
    # float32: 64 * 200 * 6297 * 4 = 322 MB
    # float16: 64 * 200 * 6297 * 2 = 161 MB
    targets_multihot = torch.zeros(
        batch_size, len_dy, target_cd_cnt, 
        dtype=torch.float16  # Changed from float32
    )
    
    for i, item in enumerate(batch):
        target_list = item['target']
        for day_idx, day_codes in enumerate(target_list):
            if day_idx < len_dy and day_codes:
                for code_idx in day_codes:
                    if 0 <= code_idx < target_cd_cnt:
                        targets_multihot[i, day_idx, code_idx] = 1.0
    
    targets_list = [item['target'] for item in batch]
    
    return {
        'age': ages,
        'gender': genders,
        'lob': lobs,
        'codes': codes,
        'dt_cnt': dt_cnts,
        'target_multihot': targets_multihot,
        'target': targets_list
    }
```

```python
def forward(
    self, 
    x: torch.Tensor,
    dt_cnt: torch.Tensor,
    targets: torch.Tensor,
    return_predictions: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    # ... existing code ...
    
    # Ensure targets match output dtype for loss computation
    # BCEWithLogitsLoss can work with float16 in autocast context
    if targets.dtype != output_flat.dtype:
        targets_flat = targets.view(batch_size * actual_len_dy, self.target_cd_cnt).to(output_flat.dtype)
    else:
        targets_flat = targets.view(batch_size * actual_len_dy, self.target_cd_cnt)
    
    # ... rest unchanged ...
```

```python
# Compute loss only on valid positions
if valid_mask.any():
    valid_output = output_flat[valid_mask]
    valid_targets = targets_flat[valid_mask]
    
    # Ensure consistent dtype for loss (important for mixed precision)
    if valid_targets.dtype != valid_output.dtype:
        valid_targets = valid_targets.to(valid_output.dtype)
    
    pred_loss = self.criterion(valid_output, valid_targets)
else:
    pred_loss = torch.tensor(0.0, device=device, requires_grad=True)
```

```python
def _embed_codes_memory_efficient(
    self, 
    x: torch.Tensor, 
    gpu_batchsize: int, 
    actual_len_dy: int, 
    actual_len_cd: int
) -> torch.Tensor:
    """
    Memory-efficient code embedding that processes in chunks.
    
    Original approach creates: [batch, len_dy, len_cd, embedding_size]
    For batch=64, len_dy=200, len_cd=80, emb=256 = 3.3GB
    
    Chunked approach processes subsets and aggregates, reducing peak memory.
    """
    device = x.device
    code_indices = x[:, :, 3:].long()  # [batch, len_dy, len_cd]
    
    # For smaller batches, use standard approach
    if gpu_batchsize * actual_len_dy * actual_len_cd <= 100000:
        cd = self.embedding_cd(code_indices)  # [batch, len_dy, len_cd, emb]
        return cd.sum(dim=-2)  # [batch, len_dy, emb]
    
    # For larger batches, process day-by-day
    cd_res = torch.zeros(
        gpu_batchsize, actual_len_dy, self.config.embedding_size,
        dtype=x.dtype, device=device
    )
    
    # Process in chunks of days to limit memory
    chunk_size = max(1, 50000 // (gpu_batchsize * actual_len_cd))
    
    for start_day in range(0, actual_len_dy, chunk_size):
        end_day = min(start_day + chunk_size, actual_len_dy)
        
        # Get embeddings for this chunk
        chunk_codes = code_indices[:, start_day:end_day, :]  # [batch, chunk_days, len_cd]
        chunk_emb = self.embedding_cd(chunk_codes)  # [batch, chunk_days, len_cd, emb]
        
        # Sum over codes within each day
        cd_res[:, start_day:end_day, :] = chunk_emb.sum(dim=-2)
        
        # Immediately free chunk memory
        del chunk_emb
    
    return cd_res
```

```python
def forward(self, x: torch.Tensor, return_moe_losses: bool = True) -> Tuple[torch.Tensor, Dict]:
    gpu_batchsize = x.shape[0]
    actual_len_dy = x.shape[1]  
    actual_len_cd = x.shape[2] - 3
    device = x.device
    
    # Extract demographics embeddings
    age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
    gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
    lob_emb = self.embedding_lob(x[:, :, 2].long())
    
    # OPTIMIZED: Memory-efficient code embedding
    cd_res = self._embed_codes_memory_efficient(
        x, gpu_batchsize, actual_len_dy, actual_len_cd
    )
    
    # Still need per-code embeddings for daily encoder
    cd = self.embedding_cd(x[:, :, 3:].long())
    
    # ... rest of forward unchanged ...
```

```python
# Try to import bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("bitsandbytes not available. Install with: pip install bitsandbytes")
```

```python
# 4. OPTIMIZER SETUP with memory optimization
if HAS_BITSANDBYTES and torch.cuda.is_available():
    # Use 8-bit Adam for ~75% memory savings on optimizer states
    # For a 100M parameter model: 
    #   - Standard AdamW: ~1.2GB optimizer states
    #   - 8-bit Adam: ~300MB optimizer states
    logger.info("Using 8-bit AdamW optimizer (bitsandbytes)")
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
else:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
```

```python
def _create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: BaseConfig,
    use_bucketing: bool,
    world_size: int = 1,
    logger: Optional[logging.Logger] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create memory-optimized dataloaders."""
    train_dataset = ClinicalDataset(train_data, config)
    val_dataset = ClinicalDataset(val_data, config)
    
    # Optimize worker count based on batch size and memory
    # Fewer workers = less memory for prefetched batches
    if config.batch_size >= 64:
        n_workers = min(2, os.cpu_count() // 2)  # Limit workers for large batches
    else:
        n_workers = max(1, os.cpu_count() // max(world_size, 1) // 2)
    
    collate_fn = create_collate_fn(config)
    
    # Common DataLoader kwargs with memory optimizations
    common_kwargs = {
        'num_workers': n_workers,
        'pin_memory': True,
        'collate_fn': collate_fn,
        'prefetch_factor': 2 if n_workers > 0 else None,  # Limit prefetch queue
        'persistent_workers': n_workers > 0
    }
    
    if use_bucketing:
        # ... bucketing code unchanged ...
        pass
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **{k: v for k, v in common_kwargs.items() if k != 'persistent_workers'}
    )
    
    if logger:
        logger.info(f"DataLoader: {n_workers} workers, prefetch_factor=2")
    
    return train_loader, val_loader
```

```python
def find_optimal_batch_size(
    model: nn.Module,
    config: BaseConfig,
    device: torch.device,
    min_batch: int = 8,
    max_batch: int = 128,
    use_mixed_precision: bool = True
) -> int:
    """
    Binary search to find maximum batch size that doesn't OOM.
    
    Usage:
        optimal_batch = find_optimal_batch_size(model, config, device)
        config.batch_size = optimal_batch
    """
    model.eval()
    
    def can_run_batch(batch_size: int) -> bool:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            # Create dummy inputs
            x = torch.randint(
                0, config.cd_cnt,
                (batch_size, config.len_dy, config.len_cd + 3),
                device=device
            )
            dt_cnt = torch.full((batch_size,), config.len_dy, device=device)
            targets = torch.zeros(
                batch_size, config.len_dy, config.target_cd_cnt,
                dtype=torch.float16 if use_mixed_precision else torch.float32,
                device=device
            )
            
            # Run forward + backward
            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if hasattr(model, 'module'):
                        out = model.module.model(x)
                    else:
                        out = model(x)
            else:
                out = model(x)
            
            if isinstance(out, tuple):
                out = out[0]
            
            loss = out.sum()
            loss.backward()
            
            del x, dt_cnt, targets, out, loss
            torch.cuda.empty_cache()
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            raise e
    
    # Binary search
    low, high = min_batch, max_batch
    best = min_batch
    
    print(f"Finding optimal batch size between {min_batch} and {max_batch}...")
    
    while low <= high:
        mid = (low + high) // 2
        print(f"  Trying batch_size={mid}...", end=" ")
        
        if can_run_batch(mid):
            print("✓")
            best = mid
            low = mid + 1
        else:
            print("✗ OOM")
            high = mid - 1
    
    # Apply safety margin (90% of max)
    safe_batch = int(best * 0.9)
    print(f"Optimal batch size: {best}, using safe value: {safe_batch}")
    
    return safe_batch
```

```python
def run_single_experiment(...):
    # ============================================================
    # 0. MEMORY OPTIMIZATION FLAGS
    # ============================================================
    if torch.cuda.is_available():
        # Enable TF32 for faster computation on Ampere+ GPUs (A100, H100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmark for optimized convolutions
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocation strategy
        # 'expandable_segments' reduces fragmentation
        if hasattr(torch.cuda.memory, '_set_allocator_settings'):
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        
        # Pre-allocate memory pool
        torch.cuda.empty_cache()
        
        # Log GPU memory configuration
        if is_main:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB")
    
    # ... rest of function ...
```

```python
@dataclass 
class MemoryOptimizationConfig:
    """Configuration for memory optimizations."""
    # Gradient checkpointing
    use_gradient_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    
    # Gradient accumulation
    accumulation_steps: int = 4  # Effective batch = batch_size * accumulation_steps
    
    # Mixed precision
    use_mixed_precision: bool = True
    targets_dtype: torch.dtype = torch.float16
    
    # 8-bit optimizer
    use_8bit_optimizer: bool = True
    
    # Memory-efficient embedding
    use_chunked_embedding: bool = True
    embedding_chunk_size: int = 50000
    
    # DataLoader
    num_workers: int = 2
    prefetch_factor: int = 2


def get_memory_optimized_config(
    base_batch_size: int = 64,
    available_memory_gb: float = 16.0  # T4 has 16GB
) -> Tuple[int, MemoryOptimizationConfig]:
    """
    Get optimized configuration based on target batch size and available memory.
    
    Returns:
        actual_batch_size: The per-GPU batch size to use
        mem_config: Memory optimization settings
    """
    mem_config = MemoryOptimizationConfig()
    
    # Estimate memory requirements (rough heuristics)
    # Base memory per sample ≈ 50MB for this model
    base_memory_per_sample = 0.05  # GB
    
    # With gradient checkpointing, reduce activation memory by ~60%
    if mem_config.use_gradient_checkpointing:
        base_memory_per_sample *= 0.4
    
    # With mixed precision, reduce by another ~40%
    if mem_config.use_mixed_precision:
        base_memory_per_sample *= 0.6
    
    # Calculate max batch size that fits in memory
    # Reserve 2GB for model weights and overhead
    usable_memory = available_memory_gb - 2.0
    max_batch_size = int(usable_memory / base_memory_per_sample)
    
    if base_batch_size <= max_batch_size:
        actual_batch_size = base_batch_size
        mem_config.accumulation_steps = 1
    else:
        # Need gradient accumulation
        actual_batch_size = max(8, max_batch_size // 2)  # Use half for safety margin
        mem_config.accumulation_steps = (base_batch_size + actual_batch_size - 1) // actual_batch_size
    
    print(f"Memory optimization config:")
    print(f"  Target batch: {base_batch_size}")
    print(f"  Actual batch: {actual_batch_size}")
    print(f"  Accumulation steps: {mem_config.accumulation_steps}")
    print(f"  Effective batch: {actual_batch_size * mem_config.accumulation_steps}")
    print(f"  Gradient checkpointing: {mem_config.use_gradient_checkpointing}")
    print(f"  Mixed precision: {mem_config.use_mixed_precision}")
    print(f"  8-bit optimizer: {mem_config.use_8bit_optimizer}")
    
    return actual_batch_size, mem_config
```

---

**User**

1. for the 5th change, is the following code correct? 
        # Extract and embed
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long()) 
        cd_res = self._embed_codes_memory_efficient(
            x, gpu_batchsize, actual_len_dy, actual_len_cd
        )
        cd = self.embedding_cd(x[:, :, 3:].long())
        cd_res = cd.sum(-2)

---

**Cursor**

```python
# Extract and embed
age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
lob_emb = self.embedding_lob(x[:, :, 2].long()) 
cd = self.embedding_cd(x[:, :, 3:].long())
cd_res = cd.sum(-2)

# Daily encoding with Flash Attention
cd = cd.reshape(gpu_batchsize * actual_len_dy, actual_len_cd, self.config.embedding_size)
```

```python
cd_res = self._embed_codes_memory_efficient(...)  # This returns the summed version
cd = self.embedding_cd(x[:, :, 3:].long())         # This creates the full embedding
cd_res = cd.sum(-2)                                  # This OVERWRITES the memory-efficient result
```

Good catch! **No, that code is NOT correct** - it has a logical bug that completely negates the memory optimization. Let me explain:

## The Problem

```python
cd_res = self._embed_codes_memory_efficient(...)  # Computes memory-efficient sum → GOOD
cd = self.embedding_cd(x[:, :, 3:].long())         # Creates FULL 3.3GB tensor → BAD
cd_res = cd.sum(-2)                                  # Overwrites the memory-efficient result → DEFEATS PURPOSE
```

The third line **overwrites** `cd_res` with the result from the full embedding, completely wasting the memory-efficient computation.

## The Core Issue

Looking at the original code flow, there are **two uses** of the code embeddings:

1. **`cd_res`** - summed embeddings for skip connection: `[batch, len_dy, emb_size]`
2. **`cd`** - full per-code embeddings for daily encoder: `[batch, len_dy, len_cd, emb_size]`

The daily encoder (attention/pooling) **requires** the full `cd` tensor to process individual code embeddings. So we can't simply avoid creating it.

## Corrected Memory-Efficient Approach

The real memory savings come from **processing the daily encoder in chunks** so we don't keep the full `cd` tensor in memory for the entire forward pass:

```python
cd_res = self._embed_codes_memory_efficient(...)  # Computes memory-efficient sum → GOOD
cd = self.embedding_cd(x[:, :, 3:].long())         # Creates FULL 3.3GB tensor → BAD
cd_res = cd.sum(-2)                                  # Overwrites the memory-efficient result → DEFEATS PURPOSE
```

Now I can see the full flow. Here's the **correct memory-efficient implementation** that processes the daily encoder in chunks:

```python
def _process_daily_encoding_memory_efficient(
    self,
    x: torch.Tensor,
    gpu_batchsize: int,
    actual_len_dy: int,
    actual_len_cd: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient daily encoding that processes in day chunks.
    
    Instead of creating the full [batch, len_dy, len_cd, emb_size] tensor (3.3GB),
    we process in chunks and only keep the reduced [batch, len_dy, emb_size] output.
    
    Returns:
        cd_res: Skip connection sum [batch, len_dy, emb_size]
        cd_daily: Daily-encoded output [batch, len_dy, emb_size]
    """
    device = x.device
    emb_size = self.config.embedding_size
    
    # Pre-allocate output tensors (much smaller than full embeddings)
    cd_res = torch.zeros(gpu_batchsize, actual_len_dy, emb_size, 
                         dtype=x.dtype, device=device)
    cd_daily = torch.zeros(gpu_batchsize, actual_len_dy, emb_size, 
                           dtype=x.dtype, device=device)
    
    # Process in day chunks to limit peak memory
    # For batch=64, len_cd=80, emb=256: each day = 64*80*256*4 = 5.2MB
    # Chunk of 25 days = 130MB (manageable)
    chunk_size = 25  # Process 25 days at a time
    
    for start_day in range(0, actual_len_dy, chunk_size):
        end_day = min(start_day + chunk_size, actual_len_dy)
        num_days_in_chunk = end_day - start_day
        
        # Extract code indices for this chunk of days
        chunk_codes = x[:, start_day:end_day, 3:].long()  # [batch, chunk_days, len_cd]
        
        # Embed codes for this chunk
        chunk_cd = self.embedding_cd(chunk_codes)  # [batch, chunk_days, len_cd, emb]
        
        # Compute skip connection (sum over codes)
        cd_res[:, start_day:end_day, :] = chunk_cd.sum(dim=-2)  # [batch, chunk_days, emb]
        
        # Process through daily encoder
        # Reshape for attention: [batch * chunk_days, len_cd, emb]
        chunk_cd = chunk_cd.reshape(gpu_batchsize * num_days_in_chunk, actual_len_cd, emb_size)
        chunk_cd = torch.swapaxes(chunk_cd, 0, 1)  # [len_cd, batch * chunk_days, emb]
        
        if self.config.use_learnt_att_pool:
            chunk_pooled = self.daily_pooling(chunk_cd)
            chunk_pooled = self.daily_mlp(chunk_pooled)
            chunk_out = self.daily_norm(chunk_pooled)
        else:
            # Pre-norm attention
            residual = chunk_cd
            chunk_cd = self.daily_norm1(chunk_cd)
            chunk_cd = self.daily_attention(chunk_cd, is_causal=False)
            chunk_cd = residual + chunk_cd
            
            # Pre-norm FFN
            residual = chunk_cd
            chunk_cd = self.daily_norm2(chunk_cd)
            chunk_cd = self.daily_ffn(chunk_cd)
            chunk_cd = residual + chunk_cd
            
            # Max pooling
            chunk_cd = chunk_cd.permute(1, 2, 0)  # [batch*chunk_days, emb, len_cd]
            chunk_cd = nn.MaxPool1d(actual_len_cd)(chunk_cd)  # [batch*chunk_days, emb, 1]
            chunk_out = chunk_cd.squeeze(-1)  # [batch*chunk_days, emb]
        
        # Reshape back and store
        cd_daily[:, start_day:end_day, :] = chunk_out.reshape(
            gpu_batchsize, num_days_in_chunk, emb_size
        )
        
        # Free chunk memory immediately
        del chunk_cd, chunk_codes
        if 'chunk_pooled' in dir():
            del chunk_pooled
    
    return cd_res, cd_daily
```

**Then modify the forward method to use it:**

```python
def forward(self, 
            x: torch.Tensor, 
            return_moe_losses: bool = True
           ) -> Tuple[torch.Tensor, Dict]:
    """Forward pass with memory-efficient daily encoding."""
    gpu_batchsize = x.shape[0]
    actual_len_dy = x.shape[1]  
    actual_len_cd = x.shape[2] - 3
    device = x.device
    
    # Extract demographic embeddings (these are small)
    age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
    gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
    lob_emb = self.embedding_lob(x[:, :, 2].long())
    
    # MEMORY-EFFICIENT: Process daily encoding in chunks
    # This avoids creating the full [batch, len_dy, len_cd, emb] tensor
    cd_res, cd = self._process_daily_encoding_memory_efficient(
        x, gpu_batchsize, actual_len_dy, actual_len_cd
    

```python
def _process_daily_encoding_memory_efficient(
    self,
    x: torch.Tensor,
    gpu_batchsize: int,
    actual_len_dy: int,
    actual_len_cd: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient daily encoding that processes in day chunks.
    
    Instead of creating the full [batch, len_dy, len_cd, emb_size] tensor (3.3GB),
    we process in chunks and only keep the reduced [batch, len_dy, emb_size] output.
    
    Returns:
        cd_res: Skip connection sum [batch, len_dy, emb_size]
        cd_daily: Daily-encoded output [batch, len_dy, emb_size]
    """
    device = x.device
    emb_size = self.config.embedding_size
    
    # Pre-allocate output tensors (much smaller than full embeddings)
    cd_res = torch.zeros(gpu_batchsize, actual_len_dy, emb_size, 
                         dtype=x.dtype, device=device)
    cd_daily = torch.zeros(gpu_batchsize, actual_len_dy, emb_size, 
                           dtype=x.dtype, device=device)
    
    # Process in day chunks to limit peak memory
    # For batch=64, len_cd=80, emb=256: each day = 64*80*256*4 = 5.2MB
    # Chunk of 25 days = 130MB (manageable)
    chunk_size = 25  # Process 25 days at a time
    
    for start_day in range(0, actual_len_dy, chunk_size):
        end_day = min(start_day + chunk_size, actual_len_dy)
        num_days_in_chunk = end_day - start_day
        
        # Extract code indices for this chunk of days
        chunk_codes = x[:, start_day:end_day, 3:].long()  # [batch, chunk_days, len_cd]
        
        # Embed codes for this chunk
        chunk_cd = self.embedding_cd(chunk_codes)  # [batch, chunk_days, len_cd, emb]
        
        # Compute skip connection (sum over codes)
        cd_res[:, start_day:end_day, :] = chunk_cd.sum(dim=-2)  # [batch, chunk_days, emb]
        
        # Process through daily encoder
        # Reshape for attention: [batch * chunk_days, len_cd, emb]
        chunk_cd = chunk_cd.reshape(gpu_batchsize * num_days_in_chunk, actual_len_cd, emb_size)
        chunk_cd = torch.swapaxes(chunk_cd, 0, 1)  # [len_cd, batch * chunk_days, emb]
        
        if self.config.use_learnt_att_pool:
            chunk_pooled = self.daily_pooling(chunk_cd)
            chunk_pooled = self.daily_mlp(chunk_pooled)
            chunk_out = self.daily_norm(chunk_pooled)
        else:
            # Pre-norm attention
            residual = chunk_cd
            chunk_cd = self.daily_norm1(chunk_cd)
            chunk_cd = self.daily_attention(chunk_cd, is_causal=False)
            chunk_cd = residual + chunk_cd
            
            # Pre-norm FFN
            residual = chunk_cd
            chunk_cd = self.daily_norm2(chunk_cd)
            chunk_cd = self.daily_ffn(chunk_cd)
            chunk_cd = residual + chunk_cd
            
            # Max pooling
            chunk_cd = chunk_cd.permute(1, 2, 0)  # [batch*chunk_days, emb, len_cd]
            chunk_cd = nn.MaxPool1d(actual_len_cd)(chunk_cd)  # [batch*chunk_days, emb, 1]
            chunk_out = chunk_cd.squeeze(-1)  # [batch*chunk_days, emb]
        
        # Reshape back and store
        cd_daily[:, start_day:end_day, :] = chunk_out.reshape(
            gpu_batchsize, num_days_in_chunk, emb_size
        )
        
        # Free chunk memory immediately
        del chunk_cd, chunk_codes
        if 'chunk_pooled' in dir():
            del chunk_pooled
    
    return cd_res, cd_daily
```

```python
def _process_daily_encoding_memory_efficient(
    self,
    x: torch.Tensor,
    gpu_batchsize: int,
    actual_len_dy: int,
    actual_len_cd: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient daily encoding that processes in day chunks.
    
    Instead of creating the full [batch, len_dy, len_cd, emb_size] tensor (3.3GB),
    we process in chunks and only keep the reduced [batch, len_dy, emb_size] output.
    
    Returns:
        cd_res: Skip connection sum [batch, len_dy, emb_size]
        cd_daily: Daily-encoded output [batch, len_dy, emb_size]
    """
    device = x.device
    emb_size = self.config.embedding_size
    
    # Pre-allocate output tensors (much smaller than full embeddings)
    cd_res = torch.zeros(gpu_batchsize, actual_len_dy, emb_size, 
                         dtype=x.dtype, device=device)
    cd_daily = torch.zeros(gpu_batchsize, actual_len_dy, emb_size, 
                           dtype=x.dtype, device=device)
    
    # Process in day chunks to limit peak memory
    # For batch=64, len_cd=80, emb=256: each day = 64*80*256*4 = 5.2MB
    # Chunk of 25 days = 130MB (manageable)
    chunk_size = 25  # Process 25 days at a time
    
    for start_day in range(0, actual_len_dy, chunk_size):
        end_day = min(start_day + chunk_size, actual_len_dy)
        num_days_in_chunk = end_day - start_day
        
        # Extract code indices for this chunk of days
        chunk_codes = x[:, start_day:end_day, 3:].long()  # [batch, chunk_days, len_cd]
        
        # Embed codes for this chunk
        chunk_cd = self.embedding_cd(chunk_codes)  # [batch, chunk_days, len_cd, emb]
        
        # Compute skip connection (sum over codes)
        cd_res[:, start_day:end_day, :] = chunk_cd.sum(dim=-2)  # [batch, chunk_days, emb]
        
        # Process through daily encoder
        # Reshape for attention: [batch * chunk_days, len_cd, emb]
        chunk_cd = chunk_cd.reshape(gpu_batchsize * num_days_in_chunk, actual_len_cd, emb_size)
        chunk_cd = torch.swapaxes(chunk_cd, 0, 1)  # [len_cd, batch * chunk_days, emb]
        
        if self.config.use_learnt_att_pool:
            chunk_pooled = self.daily_pooling(chunk_cd)
            chunk_pooled = self.daily_mlp(chunk_pooled)
            chunk_out = self.daily_norm(chunk_pooled)
        else:
            # Pre-norm attention
            residual = chunk_cd
            chunk_cd = self.daily_norm1(chunk_cd)
            chunk_cd = self.daily_attention(chunk_cd, is_causal=False)
            chunk_cd = residual + chunk_cd
            
            # Pre-norm FFN
            residual = chunk_cd
            chunk_cd = self.daily_norm2(chunk_cd)
            chunk_cd = self.daily_ffn(chunk_cd)
            chunk_cd = residual + chunk_cd
            
            # Max pooling
            chunk_cd = chunk_cd.permute(1, 2, 0)  # [batch*chunk_days, emb, len_cd]
            chunk_cd = nn.MaxPool1d(actual_len_cd)(chunk_cd)  # [batch*chunk_days, emb, 1]
            chunk_out = chunk_cd.squeeze(-1)  # [batch*chunk_days, emb]
        
        # Reshape back and store
        cd_daily[:, start_day:end_day, :] = chunk_out.reshape(
            gpu_batchsize, num_days_in_chunk, emb_size
        )
        
        # Free chunk memory immediately
        del chunk_cd, chunk_codes
        if 'chunk_pooled' in dir():
            del chunk_pooled
    
    return cd_res, cd_daily
```

```python
def forward(self, 
            x: torch.Tensor, 
            return_moe_losses: bool = True
           ) -> Tuple[torch.Tensor, Dict]:
    """Forward pass with memory-efficient daily encoding."""
    gpu_batchsize = x.shape[0]
    actual_len_dy = x.shape[1]  
    actual_len_cd = x.shape[2] - 3
    device = x.device
    
    # Extract demographic embeddings (these are small)
    age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
    gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
    lob_emb = self.embedding_lob(x[:, :, 2].long())
    
    # MEMORY-EFFICIENT: Process daily encoding in chunks
    # This avoids creating the full [batch, len_dy, len_cd, emb] tensor
    cd_res, cd = self._process_daily_encoding_memory_efficient(
        x, gpu_batchsize, actual_len_dy, actual_len_cd
    
```

---

**User**

Ok I got the following errors; inspect the root cause and analyze why and propsoed how to fix it 
🔍 GPU UTILIZATION CHECK (Batch 0):
   GPU 0: 0.11 GB allocated, 0.11 GB reserved
   GPU 1: 0.01 GB allocated, 0.02 GB reserved
   GPU 2: 0.01 GB allocated, 0.02 GB reserved
   GPU 3: 0.01 GB allocated, 0.02 GB reserved
    Loss: 0.8021 | R@10: 0.018 | R@20: 0.027 | P@10: 0.002 | P@20: 0.001 | mAP20: 0.244 | mAP50: 0.118 | Brier: 0.2924
    GPU 0: 0.81GB / 7.44GB peak
    GPU 1: 0.01GB / 6.72GB peak
    GPU 2: 0.01GB / 6.71GB peak
    GPU 3: 0.01GB / 6.70GB peak
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[42], line 2
      1 cleanup_gpu_memory_hard()
----> 2 baseline_results = run_single_experiment(
      3     exp_name=EXP_NAME,
      4     moe_config=moe_config,
      5     use_learnt_att_pool=use_learnt_att_pool,
      6     train_data=train_df,
      7     val_data=val_df,
      8     device=device,
      9     epochs=EPOCHS,
     10     experiment_round=EXPERIMENT_ROUND,
     11     embedding_size=EMBEDDING_SIZE,
     12     log_dir='logs',
     13     save_model=True
     14 )

Cell In[29], line 169, in run_single_experiment(exp_name, moe_config, use_learnt_att_pool, train_data, val_data, device, epochs, code_frequencies, log_dir, experiment_round, check_embeddings_every, log_metrics_every, resume_from, checkpoint_dir, embedding_size, local_rank, world_size, save_model)
    166 loss_tracker.reset_epoch()
    168 # Train
--> 169 train_metrics = train_epoch(
    170     model=model,
    171     dataloader=train_loader,
    172     optimizer=optimizer,
    173     scheduler=scheduler,
    174     criterion=criterion,
    175     config=config,
    176     device=device,
    177     scaler=scaler,
    178     use_mixed_precision=use_mixed_precision,
    179     moe_config=moe_config,
    180     epoch=epoch,
    181     use_bucketing=use_bucketing,
    182     log_interval=log_metrics_every,
    183     global_step=global_step,
    184     loss_tracker=loss_tracker,
    185     is_main=is_main,
    186     use_ddp=use_ddp
    187 )
    188 global_step = train_metrics['global_step']
    190 # Evaluate

Cell In[19], line 174, in train_epoch(model, dataloader, optimizer, scheduler, criterion, config, device, scaler, use_mixed_precision, moe_config, epoch, use_bucketing, log_interval, global_step, loss_tracker, is_main, use_ddp, accumulation_steps)
    172     scaler.update()
    173 else:
--> 174     total_loss.backward()
    175     torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    176     optimizer.step()

File /opt/conda/lib/python3.10/site-packages/torch/_tensor.py:625, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
    615 if has_torch_function_unary(self):
    616     return handle_torch_function(
    617         Tensor.backward,
    618         (self,),
   (...)
    623         inputs=inputs,
    624     )
--> 625 torch.autograd.backward(
    626     self, gradient, retain_graph, create_graph, inputs=inputs
    627 )

File /opt/conda/lib/python3.10/site-packages/torch/autograd/__init__.py:354, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    349     retain_graph = create_graph
    351 # The reason we repeat the same comment below is that
    352 # some Python versions print out the first line of a multi-line function
    353 # calls in the traceback and some print out the last line
--> 354 _engine_run_backward(
    355     tensors,
    356     grad_tensors_,
    357     retain_graph,
    358     create_graph,
    359     inputs_tuple,
    360     allow_unreachable=True,
    361     accumulate_grad=True,
    362 )

File /opt/conda/lib/python3.10/site-packages/torch/autograd/graph.py:841, in _engine_run_backward(t_outputs, *args, **kwargs)
    839     unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    840 try:
--> 841     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    842         t_outputs, *args, **kwargs
    843     )  # Calls into the C++ engine to run the backward pass
    844 finally:
    845     if attach_logging_hooks:

RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.

---

**Cursor**

This is a classic PyTorch autograd error. Let me analyze the root cause and look at the relevant code:

Let me see the rest of the gradient accumulation logic:

```python
   scaled_loss.backward()  # First backward - OK
```

```python
   if accumulation_counter >= accumulation_steps:
       if use_mixed_precision:
           scaler.scale(total_loss).backward()  # SECOND backward on same graph!
       else:
           total_loss.backward()  # SECOND backward on same graph!
```

## Root Cause Analysis

**I found the bug!** The gradient accumulation logic has a critical error. Look at lines 4373-4388:

```python
# Line 4373-4375: FIRST backward on scaled_loss
scaled_loss.backward()  # ← This frees the computation graph

# Lines 4380-4388: Then inside the accumulation boundary check...
if accumulation_counter >= accumulation_steps:
    if use_mixed_precision:
        scaler.scale(total_loss).backward()  # ← SECOND backward attempt - CRASH!
    else:
        total_loss.backward()  # ← SECOND backward attempt - CRASH!
```

**The problem:** After calling `scaled_loss.backward()`, the computation graph is freed. Then the code tries to call `total_loss.backward()` AGAIN inside the accumulation boundary. Since `scaled_loss = total_loss / accumulation_steps`, they share the same computation graph, and you cannot backward through it twice.

Let me also check where `optimizer.zero_grad()` is called:

```python
# Line 4373-4375: FIRST backward on scaled_loss
scaled_loss.backward()  # ← This frees the computation graph

# Lines 4380-4388: Then inside the accumulation boundary check...
if accumulation_counter >= accumulation_steps:
    if use_mixed_precision:
        scaler.scale(total_loss).backward()  # ← SECOND backward attempt - CRASH!
    else:
        total_loss.backward()  # ← SECOND backward attempt - CRASH!
```

Now I have the full picture. Here's the complete fix:

---

## The Bug

The gradient accumulation implementation is **double-calling backward**:

1. **Line 4373-4375**: First backward is called on every batch ✓
2. **Lines 4382 and 4388**: Second backward is called at accumulation boundaries ✗ (THIS IS WRONG)

After the first `.backward()`, the computation graph is freed. The second `.backward()` tries to use the already-freed graph, causing the error.

---

## The Fix

Replace the backward pass section (lines 4369-4397) with this corrected version:

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (with gradient accumulation)
        # ============================================================
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        scaled_loss = total_loss / accumulation_steps
        
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss (detach to avoid graph retention)
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        # Only step optimizer at accumulation boundaries
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                # Unscale gradients, clip, and step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients and step
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Reset accumulation tracking
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
```

**Key changes:**
1. **Removed the duplicate `.backward()` calls** inside the `if accumulation_counter >= accumulation_steps:` block
2. Only `optimizer.step()` is called at accumulation boundaries (not backward)
3. Gradients are clipped AFTER unscaling (for mixed precision) or directly (for fp32)

---

## Additional Issue: GPU Imbalance

Your GPU utilization shows a major problem:
```
GPU 0: 0.81GB / 7.44GB peak
GPU 1: 0.01GB / 6.72GB peak
GPU 2: 0.01GB / 6.71GB peak
GPU 3: 0.01GB / 6.70GB peak
```

**GPUs 1-3 are barely being used!** This suggests `nn.DataParallel` isn't properly distributing work. The likely cause is that loss is computed on GPU 0 only (after gathering outputs), not distributed.

This is actually expected behavior with your `DataParallelWrapper` design where loss computation happens inside the wrapper. Let me verify:

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (with gradient accumulation)
        # ============================================================
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        scaled_loss = total_loss / accumulation_steps
        
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss (detach to avoid graph retention)
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        # Only step optimizer at accumulation boundaries
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                # Unscale gradients, clip, and step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients and step
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Reset accumulation tracking
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
```

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (with gradient accumulation)
        # ============================================================
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        scaled_loss = total_loss / accumulation_steps
        
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss (detach to avoid graph retention)
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        # Only step optimizer at accumulation boundaries
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                # Unscale gradients, clip, and step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients and step
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Reset accumulation tracking
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
```

```plaintext
GPU 0: 0.81GB / 7.44GB peak
GPU 1: 0.01GB / 6.72GB peak
GPU 2: 0.01GB / 6.71GB peak
GPU 3: 0.01GB / 6.70GB peak
```

The GPU imbalance (0.81GB active on GPU0, 0.01GB on others) is actually **expected during the logging check** at batch 0 - you're checking memory BEFORE the forward pass distributes work. The peak memory (7.44GB vs 6.7GB) shows the model IS being replicated.

The immediate issue is the **double backward** bug. Here's the complete corrected training loop section:

---

## Complete Corrected `train_epoch` Backward Section

Replace **lines 4367-4397** with:

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (with gradient accumulation)
        # ============================================================
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        # This allows effective batch_size = actual_batch_size * accumulation_steps
        scaled_loss = total_loss / accumulation_steps
        
        # Backward pass - gradients accumulate automatically
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss (detach to break computation graph reference)
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        # Optimizer step only at accumulation boundaries
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                # 1. Unscale gradients for clipping
                scaler.unscale_(optimizer)
                # 2. Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                # 3. Optimizer step (skips if gradients are inf/nan)
                scaler.step(optimizer)
                # 4. Update scaler for next iteration
                scaler.update()
            else:
                # 1. Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                # 2. Optimizer step
                optimizer.step()
            
            # Step scheduler (if using step-based scheduling)
            if scheduler is not None:
                scheduler.step()
            
            # Reset for next accumulation cycle
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
```

---

## Summary of Changes

| Line | Old Code (Buggy) | Fixed Code |
|------|------------------|------------|
| 4372-4375 | `scaled_loss.backward()` | Keep as-is ✓ |
| 4380-4388 | `if accumulation_counter >= accumulation_steps: total_loss.backward()` | **Remove the `.backward()` call** - only keep `optimizer.step()` |

**The key insight:** With gradient accumulation, you call `.backward()` on EVERY mini-batch (which accumulates gradients in `.grad`), but only call `.step()` periodically. You do NOT call backward again at the step boundary.

---

## Optional: If Not Using Accumulation

If you want to disable gradient accumulation temporarily (to simplify debugging), set `accumulation_steps=1` when calling `train_epoch()`, or use this simplified version:

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (simple, no accumulation)
        # ============================================================
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
        
        global_step += 1
```

This matches the original (pre-accumulation) logic and should work immediately.

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (with gradient accumulation)
        # ============================================================
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        # This allows effective batch_size = actual_batch_size * accumulation_steps
        scaled_loss = total_loss / accumulation_steps
        
        # Backward pass - gradients accumulate automatically
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss (detach to break computation graph reference)
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        # Optimizer step only at accumulation boundaries
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                # 1. Unscale gradients for clipping
                scaler.unscale_(optimizer)
                # 2. Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                # 3. Optimizer step (skips if gradients are inf/nan)
                scaler.step(optimizer)
                # 4. Update scaler for next iteration
                scaler.update()
            else:
                # 1. Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                # 2. Optimizer step
                optimizer.step()
            
            # Step scheduler (if using step-based scheduling)
            if scheduler is not None:
                scheduler.step()
            
            # Reset for next accumulation cycle
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
```

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (simple, no accumulation)
        # ============================================================
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
        
        global_step += 1
```

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (with gradient accumulation)
        # ============================================================
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        # This allows effective batch_size = actual_batch_size * accumulation_steps
        scaled_loss = total_loss / accumulation_steps
        
        # Backward pass - gradients accumulate automatically
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss (detach to break computation graph reference)
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        # Optimizer step only at accumulation boundaries
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                # 1. Unscale gradients for clipping
                scaler.unscale_(optimizer)
                # 2. Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                # 3. Optimizer step (skips if gradients are inf/nan)
                scaler.step(optimizer)
                # 4. Update scaler for next iteration
                scaler.update()
            else:
                # 1. Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                # 2. Optimizer step
                optimizer.step()
            
            # Step scheduler (if using step-based scheduling)
            if scheduler is not None:
                scheduler.step()
            
            # Reset for next accumulation cycle
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
```

```python
        # ============================================================
        # STEP 5: BACKWARD PASS (simple, no accumulation)
        # ============================================================
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
        
        global_step += 1
```

---

**User**

ok, how to correctly print the GPU utilization informaiton? Explian in detail and hwo to track the real use of GPU in forrwad and backward 
The GPU imbalance (0.81GB active on GPU0, 0.01GB on others) is actually expected during the logging check at batch 0 - you're checking memory BEFORE the forward pass distributes work. The peak memory (7.44GB vs 6.7GB) shows the model IS being replicated.

---

**Cursor**

Great question! Let me explain GPU memory measurement timing and provide a proper utility for tracking GPU utilization during forward and backward passes.

## Understanding GPU Memory Measurement Timing

### Why Your Current Check Shows Imbalanced Memory

```python
# Your current code (line 4287-4294):
if batch_idx == 0 and is_main:
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n🔍 GPU UTILIZATION CHECK (Batch 0):")
        for gpu_id in range(num_gpus):
            mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
            # ...
```

**Problem:** This runs BEFORE the forward pass! At this point:
- GPU 0 has data that was just moved with `.cuda()` (line 4317-4319)
- GPUs 1-3 have nothing yet - DataParallel hasn't scattered data

### When GPU Memory Changes

| Phase | GPU 0 | GPUs 1-3 | What Happens |
|-------|-------|----------|--------------|
| Before forward | Has input data | Empty | Data loaded to GPU 0 |
| During forward (scatter) | 1/4 of batch | 1/4 of batch each | DataParallel scatters |
| During forward (compute) | Activations grow | Activations grow | Each GPU runs forward |
| After forward (gather) | All outputs | Outputs freed | Outputs gathered to GPU 0 |
| During backward | Gradients computed | Gradients computed | Each GPU runs backward |
| After backward | Has all gradients | Gradients synced | Gradient all-reduce |

---

## Correct GPU Utilization Tracking

Here's a comprehensive GPU monitoring utility you can add:

```python
class GPUMemoryTracker:
    """
    Track GPU memory at different stages of training.
    
    Usage:
        tracker = GPUMemoryTracker()
        
        # In training loop:
        tracker.record("before_forward")
        output = model(x)
        tracker.record("after_forward")
        loss.backward()
        tracker.record("after_backward")
        optimizer.step()
        tracker.record("after_optimizer")
        
        # Print summary:
        tracker.print_summary()
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.enabled else 0
        self.records = {}  # {stage_name: {gpu_id: (allocated, reserved, peak)}}
        
    def record(self, stage_name: str):
        """Record memory usage at a specific stage."""
        if not self.enabled:
            return
            
        # Synchronize to ensure all GPU operations are complete
        torch.cuda.synchronize()
        
        self.records[stage_name] = {}
        for gpu_id in range(self.num_gpus):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
            self.records[stage_name][gpu_id] = (allocated, reserved, peak)
    
    def reset_peak(self):
        """Reset peak memory stats for fresh measurement."""
        if self.enabled:
            for gpu_id in range(self.num_gpus):
                torch.cuda.reset_peak_memory_stats(gpu_id)
    
    def print_stage(self, stage_name: str):
        """Print memory for a single stage."""
        if stage_name not in self.records:
            print(f"No record for stage: {stage_name}")
            return
            
        print(f"\n📊 GPU Memory @ {stage_name}:")
        for gpu_id, (alloc, res, peak) in self.records[stage_name].items():
            print(f"   GPU {gpu_id}: {alloc:.2f}GB allocated, {res:.2f}GB reserved, {peak:.2f}GB peak")
    
    def print_summary(self):
        """Print all recorded stages."""
        if not self.records:
            print("No GPU memory records.")
            return
            
        print("\n" + "="*70)
        print("GPU MEMORY SUMMARY")
        print("="*70)
        
        # Header
        stages = list(self.records.keys())
        print(f"{'GPU':<6}", end="")
        for stage in stages:
            print(f"{stage:<20}", end="")
        print()
        print("-"*70)
        
        # Per-GPU rows
        for gpu_id in range(self.num_gpus):
            print(f"GPU {gpu_id:<2}", end="")
            for stage in stages:
                alloc, _, _ = self.records[stage][gpu_id]
                print(f"{alloc:>6.2f}GB             ", end="")
            print()
        
        print("="*70)
    
    def print_balance_check(self, stage_name: str = None):
        """Check if work is balanced across GPUs."""
        stage = stage_name or list(self.records.keys())[-1]
        if stage not in self.records:
            return
            
        allocations = [self.records[stage][i][0] for i in range(self.num_gpus)]
        mean_alloc = sum(allocations) / len(allocations)
        max_diff = max(abs(a - mean_alloc) for a in allocations)
        
        print(f"\n⚖️ Balance Check @ {stage}:")
        print(f"   Mean allocation: {mean_alloc:.2f}GB")
        print(f"   Max deviation: {max_diff:.2f}GB ({max_diff/mean_alloc*100:.1f}%)")
        
        if max_diff / mean_alloc > 0.2:
            print("   ⚠️ WARNING: GPUs are imbalanced (>20% deviation)")
        else:
            print("   ✅ GPUs are well balanced")
```

---

## How to Use in `train_epoch()`

Replace your current GPU check with this proper instrumentation:

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    # ... other params ...
    track_gpu_memory: bool = True,  # Add this parameter
    gpu_track_batches: List[int] = [0, 1, 10]  # Which batches to track
) -> Dict[str, float]:
    """Train for one epoch with optional GPU memory tracking."""
    
    model.train()
    
    # Initialize GPU tracker
    gpu_tracker = GPUMemoryTracker(enabled=track_gpu_memory and is_main)
    
    for batch_idx, batch in enumerate(dataloader):
        
        # Determine if we should track this batch
        should_track = (
            track_gpu_memory and 
            is_main and 
            batch_idx in gpu_track_batches
        )
        
        if should_track:
            gpu_tracker.reset_peak()
            print(f"\n🔍 Detailed GPU tracking for batch {batch_idx}")
        
        # ============================================================
        # STEP 1: PREPARE DATA
        # ============================================================
        if accumulation_counter == 0:
            optimizer.zero_grad(set_to_none=True)
        
        age = batch['age']
        gender = batch['gender']
        lob = batch['lob']
        codes = batch['codes']
        dt_cnt = batch['dt_cnt']
        targets_mh = batch['target_multihot']
        y = batch['target']
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        x = x.cuda(non_blocking=True)
        dt_cnt = dt_cnt.cuda(non_blocking=True)
        targets_mh = targets_mh.cuda(non_blocking=True)
        
        if should_track:
            gpu_tracker.record("1_after_data_to_gpu")
        
        # ============================================================
        # STEP 2: FORWARD PASS
        # ============================================================
        need_predictions = is_main and (batch_idx % log_interval == 0)
        
        if use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
        else:
            result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
        
        if should_track:
            gpu_tracker.record("2_after_forward")
        
        # Parse result
        if isinstance(result, tuple):
            total_loss, extras = result
            output = extras.get('predictions', None) if need_predictions else None
            moe_losses = extras.get('moe_losses', {})
            pred_loss = extras.get('pred_loss', total_loss)
            aux_loss = extras.get('aux_loss', torch.tensor(0.0))
        else:
            total_loss = result
            pred_loss = total_loss
            aux_loss = torch.tensor(0.0, device=device)
            output = None
            moe_losses = {}

        # Handle DataParallel multi-element tensors
        if total_loss.numel() > 1:
            total_loss = total_loss.mean()
        if pred_loss.numel() > 1:
            pred_loss = pred_loss.mean()
        if aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        
        # ============================================================
        # STEP 3: BACKWARD PASS
        # ============================================================
        scaled_loss = total_loss / accumulation_steps
        
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        if should_track:
            gpu_tracker.record("3_after_backward")
        
        # ============================================================
        # STEP 4: OPTIMIZER STEP (at accumulation boundary)
        # ============================================================
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            if should_track:
                gpu_tracker.record("4_after_optimizer")
            
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
        
        # ============================================================
        # PRINT GPU SUMMARY FOR TRACKED BATCHES
        # ============================================================
        if should_track:
            gpu_tracker.print_summary()
            gpu_tracker.print_balance_check("2_after_forward")
        
        # ... rest of cleanup and logging ...
```

---

## Expected Output (When Working Correctly)

When DataParallel is working properly, you should see output like:

```
🔍 Detailed GPU tracking for batch 0

======================================================================
GPU MEMORY SUMMARY
======================================================================
GPU   1_after_data_to_gpu  2_after_forward      3_after_backward     4_after_optimizer   
----------------------------------------------------------------------
GPU 0   0.15GB               1.85GB               3.42GB               2.10GB             
GPU 1   0.15GB               1.82GB               3.38GB               2.08GB             
GPU 2   0.15GB               1.83GB               3.40GB               2.09GB             
GPU 3   0.15GB               1.84GB               3.41GB               2.08GB             
======================================================================

⚖️ Balance Check @ 2_after_forward:
   Mean allocation: 1.84GB
   Max deviation: 0.02GB (1.1%)
   ✅ GPUs are well balanced
```

---

## What To Look For

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| **After forward** | All GPUs have similar memory | GPU 0 has much more than others |
| **After backward** | All GPUs have similar memory | GPU 0 has much more than others |
| **Balance deviation** | <10% difference | >20% difference |
| **Peak memory** | Similar across GPUs | GPU 0 peaks much higher |

---

## Quick Diagnostic (Add This Temporarily)

For a quick check without the full tracker, add this right after forward:

```python
# Quick GPU balance check after forward
if batch_idx == 0 and is_main:
    torch.cuda.synchronize()  # IMPORTANT: ensure forward is complete
    print("\n🔍 GPU Memory AFTER FORWARD (batch 0):")
    for gpu_id in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
        print(f"   GPU {gpu_id}: {alloc:.2f}GB")
```

The key difference from your current code is the **`torch.cuda.synchronize()`** and placing it **AFTER the forward pass**, not before.

```python
# Your current code (line 4287-4294):
if batch_idx == 0 and is_main:
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n🔍 GPU UTILIZATION CHECK (Batch 0):")
        for gpu_id in range(num_gpus):
            mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
            # ...
```

```python
class GPUMemoryTracker:
    """
    Track GPU memory at different stages of training.
    
    Usage:
        tracker = GPUMemoryTracker()
        
        # In training loop:
        tracker.record("before_forward")
        output = model(x)
        tracker.record("after_forward")
        loss.backward()
        tracker.record("after_backward")
        optimizer.step()
        tracker.record("after_optimizer")
        
        # Print summary:
        tracker.print_summary()
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.enabled else 0
        self.records = {}  # {stage_name: {gpu_id: (allocated, reserved, peak)}}
        
    def record(self, stage_name: str):
        """Record memory usage at a specific stage."""
        if not self.enabled:
            return
            
        # Synchronize to ensure all GPU operations are complete
        torch.cuda.synchronize()
        
        self.records[stage_name] = {}
        for gpu_id in range(self.num_gpus):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
            self.records[stage_name][gpu_id] = (allocated, reserved, peak)
    
    def reset_peak(self):
        """Reset peak memory stats for fresh measurement."""
        if self.enabled:
            for gpu_id in range(self.num_gpus):
                torch.cuda.reset_peak_memory_stats(gpu_id)
    
    def print_stage(self, stage_name: str):
        """Print memory for a single stage."""
        if stage_name not in self.records:
            print(f"No record for stage: {stage_name}")
            return
            
        print(f"\n📊 GPU Memory @ {stage_name}:")
        for gpu_id, (alloc, res, peak) in self.records[stage_name].items():
            print(f"   GPU {gpu_id}: {alloc:.2f}GB allocated, {res:.2f}GB reserved, {peak:.2f}GB peak")
    
    def print_summary(self):
        """Print all recorded stages."""
        if not self.records:
            print("No GPU memory records.")
            return
            
        print("\n" + "="*70)
        print("GPU MEMORY SUMMARY")
        print("="*70)
        
        # Header
        stages = list(self.records.keys())
        print(f"{'GPU':<6}", end="")
        for stage in stages:
            print(f"{stage:<20}", end="")
        print()
        print("-"*70)
        
        # Per-GPU rows
        for gpu_id in range(self.num_gpus):
            print(f"GPU {gpu_id:<2}", end="")
            for stage in stages:
                alloc, _, _ = self.records[stage][gpu_id]
                print(f"{alloc:>6.2f}GB             ", end="")
            print()
        
        print("="*70)
    
    def print_balance_check(self, stage_name: str = None):
        """Check if work is balanced across GPUs."""
        stage = stage_name or list(self.records.keys())[-1]
        if stage not in self.records:
            return
            
        allocations = [self.records[stage][i][0] for i in range(self.num_gpus)]
        mean_alloc = sum(allocations) / len(allocations)
        max_diff = max(abs(a - mean_alloc) for a in allocations)
        
        print(f"\n⚖️ Balance Check @ {stage}:")
        print(f"   Mean allocation: {mean_alloc:.2f}GB")
        print(f"   Max deviation: {max_diff:.2f}GB ({max_diff/mean_alloc*100:.1f}%)")
        
        if max_diff / mean_alloc > 0.2:
            print("   ⚠️ WARNING: GPUs are imbalanced (>20% deviation)")
        else:
            print("   ✅ GPUs are well balanced")
```

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    # ... other params ...
    track_gpu_memory: bool = True,  # Add this parameter
    gpu_track_batches: List[int] = [0, 1, 10]  # Which batches to track
) -> Dict[str, float]:
    """Train for one epoch with optional GPU memory tracking."""
    
    model.train()
    
    # Initialize GPU tracker
    gpu_tracker = GPUMemoryTracker(enabled=track_gpu_memory and is_main)
    
    for batch_idx, batch in enumerate(dataloader):
        
        # Determine if we should track this batch
        should_track = (
            track_gpu_memory and 
            is_main and 
            batch_idx in gpu_track_batches
        )
        
        if should_track:
            gpu_tracker.reset_peak()
            print(f"\n🔍 Detailed GPU tracking for batch {batch_idx}")
        
        # ============================================================
        # STEP 1: PREPARE DATA
        # ============================================================
        if accumulation_counter == 0:
            optimizer.zero_grad(set_to_none=True)
        
        age = batch['age']
        gender = batch['gender']
        lob = batch['lob']
        codes = batch['codes']
        dt_cnt = batch['dt_cnt']
        targets_mh = batch['target_multihot']
        y = batch['target']
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        x = x.cuda(non_blocking=True)
        dt_cnt = dt_cnt.cuda(non_blocking=True)
        targets_mh = targets_mh.cuda(non_blocking=True)
        
        if should_track:
            gpu_tracker.record("1_after_data_to_gpu")
        
        # ============================================================
        # STEP 2: FORWARD PASS
        # ============================================================
        need_predictions = is_main and (batch_idx % log_interval == 0)
        
        if use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
        else:
            result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
        
        if should_track:
            gpu_tracker.record("2_after_forward")
        
        # Parse result
        if isinstance(result, tuple):
            total_loss, extras = result
            output = extras.get('predictions', None) if need_predictions else None
            moe_losses = extras.get('moe_losses', {})
            pred_loss = extras.get('pred_loss', total_loss)
            aux_loss = extras.get('aux_loss', torch.tensor(0.0))
        else:
            total_loss = result
            pred_loss = total_loss
            aux_loss = torch.tensor(0.0, device=device)
            output = None
            moe_losses = {}

        # Handle DataParallel multi-element tensors
        if total_loss.numel() > 1:
            total_loss = total_loss.mean()
        if pred_loss.numel() > 1:
            pred_loss = pred_loss.mean()
        if aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        
        # ============================================================
        # STEP 3: BACKWARD PASS
        # ============================================================
        scaled_loss = total_loss / accumulation_steps
        
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        if should_track:
            gpu_tracker.record("3_after_backward")
        
        # ============================================================
        # STEP 4: OPTIMIZER STEP (at accumulation boundary)
        # ============================================================
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1
        
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            if should_track:
                gpu_tracker.record("4_after_optimizer")
            
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1
        
        # ============================================================
        # PRINT GPU SUMMARY FOR TRACKED BATCHES
        # ============================================================
        if should_track:
            gpu_tracker.print_summary()
            gpu_tracker.print_balance_check("2_after_forward")
        
        # ... rest of cleanup and logging ...
```

```plaintext
🔍 Detailed GPU tracking for batch 0

======================================================================
GPU MEMORY SUMMARY
======================================================================
GPU   1_after_data_to_gpu  2_after_forward      3_after_backward     4_after_optimizer   
----------------------------------------------------------------------
GPU 0   0.15GB               1.85GB               3.42GB               2.10GB             
GPU 1   0.15GB               1.82GB               3.38GB               2.08GB             
GPU 2   0.15GB               1.83GB               3.40GB               2.09GB             
GPU 3   0.15GB               1.84GB               3.41GB               2.08GB             
======================================================================

⚖️ Balance Check @ 2_after_forward:
   Mean allocation: 1.84GB
   Max deviation: 0.02GB (1.1%)
   ✅ GPUs are well balanced
```

```python
# Quick GPU balance check after forward
if batch_idx == 0 and is_main:
    torch.cuda.synchronize()  # IMPORTANT: ensure forward is complete
    print("\n🔍 GPU Memory AFTER FORWARD (batch 0):")
    for gpu_id in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
        print(f"   GPU {gpu_id}: {alloc:.2f}GB")
```

---

**User**

Ok now I have implemented whaty ou have suggsgted to track GPU memory. but got the following error agiain, OOM 

---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
Cell In[41], line 2
      1 cleanup_gpu_memory_hard()
----> 2 baseline_results = run_single_experiment(
      3     exp_name=EXP_NAME,
      4     moe_config=moe_config,
      5     use_learnt_att_pool=use_learnt_att_pool,
      6     train_data=train_df.sample(1280),
      7     val_data=val_df.sample(128),
      8     device=device,
      9     epochs=EPOCHS,
     10     experiment_round=EXPERIMENT_ROUND,
     11     embedding_size=EMBEDDING_SIZE,
     12     log_dir='logs',
     13     save_model=True
     14 )

Cell In[29], line 216, in run_single_experiment(exp_name, moe_config, use_learnt_att_pool, train_data, val_data, device, epochs, code_frequencies, log_dir, experiment_round, check_embeddings_every, log_metrics_every, resume_from, checkpoint_dir, embedding_size, local_rank, world_size, save_model)
    214 if epoch % check_embeddings_every == 0:
    215     logger.info("Computing embedding quality...")
--> 216     emb_metrics = compute_embedding_quality_epoch(
    217         model, val_data, config, device,
    218         num_samples=200,
    219         use_mixed_precision=use_mixed_precision
    220     )
    221     val_metrics.update(emb_metrics)
    222     logger.info(f"    Embedding std: {emb_metrics['embedding_std_mean']:.4f}")

Cell In[22], line 210, in compute_embedding_quality_epoch(model, val_data, config, device, num_samples, use_mixed_precision)
    208         _ = actual_model(x, return_moe_losses=False)
    209     else:
--> 210         _ = actual_model(x)
    212 # Get patient embeddings (last valid day)    
    213 patient_embs = extractor.get_patient_embedding(dt_cnt)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1775, in Module._wrapped_call_impl(self, *args, **kwargs)
   1773     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1774 else:
-> 1775     return self._call_impl(*args, **kwargs)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1786, in Module._call_impl(self, *args, **kwargs)
   1781 # If we don't have any hooks, we want to skip the rest of the logic in
   1782 # this function, and just call forward.
   1783 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1784         or _global_backward_pre_hooks or _global_backward_hooks
   1785         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1786     return forward_call(*args, **kwargs)
   1788 result = None
   1789 called_always_called_hooks = set()

Cell In[11], line 134, in BaselineTransformer.forward(self, x)
    131 cd = torch.swapaxes(cd, 0, 1)  # [len_cd, batch*len_dy, embedding_size]
    133 # Apply daily transformer
--> 134 cd = self.transformer_encoder_cd(cd)
    136 # Max pooling across codes dimension
    137 cd = cd.permute(1, 2, 0)  # [batch*len_dy, embedding_size, len_cd]

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1775, in Module._wrapped_call_impl(self, *args, **kwargs)
   1773     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1774 else:
-> 1775     return self._call_impl(*args, **kwargs)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1786, in Module._call_impl(self, *args, **kwargs)
   1781 # If we don't have any hooks, we want to skip the rest of the logic in
   1782 # this function, and just call forward.
   1783 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1784         or _global_backward_pre_hooks or _global_backward_hooks
   1785         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1786     return forward_call(*args, **kwargs)
   1788 result = None
   1789 called_always_called_hooks = set()

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/transformer.py:524, in TransformerEncoder.forward(self, src, mask, src_key_padding_mask, is_causal)
    521 is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
    523 for mod in self.layers:
--> 524     output = mod(
    525         output,
    526         src_mask=mask,
    527         is_causal=is_causal,
    528         src_key_padding_mask=src_key_padding_mask_for_layers,
    529     )
    531 if convert_to_nested:
    532     output = output.to_padded_tensor(0.0, src.size())

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1775, in Module._wrapped_call_impl(self, *args, **kwargs)
   1773     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1774 else:
-> 1775     return self._call_impl(*args, **kwargs)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1786, in Module._call_impl(self, *args, **kwargs)
   1781 # If we don't have any hooks, we want to skip the rest of the logic in
   1782 # this function, and just call forward.
   1783 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1784         or _global_backward_pre_hooks or _global_backward_hooks
   1785         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1786     return forward_call(*args, **kwargs)
   1788 result = None
   1789 called_always_called_hooks = set()

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/transformer.py:935, in TransformerEncoderLayer.forward(self, src, src_mask, src_key_padding_mask, is_causal)
    931     x = x + self._ff_block(self.norm2(x))
    932 else:
    933     x = self.norm1(
    934         x
--> 935         + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
    936     )
    937     x = self.norm2(x + self._ff_block(x))
    939 return x

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/transformer.py:949, in TransformerEncoderLayer._sa_block(self, x, attn_mask, key_padding_mask, is_causal)
    942 def _sa_block(
    943     self,
    944     x: Tensor,
   (...)
    947     is_causal: bool = False,
    948 ) -> Tensor:
--> 949     x = self.self_attn(
    950         x,
    951         x,
    952         x,
    953         attn_mask=attn_mask,
    954         key_padding_mask=key_padding_mask,
    955         need_weights=False,
    956         is_causal=is_causal,
    957     )[0]
    958     return self.dropout1(x)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1775, in Module._wrapped_call_impl(self, *args, **kwargs)
   1773     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1774 else:
-> 1775     return self._call_impl(*args, **kwargs)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1786, in Module._call_impl(self, *args, **kwargs)
   1781 # If we don't have any hooks, we want to skip the rest of the logic in
   1782 # this function, and just call forward.
   1783 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1784         or _global_backward_pre_hooks or _global_backward_hooks
   1785         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1786     return forward_call(*args, **kwargs)
   1788 result = None
   1789 called_always_called_hooks = set()

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/activation.py:1488, in MultiheadAttention.forward(self, query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal)
   1462     attn_output, attn_output_weights = F.multi_head_attention_forward(
   1463         query,
   1464         key,
   (...)
   1485         is_causal=is_causal,
   1486     )
   1487 else:
-> 1488     attn_output, attn_output_weights = F.multi_head_attention_forward(
   1489         query,
   1490         key,
   1491         value,
   1492         self.embed_dim,
   1493         self.num_heads,
   1494         self.in_proj_weight,
   1495         self.in_proj_bias,
   1496         self.bias_k,
   1497         self.bias_v,
   1498         self.add_zero_attn,
   1499         self.dropout,
   1500         self.out_proj.weight,
   1501         self.out_proj.bias,
   1502         training=self.training,
   1503         key_padding_mask=key_padding_mask,
   1504         need_weights=need_weights,
   1505         attn_mask=attn_mask,
   1506         average_attn_weights=average_attn_weights,
   1507         is_causal=is_causal,
   1508     )
   1509 if self.batch_first and is_batched:
   1510     return attn_output.transpose(1, 0), attn_output_weights

File /opt/conda/lib/python3.10/site-packages/torch/nn/functional.py:6307, in multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training, key_padding_mask, need_weights, attn_mask, use_separate_proj_weight, q_proj_weight, k_proj_weight, v_proj_weight, static_k, static_v, average_attn_weights, is_causal)
   6303 if not use_separate_proj_weight:
   6304     assert in_proj_weight is not None, (
   6305         "use_separate_proj_weight is False but in_proj_weight is None"
   6306     )
-> 6307     q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
   6308 else:
   6309     assert q_proj_weight is not None, (
   6310         "use_separate_proj_weight is True but q_proj_weight is None"
   6311     )

File /opt/conda/lib/python3.10/site-packages/torch/nn/functional.py:5706, in _in_projection_packed(q, k, v, w, b)
   5699     proj = linear(q, w, b)
   5700     # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
   5701     proj = (
   5702         proj.unflatten(-1, (3, E))
   5703         .unsqueeze(0)
   5704         .transpose(0, -2)
   5705         .squeeze(-2)
-> 5706         .contiguous()
   5707     )
   5708     return proj[0], proj[1], proj[2]
   5709 else:
   5710     # encoder-decoder attention

OutOfMemoryError: CUDA out of memory. Tried to allocate 5.86 GiB. GPU 0 has a total capacity of 14.57 GiB of which 5.64 GiB is free. Including non-PyTorch memory, this process has 8.92 GiB memory in use. Of the allocated memory 8.37 GiB is allocated by PyTorch, and 358.40 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

---

**Cursor**

The OOM is happening in `compute_embedding_quality_epoch()`, **not during training**! This is a different issue. Let me analyze:

## Root Cause Analysis

**Error location:** `compute_embedding_quality_epoch` → `BaselineTransformer.forward()` → `transformer_encoder_cd`

**The problem:** This function is computing embeddings after each epoch, but:
1. It's likely running with a large batch size (200 samples as shown in the call)
2. It may not be using `torch.no_grad()` properly
3. It's running on a single GPU (not through DataParallel)

Let me look at the function:

## Root Cause Found

The problem is in `compute_embedding_quality_epoch()`:

**Line 5770:** Uses `batch_size=config.batch_size` which may be scaled up for DataParallel (e.g., 128 = 32 × 4 GPUs)

**Line 5802:** Runs on the **unwrapped model** (single GPU), not through DataParallel

**Line 5777:** Has `torch.no_grad()` ✓ but the batch size is too large for a single GPU

**Memory calculation:**
- Batch size = 32-128 samples
- Sequence = 200 days × 80 codes 
- Embedding = 256 dims
- Daily attention needs: `batch × len_dy × len_cd × len_cd × heads` for attention scores
- For batch=64, len_dy=200, len_cd=80: this is ~5-6GB just for attention!

---

## The Fix

Replace the `compute_embedding_quality_epoch` function with a memory-safe version:

```python
def compute_embedding_quality_epoch(
    model: nn.Module,
    val_data: pd.DataFrame,
    config: BaseConfig,
    device: torch.device,
    num_samples: int = 200,
    use_mixed_precision: bool = False
) -> Dict[str, float]:
    """
    Evaluate embedding quality at epoch end (memory-safe version).
    
    IMPORTANT: Uses small batch size regardless of training batch size
    to avoid OOM when running on unwrapped model.
    """
    # Unwrap model for direct access
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    if isinstance(actual_model, DataParallelWrapper):
        actual_model = actual_model.model
    actual_model.eval()
    metrics = {}
    
    # CRITICAL: Clear GPU cache before embedding extraction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Sample validation data
    sample_size = min(num_samples, len(val_data))
    val_sample = val_data.sample(sample_size, random_state=42)
    val_dataset = ClinicalDataset(val_sample, config)
    
    # CRITICAL FIX: Use SMALL batch size regardless of config
    # This runs on single GPU (unwrapped model), not DataParallel
    embedding_batch_size = min(8, config.batch_size)  # Max 8 samples at a time
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=embedding_batch_size,  # ← FIXED: Use small batch
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=0  # Avoid memory from worker processes
    )
    
    all_embeddings = []
    all_targets = []
    
    with EmbeddingExtractor(actual_model) as extractor:
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                age = batch['age'].to(device)
                gender = batch['gender'].to(device)
                codes = batch['codes'].to(device)
                lob = batch['lob'].to(device)
                dt_cnt = batch['dt_cnt']
                y = batch['target']

                x = torch.cat([
                    age.unsqueeze(-1),
                    gender.unsqueeze(-1),
                    lob.unsqueeze(-1),
                    codes
                ], dim=-1)

                # Forward pass with mixed precision for memory efficiency
                try:
                    if use_mixed_precision:
                        dtype = getattr(config, 'dtype', torch.float16)
                        with torch.cuda.amp.autocast(dtype=dtype):
                            if _model_has_moe(actual_model):
                                _ = actual_model(x, return_moe_losses=False)
                            else:
                                _ = actual_model(x)
                    else:
                        if _model_has_moe(actual_model):
                            _ = actual_model(x, return_moe_losses=False)
                        else:
                            _ = actual_model(x)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️ OOM in embedding extraction at batch {batch_idx}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    raise

                # Get patient embeddings
                patient_embs = extractor.get_patient_embedding(dt_cnt)

                # Collect embeddings and targets
                for j in range(len(dt_cnt)):
                    if dt_cnt[j] > 0:
                        all_embeddings.append(patient_embs[j].cpu())
                        
                        patient_targets = y[j]
                        all_codes = set()
                        for day_codes in patient_targets:
                            all_codes.update([c for c in day_codes if c > 0])
                        all_targets.append(all_codes)
                
                # Clean up batch tensors
                del x, age, gender, codes, lob, patient_embs
                
                # Periodic cache clear for long loops
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    if len(all_embeddings) == 0:
        return {'embedding_std_mean': 0.0, 'nn_target_overlap': 0.0}
    
    embeddings_tensor = torch.stack(all_embeddings)
    
    # 1. Embedding Space Utilization
    emb_std = embeddings_tensor.std(dim=0)
    metrics['embedding_std_mean'] = emb_std.mean().item()
    
    if metrics['embedding_std_mean'] < 0.01:
        print(f"⚠️ WARNING: Embeddings collapsing! std_mean={metrics['embedding_std_mean']:.4f}")
    
    # 2. Nearest Neighbor Target Overlap
    dists = torch.cdist(embeddings_tensor, embeddings_tensor)
    
    nn_accuracies = []
    for i in range(min(100, len(embeddings_tensor))):
        _, indices = torch.topk(dists[i], k=6, largest=False)
        neighbors = indices[1:6].tolist()
        
        my_targets = all_targets[i]
        if len(my_targets) > 0:
            overlaps = []
            for nb_idx in neighbors:
                nb_targets = all_targets[nb_idx]
                if len(nb_targets) > 0:
                    overlap = len(my_targets & nb_targets) / len(my_targets | nb_targets)
                    overlaps.append(overlap)
            if overlaps:
                nn_accuracies.append(np.mean(overlaps))
    
    metrics['nn_target_overlap'] = np.mean(nn_accuracies) if nn_accuracies else 0.0
    
    return metrics
```

---

## Key Changes

| Issue | Original | Fixed |
|-------|----------|-------|
| **Batch size** | `config.batch_size` (could be 64-128) | `min(8, config.batch_size)` (max 8) |
| **Memory cleanup** | None | `torch.cuda.empty_cache()` before and during |
| **OOM handling** | Crash | Try/except with skip and continue |
| **Worker processes** | Default | `num_workers=0` to reduce memory |
| **Batch cleanup** | None | `del` tensors after each batch |

---

## Alternative: Skip Embedding Quality Check

If you want a quick workaround, you can skip this check entirely by modifying the call in `run_single_experiment`:

```python
# In run_single_experiment, around line 214-221:
if epoch % check_embeddings_every == 0:
    logger.info("Computing embedding quality...")
    try:
        torch.cuda.empty_cache()
        emb_metrics = compute_embedding_quality_epoch(
            model, val_data, config, device,
            num_samples=100,  # Reduce from 200 to 100
            use_mixed_precision=True  # Force mixed precision
        )
        val_metrics.update(emb_metrics)
        logger.info(f"    Embedding std: {emb_metrics['embedding_std_mean']:.4f}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("⚠️ Skipping embedding quality check due to OOM")
            torch.cuda.empty_cache()
        else:
            raise
```

```python
def compute_embedding_quality_epoch(
    model: nn.Module,
    val_data: pd.DataFrame,
    config: BaseConfig,
    device: torch.device,
    num_samples: int = 200,
    use_mixed_precision: bool = False
) -> Dict[str, float]:
    """
    Evaluate embedding quality at epoch end (memory-safe version).
    
    IMPORTANT: Uses small batch size regardless of training batch size
    to avoid OOM when running on unwrapped model.
    """
    # Unwrap model for direct access
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    if isinstance(actual_model, DataParallelWrapper):
        actual_model = actual_model.model
    actual_model.eval()
    metrics = {}
    
    # CRITICAL: Clear GPU cache before embedding extraction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Sample validation data
    sample_size = min(num_samples, len(val_data))
    val_sample = val_data.sample(sample_size, random_state=42)
    val_dataset = ClinicalDataset(val_sample, config)
    
    # CRITICAL FIX: Use SMALL batch size regardless of config
    # This runs on single GPU (unwrapped model), not DataParallel
    embedding_batch_size = min(8, config.batch_size)  # Max 8 samples at a time
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=embedding_batch_size,  # ← FIXED: Use small batch
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=0  # Avoid memory from worker processes
    )
    
    all_embeddings = []
    all_targets = []
    
    with EmbeddingExtractor(actual_model) as extractor:
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                age = batch['age'].to(device)
                gender = batch['gender'].to(device)
                codes = batch['codes'].to(device)
                lob = batch['lob'].to(device)
                dt_cnt = batch['dt_cnt']
                y = batch['target']

                x = torch.cat([
                    age.unsqueeze(-1),
                    gender.unsqueeze(-1),
                    lob.unsqueeze(-1),
                    codes
                ], dim=-1)

                # Forward pass with mixed precision for memory efficiency
                try:
                    if use_mixed_precision:
                        dtype = getattr(config, 'dtype', torch.float16)
                        with torch.cuda.amp.autocast(dtype=dtype):
                            if _model_has_moe(actual_model):
                                _ = actual_model(x, return_moe_losses=False)
                            else:
                                _ = actual_model(x)
                    else:
                        if _model_has_moe(actual_model):
                            _ = actual_model(x, return_moe_losses=False)
                        else:
                            _ = actual_model(x)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️ OOM in embedding extraction at batch {batch_idx}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    raise

                # Get patient embeddings
                patient_embs = extractor.get_patient_embedding(dt_cnt)

                # Collect embeddings and targets
                for j in range(len(dt_cnt)):
                    if dt_cnt[j] > 0:
                        all_embeddings.append(patient_embs[j].cpu())
                        
                        patient_targets = y[j]
                        all_codes = set()
                        for day_codes in patient_targets:
                            all_codes.update([c for c in day_codes if c > 0])
                        all_targets.append(all_codes)
                
                # Clean up batch tensors
                del x, age, gender, codes, lob, patient_embs
                
                # Periodic cache clear for long loops
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    if len(all_embeddings) == 0:
        return {'embedding_std_mean': 0.0, 'nn_target_overlap': 0.0}
    
    embeddings_tensor = torch.stack(all_embeddings)
    
    # 1. Embedding Space Utilization
    emb_std = embeddings_tensor.std(dim=0)
    metrics['embedding_std_mean'] = emb_std.mean().item()
    
    if metrics['embedding_std_mean'] < 0.01:
        print(f"⚠️ WARNING: Embeddings collapsing! std_mean={metrics['embedding_std_mean']:.4f}")
    
    # 2. Nearest Neighbor Target Overlap
    dists = torch.cdist(embeddings_tensor, embeddings_tensor)
    
    nn_accuracies = []
    for i in range(min(100, len(embeddings_tensor))):
        _, indices = torch.topk(dists[i], k=6, largest=False)
        neighbors = indices[1:6].tolist()
        
        my_targets = all_targets[i]
        if len(my_targets) > 0:
            overlaps = []
            for nb_idx in neighbors:
                nb_targets = all_targets[nb_idx]
                if len(nb_targets) > 0:
                    overlap = len(my_targets & nb_targets) / len(my_targets | nb_targets)
                    overlaps.append(overlap)
            if overlaps:
                nn_accuracies.append(np.mean(overlaps))
    
    metrics['nn_target_overlap'] = np.mean(nn_accuracies) if nn_accuracies else 0.0
    
    return metrics
```

```python
# In run_single_experiment, around line 214-221:
if epoch % check_embeddings_every == 0:
    logger.info("Computing embedding quality...")
    try:
        torch.cuda.empty_cache()
        emb_metrics = compute_embedding_quality_epoch(
            model, val_data, config, device,
            num_samples=100,  # Reduce from 200 to 100
            use_mixed_precision=True  # Force mixed precision
        )
        val_metrics.update(emb_metrics)
        logger.info(f"    Embedding std: {emb_metrics['embedding_std_mean']:.4f}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("⚠️ Skipping embedding quality check due to OOM")
            torch.cuda.empty_cache()
        else:
            raise
```

---

**User**

You are an expert in Debugging transformer pretraining code. Here is the issues I got when I ran the code @moe_flashattn_3.py (13084-13120); inspect and examine the model architecture, training strategies, configurations, forward and backpropagation procedure based on teh followng context and observations
1) the GCP vertex AI workbench instance kernel died during the training; it's been twice. one died without the data parallel wrapper and one with the data parallel wrapper; I have checked the disk storage it is good, there is no OOM issues; this is my machine type 
n1-highmem-64 (64 vCPUs, 416 GB RAM), and I use 4 Nvidia T4 GPU
2) the training loss and other metrics progresses way slower than I do with single GPU the version without the dataparallelism wrapper; I am not sure what is the root cause; examine the training and optimization configurations and methods; and check if they fit the current experiment 2 arhcitecutre nad model configurations; 
indepth and comprehsenvie analysis is needed before reaching any conclusions
Here is the issue 
07:54:40 - exp2b_flash_learned_pool - INFO - Bucketing is ENABLED via BatchSampler.
Pre-processing complete.
07:54:44 - exp2b_flash_learned_pool - INFO - Using DataLoader with 32 workers.
07:54:44 - exp2b_flash_learned_pool - INFO - Training for 1 epochs...
07:54:44 - exp2b_flash_learned_pool - INFO - 
--- Epoch 1/1 ---
  Batch 0/12335

🔍 GPU UTILIZATION CHECK (Batch 0):
   GPU 0: 0.51 GB allocated, 1.02 GB reserved
   GPU 1: 0.02 GB allocated, 0.04 GB reserved
   GPU 2: 0.02 GB allocated, 0.04 GB reserved
   GPU 3: 0.02 GB allocated, 0.04 GB reserved
    Loss: 0.8035 | R@10: 0.001 | R@20: 0.005 | P@10: 0.000 | P@20: 0.000 | mAP20: 0.109 | mAP50: 0.051 | Brier: 0.2927
    GPU 0: 0.91GB / 8.16GB peak
    GPU 1: 0.02GB / 6.74GB peak
    GPU 2: 0.02GB / 6.72GB peak
    GPU 3: 0.02GB / 6.75GB peak

🔍 Detailed GPU tracking for batch 2

======================================================================
GPU MEMORY SUMMARY
======================================================================
GPU   1_after_data_to_gpu 2_after_forward     3_after_backward    
----------------------------------------------------------------------
GPU 0   1.22GB               2.71GB               1.22GB             
GPU 1   0.02GB               1.53GB               0.02GB             
GPU 2   0.02GB               1.53GB               0.02GB             
GPU 3   0.02GB               1.53GB               0.02GB             
======================================================================

🔍 Detailed GPU tracking for batch 50

======================================================================
GPU MEMORY SUMMARY
======================================================================
GPU   1_after_data_to_gpu 2_after_forward     3_after_backward    
----------------------------------------------------------------------
GPU 0   1.41GB               2.90GB               1.41GB             
GPU 1   0.02GB               1.53GB               0.02GB             
GPU 2   0.02GB               1.53GB               0.02GB             
GPU 3   0.02GB               1.53GB               0.02GB             
======================================================================

🔍 Detailed GPU tracking for batch 100
  Batch 100/12335

======================================================================
GPU MEMORY SUMMARY
======================================================================
GPU   1_after_data_to_gpu 2_after_forward     3_after_backward    
----------------------------------------------------------------------
GPU 0   1.32GB               3.17GB               1.71GB             
GPU 1   0.02GB               1.60GB               0.02GB             
GPU 2   0.02GB               1.59GB               0.02GB             
GPU 3   0.02GB               1.59GB               0.02GB             
======================================================================
    Loss: 0.5100 | R@10: 0.004 | R@20: 0.009 | P@10: 0.000 | P@20: 0.000 | mAP20: 0.142 | mAP50: 0.079 | Brier: 0.1678
  Batch 200/12335
    Loss: 0.3384 | R@10: 0.004 | R@20: 0.010 | P@10: 0.000 | P@20: 0.000 | mAP20: 0.149 | mAP50: 0.074 | Brier: 0.0954
  Batch 300/12335
    Loss: 0.2171 | R@10: 0.004 | R@20: 0.014 | P@10: 0.000 | P@20: 0.001 | mAP20: 0.086 | mAP50: 0.052 | Brier: 0.0486
  Batch 400/12335
    Loss: 0.1504 | R@10: 0.004 | R@20: 0.011 | P@10: 0.000 | P@20: 0.001 | mAP20: 0.106 | mAP50: 0.056 | Brier: 0.0265
  Batch 500/12335
    Loss: 0.1067 | R@10: 0.005 | R@20: 0.026 | P@10: 0.001 | P@20: 0.001 | mAP20: 0.084 | mAP50: 0.046 | Brier: 0.0146
  Batch 600/12335
    Loss: 0.0814 | R@10: 0.038 | R@20: 0.084 | P@10: 0.004 | P@20: 0.004 | mAP20: 0.127 | mAP50: 0.072 | Brier: 0.0090
  Batch 700/12335
    Loss: 0.0630 | R@10: 0.165 | R@20: 0.237 | P@10: 0.017 | P@20: 0.013 | mAP20: 0.222 | mAP50: 0.167 | Brier: 0.0057
  Batch 800/12335
    Loss: 0.0517 | R@10: 0.281 | R@20: 0.324 | P@10: 0.033 | P@20: 0.021 | mAP20: 0.300 | mAP50: 0.242 | Brier: 0.0040
  Batch 900/12335
    Loss: 0.0427 | R@10: 0.275 | R@20: 0.311 | P@10: 0.037 | P@20: 0.024 | mAP20: 0.343 | mAP50: 0.274 | Brier: 0.0029
  Batch 1000/12335
    Loss: 0.0366 | R@10: 0.348 | R@20: 0.371 | P@10: 0.050 | P@20: 0.029 | mAP20: 0.404 | mAP50: 0.356 | Brier: 0.0023
    GPU 0: 1.10GB / 3.63GB peak
    GPU 1: 0.02GB / 2.41GB peak
    GPU 2: 0.02GB / 2.41GB peak
    GPU 3: 0.02GB / 2.42GB peak
  Batch 1100/12335
    Loss: 0.0321 | R@10: 0.413 | R@20: 0.440 | P@10: 0.070 | P@20: 0.041 | mAP20: 0.370 | mAP50: 0.329 | Brier: 0.0021
  Batch 1200/12335
    Loss: 0.0275 | R@10: 0.351 | R@20: 0.371 | P@10: 0.069 | P@20: 0.038 | mAP20: 0.420 | mAP50: 0.377 | Brier: 0.0016
  Batch 1300/12335
    Loss: 0.0246 | R@10: 0.424 | R@20: 0.448 | P@10: 0.079 | P@20: 0.043 | mAP20: 0.437 | mAP50: 0.392 | Brier: 0.0014
  Batch 1400/12335
    Loss: 0.0228 | R@10: 0.426 | R@20: 0.448 | P@10: 0.087 | P@20: 0.047 | mAP20: 0.395 | mAP50: 0.351 | Brier: 0.0014
  Batch 1500/12335
    Loss: 0.0199 | R@10: 0.420 | R@20: 0.456 | P@10: 0.082 | P@20: 0.045 | mAP20: 0.431 | mAP50: 0.376 | Brier: 0.0011
  Batch 1600/12335
    Loss: 0.0181 | R@10: 0.410 | R@20: 0.447 | P@10: 0.077 | P@20: 0.043 | mAP20: 0.413 | mAP50: 0.351 | Brier: 0.0010
  Batch 1700/12335
    Loss: 0.0166 | R@10: 0.436 | R@20: 0.476 | P@10: 0.096 | P@20: 0.053 | mAP20: 0.407 | mAP50: 0.345 | Brier: 0.0010
  Batch 1800/12335
    Loss: 0.0155 | R@10: 0.455 | R@20: 0.505 | P@10: 0.093 | P@20: 0.053 | mAP20: 0.396 | mAP50: 0.338 | Brier: 0.0009
  Batch 1900/12335
    Loss: 0.0145 | R@10: 0.458 | R@20: 0.508 | P@10: 0.094 | P@20: 0.055 | mAP20: 0.392 | mAP50: 0.328 | Brier: 0.0009
  Batch 2000/12335
    Loss: 0.0136 | R@10: 0.477 | R@20: 0.537 | P@10: 0.108 | P@20: 0.063 | mAP20: 0.377 | mAP50: 0.323 | Brier: 0.0009
    GPU 0: 1.10GB / 5.83GB peak
    GPU 1: 0.02GB / 2.42GB peak
    GPU 2: 0.02GB / 2.42GB peak
    GPU 3: 0.02GB / 2.42GB peak
  Batch 2100/12335
    Loss: 0.0128 | R@10: 0.495 | R@20: 0.574 | P@10: 0.099 | P@20: 0.061 | mAP20: 0.356 | mAP50: 0.302 | Brier: 0.0009
  Batch 2200/12335
    Loss: 0.0122 | R@10: 0.508 | R@20: 0.582 | P@10: 0.103 | P@20: 0.063 | mAP20: 0.358 | mAP50: 0.301 | Brier: 0.0009
  Batch 2300/12335
    Loss: 0.0112 | R@10: 0.475 | R@20: 0.562 | P@10: 0.109 | P@20: 0.068 | mAP20: 0.343 | mAP50: 0.281 | Brier: 0.0008
  Batch 2400/12335
    Loss: 0.0110 | R@10: 0.553 | R@20: 0.639 | P@10: 0.110 | P@20: 0.070 | mAP20: 0.357 | mAP50: 0.297 | Brier: 0.0009
  Batch 2500/12335
    Loss: 0.0105 | R@10: 0.536 | R@20: 0.643 | P@10: 0.106 | P@20: 0.070 | mAP20: 0.340 | mAP50: 0.275 | Brier: 0.0008
  Batch 2600/12335
    Loss: 0.0097 | R@10: 0.463 | R@20: 0.579 | P@10: 0.102 | P@20: 0.067 | mAP20: 0.317 | mAP50: 0.252 | Brier: 0.0008
  Batch 2700/12335
    Loss: 0.0093 | R@10: 0.478 | R@20: 0.594 | P@10: 0.108 | P@20: 0.071 | mAP20: 0.322 | mAP50: 0.256 | Brier: 0.0008
  Batch 2800/12335
    Loss: 0.0092 | R@10: 0.526 | R@20: 0.633 | P@10: 0.103 | P@20: 0.069 | mAP20: 0.330 | mAP50: 0.262 | Brier: 0.0008
  Batch 2900/12335
    Loss: 0.0087 | R@10: 0.502 | R@20: 0.623 | P@10: 0.104 | P@20: 0.072 | mAP20: 0.299 | mAP50: 0.234 | Brier: 0.0008
  Batch 3000/12335
    Loss: 0.0085 | R@10: 0.482 | R@20: 0.601 | P@10: 0.104 | P@20: 0.070 | mAP20: 0.326 | mAP50: 0.249 | Brier: 0.0008
    GPU 0: 1.10GB / 5.83GB peak
    GPU 1: 0.02GB / 2.42GB peak
    GPU 2: 0.02GB / 2.42GB peak
    GPU 3: 0.02GB / 2.42GB peak
  Batch 3100/12335
    Loss: 0.0082 | R@10: 0.499 | R@20: 0.591 | P@10: 0.101 | P@20: 0.067 | mAP20: 0.345 | mAP50: 0.256 | Brier: 0.0008
  Batch 3200/12335
    Loss: 0.0080 | R@10: 0.493 | R@20: 0.606 | P@10: 0.112 | P@20: 0.075 | mAP20: 0.321 | mAP50: 0.238 | Brier: 0.0008
  Batch 3300/12335
    Loss: 0.0078 | R@10: 0.525 | R@20: 0.629 | P@10: 0.110 | P@20: 0.073 | mAP20: 0.346 | mAP50: 0.253 | Brier: 0.0008
  Batch 3400/12335
    Loss: 0.0075 | R@10: 0.513 | R@20: 0.628 | P@10: 0.102 | P@20: 0.069 | mAP20: 0.337 | mAP50: 0.238 | Brier: 0.0007
  Batch 3500/12335
    Loss: 0.0071 | R@10: 0.460 | R@20: 0.573 | P@10: 0.100 | P@20: 0.068 | mAP20: 0.322 | mAP50: 0.228 | Brier: 0.0007
  Batch 3600/12335
    Loss: 0.0070 | R@10: 0.486 | R@20: 0.600 | P@10: 0.107 | P@20: 0.073 | mAP20: 0.317 | mAP50: 0.226 | Brier: 0.0007
  Batch 3700/12335
    Loss: 0.0070 | R@10: 0.510 | R@20: 0.642 | P@10: 0.105 | P@20: 0.073 | mAP20: 0.320 | mAP50: 0.224 | Brier: 0.0008
  Batch 3800/12335
    Loss: 0.0071 | R@10: 0.517 | R@20: 0.634 | P@10: 0.109 | P@20: 0.075 | mAP20: 0.330 | mAP50: 0.231 | Brier: 0.0008
  Batch 3900/12335
    Loss: 0.0077 | R@10: 0.541 | R@20: 0.652 | P@10: 0.118 | P@20: 0.080 | mAP20: 0.312 | mAP50: 0.223 | Brier: 0.0009
  Batch 4000/12335
    Loss: 0.0065 | R@10: 0.487 | R@20: 0.617 | P@10: 0.102 | P@20: 0.070 | mAP20: 0.316 | mAP50: 0.223 | Brier: 0.0007
    GPU 0: 1.10GB / 5.83GB peak
    GPU 1: 0.02GB / 2.42GB peak
    GPU 2: 0.02GB / 2.42GB peak
    GPU 3: 0.02GB / 2.42GB peak
  Batch 4100/12335
    Loss: 0.0064 | R@10: 0.514 | R@20: 0.631 | P@10: 0.108 | P@20: 0.075 | mAP20: 0.324 | mAP50: 0.225 | Brier: 0.0007
  Batch 4200/12335
    Loss: 0.0062 | R@10: 0.490 | R@20: 0.624 | P@10: 0.095 | P@20: 0.067 | mAP20: 0.303 | mAP50: 0.212 | Brier: 0.0007
  Batch 4300/12335
    Loss: 0.0063 | R@10: 0.551 | R@20: 0.674 | P@10: 0.119 | P@20: 0.080 | mAP20: 0.341 | mAP50: 0.240 | Brier: 0.0008
  Batch 4400/12335
    Loss: 0.0059 | R@10: 0.467 | R@20: 0.595 | P@10: 0.104 | P@20: 0.072 | mAP20: 0.312 | mAP50: 0.211 | Brier: 0.0007
  Batch 4500/12335
    Loss: 0.0060 | R@10: 0.506 | R@20: 0.633 | P@10: 0.101 | P@20: 0.072 | mAP20: 0.308 | mAP50: 0.214 | Brier: 0.0007
  Batch 4600/12335
    Loss: 0.0057 | R@10: 0.489 | R@20: 0.610 | P@10: 0.103 | P@20: 0.070 | mAP20: 0.317 | mAP50: 0.218 | Brier: 0.0007
  Batch 4700/12335
    Loss: 0.0059 | R@10: 0.552 | R@20: 0.651 | P@10: 0.106 | P@20: 0.073 | mAP20: 0.335 | mAP50: 0.229 | Brier: 0.0008
  Batch 4800/12335
    Loss: 0.0057 | R@10: 0.493 | R@20: 0.611 | P@10: 0.114 | P@20: 0.076 | mAP20: 0.327 | mAP50: 0.232 | Brier: 0.0007
  Batch 4900/12335
    Loss: 0.0057 | R@10: 0.465 | R@20: 0.595 | P@10: 0.108 | P@20: 0.074 | mAP20: 0.314 | mAP50: 0.215 | Brier: 0.0007
  Batch 5000/12335
    Loss: 0.0054 | R@10: 0.504 | R@20: 0.626 | P@10: 0.114 | P@20: 0.077 | mAP20: 0.328 | mAP50: 0.231 | Brier: 0.0007
    GPU 0: 1.10GB / 5.83GB peak
    GPU 1: 0.02GB / 2.42GB peak
    GPU 2: 0.02GB / 2.42GB peak
    GPU 3: 0.02GB / 2.42GB peak
  Batch 5100/12335
    Loss: 0.0055 | R@10: 0.534 | R@20: 0.645 | P@10: 0.106 | P@20: 0.072 | mAP20: 0.336 | mAP50: 0.226 | Brier: 0.0007
  Batch 5200/12335
    Loss: 0.0054 | R@10: 0.501 | R@20: 0.625 | P@10: 0.119 | P@20: 0.080 | mAP20: 0.336 | mAP50: 0.231 | Brier: 0.0007
  Batch 5300/12335
    Loss: 0.0055 | R@10: 0.477 | R@20: 0.600 | P@10: 0.107 | P@20: 0.074 | mAP20: 0.320 | mAP50: 0.219 | Brier: 0.0007
  Batch 5400/12335
    Loss: 0.0052 | R@10: 0.473 | R@20: 0.587 | P@10: 0.105 | P@20: 0.072 | mAP20: 0.322 | mAP50: 0.220 | Brier: 0.0007
  Batch 5500/12335
    Loss: 0.0054 | R@10: 0.558 | R@20: 0.672 | P@10: 0.114 | P@20: 0.077 | mAP20: 0.343 | mAP50: 0.241 | Brier: 0.0007
  Batch 5600/12335
    Loss: 0.0054 | R@10: 0.526 | R@20: 0.633 | P@10: 0.116 | P@20: 0.077 | mAP20: 0.350 | mAP50: 0.243 | Brier: 0.0008
  Batch 5700/12335
    Loss: 0.0050 | R@10: 0.467 | R@20: 0.582 | P@10: 0.101 | P@20: 0.069 | mAP20: 0.316 | mAP50: 0.217 | Brier: 0.0007
  Batch 5800/12335
    Loss: 0.0053 | R@10: 0.513 | R@20: 0.636 | P@10: 0.109 | P@20: 0.075 | mAP20: 0.328 | mAP50: 0.235 | Brier: 0.0007
  Batch 5900/12335
    Loss: 0.0050 | R@10: 0.493 | R@20: 0.609 | P@10: 0.097 | P@20: 0.066 | mAP20: 0.334 | mAP50: 0.224 | Brier: 0.0007

---

**Cursor**

This is a complex issue with multiple symptoms. Let me do a comprehensive analysis.

# Comprehensive Analysis: Kernel Death & Slow Training

Based on my inspection, I've identified **7 critical issues** causing both the kernel death and slow training.

---

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

```plaintext
Using DataLoader with 32 workers.
```

```python
n_workers = max(1, os.cpu_count() // max(world_size, 1) // 2)
# = 64 // 1 // 2 = 32 workers
```

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

```python
# Line 9246
scaled_lr = base_lr * math.sqrt(num_gpus)  # 1e-4 * sqrt(4) = 2e-4
```

```python
# If using accumulation_steps=1 (recommended):
if use_data_parallel:
    # Linear scaling with warmup is standard for DataParallel
    scaled_lr = base_lr * num_gpus  # 1e-4 * 4 = 4e-4
```

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

```python
# Calculate total optimizer steps
total_steps = len(train_loader) // accumulation_steps * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

```python
# In training loop, step scheduler ONCE per epoch, not per batch
# Move scheduler.step() outside the batch loop
```

```plaintext
After forward:   GPU 0: 2.71GB    GPUs 1-3: 1.53GB each
After backward:  GPU 0: 1.22GB    GPUs 1-3: 0.02GB each
```

```python
x = x.cuda(non_blocking=True)  # Data goes to GPU 0 first
```

```python
# Line 9262
config.batch_size = effective_batch_size  # 32 * 4 = 128

# Then at line 9287
train_loader, val_loader = _create_dataloaders(train_data, val_data, config, ...)
```

```plaintext
GPU 0: 1.22GB → 2.71GB → 1.22GB   (batch 2)
GPU 0: 1.32GB → 3.17GB → 1.71GB   (batch 100)  ← Growing!
GPU 0: 1.41GB → 2.90GB → 1.41GB   (batch 50)   
GPU 0: 1.10GB / 5.83GB peak       (batch 2000)
```

```python
# In _create_dataloaders:
n_workers = min(4, os.cpu_count() // 4)  # Was: 32
```

```python
# In run_single_experiment call to train_epoch:
accumulation_steps=1
```

```python
# In run_single_experiment:
if use_data_parallel:
    scaled_lr = base_lr * num_gpus  # Linear scaling: 4e-4 for 4 GPUs
```

```python
# Use epoch-based stepping instead of step-based:
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# Then move scheduler.step() to END of epoch (after train_epoch returns)
```

---

**User**

Follow up questions need clarificaiton and elaboration 
1) for the scheduler, what are teh difference between two solutions you proposed? pros and cons? and also 
3) Memory Leak Suspicion, how to resolve it in a protectie manner to control the peak memory to avoid OOM error? is tghere any such risk?

---

**Cursor**

# Follow-up Clarifications

## 1. Scheduler Options: Step-Based vs Epoch-Based

### Option A: Step-Based Scheduler (Current Behavior)

```python
# Scheduler is stepped EVERY optimizer step (every batch or every N batches with accumulation)
total_steps = len(train_loader) // accumulation_steps * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

# Inside train_epoch:
if accumulation_counter >= accumulation_steps:
    optimizer.step()
    scheduler.step()  # ← Called per optimizer step
```

**How it works:**
```
Epoch 1: LR decays smoothly from max → ~0 across all 12335 batches
         Step 0: LR=4e-4, Step 3000: LR=3e-4, Step 6000: LR=2e-4, Step 12335: LR≈0
```

**Pros:**
- Smoother LR decay (more granular control)
- Better for long epochs with many batches
- Industry standard for large-scale training (GPT, BERT)
- Can implement warmup more easily

**Cons:**
- Requires knowing total steps upfront
- Sensitive to changes in batch size or accumulation steps
- If training is interrupted, resume needs to restore exact step count

---

### Option B: Epoch-Based Scheduler (Simpler)

```python
# Scheduler is stepped ONCE per epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# In run_single_experiment, OUTSIDE the batch loop:
for epoch in range(epochs):
    train_metrics = train_epoch(...)  # No scheduler.step() inside
    scheduler.step()  # ← Called once per epoch
```

**How it works:**
```
Epoch 1: LR = 4e-4 (constant throughout epoch)
Epoch 2: LR = 2.8e-4 (cosine decay step)
Epoch 3: LR = 1.2e-4
Epoch 4: LR = 0.4e-4
```

**Pros:**
- Simple to implement and understand
- Robust to batch size changes
- Easy to resume training (just need epoch number)
- Works well for moderate epochs (5-20)

**Cons:**
- Less granular (LR stays constant within each epoch)
- For 1-epoch training, LR only changes once (useless)
- Can cause learning instability at epoch boundaries

---

### Recommendation for Your Case

Given:
- You're running 1 epoch initially (testing)
- 12,335 batches per epoch (large)
- Using gradient accumulation

**My recommendation: Use Step-Based with warmup**

```python
# In run_single_experiment, after creating optimizer:

# Calculate steps
steps_per_epoch = len(train_loader) // accumulation_steps
total_steps = steps_per_epoch * epochs
warmup_steps = min(1000, total_steps // 10)  # 10% warmup or 1000 steps

# Create scheduler with warmup
def get_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine decay - industry standard for transformers."""
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
```

**Why warmup matters:**
- First batches have random gradients (high variance)
- Large LR + random gradients = unstable training
- Warmup lets the model "settle" before aggressive learning

---

## 2. Memory Leak Resolution (Protective Measures)

### Identified Risk Areas

From your logs, peak memory grew from 3.17GB → 5.83GB over training. Here are the potential causes and protective measures:

---

### Risk 1: Logging Tensors Without `.item()` or `.detach()`

**Problem:**
```python
# BAD: Keeps computation graph in memory
total_pred_loss += pred_loss  # pred_loss is a tensor with graph

# GOOD: Breaks the graph
total_pred_loss += pred_loss.item()  # Scalar, no graph
# or
total_pred_loss += pred_loss.detach()  # Tensor without graph
```

**Fix - Audit all loss tracking:**
```python
# In train_epoch, around line 4404:
pred_loss_scalar = pred_loss.detach().item()  # Use detach() before item()
aux_loss_scalar = aux_loss.detach().item()

total_pred_loss += pred_loss_scalar
total_aux_loss += aux_loss_scalar
```

---

### Risk 2: Metrics Buffer Growing Unboundedly

**Problem:**
```python
batch_metrics_buffer = []  # Grows every log_interval batches
moe_metrics_buffer = []    # Grows every log_interval batches

# With 12335 batches and log_interval=100:
# = 123 entries × dict of floats ≈ small, but each dict may hold tensor refs
```

**Fix - Limit buffer size:**
```python
# Add at top of train_epoch:
MAX_METRICS_BUFFER = 100  # Keep last 100 logged batches

# When appending:
batch_metrics_buffer.append(batch_metrics)
if len(batch_metrics_buffer) > MAX_METRICS_BUFFER:
    batch_metrics_buffer.pop(0)  # Remove oldest
```

---

### Risk 3: MoE Expert Usage Tensor Accumulation

**Problem:**
```python
if 'expert_usage' in moe_losses:
    expert_usage_list.append(moe_losses['expert_usage'])  # Keeps tensor references
```

**Fix - Detach MoE metrics:**
```python
if 'expert_usage' in moe_losses:
    expert_usage_list.append(moe_losses['expert_usage'].detach().cpu())
```

---

### Risk 4: Incomplete Deletion in Cleanup

**Problem:**
```python
# Current cleanup (line ~4449)
del x, targets_mh, dt_cnt
if output is not None:
    del output
del pred_loss, total_loss, scaled_loss
```

**Missing:** `extras` dictionary which may hold tensor references.

**Fix - Comprehensive cleanup:**
```python
# After backward and before next batch:
del x, targets_mh, dt_cnt
if 'extras' in dir() and extras is not None:
    del extras
if 'output' in dir() and output is not None:
    del output
if 'moe_losses' in dir() and moe_losses:
    del moe_losses
del pred_loss, aux_loss, total_loss, scaled_loss, result
```

---

### Protective Memory Management System

Add this comprehensive memory protection to your training loop:

```python
def train_epoch(
    ...
    memory_warning_threshold_gb: float = 12.0,  # Warn at 12GB (T4 has 16GB)
    memory_critical_threshold_gb: float = 14.0,  # Critical at 14GB
) -> Dict[str, float]:
    """Train with memory protection."""
    
    # ... existing setup ...
    
    for batch_idx, batch in enumerate(dataloader):
        
        # ================================================================
        # MEMORY PROTECTION: Check before each batch
        # ================================================================
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            current_memory_gb = torch.cuda.memory_allocated() / 1024**3
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            
            # Warning level
            if peak_memory_gb > memory_warning_threshold_gb:
                if is_main:
                    print(f"⚠️ Memory warning: {peak_memory_gb:.2f}GB peak")
                # Aggressive cleanup
                gc.collect()
                torch.cuda.empty_cache()
            
            # Critical level - reduce batch processing
            if current_memory_gb > memory_critical_threshold_gb:
                if is_main:
                    print(f"🔴 Memory critical: {current_memory_gb:.2f}GB - forcing cleanup")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Skip metrics computation for this batch to save memory
                need_predictions = False
        
        # ... forward pass ...
        
        # ================================================================
        # AGGRESSIVE CLEANUP after backward
        # ================================================================
        # Delete all batch tensors
        del x, targets_mh, dt_cnt, age, gender, lob, codes
        
        # Delete forward outputs
        if 'result' in dir():
            del result
        if 'extras' in dir() and extras is not None:
            if 'predictions' in extras:
                del extras['predictions']
            if 'moe_losses' in extras:
                del extras['moe_losses']
            del extras
        if 'output' in dir() and output is not None:
            del output
        if 'moe_losses' in dir():
            del moe_losses
            
        # Delete loss tensors (CRITICAL)
        del pred_loss, aux_loss, total_loss
        if 'scaled_loss' in dir():
            del scaled_loss
        
        # Periodic deep cleanup
        if batch_idx % 100 == 0:
            gc.collect()
            
        # Reset peak memory stats periodically to track leaks
        if batch_idx % 1000 == 0 and torch.cuda.is_available():
            if is_main:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"    Peak memory this interval: {peak:.2f}GB")
            torch.cuda.reset_peak_memory_stats()
```

---

### Complete Memory-Safe Metrics Logging

Replace the metrics logging section with this safer version:

```python
# ========================================================================
# STEP 6a: COMPUTE & LOG REAL-TIME METRICS (memory-safe version)
# ========================================================================        
if is_main and batch_idx % log_interval == 0:
    # Use inference mode for metrics (even lighter than no_grad)
    with torch.inference_mode():
        # Ensure output is detached before metrics
        if output is not None:
            output_for_metrics = output.detach()
        else:
            output_for_metrics = None
        
        batch_metrics = compute_batch_metrics_lightweight(
            output_for_metrics, y, 
            dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt, 
            config, device
        )
        
        # Keep only recent metrics (prevent unbounded growth)
        batch_metrics_buffer.append(batch_metrics)
        if len(batch_metrics_buffer) > 100:
            batch_metrics_buffer = batch_metrics_buffer[-100:]
        
        # Print with explicit scalar conversion
        print(f"    Loss: {pred_loss_scalar:.4f} | "
              f"R@10: {batch_metrics['recall@10']:.3f} | "
              f"R@20: {batch_metrics['recall@20']:.3f}")
        
        # Clean up metrics tensors
        del output_for_metrics
```

---

## Summary: OOM Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| DataLoader workers | HIGH | Kernel death | Reduce to 4 workers |
| Tensor graph retention | MEDIUM | Gradual OOM | Use `.detach().item()` |
| Metrics buffer growth | LOW | Slow memory growth | Limit buffer size |
| Incomplete cleanup | MEDIUM | Gradual OOM | Comprehensive `del` |
| MoE tensor accumulation | LOW (no MoE in exp2b) | N/A | Detach before append |

**For exp2b_flash_learned_pool specifically:** Since it doesn't use MoE, the main risks are #1 (workers) and #2 (tensor graph retention). Fixing these should stabilize your training.

Would you like me to provide the complete modified `train_epoch` function with all these protections integrated?

```python
# Scheduler is stepped EVERY optimizer step (every batch or every N batches with accumulation)
total_steps = len(train_loader) // accumulation_steps * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

# Inside train_epoch:
if accumulation_counter >= accumulation_steps:
    optimizer.step()
    scheduler.step()  # ← Called per optimizer step
```

```plaintext
Epoch 1: LR decays smoothly from max → ~0 across all 12335 batches
         Step 0: LR=4e-4, Step 3000: LR=3e-4, Step 6000: LR=2e-4, Step 12335: LR≈0
```

```python
# Scheduler is stepped ONCE per epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# In run_single_experiment, OUTSIDE the batch loop:
for epoch in range(epochs):
    train_metrics = train_epoch(...)  # No scheduler.step() inside
    scheduler.step()  # ← Called once per epoch
```

```plaintext
Epoch 1: LR = 4e-4 (constant throughout epoch)
Epoch 2: LR = 2.8e-4 (cosine decay step)
Epoch 3: LR = 1.2e-4
Epoch 4: LR = 0.4e-4
```

```python
# In run_single_experiment, after creating optimizer:

# Calculate steps
steps_per_epoch = len(train_loader) // accumulation_steps
total_steps = steps_per_epoch * epochs
warmup_steps = min(1000, total_steps // 10)  # 10% warmup or 1000 steps

# Create scheduler with warmup
def get_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine decay - industry standard for transformers."""
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
```

```python
# BAD: Keeps computation graph in memory
total_pred_loss += pred_loss  # pred_loss is a tensor with graph

# GOOD: Breaks the graph
total_pred_loss += pred_loss.item()  # Scalar, no graph
# or
total_pred_loss += pred_loss.detach()  # Tensor without graph
```

```python
# In train_epoch, around line 4404:
pred_loss_scalar = pred_loss.detach().item()  # Use detach() before item()
aux_loss_scalar = aux_loss.detach().item()

total_pred_loss += pred_loss_scalar
total_aux_loss += aux_loss_scalar
```

```python
batch_metrics_buffer = []  # Grows every log_interval batches
moe_metrics_buffer = []    # Grows every log_interval batches

# With 12335 batches and log_interval=100:
# = 123 entries × dict of floats ≈ small, but each dict may hold tensor refs
```

```python
# Add at top of train_epoch:
MAX_METRICS_BUFFER = 100  # Keep last 100 logged batches

# When appending:
batch_metrics_buffer.append(batch_metrics)
if len(batch_metrics_buffer) > MAX_METRICS_BUFFER:
    batch_metrics_buffer.pop(0)  # Remove oldest
```

```python
if 'expert_usage' in moe_losses:
    expert_usage_list.append(moe_losses['expert_usage'])  # Keeps tensor references
```

```python
if 'expert_usage' in moe_losses:
    expert_usage_list.append(moe_losses['expert_usage'].detach().cpu())
```

```python
# Current cleanup (line ~4449)
del x, targets_mh, dt_cnt
if output is not None:
    del output
del pred_loss, total_loss, scaled_loss
```

```python
# After backward and before next batch:
del x, targets_mh, dt_cnt
if 'extras' in dir() and extras is not None:
    del extras
if 'output' in dir() and output is not None:
    del output
if 'moe_losses' in dir() and moe_losses:
    del moe_losses
del pred_loss, aux_loss, total_loss, scaled_loss, result
```

```python
def train_epoch(
    ...
    memory_warning_threshold_gb: float = 12.0,  # Warn at 12GB (T4 has 16GB)
    memory_critical_threshold_gb: float = 14.0,  # Critical at 14GB
) -> Dict[str, float]:
    """Train with memory protection."""
    
    # ... existing setup ...
    
    for batch_idx, batch in enumerate(dataloader):
        
        # ================================================================
        # MEMORY PROTECTION: Check before each batch
        # ================================================================
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            current_memory_gb = torch.cuda.memory_allocated() / 1024**3
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            
            # Warning level
            if peak_memory_gb > memory_warning_threshold_gb:
                if is_main:
                    print(f"⚠️ Memory warning: {peak_memory_gb:.2f}GB peak")
                # Aggressive cleanup
                gc.collect()
                torch.cuda.empty_cache()
            
            # Critical level - reduce batch processing
            if current_memory_gb > memory_critical_threshold_gb:
                if is_main:
                    print(f"🔴 Memory critical: {current_memory_gb:.2f}GB - forcing cleanup")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Skip metrics computation for this batch to save memory
                need_predictions = False
        
        # ... forward pass ...
        
        # ================================================================
        # AGGRESSIVE CLEANUP after backward
        # ================================================================
        # Delete all batch tensors
        del x, targets_mh, dt_cnt, age, gender, lob, codes
        
        # Delete forward outputs
        if 'result' in dir():
            del result
        if 'extras' in dir() and extras is not None:
            if 'predictions' in extras:
                del extras['predictions']
            if 'moe_losses' in extras:
                del extras['moe_losses']
            del extras
        if 'output' in dir() and output is not None:
            del output
        if 'moe_losses' in dir():
            del moe_losses
            
        # Delete loss tensors (CRITICAL)
        del pred_loss, aux_loss, total_loss
        if 'scaled_loss' in dir():
            del scaled_loss
        
        # Periodic deep cleanup
        if batch_idx % 100 == 0:
            gc.collect()
            
        # Reset peak memory stats periodically to track leaks
        if batch_idx % 1000 == 0 and torch.cuda.is_available():
            if is_main:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"    Peak memory this interval: {peak:.2f}GB")
            torch.cuda.reset_peak_memory_stats()
```

```python
# ========================================================================
# STEP 6a: COMPUTE & LOG REAL-TIME METRICS (memory-safe version)
# ========================================================================        
if is_main and batch_idx % log_interval == 0:
    # Use inference mode for metrics (even lighter than no_grad)
    with torch.inference_mode():
        # Ensure output is detached before metrics
        if output is not None:
            output_for_metrics = output.detach()
        else:
            output_for_metrics = None
        
        batch_metrics = compute_batch_metrics_lightweight(
            output_for_metrics, y, 
            dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt, 
            config, device
        )
        
        # Keep only recent metrics (prevent unbounded growth)
        batch_metrics_buffer.append(batch_metrics)
        if len(batch_metrics_buffer) > 100:
            batch_metrics_buffer = batch_metrics_buffer[-100:]
        
        # Print with explicit scalar conversion
        print(f"    Loss: {pred_loss_scalar:.4f} | "
              f"R@10: {batch_metrics['recall@10']:.3f} | "
              f"R@20: {batch_metrics['recall@20']:.3f}")
        
        # Clean up metrics tensors
        del output_for_metrics
```

---

**User**

for the risk1, I need detailed instruction where to add and change the code

---

**Cursor**

Let me find all the places in `train_epoch` where tensor graph retention can occur:

Now I can see all the issues. Here are the **exact line-by-line changes** needed:

---

## Risk 1 Fixes: Tensor Graph Retention

### Location 1: Line 4388 - `accumulated_loss` ✅ Already OK

```python
# Line 4388 - This is already correct:
accumulated_loss += total_loss.detach()
```

---

### Location 2: Lines 4418-4424 - Loss Scalar Tracking

**Current code (lines 4418-4424):**
```python
        # Track losses - handle DataParallel multi-GPU tensors
        pred_loss_scalar = pred_loss.mean().item() if pred_loss.numel() > 1 else pred_loss.item()
        aux_loss_scalar = aux_loss.mean().item() if aux_loss.numel() > 1 else aux_loss.item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
```

**Issue:** These lines are OK, but move them BEFORE the logging section so we can use the scalars everywhere.

**Fixed code - replace lines 4418-4424 with:**
```python
        # ============================================================
        # STEP 6: CLEANUP & LOGGING
        # ============================================================
        
        # IMMEDIATELY extract scalars and detach to prevent graph retention
        # Do this BEFORE any logging to ensure we don't keep graph alive
        with torch.no_grad():
            pred_loss_scalar = pred_loss.detach().mean().item() if pred_loss.numel() > 1 else pred_loss.detach().item()
            aux_loss_scalar = aux_loss.detach().mean().item() if aux_loss.numel() > 1 else aux_loss.detach().item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
```

**Key change:** Added `.detach()` before `.mean().item()` to explicitly break the graph.

---

### Location 3: Lines 4429-4437 - Metrics Computation with `output`

**Current code (lines 4429-4437):**
```python
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                # Compute batch metrics (FAST)
                batch_metrics = compute_batch_metrics_lightweight(
                    output, y, 
                    # convert back to list from tensor
                    dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt, 
                    config, device
                )
```

**Issue:** `output` tensor may still have graph attached when passed to metrics function.

**Fixed code - replace lines 4429-4463 with:**
```python
        # ========================================================================
        # STEP 6a: COMPUTE & LOG REAL-TIME METRICS (every log_interval batches)
        # ========================================================================        
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                # CRITICAL: Detach output before metrics to break graph
                output_detached = output.detach() if output is not None else None
                
                # Convert dt_cnt to list BEFORE any tensor ops (it will be deleted soon)
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else list(dt_cnt)
                
                # Compute batch metrics (FAST)
                batch_metrics = compute_batch_metrics_lightweight(
                    output_detached, y, 
                    dt_cnt_list, 
                    config, device
                )
                batch_metrics_buffer.append(batch_metrics)
                
                # Log to console - use already-computed scalar (don't recompute!)
                print(f"    Loss: {pred_loss_scalar:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"P@20: {batch_metrics['precision@20']:.3f} | "
                      f"mAP20: {batch_metrics['mAP@20']:.3f} | "
                      f"mAP50: {batch_metrics['mAP@50']:.3f} | "
                      f"Brier: {batch_metrics['brier_score']:.4f}")
                
                # MoE metrics if applicable
                if moe_losses and 'expert_usage' in moe_losses:
                    # CRITICAL: Detach MoE tensors before computing metrics
                    moe_losses_detached = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v 
                        for k, v in moe_losses.items()
                    }
                    moe_batch_metrics = compute_moe_batch_metrics(moe_losses_detached)
                    moe_metrics_buffer.append(moe_batch_metrics)
                    
                    print(f"    MoE: CV={moe_batch_metrics['expert_load_cv']:.3f} | "
                          f"Collapsed={moe_batch_metrics['num_collapsed_experts']} | "
                          f"Gini={moe_batch_metrics['expert_gini']:.3f}")
                    
                    if moe_batch_metrics['num_collapsed_experts'] > 0:
                        print(f"⚠️ {moe_batch_metrics['num_collapsed_experts']} experts collapsed!")
                    
                    del moe_losses_detached  # Clean up
                
                del output_detached  # Clean up detached copy
```

**Key changes:**
1. `output.detach()` before passing to metrics
2. Use `pred_loss_scalar` instead of recomputing `pred_loss.mean().item()`
3. Detach all MoE loss tensors
4. Delete temporary detached tensors

---

### Location 4: Lines 4464-4468 - Incomplete Cleanup

**Current code (lines 4464-4468):**
```python
        # Memory cleanup (NO empty_cache in loop!)
        del x, targets_mh, dt_cnt
        if 'output' in dir() and output is not None:
            del output
        del pred_loss, total_loss, scaled_loss
```

**Issue:** Missing `aux_loss`, `extras`, `moe_losses`, and `result`.

**Fixed code - replace lines 4464-4468 with:**
```python
        # ============================================================
        # MEMORY CLEANUP (COMPREHENSIVE)
        # ============================================================
        # Delete input tensors
        del x, targets_mh
        if 'dt_cnt' in dir():
            del dt_cnt
        
        # Delete forward pass results
        if 'result' in dir():
            del result
        if 'extras' in dir() and extras is not None:
            del extras
        if 'output' in dir() and output is not None:
            del output
        if 'moe_losses' in dir() and moe_losses:
            del moe_losses
        
        # Delete loss tensors (CRITICAL - these hold the entire backward graph)
        del pred_loss, aux_loss, total_loss
        if 'scaled_loss' in dir():
            del scaled_loss
```

---

## Complete Modified Section (Lines 4414-4480)

Here's the complete fixed section you can copy and replace:

```python
        if should_track:
            gpu_tracker.print_gpu_use_summary()
            
        # ============================================================
        # STEP 6: CLEANUP & LOGGING
        # ============================================================
        
        # IMMEDIATELY extract scalars and detach to prevent graph retention
        # Do this BEFORE any logging to ensure we don't keep graph alive
        with torch.no_grad():
            pred_loss_scalar = pred_loss.detach().mean().item() if pred_loss.numel() > 1 else pred_loss.detach().item()
            aux_loss_scalar = aux_loss.detach().mean().item() if aux_loss.numel() > 1 else aux_loss.detach().item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
        
        # ========================================================================
        # STEP 6a: COMPUTE & LOG REAL-TIME METRICS (every log_interval batches)
        # ========================================================================        
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                # CRITICAL: Detach output before metrics to break graph
                output_detached = output.detach() if output is not None else None
                
                # Convert dt_cnt to list BEFORE deletion
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else list(dt_cnt)
                
                # Compute batch metrics
                batch_metrics = compute_batch_metrics_lightweight(
                    output_detached, y, 
                    dt_cnt_list, 
                    config, device
                )
                batch_metrics_buffer.append(batch_metrics)
                
                # Log to console - use already-computed scalar
                print(f"    Loss: {pred_loss_scalar:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"P@20: {batch_metrics['precision@20']:.3f} | "
                      f"mAP20: {batch_metrics['mAP@20']:.3f} | "
                      f"mAP50: {batch_metrics['mAP@50']:.3f} | "
                      f"Brier: {batch_metrics['brier_score']:.4f}")
                
                # MoE metrics if applicable
                if moe_losses and 'expert_usage' in moe_losses:
                    moe_losses_detached = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v 
                        for k, v in moe_losses.items()
                    }
                    moe_batch_metrics = compute_moe_batch_metrics(moe_losses_detached)
                    moe_metrics_buffer.append(moe_batch_metrics)
                    
                    print(f"    MoE: CV={moe_batch_metrics['expert_load_cv']:.3f} | "
                          f"Collapsed={moe_batch_metrics['num_collapsed_experts']} | "
                          f"Gini={moe_batch_metrics['expert_gini']:.3f}")
                    
                    if moe_batch_metrics['num_collapsed_experts'] > 0:
                        print(f"⚠️ {moe_batch_metrics['num_collapsed_experts']} experts collapsed!")
                    
                    del moe_losses_detached
                
                del output_detached
                        
        # ============================================================
        # MEMORY CLEANUP (COMPREHENSIVE)
        # ============================================================
        del x, targets_mh
        if 'dt_cnt' in dir():
            del dt_cnt
        if 'result' in dir():
            del result
        if 'extras' in dir() and extras is not None:
            del extras
        if 'output' in dir() and output is not None:
            del output
        if 'moe_losses' in dir() and moe_losses:
            del moe_losses
        del pred_loss, aux_loss, total_loss
        if 'scaled_loss' in dir():
            del scaled_loss
```

---

## Summary of Changes

| Line | Issue | Fix |
|------|-------|-----|
| 4419-4420 | Missing `.detach()` | Add `.detach()` before `.mean().item()` |
| 4432-4436 | `output` passed with graph | Use `output.detach()` |
| 4441 | Redundant recomputation | Use `pred_loss_scalar` instead |
| 4452-4454 | MoE tensors with graph | Detach all MoE tensors |
| 4468 | Missing `aux_loss`, `extras`, etc. | Add comprehensive cleanup |

```python
# Line 4388 - This is already correct:
accumulated_loss += total_loss.detach()
```

```python
        # Track losses - handle DataParallel multi-GPU tensors
        pred_loss_scalar = pred_loss.mean().item() if pred_loss.numel() > 1 else pred_loss.item()
        aux_loss_scalar = aux_loss.mean().item() if aux_loss.numel() > 1 else aux_loss.item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
```

```python
        # ============================================================
        # STEP 6: CLEANUP & LOGGING
        # ============================================================
        
        # IMMEDIATELY extract scalars and detach to prevent graph retention
        # Do this BEFORE any logging to ensure we don't keep graph alive
        with torch.no_grad():
            pred_loss_scalar = pred_loss.detach().mean().item() if pred_loss.numel() > 1 else pred_loss.detach().item()
            aux_loss_scalar = aux_loss.detach().mean().item() if aux_loss.numel() > 1 else aux_loss.detach().item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
```

```python
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                # Compute batch metrics (FAST)
                batch_metrics = compute_batch_metrics_lightweight(
                    output, y, 
                    # convert back to list from tensor
                    dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt, 
                    config, device
                )
```

```python
        # ========================================================================
        # STEP 6a: COMPUTE & LOG REAL-TIME METRICS (every log_interval batches)
        # ========================================================================        
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                # CRITICAL: Detach output before metrics to break graph
                output_detached = output.detach() if output is not None else None
                
                # Convert dt_cnt to list BEFORE any tensor ops (it will be deleted soon)
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else list(dt_cnt)
                
                # Compute batch metrics (FAST)
                batch_metrics = compute_batch_metrics_lightweight(
                    output_detached, y, 
                    dt_cnt_list, 
                    config, device
                )
                batch_metrics_buffer.append(batch_metrics)
                
                # Log to console - use already-computed scalar (don't recompute!)
                print(f"    Loss: {pred_loss_scalar:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"P@20: {batch_metrics['precision@20']:.3f} | "
                      f"mAP20: {batch_metrics['mAP@20']:.3f} | "
                      f"mAP50: {batch_metrics['mAP@50']:.3f} | "
                      f"Brier: {batch_metrics['brier_score']:.4f}")
                
                # MoE metrics if applicable
                if moe_losses and 'expert_usage' in moe_losses:
                    # CRITICAL: Detach MoE tensors before computing metrics
                    moe_losses_detached = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v 
                        for k, v in moe_losses.items()
                    }
                    moe_batch_metrics = compute_moe_batch_metrics(moe_losses_detached)
                    moe_metrics_buffer.append(moe_batch_metrics)
                    
                    print(f"    MoE: CV={moe_batch_metrics['expert_load_cv']:.3f} | "
                          f"Collapsed={moe_batch_metrics['num_collapsed_experts']} | "
                          f"Gini={moe_batch_metrics['expert_gini']:.3f}")
                    
                    if moe_batch_metrics['num_collapsed_experts'] > 0:
                        print(f"⚠️ {moe_batch_metrics['num_collapsed_experts']} experts collapsed!")
                    
                    del moe_losses_detached  # Clean up
                
                del output_detached  # Clean up detached copy
```

```python
        # Memory cleanup (NO empty_cache in loop!)
        del x, targets_mh, dt_cnt
        if 'output' in dir() and output is not None:
            del output
        del pred_loss, total_loss, scaled_loss
```

```python
        # ============================================================
        # MEMORY CLEANUP (COMPREHENSIVE)
        # ============================================================
        # Delete input tensors
        del x, targets_mh
        if 'dt_cnt' in dir():
            del dt_cnt
        
        # Delete forward pass results
        if 'result' in dir():
            del result
        if 'extras' in dir() and extras is not None:
            del extras
        if 'output' in dir() and output is not None:
            del output
        if 'moe_losses' in dir() and moe_losses:
            del moe_losses
        
        # Delete loss tensors (CRITICAL - these hold the entire backward graph)
        del pred_loss, aux_loss, total_loss
        if 'scaled_loss' in dir():
            del scaled_loss
```

```python
        if should_track:
            gpu_tracker.print_gpu_use_summary()
            
        # ============================================================
        # STEP 6: CLEANUP & LOGGING
        # ============================================================
        
        # IMMEDIATELY extract scalars and detach to prevent graph retention
        # Do this BEFORE any logging to ensure we don't keep graph alive
        with torch.no_grad():
            pred_loss_scalar = pred_loss.detach().mean().item() if pred_loss.numel() > 1 else pred_loss.detach().item()
            aux_loss_scalar = aux_loss.detach().mean().item() if aux_loss.numel() > 1 else aux_loss.detach().item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
        
        # ========================================================================
        # STEP 6a: COMPUTE & LOG REAL-TIME METRICS (every log_interval batches)
        # ========================================================================        
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                # CRITICAL: Detach output before metrics to break graph
                output_detached = output.detach() if output is not None else None
                
                # Convert dt_cnt to list BEFORE deletion
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else list(dt_cnt)
                
                # Compute batch metrics
                batch_metrics = compute_batch_metrics_lightweight(
                    output_detached, y, 
                    dt_cnt_list, 
                    config, device
                )
                batch_metrics_buffer.append(batch_metrics)
                
                # Log to console - use already-computed scalar
                print(f"    Loss: {pred_loss_scalar:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"P@20: {batch_metrics['precision@20']:.3f} | "
                      f"mAP20: {batch_metrics['mAP@20']:.3f} | "
                      f"mAP50: {batch_metrics['mAP@50']:.3f} | "
                      f"Brier: {batch_metrics['brier_score']:.4f}")
                
                # MoE metrics if applicable
                if moe_losses and 'expert_usage' in moe_losses:
                    moe_losses_detached = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v 
                        for k, v in moe_losses.items()
                    }
                    moe_batch_metrics = compute_moe_batch_metrics(moe_losses_detached)
                    moe_metrics_buffer.append(moe_batch_metrics)
                    
                    print(f"    MoE: CV={moe_batch_metrics['expert_load_cv']:.3f} | "
                          f"Collapsed={moe_batch_metrics['num_collapsed_experts']} | "
                          f"Gini={moe_batch_metrics['expert_gini']:.3f}")
                    
                    if moe_batch_metrics['num_collapsed_experts'] > 0:
                        print(f"⚠️ {moe_batch_metrics['num_collapsed_experts']} experts collapsed!")
                    
                    del moe_losses_detached
                
                del output_detached
                        
        # ============================================================
        # MEMORY CLEANUP (COMPREHENSIVE)
        # ============================================================
        del x, targets_mh
        if 'dt_cnt' in dir():
            del dt_cnt
        if 'result' in dir():
            del result
        if 'extras' in dir() and extras is not None:
            del extras
        if 'output' in dir() and output is not None:
            del output
        if 'moe_losses' in dir() and moe_losses:
            del moe_losses
        del pred_loss, aux_loss, total_loss
        if 'scaled_loss' in dir():
            del scaled_loss
```

---

**User**

Ok now, review the code that I have jsut refactored and examine if the modificaiton is correct and there are any other remianing issues @dev/moe/moe_flashattn_3.py 

---

**Cursor**

Let me review the refactored `train_epoch` function and other key areas:

## Code Review Results

### ✅ Correctly Fixed

| Location | Fix | Status |
|----------|-----|--------|
| Line 4241 | `accumulation_steps: int = 1` default | ✅ **GOOD** - Changed from 4 to 1 |
| Line 8993 | `n_workers = min(4, os.cpu_count() // 4)` | ✅ **GOOD** - Was 32 workers |
| Lines 4421-4423 | `.detach()` added before `.item()` | ✅ **GOOD** |
| Line 4434 | `output.detach()` before metrics | ✅ **GOOD** |
| Lines 4444-4445 | Buffer limiting added | ✅ **GOOD** |
| Lines 4460-4463 | MoE losses detached | ✅ **GOOD** |
| Lines 4477-4485 | Comprehensive cleanup | ✅ **GOOD** |

---

### ⚠️ Remaining Issues Found

#### Issue 1: Line 4448 - Redundant `.item()` Call

**Current code:**
```python
loss_display = pred_loss.mean().item() if pred_loss.numel() > 1 else pred_loss.item()
```

**Problem:** This calls `.item()` on `pred_loss` AGAIN without `.detach()`. You already computed `pred_loss_scalar` at line 4422.

**Fix:** Replace line 4448-4449 with:
```python
                # Log to console - use already-computed scalar (no need to call .item() again)
                print(f"    Loss: {pred_loss_scalar:.4f} | "
```

---

#### Issue 2: Line 9316 - Scheduler T_max Still Incorrect

**Current code:**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

**Problem:** With `accumulation_steps=1` and stepping per batch, `T_max=epochs` means the cosine cycle completes in just `epochs` steps (e.g., 1 step for 1 epoch). It should be total optimizer steps.

**Fix - Option A (Step per epoch instead):**
Move `scheduler.step()` to end of epoch in `run_single_experiment`:
```python
# In run_single_experiment, after train_epoch call:
train_metrics = train_epoch(...)
scheduler.step()  # ← Move here, call once per epoch
```

And remove/comment out `scheduler.step()` from inside `train_epoch` (around line 4406).

**Fix - Option B (Correct T_max for step-based):**
```python
total_steps = len(train_loader) * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

But you'd need `len(train_loader)` before creating scheduler, which requires creating dataloader first. Easier to use Option A.

---

#### Issue 3: Missing `accumulation_steps` in train_epoch call (line 9341-9359)

**Current code:**
```python
train_metrics = train_epoch(
    model=model,
    dataloader=train_loader,
    # ... other params ...
    use_ddp=use_ddp
)  # ← accumulation_steps not passed, uses default=1
```

**Status:** This is actually **OK now** because default is 1. But for clarity and future flexibility, you might want to add it explicitly:

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
    accumulation_steps=1  # Explicit: no gradient accumulation with DataParallel
)
```

---

### Summary of Required Changes

| Priority | Issue | Line | Action |
|----------|-------|------|--------|
| 🔴 HIGH | Redundant `.item()` | 4448 | Use `pred_loss_scalar` instead |
| 🟡 MEDIUM | Scheduler T_max | 9316 | Move scheduler.step() to epoch-level |
| 🟢 LOW | Explicit accumulation | 9341-9359 | Add `accumulation_steps=1` for clarity |

---

### Quick Fix for Line 4448

Replace lines 4447-4456:

```python
                # Log to console - use already-computed scalar
                print(f"    Loss: {pred_loss_scalar:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"P@20: {batch_metrics['precision@20']:.3f} | "
                      f"mAP20: {batch_metrics['mAP@20']:.3f} | "
                      f"mAP50: {batch_metrics['mAP@50']:.3f} | "
                      f"Brier: {batch_metrics['brier_score']:.4f}")
```

---

### Quick Fix for Scheduler (Epoch-Based)

**Step 1:** In `train_epoch` (around line 4405-4406), comment out or remove the scheduler step:
```python
            if scheduler is not None:
                pass  # scheduler.step()  # ← Disabled, moved to epoch level
```

**Step 2:** In `run_single_experiment`, after the `train_epoch` call (around line 9360), add:
```python
        train_metrics = train_epoch(...)
        global_step = train_metrics['global_step']
        
        # Step scheduler once per epoch
        if scheduler is not None:
            scheduler.step()
```

This ensures the cosine schedule completes over `epochs` total epochs, not steps.

```python
loss_display = pred_loss.mean().item() if pred_loss.numel() > 1 else pred_loss.item()
```

```python
                # Log to console - use already-computed scalar (no need to call .item() again)
                print(f"    Loss: {pred_loss_scalar:.4f} | "
```

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

```python
# In run_single_experiment, after train_epoch call:
train_metrics = train_epoch(...)
scheduler.step()  # ← Move here, call once per epoch
```

```python
total_steps = len(train_loader) * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

```python
train_metrics = train_epoch(
    model=model,
    dataloader=train_loader,
    # ... other params ...
    use_ddp=use_ddp
)  # ← accumulation_steps not passed, uses default=1
```

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
    accumulation_steps=1  # Explicit: no gradient accumulation with DataParallel
)
```

```python
                # Log to console - use already-computed scalar
                print(f"    Loss: {pred_loss_scalar:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"P@20: {batch_metrics['precision@20']:.3f} | "
                      f"mAP20: {batch_metrics['mAP@20']:.3f} | "
                      f"mAP50: {batch_metrics['mAP@50']:.3f} | "
                      f"Brier: {batch_metrics['brier_score']:.4f}")
```

```python
            if scheduler is not None:
                pass  # scheduler.step()  # ← Disabled, moved to epoch level
```

```python
        train_metrics = train_epoch(...)
        global_step = train_metrics['global_step']
        
        # Step scheduler once per epoch
        if scheduler is not None:
            scheduler.step()
```

---

**User**

I have chosen to use   the folloiwng to update teh schedular, then tell me how to modify the optimizaton part in train_epoch function? and are there any ohter places that also need moficiatons?  # ============================================================
    # 4. OPTIMIZER SETUP
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

@moe_flashattn_3.py (4387-4406) 

---

**Cursor**

```python
total_steps = len(train_loader) * epochs


```

```python
total_steps = (len(train_loader) // accumulation_steps) * epochs
```

Good choice! With step-based scheduling (`T_max=total_steps`), the current code structure in `train_epoch` is **almost correct**, but I need to verify the complete block. Let me check:

The current code structure is **correct** for step-based scheduling. Here's the verification:

## ✅ Current Code is Correct

With `accumulation_steps=1`:
1. Line 4389: Counter increments to 1
2. Line 4391: `if 1 >= 1:` → True (runs every batch)
3. Lines 4392-4403: Optimizer steps
4. Lines 4405-4406: Scheduler steps
5. Lines 4407-4410: Counter resets, global_step increments

**No changes needed to lines 4387-4410.**

---

## ⚠️ But You Need to Check One More Thing

With step-based scheduling, you need `optimizer.zero_grad()` to be called **before each forward pass**. Let me verify:

```python
if accumulation_counter == 0:
    optimizer.zero_grad(set_to_none=True)
```

`zero_grad` is correctly placed at line 4306-4307:
```python
if accumulation_counter == 0:
    optimizer.zero_grad(set_to_none=True)
```

With `accumulation_steps=1`, counter resets to 0 after each batch (line 4409), so `zero_grad` runs before each batch. ✅

---

## Complete Changes Needed

### 1. In `run_single_experiment` - Scheduler Setup (around line 9316)

**Replace:**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

**With:**
```python
total_steps = len(train_loader) * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

⚠️ **BUT there's a catch:** `train_loader` is created AFTER the current scheduler line. You need to move the scheduler creation AFTER the dataloaders.

**Full fix - Reorder the sections:**
```python
    # ============================================================
    # 3. DATA PREPARATION (moved before optimizer setup)
    # ============================================================
    if code_frequencies is None:
        code_frequencies = compute_code_frequencies(train_data, config, device)
    
    train_loader, val_loader = _create_dataloaders(
        train_data, val_data, config, use_bucketing, logger=logger
    )
    
    # ============================================================
    # 4. OPTIMIZER SETUP (now after dataloaders so we have len(train_loader))
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    
    # Step-based cosine scheduler: decays over all training steps
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    logger.info(f"Scheduler: CosineAnnealingLR with T_max={total_steps} steps")
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
```

---

### 2. In `train_epoch` - No Changes Needed

Lines 4387-4410 are already correct for step-based scheduling with `accumulation_steps=1`.

---

### 3. Check End-of-Epoch Handler (lines 4499-4509)

Let me verify this section handles remaining gradients correctly:

```python
if accumulation_counter == 0:
    optimizer.zero_grad(set_to_none=True)
```

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

```python
total_steps = len(train_loader) * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

```python
    # ============================================================
    # 3. DATA PREPARATION (moved before optimizer setup)
    # ============================================================
    if code_frequencies is None:
        code_frequencies = compute_code_frequencies(train_data, config, device)
    
    train_loader, val_loader = _create_dataloaders(
        train_data, val_data, config, use_bucketing, logger=logger
    )
    
    # ============================================================
    # 4. OPTIMIZER SETUP (now after dataloaders so we have len(train_loader))
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    
    # Step-based cosine scheduler: decays over all training steps
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    logger.info(f"Scheduler: CosineAnnealingLR with T_max={total_steps} steps")
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
```

### 4. End-of-Epoch Handler (lines 4499-4509) - Minor Issue

**Current code:**
```python
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            # ...
            scaler.step(optimizer)
            scaler.update()
        else:
            # ...
            optimizer.step()
        global_step += 1  # ← But no scheduler.step()!
```

**Issue:** If this block runs (accumulation > 1), it does `optimizer.step()` but NOT `scheduler.step()`, causing a mismatch.

**With `accumulation_steps=1`:** This block will NEVER run (counter is always 0 at epoch end), so it's **not a problem**.

**For future-proofing**, you could fix it:
```python
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()  # ← ADD THIS for consistency
        
        global_step += 1
```

---

## Summary: Complete Changes Required

| Location | Change | Priority |
|----------|--------|----------|
| `run_single_experiment` lines 9298-9317 | Move dataloader creation BEFORE optimizer setup, add `total_steps = len(train_loader) * epochs` | 🔴 **Required** |
| `train_epoch` lines 4387-4410 | No changes | ✅ Already correct |
| `train_epoch` lines 4499-4509 | Add `scheduler.step()` after `optimizer.step()` | 🟢 Optional (only needed if using accumulation > 1) |

---

## Final Code Block for `run_single_experiment` (lines 9298-9320)

Replace the DATA PREPARATION and OPTIMIZER SETUP sections with:

```python
    # ============================================================
    # 3. DATA PREPARATION (BEFORE optimizer setup - need len(train_loader))
    # ============================================================
    if code_frequencies is None:
        code_frequencies = compute_code_frequencies(train_data, config, device)
    
    train_loader, val_loader = _create_dataloaders(
        train_data, val_data, config, use_bucketing, logger=logger
    )
    
    # ============================================================
    # 4. OPTIMIZER SETUP
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    
    # Step-based cosine scheduler: LR decays smoothly over all training steps
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    logger.info(f"Scheduler: CosineAnnealing with T_max={total_steps} total steps")
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # ============================================================
    # 5. RESUME FROM CHECKPOINT (if applicable)
    # ============================================================
    start_epoch, global_step, best_val_loss = 0, 0, float('inf')
    
    if is_resume:
        start_epoch, global_step, best_val_loss = _resume_from_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device, logger
        )
```

```python
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            # ...
            scaler.step(optimizer)
            scaler.update()
        else:
            # ...
            optimizer.step()
        global_step += 1  # ← But no scheduler.step()!
```

```python
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()  # ← ADD THIS for consistency
        
        global_step += 1
```

```python
    # ============================================================
    # 3. DATA PREPARATION (BEFORE optimizer setup - need len(train_loader))
    # ============================================================
    if code_frequencies is None:
        code_frequencies = compute_code_frequencies(train_data, config, device)
    
    train_loader, val_loader = _create_dataloaders(
        train_data, val_data, config, use_bucketing, logger=logger
    )
    
    # ============================================================
    # 4. OPTIMIZER SETUP
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    
    # Step-based cosine scheduler: LR decays smoothly over all training steps
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    logger.info(f"Scheduler: CosineAnnealing with T_max={total_steps} total steps")
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # ============================================================
    # 5. RESUME FROM CHECKPOINT (if applicable)
    # ============================================================
    start_epoch, global_step, best_val_loss = 0, 0, float('inf')
    
    if is_resume:
        start_epoch, global_step, best_val_loss = _resume_from_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device, logger
        )
```

```python
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            # ...
            scaler.step(optimizer)
            scaler.update()
        else:
            # ...
            optimizer.step()
        global_step += 1  # ← But no scheduler.step()!
```

```python
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()  # ← ADD THIS for consistency
        
        global_step += 1
```

```python
    # ============================================================
    # 3. DATA PREPARATION (BEFORE optimizer setup - need len(train_loader))
    # ============================================================
    if code_frequencies is None:
        code_frequencies = compute_code_frequencies(train_data, config, device)
    
    train_loader, val_loader = _create_dataloaders(
        train_data, val_data, config, use_bucketing, logger=logger
    )
    
    # ============================================================
    # 4. OPTIMIZER SETUP
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    
    # Step-based cosine scheduler: LR decays smoothly over all training steps
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    logger.info(f"Scheduler: CosineAnnealing with T_max={total_steps} total steps")
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # ============================================================
    # 5. RESUME FROM CHECKPOINT (if applicable)
    # ============================================================
    start_epoch, global_step, best_val_loss = 0, 0, float('inf')
    
    if is_resume:
        start_epoch, global_step, best_val_loss = _resume_from_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device, logger
        )
```

---

**User**

Ok there are folloiwng observations for the new run 
1) the model is learning faster but final loss and intrinsic metrics are not as high as expceted like 89%-91% before. Insepct the possible root cause and provide rationale and evidence
2) the memory across GPUs are pretty balanced; 
However, I still comes across died kernel and it seems like in validation set; the following is the result output and the kernel died in teh end; I would like the most optimal solution in terms of memory efficiency and elegance of code solution and code effectiveness. Don't do short-term solution or "patches", I wnat you to be long-sight 

  Batch 6100/10965
    Loss: 0.0032 | R@10: 0.771 | R@20: 0.841 | P@10: 0.181 | P@20: 0.115 | mAP20: 0.482 | mAP50: 0.417 | Brier: 0.0007
  Batch 6200/10965
    Loss: 0.0032 | R@10: 0.790 | R@20: 0.849 | P@10: 0.183 | P@20: 0.115 | mAP20: 0.465 | mAP50: 0.406 | Brier: 0.0007
  Batch 6300/10965
    Loss: 0.0030 | R@10: 0.818 | R@20: 0.875 | P@10: 0.190 | P@20: 0.119 | mAP20: 0.508 | mAP50: 0.448 | Brier: 0.0006
  Batch 6400/10965
    Loss: 0.0031 | R@10: 0.770 | R@20: 0.850 | P@10: 0.191 | P@20: 0.119 | mAP20: 0.489 | mAP50: 0.421 | Brier: 0.0006
  Batch 6500/10965
    Loss: 0.0030 | R@10: 0.801 | R@20: 0.863 | P@10: 0.196 | P@20: 0.122 | mAP20: 0.509 | mAP50: 0.450 | Brier: 0.0006
  Batch 6600/10965
    Loss: 0.0034 | R@10: 0.805 | R@20: 0.863 | P@10: 0.210 | P@20: 0.131 | mAP20: 0.485 | mAP50: 0.424 | Brier: 0.0007
  Batch 6700/10965
    Loss: 0.0030 | R@10: 0.763 | R@20: 0.845 | P@10: 0.194 | P@20: 0.121 | mAP20: 0.484 | mAP50: 0.420 | Brier: 0.0006
  Batch 6800/10965
    Loss: 0.0031 | R@10: 0.795 | R@20: 0.859 | P@10: 0.189 | P@20: 0.119 | mAP20: 0.495 | mAP50: 0.431 | Brier: 0.0007
  Batch 6900/10965
    Loss: 0.0032 | R@10: 0.771 | R@20: 0.856 | P@10: 0.207 | P@20: 0.131 | mAP20: 0.481 | mAP50: 0.417 | Brier: 0.0007
  Batch 7000/10965
    Loss: 0.0032 | R@10: 0.813 | R@20: 0.874 | P@10: 0.193 | P@20: 0.121 | mAP20: 0.508 | mAP50: 0.451 | Brier: 0.0007
    GPU 0: 0.79GB / 5.18GB peak
    GPU 1: 0.02GB / 2.42GB peak
    GPU 2: 0.02GB / 2.42GB peak
    GPU 3: 0.02GB / 2.43GB peak
  Batch 7100/10965
    Loss: 0.0030 | R@10: 0.808 | R@20: 0.872 | P@10: 0.190 | P@20: 0.120 | mAP20: 0.504 | mAP50: 0.446 | Brier: 0.0006
  Batch 7200/10965
    Loss: 0.0032 | R@10: 0.821 | R@20: 0.883 | P@10: 0.211 | P@20: 0.132 | mAP20: 0.493 | mAP50: 0.437 | Brier: 0.0007
  Batch 7300/10965
    Loss: 0.0031 | R@10: 0.813 | R@20: 0.883 | P@10: 0.190 | P@20: 0.120 | mAP20: 0.482 | mAP50: 0.428 | Brier: 0.0007
  Batch 7400/10965
    Loss: 0.0030 | R@10: 0.786 | R@20: 0.854 | P@10: 0.196 | P@20: 0.122 | mAP20: 0.484 | mAP50: 0.423 | Brier: 0.0006
  Batch 7500/10965
    Loss: 0.0031 | R@10: 0.814 | R@20: 0.874 | P@10: 0.200 | P@20: 0.125 | mAP20: 0.491 | mAP50: 0.432 | Brier: 0.0007
  Batch 7600/10965
    Loss: 0.0030 | R@10: 0.797 | R@20: 0.861 | P@10: 0.202 | P@20: 0.125 | mAP20: 0.518 | mAP50: 0.455 | Brier: 0.0006
  Batch 7700/10965
    Loss: 0.0030 | R@10: 0.816 | R@20: 0.890 | P@10: 0.199 | P@20: 0.126 | mAP20: 0.488 | mAP50: 0.436 | Brier: 0.0006
  Batch 7800/10965
    Loss: 0.0030 | R@10: 0.840 | R@20: 0.891 | P@10: 0.200 | P@20: 0.123 | mAP20: 0.511 | mAP50: 0.459 | Brier: 0.0006
  Batch 7900/10965
    Loss: 0.0029 | R@10: 0.799 | R@20: 0.863 | P@10: 0.192 | P@20: 0.118 | mAP20: 0.510 | mAP50: 0.448 | Brier: 0.0006
  Batch 8000/10965
    Loss: 0.0031 | R@10: 0.826 | R@20: 0.891 | P@10: 0.195 | P@20: 0.122 | mAP20: 0.511 | mAP50: 0.453 | Brier: 0.0007
    GPU 0: 0.79GB / 5.18GB peak
    GPU 1: 0.02GB / 2.42GB peak
    GPU 2: 0.02GB / 2.42GB peak
    GPU 3: 0.02GB / 2.43GB peak
  Batch 8100/10965
    Loss: 0.0031 | R@10: 0.793 | R@20: 0.863 | P@10: 0.197 | P@20: 0.123 | mAP20: 0.474 | mAP50: 0.415 | Brier: 0.0007
  Batch 8200/10965
    Loss: 0.0030 | R@10: 0.809 | R@20: 0.878 | P@10: 0.207 | P@20: 0.128 | mAP20: 0.507 | mAP50: 0.448 | Brier: 0.0006
  Batch 8300/10965
    Loss: 0.0031 | R@10: 0.814 | R@20: 0.875 | P@10: 0.204 | P@20: 0.129 | mAP20: 0.492 | mAP50: 0.435 | Brier: 0.0007
  Batch 8400/10965
    Loss: 0.0034 | R@10: 0.817 | R@20: 0.873 | P@10: 0.222 | P@20: 0.139 | mAP20: 0.503 | mAP50: 0.441 | Brier: 0.0007
  Batch 8500/10965
    Loss: 0.0030 | R@10: 0.804 | R@20: 0.875 | P@10: 0.192 | P@20: 0.121 | mAP20: 0.494 | mAP50: 0.439 | Brier: 0.0006
  Batch 8600/10965
    Loss: 0.0032 | R@10: 0.801 | R@20: 0.871 | P@10: 0.203 | P@20: 0.127 | mAP20: 0.504 | mAP50: 0.441 | Brier: 0.0007
  Batch 8700/10965
    Loss: 0.0031 | R@10: 0.808 | R@20: 0.871 | P@10: 0.192 | P@20: 0.123 | mAP20: 0.495 | mAP50: 0.434 | Brier: 0.0007
  Batch 8800/10965
    Loss: 0.0031 | R@10: 0.801 | R@20: 0.864 | P@10: 0.196 | P@20: 0.125 | mAP20: 0.497 | mAP50: 0.436 | Brier: 0.0007
  Batch 8900/10965
    Loss: 0.0032 | R@10: 0.826 | R@20: 0.885 | P@10: 0.207 | P@20: 0.129 | mAP20: 0.499 | mAP50: 0.444 | Brier: 0.0007
  Batch 9000/10965
    Loss: 0.0032 | R@10: 0.818 | R@20: 0.880 | P@10: 0.210 | P@20: 0.130 | mAP20: 0.508 | mAP50: 0.452 | Brier: 0.0007
    GPU 0: 0.79GB / 5.18GB peak
    GPU 1: 0.02GB / 2.42GB peak
    GPU 2: 0.02GB / 2.42GB peak
    GPU 3: 0.02GB / 2.43GB peak
  Batch 9100/10965
    Loss: 0.0031 | R@10: 0.788 | R@20: 0.859 | P@10: 0.179 | P@20: 0.115 | mAP20: 0.483 | mAP50: 0.419 | Brier: 0.0007
  Batch 9200/10965
    Loss: 0.0030 | R@10: 0.831 | R@20: 0.891 | P@10: 0.210 | P@20: 0.131 | mAP20: 0.507 | mAP50: 0.454 | Brier: 0.0007
  Batch 9300/10965
    Loss: 0.0031 | R@10: 0.829 | R@20: 0.894 | P@10: 0.196 | P@20: 0.122 | mAP20: 0.504 | mAP50: 0.447 | Brier: 0.0007
  Batch 9400/10965
    Loss: 0.0030 | R@10: 0.834 | R@20: 0.887 | P@10: 0.198 | P@20: 0.122 | mAP20: 0.495 | mAP50: 0.440 | Brier: 0.0007
  Batch 9500/10965
    Loss: 0.0031 | R@10: 0.813 | R@20: 0.883 | P@10: 0.193 | P@20: 0.121 | mAP20: 0.500 | mAP50: 0.441 | Brier: 0.0007
  Batch 9600/10965
    Loss: 0.0029 | R@10: 0.808 | R@20: 0.873 | P@10: 0.207 | P@20: 0.129 | mAP20: 0.511 | mAP50: 0.448 | Brier: 0.0006
  Batch 9700/10965
    Loss: 0.0031 | R@10: 0.818 | R@20: 0.881 | P@10: 0.187 | P@20: 0.118 | mAP20: 0.503 | mAP50: 0.447 | Brier: 0.0007
  Batch 9800/10965
    Loss: 0.0032 | R@10: 0.808 | R@20: 0.871 | P@10: 0.205 | P@20: 0.129 | mAP20: 0.516 | mAP50: 0.455 | Brier: 0.0007
  Batch 9900/10965
    Loss: 0.0032 | R@10: 0.803 | R@20: 0.865 | P@10: 0.202 | P@20: 0.128 | mAP20: 0.490 | mAP50: 0.427 | Brier: 0.0007
  Batch 10000/10965
    Loss: 0.0030 | R@10: 0.788 | R@20: 0.865 | P@10: 0.199 | P@20: 0.125 | mAP20: 0.495 | mAP50: 0.433 | Brier: 0.0006
    GPU 0: 0.79GB / 5.18GB peak
    GPU 1: 0.02GB / 2.42GB peak
    GPU 2: 0.02GB / 2.42GB peak
    GPU 3: 0.02GB / 2.43GB peak
  Batch 10100/10965
    Loss: 0.0030 | R@10: 0.795 | R@20: 0.871 | P@10: 0.199 | P@20: 0.124 | mAP20: 0.500 | mAP50: 0.440 | Brier: 0.0006
  Batch 10200/10965
    Loss: 0.0030 | R@10: 0.830 | R@20: 0.886 | P@10: 0.208 | P@20: 0.128 | mAP20: 0.495 | mAP50: 0.437 | Brier: 0.0007
  Batch 10300/10965
    Loss: 0.0030 | R@10: 0.793 | R@20: 0.866 | P@10: 0.206 | P@20: 0.128 | mAP20: 0.483 | mAP50: 0.424 | Brier: 0.0007
  Batch 10400/10965
    Loss: 0.0032 | R@10: 0.817 | R@20: 0.880 | P@10: 0.203 | P@20: 0.126 | mAP20: 0.497 | mAP50: 0.440 | Brier: 0.0007
  Batch 10500/10965
    Loss: 0.0036 | R@10: 0.764 | R@20: 0.849 | P@10: 0.227 | P@20: 0.147 | mAP20: 0.476 | mAP50: 0.407 | Brier: 0.0008
  Batch 10600/10965
    Loss: 0.0034 | R@10: 0.815 | R@20: 0.871 | P@10: 0.211 | P@20: 0.133 | mAP20: 0.494 | mAP50: 0.430 | Brier: 0.0007
  Batch 10700/10965
    Loss: 0.0031 | R@10: 0.796 | R@20: 0.870 | P@10: 0.183 | P@20: 0.117 | mAP20: 0.485 | mAP50: 0.427 | Brier: 0.0007
  Batch 10800/10965
    Loss: 0.0028 | R@10: 0.794 | R@20: 0.870 | P@10: 0.184 | P@20: 0.116 | mAP20: 0.484 | mAP50: 0.428 | Brier: 0.0006
  Batch 10900/10965
    Loss: 0.0032 | R@10: 0.816 | R@20: 0.878 | P@10: 0.210 | P@20: 0.130 | mAP20: 0.503 | mAP50: 0.445 | Brier: 0.0007
00:09:14 - exp2b_flash_learned_pool - INFO -   Evaluating on training subset...
00:10:17 - exp2b_flash_learned_pool - INFO -   Evaluating on validation set...

---

**Cursor**

```plaintext
00:09:14 - exp2b_flash_learned_pool - INFO -   Evaluating on training subset...
00:10:17 - exp2b_flash_learned_pool - INFO -   Evaluating on validation set...
```

Let me analyze both issues. First, let me examine the `evaluate()` function since that's where the kernel dies:

```python
all_predictions = []
all_targets = []
```

```python
all_predictions.append(valid_output.cpu())  # Appends tensors
all_targets.extend(valid_y)
```

```python
all_predictions = torch.cat(all_predictions)  # CONCATENATES ALL PREDICTIONS
```

# Comprehensive Analysis

## Issue 1: Lower Metrics (80% vs 89-91%)

**Current results:** R@10=~82%, R@20=~88% (still improving at batch 10900)

**Root Causes:**

1. **Only 1 Epoch** - Training ends while model is still learning. Look at the loss trajectory:
   - Batch 6100: Loss=0.0032, R@10=0.771
   - Batch 10900: Loss=0.0032, R@10=0.816
   - Still improving, not plateaued

2. **Cosine Scheduler Decay** - By batch 10965 (end), LR is nearly zero. The model can't learn effectively in later batches.

3. **Need More Epochs** - Your single-GPU baseline likely used more epochs or different LR schedule.

**Recommendation:** Run 3-4 epochs, or use a warmup-plateau-decay schedule instead of pure cosine.

---

## Issue 2: Kernel Dying During Validation 🔴 CRITICAL

**Root Cause: CPU Memory Exhaustion (NOT GPU)**

Looking at `evaluate()`:

```python
all_predictions = []  # Line 4584
all_targets = []      # Line 4585

# In the loop (lines 4672-4673):
all_predictions.append(valid_output.cpu())  # Accumulates on CPU
all_targets.extend(valid_y)                 # Accumulates on CPU

# After loop (line 4687):
all_predictions = torch.cat(all_predictions)  # CONCATENATES EVERYTHING
```

**The Math:**
- Validation set: ~350K samples (your data)
- Per sample: 200 days × 6297 targets × 4 bytes = 5MB
- Total: 350K × 200 × 6297 × 4 bytes = **1.7 TB of predictions**

Even if only 10% of days are valid, that's **170GB of CPU RAM** - exceeds your 416GB when combined with other processes.

**Evidence:** GPU memory is fine (0.79GB/5.18GB peak). The kernel dies = CPU memory exhaustion.

---

# The Optimal Long-Term Solution

## Redesign `evaluate()` to Be Streaming/Memory-Efficient

**Principle:** Never accumulate all predictions. Compute metrics incrementally.

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = False 
) -> Dict[str, float]:
    """
    Memory-efficient evaluation using streaming metrics.
    
    Key design:
    - NEVER accumulate all predictions (would exhaust CPU RAM)
    - Compute metrics per-batch and aggregate
    - Use online/streaming algorithms for Top-K accuracy
    """
    model.eval()
    nbatch = len(dataloader)
    batches_to_process = min(nbatch, max_batches) if max_batches else nbatch
    
    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and isinstance(model.module, DataParallelWrapper)
    )
    
    if batches_to_process == 0:
        return {'val_loss': 0.0, 'top_1_acc': 0.0, 'top_5_acc': 0.0, 
                'top_10_acc': 0.0, 'top_20_acc': 0.0}
    
    # ========================================================================
    # STREAMING METRICS: Accumulate counts, not tensors
    # ========================================================================
    total_loss = 0.0
    total_samples = 0
    
    # Top-K accuracy counters (streaming)
    top_k_correct = {1: 0, 5: 0, 10: 0, 20: 0}
    top_k_total = {1: 0, 5: 0, 10: 0, 20: 0}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            
            if batch_idx >= batches_to_process:
                if verbose:
                    print(f"    Early exit at batch {batch_idx}/{nbatch}")
                break
            
            # Periodic memory check
            if batch_idx % 100 == 0 and verbose:
                print(f"    Evaluating batch {batch_idx}/{batches_to_process}")
            
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            lob = batch['lob'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),
                codes
            ], dim=-1)
            
            # Forward pass
            if is_wrapped:
                targets_mh = batch['target_multihot'].to(device, non_blocking=True)
                dt_cnt_tensor = dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor) else torch.tensor(dt_cnt, device=device)
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                else:
                    result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                
                if isinstance(result, tuple):
                    loss_val, extras = result
                    output = extras.get('predictions')
                else:
                    loss_val = result
                    output = None
                
                loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
            else:
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if _model_has_moe(model):
                            output, _ = model(x, return_moe_losses=False)
                        else:
                            output = model(x)
                        loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
                else:
                    if _model_has_moe(model):
                        output, _ = model(x, return_moe_losses=False)
                    else:
                        output = model(x)
                    loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
            
            total_loss += loss
            
            # ================================================================
            # STREAMING TOP-K COMPUTATION (per batch, don't accumulate)
            # ================================================================
            if output is not None:
                batch_size_actual = output.shape[0]
                actual_len_dy = output.shape[1]
                output_flat = output.view(batch_size_actual * actual_len_dy, config.target_cd_cnt)
                y_flat = [item for sublist in y for item in sublist]
                dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                
                # Compute Top-K predictions ONCE (max K we need)
                max_k = 20
                top_k_preds = torch.topk(output_flat, max_k, dim=-1).indices.cpu()
                
                # Process each sample's valid days
                for j in range(batch_size_actual):
                    valid_days = min(int(dt_cnt_values[j]), actual_len_dy)
                    if valid_days <= 0:
                        continue
                    
                    start_idx = actual_len_dy * j
                    y_start = config.len_dy * j
                    
                    for day_offset in range(valid_days):
                        flat_idx = start_idx + day_offset
                        y_idx = y_start + day_offset
                        
                        if y_idx >= len(y_flat):
                            continue
                        
                        target_codes = set(y_flat[y_idx])
                        if not target_codes or target_codes == {0}:
                            continue
                        
                        pred_day = top_k_preds[flat_idx]  # [max_k]
                        
                        # Update streaming counters for each K
                        for k in [1, 5, 10, 20]:
                            pred_set = set(pred_day[:k].tolist())
                            if target_codes & pred_set:  # Any overlap = correct
                                top_k_correct[k] += 1
                            top_k_total[k] += 1
                
                total_samples += batch_size_actual
            
            # ================================================================
            # CRITICAL: Clean up batch tensors immediately
            # ================================================================
            del x, age, gender, lob, codes, output
            if 'targets_mh' in dir():
                del targets_mh
            if 'result' in dir():
                del result
            if 'extras' in dir():
                del extras
            if 'top_k_preds' in dir():
                del top_k_preds
            
            # Periodic GC for long validation loops
            if batch_idx % 50 == 0:
                gc.collect()
    
    # ========================================================================
    # COMPUTE FINAL METRICS FROM STREAMING COUNTERS
    # ========================================================================
    val_loss = total_loss / batches_to_process if batches_to_process > 0 else 0.0
    
    metrics = {'val_loss': val_loss}
    for k in [1, 5, 10, 20]:
        if top_k_total[k] > 0:
            metrics[f'top_{k}_acc'] = top_k_correct[k] / top_k_total[k]
        else:
            metrics[f'top_{k}_acc'] = 0.0
    
    return metrics
```

---

## Key Design Principles Applied

| Principle | Implementation |
|-----------|----------------|
| **Never accumulate large tensors** | Use streaming counters instead of lists |
| **Compute metrics per-batch** | Top-K computed and aggregated immediately |
| **Memory cleanup per-batch** | `del` tensors + periodic `gc.collect()` |
| **Single allocation for Top-K** | Compute `topk` once per batch, reuse for all K |

---

## Additional Changes Needed

### 1. Add `max_batches` to Validation Call (Optional Safety)

Even with the streaming solution, add a safety limit:

```python
# In run_single_experiment, around line 9376:
logger.info("  Evaluating on validation set...")
val_metrics = evaluate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    config=config,
    device=device,
    use_mixed_precision=use_mixed_precision,
    max_batches=500,  # ← Add safety limit (covers ~64K samples)
    verbose=True      # ← Enable progress logging
)
```

### 2. Clear GPU Cache Before Validation

```python
# In run_single_experiment, after train_epoch:
global_step = train_metrics['global_step']

# Clear GPU before evaluation
torch.cuda.empty_cache()
gc.collect()

logger.info("  Evaluating on training subset...")
```

---

## For Issue 1 (Lower Metrics): Increase Epochs

Change your experiment config:
```python
EPOCHS = 3  # Or 4, instead of 1
```

Or adjust the scheduler to decay slower:
```python
# Use only 80% of total_steps for cosine decay, leaving 20% at low LR
total_steps = len(train_loader) * epochs
decay_steps = int(total_steps * 0.8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps)
```

---

This streaming approach is the **optimal long-term solution** because:
1. **O(1) memory** regardless of validation set size
2. **No CPU memory exhaustion** possible
3. **Correct metrics** via streaming aggregation
4. **Production-ready** - can handle any dataset size

```python
all_predictions = []  # Line 4584
all_targets = []      # Line 4585

# In the loop (lines 4672-4673):
all_predictions.append(valid_output.cpu())  # Accumulates on CPU
all_targets.extend(valid_y)                 # Accumulates on CPU

# After loop (line 4687):
all_predictions = torch.cat(all_predictions)  # CONCATENATES EVERYTHING
```

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = False 
) -> Dict[str, float]:
    """
    Memory-efficient evaluation using streaming metrics.
    
    Key design:
    - NEVER accumulate all predictions (would exhaust CPU RAM)
    - Compute metrics per-batch and aggregate
    - Use online/streaming algorithms for Top-K accuracy
    """
    model.eval()
    nbatch = len(dataloader)
    batches_to_process = min(nbatch, max_batches) if max_batches else nbatch
    
    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and isinstance(model.module, DataParallelWrapper)
    )
    
    if batches_to_process == 0:
        return {'val_loss': 0.0, 'top_1_acc': 0.0, 'top_5_acc': 0.0, 
                'top_10_acc': 0.0, 'top_20_acc': 0.0}
    
    # ========================================================================
    # STREAMING METRICS: Accumulate counts, not tensors
    # ========================================================================
    total_loss = 0.0
    total_samples = 0
    
    # Top-K accuracy counters (streaming)
    top_k_correct = {1: 0, 5: 0, 10: 0, 20: 0}
    top_k_total = {1: 0, 5: 0, 10: 0, 20: 0}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            
            if batch_idx >= batches_to_process:
                if verbose:
                    print(f"    Early exit at batch {batch_idx}/{nbatch}")
                break
            
            # Periodic memory check
            if batch_idx % 100 == 0 and verbose:
                print(f"    Evaluating batch {batch_idx}/{batches_to_process}")
            
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            lob = batch['lob'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),
                codes
            ], dim=-1)
            
            # Forward pass
            if is_wrapped:
                targets_mh = batch['target_multihot'].to(device, non_blocking=True)
                dt_cnt_tensor = dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor) else torch.tensor(dt_cnt, device=device)
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                else:
                    result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                
                if isinstance(result, tuple):
                    loss_val, extras = result
                    output = extras.get('predictions')
                else:
                    loss_val = result
                    output = None
                
                loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
            else:
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if _model_has_moe(model):
                            output, _ = model(x, return_moe_losses=False)
                        else:
                            output = model(x)
                        loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
                else:
                    if _model_has_moe(model):
                        output, _ = model(x, return_moe_losses=False)
                    else:
                        output = model(x)
                    loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
            
            total_loss += loss
            
            # ================================================================
            # STREAMING TOP-K COMPUTATION (per batch, don't accumulate)
            # ================================================================
            if output is not None:
                batch_size_actual = output.shape[0]
                actual_len_dy = output.shape[1]
                output_flat = output.view(batch_size_actual * actual_len_dy, config.target_cd_cnt)
                y_flat = [item for sublist in y for item in sublist]
                dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                
                # Compute Top-K predictions ONCE (max K we need)
                max_k = 20
                top_k_preds = torch.topk(output_flat, max_k, dim=-1).indices.cpu()
                
                # Process each sample's valid days
                for j in range(batch_size_actual):
                    valid_days = min(int(dt_cnt_values[j]), actual_len_dy)
                    if valid_days <= 0:
                        continue
                    
                    start_idx = actual_len_dy * j
                    y_start = config.len_dy * j
                    
                    for day_offset in range(valid_days):
                        flat_idx = start_idx + day_offset
                        y_idx = y_start + day_offset
                        
                        if y_idx >= len(y_flat):
                            continue
                        
                        target_codes = set(y_flat[y_idx])
                        if not target_codes or target_codes == {0}:
                            continue
                        
                        pred_day = top_k_preds[flat_idx]  # [max_k]
                        
                        # Update streaming counters for each K
                        for k in [1, 5, 10, 20]:
                            pred_set = set(pred_day[:k].tolist())
                            if target_codes & pred_set:  # Any overlap = correct
                                top_k_correct[k] += 1
                            top_k_total[k] += 1
                
                total_samples += batch_size_actual
            
            # ================================================================
            # CRITICAL: Clean up batch tensors immediately
            # ================================================================
            del x, age, gender, lob, codes, output
            if 'targets_mh' in dir():
                del targets_mh
            if 'result' in dir():
                del result
            if 'extras' in dir():
                del extras
            if 'top_k_preds' in dir():
                del top_k_preds
            
            # Periodic GC for long validation loops
            if batch_idx % 50 == 0:
                gc.collect()
    
    # ========================================================================
    # COMPUTE FINAL METRICS FROM STREAMING COUNTERS
    # ========================================================================
    val_loss = total_loss / batches_to_process if batches_to_process > 0 else 0.0
    
    metrics = {'val_loss': val_loss}
    for k in [1, 5, 10, 20]:
        if top_k_total[k] > 0:
            metrics[f'top_{k}_acc'] = top_k_correct[k] / top_k_total[k]
        else:
            metrics[f'top_{k}_acc'] = 0.0
    
    return metrics
```

```python
# In run_single_experiment, around line 9376:
logger.info("  Evaluating on validation set...")
val_metrics = evaluate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    config=config,
    device=device,
    use_mixed_precision=use_mixed_precision,
    max_batches=500,  # ← Add safety limit (covers ~64K samples)
    verbose=True      # ← Enable progress logging
)
```

```python
# In run_single_experiment, after train_epoch:
global_step = train_metrics['global_step']

# Clear GPU before evaluation
torch.cuda.empty_cache()
gc.collect()

logger.info("  Evaluating on training subset...")
```

```python
EPOCHS = 3  # Or 4, instead of 1
```

```python
# Use only 80% of total_steps for cosine decay, leaving 20% at low LR
total_steps = len(train_loader) * epochs
decay_steps = int(total_steps * 0.8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps)
```

---

**User**

 Explain in more detial what the streaming approach really means? why it works; how it impact the metrics results i derived. Remmeber, in addiiton to accuracy, I also have other metrics to track, so your streaming solution should not sacrifice my other metrics clacualtion.  for the scheduler, Becuase in normal training I am going to have 15M records so I think 1-2 epoch is enough (is that right); so wnated to pay more attention to get an more appropriate lr scheduler strategies that fit to my cases @dev/moe/moe_flashattn_3.py 

---

**Cursor**

Let me first check what other metrics you're computing to ensure the streaming solution preserves all of them:

Good questions! Let me explain streaming in depth, then address your metrics and scheduler needs.

---

# What is Streaming Metrics Computation?

## The Problem (Accumulation Approach)

**Current code:**
```python
all_predictions = []  # Empty list
all_targets = []

for batch in dataloader:
    # Process batch...
    all_predictions.append(predictions)  # Add to list
    all_targets.extend(targets)          # Add to list

# AFTER all batches:
all_predictions = torch.cat(all_predictions)  # CONCATENATE ALL
# Compute metrics on the FULL dataset
```

**Memory usage:** `O(N)` where N = total samples

**For 15M records:**
- Predictions: 15M × 200 × 6297 × 4 bytes = **75 TB** (impossible!)
- Even with valid days only: ~7-15 TB

---

## The Solution (Streaming Approach)

**Streaming means:** Compute partial results per batch, aggregate counts (not tensors).

```python
# Initialize COUNTERS (constant memory)
total_correct = 0
total_samples = 0

for batch in dataloader:
    # Compute metric FOR THIS BATCH
    batch_correct = count_correct_in_batch(predictions, targets)
    batch_samples = len(predictions)
    
    # UPDATE COUNTERS (add scalars, not tensors)
    total_correct += batch_correct
    total_samples += batch_samples
    
    # FREE batch memory immediately
    del predictions, targets

# AFTER all batches:
final_accuracy = total_correct / total_samples
```

**Memory usage:** `O(B)` where B = batch size (constant, ~128 samples)

---

## Why Streaming Works Mathematically

### For Top-K Accuracy:

**Original formula:**
```
accuracy = (correct_1 + correct_2 + ... + correct_N) / N
```

**Streaming equivalent:**
```
accuracy = Σ(batch_correct) / Σ(batch_samples)
```

These are **mathematically identical** because addition is associative.

### For Precision@K, Recall@K, mAP@K:

These are all **averages** which can be computed as:
```
mean = Σ(batch_sums) / Σ(batch_counts)
```

**The key insight:** Any metric that can be expressed as:
- Sum of counts → streamable
- Average of values → streamable (track sum + count)
- Weighted average → streamable (track weighted_sum + total_weight)

---

## Metrics That ARE Streamable (Your Metrics)

| Metric | Formula | Streaming? | How |
|--------|---------|------------|-----|
| **Top-K Accuracy** | correct / total | ✅ | Sum counters |
| **Recall@K** | correct / total | ✅ | Sum counters |
| **Precision@K** | mean(per_sample_precision) | ✅ | Sum + count |
| **mAP@K** | mean(AP_per_sample) | ✅ | Sum + count |
| **Brier Score** | mean(squared_error) | ✅ | Sum + count |
| **Loss** | mean(batch_loss) | ✅ | Sum + count |

---

# Complete Streaming `evaluate()` with ALL Your Metrics

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = False 
) -> Dict[str, float]:
    """
    Memory-efficient streaming evaluation.
    
    Computes all metrics without accumulating predictions:
    - Loss
    - Top-K accuracy (K=1,5,10,20)
    - Recall@K (K=5,10,20,50)  
    - Precision@K (K=5,10,20,50)
    - mAP@K (K=20,50)
    - Brier score
    
    Memory: O(batch_size) constant, regardless of dataset size.
    """
    model.eval()
    nbatch = len(dataloader)
    batches_to_process = min(nbatch, max_batches) if max_batches else nbatch
    
    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and isinstance(model.module, DataParallelWrapper)
    )
    
    if batches_to_process == 0:
        return _empty_metrics()
    
    # ========================================================================
    # STREAMING ACCUMULATORS (constant memory)
    # ========================================================================
    total_loss = 0.0
    num_batches = 0
    
    # Top-K accuracy
    topk_correct = {1: 0, 5: 0, 10: 0, 20: 0}
    topk_total = {1: 0, 5: 0, 10: 0, 20: 0}
    
    # Recall@K (same structure as top-k)
    recall_correct = {5: 0, 10: 0, 20: 0, 50: 0}
    recall_total = {5: 0, 10: 0, 20: 0, 50: 0}
    
    # Precision@K (sum of precisions + count)
    precision_sum = {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
    precision_count = {5: 0, 10: 0, 20: 0, 50: 0}
    
    # mAP@K (sum of APs + count)
    map_sum = {20: 0.0, 50: 0.0}
    map_count = {20: 0, 50: 0}
    
    # Brier score (sum + count)
    brier_sum = 0.0
    brier_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            
            if batch_idx >= batches_to_process:
                break
            
            if verbose and batch_idx % 100 == 0:
                print(f"    Eval batch {batch_idx}/{batches_to_process}")
            
            # ================================================================
            # FORWARD PASS
            # ================================================================
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            lob = batch['lob'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),
                codes
            ], dim=-1)
            
            if is_wrapped:
                targets_mh = batch['target_multihot'].to(device, non_blocking=True)
                dt_cnt_tensor = dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor) else torch.tensor(dt_cnt, device=device)
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                else:
                    result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                
                if isinstance(result, tuple):
                    loss_val, extras = result
                    output = extras.get('predictions')
                else:
                    loss_val = result
                    output = None
                
                batch_loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
            else:
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if _model_has_moe(model):
                            output, _ = model(x, return_moe_losses=False)
                        else:
                            output = model(x)
                        batch_loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
                else:
                    if _model_has_moe(model):
                        output, _ = model(x, return_moe_losses=False)
                    else:
                        output = model(x)
                    batch_loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
            
            total_loss += batch_loss
            num_batches += 1
            
            # ================================================================
            # STREAMING METRICS COMPUTATION
            # ================================================================
            if output is not None:
                batch_size_actual = output.shape[0]
                actual_len_dy = output.shape[1]
                
                # Sort predictions once (reuse for all metrics)
                output_flat = output.view(batch_size_actual * actual_len_dy, config.target_cd_cnt)
                sorted_indices = torch.argsort(output_flat, dim=-1, descending=True)
                probs = torch.sigmoid(output_flat)  # For Brier score
                
                y_flat = [item for sublist in y for item in sublist]
                dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                
                # Process each sample
                for j in range(batch_size_actual):
                    valid_days = min(int(dt_cnt_values[j]), actual_len_dy)
                    if valid_days <= 0:
                        continue
                    
                    start_idx = actual_len_dy * j
                    y_start = config.len_dy * j
                    
                    for day_offset in range(valid_days):
                        flat_idx = start_idx + day_offset
                        y_idx = y_start + day_offset
                        
                        if y_idx >= len(y_flat):
                            continue
                        
                        target_codes = y_flat[y_idx]
                        true_codes = set(c for c in target_codes if c != 0)
                        
                        if not true_codes:
                            continue
                        
                        sorted_preds = sorted_indices[flat_idx].cpu()
                        
                        # --- Top-K and Recall (any overlap = correct) ---
                        for k in [1, 5, 10, 20]:
                            pred_set = set(sorted_preds[:k].tolist())
                            if true_codes & pred_set:
                                topk_correct[k] += 1
                            topk_total[k] += 1
                        
                        for k in [5, 10, 20, 50]:
                            pred_set = set(sorted_preds[:k].tolist())
                            if true_codes & pred_set:
                                recall_correct[k] += 1
                            recall_total[k] += 1
                        
                        # --- Precision@K ---
                        for k in [5, 10, 20, 50]:
                            pred_list = sorted_preds[:k].tolist()
                            hits = sum(1 for c in pred_list if c in true_codes)
                            precision_sum[k] += hits / k
                            precision_count[k] += 1
                        
                        # --- mAP@K ---
                        for k in [20, 50]:
                            hits = 0
                            precisions_at_rank = []
                            for rank, pred_code in enumerate(sorted_preds[:k].tolist(), 1):
                                if pred_code in true_codes:
                                    hits += 1
                                    precisions_at_rank.append(hits / rank)
                            if precisions_at_rank:
                                map_sum[k] += np.mean(precisions_at_rank)
                                map_count[k] += 1
                        
                        # --- Brier Score ---
                        day_probs = probs[flat_idx].cpu().numpy()
                        targets_binary = np.zeros(config.target_cd_cnt)
                        for c in true_codes:
                            if c < config.target_cd_cnt:
                                targets_binary[c] = 1.0
                        brier_sum += np.mean((day_probs - targets_binary) ** 2)
                        brier_count += 1
            
            # ================================================================
            # CLEANUP (critical for memory)
            # ================================================================
            del x, age, gender, lob, codes, output, sorted_indices
            if 'probs' in dir():
                del probs
            if 'targets_mh' in dir():
                del targets_mh
            if 'result' in dir():
                del result
            if 'extras' in dir():
                del extras
            
            if batch_idx % 50 == 0:
                gc.collect()
    
    # ========================================================================
    # AGGREGATE FINAL METRICS FROM STREAMING ACCUMULATORS
    # ========================================================================
    metrics = {
        'val_loss': total_loss / num_batches if num_batches > 0 else 0.0,
    }
    
    # Top-K accuracy
    for k in [1, 5, 10, 20]:
        metrics[f'top_{k}_acc'] = topk_correct[k] / topk_total[k] if topk_total[k] > 0 else 0.0
    
    # Recall@K
    for k in [5, 10, 20, 50]:
        metrics[f'recall@{k}'] = recall_correct[k] / recall_total[k] if recall_total[k] > 0 else 0.0
    
    # Precision@K
    for k in [5, 10, 20, 50]:
        metrics[f'precision@{k}'] = precision_sum[k] / precision_count[k] if precision_count[k] > 0 else 0.0
    
    # mAP@K
    for k in [20, 50]:
        metrics[f'mAP@{k}'] = map_sum[k] / map_count[k] if map_count[k] > 0 else 0.0
    
    # Brier score
    metrics['brier_score'] = brier_sum / brier_count if brier_count > 0 else 0.0
    
    return metrics


def _empty_metrics():
    """Return empty metrics dict."""
    return {
        'val_loss': 0.0,
        'top_1_acc': 0.0, 'top_5_acc': 0.0, 'top_10_acc': 0.0, 'top_20_acc': 0.0,
        'recall@5': 0.0, 'recall@10': 0.0, 'recall@20': 0.0, 'recall@50': 0.0,
        'precision@5': 0.0, 'precision@10': 0.0, 'precision@20': 0.0, 'precision@50': 0.0,
        'mAP@20': 0.0, 'mAP@50': 0.0,
        'brier_score': 0.0
    }
```

---

# Streaming vs Accumulation: No Metric Sacrifice

| Metric | Accumulation Result | Streaming Result | Difference |
|--------|---------------------|------------------|------------|
| Top-K Acc | `sum(correct)/sum(total)` | `sum(correct)/sum(total)` | **Identical** |
| Recall@K | `sum(hits)/sum(samples)` | `sum(hits)/sum(samples)` | **Identical** |
| Precision@K | `mean(precisions)` | `sum(precisions)/count` | **Identical** |
| mAP@K | `mean(APs)` | `sum(APs)/count` | **Identical** |
| Brier | `mean(squared_err)` | `sum(squared_err)/count` | **Identical** |

**The results are mathematically identical** because all your metrics are aggregatable.

---

# LR Scheduler for 15M Records, 1-2 Epochs

## The Challenge

With 15M records:
- Batch size 128 → ~117,000 batches per epoch
- 1-2 epochs → 117K-234K total steps

**Pure cosine decay** takes LR to near-zero by the end. With only 1-2 epochs, this can hurt:
- Model still learning at end, but LR too low
- Late-epoch samples train with very low LR (unfair)

## Recommended: Warmup + Constant + Cosine Decay

```
LR Schedule:
|   warmup   |      constant      |    cosine decay    |
0          5%                   80%                 100%
```

**Implementation:**

```python
def get_warmup_constant_cosine_scheduler(
    optimizer, 
    total_steps: int,
    warmup_fraction: float = 0.05,
    constant_fraction: float = 0.75,
    min_lr_fraction: float = 0.1  # Don't decay below 10% of max LR
):
    """
    Three-phase scheduler for large-scale 1-2 epoch training.
    
    Phase 1 (warmup): 0-5% - Linear warmup
    Phase 2 (constant): 5-80% - Constant max LR  
    Phase 3 (decay): 80-100% - Cosine decay to min_lr
    
    This keeps LR high through most of training, only decaying
    at the end for stability.
    """
    warmup_steps = int(total_steps * warmup_fraction)
    constant_end = int(total_steps * (warmup_fraction + constant_fraction))
    decay_steps = total_steps - constant_end
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        elif step < constant_end:
            # Constant LR
            return 1.0
        else:
            # Cosine decay to min_lr
            decay_progress = (step - constant_end) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return min_lr_fraction + (1 - min_lr_fraction) * cosine_decay
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Usage:**

```python
# In run_single_experiment:
optimizer = optim.AdamW(
    model.parameters(),
    lr=scaled_lr,  # e.g., 4e-4 for 4 GPUs
    weight_decay=config.weight_decay
)

total_steps = len(train_loader) * epochs
scheduler = get_warmup_constant_cosine_scheduler(
    optimizer,
    total_steps=total_steps,
    warmup_fraction=0.05,    # 5% warmup (~6K steps)
    constant_fraction=0.75,   # 75% constant (~88K steps)  
    min_lr_fraction=0.1       # Final LR = 10% of max
)

logger.info(f"Scheduler: Warmup-Constant-Cosine over {total_steps} steps")
logger.info(f"  Warmup: {int(total_steps * 0.05)} steps")
logger.info(f"  Constant: {int(total_steps * 0.75)} steps")
logger.info(f"  Decay: {int(total_steps * 0.20)} steps to {0.1*100}% of max LR")
```

---

## Alternative: Linear Warmup + Linear Decay

Simpler but also effective:

```python
def get_warmup_linear_decay_scheduler(
    optimizer, 
    total_steps: int,
    warmup_fraction: float = 0.1,
    min_lr_fraction: float = 0.0
):
    """
    Linear warmup + linear decay.
    Simpler alternative, works well for 1-2 epoch training.
    """
    warmup_steps = int(total_steps * warmup_fraction)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(min_lr_fraction, 1.0 - progress * (1.0 - min_lr_fraction))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## Comparison: Your Options

| Scheduler | When LR is High | Final LR | Best For |
|-----------|-----------------|----------|----------|
| Pure Cosine | First 50% | ~0 | Multi-epoch training |
| Warmup + Constant + Cosine | First 80% | 10% of max | **Your case (1-2 epochs)** |
| Warmup + Linear Decay | First 10% then decreasing | Configurable | Simple baseline |

---

## Is 1-2 Epochs Enough for 15M Records?

**Yes, likely sufficient because:**

1. **Per-epoch samples:** 15M samples seen per epoch
2. **Your current batch 10900 metrics:** R@10=83%, R@20=88% - still improving but convergence visible
3. **Large models converge faster with more data** - diminishing returns after 1-2 passes

**Recommendation:** 
- Start with 1 epoch + warmup-constant-cosine
- If metrics plateau early (last 20% of training shows <1% improvement), 1 epoch is enough
- If still improving at end, try 2 epochs

```python
all_predictions = []  # Empty list
all_targets = []

for batch in dataloader:
    # Process batch...
    all_predictions.append(predictions)  # Add to list
    all_targets.extend(targets)          # Add to list

# AFTER all batches:
all_predictions = torch.cat(all_predictions)  # CONCATENATE ALL
# Compute metrics on the FULL dataset
```

```python
# Initialize COUNTERS (constant memory)
total_correct = 0
total_samples = 0

for batch in dataloader:
    # Compute metric FOR THIS BATCH
    batch_correct = count_correct_in_batch(predictions, targets)
    batch_samples = len(predictions)
    
    # UPDATE COUNTERS (add scalars, not tensors)
    total_correct += batch_correct
    total_samples += batch_samples
    
    # FREE batch memory immediately
    del predictions, targets

# AFTER all batches:
final_accuracy = total_correct / total_samples
```

```plaintext
accuracy = (correct_1 + correct_2 + ... + correct_N) / N
```

```plaintext
accuracy = Σ(batch_correct) / Σ(batch_samples)
```

```plaintext
mean = Σ(batch_sums) / Σ(batch_counts)
```

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = False 
) -> Dict[str, float]:
    """
    Memory-efficient streaming evaluation.
    
    Computes all metrics without accumulating predictions:
    - Loss
    - Top-K accuracy (K=1,5,10,20)
    - Recall@K (K=5,10,20,50)  
    - Precision@K (K=5,10,20,50)
    - mAP@K (K=20,50)
    - Brier score
    
    Memory: O(batch_size) constant, regardless of dataset size.
    """
    model.eval()
    nbatch = len(dataloader)
    batches_to_process = min(nbatch, max_batches) if max_batches else nbatch
    
    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and isinstance(model.module, DataParallelWrapper)
    )
    
    if batches_to_process == 0:
        return _empty_metrics()
    
    # ========================================================================
    # STREAMING ACCUMULATORS (constant memory)
    # ========================================================================
    total_loss = 0.0
    num_batches = 0
    
    # Top-K accuracy
    topk_correct = {1: 0, 5: 0, 10: 0, 20: 0}
    topk_total = {1: 0, 5: 0, 10: 0, 20: 0}
    
    # Recall@K (same structure as top-k)
    recall_correct = {5: 0, 10: 0, 20: 0, 50: 0}
    recall_total = {5: 0, 10: 0, 20: 0, 50: 0}
    
    # Precision@K (sum of precisions + count)
    precision_sum = {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
    precision_count = {5: 0, 10: 0, 20: 0, 50: 0}
    
    # mAP@K (sum of APs + count)
    map_sum = {20: 0.0, 50: 0.0}
    map_count = {20: 0, 50: 0}
    
    # Brier score (sum + count)
    brier_sum = 0.0
    brier_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            
            if batch_idx >= batches_to_process:
                break
            
            if verbose and batch_idx % 100 == 0:
                print(f"    Eval batch {batch_idx}/{batches_to_process}")
            
            # ================================================================
            # FORWARD PASS
            # ================================================================
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            lob = batch['lob'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),
                codes
            ], dim=-1)
            
            if is_wrapped:
                targets_mh = batch['target_multihot'].to(device, non_blocking=True)
                dt_cnt_tensor = dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor) else torch.tensor(dt_cnt, device=device)
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                else:
                    result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                
                if isinstance(result, tuple):
                    loss_val, extras = result
                    output = extras.get('predictions')
                else:
                    loss_val = result
                    output = None
                
                batch_loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
            else:
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if _model_has_moe(model):
                            output, _ = model(x, return_moe_losses=False)
                        else:
                            output = model(x)
                        batch_loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
                else:
                    if _model_has_moe(model):
                        output, _ = model(x, return_moe_losses=False)
                    else:
                        output = model(x)
                    batch_loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
            
            total_loss += batch_loss
            num_batches += 1
            
            # ================================================================
            # STREAMING METRICS COMPUTATION
            # ================================================================
            if output is not None:
                batch_size_actual = output.shape[0]
                actual_len_dy = output.shape[1]
                
                # Sort predictions once (reuse for all metrics)
                output_flat = output.view(batch_size_actual * actual_len_dy, config.target_cd_cnt)
                sorted_indices = torch.argsort(output_flat, dim=-1, descending=True)
                probs = torch.sigmoid(output_flat)  # For Brier score
                
                y_flat = [item for sublist in y for item in sublist]
                dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                
                # Process each sample
                for j in range(batch_size_actual):
                    valid_days = min(int(dt_cnt_values[j]), actual_len_dy)
                    if valid_days <= 0:
                        continue
                    
                    start_idx = actual_len_dy * j
                    y_start = config.len_dy * j
                    
                    for day_offset in range(valid_days):
                        flat_idx = start_idx + day_offset
                        y_idx = y_start + day_offset
                        
                        if y_idx >= len(y_flat):
                            continue
                        
                        target_codes = y_flat[y_idx]
                        true_codes = set(c for c in target_codes if c != 0)
                        
                        if not true_codes:
                            continue
                        
                        sorted_preds = sorted_indices[flat_idx].cpu()
                        
                        # --- Top-K and Recall (any overlap = correct) ---
                        for k in [1, 5, 10, 20]:
                            pred_set = set(sorted_preds[:k].tolist())
                            if true_codes & pred_set:
                                topk_correct[k] += 1
                            topk_total[k] += 1
                        
                        for k in [5, 10, 20, 50]:
                            pred_set = set(sorted_preds[:k].tolist())
                            if true_codes & pred_set:
                                recall_correct[k] += 1
                            recall_total[k] += 1
                        
                        # --- Precision@K ---
                        for k in [5, 10, 20, 50]:
                            pred_list = sorted_preds[:k].tolist()
                            hits = sum(1 for c in pred_list if c in true_codes)
                            precision_sum[k] += hits / k
                            precision_count[k] += 1
                        
                        # --- mAP@K ---
                        for k in [20, 50]:
                            hits = 0
                            precisions_at_rank = []
                            for rank, pred_code in enumerate(sorted_preds[:k].tolist(), 1):
                                if pred_code in true_codes:
                                    hits += 1
                                    precisions_at_rank.append(hits / rank)
                            if precisions_at_rank:
                                map_sum[k] += np.mean(precisions_at_rank)
                                map_count[k] += 1
                        
                        # --- Brier Score ---
                        day_probs = probs[flat_idx].cpu().numpy()
                        targets_binary = np.zeros(config.target_cd_cnt)
                        for c in true_codes:
                            if c < config.target_cd_cnt:
                                targets_binary[c] = 1.0
                        brier_sum += np.mean((day_probs - targets_binary) ** 2)
                        brier_count += 1
            
            # ================================================================
            # CLEANUP (critical for memory)
            # ================================================================
            del x, age, gender, lob, codes, output, sorted_indices
            if 'probs' in dir():
                del probs
            if 'targets_mh' in dir():
                del targets_mh
            if 'result' in dir():
                del result
            if 'extras' in dir():
                del extras
            
            if batch_idx % 50 == 0:
                gc.collect()
    
    # ========================================================================
    # AGGREGATE FINAL METRICS FROM STREAMING ACCUMULATORS
    # ========================================================================
    metrics = {
        'val_loss': total_loss / num_batches if num_batches > 0 else 0.0,
    }
    
    # Top-K accuracy
    for k in [1, 5, 10, 20]:
        metrics[f'top_{k}_acc'] = topk_correct[k] / topk_total[k] if topk_total[k] > 0 else 0.0
    
    # Recall@K
    for k in [5, 10, 20, 50]:
        metrics[f'recall@{k}'] = recall_correct[k] / recall_total[k] if recall_total[k] > 0 else 0.0
    
    # Precision@K
    for k in [5, 10, 20, 50]:
        metrics[f'precision@{k}'] = precision_sum[k] / precision_count[k] if precision_count[k] > 0 else 0.0
    
    # mAP@K
    for k in [20, 50]:
        metrics[f'mAP@{k}'] = map_sum[k] / map_count[k] if map_count[k] > 0 else 0.0
    
    # Brier score
    metrics['brier_score'] = brier_sum / brier_count if brier_count > 0 else 0.0
    
    return metrics


def _empty_metrics():
    """Return empty metrics dict."""
    return {
        'val_loss': 0.0,
        'top_1_acc': 0.0, 'top_5_acc': 0.0, 'top_10_acc': 0.0, 'top_20_acc': 0.0,
        'recall@5': 0.0, 'recall@10': 0.0, 'recall@20': 0.0, 'recall@50': 0.0,
        'precision@5': 0.0, 'precision@10': 0.0, 'precision@20': 0.0, 'precision@50': 0.0,
        'mAP@20': 0.0, 'mAP@50': 0.0,
        'brier_score': 0.0
    }
```

```plaintext
LR Schedule:
|   warmup   |      constant      |    cosine decay    |
0          5%                   80%                 100%
```

```python
def get_warmup_constant_cosine_scheduler(
    optimizer, 
    total_steps: int,
    warmup_fraction: float = 0.05,
    constant_fraction: float = 0.75,
    min_lr_fraction: float = 0.1  # Don't decay below 10% of max LR
):
    """
    Three-phase scheduler for large-scale 1-2 epoch training.
    
    Phase 1 (warmup): 0-5% - Linear warmup
    Phase 2 (constant): 5-80% - Constant max LR  
    Phase 3 (decay): 80-100% - Cosine decay to min_lr
    
    This keeps LR high through most of training, only decaying
    at the end for stability.
    """
    warmup_steps = int(total_steps * warmup_fraction)
    constant_end = int(total_steps * (warmup_fraction + constant_fraction))
    decay_steps = total_steps - constant_end
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        elif step < constant_end:
            # Constant LR
            return 1.0
        else:
            # Cosine decay to min_lr
            decay_progress = (step - constant_end) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return min_lr_fraction + (1 - min_lr_fraction) * cosine_decay
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

```python
# In run_single_experiment:
optimizer = optim.AdamW(
    model.parameters(),
    lr=scaled_lr,  # e.g., 4e-4 for 4 GPUs
    weight_decay=config.weight_decay
)

total_steps = len(train_loader) * epochs
scheduler = get_warmup_constant_cosine_scheduler(
    optimizer,
    total_steps=total_steps,
    warmup_fraction=0.05,    # 5% warmup (~6K steps)
    constant_fraction=0.75,   # 75% constant (~88K steps)  
    min_lr_fraction=0.1       # Final LR = 10% of max
)

logger.info(f"Scheduler: Warmup-Constant-Cosine over {total_steps} steps")
logger.info(f"  Warmup: {int(total_steps * 0.05)} steps")
logger.info(f"  Constant: {int(total_steps * 0.75)} steps")
logger.info(f"  Decay: {int(total_steps * 0.20)} steps to {0.1*100}% of max LR")
```

```python
def get_warmup_linear_decay_scheduler(
    optimizer, 
    total_steps: int,
    warmup_fraction: float = 0.1,
    min_lr_fraction: float = 0.0
):
    """
    Linear warmup + linear decay.
    Simpler alternative, works well for 1-2 epoch training.
    """
    warmup_steps = int(total_steps * warmup_fraction)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(min_lr_fraction, 1.0 - progress * (1.0 - min_lr_fraction))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

**User**

the way you impelemnt new evaluation funciton looks very cumbersome,revise the impelemantion to be more pythonic, huggingface, pytorch style without losing any metrics I have newly defined and replaced several metrics in @dev/moe/moe_flashattn_3.py and you need to udnerstand the most recent updates @docs/pss/evaluation/internal_metrics_review_2.md. Then, I have implemetned the streamingmetrics, continue iwth teh impelmantions for evaluate and other related functions that needs to be adjysted. also in addiiton to modificaiton in evaluation function, are there any other functions or calsses need to be changed to accomodate the memory efficient streaming evlauation? Provides step by step entire guide for modificaiton. 
Also for the change of lr scheduler, show me all the places I neded to modifcy teh code accordingly

---

**Cursor**

I'll start by reviewing the recent metric updates and the current implementation to provide you with a comprehensive modification guide.

Let me continue reading the current `evaluate()` function and the LR scheduler implementation:

Now I have a complete picture. Here's a comprehensive step-by-step guide for implementing the memory-efficient streaming evaluation in a more Pythonic, HuggingFace/PyTorch style.

---

# Comprehensive Step-by-Step Implementation Guide

## Overview of Changes

| Component | Current State | Changes Needed |
|-----------|--------------|----------------|
| `StreamingMetrics` | Has mAP, old Brier | Update to match new metrics: micro-recall, NDCG, positive-brier |
| `evaluate()` | Accumulates tensors on CPU | Rewrite using `StreamingMetrics` |
| `comprehensive_evaluation()` | Uses old metric functions | Update to use new functions |
| LR Scheduler | Step-based inside `train_epoch` | Add warmup + move to epoch-based option |
| Helper functions | Various metric functions | Already updated per `internal_metrics_review_2.md` |

---

## STEP 1: Update `StreamingMetrics` Class (Replace entirely)

**Location:** Lines 7239-7404

Replace the current `StreamingMetrics` class with this Pythonic, HuggingFace-style implementation:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch

@dataclass
class StreamingMetricsState:
    """Internal state for streaming metrics computation."""
    # Loss
    total_loss: float = 0.0
    num_batches: int = 0
    
    # Sample counts
    num_samples: int = 0
    
    # Recall@K counters (binary: any hit in top-K)
    recall_hits: Dict[int, int] = field(default_factory=dict)
    recall_total: Dict[int, int] = field(default_factory=dict)
    
    # Micro-Recall@K (per-code: total hits / total true codes)
    micro_recall_hits: Dict[int, int] = field(default_factory=dict)
    micro_recall_true: Dict[int, int] = field(default_factory=dict)
    
    # Precision@K (sum for average)
    precision_sum: Dict[int, float] = field(default_factory=dict)
    precision_count: Dict[int, int] = field(default_factory=dict)
    
    # NDCG@K
    ndcg_sum: Dict[int, float] = field(default_factory=dict)
    ndcg_count: Dict[int, int] = field(default_factory=dict)
    
    # MRR (best-ranked true code)
    mrr_sum: float = 0.0
    mrr_count: int = 0
    
    # Positive-only Brier
    positive_brier_sum: float = 0.0
    positive_brier_count: int = 0


class StreamingMetrics:
    """
    Memory-efficient streaming metrics aggregator for clinical code prediction.
    
    Follows HuggingFace/PyTorch patterns:
    - `.reset()` clears state
    - `.update(batch)` processes one batch  
    - `.compute()` returns final metrics
    
    Key design principles:
    1. NEVER accumulate full predictions (prevents CPU OOM)
    2. Compute metrics incrementally per-batch
    3. Use only scalar counters, not tensors
    
    Example:
        metrics = StreamingMetrics(k_values=[5, 10, 20])
        for batch in dataloader:
            predictions, targets = forward(batch)
            metrics.update(predictions, targets)
        results = metrics.compute()
    """
    
    def __init__(
        self, 
        k_values: Tuple[int, ...] = (5, 10, 20, 50),
        compute_mrr: bool = True,
        compute_brier: bool = True,
        vocab_size: int = 6297
    ):
        """
        Args:
            k_values: K values for Recall@K, Precision@K, Micro-Recall@K, NDCG@K
            compute_mrr: Whether to compute Mean Reciprocal Rank
            compute_brier: Whether to compute Positive-Only Brier Score
            vocab_size: Target vocabulary size (for Brier calculation)
        """
        self.k_values = k_values
        self.compute_mrr = compute_mrr
        self.compute_brier = compute_brier
        self.vocab_size = vocab_size
        self._max_k = max(k_values)
        
        # Precompute NDCG discount factors: 1/log2(rank+2)
        self._discounts = 1.0 / np.log2(np.arange(2, self._max_k + 2))
        
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulators for a new evaluation pass."""
        self._state = StreamingMetricsState(
            recall_hits={k: 0 for k in self.k_values},
            recall_total={k: 0 for k in self.k_values},
            micro_recall_hits={k: 0 for k in self.k_values},
            micro_recall_true={k: 0 for k in self.k_values},
            precision_sum={k: 0.0 for k in self.k_values},
            precision_count={k: 0 for k in self.k_values},
            ndcg_sum={k: 0.0 for k in self.k_values},
            ndcg_count={k: 0 for k in self.k_values},
        )
    
    def update_loss(self, loss: float) -> None:
        """Accumulate loss from a batch."""
        self._state.total_loss += loss
        self._state.num_batches += 1
    
    def update(
        self,
        predictions: torch.Tensor,  # [batch_size, vocab_size] logits
        targets: List[List[int]],    # List of target code lists per sample
        probs: Optional[torch.Tensor] = None  # [batch_size, vocab_size] for Brier
    ) -> None:
        """
        Update metrics with a batch of predictions.
        
        Args:
            predictions: Model logits [batch_size, vocab_size] (will be moved to CPU)
            targets: List of target code lists (one per sample)
            probs: Optional sigmoid probabilities for Brier score
        """
        # Move to CPU and sort once (most expensive operation)
        with torch.no_grad():
            sorted_indices = torch.argsort(predictions, dim=-1, descending=True).cpu()
            if probs is None and self.compute_brier:
                probs = torch.sigmoid(predictions).cpu()
            elif probs is not None:
                probs = probs.cpu()
        
        batch_size = len(targets)
        self._state.num_samples += batch_size
        
        for i, target_codes in enumerate(targets):
            # Filter valid codes (non-zero)
            true_codes = set(c for c in target_codes if c != 0)
            if not true_codes:
                continue
            
            pred_indices = sorted_indices[i]
            num_true = len(true_codes)
            
            # === Recall@K, Micro-Recall@K, Precision@K ===
            for k in self.k_values:
                pred_set = set(pred_indices[:k].tolist())
                hits = len(true_codes & pred_set)
                
                # Recall@K (binary: any hit = success)
                if hits > 0:
                    self._state.recall_hits[k] += 1
                self._state.recall_total[k] += 1
                
                # Micro-Recall@K (per-code granularity)
                self._state.micro_recall_hits[k] += hits
                self._state.micro_recall_true[k] += num_true
                
                # Precision@K
                self._state.precision_sum[k] += hits / k
                self._state.precision_count[k] += 1
            
            # === NDCG@K ===
            for k in self.k_values:
                top_k_preds = pred_indices[:k].tolist()
                dcg = sum(
                    self._discounts[rank] 
                    for rank, pred in enumerate(top_k_preds) 
                    if pred in true_codes
                )
                num_relevant = min(num_true, k)
                idcg = sum(self._discounts[:num_relevant])
                ndcg = dcg / idcg if idcg > 0 else 0.0
                self._state.ndcg_sum[k] += ndcg
                self._state.ndcg_count[k] += 1
            
            # === MRR (best-ranked true code) ===
            if self.compute_mrr:
                best_rank = self._max_k  # Default to max if not found
                for code in true_codes:
                    rank_indices = (pred_indices[:self._max_k] == code).nonzero(as_tuple=True)[0]
                    if len(rank_indices) > 0:
                        rank = rank_indices[0].item()
                        if rank < best_rank:
                            best_rank = rank
                if best_rank < self._max_k:
                    self._state.mrr_sum += 1.0 / (best_rank + 1)
                self._state.mrr_count += 1
            
            # === Positive-Only Brier Score ===
            if self.compute_brier and probs is not None:
                for code in true_codes:
                    if 0 < code < self.vocab_size:
                        pred_prob = probs[i, code].item()
                        self._state.positive_brier_sum += (pred_prob - 1.0) ** 2
                        self._state.positive_brier_count += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated state.
        
        Returns:
            Dict with all computed metrics
        """
        metrics = {}
        
        # Loss
        metrics['val_loss'] = (
            self._state.total_loss / max(self._state.num_batches, 1)
        )
        
        # Recall@K
        for k in self.k_values:
            metrics[f'recall@{k}'] = (
                self._state.recall_hits[k] / self._state.recall_total[k]
                if self._state.recall_total[k] > 0 else 0.0
            )
        
        # Micro-Recall@K
        for k in self.k_values:
            metrics[f'micro_recall@{k}'] = (
                self._state.micro_recall_hits[k] / self._state.micro_recall_true[k]
                if self._state.micro_recall_true[k] > 0 else 0.0
            )
        
        # Precision@K
        for k in self.k_values:
            metrics[f'precision@{k}'] = (
                self._state.precision_sum[k] / self._state.precision_count[k]
                if self._state.precision_count[k] > 0 else 0.0
            )
        
        # NDCG@K
        for k in self.k_values:
            metrics[f'ndcg@{k}'] = (
                self._state.ndcg_sum[k] / self._state.ndcg_count[k]
                if self._state.ndcg_count[k] > 0 else 0.0
            )
        
        # MRR
        if self.compute_mrr:
            metrics['mrr'] = (
                self._state.mrr_sum / self._state.mrr_count
                if self._state.mrr_count > 0 else 0.0
            )
        
        # Positive-Only Brier
        if self.compute_brier:
            metrics['positive_brier'] = (
                self._state.positive_brier_sum / self._state.positive_brier_count
                if self._state.positive_brier_count > 0 else 0.0
            )
        
        # Metadata
        metrics['num_samples'] = self._state.num_samples
        metrics['num_batches'] = self._state.num_batches
        
        return metrics
    
    def __repr__(self) -> str:
        return (
            f"StreamingMetrics("
            f"k_values={self.k_values}, "
            f"samples={self._state.num_samples}, "
            f"batches={self._state.num_batches})"
        )
```

---

## STEP 2: Rewrite `evaluate()` Function (Replace entirely)

**Location:** Lines 4547-4711

Replace with this clean, Pythonic implementation:

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = False,
    k_values: Tuple[int, ...] = (1, 5, 10, 20)
) -> Dict[str, float]:
    """
    Memory-efficient evaluation using streaming metrics.
    
    Key design:
    - Uses StreamingMetrics for incremental computation
    - NEVER accumulates predictions (prevents CPU OOM on large val sets)
    - Computes recall@K, micro-recall@K, precision@K, NDCG@K, MRR, positive-Brier
    
    Args:
        model: Model to evaluate (can be wrapped in DataParallel/DataParallelWrapper)
        dataloader: Validation DataLoader
        criterion: Loss function (e.g., BCEWithLogitsLoss)
        config: Model configuration
        device: Device for computation
        use_mixed_precision: Whether to use FP16 autocast
        max_batches: If set, only evaluate first N batches (for train subset)
        verbose: Print progress every 100 batches
        k_values: K values for top-K metrics
    
    Returns:
        Dict with val_loss, recall@K, micro_recall@K, precision@K, ndcg@K, mrr, positive_brier
    """
    model.eval()
    
    # Setup
    num_batches = len(dataloader)
    batches_to_process = min(num_batches, max_batches) if max_batches else num_batches
    
    if batches_to_process == 0:
        return _get_empty_eval_metrics(k_values)
    
    # Detect wrapper type
    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and 
        isinstance(model.module, DataParallelWrapper)
    )
    
    # Initialize streaming metrics
    metrics_tracker = StreamingMetrics(
        k_values=k_values,
        compute_mrr=True,
        compute_brier=True,
        vocab_size=config.target_cd_cnt
    )
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= batches_to_process:
                break
            
            if verbose and batch_idx % 100 == 0:
                print(f"    Evaluating batch {batch_idx}/{batches_to_process}")
            
            # Forward pass
            output, loss = _forward_batch(
                model, batch, config, device, 
                criterion, use_mixed_precision, is_wrapped
            )
            
            # Update streaming metrics
            metrics_tracker.update_loss(loss)
            
            if output is not None:
                _update_streaming_metrics(
                    metrics_tracker, output, batch, config
                )
            
            # Cleanup batch tensors immediately
            del output
            if batch_idx % 50 == 0:
                gc.collect()
    
    # Compute and return final metrics
    results = metrics_tracker.compute()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


def _get_empty_eval_metrics(k_values: Tuple[int, ...]) -> Dict[str, float]:
    """Return empty metrics dict when no data to evaluate."""
    metrics = {'val_loss': 0.0, 'mrr': 0.0, 'positive_brier': 0.0}
    for k in k_values:
        metrics[f'recall@{k}'] = 0.0
        metrics[f'micro_recall@{k}'] = 0.0
        metrics[f'precision@{k}'] = 0.0
        metrics[f'ndcg@{k}'] = 0.0
    return metrics


def _forward_batch(
    model: nn.Module,
    batch: Dict,
    config: BaseConfig,
    device: torch.device,
    criterion: nn.Module,
    use_mixed_precision: bool,
    is_wrapped: bool
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Execute forward pass for a single batch.
    
    Returns:
        (output_logits, loss_scalar)
    """
    # Prepare inputs
    age = batch['age'].to(device, non_blocking=True)
    gender = batch['gender'].to(device, non_blocking=True)
    lob = batch['lob'].to(device, non_blocking=True)
    codes = batch['codes'].to(device, non_blocking=True)
    dt_cnt = batch['dt_cnt']
    y = batch['target']
    
    x = torch.cat([
        age.unsqueeze(-1),
        gender.unsqueeze(-1),
        lob.unsqueeze(-1),
        codes
    ], dim=-1)
    
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16) 
        if use_mixed_precision else nullcontext()
    )
    
    with autocast_ctx:
        if is_wrapped:
            targets_mh = batch['target_multihot'].to(device, non_blocking=True)
            dt_cnt_tensor = (
                dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor) 
                else torch.tensor(dt_cnt, device=device)
            )
            result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
            
            if isinstance(result, tuple):
                loss_val, extras = result
                output = extras.get('predictions')
            else:
                loss_val = result
                output = None
            loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
        else:
            if _model_has_moe(model):
                output, _ = model(x, return_moe_losses=False)
            else:
                output = model(x)
            loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
    
    return output, loss


def _update_streaming_metrics(
    metrics_tracker: StreamingMetrics,
    output: torch.Tensor,
    batch: Dict,
    config: BaseConfig
) -> None:
    """
    Update streaming metrics from a batch of predictions.
    
    Handles flattening over valid days and target alignment.
    """
    batch_size = output.shape[0]
    actual_len_dy = output.shape[1]
    
    dt_cnt = batch['dt_cnt']
    y = batch['target']
    
    dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
    y_flat = [item for sublist in y for item in sublist]
    
    # Flatten output over valid days
    output_flat = output.view(batch_size * actual_len_dy, config.target_cd_cnt)
    
    # Collect valid samples (only actual days, not padding)
    valid_outputs = []
    valid_targets = []
    
    for j in range(batch_size):
        valid_days = min(int(dt_cnt_values[j]), actual_len_dy)
        if valid_days <= 0:
            continue
        
        # Output indices (uses actual_len_dy)
        out_start = actual_len_dy * j
        out_end = out_start + valid_days
        valid_outputs.append(output_flat[out_start:out_end])
        
        # Target indices (uses config.len_dy since y is padded to full length)
        y_start = config.len_dy * j
        y_end = y_start + valid_days
        valid_targets.extend(y_flat[y_start:y_end])
    
    if valid_outputs:
        predictions = torch.cat(valid_outputs)
        metrics_tracker.update(predictions, valid_targets)


# Required for nullcontext in Python < 3.7
from contextlib import nullcontext
```

---

## STEP 3: Update `_build_epoch_metrics()` for New Metrics

**Location:** Lines 9470-9511

Add the new metrics to the epoch metrics dictionary:

```python
def _build_epoch_metrics(
    epoch: int,
    train_metrics: Dict,
    train_eval_metrics: Dict,
    val_metrics: Dict
) -> Dict[str, Any]:
    """Build comprehensive epoch metrics dictionary."""
    epoch_metrics = {
        'epoch': epoch + 1,
        # Training trajectory
        'train_loss': train_metrics['train_loss'],
        'train_loss_mean': train_metrics['train_loss_mean'],
        'train_loss_first': train_metrics['train_loss_first'],
        'train_loss_last': train_metrics['train_loss_last'],
        'train_loss_std': train_metrics['train_loss_std'],
        'train_loss_improvement': train_metrics['train_loss_improvement'],
        # Train evaluation (from streaming metrics)
        'eval_in_train_loss_final': train_eval_metrics['val_loss'],
        'eval_in_train_recall@1': train_eval_metrics.get('recall@1', 0.0),
        'eval_in_train_recall@5': train_eval_metrics.get('recall@5', 0.0),
        'eval_in_train_recall@10': train_eval_metrics.get('recall@10', 0.0),
        'eval_in_train_recall@20': train_eval_metrics.get('recall@20', 0.0),
        'eval_in_train_micro_recall@10': train_eval_metrics.get('micro_recall@10', 0.0),
        'eval_in_train_ndcg@20': train_eval_metrics.get('ndcg@20', 0.0),
        # Validation (from streaming metrics)
        'final_val_loss': val_metrics['val_loss'],
        'final_val_recall@1': val_metrics.get('recall@1', 0.0),
        'final_val_recall@5': val_metrics.get('recall@5', 0.0),
        'final_val_recall@10': val_metrics.get('recall@10', 0.0),
        'final_val_recall@20': val_metrics.get('recall@20', 0.0),
        'final_val_micro_recall@10': val_metrics.get('micro_recall@10', 0.0),
        'final_val_micro_recall@20': val_metrics.get('micro_recall@20', 0.0),
        'final_val_ndcg@10': val_metrics.get('ndcg@10', 0.0),
        'final_val_ndcg@20': val_metrics.get('ndcg@20', 0.0),
        'final_val_mrr': val_metrics.get('mrr', 0.0),
        'final_val_positive_brier': val_metrics.get('positive_brier', 0.0),
        'generalization_gap': train_eval_metrics['val_loss'] - val_metrics['val_loss'],
    }
    
    # Add other training metrics
    for k, v in train_metrics.items():
        if k.startswith('train_') and k not in epoch_metrics:
            epoch_metrics[k] = v
    
    # Add remaining validation metrics
    excluded_keys = {'val_loss', 'num_samples', 'num_batches'}
    for k, v in val_metrics.items():
        if k not in epoch_metrics and k not in excluded_keys:
            epoch_metrics[f'val_{k}'] = v
    
    return epoch_metrics
```

---

## STEP 4: Update `_build_final_results()` for New Metrics

**Location:** Lines 9514-9551

```python
def _build_final_results(
    exp_name: str,
    total_params: int,
    use_learnt_att_pool: bool,
    use_bucketing: bool,
    final_metrics: Dict,
    evaluation: Dict,
    epoch_history: List[Dict],
    total_time: float
) -> Dict[str, Any]:
    """Build final experiment results dictionary."""
    return {
        'experiment': exp_name,
        'parameters': total_params,
        'use_learned_pooling': use_learnt_att_pool,
        'use_bucketing': use_bucketing,
        'train_loss_mean': final_metrics['train_loss'],
        'train_loss_learned': final_metrics['train_loss_improvement'],
        'train_loss_final': final_metrics['eval_in_train_loss_final'],
        'val_loss_final': final_metrics['final_val_loss'],
        'generalization_gap': final_metrics['generalization_gap'],
        # Recall metrics
        'final_train_recall@5': final_metrics.get('eval_in_train_recall@5', 0.0),
        'final_train_recall@10': final_metrics.get('eval_in_train_recall@10', 0.0),
        'final_train_recall@20': final_metrics.get('eval_in_train_recall@20', 0.0),
        'final_val_recall@5': final_metrics.get('final_val_recall@5', 0.0),
        'final_val_recall@10': final_metrics.get('final_val_recall@10', 0.0),
        'final_val_recall@20': final_metrics.get('final_val_recall@20', 0.0),
        # New metrics
        'final_val_micro_recall@10': final_metrics.get('final_val_micro_recall@10', 0.0),
        'final_val_ndcg@20': final_metrics.get('final_val_ndcg@20', 0.0),
        'final_val_mrr': final_metrics.get('final_val_mrr', 0.0),
        'final_val_positive_brier': final_metrics.get('final_val_positive_brier', 0.0),
        # Comprehensive evaluation
        'training_time_sec': total_time,
        'precision@10': evaluation['performance'].get('precision@10', 0.0),
        'recall@10': evaluation['performance'].get('recall@10', 0.0),
        'f1@10': evaluation['performance'].get('f1@10', 0.0),
        'micro_recall@10': evaluation['performance'].get('micro_recall@10', 0.0),
        'ndcg@10': evaluation['performance'].get('ndcg@10', 0.0),
        'balanced_top10_acc': evaluation['performance'].get('balanced_top10_acc', 0.0),
        'tail_top10_acc': evaluation['performance'].get('tail_top10_acc', 0.0),
        'cost_usd': evaluation['resources'].get('cost_usd', 0.0),
        'peak_memory_gb': evaluation['resources'].get('total_peak_gb', 0.0),
        'full_evaluation': evaluation,
        'all_epochs': epoch_history
    }
```

---

## STEP 5: Update Logging Statements in `run_single_experiment()`

**Location:** ~Line 9768

```python
        logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}, "
                    f"Recall@10: {val_metrics.get('recall@10', 0):.3f}, "
                    f"μRecall@10: {val_metrics.get('micro_recall@10', 0):.3f}, "
                    f"NDCG@20: {val_metrics.get('ndcg@20', 0):.3f}")
```

---

## STEP 6: LR Scheduler Changes

### 6a. Add Warmup+Cosine Scheduler Function (Add near line 9555)

```python
import math

def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0
) -> optim.lr_scheduler.LambdaLR:
    """
    Create a cosine annealing scheduler with linear warmup.
    
    Industry-standard for transformer pretraining (GPT, BERT, etc.)
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Steps for linear warmup phase
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial (0.0 = decay to 0)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return max(min_lr_ratio, cosine_decay)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 6b. Update Scheduler Creation in `run_single_experiment()` (Lines 9695-9702)

**Find:**
```python
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

**Replace with:**
```python
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * epochs
    
    # Warmup: 10% of total steps or max 1000 steps
    warmup_steps = min(1000, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.01  # Don't decay to exactly 0
    )
    
    if is_main:
        logger.info(f"Scheduler: CosineWithWarmup (warmup={warmup_steps}, total={total_steps})")
```

### 6c. Scheduler Step Placement Options

The current code steps the scheduler **per optimizer step** (inside `train_epoch` at lines 4407-4408 and 4508-4509). This is **correct for step-based schedulers** like the warmup+cosine above.

**Current behavior (KEEP AS-IS):**
```python
# In train_epoch(), lines 4407-4409:
if scheduler is not None:
    scheduler.step()
```

If you want **epoch-based scheduling** instead (step once per epoch), you would:
1. Remove `scheduler.step()` from inside `train_epoch()` (lines 4407-4409 and 4508-4509)
2. Add it in `run_single_experiment()` after `train_epoch()` returns

**For epoch-based (ALTERNATIVE - not recommended for warmup):**
```python
# In run_single_experiment(), after train_epoch():
train_metrics = train_epoch(
    ...
    scheduler=None,  # Don't pass scheduler to train_epoch
    ...
)
if scheduler is not None:
    scheduler.step()  # Step once per epoch
```

**Recommendation:** Keep step-based scheduling (current) since warmup schedulers need fine-grained control.

---

## STEP 7: Other Functions/Classes That Need Adjustment

### 7a. `compute_embedding_quality_epoch()` (Lines 5785-5930)

Already handles memory efficiently, but verify it uses small batch sizes:

```python
# Line ~5820 - ensure small batch size
embedding_batch_size = min(8, config.batch_size)
val_loader = DataLoader(
    val_dataset, 
    batch_size=embedding_batch_size,
    num_workers=0,  # Avoid worker memory overhead
    ...
)
```

### 7b. `comprehensive_evaluation()` (Lines 7042-7235)

This function still accumulates `all_predictions` and `all_targets`. For very large validation sets, consider adding a streaming option:

```python
# At the beginning of comprehensive_evaluation(), add option:
def comprehensive_evaluation(
    model: nn.Module,
    val_dataloader: DataLoader,
    config: BaseConfig,
    device: torch.device,
    training_time_sec: float,
    epoch_history: List[Dict[str, float]],
    code_frequencies: np.ndarray,
    moe_config: Optional[MoEConfig] = None,
    use_mixed_precision: bool = False,
    max_samples: int = 10000  # NEW: Limit samples for memory safety
) -> Dict[str, any]:
```

### 7c. Test Functions

Update test functions to use new metric names:

**`test_metric_utilities()` (Line 7418):**
```python
def test_metric_utilities():
    vocab = 50
    predictions = torch.randn(32, vocab)
    targets = [[i % vocab] for i in range(32)]
    multihot = torch.randint(0, 2, (32, vocab)).float()
    freq = np.random.randint(1, 100, size=vocab)

    primary = compute_primary_task_metrics(predictions, targets, vocab)
    losses = compute_loss_metrics(predictions, multihot, nn.BCEWithLogitsLoss(), targets)
    stratified = compute_stratified_metrics(predictions, targets, freq, vocab)

    assert 'recall@10' in primary
    assert 'micro_recall@10' in primary  # NEW
    assert 'ndcg@10' in primary  # NEW
    assert 'bce_loss' in losses
    assert 'positive_brier' in losses  # NEW
    assert 'tail_top10_acc' in stratified
    print("Metric utilities ✔️")
```

---

## Summary Checklist

| Step | Component | Action | Priority |
|------|-----------|--------|----------|
| 1 | `StreamingMetrics` class | Replace entirely with new implementation | HIGH |
| 2 | `evaluate()` function | Replace with streaming version | HIGH |
| 3 | `_build_epoch_metrics()` | Add new metric keys | HIGH |
| 4 | `_build_final_results()` | Add new metric keys | HIGH |
| 5 | Logging statements | Update to show new metrics | MEDIUM |
| 6a | Warmup scheduler | Add `get_cosine_schedule_with_warmup()` | HIGH |
| 6b | Scheduler creation | Use warmup scheduler | HIGH |
| 6c | Scheduler stepping | Keep step-based (no change) | N/A |
| 7a | `compute_embedding_quality_epoch()` | Verify small batch size | MEDIUM |
| 7b | `comprehensive_evaluation()` | Add sample limit param | LOW |
| 7c | Test functions | Update metric assertions | LOW |

---

## After Making Changes

Run verification:
```bash
python -c "from moe_flashattn_3 import *; test_metric_utilities()"
```

Test streaming metrics:
```python
# Quick test
metrics = StreamingMetrics(k_values=(5, 10, 20))
predictions = torch.randn(32, 6297)
targets = [[i % 100, (i+1) % 100] for i in range(32)]
metrics.update(predictions, targets)
results = metrics.compute()
print(results)
```

---

You're currently in **Ask mode** - to apply these changes, please switch to **Agent mode** and I can help implement them directly. Alternatively, you can manually copy these code blocks into the appropriate locations in your file.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch

@dataclass
class StreamingMetricsState:
    """Internal state for streaming metrics computation."""
    # Loss
    total_loss: float = 0.0
    num_batches: int = 0
    
    # Sample counts
    num_samples: int = 0
    
    # Recall@K counters (binary: any hit in top-K)
    recall_hits: Dict[int, int] = field(default_factory=dict)
    recall_total: Dict[int, int] = field(default_factory=dict)
    
    # Micro-Recall@K (per-code: total hits / total true codes)
    micro_recall_hits: Dict[int, int] = field(default_factory=dict)
    micro_recall_true: Dict[int, int] = field(default_factory=dict)
    
    # Precision@K (sum for average)
    precision_sum: Dict[int, float] = field(default_factory=dict)
    precision_count: Dict[int, int] = field(default_factory=dict)
    
    # NDCG@K
    ndcg_sum: Dict[int, float] = field(default_factory=dict)
    ndcg_count: Dict[int, int] = field(default_factory=dict)
    
    # MRR (best-ranked true code)
    mrr_sum: float = 0.0
    mrr_count: int = 0
    
    # Positive-only Brier
    positive_brier_sum: float = 0.0
    positive_brier_count: int = 0


class StreamingMetrics:
    """
    Memory-efficient streaming metrics aggregator for clinical code prediction.
    
    Follows HuggingFace/PyTorch patterns:
    - `.reset()` clears state
    - `.update(batch)` processes one batch  
    - `.compute()` returns final metrics
    
    Key design principles:
    1. NEVER accumulate full predictions (prevents CPU OOM)
    2. Compute metrics incrementally per-batch
    3. Use only scalar counters, not tensors
    
    Example:
        metrics = StreamingMetrics(k_values=[5, 10, 20])
        for batch in dataloader:
            predictions, targets = forward(batch)
            metrics.update(predictions, targets)
        results = metrics.compute()
    """
    
    def __init__(
        self, 
        k_values: Tuple[int, ...] = (5, 10, 20, 50),
        compute_mrr: bool = True,
        compute_brier: bool = True,
        vocab_size: int = 6297
    ):
        """
        Args:
            k_values: K values for Recall@K, Precision@K, Micro-Recall@K, NDCG@K
            compute_mrr: Whether to compute Mean Reciprocal Rank
            compute_brier: Whether to compute Positive-Only Brier Score
            vocab_size: Target vocabulary size (for Brier calculation)
        """
        self.k_values = k_values
        self.compute_mrr = compute_mrr
        self.compute_brier = compute_brier
        self.vocab_size = vocab_size
        self._max_k = max(k_values)
        
        # Precompute NDCG discount factors: 1/log2(rank+2)
        self._discounts = 1.0 / np.log2(np.arange(2, self._max_k + 2))
        
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulators for a new evaluation pass."""
        self._state = StreamingMetricsState(
            recall_hits={k: 0 for k in self.k_values},
            recall_total={k: 0 for k in self.k_values},
            micro_recall_hits={k: 0 for k in self.k_values},
            micro_recall_true={k: 0 for k in self.k_values},
            precision_sum={k: 0.0 for k in self.k_values},
            precision_count={k: 0 for k in self.k_values},
            ndcg_sum={k: 0.0 for k in self.k_values},
            ndcg_count={k: 0 for k in self.k_values},
        )
    
    def update_loss(self, loss: float) -> None:
        """Accumulate loss from a batch."""
        self._state.total_loss += loss
        self._state.num_batches += 1
    
    def update(
        self,
        predictions: torch.Tensor,  # [batch_size, vocab_size] logits
        targets: List[List[int]],    # List of target code lists per sample
        probs: Optional[torch.Tensor] = None  # [batch_size, vocab_size] for Brier
    ) -> None:
        """
        Update metrics with a batch of predictions.
        
        Args:
            predictions: Model logits [batch_size, vocab_size] (will be moved to CPU)
            targets: List of target code lists (one per sample)
            probs: Optional sigmoid probabilities for Brier score
        """
        # Move to CPU and sort once (most expensive operation)
        with torch.no_grad():
            sorted_indices = torch.argsort(predictions, dim=-1, descending=True).cpu()
            if probs is None and self.compute_brier:
                probs = torch.sigmoid(predictions).cpu()
            elif probs is not None:
                probs = probs.cpu()
        
        batch_size = len(targets)
        self._state.num_samples += batch_size
        
        for i, target_codes in enumerate(targets):
            # Filter valid codes (non-zero)
            true_codes = set(c for c in target_codes if c != 0)
            if not true_codes:
                continue
            
            pred_indices = sorted_indices[i]
            num_true = len(true_codes)
            
            # === Recall@K, Micro-Recall@K, Precision@K ===
            for k in self.k_values:
                pred_set = set(pred_indices[:k].tolist())
                hits = len(true_codes & pred_set)
                
                # Recall@K (binary: any hit = success)
                if hits > 0:
                    self._state.recall_hits[k] += 1
                self._state.recall_total[k] += 1
                
                # Micro-Recall@K (per-code granularity)
                self._state.micro_recall_hits[k] += hits
                self._state.micro_recall_true[k] += num_true
                
                # Precision@K
                self._state.precision_sum[k] += hits / k
                self._state.precision_count[k] += 1
            
            # === NDCG@K ===
            for k in self.k_values:
                top_k_preds = pred_indices[:k].tolist()
                dcg = sum(
                    self._discounts[rank] 
                    for rank, pred in enumerate(top_k_preds) 
                    if pred in true_codes
                )
                num_relevant = min(num_true, k)
                idcg = sum(self._discounts[:num_relevant])
                ndcg = dcg / idcg if idcg > 0 else 0.0
                self._state.ndcg_sum[k] += ndcg
                self._state.ndcg_count[k] += 1
            
            # === MRR (best-ranked true code) ===
            if self.compute_mrr:
                best_rank = self._max_k  # Default to max if not found
                for code in true_codes:
                    rank_indices = (pred_indices[:self._max_k] == code).nonzero(as_tuple=True)[0]
                    if len(rank_indices) > 0:
                        rank = rank_indices[0].item()
                        if rank < best_rank:
                            best_rank = rank
                if best_rank < self._max_k:
                    self._state.mrr_sum += 1.0 / (best_rank + 1)
                self._state.mrr_count += 1
            
            # === Positive-Only Brier Score ===
            if self.compute_brier and probs is not None:
                for code in true_codes:
                    if 0 < code < self.vocab_size:
                        pred_prob = probs[i, code].item()
                        self._state.positive_brier_sum += (pred_prob - 1.0) ** 2
                        self._state.positive_brier_count += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated state.
        
        Returns:
            Dict with all computed metrics
        """
        metrics = {}
        
        # Loss
        metrics['val_loss'] = (
            self._state.total_loss / max(self._state.num_batches, 1)
        )
        
        # Recall@K
        for k in self.k_values:
            metrics[f'recall@{k}'] = (
                self._state.recall_hits[k] / self._state.recall_total[k]
                if self._state.recall_total[k] > 0 else 0.0
            )
        
        # Micro-Recall@K
        for k in self.k_values:
            metrics[f'micro_recall@{k}'] = (
                self._state.micro_recall_hits[k] / self._state.micro_recall_true[k]
                if self._state.micro_recall_true[k] > 0 else 0.0
            )
        
        # Precision@K
        for k in self.k_values:
            metrics[f'precision@{k}'] = (
                self._state.precision_sum[k] / self._state.precision_count[k]
                if self._state.precision_count[k] > 0 else 0.0
            )
        
        # NDCG@K
        for k in self.k_values:
            metrics[f'ndcg@{k}'] = (
                self._state.ndcg_sum[k] / self._state.ndcg_count[k]
                if self._state.ndcg_count[k] > 0 else 0.0
            )
        
        # MRR
        if self.compute_mrr:
            metrics['mrr'] = (
                self._state.mrr_sum / self._state.mrr_count
                if self._state.mrr_count > 0 else 0.0
            )
        
        # Positive-Only Brier
        if self.compute_brier:
            metrics['positive_brier'] = (
                self._state.positive_brier_sum / self._state.positive_brier_count
                if self._state.positive_brier_count > 0 else 0.0
            )
        
        # Metadata
        metrics['num_samples'] = self._state.num_samples
        metrics['num_batches'] = self._state.num_batches
        
        return metrics
    
    def __repr__(self) -> str:
        return (
            f"StreamingMetrics("
            f"k_values={self.k_values}, "
            f"samples={self._state.num_samples}, "
            f"batches={self._state.num_batches})"
        )
```

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = False,
    k_values: Tuple[int, ...] = (1, 5, 10, 20)
) -> Dict[str, float]:
    """
    Memory-efficient evaluation using streaming metrics.
    
    Key design:
    - Uses StreamingMetrics for incremental computation
    - NEVER accumulates predictions (prevents CPU OOM on large val sets)
    - Computes recall@K, micro-recall@K, precision@K, NDCG@K, MRR, positive-Brier
    
    Args:
        model: Model to evaluate (can be wrapped in DataParallel/DataParallelWrapper)
        dataloader: Validation DataLoader
        criterion: Loss function (e.g., BCEWithLogitsLoss)
        config: Model configuration
        device: Device for computation
        use_mixed_precision: Whether to use FP16 autocast
        max_batches: If set, only evaluate first N batches (for train subset)
        verbose: Print progress every 100 batches
        k_values: K values for top-K metrics
    
    Returns:
        Dict with val_loss, recall@K, micro_recall@K, precision@K, ndcg@K, mrr, positive_brier
    """
    model.eval()
    
    # Setup
    num_batches = len(dataloader)
    batches_to_process = min(num_batches, max_batches) if max_batches else num_batches
    
    if batches_to_process == 0:
        return _get_empty_eval_metrics(k_values)
    
    # Detect wrapper type
    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and 
        isinstance(model.module, DataParallelWrapper)
    )
    
    # Initialize streaming metrics
    metrics_tracker = StreamingMetrics(
        k_values=k_values,
        compute_mrr=True,
        compute_brier=True,
        vocab_size=config.target_cd_cnt
    )
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= batches_to_process:
                break
            
            if verbose and batch_idx % 100 == 0:
                print(f"    Evaluating batch {batch_idx}/{batches_to_process}")
            
            # Forward pass
            output, loss = _forward_batch(
                model, batch, config, device, 
                criterion, use_mixed_precision, is_wrapped
            )
            
            # Update streaming metrics
            metrics_tracker.update_loss(loss)
            
            if output is not None:
                _update_streaming_metrics(
                    metrics_tracker, output, batch, config
                )
            
            # Cleanup batch tensors immediately
            del output
            if batch_idx % 50 == 0:
                gc.collect()
    
    # Compute and return final metrics
    results = metrics_tracker.compute()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


def _get_empty_eval_metrics(k_values: Tuple[int, ...]) -> Dict[str, float]:
    """Return empty metrics dict when no data to evaluate."""
    metrics = {'val_loss': 0.0, 'mrr': 0.0, 'positive_brier': 0.0}
    for k in k_values:
        metrics[f'recall@{k}'] = 0.0
        metrics[f'micro_recall@{k}'] = 0.0
        metrics[f'precision@{k}'] = 0.0
        metrics[f'ndcg@{k}'] = 0.0
    return metrics


def _forward_batch(
    model: nn.Module,
    batch: Dict,
    config: BaseConfig,
    device: torch.device,
    criterion: nn.Module,
    use_mixed_precision: bool,
    is_wrapped: bool
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Execute forward pass for a single batch.
    
    Returns:
        (output_logits, loss_scalar)
    """
    # Prepare inputs
    age = batch['age'].to(device, non_blocking=True)
    gender = batch['gender'].to(device, non_blocking=True)
    lob = batch['lob'].to(device, non_blocking=True)
    codes = batch['codes'].to(device, non_blocking=True)
    dt_cnt = batch['dt_cnt']
    y = batch['target']
    
    x = torch.cat([
        age.unsqueeze(-1),
        gender.unsqueeze(-1),
        lob.unsqueeze(-1),
        codes
    ], dim=-1)
    
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16) 
        if use_mixed_precision else nullcontext()
    )
    
    with autocast_ctx:
        if is_wrapped:
            targets_mh = batch['target_multihot'].to(device, non_blocking=True)
            dt_cnt_tensor = (
                dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor) 
                else torch.tensor(dt_cnt, device=device)
            )
            result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
            
            if isinstance(result, tuple):
                loss_val, extras = result
                output = extras.get('predictions')
            else:
                loss_val = result
                output = None
            loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
        else:
            if _model_has_moe(model):
                output, _ = model(x, return_moe_losses=False)
            else:
                output = model(x)
            loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
    
    return output, loss


def _update_streaming_metrics(
    metrics_tracker: StreamingMetrics,
    output: torch.Tensor,
    batch: Dict,
    config: BaseConfig
) -> None:
    """
    Update streaming metrics from a batch of predictions.
    
    Handles flattening over valid days and target alignment.
    """
    batch_size = output.shape[0]
    actual_len_dy = output.shape[1]
    
    dt_cnt = batch['dt_cnt']
    y = batch['target']
    
    dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
    y_flat = [item for sublist in y for item in sublist]
    
    # Flatten output over valid days
    output_flat = output.view(batch_size * actual_len_dy, config.target_cd_cnt)
    
    # Collect valid samples (only actual days, not padding)
    valid_outputs = []
    valid_targets = []
    
    for j in range(batch_size):
        valid_days = min(int(dt_cnt_values[j]), actual_len_dy)
        if valid_days <= 0:
            continue
        
        # Output indices (uses actual_len_dy)
        out_start = actual_len_dy * j
        out_end = out_start + valid_days
        valid_outputs.append(output_flat[out_start:out_end])
        
        # Target indices (uses config.len_dy since y is padded to full length)
        y_start = config.len_dy * j
        y_end = y_start + valid_days
        valid_targets.extend(y_flat[y_start:y_end])
    
    if valid_outputs:
        predictions = torch.cat(valid_outputs)
        metrics_tracker.update(predictions, valid_targets)


# Required for nullcontext in Python < 3.7
from contextlib import nullcontext
```

```python
def _build_epoch_metrics(
    epoch: int,
    train_metrics: Dict,
    train_eval_metrics: Dict,
    val_metrics: Dict
) -> Dict[str, Any]:
    """Build comprehensive epoch metrics dictionary."""
    epoch_metrics = {
        'epoch': epoch + 1,
        # Training trajectory
        'train_loss': train_metrics['train_loss'],
        'train_loss_mean': train_metrics['train_loss_mean'],
        'train_loss_first': train_metrics['train_loss_first'],
        'train_loss_last': train_metrics['train_loss_last'],
        'train_loss_std': train_metrics['train_loss_std'],
        'train_loss_improvement': train_metrics['train_loss_improvement'],
        # Train evaluation (from streaming metrics)
        'eval_in_train_loss_final': train_eval_metrics['val_loss'],
        'eval_in_train_recall@1': train_eval_metrics.get('recall@1', 0.0),
        'eval_in_train_recall@5': train_eval_metrics.get('recall@5', 0.0),
        'eval_in_train_recall@10': train_eval_metrics.get('recall@10', 0.0),
        'eval_in_train_recall@20': train_eval_metrics.get('recall@20', 0.0),
        'eval_in_train_micro_recall@10': train_eval_metrics.get('micro_recall@10', 0.0),
        'eval_in_train_ndcg@20': train_eval_metrics.get('ndcg@20', 0.0),
        # Validation (from streaming metrics)
        'final_val_loss': val_metrics['val_loss'],
        'final_val_recall@1': val_metrics.get('recall@1', 0.0),
        'final_val_recall@5': val_metrics.get('recall@5', 0.0),
        'final_val_recall@10': val_metrics.get('recall@10', 0.0),
        'final_val_recall@20': val_metrics.get('recall@20', 0.0),
        'final_val_micro_recall@10': val_metrics.get('micro_recall@10', 0.0),
        'final_val_micro_recall@20': val_metrics.get('micro_recall@20', 0.0),
        'final_val_ndcg@10': val_metrics.get('ndcg@10', 0.0),
        'final_val_ndcg@20': val_metrics.get('ndcg@20', 0.0),
        'final_val_mrr': val_metrics.get('mrr', 0.0),
        'final_val_positive_brier': val_metrics.get('positive_brier', 0.0),
        'generalization_gap': train_eval_metrics['val_loss'] - val_metrics['val_loss'],
    }
    
    # Add other training metrics
    for k, v in train_metrics.items():
        if k.startswith('train_') and k not in epoch_metrics:
            epoch_metrics[k] = v
    
    # Add remaining validation metrics
    excluded_keys = {'val_loss', 'num_samples', 'num_batches'}
    for k, v in val_metrics.items():
        if k not in epoch_metrics and k not in excluded_keys:
            epoch_metrics[f'val_{k}'] = v
    
    return epoch_metrics
```

```python
def _build_final_results(
    exp_name: str,
    total_params: int,
    use_learnt_att_pool: bool,
    use_bucketing: bool,
    final_metrics: Dict,
    evaluation: Dict,
    epoch_history: List[Dict],
    total_time: float
) -> Dict[str, Any]:
    """Build final experiment results dictionary."""
    return {
        'experiment': exp_name,
        'parameters': total_params,
        'use_learned_pooling': use_learnt_att_pool,
        'use_bucketing': use_bucketing,
        'train_loss_mean': final_metrics['train_loss'],
        'train_loss_learned': final_metrics['train_loss_improvement'],
        'train_loss_final': final_metrics['eval_in_train_loss_final'],
        'val_loss_final': final_metrics['final_val_loss'],
        'generalization_gap': final_metrics['generalization_gap'],
        # Recall metrics
        'final_train_recall@5': final_metrics.get('eval_in_train_recall@5', 0.0),
        'final_train_recall@10': final_metrics.get('eval_in_train_recall@10', 0.0),
        'final_train_recall@20': final_metrics.get('eval_in_train_recall@20', 0.0),
        'final_val_recall@5': final_metrics.get('final_val_recall@5', 0.0),
        'final_val_recall@10': final_metrics.get('final_val_recall@10', 0.0),
        'final_val_recall@20': final_metrics.get('final_val_recall@20', 0.0),
        # New metrics
        'final_val_micro_recall@10': final_metrics.get('final_val_micro_recall@10', 0.0),
        'final_val_ndcg@20': final_metrics.get('final_val_ndcg@20', 0.0),
        'final_val_mrr': final_metrics.get('final_val_mrr', 0.0),
        'final_val_positive_brier': final_metrics.get('final_val_positive_brier', 0.0),
        # Comprehensive evaluation
        'training_time_sec': total_time,
        'precision@10': evaluation['performance'].get('precision@10', 0.0),
        'recall@10': evaluation['performance'].get('recall@10', 0.0),
        'f1@10': evaluation['performance'].get('f1@10', 0.0),
        'micro_recall@10': evaluation['performance'].get('micro_recall@10', 0.0),
        'ndcg@10': evaluation['performance'].get('ndcg@10', 0.0),
        'balanced_top10_acc': evaluation['performance'].get('balanced_top10_acc', 0.0),
        'tail_top10_acc': evaluation['performance'].get('tail_top10_acc', 0.0),
        'cost_usd': evaluation['resources'].get('cost_usd', 0.0),
        'peak_memory_gb': evaluation['resources'].get('total_peak_gb', 0.0),
        'full_evaluation': evaluation,
        'all_epochs': epoch_history
    }
```

```python
        logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}, "
                    f"Recall@10: {val_metrics.get('recall@10', 0):.3f}, "
                    f"μRecall@10: {val_metrics.get('micro_recall@10', 0):.3f}, "
                    f"NDCG@20: {val_metrics.get('ndcg@20', 0):.3f}")
```

```python
import math

def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0
) -> optim.lr_scheduler.LambdaLR:
    """
    Create a cosine annealing scheduler with linear warmup.
    
    Industry-standard for transformer pretraining (GPT, BERT, etc.)
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Steps for linear warmup phase
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial (0.0 = decay to 0)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return max(min_lr_ratio, cosine_decay)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

```python
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

```python
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * epochs
    
    # Warmup: 10% of total steps or max 1000 steps
    warmup_steps = min(1000, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.01  # Don't decay to exactly 0
    )
    
    if is_main:
        logger.info(f"Scheduler: CosineWithWarmup (warmup={warmup_steps}, total={total_steps})")
```

```python
# In train_epoch(), lines 4407-4409:
if scheduler is not None:
    scheduler.step()
```

```python
# In run_single_experiment(), after train_epoch():
train_metrics = train_epoch(
    ...
    scheduler=None,  # Don't pass scheduler to train_epoch
    ...
)
if scheduler is not None:
    scheduler.step()  # Step once per epoch
```

```python
# Line ~5820 - ensure small batch size
embedding_batch_size = min(8, config.batch_size)
val_loader = DataLoader(
    val_dataset, 
    batch_size=embedding_batch_size,
    num_workers=0,  # Avoid worker memory overhead
    ...
)
```

```python
# At the beginning of comprehensive_evaluation(), add option:
def comprehensive_evaluation(
    model: nn.Module,
    val_dataloader: DataLoader,
    config: BaseConfig,
    device: torch.device,
    training_time_sec: float,
    epoch_history: List[Dict[str, float]],
    code_frequencies: np.ndarray,
    moe_config: Optional[MoEConfig] = None,
    use_mixed_precision: bool = False,
    max_samples: int = 10000  # NEW: Limit samples for memory safety
) -> Dict[str, any]:
```

```python
def test_metric_utilities():
    vocab = 50
    predictions = torch.randn(32, vocab)
    targets = [[i % vocab] for i in range(32)]
    multihot = torch.randint(0, 2, (32, vocab)).float()
    freq = np.random.randint(1, 100, size=vocab)

    primary = compute_primary_task_metrics(predictions, targets, vocab)
    losses = compute_loss_metrics(predictions, multihot, nn.BCEWithLogitsLoss(), targets)
    stratified = compute_stratified_metrics(predictions, targets, freq, vocab)

    assert 'recall@10' in primary
    assert 'micro_recall@10' in primary  # NEW
    assert 'ndcg@10' in primary  # NEW
    assert 'bce_loss' in losses
    assert 'positive_brier' in losses  # NEW
    assert 'tail_top10_acc' in stratified
    print("Metric utilities ✔️")
```

```shellscript
python -c "from moe_flashattn_3 import *; test_metric_utilities()"
```

```python
# Quick test
metrics = StreamingMetrics(k_values=(5, 10, 20))
predictions = torch.randn(32, 6297)
targets = [[i % 100, (i+1) % 100] for i in range(32)]
metrics.update(predictions, targets)
results = metrics.compute()
print(results)
```

---

**User**

for teh LR Scheduler, do you think what you proposed or what I have implemtned is the best fit to my transformer? what would be other more optimal option given the industyr best practice and your epxerince? provide rational and details exlianaiton

---

**Cursor**

This is an excellent question that deserves a thorough analysis. Let me examine your specific setup and provide a detailed recommendation.

Let me also check what data sizes you're typically training with:

Now I have a complete picture. Let me provide a comprehensive analysis of LR scheduler options for your specific clinical transformer.

---

# Deep Dive: LR Scheduler Analysis for Your Clinical Transformer

## Your Current Setup

| Parameter | Value |
|-----------|-------|
| **Model** | Hierarchical Clinical Transformer + MoE |
| **Layers** | 6 temporal encoder layers |
| **Embedding** | 256 dim (SwiGLU FFN: 683) |
| **Vocab** | 75,516 input → 6,297 target codes |
| **Batch Size** | 32 per GPU × 4 GPUs = 128 effective |
| **Data Size** | ~350K to 12M patients |
| **Epochs** | 1-10 (often just 1-3) |
| **Base LR** | 1e-4 |
| **Optimizer** | AdamW (weight_decay=0.01) |
| **Current Scheduler** | `CosineAnnealingLR(T_max=total_steps)` with step-level updates |

---

## Analysis of Your Current Implementation

### What You Have (Lines 9700-9701):
```python
total_steps = len(train_loader) * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

### Current Behavior:
- LR starts at `scaled_lr` (e.g., 4e-4 with 4 GPUs using linear scaling)
- Decays following cosine curve to **0** by the end of training
- Updated every optimizer step (per-batch)

### Issues with Current Approach:

| Issue | Impact | Severity |
|-------|--------|----------|
| **No warmup** | First few steps with full LR can destabilize training, especially with MoE | ⚠️ Medium |
| **Decays to exactly 0** | Last 10-20% of training has near-zero learning, wasted compute | ⚠️ Medium |
| **Single-epoch training issue** | With epochs=1, LR is near 0 by the end of your only epoch | 🔴 High |
| **Linear LR scaling** | You use `base_lr * num_gpus` which is correct, but needs warmup to work well | ⚠️ Medium |

---

## Industry Best Practices for Transformer Pretraining

### 1. **Warmup + Cosine Decay** (GPT-2/3, BERT, LLaMA)

The gold standard for transformer pretraining:

```
LR
↑
│      ╭──────╮
│     ╱        ╲
│    ╱          ╲
│   ╱            ╲_____ min_lr
│  ╱
│ ╱
│╱
└──────────────────────→ Steps
  ↑                 ↑
  warmup           decay
```

**Why it works:**
- **Warmup (5-10% of steps)**: Gradients are noisy early on. Small LR prevents wild updates that can corrupt embeddings.
- **Cosine decay**: Smooth, continuous decay allows fine-tuning as model approaches convergence.
- **Minimum LR floor**: Keeps learning happening even late in training.

**Used by:** GPT-3, LLaMA, Chinchilla, PaLM, most modern LLMs

### 2. **Warmup + Linear Decay** (BERT, RoBERTa)

Simpler alternative:

```
LR
↑
│   ╱╲
│  ╱  ╲
│ ╱    ╲
│╱      ╲
│        ╲
│         ╲
└──────────╲──────────→ Steps
           ↘min_lr
```

**Why it works:**
- Linear is simpler, slightly faster to compute
- Works well for encoder-only models (BERT-style)

### 3. **Warmup + Constant + Decay** (T5, Flan)

Three-phase approach:

```
LR
↑
│     ╭────────────╮
│    ╱              ╲
│   ╱                ╲
│  ╱                  ╲
│ ╱                    ╲
│╱                      ╲
└────────────────────────→ Steps
  ↑      ↑           ↑
 warm  constant    decay
```

**Why it works:**
- Stable "exploration" phase at peak LR
- Good when you have long training runs (10+ epochs)

---

## Specific Analysis for Your Clinical Transformer

### Your Unique Characteristics:

1. **Multi-label BCE loss** (not cross-entropy): BCE is more sensitive to LR since each output is independent
2. **Hierarchical architecture** (daily encoder + temporal encoder): Two-level structure needs stable gradients
3. **MoE layers**: Router gradients are notoriously unstable early in training
4. **Large vocabulary** (6,297 targets): Output layer has many parameters, prone to overfit
5. **Short training** (1-3 epochs often): Every step matters; can't afford wasted epochs

### What This Means for LR Scheduling:

| Factor | Implication |
|--------|-------------|
| Multi-label BCE | Avoid aggressive LR that pushes all logits to extremes |
| MoE router | **Needs warmup** to stabilize expert selection |
| Short training | **Don't decay to 0** - you need learning until the end |
| Large vocab output | Lower peak LR is safer than linear scaling suggests |

---

## My Recommendation: **Warmup + Cosine with Floor**

For your specific setup, I recommend:

```python
def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1  # Don't go below 10% of peak LR
) -> optim.lr_scheduler.LambdaLR:
    """
    Warmup → Cosine decay → Floor
    
    Best for:
    - Short training runs (1-5 epochs)
    - MoE models (router stability)
    - Multi-label classification (BCE needs stable gradients)
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Recommended Parameters for Your Setup:

```python
# In run_single_experiment():
total_steps = len(train_loader) * epochs

# Warmup: 5% of total steps, minimum 100, maximum 2000
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))

# Min LR: 10% of peak (keeps learning happening)
min_lr_ratio = 0.1

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=min_lr_ratio
)
```

### Why These Parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Warmup 5%** | ~500 steps for 10K total | MoE router needs time to stabilize; embeddings need gradual warm-up |
| **Min LR 10%** | 4e-5 if peak is 4e-4 | BCE loss benefits from continued learning; your short epochs need it |
| **Cosine decay** | Smooth curve | Better than linear for finding fine-grained optima |

---

## Alternative Options Ranked

### 🥇 **Option 1: Warmup + Cosine with Floor (RECOMMENDED)**

```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_ratio=0.1
)
```

**Pros:**
- Best for MoE stability
- Works well for short training (1-3 epochs)
- Standard industry practice
- Keeps learning happening throughout

**Cons:**
- Slightly more complex than current

---

### 🥈 **Option 2: OneCycleLR (Good for Single Epoch)**

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=scaled_lr,
    total_steps=total_steps,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos',
    div_factor=25.0,  # Start at max_lr/25
    final_div_factor=1e4  # End at max_lr/10000
)
```

**How it works:**
```
LR
↑
│        ╭╮
│       ╱  ╲
│      ╱    ╲
│     ╱      ╲
│    ╱        ╲
│   ╱          ╲
│  ╱            ╲_____
│ ╱
└──────────────────────→ Steps
```

**Pros:**
- Specifically designed for "1 cycle" training
- Built into PyTorch
- Automatic warmup

**Cons:**
- Less flexible than custom LambdaLR
- `div_factor` and `final_div_factor` require tuning

---

### 🥉 **Option 3: Warmup + Inverse Square Root Decay (Long Training)**

```python
def inverse_sqrt_schedule(current_step: int) -> float:
    warmup_steps = 1000
    if current_step < warmup_steps:
        return float(current_step) / float(warmup_steps)
    return (warmup_steps ** 0.5) / (current_step ** 0.5)
```

**How it works:**
```
LR
↑
│  ╱\
│ ╱  \_______________
│╱
└────────────────────→ Steps
```

**Pros:**
- Excellent for very long training (10+ epochs, millions of samples)
- Used by original Transformer paper ("Attention is All You Need")
- Never reaches zero

**Cons:**
- Decays too slowly for short training
- Not ideal for your 1-3 epoch runs

---

### ❌ **Not Recommended: Your Current Approach**

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

**Problems:**
1. No warmup → MoE router instability
2. Decays to exactly 0 → Wasted last 10-20% of training
3. With 1 epoch, you're at near-zero LR for a significant portion

---

## LR Scaling for Multi-GPU

You're currently using linear scaling:
```python
scaled_lr = base_lr * num_gpus  # 1e-4 * 4 = 4e-4
```

### Analysis:

| Scaling Method | Formula | When to Use |
|----------------|---------|-------------|
| **Linear** | `lr × N` | Standard for SGD, works for most cases |
| **Square Root** | `lr × √N` | More conservative, good for unstable training |
| **None** | `lr` | When using gradient accumulation instead |

### My Recommendation for Your Setup:

For MoE with 4 GPUs, I'd recommend **square root scaling** combined with warmup:

```python
if use_data_parallel:
    scaled_lr = base_lr * math.sqrt(num_gpus)  # 1e-4 * 2 = 2e-4
else:
    scaled_lr = base_lr
```

**Rationale:**
- MoE router is sensitive to large LR updates
- Square root is safer with your 4-GPU setup
- The warmup will gradually reach the scaled LR anyway

---

## Complete Recommended Implementation

```python
import math

def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> optim.lr_scheduler.LambdaLR:
    """
    Industry-standard warmup + cosine decay scheduler.
    
    Used by GPT-3, LLaMA, Chinchilla, and most modern transformers.
    Modified with min_lr_ratio floor for short training runs.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# In run_single_experiment():
# ...

# LR Scaling (conservative for MoE)
if use_data_parallel:
    scaled_lr = base_lr * math.sqrt(num_gpus)  # sqrt scaling for stability
else:
    scaled_lr = base_lr

optimizer = optim.AdamW(
    model.parameters(),
    lr=scaled_lr,
    weight_decay=config.weight_decay
)

# Scheduler setup
total_steps = len(train_loader) * epochs
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=0.1  # Never go below 10% of peak LR
)

logger.info(f"Scheduler: Warmup({warmup_steps}) + Cosine(min_ratio=0.1)")
logger.info(f"Peak LR: {scaled_lr:.2e}, Min LR: {scaled_lr * 0.1:.2e}")
```

---

## Summary Comparison

| Scheduler | Your Current | My Recommendation |
|-----------|--------------|-------------------|
| **Type** | CosineAnnealingLR | Warmup + Cosine with Floor |
| **Warmup** | ❌ None | ✅ 5% of steps |
| **Min LR** | 0 (decays to zero) | 10% of peak |
| **LR Scaling** | Linear (4×) | Square root (2×) |
| **MoE Stability** | ⚠️ Risky | ✅ Stable |
| **Short Training** | ❌ Poor | ✅ Optimized |

### Expected Impact:
1. **Faster convergence** in early epochs (warmup prevents unstable early updates)
2. **Better final metrics** (continued learning at 10% LR floor)
3. **More stable MoE routing** (gradual LR increase lets router specialize)
4. **More efficient use of compute** (no wasted near-zero LR steps)

---

Would you like me to also analyze how this interacts with your gradient accumulation settings, or dive deeper into any specific aspect?

```python
total_steps = len(train_loader) * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

```plaintext
LR
↑
│      ╭──────╮
│     ╱        ╲
│    ╱          ╲
│   ╱            ╲_____ min_lr
│  ╱
│ ╱
│╱
└──────────────────────→ Steps
  ↑                 ↑
  warmup           decay
```

```plaintext
LR
↑
│   ╱╲
│  ╱  ╲
│ ╱    ╲
│╱      ╲
│        ╲
│         ╲
└──────────╲──────────→ Steps
           ↘min_lr
```

```plaintext
LR
↑
│     ╭────────────╮
│    ╱              ╲
│   ╱                ╲
│  ╱                  ╲
│ ╱                    ╲
│╱                      ╲
└────────────────────────→ Steps
  ↑      ↑           ↑
 warm  constant    decay
```

```python
def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1  # Don't go below 10% of peak LR
) -> optim.lr_scheduler.LambdaLR:
    """
    Warmup → Cosine decay → Floor
    
    Best for:
    - Short training runs (1-5 epochs)
    - MoE models (router stability)
    - Multi-label classification (BCE needs stable gradients)
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

```python
# In run_single_experiment():
total_steps = len(train_loader) * epochs

# Warmup: 5% of total steps, minimum 100, maximum 2000
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))

# Min LR: 10% of peak (keeps learning happening)
min_lr_ratio = 0.1

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=min_lr_ratio
)
```

```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_ratio=0.1
)
```

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=scaled_lr,
    total_steps=total_steps,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos',
    div_factor=25.0,  # Start at max_lr/25
    final_div_factor=1e4  # End at max_lr/10000
)
```

```plaintext
LR
↑
│        ╭╮
│       ╱  ╲
│      ╱    ╲
│     ╱      ╲
│    ╱        ╲
│   ╱          ╲
│  ╱            ╲_____
│ ╱
└──────────────────────→ Steps
```

```python
def inverse_sqrt_schedule(current_step: int) -> float:
    warmup_steps = 1000
    if current_step < warmup_steps:
        return float(current_step) / float(warmup_steps)
    return (warmup_steps ** 0.5) / (current_step ** 0.5)
```

```plaintext
LR
↑
│  ╱\
│ ╱  \_______________
│╱
└────────────────────→ Steps
```

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

```python
scaled_lr = base_lr * num_gpus  # 1e-4 * 4 = 4e-4
```

```python
if use_data_parallel:
    scaled_lr = base_lr * math.sqrt(num_gpus)  # 1e-4 * 2 = 2e-4
else:
    scaled_lr = base_lr
```

```python
import math

def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> optim.lr_scheduler.LambdaLR:
    """
    Industry-standard warmup + cosine decay scheduler.
    
    Used by GPT-3, LLaMA, Chinchilla, and most modern transformers.
    Modified with min_lr_ratio floor for short training runs.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# In run_single_experiment():
# ...

# LR Scaling (conservative for MoE)
if use_data_parallel:
    scaled_lr = base_lr * math.sqrt(num_gpus)  # sqrt scaling for stability
else:
    scaled_lr = base_lr

optimizer = optim.AdamW(
    model.parameters(),
    lr=scaled_lr,
    weight_decay=config.weight_decay
)

# Scheduler setup
total_steps = len(train_loader) * epochs
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=0.1  # Never go below 10% of peak LR
)

logger.info(f"Scheduler: Warmup({warmup_steps}) + Cosine(min_ratio=0.1)")
logger.info(f"Peak LR: {scaled_lr:.2e}, Min LR: {scaled_lr * 0.1:.2e}")
```

---

**User**

Ok I have implemented the get_cosine_schedule_with_warmup(), except hte run_single_experiment, are tehre any other functions I need to adjsust accodingly? like the checkpoint saving loading, model saving loading or anywhere else 

---

**Cursor**

Let me search for all scheduler-related code to identify any compatibility issues:

Good news! The `LambdaLR` scheduler (which `get_cosine_schedule_with_warmup` returns) is fully compatible with the existing checkpoint saving/loading infrastructure. Let me give you a complete analysis:

---

## Compatibility Analysis

### ✅ **Fully Compatible (No Changes Needed)**

| Function | Why It's Compatible |
|----------|---------------------|
| `save_checkpoint()` (line 4985) | Uses `scheduler.state_dict()` - works with all PyTorch schedulers including `LambdaLR` |
| `load_checkpoint()` (lines 5068-5069) | Uses `scheduler.load_state_dict()` - works with all PyTorch schedulers |
| `_resume_from_checkpoint()` (lines 9448-9449) | Same pattern, fully compatible |
| `train_epoch()` (lines 4407-4408, 4508-4509) | Calls `scheduler.step()` - universal method |

### ⚠️ **Need to Update (For Consistency)**

These test functions use `StepLR` - they'll still work, but updating them provides consistency:

| Location | Current | Recommended |
|----------|---------|-------------|
| `test_train_epoch_smoke()` (line 4879) | `StepLR` | Keep as-is (test doesn't need fancy scheduler) |
| `test_save_load_checkpoint_only()` (lines 5110, 5158) | `StepLR` | Update for resume testing |
| `test_train_epoch_full_integration()` (line 11627) | `StepLR` | Keep as-is (test doesn't need fancy scheduler) |

### 🔴 **Must Update**

| Location | Current | Action |
|----------|---------|--------|
| `run_single_experiment()` (line 9701) | `CosineAnnealingLR` | Replace with `get_cosine_schedule_with_warmup()` |

---

## What Happens During Checkpoint Resume?

When you resume from a checkpoint, `LambdaLR.load_state_dict()` restores:
- `last_epoch` (the current step count)
- `_step_count` (total steps taken)
- `base_lrs` (initial learning rates)

**Important:** The `lr_lambda` function itself is NOT serialized. This means:

1. ✅ **Resume works correctly** as long as you create the scheduler with the **same parameters** before loading
2. ⚠️ **Changing warmup/total_steps mid-training** would cause unexpected behavior

---

## Recommended Code Updates

### 1. `run_single_experiment()` - Lines 9695-9702

**Replace:**
```python
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

**With:**
```python
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    
    # Scheduler setup with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.1
    )
    
    if is_main:
        logger.info(f"Scheduler: Warmup({warmup_steps}) + Cosine (min_ratio=0.1)")
        logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.1:.2e}")
```

### 2. (Optional) Add Scheduler Config to Checkpoint

For better reproducibility, you can save scheduler configuration:

**In `save_checkpoint()` around line 4980:**
```python
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'timestamp': time.time(),
        'model_type': type(actual_model).__name__,
        # NEW: Save scheduler config for reproducibility
        'scheduler_config': {
            'type': type(scheduler).__name__,
            'total_steps': getattr(scheduler, '_total_steps', None),  # If you store it
            'warmup_steps': getattr(scheduler, '_warmup_steps', None),  # If you store it
        } if scheduler else None
    }
```

### 3. (Optional) Validate Resume Compatibility

**In `_resume_from_checkpoint()` around line 9448:**
```python
    if scheduler and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"  Restored scheduler state (last_epoch={scheduler.last_epoch})")
        except Exception as e:
            logger.warning(f"  Could not restore scheduler: {e}")
            logger.warning(f"  Continuing with fresh scheduler from step 0")
```

### 4. (Optional) Update Test for Resume Verification

If you want to verify that resume works correctly with the new scheduler, update `test_save_load_checkpoint_only()`:

**Lines 5107-5111:**
```python
    config = BaseConfig(batch_size=4, len_dy=32, len_cd=30)
    model = BaselineTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Use the same scheduler type as production
    total_steps = 100  # Mock value for test
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10, 
        num_training_steps=total_steps,
        min_lr_ratio=0.1
    )
```

**Lines 5155-5159:**
```python
        model_new = BaselineTransformer(config).to(device)
        optimizer_new = optim.AdamW(model_new.parameters(), lr=1e-4)
        
        # Must use same scheduler config for resume
        scheduler_new = get_cosine_schedule_with_warmup(
            optimizer_new, 
            num_warmup_steps=10, 
            num_training_steps=total_steps,
            min_lr_ratio=0.1
        )
```

---

## Summary Checklist

| Component | Action | Priority |
|-----------|--------|----------|
| `get_cosine_schedule_with_warmup()` | ✅ Already implemented | Done |
| `run_single_experiment()` (line 9701) | Replace scheduler creation | 🔴 Required |
| `save_checkpoint()` | No changes needed | ✅ Compatible |
| `load_checkpoint()` | No changes needed | ✅ Compatible |
| `_resume_from_checkpoint()` | Optionally add error handling | 🟡 Optional |
| `train_epoch()` | No changes needed | ✅ Compatible |
| Test functions | Optionally update for consistency | 🟡 Optional |

---

## Important Resume Consideration

When resuming training, ensure you recreate the scheduler with the **exact same parameters** before loading:

```python
# In resume logic - CORRECT approach:
total_steps = len(train_loader) * epochs  # Must be same as original
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # Same formula

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=0.1
)

# Now load state
checkpoint = torch.load('checkpoint.pt')
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# scheduler.last_epoch now reflects the resumed step
```

If you change `epochs` or `total_steps` when resuming, the LR curve will be different than expected. This is inherent to all step-based schedulers, not specific to your implementation.

```python
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

```python
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    
    # Scheduler setup with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.1
    )
    
    if is_main:
        logger.info(f"Scheduler: Warmup({warmup_steps}) + Cosine (min_ratio=0.1)")
        logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.1:.2e}")
```

```python
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'timestamp': time.time(),
        'model_type': type(actual_model).__name__,
        # NEW: Save scheduler config for reproducibility
        'scheduler_config': {
            'type': type(scheduler).__name__,
            'total_steps': getattr(scheduler, '_total_steps', None),  # If you store it
            'warmup_steps': getattr(scheduler, '_warmup_steps', None),  # If you store it
        } if scheduler else None
    }
```

```python
    if scheduler and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"  Restored scheduler state (last_epoch={scheduler.last_epoch})")
        except Exception as e:
            logger.warning(f"  Could not restore scheduler: {e}")
            logger.warning(f"  Continuing with fresh scheduler from step 0")
```

```python
    config = BaseConfig(batch_size=4, len_dy=32, len_cd=30)
    model = BaselineTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Use the same scheduler type as production
    total_steps = 100  # Mock value for test
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10, 
        num_training_steps=total_steps,
        min_lr_ratio=0.1
    )
```

```python
        model_new = BaselineTransformer(config).to(device)
        optimizer_new = optim.AdamW(model_new.parameters(), lr=1e-4)
        
        # Must use same scheduler config for resume
        scheduler_new = get_cosine_schedule_with_warmup(
            optimizer_new, 
            num_warmup_steps=10, 
            num_training_steps=total_steps,
            min_lr_ratio=0.1
        )
```

```python
# In resume logic - CORRECT approach:
total_steps = len(train_loader) * epochs  # Must be same as original
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # Same formula

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=0.1
)

# Now load state
checkpoint = torch.load('checkpoint.pt')
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# scheduler.last_epoch now reflects the resumed step
```

---

**User**

Ok here is what i have impelented hte modificaiton; closely review and inspect the changes if they are correct and any potentialisseus or bugs @dev/moe/moe_flashattn_3.py 

---

**Cursor**

Let me read the current state of your file to review the changes you've made:

Let me read the key implementation sections in detail:

Let me check the remaining parts - `_update_streaming_metrics` continuation and `_build_epoch_metrics`:

Let me check a few more potential issues:

Found an issue! Let me verify a couple more things:

---

# Code Review: Implementation Analysis

I've reviewed your implementation thoroughly. Here's my comprehensive assessment:

## ✅ **Correctly Implemented**

### 1. `get_cosine_schedule_with_warmup()` (Lines 4218-4250)
- ✅ Correct warmup phase logic
- ✅ Correct cosine decay formula
- ✅ Proper `min_lr_ratio` floor applied
- ✅ Handles edge case of `num_warmup_steps = 0`

### 2. `StreamingMetrics` Class (Lines 7374-7593)
- ✅ Proper dataclass for state management
- ✅ Correct streaming accumulator pattern
- ✅ All metrics (Recall, Micro-Recall, Precision, NDCG, MRR, Brier) implemented correctly
- ✅ Efficient use of precomputed NDCG discounts
- ✅ Proper handling of empty targets

### 3. `evaluate()` Function (Lines 4708-4799)
- ✅ Uses `StreamingMetrics` correctly
- ✅ Proper memory cleanup with `gc.collect()`
- ✅ Handles both wrapped and unwrapped models

### 4. `_build_epoch_metrics()` and `_build_final_results()` (Lines 9664-9763)
- ✅ Properly updated with new metric keys
- ✅ Uses `.get()` with defaults for safety

### 5. Scheduler Setup in `run_single_experiment()` (Lines 9912-9924)
- ✅ Correct warmup calculation formula
- ✅ Proper scheduler creation
- ✅ Resume checkpoint compatibility maintained

---

## 🔴 **Issues Found (Must Fix)**

### Issue 1: Missing `nullcontext` Import

**Location:** Line 4845 uses `nullcontext()` but it's never imported.

**Error you'll get:**
```
NameError: name 'nullcontext' is not defined
```

**Fix:** Add this import at the top of the file (around line 400-410):

```python
from contextlib import nullcontext
```

---

### Issue 2: Module-Level Test Code Executes on Import

**Location:** Lines 7601-7606

```python
metrics = StreamingMetrics(k_values=(5, 10, 20))
predictions = torch.randn(32, 6297)
targets = [[i % 100, (i+1) % 100] for i in range(32)]
metrics.update(predictions, targets)
results = metrics.compute()
print(results)
```

**Problem:** This code runs every time the module is imported, which:
- Slows down imports
- Prints output unexpectedly
- Uses GPU memory unnecessarily

**Fix:** Wrap in a function or add a guard:

```python
def test_streaming_metrics_quick():
    """Quick test for StreamingMetrics."""
    metrics = StreamingMetrics(k_values=(5, 10, 20))
    predictions = torch.randn(32, 6297)
    targets = [[i % 100, (i+1) % 100] for i in range(32)]
    metrics.update(predictions, targets)
    results = metrics.compute()
    print(results)
    print("StreamingMetrics quick test ✔️")

# Uncomment to run:
# test_streaming_metrics_quick()
```

---

### Issue 3: Incorrect End LR Log Message

**Location:** Line 9923

```python
logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.1:.2e}")
```

**Problem:** You use `min_lr_ratio=0.01` (line 9919), but the log shows `* 0.1` (10%), which is misleading.

**Fix:**

```python
logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.01:.2e}")
```

Or better, define a variable:

```python
min_lr_ratio = 0.01
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=min_lr_ratio
)
if is_main:
    logger.info(f"Scheduler: CosineWithWarmup (warmup={warmup_steps}, total={total_steps})")
    logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * min_lr_ratio:.2e}")
```

---

## ⚠️ **Potential Issues (Should Consider)**

### Issue 4: Comment/Code Mismatch for Warmup

**Location:** Line 9913

```python
# Warmup: 10% of total steps or max 1000 steps
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup
```

**Problem:** The comment says "10%" but the code uses "5%".

**Fix:** Update comment:

```python
# Warmup: 5% of total steps, clamped to [100, 2000] steps
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))
```

---

### Issue 5: NDCG Discount Array Bounds

**Location:** Line 7493-7496

```python
for k in self.k_values:
    top_k_preds = pred_indices[:k].tolist()
    dcg = sum(
        self._discounts[rank] 
        for rank, pred in enumerate(top_k_preds) 
        if pred in true_codes
    )
```

**Potential Issue:** If any `k` in `k_values` is larger than `_max_k`, you'll get an index error.

**Current Safety:** You initialize `self._max_k = max(k_values)` and size `_discounts` accordingly, so this is safe as currently written. ✅

---

### Issue 6: Missing Memory Cleanup in `_update_streaming_metrics`

**Location:** Lines 4916-4918

```python
if valid_outputs:
    predictions = torch.cat(valid_outputs)
    metrics_tracker.update(predictions, valid_targets)
```

**Improvement:** Add cleanup after processing:

```python
if valid_outputs:
    predictions = torch.cat(valid_outputs)
    metrics_tracker.update(predictions, valid_targets)
    del predictions, valid_outputs, valid_targets
```

---

## Summary of Required Fixes

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| 🔴 **Critical** | Missing `nullcontext` import | ~Line 400 | Add `from contextlib import nullcontext` |
| 🔴 **Critical** | Module-level test code | Lines 7601-7606 | Wrap in function |
| 🟡 **Medium** | Incorrect End LR log | Line 9923 | Change `0.1` to `0.01` |
| 🟢 **Minor** | Comment says 10%, code is 5% | Line 9913 | Fix comment |
| 🟢 **Minor** | Add cleanup in `_update_streaming_metrics` | Line 4918 | Add `del` statement |

---

## Quick Fix Code Block

Add this import near the top (around line 404):

```python
from contextlib import nullcontext
```

Replace lines 7601-7606 with:

```python
def test_streaming_metrics_quick():
    """Quick test for StreamingMetrics."""
    metrics = StreamingMetrics(k_values=(5, 10, 20))
    predictions = torch.randn(32, 6297)
    targets = [[i % 100, (i+1) % 100] for i in range(32)]
    metrics.update(predictions, targets)
    results = metrics.compute()
    print(results)
    print("StreamingMetrics quick test ✔️")
# test_streaming_metrics_quick()  # Uncomment to run
```

Fix line 9913 comment:
```python
# Warmup: 5% of total steps, clamped to [100, 2000] steps
```

Fix line 9923:
```python
logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.01:.2e}")
```

---

**Overall Assessment:** The implementation logic is sound. Once you fix the `nullcontext` import, it should work correctly. The other issues are minor cleanup items.

```plaintext
NameError: name 'nullcontext' is not defined
```

```python
from contextlib import nullcontext
```

```python
metrics = StreamingMetrics(k_values=(5, 10, 20))
predictions = torch.randn(32, 6297)
targets = [[i % 100, (i+1) % 100] for i in range(32)]
metrics.update(predictions, targets)
results = metrics.compute()
print(results)
```

```python
def test_streaming_metrics_quick():
    """Quick test for StreamingMetrics."""
    metrics = StreamingMetrics(k_values=(5, 10, 20))
    predictions = torch.randn(32, 6297)
    targets = [[i % 100, (i+1) % 100] for i in range(32)]
    metrics.update(predictions, targets)
    results = metrics.compute()
    print(results)
    print("StreamingMetrics quick test ✔️")

# Uncomment to run:
# test_streaming_metrics_quick()
```

```python
logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.1:.2e}")
```

```python
logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.01:.2e}")
```

```python
min_lr_ratio = 0.01
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=min_lr_ratio
)
if is_main:
    logger.info(f"Scheduler: CosineWithWarmup (warmup={warmup_steps}, total={total_steps})")
    logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * min_lr_ratio:.2e}")
```

```python
# Warmup: 10% of total steps or max 1000 steps
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup
```

```python
# Warmup: 5% of total steps, clamped to [100, 2000] steps
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))
```

```python
for k in self.k_values:
    top_k_preds = pred_indices[:k].tolist()
    dcg = sum(
        self._discounts[rank] 
        for rank, pred in enumerate(top_k_preds) 
        if pred in true_codes
    )
```

```python
if valid_outputs:
    predictions = torch.cat(valid_outputs)
    metrics_tracker.update(predictions, valid_targets)
```

```python
if valid_outputs:
    predictions = torch.cat(valid_outputs)
    metrics_tracker.update(predictions, valid_targets)
    del predictions, valid_outputs, valid_targets
```

```python
from contextlib import nullcontext
```

```python
def test_streaming_metrics_quick():
    """Quick test for StreamingMetrics."""
    metrics = StreamingMetrics(k_values=(5, 10, 20))
    predictions = torch.randn(32, 6297)
    targets = [[i % 100, (i+1) % 100] for i in range(32)]
    metrics.update(predictions, targets)
    results = metrics.compute()
    print(results)
    print("StreamingMetrics quick test ✔️")
# test_streaming_metrics_quick()  # Uncomment to run
```

```python
# Warmup: 5% of total steps, clamped to [100, 2000] steps
```

```python
logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.01:.2e}")
```

