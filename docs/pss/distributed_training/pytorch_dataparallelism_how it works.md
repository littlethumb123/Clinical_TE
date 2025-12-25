# How PyTorch DataParallel SHOULD Work (Theory vs Your Reality)

## Ideal DataParallel Operation

```
═══════════════════════════════════════════════════════════════════════════════
                    IDEAL nn.DataParallel FLOW (Theory)
═══════════════════════════════════════════════════════════════════════════════

PHASE 1: REPLICATE (Once per forward pass)
─────────────────────────────────────────────────────────────────────────────────
Master model lives on GPU 0. Parameters are BROADCAST to all GPUs.

GPU 0: [████ MODEL PARAMS ████]  ◄─── Master copy
          │
          ├──────────────────────► GPU 1: [████ MODEL PARAMS ████]  (copy)
          ├──────────────────────► GPU 2: [████ MODEL PARAMS ████]  (copy)
          └──────────────────────► GPU 3: [████ MODEL PARAMS ████]  (copy)

Note: This happens EVERY forward pass (expensive!)


PHASE 2: SCATTER (Split input data)
─────────────────────────────────────────────────────────────────────────────────
Input batch on GPU 0 is SPLIT along dimension 0 and sent to each GPU.

Input: [batch=256, seq=200, features=83] on GPU 0
                    │
    ┌───────────────┼───────────────┬───────────────┐
    ▼               ▼               ▼               ▼
GPU 0: [64,200,83] GPU 1: [64,200,83] GPU 2: [64,200,83] GPU 3: [64,200,83]

Transfer: GPU 0 → GPU 1,2,3 via NVLink/PCIe


PHASE 3: PARALLEL FORWARD (True parallelism!)
─────────────────────────────────────────────────────────────────────────────────
Each GPU runs forward pass INDEPENDENTLY on its data chunk.

Time ──────────────────────────────────────────────────────────────►

GPU 0: [════════ FORWARD PASS ════════] → output[64, 200, 6297]
GPU 1: [════════ FORWARD PASS ════════] → output[64, 200, 6297]
GPU 2: [════════ FORWARD PASS ════════] → output[64, 200, 6297]
GPU 3: [════════ FORWARD PASS ════════] → output[64, 200, 6297]
       ▲                               ▲
       │         PARALLEL!             │
       └───────────────────────────────┘


PHASE 4: GATHER (Collect outputs to GPU 0)
─────────────────────────────────────────────────────────────────────────────────
Outputs from all GPUs are CONCATENATED on GPU 0 (output_device).

GPU 0: output[64,...]  ─────┐
GPU 1: output[64,...] ──────┼──► GPU 0: [256, 200, 6297] (concatenated)
GPU 2: output[64,...] ──────┤
GPU 3: output[64,...] ──────┘


PHASE 5: LOSS + BACKWARD (Here's where it gets tricky!)
─────────────────────────────────────────────────────────────────────────────────
Loss is computed on GPU 0. Backward propagates gradients.

          GPU 0: loss = criterion(output[256,...], target[256,...])
                           │
                           ▼
          GPU 0: loss.backward()  ← Gradients flow BACK through gather
                           │
    ┌──────────────────────┼──────────────────────┬─────────────────────┐
    ▼                      ▼                      ▼                     ▼
GPU 0: grads[64]    GPU 1: grads[64]       GPU 2: grads[64]      GPU 3: grads[64]


PHASE 6: REDUCE GRADIENTS (Aggregate to GPU 0)
─────────────────────────────────────────────────────────────────────────────────
Gradients from all GPUs are SUMMED on GPU 0.

GPU 0: ∂L/∂θ₀ ──────┐
GPU 1: ∂L/∂θ₁ ──────┼──► GPU 0: Σ(∂L/∂θᵢ) / 4  (averaged gradients)
GPU 2: ∂L/∂θ₂ ──────┤
GPU 3: ∂L/∂θ₃ ──────┘


PHASE 7: OPTIMIZER STEP (Only on GPU 0)
─────────────────────────────────────────────────────────────────────────────────
Optimizer updates ONLY the master model on GPU 0.
Next iteration: Phase 1 replicates updated weights.

GPU 0: θ = θ - lr * ∇L   ◄─── Only master updated
GPU 1-3: [stale weights until next replicate]


═══════════════════════════════════════════════════════════════════════════════
                    IDEAL TIMELINE (Balanced Workload)
═══════════════════════════════════════════════════════════════════════════════

Time ─────────────────────────────────────────────────────────────────────────►
      │ Replicate │ Scatter │ Forward │ Gather │ Loss+Bwd │ Reduce │ Opt │

GPU 0: [REPLICATE] [scatter] [██████] [gather] [█loss█] [reduce] [opt]
GPU 1: [  copy   ] [receive] [██████] [ send ] [█back█] [ send ] [   ]
GPU 2: [  copy   ] [receive] [██████] [ send ] [█back█] [ send ] [   ]
GPU 3: [  copy   ] [receive] [██████] [ send ] [█back█] [ send ] [   ]

       └────┬────┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬──┘
         ~5%       ~5%     ~40%     ~5%     ~30%      ~10%    ~5%
                           ▲                ▲
                      PARALLEL!         PARALLEL!
                    (theoretical)    (if loss inside model)
```

---

## What's ACTUALLY Happening in Your Code

```
═══════════════════════════════════════════════════════════════════════════════
                    YOUR CURRENT REALITY (Broken DataParallel)
═══════════════════════════════════════════════════════════════════════════════

PROBLEM 1: Data pre-moved to GPU 0 (lines 3941-3944)
─────────────────────────────────────────────────────────────────────────────────

Your code:
    age = batch['age'].to(device, non_blocking=True)      # → GPU 0
    gender = batch['gender'].to(device, non_blocking=True) # → GPU 0
    codes = batch['codes'].to(device, non_blocking=True)   # → GPU 0

Result: Scatter starts from GPU 0, not CPU. Extra copy overhead.


PROBLEM 2: Loss computed OUTSIDE model (lines 3968)
─────────────────────────────────────────────────────────────────────────────────

Your code:
    output = model(x)  # DataParallel gathers to GPU 0
    pred_loss = compute_loss(output, y, dt_cnt, ...)  # Runs on GPU 0 ONLY!

Result: GPU 0 does ALL loss computation. GPUs 1-3 sit idle during this.


PROBLEM 3: targets (y) never moved to GPU!
─────────────────────────────────────────────────────────────────────────────────

Your code:
    y = batch['target']  # Python list! Never goes to GPU

Result: compute_loss() creates tensors on GPU 0 every batch. Slow!


═══════════════════════════════════════════════════════════════════════════════
                    YOUR ACTUAL TIMELINE (Broken)
═══════════════════════════════════════════════════════════════════════════════

Time ─────────────────────────────────────────────────────────────────────────►

GPU 0: [══ data to GPU0 ══][scatter][fwd][gather][═══ LOSS (slow!) ═══][backward][opt]
GPU 1:                     [receive][fwd][send ][======= IDLE ========][   ?    ][  ]
GPU 2:                     [receive][fwd][send ][======= IDLE ========][   ?    ][  ]
GPU 3:                     [receive][fwd][send ][======= IDLE ========][   ?    ][  ]

       └────────┬─────────┘        └─┬─┘        └─────────┬──────────┘
            Wasted               True                  Serialized!
           (data copy)         parallel            (this is your bottleneck)
             ~10%               ~15%                    ~60%


WHY GPU 1-3 SHOW 0.02 GB:
─────────────────────────────────────────────────────────────────────────────────

Memory lifecycle during YOUR training:

Step 1: DataParallel.forward() starts
        GPU 0: model params (kept)
        GPU 1-3: model params COPIED temporarily

Step 2: Forward pass runs
        GPU 0-3: activations allocated (this is the 3.26GB peak you saw)

Step 3: gather() collects outputs to GPU 0
        GPU 1-3: outputs sent to GPU 0, local tensors FREED

Step 4: compute_loss() runs on GPU 0 ONLY
        GPU 1-3: NO COMPUTATION, memory released
        GPU 1-3: Only 0.02GB = CUDA context overhead

Step 5: backward() starts from GPU 0
        Gradients flow back, but most computation on GPU 0
        GPU 1-3: Minimal gradient computation

Result: GPU 1-3 peak at 3.26GB during forward, drop to 0.02GB during loss
        GPU 0 holds everything: 4.49GB peak
```

---

## The Fix: Integrate Loss INTO the Model

```
═══════════════════════════════════════════════════════════════════════════════
                    SOLUTION 1: INTEGRATED LOSS (Target State)
═══════════════════════════════════════════════════════════════════════════════

New Flow:
─────────────────────────────────────────────────────────────────────────────────

                    ┌─────────────────────────────────────────┐
                    │     DataParallelWrapper                 │
                    │  ┌───────────────────────────────────┐  │
                    │  │  Original Model (forward pass)    │  │
                    │  └───────────────┬───────────────────┘  │
                    │                  ▼                      │
                    │  ┌───────────────────────────────────┐  │
                    │  │  Loss Computation (INSIDE!)       │  │
                    │  │  criterion(output, targets)       │  │
                    │  └───────────────┬───────────────────┘  │
                    │                  ▼                      │
                    │         return LOSS (scalar)            │
                    └─────────────────────────────────────────┘

When wrapped with nn.DataParallel:

GPU 0: [fwd + loss] ──► loss₀ (scalar) ────┐
GPU 1: [fwd + loss] ──► loss₁ (scalar) ────┼──► GPU 0: mean(loss₀,₁,₂,₃)
GPU 2: [fwd + loss] ──► loss₂ (scalar) ────┤
GPU 3: [fwd + loss] ──► loss₃ (scalar) ────┘

Key insight: DataParallel AVERAGES scalar outputs automatically!


═══════════════════════════════════════════════════════════════════════════════
                    TARGET TIMELINE (Balanced)
═══════════════════════════════════════════════════════════════════════════════

Time ─────────────────────────────────────────────────────────────────────────►

GPU 0: [scatter][████ fwd + loss ████][gather][═ backward ═][reduce][opt]
GPU 1: [receive][████ fwd + loss ████][ send ][═ backward ═][ send ][   ]
GPU 2: [receive][████ fwd + loss ████][ send ][═ backward ═][ send ][   ]
GPU 3: [receive][████ fwd + loss ████][ send ][═ backward ═][ send ][   ]

       └───┬───┘└────────┬───────────┘└──┬───┘└──────┬─────┘└──┬───┘
         ~5%         ~50%              ~5%        ~30%        ~10%
                      ▲                            ▲
                 ALL PARALLEL!               ALL PARALLEL!


Expected Speedup:
─────────────────────────────────────────────────────────────────────────────────

Before: ~60% serialized on GPU 0 → effective utilization ~40%
After:  ~10% serialized on GPU 0 → effective utilization ~90%

With 4 GPUs:
  Before: 1.0x speedup (no real parallelism)
  After:  ~3.2-3.6x speedup (realistic with overhead)
```

---

# Step-by-Step Implementation of Solution 1

- The implementation is based on @moe_flashattn_3.py

## Overview of Changes

| Step | File Location | What Changes | Why |
|------|---------------|--------------|-----|
| 1 | After line ~2860 | Add `clinical_collate_fn_v2` | Pre-compute multi-hot targets as tensors |
| 2 | After line ~2580 | Add `DataParallelWrapper` class | Integrate loss into model forward |
| 3 | Line ~8463 | Modify `_create_dataloaders` | Use new collate function |
| 4 | Line ~3872 | Modify `train_epoch` | Support wrapper model path |
| 5 | Line ~8720 | Modify `run_single_experiment` | Use wrapper for DataParallel |
| 6 | Line ~4115 | Modify `evaluate` | Handle wrapper during eval |

---

## Step 1: Add Enhanced Collate Function

**Location**: After `clinical_collate_fn` (around line 2855)

**Why**: The current collate function returns `target` as a Python list. DataParallel cannot scatter Python lists to GPUs. We need targets as pre-computed tensors.

```python
# ============================================================================
# ENHANCED COLLATE FUNCTION FOR DATAPARALLEL (Solution 1)
# ============================================================================

from functools import partial

def clinical_collate_fn_v2(batch: List[Dict], config: 'BaseConfig') -> Dict[str, Any]:
    """
    Enhanced collate function that pre-computes multi-hot targets as tensors.
    
    CRITICAL FOR DATAPARALLEL:
    - All outputs must be tensors (not Python lists)
    - Targets pre-computed to avoid GPU 0 bottleneck
    - dt_cnt as tensor for GPU scatter
    
    Args:
        batch: List of sample dicts from ClinicalDataset
        config: BaseConfig with len_dy, target_cd_cnt
    
    Returns:
        Dict with all tensor values suitable for DataParallel
    """
    batch_size = len(batch)
    len_dy = config.len_dy
    target_cd_cnt = config.target_cd_cnt
    
    # Stack standard tensors
    ages = torch.stack([item['age'] for item in batch])
    genders = torch.stack([item['gender'] for item in batch])
    lobs = torch.stack([item['lob'] for item in batch])
    codes = torch.stack([item['codes'] for item in batch])
    
    # Convert dt_cnt to tensor (was list before!)
    dt_cnts = torch.tensor([item['dt_cnt'] for item in batch], dtype=torch.long)
    
    # Pre-compute multi-hot targets: [batch, len_dy, target_cd_cnt]
    # This is the KEY change - targets become a tensor, not a nested list
    targets_multihot = torch.zeros(batch_size, len_dy, target_cd_cnt, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        target_list = item['target']  # List[List[int]] - len_dy x variable
        for day_idx, day_codes in enumerate(target_list):
            if day_idx < len_dy and day_codes:  # Check bounds and non-empty
                for code_idx in day_codes:
                    if 0 <= code_idx < target_cd_cnt:
                        targets_multihot[i, day_idx, code_idx] = 1.0
    
    # Keep original targets for metrics computation (backward compat)
    targets_list = [item['target'] for item in batch]
    
    return {
        'age': ages,                    # [batch, len_dy]
        'gender': genders,              # [batch, len_dy]
        'lob': lobs,                    # [batch, len_dy]
        'codes': codes,                 # [batch, len_dy, len_cd]
        'dt_cnt': dt_cnts,              # [batch] - NOW A TENSOR!
        'target_multihot': targets_multihot,  # [batch, len_dy, target_cd_cnt] - NEW!
        'target': targets_list          # List[List[List[int]]] - kept for metrics
    }


def create_collate_fn_v2(config: 'BaseConfig') -> Callable:
    """
    Factory to create collate function with config bound.
    
    Usage:
        collate_fn = create_collate_fn_v2(config)
        DataLoader(..., collate_fn=collate_fn)
    """
    return partial(clinical_collate_fn_v2, config=config)
```

---

## Step 2: Add DataParallelWrapper Class

**Location**: After model class definitions (around line 2580, after `FlashMoETransformer`)

**Why**: This wrapper moves loss computation INSIDE the forward pass, so each GPU computes its own loss. DataParallel then automatically averages the losses.

```python
# ============================================================================
# DATAPARALLEL WRAPPER WITH INTEGRATED LOSS (Solution 1)
# ============================================================================

class DataParallelWrapper(nn.Module):
    """
    Wrapper that integrates loss computation into the forward pass.
    
    PURPOSE:
    Standard DataParallel gathers outputs to GPU 0, then loss runs on GPU 0 only.
    This wrapper computes loss on EACH GPU, then DataParallel averages the losses.
    
    MECHANISM:
    1. Forward pass runs on each GPU (same as before)
    2. Loss computation runs on each GPU (NEW - parallel!)
    3. DataParallel gathers LOSS values (scalars), not full outputs
    4. Losses are automatically averaged across GPUs
    
    RESULT:
    - GPU 0 no longer bottlenecked by loss computation
    - All GPUs contribute equally to training
    - ~3-4x speedup with 4 GPUs
    
    Compatible with:
    - BaselineTransformer
    - FlashAttentionTransformer  
    - FlashMoETransformer
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: 'BaseConfig', 
        criterion: nn.Module,
        moe_config: Optional['MoEConfig'] = None
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = criterion
        self.moe_config = moe_config
        self.target_cd_cnt = config.target_cd_cnt
        
        # Detect model type
        self._is_moe = hasattr(model, 'moe_layers') or (
            hasattr(model, 'module') and hasattr(model.module, 'moe_layers')
        )
    
    def forward(
        self, 
        x: torch.Tensor,           # [batch, len_dy, features]
        dt_cnt: torch.Tensor,      # [batch] - valid days per sample
        targets: torch.Tensor,     # [batch, len_dy, target_cd_cnt] multi-hot
        return_predictions: bool = False  # For evaluation
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with integrated loss computation.
        
        Args:
            x: Input tensor [batch, len_dy, features]
            dt_cnt: Valid day counts [batch]
            targets: Pre-computed multi-hot targets [batch, len_dy, target_cd_cnt]
            return_predictions: If True, also return predictions for metrics
        
        Returns:
            If return_predictions=False: loss tensor (scalar per GPU, averaged by DP)
            If return_predictions=True: (loss, {'predictions': output, 'moe_losses': ...})
        """
        batch_size = x.shape[0]
        actual_len_dy = x.shape[1]
        device = x.device
        
        # ====== MODEL FORWARD ======
        if self._is_moe:
            output, moe_losses = self.model(x, return_moe_losses=True)
        else:
            output = self.model(x)
            moe_losses = {}
        
        # ====== LOSS COMPUTATION (ON THIS GPU!) ======
        # Flatten: [batch, len_dy, vocab] -> [batch * len_dy, vocab]
        output_flat = output.view(batch_size * actual_len_dy, self.target_cd_cnt)
        targets_flat = targets.view(batch_size * actual_len_dy, self.target_cd_cnt)
        
        # Create valid day mask
        # Each sample has dt_cnt[i] valid days; mask out padding days
        valid_mask = torch.zeros(
            batch_size * actual_len_dy, 
            dtype=torch.bool, 
            device=device
        )
        
        for i in range(batch_size):
            valid_days = min(int(dt_cnt[i].item()), actual_len_dy)
            if valid_days > 0:
                start_idx = i * actual_len_dy
                valid_mask[start_idx:start_idx + valid_days] = True
        
        # Compute loss only on valid positions
        if valid_mask.any():
            valid_output = output_flat[valid_mask]
            valid_targets = targets_flat[valid_mask]
            pred_loss = self.criterion(valid_output, valid_targets)
        else:
            # Edge case: no valid days (shouldn't happen with proper data)
            pred_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ====== MOE AUXILIARY LOSS ======
        aux_loss = moe_losses.get('aux_loss', torch.tensor(0.0, device=device))
        if aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        
        # Combine losses
        if self.moe_config and self.moe_config.load_balance_strategy == 'switch':
            total_loss = pred_loss + self.moe_config.aux_loss_weight * aux_loss
        else:
            total_loss = pred_loss
        
        if return_predictions:
            return total_loss, {
                'predictions': output,
                'pred_loss': pred_loss,
                'aux_loss': aux_loss,
                'moe_losses': moe_losses
            }
        else:
            return total_loss
    
    def get_inner_model(self) -> nn.Module:
        """Get the wrapped model (for checkpointing)."""
        return self.model
    
    def state_dict(self, *args, **kwargs):
        """Return inner model state dict for checkpoint compatibility."""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict to inner model."""
        return self.model.load_state_dict(state_dict, *args, **kwargs)
```

---

## Step 3: Modify `_create_dataloaders`

**Location**: Around line 8463

**What changes**: Add parameter to use enhanced collate function, and use it when DataParallel is enabled.

```python
def _create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: BaseConfig,
    use_bucketing: bool,
    world_size: int = 1,
    logger: Optional[logging.Logger] = None,
    use_enhanced_collate: bool = False  # NEW PARAMETER
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        config: Configuration with batch_size, len_dy, etc.
        use_bucketing: Whether to use length-based bucketing
        world_size: Number of processes (for DDP)
        logger: Optional logger
        use_enhanced_collate: If True, use collate_fn_v2 for DataParallel
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = ClinicalDataset(train_data, config)
    val_dataset = ClinicalDataset(val_data, config)
    
    n_workers = max(1, os.cpu_count() // max(world_size, 1) // 2)
    
    # Choose collate function based on mode
    if use_enhanced_collate:
        collate_fn = create_collate_fn_v2(config)
        if logger:
            logger.info("📦 Using enhanced collate_fn_v2 (pre-computed multi-hot targets)")
    else:
        collate_fn = clinical_collate_fn
    
    if use_bucketing:
        if logger:
            logger.info("Bucketing is ENABLED via BatchSampler.")
        train_batch_sampler = BucketingBatchSampler(
            data=train_data,
            batch_size=config.batch_size,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=collate_fn,  # Use selected collate
            persistent_workers=n_workers > 0  # Added for efficiency
        )
    else:
        if logger:
            logger.info("Using standard DataLoader (no bucketing).")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,  # Use selected collate
            persistent_workers=n_workers > 0
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn  # Use selected collate for val too
    )
    
    if logger:
        logger.info(f"Using DataLoader with {n_workers} workers.")
    
    return train_loader, val_loader
```

---

## Step 4: Modify `train_epoch`

**Location**: Around line 3872

**What changes**: Add a new code path for when using the DataParallelWrapper. The key difference is how data is passed to the model and how loss is handled.

Replace the entire `train_epoch` function with this:

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
    use_wrapper_model: bool = False  # NEW: Flag for DataParallelWrapper
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Supports two modes:
    1. Standard mode (use_wrapper_model=False): Original behavior
    2. Wrapper mode (use_wrapper_model=True): Uses DataParallelWrapper for efficient multi-GPU
    
    The wrapper mode expects:
    - batch['target_multihot']: Pre-computed multi-hot tensor
    - batch['dt_cnt']: Tensor (not list)
    - model: DataParallelWrapper or nn.DataParallel(DataParallelWrapper)
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
        
        # Progress logging
        if is_main and batch_idx % log_interval == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}')
        
        # GPU utilization check at first batch
        if batch_idx == 0 and is_main:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"\n🔍 GPU UTILIZATION CHECK (Batch 0):")
                for gpu_id in range(num_gpus):
                    mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"   GPU {gpu_id}: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        
        optimizer.zero_grad()
        
        # ================================================================
        # PATH A: WRAPPER MODEL (Efficient DataParallel)
        # ================================================================
        if use_wrapper_model:
            # Extract tensors - DON'T move to device yet!
            # DataParallel will handle device placement during scatter
            age = batch['age']
            gender = batch['gender']
            lob = batch['lob']
            codes = batch['codes']
            dt_cnt = batch['dt_cnt']           # Tensor from enhanced collate
            targets_mh = batch['target_multihot']  # Pre-computed multi-hot
            y = batch['target']                # Original list for metrics
            
            # Concatenate inputs
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),
                codes
            ], dim=-1)
            
            # Move to CUDA - DataParallel will scatter from here
            x = x.cuda(non_blocking=True)
            dt_cnt = dt_cnt.cuda(non_blocking=True)
            targets_mh = targets_mh.cuda(non_blocking=True)
            
            # Forward pass with integrated loss
            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    # Model returns loss directly (or loss + extras)
                    result = model(x, dt_cnt, targets_mh, return_predictions=True)
                    if isinstance(result, tuple):
                        total_loss, extras = result
                        output = extras.get('predictions', None)
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
                result = model(x, dt_cnt, targets_mh, return_predictions=True)
                if isinstance(result, tuple):
                    total_loss, extras = result
                    output = extras.get('predictions', None)
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
        
        # ================================================================
        # PATH B: ORIGINAL MODEL (Backward compatible)
        # ================================================================
        else:
            # Original code path - kept for non-DataParallel cases
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
            
            total_loss = torch.tensor(0.0, device=device)
            
            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if _model_has_moe(model):
                        output, moe_losses = model(x, return_moe_losses=True)
                    else:
                        output = model(x)
                        moe_losses = {}
                    
                    pred_loss = compute_loss(output, y, dt_cnt, config, criterion, device)
                    aux_loss = moe_losses.get('aux_loss', torch.tensor(0.0, device=device))
                    if aux_loss.numel() > 1:
                        aux_loss = aux_loss.mean()
                    if moe_config and moe_config.load_balance_strategy == 'switch':
                        total_loss = pred_loss + moe_config.aux_loss_weight * aux_loss
                    else:
                        total_loss = pred_loss
            else:
                if _model_has_moe(model):
                    output, moe_losses = model(x, return_moe_losses=True)
                else:
                    output = model(x)
                    moe_losses = {}
                
                pred_loss = compute_loss(output, y, dt_cnt, config, criterion, device)
                aux_loss = moe_losses.get('aux_loss', torch.tensor(0.0, device=device))
                if aux_loss.numel() > 1:
                    aux_loss = aux_loss.mean()
                if moe_config and moe_config.load_balance_strategy == 'switch':
                    total_loss = pred_loss + moe_config.aux_loss_weight * aux_loss
                else:
                    total_loss = pred_loss
        
        # ================================================================
        # BACKWARD PASS (Same for both paths)
        # ================================================================
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
        
        # ================================================================
        # LOGGING & CLEANUP
        # ================================================================
        global_step += 1
        
        # Track losses
        pred_loss_scalar = pred_loss.mean().item() if pred_loss.numel() > 1 else pred_loss.item()
        aux_loss_scalar = aux_loss.mean().item() if aux_loss.numel() > 1 else aux_loss.item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
        
        # Compute and log metrics
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                # For wrapper mode, we need to get predictions for metrics
                if use_wrapper_model and output is not None:
                    # Use the predictions from wrapper
                    batch_metrics = compute_batch_metrics_lightweight(
                        output, y, 
                        dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt,
                        config, device
                    )
                elif not use_wrapper_model:
                    batch_metrics = compute_batch_metrics_lightweight(
                        output, y, dt_cnt, config, device
                    )
                else:
                    # Fallback: minimal metrics
                    batch_metrics = {'recall@10': 0, 'recall@20': 0, 
                                   'precision@10': 0, 'precision@20': 0,
                                   'mAP@20': 0, 'mAP@50': 0, 'brier_score': 0}
                
                batch_metrics_buffer.append(batch_metrics)
                
                print(f"    Loss: {pred_loss_scalar:.4f} | "
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
        del x
        if 'output' in dir() and output is not None:
            del output
        del total_loss
        
        if batch_idx % 100 == 0:
            gc.collect()
            
            if is_main and device.type == 'cuda' and batch_idx % 1000 == 0:
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
                    print(f'    GPU {gpu_id}: {allocated:.2f}GB / {peak:.2f}GB peak')
    
    # End-of-epoch cleanup
    if device.type == 'cuda':
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    # Aggregate metrics
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

---

## Step 5: Modify `run_single_experiment` 

**Location**: Around line 8720 (the DataParallel section)

**What changes**: Use the new wrapper and pass flags to `train_epoch` and `_create_dataloaders`.

Find and replace the section from approximately line 8720 to 8800:

```python
    # ============================================================
    # DATAPARALLEL WRAPPER FOR MULTI-GPU (IMPROVED - Solution 1)
    # ============================================================    
    num_gpus = torch.cuda.device_count()
    use_data_parallel = num_gpus > 1
    use_wrapper_model = False
    use_enhanced_collate = False
    criterion = nn.BCEWithLogitsLoss()  # Define criterion early
    
    if use_data_parallel:
        logger.info(f"🚀 Enabling IMPROVED DataParallel with {num_gpus} GPUs")
        
        # Scale batch size proportionally
        effective_batch_size = config.batch_size * num_gpus
        
        # Scale learning rate (square root scaling)
        base_lr = config.learning_rate
        scaled_lr = base_lr * math.sqrt(num_gpus)
        
        logger.info(f"   Per-GPU batch size: {config.batch_size}")
        logger.info(f"   Effective batch size: {effective_batch_size}")
        logger.info(f"   Base learning rate: {base_lr}")
        logger.info(f"   Scaled learning rate: {scaled_lr:.2e}")
        
        # ====== KEY CHANGE: Wrap model with loss integration ======
        wrapped_model = DataParallelWrapper(
            model=model,
            config=config,
            criterion=criterion,
            moe_config=moe_config
        )
        
        # Then wrap with nn.DataParallel
        model = nn.DataParallel(wrapped_model)
        use_wrapper_model = True
        use_enhanced_collate = True
        
        logger.info(f"   ✅ Using DataParallelWrapper for integrated loss")
        logger.info(f"   DataParallel device_ids: {model.device_ids}")
        logger.info(f"   DataParallel output_device: {model.output_device}")
        
        # Update batch_size AFTER setting up wrapper
        config.batch_size = effective_batch_size
        
    else:
        scaled_lr = config.learning_rate
        logger.info(f"Single GPU mode (no DataParallel)")
    
    # Log config
    metrics_logger.log_config({
        'experiment': exp_name,
        'embedding_size': eff_d_model,
        'nhid': dims['nhid'],
        'nhead': dims['nhead'],
        'batch_size': config.batch_size,
        'effective_batch_size': effective_batch_size if use_data_parallel else config.batch_size,
        'use_mixed_precision': use_mixed_precision,
        'use_bucketing': use_bucketing,
        'use_learnt_att_pool': use_learnt_att_pool,
        'use_wrapper_model': use_wrapper_model,
        'moe_config': vars(moe_config) if moe_config else None
    })
    
    # ============================================================
    # 3. DATA PREPARATION
    # ============================================================
    if code_frequencies is None:
        code_frequencies = compute_code_frequencies(train_data, config, device)
    
    train_loader, val_loader = _create_dataloaders(
        train_data, val_data, config, use_bucketing, 
        logger=logger,
        use_enhanced_collate=use_enhanced_collate  # NEW PARAMETER
    )
    
    # ============================================================
    # 4. OPTIMIZER SETUP
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    # criterion already defined above
```

Then in the training loop (around line 8810), update the `train_epoch` call:

```python
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
            use_bucketing=use_bucketing,
            log_interval=log_metrics_every,
            global_step=global_step,
            loss_tracker=loss_tracker,
            is_main=is_main,
            use_ddp=use_ddp,
            use_wrapper_model=use_wrapper_model  # NEW PARAMETER
        )
```

---

## Step 6: Fix Checkpoint Saving/Loading for Wrapper

**Location**: Around line 4520 (save_checkpoint function)

**What changes**: Handle the double-wrapped model (DataParallel wrapping DataParallelWrapper).

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
    keep_last_n: int = 2,
    save_optimizer: bool = True
):
    """
    Save checkpoint with support for DataParallelWrapper.
    
    Handles three cases:
    1. Plain model
    2. nn.DataParallel(model)
    3. nn.DataParallel(DataParallelWrapper(model))
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ====== UNWRAP MODEL ======
    # Handle DataParallel
    if isinstance(model, nn.DataParallel):
        inner_model = model.module
    else:
        inner_model = model
    
    # Handle DataParallelWrapper
    if isinstance(inner_model, DataParallelWrapper):
        actual_model = inner_model.model
    else:
        actual_model = inner_model
    
    # Build checkpoint dict with ACTUAL model state
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': actual_model.state_dict(),  # Unwrapped state
        'optimizer_state_dict': optimizer.state_dict() if save_optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'timestamp': time.time(),
        'model_type': type(actual_model).__name__  # Track model type
    }
    
    # ... rest of the function stays the same ...
```

Similarly, update `load_checkpoint` (around line 4576):

```python
def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None
) -> Dict:
    """
    Load checkpoint with support for DataParallelWrapper.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # ====== UNWRAP MODEL ======
    if isinstance(model, nn.DataParallel):
        inner_model = model.module
    else:
        inner_model = model
    
    if isinstance(inner_model, DataParallelWrapper):
        actual_model = inner_model.model
    else:
        actual_model = inner_model
    
    # Load to actual model
    actual_model.load_state_dict(checkpoint['model_state_dict'])
    
    # ... rest stays the same ...
```

---

## Step 7: Fix Model Saving for Downstream Evaluation

**Location**: Around line 8100 (save_trained_model function)

```python
def save_trained_model(
    model: nn.Module,
    config: BaseConfig,
    model_name: str,
    save_dir: str,
    exp_results: Dict[str, any],
    checkpoint_dir: Optional[str] = None,
    is_best: bool = False,
    moe_config: Optional[MoEConfig] = None
) -> str:
    """Save trained model with DataParallelWrapper support."""
    os.makedirs(save_dir, exist_ok=True)
    
    # ====== UNWRAP MODEL ======
    actual_model = model
    if isinstance(actual_model, nn.DataParallel):
        actual_model = actual_model.module
    if isinstance(actual_model, DataParallelWrapper):
        actual_model = actual_model.model
    
    model_path = os.path.join(save_dir, f"{model_name}_final.pt")
    save_dict = {
        'model_state_dict': actual_model.state_dict(),
        'model_name': model_name,
        'model_type': type(actual_model).__name__,
        # ... rest of save_dict ...
    }
    # ... rest of function ...
```

---

## Summary: All Files/Functions Changed

| Function | Line (~) | Change Description |
|----------|----------|-------------------|
| `clinical_collate_fn_v2` | NEW after 2855 | New collate function returning tensors |
| `create_collate_fn_v2` | NEW after 2855 | Factory for new collate |
| `DataParallelWrapper` | NEW after 2580 | Wrapper class with integrated loss |
| `_create_dataloaders` | 8463 | Add `use_enhanced_collate` param |
| `train_epoch` | 3872 | Add `use_wrapper_model` path |
| `run_single_experiment` | 8720 | Use wrapper, pass new params |
| `save_checkpoint` | 4520 | Handle wrapped model |
| `load_checkpoint` | 4576 | Handle wrapped model |
| `save_trained_model` | 8100 | Handle wrapped model |

---

## Expected Outcome After Changes

```
═══════════════════════════════════════════════════════════════════════════════
                    AFTER SOLUTION 1: BALANCED GPU UTILIZATION
═══════════════════════════════════════════════════════════════════════════════

GPU Memory (Expected):
   GPU 0: ~1.5-2.0 GB allocated (model + activations for 64 samples)
   GPU 1: ~1.5-2.0 GB allocated (model + activations for 64 samples)
   GPU 2: ~1.5-2.0 GB allocated (model + activations for 64 samples)
   GPU 3: ~1.5-2.0 GB allocated (model + activations for 64 samples)

Timeline:
   GPU 0: [scatter][████ fwd+loss ████][gather][══ backward ══][reduce][opt]
   GPU 1: [receive][████ fwd+loss ████][ send ][══ backward ══][ send ][   ]
   GPU 2: [receive][████ fwd+loss ████][ send ][══ backward ══][ send ][   ]
   GPU 3: [receive][████ fwd+loss ████][ send ][══ backward ══][ send ][   ]

Expected Speedup: 2.5-3.5x compared to current (depending on overhead)
```
