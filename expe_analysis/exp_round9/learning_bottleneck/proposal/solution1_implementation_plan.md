# Two-Stage Decoupled Training + Co-occurrence Embedding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Break the 0% tail_top10_acc floor by (Phase 1) freezing the encoder and re-training the decoder with code-specific balanced batches, and (Phase 2) initializing code embeddings from co-occurrence statistics to address embedding homogenization.

**Architecture:** The current model is a hierarchical clinical transformer with a shared encoder producing `h ∈ ℝ^d` and a single linear decoder `nn.Linear(d, 6297)`. Stage 1 trains end-to-end as today. Stage 2 freezes the entire encoder and trains only the decoder with code-aware batch construction, ensuring every code — including tail codes — gets batches with sufficient positive examples. Phase 2 adds pre-computed PPMI+SVD code embeddings before Stage 1 to break the embedding homogenization vicious cycle (tail std=0.03 vs common std=0.27).

**Tech Stack:** PyTorch, NumPy, SciPy (sparse SVD), existing `moe_flashattn_4.py` infrastructure

**Key Reference File:** `dev/moe/moe_flashattn_4.py` (18,468 lines) — all line numbers below refer to this file unless stated otherwise.

**Execution Context:** You will manually apply these code changes in the Vertex AI Workbench environment. Each task specifies exactly which lines to modify, what code to add, and why.

---

## Phase 1: Two-Stage Decoupled Training (Solution 1)

### Overview

Phase 1 implements the most well-validated technique for long-tail classification (Kang et al., ICLR 2020). The core insight: the encoder representation `h` likely contains *some* discriminative signal for tail codes — the decoder is the bottleneck because it was trained under the same imbalanced gradient regime and never learned to extract that signal. Re-training only the decoder with balanced data gives tail codes dedicated gradient signal without corrupting the encoder.

**Critical design decision from the review:** The class-balanced sampler must be **code-specific**, not tier-specific. V5 proved tier-level sampling is insufficient. For each batch during Stage 2, we target specific codes and ensure positive examples for those codes appear in the batch.

**AdamW interaction (from review):** For Stage 2, use SGD with momentum instead of AdamW. AdamW's second-moment denominator suppresses sporadic tail gradient spikes — exactly the signal we need to preserve during decoder re-training.

---

### Task 1: Add Stage 2 Configuration Dataclass

**Why:** Stage 2 needs its own hyperparameters separate from Stage 1 (different LR, optimizer, epochs, sampling strategy). A dedicated config keeps this clean and explicit.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — insert after the existing `OptimizeConfig` dataclass

**Where to insert:** Find the `OptimizeConfig` dataclass (search for `class OptimizeConfig`). Insert the new dataclass immediately after it ends.

**Step 1: Add the dataclass**

```python
@dataclass
class Stage2Config:
    """
    Configuration for Stage 2: Decoupled decoder re-training.
    
    Stage 2 freezes the encoder and re-trains the decoder with
    code-balanced sampling to break the tail_top10_acc = 0% floor.
    """
    enabled: bool = False
    
    # Training
    learning_rate: float = 5e-5
    epochs: int = 3
    optimizer: str = 'sgd'          # SGD avoids AdamW second-moment suppression of sporadic tail gradients
    momentum: float = 0.9
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Decoder re-initialization
    reinit_rare_decoder: bool = True   # Re-init rare+tail decoder rows (remove learned suppression bias)
    reinit_tiers: tuple = ('rare', 'tail')  # Which tiers to re-initialize
    reinit_method: str = 'xavier'      # 'xavier' or 'kaiming'
    
    # Code-balanced sampling
    codes_per_batch: int = 16          # Number of target codes to enrich per batch
    positives_per_code: int = 8        # Min positive examples per target code
    batch_size: int = 128              # Total batch size (positives + random negatives)
    
    # Scheduler
    scheduler: str = 'cosine'
    warmup_fraction: float = 0.1
    
    # Monitoring
    log_interval: int = 100
    eval_every_n_batches: int = 500    # Evaluate more frequently than Stage 1
    
    # Decoder architecture (Option A/B/C from proposal)
    decoder_type: str = 'linear'       # 'linear' (Option A), 'per_tier' (B), 'mlp' (C)
```

**Step 2: Verify no syntax errors**

After adding, scroll through the file to verify the indentation and that the closing of the previous dataclass is clean.

---

### Task 2: Build the Code-Aware Batch Sampler

**Why:** This is the **single most critical missing piece** identified in the review. The fundamental problem is that any specific tail code appears in ~0.064 batches under random sampling. No loss reweighting can create information from physically absent observations. The code-aware sampler ensures every code gets batches with sufficient positive examples.

**How it works:**
1. Pre-index which patients contain each code (one-time scan)
2. Each batch: pick N target codes (cycling through all codes with inverse-frequency priority), sample patients who have those codes as positives, fill remaining slots with random patients
3. This guarantees at least `positives_per_code` positive examples for the target codes — even for tail codes that appear in only 15-57 patients total

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — insert a new class near the existing sampler code

**Where to insert:** Search for `class TierAwareBatchSampler` or the existing sampler classes. Insert the new sampler nearby (around the data loading section, after line ~3312).

**Step 1: Add the CodeBalancedBatchSampler class**

```python
class CodeBalancedBatchSampler(Sampler):
    """
    Batch sampler that ensures specific target codes have positive examples in each batch.
    
    For Stage 2 decoder re-training: cycles through all codes, sampling patients
    who contain the target code as positive examples, filling remaining batch slots
    with random patients as negatives.
    
    This addresses the fundamental information bottleneck: tail codes have ~0.064
    appearances per random batch. This sampler guarantees >= positives_per_code
    positive appearances per target code per batch.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        code_frequencies: np.ndarray,
        codes_per_batch: int = 16,
        positives_per_code: int = 8,
        batch_size: int = 128,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.codes_per_batch = codes_per_batch
        self.positives_per_code = positives_per_code
        self.rng = np.random.RandomState(seed)
        
        self.num_codes = len(code_frequencies)
        self.active_codes = np.where(code_frequencies > 0)[0]
        
        # Build code-to-patient index: which patients have each code?
        # This is the one-time cost that makes code-aware batching possible
        print(f"  Building code-to-patient index for {len(self.active_codes)} active codes...")
        self.code_to_patients = self._build_code_index(dataset)
        
        # Sampling weights: inverse frequency so tail codes are sampled more often
        freq = code_frequencies[self.active_codes].astype(np.float64)
        inv_freq = 1.0 / np.maximum(freq, 1.0)
        self.code_weights = inv_freq / inv_freq.sum()
        
        # All patient indices for negative sampling
        self.all_indices = np.arange(len(dataset))
        
        # Estimate number of batches per epoch: cycle through each active code at least once
        self.num_batches = max(len(self.active_codes) // codes_per_batch, 1) * 3  # 3 passes
        
        print(f"  CodeBalancedBatchSampler ready:")
        print(f"    Active codes: {len(self.active_codes)}")
        print(f"    Codes/batch: {codes_per_batch}, Positives/code: {positives_per_code}")
        print(f"    Batches/epoch: {self.num_batches}")
    
    def _build_code_index(self, dataset: Dataset) -> Dict[int, np.ndarray]:
        """Scan dataset to map each code to patient indices that contain it."""
        code_to_patients = defaultdict(list)
        
        for idx in range(len(dataset)):
            item = dataset[idx]
            targets = item['target']  # nested list: [[codes_day0], [codes_day1], ...]
            unique_codes = set()
            for day_codes in targets:
                for code in day_codes:
                    if code != 0:
                        unique_codes.add(code)
            for code in unique_codes:
                code_to_patients[code].append(idx)
        
        # Convert to numpy arrays for fast sampling
        result = {}
        for code, patients in code_to_patients.items():
            result[code] = np.array(patients, dtype=np.int64)
        
        codes_with_patients = len(result)
        min_patients = min(len(v) for v in result.values()) if result else 0
        max_patients = max(len(v) for v in result.values()) if result else 0
        print(f"    Codes with patients: {codes_with_patients}")
        print(f"    Patients per code: min={min_patients}, max={max_patients}")
        
        return result
    
    def __iter__(self):
        for _ in range(self.num_batches):
            # 1. Sample target codes (inverse-frequency weighted)
            target_code_indices = self.rng.choice(
                len(self.active_codes),
                size=min(self.codes_per_batch, len(self.active_codes)),
                replace=False,
                p=self.code_weights
            )
            target_codes = self.active_codes[target_code_indices]
            
            # 2. For each target code, sample positive patients
            positive_indices = set()
            for code in target_codes:
                if code in self.code_to_patients:
                    patients = self.code_to_patients[code]
                    n_sample = min(self.positives_per_code, len(patients))
                    sampled = self.rng.choice(patients, size=n_sample, replace=(n_sample > len(patients)))
                    positive_indices.update(sampled.tolist())
            
            # 3. Fill remaining batch slots with random patients
            n_positives = len(positive_indices)
            n_negatives = max(self.batch_size - n_positives, 0)
            
            if n_negatives > 0:
                negative_indices = self.rng.choice(
                    self.all_indices, size=n_negatives, replace=False
                )
                batch_indices = list(positive_indices) + negative_indices.tolist()
            else:
                batch_indices = list(positive_indices)[:self.batch_size]
            
            # Shuffle to prevent the model from learning positional patterns
            self.rng.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches
```

**Step 2: Verify the sampler integrates with DataLoader**

The sampler yields lists of indices per batch, compatible with `DataLoader(batch_sampler=...)`.

---

### Task 3: Add Encoder Freezing and Decoder Re-initialization Utilities

**Why:** Stage 2 requires (a) freezing ALL encoder parameters so only the decoder trains, and (b) re-initializing rare/tail decoder rows which have learned actively harmful weights (`w_j^T h ≈ -8.5` suppression).

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — add utility functions near the model classes

**Where to insert:** After the `DataParallelWrapper` class (after line ~1212). These are model-level utilities.

**Step 1: Add encoder freezing function**

```python
def freeze_encoder(model: nn.Module) -> Tuple[int, int]:
    """
    Freeze all encoder parameters, leaving only decoder_cd trainable.
    
    Freezes: embedding_cd, embedding_gender_cd, embedding_age_in_months,
             embedding_lob, daily_pooling/encoder, all temporal_layers, norm,
             dropout, mm (GELU activation)
    
    Returns: (frozen_count, trainable_count)
    """
    # Unwrap DataParallel / DataParallelWrapper
    actual_model = model
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    if isinstance(actual_model, DataParallelWrapper):
        actual_model = actual_model.model
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in actual_model.named_parameters():
        if 'decoder_cd' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    return frozen_count, trainable_count


def reinit_decoder_rows(
    model: nn.Module,
    code_frequencies: np.ndarray,
    tiers_to_reinit: Tuple[str, ...] = ('rare', 'tail'),
    method: str = 'xavier'
):
    """
    Re-initialize decoder_cd weight rows for specified frequency tiers.
    
    The current decoder rows for tail codes have learned actively harmful weights
    (w_j^T h ≈ -8.5 suppression). Starting from clean initialization with
    balanced gradient gives the best chance of learning useful weights.
    
    Common/medium rows are left unchanged — they're already well-trained.
    """
    actual_model = model
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    if isinstance(actual_model, DataParallelWrapper):
        actual_model = actual_model.model
    
    decoder = actual_model.decoder_cd
    
    # Compute tier boundaries (same logic as GradientTierAnalyzer, line 5228-5265)
    freq_nz = code_frequencies[code_frequencies > 0]
    percentiles = np.percentile(freq_nz, [20, 50, 80])
    
    tier_masks = {
        'tail': (code_frequencies <= percentiles[0]) & (code_frequencies > 0),
        'rare': (code_frequencies <= percentiles[1]) & (code_frequencies > percentiles[0]),
        'medium': (code_frequencies <= percentiles[2]) & (code_frequencies > percentiles[1]),
        'common': code_frequencies > percentiles[2],
    }
    
    total_reinit = 0
    with torch.no_grad():
        for tier_name in tiers_to_reinit:
            if tier_name not in tier_masks:
                continue
            mask = tier_masks[tier_name]
            indices = np.where(mask)[0]
            
            if method == 'xavier':
                # Xavier uniform, scaled to match decoder init (line 2360-2362)
                fan_in = decoder.weight.shape[1]  # embedding_size
                std = (2.0 / (fan_in + 1)) ** 0.5
                decoder.weight.data[indices] = torch.randn_like(
                    decoder.weight.data[indices]
                ) * std
            elif method == 'kaiming':
                fan_in = decoder.weight.shape[1]
                std = (2.0 / fan_in) ** 0.5
                decoder.weight.data[indices] = torch.randn_like(
                    decoder.weight.data[indices]
                ) * std
            
            # Reset biases to near-zero (not the learned suppression value)
            decoder.bias.data[indices] = 0.0
            total_reinit += len(indices)
            
            print(f"  Re-initialized {len(indices)} decoder rows for tier '{tier_name}' "
                  f"(method={method})")
    
    print(f"  Total re-initialized: {total_reinit} / {decoder.weight.shape[0]} decoder rows")
    print(f"  Kept frozen: {decoder.weight.shape[0] - total_reinit} rows (common/medium)")
```

---

### Task 4: Add the Stage 2 Training Function

**Why:** Stage 2 has fundamentally different training dynamics from Stage 1: frozen encoder, code-balanced batching, SGD optimizer, shorter epochs, more frequent evaluation. A dedicated function keeps the logic clean and separable.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — add after the existing `train_epoch` function (after line ~5945)

**Step 1: Add `train_stage2` function**

```python
def train_stage2(
    model: nn.Module,
    train_dataset: Dataset,
    val_loader: DataLoader,
    code_frequencies: np.ndarray,
    stage2_config: Stage2Config,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    scaler: Optional[GradScaler] = None,
    metrics_logger: Optional['MetricsLogger'] = None,
    logger: Optional[logging.Logger] = None,
    gradient_tier_analyzer: Optional['GradientTierAnalyzer'] = None
) -> Dict[str, Any]:
    """
    Stage 2: Decoupled decoder re-training with code-balanced sampling.
    
    1. Freeze encoder
    2. Re-initialize rare/tail decoder rows
    3. Create code-balanced batch sampler
    4. Train decoder only with SGD (avoids AdamW second-moment suppression)
    5. Evaluate frequently with tier-stratified metrics
    
    Returns:
        Dict with Stage 2 training results and metrics history
    """
    log = logger.info if logger else print
    
    log("=" * 80)
    log("STAGE 2: DECOUPLED DECODER RE-TRAINING")
    log("=" * 80)
    
    # ================================================================
    # STEP 1: FREEZE ENCODER
    # ================================================================
    frozen_count, trainable_count = freeze_encoder(model)
    log(f"  Frozen parameters: {frozen_count:,}")
    log(f"  Trainable parameters (decoder only): {trainable_count:,}")
    
    # ================================================================
    # STEP 2: RE-INITIALIZE RARE/TAIL DECODER ROWS
    # ================================================================
    if stage2_config.reinit_rare_decoder:
        reinit_decoder_rows(
            model=model,
            code_frequencies=code_frequencies,
            tiers_to_reinit=stage2_config.reinit_tiers,
            method=stage2_config.reinit_method
        )
    
    # ================================================================
    # STEP 3: CREATE CODE-BALANCED SAMPLER + DATALOADER
    # ================================================================
    sampler = CodeBalancedBatchSampler(
        dataset=train_dataset,
        code_frequencies=code_frequencies,
        codes_per_batch=stage2_config.codes_per_batch,
        positives_per_code=stage2_config.positives_per_code,
        batch_size=stage2_config.batch_size
    )
    
    stage2_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=create_collate_fn(config),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    log(f"  Stage 2 DataLoader: {len(stage2_loader)} batches/epoch")
    
    # ================================================================
    # STEP 4: CREATE SGD OPTIMIZER (decoder params only)
    # ================================================================
    # Collect only trainable (decoder) parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if stage2_config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=stage2_config.learning_rate,
            momentum=stage2_config.momentum,
            weight_decay=stage2_config.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=stage2_config.learning_rate,
            weight_decay=stage2_config.weight_decay
        )
    
    log(f"  Optimizer: {stage2_config.optimizer} (lr={stage2_config.learning_rate})")
    
    # ================================================================
    # STEP 5: CREATE SCHEDULER
    # ================================================================
    total_steps = len(stage2_loader) * stage2_config.epochs
    warmup_steps = int(total_steps * stage2_config.warmup_fraction)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )
    
    # Wrap with warmup if needed
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_steps]
        )
    
    log(f"  Scheduler: cosine with {warmup_steps} warmup steps, {total_steps} total")
    
    # ================================================================
    # STEP 6: TRAINING LOOP
    # ================================================================
    model.train()
    results_history = []
    global_step = 0
    
    # Get the criterion from the DataParallelWrapper
    actual_wrapper = model
    if isinstance(model, nn.DataParallel):
        actual_wrapper = model.module
    criterion = actual_wrapper.criterion if hasattr(actual_wrapper, 'criterion') else nn.BCEWithLogitsLoss()
    
    for epoch in range(stage2_config.epochs):
        log(f"\n  --- Stage 2 Epoch {epoch + 1}/{stage2_config.epochs} ---")
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(stage2_loader):
            optimizer.zero_grad(set_to_none=True)
            
            # Build input tensor (same as train_epoch lines 5571-5584)
            age = batch['age']
            gender = batch['gender']
            lob = batch['lob']
            codes = batch['codes']
            dt_cnt = batch['dt_cnt']
            targets_mh = batch['target_multihot']
            
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
            need_predictions = (batch_idx % stage2_config.log_interval == 0)
            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
            else:
                result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
            
            if isinstance(result, tuple):
                total_loss, extras = result
                pred_loss = extras.get('pred_loss', total_loss)
            else:
                total_loss = result
                pred_loss = total_loss
            
            if total_loss.numel() > 1:
                total_loss = total_loss.mean()
            
            # Backward pass
            if use_mixed_precision:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, stage2_config.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, stage2_config.gradient_clip
                )
                optimizer.step()
            
            scheduler.step()
            global_step += 1
            
            # Track loss
            loss_val = pred_loss.detach().mean().item() if pred_loss.numel() > 1 else pred_loss.detach().item()
            epoch_loss += loss_val
            num_batches += 1
            
            # Gradient tier analysis
            if gradient_tier_analyzer and batch_idx % stage2_config.log_interval == 0:
                tier_metrics = gradient_tier_analyzer.log_batch(model, batch_idx)
                if tier_metrics:
                    log(f"    [GradTier] Common: {tier_metrics.get('grad_tier_common_frac', 0)*100:.1f}% | "
                        f"Tail: {tier_metrics.get('grad_tier_tail_frac', 0)*100:.1f}%")
            
            # Logging
            if batch_idx % stage2_config.log_interval == 0:
                log(f"    Batch {batch_idx}/{len(stage2_loader)} | Loss: {loss_val:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                if metrics_logger:
                    metrics_logger.log_batch(epoch=epoch, batch=batch_idx, metrics={
                        'stage': 2,
                        'global_step': global_step,
                        'loss': loss_val,
                        'lr': optimizer.param_groups[0]['lr']
                    })
            
            # Memory cleanup
            del x, dt_cnt, targets_mh, total_loss, result
            if batch_idx % 100 == 0:
                gc.collect()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        log(f"  Stage 2 Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
        
        results_history.append({
            'epoch': epoch + 1,
            'stage': 2,
            'avg_loss': avg_loss,
            'global_step': global_step
        })
    
    # ================================================================
    # STEP 7: UNFREEZE MODEL FOR EVALUATION
    # ================================================================
    # Note: keep encoder frozen for final evaluation — we want to measure
    # the decoder's ability to predict given the fixed encoder representation
    
    return {
        'stage2_history': results_history,
        'final_loss': results_history[-1]['avg_loss'] if results_history else None,
        'total_steps': global_step,
        'config': vars(stage2_config)
    }
```

---

### Task 5: Integrate Stage 2 into `run_single_experiment`

**Why:** Stage 2 must be triggered after Stage 1 completes, using the same model, data, and infrastructure. The integration point is in `run_single_experiment` after the main training loop.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — modify `run_single_experiment` (starts at line 12963)

**Where to change:** There are two integration points:

#### 5a. Add `stage2_config` parameter to the function signature

**Location:** Line 12963-12984 — the function signature

**Change:** Add `stage2_config: Optional[Stage2Config] = None` as a new parameter.

Find this line:
```python
    optimize_config: Optional[OptimizeConfig] = None
    
) -> Dict[str, Any]:
```

Replace with:
```python
    optimize_config: Optional[OptimizeConfig] = None,
    stage2_config: Optional['Stage2Config'] = None
    
) -> Dict[str, Any]:
```

#### 5b. Insert Stage 2 execution after the training loop

**Location:** After the training loop ends (around line 13312, after `gradient_tier_analyzer.reset_epoch()`), and before the comprehensive evaluation section.

Find this block (around lines 13308-13312):
```python
        # Reset gradient tier analyzer for next epoch
        if gradient_tier_analyzer is not None:
            gradient_tier_analyzer.aggregate_epoch()  # Store epoch summary
            gradient_tier_analyzer.reset_epoch()
```

After the training loop's `for epoch in range(start_epoch, epochs):` block closes, insert:

```python
    # ============================================================
    # STAGE 2: DECOUPLED DECODER RE-TRAINING (optional)
    # ============================================================
    stage2_results = None
    if stage2_config is not None and stage2_config.enabled:
        logger.info("\n" + "=" * 80)
        logger.info("ENTERING STAGE 2: DECOUPLED DECODER RE-TRAINING")
        logger.info("=" * 80)
        
        # Re-create gradient tier analyzer for Stage 2 if enabled
        stage2_tier_analyzer = None
        if optimize_config and getattr(optimize_config, 'enable_gradient_tier_analysis', False):
            stage2_tier_analyzer = GradientTierAnalyzer(
                code_frequencies=code_frequencies,
                device=device,
                log_interval=stage2_config.log_interval
            )
        
        stage2_results = train_stage2(
            model=model,
            train_dataset=train_dataset,
            val_loader=val_loader,
            code_frequencies=code_frequencies,
            stage2_config=stage2_config,
            config=config,
            device=device,
            use_mixed_precision=use_mixed_precision,
            scaler=scaler,
            metrics_logger=metrics_logger,
            logger=logger,
            gradient_tier_analyzer=stage2_tier_analyzer
        )
        
        logger.info(f"Stage 2 complete: final_loss={stage2_results['final_loss']:.4f}")
```

**Important:** You will need to verify where the `for epoch` loop closes by examining the indentation carefully in the Workbench. The Stage 2 block should be at the same indentation level as the `for epoch` statement.

#### 5c. Include Stage 2 results in the final output

**Location:** Find where the final results dict is assembled (search for `final_results` or `return` at the end of `run_single_experiment`).

Add `stage2_results` to the returned dictionary:
```python
    if stage2_results is not None:
        final_results['stage2'] = stage2_results
```

---

### Task 6: Add Diagnostic Evaluation for Stage 2

**Why:** The key diagnostic for Stage 2 is: does the tail positive logit move from -14.69 toward the theoretical equilibrium of -6.2? This tells us whether the decoder is successfully learning to extract signal from `h`.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — add a diagnostic function near `compute_stratified_metrics` (line 9318)

**Where to insert:** After `compute_stratified_metrics` (around line 9411).

**Step 1: Add diagnostic function**

```python
def compute_stage2_diagnostics(
    model: nn.Module,
    val_loader: DataLoader,
    code_frequencies: np.ndarray,
    config: BaseConfig,
    device: torch.device,
    max_batches: int = 50
) -> Dict[str, float]:
    """
    Stage 2 diagnostics: measure logit distribution per tier.
    
    Key question: has the tail positive logit moved from -14.69 toward -6.2?
    """
    actual_model = model
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    if isinstance(actual_model, DataParallelWrapper):
        actual_model = actual_model.model
    
    # Tier boundaries
    freq_nz = code_frequencies[code_frequencies > 0]
    percentiles = np.percentile(freq_nz, [20, 50, 80])
    
    tier_masks = {
        'tail': torch.tensor((code_frequencies <= percentiles[0]) & (code_frequencies > 0)),
        'rare': torch.tensor((code_frequencies <= percentiles[1]) & (code_frequencies > percentiles[0])),
        'medium': torch.tensor((code_frequencies <= percentiles[2]) & (code_frequencies > percentiles[1])),
        'common': torch.tensor(code_frequencies > percentiles[2]),
    }
    
    tier_pos_logits = {t: [] for t in tier_masks}
    tier_neg_logits = {t: [] for t in tier_masks}
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            
            age = batch['age']
            gender = batch['gender']
            lob = batch['lob']
            codes = batch['codes']
            dt_cnt = batch['dt_cnt']
            targets_mh = batch['target_multihot']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),
                codes
            ], dim=-1).cuda(non_blocking=True)
            dt_cnt = dt_cnt.cuda(non_blocking=True)
            targets_mh = targets_mh.cuda(non_blocking=True)
            
            result = model(x, dt_cnt, targets_mh, return_predictions=True)
            if isinstance(result, tuple):
                _, extras = result
                logits = extras['predictions']
            else:
                continue
            
            # Flatten valid positions
            batch_size = x.shape[0]
            actual_len_dy = x.shape[1]
            logits_flat = logits.view(-1, config.target_cd_cnt).cpu()
            targets_flat = targets_mh.view(-1, config.target_cd_cnt).cpu()
            
            for tier_name, mask in tier_masks.items():
                tier_logits = logits_flat[:, mask]
                tier_targets = targets_flat[:, mask]
                
                pos_mask = tier_targets > 0.5
                neg_mask = tier_targets < 0.5
                
                if pos_mask.any():
                    tier_pos_logits[tier_name].extend(
                        tier_logits[pos_mask].tolist()
                    )
                if neg_mask.any():
                    # Sample negatives (too many to store all)
                    neg_vals = tier_logits[neg_mask]
                    sample_size = min(1000, len(neg_vals))
                    indices = torch.randperm(len(neg_vals))[:sample_size]
                    tier_neg_logits[tier_name].extend(
                        neg_vals[indices].tolist()
                    )
            
            del x, dt_cnt, targets_mh, logits
    
    model.train()
    
    diagnostics = {}
    for tier_name in tier_masks:
        pos = tier_pos_logits[tier_name]
        neg = tier_neg_logits[tier_name]
        
        diagnostics[f'{tier_name}_pos_logit_mean'] = np.mean(pos) if pos else float('nan')
        diagnostics[f'{tier_name}_pos_logit_std'] = np.std(pos) if pos else float('nan')
        diagnostics[f'{tier_name}_neg_logit_mean'] = np.mean(neg) if neg else float('nan')
        diagnostics[f'{tier_name}_margin'] = (
            (np.mean(pos) - np.mean(neg)) if pos and neg else float('nan')
        )
    
    return diagnostics
```

**Step 2: Call diagnostics before and after Stage 2**

In the Stage 2 integration block from Task 5b, add diagnostic calls:

```python
        # Before Stage 2: measure baseline logit distribution
        logger.info("  Running pre-Stage2 logit diagnostics...")
        pre_diagnostics = compute_stage2_diagnostics(
            model, val_loader, code_frequencies, config, device
        )
        logger.info(f"    PRE-S2 tail_pos_logit: {pre_diagnostics.get('tail_pos_logit_mean', 'N/A'):.2f}")
        logger.info(f"    PRE-S2 tail_margin: {pre_diagnostics.get('tail_margin', 'N/A'):.2f}")
```

And after `train_stage2` returns:

```python
        # After Stage 2: measure improved logit distribution
        logger.info("  Running post-Stage2 logit diagnostics...")
        post_diagnostics = compute_stage2_diagnostics(
            model, val_loader, code_frequencies, config, device
        )
        logger.info(f"    POST-S2 tail_pos_logit: {post_diagnostics.get('tail_pos_logit_mean', 'N/A'):.2f}")
        logger.info(f"    POST-S2 tail_margin: {post_diagnostics.get('tail_margin', 'N/A'):.2f}")
        
        stage2_results['pre_diagnostics'] = pre_diagnostics
        stage2_results['post_diagnostics'] = post_diagnostics
```

---

### Task 7: Create the Stage 2 Experiment Invocation

**Why:** You need a concrete experiment cell to run Stage 2 in the Vertex Workbench notebook.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — add at the end of the file, or create a new notebook cell

**Step 1: Add experiment invocation code**

```python
# ============================================================================
# EXPERIMENT: Stage 2 Decoupled Decoder Re-training
# ============================================================================
# Run AFTER a successful Stage 1 experiment (e.g., exp2b with R6 config)
# 
# This uses the same prepared_data and model from Stage 1.
# The stage2_config controls decoder re-training hyperparameters.

stage2_config = Stage2Config(
    enabled=True,
    learning_rate=5e-5,
    epochs=3,
    optimizer='sgd',
    momentum=0.9,
    weight_decay=1e-4,
    reinit_rare_decoder=True,
    reinit_tiers=('rare', 'tail'),
    reinit_method='xavier',
    codes_per_batch=16,
    positives_per_code=8,
    batch_size=128,
    log_interval=100,
    eval_every_n_batches=500
)

# Assuming prepared_data is already computed from Stage 1 cell:
results = run_single_experiment(
    exp_name='exp2b_flash_learned_pool',   # or whichever variant you use
    moe_config=None,                        # Dense model
    use_learnt_att_pool=True,
    prepared_data=prepared_data,
    train_data=df_train,
    device=torch.device('cuda'),
    epochs=1,                               # Stage 1: 1 epoch as current
    experiment_round='exp_round_stage2',
    log_metrics_every=500,
    embedding_size=256,                     # or 512 depending on your config
    optimize_config=OptimizeConfig(
        enable_gradient_tier_analysis=True,
        use_pos_weight=True,
        pos_weight_method='log_scaled',
    ),
    stage2_config=stage2_config
)

print(f"\nStage 2 Results:")
if results.get('stage2'):
    s2 = results['stage2']
    print(f"  Final loss: {s2['final_loss']:.4f}")
    if 'post_diagnostics' in s2:
        diag = s2['post_diagnostics']
        print(f"  Tail positive logit: {diag.get('tail_pos_logit_mean', 'N/A')}")
        print(f"  Tail margin: {diag.get('tail_margin', 'N/A')}")
```

---

## Phase 2: Co-occurrence Embedding Pre-training (Solution 3)

### Overview

Phase 2 addresses **Amplifier B: Embedding Homogenization** — the one structural barrier that Phase 1 does not touch. Tail code embeddings have std=0.03 (vs common std=0.27), meaning the encoder receives nearly identical input for different tail codes. Pre-computed embeddings from co-occurrence statistics break this by giving every code a unique embedding by construction.

**Execute Phase 2 only after evaluating Phase 1 results.** If Phase 1 Stage 2 shows no improvement in tail_top10_acc, the encoder representation likely lacks tail-specific features because the inputs were indistinguishable — and Phase 2 directly addresses that input-level barrier.

---

### Task 8: Build Co-occurrence Embedding Pre-computation Function

**Why:** Compute PPMI + SVD embeddings offline from training data. These capture which codes co-occur in the same patients, giving every code — even tail codes — a unique "signature" based on genuine domain information.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — add near the data preparation utilities (after `compute_code_frequencies`, around line 12029)

**Step 1: Add the pre-computation function**

```python
def compute_cooccurrence_embeddings(
    train_dataset: Dataset,
    embedding_dim: int,
    num_codes: int,
    target_norm: float = 1.4,
    window: str = 'patient',
    max_samples: Optional[int] = None
) -> np.ndarray:
    """
    Compute code embeddings from co-occurrence statistics via PPMI + SVD.
    
    For each pair of codes that appear in the same patient's history,
    increment the co-occurrence matrix. Then apply Positive Pointwise
    Mutual Information (PPMI) normalization and truncated SVD to produce
    d-dimensional embeddings.
    
    PPMI normalizes for frequency effects: even rare codes with 15 occurrences
    get meaningful PPMI values if they co-occur with specific codes more than chance.
    
    Levy & Goldberg (NIPS 2014) proved this is mathematically equivalent to
    Word2Vec skip-gram, but deterministic and reproducible.
    
    Args:
        train_dataset: ClinicalDataset or ClinicalDatasetLazy
        embedding_dim: Target embedding dimension (should match config.embedding_size)
        num_codes: Total code vocabulary size (config.cd_cnt, NOT target_cd_cnt)
        target_norm: L2 norm to scale embeddings to (match model's embedding scale)
        window: 'patient' (all codes in patient history) or 'day' (same-day only)
        max_samples: Limit number of patients to process (for debugging)
    
    Returns:
        embeddings: [num_codes, embedding_dim] numpy array
    """
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import svds
    
    print(f"Computing co-occurrence embeddings (dim={embedding_dim}, window={window})...")
    
    # Step 1: Build co-occurrence matrix (sparse for memory efficiency)
    cooccurrence = lil_matrix((num_codes, num_codes), dtype=np.float64)
    n_samples = min(len(train_dataset), max_samples) if max_samples else len(train_dataset)
    
    for idx in range(n_samples):
        if idx % 50000 == 0:
            print(f"  Processing patient {idx:,}/{n_samples:,}...")
        
        item = train_dataset[idx]
        codes_tensor = item['codes']  # [len_dy, len_cd]
        
        if window == 'patient':
            # All unique codes across all days for this patient
            all_codes = set()
            for day_idx in range(codes_tensor.shape[0]):
                day_codes = codes_tensor[day_idx].tolist()
                all_codes.update(c for c in day_codes if c != 0)
            
            all_codes = list(all_codes)
            for i in range(len(all_codes)):
                for j in range(i + 1, len(all_codes)):
                    ci, cj = all_codes[i], all_codes[j]
                    if ci < num_codes and cj < num_codes:
                        cooccurrence[ci, cj] += 1
                        cooccurrence[cj, ci] += 1
        
        elif window == 'day':
            for day_idx in range(codes_tensor.shape[0]):
                day_codes = [c for c in codes_tensor[day_idx].tolist() if c != 0]
                for i in range(len(day_codes)):
                    for j in range(i + 1, len(day_codes)):
                        ci, cj = day_codes[i], day_codes[j]
                        if ci < num_codes and cj < num_codes:
                            cooccurrence[ci, cj] += 1
                            cooccurrence[cj, ci] += 1
    
    cooccurrence = csr_matrix(cooccurrence)
    nnz = cooccurrence.nnz
    print(f"  Co-occurrence matrix: {num_codes}x{num_codes}, {nnz:,} non-zero entries")
    
    # Step 2: PPMI transformation
    # PPMI(i,j) = max(0, log(C[i,j] * N / (sum_k C[i,k] * sum_k C[k,j])))
    print("  Computing PPMI transformation...")
    
    row_sums = np.array(cooccurrence.sum(axis=1)).flatten()
    col_sums = np.array(cooccurrence.sum(axis=0)).flatten()
    total = cooccurrence.sum()
    
    if total == 0:
        print("  WARNING: Empty co-occurrence matrix. Returning random embeddings.")
        embeddings = np.random.randn(num_codes, embedding_dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) * target_norm
        return embeddings
    
    # Efficient PPMI: work with non-zero entries only
    ppmi = lil_matrix((num_codes, num_codes), dtype=np.float64)
    cx = cooccurrence.tocoo()
    
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if row_sums[i] > 0 and col_sums[j] > 0:
            pmi = np.log(v * total / (row_sums[i] * col_sums[j]))
            if pmi > 0:
                ppmi[i, j] = pmi
    
    ppmi = csr_matrix(ppmi)
    ppmi_nnz = ppmi.nnz
    print(f"  PPMI matrix: {ppmi_nnz:,} non-zero entries (from {nnz:,} co-occurrences)")
    
    # Step 3: Truncated SVD
    print(f"  Running truncated SVD (k={embedding_dim})...")
    k = min(embedding_dim, min(ppmi.shape) - 1)
    U, S, Vt = svds(ppmi, k=k)
    
    # Sort by decreasing singular value
    sort_idx = np.argsort(-S)
    U = U[:, sort_idx]
    S = S[sort_idx]
    
    # Embeddings = U * sqrt(S) (standard approach, balances U and V contributions)
    embeddings = U * np.sqrt(S)[np.newaxis, :]
    
    # Pad if k < embedding_dim
    if embeddings.shape[1] < embedding_dim:
        padding = np.zeros((num_codes, embedding_dim - embeddings.shape[1]))
        embeddings = np.concatenate([embeddings, padding], axis=1)
    
    # Step 4: L2-normalize to target norm
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms * target_norm
    
    embeddings = embeddings.astype(np.float32)
    
    # Step 5: Quality report
    # Measure embedding std per tier to verify distinctiveness
    print(f"\n  Embedding quality report:")
    print(f"    Shape: {embeddings.shape}")
    print(f"    Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.3f}")
    print(f"    Std of embeddings: {embeddings.std():.4f}")
    
    return embeddings
```

---

### Task 9: Add Embedding Initialization to Model Classes

**Why:** The model must accept pre-computed embeddings at initialization time to replace the default random initialization.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — modify each model class's `__init__` method

**Where to change:**

#### 9a. BaselineTransformer (line 2307)

Find:
```python
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
```

Replace with:
```python
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        
        # Support pre-computed embedding initialization
        if hasattr(config, 'pretrained_code_embeddings') and config.pretrained_code_embeddings is not None:
            with torch.no_grad():
                pretrained = config.pretrained_code_embeddings
                if isinstance(pretrained, np.ndarray):
                    pretrained = torch.from_numpy(pretrained)
                self.embedding_cd.weight.data.copy_(pretrained)
                print(f"  Initialized embedding_cd from pre-computed embeddings "
                      f"(std={self.embedding_cd.weight.data.std():.4f})")
```

#### 9b. FlashAttentionTransformer (line ~2482-2485)

Apply the **exact same change** after `self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)`.

#### 9c. FlashMoETransformer (line ~2710-2713)

Apply the **exact same change** after `self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)`.

#### 9d. Add the attribute to BaseConfig

Find the `BaseConfig` dataclass and add:

```python
    pretrained_code_embeddings: Optional[np.ndarray] = None
```

**Note:** You may need to add `import numpy as np` in the type annotation, or use `Optional[Any]` if NumPy type hints cause issues in the dataclass.

---

### Task 10: Add Staged Unfreezing for Embeddings

**Why:** If embeddings are unfrozen immediately, gradient starvation may re-homogenize them within the first few thousand steps. Staged unfreezing (freeze 50% of training, then unfreeze with 0.1× LR) follows BERT fine-tuning best practice and ensures the encoder builds features around the distinctive embeddings before they can degrade.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — add to the training loop (`train_epoch`)

**Where to change:** Inside `train_epoch` (line 5476), after the optimizer step (around line 5702).

**Step 1: Add embedding unfreezing config to Stage2Config or OptimizeConfig**

Find the `OptimizeConfig` dataclass and add:

```python
    # Embedding staged unfreezing (Phase 2 only)
    use_pretrained_embeddings: bool = False
    freeze_embeddings_fraction: float = 0.5    # Freeze for first 50% of training
    embedding_lr_multiplier: float = 0.1       # 0.1× of global LR when unfrozen
```

**Step 2: Add unfreezing logic to `train_epoch`**

Inside `train_epoch`, after the optimizer step block (around line 5702, after `global_step += 1`), add:

```python
            # Staged embedding unfreezing (Phase 2: co-occurrence embeddings)
            if (optimize_config is not None and 
                getattr(optimize_config, 'use_pretrained_embeddings', False)):
                
                total_train_steps = len(dataloader) * 1  # single epoch
                unfreeze_step = int(total_train_steps * optimize_config.freeze_embeddings_fraction)
                
                if global_step == unfreeze_step:
                    # Unfreeze embedding_cd with reduced LR
                    actual_model = model
                    if isinstance(model, nn.DataParallel):
                        actual_model = model.module
                    if isinstance(actual_model, DataParallelWrapper):
                        actual_model = actual_model.model
                    
                    for param in actual_model.embedding_cd.parameters():
                        param.requires_grad = True
                    
                    # Add embedding params to optimizer with reduced LR
                    emb_lr = optimizer.param_groups[0]['lr'] * optimize_config.embedding_lr_multiplier
                    optimizer.add_param_group({
                        'params': list(actual_model.embedding_cd.parameters()),
                        'lr': emb_lr
                    })
                    
                    if is_main:
                        print(f"\n  🔓 Unfreezing embedding_cd at step {global_step} "
                              f"(LR={emb_lr:.2e}, {optimize_config.embedding_lr_multiplier}× global)")
```

**Step 3: Freeze embeddings at model creation time**

In `run_single_experiment`, after the model is created (around line 13033-13042), add:

```python
    # Freeze embeddings if using pre-trained embeddings (Phase 2)
    if optimize_config and getattr(optimize_config, 'use_pretrained_embeddings', False):
        actual = model
        for param in actual.embedding_cd.parameters():
            param.requires_grad = False
        logger.info("  🔒 Froze embedding_cd (will unfreeze at "
                    f"{optimize_config.freeze_embeddings_fraction*100:.0f}% of training)")
```

---

### Task 11: Create Phase 2 Experiment Invocation

**Why:** Concrete experiment cell combining co-occurrence embeddings with Stage 2 re-training.

**Files:**
- Modify: `dev/moe/moe_flashattn_4.py` — add experiment cell at the end

**Step 1: Add experiment code**

```python
# ============================================================================
# EXPERIMENT: Phase 2 — Co-occurrence Embeddings + Stage 2 Decoder Re-training
# ============================================================================
# Run after Phase 1 to test if embedding homogenization was the barrier.
#
# This experiment:
# 1. Pre-computes PPMI+SVD embeddings from training co-occurrence
# 2. Initializes embedding_cd with these embeddings (frozen for first 50% of training)
# 3. Runs Stage 1 with staged unfreezing
# 4. Runs Stage 2 with code-balanced decoder re-training

# Step 1: Pre-compute embeddings (one-time, ~5-10 min on CPU)
pretrained_embeddings = compute_cooccurrence_embeddings(
    train_dataset=prepared_data.train_dataset,
    embedding_dim=256,          # match config.embedding_size
    num_codes=75516,            # match config.cd_cnt
    target_norm=1.4,
    window='patient'            # co-occurrence across full patient history
)

# Verify quality: tail embedding std should be >0.10
code_freq = prepared_data.code_frequencies
freq_nz = code_freq[code_freq > 0]
p20 = np.percentile(freq_nz, 20)
tail_mask = (code_freq <= p20) & (code_freq > 0)
tail_emb_std = pretrained_embeddings[tail_mask].std()
print(f"Tail embedding std: {tail_emb_std:.4f} (target: >0.10, current random init: ~0.03)")

# Step 2: Set config with pre-trained embeddings
optimize_config_phase2 = OptimizeConfig(
    enable_gradient_tier_analysis=True,
    use_pos_weight=True,
    pos_weight_method='log_scaled',
    use_pretrained_embeddings=True,
    freeze_embeddings_fraction=0.5,
    embedding_lr_multiplier=0.1
)

# Inject embeddings into config (before model creation)
# Note: BaseConfig needs the pretrained_code_embeddings attribute (added in Task 9d)

stage2_config_phase2 = Stage2Config(
    enabled=True,
    learning_rate=5e-5,
    epochs=3,
    optimizer='sgd',
    reinit_rare_decoder=True,
    reinit_tiers=('rare', 'tail'),
    codes_per_batch=16,
    positives_per_code=8,
    batch_size=128
)

# Step 3: Run experiment
# You will need to set config.pretrained_code_embeddings = pretrained_embeddings
# BEFORE calling run_single_experiment. This may require modifying the config
# creation inside _create_model, or passing it through the prepared_data path.
# 
# Simplest approach: after _create_model returns, manually set:
#   model.embedding_cd.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
#   for param in model.embedding_cd.parameters():
#       param.requires_grad = False

results_phase2 = run_single_experiment(
    exp_name='exp2b_flash_learned_pool',
    moe_config=None,
    use_learnt_att_pool=True,
    prepared_data=prepared_data,
    train_data=df_train,
    device=torch.device('cuda'),
    epochs=1,
    experiment_round='exp_round_phase2',
    log_metrics_every=500,
    embedding_size=256,
    optimize_config=optimize_config_phase2,
    stage2_config=stage2_config_phase2
)
```

---

## Decision Points and Success Criteria

### After Phase 1 (Tasks 1-7):

| Metric | Baseline | Target | Action if Not Met |
|--------|----------|--------|-------------------|
| `tail_pos_logit_mean` | -14.69 | > -10.0 | Proceed to Phase 2 |
| `tail_top10_acc` | 0% | > 0% (any improvement) | Breakthrough confirmed |
| `tail_margin` | ~1.76 | > 3.0 | Decoder learning signal exists |
| `common_top10_acc` | 85.9% | > 84.0% | If drops below, encoder may be affected (shouldn't with frozen encoder) |

### After Phase 2 (Tasks 8-11):

| Metric | Baseline | Target | Action if Not Met |
|--------|----------|--------|-------------------|
| `tail_embedding_std` (pre-training) | 0.03 | > 0.10 | Embedding pre-computation issue |
| `tail_embedding_std` (post-training) | 0.03 | > 0.05 | Staged unfreezing insufficient; keep frozen |
| `tail_top10_acc` (Phase 1+2) | 0% | > 0% | Fundamental architectural limit reached |

---

## Summary of All Code Changes

| Task | Location in `moe_flashattn_4.py` | What | Lines to Change |
|------|-----------------------------------|------|-----------------|
| 1 | After `OptimizeConfig` dataclass | Add `Stage2Config` dataclass | Insert new code |
| 2 | After line ~3312 (samplers section) | Add `CodeBalancedBatchSampler` class | Insert new code |
| 3 | After line ~1212 (after `DataParallelWrapper`) | Add `freeze_encoder` + `reinit_decoder_rows` | Insert new code |
| 4 | After line ~5945 (after `train_epoch`) | Add `train_stage2` function | Insert new code |
| 5a | Line 12983 | Add `stage2_config` parameter | Modify signature |
| 5b | After training loop (~line 13312) | Insert Stage 2 execution block | Insert new code |
| 5c | End of `run_single_experiment` | Add `stage2_results` to return dict | Modify existing |
| 6 | After line ~9411 | Add `compute_stage2_diagnostics` | Insert new code |
| 7 | End of file | Add Stage 2 experiment cell | Insert new code |
| 8 | After line ~12029 | Add `compute_cooccurrence_embeddings` | Insert new code |
| 9a-c | Lines 2307, ~2483, ~2711 | Add pre-trained embedding init to 3 model classes | Modify existing |
| 9d | `BaseConfig` dataclass | Add `pretrained_code_embeddings` field | Modify existing |
| 10 | `OptimizeConfig` + `train_epoch` + `run_single_experiment` | Add staged unfreezing logic | Modify existing + insert |
| 11 | End of file | Add Phase 2 experiment cell | Insert new code |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `CodeBalancedBatchSampler._build_code_index` is slow for large datasets | One-time cost; use ClinicalDatasetLazy to avoid loading all targets into memory |
| Stage 2 SGD optimizer diverges | Conservative LR (5e-5), gradient clipping (1.0), cosine schedule with warmup |
| Co-occurrence matrix is too large for memory | Using SciPy sparse matrices; 75K × 75K sparse matrix is manageable |
| Tail embedding std doesn't improve after Phase 2 | Keep embeddings frozen entirely (Option A from proposal) instead of staged unfreezing |
| Common code performance regresses in Stage 2 | Encoder is frozen — impossible by construction |
