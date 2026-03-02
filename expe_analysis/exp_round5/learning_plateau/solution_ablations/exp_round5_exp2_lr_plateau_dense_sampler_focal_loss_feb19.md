## Implementation Plan: Priority 1 & Priority 2
- Feb 19, 2026
Here's a complete guide organized by what to add and where.

### Overview

| Priority | Intervention | What it Does | Where in Notebook |
|----------|-------------|--------------|-------------------|
| **1** | Density-Aware Tier Batching | Replaces binary "has any tail code" with density-based sampling from top-20% tail-dense members | After `TierAwareBatchSampler` (after line ~6048) |
| **2** | Asymmetric Loss (ASL) | Different focusing for positives (γ+=0, keep all) vs negatives (γ-=4, aggressively down-weight easy) | After `FocalLoss` class (after line ~4277) |

---

## Priority 1: Density-Aware Tier Batch Sampler

### What the current code does wrong

The existing `TierAwareBatchSampler._build_sample_tier_mapping()` (lines 5870-5916) checks binary presence:

```5904:5916:dev/moe/moe_flashattn_4.py
            all_positive_codes = set()
            for day_codes in target_list:
                if day_codes:  # Non-empty day
                    all_positive_codes.update(day_codes)
            
            # Check tier membership - a member can be in multiple tier pools
            if all_positive_codes & medium_codes:
                self.samples_with_medium.append(idx)
            if all_positive_codes & rare_codes:
                self.samples_with_rare.append(idx)
            if all_positive_codes & tail_codes:
                self.samples_with_tail.append(idx)
```

Since 83.4% of members have at least one tail code, this pool is nearly the entire dataset. Sampling from it yields batches where tail codes are still only ~5% of occurrences.

### What the density sampler does instead

Instead of "does this member have any tail code?", it computes:

```
tail_density(member) = count_of_tail_code_occurrences / total_code_occurrences
```

Then it selects the **top-N% highest-density members** for the tail pool. These members have 15-20%+ tail codes in their sequences, so when they appear in a batch, they provide **meaningful tail gradient signal**.

### Step 1A: Add config fields to `OptimizeConfig`

**Where**: Inside the `OptimizeConfig` dataclass, after the existing tier-aware batching fields (after line 541). Add these fields right before the closing of the class.

```python
    # ============================================================
    # DENSITY-AWARE TIER BATCHING (Priority 1 - Feb 2026)
    # Replaces binary "has any tail code" with density-based selection.
    # Members in the tail pool must be in the top-N% by tail occurrence density.
    # ============================================================
    use_density_aware_batching: bool = False
    density_tail_percentile: float = 80.0     # Top 20% of members by tail density
    density_rare_percentile: float = 70.0     # Top 30% of members by rare density
    density_medium_percentile: float = 70.0   # Top 30% of members by medium density
```

**Functionality**: These fields control whether to use density-based pool selection and what percentile threshold to use. `density_tail_percentile=80` means only the top 20% of members by tail code density are eligible for the tail quota.

### Step 1B: Add `DensityTierAwareBatchSampler` class

**Where**: In a new cell right after the existing `TierAwareBatchSampler` class (after line 6048, before the `create_tier_aware_dataloader` function).

```python
class DensityTierAwareBatchSampler(Sampler):
    """
    Density-aware batch sampler that selects members with HIGH CONCENTRATION
    of rare/tail codes, not just binary presence.
    
    Key difference from TierAwareBatchSampler:
    - Old: tail_pool = members with >= 1 tail code (83.4% of members)
    - New: tail_pool = top 20% of members by tail code DENSITY
    
    Density = (number of tail code occurrences) / (total code occurrences)
    
    This ensures members selected for tail quota actually provide meaningful
    tail gradient signal (15-20%+ tail codes vs global 5.2%).
    
    Evidence basis: Jan 30 code frequency analysis showed 83.4% member-level 
    coverage but only 5.2% occurrence-level coverage for tail codes.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        code_frequencies: np.ndarray,
        batch_size: int,
        medium_quota: int = 0,
        rare_quota: int = 4,
        tail_quota: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
        percentile_boundaries: Tuple[float, float, float] = (20, 50, 80),
        density_tail_percentile: float = 80.0,
        density_rare_percentile: float = 70.0,
        density_medium_percentile: float = 70.0,
        verbose: bool = True
    ):
        """
        Args:
            dataset: ClinicalDataset with targets
            code_frequencies: Array of code occurrence counts
            batch_size: Total batch size
            medium_quota: Minimum high-density medium members per batch
            rare_quota: Minimum high-density rare members per batch
            tail_quota: Minimum high-density tail members per batch
            shuffle: Whether to shuffle within each pool
            drop_last: Whether to drop the last incomplete batch
            percentile_boundaries: (tail_thresh, rare_thresh, medium_thresh)
                for defining which CODES belong to which tier
            density_tail_percentile: Percentile threshold for tail member pool
                80.0 = only top 20% of members by tail density
            density_rare_percentile: Percentile threshold for rare member pool
            density_medium_percentile: Percentile threshold for medium member pool
            verbose: Print initialization statistics
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.medium_quota = medium_quota
        self.rare_quota = rare_quota
        self.tail_quota = tail_quota
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        self.density_tail_pct = density_tail_percentile
        self.density_rare_pct = density_rare_percentile
        self.density_medium_pct = density_medium_percentile
        
        total_quota = medium_quota + rare_quota + tail_quota
        assert total_quota <= batch_size, \
            f"Combined quotas ({total_quota}) exceed batch_size ({batch_size})"
        
        self._build_tier_indices(code_frequencies, percentile_boundaries)
        self._build_density_pools(verbose)
        self._calculate_num_batches()
    
    def _build_tier_indices(
        self,
        code_frequencies: np.ndarray,
        percentile_boundaries: Tuple[float, float, float]
    ):
        """Build tier code index sets based on frequency percentiles."""
        freq_nz = code_frequencies[code_frequencies > 0]
        if len(freq_nz) == 0:
            raise ValueError("No non-zero frequencies found in code_frequencies")
        
        percentiles = np.percentile(freq_nz, list(percentile_boundaries))
        self.tier_code_indices = {}
        
        self.tier_code_indices['common'] = set(
            np.where(code_frequencies > percentiles[2])[0]
        )
        self.tier_code_indices['medium'] = set(
            np.where((code_frequencies <= percentiles[2]) & 
                     (code_frequencies > percentiles[1]))[0]
        )
        self.tier_code_indices['rare'] = set(
            np.where((code_frequencies <= percentiles[1]) & 
                     (code_frequencies > percentiles[0]))[0]
        )
        self.tier_code_indices['tail'] = set(
            np.where((code_frequencies <= percentiles[0]) & 
                     (code_frequencies > 0))[0]
        )
        
        self.tier_thresholds = {
            'tail_upper': percentiles[0],
            'rare_upper': percentiles[1],
            'medium_upper': percentiles[2]
        }
    
    def _build_density_pools(self, verbose: bool):
        """
        Compute per-member density scores for each tier and build 
        high-density pools using percentile thresholds.
        
        Density = (occurrences of tier codes across all days) / (total occurrences)
        """
        medium_codes = self.tier_code_indices['medium']
        rare_codes = self.tier_code_indices['rare']
        tail_codes = self.tier_code_indices['tail']
        
        targets_list = self.dataset.targets
        
        tail_densities = np.zeros(self.num_samples, dtype=np.float32)
        rare_densities = np.zeros(self.num_samples, dtype=np.float32)
        medium_densities = np.zeros(self.num_samples, dtype=np.float32)
        
        tail_counts = np.zeros(self.num_samples, dtype=np.int32)
        rare_counts = np.zeros(self.num_samples, dtype=np.int32)
        medium_counts = np.zeros(self.num_samples, dtype=np.int32)
        total_counts = np.zeros(self.num_samples, dtype=np.int32)
        
        if verbose:
            print(f"DensityTierAwareBatchSampler: Computing density scores "
                  f"for {self.num_samples:,} members...")
        
        for idx in range(self.num_samples):
            if verbose and idx > 0 and idx % 500000 == 0:
                print(f"    Processed {idx:,}/{self.num_samples:,} members...")
            
            target_list = targets_list[idx]
            member_tail = 0
            member_rare = 0
            member_medium = 0
            member_total = 0
            
            for day_codes in target_list:
                if not day_codes:
                    continue
                for code in day_codes:
                    if code == 0:
                        continue
                    member_total += 1
                    if code in tail_codes:
                        member_tail += 1
                    elif code in rare_codes:
                        member_rare += 1
                    elif code in medium_codes:
                        member_medium += 1
            
            total_counts[idx] = member_total
            tail_counts[idx] = member_tail
            rare_counts[idx] = member_rare
            medium_counts[idx] = member_medium
            
            if member_total > 0:
                tail_densities[idx] = member_tail / member_total
                rare_densities[idx] = member_rare / member_total
                medium_densities[idx] = member_medium / member_total
        
        # Build pools using percentile thresholds on density scores
        # Only consider members with >0 tail occurrences for tail pool percentile
        tail_mask = tail_counts > 0
        rare_mask = rare_counts > 0
        medium_mask = medium_counts > 0
        
        # Compute percentile thresholds on members who HAVE codes from each tier
        if tail_mask.sum() > 0:
            tail_density_thresh = np.percentile(
                tail_densities[tail_mask], self.density_tail_pct
            )
        else:
            tail_density_thresh = 0.0
            
        if rare_mask.sum() > 0:
            rare_density_thresh = np.percentile(
                rare_densities[rare_mask], self.density_rare_pct
            )
        else:
            rare_density_thresh = 0.0
            
        if medium_mask.sum() > 0:
            medium_density_thresh = np.percentile(
                medium_densities[medium_mask], self.density_medium_pct
            )
        else:
            medium_density_thresh = 0.0
        
        # Select high-density members for each pool
        self.samples_with_tail = np.where(
            tail_densities >= tail_density_thresh
        )[0].tolist()
        self.samples_with_rare = np.where(
            rare_densities >= rare_density_thresh
        )[0].tolist()
        self.samples_with_medium = np.where(
            medium_densities >= medium_density_thresh
        )[0].tolist()
        self.general_samples = list(range(self.num_samples))
        
        if verbose:
            print(f"\n  Density thresholds:")
            print(f"    Tail:   density >= {tail_density_thresh:.4f} "
                  f"(top {100-self.density_tail_pct:.0f}%)")
            print(f"    Rare:   density >= {rare_density_thresh:.4f} "
                  f"(top {100-self.density_rare_pct:.0f}%)")
            print(f"    Medium: density >= {medium_density_thresh:.4f} "
                  f"(top {100-self.density_medium_pct:.0f}%)")
            
            print(f"\n  High-density pools:")
            print(f"    Tail pool:   {len(self.samples_with_tail):,} members "
                  f"({len(self.samples_with_tail)/self.num_samples:.1%})")
            print(f"    Rare pool:   {len(self.samples_with_rare):,} members "
                  f"({len(self.samples_with_rare)/self.num_samples:.1%})")
            print(f"    Medium pool: {len(self.samples_with_medium):,} members "
                  f"({len(self.samples_with_medium)/self.num_samples:.1%})")
            
            # Show density statistics for selected pool
            if len(self.samples_with_tail) > 0:
                pool_tail_densities = tail_densities[self.samples_with_tail]
                pool_tail_counts = tail_counts[self.samples_with_tail]
                pool_total_counts = total_counts[self.samples_with_tail]
                print(f"\n  Tail pool statistics:")
                print(f"    Avg tail density:    {pool_tail_densities.mean():.4f} "
                      f"(vs global {tail_densities[tail_mask].mean():.4f})")
                print(f"    Avg tail occurrences: {pool_tail_counts.mean():.1f} "
                      f"(vs global {tail_counts[tail_mask].mean():.1f})")
                print(f"    Avg total codes:     {pool_total_counts.mean():.1f}")
                
                # Estimate per-batch tail occurrence improvement
                est_tail_per_batch = (
                    self.tail_quota * pool_tail_counts.mean() +
                    (self.batch_size - self.tail_quota) * tail_counts.mean()
                )
                est_total_per_batch = (
                    self.tail_quota * pool_total_counts.mean() +
                    (self.batch_size - self.tail_quota) * total_counts.mean()
                )
                est_tail_frac = est_tail_per_batch / max(est_total_per_batch, 1)
                print(f"    Estimated tail occurrence fraction per batch: "
                      f"{est_tail_frac:.2%} (target: >5%)")
        
        self._density_stats = {
            'tail_density_threshold': float(tail_density_thresh),
            'rare_density_threshold': float(rare_density_thresh),
            'medium_density_threshold': float(medium_density_thresh),
            'tail_pool_size': len(self.samples_with_tail),
            'rare_pool_size': len(self.samples_with_rare),
            'medium_pool_size': len(self.samples_with_medium),
            'tail_pool_avg_density': float(
                tail_densities[self.samples_with_tail].mean()
            ) if len(self.samples_with_tail) > 0 else 0.0,
        }
    
    def _calculate_num_batches(self):
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with guaranteed high-density tier representation."""
        if self.shuffle:
            medium_pool = self.samples_with_medium.copy()
            rare_pool = self.samples_with_rare.copy()
            tail_pool = self.samples_with_tail.copy()
            general_pool = self.general_samples.copy()
            random.shuffle(medium_pool)
            random.shuffle(rare_pool)
            random.shuffle(tail_pool)
            random.shuffle(general_pool)
        else:
            medium_pool = self.samples_with_medium.copy()
            rare_pool = self.samples_with_rare.copy()
            tail_pool = self.samples_with_tail.copy()
            general_pool = self.general_samples.copy()
        
        used_samples = set()
        medium_idx = 0
        rare_idx = 0
        tail_idx = 0
        general_idx = 0
        batches_yielded = 0
        
        while batches_yielded < self.num_batches:
            batch = []
            
            if self.medium_quota > 0:
                medium_added = 0
                while medium_added < self.medium_quota and medium_idx < len(medium_pool):
                    sample_idx = medium_pool[medium_idx]
                    medium_idx += 1
                    if sample_idx not in used_samples:
                        batch.append(sample_idx)
                        used_samples.add(sample_idx)
                        medium_added += 1
            
            rare_added = 0
            while rare_added < self.rare_quota and rare_idx < len(rare_pool):
                sample_idx = rare_pool[rare_idx]
                rare_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    rare_added += 1
            
            tail_added = 0
            while tail_added < self.tail_quota and tail_idx < len(tail_pool):
                sample_idx = tail_pool[tail_idx]
                tail_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    tail_added += 1
            
            remaining = self.batch_size - len(batch)
            while remaining > 0 and general_idx < len(general_pool):
                sample_idx = general_pool[general_idx]
                general_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    remaining -= 1
            
            # Pool exhaustion handling with reshuffle
            if self.medium_quota > 0 and medium_idx >= len(medium_pool):
                medium_pool = self.samples_with_medium.copy()
                if self.shuffle:
                    random.shuffle(medium_pool)
                medium_idx = 0
            
            if rare_idx >= len(rare_pool):
                rare_pool = self.samples_with_rare.copy()
                if self.shuffle:
                    random.shuffle(rare_pool)
                rare_idx = 0
            
            if tail_idx >= len(tail_pool):
                tail_pool = self.samples_with_tail.copy()
                if self.shuffle:
                    random.shuffle(tail_pool)
                tail_idx = 0
            
            if general_idx >= len(general_pool):
                general_pool = self.general_samples.copy()
                if self.shuffle:
                    random.shuffle(general_pool)
                general_idx = 0
                used_samples.clear()
            
            if len(batch) >= self.batch_size or (not self.drop_last and len(batch) > 0):
                if self.shuffle:
                    random.shuffle(batch)
                yield batch[:self.batch_size]
                batches_yielded += 1
    
    def __len__(self) -> int:
        return self.num_batches
    
    def get_density_stats(self) -> dict:
        """Return density statistics for logging/analysis."""
        return self._density_stats
```

**Functionality breakdown**:

1. **`_build_tier_indices`** — Same as original: classifies which *codes* belong to which tier based on frequency percentiles.

2. **`_build_density_pools`** — The core difference. For every member, it:
   - Counts total code occurrences across all days
   - Counts occurrences in each tier (tail, rare, medium)
   - Computes density = tier_occurrences / total_occurrences
   - Selects only members above the Nth percentile of density for each tier pool
   - Prints estimated per-batch tail occurrence fraction so you can verify the fix is working

3. **`__iter__`** — Identical batching logic to the original (quota-based filling), but now drawing from the *density-filtered* pools instead of the binary-presence pools.

### Step 1C: Update `_create_dataloaders` to support density mode

**Where**: Modify the `_create_dataloaders` function (line 11704). Inside the `if use_tier_aware:` block (line 11761), add a branch that checks for density mode.

Add this block right after the existing `use_tier_aware` check at line 11761, replacing the `TierAwareBatchSampler` instantiation:

```python
    if use_tier_aware:
        if code_frequencies is None:
            raise ValueError("code_frequencies required when use_tier_aware_batching=True")
        
        use_density = (
            optimize_config is not None and
            getattr(optimize_config, 'use_density_aware_batching', False)
        )
        
        if use_density:
            if logger:
                logger.info(f"Using DENSITY-AWARE tier batching "
                           f"(medium={optimize_config.tier_medium_quota}, "
                           f"rare={optimize_config.tier_rare_quota}, "
                           f"tail={optimize_config.tier_tail_quota}, "
                           f"tail_pct={optimize_config.density_tail_percentile})")
            
            train_batch_sampler = DensityTierAwareBatchSampler(
                dataset=train_dataset,
                code_frequencies=code_frequencies,
                batch_size=config.batch_size,
                medium_quota=optimize_config.tier_medium_quota,
                rare_quota=optimize_config.tier_rare_quota,
                tail_quota=optimize_config.tier_tail_quota,
                shuffle=True,
                drop_last=True,
                density_tail_percentile=optimize_config.density_tail_percentile,
                density_rare_percentile=optimize_config.density_rare_percentile,
                density_medium_percentile=optimize_config.density_medium_percentile,
                verbose=True
            )
        else:
            if logger:
                logger.info(f"Using TIER-AWARE batching (binary presence) "
                           f"(medium={optimize_config.tier_medium_quota}, "
                           f"rare={optimize_config.tier_rare_quota}, "
                           f"tail={optimize_config.tier_tail_quota})")
            
            train_batch_sampler = TierAwareBatchSampler(
                dataset=train_dataset,
                code_frequencies=code_frequencies,
                batch_size=config.batch_size,
                medium_quota=optimize_config.tier_medium_quota,
                rare_quota=optimize_config.tier_rare_quota,
                tail_quota=optimize_config.tier_tail_quota,
                shuffle=True,
                drop_last=True,
                verbose=True
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=False
        )
```

**Important compatibility note**: I noticed a pre-existing bug at line 12211 where `run_single_experiment` passes `use_tier_aware=use_tier_aware` to `_create_dataloaders`, but the function signature doesn't have that parameter. You'll need to remove that kwarg from the call at line 12211. The function already computes `use_tier_aware` internally from `optimize_config.use_tier_aware_batching`.

### Step 1D: Usage example

Here's how you'd configure and run with density-aware batching:

```python
optimize_config = OptimizeConfig(
    # Existing settings
    use_pos_weight=True,
    pos_weight_method='log_scaled',
    pos_weight_max=35,
    
    # Enable tier-aware + density mode
    use_tier_aware_batching=True,
    use_density_aware_batching=True,
    tier_medium_quota=0,
    tier_rare_quota=8,
    tier_tail_quota=12,    # Aggressive tail quota
    
    # Density thresholds
    density_tail_percentile=80.0,   # Top 20% by tail density
    density_rare_percentile=70.0,   # Top 30% by rare density
    density_medium_percentile=70.0, # Top 30% by medium density
)
```

---

## Priority 2: Asymmetric Loss (ASL)

### Why ASL over existing FocalLoss

The existing `FocalLoss` (lines 4185-4276) uses the **same** gamma for positives and negatives. In your multi-label setting with extreme imbalance:
- **Positives are precious** (tail codes have 0.00004% probability) — you do NOT want to down-weight them
- **Negatives dominate** (most of the 75K codes are negative for any given day) — you WANT to aggressively down-weight easy negatives

ASL fixes this with separate gamma parameters: γ+=0 (keep ALL positive gradients) and γ-=4 (strongly down-weight easy negatives). It also adds a probability clipping margin that prevents very easy negatives from contributing at all.

### Step 2A: Add `AsymmetricLoss` class

**Where**: In a new cell right after the `FocalLoss` class (after line 4276, before the "Weighted loss function" section at line 4279).

```python
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (Ridnik et al., 2021).
    
    Designed specifically for multi-label problems with long-tail distributions.
    Uses different focusing parameters for positives vs negatives:
    
    L_+ = (1 - p)^γ+ × BCE(p, 1)      [positives]
    L_- = (p_m)^γ-   × BCE(p_m, 0)     [negatives, with margin]
    
    where p_m = max(p - m, 0) is the probability shifted by margin m.
    
    Key advantages over standard Focal Loss:
    - γ+ = 0: Preserves ALL positive gradients (critical for tail codes)
    - γ- = 4: Aggressively down-weights easy negatives (common codes already learned)
    - Margin m: Hard-thresholds very easy negatives to zero contribution
    
    Reference:
        "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., ICCV 2021)
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma_pos: Focusing parameter for positive examples.
                0.0 = no down-weighting (preserve all positive gradients)
                1.0+ = down-weight easy positives
            gamma_neg: Focusing parameter for negative examples.
                4.0 = strong down-weighting of easy negatives (recommended)
                2.0 = moderate
            clip: Probability margin for negatives. Shifts p down by this amount 
                before computing loss. Effectively zeros out contribution from 
                negatives with p < clip. Set 0.0 to disable.
            pos_weight: Optional per-class weights [num_classes] for additional
                frequency-based reweighting (can combine with ASL)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: [batch, ..., num_classes] raw model outputs (before sigmoid)
            targets: [batch, ..., num_classes] binary targets (0 or 1)
        
        Returns:
            ASL loss (scalar if reduction='mean' or 'sum')
        """
        if targets.dtype != logits.dtype:
            targets = targets.to(logits.dtype)
        
        # Compute probabilities
        p = torch.sigmoid(logits)
        
        # Separate positive and negative
        pos_mask = targets.bool()
        neg_mask = ~pos_mask
        
        # For negatives: apply probability margin (clipping)
        p_neg = p.clone()
        if self.clip > 0:
            p_neg = (p_neg - self.clip).clamp(min=0.0)
        
        # Compute per-element loss
        # Positive: -log(p) weighted by (1-p)^γ+
        # Negative: -log(1-p_m) weighted by (p_m)^γ-
        loss_pos = torch.zeros_like(logits)
        loss_neg = torch.zeros_like(logits)
        
        # Positive loss: (1-p)^γ+ × (-log(p))
        p_pos_safe = p.clamp(min=1e-7, max=1.0 - 1e-7)
        if self.gamma_pos > 0:
            loss_pos = ((1 - p_pos_safe) ** self.gamma_pos) * (-torch.log(p_pos_safe))
        else:
            loss_pos = -torch.log(p_pos_safe)
        
        # Negative loss: (p_m)^γ- × (-log(1 - p_m))
        p_neg_safe = p_neg.clamp(min=1e-7, max=1.0 - 1e-7)
        if self.gamma_neg > 0:
            loss_neg = (p_neg_safe ** self.gamma_neg) * (-torch.log(1 - p_neg_safe))
        else:
            loss_neg = -torch.log(1 - p_neg_safe)
        
        # Combine: positive loss where target=1, negative loss where target=0
        loss = targets * loss_pos + (1 - targets) * loss_neg
        
        # Apply per-class pos_weight if provided
        if self.pos_weight is not None:
            if self.pos_weight.device != loss.device:
                self.pos_weight = self.pos_weight.to(loss.device)
            loss = loss * self.pos_weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

**Functionality breakdown**:

1. **Positive loss path**: With γ+=0, this is just `-log(p)` — standard BCE for positives with NO down-weighting. Every positive example (including tail codes) contributes full gradient.

2. **Negative loss path**: With γ-=4, easy negatives (where p is close to 0 = already correctly classified as negative) get down-weighted by `p^4`. A common code that the model already predicts well (p=0.01 when y=0) has its loss scaled by `0.01^4 = 10^-8` — effectively zero. This prevents "gradient flooding" from easy common-code negatives.

3. **Probability clipping**: With clip=0.05, negatives with p < 0.05 get their probability shifted to 0 before loss computation, making their contribution exactly zero. This provides a hard cutoff that complements the soft γ- down-weighting.

4. **pos_weight compatibility**: Can be combined with existing per-class frequency weights for additional rebalancing.

### Step 2B: Add ASL config fields to `OptimizeConfig`

**Where**: Inside `OptimizeConfig`, after the existing focal loss fields (after line 516).

```python
    # ============================================================
    # ASYMMETRIC LOSS (Priority 2 - Feb 2026)
    # Ridnik et al. 2021 - different gamma for pos vs neg
    # Designed for multi-label with long-tail distributions
    # ============================================================
    use_asl: bool = False
    asl_gamma_pos: float = 0.0    # 0 = keep ALL positive gradients
    asl_gamma_neg: float = 4.0    # 4 = aggressively down-weight easy negatives
    asl_clip: float = 0.05        # Probability margin for negatives
```

### Step 2C: Update `create_criterion` to support ASL

**Where**: Modify the `create_criterion` function (line 4473). The change goes in the "Step 2: Create criterion" section (line 4523). Replace the if/else with a three-way check:

```python
    # Step 2: Create criterion (ASL > Focal > BCE priority)
    if optimize_config.use_asl:
        criterion = AsymmetricLoss(
            gamma_pos=optimize_config.asl_gamma_pos,
            gamma_neg=optimize_config.asl_gamma_neg,
            clip=optimize_config.asl_clip,
            pos_weight=pos_weight,
            reduction='mean'
        )
        loss_name = (f"AsymmetricLoss(γ+={optimize_config.asl_gamma_pos}, "
                     f"γ-={optimize_config.asl_gamma_neg}, "
                     f"clip={optimize_config.asl_clip})")
    elif optimize_config.use_focal_loss:
        criterion = FocalLoss(
            gamma=optimize_config.focal_gamma,
            alpha=optimize_config.focal_alpha,
            pos_weight=pos_weight,
            reduction='mean'
        )
        loss_name = f"FocalLoss(gamma={optimize_config.focal_gamma}, alpha={optimize_config.focal_alpha})"
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_name = "BCEWithLogitsLoss"
```

Also update the logging in `run_single_experiment` (around line 12126) to handle ASL:

```python
    if optimize_config is not None and (optimize_config.use_pos_weight or 
                                         optimize_config.use_focal_loss or
                                         getattr(optimize_config, 'use_asl', False)):
        criterion = create_criterion(
            code_frequencies=code_frequencies,
            device=device,
            optimize_config=optimize_config
        )
        if getattr(optimize_config, 'use_asl', False):
            logger.info(f"Using AsymmetricLoss (γ+={optimize_config.asl_gamma_pos}, "
                        f"γ-={optimize_config.asl_gamma_neg})")
        elif optimize_config.use_focal_loss:
            logger.info(f"Using FocalLoss (gamma={optimize_config.focal_gamma})")
        else:
            logger.info("Using BCEWithLogitsLoss")
        if optimize_config.use_pos_weight:
            logger.info(f"  With pos_weight method: {optimize_config.pos_weight_method}")
```

### Step 2D: Usage example

```python
optimize_config = OptimizeConfig(
    # ASL replaces BCE + pos_weight
    use_asl=True,
    asl_gamma_pos=0.0,   # Keep ALL positive gradients
    asl_gamma_neg=4.0,    # Strongly down-weight easy negatives
    asl_clip=0.05,        # Zero out very easy negatives
    
    # Can still combine with pos_weight for additional reweighting
    use_pos_weight=True,
    pos_weight_method='log_scaled',
    pos_weight_max=35,
    
    # Density-aware batching (Priority 1)
    use_tier_aware_batching=True,
    use_density_aware_batching=True,
    tier_tail_quota=12,
    tier_rare_quota=8,
    density_tail_percentile=80.0,
)
```

---

## Summary: Integration Points and Compatibility

| Component | File Location | Modifies Existing? | Backward Compatible? |
|-----------|--------------|-------------------|---------------------|
| `DensityTierAwareBatchSampler` | After line 6048 | No (new class) | Yes — only used when `use_density_aware_batching=True` |
| `AsymmetricLoss` | After line 4276 | No (new class) | Yes — only used when `use_asl=True` |
| `OptimizeConfig` fields | Lines 541, 516 | Adds fields with defaults | Yes — defaults are `False`, existing configs unchanged |
| `create_criterion` update | Line 4523 | Adds ASL branch before existing code | Yes — falls through to existing BCE/Focal |
| `_create_dataloaders` update | Line 11761 | Adds density branch inside tier-aware block | Yes — only activates when both flags are True |

**All existing experiment configurations will behave identically** because every new flag defaults to `False`. The new code paths only activate when explicitly enabled.

### Bug to Fix First

Before running, remove `use_tier_aware=use_tier_aware,` from the `_create_dataloaders` call at line 12211 in `run_single_experiment`. That parameter doesn't exist in the function signature and will cause a `TypeError`. The function already computes `use_tier_aware` internally from `optimize_config.use_tier_aware_batching`.

### Recommended First Experiment Config

For your first training run combining both priorities:

```python
optimize_config = OptimizeConfig(
    # Priority 1: Density-aware batching
    use_tier_aware_batching=True,
    use_density_aware_batching=True,
    tier_medium_quota=0,
    tier_rare_quota=8,
    tier_tail_quota=12,
    density_tail_percentile=80.0,
    density_rare_percentile=70.0,
    density_medium_percentile=70.0,
    
    # Priority 2: ASL (replaces BCE + pos_weight)
    use_asl=True,
    asl_gamma_pos=0.0,
    asl_gamma_neg=4.0,
    asl_clip=0.05,
    use_pos_weight=True,
    pos_weight_method='log_scaled',
    pos_weight_max=35,
    
    # Keep gradient tier analysis ON to verify improvement
    enable_gradient_tier_analysis=True,
    
    # Existing scheduler settings
    scheduler_type='onecycle',
    warmup_pct=0.15,
    min_lr_ratio=0.01,
)
```

**Success criteria** (from the evidence synthesis):
- `train_grad_tier_tail_frac` > 5% at end of training (was 0.1%)
- Tail embedding std increases from 0.03 toward 0.10+
- `tail_top10_acc` > 0% (any movement off zero)
- `common_top10_acc` degradation < 3%