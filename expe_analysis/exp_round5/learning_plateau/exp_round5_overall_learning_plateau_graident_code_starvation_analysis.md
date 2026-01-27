---
title: "Gradient Contribution Analysis Implementation"
date: 2026-01-23
---
# Gradient Contribution Analysis Implementation

## Overview

This analysis measures how much each code frequency tier contributes to the total gradient magnitude during training. It will help you understand:
1. **Whether rare codes are being ignored** (small gradient contribution despite high loss weight)
2. **Whether common codes dominate learning** (gradient saturation)
3. **The effective learning balance** across tiers

---

## Implementation Plan

You need to make changes in **4 locations** in `moe_flashattn_3.py`:

| Location | Purpose |
|----------|---------|
| **1. New class** (after line ~4350) | `GradientTierAnalyzer` class |
| **2. train_epoch signature** (line ~4825) | Add `gradient_analyzer` parameter |
| **3. Inside train_epoch loop** (after line ~4983-4985) | Add per-batch gradient contribution computation |
| **4. run_single_experiment** (line ~10832) | Initialize and use the analyzer |

---

## PART 1: New Class - `GradientTierAnalyzer`

**Add this code AFTER line 4338** (after the `compute_tiered_weights` function):

```python
# ============================================================
# GRADIENT CONTRIBUTION ANALYSIS BY CODE TIER
# ============================================================

@dataclass
class TierGradientMetrics:
    """Metrics for a single training step."""
    step: int
    tier_grad_norms: Dict[str, float]       # Tier name -> L2 gradient norm
    tier_grad_mean: Dict[str, float]        # Tier name -> mean absolute gradient
    tier_loss_contribution: Dict[str, float] # Tier name -> loss contribution
    total_grad_norm: float
    timestamp: float

class GradientTierAnalyzer:
    """
    Analyze gradient contribution by code frequency tier.
    
    This class computes how much each tier of codes (ultra_rare, tail, rare, 
    medium, common, very_common) contributes to the gradient during training.
    
    Key Insight: If rare tiers have tiny gradient contribution despite high 
    loss weights, the model is effectively ignoring them (gradient saturation 
    or dominated by common codes).
    
    Usage:
        analyzer = GradientTierAnalyzer(code_frequencies, device, config)
        
        # In training loop, after loss.backward():
        grad_metrics = analyzer.compute_tier_gradients(
            model, output, targets, valid_mask, criterion, step
        )
        
        # After training:
        analyzer.export_results("gradient_analysis.json")
        analyzer.print_summary()
    """
    
    def __init__(
        self,
        code_frequencies: np.ndarray,
        device: torch.device,
        target_cd_cnt: int,
        tier_config: Optional[Dict] = None,
        track_every_n_steps: int = 100  # Only track every N steps for efficiency
    ):
        """
        Initialize the analyzer.
        
        Args:
            code_frequencies: Array of shape [target_cd_cnt] with code frequencies
            device: Torch device
            target_cd_cnt: Number of target codes
            tier_config: Optional custom tier configuration
            track_every_n_steps: Track gradients every N steps (default 100)
        """
        self.device = device
        self.target_cd_cnt = target_cd_cnt
        self.track_every_n_steps = track_every_n_steps
        
        # Define tier boundaries based on percentiles
        if tier_config is None:
            tier_config = {
                'ultra_rare':  {'percentile': (0, 5),    'weight': 100},
                'tail':        {'percentile': (5, 25),   'weight': 50},
                'rare':        {'percentile': (25, 50),  'weight': 25},
                'medium':      {'percentile': (50, 75),  'weight': 10},
                'common':      {'percentile': (75, 90),  'weight': 3},
                'very_common': {'percentile': (90, 100), 'weight': 1},
            }
        
        self.tier_config = tier_config
        
        # Build tier masks (which codes belong to which tier)
        self.tier_masks = self._build_tier_masks(code_frequencies)
        
        # Storage for metrics over training
        self.history: List[TierGradientMetrics] = []
        
        # Summary statistics
        self.tier_names = list(tier_config.keys())
        
        print(f"\n📊 GradientTierAnalyzer initialized:")
        print(f"   Target codes: {target_cd_cnt}")
        print(f"   Track every: {track_every_n_steps} steps")
        for tier_name, mask in self.tier_masks.items():
            count = mask.sum().item()
            pct = 100.0 * count / target_cd_cnt
            print(f"   {tier_name:<12}: {count:>5} codes ({pct:.1f}%)")
    
    def _build_tier_masks(self, code_frequencies: np.ndarray) -> Dict[str, torch.Tensor]:
        """Build boolean masks for each tier."""
        masks = {}
        
        # Get non-zero frequencies for percentile calculation
        freq_nz = code_frequencies[code_frequencies > 0]
        if len(freq_nz) == 0:
            freq_nz = code_frequencies + 1e-10
        
        for tier_name, config in self.tier_config.items():
            p_low, p_high = config['percentile']
            
            # Calculate frequency thresholds
            thresh_low = np.percentile(freq_nz, p_low) if p_low > 0 else 0
            thresh_high = np.percentile(freq_nz, p_high) if p_high < 100 else np.inf
            
            # Create mask
            if p_low == 0:
                # Include zero-frequency codes in lowest tier
                mask = ((code_frequencies >= thresh_low) & 
                        (code_frequencies < thresh_high)) | (code_frequencies == 0)
            else:
                mask = ((code_frequencies >= thresh_low) & 
                        (code_frequencies < thresh_high))
            
            masks[tier_name] = torch.tensor(mask, dtype=torch.bool, device=self.device)
        
        return masks
    
    def should_track(self, step: int) -> bool:
        """Check if we should track this step."""
        return step % self.track_every_n_steps == 0
    
    def compute_tier_gradients(
        self,
        output: torch.Tensor,          # [batch*len_dy, target_cd_cnt] logits
        targets: torch.Tensor,         # [batch*len_dy, target_cd_cnt] multi-hot
        valid_mask: torch.Tensor,      # [batch*len_dy] boolean
        criterion: nn.Module,          # Loss function
        step: int,
        compute_full_breakdown: bool = True
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradient contribution by tier.
        
        This uses a technique called "per-output gradient probing":
        - We compute loss for each tier separately
        - Measure the gradient norm for each tier's contribution
        
        Args:
            output: Model output logits (DETACHED - we'll enable grad locally)
            targets: Target multi-hot tensor
            valid_mask: Mask for valid positions
            criterion: Loss function (BCEWithLogitsLoss with pos_weight)
            step: Current training step
            compute_full_breakdown: If True, compute detailed breakdown
        
        Returns:
            Dict with tier gradient metrics, or None if not tracking this step
        """
        if not self.should_track(step):
            return None
        
        import time
        start_time = time.time()
        
        # Get valid positions only
        if not valid_mask.any():
            return None
        
        valid_output = output[valid_mask].detach().clone().requires_grad_(True)
        valid_targets = targets[valid_mask].to(valid_output.dtype)
        
        tier_grad_norms = {}
        tier_grad_mean = {}
        tier_loss_contribution = {}
        
        # Compute gradient contribution for each tier
        for tier_name, tier_mask in self.tier_masks.items():
            # Get outputs/targets for this tier only
            tier_output = valid_output[:, tier_mask]
            tier_targets = valid_targets[:, tier_mask]
            
            # Skip if no codes in this tier
            if tier_mask.sum() == 0:
                tier_grad_norms[tier_name] = 0.0
                tier_grad_mean[tier_name] = 0.0
                tier_loss_contribution[tier_name] = 0.0
                continue
            
            # Compute loss for this tier (using the same criterion)
            # Note: pos_weight indexing needs to match tier
            if hasattr(criterion, 'pos_weight') and criterion.pos_weight is not None:
                tier_pos_weight = criterion.pos_weight[tier_mask]
                tier_criterion = nn.BCEWithLogitsLoss(
                    pos_weight=tier_pos_weight,
                    reduction='mean'
                )
            else:
                tier_criterion = nn.BCEWithLogitsLoss(reduction='mean')
            
            tier_loss = tier_criterion(tier_output, tier_targets)
            tier_loss_contribution[tier_name] = tier_loss.item()
            
            # Compute gradient w.r.t. tier output
            if valid_output.grad is not None:
                valid_output.grad.zero_()
            
            tier_loss.backward(retain_graph=True)
            
            if valid_output.grad is not None:
                # Get gradients for this tier's codes
                tier_grad = valid_output.grad[:, tier_mask]
                tier_grad_norms[tier_name] = tier_grad.norm(p=2).item()
                tier_grad_mean[tier_name] = tier_grad.abs().mean().item()
            else:
                tier_grad_norms[tier_name] = 0.0
                tier_grad_mean[tier_name] = 0.0
        
        # Compute total gradient norm
        total_grad_norm = sum(tier_grad_norms.values())
        
        # Normalize to percentages
        tier_grad_pct = {}
        for tier_name, norm in tier_grad_norms.items():
            tier_grad_pct[tier_name] = 100.0 * norm / (total_grad_norm + 1e-10)
        
        # Store metrics
        metrics = TierGradientMetrics(
            step=step,
            tier_grad_norms=tier_grad_norms,
            tier_grad_mean=tier_grad_mean,
            tier_loss_contribution=tier_loss_contribution,
            total_grad_norm=total_grad_norm,
            timestamp=time.time()
        )
        self.history.append(metrics)
        
        # Keep history bounded
        if len(self.history) > 10000:
            self.history = self.history[-5000:]
        
        return {
            'tier_grad_norms': tier_grad_norms,
            'tier_grad_pct': tier_grad_pct,
            'tier_loss_contribution': tier_loss_contribution,
            'total_grad_norm': total_grad_norm
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all tracked steps."""
        if not self.history:
            return {}
        
        summary = {
            'num_tracked_steps': len(self.history),
            'first_step': self.history[0].step,
            'last_step': self.history[-1].step,
        }
        
        # Aggregate per-tier statistics
        for tier_name in self.tier_names:
            norms = [m.tier_grad_norms.get(tier_name, 0) for m in self.history]
            losses = [m.tier_loss_contribution.get(tier_name, 0) for m in self.history]
            
            summary[f'{tier_name}_grad_norm_mean'] = np.mean(norms)
            summary[f'{tier_name}_grad_norm_std'] = np.std(norms)
            summary[f'{tier_name}_loss_mean'] = np.mean(losses)
        
        # Compute gradient contribution percentages
        total_contrib = sum(summary.get(f'{t}_grad_norm_mean', 0) for t in self.tier_names)
        for tier_name in self.tier_names:
            mean_norm = summary.get(f'{tier_name}_grad_norm_mean', 0)
            summary[f'{tier_name}_grad_pct'] = 100.0 * mean_norm / (total_contrib + 1e-10)
        
        return summary
    
    def print_summary(self):
        """Print formatted summary of gradient analysis."""
        stats = self.get_summary_stats()
        if not stats:
            print("No gradient data collected yet.")
            return
        
        print("\n" + "="*80)
        print("📊 GRADIENT CONTRIBUTION ANALYSIS BY CODE TIER")
        print("="*80)
        print(f"Tracked steps: {stats['num_tracked_steps']} "
              f"(steps {stats['first_step']} - {stats['last_step']})")
        
        print("\n" + "-"*70)
        print(f"{'Tier':<15} {'Grad Norm':<12} {'Grad %':<10} {'Loss Contrib':<12}")
        print("-"*70)
        
        for tier_name in self.tier_names:
            grad_mean = stats.get(f'{tier_name}_grad_norm_mean', 0)
            grad_pct = stats.get(f'{tier_name}_grad_pct', 0)
            loss_mean = stats.get(f'{tier_name}_loss_mean', 0)
            
            print(f"{tier_name:<15} {grad_mean:<12.4f} {grad_pct:<10.1f}% {loss_mean:<12.4f}")
        
        print("-"*70)
        
        # Interpretation
        print("\n🔍 INTERPRETATION:")
        ultra_rare_pct = stats.get('ultra_rare_grad_pct', 0)
        very_common_pct = stats.get('very_common_grad_pct', 0)
        
        if ultra_rare_pct < 5:
            print("   ⚠️  Ultra-rare codes contribute <5% of gradient - being IGNORED")
            print("      Consider: Higher weights, focal loss gamma increase, or curriculum learning")
        elif ultra_rare_pct < 15:
            print("   ⚡ Ultra-rare codes contribute 5-15% - UNDERREPRESENTED")
            print("      Consider: Moderate weight increase or gradient accumulation")
        else:
            print("   ✅ Ultra-rare codes contribute >15% - REASONABLE representation")
        
        if very_common_pct > 50:
            print("   ⚠️  Very common codes dominate (>50%) - GRADIENT SATURATION risk")
            print("      Consider: Lower weights on common, or per-tier learning rates")
        
    def export_results(self, filepath: str):
        """Export results to JSON for external analysis."""
        import json
        
        results = {
            'summary': self.get_summary_stats(),
            'tier_config': {k: v for k, v in self.tier_config.items()},
            'history': [
                {
                    'step': m.step,
                    'tier_grad_norms': m.tier_grad_norms,
                    'tier_grad_mean': m.tier_grad_mean,
                    'tier_loss_contribution': m.tier_loss_contribution,
                    'total_grad_norm': m.total_grad_norm
                }
                for m in self.history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Gradient analysis exported to: {filepath}")
```

---

## PART 2: Modify `train_epoch` Signature

**At line ~4825**, modify the function signature to add the new parameter:

```python
# Around line 4825-4847 - ADD gradient_analyzer parameter
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
    log_interval: int = 500,
    global_step: int = 0, 
    loss_tracker: Optional[LossTracker] = None,
    is_main: bool = True,
    use_ddp: bool = False,
    accumulation_steps: int = 1,
    track_gpu_memory: bool = True,
    metrics_logger: Optional['MetricsLogger'] = None,
    logger: Optional[logging.Logger] = None,
    optimize_config: Optional['OptimizeConfig'] = None,
    gradient_analyzer: Optional['GradientTierAnalyzer'] = None  # <-- ADD THIS LINE
) -> Dict[str, float]:
```

---

## PART 3: Add Gradient Computation Inside Training Loop

**After line ~4985** (after `scaled_loss.backward()`), add the gradient analysis computation:

```python
        # ... existing code: scaled_loss.backward() ...
        
        # ============================================================
        # GRADIENT TIER ANALYSIS (after backward, before optimizer step)
        # ============================================================
        if gradient_analyzer is not None and gradient_analyzer.should_track(global_step):
            with torch.no_grad():
                # Get the valid output and targets for gradient analysis
                # We need to reconstruct valid_mask here
                batch_size_local = x.shape[0]
                actual_len_dy = x.shape[1]
                
                grad_valid_mask = torch.zeros(
                    batch_size_local * actual_len_dy, 
                    dtype=torch.bool, 
                    device=device
                )
                for i in range(batch_size_local):
                    valid_days = min(int(dt_cnt[i].item()), actual_len_dy)
                    if valid_days > 0:
                        start_idx = i * actual_len_dy
                        grad_valid_mask[start_idx:start_idx + valid_days] = True
                
                # Get output from result (need to re-extract if not cached)
                if isinstance(result, tuple):
                    grad_output = result[1].get('predictions', None)
                    if grad_output is None:
                        # Need to get output - run inference only
                        with torch.no_grad():
                            if hasattr(model, 'module'):
                                actual_model = model.module
                                if hasattr(actual_model, 'model'):
                                    actual_model = actual_model.model
                            else:
                                actual_model = model
                            grad_output = actual_model(x)
                else:
                    with torch.no_grad():
                        if hasattr(model, 'module'):
                            actual_model = model.module
                            if hasattr(actual_model, 'model'):
                                actual_model = actual_model.model
                        else:
                            actual_model = model
                        grad_output = actual_model(x)
                
                if grad_output is not None:
                    grad_output_flat = grad_output.view(
                        batch_size_local * actual_len_dy, 
                        config.target_cd_cnt
                    )
                    targets_flat = targets_mh.view(
                        batch_size_local * actual_len_dy, 
                        config.target_cd_cnt
                    )
                    
                    grad_metrics = gradient_analyzer.compute_tier_gradients(
                        output=grad_output_flat,
                        targets=targets_flat,
                        valid_mask=grad_valid_mask,
                        criterion=criterion,
                        step=global_step
                    )
                    
                    # Log gradient metrics periodically
                    if grad_metrics is not None and is_main and batch_idx % log_interval == 0:
                        tier_pcts = grad_metrics['tier_grad_pct']
                        print(f"    📊 Grad%: ultra_rare={tier_pcts.get('ultra_rare', 0):.1f}% | "
                              f"rare={tier_pcts.get('rare', 0):.1f}% | "
                              f"common={tier_pcts.get('common', 0):.1f}% | "
                              f"very_common={tier_pcts.get('very_common', 0):.1f}%")
```

Also, **at the end of `train_epoch`** (before the return statement, around line 5200+), add:

```python
    # ============================================================
    # END OF EPOCH: Return gradient analysis summary
    # ============================================================
    epoch_results = {
        'pred_loss': total_pred_loss / nbatch,
        'aux_loss': total_aux_loss / nbatch if total_aux_loss > 0 else 0.0,
        'global_step': global_step,
        # ... existing keys ...
    }
    
    # Add gradient analysis summary if available
    if gradient_analyzer is not None:
        grad_summary = gradient_analyzer.get_summary_stats()
        if grad_summary:
            epoch_results['gradient_analysis'] = grad_summary
    
    return epoch_results
```

---

## PART 4: Initialize Analyzer in `run_single_experiment`

**In `run_single_experiment` (around line ~10950-11000)**, after the criterion is created and before the training loop, add:

```python
    # ============================================================
    # GRADIENT TIER ANALYZER (optional, for diagnosis)
    # ============================================================
    gradient_analyzer = None
    enable_gradient_analysis = True  # Set to True to enable
    
    if enable_gradient_analysis:
        gradient_analyzer = GradientTierAnalyzer(
            code_frequencies=code_frequencies,
            device=device,
            target_cd_cnt=config.target_cd_cnt,
            track_every_n_steps=100  # Track every 100 steps
        )
```

Then, **in the training loop call** (around line ~11100), pass the analyzer:

```python
    # In the training loop call, add gradient_analyzer parameter:
    epoch_results = train_epoch(
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
        log_interval=log_interval,
        global_step=global_step,
        loss_tracker=loss_tracker,
        is_main=is_main,
        use_ddp=use_ddp,
        accumulation_steps=accumulation_steps,
        track_gpu_memory=track_gpu_memory,
        metrics_logger=metrics_logger,
        logger=logger,
        optimize_config=optimize_config,
        gradient_analyzer=gradient_analyzer  # <-- ADD THIS
    )
```

**At the end of training**, export results:

```python
    # After training loop completes, before returning results:
    if gradient_analyzer is not None:
        gradient_analyzer.print_summary()
        
        # Export to log directory
        grad_export_path = os.path.join(log_dir, f"{exp_name}_gradient_analysis.json")
        gradient_analyzer.export_results(grad_export_path)
```

---

## Which Experiment to Run

**Use `exp2b` (Flash + Learned Pool)** for the analysis. It's your best-performing architecture without MoE complexity. Run with:

```python
# In a cell or script:
results = run_single_experiment(
    exp_name='exp2_flash_learned_pool',  # or 'exp2b_flash_learned_pool'
    moe_config=None,
    use_learnt_att_pool=True,
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=1,
    experiment_round='gradient_analysis',
    log_dir='./expe_logs/gradient_analysis/',
    save_model=False  # Don't save model, just analyze
)
```

---

## Expected Output

During training, you'll see:

```
📊 GradientTierAnalyzer initialized:
   Target codes: 6297
   Track every: 100 steps
   ultra_rare  :   315 codes (5.0%)
   tail        :  1259 codes (20.0%)
   rare        :  1574 codes (25.0%)
   medium      :  1574 codes (25.0%)
   common      :   944 codes (15.0%)
   very_common :   631 codes (10.0%)

  Batch 0/1500
    Loss: 0.0823 | R@10: 0.412 | ...
    📊 Grad%: ultra_rare=3.2% | rare=8.1% | common=22.4% | very_common=45.6%
```

At the end:

```
================================================================================
📊 GRADIENT CONTRIBUTION ANALYSIS BY CODE TIER
================================================================================
Tracked steps: 150 (steps 0 - 14900)

----------------------------------------------------------------------
Tier            Grad Norm    Grad %     Loss Contrib
----------------------------------------------------------------------
ultra_rare      0.0234       3.1%       0.2341
tail            0.0567       7.5%       0.1892
rare            0.0891       11.8%      0.1456
medium          0.1234       16.4%      0.0923
common          0.1891       25.1%      0.0654
very_common     0.2712       36.0%      0.0423
----------------------------------------------------------------------

🔍 INTERPRETATION:
   ⚠️  Ultra-rare codes contribute <5% of gradient - being IGNORED
      Consider: Higher weights, focal loss gamma increase, or curriculum learning
   ⚠️  Very common codes dominate (>50%) - GRADIENT SATURATION risk
      Consider: Lower weights on common, or per-tier learning rates
```

---

## Summary of All Changes

| File | Line | Change |
|------|------|--------|
| `moe_flashattn_3.py` | ~4340 | Add `TierGradientMetrics` dataclass and `GradientTierAnalyzer` class |
| `moe_flashattn_3.py` | ~4847 | Add `gradient_analyzer: Optional['GradientTierAnalyzer'] = None` parameter |
| `moe_flashattn_3.py` | ~4990 | Add gradient analysis computation after `backward()` |
| `moe_flashattn_3.py` | ~5200 | Add gradient summary to epoch results |
| `moe_flashattn_3.py` | ~10950 | Initialize `GradientTierAnalyzer` in `run_single_experiment` |
| `moe_flashattn_3.py` | ~11100 | Pass `gradient_analyzer` to `train_epoch` |
| `moe_flashattn_3.py` | ~11280 | Export gradient analysis results |

This gives you a complete diagnostic tool to understand exactly where gradients are flowing and whether the class imbalance mitigation is actually working at the gradient level.