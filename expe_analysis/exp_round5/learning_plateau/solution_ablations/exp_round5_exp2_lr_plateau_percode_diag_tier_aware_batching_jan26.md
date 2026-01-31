# Implementation for Priority 1 & Priority 2

Based on the code structure in `moe_flashattn_4.py` and `exp_round5_exp2_lr_plateau_gradient_result_nextstep_discussion_jan25.md`, here are the implementations designed to:
1. Work with all model types (BaselineTransformer, FlashAttentionTransformer, FlashMoETransformer)
2. Integrate seamlessly with existing code
3. Be easily copied into Jupyter notebooks

---

## Priority 1: Per-Code Logit/Embedding Analysis (Diagnostic)

**Purpose:** Analyze logit distributions and embedding norms by tier to determine if rare codes have collapsed representations or just weak signals.

**Where to add:** New cell in Jupyter notebook (can be run standalone on existing checkpoints)

```python
# ============================================================
# PRIORITY 1: PER-CODE LOGIT/EMBEDDING DIAGNOSTIC ANALYSIS
# ============================================================
# Purpose: Diagnose if rare/tail codes have:
#   - Collapsed embeddings (norms ≈ 0)
#   - Weak but non-zero logits
#   - Oscillating/unstable logits
# 
# This is a ZERO-COST diagnostic that runs on existing checkpoints.
# Run BEFORE implementing any training interventions.
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TierDiagnosticResult:
    """Container for diagnostic results per tier."""
    tier_name: str
    num_codes: int
    # Embedding analysis
    embedding_norm_mean: float
    embedding_norm_std: float
    embedding_norm_min: float
    embedding_norm_max: float
    # Logit analysis (when y=1)
    logit_when_positive_mean: float
    logit_when_positive_std: float
    logit_when_positive_min: float
    logit_when_positive_max: float
    # Logit analysis (when y=0)  
    logit_when_negative_mean: float
    logit_when_negative_std: float
    # Margin analysis
    margin_vs_threshold: float  # Mean (logit - 0.5_threshold)
    positive_rate_above_threshold: float  # % of positives with logit > 0
    
    def __repr__(self):
        return (f"TierDiagnostic({self.tier_name}): "
                f"emb_norm={self.embedding_norm_mean:.4f}±{self.embedding_norm_std:.4f}, "
                f"logit_pos={self.logit_when_positive_mean:.4f}±{self.logit_when_positive_std:.4f}, "
                f"margin={self.margin_vs_threshold:.4f}, "
                f"above_thresh={self.positive_rate_above_threshold:.2%}")


class PerCodeDiagnosticAnalyzer:
    """
    Diagnostic analyzer for per-code logit and embedding analysis.
    
    Usage (in Jupyter notebook):
        # Load model checkpoint
        model = load_trained_model(...)
        
        # Create analyzer
        analyzer = PerCodeDiagnosticAnalyzer(
            code_frequencies=prepared_data.code_frequencies,
            device=device
        )
        
        # Run diagnostic on validation data
        results = analyzer.analyze(
            model=model,
            dataloader=val_loader,
            config=config,
            num_batches=50  # Use subset for speed
        )
        
        # Print diagnosis
        analyzer.print_diagnosis(results)
        
        # Plot distributions
        analyzer.plot_distributions(results)
    """
    
    def __init__(
        self,
        code_frequencies: np.ndarray,
        device: torch.device,
        percentile_boundaries: Tuple[float, float, float] = (20, 50, 80)
    ):
        self.device = device
        self.num_codes = len(code_frequencies)
        self.code_frequencies = code_frequencies
        
        # Build tier indices (same logic as GradientTierAnalyzer)
        freq_nz = code_frequencies[code_frequencies > 0]
        if len(freq_nz) == 0:
            raise ValueError("No non-zero frequencies found")
        
        percentiles = np.percentile(freq_nz, list(percentile_boundaries))
        
        # Create tier masks
        self.tier_indices = {}
        self.tier_masks = {}
        
        # Common: above 80th percentile
        common_mask = code_frequencies > percentiles[2]
        self.tier_indices['common'] = np.where(common_mask)[0]
        self.tier_masks['common'] = common_mask
        
        # Medium: 50th to 80th percentile
        medium_mask = (code_frequencies <= percentiles[2]) & (code_frequencies > percentiles[1])
        self.tier_indices['medium'] = np.where(medium_mask)[0]
        self.tier_masks['medium'] = medium_mask
        
        # Rare: 20th to 50th percentile
        rare_mask = (code_frequencies <= percentiles[1]) & (code_frequencies > percentiles[0])
        self.tier_indices['rare'] = np.where(rare_mask)[0]
        self.tier_masks['rare'] = rare_mask
        
        # Tail: below 20th percentile (but > 0)
        tail_mask = (code_frequencies <= percentiles[0]) & (code_frequencies > 0)
        self.tier_indices['tail'] = np.where(tail_mask)[0]
        self.tier_masks['tail'] = tail_mask
        
        # Zero: never appeared in training
        zero_mask = code_frequencies == 0
        self.tier_indices['zero'] = np.where(zero_mask)[0]
        self.tier_masks['zero'] = zero_mask
        
        print(f"PerCodeDiagnosticAnalyzer initialized:")
        for tier, indices in self.tier_indices.items():
            print(f"  {tier}: {len(indices)} codes")
    
    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DataParallel/DDP to get underlying model."""
        actual_model = model
        if isinstance(model, nn.DataParallel):
            actual_model = model.module
        if hasattr(actual_model, 'model'):
            actual_model = actual_model.model
        return actual_model
    
    def _get_decoder_weights(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Extract decoder_cd weights [num_codes, d_model]."""
        actual_model = self._unwrap_model(model)
        
        if hasattr(actual_model, 'decoder_cd'):
            return actual_model.decoder_cd.weight.detach()
        
        # Search for decoder_cd in case of different naming
        for name, module in actual_model.named_modules():
            if 'decoder_cd' in name and isinstance(module, nn.Linear):
                return module.weight.detach()
        
        return None
    
    def analyze_embeddings(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """
        Analyze decoder weight embeddings per tier.
        
        The decoder_cd.weight has shape [num_codes, d_model].
        Each row is essentially the "embedding" for that code in output space.
        """
        decoder_weights = self._get_decoder_weights(model)
        if decoder_weights is None:
            print("Warning: Could not find decoder_cd weights")
            return {}
        
        # Move to CPU for analysis
        weights_cpu = decoder_weights.cpu().numpy()
        
        # Compute per-code norms
        per_code_norms = np.linalg.norm(weights_cpu, axis=1)
        
        results = {}
        for tier_name, indices in self.tier_indices.items():
            if len(indices) == 0:
                continue
            
            tier_norms = per_code_norms[indices]
            results[tier_name] = {
                'norm_mean': float(np.mean(tier_norms)),
                'norm_std': float(np.std(tier_norms)),
                'norm_min': float(np.min(tier_norms)),
                'norm_max': float(np.max(tier_norms)),
                'norm_median': float(np.median(tier_norms)),
                'num_near_zero': int(np.sum(tier_norms < 0.01)),
                'num_codes': len(indices)
            }
        
        return results
    
    @torch.no_grad()
    def analyze_logits(
        self,
        model: nn.Module,
        dataloader,
        config,
        num_batches: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze logit distributions per tier when y=1 and y=0.
        
        This is the key diagnostic: we want to know if rare/tail codes
        produce low logits even when they SHOULD be positive.
        """
        model.eval()
        actual_model = self._unwrap_model(model)
        
        # Accumulators per tier
        logits_when_positive = defaultdict(list)  # tier -> list of logits
        logits_when_negative = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx}/{num_batches}...")
            
            # Prepare input
            age = batch['age']
            gender = batch['gender']
            lob = batch['lob']
            codes = batch['codes']
            dt_cnt = batch['dt_cnt']
            targets_mh = batch['target_multihot']  # [batch, len_dy, num_codes]
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),
                codes
            ], dim=-1)
            
            x = x.to(self.device)
            dt_cnt = dt_cnt.to(self.device)
            targets_mh = targets_mh.to(self.device)
            
            # Forward pass to get logits
            # Handle different model return types
            with torch.cuda.amp.autocast(enabled=False):
                output = model(x)
            
            # Extract logits from output
            if isinstance(output, tuple):
                logits = output[0]  # First element is typically the main output
                if isinstance(logits, dict):
                    logits = logits.get('predictions', logits.get('logits', None))
            else:
                logits = output
            
            if logits is None:
                # Model might need special handling - try direct forward
                logits = actual_model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
            
            # logits: [batch, len_dy, num_codes]
            # Flatten to [batch * len_dy, num_codes]
            batch_size, len_dy, num_codes = logits.shape
            
            # Create valid day mask based on dt_cnt
            valid_mask = torch.zeros(batch_size, len_dy, device=self.device, dtype=torch.bool)
            for i, cnt in enumerate(dt_cnt):
                valid_mask[i, :cnt] = True
            
            # Flatten
            logits_flat = logits[valid_mask].cpu().numpy()  # [valid_days, num_codes]
            targets_flat = targets_mh[valid_mask].cpu().numpy()  # [valid_days, num_codes]
            
            # Accumulate per tier
            for tier_name, indices in self.tier_indices.items():
                if len(indices) == 0:
                    continue
                
                tier_logits = logits_flat[:, indices]  # [valid_days, tier_codes]
                tier_targets = targets_flat[:, indices]
                
                # Positive examples (y=1)
                pos_mask = tier_targets > 0.5
                if pos_mask.any():
                    logits_when_positive[tier_name].extend(tier_logits[pos_mask].tolist())
                
                # Negative examples (y=0) - sample to avoid memory issues
                neg_mask = tier_targets < 0.5
                if neg_mask.any():
                    neg_logits = tier_logits[neg_mask]
                    # Sample at most 10000 negatives per tier per batch
                    if len(neg_logits) > 10000:
                        neg_logits = neg_logits[np.random.choice(len(neg_logits), 10000, replace=False)]
                    logits_when_negative[tier_name].extend(neg_logits.tolist())
        
        # Compute statistics
        results = {}
        for tier_name in self.tier_indices.keys():
            pos_logits = np.array(logits_when_positive.get(tier_name, []))
            neg_logits = np.array(logits_when_negative.get(tier_name, []))
            
            results[tier_name] = {
                'num_positive_samples': len(pos_logits),
                'num_negative_samples': len(neg_logits),
            }
            
            if len(pos_logits) > 0:
                results[tier_name].update({
                    'logit_pos_mean': float(np.mean(pos_logits)),
                    'logit_pos_std': float(np.std(pos_logits)),
                    'logit_pos_min': float(np.min(pos_logits)),
                    'logit_pos_max': float(np.max(pos_logits)),
                    'logit_pos_median': float(np.median(pos_logits)),
                    'pct_pos_above_zero': float(np.mean(pos_logits > 0)),
                    'pct_pos_above_minus1': float(np.mean(pos_logits > -1)),
                    'logit_pos_25pct': float(np.percentile(pos_logits, 25)),
                    'logit_pos_75pct': float(np.percentile(pos_logits, 75)),
                })
            
            if len(neg_logits) > 0:
                results[tier_name].update({
                    'logit_neg_mean': float(np.mean(neg_logits)),
                    'logit_neg_std': float(np.std(neg_logits)),
                    'logit_neg_median': float(np.median(neg_logits)),
                })
            
            # Compute margin (separation between positive and negative)
            if len(pos_logits) > 0 and len(neg_logits) > 0:
                margin = np.mean(pos_logits) - np.mean(neg_logits)
                results[tier_name]['margin_pos_neg'] = float(margin)
        
        return results
    
    def run_full_diagnostic(
        self,
        model: nn.Module,
        dataloader,
        config,
        num_batches: int = 50
    ) -> Dict[str, Any]:
        """
        Run complete diagnostic analysis.
        
        Returns dict with:
        - embedding_analysis: per-tier embedding norm statistics
        - logit_analysis: per-tier logit statistics
        - diagnosis: interpretation of results
        """
        print("=" * 60)
        print("PRIORITY 1: PER-CODE DIAGNOSTIC ANALYSIS")
        print("=" * 60)
        
        print("\n[1/3] Analyzing decoder embeddings...")
        embedding_results = self.analyze_embeddings(model)
        
        print("\n[2/3] Analyzing logit distributions...")
        logit_results = self.analyze_logits(model, dataloader, config, num_batches)
        
        print("\n[3/3] Generating diagnosis...")
        diagnosis = self._generate_diagnosis(embedding_results, logit_results)
        
        return {
            'embedding_analysis': embedding_results,
            'logit_analysis': logit_results,
            'diagnosis': diagnosis
        }
    
    def _generate_diagnosis(
        self,
        embedding_results: Dict,
        logit_results: Dict
    ) -> Dict[str, Any]:
        """Generate diagnostic interpretation."""
        diagnosis = {
            'embedding_collapse_detected': False,
            'weak_signal_detected': False,
            'ranking_problem_detected': False,
            'recommendations': []
        }
        
        tiers_to_check = ['rare', 'tail']
        
        for tier in tiers_to_check:
            if tier not in embedding_results or tier not in logit_results:
                continue
            
            emb = embedding_results[tier]
            logit = logit_results[tier]
            
            # Check 1: Embedding collapse (norms near zero)
            if emb['norm_mean'] < 0.1 or emb['num_near_zero'] > emb['num_codes'] * 0.1:
                diagnosis['embedding_collapse_detected'] = True
                diagnosis['recommendations'].append(
                    f"{tier.upper()}: Embedding collapse detected (mean norm={emb['norm_mean']:.4f}). "
                    f"Consider embedding regularization."
                )
            
            # Check 2: Weak signal (logits when positive are low)
            if 'logit_pos_mean' in logit:
                if logit['logit_pos_mean'] < -2:
                    diagnosis['weak_signal_detected'] = True
                    diagnosis['recommendations'].append(
                        f"{tier.upper()}: Weak positive signal (mean logit={logit['logit_pos_mean']:.2f}). "
                        f"Rare codes may be under-represented."
                    )
                
                # Check 3: Poor ranking (positive logits close to negative logits)
                if 'margin_pos_neg' in logit and logit['margin_pos_neg'] < 1.0:
                    diagnosis['ranking_problem_detected'] = True
                    diagnosis['recommendations'].append(
                        f"{tier.upper()}: Small margin between pos/neg (margin={logit['margin_pos_neg']:.2f}). "
                        f"Consider sampled softmax or ranking loss."
                    )
        
        # Compare common vs tail
        if 'common' in logit_results and 'tail' in logit_results:
            common_logit = logit_results['common']
            tail_logit = logit_results['tail']
            
            if 'logit_pos_mean' in common_logit and 'logit_pos_mean' in tail_logit:
                gap = common_logit['logit_pos_mean'] - tail_logit['logit_pos_mean']
                if gap > 3:
                    diagnosis['recommendations'].append(
                        f"TIER GAP: Common codes have {gap:.2f} higher mean logits than tail. "
                        f"This suggests gradient starvation - implement tier-aware batching."
                    )
        
        if not diagnosis['recommendations']:
            diagnosis['recommendations'].append(
                "No critical issues detected. Embeddings and logits appear healthy."
            )
        
        return diagnosis
    
    def print_diagnosis(self, results: Dict[str, Any]):
        """Print formatted diagnostic results."""
        print("\n" + "=" * 60)
        print("DIAGNOSTIC RESULTS")
        print("=" * 60)
        
        # Embedding analysis
        print("\n📊 EMBEDDING ANALYSIS (decoder_cd weights)")
        print("-" * 50)
        emb = results['embedding_analysis']
        for tier in ['common', 'medium', 'rare', 'tail']:
            if tier in emb:
                e = emb[tier]
                print(f"  {tier.upper():8s}: norm={e['norm_mean']:.4f}±{e['norm_std']:.4f}, "
                      f"min={e['norm_min']:.4f}, max={e['norm_max']:.4f}, "
                      f"near_zero={e['num_near_zero']}/{e['num_codes']}")
        
        # Logit analysis
        print("\n📊 LOGIT ANALYSIS (when y=1)")
        print("-" * 50)
        logit = results['logit_analysis']
        for tier in ['common', 'medium', 'rare', 'tail']:
            if tier in logit and 'logit_pos_mean' in logit[tier]:
                l = logit[tier]
                print(f"  {tier.upper():8s}: logit={l['logit_pos_mean']:+.2f}±{l['logit_pos_std']:.2f}, "
                      f"n={l['num_positive_samples']}, "
                      f">0: {l['pct_pos_above_zero']:.1%}")
        
        # Margins
        print("\n📊 MARGIN ANALYSIS (positive - negative)")
        print("-" * 50)
        for tier in ['common', 'medium', 'rare', 'tail']:
            if tier in logit and 'margin_pos_neg' in logit[tier]:
                l = logit[tier]
                print(f"  {tier.upper():8s}: margin={l['margin_pos_neg']:+.2f}")
        
        # Diagnosis
        print("\n🔍 DIAGNOSIS")
        print("-" * 50)
        diag = results['diagnosis']
        print(f"  Embedding collapse: {'⚠️ YES' if diag['embedding_collapse_detected'] else '✅ NO'}")
        print(f"  Weak signal:        {'⚠️ YES' if diag['weak_signal_detected'] else '✅ NO'}")
        print(f"  Ranking problem:    {'⚠️ YES' if diag['ranking_problem_detected'] else '✅ NO'}")
        
        print("\n📋 RECOMMENDATIONS")
        print("-" * 50)
        for i, rec in enumerate(diag['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)
    
    def plot_distributions(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Plot embedding norm and logit distributions by tier."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        tiers = ['common', 'medium', 'rare', 'tail']
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        
        # Plot 1: Embedding norms
        ax = axes[0]
        emb = results['embedding_analysis']
        norms = [emb.get(t, {}).get('norm_mean', 0) for t in tiers]
        stds = [emb.get(t, {}).get('norm_std', 0) for t in tiers]
        ax.bar(tiers, norms, yerr=stds, color=colors, alpha=0.7, capsize=5)
        ax.set_ylabel('Embedding Norm')
        ax.set_title('Decoder Weight Norms by Tier')
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')
        ax.legend()
        
        # Plot 2: Logits when positive
        ax = axes[1]
        logit = results['logit_analysis']
        pos_means = [logit.get(t, {}).get('logit_pos_mean', 0) for t in tiers]
        pos_stds = [logit.get(t, {}).get('logit_pos_std', 0) for t in tiers]
        ax.bar(tiers, pos_means, yerr=pos_stds, color=colors, alpha=0.7, capsize=5)
        ax.set_ylabel('Logit Value')
        ax.set_title('Mean Logit When y=1')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Decision boundary')
        ax.legend()
        
        # Plot 3: Margins
        ax = axes[2]
        margins = [logit.get(t, {}).get('margin_pos_neg', 0) for t in tiers]
        ax.bar(tiers, margins, color=colors, alpha=0.7)
        ax.set_ylabel('Margin')
        ax.set_title('Margin (Positive - Negative)')
        ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Healthy margin')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


# ============================================================
# USAGE EXAMPLE (copy to Jupyter cell)
# ============================================================
"""
# Load your model and data
model = load_trained_model(
    model_path='path/to/checkpoint.pt',
    model_class=FlashAttentionTransformer,  # or FlashMoETransformer, BaselineTransformer
    config=config,
    device=device
)

# Create diagnostic analyzer
analyzer = PerCodeDiagnosticAnalyzer(
    code_frequencies=prepared_data.code_frequencies,
    device=device
)

# Run full diagnostic
results = analyzer.run_full_diagnostic(
    model=model,
    dataloader=val_loader,
    config=config,
    num_batches=50  # Adjust based on dataset size
)

# Print formatted results
analyzer.print_diagnosis(results)

# Plot distributions
analyzer.plot_distributions(results, save_path='diagnostic_plot.png')

# Decision tree based on results:
# - If embedding_collapse_detected: Implement embedding regularization
# - If weak_signal_detected: Implement tier-aware batching (Priority 2)
# - If ranking_problem_detected: Consider sampled softmax
"""
```

---

## Priority 2: Tier-Aware Batching

**Purpose:** Guarantee minimum rare/tail samples per batch to prevent gradient starvation.

**Where to add:** 
1. New cell for `TierAwareBatchSampler` class
2. Modify `create_dataloaders` function call to use this sampler

```python
# ============================================================
# PRIORITY 2: TIER-AWARE BATCH SAMPLER
# ============================================================
# Purpose: Guarantee minimum medium/rare/tail positive samples per batch
# to prevent gradient starvation during training.
#
# This directly addresses the root cause: rare codes appear too
# sporadically, causing their gradient signal to be averaged out
# by the steady stream of common code updates.
#
# Success criteria:
#   - train_grad_tier_tail_frac > 5% (>8% for 3.4M model)
#   - tail_top10_acc > 1%
#   - tail logit moves from -14.69 toward -8
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, Dataset, DataLoader
from typing import Dict, List, Iterator, Optional, Tuple, Union, Any
from collections import defaultdict
import random
import logging

# ============================================================
# Import from moe_flashattn_4_core.py when running in notebook
# ============================================================
# Add this at the start of your notebook:
#
# import sys
# sys.path.insert(0, '/path/to/dev/moe')
# from moe_flashattn_4_core import (
#     BaseConfig, FlashAttentionConfig, MoEConfig,
#     ClinicalDataset, create_collate_fn, PreparedData
# )
# ============================================================


class TierAwareBatchSampler(Sampler):
    """
    Batch sampler that guarantees minimum representation of medium/rare/tail codes.
    
    Strategy (member-level sampling with code-tier awareness):
    1. Pre-compute which MEMBERS (samples) contain medium/rare/tail positive codes
    2. Each batch draws MEMBERS from tier-specific pools:
       - `medium_quota` members that have at least one medium-tier code
       - `rare_quota` members that have at least one rare-tier code
       - `tail_quota` members that have at least one tail-tier code
       - Remaining members from general pool
    
    Key insight: The training unit is the member, but we categorize members by the
    frequency tier of the codes they contain. A member with a tail code guarantees
    that tail code will receive gradient signal in that batch.
    
    This ensures consistent gradient signal for rare/tail codes EVERY batch,
    preventing the gradient concentration collapse observed in experiments.
    
    Compatible with:
    - BaselineTransformer (exp1)
    - FlashAttentionTransformer (exp2)
    - FlashMoETransformer (exp6)
    
    Usage:
        sampler = TierAwareBatchSampler(
            dataset=train_dataset,
            code_frequencies=prepared_data.code_frequencies,
            batch_size=64,
            medium_quota=4,  # At least 4 members with medium codes per batch
            rare_quota=8,    # At least 8 members with rare codes per batch
            tail_quota=10,   # At least 10 members with tail codes per batch
            shuffle=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=create_collate_fn(config),
            num_workers=4
        )
    """
    
    def __init__(
        self,
        dataset: Dataset,
        code_frequencies: np.ndarray,
        batch_size: int,
        medium_quota: int = 0,
        rare_quota: int = 4,
        tail_quota: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        percentile_boundaries: Tuple[float, float, float] = (20, 50, 80),
        verbose: bool = True
    ):
        """
        Args:
            dataset: ClinicalDataset with targets
            code_frequencies: Array of code occurrence counts
            batch_size: Total batch size
            medium_quota: Minimum members with medium code positives per batch
            rare_quota: Minimum members with rare code positives per batch
            tail_quota: Minimum members with tail code positives per batch
            shuffle: Whether to shuffle within each pool
            drop_last: Whether to drop the last incomplete batch
            percentile_boundaries: (tail_thresh, rare_thresh, medium_thresh)
                                   E.g., (20, 50, 80) means:
                                   - Tail: freq <= 20th percentile
                                   - Rare: 20th < freq <= 50th percentile
                                   - Medium: 50th < freq <= 80th percentile
                                   - Common: freq > 80th percentile
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
        
        # Validate quotas
        total_quota = medium_quota + rare_quota + tail_quota
        assert total_quota <= batch_size, \
            f"Combined quotas ({total_quota}) exceed batch_size ({batch_size})"
        
        # Build tier code indices
        self._build_tier_indices(code_frequencies, percentile_boundaries)
        
        # Build sample-to-tier mapping (optimized for large datasets)
        self._build_sample_tier_mapping(verbose)
        
        # Calculate number of batches
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
        
        # Common: above 80th percentile
        self.tier_code_indices['common'] = set(
            np.where(code_frequencies > percentiles[2])[0]
        )
        
        # Medium: 50th to 80th percentile
        self.tier_code_indices['medium'] = set(
            np.where((code_frequencies <= percentiles[2]) & 
                     (code_frequencies > percentiles[1]))[0]
        )
        
        # Rare: 20th to 50th percentile
        self.tier_code_indices['rare'] = set(
            np.where((code_frequencies <= percentiles[1]) & 
                     (code_frequencies > percentiles[0]))[0]
        )
        
        # Tail: below 20th percentile (but > 0)
        self.tier_code_indices['tail'] = set(
            np.where((code_frequencies <= percentiles[0]) & 
                     (code_frequencies > 0))[0]
        )
        
        self.tier_thresholds = {
            'tail_upper': percentiles[0],
            'rare_upper': percentiles[1],
            'medium_upper': percentiles[2]
        }
    
    def _build_sample_tier_mapping(self, verbose: bool):
        """
        Pre-compute which members (samples) contain medium/rare/tail positive codes.
        
        This is done ONCE during initialization for efficiency.
        Optimized to avoid repeated dataset access for large datasets.
        """
        # Members that have at least one code from each tier
        self.samples_with_medium = []
        self.samples_with_rare = []
        self.samples_with_tail = []
        # General pool includes ALL samples (overlaps are OK - handled in __iter__)
        self.general_samples = list(range(self.num_samples))
        
        medium_codes = self.tier_code_indices['medium']
        rare_codes = self.tier_code_indices['rare']
        tail_codes = self.tier_code_indices['tail']
        
        if verbose:
            print(f"TierAwareBatchSampler: Building member-tier mapping for {self.num_samples:,} members...")
            print(f"  Tier code counts: medium={len(medium_codes)}, rare={len(rare_codes)}, tail={len(tail_codes)}")
        
        # Access targets directly from dataset for efficiency
        # ClinicalDataset stores targets as self.targets: List[List[List[int]]]
        targets_list = self.dataset.targets
        
        for idx in range(self.num_samples):
            if verbose and idx > 0 and idx % 500000 == 0:
                print(f"    Processed {idx:,}/{self.num_samples:,} members...")
            
            # Get target codes for this member
            # targets is List[List[int]] where each inner list is codes for one day
            target_list = targets_list[idx]
            
            # Flatten all positive codes for this member
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
        
        if verbose:
            print(f"  ✅ Members with medium codes: {len(self.samples_with_medium):,} "
                  f"({len(self.samples_with_medium)/self.num_samples:.1%})")
            print(f"  ✅ Members with rare codes: {len(self.samples_with_rare):,} "
                  f"({len(self.samples_with_rare)/self.num_samples:.1%})")
            print(f"  ✅ Members with tail codes: {len(self.samples_with_tail):,} "
                  f"({len(self.samples_with_tail)/self.num_samples:.1%})")
            
            # Warn if quotas may not be satisfiable
            if self.medium_quota > 0 and len(self.samples_with_medium) < self.medium_quota * 10:
                print(f"  ⚠️ Warning: Few members with medium codes. May need to reduce medium_quota.")
            if len(self.samples_with_rare) < self.rare_quota * 10:
                print(f"  ⚠️ Warning: Few members with rare codes. May need to reduce rare_quota.")
            if len(self.samples_with_tail) < self.tail_quota * 10:
                print(f"  ⚠️ Warning: Few members with tail codes. May need to reduce tail_quota.")
    
    def _calculate_num_batches(self):
        """Calculate number of batches per epoch."""
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with guaranteed tier representation."""
        # Copy and optionally shuffle pools
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
        
        # Track used samples to avoid duplicates within epoch
        used_samples = set()
        medium_idx = 0
        rare_idx = 0
        tail_idx = 0
        general_idx = 0
        
        batches_yielded = 0
        
        while batches_yielded < self.num_batches:
            batch = []
            
            # 1. Add medium quota (if > 0)
            if self.medium_quota > 0:
                medium_added = 0
                while medium_added < self.medium_quota and medium_idx < len(medium_pool):
                    sample_idx = medium_pool[medium_idx]
                    medium_idx += 1
                    if sample_idx not in used_samples:
                        batch.append(sample_idx)
                        used_samples.add(sample_idx)
                        medium_added += 1
            
            # 2. Add rare quota
            rare_added = 0
            while rare_added < self.rare_quota and rare_idx < len(rare_pool):
                sample_idx = rare_pool[rare_idx]
                rare_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    rare_added += 1
            
            # 3. Add tail quota
            tail_added = 0
            while tail_added < self.tail_quota and tail_idx < len(tail_pool):
                sample_idx = tail_pool[tail_idx]
                tail_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    tail_added += 1
            
            # 4. Fill remainder from general pool
            remaining = self.batch_size - len(batch)
            while remaining > 0 and general_idx < len(general_pool):
                sample_idx = general_pool[general_idx]
                general_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    remaining -= 1
            
            # Handle pool exhaustion - reset with reshuffling
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
                # Reset used_samples when general pool exhausted (epoch boundary)
                used_samples.clear()
            
            # Yield batch if it meets size requirements
            if len(batch) >= self.batch_size or (not self.drop_last and len(batch) > 0):
                if self.shuffle:
                    random.shuffle(batch)  # Shuffle within batch to avoid ordering bias
                yield batch[:self.batch_size]
                batches_yielded += 1
    
    def __len__(self) -> int:
        return self.num_batches


# ============================================================
# HELPER FUNCTION: Create Tier-Aware DataLoader
# ============================================================

def create_tier_aware_dataloader(
    dataset: Dataset,
    code_frequencies: np.ndarray,
    config,  # BaseConfig or subclass
    medium_quota: int = 0,
    rare_quota: int = 4,
    tail_quota: int = 4,
    num_workers: Optional[int] = None,
    collate_fn = None,
    verbose: bool = True
) -> DataLoader:
    """
    Factory function to create a DataLoader with tier-aware batching.
    
    Drop-in replacement for standard DataLoader creation.
    
    Args:
        dataset: ClinicalDataset
        code_frequencies: From prepared_data.code_frequencies
        config: Model config with batch_size
        medium_quota: Min members with medium codes per batch (default 0)
        rare_quota: Min members with rare codes per batch
        tail_quota: Min members with tail codes per batch
        num_workers: Number of data loading workers (default: auto-detect)
        collate_fn: Custom collate function (create_collate_fn(config))
        verbose: Print initialization statistics
    
    Returns:
        DataLoader with tier-aware batching
    
    Usage:
        train_loader = create_tier_aware_dataloader(
            dataset=prepared_data.train_dataset,
            code_frequencies=prepared_data.code_frequencies,
            config=config,
            medium_quota=4,  # Include medium codes if desired
            rare_quota=8,    # For 3.4M model: aggressive rare quota
            tail_quota=10,   # For 3.4M model: aggressive tail quota
            collate_fn=create_collate_fn(config)
        )
    """
    # Import create_collate_fn if needed
    if collate_fn is None:
        try:
            from moe_flashattn_4_core import create_collate_fn
            collate_fn = create_collate_fn(config)
        except ImportError:
            raise ValueError("collate_fn must be provided if moe_flashattn_4_core is not available")
    
    # Auto-detect workers matching existing _create_dataloaders pattern
    if num_workers is None:
        num_workers = min(4, os.cpu_count() // 4) if os.cpu_count() else 2
    
    sampler = TierAwareBatchSampler(
        dataset=dataset,
        code_frequencies=code_frequencies,
        batch_size=config.batch_size,
        medium_quota=medium_quota,
        rare_quota=rare_quota,
        tail_quota=tail_quota,
        shuffle=True,
        drop_last=True,
        verbose=verbose
    )
    
    # Match existing DataLoader configuration from _create_dataloaders
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=False  # Match moe_flashattn_4.py pattern
    )
    
    if verbose:
        print(f"\n✅ Created tier-aware DataLoader:")
        print(f"   Batch size: {config.batch_size}")
        if medium_quota > 0:
            print(f"   Medium quota: {medium_quota} members/batch")
        print(f"   Rare quota: {rare_quota} members/batch")
        print(f"   Tail quota: {tail_quota} members/batch")
        print(f"   Total batches: {len(sampler):,}")
        print(f"   Workers: {num_workers}")
    
    return loader


# ============================================================
# INTEGRATION: Modified _create_dataloaders function
# ============================================================
# This function extends the existing _create_dataloaders from moe_flashattn_4.py
# to support tier-aware batching as an additional option.

def _create_dataloaders_with_tier_aware(
    train_data: Union[pd.DataFrame, Dataset],
    val_data: Union[pd.DataFrame, Dataset],
    config,  # BaseConfig or subclass
    code_frequencies: np.ndarray,
    use_tier_aware: bool = True,
    medium_quota: int = 0,
    rare_quota: int = 4,
    tail_quota: int = 4,
    use_bucketing: bool = False,  # Mutually exclusive with tier_aware
    train_data_df: Optional[pd.DataFrame] = None,
    world_size: int = 1,
    logger: Optional[logging.Logger] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with optional tier-aware batching.
    
    This extends the existing _create_dataloaders function from moe_flashattn_4.py
    to support tier-aware batching as an additional option.
    
    Args:
        train_data: Training dataset (ClinicalDataset or DataFrame)
        val_data: Validation dataset (ClinicalDataset or DataFrame)
        config: Model configuration with batch_size
        code_frequencies: Code frequency array for tier computation
        use_tier_aware: Whether to use tier-aware batching
        medium_quota: Min medium members per batch (if tier_aware)
        rare_quota: Min rare members per batch (if tier_aware)
        tail_quota: Min tail members per batch (if tier_aware)
        use_bucketing: Whether to use bucketing (mutually exclusive with tier_aware)
        train_data_df: Original DataFrame (required for bucketing if train_data is Dataset)
        world_size: Number of distributed processes (unused, for future DDP support)
        logger: Optional logger
    
    Returns:
        (train_loader, val_loader)
    """
    # Import dependencies - handle both standalone and integrated usage
    try:
        from moe_flashattn_4_core import ClinicalDataset, create_collate_fn
    except ImportError:
        # Fallback for integrated usage
        pass
    
    # Handle both Dataset and DataFrame inputs for backward compatibility
    def is_dataset(obj):
        """Check if object is a Dataset (duck-typing for Jupyter compatibility)."""
        return hasattr(obj, '__getitem__') and hasattr(obj, '__len__') and not isinstance(obj, pd.DataFrame)
    
    if is_dataset(train_data):
        train_dataset = train_data
    else:
        train_dataset = ClinicalDataset(train_data, config)
        train_data_df = train_data  # Save for bucketing
    
    if is_dataset(val_data):
        val_dataset = val_data
    else:
        val_dataset = ClinicalDataset(val_data, config)
    
    n_workers = min(4, os.cpu_count() // 4) if os.cpu_count() else 2
    collate_fn = create_collate_fn(config)
    
    # ========================================
    # TRAINING LOADER
    # ========================================
    if use_tier_aware and not use_bucketing:
        if logger:
            logger.info(f"Using TIER-AWARE batching (medium={medium_quota}, rare={rare_quota}, tail={tail_quota})")
        else:
            print(f"Using TIER-AWARE batching (medium={medium_quota}, rare={rare_quota}, tail={tail_quota})")
        
        train_loader = create_tier_aware_dataloader(
            dataset=train_dataset,
            code_frequencies=code_frequencies,
            config=config,
            medium_quota=medium_quota,
            rare_quota=rare_quota,
            tail_quota=tail_quota,
            num_workers=n_workers,
            collate_fn=collate_fn,
            verbose=True
        )
    
    elif use_bucketing and not use_tier_aware:
        if logger:
            logger.info("Using BUCKETING batch sampler")
        
        if train_data_df is None:
            raise ValueError("train_data_df is required when use_bucketing=True and train_data is a Dataset")
        
        # Import BucketingBatchSampler from the main module
        try:
            from moe_flashattn_4 import BucketingBatchSampler
        except ImportError:
            raise ImportError("BucketingBatchSampler not available. Use tier-aware batching or provide train_data_df.")
        
        train_batch_sampler = BucketingBatchSampler(
            data=train_data_df,
            batch_size=config.batch_size,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    else:
        if logger:
            logger.info("Using STANDARD DataLoader (no special batching)")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    # ========================================
    # VALIDATION LOADER (always standard)
    # ========================================
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if logger:
        logger.info(f"Train loader: {len(train_loader):,} batches")
        logger.info(f"Val loader: {len(val_loader):,} batches")
        logger.info(f"Using DataLoader with {n_workers} workers.")
    else:
        print(f"Train loader: {len(train_loader):,} batches")
        print(f"Val loader: {len(val_loader):,} batches")
    
    return train_loader, val_loader


# ============================================================
# USAGE EXAMPLE (copy to Jupyter cell)
# ============================================================
"""
# ========================================
# Setup: Import from core module
# ========================================
import sys
sys.path.insert(0, '/path/to/dev/moe')
from moe_flashattn_4_core import (
    BaseConfig, FlashAttentionConfig, MoEConfig,
    ClinicalDataset, create_collate_fn, PreparedData
)

# ========================================
# Option 1: Use the factory function directly
# ========================================
train_loader = create_tier_aware_dataloader(
    dataset=prepared_data.train_dataset,
    code_frequencies=prepared_data.code_frequencies,
    config=config,
    medium_quota=4,  # Optional: also boost medium codes
    rare_quota=8,    # For 3.4M model: aggressive rare quota
    tail_quota=10,   # For 3.4M model: aggressive tail quota
    collate_fn=create_collate_fn(config)
)

# ========================================
# Option 2: Use the unified create_dataloaders function
# ========================================
train_loader, val_loader = _create_dataloaders_with_tier_aware(
    train_data=prepared_data.train_dataset,
    val_data=prepared_data.val_dataset,
    config=config,
    code_frequencies=prepared_data.code_frequencies,
    use_tier_aware=True,
    medium_quota=4,  # Include medium codes
    rare_quota=8,
    tail_quota=10
)

# ========================================
# Use in training loop
# ========================================
for epoch in range(num_epochs):
    train_metrics = train_epoch(
        model=model,
        dataloader=train_loader,  # <-- Uses tier-aware batching
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config,
        device=device,
        gradient_tier_analyzer=gradient_tier_analyzer  # Monitor tier fractions
    )
    
    # Check if tier-aware batching is working
    # train_grad_tier_tail_frac should be > 5% (target: >8% for 3.4M model)
    print(f"Tail gradient fraction: {train_metrics.get('train_grad_tier_tail_frac', 0):.2%}")
"""


# ============================================================
# VERIFICATION: Test tier-aware batching is working
# ============================================================

def verify_tier_aware_batching(
    dataloader: DataLoader,
    code_frequencies: np.ndarray,
    num_batches: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that tier-aware batching is producing balanced batches.
    
    Run this after creating the DataLoader to confirm it's working.
    
    Args:
        dataloader: The tier-aware DataLoader to verify
        code_frequencies: Code frequency array used to build tiers
        num_batches: Number of batches to sample for verification
        verbose: Print detailed statistics
    
    Returns:
        Dictionary with tier representation statistics per batch
    """
    # Build tier code sets
    freq_nz = code_frequencies[code_frequencies > 0]
    percentiles = np.percentile(freq_nz, [20, 50, 80])
    
    tier_codes = {
        'common': set(np.where(code_frequencies > percentiles[2])[0]),
        'medium': set(np.where((code_frequencies <= percentiles[2]) & 
                                (code_frequencies > percentiles[1]))[0]),
        'rare': set(np.where((code_frequencies <= percentiles[1]) & 
                              (code_frequencies > percentiles[0]))[0]),
        'tail': set(np.where((code_frequencies <= percentiles[0]) & 
                              (code_frequencies > 0))[0])
    }
    
    batch_tier_counts = defaultdict(list)
    batch_member_counts = defaultdict(list)  # Track members with tier codes
    
    if verbose:
        print(f"Verifying tier-aware batching over {num_batches} batches...")
        print(f"Tier code counts: common={len(tier_codes['common'])}, "
              f"medium={len(tier_codes['medium'])}, rare={len(tier_codes['rare'])}, "
              f"tail={len(tier_codes['tail'])}")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        targets_mh = batch['target_multihot']  # [batch, len_dy, num_codes]
        batch_size = targets_mh.shape[0]
        
        # Count unique codes present per tier (across whole batch)
        code_presence = (targets_mh.sum(dim=(0, 1)) > 0).numpy()
        present_codes = set(np.where(code_presence)[0])
        
        for tier_name, tier_code_set in tier_codes.items():
            tier_present = len(present_codes & tier_code_set)
            batch_tier_counts[tier_name].append(tier_present)
        
        # Count members with at least one code from each tier
        for tier_name, tier_code_set in tier_codes.items():
            tier_indices = list(tier_code_set)
            if len(tier_indices) > 0:
                # Check which members have any code from this tier
                # targets_mh: [batch, len_dy, num_codes]
                tier_mask = torch.zeros(targets_mh.shape[-1], dtype=torch.bool)
                tier_mask[tier_indices] = True
                member_has_tier = (targets_mh[:, :, tier_mask].sum(dim=(1, 2)) > 0).sum().item()
                batch_member_counts[tier_name].append(member_has_tier)
            else:
                batch_member_counts[tier_name].append(0)
    
    # Compute statistics
    results = {'tier_codes': {}, 'tier_members': {}}
    
    if verbose:
        print("\n" + "="*60)
        print("TIER REPRESENTATION PER BATCH")
        print("="*60)
        print("\nUnique codes from each tier per batch:")
        print("-" * 50)
    
    for tier_name in ['common', 'medium', 'rare', 'tail']:
        counts = batch_tier_counts[tier_name]
        if len(counts) > 0:
            mean_count = np.mean(counts)
            min_count = np.min(counts)
            max_count = np.max(counts)
        else:
            mean_count = min_count = max_count = 0
            
        results['tier_codes'][tier_name] = {
            'mean': mean_count,
            'min': min_count,
            'max': max_count,
            'all_counts': counts
        }
        if verbose:
            print(f"  {tier_name.upper():8s}: mean={mean_count:.1f} codes, "
                  f"range=[{min_count}, {max_count}]")
    
    if verbose:
        print("\nMembers with codes from each tier per batch:")
        print("-" * 50)
    
    for tier_name in ['common', 'medium', 'rare', 'tail']:
        counts = batch_member_counts[tier_name]
        if len(counts) > 0:
            mean_count = np.mean(counts)
            min_count = np.min(counts)
            max_count = np.max(counts)
        else:
            mean_count = min_count = max_count = 0
            
        results['tier_members'][tier_name] = {
            'mean': mean_count,
            'min': min_count,
            'max': max_count,
            'all_counts': counts
        }
        if verbose:
            print(f"  {tier_name.upper():8s}: mean={mean_count:.1f} members, "
                  f"range=[{min_count}, {max_count}]")
    
    # Check if tail/rare/medium are consistently present
    tail_present_rate = np.mean([c > 0 for c in batch_tier_counts['tail']]) if batch_tier_counts['tail'] else 0
    rare_present_rate = np.mean([c > 0 for c in batch_tier_counts['rare']]) if batch_tier_counts['rare'] else 0
    medium_present_rate = np.mean([c > 0 for c in batch_tier_counts['medium']]) if batch_tier_counts['medium'] else 0
    
    if verbose:
        print("\n" + "-" * 50)
        print(f"✅ Medium codes present in {medium_present_rate:.1%} of batches")
        print(f"✅ Rare codes present in {rare_present_rate:.1%} of batches")
        print(f"✅ Tail codes present in {tail_present_rate:.1%} of batches")
        
        if tail_present_rate < 0.9 or rare_present_rate < 0.9:
            print("\n⚠️ Warning: Tier-aware batching may not be achieving desired coverage!")
            print("   Consider increasing rare_quota/tail_quota or checking sample availability.")
        else:
            print("\n✅ Tier-aware batching verified!")
    
    results['presence_rates'] = {
        'medium': medium_present_rate,
        'rare': rare_present_rate,
        'tail': tail_present_rate
    }
    
    return results
```

---

## Summary: How to Use These Implementations

### Step 1: Run Priority 1 Diagnostic (Before any intervention)

Copy the `PerCodeDiagnosticAnalyzer` class to a new Jupyter cell and run:

```python
# Create analyzer
analyzer = PerCodeDiagnosticAnalyzer(
    code_frequencies=prepared_data.code_frequencies,
    device=device
)

# Run diagnostic on your trained model
results = analyzer.run_full_diagnostic(
    model=model,
    dataloader=val_loader,
    config=config,
    num_batches=50
)

# View results
analyzer.print_diagnosis(results)
analyzer.plot_distributions(results)
```

**Decision Tree:**
- If `embedding_collapse_detected` → Consider embedding regularization  
- If `weak_signal_detected` → Implement tier-aware batching (Priority 2)
- If `ranking_problem_detected` → Consider sampled softmax

### Step 2: Implement Priority 2 Tier-Aware Batching

Copy the `TierAwareBatchSampler` and helper functions, then:

```python
# Setup imports
import sys
sys.path.insert(0, '/path/to/dev/moe')
from moe_flashattn_4_core import create_collate_fn

# Create tier-aware data loader
# Note: The sampler works at MEMBER level - it ensures each batch contains
# members that have at least one code from the specified tier.
train_loader = create_tier_aware_dataloader(
    dataset=prepared_data.train_dataset,
    code_frequencies=prepared_data.code_frequencies,
    config=config,
    medium_quota=4,  # Optional: also boost medium tier exposure
    rare_quota=8,    # More aggressive for 3.4M model
    tail_quota=10,   # More aggressive for 3.4M model
    collate_fn=create_collate_fn(config)
)

# Verify it's working
verify_tier_aware_batching(train_loader, prepared_data.code_frequencies)

# Use in training
# (No changes needed to train_epoch - it will automatically benefit)
```

**Recommended quota values (based on batch_size and model scale):**

| Batch Size | Standard Model | 3.4M Model (More Aggressive) |
|------------|----------------|------------------------------|
| 32 | medium=0, rare=4, tail=4 | medium=2, rare=5, tail=6 |
| 64 | medium=0, rare=6, tail=6 | medium=4, rare=8, tail=10 |
| 128 | medium=0, rare=10, tail=10 | medium=6, rare=12, tail=16 |

**Rationale for 3.4M model aggressive quotas:**
- Tail logit degraded from -12.9 to -14.7 (worse with more data)
- Tail margin shrunk from 2.22 to 1.76 (worse discrimination)
- Need to counteract the increased gradient imbalance from more common code samples

---

# Result for priority 1: Detailed Interpretation of Diagnostic Results - 1.7M members
- Jan 28, 2026
- Use model `# MODEL_PATH='logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp2b_flash_learned_pool_v3/saved_models/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20260124_131727_final.pt'
` 
## Part 1: What Each Analysis Does

### 1.1 `analyze_embeddings` - Decoder Weight Analysis

**What it analyzes:**
The `decoder_cd` is the final linear layer in your transformer that maps from the hidden representation to output logits:

```
logits = hidden_state @ decoder_cd.weight.T + decoder_cd.bias
```

Where `decoder_cd.weight` has shape `[num_codes, embedding_size]` = `[6297, 256]`.

Each row of this weight matrix is essentially the **"output embedding"** for that code. When computing whether code `i` should be predicted, the model computes:
```
logit_i = dot_product(hidden_state, decoder_weight[i])
```

**Rationale:**
If a code's decoder weight has a **very small norm (near zero)**, this means:
1. The weight vector is collapsed to the origin
2. The dot product with ANY hidden state will be ~0
3. The model has effectively "given up" on learning that code
4. This is called **"embedding collapse"** - a known pathology in imbalanced learning

**What we measure:**
- `norm_mean`: Average L2 norm of decoder weights for codes in each tier
- `num_near_zero`: Count of codes with norm < 0.01 (collapsed)

**Expected healthy state:** All tiers should have similar, non-trivial norms (~1.0-2.0)

---

### 1.2 `analyze_logits` - Output Distribution Analysis

**What it analyzes:**
This runs the model on validation data and collects the **actual logit values** the model produces, separated by:
- **When y=1 (positive):** What logit does the model output when this code IS actually present?
- **When y=0 (negative):** What logit does the model output when this code is NOT present?

**Rationale:**
For `BCEWithLogitsLoss`:
```
probability = sigmoid(logit) = 1 / (1 + exp(-logit))
```

| Logit Value | Probability | Interpretation |
|-------------|-------------|----------------|
| 0 | 0.50 | Decision boundary |
| +2 | 0.88 | Confident positive |
| -2 | 0.12 | Confident negative |
| -5 | 0.007 | Very confident negative |
| -10 | 0.00005 | Extremely confident negative |

**Critical insight:** If `logit_pos_mean` (when y=1) is very negative, the model is saying:
> "Even when this code IS present, I think it's NOT present"

This is the **"weak signal"** or **"under-confidence"** problem.

**What we measure:**
- `logit_pos_mean`: Average logit when y=1 (should be > 0 for good prediction)
- `pct_pos_above_zero`: % of positive samples where logit > 0 (model would predict correctly)
- `margin_pos_neg`: Separation between positive and negative logit distributions

---

## Part 2: Your Results Interpretation

### 2.1 Embedding Analysis: ✅ NO COLLAPSE DETECTED

```python
'embedding_analysis': {
  'common': {'norm_mean': 1.14, 'num_near_zero': 0, 'num_codes': 1140},
  'medium': {'norm_mean': 1.11, 'num_near_zero': 0, 'num_codes': 1709},
  'rare':   {'norm_mean': 1.13, 'num_near_zero': 0, 'num_codes': 1703},
  'tail':   {'norm_mean': 1.15, 'num_near_zero': 0, 'num_codes': 1147},
  'zero':   {'norm_mean': 1.15, 'num_near_zero': 0, 'num_codes': 598}
}
```

**Key Findings:**
| Tier | Norm Mean | Std | Near Zero |
|------|-----------|-----|-----------|
| Common | 1.14 | 0.18 | 0 |
| Medium | 1.11 | 0.06 | 0 |
| Rare | 1.13 | 0.03 | 0 |
| Tail | 1.15 | 0.03 | 0 |

**Interpretation:**
1. **All tiers have similar norm means (~1.1-1.15)** - No systematic difference
2. **Zero codes near zero** - No embedding collapse detected
3. **Lower variance for rare/tail** - Interesting: these weights are MORE uniform than common codes

**This is GOOD news:** The decoder weights themselves are healthy. The model has NOT collapsed the rare/tail code representations to zero. This rules out the "dead neuron" hypothesis (Scenario A from the expert discussion).

**However, this is also SURPRISING** given the gradient starvation we observed during training (85% gradient to common codes). The explanation:
- The weights were initialized with similar norms
- Even with reduced gradient updates, the weights didn't collapse to zero
- The problem is NOT the weight magnitudes, but what the model learned

---

### 2.2 Logit Analysis: ⚠️ SEVERE UNDER-CONFIDENCE DETECTED

This is where the real problem becomes clear:

```python
'logit_analysis': {
  'common': {'logit_pos_mean': -2.4, 'pct_pos_above_zero': 18.8%, 'margin': 6.0},
  'medium': {'logit_pos_mean': -7.1, 'pct_pos_above_zero': 2.2%,  'margin': 4.8},
  'rare':   {'logit_pos_mean': -11.4, 'pct_pos_above_zero': 0.0%,  'margin': 2.9},
  'tail':   {'logit_pos_mean': -12.9, 'pct_pos_above_zero': 0.0%,  'margin': 2.2}
}
```

**Visualization of the Problem:**

```
Logit Scale (Decision boundary = 0)
─────────────────────────────────────────────────────────────────────────
                                               0
                                               │
                                               │  ← Decision boundary
                                               │
Tail positive   ██ (-12.9)                     │
Rare positive    ███ (-11.4)                   │
Medium positive      █████ (-7.1)              │
Common positive          ██████████ (-2.4)     │  Common neg (-8.4)
                                               │
```

**Key Findings by Tier:**

| Tier | Positive Samples | Logit (y=1) | % Above 0 | Margin | Probability |
|------|------------------|-------------|-----------|--------|-------------|
| **Common** | 656,382 | -2.4 | 18.8% | 6.0 | ~8% |
| **Medium** | 13,130 | -7.1 | 2.2% | 4.8 | ~0.08% |
| **Rare** | 541 | -11.4 | 0.0% | 2.9 | ~0.001% |
| **Tail** | 27 | -12.9 | 0.0% | 2.2 | ~0.0002% |

---

### 2.3 Detailed Interpretation

#### **Finding 1: Severe Logit Suppression Increases with Rarity**

There's a clear **monotonic relationship** between code frequency and logit magnitude:
- Common: logit = -2.4 (probability ~8%)
- Tail: logit = -12.9 (probability ~0.0002%)

**This means:** When a tail code IS actually present (y=1), the model outputs a logit of -12.9, which corresponds to **predicting 99.9998% probability that it's NOT present**.

The model has learned to be **extremely conservative** about rare codes.

#### **Finding 2: Zero Recall for Rare/Tail Codes**

```python
'rare':   {'pct_pos_above_zero': 0.0%}  # Model NEVER predicts rare codes
'tail':   {'pct_pos_above_zero': 0.0%}  # Model NEVER predicts tail codes
```

This is the **concrete manifestation of gradient starvation**: the model has learned that predicting "negative" for rare codes is always safe, so it never predicts them positive.

This explains why `rare_top10_acc = 0` and `tail_top10_acc = 0` in your training metrics.

#### **Finding 3: The Model CAN Distinguish, But Won't Predict**

The margin analysis reveals something important:

| Tier | logit (y=1) | logit (y=0) | Margin |
|------|-------------|-------------|--------|
| Common | -2.4 | -8.4 | 6.0 |
| Rare | -11.4 | -14.3 | 2.9 |
| Tail | -12.9 | -15.1 | 2.2 |

**Interpretation:**
- **Margins are positive** for all tiers (even tail has margin = 2.2)
- This means the model DOES produce higher logits when a code is present vs absent
- The model has learned SOME discrimination ability

**But the absolute values are the problem:**
- For tail codes: logit=-12.9 when present, logit=-15.1 when absent
- Both are FAR below the decision boundary (0)
- So even though the model "knows" a tail code is more likely when present, it still predicts "negative" because -12.9 < 0

This corresponds to **Scenario B from the expert discussion:**
> "Logits are negative but moving (Under-confident) → Needs more signal"

#### **Finding 4: The Zero-Code Anomaly**

```python
'zero': {
  'num_positive_samples': 66728,  # Unexpectedly high!
  'logit_pos_mean': +6.4,         # Highly positive!
  'margin': 21.6                  # Enormous margin!
}
```

**This is a data anomaly that needs investigation:**
- "Zero" codes are codes with `frequency = 0` in training data
- But they have 66,728 positive samples in validation
- And the model predicts them with logit +6.4 (probability ~99.8%)

**Possible explanations:**
1. **Data leakage:** These codes appear in validation but were incorrectly marked as frequency=0
2. **Code mapping issues:** Different code vocabularies between train/val
3. **Temporal drift:** New codes appeared in validation period

**Recommendation:** Investigate this before proceeding. This anomaly could indicate a data quality issue.

---

## Part 3: Connecting to the Theoretical Framework

From the expert discussion document:

### The Gradient Starvation Hypothesis is CONFIRMED

The experts hypothesized:
> "Even if a tail positive produces a large per-example gradient, tail positives appear too sporadically; their directions have high variance and get averaged out by the steady stream of head-code updates."

Your logit analysis confirms this mechanism:
1. The model receives consistent signal for common codes → learns moderate logits (-2.4)
2. The model receives sporadic signal for rare/tail → learns extremely negative logits (-12)
3. The "safe default" for any unknown code is to predict negative

### Why Embeddings Didn't Collapse Despite Gradient Starvation

The experts noted:
> "The question is whether rare code embeddings are 'collapsed' (dead) or 'weak but non-zero'"

Your results show they are **weak but non-zero**:
- Decoder weight norms are healthy (all ~1.1)
- But the LEARNED RELATIONSHIP between hidden states and these weights produces very negative logits

The weights exist, but the model hasn't learned to **activate** them appropriately.

### The Margin Analysis Supports Tier-Aware Batching

The fact that margins are positive (even for tail: 2.2) suggests:
1. The model HAS learned some representation of rare codes
2. It CAN distinguish when they should be present
3. But the overall calibration is wrong (everything shifted negative)

**This is good news for intervention:** You're not starting from scratch. The model has partial knowledge that can be amplified.

---

## Part 4: Actionable Conclusions

### Diagnosis Summary

| Check | Result | Implication |
|-------|--------|-------------|
| Embedding Collapse | ❌ Not detected | Decoder weights are healthy |
| Weak Signal | ✅ **Detected** | Rare/tail logits far below decision boundary |
| Ranking Problem | ⚠️ Partial | Margins exist but are small for rare/tail |

### Recommended Next Step: Priority 2 - Tier-Aware Batching

Based on this diagnosis, **tier-aware batching is the correct intervention** because:

1. **The problem is exposure, not capacity:** Decoder weights exist and aren't collapsed, but the model never learned when to activate them

2. **The model shows it CAN distinguish:** Positive margins indicate some learning occurred. More consistent exposure during training should amplify this

3. **The calibration shift is systematic:** All rare/tail logits are shifted ~10 units negative. Consistent gradient signal should shift this distribution toward 0

### Expected Outcome of Tier-Aware Batching

If successful, you should see:
- `logit_pos_mean` for rare/tail increase from -12 toward -2 or higher
- `pct_pos_above_zero` increase from 0% to some positive percentage
- `train_grad_tier_tail_frac` stay above 5% throughout training (instead of collapsing to 0.1%)

### Additional Investigation Needed

1. **Zero-code anomaly:** Investigate why codes with training frequency=0 have validation positives
2. **Baseline comparison:** Run same diagnostic on pos_weight=50 model to confirm pattern is consistent

## Part 5: Investigate the Zero-code Anomaly

### Understanding How Tiers Are Generated

#### Step 1: Code Frequency Computation

The `code_frequencies` array is computed **from your training data**. It has shape `[target_cd_cnt]` = `[6297]`, where each element counts how many times that code appeared as a positive target in training:

```python
# Conceptually, code_frequencies is computed like this:
code_frequencies = np.zeros(target_cd_cnt)  # [6297]

for sample in training_data:
    for day in sample.targets:
        for code_idx in day:
            code_frequencies[code_idx] += 1
```

#### Step 2: Tier Assignment

The diagnostic uses **percentiles of non-zero frequencies** to define tier boundaries:

```python
# Only look at codes that appeared at least once
freq_nz = code_frequencies[code_frequencies > 0]  # Exclude zeros
percentiles = np.percentile(freq_nz, [20, 50, 80])  
# Example: [50, 200, 1000]
```

Then each code is assigned to a tier:

| Tier | Condition | Meaning |
|------|-----------|---------|
| **Common** | freq > 80th percentile | Top 20% most frequent codes |
| **Medium** | 50th < freq ≤ 80th percentile | Next 30% |
| **Rare** | 20th < freq ≤ 50th percentile | Next 30% |
| **Tail** | 0 < freq ≤ 20th percentile | Bottom 20% of non-zero codes |
| **Zero** | freq == 0 | **Never appeared in training** |

#### Your Tier Distribution

From your results:
| Tier | # Codes | Meaning |
|------|---------|---------|
| Common | 1,140 | Appeared frequently in training |
| Medium | 1,709 | Appeared moderately in training |
| Rare | 1,703 | Appeared infrequently in training |
| Tail | 1,147 | Appeared very rarely in training |
| **Zero** | **598** | **Training frequency = 0** |

Total: 6,297 codes (matches your `target_cd_cnt`)

---

### The Zero-Code Anomaly Explained

#### What The Results Show

```python
'zero': {
  'num_positive_samples': 66728,   # These codes appear as y=1 in validation
  'logit_pos_mean': +6.4,          # Model predicts HIGH probability
  'pct_pos_above_zero': 99.99%,    # Model almost always predicts positive
  'margin': 21.6                    # Huge separation
}
```

#### What This Means

**There are 598 codes that:**
1. Have `code_frequencies[code_idx] == 0` (never appeared in training)
2. BUT appear 66,728 times as positive targets in validation data
3. AND the model predicts them with very HIGH confidence (logit = +6.4)

#### This Is Impossible Under Normal Circumstances

If a code truly never appeared in training:
- The model never received gradient signal for that code
- The decoder weight for that code was only randomly initialized
- The model should NOT be able to predict it well

**Yet the model predicts these codes with 99.99% accuracy!**

---

## Possible Explanations (Most to Least Likely)

### Explanation 1: Code Index 0 is Special (Most Likely)

Looking at your data processing in `moe_flashattn_4_core.py`:

```python
def conv_target(target: str, len_dy: int, target_cd_cnt: int) -> List[List[int]]:
    # ...
    if not day_codes:
        day_codes = [0]  # Padding with 0!
    result.append(day_codes)
```

**If code index 0 is used for padding or "no code":**
- Every day with no actual target gets `[0]` as its target
- This means code 0 appears in almost every sample
- But if `code_frequencies` was computed EXCLUDING padding, code 0 shows freq=0

**Check:** Are any of the 598 "zero" codes actually index 0 or near 0?

#### Explanation 2: Train/Val Vocabulary Mismatch

If training and validation data were processed separately:
- Training might have 5,699 unique codes
- Validation might have different codes
- The 598 codes might be validation-only codes

**This would mean:**
- `code_frequencies` was computed only on training targets
- Validation has 598 codes that weren't in training
- These are "out-of-vocabulary" codes for the model

**But this doesn't explain why the model predicts them WELL.**

#### Explanation 3: Temporal Drift / New Codes

If training and validation are from different time periods:
- New ICD/procedure codes added in validation period
- These codes have freq=0 in training
- But might co-occur with patterns the model learned

**This still doesn't explain the +6.4 logit.**

#### Explanation 4: Data Leakage

The most concerning possibility:
- Some validation data leaked into training
- Or code frequencies were computed on wrong subset
- The model actually saw these codes but frequencies don't reflect it

---

### Why +6.4 Logit is Suspicious

A logit of +6.4 means the model is **99.8% confident** these codes are present.

For a code that "never appeared in training":
- Random initialization would give logit ≈ 0
- No gradient updates means no learning
- Expected logit should be ~0 or slightly negative

**Getting +6.4 requires the model to have learned something specific about these codes.**

#### The Only Way This Makes Sense

If "zero" codes are actually **padding tokens** or **special "no-code" indicators**:

```python
# In your target encoding:
targets_multihot[i, day_idx, 0] = 1.0  # Code 0 = "valid day marker" or similar
```

Then:
- Code 0 appears in every valid day → but excluded from frequency count
- Model learns "code 0 = valid day" → predicts high
- This is NOT actually a "code" but a data artifact

---

### How to Investigate

#### Investigation 1: Check What the Zero Codes Are

```python
# Find the actual code indices in the "zero" tier
zero_mask = code_frequencies == 0
zero_code_indices = np.where(zero_mask)[0]
print(f"Zero code indices: {zero_code_indices[:20]}...")  # First 20

# Check if index 0 is included
print(f"Is index 0 in zero codes? {0 in zero_code_indices}")
```

#### Investigation 2: Check Code 0 Frequency in Targets

```python
# Check how often code index 0 appears in validation targets
code_0_count = 0
for batch in val_loader:
    targets_mh = batch['target_multihot']  # [batch, len_dy, num_codes]
    code_0_count += (targets_mh[:, :, 0] > 0).sum().item()
    
print(f"Code 0 appears {code_0_count} times in validation")
```

#### Investigation 3: Map Zero Codes Back to Original Vocabulary

```python
# If you have a code-to-description mapping:
# Check what these 598 codes actually are
for code_idx in zero_code_indices[:10]:
    code_name = ind2w.get(code_idx, "UNKNOWN")
    print(f"Code {code_idx}: {code_name}")
```

---

### My Best Hypothesis

Based on the evidence:

**The "zero" tier likely contains index 0, which is used as a padding/placeholder in your target encoding, NOT an actual medical code.**

Evidence:
1. The model predicts it with extremely high confidence (learned pattern)
2. It appears 66,728 times (consistent with "every valid day")
3. The huge margin (21.6) suggests very consistent behavior

**If confirmed, you should:**
1. Exclude index 0 from the tier analysis
2. Re-run the diagnostic with `code_frequencies[1:]` or similar
3. The "true" zero codes (never seen in training) would then show different behavior

---

### Immediate Action

Before proceeding with tier-aware batching, please investigate:

1. **What are the 598 zero-code indices?** Are they including index 0?
2. **How was `code_frequencies` computed?** Was padding excluded?
3. **Check the target encoding:** Is `[0]` used for empty days?

This investigation will ensure your diagnostic is accurate and your intervention is targeting the right problem.

# Result for priroity 1: detailed interpretation of diagnostic results - 3.4M members
- Jan 28, 2026
- Use model MODEL_PATH='logs/exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2/exp2b_flash_learned_pool/saved_models/exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20260110_112709_final.pt'


## Part 1: Embedding Analysis Results

### Raw Data Summary

| Tier | Norm Mean | Norm Std | Norm Min | Norm Max | Near Zero | # Codes |
|------|-----------|----------|----------|----------|-----------|---------|
| **Common** | 1.42 | 0.27 | 0.80 | 2.21 | 0 | 1,169 |
| **Medium** | 1.49 | 0.15 | 1.09 | 2.17 | 0 | 1,754 |
| **Rare** | 1.41 | 0.05 | 1.26 | 1.68 | 0 | 1,748 |
| **Tail** | 1.46 | 0.03 | 1.35 | 1.54 | 0 | 1,175 |
| **Zero** | 1.47 | 0.03 | 1.09 | 1.54 | 0 | 451 |

### Interpretation

#### Finding 1: ✅ NO Embedding Collapse Detected

```
Decoder Weight Norms by Tier:
─────────────────────────────────────────────────────────
Common  ████████████████████████████ 1.42 ± 0.27
Medium  █████████████████████████████ 1.49 ± 0.15
Rare    ████████████████████████████ 1.41 ± 0.05
Tail    █████████████████████████████ 1.46 ± 0.03
Zero    █████████████████████████████ 1.47 ± 0.03
        ────────────────────────────────────────────
        0        0.5        1.0        1.5        2.0
                          Collapse threshold: 0.1
```

**Key Observations:**
1. **All tiers have healthy, similar norms (~1.4-1.5)** - actually slightly HIGHER than the smaller model (~1.1)
2. **Zero codes near zero across all tiers** - no embedding collapse
3. **Variance decreases with rarity** (std: 0.27 → 0.03) - rare/tail codes have MORE uniform weights

#### Comparison with Smaller Model

| Tier | Smaller Model Norm | 3.4M Model Norm | Change |
|------|-------------------|-----------------|--------|
| Common | 1.14 | 1.42 | +24.6% |
| Medium | 1.11 | 1.49 | +34.2% |
| Rare | 1.13 | 1.41 | +24.8% |
| Tail | 1.15 | 1.46 | +27.0% |

**Interpretation:** The larger model developed higher-magnitude decoder weights overall. This suggests:
- More training → weights moved further from initialization
- The model became more "opinionated" about all codes
- But this didn't translate to better predictions for rare/tail (as we'll see in logit analysis)

---

## Part 2: Logit Analysis Results

### Raw Data Summary

| Tier | Positive Samples | Logit (y=1) | % > 0 | Logit (y=0) | Margin |
|------|------------------|-------------|-------|-------------|--------|
| **Common** | 530,594 | -2.26 | 20.1% | -8.70 | 6.44 |
| **Medium** | 10,460 | -6.39 | 2.3% | -12.62 | 6.23 |
| **Rare** | 365 | -9.68 | 0.0% | -15.01 | 5.34 |
| **Tail** | 17 | -14.69 | 0.0% | -16.45 | 1.76 |
| **Zero** | 54,464 | +4.76 | 99.8% | -16.74 | 21.49 |

### Detailed Interpretation

#### Finding 2: ⚠️ SEVERE Logit Suppression for Rare/Tail (WORSE Than Smaller Model)

```
Logit Distribution by Tier (when y=1):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                                        0 (Decision Boundary)
                                                        │
Tail    ▓▓ (-14.69)                                     │
Rare       ▓▓▓ (-9.68)                                  │
Medium         ▓▓▓▓▓ (-6.39)                            │
Common              ▓▓▓▓▓▓▓▓ (-2.26)                    │
Zero                                               ▓▓▓▓▓▓▓▓ (+4.76)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     -16    -14    -12    -10    -8     -6     -4     -2      0      2      4      6
```

**Probability Conversion (sigmoid):**

| Tier | Logit (y=1) | Probability | Interpretation |
|------|-------------|-------------|----------------|
| Common | -2.26 | ~9.4% | Low, but some chance |
| Medium | -6.39 | ~0.17% | Very low |
| Rare | -9.68 | ~0.006% | Negligible |
| Tail | -14.69 | ~0.00004% | Essentially zero |

#### Finding 3: Comparison with Smaller Model - The Problem Got WORSE for Tail

| Tier | Smaller Model Logit | 3.4M Model Logit | Change | Interpretation |
|------|---------------------|------------------|--------|----------------|
| Common | -2.41 | -2.26 | +0.15 | Slightly improved |
| Medium | -7.05 | -6.39 | +0.66 | Improved |
| Rare | -11.38 | -9.68 | +1.70 | **Improved** |
| **Tail** | -12.93 | **-14.69** | **-1.76** | **WORSE!** |

**Critical Insight:** With 10× more training data:
- Common/medium/rare codes all improved (logits moved toward 0)
- **Tail codes got WORSE** (logits moved further from 0)

This is **the Matthew Effect in action**: "The rich get richer, the poor get poorer."
- More data → more gradient updates to common codes → common improves
- More data → rare/tail still rarely seen → relative disadvantage increases

#### Finding 4: Margin Analysis - Mixed Results

| Tier | Smaller Model Margin | 3.4M Model Margin | Change |
|------|---------------------|-------------------|--------|
| Common | 6.04 | 6.44 | +0.40 ✅ |
| Medium | 4.80 | 6.23 | +1.43 ✅ |
| Rare | 2.88 | 5.34 | +2.46 ✅ |
| **Tail** | 2.22 | **1.76** | **-0.46** ⚠️ |

**Interpretation:**
- **Good news:** Discrimination IMPROVED for common/medium/rare (margins increased)
- **Bad news:** Discrimination DECREASED for tail (margin shrunk from 2.22 to 1.76)

The 3.4M model learned to better separate positive vs negative for most tiers, but for tail codes, the separation actually got WORSE. This suggests:
- With more training, the model learned to be "even more confident" that tail codes should be negative
- The rare signal from tail codes got even more overwhelmed

#### Finding 5: Zero Recall Persists

```python
'rare':   {'pct_pos_above_zero': 0.0%}  # Model NEVER predicts rare codes
'tail':   {'pct_pos_above_zero': 0.0%}  # Model NEVER predicts tail codes
```

**Identical to smaller model:** Despite 10× more data, the model still achieves **0% recall** for rare/tail codes. The problem is structural, not data-quantity-related.

#### Finding 6: The Zero-Code Anomaly Persists

```python
'zero': {
  'num_positive_samples': 54464,   # Still suspicious!
  'logit_pos_mean': +4.76,         # Highly positive
  'pct_pos_above_zero': 99.8%,     # Almost always predicts positive
  'margin': 21.49                  # Enormous margin
}
```

**Same anomaly as before:** Codes with training frequency=0 somehow have positive samples in validation and the model predicts them with high confidence.

This data issue needs investigation. 451 codes are marked as "zero frequency" but have 54,464 positive validation samples. Possible causes:
1. Code vocabulary mismatch between train/validation
2. Temporal distribution shift (new codes in validation period)
3. Incorrect frequency computation

---

## Part 3: Comprehensive Comparison Table

### Full Side-by-Side Analysis

| Metric | Smaller Model | 3.4M Model | Change | Direction |
|--------|--------------|------------|--------|-----------|
| **Embedding Norms** |
| Common norm | 1.14 | 1.42 | +24.6% | Higher |
| Tail norm | 1.15 | 1.46 | +27.0% | Higher |
| Collapse detected | No | No | Same | ✅ |
| **Logit When Positive** |
| Common logit | -2.41 | -2.26 | +0.15 | Better ✅ |
| Medium logit | -7.05 | -6.39 | +0.66 | Better ✅ |
| Rare logit | -11.38 | -9.68 | +1.70 | Better ✅ |
| Tail logit | -12.93 | -14.69 | -1.76 | **WORSE** ⚠️ |
| **Discrimination (Margins)** |
| Common margin | 6.04 | 6.44 | +0.40 | Better ✅ |
| Medium margin | 4.80 | 6.23 | +1.43 | Better ✅ |
| Rare margin | 2.88 | 5.34 | +2.46 | Better ✅ |
| Tail margin | 2.22 | 1.76 | -0.46 | **WORSE** ⚠️ |
| **Recall (% above 0)** |
| Common | 18.8% | 20.1% | +1.3% | Better ✅ |
| Medium | 2.2% | 2.3% | +0.1% | Same |
| Rare | 0.0% | 0.0% | 0 | Same |
| Tail | 0.0% | 0.0% | 0 | Same |

---

## Part 4: Theoretical Framework Connection

### The Gradient Starvation Effect is AMPLIFIED at Scale

From the expert discussion:
> "The training dynamics naturally drift into a head-dominated update regime, and neither longer training nor higher per-positive weights is addressing the mechanism that makes tail signal effectively vanish."

**Your 3.4M model demonstrates this perfectly:**

1. **More data helped common/medium/rare** - they got more samples, more gradient, better learning
2. **More data HURT tail codes** - the relative disadvantage increased; their signal was diluted further

This is the key insight: **More data without intervention makes the problem worse for the lowest-frequency codes.**

### Why Tail Codes Got Worse

The mechanism:
1. In the smaller model: tail codes appeared sporadically, learned weak negative logits (-12.9)
2. In the 3.4M model: tail codes appeared at the SAME low rate, but common codes appeared 10× more
3. The model received 10× more "pressure" to be good at common codes
4. The tail code decoder weights, while not collapsed, were pushed toward even more negative outputs

**Mathematical intuition:**
```
Total gradient ≈ Σ (gradient from each tier)
              ≈ N_common × grad_common + N_tail × grad_tail

With 10× data:
              ≈ 10×N_common × grad_common + 10×N_tail × grad_tail

The absolute increase in tail gradient (10×N_tail) is dwarfed by 
the absolute increase in common gradient (10×N_common)

If N_common >> N_tail, the relative disadvantage increases.
```

### Margin Paradox Explained

**Why did rare MARGIN improve but tail MARGIN worsen?**

Looking at the data:
- Rare: 365 positive samples (enough for some learning)
- Tail: 17 positive samples (essentially noise)

With more training:
- Rare codes (365 samples) got enough repeated exposure to learn better discrimination
- Tail codes (17 samples) didn't even appear in most batches; the few signals were drowned out

**Threshold effect:** There appears to be a minimum sample count (~100-300?) needed for a code to benefit from more training. Below this, more training makes things worse.

---

## Part 5: Diagnosis Summary

### Final Verdict

| Diagnostic Check | Result | Details |
|------------------|--------|---------|
| **Embedding Collapse** | ❌ NOT detected | All norms healthy (~1.4-1.5) |
| **Weak Signal** | ✅ **SEVERE** | Tail logit = -14.69 (prob ~0.00004%) |
| **Ranking Problem** | ⚠️ **CRITICAL for Tail** | Tail margin = 1.76 (degraded from 2.22) |

### Key Conclusions

1. **The problem is NOT capacity/representation** - decoder weights are healthy and actually stronger than smaller model

2. **The problem IS optimization dynamics** - the model learned to suppress rare/tail codes even more aggressively

3. **More data alone won't help** - in fact, it made tail codes worse (logit: -12.9 → -14.7)

4. **Tier-aware batching is even MORE critical for the 3.4M model** because:
   - The gradient imbalance is more extreme
   - Tail codes need guaranteed exposure to counteract the overwhelming common code signal
   - The model has learned stronger "priors" against tail codes that need to be overcome

### Recommended Quota for Tier-Aware Batching

Given the extreme suppression, I recommend **more aggressive quotas** for the 3.4M model:

| Batch Size | Previous Recommendation | New Recommendation | Rationale |
|------------|------------------------|-------------------|-----------|
| 32 | rare=4, tail=4 | rare=5, tail=6 | Tail needs more exposure |
| 64 | rare=6, tail=6 | rare=8, tail=10 | Compensate for worse suppression |
| 128 | rare=10, tail=10 | rare=12, tail=16 | Ensure tail sees ~12% of batch |

The goal is to ensure `tail_frac` stays above 10% of batch (not just 5%) given how severely the 3.4M model has suppressed tail codes.

### Success Criteria (Adjusted for 3.4M Model)

| Metric | Current | Target | Rationale |
|--------|---------|--------|-----------|
| tail_top10_acc | 0% | >2% | Need to move off zero |
| tail logit (y=1) | -14.69 | >-8 | Move toward decision boundary |
| tail margin | 1.76 | >3 | Restore discrimination |
| train_grad_tier_tail_frac | ~0.1% | >8% | Prevent gradient starvation |