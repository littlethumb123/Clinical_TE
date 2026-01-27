I'll first examine the existing model code to understand the structure and ensure the diagnostic and tier-aware batching implementations integrate seamlessly.

[1 tool called]

The file is quite large. Let me search for key structures to understand the architecture.

[5 tools called]

Let me look at how the training loop creates the DataLoader and examine the target structure:

[4 tools called]

Let me look at how the dataloader is created and understand the experiment run function:

[2 tools called]

Now let me understand how targets are stored in the dataset and find the comprehensive evaluation functions that compute per-code metrics:

[2 tools called]

Now I have a complete picture of the codebase. Let me provide the implementation for both Priority 1 (Diagnostic) and Priority 2 (Tier-Aware Batching).

---

# Implementation for Priority 1 & Priority 2

Based on the code structure in `moe_flashattn_4.py`, here are the implementations designed to:
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
# Purpose: Guarantee minimum rare/tail positive samples per batch
# to prevent gradient starvation during training.
#
# This directly addresses the root cause: rare codes appear too
# sporadically, causing their gradient signal to be averaged out
# by the steady stream of common code updates.
#
# Success criteria:
#   - train_grad_tier_tail_frac > 5% at end of training
#   - tail_top10_acc > 1%
# ============================================================

import torch
import numpy as np
from torch.utils.data import Sampler, Dataset
from typing import Dict, List, Iterator, Optional, Tuple
from collections import defaultdict
import random


class TierAwareBatchSampler(Sampler):
    """
    Batch sampler that guarantees minimum representation of rare/tail codes.
    
    Strategy:
    1. Pre-compute which samples contain rare/tail positive codes
    2. Each batch includes:
       - `rare_quota` samples with rare code positives
       - `tail_quota` samples with tail code positives
       - Remaining samples from general pool
    
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
            batch_size=32,
            rare_quota=4,    # At least 4 samples with rare codes per batch
            tail_quota=4,    # At least 4 samples with tail codes per batch
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
            rare_quota: Minimum samples with rare code positives per batch
            tail_quota: Minimum samples with tail code positives per batch
            shuffle: Whether to shuffle within each pool
            drop_last: Whether to drop the last incomplete batch
            percentile_boundaries: (tail_thresh, rare_thresh, medium_thresh)
            verbose: Print initialization statistics
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.rare_quota = rare_quota
        self.tail_quota = tail_quota
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        
        # Validate quotas
        assert rare_quota + tail_quota <= batch_size, \
            f"Combined quotas ({rare_quota + tail_quota}) exceed batch_size ({batch_size})"
        
        # Build tier code indices
        self._build_tier_indices(code_frequencies, percentile_boundaries)
        
        # Build sample-to-tier mapping
        self._build_sample_tier_mapping(verbose)
        
        # Calculate number of batches
        self._calculate_num_batches()
    
    def _build_tier_indices(
        self,
        code_frequencies: np.ndarray,
        percentile_boundaries: Tuple[float, float, float]
    ):
        """Build tier code index sets."""
        freq_nz = code_frequencies[code_frequencies > 0]
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
    
    def _build_sample_tier_mapping(self, verbose: bool):
        """
        Pre-compute which samples contain rare/tail positive codes.
        
        This is done ONCE during initialization for efficiency.
        """
        # Samples that have at least one rare positive code
        self.samples_with_rare = []
        # Samples that have at least one tail positive code
        self.samples_with_tail = []
        # All other samples (may overlap, but that's OK)
        self.general_samples = list(range(self.num_samples))
        
        rare_codes = self.tier_code_indices['rare']
        tail_codes = self.tier_code_indices['tail']
        
        if verbose:
            print(f"TierAwareBatchSampler: Building sample-tier mapping for {self.num_samples} samples...")
        
        for idx in range(self.num_samples):
            # Get target codes for this sample
            # The dataset stores targets as nested lists: List[List[int]]
            # Each outer list is a day, inner list is codes for that day
            sample_data = self.dataset[idx]
            target_list = sample_data['target']  # List[List[int]]
            
            # Flatten all positive codes for this sample
            all_positive_codes = set()
            for day_codes in target_list:
                if day_codes:  # Non-empty day
                    all_positive_codes.update(day_codes)
            
            # Check tier membership
            has_rare = bool(all_positive_codes & rare_codes)
            has_tail = bool(all_positive_codes & tail_codes)
            
            if has_rare:
                self.samples_with_rare.append(idx)
            if has_tail:
                self.samples_with_tail.append(idx)
        
        if verbose:
            print(f"  Samples with rare codes: {len(self.samples_with_rare)} "
                  f"({len(self.samples_with_rare)/self.num_samples:.1%})")
            print(f"  Samples with tail codes: {len(self.samples_with_tail)} "
                  f"({len(self.samples_with_tail)/self.num_samples:.1%})")
            
            # Warn if quotas may not be satisfiable
            if len(self.samples_with_rare) < self.rare_quota * 10:
                print(f"  ⚠️ Warning: Few samples with rare codes. May need to reduce rare_quota.")
            if len(self.samples_with_tail) < self.tail_quota * 10:
                print(f"  ⚠️ Warning: Few samples with tail codes. May need to reduce tail_quota.")
    
    def _calculate_num_batches(self):
        """Calculate number of batches per epoch."""
        # We need to cycle through all samples while maintaining quotas
        # The limiting factor is whichever pool is smallest relative to its quota
        
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with guaranteed tier representation."""
        # Shuffle pools if needed
        if self.shuffle:
            rare_pool = self.samples_with_rare.copy()
            tail_pool = self.samples_with_tail.copy()
            general_pool = self.general_samples.copy()
            random.shuffle(rare_pool)
            random.shuffle(tail_pool)
            random.shuffle(general_pool)
        else:
            rare_pool = self.samples_with_rare.copy()
            tail_pool = self.samples_with_tail.copy()
            general_pool = self.general_samples.copy()
        
        # Track used samples to avoid duplicates within epoch
        used_samples = set()
        rare_idx = 0
        tail_idx = 0
        general_idx = 0
        
        batches_yielded = 0
        
        while batches_yielded < self.num_batches:
            batch = []
            
            # 1. Add rare quota
            rare_added = 0
            while rare_added < self.rare_quota and rare_idx < len(rare_pool):
                sample_idx = rare_pool[rare_idx]
                rare_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    rare_added += 1
            
            # 2. Add tail quota
            tail_added = 0
            while tail_added < self.tail_quota and tail_idx < len(tail_pool):
                sample_idx = tail_pool[tail_idx]
                tail_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    tail_added += 1
            
            # 3. Fill remainder from general pool
            remaining = self.batch_size - len(batch)
            while remaining > 0 and general_idx < len(general_pool):
                sample_idx = general_pool[general_idx]
                general_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    remaining -= 1
            
            # Handle pool exhaustion - reset with reshuffling
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
                # Reset used_samples for next pass through data
                used_samples.clear()
            
            # Yield batch if it meets size requirements
            if len(batch) >= self.batch_size or (not self.drop_last and len(batch) > 0):
                if self.shuffle:
                    random.shuffle(batch)  # Shuffle within batch
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
    rare_quota: int = 4,
    tail_quota: int = 4,
    num_workers: int = 4,
    collate_fn = None
):
    """
    Factory function to create a DataLoader with tier-aware batching.
    
    Drop-in replacement for standard DataLoader creation.
    
    Args:
        dataset: ClinicalDataset
        code_frequencies: From prepared_data.code_frequencies
        config: Model config with batch_size
        rare_quota: Min samples with rare codes per batch
        tail_quota: Min samples with tail codes per batch
        num_workers: Number of data loading workers
        collate_fn: Custom collate function (create_collate_fn(config))
    
    Returns:
        DataLoader with tier-aware batching
    
    Usage:
        train_loader = create_tier_aware_dataloader(
            dataset=prepared_data.train_dataset,
            code_frequencies=prepared_data.code_frequencies,
            config=config,
            rare_quota=4,
            tail_quota=4,
            collate_fn=create_collate_fn(config)
        )
    """
    from torch.utils.data import DataLoader
    
    sampler = TierAwareBatchSampler(
        dataset=dataset,
        code_frequencies=code_frequencies,
        batch_size=config.batch_size,
        rare_quota=rare_quota,
        tail_quota=tail_quota,
        shuffle=True,
        drop_last=True,
        verbose=True
    )
    
    if collate_fn is None:
        collate_fn = create_collate_fn(config)
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    print(f"✅ Created tier-aware DataLoader:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Rare quota: {rare_quota} samples/batch")
    print(f"   Tail quota: {tail_quota} samples/batch")
    print(f"   Total batches: {len(sampler)}")
    
    return loader


# ============================================================
# INTEGRATION: Modified create_dataloaders function
# ============================================================
# Add this function to replace/augment the existing create_dataloaders

def create_dataloaders_with_tier_aware(
    train_data,  # ClinicalDataset or DataFrame
    val_data,    # ClinicalDataset or DataFrame  
    config,
    code_frequencies: np.ndarray,
    use_tier_aware: bool = True,
    rare_quota: int = 4,
    tail_quota: int = 4,
    use_bucketing: bool = False,  # Mutually exclusive with tier_aware
    train_data_df = None,
    logger = None
):
    """
    Create train and validation DataLoaders with optional tier-aware batching.
    
    This is a drop-in replacement for the existing create_dataloaders function
    that adds tier-aware batching capability.
    
    Args:
        train_data: Training dataset (ClinicalDataset or DataFrame)
        val_data: Validation dataset (ClinicalDataset or DataFrame)
        config: Model configuration
        code_frequencies: Code frequency array for tier computation
        use_tier_aware: Whether to use tier-aware batching
        rare_quota: Min rare samples per batch (if tier_aware)
        tail_quota: Min tail samples per batch (if tier_aware)
        use_bucketing: Whether to use bucketing (mutually exclusive with tier_aware)
        train_data_df: Original DataFrame for bucketing
        logger: Optional logger
    
    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Handle Dataset vs DataFrame input
    if isinstance(train_data, Dataset):
        train_dataset = train_data
    else:
        train_dataset = ClinicalDataset(train_data, config)
        train_data_df = train_data
    
    if isinstance(val_data, Dataset):
        val_dataset = val_data
    else:
        val_dataset = ClinicalDataset(val_data, config)
    
    n_workers = min(4, os.cpu_count() // 4) if os.cpu_count() else 2
    collate_fn = create_collate_fn(config)
    
    # Validation loader is always standard
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Training loader: tier-aware vs bucketing vs standard
    if use_tier_aware and not use_bucketing:
        if logger:
            logger.info(f"Using TIER-AWARE batching (rare_quota={rare_quota}, tail_quota={tail_quota})")
        
        train_loader = create_tier_aware_dataloader(
            dataset=train_dataset,
            code_frequencies=code_frequencies,
            config=config,
            rare_quota=rare_quota,
            tail_quota=tail_quota,
            num_workers=n_workers,
            collate_fn=collate_fn
        )
    
    elif use_bucketing and not use_tier_aware:
        if logger:
            logger.info("Using BUCKETING batch sampler")
        
        if train_data_df is None:
            raise ValueError("train_data_df required for bucketing")
        
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
            collate_fn=collate_fn
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
            collate_fn=collate_fn
        )
    
    if logger:
        logger.info(f"Train loader: {len(train_loader)} batches")
        logger.info(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader


# ============================================================
# USAGE EXAMPLE (copy to Jupyter cell)
# ============================================================
"""
# Option 1: Use the factory function directly
train_loader = create_tier_aware_dataloader(
    dataset=prepared_data.train_dataset,
    code_frequencies=prepared_data.code_frequencies,
    config=config,
    rare_quota=4,  # Adjust based on batch_size
    tail_quota=4,  # Adjust based on batch_size
    collate_fn=create_collate_fn(config)
)

# Option 2: Use the unified create_dataloaders function
train_loader, val_loader = create_dataloaders_with_tier_aware(
    train_data=prepared_data.train_dataset,
    val_data=prepared_data.val_dataset,
    config=config,
    code_frequencies=prepared_data.code_frequencies,
    use_tier_aware=True,
    rare_quota=4,
    tail_quota=4
)

# Then use train_loader in training as usual
for epoch in range(num_epochs):
    train_metrics = train_epoch(
        model=model,
        dataloader=train_loader,  # <-- Uses tier-aware batching
        optimizer=optimizer,
        ...
        gradient_tier_analyzer=gradient_tier_analyzer  # Monitor tier fractions
    )
    
    # Check if tier-aware batching is working
    # train_grad_tier_tail_frac should be > 5% (target)
    print(f"Tail gradient fraction: {train_metrics.get('train_grad_tier_tail_frac', 0):.2%}")
"""


# ============================================================
# VERIFICATION: Test tier-aware batching is working
# ============================================================

def verify_tier_aware_batching(
    dataloader,
    code_frequencies: np.ndarray,
    num_batches: int = 10
) -> Dict[str, Any]:
    """
    Verify that tier-aware batching is producing balanced batches.
    
    Run this after creating the DataLoader to confirm it's working.
    """
    from collections import defaultdict
    
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
    
    print(f"Verifying tier-aware batching over {num_batches} batches...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        targets_mh = batch['target_multihot']  # [batch, len_dy, num_codes]
        
        # Sum across batch and days to get code presence
        code_presence = (targets_mh.sum(dim=(0, 1)) > 0).numpy()
        present_codes = set(np.where(code_presence)[0])
        
        for tier_name, tier_code_set in tier_codes.items():
            tier_present = len(present_codes & tier_code_set)
            batch_tier_counts[tier_name].append(tier_present)
    
    # Compute statistics
    results = {}
    print("\nTier representation per batch:")
    print("-" * 50)
    for tier_name in ['common', 'medium', 'rare', 'tail']:
        counts = batch_tier_counts[tier_name]
        mean_count = np.mean(counts)
        min_count = np.min(counts)
        max_count = np.max(counts)
        results[tier_name] = {
            'mean': mean_count,
            'min': min_count,
            'max': max_count,
            'all_counts': counts
        }
        print(f"  {tier_name.upper():8s}: mean={mean_count:.1f} codes, "
              f"range=[{min_count}, {max_count}]")
    
    # Check if tail/rare are consistently present
    tail_present_rate = np.mean([c > 0 for c in batch_tier_counts['tail']])
    rare_present_rate = np.mean([c > 0 for c in batch_tier_counts['rare']])
    
    print(f"\n✅ Tail codes present in {tail_present_rate:.1%} of batches")
    print(f"✅ Rare codes present in {rare_present_rate:.1%} of batches")
    
    if tail_present_rate < 0.9 or rare_present_rate < 0.9:
        print("⚠️ Warning: Tier-aware batching may not be working correctly!")
    else:
        print("✅ Tier-aware batching verified!")
    
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
# Create tier-aware data loader
train_loader = create_tier_aware_dataloader(
    dataset=prepared_data.train_dataset,
    code_frequencies=prepared_data.code_frequencies,
    config=config,
    rare_quota=4,  # Adjust: ~10-15% of batch_size
    tail_quota=4
)

# Verify it's working
verify_tier_aware_batching(train_loader, prepared_data.code_frequencies)

# Use in training
# (No changes needed to train_epoch - it will automatically benefit)
```

**Recommended quota values:**
- For `batch_size=32`: `rare_quota=4, tail_quota=4`
- For `batch_size=64`: `rare_quota=6, tail_quota=6`  
- For `batch_size=128`: `rare_quota=10, tail_quota=10`

---

Let me know if you need me to clarify any part of the implementation or if you'd like me to elaborate on specific integration points with the existing code!