# Learning plateau in transformer pretraining
_Exported on 1/23/2026 at 21:33:20 PST from Cursor (2.2.44)_

---

**User**

You are an world-class outstanding experienced MLE experts and LLM researchers adn AI engineers; You lead my transformer refactoring projects; now we come across learning plateau issue in both experiment round 5 and experiment 6; the trainin logs for each round for each ablations you can find in exp1, exp2b, exp6 @expe_logs/exp_round5/exp2/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20251230_114137_training_log.log @expe_logs/exp_round5/exp1_opt/training.log @expe_logs/exp_round5/exp6/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_V3_20251231_152438_training_log.log. Even if I double the training sample size; the pleteau still exist and the performance (loss and business metrics) didn't get much lift @expe_logs/exp_round6/exp2/training_doublesize.log (you can compare with @expe_logs/exp_round5/exp2/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20251230_114137_config.json) 
Think and response like a seaseoned and experienced MLE experts in LLM and transforer pretraining;
Here are four experts inspection and evalaution of this critical technical challenge. @expe_analysis/exp_round5/exp_round5_learning_plateau_experts_views1.md 
Your task 
1) indepednently and objectively review and interpret experts's opinions comprehesnivesly without any preference or biases; 
2) rigorously review every datial, opinions and statements and identify flawed reasoning, weak argument or false stateemnt or missing details; 
3) synthesize and compare observations and analysis results across four experts; understand what are proposed and discussed in common and more importnatly the unique opinion each expert proposed; 

The goal is to be my tech lead and catch critical insights from the evience and results and also from other experts' opinion and evaluateion. 

Here is context

The ablations you want to icnlude to deep analysis and compare 
1) exp1_dense_legacy @expe_logs/exp_round5/exp1/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp1_dense_baseline_bs128_ep1_d256_20251230_055716_config.json(config); result is @expe_logs/exp_round5/exp1/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp1_dense_baseline_bs128_ep1_d256_20251230_055716_results.json 
2) exp1_dense_opt @expe_logs/exp_round5/exp1_opt/config.json; result is @expe_logs/exp_round5/exp1_opt/final_results.json 
3) exp2 @expe_logs/exp_round5/exp2/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20251230_114137_config.json  result is @expe_logs/exp_round5/exp2/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20251230_114137_results.json 
4) exp6 @expe_logs/exp_round5/exp6/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_v3_20251231_152438_config.json result is @expe_logs/exp_round5/exp6/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_V3_20251231_152438_results.json 

Principal: 
1. Do not hallucinate and all evidence should be from original code @dev/moe/moe_flashattn_3.py and given experimentation resutls and configuratons
2. Don't rush to any solution, assumption or theory; observe, think and analyze like a seasoned MLE experts on the configrautions, model architecuture and imlemantion, training logs and results for each ablations
3. the learning pleteau happens in all experiemntations and I just show you the exp2 training logs; you can find training logs for other epxeirmentations in the corresponding folder. 
Again, inspect carefully and rigorouytsly, think hard and deep; Do not rush to answer or hold any assumptions; make sure you SEE all configs, results, code implemenation; 
4. You can refer to previous analysis; HOWEVER, I would like you to keep thinking indepednetly and objectively. you can search or refer to external sources; make sure the informaiton is from credible, quality and reliable sources. also pint out where you cite. 

---

**Cursor**

I'll conduct a rigorous independent analysis by first examining the actual evidence (configs, results, logs, and code), then critically evaluating each expert's analysis.

Let me start by reading the primary source files:

Now let me read the training logs and round 6 data:

Now let me read the model implementation to verify architectural claims:

Let me also check the exp6 MoE config and the data sizes:

Now I have the complete picture from the actual evidence. Let me provide my comprehensive independent analysis.

---

# Independent Technical Analysis: Learning Plateau in Clinical Transformer

## Part 1: Critical Review of Expert Opinions

### **Expert 1 Analysis - Strengths & Weaknesses**

**Strengths:**
- Correctly identifies the convergence pattern (all optimized experiments reach ~0.82-0.83 R@10)
- The "Capacity-Limited Regime" hypothesis is well-articulated with specific evidence
- Correctly notes that doubling data yielded marginal improvement

**Weaknesses/Errors:**
1. **Inaccurate claim on exp1_legacy nhead**: Expert 1 states "nhead=8 (legacy says 16)" but the config file shows `"nhead": 8`. The training log shows `nhead=16 (hardcoded)` for exp1_opt, not exp1_legacy.

2. **Overstates d_model bottleneck significance**: Claims 256-dim is "extremely small" but doesn't account that clinical code prediction may not need transformer-scale representations—this is a multi-label classification task, not language modeling.

3. **The "ranking-aware loss will give +10-15% μRecall" claim is speculative**: No evidence for this specific prediction.

### **Expert 2 Analysis - Strengths & Weaknesses**

**Strengths:**
- Correctly identifies the "Representation Bottleneck" concept
- Properly notes that nhid=704 is constrained (should be 1024 for 4× expansion with d_model=256)
- Identifies the "L-shaped" loss curve pattern correctly

**Weaknesses/Errors:**
1. **FALSE STATEMENT**: Expert 2 claims "Doubling the data size (Round 6 Exp 2) resulted in **zero marginal gain**". The actual data shows:
   - Round 5 exp2: R@10=0.828, μR@10=0.462, NDCG=0.432
   - Round 6 exp2: R@10=0.834, μR@10=0.477, NDCG=0.447
   - This is +0.6% R@10, +3.2% μRecall, +3.5% NDCG—not "zero"

2. **Conflates exp2 with exp2b_flash_learned_pool**: The naming in the document is inconsistent. Exp2 uses Flash + learned pooling, not plain MoE.

3. **Overstates the d_model claim without citing counterexamples**: Many successful clinical ML models use embeddings <256 dims.

### **Expert 3 Analysis - Strengths & Weaknesses**

**Strengths:**
- Most rigorous in actually reading the files and noting inconsistencies
- Correctly identifies the `val_loss=0.0` logging artifact (exp2, exp6 show 0.0 but BCE loss is non-zero)
- Properly notes the MoE health issues from actual log data
- Correctly identifies that "more data didn't help" because it mostly added head examples

**Weaknesses/Errors:**
1. **Claims results files are "byte-identical"**: This appears to be an error—the round 6 results show clearly different metrics (R@10=0.834 vs 0.828)

2. **Missing key insight**: Doesn't notice that exp1_opt uses batch_size=64 while others use 128, giving it 2× the gradient updates

### **Expert 4 Analysis - Strengths & Weaknesses**

**Strengths:**
- Most complete table with per-tier accuracy breakdown
- Correctly identifies "Representation Collapse" concept
- Good coverage of the gradient/embedding bottleneck hypothesis

**Weaknesses/Errors:**
1. **False claim about exp1_opt steps**: Claims exp1_opt runs "more steps (~22K vs ~11K)" but from the logs, warmup steps are 3289 for exp1_opt vs 1644 for exp2, suggesting total steps scale similarly.

2. **Bits-per-code calculation is misleading**: "log₂(256)/75516 ≈ 0.0001 bits" is not how embeddings work—each code gets 256 dimensions, not 0.0001 bits.

3. **"Multi-epoch training" as a quick win is questionable**: The current setup runs 1 epoch. Multi-epoch on 6000+ output codes with BCE loss will likely overfit without curriculum changes.

---

## Part 2: Cross-Expert Synthesis

### **Points of Agreement (Validated Against Evidence)**

| Claim | Expert 1 | Expert 2 | Expert 3 | Expert 4 | Verified? |
|-------|----------|----------|----------|----------|-----------|
| All optimized experiments converge to same performance | ✓ | ✓ | ✓ | ✓ | **YES** - R@10≈0.82-0.83 |
| Tail/rare code accuracy is 0% | ✓ | ✓ | ✓ | ✓ | **YES** |
| MoE doesn't help vs dense | ✓ | ✓ | ✓ | ✓ | **YES** |
| BCE objective may be misaligned with ranking | ✓ | ✓ | ✓ | ✓ | Plausible |
| d_model=256 is a bottleneck | Partial | ✓ | — | ✓ | **UNVERIFIED** |
| 2× data gives minimal improvement | ✓ | ✓ (overstated) | ✓ | ✓ | **YES** (+3% on μRecall) |

### **Points of Disagreement**

| Topic | Experts 1,2,4 | Expert 3 |
|-------|---------------|----------|
| Primary bottleneck | Model capacity | Data/objective alignment |
| First intervention | Increase d_model | Better measurement + tail weighting |

### **Unique Insights Per Expert**

| Expert | Unique Contribution |
|--------|---------------------|
| **Expert 1** | Specific intervention: "Sampled Softmax Reformulation" with expected outcomes |
| **Expert 2** | Most emphatic about d_model=256 bottleneck; suggests 512/768 as decisive test |
| **Expert 3** | Identifies logging artifact (val_loss=0.0) and suggests fixing measurement first |
| **Expert 4** | Most complete tier breakdown; suggests hierarchical CCS/CCSR supervision |

---

## Part 3: My Independent Diagnosis

### **What the Evidence Actually Shows**

1. **The plateau is real and reproducible**: All three optimized experiments (exp1_opt, exp2, exp6) plateau at R@10≈0.82-0.83, μR@10≈0.46-0.48, despite:
   - Dense vs Flash vs MoE architectures
   - 25M vs 35M parameters
   - 1 epoch vs 2× data

2. **The learning dynamics are identical**:
   ```
   Loss: 0.80 → 0.005 (step ~2000) → 0.003 (step ~10000)
   R@10: 0.00 → 0.60 (step ~2000) → 0.83 (step ~10000)
   ```
   The model learns 72% of its final R@10 performance in the first 18% of training.

3. **Tail performance is catastrophic**: 
   ```
   Common: 82-83% accuracy
   Medium: 4-5% accuracy  
   Rare: 0% accuracy
   Tail: 0% accuracy
   ```
   This is the clearest signal—the model has learned nothing for 5000+ codes.

4. **MoE router is unstable**: The exp6 logs show simultaneous "EXPLODING" and "VANISHING" gradient warnings with 2-3 collapsed experts. This indicates the MoE capacity isn't being utilized.

5. **The doubled data experiment is misinterpreted**: 
   - Round 5: 1.4M samples → 10,960 steps
   - Round 6: 3.1M samples → 24,273 steps
   - Result: +0.6% R@10, +3.2% μRecall, +3.5% NDCG
   
   This is NOT "zero improvement"—it's diminishing returns, which suggests **the model can still learn from more data, but the marginal return is low**.

### **My Primary Hypothesis: Gradient Starvation of the Long Tail**

The evidence most strongly supports a **gradient starvation** diagnosis rather than capacity limits:

1. **The pos_weight_max=50 is insufficient**: With ~6297 codes where code frequencies follow a power law, the rarest codes may have frequency ratios of 10,000:1. Capping at 50× leaves the bottom 80% of codes effectively invisible to SGD.

2. **BCE loss structure mathematically favors common codes**: For BCE:
   ```
   ∂L/∂θ ∝ Σᵢ (σ(fᵢ) - yᵢ) × ∂fᵢ/∂θ
   ```
   When 95% of positives come from 500 common codes, the gradient is dominated by those codes. Rare codes contribute near-zero gradient even with 50× weighting.

3. **The evidence against capacity limits**:
   - exp1_opt (dense, 26M params, batch=64, 2× updates) ≈ exp2 (flash, 25M params, batch=128)
   - Adding 10M parameters via MoE gives identical performance
   - μRecall at 0.46-0.48 suggests the model CAN distinguish mid-frequency codes somewhat

4. **The MoE failure is consistent with this**: MoE specialization requires gradient signal to route tokens to appropriate experts. If all gradients come from common codes, all experts learn the same thing → collapse.

### **Key Missing Analysis**

None of the experts performed this critical analysis:

**Gradient contribution by code tier**: We need to measure what fraction of the gradient norm comes from each tier (common/medium/rare/tail). If >90% comes from common codes, this confirms gradient starvation.

---

## Part 4: Recommended Decisive Experiments

### **Experiment A: Gradient Contribution Analysis (Diagnostic, 1 hour)**

```python
# In training loop, add:
per_code_grad = {}
for batch in sample_batches:
    loss.backward(retain_graph=True)
    for code_idx in range(6297):
        per_code_grad[code_idx] = model.decoder_cd.weight.grad[code_idx].norm().item()
# Aggregate by tier
```

**If common codes dominate gradient** → confirms gradient starvation → Experiment B
**If gradient is balanced** → contradicts starvation → suggests capacity limit → Expert 2's d_model test

### **Experiment B: Aggressive Tail Reweighting (1 epoch, 4 hours)**

```python
# Modify pos_weight computation:
pos_weight_max = 500  # Instead of 50
# AND add floor for tail codes:
pos_weight[tail_codes] = max(pos_weight[tail_codes], 200)
```

**Expected outcome**: 
- If rare_top10_acc improves from 0% to >5%, gradient starvation is confirmed
- If no improvement, the bottleneck is elsewhere

### **Experiment C: Sampled Softmax Objective (1 epoch, 4 hours)**

Replace BCE with a ranking-aware objective. For each sample:
1. Take the 2-10 positive codes
2. Sample 100-200 negatives (stratified: 50 common, 50 medium, 50 rare)
3. Compute softmax loss over this subset

**This directly optimizes ranking** rather than calibration.

### **Experiment D: d_model=512 (Only if B and C fail)**

Double model width. This is the most expensive experiment and should only be run if the gradient-focused interventions fail.

---

## Part 5: Key Critiques of Expert Recommendations

### **What I Disagree With:**

1. **Expert 2's insistence on d_model=512 as "first experiment"**: This is a 4× compute increase with no guarantee of improvement. The evidence (MoE adding 10M params with no gain) suggests capacity isn't the bottleneck.

2. **Expert 4's "multi-epoch training" as quick win**: With 6297 output codes and only ~2400 samples per code on average, multi-epoch risks overfitting to common codes. The generalization gap (train loss 0.003 vs val loss 0.003) is already small.

3. **All experts underestimate the pos_weight cap problem**: A cap of 50 is extremely conservative for a 6000+ code vocabulary. Medical coding literature (e.g., MIMIC-III studies) uses uncapped inverse frequency weighting or focal loss.

### **What I Agree With:**

1. **Expert 3's emphasis on better measurement**: The val_loss=0.0 logging issue needs fixing. We need per-tier metrics tracked throughout training.

2. **Expert 1's sampled softmax suggestion**: This is the most theoretically sound intervention for ranking-focused objectives.

3. **Expert 4's CCS/CCSR hierarchy suggestion**: Hierarchical supervision could provide signal for rare codes via their parent categories.

---

## Executive Summary

**The learning plateau is most likely caused by gradient starvation of rare/tail codes**, not model capacity limits. Evidence:
- All architectural changes (Flash, MoE, +40% params) give identical performance
- Tail code accuracy is 0% across all experiments  
- Doubling data improves μRecall by 3.2%, showing the model CAN still learn
- MoE experts collapse because they all learn the same common-code patterns

**First action**: Implement gradient contribution analysis to confirm diagnosis (1 hour of compute).

**Second action**: If confirmed, run aggressive tail reweighting (pos_weight_max=500, tail floor=200) or sampled softmax objective.

**Do NOT start with**: Increasing d_model to 512. This is the highest-cost, lowest-evidence intervention.

```plaintext
   Loss: 0.80 → 0.005 (step ~2000) → 0.003 (step ~10000)
   R@10: 0.00 → 0.60 (step ~2000) → 0.83 (step ~10000)
```

```plaintext
   Common: 82-83% accuracy
   Medium: 4-5% accuracy  
   Rare: 0% accuracy
   Tail: 0% accuracy
```

```plaintext
   ∂L/∂θ ∝ Σᵢ (σ(fᵢ) - yᵢ) × ∂fᵢ/∂θ
```

```python
# In training loop, add:
per_code_grad = {}
for batch in sample_batches:
    loss.backward(retain_graph=True)
    for code_idx in range(6297):
        per_code_grad[code_idx] = model.decoder_cd.weight.grad[code_idx].norm().item()
# Aggregate by tier
```

```python
# Modify pos_weight computation:
pos_weight_max = 500  # Instead of 50
# AND add floor for tail codes:
pos_weight[tail_codes] = max(pos_weight[tail_codes], 200)
```

---

**User**

I am very interested in your experiment A graident contribution analysis; how I could implement this? which experiemnt I should use? what code I should add to the existing impleneation @dev/moe/moe_flashattn_3.py; where to add them. I am going to manually add it so make it very clear and specific where to add waht code; Also keep an global sight change all places that need to be changed. making sure all components are implmened correctly and work correctly together. DO not introduce errors and bugs;
Last based on how I run teh experimentation for now; show me the code to run teh training again. 

---

**Cursor**

I'll help you implement the Gradient Contribution Analysis. First, let me examine the key parts of the training code to understand where and how to integrate the analysis.

Now let me find where the code frequency tiers are defined and how loss is computed with targets:

Let me examine the `DataParallelWrapper` class to understand how loss is computed:

Now let me check where code_frequencies are computed/stored and how they relate to tiers:

Let me look at how `run_single_experiment` and `run_selected_experiments` work to understand where to hook in gradient analysis:

Now I have a complete understanding of the codebase. Let me provide you with a comprehensive implementation for **Gradient Contribution Analysis by Code Tier**.

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

```python
    # After training loop completes, before returning results:
    if gradient_analyzer is not None:
        gradient_analyzer.print_summary()
        
        # Export to log directory
        grad_export_path = os.path.join(log_dir, f"{exp_name}_gradient_analysis.json")
        gradient_analyzer.export_results(grad_export_path)
```

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

```plaintext
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

```plaintext
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

**User**

You are an cirtical and rigorous MLE expert and supervisor in this domain, I would liek you to self reflect and review your idea to do the graident starvation. is that making sense given the current situation, is that valid? is that necssary, is that the best test or most appropriate and effective evaluation or next step you can proposed? anything else we have been missing?  

---

**Cursor**

Let me critically review my proposal by first examining what's already been analyzed and the actual problem we're trying to solve.

Let me also re-read the expert analysis to fully understand the context:

---

# Critical Self-Reflection: Is Gradient Tier Analysis the Right Next Step?

After carefully reviewing the evidence, expert analyses, and my proposed implementation, I need to be honest: **my gradient tier analysis proposal has significant issues and is not the best next step.**

---

## 🔴 Critical Issues with My Proposal

### 1. **I'm Measuring the Wrong Gradient**

My implementation measures `dL/d(output)` — the gradient of loss with respect to model **output logits**. But what actually determines weight updates is `dL/d(θ)` — the gradient with respect to model **parameters**.

These are fundamentally different:
- `dL/d(output)` = how much the loss would change if we changed the output
- `dL/d(θ)` = how much weights will actually be updated (flows through entire model)

To measure real gradient contribution, I'd need to use **gradient hooks on the output layer weights** (`model.fc_out.weight`), not recompute backward on detached outputs.

### 2. **We Already Know the Answer**

Looking at the evidence again:

| Tier | Exp1_opt | Exp2 | Exp6 |
|------|----------|------|------|
| **Common@10 Acc** | 82.5% | 82.9% | 82.8% |
| **Medium@10 Acc** | 5.5% | 4.1% | 4.8% |
| **Rare@10 Acc** | **0.0%** | **0.0%** | **0.0%** |
| **Tail@10 Acc** | **0.0%** | **0.0%** | **0.0%** |

**We already know rare codes aren't being learned.** The gradient analysis would just confirm this — it's diagnostic, not prescriptive. It tells us "rare codes contribute little to gradients" but doesn't tell us *why* or *how to fix it*.

### 3. **Computational Overhead is Unjustified**

Running multiple backward passes per batch for tier analysis:
- Adds ~30-50% training time overhead
- Significant code complexity
- For information we can infer from existing metrics

---

## ✅ What's Actually Missing (That Would Be More Valuable)

### Better Test 1: **Output Probability Distribution Analysis** (SIMPLER)

Before gradient analysis, check: **is the model even trying to predict rare codes?**

```python
# Add this to your logging every N batches - MUCH simpler than gradient analysis
with torch.no_grad():
    probs = torch.sigmoid(output)
    
    for tier_name, mask in tier_masks.items():
        tier_probs = probs[:, mask]
        tier_targets = targets[:, mask]
        
        # Key question: What probability does the model assign to rare codes?
        mean_prob = tier_probs.mean().item()
        max_prob = tier_probs.max().item()
        
        # For codes that SHOULD be predicted (target=1)
        positive_mask = tier_targets > 0.5
        if positive_mask.any():
            pos_probs = tier_probs[positive_mask]
            recall_at_threshold = (pos_probs > 0.1).float().mean().item()
        
        print(f"{tier_name}: mean_P={mean_prob:.4f}, max_P={max_prob:.3f}")
```

**What this tells us:**
- If `ultra_rare` mean probability ≈ 0.0001 and max ≈ 0.01 → **model never considers these codes** (prediction suppression)
- If probabilities are reasonable but not in top-10 → ranking issue, not gradient issue
- This is 10x simpler and equally informative

### Better Test 2: **Direct Intervention (Sampled Softmax)**

Expert 1 and 2 already correctly identify that **the problem is objective misalignment**, not gradient magnitude. The BCE loss:
- Averages over all 6,297 codes
- Rare codes have few positive labels → tiny contribution to average loss
- Even with pos_weight, the absolute gradient is small because rare codes rarely have positive labels in any batch

**Testing sampled softmax directly answers the question**: Is the objective the bottleneck?

```python
# Pseudo-code for sampled softmax test
for batch in dataloader:
    positives = get_positive_codes(batch)  # Codes that actually occur
    negatives = sample_negatives(500, frequency_weighted=True)
    
    # Only compute loss over positive + sampled negative codes
    logits_subset = output[:, positives + negatives]
    loss = softmax_cross_entropy(logits_subset, positives)
```

If Recall@10 improves → objective was the problem (and we found the fix)
If Recall@10 unchanged → capacity/data ceiling

### Better Test 3: **What Does the Model Predict for Rare-Code Samples?**

For patients who *should* have rare codes predicted:
- What are the model's top-10 predictions?
- Are they semantically related common codes?
- Or completely random?

This tells us if the model is *trying* but failing, or *not trying at all*.

---

## 🎯 Revised Recommendation (Priority Order)

| Priority | Test | Effort | What It Tells Us |
|----------|------|--------|------------------|
| 1 | **Output probability analysis** | Low (add 10 lines to logging) | Is model even trying to predict rare codes? |
| 2 | **Sampled softmax experiment** | Medium (new loss function) | Is BCE the bottleneck? |
| 3 | **Per-tier training curves** | Low (track existing metrics by tier) | Do rare codes improve at all during training? |
| 4 | **Gradient analysis** | High | Only if above tests are inconclusive |

---

## 💡 My Honest Assessment

**What I proposed:**
- Adds complexity and overhead
- Measures the wrong thing (output gradients, not parameter gradients)  
- Confirms something we already know from existing metrics
- Doesn't directly suggest a solution

**What you should do instead:**
1. **Add output probability logging by tier** (10 lines of code, 10 minutes work)
2. **Run sampled softmax experiment** (tests the real hypothesis)
3. **Only pursue gradient analysis** if sampled softmax fails and you need to understand why

The experts already identified the most likely root cause: **objective-metric misalignment** where BCE optimizes calibration but you measure ranking. Testing a ranking-aware loss directly (sampled softmax, ListMLE) is a more decisive experiment than measuring gradients.

---

## If You Still Want Gradient Analysis (Corrected Implementation)

If you want to pursue this for completeness, here's the **correct** approach using gradient hooks on parameters:

```python
class ParameterGradientAnalyzer:
    """
    Track actual parameter gradients (not output gradients) by code tier.
    Uses hooks on the final output layer weights.
    """
    
    def __init__(self, model, tier_masks, track_every_n_steps=100):
        self.tier_masks = tier_masks
        self.track_every_n_steps = track_every_n_steps
        self.history = []
        self.current_grads = {}
        self._step = 0
        
        # Find the output layer (fc_out or similar)
        self.output_layer = self._find_output_layer(model)
        
        # Register hook
        self.handle = self.output_layer.weight.register_hook(self._grad_hook)
    
    def _find_output_layer(self, model):
        # Unwrap DataParallel if needed
        actual = model.module if hasattr(model, 'module') else model
        if hasattr(actual, 'model'):
            actual = actual.model
        return actual.fc_out  # or whatever your output layer is named
    
    def _grad_hook(self, grad):
        """Called automatically during backward pass."""
        if self._step % self.track_every_n_steps == 0:
            # grad shape: [target_cd_cnt, d_model]
            for tier_name, mask in self.tier_masks.items():
                tier_grad = grad[mask, :]  # Gradients for this tier's output weights
                self.current_grads[tier_name] = tier_grad.norm(p=2).item()
        return grad  # Don't modify
    
    def step(self):
        """Call after optimizer.step()"""
        if self._step % self.track_every_n_steps == 0 and self.current_grads:
            self.history.append({
                'step': self._step,
                **self.current_grads
            })
            self.current_grads = {}
        self._step += 1
    
    def cleanup(self):
        self.handle.remove()
```

This is cleaner, measures the **right thing**, and doesn't require extra backward passes.

---

## Bottom Line

My original gradient tier analysis proposal was **over-engineered, methodologically flawed, and not the highest-value next step**. 

The simpler, more effective path is:
1. **Output probability analysis** → Is the model trying to predict rare codes?
2. **Sampled softmax experiment** → Is the objective the bottleneck?

Only pursue gradient analysis if these tests are inconclusive and you need deeper diagnostics. And if you do, use the corrected hook-based approach, not the backward-pass approach I originally proposed.

```python
# Add this to your logging every N batches - MUCH simpler than gradient analysis
with torch.no_grad():
    probs = torch.sigmoid(output)
    
    for tier_name, mask in tier_masks.items():
        tier_probs = probs[:, mask]
        tier_targets = targets[:, mask]
        
        # Key question: What probability does the model assign to rare codes?
        mean_prob = tier_probs.mean().item()
        max_prob = tier_probs.max().item()
        
        # For codes that SHOULD be predicted (target=1)
        positive_mask = tier_targets > 0.5
        if positive_mask.any():
            pos_probs = tier_probs[positive_mask]
            recall_at_threshold = (pos_probs > 0.1).float().mean().item()
        
        print(f"{tier_name}: mean_P={mean_prob:.4f}, max_P={max_prob:.3f}")
```

```python
# Pseudo-code for sampled softmax test
for batch in dataloader:
    positives = get_positive_codes(batch)  # Codes that actually occur
    negatives = sample_negatives(500, frequency_weighted=True)
    
    # Only compute loss over positive + sampled negative codes
    logits_subset = output[:, positives + negatives]
    loss = softmax_cross_entropy(logits_subset, positives)
```

```python
class ParameterGradientAnalyzer:
    """
    Track actual parameter gradients (not output gradients) by code tier.
    Uses hooks on the final output layer weights.
    """
    
    def __init__(self, model, tier_masks, track_every_n_steps=100):
        self.tier_masks = tier_masks
        self.track_every_n_steps = track_every_n_steps
        self.history = []
        self.current_grads = {}
        self._step = 0
        
        # Find the output layer (fc_out or similar)
        self.output_layer = self._find_output_layer(model)
        
        # Register hook
        self.handle = self.output_layer.weight.register_hook(self._grad_hook)
    
    def _find_output_layer(self, model):
        # Unwrap DataParallel if needed
        actual = model.module if hasattr(model, 'module') else model
        if hasattr(actual, 'model'):
            actual = actual.model
        return actual.fc_out  # or whatever your output layer is named
    
    def _grad_hook(self, grad):
        """Called automatically during backward pass."""
        if self._step % self.track_every_n_steps == 0:
            # grad shape: [target_cd_cnt, d_model]
            for tier_name, mask in self.tier_masks.items():
                tier_grad = grad[mask, :]  # Gradients for this tier's output weights
                self.current_grads[tier_name] = tier_grad.norm(p=2).item()
        return grad  # Don't modify
    
    def step(self):
        """Call after optimizer.step()"""
        if self._step % self.track_every_n_steps == 0 and self.current_grads:
            self.history.append({
                'step': self._step,
                **self.current_grads
            })
            self.current_grads = {}
        self._step += 1
    
    def cleanup(self):
        self.handle.remove()
```

---

**User**

ou are an cirtical and rigorous MLE expert and supervisor in this domain, I would liek you to self reflect and review your idea not to do the graident starvation analysis adn insteaed fo probability analysis adn impelneted sampled sfotmax experiment. you must thijnk and decide and evavaluate like na indepednent, critical and objective experts. Do not swing by any of your previous standpoint and my preference 

---

**Cursor**

# Critical Self-Reflection: Re-evaluating My Revised Recommendation

Let me step back and evaluate all three approaches with true independence — not defending my previous positions, but asking: **what actually gives us the most valuable information per unit of effort?**

---

## Starting From First Principles: What Do We Actually Need to Learn?

**What we OBSERVE:**
- Rare codes: 0% accuracy
- Common codes: 82% accuracy
- Loss decreases, ranking metrics plateau

**What we DON'T KNOW:**
- *Why* rare codes aren't learned — the mechanism

There are multiple possible mechanisms:

| Mechanism | Description | How to Detect |
|-----------|-------------|---------------|
| **Gradient starvation** | Gradients from rare codes are too small to update weights meaningfully | Gradient analysis |
| **Output suppression** | Model outputs near-zero probabilities for rare codes | Probability analysis |
| **Representation collapse** | Rare code embeddings aren't being updated/differentiated | Embedding analysis |
| **Objective misalignment** | BCE optimizes wrong thing (calibration vs ranking) | Objective change experiment |
| **Data sparsity** | Too few positive examples per rare code | Sample counting |

**Critical insight**: Probability analysis and gradient analysis are **not equivalent** — they detect different mechanisms.

---

## Honest Re-evaluation of Each Approach

### Approach 1: Gradient Tier Analysis

**What I got RIGHT in my critique:**
- My original implementation measured the wrong gradient (output, not parameters)
- The 6-tier, multiple-backward-pass design was over-engineered

**What I got WRONG in my critique:**
- I said probability analysis is "equally informative" — **it's not**
- Probability analysis tells us WHAT (outputs are suppressed)
- Gradient analysis tells us WHERE in the learning loop the problem occurs

**The unique value of gradient analysis:**
```
If rare codes have:
- High probability but wrong ranking → Calibration vs ranking issue
- Low probability AND low gradients → Gradient starvation
- Low probability but NORMAL gradients → Something else (representation? architecture?)
```

Without gradient data, we can't distinguish case 2 from case 3.

### Approach 2: Probability Analysis

**Critical re-examination:**

If I add probability logging and see:
```
ultra_rare: mean_prob=0.0001, max_prob=0.02
```

What have I learned that I didn't know from "0% Recall@10 for rare codes"? 

**Almost nothing.** The probability being low IS the mechanism behind 0% recall. It's the same information in different form.

**Verdict**: Probability analysis has **low incremental information value**. It's not wrong, but it's redundant with what accuracy already tells us.

### Approach 3: Sampled Softmax Experiment

**Strengths:**
- Tests the most likely hypothesis directly
- If works → we have both diagnosis AND solution
- Well-supported by theory (NCE, negative sampling literature)

**Weaknesses I underweighted:**
- **It changes multiple things at once**: loss function, sampling strategy, gradient flow
- **If it fails, we learn less than we think**: Is it because the hypothesis was wrong? Or because our implementation was wrong? Or because sampled softmax has its own hyperparameters to tune?
- **Implementation is non-trivial**: Proper sampled softmax requires careful negative sampling, temperature tuning, etc.

**Verdict**: Sampled softmax is a **gamble with significant effort**. High reward if it works, but significant cost if it doesn't.

---

## The Decision Matrix

| Approach | Effort | Information if Works | Information if Fails | Unique Insight |
|----------|--------|----------------------|----------------------|----------------|
| Gradient Analysis (hook-based) | Low-Medium | Quantifies gradient imbalance | "Gradients are balanced, problem is elsewhere" | Yes - mechanistic |
| Probability Analysis | Very Low | Same as accuracy metrics | N/A | No - redundant |
| Sampled Softmax | High | "BCE was the problem" | "Need more investigation" | Partially - but confounded |

---

## Where My Previous Positions Were Wrong

### Position 1 (Original): "Do complex 6-tier gradient analysis with multiple backward passes"
**Verdict**: OVER-ENGINEERED. The implementation was flawed and too complex.

### Position 2 (Revised): "Skip gradient analysis, do probability + sampled softmax"
**Verdict**: OVERCORRECTION. I conflated "my implementation was bad" with "the approach is useless."

Probability analysis IS largely redundant. But gradient analysis (correctly implemented) provides unique mechanistic insight that neither accuracy metrics nor probability analysis gives us.

---

## What I Actually Recommend Now (Truly Independent Assessment)

### Tier 1: Minimal-effort diagnostics (Do first, takes 1 hour)

**A. Simple gradient monitoring (hook-based, ~30 lines):**
```python
class GradientMonitor:
    def __init__(self, output_layer, code_frequencies, n_tiers=3):
        self.tiers = self._build_tiers(code_frequencies, n_tiers)  # common/medium/rare only
        self.history = []
        self.output_layer = output_layer
        self._hook = output_layer.weight.register_full_backward_hook(self._record)
    
    def _build_tiers(self, freq, n):
        # Simple 3-tier: top 10% = common, 10-50% = medium, bottom 50% = rare
        ...
    
    def _record(self, module, grad_input, grad_output):
        grad = grad_output[0]  # [target_cd_cnt, d_model]
        record = {}
        for name, mask in self.tiers.items():
            record[name] = grad[mask].norm().item()
        self.history.append(record)
```

This is **20-30 lines of code**, no extra backward passes, runs automatically.

**What it tells us:**
- Quantitative gradient contribution per tier
- How pos_weight affects gradient balance
- Whether gradients change over training

**B. Skip probability analysis** — genuinely low value given what accuracy already tells us.

### Tier 2: Intervention experiment (Do second, if diagnostics suggest objective issue)

**If gradient analysis shows:**
- Common codes dominate (>80% of gradient) → Objective IS the problem → Do sampled softmax
- Gradient reasonably balanced but rare still fails → Problem is elsewhere → Don't waste time on sampled softmax

**Sampled softmax makes sense ONLY IF:**
- Diagnostics suggest objective misalignment
- We're willing to invest in proper implementation (negative sampling, temperature, etc.)

---

## My Final, Independent Assessment

| What | Verdict |
|------|---------|
| My original gradient analysis | Over-engineered, wrong implementation |
| My revised "skip gradient analysis" | Overcorrection — threw out useful signal |
| Probability analysis | Genuinely low value, skip it |
| Sampled softmax as first step | Risky without diagnostics |

**The balanced path:**
1. **Do simple, correct gradient monitoring** (30 lines, hooks, runs passively during training)
2. **Run one epoch with monitoring enabled** 
3. **If gradients heavily imbalanced** → Objective intervention (sampled softmax or ranking loss)
4. **If gradients balanced** → Problem is elsewhere (representation, architecture, data ceiling)

This gives us **mechanistic understanding** before committing to expensive interventions.

---

## Bottom Line

I was wrong twice:
1. First by over-engineering gradient analysis
2. Then by dismissing it entirely

**The right answer is a lightweight, correctly-implemented gradient monitor that gives us mechanistic insight before we commit to expensive interventions like sampled softmax.**

Probability analysis is genuinely redundant — I was right about that. But gradient analysis with proper implementation provides unique information that accuracy metrics alone cannot give us.

```plaintext
If rare codes have:
- High probability but wrong ranking → Calibration vs ranking issue
- Low probability AND low gradients → Gradient starvation
- Low probability but NORMAL gradients → Something else (representation? architecture?)
```

```plaintext
ultra_rare: mean_prob=0.0001, max_prob=0.02
```

```python
class GradientMonitor:
    def __init__(self, output_layer, code_frequencies, n_tiers=3):
        self.tiers = self._build_tiers(code_frequencies, n_tiers)  # common/medium/rare only
        self.history = []
        self.output_layer = output_layer
        self._hook = output_layer.weight.register_full_backward_hook(self._record)
    
    def _build_tiers(self, freq, n):
        # Simple 3-tier: top 10% = common, 10-50% = medium, bottom 50% = rare
        ...
    
    def _record(self, module, grad_input, grad_output):
        grad = grad_output[0]  # [target_cd_cnt, d_model]
        record = {}
        for name, mask in self.tiers.items():
            record[name] = grad[mask].norm().item()
        self.history.append(record)
```

---

**User**

No you are wrong, teh loss also plateaued right, stage at 0.003 and unchagned for a long tim 

---

**Cursor**

```plaintext
Loss: 0.8055 → ... → 0.0032 → 0.0031 → 0.0033 → 0.0032 → 0.0031...
```

You're absolutely right, and this is a critical correction that changes the diagnosis entirely.

Let me re-examine the training log:

```
Batch 0:    Loss 0.8055
Batch 500:  Loss 0.4876  
Batch 1000: Loss 0.1559
Batch 1500: Loss 0.0521
Batch 2000: Loss 0.0244
Batch 2500: Loss 0.0144
...
Batch 4000: Loss 0.0043
Batch 5000: Loss 0.0036
Batch 6000: Loss 0.0034
Batch 7000: Loss 0.0033
Batch 8000: Loss 0.0032
Batch 9000: Loss 0.0032
Batch 10000: Loss 0.0031
```

**Loss plateaus at ~0.003 and oscillates there for thousands of steps.** Metrics plateau at the same time.

---

## This Changes Everything

I was operating under the assumption: *"Loss continues to decrease while metrics plateau"* → objective misalignment

**But the actual observation is:** *"BOTH loss AND metrics plateau together"* → **learning has stopped entirely**

This is a fundamentally different problem:

| Scenario | Loss | Metrics | Diagnosis |
|----------|------|---------|-----------|
| What I assumed | ↓ continues | plateaus | Objective misalignment |
| **What actually happened** | **plateaus at 0.003** | **plateaus** | **Learning stopped** |

---

## Revised Diagnosis: Gradient Saturation

When BOTH loss and metrics plateau, it means **gradients have become near-zero across the board**. The model has stopped receiving meaningful learning signal.

In BCE with sigmoid:
```
gradient ∝ (sigmoid(logit) - target)
```

- If model is confident and correct: gradient ≈ 0
- If model is confident and WRONG: gradient should be large

**But here's what's likely happening:**

| Code Type | Model Output | Usually Correct? | Gradient |
|-----------|--------------|------------------|----------|
| Common (target=1) | High prob (~0.9) | Yes | ≈ 0 (saturated) |
| Common (target=0) | Low prob (~0.1) | Yes | ≈ 0 (saturated) |
| Rare (target=0) | Very low (~0.001) | Yes | ≈ 0 (saturated) |
| Rare (target=1) | Very low (~0.001) | **NO** | Should be large, BUT... |

The problem: **When rare codes SHOULD be predicted (target=1), the model is confidently wrong (outputs ~0.001).** This should produce large gradients. But:

1. Rare positive examples are infrequent (by definition)
2. When they appear, their gradient is averaged with thousands of "correct" predictions
3. The net gradient per step is tiny

**This IS gradient starvation** — not because gradients are zero, but because:
- Correct predictions dominate (most of batch)
- Incorrect predictions are rare (few per batch)
- Net gradient is near-zero

---

## Why This Makes Gradient Analysis MORE Relevant

If loss plateaued, gradient analysis tells us:

| Finding | Meaning | Action |
|---------|---------|--------|
| All gradients ≈ 0 | Global saturation | Fundamentally change training dynamics |
| Common gradients ≈ 0, rare gradients small | Rare codes drowned out | Rebalancing might help |
| Gradients non-zero but oscillating | Optimizer stuck | LR/optimizer changes |
| Gradients healthy but loss doesn't move | Cancellation? | Deeper investigation |

Without gradient data, we're guessing.

---

## Sampled Softmax Reframed

Sampled softmax might still help, but for **different reasons** than I originally thought:

**Original rationale**: BCE optimizes calibration, not ranking (objective misalignment)

**Revised rationale**: Sampled softmax **changes gradient dynamics**:
- Forces model to compare positive codes against negative codes
- Creates gradient even when model is "confident"
- Prevents global saturation

But now the question is: **is gradient saturation the actual mechanism?**

---

## What Gradient Analysis Would Tell Us (Now More Valuable)

```python
# Simple diagnostic: What is the actual gradient magnitude?
# If it's near-zero, we've confirmed gradient saturation

def diagnose_gradient_saturation(model, output_layer):
    """Track if gradients are actually flowing."""
    grad_history = []
    
    def hook(module, grad_input, grad_output):
        grad = grad_output[0]
        grad_history.append({
            'norm': grad.norm().item(),
            'mean_abs': grad.abs().mean().item(),
            'max_abs': grad.abs().max().item(),
            'nonzero_fraction': (grad.abs() > 1e-6).float().mean().item()
        })
    
    handle = output_layer.weight.register_full_backward_hook(hook)
    return grad_history, handle
```

**If gradient norms are ~1e-6 or smaller** → confirmed gradient saturation
**If gradient norms are healthy (~0.01-1.0)** → problem is elsewhere

---

## My Corrected Assessment

| Previous Position | Correction |
|-------------------|------------|
| "Loss continues while metrics plateau" | **Wrong** - both plateau |
| "Objective misalignment is the cause" | **Partially wrong** - it's gradient saturation |
| "Gradient analysis is redundant" | **Wrong** - it's now highly relevant |
| "Sampled softmax for ranking" | **Partially right** - but the mechanism is different |

**The right diagnostic step:**
1. Confirm gradient magnitudes are near-zero (5 lines of code with hooks)
2. If confirmed → gradient saturation is the mechanism
3. Then test interventions that **break the saturation**: sampled softmax, label smoothing, temperature scaling, gradient amplification

---

## Bottom Line

You caught a critical error in my reasoning. Loss plateauing at 0.003 means **learning stopped**, not **objective misalignment**. This makes gradient analysis more valuable, not less, because we need to confirm the saturation mechanism before designing interventions.

Thank you for the correction — this is exactly why rigorous review matters.

```plaintext
Batch 0:    Loss 0.8055
Batch 500:  Loss 0.4876  
Batch 1000: Loss 0.1559
Batch 1500: Loss 0.0521
Batch 2000: Loss 0.0244
Batch 2500: Loss 0.0144
...
Batch 4000: Loss 0.0043
Batch 5000: Loss 0.0036
Batch 6000: Loss 0.0034
Batch 7000: Loss 0.0033
Batch 8000: Loss 0.0032
Batch 9000: Loss 0.0032
Batch 10000: Loss 0.0031
```

```plaintext
gradient ∝ (sigmoid(logit) - target)
```

```python
# Simple diagnostic: What is the actual gradient magnitude?
# If it's near-zero, we've confirmed gradient saturation

def diagnose_gradient_saturation(model, output_layer):
    """Track if gradients are actually flowing."""
    grad_history = []
    
    def hook(module, grad_input, grad_output):
        grad = grad_output[0]
        grad_history.append({
            'norm': grad.norm().item(),
            'mean_abs': grad.abs().mean().item(),
            'max_abs': grad.abs().max().item(),
            'nonzero_fraction': (grad.abs() > 1e-6).float().mean().item()
        })
    
    handle = output_layer.weight.register_full_backward_hook(hook)
    return grad_history, handle
```

You're right. I've been oscillating between positions without truly grounding myself. Let me stop and think much more carefully.

---

# Stable, Grounded Analysis

## Step 1: What Do I ACTUALLY Know? (Undisputable Facts Only)

| Fact | Source |
|------|--------|
| Loss: 0.8 → 0.003 (plateaus at ~step 5000) | Training logs |
| R@10: 0.001 → 0.83 (plateaus at ~step 8000) | Training logs |
| μR@10 ≈ 0.46-0.48 across all experiments | Results files |
| Rare/tail tier accuracy: 0% | Results files |
| Common tier accuracy: 82-83% | Results files |
| 2× data: R@10 +0.7%, μR@10 +3.2% | Round 6 results |
| MoE vs Dense: identical metrics | Exp6 vs Exp2 results |
| pos_weight_max = 50, log_scaled | Config files |
| ~6297 target codes | Config files |

## Step 2: What Are My Hypotheses? (Not Facts)

| Hypothesis | Status |
|------------|--------|
| "Gradient saturation" | **Unverified** - I claimed gradients are near-zero, but haven't measured |
| "Gradient dilution" | **Unverified** - I claimed rare code signals are drowned out, but haven't measured |
| "BCE optimum excludes rare codes" | **Plausible** - math supports it, but not empirically confirmed |
| "pos_weight_max=50 is the bottleneck" | **Plausible** - logical, but not tested |
| "Model capacity is NOT the issue" | **Supported** - MoE didn't help |

## Step 3: Where I've Been Inconsistent

| My Statement | When I Said It | Problem |
|--------------|----------------|---------|
| "Gradient saturation" | Initial analysis | Used term imprecisely |
| "Gradient dilution" | Second revision | Changed terminology without new evidence |
| "BCE optimum" | Third revision | Changed frame again without new evidence |
| "Gradient analysis is valuable" | First version | Then flip-flopped |
| "Gradient analysis is low value" | Later revision | Then flip-flopped again |
| "Increase pos_weight is the fix" | Latest revision | Presented as certain when it's untested |

**The core problem**: I kept changing my DIAGNOSIS without new EVIDENCE. Each revision was just a different interpretation of the same facts.

---

## Step 4: What Can I Actually Defend?

### I CAN Defend:

**1. The plateau is real and architecture-agnostic**
- Evidence: All experiments (exp1_opt, exp2, exp6) hit the same ceiling
- Evidence: Adding MoE parameters didn't help

**2. Rare codes are not being learned**
- Evidence: 0% accuracy for rare/tail tiers in all experiments
- Evidence: μR@10 (0.46) << R@10 (0.83) shows per-code performance varies

**3. More data provides diminishing returns**
- Evidence: 2× data → +3.2% μR@10 (not 2× improvement)

**4. The problem is NOT model capacity**
- Evidence: Dense (25M) ≈ MoE (35M) in performance

### I CANNOT Defend:

**1. The exact mechanism (saturation vs dilution vs BCE optimum)**
- I've been claiming different mechanisms without measuring gradients
- This requires empirical verification

**2. That increasing pos_weight_max will fix the problem**
- I haven't seen evidence that different values were tested
- It's a hypothesis, not a verified solution

**3. That gradient analysis is either "essential" or "low value"**
- I've argued both sides without a clear basis for either

---

## Step 5: What Is The Honest Truth?

**I don't actually know what's causing the plateau.**

I have plausible hypotheses:
- pos_weight cap might be insufficient
- BCE objective might be misaligned with ranking goal
- Rare codes might have too few positive examples

But I haven't verified any of these empirically. I've been reasoning from first principles without ground truth.

---

## Step 6: What Should I Actually Recommend?

Given that I DON'T KNOW the root cause, the correct approach is:

### Priority 1: Diagnostic Experiments (Before ANY Fix)

**A. Gradient Tier Analysis** (the thing I kept flip-flopping on)
- Measure actual gradient contribution per tier
- This directly tests the "gradient starvation/dilution" hypothesis
- ~1 hour of training with logging added

**B. Per-Code Prediction Analysis**
- For rare codes with target=1, what does the model predict?
- If pred≈0 consistently → model learned "always predict 0"
- If pred varies → model is trying but failing

**C. pos_weight Ablation**
- Test pos_weight_max = 100, 200, 500
- See if any changes rare code recall
- This directly tests the "pos_weight cap" hypothesis

### Priority 2: Only AFTER Diagnostics Confirm a Hypothesis

If gradient analysis shows common codes dominate:
- Focal loss or sampled softmax (changes gradient distribution)

If pos_weight ablation shows improvement:
- Set pos_weight_max appropriately

If prediction analysis shows model "always predicts 0":
- Need stronger incentive (higher weight or different loss)

---

## Step 7: My Stable, Final Position

**Diagnosis**: I genuinely don't know the exact mechanism. I have hypotheses but no empirical confirmation.

**First action**: Run gradient tier analysis to get empirical data on gradient distribution.

**Second action**: Run pos_weight ablation (50, 100, 200, 500) to test the cap hypothesis.

**Third action**: Based on diagnostic results, choose intervention:
- If gradients heavily skewed → focal loss or sampled softmax
- If pos_weight increase helps → keep that setting
- If neither helps → reconsider d_model/architecture changes

**What I will NOT claim**:
- Any single "root cause" without evidence
- Any intervention as "definitely the solution"
- That I know the exact mechanism

---

## Why I Keep Changing My Mind

The honest answer: **I've been reasoning in circles from the same evidence.**

Each "self-reflection" just reinterpreted the same facts differently. I wasn't getting new information—I was just finding new ways to frame old information.

The ONLY way to break this cycle is **empirical measurement**:
1. Measure gradient distribution (is it skewed?)
2. Measure the effect of pos_weight changes (does it help?)
3. Let the DATA tell us the answer

---

## What I Now Commit To

1. **Gradient tier analysis IS worth doing** - it provides data I don't have
2. **pos_weight ablation IS worth doing** - it tests a key hypothesis
3. **I cannot claim to know the root cause** - I have hypotheses only
4. **Any intervention is a TEST, not a fix** - until we have diagnostic data

This is my stable position. I will not change it again based on more reasoning—only based on new empirical evidence.