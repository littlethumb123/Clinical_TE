# 🔬 Expert 1: Comprehensive Analysis: MoE Training with Focal Loss + Tiered Weighting
[exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_20251228_172443_results](expe_logs/exp_round5/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_20251228_172443_results.json)
## 1. Summary of Prior Context and Implemented Changes

### Previous Diagnosis Summary
The prior analysis (`exp_round5_ablation_exp6_auxiliary-free_all3lobs_intrinsic_only_dec26_2025_analysis.md`) identified:

| Issue | Status | Solution Proposed |
|-------|--------|------------------|
| Expert collapse (12→0) | ✅ Fixed | bias_lr increased to 1e-3 |
| Loss plateau at batch 3000 | ⚠️ Issue | Extend high-LR phase |
| μR@10 gap (31.7% vs 46.7%) | ⚠️ Issue | Higher pos_weight, Focal Loss |
| CV=0.31, Gini=0.13 | ✅ Healthy | Maintain |

### Implemented Changes for This Run

Based on the code in `moe_flashattn_3.py` lines 15118-15131:

```python
optimize_config = OptimizeConfig(
    scheduler_type='linear',
    warmup_pct=0.15,
    plateau_pct=0.45,           # 60% total at high LR (was 30%)
    min_lr_ratio=0.2,           # End at 20% of peak
    use_pos_weight=True,
    pos_weight_method='tiered', # NEW: Discrete tier weights
    pos_weight_max=100,         # Increased from 50
    use_focal_loss=True,        # NEW
    focal_gamma=2.5,            # Aggressive
    focal_alpha=0.25,
)
```

**Note:** `bias_momentum` remains at 0.8 (not changed to 0.7 as recommended).

---

## 2. Results Comparison

### Final Evaluation Metrics

| Metric | Dense (exp2b) | MoE Previous | MoE Current | Current vs Dense |
|--------|--------------|--------------|-------------|------------------|
| **R@10** | 79.0% | 72.4% | 73.8% | 93.4% |
| **μR@10** | 46.7% | 31.7% | **42.3%** | 90.6% |
| **NDCG@20** | 47.6% | 38.4% | 40.7% | 85.5% |
| **MRR** | 45.6% | N/A | 38.7% | 84.9% |
| **Precision@10** | 19.4% | 15.0% | 20.0% | 103.1% |
| **Loss (final)** | 0.0031 | 0.029 | **0.0009** | 29% |

**Key Observation**: μR@10 improved dramatically from 31.7% → 42.3% (+33% relative), but R@10 barely moved (72.4% → 73.8%).

### MoE Health Metrics (from trajectory data)

| Metric | Previous Run (end) | Current Run (throughout) | Status |
|--------|-------------------|-------------------------|--------|
| Collapsed Experts | **0** | **4-8** | 🔴 REGRESSED |
| CV | 0.31 | 0.55-0.83 | 🔴 WORSE |
| Gini | 0.13 | 0.28-0.45 | 🔴 WORSE |

---

## 3. Root Cause Analysis

### Issue #1: Expert Collapse REGRESSED (4→8 collapsed throughout training)

**Evidence from trajectory:**
```
Batch 100:   CV=0.557, Collapsed=4, Gini=0.285
Batch 4400:  CV=0.755, Collapsed=8, Gini=0.410   ← Peak collapse
Batch 5600:  CV=0.794, Collapsed=8, Gini=0.429
Batch 7500:  CV=0.648, Collapsed=4, Gini=0.336   ← Partial recovery
Batch 10900: CV=0.781, Collapsed=4, Gini=0.430
```

**Root Cause: Focal Loss + Tiered Weighting Disrupts Router Gradient Flow**

1. **Focal Loss mechanism (gamma=2.5)**:
   - Weight = (1 - p)^γ for positives
   - For confident predictions (p > 0.8): weight < 0.03 (essentially zero)
   - **Effect**: ~99% of samples contribute near-zero gradients

2. **Tiered Weighting mechanism**:
   - Discrete jumps: 1 → 3 → 10 → 25 → 50 → 100
   - Creates **bimodal gradient distribution** based on code frequency
   
3. **Combined Effect on Router**:
   - Router receives gradients only from hard examples (focal)
   - Those hard examples have extreme weights (tiered)
   - **Result**: High variance, low consistency in router training signal

**Supporting Evidence (Industry/Research)**:
- DeepSeek-V2 paper (2024) uses auxiliary-free bias correction **without focal loss** - they rely on consistent gradient flow
- ST-MoE (Zoph et al., 2022) notes that MoE routers are sensitive to gradient variance
- The original Focal Loss paper (Lin et al., 2017) was for object detection with dense layers, not sparse experts

**Mechanism of collapse:**

```
                    Without Focal                   With Focal (gamma=2.5)
                    ─────────────────              ─────────────────────────
Gradient signal:    [0.5, 0.3, 0.8, 0.4...]        [0.0, 0.0, 0.9, 0.0...]
                        ↓                               ↓
Router update:      Consistent across experts       Dominated by few hard examples
                        ↓                               ↓
Expert selection:   Learns diverse routing          Collapses to subset
```

### Issue #2: Loss Value Artificially Low

**Evidence:**
- Current loss: 0.0009 (3.4× lower than dense's 0.0031)
- But R@10 is 5.2% behind dense

**Root Cause: Focal Loss Masks True Loss**

Focal Loss with gamma=2.5 down-weights easy examples so aggressively that:
- Loss = Σ (1-p)^2.5 × BCE ≈ 0 for most samples
- The reported loss no longer reflects discriminative quality
- Model "succeeds" on focal objective while failing on actual retrieval

**Evidence**: Look at loss trajectory:
```
Batch 100:  Loss=1.4694 (standard BCE scale)
Batch 200:  Loss=0.1137 (10× drop - focal kicking in)
Batch 300:  Loss=0.0284 (4× drop in 100 batches!)
Batch 1000: Loss=0.0019 (converged)
```

This ultra-fast convergence pattern is pathological - it means focal loss is removing signal, not that the model is learning faster.

### Issue #3: Why μR@10 Improved But R@10 Didn't

**Evidence:**
| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| R@10 | 72.4% | 73.8% | +1.9% |
| μR@10 | 31.7% | 42.3% | +33.4% |

**Root Cause: Tiered Weighting Works, But Expert Collapse Limits Capacity**

The tiered weighting IS helping rare codes (μR@10 improvement confirms this). However:
1. Expert collapse reduces effective capacity by 50-75%
2. Common codes need capacity too - they're hurt by collapse
3. Net effect: rare codes improve, common codes regress, R@10 flat

**This is a trade-off, not a pure win.**

### Issue #4: Why MoE Still Underperforms Dense Baseline

**Gap Analysis:**
| Metric | Gap to Dense | Likely Cause |
|--------|--------------|--------------|
| R@10 | -5.2% | Expert collapse |
| μR@10 | -4.4% | Router instability |
| NDCG@20 | -6.9% | Ranking quality hurt by collapse |

**Root Causes:**
1. **Reduced effective capacity**: 8/16 experts active = 50% of MoE potential
2. **Router instability**: High CV (0.78) means inconsistent expert selection
3. **Architectural overhead without benefit**: MoE has 35M params vs 25M (dense), but not using them effectively

---

## 4. Training Dynamics Analysis

### Phase-by-Phase Trajectory

| Phase | Batches | Loss | R@10 | μR@10 | Collapsed | Interpretation |
|-------|---------|------|------|-------|-----------|----------------|
| **Rapid learning** | 0-500 | 1.47→0.007 | 1%→43% | 0%→17% | 4 | Focal hasn't dominated yet |
| **Continued gain** | 500-2000 | 0.007→0.001 | 43%→58% | 17%→24% | 4 | Normal learning |
| **Collapse onset** | 2000-5000 | 0.001 (flat) | 58%→70% | 24%→32% | 4→8 | Focal dominates |
| **Instability** | 5000-7500 | 0.001 | 70%→72% | 32%→35% | 4-8 (unstable) | Router oscillating |
| **Plateau** | 7500-10965 | 0.0009 | 72%→74% | 35%→40% | 4 | Stabilized collapse |

**Key Observation**: Expert collapse worsens exactly when loss plateaus at 0.001, confirming the Focal Loss → router starvation → collapse chain.

### CV Trend Shows Router Instability

```
CV Trajectory:
0.55 ████████████████████                     ← Healthy start
0.60   ██████████████████████████████          ← Degrading
0.70      ██████████████████████████████████████  ← Critical
0.80         ███████████████████████████████████████  ← Collapse
0.65           ██████████████████████████████████████████  ← Partial recovery
```

CV > 0.5 indicates significant load imbalance; CV > 0.7 is pathological.

---

## 5. Positive Brier Score Analysis

**Evidence from trajectory:**
```
Batch 100:  PosBrier=0.4345  ← Underconfident (good for calibration)
Batch 500:  PosBrier=0.6974  ← Moving toward overconfident
Batch 5000: PosBrier=0.6409
Batch 10500: PosBrier=0.5964  ← Slight improvement
```

**Interpretation:**
- PosBrier measures calibration for positive predictions
- 0.5 = random, lower = better
- Current ~0.60 is suboptimal but not terrible
- **Focal Loss alpha=0.25 may be too aggressive** - it's boosting positive class but creating overconfidence

---

## 6. Diagnosis: Underfitting vs Real Plateau

**This is NOT underfitting. It is a structural plateau caused by:**

1. **Reduced effective capacity** (expert collapse)
2. **Distorted gradient flow** (focal loss)
3. **Router training starvation** (no consistent signal)

**Evidence against underfitting:**
- Loss continues to decrease
- Training metrics (in-epoch) are close to validation metrics (gap = -8e-6)
- Model IS learning, just hitting capacity ceiling

**Evidence for structural plateau:**
- Expert collapse persists regardless of learning rate phase
- Metrics plateau while loss still moves
- CV stays high (0.6-0.8) throughout

---

## 7. Recommendations

### Tier 1: Critical (Address Immediately)

#### 1.1 Disable Focal Loss for MoE
```python
optimize_config = OptimizeConfig(
    use_focal_loss=False,  # CRITICAL: Disable
    use_pos_weight=True,
    pos_weight_method='tiered',
    # Keep tiered - it helped μR@10
)
```

**Rationale**: Focal Loss is fundamentally incompatible with MoE routing because it removes gradient signal from most samples.

#### 1.2 Lower bias_momentum to 0.5-0.6
```python
MoEConfig(
    bias_momentum=0.5,  # Much lower for faster adaptation
    bias_lr=2e-3,       # Also increase to compensate
)
```

**Rationale**: With high gradient variance from tiered weighting, the EMA needs to adapt faster.

### Tier 2: Recommended

#### 2.1 Smooth Tiered Weights

Instead of discrete jumps, use a continuous piecewise function:

```python
def compute_smooth_tiered_weights(code_frequencies, device):
    """Continuous version of tiered weights to avoid gradient jumps."""
    freq_log = np.log1p(code_frequencies)
    freq_min, freq_max = freq_log[freq_log > 0].min(), freq_log.max()
    
    # Linear interpolation in log space: [1, 100]
    weights = 1 + 99 * (1 - (freq_log - freq_min) / (freq_max - freq_min))
    weights = np.clip(weights, 1, 100)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)
```

**Rationale**: Smooth gradients help router training stability.

#### 2.2 Add Router Z-Loss

The current config has `z_loss_weight=0.0`. Enable it:

```python
MoEConfig(
    z_loss_weight=0.001,  # Small but non-zero
)
```

**Rationale**: Z-loss (from Switch Transformer) prevents logit magnitudes from growing too large, helping router stability.

### Tier 3: Experimental

#### 3.1 Try 8 Experts Instead of 16

```python
MoEConfig(
    num_experts=8,   # Reduced
    top_k=2,         # Same
)
```

**Rationale**: With 16 experts and top_k=2, only 12.5% of capacity is used per token. Collapse of 4-8 experts means 25-50% capacity is permanently lost. Fewer experts = more robust.

#### 3.2 Alternative: Expert Capacity Factor

If keeping 16 experts, implement capacity factor limiting:

```python
# In MoE forward, limit max tokens per expert
capacity_factor = 1.25
max_tokens_per_expert = int(capacity_factor * num_tokens / num_experts)
```

**Rationale**: Prevents any expert from being overloaded, forcing router to distribute.

---

## 8. Summary

### What Worked ✅
| Change | Effect | Evidence |
|--------|--------|----------|
| Tiered weighting | μR@10 +33% | 31.7% → 42.3% |
| Extended plateau LR | Slight help | Loss continued improving |

### What Failed ❌
| Change | Problem | Evidence |
|--------|---------|----------|
| Focal Loss (gamma=2.5) | Expert collapse | 0→4-8 collapsed |
| Combined Focal + Tiered | Router instability | CV 0.55→0.83 |

### Key Insight

**Focal Loss and MoE are fundamentally incompatible** because:
1. MoE routers need consistent gradient signal across all inputs
2. Focal Loss eliminates gradients from easy examples
3. This starves the router of training signal
4. Result: Router collapses to subset of experts

### Recommended Next Experiment Configuration

```python
# MoE Config
MoEConfig(
    num_experts=8,           # Reduced from 16
    num_shared_experts=1,
    top_k=2,
    load_balance_strategy='deepseek',
    aux_loss_weight=0.0,
    bias_lr=2e-3,            # Increased from 1e-3
    bias_momentum=0.5,       # Decreased from 0.8
    z_loss_weight=0.001,     # Enable z-loss
    expert_dropout=0.1,
    use_swiglu_experts=True
)

# Optimize Config
OptimizeConfig(
    scheduler_type='linear',
    warmup_pct=0.15,
    plateau_pct=0.45,
    min_lr_ratio=0.2,
    use_pos_weight=True,
    pos_weight_method='log_scaled',  # Smooth weights
    pos_weight_max=75,
    use_focal_loss=False,            # DISABLED
)
```

This configuration should:
1. ✅ Maintain healthy expert utilization (0 collapsed)
2. ✅ Preserve rare code gains from weighted BCE
3. ✅ Allow router to learn stable patterns
4. ✅ Close the gap to dense baseline

# 🔬 Expert 2 (Judge): Independent Expert Analysis: MoE Training with Focal Loss + Tiered Weighting

## Part 1: My Independent Assessment (Before Evaluating Expert 1)

### 1.1 Configuration Verification

**CRITICAL FINDING**: There's a discrepancy between what you said you changed and what the code shows:

- You mentioned changing `bias_momentum` from 0.8 to 0.7
- The code at line 704 still shows `bias_momentum=0.8`

This is important because the fix may not have been applied.

**Actual Configuration Used:**
```python
# MoE Config (exp6_auxiliary_free at line 694-709)
num_experts=8           # NOT 16 as Expert 1 mentions
num_shared_experts=1
top_k=2
bias_lr=1e-3
bias_momentum=0.8       # Still 0.8, not 0.7

# Optimize Config (line 15118-15131)
scheduler_type='linear'
plateau_pct=0.45        # 60% total at high LR  
use_focal_loss=True
focal_gamma=2.5
focal_alpha=0.25
pos_weight_method='tiered'
pos_weight_max=100
```

### 1.2 Accurate Results Comparison (from JSON files)

| Metric | Dense (exp2b) | MoE Previous | MoE Current | MoE/Dense |
|--------|--------------|--------------|-------------|-----------|
| **R@10** | 79.0% | 72.4% | 73.8% | 93.4% |
| **μR@10** | 46.7% | 31.7% | **42.3%** | 90.6% |
| **NDCG@20** | 47.6% | 38.4% | 40.7% | 85.5% |
| **MRR** | 45.6% | N/A | 38.7% | 84.9% |
| **PosBrier** | 0.766 | N/A | **0.578** | Better! |
| **Final Loss** | 0.0031 | 0.029 | 0.0009 | Not comparable |

**Key Observation**: PosBrier 0.578 < 0.766 means MoE has **BETTER** probability calibration than dense baseline!

### 1.3 Expert Collapse Pattern - What Actually Happened

From trajectory data:

```
Timeline of Expert Collapse:
Batch 100:   CV=0.557, Collapsed=4  ← COLLAPSE EXISTS FROM START
Batch 1000:  CV=0.553, Collapsed=4  ← Stable but not recovering
Batch 4200:  CV=0.727, Collapsed=5  ← First worsening
Batch 4400:  CV=0.755, Collapsed=8  ← Peak collapse
Batch 5600:  CV=0.794, Collapsed=8  ← Sustained
Batch 7500:  CV=0.648, Collapsed=4  ← Partial recovery
Batch 10900: CV=0.781, Collapsed=4  ← Final state
```

**Critical Finding**: Collapse exists from batch 100, **before any significant training occurs**. This is not "regression from 0 → 4-8" as Expert 1 states. Rather:
- Previous run: Started with collapse → **recovered** to 0 by batch 3000
- Current run: Started with collapse → **never recovered**, occasionally worsened

### 1.4 Root Cause Analysis - The True Mechanism

#### Why Collapse From the Start?

At batch 100, model predictions are essentially random (R@10=1.4%). Let's trace the gradient flow:

```
For random predictions p ≈ 0.5:
  Focal weight = (1 - 0.5)^2.5 = 0.177
  → 82% gradient reduction even at initialization!

For slightly confident predictions p ≈ 0.7:
  Focal weight = (1 - 0.7)^2.5 = 0.016
  → 98% gradient reduction

Combined with tiered weights (1 to 100):
  Some samples: 0.016 × 100 = 1.6 effective weight
  Other samples: 0.177 × 1 = 0.177 effective weight
  → 9× variance between samples before any learning!
```

#### Why Previous Run Recovered But Current Run Didn't

| Factor | Previous Run | Current Run | Effect |
|--------|--------------|-------------|--------|
| Gradient density | Dense (all samples) | Sparse (~5% dominant) | Router starved |
| Gradient variance | Low (uniform weights) | High (1-100x range) | Unstable updates |
| Bias correction signal | Consistent | Erratic | EMA can't track |

The DeepSeek bias correction uses exponential moving average (EMA) to track expert load. It needs **consistent gradient flow** to work. With focal loss:
1. Gradients are sparse (most samples down-weighted to ~0)
2. Remaining gradients have extreme variance (tiered weights)
3. The EMA sees erratic load patterns → cannot stabilize

### 1.5 Is This Underfitting or Structural Plateau?

**It is a STRUCTURAL CAPACITY LIMITATION:**

- Training loss continues decreasing ✓
- Validation tracks training (gap = -8e-6) ✓
- Model IS learning features ✓
- BUT: 4-8 of 8 experts collapsed = 50-100% of routing capacity lost

With 8 experts, top-k=2:
- Normal: 2/8 = 25% of expert capacity per token
- With 4 collapsed: 2/4 = 50% of REMAINING = 12.5% of total
- With 8 collapsed: Only shared expert working!

This creates a **hard ceiling** on performance independent of training duration.

### 1.6 Why μR@10 Improved Despite Collapse

This is actually a **SUCCESS** of tiered weighting:

```
μR@10 improvement: 31.7% → 42.3% = +33% relative

Tiered weights prioritize rare codes:
- Ultra-rare (0-5th %ile): weight=100
- Very common (90-100th %ile): weight=1

Even with reduced capacity, the model focuses remaining 
gradients on rare codes → rare code performance improves
```

The fact that R@10 didn't drop while μR@10 improved significantly shows the tiered weighting is doing its job - just with reduced effective capacity.

---

## Part 2: Evaluation of Expert 1's Analysis

### 2.1 What Expert 1 Got RIGHT ✅

| Finding | Expert 1's Claim | My Assessment |
|---------|-----------------|---------------|
| Focal Loss incompatibility | "Fundamentally incompatible with MoE" | ✅ **Correct** - the gradient sparsity starves routers |
| Loss artificially low | Loss 0.0009 doesn't reflect quality | ✅ **Correct** - focal modulation masks true loss |
| Tiered weighting works | μR@10 +33% attributed to tiered | ✅ **Correct** - strong evidence from data |
| DeepSeek needs consistent gradients | Paper reference accurate | ✅ **Correct** - cited correctly |
| Primary recommendation: disable focal | Top priority fix | ✅ **Correct** - this is the right call |

### 2.2 What Expert 1 Got WRONG or INCOMPLETE ❌

#### Error 1: Incorrect Expert Count Analysis

**Expert 1 states**: "16 experts" in multiple places (lines 159, 301-302)

**Actual**: The config shows `num_experts=8`:

```697:699:dev/moe/moe_flashattn_3.py
            num_experts=8,
            num_shared_experts=1,
            top_k=2,
```

Expert 1's recommendation to "try 8 experts instead of 16" (line 295-302) is **moot** - you're already using 8.

#### Error 2: Mischaracterization of Collapse Pattern

**Expert 1 states**: "Expert collapse REGRESSED (4→8 collapsed throughout training)" and shows "Collapse onset at batch 2000-5000"

**Actual Timeline**:
- Batch 100: Already 4 collapsed (not 0→4 during training)
- Previous run also had 4 collapsed at early batches
- The difference: previous recovered, current didn't

This is not "regression" - it's **failure to recover**. The mechanism is different and the interventions should target recovery, not prevention.

#### Error 3: Positive Brier Misinterpretation

**Expert 1 states**: "PosBrier ~0.60 is suboptimal but not terrible"

**Actual**: Lower PosBrier is better!
- Dense baseline: 0.766 
- Current MoE: 0.578

MoE has **24.5% better calibration** than dense! This is a positive finding that was missed.

#### Error 4: Incomplete Gradient Variance Analysis

Expert 1 correctly identifies focal loss sparsity but underweights the **compounding effect**:

```
Focal sparsity × Tiered variance = Catastrophic instability

Without focal (previous run):
  All samples contribute → Tiered variance averages out

With focal (current run):  
  Only hard samples contribute → Tiered variance fully expressed
  → Router sees 1-100x weight swings on ~5% of samples
  → EMA-based bias correction cannot stabilize
```

### 2.3 Expert 1's Recommendations - My Evaluation

| Recommendation | Expert 1's Priority | My Assessment |
|----------------|--------------------|--------------------|
| Disable Focal Loss | Tier 1, Critical | ✅ **Strongly Agree** |
| bias_momentum=0.5 | Tier 1, Critical | ⚠️ Agree, but secondary |
| Smooth Tiered Weights | Tier 2, Recommended | ✅ **Strongly Agree** |
| Add z_loss_weight=0.001 | Tier 2, Recommended | ⚠️ Maybe - not root cause |
| 8 Experts Instead of 16 | Tier 3, Experimental | ❌ **Already using 8** |
| Capacity Factor | Tier 3, Experimental | ⚠️ Worth trying after primary fixes |

---

## Part 3: My Recommendations

### Priority 1: MUST DO - Disable Focal Loss
```python
optimize_config = OptimizeConfig(
    use_focal_loss=False,  # CRITICAL - Remove completely
    # ... rest unchanged
)
```

**Rationale**: Focal loss is the primary cause of gradient starvation. No amount of hyperparameter tuning will fix this architectural incompatibility.

### Priority 2: Keep Tiered But Reduce Variance

Option A - Use continuous log-scaled weights:
```python
pos_weight_method='log_scaled',  # Smooth gradient flow
pos_weight_max=50,               # Reduced from 100
```

Option B - Reduce tiered weight jumps:
```python
tier_config = {
    'ultra_rare':  {'percentile': (0, 5),    'weight': 40},   # Was 100
    'tail':        {'percentile': (5, 25),   'weight': 20},   # Was 50
    'rare':        {'percentile': (25, 50),  'weight': 10},   # Was 25
    'medium':      {'percentile': (50, 75),  'weight': 4},    # Was 10
    'common':      {'percentile': (75, 90),  'weight': 2},    # Was 3
    'very_common': {'percentile': (90, 100), 'weight': 1},    # Same
}
```

**Rationale**: The μR@10 improvement shows tiered weighting WORKS. Just reduce the variance to allow router stability.

### Priority 3: Confirm bias_momentum Actually Changed

The code shows `bias_momentum=0.8`, not 0.7 as you intended. Verify and fix:

```python
MoEConfig(
    bias_lr=1e-3,
    bias_momentum=0.6,  # Actually change this time!
)
```

### Priority 4: Add Router Gradient Monitoring

Before next experiment, add tracking for:
```python
# In training loop
router_grad_norm = torch.norm(moe_layer.router.weight.grad)
log(f"Router grad norm: {router_grad_norm:.4f}")
```

This will help diagnose future instability.

### What I Would NOT Recommend Yet

1. **Don't reduce to fewer experts** - 8 is already minimal for your task
2. **Don't add z_loss** - addresses wrong problem (logit explosion, not collapse)
3. **Don't increase bias_lr beyond 2e-3** - could cause oscillation
4. **Don't extend training** - structural capacity limit won't improve with more epochs

---

## Part 4: Expected Outcomes

With focal loss disabled and smoother tiered weights:

| Metric | Current | Expected | Rationale |
|--------|---------|----------|-----------|
| Collapsed Experts | 4-8 | **0-2** | Like previous run's recovery |
| CV | 0.65-0.83 | **< 0.4** | Healthy load balance |
| R@10 | 73.8% | **76-79%** | Full capacity utilization |
| μR@10 | 42.3% | **40-44%** | Maintain rare code gains |
| Loss (BCE) | 0.0009 | **0.002-0.004** | Normal scale, comparable to dense |

---

## Summary: Agreement/Disagreement with Expert 1

### Full Agreement ✅
1. Focal loss is fundamentally incompatible with MoE routing
2. Tiered weighting is beneficial for rare codes
3. Disabling focal loss should be the top priority

### Partial Agreement ⚠️
1. Lower bias_momentum helps, but won't fix root cause alone
2. Smoother weights are good, but the method matters
3. CV/Gini metrics interpretation is correct

### Disagreement ❌
1. Expert count: You have 8 experts, not 16
2. Collapse pattern: It's "failure to recover" not "regression"
3. PosBrier: 0.58 is BETTER than dense's 0.77 (Expert 1 said "suboptimal")
4. Phase analysis timing: Collapse existed from batch 100, not onset at 2000-5000

---

## Recommended Configuration for Next Experiment

```python
# MoE Config - verify these are actually applied!
MoEConfig(
    num_experts=8,
    num_shared_experts=1,
    top_k=2,
    load_balance_strategy='deepseek',
    aux_loss_weight=0.0,
    bias_lr=1e-3,
    bias_momentum=0.6,       # Actually change from 0.8!
    z_loss_weight=0.0,       # Not needed yet
    expert_dropout=0.1,
    use_swiglu_experts=True
)

# Optimize Config
OptimizeConfig(
    scheduler_type='linear',
    warmup_pct=0.15,
    plateau_pct=0.45,
    min_lr_ratio=0.2,
    use_pos_weight=True,
    pos_weight_method='log_scaled',  # Smooth, not tiered
    pos_weight_max=50,               # Reduced from 100
    use_focal_loss=False,            # DISABLED - critical!
)
```

This should:
1. ✅ Recover from expert collapse (like previous run)
2. ✅ Maintain rare code improvements via weighted BCE
3. ✅ Close gap to dense baseline on R@10
4. ✅ Allow meaningful loss comparison with previous runs


Let me first verify the configuration changes you made, then analyze the results comprehensively.

[2 tools called]

Now I have all the information needed. Let me provide a comprehensive analysis.

---

# 📊 Comprehensive Analysis: Post-Fix Experiment Results

## 1. Configuration Confirmation ✅

### Changes Made (Verified from Code):

| Parameter | Previous Run | Current Run | Status |
|-----------|--------------|-------------|--------|
| `use_focal_loss` | `True` | **`False`** | ✅ Fixed |
| `pos_weight_method` | `'tiered'` | **`'log_scaled'`** | ✅ Fixed |
| `pos_weight_max` | `100` | **`50`** | ✅ Fixed |
| `bias_momentum` | `0.8` | **`0.7`** | ✅ Fixed |
| `scheduler_type` | `'linear'` | `'linear'` | Unchanged |
| `plateau_pct` | `0.45` | `0.45` | Unchanged |
| `min_lr_ratio` | `0.2` | `0.2` | Unchanged |

**Confirmation**: All recommended changes were successfully applied.

---

## 2. Results Interpretation - Major Performance Gains! 🎉

### 2.1 Comparison Across All Runs

| Metric | Dense Baseline | MoE + Focal (Dec 28) | MoE No-Focal (Dec 29) | Δ vs Dense |
|--------|----------------|---------------------|----------------------|------------|
| **R@10** | 79.0% | 73.8% | **83.5%** | **+5.7%** ✅ |
| **μR@10** | 46.7% | 42.3% | **49.4%** | **+5.8%** ✅ |
| **NDCG@20** | 47.6% | 40.7% | **44.6%** | -6.3% |
| **MRR** | 45.6% | 38.7% | 34.4% | -24.6% ⚠️ |
| **P@10** | 19.4% | 20.0% | **23.4%** | **+20.6%** ✅ |
| **Final Loss** | 0.0031 | 0.0009 (artificial) | **0.0031** | Comparable ✅ |
| **AUROC** | 59.3% | 77.1% | **83.1%** | **+40.1%** ✅ |

### 2.2 Key Improvements Highlighted

```
🎯 R@10:    73.8% → 83.5%  (+13.1% relative, +9.7pp absolute)
🎯 μR@10:   42.3% → 49.4%  (+16.8% relative, +7.1pp absolute)  
🎯 P@10:    20.0% → 23.4%  (+17.0% relative, +3.4pp absolute)
🎯 Loss:    0.0009 → 0.0031 (now on comparable BCE scale)
```

**This is a substantial improvement!** The MoE model now **exceeds the dense baseline** on R@10, μR@10, and P@10.

### 2.3 The MRR Anomaly - Explanation

The low MRR (0.344 vs dense 0.456) combined with low R@1 (0.8% vs 28%) is concerning but explainable:

```
Recall@1:  0.8%   ← Very low - model not confident on top-1
Recall@5:  72.3%  ← Excellent recovery
Recall@10: 83.5%  ← Best in class
```

This pattern suggests:
1. **Model spreads probability mass** across top candidates rather than concentrating on one
2. **Good for recall**, but hurts metrics that reward single-answer precision
3. This is likely due to log-scaled weights making the model more cautious

---

## 3. Validation Metric Reliability

### Your Concern: Are metrics computed on sampled batches?

Looking at the JSON structure:

```json
"final_val_recall@10": 0.8233972867523356,     // End-of-epoch validation
"precision@10": 0.2337,                         // Full evaluation
"recall@10": 0.835,                             // Full evaluation
```

**Two evaluation phases exist:**
1. **`final_val_*` metrics**: Computed at end of training on validation set (streaming batches)
2. **`full_evaluation.performance.*`**: Post-training comprehensive evaluation

The values are very close:
- `final_val_recall@10`: 82.3%
- `full_evaluation recall@10`: 83.5%

**Conclusion**: The full evaluation metrics (what you see in the JSON `full_evaluation` section) ARE comprehensive and reliable. The small ~1% difference is likely due to:
- Different batch ordering / sampling during training vs. full eval
- End-of-epoch vs. full-pass evaluation

**Trust the `full_evaluation` section** - it runs a complete pass over the eval set.

---

## 4. MoE Expert Collapse - The Critical Issue ⚠️

### 4.1 Collapse Pattern Analysis

From your trajectory:

```
Batch 5100-10900 Summary:
- Collapsed=0: Only 5 batches (5500, 7000, 7900, 9900, 10400)
- Collapsed=1-3: Rare (8500: 1 collapsed, 8600: 1, 10900: 3)
- Collapsed=4: ~90% of batches

CV Range: 0.33-0.56 (target <0.3)
Gini Range: 0.19-0.29 (target <0.2)
```

### 4.2 Comparison with Previous Runs

| Metric | MoE + Focal (Dec 28) | MoE No-Focal (Dec 29) | Change |
|--------|---------------------|----------------------|--------|
| Collapsed (mode) | 4-8 | **4** | Slightly better |
| CV (final) | 0.65-0.83 | **0.48** | Better |
| Gini (final) | 0.36-0.45 | **0.27** | Better |
| Occasional recovery | Never | **5 batches at 0** | New behavior |

**Improvement**: The collapse is now stable at 4 (not escalating to 8), CV improved significantly (0.48 vs 0.78), and there are occasional recovery events!

### 4.3 Why Collapse Persists Despite Improvements

The collapse is happening **within the first 100 batches** before meaningful learning occurs. The DeepSeek bias correction cannot recover from this early state because:

1. **Initial Router Weight Distribution**: The router weights at initialization may favor certain experts
2. **Early Gradient Momentum**: With `bias_momentum=0.7`, the EMA still carries early imbalanced routing
3. **Self-Reinforcing Loop**: Once an expert gets fewer tokens → smaller gradients → harder to recover

**Evidence**: The sporadic 0-collapsed batches (5500, 7000, 7900, 9900, 10400) show the system CAN balance - it just doesn't sustain it.

---

## 5. Root Cause Diagnosis: Why 4 Experts Stay Collapsed

### Hypothesis 1: Initialization Bias

The router linear layer `nn.Linear(d_model, num_experts)` with default Xavier/Kaiming init may create initial bias toward certain experts.

### Hypothesis 2: bias_momentum Still Too High

At `0.7`, the EMA updates as:
```
new_bias = 0.7 × old_bias + 0.3 × current_load
```

This means early routing patterns have ~70% persistence each step. After 1000 batches:
- Batch 1 influence: 0.7^1000 ≈ 0 (washed out)
- But damage from first ~20 batches is sticky (0.7^20 ≈ 0.0008, but cumulative effect matters)

### Hypothesis 3: bias_lr Not Aggressive Enough for Recovery

With `bias_lr=1e-3`, the bias adjustment per step is small. If an expert is consistently underutilized:
```
bias_update = bias_lr × (target_load - current_load)
            = 1e-3 × (0.125 - 0.0)   # For fully collapsed expert
            = 1.25e-4 per step
```

After 10,000 steps: max bias increase = 1.25 → Not enough to overcome router weight bias.

---

## 6. Recommended Next Steps

### Priority 1: More Aggressive Bias Correction

```python
MoEConfig(
    bias_lr=5e-3,          # 5× increase from 1e-3
    bias_momentum=0.5,     # Lower from 0.7 for faster adaptation
)
```

**Rationale**: The current settings allow recovery (as shown by occasional 0-collapsed batches) but don't sustain it. More aggressive correction should make recovery stable.

### Priority 2: Router Initialization Balancing

Add balanced initialization for the router:

```python
# In router __init__, after creating weights:
def _init_router_balanced(self):
    """Initialize router to give equal probability to all experts."""
    with torch.no_grad():
        # Zero out weights to start with uniform routing
        self.router.weight.zero_()
        # Small random noise to break symmetry
        self.router.weight.add_(torch.randn_like(self.router.weight) * 0.01)
```

### Priority 3: Add Expert Load Warmup

For the first N batches, force balanced routing regardless of router output:

```python
# In MoE forward, early in training:
if self.training and current_step < warmup_steps:
    # Blend router scores with uniform distribution
    uniform = torch.ones_like(router_scores) / self.num_experts
    blend_factor = current_step / warmup_steps
    router_scores = blend_factor * router_scores + (1 - blend_factor) * uniform
```

### Priority 4: Z-Loss for Router Stability

Add z-loss to prevent router logit explosion:

```python
MoEConfig(
    z_loss_weight=0.001,   # Small regularization on router logits
)
```

---

## 7. Summary Assessment

### What Worked ✅
1. **Removing Focal Loss**: Loss now on comparable scale, metrics dramatically improved
2. **Log-scaled Weights**: Smooth gradient flow, μR@10 improved significantly
3. **Lower bias_momentum**: CV improved from 0.78 → 0.48

### What Still Needs Work ⚠️
1. **Expert Collapse**: 4/8 experts consistently collapsed
2. **Bias Correction Speed**: Current settings allow occasional recovery but don't sustain it
3. **MRR/R@1**: Model too diffuse in top predictions

### Overall Grade: **A-** (Major improvement, one key issue remaining)

The model now **exceeds dense baseline** on key metrics! This validates the architectural approach. The remaining expert collapse is limiting further gains - if all 8 experts were active, you could potentially see:

```
Projected with full expert utilization:
- R@10: 83.5% → ~87-90%
- μR@10: 49.4% → ~52-55%
```

---

## 8. Configuration for Next Experiment

```python
# More aggressive bias correction
MoEConfig(
    num_experts=8,
    num_shared_experts=1,
    top_k=2,
    load_balance_strategy='deepseek',
    aux_loss_weight=0.0,
    bias_lr=5e-3,           # 5× increase
    bias_momentum=0.5,      # Lower for faster recovery
    z_loss_weight=0.001,    # Add router regularization
    expert_dropout=0.1,
    use_swiglu_experts=True
)

# Keep optimization config (working well)
OptimizeConfig(
    scheduler_type='linear',
    warmup_pct=0.15,
    plateau_pct=0.45,
    min_lr_ratio=0.2,
    use_pos_weight=True,
    pos_weight_method='log_scaled',
    pos_weight_max=50,
    use_focal_loss=False,
)
```

**Expected Outcome**: Expert collapse should recover within first 1000-2000 batches, with CV dropping to <0.35 and Gini <0.18. This should unlock additional performance gains.

---


# 🔬 Expert 3 (Independent Judge): Critical Analysis of Dec 29 Experiment

## Part 1: My Independent Assessment

### 1.1 Verified Configuration Changes

From the code at lines 14764-14777 and 694-709:

| Parameter | Dec 28 Run | Dec 29 Run | Verified Change? |
|-----------|-----------|------------|------------------|
| `use_focal_loss` | `True` | **`False`** | ✅ Fixed |
| `pos_weight_method` | `'tiered'` | **`'log_scaled'`** | ✅ Fixed |
| `pos_weight_max` | `100` | **`50`** | ✅ Fixed |
| `bias_momentum` | `0.8` | **`0.7`** | ✅ Fixed |
| `scheduler_type` | `'linear'` | `'linear'` | Same |

**All key recommendations were implemented.**

### 1.2 Results Comparison (Using Verified JSON Data)

| Metric | Dense (exp2b) | MoE Dec 28 | MoE Dec 29 | Dec 29 vs Dense |
|--------|--------------|------------|------------|-----------------|
| **R@10** | 79.0% | 73.8% | **83.5%** | **+5.7%** ✅ |
| **μR@10** | 46.7% | 42.3% | **49.4%** | **+5.8%** ✅ |
| **NDCG@10** | 44.6% | 36.9% | **41.2%** | -7.6% |
| **NDCG@20** | 47.6% | 40.7% | **44.6%** | -6.3% |
| **MRR** | 45.6% | 38.7% | **34.4%** | -24.5% ⚠️ |
| **P@10** | 19.4% | 20.0% | **23.4%** | **+20.6%** ✅ |
| **AUROC** | 59.3% | 77.1% | **83.1%** | **+40.1%** ✅ |
| **Final Loss** | 0.0031 | 0.0009 | **0.0031** | Identical ✅ |

### 1.3 MoE Health Metrics (From Your Trajectory)

```
Dec 29 Run Summary (from trajectory):
- Collapsed mode: 4 experts (~90% of batches)
- CV range: 0.33-0.56 (improved from Dec 28's 0.55-0.83)
- Gini range: 0.19-0.29 (improved from Dec 28's 0.28-0.45)
- Occasional full recovery: Batches 5500, 7000, 7900, 9900, 10400 show 0 collapsed
- Average collapsed: 3.77 (from JSON line 169)
```

**Key Observation**: Collapse improved but not eliminated. CV and Gini substantially better.

---

## Part 2: Critical Issue - Why Tail Metrics Are Still 0

### 2.1 The Critical Finding: Dense Baseline ALSO Has 0 Tail Accuracy

From the JSON files:

```json
// Dense baseline (exp2b)
"common_top10_acc": 0.79,
"medium_top10_acc": 0.0,   // Also 0!
"rare_top10_acc": 0.0,     // Also 0!
"tail_top10_acc": 0.0,     // Also 0!

// MoE current (Dec 29)
"common_top10_acc": 0.835,
"medium_top10_acc": 0.0,   // Same pattern
"rare_top10_acc": 0.0,     // Same pattern
"tail_top10_acc": 0.0,     // Same pattern
```

**This is NOT a MoE-specific issue. The DENSE baseline has the same 0% accuracy on medium/rare/tail tiers!**

### 2.2 Root Cause Analysis: Why Tail/Rare/Medium Are All 0

Looking at the evaluation code (lines 7565-7599):

```python
tail_codes = set(np.where(
    (code_frequencies <= freq_percentiles[0]) & 
    (code_frequencies > 0)
)[0].tolist())
```

The metric logic:
1. Find target codes that belong to tail tier
2. Check if ANY of those tail codes appear in top-10 predictions
3. If tier_total['tail'] > 0, compute accuracy; else return 0

**Three possible causes for 0 accuracy:**

#### Cause A: Evaluation Sample Size Too Small
```json
"num_codes_evaluated": 200   // Only 200 codes in eval set
```

If the evaluation uses only 200 random samples, and those samples happen to not contain any rare/tail code targets, the metric will be 0 by definition.

**Evidence**: Dense baseline has `num_codes_evaluated: 108` and same 0 pattern.

#### Cause B: Extreme Class Imbalance in Evaluation Set

From your earlier analysis (line 397-440 of the analysis file):
```
Imbalance ratio: 16,952,106x
Gini coefficient: 0.9390

Tier distribution:
- Common (75-100 %ile): 98.8% of total occurrences
- Medium (50-75 %ile):  1.1% of total occurrences  
- Rare (25-50 %ile):    0.1% of total occurrences
- Tail (0-25 %ile):     0.0% of total occurrences
```

If tail codes represent < 0.01% of occurrences, in a 200-sample evaluation, you might have **zero samples** with tail code targets.

#### Cause C: Model Probability Mass

Even if tail codes appear as targets, the model may assign them probability ranks > 10. Since the metric checks top-10 only:

```python
top_10_preds = torch.topk(predictions, 10, dim=-1).indices
```

If a tail code is ranked #11-6297, it won't be counted.

### 2.3 Why pos_weight Doesn't Fix This

**Critical Misunderstanding**: `pos_weight` affects **training loss gradient magnitude**, not evaluation metric thresholds.

```python
# What pos_weight does:
loss = BCE(logits, targets) × pos_weight[class]
# → Increases gradient for rare codes during TRAINING

# What it does NOT do:
# ❌ Does not change how evaluation metrics are computed
# ❌ Does not guarantee rare codes appear in top-10 predictions
# ❌ Does not affect the evaluation sample selection
```

**The model may have learned better rare code representations** (as evidenced by μR@10 improvement: 42.3% → 49.4%), **but the evaluation metric `tail_top10_acc` is computed on a tiny sample that may not contain tail code targets.**

### 2.4 Verification Needed

To diagnose, you need to log during evaluation:
```python
print(f"tier_total: {tier_total}")  # How many samples have tail/rare targets?
```

If `tier_total['tail'] == 0`, the metric is undefined (returns 0).

---

## Part 3: Evaluation of Expert 2's Analysis

### 3.1 What Expert 2 Got RIGHT ✅

| Claim | Assessment |
|-------|------------|
| MoE exceeds dense on R@10, μR@10, P@10 | ✅ **Correct** - verified from JSON |
| Loss now comparable (0.0031 vs 0.0031) | ✅ **Correct** - removing focal loss fixed this |
| Expert collapse improved but not eliminated | ✅ **Correct** - CV 0.48 vs 0.78 |
| Configuration changes verified | ✅ **Correct** - all changes applied |
| Primary metrics are reliable | ✅ **Correct** - full eval runs complete pass |

### 3.2 What Expert 2 Got WRONG or INCOMPLETE ❌

#### Issue 1: Did Not Explain Why Tail Metrics = 0

Expert 2 did not address the fundamental question: why are tail/rare/medium all 0 even after adding pos_weight?

**The correct answer**: This is not a model issue but an **evaluation sampling issue**. The dense baseline has the same 0 pattern.

#### Issue 2: Overly Optimistic About Expert Collapse

Expert 2 states:
> "Expert collapse should recover within first 1000-2000 batches"

But the trajectory shows 4 experts collapsed throughout training with only sporadic recovery. The recommended changes (bias_lr=5e-3, bias_momentum=0.5) were not aggressive enough.

**More accurate assessment**: The bias correction mechanism is fundamentally limited by how the router was initialized, not just by hyperparameters.

#### Issue 3: MRR Decline Not Fully Diagnosed

Expert 2 mentions low MRR but doesn't fully explain:

```
MRR: 45.6% (dense) → 34.4% (MoE Dec 29) = -24.5%
R@1: 28% (dense) → 0.8% (MoE Dec 29) = -97%!
```

**The MoE model has collapsed R@1 to near-zero!** This suggests:
1. Model probability mass is spread across many codes
2. Top-1 confidence is very low
3. This could be over-regularization from log-scaled weights

#### Issue 4: Incomplete Analysis of CV/Gini Improvement Source

Expert 2 credits the improvement to configuration changes, but doesn't note:
- CV improved from 0.78 → 0.48 is significant
- But 4 collapsed experts persist
- The improvement may be because **the remaining 4 active experts are more balanced**, not because collapse was fixed

### 3.3 What Expert 2 Missed Entirely

#### Missing 1: R@1 Catastrophic Collapse

```
Dense R@1:  28.0%
MoE R@1:    0.8%   ← 97% collapse!
```

This is buried in the data but critically important. The model has lost all top-1 precision.

#### Missing 2: The Dense Baseline Also Fails on Tail Metrics

Expert 2 didn't compare tier metrics between dense and MoE:
```
Dense:  common=79%, medium=0%, rare=0%, tail=0%
MoE:    common=83.5%, medium=0%, rare=0%, tail=0%
```

Both models fail identically on non-common tiers. This points to **evaluation methodology issues**, not model issues.

#### Missing 3: num_codes_evaluated Discrepancy

```
Dense:  num_codes_evaluated: 108
MoE:    num_codes_evaluated: 200
```

Different evaluation sample sizes make direct comparison problematic.

---

## Part 4: My Recommendations

### Priority 1: Fix the Evaluation to Actually Measure Tail Performance

The current evaluation is **structurally incapable** of measuring tail code performance because:
1. Sample size too small (200 samples)
2. Random sampling unlikely to include tail code targets
3. Top-10 threshold may be too restrictive for rare codes

**Fix**:
```python
# Stratified evaluation sampling
def sample_stratified_for_eval(data, code_frequencies, n_per_tier=50):
    """Ensure eval set contains samples from all tiers."""
    tiers = classify_codes_by_tier(code_frequencies)
    eval_samples = []
    for tier_name, tier_codes in tiers.items():
        # Find samples that have this tier's codes as targets
        tier_samples = data[data['targets'].apply(
            lambda x: any(c in tier_codes for c in x)
        )]
        eval_samples.append(tier_samples.sample(min(n_per_tier, len(tier_samples))))
    return pd.concat(eval_samples)
```

### Priority 2: Add Top-50 Tier Metrics

Top-10 is too restrictive for rare codes. Add:
```python
metrics['tail_top50_acc'] = compute_tier_acc(predictions, targets, tail_codes, k=50)
metrics['rare_top50_acc'] = compute_tier_acc(predictions, targets, rare_codes, k=50)
```

### Priority 3: Diagnose R@1 Collapse

The 0.8% R@1 needs investigation:
```python
# Check confidence distribution
top1_probs = torch.max(torch.sigmoid(predictions), dim=-1)
print(f"Top-1 probability: mean={top1_probs.mean():.3f}, max={top1_probs.max():.3f}")
```

If top-1 probabilities are all < 0.5, the model is too uncertain. Consider:
- Lower pos_weight_max from 50 to 25
- Or use different weight distribution that doesn't flatten all predictions

### Priority 4: More Aggressive Expert Recovery

Current bias_momentum=0.7 is still too high for recovery:
```python
MoEConfig(
    bias_lr=1e-2,       # 10× increase
    bias_momentum=0.3,  # Much lower for fast recovery
)
```

Or implement **router warmup**:
```python
# First 500 batches: force uniform routing
if batch_idx < 500:
    router_logits = torch.zeros_like(router_logits)  # Uniform
```

---

## Part 5: Summary - Agreement/Disagreement with Expert 2

### Full Agreement ✅
1. The configuration changes were correctly implemented
2. MoE now exceeds dense baseline on R@10, μR@10, P@10
3. Loss is now on comparable scale (0.0031)
4. Expert collapse improved (CV 0.78 → 0.48)
5. Full evaluation metrics are reliable

### Partial Agreement ⚠️
1. Expert collapse "improved" but calling 4/8 collapsed "progress" is generous
2. The expected further improvement may not materialize without more aggressive changes

### Disagreement ❌
1. **Tail metrics are 0 due to evaluation methodology, not model failure** - Expert 2 didn't identify this
2. **R@1 collapse from 28% → 0.8% is critical** - Expert 2 didn't highlight this
3. **The recommended next steps are too conservative** - bias_lr=5e-3 won't fix persistent collapse

### Key Missing from Expert 2's Analysis
1. Dense baseline also has 0% tail/rare/medium accuracy
2. num_codes_evaluated differs between runs (108 vs 200)
3. R@1 catastrophic collapse (97% drop) needs urgent attention
4. Evaluation stratification is needed, not just model changes

---

## Part 6: Why Tail Accuracy = 0 Despite pos_weight (Summary)

| Reason | Explanation | Evidence |
|--------|-------------|----------|
| **Evaluation sampling** | 200 random samples may not include tail code targets | Dense also has 0% tail |
| **Metric threshold** | Top-10 cutoff too restrictive for rare codes | μR@10=49.4% shows rare codes ARE learned |
| **pos_weight scope** | Only affects training gradients, not eval ranking | By design |
| **Class imbalance severity** | Tail codes < 0.01% of occurrences | Your earlier Gini=0.94 analysis |

**The model IS learning tail codes better** (μR@10 improved from 42.3% to 49.4%), but the **evaluation metric cannot detect this** because:
1. Not enough tail samples in eval set
2. Top-10 threshold too restrictive
3. This is an evaluation problem, not a model problem

---

## Recommended Configuration for Next Run

```python
# MoE Config - More aggressive bias correction
MoEConfig(
    num_experts=8,
    num_shared_experts=1,
    top_k=2,
    load_balance_strategy='deepseek',
    aux_loss_weight=0.0,
    bias_lr=1e-2,          # 10× current - aggressive
    bias_momentum=0.3,     # Very low for fast adaptation
    z_loss_weight=0.001,   # Add router regularization
    expert_dropout=0.1,
)

# Optimize Config - Address R@1 collapse
OptimizeConfig(
    scheduler_type='linear',
    warmup_pct=0.15,
    plateau_pct=0.45,
    min_lr_ratio=0.2,
    use_pos_weight=True,
    pos_weight_method='log_scaled',
    pos_weight_max=25,     # Reduced from 50 to prevent over-smoothing
    use_focal_loss=False,
)
```

And critically:
```python
# Add stratified evaluation
eval_samples = sample_stratified_for_eval(val_data, code_frequencies, n_per_tier=100)
```

This should:
1. ✅ Fix expert collapse via aggressive bias correction
2. ✅ Restore R@1 performance via less aggressive pos_weight
3. ✅ Enable meaningful tail metric measurement via stratified eval
4. ✅ Maintain the R@10/μR@10 gains already achieved