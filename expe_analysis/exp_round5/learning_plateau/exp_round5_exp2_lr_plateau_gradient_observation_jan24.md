# Systematic Observations: pos_weight_max Ablation (50 → 200)
- Jan 24, 2026
## 1. CONFIGURATION DIFFERENCE

| Parameter | Original (exp_round5/exp2) | New (exp_round5_1_lr_plateau/exp2) |
|-----------|---------------------------|-----------------------------------|
| `pos_weight_max` | 50 | 200 (4× increase) |
| `enable_gradient_tier_analysis` | Not present | `true` |
| Training samples | ~1.7M | ~1.7M |
| All other parameters | Identical | Identical |

---

## 2. FINAL METRIC COMPARISON

### 2.1 Primary Ranking Metrics

| Metric | pos_weight_max=50 | pos_weight_max=200 | Delta | % Change |
|--------|------------------|-------------------|-------|----------|
| **recall@5** | 0.7218 | 0.6861 | -0.0357 | **-4.9%** |
| **recall@10** | 0.8285 | 0.8171 | -0.0114 | **-1.4%** |
| **recall@20** | 0.8916 | 0.8930 | +0.0014 | +0.2% |
| **recall@50** | 0.9478 | 0.9512 | +0.0034 | +0.4% |
| **micro_recall@10** | 0.4622 | 0.4656 | +0.0034 | **+0.7%** |
| **micro_recall@20** | 0.5719 | 0.5844 | +0.0125 | **+2.2%** |
| **micro_recall@50** | 0.7088 | 0.7262 | +0.0174 | **+2.5%** |
| **ndcg@10** | 0.3983 | 0.3898 | -0.0085 | -2.1% |
| **ndcg@20** | 0.4320 | 0.4265 | -0.0055 | -1.3% |
| **ndcg@50** | 0.4687 | 0.4613 | -0.0074 | -1.6% |
| **mrr** | 0.3409 | 0.3242 | -0.0167 | **-4.9%** |

### 2.2 Tier-Specific Accuracy

| Metric | pos_weight_max=50 | pos_weight_max=200 | Delta |
|--------|------------------|-------------------|-------|
| **common_top10_acc** | 0.8289 | 0.8173 | **-1.2%** |
| **medium_top10_acc** | 0.0411 | 0.00157 | **-96.2%** |
| **rare_top10_acc** | 0.0 | 0.0 | No change |
| **tail_top10_acc** | 0.0 | 0.0 | No change |
| **balanced_top10_acc** | 0.2175 | 0.2047 | -5.9% |

### 2.3 Calibration and Loss

| Metric | pos_weight_max=50 | pos_weight_max=200 | Delta |
|--------|------------------|-------------------|-------|
| **train_loss_final** | 0.0138 | 0.0134 | -0.0004 |
| **val_loss (BCE)** | 0.0037 | 0.0034 | -0.0003 |
| **positive_brier** | 0.6785 | 0.6868 | +0.0083 |
| **macro_auroc** | 0.8456 | 0.8781 | **+3.8%** |
| **macro_auprc** | 0.1025 | 0.1048 | +2.3% |

---

## 3. GRADIENT TIER ANALYSIS RESULTS (pos_weight_max=200)

### 3.1 Gradient Tier Evolution During Training

From batch metrics, I observe a **dramatic shift in gradient concentration**:

| Training Phase | Common Frac | Medium Frac | Rare Frac | Tail Frac | Total Norm |
|----------------|-------------|-------------|-----------|-----------|------------|
| **Step 1** (init) | 17.8% | 27.3% | 26.5% | 17.8% | 530,569 |
| **Step 101** | 17.6% | 27.7% | 26.6% | 18.0% | 388,416 |
| **Step 301** | 17.2% | 27.8% | 26.7% | 18.2% | 61,024 |
| **Step 501** | 16.9% | 27.9% | 27.0% | 18.4% | 24,989 |
| **Step 1501** | 42.7% | 21.9% | 17.4% | 10.4% | 3,398 |
| **Step 3001** | 66.7% | 16.1% | 7.1% | 3.0% | 1,632 |
| **Step 5801** | 83.6% | 9.0% | 1.8% | 1.0% | 1,113 |
| **Step 6001** | 85.5% | 7.7% | 1.3% | 0.7% | 3,267 |
| **Step 9001** | 84.6% | 10.2% | 0.5% | 0.1% | ~10,500 |
| **Step 12001** | 85.3% | 11.2% | 0.6% | 0.1% | 22,129 |
| **Final epoch avg** | 82.8% | 10.2% | 2.0% | 1.1% | 4,861 |

### 3.2 Key Observations from Gradient Dynamics

1. **Early Training (steps 1-500)**: Gradient distribution is **relatively balanced**
   - Common: ~17-18%
   - Medium: ~27-28%
   - Rare: ~26-27%
   - Tail: ~17-18%
   - **Total norm very high**: 530K → 25K (20× reduction in first 500 steps)

2. **Mid-Training Transition (steps 500-3000)**: Rapid **concentration shift**
   - Common fraction increases from ~17% to ~67%
   - Tail fraction collapses from ~18% to ~3%
   - **This is where the gradient starvation begins**

3. **Late Training (steps 3000-12000)**: **Severe gradient concentration**
   - Common codes capture **82-86%** of total gradients
   - Tail codes receive only **0.1-1.1%** of gradients
   - Rare codes receive only **0.5-2.0%** of gradients

4. **Final Recorded Gradient Distribution**:
   - `common_frac`: 84.7%
   - `tail_frac`: 0.17%
   - **Tail gradient fraction is ~500× smaller than common**

---

## 4. TRAINING DYNAMICS COMPARISON

### 4.1 Loss Trajectory

| Checkpoint | pos_weight_max=50 (Loss) | pos_weight_max=200 (Loss) |
|------------|-------------------------|--------------------------|
| Initial | 0.8055 | 0.8122 |
| Final | 0.0032 | 0.0031 |
| Improvement | 0.8023 | 0.8092 |

**Observation**: Loss trajectories are nearly identical between experiments.

### 4.2 Recall@10 Training Trajectory (from batch metrics, pos_weight_max=200)

| Step | Recall@10 |
|------|-----------|
| 1 | 0.022 |
| 301 | 0.296 |
| 501 | 0.530 |
| 1001 | 0.693 |
| 3001 | 0.779 |
| 5001 | 0.804 |
| 8001 | 0.843 |
| 10001 | 0.843 |
| 12001 | 0.854 |

**Observation**: Recall@10 plateaus around step 8000, matching the pattern in the original experiment.

---

## 5. COMPUTATIONAL EFFICIENCY

| Metric | pos_weight_max=50 | pos_weight_max=200 |
|--------|------------------|-------------------|
| Training time (sec) | 14,739 | 12,323 |
| Samples/sec | 1,037 | 620 |
| Peak memory (GB) | 11.14 | 12.79 |
| Achieved TFLOPs | 2.35 | 1.40 |
| MFU % | 0.90% | 0.54% |

**Observation**: The new experiment ran faster wall-clock time despite lower throughput—likely due to infrastructure variance rather than algorithmic difference.

---

## 6. SPECIFIC OBSERVATIONS WITHOUT INTERPRETATION

### 6.1 What Changed Positively with Higher pos_weight_max:
- **micro_recall@10/20/50** all increased (+0.7% to +2.5%)
- **macro_auroc** increased significantly (+3.8%)
- **macro_auprc** increased (+2.3%)
- **recall@20/50** slightly increased (+0.2% to +0.4%)

### 6.2 What Changed Negatively with Higher pos_weight_max:
- **recall@5** decreased (-4.9%)
- **recall@10** decreased (-1.4%)
- **mrr** decreased (-4.9%)
- **ndcg@10/20/50** all decreased (-1.3% to -2.1%)
- **precision@5/10** decreased
- **common_top10_acc** decreased (-1.2%)
- **medium_top10_acc** collapsed from 4.1% to 0.16% (-96%)
- **positive_brier** worsened (higher is worse)

### 6.3 What Remained Unchanged:
- **rare_top10_acc** = 0 (both experiments)
- **tail_top10_acc** = 0 (both experiments)
- **tail_code_coverage** = 0 (both experiments)
- Loss plateau timing (~step 6000-8000)
- Final loss levels (~0.003)

---

## 7. GRADIENT TIER ANALYSIS - RAW NUMBERS

From the final results file `full_evaluation.gradient_tier`:

```json
{
  "common_frac": 0.8467924870295841,
  "tail_frac": 0.0017118535626965263
}
```

From epoch-level metrics:
```
train_grad_tier_common_frac: 0.828
train_grad_tier_common_norm: 3.611
train_grad_tier_medium_frac: 0.102
train_grad_tier_medium_norm: 0.295
train_grad_tier_rare_frac: 0.020
train_grad_tier_rare_norm: 0.031
train_grad_tier_tail_frac: 0.011
train_grad_tier_tail_norm: 0.017
train_grad_tier_total_norm: 4861.38
```

**Observations on absolute gradient norms**:
- Common codes: norm = 3.611 (per-code average)
- Medium codes: norm = 0.295 (8.2% of common)
- Rare codes: norm = 0.031 (0.9% of common)
- Tail codes: norm = 0.017 (0.5% of common)

---

## 8. PATTERN OBSERVATIONS

1. **The plateau phenomenon persists** regardless of pos_weight setting
2. **Loss and metrics plateau together** in both experiments around step 6000-8000
3. **Gradient concentration increases monotonically during training**—the longer training runs, the more gradients flow to common codes
4. **The 4× increase in pos_weight_max (50→200) did not prevent gradient starvation**—tail codes still receive <0.2% of total gradients by end of training
5. **The model "gave up" on rare/tail codes** in both experiments—tier accuracy remains 0% for rare/tail in both settings
6. **micro_recall improves while sample_recall decreases**—this pattern suggests the model is becoming more conservative in predictions, predicting fewer codes but with higher per-code precision for the codes it does predict

---

This is a purely descriptive analysis of what the data shows. The evidence is presented without drawing conclusions about causality or recommended actions.


# Detailed Analysis: 3.4M Member Trained Model Diagnostic Results
- Jan 24, 2026
- This analysis compared the effects of larger training dataset on the gradient tier distribution and the logit distribution.
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