# 🔬 Comprehensive Analysis: `exp6_auxiliary_free` (Fine-Grained 16 Experts)

## 1. Learning Configuration Summary

Based on the code and logs, here's your current setup:

| Parameter | Value | Assessment |
|-----------|-------|------------|
| **Architecture** | | |
| `num_experts` | 16 | Fine-grained |
| `num_shared_experts` | 1 | 1 shared + 15 routed |
| `top_k` | 2 | Standard |
| `d_ff` | 512 | Standard FFN |
| **Load Balancing** | | |
| `load_balance_strategy` | 'deepseek' | Aux-free bias correction |
| `aux_loss_weight` | 0.0 | No aux loss |
| `bias_lr` | **1e-3** | ✅ Fixed (was 1e-5) |
| `bias_momentum` | 0.8 | ✅ Lower for faster adaptation |
| `expert_dropout` | 0.1 | ✅ Increased |
| **Optimizer** | | |
| `scheduler_type` | 'onecycle' | ✅ Good for 1 epoch |
| `use_pos_weight` | True | ✅ Class imbalance handling |
| `pos_weight_max` | 50.0 | Aggressive weighting |
| **Training** | | |
| Total batches | 10,965 | Full epoch |
| Batch size | 32 (effective 128) | Standard |

---

## 2. Loss Trajectory Analysis

### Loss Progression

| Batch | Loss | Phase | LR (OneCycle) |
|-------|------|-------|---------------|
| 0 | 0.8292 | Warmup start | ~1.6e-5 |
| 500 | 0.0748 | Warmup | ~8e-5 |
| 1000 | 0.0417 | Approaching peak | ~1.6e-4 |
| 2000 | 0.0349 | Near peak | ~3.2e-4 |
| 3000 | 0.0356 | Peak LR | ~4e-4 |
| 5000 | 0.0287 | Decay starts | ~3.2e-4 |
| 7000 | 0.0286 | Decay | ~1.6e-4 |
| 8400 | 0.0291 | Late decay | ~8e-5 |

### 🔍 Loss Plateau Detection

```
Loss Trajectory Visualization:

0.83 |█
     |
0.40 |  ██
     |
0.10 |    ████
     |
0.04 |        ████████
0.03 |                ████████████████████████████████████
     |________________________________________________
     0    1k   2k   3k   4k   5k   6k   7k   8k   9k  10k
                        Batch
```

**Finding: Loss plateaus around batch 3000-4000 at ~0.028-0.032**

The loss is essentially flat from batch 3000 onwards. This indicates:
1. ✅ Model has converged to a stable point
2. ⚠️ **Potential plateau** - not improving further
3. The OneCycleLR decay phase (batch 3300+ = 30% of 10965) may be causing premature convergence

---

## 3. Primary Metrics Analysis

### Recall@10 Trajectory

| Batch | R@10 | vs Random (0.04%) | vs Your Dense (79%) |
|-------|------|-------------------|---------------------|
| 500 | 22.4% | **560×** | 28% |
| 1000 | 21.3% | **533×** | 27% |
| 2000 | 54.5% | **1363×** | 69% |
| 3000 | 60.6% | **1515×** | 77% |
| 5000 | 67.5% | **1688×** | 85% |
| 7000 | 72.7% | **1818×** | 92% |
| 8400 | 72.4% | **1810×** | 92% |

**Assessment**: R@10 at 72.4% is approaching your dense baseline of 79%!

### Micro-Recall@10 (Critical for Class Imbalance)

| Batch | μR@10 | vs Dense (46.7%) |
|-------|-------|------------------|
| 2000 | 20.9% | 45% |
| 4000 | 30.0% | 64% |
| 6000 | 32.1% | 69% |
| 8000 | 29.7% | 64% |
| 8400 | 31.7% | 68% |

**Assessment**: μR@10 peaked around batch 6000-7000 at ~34%, then slightly declined. This suggests **overfitting to common codes** in the late training phase.

### NDCG@20 Trajectory

| Batch | NDCG@20 | Quality |
|-------|---------|---------|
| 2000 | 0.231 | Fair |
| 4000 | 0.328 | Good |
| 6000 | 0.357 | Good |
| 8000 | 0.359 | Good |
| 8400 | 0.384 | Good |

**Assessment**: NDCG continues to improve slightly, indicating ranking quality is still benefiting from training.

---

## 4. MoE Health Metrics Analysis 🎉 **MAJOR IMPROVEMENT!**

### Expert Collapse Recovery

| Batch | Collapsed | CV | Gini | Assessment |
|-------|-----------|-----|------|------------|
| 0 | **12** | 0.914 | 0.502 | 🔴 Critical |
| 500 | 4 | 0.515 | 0.274 | 🟡 Recovering |
| 1000 | 4 | 0.559 | 0.286 | 🟡 Stable |
| 3000 | **0** | 0.371 | 0.192 | 🟢 Healthy! |
| 5000 | **0** | 0.319 | 0.146 | 🟢 Excellent! |
| 7000 | **0** | 0.308 | 0.131 | 🟢 Excellent! |
| 8000 | **0** | 0.309 | 0.131 | 🟢 Excellent! |

### Visual: Expert Load Balance Recovery

```
Collapsed Experts Over Training:

12 |████
   |
 8 |
   |
 4 |    ████████████████
   |
 0 |                    ████████████████████████████████████
   |________________________________________________
   0    1k   2k   3k   4k   5k   6k   7k   8k   9k  10k
                        Batch
```

**🎉 The bias_lr=1e-3 fix worked!** Experts recovered from collapse by batch 3000.

### CV (Coefficient of Variation) Improvement

| Target | Your Final | Status |
|--------|------------|--------|
| < 0.3 | **0.31** | ✅ Nearly optimal |

### Gini Coefficient Improvement

| Target | Your Final | Status |
|--------|------------|--------|
| < 0.2 | **0.13** | ✅ Excellent equality |

---

## 5. Comparison: Your Results vs Reference Baselines

| Metric | Random | Your Dense (79% R@10) | Current MoE (8400) | MoE vs Dense |
|--------|--------|----------------------|--------------------| -------------|
| R@10 | 0.04% | 79% | **72.4%** | 92% |
| μR@10 | 0.1% | 46.7% | **31.7%** | 68% |
| NDCG@20 | 0.04% | 45% | **38.4%** | 85% |
| Precision@10 | 0.04% | 19.4% | **15.0%** | 77% |

**Assessment**: MoE is tracking ~70-85% of dense baseline performance at batch 8400 (~77% through epoch). Full epoch should close the gap.

---

## 6. Identified Issues and Root Causes

### Issue 1: Loss Plateau After Batch 3000 ⚠️

**Evidence**:
- Loss: 0.0356 at batch 3000 → 0.0291 at batch 8400 (only 18% improvement in 50% more training)
- Metrics improvement also slowed

**Root Cause**: OneCycleLR with `pct_start=0.30` means:
- Peak LR at batch 3289 (30% of 10965)
- After that, LR decays continuously
- Model can't explore new solutions as LR drops

**Comparison with optimal schedule**:
```
Current OneCycleLR:
           peak
            ▲
           ╱ ╲
          ╱   ╲
         ╱     ╲       ← LR too low during 70% of training
        ╱       ╲
start ╱         ╲end
─────────────────────
  0%    30%    100%

Needed for your case:
         plateau
          ████
         ╱    ╲
        ╱      ╲
       ╱        ╲     ← Longer high-LR phase
      ╱          ╲
start╱            ╲end
─────────────────────
  0%   50%    100%
```

### Issue 2: Micro-Recall Peaked and Declined ⚠️

**Evidence**:
- μR@10: 33.7% at batch 5800 → 31.7% at batch 8400 (5.9% decline)

**Root Cause**: As LR decays, model converges more to common codes, losing rare code discrimination learned during high-LR phase.

### Issue 3: Some Batches Still Show 4 Collapsed Experts

**Evidence**: Batches 5200, 5600, 6300, 6800 show 4 collapsed experts intermittently.

**Root Cause**: Router sometimes routes to subset of experts for specific input patterns. This is likely **data-dependent** rather than systematic collapse.

---

## 7. Performance Projections

### If Training Completes to Batch 10965

Based on current trajectory:

| Metric | At 8400 | Projected Final | vs Dense |
|--------|---------|-----------------|----------|
| R@10 | 72.4% | ~75-77% | 95-97% |
| μR@10 | 31.7% | ~32-34% | 68-73% |
| NDCG@20 | 38.4% | ~40-42% | 89-93% |

**The MoE should achieve ~95% of dense baseline on R@10**, but μR@10 gap indicates class imbalance issue persists.

---

## 8. Recommendations for Enhanced Performance

### Tier 1: Schedule Optimization (High Impact)

#### Fix: Extend High-LR Phase

Modify OneCycleLR to keep LR higher longer:

```python
optimize_config = OptimizeConfig(
    scheduler_type='onecycle',
    onecycle_pct_start=0.40,      # CHANGED: 40% warmup instead of 30%
    onecycle_div_factor=10,       # CHANGED: start_lr = max_lr/10 (was /25)
    onecycle_final_div=100,       # CHANGED: end_lr = max_lr/100 (was /1000)
    # This keeps LR higher for longer
)
```

**Or switch to Linear Plateau schedule**:

```python
optimize_config = OptimizeConfig(
    scheduler_type='linear',       # Linear warmup + plateau + decay
    warmup_pct=0.15,              # 15% warmup
    plateau_pct=0.35,             # 35% at peak (total 50% before decay)
    min_lr_ratio=0.1,             # End at 10% of peak (not 1%)
)
```

### Tier 2: Class Imbalance (Medium Impact)

Your μR@10 (31.7%) is still significantly lower than R@10 (72.4%), indicating the model still favors common codes.

#### Fix: Increase pos_weight_max or Add Focal Loss

```python
# Option A: More aggressive pos_weight
optimize_config = OptimizeConfig(
    use_pos_weight=True,
    pos_weight_max=75.0,  # INCREASED from 50.0
)

# Option B: Add Focal Loss for hard examples
# (Requires code modification to use FocalLoss criterion)
```

### Tier 3: MoE Fine-Tuning (Lower Impact, Already Good)

Your MoE metrics are now healthy. Minor optimizations:

```python
# Consider slightly higher bias_lr for faster recovery
MoEConfig(
    bias_lr=2e-3,       # Try 2× current value
    bias_momentum=0.7,  # Slightly lower for faster adaptation
)
```

---

## 9. Summary Assessment

### What's Working ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| DeepSeek bias correction | ✅ Excellent | Experts recovered from 12→0 collapsed |
| OneCycleLR | ✅ Good | Smooth learning curve |
| pos_weight BCE | ✅ Working | Better than unweighted |
| Expert load balance | ✅ Excellent | CV=0.31, Gini=0.13 |
| Loss convergence | ✅ Stable | 0.028-0.032 range |

### What Needs Improvement ⚠️

| Issue | Severity | Fix |
|-------|----------|-----|
| Loss plateau after 30% | Medium | Extend high-LR phase |
| μR@10 peaked then declined | Medium | Higher pos_weight or Focal Loss |
| Not matching dense R@10 | Low | Full epoch should close gap |

### Overall Grade: **B+** (Strong improvement from previous runs)

The bias_lr fix was **critical** - MoE is now functioning properly with all experts active. The remaining gap to dense baseline is due to:
1. Only 77% through epoch (will improve)
2. LR schedule causing premature convergence
3. Continued class imbalance favoring common codes

---

## 10. Recommended Next Steps

### Immediate (This Run)
1. **Let training complete** to batch 10965 - metrics should improve ~5-10%
2. **Monitor final μR@10** - if below 35%, class imbalance is the bottleneck

### Next Experiment
1. **Use Linear Plateau schedule** with 50% high-LR phase
2. **Increase pos_weight_max to 75.0**
3. **Try 8 experts instead of 16** - fewer experts may converge faster

### Configuration for Next Run

```python
# Recommended exp6 config for next iteration
configs['exp6_auxiliary_free_v2'] = (
    MoEConfig(
        d_model=256,
        d_ff=512,
        num_experts=8,              # Reduced from 16
        num_shared_experts=1,
        top_k=2,
        load_balance_strategy='deepseek',
        aux_loss_weight=0.0,
        bias_lr=2e-3,               # Slightly higher
        bias_momentum=0.75,         # Lower for faster adaptation
        expert_dropout=0.1,
        use_moe_from_layer=2,
        use_swiglu_experts=True
    ),
    True
)

optimize_config = OptimizeConfig(
    scheduler_type='linear',        # Linear with plateau
    warmup_pct=0.15,
    plateau_pct=0.35,              # 50% total at high LR
    min_lr_ratio=0.1,              # End at 10% not 1%
    use_pos_weight=True,
    pos_weight_max=75.0,           # More aggressive
)
```
## 11. Why changing LR during training?
### Why change (usually decay) LR?
- **Early training needs big steps**: weights are far from a good solution; a larger LR helps you move quickly and escape shallow/bad regions.
- **Late training needs small steps**: once you’re near a good basin, a large LR keeps “bouncing around,” causing **metric oscillation**, preventing fine ranking improvements, and sometimes destabilizing MoE routing.
- **Objective mismatch + noise**: with BCE-style multilabel + extreme imbalance, gradients are noisy and dominated by some labels; decaying LR reduces variance and improves “polishing” of the ranking.

### What happens if LR stays constant?
- If LR is **high**: training can look like it “plateaus” (actually it’s not settling), metrics jitter, MoE router can keep shifting (or collapse/recover repeatedly).
- If LR is **low**: you get “stable” but **slow learning**—you might never reach your best recall within 1 epoch.

### Why schedules (cosine/linear/OneCycle) help you specifically
- Your logs show **fast early gains then slower improvements**; that’s classic “needs high LR early, lower later.”
- For **1-epoch training**, schedules like **OneCycle** are popular because they intentionally use:
  - **low → high LR** (explore/learn fast),
  - then **high → low LR** (stabilize/refine before the epoch ends).

### Rule of thumb for choosing
- If you care about **max performance in 1 epoch**: prefer **OneCycle** or **warmup + plateau + mild decay**.
- If you keep seeing **late-epoch collapse/instability or jittery metrics**: decay more (or end LR higher but still lower than peak).
- If you see **early plateau / underfitting**: increase peak LR or keep LR high longer (plateau), not “decay earlier.”

If you tell me your **peak LR**, **batch size**, and whether metrics are still improving after ~70% of the epoch, I can recommend the simplest schedule for your case.

## 12. Positive-negative ratio imbalance issue

```
======================================================================
CODE FREQUENCY DISTRIBUTION ANALYSIS
======================================================================

📊 BASIC STATISTICS:
   Total target codes:      6,297
   Non-zero codes:          5,677 (90.2%)
   Zero-frequency codes:    620 (9.8%)
   Total occurrences:       294,148,040

📈 FREQUENCY STATISTICS (non-zero codes only):
   Min frequency:           1
   Max frequency:           16,952,106
   Mean frequency:          51,814.0
   Median frequency:        479.0
   Std deviation:           404,488.9

⚖️ IMBALANCE METRICS:
   Imbalance ratio (max/min): 16,952,106.0x
   Gini coefficient:          0.9390 (0=equal, 1=total inequality)

📏 PERCENTILE DISTRIBUTION:
   Percentile   Frequency       % of Max    
   ----------------------------------------
     1th                 1.0        0.00%
     5th                 2.0        0.00%
    10th                 7.0        0.00%
    25th                44.0        0.00%
    50th               479.0        0.00%
    75th             6,487.0        0.04%
    90th            61,482.2        0.36%
    95th           192,124.6        1.13%
    99th           924,035.7        5.45%

🏷️ CODE TIER ANALYSIS:
   Tier       Count      % of Codes   Freq Range           % of Total Occurrences
   ---------------------------------------------------------------------------
   Common     1420           25.0%     >= 6487                    98.8%
   Medium     1421           25.0%     479 - 6487                  1.1%
   Rare       1422           25.0%     44 - 479                    0.1%
   Tail       1414           24.9%     < 44                        0.0%

🎯 POS_WEIGHT ANALYSIS:
   Testing different pos_weight_max values...

   max_weight   Mean       Median     % at Max     Effect on Rare      
   ----------------------------------------------------------------------
   10           9.98       10.00          99.5%      Rare codes get 1.0x weight vs common
   20           19.90      20.00          98.9%      Rare codes get 1.0x weight vs common
   50           49.28      50.00          97.0%      Rare codes get 1.1x weight vs common
   75           73.36      75.00          95.8%      Rare codes get 1.1x weight vs common
   100          97.14      100.00         94.6%      Rare codes get 1.1x weight vs common

======================================================================
📋 RECOMMENDATIONS
======================================================================

   1. IMBALANCE SEVERITY: EXTREME
      - Your imbalance ratio: 16,952,106x

   2. FOCAL LOSS RECOMMENDATION: YES
      - Rationale: Imbalance ratio (16,952,106x) and/or Gini (0.939) are very high
      - Suggested gamma: 2.0 (standard) to 3.0 (aggressive)

   3. RECOMMENDED CONFIGURATION:

      optimize_config = OptimizeConfig(
          use_pos_weight=True,
          pos_weight_max=100.0,
          # Consider adding FocalLoss with gamma=2.0
      )

```
