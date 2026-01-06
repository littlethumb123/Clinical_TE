# Progress Report: exp2b Flash Attention + Learned Pooling Analysis
**Date:** December 24, 2025  
**Session Focus:** Comprehensive analysis of exp2b_flash_learned_pool training results  
**Status:** ✅ Complete

---

## 📋 Executive Summary

**Key Achievements:**
- ✅ Analyzed exp2b_flash_learned_pool training results
- ✅ Identified premature loss plateau issue
- ✅ Diagnosed learning rate as too conservative
- ✅ Provided tier-based recommendations for improvement

**Key Outcomes:**
- Current R@10: 79.0% (target: 89-91%)
- Gap of 10-12% from previous best performance
- Root cause: Learning rate decays too aggressively with cosine schedule

**Current Status:** Diagnosis complete, optimization recommendations provided

**Next Steps:** Implement LR schedule improvements

---

## 🎯 Session Overview

### Context at Session Start
- Running Round 5 experiment with `exp2b_flash_learned_pool` configuration
- Flash Attention + Learned Attention Pooling architecture
- Single epoch training on 3 LOBs dataset

### Goals
1. Diagnose potential learning issues
2. Identify root causes for performance gap
3. Recommend optimizations to maximize performance in 1 epoch

---

## 📊 Detailed Technical Work

### Section 1: Current Performance Summary

| Metric | Current (1 Epoch) | Target (Previous Best) | Gap |
|--------|-------------------|------------------------|-----|
| **Recall@10** | **79.0%** | **89-91%** | **-10-12%** |
| **Recall@5** | 65.0% | ~73% | -8% |
| **Recall@20** | 86.0% | ~91% | -5% |
| **MRR** | 0.456 | Higher expected | Significant |
| **micro_recall@10** | 46.7% | Higher expected | Significant |
| **Training Time** | 20,484 sec (~5.7 hrs) | - | - |

---

### Section 2: Loss Dynamics Analysis

#### Loss Trajectory Profile
| Stage | Step Range | Loss Values | Observations |
|-------|-----------|-------------|--------------|
| Initial | 1-10 | ~0.805 | Starting BCE loss |
| Rapid Decay | 10-1000 | 0.805 → 0.05 | Good initial learning |
| Mid Training | 1000-5000 | ~0.015 → 0.005 | Reasonable progress |
| Late Training | 5000-10966 | ~0.0047 plateau | **Stagnation begins** |
| Final | ~10966+ | ~0.003 | **Premature convergence** |

#### 🚨 Key Finding: Premature Loss Plateau

1. **Fast initial convergence** (first ~1000 steps) - healthy
2. **Loss plateaus at ~0.003** by step 10,000 and barely improves
3. **Final loss std = 0.0654** indicates high variance but low mean change
4. **Train-Val gap ≈ -8e-6** (negative!) → model is **underfitting**, not overfitting

---

### Section 3: Root Cause Analysis

#### Primary Issue: Learning Rate Too Conservative

**Current Configuration:**
```
Base LR: 1e-4
Scaled LR (4 GPUs): 4e-4 (linear scaling)
Warmup: 5% of total steps (clamped 100-2000)
Cosine decay to 1% of peak (4e-6)
```

**Problem:** With ~10,966 training steps per epoch and cosine annealing:
- By step 5,500 (~50% through): LR has decayed to ~2e-4
- By step 8,200 (~75% through): LR is ~1e-4
- Final 20% of training has very low LR → **premature convergence**

#### Secondary Issues

| Issue | Evidence | Impact |
|-------|----------|--------|
| Loss plateau too early | Loss at 0.0047 by step 1000 | 90% of epoch barely improving |
| micro_recall@10 at 46.7% | Much lower than recall@10 (79%) | Not learning rare codes well |
| balanced_top10_acc = 19.75% | Imbalanced prediction | Dominated by common codes |

#### Critical Finding: Lower Loss ≠ Better Performance

| Metric | Current (1 ep) | exp3_standard_moe (2 ep) | Difference |
|--------|----------------|--------------------------|------------|
| Recall@10 | 79.0% | 83.0% | -4% |
| Final Loss | 0.00279 | 0.00339 | Lower but worse metrics! |

**Implication:** Model is optimizing wrong thing; loss doesn't correlate with ranking metrics.

---

### Section 4: Recommendations

#### Tier 1: High-Impact Changes

**1. Increase Peak Learning Rate 2-3×**
```python
# Current (too conservative)
learning_rate: float = 1e-4  # Base

# Recommended
learning_rate: float = 2e-4  # Base (2× increase)
```

**2. Use OneCycleLR (Best for 1-Epoch Training)**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=8e-4,  # Peak LR (higher than current)
    total_steps=total_steps,
    pct_start=0.3,  # 30% warmup phase
    anneal_strategy='cos',
    div_factor=25,  # start_lr = max_lr/25
    final_div_factor=1000
)
```

**3. Extend High-LR Phase**
```python
# Warmup → Plateau (30% of training) → Linear decay
plateau_end = int(0.4 * total_steps)  # Stay at peak LR until 40%
```

#### Tier 2: Medium-Impact Changes

**4. Add Focal Loss for Rare Codes**
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
```

**5. Increase Effective Batch Size**
```python
batch_size: int = 64  # Per GPU → effective = 256
accumulation_steps = 2  # Simulates 512 effective batch
```

---

## 💡 Key Insights & Learnings

### Insight 1: Underfitting, Not Overfitting
**Observation:** Negative generalization gap indicates model has capacity to learn more

**Why It Matters:**
- Can safely increase LR without overfitting risk
- Model is leaving performance on the table

**Lesson:** Monitor generalization gap, not just loss

### Insight 2: Single-Epoch Training Needs Special Schedule
**Observation:** Standard cosine decay is too aggressive for 1-epoch training

**Lesson:** Use OneCycleLR or plateau-based schedule for single-epoch runs

---

## 📅 Next Steps & Action Items

### Immediate
1. Implement OneCycleLR scheduler
2. Increase base LR to 2e-4

### Short-term
1. Add Focal Loss or positional weighting for rare codes
2. Experiment with longer plateau phase

---

## ✨ Conclusion

**Session Summary:**
Diagnosed exp2b_flash_learned_pool performance gap (79% vs 89-91% target). Root cause is learning rate decaying too aggressively, causing premature convergence. Provided tiered recommendations with OneCycleLR as top priority.

**Key Takeaway:**
> "The loss plateaus at step 1000 but training continues for 10,000 more steps with diminishing LR - this is wasted compute. Extend high-LR phase for 1-epoch training."

**Current Status:**
Diagnosis complete, ready for optimization implementation.

---

**Author:** AI Assistant  
**Date:** December 24, 2025  

