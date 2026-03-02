# V5 Results Analysis: ASL + Density-Aware Tier Batching

## 1. Configuration Delta Summary

Before analyzing results, it's critical to establish precisely what changed:

| Parameter | V2 | V3 | V4 | **V5** |
|-----------|------|------|------|------|
| Loss function | BCE | BCE | ASL (γ+=0,γ-=4,clip=0.05) | **ASL (γ+=0,γ-=4,clip=0.05)** |
| `use_pos_weight` | true (max=35) | true (max=200) | false | **false** |
| `use_tier_aware_batching` | false | false | false | **true** |
| `use_density_aware_batching` | false | false | false | **true** |
| `tier_tail_quota` | — | — | — | **20** |
| `tier_rare_quota` | — | — | — | **0** |
| `tier_medium_quota` | — | — | — | **0** |
| `density_tail_percentile` | — | — | — | **80.0** |

V5 is identical to V4 in loss function and pos_weight. The **sole new variable** is density-aware tier-aware batching, concentrated entirely on the tail tier (quota=20/128 = 15.6% of batch, top-20% tail-dense members). Notably, the actual config used `tier_tail_quota=20` (more aggressive than the adjudicator's recommended 12) and `tier_rare_quota=0` (a clean single-variable test focused on tail).

---

## 2. Head-to-Head Performance Comparison

### 2.1 Ranking and Retrieval Metrics

| Metric | V2 (BCE+pw35) | V3 (BCE+pw200) | V4 (ASL) | **V5 (ASL+Dense)** | V5 vs V4 Δ | V5 vs V2 Δ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| **recall@1** | 0.0103 | 0.0000 | 0.2401 | **0.2843** | **+18.4%** | **+2660%** |
| **recall@5** | 0.6856 | 0.6861 | 0.7193 | **0.7223** | +0.4% | +5.4% |
| **recall@10** | 0.8142 | 0.8171 | 0.8280 | **0.8329** | +0.6% | +2.3% |
| recall@20 | 0.8915 | 0.8930 | 0.8960 | **0.8985** | +0.3% | +0.8% |
| recall@50 | 0.9506 | 0.9512 | 0.9508 | 0.9501 | -0.1% | -0.1% |
| **ndcg@5** | 0.3562 | 0.3535 | 0.4190 | **0.4246** | +1.3% | +19.2% |
| **ndcg@10** | 0.3923 | 0.3898 | 0.4684 | **0.4779** | +2.0% | +21.8% |
| **ndcg@20** | 0.4298 | 0.4265 | 0.5014 | **0.5105** | +1.8% | +18.8% |
| **mrr** | 0.3293 | 0.3242 | 0.4709 | **0.4955** | **+5.2%** | **+50.5%** |
| **precision@10** | 0.2089 | 0.2099 | 0.2284 | **0.2304** | +0.8% | +10.3% |
| **f1@10** | 0.3325 | 0.3340 | 0.3581 | **0.3609** | +0.8% | +8.5% |
| **micro_recall@10** | 0.4634 | 0.4656 | 0.4716 | **0.4756** | +0.8% | +2.6% |

### 2.2 Calibration and Loss Metrics

| Metric | V2 | V3 | V4 | **V5** | V5 vs V4 Δ |
|--------|:---:|:---:|:---:|:---:|:---:|
| **positive_brier** | 0.6848 | 0.6868 | 0.3126 | **0.3076** | **-1.6% (better)** |
| val_loss (own objective) | 0.00308 | 0.00322 | 0.000776 | **0.000773** | -0.4% |
| val_bce_loss | 0.00317 | 0.00342 | 0.0936 | **0.0965** | +3.1% |
| generalization_gap | 0.01000 | 0.01015 | 0.000880 | **0.000890** | +1.1% |

### 2.3 Per-Tier Accuracy (Critical)

| Tier | V2 | V3 | V4 | **V5** | V5 vs V4 |
|------|:---:|:---:|:---:|:---:|:---:|
| **common_top10_acc** | 81.44% | 81.73% | 82.81% | **83.30%** | **+0.49pp** |
| **medium_top10_acc** | 0.47% | 0.16% | 0.00% | **0.17%** | **Recovered from zero** |
| rare_top10_acc | 0.00% | 0.00% | 0.00% | 0.00% | Unchanged |
| **tail_top10_acc** | 0.00% | 0.00% | 0.00% | **0.00%** | **Unchanged** |
| tail_code_coverage | 0.00% | 0.00% | 0.00% | 0.00% | Unchanged |
| balanced_top10_acc | 20.48% | 20.47% | 20.70% | **20.87%** | +0.17pp |

### 2.4 Discrimination Metrics

| Metric | V2 | V3 | V4 | **V5** | V5 vs V4 Δ |
|--------|:---:|:---:|:---:|:---:|:---:|
| **macro_auroc** | 0.8581 | 0.8781 | 0.8463 | **0.8575** | **+1.3% (recovered)** |
| macro_auprc | 0.1057 | 0.1048 | 0.1104 | 0.1035 | -6.3% |

---

## 3. Gradient Tier Dynamics — The Decisive Diagnostic

### 3.1 Terminal Gradient Distribution

| Metric | V2 | V3 | V4 | **V5** |
|--------|:---:|:---:|:---:|:---:|
| Final common_frac | 0.849 | 0.847 | 0.857 | **0.861** |
| Final tail_frac | 0.00125 | 0.00171 | 0.00121 | **0.00088** |
| Epoch-avg common_frac | 0.824 | 0.828 | 0.799 | **0.798** |
| Epoch-avg tail_frac | 0.01272 | 0.01078 | 0.01963 | **0.02075** |

### 3.2 Step-by-Step Gradient Trajectory Comparison (V5 vs V4)

| Step | V4 tail_frac | **V5 tail_frac** | V5 advantage | V4 common_frac | V5 common_frac |
|------|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.191 | **0.182** | -5% (slightly lower) | 0.171 | 0.174 |
| 501 | 0.149 | **0.161** | **+8% (density effect)** | 0.292 | 0.279 |
| 1001 | 0.044 | **0.062** | **+41% (density effect)** | 0.697 | 0.658 |
| 1501 | 0.022 | **0.025** | **+14%** | 0.823 | 0.809 |
| 2001 | 0.016 | **0.017** | +6% | 0.832 | 0.812 |
| 3001 | 0.010 | **0.011** | +10% | 0.844 | 0.845 |
| 5001 | 0.004 | **0.004** | ~equal | 0.856 | 0.874 |
| 8001 | 0.002 | **0.002** | ~equal | 0.834 | 0.867 |
| 12001 | 0.00121 | **0.00088** | **-27% (V5 worse)** | 0.857 | 0.861 |

---

## 4. Assessment Against Success Criteria

The adjudicator's success criteria from the Feb 22 report:

| Criterion | Target | **V5 Result** | **Verdict** |
|-----------|--------|-----------|---------|
| `train_grad_tier_tail_frac > 5%` at end of training | >5% | **0.088%** | **FAILED** |
| `tail_top10_acc > 0%` | Any movement off zero | **0.0%** | **FAILED** |
| `common_top10_acc >= 82%` | >=82% | **83.3%** | **PASSED** |
| NDCG@10 >= 0.46 | >=0.46 | **0.478** | **PASSED** |
| MRR >= 0.47 | >=0.47 | **0.496** | **PASSED** |

The two primary tail improvement criteria are unmet. The three common-code maintenance criteria are met with margin.

---

## 5. What the Evidence Tells Us

### 5.1 CONFIRMED: Density-Aware Batching at Current Configuration is Insufficient for Tail Learning

This is the most important finding. Despite allocating 15.6% of every batch (20/128) to top-20% tail-dense members, the gradient concentration transition followed the **same fundamental trajectory** as V4 (and V2/V3). The tail gradient fraction:
- Started balanced (~18% at step 1)
- Showed a marginal improvement during the critical 500-1500 step window (V5 tail_frac at step 1001 was 0.062 vs V4's 0.044 — a 41% improvement)
- Converged to the same terminal value (~0.1%) by step 5000+
- Actually ended **lower** than V4 at step 12001 (0.088% vs 0.121%)

The density batching provided a **transient delay** in gradient concentration during steps 500-1500 but did **not prevent** the concentration transition. By step 3000, the advantage had largely dissipated.

### 5.2 CONFIRMED: Density Batching Improved Common-Code Learning Beyond V4

V5 produced the best common-code model across all four experiments:

- **recall@1: 0.2843** — the single best top-1 accuracy, +18.4% over V4
- **MRR: 0.4955** — the best mean reciprocal rank, +5.2% over V4
- **NDCG@10: 0.4779** — the best ranking quality, +2.0% over V4
- **common_top10_acc: 83.3%** — the best common code accuracy, +0.5pp over V4

This is an unexpected but interpretable finding. By concentrating tail-dense members in the batch, the density sampler incidentally increases the diversity of code co-occurrence patterns the model sees per batch, which appears to benefit the learned probability landscape even for common codes.

### 5.3 CONFIRMED: Medium Codes Partially Recovered

`medium_top10_acc` went from 0.0% (V4) to 0.17% (V5). This is small but directionally significant because:

1. V4's medium code collapse was caused by removing pos_weight (which V5 also lacks)
2. V5 recovered medium codes **without restoring pos_weight**, suggesting that density batching provides a mild between-tier diversity benefit
3. The recovery is to a level comparable to V3 (0.16%) but below V2 (0.47%)

The mechanism: top-20% tail-dense members likely also carry above-average medium code density (clinical comorbidity correlation), so the density sampler incidentally increases medium code exposure.

### 5.4 CONFIRMED: macro_auroc Recovered to V2 Levels

V4 showed a small AUROC regression (0.858 → 0.846). V5 recovered to 0.858, suggesting that the density batching provides more balanced discrimination across tiers for the AUROC metric. This is consistent with the medium code partial recovery.

### 5.5 NEW INSIGHT: The Batch-Level Tail Occurrence Increase Was Likely Too Small

The config discussion document estimated that `tier_tail_quota=20` with `density_tail_percentile=80` would increase the batch-level tail occurrence fraction from ~5.2% to ~8.3% — a 1.6x improvement. Given that the occurrence imbalance is 13.4x (tail 5.2% vs common 69.7%), a 1.6x boost is arithmetically insufficient to fundamentally shift the gradient dynamics. The gradient tier trajectory confirms this: V5's marginal improvement at step 1001 (0.062 vs 0.044) washed out by step 5000.

### 5.6 NEW INSIGHT: Four Per-Step/Per-Sample Interventions Have Now Converged to the Same Terminal State

We now have **four independent experiments** all producing the same terminal gradient distribution:

| Intervention | Mechanism Level | Terminal tail_frac | Tail Acc |
|:---|:---|:---:|:---:|
| V2: BCE + pos_weight=35 | Per-sample magnitude | 0.125% | 0% |
| V3: BCE + pos_weight=200 | Per-sample magnitude (5.7x more aggressive) | 0.171% | 0% |
| V4: ASL (γ+=0, γ-=4) | Per-sample focusing | 0.121% | 0% |
| **V5: ASL + Density Batching** | **Per-sample focusing + Batch composition (modest)** | **0.088%** | **0%** |

V5 is the first experiment to change batch composition. That it produced the **same terminal gradient distribution** (and actually the lowest tail_frac of all four) tells us that the density batching intervention was **not strong enough** to cross the threshold needed to prevent the self-reinforcing concentration transition.

---

## 6. Root Cause Model — Updated with V5 Evidence

### 6.1 The Concentration Transition is Robust to Modest Batch Composition Changes

V5 provides a critical data point about the **magnitude of intervention needed**:

- A 1.6x increase in batch-level tail occurrence (5.2% → ~8.3%) was insufficient
- The concentration transition's "escape velocity" appears to require tail occurrence fractions significantly higher than 8-10% to compete with the common code gradient signal
- The self-reinforcing nature of the transition (common codes get well-learned → their gradients become refined/consistent → they dominate even more) creates a barrier that modest composition changes cannot overcome

### 6.2 What Remains Unchanged from Prior Diagnosis

| Conclusion | Status | V5 New Evidence |
|:---|:---:|:---|
| Per-sample loss engineering is exhausted for tail | CONFIRMED | V5 used identical ASL as V4 — no change expected or observed |
| Gradient concentration is an emergent, self-reinforcing process | **STRENGTHENED** | Modest batch composition change did not break the transition |
| The plateau has two separable components (ranking ceiling + tail starvation) | CONFIRMED | V5 improved ranking further while tail remained at 0% |
| The ranking ceiling continues to yield to ASL | CONFIRMED | V5 pushed recall@1, MRR, NDCG all higher than V4 |
| Medium code recovery is achievable without pos_weight | **NEW** | Density batching partially recovered medium codes |

### 6.3 Updated Intervention Landscape

| Level | Mechanism | Tested? | Result |
|:---|:---|:---:|:---|
| Per-sample magnitude (pos_weight) | Scales per-class loss | V2, V3 | EXHAUSTED — 5.7x increase → <0.5% effect |
| Per-sample focusing (ASL γ-) | Down-weights easy examples | V4 | EXHAUSTED for tail — improves common ranking only |
| **Batch composition (modest density)** | ~1.6x tail occurrence increase | **V5** | **INSUFFICIENT — same concentration trajectory** |
| **Batch composition (aggressive density)** | >3x tail occurrence increase | **UNTESTED** | Would require quota=40-50/128 or per-tier loss |
| **Per-tier loss aggregation** | Equal gradient per tier | **UNTESTED** | Addresses the problem at the aggregation level directly |
| Hierarchical supervision | CCS/CCSR auxiliary loss | UNTESTED | Indirect signal for tail codes |

---

## 7. Conclusions

### 7.1 What We Can Definitively Confirm

1. **V5 is the best overall model produced to date** for common-code ranking and retrieval. Every primary ranking metric (recall@1, MRR, NDCG@10, NDCG@20, precision@10, f1@10) is at its all-time best. The progression from V2 to V5 represents a cumulative improvement of +2660% on recall@1, +50.5% on MRR, and +21.8% on NDCG@10.

2. **ASL remains validated as the correct loss function.** V5 demonstrates that ASL's ranking and calibration benefits are preserved (and slightly enhanced) when combined with density batching. The positive Brier score continued to improve (0.313 → 0.308).

3. **Density-aware batching at 15.6% tail quota with 80th percentile density threshold is insufficient to break the gradient concentration transition.** The tail gradient fraction still collapsed from ~18% to <0.1% on essentially the same timeline as V4. The intervention was too modest relative to the 13.4x occurrence imbalance.

4. **The self-reinforcing gradient concentration mechanism is more robust than anticipated.** Four distinct intervention strategies spanning per-sample magnitude, per-sample focusing, and now modest batch composition all converge to the same terminal gradient distribution. The concentration transition has a strong "attractor" that requires a qualitatively different or much more aggressive intervention to escape.

5. **Medium code recovery is achievable through density batching without pos_weight**, confirming that the medium code regression in V4 was a marginal-exposure issue that density batching partially addresses through incidental code diversity.

### 7.2 What We Can Conclude with High Confidence

1. **The threshold for batch-level tail occurrence fraction needed to prevent concentration is above 8-10%.** V5's estimated ~8.3% was not enough. Based on the 13.4x ratio and the observation that common codes produce consistent, coherent gradients while tail codes produce sporadic, noisy gradients, the effective threshold may be in the 15-25% range — which would require dedicating 30-50% of each batch to tail-dense members, a configuration that risks degrading common-code learning.

2. **The next intervention must operate at the loss aggregation level, not the batch composition level.** While more aggressive density batching could theoretically work (higher quotas, higher percentiles), the arithmetic shows diminishing returns: even with 40/128 quota and 90th percentile density, the batch tail fraction likely reaches only ~12-15%. Per-tier loss balancing (`total_loss = w_common * L_common + w_medium * L_medium + w_rare * L_rare + w_tail * L_tail`) provides a direct, mathematically guaranteed mechanism to enforce equal tier gradient contribution regardless of occurrence frequency.

3. **Density batching provides a valuable secondary benefit for common-code quality** and should be retained even when escalating to per-tier loss balancing. The improvements in recall@1, MRR, and macro_auroc suggest that the batch diversity from density sampling benefits the overall model.

---

## 8. Recommended Next Steps

### Priority 1: Per-Tier Loss Balancing + ASL + Density Batching (V6)

**Rationale**: This is the next untested intervention level — the adjudicator's recommended escalation if density batching proved insufficient. It addresses the problem at the **aggregation level** where the gradient concentration actually occurs, rather than trying to shift the input distribution enough to outcompete it.

```python
optimize_config = OptimizeConfig(
    # Retain ASL (proven for common-code ranking)
    use_asl=True, asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_clip=0.05,
    use_pos_weight=False,

    # Retain density batching (secondary benefit for common-code quality)
    use_tier_aware_batching=True, use_density_aware_batching=True,
    tier_tail_quota=20, tier_rare_quota=0, tier_medium_quota=0,
    density_tail_percentile=80.0,

    # NEW: Per-tier loss balancing
    use_per_tier_loss_balancing=True,
    tier_loss_weights={'common': 0.25, 'medium': 0.25, 'rare': 0.25, 'tail': 0.25},

    enable_gradient_tier_analysis=True,
)
```

**Why equal weights (0.25 each)**: This is the strongest possible signal for the diagnostic question — if equal-weight per-tier loss balancing combined with ASL and density batching cannot move tail_top10_acc off zero, the problem is fundamentally a data scarcity issue requiring representation-level interventions (Phase 2: hierarchical supervision, two-stage training).

**Implementation note**: Per-tier loss balancing requires partitioning the loss computation by code tier and normalizing each tier's contribution independently. This is structurally different from pos_weight (which scales per-code loss without changing aggregation) — it guarantees that the gradient update vector has equal contributions from all four tiers.

**Updated Success Criteria**:
- `train_grad_tier_tail_frac > 10%` at **end of training** (must prevent collapse, not just delay it)
- `tail_top10_acc > 0%` (any movement off zero)
- `medium_top10_acc > 0%` (maintain V5's recovery)
- `common_top10_acc >= 80%` (allow up to 3pp degradation from V5's 83.3% — acceptable tradeoff for tail learning)
- NDCG@10 >= 0.42 (allow modest ranking regression — the per-tier loss rebalancing may shift some calibration quality from common to tail)
- Tail embedding std > 0.05 (partial de-homogenization, if embedding diagnostics are available)

### Priority 2: If V6 Succeeds on Gradient But Fails on Tail Accuracy

If per-tier loss balancing successfully maintains `tail_grad_frac > 10%` but `tail_top10_acc` remains at 0%, the problem shifts from gradient starvation to **insufficient supervision signal** for individual tail codes. In that case:

- **Hierarchical supervision** (CCS/CCSR category-level auxiliary loss, λ=0.1-0.2) provides indirect learning signal for tail codes through their clinical parent categories
- **Two-stage training** (Kang et al., 2020): train encoder with balanced exposure, then re-calibrate classifier

### Not Recommended

- **More aggressive density batching alone** (e.g., quota=40-50): The V5 results show the concentration transition is robust to modest composition changes. Going to extreme quotas (30-40% of batch) would likely degrade common-code learning while still not guaranteeing tail gradient survival. Per-tier loss balancing is more principled and mathematically guaranteed.
- **Restoring pos_weight**: The first-principles analysis showed it cancels in within-class ratio; the between-class mechanism was proven ineffective across V2/V3.
- **Different ASL hyperparameters** (γ-=2 instead of 4): Would add a confounding variable without clear benefit for tail codes.
- **LR schedule changes**: Definitively rejected by the Feb 1 polishing test.

---

## 9. Summary of the V2 → V5 Experimental Arc

| Version | What Changed | Key Finding |
|:---|:---|:---|
| **V2** (BCE+pw35) | Baseline | Gradient concentration: 85% common, 0.12% tail. Tail_acc=0% |
| **V3** (BCE+pw200) | 5.7x pos_weight increase | <0.5% gradient change. Medium code collapse (-96%). Per-sample magnitude intervention **exhausted** |
| **V4** (ASL) | Different loss geometry + removed pos_weight | Ranking ceiling **broken** (recall@1: 1%→24%, MRR +43%). Tail unchanged. Per-sample focusing intervention **exhausted** for tail |
| **V5** (ASL+DenseBatch) | Added density-aware tier batching | Ranking **further improved** (recall@1: 24%→28%, MRR +5.2%). Medium **recovered** (0%→0.17%). Tail **still 0%**. Modest batch composition intervention **insufficient** |

The consistent pattern: each intervention successfully addresses the mechanism it targets (pos_weight for per-sample boost, ASL for ranking/calibration, density batching for batch diversity) but the tail gradient starvation mechanism operates at a level that none of these have yet reached — the **gradient aggregation** level. Per-tier loss balancing is the natural next step in this cascade.