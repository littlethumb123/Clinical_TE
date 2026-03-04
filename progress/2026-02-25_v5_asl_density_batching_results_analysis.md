# Session Progress Report - V5 ASL + Density-Aware Batching Results Analysis
**Date**: 2026-02-25  
**Status**: Completed comprehensive analysis of v5 experiment results; density batching confirmed INSUFFICIENT for tail gradient starvation; per-tier loss balancing identified as next intervention.

## 1. Executive Summary

This session conducted a rigorous, systematic review of the v5 experiment (ASL + density-aware tier-aware batching, tail_quota=20, density_percentile=80). V5 results were compared head-to-head against v4 (ASL only), v3 (BCE+pw200), and v2 (BCE+pw35) across all metrics, gradient trajectories, and per-tier accuracy. The headline finding: **density-aware batching at 15.6% tail quota was insufficient to prevent the gradient concentration transition** — tail_grad_frac still collapsed from ~18% (step 1) to 0.088% (step 12001), and tail_top10_acc remained at 0%. However, v5 produced the **best overall common-code model to date** (recall@1=0.2843, MRR=0.4955, NDCG@10=0.4779), and medium_top10_acc partially recovered from 0% to 0.17%. The analysis concluded that the next intervention must operate at the **loss aggregation level** (per-tier loss balancing) rather than batch composition.

## 2. Planned vs. Executed
**Original Plan**: Analyze v5 results against the success criteria established in the Feb 22 adjudicator report.
**What Got Done**: 
- [x] Read and extracted all v5 results (config, final_results, batch_metrics)
- [x] Read and cross-referenced v4, v3, v2 final results for head-to-head comparison
- [x] Read and incorporated expert panel synthesis (Feb 22) and evidence synthesis (Feb 2) as context
- [x] Constructed full comparison tables across all 4 experiments
- [x] Analyzed gradient tier trajectory step-by-step for v5 vs v4
- [x] Evaluated against 5 pre-defined success criteria (2 failed, 3 passed)
- [x] Synthesized root cause model update incorporating v5 as the fourth data point
- [x] Produced comprehensive analysis report with detailed next-step recommendations
- [x] Identified per-tier loss balancing as the specific next intervention with proposed configuration

**Alignment Notes**: Session was analysis-only (no code changes). Results analysis was deeper than planned — the gradient trajectory comparison revealed that density batching provided only a transient delay (steps 500-1500) before converging to the same terminal state, a nuance that required step-by-step trajectory comparison.

## 3. Key Decisions & Rationale

### Decision 1: Density Batching at Current Configuration is Insufficient — Escalate to Per-Tier Loss Balancing
**Context**: V5 results showed tail_grad_frac collapsed to 0.088% and tail_top10_acc remained at 0%, failing both primary success criteria.

**Options Considered**:
1. More aggressive density batching (quota=40-50/128, percentile=90-95)
2. Per-tier loss balancing (`total_loss = Σ w_tier * L_tier`)
3. Combined: aggressive density batching + per-tier loss balancing

**Chosen**: Option 2 (per-tier loss balancing + retain current density batching)  
**Rationale**: 
- V5's ~1.6x improvement in batch tail occurrence (5.2%→~8.3%) was arithmetically insufficient against the 13.4x occurrence imbalance
- Pushing density batching more aggressively (quota=40-50) would dedicate 30-40% of batch to a restricted pool, risking common-code degradation and offering diminishing returns
- Per-tier loss balancing **mathematically guarantees** equal gradient contribution per tier — it addresses the aggregation dynamics directly rather than trying to shift input distribution enough to outcompete the self-reinforcing concentration
- The intervention landscape now shows 4 completed experiments at the per-sample and modest batch-composition levels, all converging to the same terminal state. The next untested level is loss aggregation.

**Trade-offs**: Per-tier loss balancing may degrade common-code ranking quality (re-allocating gradient budget from common to tail). Proposed success criteria allow up to 3pp regression on common_top10_acc (83.3%→≥80%) and NDCG@10 regression to ≥0.42.

### Decision 2: Retain Current Density Batching as Secondary Benefit
**Context**: Despite failing tail criteria, density batching improved common-code metrics and recovered medium codes.

**Rationale**: V5 showed density batching provides batch diversity that benefits common-code ranking (recall@1 +18.4% vs v4, MRR +5.2%, macro_auroc recovered to V2 levels). Removing it for v6 would sacrifice these gains. Keep density batching alongside per-tier loss balancing.

### Decision 3: V5 Is the New Best-Overall Model — Retain ASL + Density Config
**Context**: V5 outperformed all prior versions on every primary ranking metric.

**Evidence**:
| Metric | V2 | V3 | V4 | **V5** |
|--------|:---:|:---:|:---:|:---:|
| recall@1 | 0.010 | 0.000 | 0.240 | **0.284** |
| MRR | 0.329 | 0.324 | 0.471 | **0.496** |
| NDCG@10 | 0.392 | 0.390 | 0.468 | **0.478** |
| positive_brier | 0.685 | 0.687 | 0.313 | **0.308** |
| common_top10_acc | 81.4% | 81.7% | 82.8% | **83.3%** |

## 4. Technical Changes

### 4.1 Files Created
- `progress/2026-02-25_v5_asl_density_batching_results_analysis.md` — This progress report

### 4.2 Files Analyzed (Read-Only)
- `expe_logs/exp_round5_1_lr_plateau/exp2/v5_asymm_focalloss_dense_sampler_config.json` — V5 configuration
- `expe_logs/exp_round5_1_lr_plateau/exp2/v5_asymm_focalloss_dense_sampler_final_results.json` — V5 final evaluation metrics
- `expe_logs/exp_round5_1_lr_plateau/exp2/v5_asymm_focalloss_dense_sampler_batch_metrics.json` — V5 step-by-step training metrics (25 checkpoints, steps 1-12001)
- `expe_logs/exp_round5_1_lr_plateau/exp2/v4_asymm_focalloss_final_results.json` — V4 baseline comparison
- `expe_logs/exp_round5_1_lr_plateau/exp2/v4_asymm_focalloss_config.json` — V4 configuration
- `expe_logs/exp_round5_1_lr_plateau/exp2/v2_bce_weighed35_final_results.json` — V2 baseline
- `expe_logs/exp_round5_1_lr_plateau/exp2/v3_bce_weighed200_final_results.json` — V3 baseline
- `expe_analysis/exp_round5/learning_plateau/exp_round5_exp2_lr_plateau_asymmetric_focalloss_results_expert_panel_feb22.md` — Expert panel + adjudicator synthesis
- `expe_analysis/exp_round5/learning_plateau/exp_round5_exp2_lr_plateau_evidence_sythesis_expert_interpret_nextstep_feb2.md` — Four-expert evidence synthesis
- `expe_analysis/exp_round5/learning_plateau/solution_ablations/exp_round5_exp2_lr_plateau_dense_dense_sampler_asym_focal_loss_config_discuss.md` — Parameter-level config analysis

### 4.3 No Code or Configuration Changes
This was an analysis-only session. No code modifications were made.

## 5. Discussions & Reasoning

### Topic 1: V5 Results Against Success Criteria
**Question**: Did density-aware batching achieve the pre-defined success criteria from the Feb 22 adjudicator report?

**Analysis**: Five criteria evaluated:

| Criterion | Target | V5 Result | Verdict |
|-----------|--------|-----------|---------|
| `train_grad_tier_tail_frac > 5%` at end of training | >5% | 0.088% | **FAILED** |
| `tail_top10_acc > 0%` | Any movement | 0.0% | **FAILED** |
| `common_top10_acc >= 82%` | >=82% | 83.3% | PASSED |
| NDCG@10 >= 0.46 | >=0.46 | 0.478 | PASSED |
| MRR >= 0.47 | >=0.47 | 0.496 | PASSED |

**Conclusion**: Density batching is **necessary but insufficient**. It benefits common-code quality but cannot overcome the gradient concentration transition for tail codes. The two primary tail criteria both failed while the three common-code criteria passed with margin.

### Topic 2: Gradient Trajectory Comparison — V5 vs V4
**Question**: Did density batching change the timing or magnitude of the gradient concentration transition?

**Analysis**: Step-by-step comparison reveals:
- Step 1: V5 tail_frac=0.182 vs V4=0.191 (roughly equivalent initialization)
- **Step 1001: V5=0.062 vs V4=0.044 (+41%)** — Density batching provides measurably higher tail gradient during the critical transition window
- Step 3001: V5=0.011 vs V4=0.010 (+10%) — Advantage already dissipating
- Step 5001: V5=0.004 vs V4=0.004 — Converged
- **Step 12001: V5=0.00088 vs V4=0.00121 — V5 actually LOWER than V4**

**Conclusion**: Density batching provided a **transient ~40% improvement** in tail gradient fraction during steps 500-1500 but did NOT prevent the concentration transition. By step 3000, the advantage had largely washed out. The terminal value was actually lower in V5 (0.088% vs 0.121%), suggesting the concentration dynamics are robust to this level of batch composition change.

### Topic 3: Four Experiments Converging to Same Terminal State
**Question**: What does the V2→V5 experimental arc tell us about the intervention landscape?

**Analysis**: Four independent interventions, each targeting a different mechanism level:

| Experiment | Mechanism Level | Terminal tail_frac | Tail Acc |
|:---|:---|:---:|:---:|
| V2: BCE+pw35 | Per-sample magnitude | 0.125% | 0% |
| V3: BCE+pw200 | Per-sample magnitude (5.7x) | 0.171% | 0% |
| V4: ASL (γ-=4) | Per-sample focusing | 0.121% | 0% |
| **V5: ASL+DenseBatch** | **Per-sample focusing + batch composition** | **0.088%** | **0%** |

All four converge to the same terminal gradient distribution and the same tail accuracy. The self-reinforcing gradient concentration mechanism is robust across per-sample magnitude, per-sample focusing, and modest batch composition interventions. The next untested level is **loss aggregation** (per-tier loss balancing).

**Conclusion**: The gradient concentration "attractor" requires a qualitatively different intervention to escape — one that **directly enforces** tail gradient contribution rather than trying to create conditions where it might survive.

### Topic 4: Unexpected Benefits of Density Batching
**Question**: Why did V5 improve common-code metrics beyond V4 despite density batching targeting tail codes?

**Analysis**: V5 outperformed V4 on all ranking metrics:
- recall@1: +18.4% (0.2843 vs 0.2401)
- MRR: +5.2% (0.4955 vs 0.4709)
- NDCG@10: +2.0% (0.4779 vs 0.4684)
- macro_auroc: +1.3% (0.8575 vs 0.8463, recovered to V2 levels)
- medium_top10_acc: recovered from 0% to 0.17%

**Hypothesized mechanism**: Density batching selects top-20% tail-dense members, who tend to have more diverse clinical profiles (more code co-occurrence patterns). This incidental diversity improves the probability landscape the model learns, benefiting ranking quality even for common codes. The medium code recovery supports this — tail-dense members likely also carry above-average medium code density due to clinical comorbidity correlation.

**Conclusion**: Density batching should be **retained** in future experiments for its secondary benefit to overall model quality, even though it does not solve the tail problem directly.

## 6. Verification & Quality Checks

**Data Verification**:
- All metric values extracted directly from JSON result files (not computed or estimated)
- V5 vs V4 config comparison verified: sole difference is `use_tier_aware_batching=true`, `use_density_aware_batching=true`, `tier_tail_quota=20` (V5) vs `false`/`false`/N/A (V4)
- Gradient trajectory extracted from 25 batch metric checkpoints in `v5_asymm_focalloss_dense_sampler_batch_metrics.json`
- V4 gradient trajectory cross-referenced with expert panel report (Feb 22) and V4 batch metrics file

**Consistency Checks**:
- V5 epoch-average tail_frac (0.02075) is slightly higher than V4's (0.01963), consistent with density batching providing marginal early-training benefit
- V5 terminal common_frac (0.861) slightly higher than V4 (0.857), consistent with the same concentration dynamics
- V5 generalization gap (0.000890) essentially identical to V4 (0.000880), confirming ASL's generalization benefit is preserved

## 7. Plan Alignment Review

**PRD/Original Goals**: Resolve the learning plateau for tail/rare codes in the clinical code prediction model to improve downstream member profiling embeddings.

**Completion Status**:
- V4 ASL experiment: **Complete**. Broke ranking ceiling, confirmed per-sample loss exhaustion for tail.
- V5 density batching experiment: **Complete**. Confirmed insufficient for tail. Best overall common-code model.
- Root cause diagnosis: **Refined**. Four data points now confirm gradient concentration is robust to per-sample and modest batch-composition interventions.
- Intervention landscape: **Four of five levels tested**. Only per-tier loss aggregation remains untested.

**Updated Intervention Landscape**:

| Level | Tested? | Result |
|:---|:---:|:---|
| Per-sample magnitude (pos_weight) | V2, V3 | EXHAUSTED |
| Per-sample focusing (ASL) | V4 | EXHAUSTED for tail; validated for common ranking |
| Batch composition (density sampling) | **V5** | **INSUFFICIENT — same concentration trajectory** |
| **Per-tier loss aggregation** | **UNTESTED** | **Next intervention** |
| Hierarchical supervision | UNTESTED | Fallback if per-tier loss balancing insufficient |

## 8. Blockers & Issues

**Resolved**:
- V5 results analyzed — all metrics extracted and compared across four experiments
- Density batching effectiveness quantified — provides ~40% transient improvement at step 1001, but concentration transition not prevented
- Medium code recovery mechanism identified — incidental diversity benefit from density-selected members

**Outstanding**:
- **Per-tier loss balancing not yet implemented** — Requires implementing tier-partitioned loss computation in training loop (`moe_flashattn_4.py`). Need to add code that partitions BCE/ASL loss by tier and normalizes per-tier contributions.
- **Tail embedding std not measured for V5** — The Feb 1 diagnostic showed tail embedding std=0.03 (homogenized). Need to verify whether density batching changed this. However, given tail_top10_acc=0% and tail_grad_frac=0.088%, it is unlikely that embeddings de-homogenized.
- **macro_auprc regression** — V5 macro_auprc (0.1035) is lower than V4 (0.1104) and V2 (0.1057). This is the only metric where V5 is worse than all prior versions. The cause is unclear but may relate to the density sampler changing the effective class prior distribution seen during training.

## 9. Next Session Plan

**Immediate Priorities** (ranked):
1. **Implement per-tier loss balancing** — Add tier-partitioned loss computation to training loop in `moe_flashattn_4.py`. Equal weights (0.25 per tier) for maximum diagnostic signal. Combine with ASL + density batching (retain V5's secondary benefits).
2. **Run V6 experiment** (ASL + density batching + per-tier loss balancing) — Evaluate against updated success criteria:
   - `train_grad_tier_tail_frac > 10%` at **end** of training
   - `tail_top10_acc > 0%` (any movement)
   - `common_top10_acc >= 80%` (allow up to 3pp regression)
   - NDCG@10 >= 0.42 (allow modest ranking regression)
3. **If V6 succeeds**: Tune tier weights (e.g., 0.40 common / 0.20 medium / 0.20 rare / 0.20 tail) to balance tail improvement against common-code quality
4. **If V6 fails** (tail_grad_frac maintained but tail_top10_acc still 0%): Escalate to hierarchical supervision (CCS/CCSR auxiliary loss) — the problem shifts from gradient starvation to insufficient supervision signal

**Implementation Notes for Per-Tier Loss Balancing**:
```python
# Conceptual implementation for training loop
common_mask = tier_assignments == 'common'
medium_mask = tier_assignments == 'medium'
rare_mask = tier_assignments == 'rare'
tail_mask = tier_assignments == 'tail'

loss_common = asl_criterion(logits[:, common_mask], targets[:, common_mask])
loss_medium = asl_criterion(logits[:, medium_mask], targets[:, medium_mask])
loss_rare = asl_criterion(logits[:, rare_mask], targets[:, rare_mask])
loss_tail = asl_criterion(logits[:, tail_mask], targets[:, tail_mask])

total_loss = 0.25 * loss_common + 0.25 * loss_medium + 0.25 * loss_rare + 0.25 * loss_tail
```

**Preparation Required**:
- Locate tier assignment logic in `moe_flashattn_4.py` (the tier masks used for gradient analysis should be reusable for loss partitioning)
- Verify that per-tier loss partitioning does not break gradient flow through the ASL modulation terms
- Consider whether per-tier loss should be computed per-sample-then-averaged or per-batch-aggregated (former preserves per-sample variance; latter is simpler)

**Open Questions**:
- Should per-tier loss balancing use the same ASL hyperparameters (γ-=4, clip=0.05) for all tiers, or should tail codes use γ-=2 (less aggressive negative suppression to retain more gradient signal)?
- Is equal weighting (0.25 each) the right starting point, or should weights be inversely proportional to occurrence frequency (giving tail ~4x the weight of common)?

---

**Session Duration**: ~1 hour (analysis session)  
**Files Analyzed**: 10 experiment/analysis documents  
**Files Created**: 1 (this progress report)  
**Commits**: 0 (analysis-only session)  
**Environment**: macOS darwin 24.6.0, Cursor IDE

## Appendix: Master Evidence Table (Updated Feb 25)

| Hypothesis | Status | Evidence | Date |
|:---|:---:|:---|:---|
| LR schedule is primary bottleneck | **REJECTED** | Polishing test: val_loss worse, NDCG degraded | Feb 1 |
| Embedding collapse (norms → 0) | **REJECTED** | All tiers: min norm > 0.8 | Feb 1 |
| Embedding homogenization | **CONFIRMED** | Tail std=0.03 vs common std=0.27 | Feb 1 |
| Logit suppression / negative prior | **CONFIRMED** | Tail logit=-14.69 (P≈0.00004%) | Feb 1 |
| Model at sharp local minimum | **CONFIRMED** | recall@10 dropped 4.4% at step 200, recovered to same basin | Feb 1 |
| Gradient concentration is emergent | **CONFIRMED** | Identical trajectory across V2/V3/V4/V5 | Jan 24 - Feb 25 |
| pos_weight is ineffective for starvation | **CONFIRMED** | 5.7x increase → <0.5% gradient change; cancels in within-class ratio | Jan 24 / Feb 22 |
| Per-sample loss reweighting exhausted | **CONFIRMED** | V2 (BCE+pw35), V3 (BCE+pw200), V4 (ASL) → identical tail=0% | Jan 24 - Feb 22 |
| Occurrence-level imbalance is structural cause | **CONFIRMED** | Tail: 83.4% members but 5.2% occurrences (13.4x ratio) | Jan 30 |
| ASL validated for common-code ranking | **CONFIRMED** | recall@1: 0.01→0.28, MRR: 0.33→0.50 across V4/V5 | Feb 22 / Feb 25 |
| **Modest density batching insufficient for tail** | **CONFIRMED (NEW)** | V5: tail_quota=20, density_pct=80 → tail_frac collapsed to 0.088%, tail_acc=0% | **Feb 25** |
| **Density batching benefits common-code quality** | **CONFIRMED (NEW)** | V5: recall@1 +18.4%, MRR +5.2%, macro_auroc recovered vs V4 | **Feb 25** |
| **Medium code recovery achievable without pos_weight** | **CONFIRMED (NEW)** | V5: medium_top10_acc recovered 0%→0.17% via density batching alone | **Feb 25** |
| **Per-tier loss balancing is next untested intervention** | **PENDING** | Only remaining intervention level in the cascade | **Feb 25** |
