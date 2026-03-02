# Session Progress Report — Learning Plateau Investigation (Exp Round 5, Exp2)
**Date Range**: 2026-01-19 to 2026-02-02  
**Status**: Root cause confirmed as gradient starvation → embedding homogenization; LR schedule hypothesis eliminated; structural interventions (density-based sampling + focal loss) identified as highest-priority next steps.

---

## 1. Executive Summary

Over a two-week investigation, a systematic diagnosis of the learning plateau observed in the Clinical Transformer (Exp Round 5, Exp2) was conducted. The plateau manifests as Recall@10 stalling at ~0.83 and rare/tail code accuracy remaining at 0% across all architectural variants (dense, Flash+learned pooling, MoE) and data scales (1.7M → 3.4M samples). Through a sequence of expert-panel analyses, gradient tier instrumentation, pos_weight ablations, embedding/logit diagnostics, and a decisive LR polishing test, the root cause was confirmed as **emergent gradient starvation** — a self-reinforcing dynamic where common codes capture ~85% of gradient budget by step 3000, starving rare/tail codes of learning signal and causing their embeddings to **homogenize** (std=0.03 vs common std=0.27). This investigation eliminated several competing hypotheses (model capacity, data quantity, LR schedule, per-sample weighting) with empirical evidence, and converged on density-based/day-level sampling + focal loss as the highest-priority interventions.

---

## 2. Planned vs. Executed

**Original Plan**: Diagnose why the learning plateau exists across Exp Round 5 experiments, test competing hypotheses, and identify the minimal set of interventions.

**What Got Done**:
- [x] Initial expert panel diagnosis (5 experts) — identified capacity, objective, data, and optimization as candidate ceilings (Jan 19-23)
- [x] Gradient tier analyzer implementation plan created (Jan 23)
- [x] pos_weight ablation (50 → 200) with gradient tier instrumentation (Jan 24)
- [x] Baseline gradient tier run at pos_weight=35 for comparison (Jan 24)
- [x] Expert interpretation of gradient results — confirmed emergent gradient concentration (Jan 24-25)
- [x] Next-step proposals and expert debate on interventions (Jan 25)
- [x] Code frequency and member/day/occurrence analysis (Jan 30)
- [x] IP association analysis by code tier (Jan 30)
- [x] Per-code embedding and logit diagnostic (Feb 1)
- [x] LR polishing test (2000 steps at LR=4e-06 from plateau checkpoint) (Feb 1)
- [x] Comprehensive evidence synthesis with LR polishing results integrated (Feb 1-2)
- [ ] Density-based/day-level tier-aware sampling implementation (deferred — next phase)
- [ ] Focal loss / ASL implementation (deferred — next phase)
- [ ] Per-tier loss balancing implementation (deferred — if above insufficient)

**Alignment Notes**: The investigation proceeded deeper and more methodically than initially scoped. What began as "why is there a plateau?" evolved into a rigorous hypothesis-testing framework with empirical falsification of multiple competing hypotheses. The extra depth was justified — it prevented premature commitment to an ineffective intervention (e.g., LR schedule changes, which the polishing test later rejected).

---

## 3. Key Decisions & Rationale

### Decision 1: Implement Gradient Tier Analyzer Before Intervening
**Date**: Jan 23  
**Context**: Expert panel identified gradient starvation as a plausible hypothesis but no empirical gradient data existed.  
**Options Considered**:
1. Proceed directly to interventions (tier-aware batching, sampled softmax)
2. Build instrumentation first to measure gradient contribution by code frequency tier

**Chosen**: Option 2 — Build `GradientTierAnalyzer` class integrated into `train_epoch` in `moe_flashattn_3.py`.  
**Rationale**: Without measurement, any intervention is a guess. The analyzer (4 code locations, ~300 lines) adds minimal training overhead (track every 100 steps) while providing decisive diagnostic data.  
**Trade-offs**: Delayed intervention by ~1 day of implementation, but prevented uninformed experimentation.

### Decision 2: Run pos_weight Ablation (50 → 200) as First Experiment
**Date**: Jan 24  
**Context**: All experts recommended testing whether increasing pos_weight could overcome gradient starvation.  
**Options Considered**:
1. Increase pos_weight to 200 (test weighting hypothesis)
2. Implement tier-aware batching first (test exposure hypothesis)
3. Switch to sampled softmax (test objective hypothesis)

**Chosen**: Option 1 — pos_weight ablation with gradient tier logging.  
**Rationale**: Cheapest experiment that tests a clearly-defined hypothesis ("is per-sample weighting insufficient?") while simultaneously collecting gradient tier data for the first time.  
**Outcome**: **Hypothesis rejected** — 4× increase in pos_weight had <0.5% effect on gradient tier fractions. Additionally revealed catastrophic medium code collapse (4.1% → 0.16% accuracy).

### Decision 3: Run Baseline Gradient Tier at pos_weight=35 for Comparison
**Date**: Jan 24  
**Context**: Expert 3 (Adjudicator) correctly identified a critical gap: no baseline gradient data existed at the original pos_weight setting. Without it, the gradient concentration pattern could be an artifact of pos_weight=200.  
**Options Considered**:
1. Accept the pos_weight=200 data alone
2. Run a control experiment at pos_weight=35

**Chosen**: Option 2 — Run baseline to confirm intrinsic nature of gradient concentration.  
**Rationale**: Scientific rigor. Without the baseline, the emergent gradient concentration hypothesis could not be confirmed.  
**Outcome**: **Confirmed** — gradient concentration timeline (17% → 85% common over steps 500-3000) is nearly identical at both pos_weight=35 and pos_weight=200.

### Decision 4: Prioritize LR Polishing Test Before Structural Interventions
**Date**: Jan 25 (proposed), Feb 1 (executed)  
**Context**: Expert 4 identified that the LR schedule (`linear_plateau_cosine`, `warmup_pct=0.15`, `plateau_pct=0.45`, `min_lr_ratio=0.2`) spends 60% of training at/near peak LR with a high floor. This is a known "plateau machine" configuration.  
**Options Considered**:
1. Run LR polishing test (cheap, fast, decisive)
2. Skip to density-based sampling directly
3. Implement focal loss first

**Chosen**: Option 1 — Resume from plateau checkpoint with LR = 4e-06 (10× lower), 2000 steps.  
**Rationale**: Cheapest experiment to rule in/out a major competing hypothesis. Expert 4 was the only expert to trace through the actual scheduler code.  
**Outcome**: **LR schedule hypothesis definitively rejected.** val_loss +0.45% (worse), NDCG -0.36%, MRR -0.98%, rare/tail unchanged at 0%. Model is at a sharp local minimum — recall@10 dropped 4.4% at step 200 then slowly recovered to below baseline.

### Decision 5: Reframe Goal as "Encouragement, Not Punishment"
**Date**: Jan 25  
**Context**: The natural response to gradient concentration is "normalize gradients" or "punish common codes." The medium code collapse (-96%) from pos_weight=200 demonstrated the danger of punitive interventions.  
**Options Considered**:
1. Per-tier gradient normalization (force equal gradient contribution — "punitive")
2. Additive interventions: tier-aware batching + hierarchical supervision + focal loss ("encouragement")

**Chosen**: Shift philosophy toward additive interventions that add signal for rare codes rather than remove signal from common codes.  
**Rationale**: Evidence from medium code collapse shows multiplicative/punitive interventions carry higher risk of degrading what already works. Additive approaches fail safely.  
**Trade-offs**: May take longer to converge to a solution, but lower risk of catastrophic regression.

---

## 4. Key Conclusions (Evidence-Based)

### Conclusion 1: The Plateau is Architecture-Agnostic and Data-Quantity-Independent

**Evidence**:
| Experiment | Params | Architecture | Data | Recall@10 | μRecall@10 |
|-----------|--------|-------------|------|-----------|------------|
| exp1_opt (dense) | 26.5M | Dense FP32 | 1.7M | 0.825 | 0.478 |
| exp2 (flash) | 25.3M | Flash+Learned Pool FP16 | 1.7M | 0.828 | 0.462 |
| exp6 (MoE) | 35.4M | MoE 8-expert top-2 FP16 | 1.7M | 0.827 | 0.461 |
| exp2_doubled | 25.3M | Flash+Learned Pool FP16 | 3.4M | 0.834 | 0.477 |

All optimized experiments converge to the same performance band (R@10 ≈ 0.82-0.83, μR@10 ≈ 0.46-0.48). Adding 40% more parameters (MoE) or 100% more data yielded negligible improvement.

**Implication**: The bottleneck is neither model capacity nor data quantity.

---

### Conclusion 2: Gradient Concentration is Emergent, Progressive, and pos_weight-Independent

**Evidence** (from gradient tier analyzer):

| Training Phase | Common Frac | Tail Frac | Total Norm |
|---------------|-------------|-----------|------------|
| Step 1 | 17.8% | 17.8% | 530,569 |
| Step 500 | 16.9% | 18.4% | 24,989 |
| Step 1500 | 42.7% | 10.4% | 3,398 |
| Step 3000 | 66.7% | 3.0% | 1,632 |
| Step 12001 | 85.3% | 0.1% | 22,129 |

This evolution is **near-identical** at pos_weight=35 and pos_weight=200 (final common_frac: 84.88% vs 84.68%). A 5.7× increase in pos_weight produced <0.5% change in gradient tier distribution.

**Implication**: Per-sample weighting (pos_weight) operates at the wrong level of aggregation. The problem is batch-level gradient accumulation, not per-sample gradient magnitude.

---

### Conclusion 3: The Problem is Occurrence-Level, Not Member-Level

**Evidence** (from code frequency analysis, Jan 30):

| Level | Tail Coverage | Common Coverage | Ratio |
|-------|---------------|-----------------|-------|
| Member | 83.4% | 100.0% | 1.2× |
| Day | 22.3% | 92.2% | 4.1× |
| Occurrence | 5.2% | 69.7% | **13.4×** |

83.4% of members have at least one tail code — but tail codes appear on only 22.3% of days and represent only 5.2% of total code occurrences.

**Implication**: Member-level tier-aware batching (current implementation) is insufficient because it guarantees member presence, not occurrence density. The gradient starvation operates at the occurrence level.

---

### Conclusion 4: Embeddings are Homogenized, Not Collapsed

**Evidence** (from Feb 1 embedding/logit diagnostic):

| Tier | Mean Norm | Std | Logit (y=1) | P(y=1) | Margin |
|------|-----------|-----|-------------|---------|--------|
| common | 1.42 | **0.27** | -2.26 | ~9.4% | 6.44 |
| medium | 1.49 | **0.15** | -6.39 | ~0.17% | 6.23 |
| rare | 1.41 | **0.05** | -9.68 | ~0.006% | 5.34 |
| tail | 1.46 | **0.03** | -14.69 | ~0.00004% | 1.76 |

- All embedding norms are healthy (~1.4-1.5) — no collapse
- But tail embedding std = 0.03 (vs common = 0.27) — all 1,175 tail codes converged to near-identical "default" embedding
- The model CAN distinguish tail positive from negative (margin = 1.76), but both logits are so deeply negative that tail codes never reach top-K
- More data made tail codes **worse** (logit -12.9 → -14.69 for 3.4M model) — the "Matthew Effect"

**Implication**: The problem is not capacity but **gradient diversity**. Tail codes receive insufficient variety of gradient directions to develop distinctive representations.

---

### Conclusion 5: LR Schedule is NOT the Bottleneck

**Evidence** (from Feb 1 polishing test):

| Metric | Before | After (2000 steps at LR=4e-06) | Delta |
|--------|--------|------|-------|
| val_loss | 0.00336 | 0.00338 | +0.45% (worse) |
| recall@10 | 0.8246 | 0.8258 | +0.14% (negligible) |
| ndcg@5 | 0.3571 | 0.3558 | -0.36% |
| mrr | 0.3364 | 0.3331 | -0.98% |
| rare/tail_top10_acc | 0% | 0% | No change |

Model at sharp local minimum: recall@10 dropped 4.4% at step 200 before slowly recovering to below baseline.

**Implication**: The plateau is a **structural/data ceiling**, not an optimization ceiling. Schedule adjustments alone cannot escape this basin.

---

### Conclusion 6: Per-Sample Weighting Has a Structural Ceiling

**Evidence** (from pos_weight 50 → 200 ablation):

| Metric | pw=50 | pw=200 | Delta |
|--------|-------|--------|-------|
| medium_top10_acc | 4.1% | 0.16% | **-96.2%** |
| rare_top10_acc | 0% | 0% | No change |
| tail_top10_acc | 0% | 0% | No change |
| macro_auroc | 0.846 | 0.878 | **+3.8%** |
| recall@5 | 0.722 | 0.686 | **-4.9%** |

Higher pos_weight improved discrimination (AUROC) but **degraded ranking** (recall@5, NDCG, MRR) and catastrophically harmed medium codes. This is the signature of **BCE-metric misalignment**: the loss improves without the business metrics improving.

**Mechanistic explanation**: `Total gradient per code ∝ (samples × pos_weight × per-sample error)`. Even with 200× weight, common codes (100K samples × 1 weight × 0.1 error = 10K) still contribute more total gradient than tail codes (100 samples × 200 weight × 0.9 error = 18K) due to batch-level averaging. Common code signal is consistent every batch; tail code signal is sporadic and averaged to noise.

---

## 5. Key Insights

### Insight 1: Three-Phase Gradient Dynamics

The gradient tier evolution follows a predictable three-phase pattern:
1. **Phase 1 (Steps 0-500)**: "Golden Era" — gradients are balanced (~18% per tier) because random initialization creates uniform errors
2. **Phase 2 (Steps 500-3000)**: "Gradient Takeover" — common codes capture majority of gradient budget (17% → 67%) as the model learns easy patterns
3. **Phase 3 (Steps 3000+)**: "Terminal Concentration" — tail signal becomes noise (~0.1%); self-reinforcing because further common-code refinement requires only small-magnitude-but-consistent updates that still dominate

**Critical window**: Any intervention to prevent gradient starvation must act **before step 1500-3000**, or the model commits to a common-code basin it cannot escape.

### Insight 2: The "Volume Dominance" Contradiction

Standard optimization intuition says well-learned examples (common codes) should produce small gradients and poorly-learned examples (tail codes) should produce large gradients. The gradient data reveals the **contradiction**: despite smaller per-sample gradients, common codes still dominate total gradient because their **sheer volume** (appearing in every batch) overwhelms the weighted contributions of sporadically-appearing tail codes. This is "volume dominance."

### Insight 3: Coverage ≠ Density — The Member-Level Illusion

The code frequency analysis revealed that 83.4% of members have tail codes — seemingly high coverage. But this is misleading because:
- Those tail codes appear on only 22.3% of member-days
- They represent only 5.2% of total occurrences
- Selecting a "tail member" for a batch doesn't guarantee tail codes appear in that batch's training targets

This insight reframed the sampling strategy from "ensure tail members appear" to "ensure tail code OCCURRENCES/TARGETS appear."

### Insight 4: Homogenization ≠ Collapse — A Subtler Failure Mode

The embedding diagnostic (Feb 1) revealed a nuanced failure mode:
- Tail code embedding **norms** are healthy (~1.46) — no collapse toward zero
- But tail code embedding **variance** is tiny (std=0.03) — all 1,175 tail codes learned approximately the same vector

This is **homogenization**, not collapse. It means insufficient gradient **diversity** (not gradient magnitude) is the bottleneck. The few gradient signals tail codes receive push them all in similar directions, converging to a "default tail embedding" that is uninformative for downstream tasks.

### Insight 5: The Matthew Effect — More Data Hurts Tail Codes

Comparing 1.7M and 3.4M models:
- Common/medium/rare code logits improved (moved toward 0)
- **Tail code logits got WORSE** (-12.9 → -14.69)
- Tail margin decreased (2.22 → 1.76)

More data amplifies the gradient starvation because common code occurrences scale proportionally more than tail, increasing the relative disadvantage. **More data without structural intervention makes the problem worse for the lowest-frequency codes.**

### Insight 6: Rare Codes Associate More Strongly with IP Risk

The OR analysis (Jan 30) revealed a statistically significant gradient:
- Common median OR: 1.46, 26.7% with OR > 2
- Medium median OR: 1.76, 44.7% with OR > 2
- Rare median OR: 2.42, 58.8% with OR > 2 (Mann-Whitney p < 0.0001)

**Caveat**: This association may be confounded by healthcare utilization (more encounters → more codes observed → more rare codes → more IP events). The finding is suggestive but not causal.

---

## 6. Technical Changes

### 6.1 Files Created

| File | Date | Purpose |
|------|------|---------|
| `exp_round5_overall_learning_plateau_general.md` | Jan 19 | Foundational guide: three-budget model (optimization/capacity/data ceilings), knob interactions, 5 decisive tests, checklist |
| `exp_round5_overall_learning_plateau_experts_views1.md` | Jan 23 | 5-expert panel diagnosis of plateau root cause across exp1/exp2/exp6 |
| `exp_round5_overall_learning_plateau_graident_code_starvation_analysis.md` | Jan 23 | `GradientTierAnalyzer` class design and implementation plan for `moe_flashattn_3.py` |
| `exp_round5_exp2_lr_plateau_gradient_observation_jan24.md` | Jan 24 | Raw observations from pos_weight=200 ablation + gradient tier data + 3.4M model embedding/logit comparison |
| `exp_round5_exp2_lr_plateau_gradient_result_expert_interpret_jan24.md` | Jan 24 | 5-expert interpretation of gradient results: confirmed emergent concentration, volume dominance, pos_weight ceiling |
| `exp_round5_exp2_lr_plateau_gradient_result_nextstep_discussion_jan25.md` | Jan 25 | Reframed goal ("encouragement not punishment"), proposed 5 approaches, 3-expert review + author response, revised priority stack |
| `exp_round5_exp2_lr_plateau_percode_diag_tier_aware_batching_jan26.md` | Jan 26 | Per-code diagnostics and tier-aware batching implementation analysis (in `solution_ablations/`) |
| `exp_round5_exp2_lr_plateau_code_frequency_observation_jan30.md` | Jan 30 | Code frequency distribution (member/day/occurrence), IP-code association OR analysis |
| `exp_round5_exp2_lr_plateau_embedding_logits_LR_polishing_observation_feb1.md` | Feb 1 | LR polishing test results + per-code embedding norms + logit distributions |
| `exp_round5_exp2_lr_plateau_evidence_sythesis_expert_interpret_nextstep_feb1.md` | Feb 1 | Comprehensive synthesis integrating all evidence through Feb 1: multiple expert panels, method catalogue for member profiling |
| `exp_round5_exp2_lr_plateau_evidence_sythesis_expert_interpret_nextstep_feb_2.md` | Feb 2 | Updated synthesis incorporating polishing test rejection, embedding homogenization finding, master evidence table |

### 6.2 Code Changes Planned

| Location | Change | Status |
|----------|--------|--------|
| `moe_flashattn_3.py` ~L4340 | Add `TierGradientMetrics` dataclass + `GradientTierAnalyzer` class | Designed |
| `moe_flashattn_3.py` ~L4847 | Add `gradient_analyzer` parameter to `train_epoch` | Designed |
| `moe_flashattn_3.py` ~L4990 | Add gradient tier computation after `backward()` | Designed |
| `moe_flashattn_3.py` ~L10950 | Initialize `GradientTierAnalyzer` in `run_single_experiment` | Designed |
| `TierAwareBatchSampler` in `moe_flashattn_4.py` | Refactor from binary-presence to density-based sampling | Pending |

---

## 7. Discussions & Reasoning

### Topic 1: Capacity vs. Gradient Starvation — The Central Debate

**Question**: Is the plateau caused by model capacity limits or gradient starvation?

**Analysis**: Experts initially split — Experts 1,2,4 favored capacity (d_model=256 too small); Expert 3 favored data/objective alignment. The MoE experiment (35.4M params, exp6) scoring identically to the 25.3M dense model was strong evidence against capacity. The Adjudicator (Expert 5) noted this reasoning is partially circular: MoE capacity goes unused BECAUSE gradients are concentrated, not necessarily because capacity is sufficient.

**Conclusion**: Gradient starvation is the primary cause. Capacity is a potential secondary bottleneck that cannot be evaluated until starvation is addressed.

**Key Evidence**: All architectures converge to same performance band; gradient tier analysis shows 85% concentration on common codes.

### Topic 2: Should the Objective Be Changed (BCE → Ranking Loss)?

**Question**: Is BCE fundamentally misaligned with the task?

**Analysis**: Expert 2 provided strong evidence of objective-metric misalignment (AUROC +3.8% while recall@5 -4.9% with higher pos_weight). However, the primary goal was clarified as **member profiling** (learning embeddings for downstream tasks), not ranking per se.

**Conclusion**: BCE is well-aligned with the profiling goal. Sampled softmax/ranking losses optimize for ordering rather than representation quality. Top-K metrics serve as diagnostics but are not the primary objective. Focal Loss (which modifies gradient dynamics within BCE) is appropriate; sampled softmax is not.

### Topic 3: Member-Level vs. Occurrence-Level Sampling

**Question**: Can tier-aware batching at the member level solve gradient starvation?

**Analysis**: The Jan 30 code frequency analysis was the turning point. With 83.4% member-level tail coverage but only 5.2% occurrence-level coverage, member-level sampling is provably insufficient. Even guaranteeing 10 tail members per batch of 128, each member has ~1800 common occurrences vs ~45 tail occurrences → gradient ratio unchanged.

**Conclusion**: Sampling must operate at the day-level or density-level. Three implementation paths:
1. **Day-level sampling**: Select (member, day) pairs where tail codes appear as targets
2. **Density-based sampling**: Select members with highest tail code density (top 20th percentile)
3. **Hybrid**: Keep member as training unit but preferentially include days containing tail codes

---

## 8. Hypothesis Resolution Table

| Hypothesis | Status | Date | Key Evidence |
|-----------|--------|------|--------------|
| **H1: LR schedule is primary bottleneck** | **REJECTED** | Feb 1 | Polishing test: val_loss worse, NDCG degraded, rare/tail 0% |
| **H2: Embedding collapse (norms → 0)** | **REJECTED** | Feb 1 | All tiers: min norm > 0.8, num_near_zero = 0 |
| **H3: Model capacity is limiting** | **REJECTED** | Jan 23 | MoE (35M) ≈ Dense (25M) in all metrics |
| **H4: Data quantity is limiting** | **REJECTED** | Jan 23 | 2× data → +0.6% R@10, +3.2% μR@10 only |
| **H5: pos_weight is insufficient (needs higher)** | **REJECTED** | Jan 24 | 5.7× increase → <0.5% gradient distribution change, medium collapsed |
| **H6: Gradient starvation is static** | **REJECTED** | Jan 24 | Starts balanced (18%), concentrates dynamically to 85% |
| **H7: Gradient concentration is emergent** | **CONFIRMED** | Jan 24 | Three-phase evolution, pos_weight-independent |
| **H8: pos_weight cannot overcome sample count differential** | **CONFIRMED** | Jan 24 | 5.7× weight change, <0.5% concentration change |
| **H9: Problem is occurrence-level, not member-level** | **CONFIRMED** | Jan 30 | 83.4% member coverage but 5.2% occurrence coverage |
| **H10: Embedding homogenization (low diversity)** | **CONFIRMED** | Feb 1 | Tail std = 0.03 vs common std = 0.27 |
| **H11: Logit suppression / learned negative prior** | **CONFIRMED** | Feb 1 | Tail logit = -14.69 (P ≈ 0.00004%) |
| **H12: Model can distinguish tail pos/neg** | **CONFIRMED** | Feb 1 | Margin = 1.76 > 0 (correct direction) |
| **H13: Model is at sharp local minimum** | **CONFIRMED** | Feb 1 | R@10 dropped 4.4% at step 200, recovered to same basin |
| **H14: More data hurts tail codes** | **CONFIRMED** | Feb 1 | 3.4M: tail logit -14.69 vs 1.7M: -12.93 |
| **H15: Rare codes have stronger IP association** | **TENTATIVE** | Jan 30 | OR gradient 1.46→2.42 (p<0.0001), but confounders not ruled out |

---

## 9. Plan Alignment Review

**PRD/Original Goals**: Diagnose and resolve the learning plateau in Clinical Transformer Exp Round 5 to improve member profiling quality.

**Completion Status**:
- Diagnosis phase: **100% complete** — root cause confirmed with empirical evidence
- Intervention phase: **0% implemented** — structural interventions designed but not yet coded

**Scope Changes**: Investigation expanded from "quick ablation" to "rigorous hypothesis-testing framework" due to the complexity of interacting factors (gradient dynamics, occurrence imbalance, objective-metric misalignment, schedule). This depth was necessary — it prevented committing to interventions that later evidence showed would be ineffective (e.g., pos_weight increases, LR schedule changes).

---

## 10. Blockers & Issues

**Resolved**:
- Missing gradient data → Resolved by designing and running `GradientTierAnalyzer` (Jan 23-24)
- No baseline gradient comparison → Resolved by running pos_weight=35 control experiment (Jan 24)
- LR schedule as competing hypothesis → Resolved by polishing test (Feb 1)
- Unknown embedding health → Resolved by per-code embedding/logit diagnostic (Feb 1)

**Outstanding**:
- **Zero-code anomaly**: 451 codes with training frequency=0 have 54,464 positive validation samples and very high model confidence (logit +4.76, 99.8% above zero). Possible causes: code vocabulary mismatch between train/val, temporal distribution shift, incorrect frequency computation. Needs investigation.
- **Per-batch tail occurrence verification**: Have not yet measured what percentage of each batch's actual gradient comes from tail codes. This 1-hour measurement should precede implementation.
- **Medium code sensitivity to pos_weight**: pos_weight=50 achieved 4.1% medium accuracy vs 0.47% at pos_weight=35. This non-monotonic relationship is unexplained.

---

## 11. Next Session Plan

**Immediate Priorities** (ranked):

1. **Measure per-batch tail occurrence** [1 hour] — Quantify actual gradient imbalance per batch to confirm occurrence-level starvation before implementing fixes.

2. **Implement density-based / day-level tier-aware sampling** [1 training run] — Replace binary-presence member selection with tail-density scoring (top-20% percentile) or day-level (member, day) pair sampling where tail codes appear as targets.
   - Success criteria: `train_grad_tier_tail_frac > 5%`, tail embedding std > 0.10, `tail_top10_acc > 0%`

3. **Replace BCE + pos_weight with Focal Loss (γ=2) or ASL (γ+=0, γ-=4)** [1 training run] — Dynamic difficulty-based reweighting that adapts to model confidence, complementary to density-based sampling.

4. **If above insufficient: Per-tier loss balancing** [1 training run] — `total_loss = 0.25 * loss_common + 0.25 * loss_medium + 0.25 * loss_rare + 0.25 * loss_tail`. Forces equal tier contribution to gradient.

5. **If above insufficient: Hierarchical supervision** — Add CCS/CCSR category-level auxiliary loss (λ=0.1) to provide indirect gradient signal for rare codes via parent category membership.

**What Will NOT Work** (evidence-based):
- LR schedule changes alone (polishing test: rejected)
- Increasing pos_weight further (5.7× increase ineffective, causes medium collapse)
- Member-level tier-aware batching alone (83.4% coverage already)
- Adding model capacity (MoE ≈ Dense)
- Adding more data without structural changes (Matthew Effect)
- Embedding norm regularization (embeddings not collapsed)
- More training steps at low LR (model at sharp minimum)

**Open Questions**:
- What is the optimal tail-density threshold for sampling? (Need to compute tail-density distribution)
- Will focal loss interact well with density-based sampling, or will one dominate?
- Is the "zero-code anomaly" (451 codes, 54K positive validation samples) a data pipeline bug?
- Does the OR gradient (rare codes → higher IP association) survive adjustment for healthcare utilization?

---

**Investigation Duration**: ~2 weeks (Jan 19 – Feb 2, 2026)  
**Files Created**: 11 analysis documents  
**Key Experiments Run**: 5 (original exp round 5 suite, pos_weight=200 ablation, pos_weight=35 baseline, LR polishing test, embedding/logit diagnostic)  
**Hypotheses Tested**: 15 (6 rejected, 8 confirmed, 1 tentative)  
**Root Cause**: Emergent gradient starvation → embedding homogenization, driven by occurrence-level imbalance (13.4× ratio)  
**Environment**: macOS, PyTorch, AdamW optimizer, ~27M parameter clinical transformer, ~1.7M-3.4M training members, ~6,297 target diagnosis codes
