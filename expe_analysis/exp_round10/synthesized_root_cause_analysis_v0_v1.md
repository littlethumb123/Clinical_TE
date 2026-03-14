# Synthesized Root Cause Analysis: Systematic Comparison and Integration of V0 and V1

**Date**: 2026-03-13
**Subject**: Exp Round 10 — Why 11M formal training shows no downstream lift
**Sources**: `hypothesis_driven_root_cause_analysis_v0.md` (V0) and `hypotheiss_driven_root_cause_analysis_v1.md` (V1)
**Method**: Section-by-section analytical comparison — agreement, divergence, and resolution

---

## 0. Document Overview

| Dimension | V0 | V1 |
|-----------|----|----|
| Analyst perspective | Claims independent analysis | Claims independent analysis |
| Structural framework | 4-level priority hierarchy (Data → Loss → Training Dynamics → Architecture) | Same 4-level hierarchy |
| Emphasis | Pretraining-downstream misalignment as primary root cause | Representation monopolization → tabular redundancy as primary root cause |
| Number of hypotheses | 7 (H1.1, 1.1a, 1.1b, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2) | 8 (H1.1, H1.2, H1.3, H2.1, H2.2, H3.1, H3.2, H4.1) |
| Intervention style | Tiered A-D by cost | Tiered 1-3 by impact |
| Unique contribution | Goodhart's Law framing; config audit; literature references | Tabular redundancy hypothesis (H1.3); 4 cheap diagnostics; recommendation system analogies |

---

## 1. Phase 1: Observation & Documentation

### 1.1 Problem Statement

**AGREED**: Both documents frame the same core discrepancy:
- Expected: 7.3× data scaling (1.5M→11M) should improve downstream IP prediction
- Observed: Negligible embedding-only gain (+0.2pp), hybrid **regression** (-0.4pp) vs R6 6-8M

Both identify R6 6-8M as the peak downstream performer for hybrid models. No disagreement on the problem statement.

### 1.2 Pretraining Metrics

**AGREED on all numbers** (both cite the same source data). Key shared observations:
- R10 val_R@10 (0.853) ≈ R6 6-8M (0.855) — plateau
- R10 val_μR@10 (0.563) < R6 6-8M (0.576) — regression
- medium_top10_acc jumped 4.3%→20% at R10
- tail_top10_acc = 0% universally
- macro_AUROC improved 0.913→0.920

**MINOR DIVERGENCE**: V1 includes a more complete table with Val Loss, balanced_top10_acc, and the R9 v0 decoder experiment. V1 also notes that ASL loss values are on a different scale (†footnote), which V0 does not flag. V1's table is more precise; V0's is more readable.

**Resolution**: V1's note about ASL loss scale incompatibility is a valid methodological caveat that V0 omits. The balanced_top10_acc (V1: 26.3% for R10) provides additional context showing R10 DID improve on balanced metrics internally.

### 1.3 Downstream Metrics

**AGREED on all key findings**:
1. Embedding-only R10 oot_strict = 0.809 vs R6 = 0.807 (+0.002)
2. Hybrid R10 oot_strict = 0.831 vs R6 = 0.835 (-0.004, regression)
3. R9 v3 co-occ has best lift@1pct (20.50) in hybrid
4. Tabular baseline gap remains

**DIVERGENCE on framing**:
- V0 frames the hybrid regression as -0.4pp and highlights the non-monotonic scaling curve
- V1 adds a critical observation that V0 **does not make**: R10 hybrid oot_strict (0.831) **exactly matches** the embedding-matched tabular-only baseline (0.831), implying "zero incremental value from embeddings"
- V1 computes marginal AUC per million members (0.003/M → 0.002/M → 0.0005/M), calling it a "6x collapse in data efficiency"
- V1 includes the full-population baseline (0.838) and matched-population baseline (0.831), while V0 only references the full-population baseline
- V1 also flags that R10 val_AUC (0.784) is lower than R6 val_AUC (0.793) — a validation-level anomaly that V0 doesn't examine

**Resolution**: V1's observation that R10 hybrid = matched tabular baseline (0.831) is a **critically important finding** that V0 missed. This elevates the analysis from "scaling doesn't help much" to "embeddings provide literally zero incremental information." V1's per-million-member efficiency calculation adds quantitative rigor. V1's val_AUC anomaly (R10 < R6 on val) is also an unaddressed signal in V0.

### 1.4 Training Dynamics

**AGREED**: Both document the same R10 training characteristics:
- 84,855 batches, 1 epoch, 4× T4
- Loss 0.800→0.002, rapid convergence
- Metrics oscillation in second half

**DIVERGENCE**:
- V0 states R@10 oscillated in [0.820, 0.887] and reached ~0.84 by hour 5, concluding the last 27 hours were <1.5pp improvement
- V1 calculates that loss reached floor by ~step 15,000 and ~82% of training (70,000 steps) was at or near the floor
- V1 explicitly calculates the wasted compute: $44.53 with 82% wasted

**Resolution**: Both reach the same conclusion through slightly different calculations. V1's "82% at the loss floor" is a cleaner metric than V0's "last 27 hours produced <1.5pp." Both are valid; V1's is more precise.

### 1.5 Config Comparison

**AGREED**: Both confirm identical configs between R6 6-8M and R10 11M — data size is the only variable. Both note this makes it a clean comparison.

**DIVERGENCE**: V0 includes a detailed config comparison table. V1 states the same fact but relies on inline text rather than a table. V0 additionally notes that gradient tier analysis was disabled in R10, calling it a "missed signal." V1 also notes this (`enable_gradient_tier_analysis: false`) but frames it as limitation of evidence rather than a missed opportunity.

---

## 2. Phase 2: Hypothesis Generation

### Level 1: DATA

#### Area of Agreement

Both analyses agree that:
- Data volume is NOT the primary bottleneck
- The additional 4.2M members primarily reinforce common-code patterns
- tail_top10_acc = 0% at all data scales proves data scaling cannot fix tail codes
- The model IS learning from additional data (medium_top10_acc improves) but this doesn't translate downstream

#### Area of Divergence

**V0 has 2 sub-hypotheses under Level 1**:
- H1.1: Data information content is insufficient relative to pretraining objective
- H1.1a: Additional members are common-code-dense
- H1.1b: Pretraining-downstream misalignment is the binding constraint (V0 places this under DATA)

**V1 has 3 distinct hypotheses under Level 1**:
- H1.1: Additional data is redundant for downstream-relevant tiers
- H1.2: Distribution mismatch between training and downstream populations (V0 does not consider this)
- **H1.3: Embeddings are redundant with tabular features** (V0 does not have this hypothesis at all)

**Critical divergence — H1.2 (V1 only)**: V1 posits a distribution mismatch hypothesis, noting that R10 val_AUC (0.784) is lower than R6 val_AUC (0.793) despite more training data. V1 marks this "PARTIALLY CONFIRMED." V0 does not examine this anomaly.

**Analysis of H1.2**: The val_AUC anomaly is real and unexplained by V0. However, "distribution mismatch" may be too strong — the downstream test_AUC consistently improves with data scaling (R6: 0.799, R10: 0.815), which V1 itself acknowledges as counter-evidence. A more parsimonious explanation: the validation set composition may have shifted between experiments due to different train/val splits at different data scales. V1 correctly marks this as "contributing factor, not root cause."

**Critical divergence — H1.3 (V1 only, strongest)**: V1 introduces the "tabular redundancy" hypothesis — that embeddings encode information **identical** to what tabular features already capture. V1 supports this with:
- R10 hybrid oot_strict (0.831) = tabular baseline (0.831) — zero incremental value
- Across all experiments, hybrid models barely exceed tabular (range 0.825–0.835 vs baseline 0.831–0.838)
- lift@1% for production baseline (19.38) exceeds R10 hybrid (18.69) — embeddings make it worse

V0 does not formulate H1.3 explicitly. V0 discusses "pretraining-downstream misalignment" (H1.1b) and "common-code feature extractor" (in Phase 3) which captures **part** of the same idea — that the representation is dominated by common codes. But V0 does not take the additional step of asking: "Are those common-code features the **same** as tabular features?"

**Resolution**: V1's H1.3 is a genuinely novel and important contribution. It moves beyond "the pretraining objective is wrong" to "the pretraining objective produces features that tabular already has." This distinction matters for intervention design: if embeddings are just noisier versions of tabular features, then ANY improvement to common-code prediction (more data, better loss, etc.) will only increase redundancy. The fix must force the encoder to learn **orthogonal** information. V0's framework correctly identifies the mechanism (representation monopolization → common-code features) but doesn't connect the final dot to tabular redundancy.

**Verdict**: V0's data-level analysis is correct but incomplete. V1 adds two hypotheses (H1.2, H1.3) that V0 misses. H1.3 (tabular redundancy) is the most consequential addition.

---

### Level 2: LOSS / OBJECTIVE ALIGNMENT

#### Area of Agreement

Both analyses agree on:
- BCE loss + pos_weight creates occurrence-frequency-driven gradients (~85% common, ~0.1% tail)
- This distribution is invariant to loss function, pos_weight, data size, model capacity
- ASL fixed ranking quality but didn't change gradient distribution
- Pretraining metric improvements (macro_AUROC, medium_top10) don't map to downstream gains
- This is a confirmed primary bottleneck

Both cite the same evidence: pos_weight 35→200 changed gradient distribution <0.5%, ASL v4/v5 results, and the evidence synthesis from R5/R9 analyses.

#### Area of Divergence

**V0 presents two hypotheses**:
- H2.1: BCE creates representations optimized for code prediction, misaligned with patient risk
- H2.2: Representation monopolization is a loss-mediated structural failure

**V1 presents two hypotheses**:
- H2.1: Pretraining objective (6,297 codes) is misaligned with downstream (binary classification)
- H2.2: Loss-metric divergence is worsening with data scaling

**V1's H2.2 is distinct from V0's H2.2**: V0's H2.2 states that representation monopolization is "a loss-mediated structural failure, not a data problem." V1's H2.2 states that the gap between internal improvement and downstream stagnation **widens with scale**: at 1.5M both improve; at 6.8M internal improves more; at 11M internal continues while downstream stalls. V1's version is a stronger claim — not just that the divergence exists, but that it **intensifies**.

**V0 has a mechanistic explanation that V1 lacks**: V0 explicitly models WHY more data worsens the hybrid: "more data → more common-code gradient → more common-code-dominated representation → less useful for downstream patient risk prediction which benefits from nuanced medium/rare code patterns." V0 frames this as "the mechanism behind the hybrid regression." V1 reaches the same conclusion but through the "tabular redundancy" pathway (more data → more precise common-code estimation → MORE redundancy with tabular).

**Resolution**: The two mechanistic pathways converge to the same conclusion but via different reasoning:
- V0: more data → deeper gradient monopolization → over-specialization hurts downstream
- V1: more data → more precise common-code features → more overlap with tabular → less additive value

These are complementary, not contradictory. V0 explains the mechanism at the gradient level; V1 explains the consequence at the feature-information level. Both are valid. The synthesized view is: more data deepens gradient monopolization (V0's mechanism), which causes the representation to more precisely encode common-code statistics (V0+V1), which are the same information tabular features already provide (V1's unique insight), resulting in zero incremental hybrid value.

---

### Level 3: TRAINING DYNAMICS

#### Area of Agreement

Both analyses agree:
- Gradient starvation (85% common, <1% tail) persists at 11M (inferred from identical config)
- Gradient capture completes by step ~3,000 and is irreversible
- LR polishing test rejected schedule as root cause
- Single-epoch training is a contributing factor but not the primary bottleneck
- The remaining ~80-96% of training reinforces existing patterns

#### Area of Divergence

**V0 is more emphatic**: V0 marks training dynamics as "contributing factor but not root cause" and states "multi-epoch training alone would not fix the structural issue."

**V1 is more nuanced**: V1 notes that even at the loss floor, the model sees **fresh examples** (unique patients), and this does benefit medium codes (medium_top10_acc improved dramatically). V1 marks single-epoch as "PARTIALLY CONFIRMED — the single-epoch structure is suboptimal but not the primary bottleneck."

**V0 has H3.2 (LR schedule)**: V0 explicitly discusses the warmup→plateau→decay schedule causing irreversible gradient capture during warmup, citing the LR polishing test as rejection evidence.

**V1's H3.2 is different**: V1's H3.2 is about single-epoch diminishing returns (not LR schedule), noting that ~82% of steps are at the loss floor.

**Resolution**: V1's nuance about fresh examples benefiting medium codes is valid — it explains WHY medium_top10_acc jumped from 3.93% to 20% even though recall-based metrics plateaued. V0 misses this explanation. However, V0's explicit treatment of the LR schedule hypothesis (and its rejection) is more thorough. The synthesized view should include both: (1) V0's explicit rejection of LR schedule as root cause, and (2) V1's nuance that single-epoch training is not purely wasteful (medium codes benefit from fresh examples even at loss floor).

---

### Level 4: ARCHITECTURE / SCALING

#### Area of Agreement

Both analyses confirm:
- 256d capacity is NOT the bottleneck (R7 512d → +0.1pp, negligible)
- Shared encoder + single linear decoder is a structural constraint
- `h ∈ ℝ^256` shared by 6,297 codes is monopolized by common codes
- This is architecturally baked in

#### Area of Divergence

**V0 separates architecture from loss**: V0 treats architecture as "a necessary condition for monopolization, but the loss function is the sufficient condition." V0's Level 4 verdict is subordinate to Level 2.

**V1 elevates architecture to definitive status**: V1's H4.1 is marked "CONFIRMED — this is the structural root cause" and presents it as the **primary** root cause, above objective misalignment. V1 provides additional evidence points:
- F13: "256 dimensions to encode 6,297 binary predictions is an information-theoretic bottleneck"
- F15: "The downstream hybrid model comparison is the most telling: the embeddings capture information that is almost entirely redundant with tabular features"

**V1 adds information-theoretic framing**: V1 argues that 256 dimensions for 6,297 predictions is inherently an information bottleneck, and that common codes **consume** most of these dimensions. V0 does not make this information-theoretic argument.

**Resolution**: This is a genuine interpretive disagreement about causal primacy:
- V0: Loss/objective → is the primary cause; architecture → is the structural precondition
- V1: Architecture/gradient starvation → is the primary cause; objective misalignment → is the secondary cause

The disagreement is about the **ordering** of the causal chain, not about whether both factors contribute. In V0's view, you could keep the same architecture but change the loss (e.g., per-tier gradient normalization) and improve outcomes. In V1's view, the architecture fundamentally limits what the encoder can learn regardless of loss function changes — and V1 cites the evidence that loss function changes (ASL, pw200) have been tried and failed.

**Analytical resolution**: V1's ordering is more defensible here. The evidence shows that loss function changes (v4 ASL, v5 ASL+sampler) did NOT change gradient distribution or downstream outcomes. If loss were truly the "sufficient condition" (V0's claim), then loss changes should have had some effect. The fact that they didn't suggests the architecture mediates the relationship more strongly than V0 acknowledges. However, V0 is correct that per-tier gradient normalization (which V0/V1 both recommend) IS a loss-side intervention that could work — it just hasn't been tested yet. The truth is likely: architecture creates the precondition, loss function determines the gradient distribution WITHIN that architecture, and no tested loss variant has changed the distribution. Per-tier gradient normalization is the untested loss-side intervention that might work within the existing architecture.

---

## 3. Root Cause Synthesis

### V0's Causal Model

```
Layer 1: STRUCTURAL → Shared encoder + linear decoder → gradient aggregation
Layer 2: LOSS-MEDIATED → BCE + mean reduction → 85% common gradient monopoly
Layer 3: TASK MISALIGNMENT → Code prediction ≠ patient risk → improvements don't transfer
```

V0 names the primary root cause as "Pretraining-Downstream Objective Misalignment Compounded by Representation Monopolization."

### V1's Causal Model

```
PRIMARY: Representation monopolization → downstream redundancy
  → Encoder becomes common-code feature extractor
  → Common-code features = tabular features
  → Zero incremental value

SECONDARY: Pretraining-downstream objective misalignment
  → Code prediction optimizes for information tabular already has

CONTRIBUTING: Diminishing data returns
  → 0.0005 AUC/M at 6.8M→11M (6x collapse)
```

V1 names the primary root cause as "Representation Monopolization → Downstream Redundancy" with objective misalignment as secondary.

### Where They Agree

Both identify the same causal chain with the same components:
1. Shared encoder architecture creates gradient aggregation
2. Occurrence-frequency-driven gradients produce ~85% common-code domination
3. The encoder becomes a common-code feature extractor
4. Common-code features do not transfer to downstream tasks
5. More data amplifies the problem rather than solving it

### Where They Diverge

| Dimension | V0 | V1 |
|-----------|----|----|
| **Primary root cause label** | Pretraining-downstream misalignment | Representation monopolization → tabular redundancy |
| **Causal ordering** | Loss/objective → Architecture | Architecture/gradient → Objective |
| **Explains hybrid regression via** | Over-specialization reduces general-purpose utility | More precision on common codes → more redundancy with tabular |
| **Tabular redundancy** | Not explicitly formulated | Central thesis |
| **Goodhart's Law framing** | Present (Appendix A) | Not present |
| **Information-theoretic argument** | Not present | Present (256d for 6,297 predictions) |
| **"What would fix it" emphasis** | Fine-tuning downstream (B.1) | Force orthogonal information (auxiliary objectives, gradient rebalancing) |

### Analytical Resolution

The two analyses are describing the **same elephant from different angles**:

**V0 asks**: "Why don't pretraining improvements transfer to downstream?" Answer: because the pretraining objective is misaligned.

**V1 asks**: "Why do embeddings provide zero incremental value over tabular?" Answer: because representation monopolization makes embeddings encode the same information as tabular features.

These are the same phenomenon expressed at different levels of abstraction:
- V0 operates at the **training objective** level
- V1 operates at the **information content** level

V1's framing is more actionable because it identifies the specific information gap (tabular redundancy) and implies a specific class of solutions (force orthogonal information). V0's framing is more general and could apply to any pretraining-downstream mismatch.

**The synthesized root cause statement**:

> The shared encoder architecture, trained under BCE loss with occurrence-frequency-driven gradients (~85% common-code domination), converges to a representation `h` that encodes aggregate patient characteristics dominated by common conditions — **the same information** that tabular features (demographics, claims aggregates, historical code counts) already capture. Scaling data from 1.5M to 11M deepens this monopolization, making the representation **more precisely** encode common-code statistics without adding orthogonal information. The result is that at 11M, the embedding provides **zero incremental value** over tabular features for the downstream prediction task (R10 hybrid oot_strict = 0.831 = tabular baseline), and the hybrid model actually **regresses** vs the 6.8M model because the more specialized embedding provides less complementary signal.

---

## 4. Rejected Hypotheses

### Agreed Rejections (Both Analyses)

| Rejected Hypothesis | Evidence | V0 | V1 |
|---------------------|----------|----|----|
| "Need more data" | 7.3× more data → -0.4pp hybrid | ✓ | ✓ |
| "Need larger model (capacity)" | 512d → +0.1pp (R7) | ✓ | ✓ |
| "Need different loss function" | ASL → +0pp downstream | ✓ | ✓ |
| "Need higher pos_weight" | 200 vs 35 → <0.5% gradient change | ✓ | ✓ |
| "LR schedule is the issue" | Polishing test rejected | ✓ | ✓ |
| "Decoder is the bottleneck" | R9 v0/v1 proved h lacks signal | ✓ (implicit) | ✓ (explicit) |

### V0-Only Rejections

| Rejected Hypothesis | Evidence |
|---------------------|----------|
| "Need density-aware sampling" | v5 → best pretraining, worst downstream |
| "Need more epochs (alone)" | Won't fix gradient monopolization |

### V1-Only Rejections

V1 does not explicitly reject additional hypotheses beyond the shared list, but its H1.2 (distribution mismatch) is marked as "partially confirmed" — meaning V1 treats this as a contributing factor rather than rejecting it outright. V0 does not consider distribution mismatch at all.

---

## 5. Intervention Design

### Agreed Interventions

Both analyses recommend (in various forms):
1. **Downstream-aware fine-tuning or auxiliary objective** — directly optimize for downstream task
2. **GradNorm / per-tier gradient normalization** — rebalance gradient contributions
3. **Contrastive learning** — force representation diversity
4. **Per-tier decoder heads / MoE decoder** — break cross-code interference
5. **Co-occurrence embeddings** — proven beneficial in R9 v3

### Divergences in Intervention Design

#### Priority #1 Disagreement

**V0's #1 priority**: Downstream-aware fine-tuning of R6 6-8M model
- Cost: ~4 GPU-hours (~$1.40)
- Rationale: Directly attacks pretraining-downstream misalignment with minimal cost
- Success criterion: AUC > 0.845

**V1's #1 priority**: Downstream-aware auxiliary objective during pretraining (1A) + GradNorm (1B) + Per-tier MLP decoder (1C)
- Cost: ~1 full retraining run ($17-45)
- Rationale: Forces encoder to allocate capacity to downstream-relevant patterns AND break gradient monopolization
- No pre-registered success criterion

**Analysis**: V0 proposes a **cheap diagnostic** (fine-tune existing checkpoint) while V1 proposes a **structural fix** (retrain with new objectives). V0 explicitly pre-registers success/failure criteria; V1 does not.

V0's approach is the correct **first step** because:
1. It costs 10-30× less than V1's approach
2. It disambiguates whether the problem is "representation is damaged beyond repair" vs "representation has useful features that the current pipeline doesn't extract"
3. If B.1 succeeds (AUC > 0.845), the entire V1 retrain-from-scratch program is unnecessary
4. If B.1 fails, V1's more expensive interventions are justified

V1's approach is the correct **long-term fix** because:
1. Fine-tuning addresses the symptom (poor downstream transfer) not the cause (monopolized representation)
2. If the representation truly lacks orthogonal information (V1's H1.3), fine-tuning cannot create information that isn't there
3. V1's interventions (GradNorm, auxiliary objectives, MoE decoder) address the root cause

**Resolution**: V0's cheap diagnostic should come FIRST, followed by V1's structural fixes if the diagnostic indicates the representation itself needs changing.

#### Unique V1 Contributions to Intervention Design

**Diagnostic 1**: Embedding Feature Importance in Downstream CatBoost (~10 min CPU)
- V0 has no equivalent
- This would directly test V1's H1.3 (tabular redundancy)
- Pre-registered: if embedding features have <0.1% importance → redundancy confirmed

**Diagnostic 2**: Linear Probe on frozen h (~30 min on checkpoint)
- V0 has no equivalent
- Tests how much downstream-relevant information exists in h
- Pre-registered: if probe AUC > tabular AUC → h has unique signal; if ≤ → doesn't

**Diagnostic 3**: CKA/CCA between h and tabular features (~20 min)
- V0 has no equivalent
- Directly quantifies information overlap
- Pre-registered: if CKA > 0.8 → redundancy; if < 0.5 → orthogonal

**Diagnostic 4**: Per-code contribution to downstream AUC (~1 hour)
- V0 has no equivalent
- Maps which pretraining codes contribute to downstream utility

**Analysis**: V1's four diagnostics are excellent additions that V0 entirely lacks. They cost essentially nothing (total ~2 hours on CPU/checkpoint) and would provide definitive evidence for or against the tabular redundancy hypothesis. These should be performed BEFORE any expensive intervention.

#### Unique V0 Contributions to Intervention Design

**B.2 — Contrastive pre-training**: V0 proposes patient-level contrastive loss (similar clinical trajectories → similar h). V1 proposes "TierAwareContrastiveLoss" pushing apart patients with different code profiles. These are complementary framings of the same idea.

**B.3 — Co-occurrence embeddings + gradient rebalancing**: V0 proposes combining R9 v3's PPMI+SVD with per-tier gradient normalization as a single experiment. V1 recommends GradNorm separately.

**C.2 — Hierarchical code supervision (CCS/CCSR)**: V0 proposes adding a secondary loss predicting ~280 clinical categories. V1 does not mention this specific approach.

**D.2 — Dual-encoder architecture**: V0 proposes shared + tier-specific encoder branches. V1 proposes "Two-Tower Architecture with Downstream-Specific Head" (3A), which is architecturally different — V1's version adds a downstream head, while V0's adds tier-specific branches.

#### Unique V1 Contributions to Intervention Design

**2B — Curriculum Learning / Hard Example Mining**: V1 proposes progressively increasing rare/tail patient proportion during training. V0 does not include curriculum learning.

**3B — Pivot strategies**: V1 proposes radical alternatives if structural fixes fail:
- Residual embeddings (predict residual between tabular prediction and actual)
- Conditional embeddings (condition pretraining on tabular features)
- Direct downstream fine-tuning

V0 does not consider a "pivot" scenario. V1's residual embedding concept is particularly novel — training the model to learn ONLY what tabular features miss would directly solve the redundancy problem.

---

## 6. Cross-Validation Against Best Practices

### Agreed References

Both cite:
- Google Deep Learning Tuning Playbook (data scaling guidance)
- Kaplan et al. / Chinchilla scaling laws
- Kang et al. (2020) decoupled training

### V0-Only References

- Cui et al. (2019) class-balanced loss
- Goodfellow (2016) "more data helps underfitting not architectural limitations"
- Ruder (2019) transfer learning pretraining-downstream misalignment
- Goodhart's Law metaphor

### V1-Only References

- Yosinski et al. (2014) "How transferable are features in deep neural networks?"
- Standley et al. (2020 ICML) negative transfer in multi-task learning
- Recommendation systems analogies (YouTube, Meta)
- Chen et al. (ICML 2018) GradNorm

**Analysis**: V1's literature references are more targeted and actionable:
- Yosinski et al. explains WHY shared layers become task-specific (directly relevant)
- Standley et al. explains negative transfer from conflicting gradient signals (directly relevant)
- GradNorm provides a specific, implementable algorithm for the gradient rebalancing recommendation

V0's references are more general (Goodfellow's textbook, scaling laws) and serve more as intellectual framing than actionable guidance.

---

## 7. Confirmed Facts Inventory

### V0's Confirmed Facts (Implicit)

V0 presents findings inline without a separate "confirmed facts" section. Key facts are embedded in the hypothesis verdicts and synthesis.

### V1's Confirmed Facts (Explicit — CF1 through CF10)

V1 provides 10 numbered confirmed facts:
1. CF1: R10 hybrid = tabular baseline (0.831)
2. CF2: R10 hybrid < R6 hybrid (regression)
3. CF3: Data efficiency collapsed 6x
4. CF4: medium_top10_acc 5x improvement with zero downstream translation
5. CF5: tail_top10_acc = 0% universally
6. CF6: R@10 plateaued at ~0.855 beyond 6.8M
7. CF7: R10 config = prior rounds (no structural changes)
8. CF8: $44.53 cost, ~82% at loss floor
9. CF9: Embedding-only (0.809) is 0.029 below tabular (0.838)
10. CF10: R9 proved decoder not bottleneck, h lacks tail signal, co-occ gives +1.02 margin

**Analysis**: V1's explicit confirmed facts list is superior for traceability and auditability. Every finding is numbered and quantified. V0 presents the same information but scattered across sections. The synthesized analysis should adopt V1's practice.

---

## 8. What Each Analysis Uniquely Contributes

### V0's Unique Strengths

1. **Goodhart's Law framing**: "Optimizing the pretraining metric beyond a point degrades the downstream metric because the pretraining objective becomes an increasingly poor proxy." This is an elegant and memorable conceptual frame.

2. **Config audit table**: V0's Appendix B systematically evaluates every R10 config decision (pos_weight=200, no co-occ, no two-stage, etc.) with rationale and assessment. This is absent from V1.

3. **"What R10 Should Have Tested Instead"**: V0 explicitly states that R10 should have combined co-occurrence embeddings + per-tier gradient normalization + 3-epoch training + 6-8M data. V1 doesn't provide this retrospective recommendation.

4. **Pre-registered success/failure criteria**: V0 pre-registers outcomes for interventions (e.g., "if AUC > 0.845 → fine-tuning breaks the ceiling"). V1 pre-registers diagnostics but not interventions.

5. **Cost-first prioritization**: V0 strictly orders interventions by cost ($1.40 → $5 → $17), making the decision tree actionable.

### V1's Unique Strengths

1. **Tabular redundancy hypothesis (H1.3)**: The most important novel contribution. V0 never asks "is the embedding just a noisier version of tabular features?" V1's answer (yes, and the evidence is that hybrid = tabular baseline) reframes the entire problem.

2. **Distribution mismatch hypothesis (H1.2)**: V1 identifies the val_AUC anomaly (R10 val_AUC < R6 val_AUC) and considers population mismatch. V0 ignores this signal.

3. **Four cheap diagnostics**: Feature importance, linear probe, CKA analysis, per-code downstream contribution. Total cost ~2 hours on CPU. V0 has no diagnostic experiments.

4. **"What the embeddings COULD capture but DON'T"**: V1 explicitly lists the unique value temporal embeddings SHOULD provide (temporal dynamics, code interactions, rare event signatures, contextual meaning) and explains why the current training prevents learning them.

5. **Causal chain visualization**: V1's multi-level "WHY?" diagram is clearer than V0's three-layer text block.

6. **Recommendation system analogies**: V1 draws parallels to YouTube/Meta item embedding saturation, which provides battle-tested solution patterns (two-tower, hard negative mining, auxiliary objectives, curriculum learning).

7. **Residual embedding concept**: Training the model to predict the residual between tabular predictions and outcomes. This is a genuinely novel and targeted solution that V0 does not consider.

8. **GradNorm specificity**: V1 cites Chen et al. (ICML 2018) GradNorm and notes that simple per-tier loss decomposition is approximately a no-op (1.34× amplification, not 250×, per R9 critical review). V0 recommends "per-tier gradient normalization" without this level of specificity.

9. **Negative transfer framing**: V1 cites Standley et al. (2020) on conflicting gradient signals in multi-task learning. This frames the 6,297-code prediction problem as a multi-task learning failure — a well-studied domain with known solutions.

10. **Full experimental timeline table**: V1's appendix provides a chronological table of every experiment with date, key change, data, result, and lesson. V0 has no equivalent.

---

## 9. Synthesized Recommended Action Plan

Integrating the strengths of both analyses, ordered by cost and information value:

### Stage 0: Zero-Cost Actions (Immediate)

**0.1 — Model selection for production**: Use R6 6-8M for hybrid downstream (0.835 oot_strict) and R10 for embedding-only if needed (0.809). Use R9 v3 co-occ if lift@1pct is the priority metric (20.50). [From V0]

**0.2 — Stop scaling the same approach**: Do not invest in R11 with more data or the same architecture/loss. The evidence from 10 rounds is conclusive. [Agreed by both]

### Stage 1: Cheap Diagnostics (~2 hours CPU, <$1)

**1.1 — Embedding feature importance in CatBoost** [From V1 Diagnostic 1]
- Extract SHAP values from hybrid model
- If embedding features have <0.1% importance → tabular redundancy confirmed definitively

**1.2 — Linear probe on frozen h** [From V1 Diagnostic 2]
- Train logistic regression: h → downstream outcome
- Compare R6 vs R10 probe AUC to tabular-only
- If probe AUC ≤ tabular → h lacks unique downstream signal

**1.3 — CKA between h and tabular features** [From V1 Diagnostic 3]
- Compute CKA(h_R6, tabular) and CKA(h_R10, tabular)
- If CKA increases R6→R10 → confirms "more data → more redundancy"

**1.4 — Per-code downstream contribution** [From V1 Diagnostic 4]
- Identify which pretraining codes drive downstream AUC
- If only common codes → confirms misalignment

### Stage 2: Low-Cost Experiment (~4 GPU-hours, ~$1.40)

**2.1 — Downstream fine-tuning of R6 6-8M** [From V0 B.1]
- Fine-tune encoder end-to-end on downstream IP task
- LR 1e-5 to 1e-4, 3-5 epochs
- Pre-registered: AUC > 0.845 → fine-tuning breaks ceiling; ≤ 0.835 → representation lacks signal

**Branching logic**:
- If 2.1 succeeds → the representation HAS useful features; pipeline was the bottleneck → focus on fine-tuning workflow
- If 2.1 fails → the representation itself must change → proceed to Stage 3

### Stage 3: Structural Pretraining Fixes (~$17-45 each)

**3.1 — GradNorm + multi-epoch on 6-8M data** [From V1 1B, V0 C.1]
- Implement GradNorm (Chen et al. 2018) for dynamic per-tier gradient balancing
- Train 3-5 epochs on R6 6-8M data
- Combined with co-occurrence embedding initialization [V0 B.3]

**3.2 — Downstream-aware auxiliary objective** [From V1 1A, V0 B.2]
- Add contrastive or downstream-proxy loss during pretraining
- Forces encoder to allocate capacity to non-common-code features

**3.3 — Per-tier MLP decoder + GradNorm** [From V1 1C, V0 D.1]
- Separate nonlinear decoders for rare/tail codes
- Train end-to-end with gradient rebalancing

**3.4 — Hierarchical code supervision (CCS/CCSR)** [From V0 C.2]
- Secondary loss predicting ~280 clinical categories
- Regularizes toward clinically meaningful groupings

### Stage 4: Architectural Pivots (if Stage 3 insufficient)

**4.1 — Residual embeddings** [From V1 3B, novel]
- Train encoder to predict residual between tabular model predictions and actual outcomes
- Forces the model to learn ONLY what tabular features miss

**4.2 — Conditional embeddings** [From V1 3B]
- Condition pretraining on tabular features
- Forces orthogonal information encoding

**4.3 — Two-tower / dual-encoder** [V0 D.2 + V1 3A]
- Separate shared backbone from task-specific heads

---

## 10. Final Assessment of the Two Analyses

| Criterion | V0 | V1 | Winner |
|-----------|----|----|--------|
| **Factual accuracy** | All numbers verified against source data | All numbers verified against source data | **Tie** |
| **Completeness of evidence** | Good — covers all rounds | Better — includes val_AUC anomaly, matched-pop baseline | **V1** |
| **Hypothesis coverage** | 7 hypotheses across 4 levels | 8 hypotheses across 4 levels + tabular redundancy | **V1** |
| **Root cause identification** | Correct (misalignment + monopolization) | More precise (monopolization → tabular redundancy) | **V1** |
| **Causal mechanism** | Three-layer chain (structural → loss → task) | Five-level WHY chain with visualization | **V1** |
| **Diagnostic design** | None proposed | 4 cheap diagnostics with pre-registered criteria | **V1** |
| **Intervention prioritization** | Cost-first with pre-registered criteria | Impact-first without pre-registered criteria | **V0 for sequencing, V1 for scope** |
| **Actionability** | Highly actionable — clear first step (B.1) | More comprehensive but less decisive on first step | **V0** |
| **Literature grounding** | General references | Specific, actionable references (GradNorm, negative transfer) | **V1** |
| **Config audit / retrospective** | Present (Appendix B) | Absent | **V0** |
| **Novel insights** | Goodhart's Law framing | Tabular redundancy, residual embeddings, CKA diagnostics | **V1** |
| **Readability** | Clean, structured, tables + prose | More detailed, longer, better visualizations | **Tie (different strengths)** |

### Overall Assessment

V1 is the more complete and insightful analysis. Its identification of the tabular redundancy phenomenon (H1.3) is the single most important contribution across both documents, and its four cheap diagnostics provide a concrete path to validation. V0 is more actionable in the short term with its cost-first intervention sequencing and pre-registered success criteria.

**The ideal analysis combines V0's actionability with V1's depth** — which is what the synthesized action plan in Section 9 above achieves.

### Areas Where Neither Analysis Goes Far Enough

1. **Neither quantifies the information overlap**: Both claim embeddings are redundant with tabular, but neither has measured CKA, mutual information, or feature correlation. V1 proposes this as a diagnostic; V0 doesn't consider it.

2. **Neither examines the temporal dimension**: The temporal transformer should capture **sequence** information (order, timing, acceleration). Neither analysis investigates whether the representation actually encodes temporal structure vs. just bag-of-codes statistics. A simple diagnostic: shuffle the temporal order of codes within each patient and retrain — if metrics don't change, the model isn't learning temporal patterns.

3. **Neither considers the downstream model**: Both focus on the pretraining side. But the downstream CatBoost model may also play a role — perhaps it cannot effectively combine embedding features with tabular features (e.g., if the embedding features are on a different scale, or if CatBoost's tree splits don't capture the complementary information).

4. **Neither considers val/test split methodology**: The downstream evaluation uses 30% sample. Neither analysis examines whether the evaluation methodology itself introduces noise that obscures real improvements (e.g., if the hybrid regression from R6→R10 is within the confidence interval of the evaluation).

5. **Neither provides confidence intervals**: All downstream comparisons are point estimates. The -0.4pp hybrid regression (R6→R10) could be within statistical noise. Neither analysis computes or estimates confidence intervals for the downstream metrics.
