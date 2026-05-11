# Clinical TE: Experimentation Evolution Analysis (Rounds 1–10)

## Purpose

This document provides a detailed analysis of the experimentation design evolution across all 10 rounds, with specific attention to:
1. What data/configurations were used in each round and which comparisons are valid (same data, same config, single variable change)
2. How to properly source evidence for presentation slides 9–22 (architecture ablation + MoE investigation)
3. How the MoE section should be restructured to two slides based on the feedback

---

## Part 1: Round-by-Round Experimentation Design and Validity

### Round 1 — Architecture Search (November 2025)

**Data**: Single LOB, ~150K members, 3 epochs, single T4 GPU

**Experiments run**: 7 sub-experiments (exp1 through exp5)

| Exp | Architecture | Key Variable Tested |
|-----|-------------|-------------------|
| exp1 | Dense baseline (vanilla PyTorch transformer, FP32, nhead=16, max-pool) | Baseline |
| exp2 | Dense + Flash Attention (FP16, nhead=8, RoPE, max-pool) | Flash Attention lossless? |
| exp2b | Dense + Flash + Learned Attention Pooling | LAP vs MaxPool |
| exp3 | Flash + MoE (8 experts, top-2, Switch aux w=0.01) | Conditional computation |
| exp3b | Flash + MoE + LAP | MoE + pooling interaction |
| exp4 | Flash + MoE + 1 shared expert | Shared expert prevents collapse? |
| exp5 | Flash + MoE (16 fine-grained experts, top-5) | More smaller experts? |

**Valid comparisons within Round 1** (same data, same training, single variable):
- exp1 vs exp2: Effect of Flash Attention + FP16 (note: also changes nhead 16→8, head_dim 16→32)
- exp2 vs exp2b: Effect of LAP vs MaxPool (single variable)
- exp2b vs exp3: Effect of adding MoE to the best dense architecture (changes: +MoE layers 2-5, +aux loss)
- exp3 vs exp3b: Effect of LAP in MoE context
- exp3 vs exp4: Effect of shared expert
- exp3 vs exp5: Effect of fine-grained expert design (16 experts, top-5)

**Invalid comparison**: exp1 vs exp3/4/5 — multiple variables change simultaneously (Flash, FP16, nhead, MoE)

**Key results**:

| Exp | recall@1 | recall@10 | val_loss | Cost |
|-----|----------|-----------|----------|------|
| exp1 | 0.697 | 0.9474 | 0.00275 | $2.48 |
| exp2 | 0.697 | 0.9430 | 0.00275 | $1.87 |
| exp2b | 0.698 | 0.9472 | 0.00273 | $1.52 |
| exp3 | 0.305 | 0.7766 | 0.00346 | $2.55 |
| exp3b | 0.305 | 0.7754 | 0.00349 | $1.13 |
| exp4 | 0.305 | 0.7752 | 0.00348 | $1.11 |
| exp5 | 0.305 | 0.7752 | 0.00347 | $1.46 |

**Interpretation**:
- Dense architecture variants (exp1/2/2b) are functionally equivalent on quality (~0.947 recall@10), with exp2b cheapest. Flash Attention introduces no quality degradation; LAP is faster and marginally better.
- ALL MoE variants plateau at exactly recall@1=0.305 — a 56% drop from dense. The identical plateau across 4 different MoE designs signals a systematic failure, not a tuning problem.
- Root cause analysis (3-expert panel) identified three interacting failures: (1) aux loss 13x larger than task loss dominates gradients, (2) cold router init at std=0.01 causes arbitrary routing, (3) SwiGLU→GELU activation mismatch at the MoE boundary creates a representational bottleneck.

---

### Round 2 — MoE Ablation: Systematic Debugging (November 16, 2025)

**Data**: Single LOB, ~150K members, 2 epochs (slightly fewer than R1)

**Purpose**: Fix Round 1 MoE failures by testing one fix at a time, informed by root cause analysis.

| Exp | Fix Applied (vs R1 exp3) | Target Root Cause |
|-----|-------------------------|------------------|
| exp3 | None (baseline MoE, re-run) | Control |
| exp3a | SwiGLU activation in MoE expert FFNs | Activation mismatch |
| exp3b | SwiGLU + LAP | Activation mismatch + pooling |
| exp3c | SwiGLU + LAP + MoE from layer 4 (not 2) | Premature MoE insertion |
| exp3d | SwiGLU + LAP + layer 4 + aux_loss=0.001 | Aux loss dominance (reduced) |
| exp6 | DeepSeek auxiliary-free bias correction | Aux loss dominance (eliminated) |

**Valid comparisons** (each pair isolates a single fix):
- exp3 vs exp3a: SwiGLU activation fix alone
- exp3a vs exp3b: Adding LAP to SwiGLU-fixed MoE
- exp3b vs exp3c: Moving MoE insertion from layer 2 to layer 4
- exp3c vs exp3d: Reducing aux_loss weight 10x (0.01→0.001)
- exp3 vs exp6: DeepSeek bias correction (eliminates aux loss from gradient entirely)

**Validity note**: The comparison chain exp3→exp3a→exp3b→exp3c→exp3d is cumulative — each experiment adds one fix on top of the previous. exp6 is a clean alternative that addresses aux loss via a fundamentally different mechanism.

**Key results**:

| Exp | recall@1 | recall@10 | val_loss |
|-----|----------|-----------|----------|
| exp3 | 0.330 | 0.830 | 0.003520 |
| exp3a | 0.320 | 0.802 | 0.003532 |
| exp3b | 0.313 | 0.824 | 0.003517 |
| exp3c | 0.311 | 0.798 | 0.003532 |
| exp3d | 0.341 | 0.835 | 0.003510 |
| **exp6** | **0.530** | **0.875** | **0.003195** |

**Important**: R2 recall@1 values (0.305→0.330) differ from R1 (0.305 exactly) because R2 ran 2 epochs vs R1's 3, and different random seeds. The relative ordering is what matters, not cross-round absolute values.

**Interpretation**:
- SwiGLU fix alone (exp3a) did NOT help — slightly worse. The activation mismatch was real but not the bottleneck.
- Layer placement fix alone (exp3c) did NOT help — marginally worse.
- Reducing aux loss 10x (exp3d) helped slightly (+0.005 recall@10 vs exp3c).
- **DeepSeek auxiliary-free (exp6) was the breakthrough**: recall@1 jumped 0.330→0.530 (+60%), recall@10 jumped 0.830→0.875 (+5.4%). Removing aux loss from gradients entirely was dramatically more effective than reducing it.
- But even exp6 at recall@10=0.875 is below the R1 dense baseline of 0.947. The gap narrowed from -18% to -7.6%.

---

### Round 3 — Ablation: Larger Dim, Kaiming Init, MoE Layer Position (November 26, 2025)

**Data**: Single LOB, ~150K members

**Purpose**: Test whether remaining MoE fixes (better router init, later MoE insertion, larger dimension) can close the gap with dense.

| Exp | Architecture | Key Variable |
|-----|-------------|-------------|
| exp2b | Flash + LAP, 256d (dense control) | Dense baseline at this data |
| exp3e | MoE + SwiGLU + LAP + layer 2, aux=0.001 | Standard MoE with fixes |
| exp6 | Auxiliary-free MoE (original) | R2 best MoE |
| exp6a | Auxiliary-free + MoE from layer 4 | Later MoE insertion |
| exp6b | Auxiliary-free + no shared expert | Remove shared expert |

**Valid comparisons**:
- exp2b vs exp6a: Dense vs best MoE (on same data — this is the first fair head-to-head at this data scale)
- exp6 vs exp6a: Effect of moving MoE from layer 2 to layer 4
- exp6 vs exp6b: Effect of removing shared expert

**Key results**:

| Exp | recall@1 | recall@10 | val_loss |
|-----|----------|-----------|----------|
| exp2b (dense) | 0.747 | 0.961 | 0.002383 |
| exp3e (aux-loss MoE) | 0.531 | 0.892 | 0.003008 |
| exp6 (aux-free) | 0.755 | 0.963 | 0.002328 |
| exp6a (aux-free, layer 4) | 0.757 | 0.962 | 0.002338 |
| exp6b (aux-free, no shared) | 0.754 | 0.959 | 0.002364 |

**Interpretation**:
- **Milestone**: Auxiliary-free MoE (exp6/6a/6b) finally matches dense (exp2b). recall@10: 0.962 vs 0.961. This is the first time any MoE configuration is competitive with dense.
- The gap between aux-loss MoE (exp3e, 0.892) and aux-free MoE (exp6, 0.963) remains enormous — confirming that the auxiliary loss in the gradient is the primary pathology.
- Moving MoE to layer 4 (exp6a) gives a trivial improvement over layer 2 (exp6): 0.757 vs 0.755 recall@1.
- Removing shared expert (exp6b) slightly hurts: 0.959 vs 0.963 recall@10.
- **Critical observation**: Even at parity, MoE (exp6a) uses 30.4M params vs dense (exp2b) at 25.3M — 20% more parameters for identical quality.

---

### Round 4 — First Multi-GPU + Downstream Evaluation (December 2025)

**Data**: Single LOB (Medicaid IP), ~247K members, DDP across 4×T4 GPUs

**Purpose**: Validate multi-GPU training infrastructure AND produce the first downstream evaluation.

**Experiments**: exp1 (dense baseline) and exp2b (Flash+LAP). No MoE experiments in this round.

**Key results** (exp2b):
- Intrinsic: recall@10=0.955, val_loss=0.00397
- Downstream (Medicaid IP risk): AUC-ROC=0.717 (test), Lift@1%=4.76x (test)
- Train/test AUC gap (0.878 vs 0.717) suggests overfitting on small Medicaid sample

**Interpretation**: This round is primarily infrastructural — proving DDP works and establishing the first downstream benchmark. Not directly comparable to R1-3 (different LOB, different data size, different GPU count). Does not inform the MoE vs dense question.

---

### Round 5 — Scale to 3 LOBs, 1.5M Members (December 30, 2025)

**Data**: 3 LOBs (Commercial + Medicare + Medicaid), ~1.5M members, bs=128, multi-GPU DDP

**Purpose**: First real-scale pretraining. Test whether MoE benefits emerge at scale.

| Exp | Architecture |
|-----|-------------|
| exp1 | Dense baseline (vanilla transformer) |
| exp2b | Flash + LAP (256d) |
| exp6 | Auxiliary-free MoE (V3) |

**Valid comparisons**: exp2b vs exp6 — same data, same training, Dense vs MoE.

**Critical context**: The metric system changed at this round. Single-LOB rounds (1-4) used simple recall@K over the single LOB. Multi-LOB rounds (5+) use micro/macro recall/NDCG across 3 LOBs jointly. **R1-4 metrics are NOT directly comparable to R5+ metrics.**

**Key results**:

| Exp | recall@5 | recall@10 | ndcg@10 | mrr | macro_auroc |
|-----|----------|-----------|---------|-----|-------------|
| exp1 (baseline) | 0.423 | 0.579 | 0.170 | 0.217 | 0.503 |
| exp2b | 0.722 | 0.828 | 0.398 | 0.341 | 0.846 |
| exp6 (V3) | 0.725 | 0.827 | 0.399 | 0.343 | 0.859 |

**Interpretation**:
- exp1 catastrophically underperformed — likely not adapted to the 3-LOB evaluation framework correctly.
- **exp2b and exp6 are virtually identical**: recall@10 0.828 vs 0.827, ndcg@10 0.398 vs 0.399. MoE adds nothing at this scale.
- exp6 has marginally better macro_auroc (0.859 vs 0.846) but this is within noise for a single run.
- **This is the definitive MoE vs dense comparison at scale**: same data (1.5M, 3 LOBs), same infrastructure (4×T4 DDP), same training config. MoE reaches parity, not superiority. Given its 40% parameter overhead and 23% throughput penalty, MoE is strictly worse than dense.

---

### Round 5.1 — Loss Function / Learning Rate Plateau Investigation (January–February 2026)

**Data**: 3 LOBs, 1.5M members (same as R5)

**Purpose**: Investigate why recall@1 is near-zero in the 3-LOB setting. Test loss functions and sampling strategies.

| Variant | Loss | Sampling |
|---------|------|----------|
| v2 | BCE + pos_weight=35 | Standard |
| v3 | BCE + pos_weight=200 | Standard |
| v4 | ASL (Asymmetric Loss, gamma-=4) | Standard |
| v5 | ASL + density-aware batching | Density (quota=20) |

**Valid comparisons**: All variants use exp2b architecture on same 1.5M data. Single variable changes: v2→v3 (pos_weight), v3→v4 (loss function), v4→v5 (sampling).

**Key results**:

| Variant | recall@1 | recall@10 | ndcg@10 | positive_brier |
|---------|----------|-----------|---------|----------------|
| v2 | 0.010 | 0.814 | — | 0.68 |
| v3 | 0.000 | 0.817 | — | 0.67 |
| v4 (ASL) | 0.240 | 0.828 | 0.468 | 0.313 |
| v5 (ASL+density) | 0.284 | 0.833 | 0.478 | 0.308 |

**Interpretation**:
- ASL was the breakthrough for recall@1 (0→0.24) and calibration (brier 0.67→0.31).
- Higher pos_weight alone (v2→v3) did NOT help recall@1 and barely moved recall@10.
- ASL + density batching (v5) gave modest additional gains over ASL alone (v4).
- Gradient tier analysis showed ASL did NOT change gradient distribution (still 85% common, 0.1% tail). ASL improves prediction confidence/ranking, not the fundamental learning dynamics.

---

### Round 6 — Data Scaling: 3-4M and 6-8M Members (January–March 2026)

**Data**: 3 LOBs. Two sub-runs: 3-4M members and 6-8M members.

**Architecture**: exp2b (256d) only. No MoE.

**Valid comparisons**: R5 exp2b (1.5M) vs R6a (3-4M) vs R6b (6-8M) — same architecture, same config, only data volume changes. These are the cleanest data-scaling comparisons.

**Key results**:

| Data | recall@10 | ndcg@10 | macro_auroc | Cost |
|------|-----------|---------|-------------|------|
| 1.5M (R5) | 0.828 | 0.398 | 0.846 | $5.73 |
| 3-4M (R6a) | 0.834 | 0.410 | 0.886 | $14.36 |
| 6-8M (R6b) | 0.855 | 0.440 | 0.913 | $17.28 |

**Interpretation**:
- Consistent improvement with data: recall@10 +0.6% per doubling, macro_auroc +4-5% per doubling.
- The gain from 1.5M→6.8M (+2.7pp recall@10, +6.7pp macro_auroc) is larger than any architecture change across all prior rounds.
- Data scaling is the dominant lever — more impactful than Flash, LAP, SwiGLU, or MoE combined.
- Tail_top10_acc remains 0% across all data sizes.

---

### Round 7 — Dimension Scaling: 256d vs 512d (March 3, 2026)

**Data**: 3 LOBs, 1.5M members (controlled to isolate dimension effect)

**Architecture**: exp2b at 512d (58.6M params) vs exp2b at 256d (25.3M params, from R5)

**Valid comparison**: R5 exp2b (256d, 1.5M) vs R7 exp2b (512d, 1.5M) — same data, only dimension changes.

**Key results**:

| Dim | recall@10 | ndcg@10 | macro_auroc | Cost |
|-----|-----------|---------|-------------|------|
| 256d | 0.828 | 0.398 | 0.846 | $5.73 |
| 512d | 0.833 | 0.415 | 0.866 | $6.49 |

**Interpretation**:
- 512d provides modest improvement (+0.5pp recall@10, +2pp macro_auroc) for 2.3x parameter increase.
- The improvement is real but small — suggests the bottleneck is not model capacity.

---

### Round 8 — 512d at 6-8M Scale (March 5, 2026)

**Data**: 3 LOBs, 6-8M members

**Architecture**: exp2b at 512d (58.6M params)

**Valid comparison**: R6b (256d, 6-8M) vs R8 (512d, 6-8M) — same data, only dimension changes. This is the definitive dim-scaling test at scale.

**Key results**:

| Dim + Data | recall@10 | ndcg@10 | macro_auroc | Cost |
|------------|-----------|---------|-------------|------|
| 256d, 6-8M (R6b) | 0.855 | 0.440 | 0.913 | $17.28 |
| 512d, 6-8M (R8) | 0.858 | 0.442 | 0.914 | $19.68 |

**Interpretation**:
- At scale, 512d adds almost nothing: +0.3pp recall@10, +0.1pp macro_auroc for +14% cost and +42% memory.
- **Decision**: 256d is sufficient. The representation bottleneck is not capacity (the model has enough parameters to learn), it's what information the gradient pushes the model to encode (common-code dominance).

---

### Round 9 — Decoupled Training / Decoder-Encoder Bottleneck Test (March 10, 2026)

**Data**: 3 LOBs, 1.5M members, 256d

**Purpose**: Determine whether the learning plateau is a decoder bottleneck (fixable by retraining the decoder) or an encoder bottleneck (requires fundamental change to how representations are learned).

**Method**: Kang et al. (ICLR 2020) Two-Stage Decoupled Training:
- Stage 1: Standard pretraining (encoder + decoder trained normally)
- Stage 2: Freeze encoder, retrain decoder only on code-balanced batches

**Valid comparison**: Stage 1 output vs Stage 2 output (same model, only decoder changed)

| Metric | Stage 1 | After Stage 2 (v1) |
|--------|---------|-------------------|
| Stage 2 loss | — | 0.026 (from 0.44) |
| Tail margin | +1.66 | -0.06 |
| Tail top10_acc | 0% | 0% |

**Interpretation**:
- Stage 2 decoder loss converged 20x better — the decoder DID learn.
- But tail margin collapsed and tail_top10_acc stayed at 0%.
- **Definitive conclusion**: The bottleneck is the encoder representation `h`, not the decoder. The encoder, shaped by 85% common-code gradient, produces representations that lack discriminative features for tail codes. Tail input embeddings are homogenized (std=0.03) — the encoder receives nearly identical inputs for patients with and without tail codes.
- This eliminates all decoder-only interventions (different loss, different pos_weight, decoder architecture changes) from the solution space.

---

### Round 10 — Formal Full-Scale Training (March 12, 2026)

**Data**: 3 LOBs, full 11M members, 256d, exp2b, pos_weight=200

**Purpose**: Production-quality pretraining on maximum available data.

**Key results**:

| Metric | R10 (11M) |
|--------|-----------|
| recall@10 | 0.853 |
| micro_recall@10 | 0.563 |
| balanced_top10 | 0.263 |
| medium_top10 | 0.200 |
| tail_top10 | 0% |
| macro_auroc | 0.920 |
| Training time | 32 hours |
| Cost | $44.53 |

**Valid comparison** (data scaling series, all exp2b 256d):

| Data | recall@10 | macro_auroc | medium@10 | tail@10 |
|------|-----------|-------------|-----------|---------|
| 1.5M (R5) | 0.828 | 0.846 | 0.002 | 0% |
| 3-4M (R6a) | 0.834 | 0.886 | 0.040 | 0% |
| 6-8M (R6b) | 0.855 | 0.913 | 0.043 | 0% |
| 11M (R10) | 0.853 | 0.920 | 0.200 | 0% |

**Interpretation**:
- Medium_top10 jumped from 4.3% (6-8M) to 20% (11M) — a threshold-crossing effect. Medium codes crossed from ~150 to ~1,100 occurrences per code, enough for the model to learn.
- macro_auroc continued improving (0.913→0.920).
- recall@10 slightly decreased (0.855→0.853) — within noise, possibly due to increased code vocabulary diversity at scale.
- Tail_top10 remains 0%. Estimated data needed for tail: 100-1000x current (1-10B members). Not feasible.

---

## Part 2: Which Round's Data Should Source Each Slide

### Problem Statement

The original proposal (slides 9-22) mixes evidence from different rounds with different data sizes, different LOBs, and different metric systems. This creates invalid comparisons. Below is a corrected mapping.

### Slide 9-10: Dense Architecture Ablation

**Source**: Round 1 (single LOB, ~150K, 3 epochs). This is valid because exp1/exp2/exp2b all use the same data with single variable changes. The dense comparisons are internally consistent.

### Slide 11: MoE vs Dense (Initial Failure)

**Source**: Round 1. Valid comparison — exp2b vs exp3/3b/4/5 all on same data. The "0.305 wall" is a Round 1 finding.

### Slide 12: Multi-Round Validation

**Problem in original proposal**: The slide compares R1 (single LOB, 150K, single GPU) vs R2 (single LOB, 150K, 4 GPU) vs R4 (Medicaid, 247K, 4 GPU). These are on different data and infrastructure, making the recall@10 values not directly comparable.

**Correction**: The valid multi-round comparison for MoE vs dense is:
- Round 3: exp2b (0.961) vs exp6a (0.962) — same single LOB, ~150K data
- Round 5: exp2b (0.828) vs exp6 V3 (0.827) — same 3 LOBs, 1.5M data

These two comparisons are each internally valid and together establish the pattern across two different data scales.

### Slides 13-22 (MoE Investigation): Restructured to 2 Slides

Per the feedback, the 10-slide MoE section should collapse to 2 slides:

**MoE Slide 1: MoE Variants and Load-Balancing Experiments**

Evidence sourced from Round 1 + Round 2 + Round 3 (all on same single LOB, ~150K data, making cross-experiment comparisons valid):

| Experiment | Round | Fix Applied | recall@10 | Status |
|------------|-------|-----------|-----------|--------|
| exp3 (Switch aux w=0.01) | R1 | Baseline MoE | 0.777 | 56% below dense |
| exp3b (+ LAP) | R1 | Pooling variant | 0.775 | No improvement |
| exp4 (+ shared expert) | R1 | Shared expert | 0.775 | No improvement |
| exp5 (16 fine-grained) | R1 | More experts | 0.775 | No improvement |
| exp3a (SwiGLU experts) | R2 | Fix activation mismatch | 0.802 | Marginal |
| exp3c (layer 4 insertion) | R2 | Later MoE | 0.799 | Marginal |
| exp3d (aux=0.001) | R2 | Reduce aux loss 10x | 0.835 | Slight help |
| exp6 (DeepSeek bias) | R2 | Eliminate aux from gradient | 0.875 | Major improvement |
| exp6a (DeepSeek + layer 4) | R3 | + later insertion | 0.962 | Matches dense |
| exp6b (no shared expert) | R3 | Remove shared | 0.959 | Slightly worse |

Round 3 head-to-head (same data, same config):
- exp2b (dense): recall@10 = 0.961
- exp6a (best MoE): recall@10 = 0.962

Round 5 head-to-head (same data, same config, at scale):
- exp2b (dense): recall@10 = 0.828
- exp6 V3 (best MoE): recall@10 = 0.827

**Conclusion**: After 12 MoE experiments across 3 rounds, best MoE achieves parity — never superiority — with 20-40% more parameters.

**MoE Slide 2: Why MoE Doesn't Work Here — Mechanistic Inference**

Three structural reasons, derived from the experimental evidence above:

1. **Scale mismatch**: MoE benefits in literature emerge at >1B params. At 25-50M, routing overhead exceeds sparse computation benefit. Our model is 20-40x below the threshold.

2. **Auxiliary loss gradient conflict** (the primary pathology): At convergence, aux_loss contribution = 0.01 × 4.0 = 0.04, while prediction loss = 0.003. The balancing loss is 13x larger — gradients optimize for load balance, not code prediction. Removing aux loss entirely (DeepSeek bias correction) yielded the largest single improvement across all MoE experiments.

3. **Code frequency imbalance amplifies routing failure**: Gini=0.939 (top 25% codes = 98.8% of occurrences). Router learns to route based on dominant signal → experts specialize in common codes → sparse activation amplifies frequency bias rather than enabling patient-archetype specialization. This interacts with the domain homogeneity problem: clinical claims is a single domain where patient heterogeneity manifests as different code combinations, not fundamentally different computational patterns.

---

## Part 3: Experimentation Design Evolution — Narrative Arc

### Phase 1: Architecture Discovery (Rounds 1-3)

**Question**: What is the best transformer architecture for clinical code prediction?

**Method**: Controlled ablation on a small single-LOB dataset (~150K members). Seven architectures in Round 1, systematic MoE debugging in Rounds 2-3.

**Answer**: exp2b (Flash Attention + Learned Attention Pooling + SwiGLU + RoPE, 256d). This is the cheapest, fastest, and highest-quality architecture. MoE adds parameters without adding performance.

**Key design principle**: Each experiment changed exactly one variable. Round 1 established the baseline. Round 2 diagnosed and fixed MoE failures one-at-a-time. Round 3 validated the fixes and established MoE parity (but not superiority).

**Data consistency**: Rounds 1-3 all used the same single LOB, ~150K dataset. Cross-round comparisons within this phase are approximately valid (same data distribution, same LOB), though not perfectly controlled (different GPU counts in R2-3, different epoch counts).

### Phase 2: Infrastructure & First Downstream (Round 4)

**Question**: Does the architecture scale to multi-GPU DDP and produce useful downstream embeddings?

**Method**: First 4-GPU run on Medicaid data, with downstream IP risk prediction.

**Answer**: Yes — DDP training works. Downstream AUC=0.717, establishing the first benchmark.

**Transitional round**: This round bridges architecture search (Phase 1) and scaling (Phase 3). It does not contribute to the MoE vs dense question or the scaling analysis.

### Phase 3: Scaling the Winning Architecture (Rounds 5-8)

**Question**: How does performance scale with data volume and model dimension?

**Method**: Systematic scaling of exp2b across three axes:
- Data: 1.5M (R5) → 3-4M (R6a) → 6-8M (R6b) → 11M (R10)
- Dimension: 256d (R5/R6) vs 512d (R7/R8)
- One final MoE comparison at scale: exp6 V3 vs exp2b at 1.5M (R5)

**Answers**:
1. Data scaling is the dominant lever. 1.5M→6.8M improved macro_auroc by 6.7pp — more than any architecture change.
2. Dimension scaling (256→512) adds marginal gain (+0.3pp recall@10 at 6-8M scale) for disproportionate cost (+14% cost, +42% memory).
3. MoE at scale (1.5M, 3 LOBs) matches dense exactly — confirming the R3 finding.
4. Medium codes cross a learning threshold between 6-8M and 11M data.
5. Tail codes remain at 0% regardless of data volume — a structural problem.

**Data consistency**: Rounds 5, 6a, 6b, 7, 8, and 10 are all on the same 3-LOB joint dataset with the same metric system, making them directly comparable. The only changing variables are data volume and dimension.

### Phase 4: Understanding the Bottleneck (Rounds 5.1, 9)

**Question**: Why does performance plateau, and where is the bottleneck?

**Method**:
- R5.1: Loss function ablation (BCE vs ASL vs focal) and sampling strategy (standard vs density-aware)
- R9: Encoder vs decoder bottleneck test (Kang et al. two-stage decoupled training)

**Answers**:
1. The plateau is caused by emergent gradient starvation: common codes capture 85% of gradient by step 12K, starving tail codes.
2. ASL improves calibration and ranking but does NOT change gradient distribution.
3. The bottleneck is the encoder representation `h`, not the decoder. Decoder-only interventions (retraining, different loss) cannot fix it.
4. Tail code input embeddings are homogenized (std=0.03) — the encoder receives nearly identical inputs for tail-present and tail-absent patients.

**Data consistency**: Both R5.1 and R9 use 1.5M, 3 LOBs, exp2b — same baseline. Internally consistent.

### Phase 5: Production and Handoff (Round 10)

**Question**: What does the best model look like at maximum data scale?

**Method**: Full 11M member, 3-LOB, 256d exp2b training (32 hours, $44.53).

**Answer**: macro_auroc=0.920, medium_top10=20% (a 5x jump from 6-8M), tail_top10=0%. The model is production-quality on common and medium codes. Tail remains unsolved.

---

## Part 4: Cross-Round Comparison Validity Matrix

This matrix shows which rounds can be validly compared for which purposes:

| Comparison Purpose | Valid Rounds | Why Valid | Invalid Comparison Trap |
|-------------------|-------------|----------|------------------------|
| Dense architecture ablation | R1 (exp1 vs exp2 vs exp2b) | Same data, single variable | Don't compare R1 dense to R5 dense (different data, different metrics) |
| MoE vs Dense (small scale) | R1 (exp2b vs exp3/4/5), R3 (exp2b vs exp6a) | Same data, single variable | Don't compare R1 MoE to R5 dense (different everything) |
| MoE vs Dense (at scale) | R5 (exp2b vs exp6 V3) | Same 3-LOB 1.5M data | Don't compare to R1 MoE (different metric system) |
| MoE fix ablation | R2 (exp3→exp3a→exp3d→exp6) | Same data, cumulative fixes | R2 metrics differ from R1 (2 vs 3 epochs) |
| Data scaling | R5→R6a→R6b→R10 (all exp2b 256d) | Same arch, only data changes | Don't compare to R1-4 (different metric system) |
| Dimension scaling | R5 vs R7 (at 1.5M), R6b vs R8 (at 6-8M) | Same data, only dim changes | Must compare at same data size |
| Loss function ablation | R5.1 v2→v3→v4→v5 | Same data, single variable | Don't compare to R1 BCE results |
| Encoder vs decoder bottleneck | R9 Stage 1 vs Stage 2 | Same model, decoder-only change | Don't compare R9 to R5 (different purpose) |

---

## Part 5: Recommended Slide Structure (Revised)

Based on the feedback and validity analysis:

### Slides 9-10: Dense Architecture Ablation (Source: Round 1)
- Slide 9: Experiment matrix with controls (unchanged from proposal)
- Slide 10: Dense results table showing exp1→exp2→exp2b progression (unchanged)

### Slide 11: MoE Initial Failure (Source: Round 1)
- Table of exp2b vs exp3/3b/4/5 showing the "0.305 wall"
- All from same Round 1 data — valid comparison

### Slide 12: [REMOVED — merged into new MoE slides]

### Slide 13 (NEW MoE Slide 1): MoE Variants and Load-Balancing Journey

Content: Chronological walk through 12 MoE experiments across R1-R3, all on the same ~150K single-LOB data. Show the progression from complete failure (0.305 wall) through systematic debugging (SwiGLU fix, layer placement, aux loss reduction) to the DeepSeek bias correction breakthrough (0.875) to final parity (0.962 vs dense 0.961). Include the R5 at-scale confirmation (0.827 vs 0.828).

Key visual: Recall@10 bar chart with all MoE experiments ordered chronologically, with dense baseline as a horizontal reference line.

### Slide 14 (NEW MoE Slide 2): Why MoE Doesn't Work Here

Content: Three mechanistic reasons derived from the experimental evidence. Include the definitive head-to-head comparison table (R3 and R5), resource comparison (parameters, memory, throughput, cost), and the inference that at this model scale (25-50M params) and domain homogeneity (single clinical domain), dense architecture is strictly superior.

Key visual: Summary comparison table — Dense wins on every metric (quality, parameters, memory, speed, cost).

### Slides 15+ onward: Renumber remaining slides accordingly

The learning plateau investigation (originally slides 23-32) now starts earlier, and the total slide count decreases by 8 slides (from 42 to ~34).
