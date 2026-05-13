# Clinical TE: Experimentation Evolution Analysis (Rounds 1–10)

## Purpose

This document provides a detailed analysis of the experimentation design evolution across all 10 rounds, with specific attention to:
1. What data/configurations were used in each round and which comparisons are valid (same data, same config, single variable change)
2. Factual corrections to previously cited evidence (downstream metric inconsistency, cross-round comparison invalidity, exp1→exp2 confound)
3. How the MoE section should be restructured to two slides based on feedback
4. The complete 3-slide presentation structure: Architecture → MoE Lesson → Learning Plateau & Path Forward

---

## Part 1: Round-by-Round Experimentation Design and Validity

### Round 1 — Architecture Search (November 2025)

**Data**: Single LOB, ~150K members, 3 epochs, single T4 GPU

**Experiments run**: 7 sub-experiments (exp1 through exp5)

| Exp | Architecture | Key Variable Tested |
|-----|-------------|-------------------|
| exp1 | Dense baseline (vanilla PyTorch transformer, FP32, nhead=16, post-norm, GELU, no RoPE, max-pool) | Baseline |
| exp2 | Dense + Flash Attention + FP16 + nhead=8 + RoPE + SwiGLU + pre-norm + max-pool | Combined modernization |
| exp2b | Dense + Flash + all exp2 changes + Learned Attention Pooling | LAP vs MaxPool |
| exp3 | Flash + MoE (8 experts, top-2, Switch aux w=0.01) | Conditional computation |
| exp3b | Flash + MoE + LAP | MoE + pooling interaction |
| exp4 | Flash + MoE + 1 shared expert | Shared expert prevents collapse? |
| exp5 | Flash + MoE (16 fine-grained experts, top-5) | More smaller experts? |

**Valid comparisons within Round 1** (same data, same training):
- exp2 vs exp2b: Effect of LAP vs MaxPool — **single variable, cleanest comparison**
- exp2b vs exp3: Effect of adding MoE to the best dense architecture (changes: +MoE layers 2-5, +aux loss)
- exp3 vs exp3b: Effect of LAP in MoE context
- exp3 vs exp4: Effect of shared expert
- exp3 vs exp5: Effect of fine-grained expert design (16 experts, top-5)

**CONFOUNDED comparison — exp1 vs exp2**: This simultaneously changes **6 variables**: (1) Flash Attention kernel, (2) FP32→FP16, (3) nhead 16→8 / head_dim 16→32, (4) GELU→SwiGLU, (5) post-norm→pre-norm, (6) no RoPE→RoPE. The Round 1 root cause analysis acknowledged variables 1-3 and noted that since exp2 retains baseline accuracy, the confound doesn't alter the conclusion. But the presentation CANNOT claim "each component ablated with single variable changes" for exp1→exp2. It is a bundle of modernizations applied together, with the net effect being quality-neutral and cost-reducing.

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
- Dense architecture variants (exp1/2/2b) are functionally equivalent on quality (~0.947 recall@10), with exp2b cheapest. The bundled modernization (exp1→exp2) introduces no quality degradation while reducing cost 25%. Adding LAP (exp2→exp2b) recovers any marginal loss and further reduces cost.
- ALL MoE variants plateau at exactly recall@1=0.305 — a 56% drop from dense. The identical plateau across 4 different MoE designs signals a systematic failure, not a tuning problem.
- Root cause analysis (3-expert panel) identified three interacting failures: (1) aux loss 13x larger than task loss dominates gradients, (2) cold router init at std=0.01 causes arbitrary routing, (3) SwiGLU→GELU activation mismatch at the MoE boundary creates a representational bottleneck.

---

### Round 2 — MoE Ablation: Systematic Debugging (November 16, 2025)

**Data**: Single LOB, ~150K members, 2 epochs (slightly fewer than R1's 3 epochs)

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

**CROSS-ROUND WARNING**: R2 metrics are NOT directly comparable to R1 metrics in absolute value. R2 ran 2 epochs (vs R1's 3) with different random seeds. R2 MoE baseline exp3 shows recall@10=0.830 vs R1's 0.777 — this difference is from the epoch/seed change, not any architectural fix. Only within-R2 relative comparisons are valid.

**Key results (all within Round 2)**:

| Exp | recall@1 | recall@10 | val_loss |
|-----|----------|-----------|----------|
| exp3 | 0.330 | 0.830 | 0.003520 |
| exp3a | 0.320 | 0.802 | 0.003532 |
| exp3b | 0.313 | 0.824 | 0.003517 |
| exp3c | 0.311 | 0.798 | 0.003532 |
| exp3d | 0.341 | 0.835 | 0.003510 |
| **exp6** | **0.530** | **0.875** | **0.003195** |

**Interpretation**:
- SwiGLU fix alone (exp3a) did NOT help — slightly worse. The activation mismatch was real but not the bottleneck.
- Layer placement fix alone (exp3c) did NOT help — marginally worse.
- Reducing aux loss 10x (exp3d) helped slightly (+0.005 recall@10 vs exp3c).
- **DeepSeek auxiliary-free (exp6) was the breakthrough**: recall@1 jumped 0.330→0.530 (+60%), recall@10 jumped 0.830→0.875 (+5.4%) — both within-R2 comparisons. Removing aux loss from gradients entirely was dramatically more effective than reducing it.

---

### Round 3 — Ablation: MoE Layer Position, Shared Expert (November 26, 2025)

**Data**: Single LOB, ~150K members

**Purpose**: With DeepSeek bias correction established, test remaining MoE variants. This round includes a dense control (exp2b), enabling the first clean head-to-head at this data scale.

| Exp | Architecture | Key Variable |
|-----|-------------|-------------|
| exp2b | Flash + LAP, 256d (dense control) | Dense baseline at this data |
| exp3e | MoE + SwiGLU + LAP + layer 2, aux=0.001 | Standard MoE with all R2 fixes |
| exp6 | Auxiliary-free MoE (original config) | R2 best MoE, re-run |
| exp6a | Auxiliary-free + MoE from layer 4 | Later MoE insertion |
| exp6b | Auxiliary-free + no shared expert | Remove shared expert |

**Valid comparisons** (all within Round 3, same data):
- **exp2b vs exp6a**: Dense vs best MoE — the definitive small-scale head-to-head
- exp6 vs exp6a: Effect of moving MoE from layer 2 to layer 4
- exp6 vs exp6b: Effect of removing shared expert
- exp3e vs exp6: Aux-loss MoE vs aux-free MoE (confirms the R2 finding)

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

**METRIC SYSTEM BREAK**: The metric system changed at this round. Single-LOB rounds (1-4) used simple recall@K over one LOB. Multi-LOB rounds (5+) use micro/macro recall/NDCG across 3 LOBs jointly. **R1-4 metrics are NOT directly comparable to R5+ metrics in absolute value.** R1 exp2b recall@10=0.947 vs R5 exp2b recall@10=0.828 does NOT mean performance dropped — it means the evaluation is fundamentally different.

**Key results**:

| Exp | recall@5 | recall@10 | ndcg@10 | mrr | macro_auroc |
|-----|----------|-----------|---------|-----|-------------|
| exp1 (baseline) | 0.423 | 0.579 | 0.170 | 0.217 | 0.503 |
| exp2b | 0.722 | 0.828 | 0.398 | 0.341 | 0.846 |
| exp6 (V3) | 0.725 | 0.827 | 0.399 | 0.343 | 0.859 |

**Interpretation**:
- exp1 catastrophically underperformed — likely not adapted to the 3-LOB evaluation framework correctly.
- **exp2b and exp6 are virtually identical**: recall@10 0.828 vs 0.827, ndcg@10 0.398 vs 0.399. MoE adds nothing at this scale.
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

**Purpose**: Two interleaved goals:
1. Determine whether the learning plateau is a decoder bottleneck or an encoder bottleneck (Kang et al. ICLR 2020 two-stage decoupled training)
2. Test PPMI+SVD co-occurrence embedding pre-training to break input-level homogenization

**Method — Decoupled Training**:
- Stage 1: Standard pretraining (encoder + decoder trained normally on natural distribution)
- Stage 2: Freeze encoder, retrain decoder only on code-balanced batches with focused loss (only the 16 target codes per batch, not all 6,297)

**Valid comparison**: Stage 1 output vs Stage 2 output (same model, only decoder changed)

| Metric | Stage 1 | After Stage 2 (v1) |
|--------|---------|-------------------|
| Stage 2 loss | — | 0.026 (from 0.44 — 20x convergence) |
| Tail margin | +1.66 | -0.06 |
| Tail top10_acc | 0% | 0% |
| Tail positive logit | -13.49 | -3.93 (improved) |
| Gradient tail_frac (during S2) | 0.1% (S1 baseline) | 40.2% (targeted batching working) |

**Interpretation — Decoder Bottleneck Test**:
- Stage 2 decoder loss converged 20x better — the decoder DID learn from the balanced batches.
- Gradient tail fraction reached 40.2% during Stage 2 — the targeted batching and focused loss successfully redirected gradients to tail codes.
- But tail margin collapsed to -0.06 (noise-level) and tail_top10_acc stayed at 0%.
- **Definitive conclusion**: The bottleneck is the encoder representation `h`, not the decoder. The encoder, shaped by 85% common-code gradient during Stage 1, produces representations that lack discriminative features for tail codes. No decoder intervention can extract signal that isn't in `h`.
- **Why Kang et al. works for ImageNet but not here**: In vision, the input (pixels) contains full information about the object class regardless of training distribution. In clinical data, the input embeddings for tail codes are homogenized (std=0.03) — the encoder receives nearly identical inputs for patients with and without tail codes. Unlike pixels, input-level distinctiveness must be learned, and gradient starvation prevents that learning.
- This eliminates all decoder-only interventions (different loss, different pos_weight, decoder architecture changes, MLP decoder) from the solution space.

**Method — Co-occurrence Embedding Pre-training (PPMI+SVD)**:
- Compute Positive Pointwise Mutual Information from code co-occurrence in training data
- SVD decomposition to get dense 256-d code embeddings
- Initialize the model's code embedding layer with these vectors instead of random init
- Hypothesis: breaks input-level homogenization by encoding co-occurrence structure before training begins

**Results — PPMI+SVD**:
- Tail embedding std improved from ~0.03 (random init) to 0.077 (PPMI+SVD) — 2.5x more differentiated
- First ever positive tail margin (+1.02 in Stage 2 of v2 experiment)
- Tail top10_acc still 0% — the co-occurrence structure alone is insufficient
- Downstream (13-LOB hybrid variant, evaluated April 2026): OOT-strict Lift@1% = **20.50** vs production baseline 19.38

**Important provenance note on the 20.50 number**: This comes from `commercial_ip_1-5M_30pctsample_downstream.json`, specifically the entry `exp2b_round9_v3_+W13lobs_1-5M_catboost_hybrid`. This is a 13-LOB hybrid model (embeddings + 533 tabular features), distinct from the 3-LOB intrinsic experiment documented in the March 10 session summary. The embedding-only variant achieved Lift@1% = 17.43 (3-LOB catboost embedding-only).

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

**What the model learns well** (per-code-type intrinsic, R10):

| Code Type | Codes | micro_R@10 |
|-----------|-------|-----------|
| GPI Medications | 95 | 76.3% |
| Place of Service | 70 | 75.1% |
| Provider Taxonomy | 242 | 54.7% |
| Procedure Groups | 2,457 | 35.1% |
| ICD-10 Diagnosis | 1,931 | 31.0% |
| DRG Codes | 879 | 10.4% |

The model excels at medications (76%) and place of service (75%) — concentrated vocabularies with strong temporal patterns. These are genuinely useful clinical signals.

**Valid comparison — data scaling series (all exp2b 256d, same metric system)**:

| Data | recall@10 | macro_auroc | medium@10 | tail@10 |
|------|-----------|-------------|-----------|---------|
| 1.5M (R5) | 0.828 | 0.846 | 0.002 | 0% |
| 3-4M (R6a) | 0.834 | 0.886 | 0.040 | 0% |
| 6-8M (R6b) | 0.855 | 0.913 | 0.043 | 0% |
| 11M (R10) | 0.853 | 0.920 | 0.200 | 0% |

**Interpretation**:
- Medium_top10 jumped from 4.3% (6-8M) to 20% (11M) — a threshold-crossing effect. Medium codes crossed from ~150 to ~1,100 occurrences per code, enough for the model to learn. This validates the data scaling hypothesis and proves the architecture CAN learn lower-frequency codes given sufficient exposure.
- macro_auroc continued improving (0.913→0.920).
- recall@10 slightly decreased (0.855→0.853) — within noise, possibly due to increased code vocabulary diversity at scale.
- Tail_top10 remains 0%. At 11M members, each tail code gets ~57 occurrences. Estimated threshold: ~1,000+ occurrences. Estimated data needed: 100-1000x current (1-10B members). Not feasible by volume alone.

---

## Part 2: Factual Corrections and Errata

### Erratum 1: Downstream Metric Inconsistency (HIGH SEVERITY)

The previously cited downstream comparison used **different evaluation sets** for baselines vs TE:

| Model | Metric field actually used | Evaluation set |
|-------|---------------------------|----------------|
| PCA(256) | `oot_lift_1pct` (non-strict) | Larger, less conservative OOT set |
| AutoEncoder(256) | `oot_lift_1pct` (non-strict) | Larger, less conservative OOT set |
| SelectKBest(256) | `oot_lift_1pct` (non-strict) | Larger, less conservative OOT set |
| TE R10 | `oot_strict_lift_1pct` (strict) | Smaller, more conservative OOT set |

Source: `commercial_ip_raw_codes_vs_te_round10_pca_ae_kbest_te.json`

**Corrected comparison** (all on same strict OOT evaluation set):

| Model | OOT-strict AUC | OOT-strict Lift@1% |
|-------|----------------|-------------------|
| PCA(256) | — | **9.00** |
| TE R10 (embedding-only) | 0.810 | **18.89** |
| Improvement | — | **+110%** (not 57%) |

**Corrected comparison** (all on same non-strict OOT set):

| Model | OOT AUC | OOT Lift@1% |
|-------|---------|-------------|
| PCA(256) | 0.756 | **11.72** |
| TE R10 (embedding-only) | — | **15.93** |
| Improvement | — | **+36%** (not 57%) |

**Recommendation**: Use the strict OOT comparison for both — the improvement is actually STRONGER (110% vs 57%) and the comparison is apples-to-apples. If strict OOT numbers are not available for baselines, explicitly label the column as "non-strict OOT" and note the caveat.

### Erratum 2: exp1→exp2 Is a 6-Variable Change (MEDIUM SEVERITY)

The presentation cannot claim "each component ablated with a single variable change" for the exp1→exp2 transition. The actual changes:

1. Attention kernel: Stock PyTorch → Flash Attention (xFormers)
2. Precision: FP32 → FP16 (mixed precision)
3. Head configuration: nhead=16 (head_dim=16) → nhead=8 (head_dim=32)
4. FFN activation: GELU → SwiGLU
5. Normalization: Post-norm → Pre-norm
6. Position encoding: None → RoPE

The net effect is quality-neutral and cost-reducing (+25% speed, -25% cost). But the claim should be: "We bundled a set of modern transformer best practices (Flash, FP16, SwiGLU, RoPE, pre-norm) and confirmed the bundle is quality-neutral" — not that each was ablated individually in Round 1. The only clean single-variable ablation in Round 1 is exp2→exp2b (MaxPool→LAP).

### Erratum 3: Cross-Round MoE Progression Table (MEDIUM SEVERITY)

Previously proposed MoE tables mixed R1 and R2 recall@10 values as if they were comparable. They are not — R2 used 2 epochs vs R1's 3, producing different absolute values for the same architecture. The R2 MoE baseline (exp3=0.830) vs R1 MoE baseline (exp3=0.777) difference is from epoch count, not any fix.

**Corrected approach**: Present the MoE story as two clean snapshots, not a single progression:
- **R1 snapshot**: Dense 0.947 vs all MoE variants ~0.777 (same round, same epochs — establishes the failure)
- **R3 snapshot**: Dense 0.961 vs best MoE 0.962 (same round, same epochs — establishes parity after fixes)
- **R5 snapshot**: Dense 0.828 vs best MoE 0.827 (same round, same data at scale — confirms parity holds)
- **Separately**: List what changed between R1 and R3 as a text narrative, not as a table with recall values across rounds

### Erratum 4: R9 Co-occurrence Downstream Number Provenance (LOW SEVERITY)

The "R9 co-occur embed (hybrid) Lift@1% = 20.50" number is real but requires precise attribution:
- Source: `commercial_ip_1-5M_30pctsample_downstream.json`, entry `exp2b_round9_v3_+W13lobs_1-5M_catboost_hybrid`
- This is a **13-LOB hybrid model** (embeddings + 533 tabular features), not the 3-LOB intrinsic model from the March 10 session summary
- The embedding-only variant achieves Lift@1% = 17.43
- The PPMI+SVD pre-training was executed on March 10, 2026; the downstream evaluation was run in April 2026

When presenting, be clear that this is a hybrid model (embeddings + tabular), not embedding-only, and that it uses 13 LOBs of pre-training data.

---

## Part 3: Experimentation Design Evolution — Narrative Arc

### Phase 1: Architecture Discovery (Rounds 1-3)

**Question**: What is the best transformer architecture for clinical code prediction?

**Method**: Controlled ablation on a small single-LOB dataset (~150K members). Seven architectures in Round 1, systematic MoE debugging in Rounds 2-3.

**Answer**: exp2b (Flash Attention + Learned Attention Pooling + SwiGLU + RoPE, 256d). This is the cheapest, fastest, and highest-quality architecture. MoE adds parameters without adding performance.

**Key design principle**: Round 1 established baselines. The exp1→exp2 transition bundled multiple modernizations (confirmed quality-neutral as a bundle). The exp2→exp2b transition is a clean single-variable ablation (LAP). Round 2 diagnosed MoE failures one-at-a-time with cumulative fixes. Round 3 validated the fixes and established MoE parity (but not superiority) in a clean head-to-head.

**Data consistency**: Rounds 1-3 all used the same single LOB, ~150K dataset. Within-round comparisons are valid. Cross-round absolute values differ due to epoch count and seed differences, but relative patterns are consistent.

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
2. Dimension scaling (256→512) adds marginal gain (+0.3pp recall@10 at 6-8M scale) for disproportionate cost (+14% cost, +42% memory). Not worth it.
3. MoE at scale (1.5M, 3 LOBs) matches dense exactly — confirming the R3 finding. MoE abandoned after this round.
4. Medium codes cross a learning threshold between 6-8M and 11M data — proving the architecture CAN learn lower-frequency codes given sufficient exposure.
5. Tail codes remain at 0% regardless of data volume — a structural problem that data scaling alone cannot solve.

**Data consistency**: Rounds 5, 6a, 6b, 7, 8, and 10 are all on the same 3-LOB joint dataset with the same metric system, making them directly comparable. The only changing variables are data volume and dimension.

### Phase 4: Understanding the Bottleneck (Rounds 5.1, 9)

**Question**: Why does performance plateau, and where is the bottleneck?

**Method**:
- R5.1: Loss function ablation (BCE vs ASL vs focal) and sampling strategy (standard vs density-aware)
- R9: Encoder vs decoder bottleneck test (Kang et al. two-stage decoupled training)
- R9: PPMI+SVD co-occurrence embedding pre-training

**Answers**:
1. **Gradient starvation is the root cause**: Common codes capture 85% of gradient by step 12K, starving tail codes. This is emergent — it doesn't exist at step 1 (all tiers at 17.8%) but develops within the first few thousand steps.
2. **ASL improves calibration and ranking but does NOT change gradient distribution**. It controls how confident predictions are, not which codes the model learns to represent.
3. **The bottleneck is the encoder representation `h`, not the decoder** (R9 definitive proof). Decoder loss converged 20x better in Stage 2, but tail margin stayed at -0.06 (noise). No decoder intervention can extract signal that isn't in `h`.
4. **Tail code input embeddings are homogenized** (std=0.03) — the encoder receives nearly identical inputs for tail-present and tail-absent patients. The gradient starvation prevents the embedding layer from learning tail-code distinctiveness.
5. **PPMI+SVD partially breaks the homogenization** — tail embedding std improved from 0.03 to 0.077, and first positive tail margin was observed. But alone it is insufficient.

**Data consistency**: Both R5.1 and R9 use 1.5M, 3 LOBs, exp2b — same baseline. Internally consistent.

### Phase 5: Production and Handoff (Round 10)

**Question**: What does the best model look like at maximum data scale?

**Method**: Full 11M member, 3-LOB, 256d exp2b training (32 hours, $44.53).

**Answer**: macro_auroc=0.920, medium_top10=20% (a 5x jump from 6-8M), tail_top10=0%. The model is production-quality on common and medium codes. Tail remains unsolved but the architecture is validated — the bottleneck is data distribution, not model design.

---

## Part 4: Cross-Round Comparison Validity Matrix

| Comparison Purpose | Valid Rounds | Why Valid | Invalid Comparison Trap |
|-------------------|-------------|----------|------------------------|
| Dense architecture bundle validation | R1 (exp1 vs exp2) | Same data, confirms bundle quality-neutral | Cannot claim single-variable ablation — 6 variables changed |
| LAP vs MaxPool | R1 (exp2 vs exp2b) | Same data, single variable | Only clean single-variable ablation in R1 dense series |
| MoE vs Dense (initial failure) | R1 (exp2b vs exp3/4/5) | Same data, single variable (+ MoE) | Don't compare to R2 MoE (different epoch count) |
| MoE fix ablation | R2 (exp3→exp3a→exp3d→exp6), within-round only | Same data, cumulative single fixes | Do NOT mix R2 values into R1 tables — different epochs |
| MoE vs Dense (small scale, after fixes) | R3 (exp2b vs exp6a) | Same data, same epochs | Don't compare R3 recall to R1 recall — different absolute scales |
| MoE vs Dense (at scale) | R5 (exp2b vs exp6 V3) | Same 3-LOB 1.5M data | Don't compare to R1-4 MoE (different metric system entirely) |
| Data scaling | R5→R6a→R6b→R10 (all exp2b 256d) | Same arch, only data changes | Don't compare to R1-4 (different metric system) |
| Dimension scaling | R5 vs R7 (at 1.5M), R6b vs R8 (at 6-8M) | Same data, only dim changes | Must compare at same data size |
| Loss function ablation | R5.1 v2→v3→v4→v5 | Same data, single variable | Don't compare to R1 BCE results |
| Encoder vs decoder bottleneck | R9 Stage 1 vs Stage 2 | Same model, decoder-only change | Don't compare R9 metrics to R5 metrics |
| Downstream TE vs baselines | R10 vs baselines from same JSON | Same evaluation protocol | VERIFY same OOT set (strict vs non-strict) — see Erratum 1 |

---

## Part 5: 3-Slide Presentation Structure (Revised)

Based on the critical review, the 3-slide structure should incorporate these corrections and address the identified gaps.

### Slide 1: Architecture — "Mirror the Data, Not the Literature"

**Walk the provided diagram bottom-up.**

**Content**:
- Bottom: 80 codes/day → code_emb → Learned Attention Pooling → 1 vector/day. No positional encoding — codes within a day are an unordered set.
- Middle: Demographics (age, gender, LOB) injected via residual sum. 200 days of history.
- Top: 6-layer Temporal Encoder. Causal masking, RoPE, SwiGLU + Flash Attention. Final embedding at last day position.
- Output: predict 6,297 grouped clinical codes for next day (BCEWithLogitsLoss, dual vocabulary)

**Evidence for design choices** (from Round 1):
- Bundle of modern transformer practices (Flash, FP16, SwiGLU, RoPE, pre-norm) confirmed quality-neutral with 25% cost reduction (exp1→exp2, same data, 3 epochs)
- LAP vs MaxPool: single-variable ablation, LAP matches quality with 3-5x faster daily encoding (exp2→exp2b)
- Present as "design validated by ablation" — but do NOT claim each component was individually ablated in isolation. The exp1→exp2 step was a bundled modernization. The clean ablation is exp2→exp2b.

**What the model DOES well** (from Round 10):
- Medications prediction: 76% recall@10, Place of Service: 75% — genuinely useful clinical signals
- Medium codes went from 0% → 20% accuracy at 11M members — proves the architecture CAN learn lower-frequency codes when given sufficient data exposure
- Embedding-only downstream AUC = 0.810 — an unsupervised 256-d vector gets within 2.8pp of production tabular pipeline (0.838)
- Hybrid (embedding + tabular): AUC = 0.831, closing the gap to 0.7pp

**Slide 1 land-the-punch**: The architecture works. 25.3M parameters, $44 to train, captures temporal clinical patterns that dimensionality reduction cannot.

---

### Slide 2: Lesson 1 — "MoE: Right Idea, Wrong Scale"

**Hypothesis**: Clinical populations are heterogeneous. MoE lets experts specialize per patient archetype. Top-2 routing, same FLOPs as dense.

**Evidence — two clean snapshots, not a progression table**:

**Snapshot 1 — Round 1** (same data, same epochs, same everything except MoE):

| Architecture | recall@10 | recall@1 |
|-------------|-----------|----------|
| Dense (exp2b) | **0.947** | **0.698** |
| MoE 8 experts (exp3) | 0.777 | 0.305 |
| MoE + shared expert (exp4) | 0.775 | 0.305 |
| MoE 16 fine-grained (exp5) | 0.775 | 0.305 |

All MoE variants plateau at exactly recall@1=0.305. 56% below dense. With 27-40% more parameters.

**What we tried to make MoE work** (Rounds 2-3 — the efforts table):

Root cause analysis after Round 1 identified three interacting failures: aux loss dominating gradients (13x larger than task loss), cold router initialization, and SwiGLU→GELU activation mismatch. We systematically addressed each, one at a time:

| What we tried | Rationale | Result | Worth the cost? |
|--------------|-----------|--------|----------------|
| SwiGLU in expert FFNs (R2) | Fix activation mismatch at MoE boundary (layers 0-1 use SwiGLU, MoE experts used GELU) | recall@10: 0.802 (worse than R2 baseline 0.830) | No — not the primary bottleneck |
| MoE from layer 4 instead of 2 (R2) | Let 4 dense layers build representations before routing | recall@10: 0.799 (no improvement) | No — premature insertion was secondary |
| Reduce aux loss 10x: 0.01→0.001 (R2) | Reduce gradient conflict between balancing and prediction | recall@10: 0.835 (+0.5pp) | Marginal — conflict reduced but not eliminated |
| **DeepSeek bias correction (R2)** | **Remove aux loss from gradient entirely; load-balance via bias vector updated outside backprop** | **recall@10: 0.875 (+4.5pp)** | **Yes — single biggest improvement** |
| Shared expert (1 shared + 7 routed) (R1,R3) | Dedicated expert for general patterns; routed experts specialize | CV improved (0.484→0.310), but no quality gain | No — 1 shared helps stability, not accuracy |
| Fine-grained: 16 experts, top-5 (R1) | More smaller experts for finer specialization | 6-7 of 16 collapsed; router gradient exploded (196x norm) | No — worse than 8 experts |
| Kaiming router init (R3) | Better initialization to avoid random early routing | Slight improvement in load balance | Marginal |
| 512d + MoE (R3) | More capacity for expert specialization | Same ceiling, 2.3x parameters | No |

Net result of all fixes combined — **parity, not superiority**:

**Snapshot 2 — Round 3** (same data, same epochs — clean head-to-head after all fixes):

| Architecture | recall@10 | Params |
|-------------|-----------|--------|
| Dense (exp2b) | **0.961** | 25.3M |
| Best MoE (exp6a, aux-free, all fixes) | **0.962** | 30.4M |

Parity — with 20% more parameters, 23% slower throughput, 18% more memory, and significantly more engineering complexity.

**Confirmed at scale — Round 5** (1.5M members, 3 LOBs, same config):

| Architecture | recall@10 |
|-------------|-----------|
| Dense (exp2b) | **0.828** |
| Best MoE (exp6 V3) | **0.827** |

**The cost-benefit verdict**:

| Dimension | Dense (exp2b) | Best MoE (exp6a) | Tax paid for MoE |
|-----------|--------------|------------------|------------------|
| recall@10 | 0.961 | 0.962 | +0.1% (noise) |
| Parameters | 25.3M | 30.4M | +20% more params |
| Peak memory | 11.1 GB | 13.5 GB | +18% more memory |
| Throughput | 1,037 samp/s | 845 samp/s | -23% slower |
| Training cost | $5.73 | $7.04 | +23% more expensive |
| Engineering complexity | Simple | Router tuning, bias_lr, collapse monitoring | Significantly higher |

**Why — three structural reasons**:
1. **Scale mismatch**: MoE benefits emerge at >1B params (Mixtral, DeepSeek). At 25M, we are 40x below the threshold where routing overhead pays for itself.
2. **Aux loss gradient hijacking**: The primary failure mode. Load-balancing loss was 13x larger than task loss. Removing it entirely was the only effective fix — but even then, it only closed the gap to parity.
3. **Domain homogeneity**: MoE excels in multi-domain settings (translation/summarization/code). Clinical claims is one domain — patient heterogeneity manifests as different code combinations, not fundamentally different computational patterns. The co-occurrence analysis (Section 1.2 of the data saturation study) confirms this: 86% of all code-pair diversity involves at least one common code. There aren't distinct "patient archetypes" that need separate expert processing — there's a continuous spectrum dominated by common chronic conditions.

**Slide 2 land-the-punch**: We invested 12 experiments across 3 rounds trying to make MoE work. We fixed every identified root cause. The best MoE matches dense — never beats it. The taxes (20% params, 23% slower, higher complexity) far exceed the benefit (0.1% recall, within noise). At this scale and domain, simpler is better.

---

### Slide 3: Lesson 2 — "The Bottleneck Is Data Distribution, Not Architecture"

**The bridge** (connects Slide 2 to Slide 3): With architecture settled on exp2b, we investigated: why does performance plateau at recall@10 ~0.85 regardless of what we try?

#### 3.1 Why this matters: Rare codes are the most clinically valuable

Before showing what's broken, establish the stakes. Pre-training diagnostics on Commercial IP risk show that the codes our model fails to learn are the ones that matter most for downstream prediction:

| Code tier | Median Odds Ratio (IP risk) | Mean pre-training logit | Interpretation |
|-----------|----------------------------|------------------------|----------------|
| Common (top 20%) | 1.46 | -2.26 | Model learns these well — but they're weak predictors |
| Tail (bottom 40%) | **2.42** (65% higher, p<0.001) | **-14.69** (12.4-unit gap) | Strongest predictors — but model suppresses them |

The 12.4-unit logit gap means the model's output layer actively pushes tail code predictions toward zero. Meanwhile, embedding norms are healthy across all tiers (no encoder collapse in norm space) — the failure is specifically in discriminative feature learning, not in representation magnitude.

**Key insight**: We're not just failing on obscure codes — we're systematically failing on the codes with the highest clinical signal.

#### 3.2 The evidence — architecture-agnostic ceiling

| What we changed | tail_top10_acc |
|----------------|---------------|
| Dense 25M | 0% |
| MoE 35M | 0% |
| 512-dim 59M | 0% |
| BCE → Asymmetric Loss | 0% |
| pos_weight 35 → 200 | 0% |
| 1.5M → 11M members | 0% |

Nothing moves the needle. Not architecture, not loss function, not scale.

#### 3.3 Root cause — Emergent Gradient Starvation

Code frequency Gini = 0.939. Common codes = 69.7% of occurrences. Tail codes = 5.2%.

| Training step | Common gradient share | Tail gradient share |
|--------------|----------------------|-------------------|
| Step 1 | 17.8% | 17.8% |
| Step 3,000 | 66.7% | 3.0% |
| Step 12,000 | **85.3%** | **0.1%** |

The shared encoder becomes a common-code feature extractor. Tail embeddings homogenize (std=0.03 vs common 0.27).

**Scaling makes it worse, not better**: When we scaled from 1.5M→3.4M members, the per-code logit tier gap *widened* from 10.52→12.43. More data from the same Zipf distribution reinforces the gradient monopoly — the common codes get proportionally even more gradient.

#### 3.4 Why "just add more data" is structurally futile

The data information saturation study (target code analysis, R5) proves this is not a sample-size problem — it's a distributional invariant:

| Property | 100K members | 1M members | 10M members | Trend |
|----------|-------------|------------|-------------|-------|
| Shannon entropy | 7.831 bits | 7.833 bits | 7.834 bits | **Flat** — distribution shape is scale-invariant |
| Gini coefficient | 0.934 | 0.942 | 0.950 | **Increasing** — more data makes distribution MORE concentrated |
| Code vocabulary | ~5,800 | ~6,200 | ~6,297 | Saturates at ~200K members |

Marginal members (the next batch you'd add) are statistically indistinguishable from the core population — all distributional tests show p>0.4. Adding more members from the same claims source gives you more of the same Zipf distribution, not new information about tail codes.

**The distribution is a structural property of the clinical coding system, not an artifact of sample size.**

#### 3.5 Why self-attention can't compensate: Co-occurrence structure poverty

One might hope that attention could learn tail code meaning from context (co-occurring codes). The co-occurrence analysis shows why this fails:

| Co-occurrence metric | Value | Implication |
|---------------------|-------|-------------|
| Pair diversity involving ≥1 common code | **86%** | Common codes dominate relational structure |
| Tail-tail unique co-occurrence pairs | **21** (out of thousands possible) | Almost no relational signal between rare codes |
| Mean mutual information between code pairs | **0.005 bits** | Codes are near-conditionally-independent given the patient |
| Within-member code novelty at day 50 | **11%** (89% repeats) | Temporal context is mostly redundant |
| Within-member code novelty at day 100 | **7%** (93% repeats) | Extending the sequence window adds almost nothing |
| Skip-gram window extension gain | **<0.4%** new patterns | Wider co-occurrence windows don't help |

The attention mechanism has almost no tail-code relational structure to exploit. And within a single member's timeline, the signal saturates rapidly — 89% of codes seen by day 50 are repeats of earlier days. The 200-day temporal window mostly sees the same chronic condition codes cycling.

#### 3.6 Proving the bottleneck is the encoder, not the decoder (Round 9)

We tried Kang et al.'s approach from long-tail vision: freeze the encoder, retrain only the decoder on code-balanced batches. Result: decoder loss converged 20x better (0.44→0.026), gradient tail fraction reached 40.2%. But tail_top10_acc stayed at 0% and tail margin was -0.06 (noise). The decoder CAN learn — but the encoder representation `h` simply lacks discriminative features for tail codes. Unlike ImageNet pixels, the input-level distinctiveness for clinical codes must be learned, and gradient starvation prevents it.

#### 3.7 But data scaling DOES work for medium codes — proving the threshold mechanism

| Data | medium_top10_acc | tail_top10_acc |
|------|-----------------|---------------|
| 1.5M | 0.2% | 0% |
| 6.8M | 4.3% | 0% |
| **11M** | **20.0%** | 0% |

Medium codes crossed ~1,100 occurrences per code at 11M — enough for the gradient to learn. Tail codes at 11M have ~57 occurrences. The saturation analysis proves that reaching ~1,100 occurrences for tail codes through volume alone would require the equivalent of the entire US population, and the Zipf invariance means the distribution shape wouldn't change even then.

#### 3.8 Path forward — experimentally grounded directions

Each direction is now evaluated against the full diagnostic evidence:

**Tier 1: Tested / Partially validated**

| Approach | Mechanism | Evidence | Strength of case |
|----------|----------|---------|-----------------|
| **Pre-trained code embeddings (PPMI+SVD)** | Break input-level homogenization with co-occurrence structure | **Tested (R9)**: tail embedding std 0.03→0.077, first positive tail margin. Downstream hybrid Lift@1%=20.50 vs production 19.38 | **Strong** — directly addresses encoder bottleneck. But limited by co-occurrence structure poverty (86% common-dominated pairs, MI≈0.005). Captures distributional semantics, not deep clinical relationships |

**Tier 2: Untested but strongly motivated by diagnostics**

| Approach | Mechanism | Why diagnostics support it | Key risk |
|----------|----------|--------------------------|----------|
| **MLM-style masked code prediction** | Decouple gradient from code frequency — mask common codes, force encoder to reconstruct from context | Directly breaks the gradient starvation mechanism. Unlike next-code prediction, gradient allocation is controlled by masking schedule, not by natural frequency | Requires re-engineering the pre-training objective. Near-zero MI (0.005 bits) means masked reconstruction may be hard — codes carry little information about each other |
| **Per-tier decoder heads** | Separate decoders for common/medium/tail — eliminates cross-tier logit interference | Addresses the 12.4-unit logit gap directly. Tail decoder gets dedicated capacity without competing against common-code gradients | Doesn't fix the encoder representation; may hit the same ceiling as Kang et al. decoupled training |
| **Contrastive pre-training (patient-level)** | Learn embeddings that distinguish patients, not predict codes — exploits between-member diversity | Sidesteps the Zipf problem entirely. Even though code distributions are scale-invariant, *patient trajectories* are diverse. Shifts the learning signal from code frequency to patient identity | Requires negative sampling design; unclear how to handle the continuous spectrum of patient similarity |
| **Temporal data augmentation** | Random day dropping, code masking within days, temporal jittering | Within-member novelty is only 11% by day 50 (89% repeats). Augmentation can synthetically increase effective diversity by forcing the model to learn from partial/perturbed views | Doesn't add new information — only forces better utilization of existing signal |

**Tier 3: Hypothesis only — needs validation**

| Approach | Mechanism | Why it might help | Caveat |
|----------|----------|------------------|--------|
| **Per-code gradient normalization (GradNorm)** | Equalize gradient magnitude across frequency tiers | Directly attacks gradient concentration (85%→0.1% at step 12K) | Near-zero MI between codes (0.005 bits) means equalized gradients still lack relational context to learn from. May equalize *magnitude* without improving *signal quality* for tail codes |
| **Complementary data sources** (clinical notes, lab values, Rx fills) | Provide orthogonal signal that breaks claims-code Zipf ceiling | Code saturation is structural to the ICD/CPT system. External data operates on a different vocabulary/distribution | No experimental evidence. Data access and integration costs are high |
| **Multi-epoch training with curriculum** | Oversample tail-heavy members in later epochs | Could push tail codes past the ~1,100 occurrence threshold | Within-member temporal saturation (89% repeats by day 50) limits per-member information gain. Multi-epoch on same members yields diminishing returns |

#### 3.9 Synthesis: The three layers of the problem

The supplementary analyses reveal that the learning plateau is not one problem but three interlocking ones:

1. **Gradient starvation** (training dynamics): Common codes capture 85% of gradient by step 12K. This is the proximal cause. → Fix via masking objectives or gradient engineering.

2. **Information poverty** (data structure): Tail codes have near-zero MI with other codes (0.005 bits), only 21 tail-tail co-occurrence pairs, and 89% within-member temporal redundancy. Even with equalized gradients, the *signal* available for tail code learning is thin. → Fix via external data sources or representation pre-training that injects prior knowledge.

3. **Distributional invariance** (fundamental): The Zipf distribution is a structural property of the clinical coding system. Entropy is flat at 7.834 bits across 100x scaling. More data from the same source will never change the shape. → Fix requires changing the *type* of data or the *objective* (not volume).

No single intervention addresses all three layers. The most promising path combines: (a) PPMI+SVD pre-training to inject prior distributional knowledge (partially validated), (b) MLM-style masking to decouple gradients from frequency (addresses layer 1), and (c) contrastive learning to exploit patient-level diversity that exists despite code-level saturation (addresses layers 2+3).

**Slide 3 land-the-punch**: The architecture is sound — the bottleneck is three interlocking data distribution problems. Rare codes are clinically the most valuable (65% higher OR for IP risk), yet our model systematically suppresses them (12.4-unit logit gap). More data can't fix this — entropy is flat at 7.834 bits across 100x scaling, and Gini *increases* with scale. The co-occurrence structure is too sparse for attention to compensate (MI≈0.005 bits). But we have a partial breakthrough: PPMI+SVD pre-training moved tail embeddings from frozen to alive (std 0.03→0.077), and the medium code breakthrough at 11M proves the mechanism IS learnable — we just need to change HOW gradient reaches tail codes, not how much data we throw at them.

---

## Part 6: Narrative Coherence Check

**3-slide arc**:
1. **Slide 1** (Architecture): Here's what we built, why it's designed this way, and what it achieves. → **Establishes credibility**
2. **Slide 2** (MoE Lesson): We tried making it fancier — 12 experiments, fixed every root cause. Learned that at this scale, simpler is better. → **Demonstrates rigor** (we tested the alternative thoroughly, not just asserted it)
3. **Slide 3** (Learning Plateau): The real frontier isn't architecture — it's three interlocking data distribution problems (gradient starvation, information poverty, distributional invariance). Rare codes are clinically the most valuable, and we have the first evidence that representation pre-training can start to unlock them. → **Points forward with partial validation and quantified stakes**

**Bridge between Slide 2 and Slide 3**: "With architecture settled, we asked: why does performance plateau regardless of what we try? The answer isn't model capacity — it's gradient starvation from extreme code frequency imbalance. And the codes we're failing on turn out to be the ones that matter most clinically."

**What the audience should leave with**:
1. The architecture decisions are evidence-based, not trend-driven
2. We didn't just try one thing — we systematically tested and eliminated alternatives (12 MoE experiments, 14 hypotheses for the plateau)
3. The remaining bottleneck is understood mechanistically at three levels (gradient, information, distributional) and each level has a quantified diagnostic
4. The stakes are real: rare codes have 65% higher OR for IP risk — this isn't just a metric problem, it's a clinical value problem
5. We have a partially validated solution path (PPMI+SVD), a concrete next-step roadmap, and the medium code breakthrough proves the mechanism IS learnable
6. The model already delivers strong clinical value (medications 76%, 2.8pp gap to production AUC, 20% medium code breakthrough at 11M)

**Potential audience challenges and pre-loaded answers**:
- *"Why not just use a bigger model?"* → R7/R8: 512d adds +0.3pp for 2.3x parameters. Bottleneck is not capacity.
- *"Why not just retrain the decoder?"* → R9: decoder learned fine, tail stayed 0%. Bottleneck is the encoder.
- *"Why not just use more data?"* → Entropy is flat at 7.834 bits across 100x scaling (100K→10M members). Gini *increases* with scale. Marginal members are statistically identical to core (p>0.4). More data from the same source gives more of the same Zipf distribution, not new tail code information.
- *"Why not a different loss function?"* → R5.1: ASL improved calibration but did NOT change gradient distribution. Independent mechanisms.
- *"Is the MoE comparison fair?"* → R3 head-to-head: same data, same epochs, same config. R5 confirmed at scale. Same answer both times.
- *"Are the downstream numbers trustworthy?"* → Flag the OOT-strict vs non-strict distinction. Use consistent evaluation set when presenting.
- *"If codes are near-independent (MI≈0.005), how can attention help at all?"* → Attention helps for *common* codes where co-occurrence pairs are abundant (86% of pair diversity). It's specifically for *tail* codes that the relational structure is too sparse. This is why the problem is tier-specific, not model-wide.
- *"Why do you think masking will work if codes are independent?"* → MLM masking doesn't require codes to predict each other — it forces the *encoder* to build richer representations so the decoder can reconstruct. The gradient benefit (decoupling from frequency) is independent of inter-code MI.
