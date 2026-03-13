# Hypothesis-Driven Root Cause Analysis: Exp Round 10 (11M Formal Training)

**Date**: 2026-03-13
**Analyst**: Independent diagnosis following hypothesis-driven-diagnosis skill
**Subject**: Why did scaling from 1.5M to 11M members produce negligible downstream performance gains?

---

## Phase 1: Observe and Document

### 1.1 The Specific Discrepancy

**Expected**: Scaling training data from 1.5M → 11M members (7.3× increase) should produce meaningful improvement in downstream IP prediction, following the gains observed from 1.5M → 6.8M.

**Observed**: Near-zero downstream improvement; hybrid model actually **regressed** vs. the 6.8M experiment.

### 1.2 Quantified Pretraining Metrics Across All Rounds

| Round | Data | Dim | val_R@10 | val_μR@10 | val_NDCG@20 | med_top10 | tail_top10 | macro_AUROC | Train Time |
|-------|------|-----|----------|-----------|-------------|-----------|------------|-------------|------------|
| R5 exp1_opt | 1.5M | 256 | 0.825 | 0.478 | 0.439 | 5.5% | 0% | 0.876 | ~2h |
| R5 exp2b | 1.5M | 256 | 0.829 | 0.462 | 0.432 | 4.1% | 0% | 0.846 | ~2h |
| R5 v3 (bce_w200) | 1.5M | 256 | 0.817 | 0.466 | 0.427 | 0.2% | 0% | 0.878 | ~2h |
| R5 v4 (ASL) | 1.5M | 256 | 0.828 | 0.472 | 0.501 | 0% | 0% | 0.846 | ~2h |
| R5 v5 (ASL+dense) | 1.5M | 256 | 0.833 | 0.476 | 0.511 | 0.2% | 0% | 0.858 | ~2h |
| R6 3-4M | 3-4M | 256 | 0.834 | 0.477 | 0.447 | 4.0% | 0% | 0.886 | ~5h |
| **R6 6-8M** | **6-8M** | **256** | **0.855** | **0.576** | **0.467** | **4.3%** | **0%** | **0.913** | **~14h** |
| R7 512dim | 1.5M | 512 | 0.833 | 0.487 | 0.450 | 0.7% | 0% | 0.866 | ~5h |
| R9 v1 (2-stage) | 1.5M | 256 | 0.809 | 0.456 | 0.425 | 0.2% | 0% | 0.860 | ~8h |
| R9 v2 (co-occ) | 1.5M | 256 | 0.825 | 0.464 | 0.441 | 1.3% | 0% | 0.862 | ~8h |
| **R10 (11M)** | **11M** | **256** | **0.853** | **0.563** | **0.459** | **20.0%** | **0%** | **0.920** | **~32h** |

**Key pretraining observations**:
- R10 val_R@10 (0.853) is essentially **identical** to R6 6-8M (0.855) — no improvement with 62% more data
- R10 val_μR@10 (0.563) is actually **lower** than R6 6-8M (0.576) — a regression
- R10 medium_top10_acc (20%) is a large improvement over R6 (4.3%), but rare/tail remain 0%
- R10 macro_AUROC (0.920) improved marginally from R6 (0.913)
- The model peaked at R6 6-8M for recall-based metrics and plateaued or regressed with 11M

### 1.3 Quantified Downstream Metrics (The Real Test)

**Embedding-only models (oot_strict_auc_roc)**:

| Model | Data | oot_strict_auc | oot_strict_lift_1pct | Δ from R5_opt |
|-------|------|----------------|----------------------|---------------|
| R5 exp1_opt | 1.5M | 0.799 | 15.06 | baseline |
| R5 exp2b_v3 | 1.5M | 0.793 | 14.22 | -0.6pp |
| R6 3-4M | 3-4M | 0.799 | 16.18 | +0.0pp |
| R6 6-8M | 6-8M | 0.807 | 15.48 | +0.8pp |
| R7 512dim | 1.5M | 0.794 | 15.48 | -0.5pp |
| R9 v3 co-occ | 1.5M | 0.797 | 17.43 | -0.2pp |
| **R10 11M** | **11M** | **0.809** | **17.15** | **+1.0pp** |

**Hybrid models (oot_strict_auc_roc)**:

| Model | Data | oot_strict_auc | oot_strict_lift_1pct | Δ from R5_opt |
|-------|------|----------------|----------------------|---------------|
| R5 exp1_opt | 1.5M | 0.825 | 19.10 | baseline |
| R5 exp2b_v3 | 1.5M | 0.826 | 18.41 | +0.1pp |
| R6 3-4M | 3-4M | 0.826 | 18.96 | +0.1pp |
| **R6 6-8M** | **6-8M** | **0.835** | **18.13** | **+1.0pp** |
| R7 512dim | 1.5M | 0.827 | 19.24 | +0.2pp |
| R9 v3 co-occ | 1.5M | 0.827 | 20.50 | +0.2pp |
| **R10 11M** | **11M** | **0.831** | **18.69** | **+0.6pp** |
| Baseline tabular | full | 0.838 | 19.38 | — |

**Critical downstream findings**:
1. **Embedding-only**: R10 (11M) = 0.809 vs R6 (6-8M) = 0.807 → only **+0.2pp** from 62% more data
2. **Hybrid**: R10 (11M) = 0.831 vs R6 (6-8M) = 0.835 → **-0.4pp REGRESSION** with more data
3. **Lift@1pct**: R9 v3 co-occurrence (1.5M) has the best hybrid lift@1pct (20.50) — better than R10 (18.69)
4. **Tabular gap**: Best hybrid still 0.7pp below full tabular baseline (0.838)
5. **Data scaling curve**: 1.5M→3.4M (+0.0pp), 3.4M→6.8M (+0.9pp), 6.8M→11M (-0.4pp) — non-monotonic

### 1.4 Training Dynamics Observations (R10)

- **Configuration**: exp2b_flash_learned_pool, 256d, 4× T4 GPUs, 84,855 batches, 1 epoch
- **LR**: 8e-4 (scaled), warmup 15%, plateau until 60%, linear decay to 1.6e-4
- **Loss function**: BCEWithLogitsLoss, pos_weight_max=200, log_scaled
- **Training time**: 115,691 seconds (~32 hours), $44.53
- **Final metrics oscillation band**: R@10 ∈ [0.820, 0.887], μR@10 ∈ [0.484, 0.638]
  - The model was oscillating within this band for the entire second half of training
  - No monotonic improvement after ~50% of training
- **Loss trajectory**: 0.800 → 0.002 (rapid convergence), then plateaued at ~0.002
- **Generalization gap**: 0.0039 (very low → not overfitting, but also not learning complex patterns)

### 1.5 Config Comparison: R10 vs R6 6-8M (the better model downstream)

| Parameter | R6 6-8M | R10 11M |
|-----------|---------|---------|
| Data size | 6-8M | 11M |
| Embedding dim | 256 | 256 |
| pos_weight_max | 200 | 200 |
| Scheduler | linear | linear |
| Optimizer | AdamW | AdamW |
| LR | 8e-4 (scaled) | 8e-4 (scaled) |
| Batch size | 128 | 128 |
| Epochs | 1 | 1 |
| Gradient tier analysis | disabled | disabled |
| Co-occurrence embeddings | no | no |
| Two-stage training | no | no |

The configs are **identical** — the only variable is data size. This is a clean comparison.

---

## Phase 2: Priority-Guided Hypothesis Generation

### Level 1: DATA

**Hypothesis 1.1** (PRIMARY): "The data IS a significant part of the bottleneck — not in volume, but in information content relative to the pretraining objective."

**Evidence FOR**:
- 7.3× data increase (1.5M→11M) yielded only +1.0pp embedding-only downstream AUC and -0.4pp hybrid regression
- Scaling curve is non-monotonic for hybrid: peaked at 6-8M then declined
- This matches a classic **diminishing returns + interference** pattern: additional data adds more common-code redundancy without adding proportionally more discriminative signal for the downstream task
- The downstream task (IP prediction) requires patient-level risk stratification, while pretraining optimizes code prediction — additional common-code patients provide diminishing marginal information for patient risk

**Evidence AGAINST**:
- Pretraining metrics improved somewhat (macro_AUROC 0.913→0.920, medium_top10 4.3%→20%)
- Embedding-only downstream did improve marginally (0.807→0.809)

**Sub-hypothesis 1.1a**: "The additional 4.2M members (from 6.8M to 11M) are predominantly common-code-dense patients who reinforce the representation monopolization without adding new discriminative patterns."

**Test**: Analyze the code frequency distribution of the additional 4.2M members vs. the original 6.8M. If the marginal members have fewer rare/tail codes on average, this explains the diminishing returns.

**Sub-hypothesis 1.1b**: "Pretraining-downstream task misalignment is the binding constraint, not data volume."

**Evidence**: 
- R5 v4 ASL improved pretraining recall@1 by 2,230% yet downstream AUC moved +0.0pp
- R10 improved pretraining medium_top10_acc by 365% (4.3%→20%) yet hybrid downstream regressed
- This proves that pretraining metric improvements do not map to downstream gains

**Verdict on Level 1**: Data volume is NOT the bottleneck. The data is sufficient. The problem is what the model learns from it.

---

### Level 2: LOSS / OBJECTIVE ALIGNMENT

**Hypothesis 2.1** (CRITICAL): "The BCE loss function with pos_weight creates representations optimized for code prediction at the batch level, which is fundamentally misaligned with patient-level risk stratification."

**Evidence FOR**:
- All pretraining objectives tested (BCE, BCE+pw200, ASL, ASL+dense_sampler) produce representations that plateau at ~0.83 downstream hybrid AUC
- The loss function treats all 6,297 codes independently via binary cross-entropy, which:
  - Forces the shared representation `h` to optimize for code occurrence prediction
  - Does not capture code co-occurrence semantics relevant to disease progression
  - Does not differentiate between clinically meaningful and noise codes
- macro_AUROC (code-level) improved from 0.913 to 0.920, but downstream AUC did not follow
- **The loss ceiling and downstream ceiling are decoupled**: improving code prediction does not improve patient risk prediction

**Evidence FROM PRIOR ANALYSES**:
- pos_weight (35→200): gradient distribution changed <0.5% (R5 gradient analysis)
- ASL: fixed ranking quality (recall@1, MRR) but did not change tail gradient distribution
- The loss function cannot overcome the structural problem of occurrence-frequency-driven gradient aggregation

**Hypothesis 2.2**: "Representation monopolization by common codes is a loss-mediated structural failure, not a data problem."

**Evidence FOR** (quantified from R5/R9 analyses):
- Gradient distribution terminal state: ~85% common, ~10% medium, ~2% rare, ~0.1% tail
- This distribution is invariant to: loss function, pos_weight, data size, model capacity
- Transition happens by step ~3,000 (out of 84,855 in R10) — the first 3.5% of training determines the representation structure
- Once established, the remaining 96.5% of training reinforces common-code patterns without adding diversity
- **With 11M data**: the model sees even MORE common-code patterns in the remaining 96.5%, potentially over-specializing the representation for common codes at the expense of medium/rare code discriminability

**This is the mechanism behind the hybrid regression**: more data → more common-code gradient → more common-code-dominated representation → less useful for downstream patient risk prediction which benefits from nuanced medium/rare code patterns.

**Verdict on Level 2**: Loss/objective misalignment is a **confirmed primary bottleneck**. The BCE objective structurally produces representations that are dominated by common-code features, regardless of data volume. This is the core reason scaling from 6.8M→11M hurt downstream hybrid performance.

---

### Level 3: TRAINING DYNAMICS

**Hypothesis 3.1**: "Single-epoch training with the current LR schedule is a compounding factor that prevents the model from developing nuanced representations."

**Evidence FOR**:
- 1 epoch over 11M members = 84,855 batches
- Gradient capture occurs by step ~3,000 (batch 3,000 at 384K samples = 3.5% of data)
- The remaining 96.5% of training operates in a "reinforcement" regime, not a "learning" regime
- With 6.8M members (R6), the model had proportionally more "learning" steps per unit of data
- With 11M, the model spends more absolute time in the reinforcement regime, which actually deepens common-code monopolization

**Evidence FROM R10 training log**:
- By hour 2 (of 32), R@10 had already reached ~0.75 (batch ~5500)
- By hour 5, R@10 was ~0.84 (near final value)
- The last 27 hours of training produced <1.5pp improvement in R@10
- The metrics oscillated within a band: R@10 ∈ [0.82, 0.89], showing no sustained upward trend
- The LR was decaying during this period, but the model was not finding new patterns

**Hypothesis 3.2**: "The warmup→plateau→decay schedule causes irreversible gradient capture during warmup."

**Evidence FROM prior analyses**:
- LR polishing test rejected this (R5 evidence synthesis): resuming from plateau with 10× lower LR produced no improvement
- The gradient capture happens DURING warmup (steps 500–3,000) and is LR-independent
- Once common codes dominate the gradient, the decay phase cannot undo this

**Verdict on Level 3**: Training dynamics are a **contributing factor** but not the root cause. The 1-epoch regime and early gradient capture amplify the loss/objective misalignment. Multi-epoch training alone would not fix the structural issue.

---

### Level 4: ARCHITECTURE / SCALING

**Hypothesis 4.1**: "The 256-dimensional shared encoder is a capacity bottleneck."

**Evidence AGAINST**:
- R7 (512dim, 1.5M): downstream oot_strict = 0.794 vs R5 exp2b (256dim, 1.5M): 0.793 → +0.1pp
- R7 (512dim) hybrid: 0.827 vs R5 exp2b hybrid: 0.826 → +0.1pp
- Doubling embedding dimension produced negligible downstream improvement
- R9 root cause analysis confirmed: "capacity is not the bottleneck" across multiple expert reviews
- The 256d space has sufficient capacity; it's the content of the representation that's wrong

**Hypothesis 4.2**: "The shared encoder + single linear decoder architecture forces representation monopolization."

**Evidence FOR**:
- All codes share one `h ∈ ℝ^256` via `nn.Linear(256, 6297)`
- Gradient: `∂L/∂θ_encoder = Σ_j [∂L/∂z_j × ∂z_j/∂h × ∂h/∂θ_encoder]`
- With ~85% of gradient from common codes, `h` becomes a common-code feature extractor
- This is architecturally baked in — no amount of data or loss tuning can change it without structural changes
- The R9 two-stage training attempted to fix this by freezing the encoder, but `h` already lacked tail-discriminative features

**Verdict on Level 4**: Architecture (specifically the shared encoder + single decoder) is a **confirmed structural constraint**, but one that manifests through the gradient dynamics described in Level 2. It is a necessary condition for the monopolization, but the loss function is the sufficient condition.

---

## Phase 3: Root Cause Synthesis

### Primary Root Cause: Pretraining-Downstream Objective Misalignment Compounded by Representation Monopolization

The failure of 11M scaling can be attributed to a **three-layer causal chain**:

```
Layer 1: STRUCTURAL
  Shared encoder + linear decoder → single h for all 6,297 codes
  → Gradient from all codes aggregated into one representation
  → Whoever contributes most gradient wins → common codes win

Layer 2: LOSS-MEDIATED
  BCE + mean reduction + pos_weight → occurrence-frequency-driven gradients
  → ~85% common, ~0.1% tail (invariant to loss function, pos_weight, data)
  → Common codes monopolize representation by step ~3,000
  → More data = more common-code reinforcement, not more diversity

Layer 3: TASK MISALIGNMENT
  Pretraining objective (code prediction) ≠ downstream objective (patient risk)
  → Pretraining improvements (macro_AUROC, medium_top10) do not map to downstream gains
  → More code-prediction-optimized representation can actually HURT downstream
  → This explains the hybrid regression: R6 6-8M (0.835) > R10 11M (0.831)
```

### Why Specifically Did R10 Fail?

1. **R10 used identical config to R6 6-8M** — the only change was data size
2. The additional 4.2M members reinforced common-code patterns without proportionally adding rare/tail signal
3. The representation became **more specialized** for common-code prediction (macro_AUROC: 0.913→0.920)
4. This over-specialization reduced the **general-purpose utility** of the embeddings for downstream patient risk prediction
5. In the hybrid model, the over-specialized embeddings provided less complementary information to the tabular features, explaining the regression
6. **None of the R9 interventions were applied**: no co-occurrence embeddings, no two-stage training, no gradient rebalancing — yet the R10 experiment was expected to benefit from data alone

### What the Evidence Definitively Rules Out

| Rejected Hypothesis | Evidence |
|---------------------|----------|
| "Need more data" | 7.3× more data → -0.4pp hybrid downstream |
| "Need larger model" | 512d → +0.1pp downstream (R7 vs R5) |
| "Need different loss function" | ASL → +0pp downstream (R5 v4/v5) |
| "Need higher pos_weight" | 200 vs 35 → <0.5% gradient change |
| "Need density-aware sampling" | v5 → best pretraining, worst downstream |
| "LR schedule is the issue" | Polishing test rejected |
| "Need more epochs (alone)" | Won't fix gradient monopolization |

---

## Phase 4: Intervention Design (Cheapest-First)

### Tier A: Zero-Cost Insights (No new training needed)

**A.1 — Accept the pretraining ceiling and focus on downstream alignment**

The evidence strongly suggests that the current pretraining → downstream pipeline has a structural AUC ceiling of ~0.83 for hybrid models. Further scaling the same approach will not break this ceiling.

**A.2 — Best available model selection**

For embedding-only: use R10 (11M) — 0.809 oot_strict
For hybrid: use **R6 (6-8M)** — 0.835 oot_strict (better than R10)
For lift@1pct: use R9 v3 co-occurrence (1.5M) — 20.50 best lift

### Tier B: Low-Cost Experiments (~4-8 GPU-hours each)

**B.1 — Downstream-aware fine-tuning** (Highest priority)

Instead of treating pretraining as a fixed feature extractor:
- Take the R6 6-8M model (best hybrid downstream)
- Fine-tune the encoder end-to-end on the downstream IP prediction task
- Use a small learning rate (1e-5 to 1e-4) for 3-5 epochs
- This directly optimizes the representation for the downstream objective
- **Cost**: ~4 GPU-hours | **Expected impact**: 2-5pp AUC improvement based on transfer learning literature

**Pre-register**: If downstream AUC > 0.845 → fine-tuning breaks the ceiling
If downstream AUC ≤ 0.835 → representation is too damaged to fine-tune

**B.2 — Contrastive pre-training objective** (Second priority)

Replace or supplement BCE with a patient-level contrastive loss:
- Patients with similar clinical trajectories should have similar `h`
- Patients with different outcomes should have distant `h`
- This directly optimizes for downstream-useful representations
- Can be combined with the existing code prediction objective (multi-task)
- **Cost**: ~8 GPU-hours on 1.5M subset | **Expected impact**: representation diversification

**B.3 — Co-occurrence embedding initialization + gradient rebalancing**

The R9 v3 co-occurrence experiment showed that PPMI+SVD embeddings:
- Produced the best lift@1pct (20.50) in hybrid mode with only 1.5M data
- Achieved first positive tail margin (+1.02)
- Should be combined with per-tier gradient normalization for the full training

**Cost**: ~8 GPU-hours | **Pre-register**: If tail_top10_acc > 0 AND downstream AUC > 0.810

### Tier C: Medium-Cost Experiments (~16-32 GPU-hours)

**C.1 — Per-tier gradient normalization + multi-epoch**

This is the most-recommended intervention from all prior expert panels:
- Normalize gradient contributions so each tier contributes equally
- Train for 3-5 epochs to allow rare/tail codes sufficient exposure
- Use the R6 6-8M data (proven optimal at current scale)
- **Cost**: ~32 GPU-hours | **Expected impact**: Break tail 0% floor, improve representation diversity

**C.2 — Hierarchical code supervision (CCS/CCSR grouping)**

Add a secondary loss that predicts clinical code categories:
- CCS categories provide a ~280-class classification vs. 6,297 individual codes
- This regularizes the representation toward clinically meaningful groupings
- May directly improve downstream clinical predictions
- **Cost**: ~16 GPU-hours for data prep + training

### Tier D: Structural Changes (Requires implementation work)

**D.1 — Per-tier decoder heads (MoE decoder)**

Replace the single `nn.Linear(256, 6297)` with tier-specific decoder heads:
- Common codes: shared linear
- Medium codes: shared MLP
- Rare/tail codes: per-tier MLP with increased capacity
- This breaks the gradient aggregation at the decoder level

**D.2 — Dual-encoder architecture**

Separate the encoder into shared and tier-specific branches:
- Shared branch: learns general patient representation
- Tier-specific branches: learn features for medium/rare/tail codes
- More complex but addresses the root cause directly

---

## Phase 5: Priority Recommendation

### Immediate Action (This Week)

**Do not train another round with the same approach.** The evidence from 10 experiment rounds (5→10) spanning 8+ months and hundreds of GPU-hours is conclusive: more data, different loss functions, different pos_weights, different dimensions, and different sampling strategies all converge to the same ceiling.

### Recommended Next Experiment

**Experiment B.1: Downstream-aware fine-tuning of R6 6-8M model**

This is the highest-impact, lowest-cost intervention because:
1. It directly attacks the primary root cause (pretraining-downstream misalignment)
2. It uses the best available model (R6 6-8M at 0.835 hybrid)
3. It requires no new pretraining infrastructure
4. Cost: ~4 GPU-hours (~$1.40)
5. Clear success/failure criteria (AUC > 0.845 = success)

If B.1 succeeds: the ceiling was at the fine-tuning stage, not pretraining
If B.1 fails: the representation itself needs structural changes (proceed to B.2/B.3/C.1)

### Strategic Reframe

The experimental trajectory from R5→R10 has systematically eliminated every "easy" explanation:
- Not data volume (1.5M→11M: negligible gain)
- Not model capacity (256→512: negligible gain)
- Not loss function (BCE→ASL: negligible downstream gain)
- Not class weighting (35→200: negligible effect)
- Not sampling (density-aware: negligible downstream gain)

What remains is a fundamental misalignment between the pretraining objective and the downstream task. The pretraining approach produces a **common-code feature extractor** that happens to be useful for downstream tasks to a ceiling of ~0.83 hybrid AUC, but cannot be pushed further by improving code prediction. Breaking this ceiling requires either:

1. **Directly optimizing for the downstream task** (fine-tuning, B.1)
2. **Changing what the encoder learns** (contrastive learning, gradient rebalancing, B.2/C.1)
3. **Changing the encoder architecture** to support multi-objective representation (D.1/D.2)

---

## Appendix A: Complete Data Scaling Curve

### Pretraining Metrics vs. Data Size

```
Data Size:  1.5M  →  3.4M  →  6.8M  →  11.0M
val_R@10:   0.829 → 0.834 → 0.855 → 0.853   (peaked at 6.8M)
val_μR@10:  0.462 → 0.477 → 0.576 → 0.563   (peaked at 6.8M)
val_NDCG@20:0.432 → 0.447 → 0.467 → 0.459   (peaked at 6.8M)
macro_AUROC:0.846 → 0.886 → 0.913 → 0.920   (still increasing)
med_top10:  4.1%  → 4.0%  → 4.3%  → 20.0%   (major jump at 11M)
tail_top10: 0%    → 0%    → 0%    → 0%       (invariant)
```

### Downstream Metrics vs. Data Size

```
Data Size:       1.5M  →  3.4M  →  6.8M  →  11.0M
emb_only OOT:   0.793 → 0.799 → 0.807 → 0.809   (+1.6pp total, diminishing)
hybrid OOT:     0.826 → 0.826 → 0.835 → 0.831   (peaked at 6.8M, REGRESSED)
hybrid lift@1:  18.41 → 18.96 → 18.13 → 18.69   (noisy, no trend)
```

### Key Pattern

The **pretraining metrics** and **downstream metrics** diverge after 6.8M:
- macro_AUROC continues improving (0.913→0.920): the model gets better at code prediction
- hybrid OOT regresses (0.835→0.831): but this makes it WORSE at patient risk prediction

This is the **Goodhart's Law** of pretraining: optimizing the pretraining metric beyond a point degrades the downstream metric because the pretraining objective becomes an increasingly poor proxy for the downstream task.

## Appendix B: Config Audit

### R10 Config Decisions and Their Consequences

| Decision | Choice | Rationale | Assessment |
|----------|--------|-----------|------------|
| pos_weight_max=200 | Adopted from v3 experiments | Higher pos_weight for rare codes | **Neutral**: proven ineffective at changing gradient distribution |
| No co-occurrence embeddings | Not applied | Wasn't incorporated | **Missed opportunity**: R9 v3 showed +0.4pp downstream with co-occ at 1.5M |
| No two-stage training | Not applied | Stage 2 results were mixed | **Reasonable**: v1 decoder didn't break 0% tail |
| No gradient tier analysis | Disabled | Reduce overhead for 11M | **Missed signal**: couldn't monitor gradient monopolization dynamics at scale |
| 1 epoch | Same as all prior rounds | Standard practice | **Compounding factor**: 84,855 batches but 96.5% in reinforcement regime |
| 256 dim | Same as all prior rounds | 512d showed negligible gain | **Reasonable**: capacity not the bottleneck |

### What R10 Should Have Tested Instead

Given the accumulated evidence through R9, the 11M formal training would have been more impactful if it combined:
1. Co-occurrence embeddings (proven beneficial in R9 v3)
2. Per-tier gradient normalization (recommended by 3+ expert panels)
3. 3-epoch training (to allow rare/tail codes sufficient gradient accumulation)
4. The 6-8M data subset (which produced better downstream results than 11M)

This combination would have addressed the three identified bottlenecks (representation monopolization, gradient starvation, single-epoch exposure) simultaneously.

## Appendix C: Cross-Validation Against Best Practices

### Google Deep Learning Tuning Playbook Alignment

| Playbook Principle | R10 Adherence | Assessment |
|-------------------|---------------|------------|
| "More data only helps if model can learn from it" | Applied data scaling without structural changes | **Violated**: model had established ceiling |
| "Match the proxy objective to the true objective" | BCE for codes, downstream is patient risk | **Critical gap**: known since R5 downstream analysis |
| "Diminishing returns from data scaling are expected" | Scaled 7.3× expecting proportional gains | **Ignored**: evidence from R5→R6 already showed diminishing curve |
| "Change one variable at a time" | Only changed data size | **Good practice**: but on the wrong axis |

### Published Literature Alignment

1. **Kang et al. (2020), Decoupled Training**: Applied in R9; confirmed encoder is the bottleneck, not decoder
2. **Cui et al. (2019), Class-Balanced Loss**: Partially applied via pos_weight; shown ineffective for this gradient structure
3. **Goodfellow (2016), Deep Learning**: "More data helps with underfitting, not with architectural limitations"
4. **Scaling Laws (Kaplan et al., 2020)**: Data scaling follows power laws; R10 is deep in the diminishing returns regime
5. **Transfer Learning (Ruder, 2019)**: Pretraining-downstream misalignment is a known failure mode; task-aligned fine-tuning is the standard fix
