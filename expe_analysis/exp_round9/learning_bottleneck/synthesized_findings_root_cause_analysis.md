# Synthesized Root-Cause Analysis: Why the Loss Floor Is Invariant to Capacity and Data Scaling

**Date**: March 7, 2026
**Purpose**: Multi-expert synthesis for solution brainstorming — consolidates findings from 4 independent expert analyses, 8 experiments (v2–v5, R6–R8), gradient diagnostics, embedding/logit probes, and an LR polishing test. This document contains the confirmed facts, confirmed/rejected hypotheses, root cause mechanism, structural amplifiers, and the quantitative evidence base required to design effective interventions. This synthesized results are derived from @why_loss_reach_ceiling_regardless_dimen_training_increase.md.

---

## 1. The Question

Across all experiments — regardless of doubling model dimensions (256→512), tripling training data (1.5M→5.7M), changing loss functions (BCE→ASL), adjusting pos_weight (35→200), or applying density-aware sampling — the training loss converges to a fixed floor within each data tier, and tail/rare code accuracy remains at 0%.

**Why?** And what does the evidence tell us about where a solution must operate?

---

## 2. Complete Experimental Evidence Base

### 2.1 Experiment Matrix

| ID | Embedding | Params | Data | Loss | pos_weight | Sampling | Key Change |
|---|---|---|---|---|---|---|---|
| **v2** | 256d | 25.3M | 1.5M | BCE | pw=35 | Standard | Baseline (low pw) |
| **v3** | 256d | 25.3M | 1.5M | BCE | pw=200 | Standard | Baseline (high pw) |
| **v4** | 256d | 25.3M | 1.5M | ASL (γ-=4.0) | None | Standard | Loss function change |
| **v5** | 256d | 25.3M | 1.5M | ASL (γ-=4.0) | None | Density+tier (tail_quota=20) | Loss + sampling change |
| **R6** | 256d | 25.3M | 5.7M | BCE | pw=200 | Standard | Data scale 3.6× |
| **R7** | 512d | 58.6M | 1.5M | BCE | pw=200 | Standard | Dimension 2× |
| **R8** | 512d | 58.6M | 5.7M | BCE | pw=200 | Standard | Both dimension + data |
| **Polish** | 256d | 25.3M | — | BCE | pw=200 | — | Resume at 10× lower LR |

All experiments share: FlashAttentionTransformer, learned attention pooling, 8-head temporal encoder, linear schedule (15% warmup → 45% plateau → 40% decay), batch_size=128, AdamW, 1 epoch, multi-GPU (4× T4).

### 2.2 Loss Floor Comparison

| Run | Smoothed Train Loss | Val Loss | Generalization Gap |
|---|---|---|---|
| v2 (256d, 1.5M, pw=35) | 0.00319* | 0.00308 | 0.01000 |
| **v3 (256d, 1.5M, pw=200)** | **0.00319** | **0.00322** | **0.01015** |
| v4 (ASL, 1.5M) | 0.00166† | 0.000776† | 0.00088 |
| v5 (ASL+sampler, 1.5M) | 0.00166† | 0.000773† | 0.00089 |
| **R6 (256d, 5.7M)** | **0.00212** | **0.00205** | **0.00527** |
| **R7 (512d, 1.5M)** | **0.00318** | **0.00316** | **0.00764** |
| **R8 (512d, 5.7M)** | **0.00213** | **0.00204** | **0.00405** |

*v2 smoothed loss estimated from final results.
†v4/v5 use ASL loss which has different absolute scale; not directly comparable to BCE runs.

**Key observation**: Within each data tier under the same loss function, the floor is identical regardless of model capacity (V3 ≈ R7, R6 ≈ R8).

### 2.3 Validation Metrics Comparison

| Metric | v2 (pw=35) | v3 (pw=200) | v4 (ASL) | v5 (ASL+samp) | R6 (big data) | R7 (512d) | R8 (both) |
|---|---|---|---|---|---|---|---|
| recall@1 | — | 0.000 | **0.240** | **0.284** | 0.006 | 0.002 | 0.007 |
| recall@5 | 0.722 | 0.686 | 0.719 | 0.722 | 0.731 | 0.724 | 0.735 |
| recall@10 | 0.829 | 0.817 | 0.828 | 0.833 | 0.855 | 0.833 | 0.858 |
| recall@20 | 0.892 | 0.893 | 0.896 | 0.899 | 0.923 | 0.902 | 0.926 |
| micro_recall@10 | 0.462 | 0.466 | 0.472 | 0.476 | **0.576** | 0.487 | **0.578** |
| ndcg@10 | 0.398 | 0.390 | **0.468** | **0.478** | 0.440 | 0.415 | 0.442 |
| MRR | 0.341 | 0.324 | **0.471** | **0.496** | 0.335 | 0.345 | 0.337 |
| positive_brier | 0.679 | 0.687 | **0.313** | **0.308** | 0.639 | 0.661 | 0.638 |
| macro_auroc | 0.846 | 0.878 | 0.846 | 0.858 | **0.913** | 0.866 | **0.914** |
| macro_auprc | 0.103 | 0.105 | 0.110 | 0.103 | **0.161** | 0.135 | 0.135 |

### 2.4 Tier-Specific Code Accuracy

| Tier | v2 (pw=35) | v3 (pw=200) | v4 (ASL) | v5 (ASL+samp) | R6 (big data) | R7 (512d) | R8 (both) |
|---|---|---|---|---|---|---|---|
| common_top10_acc | 82.9% | 81.7% | 82.8% | 83.3% | **85.6%** | 83.3% | **85.9%** |
| medium_top10_acc | 4.11% | 0.16% | 0.00% | 0.17% | **4.26%** | 0.68% | 3.93% |
| rare_top10_acc | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| tail_top10_acc | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** |
| balanced_top10_acc | 21.8% | 20.5% | 20.7% | 20.9% | **22.5%** | 21.0% | 22.4% |

**Critical observation**: Tail code accuracy is **0% across ALL experiments**. No intervention tested — capacity, data, loss function, pos_weight, sampling — has moved this metric.

### 2.5 Gradient Tier Distribution (Terminal State)

| Run | common_frac | tail_frac | Source |
|---|---|---|---|
| v3 (BCE, pw=200) | 84.7% | 0.17% | Gradient tier tracking |
| v4 (ASL, no pw) | 85.7% | 0.12% | Gradient tier tracking |
| v5 (ASL+sampler) | 86.1% | 0.09% | Gradient tier tracking |

**Critical observation**: The gradient distribution is invariant to the loss function AND to density-aware sampling. This is the single most diagnostic finding.

### 2.6 Gradient Evolution Over Training (v3, pw=200)

| Training Phase | Steps | Common % | Medium % | Rare % | Tail % | Total Norm |
|---|---|---|---|---|---|---|
| Init | 1 | 17.8% | 27.3% | 26.5% | 17.8% | 530,569 |
| Early | 101 | 17.6% | 27.7% | 26.6% | 18.0% | 388,416 |
| Early | 501 | 16.9% | 27.9% | 27.0% | 18.4% | 24,989 |
| Transition | 1,501 | 42.7% | 21.9% | 17.4% | 10.4% | 3,398 |
| Mid | 3,001 | 66.7% | 16.1% | 7.1% | 3.0% | 1,632 |
| Late | 6,001 | 85.5% | 7.7% | 1.3% | 0.7% | 3,267 |
| Terminal | 12,001 | 85.3% | 11.2% | 0.6% | 0.1% | 22,129 |
| Epoch avg | — | 82.8% | 10.2% | 2.0% | 1.1% | 4,861 |

The gradient starvation transition completes by step ~3,000 and is irreversible under all tested conditions.

### 2.7 Embedding and Logit Diagnostics (v2, pw=35)

**Input Embedding Analysis:**

| Tier | Num Codes | Norm Mean | Norm Std | Interpretation |
|---|---|---|---|---|
| common | 1,169 | 1.42 | **0.27** | Rich, differentiated embeddings |
| medium | 1,754 | 1.49 | **0.15** | Moderately differentiated |
| rare | 1,748 | 1.41 | **0.05** | Weakly differentiated |
| tail | 1,175 | 1.46 | **0.03** | Homogenized — all tail codes look identical |

No embedding collapse (norms healthy at ~1.4), but severe **homogenization**: tail codes have converged to a near-identical embedding vector.

**Logit Analysis (when y=1, i.e., code actually present):**

| Tier | Mean Logit (y=1) | σ(logit) | Margin (pos-neg) | Interpretation |
|---|---|---|---|---|
| common | -2.26 | 9.4% | 6.44 | Strong learned signal |
| medium | -6.39 | 0.17% | 6.23 | Weak but present signal |
| rare | -9.68 | 0.006% | 5.34 | Very weak signal |
| tail | **-14.69** | **0.00004%** | **1.76** | Essentially no learned signal |

**Cross-code interference quantification**: For a tail code, the theoretical equilibrium decoder bias ≈ log(pw × freq) ≈ log(200 × 0.00001) ≈ **-6.2**. But the observed positive logit is **-14.69**, meaning `w_j^T h ≈ -8.5` — the common-code features in the shared representation `h` create systematic negative correlations with tail decoder rows, actively pushing tail logits 8.5 units below what the prior alone would produce.

### 2.8 LR Polishing Test Results

| Metric | Before Polish | After 2000 Steps (LR=4e-6) | Change |
|---|---|---|---|
| val_loss | 0.00336 | 0.00338 | +0.45% (WORSE) |
| recall@10 | 0.825 | 0.821 | -0.5% (WORSE) |
| recall@10 at step 200 | — | 0.788 | -4.4% (SHARP DROP) |
| NDCG@20 | 0.433 | 0.432 | -0.3% |
| rare_top10_acc | 0% | 0% | Unchanged |
| tail_top10_acc | 0% | 0% | Unchanged |
| Diagnosis | — | `STRUCTURAL_BOTTLENECK` | — |

The model at step 200 dropped 4.4% in recall, then slowly recovered to near-baseline — it is in a basin where perturbation causes regression followed by return to the same equilibrium. The polishing test **definitively rejects** the LR schedule as a contributing factor.

---

## 3. Confirmed Facts (Observable, Quotable)

These are observations directly from the data with zero interpretation:

1. **F1**: The smoothed loss floor is identical (within 0.5%) for 25.3M and 58.6M parameter models at each data tier.
2. **F2**: The loss floor drops ~35% when data scales from 1.5M to 5.7M, independent of model capacity.
3. **F3**: The model reaches the loss floor by step ~4,000 (out of ~12,335), spending ~65% of training oscillating at the floor.
4. **F4**: The gradient distribution (85% common, <1% tail) is identical under: pw=35, pw=200, ASL (γ_neg=4.0), and ASL+density sampling.
5. **F5**: Tail code accuracy is 0% across all 8 experiments tested.
6. **F6**: Tail input embeddings are homogenized (std=0.03 vs common std=0.27).
7. **F7**: Tail positive logits are ~8.5 units more negative than the theoretical bias equilibrium, indicating active cross-code suppression.
8. **F8**: LR polishing (10× lower LR for 2,000 steps from plateau) worsened val_loss and did not improve tail accuracy.
9. **F9**: ASL dramatically improves calibration (Brier 0.687→0.313) and ranking (MRR 0.324→0.496) without changing gradient distribution or tail accuracy.
10. **F10**: Data scaling produces the largest single-metric improvement: medium_top10_acc 0.16%→4.26% (27× increase from 1.5M→5.7M).

---

## 4. Hypotheses Tested and Their Status

### 4.1 Hypotheses REJECTED by Experimental Evidence

| Hypothesis | Experiment That Rejects It | Result |
|---|---|---|
| "The loss function (BCE mean reduction) determines the gradient distribution" | v4 (ASL replaces BCE) | Gradient distribution unchanged (85% common, 0.1% tail) |
| "pos_weight can overcome occurrence frequency imbalance" | v2 vs v3 (pw=35 vs pw=200) | Gradient distribution identical at both pos_weights |
| "The LR schedule prevents escape from the plateau" | Polishing test (10× lower LR) | val_loss worsened, tail accuracy unchanged |
| "Model capacity is the bottleneck" | R7, R8 (2.3× more params) | Loss floor identical within <0.5% |
| "Focal/ASL loss can break the tail code barrier" | v4, v5 | tail_top10_acc = 0% with ASL; gradient distribution unchanged |
| "Per-tier density-aware sampling fixes gradient starvation" | v5 (tier_tail_quota=20, density sampling) | gradient common_frac=86.1%, tail_frac=0.09% — WORSE than baseline |

### 4.2 Hypotheses CONFIRMED by Experimental Evidence

| Hypothesis | Evidence | Strength |
|---|---|---|
| "The gradient distribution is controlled by per-code occurrence frequency in training batches" | F4: gradient distribution invariant to loss function, pos_weight, and sampling | **Definitive** — 4 experiments confirm |
| "The shared encoder representation is monopolized by common-code gradients" | F6: embedding homogenization; F7: cross-code interference; gradient tracking | **Strong** — structural + empirical |
| "Data scaling helps by pushing medium codes above a gradient visibility threshold" | F10: medium_top10_acc 0.16%→4.26% at 3.6× data; recall@10 4.7% improvement | **Strong** — consistent across R6, R8 |
| "The model is at a structural bottleneck, not an optimization bottleneck" | F8: polishing test returned `STRUCTURAL_BOTTLENECK` | **Definitive** — direct experimental test |
| "The loss function controls calibration and ranking, not gradient distribution" | F9: ASL dramatically changes Brier/MRR while gradient distribution unchanged | **Definitive** — clean separation |

---

## 5. Root Cause: Occurrence-Frequency-Driven Per-Batch Gradient Aggregation

### 5.1 The Mechanism

The total gradient for the shared encoder parameters θ_enc at each training step is:

```
∂L/∂θ_enc = Σ_j Σ_{i ∈ batch} [∂L/∂z_ij × w_j^T × ∂h_i/∂θ_enc]
```

where:
- j indexes codes (1 to 6,297)
- i indexes patient-days in the batch (~6,400 = 128 members × ~50 valid days)
- z_ij = w_j^T h_i + b_j (the logit for code j on day i)

The number of terms where code j has y_ij = 1 (informative positive gradient) is:

| Code Tier | Frequency | Expected Positives per Batch | Per-element Gradient Amplifier | Effective Per-Batch Signal |
|---|---|---|---|---|
| Common | ~1% | ~64 | 1× (pw=1) | ~64 |
| Medium | ~0.1% | ~6.4 | ~50× (log-scaled pw) | ~320 |
| Rare | ~0.01% | ~0.64 | ~150× | ~96 |
| Tail | ~0.001% | ~0.064 | ~200× | ~12.8 |

Even with pos_weight=200, a tail code gets ~12.8 units of effective signal per batch, compared to ~64 for a common code. But critically, the tail signal is from **~0.064 samples** (i.e., most batches have ZERO tail-code-positive samples for any given tail code), while the common signal comes from **~64 consistently-present samples**.

**The gradient for a specific tail code is a near-zero-variance estimate (based on 0-1 observations) while the gradient for a specific common code is a high-precision estimate (based on ~64 observations).** AdamW's second-moment denominator further suppresses the sporadic tail gradient spikes relative to the consistent common gradient signal.

### 5.2 Why This Is the Root Cause (Not the Loss Function)

The v4/v5 experiments are the discriminating test. Two competing hypotheses:

- **Hypothesis A** (loss-function-driven): "The gradient distribution is determined by how the loss function weights easy negatives vs hard positives."
- **Hypothesis B** (occurrence-driven): "The gradient distribution is determined by per-code occurrence frequency in training batches."

Under BCE (v3), both hypotheses predict the same outcome. Under ASL (v4), they diverge:
- If A were correct: ASL (which zeroes out easy-negative gradients via p^4) should dramatically shift gradient toward positive-class signals, changing the tier distribution.
- If B were correct: ASL changes per-element weighting but not which codes appear in the batch. The distribution should be unchanged.

**Result**: Gradient distribution unchanged under ASL (85.7% common, 0.12% tail vs 84.7%/0.17% under BCE). **Hypothesis A is falsified. Hypothesis B is confirmed.**

### 5.3 Why Per-Tier Batch Enrichment Is Insufficient (v5 Evidence)

v5 uses `tier_tail_quota=20` (20 of 128 batch samples enriched with tail-code-containing patients) plus density-aware sampling. Despite this:

- gradient common_frac = 86.1% (HIGHER than v3's 84.7%)
- gradient tail_frac = 0.09% (LOWER than v3's 0.17%)
- tail_top10_acc = 0% (UNCHANGED)

**Why it fails**: 20 tail-enriched samples are shared across ~1,175 distinct tail codes. Each specific tail code appears in ~20/1,175 ≈ 0.017 samples per batch. Meanwhile, each specific common code appears in ~64/1,169 ≈ 0.055 samples per batch. The per-code ratio is still ~3:1 common:tail, far from balanced. Additionally, the tail-enriched samples ALSO contain common codes (patients have common diagnoses too), so the common-code gradient still dominates those 20 samples.

### 5.4 Causal Chain

```
Data occurrence frequency (structural, intrinsic)
    → Per-batch sample count per code (~64 for common, ~0.064 for tail)
        → Per-batch gradient contribution per code (~64× common, ~0.06× tail)
            → Encoder representation monopolization (h learns common features only)
                → Decoder failure for tail codes (w_j^T h is uninformative/actively suppressive)
                    → Loss floor = residual from unpredictable codes
```

At no point in this chain does the loss function, LR schedule, model capacity, or pos_weight magnitude play a determining role. They are all downstream of the entry point: data occurrence frequency.

---

## 6. Structural Amplifiers

While the root cause (Section 5) is sufficient to explain the loss floor invariance, three architectural and training choices amplify the severity and make recovery harder.

### 6.1 Amplifier A: Shared Encoder → Representation Monopolization

**Architecture**: The model produces a single representation `h ∈ ℝ^d` per patient-day via:

```
Input codes → Embedding(cd_cnt, d) → LearnedAttentionPooling → Temporal Transformer → h ∈ ℝ^d
```

Then ALL 6,297 codes read from this SAME `h` via `z_j = w_j^T h + b_j`.

**Impact**: Since 85% of the gradient that shapes the encoder comes from common codes, `h` becomes a common-code feature extractor. When dimension increases from 256 to 512, the additional 256 dimensions are populated by common-code features (because 85% of the gradient signal says "make these dimensions useful for common codes"). Tail codes gain 256 more decoder features to read, but those features carry no tail-relevant information.

**Evidence**:
- Embedding homogenization: tail std=0.03 vs common std=0.27 (F6)
- 512d ≈ 256d loss floor (F1)
- Cross-code interference: tail logits 8.5 units below equilibrium (F7)

**Relevance for solutions**: Any solution that leaves the shared encoder → single linear decoder architecture unchanged must change the gradient distribution (Section 5) dramatically enough to overcome the 85%/0.1% common/tail imbalance. Alternatively, architectural changes that decouple the representation (per-tier decoders, per-tier encoder branches) can address this amplifier directly.

### 6.2 Amplifier B: Input Embedding Feedback Loop

**Mechanism**: The input embedding layer `self.embedding_cd = nn.Embedding(cd_cnt, d)` is trained end-to-end. Tail code embeddings receive gradient:

```
∂L/∂e_j = Σ_{days containing code j} (∂L/∂input) × (∂input/∂e_j)
```

For tail codes, this sum has extremely few terms. Each term's upstream gradient `∂L/∂input` is itself dominated by common-code objectives (because the encoder is already monopolized). So tail code embeddings receive sparse, directionally-similar updates → they converge to nearly identical vectors → the encoder receives no distinctive input for tail codes → it cannot learn tail-specific features → tail embeddings stay homogenized.

**This is a self-reinforcing cycle at layer 0, upstream of all encoding:**

```
Homogenized tail embeddings → No distinctive encoder input → No tail-specific representation
    ↑                                                                  ↓
    ← Sparse, uniform gradient updates ← Dominated by common objectives ←
```

**Evidence**: Tail embedding std = 0.03, norm healthy at 1.46 (not collapsed, but homogenized) (F6).

**Relevance for solutions**: Even if the gradient distribution were perfectly balanced downstream, the input-level homogenization would persist because the embeddings have already converged. Solutions may need to either (a) provide pre-trained distinctive embeddings (from co-occurrence, medical ontology, etc.) or (b) decouple the embedding training from the main encoder training.

### 6.3 Amplifier C: Cross-Code Interference Through Shared Representation

**Mechanism**: The decoder computes `z_j = w_j^T h + b_j`. For a tail code, the theoretical optimal bias is:

```
b_j ≈ log(pw_j × freq_j / (1 - freq_j)) ≈ log(200 × 0.00001) ≈ -6.2
```

But the observed mean tail logit when y=1 is **-14.69**, which is ~8.5 units MORE negative. The excess comes from `w_j^T h`:

- `h` is dominated by common-code features
- The decoder row `w_j` for a tail code, trained with minimal/noisy gradient, develops negative correlations with the dominant features in `h`
- Result: `w_j^T h ≈ -8.5` for typical patient-days, pushing tail logits deeply negative beyond what the prior alone predicts

**Implication**: Adding more dimensions (512d) may make this WORSE, not better. More common-code features in `h` → more negative cross-products with poorly-trained tail decoder rows → stronger suppression of tail logits.

**Evidence**: Tail positive logit = -14.69 vs theoretical equilibrium -6.2 → excess suppression of -8.5 (F7). Tail margin (positive vs negative) = 1.76, compared to common margin = 6.44.

**Relevance for solutions**: Per-tier decoder heads would eliminate cross-code interference entirely — each tier's decoder would learn from features relevant to its own code set. A nonlinear (MLP) decoder per tier could also extract weak signals from `h` that a linear projection misses.

### 6.4 Amplifier D: Single-Epoch Rare Code Deprivation

**Mechanism**: All experiments train for exactly 1 epoch. With the linear schedule (15% warmup → 45% plateau → 40% decay):

- For 1.5M data: decay starts at step ~8,000, runs for ~4,336 steps
- For 5.7M data: decay starts at step ~26,700, runs for ~17,846 steps

A tail code with 0.001% frequency gets:
- 1.5M: ~1.58M × 0.00001 = ~16 total occurrences in the entire epoch, ~6 during the decay phase
- 5.7M: ~5.7M × 0.00001 = ~57 total occurrences, ~23 during the decay phase

But there are ~1,175 tail codes, so any SPECIFIC tail code gets:
- ~0.014 occurrences per batch during decay (1.5M) or ~0.05 occurrences per batch (5.7M)

The model effectively gets ZERO polishing-phase gradient updates for any specific tail code.

**Evidence**: The polishing test (additional 2,000 steps at 10× lower LR) showed zero improvement in rare/tail accuracy (F8), consistent with the model having no gradient signal to work with during polishing.

**Relevance for solutions**: Multi-epoch training alone would not solve the root cause (Section 5 — the gradient distribution would repeat identically each epoch). But multi-epoch training COMBINED with gradient distribution changes (Section 5 solutions) would give rare codes multiple passes through the polishing phase, allowing sparse gradient signals to accumulate coherently.

---

## 7. What the Loss Function DOES Control (v4/v5 Insight)

The v4/v5 experiments revealed an important nuance that affects solution design:

### 7.1 Loss Function Impact Domain

| What ASL Changed (vs BCE) | Magnitude | What ASL Did NOT Change |
|---|---|---|
| Calibration: positive_brier 0.687 → 0.313 | **-54%** | Gradient tier distribution: 85% common |
| Ranking: MRR 0.324 → 0.496 | **+53%** | tail_top10_acc: 0% |
| Top-1 precision: recall@1 0.000 → 0.284 | **+∞** | balanced_top10_acc: ~20.7% |
| BCE-measured val_loss: 0.00342 → 0.09356 | (different metric) | Embedding homogenization |

### 7.2 Interpretation for Solution Design

The loss function is **not irrelevant** — it significantly controls calibration and ranking quality. A future solution should:

1. **Not rely on loss function changes alone** to break the tail code barrier (proven insufficient by v4/v5)
2. **Preserve ASL/focal loss benefits** for calibration and ranking while adding orthogonal interventions for tail codes
3. **Treat loss function choice and gradient distribution as independent design dimensions** — they affect different outcome metrics

---

## 8. What Data Scaling DOES Control (R6/R8 Insight)

### 8.1 The Threshold-Crossing Effect

Data scaling from 1.5M to 5.7M produces its improvements through a threshold-crossing mechanism:

| Code Tier | Approx Frequency | Appearances at 1.5M | Appearances at 5.7M | Did It Cross Threshold? |
|---|---|---|---|---|
| Common | ~1% | ~15,000 | ~57,000 | Already above — refinement only |
| Medium | ~0.1% | ~1,500 | ~5,700 | **Yes** — medium_top10_acc 0.16%→4.26% |
| Rare | ~0.01% | ~150 | ~570 | **No** — rare_top10_acc still 0% |
| Tail | ~0.001% | ~15 | ~57 | **No** — tail_top10_acc still 0% |

The "threshold" is approximately the point where a code appears consistently enough per batch to maintain a coherent gradient signal across training steps. For medium codes, 3.6× more data pushes them above this threshold. For rare and tail codes, even 3.6× is insufficient.

### 8.2 Diminishing Returns Prediction

Extrapolating the threshold-crossing logic:
- To push rare codes above threshold: estimated ~10-50× more data (15M-75M samples)
- To push tail codes above threshold: estimated ~100-1000× more data (150M-1.5B samples)

These estimates assume the threshold is roughly "enough appearances per epoch to sustain gradient signal above noise." Since data at this scale may not be available, solutions should target the gradient distribution mechanism rather than data scaling.

### 8.3 R8 vs R6 Marginal Returns

| Metric | R6 (256d, big data) | R8 (512d, big data) | R8 Improvement Over R6 |
|---|---|---|---|
| recall@10 | 0.855 | 0.858 | +0.3% |
| micro_recall@10 | 0.576 | 0.578 | +0.4% |
| ndcg@10 | 0.440 | 0.442 | +0.5% |
| Cost (USD) | $17.28 | $19.68 | +14% cost for <0.5% gain |
| Peak memory (GB) | 12.31 | 17.51 | +42% memory |

512d adds virtually nothing on top of data scaling. The additional capacity is monopolized by common codes (Amplifier A) and provides no benefit for the underserved tiers.

---

## 9. Architecture: Relevant Code Paths for Solution Design

### 9.1 Model Forward Pass (FlashAttentionTransformer)

```
1. Extract: age, gender, lob, codes from input [batch, len_dy, 82]
2. Embed:   self.embedding_cd(codes) → [batch, len_dy, len_cd, d]
3. Residual: cd_res = codes.sum(-2) → [batch, len_dy, d]
4. Daily:   self.daily_pooling(codes) → [batch*len_dy, d]  (LearnedAttentionPooling)
5. Combine: cd_res + daily + gender + age + lob → [batch, len_dy, d]
6. GELU + LayerNorm
7. Temporal: N layers of {PreNorm → FlashAttention(causal) → PreNorm → FFN}
8. Final:   LayerNorm → Dropout → self.decoder_cd(h) → [batch, len_dy, 6297]
```

**Key intervention points**:
- Step 2: `self.embedding_cd` — where input embedding homogenization occurs
- Step 4: `daily_pooling` — where codes aggregate into a single vector per day
- Step 7: Temporal encoder — where representation monopolization occurs
- Step 8: `self.decoder_cd` — single `nn.Linear(d, 6297)`, where cross-code interference occurs

### 9.2 Loss Computation (DataParallelWrapper.forward)

```python
# valid_output: [N_valid_days, 6297]  — all codes from all valid days
# valid_targets: [N_valid_days, 6297]  — multi-hot targets
pred_loss = self.criterion(valid_output, valid_targets)  # reduction='mean' over ALL elements
```

The loss averages over `N_valid_days × 6,297` elements. For a batch of 128 members with ~50 valid days each, this is ~6,400 × 6,297 ≈ 40.3 million elements, of which ~0.2% are positive.

**Key intervention point**: `self.criterion` and its `reduction` parameter — this is where per-tier loss decomposition would operate.

### 9.3 Criterion Factory

```python
# Three loss types available:
if optimize_config.use_asl:
    criterion = AsymmetricLoss(gamma_pos, gamma_neg, clip, pos_weight, reduction='mean')
elif optimize_config.use_focal_loss:
    criterion = FocalLoss(gamma, alpha, pos_weight, reduction='mean')
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # default, reduction='mean'
```

All three use `reduction='mean'`. The framework already supports ASL and FocalLoss, demonstrating the infrastructure can accommodate loss function changes.

### 9.4 Code Tier Definitions (from gradient tier analysis)

The codebase already contains tier classification infrastructure:

| Tier | Definition | Approx Count | Frequency Range |
|---|---|---|---|
| Common | Top 25% by frequency | ~1,169 codes | >0.1% |
| Medium | 25th-50th percentile | ~1,754 codes | 0.01%-0.1% |
| Rare | 50th-75th percentile | ~1,748 codes | 0.001%-0.01% |
| Tail | Bottom 25% | ~1,175 codes | <0.001% |

The gradient tier tracking infrastructure (`enable_gradient_tier_analysis`) already computes per-tier gradient norms at each step. This infrastructure can be leveraged for per-tier loss decomposition.

---

## 10. Quantitative Constraints for Solution Design

### 10.1 Infrastructure Constraints

| Resource | Current Usage | Headroom |
|---|---|---|
| GPU type | T4 (4 GPUs) | 16 GB per GPU |
| Peak memory (256d) | 12.8 GB | ~3.2 GB spare per GPU |
| Peak memory (512d) | 17.8 GB | ~-1.8 GB (over budget on single GPU, saved by distribution) |
| Training cost (1.5M, 256d) | ~$5 per epoch | Budget-dependent |
| Training cost (5.7M, 256d) | ~$17 per epoch | Budget-dependent |
| Throughput (256d) | ~620 samples/sec | — |
| Throughput (512d) | ~459 samples/sec | ~26% slower |

### 10.2 Quantitative Targets for Breaking Through

Based on the evidence, any effective solution must change these measured quantities:

| Metric | Current (Best) | Target for Breakthrough | Measurement Method |
|---|---|---|---|
| gradient tail_frac | 0.09-0.17% | >5% (at minimum) | Gradient tier tracking |
| tail_top10_acc | 0% | >0% (any movement) | Tier-stratified evaluation |
| rare_top10_acc | 0% | >0% (any movement) | Tier-stratified evaluation |
| tail embedding std | 0.03 | >0.10 (approaching rare=0.05) | Embedding analysis |
| tail positive logit (y=1) | -14.69 | >-10.0 (within 4 units of equilibrium) | Logit analysis |
| balanced_top10_acc | 22.5% | >25% (meaningful improvement) | Tier-stratified evaluation |

### 10.3 What "Success" Looks Like at Each Tier

| Tier | Current Best | Minimum Viable Improvement | Metric |
|---|---|---|---|
| Common | 85.9% (R8) | Maintain ≥84% | common_top10_acc |
| Medium | 4.26% (R6) | Improve to >10% | medium_top10_acc |
| Rare | 0% (all) | Any positive value | rare_top10_acc |
| Tail | 0% (all) | Any positive value | tail_top10_acc |

The primary goal is to break the 0% barrier for rare and tail codes WITHOUT degrading common code performance.

---

## 11. Expert Consensus and Disagreements — Summary for Solution Design

### 11.1 All Experts Agree On

1. The shared encoder → single linear decoder architecture is the primary bottleneck
2. Gradient starvation (85% common, <1% tail) is confirmed and drives the floor
3. The LR schedule is NOT a factor (polishing test definitive)
4. Per-tier loss decomposition is a high-priority intervention
5. Separate decoder heads per tier is architecturally sound
6. Multi-epoch training could help but is insufficient alone

### 11.2 The v4/v5 Evidence Resolved Key Disagreements

| Disagreement | Before v4/v5 | After v4/v5 (Resolved) |
|---|---|---|
| Is the loss function the root cause? | Expert 1: Yes (Ceiling 1). Expert 2: No (amplifier). Expert 3: Middle ground. | **Definitively No** — ASL changes loss landscape without changing gradient distribution |
| Should focal/ASL be recommended to break tail ceiling? | All 3 experts recommend it | **No** — v4/v5 proved it insufficient for gradient distribution/tail accuracy |
| Is the root cause loss-function-driven or data-distribution-driven? | Ambiguous — both hypotheses predict same outcome under BCE | **Data-distribution-driven** — ASL falsifies loss-function hypothesis |
| Is per-tier batch enrichment sufficient? | Expert 3 recommends density sampling | **No** — v5 proved tier_tail_quota=20 insufficient (gradient distribution worsened) |
| Where must solutions operate? | Debate between loss function vs architecture vs sampling | **Batch composition (per-code level) + architecture (decoder decoupling)** |

### 11.3 Remaining Open Questions for Solution Design

1. **What per-code sampling rate equalizes gradient contribution?** The v5 evidence shows tier-level sampling is insufficient. How many tail-code-containing samples per batch are needed to achieve gradient parity for a specific tail code?

2. **Can per-tier loss decomposition overcome the root cause without sampling changes?** Per-tier loss (computing loss separately per tier and weighting tiers equally) changes `∂L/∂θ_enc` to have equal tier contributions. This addresses the gradient distribution at the loss level. Would this be sufficient, or does the batch composition issue persist even with per-tier loss?

3. **Does multi-head decoder architecture require multi-epoch training to show benefit?** If per-tier decoders are introduced, each tier's decoder sees the same `h` but has its own parameters. Would 1 epoch be sufficient for the tail decoder to learn from the sparse signal, or is multi-epoch essential?

4. **Can pre-trained code embeddings break the input-level homogenization independently?** If tail code embeddings were initialized from medical ontology or co-occurrence matrices, would this provide enough distinctive input signal to give the encoder a chance to learn tail-specific features?

5. **What is the cost of architectural changes?** Per-tier decoders, per-code sampling, multi-epoch training — what are the memory, compute, and engineering costs of each, and which are compatible with the T4 × 4 infrastructure?

---

## 12. Evidence Inventory for Solution Brainstorming

This section catalogs all evidence artifacts available for reference during solution design:

| Artifact | Location | What It Contains |
|---|---|---|
| Model code | `dev/moe/moe_flashattn_4.py` | Full architecture, loss functions, training loop |
| V3 config | `expe_logs/exp_round5_1_lr_plateau/exp2/v3_bce_weighed200_config.json` | BCE baseline config |
| V3 results | `expe_logs/exp_round5_1_lr_plateau/exp2/v3_bce_weighed200_final_results.json` | BCE baseline results |
| V4 config (ASL) | `expe_logs/exp_round5_1_lr_plateau/exp2/v4_asymm_focalloss_config.json` | ASL experiment config |
| V4 results | `expe_logs/exp_round5_1_lr_plateau/exp2/v4_asymm_focalloss_final_results.json` | ASL experiment results |
| V5 config (ASL+sampler) | `expe_logs/exp_round5_1_lr_plateau/exp2/v5_asymm_focalloss_dense_sampler_config.json` | ASL + sampling config |
| V5 results | `expe_logs/exp_round5_1_lr_plateau/exp2/v5_asymm_focalloss_dense_sampler_final_results.json` | ASL + sampling results |
| R6 results (big data) | `expe_logs/exp_round6/training_6-8M/final_results.json` | Data scaling results |
| R7 results (512d) | `expe_logs/exp_round7_512dim/exp2/final_results.json` | Dimension scaling results |
| R8 results (both) | `expe_logs/exp_round8/exp2b_512dim_6-8M/final_results.json` | Combined scaling results |
| Gradient tracking | `expe_logs/exp_round5_1_lr_plateau/exp2/v3_bce_weighed200_batch_metrics.json` | Step-by-step gradient tier data |
| Embedding/logit probe | `expe_logs/exp_round5_1_lr_plateau/exp2/exp_round5_exp2_lr_plateau_embedding_logit_check_jan_25.json` | Per-tier embedding and logit analysis |
| Polishing test | `expe_logs/exp_round5_1_lr_plateau/exp2/exp2b_flash_learned_pool_v2_gradient_polishing_test_results.json` | LR polishing test results |
| Expert analyses | `expe_analysis/exp_round5/learning_plateau/` | 17 analysis documents |
| Prior expert document | `docs/pss/learning_bottleneck/why_loss_reach_ceiling_regardless_dimen_training_increase.md` | 4 expert analyses with cross-review |
