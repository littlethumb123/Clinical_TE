# Comprehensive Comparative Analysis: v0 (Original) vs v1 (Fixed Stage 2)

## 1. What Changed Between v0 and v1

Let me first verify exactly what fixes were applied, by comparing the Stage 2 configs:

| Parameter | v0 (Original) | v1 (Fixed) | Magnitude of Change |
|-----------|---------------|------------|---------------------|
| **Learning rate** | 5e-5 | **0.005** | **100x higher** |
| **Epochs** | 3 (3,204 steps) | **20** (21,360 steps) | **6.7x more** |
| **Total steps** | 3,204 | **21,360** | **6.7x more** |
| **Warmup** | 320 steps (10%) | **1,068 steps (5%)** | Longer warmup, fills epoch 1 |
| **Focused loss** | No (`use_focused_loss`: absent) | **Yes** (`use_focused_loss: true`) | Fundamental change |
| **Warmup fraction** | 0.1 | **0.05** | Minor |

All three critical fixes from my prior recommendation were implemented: higher LR, more epochs, and focused loss. This is a properly controlled re-run.

---

## 2. Stage 1 Comparison: Confirming Reproducibility

Both v0 and v1 ran identical Stage 1 training. Let me verify they produced comparable baselines:

| Metric | v0 Stage 1 | v1 Stage 1 | Delta |
|--------|-----------|-----------|-------|
| Val loss | 0.0031 | 0.0031 | Identical |
| Recall@10 | 0.813 | 0.809 | -0.5% (noise) |
| μRecall@10 | 0.457 | 0.456 | -0.2% |
| NDCG@20 | 0.425 | 0.425 | Identical |
| Gradient Common epoch avg | 74.0% | 72.8% | -1.2% |
| Gradient Tail epoch avg | 3.7% | 4.0% | +0.3% |

Stage 1 is reproducible. Small variations are from random seed differences. Both experiments start from equivalent encoder representations.

### Pre-Stage 2 Logit Diagnostics (Baseline)

| Tier | v0 Pre-S2 logit | v1 Pre-S2 logit | v0 Pre-S2 margin | v1 Pre-S2 margin |
|------|----------------|----------------|-----------------|-----------------|
| Common | -2.41 | -2.41 | 6.72 | 6.60 |
| Medium | -7.33 | -7.24 | 5.09 | 4.98 |
| Rare | -11.62 | -11.32 | 2.97 | 2.89 |
| Tail | -13.49 | -13.26 | 2.01 | 1.66 |

Minor differences from different random seeds, but structurally identical starting points.

---

## 3. Stage 2: The Critical Comparison — Loss Trajectory

This is where the story diverges dramatically.

### v0 (broken): Loss was essentially flat

```
Epoch 1: 0.5148
Epoch 2: 0.5108
Epoch 3: 0.5092
Total reduction: 1.1%
```

### v1 (fixed): Loss dropped aggressively and continuously

```
Epoch  1: 0.4381    (-14.9% from start)
Epoch  2: 0.2254    (-56.2%)
Epoch  3: 0.1234    (-76.0%)
Epoch  4: 0.0829    (-83.9%)
Epoch  5: 0.0631    (-87.7%)
Epoch 10: 0.0329    (-93.6%)
Epoch 15: 0.0270    (-94.7%)
Epoch 20: 0.0262    (-94.9%)
```

**This is the single most important difference.** The focused loss + higher LR produced a **20x loss reduction** (0.5135 → 0.0262), compared to the v0's essentially stationary 1.1% wobble. The decoder was unambiguously learning in v1.

The loss curve shows two distinct phases:
- **Rapid learning (epochs 1-8):** Loss dropped from 0.44 to 0.039 — a 10x reduction. Batch-level loss within epoch 1 went from 0.5135 at batch 0 to 0.3352 at batch 1000 — the decoder was actively learning even within the first epoch.
- **Convergence plateau (epochs 9-20):** Loss decayed slowly from 0.035 to 0.026, with epochs 17-20 essentially flat (0.0263 → 0.0262).

This confirms that the v0 experiment was indeed catastrophically under-optimized — the decoder had massive untapped learning capacity that the low LR and few epochs never accessed.

---

## 4. Stage 2: Post-Training Diagnostics — The Verdict

### Post-Stage 2 Logit Diagnostics

| Tier | v0 Post-S2 logit | v1 Post-S2 logit | v0 Post-S2 margin | v1 Post-S2 margin |
|------|-----------------|-----------------|-------------------|-------------------|
| **Common** | -2.41 | **-2.29** | 6.73 | **6.28** |
| **Medium** | -7.33 | **-6.86** | 5.10 | **4.72** |
| **Rare** | +0.25 | **-3.45** | 0.32 | **0.45** |
| **Tail** | -0.30 | **-3.93** | -0.28 | **-0.06** |

Let me decompose this tier by tier:

### Common/Medium (Should Be Unchanged)

| Metric | v0 Pre→Post | v1 Pre→Post |
|--------|-------------|-------------|
| Common margin | 6.72 → 6.73 (+0.01) | 6.60 → 6.28 (**-0.32**) |
| Medium margin | 5.09 → 5.10 (+0.01) | 4.98 → 4.72 (**-0.26**) |

In v0, common/medium were perfectly preserved because the decoder barely moved. In v1, common margin dropped by 0.32 and medium by 0.26. This is a **small but real degradation** — the aggressive Stage 2 training is slightly disturbing the common/medium decoder rows. Since the encoder is frozen, this means the decoder weights for common/medium codes shifted slightly. This could be from: (a) the common/medium rows being trainable during Stage 2 (they weren't re-initialized but are still in the optimizer), or (b) indirect effects from weight sharing or norm interactions.

This is a **yellow flag** — not catastrophic, but indicates that Stage 2 with focused loss is still touching common/medium decoder rows when it shouldn't need to.

### Rare Codes: Genuine Improvement

| Metric | Pre-S2 | v0 Post-S2 | v1 Post-S2 |
|--------|--------|-----------|-----------|
| pos_logit | -11.32 | +0.25 | **-3.45** |
| neg_logit | -14.21 | ~-0.07 (inferred) | **-3.90** |
| margin | 2.89 | 0.32 | **0.45** |
| pos_logit_std | 1.93 | (not available) | **1.50** |

This is critically important. In v0, the rare positive logit went from -11.62 to +0.25 — a huge shift but the margin collapsed from 2.97 to 0.32. In v1, the rare positive logit went from -11.32 to -3.45 and **the margin slightly improved from 2.89 to 0.45** when comparing v0-post to v1-post... but actually **decreased from the pre-S2 baseline of 2.89**.

Wait — let me be precise here. Comparing against the v1 pre-S2 baseline:

| | v1 Pre-S2 | v1 Post-S2 | Change |
|--|-----------|-----------|--------|
| Rare margin | 2.89 | 0.45 | **-2.44** (degraded) |
| Tail margin | 1.66 | -0.06 | **-1.72** (degraded) |

The margins still collapsed, just not as catastrophically as v0. The rare margin went from 2.89 to 0.45 (v1) vs 2.97 to 0.32 (v0). The v1 result is *slightly* better (0.45 vs 0.32) but the margin still collapsed by 84%.

### Tail Codes: The Bottom Line

| Metric | v0 Post-S2 | v1 Post-S2 | v1 Better? |
|--------|-----------|-----------|-----------|
| pos_logit | -0.30 | **-3.93** | Different, not obviously better |
| margin | -0.28 | **-0.06** | Slightly less negative |
| tail_top10_acc | 0% | **0%** | **No change** |

The tail margin went from -0.28 (v0) to -0.06 (v1). Still negative, still no discrimination. The tail_top10_acc confirmed at **0%** in the final_results.json.

---

## 5. What This Result Really Means

### 5a. To the Model

The v1 experiment proves something v0 could not: **the decoder CAN be trained aggressively on balanced data (loss dropped 20x), but the resulting weights still cannot discriminate positive from negative tail cases.** This is a qualitatively different and much stronger conclusion than what v0 showed.

In v0, the flat loss (0.51 → 0.51) could be interpreted as "the decoder never got a chance to learn." In v1, the loss plummeted from 0.44 to 0.026 across 20 epochs — **the decoder clearly learned something.** But what it learned was to fit the *training distribution of the focused loss* (correctly predicting target codes on the enriched batches) without developing generalizable discrimination on the validation set.

The model is memorizing the co-occurrence patterns between target codes and the specific patient-days in the balanced batches, but this doesn't transfer to the general population. This is consistent with `h` encoding patient features that are *confounded* with common codes — when you see a patient with a tail code, their `h` looks virtually identical to patients without that tail code because `h` is dominated by the common-code features they share.

### 5b. To the Method (Two-Stage Decoupled Training)

**The Kang et al. decoupled training approach has now been given a fair test and has failed for this specific problem.**

Let me be precise about what distinguishes this from the ImageNet-LT setting where it works:

1. **In vision (ImageNet-LT):** The input (pixel image) contains full information about the object class, regardless of training distribution. A photo of a rare bird species has distinctive visual features that the encoder captures because they're physically present in the input. The decoder just needs to learn the mapping from features to class.

2. **In this clinical transformer:** The input embeddings for tail codes are homogenized (std=0.03). When a patient has a rare tail code, the only thing distinguishing their input from a similar patient without that code is the embedding of the tail code itself — which is nearly identical to other tail code embeddings. The encoder, shaped by 85% common-code gradient, has no incentive to amplify this faint input difference. So `h` for a "positive tail code patient" is nearly identical to `h` for a "negative tail code patient" — there is genuinely no signal for the decoder to extract.

The v1 result — with 100x higher LR, 7x more epochs, and focused loss — makes this distinction sharp. **The decoder was given every advantage and still failed.** We can now state with high confidence: **the bottleneck is at the encoder representation level, not the decoder level.**

### 5c. To the Resolution of the Bottleneck

This experiment resolves the central ambiguity from the v0 analysis. The three experts disagreed on whether `h` contains tail signal:

| Expert | v0 Conclusion | v1 Validation |
|--------|--------------|---------------|
| Expert 1 | "h lacks signal — bottleneck is encoder" | **Confirmed** |
| Expert 2 | "Premature — Stage 2 was under-optimized" | **Refuted** — adequate optimization still fails |
| Expert 3 (me) | "Inconclusive due to loss dilution confound" | **The confound has been removed. h genuinely lacks signal.** |

**Expert 1 was right all along.** Expert 2's concern about under-optimization was valid *procedurally* but wrong *substantively* — fixing the optimization didn't change the outcome. My prior analysis correctly identified the loss dilution as a confound that needed removal, but I overestimated the likelihood that removing it would reveal hidden signal. The signal isn't there.

---

## 6. Is Step 1 Working? Why or Why Not?

### What Worked

1. **The focused loss fix was exactly right.** The loss dropped 20x vs v0's 1.1%. The decoder was genuinely learning.
2. **The higher LR was necessary.** The warmup from 5e-4 to 5e-3 and the sustained peak LR allowed meaningful weight updates. The gradient tier tracking shows common fraction rising from 0.1% to 1.0-1.5% over epochs, indicating the decoder weights were changing enough that well-calibrated common rows started developing larger gradients (their predictions were getting slightly worse as the decoder shifted).
3. **20 epochs were sufficient.** The loss converged by epoch ~15 and was flat for epochs 17-20.

### What Did NOT Work

**The tail margin is still negative (-0.06).** Despite:
- 21,360 optimization steps (vs 3,204 in v0)
- Peak LR of 5e-3 (vs 5e-5 in v0)
- Focused loss computing only on target codes
- Loss reaching 0.026 (a well-converged state)

The decoder converged to a state where tail positive and negative logits are nearly identical (-3.93 vs -3.87). The margin of -0.06 is statistically indistinguishable from zero given the diagnostic samples only 44 tail positive instances (flagged as `tail_low_sample_warning: true`).

### Why It Didn't Work: The Definitive Diagnosis

The v1 result combined with the v0 result creates a controlled experiment that isolates the variable:

| Variable | v0 | v1 | Outcome |
|----------|---|---|---------|
| Loss scope | Global (6,297 codes) | Focused (16 target codes) | Loss dropped 20x more in v1 |
| LR | 5e-5 | 5e-3 | Decoder weights changed substantially |
| Epochs | 3 | 20 | Convergence achieved |
| **Tail margin** | **-0.28** | **-0.06** | **Both negative. No discrimination.** |

The margin went from -0.28 to -0.06 — a modest improvement that merely brought it from "anti-discriminative" to "noise-level zero discrimination." The fact that three independent fixes (each addressing a real confound) collectively produced only a 0.22-unit improvement in margin — while the loss itself dropped 20x — is the strongest possible evidence that **the information is not in `h`**.

This is the clinical version of the "polishing test" from the root cause analysis. Just as the LR polishing test proved the model was at a structural (not optimization) bottleneck in Stage 1, the v1 Stage 2 experiment proves the decoder is at a structural (not optimization) bottleneck: the representation `h` genuinely lacks features that discriminate tail code presence/absence.

---

## 7. Additional Evidence from the Gradient Dynamics

One subtle but important observation from v1's gradient tracking:

```
Epoch  1: Common 0.1% | Tail 40.2%
Epoch  5: Common 0.5% | Tail 40.1%
Epoch 10: Common 1.0% | Tail 39.8%
Epoch 15: Common 1.3% | Tail 39.7%
Epoch 20: Common 1.3% | Tail 39.6%
```

The common gradient fraction rose from 0.1% to 1.3% over 20 epochs. This means the decoder's predictions for common codes degraded slightly during Stage 2 (increasing their prediction error and thus gradient). This is consistent with the common margin dropping from 6.60 to 6.28 — the decoder is being *destabilized* at the common tier by the focused tail training. The common/medium decoder rows, though not re-initialized, are still receiving gradient (they have `requires_grad=True`), and the code-balanced batch composition perturbs them.

This is a practical concern for any future decoder-only approach: training the decoder on a radically different batch distribution than Stage 1 can subtly corrupt the well-learned common code predictions.

---

## 8. Recommended Next Steps

Given that we've now established **with high confidence** that `h` lacks discriminative features for tail codes, the next intervention must operate at the **encoder input level**. The decoder-only path has been exhausted.

### Priority 1: Co-occurrence Embedding Pre-training (Phase 2 from the implementation plan)

This is now the clear next step, for three reasons:

1. **It directly addresses the confirmed bottleneck.** The encoder receives homogenized inputs for tail codes (embedding std=0.03). PPMI+SVD embeddings give every code a unique signature by construction, breaking the vicious cycle at layer 0.

2. **It changes `h` itself.** Unlike all decoder-only interventions, distinctive embeddings give the encoder *different inputs* for different tail codes, which is a prerequisite for the encoder to build tail-specific features in `h`.

3. **The cost is low.** Pre-computation is ~5-10 minutes on CPU. A full Stage 1 retraining with new embeddings is ~$5-17. The total experiment is one training run.

**Concrete experimental plan:**

| Step | Action | Success Criterion |
|------|--------|-------------------|
| 1 | Compute PPMI+SVD embeddings from training data | tail embedding std > 0.10 |
| 2 | Run Stage 1 with pre-initialized embeddings + staged unfreezing (freeze 50%, then unfreeze at 0.1x LR) | Check post-Stage-1 tail embedding std: if still > 0.05, embeddings survived |
| 3 | Run post-Stage-1 logit diagnostics | If tail margin > 2.5 (improved from 1.66), encoder learned tail features |
| 4 | Run Stage 2 (v1-style: focused loss, LR=5e-3, 20 epochs) on the new model | tail_top10_acc > 0% is breakthrough |

### Priority 2 (If Phase 2 fails): Nonlinear (MLP) Decoder for Tail Codes

If even with distinctive embeddings the encoder doesn't build linearly-separable tail features in `h`, a 2-layer MLP decoder (`Linear(256, 128) → GELU → Linear(128, |tail|)`) could extract nonlinear combinations. But this should come after Phase 2, not before, because:
- The MLP still reads from the same `h` — if `h` genuinely contains zero tail information, a nonlinear readout won't help
- Combining distinctive embeddings + MLP decoder addresses both the input-level and readout-level barriers simultaneously

### Priority 3 (If all above fail): Accept Structural Limit or Architectural Change

At that point, the evidence would support:
- The shared encoder architecture (single `h ∈ ℝ^256` for all 6,297 codes) is fundamentally insufficient for tail codes with this data volume
- Options: per-tier encoder branches, mixture of experts, or pragmatically focusing on medium-tier improvement (where data scaling already showed 27x gains)

---

## 9. Final Summary: What We Know Now

| Question | Answer | Evidence |
|----------|--------|----------|
| Does the decoder-only approach (Kang et al.) work here? | **No** | v1 gave the decoder every advantage and tail margin stayed at -0.06 |
| Was v0's failure due to under-optimization? | **Partially yes** — loss was flat due to low LR. But fixing it didn't change the outcome. | v1 loss dropped 20x; margin still ~0 |
| Does `h` contain discriminative features for tail codes? | **No, with high confidence** | v1 definitively tested this after removing all optimization confounds |
| Where is the bottleneck? | **Encoder input level** — embedding homogenization (tail std=0.03) prevents the encoder from learning tail-specific features | Confirmed by: 10 experiments, 3 experts, 2 decoder-only attempts |
| What's next? | **Co-occurrence embedding pre-training** (Phase 2) | The only untested intervention that addresses the confirmed bottleneck |

This is a pivotal result. You've systematically eliminated the decoder as the bottleneck through two controlled experiments (v0 and v1). The research has narrowed the problem to a specific, testable hypothesis: *the encoder never receives distinctive enough input for tail codes to learn useful features.* Phase 2 directly tests this by providing distinctive input from the start.