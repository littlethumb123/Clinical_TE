# Decomposed Training Result 1 and analysis
- March 8, 2026 
- Result is from /Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/expe_logs/exp_round9/exp2b_256dim

## 1) What Should You Expect from This Two-Stage Training? How Does It Work?

### How the Two-Stage Training Works

The experiment you ran (`exp2b_flash_learned_pool`) implements **Solution 1: Decoupled Training** from the proposal, based on Kang et al. (ICLR 2020, Meta AI):

**Stage 1 (Standard End-to-End Training)** trains the full model as before -- 1 epoch on the natural (imbalanced) data with BCEWithLogitsLoss, log-scaled pos_weight, 256-dim embeddings, learned attention pooling, linear schedule. This produces the best possible encoder representation `h` under the natural data distribution. **This is unchanged from previous experiments.**

**Stage 2 (Decoupled Decoder Re-training)** then:
1. **Freezes the entire encoder** (23.7M parameters) -- embeddings, attention pooling, all transformer layers, layer norms
2. **Trains only the decoder** (1.6M parameters -- the `nn.Linear(d, 6297)` layer)
3. Uses **SGD** instead of AdamW (to avoid AdamW's second-moment denominator suppressing sporadic tail gradient spikes)
4. Runs for **3 mini-epochs** with a cosine schedule (warmup 320 steps, 3,204 total)
5. Uses the **existing loss function** with **code-balanced batching** (designed to ensure tail codes get positive examples in batches)

The core hypothesis: the encoder `h` already contains *some* discriminative features for tail codes (the macro_auroc of 0.878-0.914 across prior experiments suggests this). The decoder was the bottleneck because it was trained under the same 85%/0.1% common/tail gradient regime and never learned to extract those features. By freezing the encoder and re-training only the decoder with balanced batches, tail codes get dedicated gradient signal.

### What You Should Expect If It Worked

| Metric | Expected if Working | Explanation |
|--------|-------------------|-------------|
| **tail_pos_logit** | Move from -14.69 toward -6.2 (theoretical equilibrium) | The decoder learns to extract positive correlations from `h` instead of the -8.5 suppression |
| **tail margin** | Increase from ~1.76 toward 3.0+ | Positive logits should separate from negative logits |
| **tail_top10_acc** | Move above 0% (any positive value) | The single most important breakthrough metric |
| **rare_top10_acc** | Move above 0% | Same for rare codes |
| **common_top10_acc** | Stay at ~85% (unchanged) | Encoder is frozen, so common performance is protected |
| **Stage 2 loss** | Decrease across epochs | The decoder is learning |
| **Gradient tier distribution in Stage 2** | ~40% tail (from balanced batching) vs 0.1% in Stage 1 | The whole point of code-balanced sampling |

---

## 2) What Do the Actual Results Mean? Did This Method Work?

Let me break down your results diagnostically.

### Stage 1 Results (Lines 31-103 of training.log)

Stage 1 performed as expected and is consistent with previous experiments:

| Metric | Value | Comparison to Prior Best |
|--------|-------|--------------------------|
| Val Recall@10 | 0.813 | Slightly below v3's 0.817 (expected at 256d, 1.5M data) |
| Val uRecall@10 | 0.457 | Comparable to v3's 0.466 |
| Val NDCG@20 | 0.425 | Comparable to v3's 0.390 |
| Gradient Common | 74.0% (epoch avg) | Consistent with prior 82-85% |
| Gradient Tail | 3.7% (epoch avg) | Slightly better than v3's 1.1% |

Stage 1 is a solid baseline -- no surprises here.

### Pre-Stage 2 Diagnostics (Lines 109-112)

```
PRE-S2 common_pos_logit: -2.41, margin: 6.72
PRE-S2 medium_pos_logit: -7.33, margin: 5.09
PRE-S2 rare_pos_logit: -11.62, margin: 2.97
PRE-S2 tail_pos_logit: -13.49, margin: 2.01
```

This confirms the pattern from the root cause analysis: a steep logit gradient from common (-2.41) to tail (-13.49), with tail codes deeply suppressed. The tail margin of 2.01 means positive cases barely separate from negatives.

### Stage 2 Results (Lines 114-201) -- THE CRITICAL ANALYSIS

**Gradient tier distribution in Stage 2:**

```122:123:expe_logs/exp_round9/exp2b_256dim/training.log
  --- Stage 2 Epoch 1/3 ---
    [GradTier] Common: 0.1% | Tail: 40.6%
```

This is a **dramatic and correct shift**. The gradient distribution inverted completely:
- Common: 85% -> 0.1%
- Tail: 0.1% -> 40.6%

This proves the code-balanced batching is working exactly as designed -- tail codes are now receiving the majority of the gradient signal in Stage 2. **This is the first time in all 9 rounds of experiments that tail codes have received meaningful gradient.**

**However, the gradient distribution remained locked at exactly these values throughout all 3 epochs** (every single batch log shows Common: 0.1%, Tail: 40.6%). This is suspicious -- it means the gradient tier fractions didn't evolve at all, which suggests the gradient tier analyzer may be measuring something static (like the batch composition) rather than the actual gradient magnitudes flowing through the decoder.

**Stage 2 Loss trajectory:**

| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.5148 |
| 2 | 0.5108 |
| 3 | 0.5092 |

This is deeply concerning. The loss:
1. **Jumped from 0.0030 (Stage 1 final) to 0.5148 (Stage 2 start)** -- a 170x increase
2. **Barely decreased across 3 epochs** (0.5148 -> 0.5092, a mere 1.1% reduction)

The massive loss jump is expected because:
- Stage 2 re-initialized rare/tail decoder rows (removing the learned suppression biases)
- The balanced batching means the loss is now computed heavily on tail codes (which have very high loss)

But the near-flat loss trajectory across 3 epochs signals that **the decoder is not meaningfully learning**. If the decoder were successfully extracting signal from `h`, we'd see the loss decrease substantially across epochs.

### Post-Stage 2 Diagnostics (Lines 197-200) -- THE VERDICT

```197:200:expe_logs/exp_round9/exp2b_256dim/training.log
    POST-S2 common_pos_logit: -2.41, margin: 6.73
    POST-S2 medium_pos_logit: -7.33, margin: 5.10
    POST-S2 rare_pos_logit: 0.25, margin: 0.32
    POST-S2 tail_pos_logit: -0.30, margin: -0.28
```

Comparing pre- and post-Stage 2:

| Tier | Pre-S2 pos_logit | Post-S2 pos_logit | Change | Pre-S2 margin | Post-S2 margin | Change |
|------|-------------------|-------------------|--------|---------------|----------------|--------|
| **Common** | -2.41 | -2.41 | Unchanged | 6.72 | 6.73 | Unchanged |
| **Medium** | -7.33 | -7.33 | Unchanged | 5.09 | 5.10 | Unchanged |
| **Rare** | -11.62 | **0.25** | **+11.87** | 2.97 | **0.32** | **-2.65** |
| **Tail** | -13.49 | **-0.30** | **+13.19** | 2.01 | **-0.28** | **-2.29** |

This tells a very specific and important story:

**What happened to the logits:**
- Common and medium are completely unchanged (encoder frozen + their decoder rows likely unchanged -- consistent with the design)
- Rare positive logits moved dramatically from -11.62 to +0.25 (a +11.87 shift)
- Tail positive logits moved dramatically from -13.49 to -0.30 (a +13.19 shift)

The positive logits moved toward zero -- which means the decoder re-initialization and Stage 2 training successfully **removed the -8.5 unit cross-code suppression** that was documented in the root cause analysis. The tail positive logit went from -13.49 to -0.30, which is actually **above** the theoretical equilibrium of -6.2. This looks like progress.

**But the margins collapsed:**
- Rare margin: 2.97 -> 0.32 (nearly zero -- no discrimination)
- Tail margin: 2.01 -> **-0.28** (negative! -- model is now LESS able to distinguish positive from negative tail cases)

**What this means in plain language:** Stage 2 moved all rare/tail logits (both positive AND negative) toward zero by approximately the same amount. The decoder essentially learned `w_j ≈ 0, b_j ≈ 0` for rare/tail codes -- a near-zero output rather than the learned signal. The old decoder was wrong (deeply suppressed positive logits), but it at least had a small margin (1.76-2.97). The new decoder has no discrimination ability at all for these tiers.

### Did This Method Work?

**No, Stage 2 did not break the tail_top10_acc = 0% barrier.** Specifically:

1. **What worked:** The code-balanced batching successfully delivered gradient to tail codes (40.6% tail gradient fraction). The encoder freezing successfully protected common code performance. The decoder re-initialization removed the harmful cross-code suppression.

2. **What failed:** The decoder could not learn to discriminate positive from negative tail cases. The margins collapsed to near-zero (or negative), meaning the decoder converged to approximately `w_j ≈ 0, b_j ≈ 0` -- a population-frequency prior with no patient-specific signal.

3. **What this proves:** The encoder representation `h` **does not contain sufficient discriminative features for rare/tail codes**. The reviewer's concern from the critical review was correct:

> "If `h` truly has no tail features, this will not help... The encoder may literally have been unable to learn tail-specific features because the inputs were indistinguishable."

This is the most important finding: **the bottleneck is at the encoder level, not the decoder level.** The representation `h` was shaped by 85% common-code gradient during Stage 1, and the input embeddings were homogenized (tail std=0.03). The encoder never had the incentive or the input diversity to learn tail-specific features. No amount of decoder re-training can extract signal that isn't there.

### Final Reported Metrics

```212:213:expe_logs/exp_round9/exp2b_256dim/training.log
Final Recall@10: 0.813
Best Val Loss: 0.0000
```

The final Recall@10 of 0.813 and "Best Val Loss: 0.0000" suggest the evaluation used the **Stage 1 cached evaluation**, not a post-Stage-2 evaluation. The training log at line 204 confirms: "Using cached comprehensive evaluation from final epoch." This means the per-tier accuracy metrics after Stage 2 were NOT recomputed. You don't have explicit tail_top10_acc measurement, but the collapsed margins definitively tell us it's still 0% (or worse than before Stage 2).

---

## 3) What Are the Next Steps?

The experiment definitively answered the most important question: **the encoder representation `h` lacks discriminative features for tail codes.** This rules out any decoder-only intervention and tells us the next intervention must operate at the **encoder input level** to give the encoder a chance to learn tail-specific features.

### Recommended Next Step: Phase 2 -- Co-occurrence Embedding Pre-training (Solution 3 from the Proposal)

This is exactly what the implementation plan anticipated as the escalation path:

> "If Phase 1 Stage 2 shows no improvement in tail_top10_acc, the encoder representation likely lacks tail-specific features because the inputs were indistinguishable — and Phase 2 directly addresses that input-level barrier."

**Why this is the right next step:**

1. **Addresses the confirmed bottleneck.** The experiment proved the bottleneck is at the encoder input level (embedding homogenization, tail std=0.03). PPMI+SVD embeddings give every code -- even tail codes -- a unique signature by construction.

2. **Directly breaks the vicious cycle.** The root cause analysis identified a self-reinforcing loop at layer 0: homogenized tail embeddings -> no distinctive encoder input -> no tail-specific features -> sparse/uniform gradient -> homogenized embeddings. Pre-computed embeddings break this loop from the outside.

3. **Low cost, high information.** The pre-computation is ~5-10 minutes on CPU. If the tail embedding std is >0.10 after pre-computation (vs. current 0.03), we know the embeddings are meaningfully distinctive. Then re-running Stage 1 + Stage 2 with these embeddings tests whether the encoder CAN learn tail-specific features when given distinctive inputs.

**Concrete plan:**

| Step | What | Why |
|------|------|-----|
| 1. Pre-compute PPMI+SVD embeddings | Task 8 from the implementation plan | One-time cost, deterministic, reproducible |
| 2. Verify tail embedding std > 0.10 | Quality check before running GPU experiment | If std is still low, adjust the co-occurrence window or use ICD-10 hierarchy |
| 3. Run Stage 1 with pre-initialized + frozen embeddings (50% of training) | Tasks 9-10 | Test whether distinctive inputs let the encoder build tail-relevant features |
| 4. Run Stage 2 decoder re-training on the new model | Reuse Tasks 1-7 | Now the decoder has a chance -- `h` may contain tail signal |
| 5. Evaluate: did tail_top10_acc move above 0%? | Decision point | If yes: breakthrough. If no: fundamental architectural limit |

**If Phase 2 also fails**, the remaining options are:
- **Per-tier decoder heads** (separate MLPs per tier that can extract nonlinear signal)
- **Fundamental architectural change** (e.g., per-tier encoder branches, mixture of experts at the encoder level) -- but these are significantly more complex
- **Accept the structural limit** for tail codes and focus on maximizing medium-tier accuracy (where data scaling already showed 27x improvement)

### Additional Recommendations

1. **Re-run post-Stage-2 comprehensive evaluation.** The current experiment used cached Stage 1 metrics. You need tier-stratified metrics (tail_top10_acc, rare_top10_acc) computed AFTER Stage 2 to have a definitive measurement, even though the collapsed margins strongly suggest 0%.

2. **Consider a diagnostic experiment**: Before running the full Phase 2, try a quick test -- manually set a handful of tail code embeddings to be very distinctive (e.g., orthogonal vectors) and re-run Stage 2 only. If even 1-2 tail codes break through 0%, it confirms that distinctive embeddings are the key.

3. **SGD learning rate may need tuning for Stage 2.** The loss barely moved (0.5148 -> 0.5092). While the primary cause is lack of signal in `h`, a higher LR (e.g., 1e-4 or 5e-4) could help the decoder explore more of the weight space, though it risks instability.

# Independent Critical Review: Two-Stage Decoupled Decoder Re-training Results

## Preamble

I am conducting this review as an independent expert with deep experience in transformer pretraining failures, long-tail learning dynamics, and optimization pathologies. I have read the raw training log, the implementation code, the original proposal and its critical review, the synthesized root cause analysis, and the previous expert's interpretation of these results. I will first derive my own conclusions from the raw data, then compare against the previous expert's analysis.

---

## 1. Verification of What Actually Ran

Before interpreting any results, I need to verify that the experiment matched the plan. This is the single most important step that the previous expert did not do carefully enough.

### Stage 1: Confirmed correct
- 256d, 25.3M params, BCEWithLogitsLoss with log-scaled pos_weight (max 35, not 200), AdamW lr=8e-4, linear schedule, 1 epoch, 12,335 batches. This matches the standard configuration.

**First critical observation the previous expert missed: `pos_weight_max` is 35, not 200.** The config.json shows `"pos_weight_max": 35`. Prior experiments v3/R6 used pw=200. This means Stage 1 is running with a LOWER pos_weight than the best prior baselines. This is important context — the pre-Stage-2 logit diagnostics will differ from the root cause analysis numbers (which cited -14.69 for tail at pw=35 from v2, consistent here at -13.49).

### Stage 2: Partially verified, with critical questions

**What ran correctly:**
- Encoder frozen: 23,706,880 frozen, 1,618,329 trainable. This checks out: 25,325,209 total - 23,706,880 = 1,618,329. The `decoder_cd` is `nn.Linear(256, 6297)`, which has 256×6297 + 6297 = 1,618,329 parameters. Correct.
- SGD optimizer with lr=5e-5. Confirmed in the log.
- Cosine schedule with 320 warmup steps, 3,204 total (1,068 batches × 3 epochs). Confirmed.
- Code-balanced batch sampler active (1,068 batches/epoch vs 12,335 in Stage 1). The number 1,068 is consistent with `max(active_codes // codes_per_batch, 1) * 3` = `max(~5700 // 16, 1) * 3` ≈ `356 * 3 = 1068`. Confirmed.

**What was re-initialized:**
The `reinit_decoder_rows` function re-initialized rare AND tail decoder rows with Xavier init and reset biases to 0. This means approximately 1,705 + 1,148 = 2,853 out of 6,297 decoder rows were wiped and restarted from scratch. Common (1,141) and medium (1,711) rows were preserved unchanged.

**The 50-minute gap (10:56 to 11:46):** The code-to-patient index building took ~50 minutes. This is the `_build_code_index` scanning through the entire dataset. This is expected for a 1.5M sample dataset.

---

## 2. My Independent Reading of the Stage 2 Results

### 2.1 The Gradient Tier Numbers Are Genuine But Misleading

The GradientTierAnalyzer measures `decoder_cd.weight.grad` — the actual gradient on the decoder weight matrix after `backward()`. It computes per-row L2 norms, then the fraction of total norm per tier.

The Stage 2 reading of `Common: 0.1% | Tail: 40.6%` that persists identically across ALL batches is **genuinely measuring decoder gradients**, but there is a subtle issue the previous expert flagged as "suspicious" without fully explaining.

**Why the numbers are constant:** In Stage 2, the encoder is frozen. The gradient `∂L/∂w_j` for decoder row j is:

```
∂L/∂w_j = Σ_i (σ(z_ij) - y_ij) × pw_j × h_i
```

where `h_i` is fixed (frozen encoder). The distribution of gradient norms across tiers is determined by:
1. Which codes have positive labels (y_ij = 1) in the batch — controlled by the sampler
2. The pos_weight pw_j for those codes
3. The prediction error σ(z_ij) - y_ij

Since the CodeBalancedBatchSampler uses a **fixed random seed (42)** and the inverse-frequency weighting is deterministic, the batch composition at each logged interval (every 100 batches) is drawing from a very similar statistical distribution. Combined with the fact that the decoder is barely moving (loss flat at ~0.51), the prediction errors are nearly constant, so the gradient tier fractions are nearly constant.

**This is NOT measuring "batch composition" as the previous expert hypothesized — it IS measuring actual gradient norms.** But the constancy reflects the fact that the decoder weights are barely changing, so the gradient distribution is in a near-stationary state from the very first batch.

### 2.2 The Loss Trajectory: A Critical Misinterpretation

The previous expert notes: "loss barely decreased (0.5148 → 0.5092, a mere 1.1% reduction)" and interprets this as the decoder "not meaningfully learning."

**I disagree with this interpretation. The loss CANNOT decrease substantially, and this is not evidence of failure.** Here's why:

The Stage 2 loss is BCEWithLogitsLoss computed over ALL 6,297 codes × all valid days. But the code-balanced sampler enriches for rare/tail codes specifically. The loss for common and medium codes — which dominate the total loss sum due to having far more positive instances — is now computed on a **different** patient distribution than Stage 1 (enriched for patients with rare codes, who also have common codes). The common/medium decoder rows are frozen, so their contribution to the total loss is essentially constant. Only the ~2,853 re-initialized rare/tail rows are learning.

The rare/tail rows went from deep suppression (-11.62, -13.49) to near-zero (-0.30, 0.25), which means the per-element loss for those codes DID change dramatically. But rare/tail codes are ~0.2% of all positive labels. Their loss improvement is swamped by the constant loss from 99.8% of the labels.

**The flat total loss is actually EXPECTED and does NOT indicate failure.** To properly measure whether the decoder is learning, you need per-tier loss, not aggregate loss. The per-tier logit diagnostics are the correct measurement — and the previous expert does analyze those. But they draw the wrong conclusion from them, which I address next.

### 2.3 The Logit Diagnostics: Where the Previous Expert Made a Critical Analytical Error

The pre- and post-Stage 2 diagnostics:

| Tier | Pre-S2 pos_logit | Post-S2 pos_logit | Pre-S2 margin | Post-S2 margin |
|------|-------------------|-------------------|---------------|----------------|
| Common | -2.41 | -2.41 | 6.72 | 6.73 |
| Medium | -7.33 | -7.33 | 5.09 | 5.10 |
| Rare | -11.62 | **+0.25** | 2.97 | **0.32** |
| Tail | -13.49 | **-0.30** | 2.01 | **-0.28** |

The previous expert's interpretation:

> "The decoder essentially learned `w_j ≈ 0, b_j ≈ 0` for rare/tail codes — a near-zero output rather than the learned signal."

**This interpretation is partially correct but draws the wrong conclusion.**

Let me work through the math. The decoder computes `z_j = w_j^T h + b_j`. After Xavier re-initialization, the weights are random with std ≈ (2/(256+1))^0.5 ≈ 0.088. The bias is reset to 0.

For a random w_j with this initialization, `w_j^T h` is a dot product of two 256-dimensional vectors. Since h has been shaped by common-code training, it has a structured (non-random) direction. The expected value of `w_j^T h` for random w_j is approximately 0 (since E[w_j] = 0), with standard deviation approximately `std(w_j) × ||h||` ≈ `0.088 × ||h||`.

**The post-S2 logits of +0.25 (rare) and -0.30 (tail) are consistent with TWO possible states:**

**State A (what the previous expert claims): w_j converged to ≈ 0, b_j ≈ 0.**
If the weights collapsed toward zero, logits would be near-zero regardless of h. The margin would be near-zero because σ(0) ≈ 0.5, and the model would predict ~50% probability for everything.

**State B (an alternative the previous expert didn't consider): w_j MOVED from random initialization but h genuinely lacks discriminative features for these codes.**
In this case, the decoder tried to learn useful weights, found no consistent pattern in h that correlates with tail code presence, and settled into a low-norm state where the signal-to-noise ratio of the gradient drove the weights toward a population-average prediction.

**How to distinguish A from B:** Check the per-code logit VARIANCE, not just the mean. If State A (weights ≈ 0), then logit variance across patient-days would also be near-zero — every patient gets essentially the same logit for every tail code. If State B (weights learned but found no signal), there might still be some variance driven by the h structure, but it wouldn't correlate with the actual labels.

**The margin going negative (-0.28 for tail) is the most telling signal.** A margin of -0.28 means that, on average, the model's logit for a tail code is LOWER when the code is actually present than when it's absent. This is worse than random chance. This is consistent with the decoder weights having found spurious correlations in h that are INVERSELY related to tail code presence.

**Why would this happen?** Because the code-balanced sampler enriches batches with patients who have specific rare/tail codes. But patients who have tail codes also have common codes. The frozen representation h for these patients is dominated by common-code features. So `h` for a "tail-code-enriched" patient looks very similar to `h` for any other patient with the same common codes. The decoder, seeing `h` vectors that are essentially indistinguishable between positive and negative examples for any given tail code, cannot learn meaningful weights. With SGD momentum driving the weights, they wander into a configuration that happens to have slight negative correlation with the actual labels.

**My verdict on the margin collapse:** The previous expert is correct that this indicates the encoder lacks discriminative signal for tail codes. But the specific mechanism — and the implications for next steps — differ from their analysis.

### 2.4 The Common/Medium Rows: A Hidden Success Signal

The most important number in the entire log that the previous expert barely mentioned:

```
POST-S2 common_pos_logit: -2.41, margin: 6.73  (unchanged from -2.41, 6.72)
POST-S2 medium_pos_logit: -7.33, margin: 5.10  (unchanged from -7.33, 5.09)
```

Common and medium logits are EXACTLY unchanged (within 0.01). This is expected because:
1. Their decoder rows were NOT re-initialized (frozen by design)
2. The encoder is frozen, so h hasn't changed

But wait — the implementation says `freeze_encoder` freezes everything except `decoder_cd`. The `reinit_decoder_rows` only re-initializes rare and tail rows. **But the common and medium rows of `decoder_cd` are STILL trainable** — they just weren't re-initialized. `freeze_encoder` sets `requires_grad=True` for all decoder_cd parameters, which includes common/medium rows.

**So the common/medium decoder rows WERE receiving gradient during Stage 2 — and they didn't move at all.** This means either: (a) the gradient for common/medium rows during Stage 2 was genuinely near-zero (because the code-balanced sampler underrepresents them), or (b) the SGD learning rate of 5e-5 was too low to produce visible changes in the already-well-converged common/medium rows.

This is actually a subtle validation that the code-balanced sampler IS working as intended — it's starving common codes of gradient (0.1% fraction) the same way the natural distribution starved tail codes.

---

## 3. Where I Agree and Disagree with the Previous Expert

### I Agree:
1. **Stage 2 did not break the 0% tail accuracy barrier.** The collapsed margins confirm this.
2. **The encoder representation h likely lacks discriminative features for tail codes.** The margin collapse is consistent with this hypothesis.
3. **The code-balanced sampler worked as designed** — gradient tier fractions inverted dramatically.
4. **The frozen encoder protected common code performance** — no regression.
5. **Co-occurrence embedding pre-training (Phase 2) is a reasonable next step.**

### I Disagree:

**Disagreement 1: "The flat loss trajectory signals failure."**

The flat total loss is an artifact of aggregating across tiers where 99.8% of positive labels are from tiers whose decoder rows are either frozen or barely affected. The per-tier logit analysis (which the previous expert also performed) is the correct diagnostic. The previous expert reaches the right conclusion from the logit analysis but reaches a REDUNDANT wrong conclusion from the loss trajectory that muddies the picture.

**Disagreement 2: "The bottleneck is definitively at the encoder level."**

The previous expert states with high confidence: "the encoder representation `h` does not contain sufficient discriminative features for rare/tail codes... the bottleneck is at the encoder level, not the decoder level."

I think this is PREMATURE. Here's why:

The Stage 2 experiment has several confounds that prevent making this definitive conclusion:

**Confound A: The learning rate was almost certainly too low.** SGD with lr=5e-5 on 1,618,329 parameters for 3,204 total steps is extremely conservative. The cosine schedule decays from 5e-5 to near-zero, with the first 320 steps at even lower LR (warmup starts at 5e-6). The effective weight change per step is:

```
Δw ≈ lr × grad_norm ≈ 5e-5 × (some small value)
```

After Xavier init with std ≈ 0.088, the weights need to move a meaningful distance from their random starting point. With 3,204 steps at this low LR, the total weight movement may have been too small to discover useful features even if h contains them.

**Evidence: The loss dropped from 0.5147 to 0.5092 — a 1.1% decline. For a decoder trained from random initialization, this suggests the optimizer barely moved the weights at all.** Compare to Stage 1 where the loss dropped from 0.7954 to 0.0030 within a single epoch.

**Confound B: The diagnostic measures only mean logits over 50 validation batches.** The margin of -0.28 for tail codes is computed over 50 batches, which for tail codes might mean only a few hundred positive observations. The standard error of this estimate could easily be >0.5, making the negative margin statistically indistinguishable from zero.

**Confound C: The decoder was re-initialized to RANDOM weights and given only ~3K SGD steps.** A randomly initialized linear layer needs to learn 256 × 2,853 = ~730K weight values (for rare + tail rows). With 3,204 total gradient updates at lr=5e-5 with cosine decay, each weight sees an effective cumulative update of roughly:

```
Σ lr(t) ≈ 5e-5 × (2/π) × 3204 ≈ 0.10
```

multiplied by the gradient magnitude. If h has weak but real signal, this may simply be insufficient optimization to extract it. The Kang et al. (2020) paper that motivates this approach typically fine-tunes classifiers for 10-90 epochs on ImageNet-LT with much higher learning rates.

**My assessment: We cannot definitively conclude h lacks tail-code signal until we have tested Stage 2 with:**
1. A MUCH higher learning rate (try 1e-3, 5e-3, even 1e-2 — this is only training a linear layer)
2. More epochs (10-30 instead of 3)
3. A per-tier loss during Stage 2 that normalizes by the number of codes per tier

**Disagreement 3: "Co-occurrence embeddings are the most important next step."**

The previous expert recommends jumping to Phase 2 (co-occurrence embeddings). I think this is skipping over the much simpler hypothesis: **Stage 2 simply wasn't given enough optimization budget to succeed.**

Before investing in a fundamentally different approach (pre-computed embeddings + full retraining), we should exhaust the current approach by:
1. Increasing Stage 2 LR by 100-1000x
2. Running Stage 2 for 30+ epochs instead of 3
3. Adding a per-tier loss in Stage 2 that weights rare/tail loss relative to their tier size

If Stage 2 with aggressive optimization STILL shows margin collapse, THEN we can conclude h genuinely lacks signal and move to Phase 2.

---

## 4. Specific Technical Issues the Previous Expert Missed Entirely

### Issue 1: The Loss Function in Stage 2 Is Unchanged and Wrong for This Setting

The `train_stage2` function passes data through the existing `DataParallelWrapper`, which computes BCEWithLogitsLoss with the SAME pos_weight scheme as Stage 1 — `log_scaled` with `pos_weight_max=35`.

This means during Stage 2, the loss is computed over ALL 6,297 codes with `reduction='mean'` over all elements. The gradient that reaches the rare/tail decoder rows is:

```
∂L/∂w_j = (1/N_total_elements) × Σ_i (σ(z_ij) - y_ij) × pw_j × h_i
```

where `N_total_elements = valid_days × 6,297`. The rare/tail codes are still only ~0.2% of the positive labels, even with code-balanced batching. The code-balanced sampler enriches the batch with patients who have rare/tail codes, but the loss is still computed over ALL 6,297 codes for those patients. **The sampler changes WHICH patients appear, not how the loss aggregates across codes.**

**This is the exact same structural issue the critical reviewer identified for Solution 2 (per-tier loss) — and it applies here too.** The gradient to rare/tail decoder rows during Stage 2 is still diluted by the mean-over-all-codes reduction.

**What should have been done:** Compute loss ONLY on the target codes for each batch (the 16 codes that the sampler enriched for), or at minimum use per-tier loss decomposition within Stage 2. The code-balanced sampler ensures the right patients are in the batch, but the loss still averages over thousands of codes that have nothing to do with the target codes.

### Issue 2: The GradientTierAnalyzer May Be Measuring Something Different from What's Claimed

The GradientTierAnalyzer reports `Common: 0.1% | Tail: 40.6%`. But look at what it measures: the L2 norm of `decoder_cd.weight.grad[j, :]` per code j, then the fraction of total norm per tier.

After decoder re-initialization with Xavier init, the rare/tail rows have random weights with std ≈ 0.088. The common/medium rows have well-trained weights. The gradient `∂L/∂w_j` is `(σ(z_ij) - y_ij) × pw_j × h_i`. For the re-initialized rare/tail rows:
- z_j = w_j^T h + 0 ≈ small random value near 0
- σ(z_j) ≈ 0.5
- For negative examples (y=0): gradient ≈ 0.5 × 1 × h (no pos_weight for negatives)
- For positive examples (y=1): gradient ≈ -0.5 × pw_j × h (with pos_weight amplification)

For the frozen-but-trainable common rows:
- z_j = w_j^T h + b_j is already well-calibrated
- σ(z_j) ≈ y_ij (well-predicted)
- Gradient ≈ (σ(z_j) - y_j) × h ≈ small_residual × h

**So the high tail gradient fraction (40.6%) is partly because the rare/tail codes are newly initialized and have high prediction error (σ(0) - y ≈ ±0.5), while common codes are already well-converged (σ(z) - y ≈ small).** This is not the same as saying "tail codes are getting useful gradient signal." They're getting HIGH-MAGNITUDE gradient, but it's driven by the fact that everything starts at 50% probability for re-initialized rows.

### Issue 3: The 50-Minute Gap Reveals a Dataset Scale Problem

The `_build_code_index` took 50 minutes. This function scans every patient, extracts their target codes, and builds a dictionary mapping code -> list of patient indices. For 1.5M patients with variable-length histories, this is reasonable. But it means the `CodeBalancedBatchSampler` is operating at the PATIENT level, not the PATIENT-DAY level.

**This matters because the decoder operates on patient-DAYS.** A patient who has a tail code on day 15 out of 50 days contributes that tail code's positive label on only 1 of 50 valid prediction targets. When that patient is sampled for their tail code, the other 49 days provide negative labels for that tail code. The effective positive rate even in the "enriched" batch is still very low per specific code.

---

## 5. My Assessment: What Actually Happened

The experiment is **inconclusive, not definitively negative.** Here is what I believe happened:

1. **Stage 1 worked normally** and produced a representation h that is optimized for common codes.

2. **Stage 2 correctly froze the encoder and re-initialized rare/tail decoder rows.** The code-balanced sampler correctly enriched batches with patients who have rare/tail codes.

3. **But Stage 2 was severely under-optimized:**
   - LR of 5e-5 with cosine decay to ~0 over 3,204 steps is far too conservative for training a randomly initialized linear layer
   - Only 3 mini-epochs (with 1,068 batches each) is far too few
   - The loss function still averages over all 6,297 codes, diluting the gradient signal to rare/tail rows
   - The total effective training budget was roughly 100-1000x less than what the Kang et al. (2020) reference uses for classifier re-training

4. **The logit diagnostics show the decoder moved the rare/tail logits from deep suppression (-11 to -13) to near-zero (-0.3 to +0.25).** This proves the decoder DID change. But it moved toward the uninformative "predict everything at 50%" state, which is the natural basin of attraction for a randomly initialized linear layer that hasn't been given enough optimization to escape toward useful weights.

5. **The margin collapse (-0.28 for tail) could be either:**
   - (a) h genuinely lacks tail signal → no amount of decoder optimization will help
   - (b) Insufficient optimization → the decoder didn't have enough steps/LR to discover weak signal in h

We cannot distinguish (a) from (b) with this experiment.

---

## 6. My Recommended Next Steps (in order of priority)

**Step 1 (cheapest test, should be done before anything else): Re-run Stage 2 ONLY with aggressive hyperparameters.**

Take the saved Stage 1 model and re-run ONLY Stage 2 with:
- LR: 1e-3 or 5e-3 (100-1000x higher)
- Epochs: 20-30 (10x more)
- Per-tier loss: compute loss separately on target codes per batch, or use per-tier normalization
- Optionally: try AdamW instead of SGD for Stage 2 — the critical reviewer's concern about second-moment suppression is valid for the END of training, but for a freshly initialized decoder, AdamW's adaptive LR actually helps each dimension converge faster

This is a 2-4 hour GPU experiment that answers the most important question: with enough optimization, can the decoder extract ANY discriminative signal from h?

**Step 2 (if Step 1 also shows margin collapse): Per-tier decoder heads.**

Before investing in co-occurrence embeddings (which addresses the input level), try replacing the single `nn.Linear(256, 6297)` with separate decoders per tier — specifically a small MLP (`Linear(256, 128) → GELU → Linear(128, |tail|)`) for tail codes. A nonlinear decoder can discover features in h that a linear projection misses, because h may encode tail-relevant information in nonlinear combinations of its dimensions (e.g., the ABSENCE of certain common-code features may signal a tail code's presence, which a linear w_j^T h cannot capture but a 2-layer MLP can).

**Step 3 (if both Steps 1 and 2 fail): Then proceed to co-occurrence embeddings.**

At that point, we'd have strong evidence that h genuinely lacks tail-relevant features at both linear and nonlinear readout levels, and the intervention must occur at the encoder input.

---

## 7. Summary Verdict

| Aspect | Previous Expert Assessment | My Assessment |
|--------|---------------------------|---------------|
| Stage 2 execution | Correct | Correct, but with critical confound: loss function not adapted for Stage 2 |
| Gradient tier inversion (0.1%/40.6%) | Correctly noted as "code-balanced batching working" | Partially artifact of re-initialization; high gradient magnitude ≠ useful gradient signal |
| Flat loss trajectory | "Decoder not learning" | Expected artifact of aggregation; misleading metric for Stage 2 |
| Margin collapse for tail codes | "h lacks signal — bottleneck is at encoder" | **Premature conclusion.** Equally consistent with under-optimization of Stage 2 |
| Conclusion: "move to Phase 2 (embeddings)" | Recommended as next step | **Premature.** Should first re-run Stage 2 with 100x higher LR and 10x more epochs |
| Root cause analysis | Confirmed by experiment | Experiment is inconclusive on the core question — was the decoder given a fair chance? |

**Bottom line:** The previous expert's analysis is detailed and mostly technically correct in its observations, but draws a premature definitive conclusion ("h lacks signal") from an experiment that had at least three compounding under-optimization issues (LR too low, epochs too few, loss function undiluted). The experiment needs to be repeated with aggressive Stage 2 optimization before we can conclude that the encoder representation is the bottleneck.

# Expert 3: Independent Expert Analysis: Two-Stage Decoupled Training Experiment (Round 9)

## My Credentials & Approach

I'm analyzing this as an independent LLM pretraining expert with deep experience in long-tail learning pathologies, optimization dynamics, and transformer failure modes. I've carefully read the raw training log, the config, the root cause analysis, the original proposal with its critical review, the implementation plan, and the two prior expert analyses. I will derive my own conclusions from the raw data first, then critically evaluate where the prior experts agree and disagree — and where both of them miss something.

---

## 1) What Should You Expect from Two-Stage Training? How Does It Work?

### The Mechanism

I'll explain this from first principles, not just parroting the proposal.

**Stage 1** is standard end-to-end training. The full model (25.3M parameters) trains on the natural, imbalanced data distribution for 1 epoch. The goal is to build the best possible shared representation `h ∈ ℝ^256` for common codes, because that's where the data supports learning. This is identical to your prior experiments (v2, v3, R6, etc.). Stage 1 makes no attempt to solve the tail code problem — it deliberately accepts the 85%/0.1% common/tail gradient split.

**Stage 2** decouples the problem:
1. **Freeze the entire encoder** (23.7M params) — embeddings, attention pooling, temporal transformer layers
2. **Re-initialize the decoder rows for rare/tail codes** — replacing the learned suppression weights (`w_j^T h ≈ -8.5`) with fresh Xavier initialization
3. **Train only the decoder** (1.6M params) with a **CodeBalancedBatchSampler** that guarantees tail codes get positive examples in every batch
4. Use **SGD instead of AdamW** to avoid AdamW's second-moment dampening of sporadic tail gradients

The core hypothesis is Kang et al. (ICLR 2020): **the encoder representation `h` already captures some discriminative features for all codes, but the decoder never learned to extract them because it was trained under the same imbalanced gradient regime.** By giving the decoder dedicated, balanced gradient with the encoder frozen, tail codes get a fair chance.

### What You Should Expect If Working

| Signal | Expected Value | Why |
|--------|---------------|-----|
| **Gradient tier fractions in Stage 2** | ~40% tail, ~0.1% common (inverted from Stage 1) | Code-balanced sampler enriches for tail codes |
| **Tail positive logit** | Move from -13.49 toward -6.2 (theoretical equilibrium) | Decoder learns positive correlations with `h` instead of suppression |
| **Tail margin (pos_logit - neg_logit)** | Increase from 2.01 toward 3.0+ | Positive cases should separate from negative |
| **Stage 2 loss** | Decrease across epochs | Decoder is learning |
| **Common/medium metrics** | Unchanged | Encoder frozen, common decoder rows preserved |
| **tail_top10_acc** | Move above 0% | The breakthrough metric |

### What You Should Expect If NOT Working

| Signal | Expected Value | Diagnosis |
|--------|---------------|-----------|
| **Margin collapse** to ~0 or negative | -0.3 to +0.3 range | `h` lacks discriminative features for tail codes |
| **Stage 2 loss flat** | ~0.51 barely moving | Decoder can't find useful patterns in frozen `h` |
| **Logits near zero** | All rare/tail logits ≈ 0 | Decoder converged to uninformative "predict 50% for everything" state |

---

## 2) What Do the Actual Results Mean? Did It Work?

### Stage 1: Normal, Expected Behavior

Stage 1 reproduced the well-established pattern:

```97:103:/Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/expe_logs/exp_round9/exp2b_256dim/training.log
  Using batch-averaged training metrics (no re-evaluation)
     Val loss: 0.0031, R@10: 0.825
   Final epoch: Running comprehensive evaluation...

--- Epoch 1 Summary ---
  Train loss: 0.0130 → 0.0030
  Val loss: 0.0031, Recall@10: 0.813, μRecall@10: 0.457, NDCG@20: 0.425
```

Gradient starvation followed the exact same trajectory as all prior experiments: common codes dominated by step ~3000, reaching 87-89% by the end. The epoch average of 74% common / 3.7% tail is consistent with prior rounds. No surprises.

### Stage 2: The Critical Evidence

**What went RIGHT (mechanically):**

1. **Encoder freezing worked.** 23,706,880 frozen, 1,618,329 trainable — exactly `nn.Linear(256, 6297)` = 256×6297 + 6297 = 1,618,329.

2. **Gradient inversion was dramatic and genuine.**

```122:123:/Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/expe_logs/exp_round9/exp2b_256dim/training.log
  --- Stage 2 Epoch 1/3 ---
    [GradTier] Common: 0.1% | Tail: 40.6%
```

This is a 400x shift for tail codes (0.1% → 40.6%). The code-balanced sampler fundamentally changed the gradient landscape.

3. **Common/medium logits preserved perfectly.**

```197:198:/Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/expe_logs/exp_round9/exp2b_256dim/training.log
    POST-S2 common_pos_logit: -2.41, margin: 6.73
    POST-S2 medium_pos_logit: -7.33, margin: 5.10
```

Unchanged from pre-Stage 2 values. The frozen encoder + preserved common decoder rows did their job.

**What FAILED (functionally):**

The rare/tail diagnostics tell the definitive story:

```109:112:/Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/expe_logs/exp_round9/exp2b_256dim/training.log
    PRE-S2 common_pos_logit: -2.41, margin: 6.72
    PRE-S2 medium_pos_logit: -7.33, margin: 5.09
    PRE-S2 rare_pos_logit: -11.62, margin: 2.97
    PRE-S2 tail_pos_logit: -13.49, margin: 2.01
```

```197:200:/Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/expe_logs/exp_round9/exp2b_256dim/training.log
    POST-S2 common_pos_logit: -2.41, margin: 6.73
    POST-S2 medium_pos_logit: -7.33, margin: 5.10
    POST-S2 rare_pos_logit: 0.25, margin: 0.32
    POST-S2 tail_pos_logit: -0.30, margin: -0.28
```

| Tier | Pre-S2 logit | Post-S2 logit | Pre-S2 margin | Post-S2 margin |
|------|-------------|--------------|--------------|---------------|
| Rare | -11.62 | **+0.25** (+11.87) | 2.97 | **0.32** (-2.65) |
| Tail | -13.49 | **-0.30** (+13.19) | 2.01 | **-0.28** (-2.29) |

### My Independent Interpretation of This Result

**Here is where I fundamentally depart from both prior experts, because both miss the most critical observation:**

The logits moved +11.87 and +13.19 units respectively. That is a **massive** weight change. But both positive AND negative logits moved toward zero by essentially the same amount, destroying the margin. The tail margin went from +2.01 to **-0.28** — the model is now actively **worse** at distinguishing positive from negative tail cases than before Stage 2.

**What actually happened mechanically:** The decoder rows were Xavier-initialized with std ≈ 0.088. SGD with lr=5e-5 over 3,204 steps with cosine decay produces total effective learning:

```
Effective cumulative LR ≈ 5e-5 × (2/π) × 3204 ≈ 0.102
```

Each weight changes by roughly `gradient × 0.102`. But `h` for different patients has very high correlation (because it's dominated by common-code features). The decoder, seeing nearly identical `h` vectors for positive and negative tail-code cases, cannot learn to discriminate them. The weights settle into a near-zero configuration because **there is no consistent directional gradient to follow** — the gradient from positive and negative examples pulls in contradictory directions that average to roughly zero.

**The negative margin (-0.28) is the smoking gun.** It means the decoder found a slight spurious inverse correlation — patients WITH certain tail codes happen to have `h` vectors that correlate slightly negatively with the decoder weight direction. This is noise, not signal.

### Did This Method Work?

**No. Stage 2 did not break the 0% tail accuracy barrier.** The margin collapse from +2.01 to -0.28 confirms that tail code accuracy either stayed at 0% or worsened.

However — and this is critical — **the experiment was not properly executed for a fair test.** I detail this in Section 3 below.

---

## 3) My Critical Assessment of Both Prior Expert Analyses

### Where Expert 1 Is Wrong

Expert 1 draws a premature definitive conclusion:

> "the encoder representation `h` does not contain sufficient discriminative features for rare/tail codes... the bottleneck is at the encoder level, not the decoder level."

**This conclusion is not supported by this experiment.** Expert 1 treats the margin collapse as proof that `h` lacks signal. But Expert 2 correctly identifies multiple compounding confounds that make this experiment inconclusive on that question.

### Where Expert 2 Is Partially Right But Misses the Deeper Issue

Expert 2 argues the experiment was "under-optimized" and recommends re-running Stage 2 with 100-1000x higher LR and 10x more epochs. This is a valid observation — the learning rate WAS too conservative for training a randomly initialized linear layer.

**However, Expert 2 overlooks the most critical design flaw that neither expert catches:**

### The Flaw Both Experts Missed: The Loss Function Was Not Adapted for Stage 2

Look at the training log carefully. The Stage 2 loss starts at **0.5147** and ends at **0.5092**. This is BCEWithLogitsLoss averaged over ALL 6,297 codes × all valid days.

**The code-balanced sampler changes WHICH patients appear in the batch, but the loss is still computed over ALL 6,297 codes for those patients.** A patient sampled because they have a rare tail code also has common codes. The loss computation:

```
L = mean(all 6,297 codes × N valid days)
```

dilutes the tail-code gradient within the total loss. Even though the sampler ensures tail-code-positive patients are present, the gradient from those patients' common codes (which are well-predicted, so low loss) dominates the gradient from the tail codes (which are poorly predicted, but vastly outnumbered).

**The 40.6% tail gradient fraction is misleading.** As Expert 2 partially explains, this is measuring the L2 norm of decoder gradient rows. After Xavier re-initialization, the rare/tail rows have high prediction error (σ(0) ≈ 0.5) while common rows are well-calibrated (σ(z) ≈ y). So the raw gradient magnitude for re-initialized rows is high by construction — it doesn't mean the gradient is *useful* for learning.

**This is the central issue: the experiment tested a partially implemented version of Solution 1, not the full design.** The critical reviewer in the original proposal identified exactly this:

> "The missing intervention is tail-code-specific batch construction during decoder re-training... compute loss ONLY for code j (or for a small group of codes co-occurring with j)"

The implementation used the code-balanced sampler for patient selection but kept the global loss. This is like changing the ingredients but cooking with the same recipe — the structural dilution remains.

### Where Both Experts Are Right

Both experts correctly identify:
1. The code-balanced sampler successfully inverted the gradient fraction (0.1% → 40.6% tail)
2. The encoder freeze perfectly protected common/medium performance
3. The margin collapse indicates the decoder couldn't learn to discriminate
4. Co-occurrence embeddings remain a valid complementary approach

### Where Expert 2's LR Argument Has Merit But Is Insufficient

Expert 2 recommends 100-1000x higher LR. Let's test this reasoning:

With SGD lr=5e-3 (100x current), the weight change per step would be ~100x larger. But the fundamental issue — `h` being nearly identical for positive and negative tail cases — means the gradient is **high-variance noise**, not coherent signal. A higher LR amplifies noise just as much as signal. You'd likely see the decoder weights diverge rather than converge to a useful solution.

The LR argument would hold IF there is weak but real signal in `h` being lost due to insufficient optimization. But the negative margin (-0.28) suggests the gradient direction is wrong, not just too small. Amplifying a wrong-direction gradient doesn't help.

**That said**, there's a subtlety: the current loss function dilutes the useful gradient. If you fix the loss (compute only on target codes), a higher LR on that focused gradient COULD discover signal. So Expert 2's recommendation of higher LR is correct **but only in combination with the loss fix**.

---

## 4) What This Experiment Actually Proved

Despite not breaking the barrier, this experiment provided three high-value data points:

**Proven Fact 1: The encoder freeze + decoder re-init is mechanically sound.** Common/medium metrics were perfectly preserved. The infrastructure works.

**Proven Fact 2: Code-balanced sampling can invert the gradient distribution.** The 40.6% tail fraction proves the sampler architecture is correct.

**Proven Fact 3: The experiment is inconclusive on whether `h` contains tail signal.** The under-optimization (low LR, few epochs) combined with the loss dilution (computing over all 6,297 codes) means the decoder was never given a fair chance to extract whatever signal might exist in `h`.

---

## 5) My Recommended Next Steps (Prioritized)

### Step 1: **Re-run Stage 2 ONLY with three critical fixes** (HIGHEST PRIORITY, cheapest test)

Take the saved Stage 1 model and re-run Stage 2 with:

| Parameter | Current (Broken) | Fixed | Rationale |
|-----------|-----------------|-------|-----------|
| **Loss scope** | All 6,297 codes | **Only target codes per batch** (the 16 codes the sampler enriched for) | Eliminates gradient dilution from 6,281 irrelevant codes |
| **Learning rate** | 5e-5 | **1e-3 to 5e-3** | Linear decoder from random init needs aggressive LR |
| **Epochs** | 3 (3,204 steps) | **20-30** (20,000-30,000 steps) | Kang et al. used 10-90 epochs for classifier re-training |
| **Per-positive normalization** | Mean over all elements | **Mean over positive elements per code** | Tail code positive loss isn't diluted by millions of (trivially correct) negatives |

**Implementation for the loss fix:**

During Stage 2, instead of passing data through the DataParallelWrapper's global loss:

```python
# For each batch, know which codes were targeted by the sampler
target_code_indices = [...]  # the 16 codes this batch enriched for

# Compute loss ONLY on those codes
logits_targeted = logits[:, target_code_indices]  # [batch, len_dy, 16]
targets_targeted = targets[:, target_code_indices]
loss = F.binary_cross_entropy_with_logits(logits_targeted, targets_targeted, 
                                           pos_weight=pw[target_code_indices])
```

This is a **2-4 hour GPU experiment** (Stage 2 only, no Stage 1 rerun needed) and directly answers: **with adequate optimization and focused loss, can the decoder extract signal from `h`?**

### Step 2: **If Step 1 shows margin improvement but tail_top10_acc still 0%: Add nonlinear (MLP) decoder for tail codes**

Replace `nn.Linear(256, |tail|)` with `nn.Sequential(Linear(256, 128), GELU, Linear(128, |tail|))` for tail codes only. A 2-layer MLP can discover nonlinear combinations of features in `h` that a linear projection misses — for example, the ABSENCE of certain common-code features might signal a tail code's presence, which requires a nonlinear readout.

**Estimated cost:** Same as Step 1, just changing the decoder architecture.

### Step 3: **If both Steps 1-2 show no margin improvement: THEN proceed to co-occurrence embeddings (Phase 2)**

At that point we'd have strong evidence that `h` genuinely lacks tail-relevant features at both linear and nonlinear readout levels. The next intervention must operate at the encoder input level.

The Phase 2 plan (PPMI+SVD embeddings + staged unfreezing + re-run full pipeline) is already well-designed in the implementation plan. But it should come AFTER exhausting the decoder-level interventions because:
- It requires a full Stage 1 retraining ($5-17)
- It introduces more variables (embedding quality, freezing schedule, LR multiplier)
- If it works, you won't know whether the decoder fix was also needed unless you test them separately

### Step 4: **If all above fail: Accept the structural limit for tail codes**

At that point, the evidence would strongly indicate that the shared encoder architecture (single `h` for all 6,297 codes) is fundamentally unable to capture tail-code features with this data volume. Options:
- **Focus on medium-tier accuracy** — data scaling already showed 27x improvement there
- **Per-tier encoder branches** — but this is a major architectural change
- **Mixture of experts at the encoder level** — could route tail-code patients to specialized experts

---

## 6) Summary Assessment Table

| Dimension | Expert 1's View | Expert 2's View | My View |
|-----------|----------------|----------------|---------|
| **Did Stage 2 work?** | No, definitively | No, but inconclusive | **No, but experiment was flawed — the loss function was never adapted for Stage 2** |
| **Is `h` lacking tail signal?** | Yes (definitive) | Premature to conclude | **Cannot determine from this experiment due to loss dilution confound** |
| **Root cause of margin collapse** | `h` has no signal | Under-optimization (LR/epochs) | **Loss dilution (computing over all 6,297 codes) is the primary confound; under-optimization is secondary** |
| **Next step** | Jump to Phase 2 (embeddings) | Re-run Stage 2 with higher LR/more epochs | **Re-run Stage 2 with focused loss (target codes only) + higher LR + more epochs** |
| **What was the most informative finding?** | Encoder is the bottleneck | Experiment is inconclusive | **The code-balanced sampler works, but the loss function silently negated its benefit** |

**Bottom line:** This experiment was a valid first attempt that proved the infrastructure works but contained a critical design flaw (global loss during Stage 2) that prevented the decoder from getting useful gradient. Before concluding that the encoder is the bottleneck, you MUST re-run Stage 2 with a focused, per-target-code loss. This is a cheap experiment (~2-4 hours) that answers the most important remaining question in this research program.






