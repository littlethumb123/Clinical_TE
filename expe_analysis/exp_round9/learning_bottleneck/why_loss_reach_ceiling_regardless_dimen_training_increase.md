# Expert 1: Root-Cause Analysis: Why the Loss Floor Is Invariant to Capacity and Data Scaling
- Mar 7, 2026
## FACTS — What the Data Shows

Before forming any hypotheses, let me separate the observable facts:

**Fact 1 — Smoothed final loss converges identically within each data tier:**
| Run | Params | Data | Train Loss (smoothed) | Val Loss | Delta from V3 |
|-----|--------|------|----------------------|----------|---------------|
| V3 | 25.3M | 1.5M | 0.00319 | 0.00322 | baseline |
| R7 | 58.6M | 1.5M | 0.00318 | 0.00316 | -0.3% / -1.9% |
| R6 | 25.3M | 5.7M | 0.00212 | 0.00205 | -33.5% / -36.3% |
| R8 | 58.6M | 5.7M | 0.00213 | 0.00204 | -33.2% / -36.6% |

Doubling parameters produces <0.5% change in the loss floor. Data scaling produces ~35% change. But within each data tier, the floor is identical regardless of capacity.

**Fact 2 — Loss trajectory shows the model reaches the floor very early:**
From the batch metrics data, at step ~4000 (out of 12,335 total for the 1.5M runs), the batch loss is already at ~0.003, and it stays there for the remaining ~8,000 steps. The model spends **65% of training oscillating around the floor**, not converging toward it.

**Fact 3 — The output head is a single linear projection:**

```2573:2576:dev/moe/moe_flashattn_4.py
        self.mm = nn.GELU()
        self.decoder_cd = nn.Linear(config.embedding_size, config.target_cd_cnt)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embedding_size)
```

The final output is `decoder_cd(dropout(norm(encoder_output)))` — a single linear layer mapping from `embedding_size` to `target_cd_cnt` (6,297 codes).

**Fact 4 — The loss is BCEWithLogitsLoss with `reduction='mean'` and pos_weight:**

```4765:4766:dev/moe/moe_flashattn_4.py
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_name = "BCEWithLogitsLoss"
```

The loss averages over ALL `valid_days × target_cd_cnt` elements. With ~6,297 codes and ~0.1–0.3% positive rate, this means **~99.8% of the loss elements are negative-class terms**.

**Fact 5 — Gradient starvation is confirmed and pos_weight-independent:**
From the expert analysis, by step 3,000:
- Common codes capture 85% of gradient budget
- Tail codes receive <1%
- This is IDENTICAL at pos_weight=35 and pos_weight=200

**Fact 6 — LR polishing test was run and REJECTED:**
Resuming from the plateau checkpoint with 10x lower LR for 2,000 steps produced WORSE val_loss (+0.45%), degraded NDCG (-0.36%), and zero change in rare/tail accuracy. The model is at a **sharp local minimum** — recall@10 dropped 4.4% at step 200 before slowly recovering to the same basin.

**Fact 7 — Tail embedding homogenization:**
Tail code embeddings have std=0.03 vs common std=0.27. All ~1,175 tail codes converge to a near-identical "default embedding" — the model has effectively collapsed them into one representation.

---

## THE ROOT CAUSE — A Multi-Layer Ceiling Mechanism

The loss floor is not one single ceiling. It is the convergence point of **five interlocking mechanisms** that create a stable equilibrium neither capacity nor data can break individually.

### Ceiling 1: The BCE `reduction='mean'` Information Dilution

This is the most fundamental and least discussed ceiling. Let me derive it precisely.

The loss is computed as:

```python
pred_loss = self.criterion(valid_output, valid_targets)
# valid_output: [N_valid_days, 6297]
# valid_targets: [N_valid_days, 6297]
```

PyTorch's `BCEWithLogitsLoss(reduction='mean')` computes:

**L = -(1 / (N × C)) × Σᵢ Σⱼ [wⱼ × yᵢⱼ × log(σ(zᵢⱼ)) + (1-yᵢⱼ) × log(1 - σ(zᵢⱼ))]**

where N = valid days, C = 6,297 codes, wⱼ = pos_weight for code j.

For a well-trained model:
- **Negative terms** (y=0, ~99.8% of elements): When the model predicts correctly (zᵢⱼ << 0), the term `-log(1-σ(z)) ≈ σ(z) ≈ e^z → 0`. These contribute near-zero to the total loss.
- **Positive terms** (y=1, ~0.2% of elements): Even with pos_weight amplification, each positive contributes `wⱼ × (-log(σ(zᵢⱼ)))`, which gets divided by **N × C ≈ N × 6297**.

The consequence: **the gradient signal from any single code j is diluted by a factor of 1/6,297**. For a tail code with frequency 0.001%, the per-step gradient contribution becomes:

```
∂L/∂wⱼ ∝ (pos_weight × frequency × error) / C
         = (200 × 0.00001 × 1.0) / 6297
         ≈ 3.2 × 10⁻⁷
```

For a common code:
```
∂L/∂wⱼ ∝ (1 × 0.01 × 0.1) / 6297
         ≈ 1.6 × 10⁻⁷
```

Even with 200x pos_weight, the rare code gets only **~2x** the gradient of a common code — because the 1000x frequency disadvantage overwhelms the 200x weight advantage. The `mean` reduction flattens the loss landscape such that **no code can dominate the loss signal enough to force the model to learn it**, once the common codes are well-predicted.

**This is why the loss floor is capacity-independent**: Adding more parameters doesn't change this arithmetic. The model already has sufficient capacity to drive the negative terms to zero and the common positive terms low. The remaining loss is the unavoidable residual from poorly-predicted positives (rare/tail codes), whose gradient signal is too diluted to drive further learning.

**Mathematical estimate of the floor:**
```
L_floor ≈ positive_rate × average_positive_loss
        ≈ 0.002 × (1.0 to 1.5)  (accounting for pos_weight averaging)
        ≈ 0.002 to 0.003
```

This matches the observed smoothed losses precisely: **0.00319 (V3), 0.00318 (R7), 0.00212 (R6), 0.00213 (R8)**.

The drop from 0.003 to 0.002 with more data occurs because medium-frequency codes (which benefit from more data) start getting predicted correctly, reducing the average positive loss. But tail codes remain unpredictable, setting the new floor at ~0.002.

### Ceiling 2: The Shared-Encoder Gradient Capture Equilibrium

The architecture funnels ALL codes through a shared encoder:

```2648:2675:dev/moe/moe_flashattn_4.py
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb
        cd = self.mm(cd)
        cd = self.norm(cd)
        cd = torch.swapaxes(cd, 0, 1)
        # Temporal encoding...
        for layer in self.temporal_layers:
            residual = cd
            cd_norm = layer['norm1'](cd)
            cd_attn = layer['attention'](cd_norm, is_causal=True)
            cd = residual + cd_attn
            residual = cd
            cd_norm = layer['norm2'](cd)
            cd_ffn = layer['ffn'](cd_norm)
            cd = residual + cd_ffn
        cd = torch.swapaxes(cd, 0, 1)
        cd = self.norm(cd)
        cd = self.dropout(cd)
        cd = self.decoder_cd(cd)
```

The encoder produces a single representation `h ∈ ℝ^d` per day, and the decoder maps this to 6,297 logits via `z = Wh + b`. Each code j has its own row `wⱼ` in `W`, but **all codes share the same `h`**.

This creates a zero-sum competition: the encoder representation `h` is shaped by gradients from all codes, but 85% of the gradient budget comes from common codes. This means:
- `h` is optimized to be maximally informative for common code prediction
- Rare code rows `wⱼ` in the decoder must work with whatever features `h` provides
- If the features `h` provides are not discriminative for rare codes, the decoder can't compensate

**Why 512d doesn't help**: Doubling the embedding dimension from 256 to 512 doubles the decoder capacity (more features per code), but the encoder STILL allocates those features to common code patterns. The additional 256 dimensions are learned to be informative for common codes, not rare ones, because that's where the gradient signal comes from.

The expert analysis confirmed this with the embedding homogenization finding: tail embedding std = 0.03 (all tail codes converge to the same "default" representation) while common embedding std = 0.27 (rich, differentiated embeddings). Adding 256 more dimensions just gives the common codes 256 more dimensions to differentiate, while tail codes remain homogenized in 512 dimensions just as they were in 256.

### Ceiling 3: The Gradient Starvation Temporal Lock-In

The expert evidence shows a critical temporal pattern:

| Phase | Steps | Common Grad % | Tail Grad % | What Happens |
|-------|-------|---------------|-------------|-------------|
| Init | 0-500 | ~17% | ~18% | Random init, balanced |
| Transition | 500-3000 | 17%→67% | 18%→3% | Common captures gradient |
| Terminal | 3000+ | 85%+ | <0.1% | Locked equilibrium |

This transition is **pos_weight-independent** (identical at 35 and 200), meaning it's a property of the training dynamics, not the loss weighting.

**The mechanism**: Once common codes are partially learned (steps 500-1500), their per-sample error decreases. But because they appear in EVERY batch, the aggregate gradient from common codes (many samples × small error) still exceeds the gradient from tail codes (rare samples × high error × pos_weight). The model enters a self-reinforcing loop:
1. Common codes improve → their error drops → but aggregate gradient stays high (many samples)
2. Rare codes don't improve → their error stays high → but aggregate gradient stays low (few samples)
3. Encoder representation shifts toward common-code features
4. Rare code gradients become "noise" relative to common code refinement signal

**Why this is capacity-independent**: The gradient capture happens at the batch/step level, not the parameter level. A 512d model has more parameters, but those parameters receive the SAME gradient distribution (85% common) as a 256d model. The extra capacity has no independent information source to learn rare-code features from.

**Why this is (mostly) data-independent**: With 3.6x more data (5.7M), each code sees proportionally more examples, but the RELATIVE occurrence ratio remains the same (13.4x common:tail ratio). The gradient starvation mechanism depends on relative frequencies, not absolute counts. The exception: medium codes (25th-50th percentile) DO benefit from more data because their absolute count crosses a threshold where they appear often enough per batch to maintain gradient signal. This explains the medium_top10_acc jump from 0.16% to 4.26%.

### Ceiling 4: The 1-Epoch Training Constraint

All experiments train for exactly 1 epoch. For the 512d model on 5.7M data:
- 58.6M parameters see 5.7M samples → **0.097 samples per parameter**
- Industry norms suggest 10-100x more data per parameter for good convergence

The 1-epoch constraint means:
1. Each training sample is seen exactly once — no opportunity for the model to refine predictions on previously-seen difficult cases
2. The LR schedule is calibrated for a single pass — the "polishing" phase (last 40% of steps at decaying LR) cannot revisit early-training patterns
3. Rare codes may be seen 0-1 times during the polishing phase, providing essentially zero late-stage refinement

For the 256d model, the ratio is better (0.225 samples/param), but still far below the threshold where capacity begins to show returns.

**This explains why 512d ≈ 256d**: The larger model is severely undertrained. It cannot exploit its additional capacity within 1 epoch because it doesn't see enough diverse examples to distinguish common from rare patterns in the higher-dimensional feature space.

### Ceiling 5: The LR Schedule + Gradient Dynamics Interaction

The schedule is:
- 15% warmup → 45% plateau at peak LR → 40% linear decay to 20% of peak

For the 5.7M runs (44,546 steps):
- Warmup: steps 0-6,682 (LR ramping up)
- Plateau: steps 6,682-26,727 (LR at peak)
- Decay: steps 26,727-44,546 (LR decaying to 20% of peak)

The gradient starvation transition completes at step ~3,000 — during the **warmup phase**. This means:
1. The model locks into the common-code gradient equilibrium before the LR even reaches peak
2. By the time peak LR is reached, the representation is already shaped for common codes
3. The plateau phase (steps 6,682-26,727) spends 20,000 steps at high LR refining common codes
4. The decay phase can only polish what's already learned — it can't undo the gradient capture

The polishing test confirmed this: even 10x lower LR couldn't escape the local minimum because the basin is determined by the representation geometry (common-code features), not the step size.

---

## WHY NEITHER CAPACITY NOR DATA BREAKS THROUGH

### Why 512d ≈ 256d (Capacity Doesn't Help)

The 5 ceilings form a chain:
1. **Mean reduction** makes the loss floor a function of positive rate and average positive error
2. **Shared encoder** ensures all additional capacity serves common codes (85% gradient)
3. **Gradient starvation** prevents rare codes from claiming any of the new capacity
4. **1-epoch training** gives insufficient samples per parameter for the larger model
5. **LR schedule** cements the gradient equilibrium before peak learning even begins

Adding 256 more embedding dimensions adds capacity that is:
- Shaped by common-code gradients → learns common-code features
- Never exposed to diverse rare-code examples → cannot specialize for rare patterns
- Converges to the SAME loss floor because the same fraction of codes (rare/tail) are unpredictable

### Why 5.7M → Only Modest Improvement (Data Partially Helps)

More data DOES reduce the loss floor (0.003 → 0.002, ~35% drop), but this is entirely explained by:
- Medium codes crossing the gradient visibility threshold (medium_top10_acc: 0.16% → 4.26%)
- More diverse common-code patterns being learned

More data does NOT fix:
- The relative occurrence imbalance (tail still at 5.2% of occurrences)
- The gradient starvation mechanism (structural, ratio-dependent)
- The BCE mean reduction dilution (same formula, same sparsity rate)

### The Compounding Effect

These ceilings compound multiplicatively, not additively:

```
Achievable loss reduction ∝ (fraction of gradient to rare codes)
                          × (LR during polishing)  
                          × (epochs of training)
                          × (representation quality for rare codes)

= (0.01)     ← gradient starvation: 1% to tail
× (0.2)     ← min_lr_ratio: 20% of peak
× (1)       ← single epoch
× (0.03/0.27) ← embedding homogenization: tail/common std ratio

≈ 0.00022   ← virtually zero
```

This means the model has essentially **zero effective learning capacity** for rare/tail codes after the first few thousand steps. Neither 2.3x more parameters nor 3.6x more data changes any of these factors enough to break the compound product above zero.

---

## WHAT WOULD ACTUALLY BREAK THROUGH THE FLOOR

Based on this mechanistic analysis, breaking the loss floor requires addressing at least one of the compounding factors:

1. **Replace `reduction='mean'` with per-tier loss balancing**: This directly attacks Ceiling 1 by ensuring tail codes contribute meaningfully to the total loss regardless of their frequency. Instead of averaging over all 6,297 codes equally, partition codes into tiers and weight each tier's loss contribution equally.

2. **Multi-epoch training for the 512d model**: This addresses Ceiling 4. With 3-5 epochs, the 512d model would see each sample multiple times, giving it the opportunity to use its additional capacity for rare-code refinement. The smoothed loss floor should diverge from the 256d model after epoch 2-3 if capacity is truly available.

3. **Decouple the encoder and decoder (two-stage training)**: This addresses Ceiling 2. Train the encoder with instance-balanced sampling (Stage 1), then fine-tune the decoder with class-balanced sampling (Stage 2). This prevents common codes from monopolizing the shared representation.

4. **Use per-code or per-tier decoders**: Instead of a single `nn.Linear(d, 6297)`, use separate decoder heads for common, medium, rare, and tail codes. This breaks the shared-representation bottleneck by giving each tier its own feature extraction pathway from the encoder.

5. **Replace BCE with focal loss or ASL**: This addresses the "well-calibrated common codes contributing zero gradient but still dominating" problem by dynamically down-weighting easy examples, freeing gradient budget for hard cases.

---

## (F) Executive Summary

The loss floor at ~0.003 (1.5M) / ~0.002 (5.7M) is **not a single ceiling but a stable equilibrium** created by five interlocking mechanisms:

1. **BCE mean reduction** dilutes rare-code gradient signal by 1/6297, making pos_weight ineffective
2. **Shared encoder** allocates all representational capacity to common codes (85% of gradient)
3. **Gradient starvation** locks in by step 3000, before the LR schedule even peaks
4. **1-epoch training** denies the larger model enough passes to utilize extra capacity
5. **LR schedule** cements the equilibrium during warmup, then cannot undo it

These five factors multiply together to produce **effectively zero learning signal** for rare/tail codes. This is why 2.3x more parameters and 3.6x more data both converge to the same loss floor within each data tier — they address none of the compounding factors that create the equilibrium.

The path forward requires breaking at least one link in this chain: per-tier loss decomposition (Ceiling 1), multi-epoch training (Ceiling 4), or decoupled encoder/decoder training (Ceiling 2). The previous expert analysis already confirmed that LR schedule changes alone (Ceiling 5) cannot escape the basin.


# Expert 2 Independent Analysis: Why the Loss Floor Is Invariant to Capacity and Data Scaling

## Part 1: My Independent Diagnosis (From First Principles)

### Establishing the Observable Facts

Before forming any hypothesis, here is what the data unambiguously shows:

| Observable | V3 (256d, 1.5M) | R7 (512d, 1.5M) | R6 (256d, 5.7M) | R8 (512d, 5.7M) |
|---|---|---|---|---|
| Smoothed final train loss | 0.00319 | 0.00318 | 0.00212 | 0.00213 |
| Val loss | 0.00322 | 0.00316 | 0.00205 | 0.00204 |
| Loss reaches floor at step | ~4000 | ~4000 | ~15000 | ~15000 |
| % of training at floor | ~68% | ~68% | ~66% | ~66% |
| Tail positive logit | -14.69 | N/A | N/A | N/A |
| Tail embedding std | 0.03 | N/A | N/A | N/A |
| Common embedding std | 0.27 | N/A | N/A | N/A |
| Gradient common frac (terminal) | 85%+ | N/A | N/A | N/A |
| Gradient tail frac (terminal) | <0.1% | N/A | N/A | N/A |

### The Core Insight: This Is an Information Bottleneck, Not a Capacity Problem

Let me reason from the architecture forward.

Your model has this data flow:

```
Input codes → Embedding → Daily Pooling → Temporal Encoder → h ∈ ℝ^d → Linear(d, 6297) → logits
```

The decoder is `self.decoder_cd = nn.Linear(config.embedding_size, config.target_cd_cnt)`, which maps a single vector **h** to 6,297 logits via `z_j = w_j^T h + b_j`.

**Every code must extract its predictive signal from the same vector h.** Code j can only be predicted correctly if h contains features that correlate with code j's occurrence. The decoder row `w_j` is just a linear readout — it cannot create information that isn't in h.

**The question then becomes:** When you go from 256d to 512d, does h gain any new information relevant to rare/tail codes?

The answer is **no**, and here is the mechanistic reason:

The encoder's parameters are updated by backpropagation through all 6,297 code losses. The gradient that shapes the encoder is:

```
∂L/∂θ_encoder = Σ_j (∂L/∂z_j × ∂z_j/∂h × ∂h/∂θ_encoder)
```

Each code j contributes a gradient signal proportional to:
- **Its loss magnitude** (how wrong is the prediction?)
- **Its occurrence frequency** (how often does it appear in the batch?)
- **The pos_weight** (static amplification)

For common codes: `occurrence × small_error × weight_1 = large_aggregate`
For tail codes: `rare_occurrence × large_error × weight_200 = small_aggregate`

The empirical evidence confirms this: by step 3000, common codes capture 85% of `∂L/∂θ_encoder`. This means **85% of the encoder parameter updates serve common-code features**.

When you double the embedding dimension from 256 to 512:
- The encoder gains 256 more dimensions to populate
- But 85% of the gradient signal says "make these dimensions useful for common codes"
- So the extra 256 dimensions learn common-code features, not rare-code features
- The decoder gains 256 more features per code row, but for rare codes, those features are no more informative than the first 256

**This is why smoothed loss is 0.00318 vs 0.00319 — the extra dimensions add no new information for the codes that the model currently fails on.**

This is a single, clean mechanism: **gradient-driven encoder monopolization**. The encoder is the bottleneck, and the encoder is controlled by the gradient distribution, which is controlled by occurrence frequency, which is unchanged by model capacity.

### Why More Data Helps (But Only Partially)

More data (1.5M → 5.7M) DOES reduce the floor from ~0.003 to ~0.002. This is because:

1. **Medium-frequency codes cross a visibility threshold.** With 3.6x more data, medium codes appear ~3.6x more often per epoch. Some codes that appeared too infrequently to maintain gradient signal at 1.5M now appear consistently enough at 5.7M. Evidence: medium_top10_acc jumps from 0.16% to 4.26%.

2. **Common codes become even better calibrated.** More diverse common-code patterns → lower residual error on common codes → their contribution to total loss decreases.

3. **But the relative frequency ratio is preserved.** If tail codes were 5.2% of occurrences at 1.5M, they're still ~5.2% at 5.7M. The gradient starvation mechanism depends on the ratio, not the count.

This explains the precise observation: data reduces the floor (because it moves some codes from "can't learn" to "can learn") but doesn't make 512d better than 256d (because the same gradient dynamics apply regardless of dimension within each data tier).

### The Bayes Optimal Floor

There is a theoretical information-theoretic lower bound on the loss: the **Bayes optimal loss**. This is the loss achieved by a perfect predictor that knows the true conditional probability P(code_j | patient_history).

For the mean-reduced BCE loss:

```
L_bayes = -(1/(N×C)) × Σᵢ Σⱼ [wⱼ × P_true × log(P_true) + (1-P_true) × log(1-P_true)]
```

This depends entirely on the data distribution, not the model. It sets an absolute floor that no model can beat.

**Critical question: Is the current model near the Bayes optimal loss?**

I believe it is near-optimal for common codes but **far from optimal for rare/tail codes**. The evidence:
- Tail logit = -14.69 when y=1 (P ≈ 0.00004%). If the true P for a tail code given the right patient context is, say, 5%, then the model is predicting 0.00004% instead of 5% — a massive miscalibration.
- But because tail positives contribute only a tiny fraction of total loss elements, this miscalibration barely registers in the overall loss.

**The loss floor is NOT the Bayes optimal loss. It is the loss achieved by a model that is Bayes-optimal for common codes and maximally ignorant for tail codes.** The overall loss is dominated by common codes (where the model performs well), masking the tail code failure.

### The "Sharp Local Minimum" — What It Really Means

The LR polishing test shows the model is at a point where perturbation causes performance to drop before recovering. Let me be precise about the geometry:

The parameter space has two relevant subspaces:
1. **Common-code subspace**: The directions that affect common code predictions. The model has converged to a good solution here — the loss is low and the landscape is steep (sharp minimum).
2. **Rare-code subspace**: The directions that would affect rare code predictions. The model has NOT converged here — the relevant parameters (rare decoder rows, encoder features for rare codes) are at near-random values, and the loss landscape is FLAT (no gradient signal to guide updates).

This is better described as a **partial convergence** rather than a "sharp local minimum." The model has found a good solution in one subspace but hasn't explored the other subspace at all. Reducing LR can't help because there's no gradient signal in the rare-code subspace to follow, regardless of step size.

---

## Part 2: Critical Review of the Expert's Analysis

### Overall Assessment

The expert's analysis in `why_loss_reach_ceiling_regardless_dimen_training_increase.md` is **largely correct in diagnosis but flawed in structural presentation and contains specific analytical errors**. Let me evaluate each ceiling.

### Ceiling 1: BCE `reduction='mean'` Information Dilution

**I PARTIALLY AGREE but identify an analytical error.**

The expert's core claim is correct: the mean reduction dilutes per-code gradient signal by 1/6297. However, the expert's numerical example contains a conceptual mistake:

The expert computes:
```
Rare code:   (200 × 0.00001 × 1.0) / 6297 ≈ 3.2 × 10⁻⁷
Common code: (1 × 0.01 × 0.1) / 6297 ≈ 1.6 × 10⁻⁷
```

And concludes rare codes get "only ~2x the gradient of a common code."

**This is misleading.** This computes the gradient for a single code on a single sample. The actual gradient that reaches the encoder in a batch of 128 members × ~50 valid days = ~6,400 samples is:

```
Rare code aggregate:   ~0-1 positive samples × (200 × 1.0) / 6297 ≈ 0 to 0.032
Common code aggregate: ~50+ positive samples × (1 × 0.1) / 6297 ≈ 0.0008 × 50 = 0.04
```

The real problem isn't the per-sample ratio (which pos_weight DOES fix to ~2x) — it's the **number of samples per batch** that contribute a gradient for each code. Common codes appear in every batch; tail codes appear sporadically. The 1/6297 division is a constant that cancels out when comparing codes — it affects both equally.

The expert's mathematical estimate of the floor (`positive_rate × average_positive_loss ≈ 0.002-0.003`) is a useful approximation that agrees with observation, and I accept this as correct. But the mechanism described ("mean reduction makes pos_weight ineffective") conflates two distinct effects: the per-element averaging and the occurrence frequency imbalance.

**My verdict: The mean reduction is a contributing factor, not a root cause. The root cause is occurrence-level imbalance amplified by batch-level gradient aggregation.**

### Ceiling 2: Shared-Encoder Gradient Capture Equilibrium

**I STRONGLY AGREE. This is the primary ceiling, not ceiling #2.**

The expert correctly identifies that the shared encoder representation `h` is a zero-sum information space monopolized by common-code gradients. The statement:

> "Adding 256 more dimensions just gives the common codes 256 more dimensions to differentiate, while tail codes remain homogenized in 512 dimensions just as they were in 256."

This is precisely correct. I would elevate this to ceiling #1. The shared encoder is the fundamental architectural bottleneck. Everything else (BCE mean reduction, gradient starvation dynamics, LR schedule) is either a mechanism that creates the monopolization or a symptom of it.

### Ceiling 3: Gradient Starvation Temporal Lock-In

**I AGREE with the observation but DISAGREE with one claim.**

The temporal pattern (17% → 85% common gradient share over steps 0-3000) is well-documented and the expert's description is accurate.

However, the expert claims this is "pos_weight-independent," which I find **misleading**. The terminal distribution (85% vs 85%) IS nearly identical at pw=35 and pw=200, but this is because:
- pos_weight scales per-sample gradients linearly
- Occurrence frequency provides a multiplicative advantage that overwhelms linear scaling
- At pw=200: rare code gradient ∝ 200 × frequency_rare. At pw=35: ∝ 35 × frequency_rare. But frequency_common >> frequency_rare regardless.

The correct statement is: "pos_weight is insufficient to overcome the occurrence frequency imbalance," not "pos_weight has no effect." The distinction matters because it points to the real intervention: change the occurrence frequency (via sampling), not the per-sample weight.

### Ceiling 4: 1-Epoch Training Constraint

**I PARTIALLY DISAGREE — the framing is misleading.**

The expert uses the "0.097 samples per parameter" ratio and compares it to "industry norms of 10-100x." This comparison is inappropriate:

1. **The industry norm applies to language model pretraining** where the model must learn a distribution over natural language tokens. Clinical code prediction is a fundamentally different task — the input vocabulary is smaller (~10K codes vs ~50K+ BPE tokens), the sequences are shorter (~200 days vs ~2048+ tokens), and the output space is structured (multi-label, not autoregressive).

2. **The 256d model with 0.225 samples/param achieves identical performance to 512d with 0.097 samples/param.** If sample-per-parameter ratio were the bottleneck, the 256d model should outperform (it doesn't). This disproves the "underfitting due to insufficient data per parameter" hypothesis.

3. **The actual evidence for 1-epoch being a limitation is different**: each rare code appears 0-1 times during the late-training polishing phase, providing essentially zero refinement. Multi-epoch training might help by giving rare codes more late-phase exposure. But this is a sampling/exposure problem, not a data-per-parameter problem.

**My verdict: 1-epoch training is a contributing factor, but for the right reason (insufficient rare code exposure during polishing) rather than the stated reason (insufficient tokens per parameter).**

### Ceiling 5: LR Schedule + Gradient Dynamics Interaction

**I DISAGREE that this should be included as a ceiling.**

The LR polishing test definitively REJECTED the schedule hypothesis. The expert acknowledges this (citing the polishing test results) but still includes it as one of the "five interlocking mechanisms." This is contradictory.

The expert writes:
> "The gradient starvation transition completes at step ~3,000 — during the warmup phase"

This is an interesting observation about timing, but it doesn't make the LR schedule a *cause* of the plateau. The gradient starvation would occur regardless of whether the warmup is 15% or 5% or 30%, because the starvation is driven by occurrence frequency, not learning rate magnitude.

**My verdict: The LR schedule is NOT a ceiling. It's a non-factor that has been experimentally rejected. Including it weakens the analysis by inflating the number of claimed mechanisms without evidence.**

### The Compounding Effect Calculation

**I STRONGLY DISAGREE with this calculation.**

The expert presents:
```
Achievable loss reduction ∝ (0.01) × (0.2) × (1) × (0.03/0.27) ≈ 0.00022
```

This multiplication of:
- gradient fraction (dimensionless ratio)
- LR ratio (dimensionless ratio)
- epoch count (integer)
- embedding std ratio (dimensionless ratio)

These are four incommensurable quantities whose product has no physical meaning. The "≈ 0.00022" result creates a false sense of quantitative precision. The actual relationship between these factors and "achievable loss reduction" is highly nonlinear, context-dependent, and not representable as a simple product.

A more honest statement: "Multiple factors combine to make tail code learning extremely difficult, and none of them individually are sufficient to explain the plateau."

### What the Expert Gets RIGHT (and Should Be Emphasized More)

Despite my criticisms, the expert's analysis makes several contributions I want to highlight as correct and important:

1. **The loss floor estimate** (`positive_rate × average_positive_loss`) is a clean, testable prediction that matches observation.

2. **The shared encoder as bottleneck** is the most important structural insight. The single linear decoder from a shared representation is the architecture design that makes gradient monopolization unavoidable.

3. **The intervention recommendations** (per-tier loss balancing, multi-epoch training, decoupled encoder/decoder) are well-targeted at the actual mechanisms. I would specifically endorse:
   - **Per-tier loss decomposition** (strongest intervention for breaking ceiling 2)
   - **Separate decoder heads per tier** (directly addresses shared-representation bottleneck)
   - **Multi-epoch training** (but for the right reason: rare code re-exposure, not tokens-per-param)

---

## Part 3: My Unified Root Cause Model

After independent analysis, I propose a simpler, more mechanistically precise model than the expert's 5-ceiling framework:

### One Root Cause, Three Contributing Factors

**Root Cause: Gradient-Driven Encoder Monopolization**

The shared encoder representation `h ∈ ℝ^d` is shaped by gradients that are overwhelmingly dominated by common codes (85%+ of total gradient norm by step 3000). This creates an information bottleneck: `h` contains features useful for common codes but not for rare/tail codes. The decoder for rare codes receives an uninformative input, regardless of its own capacity or the dimension of h.

This root cause is **sufficient** to explain why:
- 256d = 512d (extra dimensions learn common features, not rare features)
- More data helps modestly (shifts some medium codes above the gradient visibility threshold)
- LR changes don't help (no gradient signal to follow, regardless of step size)
- pos_weight doesn't help (per-sample amplification can't overcome batch-level aggregation)

**Contributing Factor 1: Occurrence-Level Frequency Imbalance** (structural)

Tail codes at 5.2% of occurrences vs 69.7% for common codes (13.4x ratio). This is the DATA property that makes gradient monopolization inevitable under standard training.

**Contributing Factor 2: BCE Mean Reduction** (amplifier)

Averaging over N × 6297 elements makes each code's gradient contribution inversely proportional to the total label space. This amplifies the frequency imbalance but is not the root cause.

**Contributing Factor 3: Single-Pass Training** (amplifier)

One epoch means rare codes get zero revisits during the polishing phase (decay portion of LR schedule), denying the model any opportunity for late-stage refinement of rare predictions.

### What the Expert's 5-Ceiling Model Gets Wrong

The expert presents five "interlocking ceilings" as though they are five independent mechanisms. In reality:
- **Ceilings 2 and 3 are the same mechanism** (gradient-driven encoder monopolization IS gradient starvation)
- **Ceiling 5 has been experimentally rejected** and should not be included
- **Ceiling 1 is a secondary amplifier**, not an independent ceiling
- **Ceiling 4 is a valid contributing factor** but framed with wrong reasoning (tokens-per-param)

The compound multiplication `0.01 × 0.2 × 1 × (0.03/0.27)` is a mathematical artifact that doesn't correspond to the actual nonlinear dynamics.

A simpler, more accurate model: **One mechanism (encoder monopolization) + occurrence imbalance (structural cause) + BCE mean averaging (amplifier) + single-epoch (amplifier).**

### What Would Actually Break Through

In order of expected impact:

1. **Per-tier loss decomposition**: Compute loss separately for common/medium/rare/tail code subsets and weight them equally. This directly breaks encoder monopolization by forcing the encoder to allocate representational capacity to all tiers. This attacks the root cause.

2. **Tier-specific decoder heads**: Replace the single `nn.Linear(d, 6297)` with separate decoders per tier, each with its own mini-MLP. This allows each tier to extract different features from the encoder output, partially compensating for encoder monopolization.

3. **Multi-epoch training (2-3 epochs)**: This gives rare codes additional exposure during the decay phase. BUT — this will only help if combined with (1) or (2), because without changing the gradient distribution, multiple epochs just repeat the same common-code-dominated optimization.

4. **Density-based sampling**: Oversample patient-days with tail code targets, changing the per-batch occurrence ratio. This is the most direct attack on the structural cause (frequency imbalance).

---

## Part 4: Summary of Agreement and Disagreement

| Expert's Claim | My Position | Reasoning |
|---|---|---|
| 5 interlocking ceilings | **Disagree (overcomplication)** | 1 root cause + 3 contributing factors is more parsimonious and mechanistically precise |
| BCE mean reduction is "most fundamental" | **Disagree** | Shared encoder monopolization is more fundamental; BCE mean is an amplifier |
| Shared encoder is a zero-sum competition | **Strongly agree** | This IS the root cause; should be ceiling #1, not #2 |
| Gradient starvation is pos_weight-independent | **Agree with observation, disagree with framing** | More precise: pos_weight addresses wrong level (per-sample vs per-batch) |
| 1-epoch = insufficient tokens-per-param | **Disagree with reasoning** | The issue is insufficient rare-code revisits, not an LLM-style data scaling law |
| LR schedule is a ceiling | **Disagree** | Experimentally rejected by polishing test; should not be included |
| Compound product ≈ 0.00022 | **Strongly disagree** | Incommensurable quantities; no physical meaning; false precision |
| Loss floor ≈ positive_rate × avg_positive_loss | **Agree** | Clean prediction, matches observation |
| Interventions (per-tier loss, separate decoders, multi-epoch) | **Agree** | Well-targeted at root cause |
| The floor is capacity-independent | **Agree on conclusion, disagree on mechanism** | Not 5 ceilings but 1: encoder monopolization makes extra capacity irrelevant |

### The Bottom Line

The loss floor's invariance to capacity is explained by a single mechanism: **the shared encoder allocates representational capacity in proportion to gradient signal, and gradient signal is proportional to code occurrence frequency**. Adding more dimensions expands capacity that gets monopolized by common codes. Adding more data shifts some medium codes above the visibility threshold but doesn't change relative frequencies. The loss floor is not 5 interlocking ceilings — it is one information bottleneck (the encoder) shaped by one data property (occurrence imbalance), amplified by two training choices (BCE mean reduction and single-epoch).

The expert's analysis is directionally correct but architecturally overengineered. The intervention priorities are sound regardless: per-tier loss balancing and density-based sampling are the highest-leverage changes because they directly attack the root cause (gradient distribution → encoder representation → loss floor).

# Expert 3 Independent Expert Analysis: Why the Training Loss Floor Is Invariant to Capacity and Data Scaling

## Part 1: My Independent Diagnosis (From First Principles)

### 1.1 Precisely Stating the Observable Facts

Before I form ANY hypothesis, let me lay out what the data unambiguously shows, quoting specific values:

**Fact 1 — The loss floor is capacity-invariant within each data tier:**

| Run | Params | Data | Smoothed Train Loss | Val Loss |
|-----|--------|------|---------------------|----------|
| V3 (256d) | 25.3M | 1.5M | 0.00319 | 0.00322 |
| R7 (512d) | 58.6M | 1.5M | 0.00318 | 0.00316 |
| R6 (256d) | 25.3M | 5.7M | 0.00212 | 0.00205 |
| R8 (512d) | 58.6M | 5.7M | 0.00213 | 0.00204 |

The 256d/512d delta is <0.5% within each data tier. The 1.5M/5.7M data tier shift is ~33%.

**Fact 2 — The architecture channels ALL code predictions through a single shared representation:**

```2668:2673:dev/moe/moe_flashattn_4.py
        cd = torch.swapaxes(cd, 0, 1)
        
        cd = self.norm(cd)
        cd = self.dropout(cd)
        cd = self.decoder_cd(cd)
```

Where `decoder_cd = nn.Linear(embedding_size, 6297)` — a single linear projection from a shared `h ∈ ℝ^d` to 6,297 logits.

**Fact 3 — The loss uses `BCEWithLogitsLoss` with `reduction='mean'` over `[N_valid_days × 6,297]` elements.**

With ~0.04% average positive rate per code per day, **~99.96% of loss elements are negative-class terms**.

**Fact 4 — Gradient starvation is empirically confirmed and pos_weight-invariant:**

From gradient tier tracking (Jan 24 analysis):
- Step 1: common=17.8%, tail=17.8% (balanced at init)
- Step 3000: common=66.7%, tail=3.0% (transition)
- Step 12001: common=85.3%, tail=0.1% (terminal)
- Identical distribution at pos_weight=35 and pos_weight=200

**Fact 5 — The LR polishing test is DEFINITIVELY negative:**

Resuming from plateau with 10x lower LR (4e-6) for 2000 steps:
- val_loss: 0.00336 → 0.00338 (+0.45%, WORSE)
- recall@10: 0.8246 → 0.7883 at step 200 (DROPPED 4.4%), recovered to 0.8213 (still below baseline)
- rare_top10_acc: 0% → 0%, tail_top10_acc: 0% → 0% (UNCHANGED)
- Diagnosis field: `"STRUCTURAL_BOTTLENECK"`

**Fact 6 — Tail code INPUT EMBEDDINGS are homogenized:**

| Tier | Embedding Mean Norm | Embedding Std |
|------|---------------------|---------------|
| common | 1.42 | **0.27** |
| medium | 1.49 | **0.15** |
| rare | 1.41 | **0.05** |
| tail | 1.46 | **0.03** |

No collapse (norms are healthy), but severe **homogenization** — all ~1,175 tail codes have nearly identical learned embeddings.

**Fact 7 — Tail logits show extreme negative prior AND cross-code interference:**

| Tier | Mean Logit (y=1) | σ(logit) | Margin (pos vs neg) |
|------|------------------|----------|---------------------|
| common | -2.26 | 9.4% | 6.44 |
| medium | -6.39 | 0.17% | 6.23 |
| rare | -9.68 | 0.006% | 5.34 |
| tail | -14.69 | 0.00004% | **1.76** |

The tail margin of 1.76 (positive vs negative) proves the model CAN weakly distinguish tail positives from negatives — but the logit is so deeply negative that tail codes never surface in top-K.

### 1.2 My Root Cause Analysis (From Architecture + Optimization Theory)

I identify **one root cause** and **three structural amplifiers**. The root cause is sufficient to explain all observations. The amplifiers explain why the root cause is so difficult to overcome.

#### ROOT CAUSE: Representation Monopolization Through Gradient-Frequency Coupling

The architecture funnels all predictions through a shared encoder representation `h ∈ ℝ^d`:

```
Input codes → Embedding(cd_cnt, d) → Daily Pooling → Temporal Encoder → h ∈ ℝ^d → Linear(d, 6297) → logits
```

The encoder parameters θ_enc are updated by:

```
∂L/∂θ_enc = Σⱼ (∂L/∂zⱼ) × (wⱼ^T) × (∂h/∂θ_enc)
```

where the per-code gradient weight `∂L/∂zⱼ ∝ frequency(j) × pos_weight(j) × error(j)`.

At equilibrium (step 3000+), the product `frequency × pos_weight × error` for common codes **exceeds** that for tail codes despite 200x pos_weight amplification. This is because:
- For tail: 0.00001 freq × 200 weight × ~1.0 error = 0.002
- For common: 0.01 freq × 1 weight × ~0.1 error × (many samples per batch) = aggregate ~0.04

The aggregate per-batch gradient from common codes is ~20x that of tail codes (even with pos_weight), because common codes appear in **every batch** while tail codes appear sporadically.

This creates a self-reinforcing dynamic:
1. Common codes dominate gradients → h learns common-code features
2. h is informative for common codes → common errors decrease → but aggregate gradient stays high (many samples × small error)
3. h is uninformative for tail codes → tail errors stay high → but aggregate gradient stays low (rare samples × high error)
4. The representation locks in: h is a "common-code feature extractor"

**Why 512d cannot break this**: Adding 256 more dimensions to h gives the encoder 256 more features to learn. But 85% of the gradient signal says "make these features useful for common codes." So the extra dimensions learn common-code features. The decoder for tail codes gains 256 more features to read — but those features carry no tail-relevant information.

**Why this is the ROOT cause, not just a contributing factor**: This single mechanism explains:
- Why 256d ≈ 512d (extra capacity monopolized by common codes)
- Why more data helps modestly (some medium codes cross the gradient visibility threshold)
- Why LR changes don't help (no gradient signal in the tail subspace regardless of step size)
- Why pos_weight is ineffective (per-sample amplification can't overcome batch-level aggregation)
- Why tail embeddings homogenize (they receive uniform gradients in the h direction)

#### AMPLIFIER 1: Input Embedding Feedback Loop (Not Sufficiently Discussed by Either Expert)

Here is something I want to emphasize that both experts underweight:

The gradient starvation doesn't just affect the encoder and decoder — it affects the **input embeddings** (`self.embedding_cd`) at the very first layer. The tail code embedding at row j of the embedding table receives gradient:

```
∂L/∂eⱼ = Σ_{days containing code j} (∂L/∂input) × (∂input/∂eⱼ)
```

For a tail code appearing in 0.001% of days, this sum has extremely few terms. Each term's upstream gradient `∂L/∂input` is itself dominated by common-code objectives (because the encoder has already been monopolized). So the few gradient updates that tail code embeddings receive all point in similar directions — directions set by common code optimization, not by tail code learning.

This creates a **vicious cycle at the input level**:
1. Tail code input embeddings are near-identical (std=0.03) → no distinctive signal enters the encoder
2. Encoder receives no distinctive signal for tail codes → produces h that is uninformative for tail codes
3. Decoder for tail codes receives uninformative h → predictions are wrong → but sparse gradient can't fix embeddings
4. Return to step 1

**This is fundamentally different from the encoder monopolization**: even if you "fixed" the encoder by forcing it to allocate capacity to tail codes, the input embeddings for tail codes are still homogenized. The model literally cannot distinguish "day with tail code A" from "day with tail code B" at the input level because e_A ≈ e_B.

This makes the problem harder than either expert suggests: solving the encoder gradient distribution is necessary but insufficient — you also need to break the input embedding homogenization.

#### AMPLIFIER 2: Cross-Code Interference Through Shared Representation

Expert 1 discusses the shared encoder as a "zero-sum competition." Expert 2 discusses it as "encoder monopolization." Both are correct but miss a subtler effect: **the shared representation is not just uninformative for tail codes — it actively suppresses them**.

The decoder bias bⱼ for a tail code should equilibrate at:

```
b_j = log(pos_weight_j × frequency_j / (1 - frequency_j))
```

For a tail code with frequency 0.00001 and pos_weight 200:

```
b_j ≈ log(200 × 0.00001) = log(0.002) ≈ -6.2
```

But the observed tail logit when y=1 is **-14.69**, which is ~8.5 units MORE negative than the bias equilibrium. This excess suppression comes from the `wⱼ^T h` term in the decoder:

```
z_j = w_j^T h + b_j = (w_j^T h) + (-6.2 theoretical)
→ w_j^T h ≈ -14.69 - (-6.2) ≈ -8.5
```

The decoder weight vector `wⱼ` for tail codes has developed a negative correlation with the common-code features in h. This happens because:
- h is dominated by common-code features (encoder monopolization)
- During the few gradient updates tail codes receive, the decoder row wⱼ is pushed to have positive correlation with "patient history features" that correlate with tail code occurrence
- But these "patient history features" are encoded through the common-code lens of h, creating spurious correlations that fail to generalize
- Over training, the net effect is wⱼ^T h ≈ -8.5 for typical days, actively pushing tail logits far below what the prior alone would predict

**This cross-code interference means adding more dimensions makes the problem WORSE, not better**: more dimensions mean more common-code features in h, which means more negative dot products with poorly-trained tail decoder rows.

#### AMPLIFIER 3: Single-Epoch Rare Code Deprivation

Both experts discuss the 1-epoch constraint. Expert 1 frames it incorrectly (tokens-per-parameter ratio). Expert 2 frames it more precisely (insufficient rare-code revisits during polishing).

My framing: **The 1-epoch constraint means rare codes experience the critical LR decay phase with essentially zero exposure.**

With the linear schedule (15% warmup → 45% plateau → 40% decay):
- Decay starts at step ~8,000 (1.5M data) or ~26,700 (5.7M data)
- A tail code appearing in 0.001% of samples gets ~0.08 expected occurrences during the entire decay phase (1.5M) or ~0.29 occurrences (5.7M)
- So tail codes get **zero to one** gradient updates during the phase where the model is supposed to refine and polish predictions

Multi-epoch training would help not because of a data-per-parameter ratio, but because it gives rare codes **multiple passes through the decay/polishing phase**, allowing the few gradient updates they receive to accumulate coherently.

### 1.3 Why Data Scaling Helps (And Why Its Effect is Bounded)

The loss drop from 0.003 → 0.002 with 3.6x data is entirely explained by a **threshold-crossing effect** for medium-frequency codes:

With 1.5M data, medium codes appear ~10 times per epoch. This is below the threshold where their gradient signal is consistent enough per batch to sustain learning. With 5.7M data, the same medium codes appear ~36 times — crossing the threshold.

Evidence: medium_top10_acc jumps from 0.16% → 4.26% (27x). This directly reduces the "poorly predicted positive" contribution to the loss, lowering the floor.

But tail codes go from ~1 to ~3.6 appearances — still insufficient. So the loss can drop further only if **either** even more data is added (to push more codes above threshold) **or** the gradient distribution is changed.

The data scaling effect is bounded: eventually, only rare/tail codes remain below threshold, and adding more data yields diminishing returns because the occurrence RATIO (tail vs common) is preserved in the data distribution.

---

## Part 2: Critical Review of Expert 1 and Expert 2

### 2.1 Expert 1: "Five Interlocking Ceilings"

#### What I AGREE With (and Why)

**1. The loss floor estimate (positive_rate × average_positive_loss ≈ 0.002-0.003).**

This is a correct first-order approximation. The mean-reduced BCE loss at convergence is dominated by the residual positive-class loss diluted across the full N×C element space. The estimated value matches observation. This is analytically useful and I endorse it.

**2. The shared encoder as a zero-sum competition for representational capacity.**

Expert 1's statement that "adding 256 more dimensions just gives the common codes 256 more dimensions to differentiate, while tail codes remain homogenized" is mechanistically correct and well-evidenced. The tail embedding std evidence (0.03 vs 0.27) directly supports this.

**3. The gradient starvation temporal dynamics (17% → 85% common over steps 0-3000).**

The empirical gradient tracking data is the strongest evidence in the entire analysis. The phase transition from balanced → concentrated gradients, and its invariance to pos_weight, is the most informative single observation.

**4. The intervention recommendations (per-tier loss, separate decoders, multi-epoch).**

These are well-targeted at the root cause. Per-tier loss decomposition directly attacks gradient distribution. Separate decoder heads address the shared-representation bottleneck. Multi-epoch training addresses rare code exposure. I endorse these priorities.

#### What I DISAGREE With (and Why)

**1. Ceiling 1 (BCE mean reduction) is misattributed as "the most fundamental ceiling."**

Expert 1 computes per-sample gradient contributions:
```
Rare code:   (200 × 0.00001 × 1.0) / 6297 ≈ 3.2 × 10⁻⁷
Common code: (1 × 0.01 × 0.1) / 6297 ≈ 1.6 × 10⁻⁷
```

And concludes rare codes get "only ~2x the gradient of a common code."

**This is analytically misleading.** The 1/6297 factor cancels when comparing codes — it affects all codes identically. The real issue is not per-element dilution but **batch-level aggregation**: common codes appear in every batch (contributing consistent gradient), while tail codes appear sporadically (contributing occasional spikes that are averaged out). The mean reduction is a constant divisor that doesn't change relative gradient contributions.

Expert 1 conflates two distinct effects: (a) the absolute magnitude of per-element gradients (affected by 1/C) and (b) the relative gradient distribution across codes (NOT affected by 1/C, but by occurrence frequency × batch composition). The root cause is (b), not (a).

**2. Ceilings 2 and 3 are the SAME mechanism, not two distinct ceilings.**

Expert 1 presents:
- Ceiling 2: "Shared-Encoder Gradient Capture Equilibrium"
- Ceiling 3: "Gradient Starvation Temporal Lock-In"

But Ceiling 3 is simply the temporal DYNAMICS of Ceiling 2. The gradient starvation IS the encoder monopolization, observed over time. They are not "interlocking" — they are one mechanism described from two angles. Presenting them separately inflates the apparent complexity without adding explanatory power.

**3. Ceiling 5 (LR Schedule) has been experimentally REJECTED and should not be included.**

Expert 1 includes the LR schedule as one of five "interlocking mechanisms" while simultaneously acknowledging that the polishing test rejected it. This is contradictory. The evidence is unambiguous: polishing at 10x lower LR for 2000 steps produced WORSE val_loss (+0.45%), WORSE NDCG (-0.30%), and zero change in rare/tail accuracy. The LR schedule is demonstrably not a contributing factor.

Including a rejected hypothesis as a "ceiling" undermines the credibility of the framework. The correct response to negative experimental evidence is to remove the hypothesis, not to relabel it.

**4. The "compound multiplication" (0.01 × 0.2 × 1 × 0.03/0.27 ≈ 0.00022) is mathematically meaningless.**

This multiplies:
- Gradient fraction (dimensionless ratio, 0-1)
- Min LR ratio (dimensionless ratio, 0-1)
- Epoch count (dimensionless integer)
- Embedding std ratio (dimensionless ratio)

These are **incommensurable quantities** — their product has no physical interpretation. The relationship between these factors and "achievable loss reduction" is highly nonlinear and not representable as a simple product. Stating "≈ 0.00022" creates a false sense of quantitative precision from a dimensional analysis error. An exponential decay model or a more principled information-theoretic bound would be appropriate; this is not.

**5. The "1-epoch = insufficient tokens-per-parameter" framing (Ceiling 4) uses an inapplicable analogy.**

Expert 1 computes "0.097 samples per parameter" and compares to "industry norms of 10-100x." This comparison imports LLM pretraining scaling laws into a clinical multi-label classification problem. The domains differ in: input vocabulary size (~10K vs ~50K+), sequence length (~200 vs ~2048+), output structure (multi-label BCE vs autoregressive softmax), and learning dynamics (static code distribution vs evolving language patterns).

More importantly, the 256d model at 0.225 samples/param achieves IDENTICAL performance to 512d at 0.097 samples/param. If data-per-parameter were the limiting factor, the 256d model should outperform. The fact that it doesn't disproves this framing.

### 2.2 Expert 2: "One Root Cause, Three Contributing Factors"

#### What I AGREE With (and Why)

**1. "Gradient-Driven Encoder Monopolization" as the single root cause — this is correct and parsimonious.**

Expert 2 correctly identifies that Ceilings 2 and 3 from Expert 1 are the same mechanism, elevates the shared encoder to the primary bottleneck, and demotes BCE mean reduction and LR schedule to secondary/rejected status. This is the more accurate and more useful framework.

**2. The critique of Expert 1's Ceiling 5 (LR schedule) is correct.**

Expert 2's statement "The LR schedule is NOT a ceiling. It's a non-factor that has been experimentally rejected" is the appropriate response to the polishing test evidence. Clear, direct, and evidence-based.

**3. The critique of Expert 1's compound multiplication is correct.**

Expert 2's identification of the multiplication as "incommensurable quantities whose product has no physical meaning" is mathematically rigorous and important. This prevents a misleading pseudo-quantitative argument from becoming embedded in the analysis.

**4. The "partial convergence" framing is better than "sharp local minimum."**

Expert 2 correctly distinguishes between "converged in common-code subspace" and "unexplored in rare-code subspace." This is more mechanistically precise than calling it a "sharp minimum," which implies the model is at a single point rather than in a high-dimensional region where different parameter subsets have different convergence states.

**5. The Bayes optimal loss analysis is excellent.**

Expert 2's insight that "The loss floor is NOT the Bayes optimal loss. It is the loss achieved by a model that is Bayes-optimal for common codes and maximally ignorant for tail codes" is elegant and correct. This precisely captures the asymmetry: the model has converged for the 85% of loss elements it can optimize, and the remaining 15% is irreducible under the current gradient distribution.

#### What I DISAGREE With (and Why)

**1. BCE mean reduction is more than "just an amplifier" — it shapes the loss landscape geometry.**

Expert 2 states: "The mean reduction is a contributing factor, not a root cause. The root cause is occurrence-level imbalance amplified by batch-level gradient aggregation."

I partially disagree. While the 1/C constant doesn't change RELATIVE gradient contributions, it DOES change the ABSOLUTE loss landscape. With reduction='mean', the loss surface is extremely flat near the floor — the remaining "improvable" loss from tail codes is divided by 6,297. This flatness means:
- Gradient norms are extremely small for ALL parameters near the floor
- Weight decay competes with (and often overwhelms) the learning signal
- Numerical precision (FP16 training) may truncate small gradients

If the loss used `reduction='sum'` or per-tier weighting, the absolute gradient magnitude for tail codes would be 6,297x larger (before batch-level aggregation effects). This would not change the relative common/tail ratio, but it WOULD change the absolute landscape geometry and the interaction with weight decay and numerical precision.

So the mean reduction is not just an amplifier of occurrence imbalance — it independently reduces the absolute magnitude of ALL remaining gradients at the floor, making the floor harder to escape even in principle.

**2. Expert 2 underemphasizes the input embedding feedback loop.**

Expert 2 mentions embeddings briefly but focuses primarily on the encoder → decoder pathway. The fact that `self.embedding_cd` (the very first layer, before any encoding) shows std=0.03 for tail codes means the problem starts at the INPUT, not just the representation. Even if you solved the encoder monopolization entirely, the input signal for tail codes is still uninformative because e_A ≈ e_B for any two tail codes A, B.

This input-level homogenization is a separate mechanism from encoder monopolization. It's caused by the same gradient starvation but operates at a different layer with a different fix (e.g., pre-trained code embeddings from co-occurrence data could break this independently of encoder changes).

**3. Expert 2's unified model, while more parsimonious, slightly understates the architectural contribution.**

Expert 2 attributes everything to "gradient distribution → encoder representation → loss floor." This is the correct causal chain but it elides an important architectural question: **Why can't the decoder compensate for an uninformative encoder?**

The answer is: because `nn.Linear(d, 6297)` is a linear readout. A nonlinear decoder (e.g., per-tier MLPs) could potentially extract information from h that a linear projection cannot. The decoder architecture is a separate, independent contributor to the ceiling — not through gradient dynamics, but through **representational expressiveness**. A 2-layer MLP per code tier could learn nonlinear feature combinations that the single linear layer cannot.

Expert 2 mentions "Tier-specific decoder heads" as an intervention but doesn't integrate this into the root cause analysis. It should be there: the linear decoder is an independent architectural limitation, not just a consequence of encoder monopolization.

### 2.3 Specific Claims: My Verdict on Each

| Claim | Expert 1 | Expert 2 | My Verdict | Reasoning |
|-------|----------|----------|------------|-----------|
| **BCE mean reduction is "most fundamental"** | Claims yes (Ceiling 1) | Claims no (amplifier) | **Neither is fully correct.** It's neither the root cause nor merely an amplifier — it independently shapes loss landscape flatness and interacts with weight decay/precision. | Expert 1 overclaims; Expert 2 underclaims. The truth is in between. |
| **Shared encoder is a zero-sum bottleneck** | Ceiling 2 | Root cause | **Agree with Expert 2's elevation to root cause** | This is the single most explanatory mechanism. Expert 1 buries it as #2. |
| **Gradient starvation is pos_weight-independent** | Yes | Agrees with observation | **Agree on the observation; Expert 2's framing is more precise** | "pos_weight addresses the wrong level (per-sample vs per-batch)" is more useful than "pos_weight-independent." |
| **1-epoch = insufficient tokens-per-param** | Yes (Ceiling 4) | **Disagrees with reasoning** | **Side with Expert 2** | The LLM scaling law comparison is inapplicable. The real issue is insufficient rare code exposure during polishing, not a data/param ratio. |
| **LR schedule is a ceiling** | Yes (Ceiling 5) | **Disagrees (rejected)** | **Side with Expert 2** | The polishing test is definitive: val_loss worsened, NDCG worsened, rare/tail unchanged. Including a rejected hypothesis weakens the analysis. |
| **Compound product ≈ 0.00022** | Presents as evidence | **Strongly disagrees** | **Side with Expert 2** | Multiplying incommensurable quantities produces a dimensionally meaningless number. |
| **Loss floor estimate ≈ positive_rate × avg_positive_loss** | Presented and validated | Agrees | **Both are correct** | Clean, testable prediction that matches observation within expected accuracy. |
| **Interventions: per-tier loss, separate decoders, multi-epoch** | Recommends | Recommends | **Agree with both** | These are correctly targeted at the root cause and amplifiers. |
| **Encoder monopolization IS gradient starvation (same mechanism)** | Separates as Ceilings 2, 3 | Correctly unifies | **Side with Expert 2** | Ceiling 3 is the temporal dynamics of Ceiling 2. Separating them inflates apparent complexity. |

### 2.4 What BOTH Experts Miss or Underemphasize

**1. Cross-code interference through shared representations.**

Neither expert explicitly identifies that the shared representation h is not just uninformative for tail codes but **actively suppresses** them. The observed tail logit of -14.69 is ~8.5 units more negative than what the decoder bias equilibrium alone would produce (-6.2). The excess comes from negative `wⱼ^T h` correlations — the common-code features in h create systematic negative dot products with tail decoder rows. This means more common-code features (from 512d) could make tail prediction WORSE, not just "equally bad."

**2. The input embedding feedback loop as an independent mechanism.**

The input embedding `self.embedding_cd` for tail codes is itself homogenized (std=0.03). This means the problem starts at the FIRST layer — before the encoder, before the decoder, before the gradient dynamics. Even if the gradient distribution were perfectly balanced, the input signal for tail codes would be uninformative because all tail codes map to approximately the same vector.

This is partially addressed by the intervention recommendation of "contrastive pre-training for code embeddings," but neither expert integrates it into the root cause model. It should be there: input embedding homogenization is a self-sustaining barrier that persists even if downstream gradient issues are addressed.

**3. The decoder's linear architecture as an independent capacity constraint.**

The single `nn.Linear(d, 6297)` decoder can only compute linear readouts from h. For common codes, this is sufficient because h contains rich common-code features that are linearly separable. For tail codes, even if h contained some weakly relevant features (e.g., "patient has chronic conditions" → slightly elevated probability for related tail code), a linear decoder may not be able to extract this signal. A nonlinear decoder (per-tier MLP) could learn feature combinations that linear projection cannot.

---

## Part 3: My Unified Model

### One Root Cause + Four Structural Amplifiers

**Root Cause: Gradient-Frequency Coupling in a Shared-Representation Architecture**

The shared encoder representation `h ∈ ℝ^d` is optimized by gradients proportional to `frequency(j) × pos_weight(j) × error(j)`, summed over batch elements. At convergence, this product overwhelmingly favors common codes due to batch-level aggregation. The encoder allocates representational capacity in proportion to gradient signal, making h informative only for common codes. The decoder for tail codes reads an uninformative h, regardless of its own capacity.

**Amplifier 1: Input Embedding Feedback Loop** (layer 0 — UPSTREAM of encoder)

Tail code input embeddings receive sparse, uniform gradient updates → converge to near-identical vectors → provide no distinctive input signal to encoder → encoder cannot learn tail-specific features even if gradient budget were rebalanced → input embeddings remain uninformative. This is a self-reinforcing barrier at the very first layer.

**Amplifier 2: Cross-Code Interference** (decoder level)

The common-code features in h create negative correlations with tail decoder rows, pushing tail logits ~8.5 units below the theoretical bias equilibrium. This means adding more common-code capacity (512d) may actively WORSEN tail prediction.

**Amplifier 3: BCE Mean Reduction** (loss landscape)

Division by N×C = N×6,297 flattens the loss surface near the floor. The remaining "improvable" loss from tail codes is divided by 6,297, making absolute gradient norms extremely small. This interacts with weight decay and FP16 precision limits to further suppress tail code learning signal.

**Amplifier 4: Single-Epoch Rare Code Deprivation** (training procedure)

Tail codes get zero to one exposure during the LR decay/polishing phase, denying any opportunity for coherent late-stage refinement. Multi-epoch training is necessary for rare codes to accumulate gradient signal during the phase that matters.

### What Is NOT a Root Cause or Amplifier

- **LR schedule**: Experimentally rejected by polishing test. Not a factor.
- **Model capacity (params)**: The floor is capacity-invariant. More params get monopolized by common codes.
- **pos_weight magnitude**: Invariant gradient distribution at 35 vs 200. Addresses wrong level (per-sample vs per-batch).
- **Data quantity alone**: Helps medium codes cross threshold but does not change relative frequency ratios.

### Implications for Breaking Through

The interventions must target the root cause and at least one amplifier:

| Priority | Intervention | Target | Mechanism |
|----------|-------------|--------|-----------|
| 1 | Per-tier loss decomposition | Root cause | Forces gradient budget rebalancing by computing loss separately per tier |
| 2 | Focal loss or ASL | Amplifier 3 | Dynamically down-weights easy (common) positives, concentrating gradient on hard (tail) positives |
| 3 | Multi-epoch training (2-3 epochs) | Amplifier 4 | Gives rare codes multiple passes through the polishing phase |
| 4 | Pre-trained code embeddings (co-occurrence) | Amplifier 1 | Breaks input embedding homogenization by initializing distinctive embeddings before supervised training |
| 5 | Per-tier nonlinear decoder heads | Amplifier 2 | Eliminates cross-code interference by giving each tier its own feature extraction pathway |

The minimum viable intervention is **per-tier loss decomposition + focal loss**: this addresses the root cause (gradient distribution) and the most impactful amplifier (loss landscape flatness) without requiring architectural changes.

---

## Summary

| Aspect | Expert 1 | Expert 2 | My Position |
|--------|----------|----------|-------------|
| **Framework** | 5 interlocking ceilings | 1 root cause + 3 factors | 1 root cause + 4 amplifiers (adding input embedding loop + cross-code interference) |
| **Primary mechanism** | BCE mean reduction (Ceiling 1) | Encoder monopolization | Encoder monopolization (agree with Expert 2 on primacy) |
| **BCE mean reduction** | Most fundamental | Just an amplifier | More than an amplifier (shapes loss landscape flatness), less than root cause |
| **LR schedule** | Ceiling 5 | Not a factor (rejected) | Not a factor — polishing test is definitive |
| **1-epoch constraint** | Tokens-per-param ratio | Rare code revisits | Rare code revisits during polishing phase (Expert 2 is correct) |
| **Compound product** | 0.00022 | Meaningless | Meaningless (incommensurable quantities) |
| **Ceilings 2+3** | Two separate ceilings | Same mechanism | Same mechanism (Expert 2 is correct) |
| **Input embeddings** | Mentioned as evidence | Mentioned briefly | Elevated to independent amplifier — self-reinforcing barrier at layer 0 |
| **Cross-code interference** | Not discussed | Not discussed | Identified as Amplifier 2 — tail logits actively suppressed by ~8.5 units beyond equilibrium |
| **Overall quality** | Directionally correct, over-structured, includes rejected hypothesis | More parsimonious, more rigorous, occasionally too reductive | Both provide valuable perspectives; Expert 2 is closer to the truth but underemphasizes input-level and decoder-level constraints |

Both experts converge on the correct diagnosis (encoder monopolization as the core issue) and the correct interventions (per-tier loss, density sampling, focal loss). The disagreements are primarily about framing, ranking, and intellectual rigor — not about the fundamental mechanism or the path forward.


# Expert 4: 

I now have all the evidence needed for a thorough, independent analysis. Let me synthesize everything.

---

# Independent Expert Analysis: Why the Loss Floor Is Invariant to Capacity and Data Scaling

## Preamble — Methodology and Scope

This analysis is conducted independently from the three experts whose prior work is documented. I have examined:
- The full model architecture (`FlashAttentionTransformer` in `moe_flashattn_4.py`)
- All loss functions (BCE, FocalLoss, AsymmetricLoss) and their `reduction='mean'` behavior
- The `DataParallelWrapper.forward()` loss computation pathway
- Six experiments: v2 (pw=35), v3 (pw=200), v4 (ASL, no pw), v5 (ASL + density sampler), R6 (big data), R7 (512d), R8 (512d + big data)
- The gradient tier tracking data (step-by-step evolution)
- The embedding/logit diagnostic check
- The LR polishing test results
- All prior expert analyses

I will: (1) present my own root cause analysis, then (2) systematically evaluate each expert's claims.

---

## Part 1: My Independent Root Cause Analysis

### 1.1 Establishing the Evidence Base

**Fact 1 — The loss floor is capacity-invariant within each data tier:**

| Run | Params | Data | Smoothed Train Loss | Val Loss |
|-----|--------|------|---------------------|----------|
| V3 (256d) | 25.3M | 1.5M | 0.00319 | 0.00322 |
| R7 (512d) | 58.6M | 1.5M | 0.00318 | 0.00316 |
| R6 (256d) | 25.3M | 5.7M | 0.00212 | 0.00205 |
| R8 (512d) | 58.6M | 5.7M | 0.00213 | 0.00204 |

Within each data tier, doubling parameters yields <0.5% loss change. Across data tiers, 3.6x more data yields ~35% reduction.

**Fact 2 — The architecture channels all 6,297 code predictions through one shared vector:**

```2668:2675:dev/moe/moe_flashattn_4.py
        cd = torch.swapaxes(cd, 0, 1)
        
        cd = self.norm(cd)
        cd = self.dropout(cd)
        cd = self.decoder_cd(cd)
        
        return cd
```

Where `self.decoder_cd = nn.Linear(config.embedding_size, config.target_cd_cnt)` maps `h ∈ ℝ^d` to 6,297 logits via `z_j = w_j^T h + b_j`. Every code reads from the same `h`.

**Fact 3 — Gradient starvation is confirmed and pos_weight-invariant.** By step 3,000, common codes capture 66.7% of gradient norm; by step 12,001, 85.3%. This distribution is **identical** at pw=35 and pw=200.

**Fact 4 — The LR polishing test is definitively negative.** Resuming at 10x lower LR: val_loss worsened (+0.45%), recall@10 dropped from 0.825→0.788 at step 200, rare/tail remained at 0%. Diagnosis: `STRUCTURAL_BOTTLENECK`.

**Fact 5 — Tail code input embeddings are homogenized.** Tail embedding std = 0.03 vs common std = 0.27. All ~1,175 tail codes have near-identical learned embeddings.

**Fact 6 — Tail logits show both extreme negative prior AND cross-code interference.** Tail positive logit = -14.69. Theoretical equilibrium bias ≈ log(200 × 0.00001) ≈ -6.2. The excess ~8.5 units of suppression comes from the `w_j^T h` term — common-code features in `h` create negative correlations with tail decoder rows.

**Fact 7 — And this is the fact I want to emphasize most, because it is the most diagnostic:**

**ASL (v4/v5) fundamentally changes the loss landscape but does NOT change the gradient tier distribution:**

| Experiment | Loss Function | val_loss (own) | common_frac | tail_frac | tail_top10_acc |
|---|---|---|---|---|---|
| v3 (BCE, pw=200) | BCEWithLogits | 0.00322 | 84.7% | 0.17% | 0% |
| v4 (ASL, no pw) | ASL(γ-=4.0) | 0.000776 | 85.7% | 0.12% | 0% |
| v5 (ASL+sampler) | ASL(γ-=4.0)+density | 0.000773 | 86.1% | 0.09% | 0% |

ASL with γ_neg=4.0 explicitly down-weights easy negatives by p^4. This:
- Reduces the ASL-measured val_loss 4x
- Dramatically improves calibration (positive_brier: 0.687 → 0.313)
- Dramatically improves top-1 ranking (MRR: 0.324 → 0.496, recall@1: 0.000 → 0.284)
- But the gradient tier distribution is **UNCHANGED** (86% common, 0.1% tail)
- And tail_top10_acc remains **exactly 0%**

This is the single most diagnostic observation in the entire experimental history, and I believe none of the three experts sufficiently emphasizes its implications.

### 1.2 My Root Cause: Occurrence-Frequency-Driven Representation Monopolization

The root cause is **not** the loss function. The v4/v5 experiments prove this — changing the loss dramatically changes what the model optimizes for (better calibration, better ranking) but the gradient tier distribution is invariant.

The root cause is **not** the LR schedule. The polishing test proves this.

The root cause is **not** model capacity. R7/R8 prove this.

The root cause is that **the per-batch gradient aggregation is dominated by occurrence frequency, which is an intrinsic property of the data distribution, not the loss function or the model.**

Let me derive this precisely.

In any single training batch of ~6,400 patient-days (128 members × ~50 valid days), the gradient that reaches the shared encoder for code j is:

```
G_j = Σ_{i ∈ batch} (∂L_i/∂z_j) × w_j × (∂h_i/∂θ_enc)
```

The number of terms in this sum where z_j is "interesting" (i.e., y_ij = 1) is:

- For a common code (1% frequency): ~64 patient-days per batch
- For a tail code (0.001% frequency): ~0.064 patient-days per batch

**This is a factor of 1,000x in the number of informative gradient terms per batch.** No per-element loss reweighting can overcome this because:

1. **pos_weight=200** amplifies each tail term by 200x → ratio drops from 1000:1 to 5:1, but common codes contribute from **64 independent samples** while tail codes contribute from **0-1 samples per batch**. The variance of the tail gradient estimate is enormously higher.

2. **ASL with γ_neg=4.0** eliminates easy negative gradient but does NOT increase the number of informative positive samples. The tail code STILL appears in 0-1 patient-days per batch. The v4/v5 gradient tier data proves this: common_frac=86% with ASL, same as BCE.

3. **Focal loss** down-weights easy positives and easy negatives — but again, it doesn't change how many times a tail code appears in a batch.

4. Even **density-aware sampling** (v5, `tier_tail_quota=20`) enriches 20/128 batch samples with tail-code-containing patients. But there are ~1,175 tail codes. So each specific tail code gets gradient from ~20/1175 ≈ 0.017 samples per batch, while each specific common code gets gradient from ~64/1169 ≈ 0.055 samples per batch. This is only a 3x improvement in the common-to-tail ratio — still far from sufficient.

**The key insight**: the problem is not per-element weighting (which pos_weight, focal loss, and ASL all address) but **per-code sample count in each batch**. There are ~1,175 tail codes competing for 0-20 batch positions. Even with perfect per-element weighting, the gradient signal for any single tail code is based on 0-1 observations per batch — a hopelessly noisy estimate.

### 1.3 Why This Creates a Capacity-Independent Loss Floor

Once we understand the root cause, the capacity-independence follows directly:

1. The encoder representation `h` is updated by `∂L/∂θ_enc`, which is dominated by common-code gradient terms (85% of norm)
2. `h` becomes informative for common codes and uninformative for tail codes
3. The decoder `z_j = w_j^T h + b_j` for tail codes works with uninformative `h`
4. The loss for well-predicted codes (common) approaches zero
5. The loss for poorly-predicted codes (tail) contributes a residual that neither capacity nor loss reweighting can reduce

The loss floor is simply:

```
L_floor ≈ (1/(N×C)) × Σ_{j ∈ "unpredictable"} Σ_i [w_j × y_ij × (-log(σ(z_ij))) + (1-y_ij) × (-log(1-σ(z_ij)))]
```

Since the set of "unpredictable" codes is determined by occurrence frequency (which is data-distribution-invariant and capacity-invariant), the floor is invariant to capacity.

More data reduces the floor because it moves some codes from "unpredictable" (too few gradient samples) to "predictable" (enough gradient samples). This is exactly what we see: medium_top10_acc jumps from 0.16% to 4.26% when data scales 3.6x, because medium codes that appeared ~10 times per epoch at 1.5M now appear ~36 times at 5.7M, crossing the gradient visibility threshold.

### 1.4 The Three Structural Amplifiers

While the root cause (occurrence-frequency-driven gradient aggregation) is sufficient to explain the floor, three architectural and training choices amplify its severity:

**Amplifier A: Shared Encoder → Representation Monopolization**

The shared `h ∈ ℝ^d` is a zero-sum information space. Since 85% of gradient updates serve common codes, `h` learns common-code features. Adding more dimensions (256→512) adds capacity that gets monopolized for common-code features. The decoder for tail codes gains more features to read, but those features are no more informative.

This is an **architectural amplifier** because an alternative architecture (e.g., per-tier encoders or per-tier decoder heads) would allow different parts of the model to specialize for different code tiers, even under the same gradient distribution.

**Amplifier B: Input Embedding Feedback Loop**

The input embedding `self.embedding_cd` for tail codes is itself homogenized (std=0.03). This means the problem starts at layer 0: even before the encoder processes anything, "day with tail code A" and "day with tail code B" look identical in embedding space. This creates a vicious cycle:

1. Tail embeddings are near-identical → encoder receives no distinctive input for tail codes
2. Encoder produces `h` uninformative for tail codes → decoder can't predict them
3. Tail codes contribute near-zero gradient → their embeddings don't differentiate
4. Return to step 1

This is distinct from the encoder monopolization — even if you fixed the encoder gradient distribution, the input-level homogenization would persist as a separate barrier.

**Amplifier C: Single-Epoch Rare Code Deprivation**

With 1-epoch training and a linear schedule (15% warmup → 45% plateau → 40% decay), tail codes get 0-1 gradient updates during the entire decay/polishing phase. Multi-epoch training would give tail codes multiple passes through the polishing phase, allowing their sparse gradient updates to accumulate coherently. But without fixing Amplifier A or the root cause, multiple epochs would just repeat the same monopolized gradient pattern.

### 1.5 What Would Actually Break Through

Based on my analysis, interventions must target the root cause (per-batch occurrence imbalance) or the primary amplifier (shared representation):

| Priority | Intervention | Target | Mechanism | Expected Impact |
|----------|-------------|--------|-----------|-----------------|
| 1 | **Per-code balanced sampling** (not per-tier) | Root cause | Ensures each code gets comparable batch presence by oversampling patients with rare codes to match per-code-per-batch counts | Directly equalizes gradient budget |
| 2 | **Per-tier decoder heads** (separate MLPs per tier) | Amplifier A | Breaks shared representation bottleneck; each tier extracts different features from `h` | Allows tail decoders to find weak signals |
| 3 | **Per-tier loss decomposition** | Root cause + Amplifier A | Compute loss separately per tier, weight tiers equally → forces gradient rebalancing | Changes ∂L/∂θ_enc distribution |
| 4 | **Multi-epoch training** (2-3 epochs, combined with #1 or #3) | Amplifier C | Gives rare codes multiple polishing-phase passes | Only effective if gradient distribution changes |
| 5 | **Pre-trained code embeddings** (e.g., from co-occurrence matrix) | Amplifier B | Breaks input homogenization by initializing distinctive embeddings | Gives encoder meaningful input signal for tail codes |

Note: I exclude focal loss and ASL from recommendations because **v4/v5 experimentally proved** they do not change the gradient tier distribution or tail accuracy. They improve calibration and ranking but do not address the fundamental per-batch-per-code sample count problem.

---

## Part 2: Systematic Critical Review of Expert 1 and Expert 2/3

### 2.1 Expert 1: "Five Interlocking Ceilings" Framework

#### Claims I AGREE With

**1. The loss floor estimate: `positive_rate × average_positive_loss ≈ 0.002-0.003`**

This is a clean, first-order approximation that matches observation. The mean-reduced BCE loss at convergence is dominated by the residual positive-class loss from codes the model fails to predict. I endorse this estimate.

**Reasoning**: At convergence, negative terms contribute ~0 (model correctly predicts absence). Positive terms for well-predicted codes contribute ~0 (model correctly predicts presence). The remaining loss is from positive terms for poorly-predicted codes: `(num_poorly_predicted_positives / (N × C)) × average_positive_BCE_for_those_codes`. Given positive rate ~0.2% and ~50% of those being poorly predicted, this gives ~0.001 × ~2.0 ≈ 0.002, matching the observed 5.7M data floor. For 1.5M data, more codes are poorly predicted, giving ~0.003.

**2. Shared encoder as a zero-sum competition**

Expert 1's statement that "adding 256 more dimensions just gives the common codes 256 more dimensions to differentiate" is mechanistically correct, well-reasoned, and directly supported by the gradient tier data and embedding homogenization evidence.

**3. The gradient starvation temporal dynamics (17% → 85%)**

The empirical gradient tier tracking is the strongest evidence in the entire body of work. The phase transition is clearly documented and reproducible.

**4. The intervention recommendations (per-tier loss, separate decoders, multi-epoch)**

These are well-targeted at the actual mechanisms. However, I note that Expert 1 does not discuss why v4/v5 (ASL) failed to break the ceiling, which would have strengthened the analysis.

#### Claims I DISAGREE With

**1. Ceiling 1 (BCE mean reduction) as "the most fundamental and least discussed ceiling" — DISAGREE**

Expert 1 computes:
```
Rare code:   (200 × 0.00001 × 1.0) / 6297 ≈ 3.2 × 10⁻⁷
Common code: (1 × 0.01 × 0.1) / 6297 ≈ 1.6 × 10⁻⁷
```

And concludes "even with 200x pos_weight, the rare code gets only ~2x the gradient of a common code."

**This is analytically wrong.** The 1/6297 divisor cancels when comparing codes — it's applied equally to all codes. The per-element gradient ratio IS ~2x (200 × frequency × error vs 1 × frequency × error). But the real issue is not this per-element ratio; it's the **number of elements** that contribute. A common code has ~64 positive elements per batch; a tail code has ~0.064. The 1000x disparity in sample count overwhelms the 2x per-element advantage.

Furthermore, the v4/v5 evidence directly falsifies Expert 1's claim that mean reduction is the primary mechanism. ASL eliminates the easy-negative dominance (by down-weighting with p^4) and effectively replaces `reduction='mean'` behavior with something much more favorable to hard positives. Yet the gradient tier distribution is UNCHANGED (86% common, 0.1% tail). If the mean reduction were "the most fundamental ceiling," ASL should have dramatically altered the gradient distribution. It did not.

**My verdict**: BCE mean reduction is **not a root cause**. It contributes to the absolute flatness of the loss landscape at the floor, but it does not determine **which codes** the model fails on. The root cause is occurrence-frequency-driven batch-level gradient aggregation.

**2. Ceilings 2 and 3 are the SAME mechanism presented twice — DISAGREE with the separation**

Expert 1 lists:
- Ceiling 2: "Shared-Encoder Gradient Capture Equilibrium"
- Ceiling 3: "Gradient Starvation Temporal Lock-In"

Ceiling 3 is the temporal dynamics of Ceiling 2 — they describe the same mechanism (gradient monopolization of shared encoder) at different time points. Listing them as separate "interlocking ceilings" inflates the count without adding explanatory power. The transition from balanced (step 1) to concentrated (step 3000) to terminal (step 12000) is a single process, not two interacting ceilings.

**3. Ceiling 5 (LR Schedule) was experimentally REJECTED and should not be listed — STRONGLY DISAGREE**

Expert 1 includes the LR schedule as a ceiling while simultaneously citing the polishing test that rejected it. The polishing test data is unambiguous:
- val_loss: 0.00336 → 0.00338 (+0.45%, WORSE)
- NDCG@20: 0.433 → 0.432 (-0.34%, WORSE)
- recall@10: 0.825 → 0.821 (WORSE after partial recovery)
- tail/rare accuracy: 0% → 0% (UNCHANGED)
- Diagnosis field literally says: `"STRUCTURAL_BOTTLENECK"`

Including a hypothesis that has been experimentally rejected as one of five "interlocking mechanisms" is methodologically unsound. It undermines confidence in the entire framework.

**4. The compound multiplication `0.01 × 0.2 × 1 × (0.03/0.27) ≈ 0.00022` — STRONGLY DISAGREE**

This multiplies:
- A dimensionless gradient fraction (0.01)
- A dimensionless LR ratio (0.2) — from a rejected ceiling!
- A dimensionless count (1 epoch)
- A dimensionless embedding ratio (0.03/0.27)

These quantities are incommensurable — their product has no defined physical units or interpretation. The actual relationship between these factors and "achievable loss reduction" is highly nonlinear, depends on optimizer state, learning rate magnitude, batch composition, and hundreds of other variables. The result "≈ 0.00022" creates a false impression of quantitative rigor from what is essentially dimensional analysis malpractice.

**5. The "1-epoch = insufficient tokens-per-parameter" framing — PARTIALLY DISAGREE**

Expert 1's comparison to "industry norms of 10-100x data per parameter" imports LLM pretraining scaling laws into a clinical multi-label classification problem. These domains differ fundamentally:
- LLM: autoregressive next-token prediction over a unified vocabulary
- This task: multi-label BCE over independent code indicators
- LLM scaling laws assume the model must memorize distributional properties of natural language
- This task has structured, finite code co-occurrence patterns

More importantly, the 256d model (25.3M params, 0.225 samples/param) achieves **identical** performance to the 512d model (58.6M params, 0.097 samples/param). If tokens-per-parameter were the limiting factor, the 256d model should outperform at the same data scale. It does not. This directly falsifies the tokens-per-param framing.

The real reason 1-epoch is a limitation is that tail codes get 0-1 appearances during the polishing phase, not that the overall data/param ratio is low.

### 2.2 Expert 2: "One Root Cause, Three Contributing Factors" (in the same document)

#### Claims I AGREE With

**1. Identifying encoder monopolization as the single root cause — AGREE**

Expert 2 correctly unifies Ceilings 2 and 3 from Expert 1, elevates the shared encoder to root cause status, and demotes BCE mean reduction. This is more parsimonious and more accurate.

**2. Rejecting Ceiling 5 (LR schedule) — AGREE**

Expert 2's statement "The LR schedule is NOT a ceiling. It's a non-factor that has been experimentally rejected" is the correct response to definitive negative experimental evidence.

**3. The compound multiplication critique — AGREE**

Expert 2's identification that the multiplication involves "incommensurable quantities whose product has no physical meaning" is mathematically rigorous and important.

**4. The "partial convergence" framing — AGREE**

Calling the plateau state "partial convergence" (good solution in common-code subspace, unexplored in rare-code subspace) is more precise than "sharp local minimum," which implies the model is trapped rather than simply unfed.

**5. The Bayes optimal loss analysis — STRONGLY AGREE**

The insight that "the loss floor is NOT the Bayes optimal loss — it is the loss achieved by a model that is Bayes-optimal for common codes and maximally ignorant for tail codes" is the most elegant single statement of the problem in any of the expert analyses.

#### Claims I DISAGREE With

**1. "BCE mean reduction is just an amplifier" understates its role slightly — PARTIALLY DISAGREE**

Expert 2 is correct that the 1/C factor doesn't change *relative* gradient contributions. But the mean reduction does independently affect the absolute loss landscape: at the floor, the remaining "improvable" loss is divided by 6,297, making the absolute gradient norms extremely small. This interacts with:
- Weight decay (which may overpower tiny learning signals)
- FP16 precision limits (gradients may be truncated to zero)
- AdamW second moment estimates (tiny gradients may be further suppressed by large denominators)

That said, the v4/v5 evidence shows that even with ASL (which fundamentally changes the per-element loss landscape), the gradient distribution is unchanged. This means the mean reduction's "amplifier" effect is secondary to the occurrence-frequency root cause, supporting Expert 2's position more than mine.

**My revised position**: Expert 2 is substantially correct. The mean reduction is a secondary amplifier, not a root cause. My concern about absolute gradient magnitude is valid but insufficient to override the v4/v5 evidence.

**2. Expert 2 does not sufficiently address the v4/v5 ASL evidence — DISAGREE (by omission)**

The v4/v5 experiments are the most diagnostic evidence available, yet Expert 2 does not explicitly analyze them. The fact that ASL (which directly addresses the "easy negative dominance" that both experts discuss) FAILS to change the gradient distribution is the strongest possible evidence for Expert 2's thesis. Expert 2 should have used this to conclusively prove that the root cause is occurrence-frequency-driven, not loss-function-driven.

**3. Expert 2 underemphasizes the input embedding feedback loop — AGREE with Expert 3's critique**

The input embedding homogenization (std=0.03 for tail codes) is an independent barrier at layer 0. Even if the gradient distribution were perfectly balanced, the encoder would still receive uninformative inputs for tail codes because `e_A ≈ e_B` for any two tail codes A, B. Expert 2 mentions this but doesn't integrate it as a structural amplifier.

### 2.3 Expert 3's Analysis (Also in the Same Document)

Expert 3 provides the most comprehensive framework (1 root cause + 4 amplifiers) and identifies two mechanisms the other experts miss: (a) cross-code interference and (b) the input embedding feedback loop as an independent amplifier.

#### Claims I AGREE With

**1. Cross-code interference analysis — STRONGLY AGREE**

Expert 3's identification that the tail logit of -14.69 is ~8.5 units more negative than the theoretical bias equilibrium (-6.2) is the strongest novel contribution across all three expert analyses. The excess suppression from `w_j^T h ≈ -8.5` proves that the common-code features in `h` don't just fail to help tail codes — they **actively suppress** them. This has an important implication: adding 512d may make tail prediction *worse* (not just "equally bad") because more common-code features mean more negative cross-products with poorly-trained tail decoder rows.

**2. Input embedding feedback loop as a separate amplifier — AGREE**

This is a distinct mechanism from encoder monopolization. The vicious cycle at layer 0 (homogenized embeddings → uninformative encoder input → no gradient signal → embeddings stay homogenized) persists independently of downstream gradient distribution. Expert 3 is correct to elevate this to an independent amplifier with its own fix (pre-trained embeddings from co-occurrence data).

**3. The linear decoder as an independent architectural constraint — AGREE**

Expert 3's observation that `nn.Linear(d, 6297)` can only compute linear readouts is important. A nonlinear per-tier decoder could learn feature combinations that a linear projection cannot. This is not just a consequence of encoder monopolization — it's a separate expressiveness limitation.

#### Claims I DISAGREE With

**1. Expert 3 does not sufficiently leverage the v4/v5 ASL evidence — DISAGREE (by omission)**

Like Expert 2, Expert 3 does not explicitly analyze the v4/v5 results as a diagnostic tool. The ASL experiments are the most powerful falsification test available: they prove that loss-function interventions alone are insufficient. Expert 3 recommends "Focal loss or ASL" as priority #2, but the v4/v5 evidence already shows this doesn't change the gradient distribution or tail accuracy. This recommendation should be downgraded or removed.

**2. Expert 3's "minimum viable intervention" of "per-tier loss decomposition + focal loss" — PARTIALLY DISAGREE**

Expert 3 recommends per-tier loss decomposition + focal loss as the minimum viable intervention. Given that v4/v5 already tested ASL (which is functionally similar to focal loss in its negative-downweighting behavior) with no effect on gradient distribution, focal loss alone is insufficient. The minimum viable intervention should be **per-tier loss decomposition + per-code balanced sampling** — addressing both the gradient aggregation mechanism and the batch composition mechanism.

---

## Part 3: Synthesis — Where All Three Experts Converge and Where They Diverge

### Points of Universal Agreement

All three experts agree on:
1. **The shared encoder representation is the architectural bottleneck** — `h ∈ ℝ^d` monopolized by common codes
2. **Gradient starvation is confirmed** — 85% common, <1% tail at terminal
3. **The LR polishing test is negative** — LR changes cannot escape the basin
4. **The loss floor estimate** ≈ positive_rate × average positive loss matches observation
5. **The interventions**: per-tier loss decomposition, separate decoder heads, and multi-epoch training are the right directions

### Points of Disagreement and My Position

| Claim | Expert 1 | Expert 2 | Expert 3 | My Position |
|---|---|---|---|---|
| **Number of root causes** | 5 interlocking | 1 root + 3 factors | 1 root + 4 amplifiers | **1 root + 3 amplifiers** (root = occurrence-frequency batch aggregation, not encoder monopolization per se) |
| **Is BCE mean reduction the primary mechanism?** | Yes (Ceiling 1) | No (amplifier) | Middle ground (shapes landscape) | **No** — v4/v5 ASL disproves; it's a secondary amplifier |
| **Are gradient starvation and encoder monopolization the same thing?** | No (Ceilings 2, 3) | Yes | Yes | **Yes** — Ceiling 3 is the temporal dynamics of Ceiling 2 |
| **Is the LR schedule a factor?** | Yes (Ceiling 5) | No (rejected) | No (rejected) | **No** — polishing test is definitive |
| **Is the compound product (0.00022) meaningful?** | Yes | No | No | **No** — incommensurable quantities, no physical meaning |
| **Is tokens-per-param ratio the right framing for 1-epoch?** | Yes | No (wrong reasoning) | No (rare code revisits) | **No** — LLM scaling laws are inapplicable; the issue is rare code exposure during polishing |
| **Is focal loss / ASL a viable intervention?** | Not tested | Not explicitly discussed | Priority #2 | **No — v4/v5 already proved ASL doesn't change gradient distribution or tail accuracy** |
| **Is cross-code interference a factor?** | Not discussed | Not discussed | Yes (-8.5 units) | **Yes** — this is a critical novel insight |
| **Is input embedding feedback a separate amplifier?** | Mentioned as evidence | Mentioned briefly | Yes (Amplifier 1) | **Yes** — self-reinforcing barrier at layer 0 |

### The Critical Gap in All Three Analyses

**None of the three experts sufficiently leverages the v4/v5 ASL experimental evidence as a falsification tool.**

The v4/v5 experiments are the single most informative diagnostic in the entire experimental history because they test the hypothesis "if we change the per-element loss weighting to favor hard positives, does the gradient distribution change?" The answer is a definitive **NO**:

| | BCE (v3, pw=200) | ASL (v4, no pw) | ASL+density (v5) |
|---|---|---|---|
| common_frac | 84.7% | 85.7% | 86.1% |
| tail_frac | 0.17% | 0.12% | 0.09% |
| tail_top10_acc | 0% | 0% | 0% |
| MRR | 0.324 | 0.471 | 0.496 |
| positive_brier | 0.687 | 0.313 | 0.308 |

This proves that:
1. Per-element loss reweighting (pos_weight, focal modulation, ASL) is **irrelevant** to the gradient tier distribution
2. The gradient distribution is controlled by **occurrence frequency** in the training batches, not the loss function
3. Loss function changes CAN improve calibration and ranking (MRR, Brier) without breaking the tail code barrier
4. Therefore, the path forward must change **what the model sees per batch** (sampling), not **how it weighs each element** (loss function)

This evidence should have been the centerpiece of any analysis about why the loss floor is invariant — because it directly isolates the mechanism: changing the loss changes the landscape but not the gradient distribution, proving the distribution is data-driven, not loss-driven.

---

## Part 4: My Unified Root Cause Model

### One Root Cause

**Occurrence-frequency-driven per-batch gradient aggregation**

The total gradient for the shared encoder parameters is the sum of per-code, per-sample gradient terms. Common codes contribute ~64 informative terms per batch; tail codes contribute ~0.064. This 1000x disparity in per-batch sample count determines the gradient distribution (85% common, <1% tail), and this distribution is:
- Independent of loss function (proved by v4/v5 ASL)
- Independent of per-element weighting (proved by pw=35 vs pw=200)
- Independent of model capacity (proved by R7/R8)
- Dependent only on the data distribution's code occurrence frequencies

### Three Structural Amplifiers

1. **Shared representation monopolization** (architectural): The single `h ∈ ℝ^d` → `Linear(d, 6297)` architecture forces all codes to share features. Gradient-dominated common codes monopolize `h`, leaving it uninformative for tail codes and introducing cross-code interference (-8.5 units of active suppression).

2. **Input embedding feedback loop** (layer 0, self-reinforcing): Tail code embeddings homogenize (std=0.03) due to sparse, uniform gradient updates. This creates uninformative inputs at the very first layer, upstream of all encoding.

3. **Single-epoch rare code deprivation** (training procedure): Tail codes get 0-1 exposure during the LR decay/polishing phase, preventing any coherent late-stage refinement.

### What Is Definitively NOT a Root Cause

- **LR schedule** — experimentally rejected (polishing test)
- **Loss function** — experimentally rejected (v4/v5 ASL)
- **pos_weight magnitude** — experimentally rejected (pw=35 vs pw=200, identical gradient distribution)
- **Model capacity** — experimentally rejected (R7/R8, identical loss floor)
- **BCE mean reduction** — experimentally rejected as primary mechanism (v4/v5 ASL changes loss landscape without changing gradient distribution)

---

## Part 5: Final Assessment

| Aspect | Expert 1 | Expert 2 | Expert 3 | My Assessment |
|---|---|---|---|---|
| **Overall Quality** | Thorough but overstructured; includes rejected hypothesis; analytical errors in gradient math | Most parsimonious; methodologically rigorous; occasionally too reductive | Best novel insights (cross-code interference, input feedback loop); most comprehensive amplifier identification | Expert 2 has the most defensible framework; Expert 3 adds the most novel contributions; Expert 1 is directionally correct but structurally flawed |
| **Primary Blind Spot** | Does not analyze v4/v5 ASL as falsification evidence | Does not analyze v4/v5 ASL; underweights input embedding loop | Does not analyze v4/v5 ASL; recommends focal/ASL despite evidence against | **All three fail to use v4/v5 as the key diagnostic** |
| **Actionability** | Intervention list is correct but includes focal loss (experimentally rejected) | Intervention priorities well-ordered but missing code-level sampling | Intervention list most complete; correctly identifies pre-trained embeddings as separate fix | Per-code balanced sampling (not per-tier) is the highest-leverage intervention absent from all three analyses |

### The Bottom Line

The loss floor's invariance to capacity and data scaling is explained by one mechanism: **the per-batch gradient aggregation is proportional to per-code occurrence frequency, which is an intrinsic property of the data distribution that no model change, loss function change, or LR schedule change can alter.** Adding more parameters expands capacity that gets monopolized. Adding more data shifts some medium codes above the gradient visibility threshold but preserves relative frequency ratios. Changing the loss function (ASL) changes calibration and ranking but not the gradient distribution.

The path forward requires interventions that change **what the model sees per batch** (per-code balanced sampling, not just per-tier enrichment) combined with architectural changes that break the shared representation bottleneck (per-tier decoder heads, pre-trained code embeddings). Loss function changes (focal, ASL) are demonstrably insufficient as a standalone intervention.

# What previous three experts missed: The Epistemic Shift: What v4/v5 Changed in My Thinking

### Before v4/v5: Theory-Only Reasoning

Without the v4/v5 results, my reasoning process would have been confined to the same evidence base that all three experts worked from: the architecture code, the gradient tier tracking data, the embedding/logit analysis, and the polishing test. From those alone, I would have arrived at roughly the same diagnosis as the experts — some version of "shared encoder monopolization + gradient starvation + loss function effects."

The critical problem with theory-only reasoning here is that **multiple competing hypotheses predict the same observations**. Consider these two hypotheses:

- **Hypothesis A**: "The gradient distribution is shaped by the loss function's treatment of easy negatives — since 99.8% of loss elements are negative-class, and their near-zero contributions still accumulate to dominate, the loss function's averaging mechanics are the root cause."
- **Hypothesis B**: "The gradient distribution is shaped by per-code occurrence frequency in the training batches — common codes appear in every batch providing consistent gradient, tail codes appear sporadically, and no per-element reweighting can compensate."

Under standard BCE training (V3), **both hypotheses predict identical observations**: 85% gradient to common codes, 0.1% to tail, gradient starvation, homogenized embeddings, invariant loss floor. There is no way to distinguish them from the V3 evidence alone. All three experts attempt to do so using architectural reasoning, but architectural reasoning can only tell you what *could* be true, not what *is* true.

### The v4/v5 Evidence as a Discriminating Experiment

The v4 experiment (ASL with γ_neg=4.0, no pos_weight) is effectively a **controlled intervention** that separates Hypothesis A from Hypothesis B:

- ASL with γ_neg=4.0 applies a `p^4` modulation to negative-class gradient contributions. For an easy negative with sigmoid output p=0.001, the modulation factor is (0.001)^4 = 10^-12 — essentially zeroing out the gradient from easy negatives entirely.
- This **eliminates** the mechanism Expert 1 describes as "the most fundamental ceiling" — the mean reduction's dilution of positive-class signal by negative-class averaging.
- If Hypothesis A were correct, the gradient distribution should shift dramatically under ASL, because the easy-negative dominance is removed.

The result:

| | BCE (v3) | ASL (v4) | ASL+sampler (v5) |
|---|---|---|---|
| common_frac | 84.7% | 85.7% | 86.1% |
| tail_frac | 0.17% | 0.12% | 0.09% |

**Hypothesis A is falsified.** The gradient distribution does not change under ASL. Hypothesis B — that occurrence frequency determines the distribution — is confirmed.

This single observation eliminated an entire class of explanations from consideration. Without it, I would have been in the same position as the three experts: unable to determine whether the loss function or the data distribution is primary, forced to reason by analogy and intuition rather than evidence.

### What Specifically Changed in My Analysis

**1. The loss function dropped from "potential root cause" to "confirmed non-factor" for the gradient distribution.**

Before v4/v5: I would have treated BCE mean reduction as a plausible primary mechanism (like Expert 1 does) or at least a significant amplifier (like Experts 2 and 3 do). I would have had no empirical basis to strongly disagree with Expert 1's Ceiling 1.

After v4/v5: I could make a definitive statement — the loss function does not control the gradient tier distribution. This is not a theoretical argument; it's an empirical fact. This gave me the foundation to directly and confidently reject Expert 1's Ceiling 1 as "the most fundamental ceiling," rather than merely offering a theoretical counter-argument.

**2. I could separate "what the loss function controls" from "what it doesn't."**

The v4/v5 results show something subtle but critical:

| Metric | BCE (v3) | ASL (v4) | Change |
|---|---|---|---|
| positive_brier | 0.687 | 0.313 | -54% (massive improvement) |
| MRR | 0.324 | 0.471 | +45% (massive improvement) |
| recall@1 | 0.000 | 0.240 | +∞ (breakthrough) |
| tail_top10_acc | 0% | 0% | unchanged |
| gradient common_frac | 84.7% | 85.7% | unchanged |

The loss function **does** control calibration and ranking quality. It **does not** control gradient distribution or tail code learning. This nuance matters enormously: it means ASL/focal loss are genuinely valuable for improving model quality on *the metrics they affect* — but they are irrelevant to *the specific question of why the loss floor is capacity-invariant*.

Without v4/v5, I could not have drawn this distinction. I would have either dismissed loss function changes entirely (losing the calibration/ranking insight) or over-recommended them (as Expert 3 does by listing focal/ASL as Priority #2).

**3. I was able to identify a blind spot shared by all three experts.**

All three experts recommend focal loss or ASL as interventions to break the tail code barrier. Expert 1 suggests focal loss as part of the solution. Expert 2 recommends it implicitly. Expert 3 lists "Focal loss or ASL" as Priority #2.

The v4/v5 data shows this has already been tried and it did not change the gradient distribution or tail accuracy. This means all three experts are recommending an intervention that **has already been experimentally falsified** for the specific purpose they intend it for. This is a significant blind spot, and it could only be identified by having the v4/v5 evidence in hand.

Without v4/v5, I would have likely also recommended focal loss or ASL as an intervention, because the theoretical reasoning for why it should help is sound. The theory is wrong — but you can't know the theory is wrong without running the experiment.

**4. The causal chain in my root cause model shifted.**

Before v4/v5, the natural causal chain is:

```
loss function design → per-element gradient weights → gradient tier distribution → encoder monopolization → loss floor
```

After v4/v5, the chain becomes:

```
data occurrence frequency → per-batch sample counts per code → gradient tier distribution → encoder monopolization → loss floor
                                        ↑
                          (loss function is irrelevant here)
```

This shift has direct practical consequences. If the loss function were the entry point, the fix would be a better loss function (focal, ASL, per-tier weighting). Since batch composition is the entry point, the fix must change **what appears in each batch** — per-code balanced sampling, not per-element reweighting.

**5. v5 further narrowed the solution space.**

v5 adds density-aware sampling and tier-aware batching (tail_quota=20 out of 128 per batch) on top of ASL. Even with this, gradient distribution and tail accuracy are unchanged. This told me that **per-tier batch enrichment is insufficient** — 20 tail-enriched samples shared across 1,175 tail codes still gives each specific tail code ~0 gradient updates per batch.

This pushed me to recommend **per-code balanced sampling** rather than per-tier enrichment — a more specific and granular intervention that no expert explicitly recommends. Without the v5 evidence, I would not have made this distinction; I would have recommended tier-aware batching (as Expert 3 does) and considered the problem addressed.

### The Meta-Lesson

The v4/v5 evidence illustrates a general principle in ML diagnostics: **theory narrows the hypothesis space, but only controlled experiments can eliminate hypotheses.** 

All three experts reason from architecture, optimization theory, and gradient tracking data. This is high-quality theoretical reasoning that correctly identifies the shared encoder as a bottleneck. But it cannot determine *why* the encoder is monopolized — whether the cause is the loss function (Hypothesis A) or occurrence frequency (Hypothesis B) — because both predict the same gradient distribution under standard BCE.

The v4/v5 experiments are the discriminating test. They change one variable (the loss function) while holding everything else constant. The null result (gradient distribution unchanged) eliminates Hypothesis A and confirms Hypothesis B. This is the scientific method at work — and it's exactly the kind of evidence that transforms "plausible diagnosis" into "confirmed root cause."

Without v4/v5, my analysis would have been 80% the same in its conclusions but fundamentally weaker in its *confidence* and *specificity*. The v4/v5 evidence is what allowed me to: (a) definitively reject the loss function as a root cause, (b) identify a blind spot in all three expert analyses, (c) narrow the intervention recommendations to batch-composition-level changes, and (d) distinguish between per-tier enrichment (insufficient, per v5) and per-code balancing (the next logical step).