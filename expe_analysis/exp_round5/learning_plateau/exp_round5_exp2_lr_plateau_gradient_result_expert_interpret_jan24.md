- Jan 24, 2026
# Expert 1: Deep Technical Analysis: Gradient Tier Dynamics and the Learning Plateau

## (A) Primary Diagnosis: The Gradient Analysis Reveals a Fundamental Limitation of Per-Sample Loss Weighting

### The Core Finding

The gradient tier evolution data reveals a critical insight that reframes our understanding of the plateau:

**Gradients are NOT initially concentrated—they BECOME concentrated as training progresses.**

| Training Phase | Common | Medium | Rare | Tail | Mechanism |
|---------------|--------|--------|------|------|-----------|
| Step 1 | 17.8% | 27.3% | 26.5% | 17.8% | Random init; gradients proportional to tier size |
| Step 500 | 16.9% | 27.9% | 27.0% | 18.4% | Still balanced; model learning initial structure |
| Step 1500 | 42.7% | 21.9% | 17.4% | 10.4% | **Transition begins** - common codes pull ahead |
| Step 3000 | 66.7% | 16.1% | 7.1% | 3.0% | Common codes dominating |
| Step 12000 | 85.3% | 11.2% | 0.6% | 0.1% | **Terminal concentration** |

This progression reveals that the gradient concentration is an **emergent property of the learning dynamics**, not a fixed consequence of data imbalance.

---

## (B) Root Cause Analysis: Why pos_weight Cannot Prevent Gradient Concentration

### Hypothesis 1: The Sample Count × Per-Sample Contribution Product (HIGHEST CONFIDENCE)

The gradient contribution per code is:

```
Total_Gradient[code] ∝ Σ_samples (pos_weight × |p - y| × ∂p/∂θ)
```

For a code appearing in `N` samples with positive weight `W`:
- **Common code**: N = 100,000 samples, W = 1 → contribution ∝ 100,000
- **Rare code**: N = 100 samples, W = 200 → contribution ∝ 20,000

Even with 200× pos_weight, the common code STILL contributes 5× more gradient because **pos_weight scales per-sample, not per-code.**

**Evidence supporting this hypothesis:**
- Final gradient fractions: Common=84.7%, Tail=0.17%
- This is a 498× ratio, while the pos_weight ratio is at most 200×
- The sample count ratio must be filling the gap

### Hypothesis 2: The Confidence Asymmetry Effect (HIGH CONFIDENCE)

As training progresses, the model becomes **confident on common codes** (p → y) but **remains uncertain on rare codes** (p stays near class prior).

For BCE loss gradient: `∂L/∂θ ∝ (p - y)`

- **Common code correctly predicted**: p ≈ 0.8 when y=1 → gradient ∝ -0.2 (small)
- **Rare code defaulting to prior**: p ≈ 0.01 when y=1 → gradient ∝ -0.99 (large)

**Paradox**: If rare codes have larger per-sample gradients, why do they contribute LESS total gradient?

**Resolution**: The gradient is computed at the batch level. In any given batch:
- Common codes: 50-100 positive samples per batch → total gradient accumulates
- Rare codes: 0-1 positive samples per batch → gradient spikes are rare and averaged out

This explains why **gradient concentration INCREASES over training**: as the model learns, the per-sample gradient for common codes decreases, but the **batch-level gradient remains dominated by common codes** because they appear in every batch while rare codes appear sporadically.

### Hypothesis 3: The Medium Code Collapse Indicates Threshold Effect (MODERATE CONFIDENCE)

The catastrophic drop in medium_top10_acc (4.1% → 0.16%, a 96% collapse) reveals a critical phenomenon:

With higher pos_weight:
- The model becomes **less confident** on all predictions (loss landscape changes)
- Medium codes, which require **precise confidence calibration** to rank in top-10, are pushed below the threshold
- Common codes still dominate because their raw logits remain highest even with reduced confidence

**Evidence**:
- recall@5 dropped 4.9% (precision-sensitive)
- recall@50 INCREASED 0.4% (less precision-sensitive)
- micro_recall improved (per-code performance improved)
- macro_auroc improved +3.8% (class-balanced discrimination improved)

This pattern indicates: **the model is better at DISCRIMINATING codes, but worse at RANKING them in top-K**.

---

## (C) What the Evidence Definitively Shows

### Confirmed Facts:
1. **Gradient concentration is progressive, not static**: It evolves from balanced (18-27% per tier) to severely skewed (85% common)
2. **4× pos_weight increase (50→200) did NOT prevent gradient concentration**: Final common_frac was 84.7%
3. **Rare/tail codes receive effectively zero gradient signal by end of training**: 0.17% for tail, 2% for rare
4. **The concentration transition occurs between steps 500-3000**: This is when the model "learns" the common code prior
5. **Higher pos_weight HURTS medium codes disproportionately**: 96% collapse in medium accuracy

### Cannot Confirm Without Additional Experiments:
1. Whether even higher pos_weight (500, 1000) would eventually prevent concentration
2. Whether the concentration is bounded by architecture or by data
3. Whether focal loss would change the concentration dynamics

---

## (D) Mechanistic Explanation

The learning plateau is caused by a **self-reinforcing gradient concentration loop**:

```
Step 1: Model is initialized randomly
        → All codes have similar gradient contributions

Step 2: Common codes receive more positive examples per batch
        → Model begins to learn common code patterns
        → Loss for common codes decreases faster

Step 3: As loss decreases for common codes, their per-sample gradient decreases
        BUT the gradient contribution remains high because:
        (a) Common codes appear in EVERY batch
        (b) The model is refining its predictions (small corrections across many samples)
        
Step 4: Rare codes appear sporadically in batches
        → Their gradient signal is "averaged out" by common code gradients
        → Model never learns to predict them correctly when y=1
        
Step 5: By step 3000, the model has learned a "common code prior"
        → For any input, predict common codes with high probability
        → This explains 82% recall@10 (mostly common codes)
        
Step 6: The plateau represents the model's capacity being FULLY CONSUMED
        by common code representations. Adding capacity (MoE) doesn't help
        because all capacity goes to common codes due to gradient dominance.
```

---

## (E) Why Increasing pos_weight_max Further Is Unlikely to Help

The data shows that even 4× increase in pos_weight:
1. Did NOT change rare/tail accuracy (still 0%)
2. HURT medium code accuracy (-96%)
3. HURT top-K ranking metrics (recall@5 -4.9%, MRR -4.9%)

This suggests we're hitting a **structural limitation** of the BCE + pos_weight formulation:

**The issue is NOT the weight magnitude, but the weight APPLICATION MECHANISM.**

pos_weight scales the loss per sample, but:
- Rare codes have few samples to benefit from the scaling
- The gradient accumulation across batches still favors common codes
- No amount of per-sample scaling can overcome the batch-level dynamics

---

## (F) Actionable Next Directions

Based on the gradient tier analysis, I recommend these experiments in priority order:

### Priority 1: Per-Tier Gradient Normalization (Highest Information Gain)

**Hypothesis to test**: If we normalize gradients by tier AFTER accumulation, we can prevent concentration.

```python
# In training loop, after backward():
tier_grad_norms = compute_tier_grad_norms(model, tier_indices)
total_norm = sum(tier_grad_norms.values())

# Normalize each tier to contribute equally
for tier, indices in tier_indices.items():
    target_fraction = 0.25  # Equal contribution
    current_fraction = tier_grad_norms[tier] / total_norm
    scale_factor = target_fraction / max(current_fraction, 1e-8)
    model.decoder_cd.weight.grad[indices] *= scale_factor
```

**Expected outcome**: If gradient concentration is the cause, forced balanced gradients should improve rare/tail accuracy.

### Priority 2: Sampled Softmax Objective (Different Loss Formulation)

**Hypothesis to test**: BCE optimizes per-code independently; sampled softmax forces competition.

For each sample:
1. Take the K positive codes (typically 2-10)
2. Sample M negative codes (stratified: equal from each tier)
3. Compute softmax over the K+M candidates
4. This forces the model to rank positives ABOVE negatives from all tiers

**Expected outcome**: Direct ranking optimization should improve NDCG and rare code recall.

### Priority 3: Curriculum Learning with Tier-Aware Batching

**Hypothesis to test**: The concentration happens because rare codes are underrepresented in batches.

For each batch:
1. Ensure at least N samples with rare code positives
2. Ensure at least N samples with tail code positives
3. This guarantees rare/tail codes appear in every gradient update

**Expected outcome**: Consistent gradient signal for rare codes should prevent concentration.

### Priority 4: Focal Loss with Tier-Conditioned γ

**Hypothesis to test**: The γ parameter should be HIGHER for common codes (down-weight easy examples more).

```python
# Per-tier focal gamma
gamma_common = 3.0  # Strong down-weighting
gamma_medium = 2.0
gamma_rare = 1.0    # Mild down-weighting
gamma_tail = 0.5    # Almost no down-weighting
```

**Expected outcome**: This should reduce gradient contribution from "easy" common code predictions.

---

## (G) Plain-Language Summary

Your gradient tier analysis has provided definitive evidence for why the learning plateau exists:

1. **At the start of training**, gradients are balanced across code frequency tiers (roughly 20% each)
2. **As training progresses**, common codes capture increasingly more of the gradient signal
3. **By the end**, 85% of ALL gradient updates are directed toward the ~20% most common codes
4. **The remaining 80% of codes (medium/rare/tail) receive only 15% of learning signal**

Increasing `pos_weight_max` from 50 to 200 did NOT solve this because:
- pos_weight scales **per-sample**, not **per-code**
- Common codes appear in nearly every batch; rare codes appear sporadically
- The total gradient contribution = (per-sample weight) × (number of samples)
- Even 200× weight cannot overcome 1000× fewer samples

The path forward is NOT to increase pos_weight further, but to change HOW gradients accumulate:
1. **Normalize gradients per tier** after accumulation
2. **Switch to ranking-aware objectives** that force code competition
3. **Restructure batches** to guarantee rare code representation

This gradient analysis has transformed your diagnosis from "we don't know why rare codes don't learn" to "we know exactly why: they receive 0.17% of gradient signal by end of training."


# Expert 2: Gradient Starvation Dynamics and the pos_weight Intervention
## In-Depth Analysis: Gradient Starvation Dynamics and the pos_weight Intervention

### (A) What the Gradient Tier Evolution Definitively Reveals

The gradient tier analysis provides **empirical confirmation** of a critical insight that reframes the entire plateau diagnosis:

**The gradient concentration is NOT a static property of data imbalance—it is a DYNAMICALLY EMERGENT phenomenon of the learning process itself.**

The data shows a clear three-phase evolution:

| Phase | Steps | Gradient Distribution | Total Norm | Mechanistic Interpretation |
|-------|-------|----------------------|------------|---------------------------|
| **Initialization** | 0-500 | Balanced (17-27% per tier) | 530K → 25K | Random init; gradients proportional to output space structure |
| **Transition** | 500-3000 | Rapid skewing (17% → 67% common) | 25K → 1.6K | Model learning common code manifold; tail signal "averaged out" |
| **Terminal** | 3000-12000 | Severely concentrated (85%/10%/2%/1%) | ~2K-10K | Self-reinforcing loop; capacity fully consumed by common codes |

**Critical observation**: The gradient distribution was nearly balanced at initialization, which means the **pos_weight mechanism IS providing the intended per-sample scaling**. The concentration emerges DESPITE correct weighting because the weighting operates at the wrong level of aggregation.

---

### (B) Root Cause Analysis: The Per-Sample vs. Per-Code Gradient Accumulation Mismatch

The evidence points to a fundamental limitation in how pos_weight interacts with batch-level gradient accumulation:

#### Mechanism 1: The Sample Count × Per-Sample Gradient Product

The total gradient contribution for code $c$ is:

$$\nabla_\theta L_c = \sum_{i \in \text{batch}} w_c \cdot (\sigma(f_c^{(i)}) - y_c^{(i)}) \cdot \frac{\partial f_c^{(i)}}{\partial \theta}$$

Where $w_c$ is the pos_weight for code $c$. For a code appearing in $N$ samples:

| Code Type | Samples/Epoch | pos_weight | Effective Signal |
|-----------|---------------|------------|------------------|
| Common | ~100,000 | ~1 | ~100,000 |
| Tail | ~100 | ~200 | ~20,000 |

Even with 200× pos_weight, **common codes STILL contribute 5× more total gradient**. This explains why:
- `common_frac` = 84.7% at end
- `tail_frac` = 0.17% at end
- Ratio = 498× (far exceeding the 200× pos_weight ratio)

The sample count differential (1000× for extreme tail codes) overwhelms any per-sample weighting.

#### Mechanism 2: The Batch-Level Averaging Effect

In any given batch of 128 samples:
- **Common codes**: ~50-100 positive samples → consistent gradient signal every update
- **Tail codes**: ~0-1 positive samples → sporadic spikes that get averaged into noise

This explains why gradient concentration **accelerates** during training:
1. Early training: model uncertain everywhere → gradients come from everywhere
2. Mid training: model becoming confident on common codes → loss decreasing, but common codes still dominate batch-level updates
3. Late training: common codes require only "fine-tuning" gradients across many samples; rare codes appear too infrequently to accumulate meaningful signal

---

### (C) The Medium Code Collapse: A Critical Diagnostic Signal

The most informative signal in the new experiment is the **catastrophic collapse of medium_top10_acc** from 4.1% to 0.16% (a 96% drop):

| Metric | pos_weight_max=50 | pos_weight_max=200 | Interpretation |
|--------|------------------|-------------------|----------------|
| medium_top10_acc | 4.1% | 0.16% | Threshold effect |
| macro_auroc | 0.846 | 0.878 | Discrimination IMPROVED |
| micro_recall@10 | 0.462 | 0.466 | Per-code recall IMPROVED |
| recall@5 | 0.722 | 0.686 | Precision DEGRADED |

This pattern reveals a **ranking vs. discrimination decoupling**:

1. **Higher pos_weight improved per-code discrimination** (AUROC, micro_recall): The model is BETTER at saying "this code is positive" vs "this code is negative"

2. **But it degraded top-K ranking** (recall@5, MRR, medium_acc): The model is WORSE at putting the correct codes in the top-K positions

**Mechanistic explanation**: Higher pos_weight creates larger gradients for positive samples of rare codes. This pushes the model to output higher logits for those codes when they're positive. However, the model's **confidence calibration** becomes distorted—it now outputs similar-magnitude logits for both "genuinely likely common codes" and "occasionally-appearing medium codes."

When ranking by logit magnitude for top-K selection:
- With pos_weight=50: Common codes had clearly higher logits → dominated top-5, medium occasionally made top-10
- With pos_weight=200: Logits are "flatter" across tiers → common still wins (more samples = more robust logits), but medium codes got pushed below the threshold by the increased noise

This explains why **recall@20/50 improved slightly** (less precision-sensitive) while **recall@5/10 degraded** (more precision-sensitive).

---

### (D) What This Tells Us About the Loss Landscape

The gradient tier evolution reveals that the model is converging to a **"common code prior" attractor**:

```
                         Loss Landscape Visualization
                         
    High Loss                                                  
        │                              ┌─────────────────┐
        │    "Common Code Prior"  ────►│ Global Minimum  │
        │         Attractor            │ R@10 ≈ 0.83     │
        │            ▼                 │ Tail = 0%       │
        │    ┌──────●──────┐           └─────────────────┘
        │    │             │                    
        │    │             │           ┌─────────────────┐
        │    │             │           │ Hypothetical    │
        │    │             │      ────►│ Better Minimum  │
        │    │             └──────────►│ R@10 ≈ 0.87     │
        │    │  Basin of Attraction   │ Tail > 0%       │
        │    │                        └─────────────────┘
    Low Loss
```

The key insight: **The current optimization path is gradient-stable toward the common-code attractor**, and no amount of per-sample weighting can escape this basin because:

1. The attractor is "correct" from BCE's perspective—it minimizes average per-sample loss
2. Rare codes contribute too little cumulative signal to steer the model elsewhere
3. MoE/capacity increases just expand the attractor basin (more parameters encoding common codes)

---

### (E) Hypothesis Refinement: What We Now Know vs. Suspect

#### CONFIRMED by evidence:
1. Gradient concentration is progressive and accelerates during training
2. pos_weight=200 does NOT prevent gradient concentration (84.7% common at end)
3. Higher pos_weight DEGRADES precision metrics while IMPROVING discrimination metrics
4. The transition phase (steps 500-3000) is when gradient concentration becomes severe
5. Rare/tail accuracy remains 0% regardless of pos_weight value tested

#### SUPPORTED but not yet confirmed:
1. **The per-sample vs. per-code aggregation mismatch is the root cause**
   - Distinguishing test: Implement per-code gradient normalization and observe if concentration is prevented
   
2. **Sampled softmax would circumvent this issue**
   - Mechanism: Forces ranking competition between codes at training time
   - Distinguishing test: Compare gradient tier distribution under sampled softmax vs. BCE

3. **The capacity is being consumed by common codes, not limited overall**
   - Evidence: MoE (35M params) ≈ Dense (25M params) in final metrics
   - Further test: Freeze common code embeddings after step 3000 and continue training rare codes only

#### REFUTED by evidence:
1. ~~"pos_weight is insufficient at 50, increase to 200 will help"~~ → 200 made precision WORSE
2. ~~"Gradient starvation is a static property of data imbalance"~~ → It's dynamically emergent
3. ~~"Rare codes receive no gradient"~~ → They receive gradient initially (17-27%), but lose it progressively

---

### (F) Key Learnings and Implications

#### Learning 1: Per-Sample Weighting Has a Ceiling
There's a fundamental limit to what pos_weight can achieve. Beyond a certain value, you're fighting against batch statistics, not improving tail learning. The 96% medium collapse is evidence of overshooting.

#### Learning 2: The Metric Tradeoff Reveals Loss-Metric Misalignment
The simultaneous improvement in AUROC (+3.8%) and degradation in recall@5 (-4.9%) proves that BCE loss is NOT aligned with top-K ranking objectives. The loss can improve while business metrics degrade.

#### Learning 3: The Gradient Concentration Timeline Is Predictable
The transition phase (steps 500-3000) is where intervention is most critical. Any solution needs to act BEFORE this phase completes.

#### Learning 4: The Problem Is Gradient Aggregation, Not Weighting
The path forward requires changing HOW gradients accumulate across codes, not just scaling per-sample contributions.

---

### (G) Recommended Next Directions (Prioritized by Information Gain)

#### Direction 1: Per-Tier Gradient Normalization (Highest Priority)

**Rationale**: Directly tests the gradient aggregation hypothesis by forcing balanced tier contributions.

```python
# After loss.backward():
tier_norms = {tier: 0.0 for tier in ['common', 'medium', 'rare', 'tail']}
for tier, indices in tier_index_map.items():
    tier_norms[tier] = model.decoder_cd.weight.grad[indices].norm()

# Normalize to equal contribution
target_frac = 0.25
for tier, indices in tier_index_map.items():
    scale = (target_frac * sum(tier_norms.values())) / max(tier_norms[tier], 1e-8)
    model.decoder_cd.weight.grad[indices] *= scale
```

**Expected outcome**: If gradient concentration is causal, this should:
- Maintain gradient tier balance throughout training
- Improve rare/tail accuracy from 0% to measurable levels
- Potentially degrade common code performance (acceptable tradeoff for diagnosis)

#### Direction 2: Sampled Softmax with Tier-Stratified Negatives

**Rationale**: Circumvents BCE's per-code independence assumption by forcing ranking competition.

For each sample:
1. Positives: the 2-10 codes with y=1
2. Negatives: sample 50 from each tier (200 total)
3. Loss = CrossEntropy over the 2-10+200 candidates

**Expected outcome**: Every gradient update forces the model to rank positives above negatives from ALL tiers. This directly optimizes the ranking objective.

#### Direction 3: Two-Phase Training with Tier-Aware Curriculum

**Phase 1 (steps 0-3000)**: Train normally to learn common code structure
**Phase 2 (steps 3000+)**: 
- Freeze common code decoder weights
- Apply 10× learning rate to rare/tail decoder weights
- Use tail-boosted batching (ensure 20% of batch has rare positives)

**Rationale**: Exploits the insight that the transition happens at steps 500-3000. After common codes are learned, redirect capacity to tail.

#### Direction 4: Contrastive Loss for Tail Code Embeddings

**Rationale**: If the model "doesn't see" tail codes often enough, create synthetic supervision.

For each tail code c:
1. Find its nearest neighbors in embedding space
2. Create contrastive pairs: (anchor=c, positive=similar codes, negative=dissimilar codes)
3. Add auxiliary contrastive loss to spread tail codes in embedding space

This ensures tail codes have meaningful representations even without abundant positive samples.

---

### (H) What NOT to Do (Based on Evidence)

1. **Do NOT increase pos_weight_max further (e.g., to 500)**
   - Evidence shows diminishing returns and metric degradation
   - The medium code collapse (-96%) is a warning sign

2. **Do NOT add more model capacity (d_model=512, more layers)**
   - MoE (35M) ≈ Dense (25M) proves capacity is NOT the bottleneck
   - More capacity will just encode more common code variations

3. **Do NOT train for more epochs without addressing gradient dynamics**
   - The concentration is self-reinforcing
   - More training = more concentration = worse tail performance

4. **Do NOT trust loss improvements as a proxy for business metrics**
   - Loss decreased from 0.0037 to 0.0034 with pos_weight=200
   - But precision metrics degraded
   - This decoupling is the core signal of objective-metric misalignment

---

### (I) Executive Summary

Your gradient tier analysis has provided definitive empirical evidence for the learning plateau mechanism:

**The Root Cause**: Gradient contributions evolve from balanced (17-27% per tier) to severely concentrated (85% common, 0.17% tail) because pos_weight operates at the per-sample level while gradient accumulation occurs at the per-code level across batches. Common codes appear in every batch, accumulating consistent signal; rare codes appear sporadically, their weighted spikes averaging into noise.

**The pos_weight=200 Experiment Revealed**:
1. Increasing pos_weight cannot overcome the sample-count differential
2. Higher weighting distorts confidence calibration, hurting precision metrics
3. Medium codes are particularly vulnerable to this threshold effect (-96% accuracy)
4. The improvement in AUROC (+3.8%) while recall@5 degraded (-4.9%) proves BCE-metric misalignment

**The Path Forward** requires changing the gradient aggregation mechanism, not the per-sample weighting:
1. Per-tier gradient normalization (force balanced tier contributions)
2. Sampled softmax (directly optimize ranking, not calibration)
3. Two-phase curriculum (learn common, then redirect to tail)

This analysis transforms the diagnosis from "hypothesis: gradient starvation might cause the plateau" to "confirmed: 85% of gradients go to 20% of codes by training end, and no per-sample weighting can overcome this batch-level dynamic."


# Expert 3: Gradient Starvation Dynamics and the pos_weight Intervention

### (A) What I see (non-redundant, interpretation-only)
Your `pos_weight_max` increase did **not** “unlock” the long tail; instead it **re-shaped** learning so that **global separability** (macro AUROC/AUPRC) improved while **top‑K ordering quality** (MRR/NDCG/Recall@5/10) degraded, and the newly-instrumented gradient tiers show an **emergent collapse of learning signal into common codes over time** (tail ends near-zero share of gradient by late training).

### (B) Primary hypothesis (ranked; mechanism-first)
**H1 — The plateau is an emergent “gradient capture” fixed point of BCE-on-imbalanced-multilabel + your batch exposure, and `pos_weight` cannot change the attractor because it scales *per positive event*, not *per code’s total learning opportunity*.**  
Mechanism:

- With BCE, *total parameter update pressure per code* is roughly proportional to **(positive events seen) × (per-event weight) × (error term)**. You can increase the per-event weight (your `pos_weight_max`), but you did **not** change the **event count per step** for rare/tail codes.
- Once the model learns a decent head-code prior, training enters a regime where the optimizer spends most of its “update budget” on **tiny corrections across many head events**. Those tiny corrections still dominate the *sum* of gradients because head events are ubiquitous; rare events remain too sporadic to accumulate comparable integrated gradient.
- Your gradient-tier trace is exactly the signature of this: **initially balanced** (random init → gradients look like “tier size”), then a **rapid phase transition** where head gradients become a larger fraction, then a **late regime** where tail receives effectively no training signal. That is a structural reason the tail metrics stay flat even when you crank weights.

**H2 — Increasing `pos_weight_max` worsened top‑K ranking because it changed the geometry of logits (and their calibration) in a way that improves class-wise discrimination but reduces *margin* among the most competitive candidates.**  
Mechanism (consistent with your metric pattern):

- AUROC/AUPRC can improve when positives are, on average, scored higher than negatives (better separability), even if the **relative ordering among the top few hundred candidates** becomes worse.
- A higher positive weight pushes the model to reduce FN penalty; in multilabel settings this often **broadens** the score distribution (more codes get “somewhat high” scores), which can **hurt NDCG/MRR/Recall@5/10** because the very top positions become less “sharp,” while Recall@50 and micro-recall can improve because positives are still placed *somewhere* higher.

**H3 — The “medium tier collapse” is a symptom of the same effect: the model’s capacity/updates get re-allocated such that the middle-frequency band loses margin in the crowded top‑10 competition, while the head stays dominant and the tail remains invisible.**  
Mechanistically: medium codes need both (i) enough exposure and (ii) enough margin vs head negatives; if training shifts toward broad, conservative separability rather than sharp ranking margins, medium can be the first to fall out of top‑K.

### (C) Competing hypotheses (and what would falsify them)
- **C1 — Implementation/eval artifact** (tier boundaries, metric computation, or weight application changed in subtle ways).  
  - **Falsify/confirm**: rerun `pos_weight_max=50` with the **same gradient-tier instrumentation**, ensure identical tier definitions and evaluation path; verify the medium collapse persists only with the weight change.
- **C2 — Numerical/optimization instability induced by large weights in FP16** (gradient scaling/clipping interactions), causing poorer top‑K despite similar BCE.  
  - **Falsify/confirm**: compare grad-norm/clipping fraction/update-to-weight ratio between 50 vs 200; run BF16/FP32 loss accumulation or tighter clipping—if top‑K recovers without changing sampling/objective, it’s an optimizer/numerics issue.
- **C3 — True information ceiling / label noise for tail** (rare codes are not predictable from available features or labels are too noisy).  
  - **Falsify/confirm**: restrict to a “learnable tail” subset (min-positive-count threshold, higher label quality slice) and see if tail improves under tail-aware exposure; if not, ceiling/noise dominates.
- **C4 — Representation bottleneck (e.g., `d_model=256`)** is the real limit.  
  - **Falsify/confirm**: only worth testing *after* fixing exposure/objective; if a ranking-aware + tail-aware regimen still plateaus, then widen `d_model`/depth.

### (D) Decisive experiments (minimal set; each disambiguates cleanly)
1. **Counterfactual: same run, force tail exposure without changing loss**
   - **Change**: tier-aware batch construction (or oversampling) so each batch contains a fixed quota of rare/tail-positive samples.
   - **Measure**: does tail gradient fraction stay bounded away from ~0 late in training? do rare/tail top‑K metrics move off zero?
   - **Interpretation**: if this works, the core issue is **exposure**, not capacity.

2. **Change the learning signal from “independent BCE” to “competition/ranking”**
   - **Change**: sampled-softmax / in-batch negatives / pairwise ranking loss with **tier-stratified negatives** (so tail competes against head).
   - **Measure**: NDCG/MRR/Recall@10 and tail coverage; also whether “medium collapse” reverses.
   - **Interpretation**: if top‑K improves materially while BCE-ish loss might not, the plateau was **objective-metric misalignment**.

3. **Post-backward gradient rebalancing (surgical test of the gradient-capture hypothesis)**
   - **Change**: after `backward()`, rescale decoder gradients by tier to enforce target contributions (e.g., 25% each) for a short run.
   - **Measure**: does tail learning begin quickly (within a few thousand steps)? does head performance trade off?
   - **Interpretation**: if tail starts moving immediately, you’ve proven the bottleneck is **where updates go**, not architecture.

4. **Numerics control experiment**
   - **Change**: keep `pos_weight_max=200`, add explicit gradient clipping + FP32 loss accumulation (or BF16).
   - **Measure**: if top‑K metrics recover while gradient-tier collapse remains, then `pos_weight` was harming ranking mostly via **optimization noise**, not via fundamental exposure.

### (E) Action plan (concrete next steps + expected outcomes + risks + rollback)
- **Step 1 (measurement sanity)**: run a short `pos_weight_max=50` job with the same gradient-tier logging to get an apples-to-apples gradient trajectory baseline.  
  - **Expected**: you’ll learn whether the “late collapse to common” is inherent (likely) vs weight-induced.  
  - **Risk**: none; cheap.

- **Step 2 (highest-leverage)**: implement **tier-aware batching** (guarantee rare/tail positives appear every step).  
  - **Expected**: tail gradient fraction stops collapsing; tail coverage/rare recall should move above zero if learnable.  
  - **Risk**: may reduce head metrics; rollback by lowering quotas.

- **Step 3 (alignment)**: add a **ranking/competition component** (sampled-softmax or pairwise loss) with tier-stratified negatives.  
  - **Expected**: NDCG/MRR recover/improve; medium band should stop collapsing; top‑K improves at constant compute.  
  - **Risk**: more tuning; rollback by blending with BCE (small weight) first.

- **Step 4 (only if needed)**: consider capacity (`d_model`, depth) *after* Steps 2–3, because otherwise extra capacity just gets spent on the head again.

### (F) Plain-language explanation
Raising `pos_weight_max` changed “how hard the model tries” on positives, but it didn’t change the core fact that **rare/tail positives almost never show up in a given gradient step**, so training naturally evolves into a state where nearly all learning effort goes into refining common codes. That’s why the tail stays stuck and why you can even see overall discrimination improve while the *top‑10 ranking* gets worse: the model is learning “better separation in general,” not “the right winners in the top few slots,” and the gradient budget keeps getting captured by the head.

If you share (or point me to) the exact tier definitions + how batches are formed (random, bucketed by length, any stratification), I can make the “tier-aware batching” and “sampled-softmax” proposals maximally specific to your pipeline.


# Expert 4: Root Cause of Learning Plateau & Gradient Starvation

## 1. Executive Summary: The "Volume Dominance" Mechanism

The new evidence from the `pos_weight_max=200` experiment provides the "smoking gun" we were missing. The plateau is not due to model capacity, nor is it a simple case of "not enough weight."

**The Root Cause is "Volume Dominance":**
The gradient tier analysis reveals that gradient contribution is an **emergent property** of training dynamics.
*   **Start of Training:** Gradients are balanced (~20% per tier). The model knows nothing, so errors are uniform.
*   **End of Training:** Common codes capture **85%** of the gradient mass, while Tail codes receive **0.17%**, *despite* the model achieving high accuracy on Common codes and 0% on Tail codes.

This contradicts standard intuition. Typically, well-learned examples (Common) should produce *small* gradients (`p ≈ y`), and unlearned examples (Tail) should produce *large* gradients (`p << y`). The fact that Common codes *still* dominate means the **sheer volume of Common samples** (`N_common`) multiplied by even small residual errors overwhelms the weighted contribution of Rare samples (`N_rare × Weight`).

**Conclusion:** You cannot fix this with `pos_weight`. The class imbalance ratio ($10^4:1$) is larger than any stable scalar weight you can apply. You must intervene at the **gradient accumulation** or **objective** level.

---

## 2. In-Depth Analysis of the Evidence

### A. The "Gradient Illusion" (Time-Dependent Concentration)
The most critical finding is the temporal evolution of gradient tiers:
*   **Step 1-500:** The "Golden Era" where the model *could* learn rare codes. Gradients are balanced.
*   **Step 500-3000:** The "Takeover." Common code gradients swell from 17% to 67%.
*   **Step 3000+:** The "Starvation." The model solidifies the Common priors. Tail gradients vanish to <1%.

This explains why `pos_weight=200` failed. By the time the model settles, the optimization landscape is entirely shaped by the top 20% of codes. The rare codes are effectively "noise" to the optimizer—sporadic, high-variance spikes that get averaged out by the constant flood of common code updates.

### B. The "Medium Code Collapse" Pathology
Increasing `pos_weight_max` from 50 to 200 caused `medium_top10_acc` to collapse from **4.1% to 0.16%**.
*   **Mechanism:** By amplifying the weights of positive samples globally (or near-globally for non-common codes), you likely increased the variance of the gradients.
*   **Result:** The "Medium" codes, which were marginally learnable (on the decision boundary), were pushed into instability. The model became "confused" or the loss landscape became too jagged, causing it to default to the safer, stronger signal: the Common codes.
*   **Implication:** Simply cranking up weights is destructive. It destabilizes the "middle class" of codes without saving the "tail."

### C. The "Micro vs. Macro" Divergence
*   **Micro Recall improved (+0.7%)**: The model got better at predicting the *occurrences* of codes it already knows.
*   **Recall@5 dropped (-4.9%)**: The ranking quality at the very top deteriorated.
*   **Interpretation:** The model learned to be "safer" but less "precise" in ranking. It predicts the "sure things" (Common codes) with slightly higher probability, pushing them up the list, but loses the nuance needed for the top-5 slots.

---

## 3. Hypotheses Validated & Rejected

| Hypothesis | Status | Evidence |
| :--- | :--- | :--- |
| **H1: Capacity Limited** | **REJECTED** | Common codes consume 85% of gradients even when "learned." The model is obsessed with refining Common codes, not out of capacity. |
| **H2: Weight Insufficient** | **REJECTED** | `pos_weight=200` (4x increase) didn't move the needle on Tail acc. The mechanism is structural, not scalar. |
| **H3: Gradient Starvation** | **CONFIRMED** | Tail codes receive 0.17% of total gradient norm by end of training. |
| **H4: Volume Dominance** | **CONFIRMED** | $N_{common} \times \epsilon_{small} \gg N_{rare} \times E_{large} \times W$. The sum of many small errors > sum of few large errors. |

---

## 4. Proposed Solutions: Structural Interventions

Since `pos_weight` is mathematically insufficient to overcome the volume ratio, we must change the *rules* of accumulation or the *objective* itself.

### Intervention 1: Gradient Normalization by Tier (The "Socialist" Gradient)
Force the optimizer to respect tiers equally, regardless of sample count.

**Mechanism:**
Instead of `loss.backward()`, we manually scale gradients.
1.  Compute gradients.
2.  Calculate norm per tier ($G_{common}, G_{medium}, G_{rare}, G_{tail}$).
3.  Scale parameters for each tier $t$ by $\frac{1}{G_t}$. (Conceptually—implementation requires masking or hooks).
4.  Result: Common and Tail tiers contribute equal magnitude updates.

**Implementation Logic (PyTorch Hook approach):**
You can register a backward hook on the embedding/decoder layer.
```python
# Conceptual logic
target_norm = total_norm / 4.0
scale_common = target_norm / (grad_norm_common + 1e-6)
scale_tail = target_norm / (grad_norm_tail + 1e-6)
# Apply scaling to the specific indices of the gradient tensor
grad[common_indices] *= scale_common
grad[tail_indices] *= scale_tail
```

### Intervention 2: Sampled Softmax / Ranking Loss (The "Competition" Objective)
BCE treats every code as an independent Bernouilli trial. This allows the model to say "Yes" to Common and "No" to Rare without penalty for the relative ranking.

**Mechanism:**
Sampled Softmax forces the model to pick the *correct* code from a set of negatives.
1.  Positive: The actual next codes.
2.  Negatives: A sample of 100 codes, *heavily stratified* to include Rare/Tail codes.
3.  Objective: Maximize log-prob of Positive vs. Negatives.

This forces the model to distinguish "Rare Code X" (True) from "Common Code Y" (False), directly attacking the ranking problem.

### Intervention 3: Stratified Batching (The "Quota" System)
The "Variance" problem (Hypothesis 2B) is that Rare codes appear so infrequently that their gradient direction is unstable.

**Mechanism:**
Construct batches such that *every* batch contains at least $K$ examples of Rare/Tail codes. This ensures a consistent, non-vanishing gradient signal for the tail throughout training, preventing the "Step 500-3000" takeover.

---

## 5. Action Plan

**Recommendation:** Do NOT run another `pos_weight` ablation. It is a dead end.

**Step 1: Implement Tier-Based Gradient Normalization (High Priority)**
*   **Why:** It directly addresses the diagnosed "Volume Dominance" root cause.
*   **Effort:** Moderate (requires custom gradient manipulation in the training loop).
*   **Success Metric:** `tail_top10_acc` > 1%. `train_grad_tier_tail_frac` ≈ 0.25.

**Step 2: Implement Sampled Softmax (Medium Priority)**
*   **Why:** It aligns the objective with the metric (Ranking) and naturally handles large output spaces better than massive weighted BCE.
*   **Effort:** High (requires changing loss function and data pipeline for negative sampling).

**Step 3: Analyze "Easy" vs "Hard" Positives**
*   **Why:** To confirm if "Medium" code collapse was due to loss instability.
*   **Action:** Look at the logits of Medium codes. Are they shrinking (p -> 0) or oscillating?

### Summary for Next Step
The "Gradient Tier Analysis" was the correct diagnostic. It proved that **gradient starvation is an emergent dynamic of training volume**. The next logical step is to **force gradient equality** artificially, rather than hoping weights will balance the scales naturally overwhelming volume of common data.


# Expert 5: Adjudicator: Self-reflection 

## Executive Overview

I have conducted a rigorous, independent analysis of the four expert opinions on the gradient tier analysis results. This review evaluates the quality of reasoning, identifies flaws, and synthesizes the unique and common elements across all perspectives.

---

## Part 1: Individual Expert Critical Analysis

### Expert 1: Deep Technical Analysis

**Strengths:**
1. Correctly identifies the **emergent nature** of gradient concentration (not static from initialization)
2. Provides clear mechanistic explanation with the three-phase evolution model
3. The "Confidence Asymmetry Effect" hypothesis is mathematically grounded

**Flaws & Weak Arguments:**

| Issue | Evidence | Severity |
|-------|----------|----------|
| **Speculative math claim** | "Common code STILL contributes 5× more gradient" - uses arbitrary numbers (N=100,000 vs N=100) without citing actual sample counts from the data | Medium |
| **Missing precision** | Claims "sample count ratio must be filling the gap" but doesn't calculate actual code frequency distributions from config | Medium |
| **Hypothesis 3 (Medium Collapse)** | Explanation of "threshold effect" is plausible but untested - claims "model becomes less confident on all predictions" without evidence from logit distributions | Medium |
| **Priority 1 recommendation lacks detail** | Per-tier gradient normalization is proposed but the `target_fraction = 0.25` is arbitrary - no justification for why equal contribution is optimal | Low |

**Unique Contribution:**
- The "Confidence Asymmetry Effect" - explaining why rare codes with larger per-sample gradients (p≈0.01 when y=1) still lose the total gradient war due to batch-level averaging

---

### Expert 2: Gradient Starvation Dynamics

**Strengths:**
1. Most rigorous mathematical formulation with LaTeX notation
2. Explicit loss landscape visualization provides intuitive understanding
3. Clear "CONFIRMED / SUPPORTED / REFUTED" framework for hypothesis tracking

**Flaws & Weak Arguments:**

| Issue | Evidence | Severity |
|-------|----------|----------|
| **Overconfident in mechanism claims** | States "The per-sample vs. per-code aggregation mismatch is the root cause" but labels it only "SUPPORTED" - inconsistent confidence levels | Low |
| **Loss landscape diagram is speculative** | The "hypothetical better minimum" with "R@10 ≈ 0.87, Tail > 0%" is fabricated - no evidence such a minimum exists or is reachable | Medium |
| **Missing critical analysis of pos_weight=50 baseline** | The comparison assumes the gradient tier evolution is similar at pos_weight=50, but no data is shown for this baseline run | High |
| **"Two-Phase Training" proposal lacks evidence** | Claims "the transition happens at steps 500-3000" but proposes freezing common codes at step 3000 - what if the transition timing varies with other hyperparameters? | Medium |

**Unique Contribution:**
- Explicit "REFUTED" list that provides clear falsification of prior hypotheses:
  - ~~"pos_weight is insufficient at 50, increase to 200 will help"~~
  - ~~"Gradient starvation is a static property"~~
  - ~~"Rare codes receive no gradient"~~

---

### Expert 3: Gradient Starvation Dynamics (Duplicate Title)

**Strengths:**
1. **Most rigorous competing hypothesis framework** with explicit falsification criteria for each
2. Proposes **tier-aware batching** as a distinct intervention (not just gradient manipulation)
3. Acknowledges implementation-level concerns (C1: "Implementation/eval artifact")
4. Asks for batch formation details - showing awareness of pipeline dependencies

**Flaws & Weak Arguments:**

| Issue | Evidence | Severity |
|-------|----------|----------|
| **Hypothesis C2 (numerical instability) is speculative** | Claims "gradient scaling/clipping interactions" in FP16 could cause issues, but the observation file shows loss trajectory is nearly identical between experiments | Medium |
| **"Event count per step" claim is imprecise** | States you didn't change "event count per step" for rare codes, but with random batching, event count is stochastic - this should be "expected event count" | Low |
| **Step 1 recommendation is redundant** | Proposes running pos_weight=50 with gradient tier logging, but this is just re-running the original experiment with instrumentation - could be combined with other interventions | Low |
| **Missing quantitative threshold** | Proposes "tier-aware batching" but doesn't specify what quota of rare/tail samples per batch would be sufficient | Medium |

**Unique Contribution:**
- **Explicit falsification criteria** for each competing hypothesis (C1-C4)
- Recognition that **"tier-aware batching"** (changing sample exposure) is distinct from gradient manipulation (changing how gradients are aggregated)

---

### Expert 4: Volume Dominance Mechanism

**Strengths:**
1. Clear naming: "Volume Dominance" is intuitive and accurate
2. "Gradient Illusion" framing - well-learned examples *should* produce small gradients but still dominate
3. Practical "Socialist Gradient" metaphor makes the intervention accessible

**Flaws & Weak Arguments:**

| Issue | Evidence | Severity |
|-------|----------|----------|
| **Mathematical notation error** | Uses $N_{common} \times \epsilon_{small} \gg N_{rare} \times E_{large} \times W$ but never defines $\epsilon$ vs $E$ - confusing notation | Low |
| **"REJECTED" for H1 (Capacity Limited) is premature** | Evidence that "Common codes consume 85% of gradients even when 'learned'" doesn't prove capacity isn't limited - it proves gradient allocation is skewed, which is a different claim | Medium |
| **Step 3 recommendation is vague** | "Analyze 'Easy' vs 'Hard' Positives" - no concrete metric or threshold defined for what constitutes easy/hard | Low |
| **Missing timeline for interventions** | Proposes three interventions but doesn't specify order or compute budget constraints | Low |

**Unique Contribution:**
- The **"contradiction"** insight: in standard optimization, well-learned examples should produce small gradients, but the volume effect makes them still dominate
- The "Golden Era" framing (steps 1-500) as a critical intervention window

---

## Part 2: Cross-Expert Synthesis

### 2.1 Points of Universal Agreement

| Claim | E1 | E2 | E3 | E4 | Validity |
|-------|----|----|----|----|----------|
| Gradient concentration is **emergent**, not static | ✓ | ✓ | ✓ | ✓ | **VALID** - data shows 17-27% → 85% evolution |
| pos_weight=200 **failed** to prevent concentration | ✓ | ✓ | ✓ | ✓ | **VALID** - tail_frac=0.17% at end |
| Per-sample weighting **cannot overcome** sample count differential | ✓ | ✓ | ✓ | ✓ | **VALID** - logical given the math |
| Rare/tail accuracy **remained 0%** | ✓ | ✓ | ✓ | ✓ | **VALID** - from data |
| Medium code collapse (-96%) is a **warning sign** | ✓ | ✓ | ✓ | ✓ | **VALID** - from data |

### 2.2 Points of Disagreement or Divergence

| Topic | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|-------|----------|----------|----------|----------|
| **Primary intervention** | Per-tier gradient normalization | Gradient normalization + sampled softmax | Tier-aware batching | Gradient normalization |
| **Secondary intervention** | Sampled softmax | Two-phase training | Sampled softmax | Sampled softmax |
| **Capacity diagnosis** | Not bottleneck | Not bottleneck | Open (C4) | REJECTED |
| **Numerical issues** | Not mentioned | Not mentioned | Possible (C2) | Not mentioned |
| **Implementation artifacts** | Not mentioned | Not mentioned | Possible (C1) | Not mentioned |

### 2.3 Unique Contributions Per Expert

| Expert | Unique Insight |
|--------|----------------|
| **Expert 1** | Confidence Asymmetry Effect - explaining the paradox of why rare codes with larger per-sample gradients lose |
| **Expert 2** | Explicit hypothesis tracking with CONFIRMED/SUPPORTED/REFUTED framework; loss landscape visualization |
| **Expert 3** | Tier-aware batching as distinct from gradient manipulation; explicit falsification criteria for competing hypotheses |
| **Expert 4** | "Volume Dominance" naming; the "Golden Era" concept (steps 1-500 as intervention window); "contradiction" insight |

---

## Part 3: Critical Gaps Across All Experts

### Gap 1: No Baseline Gradient Tier Data

**Problem**: All experts interpret the pos_weight=200 gradient evolution, but none have gradient tier data from pos_weight=50 for comparison.

**Implication**: We cannot definitively say whether:
- The gradient concentration pattern is inherent to training dynamics, OR
- The concentration is accelerated/modified by pos_weight=200

**Expert 3 correctly identifies this** (Step 1 recommendation), but others assume the pattern is universal.

### Gap 2: Tier Boundary Definition

**Problem**: No expert questions how "common", "medium", "rare", and "tail" tiers are defined.

**Implication**: The 96% medium collapse could be:
- A real phenomenon, OR
- An artifact of where tier boundaries are drawn (e.g., if top-10 prediction threshold happens to align with tier boundaries)

### Gap 3: Code Frequency Distribution

**Problem**: Experts assume a power-law distribution but don't cite actual code frequencies from the dataset.

**Implication**: The "1000× sample count differential" claims are plausible but unverified. The actual ratio could be higher or lower.

### Gap 4: Logit Distribution Analysis

**Problem**: No expert examines the actual logit values for medium/rare/tail codes.

**Questions left unanswered**:
- Are rare code logits near zero (model never activates them)?
- Are rare code logits just below threshold (ranking problem)?
- Are rare code logits oscillating (training instability)?

### Gap 5: Per-Code Learning Dynamics

**Problem**: Gradient tier analysis aggregates by tier. No expert proposes per-code gradient tracking.

**Implication**: Within the "tail" tier, there may be:
- Some codes that are learning (gradient > 0)
- Some codes that are completely dead (gradient ≈ 0)

Aggregating obscures this heterogeneity.

---

## Part 4: Flawed Reasoning Patterns

### Pattern 1: Circular Reasoning on Capacity

**Observed in**: Experts 1, 2, 4

**The Argument**:
1. MoE (35M params) ≈ Dense (25M params) in metrics
2. Therefore, capacity is NOT the bottleneck

**The Flaw**:
This only proves that **adding capacity via MoE didn't help**. It doesn't prove capacity isn't limited. MoE capacity goes unused because gradients are concentrated on common codes. If you fixed gradient concentration, MoE might then outperform dense.

**Correct conclusion**: Capacity increase **without gradient rebalancing** doesn't help. Capacity may still be a secondary bottleneck.

### Pattern 2: Post-Hoc Mechanistic Attribution

**Observed in**: All experts

**The Pattern**: After seeing gradient concentration data, experts construct mechanistic explanations that fit the data. These are **consistent with** the data but not **proven by** it.

**Example**:
- Expert 1: "Batch-level averaging causes rare code gradients to be 'averaged out'"
- Expert 4: "Volume of common samples overwhelms weighted rare samples"

Both explanations fit the data, but neither is experimentally verified. The true mechanism could be different.

### Pattern 3: Assuming Interventions Will Work

**Observed in**: All experts

**The Pattern**: All propose "per-tier gradient normalization" with high confidence that it will improve tail accuracy.

**The Risk**: Forcing equal gradient contribution could:
- Destroy common code learning (catastrophic interference)
- Amplify noise from tail codes (unstable training)
- Create gradient magnitude mismatches between tiers

No expert discusses what happens if this intervention **fails** or how to detect failure early.

---

## Part 5: Synthesis and Adjudication

### 5.1 What Is Definitely True (Empirically Verified)

1. **Gradient concentration evolves from balanced to skewed** - Data shows 17-27% → 85% common by end
2. **pos_weight=200 did not prevent concentration** - Tail received 0.17% of gradients
3. **Medium codes collapsed** - 4.1% → 0.16% accuracy
4. **Rare/tail accuracy stayed at 0%** - Unchanged between experiments
5. **Macro AUROC improved (+3.8%)** while recall@5 degraded (-4.9%) - Loss-metric misalignment exists

### 5.2 What Is Plausible But Unverified

1. The "per-sample vs per-code" aggregation mismatch explanation
2. The "volume dominance" mechanism
3. That tier-aware batching or gradient normalization will fix the problem
4. That the transition at steps 500-3000 is consistent across configurations

### 5.3 What Is Missing

1. Baseline gradient tier evolution at pos_weight=50
2. Per-code logit distribution analysis
3. Tier boundary sensitivity analysis
4. Actual code frequency distribution from data
5. Failure modes for proposed interventions

### 5.4 Recommended Priority Order for Next Experiments

Based on **information gain per compute cost**:

| Priority | Experiment | Rationale |
|----------|------------|-----------|
| **1** | Run pos_weight=50 with gradient tier logging | Establishes baseline; cheap; Expert 3 correctly identifies this gap |
| **2** | Analyze logit distributions by tier | Zero-cost (use existing model); answers "is the model even trying?" |
| **3** | Tier-aware batching | Tests whether exposure alone can prevent concentration; less invasive than gradient manipulation |
| **4** | Per-tier gradient normalization | Most directly addresses diagnosed cause; but invasive and may have unintended consequences |
| **5** | Sampled softmax | Highest implementation cost; addresses objective-metric misalignment; save for later |

---

## Part 6: Final Assessment

### Overall Quality of Expert Analysis

| Expert | Rigor | Unique Value | Practical Actionability | Critical Gaps Identified |
|--------|-------|--------------|-------------------------|--------------------------|
| Expert 1 | High | Medium | High | Low |
| Expert 2 | High | Medium | Medium | Low |
| Expert 3 | Highest | High | High | Highest |
| Expert 4 | Medium | High | Medium | Low |

### Key Takeaway

All four experts converge on the same core diagnosis: **gradient concentration is emergent and per-sample weighting cannot overcome it**. This convergence increases confidence in the diagnosis.

However, all experts share a common blind spot: **they propose interventions without establishing the baseline gradient trajectory at pos_weight=50**. Before implementing gradient normalization or tier-aware batching, the immediate next step should be confirming that the concentration pattern is intrinsic to the training dynamics, not an artifact of the pos_weight=200 setting.

**The most critical missing analysis is the per-code logit distribution** - this would reveal whether the model is:
- Never activating rare codes (logits ≈ 0) → representation problem
- Activating rare codes below threshold → ranking/calibration problem  
- Oscillating on rare codes → training instability problem

Each scenario implies a different intervention.


# The third party observation and insights: Systematic Analysis: pos_weight_max=35 (Baseline) vs pos_weight_max=200

## 1. COMPREHENSIVE METRIC COMPARISON

### 1.1 Final Evaluation Metrics (Side-by-Side)

| Metric | pos_weight_max=35 | pos_weight_max=200 | Delta | % Change | Winner |
|--------|-------------------|-------------------|-------|----------|--------|
| **recall@5** | 0.6856 | 0.6861 | +0.0005 | +0.07% | ~Equal |
| **recall@10** | 0.8142 | 0.8171 | +0.0029 | +0.36% | ~Equal |
| **recall@20** | 0.8915 | 0.8930 | +0.0015 | +0.17% | ~Equal |
| **recall@50** | 0.9506 | 0.9512 | +0.0006 | +0.06% | ~Equal |
| **micro_recall@10** | 0.4634 | 0.4656 | +0.0022 | +0.47% | ~Equal |
| **micro_recall@20** | 0.5849 | 0.5844 | -0.0005 | -0.09% | ~Equal |
| **ndcg@10** | 0.3923 | 0.3898 | -0.0025 | -0.64% | ~Equal |
| **ndcg@20** | 0.4298 | 0.4265 | -0.0033 | -0.77% | ~Equal |
| **mrr** | 0.3293 | 0.3242 | -0.0051 | -1.55% | 35 (marginal) |
| **positive_brier** | 0.6848 | 0.6868 | +0.0020 | — | ~Equal |
| **common_top10_acc** | 0.8144 | 0.8173 | +0.0029 | +0.36% | ~Equal |
| **medium_top10_acc** | **0.0047** | **0.0016** | **-0.0031** | **-66.7%** | **35** |
| **rare_top10_acc** | 0.0 | 0.0 | 0 | — | Neither |
| **tail_top10_acc** | 0.0 | 0.0 | 0 | — | Neither |
| **macro_auroc** | 0.8581 | 0.8781 | +0.0200 | +2.3% | 200 |
| **macro_auprc** | 0.1057 | 0.1048 | -0.0009 | -0.85% | ~Equal |

### 1.2 Final Gradient Tier Fractions

| Tier | pos_weight_max=35 | pos_weight_max=200 | Delta |
|------|-------------------|-------------------|-------|
| **common_frac** | 84.88% | 84.68% | -0.20% |
| **tail_frac** | 0.125% | 0.171% | +0.046% |

---

## 2. GRADIENT TIER EVOLUTION COMPARISON

### 2.1 Key Timepoints During Training

| Step | pos_weight_max=35 Common | pos_weight_max=200 Common | pos_weight_max=35 Tail | pos_weight_max=200 Tail |
|------|------------------------|-------------------------|----------------------|----------------------|
| **1** | 18.0% | 17.8% | 17.7% | 17.8% |
| **101** | 18.1% | 17.6% | 18.0% | 18.0% |
| **301** | 17.7% | 17.2% | 18.0% | 18.2% |
| **501** | 16.9% | 16.4% | 18.2% | 18.4% |
| **801** | 18.0% | 18.0% | 17.8% | 17.9% |
| **1001** | 23.9% | 21.7% | 16.7% | 17.0% |
| **1501** | 39.5% | 37.3% | 13.0% | 13.6% |
| **1801** | 49.3% | 49.8% | 10.0% | 10.1% |
| **3001** | ~66-68% | ~66-67% | ~3-5% | ~3-4% |
| **6001** | ~80-82% | ~83-85% | ~1-2% | ~0.5-1% |
| **9001** | ~86-88% | ~84-88% | ~0.2% | ~0.1-0.2% |
| **12001** | 86.8% | 85.3% | 0.12% | 0.13% |

### 2.2 Gradient Tier Epoch Averages

| Tier | pos_weight_max=35 | pos_weight_max=200 | 
|------|-------------------|-------------------|
| common_frac | 82.4% | 82.8% |
| common_norm | 3.38 | 3.61 |
| medium_frac | 10.3% | 10.2% |
| medium_norm | 0.276 | 0.295 |
| rare_frac | 2.3% | 2.0% |
| rare_norm | 0.030 | 0.031 |
| tail_frac | **1.27%** | **1.08%** |
| tail_norm | 0.019 | 0.017 |
| total_norm | 4539 | 4861 |

---

## 3. CRITICAL INTERPRETATION AND ROOT CAUSE ANALYSIS

### 3.1 CONFIRMED HYPOTHESES (Now Validated by Baseline Comparison)

#### ✅ **Hypothesis: Gradient Concentration is INTRINSIC to Training Dynamics, NOT pos_weight-Induced**

**Evidence:**
- At step 1: Both experiments show ~18% common, ~17-18% tail (balanced)
- At step 12001: Both experiments show ~85% common, ~0.1% tail (severely concentrated)
- The **transition timeline is nearly identical** regardless of pos_weight setting

**Mechanistic Confirmation:**
The gradient tier evolution follows the same three-phase pattern in BOTH experiments:
1. **Phase 1 (Steps 1-500)**: Balanced gradients (~17-18% per tier)
2. **Phase 2 (Steps 500-3000)**: Rapid concentration (common rises from ~17% to ~67%)
3. **Phase 3 (Steps 3000+)**: Terminal concentration (common stabilizes at ~85%)

**Conclusion:** The gradient concentration is an **emergent property of BCE loss on imbalanced multi-label data**, not a consequence of the pos_weight magnitude. Experts 1-4 correctly hypothesized this, and the baseline now confirms it.

---

#### ✅ **Hypothesis: pos_weight Cannot Overcome Sample Count Differential**

**Evidence:**
- 5.7× increase in pos_weight (35 → 200) resulted in:
  - **No change** to rare/tail accuracy (0% → 0%)
  - **Negligible change** to gradient tier fractions (84.88% → 84.68% for common)
  - **Slight decrease** in tail_frac in epoch average (1.27% → 1.08%)

**Mechanistic Explanation:**
The per-sample weighting is overwhelmed by the sample count ratio. If tail codes appear in ~1/1000 samples and common codes appear in ~1/10 samples, then:
- pos_weight=35 for tail: 35 × (1/1000) = 0.035 effective contribution per batch
- pos_weight=200 for tail: 200 × (1/1000) = 0.2 effective contribution per batch
- pos_weight=1 for common: 1 × (1/10) = 0.1 effective contribution per batch

Even 200× pos_weight only brings tail to ~2× the common contribution **when a tail code appears** - but tail codes don't appear in most batches, making their gradient signal sporadic and averaged out.

---

#### ✅ **Hypothesis: Medium Code Vulnerability to pos_weight Changes**

**Evidence:**
- medium_top10_acc: 0.47% (pos_weight=35) → 0.16% (pos_weight=200)
- This is a **66.7% relative drop**, not 96% as previously reported

**Note on Discrepancy:** The original observation file reported pos_weight=50 baseline with medium_top10_acc=4.1%. The new baseline at pos_weight=35 shows medium_top10_acc=0.47%. This suggests the original exp_round5/exp2 run at pos_weight=50 achieved substantially better medium code performance than this pos_weight=35 run.

**Implication:** There may be a non-monotonic relationship between pos_weight and medium_top10_acc, or other factors (initialization, data ordering) are influencing medium code learning.

---

### 3.2 REFUTED OR NEEDS REVISION

#### ❌ **Expert Claim: "4× pos_weight increase (50→200) caused 96% medium collapse"**

**Reality:** The comparison is now 35 → 200 (5.7× increase), showing 66.7% relative decrease in medium_top10_acc (0.47% → 0.16%).

**Important:** The original experiment (pos_weight=50) showed medium_top10_acc=4.1%, which is ~9× higher than the pos_weight=35 baseline (0.47%). This suggests:
- Either the pos_weight=50 experiment had different conditions (data, initialization)
- Or there's a **sweet spot** around pos_weight=50 for medium codes

**This needs further investigation** before concluding that higher pos_weight universally hurts medium codes.

---

#### ❓ **Unclear: Whether macro_auroc Improvement is Meaningful**

**Evidence:**
- macro_auroc: 0.858 (pos_weight=35) → 0.878 (pos_weight=200) [+2.3%]

**Interpretation:**
This improvement suggests that higher pos_weight DOES improve per-code discrimination (the ability to separate positive from negative for each code). However, this doesn't translate to better top-K ranking because:
1. AUROC measures discrimination, not ranking
2. The logits may be "flatter" across codes, making top-K selection noisier

**Conclusion:** macro_auroc improvement is real but doesn't address the core ranking problem.

---

### 3.3 KEY NEW INSIGHTS FROM BASELINE COMPARISON

#### Insight 1: The Total Gradient Norm is Higher at pos_weight=200

| Metric | pos_weight=35 | pos_weight=200 |
|--------|---------------|----------------|
| total_norm (epoch avg) | 4539 | 4861 |
| common_norm | 3.38 | 3.61 |

**Interpretation:** Higher pos_weight increases overall gradient magnitude by ~7%. This doesn't change the concentration pattern, but it does mean the model is taking slightly larger steps, which could contribute to the slight AUROC improvement (faster learning of discriminative features).

---

#### Insight 2: The Transition Timing is Identical

The critical transition from balanced to concentrated gradients occurs at the same steps regardless of pos_weight:
- **Step 500-1500**: Transition begins
- **Step 1500-3000**: Rapid concentration
- **Step 3000+**: Plateau at ~85% common

**Implication:** Any intervention must act **before step 1500** to prevent the concentration cascade.

---

#### Insight 3: tail_frac is Actually LOWER at pos_weight=200

| Metric | pos_weight=35 | pos_weight=200 |
|--------|---------------|----------------|
| tail_frac (epoch avg) | 1.27% | 1.08% |
| tail_frac (final) | 0.125% | 0.171% |

**Contradiction:** The epoch average shows tail_frac is LOWER at pos_weight=200, but the final recorded value shows it's HIGHER. This variance suggests tail gradient contribution is noisy and batch-dependent.

**Implication:** The tail gradient signal is highly stochastic, which is why the model cannot reliably learn from it.

---

## 4. WHAT HAS BEEN DEFINITIVELY PROVEN

| Claim | Status | Evidence |
|-------|--------|----------|
| Gradient concentration is progressive, not static | **PROVEN** | Both experiments show 18% → 85% common tier evolution |
| Concentration pattern is pos_weight-independent | **PROVEN** | Near-identical evolution at 35 and 200 |
| pos_weight=200 does NOT improve rare/tail accuracy | **PROVEN** | Both remain at 0% |
| pos_weight=200 DOES improve macro_auroc | **PROVEN** | +2.3% improvement |
| Transition occurs between steps 500-3000 | **PROVEN** | Identical timeline in both experiments |
| The problem is gradient AGGREGATION, not weighting | **PROVEN** | 5.7× weight change had <0.5% effect on concentration |

---

## 5. WHAT REMAINS UNPROVEN OR NEEDS FURTHER INVESTIGATION

| Question | Status | Next Step |
|----------|--------|-----------|
| Why did pos_weight=50 achieve medium_top10_acc=4.1%? | **UNEXPLAINED** | Re-run pos_weight=50 with gradient tier logging |
| Is there a non-monotonic pos_weight effect on medium codes? | **UNCLEAR** | Test pos_weight=50, 75, 100 |
| Would tier-aware batching prevent concentration? | **UNTESTED** | Implement and run |
| Would gradient normalization prevent concentration? | **UNTESTED** | Implement and run |
| Are rare/tail codes fundamentally unpredictable? | **UNKNOWN** | Analyze per-code logit distributions |

---

## 6. DEFINITIVE ROOT CAUSE STATEMENT

Based on the now-complete baseline comparison:

### The Learning Plateau Root Cause

**The plateau is caused by an INTRINSIC property of BCE loss applied to imbalanced multi-label classification:**

1. **Early training**: Gradients are balanced because the model is randomly initialized and makes errors uniformly across codes.

2. **Mid training**: Common codes appear in nearly every batch. Their cumulative gradient signal (many samples × small per-sample error) dominates the update direction. The model begins to converge on a "common code prior."

3. **Late training**: Rare/tail codes appear sporadically. Their high-magnitude spikes are averaged out by the consistent common code signal. By step 3000, common codes capture ~67% of gradients; by step 12000, ~85%.

4. **The pos_weight mechanism operates per-sample, not per-code**. It cannot overcome the sample count differential because:
   - Total gradient per code ∝ (samples containing code) × (pos_weight) × (per-sample error)
   - For tail codes: 100 samples × 200 weight × 0.9 error = 18,000
   - For common codes: 100,000 samples × 1 weight × 0.1 error = 10,000
   - But the batch-level aggregation averages tail spikes into noise

5. **The result is a self-reinforcing loop**: As common codes improve, their per-sample gradients decrease, but their **cumulative** contribution remains dominant because they appear in every batch. Rare codes never accumulate enough consistent signal to learn.

---

## 7. RECOMMENDED NEXT STEPS (Revised Based on Baseline Data)

### Priority 1: Per-Tier Logit Distribution Analysis (Zero Cost)

**Action**: Using the existing pos_weight=200 model, compute logit distributions for each tier on validation set.

**Question Answered**: Does the model have ANY signal for rare/tail codes, or are logits near zero?

### Priority 2: Investigate pos_weight=50 Discrepancy

**Action**: Re-run pos_weight=50 experiment with gradient tier logging.

**Question Answered**: Why did the original pos_weight=50 achieve 4.1% medium_top10_acc vs 0.47% at pos_weight=35?

### Priority 3: Tier-Aware Batching (Addresses Root Cause)

**Action**: Implement batch construction that guarantees N rare/tail positive samples per batch.

**Rationale**: The baseline proves that the concentration is intrinsic to batch composition, not weight magnitude. Changing exposure directly addresses the root cause.

### Priority 4: Per-Tier Gradient Normalization

**Action**: After backward(), scale gradients by tier to enforce target fractions (e.g., 25% each).

**Rationale**: This surgically addresses the gradient aggregation problem identified in the root cause analysis.

---

## 8. SUMMARY

The baseline experiment at pos_weight_max=35 provides definitive evidence that:

1. **The gradient concentration pattern is INDEPENDENT of pos_weight** - both 35 and 200 show identical evolution from balanced (~18% per tier) to concentrated (~85% common)

2. **The pos_weight mechanism is fundamentally insufficient** - a 5.7× increase produced <0.5% change in gradient tier fractions

3. **The transition timing is consistent** - steps 500-3000 are the critical intervention window in both experiments

4. **The path forward requires structural changes** - modifying batch composition (tier-aware batching) or gradient aggregation (per-tier normalization), not per-sample weighting