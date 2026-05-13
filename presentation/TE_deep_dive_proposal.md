# Clinical Transformer Embeddings: Technical Deep Dive (3-4 Slides)

## Presentation Goal

Tell the Clinical TE story as a data science narrative: architecture validated by ablation, a rigorously tested alternative (MoE) that taught us when simpler wins, and a mechanistically understood bottleneck that points the path forward.

**Target audience**: Technical stakeholders. **Format**: 3-4 slides, ~15-20 minutes.

**Narrative arc**:
1. Here's what we built and why it works (**credibility**)
2. We tried making it fancier — learned when simpler is better (**rigor**)
3. The real frontier isn't architecture — it's data distribution (**insight + path forward**)

---

## Slide 1: Architecture — "Mirror the Data, Not the Literature"

### Walk the architecture bottom-up

**Visual**: Side-by-side — raw claims data structure (left) mirrored by model architecture (right).

```
Member 12345:
  Day 1: [E11.9, I10, Z79.4]  → Daily Encoder → d_1 (256-d)     80 codes/day
  Day 2: [E11.65, H35.01]     → Daily Encoder → d_2 (256-d)     no positional encoding
  ...                                                              (codes are unordered sets)
  Day 200: [Z00.00]           → Daily Encoder → d_200 (256-d)
                                     ↓
                    + Demographics (age, gender, LOB) via residual
                                     ↓
                    Temporal Encoder (6 layers, causal mask, RoPE, SwiGLU, Flash Attention)
                                     ↓
                    Member Embedding (256-d) → predict 6,297 grouped clinical codes
```

**Key design decisions** (each motivated by data, not trends):
- **No positional encoding in daily encoder**: codes within a day are an unordered set — ICD-10 E11.9 and I10 on the same day have no inherent sequence
- **Causal masking**: prevents temporal leakage AND defines the pre-training objective (next-day prediction) — dual purpose
- **Dual vocabulary**: 75,516 input codes (granular) → 6,297 output targets (clinically meaningful, noise-filtered)

### Design validated by ablation (Round 1, same data, 150K members)

We bundled modern transformer best practices (Flash Attention, FP16, SwiGLU, RoPE, pre-norm) and confirmed quality-neutral with 25% cost reduction (exp1 → exp2). The one clean single-variable ablation is Learned Attention Pooling vs MaxPool:

| Metric | MaxPool (exp2) | LAP (exp2b) |
|--------|---------------|-------------|
| recall@10 | 0.9430 | 0.9472 |
| Daily encoding speed | 1x | 3-5x faster |

LAP matches quality, significantly faster. exp2b becomes the winning architecture.

### What the model DOES well (Round 10, 11M members)

| Code Type | micro R@10 | Why it works |
|-----------|-----------|-------------|
| GPI Medications | **76.3%** | Concentrated vocabulary, strong temporal patterns |
| Place of Service | **75.1%** | Consistent patterns, moderate vocabulary |
| Provider Taxonomy | 54.7% | |
| Procedure Groups | 35.1% | Larger vocabulary, more rare codes |
| ICD-10 Diagnosis | 31.0% | |

- Medium codes went from **0% → 20%** accuracy between 6.8M and 11M members — proves the architecture CAN learn lower-frequency codes given sufficient data exposure
- Embedding-only downstream AUC = **0.810** — an unsupervised 256-d vector gets within 2.8pp of production tabular pipeline (0.838)
- Hybrid (embedding + tabular): AUC = 0.831, closing the gap to 0.7pp

### Land the punch

The architecture works. 25.3M parameters, $44 to train on 11M members, captures temporal clinical patterns that dimensionality reduction cannot: **+110% Lift@1%** vs PCA baselines on the same strict OOT evaluation set (18.89 vs 9.00).

---

## Slide 2: Lesson 1 — "MoE: Right Idea, Wrong Scale"

### The hypothesis

Clinical populations are heterogeneous — chronic disease patients, acute episodic patients, complex comorbidity patients. Mixture-of-Experts lets different experts specialize per patient archetype. Top-2 routing = same FLOPs as dense. Literature (Mixtral, DeepSeek) shows MoE scaling benefits.

### The evidence — two clean snapshots, not a progression table

**Snapshot 1 — Round 1** (same data, same epochs, same config — only architecture differs):

| Architecture | recall@10 | recall@1 | Params |
|-------------|-----------|----------|--------|
| **Dense (exp2b)** | **0.947** | **0.698** | 25.3M |
| MoE 8 experts (exp3) | 0.777 | 0.305 | 35.1M |
| MoE + shared expert (exp4) | 0.775 | 0.305 | 34.9M |
| MoE 16 fine-grained (exp5) | 0.775 | 0.305 | 34.4M |

All MoE variants plateau at exactly recall@1 = 0.305. **56% below dense. With 27-40% more parameters.**

### What we tried to fix it (Rounds 2-3, 12 experiments)

Root cause analysis identified three interacting failures: aux loss dominating gradients (13x larger than task loss), cold router initialization, and SwiGLU/GELU activation mismatch. We addressed each systematically:

| What we tried | Result | Worth the cost? |
|--------------|--------|----------------|
| SwiGLU in expert FFNs | recall@10: 0.802 (worse than baseline 0.830) | No |
| MoE from layer 4 instead of 2 | recall@10: 0.799 (no improvement) | No |
| Reduce aux loss 10x (0.01 → 0.001) | recall@10: 0.835 (+0.5pp) | Marginal |
| **DeepSeek bias correction** (remove aux loss from gradient entirely) | **recall@10: 0.875 (+4.5pp)** | **Yes — single biggest fix** |
| Shared expert (1 shared + 7 routed) | CV improved but no quality gain | No |
| Fine-grained: 16 experts, top-5 | 6-7 of 16 collapsed; router gradient exploded (196x norm) | No |

**Snapshot 2 — Round 3** (same data, same epochs — clean head-to-head after all fixes):

| Architecture | recall@10 | Params |
|-------------|-----------|--------|
| **Dense (exp2b)** | **0.961** | **25.3M** |
| Best MoE (exp6a, aux-free, all fixes) | **0.962** | 30.4M |

Parity. Confirmed at scale — **Round 5** (1.5M members, 3 LOBs): Dense 0.828 vs MoE 0.827.

### The cost-benefit verdict

| Dimension | Dense (exp2b) | Best MoE (exp6a) | MoE tax |
|-----------|--------------|------------------|---------|
| recall@10 | 0.961 | 0.962 | +0.1% (noise) |
| Parameters | 25.3M | 30.4M | +20% |
| Peak memory | 11.1 GB | 13.5 GB | +18% |
| Throughput | 1,037 samp/s | 845 samp/s | -23% |
| Training cost | $5.73 | $7.04 | +23% |
| Complexity | Simple | Router tuning, bias_lr, collapse monitoring | High |

### Why — three structural reasons

1. **Scale mismatch**: MoE benefits emerge at >1B params (Mixtral, DeepSeek). At 25M, we're 40x below the threshold where routing overhead pays for itself.
2. **Aux loss gradient hijacking**: Load-balancing loss was 13x larger than task loss. Removing it entirely was the only effective fix — but even then, only reached parity.
3. **Domain homogeneity**: MoE excels in multi-domain settings (translation/summarization/code). Clinical claims is one domain. Co-occurrence analysis confirms: 86% of all code-pair diversity involves at least one common code. Patient heterogeneity is a continuous spectrum, not discrete archetypes needing separate expert processing.

### Land the punch

12 experiments across 3 rounds. Fixed every identified root cause. Best MoE matches dense — never beats it. The taxes (20% params, 23% slower, higher complexity) far exceed the benefit (0.1% recall, within noise). **At this scale and domain, simpler is better.**

---

## Slide 3: Lesson 2 — "The Bottleneck Is Data Distribution, Not Architecture"

### Bridge from Slide 2

With architecture settled on exp2b, we investigated: why does performance plateau at recall@10 ~0.85 regardless of what we try?

### Why this matters: rare codes are the most clinically valuable

| Code tier | Median Odds Ratio (IP risk) | Mean pre-training logit | Model behavior |
|-----------|----------------------------|------------------------|----------------|
| Common (top 20%) | 1.46 | -2.26 | Learns well — but weak predictors |
| Tail (bottom 40%) | **2.42** (65% higher, p<0.001) | **-14.69** (12.4-unit gap) | **Strongest predictors — actively suppressed** |

We're not failing on obscure codes — we're systematically failing on the codes with the highest clinical signal.

### The evidence — architecture-agnostic ceiling

| What we changed | tail_top10_acc |
|----------------|---------------|
| Dense 25M params | 0% |
| MoE 35M params | 0% |
| 512-dim 59M params | 0% |
| BCE → Asymmetric Loss | 0% |
| pos_weight 35 → 200 | 0% |
| 1.5M → 11M members | 0% |

Nothing moves the needle. Not architecture, not loss function, not scale.

### Root cause — Emergent Gradient Starvation

Code frequency Gini = 0.939. Common codes = 69.7% of occurrences. Tail codes = 5.2%.

| Training step | Common gradient share | Tail gradient share |
|--------------|----------------------|-------------------|
| Step 1 | 17.8% | 17.8% |
| Step 3,000 | 66.7% | 3.0% |
| Step 12,000 | **85.3%** | **0.1%** |

The shared encoder becomes a common-code feature extractor. Tail embeddings homogenize to std=0.03 (vs common std=0.27). **And scaling makes it worse**: at 3.4M members, tail logits dropped further from -12.93 to -14.69. More data from the same Zipf distribution reinforces the gradient monopoly.

### Proving this is structural, not tunable

**The encoder is the bottleneck, not the decoder** (Round 9, Kang et al. decoupled training): Froze encoder, retrained decoder on code-balanced batches. Decoder loss converged 20x better, gradient tail fraction reached 40.2%. But tail_top10_acc stayed at **0%**. The encoder representation `h` simply lacks discriminative features for tail codes.

**More data can't fix the distribution** (saturation analysis):

| Property | 100K members | 1M members | 10M members | Trend |
|----------|-------------|------------|-------------|-------|
| Shannon entropy | 7.831 bits | 7.833 bits | 7.834 bits | **Flat** |
| Gini coefficient | 0.934 | 0.942 | 0.950 | **Increasing** (more concentrated) |

The Zipf distribution is a structural property of the clinical coding system, not an artifact of sample size. Marginal members are statistically indistinguishable from the core population (all distributional tests p > 0.4).

**Co-occurrence structure is too sparse for attention to compensate**:
- 86% of co-occurrence pair diversity involves at least one common code
- Only **21** unique tail-tail co-occurrence pairs (out of thousands possible)
- Mean MI between code pairs: **0.005 bits** (near-independent)
- Within-member novelty at day 50: only **11%** (89% repeats)

### But data scaling DOES work for medium codes — proving the threshold mechanism

| Data | medium_top10_acc | tail_top10_acc |
|------|-----------------|---------------|
| 1.5M | 0.2% | 0% |
| 6.8M | 4.3% | 0% |
| **11M** | **20.0%** | 0% |

Medium codes crossed ~1,100 occurrences at 11M — enough for the gradient to learn. The architecture CAN learn lower-frequency codes. We just need to change HOW gradient reaches tail codes.

### The three interlocking layers of the problem

1. **Gradient starvation** (training dynamics): Common codes capture 85% of gradient by step 12K → Fix via masking objectives or gradient engineering
2. **Information poverty** (data structure): Tail codes have near-zero MI (0.005 bits), only 21 tail-tail pairs, 89% temporal redundancy → Fix via external data or representation pre-training
3. **Distributional invariance** (fundamental): Entropy flat at 7.834 bits across 100x scaling → Fix requires changing the type of data or the objective, not volume

### Path forward — experimentally grounded

| Tier | Approach | Evidence | Status |
|------|----------|---------|--------|
| **Tested** | PPMI+SVD code embedding pre-training | Tail embedding std 0.03 → 0.077; first positive tail margin; hybrid downstream Lift@1% = **20.50** vs production 19.38 | Partially validated (R9) |
| Untested | MLM-style masked code prediction | Decouples gradient from frequency — masking schedule controls gradient allocation, not natural frequency | Strongly motivated by diagnostics |
| Untested | Contrastive pre-training (patient-level) | Sidesteps Zipf entirely — patient trajectories are diverse even when code distributions are invariant | Hypothesis stage |
| Untested | Per-code gradient normalization | Directly attacks gradient concentration (85% → 0.1%) | Hypothesis stage |

### Land the punch

The architecture is sound — the bottleneck is three interlocking data distribution problems. Rare codes are clinically the most valuable (65% higher OR for IP risk), yet our model systematically suppresses them (12.4-unit logit gap). More data can't fix this — entropy is flat at 7.834 bits across 100x scaling, and Gini *increases* with scale. But we have a partial breakthrough: PPMI+SVD pre-training moved tail embeddings from frozen to alive, and the medium code breakthrough at 11M proves the mechanism IS learnable. **We need to change how gradient reaches tail codes, not how much data we throw at them.**

---

## Optional Slide 4: Value Delivered Today

> Use this slide if the audience needs a concrete "so what" before the path-forward discussion.

### TE vs dimensionality-reduction baselines (same strict OOT evaluation)

| Representation (256-d) | OOT-strict Lift@1% |
|------------------------|-------------------|
| PCA(256) | 9.00 |
| AutoEncoder(256) | ~9 |
| SelectKBest(256) | ~9 |
| **TE R10 (embedding-only)** | **18.89** |
| **Improvement** | **+110%** |

### TE vs production pipeline

| Model | OOT-strict AUC | OOT-strict Lift@1% |
|-------|----------------|-------------------|
| Production (tabular, 533 features) | 0.838 | ~19.38 |
| TE R10 (embedding-only, 256 features) | 0.810 | 18.89 |
| TE R10 (hybrid: embedding + tabular) | 0.831 | 18.69 |
| R9 co-occur embedding (hybrid, 13 LOBs) | 0.827 | **20.50** |

An unsupervised 256-d embedding gets within 2.8pp AUC of a production pipeline with 533 hand-engineered features. The hybrid with PPMI+SVD pre-training **exceeds production** on Lift@1%.

### What the audience should leave with

1. **Architecture decisions are evidence-based, not trend-driven** — every component validated by ablation
2. **We systematically tested and eliminated alternatives** — 12 MoE experiments, 14 plateau hypotheses
3. **The bottleneck is mechanistically understood** at three levels (gradient, information, distributional) with quantified diagnostics
4. **The stakes are real** — rare codes have 65% higher OR for IP risk
5. **We have a partially validated path forward** — PPMI+SVD + the medium code breakthrough prove the mechanism is learnable
6. **The model delivers strong clinical value today** — medications 76%, 2.8pp gap to production AUC, 110% over PCA baselines

---

## Potential Audience Q&A (Pre-loaded)

| Challenge | Answer |
|-----------|--------|
| "Why not a bigger model?" | R7/R8: 512d adds +0.3pp for 2.3x params. Bottleneck is not capacity. |
| "Why not retrain the decoder?" | R9: decoder learned fine, tail stayed 0%. Bottleneck is the encoder. |
| "Why not more data?" | Entropy flat at 7.834 bits across 100x scaling. Gini *increases*. Same Zipf distribution. |
| "Why not a different loss?" | R5.1: ASL improved calibration but did NOT change gradient distribution. Independent mechanisms. |
| "Is the MoE comparison fair?" | R3 head-to-head: same data, same epochs. R5 confirmed at 1.5M 3-LOB scale. Same answer both times. |
| "Are downstream numbers trustworthy?" | All comparisons use same strict OOT evaluation set. See corrected methodology. |
| "If codes are near-independent, how can attention help?" | Attention helps for common codes where co-occurrence is abundant (86% of pair diversity). The failure is tier-specific, not model-wide. |

---

## Presentation Logistics

- **Slides**: 3 core + 1 optional value slide
- **Time**: 15-20 minutes + Q&A
- **Key visuals needed**:
  - Architecture diagram (bottom-up data flow)
  - Gradient starvation area chart (common expanding, tail collapsing over training steps)
  - Code embedding t-SNE/UMAP colored by frequency tier
  - MoE cost-benefit summary table
- **Format**: Data science storytelling — every claim backed by a specific experiment with stated comparison validity
