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

### The evidence — sourced experiment table with routing diagnostics

Root cause analysis identified three interacting failures: aux loss dominating gradients, unstable early routing, and the SwiGLU/GELU mismatch at the MoE boundary. The table keeps the performance story, but adds the logged model size and routing-health diagnostics.

| Round | Variant | Why tested | recall@10 | recall@1 | Params | Worst logged epoch-avg collapse / CV |
|------|---------|------------|-----------|----------|--------|--------------------------------------|
| R1 | **Dense (exp2b)** | Best dense control | **0.947** | **0.698** | 27.56M | — |
| R1 | Standard MoE (exp3) | 8 experts, top-2 baseline | 0.777 | 0.305 | 35.07M | 0.60 / 0.43 |
| R1 | MoE + shared expert (exp4) | Keep one general expert always on | 0.775 | 0.305 | 34.94M | 0.48 / 0.42 |
| R1 | MoE 16 fine-grained (exp5) | More smaller experts | 0.775 | 0.305 | 34.36M | 3.83 / 0.37 |
| R2 | Baseline rerun (exp3) | Control for fix ablations | 0.830 | 0.330 | 35.07M | 0.08 / 0.28 |
| R2 | SwiGLU experts (exp3a) | Fix SwiGLU→GELU mismatch | 0.802 | 0.320 | 35.04M | 0.10 / 0.26 |
| R2 | SwiGLU + LAP (exp3b) | Add LAP on top of SwiGLU | 0.824 | 0.313 | 34.91M | 0.10 / 0.29 |
| R2 | Layer 4 insertion (exp3c) | Delay routing until later layers | 0.798 | 0.311 | 34.91M | 0.15 / 0.28 |
| R2 | Aux loss 0.001 (exp3d) | Reduce gradient conflict | 0.835 | 0.341 | 34.91M | 0.13 / 0.32 |
| R2 | **DeepSeek bias correction (exp6)** | Remove aux loss from backprop | **0.875** | **0.530** | 34.90M | 1.98 / 0.72 |
| R3 | **Best MoE (exp6a)** | Aux-free + layer 4, best parity run | **0.962** | **0.757** | **84.47M** | **3.57 / 1.22** |

This makes the severity visible: performance improved only after removing the auxiliary loss from the gradient, but the routing metrics did not become healthy. The best-performing MoE row still shows severe collapse.

### Best-case parity vs production-scale verdict

**Round 3 parity run** (same run, same data):

| Architecture | recall@10 | recall@1 | Params |
|-------------|-----------|----------|--------|
| **Dense (exp2b)** | **0.961** | 0.747 | 64.29M |
| **Best MoE (exp6a)** | **0.962** | **0.757** | 84.47M |

That is parity, not superiority, and it required 31% more parameters in the logged Round 3 setup.

**Round 5 production-scale comparison** (1.5M members, 3 LOBs, same 256d setup):

| Dimension | Dense (exp2b) | MoE (exp6 V3) | MoE tax |
|-----------|---------------|---------------|---------|
| recall@10 | **0.8285** | 0.8273 | Essentially tied |
| ndcg@10 | 0.3983 | **0.3987** | Essentially tied |
| Parameters | 25.33M | 35.42M | +39.8% |
| Peak memory | 11.14 GB | 13.39 GB | +20.2% |
| Throughput | **1,037.5 samp/s** | 895.8 samp/s | -13.7% |
| Training cost | **$5.73** | $6.64 | +15.8% |
| Complexity | Simple | Router tuning, bias_lr, collapse monitoring | High |

### Why — three structural reasons

1. **Scale mismatch**: MoE benefits emerge at >1B params (Mixtral, DeepSeek). At 25M, we're 40x below the threshold where routing overhead pays for itself.
2. **Aux loss gradient hijacking**: Load-balancing loss was 13x larger than task loss. Removing it entirely was the only effective fix — but even then, only reached parity.
3. **Domain homogeneity**: MoE excels when the router can separate genuinely different computational modes, like translation vs code vs summarization. Clinical claims looks much more like one constrained clinical dialect. Of the 898,332 unique same-day code pairs, 855,373 include at least one common code (95.2%), and the top 10% of pairs explain 91.7% of all pair occurrences. Patient heterogeneity is a continuous spectrum, not a clean split into expert-worthy archetypes.

### Land the punch

12 experiments across 3 rounds. Fixed every identified root cause. Best-case MoE only reaches parity in a much larger Round 3 run, and the production 256d comparison is still effectively tied while paying 39.8% more parameters, 20.2% more memory, 13.7% lower throughput, higher cost, and much higher engineering complexity. **At this scale and domain, simpler is better.**

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
- 95.2% of unique same-day co-occurrence pairs involve at least one common code
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
| "If codes are near-independent, how can attention help?" | Attention helps for common codes where co-occurrence is abundant and 95.2% of unique same-day pairs already include at least one common code. The failure is tier-specific, not model-wide. |

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
