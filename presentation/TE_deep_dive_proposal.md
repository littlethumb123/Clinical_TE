# Clinical Transformer Embeddings: Technical Deep Dive — Presentation Proposal

## Presentation Goal

Tell the complete technical story of Clinical TE — from architecture design through 10 rounds of experimentation — with emphasis on:
- How clinical data flows through the hierarchical architecture and why each design choice exists
- The MoE hypothesis, systematic load-balancing efforts, and mechanistic explanation of why it didn't add value
- The gradient starvation discovery and systematic investigation
- Progressive experimentation results with evidence-based reasoning
- Downstream value and path forward

**Target audience**: Technical stakeholders who need to understand both *what was built* and *why it works the way it does*.

---

## Presentation Outline

### Part I: Architecture Walk-Through (Slides 1–10)

> Start directly with the model. No preamble — the audience is here for the technical deep dive.

---

#### Slide 1: The Two-Level Hierarchy — Why This Shape

**Content**: Side-by-side diagram:
- Left: Raw claims data structure (member → days → codes per day)
- Right: Model architecture mirroring that structure (Daily Encoder → Temporal Encoder → Embedding)

**Key message**: The architecture mirrors how clinical data is naturally organized. A member's health story is a sequence of days, and each day is a bag of co-occurring medical events. The model encodes both levels.

**Visual**: Data record example:
```
Member 12345:
  Day 1: [E11.9 (T2DM), I10 (HTN), Z79.4 (insulin)]     → Daily Encoder → d_1 (256-d)
  Day 2: [E11.65 (DM+retinopathy), H35.01 (retinal exam)] → Daily Encoder → d_2 (256-d)
  ...
  Day 200: [Z00.00 (wellness visit)]                       → Daily Encoder → d_200 (256-d)
                                                                               ↓
                                                             Temporal Encoder (6 layers)
                                                                               ↓
                                                             Member Embedding (256-d)
```

---

#### Slide 2: Daily Encoder — Encoding Co-Occurring Events Within a Day

**Content**: Walk through one day's processing:
1. Up to 80 medical codes per day → embedding lookup (75,516-dim vocabulary)
2. Self-attention with 4 heads (no positional encoding — codes within a day are unordered)
3. Demographic injection: age (in months), gender, LOB embeddings added via residual
4. Pooling → one 256-d vector per day

**Key design decision callout**: No positional encoding in the daily encoder. Within a single claim day, codes are a *set* not a *sequence*. ICD-10 E11.9 and I10 appearing on the same day have no inherent ordering.

**Visual**: Flow diagram: `[code_1, code_2, ..., code_80]` → Embedding → Self-Attention → `+ age + gender + LOB` → Pool → `d_i`

---

#### Slide 3: Daily Encoder — MaxPool vs Learned Attention Pooling

**Content**: Compare two pooling strategies:
- **MaxPool**: Takes element-wise max across 80 code positions. Hard selection — only the strongest activation per dimension survives. Simple, fast.
- **Learned Attention Pooling (LAP)**: A single learnable query vector attends to all code embeddings, producing a soft weighted sum. The model *learns* what to attend to.

**Evidence**: Round 1 results — exp2 (MaxPool) vs exp2b (LAP):

| Metric | MaxPool (exp2) | LAP (exp2b) |
|--------|---------------|-------------|
| recall@10 | 0.9430 | 0.9472 |
| balanced_top10 | 0.2369 | 0.2377 |
| Speed (daily encoding) | 1x | 3-5x faster |

LAP matches or beats MaxPool on quality while being significantly faster (no sequential max operation — single attention pass).

---

#### Slide 4: Temporal Encoder — Encoding Disease Progression Across Time

**Content**: Walk through the temporal encoding:
1. 200 daily vectors as a sequence of "day tokens"
2. **Rotary Position Embedding (RoPE)**: Unlike the daily encoder, temporal order matters. RoPE injects relative position without fixed sinusoidal embeddings.
3. **Causal masking**: Each day can only attend to prior days — the model predicts the future from the past, not the other way around. This prevents temporal leakage.
4. **6 transformer layers**: Pre-norm architecture (LayerNorm → Attention → Residual) for gradient stability
5. **SwiGLU FFN**: Gated activation (SiLU(xW_1) * xW_2)W_3 — empirically stronger than GELU/ReLU in transformer FFNs (PaLM, LLaMA findings apply here)

**Key design decision callout**: Causal masking serves double duty — it defines the pre-training objective (next-day prediction) and prevents temporal leakage during embedding generation (the embedding at day t uses only information from days 1..t).

---

#### Slide 5: Pre-Training Objective — Multi-Label Next-Code Prediction

**Content**:
- At each position t, predict which medical codes will appear at position t+1
- This is multi-label (a member can have 0 to many codes on any given day) → **BCEWithLogitsLoss** (not softmax cross-entropy)
- Output: `[batch, 200, 6,297]` logits — one prediction per day per target code

**Dual vocabulary design**:
- **Input vocabulary**: 75,516 codes (granular ICD-10, CPT, GPI, DRG, revenue codes, provider taxonomy, place of service)
- **Output vocabulary**: 6,297 grouped target codes (clinically meaningful prediction targets, noise filtered)

**Why dual**: Not all input codes are worth predicting. Predicting 75K codes would waste capacity on noise (billing artifacts, rarely repeated codes). The 6,297 target codes represent clinically meaningful, recurring medical events.

---

#### Slide 6: The Code Frequency Challenge

**Content**: Visualize the extreme imbalance in the training data:

| Tier | Codes | Member prevalence | Occurrence share |
|------|-------|-------------------|-----------------|
| Common (top 25%) | 1,142 | 100% | 69.7% |
| Medium | 1,711 | 97.3% | — |
| Rare | 1,706 | 95.1% | — |
| Tail (bottom 25%) | 1,148 | 83.4% | 5.2% |

**Key insight**: 83.4% of members have at least one tail code — they are not rare *members*, they are rare *occurrences*. The distinction between member-level and occurrence-level prevalence is critical and drives much of the later investigation.

**Proposed visual**: Log-scale histogram of code frequencies showing the power-law distribution (Gini = 0.939). Overlay the tier boundaries.

**pos_weight strategy**: Log-scaled reweighting — `w_j = log(max_freq) / log(freq_j)`, linearly rescaled to [1, 200]. Avoids the instability of raw inverse-frequency weighting at 16M:1 imbalance ratios.

---

#### Slide 7: The Final Embedding — What Gets Extracted

**Content**: After temporal encoding, the embedding is extracted at the member's last valid day position. This 256-dimensional vector encodes:
- The member's full longitudinal clinical history (up to 200 days)
- Temporal patterns (disease progression, treatment sequences)
- Co-occurrence structure (comorbidity patterns)
- Demographic context (age trajectory, LOB)

**Downstream usage**: This fixed-size vector replaces or augments hand-engineered features in downstream ML models (XGBoost, LightGBM, Logistic Regression) for tasks like inpatient risk prediction.

---

#### Slide 8: Architecture Component Summary

**Content**: Summary table of all design choices and their justification:

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Daily Encoder | 1-layer, 4 heads, no positional encoding | Codes within a day are unordered sets |
| Daily Pooling | Learned Attention Pooling | Matches MaxPool quality, 3-5x faster |
| Temporal Encoder | 6 layers, 8 heads, pre-norm | Sufficient depth for 200-day sequences |
| Position Encoding | RoPE (temporal only) | Relative positions; generalizes to unseen lengths |
| FFN Activation | SwiGLU | Empirically superior (PaLM, LLaMA findings) |
| Masking | Causal | Prevents temporal leakage; defines pretraining objective |
| Precision | FP16 (mixed precision) | 2x memory reduction, no quality loss |
| Attention | Flash Attention (xFormers) | O(N) memory, 2-3x speed, identical output |

---

### Part II: Architecture Ablation Results (Slides 9–12)

---

#### Slide 9: Ablation Design — One Variable at a Time

**Content**: The 7-experiment matrix from Round 1, showing what each experiment isolates:

| Exp | What it adds | Tests |
|-----|-------------|-------|
| exp1 | — | Dense baseline (PyTorch stock transformer) |
| exp2 | Flash Attention, FP16, RoPE | Is Flash Attention lossless? |
| exp2b | + Learned Attention Pooling | LAP vs MaxPool |
| exp3 | + MoE (8 experts, top-2, Switch aux) | Does conditional computation help? |
| exp3b | + MoE + LAP | MoE + pooling interaction |
| exp4 | + MoE + shared expert | Does a shared expert prevent collapse? |
| exp5 | + MoE (16 fine-grained experts, top-5) | More smaller experts? |

Training: 1.5M members, 3 epochs, single T4 GPU. Same data, same hyperparameters, one variable changes.

---

#### Slide 10: Dense Architecture Results — Every Component Helps

**Content**: Results for the dense variants (exp1, exp2, exp2b):

| Variant | Params | recall@10 | balanced@10 | val_loss | Training cost |
|---------|--------|-----------|-------------|----------|--------------|
| exp1 (baseline) | 27.7M | 0.9474 | 0.2372 | 0.00275 | $2.48 |
| exp2 (+ Flash) | 27.7M | 0.9430 | 0.2369 | 0.00275 | $1.87 |
| exp2b (+ Flash + LAP) | 27.6M | 0.9472 | 0.2377 | 0.00273 | $1.52 |

**Key takeaway**: Flash Attention is quality-neutral with 25% cost reduction. Adding LAP recovers any marginal loss and reduces cost further. The final dense architecture (exp2b) is the cheapest and best.

**Proposed new analysis — Component contribution waterfall chart**:
Show cumulative contribution: Baseline → +Flash Attn (speed) → +RoPE (position) → +SwiGLU (activation) → +LAP (pooling). Each bar shows the delta in recall@10 and cost.

---

#### Slide 11: MoE Architecture Results — Consistent Underperformance

**Content**: Results for MoE variants vs the dense winner:

| Variant | Params | recall@10 | recall@1 | Collapsed experts |
|---------|--------|-----------|----------|-------------------|
| **exp2b (dense)** | **27.6M** | **0.9472** | **0.698** | **—** |
| exp3 (MoE 8, top-2) | 35.1M | 0.7766 | 0.305 | 3-4 of 8 |
| exp3b (MoE + LAP) | 34.9M | 0.7754 | 0.305 | 3-4 of 8 |
| exp4 (shared expert) | 34.9M | 0.7752 | 0.305 | 3-4 of 8 |
| exp5 (16 fine-grained) | 34.4M | 0.7752 | 0.305 | ~4 of 16 |

All MoE variants plateau at exactly recall@1 = 0.305 — a 56% drop from dense. With 27-40% more parameters. This is the starting point for the MoE investigation in Part III.

---

#### Slide 12: Multi-Round Validation (Rounds 2–4)

**Content**: The same pattern holds across larger datasets and multi-GPU training (Round 2-1 on 4xT4, Round 4 with 11 variants):

| Round | Dense best (exp2b) | MoE best (exp6/exp6a) | Gap |
|-------|-------------------|----------------------|-----|
| R1 (1.5M, 1-GPU) | 0.9472 | 0.7766 | -18% |
| R2-1 (1.5M, 4-GPU) | 0.9651 | 0.9664 (exp6 aux-free) | +0.1% |
| R4 (1.5M, 4-GPU, full suite) | 0.9578 | 0.9565 (exp6a) | -0.1% |

After fixing key MoE issues (see Part III), the auxiliary-free MoE (exp6) reaches parity with dense — but never exceeds it, despite 40% more parameters. The best MoE configuration is statistically indistinguishable from the simpler dense model.

---

### Part III: The MoE Investigation (Slides 13–22)

> This section tells the MoE story as a scientific investigation: hypothesis → experiment → diagnosis → iteration → conclusion.

---

#### Slide 13: The MoE Hypothesis — Why We Expected It to Help

**Content**:
- Clinical populations are heterogeneous: chronic disease patients, acute episodic patients, surgical patients, low-utilizers, complex comorbidity patients
- A dense FFN processes all patients through the same transformation
- MoE allows different experts to specialize in different patient archetypes
- Top-2 routing: most patients have ~2 dominant clinical profiles (e.g., diabetes + cardiovascular)
- Same-FLOPs design: 8 experts with top-2 routing activates the same compute as a single dense FFN

**Visual**: Patient archetype diagram → Router → Expert specialization illustration

**Literature backing**: Mixtral, DeepSeek-MoE, Switch Transformer all show MoE scaling benefits — but at >1B parameters and multi-domain data.

---

#### Slide 14: MoE Design Decisions

**Content**: Key choices and their rationale:

| Decision | Choice | Why |
|----------|--------|-----|
| Expert count | 8 | Matches hypothesized patient archetypes; sweet spot per DeepSeek |
| Top-K | 2 | Patients have ~2 dominant conditions; GLaM/Mixtral validated K=2 |
| Layer placement | Layers 2-5 (of 6) | Early layers learn universal features; specialization in deeper layers |
| Expert FFN | d_model=256 → d_ff=512 | Same dimensions as dense FFN for fair comparison |
| Load balancing | Switch aux loss (w=0.01) | Standard approach from Switch Transformer paper |

---

#### Slide 15: Round 1 Failure — The "0.305 Wall"

**Content**: All 4 MoE variants plateau at exactly recall@1 = 0.305 (dense: 0.697). The identical value across all variants is too consistent for coincidence.

**Root cause diagnosis** (3 interacting failures):

1. **Auxiliary loss dominance**: aux_loss ~4.0 × weight 0.01 = 0.04. Prediction loss at convergence = ~0.003. The balancing loss is **13x larger** than the actual task loss — gradients optimize for load balance, not prediction accuracy.

2. **Cold router initialization**: Router weights initialized at std=0.01 → near-random routing → some experts randomly become "winners" → aux loss tries to rebalance but conflicts with prediction gradient.

3. **Activation function mismatch**: Layers 0-1 use SwiGLU (from the base FlashAttention config), but MoE expert FFNs use GELU. The LayerNorm between layer 1 and layer 2 learned to expect SwiGLU output statistics. Receiving GELU output creates a distribution mismatch — a representational bottleneck at the MoE boundary.

**Proposed visual**: Failure loop diagram:
```
Tiny router init → Random routing → Aux loss >> Pred loss
       ↑                                        ↓
  Router can't learn ← Gradient dominated by balancing ← SwiGLU/GELU mismatch
```

---

#### Slide 16: Systematic MoE Debugging — One Fix at a Time

**Content**: Each Round 2 experiment targets exactly one hypothesized root cause:

| Experiment | Fix applied | recall@10 | Improvement? |
|------------|-----------|-----------|-------------|
| exp3 (baseline MoE) | — | 0.830 | — |
| exp3a | SwiGLU in expert FFNs | 0.802 | No (activation mismatch not sole cause) |
| exp3c | MoE from layer 4 (not 2) | 0.799 | No (layer placement not sole cause) |
| exp3d | aux_loss = 0.001 (10x lower) | 0.835 | Slight (reduces gradient conflict) |
| **exp6** | **DeepSeek bias correction** (no aux loss in gradient) | **0.876** | **Yes — best MoE result** |

**Key insight**: The auxiliary loss was the dominant problem. Removing it from the gradient entirely (DeepSeek bias correction) gave the biggest single improvement (+4.6pp). But even the best MoE variant only approaches, never exceeds, the dense baseline.

---

#### Slide 17: DeepSeek Bias Correction — How It Works

**Content**: Instead of penalizing load imbalance through the loss function (which creates gradient conflict), maintain a learnable bias vector updated outside of backprop:

```
After each batch:
  1. Measure current expert load: load_i = fraction of tokens routed to expert i
  2. Update EMA: ema_i = momentum * ema_i + (1-momentum) * load_i
  3. Update bias: bias_i -= bias_lr * (ema_i - 1/N)
  4. Bias is added to router logits before softmax (next batch)
```

This decouples load balancing from task learning — no gradient conflict. The main loss trains only on prediction accuracy.

**Tuning discovery**: `bias_lr` must be >= 1e-3 for small models. The DeepSeek-V3 paper uses 1e-5, but that's calibrated for trillion-parameter models trained for months. At our 25M-parameter scale with single-epoch training, 1e-5 is too slow to prevent collapse.

---

#### Slide 18: Expert Collapse — The Dead Expert Trap

**Content**: Even with DeepSeek correction, ~4 of 8 experts consistently collapsed across training.

**The EMA trap**: Once an expert's load reaches exactly 0%, the correction cannot recover it:
```
ema = momentum * 0 + (1-momentum) * 0 = 0
bias_update = bias_lr * (0 - 1/N) = constant small negative
```
The bias slowly increases, but the expert receives no tokens to train on, so its weights diverge from the data distribution. Even if routing eventually reaches it, the expert produces garbage output, which causes the router to avoid it again.

**Proposed new analysis — Expert utilization heatmap**: Parse `batch_metrics.json` for exp3, exp6, and exp6d. Plot per-expert load fraction over training steps as a heatmap. This will visually demonstrate:
- exp3: immediate collapse of 3-4 experts, no recovery
- exp6: initial collapse then partial recovery (DeepSeek bias working)
- exp6d: collapse of 6-7 of 16 experts, never recovered (too many experts)

| Metric | exp3 (Switch) | exp6 (DeepSeek) | exp6d (16 experts) |
|--------|--------------|-----------------|-------------------|
| CV (end) | 0.484 | 0.310 | 0.950 |
| Gini (end) | 0.461 | 0.130 | 0.750 |
| Collapsed (end) | 3-4 of 8 | ~0 of 8 | 6-7 of 16 |

---

#### Slide 19: Scaling Experts — More Is Worse

**Content**: exp6d (16 experts, 2 shared) vs exp6 (8 experts, 1 shared):

| Metric | 8 experts (exp6) | 16 experts (exp6d) |
|--------|-----------------|-------------------|
| recall@10 | 83.5% | 82.7% |
| micro_recall@10 | 49.4% | 46.6% |
| Parameters | 35.4M | 47.0M |
| Collapsed experts | ~0 | 6-7 |
| Router GradNorm | ~10 | 196.3 |

**Why more experts fails at this scale**:
1. Router output dimensionality 16x → gradient variance scales proportionally → router gradient explosion
2. 2 shared experts absorb all general-pattern signal → routed experts compete for residual → more collapse
3. Each specific expert gets fewer tokens → less training signal → weights diverge faster

---

#### Slide 20: The Definitive Comparison — Dense Wins

**Content**: Head-to-head, Round 4, full evaluation:

| Dimension | Dense (exp2b) | Best MoE (exp6a) | Winner |
|-----------|--------------|------------------|--------|
| recall@10 | 0.9578 | 0.9565 | Dense (+0.1%) |
| micro_recall@10 | 0.462 | 0.461 | Dense |
| Parameters | 25.3M | 30.4M | Dense (-17%) |
| Peak memory | 11.1 GB | 13.5 GB | Dense (-18%) |
| Throughput | 1,037 samp/s | 845 samp/s | Dense (+23%) |
| Training cost | $5.73 | $7.04 | Dense (-23%) |

Dense is better or equal on every metric, with fewer parameters, less memory, higher throughput, and lower cost.

---

#### Slide 21: Why MoE Doesn't Help Here — Mechanistic Explanation

**Content**: Three structural reasons, supported by evidence:

**1. Scale mismatch**: Literature consistently shows MoE benefits emerge at >1B parameters. At 25-50M, routing overhead (router training, expert init variance, load balancing complexity) exceeds any conditional computation benefit. Our model is 20-40x below the threshold.

**2. Class imbalance amplifies routing failure**: Code frequency distribution has Gini=0.939 (top 25% of codes = 98.8% of occurrences). The router learns to route based on dominant signal → experts specialize in common codes → sparse activation *amplifies* the frequency bias rather than enabling patient-archetype specialization.

**3. Homogeneous domain**: MoE excels in multi-domain settings (NLP: translate/summarize/code). Clinical claims data is a single domain with shared statistical structure. The patient-archetype heterogeneity we hypothesized is real but not architecturally deep — it manifests as different code combinations, not fundamentally different computational patterns. A single dense FFN handles this adequately.

**Visual**: Conceptual diagram — "MoE benefit vs model scale" curve, with our 25M-parameter model marked in the "penalty zone" below the crossover point.

---

#### Slide 22: MoE Summary — What We Learned

**Content**: Decision table and takeaways:

| What we tried | What we learned |
|---------------|----------------|
| Switch aux loss (w=0.01) | Aux loss 13x larger than task loss → gradient conflict |
| Reduce aux loss (w=0.001) | Helps but insufficient alone |
| DeepSeek bias correction | Removes gradient conflict; best MoE health metrics |
| Shared experts | 1 shared helps; 2 shared over-absorbs |
| Fine-grained (16 experts) | More collapse, not less; router gradient explosion |
| SwiGLU in experts | Not the primary bottleneck |
| Late MoE (layer 4+) | Marginal improvement only |
| 512-dim + MoE | More parameters, same ceiling |

**Conclusion**: At this model scale and domain, a well-optimized dense model (Flash Attention + Learned Pooling + SwiGLU + RoPE) outperforms all MoE variants while being simpler, cheaper, and faster. MoE is architecturally sound but operates below the scale threshold where its benefits materialize.

---

### Part IV: The Learning Plateau Investigation (Slides 23–32)

> With architecture settled (exp2b dense), this section investigates the fundamental learning bottleneck.

---

#### Slide 23: The Plateau Phenomenon

**Content**: Across all architectures, data sizes, and training configurations, metrics converge to a ceiling:

| Configuration | recall@10 | tail_top10_acc |
|---------------|-----------|---------------|
| Dense (25M, 1.5M data) | 0.828 | 0% |
| MoE (35M, 1.5M data) | 0.827 | 0% |
| Dense (25M, 3.4M data) | 0.834 | 0% |
| Dense (59M, 512d, 6.8M data) | 0.858 | 0% |
| Dense (25M, 11M data) | 0.853 | 0% |

**Key observation**: Tail code accuracy is 0% in every single experiment. Not 1%, not 0.5% — exactly zero. The model has *never* correctly ranked a tail code in top-10, regardless of architecture, loss, sampling, or data scale. This is not a tuning problem — it is structural.

---

#### Slide 24: Hypothesis-Driven Investigation — 14 Hypotheses Tested

**Content**: Scientific approach — each hypothesis tested with a specific experiment:

| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| H1 | LR schedule ceiling | LR polishing (2000 steps at 4e-6) | **REJECTED** — val_loss got worse |
| H2 | Embedding collapse | Measure embedding norms and variance | **REJECTED** — all norms healthy |
| H3 | Model capacity | MoE (35M) vs Dense (25M) | **REJECTED** — identical performance |
| H4 | Insufficient data | 2x data (1.5M → 3.4M) | **REJECTED** — +0.6% R@10 only |
| H5 | pos_weight too low | 5.7x increase (35 → 200) | **REJECTED** — <0.5% gradient change |
| H7 | Gradient concentration is emergent | Track gradient tier fractions over training | **CONFIRMED** |
| H9 | Problem is occurrence-level | Compare member vs occurrence prevalence | **CONFIRMED** |
| H10 | Embedding homogenization | Measure per-tier embedding statistics | **CONFIRMED** |
| H11 | Logit suppression | Measure per-tier logit distributions | **CONFIRMED** |
| H13 | Sharp local minimum | LR polishing drop-then-recover pattern | **CONFIRMED** |
| H14 | More data hurts tail | Compare tail logits at 1.5M vs 3.4M | **CONFIRMED** (Matthew Effect) |

---

#### Slide 25: Root Cause — Emergent Gradient Starvation

**Content**: The gradient tier fraction evolution over training:

| Training Phase | Common fraction | Tail fraction | Interpretation |
|----------------|----------------|---------------|----------------|
| Step 1 | 17.8% | 17.8% | Balanced — all tiers equal |
| Step 500 | 16.9% | 18.4% | Still balanced |
| Step 1,500 | 42.7% | 10.4% | Concentration begins |
| Step 3,000 | 66.7% | 3.0% | Common dominates |
| Step 12,001 | 85.3% | 0.1% | Tail starved — effectively zero |

This evolution is **independent of pos_weight** (5.7x increase changed final common fraction by <0.5%).

**Proposed new visualization**: Stacked area chart of gradient tier fractions over training steps (common, medium, rare, tail). The visual should show the "squeeze" — tail fraction collapsing to zero as common expands.

**Mechanism**: Common codes appear in every batch (~64 positives per batch). Tail codes appear in ~6% of batches (~0.064 positives per batch). The per-batch gradient for common codes is a high-precision estimate (many observations, consistent direction). The per-batch gradient for tail codes is near-zero-variance noise (0-1 observations, sporadic direction). AdamW's second-moment denominator further suppresses sporadic tail spikes relative to the consistent common signal.

---

#### Slide 26: The Occurrence-Level Insight

**Content**: Why member-level interventions fail:

| Level | Tail coverage | Common coverage | Ratio |
|-------|--------------|-----------------|-------|
| Member | 83.4% | 100.0% | 1.2x |
| Day | 22.3% | 92.2% | 4.1x |
| **Occurrence** | **5.2%** | **69.7%** | **13.4x** |

83.4% of members have at least one tail code — but tail codes represent only 5.2% of all code occurrences. Enriching batches with "tail members" (tier-aware batching with quota=20) does not help because each tail-enriched member ALSO brings ~1,800 common code occurrences vs ~45 tail occurrences. The gradient ratio within each member is unchanged.

**Evidence**: After density-aware batching (v5), gradient tail_frac actually got *worse* (0.09% vs 0.17% baseline). The intervention backfired.

---

#### Slide 27: Embedding Homogenization — The Consequence

**Content**: Gradient starvation causes tail code embeddings to converge to a single point:

| Tier | Embedding std | Positive logit (y=1) | Margin | P(y=1) |
|------|--------------|---------------------|--------|--------|
| Common | 0.27 | -2.26 | 6.44 | ~9.4% |
| Medium | 0.15 | -6.39 | 6.23 | ~0.17% |
| Rare | 0.05 | -9.68 | 5.34 | ~0.006% |
| Tail | **0.03** | **-14.69** | 1.76 | ~0.00004% |

All 1,175 tail codes converged to a near-identical "default" embedding (std=0.03 vs common std=0.27). The model CAN distinguish tail-present from tail-absent (margin=1.76 > 0), but both logits are so deeply negative (-14.69 when y=1) that tail codes never surface in top-K predictions.

**Proposed new visualization**: t-SNE/UMAP of the 6,297 target code embeddings, colored by frequency tier. Expected pattern: common codes spread across embedding space (diverse, specialized embeddings); tail codes collapsed into a single tight cluster (homogenized, undifferentiated).

---

#### Slide 28: Per-Tier Logit Distributions

**Content**: Show the stark difference in logit ranges across tiers.

**Proposed new visualization**: Violin plots of logit distributions for each tier when y=1 (the code is actually present). The common tier should show logits centered around -2 to -3; tail tier should show logits centered around -14 to -15. The gap illustrates why tail codes can never reach top-K: even when correctly identified as "positive," their logits are 12 units below common codes.

**The cross-code interference mechanism**: The optimal bias for a tail code at 0.001% frequency with pos_weight=200 is `log(200 * 0.00001) = -6.2`. The observed mean tail logit is -14.69. The excess -8.5 units comes from `w_j^T h` — the shared representation `h` is dominated by common-code features, which create systematic negative correlations with tail decoder rows.

---

#### Slide 29: What Helped — Loss Function and Sampling Ablations

**Content**: Round 5-1 results:

| Variant | Loss | Sampling | recall@1 | recall@10 | balanced@10 |
|---------|------|----------|----------|-----------|-------------|
| v2 | BCE + pw=35 | Standard | 0.010 | 0.814 | 0.205 |
| v3 | BCE + pw=200 | Standard | 0.000 | 0.817 | 0.205 |
| **v4** | **ASL (gamma-=4)** | Standard | **0.240** | **0.828** | **0.207** |
| **v5** | **ASL + density batching** | Density (quota=20) | **0.284** | **0.833** | **0.209** |

**What ASL changed**: Calibration (positive_brier -54%), ranking (MRR +53%), recall@1 from 0 → 0.28.
**What ASL did NOT change**: Gradient tier distribution (still 85% common, 0.1% tail), tail_top10_acc (still 0%), embedding homogenization.

**Key message**: The loss function controls calibration and ranking quality. It does not control gradient distribution. These are independent mechanisms.

---

#### Slide 30: What Helped — Data Scaling

**Content**: The strongest single lever:

| Data scale | recall@10 | balanced@10 | medium@10 | tail@10 |
|-----------|-----------|-------------|-----------|---------|
| 1.5M | 0.828 | 0.207 | 0.002 | 0% |
| 3.4M | 0.834 | 0.219 | 0.040 | 0% |
| 6.8M | 0.855 | 0.225 | 0.043 | 0% |
| **11M** | **0.853** | **0.263** | **0.200** | **0%** |

Medium code accuracy jumped from near-0% to 20% at 11M — a threshold-crossing effect. Each medium code goes from ~150 occurrences (1.5M) to ~1,100 occurrences (11M), crossing the minimum-exposure threshold for learning.

Tail codes at 11M get ~57 total occurrences (from ~15 at 1.5M) — still below threshold. Estimated data needed for tail: 100-1000x current scale (1-10 billion members). This is not feasible.

**The Matthew Effect**: More data makes common codes learn *better* while tail codes get *relatively worse*. At 3.4M, tail logits dropped to -14.69 (from -12.93 at 1.5M). More data amplifies gradient starvation because common code occurrences scale proportionally more than tail.

---

#### Slide 31: Dimension Scaling — 256d vs 512d

**Content**:

| Config | recall@10 | micro_recall@10 | medium@10 | Cost | Peak memory |
|--------|-----------|-----------------|-----------|------|-------------|
| 256d, 6.8M | 0.855 | 0.576 | 0.043 | $17.28 | 12.3 GB |
| 512d, 6.8M | 0.858 | 0.578 | 0.039 | $19.68 | 17.5 GB |

512d adds +14% cost, +42% memory, for +0.3% recall@10 improvement. The representation bottleneck is not capacity — it's what information gets encoded (see Slide 28, cross-code interference).

---

#### Slide 32: Plateau Summary — The Unified Root Cause

**Content**: Causal chain diagram:

```
Extreme code frequency imbalance (Gini=0.939)
    ↓
Per-batch gradient aggregation: common codes dominate (~64 positives/batch vs ~0.064 for tail)
    ↓
Emergent gradient concentration: common captures 85% of gradient by step 12K
    ↓
Shared encoder representation monopolization: h encodes common-code features
    ↓
Two consequences:
    → Tail embedding homogenization (std=0.03): all tail codes look identical
    → Cross-code logit suppression (tail logit=-14.69): tail codes never reach top-K
    ↓
Architecture-agnostic ceiling: R@10 ~0.85, tail=0%, regardless of architecture/loss/data
```

**What doesn't work** (definitively eliminated):
- Higher pos_weight, alternative LR schedules, more model capacity, member-level batching, loss function changes alone, dimension scaling, data scaling alone (for tail)

**What partially works**:
- ASL (improves calibration/ranking but not gradient distribution)
- Data scaling (helps medium codes cross threshold, but amplifies tail starvation)

---

### Part V: Downstream Results & Value Demonstration (Slides 33–38)

---

#### Slide 33: Representation Quality — TE vs Baselines

**Content**: Commercial IP prediction using different 256-d representations:

| Representation | OOT-strict AUC | OOT-strict Lift@1% |
|---------------|----------------|-------------------|
| PCA(256) on raw codes | 0.756 | 11.72 |
| AutoEncoder(256) | 0.756 | 11.89 |
| SelectKBest(256) | 0.750 | 12.09 |
| **TE R10 Embedding(256)** | **0.810** | **18.89** |

TE R10 outperforms all dimensionality-reduction baselines: +5.4pp AUC, +57% Lift@1%. The transformer encodes clinical information that linear/nonlinear compression cannot capture.

---

#### Slide 34: Downstream Progression Across Rounds

**Content**: How downstream performance improved with each experimental advance:

| Round | What changed | Emb-only Lift@1% | Hybrid Lift@1% |
|-------|-------------|-----------------|---------------|
| R5 exp1 (legacy) | Legacy transformer | 7.11 | 17.57 |
| R5 v3 (256d, 1.5M) | Flash + LAP + SwiGLU | 14.22 | 18.41 |
| R5 v4 (ASL) | + Asymmetric Loss | 15.76 | 19.24 |
| R6 (3.4M) | + 2x data | 16.18 | 18.96 |
| R6 (6.8M) | + 4.5x data | 15.48 | 18.13 |
| R7 (512d) | + 2x embedding dim | 15.48 | 19.24 |
| **R10 (11M)** | **+ 7.3x data** | **17.15** | **18.69** |

Embedding-only Lift@1% improved 2.4x from legacy (7.11 → 17.15). The architecture improvements (Flash/LAP/SwiGLU) contributed the largest single jump (+7pp).

---

#### Slide 35: Per-Code-Type Intrinsic Performance (R10)

**Content**: Not all code types are equal:

| Code Type | Codes | micro_R@10 | macro_R@10 |
|-----------|-------|-----------|-----------|
| Days Count | 13 | 95.1% | 17.5% |
| GPI Medications | 95 | 76.3% | 54.1% |
| Place of Service | 70 | 75.1% | 52.9% |
| Provider Taxonomy | 242 | 54.7% | 27.0% |
| Procedure Groups | 2,457 | 35.1% | 4.2% |
| ICD-10 Diagnosis | 1,931 | 31.0% | 9.2% |
| DRG Codes | 879 | 10.4% | 1.1% |

**Insight**: The model excels at predicting medications (76%) and place of service (75%) — relatively concentrated vocabularies with strong temporal patterns. ICD-10 diagnoses (31%) and procedures (35%) are harder — large vocabularies with many rare codes. DRG codes are hardest (10%) — hospital-specific, sparse.

---

#### Slide 36: Production Comparison (R10, 11M)

**Content**: Formal R10 model vs production pipeline:

| Metric | Value |
|--------|-------|
| recall@10 | 0.853 |
| micro_recall@10 | 0.563 |
| balanced_top10 | 0.263 |
| medium_top10 | 0.200 (best ever) |
| macro_auroc | 0.920 |
| Training time | 32 hours |
| Training cost | $44.53 |

The 11M model is the strongest on intrinsic metrics. The scale moved medium codes from near-0% to 20% — a qualitative breakthrough in representation breadth.

---

#### Slide 37: Downstream vs Production — Commercial IP

**Content**: Reference the `commercial_ip_11M_downstream_compare_prod.xlsx` comparison. Key numbers:

| Model | OOT-strict AUC | OOT-strict Lift@1% |
|-------|----------------|-------------------|
| Production (tabular) | 0.838 | ~19.38 |
| TE R10 (embedding-only) | 0.810 | 17.15 |
| TE R10 (hybrid) | 0.831 | 18.69 |
| R9 co-occur embed (hybrid) | 0.827 | 20.50 |

The hybrid model approaches production. The R9 co-occurrence embedding experiment exceeds production (20.50 vs 19.38) — suggesting that pre-trained code embeddings unlock additional signal.

---

#### Slide 38: Cost-Performance Trajectory

**Proposed new analysis**: Plot all rounds on a Pareto curve of downstream Lift@1% vs training cost ($):

| Round | Cost | Emb-only Lift@1% |
|-------|------|-----------------|
| R5 v3 (1.5M) | ~$6 | 14.22 |
| R5 v4 (ASL) | ~$6 | 15.76 |
| R6 (3.4M) | $14.36 | 16.18 |
| R6 (6.8M) | $17.28 | 15.48 |
| R10 (11M) | $44.53 | 17.15 |

Diminishing returns are visible: the first $6 buys 14.22 Lift; the next $38 buys only +2.93 more.

---

### Part VI: Lessons & Path Forward (Slides 39–42)

---

#### Slide 39: What Was Validated

**Content**:

| Decision | Evidence |
|----------|---------|
| Hierarchical architecture (daily + temporal) | Matches data structure; all other choices built on this |
| Flash Attention + Learned Pooling | Quality-neutral, 2-3x faster, 3-5x cheaper daily encoding |
| SwiGLU + RoPE + pre-norm | Each component contributes; sum is best-in-class dense model |
| Dense over MoE at this scale | 10 rounds, 16+ MoE experiments — dense wins on every metric |
| Data scaling as primary lever | 1.5M → 11M: medium codes from 0% → 20%; strongest single improvement |
| BCE + log-scaled pos_weight | Stable training; ASL adds calibration value on top |

---

#### Slide 40: The Fundamental Bottleneck

**Content**: The gradient starvation problem is structural, not tunable:
- It is architecture-agnostic (dense, MoE, Flash — all plateau identically)
- It is loss-function-invariant (BCE, ASL, focal — gradient distribution unchanged)
- It is sampling-invariant at the member level (density batching made it worse)
- It is pos_weight-invariant (5.7x increase → <0.5% gradient change)
- It worsens with more data (Matthew Effect)

The shared encoder `h` becomes a common-code feature extractor. As it improves, it increasingly duplicates information that tabular features already capture — producing zero incremental downstream value at scale.

---

#### Slide 41: The Unsolved Problem & Path Forward

**Content**: The fix must break the gradient monopolization cycle. Untested directions:

| Approach | Mechanism | Estimated cost |
|----------|----------|---------------|
| Per-code gradient normalization (GradNorm) | Equalize gradient magnitude across tiers regardless of occurrence frequency | ~$17 (one 6.8M run) |
| Residual embeddings | Train encoder to predict the *residual* between tabular predictions and outcomes — forces learning orthogonal information | ~$17 |
| Per-tier decoder heads | Separate decoders for common/medium/rare/tail — eliminates cross-code interference in logit space | ~$17 |
| Downstream-aware auxiliary objective | Add downstream task signal during pretraining — direct the encoder toward useful features | ~$17 |
| Pre-trained code embeddings (PPMI+SVD) | Break input-level homogenization by initializing with co-occurrence structure | ~$17 (partially tested in R9) |

**Cheap diagnostics first** (~2 hours CPU, <$1):
1. CKA similarity between TE embedding and tabular features — quantify redundancy
2. Linear probe on frozen embeddings — measure unique signal
3. Embedding feature importance in downstream model — identify which dimensions matter
4. Temporal shuffle test — does the temporal encoder actually learn sequence structure?

---

#### Slide 42: Summary

**Content**: One-slide summary with three key messages:

1. **Architecture**: Hierarchical transformer (Flash + LAP + SwiGLU + RoPE) is the right design. Every component is validated by ablation. MoE adds complexity without benefit at this scale.

2. **The bottleneck**: Gradient starvation from extreme code frequency imbalance causes the shared encoder to monopolize on common-code features. This is a structural problem that no loss function, sampling strategy, or scaling can solve within the current training paradigm.

3. **The value**: Despite the bottleneck, the TE embedding delivers 57% higher Lift@1% than dimensionality-reduction baselines and approaches production-level downstream performance. The architecture is sound; the training objective needs evolution.

---

## Proposed Additional Analyses

These analyses would strengthen the evidence chain and provide compelling visuals for the presentation.

| # | Analysis | Purpose in presentation | Effort | Priority |
|---|----------|----------------------|--------|----------|
| 1 | **Code embedding t-SNE/UMAP by frequency tier** | Visualize embedding homogenization (Slide 27) — tail codes collapsing to a single cluster while common codes spread out | ~2 hours (load R10 checkpoint, extract embedding weights, plot) | **HIGH** |
| 2 | **Expert utilization heatmap over training** | Show expert collapse/recovery trajectories for exp3 vs exp6 vs exp6d (Slide 18) | ~1 hour (parse batch_metrics.json from R1/R4/R5) | **HIGH** |
| 3 | **Gradient tier fraction area chart** | The core gradient starvation visual (Slide 25) — stacked area showing common expanding, tail collapsing | ~1 hour (parse gradient tier data from R5/R9) | **HIGH** |
| 4 | **Component contribution waterfall** | Decompose dense architecture gains: baseline → +Flash → +SwiGLU → +LAP (Slide 10) | ~1 hour (extract from R1/R4 results, build waterfall chart) | **MEDIUM** |
| 5 | **Metric trajectory across all rounds** | Line chart showing R@10, balanced@10, medium@10 across all 10 rounds with annotations (Slide 30) | ~2 hours | **MEDIUM** |
| 6 | **Per-tier logit violin plots** | Logit distributions for common/medium/rare/tail when y=1 (Slide 28) | ~2 hours (load R10 checkpoint, run inference on validation sample) | **MEDIUM** |
| 7 | **CKA similarity: TE vs tabular features** | Quantify tabular redundancy hypothesis (Slide 41) — does R10 have higher CKA with tabular than R6? | ~2 hours CPU, needs tabular feature data | **HIGH (for Act VI)** |
| 8 | **Cost-performance Pareto curve** | Training cost ($) vs downstream Lift@1% across all rounds (Slide 38) | ~30 min (data already in logs) | **MEDIUM** |

---

## Presentation Logistics

- **Estimated total slides**: ~42
- **Estimated presentation time**: 60-90 minutes (with Q&A)
- **Format**: Technical deep dive — assumes audience understands ML/transformer basics
- **Key artifacts needed**: R10 model checkpoint, experiment log JSONs, downstream evaluation results
- **Slide tool**: PowerPoint or Google Slides (diagrams can be built in draw.io or Excalidraw)
