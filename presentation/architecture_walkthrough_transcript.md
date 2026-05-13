# Slide 1 Architecture Walk-Through: Full Transcript & Presentation Guide

## How to Use This Document

This is a **speaker transcript** for the architecture slide. It walks through the diagram bottom-up, explains what happens inside each component, what was replaced from the old model, and where to insert Round 1 evidence. Presentation suggestions (callout boxes, annotations, side panels) are marked with `[SLIDE SUGGESTION]`.

---

## Opening (5 seconds)

> "This diagram shows the full end-to-end architecture. I'm going to walk you through it bottom-up — the same direction the data flows — so you can see exactly how a member's raw claims become a 256-dimensional embedding."

---

## Section 1: Code Embedding Layer (Bottom of Diagram)

**[Point to: `code_emb [batch_size, 80, 256]` at the bottom]**

> "We start at the very bottom. Each member's clinical history is organized by day. On any given day, a member can have up to 80 medical codes — these are ICD-10 diagnoses, CPT procedures, GPI medications, DRG codes, revenue codes, provider taxonomy, and place of service. That's a total input vocabulary of 75,516 unique codes.
>
> Each code gets looked up in a learned embedding table and becomes a 256-dimensional vector. So for one day, we have a matrix of shape [80, 256] — 80 code vectors, each 256-d."

**What changed from the old model**: Nothing at this layer — both exp1 and exp2b use the same embedding lookup. The differences start in how we aggregate these 80 vectors into one.

`[SLIDE SUGGESTION]`: At the bottom of the diagram, add a small annotation: `75,516 input codes → 256-d embedding lookup`. This grounds the audience on vocabulary scale.

---

## Section 2: Learned Attention Pooling (Below the Residual Sum)

**[Point to: `learned_attention_pooling` box — "learned q → k, v proj → softmax"]**

> "Now we need to collapse those 80 code vectors into a single vector that represents the entire day. This is the Daily Encoder.
>
> We use **Learned Attention Pooling**. There's a single learned query vector — think of it as the model asking 'what happened today that matters?' This query attends to all 80 code embeddings through key and value projections, computes attention weights via softmax, and produces a weighted sum. The result is one 256-d vector per day.
>
> A critical design decision here: **there is no positional encoding** at this level. Why? Because codes within a single claim day are an unordered set. If a member has diabetes (E11.9), hypertension (I10), and an insulin prescription (Z79.4) on the same day, there's no meaningful sequence — they're co-occurring events. The model treats them as a bag, which is exactly right for this data."

**[Point to: `mlp + norm` box below the residual sum]**

> "After pooling, there's a small post-pooling MLP — a two-layer feedforward with GELU activation and LayerNorm — that adds a bit of nonlinear capacity to the daily representation before it enters the temporal encoder."

### What changed from the old model (OLD → NEW)

> "In the previous architecture, this step was completely different. The old model used a **1-layer PyTorch transformer with 4 attention heads** to process the 80 codes, followed by **MaxPool** — which just takes the element-wise maximum across all 80 positions. MaxPool is a hard selection: only the single strongest activation per dimension survives. Everything else is discarded.
>
> Learned Attention Pooling replaces both the daily transformer and the MaxPool with a single soft attention operation. It's a weighted sum, not a hard max — the model learns what to attend to."

### Round 1 Evidence

> "This is the one clean single-variable ablation from Round 1. exp2 used MaxPool, exp2b used Learned Attention Pooling — everything else identical, same data, same training."

| Metric | MaxPool (exp2) | LAP (exp2b) |
|--------|---------------|-------------|
| recall@10 | 0.9430 | **0.9472** |
| val_loss | 0.00275 | **0.00273** |
| Training cost | $1.87 | **$1.52** |
| Daily encoding speed | 1x | **3-5x faster** |

> "LAP matches or slightly beats MaxPool on quality, while being 3-5x faster on the daily encoding step and 19% cheaper overall. The speed gain comes from eliminating the sequential max operation — a single attention pass is more parallelizable."

`[SLIDE SUGGESTION]`: Place a small **comparison callout box** next to the `learned_attention_pooling` component on the diagram. Two columns: "Old: MaxPool" vs "New: Learned Attention Pooling". Below that, a single row: "recall@10: 0.943 → 0.947, 3-5x faster". Keep it tight — the audience sees the evidence right where the component lives on the architecture. This is the strongest visual proof point because it's a clean single-variable change.

---

## Section 3: Demographic Injection (Residual Sum)

**[Point to: `Residual sum: code_emb + code_pooled + age_emb + gender_emb + LOB` box]**

> "Before entering the temporal encoder, we inject demographic context. Age in months, gender, and line of business each get their own learned embedding. These are added to the pooled day vector via a residual sum — meaning the demographics modulate the representation but don't overwrite it.
>
> The output of this step is a tensor of shape [batch_size, 200, 256] — up to 200 days of history, each represented as a 256-d vector that combines clinical codes and demographics."

**What changed**: The demographic injection mechanism is the same in both models. This was a good design from the start — residual addition is the standard way to inject conditioning information without creating bottlenecks.

`[SLIDE SUGGESTION]`: The `LOB_emb`, `age_emb`, `gender_emb` boxes at the bottom of the diagram are already clear. Consider adding the tensor shape `[batch, 200, 256]` as an annotation on the arrow going into the temporal encoder stack. This tells the audience "from here, we're processing a sequence of 200 day-vectors."

---

## Section 4: The Temporal Encoder (Inter-day Encoder, 6 Layers)

**[Point to: the stacked `Layer_0` through `Layer_5` block labeled "Inter-day Encoder"]**

> "Now comes the core of the model — the temporal encoder. This is a 6-layer transformer that processes the 200-day sequence. Each day position enters as a 256-d vector, and each layer progressively enriches that representation by attending to the clinical history.
>
> Let me walk through what happens inside a single layer, then explain why stacking 6 of these creates something powerful."

### 4.1 Inside a Single Layer

**[Point to: the "Look closer" / "Next layer" detail panel on the right side of the diagram]**

> "Each layer has two blocks, and both follow the same pattern: **pre-norm → operation → residual add**."

#### Block 1: Attention

> "**First, pre-norm.** The input is normalized via LayerNorm across the 256 embedding dimensions. This stabilizes gradients and is critical for training stability — it's the 'pre-norm' pattern used by GPT and LLaMA, as opposed to the 'post-norm' pattern from the original transformer paper.
>
> **Then, Flash Attention.** The normalized input gets projected into queries, keys, and values through three separate linear layers — each 256→256, no bias. These are reshaped into 8 attention heads, each with a 32-dimensional head space.
>
> Before computing attention, we apply **Rotary Position Embedding (RoPE)** — a rotation in the complex plane applied to queries and keys. RoPE encodes relative position: how far apart two days are in the sequence. Unlike fixed sinusoidal embeddings, RoPE naturally captures that 'yesterday' matters more than '90 days ago' without hard-coding a decay function.
>
> Then comes the actual attention computation — and this is where the **causal mask** is critical."

#### The Causal Mask (Key Concept)

> "The causal mask is a lower-triangular matrix applied at every layer. It says: when computing the representation for day *t*, you can only attend to days 1 through *t*. Day *t* cannot see day *t+1* or any future day. This is a hard constraint — future attention weights are set to negative infinity before softmax, so they become exactly zero.
>
> Why is this essential? Two reasons:
>
> **First, it defines the pre-training objective.** We're predicting next-day codes. If day *t* could see day *t+1*, it would just copy the answer. The causal mask forces the model to predict the future from the past only.
>
> **Second, it prevents temporal leakage during embedding extraction.** When we extract a member's embedding at their last valid day, we need to guarantee that embedding contains only information from the member's observed history — not from future events that haven't happened yet. The causal mask guarantees this structurally, not just by convention."

`[SLIDE SUGGESTION]`: Add a small **causal mask visual** as a side annotation near the Layer stack. A 5×5 lower-triangular matrix with checkmarks (✓) and crosses (✗) is enough — don't need all 200×200. Label it: "Day *t* sees only days 1...*t*". This is the single most important concept for the audience to understand about temporal integrity.

#### What the causal mask achieves across layers

> "Here's the subtle point: the same mask shape is applied at every layer, but what's being masked is fundamentally different.
>
> After Layer 0, day 4's representation is a function of the raw inputs from days 1 through 4: `f(E₁, E₂, E₃, E₄)`.
>
> After Layer 1, day 4 attends to those blended representations. So it sees day 3's *interpretation* of days 1-3, not just raw day 3. This creates second-order temporal reasoning — patterns of patterns.
>
> By Layer 5, each day position encodes deep, multi-hop abstractions of the entire preceding history. Layer 0 might learn 'yesterday's ER visit is relevant.' Layer 3 might learn 'the medication started 30 days ago, the lab changed 20 days ago, and today's adverse event connects them.' Layer 5 might learn 'this member's overall trajectory resembles the pattern before hospitalization.'
>
> The causal mask at every layer maintains a strict invariant: **day *d*'s representation at any layer can only be a function of the original inputs E₁ through E_d.** This must hold recursively through all 6 layers for temporal integrity."

`[SLIDE SUGGESTION]`: The left side of the architecture diagram already shows the annotation `Day3_lay1 = f(Day1_lay0, Day2_lay0, Day3_lay0)` etc. When narrating, point directly to this. If space allows, consider a small 3-row progression annotation:
```
After L0: Day4 = f(E₄ only)
After L1: Day4 = f(E₁, E₂, E₃, E₄) — first-order
After L5: Day4 = deep abstraction of days 1-4
```
This makes the "representations get richer across layers" point visceral.

#### Block 2: SwiGLU FFN

**[Point to: the `W_gate → SiLU`, `W_up`, `W_down` detail in the "Next layer" panel]**

> "After attention, the second block processes each day position independently through a feed-forward network.
>
> We use **SwiGLU** — a gated activation function from the PaLM and LLaMA family. It works like this: the input is projected through two parallel paths. One path goes through a Swish (SiLU) activation to produce a gate signal. The other path produces a value. These are multiplied element-wise — the gate selectively passes information — then projected back down to 256 dimensions.
>
> The key dimensions are: input 256 → gate and value each project to 704 (that's the 8/3 × d_model scaling from LLaMA, rounded to a multiple of 64) → element-wise multiply → project back to 256."

> "Again, this block uses **pre-norm** — LayerNorm before the FFN, then residual add with the un-normalized input. And there's a residual connection: the original input to this block is added to the FFN output. This means if the FFN learns nothing useful for a particular day position, the representation passes through unchanged."

### What changed from the old model (6 variables, bundled)

> "The temporal encoder is where most of the architectural modernization happened. The old model used stock PyTorch `TransformerEncoderLayer` with:
>
> - **Post-norm** (normalize after attention, not before) — less stable gradients
> - **GELU activation** in a standard 2-layer FFN (256 → 1024 → 256) — no gating
> - **16 attention heads with 16-d head dimension** — each head has very limited capacity
> - **No RoPE** — no explicit positional encoding at all in the temporal encoder
> - **Standard PyTorch attention** in FP32 — slower, more memory
>
> The new model bundles six changes: (1) Flash Attention kernel, (2) FP32→FP16, (3) 16 heads/16-d → 8 heads/32-d, (4) GELU→SwiGLU, (5) post-norm→pre-norm, (6) no position→RoPE."

### Round 1 Evidence for the Bundle

> "I want to be precise about the evidence here. These six changes were applied together as a bundle — we did NOT ablate each one individually in Round 1. What we confirmed is that the bundle as a whole is **quality-neutral with a 25% cost reduction**."

| Metric | exp1 (old, all 6 old components) | exp2 (new, all 6 new components) |
|--------|----------------------------------|----------------------------------|
| recall@10 | 0.9474 | 0.9430 |
| val_loss | 0.00275 | 0.00275 |
| Training cost | $2.48 | $1.87 (-25%) |
| Throughput | 1x | ~1.33x |

> "Same quality, 25% cheaper, 33% faster. The bundled modernization pays for itself in throughput and cost. It doesn't add quality — it preserves quality while making training significantly more efficient. These are well-established best practices from the LLM literature (PaLM, LLaMA, Mixtral), validated here for our clinical domain.
>
> The honest claim is: 'We bundled modern transformer best practices and confirmed the bundle is quality-neutral.' Not 'we ablated each component individually.'"

`[SLIDE SUGGESTION]`: For the temporal encoder evidence, I recommend a **"What changed" comparison strip** along the right edge of the slide, keyed to the layer detail panel. Format as a compact table:

```
TEMPORAL ENCODER MODERNIZATION (exp1 → exp2, Round 1)
──────────────────────────────────────────────────────
Old                          New
──────────────────────────────────────────────────────
PyTorch attention (FP32)  →  Flash Attention (FP16)
16 heads × 16d            →  8 heads × 32d
GELU FFN (256→1024→256)   →  SwiGLU (256→704→256)
Post-norm                 →  Pre-norm
No position encoding      →  RoPE
──────────────────────────────────────────────────────
Quality: identical (R@10: 0.947 → 0.943)
Speed:   +33%  |  Cost: -25%
```

Then, next to the `learned_attention_pooling` box, a **separate, highlighted callout** for the single clean ablation:

```
DAILY POOLING (exp2 → exp2b, single variable)
──────────────────────────────────────────────
MaxPool → Learned Attention Pooling
Quality: 0.943 → 0.947  |  Speed: 3-5x faster
```

Visually distinguish these two evidence blocks: one is a confirmed bundle, the other is a clean ablation. The audience should see that the LAP result is the one you can point to with full ablation confidence.

---

## Section 5: Member Embedding Extraction

**[Point to: `Member embedding vector (last day embedding)` arrow leaving the Layer_5 box]**

> "After all 6 layers, each of the 200 day positions has a rich representation. We extract the member's embedding from the **last valid day position** — the most recent day in their history. This single 256-d vector encodes the member's entire longitudinal clinical trajectory, as seen through the causal lens of 6 transformer layers."

---

## Section 6: Prediction Head

**[Point to: `norm + dropout(0.1)` → `Predicted codes for each member next day [batch_size, 200, 6297]`]**

> "For pre-training, every day position produces a prediction. The representation goes through a final LayerNorm and dropout(0.1), then a linear projection from 256 to 6,297 — the output vocabulary of grouped clinical codes.
>
> This is a **multi-label** prediction. On any given day, a member could have zero codes or many codes. So we use BCEWithLogitsLoss, not softmax cross-entropy. Each of the 6,297 output dimensions is an independent binary prediction: will this code appear tomorrow?
>
> The **dual vocabulary** design is intentional — 75,516 input codes for full granularity, but only 6,297 output targets. Not all input codes are worth predicting. The output vocabulary is filtered to clinically meaningful, recurring codes — noise like billing artifacts and one-time codes is excluded."

`[SLIDE SUGGESTION]`: At the top of the diagram, annotate the output box with: `6,297 target codes (filtered from 75,516 input) | BCEWithLogitsLoss | multi-label`. This is a common audience question ("why different input/output sizes?") — preempting it on the slide saves time.

---

## Section 7: End-to-End Data Flow Summary (Closing)

> "So let me trace the full path one more time, quickly:
>
> 1. **Raw codes** (up to 80 per day) → embedding lookup → [80, 256]
> 2. **Learned Attention Pooling** → one 256-d vector per day — codes are an unordered set
> 3. **Demographic injection** (age, gender, LOB) via residual sum
> 4. **6-layer temporal encoder** — Flash Attention with causal mask (each day sees only past), RoPE for relative position, SwiGLU FFN for nonlinear processing, pre-norm for gradient stability
> 5. **Member embedding** extracted at last valid day — 256 dimensions capturing the full clinical trajectory
> 6. **Prediction head** projects to 6,297 target codes — next-day multi-label prediction
>
> The entire model is 25.3M parameters. It trains on 11M members in 32 hours for $44.53. Every design choice — from the unordered daily encoder to the causal temporal mask to the dual vocabulary — is motivated by how the clinical data is naturally structured."

---

## Appendix: Presentation Data Integration Strategy

### Principle: Evidence lives where the component lives

Don't put all results on a separate slide. Anchor each evidence callout physically next to the architecture component it validates. The audience should see "this component → this is why we chose it → here's the proof" in one visual scan.

### Recommended slide layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ARCHITECTURE DIAGRAM              │  EVIDENCE PANEL               │
│   (center, ~60% width)             │  (right strip, ~35% width)    │
│                                     │                               │
│   [Full bottom-up architecture      │  ┌──────────────────────┐    │
│    as in the pasted image]          │  │ TEMPORAL ENCODER      │    │
│                                     │  │ (bundled change)      │    │
│                                     │  │ Old → New             │    │
│   ← Arrow from temporal encoder     │  │ R@10: same            │    │
│     to evidence panel               │  │ Speed: +33%           │    │
│                                     │  │ Cost:  -25%           │    │
│                                     │  └──────────────────────┘    │
│   ← Arrow from LAP box             │                               │
│     to evidence panel               │  ┌──────────────────────┐    │
│                                     │  │ DAILY POOLING ★       │    │
│                                     │  │ (single-variable)     │    │
│                                     │  │ MaxPool → LAP         │    │
│                                     │  │ R@10: 0.943 → 0.947  │    │
│                                     │  │ Speed: 3-5x faster    │    │
│                                     │  └──────────────────────┘    │
│                                     │                               │
│   CAUSAL MASK mini-visual           │  ┌──────────────────────┐    │
│   (5×5 triangular matrix,           │  │ WHAT THE MODEL DOES   │    │
│    near layer stack)                │  │ Medications: 76% R@10 │    │
│                                     │  │ Place of Svc: 75%     │    │
│                                     │  │ Emb-only AUC: 0.810   │    │
│                                     │  │ vs PCA: +110% Lift@1% │    │
│                                     │  └──────────────────────┘    │
│                                     │                               │
└─────────────────────────────────────────────────────────────────────┘

 Bottom annotation bar:
 25.3M params  |  $44 to train  |  32 hours on 11M members  |  256-d embedding
```

### What to say vs what to show

| Content | Say it (transcript) | Show it (slide) |
|---------|--------------------|-----------------| 
| Causal mask mechanism | Full explanation with day-by-day trace | 5×5 triangular matrix mini-visual |
| Layer-by-layer abstraction | "Representations get richer" narrative | The existing `Day3_lay1 = f(...)` annotation on the diagram |
| exp1→exp2 bundle | "6 changes, quality-neutral, 25% cheaper" | Compact Old→New comparison strip |
| exp2→exp2b LAP | "Clean single-variable ablation, 3-5x faster" | Highlighted callout box (★ = clean ablation) |
| SwiGLU internals | Explain gate × up → down flow | The `W_gate → SiLU`, `W_up`, `W_down` detail already on diagram |
| RoPE mechanics | Brief: "rotation encodes relative position" | No separate visual needed — mention when pointing to attention box |
| Pre-norm vs post-norm | Brief: "normalize before, not after — GPT/LLaMA convention" | Include in Old→New comparison strip |
| Dual vocabulary | Explain why 75K input ≠ 6K output | Annotate top and bottom of diagram with code counts |
| Model vital stats | Close with: "25.3M params, $44, 32 hours" | Bottom annotation bar |

### Pacing guide

| Section | Time | Key moment |
|---------|------|-----------|
| Code embedding | 30s | Set the scale: 75K codes |
| Learned Attention Pooling + evidence | 90s | **Star moment**: clean ablation result |
| Demographics | 20s | Quick — nothing controversial |
| Temporal encoder + causal mask | 120s | **Deep moment**: the causal mask explanation is what makes the audience truly understand the model |
| SwiGLU + modernization evidence | 60s | Honest framing: "bundled, not individually ablated" |
| Member embedding + prediction head | 30s | Land the dual vocabulary point |
| End-to-end summary + vital stats | 30s | **Punch line**: 25.3M params, $44, 110% over PCA |
| **Total** | **~6 min** | |

---

## Key Narrative Beats to Hit

1. **"Mirror the data, not the literature"** — every design choice is motivated by how clinical data is structured (unordered codes within a day, ordered days across time, dual vocabulary for noise filtering)

2. **"The causal mask does double duty"** — it defines the training objective AND prevents temporal leakage. This is elegant engineering, not just a standard transformer trick.

3. **"Two types of evidence"** — the LAP ablation is clean (single variable). The temporal encoder modernization is a confirmed bundle (quality-neutral, cost-reducing). Be honest about the distinction — it builds credibility.

4. **"The model is small and cheap"** — 25.3M parameters is 4,000x smaller than GPT-2. $44 to train. This is not a massive compute project — it's an efficient, well-targeted architecture. This surprises most audiences and is worth emphasizing.
