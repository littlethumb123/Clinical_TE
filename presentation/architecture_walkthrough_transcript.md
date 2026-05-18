# Slides 1-2 Walk-Through: Full Transcript & Presentation Guide

## How to Use This Document

This is a **speaker transcript** for the first two slides. Slide 1 walks through the architecture bottom-up, explains what happens inside each component, what was replaced from the old model, and where to insert Round 1 evidence. Slide 2 covers the MoE lesson, including the experiment logic, routing diagnostics, and the domain-homogeneity conclusion. Presentation suggestions (callout boxes, annotations, side panels) are marked with `[SLIDE SUGGESTION]`.

---

## Opening (10 seconds)

> "This slide shows the training pipeline. The model uses a member's history to predict which clinical codes will appear on the next day. The 256-dimensional embedding is the internal summary learned for that task."

---

## Slide 1: Narrative with Technical Layer Detail (8-9 minutes)

### 1. Start with the big picture

> "The easiest way to understand this architecture is as a hierarchical encoder: first one day of claims, then up to 200 days for one member. That matches the data structure: codes within a day are an unordered set, while days across time form a sequence."

### 2. Bottom layer: turn raw codes into one day vector

**[Point to: `code_emb [batch_size, 80, 256]` and `learned_attention_pooling`]**

> "At the bottom, each day can contain up to 80 codes drawn from a vocabulary of 75,516. Each code is mapped into a 256-dimensional embedding vector.
>
> Then we collapse those 80 code vectors into one vector that represents that day. We use Learned Attention Pooling. In simple terms, the model learns how much weight to give each code and then takes a weighted combination. So one vector now represents one member's day.
>
> This replaced the old MaxPool-style collapse. MaxPool is a hard winner-take-all rule. Learned attention is a softer summary, so it keeps more information and was faster in practice."

### 3. Add demographic context

**[Point to: `Residual sum: code_emb + code_pooled + age_emb + gender_emb + LOB`]**

> "Next we add demographic context: age, gender, and line of business. These learned embeddings are added to the day representation.
>
> After this step, the member is represented as a sequence of up to 200 day vectors, each 256-dimensional."

### 4. Middle stack: encode the 200-day history

**[Point to: the stacked `Layer_0` through `Layer_5` block]**

> "The middle of the architecture is a 6-layer temporal encoder, and each layer follows the same sequence of operations.
>
> The input to one layer is a sequence of day vectors. Each vector already summarizes one day of clinical activity plus demographic context. The first step is **pre-norm**. We apply LayerNorm before the main computation so every day vector is on a stable scale. That matters because clinical histories can contain very uneven signal strength: a routine refill day and a hospitalization day should both be trainable without the larger-magnitude pattern destabilizing optimization. Pre-norm keeps the residual pathway clean and makes deep transformer training much more stable.
>
> From that normalized input, we form queries, keys, and values. Then we apply **RoPE** to the queries and keys. RoPE injects relative position directly into the attention geometry, so the model can treat the same code pattern differently depending on whether it happened yesterday, 30 days ago, or 6 months ago. For clinical data, that matters because timing changes meaning: a recent ER visit, a long-standing chronic diagnosis, and a medication started months earlier should not be interpreted the same way.
>
> Next comes **Flash Attention** with the **causal mask**. Conceptually this is still self-attention: each day looks back over earlier days and decides which past events matter for understanding the current state. The causal mask enforces the training logic by blocking any view of the future, so day t can only use days 1 through t. Flash Attention does not change that modeling idea; it changes how the computation is executed. Instead of materializing the full attention matrix in memory, it uses a fused, memory-efficient kernel, which lets us train faster and cheaper at this sequence length.
>
> The attention output is added back through a residual connection, and then the layer does a second **pre-norm** before the feed-forward block. That block is **SwiGLU**, which replaces a standard FFN. In a plain FFN, every position goes through the same nonlinear expansion and projection. In SwiGLU, one path proposes candidate features and another path gates them, so the model can selectively amplify useful signal and suppress weaker or noisier combinations. For clinical sequences, that is helpful because one day vector may mix chronic disease burden, acute utilization, medications, and administrative noise all at once. The gate gives the model finer control over what should survive.
>
> Finally, that SwiGLU output is added back through another residual connection. That final sum becomes the input to the next layer. So one layer does not just pass day t upward. It produces a more context-aware version of day t, informed by the clinically relevant parts of the member's prior timeline."

### 5. What happens to one day vector across layers

**[Point to: the left-hand annotations such as `Day3_lay1 = f(...)`]**

> "The left side of the diagram shows what happens to one specific day vector as it moves upward through the stack. Take day 4 entering layer 0. At that moment, the vector is still just a compressed summary of that day's codes and demographic context. It knows what was present on day 4, but it does not yet know what day 4 means relative to the earlier history.
>
> Inside layer 0, attention compares day 4 with days 1 through 4 and asks which earlier days are most useful for interpreting the current day. So if day 4 contains an acute event, the model can look back to recent diagnoses, prior utilization, or medication patterns and decide what context matters. The output is not a copy of one earlier day. It is an updated version of day 4 that blends its own content with the most relevant parts of the allowed past. SwiGLU then reshapes that blended signal, strengthening clinically meaningful combinations and damping weaker or noisier ones.
>
> So after layer 0, the vector still sits at the day 4 position, but its meaning has changed. It no longer means only 'what happened on day 4.' It now means 'what day 4 represents in the context of days 1 through 4.'
>
> In layer 1, the same process happens again, but now the inputs are no longer raw day summaries. They are already contextualized representations from the previous layer. That means day 4 is now attending not just to earlier events, but to earlier interpreted states. This is where the abstraction deepens. A refill after several stable outpatient days may start to look like routine maintenance. The same refill after repeated acute encounters may instead look like escalation or instability.
>
> By the higher layers, each position becomes a compact clinical state summarizing the most relevant history up to that day. The position itself does not move, and the vector stays 256-dimensional, but the information inside it becomes more contextual, more selective, and more predictive. That is what the f means. It is the whole learned layer transformation applied to that day position. So when the next layer starts, it is not receiving raw day 4 anymore. It is receiving a richer state for day 4, already informed by the member's prior trajectory, while still respecting the causal rule that nothing from the future can enter." 

### 6. Output target and member embedding

**[Point to: `Member embedding vector (last day embedding)` and `Predicted codes for each member next day [batch_size, 200, 6297]`]**

> "At the top, the model makes the training prediction. For every day position, it predicts which grouped clinical codes will appear on the next day. There are 6,297 target codes, and this is multi-label because several codes can appear together.
>
> The member embedding is produced in the middle of this process. After the 6 temporal layers, we take the representation at the last valid day. That 256-dimensional vector is the member embedding.
>
> So the full flow is: encode each day, add demographic context, model the 200-day history, predict next-day grouped codes, and use the final history representation as the embedding."

---

## Slide 1: Optional Visual Cues

`[SLIDE SUGGESTION]`: Add a small annotation at the bottom: `75,516 input codes -> embedding lookup -> one day vector`.

`[SLIDE SUGGESTION]`: Add a short label on the arrow into the temporal stack: `up to 200 days x 256-d vectors`.

`[SLIDE SUGGESTION]`: Add a small tooltip near the left-hand formula: `f = updated representation after one transformer layer`.

`[SLIDE SUGGESTION]`: Add a short label at the output: `predict 6,297 grouped next-day codes | embedding = last valid day representation`.

---

## Slide 1: Delivery Map

| Beat | What to emphasize | Approx time |
|------|-------------------|-------------|
| Training objective | Predict next-day grouped codes; embedding is intermediate | 20-30s |
| Hierarchical structure | One-day encoder, then 200-day history encoder | 25-35s |
| Daily collapse + demographic context | One day becomes one vector, then add member context | 45-60s |
| Temporal encoder | Pre-norm, RoPE, Flash Attention, causal mask, SwiGLU, residual flow | 210-240s |
| Across-layer intuition | What `f` means and how one day vector evolves | 90-120s |
| Output + embedding | 6,297 targets; last valid day becomes embedding | 30-40s |

---

## Slide 1: Key Narrative Beats

1. Start with the training objective so the audience knows the embedding is an intermediate output, not the only output.

2. Explain the hierarchy early: one day is a set, 200 days is a sequence.

3. Keep the bottom-up story simple: daily collapse, demographic context, temporal encoder, prediction head.

4. When you explain the 6-layer stack, focus on what changes for the audience: Flash Attention makes the same attention idea more efficient, and SwiGLU is a gated upgrade over a plain FFN.

5. Use the left-hand `f(...)` annotation to explain representation learning in plain language: each layer produces a more informed version of the same day.

6. End by reconnecting the two outputs: the supervised task is next-day code prediction, and the embedding is the final hidden state used for downstream work.

---

# Slide 2 MoE Lesson: Full Transcript & Presentation Guide

## How to Use This Section

This section gives you two speaker-note versions for Slide 2:

- **Version A** keeps the original shorter talk track.
- **Version B** is a punchier **2-3 minute version** that is easier to deliver live while still covering the full logic.

Use Version A if you want a tighter handoff from Slide 1. Use Version B if Slide 2 is carrying more of the experimental story.

---

## Slide 2 Framing

**Slide title**: `Mixture of Expert (MoE): right Idea, wrong scale`

**Core job of this slide**:
1. Explain MoE in plain language using the top-right FFN-to-router diagram.
2. Walk the audience through what you tried without drowning them in implementation detail.
3. Explain the collapse metric and the aux-loss vs auxiliary-free logic.
4. Land the real conclusion: the model was not finding clean specialist regimes because the claims domain is much more homogeneous than the original hypothesis assumed.

---

## Version A: Original Short Transcript (~90-120 seconds)

> "MoE has become a standard scaling pattern in frontier language models because it lets you increase total capacity without sending every token through one huge feedforward block. In plain language, instead of one FFN doing all the work, the router chooses a small number of specialized mini-networks, or experts, for each token. That is what the diagram on the right is showing: we replace the standard FFN with a router and a bank of experts. In the shared-expert version, one expert stays available as a generalist safety net while the router chooses the specialists."
>
> "Our original hypothesis was that claims data might benefit from the same idea. Clinical populations are heterogeneous: some members are dominated by chronic cardiometabolic care, some by acute events, some by oncology or high-complexity utilization. If those were truly different computational regimes, MoE should let experts specialize by patient archetype and beat the dense TE model without increasing per-token FLOPs."
>
> "So I started with the naive MoE setups in the table. The 8-expert model is the standard top-2 baseline. The 7-plus-1 shared-expert version keeps one always-on generalist so the model has a safe fallback instead of forcing every token through narrow specialists. The 16-expert version is more fine-grained: each expert is smaller, and the router can make a more granular assignment. All of those variants were materially worse than the dense TE baseline."
>
> "The main warning signal was expert collapse. In plain language, collapse means the router keeps sending most tokens to a small subset of experts while the others sit mostly idle. In our logs, an expert counts as collapsed when its load share falls below 5%, averaged over the batches in an epoch. So a value like 1.98 does not mean exactly two experts died permanently. It means that, on average across that epoch, about two experts were effectively underused. The companion CV metric tells us how uneven the overall traffic is: lower is healthier, higher means the router is concentrating load into a few experts."
>
> "That led to the next round of fixes. I first corrected the MoE boundary and moved routing later in the stack, but those changes did not solve the problem. Then I added an auxiliary load-balancing loss. The easiest way to think about that is regularization: just like L2 regularization says don't put all the weight in one place, this term says don't put all the traffic on one expert. It helped a bit, but it still stayed below the dense baseline. After that I switched to DeepSeek-style auxiliary-free bias correction. The difference is important: instead of adding another loss term to backpropagate, it nudges routing with a running bias update outside the main task gradient. That removed the worst gradient conflict and performance improved sharply. But it also exposed the dilemma: as task performance improved, routing balance often got worse. In other words, the model seemed to predict best when it kept reusing the same few experts."
>
> "At that point the conclusion changed. The story was no longer that we just needed a smarter balancing penalty. The deeper issue is that the router is not discovering clean specialist subproblems. And that is where domain homogeneity matters. In natural language, MoE can separate very different kinds of computation, like code completion, translation, summarization, or math reasoning. Claims data is different. It is more like one constrained clinical dialect: members vary in combinations and severity, but they are not speaking different computational languages. The saturation analysis makes that concrete. By day 10 only 24% of codes are still novel within a member timeline, and by day 100 that falls to 7%. 95.2% of unique same-day code pairs already involve at least one common code. The top 1% of pairs explain 66.1% of all pair occurrences. And when we scale from 1% of members to 100%, entropy barely moves and JS divergence is near zero. New members mostly add more of the same structure, not new regimes for separate experts."
>
> "So the takeaway is: MoE was the right architecture to test, and it is a proven industry pattern, but in this setting it was the wrong scale and probably the wrong data regime. After 12 experiments across three rounds, the best MoE only reached parity in a larger run, and the production 256d comparison stayed effectively tied while paying 39.8% more parameters, 20.2% more memory, 13.7% lower throughput, and more engineering complexity. For Clinical TE, the dense architecture is the better operating point."

---

## Version B: Punchy 2-3 Minute Transcript

### Opening

> "One major lesson we learned was around the use of MoE inside the transformer. I also know some folks here are exploring LLM-style solutions for other problems, so I wanted to share what we learned in case it is helpful."

> "MoE has become a standard scaling idea in large language models. In plain language, instead of one feedforward block doing all the work, you have a router that sends each token to a small number of experts. So you get more model capacity, but you do not pay the full dense compute every time."

### Explain the diagram in plain language

> "That is what the top-right diagram is showing. On the left is the normal feedforward network. On the right, we replace it with a router and a bank of experts. The router decides which experts will process a token. In the shared-expert version, one expert stays as a general fallback, so the model always has one broad expert to lean on."

### Original hypothesis

> "The motivation was straightforward. Clinical populations are heterogeneous. Different members have different disease burden, utilization patterns, and care trajectories. So the hypothesis was: maybe MoE would let experts specialize by clinical taxonomy or member archetype."

### Walk the experiment table simply

> "So I started with the straightforward setups first: 8 experts as the basic baseline, then 7 experts plus 1 shared expert, and then 16 experts as a more fine-grained version. The shared expert was meant to capture the general patterns. The 16-expert version was meant to create more granular specialization. But the first result was very clear: all of these early MoE variants were much worse than the dense TE baseline."

### Explain collapse clearly

> "The main warning sign was expert collapse. That means the router is not discovering meaningful specialization. It keeps sending most of the traffic to a small number of experts, while the others are barely used. In our metric, an expert is considered collapsed if its average load share falls below 5% over the epoch. So when you see 1.98 collapsed experts, that does not mean exactly two experts died forever. It means that on average, about two experts were effectively underused. The CV number is the companion metric. It tells us how uneven the expert load is. Higher CV means worse balance."

### Explain how the logic evolved

> "From there, the work became a step-by-step debugging sequence. I fixed the MoE boundary, changed where routing enters the stack, and then added auxiliary load-balancing loss. The easiest way to think about the auxiliary loss is as regularization for the router. It penalizes the model for overusing the same experts. That helped a little, but it still stayed below the dense baseline."

> "The biggest jump came from DeepSeek-style auxiliary-free bias correction. The difference is important. Auxiliary loss adds another objective into backpropagation. Auxiliary-free bias correction does not. It nudges the router through a bias update outside the main task gradient. Performance improved a lot, but balance got worse. That was the key dilemma. The model was basically telling us: I predict better when I keep reusing the same few experts."

### Domain homogeneity conclusion

> "That is what changed the conclusion. In natural language, MoE often works because there are genuinely different computational modes to separate, like translation, summarization, code, or reasoning. Claims data is not like that. It behaves much more like one constrained clinical language. Members vary by mix and severity, but not by fundamentally different computational patterns. The information analysis supports that. By day 50, only 11% of codes are still novel within a member timeline. 95.2% of unique same-day code pairs already include at least one common code. The top 10% of pairs explain 91.7% of all pair occurrences. And when we add more members, the distribution barely changes. So new members mostly add more of the same structure, not clean new regimes for experts to specialize on."

### Land the punch

> "So the final takeaway is simple: MoE was absolutely the right thing to test, but for this dataset it was the wrong operating point. We ran 12 experiments across three rounds, fixed every obvious pathology, and the best case still only reached parity while paying more in parameters, memory, throughput, and engineering complexity. For Clinical TE, the dense model is the better choice."

---

## Slide 2 Delivery Map

| Beat | What to emphasize | Approx time |
|------|-------------------|-------------|
| MoE concept | Industry-standard idea, router chooses experts | 20-30s |
| Diagram | FFN replaced by router + expert bank, shared expert as fallback | 20-30s |
| Hypothesis | Patient heterogeneity might create specialist regimes | 15-20s |
| Experiment table | 8 experts, shared expert, 16 fine-grained experts | 20-30s |
| Collapse metric | Underused experts + load imbalance | 20-30s |
| Aux vs aux-free | Regularization vs bias correction outside task gradient | 30-40s |
| Conclusion | Homogeneous clinical dialect, dense wins | 30-40s |

## Slide 2 Optional Callouts

`[SLIDE SUGGESTION]`: If you present the compact table on the slide, circle only three rows while speaking: the dense baseline, the first MoE baseline, and the best aux-free run. That keeps the audience oriented without making them parse every row.

`[SLIDE SUGGESTION]`: Add a small one-line tooltip near the collapse column: `collapse = experts below 5% load share; CV = load imbalance across experts`.

`[SLIDE SUGGESTION]`: If you have room for one extra annotation, add this sentence under the homogeneity bullet: `Claims varies by mix and severity, not by distinct computational language.` That line is the conceptual bridge from the routing dilemma to the final conclusion.

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
