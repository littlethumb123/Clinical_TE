This is an excellent and very common point of confusion. Let me build a concrete example to show exactly what's happening.

The short answer: **yes, the same mask shape is applied at every layer, but what's being masked is fundamentally different because the representations change after each layer.** The mask prevents future information leakage, but each layer is attending over increasingly abstract representations of the past.

Let me illustrate with a small concrete example. Suppose we have only **5 days** and **3 encoder layers** to keep it traceable.

---

## The Mask Itself Is Identical Across All Layers

The causal mask is always the same lower-triangular matrix:

```
         Day1  Day2  Day3  Day4  Day5
Day 1:  [ ✓     ✗     ✗     ✗     ✗  ]
Day 2:  [ ✓     ✓     ✗     ✗     ✗  ]
Day 3:  [ ✓     ✓     ✓     ✗     ✗  ]
Day 4:  [ ✓     ✓     ✓     ✓     ✗  ]
Day 5:  [ ✓     ✓     ✓     ✓     ✓  ]
```

This mask says: "when computing attention for day d, set the attention weights to negative infinity for all days > d, so after softmax they become zero." This is structurally identical at every layer.

## But What Changes Is WHAT Each Position Represents

Here's the key insight. Let me trace what **Day 4's representation** looks like at each layer:

### Before Layer 1 (Raw Input)

Each day is just its own fused embedding — the code pooling + demographics for that single day. Day 4 knows nothing about any other day.

```
Day 1: [raw embedding of day 1 only]          = E₁
Day 2: [raw embedding of day 2 only]          = E₂  
Day 3: [raw embedding of day 3 only]          = E₃
Day 4: [raw embedding of day 4 only]          = E₄
Day 5: [raw embedding of day 5 only]          = E₅
```

### After Layer 1: First-Order Temporal Mixing

Layer 1 applies causal attention. Day 4 can attend to days 1-4. After attention + SwiGLU + residual, Day 4's representation becomes a blend:

```
Day 4 after L1 = f(E₄, E₃, E₂, E₁)
```

It's a weighted combination: "Day 4's raw codes, informed by what happened on days 1, 2, and 3." The attention weights determine how much each past day contributes — maybe day 3 (yesterday) gets high weight because its codes are clinically related.

But importantly, **each day has only direct, first-order knowledge of other days**:

```
After Layer 1:
  Day 1' = f(E₁)                         ← only sees itself
  Day 2' = f(E₂, E₁)                     ← sees 1-2
  Day 3' = f(E₃, E₂, E₁)                ← sees 1-3
  Day 4' = f(E₄, E₃, E₂, E₁)            ← sees 1-4
  Day 5' = f(E₅, E₄, E₃, E₂, E₁)        ← sees 1-5
```

### After Layer 2: Second-Order Temporal Mixing — This Is Where It Gets Interesting

Now Layer 2 applies the **same causal mask** again. But look at what Day 4 is attending to now:

```
Day 4 attends to:
  Day 1' = f(E₁)
  Day 2' = f(E₂, E₁)                   ← Day 2 already "knows about" Day 1
  Day 3' = f(E₃, E₂, E₁)               ← Day 3 already "knows about" Days 1-2
  Day 4' = f(E₄, E₃, E₂, E₁)           ← itself, already blended
```

When Day 4 attends to Day 3' in Layer 2, it's not seeing raw Day 3 anymore — it's seeing **Day 3's interpretation of Days 1-3**. This creates **indirect multi-hop reasoning paths**:

```
After Layer 2:
  Day 4'' = g(Day4', Day3', Day2', Day1')
          = g(f(E₄,E₃,E₂,E₁), f(E₃,E₂,E₁), f(E₂,E₁), f(E₁))
```

Day 4 now has **second-order knowledge**: "Day 3 thought days 1-2 were important because of X" — it can reason about patterns-of-patterns.

### After Layer 3: Third-Order Abstractions

Layer 3 does the same again, but now each position carries even richer context:

```
After Layer 3:
  Day 4''' = h(Day4'', Day3'', Day2'', Day1'')
```

Day 4 now encodes reasoning like: "In layer 1, day 3 noticed a cardiac code. In layer 2, day 4 connected that cardiac code with a lab value from day 1. In layer 3, this pattern was recognized as a progression toward heart failure."

## Analogy: Passing Notes in a Classroom

Imagine 5 students sitting in a row. The rule is: **you can only pass notes forward** (the causal mask).

**Round 1 (Layer 1):** Each student reads the notes from everyone behind them and writes a summary.
- Student 4 reads notes from students 1, 2, 3, and their own. Writes: "Here's what I know, combined with what 1, 2, 3 told me directly."

**Round 2 (Layer 2):** Same rule — pass notes forward only. But now the notes are richer.
- Student 4 reads Student 3's summary. But Student 3's summary *already contains* Student 3's interpretation of Students 1 and 2. So Student 4 is now getting **digested, interpreted information**, not just raw facts.
- Student 4 might notice: "Student 3 thought Student 1's observation was important AND Student 2 confirmed it — that's a strong signal."

**Round 3 (Layer 3):** The notes are now summaries-of-summaries. Student 4 receives highly abstract, distilled insights about the entire history behind them.

**The causal rule never changes** — you still can't pass notes backward. But the *quality and abstraction level* of what's being communicated increases with each round.

## What Each Layer Learns to Mask (Attend to) Differently

Even though the hard mask is identical, the **learned attention patterns** (the soft weights within the allowed positions) are different at each layer because:

| Layer | What attention tends to learn | Clinical example |
|-------|-------------------------------|------------------|
| **Layer 1** | Direct co-occurrence and immediate temporal proximity | "Yesterday's ER visit is relevant to today's diagnosis" |
| **Layer 2** | Short-range temporal patterns and sequences | "The sequence of codes over the last week suggests an infection that's worsening" |
| **Layer 3** | Multi-step causal chains | "The medication started on day 10, the lab change on day 20, and the adverse event on day 30 are connected" |
| **Layer 4-5** | Higher-order trajectory patterns | "This member's trajectory over 60 days resembles the pattern before hospitalization" |
| **Layer 6** | Global patient-level abstractions | "This member's overall clinical trajectory indicates high acuity risk" |

## Why You Need All 6 Layers Causal (Not Just the First)

If you made Layer 1 causal but Layers 2-6 bidirectional, here's what would break:

```
Layer 1 (causal):
  Day 3' = f(E₃, E₂, E₁)     ← correct, no future info

Layer 2 (bidirectional — BROKEN):
  Day 3'' = g(Day1', Day2', Day3', Day4', Day5')
                                         ↑        ↑
                                    Day4' contains E₄ info!
                                    Day5' contains E₅ info!
```

Day 3 would now "see" future days 4 and 5 through Layer 2's bidirectional attention. Even though Layer 1 was causal, the information leaks immediately in Layer 2. By Layer 6, every position would effectively have full bidirectional context, and the model would learn to cheat instead of predict.

**The causal mask at every layer maintains a strict invariant: Day d's representation at any layer L can only be a function of the original inputs E₁ through E_d.** This invariant must hold recursively through all layers for the training objective (predict next-day codes) to be valid.

---

## What This Means for Model Outputs and Member-Day Evaluation

This usually raises the next question: if one member has 3 valid days, does the model run 3 separate forward passes, one for day 1, then again for day 2, then again for day 3?

The answer is **no**.

The model processes the whole member sequence in **one forward pass** and produces **one output vector per day position**.

For a member with T valid days:

- Input sequence: x = [x_1, x_2, ..., x_T]
- Output logits: [o_1, o_2, ..., o_T]
- Each o_t is a full probability/logit vector over all target codes

If the upstream target table is built for next-day prediction, the intended interpretation is:

- o_1 estimates the next-day target associated with day 1
- o_2 estimates the next-day target associated with day 2
- ...
- o_T estimates the next-day target associated with day T

In other words, the model is **sequence-in, sequence-out**, not autoregressive token-by-token generation.

### What It Does Not Do

It does **not** do this:

1. Run the model on day 1 only to get prediction 1
2. Feed prediction 1 back into the model to generate day 2 prediction
3. Feed prediction 2 back into the model to generate day 3 prediction

Instead, it does this:

1. Read the entire observed input sequence for the member
2. Build one hidden representation per day position
3. Apply temporal attention with a causal mask so day t can only use days 1 through t
4. Emit one output vector for each day position in the same pass

So for a 3-day member, there is **one forward pass** and **3 output vectors**.
For an 80-day member, there is **one forward pass** and **80 output vectors**.

---

## Shape of Input, Output, and Targets

At a high level, for one batch:

- Input tensor shape: [batch, len_dy, features]
- Output tensor shape: [batch, len_dy, target_cd_cnt]
- Target tensor shape: [batch, len_dy, target_cd_cnt]

Each day position has its own output row.

For one member i and one day position t:

- output[i, t, :] = predicted logits or probabilities for all target codes at day t
- target[i, t, :] = actual multi-hot target codes aligned to day t

Only the first dt_cnt positions are valid for that member. Padded days are ignored.

So if dt_cnt = 3:

- day positions 1, 2, 3 are scored
- day positions 4 through len_dy are ignored

If dt_cnt = 80:

- day positions 1 through 80 are scored
- day positions 81 through len_dy are ignored

This is why the analysis unit in intrinsic evaluation is a **valid member-day prediction position**, not just a member.

---

## Concrete 3-Day Example

Suppose one member has 3 valid days.

### Input History Available to the Model

- Day 1 observed input: raw codes and demographics for day 1
- Day 2 observed input: raw codes and demographics for day 2
- Day 3 observed input: raw codes and demographics for day 3

The temporal encoder sees all 3 day positions in one pass, but with the causal rule:

- Day 1 output can only depend on day 1 input
- Day 2 output can depend on day 1 and day 2 inputs
- Day 3 output can depend on day 1, day 2, and day 3 inputs

So conceptually:

- o_1 = f(x_1)
- o_2 = f(x_1, x_2)
- o_3 = f(x_1, x_2, x_3)

Not:

- o_2 = f(x_2, o_1)
- o_3 = f(x_3, o_2)

### Example Targets and Predictions

Suppose the aligned target sequence is:

- Day 1 target: A, B, C, D
- Day 2 target: A, E, D, F
- Day 3 target: A, Z, T, X, S

And the model's top-5 predictions from a **single forward pass** are:

- Day 1 prediction: A, F, Q, R, Y
- Day 2 prediction: B, R, F, L, M
- Day 3 prediction: G, E, S, A, H

Then the day-level overlaps are:

- Day 1 hit set = {A}
- Day 2 hit set = {F}
- Day 3 hit set = {A, S}

So this one member contributes **3 evaluated member-day units**.

### How Metrics Are Updated for This Member

For this member alone:

- Day 1 has at least one hit
- Day 2 has at least one hit
- Day 3 has at least one hit

So Recall@5 for this member's 3 valid days is:

- 3 successful day positions out of 3 valid day positions = 1.00

Across all true codes for this member:

- Day 1: 1 correct out of 4 true codes
- Day 2: 1 correct out of 4 true codes
- Day 3: 2 correct out of 5 true codes

So Micro Recall@5 for this member is:

- total hits / total true codes = 4 / 13

And Precision@5 across the 3 valid day positions is:

- Day 1 precision = 1 / 5
- Day 2 precision = 1 / 5
- Day 3 precision = 2 / 5
- average precision across valid day positions = ((1/5) + (1/5) + (2/5)) / 3

This is exactly why the same member can contribute multiple metric updates. A 3-day member contributes 3 scored positions. An 80-day member contributes 80 scored positions.

---

## Why This Is Still "Next-Day" Prediction

The model architecture is causal across time, which means the representation at day t cannot see future days. That makes it valid for forecasting.

So if the upstream data table defines the target at position t as the next-day codes associated with day t, then the model is learning:

- use history through day t
- predict the target aligned to position t

The intrinsic evaluation notebook then compares:

- output at day t
- against the stored target at day t

for every valid day position.

That is why intrinsic metrics in this setup are **not final-day-only metrics**. They are **all-valid-days metrics**.

---

## Intuition for Stakeholders

The easiest way to explain this is:

- One member record contains a timeline of days
- The model reads the whole observed timeline in one pass
- It produces one probability vector per day position
- Each day position is compared with its aligned actual target codes
- Metrics aggregate over all valid day positions from all members

So the model is not asking only:

- "For this member, how good was the final day prediction?"

It is asking:

- "Across every valid day in this member's history, how often did we recover the right next-day code signals?"

That is why member-day is the correct analysis unit for these intrinsic metrics.