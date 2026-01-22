You are right to hate blind tuning. The goal is to turn “turn knobs and pray” into a disciplined workflow where every change is (a) motivated by a hypothesis about training dynamics, (b) validated with targeted measurements, and (c) reproducible.

Below is a “zero-to-hero” handbook for transformer training and optimization. It is written to (1) teach the mental model (“why”), (2) give you the instrumentation and experiments (“how”), and (3) directly address your plateau pattern for a ~27M dense transformer with millions of training examples and ranking-style metrics.

---

# 1) The three-budget model: why plateaus happen at all

Nearly every plateau in transformer training is one of these ceilings:

### A) **Optimization ceiling**

You are not taking steps that refine the solution anymore (wrong step size schedule, too much gradient noise, wrong regularization, poor conditioning).
**Fix:** change *how* you optimize (LR schedule, batch, optimizer, stabilization).

### B) **Capacity ceiling**

The model cannot represent the needed distinctions, especially for long-tail or subtle ranking improvements.
**Fix:** change model size/shape, inductive bias, or routing.

### C) **Data ceiling (signal/noise ceiling)**

You are trying to learn what the data cannot reliably tell you (label noise, missing labels, ambiguity, weak supervision).
**Fix:** improve data quality, sampling, objective alignment, or labeling.

A healthy training process is:
**Diagnose which ceiling you hit → run one decisive experiment → apply the corresponding remedy.**

---

# 2) The “physics” of training: what the knobs actually do

## 2.1 Gradient descent is a step-size problem

At every step you do:

[
\theta_{t+1} = \theta_t - \alpha \cdot \widehat{g}_t
]

* (\widehat{g}_t) is your mini-batch gradient (noisy estimate of true gradient).
* (\alpha) is the effective step size (not just LR—AdamW modifies it).

A plateau occurs when:

* your steps are too large to refine (you bounce around a basin),
* or too small to escape (you creep),
* or too noisy (gradient signal drowned by variance),
* or the remaining error is not learnable from the data.

## 2.2 AdamW: why “high LR + scheduler” can still stall

AdamW update (simplified):

* Tracks moving averages of gradients (m_t) and squared gradients (v_t).
* Update roughly scales like:

[
\Delta\theta \approx \frac{\text{lr}}{\sqrt{v_t} + \epsilon}\cdot m_t ;-; \text{wd}\cdot \theta
]

Meaning:

* **LR is not the only step-size control.**
* When gradients become small or highly variable late in training, AdamW’s effective step can become “wrong-shaped” for refinement unless the LR decays appropriately.
* Weight decay (wd) is a constant “shrink” force that interacts with LR and can prevent late polishing if mis-set.

**Key lesson:** “Peak LR is high” does not guarantee good late training. What matters is the **effective update magnitude over time**, relative to curvature/noise.

## 2.3 Batch size controls gradient noise (this is huge)

Mini-batch gradients are noisy. Noise is not always bad—early training benefits from it—but late-stage refinement hates it.

* Small batch → high noise → fast exploration early, poor polishing later.
* Large batch → lower noise → slower exploration, better polishing and stability.

This is why many runs look like:

* fast early gains,
* then a long plateau where metrics jitter in a band.

**Batch size and LR are coupled**:
If you scale batch up, you can often scale LR up (to keep progress per step), but you must test stability.

## 2.4 LR schedules: warmup and decay are not optional decor

### Why warmup exists

Early training has unstable activations/gradients. Warmup prevents catastrophic early steps.

### Why decay exists

After the model finds a decent basin, it needs **smaller steps** to:

* improve ranking between hard negatives,
* refine rare-label boundaries,
* improve calibration.

If you keep LR high for too long, you get:

* early “phase change” improvement,
* then **bouncing** around a solution,
* metrics plateau.

**A schedule that delays decay is a plateau machine.**
If your schedule has a long plateau portion or a high LR floor (e.g., ending at 0.2× peak), it often prevents the “fine chiseling” phase.

## 2.5 Loss weighting (pos_weight) changes the gradient landscape

In multi-label problems, `pos_weight` multiplies the gradient contribution from positives. Capping it (`pos_weight_max`) prevents rare labels from dominating and destabilizing training.

Mechanically:

* Higher pos_weight → larger gradients on rare positives → potentially higher recall, worse precision, more instability.
* Lower pos_weight → fewer rare-label updates → potentially higher precision, lower tail recall.

**Important:** pos_weight interacts with:

* LR (big gradients + high LR = overshoot),
* mixed precision (overflow risk),
* batch sampling (how often rare positives appear).

So changing pos_weight while “testing dataset size” confounds the interpretation because you altered the gradient field, not just the data.

---

# 3) The engineer’s instrumentation: what you must log to stop guessing

If you only look at loss and Recall@10, you’re driving with a speedometer and no fuel gauge.

Log these (cheap, high value):

## 3.1 Update-to-weight ratio (U/W)

Compute:
[
\text{U/W} = \frac{|\Delta\theta|}{|\theta|}
]

* If U/W stays high late → you’re not in polishing mode (LR too high or noise too high).
* If U/W becomes tiny early → you’re under-stepping (LR too low or too much regularization).

## 3.2 Gradient norm + clipping rate

* If clipping happens often → LR too high, loss weights too aggressive, or batch outliers.
* If grad norms collapse early → underfitting or overly strong regularization.

## 3.3 Effective tokens per step and tokens seen

For scaling, the real currency is **tokens**, not “examples”:

* total tokens processed,
* tokens per parameter (“tokens/param” is a good sanity guide),
* and whether your runs are compute-matched.

If you double dataset size but keep training length fixed, you improved diversity but not optimization time.
If you double steps (epoch-matched), you changed schedule dynamics and may need different LR decay.

## 3.4 Per-frequency metrics (deciles by label frequency)

Plateau often means: head labels saturated, tail labels stuck.
Compute metrics for:

* top 10% frequent labels,
* middle,
* bottom 10% (tail).

If the tail is flat, tuning LR won’t magically fix it; you need sampling/objective changes.

---

# 4) The scientific method for tuning: how to “fish” instead of guessing

You do not tune 20 parameters. You run **discriminative experiments** that eliminate whole classes of explanations.

## 4.1 The 5 decisive tests (I’d teach any new ML engineer)

1. **Overfit test (sanity)**

   * Train on a tiny subset (e.g., 1k–10k examples) until near-perfect.
   * If you can’t, you have a bug, objective mismatch, or capacity issue.

2. **LR range test**

   * Sweep LR over a log range in a short run.
   * Find LR where loss starts to diverge; choose a peak LR below that.
   * This prevents blind “high end” LR choices.

3. **Polishing test**

   * Resume from plateau checkpoint with LR 10× lower for a short run.
   * If metrics improve: your plateau is optimization/schedule.
   * If nothing changes: it’s capacity or data ceiling.

4. **Batch-noise test**

   * Increase effective batch (gradient accumulation if needed) while adjusting LR carefully.
   * If plateau reduces/jitter decreases and metrics improve: your plateau was noise-limited.

5. **Capacity test**

   * Train a slightly bigger model for fewer steps.
   * If it clears the plateau: you were capacity-limited.

This is how you stop “tuning” and start *diagnosing*.

---

# 5) Why your performance stops increasing: the most likely mechanisms in your situation

Given:

* dense ~27M transformer,
* millions of examples,
* ranking metrics (Recall@10/NDCG/MRR) with multi-label BCE-style training,
* fast early rise then long plateau,

the dominant plateau causes are usually these:

## 5.1 You entered the “hard-negative and tail-label” regime

Early training learns:

* frequent labels,
* obvious co-occurrence rules,
* short-range correlations.

Late training requires:

* ranking among confusing candidates (hard negatives),
* tail labels that appear rarely,
* denoising label omissions.

Those improvements are **small**, require many targeted updates, and are easy to mask by variance.

## 5.2 Schedule not entering true polishing mode

If LR decays late and/or ends at a relatively high floor, you can stall at “good enough basin, too big steps to refine.”

This produces exactly what you saw: a stable band of batch metrics rather than a clear upward trend.

## 5.3 Objective–metric mismatch

BCE improves marginal probabilities; Recall@10/NDCG depend on **relative ordering** among close candidates.
Once the model has the coarse separation, BCE can keep optimizing without improving top-K ordering much.

That’s why you can feel “it’s learning, but it’s not getting better where I care.”

## 5.4 Tail is under-updated

Even with pos_weight, tail labels might not be updated enough because they appear too rarely in batches. Weighting amplifies gradient when they appear, but it does not increase how often they appear.

Sampling strategy can matter more than LR in this regime.

---

# 6) A practical handbook: how to set training configs intelligently (and how knobs interact)

Below is the “knob map” you should memorize. This is the fishing kit.

## 6.1 Learning rate (peak LR)

**Controls:** how far you move per step early/mid training.
**Too high:** divergence, heavy clipping, unstable metrics.
**Too low:** slow learning, early underfitting.

**Couples with:**

* batch size (bigger batch usually supports bigger LR),
* pos_weight (bigger gradients need lower LR),
* warmup length (higher LR needs longer warmup).

## 6.2 Warmup

**Controls:** stability during “model boot-up.”
**Too short:** early spikes, occasional collapse.
**Too long:** wastes steps at tiny LR.

**Couples with:**

* batch size, mixed precision, model depth, and any aggressive loss weighting.

## 6.3 Decay shape + final LR floor

**Controls:** whether you ever enter the refinement regime.

* If you plateau: you often need **earlier decay** and/or **lower floor**.

**Couples with:**

* evaluation metric: ranking metrics tend to need real polishing.
* EMA: if EMA is used for eval, too-high LR late can make EMA lag.

## 6.4 Batch size / gradient accumulation

**Controls:** gradient noise scale.
Plateaus with jitter often respond to larger effective batch late in training.

**Couples with:**

* LR (scaling rules),
* bucketing/sequence length (token-level batch is what matters).

## 6.5 Weight decay

**Controls:** complexity penalty / parameter shrink.
Too high can cap late performance (underfitting). Too low can overfit and harm generalization.

**Couples with:**

* LR (decoupled but still interacts in practice),
* dataset size (more data usually tolerates less regularization).

## 6.6 Dropout

**Controls:** regularization and representation robustness.
More data → you often can reduce dropout slightly.

**Couples with:**

* model size: small models can suffer more from high dropout.

## 6.7 Class weighting (pos_weight)

**Controls:** recall–precision trade-off and tail learning.
If you don’t instrument per-frequency metrics, you will misinterpret pos_weight changes.

**Couples with:**

* LR, clipping, mixed precision stability,
* sampling strategy (frequency of rare positives).

## 6.8 Loss/Objective choice

**Controls:** what the model is rewarded for.
If metric is ranking@K, consider:

* sampled softmax / in-batch negatives,
* pairwise ranking loss,
* hard-negative mining.

This is often the real “plateau breaker” when BCE has extracted most easy gains.

---

# 7) The plateau-resolution playbook for your next iteration (minimal experiments, maximal information)

This is the shortest path to an answer, not a giant grid search.

## Step 1: Confirm which ceiling you hit (one afternoon of experiments)

1. **Resume-from-plateau polishing run**

   * Take your best checkpoint.
   * Train 5–10% more steps with **LR 10× lower** than current late-training LR.
   * Keep everything else identical.

**Interpretation:**

* Metrics improve → schedule/optimization ceiling.
* No improvement → capacity/data/objective ceiling.

2. **Tail slice report**

* Report recall@10 / ndcg for labels grouped by frequency decile.

**Interpretation:**

* Tail stuck → need sampling/objective, not just LR.

## Step 2: If optimization ceiling, fix schedule scientifically

A principled schedule goal:

* warmup for stability,
* then **monotonic decay** into true polishing,
* end at a **very low LR** (so it can refine ordering and calibration).

What you do:

* remove long LR “plateau” phases,
* decay earlier,
* lower the final LR floor substantially (the floor is what prevents polishing).

Then re-run and verify:

* U/W decreases late,
* clipping decreases,
* metrics stop jittering and start drifting upward.

## Step 3: If tail/objective ceiling, change “what gradients you see”

Pick one lever (not five):

* **Sampling:** oversample rare-positive examples or construct batches that contain tail labels more often.
* **Objective alignment:** add a ranking-aware component or hard negatives.

Then verify improvement by:

* tail decile metrics,
* and top-K ordering metrics (NDCG/MRR).

## Step 4: If capacity ceiling, stop fighting physics

At 27M, it’s plausible you’re out of capacity for subtle ranking improvements.
Your confirmation is: a larger/deeper model lifts the plateau under similar token budget.

If it does, no amount of LR heroics will substitute for capacity.

---

# 8) What “mastery” looks like: your personal operating procedure

When you face a plateau in the future, you should be able to say, confidently:

1. “Is this optimization-limited?”
   I checked U/W, clipping, and ran a low-LR polishing continuation.

2. “Is this tail/data-limited?”
   I checked per-frequency deciles and data noise indicators.

3. “Is this objective-limited?”
   I verified whether loss improvements correlate with ranking@K.

4. “Is this capacity-limited?”
   I ran the smallest viable scale-up test.

If you can do those four, you are not tuning anymore. You are diagnosing.

---

# 9) A short, blunt checklist you can keep next to your terminal

### If you plateau early:

* [ ] Can I overfit a tiny subset? If no → bug/objective.
* [ ] Is clipping frequent? If yes → LR/pos_weight/batch outliers.
* [ ] Does low-LR continuation improve? If yes → schedule.
* [ ] Are tail metrics flat? If yes → sampling/objective.
* [ ] Does bigger model help? If yes → capacity.

---

If you want to make this extremely concrete, I can take your **exact scheduler implementation** (the code path, not just config fields) and help you rewrite it into a schedule that (a) has the same early speed, (b) enters polishing earlier, and (c) produces measurable late-stage gains—while also defining the specific metrics (U/W, clipping rate, tail deciles) you’ll use to prove the hypothesis rather than hope it works.
