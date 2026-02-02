# Expert 1: Comprehensive Synthesis: Learning Plateau Root Cause Analysis
- Feb 1, 2026
## Executive Summary

After reviewing all evidence systematically, I must be direct: **Tier-Aware Batching at the member level, as currently conceived, will likely NOT directly solve the learning plateau problem.** Here's why, and what the evidence actually tells us about the root cause.

---

## Part 1: What Has Been Definitively Proven

### 1.1 Empirically Confirmed Facts

| Finding | Evidence | Confidence |
|---------|----------|------------|
| **Gradient concentration is emergent, not static** | 17-27% balanced → 85% common by end (both pos_weight=35 and 200) | ✅ HIGH |
| **Concentration is pos_weight-independent** | 5.7× increase in pos_weight changed concentration by <0.5% | ✅ HIGH |
| **Transition occurs at steps 500-3000** | Identical timeline in both experiments | ✅ HIGH |
| **Rare/tail accuracy remains 0%** | Unchanged across all experiments | ✅ HIGH |
| **Embeddings are NOT collapsed** | All tiers have healthy norms (~1.1-1.5) | ✅ HIGH |
| **Logit suppression is SEVERE for tail** | Tail logit = -14.69 when y=1 (probability ~0.00004%) | ✅ HIGH |
| **Tail got WORSE with more data** | 3.4M model: -12.9 → -14.69 logit for tail | ✅ HIGH |

### 1.2 The Critical New Evidence (Jan 30)

The code frequency analysis reveals something crucial that **changes the understanding of the problem**:

```
📈 MEMBER-LEVEL ANALYSIS
   Tier            Members   % of Total
   tail          1,317,600        83.4%   ← 83.4% of members HAVE tail codes!

📅 DAY-LEVEL ANALYSIS  
   Tier               Days   % of Total
   tail         15,364,719        22.3%   ← But tail codes appear on only 22% of days

🔢 OCCURRENCE-LEVEL ANALYSIS
   Tier        Occurrences   % of Total   Ratio to Common
   tail         19,071,332         5.2%            13.4x fewer
```

**This is the smoking gun**: The gradient starvation is NOT because members with tail codes are rare (83.4% coverage). It's because:

1. **Tail codes appear on fewer days per member** (22.3% vs 92.2% for common)
2. **When they appear, they're drowned out** by 13.4× more common code occurrences

---

## Part 2: Why Tier-Aware Batching (Member-Level) May Not Directly Solve the Problem

### 2.1 The Fundamental Mismatch

The current Tier-Aware Batching implementation operates at the **member level**:
- Guarantees N members with tail codes per batch
- But 83.4% of members already have at least one tail code

**The problem is that:**
- Member-level sampling doesn't change the **occurrence-level imbalance** (5.2% vs 69.7%)
- A member selected for having a tail code might have 1 tail occurrence and 200 common occurrences
- The gradient contribution is still dominated by common codes within that member's data

### 2.2 The Math Doesn't Work Out

```
With tier-aware batching at member level:
├── Batch has 128 members
├── Guarantee 10 members have tail codes
├── But each of those 10 members has:
│   ├── ~45 days with tail codes (22.3% of ~200 days)
│   └── ~1800 common code occurrences
└── Net effect: Still ~13× more common gradients per batch
```

The member-level quota ensures tail codes **appear**, but doesn't change the **gradient magnitude ratio**.

### 2.3 What the Evidence Says About Member-Level Coverage

From the Jan 30 analysis:

> "The real problem is occurrence-level (5.2% vs 69.7%). Pos-weight is the primary mechanism for occurrence-level rebalancing. Tier-aware batching just ensures consistency."

And:

> "Given your data shows 83.4% member coverage for tail codes, aggressive quotas are less critical than I initially suggested."

---

## Part 3: The Real Root Cause (Synthesized View)

### 3.1 The Gradient Starvation Mechanism

The evidence points to a **multi-level gradient starvation mechanism**:

```
Level 1: Per-Sample (pos_weight addresses this)
├── Tail positives get higher weight per occurrence ✓
└── But this alone is insufficient

Level 2: Per-Batch (member-level batching partially addresses this)
├── Guarantee tail members appear ✓
├── But intra-member occurrence imbalance remains
└── Tail codes still drowned out within each member

Level 3: Per-Step Aggregation (NOT ADDRESSED)
├── Gradient direction from tail is high-variance
├── Gets "averaged out" by consistent common code signal
└── Model converges to "common code prior"

Level 4: Temporal Dynamics (NOT ADDRESSED)
├── Model learns common codes first (steps 0-500)
├── Common code gradients become "refined" (small magnitude, consistent direction)
├── But their cumulative effect dominates (many samples × small gradient)
└── By step 3000, tail signal is effectively noise
```

### 3.2 Why Per-Sample Weighting Can't Fix Per-Step Aggregation

The key insight from all the expert analysis:

```
Total gradient per code ∝ (samples containing code) × (pos_weight) × (per-sample error)

For tail:  100 samples × 200 weight × 0.9 error = 18,000
For common: 100,000 samples × 1 weight × 0.1 error = 10,000

Even with 200× pos_weight, common STILL contributes more due to sample count.
```

But the real killer is **batch-level averaging**:
- Common codes appear in EVERY batch → consistent gradient direction → accumulates
- Tail codes appear sporadically → high-variance direction → averages to noise

---

## Part 4: What Actually Needs to Change

Based on the evidence, the interventions must address the **gradient aggregation mechanism**, not just the weighting or batching.

### 4.1 Interventions Ranked by Evidence Support

| Intervention | Addresses Root Cause? | Evidence Support | Risk |
|--------------|----------------------|------------------|------|
| **Per-Tier Gradient Normalization** | ✅ Yes - directly fixes aggregation | Untested but mechanistically sound | Medium |
| **Sampled Softmax / Ranking Loss** | ✅ Yes - changes objective alignment | Strong theoretical, industry standard | Medium |
| **Day-Level Sampling** (not member-level) | ⚠️ Partially - addresses occurrence imbalance | Suggested but not implemented | Low |
| **Member-Level Tier-Aware Batching** | ❌ Insufficient - member coverage already 83%+ | Evidence shows it won't solve core issue | Low |
| **Increasing pos_weight** | ❌ No - proven ineffective | 5.7× increase had <0.5% effect | Low |

### 4.2 The Missing Intervention: Occurrence-Level Sampling

Instead of sampling **members** with tail codes, you should consider sampling **member-days** with tail codes:

```
Current approach (member-level):
"Give me members who have tail codes" 
→ Gets members with 1 tail out of 2000 codes

Better approach (occurrence-level):
"Give me member-days WHERE tail codes appear"
→ Gets days where tail codes actually contribute to gradient
```

This directly addresses the 22.3% day-level coverage.

---

## Part 5: Re-evaluating Training Configuration Issues

You asked me to think independently about training configuration beyond data issues. Here's what the evidence suggests:

### 5.1 Learning Rate Schedule Analysis

The plateau occurs at steps 6000-8000 across experiments. This is **after** the gradient concentration transition (steps 500-3000) completes.

**Hypothesis:** The LR schedule may not be entering true "polishing mode" early enough.

From the general plateau analysis:
> "A schedule that delays decay is a plateau machine. If your schedule has a long plateau portion or a high LR floor, it prevents the 'fine chiseling' phase."

**Key question:** What does your LR schedule look like during steps 500-3000? If LR is still high during the concentration transition, you're optimizing for the common-code basin at the expense of tail codes.

### 5.2 The Optimization Ceiling Test

The expert panel suggested a **decisive experiment**:

```
"Polishing test":
- Resume from plateau checkpoint with LR 10× lower for a short run
- If metrics improve: your plateau is optimization/schedule
- If nothing changes: it's capacity or data ceiling
```

**This test has not been run.** Before implementing tier-aware batching or gradient normalization, this 30-minute diagnostic could determine if the plateau is fundamentally an **optimization ceiling** vs a **data/objective ceiling**.

### 5.3 Batch Size and Gradient Noise

The evidence shows high jitter in tail gradient contribution (epoch average vs final value differ). This suggests:

> "The tail gradient signal is highly stochastic, which is why the model cannot reliably learn from it."

**Larger effective batch size** (via gradient accumulation) could reduce this variance, making tail gradient signal more consistent.

---

## Part 6: Recommended Action Plan (Evidence-Based Priority)

### Phase 0: Critical Diagnostics (Before ANY Intervention)

1. **LR Polishing Test** [30 min, 1 short run]
   - Resume from plateau checkpoint with LR 10× lower
   - If metrics improve → LR schedule is the problem
   - If no change → proceed to structural interventions

2. **Verify Batch Composition** [10 min, code check]
   - Confirm what % of each batch's OCCURRENCES (not members) are from each tier
   - This will quantify the actual gradient imbalance per batch

### Phase 1: If LR Schedule Is NOT the Issue

Based on evidence, priority order:

1. **Per-Tier Gradient Normalization** [Highest impact, directly addresses root cause]
   - After `backward()`, rescale gradients by tier to enforce target contributions
   - Expected: tail_frac stays at ~25% throughout training
   - Risk: May degrade common performance initially

2. **Day-Level Sampling** (Alternative to member-level batching)
   - Instead of sampling members with tail codes, sample member-days with tail codes
   - Directly addresses the 22.3% day-level coverage
   - Lower complexity than gradient normalization

3. **Sampled Softmax with Tier-Stratified Negatives**
   - Changes objective from "is this code present?" to "which codes are present?"
   - Directly addresses the ranking vs discrimination misalignment
   - Higher implementation complexity

### Phase 2: If Above Interventions Fail

- Consider hierarchical supervision (CCS category-level auxiliary loss)
- Consider contrastive pre-training for code embeddings
- Consider two-stage training (Kang et al., 2020)

---

## Part 7: Direct Answer to Your Questions

### Q1: Can Tier-Aware Batching adjust occurrence of codes instead of members?

**Yes, but it requires a different implementation.** The current `TierAwareBatchSampler` operates at member level. To address occurrence-level imbalance, you would need:

```python
# Conceptual approach: Day-Level Tier-Aware Sampling
class DayLevelTierAwareSampler:
    """
    Instead of sampling members, sample (member, day) pairs
    where the day contains tier-specific codes.
    """
    def __init__(self, dataset, code_frequencies, batch_size, tail_day_quota):
        # Pre-compute which (member, day) pairs have tail codes
        self.days_with_tail = []  # List of (member_idx, day_idx) tuples
        for member_idx in range(len(dataset)):
            for day_idx, day_codes in enumerate(dataset.targets[member_idx]):
                if any(code in tail_code_set for code in day_codes):
                    self.days_with_tail.append((member_idx, day_idx))
```

This would guarantee that **X% of the training signal per batch comes from days with tail codes**, not just members.

### Q2: Is tier imbalance the root cause of the plateau?

**Partially, but not in the way initially hypothesized.**

The tier imbalance IS a problem, but:
- It's NOT a member-level problem (83.4% coverage)
- It IS an occurrence-level problem (5.2% vs 69.7%)
- It's COMPOUNDED by gradient aggregation dynamics

The plateau is caused by the **interaction** of:
1. Occurrence-level imbalance (13.4× ratio)
2. Batch-level gradient averaging (tail spikes drowned out)
3. Temporal dynamics (concentration transition at steps 500-3000)
4. Potentially suboptimal LR schedule (not yet tested)

### Q3: What exactly can directly resolve the learning plateau?

Based on the evidence, the most direct interventions are:

1. **Per-Tier Gradient Normalization** - Forces equal contribution regardless of occurrence count
2. **Sampled Softmax** - Changes objective to ranking-aware, which directly optimizes for top-K metrics
3. **LR Schedule Adjustment** - If polishing test shows improvement, decay earlier and lower

The **least effective** intervention (given the evidence) is:
- Member-level tier-aware batching alone (won't overcome occurrence imbalance)
- Further increasing pos_weight (proven ineffective at 5.7× increase)

---

## Part 8: Final Synthesis - The Ground Truth Hypothesis

**Primary Hypothesis (Highest Confidence):**

The learning plateau is caused by a **self-reinforcing gradient capture mechanism** where:

1. **Early training (steps 0-500):** Gradients are balanced because the model is randomly initialized
2. **Transition phase (steps 500-3000):** Common codes appear in every batch with consistent gradient direction → model learns "common code prior" → common codes capture majority of gradient budget
3. **Terminal phase (steps 3000+):** Tail codes receive <1% of gradient → their signal is noise relative to common code refinement → model actively learns to suppress tail predictions (logit: -12 → -14 with more data)

**The pos_weight mechanism operates per-sample but the gradient concentration happens per-step**, which is why 200× pos_weight produces only 0.5% change in gradient tier distribution.

**The member-level coverage is already 83.4%**, so tier-aware batching at member level adds marginal value. The real problem is **occurrence-level** (5.2% vs 69.7%) and **gradient aggregation dynamics**.

---

## Recommended Immediate Actions

1. **Run the LR Polishing Test** (30 min) - Resume from plateau with 10× lower LR
2. **Compute occurrence-level batch composition** - Verify what % of gradient per batch comes from each tier
3. **If polishing test fails:** Implement per-tier gradient normalization as the most direct intervention

Would you like me to elaborate on the implementation of per-tier gradient normalization or day-level sampling, or analyze any specific aspect of the evidence in more detail?


# Expert 2: Independent Analysis: Root Cause of Learning Plateau and Actionable Next Steps

## Executive Summary

After comprehensive review of all evidence, I conclude that **tier-aware batching at the member level is insufficient and possibly misdirected** because the diagnosed problem is at the **occurrence level**, not member level. Furthermore, the evidence suggests the learning plateau may have **multiple interacting causes** beyond just rare code imbalance. Let me provide a complete evidence-based synthesis.

---

## (A) What the Evidence Definitively Shows

### 1. Gradient Concentration is Emergent and Intrinsic to Training Dynamics

| Phase | Common Frac | Tail Frac | Total Norm | Mechanism |
|-------|-------------|-----------|------------|-----------|
| Step 1 | 17.8% | 17.8% | 530,569 | Balanced at init |
| Step 500 | 16.9% | 18.4% | 24,989 | Still balanced |
| Step 1500 | 42.7% | 10.4% | 3,398 | **Transition begins** |
| Step 3000 | 66.7% | 3.0% | 1,632 | Common dominating |
| Step 12001 | 85.3% | 0.1% | 22,129 | **Terminal concentration** |

**Key Finding**: The concentration pattern is **identical at pos_weight=35 and pos_weight=200**. This proves the root cause is NOT weight magnitude—it's the **batch-level gradient accumulation dynamics**.

### 2. The Member-Level vs. Occurrence-Level Analysis Reveals the True Bottleneck

| Level | Tail Coverage | Common Coverage | Ratio |
|-------|---------------|-----------------|-------|
| **Member** | 83.4% | 100.0% | 1.2× |
| **Day** | 22.3% | 92.2% | 4.1× |
| **Occurrence** | 5.2% | 69.7% | **13.4×** |

**Critical Insight**: 83.4% of members have tail codes—but those codes appear on only 22.3% of days and represent only 5.2% of total occurrences. **Tier-aware batching at the member level cannot fix an occurrence-level problem.**

### 3. Logit Suppression is Severe but Embeddings are Healthy

| Tier | Logit (y=1) | Probability | Embedding Norm |
|------|-------------|-------------|----------------|
| Common | -2.26 | ~9.4% | 1.42 ± 0.27 |
| Medium | -6.39 | ~0.17% | 1.49 ± 0.15 |
| Rare | -9.68 | ~0.006% | 1.41 ± 0.05 |
| **Tail** | **-14.69** | **~0.00004%** | 1.46 ± 0.03 |

**Key Finding**: No embedding collapse detected. The decoder weights are healthy. The problem is that **the model has learned strong negative priors for rare codes** because it rarely sees positive examples during training.

### 4. pos_weight Increases Cause Collateral Damage

| Metric | pos_weight=50 | pos_weight=200 | Delta |
|--------|---------------|----------------|-------|
| medium_top10_acc | 4.1% | 0.16% | **-96.2%** |
| rare_top10_acc | 0% | 0% | No change |
| tail_top10_acc | 0% | 0% | No change |
| macro_auroc | 0.846 | 0.878 | **+3.8%** |
| recall@5 | 0.722 | 0.686 | **-4.9%** |

**Key Finding**: Higher pos_weight improved discrimination (AUROC) but **degraded ranking** (recall@5, NDCG) and **catastrophically harmed medium codes**. This is the signature of **objective-metric misalignment**.

---

## (B) Primary Hypothesis: Multi-Factor Root Cause

Based on the evidence, the plateau is caused by **three interacting factors**, not just one:

### Factor 1: Batch-Level Gradient Aggregation (CONFIRMED)

**Mechanism**: Even with perfect member-level coverage, each batch contains:
- Common codes: ~50-100 positive samples per batch → consistent gradient every step
- Tail codes: ~0-1 positive samples per batch → sporadic spikes averaged out

The gradient formula shows why pos_weight cannot overcome this:
```
Total_Gradient[code] ∝ Σ_batches (samples_in_batch × pos_weight × error)
```

For a tail code with 100 total samples vs. a common code with 100,000 samples:
- Tail: 100 × 200 = 20,000 (weighted)
- Common: 100,000 × 1 = 100,000 (unweighted)

**Common still contributes 5× more gradient** even with 200× weight differential.

### Factor 2: BCE Objective-Metric Misalignment (CONFIRMED)

**Evidence**: macro_auroc improved (+3.8%) while recall@5 worsened (-4.9%).

**Mechanism**: BCE optimizes per-code calibration: "Is this code likely to be positive?"
Your business metric is ranking: "Which K codes should be in the top-K?"

Once common codes are well-calibrated, BCE provides **no gradient incentive to improve ranking among competitive candidates**. The loss can decrease while ranking metrics plateau.

### Factor 3: Learning Rate Schedule May Be Preventing Polishing (HYPOTHESIS)

**Evidence to Check**:
Looking at your configuration:
```
scheduler_type='linear_plateau_cosine'
warmup_pct=0.15
plateau_pct=0.45
min_lr_ratio=0.2
```

This means:
- 15% warmup → 45% plateau (at peak LR) → 40% cosine decay → ends at 20% of peak LR

**Potential Issue**: The LR schedule has a **long plateau phase at high LR** (45% of training) and **ends at a relatively high floor** (20% of peak). This configuration is known to prevent fine-grained "polishing" of difficult examples.

From the general plateau analysis document:
> "A schedule that delays decay is a plateau machine. If your schedule has a long LR plateau or a high LR floor, it often prevents the fine chiseling phase."

**What I cannot confirm** (needs testing):
- Whether resuming from plateau checkpoint with 10× lower LR would improve metrics
- Whether the current schedule is actually the limiting factor

---

## (C) Why Tier-Aware Batching at Member Level Won't Solve the Problem

### The Logic Gap

**Current proposal**: Ensure each batch contains N members who have tail codes.

**The problem**: Having a member with tail codes doesn't mean those tail codes appear in the target labels for that batch's training days.

**Illustration**:
```
Member A (has tail codes in profile):
├── Training Day 1: [common₁, common₂, common₃] → no tail codes in TARGET
├── Training Day 2: [common₄, medium₁] → no tail codes in TARGET  
├── Training Day 3: [common₅, tail₁] → finally 1 tail code in TARGET
└── ... 200 days, only ~45 have tail codes as targets (22.3%)
```

Even if you guarantee Member A is in the batch, you only have a 22.3% chance that the specific training day contains tail codes in the target.

### What Would Actually Address Occurrence-Level Imbalance

**Option 1: Day-Level Sampling** (Most Direct)
Instead of sampling members, sample **(member, day)** pairs where that day contains tail code positives:

```python
# Construct batches where at least N samples have tail codes as TARGET
tail_positive_days = [(member_id, day_idx) for member_id, day_idx, targets 
                       in all_training_samples if any(t in tail_codes for t in targets)]
                       
# Guarantee K of these appear in each batch
batch = sample_from(general_pool, n=batch_size - K) + sample_from(tail_positive_days, n=K)
```

**Option 2: Loss-Level Rebalancing Per Tier** (Alternative)
Instead of changing batching, aggregate loss separately by tier and normalize:

```python
# In loss computation:
loss_per_tier = {tier: compute_loss(logits[:, tier_mask], targets[:, tier_mask]) 
                 for tier, tier_mask in tier_masks.items()}
total_loss = sum([weight[tier] * loss_per_tier[tier] for tier in tiers])
```

This ensures each tier contributes meaningfully to the total loss regardless of occurrence frequency.

---

## (D) Independent Assessment: What Actually Causes the Plateau?

### Ranking the Contributing Factors by Evidence Strength

| Factor | Evidence Strength | Contribution to Plateau | Intervention |
|--------|-------------------|-------------------------|--------------|
| **Gradient aggregation/starvation** | ✅ STRONG (pos_weight-independent concentration) | **Primary** | Day-level sampling or gradient normalization |
| **BCE-ranking misalignment** | ✅ STRONG (AUROC↑ while recall↓) | **Secondary** | Sampled softmax / ranking loss |
| **LR schedule preventing polishing** | ⚠️ UNVERIFIED | **Possible** | Low-LR continuation test |
| **Occurrence-level imbalance (13.4×)** | ✅ CONFIRMED | **Structural cause** | Day-level sampling |
| **Model capacity** | ❌ REJECTED (MoE=Dense) | **Not a factor** | — |
| **Data quantity** | ❌ REJECTED (2× data → +3%) | **Not primary** | — |

### My Independent Hypothesis

The plateau is primarily caused by **the combination of**:
1. **Structural**: Tail codes appear in only 5.2% of target occurrences, making their gradient signal sporadic
2. **Dynamic**: BCE loss aggregation allows common codes to dominate learning once the "easy" structure is captured
3. **Possibly**: The LR schedule doesn't provide a proper "polishing" phase at low LR

The proposed interventions (tier-aware batching, hierarchical loss, etc.) address symptoms but may not directly solve the structural issue.

---

## (E) Decisive Experiments to Distinguish Root Causes

### Experiment 1: LR Schedule / Polishing Test (2 hours, HIGHEST INFORMATION GAIN)

**Rationale**: Before complex interventions, test if this is simply a schedule issue.

```python
# Resume from plateau checkpoint with 10× lower LR
# Run for 10-20% more steps
scheduler_type='constant'  # or very slow decay
learning_rate = original_lr * 0.1
```

**Interpretation**:
- **If metrics improve** → Schedule was the bottleneck; implement earlier decay
- **If no improvement** → Proceed to structural interventions

### Experiment 2: Day-Level Tier-Aware Sampling (1 training run)

**Rationale**: This directly addresses the occurrence-level imbalance instead of member-level.

```python
# Construct training samples with guaranteed tail-positive days
class TierAwareDayLevelSampler:
    def __init__(self, dataset, tail_quota_per_batch=16):
        # Pre-compute which (member, day) pairs have tail codes in targets
        self.tail_positive_samples = self._find_tail_positive_days(dataset)
        self.general_samples = list(range(len(dataset)))
        
    def sample_batch(self, batch_size):
        # Guarantee K samples have tail codes as targets
        tail_samples = random.sample(self.tail_positive_samples, min(self.tail_quota, len(self.tail_positive_samples)))
        general_samples = random.sample(self.general_samples, batch_size - len(tail_samples))
        return tail_samples + general_samples
```

**Expected Outcome**:
- `train_grad_tier_tail_frac` should stay >5% throughout training (vs. dropping to 0.1%)
- If tail_top10_acc moves off 0%, the intervention is working

### Experiment 3: Per-Tier Loss Balancing (1 training run)

**Rationale**: Ensure each tier contributes equally to total loss regardless of occurrence frequency.

```python
# In loss computation:
loss_common = criterion(logits[:, common_mask], targets[:, common_mask])
loss_medium = criterion(logits[:, medium_mask], targets[:, medium_mask])
loss_rare = criterion(logits[:, rare_mask], targets[:, rare_mask])
loss_tail = criterion(logits[:, tail_mask], targets[:, tail_mask])

# Weight equally (or proportionally to tier importance)
total_loss = 0.25 * loss_common + 0.25 * loss_medium + 0.25 * loss_rare + 0.25 * loss_tail
```

**Expected Outcome**:
- Forces gradients to flow to all tiers equally
- May degrade common code performance (acceptable tradeoff for diagnosis)

### Experiment 4: Sampled Softmax Objective (1 training run)

**Rationale**: Directly optimizes ranking instead of calibration.

```python
# For each sample:
positives = targets[sample]  # 2-10 positive codes
negatives = sample_negatives(100, stratified_by_tier=True)  # 25 per tier
loss = cross_entropy(logits[positives + negatives], labels)
```

**Expected Outcome**:
- Should improve NDCG/MRR/recall@5 (ranking metrics)
- May worsen calibration (acceptable if ranking improves)

---

## (F) Recommended Action Plan

### Priority 1: Quick Diagnostic (Before Any Training Changes)

**1.1 LR Polishing Test** (2 hours)
- Resume from best checkpoint with LR = current_final_LR × 0.1
- Train for 10% more steps
- If metrics improve → implement schedule change first

**1.2 Analyze Update-to-Weight Ratio**
Check if training is actually still "moving" late:
```python
# Log this during training:
update_to_weight_ratio = (param.data - old_param.data).norm() / param.data.norm()
```
If U/W ratio is tiny → LR is already too low, schedule is fine
If U/W ratio is still significant → model is moving but in wrong direction

### Priority 2: Structural Intervention (Day-Level Sampling)

If Priority 1 shows schedule is not the issue:

**Implement day-level tier-aware sampling** that guarantees tail code OCCURRENCES in each batch, not just members who have tail codes somewhere in their history.

**Success criteria**:
- `train_grad_tier_tail_frac` > 5% at end of training
- `tail_top10_acc` > 1% (evidence of learning)
- `common_top10_acc` degradation < 3%

### Priority 3: Objective Alignment

If Priority 2 shows tail codes can learn but ranking metrics don't improve:

**Switch to sampled softmax** with tier-stratified negatives.

### Priority 4: Consider Hierarchical Supervision

If above interventions show improvement but don't fully solve the problem:

**Add CCS/CCSR category-level auxiliary loss** to provide signal for rare codes via their parent categories.

---

## (G) Plain-Language Summary

**What I observe**: Your gradient tier analysis definitively proves that 85% of learning signal goes to common codes by end of training, leaving tail codes effectively ignored. However, member-level tier-aware batching won't fix this because the real problem is at the occurrence level—tail codes are 13.4× rarer per occurrence than common codes.

**Root cause (my hypothesis)**: The plateau is caused by a combination of:
1. **Occurrence-level imbalance** (structural, 13.4× ratio)
2. **BCE loss aggregation dynamics** (common codes dominate gradient accumulation)
3. **Possibly** the LR schedule not providing a proper polishing phase

**What will NOT work**:
- Member-level tier-aware batching alone (83.4% member coverage is already high)
- Increasing pos_weight further (caused 96% collapse in medium codes)
- Adding more model capacity (MoE didn't help)
- Adding more data (2× data → only +3%)

**What might work**:
1. **First**: Test if LR polishing helps (cheap, fast)
2. **Then**: Day-level tier-aware sampling (guarantees tail code TARGETS in batches)
3. **Then**: Per-tier loss balancing or sampled softmax (changes what gradients look like)
4. **Finally**: Hierarchical supervision (adds auxiliary signal)

---

## (H) Answering Your Specific Questions

### 1. "Can we adjust Tier-Aware Batching to work at occurrence level instead of member level?"

**Yes, and this is essential.** The current member-level proposal is misdirected because 83.4% of members already have tail codes—the problem is those codes only appear as targets on 22.3% of days.

**Implementation approach**:
```python
# Instead of sampling MEMBERS with tail codes:
tail_members = [m for m in members if m.has_any_tail_code()]  # 83.4% coverage

# Sample TRAINING SAMPLES where tail codes are in the TARGET:
tail_positive_samples = [(member_id, day_idx) for member_id, day_idx, target_codes 
                          in training_samples if any(code in tail_codes for code in target_codes)]
```

### 2. "Is the rare/tail code imbalance really the root cause?"

**Partially yes, but it's insufficient to explain the full picture.**

The imbalance IS a structural cause (13.4× occurrence ratio), but:
- BCE loss aggregation dynamics amplify the problem
- The objective-metric misalignment means even "fixing" the imbalance may not improve ranking metrics
- The LR schedule may be contributing to the plateau

**My assessment**: Rare/tail imbalance is a **necessary but not sufficient** explanation. Addressing it alone won't solve the problem without also addressing the BCE-ranking misalignment.

### 3. "What about training configuration issues (LR, scheduler, batch size)?"

**This has NOT been adequately investigated and may be a significant factor.**

**Evidence suggesting schedule issues**:
- Your schedule has 45% plateau at peak LR + ends at 20% of peak (relatively high floor)
- Industry best practice for ranking tasks: earlier decay, lower floor
- The plateau document notes: "A schedule that delays decay is a plateau machine"

**Recommended test**: Resume from plateau with 10× lower LR. If metrics improve, implement schedule change before complex interventions.

**Batch size**: Your batch size of 128 is reasonable. However, larger effective batch (via gradient accumulation) may help late-stage stability if gradient noise is high.

---

## Final Assessment

The learning plateau is a **multi-factor problem** that requires addressing:

1. **Structural** (occurrence-level imbalance) → Day-level sampling
2. **Objective** (BCE-ranking misalignment) → Sampled softmax
3. **Possibly optimization** (schedule not polishing) → LR continuation test

Tier-aware batching at member level is **necessary but insufficient**. The path forward requires either:
- Day-level sampling that guarantees tail code TARGETS in batches, OR
- Per-tier loss balancing that forces gradients to all tiers equally

Before implementing complex interventions, run the **LR polishing test** (2 hours) to rule out schedule as the primary bottleneck.

# Expert 3: 

Based on the comprehensive review of the new evidence (Jan 30 code frequency analysis) and the experimental history, I have synthesized an independent diagnosis and action plan.

### **Executive Summary: The Diagnosis**

You asked if **Tier-Aware Batching** can be adjusted to fix the problem, and if the **tier imbalance** is really the root cause.

**The Answer:**
1.  **Yes, Tier-Aware Batching MUST be adjusted.** The current implementation is fundamentally flawed because it operates on a "Binary Presence" logic (Does this member have a tail code? Yes/No) rather than "Density" logic. Since **83.4%** of members have at least one tail code, the current sampler treats almost everyone as a "Tail Member," failing to concentrate the signal.
2.  **Yes, Occurrence Imbalance IS the Root Cause.** The problem is **Gradient Starvation** caused by the massive disparity in *occurrence* (13.4x difference), not member presence. The model learns the "Common Code Prior" effectively in steps 500-3000, and because tail codes appear in only **5.2% of occurrences** (and only on 22% of days), their gradient signal is statistically drowned out by the common signal in every standard batch.

The plateau represents the point where the model has minimized the loss for the "easy" 85% of gradients (common codes) and simply "given up" on the noisy, low-frequency tail signals.

---

### **1. How to Adjust Tier-Aware Batching (The "Density" Fix)**

You cannot simply switch the Dataloader to "Code Occurrence Level" without breaking the sequence modeling (BERT/Transformer requires the full patient timeline). However, you **can** force the Dataloader to select **"Tail-Dense" Members**.

**The Flaw in Current Implementation:**
*   **Current Logic:** `Tail_Pool = {Members with count(tail_codes) >= 1}`
    *   *Result:* Pool contains 83.4% of all members. Random sampling from this pool yields batches where tail codes are still just ~5% of tokens.
*   **Required Logic:** `Tail_Pool = {Members with count(tail_codes) >= K}` OR `Sample proportional to tail_density`.

**Proposed Solution: Density-Based Tier Sampler**
Do not just check for *presence*. You must bias the sampling towards members who provide a **strong** tail signal.

```python
# Conceptual fix for dev/moe/moe_flashattn_4.py

class DensityTierAwareSampler(Sampler):
    def __init__(self, ...):
        # 1. Calculate Tail Density Score for every member
        #    Score = (Number of Tail Codes in History) / (Total Codes in History)
        #    OR Score = Total Number of Tail Codes (absolute count)
        self.member_scores = self._compute_tail_scores(dataset)

        # 2. Define High-Signal Pools
        #    Instead of "Has > 0", use "Is in top 20% of tail density"
        threshold = np.percentile(self.member_scores, 80)
        self.tail_pool = [i for i, score in enumerate(self.member_scores) if score > threshold]
        
        # 3. Validation
        #    Verify that members in 'tail_pool' have, on average, e.g., 20% tail codes
        #    instead of the global average of 5.2%.
```

**Why this works:** It creates "Concentrated Batches". If a batch is composed of 30% "High-Density Tail Members", the `Gradient_Tail` component in that specific step becomes large enough to compete with `Gradient_Common`, preventing it from being averaged out to zero.

---

### **2. Training Configuration Issues (Independent Analysis)**

You asked to think independently about configurations (LR, Batch Size) beyond just data.

**A. Learning Rate Schedule vs. Gradient Shift**
*   **Observation:** The "Gradient Collapse" happens between **Steps 500 and 3000**.
*   **The Risk:** If you are using a standard `OneCycleLR` or `Linear` warmup that peaks around step 2000-3000, your **Maximum Learning Rate** coincides exactly with the moment the model is deciding "what to learn."
*   **Mechanism:** High LR favors the dominant gradient direction (Common Codes). By the time LR decays (Polishing phase), the weights are already stuck in a "Common Code Basin," and the small gradients from rare codes (even with pos_weight) are too weak to push them out.
*   **Recommendation:** Consider a **longer warmup** (e.g., 2000 steps) or a **lower peak LR**. This gives the "noisy" tail gradients more time to influence the trajectory before the model commits to the easy path.

**B. Batch Size vs. Signal Dilution**
*   **The Risk:** With a global occurrence rate of 5.2%, in a large batch (e.g., 256 or 512), the Law of Large Numbers ensures the gradient is *always* dominated by the mean (Common).
*   **Recommendation:** Paradoxically, **Density-Based Batching** simulates the effect of "Micro-Batching" for tail codes. It creates specific batches where the "local" occurrence rate is 20-30%, violating the global dominance of common codes.

---

### **3. Synthesized Action Plan (Direct Resolution)**

To directly resolve the plateau without changing the objective function (which you noted is for general profiling), you must mechanically force the optimizer to see tail gradients.

#### **Priority 1: Implement "Density-Aware" Batching**
Modify `TierAwareBatchSampler` to sample based on **Tail Density**, not Presence.
*   **Action:** In `dev/moe/moe_flashattn_4.py`, modify `_build_tier_indices`.
*   **Change:** Filter members not just by `any()`, but by `count > Threshold` or `density > Threshold`.
*   **Target:** Ensure the members selected for the `tail_quota` have at least 15-20% tail codes in their sequence (vs the global 5%).

#### **Priority 2: Per-Tier Gradient Normalization (The "Nuclear Option")**
If Density Batching fails, you must decouple the gradient magnitudes.
*   **Mechanism:** In the training loop, compute gradients for Common and Tail loss terms *separately*. Normalize them so `Norm(Grad_Tail) ≈ Norm(Grad_Common)` before adding them.
*   **Why:** This mathematically guarantees that the optimizer step includes a tail component, regardless of how many tail codes were in the batch.
*   **Note:** This is computationally more expensive (requires two backward passes or graph retention), so try Density Batching first.

#### **Priority 3: Adjust LR Schedule**
*   **Action:** Check your warmup steps. If it is < 1000, increase to **2000**. Ensure the model doesn't "lock in" to common codes too early.

### **Summary of Evidence-Based Reasoning**
| Phenomenon | Old Interpretation | New Interpretation (Based on Jan 30 Data) | Fix |
| :--- | :--- | :--- | :--- |
| **Imbalance** | Rare Members (Population Level) | **Rare Occurrences** (Event Level, 13x ratio) | **Density-Based Sampling** (Concentrate events in batches) |
| **Plateau** | Data Capacity Limit | **Gradient Starvation** (Tail signal drowned out) | **Inject Signal** (Via Density Batching or Grad Norm) |
| **Pos_Weight Fail**| Weight not high enough | **Signal too sparse** (Weight x 0 is still 0) | **Increase Density** (Make zeros non-zero) |

**Recommendation:** Switch to **Agent Mode** if you want me to implement the `DensityTierAwareSampler` logic in your codebase. This is the most direct, evidence-backed fix that respects your "Generalized Profile" objective.

# Expert 4: Targeting the Plateau Mechanism

Got it. I’ll (a) read the key evidence docs in `expe_analysis/exp_round5/learning_plateau/` **except** the Feb1 synthesis file you told me not to use, (b) extract concrete facts (metrics/curves/gradient stats) vs hypotheses, then (c) propose a minimal set of decisive experiments and config changes that directly target the plateau mechanism (not just code-frequency imbalance).

[5 tools called]

Nice—your own measurements already narrow the search space a lot: the plateau timing is **stable across `pos_weight_max`**, while gradient share **collapses toward common codes over training** and rare/tail top10 accuracy stays at **0**. Next I’ll pull the exact exp2 training config (LR schedule, batch, wd, warmup), and the tier-aware batching implementation, so we can tell whether we’re actually addressing an **occurrence/day-level** problem vs just **member selection**.

[9 tools called]

Aha: the exp2 configs show `scheduler_type: "linear"` with a **very high final LR floor** (`min_lr_ratio: 0.2`) plus a big `plateau_pct: 0.45`. That’s a classic “never enters polishing mode” pattern—so LR/schedule is now a top suspect independent of the code-frequency story. Next I’ll trace how those fields are implemented in the scheduler code.

[8 tools called]

Next I’ll pinpoint how `use_tier_aware_batching` is implemented (member-level vs day-level), so we can answer your exact question: whether you can directly control **tail occurrence density** rather than just picking members who *ever* have tail codes.

[8 tools called]

### (A) What I see (facts from your attached evidence)

- **Plateau is robust to pos-weight changes**: with `pos_weight_max` 50 → 200, **loss trajectories are nearly identical** and Recall@10 still plateaus (e.g. Recall@10 ~0.843 at steps 8001/10001, 0.854 at 12001 in the 200 run) (`expe_analysis/exp_round5/learning_plateau/exp_round5_exp2_lr_plateau_gradient_observation_jan24.md` L110–L127).
- **Gradient “tier share” collapses during training (emergent)**: early steps are ~balanced, then common rises to ~85% while rare/tail drop near-zero (e.g. common 17–18% at step 1 → 85.3% at step 12001; tail 17–18% → 0.1%) (same file L60–L92).
- **Member-level “tail coverage” is high; occurrence-level is low**:
  - Tail present in **83.4% of members**, but only **5.2% of total occurrences** (common is 69.7%) (`exp_round5_exp2_lr_plateau_code_frequency_check_jan30.md` L8–L16, L28–L36).
- **Your current Tier-Aware Batching is member-level “has-any”**: `TierAwareBatchSampler` flattens **all positive codes across all days of a member**, then puts the member in a tier pool if they have **≥1** code from that tier (`dev/moe/moe_flashattn_4.py` L5734–L5740, L5900–L5916).
- **Your LR schedule is structurally “high-LR for most of the run”**:
  - exp2 config uses `scheduler_type="linear"`, `warmup_pct=0.15`, `plateau_pct=0.45`, `min_lr_ratio=0.2` (`expe_logs/exp_round5/exp2/..._config.json` L10–L21).
  - Implementation is **warmup → plateau → linear decay to `min_lr_ratio`** (`dev/moe/moe_flashattn_4.py` L4841–L4859).

### (B) Primary hypothesis (ranked #1)

**H1 (most likely): this plateau is primarily an optimization/schedule ceiling (no true “polishing” phase), and the head-dominated gradient regime becomes a stable attractor early.**

Mechanism, tied to your evidence:
- The “phase transition” where gradients concentrate happens **very early (≈500–3000 steps)** (Jan24 gradient table), i.e. while you’re in warmup + high-LR plateau. With `warmup_pct + plateau_pct = 0.60`, you spend ~60% of training at/near peak LR, and even the end LR is **20% of peak**. That combination makes it hard to “refine” ranking/top‑K behavior late.
- The fact that **changing `pos_weight_max` doesn’t change plateau timing** strongly suggests the limiter is not “just not enough rare weighting”, but *the training dynamics under the current schedule*.

### (C) Competing hypotheses (and how to distinguish)

- **H2: Sampling is at the wrong granularity (member-level vs occurrence/day-level)**, so even tier-aware member quotas don’t materially change tail *occurrences* per step. Distinguish by measuring **tail occurrences per batch** before/after changes (not just “member has tail ever”).
- **H3: Data / vocab mismatch (“zero-frequency codes” showing many positives)** could break frequency stats, pos_weight, and tiering, and cap downstream utility. Distinguish by verifying **train/val share identical code IDs** and recomputing frequency on the exact training target universe (your Jan24 doc flags “zero code anomaly” as plausible).
- **H4: Objective–metric mismatch (BCE vs ranking)**: macro AUROC can improve while MRR/NDCG worsen (you observed exactly this with pos_weight_max=200) (`exp_round5_exp2_lr_plateau_gradient_observation_jan24.md` L18–L31, L49–L51). Distinguish by a tiny controlled run adding a **ranking-aware evaluation loss** (diagnostic only), without committing to downstream finetuning.

### (D) Decisive experiments (minimal set, maximum information)

1. **Polishing continuation test (schedule diagnosis)**  
   IF plateau is schedule-limited, THEN a short continuation with a much smaller LR should improve Recall/NDCG.
   - Resume the best checkpoint near plateau.
   - Run +10–20% more steps with **no plateau** and **much lower final LR**:
     - Option A: keep `linear` but set `plateau_pct=0.0–0.1`, `min_lr_ratio=0.001–0.01`
     - Option B: switch to `onecycle` (ends at `max_lr/final_div`, typically far lower than 0.2×peak)

2. **Tier-aware batching effectiveness test (sampling diagnosis)**  
   Log per batch:
   - **tail_occurrences_per_batch**, **tail_days_per_batch**, not just “#members with tail ever”.
   IF tail occurrences barely change with tier-aware batching, THEN member-level quotas can’t solve occurrence-level starvation.

3. **Vocab/frequency consistency check (data ceiling diagnosis)**  
   Verify whether any “freq==0” code has positives in train/val due to mapping drift. IF yes, fix that first; otherwise tiering/pos_weight are partially mis-calibrated.

### (E) Action plan (what to do next, directly targeting the plateau)

#### 1) Fix the schedule to create a real polishing phase (most direct, lowest conceptual risk)
- **Change**: in exp2-like runs, move from “60% at peak LR + end at 0.2×peak” to “short peak + long low-LR tail”.
  - Suggested starting point (single change family):
    - `scheduler_type="linear"`
    - `plateau_pct: 0.45 → 0.10`
    - `min_lr_ratio: 0.2 → 0.01` (or 0.001 if stable)
- **Expected outcome if H1 is correct**:
  - Recall@10 plateau shifts upward; NDCG/MRR improve late; metrics stop jittering and start drifting upward in the last 30–40% of steps.
- **Risk/rollback**: if early learning slows, raise peak LR slightly *or* keep plateau_pct but still drop `min_lr_ratio` aggressively (don’t change multiple knobs at once unless needed).

#### 2) Answer to your exact question: can Tier-Aware Batching be adjusted to control medium/rare/tail *occurrence* (not just member)?
- **As implemented today: no**. It guarantees members that *ever* have tier codes (`dev/moe/moe_flashattn_4.py` L5734–L5740, L5900–L5916). Given your stats (tail in **83.4% of members**) this can saturate quickly and still fail to raise tail *occurrence share* (only **5.2%** of occurrences) (`..._code_frequency_check...` L8–L16, L28–L36).
- **What would work (ranked by “directly controls occurrence”):**
  - **(Best within your “member-profile” framing)**: *density-aware tier batching*  
    Replace “member has ≥1 tail code ever” with a score like:
    - `tail_day_count(member)` or `tail_occurrence_count(member)`  
    Then sample from **top tail-density members** to actually increase tail occurrences per batch while keeping the training unit as a member.
  - **(More direct)**: *day-level / window-level sampling*  
    Sample **member-days** (or windows) where tail codes appear; this aligns with your finding that tail codes appear on only **22.3% of days**. This is the cleanest way to control occurrence, but it changes the effective training unit.
  - **(Hybrid)**: keep member as sample, but inside `__getitem__`/collate preferentially include days that contain tail/rare codes when building the sequence, so the model sees tail events more often without changing the external API.

#### 3) If schedule polishing helps plateau but tail still doesn’t move: then do occurrence-level sampling (not more pos_weight)
Your Jan24 evidence shows 50→200 didn’t move rare/tail top10 accuracy off 0 (`...gradient_observation...` L36–L40, L162–L166), so the next lever should be **exposure frequency**, not larger weights.

### (F) Plain-language explanation
Right now the model learns “common-code structure” fast, then gets stuck because your LR schedule stays high for most of training and never gives a long, low-LR phase where subtle ranking improvements can accumulate. Tier-aware batching (as implemented) mostly picks members who *ever* have tail codes—but since most members already do, it doesn’t reliably increase tail *occurrences* in the batches. The most direct next step is: **make the schedule actually polish**, then (only if needed) change tier-aware batching to be **tail-density/day-aware**, not just member-has-any.

If you switch to Agent mode, I can draft a concrete “density-aware tier sampler” patch that minimally modifies `TierAwareBatchSampler` while preserving your current dataset interface.


# Adjudicator: Comprehensive Expert Synthesis & Independent Critical Analysis

## 1. Deep Review of Each Expert's Core Position

### Expert 1: "Self-Reinforcing Gradient Capture Mechanism"
**Core thesis**: The plateau is caused by an emergent, self-reinforcing process where common codes "capture" the gradient budget early, and this capture is stable regardless of pos_weight adjustments.

**Key mechanistic claim**: 
> "The pos_weight mechanism operates per-sample but the gradient concentration happens per-step, which is why 200× pos_weight produces only 0.5% change in gradient tier distribution."

**Unique contribution**: Most rigorous articulation of the **multi-level gradient starvation mechanism** (per-sample → per-batch → per-step aggregation → temporal dynamics). Also clearly identifies that the problem is NOT embedding collapse (healthy norms ~1.1-1.5) but logit suppression (tail logit = -14.69).

---

### Expert 2: "Multi-Factor Interacting Causes"
**Core thesis**: Three factors interact to cause the plateau: (1) batch-level gradient aggregation, (2) BCE-ranking misalignment, (3) possibly LR schedule issues.

**Key mechanistic claim**:
> "Once common codes are well-calibrated, BCE provides **no gradient incentive to improve ranking among competitive candidates**. The loss can decrease while ranking metrics plateau."

**Unique contribution**: Most explicit about **BCE-ranking misalignment** as an independent factor. Provides the clearest evidence: macro_auroc improved (+3.8%) while recall@5 worsened (-4.9%). This is a diagnostic signature that no other expert emphasizes as strongly.

---

### Expert 3: "Binary Presence vs. Density Logic"
**Core thesis**: The current tier-aware batching uses flawed "Binary Presence" logic (does member have ≥1 tail code?) instead of "Density" logic (what fraction of member's codes are tail?).

**Key mechanistic claim**:
> "Since 83.4% of members have at least one tail code, the current sampler treats almost everyone as a 'Tail Member,' failing to concentrate the signal."

**Unique contribution**: Most **actionable framing** of the sampling fix—density-based sampling with explicit percentile thresholds. Also uniquely notes that **high LR during the gradient collapse window (steps 500-3000) favors the dominant gradient direction**.

---

### Expert 4: "Optimization/Schedule Ceiling as Primary Cause"
**Core thesis**: The plateau is primarily an optimization ceiling caused by a schedule that never enters a true "polishing" phase, making the head-dominated gradient regime a stable attractor.

**Key mechanistic claim**:
> "With warmup_pct + plateau_pct = 0.60, you spend ~60% of training at/near peak LR, and even the end LR is 20% of peak. That combination makes it hard to 'refine' ranking/top-K behavior late."

**Unique contribution**: Most **direct inspection of the actual training config** (scheduler_type="linear", min_lr_ratio=0.2, plateau_pct=0.45). Expert 4 is the only one who explicitly reads the config files and traces through the scheduler implementation code to verify the hypothesis.

---

## 2. Synthesis: Shared Conclusions vs. Divergent Positions

### ✅ **Strong Consensus (All 4 Experts Agree)**

| Conclusion | Evidence Cited | Confidence |
|------------|----------------|------------|
| Member-level tier-aware batching is **insufficient** | 83.4% member coverage, but only 5.2% occurrence coverage | HIGH |
| Gradient concentration is **emergent and pos_weight-independent** | 17% → 85% common, unchanged with 5.7× pos_weight increase | HIGH |
| The problem is **occurrence-level, not member-level** | 13.4× occurrence ratio (tail 5.2% vs common 69.7%) | HIGH |
| Tail/rare accuracy remains **0% across all experiments** | Direct observation from evaluation logs | HIGH |
| **Day-level or density-based sampling** is a valid intervention direction | Addresses occurrence-level imbalance | HIGH |

### ⚠️ **Partial Consensus (3/4 Agree)**

| Conclusion | Experts | Dissent |
|------------|---------|---------|
| LR schedule polishing test should be run first | 1, 2, 4 | Expert 3 prioritizes density-based batching first |
| Per-tier gradient normalization is mechanistically sound | 1, 2, 3 | Expert 4 treats this as secondary to schedule fix |

### ❌ **Areas of Divergence (Critical Differences)**

| Issue | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|-------|----------|----------|----------|----------|
| **Primary root cause** | Gradient aggregation dynamics | Multi-factor (aggregation + BCE + schedule) | Occurrence imbalance + binary sampling logic | Schedule ceiling (no polishing phase) |
| **Priority 1 intervention** | Per-tier gradient normalization | LR polishing test | Density-based sampling | Schedule fix (plateau_pct→0.1, min_lr_ratio→0.01) |
| **BCE-ranking misalignment** | Mentioned as secondary | **Emphasized strongly** | Not emphasized | Listed as competing hypothesis |
| **Sampled softmax/ranking loss** | Recommends | Recommends | Does not recommend (notes "generalized profile" objective) | Lists as diagnostic only |

---

## 3. Critical Evaluation: Unique Contributions & Gaps

### Expert 1's Unique Contribution: **Temporal Dynamics Framework**

Expert 1 provides the most complete **temporal model** of the plateau:
- Steps 0-500: Balanced gradients (random init)
- Steps 500-3000: Common codes capture gradient budget
- Steps 3000+: Tail signal becomes noise relative to common refinement

**My assessment**: This framework is mechanistically sound and consistent with the empirical gradient tier tracking data. However, Expert 1 **does not adequately address why** the transition happens at steps 500-3000 specifically. Expert 4 fills this gap by noting that this window coincides with the high-LR phase.

---

### Expert 2's Unique Contribution: **BCE-Ranking Misalignment Evidence**

Expert 2 provides the clearest evidence of **objective-metric misalignment**:
- macro_auroc: +3.8% (improved discrimination)
- recall@5: -4.9% (worsened ranking)
- medium_top10_acc: -96.2% (catastrophic harm)

**My assessment**: This is a critical insight. The pattern where AUROC improves but recall/NDCG worsen is a **diagnostic signature** of BCE-ranking misalignment documented in recommendation systems literature (e.g., Rendle et al., 2020, "Neural Collaborative Filtering vs. Matrix Factorization Revisited"). Expert 2 correctly identifies this as an independent factor, but **no other expert engages with this evidence as deeply**.

---

### Expert 3's Unique Contribution: **Actionable Sampling Fix**

Expert 3 provides the most **implementation-ready** fix:
```python
threshold = np.percentile(self.member_scores, 80)
self.tail_pool = [i for i, score in enumerate(self.member_scores) if score > threshold]
```

**My assessment**: This is pragmatic and aligns with the evidence. However, Expert 3 **does not adequately quantify the expected improvement**. How much would top-20% density sampling increase tail occurrence share per batch? This is calculable from the existing frequency data but is not provided.

---

### Expert 4's Unique Contribution: **Grounding in Actual Config**

Expert 4 is the only expert who **directly inspects the training config and traces through the scheduler code**:
- `scheduler_type="linear"`, `warmup_pct=0.15`, `plateau_pct=0.45`, `min_lr_ratio=0.2`
- "60% of training at/near peak LR, end LR is 20% of peak"

**My assessment**: This is the most rigorous approach. Expert 4 correctly identifies that the schedule configuration is a **testable hypothesis** that can be ruled in/out with a 2-hour polishing continuation test. However, Expert 4 **understates the BCE-ranking misalignment** evidence that Expert 2 highlights.

---

## 4. My Independent Assessment as Principal Tech Lead

### What I Agree With (High Confidence)

**1. Member-level tier-aware batching is provably insufficient.**

The evidence is unambiguous: 83.4% of members have tail codes, yet tail codes represent only 5.2% of occurrences. This is a fundamental **coverage ≠ density** problem. The current `TierAwareBatchSampler` (lines 5734-5740, 5900-5916 in `moe_flashattn_4.py`) checks for binary presence, which is mathematically insufficient given the data distribution.

**2. Gradient concentration is emergent and pos_weight-independent.**

The fact that 5.7× increase in pos_weight produced <0.5% change in gradient tier distribution is strong evidence that the problem is at the **batch-level aggregation** level, not the per-sample weighting level. This is consistent with theoretical work on gradient noise scale in SGD (Smith et al., 2017, "Don't Decay the Learning Rate, Increase the Batch Size").

**3. The LR schedule polishing test is the highest-information, lowest-cost experiment.**

Expert 4's recommendation to run a 2-hour polishing continuation (resume from plateau with 10× lower LR) is the **correct first diagnostic**. This follows the principle of "test the cheapest hypothesis first." If metrics improve, the schedule is a (or the) limiting factor. If not, we proceed to structural interventions.

---

### What I Partially Agree With (Moderate Confidence)

**4. BCE-ranking misalignment is a real factor, but its causal role is unclear.**

Expert 2's evidence (AUROC↑, recall↓) is consistent with objective-metric misalignment. However, I am cautious about attributing this as a **primary** cause vs. a **symptom** of gradient concentration. 

**Mechanistic reasoning**: If the model learns strong negative priors for tail codes (logit = -14.69), then:
- AUROC can still be high for tail codes (correctly ranking tail negatives below common positives)
- But recall@K will be low (tail positives never make it into top-K)

This is consistent with **both** BCE-ranking misalignment **and** gradient starvation as causes. I cannot disambiguate from the current evidence.

**5. Per-tier gradient normalization is mechanistically sound but may have side effects.**

Expert 1 and 3 recommend per-tier gradient normalization. This is theoretically correct: if you force `Norm(Grad_Tail) ≈ Norm(Grad_Common)`, you guarantee tail signal reaches the weights.

However, I note a potential risk: **gradient normalization can destabilize optimization** if the tier boundaries are noisy or if the per-tier sample sizes within a batch are small. With tail at 5.2% of occurrences, a batch of 128 might have 0-2 tail positives on any given step, making the gradient estimate high-variance. Normalizing a high-variance gradient to match a low-variance one can introduce oscillation.

**Mitigation**: Use EMA smoothing on per-tier gradient norms before normalization, or use per-tier loss balancing (Expert 2's suggestion) instead of explicit gradient normalization.

---

### What I Disagree With (or Require More Evidence)

**6. I am skeptical that schedule fix alone will solve the plateau.**

Expert 4 places schedule as the **primary** hypothesis. While I agree the schedule is a contributing factor and the polishing test is essential, I am skeptical that schedule fix alone will move tail/rare accuracy off 0%.

**Reasoning**: The gradient tier tracking shows that by step 3000, tail has dropped to <1% of gradient contribution. If the schedule fix allows more "polishing" after step 3000, the model will polish **what it has learned**, which is a strong common-code prior. Without changing **what gradients the model sees**, a lower LR will refine common codes, not recover tail signal.

**What I expect**: Schedule polishing might improve common code ranking (higher recall@5, lower NDCG), but tail_top10_acc will likely remain at 0% unless combined with density-based sampling or gradient normalization.

**7. Sampled softmax / ranking loss may not be appropriate for the stated objective.**

Experts 1 and 2 recommend sampled softmax or ranking loss. Expert 3 explicitly notes that the user's stated objective is "generalized profile prediction," not ranking.

**My assessment**: If the business metric is truly ranking (recall@K, NDCG), then sampled softmax is appropriate. But if the objective is probability calibration (e.g., for downstream risk scoring), then BCE is the correct objective and the ranking metrics are secondary. **The user needs to clarify the primary business objective** before committing to an objective change.

---

## 5. Evidence-Based Recommendations (Prioritized)

Based on my synthesis, here is my recommended action plan:

### Priority 0: Clarify Business Objective (Before Any Intervention)
- Is the primary goal **ranking** (recall@K, NDCG) or **calibration** (AUROC, Brier)?
- This determines whether objective changes (sampled softmax) are appropriate.

### Priority 1: LR Polishing Test (2-4 hours, Diagnostic)
- Resume from plateau checkpoint with `min_lr_ratio=0.01` (10× lower than current 0.2)
- Run for 10-20% more steps
- **Expected outcome if schedule is limiting**: Common-code recall@K improves, loss decreases
- **Expected outcome if schedule is NOT the bottleneck**: Metrics flat or worse
- **This test is decisive for Expert 4's primary hypothesis**

### Priority 2: Measure Per-Batch Tail Occurrence (1 hour, Diagnostic)
- Before/after tier-aware batching, log:
  - `tail_occurrences_per_batch` (not just "members with tail ever")
  - `tail_positive_labels_per_batch`
- **Expected outcome**: Verify that current member-level batching does NOT increase tail occurrence share
- **This validates the "member ≠ occurrence" hypothesis**

### Priority 3: Density-Based Sampling (1 training run)
- If Priority 2 confirms member-level batching is ineffective:
- Implement density-aware tier sampler (top-20% tail-density members)
- **Success criterion**: `train_grad_tier_tail_frac > 5%` at end of training

### Priority 4: Per-Tier Loss Balancing (If Priority 3 shows improvement but insufficient)
- Aggregate loss separately by tier and weight equally:
  ```python
  total_loss = 0.25 * loss_common + 0.25 * loss_medium + 0.25 * loss_rare + 0.25 * loss_tail
  ```
- **This is safer than explicit gradient normalization** (less variance amplification)

### Priority 5: Sampled Softmax (Only if objective is confirmed as ranking)
- Requires user confirmation that ranking is the primary objective
- Implement tier-stratified negative sampling (25 per tier)

---

## 6. References for Evidence-Based Decision Making

For completeness, here are key research references that support the mechanistic claims in this analysis:

1. **Gradient Noise Scale and Batch Size**: Smith et al. (2017), "Don't Decay the Learning Rate, Increase the Batch Size" — Establishes relationship between batch size, gradient noise, and learning dynamics.

2. **Class Imbalance in Multi-Label Classification**: Johnson & Khoshgoftaar (2019), "Survey on deep learning with class imbalance" — Documents gradient starvation under extreme imbalance.

3. **BCE vs. Ranking Objectives**: Rendle et al. (2020), "Neural Collaborative Filtering vs. Matrix Factorization Revisited" — Shows that pointwise losses (BCE) can optimize discrimination without improving ranking.

4. **Long-Tail Recognition**: Kang et al. (2020), "Decoupling Representation and Classifier for Long-Tailed Recognition" — Proposes two-stage training (representation → classifier) for long-tail problems.

5. **Learning Rate Schedules and Generalization**: Gotmare et al. (2019), "A Closer Look at Deep Learning Heuristics" — Analyzes how schedule choices affect final performance.

---

## 7. Final Summary

| Question | My Answer |
|----------|-----------|
| **Can tier-aware batching be adjusted to fix this?** | Yes, but it must be **density-based** (occurrence-level), not binary presence (member-level). |
| **Is tier imbalance the root cause?** | **Partially**. It is a structural cause, but the **gradient aggregation dynamics** amplify it. Schedule may also contribute. |
| **What is the most direct fix?** | LR polishing test (diagnostic) → Density-based sampling (structural) → Per-tier loss balancing (gradient-level). |
| **What will NOT work?** | Member-level tier-aware batching alone. Increasing pos_weight further. |

**Key insight from my synthesis**: All four experts converge on the diagnosis that **member-level sampling is insufficient because the problem is at the occurrence level**. The disagreement is primarily about **prioritization** (schedule vs. sampling vs. gradient normalization). My recommendation is to follow Expert 4's approach of running the cheap diagnostic (polishing test) first, then proceeding to density-based sampling if needed.





# Solution proposed by Adjudicator: 
This is a crucial clarification that fundamentally changes the analysis. If the goal is **member profiling** (learning representative embeddings for downstream tasks like classification, causal inference), then:

1. **BCE is actually well-aligned** with profiling (it learns probability estimates, not rankings)
2. **Sampled softmax / ranking losses are LESS appropriate** (they optimize for ranking, not calibration/representation)
3. **Embedding quality is the primary objective**, and top-K metrics are useful diagnostics but not the goal

Let me provide a comprehensive analysis of methods to **improve embedding representativeness** for member profiling, including those from the Jan 25 document and additional approaches from the literature.

---

## Comprehensive Methods for Improving Embedding Representativeness

### Category 1: Loss Function Modifications (Beyond BCE + pos_weight)

#### 1.1 Focal Loss (Lin et al., 2017)
**Not explicitly discussed in Jan 25 but highly relevant for your problem.**

**Mechanism**: Down-weights well-classified (easy) examples, focuses gradient on hard examples.

```python
# Focal Loss formulation
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

# γ = 0: Standard BCE
# γ = 2 (common): Heavily down-weights easy examples
# α: Class-balancing weight (like pos_weight)
```

**Why it helps for profiling**:
- Common codes become "easy" after early training → their gradients get down-weighted automatically
- Rare codes remain "hard" → their gradients stay prominent
- **Unlike pos_weight**, focal loss adapts *dynamically* based on model confidence

**Evidence supporting this for imbalanced multi-label**:
- Lin et al. (2017), "Focal Loss for Dense Object Detection" - Original paper
- Ridnik et al. (2021), "Asymmetric Loss for Multi-Label Classification" - Extension for multi-label

**Key hyperparameter**: `γ` (focusing parameter)
- `γ = 0`: Standard BCE (no focusing)
- `γ = 1-2`: Moderate focusing (recommended start)
- `γ > 3`: Aggressive focusing (risk: ignores common codes entirely)

**Implementation consideration**: Focal loss can be combined with tier-aware batching for complementary effects.

---

#### 1.2 Asymmetric Loss (ASL) (Ridnik et al., 2021)
**Designed specifically for multi-label classification with long-tail distributions.**

**Mechanism**: Different focusing parameters for positives vs negatives.

```python
# Asymmetric Loss
L_+ = (1 - p)^γ+ * log(p)     # For positive labels
L_- = (p_m)^γ- * log(1 - p_m)  # For negative labels, with margin m

# p_m = max(p - m, 0)  # Hard threshold margin
```

**Why it helps**:
- Negative examples dominate in multi-label → apply stronger down-weighting to negatives
- Positive examples are precious → preserve their gradients
- The margin `m` provides additional robustness to label noise

**Recommended hyperparameters** (from paper):
- `γ+ = 0` (no down-weighting of positives)
- `γ- = 4` (strong down-weighting of easy negatives)
- `m = 0.05` (small margin)

---

#### 1.3 Distribution-Balanced Loss (Wu et al., 2020)
**Specifically addresses the problem you're seeing: gradient concentration.**

**Mechanism**: Rebalances the negative class contribution by sampling.

```python
# Instead of computing loss over all negatives:
loss = BCE(pos) + BCE(all_negatives)  # Standard

# Distribution-Balanced Loss samples negatives:
loss = BCE(pos) + BCE(sampled_negatives_proportional_to_frequency)
```

**Why it helps for profiling**:
- For tail codes, there are many "false" negatives (codes that could have been present)
- Standard BCE treats all negatives equally → floods tail code gradients
- DB-Loss samples negatives to prevent flooding

---

### Category 2: Representation Learning Objectives (Auxiliary Losses)

#### 2.1 Hierarchical Supervision (From Jan 25 - Approach 2)
**Explicitly discussed, strongly recommended by Expert 3 in Jan 25.**

**Mechanism**: Add CCS/CCSR category-level prediction as auxiliary task.

```python
# Multi-task loss
total_loss = λ_code * BCE_code + λ_category * BCE_category

# Where category targets are derived from code targets
category_targets = aggregate_codes_to_categories(code_targets)
```

**Why it helps for profiling**:
- Tail codes inherit learning signal from their parent categories
- Category-level representations provide regularization
- Aligns with clinical ontology (ICD hierarchy is clinically meaningful)

**Key design choices**:
| Choice | Options | Recommendation |
|--------|---------|----------------|
| Aggregation | Max-pool, Avg-pool, Attention | Attention-weighted (preserves all code gradients) |
| Loss weight | Fixed λ, Learned | Fixed λ=0.1-0.2 initially |
| Hierarchy depth | CCS (coarse), CCSR (medium), Sub-categories | Start with CCS, add CCSR if needed |

---

#### 2.2 Contrastive Pre-training (From Jan 25 - Approach 3)
**Discussed, but dismissed too strongly by Expert 2.**

**Mechanism**: Learn code embeddings from co-occurrence structure before supervised training.

**Why it helps for profiling specifically**:
- Gives ALL codes (including tail) meaningful initialization
- Co-occurrence structure captures clinical relationships
- Pre-trained embeddings are more robust to gradient starvation during fine-tuning

**Implementation options**:

| Method | Positives | Negatives | Pros | Cons |
|--------|-----------|-----------|------|------|
| **Skip-gram** (Mikolov et al., 2013) | Co-occurring codes | Random codes | Simple, fast | May not capture complex relationships |
| **SimCLR-style** (Chen et al., 2020) | Augmented views of same patient | Other patients | Strong representations | Requires augmentation strategy |
| **Supervised Contrastive** (Khosla et al., 2020) | Same diagnosis group | Different groups | Uses label info | Requires grouping strategy |

**Medical-specific reference**:
- Choi et al. (2016), "Multi-layer Representation Learning for Medical Concepts" - Pre-trained medical code embeddings via co-occurrence

---

#### 2.3 Self-Supervised Pre-training (MLM-style)
**Not in Jan 25 but highly relevant for sequence-based member profiling.**

**Mechanism**: Mask random codes in a patient's history, predict them.

```python
# Masked Code Modeling (MCM)
# Similar to BERT's MLM objective

# For patient sequence: [code1, code2, code3, code4, code5]
# Masked input:         [code1, [MASK], code3, [MASK], code5]
# Task: Predict code2 and code4 from context
```

**Why it helps for profiling**:
- Every code (including tail) can be a mask target
- The objective is to predict codes from context, not just classification
- Captures temporal/sequential relationships in patient history

**Reference**:
- Li et al. (2020), "BEHRT: Transformer for Electronic Health Records" - BERT-style pre-training for EHR
- Rasmy et al. (2021), "Med-BERT: Pre-trained Contextualized Embeddings for Medical Text Mining"

---

#### 2.4 Embedding Regularization (From Jan 25 - Approach 1)
**Discussed, marked as "needs diagnostic first" - I agree.**

**Mechanism**: Add loss terms to prevent embedding collapse and encourage diversity.

```python
# Prevent collapse (codes shouldn't have zero embeddings)
min_norm_loss = F.relu(τ - embedding_norms).mean()

# Encourage diversity (codes shouldn't all be the same)
# Option A: Variance regularization
variance_loss = -embedding_variance.mean()

# Option B: Decorrelation (more principled)
cov_matrix = embeddings @ embeddings.T
decorr_loss = (cov_matrix - I).pow(2).sum() / n_codes
```

**When to use**:
- **Only if diagnostic shows embedding collapse** (tail embedding norms ≈ 0)
- Otherwise, this is treating a symptom that may not exist

---

### Category 3: Training Procedure Modifications

#### 3.1 Two-Stage Training (From Jan 25 - Approach 4)
**Discussed, unfairly dismissed by Expert 3 in Jan 25.**

**Mechanism**: Decouple representation learning from classifier training.

**Stage 1: Learn representations with balanced exposure**
```python
# Use instance-balanced sampling or tier-aware batching
# Goal: Learn good encoder representations for ALL codes
```

**Stage 2: Fine-tune classifier with class-balanced loss**
```python
# Freeze or slow-train encoder
# Focus gradient on classifier (decoder) weights
# Use tier-aware batching + focal loss
```

**Reference (critical support)**:
- **Kang et al. (2020), "Decoupling Representation and Classifier for Long-Tailed Recognition"**:
  > "We show that representation learning should use instance-balanced sampling, while classifier training should use class-balanced sampling."

Their finding: This simple decoupling achieves **state-of-the-art** on ImageNet-LT, Places-LT, and iNaturalist.

**Why it helps for profiling**:
- The encoder (your embedding generator) learns from balanced exposure in Stage 1
- The decoder (classifier) is re-calibrated in Stage 2
- **Your embeddings are the output of Stage 1**, which will be balanced

---

#### 3.2 Curriculum Learning
**Not in Jan 25 but relevant for long-tail learning.**

**Mechanism**: Order training samples by difficulty, start with easy, progress to hard.

**Options**:
| Curriculum Strategy | Mechanism | Reference |
|---------------------|-----------|-----------|
| **Anti-curriculum** | Start with hard (rare) samples | Bengio et al. (2009) |
| **Self-paced** | Model selects easy samples first | Kumar et al. (2010) |
| **Transfer curriculum** | Use pre-trained model to score difficulty | Weinshall et al. (2018) |

**For your problem**: Consider **anti-curriculum** for tail codes:
- Early training: Oversample tail/rare codes (while model is still "plastic")
- Late training: Return to natural distribution

This aligns with your evidence that **gradient concentration happens at steps 500-3000**. If tail codes get more exposure during this window, they may avoid being "starved out."

---

#### 3.3 Gradient Accumulation with Tier Quotas (From Jan 25 - Approach 5)
**Discussed, recognized as mechanistically sound but complex.**

**Simplified implementation via tier-aware batching**:
```python
# Instead of variable accumulation, just fix the batch composition
class TierAwareSampler:
    def __init__(self, dataset, tier_quotas):
        # tier_quotas = {'common': 64, 'medium': 32, 'rare': 16, 'tail': 16}
        self.quotas = tier_quotas
        
    def sample_batch(self):
        batch = []
        for tier, quota in self.quotas.items():
            batch.extend(sample_from_tier(tier, quota))
        return batch
```

**This achieves the same goal as gradient accumulation** (ensuring all tiers contribute meaningfully) with simpler implementation.

---

### Category 4: Data-Level Interventions

#### 4.1 Day-Level or Occurrence-Level Sampling
**Emerged from the expert synthesis as the most direct fix.**

**Current problem**: 83.4% of members have tail codes, but tail codes appear on only 22.3% of days and 5.2% of occurrences.

**Solution**: Sample at the **day** or **occurrence** level, not member level.

```python
# Instead of sampling members with tail codes:
tail_members = [m for m in members if has_any_tail_code(m)]

# Sample days where tail codes appear as targets:
tail_days = [(member_id, day_idx) for member_id, day_idx, targets 
             in all_training_samples if any(code in tail_codes for code in targets)]
```

**This directly addresses the occurrence-level imbalance** that member-level sampling cannot fix.

---

#### 4.2 Data Augmentation for Embeddings
**Not in Jan 25 but standard for representation learning.**

**Options**:
| Method | Mechanism | Applicability |
|--------|-----------|---------------|
| **Dropout** | Random feature masking | Standard, already used |
| **Mixup** (Zhang et al., 2018) | Interpolate between samples | Applicable to sequences |
| **CutMix** | Replace segments | Applicable to sequences |
| **Code Masking** | Randomly mask codes in history | Similar to MLM pre-training |

**For medical sequences specifically**:
- **Temporal masking**: Mask random time windows
- **Code substitution**: Replace codes with clinically similar codes (using ICD hierarchy)

---

### Category 5: Architecture Modifications

#### 5.1 Per-Tier Normalization
**Not in Jan 25 but relevant for representation learning.**

**Mechanism**: Apply separate normalization to embeddings from different tiers.

```python
# Instead of one LayerNorm for all code embeddings:
norm_common = LayerNorm(d_model)
norm_medium = LayerNorm(d_model)
norm_rare = LayerNorm(d_model)
norm_tail = LayerNorm(d_model)

# Apply per-tier normalization
embeddings = concat([norm_common(emb_common), norm_medium(emb_medium), ...])
```

**Why it helps**: 
- Normalization statistics for common codes may dominate
- Per-tier normalization ensures each tier has appropriate scale/shift

---

#### 5.2 Separate Embedding Spaces (Lightweight)
**Not in Jan 25 but addresses the capacity concern.**

**Mechanism**: Use separate embedding matrices for different tiers, then project to shared space.

```python
# Instead of one embedding matrix:
# E[n_codes, d_model]

# Use per-tier embeddings with projection:
E_common = Embedding(n_common, d_tier)
E_tail = Embedding(n_tail, d_tier)
proj = Linear(d_tier, d_model)

# Shared representation:
emb_common = proj(E_common(common_codes))
emb_tail = proj(E_tail(tail_codes))
```

**Why it helps**: Tail codes don't compete for the same embedding parameters as common codes, but they share a final representation space.

---

## Summary: Methods Ranked by Relevance for Member Profiling

| Priority | Method | Why for Profiling | Complexity | Risk |
|----------|--------|-------------------|------------|------|
| **1** | LR Polishing Test | Diagnostic first | Low | Low |
| **2** | Tier-Aware Batching (Day-Level) | Ensures all tiers get gradient | Medium | Low |
| **3** | Focal Loss / ASL | Dynamic reweighting based on difficulty | Low | Low |
| **4** | Hierarchical Supervision | Leverages clinical ontology | Medium | Low |
| **5** | Two-Stage Training | Decouples representation from classifier | Medium | Medium |
| **6** | Self-Supervised Pre-training (MCM) | Every code gets learning signal | High | Low |
| **7** | Contrastive Pre-training | Warm-starts rare code embeddings | High | Low |
| **8** | Embedding Regularization | **Only if diagnostic shows collapse** | Low | Low |
| **9** | Curriculum Learning | Focuses on tail early | Medium | Medium |
| **10** | Per-Tier Normalization | Ensures scale consistency | Low | Low |

---

## My Recommendation for Your Profiling Goal

Given that the goal is **member profiling** (not ranking), I would prioritize methods that directly improve **embedding quality for all codes**:

### Immediate Priority (Before Training)
1. **Diagnostic**: Per-code logit/embedding analysis (30 min)
2. **LR Polishing Test**: Rule out schedule as bottleneck (2 hours)

### Phase 1: Training Procedure
3. **Tier-Aware Batching (Day-Level)**: Ensure tail codes appear consistently
4. **Focal Loss (γ=2)** or **ASL**: Replace pure BCE + pos_weight

### Phase 2: Representation Enhancement
5. **Hierarchical Supervision**: Add CCS/CCSR auxiliary task (λ=0.1)
6. **Two-Stage Training**: If Phase 1 shows representation is learning but classifier is biased

### Phase 3: If Above Insufficient
7. **Self-Supervised Pre-training (MCM)**: Pre-train embeddings before supervised training
8. **Contrastive Pre-training**: Alternative if co-occurrence data is rich

### Not Recommended for Profiling
- **Sampled Softmax / Ranking Losses**: These optimize for ranking, not representation quality
- **NDCG/MRR as primary metrics**: Use AUROC, embedding quality metrics instead

---

Would you like me to elaborate on the implementation details for any of these methods, or provide a more detailed comparison between Focal Loss and ASL for your specific use case?