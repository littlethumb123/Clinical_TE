# 🔬 Comprehensive Diagnosis: Round 5 `exp2b_flash_learned_pool`

## 📊 Summary of Current Performance

| Metric | Current (1 Epoch) | Target (Previous Best) | Gap |
|--------|-------------------|------------------------|-----|
| **Recall@10** | **79.0%** | **89-91%** | **-10-12%** |
| **Recall@5** | 65.0% | ~73% | -8% |
| **Recall@20** | 86.0% | ~91% | -5% |
| **MRR** | 0.456 | Higher expected | Significant |
| **micro_recall@10** | 46.7% | Higher expected | Significant |
| **Training Time** | 20,484 sec (~5.7 hrs) | - | - |

---

## 🔍 Phase 1: Loss Dynamics Analysis

### Loss Trajectory Profile

| Stage | Step Range | Loss Values | Observations |
|-------|-----------|-------------|--------------|
| Initial | 1-10 | ~0.805 | Starting BCE loss |
| Rapid Decay | 10-1000 | 0.805 → 0.05 | Good initial learning |
| Mid Training | 1000-5000 | ~0.015 → 0.005 | Reasonable progress |
| Late Training | 5000-10966 | ~0.0047 plateau | **Stagnation begins** |
| Final | ~10966+ | ~0.003 | **Premature convergence** |

### 🚨 Key Finding: **Premature Loss Plateau**

The loss trajectory shows:
1. **Fast initial convergence** (first ~1000 steps) - healthy
2. **Loss plateaus at ~0.003** by step 10,000 and barely improves
3. **Final loss std = 0.0654** indicates high variance but low mean change
4. **Train-Val gap ≈ -8e-6** (negative!) → indicates the model is **not overfitting** but **underfitting** the data

---

## 🎯 Phase 2: Root Cause Analysis

### Primary Issue: **Learning Rate Too Conservative After Warmup**

Based on the configuration analysis:

```
Base LR: 1e-4
Scaled LR (4 GPUs): 4e-4 (linear scaling)
Warmup: 5% of total steps (clamped 100-2000)
Cosine decay to 1% of peak (4e-6)
```

**Problem**: With ~10,966 training steps per epoch and cosine annealing:
- By step 5,500 (~50% through), LR has decayed to ~2e-4
- By step 8,200 (~75% through), LR is ~1e-4
- Final 20% of training has very low LR, causing **premature convergence**

### Secondary Issues

| Issue | Evidence | Impact |
|-------|----------|--------|
| **1. Loss plateau too early** | Loss at 0.0047 by step 1000 (of 10,966) | 90% of epoch spent barely improving |
| **2. Micro-recall@10 at 46.7%** | Much lower than recall@10 (79%) | Not learning rare/tail codes well |
| **3. balanced_top10_acc = 19.75%** | Imbalanced code prediction | Dominated by common codes |
| **4. common/medium/rare_top10_acc = 0%** | Only tail codes being predicted | Category imbalance |

### Comparison with Previous MoE Results (exp3_standard_moe, 2 epochs)

| Metric | Current (1 ep) | exp3_standard_moe (2 ep) | Difference |
|--------|----------------|--------------------------|------------|
| Recall@10 | 79.0% | 83.0% | -4% |
| Final Loss | 0.00279 | 0.00339 | Lower but worse metrics! |

**This is a critical finding**: Your loss is **lower** than the MoE baseline but **performance is worse**, suggesting:
- **The model is optimizing the wrong thing** (likely averaging predictions)
- **Loss doesn't correlate well with ranking metrics** for your task

---

## 🚀 Phase 3: Recommendations to Maximize Performance in 1 Epoch

### Tier 1: High-Impact Changes (Implement First)

#### 1. **Increase Peak Learning Rate 2-3×**

```python
# Current (too conservative)
learning_rate: float = 1e-4  # Base
scaled_lr = base_lr * num_gpus  # 4e-4 for 4 GPUs

# Recommended
learning_rate: float = 2e-4  # Base (2× increase)  
# OR use more aggressive scaling
scaled_lr = base_lr * num_gpus  # Now 8e-4 for 4 GPUs
```

**Rationale**: Your loss plateaus extremely early, and there's zero overfitting (generalization gap ≈ 0). This signals capacity to learn more with higher LR.

#### 2. **Increase Warmup Steps & Use Slower Decay**

```python
# Current
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup

# Recommended: Longer plateau before decay
warmup_steps = int(0.10 * total_steps)  # 10% warmup
# Use linear schedule with longer plateau instead of pure cosine

def get_linear_warmup_linear_decay(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Warmup → Plateau (30% of training) → Linear decay"""
    plateau_end = int(0.4 * total_steps)  # Stay at peak LR until 40%
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < plateau_end:
            return 1.0  # Stay at peak
        else:
            # Linear decay from plateau_end to total_steps
            progress = (current_step - plateau_end) / (total_steps - plateau_end)
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

#### 3. **Use OneCycleLR (Best for 1-Epoch Training)**

```python
# For single epoch training, OneCycle is often best
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=8e-4,  # Peak LR (higher than current)
    total_steps=total_steps,
    pct_start=0.3,  # 30% warmup phase
    anneal_strategy='cos',
    div_factor=25,  # start_lr = max_lr/25
    final_div_factor=1000  # end_lr = max_lr/1000
)
```

### Tier 2: Medium-Impact Changes

#### 4. **Add Focal Loss or Label-Weighted BCE for Rare Codes**

Your micro_recall@10 (46.7%) is much lower than recall@10 (79%), indicating poor performance on rare codes.

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

# OR use class weights based on inverse frequency
pos_weight = torch.tensor(code_frequencies.max() / (code_frequencies + 1))
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
```

#### 5. **Increase Effective Batch Size**

With batch_size=32 and 4 GPUs, effective batch = 128. Consider:
```python
batch_size: int = 64  # Per GPU → effective = 256
# May need gradient accumulation if OOM
accumulation_steps = 2  # Simulates 512 effective batch
```

**Larger batches enable more aggressive learning rates** and smoother gradients.

### Tier 3: Architecture Considerations

#### 6. **LearnedAttentionPooling May Be Too Simple**

The current implementation uses single-head attention with a learned query. Consider:

```python
class MultiHeadLearnedPooling(nn.Module):
    """Multi-head version for better code aggregation"""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.queries = nn.Parameter(torch.randn(num_heads, 1, self.head_dim) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
```

---

## 📋 Recommended Action Plan

### Quick Wins (Try First - Minimal Code Changes)

1. **Increase base LR to 2e-4 or 3e-4** (single line change)
2. **Switch to OneCycleLR scheduler** (replace scheduler setup)
3. **Increase warmup to 10%** (single line change)

### Expected Impact

| Change | Expected Recall@10 Improvement |
|--------|-------------------------------|
| Higher LR (2×) | +3-5% |
| OneCycleLR | +2-3% |
| Focal Loss | +1-2% (especially micro_recall) |
| **Combined** | **+6-10%** (targeting 85-89%) |

---

## 🎛️ Suggested Hyperparameter Configuration

```python
@dataclass
class OptimizedConfig(BaseConfig):
    # Model (keep same)
    embedding_size: int = 256
    nhid: int = 512
    nlayers: int = 6
    dropout: float = 0.1
    
    # Training (MODIFIED)
    batch_size: int = 64  # Increase from 32
    learning_rate: float = 3e-4  # Increase from 1e-4 (3×)
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # New hyperparameters
    warmup_pct: float = 0.15  # 15% warmup (vs 5%)
    lr_scheduler: str = 'onecycle'  # vs cosine
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
```

---

## ⚠️ Key Diagnostic Insight

The **loss is low but metrics are poor** pattern indicates your model has learned to predict a "safe" average distribution rather than discriminative rankings. This is a classic symptom of:

1. **Learning rate decaying too fast** → stuck in a suboptimal minimum
2. **BCE loss optimizing wrong objective** → predicting "common codes everywhere"
3. **Not enough training signal** in late steps

The recommendations above are designed to **keep the learning rate higher for longer** and **weight rare codes appropriately**.

Let me first look at your current config file and the training code to provide accurate implementation details.

[6 tools called]

Now I have all the context. Let me provide detailed answers to each of your questions.

---

# Comprehensive Answers to Your Follow-up Questions

## 1. Learning Rate: 2e-4 vs 3e-4 — Is There a Big Difference?

### Short Answer: **Yes, the difference is significant for your case.**

### Quantitative Comparison

| Base LR | Scaled LR (4 GPUs) | % Difference from Current |
|---------|-------------------|---------------------------|
| Current: 1e-4 | 4e-4 | Baseline |
| 2e-4 | 8e-4 | **+100%** |
| 3e-4 | 1.2e-3 | **+200%** |

### Why It Matters for Your Case

**Your specific situation**:
- Loss plateaued at ~0.003 by step 10,000
- Generalization gap ≈ 0 (no overfitting)
- Recall@10 stuck at 79% vs target 89-91%

This indicates **underfitting** — the model has capacity to learn more but isn't.

| LR Choice | Behavior | Risk | Recommendation |
|-----------|----------|------|----------------|
| **2e-4 (→ 8e-4)** | More conservative, ~100% increase | Lower risk of instability | **Start here first** |
| **3e-4 (→ 1.2e-3)** | Aggressive, ~200% increase | Higher risk of oscillation/divergence | Try if 2e-4 underperforms |

### Practical Recommendation

```
Start with 2e-4 (→ 8e-4 scaled) because:
1. It's a safer 2× jump from current
2. If loss still plateaus too early → try 3e-4
3. If loss oscillates/spikes → stay at 2e-4 but adjust scheduler
```

---

## 2. Linear Decay vs Cosine Decay — What's the Difference?

### Visual Comparison

```
Learning Rate over Training Progress (0% → 100%)

Cosine Decay (your current):
1.0 |▓▓▓▓▓▓▓▓▓                      (warmup)
    |         ▓▓▓▓▓▓                 (high LR region - SHORT)
    |               ▓▓▓▓             
    |                   ▓▓▓          (rapid decay - LONG)
    |                      ▓▓▓▓▓▓▓▓▓▓
0.0 |________________________________
    0%   25%   50%   75%   100%

Linear Decay (proposed):
1.0 |▓▓▓▓▓▓▓▓▓                      (warmup)
    |         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        (plateau - LONGER)
    |                       ▓▓▓▓▓    
    |                            ▓▓▓▓ (steady decay)
0.1 |________________________________
    0%   25%   50%   75%   100%
```

### Mathematical Difference

| Phase | Cosine Decay | Linear Decay with Plateau |
|-------|--------------|---------------------------|
| 0-10% | Warmup (both same) | Warmup (same) |
| 10-40% | **Rapid decay starts** | **Stays at peak LR** |
| 40-60% | Already at ~50% peak | Still ~100% peak → starts decay |
| 60-100% | Slow decay (near min) | **Steady linear decay** |

### Why This Matters for Your 1-Epoch Training

**Problem with Cosine Decay for your case**:
- Cosine decay is designed for **multi-epoch training** where you want gradual refinement
- In 1 epoch, you need **maximum learning velocity** for longer
- By step 5,000 (of 10,966), cosine has already dropped LR by ~50%
- The last 50% of your epoch is learning at very low LR → wasted compute

**Solution with Linear Decay + Plateau**:
- Keep peak LR until 40-50% of training
- Then linear decay maintains higher LR longer
- More total learning happens in same wall-clock time

### Implementation for Linear Decay with Plateau

Add this function near line 4057 (where `get_cosine_schedule_with_warmup` is defined):

```python
def get_linear_warmup_plateau_decay(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    plateau_ratio: float = 0.3,  # Stay at peak for 30% after warmup
    min_lr_ratio: float = 0.1
) -> optim.lr_scheduler.LambdaLR:
    """
    Linear warmup → Plateau at peak → Linear decay.
    
    Better for single-epoch training where you want maximum learning early.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Steps for linear warmup
        num_training_steps: Total training steps
        plateau_ratio: Fraction of training to stay at peak LR (after warmup)
        min_lr_ratio: Minimum LR as ratio of peak (0.1 = decay to 10% of peak)
    """
    plateau_end = num_warmup_steps + int(plateau_ratio * num_training_steps)
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < plateau_end:
            # Plateau at peak LR
            return 1.0
        else:
            # Linear decay from plateau_end to end
            decay_steps = num_training_steps - plateau_end
            progress = float(current_step - plateau_end) / float(max(1, decay_steps))
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## 3. OneCycleLR — What Is It and Should You Use It?

### What is OneCycleLR?

OneCycleLR is a **super-convergence scheduler** introduced by Leslie Smith. It uses a **single cycle** of LR that:
1. Starts low
2. Ramps UP to a maximum
3. Then decays down again

### Visual Comparison

```
OneCycleLR (1 full cycle):
         peak
          ▲
         ╱ ╲
        ╱   ╲
       ╱     ╲
      ╱       ╲
start╱         ╲end
─────────────────────
  0%    50%    100%

vs. Your Current Cosine (warmup + decay):
peak
▲
│▓▓▓▓▓
│     ▓▓▓▓
│         ▓▓▓
│            ▓▓▓▓▓▓▓▓
─────────────────────
  0%    50%    100%
```

### Key Differences

| Aspect | OneCycleLR | Cosine with Warmup | Linear with Plateau |
|--------|------------|-------------------|---------------------|
| **Peak LR timing** | Middle (~30%) | Start (after warmup) | After warmup |
| **Philosophy** | "Explore then exploit" | "Start strong, decay" | "Stay strong, then decay" |
| **Best for** | 1-2 epoch training | Multi-epoch training | 1 epoch training |
| **Risk** | May need max_lr tuning | Can plateau early | Simpler to tune |
| **Momentum** | Also cycles momentum | Fixed momentum | Fixed momentum |

### Why OneCycleLR is Often Best for Single-Epoch Training

1. **Early exploration**: Low LR at start lets model find good direction
2. **Peak in middle**: Maximum learning when model has "found" the loss landscape
3. **Final refinement**: Decay at end for fine-tuning
4. **Momentum cycling**: Automatically adjusts momentum (lower when LR high, higher when LR low)

### Relationship Summary

```
                    ┌─────────────────────────────────────┐
                    │         Learning Rate Schedules      │
                    └─────────────────────────────────────┘
                                     │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
      ┌─────▼─────┐          ┌─────▼─────┐          ┌──────▼──────┐
      │  Cosine   │          │  Linear   │          │ OneCycleLR  │
      │  Decay    │          │  Decay    │          │             │
      └───────────┘          └───────────┘          └─────────────┘
            │                       │                       │
      Multi-epoch             1-epoch               1-2 epoch
      (gradual)              (sustained)           (super-convergence)
```

### My Recommendation for Your Case

**Try OneCycleLR first** because:
1. Specifically designed for limited epochs
2. The momentum cycling helps with your plateau problem
3. Well-tested for transformer training

If OneCycleLR doesn't work well (rare), fall back to Linear with Plateau.

### Implementation Details for OneCycleLR

**Step 1: Modify the scheduler setup section** (around line 9818-9829)

Replace:
```python
# OLD CODE (lines 9818-9829)
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=0.01  # Don't decay to exactly 0
)
if is_main:
    logger.info(f"Scheduler: CosineWithWarmup (warmup={warmup_steps}, total={total_steps})")
    logger.info(f"Peak LR: {scaled_lr:.2e}, End LR: {scaled_lr * 0.01:.2e}")
```

With:
```python
# NEW CODE - OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=scaled_lr,           # Peak learning rate
    total_steps=total_steps,    # Total training steps
    pct_start=0.3,              # 30% of training is warmup (ramp up)
    anneal_strategy='cos',      # Cosine annealing for decay phase
    div_factor=25,              # start_lr = max_lr / 25
    final_div_factor=1000,      # end_lr = max_lr / 1000
    three_phase=False           # Use two-phase (up-down) not three-phase
)
if is_main:
    start_lr = scaled_lr / 25
    end_lr = scaled_lr / 1000
    logger.info(f"Scheduler: OneCycleLR (pct_start=0.3, total={total_steps})")
    logger.info(f"LR: {start_lr:.2e} → {scaled_lr:.2e} → {end_lr:.2e}")
```

**Step 2: Adjust the train_epoch function call** (no changes needed)

OneCycleLR is called with `scheduler.step()` just like your current scheduler, so **no changes** are needed in the training loop.

**Step 3: Important Note about Resuming**

If you resume from a checkpoint mid-training, OneCycleLR needs to know the current step. Your existing checkpoint logic should handle this, but verify the scheduler state is being saved/loaded correctly.

---

## 4. Focal Loss vs BCEWithLogitsLoss with pos_weight

### Conceptual Comparison

| Aspect | BCEWithLogitsLoss + pos_weight | Focal Loss |
|--------|-------------------------------|------------|
| **Core idea** | Weight positive examples higher | Down-weight easy examples |
| **Formula** | `weight * BCE` | `(1-p)^γ * BCE` |
| **What it addresses** | Class imbalance (rare codes) | Easy vs hard examples |
| **Parameter** | `pos_weight` per class | `gamma` (focusing) + `alpha` (balance) |
| **Complexity** | Simple, built-in | Requires custom implementation |

### Visual Explanation

```
BCE with pos_weight:
┌──────────────────────────────────────────────────┐
│ Common codes (70% of data)  → weight = 1.0       │
│ Rare codes (30% of data)    → weight = 3.0       │
│                                                  │
│ Effect: Rare codes contribute 3× more to loss    │
└──────────────────────────────────────────────────┘

Focal Loss:
┌──────────────────────────────────────────────────┐
│ Well-predicted (p > 0.9)    → weight ≈ 0.01      │
│ Uncertain (p ≈ 0.5)         → weight ≈ 0.25      │
│ Poorly-predicted (p < 0.1)  → weight ≈ 0.81      │
│                                                  │
│ Effect: Model focuses on HARD examples           │
└──────────────────────────────────────────────────┘
```

### Pros and Cons for Your Case

| | BCEWithLogitsLoss + pos_weight | Focal Loss |
|---|------------------------------|------------|
| **Pros** | • Built-in PyTorch, no custom code<br>• Directly addresses class imbalance<br>• Stable gradient behavior<br>• Easy to tune (just set weights) | • Focuses on hard examples<br>• Can help escape plateaus<br>• Better for long-tail distribution<br>• Reduces effect of easy negatives |
| **Cons** | • Doesn't distinguish easy vs hard<br>• May overweight rare but easy examples | • Requires custom code<br>• Two hyperparameters (γ, α)<br>• Can be unstable early in training<br>• May slow early convergence |

### Which is Better for Your Case?

**Your specific symptoms**:
- `micro_recall@10 = 46.7%` (much lower than `recall@10 = 79%`)
- `common_top10_acc = 0%`, `medium_top10_acc = 0%`, `rare_top10_acc = 0%`
- `tail_top10_acc = 79%` (only predicting tail codes well)

This is **unusual** — your model predicts tail codes well but NOT common codes. This suggests:
1. Your label distribution might be inverted from typical
2. The model may be predicting "everything" (high recall) without precision

### My Recommendation

**Start with BCEWithLogitsLoss + pos_weight** because:
1. Simpler to implement and debug
2. More stable training
3. Directly addresses your imbalance issue

**Then try Focal Loss if** pos_weight doesn't help micro_recall enough.

### Implementation: BCEWithLogitsLoss with pos_weight

**Step 1: Add a function to compute pos_weights** (add near line 9803)

```python
def compute_pos_weights(code_frequencies: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute positive class weights for BCEWithLogitsLoss.
    
    Weights are inversely proportional to frequency:
    - Rare codes get higher weight
    - Common codes get lower weight
    
    Uses smoothed inverse frequency to avoid extreme weights.
    """
    # Add smoothing to avoid division by zero and extreme weights
    freq_smoothed = code_frequencies + 1  # Add 1 to all frequencies
    
    # Inverse frequency weighting
    max_freq = freq_smoothed.max()
    weights = max_freq / freq_smoothed
    
    # Clip extreme weights (common range: 1 to 10)
    weights = np.clip(weights, 1.0, 10.0)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)
```

**Step 2: Modify criterion creation** (around line 9750)

Replace:
```python
# OLD CODE (line 9750)
criterion = nn.BCEWithLogitsLoss()
```

With:
```python
# NEW CODE - Weighted BCE
# First compute pos_weights (after code_frequencies is available)
# We'll defer criterion creation until after data prep
criterion = None  # Will be set after code_frequencies is computed
```

**Step 3: Move criterion creation after data prep** (around line 9804)

After `if code_frequencies is None:`, add:
```python
# Compute class weights and create weighted criterion
pos_weights = compute_pos_weights(code_frequencies, device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
logger.info(f"Using weighted BCE loss (weight range: {pos_weights.min():.2f} - {pos_weights.max():.2f})")
```

**Step 4: Update DataParallelWrapper creation** to use the weighted criterion.

### Implementation: Focal Loss (Alternative)

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Reduces loss contribution from well-classified examples,
    focusing on hard/misclassified ones.
    
    FL(p) = -α * (1-p)^γ * log(p)     for y=1
    FL(p) = -(1-α) * p^γ * log(1-p)   for y=0
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
               - gamma=0: equivalent to standard BCE
               - gamma=2: common choice, good balance
               - gamma=5: very aggressive focusing
        alpha: Balance factor for positive class (0.25 is common)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute probabilities
        p = torch.sigmoid(logits)
        
        # Compute focal weights
        p_t = p * targets + (1 - p) * (1 - targets)  # p if y=1, else (1-p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weights (balance positive/negative)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE (stable formulation)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal modulation
        focal_loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

To use Focal Loss, replace criterion creation with:
```python
criterion = FocalLoss(gamma=2.0, alpha=0.25)
```

---

## 5. OptimizedConfig Implementation

Here's the complete implementation you can add to your code:

### Step 1: Add OptimizedConfig class (near line 400, after BaseConfig)

```python
@dataclass
class OptimizedConfig(BaseConfig):
    """
    Optimized configuration for maximizing performance in limited epochs.
    
    Changes from BaseConfig:
    - Higher learning rate (3e-4 vs 1e-4)
    - Larger batch size (64 vs 32)
    - OneCycleLR scheduler settings
    - Optional focal loss support
    - More aggressive warmup (15% vs 5%)
    
    Designed for 1-2 epoch training where you want maximum learning velocity.
    """
    # ============================================================
    # MODEL ARCHITECTURE (same as base)
    # ============================================================
    len_dy: int = 200
    len_cd: int = 80
    cd_cnt: int = 75516        # Updated based on your vocab
    target_cd_cnt: int = 8850
    embedding_size: int = 256
    nhid: int = 512
    nlayers: int = 6
    dropout: float = 0.1
    gender_vocab: int = 4
    age_vocab: int = 1440
    
    # ============================================================
    # TRAINING (OPTIMIZED)
    # ============================================================
    batch_size: int = 64        # INCREASED from 32 (try 64, fall back to 48 if OOM)
    learning_rate: float = 3e-4  # INCREASED from 1e-4 (3×)
    weight_decay: float = 0.01   # Keep same
    gradient_clip: float = 1.0   # Keep same
    
    # ============================================================
    # SCHEDULER (NEW)
    # ============================================================
    scheduler_type: str = 'onecycle'  # Options: 'cosine', 'linear_plateau', 'onecycle'
    warmup_pct: float = 0.15          # 15% warmup (vs 5%)
    min_lr_ratio: float = 0.01        # End LR = peak * this
    plateau_pct: float = 0.30         # For linear_plateau: stay at peak for this %
    
    # OneCycleLR specific
    onecycle_pct_start: float = 0.30  # Fraction of training to ramp up
    onecycle_div_factor: float = 25   # start_lr = max_lr / div_factor
    onecycle_final_div: float = 1000  # end_lr = max_lr / final_div
    
    # ============================================================
    # LOSS FUNCTION (NEW)
    # ============================================================
    use_focal_loss: bool = False      # Set True to use Focal Loss
    focal_gamma: float = 2.0          # Focusing parameter
    focal_alpha: float = 0.25         # Balance factor
    use_pos_weight: bool = True       # Use frequency-based pos_weight for BCE
    pos_weight_max: float = 10.0      # Maximum weight for rare codes
    
    # ============================================================
    # DEVICE
    # ============================================================
    device: str = 'cuda'
    criterion: str = 'BCEWithLogitsLoss'
    
    def get_scheduler(
        self, 
        optimizer: optim.Optimizer, 
        total_steps: int
    ) -> optim.lr_scheduler._LRScheduler:
        """
        Factory method to create the appropriate scheduler based on config.
        
        Returns:
            Configured LR scheduler
        """
        if self.scheduler_type == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]['lr'],  # Use current LR as max
                total_steps=total_steps,
                pct_start=self.onecycle_pct_start,
                anneal_strategy='cos',
                div_factor=self.onecycle_div_factor,
                final_div_factor=self.onecycle_final_div,
                three_phase=False
            )
        
        elif self.scheduler_type == 'linear_plateau':
            warmup_steps = int(self.warmup_pct * total_steps)
            plateau_end = warmup_steps + int(self.plateau_pct * total_steps)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                elif step < plateau_end:
                    return 1.0
                else:
                    decay_steps = total_steps - plateau_end
                    progress = float(step - plateau_end) / float(max(1, decay_steps))
                    return max(self.min_lr_ratio, 1.0 - progress * (1.0 - self.min_lr_ratio))
            
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        else:  # Default: cosine
            warmup_steps = int(self.warmup_pct * total_steps)
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=self.min_lr_ratio
            )
    
    def get_criterion(
        self, 
        code_frequencies: Optional[np.ndarray] = None,
        device: torch.device = None
    ) -> nn.Module:
        """
        Factory method to create the appropriate loss function.
        
        Args:
            code_frequencies: Array of code frequencies for pos_weight
            device: Device to place tensors on
        
        Returns:
            Loss function (BCEWithLogitsLoss or FocalLoss)
        """
        if self.use_focal_loss:
            return FocalLoss(gamma=self.focal_gamma, alpha=self.focal_alpha)
        
        elif self.use_pos_weight and code_frequencies is not None:
            # Compute inverse frequency weights
            freq_smoothed = code_frequencies + 1
            max_freq = freq_smoothed.max()
            weights = max_freq / freq_smoothed
            weights = np.clip(weights, 1.0, self.pos_weight_max)
            pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        else:
            return nn.BCEWithLogitsLoss()
```

### Step 2: Add FocalLoss class (add near line 800, before DataParallelWrapper)

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Reduces loss contribution from well-classified examples,
    focusing on hard/misclassified ones.
    
    FL(p) = -α * (1-p)^γ * log(p)     for y=1
    FL(p) = -(1-α) * p^γ * log(1-p)   for y=0
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

### Step 3: Modify run_single_experiment to use OptimizedConfig

In `run_single_experiment` (around line 9750), add logic to use the new config methods:

```python
# Around line 9750, replace criterion creation:
# OLD:
# criterion = nn.BCEWithLogitsLoss()

# NEW: Defer criterion creation until after code_frequencies is available
criterion = None  # Will be created after data prep

# ... existing code ...

# Around line 9804, after code_frequencies is set:
# Create criterion using config factory (if using OptimizedConfig)
if hasattr(config, 'get_criterion'):
    criterion = config.get_criterion(code_frequencies, device)
    logger.info(f"Using config-defined criterion: {type(criterion).__name__}")
else:
    criterion = nn.BCEWithLogitsLoss()
    logger.info("Using default BCEWithLogitsLoss")

# Update DataParallelWrapper with new criterion
wrapped_model = DataParallelWrapper(
    model=model,
    config=config,
    criterion=criterion,
    moe_config=moe_config
)

# Around line 9818, replace scheduler creation:
# OLD:
# warmup_steps = max(100, min(2000, int(0.05 * total_steps)))
# scheduler = get_cosine_schedule_with_warmup(...)

# NEW: Use config factory if available
if hasattr(config, 'get_scheduler'):
    scheduler = config.get_scheduler(optimizer, total_steps)
    logger.info(f"Using config-defined scheduler: {config.scheduler_type}")
else:
    # Fallback to original behavior
    warmup_steps = max(100, min(2000, int(0.05 * total_steps)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.01
    )
```

### Step 4: Use OptimizedConfig in your experiment

When running the experiment, use:

```python
# In your experiment setup:
config = OptimizedConfig(
    embedding_size=256,
    batch_size=64,              # Larger batch
    learning_rate=3e-4,         # Higher LR
    scheduler_type='onecycle',  # OneCycleLR
    use_pos_weight=True,        # Weighted BCE for rare codes
    use_focal_loss=False        # Start with weighted BCE
)
```

---

## Summary Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│                    RECOMMENDED CHANGES                       │
│                    (Priority Order)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 1. Increase Learning Rate     │
              │    1e-4 → 2e-4 (conservative) │
              │    1e-4 → 3e-4 (aggressive)   │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 2. Use OneCycleLR Scheduler   │
              │    Best for 1-epoch training  │
              │    pct_start=0.3              │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 3. Add pos_weight to BCE      │
              │    Helps rare code learning   │
              │    pos_weight_max=10.0        │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 4. Increase Batch Size        │
              │    32 → 64 (if memory allows) │
              │    Enables higher LR          │
              └───────────────────────────────┘
```
