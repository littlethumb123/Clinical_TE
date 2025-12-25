This is an excellent question that deserves a thorough analysis. Let me examine your specific setup and provide a detailed recommendation.

[2 tools called]

Let me also check what data sizes you're typically training with:

[2 tools called]

Now I have a complete picture. Let me provide a comprehensive analysis of LR scheduler options for your specific clinical transformer.

---

# Deep Dive: LR Scheduler Analysis for Your Clinical Transformer

## Your Current Setup

| Parameter | Value |
|-----------|-------|
| **Model** | Hierarchical Clinical Transformer + MoE |
| **Layers** | 6 temporal encoder layers |
| **Embedding** | 256 dim (SwiGLU FFN: 683) |
| **Vocab** | 75,516 input → 6,297 target codes |
| **Batch Size** | 32 per GPU × 4 GPUs = 128 effective |
| **Data Size** | ~350K to 12M patients |
| **Epochs** | 1-10 (often just 1-3) |
| **Base LR** | 1e-4 |
| **Optimizer** | AdamW (weight_decay=0.01) |
| **Current Scheduler** | `CosineAnnealingLR(T_max=total_steps)` with step-level updates |

---

## Analysis of Your Current Implementation

### What You Have (Lines 9700-9701):
```python
total_steps = len(train_loader) * epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

### Current Behavior:
- LR starts at `scaled_lr` (e.g., 4e-4 with 4 GPUs using linear scaling)
- Decays following cosine curve to **0** by the end of training
- Updated every optimizer step (per-batch)

### Issues with Current Approach:

| Issue | Impact | Severity |
|-------|--------|----------|
| **No warmup** | First few steps with full LR can destabilize training, especially with MoE | ⚠️ Medium |
| **Decays to exactly 0** | Last 10-20% of training has near-zero learning, wasted compute | ⚠️ Medium |
| **Single-epoch training issue** | With epochs=1, LR is near 0 by the end of your only epoch | 🔴 High |
| **Linear LR scaling** | You use `base_lr * num_gpus` which is correct, but needs warmup to work well | ⚠️ Medium |

---

## Industry Best Practices for Transformer Pretraining

### 1. **Warmup + Cosine Decay** (GPT-2/3, BERT, LLaMA)

The gold standard for transformer pretraining:

```
LR
↑
│      ╭──────╮
│     ╱        ╲
│    ╱          ╲
│   ╱            ╲_____ min_lr
│  ╱
│ ╱
│╱
└──────────────────────→ Steps
  ↑                 ↑
  warmup           decay
```

**Why it works:**
- **Warmup (5-10% of steps)**: Gradients are noisy early on. Small LR prevents wild updates that can corrupt embeddings.
- **Cosine decay**: Smooth, continuous decay allows fine-tuning as model approaches convergence.
- **Minimum LR floor**: Keeps learning happening even late in training.

**Used by:** GPT-3, LLaMA, Chinchilla, PaLM, most modern LLMs

### 2. **Warmup + Linear Decay** (BERT, RoBERTa)

Simpler alternative:

```
LR
↑
│   ╱╲
│  ╱  ╲
│ ╱    ╲
│╱      ╲
│        ╲
│         ╲
└──────────╲──────────→ Steps
           ↘min_lr
```

**Why it works:**
- Linear is simpler, slightly faster to compute
- Works well for encoder-only models (BERT-style)

### 3. **Warmup + Constant + Decay** (T5, Flan)

Three-phase approach:

```
LR
↑
│     ╭────────────╮
│    ╱              ╲
│   ╱                ╲
│  ╱                  ╲
│ ╱                    ╲
│╱                      ╲
└────────────────────────→ Steps
  ↑      ↑           ↑
 warm  constant    decay
```

**Why it works:**
- Stable "exploration" phase at peak LR
- Good when you have long training runs (10+ epochs)

---

## Specific Analysis for Your Clinical Transformer

### Your Unique Characteristics:

1. **Multi-label BCE loss** (not cross-entropy): BCE is more sensitive to LR since each output is independent
2. **Hierarchical architecture** (daily encoder + temporal encoder): Two-level structure needs stable gradients
3. **MoE layers**: Router gradients are notoriously unstable early in training
4. **Large vocabulary** (6,297 targets): Output layer has many parameters, prone to overfit
5. **Short training** (1-3 epochs often): Every step matters; can't afford wasted epochs

### What This Means for LR Scheduling:

| Factor | Implication |
|--------|-------------|
| Multi-label BCE | Avoid aggressive LR that pushes all logits to extremes |
| MoE router | **Needs warmup** to stabilize expert selection |
| Short training | **Don't decay to 0** - you need learning until the end |
| Large vocab output | Lower peak LR is safer than linear scaling suggests |

---

## My Recommendation: **Warmup + Cosine with Floor**

For your specific setup, I recommend:

```python
def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1  # Don't go below 10% of peak LR
) -> optim.lr_scheduler.LambdaLR:
    """
    Warmup → Cosine decay → Floor
    
    Best for:
    - Short training runs (1-5 epochs)
    - MoE models (router stability)
    - Multi-label classification (BCE needs stable gradients)
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Recommended Parameters for Your Setup:

```python
# In run_single_experiment():
total_steps = len(train_loader) * epochs

# Warmup: 5% of total steps, minimum 100, maximum 2000
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))

# Min LR: 10% of peak (keeps learning happening)
min_lr_ratio = 0.1

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=min_lr_ratio
)
```

### Why These Parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Warmup 5%** | ~500 steps for 10K total | MoE router needs time to stabilize; embeddings need gradual warm-up |
| **Min LR 10%** | 4e-5 if peak is 4e-4 | BCE loss benefits from continued learning; your short epochs need it |
| **Cosine decay** | Smooth curve | Better than linear for finding fine-grained optima |

---

## Alternative Options Ranked

### 🥇 **Option 1: Warmup + Cosine with Floor (RECOMMENDED)**

```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_ratio=0.1
)
```

**Pros:**
- Best for MoE stability
- Works well for short training (1-3 epochs)
- Standard industry practice
- Keeps learning happening throughout

**Cons:**
- Slightly more complex than current

---

### 🥈 **Option 2: OneCycleLR (Good for Single Epoch)**

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=scaled_lr,
    total_steps=total_steps,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos',
    div_factor=25.0,  # Start at max_lr/25
    final_div_factor=1e4  # End at max_lr/10000
)
```

**How it works:**
```
LR
↑
│        ╭╮
│       ╱  ╲
│      ╱    ╲
│     ╱      ╲
│    ╱        ╲
│   ╱          ╲
│  ╱            ╲_____
│ ╱
└──────────────────────→ Steps
```

**Pros:**
- Specifically designed for "1 cycle" training
- Built into PyTorch
- Automatic warmup

**Cons:**
- Less flexible than custom LambdaLR
- `div_factor` and `final_div_factor` require tuning

---

### 🥉 **Option 3: Warmup + Inverse Square Root Decay (Long Training)**

```python
def inverse_sqrt_schedule(current_step: int) -> float:
    warmup_steps = 1000
    if current_step < warmup_steps:
        return float(current_step) / float(warmup_steps)
    return (warmup_steps ** 0.5) / (current_step ** 0.5)
```

**How it works:**
```
LR
↑
│  ╱\
│ ╱  \_______________
│╱
└────────────────────→ Steps
```

**Pros:**
- Excellent for very long training (10+ epochs, millions of samples)
- Used by original Transformer paper ("Attention is All You Need")
- Never reaches zero

**Cons:**
- Decays too slowly for short training
- Not ideal for your 1-3 epoch runs

---

### ❌ **Not Recommended: Your Current Approach**

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

**Problems:**
1. No warmup → MoE router instability
2. Decays to exactly 0 → Wasted last 10-20% of training
3. With 1 epoch, you're at near-zero LR for a significant portion

---

## LR Scaling for Multi-GPU

You're currently using linear scaling:
```python
scaled_lr = base_lr * num_gpus  # 1e-4 * 4 = 4e-4
```

### Analysis:

| Scaling Method | Formula | When to Use |
|----------------|---------|-------------|
| **Linear** | `lr × N` | Standard for SGD, works for most cases |
| **Square Root** | `lr × √N` | More conservative, good for unstable training |
| **None** | `lr` | When using gradient accumulation instead |

### My Recommendation for Your Setup:

For MoE with 4 GPUs, I'd recommend **square root scaling** combined with warmup:

```python
if use_data_parallel:
    scaled_lr = base_lr * math.sqrt(num_gpus)  # 1e-4 * 2 = 2e-4
else:
    scaled_lr = base_lr
```

**Rationale:**
- MoE router is sensitive to large LR updates
- Square root is safer with your 4-GPU setup
- The warmup will gradually reach the scaled LR anyway

---

## Complete Recommended Implementation

```python
import math

def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> optim.lr_scheduler.LambdaLR:
    """
    Industry-standard warmup + cosine decay scheduler.
    
    Used by GPT-3, LLaMA, Chinchilla, and most modern transformers.
    Modified with min_lr_ratio floor for short training runs.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# In run_single_experiment():
# ...

# LR Scaling (conservative for MoE)
if use_data_parallel:
    scaled_lr = base_lr * math.sqrt(num_gpus)  # sqrt scaling for stability
else:
    scaled_lr = base_lr

optimizer = optim.AdamW(
    model.parameters(),
    lr=scaled_lr,
    weight_decay=config.weight_decay
)

# Scheduler setup
total_steps = len(train_loader) * epochs
warmup_steps = max(100, min(2000, int(0.05 * total_steps)))  # 5% warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=0.1  # Never go below 10% of peak LR
)

logger.info(f"Scheduler: Warmup({warmup_steps}) + Cosine(min_ratio=0.1)")
logger.info(f"Peak LR: {scaled_lr:.2e}, Min LR: {scaled_lr * 0.1:.2e}")
```

---

## Summary Comparison

| Scheduler | Your Current | My Recommendation |
|-----------|--------------|-------------------|
| **Type** | CosineAnnealingLR | Warmup + Cosine with Floor |
| **Warmup** | ❌ None | ✅ 5% of steps |
| **Min LR** | 0 (decays to zero) | 10% of peak |
| **LR Scaling** | Linear (4×) | Square root (2×) |
| **MoE Stability** | ⚠️ Risky | ✅ Stable |
| **Short Training** | ❌ Poor | ✅ Optimized |

### Expected Impact:
1. **Faster convergence** in early epochs (warmup prevents unstable early updates)
2. **Better final metrics** (continued learning at 10% LR floor)
3. **More stable MoE routing** (gradual LR increase lets router specialize)
4. **More efficient use of compute** (no wasted near-zero LR steps)

---

Would you like me to also analyze how this interacts with your gradient accumulation settings, or dive deeper into any specific aspect?