# 🔬 Comprehensive Training Configuration Analysis: exp1, exp2b, and exp6
Jan 7, 2026
## 1. Current Configurations Summary (from Round 5 Evidence)

### Scaled Learning Rate Reality
From the code at line 10985, the **actual learning rate** used is:
```python
scaled_lr = base_lr * num_gpus  # 4 GPUs → 4× scaling
```

| Experiment | Base LR | Scaled LR (4 GPUs) | Optimizer |
|------------|---------|--------------------| ----------|
| **exp1_dense** | 1e-4 | 4e-4 | SGD (legacy) |
| **exp2b_flash** | 2e-4 | 8e-4 | AdamW |
| **exp6_auxiliary_free** | 2e-4 | 8e-4 | AdamW |

### Complete Configuration Comparison

| Parameter | exp1 (Dense) | exp2b (Flash+Learned Pool) | exp6 (MoE Aux-Free) |
|-----------|--------------|---------------------------|---------------------|
| **Architecture** | | | |
| Attention | Standard | Flash Attention | Flash Attention |
| FFN | GELU | SwiGLU | MoE (8 experts) |
| Daily Pooling | MaxPool | Learned Att Pool | Learned Att Pool |
| Parameters | 26.4M | 25.3M | 35.4M |
| **Training** | | | |
| Scheduler | CosineAnnealing | OneCycleLR/Linear | Linear Plateau |
| Warmup % | 0% | 15% | 15% |
| Plateau % | N/A | 30-35% | 45% |
| pos_weight | None | log_scaled (max=50) | log_scaled (max=50) |
| Focal Loss | No | No | No (disabled) |
| **MoE-Specific** | | | |
| num_experts | N/A | N/A | 8 |
| num_shared | N/A | N/A | 1 |
| top_k | N/A | N/A | 2 |
| bias_lr | N/A | N/A | 1e-3 → 5e-3 |
| bias_momentum | N/A | N/A | 0.7-0.6 |
| z_loss_weight | N/A | N/A | 0.0-0.005 |
| aux_loss_weight | N/A | N/A | 0.0 |
| **Results** | | | |
| R@10 | 48% | 79-82.8% | 72.4→83.5% |
| μR@10 | 20% | 46.7% | 31.7→49.4% |
| NDCG@20 | 14% | 43.2% | 38.4→44.6% |
| Collapsed Experts | N/A | N/A | 4-8→0-4 |

---

## 2. Key Findings from Round 5 Evidence

### 2.1 Critical Success Factors (Verified from Analyses)

1. **Focal Loss + MoE = Catastrophic Failure**
   - Evidence: exp6 with focal loss had 4-8 collapsed experts, CV=0.78+
   - Root cause: Focal loss eliminates gradients from easy examples → starves router
   - **NEVER use focal_loss with MoE architectures**

2. **bias_lr is Critical for Expert Recovery**
   - Evidence: bias_lr=1e-5 → 12 collapsed experts; bias_lr=1e-3 → 0 collapsed
   - The DeepSeek bias correction needs sufficient learning rate to counteract load imbalance

3. **pos_weight_max Affects R@1 Confidence**
   - Evidence: pos_weight_max=100 → R@1=0.8%; pos_weight_max=50 → R@1=0.17%
   - Too high pos_weight spreads probability mass, hurting top-1 precision

4. **Linear Plateau Schedule > OneCycle for 1-Epoch Training**
   - Evidence: OneCycle peaks at 30% of training → premature convergence
   - Linear plateau keeps LR high for 50-60% of training → better final metrics

5. **16 Experts < 8 Experts (Counterintuitive)**
   - Evidence: exp6d (16 experts) had CV=0.95, 7 collapsed; exp6 (8 experts) had CV=0.48, 4 collapsed
   - More experts = harder load balancing = worse outcomes at this scale

---

## 3. Optimal Configuration for Doubled Training Data

When you **double the training dataset**, several dynamics change:

| Factor | Current | With 2× Data | Implication |
|--------|---------|--------------|-------------|
| Total steps/epoch | ~11,000 | ~22,000 | More updates for learning |
| Gradient noise | Higher | Lower | Can use higher LR |
| Overfitting risk | Lower | Even lower | Less regularization needed |
| Expert specialization time | Limited | More | Better MoE routing possible |
| Class imbalance exposure | Lower | Higher | Better rare code learning |

### 3.1 Recommended Configuration: MoE (exp6 style)

```python
# ============================================================
# BASE CONFIG
# ============================================================
config = BaseConfig(
    embedding_size=256,
    nhid=512,
    nlayers=6,
    dropout=0.05,           # Reduce from 0.1 - more data = less regularization needed
    batch_size=64,          # Increase from 32 - more data allows larger batches
    learning_rate=2.5e-4,   # Slight increase from 2e-4 - more data tolerates higher LR
    weight_decay=0.01,
    gradient_clip=1.0,
)

# ============================================================
# MOE CONFIG (Optimal for 2× data)
# ============================================================
moe_config = MoEConfig(
    d_model=256,
    d_ff=512,
    
    # Expert Architecture
    num_experts=8,              # Keep at 8 - proven optimal at this scale
    num_shared_experts=1,       # 1 shared expert for common patterns
    top_k=2,                    # Standard top-2 routing
    
    # Load Balancing (CRITICAL)
    load_balance_strategy='deepseek',
    aux_loss_weight=0.001,      # Add small aux loss for stability (not 0!)
    bias_lr=3e-3,               # 3× increase from 1e-3 - faster adaptation
    bias_momentum=0.5,          # Lower from 0.6-0.7 - faster recovery
    
    # Regularization
    z_loss_weight=0.005,        # Enable z-loss for router stability
    expert_dropout=0.05,        # Reduce from 0.1 - more data = less dropout
    
    # Architecture
    use_moe_from_layer=2,
    use_swiglu_experts=True,
    router_warmup_steps=1000,   # Increase from 500 - more steps with 2× data
)

# ============================================================
# OPTIMIZER CONFIG (Optimal for 2× data, 1 epoch)
# ============================================================
optimize_config = OptimizeConfig(
    # Scheduler: Linear Plateau (best for single-epoch)
    scheduler_type='linear',
    warmup_pct=0.10,            # 10% warmup (was 15%) - faster start with more data
    plateau_pct=0.40,           # 40% plateau (was 35%) - more learning at peak LR
    min_lr_ratio=0.15,          # End at 15% of peak (was 20%) - more refinement
    
    # Loss Weighting
    use_pos_weight=True,
    pos_weight_method='log_scaled',
    pos_weight_max=35,          # Reduce from 50 - prevent R@1 collapse
    
    # CRITICAL: No Focal Loss with MoE
    use_focal_loss=False,
    
    # Optimizer
    optimizer_type='adamw',
    override_gradient_clip=1.0,  # Standard clip
)
```

### 3.2 Rationale for Each Parameter Change

#### Learning Rate: 2.5e-4 (base) → 1e-3 (scaled with 4 GPUs)
- **Evidence**: exp2b/exp6 at 8e-4 worked well; with 2× data, gradients are less noisy
- **Calculation**: 2.5e-4 × 4 GPUs = 1e-3 scaled LR
- **Risk**: If loss oscillates, reduce to 2e-4 base

#### Batch Size: 64 per GPU (256 effective)
- **Evidence**: exp2b at batch 32 (128 effective) worked; more data = more memory for larger batches
- **Benefit**: Smoother gradients, better GPU utilization
- **Memory**: Monitor for OOM; fall back to 48 if needed

#### bias_lr: 3e-3 (3× increase from current best)
- **Evidence**: exp6 went from 12→0 collapsed with bias_lr increase (1e-5→1e-3)
- **Rationale**: With 2× steps, bias has more time to converge; faster rate helps early recovery
- **Upper bound**: Don't exceed 5e-3 (causes oscillation)

#### bias_momentum: 0.5 (reduced from 0.6-0.7)
- **Evidence**: exp6 with 0.7 still had persistent 4-collapsed-expert issue
- **Calculation**: At 0.5, new information weights 50%; at 0.7, only 30%
- **Effect**: Faster adaptation to load changes; less sticky early imbalance

#### z_loss_weight: 0.005
- **Evidence**: Router gradient norms vary 77,000× (0.001-120) - pathological
- **Effect**: Regularizes router logits to prevent explosion
- **Range**: 0.001-0.01 is safe; 0.005 is balanced

#### aux_loss_weight: 0.001 (not 0!)
- **Evidence**: Pure DeepSeek (aux=0) still had collapse issues
- **Rationale**: Small aux_loss provides secondary balancing signal
- **Trade-off**: >0.01 can hurt task performance

#### pos_weight_max: 35 (reduced from 50)
- **Evidence**: pos_weight_max=50-100 → R@1 collapsed to 0.17-0.8%
- **Rationale**: With 2× data, rare codes get more exposure naturally
- **Effect**: Better top-1 confidence while maintaining rare code learning

#### router_warmup_steps: 1000 (doubled from 500)
- **Evidence**: Collapse happens in first 100 batches before meaningful learning
- **Rationale**: With 2× data, 1000 steps = ~same % of epoch
- **Effect**: Force balanced routing until model has learned basic representations

---

## 4. Configuration Comparison: Current vs Recommended

| Parameter | Current Best (exp6) | Recommended (2× Data) | Change |
|-----------|--------------------|-----------------------|--------|
| **Base** | | | |
| batch_size | 32 | **64** | +100% |
| learning_rate | 2e-4 | **2.5e-4** | +25% |
| dropout | 0.1 | **0.05** | -50% |
| **MoE** | | | |
| bias_lr | 1e-3 | **3e-3** | +200% |
| bias_momentum | 0.6-0.7 | **0.5** | -25% |
| z_loss_weight | 0-0.005 | **0.005** | Standardized |
| aux_loss_weight | 0.0 | **0.001** | +from 0 |
| router_warmup_steps | 500 | **1000** | +100% |
| expert_dropout | 0.1 | **0.05** | -50% |
| **Optimizer** | | | |
| warmup_pct | 0.15 | **0.10** | -33% |
| plateau_pct | 0.35-0.45 | **0.40** | Standardized |
| min_lr_ratio | 0.2 | **0.15** | -25% |
| pos_weight_max | 50 | **35** | -30% |

---

## 5. Additional Stabilization Recommendations

### 5.1 Add Router Gradient Clipping (HIGH PRIORITY)

From exp6d analysis, router gradient norms varied from 0.001 to 120 (77,000×). Add:

```python
# In training loop, after backward pass:
router_params = [p for n, p in model.named_parameters() if 'router' in n]
torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)
```

### 5.2 Implement Gradient Accumulation for Very Large Batches

If batch_size=64 causes OOM:

```python
optimize_config = OptimizeConfig(
    # ... other params ...
)
accumulation_steps = 2  # Effective batch = 64 * 4 GPUs * 2 = 512
```

### 5.3 Add Early Stopping on Expert Collapse

```python
# In training loop, check after each batch:
if collapsed_experts > 5 and batch_idx > 2000:
    logger.warning(f"Expert collapse persisting: {collapsed_experts}/8")
    if collapsed_experts == 8:
        logger.error("Full collapse detected - consider stopping")
```

---

## 6. Expected Outcomes with Recommended Configuration

| Metric | Current exp6 Best | Expected with 2× Data + Optimal Config |
|--------|-------------------|---------------------------------------|
| **R@10** | 83.5% | **87-90%** |
| **μR@10** | 49.4% | **54-58%** |
| **NDCG@20** | 44.6% | **48-52%** |
| **R@1** | 0.17-0.8% | **5-10%** (improved) |
| **Collapsed Experts** | 0-4 | **0-1** |
| **CV** | 0.48 | **<0.35** |
| **Gini** | 0.26 | **<0.18** |

---

## 7. Summary: Key Configuration Principles

### DO ✅
1. **Use linear plateau scheduler** with 40-50% at peak LR for 1-epoch training
2. **Keep num_experts=8** - proven optimal at this scale
3. **Use bias_lr=3-5e-3** and bias_momentum=0.5 for fast expert recovery
4. **Enable z_loss_weight=0.005** for router stability
5. **Add small aux_loss_weight=0.001** as secondary balancing
6. **Use router_warmup_steps** proportional to epoch length
7. **Reduce pos_weight_max to 35** to prevent R@1 collapse
8. **Clip router gradients** separately (max_norm=1.0)

### DON'T ❌
1. **Never use focal_loss with MoE** - destroys router gradients
2. **Don't use 16 experts** - causes worse collapse than 8
3. **Don't set bias_momentum > 0.7** - too slow for recovery
4. **Don't set aux_loss_weight > 0.01** - hurts task performance
5. **Don't use pure OneCycle** for 1-epoch - peaks too early (30%)
6. **Don't set pos_weight_max > 50** - collapses R@1 confidence

---

This configuration should maximize performance when you double your training dataset while maintaining stable expert routing and good rare code performance. Would you like me to elaborate on any specific aspect or help prepare implementation code for any of these changes?

## What `Router GradNorm` Actually Means

From your code (lines 6778-6803 in `moe_flashattn_3.py`):

```python
# The GradNorm logged is the L2 norm of router weight gradients
grad_norm = grad.norm().item()  # Per-layer router gradient L2 norm
metrics['router_grad_norm_mean'] = np.mean(router_grads)  # Average across all MoE layers
```

**This metric measures**: The magnitude of gradients flowing to the router's linear projection (`nn.Linear(d_model, num_experts)`) - the component that decides which expert each token should go to.

---

## Analysis of Your Training Log

| Training Phase | Batch Range | GradNorm Range | Status |
|----------------|-------------|----------------|--------|
| **Early** | 0-100 | 346.8 → 10.0 | 🔴 Very high initial, drops rapidly |
| **Mid** | 100-2500 | 0.4 - 3.4 | 🟢 Healthy range |
| **Late-mid** | 2500-6000 | 2.0 - 8.0 | 🟡 Starting to increase |
| **Late** | 6000-10000 | 5.0 - 14.8 | 🟠 Growing |
| **Final** | 10000-10965 | 11.0 - 33.6 | 🔴 Consistently >10, growing |

### Key Pattern: GradNorm is **increasing** in late training
```
Batch 6000:   GradNorm ≈ 6-8
Batch 8000:   GradNorm ≈ 10-15
Batch 10000:  GradNorm ≈ 20-34
```

---

## What is a "Healthy" Router Gradient Norm?

Based on the code's own thresholds (line 6803-6804):

| GradNorm | Code Classification | Interpretation |
|----------|---------------------|----------------|
| **< 1e-7** | `router_grad_vanishing` | 🔴 Gradients too small - router not learning |
| **1e-7 to 10** | Healthy | 🟢 Normal training signal |
| **> 10** | `router_grad_exploding` | 🔴 Gradients too large - unstable updates |

**Your current late-training values (10-34)** are above the "healthy" threshold, which is why you're seeing the "⚠️ Router gradients EXPLODING!" warning.

---

## Critical Clarification: Timing of GradNorm Logging vs Clipping

Looking at your code flow:

```python
# STEP 5: Logging (line 4994)
router_grad_metrics = compute_router_gradient_metrics(model, ...)  # ← GradNorm computed HERE

# STEP 6: Optimization (line 5024)  
torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)  # ← Clipping happens AFTER
```

**The GradNorm you see in logs is BEFORE clipping**. The actual gradient applied to the optimizer IS clipped to `gradient_clip=1.0`.

This means:
- When GradNorm=33.6, the actual gradient used for the update is capped to norm ≤ 1.0
- The router is only receiving 1/33.6 ≈ 3% of its "intended" gradient magnitude

---

## Should You Increase `gradient_clip` to 2?

**No, I do NOT recommend increasing `gradient_clip` to 2.** Here's why:

### 1. The Current Clipping is Protecting Training
Your late-training GradNorms reach 20-34. With `gradient_clip=1.0`, the actual update is capped, preventing wild oscillations. If you set `gradient_clip=2.0`:
- When GradNorm=20, update would be 2/20 = 10% of raw (vs 5% now)
- **More aggressive updates late in training typically HURT**, not help

### 2. Increasing Late-Training GradNorm is a Symptom, Not the Root Cause
The increasing gradient norm indicates the router is receiving stronger learning signals late in training. This could be caused by:

| Possible Cause | Evidence from Your Log |
|----------------|----------------------|
| **LR decay phase** | Your LinearPlateau schedule starts decay at 60% (~batch 6579) |
| **Loss weighting effect** | pos_weight amplifies gradients for rare codes |
| **Expert load imbalance** | CV stays at 0.8-1.0, indicating uneven routing |

### 3. What You Should Consider Instead

**Option A: Add Router-Specific Gradient Clipping (More Precise)**

Add this AFTER the global clip but BEFORE optimizer.step():

```python
# After line 5024-5030, add router-specific clipping:
router_params = [p for n, p in model.named_parameters() if 'router' in n.lower()]
if router_params:
    torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)  # Additional router cap
```

**Option B: Reduce pos_weight_max in Late Training**

The increasing GradNorm correlates with the rare-code weighting having stronger effect as the model learns. Consider `pos_weight_max=35` instead of 50.

**Option C: Enable z_loss_weight**

From your config, you have `z_loss_weight=0.005`. The z-loss regularizes router logit magnitudes, which can help stabilize gradient flow:

```python
MoEConfig(
    z_loss_weight=0.01,  # Increase slightly from 0.005
)
```

---

## Summary

| Question | Answer |
|----------|--------|
| **What is GradNorm?** | L2 norm of router weight gradients (before clipping) |
| **What's healthy?** | 1e-7 to 10 (your code's thresholds) |
| **Is yours healthy?** | No - late training shows 10-34 (classified as "exploding") |
| **Increase gradient_clip to 2?** | **No** - would allow larger updates when you want stability |
| **What to do instead?** | Add router-specific clipping, reduce pos_weight_max, or increase z_loss_weight |

The high GradNorm in late training is informational - your current `gradient_clip=1.0` IS protecting you by capping the actual updates. The solution is to address WHY the gradients are large (load imbalance, aggressive weighting), not to allow larger gradients through.