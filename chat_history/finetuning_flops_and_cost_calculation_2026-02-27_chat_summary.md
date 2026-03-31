# Finetuning FLOPS and cost calculation

## Metadata

- **Conversation ID:** 357148a0-5519-479f-9af1-8d4571661ec9
- **Mode:** chat
- **Created:** 2026-02-27 06:58:10
- **Last Updated:** 2026-02-27 07:01:30
- **Branch:** main
- **Files Changed:** 0
- **Lines Added:** 0
- **Lines Removed:** 0
- **Summary:** Calculating Knowledge Distillation FLOPs and Cost

---

## Conversation

### 🧑 User

How to calculate the finetuning FLOPS and finetuning cost? step by step explain

---

### 🤖 Assistant

# Calculating Fine-Tuning FLOPs and Cost

This is a great question that spans both theoretical computation and practical budgeting. Let me walk through it step by step.

---

## Part 1: Calculating Fine-Tuning FLOPs

### Step 1: Understand the Basic FLOPs Formula

The standard approximation for **training FLOPs** of a Transformer model (from the Kaplan et al. / Chinchilla scaling laws) is:

```
FLOPs ≈ 6 × N × D
```

Where:
- **N** = number of trainable parameters
- **D** = number of tokens processed during training (dataset size × epochs)
- **6** = accounts for the forward pass (~2N per token) + backward pass (~4N per token)

This is the "6ND" rule. The factor of 6 comes from:
- Forward pass: ~2 FLOPs per parameter per token (matrix multiply = multiply + accumulate)
- Backward pass: ~4 FLOPs per parameter per token (~2× forward, since you compute gradients w.r.t. both activations and weights)

### Step 2: Adjust for Fine-Tuning Specifics

For **full fine-tuning** (all parameters trainable):

```
FLOPs_finetune = 6 × N_total × D_finetune
```

For **LoRA / PEFT** (only adapter parameters are trainable, but forward pass still traverses the full model):

```
FLOPs_finetune ≈ (2 × N_total + 4 × N_trainable) × D_finetune
```

- Forward pass still uses all N_total parameters → 2 × N_total
- Backward pass only computes gradients for trainable parameters → 4 × N_trainable

In practice, LoRA with rank r on q/v projections might make N_trainable ≈ 0.1–1% of N_total, so the backward pass cost drops dramatically.

### Step 3: Calculate D (Total Tokens Processed)

```
D_finetune = num_samples × avg_sequence_length × num_epochs
```

For example:
- 100,000 samples × 512 tokens/sample × 3 epochs = 153.6M tokens

### Step 4: Worked Example

Say you're fine-tuning a **7B parameter model** on **150M tokens** (full fine-tuning):

```
FLOPs = 6 × 7×10⁹ × 150×10⁶
      = 6 × 1.05×10¹⁸
      = 6.3×10¹⁸ FLOPs
      = 6.3 ExaFLOPs
```

### Step 5: Account for Overhead (Optional Precision)

The 6ND formula excludes:
- **Attention computation**: \( O(s^2 \cdot d \cdot L) \) where s = sequence length, d = hidden dim, L = layers. For short sequences (≤512), this is <5% of total. For long sequences (4096+), it can be 10–20%.
- **Activation checkpointing**: Adds ~33% recomputation FLOPs if used.
- **MoE routing**: If using Mixture of Experts, only a fraction of parameters are active per token. Adjust N to N_active (e.g., top-2 of 8 experts → ~25% of expert parameters active).

For your MoE model specifically, the formula becomes:

```
FLOPs_MoE ≈ 6 × (N_shared + k/E × N_expert) × D
```

Where:
- N_shared = attention + embedding parameters (always active)
- N_expert = total expert FFN parameters
- k = top-k experts selected per token
- E = total number of experts

---

## Part 2: Converting FLOPs to Wall-Clock Time

### Step 6: Know Your Hardware's Peak FLOPs

| GPU | FP16/BF16 Peak | FP32 Peak |
|-----|----------------|-----------|
| A100 (80GB) | 312 TFLOPS | 19.5 TFLOPS |
| H100 (80GB) | 990 TFLOPS | 67 TFLOPS |
| V100 (32GB) | 125 TFLOPS | 15.7 TFLOPS |
| A6000 | 155 TFLOPS | 38.7 TFLOPS |
| RTX 4090 | 165 TFLOPS | 82.6 TFLOPS |

### Step 7: Apply Model FLOPs Utilization (MFU)

Real workloads never hit peak. Typical **MFU** (Model FLOPs Utilization):

- Single GPU: 30–50%
- Multi-GPU (data parallel): 40–55%
- Multi-GPU (tensor parallel): 35–50%
- Well-optimized (DeepSpeed, FSDP): 45–60%

```
Time (seconds) = Total_FLOPs / (GPU_Peak_TFLOPS × 10¹² × MFU × num_GPUs)
```

### Step 8: Worked Example (Time)

7B model, 6.3 ExaFLOPs, single A100, 45% MFU:

```
Time = 6.3×10¹⁸ / (312×10¹² × 0.45 × 1)
     = 6.3×10¹⁸ / 1.404×10¹⁴
     = 44,872 seconds
     ≈ 12.5 GPU-hours
```

---

## Part 3: Calculating Fine-Tuning Cost

### Step 9: Know GPU Pricing

| Provider | GPU | On-Demand $/hr | Spot/Preemptible $/hr |
|----------|-----|-----------------|----------------------|
| AWS | A100 (p4d.24xlarge, 8×A100) | ~$32.77 (full instance) | ~$10–15 |
| GCP | A100 80GB | ~$3.67/GPU | ~$1.10/GPU |
| Azure | A100 80GB | ~$3.67/GPU | ~$1.47/GPU |
| Lambda Labs | A100 80GB | ~$1.10/GPU | N/A |
| RunPod | A100 80GB | ~$1.64/GPU | ~$1.24/GPU |
| Together AI (API) | Per-token pricing | Varies | N/A |

### Step 10: Calculate Cost

```
Cost = GPU_hours × num_GPUs × price_per_GPU_hour
```

From our example:
- 12.5 GPU-hours × 1 GPU × $1.10/hr (Lambda) = **$13.75**
- Same on AWS on-demand: 12.5 / 8 × $32.77 = **$51.20** (you rent 8-GPU instances)

### Step 11: Don't Forget Hidden Costs

1. **Experimentation multiplier**: Typically 3–10× the cost of a single run (hyperparameter search, failed runs, debugging)
2. **Data preprocessing**: CPU time for tokenization, filtering
3. **Evaluation runs**: Inference on validation/test sets
4. **Storage**: Model checkpoints (each checkpoint ≈ 2× model size in FP16)
5. **Data transfer**: Egress fees on cloud providers

---

## Quick Reference Formula (All-in-One)

```
Cost ($) = (6 × N × D) / (GPU_TFLOPS × 10¹² × MFU × 3600) × $/GPU-hr × experiment_multiplier
```

---

## For Your Project Specifically

Given that you're working with a **MoE architecture with FlashAttention**, a few additional considerations:

1. **FlashAttention** reduces memory but doesn't change FLOPs — it just makes the attention FLOPs more efficient (better hardware utilization), so your MFU will be higher (~50–60% vs 35–45% without it).

2. **MoE sparsity** means your effective N per token is smaller than total N. If you have the model architecture details (number of experts, top-k routing), you'd use the MoE-adjusted formula from Step 5.

3. **Mixed precision training** (BF16/FP16) means you use the half-precision TFLOPS numbers for the GPU.

Would you like me to look at your specific model architecture in `moe_flashattn_4.py` and calculate the actual FLOPs for your fine-tuning setup?

---

### 🧑 User

how about distilation

---

### 🤖 Assistant

# Calculating Knowledge Distillation FLOPs and Cost

Distillation introduces a **two-model system** — a teacher and a student — so the cost accounting is different from standard fine-tuning.

---

## Overview: What Makes Distillation Different

In distillation, you have:
1. **Teacher model** (large): runs **inference only** (forward pass) to generate soft targets / logits
2. **Student model** (small): runs **full training** (forward + backward) to learn from the teacher

The total cost is the sum of both.

---

## Part 1: FLOPs Calculation

### Step 1: Teacher Cost (Inference Only)

The teacher only does forward passes — no backward pass, no gradient computation.

```
FLOPs_teacher = 2 × N_teacher × D
```

Where:
- **2** (not 6): forward pass only = 2 FLOPs per parameter per token
- **N_teacher**: total parameters in the teacher model
- **D**: total tokens processed = samples × seq_length × epochs

**Key nuance**: If you pre-compute and cache teacher logits (offline distillation), you pay this cost **once** regardless of how many student training epochs you run. If you run online distillation (teacher generates logits on-the-fly each epoch), you pay this cost **every epoch**.

| Mode | Teacher FLOPs |
|------|---------------|
| **Offline** (pre-computed logits) | 2 × N_teacher × D_one_pass |
| **Online** (on-the-fly) | 2 × N_teacher × D_all_epochs |

### Step 2: Student Cost (Full Training)

The student trains normally, but now with a combined loss (hard labels + soft teacher targets). The loss computation itself is negligible in FLOPs — the dominant cost is still the forward/backward through the student.

```
FLOPs_student = 6 × N_student × D
```

Same as standard fine-tuning. The KL-divergence loss against teacher logits adds negligible FLOPs compared to the matrix multiplications.

### Step 3: Total Distillation FLOPs

```
FLOPs_total = FLOPs_teacher + FLOPs_student
```

**Offline distillation:**
```
FLOPs_total = 2 × N_teacher × D_one_epoch + 6 × N_student × D_all_epochs
```

**Online distillation:**
```
FLOPs_total = (2 × N_teacher + 6 × N_student) × D_all_epochs
```

### Step 4: Worked Example

Say you're distilling from a **7B teacher** into a **1.5B student**, with **150M tokens**, **3 epochs**:

**Offline distillation:**
```
FLOPs_teacher = 2 × 7×10⁹ × 150×10⁶ = 2.1×10¹⁸  (one pass)
FLOPs_student = 6 × 1.5×10⁹ × 450×10⁶ = 4.05×10¹⁸ (3 epochs)
FLOPs_total   = 2.1×10¹⁸ + 4.05×10¹⁸ = 6.15×10¹⁸ = 6.15 ExaFLOPs
```

**Online distillation:**
```
FLOPs_teacher = 2 × 7×10⁹ × 450×10⁶ = 6.3×10¹⁸  (every epoch)
FLOPs_student = 6 × 1.5×10⁹ × 450×10⁶ = 4.05×10¹⁸
FLOPs_total   = 6.3×10¹⁸ + 4.05×10¹⁸ = 10.35×10¹⁸ = 10.35 ExaFLOPs
```

Notice: offline distillation saves ~40% in this case.

### Step 5: MoE Adjustments

For MoE models (relevant to your project), adjust N to account for sparse activation:

```
N_effective = N_shared + (k/E) × N_experts
```

Apply this to whichever model (teacher or student) uses MoE architecture.

---

## Part 2: Memory & Hardware Considerations

### Step 6: Memory Requirements

This is where distillation gets tricky — you may need to hold **both models** in GPU memory simultaneously (for online distillation):

| Component | Memory |
|-----------|--------|
| Teacher weights (FP16) | 2 × N_teacher bytes |
| Student weights (FP16) | 2 × N_student bytes |
| Student optimizer states (AdamW) | 8 × N_student bytes (FP32 weights + momentum + variance) |
| Student gradients (FP16) | 2 × N_student bytes |
| Activations (student) | Varies with batch size & seq length |
| Teacher logits cache | batch_size × seq_len × vocab_size × 4 bytes |

**Example** (7B teacher → 1.5B student, online):
```
Teacher weights:    14 GB
Student weights:     3 GB
Student optimizer:  12 GB
Student gradients:   3 GB
Activations:        ~2-4 GB
─────────────────────────
Total:              ~34-36 GB  → fits on 1× A100-80GB
```

**Offline distillation** advantage: Teacher is not in memory during student training. But you need **disk storage** for cached logits:
```
Storage = num_samples × seq_len × vocab_size × bytes_per_value
```
For 100K samples × 512 tokens × 32K vocab × FP16 (2 bytes) = **3.28 TB** — this is why people often store top-k logits only (e.g., top-64 → ~100× reduction).

### Step 7: Practical Storage Optimization for Offline Distillation

Instead of storing full logit distributions:

| Strategy | Storage Reduction | Quality Impact |
|----------|-------------------|----------------|
| Top-k logits (k=64) | ~500× smaller | Negligible loss |
| Top-k logits (k=8) | ~4000× smaller | Small quality loss |
| Temperature-sharpened logits | Moderate | Negligible |
| Store hidden states instead | vocab-independent | Requires projection layer |

---

## Part 3: Cost Calculation

### Step 8: Time Estimation

You need to account for both models separately since they may run on different hardware:

**Online distillation** (both on same GPU):
```
Time = FLOPs_total / (GPU_TFLOPS × 10¹² × MFU × num_GPUs)
```

**Offline distillation** (two phases):
```
Time_teacher_inference = FLOPs_teacher / (GPU_TFLOPS × 10¹² × MFU_inference)
Time_student_training  = FLOPs_student / (GPU_TFLOPS × 10¹² × MFU_training)
Time_total = Time_teacher_inference + Time_student_training
```

Note: MFU for inference is typically higher (50–70%) than for training (35–55%) because there's no backward pass overhead.

### Step 9: Worked Example (Cost)

Using our 7B→1.5B example, offline distillation, single A100:

```
Teacher inference: 2.1×10¹⁸ / (312×10¹² × 0.60) = 11,218 sec ≈ 3.1 hrs
Student training:  4.05×10¹⁸ / (312×10¹² × 0.45) = 28,846 sec ≈ 8.0 hrs
Total: ~11.1 GPU-hours
```

At various price points:
| Provider | $/GPU-hr | Total Cost |
|----------|----------|------------|
| Lambda Labs | $1.10 | **$12.21** |
| GCP Spot | $1.10 | **$12.21** |
| RunPod | $1.64 | **$18.20** |
| AWS On-Demand (p4d) | $4.10/GPU | **$45.51** |

### Step 10: Hidden Costs Specific to Distillation

1. **Teacher training cost**: If you trained the teacher yourself, add that cost. If using an open-source model (e.g., Llama), this is free.
2. **Logit storage** (offline): Disk/S3 costs for cached outputs.
3. **Hyperparameter search**: Temperature (τ), loss weighting (α between hard and soft loss), learning rate — typically 3–5× multiplier.
4. **Multiple distillation rounds**: Progressive distillation (7B → 3B → 1.5B) doubles the pipeline cost but can improve quality.

---

## Part 4: Distillation vs. Fine-Tuning Cost Comparison

| Approach | FLOPs | GPU-Hours (A100) | Cost (Lambda) |
|----------|-------|-------------------|---------------|
| Fine-tune 7B (3 epochs, 150M tokens) | 6.3 ExaFLOPs | ~12.5 hrs | ~$13.75 |
| Fine-tune 1.5B directly | 1.35 ExaFLOPs | ~2.7 hrs | ~$2.97 |
| Distill 7B→1.5B (offline) | 6.15 ExaFLOPs | ~11.1 hrs | ~$12.21 |
| Distill 7B→1.5B (online) | 10.35 ExaFLOPs | ~18.7 hrs | ~$20.57 |

**Key insight**: Distillation costs roughly the same as fine-tuning the teacher, but you end up with a **smaller, faster model** at inference time. The long-term **inference cost savings** are where distillation pays for itself — a 1.5B model is ~4.7× cheaper to serve than a 7B model.

---

## Quick Reference Formula

```
Distillation Cost ($) = 
  [ (2 × N_teacher × D_teacher) + (6 × N_student × D_student) ]
  / (GPU_TFLOPS × 10¹² × MFU × 3600)
  × $/GPU-hr
  × experiment_multiplier
```

Would you like me to look at your specific MoE architecture and calculate concrete distillation FLOPs for your setup?

