# Parallelism Strategies: DDP vs ZeRO vs Tensor Parallelism

## Overview: What Gets Distributed?

| Strategy | Model Weights | Gradients | Optimizer States | Activations | Model Compute |
|----------|--------------|-----------|------------------|-------------|---------------|
| **DDP** | Replicated | All-reduced | Replicated | Sharded (by data) | Replicated |
| **ZeRO-1** | Replicated | All-reduced | **Sharded** | Sharded (by data) | Replicated |
| **ZeRO-2** | Replicated | **Sharded** | **Sharded** | Sharded (by data) | Replicated |
| **ZeRO-3** | **Sharded** | **Sharded** | **Sharded** | Sharded (by data) | Replicated |
| **Tensor Parallel** | **Sharded** | **Sharded** | **Sharded** | **Sharded** | **Sharded** |

---

## 1. DDP (Distributed Data Parallel) - Baseline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DDP (Data Parallelism)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GPU 0                GPU 1                GPU 2                GPU 3       │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐│
│   │ Weights │          │ Weights │          │ Weights │          │ Weights ││
│   │ (FULL)  │          │ (FULL)  │          │ (FULL)  │          │ (FULL)  ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │ Grads   │          │ Grads   │          │ Grads   │          │ Grads   ││
│   │ (FULL)  │          │ (FULL)  │          │ (FULL)  │          │ (FULL)  ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │Optimizer│          │Optimizer│          │Optimizer│          │Optimizer││
│   │ States  │          │ States  │          │ States  │          │ States  ││
│   │ (FULL)  │          │ (FULL)  │          │ (FULL)  │          │ (FULL)  ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │ Data    │          │ Data    │          │ Data    │          │ Data    ││
│   │ Shard 0 │          │ Shard 1 │          │ Shard 2 │          │ Shard 3 ││
│   └─────────┘          └─────────┘          └─────────┘          └─────────┘│
│                                                                              │
│   Memory per GPU: Model + Grads + Optimizer + Activations(1/N data)         │
│                   = 1Φ  + 1Φ   + 2Φ (Adam)  + A/N                           │
│                   = 4Φ + A/N   (where Φ = model params, A = activations)    │
│                                                                              │
│   Communication: All-reduce gradients (2Φ per step)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Memory for 1B parameter model (FP16 + Adam):**
- Weights: 2 GB
- Gradients: 2 GB  
- Optimizer: 8 GB (Adam: momentum + variance in FP32)
- **Total per GPU: ~12 GB** (replicated on ALL GPUs)

---

## 2. ZeRO-1 (Optimizer State Partitioning)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ZeRO Stage 1 (Optimizer Sharding)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GPU 0                GPU 1                GPU 2                GPU 3       │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐│
│   │ Weights │          │ Weights │          │ Weights │          │ Weights ││
│   │ (FULL)  │          │ (FULL)  │          │ (FULL)  │          │ (FULL)  ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │ Grads   │          │ Grads   │          │ Grads   │          │ Grads   ││
│   │ (FULL)  │          │ (FULL)  │          │ (FULL)  │          │ (FULL)  ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │Optimizer│          │Optimizer│          │Optimizer│          │Optimizer││
│   │ States  │          │ States  │          │ States  │          │ States  ││
│   │ (1/4)   │          │ (1/4)   │          │ (1/4)   │          │ (1/4)   ││
│   │ Shard 0 │          │ Shard 1 │          │ Shard 2 │          │ Shard 3 ││
│   └─────────┘          └─────────┘          └─────────┘          └─────────┘│
│                                                                              │
│   Memory per GPU: 1Φ + 1Φ + 2Φ/N + A/N = 2Φ + 2Φ/N + A/N                    │
│   (N=4 GPUs):     2Φ + 0.5Φ = 2.5Φ + A/N                                    │
│                                                                              │
│   Communication: All-reduce grads + All-gather weights after update         │
│                  = 2Φ + Φ = 3Φ per step                                      │
│                                                                              │
│   How it works:                                                              │
│   1. Forward/backward: same as DDP                                           │
│   2. All-reduce gradients (as DDP)                                          │
│   3. Each GPU updates only its optimizer shard                               │
│   4. All-gather to get full updated weights                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Memory for 1B parameter model (4 GPUs):**
- Weights: 2 GB
- Gradients: 2 GB
- Optimizer: 8 GB / 4 = **2 GB** (sharded!)
- **Total per GPU: ~6 GB** (vs 12 GB in DDP)

---

## 3. ZeRO-2 (Gradient + Optimizer Partitioning)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               ZeRO Stage 2 (Gradient + Optimizer Sharding)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GPU 0                GPU 1                GPU 2                GPU 3       │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐│
│   │ Weights │          │ Weights │          │ Weights │          │ Weights ││
│   │ (FULL)  │          │ (FULL)  │          │ (FULL)  │          │ (FULL)  ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │ Grads   │          │ Grads   │          │ Grads   │          │ Grads   ││
│   │ (1/4)   │          │ (1/4)   │          │ (1/4)   │          │ (1/4)   ││
│   │ Shard 0 │          │ Shard 1 │          │ Shard 2 │          │ Shard 3 ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │Optimizer│          │Optimizer│          │Optimizer│          │Optimizer││
│   │ (1/4)   │          │ (1/4)   │          │ (1/4)   │          │ (1/4)   ││
│   └─────────┘          └─────────┘          └─────────┘          └─────────┘│
│                                                                              │
│   Memory per GPU: 1Φ + Φ/N + 2Φ/N = Φ + 3Φ/N                                │
│   (N=4 GPUs):     Φ + 0.75Φ = 1.75Φ + A/N                                   │
│                                                                              │
│   Communication: Reduce-scatter grads + All-gather weights                   │
│                  = Φ + Φ = 2Φ per step (same as DDP!)                        │
│                                                                              │
│   How it works:                                                              │
│   1. Forward: same as DDP                                                    │
│   2. Backward: compute grads, then REDUCE-SCATTER (not all-reduce!)          │
│      - Each GPU receives only its shard of averaged gradients                │
│   3. Each GPU updates its shard with its optimizer shard                     │
│   4. All-gather updated weights                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Memory for 1B parameter model (4 GPUs):**
- Weights: 2 GB
- Gradients: 2 GB / 4 = **0.5 GB** (sharded!)
- Optimizer: 8 GB / 4 = **2 GB** (sharded!)
- **Total per GPU: ~4.5 GB** (vs 12 GB in DDP)

---

## 4. ZeRO-3 (Full Sharding) / FSDP

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          ZeRO Stage 3 / FSDP (Full Model + Grad + Optimizer Sharding)        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GPU 0                GPU 1                GPU 2                GPU 3       │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐│
│   │ Weights │          │ Weights │          │ Weights │          │ Weights ││
│   │ (1/4)   │          │ (1/4)   │          │ (1/4)   │          │ (1/4)   ││
│   │ Shard 0 │          │ Shard 1 │          │ Shard 2 │          │ Shard 3 ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │ Grads   │          │ Grads   │          │ Grads   │          │ Grads   ││
│   │ (1/4)   │          │ (1/4)   │          │ (1/4)   │          │ (1/4)   ││
│   ├─────────┤          ├─────────┤          ├─────────┤          ├─────────┤│
│   │Optimizer│          │Optimizer│          │Optimizer│          │Optimizer││
│   │ (1/4)   │          │ (1/4)   │          │ (1/4)   │          │ (1/4)   ││
│   └─────────┘          └─────────┘          └─────────┘          └─────────┘│
│                                                                              │
│   Memory per GPU: Φ/N + Φ/N + 2Φ/N = 4Φ/N                                   │
│   (N=4 GPUs):     Φ/4 + Φ/4 + 2Φ/4 = Φ (instead of 4Φ!)                     │
│                                                                              │
│   Communication: All-gather weights (forward) + Reduce-scatter grads         │
│                  = Φ (forward) + Φ (backward) + Φ (update) = 3Φ per step    │
│                                                                              │
│   How it works:                                                              │
│   1. Forward: ALL-GATHER weights for current layer → compute → discard       │
│   2. Backward: ALL-GATHER weights again → compute grads → REDUCE-SCATTER     │
│   3. Each GPU updates its weight shard with its optimizer shard              │
│                                                                              │
│   Trade-off: More communication for much lower memory!                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Memory for 1B parameter model (4 GPUs):**
- Weights: 2 GB / 4 = **0.5 GB** (sharded!)
- Gradients: 2 GB / 4 = **0.5 GB** (sharded!)
- Optimizer: 8 GB / 4 = **2 GB** (sharded!)
- **Total per GPU: ~3 GB** (vs 12 GB in DDP)

---

## 5. Tensor Parallelism (TP)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TENSOR PARALLELISM                                    │
│                  (Intra-layer model sharding)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Example: Linear layer W[4096, 16384] split across 4 GPUs                  │
│                                                                              │
│   GPU 0              GPU 1              GPU 2              GPU 3             │
│   ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐     │
│   │ W[:, 0:4k]│       │W[:,4k:8k]│       │W[:,8k:12k│       │W[:,12k:16k│    │
│   │ [4096,4k]│       │ [4096,4k]│       │ [4096,4k]│       │ [4096,4k]│     │
│   └────┬─────┘       └────┬─────┘       └────┬─────┘       └────┬─────┘     │
│        │                  │                  │                  │            │
│        ▼                  ▼                  ▼                  ▼            │
│   ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐     │
│   │ y0 = x@W0│       │ y1 = x@W1│       │ y2 = x@W2│       │ y3 = x@W3│     │
│   └────┬─────┘       └────┬─────┘       └────┬─────┘       └────┬─────┘     │
│        │                  │                  │                  │            │
│        └──────────────────┼──────────────────┼──────────────────┘            │
│                           ▼                                                  │
│                  ┌────────────────┐                                          │
│                  │ y = concat(y0, │  ← All-gather OR                         │
│                  │    y1, y2, y3) │    Reduce (depends on parallel type)     │
│                  └────────────────┘                                          │
│                                                                              │
│   COLUMN PARALLEL (output dim sharded):                                      │
│     y = [y0 | y1 | y2 | y3]   ← All-gather (concatenate)                    │
│                                                                              │
│   ROW PARALLEL (input dim sharded):                                          │
│     y = y0 + y1 + y2 + y3     ← All-reduce (sum)                            │
│                                                                              │
│   Memory per GPU: Model/N + Grads/N + Optimizer/N + Activations/N           │
│                   = 4Φ/N + A/N (everything divided!)                         │
│                                                                              │
│   Communication: Per layer! All-gather/reduce at layer boundaries            │
│                  Very high frequency, needs NVLink                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Difference from ZeRO-3:**
- ZeRO-3: Full weights gathered, compute on FULL layer, then discard
- Tensor Parallel: Each GPU computes on its SHARD of the layer

---

## Communication Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMMUNICATION PATTERNS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DDP:          Gradient All-Reduce (once per step, overlapped with backward)│
│                 │─────────────────│                                         │
│                       2Φ bytes                                               │
│                                                                              │
│   ZeRO-1:       All-Reduce grads + All-Gather weights                       │
│                 │─────────────────│────────│                                │
│                       2Φ             Φ       = 3Φ bytes                      │
│                                                                              │
│   ZeRO-2:       Reduce-Scatter grads + All-Gather weights                   │
│                 │────────│────────│                                         │
│                    Φ        Φ       = 2Φ bytes (same as DDP!)               │
│                                                                              │
│   ZeRO-3/FSDP:  All-Gather (fwd) + All-Gather (bwd) + Reduce-Scatter        │
│                 │────────│────────│────────│                                │
│                    Φ        Φ        Φ       = 3Φ bytes                     │
│                 But happens per LAYER (higher frequency!)                    │
│                                                                              │
│   Tensor ∥:     All-Reduce/All-Gather per LAYER (very high frequency)       │
│                 │──│ │──│ │──│ │──│ │──│ │──│ ... (every matmul)            │
│                 Requires extremely fast interconnect (NVLink mandatory)      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Efficiency Summary

For a **1B parameter model with Adam optimizer (FP16 weights, FP32 optimizer)**:

| Strategy | Weights | Gradients | Optimizer | Total/GPU | vs DDP |
|----------|---------|-----------|-----------|-----------|--------|
| **DDP** | 2 GB | 2 GB | 8 GB | **12 GB** | 1× |
| **ZeRO-1** (4 GPUs) | 2 GB | 2 GB | 2 GB | **6 GB** | 2× better |
| **ZeRO-2** (4 GPUs) | 2 GB | 0.5 GB | 2 GB | **4.5 GB** | 2.7× better |
| **ZeRO-3** (4 GPUs) | 0.5 GB | 0.5 GB | 2 GB | **3 GB** | 4× better |
| **Tensor ∥** (4 GPUs) | 0.5 GB | 0.5 GB | 2 GB | **3 GB** | 4× better |

---

## When to Use Each Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DECISION FRAMEWORK                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Model Size vs Available Memory:                                            │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │  Model fits on 1 GPU with room to spare?                            │  │
│   │  └─► Use DDP (simplest, fastest)                                    │  │
│   │                                                                      │  │
│   │  Model fits but memory tight?                                        │  │
│   │  └─► Use ZeRO-1 or ZeRO-2 (low overhead)                            │  │
│   │                                                                      │  │
│   │  Model doesn't fit on 1 GPU?                                        │  │
│   │  └─► Use ZeRO-3/FSDP (model sharding)                               │  │
│   │      └─► If high-bandwidth (NVLink): consider Tensor Parallelism    │  │
│   │                                                                      │  │
│   │  Model too large even for ZeRO-3?                                   │  │
│   │  └─► Combine: TP (intra-node) + ZeRO-3 (inter-node)                 │  │
│   │      └─► This is what Megatron-LM does for 100B+ models             │  │
│   │                                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practical Recommendations for Your Model (84M params)

| Your Scenario | Recommended Strategy | Why |
|---------------|---------------------|-----|
| **4× T4 (16 GB each)** | DDP | Model easily fits (84M = ~1 GB). DDP is simplest. |
| **4× H100 (80 GB each)** | DDP | Plenty of memory, maximize throughput |
| **Scaling to 1B params** | ZeRO-2 | Good memory savings with minimal overhead |
| **Scaling to 10B+ params** | ZeRO-3/FSDP | Required for model to fit |
| **100B+ params on 8 H100s** | TP + ZeRO-3 | TP within node, ZeRO across nodes |

---

## Hybrid Parallelism (Large Models)

For very large models, strategies are combined:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   3D PARALLELISM (LLaMA-70B example)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Node 0 (8 GPUs)                     Node 1 (8 GPUs)                       │
│   ┌─────────────────────────┐         ┌─────────────────────────┐           │
│   │  Tensor Parallel (TP=8) │ ←────── │  Tensor Parallel (TP=8) │           │
│   │  Within node (NVLink)   │  ZeRO-3 │  Within node (NVLink)   │           │
│   │                         │ ──────► │                         │           │
│   │  GPU 0  GPU 1 ... GPU 7 │         │  GPU 0  GPU 1 ... GPU 7 │           │
│   │  Layer  Layer     Layer │         │  Layer  Layer     Layer │           │
│   │  shard  shard     shard │         │  shard  shard     shard │           │
│   └─────────────────────────┘         └─────────────────────────┘           │
│             ▲                                   ▲                           │
│             │                                   │                           │
│             └───────── Pipeline Parallel ───────┘                           │
│                    (stages across nodes)                                     │
│                                                                              │
│   Tensor Parallel: Splits each layer across GPUs (NVLink required)          │
│   Pipeline Parallel: Splits model layers across nodes                        │
│   Data/ZeRO: Shards optimizer/gradients across all                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Strategy | Memory Efficiency | Communication | Complexity | Best For |
|----------|------------------|---------------|------------|----------|
| **DDP** | 1× (baseline) | 2Φ | Low | Small-medium models that fit |
| **ZeRO-1** | ~1.7× | 3Φ | Low | Memory-tight scenarios |
| **ZeRO-2** | ~2.5× | 2Φ | Low | Good balance, recommended default |
| **ZeRO-3/FSDP** | ~N× | 3Φ (per layer) | Medium | Models too large for single GPU |
| **Tensor ∥** | ~N× | High (per layer) | High | Very large models with NVLink |
| **3D Hybrid** | Optimal | Complex | Very High | 100B+ parameter models |

**For your 84M parameter model on 4× H100: DDP is sufficient and optimal.** ZeRO/FSDP becomes valuable when scaling to billions of parameters.