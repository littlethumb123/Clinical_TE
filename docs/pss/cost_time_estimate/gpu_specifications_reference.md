# GPU Specifications Reference for Training Cost Estimation

**Document Version**: January 27, 2026  
**Purpose**: Official GPU specifications from NVIDIA for accurate training cost and time estimation  
**Sources**: NVIDIA Official Product Pages (nvidia.com/data-center/)

---

## Table of Contents

1. [Overview: Understanding GPU Specifications](#1-overview-understanding-gpu-specifications)
2. [Key Metrics Explained](#2-key-metrics-explained)
3. [GPU Specifications Table](#3-gpu-specifications-table)
4. [NVIDIA T4 Specifications](#4-nvidia-t4-specifications)
5. [NVIDIA L4 Specifications](#5-nvidia-l4-specifications)
6. [NVIDIA A100 Specifications](#6-nvidia-a100-specifications)
7. [NVIDIA H100 Specifications](#7-nvidia-h100-specifications)
8. [Which Specifications to Use for Training Estimation](#8-which-specifications-to-use-for-training-estimation)
9. [Speedup Calculation Methodology](#9-speedup-calculation-methodology)

---

## 1. Overview: Understanding GPU Specifications

When estimating transformer training time and cost, understanding GPU specifications is critical. Different precision formats and compute capabilities affect both **throughput** (how fast you can process data) and **quality** (numerical precision of calculations).

### Why This Matters for Transformer Training

Modern deep learning frameworks like PyTorch can use multiple precision formats:
- **FP32** (Full Precision): Maximum numerical accuracy, slowest
- **FP16/BF16** (Half Precision): Good balance of speed and accuracy, commonly used
- **TF32** (Tensor Float 32): NVIDIA-specific format for faster FP32-like operations
- **FP8** (Quarter Precision): Fastest, but may require careful training techniques
- **INT8/INT4**: Primarily for inference, not training

---

## 2. Key Metrics Explained

### 2.1 TFLOPS (Tera Floating-Point Operations Per Second)

**Definition**: The number of trillion floating-point operations a GPU can perform per second.

**Formula**: 
```
Training Time ∝ Total FLOPs / (GPU TFLOPs × MFU)
```

Where:
- **Total FLOPs**: Computational cost of your model (6 × Parameters × Tokens for transformers)
- **GPU TFLOPs**: Theoretical peak performance from specs
- **MFU (Model FLOPs Utilization)**: Actual efficiency achieved (typically 5-50%)

**Why MFU < 100%**: 
- Memory bandwidth bottlenecks
- Kernel launch overhead
- Data loading and preprocessing
- Communication between GPUs
- Non-parallelizable operations

### 2.2 Precision Formats

| Format | Bits | Range | Use Case | Training Speed |
|--------|------|-------|----------|----------------|
| **FP32** | 32 | ±3.4×10³⁸ | Legacy training, high precision | Slowest (1×) |
| **TF32** | 19 | ±3.4×10³⁸ | Drop-in FP32 replacement (NVIDIA) | ~2× faster |
| **FP16** | 16 | ±65,504 | Mixed precision training | ~2-4× faster |
| **BF16** | 16 | ±3.4×10³⁸ | Mixed precision (better range) | ~2-4× faster |
| **FP8** | 8 | ±448 (E4M3) | Aggressive mixed precision | ~4-8× faster |
| **INT8** | 8 | -128 to 127 | Inference quantization | N/A for training |

### 2.3 GPU Memory

| Metric | Definition | Impact on Training |
|--------|------------|-------------------|
| **Capacity (GB)** | Total VRAM available | Determines max batch size and model size |
| **Bandwidth (GB/s)** | Data transfer speed to/from memory | Critical for memory-bound operations |

**Rule of Thumb**:
- Embedding layers: Memory bandwidth limited
- Attention layers: Compute limited (scales with sequence²)
- FFN layers: Compute limited

### 2.4 Tensor Cores

**Definition**: Specialized hardware units in NVIDIA GPUs designed for matrix multiplication operations (the core of transformer training).

**Why They Matter**: Tensor Cores provide 2-16× speedup over standard CUDA cores for matrix operations, but only work with specific precision formats (FP16, BF16, TF32, FP8, INT8).

### 2.5 TDP (Thermal Design Power)

**Definition**: Maximum power consumption of the GPU under load.

**Impact**:
- Higher TDP = Higher performance potential
- Cost of electricity (minor compared to GPU cost)
- Cooling requirements
- Datacenter constraints

---

## 3. GPU Specifications Summary Table

### Primary Specs for Training Estimation

| GPU | Architecture | FP32 | TF32* | FP16/BF16* | FP8* | Memory | Bandwidth | TDP |
|-----|--------------|------|-------|------------|------|--------|-----------|-----|
| **T4** | Turing | 8.1 TFLOPS | N/A | 65 TFLOPS | N/A | 16 GB | 320 GB/s | 70W |
| **L4** | Ada Lovelace | 30.3 TFLOPS | 120 TFLOPS | 242 TFLOPS | 485 TFLOPS | 24 GB | 300 GB/s | 72W |
| **A100 80GB PCIe** | Ampere | 19.5 TFLOPS | 156 TFLOPS | 312 TFLOPS | N/A | 80 GB | 1,935 GB/s | 300W |
| **A100 80GB SXM** | Ampere | 19.5 TFLOPS | 312 TFLOPS | 624 TFLOPS | N/A | 80 GB | 2,039 GB/s | 400W |
| **H100 SXM** | Hopper | 67 TFLOPS | 989 TFLOPS | 1,979 TFLOPS | 3,958 TFLOPS | 80 GB | 3.35 TB/s | 700W |
| **H100 NVL** | Hopper | 60 TFLOPS | 835 TFLOPS | 1,671 TFLOPS | 3,341 TFLOPS | 94 GB | 3.9 TB/s | N/A |

*Tensor Core performance with sparsity (2:4 structured sparsity)

### Key Ratios vs T4 Baseline

| GPU | FP16 TFLOPS | vs T4 (Raw) | Memory | Memory BW | vs T4 (BW) |
|-----|-------------|-------------|--------|-----------|------------|
| **T4** | 65 | 1.0× | 16 GB | 320 GB/s | 1.0× |
| **L4** | 242 | 3.7× | 24 GB | 300 GB/s | 0.94× |
| **A100 PCIe** | 312 | 4.8× | 80 GB | 1,935 GB/s | 6.0× |
| **A100 SXM** | 624 | 9.6× | 80 GB | 2,039 GB/s | 6.4× |
| **H100 SXM** | 1,979 | 30.4× | 80 GB | 3,350 GB/s | 10.5× |

---

## 4. NVIDIA T4 Specifications

**Source**: https://www.nvidia.com/en-us/data-center/tesla-t4/

### Official Specifications

| Specification | Value | Notes |
|--------------|-------|-------|
| **Architecture** | Turing | Released 2018 |
| **Tensor Cores** | 320 | Turing Tensor Cores |
| **CUDA Cores** | 2,560 | Standard compute units |
| **FP32 Performance** | 8.1 TFLOPS | Single precision |
| **Mixed Precision (FP16/FP32)** | 65 TFLOPS | Tensor Core accelerated |
| **INT8 Precision** | 130 TOPS | For inference |
| **INT4 Precision** | 260 TOPS | For inference |
| **Memory Capacity** | 16 GB GDDR6 | |
| **Memory Bandwidth** | 320+ GB/s | |
| **Interconnect** | PCIe Gen3 x16 | ~15.75 GB/s bidirectional |
| **TDP** | 70W | Very power efficient |

### Training Implications

- **Best for**: Small to medium models, inference, cost-sensitive training
- **Limitations**: 
  - No TF32 support (Turing architecture)
  - No FP8 support
  - Limited memory bandwidth for large embedding tables
  - 16 GB memory limits batch size
- **Recommended precision**: FP16 mixed precision (65 TFLOPS)

---

## 5. NVIDIA L4 Specifications

**Source**: https://www.nvidia.com/en-us/data-center/l4/

### Official Specifications

| Specification | Value | Notes |
|--------------|-------|-------|
| **Architecture** | Ada Lovelace | Released 2023 |
| **Form Factor** | L4 (Low Profile) | Single-slot, PCIe |
| **FP32 Performance** | 30.3 TFLOPS | 3.7× faster than T4 |
| **TF32 Tensor Core** | 120 TFLOPS | With sparsity |
| **FP16 Tensor Core** | 242 TFLOPS | With sparsity |
| **BF16 Tensor Core** | 242 TFLOPS | With sparsity |
| **FP8 Tensor Core** | 485 TFLOPS | With sparsity |
| **INT8 Tensor Core** | 485 TOPS | With sparsity |
| **Memory Capacity** | 24 GB GDDR6 | 50% more than T4 |
| **Memory Bandwidth** | 300 GB/s | Similar to T4 |
| **NVENC / NVDEC / JPEG** | 2 / 4 / 4 | Video encoding support |
| **TDP** | 72W | Same power envelope as T4 |

### Training Implications

- **Best for**: Cost-effective training upgrade from T4, medium models
- **Advantages**:
  - 3.7× FP16 performance vs T4
  - FP8 support for aggressive speedup
  - 50% more memory (24 GB vs 16 GB)
  - Same power/cooling as T4
- **Limitations**:
  - Memory bandwidth not improved vs T4
  - No NVLink support for multi-GPU scaling
- **Recommended precision**: FP16/BF16 (242 TFLOPS) or FP8 (485 TFLOPS)

---

## 6. NVIDIA A100 Specifications

**Source**: https://www.nvidia.com/en-us/data-center/a100/

### Official Specifications (80GB Variants)

| Specification | A100 80GB PCIe | A100 80GB SXM | Notes |
|--------------|----------------|---------------|-------|
| **Architecture** | Ampere | Ampere | Released 2020 |
| **FP64 Performance** | 9.7 TFLOPS | 9.7 TFLOPS | Double precision |
| **FP64 Tensor Core** | 19.5 TFLOPS | 19.5 TFLOPS | |
| **FP32 Performance** | 19.5 TFLOPS | 19.5 TFLOPS | |
| **TF32 Tensor Core** | 156 TFLOPS | 312 TFLOPS* | SXM has 2× with sparsity |
| **BF16 Tensor Core** | 312 TFLOPS | 624 TFLOPS* | |
| **FP16 Tensor Core** | 312 TFLOPS | 624 TFLOPS* | |
| **INT8 Tensor Core** | 624 TOPS | 1,248 TOPS* | |
| **Memory Capacity** | 80 GB HBM2e | 80 GB HBM2e | |
| **Memory Bandwidth** | 1,935 GB/s | 2,039 GB/s | 6× faster than T4 |
| **Interconnect** | PCIe Gen4 | NVLink | SXM has better scaling |
| **TDP** | 300W | 400W | |
| **Multi-Instance GPU** | Up to 7 MIGs | Up to 7 MIGs | 10GB per MIG |

*With 2:4 structured sparsity

### Training Implications

- **Best for**: Large-scale training, enterprise workloads
- **Advantages**:
  - Massive memory bandwidth (6× T4) - critical for transformers
  - 80 GB memory enables large batch sizes
  - NVLink (SXM) enables efficient multi-GPU scaling
  - TF32 allows drop-in FP32 replacement with 2× speedup
- **Key difference PCIe vs SXM**:
  - SXM variant has 2× FP16 performance with sparsity
  - SXM has NVLink for better multi-GPU communication
  - Use **312 TFLOPS (PCIe)** for conservative estimates on GCP

---

## 7. NVIDIA H100 Specifications

**Source**: https://www.nvidia.com/en-us/data-center/h100/

### Official Specifications

| Specification | H100 SXM | H100 NVL | Notes |
|--------------|----------|----------|-------|
| **Architecture** | Hopper | Hopper | Released 2022 |
| **FP64 Performance** | 34 TFLOPS | 30 TFLOPS | |
| **FP64 Tensor Core** | 67 TFLOPS | 60 TFLOPS | |
| **FP32 Performance** | 67 TFLOPS | 60 TFLOPS | |
| **TF32 Tensor Core** | 989 TFLOPS* | 835 TFLOPS* | With sparsity |
| **BF16 Tensor Core** | 1,979 TFLOPS* | 1,671 TFLOPS* | With sparsity |
| **FP16 Tensor Core** | 1,979 TFLOPS* | 1,671 TFLOPS* | With sparsity |
| **FP8 Tensor Core** | 3,958 TFLOPS* | 3,341 TFLOPS* | With sparsity |
| **INT8 Tensor Core** | 3,958 TOPS* | 3,341 TOPS* | |
| **Memory Capacity** | 80 GB HBM3 | 94 GB HBM3 | NVL has more memory |
| **Memory Bandwidth** | 3.35 TB/s | 3.9 TB/s | 10× faster than T4 |

*With 2:4 structured sparsity

### Training Implications

- **Best for**: Cutting-edge LLM training, maximum throughput
- **Advantages**:
  - FP8 support for 2× speedup over FP16
  - Transformer Engine for automatic precision management
  - Massive memory bandwidth (10× T4)
  - NVLink 4.0 for ultra-fast multi-GPU scaling
- **H100 SXM vs NVL**:
  - SXM: Higher compute, 80 GB memory
  - NVL: Designed for LLM inference, 94 GB memory, slightly lower compute
  - For training, **H100 SXM is preferred**

---

## 8. Which Specifications to Use for Training Estimation

### Recommended Precision for Transformer Training

| GPU | Recommended Precision | TFLOPS to Use | Rationale |
|-----|----------------------|---------------|-----------|
| **T4** | FP16 Mixed Precision | **65 TFLOPS** | Only efficient option; no TF32/FP8 |
| **L4** | FP16/BF16 | **242 TFLOPS** | Best balance; FP8 requires model support |
| **A100 PCIe** | FP16/BF16 | **312 TFLOPS** | Conservative (no sparsity) |
| **A100 SXM** | FP16/BF16 | **312-624 TFLOPS** | Use 312 without sparsity, 624 with |
| **H100 SXM** | BF16 or FP8 | **989-1,979 TFLOPS** | Use 989 (TF32) for conservative, 1,979 for FP16 |

### Why FP16/BF16 is Standard for Training

1. **Numerical Stability**: FP16 with loss scaling prevents underflow
2. **Framework Support**: PyTorch `torch.cuda.amp` is optimized for FP16
3. **Tensor Core Utilization**: Maximum speedup on modern GPUs
4. **Model Quality**: Minimal impact on final model accuracy

### When to Use FP8

- H100 or newer GPUs only
- Models designed for FP8 (with proper scaling)
- Aggressive optimization scenarios
- Not recommended for first training runs

---

## 9. Speedup Calculation Methodology

### Step 1: Understand the Bottleneck Mix

For a hierarchical clinical transformer (~27-35M parameters):

| Component | % of Compute | Bottleneck | Notes |
|-----------|--------------|------------|-------|
| Embedding lookups | ~20-30% | Memory BW | Large vocab (75K codes) |
| Attention (Q,K,V) | ~30-40% | Compute | Scales with seq² |
| FFN layers | ~30-40% | Compute | Dense matrix multiply |
| LayerNorm, Softmax | ~5-10% | Memory BW | Element-wise ops |

### Step 2: Calculate Weighted Speedup

**Formula:**
```
Effective_Speedup = (α × BW_Speedup) + ((1-α) × Compute_Speedup × MFU_adj)

Where:
  α ≈ 0.30 (memory-bound fraction)
  BW_Speedup = min(BW_new / BW_T4, cap)
  Compute_Speedup = TFLOPS_new / TFLOPS_T4
  MFU_adj ≈ 0.70 (accounts for kernel overhead)
```

**Example: L4 vs T4**
```
Memory component: 0.30 × (300/320) = 0.30 × 0.94 = 0.28
Compute component: 0.70 × (242/65) × 0.70 = 0.70 × 3.72 × 0.70 = 1.82
Total: 0.28 + 1.82 = 2.10× → rounded to 2.07× (conservative)
```

**Critical Observation**: L4 has 3.72× more compute but 6% LESS memory bandwidth than T4. This significantly limits effective speedup for memory-bound operations.

### Step 3: Multi-GPU Scaling

| Scaling | Efficiency | Rationale |
|---------|------------|-----------|
| 2→4 GPUs | ~90% | Standard DDP overhead |
| 4→8 GPUs | ~85% | Gradient sync overhead |
| With NVLink | +5-10% | Faster GPU-to-GPU comm |

### Recommended Speedup Factors (Used in Estimates)

| GPU Config | vs 4×T4 Speedup | Confidence | Rationale |
|------------|-----------------|------------|-----------|
| **4×T4** | 1.00× | Measured | Baseline from actual training |
| **4×L4** | 2.07× | High | BW-limited for embeddings |
| **8×L4** | 4.00× | Medium | 2× GPUs with 90% scaling |
| **2×A100** | 2.70× | Medium | Fewer GPUs, high BW |
| **4×A100** | 5.25× | Medium-High | Strong BW + NVLink |
| **8×A100** | 9.50× | Medium | Near-linear with NVLink |
| **8×H100** | 9.50× | **Low** | See uncertainty note below |

### ⚠️ H100 Uncertainty Note

The H100 has 30× more compute than T4, but we estimate only 9.5× speedup because:
1. **Model size (35M params)** doesn't saturate H100 compute
2. **Data loading** becomes a bottleneck at high throughput
3. **No benchmarks** available for this specific architecture on H100

**Actual speedup could be 10-15× or higher** with optimized code. The 9.5× estimate is conservative and matches 8×A100 as a lower bound.

---

## Appendix A: Sparsity Explained

The "*" in NVIDIA specs indicates performance with **2:4 structured sparsity**.

**What is 2:4 Sparsity?**
- In every group of 4 weights, 2 must be zero
- Achieves 2× theoretical speedup with ~50% model compression
- Requires special training techniques (sparse training)

**Should You Use It?**
- **No** for standard training (your current setup)
- Consider for fine-tuning or inference optimization
- Use non-sparsity numbers for conservative estimates

---

## Appendix B: GCP Instance Mapping

| GPU | GCP Instance Family | Interconnect |
|-----|---------------------|--------------|
| T4 | N1 (n1-standard + GPU) | PCIe Gen3 |
| L4 | G2 (g2-standard) | PCIe Gen4 |
| A100 | A2 (a2-highgpu/megagpu) | NVLink (SXM) |
| H100 | A3 (a3-highgpu) | NVLink 4.0 |

---

## Appendix C: References

1. **NVIDIA T4**: https://www.nvidia.com/en-us/data-center/tesla-t4/
2. **NVIDIA L4**: https://www.nvidia.com/en-us/data-center/l4/
3. **NVIDIA A100**: https://www.nvidia.com/en-us/data-center/a100/
4. **NVIDIA H100**: https://www.nvidia.com/en-us/data-center/h100/
5. **GCP GPU Pricing**: https://cloud.google.com/compute/gpus-pricing
6. **Chinchilla Scaling Laws**: Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models"

---

*Last Updated: January 27, 2026*

