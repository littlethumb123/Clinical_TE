# Training Cost & Time Summary - Quick Reference

**Document Version**: January 27, 2026  
**Model**: Hierarchical Clinical Transformer (BEHRT-style)  
**Target**: 15M members, 4 epochs, 96-hour flexibility buffer

---

## Hardware Specifications (Official NVIDIA)

| GPU | FP16 TFLOPS | Memory | Bandwidth | TDP | GCP Hourly |
|-----|-------------|--------|-----------|-----|------------|
| **T4** | 65 | 16 GB GDDR6 | 320 GB/s | 70W | $2.99 (4×) |
| **L4** | 242 | 24 GB GDDR6 | 300 GB/s | 72W | $4.00 (4×) |
| **A100** | 312 | 80 GB HBM2e | 1,935 GB/s | 300W | $14.69 (4×) |
| **H100** | 1,979 | 80 GB HBM3 | 3,350 GB/s | 700W | $88.49 (8×) |

**Source**: NVIDIA Official Specifications (user-provided)  
**Pricing**: GCP us-central1 On-Demand (user-provided)

---

## Training Time Summary (Excluding 96h Buffer)

| Configuration | exp1_dense | exp2b_flash | exp6_moe |
|---------------|------------|-------------|----------|
| **4×T4** | 424.8h (17.7d) | 193.1h (8.0d) | 217.8h (9.1d) |
| **4×L4** ⭐ | 205.2h (8.6d) | 93.3h (3.9d) | 105.2h (4.4d) |
| **8×L4** | 106.2h (4.4d) | 48.3h (2.0d) | 54.5h (2.3d) |
| **4×A100** | 80.9h (3.4d) | 36.8h (1.5d) | 41.5h (1.7d) |
| **8×A100** | 44.7h (1.9d) | 20.3h (0.8d) | 22.9h (1.0d) |
| **8×H100** | 44.7h (1.9d) | 20.3h (0.8d) | 22.9h (1.0d) |

## Total Time Including Buffer

| Configuration | exp1_dense | exp2b_flash | exp6_moe |
|---------------|------------|-------------|----------|
| **4×T4** | 520.8h (21.7d) | 289.1h (12.0d) | 313.8h (13.1d) |
| **4×L4** ⭐ | 301.2h (12.6d) | 189.3h (7.9d) | 201.2h (8.4d) |
| **8×L4** | 202.2h (8.4d) | 144.3h (6.0d) | 150.5h (6.3d) |
| **4×A100** | 176.9h (7.4d) | 132.8h (5.5d) | 137.5h (5.7d) |
| **8×A100** | 140.7h (5.9d) | 116.3h (4.8d) | 118.9h (5.0d) |

---

## Cost Summary (Including 96h Buffer)

| Configuration | exp1_dense | exp2b_flash | exp6_moe | Speedup |
|---------------|------------|-------------|----------|---------|
| **4×T4** | $1,557 | $864 | $938 | 1.0× |
| **4×L4** ⭐ | $1,205 | **$757** | **$805** | 2.07× |
| **8×L4** | $1,618 | $1,154 | $1,204 | 4.0× |
| **2×A100** | $1,859 | $1,230 | $1,297 | 2.7× |
| **4×A100** | $2,600 | $1,951 | $2,020 | 5.25× |
| **8×A100** | $4,134 | $3,417 | $3,493 | 9.5× |
| **8×H100** | $12,451 | $10,291 | $10,521 | 9.5× |

---

## Model Specifications

| Experiment | Parameters | Baseline (4×T4) | Peak Memory |
|------------|------------|-----------------|-------------|
| **exp1_dense** | 26.46M | 12.04h (0.5d) | 20.6 GB |
| **exp2b_flash** ⭐ | 25.33M | 5.47h (0.2d) | 11.1 GB |
| **exp6_moe** | 35.42M | 6.17h (0.3d) | 13.4 GB |

*Baseline = 1.7M members, 1 epoch on 4×T4*

---

## Key Recommendations

1. ✅ **Use 4×L4 for best value**: 12% cheaper than 4×T4, 2.07× faster
2. ✅ **Use exp2b_flash_learned_pool**: Fastest training, lowest cost
3. ✅ **8×L4 if time-constrained**: 48% faster for 52% more cost
4. ⚠️ **A100/H100**: Higher cost, diminishing returns for this model size

---

## Decision Matrix

```
Budget < $1,000?
  └─ 4×L4 + exp2b ($757) → 93.3h (3.9d) training, 189.3h (7.9d) total

Need results in < 1 week?
  └─ 8×L4 ($1,154) → 48.3h (2.0d) training, 144.3h (6.0d) total

Running exp1_dense (no Flash)?
  └─ 4×L4 ($1,205) → still 23% cheaper than 4×T4

Have H100 credits?
  └─ 8×H100 has same training time as 8×A100 but 3× cost
```

---

## Speedup Rationale

| GPU | FP16 TFLOPS | Memory BW | Effective Speedup |
|-----|-------------|-----------|-------------------|
| T4 | 65 | 320 GB/s | 1.0× (baseline) |
| L4 | 242 (3.72×) | 300 GB/s (0.94×) | **2.07×** |
| A100 | 312 (4.80×) | 1,935 GB/s (6.05×) | **5.25× (4 GPU)** |
| H100 | 1,979 (30×) | 3,350 GB/s (10×) | **9.5× (8 GPU)** |

**Why L4 is 2.07× not 3.72×?**
- L4 has 6% LOWER memory bandwidth than T4
- ~30% of transformer compute is memory-bound (embeddings)
- Weighted speedup: 0.30×0.94 + 0.70×3.72×0.70 = 2.07×

---

## Uncertainty Notes

| Item | Confidence | Note |
|------|------------|------|
| Baseline times | ✓ Measured | From actual exp_round5 results |
| L4 speedup (2.07×) | High | Similar architecture to T4 |
| A100 speedup (5.25×) | Medium | Based on NVLink efficiency |
| H100 speedup (9.5×) | **Low** | Model may not saturate H100 |

---

**For full calculations**: See `training_cost_estimates_15M_4epochs.md`  
**For GPU specs**: See `gpu_specifications_reference.md`  
**For methodology**: See `training_sample_time_cost_estimate.md`
