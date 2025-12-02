## **COMPLETE METRICS DICTIONARY**

### **1. Real-Time Batch Metrics** (`compute_batch_metrics_lightweight`, lines 3709-3807)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'recall@10'` | At least 1 true code in top-10 predictions | >0.20 | <0.10 | Primary clinical metric - shows if model provides useful suggestions |
| `'mAP@20'` | Mean Average Precision at 20 | >0.15 | <0.08 | Ranking quality - higher means true codes ranked earlier |
| `'brier_score'` | Mean squared probability error | <0.15 | >0.25 | Calibration quality - lower means better probability estimates |

---

### **2. Primary Task Metrics** (`compute_primary_task_metrics`, lines 4033-4112)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'recall@1'` | Any true code in top-1 | >0.05 | <0.02 | Strictest metric - first prediction correct |
| `'recall@5'` | Any true code in top-5 | >0.12 | <0.06 | Common clinical display size |
| `'recall@10'` | Any true code in top-10 | >0.20 | <0.10 | **PRIMARY METRIC** - standard clinical workflow |
| `'recall@20'` | Any true code in top-20 | >0.30 | <0.15 | Extended suggestions |
| `'recall@50'` | Any true code in top-50 | >0.45 | <0.25 | Maximum clinical utility |
| `'precision@1'` | Fraction of top-1 that are correct | >0.05 | <0.02 | Strictest precision |
| `'precision@5'` | Fraction of top-5 that are correct | >0.03 | <0.01 | False positive control |
| `'precision@10'` | Fraction of top-10 that are correct | >0.02 | <0.01 | Alert fatigue prevention |
| `'precision@20'` | Fraction of top-20 that are correct | >0.015 | <0.008 | Extended precision |
| `'precision@50'` | Fraction of top-50 that are correct | >0.01 | <0.005 | Maximum precision |
| `'mrr'` | Mean Reciprocal Rank | >0.10 | <0.05 | Average position of first true code |
| `'f1@1'` | F1 score at K=1 | >0.05 | <0.02 | Harmonic mean of P@1 and R@1 |
| `'f1@5'` | F1 score at K=5 | >0.05 | <0.02 | Harmonic mean of P@5 and R@5 |
| `'f1@10'` | F1 score at K=10 | >0.04 | <0.015 | Harmonic mean of P@10 and R@10 |
| `'f1@20'` | F1 score at K=20 | >0.03 | <0.01 | Harmonic mean of P@20 and R@20 |
| `'f1@50'` | F1 score at K=50 | >0.02 | <0.01 | Harmonic mean of P@50 and R@50 |

---

### **3. Loss & Calibration Metrics** (`compute_loss_metrics`, lines 4121-4187)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'bce_loss'` | Binary Cross-Entropy Loss (total) | <0.05 | >0.10 | Total multi-label classification loss |
| `'bce_loss_mean'` | Per-sample BCE loss (mean) | <0.05 | >0.10 | Average loss per sample |
| `'bce_loss_std'` | Per-sample BCE loss (std dev) | <0.03 | >0.08 | Loss variance across samples |
| `'ece'` | Expected Calibration Error | <0.10 | >0.20 | Probability calibration quality |
| `'brier_score'` | Brier Score (MSE of probabilities) | <0.15 | >0.25 | Probability accuracy (lower = better) |

---

### **4. Stratified Performance Metrics** (`compute_stratified_metrics`, lines 4197-4275)

**CRITICAL FOR CLINICAL AI** - Shows performance across code frequency tiers

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'common_top10_acc'` | Recall@10 for top 20% frequent codes | >0.45 | <0.30 | Performance on common diagnoses (easy) |
| `'medium_top10_acc'` | Recall@10 for 20-50% percentile codes | >0.25 | <0.15 | Performance on medium-frequency codes |
| `'rare_top10_acc'` | Recall@10 for 50-80% percentile codes | >0.12 | <0.06 | Performance on rare codes (hard) |
| `'tail_top10_acc'` | Recall@10 for bottom 20% rare codes | >0.08 | <0.03 | **CRITICAL** - Performance on very rare codes |
| `'tail_code_coverage'` | % of rare codes ever in top-50 | >0.35 | <0.20 | Does model predict rare codes at all? |
| `'balanced_top10_acc'` | Mean of 4 tier accuracies | >0.20 | <0.12 | Frequency-weighted overall performance |

**Interpretation Guide**:
- **If `common_top10_acc` >> `tail_top10_acc`**: Model memorizing frequent codes only ⚠️
- **If `tail_code_coverage` < 0.25**: Model ignoring rare codes entirely 🔴
- **If `balanced_top10_acc` ≈ `recall@10`**: Good balance across frequencies ✅

---

### **5. Convergence Metrics** (`compute_convergence_metrics`, lines 4346-4408)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'epochs_to_converge'` | Epochs to reach 95% of final loss | <5 | >15 | Training speed - lower = faster learning |
| `'convergence_rate'` | 1 / epochs_to_converge | >0.15 | <0.05 | Inverse of convergence time |
| `'loss_variance'` | Variance of smoothed loss curve | <0.005 | >0.02 | Training stability (lower = more stable) |
| `'loss_stability'` | 1 / (1 + loss_variance) | >0.98 | <0.95 | Normalized stability score |
| `'num_loss_spikes'` | Number of loss increases | <3 | >8 | Training instability indicator |
| `'best_epoch'` | Epoch with lowest val loss | N/A | N/A | When to early-stop |
| `'best_val_loss'` | Minimum validation loss | <0.04 | >0.08 | Best model performance |
| `'overfitting_gap'` | Final loss - best loss | <0.005 | >0.02 | How much overfitting occurred |
| `'auc_learning_curve'` | Area under learning curve | <5.0 | >20.0 | Learning efficiency (lower = faster) |

---

### **6. MoE Batch Metrics** (`compute_moe_batch_metrics`, lines 3973-4018)

**Only present for experiments 3-6**

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'expert_load_cv'` | Coefficient of variation of expert usage | <0.35 | >0.60 | Load balance quality (lower = better balance) |
| `'num_collapsed_experts'` | Experts with <5% usage | 0-1 | >2 | Routing failure indicator |
| `'expert_gini'` | Gini coefficient of expert usage | 0.20-0.40 | >0.60 | Inequality measure (0=perfect equality) |
| `'aux_loss'` | Auxiliary load balance loss | <0.01 | >0.05 | Switch Transformer balancing penalty |

**Critical Interpretation**:
- **If `num_collapsed_experts` > 2**: Routing has failed - some experts never used 🔴
- **If `expert_load_cv` > 0.6**: Severe load imbalance - one expert doing all work 🔴
- **If `expert_gini` < 0.15**: Too uniform - experts not specializing ⚠️
- **If `expert_gini` > 0.65**: Too peaked - routing collapsed 🔴

---

### **7. MoE Performance Metrics** (`compute_moe_performance_metrics`, lines 4417-4482)

**End-of-epoch comprehensive MoE analysis**

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'expert_load_mean'` | Average expert usage | ~0.125 (1/8) | N/A | Should be uniform ≈ 1/num_experts |
| `'expert_load_std'` | Std dev of expert usage | <0.04 | >0.08 | Usage variance |
| `'expert_load_cv'` | CV of expert usage | <0.35 | >0.60 | Same as batch metric |
| `'expert_load_min'` | Minimum expert usage | >0.03 | <0.01 | Least-used expert (should not be zero) |
| `'expert_load_max'` | Maximum expert usage | <0.25 | >0.40 | Most-used expert (should not dominate) |
| `'load_balance_score'` | 1 - CV | >0.65 | <0.40 | Overall balance quality |
| `'routing_entropy'` | Shannon entropy of routing | 1.5-2.0 | <1.0 | Routing diversity (max=log(8)≈2.08) |
| `'routing_entropy_normalized'` | Entropy / max_entropy | 0.65-0.85 | <0.50 | Normalized routing diversity |
| `'specialization_score'` | 1 - normalized_entropy | 0.15-0.35 | >0.50 | Expert specialization level |
| `'num_collapsed_experts'` | Experts with <5% usage | 0-1 | >2 | Same as batch metric |
| `'expert_collapse'` | Boolean: any expert collapsed | False | True | Binary collapse indicator |
| `'effective_experts'` | num_experts - collapsed | 7-8 | <6 | Actually utilized capacity |
| `'expert_gini'` | Gini coefficient | 0.20-0.40 | >0.60 | Same as batch metric |

**Interpretation Rules**:
- **Healthy MoE**: `load_balance_score` > 0.65, `effective_experts` = 7-8, `routing_entropy_normalized` = 0.65-0.85
- **Collapsed MoE**: `num_collapsed_experts` > 2, `expert_load_max` > 0.5
- **Over-specialized**: `routing_entropy_normalized` < 0.5
- **Under-specialized**: `specialization_score` < 0.1

---

### **8. Training Time Metrics** (`compute_training_time_metrics`, lines 4284-4337)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'total_train_time_sec'` | Total training time (seconds) | N/A | N/A | Wall-clock time |
| `'time_per_epoch_sec'` | Seconds per epoch | N/A | N/A | Epoch duration |
| `'time_per_sample_ms'` | Milliseconds per sample | <100ms | >300ms | Sample processing speed |
| `'samples_per_sec'` | Training throughput | >20 | <5 | **KEY EFFICIENCY METRIC** |
| `'tokens_per_sec'` | Tokens processed per second | >4000 | <1000 | LLM-style throughput metric |
| `'batches_per_sec'` | Batches per second | >1.0 | <0.3 | Batch processing rate |
| `'data_load_percent'` | % time loading data | <25% | >50% | I/O bottleneck indicator |
| `'forward_percent'` | % time in forward pass | 30-45% | >60% | Compute distribution |
| `'backward_percent'` | % time in backward pass | 30-45% | >60% | Gradient computation time |
| `'steps_per_sec'` | Training steps per second | >1.0 | <0.3 | Standard industry metric |

**Expected Speedups**:
- Exp2 vs Exp1: `samples_per_sec` should be 2-3× higher
- Exp2b vs Exp2: Additional 1.2-1.5× from learned pooling
- Exp3-6 vs Exp2: Similar or 0.8× (MoE overhead)

---

### **9. Memory Metrics** (`compute_memory_metrics`, lines 4491-4570)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'model_params'` | Total parameter count | N/A | N/A | Model size |
| `'model_memory_gb'` | Parameter memory (GB) | <2.0 | >5.0 | Static model size |
| `'gpu{i}_allocated_gb'` | Allocated memory on GPU i | <12GB | >14GB | Per-GPU current usage (T4 has 16GB) |
| `'gpu{i}_reserved_gb'` | Reserved memory on GPU i | <14GB | >15GB | Per-GPU reserved (includes cache) |
| `'gpu{i}_peak_gb'` | Peak memory on GPU i | <13GB | >15GB | Maximum usage during training |
| `'total_allocated_gb'` | Total allocated across GPUs | <48GB | >56GB | Aggregate allocation |
| `'total_reserved_gb'` | Total reserved across GPUs | <56GB | >60GB | Aggregate reservation |
| `'total_peak_gb'` | Total peak across GPUs | <52GB | >60GB | **KEY MEMORY METRIC** |
| `'avg_allocated_per_gpu_gb'` | Average allocated per GPU | <12GB | >14GB | Load balance check |
| `'avg_peak_per_gpu_gb'` | Average peak per GPU | <13GB | >15GB | Peak load balance |
| `'activation_memory_gb'` | Memory for activations | <10GB | >12GB | Temporary tensors |
| `'memory_per_sample_mb'` | MB per sample | <4MB | >8MB | Batch size planning |
| `'param_memory_percent'` | % memory for parameters | 15-30% | >50% | Memory breakdown |
| `'activation_memory_percent'` | % memory for activations | 60-80% | >85% | Compute vs storage |
| `'max_batch_size_theoretical'` | Theoretical max batch size | >32 | <16 | Safe batch size limit |

**Critical Thresholds (T4 GPU, 16GB)**:
- **Safe**: `total_peak_gb` < 13GB per GPU
- **Warning**: 13-15GB (may OOM with spikes)
- **Critical**: >15GB (will OOM) 🔴

---

### **10. FLOPs Metrics** (`compute_flops_metrics`, lines 4578-4700)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'forward_flops'` | FLOPs for forward pass | N/A | N/A | Theoretical compute |
| `'total_flops_per_sample'` | FLOPs per sample (fwd+bwd) | N/A | N/A | Total compute per sample |
| `'total_flops_per_batch'` | FLOPs per batch | N/A | N/A | Batch compute requirement |
| `'achieved_tflops'` | Achieved TFLOPs/sec | >1.0 | <0.5 | Actual compute throughput |
| `'mfu'` | Model FLOPs Utilization (fraction) | >0.15 | <0.08 | Hardware efficiency |
| `'mfu_percent'` | MFU as percentage | >15% | <8% | **KEY EFFICIENCY METRIC** |
| `'flops_per_param'` | FLOPs per parameter | N/A | N/A | Architectural efficiency |
| `'moe_compute_efficiency'` | Fraction of experts activated | ~0.25 | N/A | For MoE: top_k/num_experts |
| `'moe_param_efficiency'` | Active params / total params | ~0.25 | N/A | MoE parameter usage |

**Expected MFU (Model FLOPs Utilization)**:
- Exp1 (Baseline): 5-10% (standard attention is inefficient)
- Exp2 (Flash): 15-25% (kernel fusion helps)
- Exp3-6 (MoE+Flash): 20-30% (conditional compute + optimization)

**MoE Compute Savings**:
- Standard model: 100% of FLOPs
- MoE (top-2 of 8): ~25% of FLOPs (4× reduction in theory)
- Actual speedup: 1.5-2× (routing overhead)

---

### **11. Cost Metrics** (`compute_cost_metrics`, lines 4709-4781)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'cost_usd'` | Total cost for this run | N/A | N/A | Actual experiment cost |
| `'cost_per_epoch_usd'` | Cost per epoch | N/A | N/A | Epoch cost |
| `'projected_cost_{N}epochs_usd'` | Projected cost for N epochs | N/A | N/A | Scaling estimates (N=10,50,100,200) |
| `'cost_per_1k_samples_usd'` | Cost per 1000 samples | <$0.50 | >$2.00 | Sample processing cost |
| `'effective_cost_usd'` | Cost if GPU was 100% utilized | N/A | N/A | Wasted compute indicator |
| `'wasted_compute_usd'` | Money wasted on idle GPU | <20% | >50% | Efficiency loss in dollars |
| `'gpu_type'` | GPU type used | N/A | N/A | Hardware identifier |
| `'num_gpus'` | Number of GPUs | N/A | N/A | Parallelism level |
| `'hourly_rate_usd'` | Total hourly cost | N/A | N/A | Cost rate |

**For 1M samples, 10 epochs, 4×T4 GPUs (~$3.50/hr)**:
- Exp1: ~$500 (150 hours)
- Exp2: ~$250 (70 hours) - **50% cost reduction**
- Exp3-6: ~$300 (90 hours)

---

### **12. Ablation Metrics** (`compute_ablation_metrics`, lines 4790-4890)

**Cross-experiment comparisons**

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'flash_attn_acc_gain'` | Accuracy improvement from Flash | >0.02 | <0.00 | Absolute gain |
| `'flash_attn_acc_gain_percent'` | % accuracy improvement | >8% | <2% | Relative gain |
| `'flash_attn_speedup'` | Training speedup from Flash | >2.0× | <1.5× | Time reduction |
| `'learned_pool_acc_gain'` | Accuracy from learned pooling | >0.01 | <0.00 | Pooling benefit |
| `'learned_pool_speedup'` | Speedup from learned pooling | >1.2× | <1.05× | Additional speedup |
| `'moe_acc_gain'` | Accuracy from MoE | >0.03 | <0.01 | MoE benefit |
| `'moe_acc_per_param'` | Acc gain per M parameters added | >0.01 | <0.003 | Parameter efficiency |
| `'pool_moe_synergy_speedup'` | Pooling speedup with MoE | >1.3× | <1.1× | Synergy effect |
| `'pool_moe_interaction'` | Difference in pooling benefit | >0.1× | <0.0× | Interaction strength |
| `'{exp}_acc_per_dollar'` | Accuracy gain per dollar spent | >0.0001 | <0.00003 | Cost-effectiveness |
| `'{exp}_speedup_ratio'` | Speedup vs baseline | >1.5× | <1.0× | Relative speed |

**Key Ablations to Check**:
1. **Flash Attention**: Exp2 vs Exp1 → expect 2-3× speedup
2. **Learned Pooling**: Exp2b vs Exp2 → expect +0.01 accuracy, +20% speed
3. **MoE**: Exp3 vs Exp2 → expect +0.03 accuracy (especially tail codes)
4. **Shared Expert**: Exp4 vs Exp3 → expect better load balance
5. **Fine-Grained**: Exp5 vs Exp4 → expect best tail accuracy
6. **Auxiliary-Free**: Exp6 vs Exp3 → expect best load balance

---

### **13. Embedding Quality Metrics** (`compute_embedding_quality_epoch`, lines 3810-3970)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'embedding_std_mean'` | Mean std dev across dimensions | >0.05 | <0.01 | **COLLAPSE DETECTOR** - low = embeddings collapsed |
| `'nn_target_overlap'` | Jaccard similarity of NN targets | >0.20 | <0.10 | Semantic quality - high = embeddings meaningful |

**Critical Warning Signs**:
- **If `embedding_std_mean` < 0.01**: Embeddings collapsed to single point 🔴🔴🔴
  - **Action**: Reduce learning rate, add embedding normalization
- **If `nn_target_overlap` < 0.10**: Embeddings not learning relationships ⚠️
  - **Action**: Check if model is training, increase embedding size

---

## **SUMMARY: Top 10 Most Important Metrics**

| Priority | Metric Key | Category | Critical Threshold | Why It Matters |
|----------|------------|----------|-------------------|----------------|
| 1️⃣ | `'tail_top10_acc'` | Performance | >0.08 | **Rare disease detection** - clinical value |
| 2️⃣ | `'recall@10'` | Performance | >0.20 | **Primary clinical metric** - overall utility |
| 3️⃣ | `'num_collapsed_experts'` | MoE Health | ≤1 | **MoE success** - routing failure detector |
| 4️⃣ | `'samples_per_sec'` | Efficiency | >20 | **Training speed** - cost & time |
| 5️⃣ | `'embedding_std_mean'` | Quality | >0.05 | **Collapse detector** - embeddings viable |
| 6️⃣ | `'balanced_top10_acc'` | Performance | >0.20 | **Frequency-fair** - not just common codes |
| 7️⃣ | `'expert_load_cv'` | MoE Health | <0.35 | **Load balance** - expert utilization |
| 8️⃣ | `'brier_score'` | Calibration | <0.15 | **Probability quality** - confidence matters |
| 9️⃣ | `'total_peak_gb'` | Resources | <13GB | **Memory safety** - OOM prevention |
| 🔟 | `'mfu_percent'` | Efficiency | >15% | **Hardware efficiency** - GPU utilization |

This exhaustive list covers **ALL 100+ metrics** used in your evaluation framework!