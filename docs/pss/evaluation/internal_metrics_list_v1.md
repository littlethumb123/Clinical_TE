## **COMPLETE METRICS DICTIONARY**

### **1. Real-Time Batch Metrics** (`compute_batch_metrics_lightweight`, lines 6365-6517)

Lightweight metrics computed every 100 batches for real-time training monitoring. These are fast approximations that complement the loss signal during training. Full comprehensive metrics are computed at epoch end via `evaluate()`.

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'recall@1'` | Was the #1 prediction a true code? | >0.05 | <0.02 | **Strictest test** - Only considers the single highest-confidence prediction. Example: If model predicts code 45 with highest probability and patient has codes [45, 102, 78], recall@1 = 1.0 (hit). If model predicts code 99 first, recall@1 = 0.0 (miss). |
| `'recall@5'` | Any true code in top-5 predictions? | >0.12 | <0.06 | **Common clinical display** - Shows if model's top 5 suggestions include at least one correct diagnosis. Example: Patient has codes [102, 78]. Model predicts [45, 102, 33, 89, 12] → hit (102 found). |
| `'recall@10'` | Any true code in top-10 predictions? | >0.20 | <0.10 | **PRIMARY METRIC** - Standard clinical workflow metric. Represents typical physician review window. |
| `'recall@20'` | Any true code in top-20 predictions? | >0.30 | <0.15 | **Extended suggestions** - Useful for complex cases requiring broader differential diagnosis. |
| `'recall@50'` | Any true code in top-50 predictions? | >0.45 | <0.25 | **Maximum clinical utility** - Upper bound of practical suggestion list. |
| `'precision@5'` | Fraction of top-5 that are correct | >0.03 | <0.01 | **False positive control** - Of 5 predictions, how many are actual patient codes? Example: 2 codes predicted correctly out of 5 → precision@5 = 0.4. |
| `'precision@10'` | Fraction of top-10 that are correct | >0.02 | <0.01 | **Alert fatigue prevention** - High precision reduces clinician review burden. |
| `'precision@20'` | Fraction of top-20 that are correct | >0.015 | <0.008 | Extended precision measurement for larger suggestion lists. |
| `'precision@50'` | Fraction of top-50 that are correct | >0.01 | <0.005 | Maximum precision at full suggestion depth. |
| `'micro_recall@1'` | Per-code hit rate at K=1 | >0.02 | <0.01 | **Granular recall** - Total codes found / Total true codes across all samples. Unlike sample-level recall (binary per sample), this counts individual code hits. Example: 100 samples with 300 total true codes, model finds 50 → micro_recall = 0.167. |
| `'micro_recall@10'` | Per-code hit rate at K=10 | >0.08 | <0.03 | Micro-averaged recall capturing what fraction of ALL true codes are found. |
| `'micro_recall@20'` | Per-code hit rate at K=20 | >0.12 | <0.05 | Extended micro-recall for comprehensive code coverage analysis. |
| `'ndcg@20'` | Normalized Discounted Cumulative Gain | >0.25 | <0.12 | **Ranking quality with position discount** - Higher scores for correct codes ranked earlier. NDCG=1.0 means all true codes are at the very top. Example: True codes at positions [1, 5, 15] scores higher than [3, 12, 18]. |
| `'positive_brier'` | Calibration on positive labels only | <0.30 | >0.50 | **Probability calibration for true codes** - Measures how confident the model is on actual diagnoses. Lower = better. Computed as mean((predicted_prob - 1.0)²) for positive labels only. Example: If true code has prob=0.8, contribution = (0.8-1.0)² = 0.04. |

**⚠️ NOT USED in current implementation:**
| Key Name | Status | Notes |
|----------|--------|-------|
| `'mAP@20'` | ❌ NOT USED | Mean Average Precision not implemented. Use `ndcg@20` for ranking quality instead. |
| `'brier_score'` | ❌ RENAMED | Now called `positive_brier` - computed only on positive labels (not dominated by true negatives). |

---

### **2. Primary Task Metrics** (`compute_primary_task_metrics`, lines 7250-7336)

Comprehensive task performance metrics computed at epoch end during validation. These form the primary evaluation criteria for model quality and are used in BEHRT, Med-BERT, and ClinicalBERT papers.

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'recall@1'` | Any true code in top-1 | >0.05 | <0.02 | **Strictest metric** - First prediction must be correct. Rarely achievable with multi-label targets (average 3+ codes per sample). |
| `'recall@5'` | Any true code in top-5 | >0.12 | <0.06 | **Common clinical display** - Standard physician-facing suggestion list size. |
| `'recall@10'` | Any true code in top-10 | >0.20 | <0.10 | **PRIMARY METRIC** - Standard clinical workflow. Example: Patient with diabetes (E11.9) and hypertension (I10) → hit if either appears in top 10. |
| `'recall@20'` | Any true code in top-20 | >0.30 | <0.15 | **Extended suggestions** - Comprehensive differential diagnosis support. |
| `'recall@50'` | Any true code in top-50 | >0.45 | <0.25 | **Maximum clinical utility** - Full breadth of possible diagnoses. |
| `'precision@1'` | Fraction of top-1 that are correct | >0.05 | <0.02 | **Strictest precision** - Is the #1 prediction actually correct? |
| `'precision@5'` | Fraction of top-5 that are correct | >0.03 | <0.01 | **False positive control** - Balance between recall and alert fatigue. Example: 2/5 predictions correct → 0.40. |
| `'precision@10'` | Fraction of top-10 that are correct | >0.02 | <0.01 | **Alert fatigue prevention** - Clinical systems with too many false positives get ignored. |
| `'precision@20'` | Fraction of top-20 that are correct | >0.015 | <0.008 | Extended precision for larger suggestion windows. |
| `'precision@50'` | Fraction of top-50 that are correct | >0.01 | <0.005 | Maximum precision at full suggestion depth. |
| `'mrr'` | Mean Reciprocal Rank | >0.10 | <0.05 | **Average position of first correct code** - MRR=0.5 means first correct code is typically at rank 2. Formula: mean(1/rank_of_first_hit). Example: First hits at ranks [1, 3, 10] → MRR = (1 + 0.33 + 0.1)/3 = 0.477. |
| `'f1@1'` | F1 score at K=1 | >0.05 | <0.02 | **Harmonic mean of P@1 and R@1** - Balances precision and recall at strictest threshold. |
| `'f1@5'` | F1 score at K=5 | >0.05 | <0.02 | F1 at common clinical display size. |
| `'f1@10'` | F1 score at K=10 | >0.04 | <0.015 | **Balanced performance metric** - Good for comparing models when precision/recall tradeoffs differ. |
| `'f1@20'` | F1 score at K=20 | >0.03 | <0.01 | Extended F1 for larger suggestion lists. |
| `'f1@50'` | F1 score at K=50 | >0.02 | <0.01 | Maximum F1 at full suggestion depth. |
| `'micro_recall@5'` | Per-code hit rate at K=5 | >0.04 | <0.02 | **Granular recall** - Counts individual code hits, not just binary sample hits. Better reflects multi-label performance. |
| `'micro_recall@10'` | Per-code hit rate at K=10 | >0.08 | <0.04 | Micro-averaged recall at standard clinical window. |
| `'micro_recall@20'` | Per-code hit rate at K=20 | >0.12 | <0.06 | Extended micro-recall for comprehensive analysis. |
| `'micro_recall@50'` | Per-code hit rate at K=50 | >0.20 | <0.10 | Maximum micro-recall coverage. |
| `'ndcg@10'` | NDCG at K=10 | >0.20 | <0.10 | **Ranking quality** - Position-discounted gain. True codes ranked higher = better score. |
| `'ndcg@20'` | NDCG at K=20 | >0.25 | <0.12 | Extended NDCG for larger result sets. |
| `'ndcg@50'` | NDCG at K=50 | >0.30 | <0.15 | Maximum NDCG at full suggestion depth. |

**Micro-Recall vs Sample-Recall Explained:**
- **Sample Recall@10**: "Did ANY of this patient's codes appear in top-10?" → Binary (0 or 1) per sample
- **Micro-Recall@10**: "What fraction of ALL codes across ALL patients were found in top-10?" → Granular (0.0 to 1.0)
- Example: 2 patients, each with 3 codes. Patient A: 2/3 found. Patient B: 1/3 found.
  - Sample Recall@10 = (1 + 1) / 2 = 1.0 (both patients had at least 1 hit)
  - Micro-Recall@10 = (2 + 1) / 6 = 0.5 (3 of 6 total codes found)

---

### **3. Loss & Calibration Metrics** (`compute_loss_metrics`, lines 7345-7394)

Training loss and probability calibration metrics for understanding optimization dynamics.

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'bce_loss'` | Binary Cross-Entropy Loss (total) | <0.05 | >0.10 | **Primary optimization objective** - Total multi-label classification loss across all codes. Lower = better model fit. |
| `'bce_loss_mean'` | Per-sample BCE loss (mean) | <0.05 | >0.10 | **Average loss per sample** - Normalized loss for comparing different batch sizes. Example: BCE=2.5 with batch_size=50 → bce_loss_mean = 0.05. |
| `'bce_loss_std'` | Per-sample BCE loss (std dev) | <0.03 | >0.08 | **Loss variance across samples** - High std indicates model struggles with certain samples. May indicate difficult subpopulations or outliers. |
| `'positive_brier'` | Positive-Only Brier Score | <0.30 | >0.50 | **Calibration for positive labels** - Measures probability accuracy on actual diagnoses. Standard Brier is dominated by ~99.7% true negatives. This variant focuses only on true positives. Formula: mean((prob - 1)²) for positive labels. Example: True codes with probs [0.9, 0.7, 0.6] → positive_brier = mean([0.01, 0.09, 0.16]) = 0.087. |

**⚠️ NOT USED in current implementation:**
| Key Name | Status | Notes |
|----------|--------|-------|
| `'ece'` | ❌ NOT USED | Expected Calibration Error not implemented. Use `positive_brier` for calibration assessment instead. |
| `'brier_score'` | ❌ RENAMED | Now `positive_brier` - computed only on positive labels to avoid domination by true negatives. |

---

### **4. Stratified Performance Metrics** (`compute_stratified_metrics`, lines 7601-7694)

**CRITICAL FOR CLINICAL AI** - Shows performance across code frequency tiers. Medical codes follow extreme long-tail distributions where rare codes (sepsis, MI, rare diseases) are often most clinically important.

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'common_top10_acc'` | Recall@10 for top 20% frequent codes | >0.45 | <0.30 | **Performance on common diagnoses (easy)** - Codes like hypertension (I10), diabetes (E11.9). Example: If 80% of samples with common codes have at least 1 hit in top-10 → common_top10_acc = 0.80. |
| `'medium_top10_acc'` | Recall@10 for 20-50% percentile codes | >0.25 | <0.15 | **Performance on medium-frequency codes** - Moderately common conditions like asthma, GERD. |
| `'rare_top10_acc'` | Recall@10 for 50-80% percentile codes | >0.12 | <0.06 | **Performance on rare codes (hard)** - Less common conditions requiring good generalization. |
| `'tail_top10_acc'` | Recall@10 for bottom 20% rare codes | >0.08 | <0.03 | **CRITICAL - Performance on very rare codes** - Codes seen only a few times in training (rare diseases, unusual presentations). Models that fail here may miss critical diagnoses. |
| `'tail_code_coverage'` | % of rare codes ever in top-50 | >0.35 | <0.20 | **Does model predict rare codes at all?** - Measures whether the model has learned to output rare codes or ignores them entirely. Example: 100 tail codes, 40 ever appear in any patient's top-50 → coverage = 0.40. |
| `'balanced_top10_acc'` | Mean of 4 tier accuracies | >0.20 | <0.12 | **Frequency-weighted overall performance** - Equal weight to each tier prevents bias toward common codes. Formula: (common + medium + rare + tail) / 4. |

**Interpretation Guide with Examples:**
- **If `common_top10_acc` >> `tail_top10_acc`**: Model memorizing frequent codes only ⚠️
  - Example: common=0.65, tail=0.03 → Model is frequency-biased, predicting only what it's seen most often
- **If `tail_code_coverage` < 0.25**: Model ignoring rare codes entirely 🔴
  - Example: Only 15% of tail codes ever predicted → Model has collapsed to predicting only common codes
- **If `balanced_top10_acc` ≈ `recall@10`**: Good balance across frequencies ✅
  - Example: balanced=0.22, recall@10=0.24 → Model performs consistently across code frequencies

---

### **5. Convergence Metrics** (`compute_convergence_metrics`, lines 7696-7761)

Training dynamics and stability metrics for understanding learning progress.

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'epochs_to_converge'` | Epochs to reach 95% of final loss | <5 | >15 | **Training speed** - Lower = faster learning. Example: Final loss = 0.05, reached loss ≤ 0.0525 at epoch 3 → epochs_to_converge = 3. |
| `'convergence_rate'` | 1 / epochs_to_converge | >0.15 | <0.05 | **Inverse of convergence time** - Higher = faster convergence. Useful for comparing learning speed across experiments. |
| `'loss_variance'` | Variance of smoothed loss curve | <0.005 | >0.02 | **Training stability** - Lower = more stable training. High variance indicates noisy gradients or learning rate issues. |
| `'loss_stability'` | 1 / (1 + loss_variance) | >0.98 | <0.95 | **Normalized stability score** - Closer to 1.0 = more stable. Transforms variance into interpretable 0-1 scale. |
| `'num_loss_spikes'` | Number of loss increases | <3 | >8 | **Training instability indicator** - Counts epochs where val_loss increased vs previous epoch. Many spikes suggest learning rate too high or gradient explosions. |
| `'best_epoch'` | Epoch with lowest val loss | N/A | N/A | **Optimal early-stopping point** - Use this epoch's checkpoint for deployment. Example: best_epoch=7 with 10 epochs → epochs 8-10 were overfitting. |
| `'best_val_loss'` | Minimum validation loss | <0.04 | >0.08 | **Best model performance** - Lowest loss achieved during training. Use with best_epoch for model selection. |
| `'overfitting_gap'` | Final loss - best loss | <0.005 | >0.02 | **How much overfitting occurred** - Positive gap means model overfit after best_epoch. Example: final_loss=0.055, best_loss=0.045 → gap=0.01 (mild overfitting). |
| `'auc_learning_curve'` | Area under learning curve | <5.0 | >20.0 | **Learning efficiency** - Lower = faster convergence to optimal. Measures total "excess loss" during training. Computed as integral of (loss - final_loss) over epochs. |

---

### **6. MoE Batch Metrics** (`compute_moe_batch_metrics`, lines 6667-6719)

**Only present for experiments 3-6 (MoE variants)**

Real-time MoE health monitoring computed every batch during training.

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'expert_load_cv'` | Coefficient of variation of expert usage | <0.35 | >0.60 | **Load balance quality** - CV = std/mean. Lower = better balance. Example: 8 experts with usage [0.13, 0.12, 0.11, 0.14, 0.12, 0.12, 0.13, 0.13] → CV ≈ 0.08 (excellent). Usage [0.40, 0.02, 0.01, 0.02, 0.02, 0.50, 0.02, 0.01] → CV ≈ 1.2 (collapsed). |
| `'num_collapsed_experts'` | Experts with <5% usage | 0-1 | >2 | **Routing failure indicator** - Experts receiving <5% of tokens are effectively unused. Example: 8 experts, 2 with <5% → num_collapsed = 2 (routing failing). |
| `'expert_gini'` | Gini coefficient of expert usage | 0.20-0.40 | >0.60 | **Inequality measure** - 0 = perfect equality (all experts equal), 1 = total inequality (one expert gets everything). Sweet spot is 0.2-0.4 indicating moderate specialization. |
| `'aux_loss'` | Auxiliary load balance loss | <0.01 | >0.05 | **Switch Transformer balancing penalty** - Regularization term encouraging balanced routing. High aux_loss indicates load imbalance that the model is being penalized for. |

**Critical MoE Health Checks:**
- **If `num_collapsed_experts` > 2**: Routing has failed - some experts never used 🔴
  - Action: Increase aux_loss_weight, check router initialization, reduce learning rate
- **If `expert_load_cv` > 0.6**: Severe load imbalance - one expert doing all work 🔴
  - Action: The model has essentially become a single-expert dense model
- **If `expert_gini` < 0.15**: Too uniform - experts not specializing ⚠️
  - Action: May indicate router not learning; check gradient flow
- **If `expert_gini` > 0.65**: Too peaked - routing collapsed 🔴
  - Action: Same as collapsed experts - routing mechanism broken

---

### **7. MoE Performance Metrics** (`compute_moe_performance_metrics`, lines 6827-6894)

**End-of-epoch comprehensive MoE analysis**

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'expert_load_mean'` | Average expert usage | ~0.125 (1/8) | N/A | **Should be uniform ≈ 1/num_experts** - For 8 experts, ideal is 0.125. Example: mean=0.125 indicates perfect load balance on average. |
| `'expert_load_std'` | Std dev of expert usage | <0.04 | >0.08 | **Usage variance** - Low std means consistent expert usage across batches. |
| `'expert_load_cv'` | CV of expert usage | <0.35 | >0.60 | Same as batch metric - aggregated over epoch. |
| `'expert_load_min'` | Minimum expert usage | >0.03 | <0.01 | **Least-used expert** - Should not be near zero. Example: min=0.02 for 8 experts (target 0.125) indicates significant imbalance. |
| `'expert_load_max'` | Maximum expert usage | <0.25 | >0.40 | **Most-used expert** - Should not dominate. Example: max=0.45 means one expert handles 45% of tokens (others underutilized). |
| `'load_balance_score'` | 1 - CV | >0.65 | <0.40 | **Overall balance quality** - Higher = better. Transforms CV into intuitive 0-1 scale where 1.0 = perfect balance. |
| `'routing_entropy'` | Shannon entropy of routing | 1.5-2.0 | <1.0 | **Routing diversity** - Max for 8 experts = log(8) ≈ 2.08. Higher entropy = more diverse routing. Example: entropy=1.8 indicates good diversity; entropy=0.5 indicates collapsed routing. |
| `'routing_entropy_normalized'` | Entropy / max_entropy | 0.65-0.85 | <0.50 | **Normalized routing diversity** - 1.0 = perfectly uniform, 0.0 = single expert. Sweet spot 0.65-0.85 balances diversity with specialization. |
| `'specialization_score'` | 1 - normalized_entropy | 0.15-0.35 | >0.50 | **Expert specialization level** - Higher = more specialized experts. Too high (>0.5) indicates over-specialization/collapse. |
| `'num_collapsed_experts'` | Experts with <5% usage | 0-1 | >2 | Same as batch metric - aggregated over epoch. |
| `'expert_collapse'` | Boolean: any expert collapsed | False | True | **Binary collapse indicator** - Quick check for routing health. |
| `'effective_experts'` | num_experts - collapsed | 7-8 | <6 | **Actually utilized capacity** - For 8 experts, want 7-8 effective. Example: effective=5 means 3 experts are wasted capacity. |
| `'expert_gini'` | Gini coefficient | 0.20-0.40 | >0.60 | Same as batch metric - aggregated over epoch. |

**Interpretation Rules:**
- **Healthy MoE**: `load_balance_score` > 0.65, `effective_experts` = 7-8, `routing_entropy_normalized` = 0.65-0.85
- **Collapsed MoE**: `num_collapsed_experts` > 2, `expert_load_max` > 0.5
- **Over-specialized**: `routing_entropy_normalized` < 0.5
- **Under-specialized**: `specialization_score` < 0.1

---

### **8. Router Gradient Metrics** (`compute_router_gradient_metrics`, lines 6721-6821)

**MoE stability diagnostics** - Must be called AFTER backward() but BEFORE optimizer.step()!

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'router_grad_norm_mean'` | Mean L2 norm of router gradients | 0.1-5.0 | <1e-7 or >10.0 | **Gradient health indicator** - Gradients should be in moderate range for stable learning. |
| `'router_grad_norm_max'` | Max L2 norm | <10.0 | >10.0 | **Explosion detection** - Any gradient >10.0 indicates potential instability. |
| `'router_grad_norm_min'` | Min L2 norm | >1e-7 | <1e-7 | **Vanishing detection** - Near-zero gradients mean router not learning. |
| `'router_grad_norm_std'` | Std dev of gradient norms | <2.0 | >5.0 | **Gradient consistency** - High variance may indicate unstable optimization. |
| `'router_grad_exploding'` | 1 if any grad > 10.0, else 0 | 0 | 1 | **Binary explosion flag** - Quick check for gradient explosions. |
| `'router_grad_vanishing'` | 1 if any grad < 1e-7, else 0 | 0 | 1 | **Binary vanishing flag** - Quick check for vanishing gradients. |
| `'router_layers_total'` | Total router layers | N/A | N/A | **Architecture info** - Number of router/gate parameters tracked. |
| `'router_layers_healthy'` | Layers with healthy gradients | All | <Total | **Healthy layer count** - Layers with gradients in range [1e-7, 10.0]. |
| `'router_weight_mean'` | Mean of router weights | Near 0 | N/A | **Weight distribution center** - Should be near zero for balanced initialization. |
| `'router_weight_std'` | Std of router weights | >0.01 | <0.01 | **Weight variance** - Low std indicates collapsed/stuck routers. |
| `'router_weight_abs_max'` | Max absolute weight | <5.0 | >10.0 | **Weight magnitude** - Very large weights indicate instability. |
| `'router_weight_collapsed'` | 1 if weight_std < 0.01, else 0 | 0 | 1 | **Collapsed router flag** - Weights not differentiating means router stuck. |

**Warning Thresholds (Empirically Determined):**
- `grad_norm > 10.0`: Exploding gradients - reduce LR or add clipping
- `grad_norm < 1e-7`: Vanishing gradients - check focal loss/initialization
- `grad_std ≈ 0`: Router not learning - check gradient flow
- `weight_std < 0.01`: Router weights collapsed - reinitialize

---

### **9. Training Time Metrics** (`compute_training_time_metrics`, lines 7181-7237)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'total_train_time_sec'` | Total training time (seconds) | N/A | N/A | **Wall-clock time** - Total duration of training run. Example: 3600 sec = 1 hour. |
| `'time_per_epoch_sec'` | Seconds per epoch | N/A | N/A | **Epoch duration** - Useful for estimating full training time. Example: 360 sec/epoch × 50 epochs = 5 hours total. |
| `'time_per_sample_ms'` | Milliseconds per sample | <100ms | >300ms | **Sample processing speed** - End-to-end time including data loading, forward, backward, optimizer. |
| `'samples_per_sec'` | Training throughput | >20 | <5 | **KEY EFFICIENCY METRIC** - Primary throughput measure. Example: 50 samples/sec with batch_size=32 → ~1.6 batches/sec. |
| `'tokens_per_sec'` | Tokens processed per second | >4000 | <1000 | **LLM-style throughput** - Standard metric in transformer papers. tokens = samples × sequence_length. |
| `'batches_per_sec'` | Batches per second | >1.0 | <0.3 | **Batch processing rate** - samples_per_sec / batch_size. |
| `'data_load_percent'` | % time loading data | <25% | >50% | **I/O bottleneck indicator** - High percentage suggests data pipeline optimization needed (more workers, prefetching). |
| `'forward_percent'` | % time in forward pass | 30-45% | >60% | **Compute distribution** - Forward should be ~30-40% of compute for balanced training. |
| `'backward_percent'` | % time in backward pass | 30-45% | >60% | **Gradient computation time** - Backward typically slightly longer than forward (~1.5-2×). |
| `'steps_per_sec'` | Training steps per second | >1.0 | <0.3 | **Standard industry metric** - Same as batches_per_sec. Used for comparing with published baselines. |

**Expected Speedups:**
- Exp2 vs Exp1: `samples_per_sec` should be 2-3× higher (Flash Attention benefit)
- Exp2b vs Exp2: Additional 1.2-1.5× from learned pooling
- Exp3-6 vs Exp2: Similar or 0.8× (MoE adds routing overhead)

---

### **10. Memory Metrics** (`compute_memory_metrics`, lines 6897-6978)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'model_params'` | Total parameter count | N/A | N/A | **Model size** - Example: 10M params = 10,000,000 parameters. |
| `'model_memory_gb'` | Parameter memory (GB) | <2.0 | >5.0 | **Static model size** - FP32 memory: params × 4 bytes. Example: 10M params × 4 = 40MB = 0.04GB. |
| `'gpu{i}_allocated_gb'` | Allocated memory on GPU i | <12GB | >14GB | **Per-GPU current usage** - Memory actually allocated for tensors (T4 has 16GB total). |
| `'gpu{i}_reserved_gb'` | Reserved memory on GPU i | <14GB | >15GB | **Per-GPU reserved** - Includes PyTorch caching pool. May be higher than allocated. |
| `'gpu{i}_peak_gb'` | Peak memory on GPU i | <13GB | >15GB | **Maximum usage** - Highest allocation during training. Critical for OOM prevention. |
| `'total_allocated_gb'` | Total allocated across GPUs | <48GB | >56GB | **Aggregate allocation** - Sum of all GPU allocations for multi-GPU setup. |
| `'total_reserved_gb'` | Total reserved across GPUs | <56GB | >60GB | **Aggregate reservation** - Total memory reserved by PyTorch. |
| `'total_peak_gb'` | Total peak across GPUs | <52GB | >60GB | **KEY MEMORY METRIC** - Maximum total memory used. Critical for capacity planning. |
| `'avg_allocated_per_gpu_gb'` | Average allocated per GPU | <12GB | >14GB | **Load balance check** - Uneven allocation indicates data parallelism issues. |
| `'avg_peak_per_gpu_gb'` | Average peak per GPU | <13GB | >15GB | **Peak load balance** - Per-GPU peak for OOM safety margin. |
| `'activation_memory_gb'` | Memory for activations | <10GB | >12GB | **Temporary tensors** - Memory for intermediate computations. Dominates GPU memory. |
| `'memory_per_sample_mb'` | MB per sample | <4MB | >8MB | **Batch size planning** - Helps determine maximum safe batch size. Example: 4MB/sample with 12GB available → max ~3000 samples. |
| `'param_memory_percent'` | % memory for parameters | 15-30% | >50% | **Memory breakdown** - Parameters vs activations. Models are typically activation-bound. |
| `'activation_memory_percent'` | % memory for activations | 60-80% | >85% | **Compute vs storage** - High percentage normal; very high indicates inefficient architecture. |
| `'max_batch_size_theoretical'` | Theoretical max batch size | >32 | <16 | **Safe batch size limit** - Based on available memory and per-sample cost. |

**Critical Thresholds (T4 GPU, 16GB):**
- **Safe**: `total_peak_gb` < 13GB per GPU
- **Warning**: 13-15GB (may OOM with spikes)
- **Critical**: >15GB (will OOM) 🔴

---

### **11. FLOPs Metrics** (`compute_flops_metrics`, lines 6980-7103)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'forward_flops'` | FLOPs for forward pass | N/A | N/A | **Theoretical compute** - Floating point operations for one forward pass. Example: 1e10 = 10 GFLOPs. |
| `'total_flops_per_sample'` | FLOPs per sample (fwd+bwd) | N/A | N/A | **Total compute per sample** - Forward + backward ≈ 3× forward. Used for compute budgeting. |
| `'total_flops_per_batch'` | FLOPs per batch | N/A | N/A | **Batch compute requirement** - total_flops_per_sample × batch_size. |
| `'achieved_tflops'` | Achieved TFLOPs/sec | >1.0 | <0.5 | **Actual compute throughput** - Measured FLOPs/sec during training. Example: 2.5 TFLOPS = 2.5 trillion ops/sec. |
| `'mfu'` | Model FLOPs Utilization (fraction) | >0.15 | <0.08 | **Hardware efficiency** - Achieved FLOPs / Peak hardware FLOPs. MFU=0.20 means using 20% of theoretical max. |
| `'mfu_percent'` | MFU as percentage | >15% | <8% | **KEY EFFICIENCY METRIC** - Same as mfu × 100. Standard metric in LLM papers (GPT-3, PaLM). |
| `'flops_per_param'` | FLOPs per parameter | N/A | N/A | **Architectural efficiency** - Compute density of the model. |
| `'moe_compute_efficiency'` | Fraction of experts activated | ~0.25 | N/A | **MoE compute savings** - For top_k=2, num_experts=8 → 0.25 (only 25% of FFN compute used). |
| `'moe_param_efficiency'` | Active params / total params | ~0.25 | N/A | **MoE parameter usage** - Same as compute efficiency for standard MoE. |

**Expected MFU (Model FLOPs Utilization):**
- Exp1 (Baseline): 5-10% (standard attention is inefficient)
- Exp2 (Flash): 15-25% (kernel fusion helps)
- Exp3-6 (MoE+Flash): 20-30% (conditional compute + optimization)

**MoE Compute Savings:**
- Standard model: 100% of FLOPs
- MoE (top-2 of 8): ~25% of FLOPs (4× reduction in theory)
- Actual speedup: 1.5-2× (routing overhead reduces savings)

---

### **12. Cost Metrics** (`compute_cost_metrics`, lines 7105-7179)

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'cost_usd'` | Total cost for this run | N/A | N/A | **Actual experiment cost** - Based on GPU hours × hourly rate. Example: 2 hours × $1.40/hr = $2.80. |
| `'cost_per_epoch_usd'` | Cost per epoch | N/A | N/A | **Epoch cost** - Useful for budgeting longer training runs. |
| `'projected_cost_{N}epochs_usd'` | Projected cost for N epochs | N/A | N/A | **Scaling estimates** - Projections for N=10, 50, 100, 200 epochs. |
| `'cost_per_1k_samples_usd'` | Cost per 1000 samples | <$0.50 | >$2.00 | **Sample processing cost** - Normalized cost metric for comparing efficiency. |
| `'effective_cost_usd'` | Cost if GPU was 100% utilized | N/A | N/A | **Wasted compute indicator** - What you'd pay if MFU was 1.0. Shows optimization opportunity. |
| `'wasted_compute_usd'` | Money wasted on idle GPU | <20% | >50% | **Efficiency loss in dollars** - cost × (1 - mfu). Direct measure of optimization potential. |
| `'gpu_type'` | GPU type used | N/A | N/A | **Hardware identifier** - T4, V100, A100, L4, etc. |
| `'num_gpus'` | Number of GPUs | N/A | N/A | **Parallelism level** - Number of GPUs in training setup. |
| `'hourly_rate_usd'` | Total hourly cost | N/A | N/A | **Cost rate** - Combined rate for all GPUs. Example: 4× T4 @ $0.35 = $1.40/hr. |

**Cost Reference (GCP On-Demand Pricing 2024):**
| GPU Type | Per-GPU Hourly | 4-GPU Hourly |
|----------|---------------|--------------|
| T4 | $0.35 | $1.40 |
| L4 | $0.70 | $2.80 |
| V100 | $2.48 | $9.92 |
| A100 (40GB) | $3.67 | $14.68 |

**For 1M samples, 10 epochs, 4×T4 GPUs (~$1.40/hr):**
- Exp1: ~$500 (150 hours)
- Exp2: ~$250 (70 hours) - **50% cost reduction from Flash Attention**
- Exp3-6: ~$300 (90 hours) - **MoE adds overhead but improves quality**

---

### **13. Ablation Metrics** (`compute_ablation_metrics`, lines 7764-7866)

**Cross-experiment comparisons for component contribution analysis**

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'flash_attn_acc_gain'` | Accuracy improvement from Flash | >0.02 | <0.00 | **Absolute gain** - recall@10 improvement from Flash Attention. Example: Exp2 0.22 - Exp1 0.18 = 0.04 gain. |
| `'flash_attn_acc_gain_percent'` | % accuracy improvement | >8% | <2% | **Relative gain** - Percentage improvement. Example: 0.04 / 0.18 × 100 = 22% improvement. |
| `'flash_attn_speedup'` | Training speedup from Flash | >2.0× | <1.5× | **Time reduction** - Exp1_time / Exp2_time. Example: 10hr / 4hr = 2.5× speedup. |
| `'learned_pool_acc_gain'` | Accuracy from learned pooling | >0.01 | <0.00 | **Pooling benefit** - Additional accuracy from Exp2b vs Exp2. |
| `'learned_pool_speedup'` | Speedup from learned pooling | >1.2× | <1.05× | **Additional speedup** - Exp2_time / Exp2b_time. |
| `'moe_acc_gain'` | Accuracy from MoE | >0.03 | <0.01 | **MoE benefit** - Accuracy improvement from Exp3 vs Exp2. Especially important for tail codes. |
| `'moe_acc_per_param'` | Acc gain per M parameters added | >0.01 | <0.003 | **Parameter efficiency** - Accuracy gain per million additional parameters. Shows if extra params are worthwhile. |
| `'pool_moe_synergy_speedup'` | Pooling speedup with MoE | >1.3× | <1.1× | **Synergy effect** - Does pooling help more when combined with MoE? |
| `'pool_moe_interaction'` | Difference in pooling benefit | >0.1× | <0.0× | **Interaction strength** - Positive means pooling+MoE is superlinear benefit. |
| `'{exp}_acc_per_dollar'` | Accuracy gain per dollar spent | >0.0001 | <0.00003 | **Cost-effectiveness** - ROI metric for each experiment. |
| `'{exp}_speedup_ratio'` | Speedup vs baseline | >1.5× | <1.0× | **Relative speed** - How much faster than Exp1 baseline. |

**Key Ablations to Check:**
1. **Flash Attention**: Exp2 vs Exp1 → expect 2-3× speedup, minimal accuracy change
2. **Learned Pooling**: Exp2b vs Exp2 → expect +0.01 accuracy, +20% speed
3. **MoE**: Exp3 vs Exp2 → expect +0.03 accuracy (especially tail codes)
4. **Shared Expert**: Exp4 vs Exp3 → expect better load balance
5. **Fine-Grained**: Exp5 vs Exp4 → expect best tail accuracy
6. **Auxiliary-Free**: Exp6 vs Exp3 → expect best load balance without aux_loss tuning

---

### **14. Embedding Quality Metrics** (`compute_embedding_quality_epoch`, lines 6519-6664)

Embedding space quality for downstream task transfer. Run once per epoch (expensive!).

| Key Name | Meaning | Good Value | Bad Value | Interpretation |
|----------|---------|------------|-----------|----------------|
| `'embedding_std_mean'` | Mean std dev across dimensions | >0.05 | <0.01 | **COLLAPSE DETECTOR** - Measures if embeddings are spread out or collapsed to a point. Example: 256-dim embeddings, std across each dimension averaged. Low std = all patients look the same = useless embeddings. |
| `'nn_target_overlap'` | Jaccard similarity of NN targets | >0.20 | <0.10 | **Semantic quality** - Do similar embeddings have similar diagnoses? Computed as: for each patient, find 5 nearest neighbors, measure Jaccard overlap of diagnosis codes. High overlap = embeddings capture clinical similarity. |

**Why These Matter for Downstream Tasks:**
- **If embeddings collapse (low std)**: They won't transfer to downstream classifiers. All patients get same embedding = no information.
- **If NN overlap is low**: Embeddings don't capture clinical relationships. Similar conditions should cluster together.

**Critical Warning Signs:**
- **If `embedding_std_mean` < 0.01**: Embeddings collapsed to single point 🔴🔴🔴
  - **Action**: Reduce learning rate, add embedding normalization, check for NaN gradients
  - **Example**: std_mean=0.003 means all 256 dimensions have essentially same value → complete collapse
- **If `nn_target_overlap` < 0.10**: Embeddings not learning relationships ⚠️
  - **Action**: Check if model is training at all, increase embedding size, add more training data
  - **Example**: overlap=0.08 means nearest neighbors share only 8% of diagnosis codes → random clustering

---

### **15. Additional Helper Metrics**

These are computed by helper functions called within the main metric functions:

#### `compute_micro_recall_at_k` (lines 7397-7428)
Returns: `micro_recall@5`, `micro_recall@10`, `micro_recall@20`, `micro_recall@50`

Micro-averaged recall that counts individual code hits across all samples rather than binary per-sample hits.

#### `compute_ndcg_at_k` (lines 7431-7478)
Returns: `ndcg@10`, `ndcg@20`, `ndcg@50`

Normalized Discounted Cumulative Gain with position-based discounting.

#### `compute_positive_brier_score` (lines 7482-7513)
Returns: `positive_brier`

Brier score computed only on positive labels to avoid domination by true negatives.

#### `compute_auroc_auprc` (lines 7516-7598)
Returns: `macro_auroc`, `macro_auprc`, `num_codes_evaluated`

**⚠️ EXPENSIVE** - Samples codes for efficiency. Computes macro-averaged AUROC and AUPRC.

---

## **SUMMARY: Top 10 Most Important Metrics**

| Priority | Metric Key | Category | Critical Threshold | Why It Matters |
|----------|------------|----------|-------------------|----------------|
| 1️⃣ | `'tail_top10_acc'` | Performance | >0.08 | **Rare disease detection** - Clinical value is in catching rare conditions |
| 2️⃣ | `'recall@10'` | Performance | >0.20 | **Primary clinical metric** - Standard physician review window |
| 3️⃣ | `'num_collapsed_experts'` | MoE Health | ≤1 | **MoE success** - Routing failure destroys MoE benefits |
| 4️⃣ | `'samples_per_sec'` | Efficiency | >20 | **Training speed** - Directly impacts cost & iteration time |
| 5️⃣ | `'embedding_std_mean'` | Quality | >0.05 | **Collapse detector** - Zero if embeddings are useless |
| 6️⃣ | `'balanced_top10_acc'` | Performance | >0.20 | **Frequency-fair** - Ensures model works across code frequencies |
| 7️⃣ | `'expert_load_cv'` | MoE Health | <0.35 | **Load balance** - High CV means wasted expert capacity |
| 8️⃣ | `'positive_brier'` | Calibration | <0.30 | **Probability quality** - Confidence should match accuracy |
| 9️⃣ | `'total_peak_gb'` | Resources | <13GB/GPU | **Memory safety** - Prevents OOM crashes mid-training |
| 🔟 | `'mfu_percent'` | Efficiency | >15% | **Hardware efficiency** - Shows GPU utilization quality |

---

This exhaustive list covers **ALL 120+ metrics** used in your evaluation framework!
