I'll analyze the intrinsic evaluation metrics used in your transformer for both training and validation. Let me systematically examine the relevant code sections.

[6 tools called]

Let me also examine the loss computation and understand the task structure better:

[4 tools called]

Now I have a comprehensive understanding of your metrics framework. Let me provide a systematic and in-depth analysis.

---

# In-Depth Analysis of Intrinsic Evaluation Metrics for Next-Day Multi-Label Code Prediction

## 1. Summary of Current Metrics Architecture

### 1.1 Training-Time Metrics (Real-time, every `log_interval` batches)

From `compute_batch_metrics_lightweight()`:
- **Recall@K** (K=5, 10, 20, 50)
- **Precision@K** (K=5, 10, 20, 50)
- **mAP@K** (K=20, 50)
- **Brier Score**

### 1.2 Validation Metrics (End of epoch)

From `evaluate()`:
- **Validation Loss** (BCE)
- **Top-K Accuracy** (K=1, 5, 10, 20) — same as Recall@K

### 1.3 Comprehensive Evaluation Metrics

From `comprehensive_evaluation()` calling multiple functions:
- `compute_primary_task_metrics()`: Recall@K, Precision@K, F1@K, MRR
- `compute_loss_metrics()`: BCE Loss, ECE (Expected Calibration Error), Brier Score
- `compute_stratified_metrics()`: Common/Medium/Rare/Tail Code Accuracy, Tail Coverage, Balanced Top-10 Acc

---

## 2. Critical Assessment: Are These Metrics Appropriate?

### ✅ **APPROPRIATELY USED METRICS**

#### 2.1 Recall@K — **APPROPRIATE and PRIMARY for Clinical Use**

```5692:5707:dev/moe/moe_flashattn_3.py
        # ============================================================
        # 1. RECALL @ K (for K=5, 10, 20, 50)
        # ============================================================
        # Recall: "Was ANY true code in top-K predictions?"
        for k in [5, 10, 20, 50]:
            top_k_preds = sorted_indices[:, :k]
            correct = 0
            total = 0
            
            for i, target_codes in enumerate(valid_y):
                true_codes = [c for c in target_codes if c != 0]
                if len(true_codes) > 0:
                    total += 1
                    if any(code in top_k_preds[i].tolist() for code in true_codes):
                        correct += 1
```

**Validity Assessment:**
- ✅ **Clinically meaningful**: In healthcare, finding *any* correct code matters for alerting physicians
- ✅ **Standard in medical NLP**: Used by BEHRT, Med-BERT, ClinicalBERT (as noted in your docstring)
- ⚠️ **CONCERN — Binary definition is too lenient**: Your implementation counts a "hit" if *any single* true code is found. For a day with 10 true codes, predicting 1 of them gives 100% Recall@K for that sample. This **inflates reported recall** and doesn't reflect how well you capture *all* relevant codes.

**Recommendation**: Add **Micro-Recall@K** (total hits / total true codes) for a more granular view.

---

#### 2.2 Precision@K — **APPROPRIATE but with caveats**

```5712:5724:dev/moe/moe_flashattn_3.py
        # ============================================================
        # 2. PRECISION @ K (for K=5, 10, 20, 50)
        # ============================================================
        # Precision: "Of top-K predictions, how many were correct?"
        for k in [5, 10, 20, 50]:
            top_k_preds = sorted_indices[:, :k]
            precisions = []
            
            for i, target_codes in enumerate(valid_y):
                true_codes = set([c for c in target_codes if c != 0])
                if len(true_codes) > 0:
                    pred_codes = top_k_preds[i].tolist()
                    hits = sum(1 for code in pred_codes if code in true_codes)
                    precisions.append(hits / k)
```

**Validity Assessment:**
- ✅ **Correct implementation** for multi-label setting
- ⚠️ **K is fixed, but number of true labels varies**: If a sample has 3 true codes, P@20 is capped at 3/20 = 0.15 *even with perfect predictions*. This creates **systematic underestimation** for samples with few true labels.

**Recommendation**: Also report **P@K normalized by min(K, |true|)** or use **Precision at n** where n = number of true labels.

---

#### 2.3 Brier Score — **APPROPRIATE for Calibration**

```5748:5760:dev/moe/moe_flashattn_3.py
        # ============================================================
        # 4. BRIER SCORE (calibration quality)
        # ============================================================
        probs = torch.sigmoid(predictions)
        targets_binary = torch.zeros_like(predictions)
        
        for i, target_codes in enumerate(valid_y):
            for code in target_codes:
                if code > 0 and code < config.target_cd_cnt:
                    targets_binary[i, code] = 1
        
        brier = ((probs - targets_binary) ** 2).mean().item()
        metrics['brier_score'] = brier
```

**Validity Assessment:**
- ✅ **Correct implementation**: Measures (prediction - target)² averaged over all predictions
- ✅ **Critical for embeddings**: Well-calibrated probabilities produce better downstream embeddings
- ⚠️ **Dominated by negatives**: With 6,297 target codes and typically ~5-20 true codes per day, ~99.7% of entries are 0. The Brier score will be **dominated by how well you predict 0s** (i.e., true negatives), potentially masking poor positive prediction quality.

**Recommendation**: Also compute **Positive-Only Brier** (Brier score only on indices where target=1).

---

#### 2.4 Expected Calibration Error (ECE) — **APPROPRIATE**

```6102:6127:dev/moe/moe_flashattn_3.py
    # 2. Expected Calibration Error (ECE)
    # Bin predicted probabilities and check if they match empirical frequencies
    probs = torch.sigmoid(predictions)
    
    num_bins = 10
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find predictions in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        
        if in_bin.any():
            # Average predicted probability in bin
            avg_pred = probs[in_bin].mean().item()
            # Actual fraction of positives in bin
            avg_true = targets_multihot[in_bin].mean().item()
            # Weight by bin size
            bin_weight = in_bin.sum().item() / in_bin.numel()
            # ECE contribution
            ece += bin_weight * abs(avg_pred - avg_true)
```

**Validity Assessment:**
- ✅ **Standard calibration metric** for probability outputs
- ✅ **Important for clinical decision support** (confidence should match accuracy)
- ⚠️ **Same dominance issue**: Most predictions fall in [0, 0.1] bin, so ECE is heavily weighted toward low-probability predictions.

---

#### 2.5 Stratified Metrics by Code Frequency — **HIGHLY APPROPRIATE**

```6167:6219:dev/moe/moe_flashattn_3.py
    # Define frequency tiers
    freq_percentiles = np.percentile(code_frequencies[code_frequencies > 0], [20, 50, 80])
    
    common_codes = set(np.where(code_frequencies > freq_percentiles[2])[0].tolist())
    medium_codes = set(np.where(
        (code_frequencies <= freq_percentiles[2]) & 
        (code_frequencies > freq_percentiles[1])
    )[0].tolist())
    rare_codes = set(np.where(
        (code_frequencies <= freq_percentiles[1]) & 
        (code_frequencies > freq_percentiles[0])
    )[0].tolist())
    tail_codes = set(np.where(
        (code_frequencies <= freq_percentiles[0]) & 
        (code_frequencies > 0)
    )[0].tolist())
```

**Validity Assessment:**
- ✅ **Excellent for healthcare AI**: Medical codes follow extreme long-tail distribution
- ✅ **Detects "easy mode" models**: Models that only predict common codes will fail on rare/tail
- ✅ **Balanced Top-10 Acc** gives equal weight to each tier — proper for clinical evaluation

---

### ⚠️ **METRICS WITH ISSUES**

#### 2.6 mAP@K — **PARTIALLY PROBLEMATIC**

```5727:5746:dev/moe/moe_flashattn_3.py
        # ============================================================
        # 3. mAP @ K (for K=20, 50)
        # ============================================================
        # Mean Average Precision: Average of precision at each relevant item
        for k in [20, 50]:
            aps = []
            
            for i, target_codes in enumerate(valid_y):
                true_codes = set([c for c in target_codes if c != 0])
                if len(true_codes) > 0:
                    hits = 0
                    precisions_at_k = []
                    for rank, pred_code in enumerate(sorted_indices[i, :k].tolist(), 1):
                        if pred_code in true_codes:
                            hits += 1
                            precisions_at_k.append(hits / rank)
                    
                    if precisions_at_k:
                        aps.append(np.mean(precisions_at_k))
```

**Issue #1 — mAP definition may not match your needs**:
- Standard mAP averages precision at *each recall level*. Your implementation averages precision at each *hit position*.
- For samples where no hits occur in top-K, you don't append anything (samples with all misses are *excluded*), which **biases mAP upward**.

**Issue #2 — Doesn't account for codes NOT found**:
- If a sample has 10 true codes but you only find 2, your mAP still computes based on those 2 hits. It doesn't penalize *missing* the other 8.

**Recommendation**: Consider using **NDCG@K** (Normalized Discounted Cumulative Gain) which explicitly handles both ranking quality and relevance.

---

#### 2.7 MRR (Mean Reciprocal Rank) — **PROBLEMATIC for Multi-Label**

```6035:6048:dev/moe/moe_flashattn_3.py
    # Mean Reciprocal Rank
    sorted_indices = torch.argsort(predictions, dim=-1, descending=True)
    reciprocal_ranks = []
    
    for i, target_codes in enumerate(targets):
        true_codes = [c for c in target_codes if c != 0]
        if len(true_codes) > 0:
            # Find rank of first true code
            first_true = true_codes[0]
            rank = (sorted_indices[i] == first_true).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                reciprocal_ranks.append(1.0 / (rank.item() + 1))
```

**Critical Issue**:
- You compute MRR based on `true_codes[0]` — the *first* code in the list. This is **arbitrary** and doesn't reflect clinical importance.
- In a multi-label setting, MRR typically should find the rank of the *highest-ranked* true code, not the first one in the input list.

**Current behavior**: If true codes are `[123, 456, 789]` and your prediction ranks are:
- Code 456 at rank 1
- Code 789 at rank 5  
- Code 123 at rank 100

Your MRR only considers code 123 (rank 100), giving RR = 0.01, even though you correctly ranked code 456 at position 1!

**Recommendation**: Change to find the **best-ranked** true code:
```python
min_rank = min(
    (sorted_indices[i] == code).nonzero(as_tuple=True)[0].item() 
    for code in true_codes if code in sorted_indices[i]
)
reciprocal_ranks.append(1.0 / (min_rank + 1))
```

---

#### 2.8 Top-K "Accuracy" — **MISNOMER, ACTUALLY RECALL**

```4689:4703:dev/moe/moe_flashattn_3.py
    # Top-K accuracy
    top_k_results = {}
    for k in [1, 5, 10, 20]:
        top_k_preds = torch.topk(all_predictions, k, dim=-1).indices
        correct = 0
        total = 0
        
        for i, target_codes in enumerate(all_targets):
            if any(code != 0 for code in target_codes):
                total += 1
                # Check if any true code is in top-K
                if any(code in top_k_preds[i].tolist() for code in target_codes if code != 0):
                    correct += 1
        
        top_k_results[f'top_{k}_acc'] = correct / total if total > 0 else 0.0
```

**Assessment**: 
- This is **exactly the same as Recall@K**, not accuracy. Naming it "accuracy" is misleading.
- In the `evaluate()` function, you return `top_1_acc`, `top_5_acc`, etc., which are actually Recall@K.

**Recommendation**: Rename to `recall@K` for consistency and clarity.

---

### ❌ **MISSING CRITICAL METRICS**

#### 2.9 **Missing: Micro-Averaged Recall** (Per-Code Hit Rate)

Your current Recall@K is **sample-level binary**: "Was at least one code hit?"

What's missing: **Micro-Recall** = (Total correct predictions across all samples and codes) / (Total true codes across all samples)

This tells you: "Of all the codes the model should predict, what fraction did it actually capture?"

---

#### 2.10 **Missing: Label Cardinality Analysis**

You should track:
- Average number of true codes per sample (ground truth cardinality)
- Average number of predicted codes above threshold (prediction cardinality)
- Cardinality difference (|predicted| - |true|)

This reveals if the model is **over-predicting** (many false positives) or **under-predicting** (missing codes).

---

#### 2.11 **Missing: Subset Accuracy**

For multi-label classification, **subset accuracy** = fraction of samples where predicted set exactly equals true set.

While this is extremely strict (often near 0%), it's the truest measure of "getting it right."

---

#### 2.12 **Missing: AUROC / AUPRC**

For binary classification on each code, you should compute:
- **Macro-AUROC**: Average AUROC across all codes
- **Macro-AUPRC**: Average Area Under Precision-Recall Curve (better for imbalanced classes)

These are threshold-agnostic and reveal discriminative power.

---

## 3. Do These Metrics Deliver Correct Information?

### 3.1 **For Monitoring Training Progress** — Mostly Yes ✅

The training-time metrics (Recall@K, Precision@K, Brier) provide useful signals:
- Recall@K rising → Model learning to rank true codes higher
- Precision@K rising → Model not just predicting common codes
- Brier falling → Better probability calibration

### 3.2 **For Final Model Evaluation** — Partially ⚠️

| Metric | What It Claims to Measure | What It Actually Measures |
|--------|--------------------------|---------------------------|
| Recall@K | Did model find relevant codes? | Did model find *at least one* relevant code? (Binary) |
| Precision@K | How precise are top predictions? | Hits/K (but penalized when fewer than K true codes exist) |
| mAP@K | Ranking quality | Ranking quality *for hits only*, excludes total misses |
| MRR | First relevant result rank | Rank of *arbitrarily first* code in list |
| top_K_acc | Classification accuracy | Actually Recall@K (misnomer) |
| Brier | Calibration | Dominated by true negatives (~99.7% of entries) |

### 3.3 **For Comparing Architectures** — Yes ✅

The stratified metrics (`common_top10_acc`, `rare_top10_acc`, `balanced_top10_acc`) are excellent for comparing models fairly, as they prevent models from gaming metrics by only predicting frequent codes.

---

## 4. Recommendations Summary

### High Priority Fixes

1. **Fix MRR implementation**: Use best-ranked true code, not first in list
2. **Rename `top_K_acc` to `recall@K`**: Avoid confusion
3. **Add Micro-Recall@K**: `sum(hits) / sum(true_labels)` across all samples

### Medium Priority Additions

4. **Add Positive-Only Brier Score**: Evaluate calibration on positive labels only
5. **Add NDCG@K**: Standard ranking metric that handles multi-label properly
6. **Add Macro-AUROC and Macro-AUPRC**: Threshold-agnostic discriminative metrics

### Nice-to-Have

7. **Add Label Cardinality metrics**: Detect over/under-prediction tendencies
8. **Add Subset Accuracy**: Strictest measure of exact match

---

## 5. Conclusion

Your metrics framework is **reasonably well-designed** for the multi-label next-day code prediction task, with appropriate use of Recall@K, Precision@K, stratified frequency analysis, and calibration metrics. However:

1. **MRR is incorrectly implemented** for multi-label (uses arbitrary first code instead of best-ranked)
2. **Recall@K is binary** (hit/miss per sample) which inflates reported performance
3. **Brier/ECE are dominated by true negatives** due to extreme class imbalance
4. **mAP excludes samples with no hits**, biasing the metric upward
5. **Key metrics are missing**: Micro-Recall, AUROC/AUPRC, NDCG

The core metrics (loss, Recall@K, Precision@K, stratified accuracy) **do deliver correct directional information** for training and model comparison. The issues above affect absolute values and edge cases but shouldn't fundamentally mislead your architecture decisions.

# Implementations

## Overview of Changes

| Category | Action | Affected Locations |
|----------|--------|-------------------|
| **MRR** | Fix to use best-ranked | `compute_primary_task_metrics()` (lines 6035-6048) |
| **Rename** | `top_K_acc` → `recall@K` | `evaluate()`, `_build_epoch_metrics()`, `_build_final_results()`, tests, etc. (45+ occurrences) |
| **Remove** | mAP, Brier, ECE | `compute_batch_metrics_lightweight()`, `compute_loss_metrics()`, training prints |
| **Add** | Micro-Recall, AUROC/AUPRC, NDCG, Pos-Brier | New functions + integration |

---

## STEP 1: Add New Metric Functions (Insert after `compute_loss_metrics()` ~line 6133)

Add these new functions. Insert them around line 6133 (after `compute_loss_metrics()` and before `compute_stratified_metrics()`):

```python
def compute_micro_recall_at_k(
    predictions: torch.Tensor,  # [num_samples, vocab_size]
    targets: List[List[int]],   # Multi-label targets
    k_values: List[int] = [5, 10, 20, 50]
) -> Dict[str, float]:
    """
    Micro-averaged Recall@K: Total hits / Total true labels across all samples.
    
    Unlike sample-level Recall@K (binary hit/miss per sample), this measures
    what fraction of ALL true codes across the dataset are captured in top-K.
    
    Returns:
        Dict with 'micro_recall@5', 'micro_recall@10', etc.
    """
    metrics = {}
    sorted_indices = torch.argsort(predictions, dim=-1, descending=True)
    
    for k in k_values:
        top_k_preds = sorted_indices[:, :k]
        total_hits = 0
        total_true = 0
        
        for i, target_codes in enumerate(targets):
            true_codes = set(c for c in target_codes if c != 0)
            if len(true_codes) > 0:
                total_true += len(true_codes)
                pred_set = set(top_k_preds[i].tolist())
                total_hits += len(true_codes & pred_set)
        
        metrics[f'micro_recall@{k}'] = total_hits / total_true if total_true > 0 else 0.0
    
    return metrics


def compute_ndcg_at_k(
    predictions: torch.Tensor,  # [num_samples, vocab_size]
    targets: List[List[int]],   # Multi-label targets
    k_values: List[int] = [10, 20, 50]
) -> Dict[str, float]:
    """
    Normalized Discounted Cumulative Gain @ K.
    
    NDCG accounts for:
    1. Position-based discounting (earlier = better)
    2. Relevance scores (binary in our case)
    3. Normalized by ideal ranking
    
    Returns:
        Dict with 'ndcg@10', 'ndcg@20', 'ndcg@50'
    """
    metrics = {}
    sorted_indices = torch.argsort(predictions, dim=-1, descending=True)
    
    # Precompute discount factors: 1/log2(rank+2) for ranks 0,1,2,...
    max_k = max(k_values)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))  # [1/log2(2), 1/log2(3), ...]
    
    for k in k_values:
        ndcg_scores = []
        
        for i, target_codes in enumerate(targets):
            true_codes = set(c for c in target_codes if c != 0)
            if len(true_codes) == 0:
                continue
            
            # DCG: sum of discounted gains for hits in top-k
            top_k_preds = sorted_indices[i, :k].tolist()
            dcg = sum(
                discounts[rank] 
                for rank, pred in enumerate(top_k_preds) 
                if pred in true_codes
            )
            
            # Ideal DCG: if we had placed all true codes at top
            num_relevant = min(len(true_codes), k)
            idcg = sum(discounts[:num_relevant])
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    return metrics


def compute_positive_brier_score(
    predictions: torch.Tensor,   # [num_samples, vocab_size] logits
    targets: List[List[int]],    # Multi-label targets
    vocab_size: int
) -> Dict[str, float]:
    """
    Brier score computed ONLY on positive labels.
    
    Standard Brier is dominated by true negatives (~99.7% of entries).
    This variant measures calibration specifically for positive predictions.
    
    Returns:
        Dict with 'positive_brier' (lower is better, 0 = perfect)
    """
    probs = torch.sigmoid(predictions)
    
    # Collect all predicted probabilities for positive labels
    positive_probs = []
    
    for i, target_codes in enumerate(targets):
        for code in target_codes:
            if 0 < code < vocab_size:
                positive_probs.append(probs[i, code].item())
    
    if len(positive_probs) == 0:
        return {'positive_brier': 0.0}
    
    # For positive labels, target = 1, so Brier = (prob - 1)^2
    positive_probs = np.array(positive_probs)
    positive_brier = np.mean((positive_probs - 1.0) ** 2)
    
    return {'positive_brier': positive_brier}


def compute_auroc_auprc(
    predictions: torch.Tensor,   # [num_samples, vocab_size] logits
    targets: List[List[int]],    # Multi-label targets
    vocab_size: int,
    num_codes_to_sample: int = 500  # Sample codes for efficiency
) -> Dict[str, float]:
    """
    Macro-averaged AUROC and AUPRC across codes.
    
    Due to computational cost, we sample a subset of codes:
    - All codes that appear in targets (ensures coverage)
    - Random sample of additional codes
    
    Returns:
        Dict with 'macro_auroc', 'macro_auprc', 'num_codes_evaluated'
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    probs = torch.sigmoid(predictions).cpu().numpy()
    num_samples = len(predictions)
    
    # Build binary target matrix for sampled codes
    # First, find all codes that appear in targets
    target_codes_set = set()
    for target_list in targets:
        for code in target_list:
            if 0 < code < vocab_size:
                target_codes_set.add(code)
    
    # If too few positive codes, return 0
    if len(target_codes_set) < 10:
        return {'macro_auroc': 0.0, 'macro_auprc': 0.0, 'num_codes_evaluated': 0}
    
    # Sample additional codes for negative class representation
    all_codes = list(target_codes_set)
    if len(all_codes) < num_codes_to_sample:
        # Add some random codes not in targets
        remaining = list(set(range(1, vocab_size)) - target_codes_set)
        additional = np.random.choice(
            remaining, 
            min(num_codes_to_sample - len(all_codes), len(remaining)),
            replace=False
        ).tolist()
        all_codes.extend(additional)
    
    # Build target matrix for selected codes
    code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
    y_true = np.zeros((num_samples, len(all_codes)), dtype=np.float32)
    
    for i, target_list in enumerate(targets):
        for code in target_list:
            if code in code_to_idx:
                y_true[i, code_to_idx[code]] = 1.0
    
    # Get predictions for selected codes
    y_pred = probs[:, all_codes]
    
    # Compute per-code metrics (skip codes with no positives or all positives)
    aurocs = []
    auprcs = []
    
    for j in range(len(all_codes)):
        col_true = y_true[:, j]
        col_pred = y_pred[:, j]
        
        # Skip if no variance in labels
        if col_true.sum() == 0 or col_true.sum() == len(col_true):
            continue
        
        try:
            aurocs.append(roc_auc_score(col_true, col_pred))
            auprcs.append(average_precision_score(col_true, col_pred))
        except ValueError:
            continue
    
    return {
        'macro_auroc': np.mean(aurocs) if aurocs else 0.0,
        'macro_auprc': np.mean(auprcs) if auprcs else 0.0,
        'num_codes_evaluated': len(aurocs)
    }
```

---

## STEP 2: Fix MRR in `compute_primary_task_metrics()` (lines 6035-6048)

**Location:** `compute_primary_task_metrics()` around lines 6035-6048

**Current code:**
```python
    # Mean Reciprocal Rank
    sorted_indices = torch.argsort(predictions, dim=-1, descending=True)
    reciprocal_ranks = []
    
    for i, target_codes in enumerate(targets):
        true_codes = [c for c in target_codes if c != 0]
        if len(true_codes) > 0:
            # Find rank of first true code
            first_true = true_codes[0]
            rank = (sorted_indices[i] == first_true).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                reciprocal_ranks.append(1.0 / (rank.item() + 1))
    
    metrics['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
```

**Replace with:**
```python
    # Mean Reciprocal Rank (best-ranked true code)
    sorted_indices = torch.argsort(predictions, dim=-1, descending=True)
    reciprocal_ranks = []
    
    for i, target_codes in enumerate(targets):
        true_codes = [c for c in target_codes if c != 0]
        if len(true_codes) > 0:
            # Find rank of BEST-RANKED true code (not arbitrary first)
            best_rank = float('inf')
            for code in true_codes:
                rank_tensor = (sorted_indices[i] == code).nonzero(as_tuple=True)[0]
                if len(rank_tensor) > 0:
                    rank = rank_tensor.item()
                    if rank < best_rank:
                        best_rank = rank
            
            if best_rank < float('inf'):
                reciprocal_ranks.append(1.0 / (best_rank + 1))
    
    metrics['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
```

---

## STEP 3: Update `compute_primary_task_metrics()` to Include New Metrics

At the end of `compute_primary_task_metrics()` (before the `return metrics` statement, ~line 6058), add:

```python
    # Add micro-recall metrics
    metrics.update(compute_micro_recall_at_k(predictions, targets, [5, 10, 20, 50]))
    
    # Add NDCG metrics
    metrics.update(compute_ndcg_at_k(predictions, targets, [10, 20, 50]))
```

---

## STEP 4: Update `compute_loss_metrics()` - Remove ECE/Brier, Add Positive-Brier

**Location:** `compute_loss_metrics()` (lines 6067-6133)

**Replace the entire function with:**

```python
def compute_loss_metrics(
    predictions: torch.Tensor,
    targets_multihot: torch.Tensor,  # [num_samples, vocab_size]
    criterion: nn.Module,
    targets_list: Optional[List[List[int]]] = None  # For positive-only Brier
) -> Dict[str, float]:
    """
    Loss and calibration metrics.
    
    Returns:
        1. BCE Loss:
           - Primary optimization objective
           - Report both total and per-sample average
        
        2. Positive-Only Brier Score:
           - Calibration on positive labels only
           - Not dominated by true negatives
        
        3. Per-Class Loss Variance:
           - Detect if model ignores certain code categories
    """
    metrics = {}
    vocab_size = predictions.shape[1]
    
    # 1. BCE Loss (total and per-sample)
    with torch.no_grad():
        total_loss = criterion(predictions, targets_multihot)
        metrics['bce_loss'] = total_loss.item()
        
        # Per-sample loss
        per_sample_loss = F.binary_cross_entropy_with_logits(
            predictions, targets_multihot, reduction='none'
        ).mean(dim=-1)
        metrics['bce_loss_mean'] = per_sample_loss.mean().item()
        metrics['bce_loss_std'] = per_sample_loss.std().item()
    
    # 2. Positive-Only Brier Score
    if targets_list is not None:
        metrics.update(compute_positive_brier_score(predictions, targets_list, vocab_size))
    else:
        # Fallback: compute from multihot
        probs = torch.sigmoid(predictions)
        positive_mask = targets_multihot > 0.5
        if positive_mask.any():
            positive_probs = probs[positive_mask]
            metrics['positive_brier'] = ((positive_probs - 1.0) ** 2).mean().item()
        else:
            metrics['positive_brier'] = 0.0
    
    return metrics
```

---

## STEP 5: Update `compute_batch_metrics_lightweight()` - Remove mAP/Brier

**Location:** `compute_batch_metrics_lightweight()` (lines 5630-5762)

This function is used during training for real-time monitoring. Replace it entirely:

```python
def compute_batch_metrics_lightweight(
    output: torch.Tensor,
    y: List[List[List[int]]],
    dt_cnt: List[int],
    config: BaseConfig,
    device: torch.device
) -> Dict[str, float]:
    """
    Lightweight metrics for real-time training monitoring (every 100 batches).
    
    These are FAST approximations that complement loss during training.
    Full comprehensive metrics are computed at epoch end via evaluate().
    
    Metrics:
    1. Recall@5, 10, 20, 50 - Clinical utility at different cutoffs
    2. Precision@5, 10, 20, 50 - How many predictions are correct
    3. Micro-Recall@10, 20 - Per-code hit rate (more granular than sample-level)
    4. NDCG@20 - Ranking quality with position discounting
    5. Positive-Only Brier - Calibration on positive labels
    
    Returns:
        Dict with recall, precision, micro_recall, ndcg, positive_brier metrics
    """
    with torch.no_grad():
        batch_size = len(dt_cnt)
        actual_len_dy = output.shape[1]
        output_flat = output.reshape(batch_size * actual_len_dy, config.target_cd_cnt)
        y_flat = [item for sublist in y for item in sublist]
        dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
        
        # Filter valid outputs (only actual days, not padding)
        valid_outputs = []
        valid_y = []
        
        for j in range(batch_size):
            valid_days = min(int(dt_cnt_list[j]), actual_len_dy)
            if valid_days <= 0:
                continue            
            # For outputs: use actual_len_dy
            start_idx = actual_len_dy * j
            end_idx = start_idx + valid_days
            valid_outputs.append(output_flat[start_idx:end_idx])
            
            # For targets: use config.len_dy (y is always padded)
            y_start = config.len_dy * j
            y_end = y_start + valid_days
            valid_y.extend(y_flat[y_start:y_end])
        
        if len(valid_outputs) == 0:
            return {
                'recall@5': 0.0, 'recall@10': 0.0, 'recall@20': 0.0, 'recall@50': 0.0,
                'precision@5': 0.0, 'precision@10': 0.0, 'precision@20': 0.0, 'precision@50': 0.0,
                'micro_recall@10': 0.0, 'micro_recall@20': 0.0,
                'ndcg@20': 0.0,
                'positive_brier': 0.0
            }
        
        predictions = torch.cat(valid_outputs)  # [num_valid_samples, vocab_size]
        num_samples = len(predictions)
        
        metrics = {}
        sorted_indices = torch.argsort(predictions, dim=-1, descending=True)
        
        # ============================================================
        # 1. RECALL @ K (for K=5, 10, 20, 50)
        # ============================================================
        for k in [5, 10, 20, 50]:
            top_k_preds = sorted_indices[:, :k]
            correct = 0
            total = 0
            
            for i, target_codes in enumerate(valid_y):
                true_codes = [c for c in target_codes if c != 0]
                if len(true_codes) > 0:
                    total += 1
                    if any(code in top_k_preds[i].tolist() for code in true_codes):
                        correct += 1
            
            metrics[f'recall@{k}'] = correct / total if total > 0 else 0.0
        
        # ============================================================
        # 2. PRECISION @ K (for K=5, 10, 20, 50)
        # ============================================================
        for k in [5, 10, 20, 50]:
            top_k_preds = sorted_indices[:, :k]
            precisions = []
            
            for i, target_codes in enumerate(valid_y):
                true_codes = set([c for c in target_codes if c != 0])
                if len(true_codes) > 0:
                    pred_codes = top_k_preds[i].tolist()
                    hits = sum(1 for code in pred_codes if code in true_codes)
                    precisions.append(hits / k)
            
            metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
        
        # ============================================================
        # 3. MICRO-RECALL @ K (for K=10, 20) - Lightweight version
        # ============================================================
        for k in [10, 20]:
            top_k_preds = sorted_indices[:, :k]
            total_hits = 0
            total_true = 0
            
            for i, target_codes in enumerate(valid_y):
                true_codes = set(c for c in target_codes if c != 0)
                if len(true_codes) > 0:
                    total_true += len(true_codes)
                    pred_set = set(top_k_preds[i].tolist())
                    total_hits += len(true_codes & pred_set)
            
            metrics[f'micro_recall@{k}'] = total_hits / total_true if total_true > 0 else 0.0
        
        # ============================================================
        # 4. NDCG @ 20 (Lightweight - single K value for speed)
        # ============================================================
        k = 20
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        ndcg_scores = []
        
        for i, target_codes in enumerate(valid_y):
            true_codes = set(c for c in target_codes if c != 0)
            if len(true_codes) == 0:
                continue
            
            top_k_preds = sorted_indices[i, :k].tolist()
            dcg = sum(discounts[rank] for rank, pred in enumerate(top_k_preds) if pred in true_codes)
            num_relevant = min(len(true_codes), k)
            idcg = sum(discounts[:num_relevant])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        metrics['ndcg@20'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        # ============================================================
        # 5. POSITIVE-ONLY BRIER SCORE
        # ============================================================
        probs = torch.sigmoid(predictions)
        positive_probs = []
        
        for i, target_codes in enumerate(valid_y):
            for code in target_codes:
                if 0 < code < config.target_cd_cnt:
                    positive_probs.append(probs[i, code].item())
        
        if len(positive_probs) > 0:
            positive_probs = np.array(positive_probs)
            metrics['positive_brier'] = float(np.mean((positive_probs - 1.0) ** 2))
        else:
            metrics['positive_brier'] = 0.0
        
        return metrics
```

---

## STEP 6: Update Training Print Statement in `train_epoch()`

**Location:** Lines 4448-4456 in `train_epoch()`

**Find this code:**
```python
                print(f"    Loss: {loss_display:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"P@20: {batch_metrics['precision@20']:.3f} | "
                      f"mAP20: {batch_metrics['mAP@20']:.3f} | "
                      f"mAP50: {batch_metrics['mAP@50']:.3f} | "
                      f"Brier: {batch_metrics['brier_score']:.4f}")
```

**Replace with:**
```python
                print(f"    Loss: {loss_display:.4f} | "
                      f"R@10: {batch_metrics['recall@10']:.3f} | "
                      f"R@20: {batch_metrics['recall@20']:.3f} | "
                      f"μR@10: {batch_metrics['micro_recall@10']:.3f} | "
                      f"P@10: {batch_metrics['precision@10']:.3f} | "
                      f"NDCG@20: {batch_metrics['ndcg@20']:.3f} | "
                      f"PosBrier: {batch_metrics['positive_brier']:.4f}")
```

---

## STEP 7: Update `train_epoch()` Docstring

**Location:** Lines 4253-4258 in `train_epoch()`

**Find:**
```python
    Logs metrics every `log_interval` batches:
    0. Loss (BCE + aux loss if MoE)
    1. Recall@5, 10, 20, 50 - Clinical utility at different cutoffs
    2. Precision@5, 10, 20, 50 - How many predictions are correct
    3. mAP@20, mAP@50 - Ranking quality
    4. Brier score - Calibration quality (critical for embeddings)
    5 MoE health (if applicable)
```

**Replace with:**
```python
    Logs metrics every `log_interval` batches:
    0. Loss (BCE + aux loss if MoE)
    1. Recall@5, 10, 20, 50 - Clinical utility at different cutoffs
    2. Precision@5, 10, 20, 50 - How many predictions are correct
    3. Micro-Recall@10, 20 - Per-code coverage rate
    4. NDCG@20 - Ranking quality with position discounting
    5. Positive-Only Brier - Calibration on positive labels
    6. MoE health (if applicable)
```

---

## STEP 8: Rename `top_K_acc` to `recall@K` in `evaluate()`

**Location:** `evaluate()` function (lines 4546-4710)

### 8a. Update default returns (lines 4576-4580)

**Find:**
```python
        return {'val_loss': 0.0, 
                'top_1_acc': 0.0, 
                'top_5_acc': 0.0, 
                'top_10_acc': 0.0, 
                'top_20_acc': 0.0}
```

**Replace with:**
```python
        return {'val_loss': 0.0, 
                'recall@1': 0.0, 
                'recall@5': 0.0, 
                'recall@10': 0.0, 
                'recall@20': 0.0}
```

### 8b. Update empty predictions return (lines 4681-4686)

**Find:**
```python
        return {
            'val_loss': val_loss,
            'top_1_acc': 0.0,
            'top_5_acc': 0.0,
            'top_10_acc': 0.0,
            'top_20_acc': 0.0
        }
```

**Replace with:**
```python
        return {
            'val_loss': val_loss,
            'recall@1': 0.0,
            'recall@5': 0.0,
            'recall@10': 0.0,
            'recall@20': 0.0
        }
```

### 8c. Update the metric computation loop (lines 4689-4703)

**Find:**
```python
    # Top-K accuracy
    top_k_results = {}
    for k in [1, 5, 10, 20]:
        top_k_preds = torch.topk(all_predictions, k, dim=-1).indices
        correct = 0
        total = 0
        
        for i, target_codes in enumerate(all_targets):
            if any(code != 0 for code in target_codes):
                total += 1
                # Check if any true code is in top-K
                if any(code in top_k_preds[i].tolist() for code in target_codes if code != 0):
                    correct += 1
        
        top_k_results[f'top_{k}_acc'] = correct / total if total > 0 else 0.0
```

**Replace with:**
```python
    # Recall@K (previously named top_K_acc)
    recall_results = {}
    for k in [1, 5, 10, 20]:
        top_k_preds = torch.topk(all_predictions, k, dim=-1).indices
        correct = 0
        total = 0
        
        for i, target_codes in enumerate(all_targets):
            if any(code != 0 for code in target_codes):
                total += 1
                # Check if any true code is in top-K
                if any(code in top_k_preds[i].tolist() for code in target_codes if code != 0):
                    correct += 1
        
        recall_results[f'recall@{k}'] = correct / total if total > 0 else 0.0
```

### 8d. Update return statement (lines 4705-4708)

**Find:**
```python
    results = {
        'val_loss': val_loss,
        **top_k_results
    }
```

**Replace with:**
```python
    results = {
        'val_loss': val_loss,
        **recall_results
    }
```

---

## STEP 9: Update `_build_epoch_metrics()` (lines 9086-9127)

**Find all occurrences of `top_X_acc` and replace:**

```python
        # Train evaluation
        'eval_in_train_loss_final': train_eval_metrics['val_loss'],
        'eval_in_train_recall@1': train_eval_metrics['recall@1'],
        'eval_in_train_recall@5': train_eval_metrics['recall@5'],
        'eval_in_train_recall@10': train_eval_metrics['recall@10'],
        'eval_in_train_recall@20': train_eval_metrics['recall@20'],
        # Validation
        'final_val_loss': val_metrics['val_loss'],
        'final_val_recall@1': val_metrics['recall@1'],
        'final_val_recall@5': val_metrics['recall@5'],
        'final_val_recall@10': val_metrics['recall@10'],
        'final_val_recall@20': val_metrics['recall@20'],
```

Also update the exclusion filter (line 9124):
```python
        if k not in epoch_metrics and k not in ['val_loss', 'recall@1', 'recall@5', 'recall@10', 'recall@20']:
```

---

## STEP 10: Update `_build_final_results()` (lines 9140-9167)

**Find and replace all `top_X_acc` references:**

```python
        'final_train_recall@5': final_metrics['eval_in_train_recall@5'],
        'final_train_recall@10': final_metrics['eval_in_train_recall@10'],
        'final_train_recall@20': final_metrics['eval_in_train_recall@20'],
        'final_val_recall@5': final_metrics['final_val_recall@5'],
        'final_val_recall@10': final_metrics['final_val_recall@10'],
        'final_val_recall@20': final_metrics['final_val_recall@20'],
```

---

## STEP 11: Update `compute_convergence_metrics()` (line 6317)

**Find:**
```python
    top10_accs = [epoch.get('top_10_acc', 0.0) for epoch in epoch_metrics]
```

**Replace with:**
```python
    recall_at_10 = [epoch.get('recall@10', epoch.get('final_val_recall@10', 0.0)) for epoch in epoch_metrics]
```

---

## STEP 12: Update Logging Statements

### 12a. Line 9421
**Find:**
```python
        logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}, Top-10: {val_metrics['top_10_acc']:.3f}")
```
**Replace with:**
```python
        logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}, Recall@10: {val_metrics['recall@10']:.3f}")
```

### 12b. Line 9502
**Find:**
```python
    logger.info(f"Final Top-10 Acc: {epoch_history[-1]['final_val_top_10_acc']:.3f}")
```
**Replace with:**
```python
    logger.info(f"Final Recall@10: {epoch_history[-1]['final_val_recall@10']:.3f}")
```

---

## STEP 13: Update `comprehensive_evaluation()` to Include AUROC/AUPRC

**Location:** ~line 6970 in `comprehensive_evaluation()`

**Find:**
```python
    # 1. PERFORMANCE METRICS
    print("Computing performance metrics...")
    evaluation['performance'] = {
        **compute_primary_task_metrics(all_predictions, all_targets, config.target_cd_cnt),
        **compute_loss_metrics(all_predictions, all_targets_multihot, criterion),
        **compute_stratified_metrics(all_predictions, all_targets, code_frequencies, config.target_cd_cnt)
    }
```

**Replace with:**
```python
    # 1. PERFORMANCE METRICS
    print("Computing performance metrics...")
    evaluation['performance'] = {
        **compute_primary_task_metrics(all_predictions, all_targets, config.target_cd_cnt),
        **compute_loss_metrics(all_predictions, all_targets_multihot, criterion, all_targets),
        **compute_stratified_metrics(all_predictions, all_targets, code_frequencies, config.target_cd_cnt),
        **compute_auroc_auprc(all_predictions, all_targets, config.target_cd_cnt)
    }
```

---

## STEP 14: Update Test Functions

### 14a. `test_comprehensive_metrics_computation()` (lines 11500-11540)

Update print statements and assertions for the new metric names:

**Find:**
```python
    print(f"    ECE: {loss_metrics['ece']:.4f}")
```
**Replace with:**
```python
    print(f"    Positive Brier: {loss_metrics['positive_brier']:.4f}")
```

### 14b. Update test epoch history (lines 11532-11535)

**Find:**
```python
    epoch_history = [
        {'val_loss': 0.5, 'top_10_acc': 0.3},
        {'val_loss': 0.45, 'top_10_acc': 0.35},
        {'val_loss': 0.42, 'top_10_acc': 0.38},
    ]
```
**Replace with:**
```python
    epoch_history = [
        {'val_loss': 0.5, 'recall@10': 0.3},
        {'val_loss': 0.45, 'recall@10': 0.35},
        {'val_loss': 0.42, 'recall@10': 0.38},
    ]
```

### 14c. Update required columns in tests (line 11670)

**Find:**
```python
        'final_train_loss', 'final_val_loss', 'final_top_10_acc', 'final_top_5_acc',
```
**Replace with:**
```python
        'final_train_loss', 'final_val_loss', 'final_val_recall@10', 'final_val_recall@5',
```

### 14d. Update assertions (line 11687)

**Find:**
```python
    assert 0 <= results['final_top_10_acc'] <= 1, f"Top-10 acc out of range: {results['final_top_10_acc']}"
```
**Replace with:**
```python
    assert 0 <= results['final_val_recall@10'] <= 1, f"Recall@10 out of range: {results['final_val_recall@10']}"
```

---

## STEP 15: Update `compute_ablation_metrics()` (lines 6736-6838)

Search for all `final_top_10_acc` and replace with `final_val_recall@10`:

**Example changes:**
```python
    baseline_acc = baseline.get('final_val_recall@10', 0)
    # ...
    flash_acc_gain = flash_dense['final_val_recall@10'] - baseline_acc
    # ...
    pool_acc_gain = flash_learned['final_val_recall@10'] - flash_dense['final_val_recall@10']
    # ...
    moe_acc_gain = moe_standard['final_val_recall@10'] - flash_dense['final_val_recall@10']
    # ...
    acc_gain = results['final_val_recall@10'] - baseline_acc
```

---

## STEP 16: Add Required Import

At the top of the file (in the imports section, around line 430), ensure you have:

```python
from sklearn.metrics import roc_auc_score, average_precision_score
```

If sklearn is already imported elsewhere, you can skip this.

---

## Summary Checklist

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add new metric functions (micro-recall, NDCG, pos-brier, AUROC/AUPRC) | ☐ |
| 2 | Fix MRR to use best-ranked true code | ☐ |
| 3 | Integrate new metrics into `compute_primary_task_metrics()` | ☐ |
| 4 | Update `compute_loss_metrics()` (remove ECE/Brier, add pos-brier) | ☐ |
| 5 | Replace `compute_batch_metrics_lightweight()` entirely | ☐ |
| 6 | Update training print statement | ☐ |
| 7 | Update `train_epoch()` docstring | ☐ |
| 8 | Rename `top_K_acc` → `recall@K` in `evaluate()` (4 locations) | ☐ |
| 9 | Update `_build_epoch_metrics()` | ☐ |
| 10 | Update `_build_final_results()` | ☐ |
| 11 | Update `compute_convergence_metrics()` | ☐ |
| 12 | Update logging statements (2 locations) | ☐ |
| 13 | Update `comprehensive_evaluation()` to include AUROC/AUPRC | ☐ |
| 14 | Update test functions (4+ locations) | ☐ |
| 15 | Update `compute_ablation_metrics()` | ☐ |
| 16 | Add sklearn import | ☐ |

---

## After Making Changes

Run the following to verify:
1. `python -c "from moe_flashattn_3 import *; test_comprehensive_metrics_computation()"` — verify metrics compute
2. Run a small training loop to ensure print statements work
3. Check that `evaluate()` returns the new keys

Would you like me to elaborate on any specific step, or shall I provide more context for any particular function?