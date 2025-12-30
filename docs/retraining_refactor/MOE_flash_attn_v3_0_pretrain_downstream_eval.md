# Complete Usage Guide: Pretraining & Downstream Evaluation
This is a guide for running pretraining and downstream task classification
## Available Experiments

Based on your `get_experiment_configs()`, you have **14 experiments** organized into categories:

| Category | Experiment Name | Key Features |
|----------|-----------------|--------------|
| **Baselines** | `exp1_dense_baseline` | Standard Transformer, FP32 |
|  | `exp2_dense_flash` | Flash Attention, Max-Pool |
|  | `exp2b_flash_learned_pool` | Flash Attention, Learned Pooling |
| **Standard MoE** | `exp3_standard_moe` | 8 experts, top-2, GELU |
|  | `exp3a_moe_swiglu` | 8 experts, top-2, SwiGLU |
|  | `exp3b_moe_swiglu_learned_pool` | + Learned Pooling |
|  | `exp3c_moe_swiglu_learned_pool_layer4` | MoE from layer 4 |
|  | `exp3d_moe_swiglu_learned_pool_layer4_aux001` | aux_loss=0.001, layer 4 |
|  | `exp3e_moe_swiglu_learned_pool_layer2_aux001` | aux_loss=0.001, layer 2 |
| **Shared Expert** | `exp4_shared_expert` | 1 shared + 7 routed |
| **Fine-grained** | `exp5_fine_grained` | 16 experts, top-5, smaller FFN |
| **DeepSeek (Aux-free)** | `exp6_auxiliary_free` | DeepSeek balancing, no aux loss |
|  | `exp6a_auxiliary_free_layer4` | DeepSeek, MoE from layer 4 |
|  | `exp6b_auxiliary_free_no-share-exp` | DeepSeek, no shared experts |

---

## Phase 1: Pretraining

```python
# ============================================================================
# PHASE 1: TRANSFORMER PRETRAINING (Cross-LOB)
# ============================================================================
import torch
import pandas as pd
from moe_flashattn_3 import (
    run_single_experiment,
    run_selected_experiments,
    run_all_experiments,
    get_experiment_configs,
    cleanup_gpu_memory,
)

# ============================================================================
# 1. SETUP
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPUs available: {torch.cuda.device_count()}")

# ============================================================================
# 2. LOAD CROSS-LOB TRAINING DATA
# ============================================================================
# Option A: Load combined data
df_train = pd.read_feather("data/cross_lob_train.feather")
df_val = pd.read_feather("data/cross_lob_val.feather")

# Option B: Combine LOB-specific data
# commercial_train = pd.read_feather("data/commercial_train.feather")
# medicare_train = pd.read_feather("data/medicare_train.feather")
# medicaid_train = pd.read_feather("data/medicaid_train.feather")
# df_train = pd.concat([commercial_train, medicare_train, medicaid_train], ignore_index=True)

print(f"Training samples: {len(df_train):,}")
print(f"Validation samples: {len(df_val):,}")

# ============================================================================
# 3. PRETRAINING OPTIONS
# ============================================================================

# --------------------------------------
# Option 3A: Run a SINGLE experiment
# --------------------------------------
results = run_single_experiment(
    exp_name='exp3b_moe_swiglu_learned_pool',  # Best MoE variant
    moe_config=get_experiment_configs()['exp3b_moe_swiglu_learned_pool'][0],
    use_learnt_att_pool=True,
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=10,
    experiment_round='v1_production',
    embedding_size=256,
    log_dir='logs',
    save_model=True,
)

# Get model path for downstream evaluation
model_path = results['model_path']
print(f"✓ Model saved to: {model_path}")

# --------------------------------------
# Option 3B: Run SELECTED experiments (ablation study)
# --------------------------------------
# Compare baselines vs MoE variants
selected_experiments = [
    'exp2b_flash_learned_pool',          # Best dense baseline
    'exp3b_moe_swiglu_learned_pool',     # Standard MoE
    'exp4_shared_expert',                 # Shared expert
    'exp6_auxiliary_free',                # DeepSeek aux-free
]

comparison_df = run_selected_experiments(
    experiment_names=selected_experiments,
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=10,
    experiment_round='ablation_v1',
    embedding_size=256,
    save_model=True,
)

# View comparison results
print("\n" + "="*80)
print("ABLATION STUDY RESULTS")
print("="*80)
print(comparison_df[['model_path', 'best_val_loss', 'training_time']].to_string())

# --------------------------------------
# Option 3C: Run ALL experiments (full sweep)
# --------------------------------------
all_results_df = run_all_experiments(
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=10,
    experiment_round='full_sweep_v1',
)

# Save comparison table
all_results_df.to_csv('logs/experiment_comparison.csv')
```

---

## Phase 2: Downstream Classification Task

```python
# ============================================================================
# PHASE 2: DOWNSTREAM EVALUATION (Per-LOB)
# ============================================================================
from moe_flashattn_3 import (
    run_downstream_evaluation_from_saved_model,
    run_multi_lob_downstream_evaluation,
    DownstreamConfig,
    LOBData,
)

# ============================================================================
# 1. LOAD PRETRAINED MODEL PATH
# ============================================================================
# Option A: From Phase 1 results
model_path = results['model_path']

# Option B: Specify directly
# model_path = "logs/v1_production/exp3b_moe_swiglu_learned_pool/saved_models/v1_production_exp3b_moe_swiglu_learned_pool_bs32_ep10_d256_20241222_143022.pt"

print(f"Using pretrained model: {model_path}")

# ============================================================================
# 2. LOAD LOB-SPECIFIC DATA
# ============================================================================

# --- COMMERCIAL ---
commercial_features = pd.read_feather("data/commercial/features.feather")
commercial_outcomes = pd.read_feather("data/commercial/outcomes.feather")
# outcomes should have: individual_id, index_dt, acute_ip_flag (or your outcome column)

# --- MEDICARE ---
medicare_features = pd.read_feather("data/medicare/features.feather")
medicare_outcomes = pd.read_feather("data/medicare/outcomes.feather")

# --- MEDICAID ---
medicaid_features = pd.read_feather("data/medicaid/features.feather")
medicaid_outcomes = pd.read_feather("data/medicaid/outcomes.feather")

# ============================================================================
# 3. DOWNSTREAM EVALUATION OPTIONS
# ============================================================================

# --------------------------------------
# Option 3A: Evaluate a SINGLE LOB
# --------------------------------------
downstream_config = DownstreamConfig(
    task_name='medicaid_ip_risk',
    model_type='xgboost',            # 'logistic', 'xgboost', 'lightgbm'
    calibrate_proba=True,            # Isotonic calibration
    test_size=0.1,
    val_size=0.1,
    percentiles=[0.01, 0.10],        # Top 1% and 10% metrics
    lob_name='medicaid',
    outcome_column='acute_ip_flag',  # Column in outcomes_df
)

single_lob_results = run_downstream_evaluation_from_saved_model(
    model_path=model_path,
    features_df=medicaid_features,
    outcomes_df=medicaid_outcomes,
    device=device,
    downstream_config=downstream_config,
    log_dir='logs/downstream/medicaid',
)

print(f"\nMedicaid Results:")
print(f"  Test AUC-ROC: {single_lob_results['test_auc_roc']:.4f}")
print(f"  Test PR-AUC:  {single_lob_results['test_auc_pr']:.4f}")
print(f"  Test F1:      {single_lob_results['test_f1']:.4f}")
print(f"  Precision@10%: {single_lob_results['test_precision_10pct']:.4f}")
print(f"  Recall@10%:    {single_lob_results['test_recall_10pct']:.4f}")
print(f"  Lift@10%:      {single_lob_results['test_lift_10pct']:.2f}x")

# --------------------------------------
# Option 3B: Evaluate MULTIPLE LOBs with different configs
# --------------------------------------
lob_data_list = [
    # Commercial - use XGBoost
    LOBData(
        lob_name='commercial',
        features_df=commercial_features,
        outcomes_df=commercial_outcomes,
        downstream_config=DownstreamConfig(
            task_name='commercial_ip_risk',
            model_type='xgboost',
            calibrate_proba=True,
            outcome_column='acute_ip_flag',
        )
    ),
    # Medicare - use LightGBM
    LOBData(
        lob_name='medicare',
        features_df=medicare_features,
        outcomes_df=medicare_outcomes,
        downstream_config=DownstreamConfig(
            task_name='medicare_ip_risk',
            model_type='lightgbm',
            calibrate_proba=True,
            outcome_column='acute_ip_flag',
        )
    ),
    # Medicaid - use XGBoost
    LOBData(
        lob_name='medicaid',
        features_df=medicaid_features,
        outcomes_df=medicaid_outcomes,
        downstream_config=DownstreamConfig(
            task_name='medicaid_ip_risk',
            model_type='xgboost',
            calibrate_proba=True,
            outcome_column='acute_ip_flag',
        )
    ),
]

# Run multi-LOB evaluation
all_lob_results = run_multi_lob_downstream_evaluation(
    model_path=model_path,
    lob_data_list=lob_data_list,
    device=device,
    log_dir='logs/downstream',
)

# --------------------------------------
# Option 3C: Evaluate multiple LOBs with SAME config (simpler)
# --------------------------------------
base_config = DownstreamConfig(
    model_type='xgboost',
    calibrate_proba=True,
    percentiles=[0.01, 0.05, 0.10],  # Custom percentiles
    outcome_column='acute_ip_flag',
)

lob_data_simple = [
    LOBData(lob_name='commercial', features_df=commercial_features, outcomes_df=commercial_outcomes),
    LOBData(lob_name='medicare', features_df=medicare_features, outcomes_df=medicare_outcomes),
    LOBData(lob_name='medicaid', features_df=medicaid_features, outcomes_df=medicaid_outcomes),
]

all_lob_results = run_multi_lob_downstream_evaluation(
    model_path=model_path,
    lob_data_list=lob_data_simple,
    device=device,
    base_downstream_config=base_config,
    log_dir='logs/downstream',
)

# ============================================================================
# 4. ANALYZE RESULTS
# ============================================================================

# Create comparison DataFrame
comparison_data = []
for lob_name, results in all_lob_results.items():
    if 'error' not in results:
        comparison_data.append({
            'LOB': lob_name,
            'N_Test': results['test_n_samples'],
            'Prevalence': f"{results['test_prevalence']*100:.2f}%",
            'AUC-ROC': f"{results['test_auc_roc']:.4f}",
            'PR-AUC': f"{results['test_auc_pr']:.4f}",
            'F1': f"{results['test_f1']:.4f}",
            'Precision@10%': f"{results['test_precision_10pct']:.4f}",
            'Recall@10%': f"{results['test_recall_10pct']:.4f}",
            'Lift@10%': f"{results['test_lift_10pct']:.1f}x",
            'Brier': f"{results['test_brier']:.4f}",
        })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "="*100)
print("MULTI-LOB DOWNSTREAM EVALUATION SUMMARY")
print("="*100)
print(comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('logs/downstream/multi_lob_comparison.csv', index=False)
```

---

## Complete End-to-End Pipeline Script

```python
#!/usr/bin/env python
"""
Complete Clinical Transformer Training & Evaluation Pipeline
=============================================================

Phase 1: Pretrain transformer on cross-LOB data
Phase 2: Evaluate on LOB-specific downstream tasks

Usage:
    python train_and_evaluate.py
"""

import torch
import pandas as pd
from pathlib import Path
from moe_flashattn_3 import (
    # Pretraining
    run_single_experiment,
    run_selected_experiments,
    get_experiment_configs,
    cleanup_gpu_memory,
    # Downstream
    run_multi_lob_downstream_evaluation,
    DownstreamConfig,
    LOBData,
)

def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================
    EXPERIMENT_NAME = 'exp3b_moe_swiglu_learned_pool'
    EXPERIMENT_ROUND = 'production_v1'
    EPOCHS = 10
    EMBEDDING_SIZE = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running on: {device} ({torch.cuda.device_count()} GPUs)")
    
    # ============================================================
    # PHASE 1: PRETRAINING
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 1: TRANSFORMER PRETRAINING (Cross-LOB)")
    print("="*80)
    
    # Load cross-LOB training data
    df_train = pd.read_feather("data/cross_lob_train.feather")
    df_val = pd.read_feather("data/cross_lob_val.feather")
    print(f"Training: {len(df_train):,} samples")
    print(f"Validation: {len(df_val):,} samples")
    
    # Get experiment config
    moe_config, use_learnt_att_pool = get_experiment_configs()[EXPERIMENT_NAME]
    
    # Run pretraining
    pretrain_results = run_single_experiment(
        exp_name=EXPERIMENT_NAME,
        moe_config=moe_config,
        use_learnt_att_pool=use_learnt_att_pool,
        train_data=df_train,
        val_data=df_val,
        device=device,
        epochs=EPOCHS,
        experiment_round=EXPERIMENT_ROUND,
        embedding_size=EMBEDDING_SIZE,
        save_model=True,
    )
    
    model_path = pretrain_results['model_path']
    print(f"\n✓ Pretraining complete!")
    print(f"  Model saved to: {model_path}")
    print(f"  Best val loss: {pretrain_results.get('best_val_loss', 'N/A')}")
    
    # Cleanup GPU memory before downstream
    cleanup_gpu_memory()
    
    # ============================================================
    # PHASE 2: DOWNSTREAM EVALUATION
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 2: DOWNSTREAM EVALUATION (Per-LOB)")
    print("="*80)
    
    # Prepare LOB-specific data
    lob_data_list = []
    
    for lob_name in ['commercial', 'medicare', 'medicaid']:
        features_path = f"data/{lob_name}/features.feather"
        outcomes_path = f"data/{lob_name}/outcomes.feather"
        
        if Path(features_path).exists() and Path(outcomes_path).exists():
            lob_data_list.append(LOBData(
                lob_name=lob_name,
                features_df=pd.read_feather(features_path),
                outcomes_df=pd.read_feather(outcomes_path),
                downstream_config=DownstreamConfig(
                    task_name=f'{lob_name}_ip_risk',
                    model_type='xgboost',
                    calibrate_proba=True,
                    percentiles=[0.01, 0.05, 0.10],
                    outcome_column='acute_ip_flag',
                    lob_name=lob_name,
                )
            ))
            print(f"  ✓ Loaded {lob_name} data")
        else:
            print(f"  ⚠ Skipping {lob_name} (data not found)")
    
    # Run multi-LOB evaluation
    all_results = run_multi_lob_downstream_evaluation(
        model_path=model_path,
        lob_data_list=lob_data_list,
        device=device,
        log_dir=f'logs/downstream/{EXPERIMENT_ROUND}',
    )
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Pretrained model: {model_path}")
    print(f"\nDownstream Results:")
    for lob_name, results in all_results.items():
        if 'error' not in results:
            print(f"  {lob_name:12s}: AUC-ROC={results['test_auc_roc']:.4f}, "
                  f"Lift@10%={results['test_lift_10pct']:.1f}x")


if __name__ == "__main__":
    main()
```

---

## Key Metrics Explained

After downstream evaluation, you'll get these metrics for each LOB:

| Metric | Key | Description |
|--------|-----|-------------|
| **AUC-ROC** | `test_auc_roc` | Discrimination ability (0.5=random, 1.0=perfect) |
| **PR-AUC** | `test_auc_pr` | Area under Precision-Recall curve (better for imbalanced) |
| **F1** | `test_f1` | Harmonic mean of precision & recall |
| **Brier Score** | `test_brier` | Calibration quality (lower is better) |
| **Precision@10%** | `test_precision_10pct` | Of top 10% predictions, % that are true positives |
| **Recall@10%** | `test_recall_10pct` | Of all positives, % captured in top 10% |
| **Lift@10%** | `test_lift_10pct` | How much better than random (e.g., 3.0x = 3x better) |