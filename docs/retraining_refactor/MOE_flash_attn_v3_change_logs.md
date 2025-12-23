### Mixture-of-Experts (MoE) Experimentation Framework for Hierarchical Clinical Transformer
- Change from moe_flashattn_2.py to moe_flashattn_3.py; what has been changed?
- Version Summary
    - Version 1 & 2 (`moe_flashattn_2.py`)
        - Initial implementation of 5-experiment MoE ablation study
        - Base framework with Flash Attention and MoE integration
        - Single GPU training support
        - Pre-training focused evaluation metrics

    - Version 3 (`moe_flashattn_3.py`)
        - **Distributed Data Parallel (DDP) support** for multi-GPU training
        - **Line of Business (LOB) feature** added as input embedding
        - **Medicaid IP Risk downstream task evaluation** using linear probe methodology
        - **Refactored experiment running** with modular helper functions
        - **Model saving/loading utilities** for inference and deployment

- Experiment Overview

| Experiment | Model | Head Config | Activation | Load Balance | Precision | Daily Encoder |
|------------|-------|-------------|------------|--------------|-----------|---------------|
| **Exp 1: Dense Baseline** | BaselineTransformer | nhead=16, head_dim=16 | GELU only | N/A | FP32 | Standard transformer |
| **Exp 2: Dense Flash** | FlashAttentionTransformer | nhead=8, head_dim=32 | SwiGLU | N/A | FP16 | Flash Attention |
| **Exp 3: Standard Top-K MoE** | FlashMoETransformer | nhead=8, head_dim=32 | SwiGLU + GELU experts | Switch | FP16 | Flash Attention |
| **Exp 4: Shared Expert MoE** | FlashMoETransformer | nhead=8, head_dim=32 | SwiGLU + GELU experts | Switch | FP16 | Flash Attention |
| **Exp 5: Fine-Grained MoE** | FlashMoETransformer | nhead=8, head_dim=32 | SwiGLU + GELU experts | Switch | FP16 | Flash Attention |
| **Exp 6: Auxiliary-Free MoE** | FlashMoETransformer | nhead=8, head_dim=32 | SwiGLU + GELU experts | DeepSeek | FP16 | Flash Attention |

- Experiment Variants

| Experiment Name | Type | Key Features | Rationale|
|----------------|------|--------------|-----------------|
| **Baselines** ||||
| `exp1_dense_baseline` | Dense | Standard Transformer, FP32 | Reference baseline |
| `exp2_dense_flash` | Dense | Flash Attention, Max-Pool | Flash attention baseline |
| `exp2b_flash_learned_pool` | Dense | Flash Attention, Learned Pooling | Best dense model |
| **Standard MoE** ||||
| `exp3_standard_moe` | MoE | 8 experts, top-2, GELU | Basic MoE test |
| `exp3a_moe_swiglu` | MoE | 8 experts, top-2, SwiGLU | SwiGLU vs GELU |
| `exp3b_moe_swiglu_learned_pool` | MoE | + Learned pooling | Best MoE variant |
| `exp3c_moe_swiglu_learned_pool_layer4` | MoE | MoE from layer 4 | Later MoE layers |
| `exp3d_moe_swiglu_learned_pool_layer4_aux001` | MoE | aux_loss=0.001, layer 4 | Lower aux loss |
| `exp3e_moe_swiglu_learned_pool_layer2_aux001` | MoE | aux_loss=0.001, layer 2 | Lower aux loss |
| **Shared Expert MoE** ||||
| `exp4_shared_expert` | MoE | 1 shared + 7 routed | Shared expert test |
| **Fine-grained MoE** ||||
| `exp5_fine_grained` | MoE | 16 experts, top-5, smaller | Fine-grained routing |
| **Auxiliary-free MoE (DeepSeek)** ||||
| `exp6_auxiliary_free` | MoE | DeepSeek balancing, no aux loss | Aux-free MoE |
| `exp6a_auxiliary_free_layer4` | MoE | DeepSeek, MoE from layer 4 | Later DeepSeek |
| `exp6b_auxiliary_free_no-share-exp` | MoE | DeepSeek, no shared experts | Pure DeepSeek |
---

#### Version 3 Modifications

- 1. Line of Business (LOB) Feature Support

**Configuration** (`BaseConfig`):
```python
lob_vocab: int = 4  # LOB categories (0=padding, 1=Commercial, 2=Medicare, 3=Medicaid)
```

**Data Processing**:
- New `conv_lob()` function maps LOB strings to indices
- `ClinicalDataset` now processes LOB column alongside age, gender, codes
- Input tensor dimension: **83** (age, gender, lob, 80 codes) vs **82** in v2

**Model Architecture**:
- All models (`BaselineTransformer`, `FlashAttentionTransformer`, `FlashMoETransformer`) now include:
  ```python
  self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)
  ```
- LOB embedding added to combined representation:
  ```python
  cd = cd_res + cd + gender_cd + age_in_months + lob_emb
  ```

---

- 2a. **Data Parallelism (active for experimentation)**

Automatically enabled when `torch.cuda.device_count() > 1`:

```python
num_gpus = torch.cuda.device_count()
use_data_parallel = num_gpus > 1

if use_data_parallel:
    model = nn.DataParallel(model)
```

**Features**:
| Feature | Implementation |
|---------|----------------|
| Auto-detection | Enables when multiple GPUs available |
| Batch scaling | Effective batch = `batch_size * num_gpus` |
| Learning rate scaling | Square root scaling: `lr * sqrt(num_gpus)` |
| Checkpoint handling | Unwraps `model.module` for compatible saves |

**Scaling Example** (4 GPUs):
- Per-GPU batch size: 32
- Effective batch size: 128
- Base LR: 1e-4 → Scaled LR: 2e-4

**Helper Functions**:
| Function | Purpose |
|----------|---------|
| `save_checkpoint_multigpu()` | Save checkpoint compatible with DataParallel wrapper |
| `load_checkpoint_multigpu()` | Load checkpoint into wrapped or unwrapped model |
| `monitor_gpu_memory_usage()` | Track per-GPU memory for DataParallel |

---
- 2b. **Core DDP Functions (not used)**:
| Function | Description |
|----------|-------------|
| `setup_ddp()` | Initialize DDP, returns (local_rank, world_size, is_main) |
| `cleanup_ddp()` | Clean up distributed process group |
| `is_dist_initialized()` | Check if DDP is initialized |
| `get_world_size()` | Get number of processes (1 if single GPU) |
| `get_rank()` | Get current process rank |
| `is_main_process()` | Check if rank 0 |
| `reduce_tensor()` | Reduce tensor across all processes |
| `sync_metrics()` | Synchronize metrics across processes |

**Utility Functions**:
| Function | Description |
|----------|-------------|
| `print_rank()` | Print message with rank prefix |
| `print_main()` | Print only on main process |
| `barrier_with_timeout()` | Barrier with timeout for hang detection |

**Training Integration**:
- `train_epoch()` now accepts `is_main` and `use_ddp` parameters
- `run_single_experiment()` accepts `local_rank` and `world_size`
- DataLoader worker count scales with world size

---

- 3. Downstream Task Evaluation (Medicaid IP Risk)

**Configuration** (`DownstreamConfig`):
```python
@dataclass
class DownstreamConfig:
    task_name: str = "medicaid_ip_risk"
    test_size: float = 0.1          # 10% for test
    val_size: float = 0.1           # 10% for validation
    random_state: int = 42
    n_cv_folds: int = 5             # Cross-validation folds
    max_iter: int = 1000            # Max LogReg iterations
    class_weight: str = 'balanced'  # Handle class imbalance
```

**DownstreamEvaluator Class**:

| Method | Description |
|--------|-------------|
| `extract_embeddings()` | Extract member-level embeddings from trained transformer |
| `prepare_downstream_data()` | Join features with outcomes, create stratified splits |
| `train_linear_probe()` | Train logistic regression on frozen embeddings |
| `evaluate_probe()` | Compute comprehensive metrics on a data split |
| `evaluate()` | Full pipeline: extract → prepare → train → evaluate |

**Evaluation Metrics**:
- Standard: Accuracy, AUC-ROC, AUC-PR, F1, Precision, Recall, Brier Score
- Top-percentile: Lift@1%, True Positives@1%, Precision@1%, Recall@1%, F1@1%
- Dataset stats: Prevalence, sample counts

---

- 4. Model Saving/Loading Utilities

**Functions**:
| Function | Purpose |
|----------|---------|
| `generate_model_name()` | Standardized naming: `{round}_{exp}_bs{batch}_ep{epochs}_d{embedding}_{timestamp}` |
| `save_trained_model()` | Lightweight save for inference (state dict + config + results) |
| `load_trained_model()` | Load model for inference or downstream evaluation |
| `run_downstream_evaluation()` | Convenience wrapper for downstream evaluation |

**Directory Structure** (post-training):
```
logs/{experiment_round}/{exp_name}/
├── checkpoints/                    # Training resume (save_checkpoint)
│   ├── checkpoint_latest.pt
│   ├── checkpoint_best.pt
│   └── checkpoint_epoch{N}.pt
├── saved_models/                   # Inference (save_trained_model)
│   ├── {model_name}_final.pt
│   ├── {model_name}_config.json
│   └── {model_name}_results.json
├── epoch_metrics.json
├── batch_metrics.json
├── config.json
├── final_results.json
└── {exp_name}.log
```

---

- 5. Refactored Experiment Running

**Helper Functions**:
| Function | Purpose |
|----------|---------|
| `_setup_experiment_directories()` | Set up logging and checkpoint directories |
| `_create_model()` | Create model based on experiment type |
| `_create_dataloaders()` | Create train/val dataloaders with optional bucketing |
| `_resume_from_checkpoint()` | Resume training from checkpoint |
| `_build_epoch_metrics()` | Build comprehensive epoch metrics dictionary |
| `_build_final_results()` | Build final experiment results dictionary |
| `_model_has_moe()` | Check if model has MoE layers |

**Enhanced `run_single_experiment()` Parameters**:
```python
def run_single_experiment(
    # ... existing parameters ...
    local_rank: Optional[int] = None,       # DDP: local GPU rank
    world_size: Optional[int] = None,       # DDP: total GPUs
    outcomes_df: Optional[pd.DataFrame] = None,  # Downstream outcomes
    run_downstream_eval: bool = False,      # Enable downstream evaluation
    save_model: bool = True                 # Save model after training
) -> Dict[str, Any]:
```

---

- 6. Corresponding Test Functions

| Test Function | Coverage |
|---------------|----------|
| `test_conv_lob()` | LOB string conversion |
| `test_clinical_dataset_with_lob()` | Dataset with LOB support |
| `test_model_forward_with_lob()` | Model forward pass with LOB input |
| `test_ddp_initialization()` | DDP initialization on all GPUs |
| `test_model_saving_and_loading()` | Model save/load cycle |
| `test_downstream_evaluator_with_real_data()` | Downstream evaluation pipeline |
| `test_run_single_experiment_with_downstream()` | End-to-end with downstream |

---

#### Data Changes

| Version | Training Data | Validation Data |
|---------|--------------|-----------------|
| v2 | `sample_data/mdcd_train_1m.feather` | `sample_data/mdcd_val_10k.feather` |
| v3 | `sample_data/extrinsic_mdcd_ip/te_pretrain_train.feather` | `sample_data/extrinsic_mdcd_ip/te_pretrain_val_mdcd_ip_probe.feather` |

**Required Columns** (v3):
- Existing: `age_in_months`, `gender_cd`, `cd`, `target_cd`, `dt_cnt`
- New: `lob` (Line of Business)
- For downstream: `individual_id`, `index_dt`, `acute_ip_flag`