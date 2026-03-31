# Unify moe_flashattn_4_core.py with Training Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate all functions needed by `exp_round10_training_inference_headoff.ipynb` into a single `moe_flashattn_4_core.py` so the notebook imports from ONE module instead of two (`_core.py` + `moe_flashattn_4.py`).

**Architecture:** The current `_core.py` (3993 lines) contains model architecture, configs, data parsing, embedding extraction, and downstream evaluation. The notebook currently imports `OptimizeConfig`, `run_single_experiment`, `prepare_data_once`, `setup_experiment_logging`, and `ClinicalDatasetLazy` from the full `moe_flashattn_4.py` (18468 lines). We will port ONLY these 5 functions and their transitive dependencies into `_core.py`, then update the notebook to import everything from `_core.py` alone.

**Tech Stack:** Python, PyTorch, dataclasses, pandas, numpy, sklearn

---

## Current State Analysis

### What the notebook imports today

**From `moe_flashattn_4_core.py` (already there):**
- `BaseConfig`, `FlashAttentionConfig`, `MoEConfig`
- `get_experiment_configs`
- `conv_cd`, `conv_age_gender`, `conv_lob`, `conv_target`
- `ClinicalDataset`, `create_collate_fn`
- `FlashAttentionTransformer`, `FlashMoETransformer`, `BaselineTransformer`
- `DataParallelWrapper`
- `EmbeddingExtractor`
- `cleanup_gpu_memory`

**From `moe_flashattn_4.py` (needs to move to core):**
- `OptimizeConfig` (superset with 11 extra fields vs core version)
- `run_single_experiment`
- `prepare_data_once`
- `setup_experiment_logging`
- `ClinicalDatasetLazy`

### Transitive dependencies of the 5 functions

`run_single_experiment` calls these, which are NOT in `_core.py`:
- `setup_experiment_logging`
- `_setup_experiment_directories`
- `_calculate_model_dimensions`
- `_create_model`
- `_create_dataloaders`
- `_resume_from_checkpoint`
- `_build_epoch_metrics`
- `_build_final_results`
- `create_criterion` (+ `AsymmetricLoss`, `FocalLoss`, weight helpers)
- `create_optimizer`
- `create_scheduler` (+ `get_linear_warmup_plateau_decay`, `get_cosine_schedule_with_warmup`)
- `train_epoch`
- `evaluate` (standalone, NOT `DownstreamEvaluator.evaluate`)
- `comprehensive_evaluation`
- `compute_embedding_quality_epoch`
- `save_checkpoint`
- `cleanup_checkpoints_after_training`
- `LossTracker`
- `GradientTierAnalyzer`
- `build_tier_indices_streaming`
- `build_density_pools_streaming`
- `StreamingMetrics` (+ `StreamingMetricsState`)
- `TierAwareBatchSampler`
- `DensityTierAwareBatchSampler`
- `BucketingBatchSampler`
- `prepare_tensor`, `create_multihot_targets_vectorized`
- `compute_pos_weights`, `compute_log_scaled_weights`, `compute_effective_number_weights`, `compute_tiered_weights`
- `_forward_batch`, `_update_streaming_metrics`, `_get_empty_eval_metrics`
- Various metric compute functions
- `compute_batch_metrics_lightweight`, `compute_moe_batch_metrics`, etc.

`prepare_data_once` additionally calls:
- `_compute_code_frequencies_from_dataset`
- `_compute_code_frequencies_from_strings`

### What needs to change in `_core.py`

1. **Update `OptimizeConfig`** — add 11 missing fields (tier batching, density batching, ASL)
2. **Add `ClinicalDatasetLazy`** — new class after `ClinicalDataset`
3. **Add `setup_experiment_logging`** — new function
4. **Add all training pipeline functions** — the complete training pipeline from `moe_flashattn_4.py`
5. **Add all training helper classes** — `AsymmetricLoss`, `FocalLoss`, `LossTracker`, `GradientTierAnalyzer`, `StreamingMetrics`, batch samplers, etc.

### What needs to change in the notebook

1. Remove the `from moe_flashattn_4 import (...)` block
2. Add `OptimizeConfig`, `run_single_experiment`, `prepare_data_once`, `setup_experiment_logging`, `ClinicalDatasetLazy` to the existing `from moe_flashattn_4_core import (...)` block

---

## Tasks

### Task 1: Update `OptimizeConfig` in `_core.py`

**Files:**
- Modify: `dev/moe/moe_flashattn_4_core.py:210-268`

**Step 1: Add the 11 missing fields to OptimizeConfig**

The current `OptimizeConfig` ends at line 268 (after `override_gradient_clip`). Add the tier-aware batching, density-aware batching, and ASL fields from the notebook version.

Add after `override_gradient_clip: Optional[float] = None` (line 267):

```python
    # ============================================================
    # TIER-AWARE BATCHING
    # ============================================================
    use_tier_aware_batching: bool = False
    tier_medium_quota: int = 0
    tier_rare_quota: int = 8
    tier_tail_quota: int = 10

    # ============================================================
    # DENSITY-AWARE TIER BATCHING
    # ============================================================
    use_density_aware_batching: bool = False
    density_tail_percentile: float = 80.0
    density_rare_percentile: float = 70.0
    density_medium_percentile: float = 70.0

    # ============================================================
    # ASYMMETRIC LOSS (Ridnik et al. 2021)
    # ============================================================
    use_asl: bool = False
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
```

**Step 2: Verify the change**

Run: `python -c "from dev.moe.moe_flashattn_4_core import OptimizeConfig; oc = OptimizeConfig(); print(oc.use_asl, oc.use_tier_aware_batching, oc.use_density_aware_batching)"`
Expected: `False False False`

---

### Task 2: Add `ClinicalDatasetLazy` to `_core.py`

**Files:**
- Modify: `dev/moe/moe_flashattn_4_core.py` — insert after `ClinicalDataset` class (after line ~960, before `clinical_collate_fn`)

**Step 1: Add the ClinicalDatasetLazy class**

Port the class from `moe_flashattn_4.py:3392-3480`. This class depends on `conv_age_gender`, `conv_cd`, `conv_lob`, `conv_target` (all already in `_core.py`), `Dataset`, `pd`, `torch`, `time`.

```python
class ClinicalDatasetLazy(Dataset):
    """
    Memory-efficient Dataset: stores raw strings, parses on-the-fly in __getitem__.
    
    For 11M samples:
      - ClinicalDataset:     ~888 GB RAM (pre-allocated tensors + targets lists)
      - ClinicalDatasetLazy: ~130 GB RAM (raw strings only)
    
    Interface contract: __getitem__ returns identical dict as ClinicalDataset,
    so collate_fn, DataLoader, and training loop require zero changes.
    """
    def __init__(self, df: pd.DataFrame, config: BaseConfig):
        self.config = config
        self.n = len(df)
        
        print(f"ClinicalDatasetLazy: Storing {self.n:,} samples as raw strings (lazy parsing)...")
        start = time.time()
        
        self.age_strs = df['age_in_months'].tolist()
        self.gender_strs = df['gender_cd'].tolist()
        self.cd_strs = df['cd'].tolist()
        self.target_strs = df['target'].tolist()
        self.dt_cnt = df['dt_cnt'].tolist()
        self.lob_strs = df['lob'].tolist()
        
        sample_size = min(1000, self.n)
        avg_cd_len = sum(
            len(str(s)) if s and not pd.isna(s) else 0
            for s in self.cd_strs[:sample_size]
        ) / max(sample_size, 1)
        est_gb = (avg_cd_len * self.n * 1.5) / 1e9
        
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s. Estimated string memory: ~{est_gb:.1f} GB")
        print(f"  Parsing will happen on-the-fly in __getitem__ (parallelized by DataLoader workers)")
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        config = self.config
        return {
            'age': torch.tensor(
                conv_age_gender(self.age_strs[idx], config.len_dy), dtype=torch.int16
            ),
            'gender': torch.tensor(
                conv_age_gender(self.gender_strs[idx], config.len_dy, max_val=3), dtype=torch.int8
            ),
            'lob': torch.tensor(
                conv_lob(self.lob_strs[idx], config.len_dy), dtype=torch.int8
            ),
            'codes': torch.tensor(
                conv_cd(self.cd_strs[idx], config.len_dy, config.len_cd), dtype=torch.int32
            ),
            'dt_cnt': self.dt_cnt[idx],
            'target': conv_target(
                self.target_strs[idx], config.len_dy, config.target_cd_cnt
            )
        }
    
    def get_target_codes_for_member(self, idx: int) -> set:
        target_str = self.target_strs[idx]
        if not target_str or pd.isna(target_str):
            return set()
        
        codes = set()
        for day_str in target_str.split('*')[:self.config.len_dy]:
            if not day_str:
                continue
            for code_str in day_str.split(','):
                try:
                    code_val = int(code_str) if code_str else 0
                    if 0 < code_val <= self.config.target_cd_cnt:
                        code_idx = code_val - 1
                        if code_idx == 0:
                            continue
                        codes.add(code_idx)
                except ValueError:
                    pass
        return codes
```

**Step 2: Verify**

Run: `python -c "from dev.moe.moe_flashattn_4_core import ClinicalDatasetLazy; print('ClinicalDatasetLazy imported OK')"`
Expected: `ClinicalDatasetLazy imported OK`

---

### Task 3: Add `setup_experiment_logging` to `_core.py`

**Files:**
- Modify: `dev/moe/moe_flashattn_4_core.py` — insert in a new "LOGGING" section before `get_experiment_configs`

**Step 1: Add the function**

Port from `moe_flashattn_4.py:294-340`. Dependencies: `logging`, `Path`, `datetime` — all already imported in `_core.py`.

```python
def setup_experiment_logging(
    exp_name: str,
    log_dir: str = "logs",
    resume: bool = False
) -> logging.Logger:
    log_path = Path(log_dir) / exp_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    file_mode = 'a' if resume else 'w'
    file_handler = logging.FileHandler(log_path / 'training.log', mode=file_mode)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    if resume:
        logger.info(f"\n{'='*80}")
        logger.info(f"  TRAINING RESUMED")
        logger.info(f"Resume time: {datetime.now()}")
        logger.info(f"{'='*80}\n")
    
    return logger
```

---

### Task 4: Add all training pipeline functions and classes to `_core.py`

**Files:**
- Modify: `dev/moe/moe_flashattn_4_core.py` — add a new major section "TRAINING PIPELINE" after the existing "SAVE/LOAD" section and before the "DOWNSTREAM EVALUATION" section

**Step 1: Port training helper classes and functions**

Copy from `moe_flashattn_4.py` the following in order (preserving their implementations exactly):

**Loss functions and weight helpers (from moe_flashattn_4.py):**
1. `compute_pos_weights` 
2. `compute_log_scaled_weights`
3. `compute_effective_number_weights`
4. `compute_tiered_weights`
5. `class AsymmetricLoss` (line 4295)
6. `class FocalLoss` (line 4401)
7. `create_criterion` (line 4692)

**Training utilities:**
8. `class LossTracker` (line 4782)
9. `class GradientTierAnalyzer` (line 5198)
10. `prepare_tensor`
11. `create_multihot_targets_vectorized`

**Scheduler/optimizer:**
12. `get_linear_warmup_plateau_decay`
13. `get_cosine_schedule_with_warmup`
14. `create_scheduler` (line 5023)
15. `create_optimizer` (line 5119)

**Batch samplers:**
16. `class TierAwareBatchSampler` (line 5962)
17. `class DensityTierAwareBatchSampler` (line 6543)
18. `class BucketingBatchSampler` (line 6937)

**Training loop:**
19. `train_epoch` (line 5476)

**Streaming metrics:**
20. `class StreamingMetricsState` (line 9878)
21. `class StreamingMetrics` (line 9910)

**Evaluation functions:**
22. `_get_empty_eval_metrics`
23. `_forward_batch`
24. `_update_streaming_metrics`
25. `evaluate` (standalone, line 7056)
26. `compute_batch_metrics_lightweight`
27. `compute_embedding_quality_epoch` (line 8232)
28. All metric compute functions used by `comprehensive_evaluation`:
    - `compute_moe_batch_metrics`, `compute_router_gradient_metrics`
    - `compute_moe_performance_metrics`, `compute_memory_metrics`
    - `compute_flops_metrics`, `compute_cost_metrics`
    - `compute_training_time_metrics`, `compute_primary_task_metrics`
    - `compute_loss_metrics`, `compute_micro_recall_at_k`
    - `compute_ndcg_at_k`, `compute_positive_brier_score`
    - `compute_auroc_auprc`, `compute_stratified_metrics`
    - `compute_convergence_metrics`, `compute_ablation_metrics`
29. `comprehensive_evaluation` (line 9592)

**Checkpoint management:**
30. `save_checkpoint` (line 7383)
31. `cleanup_checkpoints_after_training` (line 14074)

**Experiment orchestration helpers:**
32. `_calculate_model_dimensions` (line 12037)
33. `_setup_experiment_directories` (line 12105)
34. `_create_model` (line 12124)
35. `_compute_code_frequencies_from_dataset` (line 12302)
36. `_compute_code_frequencies_from_strings` (line 12349)
37. `build_tier_indices_streaming` (line 12401)
38. `build_density_pools_streaming` (line 12473)
39. `_create_dataloaders` (line 12623)
40. `_resume_from_checkpoint` (line 12788)
41. `_build_epoch_metrics` (line 12833)
42. `_build_final_results` (line 12909)

**Top-level experiment runners:**
43. `prepare_data_once` (line 12206)
44. `run_single_experiment` (line 12963)

**Step 2: Add any missing imports**

The training pipeline needs these additional imports beyond what `_core.py` already has:
- `import shutil` (used by `cleanup_checkpoints_after_training`)
- `from torch.utils.data import Sampler` (used by batch samplers)
- `import copy` (if used by `_create_model`)

**Step 3: Fix known bugs**

In `_resume_from_checkpoint`: the notebook version uses `checkpoint.get(...)` but the loaded dict is named `checkpoint_data`. Fix to use `checkpoint_data.get(...)` consistently.

**Step 4: Verify module imports**

Run: `python -c "from dev.moe.moe_flashattn_4_core import run_single_experiment, prepare_data_once, setup_experiment_logging, ClinicalDatasetLazy, OptimizeConfig; print('All 5 target imports OK')"`
Expected: `All 5 target imports OK`

---

### Task 5: Update `exp_round10_training_inference_headoff.ipynb` imports

**Files:**
- Modify: `dev/moe/exp_round10_training_inference_headoff.ipynb` — Cell 2 (the import cell)

**Step 1: Replace the dual-import with a single import block**

Current notebook has two import blocks:

```python
from moe_flashattn_4_core import (
    BaseConfig, FlashAttentionConfig, MoEConfig,
    get_experiment_configs,
    conv_cd, conv_age_gender, conv_lob, conv_target,
    ClinicalDataset, create_collate_fn,
    FlashAttentionTransformer, FlashMoETransformer, BaselineTransformer,
    DataParallelWrapper,
    EmbeddingExtractor,
    cleanup_gpu_memory,
)

from moe_flashattn_4 import (
    OptimizeConfig,
    run_single_experiment,
    prepare_data_once,
    setup_experiment_logging,
    ClinicalDatasetLazy,
)
```

Replace with a SINGLE import block:

```python
from moe_flashattn_4_core import (
    # Configs
    BaseConfig,
    FlashAttentionConfig,
    MoEConfig,
    OptimizeConfig,
    # Experiment configs
    get_experiment_configs,
    # Data parsing utilities
    conv_cd,
    conv_age_gender,
    conv_lob,
    conv_target,
    ClinicalDataset,
    ClinicalDatasetLazy,
    create_collate_fn,
    # Model architecture
    FlashAttentionTransformer,
    FlashMoETransformer,
    BaselineTransformer,
    DataParallelWrapper,
    # Embedding extraction
    EmbeddingExtractor,
    # Training pipeline
    setup_experiment_logging,
    prepare_data_once,
    run_single_experiment,
    # Utilities
    cleanup_gpu_memory,
)
```

**Step 2: Remove the comments explaining the dual-import strategy**

Remove these comment blocks:
- `# Training pipeline imports — training orchestration + OptimizeConfig`
- `# Source: dev/moe/moe_flashattn_4.py`
- `# Note: OptimizeConfig is imported from moe_flashattn_4 (not _core) because...`

**Step 3: Update the header comment**

Update: `# Source: dev/moe/moe_flashattn_4_core.py` to be the only source reference.

---

### Task 6: Verify and Dry-Run Test

**Files:**
- Read: `dev/moe/moe_flashattn_4_core.py` (modified)
- Read: `dev/moe/exp_round10_training_inference_headoff.ipynb` (modified)

**Step 1: Syntax check the core module**

Run: `python -m py_compile dev/moe/moe_flashattn_4_core.py`
Expected: No output (success)

**Step 2: Verify all notebook imports resolve**

Run: `cd dev/moe && python -c "
from moe_flashattn_4_core import (
    BaseConfig, FlashAttentionConfig, MoEConfig, OptimizeConfig,
    get_experiment_configs,
    conv_cd, conv_age_gender, conv_lob, conv_target,
    ClinicalDataset, ClinicalDatasetLazy, create_collate_fn,
    FlashAttentionTransformer, FlashMoETransformer, BaselineTransformer,
    DataParallelWrapper, EmbeddingExtractor,
    setup_experiment_logging, prepare_data_once, run_single_experiment,
    cleanup_gpu_memory,
)
print('All imports successful')

# Verify OptimizeConfig has new fields
oc = OptimizeConfig()
assert hasattr(oc, 'use_asl'), 'Missing use_asl'
assert hasattr(oc, 'use_tier_aware_batching'), 'Missing use_tier_aware_batching'
assert hasattr(oc, 'use_density_aware_batching'), 'Missing use_density_aware_batching'
print('OptimizeConfig fields verified')

# Verify function signatures
import inspect
sig = inspect.signature(run_single_experiment)
assert 'optimize_config' in sig.parameters, 'Missing optimize_config param'
assert 'prepared_data' in sig.parameters, 'Missing prepared_data param'
print('run_single_experiment signature verified')

sig = inspect.signature(prepare_data_once)
assert 'use_lazy' in sig.parameters, 'Missing use_lazy param'
print('prepare_data_once signature verified')

print('ALL CHECKS PASSED')
"`

Expected: `ALL CHECKS PASSED`

**Step 3: Verify no circular imports or missing references**

Run: `cd dev/moe && python -c "
import moe_flashattn_4_core
attrs = dir(moe_flashattn_4_core)
required = [
    'BaseConfig', 'FlashAttentionConfig', 'MoEConfig', 'OptimizeConfig',
    'ClinicalDataset', 'ClinicalDatasetLazy', 'PreparedData',
    'FlashAttentionTransformer', 'FlashMoETransformer', 'BaselineTransformer',
    'DataParallelWrapper', 'EmbeddingExtractor', 'MetricsLogger',
    'setup_experiment_logging', 'prepare_data_once', 'run_single_experiment',
    'create_criterion', 'create_optimizer', 'create_scheduler',
    'train_epoch', 'evaluate', 'comprehensive_evaluation',
    'save_checkpoint', 'cleanup_gpu_memory',
    'LossTracker', 'GradientTierAnalyzer', 'StreamingMetrics',
]
missing = [r for r in required if r not in attrs]
if missing:
    print(f'MISSING: {missing}')
else:
    print('All required symbols exported')
"`

Expected: `All required symbols exported`

---

## Execution Notes

- The `moe_flashattn_4.py` file (18468 lines) serves as the **source of truth** for function implementations. All ported code should match that file exactly (with the `_resume_from_checkpoint` bug fix applied).
- After this unification, the notebook's runtime behavior is UNCHANGED — only the import source changes.
- The `moe_flashattn_4.py` file itself is NOT modified. It remains as the full notebook export. Only `_core.py` and the `exp_round10` notebook change.
- The `_core.py` will grow significantly but this is intentional — it becomes the single-file handoff artifact for data scientists.
