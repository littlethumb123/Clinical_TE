#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


# In[26]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # preset only 2 to the embedding generation and leave 2 for other works
import torch


# In[27]:


# Import everything from the core module
from moe_flashattn_3_core import (
    # Configurations
    BaseConfig,
    FlashAttentionConfig,
    MoEConfig,
    DownstreamConfig,
    
    # Models
    BaselineTransformer,
    FlashAttentionTransformer,
    FlashMoETransformer,
    
    # Data utilities
    ClinicalDataset,
    create_collate_fn,
    conv_cd,
    conv_age_gender,
    conv_lob,
    conv_target,
    
    # Embedding extraction
    EmbeddingExtractor,
    DownstreamEvaluator,
    
    # Model loading
    load_trained_model,
    get_experiment_configs,
    
    # GPU utilities
    cleanup_gpu_memory,
    cleanup_gpu_memory_hard,
    
    # Downstream evaluation
    run_downstream_evaluation_from_saved_model,
    run_multi_lob_downstream_evaluation,
    LOBData,
    
    # Logging
    MetricsLogger,
)
from torch.utils.data import DataLoader


# #### Small test

# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "logs/exp_round5_3lobs_pretrain_multi_gpu_test_v2/exp6_auxiliary_free_v3/saved_models/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_20251231_152438_final.pt"
checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)


# In[14]:


checkpoint_data.keys()


# In[7]:


print(f"Model type: {checkpoint_data.get('model_type')}")
print(f"Epoch: {checkpoint_data.get('moe_config')}")
print(f"Global step: {checkpoint_data.get('config')}")


# In[38]:


# Load and inspect the checkpoint
model_path = "logs/exp_round5_3lobs_pretrain_multi_gpu_test_v2/exp6_auxiliary_free_v3/saved_models/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_20251231_152438_final.pt"

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print("="*80)
print("CHECKPOINT CONTENTS:")
print("="*80)

print("\n1. Top-level keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

print("\n2. Config dict:")
config_dict = checkpoint.get('config', {})
for k, v in config_dict.items():
    print(f"  {k}: {v}")

print("\n3. MoE config dict:")
moe_config_dict = checkpoint.get('moe_config', None)
if moe_config_dict:
    for k, v in moe_config_dict.items():
        print(f"  {k}: {v}")
else:
    print("  ⚠️ moe_config is None!")

print("\n4. Model type:", checkpoint.get('model_type'))

# Calculate expected d_ff_adjusted for verification
if moe_config_dict and 'd_ff' in moe_config_dict:
    d_ff = moe_config_dict['d_ff']
    d_ff_adjusted = int((2 * d_ff) / 3)
    print(f"\n5. Expected d_ff_adjusted from checkpoint: {d_ff_adjusted}")

# Also check the actual weight shapes
print("\n6. Sample weight shapes from state_dict:")
state_dict = checkpoint['model_state_dict']
for key in list(state_dict.keys())[:10]:
    print(f"  {key}: {state_dict[key].shape}")

# Check one of the problematic layers
if 'temporal_layers.2.ffn.experts.0.ffn.w_gate.weight' in state_dict:
    shape = state_dict['temporal_layers.2.ffn.experts.0.ffn.w_gate.weight'].shape
    print(f"\n7. Expert FFN weight shape: {shape}")
    print(f"   This implies d_ff_adjusted = {shape[0]}")
    print(f"   This implies d_ff = {int(shape[0] * 3 / 2)}")


# ### Intrinsic metrics eval

# In[159]:


import json
import pandas as pd
pd.set_option('display.max_rows', None)
from pathlib import Path
from typing import Dict, List, Union, Optional

def extract_experiment_metrics(json_paths: Union[str, List[str]]) -> pd.DataFrame:
    """
    Extract all metrics from experiment result JSON files into a flat DataFrame.
    
    Handles:
    - Top-level summary metrics
    - full_evaluation.performance metrics
    - full_evaluation.efficiency metrics  
    - full_evaluation.resources metrics
    - all_epochs metrics (from final epoch)
    
    Args:
        json_paths: Single path or list of paths to result JSON files
        
    Returns:
        DataFrame with one row per experiment, all metrics as columns
    """
    if isinstance(json_paths, str):
        json_paths = [json_paths]
    
    all_records = []
    
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        
        record = {}
        
        # ==================================================================
        # 1. TOP-LEVEL SUMMARY METRICS
        # ==================================================================
        top_level_keys = [
            'experiment', 'parameters', 'use_learned_pooling', 'use_bucketing',
            'train_loss_mean', 'train_loss_learned', 'train_loss_final',
            'val_loss_final', 'generalization_gap', 'training_time_sec',
            'final_train_recall@5', 'final_train_recall@10', 'final_train_recall@20',
            'final_val_recall@5', 'final_val_recall@10', 'final_val_recall@20',
            'final_val_micro_recall@10', 'final_val_ndcg@20', 'final_val_mrr',
            'final_val_positive_brier',
            'precision@10', 'recall@10', 'f1@10', 'micro_recall@10', 'ndcg@10',
            'balanced_top10_acc', 'tail_top10_acc', 'cost_usd', 'peak_memory_gb',
            'model_name'
        ]
        for key in top_level_keys:
            if key in data:
                record[key] = data[key]
        
        # ==================================================================
        # 2. full_evaluation.performance METRICS
        #    Source: comprehensive_evaluation() -> StreamingMetrics + detailed metrics
        # ==================================================================
        if 'full_evaluation' in data and 'performance' in data['full_evaluation']:
            perf = data['full_evaluation']['performance']
            for key, value in perf.items():
                record[f'perf_{key}'] = value
        
        # ==================================================================
        # 3. full_evaluation.efficiency METRICS
        #    Source: compute_training_time_metrics()
        # ==================================================================
        if 'full_evaluation' in data and 'efficiency' in data['full_evaluation']:
            eff = data['full_evaluation']['efficiency']
            for key, value in eff.items():
                record[f'eff_{key}'] = value
        
        # ==================================================================
        # 4. full_evaluation.resources METRICS
        #    Source: compute_memory_metrics() + compute_cost_metrics() + compute_flops_metrics()
        # ==================================================================
        if 'full_evaluation' in data and 'resources' in data['full_evaluation']:
            res = data['full_evaluation']['resources']
            for key, value in res.items():
                record[f'res_{key}'] = value
        
        # ==================================================================
        # 5. all_epochs METRICS (from final epoch)
        #    Source: _build_epoch_metrics() - includes training trajectory,
        #    eval-in-train, validation, MoE, and router metrics
        # ==================================================================
        if 'all_epochs' in data and len(data['all_epochs']) > 0:
            final_epoch = data['all_epochs'][-1]  # Get last epoch
            for key, value in final_epoch.items():
                record[f'epoch_{key}'] = value
        
        record['source_file'] = Path(path).name
        all_records.append(record)
    
    df = pd.DataFrame(all_records)
    
    # Reorder columns for readability
    priority_cols = ['experiment', 'source_file', 'parameters']
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + sorted(other_cols)]
    
    return df


# In[16]:


# Check the experiment round 5 intrinsic metrics
paths = [
    "logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp1_dense_baseline_pure_legacy/saved_models/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp1_dense_baseline_bs128_ep1_d256_20251230_055716_results.json",
    "logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp1_dense_baseline_opt_config/saved_models/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2_exp1_dense_baseline_bs64_ep1_d256_20260108_183616_results.json",
    "logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp2b_flash_learned_pool_v2/saved_models/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20251230_114137_results.json",
    "logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp6_auxiliary_free_v3/saved_models/exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_20251231_152438_results.json"
]
df_exp_round5_results = extract_experiment_metrics(paths)


# In[8]:


df_exp_round5_results.T.to_excel("experiment_logs/exp_round5_3lobs_1-5M_1epoch_32batch_dim256_pretrain_multi_gpu_test_v2_intrinsic.xlsx")


# ### Model reconstruction

# In[28]:


def load_model_from_checkpoint(
    model_path: str,
    device: torch.device,
    verbose: bool = True
) -> Tuple[torch.nn.Module, BaseConfig, Optional[MoEConfig]]:
    """
    Load a pretrained model from a checkpoint file.
    'model_state_dict', 'model_name', 'model_type', 'embedding_size', 'nlayers', 
    'checkpoint_dir', 'timestamp', 'config', 'moe_config'
    
    Args:
        model_path: Path to the .pt checkpoint file
        device: Torch device to load the model onto
        verbose: Whether to print loading details
        
    Returns:
        Tuple of (model, config, moe_config)
        - model: Loaded and initialized model in eval mode
        - config: Reconstructed FlashAttentionConfig or BaseConfig
        - moe_config: MoEConfig if model is MoE, else None
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading model from: {model_path}")
    checkpoint_data = torch.load(model_path, map_location=device, weights_only=False)

    model_type = checkpoint_data.get('model_type', 'Unknown')
    config_dict = checkpoint_data.get('config', {})
    moe_config_dict = checkpoint_data.get('moe_config', None)
    
    state_dict = checkpoint_data['model_state_dict']
    
    if verbose:
        print(f"  Model type: {model_type}")
        print(f"  Embedding size: {config_dict.get('embedding_size', 256)}")
        print(f"  N layers: {config_dict.get('nlayers', 6)}")
        print(f"  Use learned attention pooling: {config_dict.get('use_learnt_att_pool', False)}")

    use_learnt_att_pool_inferred = 'daily_pooling.query' in state_dict    
    inferred_d_ff = None
    if 'FlashMoE' in model_type:
        # Try to infer d_ff from expert weight shapes
        # Look for: temporal_layers.{layer}.ffn.experts.0.ffn.w_gate.weight
        for key in state_dict.keys():
            if 'experts.0.ffn.w_gate.weight' in key:
                weight_shape = state_dict[key].shape
                d_ff_adjusted = weight_shape[0]  # Shape is [d_ff_adjusted, d_model]
                # Reverse the SwiGLU adjustment: d_ff = d_ff_adjusted * 3 / 2
                inferred_d_ff = (d_ff_adjusted * 3 + 1) // 2
                if verbose:
                    print(f"Inferred d_ff from expert weights: {inferred_d_ff} "
                          f"(d_ff_adjusted={d_ff_adjusted})")
                break
        
        # Alternative: use config.nhid which was correct
        if inferred_d_ff is None:
            inferred_d_ff = config_dict.get('nhid', 512)
            if verbose:
                print(f"Using nhid as d_ff fallback: {inferred_d_ff}")

    # For non-MoE FlashAttention models, infer nhid from dense FFN weight shapes
    inferred_nhid = None
    if 'FlashMoE' not in model_type:
        for key in state_dict.keys():
            if 'temporal_layers.0.ffn.w_gate.weight' in key:
                d_ff_adjusted = state_dict[key].shape[0]
                inferred_nhid = (d_ff_adjusted * 3 + 1) // 2
                if verbose:
                    print(f"  Inferred nhid from FFN weights: {inferred_nhid} "
                          f"(d_ff_adjusted={d_ff_adjusted})")
                break

    # Reconstruct config based on model type
    moe_config = None
    
    if 'FlashMoE' in model_type:
        # MoE model with Flash Attention
        config = FlashAttentionConfig(
            embedding_size=config_dict.get('embedding_size', 256),
            nhid=config_dict.get('nhid', 512),
            nhead=config_dict.get('nhead', 8),
            nlayers=config_dict.get('nlayers', 6),
            dropout=config_dict.get('dropout', 0.1),
            use_learnt_att_pool=use_learnt_att_pool_inferred,
            use_swiglu=config_dict.get('use_swiglu', True),
            use_rope=config_dict.get('use_rope', True),
            use_flash=config_dict.get('use_flash', True),
        )
        
        if moe_config_dict:
            d_ff_to_use = inferred_d_ff or config_dict.get('nhid', 512)
            
            if verbose and moe_config_dict.get('d_ff') != d_ff_to_use:
                print(f"⚠️ Correcting d_ff: checkpoint has {moe_config_dict.get('d_ff')}, "
                      f"using {d_ff_to_use}")
            moe_config = MoEConfig(
                d_model=moe_config_dict.get('d_model', config.embedding_size),
                d_ff=d_ff_to_use,
                num_experts=moe_config_dict.get('num_experts', 8),
                num_shared_experts=moe_config_dict.get('num_shared_experts', 1),
                top_k=moe_config_dict.get('top_k', 2),
                expert_dropout=moe_config_dict.get('expert_dropout', 0.1),
                load_balance_strategy=moe_config_dict.get('load_balance_strategy', 'deepseek'),
                aux_loss_weight=moe_config_dict.get('aux_loss_weight', 0.001),
                use_moe_from_layer=moe_config_dict.get('use_moe_from_layer', 2),
                use_swiglu_experts=moe_config_dict.get('use_swiglu_experts', True),
                router_warmup_steps=moe_config_dict.get('router_warmup_steps', 0),
                z_loss_weight=moe_config_dict.get('z_loss_weight', 0.005),
                bias_lr=moe_config_dict.get('bias_lr', 1e-3),
                bias_momentum=moe_config_dict.get('bias_momentum', 0.6),
                
            )
            if verbose:
                print(f"  MoE config: {moe_config_dict.get('num_experts')} experts, "
                      f"top-{moe_config_dict.get('top_k')}, from layer {moe_config_dict.get('use_moe_from_layer')}")
        else:
            moe_config = MoEConfig(d_model=config.embedding_size, d_ff=config.nhid)
        
        model = FlashMoETransformer(config, moe_config)
        use_mixed_precision = True
        
    elif 'FlashAttention' in model_type:
        # Flash Attention model (no MoE)
        nhid_to_use = inferred_nhid or config_dict.get('nhid', 512)
        if verbose and config_dict.get('nhid') and inferred_nhid and config_dict['nhid'] != inferred_nhid:
            print(f"  Warning: config nhid={config_dict['nhid']} vs inferred nhid={inferred_nhid}, "
                  f"using weight-inferred value")
        config = FlashAttentionConfig(
            embedding_size=config_dict.get('embedding_size', 256),
            nhid=nhid_to_use,
            nhead=config_dict.get('nhead', 8),
            nlayers=config_dict.get('nlayers', 6),
            dropout=config_dict.get('dropout', 0.1),
            use_learnt_att_pool=use_learnt_att_pool_inferred,
            use_swiglu=config_dict.get('use_swiglu', True),
            use_rope=config_dict.get('use_rope', True),
            use_flash=config_dict.get('use_flash', True),
        )
        model = FlashAttentionTransformer(config)
        use_mixed_precision = True
        
    else:
        # Baseline Transformer
        config = BaseConfig(
            embedding_size=config_dict.get('embedding_size', 256),
            nhid=config_dict.get('nhid', 512),
            nlayers=config_dict.get('nlayers', 6),
            dropout=config_dict.get('dropout', 0.1),
        )
        model = BaselineTransformer(config)
        use_mixed_precision = False
        
    # Load state dict
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Mixed precision: {use_mixed_precision}")
        print(f"   Device: {device}")
        print(f"{'='*80}\n")
    
    
    return model, config, moe_config, use_mixed_precision, model_type


# In[29]:


from torch.utils.data import Dataset
class LazyClinicalDataset(Dataset):
    """
    Memory-efficient dataset that parses data on-the-fly.
    
    Memory usage: O(1) instead of O(N)
    Trade-off: Slightly slower due to per-batch parsing, but negligible
    compared to GPU inference time.
    
    Perfect for inference on large datasets.
    """
    
    def __init__(self, df: pd.DataFrame, config: BaseConfig):
        self.config = config
        # Store only the DataFrame reference - NO tensor allocation!
        self.df = df.reset_index(drop=True)
        
        # Pre-extract columns as lists for faster access
        self.age_strs = self.df['age_in_months'].tolist()
        self.gender_strs = self.df['gender_cd'].tolist()
        self.cd_strs = self.df['cd'].tolist()
        self.lob_strs = self.df['lob'].tolist()
        self.dt_cnt = self.df['dt_cnt'].tolist()
        
        # Only needed for training - can skip for inference
        self.target_strs = self.df['target'].tolist() if 'target' in self.df.columns else None
        
        print(f"LazyClinicalDataset initialized with {len(self.df):,} samples (lazy loading)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Parse data on-demand (not pre-loaded!)
        age_list = conv_age_gender(self.age_strs[idx], self.config.len_dy)
        gender_list = conv_age_gender(self.gender_strs[idx], self.config.len_dy, max_val=3)
        cd_list = conv_cd(self.cd_strs[idx], self.config.len_dy, self.config.len_cd)
        lob_list = conv_lob(self.lob_strs[idx], self.config.len_dy)
        
        # Convert to tensors
        age = torch.tensor(age_list, dtype=torch.long)
        gender = torch.tensor(gender_list, dtype=torch.long)
        codes = torch.tensor(cd_list, dtype=torch.long)
        lob = torch.tensor(lob_list, dtype=torch.long)
        
        # Target (for training) - lazy parse
        if self.target_strs is not None:
            target_list = conv_target(self.target_strs[idx], self.config.len_dy, self.config.target_cd_cnt)
        else:
            target_list = [[0] for _ in range(self.config.len_dy)]
        
        return {
            'age': age,
            'gender': gender,
            'lob': lob,
            'codes': codes,
            'dt_cnt': self.dt_cnt[idx],
            'target': target_list
        }


# In[7]:


MODEL_PATHS


# ### Generate embeddings

# In[30]:


import time
import copy
from typing import Tuple, List, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
import threading

def generate_embeddings(
    model: torch.nn.Module,
    config: 'BaseConfig',
    data: pd.DataFrame,
    device: torch.device,
    id_column: str = 'individual_id',       # Parameterized: ID column name
    lob_value: Optional[str] = None,         # Parameterized: Auto-add LOB if specified
    desc_prefix: str = '',                   # Parameterized: Progress bar prefix
    batch_size: int = 64,
    num_workers: int = 4,
    use_mixed_precision: bool = True,
    verbose: bool = True,
    multi_gpu: bool = False,
    moe_config: Optional['MoEConfig'] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Generate embeddings for all members in the dataset (unified for all LOBs).
    
    This is the consolidated embedding generation function that works for:
    - Commercial: id_column='individual_id', lob_value=None (already in data)
    - Medicaid: id_column='asdb_member_key', lob_value='Medicaid'
    - Medicare: id_column='individual_id', lob_value='Medicare' (if needed)
    
    The embedding is extracted from the FINAL TEMPORAL REPRESENTATION:
    - BaselineTransformer: output of transformer_encoder_dy
    - FlashAttentionTransformer: input to model.norm (after all temporal layers)
    - FlashMoETransformer: input to model.norm (after all temporal + MoE layers)
    
    Patient embedding = embedding at the LAST VALID DAY (dt_cnt - 1)
    
    Optimizations:
    1. Pre-allocated pinned memory output (no vstack)
    2. Non-blocking async GPU→CPU transfers
    3. torch.inference_mode (faster than no_grad)
    4. Optional multi-GPU with true parallelism
    5. Progress bar with ETA
    
    Args:
        model: Loaded model in eval mode
        config: Model configuration
        data: DataFrame with required columns (age_in_months, gender_cd, cd, lob, dt_cnt, etc.)
        device: Primary device
        id_column: Column name for member IDs (default: 'individual_id')
        lob_value: If specified, add this as 'lob' column if missing (e.g., 'Medicaid')
        desc_prefix: Prefix for progress bar description (e.g., 'Medicaid')
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        use_mixed_precision: Use FP16 for Flash models
        verbose: Print progress
        multi_gpu: Enable multi-GPU processing
        moe_config: MoE config (required for multi-GPU with MoE models)
        
    Returns:
        embeddings: np.ndarray [num_members, embedding_size]
        member_ids: List of member IDs (from id_column)
        index_dts: List of index dates
    """
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size
    
    # Detect model type
    has_moe = (hasattr(model, 'forward') and 
               'return_moe_losses' in model.forward.__code__.co_varnames)
    
    n_gpus = torch.cuda.device_count() if multi_gpu else 1
    
    # Build description
    desc = f"{desc_prefix} " if desc_prefix else ""
    desc += "Embedding Generation"
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"{desc.upper()}")
        print(f"{'='*70}")
        print(f"Samples: {n_samples:,} | Batch: {batch_size} | GPUs: {n_gpus}")
        print(f"Workers: {num_workers} | Mixed precision: {use_mixed_precision}")
        print(f"ID column: {id_column}")
    
    # Handle LOB column (for Medicaid/Medicare where it may not exist)
    if lob_value and 'lob' not in data.columns:
        data = data.copy()
        data['lob'] = lob_value
        if verbose:
            print(f"  Added 'lob'='{lob_value}' column")
    
    # Pre-allocate pinned memory output
    embeddings_output = torch.empty(
        (n_samples, embedding_dim),
        dtype=torch.float32,
        pin_memory=True
    )
    
    # Extract IDs using the specified column
    if id_column in data.columns:
        member_ids = data[id_column].astype(str).tolist()
    else:
        # Fallback to individual_id if specified column not present
        member_ids = data['individual_id'].astype(str).tolist()
        if verbose:
            print(f"  Warning: '{id_column}' not found, using 'individual_id'")
    
    index_dts = data['index_dt'].astype(str).tolist()
    
    # Build progress description
    pbar_desc = f"Generating {desc_prefix} embeddings" if desc_prefix else "Generating embeddings"
    
    if n_gpus > 1 and multi_gpu:
        return _generate_embeddings_multi_gpu(
            model=model,
            config=config,
            data=data,
            embeddings_output=embeddings_output,
            member_ids=member_ids,
            index_dts=index_dts,
            n_gpus=n_gpus,
            batch_size=batch_size,
            num_workers=num_workers,
            use_mixed_precision=use_mixed_precision,
            has_moe=has_moe,
            moe_config=moe_config,
            verbose=verbose,
            start_time=start_time,
            pbar_desc=pbar_desc,
        )
    else:
        return _generate_embeddings_single_gpu(
            model=model,
            config=config,
            data=data,
            device=device,
            embeddings_output=embeddings_output,
            member_ids=member_ids,
            index_dts=index_dts,
            batch_size=batch_size,
            num_workers=num_workers,
            use_mixed_precision=use_mixed_precision,
            has_moe=has_moe,
            verbose=verbose,
            start_time=start_time,
            pbar_desc=pbar_desc,
        )


def _generate_embeddings_single_gpu(
    model, config, data, device, embeddings_output,
    member_ids, index_dts, batch_size, num_workers,
    use_mixed_precision, has_moe, verbose, start_time,
    pbar_desc: str = "Generating embeddings"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Single GPU optimized path (unified for all LOBs)."""
    
    n_samples = len(data)
    model.eval()
    
    dataset = LazyClinicalDataset(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    current_idx = 0
    pbar = tqdm(dataloader, desc=pbar_desc, disable=not verbose)
    
    with torch.inference_mode():
        with EmbeddingExtractor(model) as extractor:
            for batch in pbar:
                batch_size_actual = batch['age'].shape[0]
                batch_start = current_idx
                batch_end = batch_start + batch_size_actual
                
                x = torch.cat([
                    batch['age'].unsqueeze(-1),
                    batch['gender'].unsqueeze(-1),
                    batch['lob'].unsqueeze(-1),
                    batch['codes']
                ], dim=-1).to(device, non_blocking=True)
                
                dt_cnt = batch['dt_cnt']
                
                # Forward pass
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if has_moe:
                            _ = model(x, return_moe_losses=False)
                        else:
                            _ = model(x)
                else:
                    if has_moe:
                        _ = model(x, return_moe_losses=False)
                    else:
                        _ = model(x)
                
                # Extract embeddings
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                
                # Async copy to pre-allocated pinned memory
                embeddings_output[batch_start:batch_end].copy_(
                    patient_embs.float(),
                    non_blocking=True
                )
                
                current_idx = batch_end
                
                # Progress metrics
                elapsed = time.time() - start_time
                speed = batch_end / elapsed
                eta = (n_samples - batch_end) / speed if speed > 0 else 0
                pbar.set_postfix({
                    'speed': f'{speed:.0f}/s',
                    'ETA': f'{eta:.0f}s'
                })
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
        print(f"   Output: {embeddings.shape}")
    
    return embeddings, member_ids, index_dts


def _generate_embeddings_multi_gpu(
    model, config, data, embeddings_output, member_ids, index_dts,
    n_gpus, batch_size, num_workers, use_mixed_precision, has_moe,
    moe_config, verbose, start_time,
    pbar_desc: str = "Multi-GPU"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Multi-GPU path with true parallelism (unified for all LOBs)."""
    
    n_samples = len(data)
    
    if verbose:
        print(f"Multi-GPU mode: {n_gpus} GPUs")
    
    # Clone model to each GPU
    models = []
    for gpu_id in range(n_gpus):
        if verbose:
            print(f"  Cloning model to GPU {gpu_id}...")
        
        with torch.cuda.device(gpu_id):
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to(f'cuda:{gpu_id}')
            model_copy.eval()
            models.append(model_copy)
    
    # Split data into chunks
    chunk_size = (n_samples + n_gpus - 1) // n_gpus
    data_chunks = []
    start_indices = []
    
    for i in range(n_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        data_chunks.append(data.iloc[start_idx:end_idx].reset_index(drop=True))
        start_indices.append(start_idx)
        
        if verbose:
            print(f"  GPU {i}: samples {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} samples)")
    
    progress_lock = threading.Lock()
    total_processed = [0]
    errors = []
    
    def process_chunk(gpu_id: int, data_chunk: pd.DataFrame, start_idx: int):
        """Process a data chunk on a specific GPU."""
        if len(data_chunk) == 0:
            return
        
        try:
            gpu_device = torch.device(f'cuda:{gpu_id}')
            gpu_model = models[gpu_id]
            
            dataset = LazyClinicalDataset(data_chunk, config)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=create_collate_fn(config),
                num_workers=max(1, num_workers // n_gpus),
                pin_memory=True,
            )
            
            local_idx = start_idx
            
            with torch.inference_mode():
                with EmbeddingExtractor(gpu_model) as extractor:
                    for batch in dataloader:
                        batch_size_actual = batch['age'].shape[0]
                        
                        x = torch.cat([
                            batch['age'].unsqueeze(-1),
                            batch['gender'].unsqueeze(-1),
                            batch['lob'].unsqueeze(-1),
                            batch['codes']
                        ], dim=-1).to(gpu_device, non_blocking=True)
                        
                        dt_cnt = batch['dt_cnt']
                        
                        if use_mixed_precision:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                if has_moe:
                                    _ = gpu_model(x, return_moe_losses=False)
                                else:
                                    _ = gpu_model(x)
                        else:
                            if has_moe:
                                _ = gpu_model(x, return_moe_losses=False)
                            else:
                                _ = gpu_model(x)
                        
                        dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                        patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                        
                        embeddings_output[local_idx:local_idx + batch_size_actual].copy_(
                            patient_embs.float(),
                            non_blocking=True
                        )
                        
                        local_idx += batch_size_actual
                        
                        with progress_lock:
                            total_processed[0] += batch_size_actual
            
            torch.cuda.synchronize(gpu_device)
            
        except Exception as e:
            errors.append((gpu_id, str(e)))
    
    # Launch parallel processing
    if verbose:
        pbar = tqdm(total=n_samples, desc=f"{pbar_desc} ({n_gpus} GPUs)")
    
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(process_chunk, gpu_id, data_chunks[gpu_id], start_indices[gpu_id])
            for gpu_id in range(n_gpus)
        ]
        
        last_count = 0
        while not all(f.done() for f in futures):
            with progress_lock:
                current = total_processed[0]
            if verbose:
                pbar.update(current - last_count)
            last_count = current
            time.sleep(0.1)
        
        if verbose:
            pbar.update(n_samples - last_count)
            pbar.close()
        
        for f in futures:
            f.result()
    
    if errors:
        raise RuntimeError(f"GPU errors: {errors}")
    
    # Cleanup
    for m in models:
        del m
    torch.cuda.empty_cache()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
        print(f"   Effective: {n_samples/elapsed * n_gpus:,.0f} samples/s (across {n_gpus} GPUs)")
        print(f"   Output: {embeddings.shape}")
    
    return embeddings, member_ids, index_dts


# #### Save embeddings

# In[31]:


def save_embeddings(
    embeddings: np.ndarray,
    individual_ids: List[str],
    index_dts: List[str],
    output_path: str,
    model_name: str = "",
    additional_metadata: Dict = None
) -> str:
    """
    Save embeddings to disk in a structured format.
    
    Saves both:
    1. NPZ file with embeddings and metadata
    2. CSV file with IDs and index dates for easy lookup
    
    Args:
        embeddings: [num_members, embedding_dim] array
        individual_ids: List of member IDs
        index_dts: List of index dates
        output_path: Directory to save files
        model_name: Name of the model for file naming
        additional_metadata: Optional dict of additional info to save
        
    Returns:
        Path to saved NPZ file
    """
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename
    if model_name:
        filename_base = f"embeddings_{model_name}_{timestamp}"
    else:
        filename_base = f"embeddings_{timestamp}"
    
    # Save NPZ with embeddings and metadata
    npz_path = os.path.join(output_path, f"{filename_base}.npz")
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        individual_ids=np.array(individual_ids, dtype=object),
        index_dts=np.array(index_dts, dtype=object),
        embedding_dim=embeddings.shape[1],
        num_members=len(individual_ids),
        **(additional_metadata or {})
    )
    print(f"Embeddings saved to: {npz_path}")
    
    # Save CSV for easy lookup
    csv_path = os.path.join(output_path, f"{filename_base}_ids.csv")
    pd.DataFrame({
        'individual_id': individual_ids,
        'index_dt': index_dts,
        'embedding_idx': range(len(individual_ids))
    }).to_csv(csv_path, index=False)
    print(f"ID mapping saved to: {csv_path}")
    
    return npz_path


# In[32]:


from google.cloud import bigquery
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os

def save_embeddings_to_bigquery(
    embeddings: np.ndarray,
    individual_ids: list,
    index_dts: list,
    project_id: str,
    dataset_id: str,
    table_name: str,
    exp_name: str = "",
    model_type: str = "",
    if_exists: str = "replace"  # 'replace', 'append', 'fail'
) -> str:
    """
    Save embeddings to BigQuery.
    
    Args:
        embeddings: numpy array [num_members, embedding_dim]
        individual_ids: list of member IDs
        index_dts: list of index dates
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_name: Table name to create
        exp_name: Experiment name for metadata
        model_type: Model type for metadata
        if_exists: What to do if table exists ('replace', 'append', 'fail')
        
    Returns:
        Full table path
    """
    # Create DataFrame with ID columns
    df = pd.DataFrame({
        'individual_id': individual_ids,
        'index_dt': index_dts,
    })
    
    # Add embedding columns (embedding_0, embedding_1, ..., embedding_N)
    embedding_dim = embeddings.shape[1]
    for i in range(embedding_dim):
        df[f'embedding_{i}'] = embeddings[:, i].astype(np.float32)
    
    # Add metadata columns
    df['exp_name'] = exp_name
    df['model_type'] = model_type
    
    # Full table path
    full_table_id = f"{project_id}.{dataset_id}.{table_name}"
    
    print(f"Writing {len(df):,} rows to BigQuery: {full_table_id}")
    print(f"  Columns: {len(df.columns)} (embedding_dim={embedding_dim})")
    
    # Initialize BigQuery client
    client = bigquery.Client()
    
    # Configure job
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE if if_exists == "replace" 
                          else bigquery.WriteDisposition.WRITE_APPEND if if_exists == "append"
                          else bigquery.WriteDisposition.WRITE_EMPTY,
    )
    
    # Load data
    job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
    job.result()  # Wait for completion
    
    # Verify
    table = client.get_table(full_table_id)
    print(f"✅ Loaded {table.num_rows:,} rows to {full_table_id}")
    
    return full_table_id


# ##### Test

# In[40]:


PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"
TABLE_NAME = "a964286_TEST_embedding_function_validation"
table_ref = client.dataset(DATASET_ID).table(TABLE_NAME)


# In[47]:


# Create fake data
num_samples = 100
embedding_dim = 256

# Generate fake embeddings (random floats)
fake_embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)

# Generate fake individual IDs
fake_individual_ids = [f"FAKE_ID_{i:06d}" for i in range(num_samples)]

# Generate fake index dates
fake_index_dts = pd.date_range(
    start='2023-01-01', 
    periods=num_samples, 
    freq='D'
).strftime('%Y-%m-%d').tolist()

# Test configuration
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"
TABLE_NAME = "a964286_TEST_embedding_function_validation"  # Use a clearly named test table
result_table = save_embeddings_to_bigquery(
    embeddings=fake_embeddings,
    individual_ids=fake_individual_ids,
    index_dts=fake_index_dts,
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_name=TABLE_NAME,
    exp_name="test_experiment",
    model_type="test_model",
    if_exists="replace"
)


# In[ ]:





# In[ ]:





# ### Commercial IP downstream

# In[22]:


import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
client = bigquery.Client()
credentials, project= google.auth.default()
print('credentials:', credentials, ', project:', project)
import pandas as pd
from tqdm.notebook import tqdm
client = bigquery.Client()


# In[143]:


# import members not in the trainingset of the transformer
commercial_sql_code = """
select * from edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5
"""
df_cm = client.query(commercial_sql_code).to_dataframe()


# In[15]:


# sample before and after 2023-10-16 (post part will be used for oot validation)
# 0.3 samples for efficent evaluations of embeddings
df_cm['index_dt'] = pd.to_datetime(df_cm['index_dt'])
df_cm_b4_oct = df_cm[df_cm['index_dt'] <= pd.to_datetime("2023-10-16")]
df_cm_after_oct = df_cm[df_cm['index_dt'] > pd.to_datetime("2023-10-16")]
df_cm_b4_oct_sample = df_cm_b4_oct.sample(frac=0.3, random_state=42)
df_cm_after_oct_sample = df_cm_after_oct.sample(frac=0.3, random_state=42)
df_cm_sample = pd.concat([df_cm_b4_oct_sample,
                         df_cm_after_oct])


# #### Embedding generation

# In[84]:


MODEL_PATHS = {
#     # Experiment 1: Dense Baseline (no Flash Attention, no MoE)
#     'exp1_dense_baseline_pure_legacy': 
#         'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp1_dense_baseline_pure_legacy/saved_models/'
#         'exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp1_dense_baseline_bs128_ep1_d256_20251230_055716_final.pt',
    
    # Experiment 1b: Dense Baseline (same opt config as 2b and 6)
    # 'exp1_dense_baseline_opt_config': 
    #     'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp1_dense_baseline_opt_config/saved_models/'
    #     'exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2_exp1_dense_baseline_bs64_ep1_d256_20260108_183616_final.pt',

    # Experiment 2b: Flash Attention + Learned Pooling (no MoE)
    # 'exp2b_flash_learned_pool_v2': 
    #     'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp2b_flash_learned_pool_v2/saved_models/'
    #     'exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20251230_114137_final.pt',
    
    # # Experiment 6: Flash + MoE with DeepSeek auxiliary-free balancing
    # 'exp6_auxiliary_free_v3': 
    #     'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/'
    #     'exp6_auxiliary_free_v3/saved_models/'
    #     'exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_20251231_152438_final.pt',
    
    # Round 5; 
    'exp2b_flash_learned_pool': 
    'logs/exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2/'
    'exp2b_flash_learned_pool/saved_models/'
    'exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20260110_112709_final.pt'
}


# In[85]:


results = {}
batch_size = 64
# output_dir = "embedding_output/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2"
output_dir = "embedding_output/exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2"
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for exp_name, model_path in tqdm(MODEL_PATHS.items()):
    cleanup_gpu_memory(verbose=False)
    model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
        model_path=MODEL_PATHS[exp_name],
        device=device,
        verbose=True
    )
    embeddings, individual_ids, index_dts = generate_embeddings(
        model=model,
        config=config,
        data=df_cm_sample,
        device=device,
        id_column='individual_id',  # Commercial uses individual_id
        lob_value=None,              # Commercial data already has lob column
        desc_prefix='Commercial',
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        verbose=True,
        multi_gpu=True,           
        moe_config=moe_config, 
    )
    exp_output_dir = os.path.join(output_dir, exp_name)
    embeddings_path = save_embeddings(
        embeddings=embeddings,
        individual_ids=individual_ids,
        index_dts=index_dts,
        output_path=exp_output_dir,
        model_name=exp_name,
        additional_metadata={
            'model_path': model_path,
            'model_type': model_type,
            'use_mixed_precision': use_mixed_precision,
        }
    )
    # safe_exp_name = exp_name.replace('-', '_').replace('.', '_')
    # table_name = f"a965286_te4exp_3lob_exp_round5_v2_{safe_exp_name}_commercial_0p3sample_embedding"
    # bq_table_path = save_embeddings_to_bigquery(
    #     embeddings=embeddings,
    #     individual_ids=individual_ids,
    #     index_dts=index_dts,
    #     project_id=PROJECT_ID,
    #     dataset_id=DATASET_ID,
    #     table_name=table_name,
    #     exp_name=exp_name,
    #     model_type=model_type,
    #     if_exists="replace"
    # )
    results[exp_name] = {
        'embeddings_path': embeddings_path,
        'embedding_shape': embeddings.shape,
        'model_type': model_type,
        'model_path': model_path,
        'status': 'success'
    }

    # Free model memory
    del model
    del embeddings
    torch.cuda.empty_cache()


# In[ ]:





# #### Replicate IP model pipeline

# In[86]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')
import time
from dataclasses import dataclass


# In[104]:


# constant
EMBEDDING_BASE = "embedding_output/exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2"
EXPERIMENT_NAMES = [
    # 'exp1_dense_baseline_opt_config',
    # 'exp1_dense_baseline_pure_legacy', 
    'exp2b_flash_learned_pool', 
    # 'exp2b_flash_learned_pool_v2', 
    # 'exp6_auxiliary_free_v3'
    
    
]
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"
FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_commercial_ip_heldout_transformer_matched_final_dataset_4_te_experiment_round5_downstream"
OOT_CUTOFF_DATE = "2023-10-16"
TARGET_COLUMN = "ip6"
EXCLUDE_COLUMNS = frozenset([
    # Keys and identifiers
    'individual_id', 'member_id', 'index_dt', 'birth_dt', 'feature_end_dt',
    
    # Outcome columns (target and related)
    'ip6', 'sum_ip6_admits', 'sum_ip6_los', 'sum_ip6_acu_days',
    
    # Eligibility/continuity flags
    'mon_3_include', 'mon_6_include', 'mon_12_include',
    'exclude_ip', 'include_post_6_status',
    
    # Split key
    'ind_id_last_digit',
    
    # Leakage columns (cost amounts, outreach flags from previous model)
    'clm_allowed_amt_1yr', 'clm_allowed_amt_2yr', 'clm_allowed_amt_3mo', 'clm_allowed_amt_6mo',
    'clm_paid_amt_1yr', 'clm_paid_amt_2yr', 'clm_paid_amt_3mo', 'clm_paid_amt_6mo',
    'clm_par_allowed_amt_1yr', 'clm_par_allowed_amt_2yr', 'clm_par_allowed_amt_3mo', 'clm_par_allowed_amt_6mo',
    'clm_par_paid_amt_1yr', 'clm_par_paid_amt_2yr', 'clm_par_paid_amt_3mo', 'clm_par_paid_amt_6mo',
    'clm_srv_copay_amt_1yr', 'clm_srv_copay_amt_3mo', 'clm_srv_copay_amt_6mo',
    'covid_19', 'hpd_major_flag', 'chronic',
    'txt_member', 'txt_referral', 'txt_1yr_outreach', 'talked'
])
# Data-level downsampling configuration
# Previous model used 10:1 negative sampling (table: yc_a565095_cp_ip_neg_10_trs_3)
NEGATIVE_DOWNSAMPLE_RATIO = 10  # Keep 10 negatives per 1 positive
APPLY_DOWNSAMPLING = True  # Set to False to disable downsampling


# In[105]:


# Understand the time lapse between edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_experiment and edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5
# Row	discrepancy_bucket	member_count	pct_of_total	avg_diff_days	min_diff_days	max_diff_days
# 1	0: Exact match	6839956	97.53	0.0	0	0
# 2	2: 1 week - 1 month	4607	0.07	0.019101367484263269	-30	30
# 3	3: 1-3 months	21900	0.31	0.02187214611872133	-90	90
# 4	4: 3-6 months	45604	0.65	0.015261819138672406	-153	153
# 5	5: 6-12 months	100909	1.44	0.10750279955207313	-334	334


# In[106]:


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def lift_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """Calculate lift at top percentile. Lift = precision@k / baseline_prevalence."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    precision_at_k = y_true[top_k_indices].mean()
    baseline = y_true.mean()
    return precision_at_k / baseline if baseline > 0 else 0.0


def true_positives_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> int:
    """Count true positives in top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    return int(y_true[top_k_indices].sum())


def precision_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """Calculate precision at top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_indices].mean())


def compute_split_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics for a single split.
    
    Returns:
        Dict with metric names as keys (without split prefix)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    return {
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'lift_1pct': lift_at_percentage(y_true, y_prob, 0.01),
        'lift_5pct': lift_at_percentage(y_true, y_prob, 0.05),
        'lift_10pct': lift_at_percentage(y_true, y_prob, 0.10),
        'tp_1pct': true_positives_at_percentage(y_true, y_prob, 0.01),
        'precision_1pct': precision_at_percentage(y_true, y_prob, 0.01),
        'n_samples': len(y_true),
        'n_positives': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
    }


# In[107]:


# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================
import glob

def load_embeddings_from_dir(embedding_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from the NPZ file for a given experiment.
    
    Args:
        exp_name: Experiment name (e.g., 'exp1_dense_baseline')
        base_dir: Base directory containing experiment subdirectories
        
    Returns:
        embeddings: numpy array [num_members, 256]
        individual_ids: list of member IDs
        index_dts: list of index dates
    """
    if os.path.isdir(embedding_path):
        npz_files = glob.glob(os.path.join(embedding_path, "embeddings_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files found in {embedding_path}")
        npz_path = sorted(npz_files)[-1]  # Use most recent
    elif '*' in embedding_path:
        npz_files = glob.glob(embedding_path)
        if not npz_files:
            raise FileNotFoundError(f"No files matching {embedding_path}")
        npz_path = sorted(npz_files)[-1]
    else:
        npz_path = embedding_path
    
    data = np.load(npz_path, allow_pickle=True)
    
    return (
        data['embeddings'],
        data['individual_ids'],
        data['index_dts']
    )

def load_mbrs_have_embed(embedding_path: str):
    import pandas as pd
    if os.path.isdir(embedding_path):
        csv_files = glob.glob(os.path.join(embedding_path, "embeddings_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No embedding ID files found in {embedding_path}")
        csv_file = sorted(csv_files)[-1]
        return pd.read_csv(csv_file)
    
    
def create_embedding_df(embeddings: np.ndarray,
        individual_ids: np.ndarray,
        index_dts: np.ndarray
    ) -> pd.DataFrame:
    """
    Create a DataFrame from embedding data.
    col: individual_id, index_dt, embedding_0...embedding_255
    """
    embedding_dim = embeddings.shape[1]
    embedding_cols = [f'embedding_{i}' for i in range(embedding_dim)]
    
    df = pd.DataFrame({
        'individual_id': individual_ids,
        'index_dt': pd.to_datetime(index_dts).strftime('%Y-%m-%d')
    })
    
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    return pd.concat([df, embedding_df], axis=1)

def join_embeddings_with_features(
    emb_df: pd.DataFrame,
    df_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Join embeddings with features on (individual_id, index_dt).
    Applies eligibility filters automatically.
    """
    # Standardize date format
    df_features = df_features.copy()
    df_features['index_dt'] = pd.to_datetime(df_features['index_dt']).dt.strftime('%Y-%m-%d')
    emb_df['index_dt'] = pd.to_datetime(emb_df['index_dt']).dt.strftime('%Y-%m-%d')
    emb_df['individual_id'] = emb_df['individual_id'].astype(str)
    df_features['individual_id'] = df_features['individual_id'].astype(str)
    # Inner join
    df_merged = df_features.merge(
        emb_df,
        on=['individual_id', 'index_dt'],
        how='inner'
    )
    
    # # mon_6_include filter is done in sql and the other two columsn do not present
    # if 'mon_6_include' in df_merged.columns:
    #     df_merged = df_merged[df_merged['mon_6_include'] == 1]
    # if 'exclude_ip' in df_merged.columns:
    #     df_merged = df_merged[(df_merged['exclude_ip'] == 0) | (df_merged['exclude_ip'].isna())]
    # if 'include_post_6_status' in df_merged.columns:
    #     df_merged = df_merged[df_merged['include_post_6_status'] == 1]
    
    # Remove duplicates
    df_merged = df_merged.drop_duplicates(
        subset=['individual_id', 'index_dt'], 
        keep='last'
    )
    
    return df_merged

def create_data_splits(
    df: pd.DataFrame,
    oot_cutoff_date: str = OOT_CUTOFF_DATE
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test/OOT splits based on ind_id_last_digit and date.
    
    Split Logic (matching previous pipeline with improved consistency):
        - Train: ind_id_last_digit 0-7 AND date <= cutoff
        - Val:   ind_id_last_digit 8 AND date <= cutoff
        - Test:  ind_id_last_digit 9 AND date <= cutoff
        - OOT:   date > cutoff (all ind_id_last_digit values)
        - OOT_strict: date > cutoff AND ind_id_last_digit 9
    
    Note: OOT may contain members also present in train/val/test at earlier dates.
    This tests temporal generalization (model performance on future time periods).
    
    Returns:
        Dict with keys 'train', 'val', 'test', 'oot', 'oot_strict'
    """
    df = df.copy()
    df['_index_dt_parsed'] = pd.to_datetime(df['index_dt'])
    
    oot_cutoff = pd.to_datetime(oot_cutoff_date)
    
    splits = {
        'train': df[(df['ind_id_last_digit'].isin([0,1,2,3,4,5,6,7]))&(df['_index_dt_parsed'] <= oot_cutoff)],
        'val': df[(df['ind_id_last_digit'] == 8) & (df['_index_dt_parsed'] <= oot_cutoff)],
        'test': df[(df['ind_id_last_digit'] == 9) & (df['_index_dt_parsed'] <= oot_cutoff)],
        'oot': df[df['_index_dt_parsed'] > oot_cutoff],
        'oot_strict': df[(df['_index_dt_parsed'] > oot_cutoff) & (df['ind_id_last_digit'] == 9)]
    }
    print("Data splits created:")
    for name, split_df in splits.items():
        if len(split_df) > 0:
            prevalence = split_df[TARGET_COLUMN].mean() * 100
            print(f"  {name}: {len(split_df):,} rows, {int(split_df[TARGET_COLUMN].sum()):,} positives ({prevalence:.2f}%)")
        else:
            print(f"  {name}: EMPTY")
            
    # Remove temp column
    for key in splits:
        splits[key] = splits[key].drop(columns=['_index_dt_parsed'])
    
    return splits

def identify_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify embedding and tabular feature columns.
    
    Returns:
        Tuple of (embedding_features, tabular_features)
    """
    all_cols = set(df.columns)
    
    embedding_features = sorted([c for c in all_cols if c.startswith('embedding_')])
    
    excluded = EXCLUDE_COLUMNS | set(embedding_features) | {'_exp_name', 'index_dt_parsed', '_index_dt_parsed'}
    tabular_features = sorted([
        c for c in all_cols 
        if c not in excluded and c != TARGET_COLUMN
    ])
    
    return embedding_features, tabular_features

def downsample_negatives(
    X: pd.DataFrame, 
    y: pd.Series,
    ratio: int = NEGATIVE_DOWNSAMPLE_RATIO,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Downsample negative class to achieve target ratio (matching previous pipeline).
    
    The previous model used a pre-sampled table (yc_a565095_cp_ip_neg_10_trs_3)
    with approximately 10:1 negative-to-positive ratio. This function replicates
    that data-level rebalancing strategy.
    
    Args:
        X: Feature DataFrame
        y: Target Series (0/1)
        ratio: Number of negatives to keep per positive (default: 10)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_resampled, y_resampled)
        
    Example:
        If y has 1000 positives and 50000 negatives with ratio=10:
        - Keep all 1000 positives
        - Randomly sample 10000 negatives (10 * 1000)
        - Return resampled data with ~10:1 ratio
    """
    np.random.seed(random_state)
    
    # Separate positive and negative indices
    pos_mask = y == 1
    neg_mask = y == 0
    
    pos_indices = X.index[pos_mask].tolist()
    neg_indices = X.index[neg_mask].tolist()
    
    n_positives = len(pos_indices)
    n_negatives = len(neg_indices)
    target_n_negatives = int(n_positives * ratio)
    
    # If we already have fewer negatives than target, keep all
    if n_negatives <= target_n_negatives:
        print(f"  Downsampling: No action needed (current ratio: {n_negatives/n_positives:.1f}:1)")
        return X, y
    
    # Randomly sample negatives
    sampled_neg_indices = np.random.choice(neg_indices, size=target_n_negatives, replace=False)
    
    # Combine indices
    keep_indices = pos_indices + sampled_neg_indices.tolist()
    
    X_resampled = X.loc[keep_indices].copy()
    y_resampled = y.loc[keep_indices].copy()
    
    # Shuffle to mix positives and negatives
    shuffle_idx = np.random.permutation(len(X_resampled))
    X_resampled = X_resampled.iloc[shuffle_idx].reset_index(drop=True)
    y_resampled = y_resampled.iloc[shuffle_idx].reset_index(drop=True)
    
    print(f"  Downsampling: {n_negatives}:{n_positives} ({n_negatives/n_positives:.1f}:1) → "
          f"{target_n_negatives}:{n_positives} ({ratio}:1)")
    
    return X_resampled, y_resampled


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target y, handling missing values.
    """
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int)
    
    # Fill missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    X[cat_cols] = X[cat_cols].fillna('missing')
    
    return X, y



# In[145]:


from dataclasses import dataclass
@dataclass
class PreparedData:
    """Container for prepared evaluation data. Prepare once, evaluate many models."""
    X_splits: Dict[str, pd.DataFrame]
    y_splits: Dict[str, pd.Series]
    feature_cols: List[str]
    embedding_features: List[str]
    tabular_features: List[str]
    cat_feature_indices: List[int]  # Column indices for CatBoost
    feature_set: str
    embedding_path: str
    downsampled: bool = True  # Whether training data was downsampled

def prepare_evaluation_data(
    df_features: pd.DataFrame,
    embedding_location_path: str = "",
    feature_set: str = 'embedding_only',
    oot_cutoff_date: str = OOT_CUTOFF_DATE,
    downsample_ratio: Optional[float] = None,
    random_state: int = 42
) -> PreparedData:
    """
    Prepare data once for multiple model evaluations.
    
    This function decouples data preparation from model training, allowing
    the same prepared data to be used across multiple models efficiently.
    
    Args:
        df_features: DataFrame with features and outcomes (from BigQuery)
        embedding_location_path: Path to NPZ file or directory (not needed for tabular_only)
        feature_set: One of 'embedding_only', 'tabular_only', 'hybrid'
        oot_cutoff_date: Date string for OOT split cutoff
        downsample_ratio: If provided, downsample training negatives to this ratio 
                          (e.g., 10.0 for 10:1 negative:positive ratio). 
                          Only applied to training set. Set to 10.0 to match previous model.
        random_state: Random seed for downsampling reproducibility
        
    Returns:
        PreparedData object containing X_splits, y_splits, and metadata
    """
    total_start_time = time.time()
    step_times = {}
    
    # Validate feature_set
    valid_feature_sets = {'embedding_only', 'tabular_only', 'hybrid'}
    if feature_set not in valid_feature_sets:
        raise ValueError(f"feature_set must be one of {valid_feature_sets}")
    
    step_start = time.time()
    print(f"\n Loading and preparing data...")
    # Load embeddings (skip for tabular_only)
    if feature_set != 'tabular_only':
        print(f"  Loading embeddings from: {embedding_location_path}")
        embeddings, individual_ids, index_dts = load_embeddings_from_dir(embedding_location_path)
        print(f"  Creating embedding DataFrame...")
        emb_df = create_embedding_df(embeddings, individual_ids, index_dts)
        
        print(f"  Joining embeddings with features...")
        if feature_set == 'embedding_only':
            df_merged = join_embeddings_with_features(emb_df, df_features[['individual_id', 'index_dt', TARGET_COLUMN, 'ind_id_last_digit']])
        else:
            df_merged = join_embeddings_with_features(emb_df, df_features)
    else:
        if embedding_location_path:
            print(f"  Preparing tabular-only data (no embeddings) joined with member ID im embedding table...")
            # For tabular-only, just apply filters directly
            df_members = load_mbrs_have_embed(embedding_location_path)
            df_merged = join_embeddings_with_features(df_members[['individual_id', 'index_dt']], df_features)
        # using entire table without joining to match 30% samples
        else:
            # For tabular-only, just apply filters directly
            df_merged = df_features.copy()
            df_merged['index_dt'] = pd.to_datetime(df_merged['index_dt']).dt.strftime('%Y-%m-%d')
            if 'mon_6_include' in df_merged.columns:
                df_merged = df_merged[df_merged['mon_6_include'] == 1]
            if 'exclude_ip' in df_merged.columns:
                df_merged = df_merged[(df_merged['exclude_ip'] == 0) | (df_merged['exclude_ip'].isna())]
            if 'include_post_6_status' in df_merged.columns:
                df_merged = df_merged[df_merged['include_post_6_status'] == 1]
            df_merged = df_merged.drop_duplicates(subset=['individual_id', 'index_dt'], keep='last')    
            
    step_times['step1_data_loading'] = time.time() - step_start
    print(f"  Step 1 complete ({step_times['step1_data_loading']:.2f}s)")
    print(f"\n[Step 2/6] Creating data splits...")
    # Create splits
    step_start = time.time()
    splits = create_data_splits(df_merged)
    step_times['step2_create_splits'] = time.time() - step_start
    print(f"  Step 2 complete ({step_times['step2_create_splits']:.2f}s)")
    
    # Identify feature columns
    embedding_features, tabular_features = identify_feature_columns(df_merged)

    
    # Select features based on feature_set
    if feature_set == 'embedding_only':
        feature_cols = embedding_features
    elif feature_set == 'tabular_only':
        feature_cols = tabular_features
    else:  # hybrid
        feature_cols = tabular_features + embedding_features
    
    step_start = time.time()
    print(f"\n[Step 4/6] Preparing feature matrices for each split...")
        
    # Prepare feature matrices for each split
    X_splits, y_splits = {}, {}
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            X_splits[split_name], y_splits[split_name] = prepare_features(split_df, feature_cols)
            
    step_times['step4_prepare_features'] = time.time() - step_start
    print(f"  Step 4 complete ({step_times['step4_prepare_features']:.2f}s)")
    
   
    # Apply downsampling to training set only (if requested)
    downsampled = False
    if downsample_ratio is not None and 'train' in X_splits:
        step_start = time.time()
        print(f"\n[Step 5/6] Rebalance the training dataset with a ratio of {downsample_ratio}...") 
        X_splits['train'], y_splits['train'] = downsample_negatives(
            X_splits['train'], 
            y_splits['train'], 
            ratio=downsample_ratio,
            random_state=random_state
        )
        downsampled = True    
    
    print(f"Finish data preparation, total time: {time.time() - total_start_time}")
    # Pre-compute categorical column indices for CatBoost (only for tabular/hybrid)
    cat_feature_indices = []
    if feature_set != 'embedding_only' and 'train' in X_splits:
        cat_cols = X_splits['train'].select_dtypes(include=['object', 'category']).columns
        cat_feature_indices = [X_splits['train'].columns.get_loc(c) for c in cat_cols]
    
    return PreparedData(
        X_splits=X_splits,
        y_splits=y_splits,
        feature_cols=feature_cols,
        embedding_features=embedding_features,
        tabular_features=tabular_features,
        cat_feature_indices=cat_feature_indices,
        feature_set=feature_set,
        embedding_path=embedding_location_path,
        downsampled=downsampled
    )


# In[146]:


# =============================================================================
# MODEL EVALUATION FUNCTION
# =============================================================================
def evaluate_model_on_splits(
    model,
    X_splits: Dict[str, pd.DataFrame],
    y_splits: Dict[str, pd.Series],
    apply_scaling: bool = False,
    cat_feature_indices: Optional[List[int]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Train model on train split, evaluate on all splits.
    
    Args:
        model: sklearn-compatible model with fit() and predict_proba()
        X_splits: Dict with 'train', 'val', 'test', 'oot' DataFrames
        y_splits: Dict with corresponding target Series
        apply_scaling: Whether to apply StandardScaler
        cat_feature_indices: List of categorical column indices (for CatBoost)
    
    Returns:
        Dict with split names as keys, metrics dict as values
    """
    total_start_time = time.time()
    step_times = {}
    
    # Clone model to avoid modifying original
    model = clone(model)
    
    X_train, y_train = X_splits['train'], y_splits['train']
    
    
    step_start = time.time()

    # Handle scaling
    scaler = None
    if apply_scaling:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        step_times['scaling'] = time.time() - step_start
        print(f"\n Scaling done {step_times['scaling']}...")
    else:
        X_train_processed = X_train

    
    # Handle CatBoost-specific training
    model_type = type(model).__name__
    
    
    step_start = time.time()
    if model_type == 'CatBoostClassifier':
        # CatBoost uses Pool for categorical features
        from catboost import Pool
        cat_indices = cat_feature_indices if cat_feature_indices else []
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_splits['val'], y_splits['val'], cat_features=cat_indices)
        
        model.fit(train_pool, eval_set=val_pool, verbose=0)
    else:
        model.fit(X_train_processed, y_train)
    step_times['model_fit'] = time.time() - step_start
    print(f"\n Fit model done {model_type}: {step_times['model_fit']}")
    
    # Evaluate on all splits
    step_start = time.time()
    results = {}
    for split_name in tqdm(['val', 'test', 'oot', 'oot_strict']):
        print(f"\n Evaluating {split_name}...")
        X_split = X_splits.get(split_name)
        y_split = y_splits.get(split_name)
        
        if X_split is None or len(X_split) == 0:
            continue
        
        # Apply same preprocessing
        if apply_scaling and scaler is not None:
            X_processed = scaler.transform(X_split)
        else:
            X_processed = X_split
        
        # Predict
        if model_type == 'CatBoostClassifier' and cat_feature_indices:
            from catboost import Pool
            pool = Pool(X_split, cat_features=cat_feature_indices)
            y_prob = model.predict_proba(pool)[:, 1]
        else:
            y_prob = model.predict_proba(X_processed)[:, 1]
        
        # Compute metrics
        results[split_name] = compute_split_metrics(np.array(y_split), y_prob)
    step_times['model_eval'] = time.time() - step_start
    print(f"\n Evaluate model done {model_type}: {step_times['model_eval']}")
    return results


# In[147]:


def evaluate_with_prepared_data(
    prepared_data: PreparedData,
    ml_model_object: Any,
    exp_name: str,
    apply_scaling: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a model using pre-prepared data.
    
    This is the efficient way to evaluate multiple models on the same dataset.
    Call prepare_evaluation_data() once, then call this function for each model.
    
    Args:
        prepared_data: PreparedData object from prepare_evaluation_data()
        ml_model_object: Pre-configured sklearn-compatible model
        exp_name: Experiment name for result identification
        apply_scaling: Whether to apply StandardScaler (True for LR, False for tree-based)
    
    Returns:
        Dict with exp_name, model_type, feature_set, and all metrics
    """
    # Determine if we should use categorical features
    use_cat_features = (
        prepared_data.feature_set != 'embedding_only' and 
        len(prepared_data.cat_feature_indices) > 0
    )
    
    # Evaluate model
    split_results = evaluate_model_on_splits(
        model=ml_model_object,
        X_splits=prepared_data.X_splits,
        y_splits=prepared_data.y_splits,
        apply_scaling=apply_scaling,
        cat_feature_indices=prepared_data.cat_feature_indices if use_cat_features else None
    )
    
    # Build output dictionary
    output = {
        'exp_name': exp_name,
        'model_type': type(ml_model_object).__name__,
        'feature_set': prepared_data.feature_set,
        'n_features': len(prepared_data.feature_cols),
    }
    
    # Flatten split results with prefixes
    for split_name, metrics in split_results.items():
        for metric_name, value in metrics.items():
            output[f'{split_name}_{metric_name}'] = value
    
    return output


def evaluate_all_experiments(
    experiment_configs: List[Dict],
    df_features: pd.DataFrame,
    downsample_ratio: Optional[float] = None
) -> pd.DataFrame:
    """
    Evaluate multiple experiments efficiently by grouping by data requirements.
    
    Data is prepared once per unique (embedding_path, feature_set, downsample_ratio) 
    combination, then reused for all models in that group.
    
    Args:
        experiment_configs: List of dicts, each with:
            - embedding_location_path: str
            - ml_model_object: model
            - exp_name: str
            - feature_set: str (optional, default 'embedding_only')
            - apply_scaling: bool (optional, default False)
            - downsample_ratio: float (optional, overrides global downsample_ratio)
        df_features: DataFrame with features and outcomes
        downsample_ratio: Global downsample ratio for all experiments.
                          Use 10.0 to match previous model's 10:1 negative sampling.
                          Can be overridden per-experiment in config.
    
    Returns:
        DataFrame with one row per experiment, all metrics as columns
    """
    from collections import defaultdict
    
    # Group configs by (embedding_path, feature_set, downsample_ratio) to avoid redundant data preparation
    groups = defaultdict(list)

    for config in tqdm(experiment_configs):
        embedding_path = config.get('embedding_location_path', '')
        feature_set = config.get('feature_set', 'embedding_only')
        # Per-experiment downsample_ratio overrides global
        ds_ratio = config.get('downsample_ratio', downsample_ratio)
        # Use tuple as key for grouping (include downsample_ratio since it affects data)
        key = (embedding_path, feature_set, ds_ratio)
        groups[key].append(config)
    
    results = []
    prepared_cache = {}  # Cache prepared data for reuse  
    for (embedding_path, feature_set, ds_ratio), group_configs in tqdm(groups.items()):
        print(f"==========================================")
        # Prepare data once for this group
        cache_key = (embedding_path, feature_set, ds_ratio)
        if cache_key not in prepared_cache:
            ds_str = f", downsample={ds_ratio}:1" if ds_ratio else ""
            print(f"Preparing data for: feature_set={feature_set}{ds_str}, path={embedding_path[:50] if embedding_path else 'N/A'}...")
            prepared_cache[cache_key] = prepare_evaluation_data(
                df_features=df_features,
                embedding_location_path=embedding_path,
                feature_set=feature_set,
                downsample_ratio=ds_ratio
            )
        
        prepared_data = prepared_cache[cache_key]
        
        # Evaluate each model in this group using prepared data
        for config in group_configs:
            model = config['ml_model_object']
            exp_name = config['exp_name']
            apply_scaling = config.get('apply_scaling', False)
            
            print(f"  Evaluating: {exp_name} ({type(model).__name__})")
            
            result = evaluate_with_prepared_data(
                prepared_data=prepared_data,
                ml_model_object=model,
                exp_name=exp_name,
                apply_scaling=apply_scaling
            )
            results.append(result)
    
    return pd.DataFrame(results)


# In[148]:


lr_model = LogisticRegression(
    max_iter=1000, 
    solver='lbfgs', 
    class_weight='balanced',
    random_state=42
)
catboost_model = CatBoostClassifier(
    iterations=2500,
    depth=7,
    learning_rate=0.025,
    grow_policy='SymmetricTree',
    auto_class_weights='Balanced',
    od_wait=80,
    use_best_model=True,
    random_seed=42,
    verbose=0
)
# Match previous model's best configuration
catboost_model_legacy = CatBoostClassifier(
    iterations=2436,
    depth=7,
    learning_rate=0.027,  # Rounded from 0.026766501358942353
    random_strength=3,
    l2_leaf_reg=2.95,
    border_count=136,
    min_data_in_leaf=30,
    grow_policy='SymmetricTree',
    od_wait=84,
    bootstrap_type='Bernoulli',
    subsample=0.79,
    leaf_estimation_iterations=8,
    loss_function='Logloss',
    eval_metric='AUC',
    od_type='Iter',
    use_best_model=True,
    random_seed=42,
    thread_count=-1,
    verbose=0
)


# In[114]:


experiment_configs = [
    # These 3 share (exp1 path, embedding_only) - data prepared ONCE
    # {
    #     'embedding_location_path': f"{EMBEDDING_BASE}/exp1_dense_baseline_pure_legacy",
    #     'ml_model_object': catboost_model,
    #     'exp_name': "exp1_legacy_catboost_emb_only",
    #     'feature_set': 'embedding_only',
    #     'apply_scaling': False
    # },
    {
        'embedding_location_path': f"{EMBEDDING_BASE}/exp2b_flash_learned_pool", # exp_round6
        'ml_model_object': catboost_model,
        'exp_name': "exp_round6_exp2b_catboost_emb_only",
        'feature_set': 'embedding_only',
        'apply_scaling': False
    },
    # {
    #     'embedding_location_path': f"{EMBEDDING_BASE}/exp2b_flash_learned_pool_v2",
    #     'ml_model_object': catboost_model,
    #     'exp_name': "exp2b_catboost_emb_only",
    #     'feature_set': 'embedding_only',
    #     'apply_scaling': False
    # },
    # {
    #     'embedding_location_path': f"{EMBEDDING_BASE}/exp6_auxiliary_free_v3",
    #     'ml_model_object': catboost_model,
    #     'exp_name': "exp6_catboost_emb_only",
    #     'feature_set': 'embedding_only',
    #     'apply_scaling': False
    # },
    # # These 3 share (exp1 path, embedding_only) - join with embedding taboe
    # {
    #     'embedding_location_path': f"{EMBEDDING_BASE}/exp1_dense_baseline_pure_legacy",
    #     'ml_model_object': catboost_model,
    #     'exp_name': "tabular_only_catboost",
    #     'feature_set': 'tabular_only',
    #     'apply_scaling': False
    # },
    # {
    #     'embedding_location_path': f"{EMBEDDING_BASE}/exp1_dense_baseline_pure_legacy",
    #     'ml_model_object': catboost_model,
    #     'exp_name': "exp1_legacy_catboost_hybrid",
    #     'feature_set': 'hybrid',
    #     'apply_scaling': False
    # },
    {
        'embedding_location_path': f"{EMBEDDING_BASE}/exp2b_flash_learned_pool",
        'ml_model_object': catboost_model,
        'exp_name': "exp_round6_exp2b_catboost_hybrid",
        'feature_set': 'hybrid',
        'apply_scaling': False
    },
    # {
    #     'embedding_location_path': f"{EMBEDDING_BASE}/exp2b_flash_learned_pool_v2",
    #     'ml_model_object': catboost_model,
    #     'exp_name': "exp2b_catboost_hybrid",
    #     'feature_set': 'hybrid',
    #     'apply_scaling': False
    # },
    # {
    #     'embedding_location_path': f"{EMBEDDING_BASE}/exp6_auxiliary_free_v3",
    #     'ml_model_object': catboost_model,
    #     'exp_name': "exp6_catboost_hybrid",
    #     'feature_set': 'hybrid',
    #     'apply_scaling': False
    # }, 
    # {
    #     'embedding_location_path': f"", # use full dataset commericial what is the ceiline
    #     'ml_model_object': catboost_model,
    #     'exp_name': "full_tabular_only_catboost",
    #     'feature_set': 'tabular_only',
    #     'apply_scaling': False
    # }

]


# ##### import feature tables

# In[66]:


import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
client = bigquery.Client()
credentials, project_id= google.auth.default()
print('credentials:', credentials, ', project:', project)
import pandas as pd
from tqdm.notebook import tqdm
client = bigquery.Client()


# In[67]:


feature_sql = """
SELECT 
* from edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_ip_heldout_transformer_matched_final_dataset_4_te_experiment_round5_downstream
"""


# In[68]:


df_ip_features = client.query(feature_sql).to_dataframe()


# In[95]:


df_ip_features['ip6'].value_counts()


# In[ ]:


df_ip_features['index_dt'] = pd.to_datetime(df_ip_features['index_dt']).dt.strftime('%Y-%m-%d')
embedding_dfs['exp1_dense_baseline']['index_dt'] = pd.to_datetime(embedding_dfs['exp1_dense_baseline']['index_dt']).dt.strftime('%Y-%m-%d')


# #### Evaluate

# In[115]:


results_df_exp_round6 = evaluate_all_experiments(experiment_configs, df_ip_features, downsample_ratio=10.0)


# In[168]:


results_df.T


# In[75]:


results_df_fulltabular.T


# In[5]:


import pandas as pd
import numpy as np

# Set a seed for reproducibility (optional)
np.random.seed(42)

# Create a DataFrame with 100 rows and 4 columns ('A', 'B', 'C', 'D')
# Integers will be between 0 (inclusive) and 100 (exclusive)
df = pd.DataFrame(
    np.random.randint(0, 100, size=(100, 4)),
    columns=list('ABCD')
)


# In[6]:


df


# In[10]:


# Initialize BigQuery client
# client = bigquery.Client(project=project_id)

# Configure job
job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
)
full_table_id = "edp-prod-storage.edp_ent_sdoheir_cns.a964286_test_load_table_to_bigquery"
# Load data
job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
job.result()  # Wait for completion


# ### Medicare embedding generation

# In[126]:


import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
client = bigquery.Client()
credentials, project= google.auth.default()
print('credentials:', credentials, ', project:', project)
import pandas as pd
from tqdm.notebook import tqdm
client = bigquery.Client()


# In[128]:


# import members not in the trainingset of the transformer
# Didn't sampled and 30% generate all 2.78M
medicare_sql_code = """
select * from edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_holdout_members_with_features
"""
df_me = client.query(medicare_sql_code).to_dataframe()


# In[151]:


df_me['lob'] = 'Medicare'


# In[152]:


# sample before and after 2023-10-16 (post part will be used for oot validation)
# 0.3 samples for efficent evaluations of embeddings
df_me['index_dt'] = pd.to_datetime(df_me['index_dt'])
df_me_b4_oct = df_me[df_me['index_dt'] <= pd.to_datetime("2023-10-16")]
df_me_after_oct = df_me[df_me['index_dt'] > pd.to_datetime("2023-10-16")]
df_me_b4_oct_sample = df_me_b4_oct.sample(frac=0.3, random_state=42)
df_me_after_oct_sample = df_me_after_oct.sample(frac=0.3, random_state=42)
df_me_sample = pd.concat([df_me_b4_oct_sample,
                         df_me_after_oct_sample])


# In[156]:


df_me_sample.head()


# In[ ]:


del df_me, df_me_b4_oct, df_me_after_oct, df_me_b4_oct_sample, df_me_after_oct_sample


# In[154]:


MODEL_PATHS = {
    # Experiment 1: Dense Baseline (no Flash Attention, no MoE)
    'exp1_dense_baseline_pure_legacy': 
        'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp1_dense_baseline_pure_legacy/saved_models/'
        'exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp1_dense_baseline_bs128_ep1_d256_20251230_055716_final.pt', 
    
    # Experiment 1b: Dense Baseline (same opt config as 2b and 6)
    'exp1_dense_baseline_opt_config': 
        'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp1_dense_baseline_opt_config/saved_models/'
        'exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2_exp1_dense_baseline_bs64_ep1_d256_20260108_183616_final.pt',

    # Experiment 2b: Flash Attention + Learned Pooling (no MoE)
    'exp2b_flash_learned_pool_v2': 
        'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/exp2b_flash_learned_pool_v2/saved_models/'
        'exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20251230_114137_final.pt',
    
    # Experiment 6: Flash + MoE with DeepSeek auxiliary-free balancing
    'exp6_auxiliary_free_v3': 
        'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/'
        'exp6_auxiliary_free_v3/saved_models/'
        'exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_20251231_152438_final.pt',
    
    # Round 6; 
    # 'exp2b_flash_learned_pool': 
    # 'logs/exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2/'
    # 'exp2b_flash_learned_pool/saved_models/'
    # 'exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_20260110_112709_final.pt'
}


# In[ ]:


import time

results = {}
batch_size = 64
output_dir = "embedding_output/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2"
# output_dir = "embedding_output/exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2"
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"
LOB = 'medicare'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for exp_name, model_path in tqdm(MODEL_PATHS.items()):
    cleanup_gpu_memory(verbose=False)
    model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
        model_path=MODEL_PATHS[exp_name],
        device=device,
        verbose=True
    )
    
    inference_start_time = time.time()
    embeddings, individual_ids, index_dts = generate_embeddings(
        model=model,
        config=config,
        data=df_me_sample,
        device=device,
        id_column='individual_id',  # Commercial uses individual_id
        lob_value=None,              # Medicare data already has lob column
        desc_prefix='Mecicare',
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        verbose=True,
        multi_gpu=True,           
        moe_config=moe_config, 
    )
    inference_duration = time.time() - inference_start_time
    print(f"Inference duration for {exp_name}: {round(inference_duration/3600, 2):.2f} hr)")
    exp_output_dir = os.path.join(output_dir, exp_name)
    # embeddings_path = save_embeddings(
    #     embeddings=embeddings,
    #     individual_ids=individual_ids,
    #     index_dts=index_dts,
    #     output_path=exp_output_dir,
    #     model_name=exp_name,
    #     additional_metadata={
    #         'model_path': model_path,
    #         'model_type': model_type,
    #         'use_mixed_precision': use_mixed_precision,
    #     }
    # )
    safe_exp_name = exp_name.replace('-', '_').replace('.', '_')
    table_name = f"a964286_te4exp_3lob_exp_round5_v2_{safe_exp_name}_{LOB}_all_sample_embedding"
    bq_table_path = save_embeddings_to_bigquery(
        embeddings=embeddings,
        individual_ids=individual_ids,
        index_dts=index_dts,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_name=table_name,
        exp_name=exp_name,
        model_type=model_type,
        if_exists="replace"
    )
    results[exp_name] = {
        'embeddings_path': embeddings_path,
        'embedding_shape': embeddings.shape,
        'model_type': model_type,
        'model_path': model_path,
        'inference_duration_hr': round(inference_duration/3600, 2),
        'status': 'success'
    }

    # Free model memory
    del model
    del embeddings
    torch.cuda.empty_cache()


# In[ ]:





# ### Medicaid embedding generation

# In[34]:


import os
import sys
import glob
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.base import clone
from catboost import CatBoostClassifier, Pool

# BigQuery
import google.auth
from google.cloud import bigquery


# #### Configs

# In[38]:


# BigQuery Tables - CORRECTED
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"

# =============================================================================
# HELDOUT TABLES (members NOT in TE pretraining 10% sample)
# These are used for downstream evaluation to avoid data leakage
# =============================================================================
HELDOUT_FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_heldout_non_embedding_features"
HELDOUT_OUTCOME_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_heldout_outcome_ip"
HELDOUT_TE_INPUT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_heldout_te_inference_input"

# =============================================================================
# FULL DATASET TABLES (all members, including those in pretrain)
# Only use these for reference/comparison, NOT for downstream evaluation
# =============================================================================
FULL_FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_non_embedding_features"
FULL_OUTCOME_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip"

# =============================================================================
# TE CROSSWALK TABLES (for ID mapping between formats)
# =============================================================================
MEMBER_CROSSWALK_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_Medicaid_member_train_ending"
TE_SEQUENCE_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_Medicaid_o3_train_ending"
PRETRAIN_10PCT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_Combined_All_LOB_o3_train_10pct_sample"

# Legacy table references (from Eric Ma's original pipeline - for reference only)
# These are from the anbc-hcb-dev project, not used in current downstream eval
# LEGACY_FEATURES_TABLE = "anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_non_embedding_features"
# LEGACY_EMBEDDINGS_TABLE = "anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_embeddings"
# LEGACY_OUTCOME_TABLE = "anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip"

# Default to HELDOUT tables for downstream evaluation
FEATURES_TABLE = HELDOUT_FEATURES_TABLE
OUTCOME_TABLE = HELDOUT_OUTCOME_TABLE

# Note: Embeddings will come from transformer inference on HELDOUT_TE_INPUT_TABLE
# No pre-computed embeddings table for heldout - we generate them fresh

# Target variable
TARGET_COLUMN = "acute_ip_flag"

# Member ID column (primary key for joining)
MEMBER_KEY = "asdb_member_key"

# Random seeds for reproducibility (matches original Eric Ma pipeline)
RANDOM_STATE = 35  # For train/test split (matches Eric's train_test_split random_state)
UNDERSAMPLE_RANDOM_STATE = 53  # For undersampling (Eric uses 53 for RandomUnderSampler)
CATBOOST_RANDOM_SEED = 53  # For CatBoost model (Eric uses 53 for random_seed)

# Class imbalance handling
# Original finding: CatBoost works better with 0.2 undersampling ratio
CATBOOST_UNDERSAMPLE_RATIO = 0.2  # 20% minority-to-majority (5:1)
XGBOOST_UNDERSAMPLE_RATIO = 0.03  # 3% minority-to-majority (~33:1)

# Train/Val/Test split ratios (original: 80/10/10)
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# =============================================================================
# OUT-OF-TIME (OOT) VALIDATION CONFIGURATION
# =============================================================================
# For time-based train/test split similar to commercial IP:
# - Data before cutoff: used for train/val/test (stratified random split)
# - Data after cutoff: used for OOT (out-of-time) validation
# This tests temporal generalization of the model.
#
# Note: Commercial IP uses ind_id_last_digit for deterministic splitting.
# Medicaid doesn't have this column, so we use stratified random split
# for train/val/test, matching Eric's original Medicaid IP pipeline.
OOT_CUTOFF_DATE = "2023-10-16"  # Same as commercial IP for consistency

# Sampling fraction for efficient embedding generation (optional)
# Set to None to use full data, or 0.3 for 30% sample like commercial
EMBEDDING_SAMPLE_FRAC = None  # Use full data for Medicaid heldout (already ~90% of total)

# =============================================================================
# CATBOOST TUNED HYPERPARAMETERS
# =============================================================================
# From Optuna optimization in original pipeline (optuna_results_catboost.csv)
# Best trial 42: AUC = 0.8737 (from optuna_catboost.log)
# Note: Eric's final model uses params from lines 513-521 of catboost.py
CATBOOST_TUNED_PARAMS = {
    'learning_rate': 0.015742881221129403,
    'iterations': 2665,
    'l2_leaf_reg': 0.222046549398224,
    'depth': 7,
    'random_seed': CATBOOST_RANDOM_SEED,  # Eric uses 53 for CatBoost
    'verbose': 0,
    'thread_count': -1,  # Use all available threads (-1), Eric used 15
    'use_best_model': True,
    # Note: Original didn't use auto_class_weights, relied on undersampling instead
}

# Alternative: Balanced class weights model (without undersampling)
CATBOOST_BALANCED_PARAMS = {
    'iterations': 2500,
    'depth': 7,
    'learning_rate': 0.025,
    'grow_policy': 'SymmetricTree',
    'auto_class_weights': 'Balanced',
    'od_wait': 80,
    'use_best_model': True,
    'random_seed': CATBOOST_RANDOM_SEED,  # Use same seed as tuned params
    'verbose': 0,
    'thread_count': -1,
}

# =============================================================================
# SELECTED FEATURES FROM RFECV
# =============================================================================
# These 243 non-embedding features were selected by RFECV in the original pipeline
# The full list (499) includes these + 256 embedding features (emb0-emb255)

SELECTED_TABULAR_FEATURES = [
    # COA Population Group (categorical)
    'coa_population_group',
    
    # ED Visits - Year 1
    'sum_ed_visits_yr1', 'ed_flag_yr1', 'sum_avoidable_yr1', 'sum_unnecessary_yr1',
    'sum_preventable_yr1', 'low_sev_ed_visits_yr1', 'low_med_sev_ed_visits_yr1',
    'med_sev_ed_visits_yr1', 'med_high_sev_ed_visits_yr1', 'high_sev_ed_visits_yr1',
    'high_sev_ed_flag_yr1',
    
    # ED Visits - Year 2
    'sum_ed_visits_yr2', 'sum_avoidable_yr2', 'sum_preventable_yr2',
    'med_sev_ed_visits_yr2', 'med_high_sev_ed_visits_yr2', 'high_sev_ed_visits_yr2',
    
    # IP Admits
    'sum_acute_ip_admits_yr1', 'sum_acute_calc_los_yr1',
    'sum_acute_ip_admits_yr2', 'sum_acute_calc_los_yr2',
    
    # OP Visits
    'sum_op_visits_yr1', 'sum_op_visits_yr2',
    
    # EMIS Claims - Year 1
    'emis_community_clm_yr1', 'emis_ed_clm_yr1', 'emis_hh_clm_yr1', 'emis_home_clm_yr1',
    'emis_ip_clm_yr1', 'emis_ins_clm_yr1', 'emis_lab_clm_yr1', 'emis_mrx_clm_yr1',
    'emis_mh_clm_yr1', 'emis_misc_clm_yr1', 'emis_pcp_clm_yr1', 'emis_radio_clm_yr1',
    'emis_ambul_clm_yr1', 'emis_spec_clm_yr1',
    
    # LTC and COE Claims - Year 1
    'ltc_clm_yr1', 'coe_ip_hos_clm_yr1', 'coe_ip_non_hos_clm_yr1', 'coe_lab_clm_yr1',
    'coe_ltc_community_clm_yr1', 'coe_ltc_home_clm_yr1', 'coe_ltc_ins_clm_yr1',
    'coe_other_clm_yr1', 'coe_op_hos_clm_yr1', 'coe_op_non_hos_clm_yr1',
    'coe_anesth_clm_yr1', 'coe_eval_clm_yr1', 'coe_maternity_clm_yr1',
    'coe_mrx_clm_yr1', 'coe_mh_clm_yr1', 'coe_phy_clm_yr1', 'coe_surg_clm_yr1',
    'coe_radio_clm_yr1', 'uc_clm_yr1', 'obs_clm_yr1',
    
    # EMIS Claims - Year 2
    'emis_community_clm_yr2', 'emis_ed_clm_yr2', 'emis_hh_clm_yr2', 'emis_home_clm_yr2',
    'emis_ip_clm_yr2', 'emis_ins_clm_yr2', 'emis_lab_clm_yr2', 'emis_mrx_clm_yr2',
    'emis_mh_clm_yr2', 'emis_misc_clm_yr2', 'emis_pcp_clm_yr2', 'emis_radio_clm_yr2',
    'emis_ambul_clm_yr2', 'emis_spec_clm_yr2',
    
    # LTC and COE Claims - Year 2
    'ltc_clm_yr2', 'coe_ip_hos_clm_yr2', 'coe_ip_non_hos_clm_yr2', 'coe_lab_clm_yr2',
    'coe_other_clm_yr2', 'coe_op_hos_clm_yr2', 'coe_op_non_hos_clm_yr2',
    'coe_anesth_clm_yr2', 'coe_eval_clm_yr2', 'coe_maternity_clm_yr2',
    'coe_mrx_clm_yr2', 'coe_mh_clm_yr2', 'coe_phy_clm_yr2', 'coe_surg_clm_yr2',
    'coe_radio_clm_yr2', 'uc_clm_yr2', 'obs_clm_yr2',
    
    # Chronic Conditions (binary flags)
    'IDA', 'ANX', 'OST', 'AST', 'CHO', 'burns', 'CBD', 'CHF', 'CRF', 'CHD',
    'COP', 'DIA', 'esrd', 'EPL', 'CRO', 'MOH', 'HepC', 'HYP', 'HYC',
    'meta_cancer', 'liver_dis', 'MSS', 'OBE', 'oud', 'paralysis', 'hmd',
    'PVD', 'autoimmune', 'SCA', 'spinal_inj', 'back', 'substance', 'ALC', 'psychoses',
    'major_chronic_cnt',
    
    # Pharmacy Features - Year 1
    'rx_claim_cnt_yr1', 'days_supply_sum_yr1', 'ndc_cnt_yr1', 'gpi_cnt_yr1',
    'gpi4_cnt_yr1', 'gpi2_cnt_yr1', 'retail_fills_yr1', 'mail_order_fills_yr1',
    'generic_fills_yr1', 'branded_generic_fills_yr1', 'ss_brand_fills_yr1',
    'ms_brand_fills_yr1', 'formulary_fills_yr1', 'maint_drug_fills_yr1',
    'antidiabetic_scripts_yr1', 'antidiabetic_days_supply_yr1',
    'beta_blocker_scripts_yr1', 'beta_blocker_days_supply_yr1',
    'antihypertensive_scripts_yr1', 'antihypertensive_days_supply_yr1',
    'lipid_lowering_scripts_yr1', 'lipid_lowering_days_supply_yr1',
    'calcium_channel_blk_scripts_yr1', 'calcium_channel_blk_days_supply_yr1',
    'diuretic_scripts_yr1', 'diuretic_days_supply_yr1',
    'antianginal_agent_scripts_yr1', 'antianginal_agent_days_supply_yr1',
    'antidepressant_scripts_yr1', 'antidepressant_days_supply_yr1',
    'antipsychotic_scripts_yr1', 'antipsychotic_days_supply_yr1',
    'antianxiety_days_supply_yr1', 'anticonvulsant_scripts_yr1',
    'anticonvulsant_days_supply_yr1', 'inhaled_steroid_scripts_yr1',
    'inhaled_steroid_days_supply_yr1',
    
    # Pharmacy Features - Year 2
    'rx_claim_cnt_yr2', 'days_supply_sum_yr2', 'ndc_cnt_yr2', 'gpi_cnt_yr2',
    'gpi4_cnt_yr2', 'gpi2_cnt_yr2', 'retail_fills_yr2', 'generic_fills_yr2',
    'branded_generic_fills_yr2', 'ss_brand_fills_yr2', 'ms_brand_fills_yr2',
    'formulary_fills_yr2', 'maint_drug_fills_yr2',
    'antidiabetic_scripts_yr2', 'antidiabetic_days_supply_yr2',
    'beta_blocker_scripts_yr2', 'beta_blocker_days_supply_yr2',
    'antihypertensive_scripts_yr2', 'antihypertensive_days_supply_yr2',
    'lipid_lowering_scripts_yr2', 'lipid_lowering_days_supply_yr2',
    'calcium_channel_blk_scripts_yr2', 'calcium_channel_blk_days_supply_yr2',
    'diuretic_days_supply_yr2', 'antianginal_agent_scripts_yr2',
    'antidepressant_scripts_yr2', 'antidepressant_days_supply_yr2',
    'antipsychotic_scripts_yr2', 'antipsychotic_days_supply_yr2',
    'antianxiety_scripts_yr2', 'antianxiety_days_supply_yr2',
    'anticonvulsant_scripts_yr2', 'anticonvulsant_days_supply_yr2',
    'inhaled_steroid_days_supply_yr2',
    
    # Demographics
    'agenbr', 'gender', 'ethnicity_code', 'primarylanguage_desc',
    'tenure_yr1', 'tenure_yr2', 'urbsubr',
    
    # SDOH Scores
    'zip_weight_avg_medinc', 'acs_social_risk_score', 'sdi_score', 'svi_score',
    'adi_score', 'citizenship_index', 'education_index', 'food_access',
    'health_access', 'health_habits', 'housing_desert', 'housing_ownership',
    'housing_quality', 'income_index', 'income_inequality', 'language_score',
    'natural_disaster', 'poverty_score', 'proactive_health', 'racial_diversity',
    'social_isolation', 'technology_access', 'transport_access',
    'unemployment_index', 'water_quality', 'disability_score', 'health_infra',
    'csdi_social_risk_score',
    
    # Healthcare Utilization
    'sum_pcp', 'sum_spec', 'sum_ob', 'sum_dme', 'sum_chol_lab', 'sum_a1c_lab',
    'sum_chemo',
    
    # CMS Screening Flags
    'cms_alc_scrn', 'cms_col_scrn', 'cms_hepb_scrn', 'cms_nutrition',
    'cms_sti_scrn', 'cms_mam_scrn',
]

# Embedding features (256 dimensions)
EMBEDDING_FEATURES = [f'emb{i}' for i in range(256)]

# Categorical features requiring special handling
CATEGORICAL_FEATURES = [
    'coa_population_group', 'gender', 'ethnicity_code',
    'primarylanguage_desc', 'urbsubr',
    'cms_alc_scrn', 'cms_col_scrn', 'cms_hepb_scrn', 'cms_nutrition',
    'cms_sti_scrn', 'cms_mam_scrn',
]


# #### Eval metrics

# In[24]:


# =============================================================================
# EVALUATION METRICS FUNCTIONS
# =============================================================================
# These match the original Medicaid IP model evaluation exactly

def lift_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """
    Calculate lift at top percentile.
    
    Lift = precision@k / baseline_prevalence
    
    This is the primary metric for the Medicaid IP model, measuring how
    many times better the model is at identifying positives in the top k%.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        pct: Percentile (0.01 for 1%, 0.10 for 10%)
        
    Returns:
        Lift value (e.g., 20 means 20x better than random)
    """
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    precision_at_k = y_true[top_k_indices].mean()
    baseline = y_true.mean()
    return precision_at_k / baseline if baseline > 0 else 0.0


def true_positives_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> int:
    """Count true positives in top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    return int(y_true[top_k_indices].sum())


def precision_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """Calculate precision at top percentile (PPV@k)."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_indices].mean())


def sensitivity_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """
    Calculate sensitivity at top percentile.
    
    Sensitivity = TP / (TP + FN) for members in top k%
    """
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    
    # Binary prediction: 1 for top k%, 0 for rest
    y_pred_binary = np.zeros(n, dtype=int)
    y_pred_binary[top_k_indices] = 1
    
    tp = (y_true[top_k_indices] == 1).sum()
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum()
    
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def compute_split_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single split.
    
    This function replicates the exact metrics from the original Medicaid IP model:
    - ROC-AUC: Overall discrimination ability
    - AUC-PR: Precision-Recall AUC (important for imbalanced data)
    - Brier Score: Calibration metric
    - Lift@1%, Lift@5%, Lift@10%: Key business metrics
    - PPV@1%, PPV@10%: Precision at top percentiles
    - Sensitivity@1%, Sensitivity@10%: Recall at top percentiles
    - TP@1%: True positives captured in top 1%
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        
    Returns:
        Dict with metric names as keys
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    return {
        # Discrimination metrics
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        
        # Lift metrics (primary business metrics)
        'lift_1pct': lift_at_percentage(y_true, y_prob, 0.01),
        'lift_5pct': lift_at_percentage(y_true, y_prob, 0.05),
        'lift_10pct': lift_at_percentage(y_true, y_prob, 0.10),
        
        # PPV (Precision) at percentiles
        'ppv_1pct': precision_at_percentage(y_true, y_prob, 0.01) * 100,
        'ppv_10pct': precision_at_percentage(y_true, y_prob, 0.10) * 100,
        
        # Sensitivity (Recall) at percentiles
        'sensitivity_1pct': sensitivity_at_percentage(y_true, y_prob, 0.01) * 100,
        'sensitivity_10pct': sensitivity_at_percentage(y_true, y_prob, 0.10) * 100,
        
        # True positives captured
        'tp_1pct': true_positives_at_percentage(y_true, y_prob, 0.01),
        
        # Sample info
        'n_samples': len(y_true),
        'n_positives': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
    }


# #### Load data

# In[23]:


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_medicaid_heldout_data(
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load HELDOUT Medicaid IP data from BigQuery for downstream evaluation.
    
    This function loads from the HELDOUT tables which contain members
    NOT in the TE pretraining 10% sample, ensuring no data leakage.
    
    Tables loaded:
    - HELDOUT_FEATURES_TABLE: Non-embedding features (already filtered)
    - HELDOUT_OUTCOME_TABLE: Target variable (acute_ip_flag)
    
    Note: Embeddings are NOT loaded here - they must be generated via
    transformer inference on HELDOUT_TE_INPUT_TABLE and merged separately.
    
    Args:
        sample_frac: Optional sampling fraction for testing (e.g., 0.1 for 10%)
        random_state: Random seed for sampling
        verbose: Print progress information
        
    Returns:
        DataFrame with features and outcome (no embeddings)
    """
    client = bigquery.Client()
    
    if verbose:
        print(f"\n{'='*70}")
        print("LOADING MEDICAID IP HELDOUT DATA FROM BIGQUERY")
        print(f"{'='*70}")
        print(f"Features table: {HELDOUT_FEATURES_TABLE}")
        print(f"Outcome table: {HELDOUT_OUTCOME_TABLE}")
    
    # Step 1: Load heldout non-embedding features (already filtered)
    if verbose:
        print("\n[Step 1/3] Loading heldout non-embedding features...")
    
    features_sql = f"""
    SELECT *
    FROM `{HELDOUT_FEATURES_TABLE}`
    """
    
    df_features = client.query(features_sql).to_dataframe()
    if verbose:
        print(f"  Features loaded: {len(df_features):,} rows, {len(df_features.columns)} columns")
    
    # Step 2: Load heldout outcomes
    if verbose:
        print("\n[Step 2/3] Loading heldout outcomes...")
    
    outcomes_sql = f"""
    SELECT
        asdb_member_key,
        acute_ip_flag
    FROM `{HELDOUT_OUTCOME_TABLE}`
    """
    
    df_outcomes = client.query(outcomes_sql).to_dataframe()
    if verbose:
        print(f"  Outcomes loaded: {len(df_outcomes):,} rows")
        print(f"  Positive rate: {df_outcomes[TARGET_COLUMN].mean()*100:.2f}%")
    
    # Step 3: Merge features with outcomes
    if verbose:
        print("\n[Step 3/3] Merging features and outcomes...")
    
    df_features = df_features.set_index(MEMBER_KEY)
    df_outcomes = df_outcomes.set_index(MEMBER_KEY)
    
    df_merged = df_features.merge(df_outcomes, left_index=True, right_index=True, how='inner')
    df_merged = df_merged.reset_index()
    
    if verbose:
        print(f"  Merged dataset: {len(df_merged):,} rows, {len(df_merged.columns)} columns")
    
    # Optional sampling for testing
    if sample_frac is not None:
        if verbose:
            print(f"\n  Sampling {sample_frac*100:.0f}% of data...")
        df_merged = df_merged.sample(frac=sample_frac, random_state=random_state)
        if verbose:
            print(f"  Sampled dataset: {len(df_merged):,} rows")
    
    if verbose:
        print(f"\n✅ Data loading complete!")
        print(f"   Final shape: {df_merged.shape}")
        print(f"   Positive rate: {df_merged[TARGET_COLUMN].mean()*100:.2f}%")
        print(f"   ⚠️  Note: Embeddings NOT included - merge from NPZ after inference")
        print(f"{'='*70}\n")
    
    return df_merged


def load_te_inference_input(
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load TE inference input data for generating embeddings for heldout members.
    
    This table contains the raw TE sequences (cd, gender_cd, age_in_months)
    needed to run transformer inference and generate embeddings.
    
    Returns:
        DataFrame with columns: asdb_member_key, individual_id, index_dt,
        gender_cd, age_in_months, cd, dt_cnt
    """
    client = bigquery.Client()
    
    if verbose:
        print(f"\n{'='*70}")
        print("LOADING TE INFERENCE INPUT FOR HELDOUT MEMBERS")
        print(f"{'='*70}")
        print(f"Table: {HELDOUT_TE_INPUT_TABLE}")
    
    sql = f"""
    SELECT *
    FROM `{HELDOUT_TE_INPUT_TABLE}`
    """
    
    df = client.query(sql).to_dataframe()
    
    if verbose:
        print(f"  Loaded: {len(df):,} rows")
        print(f"  Unique members: {df[MEMBER_KEY].nunique():,}")
        print(f"  Avg sequence days: {df['dt_cnt'].mean():.1f}")
    
    # Optional sampling
    if sample_frac is not None:
        if verbose:
            print(f"\n  Sampling {sample_frac*100:.0f}% of data...")
        df = df.sample(frac=sample_frac, random_state=random_state)
        if verbose:
            print(f"  Sampled: {len(df):,} rows")
    
    if verbose:
        print(f"{'='*70}\n")
    
    return df


# Legacy function for backward compatibility with original Eric Ma pipeline
def load_medicaid_data_from_bigquery(
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    DEPRECATED: Use load_medicaid_heldout_data() for downstream evaluation.
    
    This function is kept for backward compatibility but now loads from
    heldout tables. For the full pipeline, use load_medicaid_heldout_data()
    and merge embeddings separately.
    """
    if verbose:
        print("⚠️  Note: load_medicaid_data_from_bigquery() now loads from HELDOUT tables")
        print("   For new embeddings, use load_medicaid_heldout_data() + merge_new_embeddings_with_features()")
    
    return load_medicaid_heldout_data(
        sample_frac=sample_frac,
        random_state=random_state,
        verbose=verbose
    )




# #### Data and feature preprocessing

# In[21]:


def preprocess_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Preprocess features replicating the original Medicaid IP pipeline.
    
    Preprocessing steps:
    1. Fill embedding columns (emb0-emb255) with 0
    2. Fill numeric columns with 0
    3. Fill string/categorical columns with empty string
    4. Encode gender: M -> 1, F -> 0, other -> -1
    
    Note: CatBoost handles categorical features natively, so we don't
    need to one-hot encode them. We just ensure proper types.
    
    Args:
        df: Raw DataFrame
        verbose: Print progress
        
    Returns:
        Preprocessed DataFrame
    """
    import re
    from pandas.api.types import is_integer_dtype, is_float_dtype
    
    df = df.copy()
    
    if verbose:
        print("Preprocessing features...")
    
    # Step 1: Fill embedding columns with 0
    emb_pattern = r'^emb\d+$'
    emb_cols = [col for col in df.columns if re.match(emb_pattern, col)]
    if emb_cols:
        df[emb_cols] = df[emb_cols].fillna(0)
        if verbose:
            print(f"  Filled {len(emb_cols)} embedding columns with 0")
    
    # Step 2: Fill numeric columns with 0, string columns with ''
    numeric_filled = 0
    string_filled = 0
    
    for col in df.columns:
        if col in emb_cols or col == TARGET_COLUMN or col == MEMBER_KEY:
            continue
            
        if is_integer_dtype(df[col]) or is_float_dtype(df[col]):
            df[col] = df[col].fillna(0)
            numeric_filled += 1
        else:
            try:
                df[col] = df[col].fillna('')
                string_filled += 1
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not process column {col}: {e}")
    
    if verbose:
        print(f"  Filled {numeric_filled} numeric columns with 0")
        print(f"  Filled {string_filled} string columns with ''")
    
    # Step 3: Encode gender (matches original)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(-1).astype(int)
        if verbose:
            print("  Encoded gender: M->1, F->0, other->-1")
    
    if verbose:
        print("✅ Preprocessing complete!")
    
    return df


def downsample_negatives(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = CATBOOST_UNDERSAMPLE_RATIO,
    random_state: int = UNDERSAMPLE_RANDOM_STATE  # Eric uses 53 for undersampling
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Downsample negative class to achieve target ratio.
    
    The original Medicaid IP model used undersampling as the primary
    class imbalance strategy. CatBoost worked best with 0.2 ratio
    (20% minority, meaning 5:1 negative-to-positive ratio).
    
    Args:
        X: Feature DataFrame
        y: Target Series
        ratio: Minority class ratio (e.g., 0.2 for 5:1)
        random_state: Random seed
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    np.random.seed(random_state)
    
    pos_mask = y == 1
    neg_mask = y == 0
    
    pos_indices = X.index[pos_mask].tolist()
    neg_indices = X.index[neg_mask].tolist()
    
    n_positives = len(pos_indices)
    n_negatives = len(neg_indices)
    
    # Calculate target number of negatives based on ratio
    # ratio = n_positives / (n_positives + n_negatives_target)
    # Solving: n_negatives_target = n_positives * (1 - ratio) / ratio
    target_n_negatives = int(n_positives * (1 - ratio) / ratio)
    
    if n_negatives <= target_n_negatives:
        print(f"  Downsampling: No action needed (current ratio: {n_positives/(n_positives+n_negatives):.3f})")
        return X, y
    
    # Randomly sample negatives
    sampled_neg_indices = np.random.choice(neg_indices, size=target_n_negatives, replace=False)
    keep_indices = pos_indices + sampled_neg_indices.tolist()
    
    X_resampled = X.loc[keep_indices].copy()
    y_resampled = y.loc[keep_indices].copy()
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_resampled))
    X_resampled = X_resampled.iloc[shuffle_idx].reset_index(drop=True)
    y_resampled = y_resampled.iloc[shuffle_idx].reset_index(drop=True)
    
    new_ratio = y_resampled.sum() / len(y_resampled)
    print(f"  Downsampling: {n_negatives}:{n_positives} -> "
          f"{target_n_negatives}:{n_positives} (ratio: {new_ratio:.3f})")
    
    return X_resampled, y_resampled


# =============================================================================
# DATA PREPARATION
# =============================================================================

@dataclass
class MedicaidPreparedData:
    """
    Container for prepared Medicaid IP evaluation data.
    
    Prepare data once using prepare_medicaid_evaluation_data(), 
    then evaluate multiple models using evaluate_with_prepared_data().
    """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_cols: List[str]
    embedding_features: List[str]
    tabular_features: List[str]
    cat_feature_indices: List[int]
    feature_set: str
    downsampled: bool
    original_train_size: int
    

def create_train_val_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.
    
    Replicates the original Medicaid IP model split strategy:
    - 80% train, 10% validation, 10% test
    - Stratified by target variable to preserve class distribution
    
    Args:
        df: Full DataFrame with features and target
        test_size: Fraction for test set (default 0.1)
        val_size: Fraction for validation set (default 0.1)
        random_state: Random seed
        verbose: Print info
        
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    if verbose:
        print("\nCreating stratified train/val/test splits...")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN]
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val[TARGET_COLUMN]
    )
    
    splits = {
        'train': train.reset_index(drop=True),
        'val': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
    }
    
    if verbose:
        for name, split_df in splits.items():
            prevalence = split_df[TARGET_COLUMN].mean() * 100
            print(f"  {name}: {len(split_df):,} rows, "
                  f"{int(split_df[TARGET_COLUMN].sum()):,} positives ({prevalence:.2f}%)")
    
    return splits


def prepare_medicaid_evaluation_data(
    df: pd.DataFrame,
    feature_set: str = 'hybrid',
    apply_downsampling: bool = True,
    downsample_ratio: float = CATBOOST_UNDERSAMPLE_RATIO,
    split_random_state: int = RANDOM_STATE,
    undersample_random_state: int = UNDERSAMPLE_RANDOM_STATE,
    verbose: bool = True
) -> MedicaidPreparedData:
    """
    Prepare Medicaid IP data for model evaluation.
    
    This function encapsulates the complete data preparation pipeline
    from the original Medicaid IP model:
    1. Preprocessing (missing values, encoding)
    2. Train/val/test splitting (stratified)
    3. Feature selection based on feature_set
    4. Optional downsampling of training set
    
    Args:
        df: Raw Medicaid IP DataFrame
        feature_set: One of 'embedding_only', 'tabular_only', 'hybrid'
        apply_downsampling: Whether to downsample training set
        downsample_ratio: Undersampling ratio (default 0.2 for CatBoost)
        split_random_state: Random seed for train/test split (default 35, Eric's)
        undersample_random_state: Random seed for undersampling (default 53, Eric's)
        verbose: Print progress
        
    Returns:
        MedicaidPreparedData object ready for model evaluation
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PREPARING MEDICAID IP DATA FOR EVALUATION")
        print(f"{'='*70}")
        print(f"Feature set: {feature_set}")
        print(f"Downsampling: {apply_downsampling} (ratio: {downsample_ratio})")
    
    # Validate feature_set
    valid_feature_sets = {'embedding_only', 'tabular_only', 'hybrid'}
    if feature_set not in valid_feature_sets:
        raise ValueError(f"feature_set must be one of {valid_feature_sets}")
    
    # Step 1: Preprocess
    if verbose:
        print("\n[Step 1/4] Preprocessing features...")
    df_processed = preprocess_features(df, verbose=verbose)
    
    # Step 2: Split data (using split_random_state=35 to match Eric's train_test_split)
    if verbose:
        print("\n[Step 2/4] Creating train/val/test splits...")
    splits = create_train_val_test_split(df_processed, random_state=split_random_state, verbose=verbose)
    
    # Step 3: Select features based on feature_set
    if verbose:
        print(f"\n[Step 3/4] Selecting features for '{feature_set}'...")
    
    # Identify available features
    available_tabular = [f for f in SELECTED_TABULAR_FEATURES if f in df_processed.columns]
    available_embedding = [f for f in EMBEDDING_FEATURES if f in df_processed.columns]
    
    if verbose:
        print(f"  Available tabular features: {len(available_tabular)}")
        print(f"  Available embedding features: {len(available_embedding)}")
    
    if feature_set == 'embedding_only':
        feature_cols = available_embedding
    elif feature_set == 'tabular_only':
        feature_cols = available_tabular
    else:  # hybrid
        feature_cols = available_tabular + available_embedding
    
    if verbose:
        print(f"  Selected features: {len(feature_cols)}")
    
    # Prepare X and y for each split
    X_train = splits['train'][feature_cols].copy()
    X_val = splits['val'][feature_cols].copy()
    X_test = splits['test'][feature_cols].copy()
    y_train = splits['train'][TARGET_COLUMN].astype(int)
    y_val = splits['val'][TARGET_COLUMN].astype(int)
    y_test = splits['test'][TARGET_COLUMN].astype(int)
    
    original_train_size = len(X_train)
    
    # Step 4: Apply downsampling to training set (using undersample_random_state=53 to match Eric's)
    downsampled = False
    if apply_downsampling:
        if verbose:
            print(f"\n[Step 4/4] Applying downsampling to training set...")
        X_train, y_train = downsample_negatives(
            X_train, y_train, 
            ratio=downsample_ratio, 
            random_state=undersample_random_state  # Eric uses 53 for undersampling
        )
        downsampled = True
    else:
        if verbose:
            print(f"\n[Step 4/4] Skipping downsampling...")
    
    # Identify categorical columns for CatBoost
    cat_feature_indices = []
    if feature_set != 'embedding_only':
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        cat_feature_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
        if verbose:
            print(f"  Categorical features for CatBoost: {len(cat_feature_indices)}")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✅ Data preparation complete! ({elapsed:.1f}s)")
        print(f"   Train: {len(X_train):,} samples ({y_train.sum():,} positives)")
        print(f"   Val: {len(X_val):,} samples ({y_val.sum():,} positives)")
        print(f"   Test: {len(X_test):,} samples ({y_test.sum():,} positives)")
        print(f"{'='*70}\n")
    
    return MedicaidPreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_cols=feature_cols,
        embedding_features=available_embedding,
        tabular_features=available_tabular,
        cat_feature_indices=cat_feature_indices,
        feature_set=feature_set,
        downsampled=downsampled,
        original_train_size=original_train_size,
    )


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model_on_splits(
    model: Any,
    prepared_data: MedicaidPreparedData,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Train model on training set and evaluate on all splits.
    
    Args:
        model: CatBoostClassifier or compatible model
        prepared_data: MedicaidPreparedData from prepare_medicaid_evaluation_data()
        verbose: Print progress
        
    Returns:
        Dict with split names as keys, metrics dict as values
    """
    start_time = time.time()
    
    # Clone model to avoid modifying original
    model = clone(model)
    model_type = type(model).__name__
    
    if verbose:
        print(f"\nTraining {model_type}...")
    
    # Train model
    if model_type == 'CatBoostClassifier':
        train_pool = Pool(
            prepared_data.X_train, 
            prepared_data.y_train,
            cat_features=prepared_data.cat_feature_indices if prepared_data.cat_feature_indices else None
        )
        val_pool = Pool(
            prepared_data.X_val,
            prepared_data.y_val,
            cat_features=prepared_data.cat_feature_indices if prepared_data.cat_feature_indices else None
        )
        model.fit(train_pool, eval_set=val_pool, verbose=0)
    else:
        model.fit(prepared_data.X_train, prepared_data.y_train)
    
    train_time = time.time() - start_time
    if verbose:
        print(f"  Training completed in {train_time:.1f}s")
    
    # Evaluate on all splits
    results = {}
    splits_data = {
        'train': (prepared_data.X_train, prepared_data.y_train),
        'val': (prepared_data.X_val, prepared_data.y_val),
        'test': (prepared_data.X_test, prepared_data.y_test),
    }
    
    for split_name, (X_split, y_split) in splits_data.items():
        if verbose:
            print(f"  Evaluating on {split_name}...")
        
        # Predict probabilities
        if model_type == 'CatBoostClassifier' and prepared_data.cat_feature_indices:
            pool = Pool(X_split, cat_features=prepared_data.cat_feature_indices)
            y_prob = model.predict_proba(pool)[:, 1]
        else:
            y_prob = model.predict_proba(X_split)[:, 1]
        
        # Compute metrics
        results[split_name] = compute_split_metrics(np.array(y_split), y_prob)
    
    # Add training time
    results['_training_time_sec'] = train_time
    
    return results


def evaluate_with_prepared_data(
    prepared_data: MedicaidPreparedData,
    model: Any,
    exp_name: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a model using pre-prepared data.
    
    Args:
        prepared_data: MedicaidPreparedData from prepare_medicaid_evaluation_data()
        model: Pre-configured model (e.g., CatBoostClassifier)
        exp_name: Experiment name for result identification
        verbose: Print progress
        
    Returns:
        Dict with exp_name, model_type, feature_set, and all metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {exp_name}")
        print(f"{'='*70}")
    
    # Evaluate model
    split_results = evaluate_model_on_splits(model, prepared_data, verbose=verbose)
    
    # Build output dictionary
    output = {
        'exp_name': exp_name,
        'model_type': type(model).__name__,
        'feature_set': prepared_data.feature_set,
        'n_features': len(prepared_data.feature_cols),
        'n_embedding_features': len(prepared_data.embedding_features),
        'n_tabular_features': len(prepared_data.tabular_features),
        'downsampled': prepared_data.downsampled,
        'original_train_size': prepared_data.original_train_size,
        'actual_train_size': len(prepared_data.X_train),
        'training_time_sec': split_results.pop('_training_time_sec', 0),
    }
    
    # Flatten split results with prefixes
    for split_name, metrics in split_results.items():
        for metric_name, value in metrics.items():
            output[f'{split_name}_{metric_name}'] = value
    
    if verbose:
        print(f"\n📊 Key Results for {exp_name}:")
        print(f"   Test AUC-ROC: {output.get('test_auc_roc', 0):.4f}")
        print(f"   Test Lift@1%: {output.get('test_lift_1pct', 0):.2f}x")
        print(f"   Test Lift@10%: {output.get('test_lift_10pct', 0):.2f}x")
        print(f"   Test PPV@1%: {output.get('test_ppv_1pct', 0):.2f}%")
        print(f"{'='*70}\n")
    
    return output


# =============================================================================
# EMBEDDING INTEGRATION
# =============================================================================

def load_embeddings_from_npz(
    embedding_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from NPZ file (generated by embedding generation pipeline).
    
    Args:
        embedding_path: Path to NPZ file or directory containing NPZ files
        
    Returns:
        Tuple of (embeddings, individual_ids, index_dts)
    """
    if os.path.isdir(embedding_path):
        npz_files = glob.glob(os.path.join(embedding_path, "embeddings_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files found in {embedding_path}")
        npz_path = sorted(npz_files)[-1]  # Use most recent
    else:
        npz_path = embedding_path
    
    print(f"Loading embeddings from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    return (
        data['embeddings'],
        data['individual_ids'],
        data['index_dts']
    )


def merge_new_embeddings_with_features(
    df_features: pd.DataFrame,
    embeddings: np.ndarray,
    individual_ids: np.ndarray,
    merge_key: str = MEMBER_KEY
) -> pd.DataFrame:
    """
    Replace existing embeddings with new transformer embeddings.
    
    This function is used when you want to evaluate new embeddings
    (e.g., from a new transformer model) against the same Medicaid IP task.
    
    Args:
        df_features: DataFrame with existing features (may include old embeddings)
        embeddings: New embeddings array [num_members, embedding_dim]
        individual_ids: Member IDs corresponding to embeddings
        merge_key: Column name to join on
        
    Returns:
        DataFrame with old embeddings replaced by new embeddings
    """
    # Create embedding DataFrame
    embedding_dim = embeddings.shape[1]
    embedding_cols = [f'emb{i}' for i in range(embedding_dim)]
    
    df_emb = pd.DataFrame(embeddings, columns=embedding_cols)
    df_emb[merge_key] = individual_ids
    
    # Remove old embeddings from features
    old_emb_cols = [c for c in df_features.columns if c.startswith('emb')]
    df_features_no_emb = df_features.drop(columns=old_emb_cols, errors='ignore')
    
    # Merge new embeddings
    df_merged = df_features_no_emb.merge(df_emb, on=merge_key, how='inner')
    
    print(f"Merged new embeddings: {len(df_merged):,} rows (from {len(df_features):,})")
    print(f"  Embedding dim: {embedding_dim}")
    
    return df_merged


# #### Generic pipeline

# In[22]:


# =============================================================================
# HIGH-LEVEL EVALUATION FUNCTIONS
# =============================================================================

def run_medicaid_ip_evaluation(
    embedding_path: Optional[str] = None,
    feature_set: str = 'hybrid',
    sample_frac: Optional[float] = None,
    use_tuned_params: bool = True,
    apply_downsampling: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete Medicaid IP evaluation pipeline.
    
    This is the main entry point for evaluating embeddings on the
    Medicaid IP hospitalization prediction task.
    
    Args:
        embedding_path: Path to new embeddings NPZ file (None = use original)
        feature_set: 'embedding_only', 'tabular_only', or 'hybrid'
        sample_frac: Sample fraction for testing (None = full data)
        use_tuned_params: Use Optuna-tuned hyperparameters
        apply_downsampling: Apply class imbalance handling
        verbose: Print progress
        
    Returns:
        Dict with all evaluation results
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'#'*70}")
        print("MEDICAID IP DOWNSTREAM EVALUATION")
        print(f"{'#'*70}")
        print(f"Feature set: {feature_set}")
        print(f"New embeddings: {embedding_path or 'Using original'}")
        print(f"Sample fraction: {sample_frac or 'Full data'}")
    
    # Step 1: Load data from BigQuery
    df = load_medicaid_data_from_bigquery(
        sample_frac=sample_frac,
        verbose=verbose
    )
    
    # Step 2: Replace embeddings if new path provided
    if embedding_path is not None:
        if verbose:
            print("\nReplacing embeddings with new transformer embeddings...")
        embeddings, individual_ids, index_dts = load_embeddings_from_npz(embedding_path)
        df = merge_new_embeddings_with_features(df, embeddings, individual_ids)
    
    # Step 3: Prepare data
    prepared_data = prepare_medicaid_evaluation_data(
        df=df,
        feature_set=feature_set,
        apply_downsampling=apply_downsampling,
        verbose=verbose
    )
    
    # Step 4: Configure model
    if use_tuned_params:
        model = CatBoostClassifier(**CATBOOST_TUNED_PARAMS)
    else:
        model = CatBoostClassifier(**CATBOOST_BALANCED_PARAMS)
    
    # Step 5: Evaluate
    exp_name = f"medicaid_ip_{feature_set}"
    if embedding_path:
        exp_name += f"_new_emb"
    
    results = evaluate_with_prepared_data(
        prepared_data=prepared_data,
        model=model,
        exp_name=exp_name,
        verbose=verbose
    )
    
    # Add metadata
    results['embedding_path'] = embedding_path
    results['sample_frac'] = sample_frac
    results['use_tuned_params'] = use_tuned_params
    results['total_time_sec'] = time.time() - start_time
    
    return results


# #### Embedding generation

# In[36]:


medicaid_sql = f"""
SELECT *
FROM `{HELDOUT_TE_INPUT_TABLE}`
"""
client = bigquery.Client()
df_te_input = client.query(medicaid_sql).to_dataframe()


# In[39]:


# Get OOT dataset
df_te_input['index_dt'] = pd.to_datetime(df_te_input['index_dt'])
df_pre_oot = df_te_input[df_te_input['index_dt'] <= pd.to_datetime(OOT_CUTOFF_DATE)]
df_oot = df_te_input[df_te_input['index_dt'] > pd.to_datetime(OOT_CUTOFF_DATE)]
print(f"\nTime distribution (OOT cutoff: {OOT_CUTOFF_DATE}):")
print(f"  Pre-OOT (≤{OOT_CUTOFF_DATE}): {len(df_pre_oot):,} members")
print(f"  OOT (>{OOT_CUTOFF_DATE}): {len(df_oot):,} members")


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}    
for exp_name, model_path in tqdm(MODEL_PATHS.items(), desc="Processing models"):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")

    cleanup_gpu_memory(verbose=False)

    # Load model
    model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
        model_path=model_path,
        device=device,
        verbose=verbose
    )

    # Generate embeddings
    inference_start = time.time()
    embeddings, member_keys, index_dts = generate_medicaid_embeddings(
        model=model,
        config=config,
        data=df_te_input,
        device=device,
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        verbose=verbose,
        multi_gpu=multi_gpu,
        moe_config=moe_config,
    )


# In[ ]:




