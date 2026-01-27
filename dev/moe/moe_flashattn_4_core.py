"""
moe_flashattn_4_core.py

Core components for clinical transformer embedding generation and downstream tasks.
Extracted from moe_flashattn_4.py

Version 4 Changes from Version 3:
- OptimizeConfig: Added `enable_gradient_tier_analysis` field for gradient tier analysis
- OptimizeConfig: Changed `pos_weight_max` default from 50 to 35 for better gradient stability

Usage:
    from moe_flashattn_4_core import (
        FlashMoETransformer, FlashAttentionConfig, MoEConfig,
        EmbeddingExtractor, ClinicalDataset, create_collate_fn,
        load_trained_model, DownstreamEvaluator, DownstreamConfig,
    )
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import os
import json
import time
import math
import gc
import warnings
from datetime import datetime
from pathlib import Path
from collections import Counter
from contextlib import nullcontext
from functools import partial
from dataclasses import dataclass, field

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.utils.checkpoint import checkpoint
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
import pandas as pd
import numpy as np
from scipy import stats
import logging

# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.calibration import CalibratedClassifierCV

# Gradient boosting
import xgboost as xgb
import lightgbm as lgb

# Type hints
from typing import Dict, Optional, Tuple, List, Any, Union

warnings.filterwarnings("ignore")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class BaseConfig:
    """
    Base configuration shared across all experiments.
    
    Parameters match your updated specifications:
    - len_dy: 200 days (sequence length)
    - len_cd: 80 codes per day
    - cd_cnt:
        - derive from 
            SELECT COUNT(*) AS cd_cnt
            FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`;
    - target_cd_cnt: 8850 target codes
        - derive from the following target_cd_cnt = total_codes = max(ind) + 1
            -- SELECT 
            --     'w2ind_target (OUTPUT)' AS vocabulary,
            --     COUNT(*) AS total_codes
            -- FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`
    - Multi-label loss (BCEWithLogitsLoss)
    """
    # Data dimensions (from your specifications)
    len_dy: int = 200          # Days in sequence
    len_cd: int = 80           # Codes per day
    cd_cnt: int = 75516        # Input vocabulary size
    target_cd_cnt: int = 6297  # Target vocabulary (updated from 2767, 8850(experiment))

    # Model architecture
    embedding_size: int = 256  # Embedding dimension
    nhid: int = 512           # FFN hidden dimension
    nlayers: int = 6          # Number of temporal encoder layers
    dropout: float = 0.1      # Dropout rate (updated from 0.05)
    
    # Embeddings
    gender_vocab: int = 4     # Gender categories
    age_vocab: int = 1440     # Age in months (120 years)
    lob_vocab: int = 4        # LOB categories (0=padding, 1=Commercial, 2=Medicare, 3=Medicaid)
    
    # Training
    batch_size: int = 32     # Batch size per GPU (change from 16 to 32, 64 is a aggressive and generate oom error 
    learning_rate: float = 2e-4 # 1e-4 is a little conservative
    weight_decay: float = 0.01
    gradient_clip: float = 1.0  # Gradient clipping norm
    
    # Device
    device: str = 'cuda'
    
    # Loss function
    criterion: str = 'BCEWithLogitsLoss'  # Multi-label loss

    
@dataclass
class BaseExpDenseConfig(BaseConfig):
    """
    Base configuration shared across all experiments.
    
    Parameters match your updated specifications:
    - len_dy: 200 days (sequence length)
    - len_cd: 80 codes per day
    - target_cd_cnt: 8850 target codes
    - Multi-label loss (BCEWithLogitsLoss)
    """
    embedding_size: int = 256
    nhid: int = 2048              # ← Standard 4x for GELU (512 * 4)
    
@dataclass
class FlashAttentionConfig(BaseConfig):
    """
    Configuration for Flash Attention experiments.
    
    Key differences from baseline:
    - Uses xFormers memory-efficient attention
    - Requires specific head dimensions
    - Mixed precision training (FP16)
    """
    # Flash Attention specific
    use_flash: bool = True
    use_rope: bool = True      # Rotary position embeddings
    use_swiglu: bool = True    # SwiGLU activation
    use_prenorm: bool = True   # Pre-normalization
    dtype: torch.dtype = torch.float16  # FP16 for T4 GPU
    use_learnt_att_pool: bool = False # Use learned attention pooling instead of transformer for daily code level encoder
    # CHOICE REQUIRED: Head configuration
    nhead: int = 8            # Option A: 8 heads (head_dim=32)
    # nhead: int = 16         # Option B: 16 heads (head_dim=16)
    use_gradient_checkpointing: bool = True  # Enable by default for batch_size >= 32
    checkpoint_every_n_layers: int = 2  # Checkpoint every 2 layers (balance speed/memory)    
    
    

@dataclass
class MoEConfig:
    """
    Configuration for Mixture of Experts layer.
    
    Supports all 5 experimental configurations:
    1. Dense (no MoE)
    2. Standard MoE (8 experts, top-2)
    3. Shared Expert MoE (1 shared + 7 routed)
    4. Fine-grained MoE (1 shared + 15 routed, smaller)
    5. Auxiliary-free MoE (DeepSeek balancing)
    """
    # Model dimensions
    d_model: int = 256
    d_ff: int = 512

    # Expert configuration
    num_experts: int = 8
    num_shared_experts: int = 0
    top_k: int = 2
    expert_dropout: float = 0.1 # increase the expert dropout rate 
    
    # Load balancing
    load_balance_strategy: str = 'switch'  # 'switch' or 'deepseek'
    aux_loss_weight: float = 0.01
    bias_lr: float = 5e-4 # increase the bais_lr to keep up with router learning
    bias_momentum: float = 0.9
    
    # Optional
    z_loss_weight: float = 0.0
    use_moe_from_layer: int = 2  # Start MoE from layer 2  by default
    use_swiglu_experts: bool = False
    router_warmup_steps: int = 500 # First 500 batches use balanced routing for warmup

@dataclass
class OptimizeConfig:
    """
    - Higher learning rate (2e-4 vs 1e-4)
    - OneCycleLR scheduler (default)
    - BCE with pos_weight for rare code handling
    """
    # ============================================================
    # SCHEDULER
    # options include linear, cosine and onecycle; choose onecycle first
    # ============================================================
    scheduler_type: str = 'onecycle'  # 'onecycle' | 'linear' | 'cosine'
    warmup_pct: float = 0.15          # Warmup as fraction of total steps, for the first 15% of total steps, LR ramps linearly from ~0 to peak.
    min_lr_ratio: float = 0.01        # End LR = peak * min_lr_ratio, the LR decays down to 20%
    
    # OneCycle specific
    onecycle_pct_start: float = 0.30  # Fraction of training to ramp up
    onecycle_div_factor: float = 25   # start_lr = max_lr / div_factor
    onecycle_final_div: float = 1000  # end_lr = max_lr / final_div
    
    # Linear specific
    plateau_pct: float = 0.30         # Stay at peak for this fraction after warmup, after warmup, LR stays at peak for 35% of total steps.
    
    # ============================================================
    # LOSS FUNCTION
    # ============================================================
    use_pos_weight: bool = True       # Enable frequency-based BCE weighting
    pos_weight_max: float = 35         # Cap weight to avoid instability, too large value like 50 can increase gradients largely and unstablize the MOE router
    pos_weight_method: str = 'log_scaled'     # Options: 'inverse', 'log_scaled', 'ens', 'tiered'
    
    # Tiered weighting configuration (when pos_weight_method='tiered')
    tier_weights: dict = None  # Will use default if None
    enable_gradient_tier_analysis: bool = False  # Enable per-tier gradient analysis during training
    
    # Effective Number of Samples (when pos_weight_method='ens') optional, need tune on beta
    ens_beta: float = 0.9999              # Higher = more aggressive reweighting
    
    # ============================================================
    # FOCAL LOSS; consider add weights to hard classified cases (too much maybe, not compatible with MOE) 
    # ============================================================
    use_focal_loss: bool = False          # Set True to enable focal loss
    focal_gamma: float = 2.0              # Focusing parameter (0=BCE, 2=standard, 3=aggressive)
    focal_alpha: float = 0.25             # Balance factor for positive class   
    
    # ============================================================
    # OPTIMIZER CONFIGURATION
    # Options: 'adamw' (default) | 'sgd'
    # ============================================================
    optimizer_type: str = 'adamw'     # 'adamw' | 'sgd'
    
    # SGD-specific parameters (only used when optimizer_type='sgd')
    sgd_momentum: float = 0.9         # Momentum for SGD (legacy uses 0.9)
    sgd_nesterov: bool = False        # Use Nesterov momentum
    
    # Override defaults (optional - None means use BaseConfig defaults)
    override_lr: Optional[float] = None           # Override learning rate
    override_weight_decay: Optional[float] = None # Override weight_decay (0 for legacy SGD)
    override_gradient_clip: Optional[float] = None # Override gradient clip (0.25 for legacy)    


@dataclass
class DownstreamConfig:
    """Configuration for downstream task evaluation."""
    task_name: str = "medicaid_ip_risk"
    test_size: float = 0.1        # 10% for test
    val_size: float = 0.1         # 10% of remaining for validation (= 8% of total)
    random_state: int = 42
    percentiles: List[float] = field(default_factory=lambda: [0.1, 0.01])
    n_cv_folds: int = 5           # optional, this takes long time; cross-validation folds for probe
    max_iter: int = 1000          # Max iterations for logistic regression
    class_weight: str = 'balanced'  # Handle class imbalance
    model_type: str = 'xgboost'  # 'logistic', 'xgboost', 'lightgbm'
    calibrate_proba: bool = True  # Whether to calibrate probabilities
    lob_name: Optional[str] = None  # e.g., 'commercial', 'medicare', 'medicaid'
    outcome_column: str = 'acute_ip_flag'  # Column name for outcome in outcomes_df

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_experiment_configs() -> Dict[str, Tuple[Optional[MoEConfig], bool]]:
    """
    Define all experiment configurations.
    
    Returns:
        Dict mapping experiment_name to (moe_config, use_learnt_att_pool)
        
    Each experiment is a tuple:
        - moe_config: MoEConfig or None
        - use_learnt_att_pool: bool (True for learned pooling, False for transformer)
    """
    configs = {}
    
    # ============================================================
    # BASELINE EXPERIMENTS (No MoE)
    # ============================================================
    
    # Exp 1: Pure baseline for experimentation (standard everything)
    configs['exp1_dense_baseline'] = (
        None,   # No MoE
        False   # Standard transformer (not Flash, so pooling flag ignored)
    )
    
    # Exp 2: Flash Attention with standard pooling
    configs['exp2_dense_flash'] = (
        None,   # No MoE
        False   # Flash Attention + Max-Pool (baseline for pooling comparison)
    )
    
    # Exp 2b: Flash Attention with learned pooling
    configs['exp2b_flash_learned_pool'] = (
        None,   # No MoE
        True    # Learned Attention Pooling (test pooling improvement)
    )
    
    # ============================================================
    # MOE EXPERIMENTS (All use Flash Attention)
    # ============================================================
    
    # Exp 3: Standard MoE with standard pooling
    configs['exp3_standard_moe'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            load_balance_strategy='switch',
            aux_loss_weight=0.01,
            expert_dropout=0.05,
            use_moe_from_layer=2,
            use_swiglu_experts = False
        ),
        False  # Flash Attention + Max-Pool (baseline)
    )
    configs['exp3a_moe_swiglu'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            load_balance_strategy='switch',
            aux_loss_weight=0.01,
            expert_dropout=0.05,
            use_moe_from_layer=2,
            use_swiglu_experts = True
        ),   
        False  # Flash Attention + Max-Pool (Baseline GELU vs. Swiglu)
    )
    # Exp 3b: Standard MoE with learned pooling and Swiglu
    configs['exp3b_moe_swiglu_learned_pool'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            load_balance_strategy='switch',
            aux_loss_weight=0.01,
            expert_dropout=0.05,
            use_moe_from_layer=2,
            use_swiglu_experts = True
        ),
        True  # Learned Attention Pooling (test with standard MoE)
    )
    # Exp 3c: Standard MoE with learned pooling and Swiglu but set MOE from the 4th layer, not the 2nd
    # Reason being that the first 0-3 layers may still learning new stuff and the last two layers are 
    # Starting to pick up specialized knowledge
    configs['exp3c_moe_swiglu_learned_pool_layer4'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            load_balance_strategy='switch',
            aux_loss_weight=0.01,
            expert_dropout=0.05,
            use_moe_from_layer=4,
            use_swiglu_experts = True
        ),
        True  # Learned Attention Pooling (test with standard MoE)
    )    
    # Exp 3d: lower the aux_loss to 0.001 not 0.01 to reduce the impact of the auxilary loss 
    configs['exp3d_moe_swiglu_learned_pool_layer4_aux001'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            load_balance_strategy='switch',
            aux_loss_weight=0.001,
            expert_dropout=0.05,
            use_moe_from_layer=2,
            use_swiglu_experts = True
        ),
        True  # Learned Attention Pooling (test with standard MoE)
    )  
    # Exp 3e: lower the aux_loss to 0.001 not 0.01 to reduce the impact of the auxilary loss 
    configs['exp3e_moe_swiglu_learned_pool_layer2_aux001'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            load_balance_strategy='switch',
            aux_loss_weight=0.001,
            expert_dropout=0.05,
            use_moe_from_layer=2,
            use_swiglu_experts = True
        ),
        True  # Learned Attention Pooling (test with standard MoE)
    )  
    
    # Exp 4: Shared Expert MoE (with learned pooling)
    configs['exp4_shared_expert'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=1,
            top_k=2,
            load_balance_strategy='switch',
            aux_loss_weight=0.001, # Change to 0.001 based on the experiment 3d
            expert_dropout=0.05,
            use_moe_from_layer=2,
            use_swiglu_experts = True
        ),
        True  # Use learned pooling (recommended for MoE)
    )
    
    # Exp 5: Fine-grained MoE (with learned pooling)
    configs['exp5_fine_grained'] = (
        MoEConfig(
            d_model=256,
            d_ff=238,  # Smaller experts
            num_experts=16,
            num_shared_experts=1,
            top_k=5,
            load_balance_strategy='switch',
            aux_loss_weight=0.001,
            expert_dropout=0.05,
            use_moe_from_layer=2,
            use_swiglu_experts = True
        ),
        True  # Use learned pooling
    )
    
    # Exp 6: Auxiliary-free MoE with DeepSeek balancing (with learned pooling)
    configs['exp6_auxiliary_free'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=1,
            top_k=2,
            load_balance_strategy='deepseek',  # DeepSeek bias correction
            aux_loss_weight=0.001,  # add a little auxiliary loss
            bias_lr=3e-3,          # 100× increase from 1e-5
            bias_momentum=0.6,     # Lower momentum for faster adaptation
            expert_dropout=0,    # Increase from 0.05
            use_moe_from_layer=2,
            use_swiglu_experts = True,
            router_warmup_steps = 0, # v2 add a router warmup; v3 remove it 
            z_loss_weight=0.005 # add z-loss to 0.01
        ),
        True  # Use learned pooling
    )
    # Exp 6a: Auxiliary-free MoE with DeepSeek balancing (with learned pooling)
    configs['exp6a_auxiliary_free_layer4'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=1,
            top_k=2,
            load_balance_strategy='deepseek',  # DeepSeek bias correction
            aux_loss_weight=0.0,  # No auxiliary loss
            bias_lr=3e-3,          # 100× increase from 1e-5
            bias_momentum=0.7,     # Lower momentum for faster adaptation
            expert_dropout=0.1,    # Increase from 0.05
            use_moe_from_layer=4,
            use_swiglu_experts = True
        ),
        True  # Use learned pooling
    )
    configs['exp6b_auxiliary_free_no-share-exp'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            load_balance_strategy='deepseek',  # DeepSeek bias correction
            aux_loss_weight=0.0,  # No auxiliary loss
            bias_lr=3e-3,          # 100× increase from 1e-5
            bias_momentum=0.6,     # Lower momentum for faster adaptation
            expert_dropout=0.1,    # Increase from 0.05
            use_moe_from_layer=2,
            use_swiglu_experts = True,
            router_warmup_steps = 500,
            z_loss_weight=0.001
        ),
        True  # Use learned pooling
    )    
    configs['exp6c_auxiliary_free_fine-grained16'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=16,
            num_shared_experts=1,
            top_k=2,
            load_balance_strategy='deepseek',  # DeepSeek bias correction
            aux_loss_weight=0.0,  # No auxiliary loss
            bias_lr=3e-3,          # 100× increase from 1e-5
            bias_momentum=0.6,     # Lower momentum for faster adaptation
            expert_dropout=0.1,    # Increase from 0.05
            use_moe_from_layer=2,
            use_swiglu_experts = True,
            z_loss_weight=0.001
        ),
        True  # Use learned pooling
    ) 
    configs['exp6d_auxiliary_free_fine-grained16_shared2'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=16,
            num_shared_experts=2,
            top_k=2,
            load_balance_strategy='deepseek',  # DeepSeek bias correction
            aux_loss_weight=0.0,  # No auxiliary loss
            bias_lr=5e-3,          # 100× increase from 1e-5
            bias_momentum=0.6,     # Lower momentum for faster adaptation
            expert_dropout=0.1,    # Increase from 0.05
            use_moe_from_layer=2,
            use_swiglu_experts = True
        ),
        True  # Use learned pooling
    )    
    configs['exp6e_auxiliary_free_fine-grained32-shared2'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=32,
            num_shared_experts=2,
            top_k=2,
            load_balance_strategy='deepseek',  # DeepSeek bias correction
            aux_loss_weight=0.0,  # No auxiliary loss
            bias_lr=5e-3,          # 100× increase from 1e-5
            bias_momentum=0.7,     # Lower momentum for faster adaptation
            expert_dropout=0.1,    # Increase from 0.05
            use_moe_from_layer=2,
            use_swiglu_experts = True
        ),
        True  # Use learned pooling
    ) 
    return configs


def cleanup_gpu_memory(verbose = True):
    """
    Comprehensive GPU memory cleanup before training.
    
    Clears:
    - Python garbage collector
    - PyTorch CUDA cache
    - All cached allocations
    - Resets memory statistics
    """
    # Collect Python garbage first
    gc.collect()
    
    if not torch.cuda.is_available():
        if verbose:
            print("No CUDA devices available")
        return
    
    num_gpus = torch.cuda.device_count()
    
    # Synchronize ALL GPUs before cleanup
    for device_id in range(num_gpus):
        torch.cuda.synchronize(device_id)
    
    # Clear CUDA cache (global operation, but call after sync)
    torch.cuda.empty_cache()
    
    # Reset memory statistics for ALL GPUs
    for device_id in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.reset_accumulated_memory_stats(device_id)
    
    # Print memory status for each GPU
    if verbose:
        print(f"\n{'='*80}")
        print(f"GPU MEMORY STATUS ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")
        print(f"{'='*80}")
        
        for device_id in range(num_gpus):
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
            
            print(f"GPU {device_id}:")
            print(f"  Allocated:     {allocated:.2f} GB")
            print(f"  Reserved:      {reserved:.2f} GB")
            print(f"  Peak Allocated: {max_allocated:.2f} GB")
            
            # Calculate fragmentation
            if reserved > 0:
                fragmentation = (reserved - allocated) / reserved * 100
                print(f"  Fragmentation: {fragmentation:.1f}%")
            print()
        
        print(f"{'='*80}\n")

def cleanup_gpu_memory_hard(device_ids: list = None):
    """
    AGGRESSIVE cleanup for stubborn memory leaks.
    
    Use when normal cleanup doesn't free enough memory.
    
    Args:
        device_ids: List of GPU IDs to clean. If None, clean all GPUs.
    """
    # Multiple garbage collection passes
    for _ in range(3):
        gc.collect()
    
    if not torch.cuda.is_available():
        return
    
    num_gpus = torch.cuda.device_count()
    if device_ids is None:
        device_ids = list(range(num_gpus))
    
    # Synchronize all operations
    for device_id in device_ids:
        with torch.cuda.device(device_id):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Clean up IPC handles
    
    # Final global cleanup
    torch.cuda.empty_cache()
    
    # Multiple GC passes again
    for _ in range(2):
        gc.collect()
    
    print(f"✅ Aggressive cleanup completed for GPUs: {device_ids}")      

# ============================================================================
# DATASET AND DATA LOADING
# ============================================================================

def conv_cd(ipt: str, len_dy: int, len_cd: int) -> List[List[int]]:
    """
    Convert code string to 2D list.
    
    Format: "code1,code2*code3,code4*..." 
    where * separates days, , separates codes within day
    
    Corrected from original to handle edge cases better.
    """
    if not ipt or pd.isna(ipt):
        return [[0] * len_cd for _ in range(len_dy)]
    
    days = ipt.split('*')
    days = days[:len_dy]  # Truncate to max days
    
    result = []
    for day_str in days:
        if not day_str:
            day_codes = [0] * len_cd
        else:
            codes = day_str.split(',')
            day_codes = []
            for code in codes[:len_cd]:  # Truncate to max codes
                try:
                    day_codes.append(int(code) if code else 0)
                except ValueError:
                    day_codes.append(0)
            # Pad to len_cd
            day_codes.extend([0] * (len_cd - len(day_codes)))
        result.append(day_codes)
    
    # Pad to len_dy days
    while len(result) < len_dy:
        result.append([0] * len_cd)
    
    return result

def conv_age_gender(ipt: str, len_dy: int, max_val: int = 1439) -> List[int]:
    """
    Convert age/gender string to list.
    
    Format: "value1*value2*..."
    Clips age to max_val (1439 months = 120 years)
    """
    if not ipt or pd.isna(ipt):
        return [0] * len_dy
    
    values = ipt.split('*')
    values = values[:len_dy]
    
    result = []
    for val in values:
        try:
            result.append(min(int(val), max_val) if val else 0)
        except ValueError:
            result.append(0)
    
    # Pad or forward-fill
    if result:
        last_val = result[-1]
        while len(result) < len_dy:
            result.append(last_val)
    else:
        result = [0] * len_dy
    
    return result

def conv_lob(ipt: str, len_dy: int) -> List[int]:
    """
    Convert LOB (Line of Business) string to list.
    
    Format: "value1*value2*..." (same format as age/gender, but with string values)
    Maps: Commercial=1, Medicare=2, Medicaid=3, padding/unknown=0
    
    Args:
        ipt: LOB string from data (e.g., "Medicaid*Medicaid*..." or single value "Medicaid")
        len_dy: Target sequence length (200)
    
    Returns:
        List of LOB indices [len_dy]
    """
    # LOB mapping (case-insensitive)
    lob_map = {
        'commercial': 1,
        'medicare': 2,
        'medicaid': 3
    }
    
    if not ipt or pd.isna(ipt):
        return [0] * len_dy
    
    # Handle both single value and asterisk-separated formats
    if '*' in str(ipt):
        values = str(ipt).split('*')
    else:
        # Single value - repeat for all days
        values = [str(ipt)] * len_dy
    
    values = values[:len_dy]  # Truncate to max days
    
    result = []
    for val in values:
        val_clean = val.strip().lower() if val else ''
        if val_clean in lob_map:
            result.append(lob_map[val_clean])
        else:
            result.append(0)  # Unknown LOB
    
    # Forward-fill with last valid value (LOB typically doesn't change within sequence)
    if result:
        last_val = result[-1] if result[-1] != 0 else 3  # Default to Medicaid if all zeros
        while len(result) < len_dy:
            result.append(last_val)
    else:
        result = [3] * len_dy  # Default to Medicaid (since this is Medicaid data)
    
    return result

def conv_target(target: str, len_dy: int, target_cd_cnt: int) -> List[List[int]]:
    """
    Convert target string to nested list (multi-label).
    
    CRITICAL: Each day can have multiple target codes!
    Format: "code1,code2*code3*code4,code5,code6*..."
    
    Returns: List[List[int]] where each inner list contains all codes for that day
    """
    if not target or pd.isna(target):
        return [[0] for _ in range(len_dy)]
    
    days = target.split('*')
    days = days[:len_dy]
    
    result = []
    for day_str in days:
        if not day_str:
            day_codes = [0]
        else:
            codes = day_str.split(',')
            day_codes = []
            for code in codes:
                try:
                    code_val = int(code) if code else 0
                    # Validate code is in vocabulary
                    if 0 < code_val <= target_cd_cnt:
                        # Convert to 0-based index before appending
                        day_codes.append(code_val-1)
                    elif code_val > target_cd_cnt:
                        print(f"Warning: Target code {code_val} >= vocab size {target_cd_cnt}")
                except ValueError:
                    pass
            if not day_codes:
                day_codes = [0]
        result.append(day_codes)
    
    # Pad to len_dy
    while len(result) < len_dy:
        result.append([0])
    
    return result


class ClinicalDataset(Dataset):
    """
    PyTorch Dataset for clinical transformer.
    Pre-processes all string parsing once during initialization for high performance.
    """
    def __init__(self, df: pd.DataFrame, config: BaseConfig):
        self.config = config
        print(f"Pre-processing {len(df)} samples into tensors (one-time cost)...")
        
        # Extract series for faster processing
        age_strs = df['age_in_months'].tolist()
        gender_strs = df['gender_cd'].tolist()
        cd_strs = df['cd'].tolist()
        target_strs = df['target'].tolist()
        self.dt_cnt = df['dt_cnt'].tolist()
        lob_strs = df['lob'].tolist()
        
        # Pre-allocate tensors
        self.ages = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.genders = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.codes = torch.zeros(len(df), config.len_dy, config.len_cd, dtype=torch.long)
        self.lobs = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.targets = []

        # Process all samples
        for i in range(len(df)):
            if i > 0 and i % 50000 == 0:
                print(f"  Processed {i}/{len(df)} samples...")

            age_list = conv_age_gender(age_strs[i], config.len_dy)
            gender_list = conv_age_gender(gender_strs[i], config.len_dy, max_val=3)
            cd_list = conv_cd(cd_strs[i], config.len_dy, config.len_cd)
            lob_list = conv_lob(lob_strs[i], config.len_dy) 
            target_list = conv_target(target_strs[i], config.len_dy, config.target_cd_cnt)

            self.ages[i] = torch.tensor(age_list, dtype=torch.long)
            self.genders[i] = torch.tensor(gender_list, dtype=torch.long)
            self.codes[i] = torch.tensor(cd_list, dtype=torch.long)
            self.lobs[i] = torch.tensor(lob_list, dtype=torch.long)
            self.targets.append(target_list)
        
        print("Pre-processing complete.")

    def __len__(self):
        return len(self.dt_cnt)

    def __getitem__(self, idx):
        return {
            'age': self.ages[idx],
            'gender': self.genders[idx],
            'lob': self.lobs[idx], 
            'codes': self.codes[idx],
            'dt_cnt': self.dt_cnt[idx],
            'target': self.targets[idx]
        }

@dataclass
class PreparedData:
    """
    Container for pre-computed expensive data artifacts.
    Reuse across multiple experiment runs to avoid redundant computation.
    """
    train_dataset: ClinicalDataset
    val_dataset: ClinicalDataset
    code_frequencies: np.ndarray
    config: BaseConfig  # The config used for dataset creation
    
    def __repr__(self):
        return (f"PreparedData(train_samples={len(self.train_dataset)}, "
                f"val_samples={len(self.val_dataset)}, "
                f"code_frequencies_shape={self.code_frequencies.shape})")


def clinical_collate_fn(batch: List[Dict], config: 'BaseConfig') -> Dict[str, Any]:
    """
    Custom collate function for clinical data.
    
    Handles the special case of 'target' which is a nested list with variable-length sublists.
    PyTorch's default_collate cannot handle this, so keep it as a Python list.
    v2: Enhanced collate function that pre-computes multi-hot targets as tensors.
    
    Args:
        batch: List of dictionaries from ClinicalDataset.__getitem__
    
    Returns:
        Batched dictionary with:
        - age, gender, codes: Stacked tensors
        - dt_cnt: List of integers
        - target: List of nested lists (NOT converted to tensor)
    """
    batch_size = len(batch)
    len_dy = config.len_dy
    target_cd_cnt = config.target_cd_cnt
    
    # Extract each field
    ages = torch.stack([item['age'] for item in batch])
    genders = torch.stack([item['gender'] for item in batch])
    lobs = torch.stack([item['lob'] for item in batch])
    codes = torch.stack([item['codes'] for item in batch])
    dt_cnts = torch.tensor([item['dt_cnt'] for item in batch], dtype=torch.long)
    # Pre-compute multi-hot targets: [batch, len_dy, target_cd_cnt]
    targets_multihot = torch.zeros(batch_size, len_dy, target_cd_cnt, dtype=torch.float16)
    
    for i, item in enumerate(batch):
        target_list = item['target']  # List[List[int]] - len_dy x variable
        for day_idx, day_codes in enumerate(target_list):
            if day_idx < len_dy and day_codes:  # Check bounds and non-empty
                for code_idx in day_codes:
                    if 0 <= code_idx < target_cd_cnt:
                        targets_multihot[i, day_idx, code_idx] = 1.0
    
    # Keep original targets for metrics computation (backward compat)
    targets_list = [item['target'] for item in batch]
    
    return {
        'age': ages,
        'gender': genders,
        'lob': lobs,
        'codes': codes,
        'dt_cnt': dt_cnts,
        'target_multihot': targets_multihot,  # [batch, len_dy, target_cd_cnt]
        'target': targets_list          # List[List[List[int]]] - kept for metrics
    }

def create_collate_fn(config: 'BaseConfig'):
    #     Factory to create collate function with config bound.

    #     Usage:
    #         collate_fn = create_collate_fn(config)
    #         DataLoader(..., collate_fn=collate_fn)
    return partial(clinical_collate_fn, config=config)

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better temporal modeling.
    
    Why use RoPE?
    1. Captures relative positions naturally (important for medical sequences)
    2. Extrapolates to longer sequences than training
    3. No additional parameters to learn
    4. Used in modern LLMs (LLaMA, Mistral)
    
    Mathematical formulation:
    - For position m and dimension i: θ_i = base^(-2i/d)
    - Rotation in complex plane preserves relative distances
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            dim: Dimension per attention head (head_dim)
            max_seq_len: Maximum sequence length (200 for your data)
            base: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies for efficiency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache rotations for all positions
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        """Precompute cos/sin values for all positions."""
        # Position indices
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        
        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Duplicate for complex representation
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Cache cos and sin
        cos_cached = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        sin_cached = emb.sin()[None, None, :, :]
        
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q, k: [batch, nhead, seq_len, head_dim]
        
        Returns:
            q_rot, k_rot: Rotated queries and keys
        """
        seq_len = q.shape[2]
        
        # Get cached values
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Apply rotation
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot



class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit activation.
    
    Why SwiGLU over GELU?
    1. Empirically better performance in transformers
    2. Gating allows selective information flow
    3. Used in state-of-the-art models (LLaMA, PaLM)
    
    Note: To maintain parameter count equivalence with standard FFN,
    we adjust hidden dimension: d_ff_adjusted = (2/3) * d_ff
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        # Adjust dimension for parameter equivalence
        # Standard FFN: 2 * (d_model * d_ff) parameters
        # SwiGLU: 3 * (d_model * d_ff_adjusted) parameters
        d_ff_adjusted = int((2 * d_ff) / 3)
        
        self.w_gate = nn.Linear(d_model, d_ff_adjusted, bias=False)
        self.w_up = nn.Linear(d_model, d_ff_adjusted, bias=False)
        self.w_down = nn.Linear(d_ff_adjusted, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: gate * up_projection."""
        gate = F.silu(self.w_gate(x))  # Swish activation
        up = self.w_up(x)
        hidden = gate * up  # Gating
        output = self.w_down(hidden)
        output = self.dropout(output)
        return output


class FlashAttentionLayer(nn.Module):
    """
    Multi-head attention with Flash Attention (xFormers) support.
    
    This layer can be used in both standard and MoE transformers.
    
    Key features:
    1. xFormers memory-efficient attention for long sequences
    2. Rotary position embeddings for temporal modeling
    3. Pre-normalization for training stability
    4. Mixed precision (FP16) support
    
    Architecture choice explanations:
    - Pre-norm: Better gradient flow in deep networks
    - No bias in projections: Following modern practices
    - Separate Q,K,V projections: More flexibility than single projection
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_flash: bool = True,
        max_seq_len: int = 200,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize Flash Attention layer.
        
        Args:
            d_model: Model dimension (256)
            nhead: Number of heads (8 for head_dim=32, or 16 for head_dim=16)
            dropout: Dropout rate
            use_rope: Whether to use rotary embeddings
            use_flash: Whether to use Flash Attention
            max_seq_len: Maximum sequence length (200 days)
            dtype: Data type for mixed precision
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_rope = use_rope
        self.use_flash = use_flash
        self.dtype = dtype
        
        # Validate head dimension for Flash Attention
        if use_flash and self.head_dim not in [32, 64, 128]:
            print(f"⚠️ Warning: head_dim={self.head_dim} not optimal for xFormers.")
            print(f"   Recommended: 32, 64, or 128. Current: {self.head_dim}")
        
        # Query, Key, Value projections (no bias following modern practices)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary position embeddings
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=self.head_dim,
                max_seq_len=max_seq_len,
                base=10000.0
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Check xFormers availability
        self.xformers_available = False
        if use_flash:
            try:
                from xformers.ops import memory_efficient_attention
                self.xformers_attention = memory_efficient_attention
                self.xformers_available = True
            except ImportError:
                pass
        
        self._init_weights()
            
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Small initialization for attention weights
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2))
        
        # Zero initialization for output (residual starts at identity)
        nn.init.zeros_(self.out_proj.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with Flash Attention.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model] or [batch, seq, d_model]
            mask: Attention mask (optional)
            is_causal: Whether to apply causal masking (for temporal encoder)
        
        Returns:
            Output tensor same shape as input
        """
        # Handle both PyTorch formats
        if x.dim() == 3 and x.shape[0] > x.shape[1]:
            # Assume [seq_len, batch, d_model] format
            seq_first = True
            seq_len, batch_size, d_model = x.shape
        else:
            # Assume [batch, seq_len, d_model] format
            seq_first = False
            batch_size, seq_len, d_model = x.shape
            x = x.transpose(0, 1)  # Convert to seq_first
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [seq_len, batch, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # [seq_len, batch, d_model] -> [batch, nhead, seq_len, head_dim]
        q = q.view(seq_len, batch_size, self.nhead, self.head_dim)
        q = q.permute(1, 2, 0, 3)
        
        k = k.view(seq_len, batch_size, self.nhead, self.head_dim)
        k = k.permute(1, 2, 0, 3)
        
        v = v.view(seq_len, batch_size, self.nhead, self.head_dim)
        v = v.permute(1, 2, 0, 3)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k)
        
        # Apply attention
        if self.use_flash and self.xformers_available:
            attn_output = self._xformers_attention(q, k, v, is_causal)
        else:
            attn_output = self._standard_attention(q, k, v, mask, is_causal)
        
        # Reshape back
        # [batch, nhead, seq_len, head_dim] -> [seq_len, batch, d_model]
        attn_output = attn_output.permute(2, 0, 1, 3)
        attn_output = attn_output.contiguous().view(seq_len, batch_size, d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Convert back to original format if needed
        if not seq_first:
            output = output.transpose(0, 1)
        
        return output
    
    def _xformers_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        is_causal: bool
    ) -> torch.Tensor:
        """
        Apply xFormers memory-efficient attention.
        
        This is the key optimization that enables:
        1. Linear memory complexity (vs quadratic)
        2. 2-3x speedup on long sequences
        3. Larger batch sizes
        """
        # Convert to xFormers format and dtype
        q = q.to(dtype=self.dtype)
        k = k.to(dtype=self.dtype)
        v = v.to(dtype=self.dtype)
        
        # xFormers expects [batch, seq_len, nhead, head_dim]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Create causal mask if needed
        attn_bias = None
        if is_causal:
            from xformers.ops import LowerTriangularMask
            attn_bias = LowerTriangularMask()
        
        # Apply memory-efficient attention
        output = self.xformers_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=self.dropout.p if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim)
        )
        
        # Convert back to [batch, nhead, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous()
        
        return output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """Fallback to standard attention if Flash not available."""
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if is_causal:
            # Create causal mask
            seq_len = q.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device),
                diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask, float('-inf'))
        
        if mask is not None:
            scores += mask
        
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout.p, training=self.training)
        output = torch.matmul(attn, v)
        
        return output


class LearnedAttentionPooling(nn.Module):
    """
    Learned attention pooling for daily code aggregation.
    
    Instead of:
        codes → Transformer(80 tokens) → Max-Pool → vector
    
    We do:
        codes → Attention(learned query) → vector
    
    Benefits:
    - 3-5× faster than full transformer
    - Learnable soft aggregation (better than hard max)
    - No position encoding needed (codes are unordered)
    - Same memory footprint as mean/max pooling
    
    Architecture:
        Query: Learnable [1, d_model] vector
        Keys/Values: Code embeddings [N, d_model]
        Output: Attention-weighted sum [d_model]
    """
    
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        
        # Learnable query (what to look for in codes)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Attention projections
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Optional: Multi-head version (use 2-4 heads)
        # For simplicity, we use single-head here
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence using learned attention.
        
        Args:
            x: [seq_len, batch, d_model] - code embeddings
        
        Returns:
            pooled: [batch, d_model] - aggregated representation
        """
        seq_len, batch_size, d_model = x.shape
        
        # Expand query for batch
        q = self.query.expand(-1, batch_size, -1)  # [1, batch, d_model]
        
        # Project keys and values
        k = self.k_proj(x)  # [seq_len, batch, d_model]
        v = self.v_proj(x)  # [seq_len, batch, d_model]
        
        # Compute attention scores
        # q: [1, batch, d_model]
        # k: [seq_len, batch, d_model]
        # scores: [batch, 1, seq_len]
        q = q.transpose(0, 1)  # [batch, 1, d_model]
        k = k.permute(1, 2, 0)  # [batch, d_model, seq_len]
        
        scores = torch.bmm(q, k) / math.sqrt(d_model)  # [batch, 1, seq_len]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        v = v.permute(1, 0, 2)  # [batch, seq_len, d_model]
        pooled = torch.bmm(attn_weights, v)  # [batch, 1, d_model]
        pooled = pooled.squeeze(1)  # [batch, d_model]
        
        return pooled


class SwitchAuxiliaryLoss(nn.Module):
    """
    Switch Transformer load balancing loss.
    
    Encourages uniform expert usage by minimizing:
    L_aux = N × Σ_i (importance_i × load_i)
    
    Where:
    - importance_i = average router probability for expert i
    - load_i = fraction of tokens actually routed to expert i
    """
    
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
    
    def forward(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss.
        
        Args:
            router_probs: [num_tokens, num_experts] softmax probabilities
            expert_indices: [num_tokens, top_k] selected expert indices
        
        Returns:
            Scalar loss value
        """
        # Importance: mean probability per expert
        importance = router_probs.mean(dim=0)  # [num_experts]
        
        # Load: actual usage per expert
        batch_size = expert_indices.shape[0]
        top_k = expert_indices.shape[1]
        load = torch.zeros(self.num_experts, device=expert_indices.device)
        
        for k in range(top_k):
            load.scatter_add_(
                0,
                expert_indices[:, k],
                torch.ones(batch_size, device=expert_indices.device)
            )
        
        load = load / (batch_size * top_k)
        
        # Switch loss encourages importance ≈ load ≈ 1/N
        aux_loss = self.num_experts * torch.sum(importance * load)
        
        return aux_loss

class DeepSeekBiasCorrection(nn.Module):
    """
    DeepSeek auxiliary-free load balancing.
    
    Instead of adding loss term, directly adjust router bias:
    bias_i -= α × (load_i - 1/N)
    
    This avoids gradient conflicts between task loss and balancing loss.
    """
    
    def __init__(self, num_experts: int, bias_lr: float = 1e-5, momentum: float = 0.9):
        super().__init__()
        self.num_experts = num_experts
        self.bias_lr = bias_lr
        self.momentum = momentum
        
        # Learnable bias (updated outside gradient computation)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        
        # Exponential moving average of loads
        self.register_buffer('expert_load_ema', torch.ones(num_experts) / num_experts)
    
    def get_bias(self) -> torch.Tensor:
        """Get current bias to add to router logits."""
        return self.expert_bias
    
    def update_bias(self, expert_indices: torch.Tensor) -> None:
        """
        Update bias based on current batch usage.
        
        Called AFTER backward pass to avoid interfering with gradients.
        """
        with torch.no_grad():
            batch_size = expert_indices.shape[0]
            top_k = expert_indices.shape[1]
            
            # Compute current load
            current_load = torch.zeros(self.num_experts, device=expert_indices.device)
            for k in range(top_k):
                current_load.scatter_add_(
                    0,
                    expert_indices[:, k],
                    torch.ones(batch_size, device=expert_indices.device)
                )
            current_load = current_load / (batch_size * top_k)
            
            # Update EMA
            self.expert_load_ema = (
                self.momentum * self.expert_load_ema + 
                (1 - self.momentum) * current_load
            )
            
            # Update bias: reduce probability for overused experts
            target_load = 1.0 / self.num_experts
            bias_gradient = self.expert_load_ema - target_load
            self.expert_bias -= self.bias_lr * bias_gradient

class ExpertLayer(nn.Module):
    """
    Single expert: 2-layer FFN.
    
    Can use either standard GELU or SwiGLU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.05,
        use_swiglu: bool = False # Stability, efficiency, fair comparison
    ):
        super().__init__()
        
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, dropout)
        else:
            # Standard 2-layer FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Supports:
    1. Variable number of experts
    2. Shared experts (always active)
    3. Top-K routing
    4. Multiple load balancing strategies
    
    Architecture:
    - Router selects K experts per token
    - Each token processed by selected experts
    - Outputs combined with router weights
    - Shared experts (if any) always contribute
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_experts - config.num_shared_experts
        self.top_k = config.top_k
        self.router_warmup_steps = config.router_warmup_steps
        self.register_buffer('_warmup_step', torch.tensor(0, dtype=torch.long))
        
        # Router network
        self.router = nn.Linear(config.d_model, self.num_routed_experts, bias=False)
        # Kaiming initialization for better routing diversity at startup
        # Formula: std = sqrt(2 / fan_in), where fan_in = d_model
        # For d_model=256: std ≈ 0.088 (9x larger than previous 0.01)
        # For d_model=512: std ≈ 0.063 (6x larger than previous 0.01)
        fan_in = config.d_model
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(self.router.weight, mean=0.0, std=std)
        
        # Routed experts
        self.experts = nn.ModuleList([
            ExpertLayer(
                config.d_model,
                config.d_ff,
                config.expert_dropout,
                use_swiglu=config.use_swiglu_experts
            )
            for _ in range(self.num_routed_experts)
        ])
        
        # Shared experts (always active)
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertLayer(
                    config.d_model,
                    config.d_ff,
                    config.expert_dropout,
                    use_swiglu=config.use_swiglu_experts
                )
                for _ in range(self.num_shared_experts)
            ])
        
        # Load balancing
        if config.load_balance_strategy == 'switch':
            self.aux_loss_fn = SwitchAuxiliaryLoss(self.num_routed_experts)
        elif config.load_balance_strategy == 'deepseek':
            self.bias_correction = DeepSeekBiasCorrection(
                self.num_routed_experts,
                config.bias_lr,
                config.bias_momentum
            )

    def set_warmup_step(self, step: int) -> None:
        """Update current warmup step. Called from training loop."""
        self._warmup_step.fill_(step)

    def reset_warmup(self) -> None:
        """Reset warmup counter to 0."""
        self._warmup_step.zero_()

    @property
    def is_in_warmup(self) -> bool:
        """Check if currently in warmup phase."""
        return self._warmup_step.item() < self.router_warmup_steps    
    
    def forward(
        self,
        x: torch.Tensor,
        train: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Process:
        1. Sort tokens by expert assignment
        2. Batch process all tokens for same expert
        3. Vectorized scatter-add for output combination
        4. Add shared expert outputs (if any)
        
        Args:
            x: [seq_len, batch_size, d_model] input
            train: Whether in training mode
        
        Returns:
            output: Same shape as input
            losses: Dictionary with 'aux_loss' and 'expert_usage'
        """
        seq_len, batch_size, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # [num_tokens, d_model]
        num_tokens = x_flat.shape[0]

        # ========================================================================
        # STEP 1: Router computation
        # ========================================================================
        router_logits = self.router(x_flat)  # [num_tokens, num_experts] 
            
        # Add router balance warmup for stability
        if train and self.is_in_warmup:
            # Blend router logits with uniform distribution
            # blend_factor: 0 at start (uniform) → 1 at end (learned)
            blend_factor = float(self._warmup_step.item()) / float(self.router_warmup_steps)
            uniform_logits = torch.zeros_like(router_logits)
            router_logits = blend_factor * router_logits + (1.0 - blend_factor) * uniform_logits
        
        if self.config.load_balance_strategy == 'deepseek':
            bias = self.bias_correction.get_bias()
            router_logits = router_logits + bias.unsqueeze(0)

        # Z-loss = mean(log(sum(exp(logits))))^2
        # Penalizes large logit magnitudes to prevent softmax saturation
        if train and self.config.z_loss_weight > 0:
            z_loss = torch.logsumexp(router_logits, dim=-1).mean() ** 2
        else:
            z_loss = torch.tensor(0.0, device=x.device, requires_grad=False) 
            
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_gates = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # ========================================================================
        # STEP 2: Create dispatch mask (vectorized approach)
        # ========================================================================
        # Build a [num_tokens, num_experts] mask with gate values
        dispatch_mask = torch.zeros(
            num_tokens, self.num_routed_experts,
            dtype=x.dtype, device=x.device
        )

        # Scatter gate values into dispatch mask
        for k in range(self.top_k):
            dispatch_mask.scatter_add_(
                1,  # dim: experts
                top_k_indices[:, k:k+1],  # indices
                top_k_gates[:, k:k+1]  # values
            )

        # ========================================================================
        # STEP 3: Process each expert with its assigned tokens
        # ========================================================================
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_routed_experts):
            # Get gate weights for this expert
            gates = dispatch_mask[:, expert_idx]  # [num_tokens]

            # Find tokens with non-zero gates (assigned to this expert)
            expert_mask = gates > 0

            if not expert_mask.any():
                continue  # Skip if no tokens for this expert

            # Get tokens and gates for this expert
            expert_tokens = x_flat[expert_mask]  # [num_expert_tokens, d_model]
            expert_gates = gates[expert_mask]  # [num_expert_tokens]

            # Forward through expert (BATCHED!)
            expert_output = self.experts[expert_idx](expert_tokens)

            # Weighted scatter back (VECTORIZED!)
            output[expert_mask] += expert_output * expert_gates.unsqueeze(-1)

        # ========================================================================
        # STEP 4: Add shared experts
        # ========================================================================
        if self.num_shared_experts > 0:
            for shared_expert in self.shared_experts:
                shared_output = shared_expert(x_flat)
                output += shared_output / self.num_shared_experts

        # Reshape back to sequence format
        output = output.reshape(seq_len, batch_size, d_model)

        # ========================================================================
        # STEP 5: Compute losses (same as before)
        # ========================================================================
        losses = {}
        if train:
            if self.config.load_balance_strategy == 'switch':
                losses['aux_loss'] = self.aux_loss_fn(router_probs, top_k_indices)
            else:
                losses['aux_loss'] = torch.tensor(0.0, device=x.device)
            
            # Inclue z-loss for MOE stability
            losses['z_loss'] = z_loss if self.config.z_loss_weight > 0 else torch.tensor(0.0, device=x.device)
            
            if self.config.load_balance_strategy == 'deepseek':
                self.bias_correction.update_bias(top_k_indices)
            

            # Track expert usage (vectorized)
            with torch.no_grad():
                expert_usage = torch.zeros(self.num_routed_experts, device=x.device)
                for k in range(self.top_k):
                    expert_usage.scatter_add_(0, top_k_indices[:, k], torch.ones(num_tokens, device=x.device))
                losses['expert_usage'] = expert_usage / (num_tokens * self.top_k)

        return output, losses


class DataParallelWrapper(nn.Module):
    """
    Wrapper that integrates loss computation into the forward pass.
    
    PURPOSE:
    Standard DataParallel gathers outputs to GPU 0, then loss runs on GPU 0 only.
    This wrapper computes loss on EACH GPU, then DataParallel averages the losses.
    
    MECHANISM:
    1. Forward pass runs on each GPU
    2. Loss computation runs on each GPU
    3. DataParallel gathers LOSS values (scalars) to GPU0
    4. Losses are automatically averaged across GPUs
    
    RESULT:
    - GPU 0 no longer bottlenecked by loss computation
    - All GPUs contribute equally to training
    - ~3-4x speedup with 4 GPUs
    
    Compatible with:
    - BaselineTransformer
    - FlashAttentionTransformer  
    - FlashMoETransformer
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: 'BaseConfig', 
        criterion: nn.Module,
        moe_config: Optional['MoEConfig'] = None
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = criterion
        self.moe_config = moe_config
        self.target_cd_cnt = config.target_cd_cnt
        
        # Detect model type for proper forward handling
        self._is_moe = _model_has_moe(model)
    
    def forward(
        self, 
        x: torch.Tensor,           # [batch, len_dy, features]
        dt_cnt: torch.Tensor,      # [batch] - valid days per sample
        targets: torch.Tensor,     # [batch, len_dy, target_cd_cnt] multi-hot
        return_predictions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with integrated loss computation.
        
        Args:
            x: Input tensor [batch, len_dy, features]
            dt_cnt: Valid day counts [batch]
            targets: Pre-computed multi-hot targets [batch, len_dy, target_cd_cnt]
            return_predictions: If True, also return predictions for metrics
        
        Returns:
            If return_predictions=False: loss tensor (scalar per GPU, averaged by DP)
            If return_predictions=True: (loss, {'predictions': output, 'moe_losses': ...})
        """
        batch_size = x.shape[0]
        actual_len_dy = x.shape[1]
        device = x.device
        
        # ====================================================================
        # STEP 1: MODEL FORWARD (handles both dense and MoE)
        # ====================================================================
        if self._is_moe:
            # MoE models return (output, moe_losses)
            output, moe_losses = self.model(x, return_moe_losses=True)
        else:
            # Dense models return just output
            output = self.model(x)
            moe_losses = {}
        
        # ====================================================================
        # STEP 2: LOSS COMPUTATION (same for all models)
        # ====================================================================
        output_flat = output.view(batch_size * actual_len_dy, self.target_cd_cnt)
        if targets.dtype != output_flat.dtype:
            targets_flat = targets.view(batch_size * actual_len_dy, self.target_cd_cnt).to(output_flat.dtype)
        else:
            targets_flat = targets.view(batch_size * actual_len_dy, self.target_cd_cnt)
        
        # Create valid day mask
        valid_mask = torch.zeros(
            batch_size * actual_len_dy, 
            dtype=torch.bool, 
            device=device
        )
        
        for i in range(batch_size):
            valid_days = min(int(dt_cnt[i].item()), actual_len_dy)
            if valid_days > 0:
                start_idx = i * actual_len_dy
                valid_mask[start_idx:start_idx + valid_days] = True
                
        # Compute loss only on valid positions
        if valid_mask.any():
            valid_output = output_flat[valid_mask]
            valid_targets = targets_flat[valid_mask]
            if valid_targets.dtype != valid_output.dtype:
                valid_targets = valid_targets.to(valid_output.dtype)
            pred_loss = self.criterion(valid_output, valid_targets)
        else:
            pred_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ====================================================================
        # STEP 3: HANDLE MOE AUXILIARY LOSS
        # ====================================================================
        aux_loss = moe_losses.get('aux_loss', torch.tensor(0.0, device=device))
        z_loss = moe_losses.get('z_loss', torch.tensor(0.0, device=device))
        
        # Handle potential multi-element tensor from layer accumulation
        if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        if isinstance(z_loss, torch.Tensor) and z_loss.numel() > 1:
            z_loss = z_loss.mean()
            
        # Combine losses based on MoE config
        if self.moe_config is not None:
            if self.moe_config.load_balance_strategy == 'switch':
                # Switch loss: add weighted auxiliary loss
                total_loss = (pred_loss 
                              + self.moe_config.aux_loss_weight * aux_loss
                              + self.moe_config.z_loss_weight * z_loss)
            else:
                # DeepSeek or other: no auxiliary loss in total
                # add z-loss for better stable
                # (bias correction happens inside model forward)
                total_loss = pred_loss + self.moe_config.z_loss_weight * z_loss
        else:
            # Dense model: just prediction loss
            total_loss = pred_loss
        
        # ====================================================================
        # STEP 4: RETURN (with optional extras for monitoring)
        # ====================================================================
        if return_predictions:
            extras = {
                'predictions': output,
                'pred_loss': pred_loss,
                'aux_loss': aux_loss,
                'z_loss': z_loss,
                'moe_losses': moe_losses  # includes expert_usage for monitoring
            }
            return total_loss, extras
        else:
            return total_loss
    
    # ========================================================================
    # CHECKPOINT COMPATIBILITY METHODS
    # ========================================================================
    
    def get_inner_model(self) -> nn.Module:
        """Get the wrapped model for direct access."""
        return self.model
    
    def state_dict(self, *args, **kwargs):
        """Return inner model state dict (not wrapper state)."""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict to inner model."""
        return self.model.load_state_dict(state_dict, *args, **kwargs)


# Model type detection (used elsewhere)
def _model_has_moe(model: nn.Module) -> bool:
    """
    Check if model is MoE variant.
    
    Handles:
    - Direct model
    - nn.DataParallel wrapped
    - DataParallelWrapper wrapped
    - Double-wrapped (DataParallel + DataParallelWrapper)
    """
    # Unwrap DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module
    
    # Unwrap DataParallelWrapper
    if isinstance(model, DataParallelWrapper):
        model = model.model
    
    # Check for MoE layers
    if hasattr(model, 'temporal_layers'):
        for layer in model.temporal_layers:
            if isinstance(layer, nn.ModuleDict) and 'ffn' in layer:
                if isinstance(layer['ffn'], MoELayer):
                    return True
    
    return False

# ============================================================================
# TRANSFORMER MODELS
# ============================================================================

class BaselineTransformer(nn.Module):
    """
    Baseline hierarchical clinical transformer.
    
    This replicates min_transformer_finetune.py with corrections:
    1. Updated parameters (200 days, 80 codes, 8850 targets)
    2. Fixed multi-label loss handling
    3. Cleaner implementation
    
    Architecture:
    - Daily encoder: 1 layer, 4 heads (encodes codes within each day)
    - Temporal encoder: 6 layers, 16 heads (models patterns across days)
    - No Flash Attention, no MoE
    """
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        
        # ============================================================
        # EMBEDDINGS
        # ============================================================
        # Medical code embeddings
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        
        # Demographics embeddings
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)
        
        # ============================================================
        # DAILY CODE ENCODER
        # ============================================================
        # Encodes co-occurring codes within same day
        # Uses 4 heads as in original
        encoder_layers_cd = TransformerEncoderLayer(
            d_model=config.embedding_size,
            nhead=4,  # Fixed at 4 heads for daily encoder
            dim_feedforward=config.embedding_size,
            dropout=0.0,  # No dropout in daily encoder (as original)
            batch_first=False  # PyTorch default
        )
        self.transformer_encoder_cd = TransformerEncoder(encoder_layers_cd, num_layers=1)
        
        # ============================================================
        # TEMPORAL ENCODER
        # ============================================================
        # Models patterns across days
        encoder_layers_dy = TransformerEncoderLayer(
            d_model=config.embedding_size,
            nhead=16,  # 16 heads as original in min's transformer (head_dim=16)
            dim_feedforward=config.nhid,
            dropout=config.dropout,
            batch_first=False
        )
        self.transformer_encoder_dy = TransformerEncoder(encoder_layers_dy, num_layers=config.nlayers)
        
        # ============================================================
        # OUTPUT LAYERS
        # ============================================================
        self.mm = nn.GELU()  # Activation before temporal encoder
        self.decoder_cd = nn.Linear(config.embedding_size, config.target_cd_cnt)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embedding_size)
        
        # Initialize weights
        self.init_weights()
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for temporal attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        """Initialize output layer weights."""
        initrange = 0.1
        nn.init.zeros_(self.decoder_cd.bias)
        nn.init.uniform_(self.decoder_cd.weight, -initrange, initrange)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical transformer.
        
        Args:
            x: [batch, len_dy, 82] where 82 = [age, gender, 80 codes]
        
        Returns:
            output: [batch, len_dy, target_cd_cnt] logits for multi-label prediction
        
        Process:
        1. Extract age, gender, medical codes
        2. Embed each component
        3. Encode codes within each day
        4. Combine with demographics
        5. Encode temporal patterns across days
        6. Project to target vocabulary
        """
        gpu_batchsize = x.shape[0]
        actual_len_dy = x.shape[1]
        actual_len_cd = x.shape[2] - 3 # demographics (age, gender, lob)
        
        # ============================================================
        # STEP 1: EXTRACT COMPONENTS
        # ============================================================
        age_in_months = x[:, :, 0].long()  # [batch, len_dy]
        gender_cd = x[:, :, 1].long()       # [batch, len_dy]
        lob = x[:, :, 2].long()
        cd = x[:, :, 3:].long()             # [batch, len_dy, len_cd]
        
        # ============================================================
        # STEP 2: EMBED
        # ============================================================
        gender_cd = self.embedding_gender_cd(gender_cd)      # [batch, len_dy, embedding_size]
        age_in_months = self.embedding_age_in_months(age_in_months)  # [batch, len_dy, embedding_size]
        lob_emb = self.embedding_lob(lob)      # [batch, len_dy, embedding_size]
        cd = self.embedding_cd(cd)                           # [batch, len_dy, len_cd, embedding_size]
        
        # Residual connection: sum of all code embeddings
        cd_res = cd.sum(-2)  # [batch, len_dy, embedding_size]
        
        # ============================================================
        # STEP 3: DAILY CODE ENCODING
        # ============================================================
        # Reshape to process all days in parallel
        cd = cd.reshape(gpu_batchsize * actual_len_dy, actual_len_cd, self.config.embedding_size)
        cd = torch.swapaxes(cd, 0, 1)  # [len_cd, batch*len_dy, embedding_size]
        
        # Apply daily transformer
        cd = self.transformer_encoder_cd(cd)
        
        # Max pooling across codes dimension
        cd = cd.permute(1, 2, 0)  # [batch*len_dy, embedding_size, len_cd]
        cd = nn.MaxPool1d(actual_len_cd)(cd)  # [batch*len_dy, embedding_size, 1]
        cd = cd.reshape(gpu_batchsize, actual_len_dy, self.config.embedding_size)
        
        # ============================================================
        # STEP 4: COMBINE REPRESENTATIONS
        # ============================================================
        # Add all embeddings: residual codes + encoded codes + demographics
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb
        cd = self.mm(cd)  # GELU activation
        cd = self.norm(cd)
        
        # ============================================================
        # STEP 5: TEMPORAL ENCODING
        # ============================================================
        # Convert to sequence-first format
        cd = torch.swapaxes(cd, 0, 1)  # [len_dy, batch, embedding_size]
        
        # Generate causal mask
        mth_mask = self._generate_square_subsequent_mask(actual_len_dy).to(x.device)
        
        # Apply temporal transformer
        cd = self.transformer_encoder_dy(cd, mth_mask)
        
        # Convert back to batch-first
        cd = torch.swapaxes(cd, 0, 1)  # [batch, len_dy, embedding_size]
        
        # ============================================================
        # STEP 6: OUTPUT PROJECTION
        # ============================================================
        cd = self.norm(cd)
        cd = self.dropout(cd)
        cd = self.decoder_cd(cd)  # [batch, len_dy, target_cd_cnt]
        
        # NOTE: For BCEWithLogitsLoss, we return raw logits (no softmax)
        # Original used log_softmax which is incorrect for multi-label
        
        return cd


class FlashAttentionTransformer(nn.Module):
    """
    Hierarchical transformer with Flash Attention.
    
    Improvements over baseline:
    1. Flash Attention for memory efficiency
    2. RoPE for better temporal modeling
    3. Pre-normalization for stability
    4. Optional SwiGLU activation
    5. Mixed precision support
    """
    
    def __init__(self, config: FlashAttentionConfig):
        super().__init__()
        self.config = config
        
        # Embeddings (same as baseline)
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)
        
        # ============================================================
        # DAILY ENCODER (can use Flash or standard)
        # ============================================================
        if config.use_flash:
            if config.use_learnt_att_pool:
                self.daily_pooling = LearnedAttentionPooling(
                    d_model=config.embedding_size,
                    dropout=0.0
                )
                # Add optional MLP for extra capacity
                self.daily_mlp = nn.Sequential(
                    nn.Linear(config.embedding_size, config.embedding_size),
                    nn.GELU(),
                    nn.Linear(config.embedding_size, config.embedding_size)
                )
                self.daily_norm = nn.LayerNorm(config.embedding_size)
            else: 
                # Custom Flash Attention layer for daily encoding
                self.daily_attention = FlashAttentionLayer(
                    d_model=config.embedding_size,
                    nhead=4,  # Keep 4 heads for daily encoder
                    dropout=0.0,
                    use_rope=False,  # No position encoding needed within day
                    use_flash=True,
                    max_seq_len=config.len_cd,
                    dtype=config.dtype
                )
                # Add FFN for daily encoder
                self.daily_ffn = nn.Sequential(
                    nn.Linear(config.embedding_size, config.embedding_size),
                    nn.GELU(),
                    nn.Linear(config.embedding_size, config.embedding_size)
                )
                self.daily_norm1 = nn.LayerNorm(config.embedding_size)
                self.daily_norm2 = nn.LayerNorm(config.embedding_size)
        else:
            # Standard transformer encoder
            encoder_layers_cd = TransformerEncoderLayer(
                d_model=config.embedding_size,
                nhead=4,
                dim_feedforward=config.embedding_size,
                dropout=0.0,
                batch_first=False
            )
            self.transformer_encoder_cd = TransformerEncoder(encoder_layers_cd, num_layers=1)
        
        # ============================================================
        # TEMPORAL ENCODER WITH FLASH ATTENTION
        # ============================================================
        self.temporal_layers = nn.ModuleList()
        
        for i in range(config.nlayers):
            # Flash Attention layer
            attn = FlashAttentionLayer(
                d_model=config.embedding_size,
                nhead=config.nhead,  # 8 or 16 depending on choice
                dropout=config.dropout,
                use_rope=config.use_rope,
                use_flash=config.use_flash,
                max_seq_len=config.len_dy,
                dtype=config.dtype
            )
            
            # Feed-forward network
            if config.use_swiglu:
                ffn = SwiGLU(config.embedding_size, config.nhid, config.dropout)
            else:
                ffn = nn.Sequential(
                    nn.Linear(config.embedding_size, config.nhid),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.nhid, config.embedding_size),
                    nn.Dropout(config.dropout)
                )
            
            # Layer norms (pre-norm architecture)
            norm1 = nn.LayerNorm(config.embedding_size)
            norm2 = nn.LayerNorm(config.embedding_size)
            
            self.temporal_layers.append(nn.ModuleDict({
                'attention': attn,
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2
            }))
        
        # Output layers
        self.mm = nn.GELU()
        self.decoder_cd = nn.Linear(config.embedding_size, config.target_cd_cnt)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embedding_size)
        
        self.init_weights()
            
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        nn.init.zeros_(self.decoder_cd.bias)
        nn.init.uniform_(self.decoder_cd.weight, -initrange, initrange)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Flash Attention.
        
        Same structure as baseline but with:
        - Flash Attention for efficiency
        - Pre-normalization
        - Optional RoPE and SwiGLU
        """
        gpu_batchsize = x.shape[0]
        actual_len_dy = x.shape[1]
        actual_len_cd = x.shape[2] - 3 # demographics age, gender, lob
        
        # Extract and embed (same as baseline)
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long()) 
        cd = self.embedding_cd(x[:, :, 3:].long())
        cd_res = cd.sum(-2)
        
        # Daily encoding
        cd = cd.reshape(gpu_batchsize * actual_len_dy, actual_len_cd, self.config.embedding_size)
        cd = torch.swapaxes(cd, 0, 1)  # [len_cd, batch*len_dy, embedding_size]
        # Flash Attention version
        if self.config.use_flash:
            if self.config.use_learnt_att_pool:
                # Learned attention pooling (replaces transformer + max-pool)
                cd = self.daily_pooling(cd)  # [batch*len_dy, embedding_size]

                # Optional MLP for capacity
                cd = self.daily_mlp(cd)
                cd = self.daily_norm(cd)   
                # Regular transformer + max-pool
            else: 
                # Pre-norm attention
                residual = cd
                cd = self.daily_norm1(cd)
                cd = self.daily_attention(cd, is_causal=False)  # No causal mask within day
                cd = residual + cd

                # Pre-norm FFN
                residual = cd
                cd = self.daily_norm2(cd)
                cd = self.daily_ffn(cd)
                cd = residual + cd

                # Max pooling
                cd = cd.permute(1, 2, 0)  # [batch*len_dy, embedding_size, len_cd]
                cd = nn.MaxPool1d(actual_len_cd)(cd)  # [batch*len_dy, embedding_size, 1]
                cd = cd.squeeze(-1)  # [batch*len_dy, embedding_size]
            
            
        else:
            # Standard encoding
            cd = self.transformer_encoder_cd(cd)
            cd = cd.permute(1, 2, 0)
            cd = nn.MaxPool1d(actual_len_cd)(cd)
        
        # Reshape back
        cd = cd.reshape(gpu_batchsize, actual_len_dy, self.config.embedding_size)
        
        # Combine representations
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb
        cd = self.mm(cd)
        cd = self.norm(cd)
        cd = torch.swapaxes(cd, 0, 1)  # [len_dy, batch, embedding_size]
        
        # Temporal encoding with Flash Attention
        for layer in self.temporal_layers:
            # Pre-norm attention block
            residual = cd
            cd_norm = layer['norm1'](cd)
            cd_attn = layer['attention'](cd_norm, is_causal=True)
            cd = residual + cd_attn
            
            # Pre-norm FFN block
            residual = cd
            cd_norm = layer['norm2'](cd)
            cd_ffn = layer['ffn'](cd_norm)
            cd = residual + cd_ffn
        
        # Output projection
        cd = torch.swapaxes(cd, 0, 1)
        
        cd = self.norm(cd)
        cd = self.dropout(cd)
        cd = self.decoder_cd(cd)
        
        return cd


class FlashMoETransformer(nn.Module):
    """
    Hierarchical transformer with Flash Attention + MoE.
    
    Combines:
    1. Flash Attention for memory efficiency
    2. MoE for conditional computation
    3. Flexible configuration for 5 experiments
    
    MoE placement: layers 2-5 (after learning basic patterns); may change to 4-5
    """
    
    def __init__(self, config: FlashAttentionConfig, moe_config: Optional[MoEConfig] = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        self.use_moe_from_layer = moe_config.use_moe_from_layer if moe_config else 999

        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        self.checkpoint_every_n_layers = getattr(config, 'checkpoint_every_n_layers', 2)
        
        # Embeddings
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)
        
        # Daily encoder (Flash Attention, no MoE)
        if self.config.use_learnt_att_pool:
            self.daily_pooling = LearnedAttentionPooling(
                d_model=config.embedding_size,
                dropout=0.0
            )
            self.daily_mlp = nn.Sequential(
                nn.Linear(config.embedding_size, config.embedding_size),
                nn.GELU(),
                nn.Linear(config.embedding_size, config.embedding_size)
            )
            self.daily_norm = nn.LayerNorm(config.embedding_size)            
        else:
            self.daily_attention = FlashAttentionLayer(
                d_model=config.embedding_size,
                nhead=4,
                dropout=0.0,
                use_rope=False,
                use_flash=config.use_flash,
                max_seq_len=config.len_cd,
                dtype=config.dtype
            )
            self.daily_ffn = nn.Sequential(
                nn.Linear(config.embedding_size, config.embedding_size),
                nn.GELU(),
                nn.Linear(config.embedding_size, config.embedding_size)
            )
            self.daily_norm1 = nn.LayerNorm(config.embedding_size)
            self.daily_norm2 = nn.LayerNorm(config.embedding_size)
        
        # Temporal encoder with Flash + MoE
        self.temporal_layers = nn.ModuleList()
        
        for i in range(config.nlayers):
            # Always use Flash Attention
            attn = FlashAttentionLayer(
                d_model=config.embedding_size,
                nhead=config.nhead,
                dropout=config.dropout,
                use_rope=config.use_rope,
                use_flash=config.use_flash,
                max_seq_len=config.len_dy,
                dtype=config.dtype
            )
            
            # FFN: MoE or standard depending on layer
            if moe_config and i >= self.use_moe_from_layer:
                ffn = MoELayer(moe_config)
                is_moe = True
            else:
                if config.use_swiglu:
                    ffn = SwiGLU(config.embedding_size, config.nhid, config.dropout)
                else:
                    ffn = nn.Sequential(
                        nn.Linear(config.embedding_size, config.nhid),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.nhid, config.embedding_size),
                        nn.Dropout(config.dropout)
                    )
                is_moe = False
            
            # Layer norms
            norm1 = nn.LayerNorm(config.embedding_size)
            norm2 = nn.LayerNorm(config.embedding_size)
            self.temporal_layers.append(nn.ModuleDict({
                'attention': attn,
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2,
            }))
            
            
        # Output layers
        self.mm = nn.GELU()
        self.decoder_cd = nn.Linear(config.embedding_size, config.target_cd_cnt)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embedding_size)
        
        self.init_weights()
            
    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder_cd.bias)
        nn.init.uniform_(self.decoder_cd.weight, -initrange, initrange)

    def _checkpointed_layer_forward(
        self, 
        layer_dict: nn.ModuleDict, 
        cd_input: torch.Tensor, 
        layer_idx: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Wrapper for gradient checkpointing.
        Must be a separate function that doesn't capture state (for checkpoint compatibility).
        """
        def layer_fn(x):
            residual = x
            x_norm = layer_dict['norm1'](x)
            x_attn = layer_dict['attention'](x_norm, is_causal=True)
            x = residual + x_attn
            
            residual = x
            x_norm = layer_dict['norm2'](x)
            
            if isinstance(layer_dict['ffn'], MoELayer):
                x_ffn, moe_losses = layer_dict['ffn'](x_norm, train=self.training)
                # Note: moe_losses returned separately since checkpoint doesn't handle dicts
                return residual + x_ffn
            else:
                x_ffn = layer_dict['ffn'](x_norm)
                return residual + x_ffn
        
        return layer_fn
    
    def set_moe_warmup_step(self, step: int) -> None:
        """Propagate warmup step to all MoE layers."""
        for layer in self.temporal_layers:
            if isinstance(layer['ffn'], MoELayer):
                layer['ffn'].set_warmup_step(step)

    def reset_moe_warmup(self) -> None:
        """Reset warmup for all MoE layers."""
        for layer in self.temporal_layers:
            if isinstance(layer['ffn'], MoELayer):
                layer['ffn'].reset_warmup()  
                
    def forward(self, 
                x: torch.Tensor, 
                return_moe_losses: bool = True
               ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with Flash Attention + MoE.
        
        Returns:
            output: [batch, len_dy, target_cd_cnt]
            moe_losses: Dictionary with auxiliary losses
        """
        gpu_batchsize = x.shape[0]
        actual_len_dy = x.shape[1]  
        actual_len_cd = x.shape[2] - 3
        device = x.device
        
        # Extract and embed
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long()) 
        cd = self.embedding_cd(x[:, :, 3:].long())
        cd_res = cd.sum(-2)
        
        # Daily encoding with Flash Attention
        cd = cd.reshape(gpu_batchsize * actual_len_dy, actual_len_cd, self.config.embedding_size)
        cd = torch.swapaxes(cd, 0, 1)

        if self.config.use_learnt_att_pool:
            cd_pooled = self.daily_pooling(cd)
            cd_pooled = self.daily_mlp(cd_pooled) # Optional
            cd = self.daily_norm(cd_pooled)   
        else: 
            # Pre-norm attention
            residual = cd
            cd = self.daily_norm1(cd)
            cd = self.daily_attention(cd, is_causal=False)
            cd = residual + cd

            # Pre-norm FFN
            residual = cd
            cd = self.daily_norm2(cd)
            cd = self.daily_ffn(cd)
            cd = residual + cd

            # Max pooling
            cd = cd.permute(1, 2, 0)  # [batch*len_dy, embedding_size, len_cd]
            cd = nn.MaxPool1d(actual_len_cd)(cd)  # [batch*len_dy, embedding_size, 1]
            cd = cd.squeeze(-1)  # [batch*len_dy, embedding_size]
        
        # Reshape it back
        cd = cd.reshape(gpu_batchsize, actual_len_dy, self.config.embedding_size)
        
        # Combine
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb
        cd = self.mm(cd)
        cd = self.norm(cd)
        cd = torch.swapaxes(cd, 0, 1)
        
        # Temporal encoding with Flash + MoE
        total_aux_loss = torch.tensor(0.0, device=device)
        total_z_loss = torch.tensor(0.0, device=device)
        expert_usage_list = []
        
        for i, layer in enumerate(self.temporal_layers):

            # Determine if this layer should be checkpointed
            should_checkpoint = (
                self.training and 
                self.use_gradient_checkpointing and
                (i % self.checkpoint_every_n_layers == 0)
            )
            
            if should_checkpoint and not isinstance(layer['ffn'], MoELayer):
                # Use gradient checkpointing for non-MoE layers
                # (MoE layers have auxiliary losses that complicate checkpointing)
                def create_custom_forward(layer_module):
                    def custom_forward(x):
                        residual = x
                        x_norm = layer_module['norm1'](x)
                        x_attn = layer_module['attention'](x_norm, is_causal=True)
                        x = residual + x_attn
                        
                        residual = x
                        x_norm = layer_module['norm2'](x)
                        x_ffn = layer_module['ffn'](x_norm)
                        return residual + x_ffn
                    return custom_forward
                
                cd = checkpoint(
                    create_custom_forward(layer),
                    cd,
                    use_reentrant=False
                )
            else:
                # Standard forward (MoE layers or non-checkpointed)
                # Flash Attention block
                residual = cd
                cd_norm = layer['norm1'](cd)
                cd_attn = layer['attention'](cd_norm, is_causal=True)
                cd = residual + cd_attn

                # FFN block (MoE or standard)
                residual = cd
                cd_norm = layer['norm2'](cd)

                # determine if the ffn is MOE or standard FFN
                if isinstance(layer['ffn'], MoELayer):
                    cd_ffn, moe_losses = layer['ffn'](cd_norm, train=self.training)
                    if self.training and return_moe_losses:
                        total_aux_loss += moe_losses['aux_loss']
                        if 'z_loss' in moe_losses:
                            total_z_loss += moe_losses['z_loss']
                        if 'expert_usage' in moe_losses:
                            expert_usage_list.append(moe_losses['expert_usage'].detach())
                else:
                    cd_ffn = layer['ffn'](cd_norm)

                cd = residual + cd_ffn
        
        # Output
        cd = torch.swapaxes(cd, 0, 1)
        
        cd = self.norm(cd)
        cd = self.dropout(cd)
        cd = self.decoder_cd(cd)
        
        # Prepare losses
        moe_losses = {}
        if return_moe_losses and self.training:
            moe_losses['aux_loss'] = total_aux_loss
            moe_losses['z_loss'] = total_z_loss
            if expert_usage_list:
                moe_losses['expert_usage'] = torch.stack(expert_usage_list).mean(dim=0).unsqueeze(0)
        
        return cd, moe_losses

# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

class EmbeddingExtractor:
    """
    Extract embeddings from any model using PyTorch forward hooks.
    
    This is the standard PyTorch way to capture intermediate activations
    without modifying the forward() method.
    
    Usage:
        extractor = EmbeddingExtractor(model)
        
        # Run forward pass normally
        output = model(x)
        
        # Get embeddings (captured automatically)
        embeddings = extractor.get_embeddings()  # [batch, len_dy, embedding_size]
        
        # Clean up when done
        extractor.remove()
    
    Works with: BaselineTransformer, FlashAttentionTransformer, FlashMoETransformer
    """
    
    def __init__(self, model: nn.Module):
        self.wrapped_model = model
        inner = model.module if isinstance(model, nn.DataParallel) else model
        if isinstance(inner, DataParallelWrapper):
            inner = inner.model
        self.model = inner
        self.embeddings = None
        self._hook_handle = None
        
        # Register hook on the appropriate layer based on model type
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook on the layer BEFORE the decoder."""
        
        def hook_fn(module, input, output):
            """Capture the output of the target layer."""
            # Handle different output formats
            if isinstance(output, tuple):
                # Some modules return (output, other_stuff)
                self.embeddings = output[0].detach()
            else:
                self.embeddings = output.detach()
        
        # Determine which layer to hook based on model type
        if isinstance(self.model, BaselineTransformer):
            # Hook the temporal encoder output
            target_layer = self.model.transformer_encoder_dy
            self._hook_handle = target_layer.register_forward_hook(hook_fn)
            
        elif isinstance(self.model, (FlashAttentionTransformer, FlashMoETransformer)):
            # Hook the final temporal layer's output
            # We need to hook the norm BEFORE decoder
            target_layer = self.model.norm  # Final LayerNorm before decoder
            
            # Custom hook that captures BEFORE norm (the raw temporal output)
            def pre_decoder_hook(module, input, output):
                # input[0] is what goes INTO the norm layer = our embedding
                self.embeddings = input[0].detach()
            
            self._hook_handle = target_layer.register_forward_hook(pre_decoder_hook)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model).__name__}")
    
    def get_embeddings(self) -> torch.Tensor:
        """
        Get the captured embeddings.
        
        Returns:
            embeddings: [batch, len_dy, embedding_size] for Flash/MoE models
                        [len_dy, batch, embedding_size] for Baseline (needs transpose)
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings captured. Did you run a forward pass?")
        
        emb = self.embeddings
        
        # Normalize shape to [batch, len_dy, embedding_size]
        if isinstance(self.model, BaselineTransformer):  # Uses unwrapped model
            # Baseline returns [len_dy, batch, embedding_size]
            emb = emb.permute(1, 0, 2)
        
        return emb
    
    def get_patient_embedding(self, dt_cnt: List[int]) -> torch.Tensor:
        """
        Get the embedding for each patient's LAST valid day.
        
        This is what you use for downstream tasks - one embedding per patient.
        
        Args:
            dt_cnt: List of valid day counts per patient
            
        Returns:
            embeddings: [batch, embedding_size]
        """
        embeddings = self.get_embeddings()  # [batch, len_dy, embedding_size]
        if isinstance(dt_cnt, torch.Tensor):
            dt_cnt = dt_cnt.tolist()        
        patient_embeddings = []
        for i, valid_days in enumerate(dt_cnt):
            if valid_days > 0:
                # Get embedding at last valid day
                patient_embeddings.append(embeddings[i, valid_days - 1, :])
            else:
                # Fallback to first position if no valid days
                patient_embeddings.append(embeddings[i, 0, :])
        
        return torch.stack(patient_embeddings)  # [batch, embedding_size]
    
    def remove(self):
        """Remove the hook. Call this when done to free memory."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.embeddings = None
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-cleanup when exiting context."""
        self.remove()
        return False

# ============================================================================
# METRICS LOGGER
# ============================================================================

class MetricsLogger:
    """
    JSON-based metrics logger for structured experiment tracking.
    
    Usage:
        logger = MetricsLogger("exp1_baseline", log_dir="logs")
        logger.log_epoch(epoch=1, metrics={'train_loss': 0.5, 'val_loss': 0.6})
        logger.log_batch(epoch=1, batch=100, metrics={'loss': 0.55, 'recall@10': 0.3})
        logger.save()
    """
    
    def __init__(self, exp_name: str, 
                 log_dir: str = "logs", 
                 resume: bool = False):
        self.exp_name = exp_name
        self.log_path = Path(log_dir) / exp_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        if resume: self.init_resume()
        self.epoch_metrics = []
        self.batch_metrics = []
        self.config = {}
    
    def init_resume(self):
        """initialize the log set up for training resume"""
        epoch_file = self.log_path / 'epoch_metrics.json'
        if epoch_file.exists():
            try:
                with open(epoch_file, 'r') as f:
                    self.epoch_metrics = json.load(f)
                print(f" Loaded {len(self.epoch_metrics)} existing epochs")
            except Exception as e:
                print(f" Could not load existing metrics: {e}")
                self.epoch_metrics = []
        else:
            self.epoch_metrics = []

        batch_file = self.log_path / 'batch_metrics.json'
        if batch_file.exists():
            try:
                with open(batch_file, 'r') as f:
                    self.batch_metrics = json.load(f)
            except:
                self.batch_metrics = []
        else:
            self.batch_metrics = []        
    
    def log_config(self, config: Dict):
        """Log experiment configuration."""
        self.config = config
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        entry = {'epoch': epoch, **metrics}
        self.epoch_metrics.append(entry)
    
    def log_batch(self, epoch: int, batch: int, metrics: Dict[str, float]):
        """Log batch-level metrics (for real-time monitoring)."""
        entry = {'epoch': epoch, 'batch': batch, **metrics}
        self.batch_metrics.append(entry)
    
    @staticmethod
    def convert_to_serializable(obj):
        """Recursively convert numpy/torch types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: MetricsLogger.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [MetricsLogger.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.item() if obj.numel() == 1 else obj.cpu().tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, torch.dtype):
            return str(obj)
        else:
            return obj

    def save(self):
        """Save all metrics to JSON files."""
        # Save epoch metrics
        with open(self.log_path / 'epoch_metrics.json', 'w') as f:
            json.dump(self.convert_to_serializable(self.epoch_metrics), f, indent=2)

        # Save batch metrics
        with open(self.log_path / 'batch_metrics.json', 'w') as f:
            json.dump(self.convert_to_serializable(self.batch_metrics), f, indent=2)

        # Save config
        if self.config:
            with open(self.log_path / 'config.json', 'w') as f:
                json.dump(self.convert_to_serializable(self.config), f, indent=2)
                
    def save_final_results(self, results: Dict):
        """Save complete experiment results to JSON for later comparison."""
        results_path = self.log_path / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.convert_to_serializable(results), f, indent=2)

# ============================================================================
# MODEL LOADING
# ============================================================================

def generate_model_name(
    exp_name: str,
    experiment_round: Optional[str] = None,
    batch_size: int = 32,
    epochs: int = 1,
    embedding_size: int = 256 # 256 by default
) -> str:
    """
    Generate a standardized model name for saving/loading.
    
    Format: {experiment_round}_{exp_name}_bs{batch_size}_ep{epochs}_d{embedding_size}_{timestamp}
    
    Example: round3_exp3_standard_moe_bs32_ep10_d256_20241211_143022
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parts = []
    if experiment_round:
        parts.append(experiment_round)
    parts.append(exp_name)
    parts.append(f"bs{batch_size}")
    parts.append(f"ep{epochs}")
    parts.append(f"d{embedding_size}")
    parts.append(timestamp)
    
    return "_".join(parts)


def save_trained_model(
    model: nn.Module,
    config: BaseConfig,
    model_name: str,
    save_dir: str,
    exp_results: Dict[str, any],
    checkpoint_dir: Optional[str] = None,  # Link to checkpoint for resume
    is_best: bool = False,
    moe_config: Optional[MoEConfig] = None
) -> str:
    """
    Save trained model for inference/downstream evaluation.
    
    This is SEPARATE from save_checkpoint() which is for training resume.
    This creates lightweight, portable model files for:
    - Loading in production
    - Running downstream evaluations
    - Sharing/deploying models
    
    Directory structure after training:
    logs/{experiment_round}/{exp_name}/
    ├── checkpoints/                    # Training resume (save_checkpoint)
    │   ├── checkpoint_latest.pt
    │   ├── checkpoint_best.pt
    │   └── checkpoint_epoch{N}.pt
    ├── saved_models/                   # Inference (save_trained_model)
    │   ├── {model_name}_final.pt       # Lightweight: just weights
    │   ├── {model_name}_config.json
    │   └── {model_name}_results.json
    ├── epoch_metrics.json              # MetricsLogger
    ├── batch_metrics.json
    ├── config.json
    ├── final_results.json
    └── {exp_name}.log                  # Python logger
    
    Args:
        model: Trained model
        config: Model configuration
        model_name: Generated model name (for traceability)
        save_dir: Directory to save model files
        exp_results: Experiment results dictionary
        checkpoint_dir: Path to checkpoint directory (for linking)
        is_best: Whether this is the best model
        
    Returns:
        Path to saved model
    """
    os.makedirs(save_dir, exist_ok=True)
    actual_model = model
    if isinstance(actual_model, nn.DataParallel):
        actual_model = actual_model.module
    if isinstance(actual_model, DataParallelWrapper):
        actual_model = actual_model.model
        
    # 1. Save lightweight model (just state dict + model info)
    model_path = os.path.join(save_dir, f"{model_name}_final.pt")

    save_dict = {
        'model_state_dict': actual_model.state_dict(),
        'model_name': model_name,
        'model_type': type(actual_model).__name__,
        'embedding_size': config.embedding_size,
        'nlayers': config.nlayers,
        # Link back to full checkpoint for resume if needed
        'checkpoint_dir': checkpoint_dir,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'embedding_size': config.embedding_size,
            'nhid': config.nhid,
            'nhead': getattr(config, 'nhead', 8),
            'nlayers': config.nlayers,
            'dropout': config.dropout,
            # FlashAttention-specific configs
            'use_learnt_att_pool': getattr(config, 'use_learnt_att_pool', False),
            'use_swiglu': getattr(config, 'use_swiglu', True),
            'use_rope': getattr(config, 'use_rope', True),
            'use_flash': getattr(config, 'use_flash', True),
        },
        'moe_config': vars(moe_config) if moe_config else None, 
    }
    torch.save(save_dict, model_path)
    print(f"Model saved to: {model_path}")
    
    # 2. Save config (human-readable)
    config_path = os.path.join(save_dir, f"{model_name}_config.json")
    config_dict = {}
    for k, v in config.__dict__.items():
        if isinstance(v, torch.dtype):
            config_dict[k] = str(v)
        elif callable(v):
            config_dict[k] = str(v)
        else:
            config_dict[k] = v
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 3. Save results (with downstream metrics if available)
    results_path = os.path.join(save_dir, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(MetricsLogger.convert_to_serializable(exp_results), f, indent=2)
    
    # 4. Also save as "best" for easy access
    if is_best:
        best_path = os.path.join(save_dir, f"{model_name}_best.pt")
        torch.save(save_dict, best_path)
        print(f"Best model saved to: {best_path}")
    
    return model_path


def load_trained_model(
    model_path: str,
    model_class: type,
    config: BaseConfig,
    device: torch.device
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved .pt file
        model_class: Class of the model (BaselineTransformer, FlashAttentionTransformer, etc.)
        config: Model configuration
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    checkpoint_data = torch.load(model_path, map_location=device)
    moe_config_dict = checkpoint_data.get('moe_config', None) 
    
    # Create model instance
    if model_class == FlashMoETransformer:
        # Additional configurations for MOE
        if moe_config_dict is not None:
            moe_config = MoEConfig(
                d_model=moe_config_dict.get('d_model', config.embedding_size),
                d_ff=moe_config_dict.get('d_ff', config.nhid),
                num_experts=moe_config_dict.get('num_experts', 8),
                num_shared_experts=moe_config_dict.get('num_shared_experts', 0),
                top_k=moe_config_dict.get('top_k', 2),
                expert_dropout=moe_config_dict.get('expert_dropout', 0.05),
                load_balance_strategy=moe_config_dict.get('load_balance_strategy', 'switch'),
                aux_loss_weight=moe_config_dict.get('aux_loss_weight', 0.01),
                bias_lr=moe_config_dict.get('bias_lr', 1e-5),
                bias_momentum=moe_config_dict.get('bias_momentum', 0.9),
                z_loss_weight=moe_config_dict.get('z_loss_weight', 0.0),
                use_moe_from_layer=moe_config_dict.get('use_moe_from_layer', 2),
                use_swiglu_experts=moe_config_dict.get('use_swiglu_experts', False),
                router_warmup_steps=moe_config_dict.get('router_warmup_steps', 500),
            )
        else:
            moe_config = MoEConfig(d_model=config.embedding_size, d_ff=config.nhid)
        model = model_class(config, moe_config)
    else:
        model = model_class(config)
    
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Model type: {checkpoint_data.get('model_type', 'Unknown')}")
    
    return model

# ============================================================================
# DOWNSTREAM EVALUATION
# ============================================================================

# Import downstream metrics - these should be in a separate utils module
# If not available, define placeholders
try:
    from utils.metrics import (lift_at_percentage, 
                               true_positives_at_percentage, 
                               num_samples_at_percentage, 
                               precision_at_percentage, 
                               recall_at_percentage, 
                               f1_at_percentage, 
                               pr_auc_at_percentage, 
                               roc_auc_at_percentage)
except ImportError:
    # Define placeholder functions if metrics module not available
    def lift_at_percentage(y_true, y_prob, pct):
        """Calculate lift at top percentile."""
        n = len(y_true)
        k = max(1, int(n * pct))
        indices = np.argsort(y_prob)[::-1][:k]
        precision_at_k = np.mean(y_true[indices])
        baseline = np.mean(y_true)
        return precision_at_k / baseline if baseline > 0 else 0.0
    
    def true_positives_at_percentage(y_true, y_prob, pct):
        """Count true positives in top percentile."""
        n = len(y_true)
        k = max(1, int(n * pct))
        indices = np.argsort(y_prob)[::-1][:k]
        return int(np.sum(y_true[indices]))
    
    def num_samples_at_percentage(y_true, pct):
        """Count samples at top percentile."""
        return max(1, int(len(y_true) * pct))
    
    def precision_at_percentage(y_true, y_prob, pct):
        """Calculate precision at top percentile."""
        n = len(y_true)
        k = max(1, int(n * pct))
        indices = np.argsort(y_prob)[::-1][:k]
        return float(np.mean(y_true[indices]))
    
    def recall_at_percentage(y_true, y_prob, pct):
        """Calculate recall at top percentile."""
        n = len(y_true)
        k = max(1, int(n * pct))
        indices = np.argsort(y_prob)[::-1][:k]
        tp = np.sum(y_true[indices])
        total_positives = np.sum(y_true)
        return float(tp / total_positives) if total_positives > 0 else 0.0
    
    def f1_at_percentage(y_true, y_prob, pct):
        """Calculate F1 at top percentile."""
        prec = precision_at_percentage(y_true, y_prob, pct)
        rec = recall_at_percentage(y_true, y_prob, pct)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    def pr_auc_at_percentage(y_true, y_prob, pct):
        """Calculate PR-AUC at top percentile."""
        return average_precision_score(y_true, y_prob)
    
    def roc_auc_at_percentage(y_true, y_prob, pct):
        """Calculate ROC-AUC at top percentile."""
        return roc_auc_score(y_true, y_prob)


class DownstreamEvaluator:
    """
    Evaluates transformer embeddings on downstream classification tasks.
    
    Implements the "linear probe" methodology:
    1. Extract embeddings from trained transformer (frozen)
    2. Train a simple logistic regression on top
    3. Evaluate on held-out test set
    
    # why linear probes? 
    Reference: Radford et al. (2021) "CLIP" uses linear probe as primary metric
    
    Usage:
        evaluator = DownstreamEvaluator(model, config, device)
        results = evaluator.evaluate(
            features_df=train_data,
            outcomes_df=outcomes_data,
            downstream_config=DownstreamConfig()
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_config: BaseConfig,
        device: torch.device,
        use_mixed_precision: bool = False
    ):
        # Enable multi-GPU wrapper
        inner_model = model.module if isinstance(model, nn.DataParallel) else model
        # Unwrap DataParallelWrapper
        if isinstance(inner_model, DataParallelWrapper):
            inner_model = inner_model.model
        self.model = inner_model
        self.model_config = model_config
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def extract_embeddings(
        self,
        data: pd.DataFrame,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract member-level embeddings from the transformer.
        
        Args:
            data: DataFrame with transformer input columns + individual_id, index_dt
            batch_size: Batch size for inference
            
        Returns:
            embeddings: np.ndarray [num_patients, embedding_dim]
            individual_ids: List of individual_id strings
            index_dts: List of index_dt strings
        """

        
        # Create dataset and dataloader
        dataset = ClinicalDataset(data, self.model_config)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.model_config),
            num_workers=0
        )
        
        all_embeddings = []
        individual_ids = data['individual_id'].tolist()
        index_dts = data['index_dt'].astype(str).tolist()
        
        self.model.eval()
        
        with torch.no_grad():
            with EmbeddingExtractor(self.model) as extractor:
                for batch_idx, batch in enumerate(dataloader):
                    # Prepare input
                    age = batch['age'].to(self.device)
                    gender = batch['gender'].to(self.device)
                    lob = batch['lob'].to(self.device)
                    codes = batch['codes'].to(self.device)
                    dt_cnt = batch['dt_cnt'] # This is a tensor
                    
                    x = torch.cat([
                        age.unsqueeze(-1),
                        gender.unsqueeze(-1),
                        lob.unsqueeze(-1),
                        codes
                    ], dim=-1)
                    
                    # Forward pass
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            if hasattr(self.model, 'forward') and 'return_moe_losses' in self.model.forward.__code__.co_varnames:
                                _ = self.model(x, return_moe_losses=False)
                            else:
                                _ = self.model(x)
                    else:
                        if hasattr(self.model, 'forward') and 'return_moe_losses' in self.model.forward.__code__.co_varnames:
                            _ = self.model(x, return_moe_losses=False)
                        else:
                            _ = self.model(x)
                    
                    # Get patient-level embeddings (last valid day)
                    dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                    patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                    all_embeddings.append(patient_embs.cpu().numpy())
                    
                    if batch_idx % 100 == 0:
                        print(f"  Extracted embeddings: {(batch_idx + 1) * batch_size}/{len(data)}")
        
        embeddings = np.vstack(all_embeddings)
        print(f"  Total embeddings extracted: {embeddings.shape}")
        
        return embeddings, individual_ids, index_dts
    
    def prepare_downstream_data(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        downstream_config: DownstreamConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Join features with outcomes and create train/val/test splits.
        
        Args:
            features_df: Transformer training data with individual_id, index_dt
            outcomes_df: Outcome table with individual_id, index_dt, acute_ip_flag
            downstream_config: Configuration for splitting
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Preparing downstream data...")
        
        outcome_col = downstream_config.outcome_column
        # Extract embeddings
        embeddings, individual_ids, index_dts = self.extract_embeddings(features_df)
        
        # Create embedding DataFrame for joining
        emb_df = pd.DataFrame({
            'individual_id': individual_ids,
            'index_dt': index_dts
        })
        emb_df['embedding_idx'] = range(len(emb_df))
        emb_df['individual_id'] = emb_df['individual_id'].astype(str)
        
        # Ensure outcomes_df has correct types
        outcomes_df = outcomes_df.copy()
        outcomes_df['individual_id'] = outcomes_df['individual_id'].astype(str)
        outcomes_df['index_dt'] = outcomes_df['index_dt'].astype(str)
        
        # Left join embeddings with outcomes
        merged = emb_df.merge(
            outcomes_df[['individual_id', 'index_dt', outcome_col]],
            on=['individual_id'],
            how='left'
        )
        
        # Handle missing outcomes (default to 0)
        merged[outcome_col] = merged[outcome_col].fillna(0).astype(int)
        
        print(f"  Matched {len(merged)} records")
        print(f"  Label distribution: {merged[outcome_col].value_counts().to_dict()}")
        
        # Get embeddings and labels
        X = embeddings[merged['embedding_idx'].values]
        y = merged[outcome_col].values
        
        # Stratified train/test split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=downstream_config.test_size,
            random_state=downstream_config.random_state,
            stratify=y
        )
        
        # Stratified train/val split from trainval
        val_ratio = downstream_config.val_size / (1 - downstream_config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=downstream_config.random_state,
            stratify=y_trainval
        )
        
        print(f"  Train: {len(X_train)} (pos: {y_train.sum()})")
        print(f"  Val:   {len(X_val)} (pos: {y_val.sum()})")
        print(f"  Test:  {len(X_test)} (pos: {y_test.sum()})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_linear_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        downstream_config: DownstreamConfig
    ) -> Tuple[LogisticRegression, StandardScaler]:
        """
        Train a linear probe (logistic regression) on embeddings.
        
        Args:
            X_train: Training embeddings
            y_train: Training labels
            downstream_config: Configuration
            
        Returns:
            Trained classifier and scaler
        """
        print("Training linear probe...")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train logistic regression
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            classifier = LogisticRegression(
                max_iter=downstream_config.max_iter,
                solver='lbfgs',
                class_weight=downstream_config.class_weight,
                random_state=downstream_config.random_state,
                n_jobs=-1
            )
            classifier.fit(X_train_scaled, y_train)
        
        print("  Linear probe trained successfully")
        return classifier, scaler
    
    ##############################################
    # Xgboost and lightbgm
    ##############################################
    def _compute_scale_pos_weight(self, y_train: np.ndarray) -> float:
        """
        Compute scale_pos_weight for handling class imbalance.
        
        Formula: scale_pos_weight = n_negative / n_positive
        """
        n_positive = np.sum(y_train == 1)
        n_negative = np.sum(y_train == 0)
        
        if n_positive == 0:
            return 1.0
        
        scale_pos_weight = n_negative / n_positive
        print(f"  Class distribution: {n_negative} neg / {n_positive} pos")
        print(f"  Computed scale_pos_weight: {scale_pos_weight:.2f}")
        return scale_pos_weight
    
    
    def train_xgboost_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        downstream_config: DownstreamConfig
    ) -> Tuple[Any, StandardScaler, Any]:
        """
        Train XGBoost classifier with proper imbalance handling and calibration.
        
        Returns:
            - classifier: XGBoost model (or calibrated wrapper)
            - scaler: StandardScaler
            - calibrator: CalibratedClassifierCV if calibration enabled, else None
        """
        print("Training XGBoost probe...")
        
        # Standardize features (less critical for trees, but keeps consistency)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Compute scale_pos_weight for imbalance
        scale_pos_weight = self._compute_scale_pos_weight(y_train)
        
        # XGBoost with default hyperparameters + imbalance handling
        classifier = xgb.XGBClassifier(
            n_estimators=100,          # Default
            max_depth=6,               # Default
            learning_rate=0.2,         # Default
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=downstream_config.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        classifier.fit(X_train_scaled, y_train)
        print("  XGBoost base model trained")
        
        # Calibration using validation set
        calibrator = None
        if downstream_config.calibrate_proba:
            print("  Calibrating probabilities with isotonic regression...")
            # Use isotonic regression for better calibration on large datasets
            # cv='prefit' means we use the already-fitted classifier
            calibrator = CalibratedClassifierCV(
                classifier, 
                method='isotonic',  # Better for large datasets
                cv='prefit'
            )
            calibrator.fit(X_val_scaled, y_val)
            print("  Probability calibration complete")
        
        return classifier, scaler, calibrator

    def train_lightgbm_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        downstream_config: DownstreamConfig
    ) -> Tuple[Any, StandardScaler, Any]:
        """
        Train LightGBM classifier with proper imbalance handling and calibration.
        
        Returns:
            - classifier: LightGBM model (or calibrated wrapper)
            - scaler: StandardScaler
            - calibrator: CalibratedClassifierCV if calibration enabled, else None
        """
        print("Training LightGBM probe...")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Compute scale_pos_weight for imbalance
        scale_pos_weight = self._compute_scale_pos_weight(y_train)
        
        # LightGBM with default hyperparameters + imbalance handling
        classifier = lgb.LGBMClassifier(
            n_estimators=100,          # Default
            max_depth=-1,              # Default (no limit)
            learning_rate=0.1,         # Default
            num_leaves=31,             # Default
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            objective='binary',
            random_state=downstream_config.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        classifier.fit(X_train_scaled, y_train)
        print("  LightGBM base model trained")
        
        # Calibration using validation set
        calibrator = None
        if downstream_config.calibrate_proba:
            print("  Calibrating probabilities with isotonic regression...")
            calibrator = CalibratedClassifierCV(
                classifier,
                method='isotonic',
                cv='prefit'
            )
            calibrator.fit(X_val_scaled, y_val)
            print("  Probability calibration complete")
        
        return classifier, scaler, calibrator    
    
    
    def evaluate_probe(
        self,
        classifier: Any,
        scaler: StandardScaler,
        X: np.ndarray,
        y: np.ndarray,
        percentiles: List[float] = [0.01, 0.1],
        split_name: str = "test",
        calibrator: Any = None
    ) -> Dict[str, float]:
        """
        Evaluate the linear probe on a dataset split.
        
        Args:
            classifier: Trained logistic regression
            scaler: Fitted scaler
            X: Embeddings
            y: True labels
            split_name: Name of split for logging
            
        Returns:
            Dictionary of metrics
        """
        X_scaled = scaler.transform(X)
        
        # Predictions
        if calibrator is not None:
            # Use calibrated probabilities
            y_prob = calibrator.predict_proba(X_scaled)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = classifier.predict(X_scaled)
            y_prob = classifier.predict_proba(X_scaled)[:, 1]
        
        
        # Compute metrics
        metrics = {
            f'{split_name}_accuracy': float((y_pred == y).mean()),
            f'{split_name}_auc_roc': float(roc_auc_score(y, y_prob)),
            f'{split_name}_auc_pr': float(average_precision_score(y, y_prob)),
            f'{split_name}_f1': float(f1_score(y, y_pred)),
            f'{split_name}_precision': float(precision_score(y, y_pred, zero_division=0)),
            f'{split_name}_recall': float(recall_score(y, y_pred, zero_division=0)),
            f'{split_name}_brier': float(brier_score_loss(y, y_prob)),
            f'{split_name}_prevalence': float(y.mean()),
            f'{split_name}_n_samples': int(len(y)),
            f'{split_name}_n_positive': int(y.sum()),
        }
        for pct in percentiles:
            pct_str = f"{int(pct * 100)}pct"  # e.g., "1pct", "5pct", "10pct", "20pct"
            # Lift at top X%
            metrics[f'{split_name}_lift_{pct_str}'] = lift_at_percentage(y, y_prob, pct)
            # ROC_AUC at top X%
            metrics[f'{split_name}_roc_auc_{pct_str}'] = roc_auc_at_percentage(y, y_prob, pct) 
            # PR_AUC at top X%
            metrics[f'{split_name}_pr_auc_{pct_str}'] = pr_auc_at_percentage(y, y_prob, pct) 
            # True positives at top X%
            metrics[f'{split_name}_true_positives_{pct_str}'] = true_positives_at_percentage(y, y_prob, pct)
            # Number of samples at top X%
            metrics[f'{split_name}_n_samples_{pct_str}'] = num_samples_at_percentage(y, pct)
            # Precision at top X% (how many in top X% are actual positives)
            metrics[f'{split_name}_precision_{pct_str}'] = precision_at_percentage(y, y_prob, pct)
            # Recall at top X% (what fraction of all positives are captured)
            metrics[f'{split_name}_recall_{pct_str}'] = recall_at_percentage(y, y_prob, pct)
            # F1 at top X%
            metrics[f'{split_name}_f1_{pct_str}'] = f1_at_percentage(y, y_prob, pct)      
            
        return metrics
    
    def evaluate(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        downstream_config: Optional[DownstreamConfig] = None
    ) -> Dict[str, any]:
        """
        Full downstream evaluation pipeline.
        
        Args:
            features_df: Transformer training data with individual_id, index_dt
            outcomes_df: Outcome table with individual_id, index_dt, acute_ip_flag
            downstream_config: Configuration for evaluation
            
        Returns:
            Dictionary with all metrics for train/val/test splits
        """
        if downstream_config is None:
            downstream_config = DownstreamConfig()
        
        print(f"\n{'='*60}")
        print(f"DOWNSTREAM EVALUATION: {downstream_config.task_name}")
        print(f"Model Type: {downstream_config.model_type}")
        print(f"{'='*60}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_downstream_data(
            features_df, outcomes_df, downstream_config
        )
        
        # Train LR, xgboost, lightgbm
        calibrator = None
        if downstream_config.model_type == 'xgboost':
            classifier, scaler, calibrator = self.train_xgboost_probe(
                X_train, y_train, X_val, y_val, downstream_config
            )
        elif downstream_config.model_type == 'lightgbm':
            classifier, scaler, calibrator = self.train_lightgbm_probe(
                X_train, y_train, X_val, y_val, downstream_config
            )
        else:  # default: logistic regression
            classifier, scaler = self.train_linear_probe(X_train, y_train, downstream_config)
        
        # Evaluate on all splits'
        train_metrics = self.evaluate_probe(classifier, scaler, X_train, y_train, downstream_config.percentiles, 'train', calibrator)
        val_metrics = self.evaluate_probe(classifier, scaler, X_val, y_val, downstream_config.percentiles, 'val', calibrator)
        test_metrics = self.evaluate_probe(classifier, scaler, X_test, y_test, downstream_config.percentiles, 'test', calibrator)
        
        # Combine results
        results = {
            'task_name': downstream_config.task_name,
            'model_type': downstream_config.model_type,
            'calibrated': downstream_config.calibrate_proba and calibrator is not None,
            'embedding_dim': X_train.shape[1],
            **train_metrics,
            **val_metrics,
            **test_metrics,
        }
        
        # Print summary
        print(f"\n--- {downstream_config.model_type.upper()} Probe Results ---")
        print(f"Train AUC-ROC: {train_metrics['train_auc_roc']:.4f}")
        print(f"Val AUC-ROC:   {val_metrics['val_auc_roc']:.4f}")
        print(f"Test AUC-ROC:  {test_metrics['test_auc_roc']:.4f}")
        print(f"Test F1:       {test_metrics['test_f1']:.4f}")
        print(f"Test Recall:   {test_metrics['test_recall']:.4f}")
        print(f"\n--- Top 10% Metrics ---")
        print(f"Test Precision@10%: {test_metrics.get('test_precision_10pct', 'N/A')}")
        print(f"Test Recall@10%:    {test_metrics.get('test_recall_10pct', 'N/A')}")
        print(f"Test F1@10%:        {test_metrics.get('test_f1_10pct', 'N/A')}")
        print(f"Test Lift@10%:      {test_metrics.get('test_lift_10pct', 'N/A')}")
        print(f"{'='*60}\n")
        
        return results


def run_downstream_evaluation(
    model: nn.Module,
    model_config: BaseConfig,
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    device: torch.device,
    use_mixed_precision: bool = False,
    downstream_config: Optional[DownstreamConfig] = None
) -> Dict[str, any]:
    """
    Convenience function to run downstream evaluation on a trained model.
    
    Args:
        model: Trained transformer model
        model_config: Model configuration
        features_df: Training data with individual_id, index_dt, and transformer features
        outcomes_df: Outcomes with individual_id, index_dt, acute_ip_flag
        device: Torch device
        use_mixed_precision: Whether to use FP16 for inference
        downstream_config: Optional downstream evaluation configuration
        
    Returns:
        Dictionary with downstream evaluation metrics
    """
    evaluator = DownstreamEvaluator(
        model=model,
        model_config=model_config,
        device=device,
        use_mixed_precision=use_mixed_precision
    )
    if not downstream_config:
        downstream_config = DownstreamConfig(task_name="medicaid_ip_risk", 
                                             model_type = 'xgboost'
                                            )    
    return evaluator.evaluate(
        features_df=features_df,
        outcomes_df=outcomes_df,
        downstream_config=downstream_config
    )

def run_downstream_evaluation_from_saved_model(
    model_path: str,
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    device: torch.device,
    downstream_config: Optional[DownstreamConfig] = None,
    log_dir: str = "logs/downstream",
) -> Dict[str, Any]:
    """
    Run downstream evaluation using a pretrained model loaded from disk.
    
    This is the standalone downstream evaluation pipeline that can be run
    independently of pretraining.
    
    Args:
        model_path: Path to saved model (.pt file)
        features_df: DataFrame with transformer input columns + individual_id, index_dt
        outcomes_df: DataFrame with individual_id, index_dt, and outcome column
        device: Torch device
        downstream_config: Configuration for downstream evaluation
        log_dir: Directory to save evaluation logs
        
    Returns:
        Dictionary with downstream evaluation metrics
    """
    if downstream_config is None:
        downstream_config = DownstreamConfig()
    
    # ============================================================
    # 1. LOAD MODEL AND CONFIG
    # ============================================================
    print(f"\n{'='*80}")
    print(f"DOWNSTREAM EVALUATION FROM SAVED MODEL")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    print(f"LOB: {downstream_config.lob_name or 'Not specified'}")
    print(f"Task: {downstream_config.task_name}")
    print(f"Model type: {downstream_config.model_type}")
    
    # Load checkpoint
    checkpoint_data = torch.load(model_path, map_location=device)
    
    # Determine model class from checkpoint
    model_type = checkpoint_data.get('model_type', 'FlashAttentionTransformer')
    config_dict = checkpoint_data.get('config', {})
    moe_config_dict = checkpoint_data.get('moe_config', None)
    
    # Reconstruct config
    if 'FlashMoE' in model_type:
        config = FlashAttentionConfig(
            embedding_size=config_dict.get('embedding_size', 256),
            nhid=config_dict.get('nhid', 512),
            nhead=config_dict.get('nhead', 8),
            nlayers=config_dict.get('nlayers', 6),
            dropout=config_dict.get('dropout', 0.1),
            use_learnt_att_pool=config_dict.get('use_learnt_att_pool', False),
            use_swiglu=config_dict.get('use_swiglu', True),
            use_rope=config_dict.get('use_rope', True),
            use_flash=config_dict.get('use_flash', True),
        )
        if moe_config_dict is not None:
            moe_config = MoEConfig(
                d_model=moe_config_dict.get('d_model', config.embedding_size),
                d_ff=moe_config_dict.get('d_ff', config.nhid),
                num_experts=moe_config_dict.get('num_experts', 8),
                num_shared_experts=moe_config_dict.get('num_shared_experts', 0),
                top_k=moe_config_dict.get('top_k', 2),
                expert_dropout=moe_config_dict.get('expert_dropout', 0.05),
                load_balance_strategy=moe_config_dict.get('load_balance_strategy', 'switch'),
                aux_loss_weight=moe_config_dict.get('aux_loss_weight', 0.01),
                bias_lr=moe_config_dict.get('bias_lr', 1e-5),
                bias_momentum=moe_config_dict.get('bias_momentum', 0.9),
                z_loss_weight=moe_config_dict.get('z_loss_weight', 0.0),
                use_moe_from_layer=moe_config_dict.get('use_moe_from_layer', 2),
                use_swiglu_experts=moe_config_dict.get('use_swiglu_experts', False),
                router_warmup_steps=moe_config_dict.get('router_warmup_steps', 500),
            )
            print(f"  Restored MoE config: {moe_config_dict.get('num_experts')} experts, "
                  f"top-{moe_config_dict.get('top_k')}, from layer {moe_config_dict.get('use_moe_from_layer')}")
        else:
            print("  Warning: moe_config not found in checkpoint, using defaults")
            moe_config = MoEConfig(
                d_model=config.embedding_size,
                d_ff=config.nhid
            )            
        model_class = FlashMoETransformer
        model = model_class(config, moe_config)
    elif 'FlashAttention' in model_type:
        config = FlashAttentionConfig(
            embedding_size=config_dict.get('embedding_size', 256),
            nhid=config_dict.get('nhid', 512),
            nhead=config_dict.get('nhead', 8),
            nlayers=config_dict.get('nlayers', 6),
            dropout=config_dict.get('dropout', 0.1),
            use_learnt_att_pool=config_dict.get('use_learnt_att_pool', False),
            use_swiglu=config_dict.get('use_swiglu', True),
            use_rope=config_dict.get('use_rope', True),
            use_flash=config_dict.get('use_flash', True),
        )
        model_class = FlashAttentionTransformer
        model = model_class(config)
    else:
        config = BaseConfig(
            embedding_size=config_dict.get('embedding_size', 256),
            nhid=config_dict.get('nhid', 512),
            nlayers=config_dict.get('nlayers', 6),
            dropout=config_dict.get('dropout', 0.1),
        )
        model_class = BaselineTransformer
        model = model_class(config)
    
    # Load state dict
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model_type}")
    print(f"Embedding size: {config.embedding_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Determine if mixed precision was used during training
    use_mixed_precision = 'Flash' in model_type
    
    # ============================================================
    # 2. RUN DOWNSTREAM EVALUATION
    # ============================================================
    evaluator = DownstreamEvaluator(
        model=model,
        model_config=config,
        device=device,
        use_mixed_precision=use_mixed_precision
    )
    
    results = evaluator.evaluate(
        features_df=features_df,
        outcomes_df=outcomes_df,
        downstream_config=downstream_config
    )
    
    # Add metadata
    results['model_path'] = model_path
    results['model_type'] = model_type
    results['lob_name'] = downstream_config.lob_name
    
    # ============================================================
    # 3. SAVE RESULTS
    # ============================================================
    os.makedirs(log_dir, exist_ok=True)
    
    lob_suffix = f"_{downstream_config.lob_name}" if downstream_config.lob_name else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"downstream_{downstream_config.task_name}{lob_suffix}_{timestamp}.json"
    results_path = os.path.join(log_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(MetricsLogger.convert_to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"{'='*80}\n")
    
    return results


@dataclass
class LOBData:
    """Container for LOB-specific data."""
    lob_name: str
    features_df: pd.DataFrame
    outcomes_df: pd.DataFrame
    downstream_config: Optional[DownstreamConfig] = None


def run_multi_lob_downstream_evaluation(
    model_path: str,
    lob_data_list: List[LOBData],
    device: torch.device,
    base_downstream_config: Optional[DownstreamConfig] = None,
    log_dir: str = "logs/downstream",
) -> Dict[str, Dict[str, Any]]:
    """
    Run downstream evaluation for MULTIPLE LOBs using a single pretrained model.
    
    This is the main entry point for multi-LOB downstream evaluation after
    pretraining a cross-LOB transformer model.
    
    Args:
        model_path: Path to pretrained model (.pt file)
        lob_data_list: List of LOBData objects, each containing:
            - lob_name: Name of the LOB (e.g., 'commercial', 'medicare', 'medicaid')
            - features_df: DataFrame with transformer input features for this LOB
            - outcomes_df: DataFrame with outcomes for this LOB
            - downstream_config: Optional LOB-specific config (overrides base_config)
        device: Torch device
        base_downstream_config: Base configuration (can be overridden per-LOB)
        log_dir: Directory to save evaluation logs
        
    Returns:
        Dictionary mapping LOB names to their evaluation results
    """
    print(f"\n{'='*80}")
    print(f"MULTI-LOB DOWNSTREAM EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"LOBs to evaluate: {[lob.lob_name for lob in lob_data_list]}")
    print(f"{'='*80}\n")
    
    if base_downstream_config is None:
        base_downstream_config = DownstreamConfig()
    
    all_results = {}
    
    for lob_data in lob_data_list:
        print(f"\n{'─'*60}")
        print(f"Processing LOB: {lob_data.lob_name}")
        print(f"{'─'*60}")
        
        # Use LOB-specific config or fall back to base config
        if lob_data.downstream_config is not None:
            downstream_config = lob_data.downstream_config
        else:
            # Clone base config and set LOB name
            downstream_config = DownstreamConfig(
                task_name=base_downstream_config.task_name,
                test_size=base_downstream_config.test_size,
                val_size=base_downstream_config.val_size,
                random_state=base_downstream_config.random_state,
                percentiles=base_downstream_config.percentiles,
                n_cv_folds=base_downstream_config.n_cv_folds,
                max_iter=base_downstream_config.max_iter,
                class_weight=base_downstream_config.class_weight,
                model_type=base_downstream_config.model_type,
                calibrate_proba=base_downstream_config.calibrate_proba,
                lob_name=lob_data.lob_name,
                outcome_column=base_downstream_config.outcome_column,
            )
        
        # Ensure LOB name is set
        downstream_config.lob_name = lob_data.lob_name
        
        # Create LOB-specific log directory
        lob_log_dir = os.path.join(log_dir, lob_data.lob_name)
        
        # Run evaluation for this LOB
        try:
            lob_results = run_downstream_evaluation_from_saved_model(
                model_path=model_path,
                features_df=lob_data.features_df,
                outcomes_df=lob_data.outcomes_df,
                device=device,
                downstream_config=downstream_config,
                log_dir=lob_log_dir,
            )
            all_results[lob_data.lob_name] = lob_results
            
            print(f"  ✓ {lob_data.lob_name} - Test AUC-ROC: {lob_results['test_auc_roc']:.4f}")
            
        except Exception as e:
            print(f"  ✗ {lob_data.lob_name} - Error: {str(e)}")
            all_results[lob_data.lob_name] = {'error': str(e)}
    
    # ============================================================
    # AGGREGATE SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print("MULTI-LOB EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    summary_table = []
    for lob_name, results in all_results.items():
        if 'error' not in results:
            summary_table.append({
                'LOB': lob_name,
                'N': results.get('test_n_samples', 0),
                'Prevalence': results.get('test_prevalence', 0),
                'AUC-ROC': results.get('test_auc_roc', 0),
                'PR-AUC': results.get('test_auc_pr', 0),
                'F1': results.get('test_f1', 0),
                'Precision@10%': results.get('test_precision_10pct', 0),
                'Recall@10%': results.get('test_recall_10pct', 0),
                'Lift@10%': results.get('test_lift_10pct', 0),
            })
    
    if summary_table:
        summary_df = pd.DataFrame(summary_table)
        print(summary_df.to_string(index=False))
    
    # Save aggregate summary
    summary_path = os.path.join(log_dir, f"multi_lob_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(MetricsLogger.convert_to_serializable(all_results), f, indent=2)
    
    print(f"\nAggregate summary saved to: {summary_path}")
    print(f"{'='*80}\n")
    
    return all_results


