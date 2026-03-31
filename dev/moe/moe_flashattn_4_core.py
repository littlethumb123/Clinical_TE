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
# LOGGING
# ============================================================================

def setup_experiment_logging(
    exp_name: str,
    log_dir: str = "logs",
    resume: bool = False
) -> logging.Logger:
    """
    Set up comprehensive logging for experiment tracking.
    
    Creates:
    1. Console logger (INFO level)
    2. File logger (DEBUG level) - saves to logs/{exp_name}/training.log
    """
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
        """
        Parse target string for a single member and return the set of unique
        positive target code indices. Used by streaming tier computation.
        """
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
        return results_path

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.epoch_metrics:
            return {}
        
        real_epochs = [m for m in self.epoch_metrics if 'resume_event' not in m]
        if not real_epochs:
            return {}

        final_epoch = real_epochs[-1]
        best_val_loss_epoch = min(real_epochs, key=lambda x: x.get('val_loss', float('inf')))

        return {
            'num_epochs': len(real_epochs),
            'final_train_loss': final_epoch.get('train_loss', 0),
            'final_val_loss': final_epoch.get('val_loss', 0),
            'best_val_loss': best_val_loss_epoch.get('val_loss', 0),
            'best_epoch': best_val_loss_epoch.get('epoch', 0)
        }

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


# ============================================================================
# TRAINING PIPELINE
# ============================================================================
# Functions below support the full training workflow:
# - Loss functions and weight helpers
# - Scheduler/optimizer factories
# - Training loop and evaluation
# - Checkpoint management
# - Experiment orchestration
# ============================================================================

import shutil
from torch.utils.data import Sampler
import copy

# --- Data preparation utilities ---

def prepare_tensor(
    batch: pd.DataFrame,
    config: BaseConfig,
    device: torch.device
) -> Tuple[List[int], torch.Tensor, List[List[List[int]]]]:
    """
    Prepare batch for model input.
    1. Pre-allocate tensors instead of building lists
    2. Batch conversion operations
    3. Minimize list comprehensions    
    Returns:
        dt_cnt: List of actual day counts per sample
        x: Input tensor [batch_size, len_dy, 2 + len_cd]
        y: Target codes List[List[List[int]]] - nested list for multi-label
    """
    batch_size = len(batch)
    
    # Extract all columns at once (faster than row iteration)
    age_strs = batch['age_in_months'].tolist()
    gender_strs = batch['gender_cd'].tolist()
    cd_strs = batch['cd'].tolist()
    target_strs = batch['target'].tolist() # In raw feature table, this is target column
    dt_cnt = batch['dt_cnt'].tolist()
    
    # Pre-allocate output tensors
    age_tensor = torch.zeros(batch_size, config.len_dy, dtype=torch.long, device=device)
    gender_tensor = torch.zeros(batch_size, config.len_dy, dtype=torch.long, device=device)
    cd_tensor = torch.zeros(batch_size, config.len_dy, config.len_cd, dtype=torch.long, device=device)
    
    # Parallel parsing (can be further optimized with numba/cython if needed)
    for i in range(batch_size):
        age_list = conv_age_gender(age_strs[i], config.len_dy)
        gender_list = conv_age_gender(gender_strs[i], config.len_dy, max_val=3)
        cd_list = conv_cd(cd_strs[i], config.len_dy, config.len_cd)
        
        age_tensor[i] = torch.tensor(age_list, dtype=torch.long)
        gender_tensor[i] = torch.tensor(gender_list, dtype=torch.long)
        cd_tensor[i] = torch.tensor(cd_list, dtype=torch.long)
    
    # Concatenate
    x = torch.cat([
        age_tensor.unsqueeze(-1),
        gender_tensor.unsqueeze(-1),
        cd_tensor
    ], dim=-1)
    
    # Parse targets (keep as list for multi-label)
    y = [conv_target(target_strs[i], config.len_dy, config.target_cd_cnt) 
         for i in range(batch_size)]
    
    return dt_cnt, x, y


def create_multihot_targets_vectorized(
    y_flat: List[List[int]],
    num_samples: int,
    vocab_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Vectorized multi-hot target construction.
    
    Instead of:
        for j in range(num_samples):
            for k in y_flat[j]:
                y_cd[j, k] = 1
    
    We build index tensors once and scatter:
        y_cd.scatter_(1, indices, 1.0)
    
    Speedup: ~20-50× faster than nested loops for large vocab.
    
    Args:
        y_flat: List[List[int]] - target codes per sample
        num_samples: Number of samples
        vocab_size: Target vocabulary size
        device: Target device
    
    Returns:
        y_cd: [num_samples, vocab_size] multi-hot tensor
    """
    # Pre-allocate output
    y_cd = torch.zeros(num_samples, vocab_size, device=device)
    
    # Build index lists
    row_indices = []
    col_indices = []
    
    for j in range(num_samples):
        for k in y_flat[j]:
            if k != 0 and k < vocab_size:
                row_indices.append(j)
                col_indices.append(k)
    
    # Early exit if no valid targets
    if len(row_indices) == 0:
        return y_cd
    
    # Convert to tensors (single GPU transfer)
    row_idx = torch.tensor(row_indices, dtype=torch.long, device=device)
    col_idx = torch.tensor(col_indices, dtype=torch.long, device=device)
    
    # Vectorized scatter (single kernel call)
    y_cd[row_idx, col_idx] = 1.0
    
    return y_cd





# --- Loss weight computation ---

# ============================================================
# Pos weighting methods 
# Log-Scaled	Smooth gradients	Continuous, no sudden jumps	Hard to interpret exact boosts
# ENS	Theoretical rigor	Single β parameter, principled	Sensitive to β choice
# Tiered	Explicit control	Interpretable, tunable	Discrete (may miss nuance)
# ============================================================
def compute_pos_weights(
    code_frequencies: np.ndarray,
    device: torch.device,
    max_weight: float = 100.0
) -> torch.Tensor:
    """
    Original inverse frequency weighting with capping.
    
    Formula: weight = max_freq / freq, capped at max_weight.
    
    Note: This method suffers from extreme capping when imbalance is severe
    (e.g., 16M:1 means 95%+ codes get capped to max_weight).
    Consider using 'log_scaled', 'ens', or 'tiered' methods instead.
    """
    # Add smoothing to avoid division by zero
    freq_safe = code_frequencies.astype(np.float64) + 1.0
    max_freq = freq_safe.max()
    
    # Inverse frequency weighting
    weights = max_freq / freq_safe
    
    # Cap weights
    weights = np.clip(weights, 1.0, max_weight)
    
    print(f"  Inverse freq weights: min={weights.min():.2f}, max={weights.max():.2f}, "
          f"mean={weights.mean():.2f}, median={np.median(weights):.2f}")
    
    return torch.tensor(weights, dtype=torch.float32, device=device)

def compute_log_scaled_weights(
    code_frequencies: np.ndarray,
    device: torch.device,
    max_weight: float = 100.0,
    min_weight: float = 1.0
) -> torch.Tensor:
    """
    Log-scaled inverse frequency weighting.
    
    Compresses extreme imbalance ratios (e.g., 16M:1) to manageable range
    while preserving relative ordering.
    
    Formula: weight = log(max_freq + 1) / log(freq + 1), then scaled to [min, max]
    
    Example for 16M:1 imbalance:
        - Freq=1 → weight ≈ 100 (not 16M!)
        - Freq=479 → weight ≈ 38
        - Freq=16.9M → weight ≈ 1
    """
    # Add 1 to handle zero frequencies
    freq_safe = code_frequencies.astype(np.float64) + 1.0
    
    # Log-transform
    log_freq = np.log(freq_safe)
    log_max = np.log(freq_safe.max())
    
    # Inverse log ratio
    weights = log_max / np.maximum(log_freq, 1e-8)
    
    # Scale to desired range [min_weight, max_weight]
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    weights_scaled = min_weight + weights_norm * (max_weight - min_weight)
    
    # Final clipping for safety
    weights_final = np.clip(weights_scaled, min_weight, max_weight)
    
    print(f"  Log-scaled weights: min={weights_final.min():.2f}, max={weights_final.max():.2f}, "
          f"mean={weights_final.mean():.2f}, median={np.median(weights_final):.2f}")
    
    return torch.tensor(weights_final, dtype=torch.float32, device=device)


def compute_effective_number_weights(
    code_frequencies: np.ndarray,
    device: torch.device,
    beta: float = 0.9999,
    max_weight: float = 100.0,
    min_weight: float = 1.0
) -> torch.Tensor:
    """
    Class-balanced loss using Effective Number of Samples (ENS).
    
    From: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., CVPR 2019)
    
    Formula: 
        E_n = (1 - β^n) / (1 - β)
        weight = 1 / E_n
    
    Args:
        beta: Controls how fast returns diminish
              - beta=0.9:    Mild reweighting
              - beta=0.999:  Moderate
              - beta=0.9999: Aggressive (for extreme imbalance like yours)
    """
    freq_safe = code_frequencies.astype(np.float64)
    freq_safe[freq_safe == 0] = 1  # Handle zero frequencies
    
    # Effective number: E_n = (1 - β^n) / (1 - β)
    effective_n = (1.0 - np.power(beta, freq_safe)) / (1.0 - beta)
    
    # Weight inversely proportional to effective number
    weights = 1.0 / effective_n
    
    # Normalize to [min_weight, max_weight]
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    weights_scaled = min_weight + weights_norm * (max_weight - min_weight)
    
    weights_final = np.clip(weights_scaled, min_weight, max_weight)
    
    print(f"  ENS weights (beta={beta}): min={weights_final.min():.2f}, max={weights_final.max():.2f}, "
          f"mean={weights_final.mean():.2f}, median={np.median(weights_final):.2f}")
    
    return torch.tensor(weights_final, dtype=torch.float32, device=device)


def compute_tiered_weights(
    code_frequencies: np.ndarray,
    device: torch.device,
    tier_config: Optional[dict] = None
) -> torch.Tensor:
    """
    Quantile-based tiered weighting with explicit control.
    
    Assigns discrete weights to frequency tiers, giving explicit,
    interpretable control over the boost for each tier.
    
    Default tiers (based on percentiles of non-zero frequencies):
        - Ultra-rare (0-5th percentile):   weight = 100
        - Tail (5-25th percentile):        weight = 50
        - Rare (25-50th percentile):       weight = 25
        - Medium (50-75th percentile):     weight = 10
        - Common (75-90th percentile):     weight = 3
        - Very common (>90th percentile):  weight = 1
    
    Args:
        code_frequencies: Array of code frequencies
        device: Torch device
        tier_config: Optional dict to override default tier weights
    """
    # better specify the tier config ourselves otherwise variance can results in unstabilization 
    # in token routing (with deepseek bias) Don't do very large
    if tier_config is None:
        tier_config = {
            'ultra_rare':  {'percentile': (0, 5),    'weight': 100},
            'tail':        {'percentile': (5, 25),   'weight': 50},
            'rare':        {'percentile': (25, 50),  'weight': 25},
            'medium':      {'percentile': (50, 75),  'weight': 10},
            'common':      {'percentile': (75, 90),  'weight': 3},
            'very_common': {'percentile': (90, 100), 'weight': 1},
        }
    
    # Get non-zero frequencies for percentile calculation
    freq_nz = code_frequencies[code_frequencies > 0]
    
    # Initialize all weights to 1
    weights = np.ones(len(code_frequencies), dtype=np.float32)
    
    print("  Tiered weights distribution:")
    for tier_name, config in tier_config.items():
        p_low, p_high = config['percentile']
        weight = config['weight']
        
        # Calculate frequency thresholds
        thresh_low = np.percentile(freq_nz, p_low) if p_low > 0 else 0
        thresh_high = np.percentile(freq_nz, p_high) if p_high < 100 else np.inf
        
        # Create mask for this tier
        if p_low == 0:
            # Include zero-frequency codes in the lowest tier
            mask = (code_frequencies >= thresh_low) & (code_frequencies < thresh_high)
            mask = mask | (code_frequencies == 0)
        else:
            mask = (code_frequencies >= thresh_low) & (code_frequencies < thresh_high)
        
        weights[mask] = weight
        
        count = mask.sum()
        print(f"    {tier_name:<12}: {count:>5} codes (freq {thresh_low:.0f}-{thresh_high:.0f}), weight={weight}")
    
    print(f"  Final: min={weights.min():.2f}, max={weights.max():.2f}, "
          f"mean={weights.mean():.2f}, median={np.median(weights):.2f}")
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


# --- Loss function classes ---

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (Ridnik et al., 2021).
    
    Designed specifically for multi-label problems with long-tail distributions.
    Uses different focusing parameters for positives vs negatives:
    
    L_+ = (1 - p)^γ+ × BCE(p, 1)      [positives]
    L_- = (p_m)^γ-   × BCE(p_m, 0)     [negatives, with margin]
    
    where p_m = max(p - m, 0) is the probability shifted by margin m.
    
    Key advantages over standard Focal Loss:
    - γ+ = 0: Preserves ALL positive gradients (critical for tail codes)
    - γ- = 4: Aggressively down-weights easy negatives (common codes already learned)
    - Margin m: Hard-thresholds very easy negatives to zero contribution
    
    Reference:
        "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., ICCV 2021)
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma_pos: Focusing parameter for positive examples.
                0.0 = no down-weighting (preserve all positive gradients)
                1.0+ = down-weight easy positives
            gamma_neg: Focusing parameter for negative examples.
                4.0 = strong down-weighting of easy negatives (recommended)
                2.0 = moderate
            clip: Probability margin for negatives. Shifts p down by this amount 
                before computing loss. Effectively zeros out contribution from 
                negatives with p < clip. Set 0.0 to disable.
            pos_weight: Optional per-class weights [num_classes] for additional
                frequency-based reweighting (can combine with ASL)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: [batch, ..., num_classes] raw model outputs (before sigmoid)
            targets: [batch, ..., num_classes] binary targets (0 or 1)
        
        Returns:
            ASL loss (scalar if reduction='mean' or 'sum')
        """
        if targets.dtype != logits.dtype:
            targets = targets.to(logits.dtype)
        
        # Compute probabilities
        p = torch.sigmoid(logits)
        
        # Numerically stable BCE per-element (handles float16 safely)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # For negatives: apply probability margin (clipping)
        p_neg = p
        if self.clip > 0:
            p_neg = (p - self.clip).clamp(min=0.0)
        
        # Asymmetric focusing weights
        # Positives: (1-p)^γ+  (when γ+=0, this is 1.0 — no modulation)
        # Negatives: (p_m)^γ-  (when γ-=4, easy negatives get strongly down-weighted)
        pos_weight_factor = (1.0 - p) ** self.gamma_pos if self.gamma_pos > 0 else 1.0
        neg_weight_factor = p_neg ** self.gamma_neg if self.gamma_neg > 0 else 1.0
        
        # Combine: apply pos weights where target=1, neg weights where target=0
        modulation = targets * pos_weight_factor + (1 - targets) * neg_weight_factor
        
        loss = modulation * bce
        
        # Apply per-class pos_weight if provided
        if self.pos_weight is not None:
            loss = loss * self.pos_weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification with extreme class imbalance.
    
    Focal Loss down-weights easy examples to focus training on hard ones.
    Can be combined with pos_weight for class-balanced focal loss.
    
    Formula:
        FL(p, y) = -α × (1-p)^γ × log(p) × y  
                 - (1-α) × p^γ × log(1-p) × (1-y)
    
    When combined with pos_weight:
        Combined = pos_weight[class] × FL(p, y)
    
    Args:
        gamma: Focusing parameter
               - gamma=0: Equivalent to BCE
               - gamma=2: Standard (recommended)
               - gamma=3+: Aggressive (for extreme imbalance)
        alpha: Balance between positive/negative
               - alpha=0.25: Typical for many negatives
               - alpha=0.5: Balanced
        pos_weight: Optional per-class weights [num_classes]
        reduction: 'mean', 'sum', or 'none'
    
    Reference:
        "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: [batch, ..., num_classes] raw model outputs (before sigmoid)
            targets: [batch, ..., num_classes] binary targets (0 or 1)
        
        Returns:
            Focal loss (scalar if reduction='mean' or 'sum')
        """
        # Ensure same dtype
        if targets.dtype != logits.dtype:
            targets = targets.to(logits.dtype)
        
        # Compute probabilities (numerically stable)
        p = torch.sigmoid(logits)
        
        # Compute focal modulation weights
        # For positives (y=1): weight = (1-p)^γ  → down-weight when p is high (easy)
        # For negatives (y=0): weight = p^γ     → down-weight when p is low (easy)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weights (balance positive/negative contribution)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE component (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal modulation and alpha
        focal_loss = alpha_weight * focal_weight * bce
        
        # Apply per-class pos_weight if provided (for class imbalance)
        if self.pos_weight is not None:
            # Ensure pos_weight is on same device
            if self.pos_weight.device != focal_loss.device:
                self.pos_weight = self.pos_weight.to(focal_loss.device)
            
            # pos_weight shape: [num_classes]
            # focal_loss shape: [batch, ..., num_classes]
            # Broadcast and multiply
            focal_loss = focal_loss * self.pos_weight
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# #### Weighted loss function(weighted + Focal loss)



def create_criterion(
    code_frequencies: np.ndarray,
    device: torch.device,
    optimize_config: 'OptimizeConfig'
) -> nn.Module:
    """
    Factory function to create the appropriate loss criterion.
    
    Handles:
    1. Weight computation method (inverse, log_scaled, ens, tiered)
    2. BCE vs Focal Loss selection
    3. Combining pos_weight with loss function
    
    Args:
        code_frequencies: Array of code frequencies
        device: Torch device
        optimize_config: Configuration with loss settings
    
    Returns:
        Configured loss criterion (BCEWithLogitsLoss or FocalLoss)
    """
    pos_weight = None
    if optimize_config.use_asl and optimize_config.use_focal_loss:
        print("  ⚠️ Warning: Both use_asl and use_focal_loss are True. Using ASL (takes priority).")    
    # Step 1: Compute pos_weight if enabled
    if optimize_config.use_pos_weight:
        method = optimize_config.pos_weight_method
        max_weight = optimize_config.pos_weight_max
        
        print(f"  Computing pos_weight using method: '{method}'")
        
        if method == 'log_scaled':
            pos_weight = compute_log_scaled_weights(
                code_frequencies, device, max_weight=max_weight
            )
        elif method == 'ens':
            pos_weight = compute_effective_number_weights(
                code_frequencies, device, 
                beta=optimize_config.ens_beta,
                max_weight=max_weight
            )
        elif method == 'tiered':
            pos_weight = compute_tiered_weights(
                code_frequencies, device,
                tier_config=optimize_config.tier_weights
            )
        else:  # Default: 'inverse' - original method
            pos_weight = compute_pos_weights(
                code_frequencies, device, max_weight=max_weight
            )
    
    # Step 2: Create criterion (ASL > Focal > BCE priority)

    if optimize_config.use_asl:
        criterion = AsymmetricLoss(
            gamma_pos=optimize_config.asl_gamma_pos,
            gamma_neg=optimize_config.asl_gamma_neg,
            clip=optimize_config.asl_clip,
            pos_weight=pos_weight,
            reduction='mean'
        )
        loss_name = (f"AsymmetricLoss(γ+={optimize_config.asl_gamma_pos}, "
                     f"γ-={optimize_config.asl_gamma_neg}, "
                     f"clip={optimize_config.asl_clip})")
    elif optimize_config.use_focal_loss:
        criterion = FocalLoss(
            gamma=optimize_config.focal_gamma,
            alpha=optimize_config.focal_alpha,
            pos_weight=pos_weight,
            reduction='mean'
        )
        loss_name = f"FocalLoss(gamma={optimize_config.focal_gamma}, alpha={optimize_config.focal_alpha})"
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_name = "BCEWithLogitsLoss"
        
    # Log what was created
    if pos_weight is not None:
        print(f"  Created: {loss_name} with pos_weight ({optimize_config.pos_weight_method})")
    else:
        print(f"  Created: {loss_name} without pos_weight")
    
    return criterion


# #### Loss logger



# --- Training utilities ---

class LossTracker:
    """
    Track training loss trajectory for learning curve analysis.
    
    Similar to HuggingFace TrainerState - tracks:
    1. Per-batch losses (for learning curves)
    2. Per-epoch statistics (mean, std, min, max)
    3. Convergence diagnostics
    
    Design:
    - Lightweight: Only stores statistics, not all values
    - Efficient: Rolling statistics for large epochs
    - Compatible: Works with existing metrics system
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: How many recent batches to keep for rolling stats
        """
        self.window_size = window_size
        self.reset_epoch()
        
        # Cross-epoch tracking
        self.epoch_summaries = []  # Summary per epoch
    
    def reset_epoch(self):
        """Reset for new epoch"""
        self.batch_losses = []        # All batch losses this epoch
        self.batch_steps = []         # Global step for each batch
        self.running_sum = 0.0        # For efficient mean calculation
        self.running_count = 0
    
    def log_batch(self, loss: float, step: int):
        """
        Log a single batch loss.
        
        Args:
            loss: Loss value from this batch
            step: Global training step
        """
        self.batch_losses.append(loss)
        self.batch_steps.append(step)
        self.running_sum += loss
        self.running_count += 1
    
    def get_recent_losses(self, n: int = None) -> List[float]:
        """Get last N losses (for smoothing)"""
        if n is None:
            n = self.window_size
        return self.batch_losses[-n:] if len(self.batch_losses) >= n else self.batch_losses
    
    def get_epoch_summary(self) -> Dict[str, float]:
        """
        Get statistical summary of this epoch.
        
        Returns comprehensive statistics for monitoring:
        - Mean, std, min, max
        - First batch vs last batch (learning delta)
        - Convergence indicators
        """
        if len(self.batch_losses) == 0:
            return {}
        
        losses_array = np.array(self.batch_losses)
        
        summary = {
            'train_loss_mean': float(np.mean(losses_array)),      # Average over epoch
            'train_loss_std': float(np.std(losses_array)),        # Variance (stability)
            'train_loss_min': float(np.min(losses_array)),        # Best batch
            'train_loss_max': float(np.max(losses_array)),        # Worst batch
            'train_loss_first': float(losses_array[0]),           # Epoch start
            'train_loss_last': float(losses_array[-1]),           # Epoch end (final model)
            'train_loss_improvement': float(losses_array[0] - losses_array[-1]),  # Learning delta
        }
        
        # Smoothed loss (last 100 batches)
        if len(losses_array) >= 100:
            smoothed = np.convolve(losses_array, np.ones(100)/100, mode='valid')
            summary['train_loss_smoothed'] = float(smoothed[-1])
        
        # Store for cross-epoch analysis
        self.epoch_summaries.append(summary)
        
        return summary
    
    def save_trajectory(self, filepath: str):
        """Save full loss trajectory for plotting"""
        trajectory = {
            'steps': self.batch_steps,
            'losses': self.batch_losses,
            'epoch_summaries': self.epoch_summaries
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(trajectory, f, indent=2)
    
    def should_stop_early(self, patience: int = 3) -> bool:
        """
        Check if training should stop (loss not improving).
        
        Args:
            patience: Number of epochs without improvement
        
        Returns:
            True if should stop early
        """
        if len(self.epoch_summaries) < patience + 1:
            return False
        
        # Check if loss increased for 'patience' consecutive epochs
        recent_means = [s['train_loss_mean'] for s in self.epoch_summaries[-patience-1:]]
        
        # If all recent epochs have higher loss than the best, stop
        best_loss = min(s['train_loss_mean'] for s in self.epoch_summaries)
        recent_worse = all(loss > best_loss * 1.05 for loss in recent_means[-patience:])
        
        return recent_worse




# --- Schedulers ---

import torch.optim as optim

def get_linear_warmup_plateau_decay(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    plateau_ratio: float = 0.3,
    min_lr_ratio: float = 0.1
) -> optim.lr_scheduler.LambdaLR:
    """
    Linear warmup → Plateau at peak → Linear decay.
    
    Better for single-epoch training where you want maximum learning early.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Steps for linear warmup
        num_training_steps: Total training steps
        plateau_ratio: Fraction of training to stay at peak LR (after warmup)
        min_lr_ratio: Minimum LR as ratio of peak (0.1 = decay to 10% of peak)
    
    Returns:
        LambdaLR scheduler
    """
    plateau_end = num_warmup_steps + int(plateau_ratio * num_training_steps)
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < plateau_end:
            # Plateau at peak LR
            return 1.0
        else:
            # Linear decay from plateau_end to end
            decay_steps = num_training_steps - plateau_end
            progress = float(current_step - plateau_end) / float(max(1, decay_steps))
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0
) -> optim.lr_scheduler.LambdaLR:
    """
    Create a cosine annealing scheduler with linear warmup.
    
    Industry-standard for transformer pretraining (GPT, BERT, etc.)
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Steps for linear warmup phase
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial (0.0 = decay to 0)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return max(min_lr_ratio, cosine_decay)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)




def create_scheduler(
    optimizer: optim.Optimizer,
    optimize_config: Optional[OptimizeConfig], 
    total_steps: int,
    scaled_lr: float,
    logger: Optional[logging.Logger] = None
) -> Tuple[optim.lr_scheduler._LRScheduler, str]:
    """
    Unified scheduler factory supporting multiple LR schedule types.
    
    Args:
        optimizer: The optimizer to schedule
        config: Config object (OptimizedConfig)
        total_steps: Total training steps
        scaled_lr: The peak learning rate (after multi-GPU scaling)
        logger: Optional logger for info messages
    
    Returns:
        Tuple of (scheduler, description_string)
    
    Supported scheduler_type values:
        - 'onecycle': OneCycleLR (default, best for 1-2 epochs)
        - 'linear': Linear warmup → plateau → linear decay
        - 'cosine': Linear warmup → cosine decay (original)
    """
    scheduler_type = getattr(optimize_config, 'scheduler_type', 'onecycle')
    warmup_pct = getattr(optimize_config, 'warmup_pct', 0.15)
    min_lr_ratio = getattr(optimize_config, 'min_lr_ratio', 0.01)
    
    if scheduler_type == 'onecycle':
        # OneCycleLR: ramp up → ramp down (best for single epoch)
        pct_start = getattr(optimize_config, 'onecycle_pct_start', 0.30)
        div_factor = getattr(optimize_config, 'onecycle_div_factor', 25)
        final_div = getattr(optimize_config, 'onecycle_final_div', 1000)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scaled_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=div_factor,
            final_div_factor=final_div,
            three_phase=False
        )
        
        start_lr = scaled_lr / div_factor
        end_lr = scaled_lr / final_div
        desc = f"OneCycleLR: {start_lr:.2e} → {scaled_lr:.2e} → {end_lr:.2e} (pct_start={pct_start})"
    
    elif scheduler_type == 'linear':
        # Linear warmup → plateau → linear decay
        plateau_ratio = getattr(optimize_config, 'plateau_pct', 0.30)  # Read from config
        warmup_steps = int(warmup_pct * total_steps)
        
        # Create scheduler using the helper function
        scheduler = get_linear_warmup_plateau_decay(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            plateau_ratio=plateau_ratio,
            min_lr_ratio=min_lr_ratio
        )
        
        end_lr = scaled_lr * min_lr_ratio
        plateau_end_pct = warmup_pct + plateau_ratio  # For description
        desc = (f"LinearPlateau: warmup={warmup_steps} steps ({warmup_pct*100:.0f}%), "
                f"plateau until {plateau_end_pct*100:.0f}%, "
                f"decay to {end_lr:.2e}")
    
    else:  # 'cosine' or default fallback
        # Cosine with warmup (original scheduler)
        if warmup_pct == 0.0:
            warmup_steps = 0  # True legacy mode: no warmup
        else:
            warmup_steps = max(100, min(2000, int(warmup_pct * total_steps)))
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )
        
        end_lr = scaled_lr * min_lr_ratio
        desc = f"CosineWarmup: warmup={warmup_steps}, peak={scaled_lr:.2e}, end={end_lr:.2e}"
    
    if logger:
        logger.info(f"Scheduler: {desc}")
    
    return scheduler, desc




# --- Optimizer ---

def create_optimizer(
    model: nn.Module,
    base_config: 'BaseConfig',
    optimize_config: Optional['OptimizeConfig'],
    scaled_lr: float,
    logger: Optional[logging.Logger] = None
) -> Tuple[optim.Optimizer, str]:
    """
    Unified optimizer factory supporting AdamW and SGD.
    
    Args:
        model: The model to optimize
        base_config: BaseConfig with default weight_decay
        optimize_config: OptimizeConfig with optimizer settings
        scaled_lr: Learning rate (potentially scaled for multi-GPU)
        logger: Optional logger for info messages
    
    Returns:
        Tuple of (optimizer, description_string)
    
    Supported optimizer_type values:
        - 'adamw': AdamW (default, standard for transformer training)
        - 'sgd': SGD with momentum (legacy compatibility, good for baselines)
    """
    # Determine optimizer type (default to AdamW)
    optimizer_type = getattr(optimize_config, 'optimizer_type', 'adamw').lower()
    
    # Determine learning rate (allow override)
    override_lr = getattr(optimize_config, 'override_lr', None)
    lr = override_lr if override_lr is not None else scaled_lr
    
    # Determine weight decay (allow override)
    override_wd = getattr(optimize_config, 'override_weight_decay', None)
    weight_decay = override_wd if override_wd is not None else base_config.weight_decay
    
    if optimizer_type == 'sgd':
        # SGD with momentum
        momentum = getattr(optimize_config, 'sgd_momentum', 0.9)
        nesterov = getattr(optimize_config, 'sgd_nesterov', False)
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        desc = (f"SGD: lr={lr:.2e}, momentum={momentum}, "
                f"weight_decay={weight_decay}, nesterov={nesterov}")
        
        if logger:
            logger.info(f"Optimizer: {desc}")
            
    else:  # 'adamw' (default)
        # AdamW - default for transformer training
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        desc = f"AdamW: lr={lr:.2e}, weight_decay={weight_decay}"
        
        if logger:
            logger.info(f"Optimizer: {desc}")
    
    return optimizer, desc


# #### Gradient tier inspection


# ============================================================
# GRADIENT TIER ANALYSIS
# Follows same pattern as MoE metrics and router gradient logging


# --- Gradient tier analysis ---

class GradientTierAnalyzer:
    """
    Analyzes gradient contribution per code frequency tier.
    
    Purpose: Diagnose if rare/tail codes receive insufficient gradient signal
    
    Follows the same pattern as MoE metrics and router gradient tracking:
    - log_batch() returns metrics dict (added to batch_entry)
    - aggregate_epoch() returns summary (added to epoch_metrics)
    - get_summary_for_results() returns data for final_results.json
    
    Usage in train_epoch:
        analyzer = GradientTierAnalyzer(code_frequencies, device)
        # After backward():
        tier_metrics = analyzer.log_batch(model, batch_idx)
        batch_entry.update(tier_metrics)  # Goes to batch_metrics.json
        # At epoch end:
        epoch_metrics.update(analyzer.aggregate_epoch())
    """
    
    def __init__(
        self,
        code_frequencies: np.ndarray,
        device: torch.device,
        log_interval: int = 500
    ):
        self.device = device
        self.log_interval = log_interval
        self.num_codes = len(code_frequencies)
        
        # Build tier indices (same logic as compute_stratified_metrics)
        freq_nz = code_frequencies[code_frequencies > 0]
        if len(freq_nz) == 0:
            raise ValueError("No non-zero frequencies found")
        
        percentiles = np.percentile(freq_nz, [20, 50, 80])
        
        # Create tier masks
        self.tier_indices = {}
        self.tier_sizes = {}
        
        # Common: above 80th percentile
        common_mask = code_frequencies > percentiles[2]
        self.tier_indices['common'] = torch.tensor(
            np.where(common_mask)[0], dtype=torch.long
        )
        self.tier_sizes['common'] = int(common_mask.sum())
        
        # Medium: 50th to 80th percentile
        medium_mask = (code_frequencies <= percentiles[2]) & (code_frequencies > percentiles[1])
        self.tier_indices['medium'] = torch.tensor(
            np.where(medium_mask)[0], dtype=torch.long
        )
        self.tier_sizes['medium'] = int(medium_mask.sum())
        
        # Rare: 20th to 50th percentile
        rare_mask = (code_frequencies <= percentiles[1]) & (code_frequencies > percentiles[0])
        self.tier_indices['rare'] = torch.tensor(
            np.where(rare_mask)[0], dtype=torch.long
        )
        self.tier_sizes['rare'] = int(rare_mask.sum())
        
        # Tail: below 20th percentile (but > 0)
        tail_mask = (code_frequencies <= percentiles[0]) & (code_frequencies > 0)
        self.tier_indices['tail'] = torch.tensor(
            np.where(tail_mask)[0], dtype=torch.long
        )
        self.tier_sizes['tail'] = int(tail_mask.sum())
        
        # Buffer for epoch aggregation (same pattern as moe_metrics_buffer)
        self.batch_buffer = []
        self.epoch_summaries = []
        
        print(f"  GradientTierAnalyzer initialized:")
        print(f"    Common: {self.tier_sizes['common']} codes")
        print(f"    Medium: {self.tier_sizes['medium']} codes")
        print(f"    Rare:   {self.tier_sizes['rare']} codes")
        print(f"    Tail:   {self.tier_sizes['tail']} codes")
    
    def _get_decoder_gradients(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Extract decoder_cd.weight gradients, handling DataParallel wrapping."""
        actual_model = model
        
        # Unwrap DataParallel
        if isinstance(model, nn.DataParallel):
            actual_model = model.module
        # Unwrap DataParallelWrapper
        if hasattr(actual_model, 'model'):
            actual_model = actual_model.model
        
        # Find decoder_cd
        decoder = None
        if hasattr(actual_model, 'decoder_cd'):
            decoder = actual_model.decoder_cd
        else:
            for name, module in actual_model.named_modules():
                if 'decoder_cd' in name and isinstance(module, nn.Linear):
                    decoder = module
                    break
        
        if decoder is None or decoder.weight.grad is None:
            return None
        
        return decoder.weight.grad.detach()
    
    def log_batch(
        self,
        model: nn.Module,
        batch_idx: int
    ) -> Dict[str, float]:
        """
        Compute and return gradient tier metrics for this batch.
        
        Call AFTER backward() but BEFORE optimizer.step().
        Returns dict with keys like 'grad_tier_common_frac', etc.
        These can be added to batch_entry for batch_metrics.json.
        
        Returns empty dict if not at log interval or no gradients available.
        """
        if batch_idx % self.log_interval != 0:
            return {}
        
        grad = self._get_decoder_gradients(model)
        if grad is None:
            return {}
        
        # Move to CPU for computation
        grad_cpu = grad.cpu()
        
        # Per-code gradient norms: [num_codes]
        per_code_norm = torch.norm(grad_cpu, dim=1)
        total_norm = per_code_norm.sum().item()
        
        if total_norm < 1e-12:
            return {}
        
        # Compute per-tier statistics
        metrics = {}
        for tier_name, indices in self.tier_indices.items():
            if len(indices) == 0:
                metrics[f'grad_tier_{tier_name}_frac'] = 0.0
                metrics[f'grad_tier_{tier_name}_norm'] = 0.0
                continue
            
            tier_norms = per_code_norm[indices]
            tier_total = tier_norms.sum().item()
            
            metrics[f'grad_tier_{tier_name}_frac'] = tier_total / total_norm
            metrics[f'grad_tier_{tier_name}_norm'] = tier_norms.mean().item()
        
        metrics['grad_tier_total_norm'] = total_norm
        
        # Add to buffer for epoch aggregation
        self.batch_buffer.append(metrics)
        
        return metrics
    
    def aggregate_epoch(self) -> Dict[str, float]:
        """
        Aggregate batch metrics into epoch-level summary.
        Call at end of epoch, returns dict for epoch_metrics.
        Same pattern as MoE/router gradient aggregation.
        """
        if not self.batch_buffer:
            return {}
        
        epoch_summary = {
            'train_grad_tier_samples': len(self.batch_buffer)
        }
        
        # Aggregate all numeric metrics
        for key in self.batch_buffer[0].keys():
            values = [m[key] for m in self.batch_buffer]
            epoch_summary[f'train_{key}'] = np.mean(values)
            epoch_summary[f'train_{key}_std'] = np.std(values)
        
        # Store for final results
        self.epoch_summaries.append(epoch_summary)
        
        return epoch_summary
    
    def get_diagnosis(self) -> Dict[str, Any]:
        """
        Generate diagnosis dict for comprehensive_evaluation results.
        """
        if not self.epoch_summaries:
            return {}
        
        latest = self.epoch_summaries[-1]
        common_frac = latest.get('train_grad_tier_common_frac', 0.0)
        tail_frac = latest.get('train_grad_tier_tail_frac', 0.0)
        
        diagnosis = {
            'tier_sizes': self.tier_sizes,
            'final_epoch_summary': latest,
            'all_epoch_summaries': self.epoch_summaries,
            'gradient_imbalance_ratio': common_frac / max(tail_frac, 1e-8),
            'diagnosis': 'balanced'
        }
        
        if common_frac > 0.8:
            diagnosis['diagnosis'] = 'severe_starvation'
            diagnosis['recommendation'] = 'Increase pos_weight_max significantly (200-500)'
        elif common_frac > 0.6:
            diagnosis['diagnosis'] = 'moderate_imbalance'
            diagnosis['recommendation'] = 'Consider increasing pos_weight_max or using focal loss'
        else:
            diagnosis['diagnosis'] = 'balanced'
            diagnosis['recommendation'] = 'Gradient distribution is healthy'
        
        return diagnosis
    
    def reset_epoch(self):
        """Clear batch buffer for new epoch."""
        self.batch_buffer = []
    
    def print_summary(self, epoch: int = 0, logger: Optional[logging.Logger] = None):
        """Print formatted summary (same pattern as MoE health logging)."""
        if not self.batch_buffer:
            return
        
        # Compute current averages
        common_frac = np.mean([m.get('grad_tier_common_frac', 0) for m in self.batch_buffer])
        medium_frac = np.mean([m.get('grad_tier_medium_frac', 0) for m in self.batch_buffer])
        rare_frac = np.mean([m.get('grad_tier_rare_frac', 0) for m in self.batch_buffer])
        tail_frac = np.mean([m.get('grad_tier_tail_frac', 0) for m in self.batch_buffer])

        zero_frac = 1.0 - (common_frac + medium_frac + rare_frac + tail_frac)
        zero_codes = self.num_codes - sum(self.tier_sizes.values())        
        
        summary_msg = (
            f"\n  📊 GRADIENT TIER ANALYSIS (Epoch {epoch + 1})\n"
            f"  {'─' * 60}\n"
            f"  {'Tier':<12} {'Codes':>8} {'Gradient Fraction':>20}\n"
            f"  {'─' * 60}\n"
            f"  {'Common':<12} {self.tier_sizes['common']:>8} {common_frac * 100:>19.1f}%\n"
            f"  {'Medium':<12} {self.tier_sizes['medium']:>8} {medium_frac * 100:>19.1f}%\n"
            f"  {'Rare':<12} {self.tier_sizes['rare']:>8} {rare_frac * 100:>19.1f}%\n"
            f"  {'Tail':<12} {self.tier_sizes['tail']:>8} {tail_frac * 100:>19.1f}%\n"
            f"  {'Zero-freq':<12} {zero_codes:>8} {zero_frac * 100:>19.1f}%\n"
            f"  {'─' * 60}\n"
            f"  {'TOTAL':<12} {self.num_codes:>8} {'100.0%':>20}\n"
        )
        
        # Diagnosis
        positive_frac = common_frac + medium_frac + rare_frac + tail_frac
        common_of_positive = common_frac / max(positive_frac, 1e-8)

        if zero_frac > 0.5:
            summary_msg += f"\n  ⚠️  WARNING: {zero_frac*100:.1f}% of gradient goes to ZERO-FREQUENCY codes\n"
            summary_msg += f"       These are codes that never appear in training - consider reducing target vocabulary\n"

        if common_of_positive > 0.6:
            summary_msg += f"  ⚠️  Among POSITIVE codes: Common receives {common_of_positive*100:.1f}% of positive-tier gradient\n"
            summary_msg += f"       Tail receives only {tail_frac/max(positive_frac, 1e-8)*100:.1f}% - gradient starvation likely\n"
        elif common_of_positive > 0.4:
            summary_msg += f"  ⚡ MODERATE IMBALANCE among positive codes\n"
        else:
            summary_msg += f"  ✅ Gradient distribution among positive codes appears balanced\n"

        print(summary_msg)
        if logger:
            logger.info(summary_msg)



# --- Training loop ---

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_mixed_precision: bool = False,
    moe_config: Optional[MoEConfig] = None,
    epoch: int = 1,
    use_bucketing: bool = False,
    log_interval: int = 500,
    global_step: int = 0, 
    loss_tracker: Optional[LossTracker] = None,
    is_main: bool = True,
    use_ddp: bool = False,
    accumulation_steps: int = 1,
    track_gpu_memory: bool = True,
    metrics_logger: Optional['MetricsLogger'] = None,  # Batch-metrics logger
    logger: Optional[logging.Logger] = None,           # General training logger
    optimize_config: Optional['OptimizeConfig'] = None,
    gradient_tier_analyzer: Optional['GradientTierAnalyzer'] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Clean design:
    - Step 1: Build batch list (either bucketed or sequential)
    - Step 2: Iterate over batch list uniformly
    - Step 3: Dynamic truncation for bucketed batches
    
    Logs metrics every `log_interval` batches:
    0. Loss (BCE + aux loss if MoE)
    1. Recall@5, 10, 20, 50 - Clinical utility at different cutoffs
    2. Precision@5, 10, 20, 50 - How many predictions are correct
    3. Micro-Recall@10, 20 - Per-code coverage rate
    4. NDCG@20 - Ranking quality with position discounting
    5. Positive-Only Brier - Calibration on positive labels
    6. MoE health (if applicable)
    
    # Track the training procedure with global_step: int = 0,
    
    """
    model.train()
    gpu_tracker = GPUMemoryTracker(enabled=track_gpu_memory and is_main)
    nbatch = len(dataloader)
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    batch_metrics_buffer = []  
    moe_metrics_buffer = []
    router_grad_metrics_buffer = [] # track routing gradients stability
    gradient_tier_buffer = []
    
    if loss_tracker is None:
        loss_tracker = LossTracker()    
        
    # Track accumulated loss for proper averaging
    accumulated_loss = 0.0
    accumulation_counter = 0
    
    # ============================================================
    # STEP 2: ITERATE OVER BATCHES (UNIFORM LOGIC)
    # ============================================================
    for batch_idx, batch in enumerate(dataloader):

        should_track = (
            track_gpu_memory and 
            is_main and 
            batch_idx in [2, 50, 100]
        )
        if should_track:
            gpu_tracker.reset_peak()
            print(f"\n🔍 Detailed GPU tracking for batch {batch_idx}")
            
        # Only main process prints progress
        if is_main and batch_idx % log_interval == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}')
            
        # Middle check multi-GPU works
        if batch_idx == 0 and is_main:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"\n🔍 GPU UTILIZATION CHECK (Batch 0):")
                for gpu_id in range(num_gpus):
                    mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"   GPU {gpu_id}: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")        
        
        if accumulation_counter == 0:
            optimizer.zero_grad(set_to_none=True) 
        
        # Get batch data
        # Extract tensors, DataParallel will handle device placement during scatter
        age = batch['age']
        gender = batch['gender']
        lob = batch['lob']
        codes = batch['codes']
        dt_cnt = batch['dt_cnt']
        targets_mh = batch['target_multihot']  # Pre-computed multi-hot     
        y = batch['target']
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        # Move to cuda and scatter data to different GPUs
        x = x.cuda(non_blocking=True)
        dt_cnt = dt_cnt.cuda(non_blocking=True)
        targets_mh = targets_mh.cuda(non_blocking=True)

        if should_track:
            gpu_tracker.record("1_after_data_to_gpu")
        
        # ============================================================
        # STEP 4: FORWARD PASS
        # ============================================================
        total_loss = torch.tensor(0.0, device=device)
        need_predictions = is_main and (batch_idx % log_interval == 0)
        if use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Model forward
                result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
        else:
            # Standard precision (baseline)
            result = model(x, dt_cnt, targets_mh, return_predictions=need_predictions)
            
        if should_track:
            gpu_tracker.record("2_after_forward")            
            
        if isinstance(result, tuple):
            total_loss, extras = result
            output = extras.get('predictions', None) if need_predictions else None
            moe_losses = extras.get('moe_losses', {})
            pred_loss = extras.get('pred_loss', total_loss)
            aux_loss = extras.get('aux_loss', torch.tensor(0.0))
        else:
            total_loss = result
            pred_loss = total_loss
            aux_loss = torch.tensor(0.0, device=device)
            output = None
            moe_losses = {}

        # Handle DataParallel multi-element tensors
        if total_loss.numel() > 1:
            total_loss = total_loss.mean()
        if pred_loss.numel() > 1:
            pred_loss = pred_loss.mean()
        if aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        # GRADIENT ACCUMULATION: Scale loss by accumulation steps
        scaled_loss = total_loss / accumulation_steps                
        # ============================================================
        # STEP 5: BACKWARD PASS
        # ============================================================
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        if should_track:
            gpu_tracker.record("3_after_backward")

        # ============================================================
        # STEP 5.1 GRADIENT TIER ANALYSIS (after backward, before optimizer step)
        # Same pattern as router gradient monitoring
        # ============================================================
        if gradient_tier_analyzer is not None and is_main and batch_idx % log_interval == 0:
            tier_metrics = gradient_tier_analyzer.log_batch(model, batch_idx)
            if tier_metrics:
                gradient_tier_buffer.append(tier_metrics)
                # Keep buffer bounded (same as router_grad_metrics_buffer)
                if len(gradient_tier_buffer) > 100:
                    gradient_tier_buffer = gradient_tier_buffer[-100:]
            
            
        # ============================================================
        # STEP 5.2 ROUTER GRADIENT MONITORING (before optimizer.step)
        # ============================================================
        if is_main and moe_config is not None and batch_idx % log_interval == 0:
            router_grad_metrics = compute_router_gradient_metrics(
                model, moe_config, log_all_layers=False
            )
            if router_grad_metrics:
                router_grad_metrics_buffer.append(router_grad_metrics)
                
                # Keep buffer bounded
                if len(router_grad_metrics_buffer) > 100:
                    router_grad_metrics_buffer = router_grad_metrics_buffer[-100:]
            
        # ============================================================
        # STEP 6: Optimization
        # ============================================================
        # Resolve gradient clip value (override takes precedence)
        gradient_clip = config.gradient_clip  # default from BaseConfig
        if optimize_config is not None:
            override_clip = getattr(optimize_config, 'override_gradient_clip', None)
            if override_clip is not None:
                gradient_clip = override_clip      
                
        # Track accumulated loss
        accumulated_loss += total_loss.detach()
        accumulation_counter += 1        
        
        if accumulation_counter >= accumulation_steps:
            if use_mixed_precision:
                # 1. Unscale gradients for clipping
                scaler.unscale_(optimizer)
                # 2. Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                # 3. Optimizer step (skips if gradients are inf/nan)
                scaler.step(optimizer)
                # 4. Update scaler for next iteration
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        
            if scheduler is not None:
                scheduler.step()
            # Reset accumulation
            accumulated_loss = 0.0
            accumulation_counter = 0
            global_step += 1

            # UPDATE MOE ROUTER WARMUP          
            if moe_config is not None and moe_config.router_warmup_steps > 0:
                # Get the actual model (unwrap DataParallel/DataParallelWrapper)
                actual_model = model
                if isinstance(model, nn.DataParallel):
                    actual_model = model.module
                if isinstance(actual_model, DataParallelWrapper):
                    actual_model = actual_model.model
                
                # Update warmup step if model has the method
                if hasattr(actual_model, 'set_moe_warmup_step'):
                    actual_model.set_moe_warmup_step(global_step)   
                    
        if should_track:
            gpu_tracker.print_gpu_use_summary()    
            
        # ============================================================
        # STEP 7: CLEANUP & LOGGING
        # ============================================================

        # Track losses - handle DataParallel multi-GPU tensors
        # extract scalars and detach to prevent graph retention
        # deactivate the graph
        with torch.no_grad():
            pred_loss_scalar = pred_loss.detach().mean().item() if pred_loss.numel() > 1 else pred_loss.detach().item()
            aux_loss_scalar = aux_loss.detach().mean().item() if aux_loss.numel() > 1 else aux_loss.detach().item()
        
        total_pred_loss += pred_loss_scalar
        total_aux_loss += aux_loss_scalar
        loss_tracker.log_batch(pred_loss_scalar, global_step)
        
        # ========================================================================
        # STEP 6a: COMPUTE & LOG REAL-TIME METRICS (every log_interval batches)
        # ========================================================================        
        if is_main and batch_idx % log_interval == 0:
            with torch.no_grad():
                output_detached = output.detach() if output is not None else None
                # Compute batch metrics (FAST)
                batch_metrics = compute_batch_metrics_lightweight(
                    output_detached, y, 
                    # convert back to list from tensor
                    dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt, 
                    config, device
                )
                # Keep only recent metrics (prevent unbounded growth)
                batch_metrics_buffer.append(batch_metrics)
                if len(batch_metrics_buffer) > 100:
                    batch_metrics_buffer = batch_metrics_buffer[-100:]

                batch_log_msg = (
                    f"    Loss: {pred_loss_scalar:.4f} | "
                    f"R@10: {batch_metrics['recall@10']:.3f} | "
                    f"R@20: {batch_metrics['recall@20']:.3f} | "
                    f"μR@10: {batch_metrics['micro_recall@10']:.3f} | "
                    f"P@10: {batch_metrics['precision@10']:.3f} | "
                    f"NDCG@20: {batch_metrics['ndcg@20']:.3f} | "
                    f"PosBrier: {batch_metrics['positive_brier']:.4f}"
                )                
                print(batch_log_msg)

                # General training logger (training.log)
                if logger:
                    logger.debug(batch_log_msg)

                # Collect batch metrics for batch_metrics.json
                batch_entry = {
                    'global_step': global_step,
                    'loss': pred_loss_scalar,
                    **batch_metrics  # recall@10, precision@10, etc.
                }
                
                # MoE metrics if applicable
                if moe_losses and 'expert_usage' in moe_losses:
                    moe_losses_detached = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v 
                        for k, v in moe_losses.items()
                    }
                    moe_batch_metrics = compute_moe_batch_metrics(moe_losses_detached)
                    moe_metrics_buffer.append(moe_batch_metrics)
                    
                    if router_grad_metrics_buffer:
                        latest_router = router_grad_metrics_buffer[-1]
                    else:
                        latest_router = {}  # Default empty dict

                    moe_log_msg = (
                        f"    MoE: CV={moe_batch_metrics['expert_load_cv']:.3f} | "
                        f"Collapsed={moe_batch_metrics['num_collapsed_experts']} | "
                        f"Gini={moe_batch_metrics['expert_gini']:.3f}"
                    )
                    if latest_router:
                        moe_log_msg += (
                            f" | Router: GradNorm={latest_router.get('router_grad_norm_mean', 0):.4f} | "
                            f"WeightStd={latest_router.get('router_weight_std', 0):.4f}"
                        )
                        if latest_router.get('router_grad_exploding', 0):
                            moe_log_msg += (
                                f" ⚠️ Router gradients EXPLODING! Consider reducing LR or adding clipping"
                            )
                        if latest_router.get('router_grad_vanishing', 0):
                            moe_log_msg += (
                                f" ⚠️ Router gradients VANISHING! Check focal loss or initialization"
                            )
                            
                    print(moe_log_msg)
                        
                    if logger:
                        logger.debug(moe_log_msg)

                    # Add MoE metrics to batch_metrics.json
                    batch_entry.update({
                        'moe_cv': moe_batch_metrics['expert_load_cv'],
                        'moe_collapsed': moe_batch_metrics['num_collapsed_experts'],
                        'moe_gini': moe_batch_metrics['expert_gini'],
                        'router_gradnorm_mean': latest_router.get('router_grad_norm_mean', 0),
                        'router_weight_std': latest_router.get('router_weight_std', 0),
                        'router_grad_exploding': latest_router.get('router_grad_exploding', 0),
                        'router_grad_vanishing': latest_router.get('router_grad_vanishing', 0)
                    })
                        
                    # WARNING if experts collapsing
                    if moe_batch_metrics['num_collapsed_experts'] > 0:
                        print(f" {moe_batch_metrics['num_collapsed_experts']} experts collapsed!")
                    del moe_losses_detached
                    
                # Gradient tier metrics (if available)
                if gradient_tier_buffer:
                    latest_tier = gradient_tier_buffer[-1]
                    batch_entry.update({
                        'grad_tier_common_frac': latest_tier.get('grad_tier_common_frac', 0.0),
                        'grad_tier_medium_frac': latest_tier.get('grad_tier_medium_frac', 0.0),
                        'grad_tier_rare_frac': latest_tier.get('grad_tier_rare_frac', 0.0),
                        'grad_tier_tail_frac': latest_tier.get('grad_tier_tail_frac', 0.0),
                        'grad_tier_total_norm': latest_tier.get('grad_tier_total_norm', 0.0),
                    })
                    
                    # Log to console (compact format, same pattern as MoE logging)
                    tier_log_msg = (
                        f"    [GradTier] Common: {latest_tier.get('grad_tier_common_frac', 0)*100:.1f}% | "
                        f"Tail: {latest_tier.get('grad_tier_tail_frac', 0)*100:.1f}%"
                    )
                    print(tier_log_msg)
                    if logger:
                        logger.debug(tier_log_msg)                    
                    
                    
                if metrics_logger:
                    metrics_logger.log_batch(epoch=epoch, batch=batch_idx, metrics=batch_entry)
                    
                del output_detached  # Clean up detached copy
                        
        # Memory cleanup (NO empty_cache in loop!)
        del x, targets_mh, dt_cnt
        if 'extras' in dir() and extras is not None:
            del extras
        if 'output' in dir() and output is not None:
            del output
        del pred_loss, aux_loss, total_loss, scaled_loss, result
        
        if batch_idx % 100 == 0:
            gc.collect()  # Python GC only
            
            # Optional memory monitoring
            if is_main and device.type == 'cuda' and batch_idx % 1000 == 0:
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
                    print(f'    GPU {gpu_id}: {allocated:.2f}GB / {peak:.2f}GB peak')

                    
                    
    # Handle remaining gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        global_step += 1
                    
    # End-of-epoch cleanup
    if device.type == 'cuda':
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    # ========================================================================
    # AGGREGATE EPOCH METRICS
    # ========================================================================
    loss_summary = loss_tracker.get_epoch_summary()
    epoch_metrics = {
        'train_loss': total_pred_loss / nbatch,
        **loss_summary, 
        'aux_loss': total_aux_loss / nbatch
    }
    
    # Add averaged batch metrics
    if batch_metrics_buffer:
        for key in batch_metrics_buffer[0].keys():
            epoch_metrics[f'train_{key}'] = np.mean([m[key] for m in batch_metrics_buffer])
    
    # Add averaged MoE metrics
    if moe_metrics_buffer:
        for key in moe_metrics_buffer[0].keys():
            epoch_metrics[f'train_{key}'] = np.mean([m[key] for m in moe_metrics_buffer])
        
        # Store final expert usage for comprehensive evaluation
        if 'expert_usage' in moe_losses:
            epoch_metrics['expert_usage'] = moe_losses['expert_usage']
    
    # Router gradients
    if router_grad_metrics_buffer:
        for key in router_grad_metrics_buffer[0].keys():
            if isinstance(router_grad_metrics_buffer[0][key], (int, float)):
                epoch_metrics[f'train_{key}'] = np.mean([m[key] for m in router_grad_metrics_buffer])
        
        # Store final gradient health for comprehensive evaluation
        epoch_metrics['router_grad_final_norm'] = router_grad_metrics_buffer[-1].get('router_grad_norm_mean', 0)
        epoch_metrics['router_grad_healthy_pct'] = np.mean([
            m.get('router_layers_healthy', 0) / max(m.get('router_layers_total', 1), 1) 
            for m in router_grad_metrics_buffer
        ])
    
    # Aggregate gradient tier metrics
    if gradient_tier_buffer:
        for key in gradient_tier_buffer[0].keys():
            if isinstance(gradient_tier_buffer[0][key], (int, float)):
                epoch_metrics[f'train_{key}'] = np.mean([m[key] for m in gradient_tier_buffer])
        
        # Store summary for comprehensive evaluation
        epoch_metrics['gradient_tier_common_frac_final'] = gradient_tier_buffer[-1].get('grad_tier_common_frac', 0)
        epoch_metrics['gradient_tier_tail_frac_final'] = gradient_tier_buffer[-1].get('grad_tier_tail_frac', 0)
    
    # Print gradient tier summary if analyzer provided
    if gradient_tier_analyzer is not None and is_main:
        gradient_tier_analyzer.print_summary(epoch, logger)
        
    # add global step 
    epoch_metrics['global_step'] = global_step
        
    return epoch_metrics


# #### Tier-aware batch sampler


from torch.utils.data import Sampler, Dataset, DataLoader


# --- Batch samplers ---

class TierAwareBatchSampler(Sampler):
    """
    Batch sampler that guarantees minimum representation of medium/rare/tail codes.
    
    Strategy (member-level sampling with code-tier awareness):
    1. Pre-compute which MEMBERS (samples) contain medium/rare/tail positive codes
    2. Each batch draws MEMBERS from tier-specific pools:
       - `medium_quota` members that have at least one medium-tier code
       - `rare_quota` members that have at least one rare-tier code
       - `tail_quota` members that have at least one tail-tier code
       - Remaining members from general pool
    
    Key insight: The training unit is the member, but we categorize members by the
    frequency tier of the codes they contain. A member with a tail code guarantees
    that tail code will receive gradient signal in that batch.
    
    This ensures consistent gradient signal for rare/tail codes EVERY batch,
    preventing the gradient concentration collapse observed in experiments.
    
    Compatible with:
    - BaselineTransformer (exp1)
    - FlashAttentionTransformer (exp2)
    - FlashMoETransformer (exp6)
    
    Usage:
        sampler = TierAwareBatchSampler(
            dataset=train_dataset,
            code_frequencies=prepared_data.code_frequencies,
            batch_size=64,
            medium_quota=4,  # At least 4 members with medium codes per batch
            rare_quota=8,    # At least 8 members with rare codes per batch
            tail_quota=10,   # At least 10 members with tail codes per batch
            shuffle=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=create_collate_fn(config),
            num_workers=4
        )
    """
    
    def __init__(
        self,
        dataset: Dataset,
        code_frequencies: np.ndarray,
        batch_size: int,
        medium_quota: int = 0,
        rare_quota: int = 4,
        tail_quota: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        percentile_boundaries: Tuple[float, float, float] = (20, 50, 80),
        verbose: bool = True,
        precomputed_tier_indices: Optional[dict] = None
    ):
        """
        Args:
            dataset: ClinicalDataset with targets
            code_frequencies: Array of code occurrence counts
            batch_size: Total batch size
            medium_quota: Minimum members with medium code positives per batch
            rare_quota: Minimum members with rare code positives per batch
            tail_quota: Minimum members with tail code positives per batch
            shuffle: Whether to shuffle within each pool
            drop_last: Whether to drop the last incomplete batch
            percentile_boundaries: (tail_thresh, rare_thresh, medium_thresh)
                                   E.g., (20, 50, 80) means:
                                   - Tail: freq <= 20th percentile
                                   - Rare: 20th < freq <= 50th percentile
                                   - Medium: 50th < freq <= 80th percentile
                                   - Common: freq > 80th percentile
            verbose: Print initialization statistics
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.medium_quota = medium_quota
        self.rare_quota = rare_quota
        self.tail_quota = tail_quota
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        
        # Validate quotas
        total_quota = medium_quota + rare_quota + tail_quota
        assert total_quota <= batch_size, \
            f"Combined quotas ({total_quota}) exceed batch_size ({batch_size})"
        
        if precomputed_tier_indices is not None:
            self.tier_code_indices = precomputed_tier_indices['tier_code_indices']
            self.tier_thresholds = precomputed_tier_indices['tier_thresholds']
            self.samples_with_medium = precomputed_tier_indices['samples_with_medium']
            self.samples_with_rare = precomputed_tier_indices['samples_with_rare']
            self.samples_with_tail = precomputed_tier_indices['samples_with_tail']
            self.general_samples = list(range(self.num_samples))
            if verbose:
                print(f"TierAwareBatchSampler: Using pre-computed tier indices")
                print(f"  Members with medium: {len(self.samples_with_medium):,}")
                print(f"  Members with rare: {len(self.samples_with_rare):,}")
                print(f"  Members with tail: {len(self.samples_with_tail):,}")
        else:
            self._build_tier_indices(code_frequencies, percentile_boundaries)
            self._build_sample_tier_mapping(verbose)
        
        self._calculate_num_batches()
    
    def _build_tier_indices(
        self,
        code_frequencies: np.ndarray,
        percentile_boundaries: Tuple[float, float, float]
    ):
        """Build tier code index sets based on frequency percentiles."""
        freq_nz = code_frequencies[code_frequencies > 0]
        if len(freq_nz) == 0:
            raise ValueError("No non-zero frequencies found in code_frequencies")
        
        percentiles = np.percentile(freq_nz, list(percentile_boundaries))
        self.tier_code_indices = {}
        
        # Common: above 80th percentile
        self.tier_code_indices['common'] = set(
            np.where(code_frequencies > percentiles[2])[0]
        )
        
        # Medium: 50th to 80th percentile
        self.tier_code_indices['medium'] = set(
            np.where((code_frequencies <= percentiles[2]) & 
                     (code_frequencies > percentiles[1]))[0]
        )
        
        # Rare: 20th to 50th percentile
        self.tier_code_indices['rare'] = set(
            np.where((code_frequencies <= percentiles[1]) & 
                     (code_frequencies > percentiles[0]))[0]
        )
        
        # Tail: below 20th percentile (but > 0)
        self.tier_code_indices['tail'] = set(
            np.where((code_frequencies <= percentiles[0]) & 
                     (code_frequencies > 0))[0]
        )
        
        self.tier_thresholds = {
            'tail_upper': percentiles[0],
            'rare_upper': percentiles[1],
            'medium_upper': percentiles[2]
        }
    
    def _build_sample_tier_mapping(self, verbose: bool):
        """
        Pre-compute which members (samples) contain medium/rare/tail positive codes.
        
        This is done ONCE during initialization for efficiency.
        Optimized to avoid repeated dataset access for large datasets.
        """
        # Members that have at least one code from each tier
        self.samples_with_medium = []
        self.samples_with_rare = []
        self.samples_with_tail = []
        # General pool includes ALL samples (overlaps are OK - handled in __iter__)
        self.general_samples = list(range(self.num_samples))
        
        medium_codes = self.tier_code_indices['medium']
        rare_codes = self.tier_code_indices['rare']
        tail_codes = self.tier_code_indices['tail']
        
        if verbose:
            print(f"TierAwareBatchSampler: Building member-tier mapping for {self.num_samples:,} members...")
            print(f"  Tier code counts: medium={len(medium_codes)}, rare={len(rare_codes)}, tail={len(tail_codes)}")
        
        # Access targets directly from dataset for efficiency
        # ClinicalDataset stores targets as self.targets: List[List[List[int]]]
        targets_list = self.dataset.targets
        
        for idx in range(self.num_samples):
            if verbose and idx > 0 and idx % 500000 == 0:
                print(f"    Processed {idx:,}/{self.num_samples:,} members...")
            
            # Get target codes for this member
            # targets is List[List[int]] where each inner list is codes for one day
            target_list = targets_list[idx]
            
            # Flatten all positive codes for this member
            all_positive_codes = set()
            for day_codes in target_list:
                if day_codes:  # Non-empty day
                    all_positive_codes.update(day_codes)
            
            # Check tier membership - a member can be in multiple tier pools
            if all_positive_codes & medium_codes:
                self.samples_with_medium.append(idx)
            if all_positive_codes & rare_codes:
                self.samples_with_rare.append(idx)
            if all_positive_codes & tail_codes:
                self.samples_with_tail.append(idx)
        
        if verbose:
            print(f"  ✅ Members with medium codes: {len(self.samples_with_medium):,} "
                  f"({len(self.samples_with_medium)/self.num_samples:.1%})")
            print(f"  ✅ Members with rare codes: {len(self.samples_with_rare):,} "
                  f"({len(self.samples_with_rare)/self.num_samples:.1%})")
            print(f"  ✅ Members with tail codes: {len(self.samples_with_tail):,} "
                  f"({len(self.samples_with_tail)/self.num_samples:.1%})")
            
            # Warn if quotas may not be satisfiable
            if self.medium_quota > 0 and len(self.samples_with_medium) < self.medium_quota * 10:
                print(f"  ⚠️ Warning: Few members with medium codes. May need to reduce medium_quota.")
            if len(self.samples_with_rare) < self.rare_quota * 10:
                print(f"  ⚠️ Warning: Few members with rare codes. May need to reduce rare_quota.")
            if len(self.samples_with_tail) < self.tail_quota * 10:
                print(f"  ⚠️ Warning: Few members with tail codes. May need to reduce tail_quota.")
    
    def _calculate_num_batches(self):
        """Calculate number of batches per epoch."""
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with guaranteed tier representation."""
        # Copy and optionally shuffle pools
        if self.shuffle:
            medium_pool = self.samples_with_medium.copy()
            rare_pool = self.samples_with_rare.copy()
            tail_pool = self.samples_with_tail.copy()
            general_pool = self.general_samples.copy()
            random.shuffle(medium_pool)
            random.shuffle(rare_pool)
            random.shuffle(tail_pool)
            random.shuffle(general_pool)
        else:
            medium_pool = self.samples_with_medium.copy()
            rare_pool = self.samples_with_rare.copy()
            tail_pool = self.samples_with_tail.copy()
            general_pool = self.general_samples.copy()
        
        # Track used samples to avoid duplicates within epoch
        used_samples = set()
        medium_idx = 0
        rare_idx = 0
        tail_idx = 0
        general_idx = 0
        
        batches_yielded = 0
        
        while batches_yielded < self.num_batches:
            batch = []
            
            # 1. Add medium quota (if > 0)
            if self.medium_quota > 0:
                medium_added = 0
                while medium_added < self.medium_quota and medium_idx < len(medium_pool):
                    sample_idx = medium_pool[medium_idx]
                    medium_idx += 1
                    if sample_idx not in used_samples:
                        batch.append(sample_idx)
                        used_samples.add(sample_idx)
                        medium_added += 1
            
            # 2. Add rare quota
            rare_added = 0
            while rare_added < self.rare_quota and rare_idx < len(rare_pool):
                sample_idx = rare_pool[rare_idx]
                rare_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    rare_added += 1
            
            # 3. Add tail quota
            tail_added = 0
            while tail_added < self.tail_quota and tail_idx < len(tail_pool):
                sample_idx = tail_pool[tail_idx]
                tail_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    tail_added += 1
            
            # 4. Fill remainder from general pool
            remaining = self.batch_size - len(batch)
            while remaining > 0 and general_idx < len(general_pool):
                sample_idx = general_pool[general_idx]
                general_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    remaining -= 1
            
            # Handle pool exhaustion - reset with reshuffling
            if self.medium_quota > 0 and medium_idx >= len(medium_pool):
                medium_pool = self.samples_with_medium.copy()
                if self.shuffle:
                    random.shuffle(medium_pool)
                medium_idx = 0
            
            if rare_idx >= len(rare_pool):
                rare_pool = self.samples_with_rare.copy()
                if self.shuffle:
                    random.shuffle(rare_pool)
                rare_idx = 0
            
            if tail_idx >= len(tail_pool):
                tail_pool = self.samples_with_tail.copy()
                if self.shuffle:
                    random.shuffle(tail_pool)
                tail_idx = 0
            
            if general_idx >= len(general_pool):
                general_pool = self.general_samples.copy()
                if self.shuffle:
                    random.shuffle(general_pool)
                general_idx = 0
                # Reset used_samples when general pool exhausted (epoch boundary)
                used_samples.clear()
            
            # Yield batch if it meets size requirements
            if len(batch) >= self.batch_size or (not self.drop_last and len(batch) > 0):
                if self.shuffle:
                    random.shuffle(batch)  # Shuffle within batch to avoid ordering bias
                yield batch[:self.batch_size]
                batches_yielded += 1
    
    def __len__(self) -> int:
        return self.num_batches



def create_tier_aware_dataloader(
    dataset: Dataset,
    code_frequencies: np.ndarray,
    config,  # BaseConfig or subclass
    medium_quota: int = 0,
    rare_quota: int = 4,
    tail_quota: int = 4,
    num_workers: Optional[int] = None,
    collate_fn = None,
    verbose: bool = True
) -> DataLoader:
    """
    Factory function to create a DataLoader with tier-aware batching.
    
    Drop-in replacement for standard DataLoader creation.
    
    Args:
        dataset: ClinicalDataset
        code_frequencies: From prepared_data.code_frequencies
        config: Model config with batch_size
        medium_quota: Min members with medium codes per batch (default 0)
        rare_quota: Min members with rare codes per batch
        tail_quota: Min members with tail codes per batch
        num_workers: Number of data loading workers (default: auto-detect)
        collate_fn: Custom collate function (create_collate_fn(config))
        verbose: Print initialization statistics
    
    Returns:
        DataLoader with tier-aware batching
    
    Usage:
        train_loader = create_tier_aware_dataloader(
            dataset=prepared_data.train_dataset,
            code_frequencies=prepared_data.code_frequencies,
            config=config,
            medium_quota=4,  # Include medium codes if desired
            rare_quota=8,    # For 3.4M model: aggressive rare quota
            tail_quota=10,   # For 3.4M model: aggressive tail quota
            collate_fn=create_collate_fn(config)
        )
    """
    # Import create_collate_fn if needed
    if collate_fn is None:
        try:
            from moe_flashattn_4_core import create_collate_fn
            collate_fn = create_collate_fn(config)
        except ImportError:
            raise ValueError("collate_fn must be provided if moe_flashattn_4_core is not available")
    
    # Auto-detect workers matching existing _create_dataloaders pattern
    if num_workers is None:
        num_workers = min(4, os.cpu_count() // 4) if os.cpu_count() else 2
    
    sampler = TierAwareBatchSampler(
        dataset=dataset,
        code_frequencies=code_frequencies,
        batch_size=config.batch_size,
        medium_quota=medium_quota,
        rare_quota=rare_quota,
        tail_quota=tail_quota,
        shuffle=True,
        drop_last=True,
        verbose=verbose
    )
    
    # Match existing DataLoader configuration from _create_dataloaders
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=False  # Match moe_flashattn_4.py pattern
    )
    
    if verbose:
        print(f"\n✅ Created tier-aware DataLoader:")
        print(f"   Batch size: {config.batch_size}")
        if medium_quota > 0:
            print(f"   Medium quota: {medium_quota} members/batch")
        print(f"   Rare quota: {rare_quota} members/batch")
        print(f"   Tail quota: {tail_quota} members/batch")
        print(f"   Total batches: {len(sampler):,}")
        print(f"   Workers: {num_workers}")
    
    return loader


# ============================================================
# INTEGRATION: Modified _create_dataloaders function
# ============================================================
# This function extends the existing _create_dataloaders from moe_flashattn_4.py
# to support tier-aware batching as an additional option.

def _create_dataloaders_with_tier_aware(
    train_data: Union[pd.DataFrame, Dataset],
    val_data: Union[pd.DataFrame, Dataset],
    config,  # BaseConfig or subclass
    code_frequencies: np.ndarray,
    use_tier_aware: bool = True,
    medium_quota: int = 0,
    rare_quota: int = 4,
    tail_quota: int = 4,
    use_bucketing: bool = False,  # Mutually exclusive with tier_aware
    train_data_df: Optional[pd.DataFrame] = None,
    world_size: int = 1,
    logger: Optional[logging.Logger] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with optional tier-aware batching.
    
    This extends the existing _create_dataloaders function from moe_flashattn_4.py
    to support tier-aware batching as an additional option.
    
    Args:
        train_data: Training dataset (ClinicalDataset or DataFrame)
        val_data: Validation dataset (ClinicalDataset or DataFrame)
        config: Model configuration with batch_size
        code_frequencies: Code frequency array for tier computation
        use_tier_aware: Whether to use tier-aware batching
        medium_quota: Min medium members per batch (if tier_aware)
        rare_quota: Min rare members per batch (if tier_aware)
        tail_quota: Min tail members per batch (if tier_aware)
        use_bucketing: Whether to use bucketing (mutually exclusive with tier_aware)
        train_data_df: Original DataFrame (required for bucketing if train_data is Dataset)
        world_size: Number of distributed processes (unused, for future DDP support)
        logger: Optional logger
    
    Returns:
        (train_loader, val_loader)
    """
    # Import dependencies - handle both standalone and integrated usage
    try:
        from moe_flashattn_4_core import ClinicalDataset, create_collate_fn
    except ImportError:
        # Fallback for integrated usage
        pass
    
    # Handle both Dataset and DataFrame inputs for backward compatibility
    def is_dataset(obj):
        """Check if object is a Dataset (duck-typing for Jupyter compatibility)."""
        return hasattr(obj, '__getitem__') and hasattr(obj, '__len__') and not isinstance(obj, pd.DataFrame)
    
    if is_dataset(train_data):
        train_dataset = train_data
    else:
        train_dataset = ClinicalDataset(train_data, config)
        train_data_df = train_data  # Save for bucketing
    
    if is_dataset(val_data):
        val_dataset = val_data
    else:
        val_dataset = ClinicalDataset(val_data, config)
    
    n_workers = min(4, os.cpu_count() // 4) if os.cpu_count() else 2
    collate_fn = create_collate_fn(config)
    
    # ========================================
    # TRAINING LOADER
    # ========================================
    if use_tier_aware and not use_bucketing:
        if logger:
            logger.info(f"Using TIER-AWARE batching (medium={medium_quota}, rare={rare_quota}, tail={tail_quota})")
        else:
            print(f"Using TIER-AWARE batching (medium={medium_quota}, rare={rare_quota}, tail={tail_quota})")
        
        train_loader = create_tier_aware_dataloader(
            dataset=train_dataset,
            code_frequencies=code_frequencies,
            config=config,
            medium_quota=medium_quota,
            rare_quota=rare_quota,
            tail_quota=tail_quota,
            num_workers=n_workers,
            collate_fn=collate_fn,
            verbose=True
        )
    
    elif use_bucketing and not use_tier_aware:
        if logger:
            logger.info("Using BUCKETING batch sampler")
        
        if train_data_df is None:
            raise ValueError("train_data_df is required when use_bucketing=True and train_data is a Dataset")
        
        # Import BucketingBatchSampler from the main module
        try:
            from moe_flashattn_4 import BucketingBatchSampler
        except ImportError:
            raise ImportError("BucketingBatchSampler not available. Use tier-aware batching or provide train_data_df.")
        
        train_batch_sampler = BucketingBatchSampler(
            data=train_data_df,
            batch_size=config.batch_size,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    else:
        if logger:
            logger.info("Using STANDARD DataLoader (no special batching)")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    # ========================================
    # VALIDATION LOADER (always standard)
    # ========================================
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if logger:
        logger.info(f"Train loader: {len(train_loader):,} batches")
        logger.info(f"Val loader: {len(val_loader):,} batches")
        logger.info(f"Using DataLoader with {n_workers} workers.")
    else:
        print(f"Train loader: {len(train_loader):,} batches")
        print(f"Val loader: {len(val_loader):,} batches")
    
    return train_loader, val_loader


# #### Dense tier aware batch sampler



class DensityTierAwareBatchSampler(Sampler):
    """
    Density-aware batch sampler that selects members with HIGH CONCENTRATION
    of rare/tail codes, not just binary presence.
    
    Key difference from TierAwareBatchSampler:
    - Old: tail_pool = members with >= 1 tail code (83.4% of members)
    - New: tail_pool = top 20% of members by tail code DENSITY
    
    Density = (number of tail code occurrences) / (total code occurrences)
    
    This ensures members selected for tail quota actually provide meaningful
    tail gradient signal (15-20%+ tail codes vs global 5.2%).
    
    Evidence basis: Jan 30 code frequency analysis showed 83.4% member-level 
    coverage but only 5.2% occurrence-level coverage for tail codes.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        code_frequencies: np.ndarray,
        batch_size: int,
        medium_quota: int = 0,
        rare_quota: int = 4,
        tail_quota: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
        percentile_boundaries: Tuple[float, float, float] = (20, 50, 80),
        density_tail_percentile: float = 80.0,
        density_rare_percentile: float = 70.0,
        density_medium_percentile: float = 70.0,
        verbose: bool = True,
        precomputed_density_pools: Optional[dict] = None
    ):
        """
        Args:
            dataset: ClinicalDataset with targets
            code_frequencies: Array of code occurrence counts
            batch_size: Total batch size
            medium_quota: Minimum high-density medium members per batch
            rare_quota: Minimum high-density rare members per batch
            tail_quota: Minimum high-density tail members per batch
            shuffle: Whether to shuffle within each pool
            drop_last: Whether to drop the last incomplete batch
            percentile_boundaries: (tail_thresh, rare_thresh, medium_thresh)
                for defining which CODES belong to which tier
            density_tail_percentile: Percentile threshold for tail member pool
                80.0 = only top 20% of members by tail density
            density_rare_percentile: Percentile threshold for rare member pool
            density_medium_percentile: Percentile threshold for medium member pool
            verbose: Print initialization statistics
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.medium_quota = medium_quota
        self.rare_quota = rare_quota
        self.tail_quota = tail_quota
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        self.density_tail_pct = density_tail_percentile
        self.density_rare_pct = density_rare_percentile
        self.density_medium_pct = density_medium_percentile
        
        total_quota = medium_quota + rare_quota + tail_quota
        assert total_quota <= batch_size, \
            f"Combined quotas ({total_quota}) exceed batch_size ({batch_size})"
        
        if precomputed_density_pools is not None:
            self.tier_code_indices = precomputed_density_pools['tier_code_indices']
            self.tier_thresholds = precomputed_density_pools['tier_thresholds']
            self.samples_with_medium = precomputed_density_pools['samples_with_medium']
            self.samples_with_rare = precomputed_density_pools['samples_with_rare']
            self.samples_with_tail = precomputed_density_pools['samples_with_tail']
            self.general_samples = list(range(self.num_samples))
            self._density_stats = precomputed_density_pools.get('density_stats', {})
            if verbose:
                print(f"DensityTierAwareBatchSampler: Using pre-computed density pools")
                print(f"  Tail pool: {len(self.samples_with_tail):,}")
                print(f"  Rare pool: {len(self.samples_with_rare):,}")
                print(f"  Medium pool: {len(self.samples_with_medium):,}")
        else:
            self._build_tier_indices(code_frequencies, percentile_boundaries)
            self._build_density_pools(verbose)
        
        self._calculate_num_batches()
    
    def _build_tier_indices(
        self,
        code_frequencies: np.ndarray,
        percentile_boundaries: Tuple[float, float, float]
    ):
        """Build tier code index sets based on frequency percentiles."""
        freq_nz = code_frequencies[code_frequencies > 0]
        if len(freq_nz) == 0:
            raise ValueError("No non-zero frequencies found in code_frequencies")
        
        percentiles = np.percentile(freq_nz, list(percentile_boundaries))
        self.tier_code_indices = {}
        
        self.tier_code_indices['common'] = set(
            np.where(code_frequencies > percentiles[2])[0]
        )
        self.tier_code_indices['medium'] = set(
            np.where((code_frequencies <= percentiles[2]) & 
                     (code_frequencies > percentiles[1]))[0]
        )
        self.tier_code_indices['rare'] = set(
            np.where((code_frequencies <= percentiles[1]) & 
                     (code_frequencies > percentiles[0]))[0]
        )
        self.tier_code_indices['tail'] = set(
            np.where((code_frequencies <= percentiles[0]) & 
                     (code_frequencies > 0))[0]
        )
        
        self.tier_thresholds = {
            'tail_upper': percentiles[0],
            'rare_upper': percentiles[1],
            'medium_upper': percentiles[2]
        }
    
    def _build_density_pools(self, verbose: bool):
        """
        Compute per-member density scores for each tier and build 
        high-density pools using percentile thresholds.
        
        Density = (occurrences of tier codes across all days) / (total occurrences)
        """
        medium_codes = self.tier_code_indices['medium']
        rare_codes = self.tier_code_indices['rare']
        tail_codes = self.tier_code_indices['tail']
        
        targets_list = self.dataset.targets
        
        tail_densities = np.zeros(self.num_samples, dtype=np.float32)
        rare_densities = np.zeros(self.num_samples, dtype=np.float32)
        medium_densities = np.zeros(self.num_samples, dtype=np.float32)
        
        tail_counts = np.zeros(self.num_samples, dtype=np.int32)
        rare_counts = np.zeros(self.num_samples, dtype=np.int32)
        medium_counts = np.zeros(self.num_samples, dtype=np.int32)
        total_counts = np.zeros(self.num_samples, dtype=np.int32)
        
        if verbose:
            print(f"DensityTierAwareBatchSampler: Computing density scores "
                  f"for {self.num_samples:,} members...")
        
        for idx in range(self.num_samples):
            if verbose and idx > 0 and idx % 500000 == 0:
                print(f"    Processed {idx:,}/{self.num_samples:,} members...")
            
            target_list = targets_list[idx]
            member_tail = 0
            member_rare = 0
            member_medium = 0
            member_total = 0
            
            for day_codes in target_list:
                if not day_codes:
                    continue
                for code in day_codes:
                    if code == 0:
                        continue
                    member_total += 1
                    if code in tail_codes:
                        member_tail += 1
                    elif code in rare_codes:
                        member_rare += 1
                    elif code in medium_codes:
                        member_medium += 1
            
            total_counts[idx] = member_total
            tail_counts[idx] = member_tail
            rare_counts[idx] = member_rare
            medium_counts[idx] = member_medium
            
            if member_total > 0:
                tail_densities[idx] = member_tail / member_total
                rare_densities[idx] = member_rare / member_total
                medium_densities[idx] = member_medium / member_total
        
        # Build pools using percentile thresholds on density scores
        # Only consider members with >0 tail occurrences for tail pool percentile
        tail_mask = tail_counts > 0
        rare_mask = rare_counts > 0
        medium_mask = medium_counts > 0
        
        # Compute percentile thresholds on members who HAVE codes from each tier
        if tail_mask.sum() > 0:
            tail_density_thresh = np.percentile(
                tail_densities[tail_mask], self.density_tail_pct
            )
        else:
            tail_density_thresh = 0.0
            
        if rare_mask.sum() > 0:
            rare_density_thresh = np.percentile(
                rare_densities[rare_mask], self.density_rare_pct
            )
        else:
            rare_density_thresh = 0.0
            
        if medium_mask.sum() > 0:
            medium_density_thresh = np.percentile(
                medium_densities[medium_mask], self.density_medium_pct
            )
        else:
            medium_density_thresh = 0.0
        
        # Select high-density members for each pool
        # Add the mask to ensure only members with actual tail codes qualify
        self.samples_with_tail = np.where(
            (tail_densities >= tail_density_thresh) & (tail_counts > 0)
        )[0].tolist()
        self.samples_with_rare = np.where(
            (rare_densities >= rare_density_thresh) & (rare_counts > 0)
        )[0].tolist()
        self.samples_with_medium = np.where(
            (medium_densities >= medium_density_thresh) & (medium_counts > 0)
        )[0].tolist()
        self.general_samples = list(range(self.num_samples))
        
        if verbose:
            print(f"\n  Density thresholds:")
            print(f"    Tail:   density >= {tail_density_thresh:.4f} "
                  f"(top {100-self.density_tail_pct:.0f}%)")
            print(f"    Rare:   density >= {rare_density_thresh:.4f} "
                  f"(top {100-self.density_rare_pct:.0f}%)")
            print(f"    Medium: density >= {medium_density_thresh:.4f} "
                  f"(top {100-self.density_medium_pct:.0f}%)")
            
            print(f"\n  High-density pools:")
            print(f"    Tail pool:   {len(self.samples_with_tail):,} members "
                  f"({len(self.samples_with_tail)/self.num_samples:.1%})")
            print(f"    Rare pool:   {len(self.samples_with_rare):,} members "
                  f"({len(self.samples_with_rare)/self.num_samples:.1%})")
            print(f"    Medium pool: {len(self.samples_with_medium):,} members "
                  f"({len(self.samples_with_medium)/self.num_samples:.1%})")
            
            # Show density statistics for selected pool
            if len(self.samples_with_tail) > 0:
                pool_tail_densities = tail_densities[self.samples_with_tail]
                pool_tail_counts = tail_counts[self.samples_with_tail]
                pool_total_counts = total_counts[self.samples_with_tail]
                print(f"\n  Tail pool statistics:")
                print(f"    Avg tail density:    {pool_tail_densities.mean():.4f} "
                      f"(vs global {tail_densities[tail_mask].mean():.4f})")
                print(f"    Avg tail occurrences: {pool_tail_counts.mean():.1f} "
                      f"(vs global {tail_counts[tail_mask].mean():.1f})")
                print(f"    Avg total codes:     {pool_total_counts.mean():.1f}")
                
                # Estimate per-batch tail occurrence improvement
                est_tail_per_batch = (
                    self.tail_quota * pool_tail_counts.mean() +
                    (self.batch_size - self.tail_quota) * tail_counts.mean()
                )
                est_total_per_batch = (
                    self.tail_quota * pool_total_counts.mean() +
                    (self.batch_size - self.tail_quota) * total_counts.mean()
                )
                est_tail_frac = est_tail_per_batch / max(est_total_per_batch, 1)
                print(f"    Estimated tail occurrence fraction per batch: "
                      f"{est_tail_frac:.2%} (target: >5%)")
        
        self._density_stats = {
            'tail_density_threshold': float(tail_density_thresh),
            'rare_density_threshold': float(rare_density_thresh),
            'medium_density_threshold': float(medium_density_thresh),
            'tail_pool_size': len(self.samples_with_tail),
            'rare_pool_size': len(self.samples_with_rare),
            'medium_pool_size': len(self.samples_with_medium),
            'tail_pool_avg_density': float(
                tail_densities[self.samples_with_tail].mean()
            ) if len(self.samples_with_tail) > 0 else 0.0,
        }
    
    def _calculate_num_batches(self):
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with guaranteed high-density tier representation."""
        if self.shuffle:
            medium_pool = self.samples_with_medium.copy()
            rare_pool = self.samples_with_rare.copy()
            tail_pool = self.samples_with_tail.copy()
            general_pool = self.general_samples.copy()
            random.shuffle(medium_pool)
            random.shuffle(rare_pool)
            random.shuffle(tail_pool)
            random.shuffle(general_pool)
        else:
            medium_pool = self.samples_with_medium.copy()
            rare_pool = self.samples_with_rare.copy()
            tail_pool = self.samples_with_tail.copy()
            general_pool = self.general_samples.copy()
        
        used_samples = set()
        medium_idx = 0
        rare_idx = 0
        tail_idx = 0
        general_idx = 0
        batches_yielded = 0
        
        while batches_yielded < self.num_batches:
            batch = []
            
            if self.medium_quota > 0:
                medium_added = 0
                while medium_added < self.medium_quota and medium_idx < len(medium_pool):
                    sample_idx = medium_pool[medium_idx]
                    medium_idx += 1
                    if sample_idx not in used_samples:
                        batch.append(sample_idx)
                        used_samples.add(sample_idx)
                        medium_added += 1
            
            rare_added = 0
            while rare_added < self.rare_quota and rare_idx < len(rare_pool):
                sample_idx = rare_pool[rare_idx]
                rare_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    rare_added += 1
            
            tail_added = 0
            while tail_added < self.tail_quota and tail_idx < len(tail_pool):
                sample_idx = tail_pool[tail_idx]
                tail_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    tail_added += 1
            
            remaining = self.batch_size - len(batch)
            while remaining > 0 and general_idx < len(general_pool):
                sample_idx = general_pool[general_idx]
                general_idx += 1
                if sample_idx not in used_samples:
                    batch.append(sample_idx)
                    used_samples.add(sample_idx)
                    remaining -= 1
            
            # Pool exhaustion handling with reshuffle
            if self.medium_quota > 0 and medium_idx >= len(medium_pool):
                medium_pool = self.samples_with_medium.copy()
                if self.shuffle:
                    random.shuffle(medium_pool)
                medium_idx = 0
            
            if rare_idx >= len(rare_pool):
                rare_pool = self.samples_with_rare.copy()
                if self.shuffle:
                    random.shuffle(rare_pool)
                rare_idx = 0
            
            if tail_idx >= len(tail_pool):
                tail_pool = self.samples_with_tail.copy()
                if self.shuffle:
                    random.shuffle(tail_pool)
                tail_idx = 0
            
            if general_idx >= len(general_pool):
                general_pool = self.general_samples.copy()
                if self.shuffle:
                    random.shuffle(general_pool)
                general_idx = 0
                used_samples.clear()
            
            if len(batch) >= self.batch_size or (not self.drop_last and len(batch) > 0):
                if self.shuffle:
                    random.shuffle(batch)
                yield batch[:self.batch_size]
                batches_yielded += 1
    
    def __len__(self) -> int:
        return self.num_batches
    
    def get_density_stats(self) -> dict:
        """Return density statistics for logging/analysis."""
        return self._density_stats


# #### Bucketing batch sampler



class BucketingBatchSampler:
    """
    Batch sampler that groups samples by similar sequence length.
    
    Strategy:
    1. Sort samples by dt_cnt (actual days)
    2. Create buckets of similar lengths
    3. Sample batches within buckets
    4. Shuffle buckets between epochs
    
    Benefits:
    - Reduces padding waste by ~50%
    - Flash Attention only computes over actual length
    - Compatible with existing code (just changes batch composition)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        bucket_boundaries: List[int] = [50, 100, 150, 200],
        shuffle: bool = True,
        drop_last: bool = True
    ):
        self.data = data
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get lengths
        self.lengths = data['dt_cnt'].values
        self.indices = np.arange(len(data))
        
        # Build buckets
        self._build_buckets()
    
    def _build_buckets(self):
        """Assign each sample to a bucket based on length."""
        self.buckets = [[] for _ in range(len(self.bucket_boundaries) + 1)]
        
        for idx, length in zip(self.indices, self.lengths):
            # Find bucket
            bucket_idx = 0
            for i, boundary in enumerate(self.bucket_boundaries):
                if length <= boundary:
                    bucket_idx = i
                    break
            else:
                bucket_idx = len(self.bucket_boundaries)
            
            self.buckets[bucket_idx].append(idx)
        
        # Sort within buckets for minimal padding
        for bucket in self.buckets:
            bucket.sort(key=lambda idx: self.lengths[idx])
    
    def __iter__(self):
        """Generate batches."""
        all_batches = []
        
        for bucket in self.buckets:
            if len(bucket) == 0:
                continue
            
            # Optionally shuffle within bucket
            if self.shuffle:
                np.random.shuffle(bucket)
            
            # Create batches from this bucket
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle batches
        if self.shuffle:
            np.random.shuffle(all_batches)
        
        return iter(all_batches)
    
    def __len__(self):
        """Total number of batches."""
        total = 0
        for bucket in self.buckets:
            total += len(bucket) // self.batch_size
            if not self.drop_last and len(bucket) % self.batch_size > 0:
                total += 1
        return total


def create_bucketing_dataloader(
    data: pd.DataFrame,
    batch_size: int,
    shuffle: bool = True
) -> Tuple[BucketingBatchSampler, int]:
    """
    Create bucketing batch sampler for dynamic padding.
    
    Returns:
        sampler: Batch sampler
        num_batches: Total number of batches
    """
    sampler = BucketingBatchSampler(
        data=data,
        batch_size=batch_size,
        bucket_boundaries=[50, 100, 150, 200],
        shuffle=shuffle,
        drop_last=True
    )
    
    return sampler, len(sampler)


# #### Validate



# --- Evaluation ---

def evaluate(
    model: nn.Module,
    dataloader: DataLoader, 
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = False,
    k_values: Tuple[int, ...] = (1, 5, 10, 20)
) -> Dict[str, float]:
    """
    Memory-efficient evaluation using streaming metrics.
    
    Key design:
    - Uses StreamingMetrics for incremental computation
    - NEVER accumulates predictions (prevents CPU OOM on large val sets)
    - Computes recall@K, micro-recall@K, precision@K, NDCG@K, MRR, positive-Brier
    
    Args:
        model: Model to evaluate (can be wrapped in DataParallel/DataParallelWrapper)
        dataloader: Validation DataLoader
        criterion: Loss function (e.g., BCEWithLogitsLoss)
        config: Model configuration
        device: Device for computation
        use_mixed_precision: Whether to use FP16 autocast
        max_batches: If set, only evaluate first N batches (for train subset)
        verbose: Print progress every 100 batches
        k_values: K values for top-K metrics
    
    Returns:
        Dict with val_loss, recall@K, micro_recall@K, precision@K, ndcg@K, mrr, positive_brier
    """
    model.eval()
    
    # Setup
    num_batches = len(dataloader)
    batches_to_process = min(num_batches, max_batches) if max_batches else num_batches
    
    if batches_to_process == 0:
        return _get_empty_eval_metrics(k_values)
    
    # Detect wrapper type
    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and 
        isinstance(model.module, DataParallelWrapper)
    )
    
    # Initialize streaming metrics
    metrics_tracker = StreamingMetrics(
        k_values=k_values,
        compute_mrr=True,
        compute_brier=True,
        vocab_size=config.target_cd_cnt
    )
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= batches_to_process:
                break
            
            if verbose and batch_idx % 100 == 0:
                print(f"    Evaluating batch {batch_idx}/{batches_to_process}")
            
            # Forward pass
            output, loss = _forward_batch(
                model, batch, config, device, 
                criterion, use_mixed_precision, is_wrapped
            )
            
            # Update streaming metrics
            metrics_tracker.update_loss(loss)
            
            if output is not None:
                _update_streaming_metrics(
                    metrics_tracker, output, batch, config
                )
            
            # Cleanup batch tensors immediately
            del output
            if batch_idx % 1000 == 0:
                gc.collect()
    
    # Compute and return final metrics
    results = metrics_tracker.compute()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return results



def _get_empty_eval_metrics(k_values: Tuple[int, ...]) -> Dict[str, float]:
    """Return empty metrics dict when no data to evaluate."""
    metrics = {'val_loss': 0.0, 'mrr': 0.0, 'positive_brier': 0.0}
    for k in k_values:
        metrics[f'recall@{k}'] = 0.0
        metrics[f'micro_recall@{k}'] = 0.0
        metrics[f'precision@{k}'] = 0.0
        metrics[f'ndcg@{k}'] = 0.0
    return metrics


def _forward_batch(
    model: nn.Module,
    batch: Dict,
    config: BaseConfig,
    device: torch.device,
    criterion: nn.Module,
    use_mixed_precision: bool,
    is_wrapped: bool
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Execute forward pass for a single batch.
    
    Returns:
        (output_logits, loss_scalar)
    """
    # Prepare inputs
    age = batch['age'].to(device, non_blocking=True)
    gender = batch['gender'].to(device, non_blocking=True)
    lob = batch['lob'].to(device, non_blocking=True)
    codes = batch['codes'].to(device, non_blocking=True)
    dt_cnt = batch['dt_cnt']
    y = batch['target']
    
    x = torch.cat([
        age.unsqueeze(-1),
        gender.unsqueeze(-1),
        lob.unsqueeze(-1),
        codes
    ], dim=-1)
    
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16) 
        if use_mixed_precision else nullcontext()
    )
    
    with autocast_ctx:
        if is_wrapped:
            targets_mh = batch['target_multihot'].to(device, non_blocking=True)
            dt_cnt_tensor = (
                dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor) 
                else torch.tensor(dt_cnt, device=device)
            )
            result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
            
            if isinstance(result, tuple):
                loss_val, extras = result
                output = extras.get('predictions')
            else:
                loss_val = result
                output = None
            loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
        else:
            if _model_has_moe(model):
                output, _ = model(x, return_moe_losses=False)
            else:
                output = model(x)
            loss = compute_loss(output, y, dt_cnt, config, criterion, device).item()
    
    return output, loss


def _update_streaming_metrics(
    metrics_tracker: 'StreamingMetrics',
    output: torch.Tensor,
    batch: Dict,
    config: BaseConfig
) -> None:
    """
    Update streaming metrics from a batch of predictions.
    
    Handles flattening over valid days and target alignment.
    """
    batch_size = output.shape[0]
    actual_len_dy = output.shape[1]
    
    dt_cnt = batch['dt_cnt']
    y = batch['target']
    
    dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
    y_flat = [item for sublist in y for item in sublist]
    
    # Flatten output over valid days
    output_flat = output.view(batch_size * actual_len_dy, config.target_cd_cnt)
    
    # Collect valid samples (only actual days, not padding)
    valid_outputs = []
    valid_targets = []
    
    for j in range(batch_size):
        valid_days = min(int(dt_cnt_values[j]), actual_len_dy)
        if valid_days <= 0:
            continue
        
        # Output indices (uses actual_len_dy)
        out_start = actual_len_dy * j
        out_end = out_start + valid_days
        valid_outputs.append(output_flat[out_start:out_end])
        
        # Target indices (uses config.len_dy since y is padded to full length)
        y_start = config.len_dy * j
        y_end = y_start + valid_days
        valid_targets.extend(y_flat[y_start:y_end])
    
    if valid_outputs:
        predictions = torch.cat(valid_outputs)
        metrics_tracker.update(predictions, valid_targets)
        del predictions, valid_outputs, valid_targets


# #### Test


def test_prepare_tensor_and_multihot():
    config = BaseConfig(batch_size=4, len_dy=200, len_cd=80, device=device.type)
    dt_cnt, x, y = prepare_tensor(df_train.head(config.batch_size), config, device)

    assert x.shape == (config.batch_size, config.len_dy, 2 + config.len_cd)
    assert len(y) == config.batch_size

    y_flat = [codes for day_list in y for codes in day_list]
    multihot = create_multihot_targets_vectorized(
        y_flat[:10],
        num_samples=10,
        vocab_size=config.target_cd_cnt,
        device=device
    )

    assert multihot.shape == (10, config.target_cd_cnt)
    print("prepare_tensor + multihot ✔️")
test_prepare_tensor_and_multihot()



def test_compute_loss_smoke():
    config = BaseConfig(batch_size=4, len_dy=200, len_cd=80, device=device.type)
    dt_cnt, x, y = prepare_tensor(df_train.head(config.batch_size), config, device)

    model = BaselineTransformer(config).to(device)
    with torch.no_grad():
        logits = model(x.to(device))

    crit = nn.BCEWithLogitsLoss()
    loss = compute_loss(logits, y, dt_cnt, config, crit, device)

    assert loss.ndim == 0 and torch.isfinite(loss)
    print("compute_loss ✔️")
test_compute_loss_smoke()



def test_train_epoch_smoke():
    config = BaseConfig(batch_size=4, len_dy=200, len_cd=80, learning_rate=1e-3, device=device.type)
    model = BaselineTransformer(config).to(device)
    opt = optim.AdamW(model.parameters(), lr=config.learning_rate)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    crit = nn.BCEWithLogitsLoss()

    train_subset = df_train.head(config.batch_size * 2)  # Make sure we have at least one full batch
    train_dataset = ClinicalDataset(train_subset, config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                              collate_fn=create_collate_fn(config), drop_last=True)
    metrics = train_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=opt,
        scheduler=sched,
        criterion=crit,
        config=config,
        device=device,
        use_mixed_precision=False,
        use_bucketing=False
    )

    assert 'train_loss' in metrics and 'aux_loss' in metrics
    print("train_epoch smoke ✔️")
test_train_epoch_smoke()



def test_evaluate_smoke():
    config = BaseConfig(batch_size=4, len_dy=200, len_cd=80, device=device.type)
    model = BaselineTransformer(config).to(device)
    crit = nn.BCEWithLogitsLoss()

    # Prime the model with one forward so embeddings are on-device
    
    dt_cnt, x, y = prepare_tensor(df_train.head(config.batch_size*2), config, device)
    with torch.no_grad():
        model(x.to(device))
        
    # --- NEW: Create Dataset and DataLoader for the test ---
    val_subset = df_val.head(config.batch_size)
    val_dataset = ClinicalDataset(val_subset, config)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=create_collate_fn(config))

    val_metrics = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=crit,
        config=config,
        device=device,
        use_mixed_precision=False
    )

    assert 'val_loss' in val_metrics and 'recall@10' in val_metrics
    print("evaluate smoke ✔️")
test_evaluate_smoke()


# ### Training save and reload



# --- Checkpoint management ---

def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    metrics: Dict,
    is_best: bool = False,
    keep_last_n: int = 1,  # Only keep last N epoch checkpoints
    save_optimizer: bool = True  # Option to skip optimizer for final save
):
    """
    Save checkpoint - PyTorch official pattern.
    
    This is what every major lab uses. Simple, tested, works.
    
    Saves 3 files:
    - checkpoint_latest.pt (always)
    - checkpoint_epoch{N}.pt (every epoch)
    - checkpoint_best.pt (when val loss improves)
    """
    import glob
    import shutil
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Handle DataParallel
    if isinstance(model, nn.DataParallel):
        inner_model = model.module
    else:
        inner_model = model   
    # Handle DataParallelWrapper
    if isinstance(inner_model, DataParallelWrapper):
        actual_model = inner_model.model
    else:
        actual_model = inner_model
    # Build checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'timestamp': time.time(),
        'model_type': type(actual_model).__name__,
        'scheduler_config': {
            'type': type(scheduler).__name__,
            'total_steps': getattr(scheduler, '_total_steps', None),  # If you store it
            'warmup_steps': getattr(scheduler, '_warmup_steps', None),  # If you store it
        } if scheduler else None
    }
    # ============================================================
    # ROLLING CLEANUP: Remove old epoch checkpoints BEFORE saving
    # Prevents disk full errors during long training runs
    # ============================================================
    existing_checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch*.pt')),
        key=lambda x: int(x.split('epoch')[-1].replace('.pt', ''))
    )    
    # Remove oldest checkpoints to maintain only (keep_last_n - 1)
    # only keep (keep_last_n - 1) because we're about to add one more
    while len(existing_checkpoints) >= keep_last_n:
        oldest = existing_checkpoints.pop(0)
        if os.path.exists(oldest):
            try:
                os.remove(oldest)
                print(f"🗑️ Removed old checkpoint: {os.path.basename(oldest)}")
            except OSError as e:
                print(f"⚠️ Could not remove {oldest}: {e}")    
                
    # Save latest (for resume)
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    print(f"💾 Saved: checkpoint_latest.pt")
    
    # Save epoch checkpoint (every epoch)
    epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, epoch_path)
    
    # Save best (when val loss improves)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        shutil.copy(epoch_path, best_path)
        if metrics and isinstance(metrics, list) and len(metrics) > 0:
            last_val_loss = metrics[-1].get('val_loss', 0)
            print(f"✅ New best! Val loss: {last_val_loss:.4f}")
        else:
            print(f"✅ New best checkpoint saved!")
    
    return latest_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None
) -> Dict:
    """
    Load checkpoint - PyTorch official pattern.
    
    Returns:
        Dict with: epoch, global_step, metrics
    """
    
    print(f" Loading checkpoint: {checkpoint_path}")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only = False)
    # ====== UNWRAP MODEL ======
    if isinstance(model, nn.DataParallel):
        inner_model = model.module
    else:
        inner_model = model
    
    if isinstance(inner_model, DataParallelWrapper):
        actual_model = inner_model.model
    else:
        actual_model = inner_model
        
    # Restore states
    actual_model.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
    if scheduler and checkpoint_data.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
    
    if scaler and checkpoint_data.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
    
    print(f"✅ Resumed from epoch {checkpoint_data['epoch']}, step {checkpoint_data['global_step']}")
    
    return {
        'epoch': checkpoint_data['epoch'],
        'global_step': checkpoint_data['global_step'],
        'metrics': checkpoint_data.get('metrics', {})
    }


# #### Test




import shutil
def test_save_load_checkpoint_only():
    """
    Test ONLY save_checkpoint() and load_checkpoint() functions.
    
    Validates:
    - Checkpoint files created
    - All state saved correctly
    - State can be loaded
    - Weights match exactly after load
    """
    
    print("\n" + "="*80)
    print("🧪 TEST: save_checkpoint() + load_checkpoint()")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple model
    config = BaseConfig(batch_size=4, len_dy=32, len_cd=30)
    model = BaselineTransformer(config).to(device)


# --- Metrics computation ---

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
    1. Recall@1, 5, 10, 20, 50 - Clinical utility at different cutoffs
    2. Precision@5, 10, 20, 50 - How many predictions are correct
    3. Micro-Recall@1, 10, 20 - Per-code hit rate (more granular than sample-level)
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
            y_end = y_start + valid_days  # use same valid_days as model output
            valid_y.extend(y_flat[y_start:y_end])
        
        if len(valid_outputs) == 0:
            return {
               'recall@1': 0.0, 'recall@5': 0.0, 'recall@10': 0.0, 'recall@20': 0.0, 'recall@50': 0.0,
                'precision@5': 0.0, 'precision@10': 0.0, 'precision@20': 0.0, 'precision@50': 0.0,
                'micro_recall@1': 0.0, 'micro_recall@10': 0.0, 'micro_recall@20': 0.0,
                'ndcg@20': 0.0, 'positive_brier': 0.0
            }
        
        predictions = torch.cat(valid_outputs)  # [num_valid_samples, vocab_size]
        num_samples = len(predictions)
        
        metrics = {}
        sorted_indices = torch.argsort(predictions, dim=-1, descending=True)
        # ============================================================
        # 1. RECALL @ K (for K=1, 5, 10, 20, 50)
        # ============================================================
        # Recall: "Was ANY true code in top-K predictions?"
        for k in [1, 5, 10, 20, 50]:
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
            
            metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
        
        # ============================================================
        # 3. MICRO-RECALL @ K (for K=1, 5, 10, 20) - Lightweight version
        # ============================================================
        for k in [1, 5, 10, 20]:
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



def compute_embedding_quality_epoch(
    model: nn.Module,
    val_data: pd.DataFrame,
    config: BaseConfig,
    device: torch.device,
    num_samples: int = 200,
    use_mixed_precision: bool = False
) -> Dict[str, float]:
    """
    Evaluate embedding quality at epoch end.
    
    Run this ONCE per epoch (expensive!) to check if embeddings are useful
    for downstream tasks.
    
    Metrics computed:
    1. Embedding std_mean - Detects embedding collapse (should be > 0.05)
    2. NN target overlap - Do similar embeddings have similar codes? (higher = better)
    
    Why these matter for downstream tasks:
    - If embeddings collapse (low std), they won't transfer to downstream classifiers
    - If NN overlap is low, embeddings don't capture clinical similarity
    
    Returns:
        Dict with 'embedding_std_mean', 'nn_target_overlap'
    """
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    if isinstance(actual_model, DataParallelWrapper):
        actual_model = actual_model.model
    actual_model.eval()
    metrics = {}
    
    # Sample validation data
    sample_size = min(num_samples, len(val_data))
    val_sample = val_data.sample(sample_size, random_state=42)
    val_dataset = ClinicalDataset(val_sample, config)

    # This runs on single GPU (unwrapped model), not DataParallel
    # so set up maximum 8 batch size 
    embedding_batch_size = min(8, config.batch_size)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=embedding_batch_size, 
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers = 0
    )        
    all_embeddings = []
    all_targets = []
    with EmbeddingExtractor(actual_model) as extractor:
        with torch.no_grad():  
            for batch in val_loader:  # ← Use DataLoader instead of manual iteration
                age = batch['age'].to(device)
                gender = batch['gender'].to(device)
                codes = batch['codes'].to(device)
                lob = batch['lob'].to(device)
                dt_cnt = batch['dt_cnt']
                y = batch['target']

                x = torch.cat([
                    age.unsqueeze(-1),
                    gender.unsqueeze(-1),
                    lob.unsqueeze(-1),
                    codes
                ], dim=-1)

                # ============================================================
                # FORWARD PASS - hook captures embeddings automatically
                # ============================================================
                try:
                    if use_mixed_precision:
                        dtype = getattr(config, 'dtype', torch.float16)
                        with torch.cuda.amp.autocast(dtype=dtype):
                            if _model_has_moe(actual_model):
                                _ = actual_model(x, return_moe_losses=False)
                            else:
                                _ = actual_model(x)
                    else:
                        if _model_has_moe(actual_model):
                            _ = actual_model(x, return_moe_losses=False)
                        else:
                            _ = actual_model(x)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️ OOM in embedding extraction at batch {batch_idx}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    raise

                # Get patient embeddings (last valid day)    
                patient_embs = extractor.get_patient_embedding(dt_cnt)

                # Collect embeddings and targets
                for j in range(len(dt_cnt)):
                    if dt_cnt[j] > 0:
                        all_embeddings.append(patient_embs[j].cpu())
                        
                        # Aggregate all target codes for this patient
                        patient_targets = y[j]
                        all_codes = set()
                        for day_codes in patient_targets:
                            all_codes.update([c for c in day_codes if c > 0])
                        all_targets.append(all_codes)
                del x, age, gender, codes, lob, patient_embs
                # Periodic cache clear for long loops
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
    # Final cleanup
    torch.cuda.empty_cache()
    
    if len(all_embeddings) == 0:
        return {'embedding_std_mean': 0.0, 'nn_target_overlap': 0.0}
    
    embeddings_tensor = torch.stack(all_embeddings)  # [num_patients, 256]
    
    # 1. Embedding Space Utilization (detect collapse)
    emb_std = embeddings_tensor.std(dim=0)
    metrics['embedding_std_mean'] = emb_std.mean().item()
    
    # WARNING if collapsed
    if metrics['embedding_std_mean'] < 0.01:
        print(f"⚠️ WARNING: Embeddings collapsing! std_mean={metrics['embedding_std_mean']:.4f}")
    
    # 2. Nearest Neighbor Target Overlap (do similar embeddings have similar codes?)
    dists = torch.cdist(embeddings_tensor, embeddings_tensor)
    
    nn_accuracies = []
    for i in range(min(100, len(embeddings_tensor))):  # Sample 100 for speed
        # Get 5 nearest neighbors
        _, indices = torch.topk(dists[i], k=6, largest=False)
        neighbors = indices[1:6].tolist()  # Exclude self
        
        my_targets = all_targets[i]
        if len(my_targets) > 0:
            overlaps = []
            for nb_idx in neighbors:
                nb_targets = all_targets[nb_idx]
                if len(nb_targets) > 0:
                    # Jaccard similarity
                    overlap = len(my_targets & nb_targets) / len(my_targets | nb_targets)
                    overlaps.append(overlap)
            if overlaps:
                nn_accuracies.append(np.mean(overlaps))
    
    metrics['nn_target_overlap'] = np.mean(nn_accuracies) if nn_accuracies else 0.0
    
    return metrics


def compute_moe_batch_metrics(
    moe_losses: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Extract MoE health metrics from a single batch.
    
    Call this every batch during training to track MoE routing in real-time.
    
    Metrics:
    1. Expert load CV - Coefficient of variation (lower = better balance)
    2. Num collapsed experts - Experts with <5% usage (should be 0)
    3. Expert Gini - Inequality metric (0 = perfect equality, 1 = total inequality)
    
    Returns:
        Dict with MoE health metrics
    """


    if 'expert_usage' not in moe_losses:
        return {}
    
    metrics = {}
    usage = moe_losses['expert_usage']
    
    # Handle DataParallel, expert_usage might be [num_gpus, num_experts]
    # Average across GPUs
    if usage.dim() > 1:
        usage = usage.mean(dim=0)  
        
    usage = usage.cpu().numpy()
    # 1. Load balance CV
    if usage.mean() > 0:
        metrics['expert_load_cv'] = usage.std() / usage.mean()
    else:
        metrics['expert_load_cv'] = 0.0
    
    # 2. Collapsed experts
    metrics['num_collapsed_experts'] = int((usage < 0.05).sum())
    
    # 3. Gini coefficient
    sorted_usage = np.sort(usage)
    n = len(sorted_usage)
    if sorted_usage.sum() > 0:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_usage)) / (n * np.sum(sorted_usage)) - (n + 1) / n
        metrics['expert_gini'] = gini
    else:
        metrics['expert_gini'] = 0.0
    
    # 4. Aux loss
    if 'aux_loss' in moe_losses:
        aux = moe_losses['aux_loss']
        metrics['aux_loss'] = aux.mean().item() if aux.numel() > 1 else aux.item()
    
    return metrics

def compute_router_gradient_metrics(
    model: nn.Module,
    moe_config: Optional['MoEConfig'] = None,
    log_all_layers: bool = False
) -> Dict[str, float]:
    """
    Monitor router gradient health for MoE stability diagnostics.
    
    Must be called AFTER backward() but BEFORE optimizer.step()!
    
    Call this every log_interval batches to track:
    1. Router gradient norms (should be stable, not exploding/vanishing)
    2. Router weight statistics (for detecting stuck routers)
    3. Per-layer breakdown (optional, for debugging)
    
    Warning thresholds (empirically determined):
    - grad_norm > 10.0: Exploding gradients - reduce LR or add clipping
    - grad_norm < 1e-7: Vanishing gradients - check focal loss/initialization  
    - grad_std ≈ 0: Router not learning - check gradient flow
    - weight_std < 0.01: Router weights collapsed - reinitialize
    
    Args:
        model: The model (handles DataParallel wrapping automatically)
        moe_config: MoE configuration (if None, returns empty dict)
        log_all_layers: If True, log per-layer breakdown (more overhead)
    
    Returns:
        Dict with router gradient metrics:
        - router_grad_norm_mean: Mean L2 norm of router gradients
        - router_grad_norm_max: Max L2 norm (detect explosions)
        - router_grad_norm_min: Min L2 norm (detect vanishing)
        - router_grad_exploding: 1 if any grad > 10.0, else 0
        - router_grad_vanishing: 1 if any grad < 1e-7, else 0
        - router_weight_mean: Mean of router weights
        - router_weight_std: Std of router weights (should have variance)
        - router_weight_abs_max: Max absolute weight (detect explosion)
        - router_grad_sparsity: Fraction of near-zero gradients
    """
    metrics = {}
    
    if moe_config is None:
        return metrics
    
    # Find router layers in the model (handle wrappers)
    router_grads = []
    router_weights = []
    router_layer_names = []  # For per-layer debugging
    
    # Unwrap DataParallel/DataParallelWrapper
    actual_model = model
    if hasattr(model, 'module'):
        actual_model = model.module
    if hasattr(actual_model, 'model'):
        actual_model = actual_model.model
    
    # Search for router parameters
    for name, param in actual_model.named_parameters():
        if 'router' in name.lower() or 'gate' in name.lower():
            if param.grad is not None:
                # Compute gradient norm efficiently (no graph retention)
                grad = param.grad.detach()
                grad_norm = grad.norm().item()
                router_grads.append(grad_norm)
                router_layer_names.append(name)
                
                # Optional: compute gradient sparsity (near-zero elements)
                if log_all_layers:
                    grad_sparsity = (grad.abs() < 1e-8).float().mean().item()
                    metrics[f'router_grad_sparsity_{name}'] = grad_sparsity
            
            # Weight statistics (always available)
            weight = param.detach()
            router_weights.append(weight)
    
    # Aggregate gradient metrics
    if router_grads:
        metrics['router_grad_norm_mean'] = np.mean(router_grads)
        metrics['router_grad_norm_max'] = np.max(router_grads)
        metrics['router_grad_norm_min'] = np.min(router_grads)
        metrics['router_grad_norm_std'] = np.std(router_grads) if len(router_grads) > 1 else 0.0
        
        # Detect gradient issues (binary flags for alerting)
        metrics['router_grad_exploding'] = int(max(router_grads) > 10.0)
        metrics['router_grad_vanishing'] = int(min(router_grads) < 1e-7)
        
        # Count healthy vs problematic layers
        metrics['router_layers_total'] = len(router_grads)
        metrics['router_layers_healthy'] = sum(1 for g in router_grads if 1e-7 < g < 10.0)
    
    # Aggregate weight metrics
    if router_weights:
        # Concatenate all router weights for global statistics
        all_weights = torch.cat([w.flatten() for w in router_weights])
        metrics['router_weight_mean'] = all_weights.mean().item()
        metrics['router_weight_std'] = all_weights.std().item()
        metrics['router_weight_abs_max'] = all_weights.abs().max().item()
        
        # Detect collapsed routers (very low variance = not differentiating)
        metrics['router_weight_collapsed'] = int(metrics['router_weight_std'] < 0.01)
    
    return metrics


# #### Resource metrics


def compute_moe_performance_metrics(
    expert_usage: torch.Tensor,  # [num_experts] usage distribution
    router_probs_history: List[torch.Tensor],  # Router probabilities over time
    num_experts: int
) -> Dict[str, float]:
    """
    MoE-specific quality metrics.
        - Expert specialization
        - Load balancing quality
        - Routing entropy    
    Returns:
        1. Load Balance Score:
           - Coefficient of variation of expert loads
           - Standard deviation / mean
           - Lower = better balance
        
        2. Expert Specialization:
           - Entropy of routing distribution
           - Higher = more specialized
        
        3. Expert Collapse:
           - Number of experts with <5% usage
           - Binary: any expert collapsed?
    """
    metrics = {}
    
    # 1. Load balance (Switch Transformer metric)
    expert_loads = expert_usage.cpu().numpy()
    
    metrics['expert_load_mean'] = expert_loads.mean()
    metrics['expert_load_std'] = expert_loads.std()
    metrics['expert_load_cv'] = expert_loads.std() / expert_loads.mean() if expert_loads.mean() > 0 else 0.0
    metrics['expert_load_min'] = expert_loads.min()
    metrics['expert_load_max'] = expert_loads.max()
    metrics['load_balance_score'] = 1.0 - metrics['expert_load_cv']  # Higher = better (0-1)
    
    # 2. Expert specialization (routing entropy)
    # Average entropy across all tokens
    if len(router_probs_history) > 0:
        avg_router_probs = torch.stack(router_probs_history).mean(dim=0)  # [num_experts]
        
        # Entropy: H = -Σ p_i log(p_i)
        # High entropy = uniform routing (less specialized)
        # Low entropy = peaked routing (more specialized)
        entropy = -(avg_router_probs * torch.log(avg_router_probs + 1e-10)).sum()
        max_entropy = np.log(num_experts)  # Uniform distribution
        
        metrics['routing_entropy'] = entropy.item()
        metrics['routing_entropy_normalized'] = entropy.item() / max_entropy  # 0-1
        metrics['specialization_score'] = 1.0 - (entropy.item() / max_entropy)  # Higher = more specialized
    
    # 3. Expert collapse detection
    collapse_threshold = 0.05  # Expert used <5% of time
    collapsed_experts = (expert_loads < collapse_threshold).sum()
    
    metrics['num_collapsed_experts'] = int(collapsed_experts)
    metrics['expert_collapse'] = bool(collapsed_experts > 0)
    metrics['effective_experts'] = num_experts - collapsed_experts  # How many actually used
    
    # 4. Expert diversity (Gini coefficient)
    # Measures inequality of expert usage (0 = perfect equality, 1 = one expert does everything)
    sorted_loads = np.sort(expert_loads)
    n = len(sorted_loads)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_loads)) / (n * np.sum(sorted_loads)) - (n + 1) / n
    metrics['expert_gini'] = gini
    
    return metrics


def compute_memory_metrics(
    device: torch.device,
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_gpus: int = 1
) -> Dict[str, float]:
    """
    GPU memory usage metrics.
        - Peak memory (limits batch size)
        - Memory efficiency (Flash Attention benefit)
        - Per-GPU memory for multi-GPU    
    Returns:
        1. Peak Memory:
           - Maximum allocated during training
           - Per-GPU breakdown for DataParallel
        
        2. Memory Efficiency:
           - Memory per parameter
           - Memory per sample
           - Activation memory vs parameter memory
        
        3. Practical Limits:
           - Max batch size achievable
           - Estimated memory for larger models
    """
    metrics = {}
    
    if device.type != 'cuda':
        return metrics
    
    # Model parameters memory
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = total_params * 4 / (1024 ** 3)  # FP32
    
    metrics['model_params'] = total_params
    metrics['model_memory_gb'] = param_memory_gb
    
    # Runtime memory (across all GPUs)
    total_allocated = 0
    total_reserved = 0
    total_peak = 0
    
    for gpu_id in range(num_gpus):
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)
        
        total_allocated += allocated


        total_reserved += reserved
        total_peak += peak
        
        # Per-GPU metrics
        metrics[f'gpu{gpu_id}_allocated_gb'] = allocated
        metrics[f'gpu{gpu_id}_reserved_gb'] = reserved
        metrics[f'gpu{gpu_id}_peak_gb'] = peak
    
    # Aggregate metrics
    metrics['total_allocated_gb'] = total_allocated
    metrics['total_reserved_gb'] = total_reserved
    metrics['total_peak_gb'] = total_peak
    metrics['avg_allocated_per_gpu_gb'] = total_allocated / num_gpus
    metrics['avg_peak_per_gpu_gb'] = total_peak / num_gpus
    
    # Memory efficiency
    activation_memory = total_peak - param_memory_gb * num_gpus  # Approx activations
    metrics['activation_memory_gb'] = max(0, activation_memory)
    metrics['memory_per_sample_mb'] = (activation_memory * 1024) / batch_size if batch_size > 0 else 0
    
    # Memory breakdown
    metrics['param_memory_percent'] = (param_memory_gb * num_gpus / total_peak) * 100 if total_peak > 0 else 0
    metrics['activation_memory_percent'] = (activation_memory / total_peak) * 100 if total_peak > 0 else 0
    
    # Theoretical max batch size (assume 80% GPU memory usage is safe)
    gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    safe_memory = gpu_total_memory * 0.8
    memory_per_sample = metrics['memory_per_sample_mb'] / 1024  # Convert to GB
    
    if memory_per_sample > 0:
        metrics['max_batch_size_theoretical'] = int((safe_memory - param_memory_gb) / memory_per_sample)
    
    return metrics

def compute_flops_metrics(
    model_config: BaseConfig,
    batch_size: int,
    seq_len: int,
    num_experts: int = None,
    top_k: int = None,
    use_moe_from_layer: int = None, 
    actual_throughput: float = None  # tokens/sec from training
) -> Dict[str, float]:
    """
    Computational cost metrics.
        - FLOPs (floating point operations)
        - MFU (Model FLOPs Utilization)
        - Hardware efficiency    
    Returns:
        1. Theoretical FLOPs:
           - Forward pass FLOPs
           - Total FLOPs (forward + backward ≈ 3× forward)
           - FLOPs comparison vs baseline
        
        2. Model FLOPs Utilization (MFU):
           - Achieved FLOPs / Peak hardware FLOPs
           - Standard metric in LLM papers (GPT-3, PaLM)
        
        3. FLOPs Efficiency:
           - FLOPs per parameter
           - Effective compute (for MoE: top_k/num_experts × base_flops)
    """
    metrics = {}
    
    d_model = model_config.embedding_size
    d_ff = model_config.nhid
    n_layers = model_config.nlayers
    
    # ============================================================
    # 1. ATTENTION FLOPS
    # ============================================================
    # Per layer: QKV projections + attention + output projection
    # QKV: 3 × (2 × seq_len × d_model × d_model)
    # Attention: 2 × seq_len^2 × d_model
    # Output: 2 × seq_len × d_model × d_model
    
    attn_flops_per_layer = (
        3 * (2 * seq_len * d_model * d_model) +  # QKV
        2 * seq_len * seq_len * d_model +         # Attention scores
        2 * seq_len * d_model * d_model           # Output projection
    )
    
    # ============================================================
    # 2. FFN FLOPS
    # ============================================================
    if num_experts is None or top_k is None:
        # Dense FFN
        ffn_flops_per_layer = 2 * seq_len * d_model * d_ff + 2 * seq_len * d_ff * d_model
        effective_ffn_layers = n_layers
    else:
        # MoE FFN (only top-k experts activated)
        base_ffn_flops = 2 * seq_len * d_model * d_ff + 2 * seq_len * d_ff * d_model
        # Router overhead
        router_flops = 2 * seq_len * d_model * num_experts
        # Effective FFN FLOPs (only top-k experts)
        effective_ffn_flops = router_flops + (top_k / num_experts) * base_ffn_flops
        
        ffn_flops_per_layer = effective_ffn_flops
        effective_ffn_layers = n_layers - use_moe_from_layer if use_moe_from_layer else n_layers  # Only MoE layers
    
    # ============================================================
    # 3. TOTAL FLOPS
    # ============================================================
    # Attention FLOPs (all layers)
    total_attn_flops = attn_flops_per_layer * n_layers
    
    # FFN FLOPs (dense + MoE layers)
    dense_ffn_layers = use_moe_from_layer if (num_experts and use_moe_from_layer) else n_layers
    dense_ffn_flops = (2 * seq_len * d_model * d_ff + 2 * seq_len * d_ff * d_model) * dense_ffn_layers
    moe_ffn_flops = ffn_flops_per_layer * effective_ffn_layers if num_experts else 0
    total_ffn_flops = dense_ffn_flops + moe_ffn_flops
    
    # Embeddings and output
    embed_flops = 2 * seq_len * d_model * model_config.target_cd_cnt  # Output projection
    
    # Forward pass total
    forward_flops = total_attn_flops + total_ffn_flops + embed_flops
    
    # Total (forward + backward ≈ 3× forward)
    total_flops = forward_flops * 3
    
    metrics['forward_flops'] = forward_flops
    metrics['total_flops_per_sample'] = total_flops
    metrics['total_flops_per_batch'] = total_flops * batch_size
    
    # ============================================================
    # 4. MODEL FLOPS UTILIZATION (MFU)
    # ============================================================
    # T4 GPU: 8.1 TFLOPS (FP32), 65 TFLOPS (FP16 Tensor Core)
    # For 4× T4: Total peak = 260 TFLOPS (FP16)
    
    peak_flops_t4_fp16 = 65e12  # Per GPU
    peak_flops_total = peak_flops_t4_fp16 * 4  # 4 GPUs
    
    if actual_throughput is not None:  # tokens/sec
        # Achieved FLOPs/sec
        achieved_flops = (forward_flops / seq_len) * actual_throughput
        
        # MFU = achieved / peak
        mfu = achieved_flops / peak_flops_total
        metrics['achieved_tflops'] = achieved_flops / 1e12
        metrics['mfu'] = mfu
        metrics['mfu_percent'] = mfu * 100
    
    # ============================================================
    # 5. EFFICIENCY COMPARISONS
    # ============================================================
    # FLOPs per parameter (architecture efficiency)
    metrics['flops_per_param'] = total_flops / total_params if hasattr(model_config, 'total_params') else 0
    
    # MoE efficiency (if applicable)
    if num_experts and top_k:
        # Compute efficiency: fraction of experts used
        metrics['moe_compute_efficiency'] = top_k / num_experts
        # Parameter efficiency: total params vs active params
        metrics['moe_param_efficiency'] = top_k / num_experts
    
    return metrics

def compute_cost_metrics(
    training_time_sec: float,
    num_epochs: int,
    gpu_type: str = "T4",
    num_gpus: int = 4,
    region: str = "us-central1"  # GCP region
) -> Dict[str, float]:
    """
    Training cost estimation.
        - Training cost in dollars
        - Cost per experiment
        - Cost projections for full training    
    Returns:
        1. Actual Cost:
           - Cost for this experiment
           - Cost per epoch
           - Cost per 1000 samples
        
        2. Projected Costs:
           - Full training (100 epochs)
           - Cost comparison vs baseline
        
        3. Cost Breakdown:
           - Compute cost
           - Memory cost (if separated)
           - Storage cost (checkpoints)
    """
    metrics = {}
    
    # GCP pricing (approximate, as of 2024)
    # https://cloud.google.com/compute/gpus-pricing
    gpu_hourly_rates = {
        'T4': 0.35,      # per GPU per hour (GCP on-demand)
        'V100': 2.48,    # per GPU per hour
        'A100': 3.67,    # per GPU per hour (40GB)
        'L4': 0.70       # per GPU per hour
    }
    
    # Get hourly rate
    rate_per_gpu = gpu_hourly_rates.get(gpu_type, 0.35)
    rate_total = rate_per_gpu * num_gpus
    
    # Training time in hours
    training_hours = training_time_sec / 3600
    
    # 1. Actual cost for this run
    metrics['cost_usd'] = training_hours * rate_total
    metrics['cost_per_epoch_usd'] = metrics['cost_usd'] / max(num_epochs, 1)   
    
    # 2. Projected costs
    # Typical clinical transformer: 100-300 epochs for convergence
    for num_epochs_projection in [10, 50, 100, 200]:
        cost_projection = (training_time_sec / max(num_epochs, 1)) * num_epochs_projection / 3600 * rate_total
        metrics[f'projected_cost_{num_epochs}epochs_usd'] = cost_projection
    
    # 3. Cost efficiency
    # Cost per 1000 samples
    samples_per_sec = metrics.get('samples_per_sec', 0)
    if samples_per_sec > 0:
        cost_per_1k_samples = (rate_total / 3600) / samples_per_sec * 1000
        metrics['cost_per_1k_samples_usd'] = cost_per_1k_samples
    
    # 4. Hardware utilization cost
    # What fraction of GPU time is actually computing vs idle?
    mfu = metrics.get('mfu', 0)
    if mfu > 0:
        # Effective cost (if GPU was 100% utilized)
        metrics['effective_cost_usd'] = metrics['cost_usd'] / mfu
        metrics['wasted_compute_usd'] = metrics['cost_usd'] * (1 - mfu)
    
    metrics['gpu_type'] = gpu_type
    metrics['num_gpus'] = num_gpus
    metrics['hourly_rate_usd'] = rate_total
    
    return metrics

def compute_training_time_metrics(
    total_train_time: float,  # Seconds
    num_epochs: int,
    num_samples: int,
    num_tokens: int,  # batch_size * len_dy * num_batches


    batch_size: int,
    data_load_time: float = 0.0,  # Optional profiling
    forward_time: float = 0.0,
    backward_time: float = 0.0
) -> Dict[str, float]:
    """
    Training time and throughput metrics.
    CRITICAL for comparing architectures:
    - Absolute time (wall-clock)
    - Normalized throughput (tokens/sec, samples/sec)
    - Time breakdown (data loading vs compute)    
    Returns:
        1. Wall-Clock Time:
           - Total training time
           - Time per epoch
           - Time per sample
        
        2. Throughput:
           - Samples per second
           - Tokens per second (standard LLM metric)
           - Batches per second
        
        3. Time Breakdown (if profiled):
           - Data loading %
           - Forward pass %
           - Backward pass %
           - Optimizer step %
    """
    metrics = {}
    
    # Absolute time
    metrics['total_train_time_sec'] = total_train_time
    metrics['time_per_epoch_sec'] = total_train_time / num_epochs
    metrics['time_per_sample_ms'] = (total_train_time / num_samples) * 1000
    
    # Throughput
    metrics['samples_per_sec'] = num_samples / total_train_time
    metrics['tokens_per_sec'] = num_tokens / total_train_time
    metrics['batches_per_sec'] = (num_samples / batch_size) / total_train_time if total_train_time > 0 and batch_size > 0 else 0
    
    # Time breakdown (if profiled)
    if data_load_time > 0 or forward_time > 0:
        total_profiled = data_load_time + forward_time + backward_time
        metrics['data_load_percent'] = (data_load_time / total_profiled) * 100
        metrics['forward_percent'] = (forward_time / total_profiled) * 100
        metrics['backward_percent'] = (backward_time / total_profiled) * 100
    
    # Training speed (industry standard: steps per second)
    # Useful for comparing with published baselines
    metrics['steps_per_sec'] = (num_samples / batch_size) / total_train_time if total_train_time > 0 and batch_size > 0 else 0
    
    return metrics


# #### Primary functional metrics


"""
CRITICAL for healthcare AI publication:
- Top-K accuracy reflects clinical workflow (doctors review multiple suggestions)
- Multi-label aware (each day has multiple diagnoses/procedures)
- Stratified by code frequency (rare codes often most important clinically)
"""

def compute_primary_task_metrics(
    predictions: torch.Tensor,  # [num_samples, vocab_size]
    targets: List[List[int]],   # Multi-label targets
    vocab_size: int
) -> Dict[str, float]:
    """
    Primary metrics for medical code prediction.
    
    Returns:
        1. Top-K Recall@K (K=1,5,10,20,50): 
           "Was ANY true code in top-K predictions?"
           - Most important for clinical utility
           - Used in BEHRT, Med-BERT, ClinicalBERT papers
        
        2. Mean Reciprocal Rank (MRR):
           "Average rank of first correct code"
           - Ranking quality metric
           - Standard in information retrieval
        
        3. Precision@K and F1@K:
           "Of top-K predictions, how many were correct?"
           - Balances recall with false positives
           - Important for alert fatigue in healthcare
    """
    metrics = {}
    
    # Top-K Recall (Primary metric)
    for k in [1, 5, 10, 20, 50]:
        top_k_preds = torch.topk(predictions, k, dim=-1).indices
        correct = 0
        total = 0
        
        for i, target_codes in enumerate(targets):
            true_codes = [c for c in target_codes if c != 0]
            if len(true_codes) > 0:
                total += 1
                # Hit if ANY true code in top-K
                if any(code in top_k_preds[i].tolist() for code in true_codes):
                    correct += 1
        
        metrics[f'recall@{k}'] = correct / total if total > 0 else 0.0
        
    # Precision@K (Fraction of top-K that are correct)
    for k in [1, 5, 10, 20, 50]:
        top_k_preds = torch.topk(predictions, k, dim=-1).indices
        precisions = []
        
        for i, target_codes in enumerate(targets):
            true_codes = set([c for c in target_codes if c != 0])
            if len(true_codes) > 0:
                pred_codes = top_k_preds[i].tolist()
                hits = sum(1 for code in pred_codes if code in true_codes)
                precisions.append(hits / k)
        
        metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
    
    # Mean Reciprocal Rank
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
    # F1@K (Harmonic mean of precision and recall)
    for k in [1, 5, 10, 20, 50]:
        if f'recall@{k}' in metrics and f'precision@{k}' in metrics:
            r = metrics[f'recall@{k}']
            p = metrics[f'precision@{k}']
            metrics[f'f1@{k}'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0 
    # Add micro-recall metrics
    metrics.update(compute_micro_recall_at_k(predictions, targets, [5, 10, 20, 50]))
    # Add NDCG metrics
    metrics.update(compute_ndcg_at_k(predictions, targets, [10, 20, 50])) 
    return metrics

"""
CRITICAL for understanding training dynamics:
- Multi-label BCE loss
- Per-class calibration
- Convergence metrics
"""

def compute_loss_metrics(
    predictions: torch.Tensor,
    targets_multihot: torch.Tensor,  # [num_samples, vocab_size]
    criterion: nn.Module,
    targets_list: Optional[List[List[int]]] = None
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
           - Important for rare disease detection
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
    num_codes_to_sample: int = 200  # Sample codes for efficiency
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
    if len(all_codes) > num_codes_to_sample:
        # Sample a subset of codes to be memory efficient
        all_codes = list(np.random.choice(all_codes, num_codes_to_sample, replace=False))
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


def compute_stratified_metrics(
    predictions: torch.Tensor,
    targets: List[List[int]],
    code_frequencies: np.ndarray,  # From training data
    vocab_size: int
) -> Dict[str, float]:
    """
    Stratified performance by code frequency(Rare Code Analysis)
    CRITICAL for clinical AI publication:
    - Medical codes have extreme long-tail distribution
    - Rare codes (sepsis, MI, rare diseases) are most important
    - Must show model doesn't just predict common codes

    Returns:
        1. Rare Code Performance:
           - Top-10 accuracy for codes in bottom 20% frequency
           - Critical for healthcare (rare = important)
        
        2. Common Code Performance:
           - Top-10 accuracy for codes in top 20% frequency
           - Shows model learns frequent patterns
        
        3. Tail Code Coverage:
           - What fraction of rare codes ever predicted in top-50?
           - Detects if model ignores rare codes entirely
    """
    metrics = {}
    
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
    
    print(f"  Tier sizes: common={len(common_codes)}, medium={len(medium_codes)}, "
          f"rare={len(rare_codes)}, tail={len(tail_codes)}") 
    
    top_10_preds = torch.topk(predictions, 10, dim=-1).indices  
    # Top-10 accuracy per tier
    # Single-pass: compute all tiers at once
    tier_correct = {'common': 0, 'medium': 0, 'rare': 0, 'tail': 0}
    tier_total = {'common': 0, 'medium': 0, 'rare': 0, 'tail': 0}
    tier_sets = {
        'common': common_codes, 
        'medium': medium_codes, 
        'rare': rare_codes, 
        'tail': tail_codes
    }

    for i, target_codes in enumerate(targets):
        pred_set = set(top_10_preds[i].tolist())  # Convert once per sample
        for tier_name, code_set in tier_sets.items():
            # Find target codes in this tier
            tier_true = [c for c in target_codes if c in code_set and c != 0]
            if len(tier_true) > 0:
                tier_total[tier_name] += 1
                # Check if any tier code is in top-10 predictions
                if any(c in pred_set for c in tier_true):
                    tier_correct[tier_name] += 1

    metrics['common_top10_acc'] = tier_correct['common'] / tier_total['common'] if tier_total['common'] > 0 else 0.0
    metrics['medium_top10_acc'] = tier_correct['medium'] / tier_total['medium'] if tier_total['medium'] > 0 else 0.0
    metrics['rare_top10_acc'] = tier_correct['rare'] / tier_total['rare'] if tier_total['rare'] > 0 else 0.0
    metrics['tail_top10_acc'] = tier_correct['tail'] / tier_total['tail'] if tier_total['tail'] > 0 else 0.0
    
    # Tail code coverage (what % of rare codes ever predicted)
    top_50_preds = torch.topk(predictions, 50, dim=-1).indices
    predicted_codes = set(top_50_preds.flatten().tolist())
    
    tail_coverage = len(predicted_codes & tail_codes) / len(tail_codes) if tail_codes else 0.0
    metrics['tail_code_coverage'] = tail_coverage
    
    # Frequency-weighted accuracy (prevents bias to common codes)
    # Give equal weight to each frequency tier
    tier_accs = [
        metrics['common_top10_acc'],
        metrics['medium_top10_acc'],
        metrics['rare_top10_acc'],
        metrics['tail_top10_acc']
    ]
    metrics['balanced_top10_acc'] = np.mean(tier_accs)
    
    return metrics

def compute_convergence_metrics(
    epoch_losses: List[float],  # Validation losses per epoch
    epoch_metrics: List[Dict[str, float]],  # All metrics per epoch
    smoothing_window: int = 3
) -> Dict[str, float]:
    """
    Convergence speed and stability metrics.
        Understanding training dynamics:
        - How fast does model learn?
        - Is training stable?
        - When do early-stop?    
    Returns:
        1. Convergence Speed:
           - Epochs to reach 95% of final performance
           - Area under learning curve
        
        2. Training Stability:
           - Loss variance across epochs
           - Number of loss spikes (increases)
        
        3. Early Stopping Point:
           - Optimal epoch (before overfitting)
           - Patience needed (for early stopping callback)
    """
    metrics = {}
    
    # Extract validation losses and top-10 accuracy
    val_losses = [epoch['val_loss'] for epoch in epoch_metrics]
    recall_at_10 = [epoch.get('recall@10', epoch.get('final_val_recall@10', 0.0)) for epoch in epoch_metrics]
    
    # 1. Convergence speed
    final_loss = val_losses[-1]
    target_loss = final_loss * 1.05  # Within 5% of final
    
    converged_epoch = len(val_losses)  # Default: never converged
    for i, loss in enumerate(val_losses):
        if loss <= target_loss:
            converged_epoch = i + 1
            break
    
    metrics['epochs_to_converge'] = converged_epoch
    metrics['convergence_rate'] = 1.0 / converged_epoch if converged_epoch > 0 else 0.0
    
    # 2. Training stability (smoothed loss variance)
    if len(val_losses) >= smoothing_window:
        smoothed = np.convolve(val_losses, np.ones(smoothing_window)/smoothing_window, mode='valid')
        metrics['loss_variance'] = np.var(smoothed)
        metrics['loss_stability'] = 1.0 / (1.0 + metrics['loss_variance'])  # Higher = more stable
    
    # Count loss spikes (validation loss increases)
    spikes = sum(1 for i in range(1, len(val_losses)) if val_losses[i] > val_losses[i-1])
    metrics['num_loss_spikes'] = spikes
    
    # 3. Optimal stopping point (best validation loss)
    best_epoch = np.argmin(val_losses) + 1
    best_loss = np.min(val_losses)
    metrics['best_epoch'] = best_epoch
    metrics['best_val_loss'] = best_loss
    metrics['overfitting_gap'] = val_losses[-1] - best_loss  # Positive = overfitting
    
    # Area under learning curve (lower = faster learning)
    # Normalize by final performance
    normalized_losses = [(loss - final_loss) for loss in val_losses]
    metrics['auc_learning_curve'] = np.trapz(normalized_losses)
    
    return metrics


def compute_ablation_metrics(
    all_experiment_results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Ablation analysis across experiments.
        - Component contribution
        - Architecture variants
        - Marginal benefit of each improvement    
    Returns:
        1. Component Contributions:
           - Flash Attention impact (exp2 vs exp1)
           - Learned pooling impact (exp2b vs exp2)
           - MoE impact (exp3 vs exp2)
        
        2. Interaction Effects:
           - Flash + MoE synergy
           - Learned pooling + MoE synergy
        
        3. Diminishing Returns:
           - Cost-benefit ratio for each component
    """
    metrics = {}
    
    # Baseline reference
    baseline = all_experiment_results.get('exp1_dense_baseline', {})
    baseline_acc = baseline.get('final_val_recall@10', 0)
    baseline_time = baseline.get('training_time_sec', 1)
    
    # ============================================================
    # 1. FLASH ATTENTION IMPACT
    # ============================================================
    flash_dense = all_experiment_results.get('exp2_dense_flash', {})
    if flash_dense:
        # Accuracy impact
        flash_acc_gain = flash_dense['final_val_recall@10'] - baseline_acc
        metrics['flash_attn_acc_gain'] = flash_acc_gain
        metrics['flash_attn_acc_gain_percent'] = (flash_acc_gain / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        # Speed impact
        flash_speedup = baseline_time / flash_dense['training_time_sec']
        metrics['flash_attn_speedup'] = flash_speedup
    
    # ============================================================
    # 2. LEARNED POOLING IMPACT
    # ============================================================
    flash_learned = all_experiment_results.get('exp2b_flash_learned_pool', {})
    if flash_dense and flash_learned:
        # Accuracy impact
        pool_acc_gain = flash_learned['final_val_recall@10'] - flash_dense['final_val_recall@10']
        metrics['learned_pool_acc_gain'] = pool_acc_gain
        
        # Speed impact
        pool_speedup = flash_dense['training_time_sec'] / flash_learned['training_time_sec']
        metrics['learned_pool_speedup'] = pool_speedup
    
    # ============================================================
    # 3. MOE IMPACT
    # ============================================================
    moe_standard = all_experiment_results.get('exp3_standard_moe', {})
    if flash_dense and moe_standard:
        # Accuracy impact
        moe_acc_gain = moe_standard['final_val_recall@10'] - flash_dense['final_val_recall@10']
        metrics['moe_acc_gain'] = moe_acc_gain
        
        # Efficiency: accuracy gain per parameter increase
        param_increase = moe_standard.get('parameters', 0) - flash_dense.get('parameters', 0)
        if param_increase > 0:
            metrics['moe_acc_per_param'] = moe_acc_gain / (param_increase / 1e6)  # Per million params
    
    # ============================================================
    # 4. INTERACTION EFFECTS
    # ============================================================
    moe_learned = all_experiment_results.get('exp3b_moe_learned_pool', {})
    if moe_standard and moe_learned:
        # Learned pooling benefit with MoE
        pool_with_moe_speedup = moe_standard['training_time_sec'] / moe_learned['training_time_sec']
        metrics['pool_moe_synergy_speedup'] = pool_with_moe_speedup
        
        # Compare to learned pooling benefit without MoE
        if flash_dense and flash_learned:
            pool_no_moe_speedup = flash_dense['training_time_sec'] / flash_learned['training_time_sec']
            # Synergy: does pooling help more with MoE?
            metrics['pool_moe_interaction'] = pool_with_moe_speedup - pool_no_moe_speedup
    
    # ============================================================
    # 5. COST-BENEFIT RATIO
    # ============================================================
    for exp_name, results in all_experiment_results.items():
        if exp_name == 'exp1_dense_baseline':
            continue
        
        # Accuracy improvement per dollar
        acc_gain = results['final_val_recall@10'] - baseline_acc
        cost = results.get('cost_usd', 0)
        
        if cost > 0:
            metrics[f'{exp_name}_acc_per_dollar'] = acc_gain / cost
        
        # Speedup vs cost ratio
        speedup = baseline_time / results['training_time_sec']
        metrics[f'{exp_name}_speedup_ratio'] = speedup
    
    return metrics



# #### Comprehensive evaluation metrics



def comprehensive_evaluation(
    model: nn.Module,
    val_dataloader: DataLoader,
    config: BaseConfig,
    device: torch.device,
    training_time_sec: float,
    epoch_history: List[Dict[str, float]],
    code_frequencies: np.ndarray,
    moe_config: Optional[MoEConfig] = None,
    use_mixed_precision: bool = False,
    max_samples_for_detailed_metrics = 10000,
    current_train_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, any]:
    """
    Comprehensive evaluation with all metrics.
    
    Returns dictionary organized by category:
    - performance: Task performance metrics
    - efficiency: Time and throughput metrics
    - cost: Resource usage and cost estimates
    - moe: MoE-specific metrics (if applicable)
    - sample up to max_samples_for_detailed_metrics to prevent CPU OOM.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    
    model.eval()

    is_wrapped = isinstance(model, DataParallelWrapper) or (
        isinstance(model, nn.DataParallel) and isinstance(model.module, DataParallelWrapper)
    )
    print("Computing streaming metrics (memory-safe)...")
    metrics_tracker = StreamingMetrics(
        k_values=(1, 5, 10),
        compute_mrr=True,
        compute_brier=True,
        vocab_size=config.target_cd_cnt
    )
    sampled_predictions = []
    sampled_targets = []
    sampled_multihot = []
    samples_collected = 0
    
    criterion = nn.BCEWithLogitsLoss()

    nbatch = len(val_dataloader)
    if nbatch == 0:
        print(" Validation set too small, skipping detailed metrics")
        return {
            'performance': {},
            'efficiency': {},
            'resources': {},
        }     
    with torch.no_grad():
        
        for batch_idx, batch in enumerate(val_dataloader): # ← Iterate over DataLoader
            if batch_idx % 200 == 0:
                print(f"  Processing batch {batch_idx}/{len(val_dataloader)}...")
            age = batch['age'].to(device)
            gender = batch['gender'].to(device)
            lob = batch['lob'].to(device)
            codes = batch['codes'].to(device)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1), 
                codes
            ], dim=-1)
            
            # Forward
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.float16) 
                if use_mixed_precision else nullcontext()
            )
            
            with autocast_ctx:
                if is_wrapped:
                    targets_mh = batch['target_multihot'].to(device, non_blocking=True)
                    dt_cnt_tensor = (
                        dt_cnt.to(device) if isinstance(dt_cnt, torch.Tensor) 
                        else torch.tensor(dt_cnt, device=device)
                    )
                    result = model(x, dt_cnt_tensor, targets_mh, return_predictions=True)
                    if isinstance(result, tuple):
                        loss_val, extras = result = result
                        output = extras.get('predictions')
                    else:
                        loss_val = result
                        output = None
                    # record loss
                    if loss_val is not None:
                        batch_loss = loss_val.mean().item() if loss_val.numel() > 1 else loss_val.item()
                        metrics_tracker.update_loss(batch_loss)
                        
                else:
                    if _model_has_moe(model):
                        output, _ = model(x, return_moe_losses=False)
                    else:
                        output = model(x)
                        
            if output is None: continue
            
            # Get actual batch size from output
            batch_size_actual = output.shape[0]
            # Get actual length of day
            actual_len_dy = output.shape[1]
            # Process outputs
            output_flat = output.reshape(batch_size_actual * actual_len_dy, config.target_cd_cnt)
            y_flat = [item for sublist in y for item in sublist]
            dt_cnt_values = dt_cnt.cpu().tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
            
            # Collect valid predictions
            valid_outputs = []
            valid_targets = []
            
            # Filter valid days
            for j in range(batch_size_actual):
                valid_days = min(dt_cnt[j].item(), actual_len_dy)
                if valid_days <= 0:
                    continue
                out_start = actual_len_dy * j
                out_end = out_start + valid_days
                valid_outputs.append(output_flat[out_start:out_end])
                
                y_start = config.len_dy * j
                y_end = y_start + valid_days
                valid_targets.extend(y_flat[y_start:y_end])
                
            if valid_outputs:
                batch_preds = torch.cat(valid_outputs)

                # Update streaming metrics (ALL data)
                metrics_tracker.update(batch_preds, valid_targets)

                # Update loss for non-wrapped models
                if not is_wrapped:
                    # Create multi-hot targets for this batch
                    targets_multihot = create_multihot_targets_vectorized(
                        valid_targets, len(batch_preds), config.target_cd_cnt, device
                    )
                    batch_loss = criterion(batch_preds, targets_multihot).item()
                    metrics_tracker.update_loss(batch_loss)
                
                # Sample for detailed metrics for memory efficiency
                if samples_collected < max_samples_for_detailed_metrics:
                    remaining = max_samples_for_detailed_metrics - samples_collected
                    to_take = min(len(batch_preds), remaining)

                    sampled_predictions.append(batch_preds[:to_take].cpu())
                    sampled_targets.extend(valid_targets[:to_take])

                    # Create multihot for sampled
                    multihot = create_multihot_targets_vectorized(
                        valid_targets[:to_take], to_take, config.target_cd_cnt, device
                    )
                    sampled_multihot.append(multihot.cpu())
                    samples_collected += to_take

            # Memory cleanup
            del output, output_flat, valid_outputs
            if batch_idx % 100 == 0:
                gc.collect()
    
    # ============================================================
    # COMPUTE ALL METRICS
    # ============================================================
    streaming_results = metrics_tracker.compute()
    print(f"Computing detailed metrics on {samples_collected} sampled predictions...")
    evaluation = {}
    performance_metrics = {k: v for k, v in streaming_results.items() 
                           if k not in ['num_samples', 'num_batches']}

    # Ensure all keys expected by _build_epoch_metrics() are present
    # The streaming metrics should already have these, but ensure defaults
    # resuse the streaming results
    required_keys = {
        'val_loss': performance_metrics.get('val_loss', 0.0),
        'recall@1': performance_metrics.get('recall@1', 0.0),
        'recall@5': performance_metrics.get('recall@5', 0.0),
        'recall@10': performance_metrics.get('recall@10', 0.0),
        'recall@20': performance_metrics.get('recall@20', 0.0),
        'micro_recall@10': performance_metrics.get('micro_recall@10', 0.0),
        'micro_recall@20': performance_metrics.get('micro_recall@20', 0.0),
        'ndcg@10': performance_metrics.get('ndcg@10', 0.0),
        'ndcg@20': performance_metrics.get('ndcg@20', 0.0),
        'mrr': performance_metrics.get('mrr', 0.0),
        'positive_brier': performance_metrics.get('positive_brier', 0.0),
    }    
    # Merge (required_keys first so they're guaranteed, then rest)
    performance_metrics = {**required_keys, **performance_metrics}
    # here only add detailed metrics
    if sampled_predictions:
        all_predictions = torch.cat(sampled_predictions)
        all_targets_multihot = torch.cat(sampled_multihot)
        performance_metrics.update(
            compute_primary_task_metrics(all_predictions, sampled_targets, config.target_cd_cnt)
        )
        performance_metrics.update(
            compute_loss_metrics(all_predictions, all_targets_multihot, criterion, sampled_targets)
        )
        performance_metrics.update(
            compute_stratified_metrics(all_predictions, sampled_targets, code_frequencies, config.target_cd_cnt)
        )
        performance_metrics.update(
            compute_auroc_auprc(all_predictions, sampled_targets, config.target_cd_cnt)
        )
        # Cleanup
        del all_predictions, all_targets_multihot, sampled_predictions, sampled_multihot
        gc.collect()
        
    evaluation['performance'] = performance_metrics    
    
    # 2. EFFICIENCY METRICS
    print("Computing efficiency metrics...")
    num_samples = streaming_results.get('num_samples', 0)
    num_tokens = num_samples * config.len_dy
    actual_epochs = len(epoch_history) + 1 if current_train_metrics is not None else max(len(epoch_history), 1)
    evaluation['efficiency'] = compute_training_time_metrics(
        total_train_time=training_time_sec,
        num_epochs=actual_epochs,
        num_samples=num_samples * actual_epochs,
        num_tokens=num_tokens * actual_epochs,
        batch_size=config.batch_size
    )
    
    print("Computing resource metrics...")
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 1
    
    evaluation['resources'] = {
        **compute_memory_metrics(device, model, config.batch_size, config.len_dy, num_gpus),
        **compute_flops_metrics(
            config, config.batch_size, config.len_dy,
            num_experts=moe_config.num_experts if moe_config else None,
            top_k=moe_config.top_k if moe_config else None,
            use_moe_from_layer=moe_config.use_moe_from_layer if moe_config else None,
            actual_throughput=evaluation['efficiency'].get('tokens_per_sec', 0)
        ),
        **compute_cost_metrics(training_time_sec, len(epoch_history), "T4", num_gpus)
    }
    
    # 4. MOE METRICS (if applicable)
    if moe_config:
        print("Computing MoE metrics...")
        if current_train_metrics is not None:
            expert_usage = current_train_metrics.get('expert_usage', None)
        elif epoch_history:
            expert_usage = epoch_history[-1].get('expert_usage', None)
        else:
            expert_usage = None
            
        if expert_usage:
            evaluation['moe'] = compute_moe_performance_metrics(
                expert_usage=expert_usage,
                router_probs_history=[],
                num_experts=moe_config.num_experts - moe_config.num_shared_experts
            )
            
    if current_train_metrics is not None:
        grad_tier_data = {
            'common_frac': current_train_metrics.get('gradient_tier_common_frac_final', None),
            'tail_frac': current_train_metrics.get('gradient_tier_tail_frac_final', None),
        }
        if grad_tier_data['common_frac'] is not None:
            evaluation['gradient_tier'] = grad_tier_data
            
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
    return evaluation


# #### Streaming Metrics for evaluation


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple



# --- Streaming metrics ---

@dataclass
class StreamingMetricsState:
    """Internal state for streaming metrics computation."""
    # Loss
    total_loss: float = 0.0
    num_batches: int = 0
    
    # Sample counts
    num_samples: int = 0
    
    # Recall@K counters (binary: any hit in top-K)
    recall_hits: Dict[int, int] = field(default_factory=dict)
    recall_total: Dict[int, int] = field(default_factory=dict)
    
    # Micro-Recall@K (per-code: total hits / total true codes)
    micro_recall_hits: Dict[int, int] = field(default_factory=dict)
    micro_recall_true: Dict[int, int] = field(default_factory=dict)
    
    # Precision@K (sum for average)
    precision_sum: Dict[int, float] = field(default_factory=dict)
    precision_count: Dict[int, int] = field(default_factory=dict)
    
    # NDCG@K
    ndcg_sum: Dict[int, float] = field(default_factory=dict)
    ndcg_count: Dict[int, int] = field(default_factory=dict)
    
    # MRR (best-ranked true code)
    mrr_sum: float = 0.0
    mrr_count: int = 0
    
    # Positive-only Brier
    positive_brier_sum: float = 0.0
    positive_brier_count: int = 0
class StreamingMetrics:
    """
    Memory-efficient streaming metrics aggregator for evaluation
    Computes metrics incrementally without storing all predictions.
    implemented in round 5 experimentation due to memory constraints
    Follows HuggingFace/PyTorch patterns:
    - `.reset()` clears state
    - `.update(batch)` processes one batch  
    - `.compute()` returns final metrics
    
    Key design principles:
    1. never accumulate full predictions (prevents CPU OOM)
    2. Compute metrics incrementally per-batch
    3. Use only scalar counters, not tensors    
    Usage:
        metrics = StreamingMetrics()
        for batch in dataloader:
            metrics.update(predictions, targets, loss)
        results = metrics.compute()
    """
    
    def __init__(
        self, 
        k_values: Tuple[int, ...] = (5, 10, 20),
        compute_mrr: bool = True,
        compute_brier: bool = True,
        vocab_size: int = 6297
    ):
        """
        Args:
            k_values: K values for Recall@K, Precision@K, Micro-Recall@K, NDCG@K
            compute_mrr: Whether to compute Mean Reciprocal Rank
            compute_brier: Whether to compute Positive-Only Brier Score
            vocab_size: Target vocabulary size (for Brier calculation)
        """
        self.k_values = k_values
        self.compute_mrr = compute_mrr
        self.compute_brier = compute_brier
        self.vocab_size = vocab_size
        self._max_k = max(k_values)
        
        # Precompute NDCG discount factors: 1/log2(rank+2)
        self._discounts = 1.0 / np.log2(np.arange(2, self._max_k + 2))
        # Precompute cumulative sums for IDCG (used in vectorized NDCG)
        self._discount_cumsum = np.cumsum(self._discounts)
        # Cache for device-specific tensors (lazy initialization)
        self._cached_device = None
        self._discounts_tensor = None
        self._discount_cumsum_tensor = None        
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulators for a new evaluation pass."""
        self._state = StreamingMetricsState(
            recall_hits={k: 0 for k in self.k_values},
            recall_total={k: 0 for k in self.k_values},
            micro_recall_hits={k: 0 for k in self.k_values},
            micro_recall_true={k: 0 for k in self.k_values},
            precision_sum={k: 0.0 for k in self.k_values},
            precision_count={k: 0 for k in self.k_values},
            ndcg_sum={k: 0.0 for k in self.k_values},
            ndcg_count={k: 0 for k in self.k_values},
        )
        
    def _get_tensors_for_device(self, device: torch.device):
        """Lazy initialization of device-specific tensors."""
        if self._cached_device != device:
            self._discounts_tensor = torch.tensor(
                self._discounts, dtype=torch.float32, device=device
            )
            self._discount_cumsum_tensor = torch.tensor(
                self._discount_cumsum, dtype=torch.float32, device=device
            )
            self._cached_device = device
        return self._discounts_tensor, self._discount_cumsum_tensor  
    
    def update_loss(self, loss: float) -> None:
        """Accumulate loss from a batch."""
        self._state.total_loss += loss
        self._state.num_batches += 1
    
    def update(
        self,
        predictions: torch.Tensor,  # [batch_size, vocab_size] logits
        targets: List[List[int]],    # List of target code lists per sample
        probs: Optional[torch.Tensor] = None  # [batch_size, vocab_size] for Brier
    ) -> None:
        """
        Update metrics with a batch of predictions.
        
        Args:
            predictions: Model logits [batch_size, vocab_size] (will be moved to CPU)
            targets: List of target code lists (one per sample)
            probs: Optional sigmoid probabilities for Brier score
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        # Get device-cached tensors for NDCG
        discounts_tensor, discount_cumsum_tensor = self._get_tensors_for_device(device)
        
        # ================================================================
        # STEP 1: Get top-K predictions (ONCE for all metrics)
        # ================================================================
        with torch.no_grad():
            # topk is O(n) vs argsort O(n log n)
            _, top_k_indices = torch.topk(predictions, self._max_k, dim=-1)
            # Keep on GPU for gather, move to CPU only for final extraction

            # Compute probs for Brier if needed (keep on GPU for vectorized Brier)
            probs_tensor = None
            if self.compute_brier:
                if probs is None:
                    probs_tensor = torch.sigmoid(predictions)
                else:
                    probs_tensor = probs

        # ================================================================
        # STEP 2: Build target tensor for vectorized comparison
        # ================================================================
        # Create boolean target tensor [batch_size, vocab_size]
        target_tensor = torch.zeros(batch_size, self.vocab_size, dtype=torch.bool, device=device)

        # Track which samples have valid targets (for filtering)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        num_true_per_sample = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Build target tensor (this loop is unavoidable but fast - just indexing)
        for i, target_codes in enumerate(targets):
            valid_codes = [c for c in target_codes if 0 < c < self.vocab_size]
            if valid_codes:
                target_tensor[i, valid_codes] = True
                valid_mask[i] = True
                num_true_per_sample[i] = len(valid_codes)

        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            return

        self._state.num_samples += num_valid

        # ================================================================
        # STEP 3: Vectorized hit detection for Recall, Precision, Micro-Recall
        # ================================================================
        for k in self.k_values:
            top_k = top_k_indices[:, :k]  # [batch, k]

            # Gather target values at predicted positions: [batch, k]
            hits_matrix = torch.gather(target_tensor, 1, top_k)
            hits_per_sample = hits_matrix.sum(dim=1)  # [batch]

            # Only count valid samples
            valid_hits = hits_per_sample[valid_mask]
            valid_num_true = num_true_per_sample[valid_mask]

            # Recall@K (binary: any hit = success)
            self._state.recall_hits[k] += (valid_hits > 0).sum().item()
            self._state.recall_total[k] += num_valid

            # Micro-Recall@K (per-code: total hits / total true codes)
            self._state.micro_recall_hits[k] += valid_hits.sum().item()
            self._state.micro_recall_true[k] += valid_num_true.sum().item()

            # Precision@K: hits / k (averaged over samples)
            precision_per_sample = valid_hits.float() / k
            self._state.precision_sum[k] += precision_per_sample.sum().item()
            self._state.precision_count[k] += num_valid

        # ================================================================
        # STEP 4: Vectorized NDCG@K
        # ================================================================
        for k in self.k_values:
            top_k = top_k_indices[:, :k]  # [batch, k]

            # hits_matrix: [batch, k] - True where prediction is correct
            hits_matrix = torch.gather(target_tensor, 1, top_k).float()

            # DCG: sum of discounts where hits occur
            # discounts[:k] is [k], hits_matrix is [batch, k]
            dcg_per_sample = (hits_matrix * discounts_tensor[:k]).sum(dim=1)  # [batch]

            # IDCG: sum of first min(num_true, k) discounts
            # For each sample, IDCG = sum(discounts[:min(num_true, k)])
            valid_num_true_k = torch.clamp(num_true_per_sample, max=k)  # [batch]

            # VECTORIZED IDCG: index into cumsum (handle 0 case)
            # discount_cumsum_tensor[i-1] for i > 0, else 0
            idcg_indices = (valid_num_true_k - 1).clamp(min=0)  # [batch]
            idcg_per_sample = torch.where(
                valid_num_true_k > 0,
                discount_cumsum_tensor[idcg_indices],
                torch.zeros(batch_size, device=device)
            )

            # NDCG = DCG / IDCG (avoid div by zero)
            ndcg_per_sample = torch.where(
                idcg_per_sample > 0,
                dcg_per_sample / idcg_per_sample,
                torch.zeros_like(dcg_per_sample)
            )

            # Only count valid samples
            self._state.ndcg_sum[k] += ndcg_per_sample[valid_mask].sum().item()
            self._state.ndcg_count[k] += num_valid

        # ================================================================
        # STEP 5: Vectorized MRR (Mean Reciprocal Rank)
        # ================================================================
        if self.compute_mrr:
            # hits_matrix for max_k: [batch, max_k]
            hits_matrix = torch.gather(target_tensor, 1, top_k_indices).float()

            # Find first hit position for each sample
            # Use argmax on cumsum - first True gives cumsum=1
            cumsum_hits = hits_matrix.cumsum(dim=1)  # [batch, max_k]

            # First hit position: where cumsum first becomes 1
            # If no hits, argmax returns 0 but we need to check separately
            has_any_hit = hits_matrix.sum(dim=1) > 0  # [batch]

            # Get rank of first hit (0-indexed)
            # argmax on (cumsum >= 1) gives first position where True
            first_hit_mask = cumsum_hits >= 1
            # Set False entries to max_k so they don't affect argmax
            first_hit_positions = first_hit_mask.float().argmax(dim=1)  # [batch]

            # Reciprocal rank: 1 / (rank + 1), only for samples with hits
            reciprocal_ranks = torch.where(
                has_any_hit & valid_mask,
                1.0 / (first_hit_positions.float() + 1),
                torch.zeros(batch_size, device=device)
            )

            self._state.mrr_sum += reciprocal_ranks.sum().item()
            self._state.mrr_count += num_valid

        # ================================================================
        # STEP 6: Vectorized Positive-Only Brier Score
        # ================================================================
        if self.compute_brier and probs_tensor is not None:
            # Compute Brier only at positive positions: (p - 1)^2
            # target_tensor is [batch, vocab], probs_tensor is [batch, vocab]
            brier_contrib = ((probs_tensor - 1.0) ** 2) * target_tensor.float()

            # Sum over all positive positions
            self._state.positive_brier_sum += brier_contrib.sum().item()
            self._state.positive_brier_count += target_tensor.sum().item()

        # Cleanup
        del target_tensor, top_k_indices
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated state.
        
        Returns:
            Dict with all computed metrics
        """
        metrics = {}
        
        # Loss
        metrics['val_loss'] = (
            self._state.total_loss / max(self._state.num_batches, 1)
        )
        
        # Recall@K
        for k in self.k_values:
            metrics[f'recall@{k}'] = (
                self._state.recall_hits[k] / self._state.recall_total[k]
                if self._state.recall_total[k] > 0 else 0.0
            )
        
        # Micro-Recall@K
        for k in self.k_values:
            metrics[f'micro_recall@{k}'] = (
                self._state.micro_recall_hits[k] / self._state.micro_recall_true[k]
                if self._state.micro_recall_true[k] > 0 else 0.0
            )
        
        # Precision@K
        for k in self.k_values:
            metrics[f'precision@{k}'] = (
                self._state.precision_sum[k] / self._state.precision_count[k]
                if self._state.precision_count[k] > 0 else 0.0
            )
        
        # NDCG@K
        for k in self.k_values:
            metrics[f'ndcg@{k}'] = (
                self._state.ndcg_sum[k] / self._state.ndcg_count[k]
                if self._state.ndcg_count[k] > 0 else 0.0
            )
        
        # MRR
        if self.compute_mrr:
            metrics['mrr'] = (
                self._state.mrr_sum / self._state.mrr_count
                if self._state.mrr_count > 0 else 0.0
            )
        
        # Positive-Only Brier
        if self.compute_brier:
            metrics['positive_brier'] = (
                self._state.positive_brier_sum / self._state.positive_brier_count
                if self._state.positive_brier_count > 0 else 0.0
            )
        
        # Metadata
        metrics['num_samples'] = self._state.num_samples
        metrics['num_batches'] = self._state.num_batches
        
        return metrics
    
    def __repr__(self) -> str:
        return (
            f"StreamingMetrics("
            f"k_values={self.k_values}, "
            f"samples={self._state.num_samples}, "
            f"batches={self._state.num_batches})"
        )


# #### Test


def test_streaming_metrics_quick():
    """Quick test for StreamingMetrics."""
    metrics = StreamingMetrics(k_values=(5, 10, 20))
    predictions = torch.randn(32, 6297)
    targets = [[i % 100, (i+1) % 100] for i in range(32)]
    metrics.update(predictions, targets)
    results = metrics.compute()
    print(results)
    print("StreamingMetrics quick test ✔️")

# Uncomment to run:
test_streaming_metrics_quick()



def test_metric_utilities():
    vocab = 50
    predictions = torch.randn(32, vocab)
    targets = [[i % vocab] for i in range(32)]
    multihot = torch.randint(0, 2, (32, vocab)).float()
    freq = np.random.randint(1, 100, size=vocab)

    primary = compute_primary_task_metrics(predictions, targets, vocab)
    losses = compute_loss_metrics(predictions, multihot, nn.BCEWithLogitsLoss())
    stratified = compute_stratified_metrics(predictions, targets, freq, vocab)

    assert 'recall@10' in primary
    assert 'bce_loss' in losses
    assert 'tail_top10_acc' in stratified
    print("Metric utilities ✔️")
test_metric_utilities()



def test_comprehensive_evaluation_dense():
    config = BaseConfig(batch_size=4, len_dy=200, len_cd=80, device=device.type)
    model = BaselineTransformer(config).to(device)
    criterion = nn.BCEWithLogitsLoss()

    train_subset = df_train.head(config.batch_size*2)
    val_subset = df_val.head(config.batch_size)
    epoch_history = [{'val_loss': 1.0, 'recall@10': 0.1}]
    code_freq = np.ones(config.target_cd_cnt, dtype=np.int32)

    # Create DataLoader for validation
    val_dataset = ClinicalDataset(val_subset, config)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        collate_fn=create_collate_fn(config)
    )

    previous = globals().get('config')
    globals()['config'] = config  # required by compute_training_time_metrics

    evaluation = comprehensive_evaluation(
        model=model,
        val_dataloader=val_loader,  # ← Pass DataLoader instead of DataFrame
        config=config,
        device=device,
        training_time_sec=1.0,
        epoch_history=epoch_history,
        code_frequencies=code_freq,
        moe_config=None,
        use_mixed_precision=False
    )

    assert 'performance' in evaluation
    assert 'efficiency' in evaluation
    print("comprehensive_evaluation (dense) ✔️")

    if previous is not None:
        globals()['config'] = previous
    else:
        del globals()['config']
test_comprehensive_evaluation_dense()


# ### Extract embedding for each member


# ============================================================================
# EMBEDDING EXTRACTOR (Pythonic hook-based approach)
# ============================================================================



# --- GPU memory tracking ---

class GPUMemoryTracker:
    """
    Track GPU memory at different stages of training.
    
    Usage:
        tracker = GPUMemoryTracker()
        
        # In training loop:
        tracker.record("before_forward")
        output = model(x)
        tracker.record("after_forward")
        loss.backward()
        tracker.record("after_backward")
        optimizer.step()
        tracker.record("after_optimizer")
        
        # Print summary:
        tracker.print_summary()
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.enabled else 0
        self.records = {}  # {stage_name: {gpu_id: (allocated, reserved, peak)}}
        
    def record(self, stage_name: str):
        """Record memory usage at a specific stage."""
        if not self.enabled:
            return
            
        # Synchronize to ensure all GPU operations are complete
        torch.cuda.synchronize()
        
        self.records[stage_name] = {}
        for gpu_id in range(self.num_gpus):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
            self.records[stage_name][gpu_id] = (allocated, reserved, peak)
    
    def reset_peak(self):
        """Reset peak memory stats for fresh measurement."""
        if self.enabled:
            for gpu_id in range(self.num_gpus):
                torch.cuda.reset_peak_memory_stats(gpu_id)
    
    def print_stage(self, stage_name: str):
        """Print memory for a single stage."""
        if stage_name not in self.records:
            print(f"No record for stage: {stage_name}")
            return
            
        print(f"\n📊 GPU Memory @ {stage_name}:")
        for gpu_id, (alloc, res, peak) in self.records[stage_name].items():
            print(f"   GPU {gpu_id}: {alloc:.2f}GB allocated, {res:.2f}GB reserved, {peak:.2f}GB peak")
    
    def print_gpu_use_summary(self):
        """Print all recorded stages."""
        if not self.records:
            print("No GPU memory records.")
            return
            
        print("\n" + "="*70)
        print("GPU MEMORY SUMMARY")
        print("="*70)
        
        # Header
        stages = list(self.records.keys())
        print(f"{'GPU':<6}", end="")
        for stage in stages:
            print(f"{stage:<20}", end="")
        print()
        print("-"*70)
        
        # Per-GPU rows
        for gpu_id in range(self.num_gpus):
            print(f"GPU {gpu_id:<2}", end="")
            for stage in stages:
                alloc, _, _ = self.records[stage][gpu_id]
                print(f"{alloc:>6.2f}GB             ", end="")
            print()
        
        print("="*70)


# ### Memory management


import torch
import gc


# --- Experiment orchestration helpers ---

def _calculate_model_dimensions(embedding_size: int, 
                                use_swiglu: bool = False) -> dict:
    """
    Calculate optimal nhead and nhid based on embedding_size and industry best practices.
    
    Industry Standards:
    - nhead: Chosen to give head_dim of 32, 64, or 128 (optimal for Flash Attention)
    - nhid: For standard FFN: 4x embedding_size
            For SwiGLU: ~8/3 x embedding_size (LLaMA, PaLM standard)
    
    Args:
        embedding_size: Model dimension (d_model)
        use_swiglu: Whether using SwiGLU activation (True for Flash/MoE models)
    
    Returns:
        dict with 'nhead' and 'nhid'
    """
    # ============================================================
    # NHEAD CALCULATION (Target head_dim: 64 preferred, 32 or 128 acceptable)
    # ============================================================
    if embedding_size <= 256:
        # 256 / 8 = 32 (good for Flash Attention)
        nhead = 8
    elif embedding_size <= 512:
        # 512 / 8 = 64 (optimal for Flash Attention)
        nhead = 8
    elif embedding_size <= 768:
        # 768 / 12 = 64 (BERT-base standard)
        nhead = 12
    elif embedding_size <= 1024:
        # 1024 / 16 = 64 (optimal)
        nhead = 16
    elif embedding_size <= 2048:
        # 2048 / 32 = 64 (optimal)
        nhead = 32
    else:
        # For very large models, target head_dim=128
        nhead = embedding_size // 128
    
    # ============================================================
    # NHID CALCULATION
    # ============================================================
    if use_swiglu:
        # SwiGLU uses 3 projections instead of 2
        # To match parameter count: nhid_swiglu = (2/3) * nhid_standard
        # Standard: nhid = 4 * d_model
        # SwiGLU: nhid = (2/3) * 4 * d_model = (8/3) * d_model ≈ 2.67 * d_model
        # Round to nearest multiple of 64 for memory alignment
        nhid_raw = int((8 / 3) * embedding_size)
        nhid = ((nhid_raw + 63) // 64) * 64  # Round up to multiple of 64
    else:
        # Standard FFN: 4x expansion (Vaswani et al. 2017)
        nhid = 4 * embedding_size
    
    return {
        'nhead': nhead,
        'nhid': nhid,
        'head_dim': embedding_size // nhead  # For logging
    }



# ============================================================================
# EXPERIMENT HELPER FUNCTIONS
# ============================================================================


def _setup_experiment_directories(
    log_dir: str,
    experiment_round: Optional[str],
    exp_name: str,
    checkpoint_dir: Optional[str]
) -> Tuple[str, str]:
    """Set up logging and checkpoint directories."""
    if experiment_round is not None:
        effective_log_dir = os.path.join(log_dir, experiment_round)
    else:
        datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        effective_log_dir = os.path.join(log_dir, f"exp_{datetime_string}")
    
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(effective_log_dir, exp_name, 'checkpoints')
    
    return effective_log_dir, checkpoint_dir



def _create_model(
    exp_name: str,
    eff_d_model: int,
    eff_nhid: int,
    eff_nhead: int,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> Tuple[nn.Module, BaseConfig, Optional[MoEConfig], bool, bool]:
    """
    Create model based on experiment type.
    
    Returns:
        model, config, use_mixed_precision, use_bucketing
    """
    is_baseline = exp_name == 'exp1_dense_baseline'
    is_flash_dense = exp_name in ['exp2_dense_flash', 'exp2b_flash_learned_pool']
    is_moe = moe_config is not None and not is_baseline and not is_flash_dense
    
    if is_baseline:
        config = BaseConfig(embedding_size=eff_d_model, nhid=eff_nhid)
        model = BaselineTransformer(config).to(device)
        use_mixed_precision = False
        use_bucketing = False
        moe_config_out = None
        if logger:
            logger.info(f"Model: Baseline Transformer (FP32)")
            logger.info(f"  d_model={eff_d_model}, nhid={eff_nhid}, nhead=16 (hardcoded)")
            
    elif is_flash_dense:
        config = FlashAttentionConfig(
            embedding_size=eff_d_model,
            nhid=eff_nhid,
            nhead=eff_nhead,
            use_swiglu=True,
            dtype=torch.float16,
            use_learnt_att_pool=use_learnt_att_pool
        )
        model = FlashAttentionTransformer(config).to(device)
        use_mixed_precision = True
        use_bucketing = True
        moe_config_out = None
        if logger:
            pooling_str = "Learned Attention Pooling" if use_learnt_att_pool else "Flash Attention + Max-Pool"
            logger.info(f"Model: Flash Attention Transformer (FP16)")
            logger.info(f"  d_model={eff_d_model}, nhid={eff_nhid}, nhead={eff_nhead}")
            logger.info(f"  Daily Encoder: {pooling_str}")
            
    else:
        # MoE variant
        config = FlashAttentionConfig(
            embedding_size=eff_d_model,
            nhid=eff_nhid,
            nhead=eff_nhead,
            use_swiglu=True,
            dtype=torch.float16,
            use_learnt_att_pool=use_learnt_att_pool
        )
        
        if moe_config:
            import copy
            moe_config_out = copy.deepcopy(moe_config)
            moe_config_out.d_model = eff_d_model
            moe_config_out.d_ff = eff_nhid
        else:
            moe_config_out = None
            
        model = FlashMoETransformer(config, moe_config_out).to(device)
        use_mixed_precision = True
        use_bucketing = True
        if logger:
            pooling_str = "Learned Attention Pooling" if use_learnt_att_pool else "Flash Attention + Max-Pool"
            logger.info(f"Model: Flash + MoE Transformer (FP16)")
            logger.info(f"  d_model={eff_d_model}, nhid={eff_nhid}, nhead={eff_nhead}")
            logger.info(f"  Daily Encoder: {pooling_str}")
            logger.info(f"  MoE: {moe_config_out.num_experts} experts, top-{moe_config_out.top_k}")
            logger.info(f"  MoE d_ff: {moe_config_out.d_ff}") 
    
    return model, config, moe_config_out, use_mixed_precision, use_bucketing


# --- Data preparation ---

def prepare_data_once(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: Optional[BaseConfig] = None,
    device: torch.device = None,
    code_freq_sample_fraction: float = 1.0,
    use_lazy: bool = False
) -> PreparedData:
    """
    Prepare datasets and code frequencies ONCE for reuse across multiple experiments.
    
    This is the expensive operation that should only run once.
    All experiments can then share the prepared data.
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame  
        config: BaseConfig with data dimensions (len_dy, len_cd, target_cd_cnt)
                If None, uses default BaseConfig()
        device: Torch device (for code frequency computation)
        code_freq_sample_fraction: Fraction of data to use for code frequency (1.0 = all)
    
    Returns:
        PreparedData containing:
        - train_dataset: Pre-processed ClinicalDataset
        - val_dataset: Pre-processed ClinicalDataset
        - code_frequencies: Pre-computed code frequencies
        - config: The config used
    
    Example:
        # Prepare once
        prepared = prepare_data_once(df_train, df_val, device=device)
        
        # Run multiple experiments with prepared data
        for exp_name in ['exp1_dense_baseline', 'exp2_dense_flash', 'exp3_standard_moe']:
            results = run_single_experiment(
                exp_name=exp_name,
                prepared_data=prepared,  # Reuse prepared data
                ...
            )
    """
    import gc
    if config is None:
        config = BaseConfig()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*80)
    print("PREPARING DATA (ONE-TIME OPERATION)")
    print("="*80)
    start_time = time.time()
    
    # ============================================================
    # STEP 1: Create Datasets
    # ============================================================
    DatasetClass = ClinicalDatasetLazy if use_lazy else ClinicalDataset
    
    print(f"\n[1/3] Creating training dataset ({'lazy' if use_lazy else 'eager'})...")
    train_dataset = DatasetClass(train_data, config)
    del train_data
    gc.collect()    
    print(f"\n[2/3] Creating validation dataset ({'lazy' if use_lazy else 'eager'})...")
    val_dataset = DatasetClass(val_data, config)
    del val_data
    gc.collect()    
    # ============================================================
    # STEP 2: Compute Code Frequencies
    # ============================================================
    print("\n[3/3] Computing code frequencies...")
    if use_lazy:
        code_frequencies = _compute_code_frequencies_from_strings(
            train_dataset.target_strs, config,
            sample_fraction=code_freq_sample_fraction
        )
    else:
        code_frequencies = _compute_code_frequencies_from_dataset(
            train_dataset, config,
            sample_fraction=code_freq_sample_fraction
        )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Data preparation complete in {elapsed:.1f}s")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples: {len(val_dataset):,}")
    print(f"   Unique codes: {np.sum(code_frequencies > 0):,}")
    print("="*80 + "\n")
    
    return PreparedData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        code_frequencies=code_frequencies,
        config=config
    )



def _compute_code_frequencies_from_dataset(
    dataset: ClinicalDataset,
    config: BaseConfig,
    sample_fraction: float = 1.0
) -> np.ndarray:
    """
    Compute code frequencies directly from a ClinicalDataset (no re-parsing).
    
    This is much faster because:
    - Uses already-parsed dataset (no string parsing)
    - No DataLoader overhead for simple iteration
    """
    code_frequencies = np.zeros(config.target_cd_cnt, dtype=np.int64)
    train_code_counts = Counter()
    
    n_samples = len(dataset)
    if sample_fraction < 1.0:
        n_samples = int(n_samples * sample_fraction)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
    else:
        indices = range(n_samples)
    
    print(f"  Processing {n_samples:,} samples...")
    
    for idx_count, idx in enumerate(indices):
        item = dataset[idx]
        target_list = item['target']  # List[List[int]]
        
        for day_codes in target_list:
            for code in day_codes:
                if code != 0:
                    train_code_counts[code] += 1
        
        if (idx_count + 1) % 100000 == 0:
            print(f"    Processed {idx_count + 1:,}/{n_samples:,} samples...")
    
    # Convert Counter to array
    for code_idx, count in train_code_counts.items():
        if 0 <= code_idx < config.target_cd_cnt:
            code_frequencies[code_idx] = count
    
    non_zero_codes = np.sum(code_frequencies > 0)
    print(f"  ✅ Found {non_zero_codes:,} unique codes")
    
    return code_frequencies



def _compute_code_frequencies_from_strings(
    target_strs: list,
    config: BaseConfig,
    sample_fraction: float = 1.0
) -> np.ndarray:
    """
    Compute code frequencies directly from raw target strings.
    Used with ClinicalDatasetLazy to avoid triggering full __getitem__ parsing.
    
    Matches _compute_code_frequencies_from_dataset behavior:
    - Skips code_idx=0 (consistent with `if code != 0` in the existing function)
    - 0-indexed code values from conv_target's code_val-1 mapping
    """
    code_frequencies = np.zeros(config.target_cd_cnt, dtype=np.int64)
    
    n = len(target_strs)
    if sample_fraction < 1.0:
        n_process = int(n * sample_fraction)
        indices = np.random.choice(n, n_process, replace=False)
    else:
        n_process = n
        indices = range(n)
    
    print(f"  Computing code frequencies from {n_process:,} target strings...")
    
    for count, idx in enumerate(indices):
        target_str = target_strs[idx]
        if not target_str or pd.isna(target_str):
            continue
        
        for day_str in target_str.split('*')[:config.len_dy]:
            if not day_str:
                continue
            for code_str in day_str.split(','):
                try:
                    code_val = int(code_str) if code_str else 0
                    if 0 < code_val <= config.target_cd_cnt:
                        code_idx = code_val - 1
                        if code_idx == 0:
                            continue
                        code_frequencies[code_idx] += 1
                except ValueError:
                    pass
        
        if (count + 1) % 1_000_000 == 0:
            print(f"    {count + 1:,}/{n_process:,} processed...")
    
    non_zero = np.sum(code_frequencies > 0)
    print(f"  Found {non_zero:,} unique codes")
    return code_frequencies



def build_tier_indices_streaming(
    dataset,
    code_frequencies: np.ndarray,
    percentile_boundaries: Tuple[float, float, float] = (20, 50, 80)
) -> dict:
    """
    Stream through ClinicalDatasetLazy.target_strs to build tier membership indices.
    Memory: ~50 MB (index lists only) vs ~176 GB (full targets list).
    
    Matches TierAwareBatchSampler._build_sample_tier_mapping behavior:
    - Uses get_target_codes_for_member which skips code_idx=0
    """
    freq_nz = code_frequencies[code_frequencies > 0]
    if len(freq_nz) == 0:
        raise ValueError("No non-zero frequencies found")
    
    percentiles = np.percentile(freq_nz, list(percentile_boundaries))
    
    tier_code_indices = {
        'common': set(np.where(code_frequencies > percentiles[2])[0]),
        'medium': set(np.where(
            (code_frequencies <= percentiles[2]) & (code_frequencies > percentiles[1])
        )[0]),
        'rare': set(np.where(
            (code_frequencies <= percentiles[1]) & (code_frequencies > percentiles[0])
        )[0]),
        'tail': set(np.where(
            (code_frequencies <= percentiles[0]) & (code_frequencies > 0)
        )[0]),
    }
    
    medium_codes = tier_code_indices['medium']
    rare_codes = tier_code_indices['rare']
    tail_codes = tier_code_indices['tail']
    
    samples_with_medium = []
    samples_with_rare = []
    samples_with_tail = []
    
    n = len(dataset)
    print(f"  Streaming tier classification for {n:,} members...")
    
    for idx in range(n):
        positive_codes = dataset.get_target_codes_for_member(idx)
        
        if positive_codes & medium_codes:
            samples_with_medium.append(idx)
        if positive_codes & rare_codes:
            samples_with_rare.append(idx)
        if positive_codes & tail_codes:
            samples_with_tail.append(idx)
        
        if (idx + 1) % 1_000_000 == 0:
            print(f"    {idx + 1:,}/{n:,} classified...")
    
    print(f"  Members with medium: {len(samples_with_medium):,} ({len(samples_with_medium)/n:.1%})")
    print(f"  Members with rare: {len(samples_with_rare):,} ({len(samples_with_rare)/n:.1%})")
    print(f"  Members with tail: {len(samples_with_tail):,} ({len(samples_with_tail)/n:.1%})")
    
    return {
        'samples_with_medium': samples_with_medium,
        'samples_with_rare': samples_with_rare,
        'samples_with_tail': samples_with_tail,
        'tier_code_indices': tier_code_indices,
        'tier_thresholds': {
            'tail_upper': percentiles[0],
            'rare_upper': percentiles[1],
            'medium_upper': percentiles[2],
        }
    }



def build_density_pools_streaming(
    dataset,
    code_frequencies: np.ndarray,
    percentile_boundaries: Tuple[float, float, float] = (20, 50, 80),
    density_tail_percentile: float = 80.0,
    density_rare_percentile: float = 70.0,
    density_medium_percentile: float = 70.0,
    verbose: bool = True
) -> dict:
    """
    Stream through ClinicalDatasetLazy to build density-aware tier pools.
    Replaces DensityTierAwareBatchSampler._build_density_pools for lazy datasets.
    
    Matches existing _build_density_pools behavior:
    - Skips code_idx=0 (consistent with `if code == 0: continue` at line 6587)
    - Counts per-tier occurrences (not just unique codes) for density scoring
    """
    freq_nz = code_frequencies[code_frequencies > 0]
    if len(freq_nz) == 0:
        raise ValueError("No non-zero frequencies found")
    
    percentiles = np.percentile(freq_nz, list(percentile_boundaries))
    
    tier_code_indices = {
        'common': set(np.where(code_frequencies > percentiles[2])[0]),
        'medium': set(np.where(
            (code_frequencies <= percentiles[2]) & (code_frequencies > percentiles[1])
        )[0]),
        'rare': set(np.where(
            (code_frequencies <= percentiles[1]) & (code_frequencies > percentiles[0])
        )[0]),
        'tail': set(np.where(
            (code_frequencies <= percentiles[0]) & (code_frequencies > 0)
        )[0]),
    }
    
    medium_codes = tier_code_indices['medium']
    rare_codes = tier_code_indices['rare']
    tail_codes = tier_code_indices['tail']
    
    n = len(dataset)
    tail_densities = np.zeros(n, dtype=np.float32)
    rare_densities = np.zeros(n, dtype=np.float32)
    medium_densities = np.zeros(n, dtype=np.float32)
    tail_counts = np.zeros(n, dtype=np.int32)
    rare_counts = np.zeros(n, dtype=np.int32)
    medium_counts = np.zeros(n, dtype=np.int32)
    total_counts = np.zeros(n, dtype=np.int32)
    
    if verbose:
        print(f"  Computing density scores for {n:,} members (streaming)...")
    
    for idx in range(n):
        if verbose and idx > 0 and idx % 1_000_000 == 0:
            print(f"    {idx:,}/{n:,} processed...")
        
        target_str = dataset.target_strs[idx]
        if not target_str or pd.isna(target_str):
            continue
        
        member_tail = 0
        member_rare = 0
        member_medium = 0
        member_total = 0
        
        for day_str in target_str.split('*')[:dataset.config.len_dy]:
            if not day_str:
                continue
            for code_str in day_str.split(','):
                try:
                    code_val = int(code_str) if code_str else 0
                    if 0 < code_val <= dataset.config.target_cd_cnt:
                        code_idx = code_val - 1
                        if code_idx == 0:
                            continue
                        member_total += 1
                        if code_idx in tail_codes:
                            member_tail += 1
                        elif code_idx in rare_codes:
                            member_rare += 1
                        elif code_idx in medium_codes:
                            member_medium += 1
                except ValueError:
                    pass
        
        total_counts[idx] = member_total
        tail_counts[idx] = member_tail
        rare_counts[idx] = member_rare
        medium_counts[idx] = member_medium
        
        if member_total > 0:
            tail_densities[idx] = member_tail / member_total
            rare_densities[idx] = member_rare / member_total
            medium_densities[idx] = member_medium / member_total
    
    tail_mask = tail_counts > 0
    rare_mask = rare_counts > 0
    medium_mask = medium_counts > 0
    
    tail_density_thresh = (
        np.percentile(tail_densities[tail_mask], density_tail_percentile)
        if tail_mask.sum() > 0 else 0.0
    )
    rare_density_thresh = (
        np.percentile(rare_densities[rare_mask], density_rare_percentile)
        if rare_mask.sum() > 0 else 0.0
    )
    medium_density_thresh = (
        np.percentile(medium_densities[medium_mask], density_medium_percentile)
        if medium_mask.sum() > 0 else 0.0
    )
    
    samples_with_tail = np.where(
        (tail_densities >= tail_density_thresh) & (tail_counts > 0)
    )[0].tolist()
    samples_with_rare = np.where(
        (rare_densities >= rare_density_thresh) & (rare_counts > 0)
    )[0].tolist()
    samples_with_medium = np.where(
        (medium_densities >= medium_density_thresh) & (medium_counts > 0)
    )[0].tolist()
    
    if verbose:
        print(f"  Density thresholds: tail>={tail_density_thresh:.4f}, "
              f"rare>={rare_density_thresh:.4f}, medium>={medium_density_thresh:.4f}")
        print(f"  Tail pool: {len(samples_with_tail):,} ({len(samples_with_tail)/n:.1%})")
        print(f"  Rare pool: {len(samples_with_rare):,} ({len(samples_with_rare)/n:.1%})")
        print(f"  Medium pool: {len(samples_with_medium):,} ({len(samples_with_medium)/n:.1%})")
    
    return {
        'samples_with_medium': samples_with_medium,
        'samples_with_rare': samples_with_rare,
        'samples_with_tail': samples_with_tail,
        'tier_code_indices': tier_code_indices,
        'tier_thresholds': {
            'tail_upper': percentiles[0],
            'rare_upper': percentiles[1],
            'medium_upper': percentiles[2],
        },
        'density_stats': {
            'tail_density_threshold': float(tail_density_thresh),
            'rare_density_threshold': float(rare_density_thresh),
            'medium_density_threshold': float(medium_density_thresh),
            'tail_pool_size': len(samples_with_tail),
            'rare_pool_size': len(samples_with_rare),
            'medium_pool_size': len(samples_with_medium),
        }
    }



# --- DataLoader creation ---

def _create_dataloaders(
    train_data: Union[pd.DataFrame, ClinicalDataset],
    val_data: Union[pd.DataFrame, ClinicalDataset],
    config: BaseConfig,
    use_bucketing: bool,
    train_data_df: Optional[pd.DataFrame] = None,
    world_size: int = 1,
    logger: Optional[logging.Logger] = None,
    optimize_config: Optional[OptimizeConfig] = None,
    code_frequencies: Optional[np.ndarray] = None,
    precomputed_tier_indices: Optional[dict] = None
    
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_data: Either a DataFrame (legacy) or pre-created ClinicalDataset
        val_data: Either a DataFrame (legacy) or pre-created ClinicalDataset
        config: Model configuration with batch_size
        use_bucketing: Whether to use bucketing batch sampler
        train_data_df: Original DataFrame (required for bucketing if train_data is Dataset)
        world_size: Number of distributed processes
        logger: Optional logger
        code_frequencies: Code frequency array (required if use_tier_aware=True)
        optimize_config: OptimizeConfig with tier-aware batching settings
        
    Returns:
        (train_loader, val_loader)
    """
    # Handle both Dataset and DataFrame inputs for backward compatibility
    def is_dataset(obj):
        """Check if object is a Dataset (duck-typing for Jupyter compatibility)."""
        return hasattr(obj, '__getitem__') and hasattr(obj, '__len__') and not isinstance(obj, pd.DataFrame)
    
    if is_dataset(train_data):
        train_dataset = train_data
    else:
        train_dataset = ClinicalDataset(train_data, config)
        train_data_df = train_data  # Save for bucketing
    
    if is_dataset(val_data):
        val_dataset = val_data
    else:
        val_dataset = ClinicalDataset(val_data, config)
    
    n_workers = min(4, os.cpu_count() // 4)
    collate_fn = create_collate_fn(config)
    
    use_tier_aware = (
        optimize_config is not None and 
        optimize_config.use_tier_aware_batching
    )
    
    if use_tier_aware:
        if code_frequencies is None:
            raise ValueError("code_frequencies required when use_tier_aware_batching=True")
        
        use_density = (
            optimize_config is not None and
            getattr(optimize_config, 'use_density_aware_batching', False)
        )
        
        if use_density:
            if logger:
                logger.info(f"Using DENSITY-AWARE tier batching "
                           f"(medium={optimize_config.tier_medium_quota}, "
                           f"rare={optimize_config.tier_rare_quota}, "
                           f"tail={optimize_config.tier_tail_quota}, "
                           f"tail_pct={optimize_config.density_tail_percentile})")
            
            train_batch_sampler = DensityTierAwareBatchSampler(
                dataset=train_dataset,
                code_frequencies=code_frequencies,
                batch_size=config.batch_size,
                medium_quota=optimize_config.tier_medium_quota,
                rare_quota=optimize_config.tier_rare_quota,
                tail_quota=optimize_config.tier_tail_quota,
                shuffle=True,
                drop_last=True,
                density_tail_percentile=optimize_config.density_tail_percentile,
                density_rare_percentile=optimize_config.density_rare_percentile,
                density_medium_percentile=optimize_config.density_medium_percentile,
                verbose=True,
                precomputed_density_pools=precomputed_tier_indices
            )
        else:
            if logger:
                logger.info(f"Using TIER-AWARE batching (binary presence) "
                           f"(medium={optimize_config.tier_medium_quota}, "
                           f"rare={optimize_config.tier_rare_quota}, "
                           f"tail={optimize_config.tier_tail_quota})")
            
            train_batch_sampler = TierAwareBatchSampler(
                dataset=train_dataset,
                code_frequencies=code_frequencies,
                batch_size=config.batch_size,
                medium_quota=optimize_config.tier_medium_quota,
                rare_quota=optimize_config.tier_rare_quota,
                tail_quota=optimize_config.tier_tail_quota,
                shuffle=True,
                drop_last=True,
                verbose=True,
                precomputed_tier_indices=precomputed_tier_indices
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=False
        )
        
    elif use_bucketing:
        if train_data_df is None:
            raise ValueError("train_data_df is required when use_bucketing=True and train_data is a Dataset")
        if logger:
            logger.info("Bucketing is ENABLED via BatchSampler.")
        train_batch_sampler = BucketingBatchSampler(
            data=train_data_df,
            batch_size=config.batch_size,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    else:
        if logger:
            logger.info("Using standard DataLoader (no bucketing).")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    # VALIDATION LOADER (always standard - no tier-aware needed for eval)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if logger:
        logger.info(f"Train loader: {len(train_loader):,} batches")
        logger.info(f"Val loader: {len(val_loader):,} batches")
        logger.info(f"Using DataLoader with {n_workers} workers.")
    
    return train_loader, val_loader



# --- Checkpoint resume ---

def _resume_from_checkpoint(
    resume_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[GradScaler],
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> Tuple[int, int, float]:
    """
    Resume training from checkpoint.
    
    Returns:
        start_epoch, global_step, best_val_loss
    """
    checkpoint_data = torch.load(resume_path, map_location=device)
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint_data['model_state_dict'])
    else:
        model.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
    
    start_epoch = checkpoint_data['epoch'] + 1
    global_step = checkpoint_data.get('global_step', 0)
    
    best_val_loss = float('inf')
    if checkpoint_data.get('metrics'):
        valid_losses = [m.get('val_loss', float('inf')) for m in checkpoint_data['metrics'] if 'val_loss' in m]
        if valid_losses:
            best_val_loss = min(valid_losses)
    
    if logger:
        logger.info(f"✅ Resumed from epoch {start_epoch}, step {global_step}")
        logger.info(f"   Previous best val loss: {best_val_loss:.4f}")
    
    return start_epoch, global_step, best_val_loss



# --- Metrics building ---

def _build_epoch_metrics(
    epoch: int,
    train_metrics: Dict,
    train_eval_metrics: Dict,
    val_metrics: Dict
) -> Dict[str, Any]:
    """Build comprehensive epoch metrics dictionary. 
    
        train_metrics (from train_epoch()):
            - Computed during TRAINING MODE (dropout active)
            - Loss: averaged over ALL batches
            - Other metrics: averaged over SAMPLED batches (every log_interval) 
        
        train_eval_metrics (from train_metrics):
            - Derived from train_metrics batch averages
            - NOT a separate forward pass in eval mode
            - Slightly noisier than true eval-mode metrics
            - Purpose: Training monitoring, not final model comparison

        val_metrics (from evaluate() or comprehensive_evaluation()):
            - Computed during EVAL MODE (no dropout)
            - Full forward pass on validation data
            - These are the authoritative metrics for model selection
            - Final epoch uses comprehensive_evaluation()

        The 'generalization_gap' metric uses train_eval_metrics['val_loss'] (which is
        actually train loss) minus val_metrics['val_loss'] (true validation loss).
        A positive gap indicates the model performs better on training than validation.
    
    """
    epoch_metrics = {
        'epoch': epoch + 1,
        # Training trajectory
        'train_loss': train_metrics['train_loss'],
        'train_loss_mean': train_metrics['train_loss_mean'],
        'train_loss_first': train_metrics['train_loss_first'],
        'train_loss_last': train_metrics['train_loss_last'],
        'train_loss_std': train_metrics['train_loss_std'],
        'train_loss_improvement': train_metrics['train_loss_improvement'],
        # Train evaluation
        'eval_in_train_loss_final': train_eval_metrics['val_loss'],
        'eval_in_train_recall@1': train_eval_metrics['recall@1'],
        'eval_in_train_recall@5': train_eval_metrics['recall@5'],
        'eval_in_train_recall@10': train_eval_metrics['recall@10'],
        'eval_in_train_recall@20': train_eval_metrics['recall@20'],
        'eval_in_train_micro_recall@10': train_eval_metrics.get('micro_recall@10', 0.0),
        'eval_in_train_ndcg@20': train_eval_metrics.get('ndcg@20', 0.0),
        # Validation
        'final_val_loss': val_metrics['val_loss'],
        'final_val_recall@1': val_metrics['recall@1'],
        'final_val_recall@5': val_metrics['recall@5'],
        'final_val_recall@10': val_metrics['recall@10'],
        'final_val_recall@20': val_metrics['recall@20'],
        'final_val_micro_recall@10': val_metrics.get('micro_recall@10', 0.0),
        'final_val_micro_recall@20': val_metrics.get('micro_recall@20', 0.0),
        'final_val_ndcg@10': val_metrics.get('ndcg@10', 0.0),
        'final_val_ndcg@20': val_metrics.get('ndcg@20', 0.0),
        'final_val_mrr': val_metrics.get('mrr', 0.0),
        'final_val_positive_brier': val_metrics.get('positive_brier', 0.0),
        'generalization_gap': train_eval_metrics['val_loss'] - val_metrics['val_loss'],
    }
    
    # Add other training metrics
    for k, v in train_metrics.items():
        if k.startswith('train_') and k not in epoch_metrics:
            epoch_metrics[k] = v
    
    # Add other validation metrics (embedding quality, etc.)
    excluded_keys = {'val_loss', 'num_samples', 'num_batches'}
    for k, v in val_metrics.items():
        if k not in epoch_metrics and k not in excluded_keys:
            epoch_metrics[f'val_{k}'] = v
    
    return epoch_metrics



def _build_final_results(
    exp_name: str,
    total_params: int,
    use_learnt_att_pool: bool,
    use_bucketing: bool,
    final_metrics: Dict,
    evaluation: Dict,
    epoch_history: List[Dict],
    total_time: float
) -> Dict[str, Any]:
    """Build final experiment results dictionary."""
    return {
        'experiment': exp_name,
        'parameters': total_params,
        'use_learned_pooling': use_learnt_att_pool,
        'use_bucketing': use_bucketing,
        'train_loss_mean': final_metrics['train_loss'],
        'train_loss_learned': final_metrics['train_loss_improvement'],
        'train_loss_final': final_metrics['eval_in_train_loss_final'],
        'val_loss_final': final_metrics['final_val_loss'],
        'generalization_gap': final_metrics['generalization_gap'],
        # Recall metrics
        'final_train_recall@5': final_metrics.get('eval_in_train_recall@5', 0.0),
        'final_train_recall@10': final_metrics.get('eval_in_train_recall@10', 0.0),
        'final_train_recall@20': final_metrics.get('eval_in_train_recall@20', 0.0),
        'final_val_recall@5': final_metrics.get('final_val_recall@5', 0.0),
        'final_val_recall@10': final_metrics.get('final_val_recall@10', 0.0),
        'final_val_recall@20': final_metrics.get('final_val_recall@20', 0.0),
        # New metrics
        'final_val_micro_recall@10': final_metrics.get('final_val_micro_recall@10', 0.0),
        'final_val_ndcg@20': final_metrics.get('final_val_ndcg@20', 0.0),
        'final_val_mrr': final_metrics.get('final_val_mrr', 0.0),
        'final_val_positive_brier': final_metrics.get('final_val_positive_brier', 0.0),
        # Comprehensive evaluation
        'training_time_sec': total_time,
        'precision@10': evaluation['performance'].get('precision@10', 0.0),
        'recall@10': evaluation['performance'].get('recall@10', 0.0),
        'f1@10': evaluation['performance'].get('f1@10', 0.0),
        'micro_recall@10': evaluation['performance'].get('micro_recall@10', 0.0),
        'ndcg@10': evaluation['performance'].get('ndcg@10', 0.0),
        'balanced_top10_acc': evaluation['performance'].get('balanced_top10_acc', 0.0),
        'tail_top10_acc': evaluation['performance'].get('tail_top10_acc', 0.0),
        'cost_usd': evaluation['resources'].get('cost_usd', 0.0),
        'peak_memory_gb': evaluation['resources'].get('total_peak_gb', 0.0),
        'full_evaluation': evaluation,
        'all_epochs': epoch_history
    }


# #### Run experimentation



# --- Main experiment runner ---

def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    prepared_data: Optional[PreparedData] = None,
    train_data: Optional[pd.DataFrame] = None, # needed for bucketing sampler
    val_data: Optional[pd.DataFrame] = None, # deprecated
    device: torch.device = None,
    epochs: int = 2,
    log_dir: str = "logs",
    experiment_round: Optional[str] = None,
    check_embeddings_every: Optional[int] = None,
    log_metrics_every: int = 100,
    resume_from: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    embedding_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    save_model: bool = True,
    eval_max_batches: Optional[int] = None,
    optimize_config: Optional[OptimizeConfig] = None
    
) -> Dict[str, Any]:
    """
    Run a single experiment with optional downstream evaluation.
    V2: clean up the messy implemenation and put the V1 to legacy
    This is the main entry point for training a model variant.
    
    Training Flow:
    1. Setup directories, logging
    2. Create model
    3. Create dataloaders  
    4. Setup optimizer/scheduler
    5. Training loop (epochs)
       - Each epoch: train → evaluate → save_checkpoint()
    6. total_time = time.time() - start_time
    7. comprehensive_evaluation()
    8. Save final model
    9. Cleanup checkpoints
    10. Finalize & return results
    """
    # ============================================================
    # 1. SETUP
    # ============================================================
    # DDP is disabled for now
    use_ddp = False
    is_main = True
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_resume = resume_from is not None
    effective_log_dir, checkpoint_dir = _setup_experiment_directories(
        log_dir, experiment_round, exp_name, checkpoint_dir
    )
    
    # Setup logging
    logger = setup_experiment_logging(exp_name, effective_log_dir, resume=is_resume)
    metrics_logger = MetricsLogger(exp_name, effective_log_dir, resume=is_resume)
    loss_tracker = LossTracker(window_size=100)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPERIMENT: {exp_name}")
    logger.info(f"{'='*80}")
    
    # ============================================================
    # 2. MODEL CREATION
    # ============================================================
    eff_d_model = embedding_size if embedding_size is not None else 256
    uses_swiglu = exp_name not in ['exp1_dense_baseline']
    dims = _calculate_model_dimensions(eff_d_model, use_swiglu=uses_swiglu)    
    model, config, moe_config, use_mixed_precision, use_bucketing = _create_model(
        exp_name=exp_name,
        eff_d_model=eff_d_model,
        eff_nhid=dims['nhid'],
        eff_nhead=dims['nhead'],
        moe_config=moe_config,
        use_learnt_att_pool=use_learnt_att_pool,
        device=device,
        logger=logger
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # ============================================================
    # GET DATASET AND CODE FREQUENCY
    # ============================================================
    
    if prepared_data is not None:
        train_dataset = prepared_data.train_dataset
        val_dataset = prepared_data.val_dataset
        code_frequencies = prepared_data.code_frequencies
        # For bucketing, we need the original DataFrame - store reference
        train_data_df = train_data if train_data is not None else None
        logger.info("✅ Using pre-prepared data")    
    else:
        # LEGACY PATH: Create datasets from DataFrames; will deprecate this soon; takes long time
        if train_data is None or val_data is None:
            raise ValueError("Either prepared_data or (train_data, val_data) must be provided")
        
        logger.info("⚠️ Using legacy mode: parsing datasets from DataFrames")
        train_dataset = ClinicalDataset(train_data, config)
        val_dataset = ClinicalDataset(val_data, config)
        train_data_df = train_data
        
        # Calculate code frequency if not provided
        code_frequencies = _compute_code_frequencies_from_dataset(
            train_dataset, config, sample_fraction=1.0
        )    

    # ============================================================
    # CRITERION CREATION (Focal Loss + multiple weight methods)
    # ============================================================
    # Create criterion with pos_weight
    if optimize_config is not None and (optimize_config.use_pos_weight or 
                                         optimize_config.use_focal_loss or
                                         getattr(optimize_config, 'use_asl', False)):
        criterion = create_criterion(
            code_frequencies=code_frequencies,
            device=device,
            optimize_config=optimize_config
        )
        if getattr(optimize_config, 'use_asl', False):
            logger.info(f"Using AsymmetricLoss (γ+={optimize_config.asl_gamma_pos}, "
                        f"γ-={optimize_config.asl_gamma_neg})")
        elif optimize_config.use_focal_loss:
            logger.info(f"Using FocalLoss (gamma={optimize_config.focal_gamma})")
        else:
            logger.info("Using BCEWithLogitsLoss")
        if optimize_config.use_pos_weight:
            logger.info(f"  With pos_weight method: {optimize_config.pos_weight_method}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Using BCEWithLogitsLoss without pos_weight")        
    # ============================================================
    # DATAPARALLEL WRAPPER FOR MULTI-GPU
    # ============================================================    
    num_gpus = torch.cuda.device_count()
    use_data_parallel = num_gpus > 1
    
    # Always wrap model regardless of num_gpus
    wrapped_model = DataParallelWrapper(
        model=model,
        config=config,
        criterion=criterion,
        moe_config=moe_config
    )    
    if use_data_parallel:
        logger.info(f" Enabling DataParallel with {num_gpus} GPUs")
        # Scale batch size proportionally (effective batch = batch_size * num_gpus)
        effective_batch_size = config.batch_size * num_gpus
        
        # Scale learning rate (square root scaling - more conservative)
        base_lr = config.learning_rate  # 1e-4 from your config 
        # scaled_lr = base_lr * math.sqrt(num_gpus)  # ~2e-4 for 4 GPUs, this is conservative in exp round 5
        # Alternative: Linear scaling (more aggressive)
        scaled_lr = base_lr * num_gpus  # 4e-4 for 4 GPUs
        
        logger.info(f"   Per-GPU batch size: {config.batch_size}")
        logger.info(f"   Effective batch size: {effective_batch_size}")
        logger.info(f"   Base learning rate: {base_lr}")
        logger.info(f"   Scaled learning rate: {scaled_lr:.2e}")
        
        # Use cusotomized data parallel wrapper
        model = nn.DataParallel(wrapped_model)  
        # Verification: Check DataParallel is set up correctly
        logger.info(f"   DataParallel device_ids: {model.device_ids}")
        logger.info(f"   DataParallel output_device: {model.output_device}")
        logger.info(f"   ✅ Using DataParallelWrapper for integrated loss")
        # Update batch_size to effective_batch_size
        config.batch_size = effective_batch_size
        
    else:
        scaled_lr = config.learning_rate 
        model = wrapped_model
        logger.info(f" Single GPU mode with DataParallelWrapper")        
    # Log config
    metrics_logger.log_config({
        'experiment': exp_name,
        'embedding_size': eff_d_model,
        'nhid': dims['nhid'],
        'nhead': dims['nhead'],
        'batch_size': config.batch_size,
        'use_mixed_precision': use_mixed_precision,
        'use_bucketing': use_bucketing,
        'use_learnt_att_pool': use_learnt_att_pool,
        'optimize_config': vars(optimize_config) if optimize_config else None,
        'moe_config': vars(moe_config) if moe_config else None
    
    })
    
    # ============================================================
    # CONVERT DATASET TO DATALOADER 
    # Have to come after the config.batch_size = effective_batch_size gets updated
    # ============================================================
    # pre-computation block 
    precomputed_tier = None
    is_lazy = isinstance(train_dataset, ClinicalDatasetLazy)
    use_tier_aware = (
        optimize_config is not None and 
        optimize_config.use_tier_aware_batching
    )
    if is_lazy and use_tier_aware:
        use_density = (
            optimize_config is not None and
            getattr(optimize_config, 'use_density_aware_batching', False)
        )
        if use_density:
            logger.info("Pre-computing density pools for lazy dataset...")
            precomputed_tier = build_density_pools_streaming(
                train_dataset, code_frequencies,
                density_tail_percentile=optimize_config.density_tail_percentile,
                density_rare_percentile=optimize_config.density_rare_percentile,
                density_medium_percentile=optimize_config.density_medium_percentile
            )
        else:
            logger.info("Pre-computing tier indices for lazy dataset...")
            precomputed_tier = build_tier_indices_streaming(
                train_dataset, code_frequencies
            )
    
    train_loader, val_loader = _create_dataloaders(
        train_data=train_dataset, 
        val_data=val_dataset, 
        config=config, 
        use_bucketing=use_bucketing, 
        train_data_df=train_data_df, 
        logger=logger,
        optimize_config=optimize_config,
        code_frequencies=code_frequencies,
        precomputed_tier_indices=precomputed_tier
    )
    # ============================================================
    # 4. OPTIMIZER SETUP
    # ============================================================
    optimizer, optimizer_desc = create_optimizer(
        model=model,
        base_config=config,
        optimize_config=optimize_config,
        scaled_lr=scaled_lr,
        logger=logger
    )
    total_steps = len(train_loader) * epochs
    
    scheduler, scheduler_desc = create_scheduler(
        optimizer=optimizer,
        optimize_config=optimize_config,
        total_steps=total_steps,
        scaled_lr=scaled_lr,
        logger=logger
    )  
    
    if is_main:
        logger.info(f"Optimizer: {optimizer_desc}")
        logger.info(f"Scheduler: {scheduler_desc}") 

    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # ============================================================
    # 5. RESUME FROM CHECKPOINT (if applicable)
    # ============================================================
    start_epoch, global_step, best_val_loss = 0, 0, float('inf')
    
    if is_resume:
        start_epoch, global_step, best_val_loss = _resume_from_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device, logger
        )

    # ============================================================
    # GRADIENT TIER ANALYSIS (optional diagnostic)
    # ============================================================
    gradient_tier_analyzer = None
    if optimize_config and getattr(optimize_config, 'enable_gradient_tier_analysis', False):
        gradient_tier_analyzer = GradientTierAnalyzer(
            code_frequencies=code_frequencies,
            device=device,
            log_interval=log_metrics_every
        )
        logger.info("📊 Gradient Tier Analysis ENABLED")
        
    # ============================================================
    # 6. TRAINING LOOP
    # ============================================================
    logger.info(f"Training for {epochs} epochs...")
    epoch_history = []
    start_time = time.time()
    final_comprehensive_evaluation = None
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
        loss_tracker.reset_epoch()
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config,
            device=device,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision,
            moe_config=moe_config,
            epoch=epoch,
            use_bucketing=use_bucketing,
            log_interval=log_metrics_every,
            global_step=global_step,
            loss_tracker=loss_tracker,
            is_main=is_main,
            use_ddp=use_ddp,
            accumulation_steps=1,  # no gradient accumulation with DataParallel
            metrics_logger = metrics_logger,
            logger = logger,
            optimize_config=optimize_config,
            gradient_tier_analyzer=gradient_tier_analyzer
        )
        global_step = train_metrics['global_step']
        
        # Evaluate
        logger.info("  Using batch-averaged training metrics (no re-evaluation)")
        train_eval_metrics = {
                # Use train_loss as "val_loss" for the expected key format
                # This is the average loss over all batches (not sampled)
                'val_loss': train_metrics.get('train_loss', 0.0),

                # Recall metrics (from batch sampling)
                'recall@1': train_metrics.get('train_recall@1', 0.0),
                'recall@5': train_metrics.get('train_recall@5', 0.0),
                'recall@10': train_metrics.get('train_recall@10', 0.0),
                'recall@20': train_metrics.get('train_recall@20', 0.0),

                # Micro-recall (from batch sampling)
                'micro_recall@10': train_metrics.get('train_micro_recall@10', 0.0),
                'micro_recall@20': train_metrics.get('train_micro_recall@20', 0.0),

                # NDCG (from batch sampling)
                'ndcg@10': train_metrics.get('train_ndcg@10', 0.0),
                'ndcg@20': train_metrics.get('train_ndcg@20', 0.0),

                # Other metrics
                'mrr': train_metrics.get('train_mrr', 0.0),
                'positive_brier': train_metrics.get('train_positive_brier', 0.0),
            }
        
        # Reset gradient tier analyzer for next epoch
        if gradient_tier_analyzer is not None:
            gradient_tier_analyzer.aggregate_epoch()  # Store epoch summary
            gradient_tier_analyzer.reset_epoch()
        
        # if this is the final epoch; then directly compute the omprehensive evaluation
        if epoch == epochs - 1:
            # FINAL EPOCH: Run comprehensive_evaluation directly
            # This computes val metrics + detailed analysis in one pass
            logger.info("  Final epoch: Running comprehensive evaluation...")
            
            # Get time up to this point for the comprehensive evaluation
            current_time = time.time() - start_time
            
            comprehensive_result = comprehensive_evaluation(
                model=model,
                val_dataloader=val_loader,
                config=config,
                device=device,
                training_time_sec=current_time,
                epoch_history=epoch_history,  # Previous epochs only
                code_frequencies=code_frequencies,
                moe_config=moe_config,
                use_mixed_precision=use_mixed_precision,
                current_train_metrics=train_metrics
            )
            
            # Extract val_metrics from comprehensive_evaluation
            val_metrics = comprehensive_result['performance']
            
            # Store comprehensive result for later (skip re-running after loop)
            final_comprehensive_evaluation = comprehensive_result
            
        else:
            # NON-FINAL EPOCHS: Use lightweight evaluate()
            logger.info("  Evaluating on validation set...")
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                config=config,
                device=device,
                use_mixed_precision=use_mixed_precision,
                max_batches=eval_max_batches
            )
        
        # Embedding quality check
        if epoch != epochs - 1 and check_embeddings_every and epoch % check_embeddings_every == 0:
            if val_data is not None:
                logger.info("Computing embedding quality...")
                emb_metrics = compute_embedding_quality_epoch(
                    model, val_data, config, device,
                    num_samples=100,
                    use_mixed_precision=True
                )
                val_metrics.update(emb_metrics)
                logger.info(f"    Embedding std: {emb_metrics['embedding_std_mean']:.4f}")
                logger.info(f"    NN overlap: {emb_metrics['nn_target_overlap']:.3f}")
            else:
                logger.debug("  Skipping embedding quality check (val_data DataFrame not provided)")
        
        # Build epoch metrics
        epoch_metrics = _build_epoch_metrics(epoch, train_metrics, train_eval_metrics, val_metrics)
        epoch_history.append(epoch_metrics)
        
        # Save checkpoint
        is_best = epoch_metrics['final_val_loss'] < best_val_loss
        if is_best:
            best_val_loss = epoch_metrics['final_val_loss']
        
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=epoch_history,
            is_best=is_best
        )
        
        # Log summary
        logger.info(f"\n--- Epoch {epoch+1} Summary ---")
        logger.info(f"  Train loss: {train_metrics['train_loss']:.4f} → {train_metrics['train_loss_last']:.4f}")
        logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}, "
                    f"Recall@10: {val_metrics.get('recall@10', 0):.3f}, "
                    f"μRecall@10: {val_metrics.get('micro_recall@10', 0):.3f}, "
                    f"NDCG@20: {val_metrics.get('ndcg@20', 0):.3f}")
        
        metrics_logger.log_epoch(epoch + 1, epoch_metrics)
        loss_tracker.save_trajectory(
            filepath=os.path.join(effective_log_dir, exp_name, f'loss_trajectory_epoch{epoch}.json')
        )
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time:.1f}s")
    
    # ============================================================
    # 7. FINAL EVALUATION
    # ============================================================
    if final_comprehensive_evaluation:
        logger.info("  Using cached comprehensive evaluation from final epoch")
        evaluation = final_comprehensive_evaluation
    else:
        # Fallback: run comprehensive evaluation (only if epochs=0 or some edge case)
        logger.info("  Running comprehensive evaluation...")
        evaluation = comprehensive_evaluation(
            model=model,
            val_dataloader=val_loader,
            config=config,
            device=device,
            training_time_sec=total_time,
            epoch_history=epoch_history,
            code_frequencies=code_frequencies,
            moe_config=moe_config,
            use_mixed_precision=use_mixed_precision
        )
        
    # Build results
    results = _build_final_results(
        exp_name, total_params, use_learnt_att_pool, use_bucketing,
        epoch_history[-1], evaluation, epoch_history, total_time
    )
    
    # ============================================================
    # 8. SAVE MODEL (if requested)
    # ============================================================
    if save_model:
        model_name = generate_model_name(
            exp_name=exp_name,
            experiment_round=experiment_round,
            batch_size=config.batch_size,
            epochs=epochs,
            embedding_size=eff_d_model
        )
        results['model_name'] = model_name
        
        model_save_dir = os.path.join(effective_log_dir, exp_name, 'saved_models')
        model_path = save_trained_model(
            model=model,
            config=config,
            model_name=model_name,
            save_dir=model_save_dir,
            exp_results=results,
            checkpoint_dir=checkpoint_dir,
            is_best=True,
            moe_config = moe_config
        )
        logger.info(f"Model saved as: {model_name}")
        results['model_path'] = model_path
        
        # ============================================================
        # 8b. CLEANUP CHECKPOINTS (model saved, no longer needed) to RELEASE MEMORY
        # ============================================================
        # implement inside the memory management session
        cleanup_checkpoints_after_training(
            checkpoint_dir=checkpoint_dir,
            keep_best=False,  # Set to true if keep checkpoint_best.pt
            logger=logger
        ) 
    
    # ============================================================
    # 10. FINALIZE
    # ============================================================
    results_path = metrics_logger.save_final_results(results)
    logger.info(f"Complete results saved to {results_path}")
    
    
    metrics_logger.save()
    
    summary = metrics_logger.get_summary()
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPERIMENT COMPLETE: {exp_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Final Recall@10: {epoch_history[-1]['final_val_recall@10']:.3f}")
    logger.info(f"Best Val Loss: {summary['best_val_loss']:.4f}")
    logger.info(f"Training Time: {total_time:.1f}s")
    logger.info(f"{'='*80}\n")
    
    return results


# --- Post-training cleanup ---

def cleanup_checkpoints_after_training(
    checkpoint_dir: str,
    keep_best: bool = True,
    logger: Optional[logging.Logger] = None
):
    """
    Clean up checkpoint files after training is complete.
    Called after save_trained_model() to reclaim disk space in vertexAI workbench
    
    Args:
        checkpoint_dir: Path to checkpoints directory
        keep_best: If True, keeps checkpoint_best.pt for reference
        logger: Logger for output
    """
    import glob
    import shutil
    
    if not os.path.exists(checkpoint_dir):
        return
    
    files_removed = 0
    bytes_freed = 0
    
    # Remove all epoch checkpoints
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch*.pt')):
        bytes_freed += os.path.getsize(f)
        os.remove(f)
        files_removed += 1
    
    # Remove latest checkpoint
    latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    if os.path.exists(latest):
        bytes_freed += os.path.getsize(latest)
        os.remove(latest)
        files_removed += 1
    
    # Optionally remove best checkpoint
    if not keep_best:
        best = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        if os.path.exists(best):
            bytes_freed += os.path.getsize(best)
            os.remove(best)
            files_removed += 1
    
    # Remove empty directory
    remaining = os.listdir(checkpoint_dir)
    if not remaining:
        os.rmdir(checkpoint_dir)
    
    gb_freed = bytes_freed / (1024 ** 3)
    msg = f"🗑️ Cleaned up {files_removed} checkpoint files, freed {gb_freed:.2f} GB"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ### Time and cost estimation



