#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Mixture-of-Experts (MoE) Experimentation Framework for Hierarchical Clinical Transformer

This module implements a comprehensive 5-experiment ablation study to evaluate MoE integration
into the hierarchical clinical transformer architecture (min_transformer.py).

Experiment Overview:
- Exp 1: Dense Baseline (reference performance)
- Exp 2: Standard Top-K MoE (8 experts, top-2)
- Exp 3: Shared Expert MoE (1 shared + 7 routed)
- Exp 4: Fine-Grained MoE (1 shared + 15 routed, smaller experts)
- Exp 5: Auxiliary-Free MoE (DeepSeek bias-based load balancing)

Based on:
- DeepSeek-MoE ablation methodology
- Switch Transformer load balancing
- BEHRT clinical evaluation metrics

Author: Daniel Xing
Date: 2025-10-24
"""


# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from collections import Counter
import time


# In[4]:


# ============================================================================
# SECTION 1: CONFIGURATION CLASSES
# ============================================================================


from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    """
    # Column names in your DataFrame
    input_code_column: str = 'cd'           # Input medical codes
    target_code_column: str = 'target_cd'   # Target codes (YOUR COLUMN NAME!)
    age_column: str = 'age_in_months'
    gender_column: str = 'gender_cd'
    dt_cnt_column: str = 'dt_cnt'
    id_column: str = 'individual_id'
    
    # Sequence parameters
    len_dy: int = 70        # Max days per sequence (check your data!)
    len_cd: int = 25        # Max codes per day (check your data!)
    max_age: int = 1439     # Max age in months (120 years)
    
    # Data paths (optional)
    train_query: Optional[str] = None
    val_query: Optional[str] = None
    test_query: Optional[str] = None

    
@dataclass
class ModelConfig:
    """
    Configuration for model architecture (excluding MoE).
    """
    # Vocabulary sizes
    cd_cnt: int = 84010              # Input code vocabulary size
    target_cd_cnt: int = 8850        # Target code vocabulary (YOUR VALUE!)
    
    # Architecture
    embedding_size: int = 256
    nhead: int = 16                  # Attention heads for temporal encoder
    nhid: int = 512                  # FFN hidden dimension
    nlayers: int = 6                 # Number of temporal layers
    dropout: float = 0.1
    
    # MoE placement
    use_moe_from_layer: int = 2      # Which layer to start MoE (0-5)
    
    # Embeddings
    num_gender_categories: int = 4   # Gender embedding vocab    


@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters.
    """
    # Optimization
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 1
    
    # Scheduler
    scheduler_type: str = 'cosine'   # 'cosine', 'step', 'none'
    warmup_steps: int = 0
    
    # Regularization
    gradient_clip_norm: float = 1.0
    
    # Logging
    log_interval: int = 100          # Batches between logging
    eval_frequency: int = 1          # Epochs between evaluation
    
    # Device & Parallelism
    device: str = 'cuda'  # ← Changed from 'cuda:0'
    parallel: bool = True  # ← Enable DataParallel
    num_gpus: int = 4      # ← Number of GPUs to use
    
@dataclass
class MoEConfig:
    """
    Configuration dataclass for Mixture-of-Experts layer.
    
    Attributes:
        d_model: Embedding dimension (256 from min_transformer.py)
        d_ff: Feed-forward hidden dimension (512 standard)
        num_experts: Total number of experts (shared + routed)
        num_shared_experts: Number of always-activated shared experts (0 for Exp 2)
        top_k: Number of experts activated per token
        expert_dropout: Dropout rate within expert FFN
        load_balance_strategy: 'switch' or 'deepseek' balancing method
        aux_loss_weight: Weight for Switch auxiliary loss (typically 0.01)
        bias_lr: Learning rate for DeepSeek bias correction
        bias_momentum: EMA momentum for DeepSeek bias correction
        z_loss_weight: Router Z-loss weight (optional, not used in 5 experiments)
    """
    # Model dimensions
    d_model: int = 256
    d_ff: int = 512
    
    # Expert architecture
    num_experts: int = 8
    num_shared_experts: int = 0
    top_k: int = 2
    expert_dropout: float = 0.05
    
    # Load balancing
    load_balance_strategy: str = 'switch'  # 'switch' or 'deepseek'
    aux_loss_weight: float = 0.01
    bias_lr: float = 1e-5
    bias_momentum: float = 0.9
    
    # Optional
    z_loss_weight: float = 0.0


# In[6]:


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

def get_experiment_configs() -> Dict[str, Optional[MoEConfig]]:
    """
    Define all 5 experiment configurations for ablation study.
    
    Returns:
        Dictionary mapping experiment name to MoEConfig (None for dense baseline)
        
    Experiments:
        exp1_dense: No MoE, use original TransformerModel from min_transformer.py
        exp2_standard_moe: 8 experts, top-2, all routed
        exp3_shared_expert: 1 shared + 7 routed, top-1 routed (2 total active)
        exp4_fine_grained: 1 shared + 15 routed (238 dim), top-4 routed (5 total active)
        exp5_auxiliary_free: Same as exp3 but with DeepSeek bias balancing
    """
    configs = {}
    
    # Exp 1: Dense Baseline (Min's transformer    )
    configs['exp1_dense'] = None
    
    # Exp 2: Standard Top-K MoE
    configs['exp2_standard_moe'] = MoEConfig(
        d_model=256,
        d_ff=512,
        num_experts=8,
        num_shared_experts=0,
        top_k=2,
        load_balance_strategy='switch',
        aux_loss_weight=0.01,
        expert_dropout=0.05,
    )
    
    # Exp 3: Shared Expert MoE
    configs['exp3_shared_expert'] = MoEConfig(
        d_model=256,
        d_ff=512,
        num_experts=8,
        num_shared_experts=1,
        top_k=2,  # 1 shared (always) + 1 routed = 2 total
        load_balance_strategy='switch',
        aux_loss_weight=0.01,
        expert_dropout=0.05,
    )
    
    # Exp 4: Fine-Grained MoE
    # Calculate expert dimension to maintain ~2.1M params per layer:
    # Shared: 1 × (256 × 512 × 2) = 262K
    # Routed: 15 × (256 × d_ff × 2) ≈ 1,835K
    # d_ff ≈ 238
    configs['exp4_fine_grained'] = MoEConfig(
        d_model=256,
        d_ff=238,  # Smaller experts for fine-grained specialization
        num_experts=16,
        num_shared_experts=1,
        top_k=5,  # 1 shared + 4 routed = 5 total
        load_balance_strategy='switch',
        aux_loss_weight=0.01,
        expert_dropout=0.05,
    )
    
    # Exp 5: Auxiliary-Free MoE (same architecture as Exp 3, different balancing)
    configs['exp5_auxiliary_free'] = MoEConfig(
        d_model=256,
        d_ff=512,
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
        load_balance_strategy='deepseek',  # Key difference
        bias_lr=1e-5,
        bias_momentum=0.9,
        aux_loss_weight=0.0,  # Not used for DeepSeek
        expert_dropout=0.05,
    )
    
    return configs


# ============================================================================
# SECTION 2: LOAD BALANCING MECHANISMS
# ============================================================================

class SwitchAuxiliaryLoss(nn.Module):
    """
    Switch Transformer auxiliary loss for load balancing (Fedus et al. 2021).
    
    Encourages uniform expert utilization by penalizing imbalanced routing.
    
    Loss Formula:
        L_aux = N × Σ_i (importance_i × load_i)
        
    Where:
        - N = number of experts
        - importance_i = mean router probability for expert i (what router thinks)
        - load_i = fraction of tokens routed to expert i (actual usage)
        
    Minimized when importance ≈ load ≈ 1/N (uniform distribution).
    
    Reference:
        "Switch Transformers: Scaling to Trillion Parameter Models with Simple 
        and Efficient Sparsity" (https://arxiv.org/abs/2101.03961)
    """
    
    def __init__(self, num_experts: int):
        """
        Initialize Switch auxiliary loss.
        
        Args:
            num_experts: Number of routed experts (excludes shared experts)
        """
        super().__init__()
        self.num_experts = num_experts
    
    def forward(self, router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        Args:
            router_probs: [num_tokens, num_experts] - softmax probabilities from router
            expert_indices: [num_tokens, top_k] - indices of selected experts
            
        Returns:
            aux_loss: Scalar loss value
            
        Example:
            If all experts equally used: loss ≈ 1.0 (minimum)
            If imbalanced (e.g., 2 experts do all work): loss > 1.0 (penalized)
        """
        # Importance: average probability mass given to each expert by router
        importance = router_probs.mean(dim=0)  # [num_experts]
        
        # Load: actual fraction of tokens routed to each expert
        batch_size = expert_indices.shape[0]
        top_k = expert_indices.shape[1]
        load = torch.zeros(self.num_experts, device=expert_indices.device)
        
        # Count how many tokens assigned to each expert (across all top-k positions)
        for k in range(top_k):
            load.scatter_add_(0, expert_indices[:, k], 
                            torch.ones(batch_size, device=expert_indices.device))
        
        # Normalize by total token-expert assignments
        load = load / (batch_size * top_k)
        
        # Switch loss: N × dot_product(importance, load)
        aux_loss = self.num_experts * torch.sum(importance * load)
        
        return aux_loss


class DeepSeekBiasCorrection(nn.Module):
    """
    DeepSeek-V3 auxiliary-loss-free load balancing via learnable bias.
    
    Key Idea:
        Instead of adding aux loss to main objective (creating gradient conflicts),
        adjust router bias separately to encourage balanced expert usage.
        
    Update Rule:
        bias_i(t+1) = bias_i(t) - α × [load_i(t) - 1/N]
        
    Effect:
        - Overused expert (load > 1/N): bias decreases → lower selection probability
        - Underused expert (load < 1/N): bias increases → higher selection probability
        
    Advantages:
        1. No auxiliary loss hyperparameter to tune
        2. No gradient conflict between prediction and balancing
        3. More stable training (empirically validated in DeepSeek-V3)
        
    Reference:
        "DeepSeek-V3 Technical Report" Section 3.2
        (https://arxiv.org/abs/2412.19437)
    """
    
    def __init__(self, num_experts: int, bias_lr: float = 1e-5, momentum: float = 0.9):
        """
        Initialize DeepSeek bias correction mechanism.
        
        Args:
            num_experts: Number of routed experts
            bias_lr: Learning rate for bias updates (typically 1e-5)
            momentum: EMA momentum for load smoothing (typically 0.9)
        """
        super().__init__()
        self.num_experts = num_experts
        self.bias_lr = bias_lr
        self.momentum = momentum
        
        # Bias vector (not trained by optimizer, updated manually)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        
        # EMA of expert loads for stability
        self.register_buffer('expert_load_ema', torch.ones(num_experts) / num_experts)
    
    def get_bias(self) -> torch.Tensor:
        """
        Return current bias vector to add to router logits.
        
        Returns:
            expert_bias: [num_experts] bias values
        """
        return self.expert_bias
    
    def update_bias(self, expert_indices: torch.Tensor) -> None:
        """
        Update bias based on current batch's expert assignments.
        
        Called AFTER forward pass, OUTSIDE of backpropagation to avoid
        interfering with gradient flow for main prediction task.
        
        Args:
            expert_indices: [num_tokens, top_k] - selected expert indices
            
        Algorithm:
            1. Compute current expert loads from assignments
            2. Update EMA of loads for stability
            3. Compute bias gradient: load - target_load (1/N)
            4. Update bias: bias -= learning_rate × gradient
        """
        with torch.no_grad():
            batch_size = expert_indices.shape[0]
            top_k = expert_indices.shape[1]
            
            # Compute current load distribution
            current_load = torch.zeros(self.num_experts, device=expert_indices.device)
            for k in range(top_k):
                current_load.scatter_add_(0, expert_indices[:, k],
                                         torch.ones(batch_size, device=expert_indices.device))
            current_load = current_load / (batch_size * top_k)
            
            # Update exponential moving average of load
            self.expert_load_ema = (self.momentum * self.expert_load_ema + 
                                   (1 - self.momentum) * current_load)
            
            # Compute bias update
            target_load = 1.0 / self.num_experts  # Uniform target
            bias_gradient = self.expert_load_ema - target_load
            
            # Gradient descent on bias
            self.expert_bias -= self.bias_lr * bias_gradient


# In[7]:


# ============================================================================
# SECTION 3: MOE CORE COMPONENTS
# ============================================================================

class ExpertLayer(nn.Module):
    """
    Single expert: standard 2-layer feed-forward network.
    
    Architecture:
        x → Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model) → Dropout → output
        
    This is identical to the FFN in PyTorch's TransformerEncoderLayer,
    matching the dense baseline architecture from min_transformer.py.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.05):
        """
        Initialize expert feed-forward network.
        
        Args:
            d_model: Input/output dimension (256)
            d_ff: Hidden layer dimension (512 for standard, 238 for fine-grained)
            dropout: Dropout probability
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)      # Up-projection
        self.w2 = nn.Linear(d_ff, d_model)      # Down-projection
        self.activation = nn.GELU()             # Non-linearity
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert FFN.
        
        Args:
            x: [num_tokens, d_model] input tokens
            
        Returns:
            output: [num_tokens, d_model] expert output
        """
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class MoELayer(nn.Module):
    """
    Flexible Mixture-of-Experts layer supporting multiple configurations.
    
    Features:
        - Standard top-K routing (Exp 2)
        - Shared expert isolation (Exp 3-5)
        - Variable expert counts and sizes (Exp 4)
        - Multiple load balancing strategies (Switch, DeepSeek)
        
    Architecture:
        Input → Router → Top-K Selection → Expert Computation → Weighted Combination → Output
        
    For shared expert variant:
        Output = shared_expert(input) + Σ(gate_i × routed_expert_i(input))
        
    Reference:
        - Switch Transformer (Fedus et al. 2021)
        - DeepSeek-MoE (Dai et al. 2024)
    """
    
    def __init__(self, config: MoEConfig):
        """
        Initialize MoE layer with given configuration.
        
        Args:
            config: MoEConfig specifying architecture and balancing strategy
        """
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_experts - config.num_shared_experts
        self.top_k = config.top_k
        
        # Router: learns to select appropriate experts for each token
        # Only routes to routed experts (shared experts always active)
        self.router = nn.Linear(config.d_model, self.num_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)  # Small init for stability
        
        # Routed experts
        self.experts = nn.ModuleList([
            ExpertLayer(config.d_model, config.d_ff, config.expert_dropout)
            for _ in range(self.num_routed_experts)
        ])
        
        # Shared experts (always activated, if any)
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertLayer(config.d_model, config.d_ff if i == 0 else config.d_ff, 
                          config.expert_dropout)
                for i in range(self.num_shared_experts)
            ])
        
        # Load balancing mechanism
        if config.load_balance_strategy == 'switch':
            self.aux_loss_fn = SwitchAuxiliaryLoss(self.num_routed_experts)
        elif config.load_balance_strategy == 'deepseek':
            self.bias_correction = DeepSeekBiasCorrection(
                self.num_routed_experts, config.bias_lr, config.bias_momentum
            )
    
    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Process:
            1. Flatten input to [num_tokens, d_model]
            2. Compute router logits (optionally with DeepSeek bias)
            3. Select top-k experts per token
            4. Compute expert outputs (only for assigned tokens - efficient!)
            5. Combine weighted expert outputs
            6. Add shared expert outputs (if any)
            7. Compute load balancing loss
            8. Reshape back to original format
            
        Args:
            x: [seq_len, batch_size, d_model] - transformer format
            train: Whether in training mode (for loss computation)
            
        Returns:
            output: [seq_len, batch_size, d_model] - MoE output
            losses: dict with 'aux_loss' and 'expert_usage' statistics
        """
        seq_len, batch_size, d_model = x.shape
        
        # Flatten: [seq_len, batch_size, d_model] → [num_tokens, d_model]
        x_flat = x.reshape(-1, d_model)
        num_tokens = x_flat.shape[0]
        
        # === ROUTER COMPUTATION ===
        router_logits = self.router(x_flat)  # [num_tokens, num_routed_experts]
        
        # Apply DeepSeek bias if using that strategy
        if self.config.load_balance_strategy == 'deepseek':
            bias = self.bias_correction.get_bias()
            router_logits = router_logits + bias.unsqueeze(0)
        
        # Router probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # === TOP-K SELECTION ===
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # top_k_probs: [num_tokens, top_k] - probabilities of selected experts
        # top_k_indices: [num_tokens, top_k] - indices of selected experts
        
        # Renormalize top-k probabilities to sum to 1
        top_k_gates = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # === EXPERT COMPUTATION ===
        output = torch.zeros_like(x_flat)
        
        # Process routed experts
        for expert_idx in range(self.num_routed_experts):
            # Find tokens assigned to this expert
            expert_mask = (top_k_indices == expert_idx)  # [num_tokens, top_k]
            tokens_for_expert_mask = expert_mask.any(dim=-1)  # [num_tokens]
            
            if not tokens_for_expert_mask.any():
                continue  # Skip unused expert (efficient!)
            
            # Get tokens for this expert
            expert_tokens = x_flat[tokens_for_expert_mask]  # [num_expert_tokens, d_model]
            
            # Expert forward pass
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Get corresponding gate weights
            expert_gates = torch.zeros(tokens_for_expert_mask.sum(), device=x.device)
            token_positions = torch.where(tokens_for_expert_mask)[0]
            
            for i, token_idx in enumerate(token_positions):
                # Find which k position has this expert
                k_positions = torch.where(top_k_indices[token_idx] == expert_idx)[0]
                if len(k_positions) > 0:
                    expert_gates[i] = top_k_gates[token_idx, k_positions[0]]
            
            # Add weighted expert output
            output[tokens_for_expert_mask] += expert_output * expert_gates.unsqueeze(-1)
        
        # Add shared expert outputs (if any)
        if self.num_shared_experts > 0:
            for shared_expert in self.shared_experts:
                shared_output = shared_expert(x_flat)
                output += shared_output / self.num_shared_experts
        
        # Reshape back to sequence format
        output = output.reshape(seq_len, batch_size, d_model)
        
        # === COMPUTE LOSSES ===
        losses = {}
        
        if train:
            # Auxiliary loss (Switch)
            if self.config.load_balance_strategy == 'switch':
                losses['aux_loss'] = self.aux_loss_fn(router_probs, top_k_indices)
            else:
                losses['aux_loss'] = torch.tensor(0.0, device=x.device)
            
            # Update DeepSeek bias (outside backprop)
            if self.config.load_balance_strategy == 'deepseek':
                self.bias_correction.update_bias(top_k_indices)
            
            # Track expert usage for monitoring
            with torch.no_grad():
                expert_usage = torch.zeros(self.num_routed_experts, device=x.device)
                for k in range(self.top_k):
                    expert_usage.scatter_add_(0, top_k_indices[:, k],
                                            torch.ones(num_tokens, device=x.device))
                losses['expert_usage'] = expert_usage / (num_tokens * self.top_k)
        
        return output, losses


class MoETransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with MoE feed-forward network.
    
    Replaces standard FFN in PyTorch's TransformerEncoderLayer with MoE.
    Keeps multi-head attention unchanged.
    
    Architecture:
        Input → Self-Attention → Add & Norm → MoE FFN → Add & Norm → Output
        
    This maintains compatibility with the hierarchical transformer from min_transformer.py
    while adding conditional computation in the FFN.
    """
    
    def __init__(self, moe_config: MoEConfig, nhead: int = 16, dropout: float = 0.1):
        """
        Initialize MoE transformer encoder layer.
        
        Args:
            moe_config: MoEConfig for feed-forward network
            nhead: Number of attention heads (16 from min_transformer.py)
            dropout: Dropout probability
        """
        super().__init__()
        d_model = moe_config.d_model
        
        # Multi-head attention (unchanged from standard transformer)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # MoE FFN (replaces standard FFN)
        self.moe = MoELayer(moe_config)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                train: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE transformer layer.
        
        Args:
            src: [seq_len, batch_size, d_model] input
            src_mask: Attention mask (causal mask for temporal modeling)
            train: Training mode flag
            
        Returns:
            output: [seq_len, batch_size, d_model]
            moe_losses: Dictionary of MoE-specific losses
        """
        # Self-attention block (standard)
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # MoE FFN block
        src2, moe_losses = self.moe(src, train=train)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, moe_losses


# In[8]:


# ============================================================================
# SECTION 4: HIERARCHICAL MOE TRANSFORMER (FULL MODEL)
# ============================================================================

class HierarchicalMoETransformer(nn.Module):
    """
    Hierarchical transformer with MoE in temporal encoder.
    
    Based on min_transformer.py architecture:
        1. Daily code encoder (1 layer, 4 heads) - encodes co-occurring codes within each day
        2. Temporal encoder (6 layers, 16 heads) - encodes temporal patterns across days
           → Layers 2-5 replaced with MoE for conditional computation
           
    Input: [batch, 200 days, 82 features] where features = [age, gender, 80 medical codes]
    Output: [batch, 200 days, 8849 target codes] - next code predictions
    
    Key Design Decisions:
        - Daily encoder stays dense (simple aggregation, doesn't need specialization)
        - Temporal encoder layers 0-1 stay dense (learn basic temporal patterns)
        - Temporal encoder layers 2-5 use MoE (learn specialized patient trajectories)
    """
    
    def __init__(self, cd_cnt: int, 
                 target_cd_cnt: int, 
                 embedding_size: int = 256,
                 moe_config: Optional[MoEConfig] = None,
                 use_moe_from_layer: int = 2,
                 nlayers: int = 6, nhead: int = 16, dropout: float = 0.1,
                 len_dy: int = 200, len_cd: int = 80):
        """50
        Initialize hierarchical MoE transformer.
        
        Args:
            cd_cnt: Size of medical code vocabulary (84010 in data)
            target_cd_cnt: Number of target prediction classes (2767 original in data, test data is 8850)
            embedding_size: Embedding dimension (256)
            moe_config: MoEConfig for temporal encoder (None for dense baseline)
            use_moe_from_layer: Which temporal layer to start using MoE (2 = layers 2-5)
            nlayers: Number of temporal encoder layers (6)
            nhead: Number of attention heads for temporal encoder (16)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding_size = embedding_size
        self.len_dy = len_dy  # Sequence length in days
        self.len_cd = len_cd   # Max codes per day
        self.use_moe_from_layer = use_moe_from_layer
        
        # === EMBEDDINGS ===
        self.embedding_cd = nn.Embedding(cd_cnt, embedding_size)
        self.embedding_gender_cd = nn.Embedding(4, embedding_size)
        self.embedding_age_in_months = nn.Embedding(1440, embedding_size)
        
        # === DAILY CODE ENCODER (Level 1) ===
        # 1 layer, 4 heads, encodes co-occurring codes within same day
        encoder_layers_cd = TransformerEncoderLayer(
            embedding_size, 4, embedding_size, dropout, batch_first=False
        )
        self.transformer_encoder_cd = TransformerEncoder(encoder_layers_cd, 1)
        
        # === TEMPORAL ENCODER WITH MOE (Level 2) ===
        self.temporal_layers = nn.ModuleList()
        
        # Set default MoE config if not provided
        if moe_config is None:
            moe_config = MoEConfig(
                d_model=embedding_size,
                d_ff=512,
                num_experts=8,
                num_shared_experts=0,
                top_k=2,
                load_balance_strategy='switch',
                aux_loss_weight=0.01,
            )
        
        # Build temporal layers: 0-1 dense, 2-5 MoE
        for i in range(nlayers):
            if i >= use_moe_from_layer:
                # MoE layers (conditional computation)
                self.temporal_layers.append(
                    MoETransformerEncoderLayer(moe_config, nhead, dropout)
                )
            else:
                # Standard dense layers (basic temporal patterns)
                self.temporal_layers.append(
                    TransformerEncoderLayer(embedding_size, nhead, 512, dropout, batch_first=False)
                )
        
        # === OUTPUT LAYERS ===
        self.mm = nn.GELU()
        self.decoder_cd = nn.Linear(embedding_size, target_cd_cnt)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_size)
        
        self.init_weights()
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask to prevent attending to future days.
        
        Args:
            sz: Sequence length (200 days)
            
        Returns:
            mask: [sz, sz] causal mask with -inf for future positions
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        """Initialize output layer weights."""
        initrange = 0.1
        nn.init.zeros_(self.decoder_cd.bias)
        nn.init.uniform_(self.decoder_cd.weight, -initrange, initrange)
    
    def forward(self, x: torch.Tensor, return_moe_losses: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through hierarchical MoE transformer.
        
        Process:
            1. Extract and embed age, gender, medical codes
            2. Daily encoding: aggregate codes within each day using attention + pooling
            3. Combine daily representations with demographics
            4. Temporal encoding: model patterns across days with MoE in layers 2-5
            5. Output projection: predict next medical codes
            
        Args:
            x: [batch, 200 days, 82 features] where features = [age, gender, 80 codes]
            return_moe_losses: Whether to return MoE auxiliary losses
            
        Returns:
            cd: [batch, 200, target_cd_cnt] - log probabilities for next code predictions
            moe_losses: dict with 'aux_loss' and 'expert_usage' (if return_moe_losses=True)
        """
        gpu_batchsize = x.shape[0]
        device = x.device
        
        # === EXTRACT AND EMBED INPUTS ===
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())  # [batch, 200, 256]
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())          # [batch, 200, 256]
        cd = self.embedding_cd(x[:, :, 2:].long())                        # [batch, 200, 80, 256]
        cd_res = cd.sum(-2)  # Residual connection: [batch, 200, 256]
        
        # === DAILY CODE ENCODING ===
        # Reshape to process all days in parallel
        cd = cd.reshape(gpu_batchsize * self.len_dy, self.len_cd, self.embedding_size)
        cd = torch.swapaxes(cd, 0, 1)  # [80 codes, batch*200 days, 256]
        
        # Apply daily encoder (captures co-occurring codes)
        cd = self.transformer_encoder_cd(cd)
        
        # Max pooling across codes dimension
        cd = cd.permute(1, 2, 0)  # [batch*200, 256, 80]
        cd = nn.MaxPool1d(self.len_cd)(cd)  # [batch*200, 256, 1]
        cd = cd.reshape(gpu_batchsize, self.len_dy, self.embedding_size)  # [batch, 200, 256]
        
        # === COMBINE WITH DEMOGRAPHICS ===
        cd = cd_res + cd + gender_cd + age_in_months
        cd = self.mm(cd)
        cd = self.norm(cd)
        cd = torch.swapaxes(cd, 0, 1)  # [200 seq_len, batch, 256] - sequence first for transformer
        
        # === TEMPORAL ENCODING WITH MOE ===
        mth_mask = self._generate_square_subsequent_mask(self.len_dy).to(device)
        
        # Accumulate MoE losses
        total_aux_loss = torch.tensor(0.0, device=device)
        expert_usage_list = []
        
        for i, layer in enumerate(self.temporal_layers):
            if i >= self.use_moe_from_layer:
                # MoE layer
                cd, moe_losses = layer(cd, src_mask=mth_mask, train=self.training)
                if self.training and return_moe_losses:
                    total_aux_loss += moe_losses['aux_loss']
                    if 'expert_usage' in moe_losses:
                        expert_usage_list.append(moe_losses['expert_usage'])
            else:
                # Standard dense layer
                cd = layer(cd, src_mask=mth_mask)
        
        # === OUTPUT PROCESSING ===
        cd = torch.swapaxes(cd, 0, 1)  # [batch, 200, 256]
        cd = self.norm(cd)
        cd = self.dropout(cd)
        cd = self.decoder_cd(cd)  # [batch, 200, target_cd_cnt]
        
        if return_moe_losses and self.training:
            moe_losses = {
                'aux_loss': total_aux_loss,
            }
            if expert_usage_list:
                moe_losses['expert_usage'] = torch.stack(expert_usage_list).mean(dim=0)
            return cd, moe_losses
        
        return cd, {}


# In[9]:


def compute_moe_specific_metrics(
    model: HierarchicalMoETransformer,
    val_data: pd.DataFrame,
    prepare_tensor_fn,
    device: torch.device,
    batch_size: int = 16
) -> Optional[Dict[str, any]]:
    """
    Compute MoE-specific diagnostic metrics.
    
    Metrics:
        - Expert loads: Fraction of tokens routed to each expert
        - Balance score: Standard deviation of loads (lower = more balanced)
        - Expert collapse: Any expert with < 5% usage (indicates problem)
        
    Args:
        model: HierarchicalMoETransformer
        val_data: Validation DataFrame
        prepare_tensor_fn: Tensor preparation function
        device: Torch device
        batch_size: Batch size
        
    Returns:
        metrics: Dict with MoE diagnostics, or None if not an MoE model
    """
    model.eval()
    
    expert_loads = []
    
    with torch.no_grad():
        nbatch = len(val_data) // batch_size
        
        for i in range(nbatch):
            batch = val_data.iloc[i*batch_size:(i+1)*batch_size]
            dt_cnt, x, y = prepare_tensor_fn(batch, device)
            
            # Forward pass
            _, moe_losses = model(x, return_moe_losses=True)
            
            if 'expert_usage' in moe_losses:
                expert_loads.append(moe_losses['expert_usage'].cpu())
    
    if not expert_loads:
        return None
    
    # Aggregate across batches
    expert_loads = torch.stack(expert_loads).mean(dim=0).numpy()
    
    metrics = {
        'expert_loads': expert_loads,
        'balance_score': expert_loads.std(),
        'min_expert_usage': expert_loads.min(),
        'max_expert_usage': expert_loads.max(),
        'expert_collapse': (expert_loads < 0.05).any(),
    }
    
    return metrics


# In[10]:


# Add these helper functions after the imports in moe_experiments.py
# (around line 34, after "import time")

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def conv_cd(ipt: str, data_config: DataConfig) -> List[List[int]]:
    """
    Convert code string to 2D list of integers.
    
    Format: "code1,code2*code3,code4*..." where * separates days, , separates codes within day
    
    Args:
        ipt: Input string with format "day1codes*day2codes*..."
        len_dy: Maximum number of days (200)
        len_cd: Maximum codes per day (80)
        
    Returns:
        List of lists: [[day1_codes], [day2_codes], ...] padded to [len_dy, len_cd]
        
    Example:
        "1,2,3*4,5*6" -> [[1,2,3,0,...], [4,5,0,...], [6,0,...], [0,0,...], ...]
    """
    ipt = ipt.split('*')
    ipt = ipt[:data_config.len_dy]
    ipt = ipt + (data_config.len_dy - len(ipt)) * ['']
    ipt = [dy.split(',') for dy in ipt]
    ipt = [[int(cd) if cd != '' else 0 for cd in dy] for dy in ipt]
    ipt = [dy + (data_config.len_cd - len(dy)) * [0] for dy in ipt]
    return ipt



def conv_age_gender(ipt: str, data_config: DataConfig) -> List[int]:
    """
    Convert age/gender string to list of integers.
    
    Format: "value1*value2*..." where each value is age in months or gender code
    
    Args:
        ipt: Input string with format "val1*val2*..."
        len_dy: Maximum number of days (200)
        max_age: Maximum age value (1439 for 120 years in months)
        
    Returns:
        List of integers padded to len_dy
        
    Example:
        "360*361*362" -> [360, 361, 362, 0, 0, ...] (up to 200 values)
    """
    ipt = ipt.split('*')
    ipt = ipt[:data_config.len_dy]
    ipt = [min(int(cd), data_config.max_age) for cd in ipt]
    ipt = ipt + (data_config.len_dy - len(ipt)) * [0]
    return ipt


def conv_target(target, data_config: DataConfig, model_config: ModelConfig):
    """Parse target codes string into nested list (multiple codes per day).
    
    Args:
        target: Target string "codes_day0*codes_day1*..."
        data_config: DataConfig with len_dy
    """
    target = target.split('*')
    target = target[:data_config.len_dy]
    # Add pad to the days
    target = target + (data_config.len_dy - len(target)) * ['']
    target = [dy.split(',') for dy in target]
    # Convert 1-indexed to 0-indexed
    # Convert with strict bounds checking
    result = []
    for dy in target:
        day_codes = []
        for cd in dy:
            if cd != '' and cd != '0':
                code_idx = int(cd)  # 0 indexed cd
                # Clip to valid range [0, target_cd_cnt]
                if code_idx >= model_config.target_cd_cnt:
                    # This should NEVER happen if target_cd_cnt is correct
                    raise ValueError(
                        f"Target code {code_idx} >= vocab size {model_config.target_cd_cnt}. "
                        f"Max code seen: {max_seen}. "
                        f"Fix: Set target_cd_cnt = {max_seen + 1}"
                    )
                day_codes.append(code_idx)
            else:
                day_codes.append(0)
        result.append(day_codes if day_codes else [0])
    
    return result


def prepare_tensor(batch: pd.DataFrame, 
                   device: torch.device,
                   data_config: DataConfig,
                   model_config: ModelConfig) -> Tuple[List[int], torch.Tensor, List[List[int]]]:
    """
    Prepare tensors from DataFrame batch for model input.
    
    This function converts raw data (stored as strings) into PyTorch tensors.
    Compatible with both training (returns targets) and inference (can ignore targets).
    
    Expected DataFrame columns:
        - age_in_months: String format "age1*age2*..." (age in months per day)
        - gender_cd: String format "gender1*gender2*..." (gender code per day)
        - cd: String format "code1,code2*code3,code4*..." (medical codes)
        - dt_cnt: Integer, number of actual days in sequence (rest is padding)
        
    Args:
        batch: DataFrame with batch_size rows
        device: PyTorch device (cuda or cpu)
        data_config: DataConfig with column names and sequence params
        model_config: ModelConfig
        
    Returns:
        dt_cnt: List of actual day counts per sample in batch
        x: Tensor [batch_size, 200, 82] where 82 = [age, gender, 80 codes]
        y: List of lists, target codes for each sample (for loss computation)
        
    Example shape:
        dt_cnt: List of actual day counts per sample
        x: Tensor [batch_size, len_dy, 2+len_cd] where 2 = [age, gender]
        y: List of lists, target codes [[day0_codes], [day1_codes], ...]
    """
    batch_size = len(batch)
    
    # Parse age
    age_in_months = [conv_age_gender(ipt, data_config) 
                     for ipt in batch[data_config.age_column].tolist()]
    age_in_months = torch.tensor(age_in_months, dtype=torch.long).to(device)
    age_in_months = age_in_months.reshape(batch_size, data_config.len_dy, 1)
    
    # Parse gender
    gender_cd = [conv_age_gender(ipt, data_config) 
                 for ipt in batch[data_config.gender_column].tolist()]
    gender_cd = torch.tensor(gender_cd, dtype=torch.long).to(device)
    gender_cd = gender_cd.reshape(batch_size, data_config.len_dy, 1)
    
    # Parse input codes
    cd = [conv_cd(ipt, data_config) 
          for ipt in batch[data_config.input_code_column].tolist()]
    cd = torch.tensor(cd, dtype=torch.long).to(device)
    
    # Concatenate: [batch, len_dy, 1+1+len_cd]
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)
    
    # Parse targets (nested list format)
    dt_cnt = batch[data_config.dt_cnt_column].tolist()
    y = [conv_target(target, data_config, model_config) 
         for target in batch[data_config.target_code_column].tolist()]
    
    return dt_cnt, x, y

def create_multihot_encoding(
    y: List[List[int]], 
    num_samples: int,
    vocab_size: int,
    device: torch.device,
    validate: bool = True
) -> torch.Tensor:
    """
    Create multi-hot encoding with optional validation.
    
    Args:
        y: Nested list of target codes [[day0_codes], [day1_codes], ...]
        num_samples: Number of samples (after dt_cnt filtering)
        vocab_size: Target vocabulary size
        device: PyTorch device
        validate: Whether to validate code ranges (slower but safer)
        
    Returns:
        y_cd: Multi-hot tensor [num_samples, vocab_size]
    """
    y_cd = torch.zeros(num_samples, vocab_size, device=device)
    
    invalid_count = 0
    max_code_seen = 0
    
    for j in range(num_samples):
        for k in y[j]:
            if k == 0:
                continue  # Skip padding
            
            max_code_seen = max(max_code_seen, k)
            
            if 0 < k < vocab_size:
                y_cd[j, k] = 1
            else:
                invalid_count += 1
                if validate and invalid_count == 1:
                    raise ValueError(
                        f"Code {k} out of bounds [0, {vocab_size}). "
                        f"Max code: {max_code_seen}. "
                        f"Increase target_cd_cnt to {max_code_seen + 1}"
                    )
    
    if invalid_count > 0 and not validate:
        print(f"    ⚠️ Skipped {invalid_count} out-of-bounds codes")
    
    return y_cd


# ============================================================================
# BENCHMARKING UTILITY (OPTIONAL)
# ============================================================================

def benchmark_throughput(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    num_iters: int = 100,
    warmup_iters: int = 10
) -> float:
    """
    Benchmark model throughput (samples per second).
    
    Useful for estimating training time and costs.
    
    Args:
        model: Model to benchmark
        device: Device to run on
        batch_size: Batch size
        num_iters: Number of iterations to measure
        warmup_iters: Warmup iterations (excluded from timing)
        
    Returns:
        samples_per_second: Throughput in samples/sec
        
    Example usage:
        model = HierarchicalMoETransformer(...)
        throughput = benchmark_throughput(model, device)
        print(f"Throughput: {throughput:.2f} samples/sec")
    """
    model.eval()
    
    print("Generating dummy data...")
    x = torch.zeros((batch_size, 200, 82), dtype=torch.long).to(device)
    
    # Age: random values in [0, 1439]
    x[:, :, 0] = torch.randint(0, 1440, (batch_size, 200), dtype=torch.long).to(device)
    
    # Gender: random values in [0, 3] (4 categories)
    x[:, :, 1] = torch.randint(0, 4, (batch_size, 200), dtype=torch.long).to(device)
    
    # Medical codes: random values in [0, min(10000, cd_cnt))
    # Use smaller range for efficiency, but ensure within vocab
    max_code = 10000  # Most codes will be in lower range anyway
    x[:, :, 2:] = torch.randint(0, max_code, (batch_size, 200, 80), dtype=torch.long).to(device)
    
    # Warmup
    print(f"Warming up ({warmup_iters} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            if isinstance(model, HierarchicalMoETransformer):
                _ = model(x, return_moe_losses=False)
            else:
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({num_iters} iterations)...")
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            if isinstance(model, HierarchicalMoETransformer):
                _ = model(x, return_moe_losses=False)
            else:
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    elapsed = time.time() - start
    
    samples_per_second = (num_iters * batch_size) / elapsed
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Time per batch: {elapsed/num_iters*1000:.2f} ms")
    print(f"Throughput: {samples_per_second:.2f} samples/sec")
    print(f"{'='*60}\n")
    
    return samples_per_second


# In[35]:


# ============================================================================
# SECTION 6: TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_data: pd.DataFrame,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],  # ← Add this line!
    criterion: nn.Module,
    device: torch.device,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    moe_config: Optional[MoEConfig],
    epoch: int
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_data: Training DataFrame
        prepare_tensor_fn: Function to prepare tensors from DataFrame
        optimizer: Optimizer
        criterion: Loss function
        device: Torch device
        batch_size: Batch size
        moe_config: MoE configuration (None for dense)
        epoch: Current epoch number
        log_interval: Batches between logging
        
    Returns:
        metrics: Dictionary with training metrics
    """
    model.train()
    nbatch = len(train_data) // training_config.batch_size
    
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    total_loss_sum = 0.0
    
    for i in range(nbatch):
        if i % 1000 == 0:
            print(f'  Epoch {epoch}, Batch {i}/{nbatch}')
            # Report GPU memory usage for multiple GPUs
            if device.type == 'cuda':
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f'    GPU {gpu_id}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved')
        
        optimizer.zero_grad()
        
        # Prepare batch
        batch = train_data.iloc[i*training_config.batch_size:
                                (i+1)*training_config.batch_size]
        dt_cnt, x, y = prepare_tensor(batch, device, data_config, model_config) 
      
        # Forward pass
        is_moe = isinstance(model, HierarchicalMoETransformer)
        if is_moe:
            opt, moe_losses = model(x, return_moe_losses=True)
        else:
            opt, _ = model(x)
            moe_losses = {'aux_loss': torch.tensor(0.0, device=device)}
        
        # Reshape outputs (using config values!)
        opt = opt.reshape(training_config.batch_size * data_config.len_dy, 
                         model_config.target_cd_cnt)
        y = [item for sublist in y for item in sublist]  # Flatten: now length = batch*len_dy
        # Filter opt by dt_cnt
        opt = torch.cat([opt[data_config.len_dy*j:data_config.len_dy*j+dt_cnt[j], :] 
                        for j in range(training_config.batch_size)], dim=0)
        
        # Create multi-hot encoding - iterate over filtered opt length
        y_cd = torch.zeros(len(opt), model_config.target_cd_cnt, device=device)

        for j in range(len(opt)):  # ← Use len(opt), not num_samples
            # j corresponds to j-th filtered day across all patients
            for k in y[j]:  # ← Access y directly (works because of consistent ordering)
                if k != 0 and k < model_config.target_cd_cnt:
                    y_cd[j, k] = 1
                elif k >= model_config.target_cd_cnt:
                    print(f"⚠️ Code {k} >= vocab {model_config.target_cd_cnt}")
                    
        # Compute loss
        pred_loss = criterion(opt, y_cd)  # BCEWithLogitsLoss
        aux_loss = moe_losses['aux_loss']
        
        # Total loss
        if moe_config and moe_config.load_balance_strategy == 'switch':
            total_loss = pred_loss + moe_config.aux_loss_weight * aux_loss
        else:
            total_loss = pred_loss
        
        total_loss.backward()
        # gradient norm logging
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        # track abnormal gradient norm
        if total_grad_norm > training_config.gradient_clip_norm * 2:
            print(f"    ⚠️ High gradient norm: {total_grad_norm:.2f}")
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                      training_config.gradient_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_pred_loss += pred_loss.item()
        total_aux_loss += aux_loss.item()
        total_loss_sum += total_loss.item()
        
        # Log periodically
        if i % training_config.log_interval == 0 and i > 0:
            avg_pred = total_pred_loss / training_config.log_interval
            avg_aux = total_aux_loss / training_config.log_interval
            avg_total = total_loss_sum / training_config.log_interval
            print(f'    Pred Loss: {avg_pred:.4f}, Aux Loss: {avg_aux:.4f}, Total: {avg_total:.4f}')
            
            if 'expert_usage' in moe_losses:
                usage = moe_losses['expert_usage'].cpu().numpy()
                print(f'    Expert Usage: {usage}')
                usage_std = usage.std()
                if usage_std > 0.1:
                    print(f'    ⚠️ WARNING: Expert imbalance (std={usage_std:.4f})')
            
            total_pred_loss = 0.0
            total_aux_loss = 0.0
            total_loss_sum = 0.0
    
    return {'train_loss': total_loss_sum / nbatch if nbatch > 0 else 0.0}


# In[42]:


# ============================================================================
# SECTION 5: EVALUATION METRICS
# ============================================================================

def compute_comprehensive_internal_metrics(
    model: nn.Module,
    val_data: pd.DataFrame,
    criterion: nn.Module,
    device: torch.device,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    code_frequencies: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive internal evaluation metrics for medical code prediction.
    
    Healthcare-Optimized Metrics:
        - Top-K Accuracy (K=1,5,10,20): Clinical utility - "correct code in top-K suggestions"
        - MRR (Mean Reciprocal Rank): Ranking quality across all predictions
        - Stratified Performance: Common vs. Rare vs. Tail code accuracy
        - Validation bce: Optimization objective
        - Perplexity: For literature comparison (secondary metric) not used anymore
        
    Why NOT use perplexity?
        - Confirm that the BCE entropy loss is used and 
        - Perplexity averages over all codes equally (doesn't highlight rare code performance)
        - Clinical reality: rare codes (sepsis, MI) often most important
        - Top-K matches clinical workflow: doctors review multiple suggestions
        
    Reference:
        BEHRT (Li et al. 2020) - uses Top-K as primary metric for clinical transformers
        
    Args:
        model: Trained model (dense or MoE)
        val_data: Validation DataFrame
        prepare_tensor_fn: Function to convert DataFrame rows to tensors
        criterion: Loss function (nn.BCELoss)
        device: Torch device
        code_frequencies: [target_cd_cnt] frequency of each code in training data
        batch_size: Batch size for evaluation
        
    Returns:
        metrics: Dictionary with all internal metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_bce = 0.0
    num_predictions = 0
    
    with torch.no_grad():
        nbatch = len(val_data) // training_config.batch_size
        
        for i in range(nbatch):
            batch = val_data.iloc[i*training_config.batch_size:
                                 (i+1)*training_config.batch_size]
            dt_cnt, x, y = prepare_tensor(batch, device, data_config, model_config)  # ← 3 values!
            # Get the actual model (unwrap if DataParallel)
            actual_model = model.module if isinstance(model, nn.DataParallel) else model            
            # Forward pass
            if isinstance(actual_model, HierarchicalMoETransformer):
                opt, _ = model(x, return_moe_losses=False)
            else:
                opt = model(x)
            
            # Reshape for loss computation
            opt = opt.reshape(training_config.batch_size * data_config.len_dy, 
                            model_config.target_cd_cnt)
            y_list = [item for sublist in y for item in sublist]
            opt = torch.cat([opt[data_config.len_dy*j:data_config.len_dy*j+dt_cnt[j], :] 
                           for j in range(training_config.batch_size)], dim=0)
            
            # Create multi-hot targets
            y_cd = torch.zeros(len(opt), model_config.target_cd_cnt).to(device)
            for j in range(len(opt)):
                for k in y_list[j]:
                    if k != 0:
                        y_cd[j, k] = 1
            
            # Compute bce
            bce_loss = criterion(opt, y_cd)
            total_bce += bce_loss.item() * len(opt)
            num_predictions += len(opt)
            
            # Store for ranking metrics
            all_predictions.append(opt.cpu())
            all_targets.append(y_cd.cpu())
    
    # Aggregate predictions
    val_bce = total_bce / num_predictions
    all_predictions = torch.cat(all_predictions)  # [num_predictions, target_cd_cnt]
    all_targets = torch.cat(all_targets)          # [num_predictions, target_cd_cnt] (multi-hot)
    
    # === TOP-K ACCURACY (Multi-label aware) ===
    top_k_results = {}
    for k in [1, 5, 10, 20]:
        top_k_preds = torch.topk(all_predictions, k, dim=-1).indices  # [N, k]
        
        # For each sample, check if ANY true code is in top-K
        correct = []
        for i in range(len(all_targets)):
            true_codes = all_targets[i].nonzero(as_tuple=True)[0]  # Get all true codes
            if len(true_codes) > 0:
                # Check if any true code is in top-K predictions
                hit = any(code.item() in top_k_preds[i].tolist() for code in true_codes)
                correct.append(hit)
        
        top_k_results[f'top_{k}_acc'] = sum(correct) / len(correct) if correct else 0.0
    
    # === MEAN RECIPROCAL RANK (First true code) ===
    sorted_indices = torch.argsort(all_predictions, dim=-1, descending=True)
    reciprocal_ranks = []
    for i in range(len(all_targets)):
        true_codes = all_targets[i].nonzero(as_tuple=True)[0]
        if len(true_codes) > 0:
            # Find rank of first true code
            first_true = true_codes[0].item()
            rank = (sorted_indices[i] == first_true).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # === STRATIFIED PERFORMANCE (RARE CODE ANALYSIS) ===
    # For multi-label, analyze performance by code frequency
    freq_percentiles = np.percentile(code_frequencies[code_frequencies > 0], [10, 50, 80])

    # Categorize codes into tiers
    common_codes = np.where(code_frequencies > freq_percentiles[2])[0]
    rare_codes = np.where((code_frequencies < freq_percentiles[1]) & (code_frequencies > 0))[0]
    tail_codes = np.where((code_frequencies < freq_percentiles[0]) & (code_frequencies > 0))[0]

    # Compute top-10 accuracy for each tier
    def compute_tier_accuracy(predictions, targets, code_set):
        """Compute accuracy for specific code tier."""
        if len(code_set) == 0:
            return 0.0

        correct = 0
        total = 0
        top_10_preds = torch.topk(predictions, 10, dim=-1).indices

        for i in range(len(targets)):
            true_codes = targets[i].nonzero(as_tuple=True)[0].tolist()
            # Only count samples with at least one code in this tier
            tier_true_codes = [c for c in true_codes if c in code_set]
            if len(tier_true_codes) > 0:
                total += 1
                # Check if any tier code is in top-10
                if any(c in top_10_preds[i].tolist() for c in tier_true_codes):
                    correct += 1

        return correct / total if total > 0 else 0.0

    stratified = {
        'common_codes_top10': compute_tier_accuracy(all_predictions, all_targets, set(common_codes)),
        'rare_codes_top10': compute_tier_accuracy(all_predictions, all_targets, set(rare_codes)),
        'tail_codes_top10': compute_tier_accuracy(all_predictions, all_targets, set(tail_codes)),
    }
    
    # === COMBINE ALL METRICS ===
    metrics = {
        # Primary metrics (clinical decision-making)
        'val_bce': val_bce,
        'top_1_acc': top_k_results['top_1_acc'],
        'top_5_acc': top_k_results['top_5_acc'],
        'top_10_acc': top_k_results['top_10_acc'],
        'top_20_acc': top_k_results['top_20_acc'],
        'mrr': mrr,
        # Secondary metric (literature comparison)
        # 'perplexity': np.exp(val_nll),
        
        # Stratified accuracy
        'common_codes_top10': stratified['common_codes_top10'],
        'rare_codes_top10': stratified['rare_codes_top10'],
        'tail_codes_top10': stratified['tail_codes_top10']
    }
    
    return metrics


# #### Memory management

# In[32]:


import torch
import gc

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
    import gc
    import torch
    
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
    
def monitor_gpu_memory_usage(
    model: torch.nn.Module = None,
    detailed: bool = False,
    return_stats: bool = False
) -> dict:
    """
    Monitor current GPU memory usage across all devices.
    
    Useful for debugging memory issues during training.
    
    Args:
        model: If provided, also report model size
        detailed: If True, show memory breakdown
        
    Returns:
        Dictionary with memory stats for each GPU
    """
    if not torch.cuda.is_available():
        return {}
    
    num_gpus = torch.cuda.device_count()
    stats = {}
    
    print(f"\n{'='*80}")
    print(f"DETAILED GPU MEMORY MONITOR")
    print(f"{'='*80}")
    
    for device_id in range(num_gpus):
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
        max_reserved = torch.cuda.max_memory_reserved(device_id) / 1024**3
        
        # Get total GPU memory
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory / 1024**3
        
        print(f"GPU {device_id} ({props.name}):")
        print(f"  Total Memory:   {total_memory:.2f} GB")
        print(f"  Allocated:      {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
        print(f"  Reserved:       {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
        print(f"  Free:           {total_memory - reserved:.2f} GB")
        
        if detailed:
            print(f"  Peak Allocated: {max_allocated:.2f} GB")
            print(f"  Peak Reserved:  {max_reserved:.2f} GB")
            
            # Memory segments info (PyTorch internal)
            try:
                memory_stats = torch.cuda.memory_stats(device_id)
                active_bytes = memory_stats.get('active_bytes.all.current', 0) / 1024**3
                inactive_bytes = memory_stats.get('inactive_split_bytes.all.current', 0) / 1024**3
                print(f"  Active:         {active_bytes:.2f} GB")
                print(f"  Inactive:       {inactive_bytes:.2f} GB")
            except:
                pass
        if return_stats:
            stats[f'gpu_{device_id}'] = {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'max_reserved': max_reserved,
            'total': total_memory,
            'free': total_memory - reserved
            }
        print()
    
    # Model size if provided
    if model is not None:
        model_params = sum(p.numel() for p in model.parameters())
        model_size = model_params * 4 / 1024**3  # Assume FP32
        
        # If model is DataParallel, calculate replicated size
        if isinstance(model, torch.nn.DataParallel):
            replicated_size = model_size * len(model.device_ids)
            print(f"Model Size (per GPU):  {model_size:.2f} GB")
            print(f"Model Size (total):    {replicated_size:.2f} GB (replicated across {len(model.device_ids)} GPUs)")
        else:
            print(f"Model Size: {model_size:.2f} GB")
        print()
    
    print(f"{'='*80}\n")
    if return_stats: 
        return stats
    
def save_checkpoint_multigpu(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    filepath: str
):
    """
    Save checkpoint compatible with DataParallel.
    
    CRITICAL: DataParallel wraps the model, so state_dict has 'module.' prefix.
    We save the unwrapped state_dict for compatibility.
    """
    # Check if model is wrapped in DataParallel
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()  # ← Unwrap!
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved: {filepath}")


def load_checkpoint_multigpu(
    model: nn.Module,
    optimizer: optim.Optimizer,
    filepath: str,
    device: torch.device
):
    """
    Load checkpoint compatible with DataParallel.
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # If model is wrapped in DataParallel, load to module
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint loaded: {filepath}")
    return checkpoint['epoch'], checkpoint['metrics']
        
        


# #### Run experimentation

# In[11]:


def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    prepare_tensor_fn,
    model_params: Dict,
    training_params: Dict,
    code_frequencies: np.ndarray,
    device: torch.device
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Run a single experiment (one of the 5 configurations).
    
    Args:
        exp_name: Experiment name
        moe_config: MoE configuration (None for dense baseline)
        train_data: Training DataFrame
        val_data: Validation DataFrame  
        prepare_tensor_fn: Tensor preparation function
        model_params: Model hyperparameters (cd_cnt, target_cd_cnt, etc.)
        training_params: Training hyperparameters (lr, epochs, etc.)
        code_frequencies: Code frequency array for stratified evaluation
        device: Torch device
        
    Returns:
        model: Trained model
        metrics: Final evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    
    # Create model
    if moe_config is None:
        # Dense baseline - would use original TransformerModel from min_transformer.py
        # For this implementation, we use HierarchicalMoETransformer without MoE
        print("Model: Dense Baseline (no MoE)")
        model = HierarchicalMoETransformer(
            cd_cnt=model_params['cd_cnt'],
            target_cd_cnt=model_params['target_cd_cnt'],
            embedding_size=model_params['embedding_size'],
            moe_config=None,
            use_moe_from_layer=999,  # Never use MoE
            nlayers=6,
            nhead=16,
            dropout=0.1
        ).to(device)
    else:
        print(f"Model: MoE - {moe_config.num_experts} experts, "
              f"{moe_config.num_shared_experts} shared, top-{moe_config.top_k}")
        model = HierarchicalMoETransformer(
            cd_cnt=model_params['cd_cnt'],
            target_cd_cnt=model_params['target_cd_cnt'],
            embedding_size=model_params['embedding_size'],
            moe_config=moe_config,
            use_moe_from_layer=2,
            nlayers=6,
            nhead=16,
            dropout=0.1
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), 
                           lr=training_params['lr'], 
                           weight_decay=training_params['weight_decay'])
    
    # The task is multi-label code prediction (multiple codes per day), 
    # not single-label. BCEoss expects integer class indices; BCEWithLogitsLoss expects multi-hot vectors.
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print(f"\nTraining for {training_params['epochs']} epochs...")
    training_metrics = []
    
    for epoch in range(training_params['epochs']):
        print(f"\nEpoch {epoch+1}/{training_params['epochs']}")
        
       # Create wrapped prepare_tensor function with prediction mode and target_cd_cnt
        def prepare_tensor_with_mode(batch, device):
            return prepare_tensor_fn(batch, device, prediction_mode, model_params['target_cd_cnt'])
        
        
        # Train
        train_metrics = train_epoch(
            model, train_data, prepare_tensor_fn, optimizer, criterion,
            device, training_params['batch_size'], moe_config, epoch+1
        )
        
        # Evaluate
        print("  Evaluating...")
        internal_metrics = compute_comprehensive_internal_metrics(
            model, val_data, prepare_tensor_fn, criterion, device, 
            code_frequencies, training_params['batch_size']
        )
        
        # MoE-specific metrics
        moe_metrics = None
        if moe_config is not None:
            moe_metrics = compute_moe_specific_metrics(
                model, val_data, prepare_tensor_fn, device, training_params['batch_size']
            )
        
        # Display key metrics
        print(f"\n  PRIMARY METRICS:")
        print(f"    Val BCE:         {internal_metrics['val_bce']:.4f}")
        print(f"    Top-5 Acc:       {internal_metrics['top_5_acc']:.3f} ⭐")
        print(f"    Top-10 Acc:      {internal_metrics['top_10_acc']:.3f} ⭐")
        print(f"    MRR:             {internal_metrics['mrr']:.4f} ⭐")
        print(f"  STRATIFIED (Top-10):")
        print(f"    Common Codes:    {internal_metrics['common_codes_top10']:.3f}")
        print(f"    Rare Codes:      {internal_metrics['rare_codes_top10']:.3f}")
        print(f"    Tail Codes:      {internal_metrics['tail_codes_top10']:.3f} ⭐")
        
        if moe_metrics:
            print(f"  MoE METRICS:")
            print(f"    Balance Score:   {moe_metrics['balance_score']:.4f}")
            print(f"    Expert Loads:    {moe_metrics['expert_loads']}")
            if moe_metrics['expert_collapse']:
                print(f"    ⚠️ WARNING: Expert collapse detected!")
        
        # Store metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['train_loss'],
            **internal_metrics
        }
        if moe_metrics:
            epoch_metrics.update({
                'expert_balance': moe_metrics['balance_score'],
                'expert_collapse': moe_metrics['expert_collapse']
            })
        training_metrics.append(epoch_metrics)
    
    # Final metrics
    final_metrics = training_metrics[-1]
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print(f"{'='*80}")
    print(f"Final Top-10 Accuracy: {final_metrics['top_10_acc']:.3f}")
    print(f"Final Tail Code Top-10: {final_metrics['tail_codes_top10']:.3f}")
    print(f"Final Val bce: {final_metrics['val_bce']:.4f}")
    
    return model, final_metrics


def run_all_experiments(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    prepare_tensor_fn,
    model_params: Dict,
    training_params: Dict,
    device: torch.device
) -> Dict[str, Tuple[nn.Module, Dict]]:
    """
    Run all 5 experiments and compare results.
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        prepare_tensor_fn: Tensor preparation function
        model_params: Model hyperparameters
        training_params: Training hyperparameters
        device: Torch device
        
    Returns:
        results: Dictionary mapping experiment name to (model, metrics)
    """
    print("\n" + "="*80)
    print("STARTING 5-EXPERIMENT ABLATION STUDY")
    print("="*80)
    
    # Prepare code frequencies for stratified evaluation
    print("\nPreparing code frequencies...")
    code_frequencies = np.zeros(model_params['target_cd_cnt'])
    nbatch = len(train_data) // training_params['batch_size']
    train_code_counts = Counter()
    
    for i in range(min(nbatch, 1000)):  # Sample for efficiency
        batch = train_data.iloc[i*training_params['batch_size']:(i+1)*training_params['batch_size']]
        _, _, y  = prepare_tensor_fn(batch, device, 'same_day', model_params['target_cd_cnt'])
        y_flat = [item for sublist in y for item in sublist]
        train_code_counts.update(y_flat)
    
    for code_idx, count in train_code_counts.items():
        if code_idx < len(code_frequencies):
            code_frequencies[code_idx] = count
    
    print(f"Computed frequencies for {len(train_code_counts)} unique codes")
    
    # Get experiment configurations
    configs = get_experiment_configs()
    
    # Run all experiments
    results = {}
    all_metrics = []
    
    for exp_name, moe_config in configs.items():
        start_time = time.time()
        
        model, metrics = run_single_experiment(
            exp_name=exp_name,
            moe_config=moe_config,
            train_data=train_data,
            val_data=val_data,
            prepare_tensor_fn=prepare_tensor_fn,
            model_params=model_params,
            training_params=training_params,
            code_frequencies=code_frequencies,
            device=device
        )
        
        elapsed = time.time() - start_time
        metrics['training_time_seconds'] = elapsed
        metrics['experiment'] = exp_name
        
        results[exp_name] = (model, metrics)
        all_metrics.append(metrics)
    
    # === FINAL COMPARISON ===
    print("\n" + "="*80)
    print("FINAL COMPARISON: ALL 5 EXPERIMENTS")
    print("="*80)
    
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.set_index('experiment')
    
    # Display key metrics
    key_metrics = ['top_10_acc', 'tail_codes_top10', 'mrr', 'val_bce', 'top_5_acc']
    print("\nPrimary Metrics:")
    print(comparison_df[key_metrics].to_string())
    
    # Rank experiments
    comparison_df['rank_top10'] = comparison_df['top_10_acc'].rank(ascending=False)
    comparison_df['rank_tail'] = comparison_df['tail_codes_top10'].rank(ascending=False)
    comparison_df['rank_bce'] = comparison_df['val_bce'].rank()  # Lower is better
    comparison_df['overall_rank'] = (comparison_df['rank_top10'] + 
                                     comparison_df['rank_tail'] + 
                                     comparison_df['rank_bce']) / 3
    
    comparison_df = comparison_df.sort_values('overall_rank')
    
    print("\n" + "="*80)
    print("OVERALL RANKING")
    print("="*80)
    print(comparison_df[['top_10_acc', 'tail_codes_top10', 'val_bce', 'overall_rank']].to_string())
    
    return results


# ### Training

# In[16]:


import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
client = bigquery.Client()
credentials, project= google.auth.default()
print('credentials:', credentials, ', project:', project)


# In[18]:


# Load data
input_sql = """
select * from
anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_o3_score_ending
limit 2000
"""
# input_data = client.query(input_sql).to_dataframe() 


# In[22]:


import pandas as pd
df_train = pd.read_feather("sample_data/mdcd_train_8000.feather")
df_val = pd.read_feather("sample_data/mdcd_val_2000.feather")


# In[13]:


df_train.head()


# #### Initiate parameters

# In[13]:


# === 1. CREATE CONFIGURATIONS ===
data_config = DataConfig(
    input_code_column='cd',
    target_code_column='target_cd',  # ← YOUR COLUMN NAME!
    age_column='age_in_months',
    gender_column='gender_cd',
    dt_cnt_column='dt_cnt',
    len_dy=200,   
    len_cd=80
)

model_config = ModelConfig(
    cd_cnt=84010,           # Input vocabulary
    target_cd_cnt=8850,    # Predicted target dimension
    embedding_size=256,
    nhead=16,
    nhid=512,
    nlayers=6,
    use_moe_from_layer=2,
)

training_config = TrainingConfig(
    batch_size=128,
    learning_rate=1e-4,
    weight_decay=0.01,
    epochs=1,
    gradient_clip_norm=1.0,
    device='cuda',
    num_gpus=4,
    parallel=True
)

moe_config = MoEConfig(
    d_model=256,
    d_ff=512,
    num_experts=8,
    num_shared_experts=1,
    top_k=2,
    load_balance_strategy='switch',
    aux_loss_weight=0.01,
)


# In[15]:


# Check the max number of voc counts
all_target_codes = []
for target_str in df_train['target_cd'].tolist():
    codes = [int(c) for c in target_str.replace('*', ',').split(',') if c and c != '0']
    all_target_codes.extend(codes)

max_target = max(all_target_codes)
min_target = min(all_target_codes)
unique_target = len(set(all_target_codes))

print(f"\n{'='*80}")
print(f"ACTUAL DATA VOCABULARY ANALYSIS")
print(f"{'='*80}")
print(f"Target codes (target_cd):")
print(f"  Min value: {min_target}")
print(f"  Max value: {max_target}")
print(f"  Unique codes: {unique_target}")
print(f"  Expected vocab size: {max_target + 1}")
print(f"\nCurrent config:")
print(f"  target_cd_cnt: {model_config.target_cd_cnt}")
print(f"\nREQUIRED FIX:")
print(f"  target_cd_cnt = {max_target + 1}")
print(f"{'='*80}\n")


# In[15]:


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[18]:


def init_model_multgpu(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    moe_config: Optional[MoEConfig] = None
) -> nn.Module:
    """
    Create model with DataParallel support.
    
    DataParallel splits the batch across GPUs:
    - GPU 0: samples 0-31
    - GPU 1: samples 32-63
    - GPU 2: samples 64-95
    - GPU 3: samples 96-127
    
    Each GPU:
    1. Receives its portion of the batch
    2. Runs forward pass independently
    3. Computes gradients
    4. GPU 0 gathers and averages all gradients
    5. Updates weights (synchronized across all GPUs)
    """
    
    # Create model on CPU first
    model = HierarchicalMoETransformer(
        cd_cnt=model_config.cd_cnt,
        target_cd_cnt=model_config.target_cd_cnt,
        embedding_size=model_config.embedding_size,
        moe_config=moe_config,
        use_moe_from_layer=model_config.use_moe_from_layer,
        nlayers=model_config.nlayers,
        nhead=model_config.nhead,
        dropout=model_config.dropout,
        len_dy=data_config.len_dy,
        len_cd=data_config.len_cd
    )
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*80}")
    print(f"GPU CONFIGURATION")
    print(f"{'='*80}")
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("⚠️ No GPUs detected! Using CPU...")
        device = torch.device('cpu')
        return model.to(device)
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Move to GPU and wrap with DataParallel
    if training_config.parallel and num_gpus > 1:
        # Use specified number of GPUs (or all available)
        gpu_ids = list(range(min(training_config.num_gpus, num_gpus)))
        print(f"\n✅ Using DataParallel with GPUs: {gpu_ids}")
        print(f"   Batch size per GPU: {training_config.batch_size // len(gpu_ids)}")
        
        model = model.to('cuda:0')  # Move to first GPU
        model = nn.DataParallel(model, device_ids=gpu_ids)
        
        device = torch.device('cuda:0')  # Primary device
    else:
        print(f"\n✅ Using single GPU: cuda:0")
        device = torch.device('cuda:0')
        model = model.to(device)
    
    print(f"{'='*80}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model, device



# # Single GPU model initiation
# model = HierarchicalMoETransformer(
#     cd_cnt=model_config.cd_cnt,
#     target_cd_cnt=model_config.target_cd_cnt,
#     embedding_size=model_config.embedding_size,
#     moe_config=moe_config,
#     use_moe_from_layer=model_config.use_moe_from_layer,
#     nlayers=model_config.nlayers,
#     nhead=model_config.nhead,
#     dropout=model_config.dropout,
#     len_dy=data_config.len_dy,
#     len_cd=data_config.len_cd  
# ).to(device)


# In[19]:


# Create model with DataParallel
model, device = init_model_multgpu(
    model_config, training_config, moe_config
)


# In[20]:


# === 4. OPTIMIZER & CRITERION ===
optimizer = optim.AdamW(
    model.parameters(),
    lr=training_config.learning_rate,
    weight_decay=training_config.weight_decay
)
if training_config.scheduler_type == 'cosine':
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=training_config.epochs)
elif training_config.scheduler_type == 'step':
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=training_config.epochs // 3, gamma=0.1)
else:
    scheduler = None
criterion = nn.BCEWithLogitsLoss()  # ← Multi-label loss


# In[23]:


# === 5. COMPUTE CODE FREQUENCIES FOR STRATIFIED EVAL ===
print("\nComputing code frequencies from training data...")
code_frequencies = np.zeros(model_config.target_cd_cnt)
train_code_counts = Counter()

# Sample training data to compute frequencies
nbatch = min(len(df_train) // training_config.batch_size, 1000)
for i in range(nbatch):
    batch = df_train.iloc[i*training_config.batch_size:(i+1)*training_config.batch_size]
    _, _, y = prepare_tensor(batch, device, data_config, model_config)
    y_flat = [code 
          for patient in y           # Level 1: Iterate patients
          for day in patient          # Level 2: Iterate days per patient
          for code in day             # Level 3: Iterate codes per day
          if code != 0]               # Filter out padding
    train_code_counts.update(y_flat)

for code_idx, count in train_code_counts.items():
    if 0 <= code_idx < model_config.target_cd_cnt:
        code_frequencies[code_idx] = count


# In[33]:


monitor_gpu_memory_usage()


# In[43]:


# cleanup_gpu_memory()

for epoch in range(1):
    print(f"\nEpoch {epoch+1}/{training_config.epochs}")

    # Train
    # train_metrics = train_epoch(
    #     model, df_train, optimizer, None, criterion, device,  # ← Add None here!
    #     data_config, model_config, training_config, moe_config, 
    #     epoch+1
    # )

    # Evaluate
    if (epoch + 1) % training_config.eval_frequency == 0:
        val_metrics = compute_comprehensive_internal_metrics(
            model, df_val, criterion, device,
            data_config, model_config, training_config, code_frequencies
        )

        print(f"\n  Validation Metrics:")
        print(f"    Val bce:     {val_metrics['val_bce']:.4f}")
        print(f"    Top-10 Acc:  {val_metrics['top_10_acc']:.3f}")
        print(f"    MRR:         {val_metrics['mrr']:.4f}")


# In[38]:


train_metrics                                                                                                                                                      


# In[45]:


val_metrics


# #### Time estimate

# In[16]:


total_params = sum(p.numel() for p in base_model.parameters())
print(f"Total parameters: {total_params:,}")


# In[17]:


throughput = benchmark_throughput(base_model, device, batch_size=16)


# In[23]:


# Prepare code frequencies for stratified evaluation
print("\nPreparing code frequencies...")
code_frequencies = np.zeros(model_params['target_cd_cnt'])
nbatch = len(df_train) // training_params['batch_size']
train_code_counts = Counter()

for i in range(min(nbatch, 1000)):  # Sample for efficiency
    batch = df_train.iloc[i*training_params['batch_size']:(i+1)*training_params['batch_size']]
    _, _, y, _ = prepare_tensor(batch, device, 'same_day', model_params['target_cd_cnt'])
    y_flat = [item for sublist in y for item in sublist]
    train_code_counts.update(y_flat)
for code_idx, count in train_code_counts.items():
    if code_idx < len(code_frequencies):
        code_frequencies[code_idx] = count


# In[53]:


# Get experiment configurations
configs = get_experiment_configs()

# Run all experiments
results = {}
all_metrics = []

for exp_name, (moe_config, prediction_mode) in configs.items():
    start_time = time.time()

    model, metrics = run_single_experiment(
        exp_name=exp_name,
        moe_config=moe_config,
        train_data=df_train,
        val_data=df_val,
        prepare_tensor_fn=prepare_tensor,
        model_params=model_params,
        training_params=training_params,
        code_frequencies=code_frequencies,
        device=device
    )

    elapsed = time.time() - start_time
    metrics['training_time_seconds'] = elapsed
    metrics['experiment'] = exp_name

    results[exp_name] = (model, metrics)
    all_metrics.append(metrics)


# In[37]:




