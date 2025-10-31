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


# In[2]:


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

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


# In[3]:


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


# In[4]:


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
    Output: [batch, 200 days, 2767 target codes] - next code predictions
    
    Key Design Decisions:
        - Daily encoder stays dense (simple aggregation, doesn't need specialization)
        - Temporal encoder layers 0-1 stay dense (learn basic temporal patterns)
        - Temporal encoder layers 2-5 use MoE (learn specialized patient trajectories)
    """
    
    def __init__(self, cd_cnt: int, target_cd_cnt: int, embedding_size: int = 256,
                 moe_config: Optional[MoEConfig] = None,
                 use_moe_from_layer: int = 2,
                 nlayers: int = 6, nhead: int = 16, dropout: float = 0.1):
        """
        Initialize hierarchical MoE transformer.
        
        Args:
            cd_cnt: Size of medical code vocabulary (84010 in data)
            target_cd_cnt: Number of target prediction classes (2767 in data)
            embedding_size: Embedding dimension (256)
            moe_config: MoEConfig for temporal encoder (None for dense baseline)
            use_moe_from_layer: Which temporal layer to start using MoE (2 = layers 2-5)
            nlayers: Number of temporal encoder layers (6)
            nhead: Number of attention heads for temporal encoder (16)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding_size = embedding_size
        self.len_dy = 200  # Sequence length in days
        self.len_cd = 80   # Max codes per day
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
        cd = F.log_softmax(cd, dim=-1)
        
        if return_moe_losses and self.training:
            moe_losses = {
                'aux_loss': total_aux_loss,
            }
            if expert_usage_list:
                moe_losses['expert_usage'] = torch.stack(expert_usage_list).mean(dim=0)
            return cd, moe_losses
        
        return cd, {}


# In[5]:


# ============================================================================
# SECTION 5: EVALUATION METRICS
# ============================================================================

def compute_comprehensive_internal_metrics(
    model: nn.Module,
    val_data: pd.DataFrame,
    prepare_tensor_fn,
    criterion: nn.Module,
    device: torch.device,
    code_frequencies: np.ndarray,
    batch_size: int = 16
) -> Dict[str, float]:
    """
    Compute comprehensive internal evaluation metrics for medical code prediction.
    
    Healthcare-Optimized Metrics:
        - Top-K Accuracy (K=1,5,10,20): Clinical utility - "correct code in top-K suggestions"
        - MRR (Mean Reciprocal Rank): Ranking quality across all predictions
        - Stratified Performance: Common vs. Rare vs. Tail code accuracy
        - Validation NLL: Optimization objective
        - Perplexity: For literature comparison (secondary metric)
        
    Why NOT just perplexity?
        - Perplexity averages over all codes equally (doesn't highlight rare code performance)
        - Clinical reality: rare codes (sepsis, MI) often most important
        - Top-K matches clinical workflow: doctors review multiple suggestions
        
    Reference:
        BEHRT (Li et al. 2020) - uses Top-K as primary metric for clinical transformers
        
    Args:
        model: Trained model (dense or MoE)
        val_data: Validation DataFrame
        prepare_tensor_fn: Function to convert DataFrame rows to tensors
        criterion: Loss function (nn.NLLLoss)
        device: Torch device
        code_frequencies: [target_cd_cnt] frequency of each code in training data
        batch_size: Batch size for evaluation
        
    Returns:
        metrics: Dictionary with all internal metrics
    """
    model.eval()
    
    all_predictions = []  # Log probabilities
    all_targets = []      # True code indices
    total_nll = 0.0
    num_predictions = 0
    
    with torch.no_grad():
        nbatch = len(val_data) // batch_size
        
        for i in range(nbatch):
            batch = val_data.iloc[i*batch_size:(i+1)*batch_size]
            dt_cnt, x, y, day_indices = prepare_tensor_fn(batch, device)
            
            # Forward pass
            if isinstance(model, HierarchicalMoETransformer):
                opt, _ = model(x, return_moe_losses=False)
            else:
                opt = model(x)
            
            # Reshape for loss computation
            opt = opt.reshape(batch_size * 200, -1)
            y_list = [item for sublist in y for item in sublist]
            opt = torch.cat([opt[200*j:200*j+dt_cnt[j], :] for j in range(batch_size)], dim=0)
            y_tensor = torch.tensor(y_list, dtype=torch.long).to(device)
            
            # Compute NLL
            nll = criterion(opt, y_tensor)
            total_nll += nll.item() * len(y_tensor)
            num_predictions += len(y_tensor)
            
            # Store for ranking metrics
            all_predictions.append(opt.cpu())
            all_targets.extend(y_list)
    
    # Aggregate predictions
    val_nll = total_nll / num_predictions
    all_predictions = torch.cat(all_predictions)  # [num_predictions, target_cd_cnt]
    all_targets = torch.tensor(all_targets)        # [num_predictions]
    
    # === TOP-K ACCURACY (PRIMARY CLINICAL METRIC) ===
    # For each prediction day:
    # - Get top-K predicted codes
    # - Check if ANY of the true codes (from multi-hot) appear in top-K
        top_k_results = {}
    for k in [1, 5, 10, 20]:
        top_k_preds = torch.topk(all_predictions, k, dim=-1).indices
        in_top_k = (top_k_preds == all_targets.unsqueeze(1)).any(dim=1)
        for i in range(len(all_targets)):
            true_codes = all_targets[i].nonzero(as_tuple=True)[0]  # Get all non-zero positions
            if len(true_codes) > 0:
                # Check if any true code is in top-K predictions
                in_top_k[i] = (top_k_preds[i].unsqueeze(0) == true_codes.unsqueeze(1)).any()
        top_k_results[f'top_{k}_acc'] = in_top_k.float().mean().item()
    
    # === MEAN RECIPROCAL RANK ===
    sorted_indices = torch.argsort(all_predictions, dim=-1, descending=True)
    reciprocal_ranks = []
    for i in range(len(all_targets)):
        rank = (sorted_indices[i] == all_targets[i]).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / rank)
    mrr = np.mean(reciprocal_ranks)
    
    # === STRATIFIED PERFORMANCE (RARE CODE ANALYSIS) ===
    freq_percentiles = np.percentile(code_frequencies, [10, 50, 80])
    target_freqs = code_frequencies[all_targets.numpy()]
    
    # Masks for different code frequency tiers
    common_mask = target_freqs > freq_percentiles[2]  # Top 20% most frequent
    rare_mask = target_freqs < freq_percentiles[1]    # Bottom 50%
    tail_mask = target_freqs < freq_percentiles[0]    # Bottom 10% (very rare)
    
    # Top-10 accuracy for each tier
    top_10_preds = torch.topk(all_predictions, 10, dim=-1).indices
    correct_in_top10 = (top_10_preds == all_targets.unsqueeze(1)).any(dim=1)
    
    stratified = {
        'common_codes_top10': correct_in_top10[common_mask].float().mean().item() if common_mask.any() else 0,
        'rare_codes_top10': correct_in_top10[rare_mask].float().mean().item() if rare_mask.any() else 0,
        'tail_codes_top10': correct_in_top10[tail_mask].float().mean().item() if tail_mask.any() else 0,
    }
    
    # === COMBINE ALL METRICS ===
    metrics = {
        # Primary metrics (clinical decision-making)
        'val_nll': val_nll,
        'top_1_acc': top_k_results['top_1_acc'],
        'top_5_acc': top_k_results['top_5_acc'],
        'top_10_acc': top_k_results['top_10_acc'],
        'top_20_acc': top_k_results['top_20_acc'],
        'mrr': mrr,
        
        # Stratified performance (critical for healthcare)
        'common_codes_top10': stratified['common_codes_top10'],
        'rare_codes_top10': stratified['rare_codes_top10'],
        'tail_codes_top10': stratified['tail_codes_top10'],
        
        # Secondary metric (literature comparison)
        'perplexity': np.exp(val_nll),
    }
    
    return metrics


# In[6]:


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
            dt_cnt, x, y, day_indices = prepare_tensor_fn(batch, device)
            
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


# In[ ]:





# In[7]:


# ============================================================================
# SECTION 6: TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_data: pd.DataFrame,
    prepare_tensor_fn,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    moe_config: Optional[MoEConfig],
    epoch: int,
    log_interval: int = 100
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
    nbatch = len(train_data) // batch_size
    
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    total_loss_sum = 0.0
    
    for i in range(nbatch):
        if i % 1000 == 0:
            print(f'  Epoch {epoch}, Batch {i}/{nbatch}')
        
        optimizer.zero_grad()
        
        # Prepare batch
        batch = train_data.iloc[i*batch_size:(i+1)*batch_size]
        dt_cnt, x, y, day_indices = prepare_tensor_fn(batch, device)  # ← NOW RETURNS day_indices
        
        # Forward pass
        is_moe = isinstance(model, HierarchicalMoETransformer)
        if is_moe:
            opt, moe_losses = model(x, return_moe_losses=True)
        else:
            opt = model(x)
            moe_losses = {'aux_loss': torch.tensor(0.0, device=device)}
        
        opt = opt.reshape(batch_size * len_dy, target_cd_cnt)  # Use len_dy, not hardcoded 200
        y = [item for sublist in y for item in sublist]  # Flatten: [[codes_day0], [codes_day1], ...] → [codes_day0, codes_day1, ...]
        opt = torch.cat([opt[len_dy*j:len_dy*j+dt_cnt[j], :] for j in range(batch_size)], dim=0)
        
        # Create multi-hot encoding
        y_cd = torch.zeros(len(opt), target_cd_cnt).to(device)  # [num_days, target_cd_cnt]

        for j in range(len(opt)):
            for k in y[j]:  # y[j] is a LIST of codes (could be multiple per day)
                if k != 0:
                    y_cd[j, k] = 1  # Multi-hot: multiple 1s possible

        pred_loss = criterion(opt, y_cd)  # BCEWithLogitsLoss expects multi-hot
        
        # MoE auxiliary loss
        aux_loss = moe_losses['aux_loss']
        
        # Total loss
        if moe_config and moe_config.load_balance_strategy == 'switch':
            total_loss = pred_loss + moe_config.aux_loss_weight * aux_loss
        else:
            total_loss = pred_loss
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Track metrics
        total_pred_loss += pred_loss.item()
        total_aux_loss += aux_loss.item()
        total_loss_sum += total_loss.item()
        
        # Log periodically
        if i % log_interval == 0 and i > 0:
            avg_pred = total_pred_loss / log_interval
            avg_aux = total_aux_loss / log_interval
            avg_total = total_loss_sum / log_interval
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
    
    return {
        'train_loss': total_loss_sum / nbatch if nbatch > 0 else 0.0
    }


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
    # not single-label. NLLLoss expects integer class indices; BCEWithLogitsLoss expects multi-hot vectors.
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
        print(f"    Val NLL:         {internal_metrics['val_nll']:.4f}")
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
    print(f"Final Val NLL: {final_metrics['val_nll']:.4f}")
    
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
        _, _, y, _ = prepare_tensor_fn(batch, device, 'same_day', model_params['target_cd_cnt'])
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
    key_metrics = ['top_10_acc', 'tail_codes_top10', 'mrr', 'val_nll', 'top_5_acc']
    print("\nPrimary Metrics:")
    print(comparison_df[key_metrics].to_string())
    
    # Rank experiments
    comparison_df['rank_top10'] = comparison_df['top_10_acc'].rank(ascending=False)
    comparison_df['rank_tail'] = comparison_df['tail_codes_top10'].rank(ascending=False)
    comparison_df['rank_nll'] = comparison_df['val_nll'].rank()  # Lower is better
    comparison_df['overall_rank'] = (comparison_df['rank_top10'] + 
                                     comparison_df['rank_tail'] + 
                                     comparison_df['rank_nll']) / 3
    
    comparison_df = comparison_df.sort_values('overall_rank')
    
    print("\n" + "="*80)
    print("OVERALL RANKING")
    print("="*80)
    print(comparison_df[['top_10_acc', 'tail_codes_top10', 'val_nll', 'overall_rank']].to_string())
    
    return results


# In[12]:


# Add these helper functions after the imports in moe_experiments.py
# (around line 34, after "import time")

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def conv_cd(ipt: str, len_dy: int = 200, len_cd: int = 80) -> List[List[int]]:
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
    ipt = ipt[:len_dy]
    ipt = ipt + (len_dy - len(ipt)) * ['']
    ipt = [dy.split(',') for dy in ipt]
    ipt = [[int(cd) if cd != '' else 0 for cd in dy] for dy in ipt]
    ipt = [dy + (len_cd - len(dy)) * [0] for dy in ipt]
    return ipt  # [len_dy, len_cd]


def conv_age_gender(ipt: str, len_dy: int = 200, max_age: int = 1439) -> List[int]:
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
    ipt = ipt[:len_dy]
    ipt = [min(int(cd), 1439) for cd in ipt]  # Clip age to max 1439 months
    ipt = ipt + (len_dy - len(ipt)) * [0]
    return ipt  # [len_dy]


def conv_target(target, len_dy=200):
    """Parse target codes string into nested list (multiple codes per day)."""
    target = target.split('*')
    target = target[:len_dy]
    target = [dy.split(',') for dy in target]  # Split codes within each day
    target = [[int(cd) if cd != '' else 0 for cd in dy] for dy in target]
    return target  # [[codes_day0], [codes_day1], ...] - nested list


def prepare_tensor(batch: pd.DataFrame, device: torch.device, prediction_mode: str = 'same_day',
                  target_cd_cnt: int = 2767) -> Tuple[List[int], torch.Tensor, List[List[int]], List[List[int]]]:
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
        prediction_mode: 'same_day' or 'next_day'
        target_cd_cnt: Number of target classes (for validation)
        
    Returns:
        dt_cnt: List of actual day counts per sample in batch
        x: Tensor [batch_size, 200, 82] where 82 = [age, gender, 80 codes]
        y: List of lists, target codes for each sample (for loss computation)
        day_indices: List of lists, day index for each target code
        
    Example shape:
        batch_size=16, len_dy=200, len_cd=80
        x shape: [16, 200, 82]
        dt_cnt: [150, 180, 200, ...] (16 values)
        y: [[code1, code2, ...], [code3, code4, ...], ...] (16 lists)
        day_indices: [[0,0,1,1,2,...], [0,0,0,1,...], ...] (16 lists)
    """
    batch_size = len(batch)
    
    # Parse age
    age_in_months = [conv_age_gender(ipt, len_dy) for ipt in batch['age_in_months'].tolist()]
    age_in_months = torch.tensor(age_in_months, dtype=torch.long).to(device)
    age_in_months = age_in_months.reshape(batch_size, len_dy, 1)
    
    # Parse gender
    gender_cd = [conv_age_gender(ipt, len_dy) for ipt in batch['gender_cd'].tolist()]
    gender_cd = torch.tensor(gender_cd, dtype=torch.long).to(device)
    gender_cd = gender_cd.reshape(batch_size, len_dy, 1)
    
    # Parse input codes
    cd = [conv_cd(ipt, len_dy, len_cd) for ipt in batch['cd'].tolist()]
    cd = torch.tensor(cd, dtype=torch.long).to(device)  # [batch, len_dy, len_cd]
    
    # Concatenate: [batch, len_dy, 1+1+len_cd]
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)
    
    # Parse targets (nested list format)
    dt_cnt = batch['dt_cnt'].tolist()
    y = [conv_target(target, len_dy) for target in batch[target_column].tolist()]
    
    return dt_cnt, x, y


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


# ### Training

# In[16]:


import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
client = bigquery.Client()
credentials, project= google.auth.default()
print('credentials:', credentials, ', project:', project)


# In[17]:


# === MODEL PARAMETERS ===
model_params = {
    'cd_cnt': 84010,           # Medical code vocabulary size, expected to change
    'target_cd_cnt': 2767,     # Target prediction classes
    'embedding_size': 256,
}

# === TRAINING PARAMETERS ===
training_params = {
    'batch_size': 16,
    'epochs': 1,               # Increase for real training
    'lr': 1e-4,
    'weight_decay': 0.01,
}


# In[18]:


# Load data
input_sql = """
select * from
anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_o3_score_ending
limit 2000
"""
input_data = client.query(input_sql).to_dataframe() 


# In[19]:


df_train = input_data.iloc[:1500]
df_val = input_data.iloc[1500]


# In[24]:


df_train.head()


# In[20]:


# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[21]:


base_model = HierarchicalMoETransformer(
    cd_cnt=model_params['cd_cnt'],
    target_cd_cnt=model_params['target_cd_cnt'],
    embedding_size=model_params['embedding_size'],
    moe_config=None,
    use_moe_from_layer=999,  # Never use MoE
    nlayers=6,
    nhead=16,
    dropout=0.1
).to(device)


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


# In[22]:


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


# In[ ]:




