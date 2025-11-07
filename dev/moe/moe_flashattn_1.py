#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Mixture-of-Experts (MoE) Experimentation Framework for Hierarchical Clinical Transformer

This module implements a comprehensive 5-experiment ablation study to evaluate MoE integration
into the hierarchical clinical transformer architecture (min_transformer.py).

Experiment Overview:
- Exp 1: Dense Baseline (reference performance)
    - Model: 
        - Head config: nhead=16, head_dim=16 (fixed, matches original)
        - Activation: GELU only
        - Load balance: N/A (no MoE)
        - Mixed precision: FP32 (use_mixed_precision=False)
        - Daily encoder: Standard transformer
- Exp 2: Dense Flash
    - Model: FlashAttentionTransformer
        - Head config: nhead=8, head_dim=32 (from FlashAttentionConfig)
        - Activation: SwiGLU (config.use_swiglu=True)
        - Load balance: N/A (no MoE)
        - Mixed precision: FP16 (use_mixed_precision=True)
        - Daily encoder: Flash Attention
- Exp 3: Standard Top-K MoE (8 experts, top-2)
- Exp 4: Shared Expert MoE (1 shared + 7 routed)
- Exp 5: Fine-Grained MoE (1 shared + 15 routed, smaller experts)
- Exp 6: Auxiliary-Free MoE (DeepSeek bias-based load balancing)
    - Model: FlashMoETransformer
        - Head config: nhead=8, head_dim=32
        - Activation: SwiGLU in temporal layers, GELU in experts
        - Load balance: Switch (exp 3-5) or DeepSeek (exp 6)
        - Mixed precision: FP16
        - Daily encoder: Flash Attention

Based on:
- DeepSeek-MoE ablation methodology
- Switch Transformer load balancing
- BEHRT clinical evaluation metrics

Author: Daniel Xing
Date: 2025-10-24
"""


# In[ ]:


# ============================================================================
# CONFLICT #1: Head Configuration
# ============================================================================
# flash_attention.py uses: nhead=8, head_dim=32 (256/8)
# moe_1.py uses: nhead=16, head_dim=16 (256/16)
# 
# REASON FOR CONFLICT:
# - Flash Attention (xFormers) performs optimally with head_dim=32, 64, or 128
# - Original min_transformer uses nhead=16 which gives head_dim=16 (suboptimal)
#
# QUESTION: Which configuration should we use?
# Option A: nhead=8, head_dim=32 (optimal for Flash Attention)
# Option B: nhead=16, head_dim=16 (matches original, but slower)
# ============================================================================
# CONFLICT #2: Data Preparation - Target Format
# ============================================================================
# flash_attention.py: Parses targets as nested lists (multi-label per day)
# moe_1.py: Same approach but different implementation details
# min_transformer_finetune.py: Original implementation
#
# Key difference: How to handle multi-label targets
# - Each day can have multiple target codes
# - Need to create multi-hot encoding
# ============================================================================
# ============================================================================
# CONFLICT #3: Activation Functions
# ============================================================================
# flash_attention.py: Uses SwiGLU activation
# moe_1.py: Uses standard GELU
# 
# SwiGLU advantages:
# - Better performance in LLMs (used in LLaMA, PaLM)
# - Gating mechanism for selective information flow
# - Requires parameter adjustment for fairness
# 
# Resolution
# - Baseline model: Uses GELU (can't change, matches original)
# - Flash Attention models: Defaults to SwiGLU (use_swiglu=True)
# - MoE experts: Defaults to SwiGELU (use_swiglu=True in ExpertLayer)
# ============================================================================
# ============================================================================
# CONFLICT #6: Mixed Precision Training
# ============================================================================
# flash_attention.py uses mixed precision with GradScaler
# moe_1.py doesn't use mixed precision
# 
# Mixed precision benefits:
# - 2x speedup from FP16 operations
# - 2x memory reduction
# - Maintained accuracy with loss scaling
# ============================================================================


# In[1]:


import pandas as pd
df_train = pd.read_feather("sample_data/mdcd_train_8000.feather")
df_val = pd.read_feather("sample_data/mdcd_val_2000.feather")


# In[2]:


"""
Unified Flash Attention + MoE Clinical Transformer Implementation
================================================================

This module provides a complete implementation integrating:
- Baseline transformer (replicating min_transformer_finetune.py)
- Flash Attention with xFormers
- Mixture of Experts (MoE) with multiple configurations
- 5 experiments as planned

Author: Clinical Transformer Team
Date: 2024
"""

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
import math
import gc
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ### Configurations

# In[3]:


@dataclass
class BaseConfig:
    """
    Base configuration shared across all experiments.
    
    Parameters match your updated specifications:
    - len_dy: 200 days (sequence length)
    - len_cd: 80 codes per day
    - target_cd_cnt: 8850 target codes
    - Multi-label loss (BCEWithLogitsLoss)
    """
    # Data dimensions (from your specifications)
    len_dy: int = 200          # Days in sequence (updated from 70)
    len_cd: int = 80           # Codes per day (updated from 25)
    cd_cnt: int = 84010        # Input vocabulary size
    target_cd_cnt: int = 8850  # Target vocabulary (updated from 2767)
    
    # Model architecture
    embedding_size: int = 256  # Embedding dimension
    nhid: int = 512           # FFN hidden dimension
    nlayers: int = 6          # Number of temporal encoder layers
    dropout: float = 0.1      # Dropout rate (updated from 0.05)
    
    # Embeddings
    gender_vocab: int = 4     # Gender categories
    age_vocab: int = 1440     # Age in months (120 years)
    
    # Training
    batch_size: int = 64     # Batch size
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0  # Gradient clipping norm
    
    # Device
    device: str = 'cuda'
    
    # Loss function
    criterion: str = 'BCEWithLogitsLoss'  # Multi-label loss

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
    expert_dropout: float = 0.05
    n_head = 8
    head_dim = 16
    
    # Load balancing
    load_balance_strategy: str = 'switch'  # 'switch' or 'deepseek'
    aux_loss_weight: float = 0.01
    bias_lr: float = 1e-5
    bias_momentum: float = 0.9
    
    # Optional
    z_loss_weight: float = 0.0
    use_moe_from_layer: int = 2  # Start MoE from layer 2



# In[4]:


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
    
    # Exp 1: Pure baseline (standard everything)
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
            use_moe_from_layer=2
        ),
        False  # Flash Attention + Max-Pool (baseline)
    )
    
    # Exp 3b: Standard MoE with learned pooling
    configs['exp3b_moe_learned_pool'] = (
        MoEConfig(
            d_model=256,
            d_ff=512,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            load_balance_strategy='switch',
            aux_loss_weight=0.01,
            expert_dropout=0.05,
            use_moe_from_layer=2
        ),
        True  # Learned Attention Pooling (test with MoE)
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
            aux_loss_weight=0.01,
            expert_dropout=0.05,
            use_moe_from_layer=2
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
            aux_loss_weight=0.01,
            expert_dropout=0.05,
            use_moe_from_layer=2
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
            aux_loss_weight=0.0,  # No auxiliary loss
            bias_lr=1e-5,
            bias_momentum=0.9,
            expert_dropout=0.05,
            use_moe_from_layer=2
        ),
        True  # Use learned pooling
    )
    
    return configs


# ### RPE and Swiglu

# In[5]:


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


# In[6]:


def test_rotary_position_embedding():
    rope = RotaryPositionEmbedding(dim=32, max_seq_len=16).to(device)
    q = torch.randn(2, 4, 16, 32, device=device)
    k = torch.randn(2, 4, 16, 32, device=device)
    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape, "q not equal"
    assert k_rot.shape == k.shape, "k not equal"
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4), "q close failed"
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4), "k close failed"
    print("RoPE forward ✔️")
test_rotary_position_embedding()


# In[7]:


def test_swiglu_forward():
    layer = SwiGLU(d_model=256, d_ff=512, dropout=0.1).to(device)
    x = torch.randn(6, 256, device=device)
    y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print("SwiGLU forward ✔️")
test_swiglu_forward()


# ### Flash attention

# In[8]:


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
                print(f"✓ xFormers available for Flash Attention")
            except ImportError:
                print("xFormers not available - will use standard attention")
        
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


# In[9]:


def test_flash_attention_layer_fallback():
    layer = FlashAttentionLayer(
        d_model=256,
        nhead=8,
        dropout=0.1,
        use_rope=True,
        use_flash=False,  # ensures no xFormers dependency for the smoke test
        max_seq_len=32,
        dtype=torch.float32
    ).to(device)

    x = torch.randn(32, 3, 256, device=device)  # [seq, batch, dim]
    out = layer(x, is_causal=True)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    print("FlashAttentionLayer fallback path ✔️")
test_flash_attention_layer_fallback()


# ### Learned Attention Pooling for daily encoder (Optional and only apply to MOE experimentation set up)

# In[11]:


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


# In[12]:


def test_learned_attention_pooling():
    pooling = LearnedAttentionPooling(d_model=256, dropout=0.0).to(device)
    x = torch.randn(80, 5, 256, device=device)  # [seq, batch, dim]
    pooled = pooling(x)

    assert pooled.shape == (5, 256)
    assert torch.isfinite(pooled).all()
    print("LearnedAttentionPooling ✔️")
test_learned_attention_pooling()


# ### MOE components

# In[14]:


# ============================================================================
# CONFLICT #4: Load Balancing Strategy
# ============================================================================
# moe_1.py implements both Switch and DeepSeek strategies
# flash_attention.py doesn't have MoE
#
# Question: Should we keep both strategies or focus on one?
# Option A: Keep both (more flexible, as in moe_1.py)
# Option B: Use only Switch (simpler, well-tested)
# Option C: Use only DeepSeek (no aux loss, newer)
# ============================================================================

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
        
        # Router network
        self.router = nn.Linear(config.d_model, self.num_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        
        # Routed experts
        self.experts = nn.ModuleList([
            ExpertLayer(
                config.d_model,
                config.d_ff,
                config.expert_dropout,
                use_swiglu=False  # Can be configured
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
                    use_swiglu=False
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
    
    def forward(
        self,
        x: torch.Tensor,
        train: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Process:
        1. Router selects top-K experts per token
        2. Each token processed only by selected experts (efficient!)
        3. Weighted combination of expert outputs
        4. Add shared expert outputs (if any)
        
        Args:
            x: [seq_len, batch_size, d_model] input
            train: Whether in training mode
        
        Returns:
            output: Same shape as input
            losses: Dictionary with 'aux_loss' and 'expert_usage'
        """
        seq_len, batch_size, d_model = x.shape
        
        # Flatten for routing
        x_flat = x.reshape(-1, d_model)  # [num_tokens, d_model]
        num_tokens = x_flat.shape[0]
        
        # Router computation
        router_logits = self.router(x_flat)  # [num_tokens, num_routed_experts]
        
        # Apply DeepSeek bias if used
        if self.config.load_balance_strategy == 'deepseek':
            bias = self.bias_correction.get_bias()
            router_logits = router_logits + bias.unsqueeze(0)
        
        # Get router probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-K experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Renormalize gates
        top_k_gates = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Process tokens through experts
        output = torch.zeros_like(x_flat)
        
        # Efficient expert processing
        for expert_idx in range(self.num_routed_experts):
            # Find tokens for this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue  # Skip unused expert
            
            # Get tokens
            expert_tokens = x_flat[expert_mask]
            
            # Expert forward
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Get gates for these tokens
            token_positions = torch.where(expert_mask)[0]
            expert_gates = torch.zeros(len(token_positions), device=x.device)
            
            for i, pos in enumerate(token_positions):
                k_positions = torch.where(top_k_indices[pos] == expert_idx)[0]
                if len(k_positions) > 0:
                    expert_gates[i] = top_k_gates[pos, k_positions[0]]
            
            # Add weighted output
            output[expert_mask] += expert_output * expert_gates.unsqueeze(-1)
        
        # Add shared experts
        if self.num_shared_experts > 0:
            for shared_expert in self.shared_experts:
                shared_output = shared_expert(x_flat)
                output += shared_output / self.num_shared_experts
        
        # Reshape back
        output = output.reshape(seq_len, batch_size, d_model)
        
        # Compute losses
        losses = {}
        
        if train:
            if self.config.load_balance_strategy == 'switch':
                losses['aux_loss'] = self.aux_loss_fn(router_probs, top_k_indices)
            else:
                losses['aux_loss'] = torch.tensor(0.0, device=x.device)
            
            if self.config.load_balance_strategy == 'deepseek':
                self.bias_correction.update_bias(top_k_indices)
            
            # Track usage
            with torch.no_grad():
                expert_usage = torch.zeros(self.num_routed_experts, device=x.device)
                for k in range(self.top_k):
                    expert_usage.scatter_add_(
                        0,
                        top_k_indices[:, k],
                        torch.ones(num_tokens, device=x.device)
                    )
                losses['expert_usage'] = expert_usage / (num_tokens * self.top_k)
        
        return output, losses


# In[15]:


def test_switch_auxiliary_loss():
    loss_fn = SwitchAuxiliaryLoss(num_experts=4).to(device)
    router_probs = torch.softmax(torch.randn(20, 4, device=device), dim=-1)
    expert_indices = torch.randint(0, 4, (20, 2), device=device)
    loss = loss_fn(router_probs, expert_indices)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    print("SwitchAuxiliaryLoss ✔️")
test_switch_auxiliary_loss()


# In[16]:


def test_deepseek_bias_correction():
    correction = DeepSeekBiasCorrection(num_experts=4, bias_lr=0.1, momentum=0.5)
    before = correction.get_bias().clone()
    expert_indices = torch.randint(0, 4, (16, 2))
    correction.update_bias(expert_indices)
    after = correction.get_bias()

    assert not torch.equal(before, after)
    print("DeepSeekBiasCorrection ✔️")
test_deepseek_bias_correction()


# In[17]:


def test_expert_layer_forward():
    expert = ExpertLayer(d_model=256, d_ff=128, dropout=0.1, use_swiglu=False).to(device)
    x = torch.randn(10, 256, device=device)
    y = expert(x)

    assert y.shape == x.shape
    print("ExpertLayer ✔️")
test_expert_layer_forward()


# In[18]:


def test_moe_layer_forward():
    moe_cfg = MoEConfig(
        d_model=256,
        d_ff=128,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        load_balance_strategy='switch',
        aux_loss_weight=0.1,
        expert_dropout=0.0,
        use_moe_from_layer=0
    )
    moe = MoELayer(moe_cfg).to(device)
    x = torch.randn(12, 3, 256, device=device)  # [seq, batch, dim]
    out, losses = moe(x, train=True)

    assert out.shape == x.shape
    assert 'aux_loss' in losses
    assert torch.isfinite(losses['aux_loss'])
    print("MoELayer ✔️")
test_moe_layer_forward()


# ### Model architecture

# In[33]:


# ============================================================================
# MODEL 1: BASELINE TRANSFORMER (Replicating min_transformer_finetune.py)
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
        
        # ============================================================
        # STEP 1: EXTRACT COMPONENTS
        # ============================================================
        age_in_months = x[:, :, 0].long()  # [batch, len_dy]
        gender_cd = x[:, :, 1].long()       # [batch, len_dy]
        cd = x[:, :, 2:].long()             # [batch, len_dy, len_cd]
        
        # ============================================================
        # STEP 2: EMBED
        # ============================================================
        gender_cd = self.embedding_gender_cd(gender_cd)      # [batch, len_dy, embedding_size]
        age_in_months = self.embedding_age_in_months(age_in_months)  # [batch, len_dy, embedding_size]
        cd = self.embedding_cd(cd)                           # [batch, len_dy, len_cd, embedding_size]
        
        # Residual connection: sum of all code embeddings
        cd_res = cd.sum(-2)  # [batch, len_dy, embedding_size]
        
        # ============================================================
        # STEP 3: DAILY CODE ENCODING
        # ============================================================
        # Reshape to process all days in parallel
        cd = cd.reshape(gpu_batchsize * self.config.len_dy, self.config.len_cd, self.config.embedding_size)
        cd = torch.swapaxes(cd, 0, 1)  # [len_cd, batch*len_dy, embedding_size]
        
        # Apply daily transformer
        cd = self.transformer_encoder_cd(cd)
        
        # Max pooling across codes dimension
        cd = cd.permute(1, 2, 0)  # [batch*len_dy, embedding_size, len_cd]
        cd = nn.MaxPool1d(self.config.len_cd)(cd)  # [batch*len_dy, embedding_size, 1]
        cd = cd.reshape(gpu_batchsize, self.config.len_dy, self.config.embedding_size)
        
        # ============================================================
        # STEP 4: COMBINE REPRESENTATIONS
        # ============================================================
        # Add all embeddings: residual codes + encoded codes + demographics
        cd = cd_res + cd + gender_cd + age_in_months
        cd = self.mm(cd)  # GELU activation
        cd = self.norm(cd)
        
        # ============================================================
        # STEP 5: TEMPORAL ENCODING
        # ============================================================
        # Convert to sequence-first format
        cd = torch.swapaxes(cd, 0, 1)  # [len_dy, batch, embedding_size]
        
        # Generate causal mask
        mth_mask = self._generate_square_subsequent_mask(self.config.len_dy).to(x.device)
        
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

# ============================================================================
# MODEL 2: FLASH ATTENTION TRANSFORMER
# ============================================================================

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
        
        # Extract and embed (same as baseline)
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        cd = self.embedding_cd(x[:, :, 2:].long())
        cd_res = cd.sum(-2)
        
        # Daily encoding
        cd = cd.reshape(gpu_batchsize * self.config.len_dy, self.config.len_cd, self.config.embedding_size)
        cd = torch.swapaxes(cd, 0, 1)  # [len_cd, batch*len_dy, embedding_size]
        # Flash Attention version
        if self.config.use_flash:
            if config.use_learnt_att_pool:
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
                cd = nn.MaxPool1d(self.config.len_cd)(cd)  # [batch*len_dy, embedding_size, 1]
                cd = cd.squeeze(-1)  # [batch*len_dy, embedding_size]
            
            
        else:
            # Standard encoding
            cd = self.transformer_encoder_cd(cd)
            cd = cd.permute(1, 2, 0)
            cd = nn.MaxPool1d(self.config.len_cd)(cd)
        
        # Reshape back
        cd = cd.reshape(gpu_batchsize, self.config.len_dy, self.config.embedding_size)
        
        # Combine representations
        cd = cd_res + cd + gender_cd + age_in_months
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

# ============================================================================
# MODEL 3: FLASH ATTENTION + MOE TRANSFORMER
# ============================================================================

class FlashMoETransformer(nn.Module):
    """
    Hierarchical transformer with Flash Attention + MoE.
    
    Combines:
    1. Flash Attention for memory efficiency
    2. MoE for conditional computation
    3. Flexible configuration for 5 experiments
    
    MoE placement: layers 2-5 (after learning basic patterns)
    """
    
    def __init__(self, config: FlashAttentionConfig, moe_config: Optional[MoEConfig] = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        self.use_moe_from_layer = moe_config.use_moe_from_layer if moe_config else 999
        
        # Embeddings
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        
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
            layer_dict = nn.ModuleDict({
                'attention': attn,
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2,
            })
            layer_dict.is_moe = is_moe
            self.temporal_layers.append(layer_dict)
            
            
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
    
    def forward(self, x: torch.Tensor, return_moe_losses: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with Flash Attention + MoE.
        
        Returns:
            output: [batch, len_dy, target_cd_cnt]
            moe_losses: Dictionary with auxiliary losses
        """
        gpu_batchsize = x.shape[0]
        device = x.device
        
        # Extract and embed
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        cd = self.embedding_cd(x[:, :, 2:].long())
        cd_res = cd.sum(-2)
        
        # Daily encoding with Flash Attention
        cd = cd.reshape(gpu_batchsize * self.config.len_dy, self.config.len_cd, self.config.embedding_size)
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
            cd = nn.MaxPool1d(self.config.len_cd)(cd)  # [batch*len_dy, embedding_size, 1]
            cd = cd.squeeze(-1)  # [batch*len_dy, embedding_size]
        
        # Reshape it back
        cd = cd.reshape(gpu_batchsize, self.config.len_dy, self.config.embedding_size)
        
        # Combine
        cd = cd_res + cd + gender_cd + age_in_months
        cd = self.mm(cd)
        cd = self.norm(cd)
        cd = torch.swapaxes(cd, 0, 1)
        
        # Temporal encoding with Flash + MoE
        total_aux_loss = torch.tensor(0.0, device=device)
        expert_usage_list = []
        
        for i, layer in enumerate(self.temporal_layers):
            # Flash Attention block
            residual = cd
            cd_norm = layer['norm1'](cd)
            cd_attn = layer['attention'](cd_norm, is_causal=True)
            cd = residual + cd_attn
            
            # FFN block (MoE or standard)
            residual = cd
            cd_norm = layer['norm2'](cd)
            
            if layer.is_moe:
                cd_ffn, moe_losses = layer['ffn'](cd_norm, train=self.training)
                if self.training and return_moe_losses:
                    total_aux_loss += moe_losses['aux_loss']
                    if 'expert_usage' in moe_losses:
                        expert_usage_list.append(moe_losses['expert_usage'])
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
            if expert_usage_list:
                moe_losses['expert_usage'] = torch.stack(expert_usage_list).mean(dim=0)
        
        return cd, moe_losses


# In[30]:


def test_baseline_transformer_forward():
    cfg = BaseConfig(len_dy=32, len_cd=40, batch_size=4, device=device.type)
    batch = df_train.head(cfg.batch_size).copy()
    dt_cnt, x, y = prepare_tensor(batch, cfg, device)

    model = BaselineTransformer(cfg).to(device)
    with torch.no_grad():
        out = model(x.to(device))

    assert out.shape == (cfg.batch_size, cfg.len_dy, cfg.target_cd_cnt)
    print("BaselineTransformer forward ✔️")
test_baseline_transformer_forward()


# In[31]:


def test_flash_attention_transformer_forward():
    cfg = FlashAttentionConfig(
        len_dy=32,
        len_cd=40,
        batch_size=4,
        device=device.type,
        use_flash=False,          # fallback path for portability
        use_learnt_att_pool=False,
        dtype=torch.float32,
        nhead=8
    )
    batch = df_train.head(cfg.batch_size).copy()
    dt_cnt, x, y = prepare_tensor(batch, cfg, device)

    model = FlashAttentionTransformer(cfg).to(device)
    with torch.no_grad():
        out = model(x.to(device))

    assert out.shape == (cfg.batch_size, cfg.len_dy, cfg.target_cd_cnt)
    print("FlashAttentionTransformer forward ✔️")
test_flash_attention_transformer_forward()


# In[34]:


def test_flash_moe_transformer_forward():
    cfg = FlashAttentionConfig(
        len_dy=32,
        len_cd=40,
        batch_size=4,
        device=device.type,
        use_flash=False,
        use_learnt_att_pool=True,
        dtype=torch.float32,
        nhead=8
    )
    moe_cfg = MoEConfig(
        d_model=cfg.embedding_size,
        d_ff=128,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        load_balance_strategy='switch',
        aux_loss_weight=0.1,
        expert_dropout=0.0,
        use_moe_from_layer=0
    )
    batch = df_train.head(cfg.batch_size).copy()
    dt_cnt, x, y = prepare_tensor(batch, cfg, device)

    model = FlashMoETransformer(cfg, moe_cfg).to(device)
    with torch.no_grad():
        out, moe_losses = model(x.to(device), return_moe_losses=True)

    assert out.shape == (cfg.batch_size, cfg.len_dy, cfg.target_cd_cnt)
    assert 'aux_loss' in moe_losses
    print("FlashMoETransformer forward ✔️")
test_flash_moe_transformer_forward()


# ### Training session

# In[21]:


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
                    if 0 < code_val < target_cd_cnt:
                        day_codes.append(code_val)
                    elif code_val >= target_cd_cnt:
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
    target_strs = batch['target_cd'].tolist()
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


def compute_loss(
    output: torch.Tensor,
    y: List[List[List[int]]],
    dt_cnt: List[int],
    config: BaseConfig,
    criterion: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """
    Compute multi-label loss.
    
    Process:
    1. Flatten predictions by valid days
    2. Create multi-hot encoding for targets
    3. Apply BCEWithLogitsLoss
    """
    batch_size = len(dt_cnt)
    
    # Reshape output
    output = output.reshape(batch_size * config.len_dy, config.target_cd_cnt)
    
    # Flatten targets
    y_flat = [item for sublist in y for item in sublist]
    
    # Filter by valid days
    valid_outputs = []
    valid_y_indices = []
    
    for j in range(batch_size):
        start_idx = config.len_dy * j
        end_idx = start_idx + dt_cnt[j]
        valid_outputs.append(output[start_idx:end_idx])
        valid_y_indices.extend(range(start_idx, end_idx))
    
    output = torch.cat(valid_outputs, dim=0)
    
    # Extract valid targets
    y_valid = [y_flat[i] for i in valid_y_indices]
    
    # VECTORIZED: Create multi-hot encoding
    y_cd = create_multihot_targets_vectorized(
        y_valid,
        len(output),
        config.target_cd_cnt,
        device
    )
    
    # Compute loss
    loss = criterion(output, y_cd)
    
    return loss

# Training each epoch
def train_epoch(
    model: nn.Module,
    train_data: pd.DataFrame,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    moe_config: Optional[MoEConfig] = None,
    epoch: int = 0,
    use_bucketing: bool = False
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Clean design:
    - Step 1: Build batch list (either bucketed or sequential)
    - Step 2: Iterate over batch list uniformly
    - Step 3: Dynamic truncation for bucketed batches
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # ============================================================
    # STEP 1: BUILD BATCH LIST
    # ============================================================
    if use_bucketing:
        # Create bucketed batches (groups similar lengths)
        batch_sampler, nbatch = create_bucketing_dataloader(
            train_data, config.batch_size, shuffle=True
        )
        batch_list = list(batch_sampler)  # List of index arrays
        print(f"  Using bucketing: {nbatch} batches")
    else:
        # Create sequential batches
        nbatch = len(train_data) // config.batch_size
        batch_list = [
            list(range(i * config.batch_size, (i + 1) * config.batch_size))
            for i in range(nbatch)
        ]
        print(f"  Using sequential batching: {nbatch} batches")
    
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    
    # ============================================================
    # STEP 2: ITERATE OVER BATCHES (UNIFORM LOGIC)
    # ============================================================
    for batch_idx, indices in enumerate(batch_list):
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{nbatch}')
        
        optimizer.zero_grad()
        
        # Get batch data
        batch = train_data.iloc[indices]
        
        # ============================================================
        # STEP 3: DYNAMIC TRUNCATION (only for bucketed batches)
        # ============================================================
        if use_bucketing:
            # Calculate max actual length in this batch
            max_len = int(batch['dt_cnt'].max())
            # Store original for restoration
            original_len_dy = config.len_dy
            # Truncate config temporarily
            if max_len < config.len_dy:
                config.len_dy = max_len
        else:
            original_len_dy = None  # No truncation needed
        
        # Prepare tensors
        dt_cnt, x, y = prepare_tensor(batch, config, device)
        
        # ============================================================
        # STEP 4: FORWARD PASS
        # ============================================================
        if use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Model forward
                if hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames:
                    output, moe_losses = model(x, return_moe_losses=True)
                else:
                    output = model(x)
                    moe_losses = {}
                
                # Compute loss (vectorized!)
                pred_loss = compute_loss(output, y, dt_cnt, config, criterion, device)
                
                # Add auxiliary loss if MoE
                aux_loss = moe_losses.get('aux_loss', torch.tensor(0.0, device=device))
                if moe_config and moe_config.load_balance_strategy == 'switch':
                    total_loss = pred_loss + moe_config.aux_loss_weight * aux_loss
                else:
                    total_loss = pred_loss
        else:
            # Standard precision (baseline)
            if hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames:
                output, moe_losses = model(x, return_moe_losses=True)
            else:
                output = model(x)
                moe_losses = {}
            
            pred_loss = compute_loss(output, y, dt_cnt, config, criterion, device)
            aux_loss = moe_losses.get('aux_loss', torch.tensor(0.0, device=device))
            
            if moe_config and moe_config.load_balance_strategy == 'switch':
                total_loss = pred_loss + moe_config.aux_loss_weight * aux_loss
            else:
                total_loss = pred_loss
        
        # ============================================================
        # STEP 5: BACKWARD PASS
        # ============================================================
        if use_mixed_precision:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # ============================================================
        # STEP 6: CLEANUP & LOGGING
        # ============================================================
        
        # Restore config if we modified it
        if original_len_dy is not None:
            config.len_dy = original_len_dy
        
        # Track losses
        total_pred_loss += pred_loss.item()
        total_aux_loss += aux_loss.item()
        
        # Log expert usage
        if 'expert_usage' in moe_losses and batch_idx % 100 == 0:
            usage = moe_losses['expert_usage'].cpu().numpy()
            print(f'    Expert usage: {usage}')
            if usage.std() > 0.1:
                print(f'    ⚠️ Expert imbalance (std={usage.std():.3f})')
        
        # Memory cleanup (NO empty_cache in loop!)
        del x, output, pred_loss, total_loss
        if batch_idx % 100 == 0:
            gc.collect()  # Python GC only
            
            # Optional memory monitoring
            if device.type == 'cuda' and batch_idx % 1000 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f'    GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB')
    
    # End-of-epoch cleanup
    if device.type == 'cuda':
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    return {
        'train_loss': total_pred_loss / nbatch,
        'aux_loss': total_aux_loss / nbatch
    }

def evaluate(
    model: nn.Module,
    val_data: pd.DataFrame,
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Computes:
    1. Validation loss
    2. Top-K accuracy (1, 5, 10, 20)
    3. Mean Reciprocal Rank
    """
    model.eval()
    
    nbatch = len(val_data) // config.batch_size
    total_loss = 0.0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(nbatch):
            batch = val_data.iloc[i*config.batch_size:(i+1)*config.batch_size]
            dt_cnt, x, y = prepare_tensor(batch, config, device)
            
            # Forward
            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames:
                        output, _ = model(x, return_moe_losses=False)
                    else:
                        output = model(x)
                    
                    loss = compute_loss(output, y, dt_cnt, config, criterion, device)
            else:
                if hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames:
                    output, _ = model(x, return_moe_losses=False)
                else:
                    output = model(x)
                
                loss = compute_loss(output, y, dt_cnt, config, criterion, device)
            
            total_loss += loss.item()
            
            # Store predictions for metrics
            output = output.reshape(config.batch_size * config.len_dy, config.target_cd_cnt)
            y_flat = [item for sublist in y for item in sublist]
            
            # Filter by valid days
            for j in range(config.batch_size):
                start_idx = config.len_dy * j
                end_idx = start_idx + dt_cnt[j]
                valid_output = output[start_idx:end_idx]
                valid_y = y_flat[start_idx:end_idx]
                
                all_predictions.append(valid_output.cpu())
                all_targets.append(valid_y)
    
    val_loss = total_loss / nbatch
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    
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
    
    return {
        'val_loss': val_loss,
        **top_k_results
    }


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
        drop_last: bool = False
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
        drop_last=False
    )
    
    return sampler, len(sampler)


# #### Test

# In[35]:


def test_prepare_tensor_and_multihot():
    cfg = BaseConfig(batch_size=4, len_dy=32, len_cd=40, device=device.type)
    dt_cnt, x, y = prepare_tensor(df_train.head(cfg.batch_size), cfg, device)

    assert x.shape == (cfg.batch_size, cfg.len_dy, 2 + cfg.len_cd)
    assert len(y) == cfg.batch_size

    y_flat = [codes for day_list in y for codes in day_list]
    multihot = create_multihot_targets_vectorized(
        y_flat[:10],
        num_samples=10,
        vocab_size=cfg.target_cd_cnt,
        device=device
    )

    assert multihot.shape == (10, cfg.target_cd_cnt)
    print("prepare_tensor + multihot ✔️")
test_prepare_tensor_and_multihot()


# In[36]:


def test_compute_loss_smoke():
    cfg = BaseConfig(batch_size=4, len_dy=32, len_cd=40, device=device.type)
    dt_cnt, x, y = prepare_tensor(df_train.head(cfg.batch_size), cfg, device)

    model = BaselineTransformer(cfg).to(device)
    with torch.no_grad():
        logits = model(x.to(device))

    crit = nn.BCEWithLogitsLoss()
    loss = compute_loss(logits, y, dt_cnt, cfg, crit, device)

    assert loss.ndim == 0 and torch.isfinite(loss)
    print("compute_loss ✔️")
test_compute_loss_smoke()


# In[37]:


def test_train_epoch_smoke():
    cfg = BaseConfig(batch_size=4, len_dy=16, len_cd=30, learning_rate=1e-3, device=device.type)
    model = BaselineTransformer(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    crit = nn.BCEWithLogitsLoss()

    train_subset = df_train.head(cfg.batch_size)  # single batch
    metrics = train_epoch(
        model=model,
        train_data=train_subset,
        optimizer=opt,
        scheduler=sched,
        criterion=crit,
        config=cfg,
        device=device,
        use_mixed_precision=False,
        use_bucketing=False
    )

    assert 'train_loss' in metrics and 'aux_loss' in metrics
    print("train_epoch smoke ✔️")
test_train_epoch_smoke()


# In[38]:


def test_evaluate_smoke():
    cfg = BaseConfig(batch_size=4, len_dy=16, len_cd=30, device=device.type)
    model = BaselineTransformer(cfg).to(device)
    crit = nn.BCEWithLogitsLoss()

    # Prime the model with one forward so embeddings are on-device
    dt_cnt, x, y = prepare_tensor(df_train.head(cfg.batch_size), cfg, device)
    with torch.no_grad():
        model(x.to(device))

    val_metrics = evaluate(
        model=model,
        val_data=df_val.head(cfg.batch_size),
        criterion=crit,
        config=cfg,
        device=device,
        use_mixed_precision=False
    )

    assert 'val_loss' in val_metrics and 'top_10_acc' in val_metrics
    print("evaluate smoke ✔️")
test_evaluate_smoke()


# ### Evaluation metrics

# In[40]:


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
    for k in [5, 10, 20]:
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
            # Find rank of first true code
            first_true = true_codes[0]
            rank = (sorted_indices[i] == first_true).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                reciprocal_ranks.append(1.0 / (rank.item() + 1))
    
    metrics['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # F1@K (Harmonic mean of precision and recall)
    for k in [10, 20]:
        if f'recall@{k}' in metrics and f'precision@{k}' in metrics:
            r = metrics[f'recall@{k}']
            p = metrics[f'precision@{k}']
            metrics[f'f1@{k}'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
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
    criterion: nn.Module
) -> Dict[str, float]:
    """
    Loss and calibration metrics.
    
    Returns:
        1. BCE Loss:
           - Primary optimization objective
           - Report both total and per-sample average
        
        2. Calibration Error (ECE):
           - How well do predicted probabilities match actual frequencies?
           - Important for healthcare (confidence matters!)
        
        3. Per-Class Loss Variance:
           - Detect if model ignores certain code categories
           - Important for rare disease detection
    """
    metrics = {}
    
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
    
    metrics['ece'] = ece
    
    # 3. Brier Score (alternative calibration metric)
    brier = ((probs - targets_multihot) ** 2).mean().item()
    metrics['brier_score'] = brier
    
    return metrics

"""
Stratified Performance (Rare Code Analysis)
CRITICAL for clinical AI publication:
- Medical codes have extreme long-tail distribution
- Rare codes (sepsis, MI, rare diseases) are most important
- Must show model doesn't just predict common codes
"""

def compute_stratified_metrics(
    predictions: torch.Tensor,
    targets: List[List[int]],
    code_frequencies: np.ndarray,  # From training data
    vocab_size: int
) -> Dict[str, float]:
    """
    Stratified performance by code frequency.
    
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
    
    # Top-10 accuracy per tier
    def tier_accuracy(code_set, k=10):
        top_k_preds = torch.topk(predictions, k, dim=-1).indices
        correct = 0
        total = 0
        
        for i, target_codes in enumerate(targets):
            tier_true = [c for c in target_codes if c in code_set and c != 0]
            if len(tier_true) > 0:
                total += 1
                if any(c in top_k_preds[i].tolist() for c in tier_true):
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    metrics['common_top10_acc'] = tier_accuracy(common_codes, k=10)
    metrics['medium_top10_acc'] = tier_accuracy(medium_codes, k=10)
    metrics['rare_top10_acc'] = tier_accuracy(rare_codes, k=10)
    metrics['tail_top10_acc'] = tier_accuracy(tail_codes, k=10)
    
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

"""
CRITICAL for comparing architectures:
- Absolute time (wall-clock)
- Normalized throughput (tokens/sec, samples/sec)
- Time breakdown (data loading vs compute)
"""

def compute_training_time_metrics(
    total_train_time: float,  # Seconds
    num_epochs: int,
    num_samples: int,
    num_tokens: int,  # batch_size * len_dy * num_batches
    data_load_time: float = 0.0,  # Optional profiling
    forward_time: float = 0.0,
    backward_time: float = 0.0
) -> Dict[str, float]:
    """
    Training time and throughput metrics.
    
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
    metrics['batches_per_sec'] = (num_samples / config.batch_size) / total_train_time
    
    # Time breakdown (if profiled)
    if data_load_time > 0 or forward_time > 0:
        total_profiled = data_load_time + forward_time + backward_time
        metrics['data_load_percent'] = (data_load_time / total_profiled) * 100
        metrics['forward_percent'] = (forward_time / total_profiled) * 100
        metrics['backward_percent'] = (backward_time / total_profiled) * 100
    
    # Training speed (industry standard: steps per second)
    # Useful for comparing with published baselines
    metrics['steps_per_sec'] = (num_samples / config.batch_size) / total_train_time
    
    return metrics

"""
Understanding training dynamics:
- How fast does model learn?
- Is training stable?
- When do early-stop?
"""

def compute_convergence_metrics(
    epoch_losses: List[float],  # Validation losses per epoch
    epoch_metrics: List[Dict[str, float]],  # All metrics per epoch
    smoothing_window: int = 3
) -> Dict[str, float]:
    """
    Convergence speed and stability metrics.
    
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
    top10_accs = [epoch.get('top_10_acc', 0.0) for epoch in epoch_metrics]
    
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

"""
CRITICAL for MoE publication:
- Expert specialization
- Load balancing quality
- Routing entropy
"""

def compute_moe_performance_metrics(
    expert_usage: torch.Tensor,  # [num_experts] usage distribution
    router_probs_history: List[torch.Tensor],  # Router probabilities over time
    num_experts: int
) -> Dict[str, float]:
    """
    MoE-specific quality metrics.
    
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

"""
CRITICAL for deployment and cost estimation:
- Peak memory (limits batch size)
- Memory efficiency (Flash Attention benefit)
- Per-GPU memory for multi-GPU
"""

def compute_memory_metrics(
    device: torch.device,
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_gpus: int = 1
) -> Dict[str, float]:
    """
    GPU memory usage metrics.
    
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

"""
CRITICAL for fair comparison:
- FLOPs (floating point operations)
- MFU (Model FLOPs Utilization)
- Hardware efficiency
"""

def compute_flops_metrics(
    model_config: BaseConfig,
    batch_size: int,
    seq_len: int,
    num_experts: int = None,
    top_k: int = None,
    actual_throughput: float = None  # tokens/sec from training
) -> Dict[str, float]:
    """
    Computational cost metrics.
    
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
        effective_ffn_layers = n_layers - model_config.use_moe_from_layer  # Only MoE layers
    
    # ============================================================
    # 3. TOTAL FLOPS
    # ============================================================
    # Attention FLOPs (all layers)
    total_attn_flops = attn_flops_per_layer * n_layers
    
    # FFN FLOPs (dense + MoE layers)
    dense_ffn_layers = model_config.use_moe_from_layer if num_experts else n_layers
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

"""
CRITICAL for practical deployment:
- Training cost in dollars
- Cost per experiment
- Cost projections for full training
"""

def compute_cost_metrics(
    training_time_sec: float,
    gpu_type: str = "T4",
    num_gpus: int = 4,
    region: str = "us-central1"  # GCP region
) -> Dict[str, float]:
    """
    Training cost estimation.
    
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
    metrics['cost_per_epoch_usd'] = metrics['cost_usd'] / epochs if hasattr(config, 'epochs') else 0
    
    # 2. Projected costs
    # Typical clinical transformer: 100-300 epochs for convergence
    for num_epochs in [10, 50, 100, 200]:
        cost_projection = (training_time_sec / epochs if hasattr(config, 'epochs') else training_time_sec) * num_epochs / 3600 * rate_total
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

"""
CRITICAL for understanding what works:
- Component contribution
- Architecture variants
- Marginal benefit of each improvement
"""

def compute_ablation_metrics(
    all_experiment_results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Ablation analysis across experiments.
    
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
    baseline_acc = baseline.get('final_top_10_acc', 0)
    baseline_time = baseline.get('training_time_sec', 1)
    
    # ============================================================
    # 1. FLASH ATTENTION IMPACT
    # ============================================================
    flash_dense = all_experiment_results.get('exp2_dense_flash', {})
    if flash_dense:
        # Accuracy impact
        flash_acc_gain = flash_dense['final_top_10_acc'] - baseline_acc
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
        pool_acc_gain = flash_learned['final_top_10_acc'] - flash_dense['final_top_10_acc']
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
        moe_acc_gain = moe_standard['final_top_10_acc'] - flash_dense['final_top_10_acc']
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
        acc_gain = results['final_top_10_acc'] - baseline_acc
        cost = results.get('cost_usd', 0)
        
        if cost > 0:
            metrics[f'{exp_name}_acc_per_dollar'] = acc_gain / cost
        
        # Speedup vs cost ratio
        speedup = baseline_time / results['training_time_sec']
        metrics[f'{exp_name}_speedup_ratio'] = speedup
    
    return metrics

def comprehensive_evaluation(
    model: nn.Module,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: BaseConfig,
    device: torch.device,
    training_time_sec: float,
    epoch_history: List[Dict[str, float]],
    code_frequencies: np.ndarray,
    moe_config: Optional[MoEConfig] = None,
    use_mixed_precision: bool = False
) -> Dict[str, any]:
    """
    Comprehensive evaluation with all metrics.
    
    Returns dictionary organized by category:
    - performance: Task performance metrics
    - efficiency: Time and throughput metrics
    - cost: Resource usage and cost estimates
    - moe: MoE-specific metrics (if applicable)
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_targets_multihot = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    # ============================================================
    # COLLECT VALIDATION PREDICTIONS
    # ============================================================
    print("Collecting predictions...")
    with torch.no_grad():
        nbatch = len(val_data) // config.batch_size
        
        for i in range(nbatch):
            batch = val_data.iloc[i*config.batch_size:(i+1)*config.batch_size]
            dt_cnt, x, y = prepare_tensor(batch, config, device)
            
            # Forward
            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames:
                        output, _ = model(x, return_moe_losses=False)
                    else:
                        output = model(x)
            else:
                if hasattr(model, 'forward') and 'return_moe_losses' in model.forward.__code__.co_varnames:
                    output, _ = model(x, return_moe_losses=False)
                else:
                    output = model(x)
            
            # Process outputs
            output_flat = output.reshape(config.batch_size * config.len_dy, config.target_cd_cnt)
            y_flat = [item for sublist in y for item in sublist]
            
            # Filter valid days
            for j in range(config.batch_size):
                start_idx = config.len_dy * j
                end_idx = start_idx + dt_cnt[j]
                
                valid_output = output_flat[start_idx:end_idx]
                valid_y = y_flat[start_idx:end_idx]
                
                all_predictions.append(valid_output.cpu())
                all_targets.append(valid_y)
                
                # Create multihot for this sample
                for sample_output, sample_y in zip(valid_output, valid_y):
                    multihot = torch.zeros(config.target_cd_cnt)
                    for code in sample_y:
                        if code != 0 and code < config.target_cd_cnt:
                            multihot[code] = 1
                    all_targets_multihot.append(multihot)
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions)
    all_targets_multihot = torch.stack(all_targets_multihot)
    
    # ============================================================
    # COMPUTE ALL METRICS
    # ============================================================
    
    evaluation = {}
    
    # 1. PERFORMANCE METRICS
    print("Computing performance metrics...")
    evaluation['performance'] = {
        **compute_primary_task_metrics(all_predictions, all_targets, config.target_cd_cnt),
        **compute_loss_metrics(all_predictions, all_targets_multihot, criterion),
        **compute_stratified_metrics(all_predictions, all_targets, code_frequencies, config.target_cd_cnt)
    }
    
    # 2. EFFICIENCY METRICS
    print("Computing efficiency metrics...")
    num_samples = len(all_predictions)
    num_tokens = num_samples * config.len_dy
    
    evaluation['efficiency'] = compute_training_time_metrics(
        total_train_time=training_time_sec,
        num_epochs=len(epoch_history),
        num_samples=num_samples * len(epoch_history),  # Total across epochs
        num_tokens=num_tokens * len(epoch_history),
        data_load_time=0.0,  # TODO: Add profiling
        forward_time=0.0,
        backward_time=0.0
    )
    
    # 3. RESOURCE METRICS
    print("Computing resource metrics...")
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 1
    
    evaluation['resources'] = {
        **compute_memory_metrics(device, model, config.batch_size, config.len_dy, num_gpus),
        **compute_flops_metrics(
            config,
            config.batch_size,
            config.len_dy,
            num_experts=moe_config.num_experts if moe_config else None,
            top_k=moe_config.top_k if moe_config else None,
            actual_throughput=evaluation['efficiency']['tokens_per_sec']
        ),
        **compute_cost_metrics(training_time_sec, gpu_type="T4", num_gpus=num_gpus)
    }
    
    # 4. MOE METRICS (if applicable)
    if moe_config is not None:
        print("Computing MoE metrics...")
        # Extract from last epoch
        expert_usage = epoch_history[-1].get('expert_usage', None)
        if expert_usage is not None:
            evaluation['moe'] = compute_moe_performance_metrics(
                expert_usage=expert_usage,
                router_probs_history=[],  # TODO: Collect during training
                num_experts=moe_config.num_experts - moe_config.num_shared_experts
            )
    
    return evaluation


# In[41]:


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


# In[42]:


def test_comprehensive_evaluation_dense():
    cfg = BaseConfig(batch_size=4, len_dy=16, len_cd=30, device=device.type)
    model = BaselineTransformer(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()

    train_subset = df_train.head(cfg.batch_size)
    val_subset = df_val.head(cfg.batch_size)
    epoch_history = [{'val_loss': 1.0, 'top_10_acc': 0.1}]
    code_freq = np.ones(cfg.target_cd_cnt, dtype=np.int32)

    previous = globals().get('config')
    globals()['config'] = cfg  # required by compute_training_time_metrics

    evaluation = comprehensive_evaluation(
        model=model,
        train_data=train_subset,
        val_data=val_subset,
        config=cfg,
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


# ### Run experimentation

# In[32]:


def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 10,
    code_frequencies: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Run a SINGLE experiment.
    
    Args:
        exp_name: Experiment identifier
        moe_config: MoE configuration (None for dense models)
        use_learnt_att_pool: Whether to use learned attention pooling
        train_data: Training DataFrame
        val_data: Validation DataFrame
        device: Torch device
        epochs: Number of epochs
        code_frequencies: Pre-computed code frequencies (optional)
    
    Returns:
        Dictionary with:
        - experiment: name
        - parameters: model size
        - final_train_loss: final training loss
        - final_val_loss: final validation loss
        - final_top_10_acc: final top-10 accuracy
        - training_time_sec: total training time
        - all_epochs: list of per-epoch metrics
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    
    # ============================================================
    # MODEL CREATION (3 model types)
    # ============================================================
    
    if exp_name == 'exp1_dense_baseline':
        # Type 1: Pure baseline (no Flash, no MoE)
        config = BaseConfig()
        model = BaselineTransformer(config).to(device)
        use_mixed_precision = False
        use_bucketing = False  # Baseline doesn't benefit from bucketing
        print("Model Type: Baseline Transformer")
        print("  Precision: FP32")
        print("  Daily Encoder: Standard Transformer + Max-Pool")
        
    elif exp_name in ['exp2_dense_flash', 'exp2b_flash_learned_pool']:
        # Type 2: Flash Attention (no MoE)
        config = FlashAttentionConfig(
            nhead=8,
            use_swiglu=True,
            dtype=torch.float16,
            use_learnt_att_pool=use_learnt_att_pool
        )
        model = FlashAttentionTransformer(config).to(device)
        use_mixed_precision = True
        use_bucketing = True  # Flash benefits from bucketing
        
        pooling_str = "Learned Attention Pooling" if use_learnt_att_pool else "Flash Attention + Max-Pool"
        print("Model Type: Flash Attention Transformer")
        print(f"  Precision: FP16")
        print(f"  Daily Encoder: {pooling_str}")
        
    else:
        # Type 3: Flash + MoE
        config = FlashAttentionConfig(
            nhead=8,
            use_swiglu=True,
            dtype=torch.float16,
            use_learnt_att_pool=use_learnt_att_pool
        )
        model = FlashMoETransformer(config, moe_config).to(device)
        use_mixed_precision = True
        use_bucketing = True
        
        pooling_str = "Learned Attention Pooling" if use_learnt_att_pool else "Flash Attention + Max-Pool"
        print("Model Type: Flash + MoE Transformer")
        print(f"  Precision: FP16")
        print(f"  Daily Encoder: {pooling_str}")
        print(f"  MoE: {moe_config.num_experts} experts, {moe_config.num_shared_experts} shared, top-{moe_config.top_k}")
        print(f"  Load Balance: {moe_config.load_balance_strategy}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Bucketing: {'Enabled' if use_bucketing else 'Disabled'}")

    # Compute code frequencies if not provided
    if code_frequencies is None:
        code_frequencies = compute_code_frequencies(train_data, config, device)
    
    
    # ============================================================
    # TRAINING SETUP
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    print(f"\nTraining for {epochs} epochs...")
    epoch_results = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config,
            device=device,
            use_mixed_precision=use_mixed_precision,
            moe_config=moe_config,
            epoch=epoch,
            use_bucketing=use_bucketing
        )
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            val_data=val_data,
            criterion=criterion,
            config=config,
            device=device,
            use_mixed_precision=use_mixed_precision
        )
        
        # Combine metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            **train_metrics,
            **val_metrics
        }
        epoch_results.append(epoch_metrics)
        
        # Log
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Top-10 Acc: {val_metrics['top_10_acc']:.3f}")
    
    total_time = time.time() - start_time
    
    evaluation = comprehensive_evaluation(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        device=device,
        training_time_sec=total_time,
        epoch_history=epoch_history,
        code_frequencies=code_frequencies,
        moe_config=moe_config,
        use_mixed_precision=use_mixed_precision
    )
    
    # ============================================================
    # FINAL RESULTS
    # ============================================================
    final_metrics = epoch_results[-1]
    
    results = {
        'experiment': exp_name,
        'parameters': total_params,
        'use_learned_pooling': use_learnt_att_pool,
        'use_bucketing': use_bucketing,
        'final_train_loss': final_metrics['train_loss'],
        'final_val_loss': final_metrics['val_loss'],
        'final_top_10_acc': final_metrics['top_10_acc'],
        'final_top_5_acc': final_metrics['top_5_acc'],
        'training_time_sec': total_time,
        'recall@10': evaluation['performance']['recall@10'],
        'tail_top10_acc': evaluation['performance']['tail_top10_acc'],
        'cost_usd': evaluation['resources']['cost_usd'],
        'peak_memory_gb': evaluation['resources']['total_peak_gb'],
        'full_evaluation': evaluation,
        'all_epochs': epoch_results
    }
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print(f"{'='*80}")
    print(f"  Final Top-10 Acc: {final_metrics['top_10_acc']:.3f}")
    print(f"  Final Val Loss: {final_metrics['val_loss']:.4f}")
    print(f"  Training Time: {total_time:.1f}s")
    
    return results


def run_selected_experiments(
    experiment_names: List[str],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 10
) -> pd.DataFrame:
    """
    Run SELECTED experiments (flexible subset).
    
    Args:
        experiment_names: List of experiment names to run
        train_data: Training DataFrame
        val_data: Validation DataFrame
        device: Torch device
        epochs: Number of epochs per experiment
    
    Returns:
        DataFrame with comparison of all selected experiments
    
    Example usage:
        # Run only baseline and one Flash variant
        results = run_selected_experiments(
            ['exp1_dense_baseline', 'exp2b_flash_learned_pool'],
            df_train, df_val, device
        )
        
        # Run only MoE variants
        results = run_selected_experiments(
            ['exp3_standard_moe', 'exp4_shared_expert'],
            df_train, df_val, device
        )
    """
    print("\n" + "="*80)
    print(f"RUNNING {len(experiment_names)} SELECTED EXPERIMENTS")
    print("="*80)
    
    # Get all available configurations
    all_configs = get_experiment_configs()
    
    # Validate experiment names
    for exp_name in experiment_names:
        if exp_name not in all_configs:
            raise ValueError(f"Unknown experiment: {exp_name}. Available: {list(all_configs.keys())}")
    
    # Run selected experiments
    all_results = []
    
    for exp_name in experiment_names:
        # Clean GPU before each experiment
        if device.type == 'cuda':
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        
        # Get configuration
        moe_config, use_learnt_att_pool = all_configs[exp_name]
        
        # Run experiment
        results = run_single_experiment(
            exp_name=exp_name,
            moe_config=moe_config,
            use_learnt_att_pool=use_learnt_att_pool,
            train_data=train_data,
            val_data=val_data,
            device=device,
            epochs=epochs
        )
        
        all_results.append(results)
    
    # ============================================================
    # CREATE COMPARISON TABLE
    # ============================================================
    df_results = pd.DataFrame(all_results).set_index('experiment')
    
    return df_results


def run_all_experiments(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 10
) -> pd.DataFrame:
    """
    Run ALL experiments (convenience wrapper).
    
    This runs all 7 experiments defined in get_experiment_configs().
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        device: Torch device
        epochs: Number of epochs per experiment
    
    Returns:
        DataFrame with all experiment results
    """
    # Get all experiment names
    all_configs = get_experiment_configs()
    experiment_names = list(all_configs.keys())
    
    print(f"\nRunning ALL {len(experiment_names)} experiments:")
    for i, name in enumerate(experiment_names, 1):
        _, use_pool = all_configs[name]
        pool_str = " (with learned pooling)" if use_pool else ""
        print(f"  {i}. {name}{pool_str}")
    
    # Run all experiments
    return run_selected_experiments(
        experiment_names=experiment_names,
        train_data=train_data,
        val_data=val_data,
        device=device,
        epochs=epochs
    )


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
        
        


# ### Time and cost estimation

# In[1]:


import numpy as np
from typing import Dict, Tuple

def estimate_training_time_cost(
    num_members: int,
    epochs: int,
    model_type: str = "baseline",  # "baseline" or "flash_moe"
    hardware: str = "T4x4"  # "T4x4" or "A100x4"
) -> Dict[str, float]:
    """
    Estimate training time and cost for clinical transformer.
    
    Args:
        num_members: Number of patient records to train on
        epochs: Number of training epochs
        model_type: "baseline" or "flash_moe"
        hardware: "T4x4" (4× T4) or "A100x4" (4× A100)
    
    Returns:
        Dict with keys: 'hours', 'cost_usd', 'days', 'samples_per_sec'
    """
    
    # Base FLOPs calculation (from methodology)
    # 257 ExaFLOPs for 12M members, 10 epochs
    base_flops = 257e18  # ExaFLOPs
    flops_per_member_epoch = base_flops / (12e6 * 10)
    total_flops = flops_per_member_epoch * num_members * epochs
    
    # Hardware specifications
    hardware_specs = {
        'T4x4': {
            'peak_tflops': 260,  # 4 × 65 TFLOPs
            'memory_gb': 64,     # 4 × 16 GB
            'hourly_cost': 3.472,  # $2.072 + 4×$0.35
            'nvlink': False
        },
        'A100x4': {
            'peak_tflops': 1248,  # 4 × 312 TFLOPs
            'memory_gb': 160,     # 4 × 40 GB
            'hourly_cost': 11.992,  # $2.072 + 4×$2.48
            'nvlink': True
        }
    }
    
    # Model configurations with practical adjustments
    model_configs = {
        'baseline': {
            'T4x4': {
                'batch_size': 64,
                'mfu': 0.09,  # 9% - verified from PaLM paper
                'grad_accum_steps': 2,
                'data_overhead': 0.25,
                'comm_overhead': 0.12,
                'misc_overhead': 0.03
            },
            'A100x4': {
                'batch_size': 256,
                'mfu': 0.20,  # 20% - better hardware utilization
                'grad_accum_steps': 1,
                'data_overhead': 0.20,
                'comm_overhead': 0.05,  # NVLink benefit
                'misc_overhead': 0.03
            }
        },
        'flash_moe': {
            'T4x4': {
                'batch_size': 128,
                'mfu': 0.16,  # Practical speedup factors applied
                'grad_accum_steps': 1,
                'data_overhead': 0.20,  # Bucketing optimization
                'comm_overhead': 0.10,
                'misc_overhead': 0.03
            },
            'A100x4': {
                'batch_size': 512,
                'mfu': 0.40,  # Near-optimal for Flash+MoE
                'grad_accum_steps': 1,
                'data_overhead': 0.15,
                'comm_overhead': 0.03,
                'misc_overhead': 0.02
            }
        }
    }
    
    # Get configuration
    config = model_configs[model_type][hardware]
    hw_spec = hardware_specs[hardware]
    
    # Calculate effective throughput
    effective_tflops_per_sec = hw_spec['peak_tflops'] * 1e12 * config['mfu']
    
    # Pure compute time
    compute_seconds = total_flops / effective_tflops_per_sec
    compute_hours = compute_seconds / 3600
    
    # Add overheads (linear, not multiplicative)
    grad_accum_overhead = compute_hours * 0.05 * (config['grad_accum_steps'] - 1)
    data_overhead = compute_hours * config['data_overhead']
    comm_overhead = compute_hours * config['comm_overhead']
    misc_overhead = compute_hours * config['misc_overhead']
    
    total_hours = compute_hours + grad_accum_overhead + data_overhead + comm_overhead + misc_overhead
    
    # Calculate cost
    total_cost = total_hours * hw_spec['hourly_cost']
    
    # Calculate throughput metrics
    samples_per_sec = (num_members * epochs) / (total_hours * 3600)
    
    return {
        'hours': round(total_hours, 1),
        'cost_usd': round(total_cost, 2),
        'days': round(total_hours / 24, 1),
        'samples_per_sec': round(samples_per_sec, 2),
        'compute_hours': round(compute_hours, 1),
        'overhead_hours': round(total_hours - compute_hours, 1),
        'mfu_percent': round(config['mfu'] * 100, 1),
        'batch_size': config['batch_size']
    }
import numpy as np
from typing import Dict

def h100_time_cost(
    num_members: int,
    epochs: int,
    model: str = "baseline",  # "baseline" or "flash_moe"
    num_h100: int = 1
) -> Dict[str, float]:
    """
    Estimate training time & cost on NVIDIA H100 80-GB GPUs.
    """
    # 1. FLOPs scaling factor
    base_flops = 257e18         # 257 EFLOPs for 12 M × 10
    flop_per_ex = base_flops / (12_000_000 * 10)
    total_flops = flop_per_ex * num_members * epochs

    # 2. Hardware
    peak_tf = 395 * num_h100        # TFLOPs/s cluster
    hourly_rate = 18.191 * num_h100 # USD / h

    # 3. Model-specific config
    conf = {
        ("baseline", 1): dict(mfu=0.25, data=0.20, comm=0.00, grad=0.05, misc=0.03),
        ("baseline", "multi"): dict(mfu=0.25, data=0.20, comm=0.03, grad=0.05, misc=0.03),
        ("flash_moe", 1): dict(mfu=0.50, data=0.15, comm=0.00, grad=0.00, misc=0.02),
        ("flash_moe", "multi"): dict(mfu=0.50, data=0.15, comm=0.03, grad=0.00, misc=0.02),
    }
    key = (model, 1 if num_h100 == 1 else "multi")
    cfg = conf[key]

    # 4. Pure compute time
    compute_h = total_flops / (peak_tf * 1e12 * cfg["mfu"]) / 3600

    # 5. Overheads
    overhead_h = compute_h * (
        cfg["data"] + cfg["comm"] + cfg["misc"]
        + cfg["grad"]    # grad_accum=2 only for baseline small batches -> already accounted
    )
    total_h = compute_h + overhead_h
    cost = total_h * hourly_rate

    return {
        "hours": round(total_h, 2),
        "days": round(total_h / 24, 2),
        "cost_usd": round(cost, 0),
        "compute_only_h": round(compute_h, 2),
        "mfu_percent": round(cfg["mfu"] * 100, 1)
    }


# In[4]:


for n in [1, 2, 4]:
    print("Baseline", n, "GPU", h100_time_cost(12_000_000, 1, "flash_moe", n))
    print("Flash+MoE", n, "GPU:", h100_time_cost(12_000_000, 1, "flash_moe", n))


# In[7]:


scenarios = [
    ("Baseline", "baseline", "T4x4"),
    ("Flash+MoE", "flash_moe", "T4x4"),
    ("Baseline", "baseline", "A100x4"),
    ("Flash+MoE", "flash_moe", "A100x4"),
]

for name, model, hw in scenarios:
    result = estimate_training_time_cost(
        num_members=1_000_000,
        epochs=1,
        model_type=model,
        hardware=hw
    )

    print(f"\n{name} on {hw}:")
    print(f"  Time: {result['hours']} hours ({result['days']} days)")
    print(f"  Cost: ${result['cost_usd']}")
    print(f"  Batch Size: {result['batch_size']}")
    print(f"  MFU: {result['mfu_percent']}%")
    print(f"  Throughput: {result['samples_per_sec']} samples/sec")

# Extended example: Different scales
print("\n\nScaling Analysis (Flash+MoE on A100x4)")
print("=" * 50)

for members in [100_000, 500_000, 1_000_000, 5_000_000, 12_000_000]:
    for epochs in [1, 10]:
        result = estimate_training_time_cost(
            num_members=members,
            epochs=epochs,
            model_type="flash_moe",
            hardware="A100x4"
        )
        print(f"{members/1e6:.1f}M members, {epochs} epochs: "
              f"{result['hours']:.1f}h (${result['cost_usd']})")


# ### Training

# In[16]:


import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
client = bigquery.Client()
credentials, project= google.auth.default()
print('credentials:', credentials, ', project:', project)


# In[15]:


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


# ### Start experimentation

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




