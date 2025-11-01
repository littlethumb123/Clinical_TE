#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Flash Attention Baseline Implementation for Clinical Transformer
================================================================

This module implements a complete Flash Attention baseline with modern optimizations:
- PyTorch 2.0 native Flash Attention (scaled_dot_product_attention)
- Pre-normalization (GPT-2+ style)
- Rotary Position Embeddings (RoPE) for better temporal modeling
- SwiGLU activation for improved FFN expressivity
- Mixed precision training (BF16)
- Comprehensive benchmarking utilities

Module Structure:
-----------------
1. Position Encodings (RoPE)
2. Activation Functions (SwiGLU)
3. Core Attention Layers (Flash-enabled)
4. Full Transformer Model
5. Training Utilities
6. Benchmarking & Validation

Author: Clinical Transformer Team
Date: 2025-01-25
References:
- Flash Attention: Dao et al. 2022 (https://arxiv.org/abs/2205.14135)
- Flash Attention 2: Dao 2023 (https://arxiv.org/abs/2307.08691)
- RoPE: Su et al. 2021 (https://arxiv.org/abs/2104.09864)
- SwiGLU: Shazeer 2020 (https://arxiv.org/abs/2002.05202)
"""


# In[44]:


get_ipython().system('pip install xformers')


# In[1]:


# Test with T4-compatible settings
import torch
from xformers.ops import memory_efficient_attention, LowerTriangularMask

device = torch.device('cuda')

# FP16 + head_dim=32
q = torch.randn(2, 200, 8, 32, device=device, dtype=torch.float32)
k = torch.randn(2, 200, 8, 32, device=device, dtype=torch.float32)
v = torch.randn(2, 200, 8, 32, device=device, dtype=torch.float32)

try:
    output = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())
    print(f"✓ xFormers working on T4")
    print(f"  Config: FP32, nhead=8, head_dim=32")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")


# * Consider using FP16 or FP32? 
#     - T4 GPU Specifications:
#     - FP32 performance: 8.1 TFLOPS
#     - FP16 Tensor Core performance: 65 TFLOPS (8× faster!)
#     - BF16: Not well supported on T4
# * Why using xformers instead of pytorch built-in or flash attention 

# In[ ]:





# ### Flash attention implementation

# In[2]:


import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

@dataclass
class FlashAttentionConfig:
    """
    Configuration for Flash Attention transformer.
    
    Attributes:
        # Model architecture
        embedding_size: Dimension of token embeddings (256 for clinical transformer)
        nhead: Number of attention heads (16 for temporal encoder)
        nlayers: Number of transformer layers (6 for temporal encoder)
        nhid: Hidden dimension of FFN (512 standard)
        dropout: Dropout rate (0.1 standard)
        
        # Vocabulary sizes
        cd_cnt: Number of medical codes (84,010)
        target_cd_cnt: Number of prediction targets (8849 test_max_value, not formal retraining target size)
        gender_vocab: Gender categories (4)
        age_vocab: Age in months vocabulary (1,440)
        
        # Sequence dimensions
        len_dy: Number of days in sequence (200)
        len_cd: Number of codes per day (80)
        
        # Optimization flags
        use_rope: Whether to use Rotary Position Embeddings
        use_swiglu: Whether to use SwiGLU activation
        use_prenorm: Whether to use pre-normalization (recommended)
        use_flash: Whether to enable Flash Attention (requires PyTorch 2.0+)
        
        # Training configuration
        dtype: Training data type (torch.bfloat16 recommended)
    """
    # Model architecture (from min_transformer.py)
    embedding_size: int = 256
    nhead: int = 8 # Enable head_dim = 32
    nlayers: int = 6
    nhid: int = 512
    dropout: float = 0.1
    
    # Vocabulary sizes
    cd_cnt: int = 84010
    target_cd_cnt: int = 8849 # test_max_value, not formal retraining target size
    gender_vocab: int = 4
    age_vocab: int = 1440
    
    # Sequence dimensions
    len_dy: int = 200  # Days
    len_cd: int = 80   # Codes per day
    
    # Optimization flags
    use_rope: bool = True
    use_swiglu: bool = True
    use_prenorm: bool = True
    use_flash: bool = True
    
    # Training
    dtype: torch.dtype = torch.float16


# ============================================================================
# SECTION 2: ROTARY POSITION EMBEDDINGS (RoPE)
# ============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for temporal modeling.
    
    RoPE encodes position information by rotating query and key vectors in the
    complex plane. This provides several advantages over learned position embeddings:
    1. Naturally captures relative distances between positions
    2. Enables length extrapolation (can handle sequences longer than training)
    3. No additional parameters to learn
    
    Mathematical Formulation:
    -------------------------
    For position m and dimension i:
    θ_i = base^(-2i/d)
    
    RoPE(x, m) = [
        x_0 * cos(m*θ_0) - x_1 * sin(m*θ_0),
        x_0 * sin(m*θ_0) + x_1 * cos(m*θ_0),
        x_2 * cos(m*θ_1) - x_3 * sin(m*θ_1),
        ...
    ]
    
    This rotation preserves dot products as a function of relative distance:
    q_m^T k_n depends only on (m-n), not absolute positions.
    
    Args:
        dim: Dimension per attention head (embedding_size // nhead)
        max_seq_len: Maximum sequence length (200 for clinical data)
        base: Base for frequency computation (10000 standard, from original paper)
    
    Reference:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        Su et al. 2021 (https://arxiv.org/abs/2104.09864)
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies: θ_i = base^(-2i/d)
        # Shape: [dim // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute rotation matrices for all positions
        # This is more efficient than computing on-the-fly
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        """
        Precompute cos and sin values for all positions up to max_seq_len.
        
        This caching strategy is critical for performance:
        - Computed once during initialization
        - Reused for all forward passes
        - Stored as buffers (automatically moved to correct device)
        """
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        
        # Outer product: [max_seq_len, dim//2]
        # freqs[i, j] = i * inv_freq[j]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Duplicate for complex representation
        # [max_seq_len, dim//2] -> [max_seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Precompute cos and sin
        # Shape: [1, 1, max_seq_len, dim]
        # Dimensions: [batch=1, nhead=1, seq_len, head_dim]
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dimensions of the input.
        
        This implements the rotation in complex plane:
        [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
        
        Args:
            x: Input tensor [..., dim]
        
        Returns:
            Rotated tensor [..., dim]
        """
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to queries and keys.
        
        The rotation is applied independently to Q and K, which means their
        dot products will encode relative position information.
        
        Args:
            q: Query tensor [batch, nhead, seq_len, head_dim]
            k: Key tensor [batch, nhead, seq_len, head_dim]
        
        Returns:
            q_rot: Rotated queries [batch, nhead, seq_len, head_dim]
            k_rot: Rotated keys [batch, nhead, seq_len, head_dim]
        """
        seq_len = q.shape[2]
        
        # Get cached cos/sin values for this sequence length
        # [:, :seq_len, :, :] handles variable length sequences
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Apply rotation
        # q_rot = q * cos + rotate_half(q) * sin
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot


# ============================================================================
# SECTION 3: SWIGLU ACTIVATION
# ============================================================================

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function.
    
    SwiGLU is a gated activation that has shown improved performance over
    standard GELU/ReLU in large language models (LLaMA, PaLM).
    
    Mathematical Formulation:
    -------------------------
    SwiGLU(x) = Swish(W_1 x) ⊙ (W_2 x)
    where Swish(x) = x * sigmoid(x)
    
    Key Properties:
    1. Gating mechanism allows selective information flow
    2. Non-monotonic (unlike ReLU) - can suppress and amplify
    3. Smooth gradients (unlike ReLU) - better optimization
    
    Implementation Note:
    For parameter equivalence with standard 2-layer FFN:
    - Standard FFN: d_model -> d_ff -> d_model (2 * d_model * d_ff params)
    - SwiGLU: d_model -> d_ff -> d_model with gating (3 * d_model * d_ff_adjusted)
    - Therefore: d_ff_adjusted = (2/3) * d_ff to maintain same parameter count
    
    Args:
        d_model: Input/output dimension (256 for clinical transformer)
        d_ff: Hidden dimension (512 for standard FFN)
        dropout: Dropout rate applied after activation
    
    Reference:
        GLU Variants Improve Transformer
        Shazeer 2020 (https://arxiv.org/abs/2002.05202)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        # Adjust hidden dimension for parameter equivalence
        # Standard FFN: 2 * (d_model * d_ff) parameters
        # SwiGLU: 3 * (d_model * d_ff_adjusted) parameters
        # Solve: 2 * d_ff = 3 * d_ff_adjusted => d_ff_adjusted = (2/3) * d_ff
        d_ff_adjusted = int((2 * d_ff) / 3)
        
        # Gate projection: produces activation values
        self.w_gate = nn.Linear(d_model, d_ff_adjusted, bias=False)
        
        # Up projection: produces values to be gated
        self.w_up = nn.Linear(d_model, d_ff_adjusted, bias=False)
        
        # Down projection: back to model dimension
        self.w_down = nn.Linear(d_ff_adjusted, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: Input tensor [..., d_model]
        
        Returns:
            Output tensor [..., d_model]
        """
        # Swish activation: x * sigmoid(x)
        gate = F.silu(self.w_gate(x))  # SiLU is Swish
        
        # Element-wise gating
        up = self.w_up(x)
        hidden = gate * up
        
        # Project back to model dimension
        output = self.w_down(hidden)
        output = self.dropout(output)
        
        return output



# In[3]:


# ============================================================================
# SECTION 4: FLASH ATTENTION ENCODER LAYER
# ============================================================================

class FlashAttentionEncoderLayer(nn.Module):
    """
    Transformer encoder layer with Flash Attention and modern optimizations.
    
    This layer combines several architectural improvements:
    1. Flash Attention: Memory-efficient O(N) attention via tiling
    2. Pre-normalization: Norm before attention/FFN (better gradient flow)
    3. RoPE: Rotary position embeddings (better temporal modeling)
    4. SwiGLU: Gated activation (improved expressivity)
    
    Architecture:
    -------------
    x_norm = LayerNorm(x)
    x = x + FlashAttention(x_norm) with RoPE
    x_norm = LayerNorm(x)
    x = x + SwiGLU(x_norm)
    
    This is the "pre-norm" configuration used in:
    - GPT-2, GPT-3, GPT-4 (OpenAI)
    - LLaMA, LLaMA 2 (Meta)
    - Mistral (Mistral AI)
    
    Advantages over post-norm (original Transformer):
    - More stable training (better gradient flow)
    - Can train deeper models without warm-up tricks
    - Faster convergence
    
    Args:
        config: FlashAttentionConfig with model parameters
    """
    
    def __init__(self, config: FlashAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.embedding_size
        self.nhead = config.nhead
        self.head_dim = self.d_model // self.nhead
        # Validate for xFormers
        if config.use_flash:
            if config.dtype == torch.bfloat16:
                print("⚠️  Warning: BF16 not supported on T4 with xFormers. Switching to FP16.")
                config.dtype = torch.float16        
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Rotary position embeddings
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=self.head_dim,
                max_seq_len=config.len_dy,  # 200 days
                base=10000.0
            )
        
        # Feed-forward network
        if config.use_swiglu:
            self.ffn = SwiGLU(
                d_model=self.d_model,
                d_ff=config.nhid,
                dropout=config.dropout
            )
        else:
            # Standard 2-layer MLP
            self.ffn = nn.Sequential(
                nn.Linear(self.d_model, config.nhid),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.nhid, self.d_model),
                nn.Dropout(config.dropout)
            )
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Check xFormers availability
        self.xformers_available = False
        if config.use_flash:
            try:
                from xformers.ops import memory_efficient_attention
                self.xformers_attention = memory_efficient_attention
                self.xformers_available = True
                if not hasattr(FlashAttentionEncoderLayer, '_xformers_logged'):
                    print(f"✓ xFormers available (using FP16, head_dim={self.head_dim})")
                    FlashAttentionEncoderLayer._xformers_logged = True
            except ImportError:
                if not hasattr(FlashAttentionEncoderLayer, '_xformers_warned'):
                    print("⚠️  xFormers not available - using PyTorch SDPA fallback")
                    FlashAttentionEncoderLayer._xformers_warned = True
        
        # Initialize weights
        self._init_weights()
    
    
    def _init_weights(self):
        """
        Initialize weights for stable training.
        
        Strategy:
        - QKV projections: Xavier uniform with small gain
        - Output projection: Zero init (residual path starts at identity)
        - FFN: Standard initialization
        
        This initialization helps with:
        - Stable gradient flow in deep networks
        - Faster convergence
        - Avoiding gradient explosion in early training
        """
        # QKV projections with scaled init
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2))
        
        # Output projection starts at zero (residual path)
        nn.init.zeros_(self.out_proj.weight)
        
        # FFN initialization (if not SwiGLU)
        if not self.config.use_swiglu and isinstance(self.ffn, nn.Sequential):
            nn.init.xavier_uniform_(self.ffn[0].weight)
            nn.init.zeros_(self.ffn[3].weight)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with Flash Attention.
        
        Args:
            src: Input tensor [seq_len, batch_size, d_model]
            src_mask: Optional attention mask [seq_len, seq_len]
            is_causal: Whether to apply causal masking (True for temporal encoder)
        
        Returns:
            output: Tensor [seq_len, batch_size, d_model]
        """
        seq_len, batch_size, d_model = src.shape
        
        # ---- Attention Block (Pre-Norm) ----
        
        # Pre-normalization
        x = src
        x_norm = self.norm1(x)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        # Ensure consistent dtype immediately after projection
        if self.config.use_flash:
            target_dtype = self.config.dtype
            q = q.to(dtype=target_dtype)
            k = k.to(dtype=target_dtype)
            v = v.to(dtype=target_dtype)    
        
        # Reshape for multi-head attention
        # [seq_len, batch, d_model] -> [batch, nhead, seq_len, head_dim]
        q = q.view(seq_len, batch_size, self.nhead, self.head_dim)
        q = q.permute(1, 2, 0, 3)  # [batch, nhead, seq_len, head_dim]
        
        k = k.view(seq_len, batch_size, self.nhead, self.head_dim)
        k = k.permute(1, 2, 0, 3)
        
        v = v.view(seq_len, batch_size, self.nhead, self.head_dim)
        v = v.permute(1, 2, 0, 3)
        
        # Apply RoPE if enabled
        if self.config.use_rope:
            q, k = self.rope(q, k)
        
        # Flash Attention using Xform
        if self.config.use_flash and self.xformers_available:
            # Use xFormers memory-efficient attention (compatible with T4)
            attn_output = self._xformers_attention(q, k, v, is_causal)
            
        elif self.config.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch SDPA (auto-selects best backend)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else src_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Fallback to standard attention
            attn_output = self._manual_attention(q, k, v, src_mask, is_causal, seq_len)
        
        # Reshape back to [seq_len, batch, d_model]
        attn_output = attn_output.permute(2, 0, 1, 3)  # [seq_len, batch, nhead, head_dim]
        attn_output = attn_output.contiguous().view(seq_len, batch_size, d_model)
        
        # Output projection and residual
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        src = src + attn_output
        
        # ---- FFN Block (Pre-Norm) ----
        
        x = src
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        src = src + ffn_output
        
        return src

    def _xformers_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool
    ) -> torch.Tensor:
        """
        Apply xFormers memory-efficient attention.
        
        Args:
            q, k, v: [batch, nhead, seq_len, head_dim]
            is_causal: Whether to use causal masking
        
        Returns:
            output: [batch, nhead, seq_len, head_dim]
        """
        batch_size, nhead, seq_len, head_dim = q.shape
        
        # Ensure all tensors have same dtype
        target_dtype = self.config.dtype
        q = q.to(dtype=target_dtype)
        k = k.to(dtype=target_dtype)
        v = v.to(dtype=target_dtype)  
        
        # xFormers expects [batch, seq_len, nhead, head_dim]
        q = q.transpose(1, 2).contiguous()  # [batch, seq_len, nhead, head_dim]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Create attention bias for causal masking
        attn_bias = None
        if is_causal:
            from xformers.ops import LowerTriangularMask
            attn_bias = LowerTriangularMask()
        
        # Apply memory-efficient attention
        attn_output = self.xformers_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=self.dropout.p if self.training else 0.0,
            scale=1.0 / math.sqrt(head_dim)
        )
        
        # Reshape back to [batch, nhead, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        return attn_output
    
    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        is_causal: bool,
        seq_len: int
    ) -> torch.Tensor:
        """
        Fallback manual attention implementation.
        
        Args:
            q, k, v: [batch, nhead, seq_len, head_dim]
            src_mask: Optional mask
            is_causal: Whether to use causal masking
            seq_len: Sequence length
        
        Returns:
            output: [batch, nhead, seq_len, head_dim]
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if is_causal:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device),
                diagonal=1
            ).bool()
            attn.masked_fill_(causal_mask, float('-inf'))
        
        if src_mask is not None:
            attn += src_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout.p, training=self.training)
        attn_output = torch.matmul(attn, v)
        
        return attn_output
    
    
# ============================================================================
# SECTION 5: FLASH ATTENTION TRANSFORMER MODEL
# ============================================================================

class FlashClinicalTransformer(nn.Module):
    """
    Hierarchical clinical transformer with Flash Attention.
    
    This model processes hierarchical medical claims data:
    1. Daily level: Multiple medical codes per day
    2. Temporal level: Sequence of days (up to 200 days)
    
    Architecture:
    -------------
    Input: [batch, 200 days, 82 features]
           where features = [age, gender, 80 medical codes]
    
    Stage 1 - Daily Encoder:
    ├── Embed codes: [batch, 200, 80, 256]
    ├── 1-layer transformer over 80 codes
    ├── Max pool: [batch, 200, 256]
    └── Add demographics: [batch, 200, 256]
    
    Stage 2 - Temporal Encoder (Flash Attention):
    ├── 6 layers of Flash Attention
    ├── Causal masking (can't see future)
    ├── RoPE for temporal position encoding
    └── Output: [batch, 200, 256]
    
    Output:
    └── Project to [batch, 200, 2767] medical code predictions
    
    This architecture is based on BEHRT (Li et al. 2020) with improvements:
    - Flash Attention for efficiency
    - Pre-normalization for stability
    - RoPE for better temporal modeling
    - SwiGLU for expressivity
    
    Args:
        config: FlashAttentionConfig with all model parameters
    """
    
    def __init__(self, config: FlashAttentionConfig):
        super().__init__()
        self.config = config
        
        # ---- Embeddings ----
        
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        
        # ---- Daily Code Encoder (Standard) ----
        # Keep standard transformer for daily encoder (sequences are short: 80 codes)
        # Flash Attention benefit is minimal for N < 100
        if config.use_flash:
            # Use Flash Attention for daily encoder
            daily_config = FlashAttentionConfig(
                embedding_size=config.embedding_size,
                nhead=4,  # 4 heads for daily encoder
                nlayers=1,
                nhid=config.embedding_size,
                dropout=0.0,
                len_dy=config.len_cd,  # Sequence length = 80
                use_flash=True,
                use_rope=False,  # No need for position encoding in daily codes
                use_swiglu=config.use_swiglu,
                use_prenorm=config.use_prenorm,
                dtype=config.dtype
            )

            # Single layer encoder
            self.transformer_encoder_cd = FlashAttentionEncoderLayer(daily_config)
        else:
            # Fallback to standard
            encoder_layer_cd = TransformerEncoderLayer(
                d_model=config.embedding_size,
                nhead=4,
                dim_feedforward=config.embedding_size,
                dropout=0.0,
                batch_first=False
            )
            self.transformer_encoder_cd = TransformerEncoder(encoder_layer_cd, num_layers=1)
        
        # ---- Temporal Encoder (Flash Attention) ----
        # This is the critical path - 200 day sequences benefit greatly from Flash
        
        self.temporal_layers = nn.ModuleList([
            FlashAttentionEncoderLayer(config)
            for _ in range(config.nlayers)
        ])
        
        # ---- Output Layers ----
        
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(config.embedding_size)
        self.dropout = nn.Dropout(0.1)
        self.decoder_cd = nn.Linear(config.embedding_size, config.target_cd_cnt)
        
        # Initialize output projection
        nn.init.zeros_(self.decoder_cd.weight)
        nn.init.uniform_(self.decoder_cd.weight, -0.1, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical transformer.
        
        Args:
            x: Input tensor [batch, 200, 82]
               where 82 = [age, gender, 80 medical codes]
        
        Returns:
            predictions: Log probabilities [batch, 200, target_cd_cnt]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # ---- Stage 1: Extract and Embed Inputs ----
        
        # Extract components
        age_in_months = x[:, :, 0].long()  # [batch, 200]
        gender_cd = x[:, :, 1].long()      # [batch, 200]
        codes = x[:, :, 2:].long()         # [batch, 200, 80]
        
        # Embed
        age_emb = self.embedding_age_in_months(age_in_months)      # [batch, 200, 256]
        gender_emb = self.embedding_gender_cd(gender_cd)           # [batch, 200, 256]
        codes_emb = self.embedding_cd(codes)                       # [batch, 200, 80, 256]
        
        # Residual connection: sum of all code embeddings per day
        codes_sum = codes_emb.sum(dim=-2)  # [batch, 200, 256]
        
        # ---- Stage 2: Daily Code Encoding ----
        
        # Reshape for transformer: [batch*200, 80, 256]
        codes_flat = codes_emb.reshape(batch_size * self.config.len_dy, self.config.len_cd, self.config.embedding_size)
        
        # Swap axes for sequence-first format: [80, batch*200, 256]
        codes_flat = codes_flat.transpose(0, 1)
        
        # Apply daily transformer
        codes_encoded = self.transformer_encoder_cd(codes_flat)  # [80, batch*200, 256]
        
        # Max pool across 80 codes: [batch*200, 256, 80] -> [batch*200, 256]
        codes_encoded = codes_encoded.permute(1, 2, 0)  # [batch*200, 256, 80]
        codes_pooled = F.max_pool1d(codes_encoded, kernel_size=self.config.len_cd)  # [batch*200, 256, 1]
        codes_pooled = codes_pooled.squeeze(-1)  # [batch*200, 256]
        
        # Reshape back: [batch, 200, 256]
        codes_pooled = codes_pooled.view(batch_size, self.config.len_dy, self.config.embedding_size)
        
        # ---- Stage 3: Combine with Demographics ----
        
        # Add all embeddings
        combined = codes_sum + codes_pooled + gender_emb + age_emb  # [batch, 200, 256]
        
        # Activation and normalization
        combined = self.activation(combined)
        combined = self.norm(combined)
        
        # ---- Stage 4: Temporal Encoding (Flash Attention) ----
        
        # Swap to sequence-first: [200, batch, 256]
        temporal_input = combined.transpose(0, 1)
        
        # Apply Flash Attention layers with causal masking
        for layer in self.temporal_layers:
            temporal_input = layer(temporal_input, is_causal=True)
        
        # Swap back to batch-first: [batch, 200, 256]
        temporal_output = temporal_input.transpose(0, 1)
        
        # Final normalization and dropout
        temporal_output = self.norm(temporal_output)
        temporal_output = self.dropout(temporal_output)
        
        # ---- Stage 5: Output Projection ----
        
        # Project to target vocabulary
        predictions = self.decoder_cd(temporal_output)  # [batch, 200, target_cd_cnt]
        
        return predictions


# In[4]:


# ============================================================================
# SECTION 6: TRAINING UTILITIES
# ============================================================================

class MixedPrecisionTrainer:
    """
    Mixed precision training utilities for Flash Attention model.
    
    Mixed precision (BF16/FP16) provides:
    1. 2× speedup from faster FP16 math
    2. 2× memory reduction from smaller tensors
    3. Maintained accuracy with gradient scaling
    
    BF16 vs FP16:
    - BF16: Better for transformers (same exponent range as FP32)
    - FP16: Requires more careful gradient scaling (can overflow)
    - Recommendation: Use BF16 if available (Ampere+ GPUs)
    
    Args:
        model: FlashClinicalTransformer instance
        device: torch.device for training
        dtype: torch.float16 or torch.float16
    """
    
    def __init__(
        self, 
        model: FlashClinicalTransformer,
        device: torch.device,
        dtype: torch.dtype = torch.float16
    ):
        self.model = model.to(device).to(dtype)
        self.device = device
        self.dtype = dtype
        self.scaler = torch.cuda.amp.GradScaler() if dtype == torch.float16 else None
        
        # Enable TF32 for additional speedup on Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable Flash Attention
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
    
    def training_step(
        self,
        batch: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        Single training step with mixed precision.
        
        Args:
            batch: Input tensor [batch, 200, 82]
            targets: Target tensor (depends on task)
            optimizer: PyTorch optimizer
            criterion: Loss function (typically NLLLoss)
        
        Returns:
            loss: Scalar loss value
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(dtype=self.dtype, enabled=True):
            outputs = self.model(batch)
            loss = criterion(outputs, targets)
        
        # Backward pass with gradient scaling (for FP16)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        
        return loss.item()
# ============================================================================
# SECTION 7: BENCHMARKING & VALIDATION
# ============================================================================

class FlashAttentionBenchmark:
    """
    Comprehensive benchmarking suite for Flash Attention.
    
    Measures:
    1. Training throughput (tokens/second)
    2. Memory usage (peak GPU memory)
    3. Inference latency (forward pass time)
    4. Numerical accuracy (vs standard attention)
    
    This validates that Flash Attention provides:
    - Significant speedup (2-3× expected)
    - Memory reduction (30-40% expected)
    - Identical results to standard attention
    """
    
    def __init__(self, device: torch.device = torch.device('cuda')):
        self.device = device
    
    def benchmark_throughput(
        self,
        model: FlashClinicalTransformer,
        batch_size: int = 16,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Measure training throughput.
        
        Returns:
            Dictionary with:
            - tokens_per_second: Throughput in tokens/sec
            - time_per_batch: Average time per batch (seconds)
            - memory_allocated: Peak GPU memory (MB)
        """
        model.eval()
        # Create properly bounded dummy input
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create empty tensor
        dummy_input = torch.zeros((batch_size, config.len_dy, 82), device=device, dtype=torch.long)

        # Fill each feature with appropriate ranges
        dummy_input[:, :, 0] = torch.randint(0, config.age_vocab, (batch_size, config.len_dy), device=device)  # Age: 0-1439
        dummy_input[:, :, 1] = torch.randint(0, config.gender_vocab, (batch_size, config.len_dy), device=device)  # Gender: 0-3
        dummy_input[:, :, 2:] = torch.randint(0, config.cd_cnt, (batch_size, config.len_dy, 80), device=device)  # Codes: 0-84009

        # Convert to float for model input
        dummy_input = dummy_input.float()
        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        time_per_batch = total_time / num_iterations
        total_tokens = batch_size * model.config.len_dy * num_iterations
        tokens_per_second = total_tokens / total_time
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return {
            'tokens_per_second': tokens_per_second,
            'time_per_batch': time_per_batch,
            'memory_mb': memory_mb
        }
    
    def compare_standard_vs_flash(
        self,
        config: FlashAttentionConfig,
        batch_size: int = 16
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare standard attention vs Flash Attention.
        
        Returns:
            Dictionary with results for 'standard' and 'flash'
        """
        results = {}
        
        # Test standard attention
        config_standard = FlashAttentionConfig(**vars(config))
        config_standard.use_flash = False
        model_standard = FlashClinicalTransformer(config_standard).to(self.device)
        
        print("Benchmarking standard attention...")
        results['standard'] = self.benchmark_throughput(model_standard, batch_size)
        
        # Test Flash Attention
        config_flash = FlashAttentionConfig(**vars(config))
        config_flash.use_flash = True
        model_flash = FlashClinicalTransformer(config_flash).to(self.device)
        
        print("Benchmarking Flash Attention...")
        results['flash'] = self.benchmark_throughput(model_flash, batch_size)
        
        # Calculate improvements
        speedup = results['flash']['tokens_per_second'] / results['standard']['tokens_per_second']
        memory_reduction = 1.0 - (results['flash']['memory_mb'] / results['standard']['memory_mb'])
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Standard Attention:")
        print(f"  Throughput: {results['standard']['tokens_per_second']:.1f} tokens/sec")
        print(f"  Memory: {results['standard']['memory_mb']:.1f} MB")
        print(f"\nFlash Attention:")
        print(f"  Throughput: {results['flash']['tokens_per_second']:.1f} tokens/sec")
        print(f"  Memory: {results['flash']['memory_mb']:.1f} MB")
        print(f"\nImprovements:")
        print(f"  Speedup: {speedup:.2f}×")
        print(f"  Memory reduction: {memory_reduction*100:.1f}%")
        print("="*60)
        
        return results
    
    def validate_numerical_accuracy(
        self,
        config: FlashAttentionConfig,
        batch_size: int = 4,
        tolerance: float = 1e-3
    ) -> bool:
        """
        Validate that Flash Attention produces same results as standard.
        
        Args:
            config: Model configuration
            batch_size: Batch size for testing
            tolerance: Maximum allowed difference
        
        Returns:
            True if outputs match within tolerance
        """
        print("Validating numerical accuracy...")
        
        # Create models
        config_standard = FlashAttentionConfig(**vars(config))
        config_standard.use_flash = False
        model_standard = FlashClinicalTransformer(config_standard).to(self.device).eval()
        
        config_flash = FlashAttentionConfig(**vars(config))
        config_flash.use_flash = True
        model_flash = FlashClinicalTransformer(config_flash).to(self.device).eval()
        
        # Copy weights from standard to flash
        model_flash.load_state_dict(model_standard.state_dict())
        
        # Create properly bounded dummy input
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create empty tensor
        dummy_input = torch.zeros((batch_size, config.len_dy, 82), device=device, dtype=torch.long)

        # Fill each feature with appropriate ranges
        dummy_input[:, :, 0] = torch.randint(0, config.age_vocab, (batch_size, config.len_dy), device=device)  # Age: 0-1439
        dummy_input[:, :, 1] = torch.randint(0, config.gender_vocab, (batch_size, config.len_dy), device=device)  # Gender: 0-3
        dummy_input[:, :, 2:] = torch.randint(0, config.cd_cnt, (batch_size, config.len_dy, 80), device=device)  # Codes: 0-84009

        # Convert to float for model input
        dummy_input = dummy_input.float()
        
        # Forward pass
        with torch.no_grad():
            output_standard = model_standard(dummy_input)
            output_flash = model_flash(dummy_input)
        
        # Compare
        max_diff = (output_standard - output_flash).abs().max().item()
        mean_diff = (output_standard - output_flash).abs().mean().item()
        
        passed = max_diff < tolerance
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Tolerance: {tolerance:.6f}")
        print(f"Validation: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        return passed


# ### Validation and test flash attention (with Xformers) with dummy dataset

# In[5]:


config = FlashAttentionConfig(
    use_flash=True,
    use_rope=True,
    use_swiglu=True,
    use_prenorm=True
)

model = FlashClinicalTransformer(config)


# In[6]:


# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created successfully")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / (1024**2):.1f} MB (FP32)")


# In[7]:


# Test forward pass
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# Create properly bounded dummy input
batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create empty tensor
dummy_input = torch.zeros((batch_size, config.len_dy, 82), device=device, dtype=torch.long)

# Fill each feature with appropriate ranges
dummy_input[:, :, 0] = torch.randint(0, config.age_vocab, (batch_size, config.len_dy), device=device)  # Age: 0-1439
dummy_input[:, :, 1] = torch.randint(0, config.gender_vocab, (batch_size, config.len_dy), device=device)  # Gender: 0-3
dummy_input[:, :, 2:] = torch.randint(0, config.cd_cnt, (batch_size, config.len_dy, 80), device=device)  # Codes: 0-84009

# Convert to float for model input
dummy_input = dummy_input.float()

print(f"\nTesting forward pass...")
print(f"Input shape: {dummy_input.shape}")

with torch.no_grad():
    output = model(dummy_input)

print(f"Output shape: {output.shape}")
print(f"Expected shape: [{batch_size}, {config.len_dy}, {config.target_cd_cnt}]")
print(f"Forward pass: ✓")

print("="*60)


# ### Run comparison with dummy variables

# In[27]:


# Test standard attention
config = FlashAttentionConfig(
    use_flash=True,
    use_rope=True,
    use_swiglu=True,
    use_prenorm=True
)
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_standard = FlashAttentionConfig(**vars(config))
config_standard.use_flash = False
model_standard = FlashClinicalTransformer(config_standard).to(device)
config_flash = FlashAttentionConfig(**vars(config))
config_flash.use_flash = True
model_flash = FlashClinicalTransformer(config_flash).to(device)


# In[28]:


fa_benchmark = FlashAttentionBenchmark(device)
standard_benchmark_througput = fa_benchmark.benchmark_throughput(model_standard, batch_size)
flash_benchmark_througput = fa_benchmark.benchmark_throughput(model_flash, batch_size)


# In[29]:


standard_benchmark_througput


# In[30]:


flash_benchmark_througput


# ### Real dataset

# In[4]:


import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
import pandas as pd
client = bigquery.Client()
credentials, project= google.auth.default()
print('credentials:', credentials, ', project:', project)


# In[5]:


"""
Training Script for Flash Attention Baseline with Real Data

Integrates Flash Attention model with actual clinical claims data.
Handles data preparation, training loop, and validation.

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

# Import Flash Attention baseline
# from fa_baseline import (
#     FlashClinicalTransformer,
#     FlashAttentionConfig,
#     FlashAttentionBenchmark,
#     verify_environment
# )


# =============================================================================
# DATA PREPARATION FOR CLINICAL CLAIMS
# =============================================================================

class ClinicalDataPreparator:
    """
    Prepares clinical claims data for Flash Attention model.
    """
    
    def __init__(self, len_dy: int = 200, len_cd: int = 80, target_cd_cnt: int = 8849):
        self.len_dy = len_dy
        self.len_cd = len_cd
        self.target_cd_cnt = target_cd_cnt
    
    def parse_codes(self, cd_str: str) -> List[List[int]]:
        """Parse medical codes from string format."""
        if pd.isna(cd_str) or cd_str == '':
            return [[0] * self.len_cd for _ in range(self.len_dy)]
        
        # Split by asterisk for multi-day sequences
        days = cd_str.split('*')
        days = days[:self.len_dy]
        
        # Parse each day's codes
        parsed_days = []
        for day_str in days:
            codes = day_str.split(',')
            codes = [int(c) if c.strip() != '' else 0 for c in codes]
            codes = (codes + [0] * self.len_cd)[:self.len_cd]
            parsed_days.append(codes)
        
        # Pad to len_dy days
        while len(parsed_days) < self.len_dy:
            parsed_days.append([0] * self.len_cd)
        
        return parsed_days
    
    def parse_age_gender(self, value, num_days: int = None) -> List[int]:
        """Parse age or gender values."""
        if num_days is None:
            num_days = self.len_dy
        
        if isinstance(value, str):
            values = value.split('*')
            # Clip to max age (1439)
            values = [min(int(v), 1439) if v.strip() != '' else 0 for v in values]
        else:
            values = [min(int(value), 1439)] * num_days
        
        values = values[:self.len_dy]
        while len(values) < self.len_dy:
            values.append(values[-1] if values else 0)
        
        return values

    def parse_target(self, target_str: str) -> List[List[int]]:
        """
        Parse target string to nested list of codes (multi-label).
        Matches conv_target() from min_transformer_finetune.py lines 141-147.
        """
        if pd.isna(target_str) or target_str == '':
            return [[0] for _ in range(self.len_dy)]

        # Split by asterisk for multi-day sequences
        days = target_str.split('*')
        days = days[:self.len_dy]

        # Parse each day's target codes
        parsed_days = []
        for day_str in days:
            codes = day_str.split(',')
            codes = [min(int(c), self.target_cd_cnt-1) if c.strip() != '' else 0 for c in codes]
            parsed_days.append(codes)  # ✅ Keep ALL codes per day

        # Pad to len_dy days
        while len(parsed_days) < self.len_dy:
            parsed_days.append([0])

        return parsed_days  # Returns [[15,42,7258], [156], ...]
    
    
    def prepare_batch(
        self, 
        batch_df: pd.DataFrame,
        device: torch.device
    ) -> Tuple[List[int], torch.Tensor, List[List[int]]]:
        """
        Prepare batch for Flash Attention model..
        """
        batch_size = len(batch_df)

        age_batch = []
        gender_batch = []
        codes_batch = []
        dt_cnt = []
        targets = []

        for idx, row in batch_df.iterrows():
            day_count = int(row.get('dt_cnt', 1))
            dt_cnt.append(day_count)

            age_seq = self.parse_age_gender(row['age_in_months'], day_count)
            age_batch.append(age_seq)

            gender_seq = self.parse_age_gender(row['gender_cd'], day_count)
            gender_batch.append(gender_seq)

            codes_seq = self.parse_codes(row['cd'])
            codes_batch.append(codes_seq)

            # ✅ Parse target_cd column directly (multi-label)
            target_seq = self.parse_target(row['target_cd'])
            targets.append(target_seq)  # ✅ Returns [[15,42,7258], [156], ...]

        # Convert to tensors
        age_tensor = torch.tensor(age_batch, dtype=torch.long, device=device)
        gender_tensor = torch.tensor(gender_batch, dtype=torch.long, device=device)
        codes_tensor = torch.tensor(codes_batch, dtype=torch.long, device=device)

        age_tensor = age_tensor.unsqueeze(-1)
        gender_tensor = gender_tensor.unsqueeze(-1)

        x = torch.cat([age_tensor, gender_tensor, codes_tensor], dim=-1)
        x = x.float()

        return dt_cnt, x, targets  # ✅ targets is List[List[List[int]]]


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: FlashClinicalTransformer,
    data: pd.DataFrame,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    preparator: ClinicalDataPreparator,
    batch_size: int,
    device: torch.device,
    config: FlashAttentionConfig,
    epoch: int = 0
) -> dict:
    """
    Train for one epoch with Flash Attention model.
    
    Args:
        model: Flash Attention model
        data: Training DataFrame
        optimizer: Optimizer
        criterion: Loss function (BCEWithLogitsLoss)
        preparator: Data preparator
        batch_size: Batch size
        device: Device
        config: Model configuration
        epoch: Current epoch number
    
    Returns:
        Dictionary with training statistics
    """
    model.train()
    
    num_batches = len(data) // batch_size
    total_loss = 0.0
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.dtype == torch.float16 else None
    
    start_time = time.time()
    
    for i in range(num_batches):
        if i % 100 == 0:
            print(f'Epoch {epoch}, Batch {i}/{num_batches}, Time: {time.time()-start_time:.2f}s')
        
        # Get batch
        batch_df = data.iloc[i*batch_size:(i+1)*batch_size]
        
        # Prepare tensors
        dt_cnt, x, y = preparator.prepare_batch(batch_df, device)
        # y is now List[List[List[int]]], e.g., [[[15,42], [156], ...], [[22,15], ...]]
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=config.dtype):
            output = model(x)  # [batch, 200, target_cd_cnt]

            # Reshape for loss computation
            output = output.reshape(batch_size * config.len_dy, config.target_cd_cnt)

            # Flatten targets (lines 182-183 in min_transformer_finetune.py)
            y_flat = [item for sublist in y for item in sublist]
            # y_flat is now [[15,42], [156], [22,15], ...] - list of code lists per day

            # Select only valid days (lines 184 in min_transformer_finetune.py)
            valid_outputs = []
            for j in range(batch_size):
                start_idx = config.len_dy * j
                end_idx = start_idx + dt_cnt[j]
                valid_outputs.append(output[start_idx:end_idx])

            output = torch.cat(valid_outputs, dim=0)  # [total_valid_days, target_cd_cnt]

            # Create multi-hot encoding (lines 186-191 in min_transformer_finetune.py)
            y_cd = torch.zeros(len(output), config.target_cd_cnt, device=device)

            for j in range(len(output)):  # For each valid day
                for k in y_flat[j]:  # For each target code on this day
                    if k != 0:
                        if k < config.target_cd_cnt:
                            y_cd[j, k] = 1
                        else:
                            # Warnings for out-of-bounds targets
                            print(f"Warning: Target code {k} >= {config.target_cd_cnt}, skipping")

            # BCEWithLogitsLoss with multi-hot targets
            loss = criterion(output, y_cd)
        
        # Backward with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Memory cleanup
        del x, output, loss, y_cd, dt_cnt, y
        if i % 100 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    
    print(f"\nEpoch {epoch} completed: Avg Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
    
    return {
        'avg_loss': avg_loss,
        'epoch_time': epoch_time,
        'num_batches': num_batches
    }


def validate_epoch(
    model: FlashClinicalTransformer,
    data: pd.DataFrame,
    criterion: nn.Module,
    preparator: ClinicalDataPreparator,
    batch_size: int,
    device: torch.device,
    config: FlashAttentionConfig
) -> dict:
    """Validation loop."""
    model.eval()
    
    num_batches = len(data) // batch_size
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_df = data.iloc[i*batch_size:(i+1)*batch_size]
            dt_cnt, x, y = preparator.prepare_batch(batch_df, device)
            
            output = model(x)
            output = output.reshape(batch_size * config.len_dy, config.target_cd_cnt)
            
            y_flat = [item for sublist in y for item in sublist]
            
            valid_outputs = []
            for j in range(batch_size):
                start_idx = config.len_dy * j
                end_idx = start_idx + dt_cnt[j]
                valid_outputs.append(output[start_idx:end_idx])

            output = torch.cat(valid_outputs, dim=0)

            # ✅ Create multi-hot encoding (same as training)
            y_cd = torch.zeros(len(output), config.target_cd_cnt, device=device)

            for j in range(len(output)):
                for k in y_flat[j]:
                    if k != 0 and k < config.target_cd_cnt:
                        y_cd[j, k] = 1

            loss = criterion(output, y_cd)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")
    
    return {'val_loss': avg_loss}


# #### Load data

# In[6]:


import logging

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# In[6]:


# Load data
input_sql = """
select * from
anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_o3_score_ending
limit 2000
"""
input_data = client.query(input_sql).to_dataframe() 


# In[7]:


input_data.head()


# In[7]:


import pandas as pd
df_train = pd.read_feather("sample_data/mdcd_train_8000.feather")
df_val = pd.read_feather("sample_data/mdcd_val_2000.feather")


# In[12]:


df_train.head()


# In[8]:


import torch
import gc

def cleanup_gpu_memory():
    """
    Comprehensive GPU memory cleanup before training.
    
    Clears:
    - Python garbage collector
    - PyTorch CUDA cache
    - All cached allocations
    - Resets memory statistics
    """
    # Collect Python garbage
    gc.collect()
    
    if torch.cuda.is_available():
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Synchronize all CUDA operations
        torch.cuda.synchronize()
        
        # Reset peak memory stats (optional, useful for profiling)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Print current memory status
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved_memory = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {current_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB")


# Add this at the very beginning of your training script (before model creation):



# In[14]:


cleanup_gpu_memory()


# ### Test availability of flash attention implementations

# In[7]:


# Pytorch built-in flash attention is not available for T4 for pytorch 028
# Then turn to xformers


import torch
import torch.nn.functional as F

print("="*70)
print("FLASH ATTENTION COMPATIBILITY DIAGNOSTIC")
print("="*70)

# 1. PyTorch Version
print(f"\n1. PyTorch Version: {torch.__version__}")
pytorch_ok = torch.__version__ >= "2.0.0"
print(f"   Required: >= 2.0.0")
print(f"   Status: {'✓ PASS' if pytorch_ok else '✗ FAIL'}")

# 2. CUDA Availability
cuda_available = torch.cuda.is_available()
print(f"\n2. CUDA Available: {cuda_available}")
print(f"   Status: {'✓ PASS' if cuda_available else '✗ FAIL'}")

if cuda_available:
    # 3. CUDA Version
    print(f"\n3. CUDA Version: {torch.version.cuda}")
    
    # 4. GPU Model
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n4. GPU: {gpu_name}")
    
    # 5. Compute Capability
    major, minor = torch.cuda.get_device_capability(0)
    compute_cap = major + minor / 10
    print(f"\n5. CUDA Compute Capability: {compute_cap:.1f}")
    print(f"   Required: >= 7.5 (Volta/Turing/Ampere/Hopper)")
    compute_ok = compute_cap >= 7.5
    print(f"   Status: {'✓ PASS' if compute_ok else '✗ FAIL - TOO OLD!'}")
    
    # 6. Check if Flash Attention is compiled
    print(f"\n6. Checking PyTorch Flash Attention support...")
    print(f"   torch.backends.cuda.flash_sdp_enabled(): {torch.backends.cuda.flash_sdp_enabled()}")
    
    # 7. Test all backends
    print(f"\n7. Testing SDPA backends:")
    from torch.backends.cuda import sdp_kernel
    
    dummy_q = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
    dummy_k = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
    dummy_v = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
    
    backends = {
        'Flash Attention': (True, False, False),
        'Memory Efficient': (False, False, True),
        'Math (fallback)': (False, True, False),
    }
    
    for name, (flash, math, mem_eff) in backends.items():
        try:
            with sdp_kernel(enable_flash=flash, enable_math=math, enable_mem_efficient=mem_eff):
                output = F.scaled_dot_product_attention(
                    dummy_q, dummy_k, dummy_v,
                    attn_mask=None,
                    is_causal=True
                )
            print(f"   {name}: ✓ AVAILABLE")
        except Exception as e:
            print(f"   {name}: ✗ UNAVAILABLE ({str(e)[:50]})")
    
    # 8. Test different head dimensions
    print(f"\n8. Testing head dimensions for Flash Attention:")
    for head_dim in [8, 16, 32, 64, 128, 256]:
        try:
            test_q = torch.randn(1, 8, 128, head_dim, device='cuda', dtype=torch.float16)
            test_k = torch.randn(1, 8, 128, head_dim, device='cuda', dtype=torch.float16)
            test_v = torch.randn(1, 8, 128, head_dim, device='cuda', dtype=torch.float16)
            
            with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                output = F.scaled_dot_product_attention(test_q, test_k, test_v, is_causal=True)
            print(f"   head_dim={head_dim}: ✓ WORKS")
        except Exception as e:
            print(f"   head_dim={head_dim}: ✗ FAILS")
    
    # 9. Check your actual dimensions
    print(f"\n9. Your Model Configuration:")
    embedding_size = 256
    nhead = 8
    head_dim = embedding_size // nhead
    print(f"   embedding_size: {embedding_size}")
    print(f"   nhead: {nhead}")
    print(f"   head_dim: {head_dim}")
    print(f"   Status: {'✓ VALID (divisible by 8)' if head_dim % 8 == 0 else '✗ INVALID'}")
    
    # 10. Test with your exact dimensions
    print(f"\n10. Testing with YOUR exact model dimensions:")
    print(f"    [batch=1, nhead=16, seq=200, head_dim=16]")
    try:
        your_q = torch.randn(1, 16, 200, 16, device='cuda', dtype=torch.float16)
        your_k = torch.randn(1, 16, 200, 16, device='cuda', dtype=torch.float16)
        your_v = torch.randn(1, 16, 200, 16, device='cuda', dtype=torch.float16)
        
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = F.scaled_dot_product_attention(your_q, your_k, your_v, is_causal=True)
        print(f"    ✓ SUCCESS - Flash Attention works with your dimensions!")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        print(f"\n    Trying with default backends (auto-select):")
        try:
            output = F.scaled_dot_product_attention(your_q, your_k, your_v, is_causal=True)
            print(f"    ✓ Works with auto-backend selection")
        except Exception as e2:
            print(f"    ✗ Still fails: {e2}")

else:
    print("\n✗ CUDA not available. Flash Attention requires CUDA.")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if not cuda_available:
    print("⚠️  CRITICAL: No CUDA available. You're running on CPU.")
    print("   Flash Attention requires CUDA GPU.")
elif not compute_ok:
    print("⚠️  CRITICAL: GPU compute capability too low.")
    print(f"   Your GPU: {gpu_name} (compute {compute_cap:.1f})")
    print("   Required: Compute capability >= 7.5")
    print("   Compatible GPUs: V100, T4, A100, A10, RTX 2060+, RTX 3000+, RTX 4000+")
else:
    print("ℹ️  Hardware requirements met, but Flash Attention still unavailable.")
    print("   This likely means PyTorch wasn't compiled with Flash Attention support.")


# In[11]:


# Verify use of xforms
def verify_xformers():
    """Verify xFormers is properly installed and working."""
    print("\n" + "="*70)
    print("VERIFYING XFORMERS INSTALLATION")
    print("="*70)
    
    try:
        import xformers
        print(f"✓ xFormers version: {xformers.__version__}")
        
        # Test memory-efficient attention
        from xformers.ops import memory_efficient_attention, LowerTriangularMask
        
        # Create test tensors
        batch, seq_len, nhead, head_dim = 2, 200, 16, 16
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        q = torch.randn(batch, seq_len, nhead, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, seq_len, nhead, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq_len, nhead, head_dim, device=device, dtype=torch.float16)
        
        # Test causal attention
        output = memory_efficient_attention(
            q, k, v,
            attn_bias=LowerTriangularMask(),
            p=0.0
        )
        
        print(f"✓ Memory-efficient attention working")
        print(f"  Input shape: {q.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Device: {device}")
        
        # Quick benchmark
        import time
        warmup = 5
        iterations = 20
        
        for _ in range(warmup):
            _ = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (batch * seq_len * iterations) / elapsed
        print(f"✓ Throughput: {throughput:.0f} tokens/sec")
        
        print("\n✓ xFormers is ready to use!")
        return True
        
    except ImportError as e:
        print(f"✗ xFormers not installed: {e}")
        print("  Install with: pip install xformers")
        return False
    except Exception as e:
        print(f"✗ xFormers test failed: {e}")
        return False

# Run verification
xformers_ok = verify_xformers()


# In[9]:


# 2. Import and verify
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# 3. Verify installation
try:
    import xformers
    from xformers.ops import memory_efficient_attention, LowerTriangularMask
    print(f"✓ xFormers {xformers.__version__} installed successfully")
    XFORMERS_AVAILABLE = True
except ImportError:
    print("✗ xFormers not available - will use fallback")
    XFORMERS_AVAILABLE = False

device = torch.device('cuda')

# Use FP16 (float16) instead of BF16
q = torch.randn(2, 200, 8, 32, device=device, dtype=torch.float16)  # Changed to float16, head_dim=32
k = torch.randn(2, 200, 8, 32, device=device, dtype=torch.float16)
v = torch.randn(2, 200, 8, 32, device=device, dtype=torch.float16)

try:
    output = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())
    print(f"✓ xFormers working with FP16 on {torch.cuda.get_device_name(0)}")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Still failed: {e}")


# In[ ]:





# ### Training

# #### Test single epoch

# In[29]:


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

print("="*70)
print("Cleaning up GPU memory from previous runs...")
print("="*70)
cleanup_gpu_memory()

config = FlashAttentionConfig(
    use_flash=True,
    use_rope=True,
    use_swiglu=True,
    use_prenorm=True,
    dtype=torch.float16,
    target_cd_cnt=8849 # Only for test, this is different when for formal retraining
)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\nStep 1: Creating Flash Attention model...")
model = FlashClinicalTransformer(config).to(device)

print("\nStep 2: Setting up training...")
batch_size = 16

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.95)
)

criterion = nn.BCEWithLogitsLoss()  # Multi-label loss for next days multiple code prediction
preparator = ClinicalDataPreparator(len_dy=200, len_cd=80, target_cd_cnt=config.target_cd_cnt)




# In[32]:


# Train
print("Training...")
print("="*70)

num_epochs = 1

for epoch in range(num_epochs):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*70}")

    # Train
    train_stats = train_epoch(
        model, df_train, optimizer, criterion,
        preparator, batch_size, device, config, epoch
    )

    # Validate
    val_stats = validate_epoch(
        model, df_val, criterion,
        preparator, batch_size, device, config
    )
    
    # Log epoch metrics
    logger.info(f"Epoch {epoch+1}/{num_epochs} completed")
    logger.info(f"  Train Loss: {train_stats['avg_loss']:.4f}")
    logger.info(f"  Val Loss: {val_stats['val_loss']:.4f}")
    logger.info(f"  Epoch Time: {train_stats['epoch_time']:.2f}s")
    logger.info(f"  Throughput: {train_stats.get('num_batches', 0) / train_stats['epoch_time']:.2f} batches/sec")

    # # Save checkpoint
    # if epoch % 2 == 0:
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'train_loss': train_stats['avg_loss'],
    #         'val_loss': val_stats['val_loss'],
    #         'config': config,
    #         'code_to_target': preparator.code_to_target  # Save mapping!
    #     }
    #     torch.save(checkpoint, f'flash_checkpoint_epoch{epoch}.pt')
    #     print(f"✓ Checkpoint saved: flash_checkpoint_epoch{epoch}.pt")

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)


# ### Benchmark

# In[9]:


# =============================================================================
# PERFORMANCE COMPARISON: DENSE vs FLASH ATTENTION
# =============================================================================

"""
Comprehensive Benchmark: Standard Attention vs Flash Attention

Metrics Tracked:
1. Training time per epoch
2. Validation time
3. GPU memory usage (peak and average)
4. Training loss
5. Validation loss
6. Throughput (samples/sec, tokens/sec)
7. Estimated GPU cost (based on A100 pricing)
"""

import torch
import gc
import time
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BenchmarkMetrics:
    """Store metrics for model comparison."""
    model_name: str
    train_time: float
    val_time: float
    train_loss: float
    val_loss: float
    peak_memory_mb: float
    avg_memory_mb: float
    samples_per_sec: float
    tokens_per_sec: float
    total_time: float
    
    def to_dict(self):
        return {
            'Model': self.model_name,
            'Train Time (s)': f"{self.train_time:.2f}",
            'Val Time (s)': f"{self.val_time:.2f}",
            'Total Time (s)': f"{self.total_time:.2f}",
            'Train Loss': f"{self.train_loss:.4f}",
            'Val Loss': f"{self.val_loss:.4f}",
            'Peak Memory (GB)': f"{self.peak_memory_mb/1024:.2f}",
            'Avg Memory (GB)': f"{self.avg_memory_mb/1024:.2f}",
            'Samples/sec': f"{self.samples_per_sec:.2f}",
            'Tokens/sec': f"{self.tokens_per_sec:.0f}"
        }


class ModelBenchmark:
    """Benchmark framework for comparing models."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results: List[BenchmarkMetrics] = []
    
    def benchmark_model(
        self,
        model_name: str,
        model: nn.Module,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        config: FlashAttentionConfig,
        preparator: ClinicalDataPreparator,
        criterion: nn.Module,
        batch_size: int = 16,
        num_epochs: int = 1
    ) -> BenchmarkMetrics:
        """
        Benchmark a single model.
        
        Args:
            model_name: Name for display
            model: The model to benchmark
            train_data: Training DataFrame
            val_data: Validation DataFrame
            config: Model configuration
            preparator: Data preparator
            criterion: Loss function
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
        
        Returns:
            BenchmarkMetrics with all tracked metrics
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {model_name}")
        print(f"{'='*70}")
        
        # Reset GPU stats
        cleanup_gpu_memory()
        torch.cuda.reset_peak_memory_stats()
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Track metrics
        memory_samples = []
        train_time_total = 0.0
        val_time_total = 0.0
        final_train_loss = 0.0
        final_val_loss = 0.0
        total_samples = 0
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_start = time.time()
            train_stats = self._train_epoch(
                model, train_data, optimizer, criterion,
                preparator, batch_size, config, epoch
            )
            train_time = time.time() - train_start
            train_time_total += train_time
            final_train_loss = train_stats['avg_loss']
            total_samples += len(train_data)
            
            # Track memory during training
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                memory_samples.append(memory_mb)
            
            # Validation
            val_start = time.time()
            val_stats = self._validate(
                model, val_data, criterion,
                preparator, batch_size, config
            )
            val_time = time.time() - val_start
            val_time_total += val_time
            final_val_loss = val_stats['val_loss']
            
            print(f"  Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}")
            print(f"  Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s")
        
        # Get memory stats
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            avg_memory_mb = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        else:
            peak_memory_mb = 0
            avg_memory_mb = 0
        
        # Calculate throughput
        total_time = train_time_total + val_time_total
        samples_per_sec = total_samples / train_time_total if train_time_total > 0 else 0
        # Assume avg sequence length = 100 days (conservative estimate)
        avg_seq_len = 100
        tokens_per_sec = samples_per_sec * avg_seq_len
        
        metrics = BenchmarkMetrics(
            model_name=model_name,
            train_time=train_time_total,
            val_time=val_time_total,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            samples_per_sec=samples_per_sec,
            tokens_per_sec=tokens_per_sec,
            total_time=total_time
        )
        
        self.results.append(metrics)
        return metrics
    
    def _train_epoch(
        self,
        model: nn.Module,
        data: pd.DataFrame,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        preparator: ClinicalDataPreparator,
        batch_size: int,
        config: FlashAttentionConfig,
        epoch: int
    ) -> Dict:
        """Single training epoch."""
        model.train()
        num_batches = len(data) // batch_size
        total_loss = 0.0
        
        scaler = torch.cuda.amp.GradScaler() if config.dtype == torch.float16 else None
        
        for i in range(num_batches):
            batch_df = data.iloc[i*batch_size:(i+1)*batch_size]
            dt_cnt, x, y = preparator.prepare_batch(batch_df, self.device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=config.dtype):
                output = model(x)
                output = output.reshape(batch_size * config.len_dy, config.target_cd_cnt)
                
                y_flat = [item for sublist in y for item in sublist]
                
                valid_outputs = []
                for j in range(batch_size):
                    start_idx = config.len_dy * j
                    end_idx = start_idx + dt_cnt[j]
                    valid_outputs.append(output[start_idx:end_idx])
                
                output = torch.cat(valid_outputs, dim=0)
                
                y_cd = torch.zeros(len(output), config.target_cd_cnt, device=self.device)
                for j in range(len(output)):
                    for k in y_flat[j]:
                        if k != 0 and k < config.target_cd_cnt:
                            y_cd[j, k] = 1
                
                loss = criterion(output, y_cd)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
            del x, output, loss, y_cd
            if i % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'avg_loss': avg_loss}
    
    def _validate(
        self,
        model: nn.Module,
        data: pd.DataFrame,
        criterion: nn.Module,
        preparator: ClinicalDataPreparator,
        batch_size: int,
        config: FlashAttentionConfig
    ) -> Dict:
        """Validation loop."""
        model.eval()
        num_batches = len(data) // batch_size
        total_loss = 0.0
        
        with torch.no_grad():
            for i in range(num_batches):
                batch_df = data.iloc[i*batch_size:(i+1)*batch_size]
                dt_cnt, x, y = preparator.prepare_batch(batch_df, self.device)
                with torch.cuda.amp.autocast(dtype=config.dtype):
                    output = model(x)
                    output = output.reshape(batch_size * config.len_dy, config.target_cd_cnt)

                    y_flat = [item for sublist in y for item in sublist]

                    valid_outputs = []
                    for j in range(batch_size):
                        start_idx = config.len_dy * j
                        end_idx = start_idx + dt_cnt[j]
                        valid_outputs.append(output[start_idx:end_idx])

                    output = torch.cat(valid_outputs, dim=0)

                    y_cd = torch.zeros(len(output), config.target_cd_cnt, device=self.device)
                    for j in range(len(output)):
                        for k in y_flat[j]:
                            if k != 0 and k < config.target_cd_cnt:
                                y_cd[j, k] = 1

                    loss = criterion(output, y_cd)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'val_loss': avg_loss}
    
    def compare_models(self) -> pd.DataFrame:
        """Generate comparison table."""
        if len(self.results) < 2:
            print("Need at least 2 models to compare")
            return None
        
        # Create comparison DataFrame
        df = pd.DataFrame([m.to_dict() for m in self.results])
        
        # Calculate improvements
        baseline = self.results[0]  # First model is baseline
        flash = self.results[1]     # Second model is Flash Attention
        
        speedup = baseline.total_time / flash.total_time if flash.total_time > 0 else 0
        memory_reduction = (1 - flash.peak_memory_mb / baseline.peak_memory_mb) * 100
        train_loss_diff = ((flash.train_loss - baseline.train_loss) / baseline.train_loss) * 100
        val_loss_diff = ((flash.val_loss - baseline.val_loss) / baseline.val_loss) * 100
        
        print(f"\n{'='*70}")
        print(f"PERFORMANCE COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print(f"IMPROVEMENT ANALYSIS")
        print(f"{'='*70}")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Memory Reduction: {memory_reduction:.1f}%")
        print(f"Train Loss Difference: {train_loss_diff:+.2f}%")
        print(f"Val Loss Difference: {val_loss_diff:+.2f}%")
        
        return df
    
    def estimate_cost(
        self,
        gpu_type: str = "T4",
        hours_per_epoch: float = None
    ):
        """
        Estimate training cost based on GPU pricing.
        
        GPU Hourly Rates (approximate, 2025):
        - A100-40GB: $3.57/hour
        - A100-80GB: $4.60/hour
        - V100-32GB: $2.58/hour
        - T4: $0.4025/hour
        """
        gpu_prices = {
            'A100': 3.57,      # AWS p4d.24xlarge / GCP a2-highgpu-1g
            'V100': 2.852,      # AWS p3.2xlarge / GCP n1-standard-8
            'T4': 0.4025,       # AWS g4dn.xlarge
        }
        
        hourly_rate = gpu_prices.get(gpu_type, 2.50)
        
        print(f"\n{'='*70}")
        print(f"COST ESTIMATION ({gpu_type} @ ${hourly_rate}/hour)")
        print(f"{'='*70}")
        
        for metrics in self.results:
            # Calculate cost for measured time
            hours = metrics.total_time / 3600
            cost = hours * hourly_rate
            
            print(f"\n{metrics.model_name}:")
            print(f"  Measured Time: {metrics.total_time:.2f}s ({hours:.4f} hours)")
            print(f"  Cost for Benchmark: ${cost:.4f}")
            
            # Estimate full training cost (if provided)
            if hours_per_epoch is not None:
                full_cost_per_epoch = hours_per_epoch * hourly_rate
                print(f"  Estimated Cost per Full Epoch: ${full_cost_per_epoch:.2f}")
                
                # Common training scenarios
                for epochs in [10, 50, 100]:
                    total_cost = full_cost_per_epoch * epochs
                    print(f"    {epochs} epochs: ${total_cost:.2f}")
        
        # Calculate savings
        if len(self.results) >= 2:
            baseline_hours = self.results[0].total_time / 3600
            flash_hours = self.results[1].total_time / 3600
            time_saved = baseline_hours - flash_hours
            cost_saved = time_saved * hourly_rate
            
            print(f"\n{'='*70}")
            print(f"COST SAVINGS (Flash Attention vs Baseline)")
            print(f"{'='*70}")
            print(f"Time Saved per Run: {time_saved * 3600:.2f}s ({time_saved:.4f} hours)")
            print(f"Cost Saved per Run: ${cost_saved:.4f}")
            
            if hours_per_epoch is not None:
                # Extrapolate to full training
                speedup = self.results[0].total_time / self.results[1].total_time
                flash_hours_full = hours_per_epoch / speedup
                time_saved_full = hours_per_epoch - flash_hours_full
                cost_saved_per_epoch = time_saved_full * hourly_rate
                
                print(f"\nEstimated Savings per Full Epoch:")
                print(f"  Time Saved: {time_saved_full:.2f} hours")
                print(f"  Cost Saved: ${cost_saved_per_epoch:.2f}")
                
                for epochs in [10, 50, 100]:
                    total_saved = cost_saved_per_epoch * epochs
                    print(f"    {epochs} epochs: ${total_saved:.2f}")


# =============================================================================
# RUN COMPARISON
# =============================================================================

def run_comparison_benchmark(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    batch_size: int = 16,
    num_epochs: int = 1,
    use_subset: bool = False,
    subset_size: int = 2000
):
    """
    Run comprehensive comparison between Dense and Flash Attention models.
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        use_subset: Whether to use a subset for faster comparison
        subset_size: Size of subset if use_subset=True
    """
    
    # Use subset for faster benchmarking
    if use_subset:
        train_subset = train_data.head(subset_size)
        val_subset = val_data.head(subset_size // 4)
        print(f"\nUsing subset: {len(train_subset)} train, {len(val_subset)} val samples")
    else:
        train_subset = train_data
        val_subset = val_data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benchmark = ModelBenchmark(device)
    
    # Configuration
    config_base = FlashAttentionConfig(
        use_flash=False,  # Standard attention
        use_rope=True,
        use_swiglu=True,
        use_prenorm=True,
        dtype=torch.float16,
        target_cd_cnt=8849
    )
    
    config_flash = FlashAttentionConfig(
        use_flash=True,   # Flash attention
        use_rope=True,
        use_swiglu=True,
        use_prenorm=True,
        dtype=torch.float16,
        target_cd_cnt=8849
    )
    
    criterion = nn.BCEWithLogitsLoss()
    preparator = ClinicalDataPreparator(len_dy=200, len_cd=80, target_cd_cnt=8849)
    
    # Benchmark 1: Standard Dense Attention
    print("\n" + "="*70)
    print("CREATING BASELINE MODEL (Standard Attention)")
    print("="*70)
    model_dense = FlashClinicalTransformer(config_base).to(device)
    
    metrics_dense = benchmark.benchmark_model(
        model_name="Standard Attention",
        model=model_dense,
        train_data=train_subset,
        val_data=val_subset,
        config=config_base,
        preparator=preparator,
        criterion=criterion,
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    
    # Clean up
    del model_dense
    cleanup_gpu_memory()
    
    # Benchmark 2: Flash Attention
    print("\n" + "="*70)
    print("CREATING FLASH ATTENTION MODEL")
    print("="*70)
    model_flash = FlashClinicalTransformer(config_flash).to(device)
    
    metrics_flash = benchmark.benchmark_model(
        model_name="Flash Attention",
        model=model_flash,
        train_data=train_subset,
        val_data=val_subset,
        config=config_flash,
        preparator=preparator,
        criterion=criterion,
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    
    # Generate comparison
    df_comparison = benchmark.compare_models()
    
    # Estimate costs
    benchmark.estimate_cost(
        gpu_type="T4",
        hours_per_epoch=None  # Set to actual hours if known
    )
    
    return df_comparison, benchmark.results


    


# In[10]:


# Option 1: Quick comparison on subset (recommended for testing)
print("\n" + "="*70)
print("RUNNING QUICK COMPARISON (Subset)")
print("="*70)
cleanup_gpu_memory()
# df_comparison, results = run_comparison_benchmark(
#     train_data=df_train,
#     val_data=df_val,
#     batch_size=16,
#     num_epochs=1,
#     use_subset=False,
#     subset_size=2000  # Use 2000 samples for quick test
# )
    
# Option 2: Full comparison (takes longer)
# print("\n" + "="*70)
# print("RUNNING FULL COMPARISON")
# print("="*70)
# 
df_comparison, results = run_comparison_benchmark(
    train_data=df_train,
    val_data=df_val,
    batch_size=16,
    num_epochs=1,
    use_subset=False
)

# Save results
if df_comparison is not None:
    df_comparison.to_csv('flash_attention_comparison.csv', index=False)
    print("\n✓ Results saved to 'flash_attention_comparison.csv'")


# In[ ]:




