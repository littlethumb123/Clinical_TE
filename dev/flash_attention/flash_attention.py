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


# ### Flash attention implementation

# In[1]:


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
        target_cd_cnt: Number of prediction targets (2,767)
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
    nhead: int = 16
    nlayers: int = 6
    nhid: int = 512
    dropout: float = 0.1
    
    # Vocabulary sizes
    cd_cnt: int = 84010
    target_cd_cnt: int = 2767
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
    dtype: torch.dtype = torch.bfloat16


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
        # Shape: [1, max_seq_len, 1, dim]
        # Dimensions: [batch, seq_len, nhead, head_dim]
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



# In[2]:


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
        
        # Flash Attention (PyTorch 2.0+)
        # This automatically uses Flash Attention when:
        # - CUDA capability >= 7.5 (Volta, Turing, Ampere, Hopper)
        # - head_dim is divisible by 8
        # - Using FP16 or BF16
        # - No custom attention mask (or only causal)
        if self.config.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=src_mask,  # Can be None
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal  # Optimized causal mask
            )
        else:
            # Fallback to standard attention if Flash not available
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
        
        encoder_layer_cd = TransformerEncoderLayer(
            d_model=config.embedding_size,
            nhead=4,  # 4 heads for daily encoder
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
        
        # Log softmax for NLLLoss
        predictions = F.log_softmax(predictions, dim=-1)
        
        return predictions


# In[3]:


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
        dtype: torch.bfloat16 or torch.float16
    """
    
    def __init__(
        self, 
        model: FlashClinicalTransformer,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16
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


# ### Validation and test flash attention with dummy dataset

# In[6]:


# Requirements:
# - PyTorch >= 2.0
# - CUDA compute capability >= 7.5 (Volta, Turing, Ampere, Hopper)
# - CUDA available
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Compute capability
    major, minor = torch.cuda.get_device_capability(0)
    compute_capability = major + minor / 10
    print(f"Compute capability: {compute_capability:.1f}")
    flash_ok = compute_capability >= 7.5
    print(f"  Flash Attention supported: {'✓' if flash_ok else '✗'}")

    # Check for Flash Attention function
    has_sdpa = hasattr(F, 'scaled_dot_product_attention')
    print(f"  scaled_dot_product_attention available: {'✓' if has_sdpa else '✗'}")
# BF16 support
if torch.cuda.is_available():
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"\nBF16 supported: {'✓' if bf16_ok else '✗'}")

print("="*60)


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
    
    def __init__(self, len_dy: int = 200, len_cd: int = 80, target_cd_cnt: int = 2767):
        self.len_dy = len_dy
        self.len_cd = len_cd
        self.target_cd_cnt = target_cd_cnt
        
        # Code mapping will be built from data
        self.code_to_target = None
    
    def build_code_mapping(self, data: pd.DataFrame):
        """
        Build mapping from raw medical codes to target indices.
        
        Strategy: Use top N most frequent codes as targets.
        All other codes map to index 0 (unknown/other).
        
        Args:
            data: Training DataFrame with 'cd' column
        """
        from collections import Counter
        
        print(f"Building code mapping for top {self.target_cd_cnt} codes...")
        
        # Collect all codes from training data
        all_codes = []
        for cd_str in data['cd']:
            if pd.notna(cd_str) and cd_str != '':
                # Parse all codes (handle both formats)
                codes = cd_str.replace('*', ',').split(',')
                codes = [int(c) for c in codes if c.strip() != '']
                all_codes.extend(codes)
        
        # Count frequencies
        code_counts = Counter(all_codes)
        
        # Get top N most frequent codes
        most_common = code_counts.most_common(self.target_cd_cnt)
        
        # Create mapping: raw_code -> target_index
        self.code_to_target = {}
        for target_idx, (code, count) in enumerate(most_common):
            self.code_to_target[code] = target_idx
        
        print(f"✓ Mapped {len(self.code_to_target)} codes to target indices")
        print(f"  Most common code: {most_common[0][0]} (count: {most_common[0][1]})")
        print(f"  Least common in targets: {most_common[-1][0]} (count: {most_common[-1][1]})")
        
        # Coverage statistics
        total_codes = len(all_codes)
        mapped_codes = sum(1 for c in all_codes if c in self.code_to_target)
        coverage = mapped_codes / total_codes * 100
        print(f"  Coverage: {coverage:.2f}% of training codes in target vocabulary")
    
    def map_code_to_target(self, code: int) -> int:
        """
        Map raw medical code to target index.
        
        Args:
            code: Raw medical code (0-84009)
        
        Returns:
            Target index (0-2766), or 0 if code not in target vocabulary
        """
        if self.code_to_target is None:
            raise ValueError("Code mapping not built. Call build_code_mapping() first!")
        
        return self.code_to_target.get(code, 0)  # Unknown codes map to 0
    
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
            values = [int(v) if v.strip() != '' else 0 for v in values]
        else:
            values = [int(value)] * num_days
        
        values = values[:self.len_dy]
        while len(values) < self.len_dy:
            values.append(values[-1] if values else 0)
        
        return values
    
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
            
            # **FIX**: Map codes to target indices
            day_targets = []
            for day_idx in range(day_count):
                # Get non-zero codes for this day
                day_codes = [c for c in codes_seq[day_idx] if c != 0]
                if day_codes:
                    # Map first code to target index
                    raw_code = day_codes[0]
                    target_idx = self.map_code_to_target(raw_code)
                    day_targets.append(target_idx)
                else:
                    day_targets.append(0)
            
            targets.append(day_targets)
        
        # Convert to tensors
        age_tensor = torch.tensor(age_batch, dtype=torch.long, device=device)
        gender_tensor = torch.tensor(gender_batch, dtype=torch.long, device=device)
        codes_tensor = torch.tensor(codes_batch, dtype=torch.long, device=device)
        
        age_tensor = age_tensor.unsqueeze(-1)
        gender_tensor = gender_tensor.unsqueeze(-1)
        
        x = torch.cat([age_tensor, gender_tensor, codes_tensor], dim=-1)
        x = x.float()
        
        return dt_cnt, x, targets


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
        criterion: Loss function (NLLLoss)
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
    scaler = torch.cuda.amp.GradScaler()
    
    start_time = time.time()
    
    for i in range(num_batches):
        if i % 100 == 0:
            print(f'Epoch {epoch}, Batch {i}/{num_batches}, Time: {time.time()-start_time:.2f}s')
        
        # Get batch
        batch_df = data.iloc[i*batch_size:(i+1)*batch_size]
        
        # Prepare tensors
        dt_cnt, x, y = preparator.prepare_batch(batch_df, device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(dtype=config.dtype):
            output = model(x)  # [batch, 200, target_cd_cnt]
            
            # Reshape for loss computation
            output = output.reshape(batch_size * config.len_dy, config.target_cd_cnt)
            
            # Flatten targets
            y_flat = [item for sublist in y for item in sublist]
            
            # Select only valid days
            valid_outputs = []
            for j in range(batch_size):
                start_idx = config.len_dy * j
                end_idx = start_idx + dt_cnt[j]
                valid_outputs.append(output[start_idx:end_idx])
            
            output = torch.cat(valid_outputs, dim=0)
            y_tensor = torch.tensor(y_flat, device=device, dtype=torch.long)
            
            # Ensure same length
            min_len = min(len(output), len(y_tensor))
            output = output[:min_len]
            y_tensor = y_tensor[:min_len]
            
            # Compute loss
            loss = criterion(output, y_tensor)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Memory cleanup
        del x, output, loss
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


def validate(
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
            y_tensor = torch.tensor(y_flat, device=device, dtype=torch.long)
            
            min_len = min(len(output), len(y_tensor))
            output = output[:min_len]
            y_tensor = y_tensor[:min_len]
            
            loss = criterion(output, y_tensor)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")
    
    return {'val_loss': avg_loss}


# #### Load data

# In[8]:


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


# In[11]:


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



# In[12]:


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

print("="*70)
print("Cleaning up GPU memory from previous runs...")
print("="*70)
cleanup_gpu_memory()

train_data = input_data.iloc[:1500]
val_data = input_data.iloc[1501:2000]

config = FlashAttentionConfig(
    use_flash=True,
    use_rope=True,
    use_swiglu=True,
    use_prenorm=True,
    dtype=torch.bfloat16
)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\nStep 1: Creating Flash Attention model...")
model = FlashClinicalTransformer(config).to(device)

print("\nStep 2: Setting up training...")
batch_size = 32

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.95)
)

criterion = nn.NLLLoss()
preparator = ClinicalDataPreparator(len_dy=200, len_cd=80)
preparator.build_code_mapping(train_data)


# Step 6: Train
print("\nStep 3: Training...")
print("="*70)

num_epochs = 5

for epoch in range(num_epochs):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*70}")

    # Train
    train_stats = train_epoch(
        model, train_data, optimizer, criterion,
        preparator, batch_size, device, config, epoch
    )

    # Validate
    val_stats = validate(
        model, val_data, criterion,
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


# OUTPUT
# ======================================================================
# Cleaning up GPU memory from previous runs...
# ======================================================================
# GPU Memory - Allocated: 0.70 GB, Reserved: 1.83 GB
# Using device: cuda

# Step 1: Creating Flash Attention model...

# Step 2: Setting up training...
# Building code mapping for top 2767 codes...
# ✓ Mapped 1779 codes to target indices
#   Most common code: 1 (count: 1499)
#   Least common in targets: 5001 (count: 1)
#   Coverage: 100.00% of training codes in target vocabulary

# Step 3: Training...
# ======================================================================

# ======================================================================
# Epoch 1/5
# ======================================================================
# Epoch 0, Batch 0/46, Time: 0.00s

# Epoch 0 completed: Avg Loss = 6.7877, Time = 56.73s
# 2025-10-26 06:26:48 - INFO - Epoch 1/5 completed
# 2025-10-26 06:26:48 - INFO -   Train Loss: 6.7877
# 2025-10-26 06:26:48 - INFO -   Val Loss: 5.9962
# 2025-10-26 06:26:48 - INFO -   Epoch Time: 56.73s
# 2025-10-26 06:26:48 - INFO -   Throughput: 0.81 batches/sec
# Validation Loss: 5.9962

# ======================================================================
# Epoch 2/5
# ======================================================================
# Epoch 1, Batch 0/46, Time: 0.00s

# Epoch 1 completed: Avg Loss = 5.9135, Time = 58.28s
# 2025-10-26 06:27:51 - INFO - Epoch 2/5 completed
# 2025-10-26 06:27:51 - INFO -   Train Loss: 5.9135
# 2025-10-26 06:27:51 - INFO -   Val Loss: 5.6089
# 2025-10-26 06:27:51 - INFO -   Epoch Time: 58.28s
# 2025-10-26 06:27:51 - INFO -   Throughput: 0.79 batches/sec
# Validation Loss: 5.6089

# ======================================================================
# Epoch 3/5
# ======================================================================
# Epoch 2, Batch 0/46, Time: 0.00s

# Epoch 2 completed: Avg Loss = 5.5273, Time = 58.20s
# 2025-10-26 06:28:54 - INFO - Epoch 3/5 completed
# 2025-10-26 06:28:54 - INFO -   Train Loss: 5.5273
# 2025-10-26 06:28:54 - INFO -   Val Loss: 5.4233
# 2025-10-26 06:28:54 - INFO -   Epoch Time: 58.20s
# 2025-10-26 06:28:54 - INFO -   Throughput: 0.79 batches/sec
# Validation Loss: 5.4233

# ======================================================================
# Epoch 4/5
# ======================================================================
# Epoch 3, Batch 0/46, Time: 0.00s

# Epoch 3 completed: Avg Loss = 5.3262, Time = 58.26s
# 2025-10-26 06:29:58 - INFO - Epoch 4/5 completed
# 2025-10-26 06:29:58 - INFO -   Train Loss: 5.3262
# 2025-10-26 06:29:58 - INFO -   Val Loss: 5.3796
# 2025-10-26 06:29:58 - INFO -   Epoch Time: 58.26s
# 2025-10-26 06:29:58 - INFO -   Throughput: 0.79 batches/sec
# Validation Loss: 5.3796

# ======================================================================
# Epoch 5/5
# ======================================================================
# Epoch 4, Batch 0/46, Time: 0.00s

# Epoch 4 completed: Avg Loss = 5.2410, Time = 58.22s
# 2025-10-26 06:31:01 - INFO - Epoch 5/5 completed
# 2025-10-26 06:31:01 - INFO -   Train Loss: 5.2410
# 2025-10-26 06:31:01 - INFO -   Val Loss: 5.4012
# 2025-10-26 06:31:01 - INFO -   Epoch Time: 58.22s
# 2025-10-26 06:31:01 - INFO -   Throughput: 0.79 batches/sec
# Validation Loss: 5.4012

# ======================================================================
# TRAINING COMPLETED!
# ======================================================================

# In[ ]:




