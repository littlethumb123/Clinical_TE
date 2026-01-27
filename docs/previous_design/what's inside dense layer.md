# 🔬 Inside a Temporal Layer: Complete Architecture

## 📦 Layer Structure Overview

Each temporal layer is a `ModuleDict` containing 4 components:

```python
# From moe_flashattn_3.py (lines 2455-2460)
self.temporal_layers.append(nn.ModuleDict({
    'attention': attn,      # FlashAttentionLayer
    'ffn': ffn,             # SwiGLU or MoELayer
    'norm1': norm1,         # LayerNorm (before attention)
    'norm2': norm2          # LayerNorm (before FFN)
}))
```

---

## 🏗️ Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SINGLE TEMPORAL LAYER                                  │
│                         (Pre-Norm Architecture)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT: x [seq_len=200, batch, d_model=256]                                     │
│         ↓                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  ═══════════════════════ ATTENTION BLOCK ════════════════════════════   │   │
│  │                                                                          │   │
│  │   residual = x ─────────────────────────────────────────────────┐        │   │
│  │         ↓                                                       │        │   │
│  │   ┌─────────────────────────────────────────────────────────┐   │        │   │
│  │   │              norm1: LayerNorm(256)                      │   │        │   │
│  │   │              Normalize across embedding dimension       │   │        │   │
│  │   └────────────────────────┬────────────────────────────────┘   │        │   │
│  │                            ↓                                     │        │   │
│  │   ┌─────────────────────────────────────────────────────────┐   │        │   │
│  │   │         FlashAttentionLayer (is_causal=True)            │   │        │   │
│  │   │  ┌─────────────────────────────────────────────────┐    │   │        │   │
│  │   │  │  1. Linear Projections (no bias)                │    │   │        │   │
│  │   │  │     q_proj: Linear(256 → 256)                   │    │   │        │   │
│  │   │  │     k_proj: Linear(256 → 256)                   │    │   │        │   │
│  │   │  │     v_proj: Linear(256 → 256)                   │    │   │        │   │
│  │   │  ├─────────────────────────────────────────────────┤    │   │        │   │
│  │   │  │  2. Reshape to Multi-Head                       │    │   │        │   │
│  │   │  │     [seq, batch, 256] → [batch, 8, seq, 32]     │    │   │        │   │
│  │   │  │     (8 heads × 32 head_dim)                     │    │   │        │   │
│  │   │  ├─────────────────────────────────────────────────┤    │   │        │   │
│  │   │  │  3. Rotary Position Embedding (RoPE)            │    │   │        │   │
│  │   │  │     Apply rotation to Q and K                   │    │   │        │   │
│  │   │  │     q_rot = q * cos + rotate_half(q) * sin      │    │   │        │   │
│  │   │  │     k_rot = k * cos + rotate_half(k) * sin      │    │   │        │   │
│  │   │  ├─────────────────────────────────────────────────┤    │   │        │   │
│  │   │  │  4. Flash Attention (xFormers)                  │    │   │        │   │
│  │   │  │     memory_efficient_attention(q, k, v,         │    │   │        │   │
│  │   │  │         attn_bias=LowerTriangularMask(),        │    │   │        │   │
│  │   │  │         scale=1/√32)                            │    │   │        │   │
│  │   │  │                                                 │    │   │        │   │
│  │   │  │     Causal Mask (automatically applied):        │    │   │        │   │
│  │   │  │     ┌─────────────────────────────┐             │    │   │        │   │
│  │   │  │     │ 1  0  0  0  0  ...  0  0    │             │    │   │        │   │
│  │   │  │     │ 1  1  0  0  0  ...  0  0    │             │    │   │        │   │
│  │   │  │     │ 1  1  1  0  0  ...  0  0    │             │    │   │        │   │
│  │   │  │     │ ...                         │             │    │   │        │   │
│  │   │  │     │ 1  1  1  1  1  ...  1  1    │ (200×200)   │    │   │        │   │
│  │   │  │     └─────────────────────────────┘             │    │   │        │   │
│  │   │  ├─────────────────────────────────────────────────┤    │   │        │   │
│  │   │  │  5. Reshape Back                                │    │   │        │   │
│  │   │  │     [batch, 8, seq, 32] → [seq, batch, 256]     │    │   │        │   │
│  │   │  ├─────────────────────────────────────────────────┤    │   │        │   │
│  │   │  │  6. Output Projection + Dropout                 │    │   │        │   │
│  │   │  │     out_proj: Linear(256 → 256)                 │    │   │        │   │
│  │   │  │     dropout(p=0.05)                             │    │   │        │   │
│  │   │  └─────────────────────────────────────────────────┘    │   │        │   │
│  │   └────────────────────────┬────────────────────────────────┘   │        │   │
│  │                            ↓                                     │        │   │
│  │                       ADD (Residual)  ←─────────────────────────┘        │   │
│  │                            ↓                                              │   │
│  └────────────────────────────┬─────────────────────────────────────────────┘   │
│                               ↓                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  ═══════════════════════ FFN BLOCK ══════════════════════════════════   │   │
│  │                                                                          │   │
│  │   residual = x ─────────────────────────────────────────────────┐        │   │
│  │         ↓                                                       │        │   │
│  │   ┌─────────────────────────────────────────────────────────┐   │        │   │
│  │   │              norm2: LayerNorm(256)                      │   │        │   │
│  │   └────────────────────────┬────────────────────────────────┘   │        │   │
│  │                            ↓                                     │        │   │
│  │   ╔═════════════════════════════════════════════════════════╗   │        │   │
│  │   ║  OPTION A: SwiGLU (Dense Model)                         ║   │        │   │
│  │   ║  ┌───────────────────────────────────────────────────┐  ║   │        │   │
│  │   ║  │  w_gate: Linear(256 → 341)   # (2/3) × 512        │  ║   │        │   │
│  │   ║  │  w_up:   Linear(256 → 341)                        │  ║   │        │   │
│  │   ║  │  w_down: Linear(341 → 256)                        │  ║   │        │   │
│  │   ║  │                                                   │  ║   │        │   │
│  │   ║  │  gate = SiLU(w_gate(x))    # Swish activation     │  ║   │        │   │
│  │   ║  │  up = w_up(x)                                     │  ║   │        │   │
│  │   ║  │  hidden = gate * up        # Gating               │  ║   │        │   │
│  │   ║  │  output = dropout(w_down(hidden))                 │  ║   │        │   │
│  │   ║  └───────────────────────────────────────────────────┘  ║   │        │   │
│  │   ╠═════════════════════════════════════════════════════════╣   │        │   │
│  │   ║  OPTION B: MoELayer (Layers 2-5 in MoE Model)           ║   │        │   │
│  │   ║  ┌───────────────────────────────────────────────────┐  ║   │        │   │
│  │   ║  │  1. ROUTER                                        │  ║   │        │   │
│  │   ║  │     router: Linear(256 → num_routed_experts)      │  ║   │        │   │
│  │   ║  │     logits = router(x)                            │  ║   │        │   │
│  │   ║  │     probs = softmax(logits)                       │  ║   │        │   │
│  │   ║  │     top_k_indices = topk(probs, k=2)              │  ║   │        │   │
│  │   ║  │                                                   │  ║   │        │   │
│  │   ║  │  2. EXPERT DISPATCH                               │  ║   │        │   │
│  │   ║  │     For each token:                               │  ║   │        │   │
│  │   ║  │       - Route to top-K experts (K=2)              │  ║   │        │   │
│  │   ║  │       - Each expert is an ExpertLayer (FFN)       │  ║   │        │   │
│  │   ║  │                                                   │  ║   │        │   │
│  │   ║  │  3. EXPERTS (8 total, each is 2-layer FFN)        │  ║   │        │   │
│  │   ║  │     ┌─────────────────────────────────────────┐   │  ║   │        │   │
│  │   ║  │     │ Expert 0: Linear(256→512)→GELU→Linear   │   │  ║   │        │   │
│  │   ║  │     │ Expert 1: Linear(256→512)→GELU→Linear   │   │  ║   │        │   │
│  │   ║  │     │ Expert 2: Linear(256→512)→GELU→Linear   │   │  ║   │        │   │
│  │   ║  │     │ ...                                     │   │  ║   │        │   │
│  │   ║  │     │ Expert 7: Linear(256→512)→GELU→Linear   │   │  ║   │        │   │
│  │   ║  │     └─────────────────────────────────────────┘   │  ║   │        │   │
│  │   ║  │                                                   │  ║   │        │   │
│  │   ║  │  4. COMBINE                                       │  ║   │        │   │
│  │   ║  │     output = Σ (expert_i(x) × weight_i)           │  ║   │        │   │
│  │   ║  │                                                   │  ║   │        │   │
│  │   ║  │  5. SHARED EXPERTS (Optional)                     │  ║   │        │   │
│  │   ║  │     If num_shared_experts > 0:                    │  ║   │        │   │
│  │   ║  │       output += shared_expert(x)                  │  ║   │        │   │
│  │   ║  │                                                   │  ║   │        │   │
│  │   ║  │  6. AUXILIARY LOSSES (Training only)              │  ║   │        │   │
│  │   ║  │     - aux_loss: Load balancing                    │  ║   │        │   │
│  │   ║  │     - z_loss: Router z-regularization             │  ║   │        │   │
│  │   ║  └───────────────────────────────────────────────────┘  ║   │        │   │
│  │   ╚═════════════════════════════════════════════════════════╝   │        │   │
│  │                            ↓                                     │        │   │
│  │                       ADD (Residual)  ←─────────────────────────┘        │   │
│  │                            ↓                                              │   │
│  └────────────────────────────┬─────────────────────────────────────────────┘   │
│                               ↓                                                 │
│  OUTPUT: x [seq_len=200, batch, d_model=256]                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Component-by-Component Breakdown

### 1️⃣ LayerNorm (norm1, norm2)

```python
# From moe_flashattn_3.py (lines 2452-2453)
norm1 = nn.LayerNorm(config.embedding_size)  # 256
norm2 = nn.LayerNorm(config.embedding_size)  # 256
```

| Property | Value |
|----------|-------|
| **Parameters** | 2 × 256 = 512 (scale + bias) |
| **Operation** | Normalize across embedding dim |
| **Purpose** | Stabilize training (pre-norm pattern) |

---

### 2️⃣ FlashAttentionLayer

```python
# From moe_flashattn_3.py (lines 1430-1445)
self.q_proj = nn.Linear(d_model, d_model, bias=False)  # 256 × 256
self.k_proj = nn.Linear(d_model, d_model, bias=False)  # 256 × 256
self.v_proj = nn.Linear(d_model, d_model, bias=False)  # 256 × 256
self.out_proj = nn.Linear(d_model, d_model, bias=False)  # 256 × 256

if use_rope:
    self.rope = RotaryPositionEmbedding(dim=self.head_dim, max_seq_len=200)
```

| Sub-Component | Parameters | Purpose |
|---------------|------------|---------|
| **q_proj** | 65,536 | Query projection |
| **k_proj** | 65,536 | Key projection |
| **v_proj** | 65,536 | Value projection |
| **out_proj** | 65,536 | Output projection |
| **RoPE** | 0 (cached buffers) | Relative position encoding |
| **Total** | **262,144** | ~262K params per layer |

**Attention Flow:**
```
x → q_proj(x), k_proj(x), v_proj(x)
  → reshape to [batch, 8 heads, 200 seq, 32 head_dim]
  → apply RoPE (rotation based on position)
  → xFormers memory_efficient_attention with causal mask
  → reshape back to [200, batch, 256]
  → out_proj + dropout
```

---

### 3️⃣ Rotary Position Embedding (RoPE)

```python
# From moe_flashattn_3.py (lines 1303-1304)
q_rot = (q * cos) + (self.rotate_half(q) * sin)
k_rot = (k * cos) + (self.rotate_half(k) * sin)
```

| Property | Value |
|----------|-------|
| **Parameters** | 0 (precomputed buffers) |
| **Cached Tensors** | cos_cached, sin_cached [1, 1, 200, 32] |
| **Operation** | Rotate Q and K in complex plane |
| **Benefit** | Captures relative position; extrapolates to longer sequences |

---

### 4️⃣ SwiGLU FFN (Dense Model)

```python
# From moe_flashattn_3.py (lines 1331-1342)
d_ff_adjusted = int((2 * d_ff) / 3)  # 341 for d_ff=512

self.w_gate = nn.Linear(d_model, d_ff_adjusted, bias=False)  # 256 → 341
self.w_up = nn.Linear(d_model, d_ff_adjusted, bias=False)    # 256 → 341
self.w_down = nn.Linear(d_ff_adjusted, d_model, bias=False)  # 341 → 256

def forward(self, x):
    gate = F.silu(self.w_gate(x))  # Swish activation
    up = self.w_up(x)
    hidden = gate * up              # Gating
    output = self.w_down(hidden)
    return self.dropout(output)
```

| Sub-Component | Parameters | Purpose |
|---------------|------------|---------|
| **w_gate** | 256 × 341 = 87,296 | Gating signal |
| **w_up** | 256 × 341 = 87,296 | Expansion |
| **w_down** | 341 × 256 = 87,296 | Projection back |
| **Total** | **261,888** | ~262K params per layer |

---

### 5️⃣ MoELayer (For MoE Models, Layers 2-5)

```python
# From moe_flashattn_3.py (lines 1917-1947)
self.router = nn.Linear(config.d_model, self.num_routed_experts, bias=False)  # 256 → 8

self.experts = nn.ModuleList([
    ExpertLayer(d_model=256, d_ff=512, use_swiglu=False)  # GELU FFN
    for _ in range(8)  # 8 experts
])

# Optional shared experts (always active)
if self.num_shared_experts > 0:
    self.shared_experts = nn.ModuleList([...])
```

| Sub-Component | Parameters | Purpose |
|---------------|------------|---------|
| **Router** | 256 × 8 = 2,048 | Select top-K experts per token |
| **Expert (×8)** | Each: 256×512 + 512×256 = 262,144 | FFN for specialized processing |
| **Total Experts** | 8 × 262,144 = 2,097,152 | But only K=2 active per token |
| **Shared Expert** | (Optional) 262,144 | Always active |

**MoE Flow:**
```
x → router(x) → softmax → select top-K experts
  → dispatch tokens to selected experts
  → expert_outputs = [expert_i(x) for i in selected]
  → weighted_sum = Σ (expert_output_i × router_weight_i)
  → (optional) + shared_expert(x)
  → return output, {aux_loss, z_loss}
```

---

## 📊 Parameter Count per Temporal Layer

### Dense Model (SwiGLU FFN)

| Component | Parameters |
|-----------|------------|
| norm1 | 512 |
| FlashAttentionLayer | 262,144 |
| norm2 | 512 |
| SwiGLU | 261,888 |
| **Total per Layer** | **525,056** (~525K) |

**6 Layers Total**: 6 × 525K = **~3.15M parameters** (temporal encoder only)

---

### MoE Model (Layers 0-1: SwiGLU, Layers 2-5: MoE)

| Layer | Component | Parameters |
|-------|-----------|------------|
| 0-1 | SwiGLU (×2) | 2 × 525K = 1.05M |
| 2-5 | MoE (×4) | 4 × (262K attention + 512 norms + 2.1M experts) = 9.5M |
| **Total** | | **~10.5M parameters** (temporal encoder only) |

---

## 🔄 Forward Pass Execution (Code Evidence)

From `moe_flashattn_3.py` (lines 2823-2847):

```python
# Standard forward (MoE layers or non-checkpointed)
# Flash Attention block
residual = cd
cd_norm = layer['norm1'](cd)                    # Pre-norm
cd_attn = layer['attention'](cd_norm, is_causal=True)  # FlashAttention
cd = residual + cd_attn                          # Residual connection

# FFN block (MoE or standard)
residual = cd
cd_norm = layer['norm2'](cd)                    # Pre-norm

if isinstance(layer['ffn'], MoELayer):
    cd_ffn, moe_losses = layer['ffn'](cd_norm, train=self.training)
else:
    cd_ffn = layer['ffn'](cd_norm)              # SwiGLU or standard FFN

cd = residual + cd_ffn                          # Residual connection
```

---

## 🎯 Summary: What Each Component Does

| Component | What It Learns |
|-----------|---------------|
| **norm1/norm2** | Stabilize gradients; normalize feature scales |
| **Q/K/V Projections** | What to "query", what to "match", what to "retrieve" |
| **RoPE** | Relative temporal relationships between days |
| **Causal Attention** | "Given days 0...t, what patterns emerge?" |
| **SwiGLU/FFN** | Non-linear transformations; pattern refinement |
| **MoE Router** | "Which expert should handle this clinical pattern?" |
| **Experts** | Specialized pattern processing (e.g., cardiac, diabetes, etc.) |
| **Residual Connections** | Identity shortcuts for gradient flow |




# Deep Dive: LayerNorm, Dropout, GELU, and SwiGLU
## 1. Layer Normalization (LayerNorm)

### What It Does
LayerNorm normalizes the activations **across the feature dimension** for each sample independently, stabilizing training and enabling faster convergence.

### Mathematical Formulation

For an input vector $\mathbf{x} = [x_1, x_2, ..., x_d]$ where $d$ is the feature dimension (256 in this model):

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- **Mean**: $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$
- **Variance**: $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$
- $\gamma, \beta \in \mathbb{R}^d$ are **learnable** scale and shift parameters
- $\epsilon \approx 10^{-5}$ prevents division by zero
- $\odot$ denotes element-wise multiplication

### Step-by-Step Example

```
Input: x = [2.0, 4.0, 6.0, 8.0]  (d=4)

Step 1: Compute mean
μ = (2 + 4 + 6 + 8) / 4 = 5.0

Step 2: Compute variance
σ² = [(2-5)² + (4-5)² + (6-5)² + (8-5)²] / 4
   = [9 + 1 + 1 + 9] / 4 = 5.0

Step 3: Normalize
x_norm = (x - μ) / √(σ² + ε)
       = [-3, -1, 1, 3] / √5.0
       = [-1.34, -0.45, 0.45, 1.34]

Step 4: Scale and shift (learned γ=1, β=0 initially)
output = γ * x_norm + β = [-1.34, -0.45, 0.45, 1.34]
```

### Why Use LayerNorm?

| Benefit | Explanation |
|---------|-------------|
| **Stable Gradients** | Prevents activations from exploding or vanishing |
| **Faster Convergence** | Normalized inputs allow larger learning rates |
| **Batch-Size Independent** | Works with any batch size (unlike BatchNorm) |
| **Sequence-Friendly** | Each position normalized independently |

### In This Model

```python
# Pre-norm architecture (GPT-style)
x_norm = self.norm1(x)      # Normalize BEFORE attention
attn_out = attention(x_norm)
x = x + attn_out            # Residual with unnormalized x
```

---

## 2. Dropout

### What It Does
Dropout **randomly zeros out** a fraction of neurons during training, preventing co-adaptation and acting as regularization.

### Mathematical Formulation

During **training** with dropout probability $p$:

$$\text{Dropout}(\mathbf{x})_i = \begin{cases} 
0 & \text{with probability } p \\
\frac{x_i}{1-p} & \text{with probability } 1-p
\end{cases}$$

The scaling by $\frac{1}{1-p}$ ensures expected value remains unchanged:

$$\mathbb{E}[\text{Dropout}(\mathbf{x})] = \mathbf{x}$$

During **inference**: Dropout is disabled, output = input.

### Step-by-Step Example

```
Training with p = 0.1 (10% dropout):

Input:  x = [0.5, 0.8, 0.3, 0.6, 0.9]

Step 1: Generate random mask (Bernoulli sampling)
mask = [1, 0, 1, 1, 1]  (0 = drop, 1 = keep)

Step 2: Apply mask and scale
scale = 1 / (1 - 0.1) = 1.111

output = x * mask * scale
       = [0.5×1×1.111, 0.8×0×1.111, 0.3×1×1.111, 0.6×1×1.111, 0.9×1×1.111]
       = [0.556, 0.0, 0.333, 0.667, 1.0]
```

### Why Use Dropout?

| Benefit | Explanation |
|---------|-------------|
| **Prevents Overfitting** | Forces network to learn redundant representations |
| **Ensemble Effect** | Training samples different "sub-networks" |
| **Reduces Co-adaptation** | Neurons can't rely on specific other neurons |

### In This Model

```python
self.dropout = nn.Dropout(0.1)  # 10% dropout

# Applied after attention and FFN
attn_output = self.out_proj(attn_output)
attn_output = self.dropout(attn_output)  # ← Here
x = x + attn_output
```

---

## 3. GELU (Gaussian Error Linear Unit)

### What It Does
GELU is a **smooth, non-monotonic activation function** that applies a probabilistic gating based on the Gaussian distribution.

### Mathematical Formulation

**Exact form:**

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

Where $\Phi(x)$ is the CDF of standard normal distribution.

**Fast approximation (used in practice):**

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

### Intuition

GELU can be interpreted as:
- **Probabilistic gating**: Each input has a probability of being "passed through"
- **Smooth ReLU**: Unlike ReLU's hard cutoff at 0, GELU transitions smoothly
- **Non-monotonic**: Can suppress small positive values (unlike ReLU)

### Comparison with Other Activations

| x value | ReLU | GELU | Interpretation |
|---------|------|------|----------------|
| -2.0 | 0.0 | -0.045 | GELU slightly negative |
| -1.0 | 0.0 | -0.159 | GELU more negative |
| 0.0 | 0.0 | 0.0 | Same at origin |
| 0.5 | 0.5 | 0.345 | GELU suppresses small positives |
| 1.0 | 1.0 | 0.841 | GELU slightly less |
| 2.0 | 2.0 | 1.955 | Nearly linear for large x |

### Visualization

```
        GELU vs ReLU
  y │
  2 │                    ╱ ReLU
    │                  ╱
  1 │              ╱ ╱ GELU
    │          ╱ ╱
  0 │─────────●───────────────
    │      ╱   ╲
 -1 │    ╱      ╲ (GELU dips below 0)
    └──────────────────────── x
     -2  -1   0   1   2
```

### Why GELU Over ReLU?

| Property | ReLU | GELU |
|----------|------|------|
| Smooth | ❌ No (sharp corner at 0) | ✅ Yes (infinitely differentiable) |
| Non-monotonic | ❌ No | ✅ Yes (can suppress small values) |
| Gradient at 0 | Undefined | 0.5 |
| "Dead neurons" | ❌ Yes (x<0 → gradient=0) | ✅ No (always has gradient) |

GELU has shown empirically better performance in transformers (BERT, GPT).

---

## 4. SwiGLU (Swish-Gated Linear Unit)

### What It Does
SwiGLU is a **gated activation** that uses two parallel linear projections with element-wise gating, providing richer feature interactions.

### Mathematical Formulation

For input $\mathbf{x} \in \mathbb{R}^{d}$:

$$\text{SwiGLU}(\mathbf{x}) = \text{Swish}(\mathbf{W}_{\text{gate}}\mathbf{x}) \odot (\mathbf{W}_{\text{up}}\mathbf{x})$$

$$\text{Output} = \mathbf{W}_{\text{down}}[\text{SwiGLU}(\mathbf{x})]$$

Where:
- **Swish** (also called SiLU): $\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$
- $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d' \times d}$ — gate projection
- $\mathbf{W}_{\text{up}} \in \mathbb{R}^{d' \times d}$ — value projection  
- $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d \times d'}$ — output projection
- $\odot$ — element-wise multiplication (gating)

### Step-by-Step Example

```
Input: x = [1.0, 2.0]  (d=2)
Hidden: d' = 3

W_gate = [[0.5, 0.3],    W_up = [[0.4, 0.2],
          [0.2, 0.6],            [0.3, 0.5],
          [0.4, 0.1]]            [0.1, 0.4]]

Step 1: Gate projection
gate_linear = W_gate @ x = [0.5×1+0.3×2, 0.2×1+0.6×2, 0.4×1+0.1×2]
            = [1.1, 1.4, 0.6]

Step 2: Apply Swish to gate
Swish(1.1) = 1.1 × σ(1.1) = 1.1 × 0.75 = 0.825
Swish(1.4) = 1.4 × σ(1.4) = 1.4 × 0.80 = 1.12
Swish(0.6) = 0.6 × σ(0.6) = 0.6 × 0.65 = 0.39

gate = [0.825, 1.12, 0.39]

Step 3: Value projection
up = W_up @ x = [0.4×1+0.2×2, 0.3×1+0.5×2, 0.1×1+0.4×2]
   = [0.8, 1.3, 0.9]

Step 4: Element-wise gating
hidden = gate ⊙ up = [0.825×0.8, 1.12×1.3, 0.39×0.9]
       = [0.66, 1.456, 0.351]

Step 5: Down projection
output = W_down @ hidden  (back to d=2)
```

### Code Implementation

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        # Adjust for parameter equivalence with standard FFN
        d_ff_adjusted = int((2 * d_ff) / 3)  # 512 → 341
        
        self.w_gate = nn.Linear(d_model, d_ff_adjusted, bias=False)
        self.w_up = nn.Linear(d_model, d_ff_adjusted, bias=False)
        self.w_down = nn.Linear(d_ff_adjusted, d_model, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish activation
        up = self.w_up(x)
        hidden = gate * up              # Element-wise gating
        return self.w_down(hidden)
```

---

## 5. GELU vs SwiGLU: Detailed Comparison

### Architecture Difference

```
STANDARD FFN (with GELU):
─────────────────────────
x ──→ Linear(256→512) ──→ GELU ──→ Linear(512→256) ──→ output
         W₁                           W₂

SWIGLU FFN:
───────────
       ┌──→ Linear(256→341) ──→ Swish ──┐
x ──→──┤                                 ├──→ ⊙ ──→ Linear(341→256) ──→ output
       └──→ Linear(256→341) ─────────────┘
            W_gate            W_up               W_down
```

### Mathematical Comparison

| Aspect | Standard FFN + GELU | SwiGLU |
|--------|---------------------|--------|
| **Formula** | $\text{GELU}(\mathbf{W}_1\mathbf{x})\mathbf{W}_2$ | $\text{Swish}(\mathbf{W}_g\mathbf{x}) \odot (\mathbf{W}_u\mathbf{x})\mathbf{W}_d$ |
| **Activation** | GELU (probabilistic gate) | Swish (smooth gate) |
| **Gating** | Implicit (activation only) | Explicit (two projections) |
| **Parameters** | $2 \times d \times d_{ff}$ | $3 \times d \times d'_{ff}$ |
| **With equivalence** | $2 \times 256 \times 512 = 262k$ | $3 \times 256 \times 341 = 262k$ |

### Why SwiGLU is Better

1. **Richer Feature Interactions**
   - Two separate projections learn different aspects
   - Gating allows selective information flow
   
2. **Better Gradient Flow**
   - Swish is smooth everywhere (unlike ReLU)
   - Gating provides multiplicative gradients

3. **Empirical Results** (from LLaMA, PaLM papers)
   - 1-2% improvement in language modeling perplexity
   - Better transfer learning performance

### Gradient Comparison

**GELU gradient:**
$$\frac{\partial \text{GELU}}{\partial x} = \Phi(x) + x \cdot \phi(x)$$

where $\phi(x)$ is the standard normal PDF.

**Swish gradient:**
$$\frac{\partial \text{Swish}}{\partial x} = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$$

**SwiGLU gradient** (for the gating path):
$$\frac{\partial}{\partial x}[\text{Swish}(g) \cdot u] = \text{Swish}'(g) \cdot u \cdot \frac{\partial g}{\partial x} + \text{Swish}(g) \cdot \frac{\partial u}{\partial x}$$

The multiplicative term in SwiGLU provides **richer gradient signals**.

---

## 6. Summary Table

| Component | Purpose | Key Equation | In This Model |
|-----------|---------|--------------|---------------|
| **LayerNorm** | Normalize activations | $\gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | Pre-norm at each sub-block |
| **Dropout** | Regularization | $x_i \cdot \text{Bernoulli}(1-p) / (1-p)$ | 10% after attention & FFN |
| **GELU** | Smooth activation | $x \cdot \Phi(x)$ | Standard FFN option |
| **SwiGLU** | Gated activation | $\text{Swish}(W_g x) \odot (W_u x)$ | Advanced FFN option |