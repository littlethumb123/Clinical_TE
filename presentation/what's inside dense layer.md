[8 tools called]

Now I have all the details. Here's the complete, in-depth picture of what's inside each temporal layer:

---

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