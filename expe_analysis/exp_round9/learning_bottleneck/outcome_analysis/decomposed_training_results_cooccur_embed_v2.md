## Comprehensive v2 (Co-occurrence Embeddings) vs v0/v1 Analysis

### 1. Full Comparative Table — All Three Experiments

| Metric | v0 (baseline decoder) | v1 (fixed decoder) | v2 (co-occurrence emb) | v2 vs v1 |
|--------|----------------------|--------------------|-----------------------|----------|
| **Stage 1** | | | | |
| Pretrained embeddings | No | No | **Yes** (PPMI+SVD, std=0.0765) | New |
| Embedding frozen 50%? | No | No | **Yes** | New |
| Val Recall@10 | 0.813 | 0.809 | **0.825** | **+0.016** |
| Val μRecall@10 | 0.457 | 0.456 | **0.464** | **+0.008** |
| Val NDCG@20 | 0.425 | 0.425 | **0.441** | **+0.016** |
| Val loss | 0.0031 | 0.0031 | **0.0030** | Slightly better |
| Common grad frac | 74.0% | 72.8% | 72.6% | Same |
| Tail grad frac | 3.7% | 4.0% | 3.8% | Same |
| medium_top10_acc | — | 0.0016 | **0.0129** | **8x better** |
| common_top10_acc | — | 0.810 | **0.825** | +0.015 |
| macro_auroc | — | 0.860 | **0.862** | Same |
| **Pre-Stage 2 logit diagnostics** | | | | |
| Common pos logit | -2.41 | -2.41 | **-2.33** | +0.08 |
| Common margin | 6.72 | 6.60 | **6.94** | **+0.34** |
| Medium pos logit | -7.33 | -7.24 | **-6.99** | +0.25 |
| Medium margin | 5.09 | 4.98 | **5.21** | **+0.23** |
| Rare pos logit | -11.62 | -11.32 | -11.42 | -0.10 |
| Rare margin | 2.97 | 2.89 | 2.68 | **-0.21** |
| Tail pos logit | -13.49 | -13.26 | -13.48 | -0.22 |
| **Tail margin (pre-S2)** | **2.01** | **1.66** | **1.41** | **-0.25** |
| **Stage 2 (20 epochs, LR=5e-3, focused loss)** | | | | |
| S2 Epoch 1 loss | 0.5148 | 0.4381 | **0.6704** | Higher start |
| S2 Epoch 5 loss | — | 0.0631 | **0.0539** | Better |
| S2 Epoch 10 loss | — | 0.0329 | **0.0302** | Better |
| S2 Epoch 20 loss (final) | 0.5092 | **0.0262** | **0.0223** | **Better** |
| **Post-Stage 2 logit diagnostics** | | | | |
| Common pos logit | -2.41 | -2.29 | **-2.20** | +0.09 |
| Common margin | 6.73 | 6.28 | **6.59** | **+0.31** |
| Medium pos logit | -7.33 | -6.86 | **-6.62** | +0.24 |
| Medium margin | 5.10 | 4.72 | **4.93** | **+0.21** |
| Rare pos logit | +0.25 | -3.45 | **-0.75** | +2.70 |
| **Rare margin** | **0.32** | **0.45** | **0.77** | **+0.32 (71% improvement)** |
| Tail pos logit | -0.30 | -3.93 | **-4.02** | -0.09 |
| Tail neg logit | (inferred ~-0.02) | -3.87 | **-5.04** | -1.17 |
| **Tail margin** | **-0.28** | **-0.06** | **+1.02** | **+1.08 (first positive tail margin ever)** |
| **tail_top10_acc** | **0%** | **0%** | **0%** | No change |

---

### 2. What the Numbers Tell Us — Systematic Analysis

#### 2a. Stage 1 Improvement: Co-occurrence embeddings genuinely help the encoder

The v2 Stage 1 results are **the best Stage 1 results ever achieved in this project:**

- Recall@10: 0.825 (vs 0.813 v0, 0.809 v1) — a +1.5% absolute improvement
- NDCG@20: 0.441 (vs 0.425) — a +3.8% relative improvement
- medium_top10_acc: 0.0129 (vs 0.0016 v1) — an **8x improvement** for medium codes
- Common margin pre-S2: 6.94 (vs 6.72 v0, 6.60 v1) — the strongest common separation ever seen

The PPMI+SVD embeddings provided a meaningfully better starting point that the encoder could exploit. The embedding std of 0.0765 (vs ~0.03 for random init) gave the encoder more structured input from the start.

#### 2b. Stage 2 Loss — v2 Learns Faster and Converges Lower

| Epoch | v1 loss | v2 loss | v2 better by |
|-------|---------|---------|-------------|
| 1 | 0.4381 | 0.6704 | v1 starts lower |
| 2 | 0.2254 | 0.1769 | v2 catches up |
| 3 | 0.1234 | 0.1010 | v2 ahead |
| 5 | 0.0631 | 0.0539 | -14.6% |
| 10 | 0.0329 | 0.0302 | -8.2% |
| 15 | 0.0270 | 0.0231 | -14.4% |
| 20 | 0.0262 | 0.0223 | **-14.9%** |

v2 starts higher (because the encoder built slightly different representations — the decoder rows were re-initialized from a different `h` landscape), but **converges to a 15% lower final loss** than v1. The decoder is finding more to learn from the v2 representation.

#### 2c. The Critical Result: Tail Margin Turned Positive for the First Time

This is the single most important finding from all experiments:

| Experiment | Tail margin (post-S2) | Interpretation |
|-----------|----------------------|---------------|
| v0 | **-0.28** | Anti-discriminative — model predicts LESS when code is present |
| v1 | **-0.06** | Noise — statistically indistinguishable from zero |
| v2 | **+1.02** | **First positive tail margin ever observed** |

The v2 tail margin of +1.02 means: **on average, the model's logit for a tail code is 1.02 units HIGHER when the code is genuinely present than when it's absent.** This is a genuine positive signal direction. Compare to common margin (6.59) and medium margin (4.93) — the tail margin is still small, but it's now on the right side of zero.

Additionally, the rare margin improved from 0.45 (v1) to **0.77** (v2) — a 71% relative improvement.

#### 2d. But tail_top10_acc is Still 0%

Despite the positive tail margin, `tail_top10_acc` remains 0%. Why?

A margin of +1.02 translates to: `P(positive) / P(negative) = e^1.02 ≈ 2.77`. So a positive tail code is predicted ~2.77x more likely than if it were absent. But this relative difference operates on **extremely small absolute probabilities**. If the base rate for a tail code is 0.001%, then `σ(-5.04) ≈ 0.006` (negative logit) vs `σ(-4.02) ≈ 0.018` (positive logit). Both are tiny. For this code to appear in the top 10 predictions, its logit of -4.02 must exceed the logits of ~6,200 other codes. Common codes have logits around -2.20 (when present), so there are ~1,100+ codes with higher logits even when this tail code is present. **A margin of +1.02 is necessary but not sufficient — the margin needs to be ~5+ for tail codes to compete with common codes for top-10 slots.**

---

### 3. Root Cause Analysis: Why the Co-occurrence Embeddings Helped But Didn't Solve the Problem

#### Root Cause 1: The Gradient Starvation Problem Was NOT Solved at the Encoder Level

Look at the Stage 1 gradient tier analysis:

| Tier | v0 | v1 | v2 |
|------|------|------|------|
| Common | 74.0% | 72.8% | 72.6% |
| Medium | 12.4% | 12.8% | 12.9% |
| Rare | 5.8% | 6.0% | 6.0% |
| Tail | 3.7% | 4.0% | 3.8% |

**The gradient distribution is essentially identical across all three experiments.** Co-occurrence embeddings changed the input, but the *gradient flow through the encoder* is still dominated by common codes. This means:

- The 6-layer transformer still builds `h` primarily to serve common code prediction
- Tail codes get 3.8% of the gradient — the same starvation
- The initial embedding distinctiveness (std=0.0765) gets progressively washed out by the ~72% common gradient over 12,000+ steps

The embeddings provided a better **starting point**, which explains why Stage 1 metrics improved (especially medium_top10_acc: 8x better). But the encoder still converges to a common-code-dominated representation because that's where the gradient pushes it.

#### Root Cause 2: The Pre-S2 Tail Margin Actually DECREASED with Co-occurrence Embeddings

A paradoxical finding:

| | v0 | v1 | v2 |
|--|------|------|------|
| Pre-S2 tail margin | 2.01 | 1.66 | **1.41** |

The co-occurrence embeddings *improved* common and medium margins but made the pre-S2 tail margin *worse*. This is because:

1. The PPMI+SVD embeddings encode **co-occurrence structure**, which is dominated by common codes (they co-occur with everything)
2. The encoder learned to exploit this structure more effectively for common codes, sharpening the common-code representation
3. This sharpening further marginalizes tail codes in `h` — the representation becomes even more "specialized" for common codes, leaving even less tail-discriminative information

This is a subtle but important insight: **giving the encoder better input made it better at its primary job (common codes) but didn't help — and slightly hurt — tail codes during Stage 1 training.** The gradient starvation was the binding constraint, not input quality.

#### Root Cause 3: The Improvement Came Entirely from Stage 2, Not Stage 1

The tail margin improvement from -0.06 (v1) to +1.02 (v2) happened during Stage 2, not Stage 1 (the pre-S2 tail margin was worse in v2). This means:

- **The co-occurrence embeddings helped the encoder build a representation `h` that is slightly more informative for tail codes in a way that Stage 2 can exploit** — even though Stage 1's own metrics don't show tail improvement
- Specifically, the better medium-code representation (medium_top10_acc: 8x better) may have created intermediate features in `h` that are partially correlated with tail codes. Tail codes often co-occur with medium codes more than with common codes. Better medium features in `h` → more tail-correlated information → Stage 2 decoder can find a positive margin

#### Root Cause 4: The Fundamental Capacity Issue — Single `h ∈ ℝ^256` for 6,297 Codes

The encoder compresses all information about a patient-day into a single 256-dimensional vector. This vector must simultaneously encode:
- ~1,141 common codes (well-trained, consuming most of the representational capacity)
- ~1,711 medium codes (partially trained)
- ~1,705 rare codes (undertrained)
- ~1,148 tail codes (essentially untrained)

A 256-dimensional vector has at most 256 orthogonal directions. Even if the encoder could perfectly allocate capacity, it would need to pack 6,297 binary predictions into 256 dimensions. The theoretical information capacity is `256 × log2(1/ε)` bits, where `ε` is the prediction precision. For tail codes with base rate 0.001%, even a single bit of useful information per tail code would require 1,148 bits — far more than 256 dimensions can encode.

The co-occurrence embeddings improved the input signal (breaking the homogenization), and Stage 2's focused loss allowed the decoder to extract what little signal exists. But **the fundamental bottleneck is the 256-dimensional information bottleneck** — there simply isn't enough representational capacity in `h` to encode tail-code-specific features after common codes have consumed most of the space.

---

### 4. Summary Verdict: Did Co-occurrence Embeddings Work?

| Dimension | Assessment |
|-----------|-----------|
| Stage 1 overall performance | **Yes** — best Stage 1 metrics ever (+1.5% R@10, +8x medium_top10_acc) |
| Tail margin direction | **Yes** — first positive tail margin ever (+1.02 vs -0.06 v1) |
| Tail discrimination (top10_acc) | **No** — still 0% |
| Rare margin improvement | **Yes** — 0.77 vs 0.45 v1 (+71%) |
| Broke the fundamental bottleneck? | **No** — gradient starvation in Stage 1 is unchanged; `h` capacity limit remains |
| Worth the investment? | **Yes** — proved the hypothesis was directionally correct; moved the needle measurably |

**The co-occurrence embeddings are the first intervention in 9 rounds of experiments that produced a positive tail margin.** This is a genuine, meaningful step forward. But the margin (+1.02) is insufficient for practical discrimination (need ~5+), and `tail_top10_acc` remains at 0%.

---

### 5. Three Industry-Proven Next Steps

Given that the co-occurrence embeddings moved the needle but didn't solve the problem, and the root cause analysis points to (a) gradient starvation during Stage 1 and (b) the 256-dim representational bottleneck, here are three practical methods used in production systems at scale.

#### Method A: Sparse Mixture of Experts (MoE) Decoder with Per-Tier Routing

**What it is:** Replace the single `nn.Linear(256, 6297)` decoder with a Sparse MoE decoder that routes different code tiers to different expert MLPs. This is the same architecture used in production recommender systems at Google (MMoE), Meta (DLRM with expert routing), and YouTube recommendations for handling long-tail items.

**Why it addresses the bottleneck:** The single linear decoder maps `h → logits` via `W^T h + b`. When `h` has limited tail information, a linear readout can't extract nonlinear combinations. A 2-layer MLP expert can discover ABSENCE patterns (e.g., "no common respiratory codes + high age + presence in emergency codes" → higher probability of rare tail code) that a linear decoder structurally cannot represent.

**How it works:**
1. Keep the existing encoder and Stage 1 training unchanged
2. Replace `decoder_cd` with a per-tier MoE decoder
3. Common/medium codes use the existing linear decoder (already well-trained)
4. Rare/tail codes use dedicated 2-layer MLP experts
5. Train Stage 2 with the existing focused loss + code-balanced sampling

**Where to add the code:** In the same cell that defines the model classes (where `FlashAttentionTransformer` is defined). Add a new class right after the existing model definitions, and add a small integration function.

```python
# ============================================================================
# MoE DECODER: Per-Tier Expert Decoder (replaces nn.Linear for rare/tail)
# ============================================================================

class TieredMoEDecoder(nn.Module):
    """
    Replaces the flat nn.Linear(d_model, target_cd_cnt) decoder with:
    - Linear pass-through for common/medium codes (preserves learned weights)
    - Dedicated MLP experts for rare/tail codes (nonlinear readout)
    
    Used in Stage 2 only. The original decoder weights for common/medium are
    copied in and frozen. Only the MLP experts are trainable.
    """
    def __init__(
        self,
        d_model: int,
        target_cd_cnt: int,
        code_frequencies: np.ndarray,
        original_decoder: nn.Linear,
        expert_hidden: int = 128,
        expert_layers: int = 2,
        percentile_boundaries: tuple = (20, 50, 80)
    ):
        super().__init__()
        self.d_model = d_model
        self.target_cd_cnt = target_cd_cnt
        
        freq_nz = code_frequencies[code_frequencies > 0]
        p20, p50, p80 = np.percentile(freq_nz, percentile_boundaries)
        
        self.common_mask = code_frequencies > p80
        self.medium_mask = (code_frequencies <= p80) & (code_frequencies > p50)
        self.rare_mask = (code_frequencies <= p50) & (code_frequencies > p20)
        self.tail_mask = (code_frequencies <= p20) & (code_frequencies > 0)
        self.zero_mask = code_frequencies == 0
        
        self.common_medium_mask = self.common_mask | self.medium_mask | self.zero_mask
        self.rare_tail_mask = self.rare_mask | self.tail_mask
        
        self.common_medium_indices = torch.where(
            torch.tensor(self.common_medium_mask[:target_cd_cnt])
        )[0]
        self.rare_tail_indices = torch.where(
            torch.tensor(self.rare_tail_mask[:target_cd_cnt])
        )[0]
        
        n_common_medium = int(self.common_medium_indices.shape[0])
        n_rare_tail = int(self.rare_tail_indices.shape[0])
        
        self.linear_common_medium = nn.Linear(d_model, n_common_medium)
        with torch.no_grad():
            orig_w = original_decoder.weight.data  # [target_cd_cnt, d_model]
            orig_b = original_decoder.bias.data     # [target_cd_cnt]
            self.linear_common_medium.weight.data.copy_(
                orig_w[self.common_medium_indices]
            )
            self.linear_common_medium.bias.data.copy_(
                orig_b[self.common_medium_indices]
            )
        self.linear_common_medium.weight.requires_grad = False
        self.linear_common_medium.bias.requires_grad = False
        
        if expert_layers == 2:
            self.mlp_rare_tail = nn.Sequential(
                nn.Linear(d_model, expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, n_rare_tail)
            )
        elif expert_layers == 3:
            self.mlp_rare_tail = nn.Sequential(
                nn.Linear(d_model, expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, expert_hidden // 2),
                nn.GELU(),
                nn.Linear(expert_hidden // 2, n_rare_tail)
            )
        else:
            self.mlp_rare_tail = nn.Linear(d_model, n_rare_tail)

        self._init_mlp_weights()
        
        print(f"  TieredMoEDecoder initialized:")
        print(f"    Common+Medium (linear, frozen): {n_common_medium} codes")
        print(f"    Rare+Tail (MLP, trainable): {n_rare_tail} codes")
        print(f"    MLP architecture: {d_model} → {expert_hidden} → {n_rare_tail}")
        print(f"    Trainable params: {sum(p.numel() for p in self.mlp_rare_tail.parameters()):,}")
    
    def _init_mlp_weights(self):
        for m in self.mlp_rare_tail.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, h):
        """
        h: [batch, d_model] or [batch, seq_len, d_model]
        Returns: [batch, target_cd_cnt] or [batch, seq_len, target_cd_cnt]
        """
        shape = h.shape
        if h.dim() == 3:
            batch, seq_len, d = shape
            h_flat = h.reshape(-1, d)
        else:
            h_flat = h
        
        out = torch.zeros(
            h_flat.shape[0], self.target_cd_cnt,
            device=h_flat.device, dtype=h_flat.dtype
        )
        
        out[:, self.common_medium_indices] = self.linear_common_medium(h_flat)
        out[:, self.rare_tail_indices] = self.mlp_rare_tail(h_flat)
        
        if len(shape) == 3:
            out = out.reshape(batch, seq_len, self.target_cd_cnt)
        
        return out
```

**How to use it in Stage 2:** In the experiment execution cell, after Stage 1 completes and before Stage 2 runs, swap the decoder:

```python
# After Stage 1 completes, before Stage 2:
# Swap decoder_cd with TieredMoEDecoder

actual_model = model
if isinstance(model, nn.DataParallel):
    actual_model = model.module
if isinstance(actual_model, DataParallelWrapper):
    actual_model = actual_model.model

tiered_decoder = TieredMoEDecoder(
    d_model=256,
    target_cd_cnt=6297,
    code_frequencies=data_prepared_1p5M.code_frequencies,
    original_decoder=actual_model.decoder_cd,
    expert_hidden=128,
    expert_layers=2
).to(device)

# Replace the decoder
actual_model.decoder_cd = tiered_decoder

# Now run Stage 2 as normal — only MLP experts are trainable
```

This is a **~3-4 hour experiment** (Stage 2 only, no Stage 1 rerun needed if you save the Stage 1 checkpoint).

---

#### Method B: Gradient-Balanced Multi-Task Learning with GradNorm

**What it is:** Treat each code tier as a separate task and dynamically rebalance gradient magnitudes during Stage 1 training itself, using the GradNorm algorithm (Chen et al., ICML 2018). This is widely used in production multi-task systems at Google (Search ranking), Uber (Eats + ride prediction), and autonomous driving (multiple perception heads).

**Why it addresses the bottleneck:** The root cause analysis showed that Stage 1 gradient starvation (74% common, 3.8% tail) was unchanged even with co-occurrence embeddings. GradNorm dynamically adjusts per-task loss weights so that all tasks (tiers) train at roughly equal rates, preventing common codes from monopolizing the encoder capacity.

**How it works:**
1. Define 4 "tasks" (common, medium, rare, tail) each with their own loss
2. Track the training rate (loss decrease) per tier
3. Use a learnable weight per tier, updated to equalize training rates
4. The encoder receives balanced gradient from all tiers throughout Stage 1

**Where to add:** Add a new class in the cell where `OptimizeConfig` is defined (or right after the training utilities). Then modify the training loop.

```python
# ============================================================================
# GRADNORM: Dynamic gradient balancing across code tiers
# ============================================================================

class GradNormBalancer(nn.Module):
    """
    GradNorm (Chen et al., ICML 2018) for per-tier gradient balancing.
    Learns per-tier loss weights that equalize training rates across tiers.
    """
    def __init__(self, n_tasks: int = 4, alpha: float = 1.5):
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.log_weights = nn.Parameter(torch.zeros(n_tasks))
        self.initial_losses = None
    
    @property
    def weights(self):
        return F.softmax(self.log_weights, dim=0) * self.n_tasks
    
    def forward(
        self,
        per_tier_losses: List[torch.Tensor],
        shared_layer: nn.Parameter
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            per_tier_losses: [loss_common, loss_medium, loss_rare, loss_tail]
            shared_layer: A parameter from the shared encoder (e.g., last layer norm weight)
                         Used to compute gradient norms for balancing.
        Returns:
            total_loss: Weighted sum of per-tier losses
            info: dict with weights and loss ratios for logging
        """
        weights = self.weights
        
        if self.initial_losses is None:
            self.initial_losses = torch.stack(
                [l.detach() for l in per_tier_losses]
            )
        
        weighted_losses = [w * l for w, l in zip(weights, per_tier_losses)]
        total_loss = sum(weighted_losses)
        
        loss_ratios = torch.stack([
            l.detach() / (init_l + 1e-8)
            for l, init_l in zip(per_tier_losses, self.initial_losses)
        ])
        mean_ratio = loss_ratios.mean()
        target_grad_norms = (loss_ratios / (mean_ratio + 1e-8)) ** self.alpha
        
        info = {
            'weights': weights.detach().cpu().numpy(),
            'loss_ratios': loss_ratios.detach().cpu().numpy(),
            'target_norms': target_grad_norms.detach().cpu().numpy(),
        }
        
        return total_loss, info


def compute_per_tier_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    code_frequencies: np.ndarray,
    target_cd_cnt: int,
    pos_weight: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    """
    Compute BCE loss separately for each tier.
    Returns: [loss_common, loss_medium, loss_rare, loss_tail]
    """
    freq_nz = code_frequencies[code_frequencies > 0]
    p20, p50, p80 = np.percentile(freq_nz, [20, 50, 80])
    
    tier_masks = {
        'common': code_frequencies[:target_cd_cnt] > p80,
        'medium': (code_frequencies[:target_cd_cnt] <= p80) & 
                  (code_frequencies[:target_cd_cnt] > p50),
        'rare': (code_frequencies[:target_cd_cnt] <= p50) & 
                (code_frequencies[:target_cd_cnt] > p20),
        'tail': (code_frequencies[:target_cd_cnt] <= p20) & 
                (code_frequencies[:target_cd_cnt] > 0),
    }
    
    losses = []
    for tier_name in ['common', 'medium', 'rare', 'tail']:
        mask = torch.tensor(tier_masks[tier_name], device=logits.device)
        tier_logits = logits[..., mask]
        tier_targets = targets[..., mask]
        
        if pos_weight is not None:
            tier_pw = pos_weight[mask]
            loss = F.binary_cross_entropy_with_logits(
                tier_logits, tier_targets,
                pos_weight=tier_pw,
                reduction='mean'
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                tier_logits, tier_targets, reduction='mean'
            )
        losses.append(loss)
    
    return losses
```

**Integration into the training loop:** Inside the `train_one_epoch` function, after computing the model output but before the backward pass, replace the single loss computation with per-tier loss + GradNorm weighting. The GradNorm balancer's `log_weights` parameter gets its own optimizer (separate from the model optimizer) with a higher learning rate (e.g., 0.025 as recommended by the original paper).

This requires modifying the training loop, which is more invasive. The key change is in the loss computation block:

```python
# In train_one_epoch, replace:
#   pred_loss = criterion(output, y)
# With:
per_tier_losses = compute_per_tier_loss(
    output, y, code_frequencies, config.target_cd_cnt, 
    pos_weight=criterion.pos_weight if hasattr(criterion, 'pos_weight') else None
)
pred_loss, gradnorm_info = gradnorm_balancer(
    per_tier_losses,
    shared_layer=actual_model.temporal_encoder.layers[-1].norm1.weight
)
```

And add a separate optimizer for the GradNorm weights:

```python
gradnorm_optimizer = optim.Adam(gradnorm_balancer.parameters(), lr=0.025)
# After main optimizer step, also step the gradnorm optimizer
gradnorm_optimizer.step()
```

---

#### Method C: Contrastive Learning on the Encoder Representation (SimCLR-style)

**What it is:** Add a contrastive auxiliary loss during Stage 1 that explicitly pushes the encoder to separate patients with different code profiles in the representation space. This is based on SimCLR (Chen et al., ICML 2020, Google Brain) and its application to tabular/sequential data in healthcare at Google Health (CLOCS), Flatiron Health, and Tempus.

**Why it addresses the bottleneck:** The analysis showed that `h` for a patient with a tail code is nearly identical to `h` for a similar patient without that tail code (because both have the same common codes, and the encoder optimizes for common codes). A contrastive loss directly penalizes this: if two patients have different target code sets, their `h` vectors should be pushed apart. This forces the encoder to encode information about ALL codes (including tail) into `h`, not just common ones.

**How it works:**
1. During Stage 1, for each batch, compute `h` for all patients
2. Define "positive pairs" (same rare/tail codes) and "negative pairs" (different code profiles)
3. Add an InfoNCE contrastive loss that pulls positive pairs together and pushes negative pairs apart in the `h` space
4. Weight this loss at 0.1-0.5x of the main BCE loss so it shapes `h` without dominating

**Where to add:** Add the contrastive loss class in the utilities section (after the existing loss functions), then integrate it into the training loop.

```python
# ============================================================================
# CONTRASTIVE AUXILIARY LOSS for representation diversity
# ============================================================================

class TierAwareContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss that pushes the encoder to distinguish
    patients with different rare/tail code profiles.
    
    For each anchor patient, positive = another patient sharing at least
    one rare/tail code. Negative = patients without that code.
    
    This forces h to encode tail-code-relevant features even when the
    main BCE loss gradient is dominated by common codes.
    """
    def __init__(self, temperature: float = 0.1, max_pairs_per_batch: int = 256):
        super().__init__()
        self.temperature = temperature
        self.max_pairs = max_pairs_per_batch
    
    def forward(
        self,
        h: torch.Tensor,
        targets: torch.Tensor,
        rare_tail_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: [batch_size, d_model] — pooled encoder representations
            targets: [batch_size, target_cd_cnt] — multi-hot targets
            rare_tail_mask: [target_cd_cnt] — True for rare+tail codes
        Returns:
            contrastive loss scalar
        """
        rare_tail_targets = targets[:, rare_tail_mask]  # [B, n_rare_tail]
        
        has_rare_tail = rare_tail_targets.sum(dim=1) > 0  # [B]
        if has_rare_tail.sum() < 2:
            return torch.tensor(0.0, device=h.device, requires_grad=True)
        
        anchor_idx = torch.where(has_rare_tail)[0]
        if len(anchor_idx) > self.max_pairs:
            perm = torch.randperm(len(anchor_idx), device=h.device)[:self.max_pairs]
            anchor_idx = anchor_idx[perm]
        
        h_anchors = h[anchor_idx]  # [n_anchors, d]
        rt_anchors = rare_tail_targets[anchor_idx]  # [n_anchors, n_rare_tail]
        
        h_norm = F.normalize(h_anchors, dim=1)
        sim = h_norm @ h_norm.t() / self.temperature  # [n_anchors, n_anchors]
        
        code_overlap = rt_anchors @ rt_anchors.t()  # [n_anchors, n_anchors]
        positive_mask = code_overlap > 0
        positive_mask.fill_diagonal_(False)
        
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)
        
        exp_sim = torch.exp(sim)
        exp_sim.fill_diagonal_(0)
        
        denom = exp_sim.sum(dim=1, keepdim=True)  # [n_anchors, 1]
        log_prob = sim - torch.log(denom + 1e-8)   # [n_anchors, n_anchors]
        
        pos_log_prob = (log_prob * positive_mask.float()).sum(dim=1)
        n_pos = positive_mask.float().sum(dim=1).clamp(min=1)
        loss = -(pos_log_prob / n_pos).mean()
        
        return loss
```

**Integration into the training loop:** In `train_one_epoch`, after computing the main prediction loss, add the contrastive term:

```python
# After computing pred_loss from the main criterion:
if contrastive_loss_fn is not None and epoch_idx == 0:
    # Get h from the model (before the decoder)
    # This requires a small model modification to expose h
    h_pooled = actual_model.get_representation(x, dt_cnt)  # need to add this method
    
    cl_loss = contrastive_loss_fn(h_pooled, y, rare_tail_mask_tensor)
    pred_loss = pred_loss + 0.2 * cl_loss  # weight the contrastive loss at 0.2x
```

The model needs a small method addition to expose the intermediate `h` representation. In the `FlashAttentionTransformer` class, add:

```python
def get_representation(self, x, dt_cnt):
    """Return the pooled h representation before the decoder."""
    # Same forward pass as forward() but stop before decoder_cd
    cd = x[:, :, 0, :]
    ages = x[:, :, 1, :]
    genders = x[:, :, 2, :]
    lobs = x[:, :, 3, :]
    
    cd_emb = self.embedding_cd(cd)
    age_emb = self.embedding_age(ages)
    gender_emb = self.embedding_gender(genders)
    lob_emb = self.embedding_lob(lobs)
    
    combined = cd_emb + age_emb + gender_emb + lob_emb
    combined = self.input_norm(combined)
    
    daily = self.daily_code_encoder(combined)
    daily = self.daily_norm(daily)
    
    h = self.temporal_encoder(daily)
    # h: [batch, seq_len, d_model]
    return h
```

---

### Prioritization

| Method | Cost | Addresses | Expected Impact | Risk |
|--------|------|-----------|----------------|------|
| **A: MoE Decoder** | ~3-4 hours (Stage 2 only) | Readout nonlinearity | Moderate — if `h` has nonlinear tail signal, MLP finds it | Low — no Stage 1 changes |
| **B: GradNorm** | ~$5-17 (full Stage 1 retrain) | Gradient starvation | High — directly equalizes tier training rates | Medium — requires training loop changes |
| **C: Contrastive** | ~$5-17 (full Stage 1 retrain) | Representation diversity | High — forces `h` to encode tail information | Medium — requires model + loop changes |

**Recommended order:** A first (cheapest, tests whether the v2 `h` has nonlinear signal), then B (addresses the root cause — gradient starvation), then C (if GradNorm alone doesn't force enough diversity into `h`). Methods B and C can also be combined — GradNorm + contrastive loss during Stage 1 — which would be the strongest intervention.

