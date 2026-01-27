Thank you for the important corrections and the reframing. You're absolutely right - I was getting lost in the weeds of small metric differences and focused too much on "punitive" redistribution rather than **constructive improvement**.

Let me reframe the entire approach around your insight:

---

# Reframed Goal: Improve Overall Model Quality Through Better Rare/Tail Representation

## The Core Insight

The goal is NOT:
- ❌ Punish common codes to give gradient budget to rare codes
- ❌ Redistribute learning at the expense of common code performance
- ❌ Trade off head performance for tail performance

The goal IS:
- ✅ **Improve the quality of embeddings for ALL codes**
- ✅ **Enable the model to learn rare/tail code representations WITHOUT degrading common**
- ✅ **Increase the model's overall representational capacity and discriminative power**

---

## Mechanistic Understanding (Focused on Root Cause)

**The root cause is NOT that common codes receive "too much" gradient.**

**The root cause is that rare/tail codes receive INSUFFICIENT and INCONSISTENT learning signal to develop meaningful representations.**

This is a subtle but crucial distinction:
- Common codes work fine because they receive consistent, frequent signal
- Rare codes fail because their signal is sporadic and averaged into noise
- The solution is to **ADD signal for rare codes**, not **REMOVE signal from common codes**

---

## Solution Approaches Framed as "Encouragement"

### Approach 1: Auxiliary Representation Learning (Add Signal, Don't Redistribute)

**Concept**: Add a secondary learning objective that forces the model to learn meaningful embeddings for ALL codes, including rare/tail.

**Mechanism**:
- Add a **code embedding regularization loss** that encourages:
  1. Codes with similar clinical meaning to be nearby in embedding space
  2. Codes with different clinical meaning to be separated
  3. ALL codes (including rare) to have distinct, non-trivial embeddings

**Implementation Idea**:
```python
# In addition to BCE loss, add embedding quality loss
def embedding_regularization_loss(code_embeddings, tier_indices):
    """Encourage all code embeddings to be well-distributed in space"""
    
    # 1. Prevent embedding collapse (rare codes shouldn't be near zero)
    embedding_norms = code_embeddings.norm(dim=1)
    min_norm_loss = F.relu(1.0 - embedding_norms).mean()  # Encourage norm > 1
    
    # 2. Encourage diversity within each tier
    diversity_loss = 0
    for tier, indices in tier_indices.items():
        tier_embeddings = code_embeddings[indices]
        # Encourage embeddings to spread out (not collapse to same vector)
        pairwise_dist = torch.cdist(tier_embeddings, tier_embeddings)
        diversity_loss += F.relu(0.5 - pairwise_dist).mean()  # Penalize if too close
    
    return min_norm_loss + 0.1 * diversity_loss
```

**Why This is "Encouraging"**: This adds learning signal to ALL codes equally. It doesn't take away from common codes; it gives rare codes an additional reason to develop meaningful representations.

---

### Approach 2: Hierarchical Code Supervision (Leverage ICD Structure)

**Concept**: ICD codes have hierarchical structure (CCS categories, CCSR groups). Use this structure to provide supervision signal for rare codes.

**Mechanism**:
- Rare code X belongs to CCS category Y
- Common codes A, B, C also belong to CCS category Y
- The model should learn that X is "similar" to A, B, C even if X rarely appears

**Implementation Idea**:
```python
# Add category-level prediction as auxiliary task
# For each patient: predict CCS categories (coarser granularity)
# This gives signal to rare codes through their parent category

def hierarchical_loss(code_logits, code_targets, code_to_category_map):
    """Add category-level supervision"""
    # Aggregate code logits to category level (max-pooling within category)
    category_logits = aggregate_to_category(code_logits, code_to_category_map)
    category_targets = aggregate_to_category(code_targets, code_to_category_map)
    
    return F.binary_cross_entropy_with_logits(category_logits, category_targets)
```

**Why This is "Encouraging"**: Rare codes that belong to common categories receive "transferred" supervision from their category membership. The category-level signal flows back to improve rare code embeddings.

---

### Approach 3: Contrastive Pre-training for Code Embeddings

**Concept**: Before the main prediction task, pretrain the code embeddings using contrastive learning to ensure all codes (including rare) have distinctive, meaningful representations.

**Mechanism**:
- Create positive pairs: (code A, code B) that co-occur in same patients
- Create negative pairs: (code A, code C) that never co-occur
- Train embeddings to pull positives together, push negatives apart
- This gives EVERY code (including rare) learning signal

**Why This is "Encouraging"**: This gives rare codes learning signal from their co-occurrence patterns, not just from direct prediction. A rare code that co-occurs with common codes learns a meaningful embedding through association.

---

### Approach 4: Two-Stage Training with Capacity Expansion (Not Punishment)

**Concept**: Instead of freezing common codes (punitive), ADD dedicated capacity for rare/tail codes in the second stage.

**Mechanism**:
- **Stage 1**: Train full model on all data normally. Common codes learn well.
- **Stage 2**: Add a **separate rare/tail-specific module** (e.g., additional attention heads or MoE experts) that specializes in rare codes. Common code pathways remain intact.

**Implementation Idea**:
```python
class TierAwareDecoder(nn.Module):
    def __init__(self, d_model, num_codes, tier_indices):
        self.common_decoder = nn.Linear(d_model, len(tier_indices['common']))
        self.medium_decoder = nn.Linear(d_model, len(tier_indices['medium']))
        self.rare_decoder = nn.Linear(d_model, len(tier_indices['rare']))
        self.tail_decoder = nn.Linear(d_model, len(tier_indices['tail']))
        
    def forward(self, x, stage='full'):
        if stage == 'stage1':
            # Only train common/medium decoders
            return self.common_decoder(x), self.medium_decoder(x)
        else:
            # Stage 2: Add rare/tail decoders with potentially higher LR
            return (self.common_decoder(x), self.medium_decoder(x),
                    self.rare_decoder(x), self.tail_decoder(x))
```

**Why This is "Encouraging"**: Common codes keep their learned representations. Rare codes get dedicated capacity and learning rate, allowing them to develop without competing for the same parameters.

---

### Approach 5: Balanced Gradient Accumulation (Not Normalization)

**Concept**: Instead of normalizing gradients (which can destabilize training), use **gradient accumulation with tier-aware weighting**.

**Mechanism**:
- Accumulate gradients over multiple batches before stepping
- During accumulation, ensure that the total gradient contribution from each tier is approximately equal
- This is different from normalization: we're ADDING more batches until rare codes contribute enough, not SCALING DOWN common codes

**Implementation Idea**:
```python
# Instead of normalizing, accumulate until balanced
tier_grad_accum = {'common': 0, 'medium': 0, 'rare': 0, 'tail': 0}
target_contribution = 0.25

while min(tier_grad_accum.values()) < target_contribution:
    batch = sample_batch_with_rare_preference()  # Oversample rare/tail
    loss = compute_loss(batch)
    loss.backward()
    
    # Track tier contributions
    for tier, indices in tier_indices.items():
        tier_grad_accum[tier] += compute_tier_grad_norm(model, indices)

# Now step with balanced accumulated gradients
optimizer.step()
```

**Why This is "Encouraging"**: We're ADDING more rare code signal rather than REDUCING common code signal. The optimizer step happens when all tiers have contributed meaningfully.

---

## Recommended Action Plan (Prioritized by Constructiveness)

### Phase 1: Diagnostic (Before Any Intervention)

**1.1 Per-Code Logit Analysis** (30 min)
- Check if rare code embeddings are near-zero (collapsed) or just weak
- If collapsed: Need embedding regularization (Approach 1)
- If weak but non-zero: Need more signal (Approaches 2, 3, 4)

### Phase 2: Low-Risk Constructive Interventions

**2.1 Add Embedding Regularization Loss** (Approach 1)
- Feasibility: ★★★★★ (just add a loss term)
- Risk: Low (additive, doesn't change main training)
- Expected benefit: Ensures all codes have non-trivial embeddings

**2.2 Add Hierarchical Supervision** (Approach 2)
- Feasibility: ★★★★☆ (need CCS mapping, which you likely have)
- Risk: Low (additive auxiliary task)
- Expected benefit: Rare codes get signal through category membership

### Phase 3: Moderate-Effort Constructive Interventions

**3.1 Tier-Aware Batching WITH Higher LR for Rare** (Modified Approach 4)
- Instead of just guaranteeing rare codes appear, also apply slightly higher LR to rare/tail decoder weights
- This ADDS learning speed for rare codes without reducing common

**3.2 Contrastive Pre-training Phase** (Approach 3)
- Before main training, learn code embeddings from co-occurrence
- Then continue with supervised training
- Rare codes start with meaningful embeddings

---

## The Key Mental Model Shift

| Old Framing | New Framing |
|-------------|-------------|
| Common codes get too much gradient | Rare codes don't get enough signal |
| Normalize gradients to redistribute | Add auxiliary signal for rare codes |
| Freeze common codes | Add capacity for rare codes |
| Trade off head for tail | Improve tail WITHOUT hurting head |
| Zero-sum competition | Positive-sum improvement |

---

## Concrete Next Steps

1. **Logit analysis** - Understand if rare code embeddings are collapsed or just weak
2. **Implement embedding regularization** - Ensure all codes have distinctive embeddings
3. **Add hierarchical loss** - Give rare codes category-level supervision
4. **Tier-aware batching + learning rate scheduling** - More signal + faster learning for rare codes

Would you like me to elaborate on any of these constructive approaches or help design the implementation for the most promising one?