# Asymmetric focal loss and dense sampler batching

## Metadata

- **Conversation ID:** 968ae281-c4ba-4aeb-9be1-a47e4223ff6c
- **Mode:** chat
- **Created:** 2026-03-04 22:44:31
- **Last Updated:** 2026-03-05 07:44:22
- **Branch:** main
- **Files Changed:** 0
- **Lines Added:** 0
- **Lines Removed:** 0
- **Summary:** Read exp_round5_exp2_lr_plateau_asymmetric_focalloss_results_expert_panel_feb22.md, moe_flashattn_4.py, exp_round5_exp2_lr_plateau_evidence_sythesis_expert_interpret_nextstep_feb2.md, is_focal_loss_necessary_which_variant_to_use.md

---

## Conversation

### 🧑 User

Explain in detail, how the asymmetric focal loss and dense sampler batching actually works and why it is chosen to resolve the code underrepresentative issues? explian in technical deteails but also provide a plain lanaguge version of explanations.

---

### 🤖 Assistant

Searching the codebase for asymmetric focal loss and dense-sampler batching and how they address code underrepresentation.

---

### 🤖 Assistant

Fetching the sampler's batch-construction logic and any density-based variant.

---

### 🤖 Assistant

Reading the full `DensityTierAwareBatchSampler` implementation:

---

### 🤖 Assistant

Checking how the two samplers are selected in the training config:

---

### 🤖 Assistant

Here’s a concise technical and plain-language explanation of how asymmetric focal loss and dense (density-based) sampler batching work and why they’re used for code underrepresentation.

---

## 1. Asymmetric Focal Loss (ASL)

### What problem it addresses

- You have **extreme multi-label** (e.g. ~75K codes per day, only ~80 positives → ~0.1% positive rate).
- **Common codes** are learned quickly; their predictions become “easy” (high confidence).
- Under **standard BCE**, gradient for a negative (y=0) is ∝ σ(z). So every “easy” negative (σ(z) ≈ 0.01) still contributes a small gradient. With millions of such (sample, class) pairs, their **total** gradient dominates.
- **Rare/tail positives** get a tiny share of the total gradient → their embeddings get “drowned out” and barely move (e.g. tail logits stuck around -14.69).

So the issue is: **too much gradient from easy negatives**, not just “not enough positive weight.”

### How ASL works (formulas)

From `AsymmetricLoss` in `dev/moe/moe_flashattn_4.py`:

- **Positives (y=1):**  
  \(L_+ = (1-p)^{\gamma_+} \times \text{BCE}(p, 1)\)  
  With **γ₊ = 0**: weight is 1 → **no down-weighting**; every positive keeps full BCE gradient.

- **Negatives (y=0):**  
  Use shifted probability \(p_m = \max(p - m,\, 0)\) (margin/clip), then  
  \(L_- = (p_m)^{\gamma_-} \times \text{BCE}(p_m, 0)\).  
  With **γ₋ = 4** and **clip m = 0.05**:
  - Easy negatives (p &lt; 0.05): \(p_m = 0\) → **no gradient**.
  - Harder negatives (e.g. p = 0.1): weight ∝ \(0.05^4\) → very small gradient.

So:

- **γ₊ = 0**: Preserve **all** positive gradient (critical for rare/tail codes where every positive counts).
- **γ₋ = 4**: Strongly down-weight easy negatives (p^4 makes them negligible).
- **clip = 0.05**: Hard floor: negatives with p &lt; 0.05 contribute **zero** loss/gradient.

Implementation-wise (same file): BCE is computed per element; for negatives `p_neg = (p - clip).clamp(min=0)`; modulation is `targets * (1-p)^γ₊ + (1-targets) * p_neg^γ₋`; loss = modulation × BCE, then optional `pos_weight` and reduction.

### Why ASL is chosen for underrepresentation

- **pos_weight** only scales the **positive** term. It does nothing for the 99.99% of (sample, code) pairs that are negative; the **sum** of small negative gradients still dominates. So pos_weight alone cannot fix “gradient drowning.”
- **Standard focal loss** (one γ, often with α=0.25) down-weights **all** positives by α and uses the same γ for pos/neg. With 0.1% positive rate, the aggregate negative gradient still wins; you also don’t want to shrink the few tail positive gradients.
- **ASL** is chosen because it:
  - Keeps **full** gradient for every positive (γ₊=0, no α on positives).
  - **Crushes** easy-negative gradient (γ₋=4, clip) so the total gradient is no longer dominated by “already correct” negatives.

So it directly targets the **per-sample, per-class** imbalance: reduce easy-negative gradient, preserve rare-positive gradient.

### Plain-language summary (ASL)

- **Problem:** The model spends almost all gradient on “obvious non-codes” and almost none on the rare codes we care about.
- **ASL:** “Ignore predictions the model already has right (p &lt; 0.05 for negatives), and don’t shrink the learning signal for the rare codes that are actually present.” So it rebalances **how** each (sample, code) contributes to the loss so that rare positives aren’t drowned out by mass of easy negatives.

---

## 2. Dense (density-based) sampler batching

### What problem it addresses

- Training unit is the **member** (full timeline). You can’t sample at “code occurrence” level without breaking the sequence model.
- **Member-level “presence”** (has ≥1 tail code) is weak: ~83.4% of members have at least one tail code, but **only ~5.2% of all code occurrences** are tail. So with presence-only sampling, most batches still have only ~5% tail in the target matrix → tail gradient is a tiny fraction of the batch gradient every step (e.g. tail gradient share collapses to ~0.1%).
- So the issue is **batch composition**: tail codes are **underrepresented in each batch**, so no amount of loss reweighting (ASL, pos_weight) can give tail codes a big enough share of the **batch** gradient.

### Two samplers in the codebase

1. **TierAwareBatchSampler** (presence-based)  
   - Tier = code frequency (e.g. percentiles: tail &lt; 20th, rare 20–50th, medium 50–80th, common &gt; 80th).  
   - Pools: “samples with ≥1 medium code”, “≥1 rare code”, “≥1 tail code”.  
   - Each batch: fill **medium_quota**, **rare_quota**, **tail_quota** from those pools (no duplicate sample in one epoch), then fill the rest from a general pool.  
   - Problem: tail pool is 83.4% of members → random within pool still gives ~5.2% tail **occurrence** rate in the batch.

2. **DensityTierAwareBatchSampler** (“dense” sampler)  
   - For each member, compute **density** per tier, e.g.  
     tail_density = (number of tail code occurrences in that member) / (total code occurrences for that member).  
   - Build **high-density** pools using percentiles, e.g. tail pool = members in the **top 20%** of tail density (among those with at least one tail code) → `density_tail_percentile=80`.  
   - Same quota structure: each batch gets **tail_quota** (and rare/medium quotas) from these **density** pools, rest from general.  
   - Effect: members in the tail quota have **high concentration** of tail codes (e.g. 15–20% of their codes are tail vs 5.2% globally), so in those slots the **occurrence** share of tail in the batch is much higher.

So “dense” = **density-based** tier-aware batching: bias batches toward members with a **high fraction** of rare/tail codes, not just “has at least one.”

### Why density batching is chosen for underrepresentation

- **Evidence (from your analyses):** With only loss changes (BCE, pos_weight, ASL), tail gradient fraction and tail accuracy stay ~0%; gradient concentration is **invariant** to the loss. So the bottleneck is **how often tail codes appear in each batch**, not how they’re weighted inside the batch.
- **Density batching** increases the **tail occurrence rate within each batch** (e.g. from ~5.2% to something like 15–20% in the quota part of the batch), so the **batch-level** tail gradient can compete with the common-code gradient instead of being averaged down to ~0.1%.

### Plain-language summary (dense sampler)

- **Problem:** Rare codes show up in only a small fraction of all code slots. So in a typical batch, almost all “code slots” are common → the model mostly gets gradient from common codes and barely sees rare ones.
- **Dense batching:** “In every batch, reserve a fixed number of slots for patients whose history is **rich** in rare/tail codes (not just ‘has at least one’).” So you **force** each batch to contain enough high-tail-density members that the rare codes actually get a meaningful share of the gradient in that step.

---

## 3. Why both together for code underrepresentation

- **ASL** fixes the **per-sample, per-class** imbalance: it reduces gradient from easy negatives and keeps full gradient for positives. In your experiments this improved **ranking and calibration** (e.g. recall@1, NDCG, Brier) but **did not** increase tail gradient share or tail accuracy. So ASL addresses “how we weigh what’s in the batch.”
- **Density batching** fixes the **batch-level** imbalance: it makes sure each batch **contains** enough tail-heavy members so that tail codes appear often enough in that batch for their gradient to matter. So it addresses “what the model sees.”
- **Underrepresentation** has two layers:  
  1) **In the batch:** tail is a tiny fraction of (sample, code) pairs → fixed by **dense (density) batching**.  
  2) **In the loss:** easy negatives dominate total gradient → fixed by **ASL**.

So they’re complementary: **ASL** makes the best use of each batch (don’t waste gradient on easy negatives; keep full signal for positives), and **dense sampler batching** ensures each batch has enough rare/tail code occurrences so that signal exists to begin with. Together they target both “see enough tail” and “weight tail vs easy negatives correctly.”

---

### 🧑 User

create a new table using select * from edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round7_512dim_exp2b_flash_learned_pool_medicare_all_sample_embedding; the new table in bigquery will be edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round6_6_8M_exp2b_flash_learned_pool_weight200_medicare_all_sample_embedding

---

### 🤖 Assistant

In **Ask mode** I can’t run BigQuery for you, but you can create the new table with this SQL in the BigQuery console (or `bq query`):

```sql
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round6_6_8M_exp2b_flash_learned_pool_weight200_medicare_all_sample_embedding`
AS
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round7_512dim_exp2b_flash_learned_pool_medicare_all_sample_embedding`;
```

- **CREATE OR REPLACE TABLE** creates the table and overwrites it if it already exists. Use **CREATE TABLE** (and drop the target first if needed) if you want to avoid overwriting.
- The new table will have the same schema and data as the source.

From the shell you can run:

```bash
bq query --use_legacy_sql=false '
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round6_6_8M_exp2b_flash_learned_pool_weight200_medicare_all_sample_embedding`
AS
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round7_512dim_exp2b_flash_learned_pool_medicare_all_sample_embedding`;
'
```

If you want this executed automatically from your project (e.g., via a script or Cursor in Agent mode), say how you run BigQuery (console vs `bq` vs Python) and we can wire it up.

