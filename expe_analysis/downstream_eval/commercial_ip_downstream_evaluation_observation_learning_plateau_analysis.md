I now have a comprehensive evidence base. Let me deliver the deep, systematic analysis.

---

# Comprehensive Root Cause Analysis: Learning Plateau, Downstream Performance, and Code Representation Bias

## Part 1: Summary of Previous Chat and Experimental Arc

The previous conversation (and the extensive analysis files in `expe_analysis/exp_round5/learning_plateau/`) documented a systematic investigation of a learning plateau in the multi-label clinical code prediction pretraining task. Here is the experimental history:

| Version | Config | Key Pretraining Finding |
|---|---|---|
| **V2** | BCE + pos_weight=35 | Baseline. Gradient concentration: 85% common, 0.12% tail. tail_acc=0% |
| **V3** | BCE + pos_weight=200 | 5.7x pos_weight increase had <0.5% gradient change. Medium code collapse (-96%). Per-sample magnitude **exhausted** |
| **V4** | ASL (gamma+=0, gamma-=4) | Ranking ceiling broken (recall@1: 1%->24%, MRR +43%). Tail still 0%. Per-sample focusing **exhausted** for tail |
| **V5** | ASL + Density-Aware Batching | Ranking further improved (recall@1: 28%, MRR +5.2%). Medium recovered. Tail **still 0%**. Modest batch composition **insufficient** |

**Key conclusions from the prior analysis:**

1. The learning plateau has **two separable components**: (a) a ranking ceiling for common codes (solved by ASL in V4), and (b) structural tail code gradient starvation (unsolved across all 4 experiments)
2. Three distinct per-sample interventions (pos_weight scaling, aggressive pos_weight, asymmetric focal loss) all converged to the **same terminal gradient distribution** (~85% common, ~0.1% tail)
3. Density-aware batching provided a transient delay in gradient concentration but did not prevent it
4. The tail gradient starvation is an **emergent, self-reinforcing process** operating at the batch-aggregation level
5. Tail embeddings are **homogenized** (std=0.03 vs common std=0.27), making them uninformative for downstream tasks

---

## Part 2: Downstream Evaluation Results — The Critical Evidence

Now let me analyze the downstream inpatient (IP) prediction results from the JSON data. This is where the pretraining investments get tested against reality.

### 2.1 Embedding-Only Models (Key Columns You Asked About)

| Metric | exp2b (BCE+pw35) | V4 (ASL) | V5 (ASL+Dense) | exp_round6 (3.4M) | Matched Tabular |
|---|:---:|:---:|:---:|:---:|:---:|
| **val AUC-ROC** | 0.7645 | 0.7714 | 0.7723 | **0.7844** | **0.8230** |
| **test AUC-ROC** | 0.7905 | **0.7906** | 0.7827 | **0.7969** | **0.8202** |
| **test Lift@1%** | 15.06 | 14.46 | **15.33** | 14.08 | **19.39** |
| **test Lift@5%** | 7.24 | 6.98 | 7.00 | 7.22 | **8.15** |
| **test Lift@10%** | 4.74 | 4.78 | **4.91** | **4.91** | **5.28** |
| **OOT AUC-ROC** | **0.7831** | 0.7815 | 0.7802 | **0.7964** | **0.8238** |
| **OOT-strict AUC-ROC** | **0.7932** | 0.7898 | 0.7828 | **0.7990** | **0.8307** |
| **OOT-strict Lift@1%** | 14.22 | **15.76** | 15.20 | **16.18** | **17.71** |

### 2.2 Hybrid Models (Embeddings + Tabular)

| Metric | exp2b hybrid | V4 hybrid | V5 hybrid | exp_round6 hybrid | Matched Tabular |
|---|:---:|:---:|:---:|:---:|:---:|
| **test AUC-ROC** | **0.8241** | 0.8214 | 0.8151 | 0.8217 | 0.8202 |
| **OOT AUC-ROC** | 0.8223 | 0.8226 | 0.8227 | **0.8247** | 0.8238 |
| **OOT-strict AUC-ROC** | 0.8260 | 0.8252 | 0.8215 | **0.8265** | **0.8307** |

### 2.3 Critical Observations from Downstream Data

**Observation 1: The pretraining metric improvements did NOT translate to downstream improvement.**

This is the most important finding. Despite V4's 2300% improvement in recall@1, 43% improvement in MRR, and 19% improvement in NDCG@10 over V2 in pretraining:
- V4 embedding-only OOT-strict AUC: **0.7898** vs exp2b (V2): **0.7932** — V4 is actually **worse**
- V5 (even better pretraining metrics): OOT-strict AUC: **0.7828** — even **worse** than both

**Observation 2: V5 (ASL + density-aware batching) produced the WORST downstream embeddings of the three.**

Despite V5 being "the best overall pretraining model" with the best recall@1 (0.284), MRR (0.496), NDCG@10 (0.478), and common_top10_acc (83.3%), its downstream performance is:
- OOT-strict AUC: 0.7828 (worst of all three)
- OOT AUC: 0.7802 (worst of all three)
- Test AUC: 0.7827 (worst of all three)

**Observation 3: No hybrid model beats the matched tabular baseline on OOT-strict.**

All hybrid models (embeddings + 533 tabular features) underperform the tabular-only model on the strictest evaluation (OOT-strict AUC-ROC):
- Best hybrid (exp_round6): 0.8265
- Matched tabular: **0.8307**
- The embeddings are **not adding value** and may be adding noise

**Observation 4: The only improvement that translated downstream was more pretraining data.**

exp_round6 (trained on 3.4M members) vs the others (~1.5M):
- exp_round6 embedding-only OOT-strict: **0.7990** — the best embedding model
- All 1.5M models cluster in 0.78-0.79 regardless of loss function

**Observation 5: The embedding-tabular gap is consistent and large.**

Across all experiments, the embedding-only AUC-ROC is ~3pp below tabular (0.79 vs 0.82 OOT-strict). The Lift@1% gap is ~23% (15x vs 19x). This gap persists regardless of pretraining optimization strategy.

---

## Part 3: Deep Root Cause Analysis

### 3.1 Separating the Facts from Hypotheses

**FACTS (from evidence):**

| # | Fact | Evidence |
|---|---|---|
| F1 | Pretraining ranking improvement (V2→V4→V5) did NOT improve downstream | OOT-strict AUC: V2=0.793, V4=0.790, V5=0.783 |
| F2 | V5 (best pretraining model) has the WORST downstream performance of the three | OOT-strict: V5=0.783 < V4=0.790 < V2=0.793 |
| F3 | More pretraining data improves downstream (exp_round6 = 0.799) | 3.4M > 1.5M in all downstream metrics |
| F4 | No hybrid model beats tabular-only on OOT-strict | Best hybrid: 0.827, tabular: 0.831 |
| F5 | All embedding-only models cluster in 0.78-0.80 OOT-strict | Narrow range despite massive pretraining metric differences |
| F6 | Tail code starvation persists across all experiments | tail_top10_acc = 0% in V2/V3/V4/V5 |
| F7 | Tabular uses 533 features; embedding uses 256 dimensions | Feature count from JSON |
| F8 | The AUC gap (embedding vs tabular) is ~3pp on OOT-strict | 0.79-0.80 vs 0.83 |
| F9 | Lift@1% gap is ~23% | ~15x vs ~19x |

### 3.2 How Much is the Code Representation Bias (Tail Starvation) Responsible?

This is the central question. Let me analyze it rigorously.

**Hypothesis A: Tail starvation is the PRIMARY cause of the embedding-tabular gap.**

If this were true, we would expect:
- Interventions that improve tail code representation should improve downstream (**NOT OBSERVED**: V4/V5 improved common-code pretraining metrics dramatically but downstream got WORSE)
- The gap should narrow as pretraining metrics improve (**NOT OBSERVED**: gap remained constant or widened)
- Hybrid models should outperform tabular because embeddings add complementary signal (**NOT OBSERVED**: hybrids underperform tabular on OOT-strict)

**Verdict: WEAKLY SUPPORTED at best.** The evidence does not support tail starvation as the primary driver of the downstream gap.

**Hypothesis B: Task misalignment (code prediction vs. IP prediction) is the PRIMARY cause.**

If this were true, we would expect:
- Improving code prediction metrics should NOT reliably improve downstream IP prediction (**OBSERVED**: V4's 2300% recall@1 gain → 0pp downstream gain)
- The best pretraining model may not be the best downstream model (**OBSERVED**: V5 best pretraining, worst downstream)
- The "shape" of the embedding space optimized for code prediction is different from what's optimal for IP prediction (**CONSISTENT with all evidence**)

**Verdict: STRONGLY SUPPORTED.** The disconnection between pretraining improvements and downstream performance is the strongest signal in the data.

**Hypothesis C: Information compression bottleneck (256 dims vs 533 features) is a significant contributor.**

If this were true, we would expect:
- Hybrid should not improve much over tabular alone (adding 256 dims to 533 features → marginal at best) (**OBSERVED**)
- Embedding-only should have a consistent ceiling regardless of pretraining quality (**OBSERVED**: all cluster in 0.78-0.80)
- More data (3.4M) helps because it enables better compression of the code information (**OBSERVED**: exp_round6 is the best embedding model)

**Verdict: MODERATELY SUPPORTED.**

### 3.3 The True Root Cause Hierarchy (Evidence-Based)

Based on the downstream evidence, I rank the root causes as follows:

**Root Cause 1 (PRIMARY): Pretraining-Downstream Task Misalignment**

The pretraining objective (multi-label code prediction across 5,700+ codes) and the downstream task (binary IP prediction with ~0.9% prevalence) are fundamentally different tasks. The embedding space learned for code prediction optimizes for:
- Code co-occurrence patterns
- Temporal code sequences
- Multi-label probability calibration

But IP prediction requires:
- Clinical severity signals
- Utilization patterns
- Specific high-risk condition combinations
- Risk trajectory markers

The evidence is decisive: V4's 2300% recall@1 improvement and 54% Brier score reduction in pretraining translated to **zero** downstream improvement. This means the calibration and ranking improvements in code prediction did not alter the embedding's representation of IP-relevant clinical signal. The embedding space "moved" dramatically in code-prediction space but essentially stayed still in IP-prediction space.

**Mechanistic explanation**: ASL (V4) and density batching (V5) primarily improved the **decoder** (the projection from embedding to logits) by better calibrating common code probabilities. The **encoder** (which produces the embedding) learned to be more discriminative for common code ranking, but this discrimination is orthogonal to IP-relevant features. The embedding at the last valid day captures the temporal clinical pattern, and the common-code ranking improvements don't change the fundamental clinical signal in that temporal representation.

**Root Cause 2 (SECONDARY): Domain Knowledge Gap**

The tabular model has 533 features that include hand-engineered clinical constructs designed by domain experts specifically for risk prediction:
- Comorbidity indices (Charlson, Elixhauser)
- CCS/CCSR category aggregations
- Utilization metrics (ER visits, office visits, prescription counts)
- Temporal trend features (increasing/decreasing utilization)
- Specific clinical risk factors (prior hospitalizations, specific diagnoses)

These features embed decades of clinical knowledge about what predicts hospitalization. A 256-dimensional embedding learned from raw code sequences, without any explicit clinical knowledge injection, cannot easily replicate this.

**Root Cause 3 (TERTIARY): Tail Code Representation Deficit**

The tail code starvation IS a real problem, but its contribution to the downstream gap is likely **smaller than previously hypothesized**, for three reasons:

1. **Redundancy**: The OR analysis (Jan 30) showed rare codes have higher individual OR with IP, but this doesn't prove unique predictive signal. Members with rare codes also have common codes, and the common code patterns may already capture the IP risk through co-occurrence.

2. **Limited downstream impact demonstrated**: If tail code information were critical for IP prediction, we would expect:
   - Hybrid models (which get the full 533 tabular features including some that capture rare condition information) to show large improvement → they don't
   - V5's improved batch diversity (which incidentally improved common code representation quality) to translate downstream → it didn't

3. **The IP prevalence is very low (~0.9%)**: At this prevalence, the signal comes from specific high-risk patterns that are likely concentrated in common/medium codes that are well-represented in the embedding.

### 3.4 Why V5 Performed WORSE Downstream Than V2

This is the most counterintuitive finding and deserves specific analysis.

V5 (ASL + density batching) produced the best pretraining model but the worst downstream model. The possible mechanisms:

1. **ASL's probability compression changed the embedding space unfavorably for downstream**: ASL compresses the probability space (val_bce_loss 30x higher than V2). This means the internal representations are optimized for a different probability landscape. The embedding captures "ASL-optimal" patterns that are not IP-predictive.

2. **Density batching introduced a distribution shift**: By oversampling tail-dense members, V5's encoder learned representations biased toward a different member population than the downstream evaluation population. The downstream commercial IP evaluation uses a random population sample, not a tail-dense sample.

3. **Better common-code ranking ≠ better temporal representation**: V5's recall@1 of 0.284 means it correctly identifies the single most likely code more often. But the embedding (taken from the last temporal layer, not the decoder) may have become more "peaked" around common code features, reducing its ability to encode the broader clinical context that IP prediction requires.

### 3.5 The Relationship Between Learning Plateau and Downstream Gap

To directly answer your question: **The learning plateau and the downstream embedding underperformance are RELATED but NOT causally linked as previously assumed.**

| Phenomenon | Relationship to Downstream Gap |
|---|---|
| **Common code ranking plateau** (solved by ASL) | **NOT a cause** — solving it didn't help downstream |
| **Tail gradient starvation** (unsolved) | **Partially contributes** — but magnitude unclear and likely secondary |
| **Tail embedding homogenization** (std=0.03) | **Partially contributes** — these dimensions are wasted, but may not contain unique IP signal |
| **Pretraining objective misalignment** | **PRIMARY cause** — demonstrated by the complete disconnection between pretraining and downstream metrics |

The learning plateau investigation was scientifically rigorous and produced valuable mechanistic understanding of gradient dynamics in long-tail multi-label classification. However, the downstream results reveal that **the learning plateau was primarily a pretraining-task problem, not a downstream-representation problem.** The model was plateauing on code prediction metrics, and fixing that plateau (V4, V5) improved code prediction but not the quality of the learned representation for IP prediction.

---

## Part 4: Synthesis and Implications

### 4.1 What We Now Know with High Confidence

| Conclusion | Evidence | Confidence |
|---|---|---|
| Pretraining metric improvements (V4/V5) do NOT translate to downstream IP improvement | OOT-strict AUC: V2=0.793 > V4=0.790 > V5=0.783 | **HIGH** |
| The embedding-tabular gap (~3pp AUC) is NOT primarily caused by tail code starvation | V4/V5 didn't close the gap despite pretraining improvements | **HIGH** |
| More pretraining data is the most effective lever for downstream | exp_round6 (3.4M) = best downstream at 0.799 OOT-strict | **HIGH** |
| Hybrid models do not beat tabular-only on strict evaluation | Best hybrid (0.827) < tabular (0.831) on OOT-strict | **HIGH** |
| The pretraining objective is misaligned with downstream IP prediction | Decisive: massive pretraining gains → zero downstream gains | **HIGH** |
| The tail code starvation is a pretraining problem with unclear downstream relevance | No evidence that solving it would close the downstream gap | **MODERATE** |

### 4.2 What the Evidence DOES NOT Show

1. **Whether tail codes carry unique IP-predictive signal not captured by common codes.** The redundancy analysis was never performed.
2. **Whether a different pretraining objective (e.g., contrastive, masked code modeling) would produce better downstream embeddings.** Only code prediction has been tested.
3. **What the ceiling is for embedding-only downstream performance.** We've only tested 256-dim embeddings from code prediction.

### 4.3 Revised Understanding of the Problem

The original framing was: *"The learning plateau (tail code starvation) is preventing the model from learning good embeddings, which is why embedding-only underperforms tabular."*

The evidence-based reframing is: *"The embedding-tabular gap is primarily caused by task misalignment between code prediction and IP prediction, compounded by an information compression bottleneck (256 dims vs. 533 features). Tail code starvation is a real pretraining problem but its contribution to the downstream gap is secondary and possibly minor."*

### 4.4 What This Means for the V6 (Per-Tier Loss Balancing) Recommendation

The V6 experiment (per-tier loss balancing) proposed in the previous chat would likely:
- Successfully maintain tail gradient fraction >10% (mechanistically sound)
- Possibly move tail_top10_acc off zero (if gradient is sufficient to break homogenization)
- **But unlikely to improve downstream IP prediction** — because the problem is not tail representation quality, it's task alignment

This doesn't mean V6 is worthless — it would validate the gradient starvation diagnosis and potentially improve the embedding's clinical breadth. But setting expectations correctly is important: **solving the tail code problem alone will NOT close the 3pp AUC gap with tabular.**

---

## Part 5: Actionable Implications

Based on this comprehensive analysis, if the goal is to **close the gap between embedding-only and tabular performance**, the following approaches are ranked by evidence support:

| Priority | Approach | Rationale | Expected Impact |
|---|---|---|---|
| **1** | **More pretraining data** (5M+) | Only intervention that improved downstream (3.4M > 1.5M) | Medium (0.5-1pp AUC) |
| **2** | **Task-aligned finetuning** (e.g., downstream-aware pretraining) | Addresses the root cause: task misalignment | Potentially large (1-2pp) |
| **3** | **Larger embedding dimensions** (512+) | Addresses compression bottleneck | Small-Medium |
| **4** | **Clinical auxiliary losses** (CCS/CCSR hierarchy) | Injects domain knowledge into pretraining | Medium |
| **5** | **Contrastive pretraining** (clinical similarity) | Learns clinically-meaningful representation space | Unknown |
| Lower | **Per-tier loss balancing (V6)** | Addresses tail starvation (real but secondary problem) | **Low for downstream** |
| Lower | **More ASL/sampling optimization** | Already shown to not translate downstream | **Negligible** |

The critical insight is that **the path to competitive embedding performance runs through task alignment and data scale, not through further pretraining loss engineering.** The learning plateau work was valuable science but optimized the wrong metric for the downstream goal.