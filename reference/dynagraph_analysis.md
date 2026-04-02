Now I have comprehensive context on both the paper and your project. Let me provide the thorough walkthrough.

---

# DynaGraph Critical Review: Complete Technical Walkthrough

## Part 1: Rationale and Unique Contributions

### What Problem Does DynaGraph Actually Solve?

The paper targets a genuine and under-addressed gap in clinical ML: **the static relationship assumption**. Here is the crux of the problem, stripped to its essence:

When a patient develops septic shock, the relationship between creatinine and blood pressure *changes* fundamentally. During normal physiology, these variables are loosely connected. During cardiorenal syndrome, they become tightly coupled. During recovery, they decouple. Existing models either:

1. **Ignore inter-variable relationships entirely** (LSTM, GRU, Transformer, TCN) -- each feature is processed as an independent time series and the model must learn any cross-feature interactions implicitly through shared hidden states.
2. **Use fixed graphs from medical ontologies** (GCN, GAT, most GNN methods) -- "creatinine connects to potassium" with a static edge weight that never changes, whether the patient is healthy or in multi-organ failure.
3. **Use correlation-based graphs with fixed thresholds** (SimTSC, some temporal GNNs) -- slightly better, but the graph structure is determined once at preprocessing and frozen.

DynaGraph's core claim is: **learn the graph structure itself as a function of time, jointly with the prediction task, end-to-end from data**.

### How Is It Genuinely Different From Prior Work?

Let me be precise about what is and is not novel:

**Genuinely novel (strong claims):**

- **Fully learnable, time-varying adjacency matrices** -- No prior method in the clinical domain learns both the graph topology AND its temporal evolution end-to-end without predefined structure. The adjacency matrix entries are learnable parameters that are trained through backpropagation on the clinical prediction objective. This means the model can discover that the creatinine-urea edge strengthens during acute kidney injury and weakens during recovery, without anyone telling it that these variables are related.

- **Joint spatio-temporal + sequential modeling** -- The parallel LSTM pathway captures *within-variable* temporal patterns (e.g., creatinine rising from 1.2 to 3.5 mg/dL), while the graph pathway captures *between-variable* relationships (e.g., creatinine-urea coupling). The two are fused via element-wise addition then modulated by interpretability weights. This dual representation is architecturally distinct from methods that do one or the other.

- **Integrated gradient-based pseudo-attention for temporal interpretability** -- The interpretability matrix `I` is not a post-hoc explanation (like SHAP or GradCAM applied after training). It is a learnable weight matrix that is part of the forward pass and updated based on gradient magnitudes during training. This means interpretability is baked into the learned representation, not bolted on.

**Less novel than claimed (honest assessment):**

- **Contrastive graph augmentation** -- While applied creatively (shuffling along time axis, node dropping, edge perturbation), contrastive learning on graphs is well-established (GraphCL, GRACE, etc.). The paper adapts known techniques rather than inventing new augmentation strategies.

- **Focal loss for class imbalance** -- The paper correctly notes this is "a necessary implementation consideration rather than our core methodological innovation." However, it is presented prominently in the multi-loss objective. Focal loss is standard in imbalanced classification since Lin et al. (2017).

- **VGAE with GIN encoder** -- Variational graph autoencoders and Graph Isomorphism Networks are both established components. The contribution is their specific combination and integration into the DynaGraph pipeline, not the components themselves.

**Critical comparison to methods most similar to DynaGraph:**

| Method | Learned Graph? | Time-Varying? | Interpretable? | Clinical Validation? |
|--------|---------------|---------------|----------------|---------------------|
| MTGNN (Wu et al., 2020) | Yes | No (one static learned graph) | No | No (traffic/weather) |
| DCRNN (Li et al., 2018) | No (predefined) | Diffusion over time | No | No (traffic) |
| TodyNet (2022) | No | Time-aware | Partial | No |
| MedGNN | No (medical ontology) | No | No | Yes |
| STraTS (2022) | No | Neural ODE for irregularity | Partial | Yes |
| **DynaGraph** | **Yes** | **Yes (per-window)** | **Yes (gradient-based)** | **Yes (4 datasets)** |

The unique intersection is: *learned + time-varying + interpretable + clinically validated*. No single prior method occupies this same cell.

---

## Part 2: Technical Components -- Deep Walkthrough

### 2.1 Input Representation and Time Windowing

**What happens:** A 24-hour patient trajectory with `d` clinical features (vital signs, labs, demographics) sampled hourly is divided into `s = 6` equal windows of 4 hours each. This gives a tensor `X in R^{d x l x s}` where `d` is the number of features, `l` is the number of time points per window, and `s` is the number of windows.

**Plain language:** Imagine laying out a patient's ICU data as a spreadsheet. Rows are features (heart rate, creatinine, etc.), columns are hours. You slice this spreadsheet into 6 chunks of 4 hours each. Each chunk becomes one "time slice" that the model processes.

**Design rationale:** The 4-hour windowing serves two purposes:
1. It creates discrete time steps where graph structure can meaningfully change (physiological relationships don't restructure minute-by-minute, but can shift over 4-hour blocks)
2. It makes the model computationally tractable -- instead of learning a d x d adjacency matrix at every hour, you learn one per 4-hour window

**Critical note:** The choice of 6 windows over 24 hours is a hyperparameter that encodes an assumption about the temporal resolution at which physiological relationships change. The paper uses grid search on the validation set, finding 6 optimal. Finer granularity (more windows) would capture faster transitions but increase computation quadratically.

### 2.2 Dynamic Graph Construction Through Information Propagation

This is the core innovation. Let me walk through it step by step.

**Step 1: Initial Graph (t=1)**

For the first time window, the model constructs an adjacency matrix:

\[A = \Theta^T \cdot \Psi \in \mathbb{R}^{d \times d}\]

where `Theta` and `Psi` are both vectors of length `d` with **learnable parameters initialized randomly**.

**What this means in plain language:** Each clinical feature gets two learnable numbers -- a "source score" (in Theta) and a "target score" (in Psi). The relationship strength between any two features is the product of one feature's source score and the other's target score. This is a low-rank factorization of the adjacency matrix -- instead of learning `d^2` independent parameters, you learn `2d` parameters and their outer product gives you the full `d x d` matrix.

**Mathematical intuition:** This is essentially the same idea as how word embeddings work in Word2Vec's skip-gram. Each feature has an embedding as a "source" and as a "target," and their dot product captures how strongly they relate. The key difference: these embeddings are optimized for clinical prediction, so features that co-contribute to mortality risk will develop strong edges.

**Step 2: Sparsification via top-k**

\[\text{idx, idy} = \text{argtop-k}(A[:, :])\]
\[A[\text{-idx, -idy}] = 0\]

**Plain language:** The full d x d matrix would connect every feature to every other feature. That's wasteful -- not all features are meaningfully related. So they keep only the k strongest connections and zero out the rest. This makes the graph sparse and computationally cheaper.

**Design rationale:** Sparsification serves double duty: it reduces computation and acts as an implicit regularizer by forcing the model to commit to the most important relationships rather than spreading weight thinly across all pairs.

**Step 3: Temporal Propagation (t=2 through t=s)**

For subsequent time windows, new vertices representing the previous time step's features are added, connected to their counterparts in the current window. Then node embeddings are aggregated and redundant vertices are removed.

**Plain language:** Think of it this way -- at time window 2, the model sees both "creatinine at window 1" and "creatinine at window 2" as separate nodes, connected by an edge. The message passing allows information to flow from past states to current states. Then the past nodes are folded back into the current nodes via aggregation. This prevents the graph from growing exponentially while maintaining temporal memory.

**Mathematical mechanism:** This is an adaptation of temporal message passing from temporal graph networks (TGAT), but with the crucial difference that the graph structure itself (which nodes connect to which) is learned rather than predetermined.

**The result:** A sequence of adjacency matrices `A = {A_1, A_2, ..., A_s}` where each `A_t` is a d x d matrix capturing the inter-feature relationships at time window t. These matrices are different from each other -- the creatinine-urea edge might be 0.3 at t=1 and 0.8 at t=5 if the patient develops kidney injury.

### 2.3 LSTM Embeddings (Parallel Pathway)

**What happens:** The same input `X` is simultaneously processed by an LSTM to produce embedding matrices `E = {E_1, E_2, ..., E_s}`, each of dimension `d x d x s`.

**Plain language:** While the graph is learning *which features relate to which*, the LSTM is learning *how each individual feature evolves over time*. The LSTM sees a feature's values across time windows and produces a hidden state that encodes its temporal trajectory. For example, the LSTM for creatinine would encode "rising steadily" differently from "spiked and recovered."

**Why LSTM and not the graph alone?** The graph captures *between-feature* relationships at each time snapshot. But it doesn't natively model the *within-feature* temporal dynamics. A rising creatinine has different clinical meaning than a stable creatinine at the same absolute value. The LSTM provides this complementary temporal context.

**Design rationale:** This dual-pathway design is the paper's answer to a fundamental tension in temporal graph learning: graphs are inherently about *structure* (relationships between things), while time series are inherently about *evolution* (how things change). By processing both in parallel and fusing them, DynaGraph captures both dimensions.

### 2.4 Interpretability Mechanism (Pseudo-Attention)

This is perhaps the most practically relevant component. Let me unpack it carefully.

**The mechanism:** For each time slice, a weight matrix `I_t` of dimension d x d is maintained. This matrix starts uniform (all weights equal) and is updated during training based on which nodes and edges contribute most to the loss gradient.

**Node importance (Eq. 4):**

\[I_{vv} = \|\nabla_{h_v} \mathcal{L}\|\]

**Plain language:** How important is feature `v` to the prediction? Compute the gradient of the loss with respect to feature v's hidden representation. If changing feature v's representation would dramatically change the loss (large gradient magnitude), it's important. If the loss barely changes (small gradient), the feature doesn't matter much at this time point.

**Edge importance (Eq. 5):**

\[I_{vu} = \|\nabla_{e_{v,u}} \mathcal{L}\|\]

**Plain language:** How important is the *relationship* between features v and u? Same logic -- if the loss is sensitive to the edge weight between creatinine and urea, that relationship matters for prediction.

**Combined feature importance (Eq. 3):**

\[I_v = \alpha \cdot I_{vv} + (1-\alpha) \cdot \bar{I}_{vu}\]

**Plain language:** A feature's total importance combines its direct importance (how much it matters on its own) with the average importance of its relationships with other features. The parameter alpha controls the balance. If alpha = 1, only the node itself matters. If alpha = 0, only its connections matter. In practice, both matter.

**Why "pseudo-attention" and not actual attention?** True attention (as in Transformers) computes query-key-value products and uses softmax to produce weights. This mechanism instead uses gradient magnitudes as importance scores. It is called "pseudo-attention" because it serves a similar function -- weighting different parts of the input based on their relevance -- but through a fundamentally different mechanism (gradient-based rather than query-key).

**Critical assessment:** The interpretability claim is strong for *correlational* understanding (which features co-vary with the prediction) but explicitly does not provide *causal* guarantees. The paper correctly acknowledges this limitation. Knowing that creatinine-urea coupling is important at t=5 tells you the model relies on this relationship, but not whether creatinine *caused* urea changes or vice versa.

### 2.5 Graph Fusion (Eq. 8)

\[G^{(i)} = (A^{(i)} + E^{(i)}) \odot I^{(i)}\]

**Plain language:** For each time slice, take the learned adjacency matrix (inter-feature structure), add the LSTM-derived embedding affinity matrix (temporal evolution), then multiply element-wise by the interpretability weights (importance modulation). The result is a final graph that encodes:
- Which features are related (from A)
- How they're evolving (from E)
- How much each relationship matters for prediction (from I)

**Design rationale for additive fusion of A and E:** Addition rather than concatenation or multiplication means that either pathway can contribute independently. If the graph discovers a strong creatinine-urea edge (high A entry) but the LSTM sees both features are stable (low E entry), the relationship is still represented. If the LSTM detects rapid change in both (high E entry) but the graph hasn't learned a strong edge (low A entry), the temporal signal still propagates. The Hadamard product with I then gates the combined signal based on prediction-relevant importance.

### 2.6 GIN Encoder and VGAE

**What happens:** The fused graph G is processed by a Variational Graph Autoencoder (VGAE) that uses Graph Isomorphism Networks (GINs) as both encoder and decoder.

**GIN update rule (Eq. 6):**

\[h_v^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)\]

**Plain language:** For each node v at layer k, take its own representation from the previous layer, scale it by (1 + epsilon), add up all its neighbors' representations, and pass the sum through an MLP. The epsilon parameter lets the model control how much a node's own features matter relative to its neighborhood.

**Why GIN specifically?** GINs are provably as powerful as the Weisfeiler-Lehman graph isomorphism test -- meaning they can distinguish between different graph structures as well as any message-passing neural network can. This is important because DynaGraph's learned graphs have different structures at different time points, and the encoder needs to be sensitive to these structural differences.

**The VGAE component:** The encoder maps each time-slice graph to a distribution in latent space (mean and variance parameters). Samples from this distribution are decoded back to reconstruct the graph. The VGAE loss (Eq. 13) combines reconstruction loss (BCE between original and reconstructed graph) with a KL divergence regularizer that keeps the latent distributions close to a standard normal.

**Design rationale:** The VGAE serves two purposes:
1. It provides a compressed, regularized representation of each time-slice graph
2. The KL regularization prevents the latent space from collapsing to degenerate solutions where all graphs look the same

### 2.7 Temporal Pooling

**What happens:** After the VGAE, the output graphs are pooled to reduce the number of nodes using a hierarchical, differentiable pooling mechanism based on 2D CNNs.

**Equations 14-15:**

\[X_t^{(\ell+1)} = \text{CNN}^{(\ell)}(X_t^{(\ell)})\]
\[G_t^{(\ell+1)} = M^{(\ell)} G_t^{(\ell)} M^{(\ell)\top}\]

**Plain language:** The CNN produces a soft assignment matrix M that clusters nodes. Rather than hard-assigning each node to a cluster, each node has a probability distribution over clusters. The graph is then "coarsened" -- many nodes become fewer super-nodes, with edges between super-nodes reflecting the aggregate edge patterns of their constituent nodes.

**Design rationale:** Without pooling, the final representation would be d x d x s in dimension, which for 44 features and 6 time slices would be ~11,616 dimensions feeding into the MLP classifier. Pooling progressively reduces this while preserving the structural patterns that matter for prediction.

### 2.8 Multi-Loss Training Objective

The total loss (Eq. 20) combines five components:

\[\mathcal{L}_{total} = \alpha \mathcal{L}_{contrast} + \epsilon \mathcal{L}_{focal} + \lambda \mathcal{L}_{reg} + \mu \mathcal{L}_{structure} + \beta \mathcal{L}_{VGAE}\]

Let me explain each and *why* it's needed:

**1. Contrastive Loss (Eq. 16)** -- Largest ablation impact (-0.048 AUROC when removed)

Augmented graphs (time shuffled, node-dropped, edge-perturbed) should be similar to the original; random negative pairs should be dissimilar. This forces the model to learn graph representations that are robust to noise and capture essential structure rather than superficial patterns.

**Why it matters:** Clinical data is noisy, irregularly sampled, and full of measurement artifacts. Without contrastive training, the model might learn representations that are fragile to small perturbations in input timing or missing values.

**2. Focal Loss (Eq. 17)** -- Second largest impact (-0.035 AUROC when removed)

\[\mathcal{L}_{focal}(\hat{y}) = -(1-\hat{y})^\gamma \cdot \log(\hat{y})\]

**Plain language:** Standard cross-entropy treats all examples equally. Focal loss down-weights easy examples (where the model is already confident) and up-weights hard examples (where the model is uncertain). The gamma parameter controls how aggressively easy examples are suppressed.

**Why it matters:** With prevalences ranging from 2% to 38%, the model would otherwise overwhelmingly optimize for predicting the majority class. Focal loss ensures rare outcomes still receive meaningful gradient signal.

**3. Regularization Loss (Eq. 18)** -- Moderate impact

\[\mathcal{L}_{reg} = \lambda \sum_{(i,j) \in E} \|h_i - h_j\|^2\]

**Plain language:** Connected nodes should have somewhat similar representations. This prevents the graph structure from being ignored -- if two features are connected by a strong edge, their learned representations shouldn't be wildly different.

**4. Structural Loss (Eq. 19)** -- Smallest individual impact but prevents instability

\[\mathcal{L}_{structure} = \mu \left(1 - \frac{\sum_{i,j} A_{ij} \cdot A'_{ij}}{\sqrt{\sum_{i,j} A_{ij}^2} \cdot \sqrt{\sum_{i,j} A'^2_{ij}}}\right)\]

**Plain language:** This is cosine dissimilarity between consecutive adjacency matrices. It penalizes the graph structure from changing too drastically between adjacent time windows. If the creatinine-urea edge is 0.7 at t=3, it can't jump to 0.1 at t=4 without incurring a penalty.

**Why it matters:** Without this, the learned graphs could oscillate wildly between time windows, producing uninterpretable dynamics. The structural loss ensures smooth, gradual transitions that mirror how physiological relationships actually evolve.

**5. VGAE Loss (Eq. 13)** -- Standard component

Combines graph reconstruction loss with KL divergence regularization.

---

## Part 3: Implications for Your Transformer Embedding Design

This is where DynaGraph becomes directly actionable for your Clinical TE project. Based on my understanding of your architecture (hierarchical Flash-attention transformer with optional MoE, multi-label BCE/ASL pretraining, ~84k code vocabulary, 200-day temporal windows), here are the specific insights and why they matter:

### 3.1 The Fundamental Diagnosis DynaGraph Validates About Your Architecture

Your temporal architecture research document identified the core problem precisely: **the TE converges to bag-of-codes aggregate statistics redundant with tabular features**. DynaGraph provides independent evidence for *why* this happens and points toward what to do about it.

**The why:** Your current architecture processes features independently through time (each code gets an embedding, codes within a day are pooled, days are processed sequentially with causal attention). The self-attention mechanism in the temporal encoder CAN in principle learn cross-feature relationships, but it must discover them implicitly through shared hidden states. DynaGraph shows that **explicit relational modeling** (learned inter-variable graphs) captures dynamics that implicit sequence modeling misses. The 6-8% AUPRC improvement over vanilla Transformers in Table 1 quantifies this gap.

**The relevance to your specific problem:** Your TE has 9 code types (diagnoses, procedures, medications, etc.) that interact in clinically meaningful ways that change over time. A patient starting a statin (GPI medication code) after an MI diagnosis (ICD-10 code) with a concurrent procedure code (cardiac cath) represents a coherent clinical trajectory. Your current architecture treats these as independent tokens in the daily set, relying on the attention mechanism to figure out the relationships. DynaGraph's results suggest that **explicitly modeling how code-type relationships evolve over the patient's timeline** would capture information your current architecture misses.

### 3.2 Specific Design Ideas to Borrow

**Idea 1: Learned Inter-Code-Type Adjacency (Highest Priority)**

DynaGraph learns a d x d adjacency matrix where d is the number of clinical features. In your setting, you could learn a **code-type adjacency matrix** that evolves over time windows. Instead of 44 features x 44 features, you'd have 9 code types x 9 code types (or finer: code-group x code-group) with learned, time-varying edge weights.

**Why this is specifically valuable for you:** Your temporal architecture research identified that the TE currently treats all code types as interchangeable tokens in the daily embedding. A learned adjacency matrix would force the model to discover that diagnosis-medication relationships strengthen when a new treatment begins, or that procedure-diagnosis relationships shift after a surgery. This is exactly the kind of temporal relational information that bag-of-codes methods cannot capture.

**Implementation concept:** After your daily encoder pools codes into a daily representation, add a **code-type graph layer** that learns a small adjacency matrix over code types per day/window. The graph edges would capture how strongly diagnosis codes relate to procedure codes, medication codes relate to diagnosis codes, etc., and how these relationships change across the patient's timeline.

**Idea 2: Structural Regularization for Temporal Stability (Medium Priority)**

Your learning plateau investigations found that gradient mass concentrates on common codes while tail codes starve. DynaGraph's structural loss (Eq. 19) provides a principle that could help: **prevent the learned representations from changing too drastically between adjacent time steps**.

**Why this matters for your plateau:** If your temporal encoder's hidden states fluctuate wildly between adjacent days, the model is effectively "resetting" its temporal memory at each step, which forces it to rely on aggregate statistics (the bag-of-codes trap). A structural consistency penalty on adjacent day representations would encourage smooth temporal trajectories that accumulate meaningful temporal information.

**Idea 3: Contrastive Augmentation for Temporal Robustness (Medium Priority)**

DynaGraph's contrastive loss had the single largest ablation impact (-0.048 AUROC). For your setting, a contrastive objective that augments patient timelines (temporal shuffling within windows, random code dropout, time-window shifting) could force the encoder to learn representations robust to the noise and irregularity in claims data.

**Specific formulation for your architecture:** Take a patient's 200-day sequence, create augmented views by (a) randomly dropping 10-15% of codes within days, (b) swapping adjacent days within short windows, (c) adding random temporal jitter. Train a contrastive head alongside your BCE objective that pushes augmented views of the same patient together and different patients apart.

**Why this specifically addresses your bag-of-codes problem:** If the TE relies on aggregate code frequencies, time-shuffling augmentations should NOT change the representation much (shuffling doesn't change frequencies). By training the contrastive loss to treat time-shuffled sequences as "different" from the original, you explicitly force the model to encode *temporal order*, breaking the bag-of-codes equilibrium.

**Idea 4: Gradient-Based Interpretability Matrix (Lower Priority, But High Value)**

DynaGraph's pseudo-attention mechanism using gradient magnitudes (Eqs. 3-5) is conceptually simple to adapt to your architecture. At each temporal position in your encoder, compute the gradient magnitude of the embedding with respect to the prediction loss. This gives you **time-resolved feature importance** for free, without any external explanation framework.

**Why this is better than post-hoc SHAP for your use case:** Your March 15 plan discusses SHAP + multi-checkpoint embedding generation. SHAP is computationally expensive (requires many forward passes per explanation) and operates post-hoc. DynaGraph's approach integrates importance scores into the training loop itself, making them available at inference time with zero additional cost.

### 3.3 What NOT to Borrow (Critical Caveats)

**The LSTM parallel pathway is not needed for you.** Your transformer's temporal encoder already captures within-feature temporal patterns through causal self-attention. DynaGraph uses LSTM because its graph pathway alone doesn't model within-feature evolution. Your architecture already handles this.

**The 24-hour window / 4-hour granularity doesn't map to your domain.** Your data operates on a day-level (200 days), not hour-level (24 hours). The relevant temporal resolution for claims data is weeks-to-months, not hours. If you adopt window-based graph learning, your windows should be on the order of 1-4 weeks, not hours.

**The VGAE may be over-engineered for your setting.** The VGAE provides latent space regularization for the learned graphs, but your code-type graph would be small (9 x 9 or similar) and wouldn't benefit from the complexity of variational inference. A simpler graph encoder (standard GIN or even a single attention layer over code types) would likely suffice.

**The specific multi-loss balancing is dataset-specific.** DynaGraph's five-component loss with hyperparameter balancing (alpha, epsilon, lambda, mu, beta) is tuned to their specific datasets and class distributions. Directly copying these weights would not work. However, the *principle* of combining a contrastive term with a structural stability term and a class-imbalance-aware task loss is sound and applicable.

### 3.4 The Strategic Takeaway: Where DynaGraph Points for Your TE

Your temporal architecture research document concluded that the TE needs to capture information that tabular features fundamentally cannot represent. DynaGraph provides **empirical proof** that this information exists and is capturable -- specifically, the **temporal evolution of inter-variable relationships** accounts for 6-8% AUPRC improvement over methods that treat variables independently.

For your TE, this translates to: **the temporal evolution of inter-code-type relationships** (how diagnosis-procedure-medication interactions change over a patient's clinical trajectory) is the signal that would differentiate your embeddings from simple bag-of-codes aggregates. This aligns perfectly with the direction your research already identified, and DynaGraph provides both a proof of concept and specific architectural mechanisms to implement it.

The most impactful single change would be to add a lightweight **temporal code-type graph layer** between your daily encoder and your temporal encoder, learning a small adjacency matrix over code types that evolves across the patient timeline. This directly addresses the bag-of-codes plateau by giving the model an explicit mechanism to capture relational dynamics -- exactly the information that PCA, AutoEncoder, and SelectKBest fundamentally cannot represent.