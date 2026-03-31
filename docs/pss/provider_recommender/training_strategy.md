# SmartRec Training Strategy Deep Dive: Data, Negative Sampling, Multi-Stage Training, and Inference

**Purpose:** A code-level walkthrough of the complete training-to-inference pipeline — how the training data is constructed, how negative sampling works, what each training stage does, and how the system produces recommendations at inference time.

---

## Table of Contents

1. [System Overview: Three-Stage Pipeline](#1-system-overview)
2. [Training Data Construction: Where Do the Pairs Come From?](#2-training-data)
3. [Negative Sampling: How the Model Learns "This Is Not a Match"](#3-negative-sampling)
4. [Stage 0: Data Preprocessing with NVTabular (GPU-Accelerated ETL)](#4-stage0-preprocessing)
5. [Stage 1: Standalone Embedding Models (VAE + Autoencoder)](#5-stage1-embeddings)
6. [Stage 2: SmartRec Ranking Model Training](#6-stage2-smartrec)
7. [Training Optimization Deep Dive](#7-training-optimization)
8. [Inference Pipeline: From Model to Recommendation](#8-inference)
9. [Complete Pipeline Timeline](#9-timeline)

---

## 1. System Overview: Three-Stage Pipeline {#1-system-overview}

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 0: DATA PREPROCESSING (Runs once, offline)                    │
│                                                                     │
│ BigQuery table → Dask-CUDA + NVTabular workflow → Parquet files    │
│ Scripts: train_data_processor.py, get_processed_data.py             │
│ Infra: 4-GPU Dask-CUDA cluster, 22GB RMM pool per GPU              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: STANDALONE EMBEDDING MODELS (Independent training)         │
│                                                                     │
│ 1a. Provider VAE:        train_vae.py       → provider embeddings  │
│ 1b. Provider Autoencoder: train_provider_autoencoder.py → embeddings│
│ 1c. Member Autoencoder:  train_member_autoencoder.py → embeddings  │
│                                                                     │
│ Purpose: Generate entity-level embeddings for candidate retrieval   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SMARTREC RANKING MODEL (Main model training)               │
│                                                                     │
│ Script: train_smartrec.py → SmartRecLightning wrapper               │
│ Input: (member, provider) pairs with visit labels                  │
│ Output: Ranking model that scores any (member, provider) pair       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ INFERENCE: EMBEDDING EXTRACTION                                     │
│                                                                     │
│ get_vae_embedding.py         → provider embeddings (.npy)          │
│ get_autoencoder_embedding.py → provider embeddings (.npy)          │
│ get_smartrec_embedding.py    → user + item embeddings (.npy)       │
│                                                                     │
│ These go into ANN index for candidate generation + ranking.         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Training Data Construction: Where Do the Pairs Come From? {#2-training-data}

### Data Source

The training data originates from a BigQuery table:

```
project_id:  "anbc-hcb-dev"
dataset_id:  "provider_ds_hcb_dev"
table_id:    "im_dev_traindata_202312_all_specialty_traindata"
```

This table is a **pre-materialized table of (member, provider) pairs** with historical visit labels. Based on the table name and the target columns, this data represents December 2023 historical interactions across all specialties.

### What Each Row Represents

Each row in the training data is a **(member, provider) pair** containing:

```
One row = {
    UID columns:       idx (member ID), epdb_dw_prvdr_id (provider ID), srv_location_nbr
    Member features:   5 categorical + 874 numerical features
    Provider features: 2 categorical + 7 list-type + 408 numerical features
    Pair features:     pairwise_distance, is_in_network, epdb_accepting_pts, rand_feature
    Target labels:     visited, new_visit, newvisit_3mo, visited_3mo, newvisit_6mo,
                       visited_6mo, newvisit_9mo, visited_9mo, newvisit_12mo, visited_12mo
}
```

### Target Labels: Multi-Horizon Visit Prediction

The SmartRec model predicts visit likelihood across **multiple time horizons**:

| Target Column | Meaning | Weight in Loss |
|---------------|---------|----------------|
| `visited` | Has visited this provider (historically) | **10** |
| `visited_6mo` | Visited within last 6 months | **5** |
| `visited_9mo` | Visited within last 9 months | **2** |
| `visited_12mo` | Visited within last 12 months | **2** |

The active targets in `config/smartrec/features.py` show these 4 are enabled (others like `new_visit`, `newvisit_3mo` are commented out). The asymmetric weighting `[10, 5, 2, 2]` tells the model to prioritize predicting recent visits (`visited` with weight 10) over longer-horizon visits (`visited_12mo` with weight 2).

There are also 2 additional targets with weight 1 in the `config/smartrec/vars.py` config (`target_weights: [10, 5, 2, 2, 1, 1]`), indicating 6 active targets in the latest config version. This minor discrepancy between `features.py` (4 targets) and `vars.py` (6 targets) suggests iteration — the config was expanded.

---

## 3. Negative Sampling: How the Model Learns "This Is Not a Match" {#3-negative-sampling}

This is a critical design question. Based on thorough code inspection, SmartRec uses a **multi-layered negative sampling strategy** across different levels:

### Level 1: Pre-Constructed Negatives in the Training Table (Offline)

The BigQuery table `im_dev_traindata_202312_all_specialty_traindata` contains **both positive and negative (member, provider) pairs**. This is evident from:

1. The label structure: multiple binary columns (`visited=0/1`, `visited_6mo=0/1`, etc.) — if the table contained only positive pairs, these would all be 1.
2. The triplet loss code guards for the case where `y_true.sum() > 0` AND `y_true.sum() < y_true.numel()` — confirming that batches contain a **mix** of positive and negative samples.
3. The BCE loss setup with weighted targets — weighted BCE only makes sense when you have both positive and negative examples.

The exact negative sampling ratio and strategy (random negatives, popularity-based, geographic proximity-based) is handled upstream in the BigQuery data preparation pipeline, outside this codebase. However, the `rand_feature` column in pair features suggests randomization is involved in the data construction process.

### Level 2: In-Batch Negatives for Contrastive Losses (Online, During Training)

The contrastive losses construct additional negatives **within each mini-batch** during training. This is where the most interesting negative sampling logic lives:

#### Triplet Loss — Explicit Positive/Negative Mining

From `_calculate_loss()` in `lightning_smartrec.py` (lines 500-540):

```
For each sample i in the batch:
    1. Look at y_true[i] = [visited, visited_6mo, visited_9mo, visited_12mo]
    2. pos_indices = indices where y_true[i] == 1 (positive target columns)
    3. neg_indices = indices where y_true[i] == 0 (negative target columns)
    4. Randomly pick one positive index, one negative index
    5. Use corresponding item_embeddings as pos/neg items
    
    anchor = user_embeddings[i]
    positive = item_embeddings[random positive]
    negative = item_embeddings[random negative]
    
    loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
```

**Key insight:** The triplet mining here is **within-sample** across target columns, NOT across different samples in the batch. If `visited=1` but `visited_12mo=0`, the model treats the same provider as "positive" for the first target and "negative" for the last — a nuanced temporal signal.

#### InfoNCE Loss — All-Pairs In-Batch Negatives

From `info_nce_loss.py`:

```
For multi-label scenario:
    similarity = user_embeddings × item_embeddings^T / temperature
    → [batch_size × batch_size] similarity matrix
    
    label_similarity = labels × labels^T
    → Positive pairs: share at least one label (label_similarity > 0)
    → Negative pairs: share no labels (label_similarity == 0)
    
    For each user i:
        positives = all items j where labels share overlap with user i
        negatives = all items j where labels share no overlap
        
        loss = -log( Σ exp(sim_pos) / (Σ exp(sim_pos) + Σ exp(sim_neg)) )
```

This creates O(batch_size²) implicit pair comparisons. With batch_size=12,288, that is ~150 million implicit comparisons per batch, far more than explicit triplets.

#### Ranking Contrastive Loss — Hard Negative Weighting

From `ranking_contrastive_loss.py`:

```
Like InfoNCE, but with HARD NEGATIVE emphasis:

1. Sort all items by similarity to user (descending)
2. For negative items (share no labels):
   weight = exp(similarity × lambda_hard)
   → Items that are SIMILAR to the user but NOT matches get HIGHER weight
   → "Hard negatives" = providers that look like a match but aren't
   
3. loss = -log( pos_score / (pos_score + Σ weighted_neg_scores) )

lambda_hard = 2.0  ← amplifies hard negative contribution
temperature = 0.1  ← sharper similarity distribution than InfoNCE's 0.07
```

This is particularly powerful for healthcare: a cardiologist in the wrong network or too far away looks like a great match by clinical features but should be penalized.

### Summary of Negative Sampling Strategy

```
OFFLINE (upstream SQL/data pipeline):
  ┌──────────────────────────────────────────┐
  │ Pre-materialized positive + negative     │
  │ (member, provider) pairs in BigQuery     │
  │ with multi-horizon visit labels          │
  └──────────────────────────────────────────┘

ONLINE (during SmartRec training):
  ┌──────────────────────────────────────────┐
  │ Level 1: BCE Loss                        │
  │   Direct binary classification on        │
  │   pre-labeled pairs                      │
  │                                          │
  │ Level 2: Triplet Loss                    │
  │   Within-sample cross-target mining      │
  │   Explicit anchor/pos/neg triplets       │
  │                                          │
  │ Level 3: InfoNCE Contrastive             │
  │   All-pairs in-batch comparisons         │
  │   O(batch_size²) implicit negatives      │
  │   Multi-label aware (shared labels)      │
  │                                          │
  │ Level 4: Ranking Contrastive             │
  │   Hard negative weighting                │
  │   Focuses on challenging negatives       │
  │   that look like positives               │
  └──────────────────────────────────────────┘
```

---

## 4. Stage 0: Data Preprocessing with NVTabular (GPU-Accelerated ETL) {#4-stage0-preprocessing}

### Script 1: `train_data_processor.py` — Fit the NVTabular Workflow

This runs **once** to learn the preprocessing statistics (category vocabularies, normalization parameters):

```
INFRASTRUCTURE:
  4 NVIDIA GPUs
  22GB RMM (RAPIDS Memory Manager) pool per GPU
  Dask-CUDA cluster for distributed GPU processing

WORKFLOW DEFINITION:
  uid_raw:   [idx, epdb_dw_prvdr_id, srv_location_nbr]
             → FillMissing(0) → cast int64 → TagAsUserID

  var_cont:  [874 member numerical + 408 provider numerical + 4 pair features]
             → FillMissing(0) → cast float32 → replace inf→0 → Tag CONTINUOUS

  var_cat:   [5 user cats + 2 item cats + 7 item list features]
             → Categorify(freq_threshold=1) → cast int32 → Tag CATEGORICAL

  targets:   [visited, new_visit, ..., visited_12mo]
             → cast int32 → FillMissing(0) → Tag TARGET

OUTPUT:
  NVTabular Workflow saved to GCS: gs://provider-ds-data-hcb-dev/.../workflow
  (Contains learned Categorify mappings, statistics, etc.)
```

### Script 2: `get_processed_data.py` — Apply Workflow and Save Parquet

```
1. Load the saved workflow from GCS
2. Load raw data from BigQuery (same table)
3. Transform using workflow
4. Write to Parquet:
   - 256 output files
   - PER_PARTITION shuffle
   - Separated into cats/conts/labels columns
   - Saved to GCS_DATA_PATH
```

The PER_PARTITION shuffle is a lightweight shuffle within each Dask partition, providing some randomization without a full global shuffle (which is expensive at scale).

---

## 5. Stage 1: Standalone Embedding Models (VAE + Autoencoder) {#5-stage1-embeddings}

### Stage 1a: Provider VAE (`train_vae.py`)

**What it trains:** A Variational Autoencoder on **provider-only** features (no member data).

**Data:** Provider feature parquet files from `/projects/dlrm-poc/data/processed-parquet-data/`

**Architecture:**
```
FeatureEmbeddingLayer (provider cats + numericals)
    → input_dim = 2 cats × 8 embed + 460 numericals = 476
    
Encoder: 476 → 2048 → 512 → 128 → 32 (hidden_size)
    
Reparameterize: μ, σ → z = μ + ε×σ  (latent_dim=32)
    
Decoder: 32 → 128 → 512 → 2048 → 476

Loss = MSE(reconstruction, original) + KL_divergence
```

**Training config:**
| Parameter | Value |
|-----------|-------|
| Max epochs | 8,000 |
| Batch size | 32,768 |
| Learning rate | 4e-5 |
| Optimizer | AdamW (amsgrad=True, weight_decay=0.01) |
| Scheduler | CyclicLR (triangular, step_size_up=30/down=50) |
| SWA | Enabled, starts at epoch 50, anneals 50 epochs |
| Precision | bf16-mixed |
| Gradient clipping | Not explicit (via framework) |
| Checkpoints | Every 5 epochs, keep all |

### Stage 1b: Provider Autoencoder (`train_provider_autoencoder.py`)

Same data as VAE but uses a **boosted deterministic autoencoder** instead:

```
FeatureEmbeddingLayer → 4× Encoder ensemble (nboost=4)
    → Concatenate → Optional latent projection → Decoder
    
Loss = MSE(reconstruction, original)
```

**Key difference from VAE:** No KL divergence, no reparameterization trick. The 4 boosted encoders provide diversity through multiple encoder perspectives, similar to the SmartRec internal autoencoder design.

**Training config:**
| Parameter | Value |
|-----------|-------|
| Max epochs | 8,000 |
| Batch size | 32,768 |
| Learning rate | 4e-5 |
| SWA | Enabled, starts at epoch 400, anneals 200 epochs, lr=2e-5 |
| Latent dim | 32 |

### Stage 1c: Member Autoencoder (`train_member_autoencoder.py`)

Same architecture as provider autoencoder but on **member-only** features:

```
config/member/vars.py:
  6 categorical features (gender_cd, age_band, business_ln_cd, fund_ctg_cd, state_code, mbrlang_abr)
  ~835 numerical features (dx_ctg_*, lab_*, rx_gpi2_*, prcdr_ctg_*, etc.)
  
Architecture:
  Encoder: input → 2048 → 512 → 128  (3 hidden layers, no hidden_size bottleneck)
  4× boosted encoders
  Latent layer: 128×4 = 512 → 32
  Decoder: 32 → 128 → 512 → 2048 → input
```

**Training config:**
| Parameter | Value |
|-----------|-------|
| Max epochs | 6,000 |
| Batch size | 16,384 |
| Learning rate | 4e-5 |
| Train/Test split | 93.75% / 6.25% |
| SWA | Enabled, starts at epoch 400, anneals 200 epochs |

### Key Differences Between Stage 1 Models

| Aspect | Provider VAE | Provider AE | Member AE |
|--------|-------------|-------------|-----------|
| Architecture | Single encoder + KL | 4× boosted encoders | 4× boosted encoders |
| Loss | MSE + KL divergence | MSE only | MSE only |
| Latent space | Gaussian (smooth) | Deterministic | Deterministic |
| Features | Provider only | Provider only | Member only |
| Embedding output | 32d | 32d | 32d |
| Best for | Generative retrieval | Discriminative retrieval | Member representation |

---

## 6. Stage 2: SmartRec Ranking Model Training {#6-stage2-smartrec}

### Data Loading Pipeline

```
train_smartrec.py:

1. Load NVTabular-processed parquet files from GCS or local disk
   → Separate train/test folders (pre-split, NOT runtime split)

2. Wrap in NVTabular Dataset:
   nvt.Dataset(paths, engine="parquet", part_size="256MB", cpu=False)
   → GPU-resident data, streamed in 256MB parts

3. Create TorchAsyncItr (async GPU data iterator):
   TorchAsyncItr(
       dataset,
       cats=CAT_COLS,       # 5 user + 2 item + 7 item list = 14 columns
       conts=NUM_COLS,      # 874 user + 408 item + 4 pair = 1286 columns  
       labels=target_cols,  # 4-6 target columns
       batch_size=12288,    # 1024 × 12
       shuffle=True,        # Shuffles within NVTabular partitions
   )

4. Wrap in ResumableDataLoader:
   → Extends DLDataLoader with state_dict() / load_state_dict()
   → Enables checkpoint-resumable training
   → batch_size=None (batching handled by TorchAsyncItr)
   → collate_fn=lambda x: x (no-op, data already collated by NVTabular)
   → num_workers=0 (GPU-to-GPU, no CPU workers needed)
```

### Model Instantiation

```python
smartrec_model = SmartRec(config)           # Create model
smartrec_model = AveragedModel(smartrec_model)  # Wrap for SWA
lightning_smartrec = SmartRecLightning(
    smartrec_model, config,
    batch_size=12288,
    train_dir=train_dataloader,    # Pre-built dataloader
    test_dir=test_dataloader,
    learning_rate=4e-4,
)
```

### Training Loop (Manual Optimization)

The SmartRec training step uses `automatic_optimization = False` for explicit control:

```
training_step(batch, batch_idx):
    1. Process batch → separate features (x) from targets (y)
    2. Forward pass: y_pred = model(x)
    3. Calculate all 8+ losses via _calculate_loss()
    4. Apply uncertainty weighting → total_loss
    
    5. Manual optimization:
       optimizer.zero_grad()
       self.manual_backward(total_loss)
       self.clip_gradients(optimizer, gradient_clip_val=5.0, algorithm="norm")
       optimizer.step()
       lr_scheduler.step()  ← stepped EVERY BATCH, not per epoch
```

### SmartRec Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for deep learning, weight decay built-in |
| Learning rate | 4e-4 | Moderate, explored via LR finder |
| Weight decay | 0.001 | Light regularization |
| Scheduler | CyclicLR (exp_range) | Periodically increases LR for escaping local minima |
| CyclicLR base_lr | 4e-5 (lr/10) | Lower bound of cycle |
| CyclicLR max_lr | 4e-4 (lr) | Upper bound of cycle |
| CyclicLR step_size_up | 750 (limit_batches/10) | Steps to ramp up |
| CyclicLR step_size_down | 6750 (limit_batches×9/10) | Steps to ramp down (asymmetric!) |
| CyclicLR gamma | 0.99994 | Exponential decay per cycle |
| Gradient clipping | 5.0 (norm) | Prevent gradient explosion |
| Batch size | 12,288 | Large batches for contrastive learning |
| Precision | bf16-mixed | Speed + memory savings |
| Strategy | DDP | Distributed Data Parallel across GPUs |
| Max epochs | 10 | Short, relies on large data volume |
| Limit train batches | 7,500 per epoch | Limits per-epoch iteration |
| Limit val batches | 100 | Quick validation |
| Val check interval | Every 50 steps | Frequent validation for early stopping |
| Checkpoint | Every 500 steps | Fine-grained checkpointing |
| Sync batchnorm | True | Required for multi-GPU consistency |
| Reload dataloaders | Every 50 epochs | Refresh NVTabular iterator |

### Loss Computation Sequence

Each `training_step` computes losses in this order:

```
1. BCE Loss (primary)
   → y_pred vs y_true for each target, weighted [10, 5, 2, 2, 1, 1]

2. Extract user_embeddings and item_embeddings by running towers separately

3. Effective Rank Loss
   → SVD of user and item embedding matrices
   → Maximize entropy of singular values

4. Triplet Loss (if batch has both positive and negative labels)
   → Mine within-sample pos/neg across target columns
   → Compute margin-based triplet loss

5. InfoNCE Contrastive Loss
   → All-pairs user-item similarity matrix
   → Multi-label positive/negative partitioning

6. Ranking Contrastive Loss
   → Same as InfoNCE but with hard negative weighting (lambda=2.0)

7. Orthogonality Regularization
   → Gram matrix of embeddings should approximate identity

8. Variance Preservation Loss
   → Each embedding dimension should maintain minimum variance

9. Category Alignment Loss (if specialty_ctg_cd available)
   → Covariance matrices across specialty categories should align

10. (Optional) Module-specific losses
    → Attention pattern/feature diversity losses
    → DCN variance/orthogonality losses
    → Collected from internal module buffers

11. Uncertainty Weighting (Kendall et al., 2018)
    total = Σ [ 0.5 × exp(-log_var_i) × loss_i + 0.5 × log_var_i ]
```

---

## 7. Training Optimization Deep Dive {#7-training-optimization}

### Why CyclicLR with Exponential Range?

```
Standard training:  lr decreases monotonically → gets stuck in sharp minima
CyclicLR:           lr oscillates between base and max → explores loss landscape

exp_range mode:     Each cycle's peak decreases by gamma^(cycle_iterations)
                    gamma=0.99994 → gradual decay of max_lr over training

Asymmetric step sizes:
  step_size_up   = 750  (10% of epoch)   ← fast ramp-up
  step_size_down = 6750 (90% of epoch)   ← slow annealing

This "fast climb, slow descend" pattern means:
  - The model briefly explores with high LR (10% of time)
  - Then carefully fine-tunes with decreasing LR (90% of time)
```

### Why Stochastic Weight Averaging (SWA)?

For Stage 1 models (VAE/AE), SWA is explicitly enabled:

```
StochasticWeightAveraging(
    swa_lrs=learning_rate/2,  # SWA at half the base LR
    swa_epoch_start=50-400,   # Start averaging after initial training
    annealing_epochs=50-200,  # Gradually reduce LR during SWA
    annealing_strategy="cos"  # Cosine annealing
)
```

For Stage 2 (SmartRec), SWA is applied via `AveragedModel` wrapping:
```python
smartrec_model = AveragedModel(SmartRec(config))
```
This maintains a running average of model weights alongside the training weights, producing a smoother loss landscape and better generalization.

### bf16-mixed Precision Strategy

All training scripts use `precision="bf16-mixed"`:
- Forward pass: bf16 for speed (16-bit brain float)
- Loss computation: fp32 for accuracy  
- Backward pass: fp32 gradients
- Weight updates: fp32 master weights

This roughly halves GPU memory and doubles throughput with minimal accuracy loss.

### TF32 CUDA Optimization

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

TF32 (TensorFloat-32) uses 10-bit mantissa for matrix multiplications on Ampere+ GPUs. Combined with bf16-mixed, this provides ~3x speedup over pure fp32.

### Flash Attention (SmartRec only)

```python
if torch.backends.cuda.is_flash_attention_available():
    torch.backends.cuda.enable_flash_sdp(True)
```

When available, SmartRec enables Flash Attention for its attention layers, reducing memory from O(n²) to O(n) and improving throughput significantly.

---

## 8. Inference Pipeline: From Model to Recommendation {#8-inference}

### Stage 1 Inference: Entity Embedding Extraction

#### `get_vae_embedding.py` — Extract Provider Embeddings via VAE

```
1. Load trained VAE from checkpoint
   (e.g., Provider_VAE_epoch=2499-train_loss=-0.02519.ckpt)

2. Load ALL provider data via Merlin Dataset + Loader
   batch_size=40960 (much larger than training — no backprop needed)

3. For each batch:
   x = model.embedding_layer(batch)     # Raw features → embeddings
   h = model.encoder(x)                 # Encode to bottleneck
   
   if embedding_type == "latent":
       mu = model.fc_mu(h)
       logvar = model.fc_logvar(h)
       z = model.reparameterize(mu, logvar)  # Sample from latent
       embedding = z   (32d)
   else:  # "encoder"
       embedding = h   (32d, deterministic)
   
   Prepend ID columns: [specialty_ctg_cd, epdb_dw_prvdr_id, srv_location_nbr, tax_id_nbr]

4. Save: np.save("vae_embeddings.npy", all_embeddings)
```

#### `get_autoencoder_embedding.py` — Extract Provider Embeddings via AE

Same flow but using the boosted Autoencoder:
```
h = model.encode(batch)       # FeatureEmbedding → 4×Encoder → concat
if latent_layer:
    h = model.latent_layer(h) # Project to 32d
```

### Stage 2 Inference: SmartRec Embedding Extraction

`get_smartrec_embedding.py` extracts **both user and item embeddings** from the trained ranking model:

```
1. Load SmartRec model from checkpoint
   smartrec_model = SmartRec(config)
   smartrec_model = AveragedModel(smartrec_model)
   lightning = SmartRecLightning.load_from_checkpoint(ckpt, model=smartrec, config=config)
   model = lightning.model.module  # Unwrap AveragedModel

2. Wrap with DataParallel for multi-GPU inference
   model = torch.nn.DataParallel(model)
   batch_size = 81920 × 2 = 163,840

3. For each batch, extract BOTH tower outputs:
   one_tower(user_model, batch) → 32d user_latent
   one_tower(item_model, batch) → 32d item_latent
   
   Where one_tower() does:
     embedded = model.embedding_layer(batch)
     → pre_attention_layer
     → attention_layers (with residual connections)
     → residual_dcn (parallel path)
     → concatenate attention + dcn
     → mmoe
     → autoencoder.encode() → 32d
   
4. Prepend ID columns:
   user: [idx] + 32d embedding
   item: [epdb_dw_prvdr_id, srv_location_nbr, specialty_ctg_cd] + 32d embedding

5. Save separately:
   np.save("smartrec_embeddings_user.npy", user_embeddings)
   np.save("smartrec_embeddings_item.npy", item_embeddings)
```

### Deployment Use: How These Become Recommendations

```
CANDIDATE GENERATION (from Stage 1 embeddings):
  1. All provider AE/VAE embeddings → loaded into ANN index (e.g., FAISS, ScaNN)
  2. For incoming member → compute member AE embedding
  3. ANN search → top-K candidate providers (e.g., K=100-500)

RANKING (from Stage 2 model):
  4. For each (member, candidate_provider) pair:
     → Run full SmartRec forward pass
     → Get [visited, visited_6mo, visited_9mo, visited_12mo] probabilities
  5. Score = weighted combination of multi-horizon predictions
  6. Sort by score → Final ranked recommendation list

ALTERNATIVELY (from Stage 2 embeddings):
  4'. Load pre-computed SmartRec user/item embeddings
  5'. Cosine similarity between user embedding and item embeddings
  6'. This is faster but less accurate (skips cross-attention, pair features, DCN)
```

---

## 9. Complete Pipeline Timeline {#9-timeline}

```
DAY 1-2: DATA PREPARATION
  ┌─────────────────────────────────────────────┐
  │ BigQuery materialization (SQL, upstream)     │
  │ train_data_processor.py: Fit NVTabular      │
  │ get_processed_data.py: Transform → Parquet  │
  └─────────────────────────────────────────────┘

DAY 3-7: STAGE 1 TRAINING (can run in parallel)
  ┌────────────────────────┐
  │ Provider VAE           │ 8000 epochs, batch=32K
  │ train_vae.py           │ ~3-5 days on multi-GPU
  └────────────────────────┘
  ┌────────────────────────┐
  │ Provider Autoencoder   │ 8000 epochs, batch=32K
  │ train_provider_ae.py   │ ~3-5 days
  └────────────────────────┘
  ┌────────────────────────┐
  │ Member Autoencoder     │ 6000 epochs, batch=16K
  │ train_member_ae.py     │ ~3-4 days
  └────────────────────────┘

DAY 5-7: STAGE 2 TRAINING
  ┌────────────────────────┐
  │ SmartRec               │ 10 epochs × 7500 batches = 75K steps
  │ train_smartrec.py      │ batch=12K, bf16, DDP
  │                        │ ~1-2 days on multi-GPU
  └────────────────────────┘

DAY 8: INFERENCE / EMBEDDING EXPORT
  ┌────────────────────────┐
  │ get_vae_embedding.py   │ batch=40K, minutes
  │ get_ae_embedding.py    │ batch=40K, minutes
  │ get_smartrec_embed.py  │ batch=160K, DataParallel
  └────────────────────────┘

DAY 8+: DEPLOYMENT
  ┌────────────────────────┐
  │ Load embeddings → ANN  │
  │ Serve recommendations  │
  └────────────────────────┘
```

### Training Parameters Comparison Across All Stages

| Parameter | Provider VAE | Provider AE | Member AE | SmartRec |
|-----------|-------------|-------------|-----------|----------|
| Script | `train_vae.py` | `train_provider_autoencoder.py` | `train_member_autoencoder.py` | `train_smartrec.py` |
| Config | `config/vars.py` | `config/vars.py` | `config/member/vars.py` | `config/smartrec/vars.py` |
| Epochs | 8,000 | 8,000 | 6,000 | 10 |
| Batch size | 32,768 | 32,768 | 16,384 | 12,288 |
| Learning rate | 4e-5 | 4e-5 | 4e-5 | 4e-4 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Scheduler | CyclicLR triangular | CyclicLR triangular | CyclicLR triangular | CyclicLR exp_range |
| SWA | Yes (epoch 50) | Yes (epoch 400) | Yes (epoch 400) | AveragedModel |
| Precision | bf16-mixed | bf16-mixed | bf16-mixed | bf16-mixed |
| Strategy | auto | auto | auto | DDP |
| Loss | MSE + KL | MSE | MSE | 8 losses + uncertainty |
| Gradient clip | — | norm=1.0 | norm=1.0 | norm=5.0 |
| Features | Provider only | Provider only | Member only | Member + Provider + Pair |
| Data entity | Provider rows | Provider rows | Member rows | (Member, Provider) pairs |
| Output dim | 32 | 32 | 32 | 32 (per tower) |
