# Transformer Model Architecture and Encoding Guide

## Introduction

This doc provides a detailed technical explanation of how the clinical transformer model processes member's claims data to generate patient embeddings. The transformer uses a multi-level encoding architecture to capture both daily medical events and temporal patterns across a patient's medical history.

## Model Architecture Overview

### Core Architecture Components

The transformer model (`TransformerModel` class) implements a **hierarchical encoding strategy** with the following key components:

```python
class TransformerModel(nn.Module):
    def __init__(self, nhead, nhid, nlayers, dropout=0.05):
        # Embedding layers for different input types
        self.embedding_cd = nn.Embedding(cd_cnt, embedding_size)           # Medical codes
        self.embedding_gender_cd = nn.Embedding(4, embedding_size)         # Gender
        self.embedding_age_in_months = nn.Embedding(1440, embedding_size)  # Age
        
        # Two-level transformer architecture
        self.transformer_encoder_cd = TransformerEncoder(...)  # Daily code encoder
        self.transformer_encoder_dy = TransformerEncoder(...)  # Temporal sequence encoder
        
        # Output processing
        self.decoder_cd = nn.Linear(embedding_size, target_cd_cnt)
```

**Key Parameters:**
- **Embedding Size**: 256 dimensions for all embeddings
- **Sequence Length**: 200 days maximum
- **Codes per Day**: 80 medical codes maximum
- **Vocabulary Size**: 84,010 (Com_ME) or 98,041 (Medicaid) unique medical codes
- **Target Codes**: 2,767 (Com_ME) or 2 (Medicaid) prediction targets

## Hierarchical Encoding Strategy

### Level 1: Daily Code Encoding (`transformer_encoder_cd`)

**Purpose**: Encode multiple medical codes within a single day into a unified daily representation. This layer captures relationships between codes that co-occur on the same day, and compresses up to 80 codes into a single vector.

```python
# Daily transformer configuration
encoder_layers_cd = TransformerEncoderLayer(
    d_model=embedding_size,    # 256 dimensions
    nhead=4,                   # 4 attention heads
    dim_feedforward=embedding_size,  # 256 FFN dimensions
    dropout=0                  # No dropout for daily encoding
)
self.transformer_encoder_cd = TransformerEncoder(encoder_layers_cd, num_layers=1)
```

**Input Processing:**
```python
# Input shape: [batch_size, 200_days, 80_codes]
cd = x[:,:,2:]  # Extract medical codes (skip age/gender)
cd = self.embedding_cd(cd)  # [batch, 200, 80, 256] - embed each code

# Reshape for daily processing: combine batch and day dimensions
cd = cd.reshape(gpu_batchsize*len_dy, len_cd, embedding_size)  # [batch*200, 80, 256]
cd = torch.swapaxes(cd, 0, 1)  # [80, batch*200, 256] - sequence first for transformer
```

**Daily Attention Mechanism:**
```python
# Self-attention across codes within each day
cd = self.transformer_encoder_cd(cd)  # [80, batch*200, 256]

# Aggregate codes into single daily representation
cd = cd.permute(1,2,0)  # [batch*200, 256, 80]
cd = nn.MaxPool1d(len_cd)(cd)  # [batch*200, 256, 1] - max pooling across codes
cd = cd.reshape(gpu_batchsize, len_dy, embedding_size)  # [batch, 200, 256]
```

### Level 2: Temporal Sequence Encoding (`transformer_encoder_dy`)

**Purpose**: Encode the sequence of daily representations across the 200-day timeline to capture disease progression and treatment patterns, modeling how medical conditions evolve over time. It uses causal masking to ensure the model only attends to past days and understands progression patterns and episodic care.

```python
# Temporal transformer configuration
encoder_layers_dy = TransformerEncoderLayer(
    d_model=embedding_size,    # 256 dimensions
    nhead=16,                  # 16 attention heads (more complex temporal relationships)
    dim_feedforward=512,       # 512 FFN dimensions (larger for complex patterns)
    dropout=0.1                # Dropout for regularization
)
self.transformer_encoder_dy = TransformerEncoder(encoder_layers_dy, num_layers=6)
```

**Temporal Input Preparation:**
```python
# Combine daily codes with demographics
cd_res = cd.sum(-2)  # Residual connection from original embeddings
cd = cd_res + cd + gender_cd + age_in_months  # [batch, 200, 256]

# Apply activation and normalization
cd = self.mm(cd)      # GELU activation
cd = self.norm(cd)    # Layer normalization
cd = torch.swapaxes(cd, 0, 1)  # [200, batch, 256] - sequence first
```

**Causal Masking for Temporal Ordering:**
```python
def _generate_square_subsequent_mask(self, sz):
    # Create causal mask to prevent looking at future days
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Apply mask during forward pass
mth_mask = self._generate_square_subsequent_mask(len_dy).to(device)
cd = self.transformer_encoder_dy(cd, mth_mask)  # [200, batch, 256]
```

## Embedding Strategy Details

### Input Embeddings

#### 1. Medical Code Embeddings
```python
self.embedding_cd = nn.Embedding(cd_cnt, embedding_size)
# Maps integer code IDs (1-84010) to 256-dimensional vectors
# Learned representations capture semantic relationships between medical codes
```

**Example Learned Relationships:**
- Diabetes codes (250.x) cluster together in embedding space
- Related procedures (glucose tests) positioned near diabetes diagnoses
- Similar medications grouped by therapeutic class

#### 2. Demographic Embeddings
```python
# Gender embedding
self.embedding_gender_cd = nn.Embedding(4, embedding_size)
# 4 categories: 0=Female, 1=Male, 2=Other, 3=Unknown

# Age embedding  
self.embedding_age_in_months = nn.Embedding(1440, embedding_size)
# 1440 = 120 years * 12 months, captures age-related medical patterns
```

**Demographic Integration:**
```python
# Demographics are added to each day's representation
cd = cd_res + cd + gender_cd + age_in_months
# This allows the model to condition medical interpretations on patient demographics
```

### Output Embeddings and Final Representation

#### Patient History Embedding Extraction

The final patient embedding is extracted from the **temporal transformer's output** at the last actual day of data:

```python
# During inference (score function)
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.transformer_encoder_dy.register_forward_hook(get_activation('transformer_encoder_dy'))

# Extract embedding from the last day with actual data
intermedia_output = activation['transformer_encoder_dy']
intermedia_output = [intermedia_output[dt_cnt[i], i, :].reshape(1, -1) for i in range(batch_size)]
```

## Training Strategies
I'll read the current `Understand_transformer.md` file to see the existing structure and then provide you with the text to add under the "Training Strategies" section.

[1 tool called]

Perfect! I can see the document already has a "## Training Strategies" section header at line 169, but it's empty. Here's the text you should add to complete that section:

---


I'll trace through the code to show you exactly how the autoregressive multi-label prediction works. Let me read the relevant sections.

[8 tools called]

Now let me look at how the loss is computed and how targets align with predictions:

[2 tools called]

Now I have all the details. Let me create a comprehensive step-by-step explanation based on the actual code.

---

# Complete Step-by-Step: How the Model Uses Daily Codes to Predict Future Codes

## 🎯 IMPORTANT CLARIFICATION: Same-Day vs Next-Day Prediction

Based on the code, the model actually performs **same-day reconstruction** (NOT next-day prediction). Let me show you exactly what this means:

```
Input day t:    [age_t, gender_t, lob_t, code1_t, code2_t, ..., code80_t]
Target day t:   [target_code1_t, target_code2_t, ...]  ← SAME day codes

NOT:
Input day t:    → predicting codes on day t+1
```

The **causal masking** ensures that when predicting day t, the model can only see days 1 through t (not t+1 onwards), but it's still predicting codes that occurred on day t itself.

---

## 📋 Concrete Example: End-to-End Pipeline

### Step 0: Raw Data Format

Let's trace a **real example patient** through the system.

**Raw Data in DataFrame:**
```python
# Example patient row from training data
{
    'individual_id': 'MEMBER_12345',
    'age_in_months': "720*720*721*721*722",  # Age at each day (60 years → 60 years 2 mo)
    'gender_cd': "1*1*1*1*1",                 # Male (1) all days
    'lob': "3*3*3*3*3",                       # Medicaid (3) all days
    'cd': "45001,45002,12003*45001*45001,67890,11111*45001*45001,45002",  # Medical codes per day
    'target': "6001,6002*6003*6004,6005,6006*6007*6008,6009",  # Target codes per day
    'dt_cnt': 5                               # 5 valid days in this sequence
}
```

**Visual representation:**
```
Day 0: codes=[45001,45002,12003]  → targets=[6001,6002]  (2 codes to predict)
Day 1: codes=[45001]             → targets=[6003]       (1 code)
Day 2: codes=[45001,67890,11111] → targets=[6004,6005,6006]  (3 codes)
Day 3: codes=[45001]             → targets=[6007]       (1 code)
Day 4: codes=[45001,45002]       → targets=[6008,6009]  (2 codes)
```

---

### Step 1: Data Preprocessing (`conv_*` functions)

**From code lines 3449-3478:**

```python
def conv_target(target: str, len_dy: int, target_cd_cnt: int) -> List[List[int]]:
    """
    Convert target string to nested list (multi-label).
    
    CRITICAL: Each day can have multiple target codes!
    Format: "code1,code2*code3*code4,code5,code6*..."
    
    Returns: List[List[int]] where each inner list contains all codes for that day
    """
```

**For our example:**
```python
target_str = "6001,6002*6003*6004,6005,6006*6007*6008,6009"

# After conv_target:
targets = [
    [6000, 6001],           # Day 0: codes 6001,6002 → 0-indexed: [6000, 6001]
    [6002],                 # Day 1: code 6003 → [6002]
    [6003, 6004, 6005],     # Day 2: codes 6004,6005,6006 → [6003, 6004, 6005]
    [6006],                 # Day 3: code 6007 → [6006]
    [6007, 6008],           # Day 4: codes 6008,6009 → [6007, 6008]
]
# Note: codes are converted to 0-based indices (subtract 1)
```

**Similarly for input codes (`conv_cd`):**
```python
codes = [
    [45000, 45001, 12002, 0, 0, ..., 0],  # Day 0: padded to 80 codes
    [45000, 0, 0, ..., 0],                 # Day 1
    [45000, 67889, 11110, 0, 0, ..., 0],   # Day 2
    [45000, 0, 0, ..., 0],                 # Day 3
    [45000, 45001, 0, 0, ..., 0],          # Day 4
    [0, 0, 0, ..., 0],                     # Days 5-199: all zeros (padding)
    ...
]
# Shape: [200, 80]
```

---

### Step 2: Collate Function Creates Batch (lines 3195-3251)

```python
def clinical_collate_fn(batch: List[Dict], config: 'BaseConfig') -> Dict[str, Any]:
    """
    Pre-compute multi-hot targets as tensors.
    """
    # Pre-compute multi-hot targets: [batch, len_dy, target_cd_cnt]
    targets_multihot = torch.zeros(batch_size, len_dy, target_cd_cnt, dtype=torch.float16)
    
    for i, item in enumerate(batch):
        target_list = item['target']  # List[List[int]] - len_dy x variable
        for day_idx, day_codes in enumerate(target_list):
            if day_idx < len_dy and day_codes:
                for code_idx in day_codes:
                    if 0 <= code_idx < target_cd_cnt:
                        targets_multihot[i, day_idx, code_idx] = 1.0
```

**For our example patient (patient 0 in batch):**
```
targets_multihot[0] = Tensor of shape [200, 6297]

# Day 0:
targets_multihot[0, 0, :] = [0,0,0,...,0,1,1,0,...,0]
                                        ^  ^
                                   idx 6000, 6001 = 1.0

# Day 1:
targets_multihot[0, 1, :] = [0,0,0,...,0,0,0,1,0,...,0]
                                            ^
                                       idx 6002 = 1.0

# Day 2:
targets_multihot[0, 2, :] = [0,0,0,...,1,1,1,0,...,0]
                                      ^  ^  ^
                                 idx 6003,6004,6005 = 1.0
# ... and so on
```

---

### Step 3: Model Input Construction (train_epoch lines 5032-5042)

```python
# Combine all features into input tensor
x = torch.cat([
    age.unsqueeze(-1),      # [batch, 200, 1]
    gender.unsqueeze(-1),   # [batch, 200, 1]
    lob.unsqueeze(-1),      # [batch, 200, 1]
    codes                   # [batch, 200, 80]
], dim=-1)
# Result: x.shape = [batch, 200, 83]
```

**For our example:**
```
x[0] = Tensor of shape [200, 83]

# Day 0 row:
x[0, 0, :] = [720,    1,    3,    45000, 45001, 12002, 0, 0, ..., 0]
              ^       ^     ^     ^-------------------------^
              age   gender  lob          80 medical codes

# Day 4 row:
x[0, 4, :] = [722,    1,    3,    45000, 45001, 0, 0, ..., 0]
```

---

### Step 4: Embedding Layer (FlashAttentionTransformer.forward lines 2552-2557)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    gpu_batchsize = x.shape[0]
    actual_len_dy = x.shape[1]
    actual_len_cd = x.shape[2] - 3  # = 80
    
    # Extract and embed
    age_in_months = self.embedding_age_in_months(x[:, :, 0].long())   # [batch, 200, 256]
    gender_cd = self.embedding_gender_cd(x[:, :, 1].long())           # [batch, 200, 256]
    lob_emb = self.embedding_lob(x[:, :, 2].long())                   # [batch, 200, 256]
    cd = self.embedding_cd(x[:, :, 3:].long())                        # [batch, 200, 80, 256]
    
    cd_res = cd.sum(-2)  # Sum across codes dimension → [batch, 200, 256]
```

**For our example (Day 0):**
```
x[0, 0, 0] = 720 (age in months)
    ↓
age_embedding[720] → 256-dim vector representing "60 years old"

x[0, 0, 3:] = [45000, 45001, 12002, 0, 0, ...]
    ↓
code_embeddings = [
    embed(45000),  # 256-dim
    embed(45001),  # 256-dim  
    embed(12002),  # 256-dim
    embed(0),      # PAD embedding
    ...
]
# Shape: [80, 256]

cd_res[0, 0] = sum of all 80 code embeddings → [256]
```

---

### Step 5: Daily Encoder - Learned Attention Pooling (lines 2564-2570)

```python
if self.config.use_learnt_att_pool:
    # Learned attention pooling (replaces transformer + max-pool)
    cd = self.daily_pooling(cd)  # [batch*len_dy, embedding_size]
    cd = self.daily_mlp(cd)
    cd = self.daily_norm(cd)
```

**How LearnedAttentionPooling works (lines 1684-1761):**
```python
class LearnedAttentionPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool 80 code embeddings into 1 day embedding.
        
        Args:
            x: [80, batch*200, 256] - code embeddings
        
        Returns:
            pooled: [batch*200, 256] - one vector per day
        """
        # Learnable query: "What should I focus on today?"
        q = self.query  # [1, 256] - learned during training
        
        # Compute attention over 80 codes
        scores = q @ x.T / sqrt(256)  # [1, 80] attention weights
        attn_weights = softmax(scores)  # Sum to 1
        
        # Weighted sum: codes with higher attention get more weight
        pooled = attn_weights @ x  # [1, 256]
```

**For Day 0 of our example:**
```
Input: 80 code embeddings (including 77 padding zeros)
       [embed(45000), embed(45001), embed(12002), PAD, PAD, ...]

Attention learns: "Focus on diabetes (45000) and hypertension (45001)"
       weights = [0.45, 0.35, 0.15, 0.01, 0.01, ...] 

Output: Weighted combination → [256] vector representing "Day 0"
```

---

### Step 6: Combine All Representations (lines 2598-2604)

```python
# Reshape back to [batch, 200, 256]
cd = cd.reshape(gpu_batchsize, actual_len_dy, self.config.embedding_size)

# Add all embeddings together
cd = cd_res + cd + gender_cd + age_in_months + lob_emb
# Each is [batch, 200, 256]

cd = self.mm(cd)   # GELU activation
cd = self.norm(cd) # LayerNorm
cd = torch.swapaxes(cd, 0, 1)  # → [200, batch, 256] for temporal encoder
```

**For our example:**
```
day_0_vector = (
    sum_of_code_embeddings      # What happened today (codes)
    + pooled_code_attention     # Learned summary of today
    + gender_embedding          # Male patient context
    + age_embedding             # 60 years old context
    + lob_embedding             # Medicaid context
)
# Result: [256] vector capturing all information about Day 0
```

---

### Step 7: Temporal Encoder with CAUSAL MASKING 🔑 (lines 2606-2618)

This is where **autoregression** happens:

```python
# Temporal encoding with Flash Attention
for layer in self.temporal_layers:  # 6 layers
    # Pre-norm attention block
    residual = cd
    cd_norm = layer['norm1'](cd)
    cd_attn = layer['attention'](cd_norm, is_causal=True)  # ← CAUSAL MASKING
    cd = residual + cd_attn
    
    # Pre-norm FFN block
    residual = cd
    cd_norm = layer['norm2'](cd)
    cd_ffn = layer['ffn'](cd_norm)
    cd = residual + cd_ffn
```

**How causal masking works (lines 1635-1642):**

```python
def _standard_attention(self, q, k, v, mask, is_causal):
    scale = 1.0 / math.sqrt(self.head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if is_causal:
        # Create causal mask - upper triangular = -inf
        seq_len = q.size(2)  # 200 days
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device),
            diagonal=1
        ).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
```

**Visual of Causal Mask (200×200):**
```
                Day 0  Day 1  Day 2  Day 3  Day 4  ...  Day 199
Query Day 0:    [1.0,  -inf,  -inf,  -inf,  -inf, ...,  -inf ]
Query Day 1:    [0.3,   0.7,  -inf,  -inf,  -inf, ...,  -inf ]
Query Day 2:    [0.2,   0.3,   0.5,  -inf,  -inf, ...,  -inf ]
Query Day 3:    [0.1,   0.2,   0.3,   0.4,  -inf, ...,  -inf ]
Query Day 4:    [0.1,   0.15,  0.2,   0.25,  0.3, ...,  -inf ]
...
Query Day 199:  [0.01, 0.01, 0.01, 0.01, 0.01, ...,   0.5  ]

-inf → softmax → 0 (cannot attend to future days)
```

**For our example - What Day 4 can "see":**
```
When computing representation for Day 4:
✓ Can attend to: Day 0, Day 1, Day 2, Day 3, Day 4
✗ Cannot attend to: Day 5, Day 6, ..., Day 199

Attention weights after softmax:
Day 4 attends to = [
    Day 0 (weight 0.15): "Saw diabetes diagnosis"
    Day 1 (weight 0.10): "Routine visit"
    Day 2 (weight 0.25): "Multiple procedures" 
    Day 3 (weight 0.20): "Follow-up"
    Day 4 (weight 0.30): "Current day - strongest signal"
]
```

---

### Step 8: Output Projection → Predictions (lines 2620-2627)

```python
# Output projection
cd = torch.swapaxes(cd, 0, 1)  # → [batch, 200, 256]

cd = self.norm(cd)
cd = self.dropout(cd)
cd = self.decoder_cd(cd)  # Linear: 256 → 6297

return cd  # [batch, 200, 6297] - RAW LOGITS (no softmax!)
```

**For our example:**
```
output[0] = Tensor of shape [200, 6297]

# Day 0 predictions (can only see Day 0):
output[0, 0, :] = [-2.1, -1.5, ..., 3.2, 2.8, ..., -0.5]
                                    ^    ^
                               logit for code 6000, 6001
                               (higher = more likely)

# Day 4 predictions (can see Days 0-4):
output[0, 4, :] = [-1.8, -1.2, ..., 2.9, 2.1, ..., 3.5, 2.8]
                                                   ^    ^
                                              logit for 6007, 6008
```

---

### Step 9: Loss Computation with BCEWithLogitsLoss (DataParallelWrapper lines 1085-1110)

```python
# In DataParallelWrapper.forward():

# Flatten for loss computation
output_flat = output.view(batch_size * actual_len_dy, self.target_cd_cnt)
# Shape: [batch*200, 6297]

targets_flat = targets.view(batch_size * actual_len_dy, self.target_cd_cnt)
# Shape: [batch*200, 6297]

# Create valid day mask (only compute loss for real days, not padding)
valid_mask = torch.zeros(batch_size * actual_len_dy, dtype=torch.bool, device=device)

for i in range(batch_size):
    valid_days = min(int(dt_cnt[i].item()), actual_len_dy)
    if valid_days > 0:
        start_idx = i * actual_len_dy
        valid_mask[start_idx:start_idx + valid_days] = True

# Compute loss only on valid positions
valid_output = output_flat[valid_mask]   # [total_valid_days, 6297]
valid_targets = targets_flat[valid_mask] # [total_valid_days, 6297]

pred_loss = self.criterion(valid_output, valid_targets)  # BCEWithLogitsLoss
```

**For our example (dt_cnt=5):**
```
valid_mask for patient 0: [True, True, True, True, True, False, False, ..., False]
                           ^     ^     ^     ^     ^      ^
                          Day0  Day1  Day2  Day3  Day4   Day5-199 (padding)

Loss computed only on Days 0-4:
- Day 0: BCE(logits for 6297 codes, target=[0,0,...,1,1,...,0])
- Day 1: BCE(logits for 6297 codes, target=[0,0,...,0,0,1,...,0])
- ...
- Day 4: BCE(logits for 6297 codes, target=[0,0,...,0,0,0,...,1,1])
```

---

## 🔄 Summary: The Complete Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE AUTOREGRESSIVE FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAW DATA                                                                   │
│  ────────                                                                   │
│  age="720*720*721"  gender="1*1*1"  lob="3*3*3"                            │
│  cd="45001,45002*45001*45001,67890"                                        │
│  target="6001,6002*6003*6004,6005"                                         │
│                                                                             │
│         ↓ ClinicalDataset + collate_fn                                     │
│                                                                             │
│  TENSORS                                                                    │
│  ───────                                                                    │
│  x: [batch, 200, 83]         # age + gender + lob + 80 codes               │
│  targets: [batch, 200, 6297] # multi-hot encoding                          │
│  dt_cnt: [batch]             # valid days per patient                      │
│                                                                             │
│         ↓ Embedding Layers                                                 │
│                                                                             │
│  EMBEDDINGS                                                                 │
│  ──────────                                                                 │
│  age_emb: [batch, 200, 256]                                                │
│  gender_emb: [batch, 200, 256]                                             │
│  lob_emb: [batch, 200, 256]                                                │
│  code_emb: [batch, 200, 80, 256]                                           │
│                                                                             │
│         ↓ Daily Encoder (Learned Pooling)                                  │
│                                                                             │
│  DAY REPRESENTATIONS                                                        │
│  ───────────────────                                                        │
│  day_vectors: [batch, 200, 256]                                            │
│  = code_sum + pooled + age + gender + lob                                  │
│                                                                             │
│         ↓ Temporal Encoder (6 layers, CAUSAL MASK)                         │
│                                                                             │
│  CONTEXTUALIZED REPRESENTATIONS                                             │
│  ───────────────────────────────                                            │
│  For Day t: attends to Days 0, 1, 2, ..., t ONLY                           │
│  Cannot see Days t+1, t+2, ..., 199                                        │
│                                                                             │
│         ↓ Output Projection (Linear 256 → 6297)                            │
│                                                                             │
│  RAW LOGITS                                                                 │
│  ──────────                                                                 │
│  output: [batch, 200, 6297]                                                │
│  Each position: 6297 independent binary predictions                        │
│                                                                             │
│         ↓ BCEWithLogitsLoss (sigmoid + binary cross-entropy)               │
│                                                                             │
│  LOSS                                                                       │
│  ────                                                                       │
│  For each day t, for each code c:                                          │
│    loss += -[y_c * log(σ(logit_c)) + (1-y_c) * log(1-σ(logit_c))]         │
│                                                                             │
│  Only computed for valid days (dt_cnt)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Insight: Why It's "Autoregressive" but Same-Day Prediction

The model is **autoregressive in its attention mechanism** but performs **same-day reconstruction**:

| Aspect | What Happens |
|--------|--------------|
| **Autoregressive Attention** | Day t can only attend to Days 0...t (cannot peek at future) |
| **Prediction Target** | Predict codes that occurred on Day t (same day) |
| **Multi-label** | Each day can have multiple true codes (BCEWithLogitsLoss) |
| **Training Signal** | Learn patterns: "Given patient history up to day t, what codes are on day t?" |

**Why this design?**
- The model learns to recognize: "A patient with diabetes history who presents with chest pain today → likely has codes for cardiac workup today"
- It's learning the **clinical context** that predicts what happens on a given day
- The causal mask prevents "cheating" by looking at future days during training



# 🔍 Is This Model Autoregressive?

## ⚠️ SHORT ANSWER: **Partially Yes, But Not Truly Autoregressive**

The model has **autoregressive architectural components** (causal masking) but is **NOT a true autoregressive generative model** due to a fundamental vocabulary mismatch.

---

## 📋 Evidence-Based Analysis

### ✅ What Makes It LOOK Autoregressive (Architecture)

**Evidence 1: Causal Masking in Temporal Encoder**

```2702:2703:dev/moe/moe_flashattn_3.py
x_attn = layer_dict['attention'](x_norm, is_causal=True)
```

The temporal encoder uses `is_causal=True`, which creates a lower-triangular attention mask:

```
Position:    0   1   2   3   4
Day 0:      [✓] [✗] [✗] [✗] [✗]   ← Can only see itself
Day 1:      [✓] [✓] [✗] [✗] [✗]   ← Can see day 0-1
Day 2:      [✓] [✓] [✓] [✗] [✗]   ← Can see day 0-2
Day 3:      [✓] [✓] [✓] [✓] [✗]   ← Can see day 0-3
Day 4:      [✓] [✓] [✓] [✓] [✓]   ← Can see all previous
```

**Evidence 2: Sequential Prediction**

At each position t, the model produces a prediction for the next day (t+1) using only information from days 0...t.

---

### ❌ What Makes It NOT Truly Autoregressive

**Evidence 1: Vocabulary Mismatch (CRITICAL)**

```358:360:dev/moe/moe_flashattn_3.py
cd_cnt: int = 75516        # Input vocabulary size
target_cd_cnt: int = 6297  # Target vocabulary (updated from 2767, 8850)
```

| Vocabulary | Size | Example |
|------------|------|---------|
| **Input (cd_cnt)** | 75,516 codes | `E11.65`, `99213`, `gpi2210` |
| **Output (target_cd_cnt)** | 6,297 codes | `icd9_dx_cdE11`, `prcdr_group_992`, `gpi22` |

**The model CANNOT feed its outputs back as inputs because they're in different vocabularies!**

This is fundamentally different from GPT, where:
- Input vocabulary = Output vocabulary = same token set
- Next token prediction can be fed back as input

**Evidence 2: No Recurrent/Iterative Generation**

The model processes the **entire input sequence at once** and produces predictions for all positions **in parallel**:

```python
# Forward pass produces all predictions simultaneously
output = model(x)  # Shape: [batch, len_dy, target_cd_cnt]
# All 200 days predicted in one forward pass
```

A true autoregressive model would:
1. Predict position 1
2. Feed prediction back as input
3. Predict position 2
4. Repeat...

**Evidence 3: Teacher Forcing Only (No Generation Mode)**

During training, the model uses **teacher forcing**:
- Ground truth inputs (days 0...t) are provided
- Model predicts target for day t+1
- Model never sees its own predictions as input

There's no evidence of a generation/inference mode that feeds outputs back.

---

## 🏷️ Correct Classification

| Property | True Autoregressive (GPT) | This Model |
|----------|---------------------------|------------|
| Causal masking | ✅ Yes | ✅ Yes |
| Same input/output vocab | ✅ Yes | ❌ No (75k vs 6k) |
| Can feed output → input | ✅ Yes | ❌ No |
| Generates sequentially | ✅ Yes | ❌ No (parallel) |
| Uses own predictions | ✅ Yes | ❌ No (teacher forcing) |

**This model is more accurately described as:**

> **A Causal Sequence-to-Sequence Model with Multi-Label Classification**

Or:

> **A Masked Language Model with Causal Constraints for Classification**

---

## 📊 Visual Comparison

```
TRUE AUTOREGRESSIVE (GPT):
──────────────────────────
Input vocab = Output vocab = 50,000 tokens

Training:   [The] [cat] [sat] → predict [on]
Inference:  [The] → [cat] → [sat] → [on] → [the] → [mat]
                 ↑       ↑        ↑       ↑        ↑
            (feed back predictions as input)


THIS CLINICAL MODEL:
────────────────────
Input vocab = 75,516 detailed codes
Output vocab = 6,297 grouped codes

Training:   [codes_day0] [codes_day1] [codes_day2] → predict [grouped_targets]
            (all at once, teacher forcing)

Inference:  [codes_day0...day199] → [predictions for all 200 days]
            (single forward pass, no feedback loop)
            
            ❌ Cannot do: predict_day1 → feed_back → predict_day2
               because output (grouped) ≠ input (detailed)
```

---

## 🎯 Why This Design Choice?

1. **Computational Efficiency**: Predicting 6k classes is much faster than 75k
2. **Better Generalization**: Grouped codes reduce sparsity and improve learning
3. **Task Alignment**: The goal is forecasting categories of events, not generating exact code sequences
4. **Parallelization**: Non-autoregressive generation allows parallel prediction (faster inference)

---

## 📝 Summary

| Question | Answer |
|----------|--------|
| Has causal masking? | ✅ Yes |
| Is architecturally autoregressive? | ✅ Partially (causal attention) |
| Is behaviorally autoregressive? | ❌ No (can't generate) |
| Is a true autoregressive model? | ❌ **No** |
| Correct term | **Causal sequence classifier** or **Teacher-forced causal model** |

The model borrows the **causal constraint** from autoregressive models (no peeking at future) but serves a **classification purpose** rather than **generation**. It's designed for **prediction** (what happens next?), not **generation** (produce a sequence one token at a time).

[4 tools called]

Now let me find the embedding extraction logic in the new MoE model:

[1 tool called]

Let me check the model forward pass to understand the exact layer where embeddings are extracted:

[6 tools called]

Now I have all the evidence. Let me provide a comprehensive, evidence-based answer:

---

# 🔍 Embedding Extraction: Original vs New Models

## ✅ Summary Answer

| Aspect | Original (min_transformer.py) | New (FlashAttention/MoE) |
|--------|-------------------------------|--------------------------|
| **Extraction Layer** | Output of `transformer_encoder_dy` | Input to final `norm` layer (BEFORE LayerNorm) |
| **Layer Position** | After 6-layer temporal encoder | After 6-layer temporal encoder |
| **Shape** | `[len_dy, batch, 256]` → transposed | `[batch, len_dy, 256]` |
| **Patient Embedding** | `embeddings[dt_cnt[i], i, :]` | `embeddings[i, dt_cnt-1, :]` |
| **Semantically Same?** | **YES** - Both are final temporal representations |

---

## 📋 Evidence-Based Comparison

### Original Model (`min_transformer.py`)

**Evidence 1: Hook Registration (lines 200, 266)**

```python
model.transformer_encoder_dy.register_forward_hook(get_activation('transformer_encoder_dy'))
```

**Evidence 2: Embedding Extraction (lines 224-226)**

```python
intermedia_output = activation['transformer_encoder_dy']       
intermedia_output = [intermedia_output[dt_cnt[i], i, :].reshape(1,-1) for i in range(batch_size)]
intermedia_output = torch.cat(intermedia_output)
```

**Evidence 3: Architecture Flow (lines 107-116)**

```python
mth_mask = self._generate_square_subsequent_mask(len_dy).to(device)      
cd = self.transformer_encoder_dy(cd, mth_mask)  # ← HOOK CAPTURES THIS OUTPUT
cd = torch.swapaxes(cd, 0, 1)
cd = self.norm(cd)
cd = self.dropout(cd)
cd = self.decoder_cd(cd)  # → Final prediction
```

**The original model extracts from:** Output of 6-layer temporal transformer BEFORE norm/dropout/decoder.

---

### New Models (`moe_flashattn_3.py`)

**Evidence 1: EmbeddingExtractor Hook Registration (lines 8641-8651)**

```python
elif isinstance(self.model, (FlashAttentionTransformer, FlashMoETransformer)):
    # Hook the final temporal layer's output
    target_layer = self.model.norm  # Final LayerNorm before decoder
    
    # Custom hook that captures BEFORE norm (the raw temporal output)
    def pre_decoder_hook(module, input, output):
        # input[0] is what goes INTO the norm layer = our embedding
        self.embeddings = input[0].detach()
    
    self._hook_handle = target_layer.register_forward_hook(pre_decoder_hook)
```

**Evidence 2: FlashAttentionTransformer Architecture (lines 2557-2562)**

```python
# Output projection
cd = torch.swapaxes(cd, 0, 1)

cd = self.norm(cd)    # ← HOOK CAPTURES input TO this layer
cd = self.dropout(cd)
cd = self.decoder_cd(cd)
```

**Evidence 3: FlashMoETransformer Architecture (lines 2849-2854)**

```python
# Output
cd = torch.swapaxes(cd, 0, 1)

cd = self.norm(cd)    # ← SAME: HOOK CAPTURES input TO this layer
cd = self.dropout(cd)
cd = self.decoder_cd(cd)
```

**The new models extract from:** The tensor that goes INTO `self.norm()` - which is EXACTLY the output of the 6-layer temporal encoder after final residual connection.

---

## 🔬 What Do These Embeddings Represent?

### Architecture Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        HIERARCHICAL TRANSFORMER ARCHITECTURE                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT LAYER (Day t)                                                         │
│  ─────────────────────                                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ age_in_months│ │  gender_cd  │ │    lob      │ │  cd (80 codes/day)     ││
│  │ Embedding    │ │ Embedding   │ │ Embedding   │ │  Embedding → Pooling   ││
│  └─────┬───────┘ └─────┬───────┘ └─────┬───────┘ └───────────┬─────────────┘│
│        │               │               │                     │              │
│        └───────────────┼───────────────┼─────────────────────┘              │
│                        │               │                                    │
│                        ▼               ▼                                    │
│               ┌────────────────────────────────────┐                        │
│               │        SUM + GELU + LayerNorm       │                        │
│               │  [batch, len_dy, 256]               │                        │
│               └────────────────┬───────────────────┘                        │
│                                │                                            │
│  TEMPORAL ENCODER (6 LAYERS)   │                                            │
│  ───────────────────────────   ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Layer 0: [Norm1 → Attn(causal) → Residual] → [Norm2 → FFN → Residual]  ││
│  │ Layer 1: [Norm1 → Attn(causal) → Residual] → [Norm2 → FFN → Residual]  ││
│  │ Layer 2: [Norm1 → Attn(causal) → Residual] → [Norm2 → FFN/MoE → Res]   ││
│  │ Layer 3: [Norm1 → Attn(causal) → Residual] → [Norm2 → FFN/MoE → Res]   ││
│  │ Layer 4: [Norm1 → Attn(causal) → Residual] → [Norm2 → FFN/MoE → Res]   ││
│  │ Layer 5: [Norm1 → Attn(causal) → Residual] → [Norm2 → FFN/MoE → Res]   ││
│  └────────────────────────────┬────────────────────────────────────────────┘│
│                               │                                             │
│                               ▼                                             │
│               ┌────────────────────────────────────┐                        │
│               │     ⭐ EMBEDDING EXTRACTED HERE ⭐   │                        │
│               │  [batch, len_dy, 256]               │                        │
│               │  After 6 layers of causal attention │                        │
│               │  Contains full patient history      │                        │
│               └────────────────┬───────────────────┘                        │
│                                │                                            │
│  OUTPUT LAYER                  ▼                                            │
│  ─────────────  ┌─────────────────────────────────────────────────────────┐ │
│                 │  LayerNorm → Dropout → Linear(256 → 6297) → logits      │ │
│                 │  [batch, len_dy, target_cd_cnt]                         │ │
│                 └─────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 What the Embedding Semantically Represents

The extracted embedding at position `t` (256-dimensional vector) represents:

| Property | Description |
|----------|-------------|
| **Temporal Context** | All medical events from day 0 to day t (via causal masking) |
| **Demographics** | Age, gender, and LOB encoded at each time step |
| **Clinical Patterns** | Learned relationships between diagnoses, procedures, medications |
| **Predictive Features** | Features optimized to predict next-day medical events |

**For Downstream Tasks (IP Risk):**

The **last valid day's embedding** (`embeddings[patient_i, dt_cnt-1, :]`) is used:

```python
# From EmbeddingExtractor.get_patient_embedding() (lines 8687-8694)
for i, valid_days in enumerate(dt_cnt):
    if valid_days > 0:
        # Get embedding at last valid day
        patient_embeddings.append(embeddings[i, valid_days - 1, :])
```

This represents the **entire patient's medical history** compressed into a 256-dimensional vector:
- Day 0: Only knows about day 0
- Day 50: Knows about days 0-50
- Day 199 (last): Knows about entire 200-day history

---

## ⚠️ Are They Semantically Identical?

**YES, they are semantically the same.**

| Evidence | Explanation |
|----------|-------------|
| **Same Layer** | Both extract from the OUTPUT of the 6-layer temporal encoder |
| **Same Position** | Both extract BEFORE LayerNorm + Dropout + Decoder |
| **Same Causal Context** | Both have attended to all previous days (0...t) |
| **Same Dimension** | Both are [256] per day per patient |

**The only difference is implementation:**

| Model | Hook Target | What's Captured |
|-------|-------------|-----------------|
| Original | `transformer_encoder_dy` output | Direct output of PyTorch TransformerEncoder |
| New Flash | `self.norm` input | Input to final LayerNorm (= temporal encoder output) |
| New MoE | `self.norm` input | Input to final LayerNorm (= temporal encoder output) |

---

## 📊 Practical Implications

1. **Downstream task embeddings are compatible**: You can use embeddings from either model for the same downstream classifier (e.g., IP risk prediction).

2. **MoE embeddings have additional computation**: In MoE models, the embedding passes through Mixture-of-Experts layers (layers 2-5), so the final representation has been processed by conditionally-activated expert networks.

3. **Patient-level embedding is the last day**: Both implementations use `dt_cnt - 1` to get the "full history" embedding for downstream tasks.


# 🎯 DEFINITIVE ANSWER: What the Model is Trained to Predict

## ✅ The Model Predicts **NEXT-DAY CODES (Condensed)**

Based on comprehensive analysis of both the SQL pipeline and Python model code, the model is trained to predict:

> **Given medical codes from day N → Predict condensed/grouped codes for day N+1**

---

## 📋 Evidence Chain

### Evidence 1: SQL Data Pipeline Uses LEAD() Function

From `procedure_explained.md` (lines 349-353):

```sql
z2 AS (
    -- Apply LEAD for next-day prediction
    SELECT ..., LEAD(target, 1) OVER (PARTITION BY individual_id ORDER BY dt ASC) AS target_next_day
    FROM z1
),
```

**This explicitly shifts targets by 1 day using SQL's LEAD() window function.**

---

### Evidence 2: SQL Documentation Explicitly States Next-Day

From `combine_all_lobs_for_training.sql` (lines 64-67):
```sql
4. Target column enables next-day prediction training
   - Input (cd): Codes from day N
   - Target (target): Codes from day N+1
   - Model learns to predict tomorrow's healthcare events
```

From `combine_all_lobs_for_training.sql` (line 41):
```sql
- target: STRING (TARGET code sequences for next-day prediction: "45*67*89*..." ~5k vocab)
```

---

### Evidence 3: Target Vocabulary is Condensed (~13:1 Compression)

From `create_w2ind_target_from_w2ind.sql` (lines 14-17):
```sql
VOCABULARY TRANSFORMATION:
- w2ind (INPUT):        ~80k codes → What transformer READS (encoder)
- w2ind_target (OUTPUT): ~6k codes → What transformer PREDICTS (decoder)
- Compression Ratio:     ~13:1 reduction
```

**Grouping examples from the SQL:**
| Input Code | Target Group | Rule |
|------------|--------------|------|
| `E11.65` (Type 2 DM w/ complications) | `icd9_dx_cdE11` | First 3 chars |
| `E11.9` (Type 2 DM unspecified) | `icd9_dx_cdE11` | First 3 chars |
| `99213` (Office visit Level 3) | `prcdr_group_992` | First 3 digits |
| `99214` (Office visit Level 4) | `prcdr_group_992` | First 3 digits |
| `gpi2210` (Metformin) | `gpi22` | First 2 digits |
| `gpi2215` (Glipizide) | `gpi22` | First 2 digits |

---

### Evidence 4: Causal Masking is for Teacher Forcing (Not Prediction Target)

From `moe_flashattn_3.py` (lines 1635-1642):
```python
if is_causal:
    # Create causal mask
    seq_len = q.size(2)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device),
        diagonal=1
    ).bool()
    scores.masked_fill_(causal_mask, float('-inf'))
```

**The causal mask ensures:**
- When computing representation for position t, model can only see positions 0...t
- This is **teacher forcing** — during training, the model sees all previous days' inputs
- The **prediction target** at position t is the **next day's codes (t+1)**, not same-day

---

### Evidence 5: Data Alignment in Training

From `ClinicalDataset` and `conv_target` in `moe_flashattn_3.py` (lines 3118-3157):
```python
# Target is already shifted in SQL via LEAD()
# So targets[day_idx] = codes that occurred on day (day_idx + 1)
self.targets.append(target_list)  # Already shifted by SQL LEAD()
```

The data arrives pre-shifted from SQL:
```
Day position 0: input=codes_day_0, target=codes_day_1
Day position 1: input=codes_day_1, target=codes_day_2
...
Day position N: input=codes_day_N, target=codes_day_N+1
```

---

## 🔍 Visual Representation

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     THE ACTUAL TRAINING TASK                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SQL PIPELINE (Data Prep):                                                   │
│  ────────────────────────                                                    │
│  Raw codes per day → Group/condense → Apply LEAD() shift                     │
│                                                                              │
│  Day 0: input_codes=[99213, E11.65, gpi2210]                                │
│           ↓ LEAD(1)                                                         │
│         target_for_day0 = condensed(codes_day1)                             │
│                         = [prcdr_group_992, icd9_dx_cdE11, ...]             │
│                                                                              │
│  TRAINING DATA FORMAT:                                                       │
│  ─────────────────────                                                       │
│  Position │ cd (Input ~84k vocab)        │ target (Next-day ~6k vocab)       │
│  ─────────┼──────────────────────────────┼───────────────────────────────────│
│     0     │ 1523,8492,45321              │ 101,203,89 (codes for DAY 1)     │
│     1     │ 7832,1120,8492               │ 45,890,101 (codes for DAY 2)     │
│     2     │ 3344,5678,9900               │ 23,456     (codes for DAY 3)     │
│    ...    │ ...                          │ ...                              │
│    199    │ 1234,5678                    │ (no target - last day)           │
│                                                                              │
│  MODEL PREDICTION:                                                           │
│  ─────────────────                                                           │
│  At position t, model sees: days 0, 1, 2, ..., t (via causal mask)          │
│  Model predicts: codes for day t+1 (the NEXT day)                           │
│  Loss computed: BCE(predicted, target[t]) where target[t] = actual day t+1  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Summary: The Complete Picture

| Aspect | Details | Evidence Source |
|--------|---------|-----------------|
| **Input vocabulary** | ~84,000 detailed codes | `create_w2ind_target_from_w2ind.sql` line 15 |
| **Target vocabulary** | ~6,000 grouped codes | `create_w2ind_target_from_w2ind.sql` line 16 |
| **Compression** | 13:1 reduction | Same file, line 17 |
| **Prediction task** | Next-day codes | `combine_all_lobs_for_training.sql` lines 64-67 |
| **Shift mechanism** | SQL LEAD(target, 1) | `procedure_explained.md` lines 349-353 |
| **Loss function** | BCEWithLogitsLoss (multi-label) | `moe_flashattn_3.py` line 400 |
| **Causal masking** | Teacher forcing (model sees days 0...t) | `moe_flashattn_3.py` lines 1635-1642 |

---

## ⚠️ Important Clarification on Causal Masking

The **causal mask does NOT define what is being predicted**. It defines **what information the model can use**.

```
CAUSAL MASK PURPOSE:
─────────────────────
• During training: Prevents model from "cheating" by looking at future days
• At position t: Model can attend to positions 0, 1, 2, ..., t
• At position t: Model CANNOT attend to positions t+1, t+2, ...

PREDICTION TARGET (determined by data, not mask):
─────────────────────────────────────────────────
• At position t: Target is codes from day t+1
• This is achieved via LEAD() in SQL, NOT via mask
• Mask = information restriction; LEAD = target shift
```

---

## 🏥 Clinical Interpretation

The model is trained to answer:

> **"Given a patient's medical history through today, what CATEGORIES of medical events will likely occur tomorrow?"**

- **Not exact codes**: Predicts `icd9_dx_cdE11` (Type 2 Diabetes family), not `E11.65` vs `E11.9`
- **Multi-label**: A patient can have multiple predictions for the next day
- **Clinically meaningful groups**: Grouping follows medical code hierarchies (CPT sections, ICD families, drug classes)

This is a **next-day clinical event forecasting** task with condensed output vocabulary for computational efficiency and better generalization.