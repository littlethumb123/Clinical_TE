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