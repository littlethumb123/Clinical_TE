I'll help you modify the code to incorporate the LOB (Line of Business) column. This requires changes across several components: configuration, data parsing, dataset preparation, and the model architectures.

## Key Changes Needed

Here's a comprehensive guide to add LOB support:

### 1. **Update Configuration Classes**

```python:dev/moe/moe_flashattn_2.py
# Around line 239-276 in BaseConfig
@dataclass
class BaseConfig:
    """
    Base configuration shared across all experiments.
    
    Parameters match your updated specifications:
    - len_dy: 200 days (sequence length)
    - len_cd: 80 codes per day
    - target_cd_cnt: 8850 target codes
    - Multi-label loss (BCEWithLogitsLoss)
    """
    # Data dimensions (from your specifications)
    len_dy: int = 200          # Days in sequence (updated from 70)
    len_cd: int = 80           # Codes per day (updated from 25)
    cd_cnt: int = 84010        # Input vocabulary size
    target_cd_cnt: int = 8850  # Target vocabulary (updated from 2767)

    # Model architecture
    embedding_size: int = 256  # Embedding dimension
    nhid: int = 512           # FFN hidden dimension
    nlayers: int = 6          # Number of temporal encoder layers
    dropout: float = 0.1      # Dropout rate (updated from 0.05)
    
    # Embeddings
    gender_vocab: int = 4     # Gender categories
    age_vocab: int = 1440     # Age in months (120 years)
    lob_vocab: int = 4        # ← ADD THIS: LOB categories (0=padding, 1=Commercial, 2=Medicare, 3=Medicaid)
    
    # Training
    batch_size: int = 16     # Batch size
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0  # Gradient clipping norm
    
    # Device
    device: str = 'cuda'
    
    # Loss function
    criterion: str = 'BCEWithLogitsLoss'  # Multi-label loss
```

### 2. **Add LOB Data Parsing Function**

```python:dev/moe/moe_flashattn_2.py
# Around line 2240-2308, after conv_age_gender function
def conv_lob(ipt: str, len_dy: int) -> List[int]:
    """
    Convert LOB (Line of Business) string to list.
    
    Format: "value1*value2*..." (same format as age/gender)
    Maps: Commercial=1, Medicare=2, Medicaid=3, padding=0
    
    Args:
        ipt: LOB string from data (e.g., "Medicaid*Medicaid*...")
        len_dy: Target sequence length (200)
    
    Returns:
        List of LOB indices [len_dy]
    """
    # LOB mapping
    lob_map = {
        'Commercial': 1,
        'Medicare': 2,
        'Medicaid': 3
    }
    
    if not ipt or pd.isna(ipt):
        return [0] * len_dy
    
    values = ipt.split('*')
    values = values[:len_dy]
    
    result = []
    for val in values:
        val_clean = val.strip() if val else ''
        if val_clean in lob_map:
            result.append(lob_map[val_clean])
        else:
            result.append(0)  # Unknown LOB
    
    # Forward-fill with last valid value (LOB typically doesn't change)
    if result:
        last_val = result[-1] if result[-1] != 0 else 1  # Default to Commercial if all zeros
        while len(result) < len_dy:
            result.append(last_val)
    else:
        result = [1] * len_dy  # Default to Commercial
    
    return result
```

### 3. **Update ClinicalDataset Class**

```python:dev/moe/moe_flashattn_2.py
# Around line 2150-2201, modify ClinicalDataset.__init__
class ClinicalDataset(Dataset):
    """
    PyTorch Dataset for clinical transformer.
    Pre-processes all string parsing once during initialization for high performance.
    """
    def __init__(self, df: pd.DataFrame, config: BaseConfig):
        self.config = config
        print(f"Pre-processing {len(df)} samples into tensors (one-time cost)...")
        
        # Extract series for faster processing
        age_strs = df['age_in_months'].tolist()
        gender_strs = df['gender_cd'].tolist()
        lob_strs = df['lob'].tolist()  # ← ADD THIS
        cd_strs = df['cd'].tolist()
        target_strs = df['target_cd'].tolist()
        self.dt_cnt = df['dt_cnt'].tolist()

        # Pre-allocate tensors
        self.ages = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.genders = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.lobs = torch.zeros(len(df), config.len_dy, dtype=torch.long)  # ← ADD THIS
        self.codes = torch.zeros(len(df), config.len_dy, config.len_cd, dtype=torch.long)
        self.targets = []

        # Process all samples
        for i in range(len(df)):
            if i > 0 and i % 50000 == 0:
                print(f"  Processed {i}/{len(df)} samples...")

            age_list = conv_age_gender(age_strs[i], config.len_dy)
            gender_list = conv_age_gender(gender_strs[i], config.len_dy, max_val=3)
            lob_list = conv_lob(lob_strs[i], config.len_dy)  # ← ADD THIS
            cd_list = conv_cd(cd_strs[i], config.len_dy, config.len_cd)
            target_list = conv_target(target_strs[i], config.len_dy, config.target_cd_cnt)

            self.ages[i] = torch.tensor(age_list, dtype=torch.long)
            self.genders[i] = torch.tensor(gender_list, dtype=torch.long)
            self.lobs[i] = torch.tensor(lob_list, dtype=torch.long)  # ← ADD THIS
            self.codes[i] = torch.tensor(cd_list, dtype=torch.long)
            self.targets.append(target_list)
        
        print("Pre-processing complete.")

    def __len__(self):
        return len(self.dt_cnt)

    def __getitem__(self, idx):
        return {
            'age': self.ages[idx],
            'gender': self.genders[idx],
            'lob': self.lobs[idx],  # ← ADD THIS
            'codes': self.codes[idx],
            'dt_cnt': self.dt_cnt[idx],
            'target': self.targets[idx]
        }
```

### 4. **Update clinical_collate_fn**

```python:dev/moe/moe_flashattn_2.py
# Around line 2206-2236
def clinical_collate_fn(batch):
    """
    Custom collate function for clinical data.
    
    Handles the special case of 'target' which is a nested list with variable-length sublists.
    PyTorch's default_collate cannot handle this, so we keep it as a Python list.
    
    Args:
        batch: List of dictionaries from ClinicalDataset.__getitem__
    
    Returns:
        Batched dictionary with:
        - age, gender, lob, codes: Stacked tensors
        - dt_cnt: List of integers
        - target: List of nested lists (NOT converted to tensor)
    """
    # Extract each field
    ages = torch.stack([item['age'] for item in batch])
    genders = torch.stack([item['gender'] for item in batch])
    lobs = torch.stack([item['lob'] for item in batch])  # ← ADD THIS
    codes = torch.stack([item['codes'] for item in batch])
    dt_cnts = [item['dt_cnt'] for item in batch]  # Keep as list
    targets = [item['target'] for item in batch]  # Keep as list of lists
    
    return {
        'age': ages,
        'gender': genders,
        'lob': lobs,  # ← ADD THIS
        'codes': codes,
        'dt_cnt': dt_cnts,
        'target': targets
    }
```

### 5. **Update prepare_tensor Function** (if still used)

```python:dev/moe/moe_flashattn_2.py
# Around line 2353-2404, modify prepare_tensor
def prepare_tensor(
    batch: pd.DataFrame,
    config: BaseConfig,
    device: torch.device
) -> Tuple[List[int], torch.Tensor, List[List[List[int]]]]:
    """
    Prepare batch for model input.
    
    Returns:
        dt_cnt: List of actual day counts per sample
        x: Input tensor [batch_size, len_dy, 3 + len_cd]  # ← CHANGED from 2 + len_cd
        y: Target codes List[List[List[int]]] - nested list for multi-label
    """
    batch_size = len(batch)
    
    # Extract all columns at once (faster than row iteration)
    age_strs = batch['age_in_months'].tolist()
    gender_strs = batch['gender_cd'].tolist()
    lob_strs = batch['lob'].tolist()  # ← ADD THIS
    cd_strs = batch['cd'].tolist()
    target_strs = batch['target_cd'].tolist()
    dt_cnt = batch['dt_cnt'].tolist()
    
    # Pre-allocate output tensors
    age_tensor = torch.zeros(batch_size, config.len_dy, dtype=torch.long, device=device)
    gender_tensor = torch.zeros(batch_size, config.len_dy, dtype=torch.long, device=device)
    lob_tensor = torch.zeros(batch_size, config.len_dy, dtype=torch.long, device=device)  # ← ADD THIS
    cd_tensor = torch.zeros(batch_size, config.len_dy, config.len_cd, dtype=torch.long, device=device)
    
    # Parallel parsing
    for i in range(batch_size):
        age_list = conv_age_gender(age_strs[i], config.len_dy)
        gender_list = conv_age_gender(gender_strs[i], config.len_dy, max_val=3)
        lob_list = conv_lob(lob_strs[i], config.len_dy)  # ← ADD THIS
        cd_list = conv_cd(cd_strs[i], config.len_dy, config.len_cd)
        
        age_tensor[i] = torch.tensor(age_list, dtype=torch.long)
        gender_tensor[i] = torch.tensor(gender_list, dtype=torch.long)
        lob_tensor[i] = torch.tensor(lob_list, dtype=torch.long)  # ← ADD THIS
        cd_tensor[i] = torch.tensor(cd_list, dtype=torch.long)
    
    # Concatenate
    x = torch.cat([
        age_tensor.unsqueeze(-1),
        gender_tensor.unsqueeze(-1),
        lob_tensor.unsqueeze(-1),  # ← ADD THIS
        cd_tensor
    ], dim=-1)
    
    # Parse targets (keep as list for multi-label)
    y = [conv_target(target_strs[i], config.len_dy, config.target_cd_cnt) 
         for i in range(batch_size)]
    
    return dt_cnt, x, y
```

### 6. **Update BaselineTransformer Model**

```python:dev/moe/moe_flashattn_2.py
# Around line 1450-1617, modify BaselineTransformer
class BaselineTransformer(nn.Module):
    """
    Baseline hierarchical clinical transformer.
    """
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        
        # ============================================================
        # EMBEDDINGS
        # ============================================================
        # Medical code embeddings
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        
        # Demographics embeddings
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # ← ADD THIS
        
        # ... rest of __init__ stays the same ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical transformer.
        
        Args:
            x: [batch, len_dy, 83] where 83 = [age, gender, lob, 80 codes]  # ← CHANGED
        
        Returns:
            output: [batch, len_dy, target_cd_cnt] logits for multi-label prediction
        """
        gpu_batchsize = x.shape[0]
        actual_len_dy = x.shape[1]
        actual_len_cd = x.shape[2] - 3  # ← CHANGED from -2
        
        # ============================================================
        # STEP 1: EXTRACT COMPONENTS
        # ============================================================
        age_in_months = x[:, :, 0].long()  # [batch, len_dy]
        gender_cd = x[:, :, 1].long()       # [batch, len_dy]
        lob = x[:, :, 2].long()             # ← ADD THIS: [batch, len_dy]
        cd = x[:, :, 3:].long()             # ← CHANGED from 2: to 3:
        
        # ============================================================
        # STEP 2: EMBED
        # ============================================================
        gender_cd = self.embedding_gender_cd(gender_cd)      # [batch, len_dy, embedding_size]
        age_in_months = self.embedding_age_in_months(age_in_months)  # [batch, len_dy, embedding_size]
        lob_emb = self.embedding_lob(lob)                    # ← ADD THIS: [batch, len_dy, embedding_size]
        cd = self.embedding_cd(cd)                           # [batch, len_dy, len_cd, embedding_size]
        
        # Residual connection: sum of all code embeddings
        cd_res = cd.sum(-2)  # [batch, len_dy, embedding_size]
        
        # ... daily encoding stays the same ...
        
        # ============================================================
        # STEP 4: COMBINE REPRESENTATIONS
        # ============================================================
        # Add all embeddings: residual codes + encoded codes + demographics + LOB
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # ← CHANGED: added lob_emb
        cd = self.mm(cd)  # GELU activation
        cd = self.norm(cd)
        
        # ... rest stays the same ...
```

### 7. **Update FlashAttentionTransformer Model**

```python:dev/moe/moe_flashattn_2.py
# Around line 1628-1835, modify FlashAttentionTransformer similarly
class FlashAttentionTransformer(nn.Module):
    
    def __init__(self, config: FlashAttentionConfig):
        super().__init__()
        self.config = config
        
        # Embeddings (same as baseline)
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # ← ADD THIS
        
        # ... rest of __init__ stays the same ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Flash Attention."""
        gpu_batchsize = x.shape[0]
        actual_len_dy = x.shape[1]
        actual_len_cd = x.shape[2] - 3  # ← CHANGED from -2
        
        # Extract and embed (same as baseline)
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long())  # ← ADD THIS
        cd = self.embedding_cd(x[:, :, 3:].long())  # ← CHANGED from 2:
        cd_res = cd.sum(-2)
        
        # ... daily encoding stays the same ...
        
        # Combine representations
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # ← CHANGED: added lob_emb
        cd = self.mm(cd)
        cd = self.norm(cd)
        
        # ... rest stays the same ...
```

### 8. **Update FlashMoETransformer Model**

```python:dev/moe/moe_flashattn_2.py
# Around line 1847-2051, modify FlashMoETransformer similarly
class FlashMoETransformer(nn.Module):
    
    def __init__(self, config: FlashAttentionConfig, moe_config: Optional[MoEConfig] = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        self.use_moe_from_layer = moe_config.use_moe_from_layer if moe_config else 999
        
        # Embeddings
        self.embedding_cd = nn.Embedding(config.cd_cnt, config.embedding_size)
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # ← ADD THIS
        
        # ... rest of __init__ stays the same ...
    
    def forward(self, x: torch.Tensor, return_moe_losses: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with Flash Attention + MoE."""
        gpu_batchsize = x.shape[0]
        actual_len_dy = x.shape[1]  
        actual_len_cd = x.shape[2] - 3  # ← CHANGED from -2
        device = x.device
        
        # Extract and embed
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long())  # ← ADD THIS
        cd = self.embedding_cd(x[:, :, 3:].long())  # ← CHANGED from 2:
        cd_res = cd.sum(-2)
        
        # ... daily encoding stays the same ...
        
        # Combine
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # ← CHANGED: added lob_emb
        cd = self.mm(cd)
        cd = self.norm(cd)
        
        # ... rest stays the same ...
```

### 9. **Update train_epoch Function**

```python:dev/moe/moe_flashattn_2.py
# Around line 3096-3278, update train_epoch to handle LOB
def train_epoch(...):
    """Train for one epoch."""
    # ... existing code ...
    
    for batch_idx, batch in enumerate(dataloader):
        # ... existing code ...
        
        # Get batch data
        age = batch['age'].to(device, non_blocking=True)
        gender = batch['gender'].to(device, non_blocking=True)
        lob = batch['lob'].to(device, non_blocking=True)  # ← ADD THIS
        codes = batch['codes'].to(device, non_blocking=True)
        dt_cnt = batch['dt_cnt']
        y = batch['target']
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),  # ← ADD THIS
            codes
        ], dim=-1)
        
        # ... rest stays the same ...
```

### 10. **Update evaluate Function**

```python:dev/moe/moe_flashattn_2.py
# Around line 3279-3429, update evaluate similarly
def evaluate(...):
    """Evaluate model on validation set."""
    # ... existing code ...
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # ... existing code ...
            
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            lob = batch['lob'].to(device, non_blocking=True)  # ← ADD THIS
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),  # ← ADD THIS
                codes
            ], dim=-1)
            
            # ... rest stays the same ...
```

## Summary of Changes

1. **Configuration**: Added `lob_vocab: int = 4` to `BaseConfig`
2. **Data parsing**: Created `conv_lob()` function to map LOB strings to indices
3. **Dataset**: Updated `ClinicalDataset` to process and store LOB data
4. **Collate function**: Added LOB to batching
5. **Tensor preparation**: Updated `prepare_tensor()` to include LOB (now `[batch, len_dy, 3 + len_cd]`)
6. **Models**: Added LOB embedding layer to all three transformer variants
7. **Training/Eval**: Updated data loading to concatenate LOB into input tensor

The LOB information will now be embedded and combined with age, gender, and medical codes to provide richer patient context for the model.