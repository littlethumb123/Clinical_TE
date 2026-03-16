# Legacy Model Replication (Full Training) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a standalone Jupyter notebook that faithfully replicates the legacy transformer training on the full `a834793_Combined_All_LOB_o3_train_ending` dataset, serving as both a production backup and a reference baseline for round 10 experiments.

**Architecture:** Independent notebook with the legacy 2-level hierarchical transformer (daily encoder + temporal encoder), trained with BCEWithLogitsLoss on multi-label targets (6297 codes), using SGD optimization with legacy hyperparameters. The log_softmax bug is corrected, the double-update bug is removed, and gradient clipping order is fixed.

**Tech Stack:** PyTorch, BigQuery (data loading), Google Cloud Storage (model checkpoints), pandas, numpy

---

## Background: Why This Plan Exists

The refactored `moe_flashattn_4.ipynb` framework, while more capable, introduced multiple subtle changes from the legacy training pipeline that compound into potentially significant differences in training outcomes. After thorough analysis, creating a standalone notebook is the safest path because:

1. The existing `exp1_dense_baseline` uses nhid=1024 (should be 512), adds LOB embedding, uses per-step scheduling, and wraps the model in DataParallelWrapper -- none of which match legacy.
2. Modifying the shared framework risks regressions across all other experiment configurations.
3. A self-contained notebook is easier to audit and verify correctness against the original.

## Critical Bug Fixes Applied (vs. Legacy Scripts)

These corrections from the refactored code are **intentionally preserved** because they fix genuine bugs:

| Bug | In Original (`min_transformer_train.py`) | Fix Applied |
|---|---|---|
| **Double weight update** | `optimizer.step()` + `p.data.add_(p.grad, alpha=-lr)` | Only `optimizer.step()` |
| **Gradient clip after step** | `clip_grad_norm_` called after `optimizer.step()` | Clip before step |
| **log_softmax + BCEWithLogitsLoss** | Present in `transformer_training_pipeline.py` | Return raw logits (no log_softmax) |

## Configuration Decisions

| Parameter | Legacy Value | Value Used | Rationale |
|---|---|---|---|
| `len_dy` | 200 | 200 | Match current data format |
| `len_cd` | 80 | 80 | Match current data format |
| `cd_cnt` | 75516 | 75516 | From `a834793_member_w2ind` |
| `target_cd_cnt` | 6297 | 6297 | From `a834793_member_w2ind_target` |
| `embedding_size` | 256 | 256 | Unchanged |
| `nhid` | 512 | 512 | Legacy value (NOT 1024 from exp1_dense_baseline) |
| `nlayers` | 6 | 6 | Unchanged |
| `nhead` | 16 | 16 | Unchanged |
| `dropout` | 0.05 | 0.05 | Match `transformer_training_pipeline.py` |
| `batch_size` | 512 | 512 | Match `transformer_training_pipeline.py` |
| `optimizer` | SGD(lr=1e-3, momentum=0.9) | SGD(lr=1e-3, momentum=0.9) | Match `transformer_training_pipeline.py` |
| `scheduler` | CosineAnnealingLR(T_max=num_epochs) | CosineAnnealingLR(T_max=num_epochs) | Per-epoch stepping |
| `gradient_clip` | 0.25 | 0.25 | Legacy value |
| `loss` | BCEWithLogitsLoss | BCEWithLogitsLoss | Multi-label (no pos_weight) |
| `val_split` | 80/20 random_split | 80/20 random_split | Standard approach |
| `num_epochs` | 10 | 10 | Legacy value |
| `LOB embedding` | Not present | **Not included** | Legacy architecture has no LOB |
| `parallel` | True (DataParallel) | True (DataParallel) | Multi-GPU |
| `min_dt_cnt_filter` | 180 | 180 | Match `transformer_training_pipeline.py` |

### Why SGD lr=1e-3 (not 1e-2)?

The original `min_transformer_train.py` used lr=1e-2 but with a double-update bug that effectively made the LR ~2-3x higher. The `transformer_training_pipeline.py` adjusted to lr=1e-3 to compensate after removing the double update. Since we fix the double-update bug, lr=1e-3 is the correct equivalent.

---

## Task 1: Create the Standalone Notebook File

**Files:**
- Create: `dev/legacy/legacy_full_training.ipynb`

### Step 1: Create notebook with configuration cell

Create `dev/legacy/legacy_full_training.ipynb` with the first cell containing:

```python
"""
Legacy Transformer Full Training - Standalone Replication
=========================================================
Purpose: Faithful replication of legacy training pipeline on full dataset.
         Serves as backup and reference baseline for round 10.
         
Bug fixes applied from refactored code:
  1. Removed log_softmax (correct for BCEWithLogitsLoss)
  2. Fixed gradient clipping order (before optimizer.step)
  3. Removed double weight update
  
Reference files:
  - Original: data_ingestion/Legacy/Train/python/min_transformer_train.py
  - Regenerated: dev/transformer_training_pipeline.py
  - Cleaned: dev/legacy/transformer_training_scoring.py
"""

import random
random.seed(1234)
import pandas as pd
import numpy as np
import gc
gc.collect()
import os
import torch
torch.manual_seed(123)
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import pytz
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURATION - Legacy values with corrections noted
# ===========================================================================
BIGQUERY_TABLE = 'edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending'
GCP_PROJECT = 'edp-prod-storage'
GCS_BUCKET = 'us-east4-edp-prod-css-sdoh--1b0f6fa9-bucket'
MODEL_PATH = 'a834793_transformer/Model/legacy_replication'

batch_size = 512
embedding_size = 256
minimum_mth_training = 180      # days, not months
len_dy = 200                     # sequence length (days)
len_cd = 80                      # codes per day
nhead = 16                       # temporal encoder heads
nhid = 512                       # FFN hidden dim (legacy value, NOT 1024)
nlayers = 6                      # temporal encoder layers
ndropout = 0.05                  # dropout rate
cd_cnt = 75516                   # input vocabulary size
target_cd_cnt = 6297             # target vocabulary size
parallel = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
entity_id = 'individual_id'
target = 'target'

NUM_EPOCHS = 10
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-3             # Adjusted from legacy 1e-2 (double-update bug removed)
GRADIENT_CLIP = 0.25

print(f"Device: {device}")
print(f"GPUs available: {torch.cuda.device_count()}")
print(f"Config: batch={batch_size}, emb={embedding_size}, nhid={nhid}, "
      f"nhead={nhead}, nlayers={nlayers}, dropout={ndropout}")
print(f"Data: len_dy={len_dy}, len_cd={len_cd}, cd_cnt={cd_cnt}, "
      f"target_cd_cnt={target_cd_cnt}")
```

### Step 2: Create model definition cell

```python
# ===========================================================================
# MODEL DEFINITION
# Matches legacy TransformerModel exactly, with log_softmax removed
# ===========================================================================

class LegacyTransformerModel(nn.Module):
    """
    Legacy hierarchical clinical transformer.
    
    Architecture (unchanged from min_transformer_train.py):
    - Daily encoder: 1 layer, 4 heads, d_ff=embedding_size, dropout=0
    - Temporal encoder: nlayers layers, nhead heads, d_ff=nhid, dropout=ndropout
    - Causal mask on temporal encoder
    - Max pooling on daily encoder output
    - Residual sum: code_sum + max_pool + gender + age
    
    Bug fix: Returns raw logits instead of log_softmax output.
    No LOB embedding (not present in legacy architecture).
    """
    def __init__(self, nhead, nhid, nlayers, dropout=0.05):
        super(LegacyTransformerModel, self).__init__()
        
        self.embedding_cd = nn.Embedding(cd_cnt, embedding_size)
        self.embedding_cd.weight.requires_grad = True
        self.embedding_gender_cd = nn.Embedding(4, embedding_size)
        self.embedding_gender_cd.weight.requires_grad = True
        self.embedding_age_in_months = nn.Embedding(1440, embedding_size)
        self.embedding_age_in_months.weight.requires_grad = True
        
        # Daily code encoder: 1 layer, 4 heads, no dropout
        encoder_layers_cd = TransformerEncoderLayer(embedding_size, 4, embedding_size, 0)
        self.transformer_encoder_cd = TransformerEncoder(encoder_layers_cd, 1)
        
        # Temporal encoder: 6 layers, 16 heads
        encoder_layers_dy = TransformerEncoderLayer(embedding_size, nhead, nhid, dropout)
        self.transformer_encoder_dy = TransformerEncoder(encoder_layers_dy, nlayers)
        
        self.mm = nn.GELU()
        self.decoder_cd = nn.Linear(embedding_size, target_cd_cnt)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embedding_size)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder_cd.weight)
        nn.init.uniform_(self.decoder_cd.weight, -initrange, initrange)

    def forward(self, x):
        gpu_batchsize = x.shape[0]
        age_in_months = x[:, :, 0]
        gender_cd = x[:, :, 1]
        
        gender_cd = self.embedding_gender_cd(gender_cd)
        age_in_months = self.embedding_age_in_months(age_in_months)
        
        cd = x[:, :, 2:]
        cd = self.embedding_cd(cd)
        cd_res = cd.sum(-2)
        cd = cd.reshape(gpu_batchsize * len_dy, len_cd, embedding_size)
        cd = torch.swapaxes(cd, 0, 1)
        cd = self.transformer_encoder_cd(cd)
        cd = cd.permute(1, 2, 0)
        cd = nn.MaxPool1d(len_cd)(cd)
        cd = cd.reshape(gpu_batchsize, len_dy, embedding_size)
        cd = cd_res + cd + gender_cd + age_in_months
        cd = self.mm(cd)
        cd = self.norm(cd)
        cd = torch.swapaxes(cd, 0, 1)
        
        mth_mask = self._generate_square_subsequent_mask(len_dy).to(x.device)
        cd = self.transformer_encoder_dy(cd, mth_mask)
        cd = torch.swapaxes(cd, 0, 1)
        cd = self.norm(cd)
        cd = self.dropout(cd)
        cd = self.decoder_cd(cd)
        
        # BUG FIX: Return raw logits for BCEWithLogitsLoss
        # Original had F.log_softmax(cd, dim=-1) which is incorrect for BCE
        return cd
```

### Step 3: Create dataset and data loading cells

```python
# ===========================================================================
# DATASET & DATA LOADING
# Same ClinicalDataset as transformer_training_scoring.py
# ===========================================================================

class ClinicalDataset(Dataset):
    def __init__(self, df, target_col='target'):
        self.samples = []
        self.target_col = target_col
        if minimum_mth_training > 0:
            df = df[df['dt_cnt'] >= minimum_mth_training].reset_index(drop=True)
            print(f"After filtering dt_cnt >= {minimum_mth_training}: {len(df)} samples")
        
        for idx in range(len(df)):
            if idx % 50000 == 0:
                print(f"  Pre-processing {idx}/{len(df)}...")
            row = df.iloc[idx]
            age = self._parse_age_gender(row['age_in_months'])
            gender = self._parse_age_gender(row['gender_cd'])
            codes = self._parse_codes(row['cd'])
            if target_col in row and pd.notna(row[target_col]):
                target_val = self._parse_target(row[target_col])
            else:
                target_val = []
            self.samples.append({
                'age': np.array(age, dtype=np.int64),
                'gender': np.array(gender, dtype=np.int64),
                'codes': np.array(codes, dtype=np.int64),
                'dt_cnt': int(row['dt_cnt']),
                'target': target_val,
                entity_id: row[entity_id] if entity_id in row.index else None
            })
        print(f"Pre-processing complete: {len(self.samples)} samples")

    def _parse_age_gender(self, ipt):
        ipt = ipt.split('*')
        ipt = ipt[:len_dy]
        ipt = [min(int(cd), 1439) if cd != '' else 0 for cd in ipt]
        ipt = ipt + (len_dy - len(ipt)) * [0]
        return ipt

    def _parse_codes(self, ipt):
        ipt = ipt.split('*')
        ipt = ipt[:len_dy]
        ipt = ipt + (len_dy - len(ipt)) * ['']
        ipt = [dy.split(',') for dy in ipt]
        ipt = [[int(cd) if cd != '' else 0 for cd in dy] for dy in ipt]
        ipt = [dy + (len_cd - len(dy)) * [0] for dy in ipt]
        return ipt

    def _parse_target(self, target_str):
        target_str = target_str.split('*')
        target_str = target_str[:len_dy]
        target_str = [dy.split(',') for dy in target_str]
        target_str = [[int(cd) if cd != '' else 0 for cd in dy] for dy in target_str]
        return target_str

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'age': torch.from_numpy(sample['age']),
            'gender': torch.from_numpy(sample['gender']),
            'codes': torch.from_numpy(sample['codes']),
            'dt_cnt': sample['dt_cnt'],
            'target': sample['target'],
            entity_id: sample[entity_id]
        }
```

```python
# ===========================================================================
# DATA LOADING FROM BIGQUERY
# ===========================================================================
from google.cloud import bigquery

def load_full_training_data():
    """Load full training dataset from BigQuery."""
    client = bigquery.Client(project=GCP_PROJECT)
    
    query = f"""
    SELECT 
        individual_id,
        age_in_months,
        gender_cd,
        cd,
        dt_cnt,
        target
    FROM `{BIGQUERY_TABLE}`
    """
    
    print(f"Loading data from {BIGQUERY_TABLE}...")
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df):,} rows")
    
    # Deduplicate: keep only members with exactly 1 record
    member_counts = df.groupby('individual_id').size()
    single_record = member_counts[member_counts == 1].index
    df = df[df['individual_id'].isin(single_record)].copy()
    print(f"After dedup: {len(df):,} unique members")
    
    return df
```

### Step 4: Create training loop cell

```python
# ===========================================================================
# TRAINING & VALIDATION FUNCTIONS
# ===========================================================================

def currentTime():
    tz = pytz.timezone("America/New_York")
    return datetime.now(tz).strftime("%D %H:%M:%S")

def train_epoch(model, dataloader, optimizer, criterion):
    """Train one epoch. Matches legacy flow with bug fixes."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{num_batches}  {currentTime()}')
        
        optimizer.zero_grad()
        
        age = batch['age'].to(device, non_blocking=True)
        gender = batch['gender'].to(device, non_blocking=True)
        codes = batch['codes'].to(device, non_blocking=True)
        dt_cnt = batch['dt_cnt']
        targets = batch['target']
        
        # Build input tensor: [batch, len_dy, 2+len_cd]
        x = torch.cat([age.unsqueeze(-1), gender.unsqueeze(-1), codes], dim=-1)
        
        output = model(x)
        output = output.reshape(-1, target_cd_cnt)
        
        # Flatten targets and select valid timesteps
        targets_flat = [item for sublist in targets for item in sublist]
        valid_outputs = torch.cat(
            [output[len_dy * i:len_dy * i + dt_cnt[i], :] for i in range(len(dt_cnt))],
            dim=0
        )
        
        # Build multi-hot target tensor
        y_cd = torch.zeros(len(valid_outputs), target_cd_cnt, device=device)
        for j in range(len(valid_outputs)):
            for k in targets_flat[j]:
                if k != 0:
                    y_cd[j, k] = 1
        
        loss = criterion(valid_outputs, y_cd)
        total_loss += loss.item()
        
        loss.backward()
        
        # BUG FIX: Clip BEFORE step (legacy clipped after step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        # BUG FIX: No manual weight update (legacy had double update)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def val_epoch(model, dataloader, criterion):
    """Validate one epoch."""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f'  Val Batch {batch_idx}/{num_batches}  {currentTime()}')
            
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            targets = batch['target']
            
            x = torch.cat([age.unsqueeze(-1), gender.unsqueeze(-1), codes], dim=-1)
            
            output = model(x)
            output = output.reshape(-1, target_cd_cnt)
            
            targets_flat = [item for sublist in targets for item in sublist]
            valid_outputs = torch.cat(
                [output[len_dy * i:len_dy * i + dt_cnt[i], :] for i in range(len(dt_cnt))],
                dim=0
            )
            
            y_cd = torch.zeros(len(valid_outputs), target_cd_cnt, device=device)
            for j in range(len(valid_outputs)):
                for k in targets_flat[j]:
                    if k != 0:
                        y_cd[j, k] = 1
            
            loss = criterion(valid_outputs, y_cd)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss
```

### Step 5: Create checkpoint and model management cell

```python
# ===========================================================================
# CHECKPOINT MANAGEMENT
# ===========================================================================
from google.cloud import storage
import joblib
from io import BytesIO

def save_checkpoint_gcs(model, optimizer, epoch, val_loss, filename='checkpoint_latest.pt'):
    """Save checkpoint to GCS."""
    checkpoint = {
        'timestamp': str(currentTime()),
        'model': model.module.state_dict() if parallel and isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    blob = storage.Client().bucket(GCS_BUCKET).blob(
        os.path.join(MODEL_PATH, filename)
    )
    buf = BytesIO()
    joblib.dump(checkpoint, buf)
    buf.seek(0)
    blob.upload_from_file(buf)
    print(f"  Checkpoint saved to gs://{GCS_BUCKET}/{MODEL_PATH}/{filename}")

def save_checkpoint_local(model, optimizer, epoch, val_loss, filepath):
    """Save checkpoint locally."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'timestamp': str(currentTime()),
        'model': model.module.state_dict() if parallel and isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(checkpoint, filepath)
    print(f"  Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load checkpoint from local file."""
    checkpoint = torch.load(filepath, map_location=device)
    if parallel and isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))
```

### Step 6: Create main training orchestration cell

```python
# ===========================================================================
# MAIN TRAINING PIPELINE
# ===========================================================================

# 1. Load data
df = load_full_training_data()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Create dataset
print("\nCreating dataset...")
dataset = ClinicalDataset(df, target_col=target)
del df
gc.collect()

# 3. Split train/val
train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train: {train_size:,} | Val: {val_size:,}")

# 4. Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=8, pin_memory=True, prefetch_factor=2,
    persistent_workers=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=8, pin_memory=True, prefetch_factor=2,
    persistent_workers=True, drop_last=False
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# 5. Create model
model = LegacyTransformerModel(nhead, nhid, nlayers, ndropout)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
if parallel and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
model = model.to(device)

# 6. Setup optimization (legacy config)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.BCEWithLogitsLoss()

# 7. Training loop
print(f"\n{'='*80}")
print(f"Starting training: {NUM_EPOCHS} epochs")
print(f"Optimizer: SGD(lr={LEARNING_RATE}, momentum=0.9)")
print(f"Scheduler: CosineAnnealingLR(T_max={NUM_EPOCHS})")
print(f"Loss: BCEWithLogitsLoss (no pos_weight)")
print(f"Gradient clip: {GRADIENT_CLIP}")
print(f"{'='*80}\n")

best_val_loss = None
training_history = []

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'='*60}")
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    print(f"  Train loss: {train_loss:.6f}")
    
    val_loss = val_epoch(model, val_loader, criterion)
    print(f"  Val loss:   {val_loss:.6f}")
    
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lr': optimizer.param_groups[0]['lr']
    })
    
    # Save best model
    if best_val_loss is None or val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint_local(
            model, optimizer, epoch, val_loss,
            'logs/legacy_replication/best_model.pt'
        )
        save_checkpoint_gcs(model, optimizer, epoch, val_loss, 'best_model.pt')
        print(f"  *** New best model! Val loss: {best_val_loss:.6f}")
    
    # Save epoch checkpoint
    save_checkpoint_local(
        model, optimizer, epoch, val_loss,
        f'logs/legacy_replication/checkpoint_epoch{epoch}.pt'
    )
    
    # Step scheduler (per-epoch, matching legacy behavior)
    scheduler.step()

print(f"\n{'='*80}")
print(f"Training complete!")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"{'='*80}")

# Print history
print("\nTraining History:")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>12}")
for h in training_history:
    print(f"{h['epoch']:>6} {h['train_loss']:>12.6f} {h['val_loss']:>12.6f} {h['lr']:>12.8f}")
```

### Step 7: Create embedding extraction cell (for downstream evaluation)

```python
# ===========================================================================
# EMBEDDING EXTRACTION (for downstream task evaluation)
# Matches transformer_training_scoring.py's score() function
# ===========================================================================

def extract_embeddings(model, data_df, batch_size=512):
    """
    Extract member-level embeddings from the last valid timestep.
    Uses forward hook on transformer_encoder_dy.
    """
    model.eval()
    activation = {}

    def get_activation(name):
        def hook(model_hook, input_hook, output_hook):
            activation[name] = output_hook.detach()
        return hook

    # Get the actual model (unwrap DataParallel)
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    handle = actual_model.transformer_encoder_dy.register_forward_hook(
        get_activation('transformer_encoder_dy')
    )

    # Pad data to be divisible by batch_size
    dsize = data_df.shape[0]
    nbatch = int(dsize / batch_size)
    if dsize - nbatch * batch_size > 0:
        k = batch_size - (dsize - nbatch * batch_size)
        data_df = pd.concat([data_df, pd.concat([data_df.head(1)] * k, ignore_index=True)])
    data_df = data_df.reset_index(drop=True)
    nbatch = int(data_df.shape[0] / batch_size)

    # Helper functions for raw DataFrame processing
    def _conv_cd(ipt):
        ipt = ipt.split('*')[:len_dy]
        ipt = ipt + (len_dy - len(ipt)) * ['']
        ipt = [dy.split(',') for dy in ipt]
        ipt = [[int(cd) if cd != '' else 0 for cd in dy] for dy in ipt]
        ipt = [dy + (len_cd - len(dy)) * [0] for dy in ipt]
        return ipt

    def _conv_age_gender(ipt):
        ipt = ipt.split('*')[:len_dy]
        ipt = [min(int(cd), 1439) for cd in ipt]
        ipt = ipt + (len_dy - len(ipt)) * [0]
        return ipt

    ys = []
    with torch.no_grad():
        for i in tqdm(range(nbatch), desc="Extracting embeddings"):
            batch = data_df.iloc[i * batch_size:i * batch_size + batch_size, :]
            
            # Build tensor from raw DataFrame
            ages = torch.tensor([_conv_age_gender(v) for v in batch['age_in_months'].tolist()]).to(device)
            genders = torch.tensor([_conv_age_gender(v) for v in batch['gender_cd'].tolist()]).to(device)
            codes = torch.tensor([_conv_cd(v) for v in batch['cd'].tolist()]).to(device)
            dt_cnt = batch['dt_cnt'].tolist()
            
            x = torch.cat([ages.unsqueeze(-1), genders.unsqueeze(-1), codes], dim=-1)
            _ = model(x)
            
            enc_out = activation['transformer_encoder_dy']
            # Extract embedding from last valid timestep per member
            embeddings = torch.stack([
                enc_out[dt_cnt[j], j, :] for j in range(batch_size)
            ])
            ys.append(embeddings)

    handle.remove()
    ys = torch.cat(ys).cpu().numpy()
    result = pd.DataFrame(ys, columns=[f'emb{i}' for i in range(embedding_size)])
    result[entity_id] = data_df[entity_id].values
    result = result.head(dsize)
    return result
```

---

## Task 2: Validate Model Architecture Parity

**Purpose:** Before running the full training, confirm the standalone model has identical architecture to the legacy.

### Step 1: Parameter count verification cell

```python
# ===========================================================================
# VALIDATION: Architecture parity check
# ===========================================================================
model_test = LegacyTransformerModel(nhead, nhid, nlayers, ndropout)
params = {name: p.shape for name, p in model_test.named_parameters()}

print("Layer-by-layer parameter shapes:")
total = 0
for name, shape in params.items():
    n = 1
    for s in shape:
        n *= s
    total += n
    print(f"  {name:<50} {str(shape):<25} {n:>12,}")
print(f"\n  {'TOTAL':<50} {'':25} {total:>12,}")

# Verify critical architecture properties
assert model_test.transformer_encoder_cd.layers[0].self_attn.num_heads == 4, "Daily encoder should have 4 heads"
assert len(model_test.transformer_encoder_cd.layers) == 1, "Daily encoder should have 1 layer"
assert len(model_test.transformer_encoder_dy.layers) == 6, "Temporal encoder should have 6 layers"
assert model_test.transformer_encoder_dy.layers[0].self_attn.num_heads == 16, "Temporal encoder should have 16 heads"
assert model_test.decoder_cd.out_features == target_cd_cnt, f"Output should be {target_cd_cnt}"

# Verify NO log_softmax in output
x_test = torch.randint(0, 100, (2, len_dy, 2 + len_cd))
out = model_test(x_test)
assert (out > 0).any(), "Output should contain positive values (raw logits, not log_softmax)"
print("\nAll architecture checks passed!")
del model_test
```

---

## Task 3: Smoke Test with Small Sample

**Purpose:** Run 1-2 epochs on a tiny subset to verify the full pipeline works end-to-end before committing GPU hours.

### Step 1: Create smoke test cell

```python
# ===========================================================================
# SMOKE TEST: Quick validation with small sample
# ===========================================================================
from google.cloud import bigquery

client = bigquery.Client(project=GCP_PROJECT)
test_sql = f"""
SELECT individual_id, age_in_months, gender_cd, cd, dt_cnt, target
FROM `{BIGQUERY_TABLE}`
WHERE dt_cnt >= {minimum_mth_training}
LIMIT 2000
"""
df_test = client.query(test_sql).to_dataframe()
print(f"Smoke test data: {len(df_test)} rows")

dataset_test = ClinicalDataset(df_test, target_col=target)
train_size = int(0.8 * len(dataset_test))
val_size = len(dataset_test) - train_size
train_ds, val_ds = random_split(dataset_test, [train_size, val_size])

train_loader_test = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
val_loader_test = DataLoader(val_ds, batch_size=32, shuffle=False)

model_test = LegacyTransformerModel(nhead, nhid, nlayers, ndropout).to(device)
optimizer_test = optim.SGD(model_test.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion_test = nn.BCEWithLogitsLoss()

print("\nRunning 2 smoke test epochs...")
for ep in range(2):
    train_loss = train_epoch(model_test, train_loader_test, optimizer_test, criterion_test)
    val_loss = val_epoch(model_test, val_loader_test, criterion_test)
    print(f"  Epoch {ep+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

print("\nSmoke test passed! Ready for full training.")
del model_test, optimizer_test, dataset_test
gc.collect()
torch.cuda.empty_cache()
```

---

## Summary of Key Differences from exp1_dense_baseline

| What | `exp1_dense_baseline` (moe_flashattn_4) | This Legacy Replication |
|---|---|---|
| nhid (FFN dim) | 1024 (4x expansion) | **512** (legacy value) |
| LOB embedding | Yes (nn.Embedding(4, 256)) | **No** (not in legacy) |
| Input features | [age, gender, lob, codes] | **[age, gender, codes]** |
| Output activation | Raw logits (correct) | Raw logits (correct) |
| cd_cnt | 75516 | 75516 |
| target_cd_cnt | 6297 | 6297 |
| Optimizer | AdamW(lr=2e-4, wd=0.01) | **SGD(lr=1e-3, momentum=0.9)** |
| Scheduler | Linear warmup+plateau+decay (per-step) | **CosineAnnealingLR (per-epoch)** |
| Gradient clip | 1.0 | **0.25** |
| Batch size | 32 * num_gpus | **512** |
| Loss | BCEWithLogitsLoss + pos_weight | **BCEWithLogitsLoss (no weighting)** |
| Loss integration | DataParallelWrapper (inside forward) | Standard (outside forward) |
| Val split | 99/1 stratified by LOB | **80/20 random_split** |
| Dropout | 0.05 | 0.05 |
| Data parallel | DataParallelWrapper | **nn.DataParallel** |
