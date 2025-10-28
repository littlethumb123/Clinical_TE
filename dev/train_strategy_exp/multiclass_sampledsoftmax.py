#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
APPROACH 1: Multi-Label Classification with BCEWithLogitsLoss
============================================================

Treats prediction as independent binary classification for each of 84K codes.
Each day can predict multiple codes simultaneously (multi-hot encoding).

Pros:
  ✅ Simple implementation (just change target_cd_cnt)
  ✅ Works seamlessly with MoE (no architectural changes)
  ✅ Natural for medical coding (multiple diagnoses per day)
  ✅ No approximation - exact gradient for all classes

Cons:
  ⚠️ Slower training (~30% slower than 2767 classes)
  ⚠️ Larger memory footprint for final layer
  ⚠️ Gradient updates for all 84K classes every step
"""

"""
APPROACH 2: Sampled Softmax
===========================

Only compute loss over positive classes + random negative samples.
Reduces computation from O(84K) to O(2K) per step.

Pros:
  ✅ Much faster training (~3-5× faster than full 84K)
  ✅ Lower memory usage (smaller gradient tensors)
  ✅ Proven effective (Word2Vec, BERT, GPT pre-training)

Cons:
  ⚠️ Approximate gradients (but empirically works well)
  ⚠️ More complex implementation
  ⚠️ Requires different inference logic
  ⚠️ MoE integration requires care (see below)
"""



# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import time
import gc
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model hyperparameters (from min_transformer_finetune.py)
batch_size = 32  # Reduce if OOM
embedding_size = 256
len_dy = 200      # Max days in sequence
len_cd = 80      # Max codes per day
nhead = 16
nhid = 512
nlayers = 6
ndropout = 0.05
cd_cnt = 84010           # Input vocabulary
target_cd_cnt = 84010    # Target vocabulary (full 84K)

# Sampled softmax settings
num_sampled = 2767

# Training settings
num_epochs = 3           # For comparison (use more for real training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")


# In[3]:


# ============================================================================
# MODEL: Multi-Label Transformer (Compatible with MoE)
# ============================================================================

class MultiLabelHierarchicalTransformer(nn.Module):
    """
    Hierarchical transformer with multi-label output.
    Identical to your current architecture, just larger output layer.
    """
    
    def __init__(self, nhead, nhid, nlayers, dropout=0.05, 
                 cd_cnt=84010, target_cd_cnt=84010, embedding_size=256,
                 moe_config=None):
        super().__init__()
        
        # === EMBEDDINGS ===
        self.embedding_cd = nn.Embedding(cd_cnt, embedding_size)
        self.embedding_gender_cd = nn.Embedding(4, embedding_size)
        self.embedding_age_in_months = nn.Embedding(1440, embedding_size)
        
        # === CODE-LEVEL ENCODER ===
        encoder_layers_cd = TransformerEncoderLayer(embedding_size, 4, embedding_size, 0)
        self.transformer_encoder_cd = TransformerEncoder(encoder_layers_cd, 1)
        
        # === DAY-LEVEL ENCODER (with optional MoE) ===
        if moe_config is not None:
            # Use MoE layers (from your moe_experiments.py)
            from moe_experiments import MoETransformerEncoder
            self.transformer_encoder_dy = MoETransformerEncoder(
                embedding_size, nhead, nlayers, dropout, moe_config
            )
            self.use_moe = True
        else:
            # Dense baseline
            encoder_layers_dy = TransformerEncoderLayer(embedding_size, nhead, nhid, dropout)
            self.transformer_encoder_dy = TransformerEncoder(encoder_layers_dy, nlayers)
            self.use_moe = False
        
        # === OUTPUT LAYER ===
        self.mm = nn.GELU()
        self.decoder_cd = nn.Linear(embedding_size, target_cd_cnt)  # 84K outputs
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_size)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder_cd.bias)
        nn.init.uniform_(self.decoder_cd.weight, -initrange, initrange)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x, return_moe_losses=False):
        """
        Forward pass.
        
        Args:
            x: [batch, len_dy, 2+len_cd] - age, gender, codes
            return_moe_losses: If True and using MoE, return auxiliary losses
            
        Returns:
            logits: [batch, len_dy, 84010] - raw logits for BCELoss
            moe_losses: Dict (if using MoE and return_moe_losses=True)
        """
        batch_size = x.shape[0]
        len_dy = x.shape[1]
        len_cd = x.shape[2] - 2
        device = x.device
        
        # === EMBEDDING ===
        age_in_months = x[:, :, 0].long()
        gender_cd = x[:, :, 1].long()
        cd = x[:, :, 2:].long()
        
        age_emb = self.embedding_age_in_months(age_in_months)
        gender_emb = self.embedding_gender_cd(gender_cd)
        cd_emb = self.embedding_cd(cd)
        
        # === CODE-LEVEL ENCODING ===
        cd_res = cd_emb.sum(-2)
        cd_flat = cd_emb.reshape(batch_size * len_dy, len_cd, -1)
        cd_flat = cd_flat.transpose(0, 1)
        cd_encoded = self.transformer_encoder_cd(cd_flat)
        cd_encoded = cd_encoded.permute(1, 2, 0)
        cd_encoded = nn.MaxPool1d(len_cd)(cd_encoded)
        cd_encoded = cd_encoded.reshape(batch_size, len_dy, -1)
        
        # === COMBINE FEATURES ===
        cd_combined = cd_res + cd_encoded + age_emb + gender_emb
        cd_combined = self.mm(cd_combined)
        cd_combined = self.norm(cd_combined)
        cd_combined = cd_combined.transpose(0, 1)  # [len_dy, batch, 256]
        
        # === DAY-LEVEL ENCODING ===
        mth_mask = self._generate_square_subsequent_mask(len_dy).to(device)
        
        if self.use_moe and return_moe_losses:
            cd_encoded, moe_losses = self.transformer_encoder_dy(
                cd_combined, mth_mask, return_losses=True
            )
        else:
            cd_encoded = self.transformer_encoder_dy(cd_combined, mth_mask)
            moe_losses = {}
        
        cd_encoded = cd_encoded.transpose(0, 1)  # [batch, len_dy, 256]
        cd_encoded = self.norm(cd_encoded)
        cd_encoded = self.dropout(cd_encoded)
        
        # === OUTPUT (NO SOFTMAX for BCELoss!) ===
        logits = self.decoder_cd(cd_encoded)  # [batch, len_dy, 84010]
        
        if return_moe_losses and self.use_moe:
            return logits, moe_losses
        else:
            return logits


# ============================================================================
# TRAINING: Multi-Label Loss
# ============================================================================

def train_epoch_multilabel(model, data, optimizer, device, use_amp=True):
    """Train one epoch with multi-label loss - CORRECTED for next-day prediction."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if use_amp else None
    
    nbatch = len(data) // batch_size
    total_loss = 0
    epoch_start = time.time()
    
    for i in range(nbatch):
        batch = data.iloc[i*batch_size:(i+1)*batch_size]
        dt_cnt, x, y = prepare_tensor_multilabel(batch, device, batch_size, len_dy, len_cd)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            logits = model(x)  # [batch, len_dy, 84010]
            logits_flat = logits.reshape(batch_size * len_dy, target_cd_cnt)
            
            # CHANGE: Extract predictions for days 0 to dt_cnt-2 (to match targets)
            # Because y has targets for days 1 to dt_cnt-1
            predictions = []
            for j in range(batch_size):
                # Get predictions for days [0, dt_cnt-2] to predict days [1, dt_cnt-1]
                pred_slice = logits_flat[len_dy*j : len_dy*j + (dt_cnt[j] - 1), :]
                predictions.append(pred_slice)
            
            predictions = torch.cat(predictions, dim=0)  # [sum(dt_cnt-1), 84010]
            
            # Create multi-hot targets
            y_flat = [item for sublist in y for item in sublist]
            y_multihot = torch.zeros(len(predictions), target_cd_cnt, device=device)
            
            for day_idx in range(len(predictions)):
                for code_id in y_flat[day_idx]:
                    if 0 < code_id < target_cd_cnt:
                        y_multihot[day_idx, code_id] = 1.0
            
            loss = criterion(predictions, y_multihot)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / nbatch
    
    return avg_loss, epoch_time





# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# ============================================================================
# SAMPLED SOFTMAX LOSS LAYER
# ============================================================================

class SampledSoftmaxLoss(nn.Module):
    """
    Sampled softmax loss for efficient training with 84K classes.
    
    Key Idea:
        For each training example with K positive classes:
        - Compute scores for K positive classes (always)
        - Sample N negative classes (typically 2000)
        - Compute loss over K+N classes instead of full 84K
        
    This reduces:
        - Forward compute: 84K → 2K matrix multiplications
        - Backward compute: 84K → 2K gradient computations
        - Memory: 84K → 2K gradient storage
    """
    
    def __init__(self, embedding_size, num_classes, num_sampled=2000, 
                 sampling_strategy='log_uniform'):
        """
        Args:
            embedding_size: Hidden dimension (256)
            num_classes: Total vocabulary size (84010)
            num_sampled: Number of negative samples per example
            sampling_strategy: 'uniform' or 'log_uniform' (better for power law)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.sampling_strategy = sampling_strategy
        
        # Output embeddings (same as nn.Linear, but we'll use them differently)
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        
        # Pre-compute log-uniform sampling probabilities (for medical codes)
        if sampling_strategy == 'log_uniform':
            # Sample more frequent codes more often (matches power law distribution)
            # You can replace this with actual code frequencies from your data
            log_probs = np.log(np.arange(1, num_classes + 1) + 1)
            log_probs = log_probs[::-1]  # More frequent = lower index
            self.sampling_probs = torch.from_numpy(log_probs / log_probs.sum()).float()
        else:
            self.sampling_probs = None
    
    def sample_negatives(self, batch_size, positive_classes, device):
        """
        Sample negative classes for each example in batch.
        
        Args:
            batch_size: Number of examples
            positive_classes: List of lists, positive class IDs per example
            device: torch.device
            
        Returns:
            negative_samples: [batch_size, num_sampled] tensor of negative class IDs
            log_q: [batch_size, num_sampled] log sampling probabilities (for bias correction)
        """
        negative_samples = []
        log_q_values = []
        
        for i in range(batch_size):
            pos_set = set(positive_classes[i])
            
            # Sample from [0, num_classes) excluding positives
            candidates = list(set(range(self.num_classes)) - pos_set)
            
            if self.sampling_probs is not None:
                # Log-uniform sampling
                candidate_probs = self.sampling_probs[candidates].numpy()
                candidate_probs = candidate_probs / candidate_probs.sum()
                sampled_indices = np.random.choice(
                    len(candidates), 
                    size=min(self.num_sampled, len(candidates)),
                    replace=False,
                    p=candidate_probs
                )
                sampled = [candidates[idx] for idx in sampled_indices]
                log_q = np.log(candidate_probs[sampled_indices] + 1e-10)
            else:
                # Uniform sampling
                sampled = np.random.choice(
                    candidates,
                    size=min(self.num_sampled, len(candidates)),
                    replace=False
                )
                log_q = np.log(1.0 / len(candidates)) * np.ones(len(sampled))
            
            negative_samples.append(sampled)
            log_q_values.append(log_q)
        
        # Pad to num_sampled (in case some examples have fewer candidates)
        max_len = max(len(s) for s in negative_samples)
        negative_samples_padded = []
        log_q_padded = []
        
        for i in range(batch_size):
            padded = negative_samples[i] + [0] * (max_len - len(negative_samples[i]))
            negative_samples_padded.append(padded[:self.num_sampled])
            
            padded_log_q = list(log_q_values[i]) + [-100.0] * (max_len - len(log_q_values[i]))
            log_q_padded.append(padded_log_q[:self.num_sampled])
        
        return (torch.tensor(negative_samples_padded, dtype=torch.long, device=device),
                torch.tensor(log_q_padded, dtype=torch.float, device=device))
    
    def forward(self, hidden, target_classes_list):
        """
        Compute sampled softmax loss.
        
        Args:
            hidden: [batch_size, embedding_size] - model outputs
            target_classes_list: List of lists - positive class IDs per example
                Example: [[15, 42], [156, 823], [5042]]
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = hidden.size(0)
        device = hidden.device
        
        # Sample negatives
        negative_samples, log_q_neg = self.sample_negatives(
            batch_size, target_classes_list, device
        )
        
        # Compute loss for each example
        losses = []
        
        for i in range(batch_size):
            pos_classes = target_classes_list[i]
            if len(pos_classes) == 0 or pos_classes[0] == 0:
                continue
            
            # Filter out padding (0) from positive classes
            pos_classes = [c for c in pos_classes if c > 0]
            if len(pos_classes) == 0:
                continue
            
            pos_classes_tensor = torch.tensor(pos_classes, dtype=torch.long, device=device)
            neg_classes = negative_samples[i]
            
            # Combine positive and negative classes
            all_classes = torch.cat([pos_classes_tensor, neg_classes])
            
            # Compute logits for sampled classes
            sampled_weights = self.weight[all_classes]  # [num_pos + num_sampled, 256]
            sampled_biases = self.bias[all_classes]     # [num_pos + num_sampled]
            
            logits = torch.matmul(sampled_weights, hidden[i]) + sampled_biases
            # [num_pos + num_sampled]
            
            # Binary labels: 1 for positives, 0 for negatives
            labels = torch.zeros_like(logits)
            labels[:len(pos_classes)] = 1.0
            
            # Binary cross-entropy loss
            loss_i = F.binary_cross_entropy_with_logits(logits, labels)
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()
    
    def full_forward(self, hidden):
        """
        Full forward pass for inference (compute logits for all 84K classes).
        
        Args:
            hidden: [batch_size, embedding_size]
            
        Returns:
            logits: [batch_size, 84010]
        """
        logits = F.linear(hidden, self.weight, self.bias)
        return logits


# ============================================================================
# MODEL: Sampled Softmax Transformer
# ============================================================================

class SampledSoftmaxTransformer(nn.Module):
    """
    Hierarchical transformer with sampled softmax output.
    
    Key difference from multi-label: the decoder_cd is now a SampledSoftmaxLoss layer.
    """
    
    def __init__(self, nhead, nhid, nlayers, dropout=0.05, 
                 cd_cnt=84010, target_cd_cnt=84010, embedding_size=256,
                 num_sampled=2000, moe_config=None):
        super().__init__()
        
        # === EMBEDDINGS (same as before) ===
        self.embedding_cd = nn.Embedding(cd_cnt, embedding_size)
        self.embedding_gender_cd = nn.Embedding(4, embedding_size)
        self.embedding_age_in_months = nn.Embedding(1440, embedding_size)
        
        # === CODE-LEVEL ENCODER (same as before) ===
        encoder_layers_cd = TransformerEncoderLayer(embedding_size, 4, embedding_size, 0)
        self.transformer_encoder_cd = TransformerEncoder(encoder_layers_cd, 1)
        
        # === DAY-LEVEL ENCODER (same as before, with optional MoE) ===
        if moe_config is not None:
            from moe_experiments import MoETransformerEncoder
            self.transformer_encoder_dy = MoETransformerEncoder(
                embedding_size, nhead, nlayers, dropout, moe_config
            )
            self.use_moe = True
        else:
            encoder_layers_dy = TransformerEncoderLayer(embedding_size, nhead, nhid, dropout)
            self.transformer_encoder_dy = TransformerEncoder(encoder_layers_dy, nlayers)
            self.use_moe = False
        
        # === OUTPUT LAYER: DIFFERENT! ===
        self.mm = nn.GELU()
        self.decoder_cd = SampledSoftmaxLoss(
            embedding_size=embedding_size,
            num_classes=target_cd_cnt,
            num_sampled=num_sampled
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_size)
        
        self.len_dy = 200  # Store for reshaping
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x, target_classes=None, return_moe_losses=False):
        """
        Forward pass.
        
        Args:
            x: [batch, len_dy, 2+len_cd]
            target_classes: List of lists of lists (for training)
                Shape: [batch_size][day][codes]
                Example: [[[15, 42], [156]], [[823, 5042], [100]]]
            return_moe_losses: Return MoE auxiliary losses
            
        Returns:
            If training:
                loss: Sampled softmax loss
                moe_losses: Dict (if using MoE)
            If inference:
                logits: [batch, len_dy, 84010]
        """
        batch_size = x.shape[0]
        len_dy = x.shape[1]
        len_cd = x.shape[2] - 2
        device = x.device
        
        # === ENCODING (same as multi-label) ===
        age_in_months = x[:, :, 0].long()
        gender_cd = x[:, :, 1].long()
        cd = x[:, :, 2:].long()
        
        age_emb = self.embedding_age_in_months(age_in_months)
        gender_emb = self.embedding_gender_cd(gender_cd)
        cd_emb = self.embedding_cd(cd)
        
        cd_res = cd_emb.sum(-2)
        cd_flat = cd_emb.reshape(batch_size * len_dy, len_cd, -1)
        cd_flat = cd_flat.transpose(0, 1)
        cd_encoded = self.transformer_encoder_cd(cd_flat)
        cd_encoded = cd_encoded.permute(1, 2, 0)
        cd_encoded = nn.MaxPool1d(len_cd)(cd_encoded)
        cd_encoded = cd_encoded.reshape(batch_size, len_dy, -1)
        
        cd_combined = cd_res + cd_encoded + age_emb + gender_emb
        cd_combined = self.mm(cd_combined)
        cd_combined = self.norm(cd_combined)
        cd_combined = cd_combined.transpose(0, 1)
        
        mth_mask = self._generate_square_subsequent_mask(len_dy).to(device)
        
        if self.use_moe and return_moe_losses:
            cd_encoded, moe_losses = self.transformer_encoder_dy(
                cd_combined, mth_mask, return_losses=True
            )
        else:
            cd_encoded = self.transformer_encoder_dy(cd_combined, mth_mask)
            moe_losses = {}
        
        cd_encoded = cd_encoded.transpose(0, 1)  # [batch, len_dy, 256]
        cd_encoded = self.norm(cd_encoded)
        cd_encoded = self.dropout(cd_encoded)
        
        # === OUTPUT: DIFFERENT! ===
        if self.training and target_classes is not None:
            # Flatten embeddings and targets for sampled softmax
            cd_flat = cd_encoded.reshape(-1, cd_encoded.size(-1))  # [batch*len_dy, 256]
            targets_flat = target_classes
            
            # Compute sampled softmax loss
            loss_pred = self.decoder_cd(cd_flat, targets_flat)
            
            if return_moe_losses and self.use_moe:
                return loss_pred, moe_losses
            else:
                return loss_pred
        else:
            # Inference: compute full logits
            logits = self.decoder_cd.full_forward(cd_encoded.reshape(-1, cd_encoded.size(-1)))
            logits = logits.reshape(batch_size, len_dy, -1)
            logits = F.log_softmax(logits, dim=-1)
            
            if return_moe_losses and self.use_moe:
                return logits, moe_losses
            else:
                return logits


# ============================================================================
# TRAINING: Sampled Softmax
# ============================================================================

def train_epoch_sampled(model, data, optimizer, device, use_amp=True):
    """Train one epoch with sampled softmax - CORRECTED for next-day prediction."""
    model.train()
    scaler = GradScaler() if use_amp else None
    
    nbatch = len(data) // batch_size
    total_loss = 0
    epoch_start = time.time()
    
    for i in range(nbatch):
        batch = data.iloc[i*batch_size:(i+1)*batch_size]
        dt_cnt, x, y = prepare_tensor_multilabel(batch, device, batch_size, len_dy, len_cd)

        # don't pad to len_dy, only pad within the batch
        max_days_in_batch = max(len(y[patient_idx]) for patient_idx in range(batch_size))

        #  Flatten targets for next-day prediction
        # y structure: [batch][day_0_to_dt_cnt-2][codes]
        # We need flat list matching predictions from days [0, dt_cnt-2]
        targets_flat = []
        for patient_idx in range(batch_size):
            for day_idx in range(len(y[patient_idx])):  # y already has dt_cnt-1 entries
                targets_flat.append(y[patient_idx][day_idx])
            # Pad remaining days with [0]
            for day_idx in range(len(y[patient_idx]), max_days_in_batch):
                targets_flat.append([0])
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            # ALSO: Only pass the relevant portion of x and ignore the rest padding
            x_trimmed = x[:, :max_days_in_batch, :]
            loss = model(x_trimmed, targets_flat)
        
        # Handle zero loss gracefully
        if loss.item() > 0:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / nbatch
    
    return avg_loss, epoch_time


# In[10]:


# ============================================================================
# DATA PREPROCESSING (from min_transformer_finetune.py)
# ============================================================================

def conv_cd(ipt, max_days=len_dy, max_codes_per_day=len_cd):
    """
    Convert comma-separated code string to tensor format.
    (From min_transformer_finetune.py lines 125-132)
    
    Format: "code1,code2*code3,code4*..." 
    Where * separates days, , separates codes within a day
    
    Args:
        ipt: String like "20585,61238*83835,20*1,12081,28"
        
    Returns:
        List[List[int]]: [[20585, 61238, 0, ...], [83835, 20, 0, ...], ...]
    """
    ipt = ipt.split('*')
    ipt = ipt[:max_days]
    ipt = ipt + (max_days - len(ipt)) * ['']
    ipt = [dy.split(',') for dy in ipt]
    ipt = [[int(cd) if cd != '' else 0 for cd in dy] for dy in ipt]
    ipt = [dy + (max_codes_per_day - len(dy)) * [0] for dy in ipt]
    return ipt


def conv_age_gender(ipt, max_days=len_dy):
    """
    Convert age/gender string to tensor format.
    (From min_transformer_finetune.py lines 134-139)
    
    IMPORTANT: This function clips to 1439 max, suitable for age.
    For gender_cd, values should already be 0-3 in the data.
    
    Format: "value1*value2*..." where * separates days
    
    Args:
        ipt: String like "68*69*70"
        
    Returns:
        List[int]: [68, 69, 70, 0, ...] padded to max_days
    """
    ipt = ipt.split('*')
    ipt = ipt[:max_days]
    ipt = [min(int(cd), 1439) for cd in ipt]  # Original clips to 1439
    ipt = ipt + (max_days - len(ipt)) * [0]
    return ipt


def conv_target(target, max_days=len_dy):
    """
    Convert target codes string to list format.
    (From min_transformer_finetune.py lines 141-147)
    
    Format: "code1,code2*code3*..." where * separates days
    
    Args:
        target: String like "15,42*156*5042"
        
    Returns:
        List[List[int]]: [[15, 42], [156], [5042], ...]
    """
    target = target.split('*')
    target = target[:max_days]
    target = [dy.split(',') for dy in target]
    target = [[int(cd) if cd != '' else 0 for cd in dy] for dy in target]
    return target


def prepare_tensor_multilabel(batch, device, batch_size=batch_size, len_dy=len_dy, len_cd=len_cd):
    """
    Prepare tensors for multi-label model.
    (Based on min_transformer_finetune.py lines 149-166)
    
    CHANGES FROM BUGGY VERSION:
    1. Separate processing for age and gender (no shared function with max_value parameter)
    2. Gender clipped to [0, 3] explicitly
    3. Target extraction for NEXT DAY prediction (not same day)
    
    Returns:
        dt_cnt: List[int] - actual days per patient
        x: [batch, len_dy, 2+len_cd] - concatenated features
        y: List[List[List[int]]] - target codes per patient per NEXT day
    """
    # Process age_in_months (clip to 1439)
    age_in_months = [conv_age_gender(ipt, len_dy) for ipt in batch['age_in_months'].tolist()]
    age_in_months = torch.tensor(age_in_months, dtype=torch.long).to(device)
    age_in_months = age_in_months.reshape(batch_size, len_dy, 1)
    
    # Process gender_cd (MUST clip to 0-3 for nn.Embedding(4, ...))
    gender_cd = [conv_age_gender(ipt, len_dy) for ipt in batch['gender_cd'].tolist()]
    gender_cd = torch.tensor(gender_cd, dtype=torch.long).to(device)
    gender_cd = gender_cd.reshape(batch_size, len_dy, 1)
    # CRITICAL: Gender must be in [0, 3] range
    gender_cd = torch.clamp(gender_cd, 0, 3)
    
    # Process medical codes
    cd = [conv_cd(ipt, len_dy, len_cd) for ipt in batch['cd'].tolist()]
    cd = torch.tensor(cd, dtype=torch.long).to(device)
    
    # Concatenate all features
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)
    
    dt_cnt = batch['dt_cnt'].tolist()
    
    # ========================================================================
    # CRITICAL FIX: Extract targets for NEXT DAY prediction
    # ========================================================================
    # Original buggy code extracted same-day codes
    # Correct approach: predict codes on day t+1 given history up to day t
    
    y = []
    for i in range(batch_size):
        patient_targets = []
        
        # For each day t, predict codes on day t+1
        for day_idx in range(dt_cnt[i] - 1):  # Stop at dt_cnt-1 (no target for last day)
            # Target = codes on NEXT day (day_idx + 1)
            next_day_codes = [c for c in cd[i, day_idx + 1].cpu().numpy() if c > 0]
            patient_targets.append(next_day_codes if len(next_day_codes) > 0 else [0])
        
        y.append(patient_targets)
    
    return dt_cnt, x, y


# ============================================================================
# METRICS COLLECTION
# ============================================================================

def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_gradient_memory(model):
    """Calculate total gradient memory in MB."""
    total_grad_size = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_size += param.grad.numel() * param.grad.element_size()
    return total_grad_size / 1024**2


def get_model_size(model):
    """Get model parameter size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / 1024**2


# In[6]:


import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
client = bigquery.Client()
credentials, project= google.auth.default()
print('credentials:', credentials, ', project:', project)


# In[7]:


# Load data
input_sql = """
select * from
anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_o3_score_ending
limit 4000
"""
input_data = client.query(input_sql).to_dataframe() 
df_train = input_data.iloc[:3000]
df_val = input_data.iloc[3000:]


# In[12]:


results = {
    'multilabel': defaultdict(list),
    'sampled_softmax': defaultdict(list)
}

print("="*80)
print("BENCHMARK: Multi-Label vs Sampled Softmax")
print("="*80)
print(f"Dataset size: {len(df_train)} samples")
print(f"Batch size: {batch_size}")
print(f"Target vocabulary: {target_cd_cnt:,} codes")
print(f"Sampled negatives: {num_sampled}")
print(f"Device: {device}")
print()


# In[11]:


print("\n" + "="*80)
print("TRAINING: Sampled Softmax (2000 samples)")
print("="*80)

model_ss = SampledSoftmaxTransformer(nhead=nhead, 
    nhid=nhid, 
    nlayers=nlayers, 
    dropout=ndropout,
    cd_cnt=cd_cnt,
    target_cd_cnt=target_cd_cnt,
    embedding_size=embedding_size,
    num_sampled=num_sampled).to(device)
optimizer_ss = optim.SGD(model_ss.parameters(), lr=1e-3, momentum=0.9)

# Measure model size
model_size_ss = get_model_size(model_ss)
results['sampled_softmax']['model_size_mb'] = model_size_ss
print(f"Model size: {model_size_ss:.2f} MB")

# Measure initial memory
initial_mem_ss = get_gpu_memory()
results['sampled_softmax']['initial_memory_mb'] = initial_mem_ss
print(f"Initial GPU memory: {initial_mem_ss:.2f} MB")

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    avg_loss, epoch_time = train_epoch_sampled(model_ss, df_train, optimizer_ss, device)

    # Measure gradient size
    grad_size_ss = get_gradient_memory(model_ss)
    peak_mem_ss = get_gpu_memory()

    results['sampled_softmax']['epoch_loss'].append(avg_loss)
    results['sampled_softmax']['epoch_time'].append(epoch_time)
    results['sampled_softmax']['gradient_size_mb'].append(grad_size_ss)
    results['sampled_softmax']['peak_memory_mb'].append(peak_mem_ss)

    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Time: {epoch_time:.2f}s")
    print(f"  Gradient size: {grad_size_ss:.2f} MB")
    print(f"  Peak memory: {peak_mem_ss:.2f} MB")

# Final metrics
results['sampled_softmax']['total_time'] = sum(results['sampled_softmax']['epoch_time'])
results['sampled_softmax']['avg_loss'] = np.mean(results['sampled_softmax']['epoch_loss'])
results['sampled_softmax']['final_loss'] = results['sampled_softmax']['epoch_loss'][-1]

del model_ss, optimizer_ss
gc.collect()
torch.cuda.empty_cache()


# In[ ]:





# In[ ]:




