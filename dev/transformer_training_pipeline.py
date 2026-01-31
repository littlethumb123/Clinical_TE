###############################
### train general #############
###############################

# =============================================================================
# CONFIGURATION
# =============================================================================
RUN_TRAINING = False  # Change to True to start training

# Data source configuration
BIGQUERY_TABLE = 'edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending'
GCP_PROJECT = 'edp-prod-storage'
GCS_BUCKET = 'us-east4-edp-prod-css-sdoh--1b0f6fa9-bucket'
GCS_DATA_PATH = 'a834793_transformer/TrainingData'  # Where pickles will be stored

# Choose data loading method
USE_GCS_PICKLES = True  # True = load from GCS pickles (fast, cheap)
                        # False = load from BigQuery (flexible, costs per query)

# For testing with small sample (only used when USE_GCS_PICKLES=False)
SAMPLE_SIZE = None  # Set to e.g., 10000 for quick testing, None for full data
# =============================================================================

import random
random.seed(1234)
import pandas as pd
import numpy as np
import gc
gc.collect()
from sklearn.model_selection import train_test_split
import os
import math
import torch
torch.manual_seed(123)
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from joblib import Parallel, delayed
import multiprocessing
from io import open
import argparse
import time
import math
import torch.onnx
import h5py
import os 
import pickle
from multiprocessing import cpu_count, Pool
from datetime import datetime
from google.cloud import storage
import joblib
from io import BytesIO
import google.auth
from google.auth import impersonated_credentials
from datetime import datetime
import pytz

class TransformerModel(nn.Module):
    def __init__(self, nhead, nhid, nlayers, dropout=0.05):
        super(TransformerModel, self).__init__()
        
        self.embedding_cd = nn.Embedding(cd_cnt,embedding_size)
        self.embedding_cd.weight.requires_grad = True
        self.embedding_gender_cd = nn.Embedding(4,embedding_size)
        self.embedding_gender_cd.weight.requires_grad = True
        self.embedding_age_in_months = nn.Embedding(1440,embedding_size)  
        self.embedding_age_in_months.weight.requires_grad = True
        
        encoder_layers_cd = TransformerEncoderLayer(embedding_size, 4, embedding_size, 0)
        self.transformer_encoder_cd = TransformerEncoder(encoder_layers_cd, 1)        
        
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
        age_in_months = x[:,:,0]
        gender_cd = x[:,:,1]
        
        gender_cd = self.embedding_gender_cd(gender_cd)
        age_in_months = self.embedding_age_in_months(age_in_months)
        
        cd = x[:,:,2:]
        cd = self.embedding_cd(cd)
        cd_res = cd.sum(-2)
        # print(cd_res.shape)
        cd = cd.reshape(gpu_batchsize*len_dy,len_cd,embedding_size)
        cd = torch.swapaxes(cd, 0, 1) 
        cd = self.transformer_encoder_cd(cd)
        cd = cd.permute(1,2,0)
        cd = nn.MaxPool1d(len_cd)(cd)
        cd = cd.reshape(gpu_batchsize,len_dy,embedding_size)
        # print(cd.shape)
        cd = cd_res+cd + gender_cd + age_in_months
        # print(cd.shape)
        cd = self.mm(cd)
        cd = self.norm(cd)
        cd = torch.swapaxes(cd, 0, 1)

        mth_mask = self._generate_square_subsequent_mask(len_dy).to(device)      
        cd = self.transformer_encoder_dy(cd, mth_mask)
        cd = torch.swapaxes(cd, 0, 1)
        cd = self.norm(cd)
        cd = self.dropout(cd)

        cd = self.decoder_cd(cd)
        cd = F.log_softmax(cd, dim=-1)

        return cd


# OLD APPROACH - commented out (slow, I/O bottleneck)
# def dataLoader(fileid):
#     blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(fileid)
#     data = BytesIO()
#     blob.download_to_file(data)
#     data=joblib.load(data)
#     data = data.sample(frac=1)
#     return data


# ============================================================================
# NEW APPROACH - PyTorch DataLoader (faster, better GPU utilization)
# ============================================================================

from torch.utils.data import Dataset, DataLoader

class ClinicalDataset(Dataset):
    """
    PyTorch Dataset for clinical transformer training.
    Pre-processes all string parsing once during initialization.
    """
    def __init__(self, df, target_col='target'):
        """
        Args:
            df: pandas DataFrame with columns: age_in_months, gender_cd, cd, dt_cnt, target
            target_col: name of target column
        """
        print(f"Pre-processing {len(df)} samples (one-time cost)...")
        self.samples = []
        self.target_col = target_col
        
        # Filter by minimum training months
        if minimum_mth_training > 0:
            df = df[df['dt_cnt'] >= minimum_mth_training].reset_index(drop=True)
            print(f"After filtering: {len(df)} samples with dt_cnt >= {minimum_mth_training}")
        
        for idx in range(len(df)):
            if idx % 10000 == 0:
                print(f"  Processed {idx}/{len(df)} samples...")
            
            row = df.iloc[idx]
            
            # Parse age (do once, not per batch!)
            age = self._parse_age_gender(row['age_in_months'])
            
            # Parse gender (do once, not per batch!)
            gender = self._parse_age_gender(row['gender_cd'])
            
            # Parse medical codes (do once, not per batch!)
            codes = self._parse_codes(row['cd'])
            
            # Parse target
            if target_col in row:
                target = self._parse_target(row[target_col])
            else:
                target = []
            
            # Store as numpy arrays (convert to tensors in __getitem__)
            self.samples.append({
                'age': np.array(age, dtype=np.int64),
                'gender': np.array(gender, dtype=np.int64),
                'codes': np.array(codes, dtype=np.int64),
                'dt_cnt': int(row['dt_cnt']),
                'target': target  # Keep as list for now
            })
        
        print(f"Pre-processing complete! {len(self.samples)} samples ready.")
    
    def _parse_age_gender(self, ipt):
        """Parse age or gender string: '540*541*542*...' -> [540, 541, 542, ...]"""
        ipt = ipt.split('*')
        ipt = ipt[:len_dy]
        ipt = [min(int(cd), 1439) if cd != '' else 0 for cd in ipt]
        ipt = ipt + (len_dy - len(ipt)) * [0]
        return ipt
    
    def _parse_codes(self, ipt):
        """Parse medical codes: '123,456*789,101*...' -> [[123,456,...], [789,101,...], ...]"""
        ipt = ipt.split('*')
        ipt = ipt[:len_dy]
        ipt = ipt + (len_dy - len(ipt)) * ['']
        ipt = [dy.split(',') for dy in ipt]
        ipt = [[int(cd) if cd != '' else 0 for cd in dy] for dy in ipt]
        ipt = [dy + (len_cd - len(dy)) * [0] for dy in ipt]
        return ipt
    
    def _parse_target(self, target):
        """Parse target codes"""
        target = target.split('*')
        target = target[:len_dy]
        target = [dy.split(',') for dy in target]
        target = [[int(cd) if cd != '' else 0 for cd in dy] for dy in target]
        return target
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return one sample as tensors"""
        sample = self.samples[idx]
        
        return {
            'age': torch.from_numpy(sample['age']),
            'gender': torch.from_numpy(sample['gender']),
            'codes': torch.from_numpy(sample['codes']),
            'dt_cnt': sample['dt_cnt'],
            'target': sample['target']  # Keep as list
        }


def load_from_bigquery(table_name, project_id, sample_size=None):
    """
    Load training data from BigQuery table.
    
    Args:
        table_name: Full BigQuery table name (e.g., 'dataset.table')
        project_id: GCP project ID
        sample_size: Optional - limit number of rows for testing
    
    Returns:
        pandas DataFrame with training data
    """
    from google.cloud import bigquery
    
    print(f"\n{'='*80}")
    print(f"Loading data from BigQuery...")
    print(f"{'='*80}")
    print(f"Table: {table_name}")
    
    client = bigquery.Client(project=project_id)
    
    # Build query
    query = f"""
    SELECT 
        individual_id,
        age_in_months,
        gender_cd,
        cd,
        dt_cnt,
        target
    FROM `{table_name}`
    WHERE dt_cnt >= {minimum_mth_training}  -- Filter minimum history
    """
    
    if sample_size:
        query += f"\nLIMIT {sample_size}"
    
    print(f"\nExecuting query...")
    print(f"Minimum history filter: {minimum_mth_training} days")
    if sample_size:
        print(f"Sample size limit: {sample_size:,} rows")
    
    # Load data
    df = client.query(query).to_dataframe()
    
    print(f"✓ Loaded {len(df):,} samples from BigQuery")
    print(f"{'='*80}\n")
    
    return df


def export_bigquery_to_gcs_pickles(table_name, project_id, bucket_name, gcs_path, num_shards=10):
    """
    Export BigQuery table to GCS as pickle files (one-time setup).
    
    Args:
        table_name: BigQuery table name
        project_id: GCP project ID  
        bucket_name: GCS bucket name
        gcs_path: Path within bucket (e.g., 'data/training')
        num_shards: Number of pickle files to create (for parallel loading)
    """
    print("\n" + "="*80)
    print("📦 EXPORTING BIGQUERY DATA TO GCS PICKLES (ONE-TIME SETUP)")
    print("="*80)
    
    # Load full dataset from BigQuery
    df = load_from_bigquery(table_name, project_id, sample_size=None)
    
    print(f"\nSharding data into {num_shards} files...")
    shard_size = len(df) // num_shards
    
    # Upload to GCS
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < num_shards - 1 else len(df)
        shard = df.iloc[start_idx:end_idx]
        
        # Save locally first
        local_path = f'/tmp/data_shard_{i}.pkl'
        shard.to_pickle(local_path)
        
        # Upload to GCS
        gcs_file_path = f'{gcs_path}/data_shard_{i}.pkl'
        blob = bucket.blob(gcs_file_path)
        blob.upload_from_filename(local_path)
        
        print(f"  ✓ Uploaded shard {i+1}/{num_shards} ({len(shard):,} samples) to gs://{bucket_name}/{gcs_file_path}")
        
        # Clean up local file
        os.remove(local_path)
    
    print("\n" + "="*80)
    print(f"✅ Export complete! {num_shards} files saved to gs://{bucket_name}/{gcs_path}/")
    print("="*80)
    print("You can now set USE_GCS_PICKLES = True for fast, cost-free loading!")
    print("="*80 + "\n")


def load_from_gcs_pickles(bucket_name, gcs_path, project_id, num_shards=10):
    """
    Load data from GCS pickle files.
    
    Args:
        bucket_name: GCS bucket name
        gcs_path: Path within bucket where pickles are stored
        project_id: GCP project ID
        num_shards: Number of pickle files to load
    
    Returns:
        Combined pandas DataFrame
    """
    print(f"\n{'='*80}")
    print(f"Loading data from GCS pickles...")
    print(f"{'='*80}")
    print(f"Location: gs://{bucket_name}/{gcs_path}/")
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    dfs = []
    for i in range(num_shards):
        gcs_file_path = f'{gcs_path}/data_shard_{i}.pkl'
        blob = bucket.blob(gcs_file_path)
        
        # Download to memory
        pickle_bytes = BytesIO()
        blob.download_to_file(pickle_bytes)
        pickle_bytes.seek(0)
        
        # Load pickle
        df = pd.read_pickle(pickle_bytes)
        dfs.append(df)
        
        print(f"  ✓ Loaded shard {i+1}/{num_shards} ({len(df):,} samples)")
    
    # Combine all shards
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n✓ Total loaded: {len(combined_df):,} samples")
    print(f"{'='*80}\n")
    
    return combined_df


def load_and_prepare_data(shuffle=True):
    """
    Load data and create PyTorch DataLoader.
    Uses configuration from top of file (USE_GCS_PICKLES, BIGQUERY_TABLE, etc.)
    
    Args:
        shuffle: whether to shuffle data
    
    Returns:
        DataLoader ready for training
    """
    if USE_GCS_PICKLES:
        # Load from GCS pickles (fast, no query cost)
        df = load_from_gcs_pickles(
            bucket_name=GCS_BUCKET,
            gcs_path=GCS_DATA_PATH,
            project_id=GCP_PROJECT,
            num_shards=10
        )
    else:
        # Load from BigQuery (flexible, costs per query)
        df = load_from_bigquery(
            table_name=BIGQUERY_TABLE,
            project_id=GCP_PROJECT,
            sample_size=SAMPLE_SIZE
        )
    
    # Shuffle if requested
    if shuffle:
        print("Shuffling data...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create PyTorch Dataset (this pre-processes everything)
    print("\nCreating PyTorch Dataset...")
    dataset = ClinicalDataset(df, target_col=target)
    
    # Create DataLoader with multiple workers
    print(f"\nCreating DataLoader with batch_size={batch_size}...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Already shuffled above
        num_workers=8,   # 8 parallel workers for data loading
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Pre-load 2 batches per worker
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True  # Drop incomplete last batch
    )
    
    print(f"DataLoader ready! {len(dataloader)} batches per epoch")
    print(f"{'='*80}\n")
    
    return dataloader


def currentTime():
    newYorkTz = pytz.timezone("America/New_York") 
    timeInNewYork = datetime.now(newYorkTz)
    currentTimeInNewYork = timeInNewYork.strftime("%D %H:%M:%S")
    return currentTimeInNewYork

# HELPER PARSING FUNCTIONS - kept for fine-tuning section compatibility
def conv_cd(ipt):
    ipt = ipt.split('*')
    ipt = ipt[:len_dy]
    ipt = ipt + (len_dy-len(ipt))*['']
    ipt = [dy.split(',') for dy in ipt]
    ipt = [[int(cd) if cd!='' else 0 for cd in dy] for dy in ipt]
    ipt = [dy + (len_cd-len(dy))*[0] for dy in ipt]
    return ipt

def conv_age_gender(ipt):
    ipt = ipt.split('*')
    ipt = ipt[:len_dy]
    ipt = [min(int(cd),1439) for cd in ipt]
    ipt = ipt + (len_dy-len(ipt))*[0]
    return ipt

def conv_target(target):
    target = target.split('*')
    target = target[:len_dy]
    target = target
    target = [dy.split(',') for dy in target]
    target = [[int(cd) if cd!='' else 0 for cd in dy] for dy in target]
    return target

# def prepare_tensor(batch):
#     age_in_months = [conv_age_gender(ipt) for ipt in batch['age_in_months'].tolist()]
#     age_in_months = torch.tensor(age_in_months).to(device)
#     age_in_months = age_in_months.reshape(batch_size,len_dy,1)
#     
#     gender_cd = [conv_age_gender(ipt) for ipt in batch['gender_cd'].tolist()]
#     gender_cd = torch.tensor(gender_cd).to(device)
#     gender_cd = gender_cd.reshape(batch_size,len_dy,1)    
#     
#     cd = [conv_cd(ipt) for ipt in batch['cd'].tolist()]
#     cd = torch.tensor(cd).to(device)
#     
#     x = torch.cat([age_in_months,gender_cd,cd],dim=-1)    
#     
#     dt_cnt = batch['dt_cnt'].tolist()
#     y = [conv_target(target) for target in batch[target].tolist()]
#     
#     return dt_cnt,x,y


# OLD TRAINING FUNCTIONS - commented out (replaced by train_epoch/val_epoch)
# def train(data):
#     model.train()
#     
#     nbatch = int(data.shape[0]/batch_size)
#     for i in range(nbatch):
#         if i%1000 == 0:
#             print('batch',i,currentTime())
#         optimizer.zero_grad()
#         batch = data.iloc[i*batch_size:i*batch_size+batch_size,:]
#         dt_cnt,x,y = prepare_tensor(batch)
#
#         opt = model(x)
#         opt = opt.reshape(batch_size*len_dy,target_cd_cnt)
#         y = [item for sublist in y for item in sublist]
#         
#         opt = torch.cat([opt[len_dy*i:len_dy*i+dt_cnt[i],:] for i in range(batch_size)],dim=0)
#
#         y_cd = torch.zeros(len(opt),target_cd_cnt).to(device)
#
#         for j in range(len(opt)):
#             for k in y[j]:
#                 if k!=0:
#                     y_cd[j,k]=1         
#
#         loss = criterion(opt, y_cd)        
#         
#         loss.backward()
#         optimizer.step()
#         
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
#         for p in model.parameters():
#             p.data.add_(p.grad, alpha=-optimizer.param_groups[0]['lr'])
#
#         del batch,x,y_cd,opt,loss
#         gc.collect()
#         torch.cuda.empty_cache()
#         
# def val(data):
#     model.eval()
#     nbatch = int(data.shape[0]/batch_size)
#     total_loss = 0
#     for i in range(nbatch):
#         if i%1000 == 0:
#             print('batch',i,currentTime())
#         optimizer.zero_grad()
#         batch = data.iloc[i*batch_size:i*batch_size+batch_size,:]
#         dt_cnt,x,y = prepare_tensor(batch)
#
#         opt = model(x)
#         opt = opt.reshape(batch_size*len_dy,target_cd_cnt)
#         y = [item for sublist in y for item in sublist]
#         
#         opt = torch.cat([opt[len_dy*i:len_dy*i+dt_cnt[i],:] for i in range(batch_size)],dim=0)
#
#         y_cd = torch.zeros(len(opt),target_cd_cnt).to(device)
#
#         for j in range(len(opt)):
#             for k in y[j]:
#                 if k!=0:
#                     y_cd[j,k]=1         
#
#         loss = criterion(opt, y_cd)        
#         total_loss += float(loss)
#
#         del batch,x,y_cd,opt,loss
#         gc.collect()
#         torch.cuda.empty_cache()
#     return total_loss/(nbatch*batch_size)


# ============================================================================
# NEW TRAINING FUNCTIONS - work with PyTorch DataLoader
# ============================================================================

def train_epoch(dataloader):
    """Train for one epoch using PyTorch DataLoader"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{num_batches}', currentTime())
        
        optimizer.zero_grad()
        
        # Unpack batch (already pre-processed!)
        age_in_months = batch['age'].to(device, non_blocking=True)  # [batch, 200]
        gender_cd = batch['gender'].to(device, non_blocking=True)    # [batch, 200]
        codes = batch['codes'].to(device, non_blocking=True)         # [batch, 200, 80]
        dt_cnt = batch['dt_cnt']  # List of ints
        targets = batch['target']  # List of lists
        
        # Reshape for model input
        age_in_months = age_in_months.unsqueeze(-1)  # [batch, 200, 1]
        gender_cd = gender_cd.unsqueeze(-1)          # [batch, 200, 1]
        
        # Concatenate inputs
        x = torch.cat([age_in_months, gender_cd, codes], dim=-1)  # [batch, 200, 82]
        
        # Forward pass
        output = model(x)  # [batch, 200, target_cd_cnt]
        output = output.reshape(-1, target_cd_cnt)  # [batch*200, target_cd_cnt]
        
        # Flatten targets
        targets_flat = [item for sublist in targets for item in sublist]
        
        # Select only valid timesteps (based on dt_cnt)
        valid_outputs = torch.cat([output[len_dy*i:len_dy*i+dt_cnt[i],:] for i in range(len(dt_cnt))], dim=0)
        
        # Create target tensor (multi-label)
        y_cd = torch.zeros(len(valid_outputs), target_cd_cnt, device=device)
        for j in range(len(valid_outputs)):
            for k in targets_flat[j]:
                if k != 0:
                    y_cd[j, k] = 1
        
        # Compute loss
        loss = criterion(valid_outputs, y_cd)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        
    avg_loss = total_loss / num_batches
    print(f'Training complete. Average loss: {avg_loss:.4f}')
    return avg_loss


def val_epoch(dataloader):
    """Validate for one epoch using PyTorch DataLoader"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():  # No gradients needed for validation
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f'Val Batch {batch_idx}/{num_batches}', currentTime())
            
            # Unpack batch
            age_in_months = batch['age'].to(device, non_blocking=True)
            gender_cd = batch['gender'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            targets = batch['target']
            
            # Reshape for model input
            age_in_months = age_in_months.unsqueeze(-1)
            gender_cd = gender_cd.unsqueeze(-1)
            
            # Concatenate inputs
            x = torch.cat([age_in_months, gender_cd, codes], dim=-1)
            
            # Forward pass
            output = model(x)
            output = output.reshape(-1, target_cd_cnt)
            
            # Flatten targets
            targets_flat = [item for sublist in targets for item in sublist]
            
            # Select only valid timesteps
            valid_outputs = torch.cat([output[len_dy*i:len_dy*i+dt_cnt[i],:] for i in range(len(dt_cnt))], dim=0)
            
            # Create target tensor
            y_cd = torch.zeros(len(valid_outputs), target_cd_cnt, device=device)
            for j in range(len(valid_outputs)):
                for k in targets_flat[j]:
                    if k != 0:
                        y_cd[j, k] = 1
            
            # Compute loss
            loss = criterion(valid_outputs, y_cd)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f'Validation complete. Average loss: {avg_loss:.4f}')
    return avg_loss
        
def save_checkpoint(model,optimizer,epoch,stage,dataid):
    checkpoint = dict()
    checkpoint['timestamp'] = str(currentTime())
    if parallel == True:
        checkpoint['model'] = model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
    checkpoint['optimizer'] = optimizer
    checkpoint['current_epoch'] = epoch  
    checkpoint['current_stage'] = stage
    checkpoint['current_dataid'] = dataid
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'checkpoint'))
    with blob.open("wb", ignore_flush=True) as f:
        joblib.dump(checkpoint, f) 

def save_bestmodel(model,optimizer,epoch,best_val_loss):
    global bestModel
    bestModel['timestamp'] = str(currentTime())
    if parallel == True:
        bestModel['model'] = model.module.state_dict()
    else:
        bestModel['model'] = model.state_dict()
    bestModel['optimizer'] = optimizer
    bestModel['result_epoch'+str(epoch)] = best_val_loss  
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'bestModel'))
    with blob.open("wb", ignore_flush=True) as f:
        joblib.dump(bestModel, f) 
        
        
# OLD run_epochs function - commented out (uses old dataLoader/train/val functions)
# def run_epochs(total_epochs,val_firstid,val_lastid):
#     global bestModel,unfinished_epoch,unfinished_stage,unfinished_dataid,best_val_loss,model
#
#     epoch = unfinished_epoch
#     
#     while epoch>=unfinished_epoch and epoch<total_epochs:
#         print('#########################')
#         print('working on epoch',epoch)
#         if unfinished_stage == 'training':
#             for dataid in range(unfinished_dataid,val_firstid):
#                 print('training...',dataid,currentTime())
#                 fileid = data_source+str(dataid)+'.p' 
#                 data = dataLoader(fileid)
#                 # data = data.head(1000)
#                 if minimum_mth_training>0:
#                     data = data[data['dt_cnt']>=minimum_mth_training].reset_index(drop=True)
#                 train(data)
#                 if dataid == val_firstid-1:
#                     save_checkpoint(model,optimizer,epoch,'validating',dataid + 1)
#                 else:
#                     save_checkpoint(model,optimizer,epoch,'training',dataid + 1)
#             unfinished_stage = 'validating'
#         else:
#             total_loss = 0
#             for dataid in range(val_firstid,val_lastid+1):
#                 print('validating...',dataid,currentTime())
#                 fileid = data_source+str(dataid)+'.p' 
#                 data = dataLoader(fileid) 
#                 total_loss += val(data)
#             total_loss = total_loss/(val_lastid-val_firstid+1)
#
#             if (not best_val_loss) or (total_loss < best_val_loss):
#                 best_val_loss = total_loss
#                 save_bestmodel(model,optimizer,epoch,best_val_loss)             
#                 unfinished_stage = 'training'
#                 unfinished_dataid = 0
#                 epoch += 1
#                 unfinished_epoch += 1
#                 save_checkpoint(model,optimizer,epoch,unfinished_stage,unfinished_dataid)
#                 print('improved...saved best model','loss:',best_val_loss)
#                 print('before',optimizer.param_groups[0]["lr"])
#                 scheduler.step()
#                 print('after',optimizer.param_groups[0]["lr"])
#             else:
#                 print('stopped....no improving')
#                 break    

# Model checkpoint storage (for saving/loading trained models)
bucket_name = "us-east4-edp-prod-css-sdoh--1b0f6fa9-bucket"  # Where to save model checkpoints
model_path = 'a834793_transformer/Model'  # Path for checkpoint and bestModel files

# Old data_source variable (no longer used with new PyTorch DataLoader approach)
# data_source = 'a321276/TransformerV9/Data/a321276_o3_'  # Removed - using GCS_DATA_PATH instead


batch_size = 512
embedding_size = 256
minimum_mth_training = 180   # Minimum days of history (180 days ≈ 6 months)
len_dy = 200 # how many days in the seq
len_cd = 80 # within a day how many cds. 
nhead = 16 # heads of transformer - double transformer share same feature...
nhid = 512 # number of hidden of transformer - double transformer share same feature...
nlayers = 6 # number of layers of transformer - double transformer share same feature...
ndropout = 0.05 # dropout rate of transformer - double transformer share same feature...
cd_cnt = 84010 # numbr of codes used in embedding matrix
target_cd_cnt = 2767 # numbr of target codes 
criterion = nn.BCEWithLogitsLoss() # multi-label loss
parallel = True
device = torch.device("cuda")
target = 'target'
try:
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'bestModel'))
    bestModel = BytesIO()
    blob.download_to_file(bestModel)
    bestModel=joblib.load(bestModel)  
    best_model = bestModel['model']
    best_val_loss = 'result_epoch'+str(max([int(key.split('result_epoch')[1]) for key in bestModel.keys()  if 'result_epoch' in key]))
    best_val_loss = bestModel[best_val_loss]
    print('results loaded','best_loss',best_val_loss)
except:
    bestModel = dict()
    best_val_loss = None
    print('no result found')
    
try:
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'checkpoint'))
    checkpoint = BytesIO()
    blob.download_to_file(checkpoint)
    checkpoint=joblib.load(checkpoint)  
    model = TransformerModel(nhead, nhid, nlayers, ndropout)
    model.load_state_dict(checkpoint['model'])
    if parallel==True:
        model= nn.DataParallel(model)
    model = model.to(device)    
    optimizer = checkpoint['optimizer']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    unfinished_epoch = checkpoint['current_epoch']
    unfinished_stage = checkpoint['current_stage']
    unfinished_dataid = checkpoint['current_dataid']
    print('model loaded','unfinished_epoch',unfinished_epoch,'unfinished_stage',unfinished_stage,'unfinished_dataid',unfinished_dataid)

except:
    print('new model')
    model = TransformerModel(nhead, nhid, nlayers, ndropout)
    if parallel==True:
        model= nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    unfinished_epoch = 0
    unfinished_stage = 'training'
    unfinished_dataid = 0


# =============================================================================
# TRAINING EXECUTION
# =============================================================================

if RUN_TRAINING:
    print("\n" + "="*80)
    print("🚀 Starting training with PyTorch DataLoader approach...")
    print("="*80)
    
    # Load and prepare data (train/val split)
    print("\nLoading and preparing data...")
    full_loader = load_and_prepare_data(shuffle=True)
    
    # Get full dataset for train/val split
    full_dataset = full_loader.dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create separate dataloaders for train and val
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False
    )
    
    print(f"✓ Training set: {len(train_dataset):,} samples ({len(train_loader)} batches)")
    print(f"✓ Validation set: {len(val_dataset):,} samples ({len(val_loader)} batches)")
    
    # Training loop
    for epoch in range(10):
        print(f"\n{'='*80}")
        print(f"📊 Epoch {epoch+1}/10")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(train_loader)
        
        # Validate
        val_loss = val_epoch(val_loader)
        
        # Save best model
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            save_bestmodel(model, optimizer, epoch, best_val_loss)
            print(f"✅ New best model saved! Validation loss: {best_val_loss:.4f}")
        else:
            print(f"⚠️  No improvement. Best loss: {best_val_loss:.4f}, Current: {val_loss:.4f}")
        
        # Learning rate schedule
        scheduler.step()
        print(f"📉 Learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")
    
    print("\n🎉 Training complete!")

else:
    print("\n" + "="*80)
    print("ℹ️  Training is DISABLED")
    print("="*80)
    print("To start training, set RUN_TRAINING = True at the top of this file (line 8)")
    print("="*80 + "\n")


# =============================================================================
# OPTIONAL: Export BigQuery data to GCS pickles (run this once)
# =============================================================================
# 
# To export your BigQuery data to GCS pickles (recommended for cost savings):
#
# 1. Uncomment the code below
# 2. Run this script once
# 3. After export completes, set USE_GCS_PICKLES = True at the top
# 4. Comment this section out again
#
"""
print("\n" + "="*80)
print("🔧 ONE-TIME SETUP: Exporting BigQuery data to GCS pickles")
print("="*80)

export_bigquery_to_gcs_pickles(
    table_name=BIGQUERY_TABLE,
    project_id=GCP_PROJECT,
    bucket_name=GCS_BUCKET,
    gcs_path=GCS_DATA_PATH,
    num_shards=10
)

print("✅ Export complete! You can now:")
print("1. Set USE_GCS_PICKLES = True at the top of this file")
print("2. Comment out this export section")  
print("3. Run training - it will use the pickles (fast & cheap!)")
print("="*80 + "\n")
"""