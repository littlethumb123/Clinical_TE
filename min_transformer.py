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
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")


batch_size = 16  #512, Jane changed to run on t4!!!
embedding_size = 256
minimum_mth_training = 6   #filter out data points which short month length
len_dy = 200 # how many days in th seq
len_cd = 80 # within a day how many cds. 
nhead = 16 # heads of transformer - double transformer share same feature...
nhid = 512 # number of hidden of transformer - double transformer share same feature...
nlayers = 6 # number of layers of transformer - double transformer share same feature...
ndropout = 0.1 # dropout rate of transformer - double transformer share same feature...
cd_cnt = 84010 # numbr of codes used in embedding matrix
target_cd_cnt = 2767 # numbr of target codes 
criterion = nn.NLLLoss()
parallel = True
device = torch.device("cuda:0") 
#device = torch.device("cuda") #Jane changed to run on t4!!!


class TransformerModel(nn.Module):
    def __init__(self, nhead, nhid, nlayers, dropout=0.05):
        super(TransformerModel, self).__init__()
        self.embedding_cd = nn.Embedding(cd_cnt,embedding_size)
        self.embedding_cd.weight.requires_grad = True
        self.embedding_gender_cd = nn.Embedding(4,embedding_size)
        self.embedding_gender_cd.weight.requires_grad = True
        self.embedding_age_in_months = nn.Embedding(1440, embedding_size)  
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
    

def dataLoader(fileid):
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(fileid)
    data = BytesIO()
    blob.download_to_file(data)
    data=joblib.load(data)
    return data
def currentTime():
    newYorkTz = pytz.timezone("America/New_York") 
    timeInNewYork = datetime.now(newYorkTz)
    currentTimeInNewYork = timeInNewYork.strftime("%D %H:%M:%S")
    return currentTimeInNewYork
 
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
 
def conv_dy(x):
    x = x.split('*')
    x = x[:len_dy]
    x = [int(cd) for cd in x]
    return x
def prepare_tensor(batch):
    age_in_months = [conv_age_gender(ipt) for ipt in batch['age_in_months'].tolist()]
    age_in_months = torch.tensor(age_in_months).to(device)
    age_in_months = age_in_months.reshape(batch_size,len_dy,1)
    gender_cd = [conv_age_gender(ipt) for ipt in batch['gender_cd'].tolist()]
    gender_cd = torch.tensor(gender_cd).to(device)
    gender_cd = gender_cd.reshape(batch_size,len_dy,1)    
    cd = [conv_cd(ipt) for ipt in batch['cd'].tolist()]
    cd = torch.tensor(cd).to(device)
    x = torch.cat([age_in_months,gender_cd,cd],dim=-1)
 
    dt_cnt = batch['dt_cnt'].tolist()
 
    return dt_cnt,x
def train(data):
    model.train()
    nbatch = int(data.shape[0]/batch_size)
    for i in range(nbatch):
        if i%1000 == 0:
            print('batch',i,currentTime())
        optimizer.zero_grad()
        batch = data.iloc[i*batch_size:i*batch_size+batch_size,:]
        dt_cnt,x,y = prepare_tensor(batch)
        opt = model(x)
        opt = opt.reshape(batch_size*len_dy,target_cd_cnt)
        y = [item for sublist in y for item in sublist]
        opt = torch.cat([opt[len_dy*i:len_dy*i+dt_cnt[i],:] for i in range(batch_size)],dim=0)
        y = torch.tensor(y).to(device)
        loss = criterion(opt, y)        
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-optimizer.param_groups[0]['lr'])
 
        del batch,x,y,opt,loss
        gc.collect()
        torch.cuda.empty_cache()
        
def score(model, data):
    model.eval()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
 
    model.transformer_encoder_dy.register_forward_hook(get_activation('transformer_encoder_dy'))
 
    dsize = data.shape[0]
    nbatch = int(dsize/batch_size)
    
    #print('dsize:', dsize, 'nbatch:', nbatch)
    
    #print('exec, calc dsize-nbatch*batch_size:', dsize-nbatch*batch_size, dsize, nbatch, batch_size)
    
    if dsize-nbatch*batch_size>0:
        k = batch_size - (dsize-nbatch*batch_size) # fill some records to build last trunk; these will be deleted in the end        
        data = pd.concat([data, pd.concat([data.head(1)]* k,  ignore_index=True)])  # changed in case k is greater than dsize
        #print('1st appended, k=', k, ', dsize=', dsize, ', batch_size=', batch_size, 'nbatch =', nbatch, 'condition:', dsize-nbatch*batch_size, 'fill=', fill.shape)
                    
    data = data.reset_index(drop=True)
    nbatch = int(data.shape[0]/batch_size)
    
    ys = []
    for i in range(nbatch):
        batch = data.iloc[i*batch_size:i*batch_size+batch_size,:]
        #print('batch:', batch.shape, i)
        
        dt_cnt,x = prepare_tensor(batch)
        opts = model(x)
        intermedia_output = activation['transformer_encoder_dy']       
        intermedia_output = [intermedia_output[dt_cnt[i],i,:].reshape(1,-1) for i in range(batch_size)]
        intermedia_output = torch.cat(intermedia_output)       
        #print('intermedia_output:', intermedia_output.shape, i)
        
        ys.append(intermedia_output)
        
    ys = torch.cat(ys).cpu().numpy()
    ys = pd.DataFrame(ys,columns = ['emb'+str(i) for i in range(embedding_size)])
    ys[entity_id] = data[entity_id]
    ys = ys.head(dsize)
    return ys   



def load_model_state():
    # Read in parameters from GCP bucket
    bucket_name = "clin-analytics-data-hcb-dev" # model from Elle
    model_path = 'a534354/TransformerV10/Model'
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'bestModel'))
    bestModel = BytesIO()
    blob.download_to_file(bestModel)
    bestModel=joblib.load(bestModel)  

    # Create transformer model
    model = TransformerModel(nhead, nhid, nlayers, 0)
    model.load_state_dict(bestModel['model'])
    model = model.to(device)


# Create a function for extracting one's daily embedding and used that for LIME interpretation. 
def get_daily_embedding(model, data):
    """ Get member's daily embedding
    """
    model.eval()
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model.transformer_encoder_dy.register_forward_hook(get_activation('transformer_encoder_dy'))
 
    dsize = data.shape[0]
    nbatch = int(dsize/batch_size)
    
    if dsize-nbatch*batch_size>0:
        k = batch_size - (dsize-nbatch*batch_size) # fill some records to build last trunk; these will be deleted in the end        
        data = pd.concat([data, pd.concat([data.head(1)]* k,  ignore_index=True)])  # changed in case k is greater than dsize
                    
    data = data.reset_index(drop=True)
    nbatch = int(data.shape[0]/batch_size)
    all_embeddings = []
    ys = []
    for i in tqdm(range(nbatch)):
        batch = data.iloc[i*batch_size:i*batch_size+batch_size,:]
        
        # get patient IDs before tensor conversion
        individual_ids = batch['individual_id'].tolist()
        
        dt_cnt,x = prepare_tensor(batch)
        opts = model(x)
        day_embeddings = activation['transformer_encoder_dy']  
        day_embeddings = torch.swapaxes(day_embeddings, 0, 1)
        
        # collect daily embedding for each member
        for mbr_idx in range(batch_size):
            # Skip padding entries
            if i*batch_size + mbr_idx >= dsize:
                continue
                
            mbr_id = batch[entity_id].iloc[mbr_idx]
            valid_days = dt_cnt[mbr_idx] + 1 # adjust day by adding 1 
            
            # extract embeddings for each day of this patient's history
            for day_idx in range(1, valid_days):  # from day1 
                # get embedding for this specific day
                embedding = day_embeddings[mbr_idx, day_idx, :].cpu().numpy()
                
                # create a row with patient ID, day index, and embedding values
                embedding_dict = {
                    entity_id: mbr_id,
                    'day_idx': day_idx,
                }
                
                # add embedding dimensions
                for j in range(embedding_size):
                    embedding_dict[f'emb{j}'] = embedding[j]
                
                all_embeddings.append(embedding_dict)
        # clear memory before each batch to avoid memory overflow
        torch.cuda.empty_cache()
        gc.collect()
    return pd.DataFrame(all_embeddings)    