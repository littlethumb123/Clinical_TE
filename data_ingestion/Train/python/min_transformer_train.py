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

def dataLoader(fileid):
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(fileid)
    data = BytesIO()
    blob.download_to_file(data)
    data=joblib.load(data)
    data = data.sample(frac=1)
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

    if target in batch.columns:
        y = [conv_dy(target) for target in batch[target].tolist()]
        return dt_cnt,x,y
    else:
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
        
def val(data):
    model.eval()
    nbatch = int(data.shape[0]/batch_size)
    total_loss = 0
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
        total_loss += float(loss)
        
        del batch,x,y,opt,loss
        gc.collect()
        torch.cuda.empty_cache()
    return total_loss/(nbatch*batch_size)
        
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
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'checkpoint_ip'))
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
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'bestModel_ip'))
    with blob.open("wb", ignore_flush=True) as f:
        joblib.dump(bestModel, f) 
        
def run_epochs(total_epochs,val_firstid,val_lastid):
    global bestModel,unfinished_epoch,unfinished_stage,unfinished_dataid,best_val_loss,model

    epoch = unfinished_epoch
    
    while epoch>=unfinished_epoch and epoch<total_epochs:
        print('#########################')
        print('working on epoch',epoch)
        if unfinished_stage == 'training':
            for dataid in range(unfinished_dataid,val_firstid):
                print('training...',dataid,currentTime())
                fileid = data_source+str(dataid)+'.p' 
                data = dataLoader(fileid)
                # data = data.head(1000)
                if minimum_mth_training>0:
                    data = data[data['dt_cnt']>=minimum_mth_training].reset_index(drop=True)
                train(data)
                if dataid == val_firstid-1:
                    save_checkpoint(model,optimizer,epoch,'validating',dataid + 1)
                else:
                    save_checkpoint(model,optimizer,epoch,'training',dataid + 1)
            unfinished_stage = 'validating'
        else:
            total_loss = 0
            for dataid in range(val_firstid,val_lastid+1):
                print('validating...',dataid,currentTime())
                fileid = data_source+str(dataid)+'.p' 
                data = dataLoader(fileid) 
                # data = data.head(1000)
                total_loss += val(data)
            total_loss = total_loss/(val_lastid-val_firstid+1)

            if (not best_val_loss) or (total_loss < best_val_loss):
                best_val_loss = total_loss
                save_bestmodel(model,optimizer,epoch,best_val_loss)             
                unfinished_stage = 'training'
                unfinished_dataid = 0
                epoch += 1
                unfinished_epoch += 1
                save_checkpoint(model,optimizer,epoch,unfinished_stage,unfinished_dataid)
                print('improved...saved best model','loss:',best_val_loss)
                print('before',optimizer.param_groups[0]["lr"])
                scheduler.step()
                print('after',optimizer.param_groups[0]["lr"])

            else:
                print('stopped....no improving')
                break    
                
bucket_name = "provider-ds-data-hcb-dev"
data_source = 'a321276/TransformerV9/Data/a321276_o3_'
model_path = 'a321276/TransformerV9/Model'


batch_size = 256
embedding_size = 256
minimum_mth_training = 6   #filter out data points which short month length
len_dy = 70 # how many days in th seq
len_cd = 25 # within a day how many cds. 
nhead = 16 # heads of transformer - double transformer share same feature...
nhid = 512 # number of hidden of transformer - double transformer share same feature...
nlayers = 6 # number of layers of transformer - double transformer share same feature...
ndropout = 0.1 # dropout rate of transformer - double transformer share same feature...
cd_cnt = 98041 # numbr of codes used in embedding matrix
target_cd_cnt = 2 # numbr of target codes 
criterion = nn.NLLLoss()
parallel = False
device = torch.device("cuda:0")


try:
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'bestModel_ip'))
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
    blob = storage.Client(credentials=google.auth.default()[0]).bucket(bucket_name).blob(os.path.join(model_path,'checkpoint_ip'))
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
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    unfinished_epoch = 0
    unfinished_stage = 'training'
    unfinished_dataid = 0
    
target = 'ip_3m'
run_epochs(10,val_firstid=9,val_lastid=9)
    
    
    
 