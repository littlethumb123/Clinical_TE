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

batch_size = 512
embedding_size = 256
minimum_mth_training = 180
len_dy = 200
len_cd = 80
nhead = 16
nhid = 512
nlayers = 6
ndropout = 0.05
cd_cnt = 84010
target_cd_cnt = 2767
parallel = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
entity_id = 'individual_id'
target = 'target'


class TransformerModel(nn.Module):
    def __init__(self, nhead, nhid, nlayers, dropout=0.05):
        super(TransformerModel, self).__init__()
        self.embedding_cd = nn.Embedding(cd_cnt, embedding_size)
        self.embedding_cd.weight.requires_grad = True
        self.embedding_gender_cd = nn.Embedding(4, embedding_size)
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
        cd = F.log_softmax(cd, dim=-1)
        return cd


class ClinicalDataset(Dataset):
    def __init__(self, df, target_col='target'):
        self.samples = []
        self.target_col = target_col
        if minimum_mth_training > 0:
            df = df[df['dt_cnt'] >= minimum_mth_training].reset_index(drop=True)
        for idx in range(len(df)):
            row = df.iloc[idx]
            age = self._parse_age_gender(row['age_in_months'])
            gender = self._parse_age_gender(row['gender_cd'])
            codes = self._parse_codes(row['cd'])
            if target_col in row:
                target_val = self._parse_target(row[target_col])
            else:
                target_val = []
            self.samples.append({
                'age': np.array(age, dtype=np.int64),
                'gender': np.array(gender, dtype=np.int64),
                'codes': np.array(codes, dtype=np.int64),
                'dt_cnt': int(row['dt_cnt']),
                'target': target_val,
                entity_id: row[entity_id] if entity_id in row else None
            })

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


def currentTime():
    newYorkTz = pytz.timezone("America/New_York")
    timeInNewYork = datetime.now(newYorkTz)
    return timeInNewYork.strftime("%D %H:%M:%S")


def conv_cd(ipt):
    ipt = ipt.split('*')
    ipt = ipt[:len_dy]
    ipt = ipt + (len_dy - len(ipt)) * ['']
    ipt = [dy.split(',') for dy in ipt]
    ipt = [[int(cd) if cd != '' else 0 for cd in dy] for dy in ipt]
    ipt = [dy + (len_cd - len(dy)) * [0] for dy in ipt]
    return ipt


def conv_age_gender(ipt):
    ipt = ipt.split('*')
    ipt = ipt[:len_dy]
    ipt = [min(int(cd), 1439) for cd in ipt]
    ipt = ipt + (len_dy - len(ipt)) * [0]
    return ipt


def conv_target(target_str):
    target_str = target_str.split('*')
    target_str = target_str[:len_dy]
    target_str = [dy.split(',') for dy in target_str]
    target_str = [[int(cd) if cd != '' else 0 for cd in dy] for dy in target_str]
    return target_str


def prepare_tensor(batch, current_batch_size):
    age_in_months = [conv_age_gender(ipt) for ipt in batch['age_in_months'].tolist()]
    age_in_months = torch.tensor(age_in_months).to(device)
    age_in_months = age_in_months.reshape(current_batch_size, len_dy, 1)
    gender_cd = [conv_age_gender(ipt) for ipt in batch['gender_cd'].tolist()]
    gender_cd = torch.tensor(gender_cd).to(device)
    gender_cd = gender_cd.reshape(current_batch_size, len_dy, 1)
    cd = [conv_cd(ipt) for ipt in batch['cd'].tolist()]
    cd = torch.tensor(cd).to(device)
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)
    dt_cnt = batch['dt_cnt'].tolist()
    return dt_cnt, x


def create_dataloader(df, shuffle=True):
    dataset = ClinicalDataset(df, target_col=target)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    return dataloader


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{num_batches}', currentTime())
        optimizer.zero_grad()
        age_in_months = batch['age'].to(device, non_blocking=True)
        gender_cd = batch['gender'].to(device, non_blocking=True)
        codes = batch['codes'].to(device, non_blocking=True)
        dt_cnt = batch['dt_cnt']
        targets = batch['target']
        age_in_months = age_in_months.unsqueeze(-1)
        gender_cd = gender_cd.unsqueeze(-1)
        x = torch.cat([age_in_months, gender_cd, codes], dim=-1)
        output = model(x)
        output = output.reshape(-1, target_cd_cnt)
        targets_flat = [item for sublist in targets for item in sublist]
        valid_outputs = torch.cat([output[len_dy * i:len_dy * i + dt_cnt[i], :] for i in range(len(dt_cnt))], dim=0)
        y_cd = torch.zeros(len(valid_outputs), target_cd_cnt, device=device)
        for j in range(len(valid_outputs)):
            for k in targets_flat[j]:
                if k != 0:
                    y_cd[j, k] = 1
        loss = criterion(valid_outputs, y_cd)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
    avg_loss = total_loss / num_batches
    print(f'Training complete. Average loss: {avg_loss:.4f}')
    return avg_loss


def val_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f'Val Batch {batch_idx}/{num_batches}', currentTime())
            age_in_months = batch['age'].to(device, non_blocking=True)
            gender_cd = batch['gender'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            targets = batch['target']
            age_in_months = age_in_months.unsqueeze(-1)
            gender_cd = gender_cd.unsqueeze(-1)
            x = torch.cat([age_in_months, gender_cd, codes], dim=-1)
            output = model(x)
            output = output.reshape(-1, target_cd_cnt)
            targets_flat = [item for sublist in targets for item in sublist]
            valid_outputs = torch.cat([output[len_dy * i:len_dy * i + dt_cnt[i], :] for i in range(len(dt_cnt))], dim=0)
            y_cd = torch.zeros(len(valid_outputs), target_cd_cnt, device=device)
            for j in range(len(valid_outputs)):
                for k in targets_flat[j]:
                    if k != 0:
                        y_cd[j, k] = 1
            loss = criterion(valid_outputs, y_cd)
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    print(f'Validation complete. Average loss: {avg_loss:.4f}')
    return avg_loss


def score(model, data):
    model.eval()
    activation = {}

    def get_activation(name):
        def hook(model_hook, input_hook, output_hook):
            activation[name] = output_hook.detach()
        return hook

    handle = model.transformer_encoder_dy.register_forward_hook(get_activation('transformer_encoder_dy'))
    dsize = data.shape[0]
    nbatch = int(dsize / batch_size)
    if dsize - nbatch * batch_size > 0:
        k = batch_size - (dsize - nbatch * batch_size)
        data = pd.concat([data, pd.concat([data.head(1)] * k, ignore_index=True)])
    data = data.reset_index(drop=True)
    nbatch = int(data.shape[0] / batch_size)
    ys = []
    with torch.no_grad():
        for i in range(nbatch):
            batch = data.iloc[i * batch_size:i * batch_size + batch_size, :]
            dt_cnt, x = prepare_tensor(batch, batch_size)
            _ = model(x)
            intermedia_output = activation['transformer_encoder_dy']
            intermedia_output = [intermedia_output[dt_cnt[j], j, :].reshape(1, -1) for j in range(batch_size)]
            intermedia_output = torch.cat(intermedia_output)
            ys.append(intermedia_output)
    handle.remove()
    ys = torch.cat(ys).cpu().numpy()
    ys = pd.DataFrame(ys, columns=['emb' + str(i) for i in range(embedding_size)])
    ys[entity_id] = data[entity_id]
    ys = ys.head(dsize)
    return ys


def get_daily_embedding(model, data):
    model.eval()
    activation = {}

    def get_activation(name):
        def hook(model_hook, input_hook, output_hook):
            activation[name] = output_hook.detach()
        return hook

    handle = model.transformer_encoder_dy.register_forward_hook(get_activation('transformer_encoder_dy'))
    dsize = data.shape[0]
    nbatch = int(dsize / batch_size)
    if dsize - nbatch * batch_size > 0:
        k = batch_size - (dsize - nbatch * batch_size)
        data = pd.concat([data, pd.concat([data.head(1)] * k, ignore_index=True)])
    data = data.reset_index(drop=True)
    nbatch = int(data.shape[0] / batch_size)
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(nbatch)):
            batch = data.iloc[i * batch_size:i * batch_size + batch_size, :]
            dt_cnt, x = prepare_tensor(batch, batch_size)
            _ = model(x)
            day_embeddings = activation['transformer_encoder_dy']
            day_embeddings = torch.swapaxes(day_embeddings, 0, 1)
            for mbr_idx in range(batch_size):
                if i * batch_size + mbr_idx >= dsize:
                    continue
                mbr_id = batch[entity_id].iloc[mbr_idx]
                valid_days = dt_cnt[mbr_idx] + 1
                for day_idx in range(1, valid_days):
                    embedding = day_embeddings[mbr_idx, day_idx, :].cpu().numpy()
                    embedding_dict = {
                        entity_id: mbr_id,
                        'day_idx': day_idx,
                    }
                    for j in range(embedding_size):
                        embedding_dict[f'emb{j}'] = embedding[j]
                    all_embeddings.append(embedding_dict)
            torch.cuda.empty_cache()
            gc.collect()
    handle.remove()
    return pd.DataFrame(all_embeddings)


def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'timestamp': str(currentTime()),
        'model': model.module.state_dict() if parallel else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0)


def create_model():
    model = TransformerModel(nhead, nhid, nlayers, ndropout)
    if parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    return model


def train(df, num_epochs=10, val_split=0.2, checkpoint_path=None):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = ClinicalDataset(df, target_col=target)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
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
    model = create_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = None
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = val_epoch(model, val_loader, criterion)
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, checkpoint_path.replace('.pt', '_best.pt'))
            print(f'New best model saved! Validation loss: {best_val_loss:.4f}')
        scheduler.step()
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
    return model, best_val_loss


if __name__ == '__main__':
    print(f'Device: {device}')
    print(f'Batch size: {batch_size}')
    print(f'Embedding size: {embedding_size}')
    print(f'Number of heads: {nhead}')
    print(f'Hidden size: {nhid}')
    print(f'Number of layers: {nlayers}')
    print(f'Dropout: {ndropout}')



