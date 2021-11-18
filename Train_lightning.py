import numpy as np
import os
import time
import pickle as pkl
import random

import torch
# import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from model import GRUclassifier, Transformerclassifier, Transformerclassifier_concat
from prefetch_generator import BackgroundGenerator
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
import pdb
from scipy import interpolate
import random

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        inputs=inputs.reshape(-1,50)
        N = inputs.size(0)
        
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        
        probs = (P*class_mask).sum(1).view(-1,1)
        a=1-probs
        b=probs
        batch_loss = -1.0*(torch.pow(a, self.gamma))*torch.log(b)
        
        batch_loss = batch_loss.view(targets.shape[0],-1)
        batch_loss *= weights.unsqueeze(1)
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        
        return loss

def pad_data(inputs):

    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

#         s = np.shape(x)[1]
        x = np.pad(x, (0, max_len - np.shape(x)
                       [0]), mode='constant', constant_values=0)

        return x

    #max_len = max(np.shape(x)[0] for x in inputs)
    max_len = 1024
    pad_output = np.stack([pad(x, max_len) for x in inputs])

    return pad_output

def collate_fn(batch):
    f0s = [b[0] for b in batch]
    energys = [b[1] for b in batch]
    codings = [b[2] for b in batch]
    targets = [b[3] for b in batch]
    weights = [b[4] for b in batch]

    f0_padded=pad_data(f0s)
    energy_padded=pad_data(energys)
    coding_padded=pad_data(codings)
    target_padded=pad_data(targets)
    weights=np.concatenate(weights,0)
#     print(weights.shape)
#     print(weights)
#     print(f0_padded.shape,energy_padded.shape,coding_padded.shape,target_padded.shape)

    return (torch.LongTensor(f0_padded),
           torch.LongTensor(energy_padded),
           torch.LongTensor(coding_padded)),(torch.LongTensor(target_padded)),torch.FloatTensor(weights)
def herz2note(x):
    x = np.where(x > 1, x, 1)
    y = 69 + 12 * np.log(x / 440.0) / np.log(2)
    return np.where(x > 1, y, 0)

def note2herz(x):
    x = np.where(x > 1, x, 1)
    y = np.exp(np.log(2) * (x - 69) / 12) * 440.0
    return np.where(x > 1, y, 0)

#def data_pading(f0, energy, target):
#    if (target!=0).sum() == 0:
#        return f0, energy, target
#    add_num = np.random.randint(-target[target!=0].min()+1, 49-target.max())
#    target = target[target!=0] + add_num
#    f0 = f0[f0!=0] + add_num
#    energy_num = np.random.normal(1, 1)
#    energy_num = np.clip(energy_num , 0.5 , 3)
#    energy *= energy_num
#    return f0, energy, target

class ScoreDataset(data.Dataset):
    def __init__(self, dataset, is_train = True):
        self.dataset = dataset
        self.is_train = is_train
        self.songs_phoneme, self.songs_melody, self.songs_f0, self.songs_energy = [], [], [], []
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                self.songs_phoneme.append(dataset[i][j][0])
                self.songs_melody.append(dataset[i][j][1])
                self.songs_f0.append(dataset[i][j][2])
                self.songs_energy.append(dataset[i][j][3])

    def __getitem__(self, idx: int):
        f0 =  self.songs_f0[idx]
        f0 = herz2note(f0)
        energy = self.songs_energy[idx]
        phoneme = self.songs_phoneme[idx]
        melody = self.songs_melody[idx]-40
        target = melody.astype(np.int64)
        coding = phoneme
        #coding=np.concatenate([np.linspace(0,31,f0.shape[0]//2), np.linspace(31,0,f0.shape[0]-f0.shape[0]//2)],0)

        target=np.clip(target,0,49)

        if (target!=0).sum() != 0 and self.is_train:
            #add_num = np.random.randint(-target[target!=0].min()+1, 49-target.max())
            add_num = random.randint(max(-49, -target[target!=0].min()+1), min(49, 49-target.max()-1))
            target[target!=0] += add_num
            f0[f0>40] += add_num
            energy_num = np.random.normal(1, 0.2)
            #energy_num = np.clip(energy_num , 0.5 , 3)
            energy_num = max(min(1.6, energy_num), 0.4)
            energy *= energy_num 

            x = np.arange(0, f0.shape[0], 1)
            f_energy = interpolate.interp1d(x, energy, kind='linear')
            f_f0 = interpolate.interp1d(x, f0, kind='linear')
            f_target = interpolate.interp1d(x, target, kind='nearest')
            f_coding = interpolate.interp1d(x, coding, kind='nearest')

            d_len = f0.shape[0] + int(max(min(random.gauss(0,200),600),-600))
            d_len = min(1024, d_len)
            xnew = np.arange(0, f0.shape[0], f0.shape[0]/(d_len))
            xnew = np.clip(xnew, 0, f0.shape[0]-1)

            energy = f_energy(xnew)
            f0 = f_f0(xnew)
            target = f_target(xnew)
            coding = f_coding(xnew)
            #plt.plot(x, energy, 'o', xnew, energy_new, '-')
            #plt.savefig('plot.png')

        f0=((np.clip(f0,40.0,90.0)-40.0)/50.0*511).astype(np.int64)

        #x = np.arange(0,f0.shape[0],1)
        #plt.plot(x, target, 'r')
        #plt.plot(x, f0, 'g')
        #plt.savefig('./plot/'+str(idx)+'predict.png')

        energy=((np.clip(energy,0,1))*255).astype(np.int64) #fanwei
        coding=coding.astype(np.int64)
        target = target.astype(np.int64)
        weight=np.array([1.0])

        return (f0,energy,coding,target,weight)

    def __len__(self):
        return len(self.songs_phoneme)

class Pipeline(LightningModule):
    def __init__(self,
        learning_rate: float = 0.0001,
        batch_size: int = 8,
        num_workers: int = 8,
        val_rate=0.05,
    ):
        super().__init__()
        
        self.val_rate=val_rate #0.05
        self.save_hyperparameters()
        #self.net = Transformerclassifier()
        self.net = Transformerclassifier_concat()
        #self.net = GRUclassifier()
        self.criterion = FocalLoss(50)

    def forward(self, x):
        return self.net(x)
        
    def training_step(self, batch, batch_idx):
        data, target, weights = batch
        output = self(data)
        loss = self.criterion(output,target,weights)

        softmax = nn.Softmax(dim = 2)
        predict = softmax(output)
        predict = predict.argmax(dim=2)
        total_acc = (predict == target).sum()
        total_pre = output.shape[0]*output.shape[1]
        train_acc = total_acc / total_pre

        #x = np.arange(0,1024,1)
        #y1 = predict[0].cpu().detach().numpy()
        #y2 = target[0].cpu().detach().numpy()
        #y3 = data[0][0].cpu().detach().numpy() / 511 * 50
        #plt.plot(x, y1, 'b')
        #plt.plot(x, y2, 'r')
        #plt.plot(x, y3, 'g')
        #plt.savefig('./plot/'+str(batch_idx)+'predict.png')
        #print('train/acc', train_acc)
        self.log('train/acc', train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target, weights = batch
        output = self(data)
        val_loss = self.criterion(output,target,weights)

        softmax = nn.Softmax(dim = 2)
        predict = softmax(output)
        predict = predict.argmax(dim=2)
        total_acc = (predict == target).sum()
        total_pre = output.shape[0]*output.shape[1]
        test_acc = total_acc / total_pre
        #print('val/acc', test_acc)
        self.log('val/acc', test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        lr_scheduler = {"scheduler": scheduler }
        return [optimizer], [lr_scheduler]

    def setup(self, stage):
        dataset = np.load("dataset_10_fold_1024.npy")
        fold = 9
        train_dataset = np.concatenate((dataset[0:fold], dataset[fold+1:]), axis=0)
        test_dataset = dataset[fold:fold+1]
        self.train_dataset = ScoreDataset(train_dataset, is_train = True)
        #data = self.train_dataset[10]
        #for idx in range(len(self.train_dataset)):
        #    data = self.train_dataset[idx]
        self.validation_dataset = ScoreDataset(test_dataset, is_train = False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size = self.hparams.batch_size,
            collate_fn = collate_fn, 
            shuffle=True, 
            drop_last=True, 
            num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, 
            batch_size=self.hparams.batch_size, 
            collate_fn = collate_fn, 
            shuffle=False, 
            drop_last=False, 
            num_workers=self.hparams.num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='score_labeler.py')
    parser.add_argument('--batch_size', type=str, default=8, help="batch size")
    #parser.add_argument('--seqlen', type=str, default=1024, help="seqlen")
    parser.add_argument('--learning_rate', type=str, default=0.0001, help="learning rate")
    parser.add_argument('--num_epochs', type=str, default=10, help="num pochs")
    opt = parser.parse_args()

    model=Pipeline(learning_rate = opt.learning_rate, batch_size = opt.batch_size, )
    checkpoint_callback = ModelCheckpoint(monitor='val/loss',save_top_k = -1)
    trainer = Trainer(gpus=6, max_epochs=80,
                  distributed_backend="ddp",
                  log_every_n_steps=1,
#                   resume_from_checkpoint="lightning_logs/version_32/checkpoints/epoch=0.ckpt",
                  callbacks=[checkpoint_callback],
                  progress_bar_refresh_rate=1)
    trainer.fit(model)