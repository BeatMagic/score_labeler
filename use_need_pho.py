import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from Train_lightning import Pipeline
from collections import Counter
import pdb

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

def pad_data_(inputs):
    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        x = np.pad(x, ((0, 0),(0, max_len - np.shape(x)[1])), mode='constant', constant_values=0.0)

        return x

    max_len = 1024
    pad_output = np.stack([pad(x, max_len) for x in inputs])

    return pad_output

def pad_data(inputs):

    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        x = np.pad(x, (0, max_len - np.shape(x)[0]), mode='constant', constant_values=0)

        return x

    max_len = 1024
    pad_output = np.stack([pad(x, max_len) for x in inputs])

    return pad_output

def collate_fn(batch):
    f0s = [b[0] for b in batch]
    energys = [b[1] for b in batch]
    codings = [b[2] for b in batch]
    mfcc    = [b[3] for b in batch]
    mel     = [b[4] for b in batch]
    targets = [b[5] for b in batch]
    weights = [b[6] for b in batch]

    f0_padded=pad_data(f0s)
    energy_padded=pad_data(energys)
    coding_padded=pad_data(codings)
    target_padded=pad_data(targets)
    weights=np.concatenate(weights,0)
    mfcc_padded=pad_data_(mfcc)
    mel_padded=pad_data_(mel)

    return (torch.LongTensor(f0_padded),
           torch.LongTensor(energy_padded),
           torch.LongTensor(coding_padded),
           torch.Tensor(mfcc_padded),
           torch.Tensor(mel_padded)),(torch.LongTensor(target_padded)),torch.FloatTensor(weights)

def herz2note(x):
    x = np.where(x > 1, x, 1)
    y = 69 + 12 * np.log(x / 440.0) / np.log(2)
    return np.where(x > 1, y, 0)

def note2herz(x):
    x = np.where(x > 1, x, 1)
    y = np.exp(np.log(2) * (x - 69) / 12) * 440.0
    return np.where(x > 1, y, 0)

class ScoreDataset(data.Dataset):
    def __init__(self, dataset, is_train = True):
        self.dataset = dataset
        self.is_train = is_train
        self.songs_phoneme, self.songs_melody, self.songs_f0, self.songs_energy = [], [], [], []
        self.songs_mfcc, self.songs_mel = [], []
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                self.songs_phoneme.append(dataset[i][j][0])
                self.songs_melody.append(dataset[i][j][1])
                self.songs_f0.append(dataset[i][j][2])
                self.songs_energy.append(dataset[i][j][3])
                self.songs_mfcc.append(dataset[i][j][4])
                self.songs_mel.append(dataset[i][j][5])

    def __getitem__(self, idx: int):
        f0 =  self.songs_f0[idx]
        f0 = herz2note(f0)
        energy = self.songs_energy[idx]
        phoneme = self.songs_phoneme[idx]
        melody = self.songs_melody[idx]-40
        target = melody.astype(np.int64)
        coding = phoneme
        mfcc = self.songs_mfcc[idx]
        mel = self.songs_mel[idx]
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
            d_len = np.clip(d_len, 650, 1024)
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

        return (f0,energy,coding,mfcc,mel,target,weight)

    def __len__(self):
        return len(self.songs_phoneme)

def process_function(predict, coding):
    batch_size, seq_len = predict.shape[0], predict.shape[1]
    for i in range(batch_size):
        predict_stp = predict[i]
        coding_stp = coding[i]
        #target_stp = target[i]
        start = 0
        for j in range(1, seq_len+1):
            if j == seq_len or coding_stp[j] != coding_stp[j-1]:
                end = j
                process_predict = predict_stp[start:end]
                #process_predict[0] = 1
                #process_predict[46:50] = 2
                length = end-start
                if length == 0:
                    continue
                value_start = 0
                for k in range(1, length):
                    if process_predict[k] != process_predict[k-1]:
                        if k - value_start >= 16:
                            value_start = k
                        elif k - value_start < 16:
                            process_predict[value_start:k] = process_predict[k]
                predict[i][start:end] = process_predict[:]
                start = end
    return predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score_labeler.py')
    parser.add_argument('--batch_size', type=str, default=8, help="batch size")
    parser.add_argument('--seqlen', type=str, default=1024, help="seqlen")
    parser.add_argument('--learning_rate', type=str, default=0.0001, help="learning rate")
    parser.add_argument('--num_epochs', type=str, default=10, help="num pochs")

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save np.load
    #np_load_old = np.load

    # modify the default parameters of np.load
    #np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    dataset = np.load("dataset_10_fold_1024.npy")
    test_dataset = dataset[9:10]
    test_dataset = ScoreDataset(test_dataset, is_train = False)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, collate_fn = collate_fn, shuffle=False, drop_last=False, num_workers=4)
    criterion = FocalLoss(50)
    PATH = "/home/baochunhui/score_labeler/lightning_logs/add_mfcc_mel/checkpoints/epoch=190-step=27885.ckpt"
    Pipeline = Pipeline.load_from_checkpoint(PATH)
    model = Pipeline.net
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total = 0.0
    total_acc = 0.0
    total_pre = 0.0
    for iter, testdata in enumerate(test_loader):
        with torch.no_grad():
            data, target, weights = testdata
            data = list(data)
            data[0] = Variable(data[0].to(device))
            data[1] = Variable(data[1].to(device))
            data[2] = Variable(data[2].to(device))
            data[3] = Variable(data[3].to(device))
            data[4] = Variable(data[4].to(device))
            data = tuple(data)
            target = Variable(target.to(device))
            weights = Variable(weights.to(device))

            output = model(data)
            loss = criterion(output,target,weights)

            total += output.shape[0]
            total_loss += loss

            softmax = nn.Softmax(dim = 2)
            predict = softmax(output)
            predict = predict.argmax(dim=2)

            predict = process_function(predict, data[2])
            '''
            coding = data[2][2].cpu().detach().numpy()
            x = np.arange(0,1024,1)
            y1 = predict[2].cpu().detach().numpy()
            y2 = target[2].cpu().detach().numpy()
            y3 = data[0][2].cpu().detach().numpy() / 511 * 50
            plt.plot(x, y1, 'b')
            plt.plot(x, y2, 'r')
            plt.plot(x, y3, 'g')
            pdb.set_trace()
            plt.savefig('predict.png')
            '''
            total_acc += (predict == target).sum()
            total_pre += output.shape[0]*output.shape[1]

    test_loss_= total_loss / total
    test_acc_ = (total_acc / total_pre).item()

    print('Testing Loss: %.5f, Testing Acc: %.5f' % (test_loss_, test_acc_))