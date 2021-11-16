import torch
# import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from model import GRUclassifier, Transformerclassifier
import numpy as np
import argparse
from torch.autograd import Variable
import tqdm
import pdb

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

frame = 256/44100

def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1

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
#         print(batch_loss.shape,weights.shape)
        batch_loss *= weights.unsqueeze(1)
#         for i in range(targets.shape[0]):
#             targets_=targets.float()[i]
#             omega=1.0+targets_[torch.nonzero(targets_)].std()
#             if omega.isnan():
#                 omega=1.0
# #             print(i,omega)
#             batch_loss[i]*=omega
        
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
    max_len = opt.seqlen
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

class ScoreDataset(data.Dataset):
    def __init__(self, dataset, frame = 256/44100):
        self.dataset = dataset
        self.songs_phoneme, self.songs_melody, self.songs_f0, self.songs_energy = [], [], [], []
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                self.songs_phoneme.append(dataset[i][j][0])
                self.songs_melody.append(dataset[i][j][1])
                self.songs_f0.append(dataset[i][j][2])
                self.songs_energy.append(dataset[i][j][3])
        #self.songs = list(dataset.keys())
        #self.number_of_songs = len(dataset)

    def __getitem__(self, idx: int):
        '''
        song = self.songs[idx]
        data = self.dataset[song]
        f0 = data['f0'] #shape (15371,)
        energy = data['energy'].squeeze(0) # (1, 15371)
        phoneme = data['phoneme'] # 268
        melody = data['melody'] # 219 
        #total_second = min(melody[-1][-1], phoneme[-1][-1])
        phoneme_frame = []
        melody_frame = []
        phoneme_id, melody_id = 0, 0
        for i in range(1, f0.shape[0]+1):
            frame_end_time = i*frame
            phoneme_end_time = phoneme[phoneme_id][2]
            melody_end_time = melody[melody_id][2]
            if frame_end_time <= phoneme_end_time:
                phoneme_frame.append(phoneme[phoneme_id][0])
            else:
                phoneme_id += 1
                if phoneme_id >= len(phoneme):
                    break
                phoneme_frame.append(phoneme[phoneme_id][0])

            if frame_end_time <= melody_end_time:
                melody_frame.append(melody[melody_id][0])
            else:
                melody_id += 1
                if melody_id >= len(melody):
                    break
                melody_frame.append(melody[melody_id][0])   

        target = np.array(melody_frame)
        '''

        f0 =  self.songs_f0[idx]
        f0 = herz2note(f0)
        energy = self.songs_energy[idx]
        phoneme = self.songs_phoneme[idx]
        melody = self.songs_melody[idx]-40
        target = melody.astype(np.int64)
        #coding = np.zeros(phoneme.shape[0])
        #start = 0
        #code = 1
        #for i in range(1, phoneme.shape[0]):
        #    if phoneme[i] != phoneme[i-1]:
        #        coding[start:i] = code
        #        code += 1
        #        start = i
        #coding[start:] = code
        #coding = np.array(coding)
        coding=np.concatenate([np.linspace(0,31,f0.shape[0]//2), np.linspace(31,0,f0.shape[0]-f0.shape[0]//2)],0)
        target=np.clip(target,0,49)
        f0=((np.clip(f0,40.0,90.0)-40.0)/50.0*511).astype(np.int64)
        energy=((np.clip(energy,0,300.0)/300.0)*255).astype(np.int64) #fanwei
        coding=coding.astype(np.int64)
        weight=np.array([1.0])
        return (f0,energy,coding,target,weight)

    def __len__(self):
        return len(self.songs_phoneme)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='score_labeler.py')
    parser.add_argument('--batch_size', type=str, default=8, help="batch size")
    parser.add_argument('--seqlen', type=str, default=1024, help="seqlen")
    parser.add_argument('--learning_rate', type=str, default=0.0001, help="learning rate")
    parser.add_argument('--num_epochs', type=str, default=10, help="num pochs")

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = np.load("dataset_10_fold_1024.npy") #(10, 780, 4), 10 fold, every fold 780 data, every data 4 part: (phoneme, melody, f0, energy) 

    #train_set = ScoreDataset(dataset)
    #data = train_set[0]
    #train_loader = DataLoader(train_set, batch_size=opt.batch_size, collate_fn = collate_fn, shuffle=True, drop_last=True, num_workers=4)

    #model = GRUclassifier()
    
    model = Transformerclassifier()
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    #model = nn.DataParallel(model)
    model = model.to(device)
    #pdb.set_trace()


    for fold in range(10):
        train_dataset = np.concatenate((dataset[0:fold], dataset[fold+1:]), axis=0)
        test_dataset = dataset[fold:fold+1]

        train_dataset = ScoreDataset(train_dataset)
        data = train_dataset[3]
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn = collate_fn, shuffle=True, drop_last=True, num_workers=4)

        test_dataset = ScoreDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, collate_fn = collate_fn, shuffle=False, drop_last=False, num_workers=4)

        optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate)
        criterion = FocalLoss(50)

        train_loss_ = []
        test_loss_ = []
        train_acc_ = []
        test_acc_ = []

        for epoch in range(opt.num_epochs):
            model = model.to(device)
            model.train()
            total_loss = 0.0
            total = 0.0
            total_acc = 0.0
            total_pre = 0.0
            for iter, traindata in enumerate(train_loader):
                data, target, weights = traindata
                data = list(data)
                data[0] = Variable(data[0].to(device))
                data[1] = Variable(data[1].to(device))
                data[2] = Variable(data[2].to(device))
                data = tuple(data)
                target = Variable(target.to(device))
                weights = Variable(weights.to(device))

                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output,target,weights)

                loss.backward()
                optimizer.step()

                total += output.shape[0]
                total_loss += loss
                softmax = nn.Softmax(dim = 2)
                predict = softmax(output)
                predict = predict.argmax(dim=2)
                total_acc += (predict == target).sum()
                total_pre += output.shape[0]*output.shape[1]


                if iter % 50 == 0:
                    print('[Epoch: %3d/%3d] iter: %d loss: %.3f train_acc: %.3f' % 
                        (epoch, opt.num_epochs, iter, loss, ((predict == target).sum()/(output.shape[0]*output.shape[1])) ))

            train_loss_.append(total_loss / total)
            train_acc_.append(total_acc / total_pre)
            '''
            model.eval()
            total_loss = 0.0
            total = 0.0
            total_acc = 0.0
            total_pre = 0.0
            model = model.cpu()
            for iter, testdata in enumerate(test_loader):
                data, target, weights = testdata
                data = list(data)
                data[0] = Variable(data[0])
                data[1] = Variable(data[1])
                data[2] = Variable(data[2])
                data = tuple(data)
                target = Variable(target)
                weights = Variable(weights)

                output = model(data)
                loss = criterion(output,target,weights)

                total += output.shape[0]
                total_loss += loss

                softmax = nn.Softmax(dim = 2)
                predict = softmax(output)
                predict = predict.argmax(dim=2)
                total_acc += (predict == target).sum()
                total_pre += output.shape[0]*output.shape[1]

            test_loss_.append(total_loss / total)
            test_acc_.append(total_acc / total_pre)

            print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, opt.num_epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))
            '''
            filename = 'score_labeler'+'_fold_'+str(fold)+'_epoch_'+str(epoch)+'_clf.pkl'
            torch.save(model.state_dict(), filename)
            print('File %s is saved.' % filename)