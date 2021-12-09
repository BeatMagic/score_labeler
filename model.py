import torch
# import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import pdb

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

sinusoid_encoding = get_sinusoid_encoding_table(1024, 512)
sinusoid_encoding = sinusoid_encoding.unsqueeze(dim=0)

class GRUclassifier(nn.Module):
    def __init__(self):
        super().__init__()
#         self.linears=nn.ModuleList([nn.Linear(32, 256),
#                       nn.Linear(128, 256),
#                       nn.Linear(80, 256),
#                       nn.Linear(20, 256),
#                       nn.Linear(7, 256),
#                       nn.Linear(256, 256),
#                       nn.Linear(64, 256)])
        self.embeddings = nn.ModuleList([nn.Embedding(512, 512),
                                         nn.Embedding(256, 512),
                                         nn.Embedding(105, 512)
                                        ])
        self.gru =  nn.GRU(input_size=512,hidden_size=256,num_layers=4,bidirectional=True,dropout=0.2)

        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        #self.Transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.classifier = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 50),
        )
#         self.linear=nn.Linear(3, 1)

    def forward(self, features):
        gru_in=self.embeddings[0](features[0])+self.embeddings[1](features[1])+self.embeddings[2](features[2])
        gru_in += sinusoid_encoding.cuda()
        gru_out,_=self.gru(gru_in.transpose(0,1))
        result=self.classifier(gru_out)
        return result.transpose(0,1)

class Transformerclassifier(nn.Module):
    def __init__(self):
        super().__init__()
#         self.linears=nn.ModuleList([nn.Linear(32, 256),
#                       nn.Linear(128, 256),
#                       nn.Linear(80, 256),
#                       nn.Linear(20, 256),
#                       nn.Linear(7, 256),
#                       nn.Linear(256, 256),
#                       nn.Linear(64, 256)])
        self.embeddings = nn.ModuleList([nn.Embedding(512, 512),
                                         nn.Embedding(256, 512),
                                         nn.Embedding(105, 512)
                                        ])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout = 0.5, batch_first=True)
        self.Transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 50),
        )
#         self.linear=nn.Linear(3, 1)

    def forward(self, features):
        inputs=self.embeddings[0](features[0])+self.embeddings[1](features[1])+self.embeddings[2](features[2])
        device = inputs.device
        inputs += sinusoid_encoding.to(device)
        outputs = self.Transformer(inputs)
        result=self.classifier(outputs)
        return result

class Transformerclassifier_concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(512, 256),
                                         nn.Embedding(256, 64),
                                         nn.Embedding(105, 64)
                                        ])
        self.linears = nn.ModuleList([nn.Linear(20, 34),
                                         nn.Linear(80, 94)
                                        ])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout = 0.2, batch_first=True)
        self.Transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.classifier = nn.Sequential(
            nn.Linear(512,512),
            nn.LeakyReLU(inplace=True), 
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 50),
        )

    def forward(self, features):
        inputs = torch.cat((self.embeddings[0](features[0]), self.embeddings[1](features[1]), self.embeddings[2](features[2]),self.linears[0](features[3].transpose(1,2)), self.linears[1](features[4].transpose(1,2))), 2)
        if inputs.shape[1] != 1024:
            sinusoid_encoding_ = get_sinusoid_encoding_table(inputs.shape[1], 512)
            sinusoid_encoding_ = sinusoid_encoding_.unsqueeze(dim=0)
            device = inputs.device
            inputs += sinusoid_encoding_.to(device)
        #pdb.set_trace()
        #features[3] = features[3].transpose(1,2)
        #features[4] = features[4].transpose(1,2)
        #self.linears[0](features[3].transpose(1,2))
        #inputs=self.embeddings[0](features[0])+self.embeddings[1](features[1])+self.embeddings[2](features[2])
        else:
            device = inputs.device
            inputs += sinusoid_encoding.to(device)
        outputs = self.Transformer(inputs)
        result=self.classifier(outputs)
        return result 