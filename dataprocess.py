import numpy as np
import re
import csv
import tqdm
import time

import pdb

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

dataset = np.load("dataset.npy")
dataset = dataset.item()

def split(phoneme, melody, f0, energy, mfcc, mel):
    data = []
    length = phoneme.shape[0]
    crr_begin = 0
    if_changes = False
    total_len = 0
    total_begin = 0
    for i in range(length-1):
        if phoneme[i+1] != phoneme[i]:
            crr_len = i+1-crr_begin
            if total_len + crr_len <= 1024:
                total_len += crr_len
                crr_begin = i+1
            elif total_len + crr_len > 1024:
                pho = phoneme[total_begin:crr_begin]
                melo = melody[total_begin:crr_begin]
                f   = f0[total_begin:crr_begin]
                ene = energy[total_begin:crr_begin]
                mf  = mfcc[:, total_begin:crr_begin]
                me  = mel[:, total_begin:crr_begin]
                data.append((pho, melo, f, ene, mf, me)) #phoneme, melody, f0, energy
                total_len = crr_len
                total_begin = crr_begin
                crr_begin = i+1
    return data

split_data = []
for song in dataset.keys():
    phoneme = dataset[song]['phoneme']
    melody = dataset[song]['melody']
    f0 = dataset[song]['f0']
    energy = dataset[song]['energy'].squeeze(0)
    mfcc = dataset[song]['mfcc']
    mel = dataset[song]['mel']
    data = split(phoneme, melody, f0, energy, mfcc, mel)
    split_data += data

index = []
for i in range(len(split_data)):
    if split_data[i][0].shape[0] > 1024 or split_data[i][0].shape[0] == 0:
        index.append(i)
for i in index[::-1]:
    split_data.pop(i)
split_data.append(split_data[0])
print(len(split_data))
length = len(split_data) // 10

dataset_10_fold_1024 = []

for i in range(10):
    dataset_10_fold_1024.append(split_data[(i*length):((i*length)+length)])

pdb.set_trace()
dataset_10_fold_1024 = np.array(dataset_10_fold_1024)
np.save('dataset_10_fold_1024.npy', dataset_10_fold_1024)