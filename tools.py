import argparse
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Train_lightning import Pipeline
from collections import Counter
import pdb
import os
from textgrid import *
import textgrid
import json
from read_wav import *
import librosa
import pretty_midi
import time

f0_min, f0_max = 20, 1500

frame = 256/44100

def get_f0_from_wav_fast(y,sr):

    f0_dio,_ = pw.dio(y,sr,f0_floor=f0_min, f0_ceil=f0_max,
                                          frame_period=(256/sr*1000))

    return f0_dio

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

def herz2note(x):
    x = np.where(x > 1, x, 1)
    y = 69 + 12 * np.log(x / 440.0) / np.log(2)
    return np.where(x > 1, y, 0)
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

def json_load(vocabulary):
    with open(vocabulary,'r') as f:
        content = f.read()
        data = json.loads(content)
        return data

def pre(phoneme, f0, energy, mfcc, mel, model, device):
    length = phoneme.shape[0]
    phoneme=phoneme.astype(np.int64)
    f0 = f0.astype(np.int64)
    energy=((np.clip(energy,0,1))*255).astype(np.int64)
    f0=np.expand_dims(f0, 0)
    energy=np.expand_dims(energy, 0)
    phoneme=np.expand_dims(phoneme, 0)
    mfcc=np.expand_dims(mfcc, 0)
    mel=np.expand_dims(mel, 0)
    if length < 1024:
        f0=pad_data(f0)
        energy=pad_data(energy)
        phoneme=pad_data(phoneme)
        mfcc=pad_data_(mfcc)
        mel=pad_data_(mel)
    with torch.no_grad():
        data = [torch.LongTensor(f0), torch.LongTensor(energy), torch.LongTensor(phoneme), torch.Tensor(mfcc), torch.Tensor(mel)]
        data[0] = Variable(data[0].to(device))
        data[1] = Variable(data[1].to(device))
        data[2] = Variable(data[2].to(device))
        data[3] = Variable(data[3].to(device))
        data[4] = Variable(data[4].to(device))
        data = tuple(data)
        output = model(data)
        softmax = nn.Softmax(dim = 2)
        predict = softmax(output)
        predict = predict.argmax(dim=2)
        predict = process_function(predict, data[2])
        predict = predict.squeeze(0)
        predict = predict.cpu().detach().numpy().tolist()
        predict = predict[:length]
    return predict

def output_midi_(predict_pitch, time_point, shortest_note):
    predict_pitch = np.array(predict_pitch)
    predict_pitch[predict_pitch!=0] += 40
    predict_pitch = predict_pitch.tolist()
    new_midi = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(1)
    predict_notes = []
    for t in range(len(time_point)-1):
        start = int(time_point[t]/frame)
        end = int(time_point[t+1]/frame)
        if end >= len(predict_pitch):
            process_pitch = predict_pitch[start:]
        else:
            process_pitch = predict_pitch[start:end]
        # process pitch
        sta = 0 
        notes = []
        for i in range(1, len(process_pitch)):
            if process_pitch[i] != process_pitch[i-1] or i == len(process_pitch)-1:
                begin = start*frame + sta*frame
                time_long = (i - sta)*frame
                notes.append([process_pitch[sta], begin, time_long])
                sta = i
        if len(notes) == 1:
            if notes[0][0] != 0:
                note = pretty_midi.Note(velocity=100, pitch=notes[0][0], start=notes[0][1], end=notes[0][1]+notes[0][2])
                voice.notes.append(note)
            predict_notes += notes
        else:
            for i in range(len(notes)-1, -1, -1):
                if notes[i][2] < shortest_note:
                    if i == 0 and len(notes) > 1:
                        notes[i+1][1] = notes[i][1]
                        notes[i+1][2] += notes[i][2]
                    elif i == len(notes)-1:
                        notes[i-1][2] += notes[i][2]
                    else:
                        notes[i-1][2] += notes[i][2]
                    notes.pop(i)
            for n in notes:
            	if n[0] != 0:
                    note = pretty_midi.Note(velocity=100, pitch=n[0], start=n[1], end=n[1]+n[2])
                    voice.notes.append(note)
            predict_notes += notes
    new_midi.instruments.append(voice)
    return new_midi, predict_notes