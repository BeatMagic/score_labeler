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
import json
from read_wav import *
import librosa
import pretty_midi

wav_data_path = '/home/NAS/vocal_data_archive/step2数据准备与标注/米六_海天转标贝/0520/'
interval_data_path = '/home/baochunhui/score_labeler/4_dayu.interval'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f0_min, f0_max = 20, 1500

frame = 256/44100

def json_load(vocabulary):
    with open(vocabulary,'r') as f:
        content = f.read()
        data = json.loads(content)
        return data

cn_vocabulary = json_load('zh_plan (3).json')
jp_vocabulary = json_load('jp_plan (3).json')
add_vocabulary = {
    'none':0,
    'pau':1,
    'br':2,
    'sil':3,
    'sp':4,
    'pad_sil':6,
    'pad_phn':7,
    'pad_sp':8,
    'pad_br':9
}
cn_vocabulary['phon_id'].update(add_vocabulary)
for i in cn_vocabulary['phon_id']:
    cn_vocabulary['phon_id'][i] += 10

special = [i for i in range(10)]
tail = cn_vocabulary['phon_class']['tail']
tail_idx = []
for i in range(len(tail)):
    if tail[i] in cn_vocabulary['phon_id']:
        tail_idx.append(cn_vocabulary['phon_id'][tail[i]])

phoneme = []
textread = TextGrid()
textread.read(interval_data_path)
for text in textread.tiers[0]:
    start = text.minTime
    end = text.maxTime
    phon = text.mark
    if phon in cn_vocabulary['phon_id']:
        phon_id = cn_vocabulary['phon_id'][phon]
    else:
        phon_id = 104
    phoneme.append((phon_id, start, end))

time_point = []
for i in range(len(phoneme)):
    if i == 0:
        time_point.append(phoneme[i][1])
    if phoneme[i][0] in tail_idx:
        time_point.append(phoneme[i][1])
    if i > 0 and phoneme[i][0] in special and phoneme[i-1][0] in tail_idx:
        time_point.append(phoneme[i][1])
    if i == len(phoneme)-1:
        time_point.append(phoneme[i][2])

x, sr = librosa.load(wav_data_path, sr=44100)
mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20, hop_length=256)
mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=80, hop_length=256)
x = x.astype(np.float64)
f0 = get_f0_from_wav(x, sr)
energy = rms(y=x, S=sr, frame_length=256, hop_length=256, center=True, pad_mode='reflect')
energy = energy.squeeze(0)
'''
np.save('f0.npy', f0)
np.save('energy.npy', energy)
np.save('mfcc.npy', mfcc)
np.save('mel.npy', mel)

f0 = np.load('f0.npy')
energy = np.load('energy.npy')
mfcc = np.load('mfcc.npy')
mel = np.load('mel.npy')
'''
f0 = herz2note(f0)
f0=((np.clip(f0,40.0,90.0)-40.0)/50.0*511)

phoneme_frame = []
phoneme_id = 0
for i in range(1, f0.shape[0]+1):
    frame_end_time = i*frame
    phoneme_end_time = phoneme[phoneme_id][2]
    if frame_end_time <= phoneme_end_time:
        phoneme_frame.append(phoneme[phoneme_id][0])
    else:
        phoneme_id += 1
        if phoneme_id >= len(phoneme):
            break
        phoneme_frame.append(phoneme[phoneme_id][0])
phoneme_frame = np.array(phoneme_frame)
phoneme = phoneme_frame

if phoneme.shape[0] < f0.shape[0]:
    phoneme = np.pad(phoneme, (0, f0.shape[0] - np.shape(phoneme)[0]), mode='constant', constant_values=phoneme[-1])
if phoneme.shape[0] > f0.shape[0]:
    phoneme = phoneme[:f0.shape[0]]

PATH = "/home/baochunhui/score_labeler/lightning_logs/add_mfcc_mel/checkpoints/epoch=190-step=27885.ckpt"
Pipeline = Pipeline.load_from_checkpoint(PATH)
model = Pipeline.net
model = model.to(device)
model.eval()

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
def pre(phoneme, f0, energy, mfcc, mel):
    length = phoneme.shape[0]
    phoneme=phoneme.astype(np.int64)
    f0 = f0.astype(np.int64)
    #f0 = herz2note(f0)
    #f0=((np.clip(f0,40.0,90.0)-40.0)/50.0*511).astype(np.int64)
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
        '''
        pdb.set_trace()
        x = np.arange(0,1024,1)
        y1 = predict[0].cpu().detach().numpy()
        #y2 = target[0].cpu().detach().numpy()
        y3 = data[0][0].cpu().detach().numpy() / 511 * 50
        plt.plot(x, y1, 'b')
        #plt.plot(x, y2, 'r')
        plt.plot(x, y3, 'g')
        pdb.set_trace()
        plt.savefig('predict.png')
        '''
        predict = predict.squeeze(0)
        predict = predict.cpu().detach().numpy().tolist()
        predict = predict[:length]
    return predict

predict_pitch = []
with torch.no_grad():
    length = phoneme.shape[0]
    crr_begin = 0
    total_len = 0
    total_begin = 0
    for i in range(1, length):
        if phoneme[i] != phoneme[i-1]:
            crr_len = i-crr_begin
            if total_len + crr_len <= 1024:
                total_len += crr_len
                crr_begin = i
            elif total_len + crr_len > 1024:
                pho = phoneme[total_begin:crr_begin]
                f   = f0[total_begin:crr_begin]
                ene = energy[total_begin:crr_begin]
                mf  = mfcc[:, total_begin:crr_begin]
                me  = mel[:, total_begin:crr_begin]
                '''
                if pho.shape[0] > 1024:
                    pho = pho[:1024]
                    f = f[:1024]
                    ene = ene[:1024]
                    mf = mf[:, :1024]
                    me = me[:, :1024]
                    predict = pre(pho, f, ene, mf, me)
                    total_len = 0
                    total_begin = i
                    crr_begin = i
                else:
                	'''
                predict = pre(pho, f, ene, mf, me)
                total_len = crr_len
                total_begin = crr_begin
                crr_begin = i
                predict_pitch += predict
        if i == length-1:
            pho = phoneme[total_begin:]
            f   = f0[total_begin:]
            ene = energy[total_begin:]
            mf  = mfcc[:, total_begin:]
            me  = mel[:, total_begin:] 
            '''
            if pho.shape[0] > 1024:
                pho = pho[:1024]
                f = f[:1024]
                ene = ene[:1024]
                mf = mf[:, :1024]
                me = me[:, :1024]
            '''
            predict = pre(pho, f, ene, mf, me)
            predict_pitch += predict

def output_midi(predict_pitch):
    #for i in range(len(predict_pitch)):
    #    if predict_pitch[i] != 0: 
    #        break
    #for j in range(len(predict_pitch)-1, -1, -1):
    #    if predict_pitch[j] != 0:
    #        break
    #predict_pitch = predict_pitch[i:j+1]
    predict_pitch = np.array(predict_pitch)
    predict_pitch[predict_pitch!=0] += 40
    predict_pitch = predict_pitch.tolist()
    new_midi = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(1)
    start_time = 0.0
    start = 0
    for i in range(1, len(predict_pitch)):
        if predict_pitch[i] != predict_pitch[i-1]:
            end_time = i*frame
            if end_time - start_time < 0.15:
                continue
            pitch = predict_pitch[i-1]
            if pitch != 0:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time,end=end_time)
                voice.notes.append(note)
            start_time = end_time
            start = i
        elif phoneme[i] != phoneme[i-1]:
            end_time = i*frame
            if end_time - start_time < 0.15:
                continue
            pitch = predict_pitch[i-1]
            if pitch != 0:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time,end=end_time)
                voice.notes.append(note)
            start_time = end_time
            start = i
    new_midi.instruments.append(voice)
    return new_midi

def output_midi_(predict_pitch, time_point):
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
                if notes[i][2] < 0.16:
                    if i == 0:
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

new_midi, predict_notes = output_midi_(predict_pitch, time_point)

x = np.arange(0, 3000, 1)
y1 = np.array(predict_pitch)
y1 += 40
y2 = np.zeros(f0.shape[0])
for note in predict_notes:
    y2[int(note[1]/frame):int((note[1]+note[2])/frame)+1] = note[0]  
y2[y2==0] += 40
#y2 = target[2].cpu().detach().numpy()
y3 = f0 / 511 * 50
y3 += 40
y4 = []
x2 = []
for i in range(3000, 6000):
    if phoneme[i] != phoneme[i-1] and (phoneme[i] in tail_idx or (phoneme[i] in special and phoneme[i-1] in tail_idx)):
        y4.append(y2[i])
        x2.append(i-3000)
y4 = np.array(y4)
x2 = np.array(x2)
y4[y4 == 0.0] += 40
plt.plot(x, y1[3000:6000], 'b')
plt.plot(x, y2[3000:6000], 'r')
plt.plot(x, y3[3000:6000], 'g')
plt.plot(x2, y4, 'o', markersize=3., color='r')
plt.ylim(ymin=40)
plt.savefig('predict4.png')
destination = 'dayu.mid'
new_midi.write(destination)