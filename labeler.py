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
from tools import *
import time

parser = argparse.ArgumentParser(description='labeler.py')
parser.add_argument('--data_path', type=str, default='./testfile', help="data_path")
parser.add_argument('--shortest_note', type=float, default=0.06, help="shortest_note")
parser.add_argument('--interval_tier', type=int, default=0, help="phoneme is in which tier")
parser.add_argument('--model_path', type=str, default='selected_model.ckpt', help="model_path")
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f0_min, f0_max = 20, 1500

frame = 256/44100

cn_vocabulary = json_load('zh_plan (3).json')
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
for i in cn_vocabulary['phon_id']:
    cn_vocabulary['phon_id'][i] += 10
cn_vocabulary['phon_id'].update(add_vocabulary)

special = [i for i in range(10)]
tail = cn_vocabulary['phon_class']['tail']
tail_idx = []
for i in range(len(tail)):
    if tail[i] in cn_vocabulary['phon_id']:
        tail_idx.append(cn_vocabulary['phon_id'][tail[i]])

Pipeline = Pipeline.load_from_checkpoint(opt.model_path)
model = Pipeline.net
model = model.to(device)
model.eval()

seen = set()
dirs = os.listdir(opt.data_path)
for crr_dir in dirs:
    song = os.path.splitext(crr_dir)[0]
    if song in seen:
        continue

    wav_data_path = opt.data_path + '/' + song + '.wav'
    if not os.path.exists(wav_data_path):
        print('wav file for', song, 'is not exist')
        continue

    interval_data_path = opt.data_path + '/' + song + '.interval'
    if not os.path.exists(interval_data_path):
        print('interval file for', song, 'is not exist')
        continue

    seen.add(song)
    print('processing', song)
    destination = opt.data_path + '/' + song + '.mid'
    phoneme = []
    textread = TextGrid()
    textread.read(interval_data_path)
    for text in textread.tiers[opt.interval_tier]:
        start = text.minTime
        end = text.maxTime
        phon = text.mark
        if phon in cn_vocabulary['phon_id']:
            phon_id = cn_vocabulary['phon_id'][phon]
        else:
            phon_id = 104
            print(phon, 'is not in dictionary')
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

    #load_s = time.time()
    x, sr = librosa.load(wav_data_path, sr=44100)
    #load_end = time.time()
    #print('loading time: ', load_end-load_s, 's')
    #mfcc_s = time.time()
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20, hop_length=256)
    #mfcc_end = time.time()
    #print('computing mfcc: ', mfcc_end-mfcc_s, 's')
    #mel_s = time.time()
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=80, hop_length=256)
    #mel_end = time.time()
    #print('computing mel: ', mel_end-mel_s, 's')
    x = x.astype(np.float64)
    #f0_s = time.time()
    f0 = get_f0_from_wav_fast(x, sr)
    #f0 = get_f0_from_wav(x, sr)
    #f0_end = time.time()
    #print('computing f0: ', f0_end-f0_s, 's')
    #energy_s = time.time()
    energy = rms(y=x, S=sr, frame_length=256, hop_length=256, center=True, pad_mode='reflect')
    #energy_end = time.time()
    #print('computing energy: ', energy_end-energy_s, 's')
    energy = energy.squeeze(0)

    #predict_start = time.time()
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
                    predict = pre(pho, f, ene, mf, me, model, device)
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
                predict = pre(pho, f, ene, mf, me, model, device)
                predict_pitch += predict
    #predict_end = time.time()
    #print('predict time: ', predict_end-predict_start, 's')
    #output_s = time.time()

    for i in range(len(predict_pitch)):
        if phoneme[i] in [2, 3, 4]:
            predict_pitch[i] = 0
    new_midi, predict_notes = output_midi_(predict_pitch, time_point, opt.shortest_note)
    new_midi.write(destination)
    #output_end = time.time()
    #print('output time: ', output_end-output_s, 's')

    test_midi = destination
    test_txtgrid = interval_data_path
    #add_midi_interval(test_midi, test_txtgrid)
    py_grid = textgrid.TextGrid.fromFile(test_txtgrid)

    melody = []
    midi_data = pretty_midi.PrettyMIDI(test_midi)
    instrument = midi_data.instruments[0]
    for note in instrument.notes:
        pitch = note.pitch
        start = note.start
        end = note.end
        melody.append([pitch, start, end])

    if melody[0][0] != 0.0:
        melody.insert(0, [0, 0.0, melody[0][1]])
    for m in range(2*len(melody)):
        if m == len(melody)-1:
            break
        if melody[m][2] != melody[m+1][1]:
            if melody[m][2] > melody[m+1][1]:
                melody[m+1][1] = melody[m][2]
            elif melody[m][2] < melody[m+1][1]:
                melody.insert(m+1, [0, melody[m][2], melody[m+1][1]])

    seg_tier = textgrid.IntervalTier('note')
    for note in melody:
        seg_tier.add(note[1], note[2], str(note[0]))

    if py_grid.getFirst('note'):
        for i in range(len(py_grid.tiers)):
            if py_grid.tiers[i].name == 'note':
                py_grid.tiers[i] = seg_tier
    else:
        py_grid.append(seg_tier)
    py_grid.write(test_txtgrid)