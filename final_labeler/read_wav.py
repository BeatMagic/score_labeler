import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
import hparams as hp
import pdb
import pyworld as pw
from scipy.io.wavfile import read
import tgt
import librosa
import scipy
import soundfile as sf
from librosa.feature import rms

data_wav = '/home/NAS/vocal_data_archive/note标注/楚瓷_note202107/1guangnianzhiwai.wav'
data_interval = '/home/NAS/vocal_data_archive/note标注/楚瓷_note202107/1guangnianzhiwai.interval'
#y, sr = librosa.load(data_wav)
#y = y.astype(np.float64)

#y,sr = sf.read(data_wav)
x, sr2 = librosa.load(data_wav, sr=44100)
x = x.astype(np.float64)
x = x[0:100000]
#a = read(data_wav)
#wav = np.array(a[1],dtype=float) #len = 7056975

data = np.load('/home/baochunhui/score_labeler/Data/data_score_label/fastsing_dataset-b6-092_9-3.npz')

f0_min, f0_max = 20, 1500

def get_f0_from_wav(y,sr):
    def de_sudden_change(x):
        for i in range(len(x)-3):
            j=i+1
            if x[j-1]!=0 and x[j]==0 and x[j+1]!=0:
                x[j]=(x[j-1]+x[j+1])/2
            if x[j-1]!=0 and x[j]==0 and x[j+1]==0 and x[j+2]!=0:
                x[j]=x[j+1]=(x[j-1]+x[j+2])/2
        return x

    f0_dio,_ = pw.dio(y,sr,f0_floor=f0_min, f0_ceil=f0_max, frame_period=(256/sr*1000))

    f0_dio=de_sudden_change(f0_dio)

    f0_harvest,_ = pw.harvest(y,sr,f0_floor=f0_min, f0_ceil=f0_max, frame_period=(256/sr*1000))

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=256, center=True, pad=True)[0]
    zcr_d_a = scipy.ndimage.gaussian_filter1d(np.pad(np.abs(zcr[1:]-zcr[:-1]),[0,1],mode='reflect'),4)

    f0_harvest=np.where(zcr_d_a<0.002,f0_harvest,np.where(f0_dio>0,f0_harvest,0))

    return f0_harvest

f0 = get_f0_from_wav(x, sr2)

hop_length = 256
frame_length = 256

#energy = np.array([sum(abs(x[i:i+frame_length]**2)) for i in range(0, len(x), hop_length)])
energy = rms(y=x, S=sr2, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect')

def herz2note(x):
    x = np.where(x > 1, x, 1)
    y = 69 + 12 * np.log(x / 440.0) / np.log(2)
    return np.where(x > 1, y, 0)

def note2herz(x):
    x = np.where(x > 1, x, 1)
    y = np.exp(np.log(2) * (x - 69) / 12) * 440.0
    return np.where(x > 1, y, 0)

