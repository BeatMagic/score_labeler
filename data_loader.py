
import os
from textgrid import *
import json
from read_wav import *
import pdb
import librosa
import pretty_midi
import numpy as np
import torch.utils.data as data

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

data_path = '/home/NAS/vocal_data_archive/note标注/云灏_note202107/1-137/'
f0_min, f0_max = 20, 1500

frame = 256/44100

dirs = os.listdir(data_path)

def json_load(vocabulary):
    with open(vocabulary,'r') as f:
        content = f.read()
        data = json.loads(content)
        return data
#language, dict, phon_id
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

def nearest_bd(time, time_bds):
    '''
    time：时间戳
    time_bds：list，存着时间边界
    return:
        离time最近的time_bd，
        如果最近的time_bd离time大于0.01秒，返回time
    '''
    time_ds = []
    for time_bd in time_bds:
        time_d = time_bd - time
        if abs(time_d) < 0.01:
            time_ds.append(time_d)
    if len(time_ds) != 0:
        time_ds.sort(key=lambda x: abs(x))
        return time + time_ds[0]
    else:
        return time

for i in cn_vocabulary['phon_id']:
	cn_vocabulary['phon_id'][i] += 10

cn_vocabulary['phon_id'].update(add_vocabulary)

songs = {}
for i in dirs:
	song = os.path.splitext(i)[0]
	if song not in songs:
		songs[song] = [i]
	else:
		songs[song].append(i)

processed_song = {}
i = 1
for song in songs:
    if song == '.DS_Store':
        continue
    print(i)
    i += 1
    processed_song[song] = {}
    for file in songs[song]:
        if os.path.splitext(file)[1] == '.wav':
            x, sr = librosa.load(data_path+file, sr=44100)
            x = x.astype(np.float64)
            f0 = get_f0_from_wav(x, sr2)
            energy = rms(y=x, S=sr2, frame_length=256, hop_length=256, center=True, pad_mode='reflect')
            processed_song[song]['f0'] = f0
            processed_song[song]['energy'] = energy

        elif os.path.splitext(file)[1] == '.mid':
            processed_song[song]['melody'] = []
            midi_data = pretty_midi.PrettyMIDI(data_path+file)
            instrument = midi_data.instruments[0]
            for note in instrument.notes:
                pitch = note.pitch
                start = note.start
                end = note.end
                processed_song[song]['melody'].append((pitch, start, end))
            #midi_data.instruments

        elif os.path.splitext(file)[1] == '.interval':
            processed_song[song]['phoneme'] = []
            textread = TextGrid()
            textread.read(data_path+file)
            for text in textread.tiers[2]:
                start = text.minTime
                end = text.maxTime
                phon = text.mark
                if phon in cn_vocabulary['phon_id']:
                    phon_id = cn_vocabulary['phon_id'][phon]
                else:
                    phon_id = 104
                processed_song[song]['phoneme'].append((phon_id, start, end))
#pdb.set_trace()
#np.save("yunhao.npy", processed_song)

#dataset = np.load("yunhao.npy")
#dataset = dataset.item()
dataset = processed_song
#对齐时间
for song in dataset:
    time_bds = [dataset[song]['phoneme'][0][1]]
    for phon in dataset[song]['phoneme']:
        time_bds.append(phon[2])
    #melody_time = [data[song]['melody'][0][1]]
    melody = []
    for melo in dataset[song]['melody']:
        melody_time1 = melo[1]
        melody_time2 = melo[2]
        time1 = nearest_bd(melody_time1, time_bds)
        time2 = nearest_bd(melody_time2, time_bds)
        melody.append((melo[0], time1, time2))
    dataset[song]['melody'] = melody
np.save("fengliyuan5.npy", dataset)

dataset = {}
chuci = np.load("chuci.npy")
chuci = chuci.item()
dataset.update(chuci)
yunhao = np.load("yunhao.npy")
yunhao = yunhao.item()
dataset.update(yunhao)
#chuci.keys() & yunhao.keys()
qixuan = np.load("qixuan.npy")
qixuan = qixuan.item()
dataset.update(qixuan)
youwuyueshan1 = np.load("youwuyueshan1.npy")
youwuyueshan1 = youwuyueshan1.item()
youwuyueshan1['youwu_8_gouzhiqishi'] = youwuyueshan1.pop('8_gouzhiqishi')
dataset.update(youwuyueshan1)
youwuyueshan2 = np.load("youwuyueshan2.npy")
youwuyueshan2 = youwuyueshan2.item()
dataset.update(youwuyueshan2)
youwuyueshan3 = np.load("youwuyueshan3.npy")
youwuyueshan3 = youwuyueshan3.item()
dataset.update(youwuyueshan3)
youwuyueshan4 = np.load("youwuyueshan4.npy")
youwuyueshan4 = youwuyueshan4.item()
dataset.update(youwuyueshan4)
xiaoye1 = np.load("xiaoye1.npy")
xiaoye2 = np.load("xiaoye2.npy")
xiaoye3 = np.load("xiaoye3.npy")
xiaoye4 = np.load("xiaoye4.npy")
xiaoye5 = np.load("xiaoye5.npy")
xiaoye6 = np.load("xiaoye6.npy")
xiaoye7 = np.load("xiaoye7.npy")
xiaoye1 = xiaoye1.item()
xiaoye2 = xiaoye2.item()
xiaoye3 = xiaoye3.item()
xiaoye4 = xiaoye4.item()
xiaoye5 = xiaoye5.item()
xiaoye6 = xiaoye6.item()
xiaoye7 = xiaoye7.item()
dataset.update(xiaoye1)
dataset.update(xiaoye2)
dataset.update(xiaoye3)
dataset.update(xiaoye4)
dataset.update(xiaoye5)
dataset.update(xiaoye6)
dataset.update(xiaoye7)

qi1 = np.load("qi1.npy")
qi2 = np.load("qi2.npy")
qi3 = np.load("qi3.npy")
qi4 = np.load("qi4.npy")
qi5 = np.load("qi5.npy")

qi1 = qi1.item()
qi2 = qi2.item()
qi3 = qi3.item()
qi4 = qi4.item()
qi5 = qi5.item()

dataset.update(qi1)
dataset.update(qi2)
dataset.update(qi3)
dataset.update(qi4)
dataset.update(qi5)

fengliyuan1 = np.load("fengliyuan1.npy")
fengliyuan2 = np.load("fengliyuan2.npy")
fengliyuan3 = np.load("fengliyuan3.npy")
fengliyuan4 = np.load("fengliyuan4.npy")
fengliyuan5 = np.load("fengliyuan5.npy")

fengliyuan1 = fengliyuan1.item()
fengliyuan2 = fengliyuan2.item()
fengliyuan3 = fengliyuan3.item()
fengliyuan4 = fengliyuan4.item()
fengliyuan5 = fengliyuan5.item()
#dataset.keys() & fengliyuan5.keys()
dataset.update(fengliyuan1)
dataset.update(fengliyuan2)
dataset.update(fengliyuan3)
dataset.update(fengliyuan4)
dataset.update(fengliyuan5)

pdb.set_trace()
np.save("dataset.npy", dataset)


dataset = np.load("dataset.npy")
dataset = dataset.item()

for song in dataset:
    data = dataset[song]
    f0 = data['f0'] #shape (15371,)
    energy = data['energy'].squeeze(0) # (1, 15371)
    phoneme = data['phoneme'] # 268
    melody = data['melody'] # 219 

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
    dataset[song]['phoneme'] = np.array(phoneme_frame)
    dataset[song]['melody'] = np.array(melody_frame)
pdb.set_trace()
np.save("dataset.npy", dataset)