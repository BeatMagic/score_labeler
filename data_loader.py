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
            f0 = get_f0_from_wav(x, sr)
            energy = rms(y=x, S=sr, frame_length=256, hop_length=256, center=True, pad_mode='reflect')
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

data_path = '/home/NAS/vocal_data_archive/note标注/楚瓷_note202107/'
dataset = {}
chuci = np.load("chuci.npy")
chuci = chuci.item()
for song in chuci:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    chuci[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    chuci[song]['mel'] = mel
dataset.update(chuci)
print('chuci completed')

data_path = '/home/NAS/vocal_data_archive/note标注/云灏_note202107/1-137/'
yunhao = np.load("yunhao.npy")
yunhao = yunhao.item()
for song in yunhao:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    yunhao[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    yunhao[song]['mel'] = mel
dataset.update(yunhao)
print('yunhao completed')
#chuci.keys() & yunhao.keys()
data_path = '/home/NAS/vocal_data_archive/note标注/绮萱_note202108/1_batch/'
qixuan = np.load("qixuan.npy")
qixuan = qixuan.item()
for song in qixuan:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20 ,hop_length=256)
    qixuan[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    qixuan[song]['mel'] = mel
dataset.update(qixuan)
print('qixuan completed')

data_path = '/home/NAS/vocal_data_archive/note标注/幽舞越山_note202108/0610/'
youwuyueshan1 = np.load("youwuyueshan1.npy")
youwuyueshan1 = youwuyueshan1.item()
for song in youwuyueshan1:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    youwuyueshan1[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    youwuyueshan1[song]['mel'] = mel
youwuyueshan1['youwu_8_gouzhiqishi'] = youwuyueshan1.pop('8_gouzhiqishi')
dataset.update(youwuyueshan1)
print('youwuyueshan1 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/幽舞越山_note202108/0616/'
youwuyueshan2 = np.load("youwuyueshan2.npy")
youwuyueshan2 = youwuyueshan2.item()
for song in youwuyueshan2:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    youwuyueshan2[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    youwuyueshan2[song]['mel'] = mel
dataset.update(youwuyueshan2)
print('youwuyueshan2 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/幽舞越山_note202108/0617/'
youwuyueshan3 = np.load("youwuyueshan3.npy")
youwuyueshan3 = youwuyueshan3.item()
for song in youwuyueshan3:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    youwuyueshan3[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    youwuyueshan3[song]['mel'] = mel
dataset.update(youwuyueshan3)
print('youwuyueshan3 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/幽舞越山_note202108/0618/'
youwuyueshan4 = np.load("youwuyueshan4.npy")
youwuyueshan4 = youwuyueshan4.item()
for song in youwuyueshan4:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    youwuyueshan4[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    youwuyueshan4[song]['mel'] = mel
dataset.update(youwuyueshan4)
print('youwuyueshan4 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/小夜_note202107/小夜_note/1124_clean_wu/'
xiaoye1 = np.load("xiaoye1.npy")
xiaoye1 = xiaoye1.item()
for song in xiaoye1:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    xiaoye1[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    xiaoye1[song]['mel'] = mel
dataset.update(xiaoye1)
print('xiaoye1 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/小夜_note202107/小夜_note/1125_clean_wu/'
xiaoye2 = np.load("xiaoye2.npy")
xiaoye2 = xiaoye2.item()
for song in xiaoye2:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    xiaoye2[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    xiaoye2[song]['mel'] = mel
dataset.update(xiaoye2)
print('xiaoye2 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/小夜_note202107/小夜_note/1128_clean_wu/'
xiaoye3 = np.load("xiaoye3.npy")
xiaoye3 = xiaoye3.item()
for song in xiaoye3:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    xiaoye3[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    xiaoye3[song]['mel'] = mel
dataset.update(xiaoye3)
print('xiaoye3 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/小夜_note202107/小夜_note/1129_clean_wu/'
xiaoye4 = np.load("xiaoye4.npy")
xiaoye4 = xiaoye4.item()
for song in xiaoye4:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    xiaoye4[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    xiaoye4[song]['mel'] = mel
dataset.update(xiaoye4)
print('xiaoye4 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/小夜_note202107/小夜_note/1130_clean_wu/'
xiaoye5 = np.load("xiaoye5.npy")
xiaoye5 = xiaoye5.item()
for song in xiaoye5:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    xiaoye5[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    xiaoye5[song]['mel'] = mel
dataset.update(xiaoye5)
print('xiaoye5 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/小夜_note202107/小夜_note/1201_clean_wu/'
xiaoye6 = np.load("xiaoye6.npy")
xiaoye6 = xiaoye6.item()
for song in xiaoye6:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    xiaoye6[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    xiaoye6[song]['mel'] = mel
dataset.update(xiaoye6)
print('xiaoye6 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/小夜_note202107/小夜_note/1202_clean_wu/'
xiaoye7 = np.load("xiaoye7.npy")
xiaoye7 = xiaoye7.item()
for song in xiaoye7:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    xiaoye7[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    xiaoye7[song]['mel'] = mel
dataset.update(xiaoye7)
print('xiaoye7 completed')


data_path = '/home/NAS/vocal_data_archive/note标注/祈_note202107/祈_rec_ch_1-191(189)_cutted/'
qi1 = np.load("qi1.npy")
qi1 = qi1.item()
for song in qi1:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    qi1[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    qi1[song]['mel'] = mel
dataset.update(qi1)
print('qi1 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/祈_note202107/qi_af_ch_1-54_cut/'
qi2 = np.load("qi2.npy")
qi2 = qi2.item()
for song in qi2:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    qi2[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    qi2[song]['mel'] = mel
dataset.update(qi2)
print('qi2 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/祈_note202107/qi_af_ch_55-161_cut/'
qi3 = np.load("qi3.npy")
qi3 = qi3.item()
for song in qi3:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    qi3[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    qi3[song]['mel'] = mel
dataset.update(qi3)
print('qi3 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/祈_note202107/qi_af_ch_162-257_cut/'
qi4 = np.load("qi4.npy")
qi4 = qi4.item()
for song in qi4:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    qi4[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    qi4[song]['mel'] = mel
dataset.update(qi4)
print('qi4 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/祈_note202107/qi_af_ch_258-278_cut/'
qi5 = np.load("qi5.npy")
qi5 = qi5.item()
for song in qi5:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    qi5[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    qi5[song]['mel'] = mel
dataset.update(qi5)
print('qi5 completed')

data_path = '/home/NAS/vocal_data_archive/note标注/冯丽媛_note202108/0509/'
fengliyuan1 = np.load("fengliyuan1.npy")
fengliyuan1 = fengliyuan1.item()
for song in fengliyuan1:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    fengliyuan1[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    fengliyuan1[song]['mel'] = mel
dataset.update(fengliyuan1)
print('fengliyuan1')

data_path = '/home/NAS/vocal_data_archive/note标注/冯丽媛_note202108/0510/'
fengliyuan2 = np.load("fengliyuan2.npy")
fengliyuan2 = fengliyuan2.item()
for song in fengliyuan2:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    fengliyuan2[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    fengliyuan2[song]['mel'] = mel
dataset.update(fengliyuan2)
print('fengliyuan2')

data_path = '/home/NAS/vocal_data_archive/note标注/冯丽媛_note202108/0513/'
fengliyuan3 = np.load("fengliyuan3.npy")
fengliyuan3 = fengliyuan3.item()
for song in fengliyuan3:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    fengliyuan3[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    fengliyuan3[song]['mel'] = mel
dataset.update(fengliyuan3)
print('fengliyuan3')

data_path = '/home/NAS/vocal_data_archive/note标注/冯丽媛_note202108/0630/'
fengliyuan4 = np.load("fengliyuan4.npy")
fengliyuan4 = fengliyuan4.item()
for song in fengliyuan4:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    fengliyuan4[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    fengliyuan4[song]['mel'] = mel
dataset.update(fengliyuan4)
print('fengliyuan4')

data_path = '/home/NAS/vocal_data_archive/note标注/冯丽媛_note202108/0702/'
fengliyuan5 = np.load("fengliyuan5.npy")
fengliyuan5 = fengliyuan5.item()
for song in fengliyuan5:
    file_name = data_path + song + '.wav'
    y, sr = librosa.load(file_name, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    fengliyuan5[song]['mfcc'] = mfcc
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
    fengliyuan5[song]['mel'] = mel
dataset.update(fengliyuan5)
print('fengliyuan5')


#pdb.set_trace()
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
#pdb.set_trace()
np.save("dataset.npy", dataset)