#-*- coding: utf-8 -*-
import mido
import pdb

data_midi = '/home/NAS/vocal_data_archive/note标注/楚瓷_note202107/1guangnianzhiwai.mid'
mid = mido.MidiFile(data_midi)

#for i, track in enumerate(mid.tracks):#enumerate()：创建索引序列，索引初始为0
    #print('Track {}: {}'.format(i, track.name))
    #for msg in track:#每个音轨的消息遍历
        #print(msg)

track = mid.tracks[0]
for msg in track:
	pdb.set_trace()
	print(msg)

pdb.set_trace()

x = 1