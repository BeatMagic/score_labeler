from textgrid import *
import pdb

data_interval = '/home/NAS/vocal_data_archive/note标注/楚瓷_note202107/1guangnianzhiwai.interval'

textread = TextGrid()
textread.read(data_interval)

for i in textread.tiers[0]:
    pdb.set_trace()
    print(i)

pdb.set_trace()

x = 1