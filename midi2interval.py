import pretty_midi
import textgrid


def add_midi_interval(midi_path, txtgrid_path):
    try:
        py_grid = textgrid.TextGrid.fromFile(txtgrid_path)
    except FileNotFoundError:
        return None
    phn_tier = py_grid.getFirst('phoneme')
    maxTime = max([interval.maxTime for interval in phn_tier])

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = midi_data.instruments[0].notes

    midi_start = notes[0].start
    first_normal_phn = phn_tier.intervals[0].mark
    temp_index = 0
    while first_normal_phn in ['sil', 'sp', 'br', 'cl']:
        temp_index += 1
        first_normal_phn = phn_tier.intervals[temp_index].mark
    phn_start = phn_tier.intervals[temp_index].minTime

    midi_bias = (phn_start - midi_start)

    seg_tier = textgrid.IntervalTier('note')

    last_end = 0
    last_phn = '0'
    for i, note in enumerate(notes):
        start = note.start + midi_bias
        end = note.end + midi_bias

        if start - last_end > 0:
            seg_tier.add(last_end, start, last_phn)

        last_end = start
        last_phn = str(note.pitch)

        if i == len(notes) - 1:
            seg_tier.add(last_end, end, last_phn)
            seg_tier.add(end, maxTime, '0')

    if py_grid.getFirst('note'):
        for i in range(len(py_grid.tiers)):
            if py_grid.tiers[i].name == 'note':
                py_grid.tiers[i] = seg_tier
    else:
        py_grid.append(seg_tier)
    py_grid.write(txtgrid_path)


if __name__ == "__main__":
    test_midi = '/home/baochunhui/score_labeler/shijianzhuyu.mid'
    test_txtgrid = '/home/baochunhui/score_labeler/39_shijianzhuyu.interval'
    add_midi_interval(test_midi, test_txtgrid)