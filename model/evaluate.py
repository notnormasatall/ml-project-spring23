# Utils
from os.path import join as pjoin
import numpy as np

# MIDI
import pretty_midi
import mir_eval
import librosa

CHECKPOINT_PATH = "../../../extra_space2/solodzhuk_rodzin/models/ml-project/"
# FILE = "../../../extra_space2/solodzhuk_rodzin/midi/MIDI_Unprocessed_02_R2_2008_01_05_ORIG_MID_AUDIO_02_R2_2008_wav.midi"
FILE = "../../../extra_space2/solodzhuk_rodzin/midi/MIDI-Unprocessed_17_R1_2009_01-03_ORIG_MID--AUDIO_17_R1_2009_17_R1_2009_01_WAV.midi"


def write_convention_file(path_to_orig_midi, path_to_trans_midi):
    orig_instrument = pretty_midi.PrettyMIDI(path_to_orig_midi).instruments[0]
    trans_instrument = pretty_midi.PrettyMIDI(path_to_trans_midi).instruments[0]
    ref_time = np.array([])
    ref_pitch = np.array([])
    ref_vel = np.array([])
    est_time = np.array([])
    est_pitch = np.array([])
    est_vel = np.array([])
    for note in orig_instrument.notes:
        # print(note)
        ref_time = np.append(ref_time, np.array([note.start, note.end]))
        ref_pitch = np.append(ref_pitch, librosa.midi_to_hz(note.pitch))
        ref_vel = np.append(ref_vel, note.velocity)
    for note in trans_instrument.notes:
        est_time = np.append(est_time, np.array([note.start, note.end]))
        est_pitch = np.append(est_pitch, librosa.midi_to_hz(note.pitch))
        est_vel = np.append(est_vel, note.velocity)
    ref_time = ref_time.reshape(ref_time.shape[0]//2, 2)
    est_time = est_time.reshape(est_time.shape[0]//2, 2)
    results = mir_eval.transcription_velocity.precision_recall_f1_overlap(ref_time, ref_pitch, ref_vel, est_time, est_pitch, est_vel)
    # results = mir_eval.transcription.precision_recall_f1_overlap(ref_time, ref_pitch, est_time, est_pitch)
    return results[2]

if __name__ == "__main__":
    print(f"""F1-score is: {100*write_convention_file(FILE, pjoin(CHECKPOINT_PATH, "test_song.mid")):.02f}%""")