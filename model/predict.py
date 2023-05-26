from model import SimpleModel
from os.path import join as pjoin
from dataset import get_data
from vocab import detokenizer_vocab
import pretty_midi
import numpy as np
import os

# MODEL_PATH = "../../../extra_space2/solodzhuk_rodzin/models/new_tokenization.ckpt"
MODEL_PATH = "../../../extra_space2/solodzhuk_rodzin/models/improved_model.ckpt"
CHECKPOINT_PATH = "../../../extra_space2/solodzhuk_rodzin/models/ml-project/"
WAV_WORK_DIR = "../../../extra_space2/solodzhuk_rodzin/wav/"
FILES = ["".join(i.split(".")[:-1]) for i in os.listdir(WAV_WORK_DIR)]
FILE = ["MIDI_Unprocessed_02_R2_2008_01_05_ORIG_MID_AUDIO_02_R2_2008_wav"]

def generate_midi_file(predicted_tokens, max_seq_len = 4088):
    tokens = []
    for sequence in predicted_tokens:
        start_flag = False
        for token in sequence:
            if token == 1:
                if not start_flag:
                    start_flag = True
                else:
                    break
            else:
                tokens.append(token.item())

    mid = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    prev_time = 0
    curr_time = 0
    curr_vel = 0

    abs_time = 0

    queue = {}

    for token in tokens:
        if "Velocity" in detokenizer_vocab[token]:
            curr_vel = int(detokenizer_vocab[token].split("_")[1])
        elif "NoteOn" in detokenizer_vocab[token]:
            queue[int(detokenizer_vocab[token].split("_")[1])] = (abs_time, curr_vel)
        elif "NoteOff" in detokenizer_vocab[token]:
            try:
                start, vel = queue[int(detokenizer_vocab[token].split("_")[1])]
                pitch = int(detokenizer_vocab[token].split("_")[1])
                end = abs_time
                piano.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
            except KeyError:
                pass
        else: #timeshift
            curr_time = int(detokenizer_vocab[token].split("_")[1])
            if curr_time >= prev_time:
                abs_time += (curr_time - prev_time)/1000
            else:
                abs_time += (max_seq_len - prev_time + curr_time)/1000
            prev_time = curr_time
    
    mid.instruments.append(piano)
    mid.write(pjoin(CHECKPOINT_PATH, "test_song.mid"))


if __name__ == "__main__":
    output, mask, input, output_mask = get_data(FILE)
    model = SimpleModel.load_from_checkpoint(MODEL_PATH)
    tokens = model.predict(input, mask)
    generate_midi_file(tokens)