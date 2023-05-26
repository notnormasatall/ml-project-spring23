# General
import os
from os.path import join as pjoin
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

from typing import List, Dict
from pathlib import Path

import heapq

# NN
from torch.utils.data import Dataset

# Audiofile utils
import librosa
from miditoolkit import MidiFile
from vocab import tokenizer_vocab
import pretty_midi

WAV_WORK_DIR = "../../../extra_space2/solodzhuk_rodzin/wav/"
MID_WORK_DIR = "../../../extra_space2/solodzhuk_rodzin/midi/"

SAMPLE_RATE = 16000
HOP_WIDTH = 128
SEG_WIDTH = 511
MEL_BINS = 512
N_FFT = 2048

FILES = ["".join(i.split(".")[:-1]) for i in os.listdir(WAV_WORK_DIR)]

def load_data(fname, sr = SAMPLE_RATE):
    '''
    Given the path, returns .wav and the corresponding .midi files.
    '''
    wav_f = librosa.load(pjoin(WAV_WORK_DIR, fname + ".wav"), sr=sr)[0]
    midi_f = MidiFile(pjoin(MID_WORK_DIR, fname + ".midi"), clip=True)

    return wav_f, midi_f

def get_spectogram(audio, sr=SAMPLE_RATE, hl=HOP_WIDTH, mn=MEL_BINS, n_fft=N_FFT):
    '''
    Returns a melspectogram of an audiofile on a given path.
    '''
    return librosa.feature.melspectrogram(y = audio, sr=sr, n_fft=n_fft, 
                                          hop_length=hl, n_mels=mn).T

def pad_spectrogram(spectrogram, seq_size=511, log_mel_bins = 512):
    '''
    Splits spectogram into batches, each having a mask.
    '''
    batches = []
    attention_mask = []

    for i in range(0, spectrogram.shape[0], seq_size):

        if i+seq_size > spectrogram.shape[0]:
            result = np.append(spectrogram[i:i+seq_size], 
                               np.array([1 for _ in range(log_mel_bins)]).reshape(1, log_mel_bins), axis=0)      
            batches.append(np.append(result, 
                                     np.array([0 for _ in range((i+seq_size - spectrogram.shape[0])*(log_mel_bins))]).reshape(i+seq_size - spectrogram.shape[0], log_mel_bins), axis=0))
            attention_mask.append([1 for _ in range(spectrogram.shape[0] - i + 1)] + [0 for _ in range(i+seq_size - spectrogram.shape[0])])
            continue
        batches.append(np.append(spectrogram[i:i+seq_size], np.array([1 for _ in range(log_mel_bins)]).reshape(1, log_mel_bins), axis=0))
        attention_mask.append([1 for _ in range(seq_size+1)])
    
    batches, attention_mask = np.array(batches).reshape(len(batches), seq_size + 1, log_mel_bins), np.array(attention_mask).reshape(len(attention_mask), seq_size + 1)
    batches = batches.astype(np.float32)
    attention_mask = attention_mask.astype(np.float32)
    return batches, attention_mask


class MIDIDataset(Dataset):
    """
    Dataset for generator training

    :param files_paths: list of paths to files to load.
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(self, files_paths: List[Path], max_seq_len: int, batch_size = 512):
        samples = []
        masks = []

        spectrograms = []
        input_masks = []

        print("Loading data:")
        for file_path in tqdm(files_paths):
            midi_path = pjoin(MID_WORK_DIR, file_path + ".midi")

            batches = []
            batches_mask = []

            mid = pretty_midi.PrettyMIDI(midi_path).instruments[0]
            queue = []
            curr_time = 0
            curr_abs_time = 0
            curr_vel = 0
            current_seq = 0

            for note in mid.notes:
                print(note)
                heapq.heappush(queue, (note.start, "NoteOn", note.pitch, note.velocity))
                heapq.heappush(queue, (note.end, "NoteOff", note.pitch, note.velocity))

            sequence = [] 
            too_long_batches_idx = []
            
            while queue:
                event_time, event_type, event_pitch, event_velocity = heapq.heappop(queue)
                if event_time > curr_time: #AbsoluteTime
                    if event_time > max_seq_len*(current_seq+1): #time to create new sequence
                        if len(sequence + [0]) > batch_size:
                            too_long_batches_idx.append(current_seq)
                            sequence = [] 
                            current_seq += 1
                        else:
                            padding = [0]*(batch_size - len(sequence + [0]))
                            batches.append(np.array(sequence + [1] + padding))
                            batches_mask.append(np.array([1] * len(sequence + [0]) + [0]*(batch_size - len(sequence + [1]))))
                            sequence = [] 
                            current_seq += 1
                    while event_time > max_seq_len*(current_seq+1):
                        too_long_batches_idx.append(current_seq)
                        current_seq += 1
                    if int(1000*(event_time - max_seq_len*current_seq)) - int(1000*(event_time - max_seq_len*current_seq))%10 != curr_abs_time:
                        sequence.append(tokenizer_vocab[f"AbsoluteTime_{int(1000*(event_time - max_seq_len*current_seq)) - int(1000*(event_time - max_seq_len*current_seq))%10}"]) #AbsoluteTime
                        curr_abs_time = int(1000*(event_time - max_seq_len*current_seq)) - int(1000*(event_time - max_seq_len*current_seq))%10
                    curr_time = event_time
                if event_velocity != curr_vel and event_type == "NoteOn": #when to consider Velocity
                    sequence.append(tokenizer_vocab[f"Velocity_{event_velocity}"])
                sequence.append(tokenizer_vocab[f"{event_type}_{event_pitch}"]) #NoteOn and NoteOff
            if len(sequence + [0]) > batch_size:
                too_long_batches_idx.append(current_seq)
            else:
                padding = [0]*(batch_size - len(sequence + [0]))
                batches.append(np.array(sequence + [1] + padding))
                batches_mask.append(np.array([1] * len(sequence + [0]) + [0]*(batch_size - len(sequence + [1]))))

            spect = get_spectogram(load_data(file_path)[0])
            input_batches, input_mask = pad_spectrogram(spect)

            mask = np.ones(input_batches.shape[0], dtype=bool)
            mask[too_long_batches_idx] = False

            input_batches = input_batches[mask]
            input_mask = input_mask[mask]

            min_val = min(len(batches), input_batches.shape[0])
            spectrograms.append(np.array(input_batches[:min_val]))
            input_masks.append(np.array(input_mask[:min_val]))
            masks.append(np.array(batches_mask[:min_val]))
            samples.append(np.array(batches[:min_val]))

        self.masks = np.array(masks) 
        self.samples = np.array(samples)
        self.spectrograms = np.array(spectrograms)
        self.input_masks = np.array(input_masks)

    def __getitem__(self, idx) -> Dict[str, np.array]:
        return {"input_spectrograms": self.spectrograms[idx], "input_mask": self.input_masks[idx], "output_mask": self.masks[idx], "output": self.samples[idx]}
    
    def __len__(self) -> int: return len(self.samples)

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(self) == 0 else f'{len(self.samples)} samples'

class MyDataset(Dataset):
    def __init__(self, decoder_input_ids, attention_mask, inputs_embeds, decoder_attention_mask):
        self.decoder_input_ids = decoder_input_ids
        self.attention_mask = attention_mask
        self.inputs_embeds = inputs_embeds
        self.decoder_attention_mask = decoder_attention_mask

    def __len__(self):
        return len(self.decoder_input_ids)

    def __getitem__(self, index):
        return {
            "decoder_input_ids": self.decoder_input_ids[index],
            "attention_mask": self.attention_mask[index],
            "inputs_embeds": self.inputs_embeds[index],
            "decoder_attention_mask": self.decoder_attention_mask[index]
        }

def get_data(paths):
    dataset = MIDIDataset(paths, 4.088)
    for i in range(len(paths)):
        if not i:
            input = dataset[i]["input_spectrograms"].reshape(dataset[i]["input_spectrograms"].shape[0], 512, 512)
            output = dataset[i]["output"]
            output_mask = dataset[i]["output_mask"]
            input_mask = dataset[i]["input_mask"].reshape(dataset[i]["input_mask"].shape[0], 512)
            continue
        input = np.append(input, dataset[i]["input_spectrograms"], axis = 0)
        output = np.append(output, dataset[i]["output"], axis = 0)
        output_mask = np.append(output_mask, dataset[i]["output_mask"], axis = 0)
        input_mask = np.append(input_mask, dataset[i]["input_mask"], axis = 0)

    return output, input_mask, input, output_mask


if __name__=="__main__":
    output, mask, input, output_mask = get_data(FILES[:10])
    print("\noutput:",
          output.shape, 
          "\nmask:",
          mask.shape, 
          "\ninput:",
          input.shape, 
          "\noutput mask:",
          output_mask.shape)