# General
import librosa
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np
from vocab import detokenizer_vocab
import warnings
import pretty_midi
warnings.filterwarnings('ignore')

SAMPLE_RATE = 16000
HOP_WIDTH = 128
SEG_WIDTH = 511
MEL_BINS = 512
N_FFT = 2048


def load_data(fname, sr=SAMPLE_RATE):
    '''
    Given the path, returns .wav and the corresponding .midi files.
    '''
    wav_f = librosa.load(fname, sr=sr)[0]
    return wav_f


def get_spectogram(audio, sr=SAMPLE_RATE, hl=HOP_WIDTH, mn=MEL_BINS, n_fft=N_FFT):
    '''
    Returns a melspectogram of an audiofile on a given path.
    '''
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
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

    def __init__(self, files_paths: List[Path], max_seq_len: int, batch_size=512):
        samples = []
        masks = []

        spectrograms = []
        input_masks = []

        print("Loading data:")
        for file_path in tqdm(files_paths):

            batches = []
            batches_mask = []
            spect = get_spectogram(load_data(file_path))
            input_batches, input_mask = pad_spectrogram(spect)

            spectrograms.append(np.array(input_batches))
            input_masks.append(np.array(input_mask))
            masks.append(np.array(batches_mask))
            samples.append(np.array(batches))

        self.masks = np.array(masks)
        self.samples = np.array(samples)
        self.spectrograms = np.array(spectrograms)
        self.input_masks = np.array(input_masks)

    def __getitem__(self, idx) -> Dict[str, np.array]:
        return {"input_spectrograms": self.spectrograms[idx], "input_mask": self.input_masks[idx], "output_mask": self.masks[idx], "output": self.samples[idx]}

    def __len__(self) -> int: return len(self.samples)

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(
        self) == 0 else f'{len(self.samples)} samples'


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


def get_data(path):
    dataset = MIDIDataset([path], 4.088)
    for i in range(1):

        if not i:
            input = dataset[i]["input_spectrograms"].reshape(
                dataset[i]["input_spectrograms"].shape[0], 512, 512)
            output = dataset[i]["output"]
            output_mask = dataset[i]["output_mask"]
            input_mask = dataset[i]["input_mask"].reshape(
                dataset[i]["input_mask"].shape[0], 512)
            continue
        input = np.append(input, dataset[i]["input_spectrograms"], axis=0)
        output = np.append(output, dataset[i]["output"], axis=0)
        output_mask = np.append(output_mask, dataset[i]["output_mask"], axis=0)
        input_mask = np.append(input_mask, dataset[i]["input_mask"], axis=0)

    return output, input_mask, input, output_mask


def generate_midi_file(predicted_tokens, path="", max_seq_len=4088):
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
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    prev_time = 0
    curr_time = 0
    curr_vel = 0

    abs_time = 0

    queue = {}

    for token in tokens:
        if token:
            if "Velocity" in detokenizer_vocab[token]:
                curr_vel = int(detokenizer_vocab[token].split("_")[1])
            elif "NoteOn" in detokenizer_vocab[token]:
                queue[int(detokenizer_vocab[token].split("_")[1])] = (
                    abs_time, curr_vel)
            elif "NoteOff" in detokenizer_vocab[token]:
                try:
                    start, vel = queue[int(detokenizer_vocab[token].split("_")[1])]
                    pitch = int(detokenizer_vocab[token].split("_")[1])
                    end = abs_time
                    piano.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=pitch, start=start, end=end))
                except KeyError:
                    pass
            else:  # timeshift
                curr_time = int(detokenizer_vocab[token].split("_")[1])
                if curr_time >= prev_time:
                    abs_time += (curr_time - prev_time)/1000
                else:
                    abs_time += (max_seq_len - prev_time + curr_time)/1000
                prev_time = curr_time

    mid.instruments.append(piano)
    mid.write(path)
