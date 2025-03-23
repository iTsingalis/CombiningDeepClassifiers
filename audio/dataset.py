import os
import torch
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wavfile
from scipy.linalg import norm as scipy_norm
# from scipy.fft import rfft
import matplotlib.pyplot as plt
import torchaudio
import librosa
from torch.fft import rfft
import torchvision


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def normalize_l2(i, n):
    return i / n


class AudioDataset1D(Dataset):
    def __init__(self, audio_frame_pkl_idx, audio_frame_indices_dir, extracted_wav_dir,
                 transforms=None, sample_prc=-1, nfft_scale=None, log_mel=None, n_mels=128):
        self.data = []
        self.audio_frame_indices_dir = audio_frame_indices_dir
        self.extracted_wav_dir = extracted_wav_dir
        self.transforms = transforms
        self.nfft_scale = nfft_scale
        self.log_mel = log_mel
        self.n_mels = n_mels

        with open(audio_frame_pkl_idx, "rb") as f:
            self.data = pickle.load(f)

        self.device_names = set([d['device_name'] for d in self.data])

        self.wav_l2_norm = {}

        random.seed(108)
        random.shuffle(self.data)

        if sample_prc > 0:
            assert isinstance(sample_prc, float), 'sample_prc must be a float'
            assert 0 <= sample_prc <= 0.999, "sample_prc must be between 0 and 1"
            sample_prc = 0.999 if sample_prc == 1. else sample_prc
            self.data, _ = train_test_split(self.data, train_size=sample_prc,
                                            stratify=[ll['target'] for ll in self.data])
        self.n_classes = len(np.unique([d['target'] for d in self.data]))
        self.crop_size = (256, 256, 3)

    def __len__(self):
        return len(self.data)

    def get_raw_signal_dim(self):
        return self.raw_signal_dim

    def get_nfft_signal_dim(self):
        return self.fft_signal_dim

    def __getitem__(self, idx):

        entry = self.data[idx]
        audio_frame_pkl_idx = os.path.join(self.audio_frame_indices_dir, entry['model_id'],
                               entry['device_name'], entry['audio_image_frame_name'] + '.pickle')

        audio_image_frame_indices = entry['audio_image_frame_name'].split('-')[-1]

        with open(audio_frame_pkl_idx, "rb") as f:
            audio_frame_idx = pickle.load(f)

        wav_file_path = os.path.join(self.extracted_wav_dir,
                                     entry['model_id'],
                                     entry['device_name'],
                                     entry['device_name'] + '.wav')
        # audio_clip, sr = librosa.load(wav_file_path)
        # sr: 22050
        sr, audio_clip = wavfile.read(wav_file_path)
        if audio_clip.ndim > 1:
            audio_clip = np.mean(audio_clip, axis=1)
        audio_frame = [audio_clip[i % len(audio_clip)]
                       for i in range(audio_frame_idx['start'], audio_frame_idx['stop'])]

        self.raw_signal_dim = len(audio_frame)
        audio_frame = torch.Tensor(audio_frame)

        # audio_frame /= torch.linalg.norm(audio_frame)
        # https://stackoverflow.com/questions/14738221/which-difference-between-normalize-signal-before-or-after-fft
        if not self.log_mel:
            self.fft_signal_dim = next_power_of_2(int(self.nfft_scale * self.raw_signal_dim))
            audio_frame = torch.abs(rfft(audio_frame, n=self.fft_signal_dim, dim=0))[1:]
        elif self.log_mel:

            """
            Smaller Window Size: Better time resolution but worse frequency resolution.
            Larger Window Size: Better frequency resolution but worse time resolution.
            Smaller Hop Size: Better time resolution and more overlap, leading to higher computational cost.
            Larger Hop Size: Worse time resolution and less overlap, leading to lower computational cost.
            """
            window_size_ms = [25, 50, 100]    # msec
            hop_sizes_ms = [10, 25, 50]        # msec
            min_spec_y_dim = np.inf

            specs = []
            num_channels = 3
            for i in range(num_channels):
                window_length = int(round(window_size_ms[i] * sr / 1000))
                hop_length = int(round(hop_sizes_ms[i] * sr / 1000))

                # Ensure n_fft is at least the size of window_length
                # n_fft = max(window_length, 2048)  # Setting a minimum size for n_fft
                # Ensure n_fft is at least the size of window_length and a power of 2
                # n_fft = max(2 ** int(np.ceil(np.log2(window_length))), 2048)  # Setting a minimum size for n_fft
                n_fft = next_power_of_2(int(self.nfft_scale * window_length))
                # Generate Mel spectrogram
                clip = torch.Tensor(audio_frame)
                spec = torchaudio.transforms.MelSpectrogram(sample_rate=int(sr),
                                                            n_fft=n_fft,
                                                            win_length=window_length,
                                                            hop_length=hop_length,
                                                            n_mels=self.n_mels)(clip)
                plt_mel = False
                if plt_mel:
                    fig, ax = plt.subplots()
                    S_dB = librosa.power_to_db(spec, ref=np.max)
                    img = librosa.display.specshow(S_dB, x_axis='time',
                                                   y_axis='mel', sr=sr,
                                                   fmax=8000,
                                                   ax=ax)
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    ax.set(title='Mel-frequency spectrogram')

                    plt.show()

                spec = torch.log(spec + 1e-6)

                spec_x_dim, spec_y_dim = spec.shape[0], spec.shape[1]
                if spec_y_dim < min_spec_y_dim:
                    min_spec_y_dim = spec_y_dim
                # print(spec_y_dim)
                spec = spec.view(1, 1, spec_x_dim, spec_y_dim)
                # spec = torchvision.transforms.Resize((self.n_mels, 1000))(spec).squeeze(1)
                specs.append(spec)
            specs = [torchvision.transforms.Resize((self.n_mels, min_spec_y_dim))(sp).squeeze(1) for sp in specs]
            audio_frame = torch.cat(specs, dim=0)

        if self.transforms:
            audio_frame = self.transforms(audio_frame)
        target = torch.LongTensor([entry["target"]])

        return audio_frame, target, entry['device_name'], audio_image_frame_indices


class AudioDataset(Dataset):
    def __init__(self, split_dir, mel_dir, transforms=None):
        self.mel_dir = mel_dir
        self.data = []
        self.length = 1500
        self.transforms = transforms
        self.data = pd.read_csv(split_dir)
        self.n_classes = len(np.unique(self.data['dev_numerical_id'].values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        mel_path = os.path.join(self.mel_dir,
                                item['dev_alphabetical_id'],
                                item['dev_alphabetical_id'] + '.pkl')
        with open(mel_path, "rb") as f:
            values = pickle.load(f)

        mel = values['mel'].reshape(-1, 128, self.length)
        mel = torch.Tensor(mel)
        if self.transforms:
            mel = self.transforms(mel)
        target = torch.LongTensor([item['dev_numerical_id']])
        return mel, target, item['dev_alphabetical_id']
