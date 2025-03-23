import os
import torch
import pyvips
import pickle
import random
import numpy as np
import image.prnu as prnu
import scipy.io.wavfile as wavfile
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import librosa
import torchaudio
import torchvision
from torch.fft import rfft


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class ImageAudioDataset(Dataset):
    def __init__(self, pkl_fold_dir,
                 visual_frame_dir,
                 wav_dir,
                 audio_frame_indices_dir,
                 fft2_enabled=False, prnu_enabled=False,
                 nfft_scale=None, log_mel=None, n_mels=128,
                 gaussian_noise_flag=False,
                 center_crop=False,
                 sample_prc=-1):
        # Visual attributes
        self.visual_frame_dir = visual_frame_dir
        self.fft2_enabled = fft2_enabled
        self.prnu_enabled = prnu_enabled
        self.gaussian_noise_flag = gaussian_noise_flag
        self.center_crop = center_crop

        # Audio attributes
        self.audio_frame_indices_dir = audio_frame_indices_dir
        self.wav_dir = wav_dir
        self.nfft_scale = nfft_scale
        self.log_mel = log_mel
        self.n_mels = n_mels

        self.mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        assert not (fft2_enabled and prnu), 'fft2 and prnu cannot be enabled simultaneously in visual content'

        with open(pkl_fold_dir, "rb") as f:
            self.fold_data = pickle.load(f)

        random.seed(108)
        random.shuffle(self.fold_data)

        self.device_names = set([d['device_name'] for d in self.fold_data])

        if sample_prc > 0:
            assert isinstance(sample_prc, float), 'sample_prc must be a float'
            assert 0 <= sample_prc <= 0.999, "sample_prc must be between 0 and 1"
            sample_prc = 0.999 if sample_prc == 1. else sample_prc
            self.fold_data, _ = train_test_split(self.fold_data, train_size=sample_prc,
                                                 stratify=[l['target'] for l in self.fold_data])
        self.n_classes = len(np.unique([d['target'] for d in self.fold_data]))
        self.visual_frame_crop_size = (224, 224, 3)

    def __len__(self):
        return len(self.fold_data)

    def add_gaussian_noise(self, image, mean=0.0, var=0.01, clip=True):

        # Generate Gaussian noise
        sigma = np.sqrt(var)
        gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

        # Add the noise to the image
        noisy_image = image + gaussian_noise

        # Clip values if required to keep them in the [0, 1] range
        if clip:
            noisy_image = np.clip(noisy_image, 0, 1)

        return noisy_image

    # Normalize the image (assuming image pixel values are in range [0, 1])
    def normalize_image(self, image, gaussian_noise_flag=False):
        # Ensure the image is a float array
        image = image.numpy(dtype=np.float32) / 255.0

        if gaussian_noise_flag:
            image = self.add_gaussian_noise(image, mean=0.0, var=0.01, clip=True)

        # Normalize per channel: (value - mean) / std
        normalized_image = (image - self.mean_rgb) / self.std_rgb

        return normalized_image

    def usingVIPSandShrink(self, f, normalize=True, gaussian_noise_flag=False):
        image = pyvips.Image.new_from_file(f, access="sequential")  # RGB image
        if normalize:
            # image = image.numpy(dtype=np.float32) / 255.0
            image = self.normalize_image(image, gaussian_noise_flag)
        else:
            image = image.numpy(dtype=np.uint8)
        return image

    def __getitem__(self, idx):
        entry = self.fold_data[idx]

        ######################################
        ###### Load Image Data
        ######################################

        img_path = os.path.join(self.visual_frame_dir, entry['model_id'],
                                entry['device_name'], entry['audio_image_frame_name'] + '.png')

        visual_values = self.usingVIPSandShrink(img_path,
                                                normalize=not self.prnu_enabled,
                                                gaussian_noise_flag=self.gaussian_noise_flag)

        # If center_crop is not enabled, then the random crop is performed.
        visual_values = prnu.cut_ctr(visual_values,
                                     sizes=self.visual_frame_crop_size,
                                     center_crop=self.center_crop)

        if self.prnu_enabled:
            visual_values = np.dstack((prnu.extract_single(visual_values[:, :, 0]),
                                       prnu.extract_single(visual_values[:, :, 1]),
                                       prnu.extract_single(visual_values[:, :, 2])))

        visual_values = np.transpose(visual_values, (2, 0, 1))  # BRG preferred by pytorch

        visual_values = torch.from_numpy(visual_values)

        if self.fft2_enabled:
            import matplotlib.pyplot as plt
            # Apply 2D Fourier transform using torch.fft.fft2_enabled
            visual_values = torch.fft.fft2_enabled(visual_values)

            # Shift the zero frequency to the center using torch.fft.fftshift
            visual_values = torch.fft.fftshift(visual_values)

            # Compute the magnitude (absolute value) of the shifted Fourier transform
            visual_values = torch.abs(visual_values)

        ######################################
        ###### Load Audio Data
        ######################################

        audio_frame_buffer_indices = os.path.join(self.audio_frame_indices_dir,
                                                  entry['model_id'],
                                                  entry['device_name'], entry['audio_image_frame_name'] + '.pickle')

        audio_image_frame_indices = entry['audio_image_frame_name'].split('-')[-1]

        with open(audio_frame_buffer_indices, "rb") as f:
            audio_frame_idx = pickle.load(f)

        wav_file_path = os.path.join(self.wav_dir,
                                     entry['model_id'],
                                     entry['device_name'],
                                     entry['device_name'] + '.wav')
        # audio_clip, sr = librosa.load(wav_file_path)
        # sr: 22050
        sr, audio_clip = wavfile.read(wav_file_path)
        if audio_clip.ndim > 1:
            audio_clip = np.mean(audio_clip, axis=1)
        audio_values = [audio_clip[i % len(audio_clip)]
                        for i in range(audio_frame_idx['start'], audio_frame_idx['stop'])]

        self.raw_signal_dim = len(audio_values)
        audio_values = torch.Tensor(audio_values)

        # audio_values /= torch.linalg.norm(audio_values)
        # https://stackoverflow.com/questions/14738221/which-difference-between-normalize-signal-before-or-after-fft
        if not self.log_mel:
            self.fft_signal_dim = next_power_of_2(int(self.nfft_scale * self.raw_signal_dim))
            audio_values = torch.abs(rfft(audio_values, n=self.fft_signal_dim, dim=0))[1:]
        elif self.log_mel:

            """
            Smaller Window Size: Better time resolution but worse frequency resolution.
            Larger Window Size: Better frequency resolution but worse time resolution.
            Smaller Hop Size: Better time resolution and more overlap, leading to higher computational cost.
            Larger Hop Size: Worse time resolution and less overlap, leading to lower computational cost.
            """
            window_size_ms = [25, 50, 100]  # msec
            hop_sizes_ms = [10, 25, 50]  # msec
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
                clip = torch.Tensor(audio_values)
                spec = torchaudio.transforms.MelSpectrogram(sample_rate=int(sr),
                                                            n_fft=n_fft,
                                                            win_length=window_length,
                                                            hop_length=hop_length,
                                                            n_mels=self.n_mels)(clip)
                plt_mel = False
                if plt_mel:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    S_dB = librosa.power_to_db(spec, ref=np.max)
                    img = librosa.display.specshow(S_dB, x_axis='time',
                                                   y_axis='mel', sr=sr,
                                                   fmax=8000,
                                                   ax=ax)
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    ax.set(title='Mel-frequency spectrogram')

                    # plt.savefig(os.path.join(mel_path, f'{Path(mel_path).stem}_chanel{i}.png'),
                    #             bbox_inches='tight', pad_inches=0, transparent=True)
                    # plt.close()
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

            audio_values = torch.cat(specs, dim=0)

        target = torch.LongTensor([entry["target"]])

        assert 0 <= target <= 33, "Value is out of the expected range (0-33)"

        return visual_values, audio_values, target, entry['device_name'], audio_image_frame_indices
