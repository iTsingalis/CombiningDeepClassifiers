
import os
import torch
import librosa
import torchaudio
import numpy as np
import torchvision
import pickle as pkl
from glob import glob
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def extract_spectrogram(sr, clip, target, mel_path):
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i] * sr / 1000))
        hop_length = int(round(hop_sizes[i] * sr / 1000))

        clip = torch.Tensor(clip)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                    n_fft=4410,
                                                    win_length=window_length,
                                                    hop_length=hop_length,
                                                    n_mels=128)(clip)  # Check this otherwise use 2400
        plt_mel = True
        if plt_mel:
            fig, ax = plt.subplots()
            S_dB = librosa.power_to_db(spec, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time',
                                           y_axis='mel', sr=sr,
                                           fmax=8000,
                                           ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')

            plt.savefig(mel_path + '.png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            plt.show()

        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec + eps)

        # Resize is scaling, not crop
        spec = np.asarray(torchvision.transforms.Resize((128, 1500))(Image.fromarray(spec)))
        specs.append(spec)

    new_entry = {}
    new_entry["audio"] = clip.numpy()
    new_entry["values"] = np.array(specs)
    new_entry["target"] = target

    return [new_entry]


def process_brand(dev_ids):

    for n_dev, dev_id in enumerate(dev_ids):
        print(f'Dev_id: {dev_id}')

        mel_path = os.path.join(output_folder, dev_id)
        if os.path.exists(mel_path):
            continue

        wav_path = glob(os.path.join(devices_folder, dev_id, '*.wav'))
        if len(wav_path):
            wav_path = glob(os.path.join(devices_folder, dev_id, '*.wav')).pop()
        else:
            print(f'No wav found for {dev_id}')
            continue

        Path(mel_path).mkdir(parents=True, exist_ok=True)

        clip, sr = librosa.load(wav_path)

        num_channels = 3
        window_sizes = [25, 50, 100]
        hop_sizes = [10, 25, 50]

        specs = []
        for i in range(num_channels):
            window_length = int(round(window_sizes[i] * sr / 1000))
            hop_length = int(round(hop_sizes[i] * sr / 1000))
            # Generate Mel spectrogram

            clip = torch.Tensor(clip)
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=int(sr),
                                                        n_fft=window_length,
                                                        win_length=window_length,
                                                        hop_length=hop_length,
                                                        n_mels=128)(clip)  # Check this otherwise use 2400
            plt_mel = True
            if plt_mel:
                fig, ax = plt.subplots()
                S_dB = librosa.power_to_db(spec, ref=np.max)
                img = librosa.display.specshow(S_dB, x_axis='time',
                                               y_axis='mel', sr=sr,
                                               fmax=8000,
                                               ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set(title='Mel-frequency spectrogram')

                plt.savefig(os.path.join(mel_path, f'{Path(mel_path).stem}_chanel{i}.png'),
                            bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
                plt.show()

            eps = 1e-6
            spec = spec.numpy()
            spec = np.log(spec + eps)

            # Resize is scaling, not crop
            spec = np.asarray(torchvision.transforms.Resize((128, 1500))(Image.fromarray(spec)))
            specs.append(spec)

        mel_dict = {"mel": np.array(specs), "label": dev_id}
        with open(os.path.join(mel_path, f'{Path(mel_path).stem}.pkl'), "wb") as handler:
            pkl.dump(mel_dict, handler, protocol=pkl.HIGHEST_PROTOCOL)


def main():
    dev_ids = [name for name in os.listdir(devices_folder)]

    process_brand(dev_ids=dev_ids)

    print('That is all folks...')


if __name__ == '__main__':
    devices_folder = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav'
    output_folder = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedMel'

    main()
