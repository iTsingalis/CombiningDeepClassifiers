import os
import torch
import pyvips
import pickle
import random
import numpy as np
import image.prnu as prnu
from torch.utils.data import Dataset
from multiprocessing import cpu_count, Pool
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    def __init__(self, pkl_dir, visual_frame_dir,
                 crop_size=(256, 256, 3),
                 fft2=False,
                 prnu_enabled=False,
                 transforms=None,
                 gaussian_noise_flag=False,
                 center_crop=False,
                 sample_prc=-1):
        self.data = []
        self.visual_frame_dir = visual_frame_dir
        self.transforms = transforms
        self.fft2 = fft2
        self.prnu_enabled = prnu_enabled
        self.gaussian_noise_flag = gaussian_noise_flag
        self.center_crop = center_crop

        self.mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.visual_frame_crop_size = (224, 224, 3)

        assert not (fft2 and prnu_enabled), 'fft2 and prnu cannot be enabled simultaneously'

        with open(pkl_dir, "rb") as f:
            self.data = pickle.load(f)

        random.seed(108)
        random.shuffle(self.data)

        self.device_names = set([d['device_name'] for d in self.data])

        if sample_prc > 0:
            assert isinstance(sample_prc, float), 'sample_prc must be a float'
            assert 0 <= sample_prc <= 0.999, "sample_prc must be between 0 and 1"
            sample_prc = 0.999 if sample_prc == 1. else sample_prc
            self.data, _ = train_test_split(self.data, train_size=sample_prc,
                                            stratify=[l['target'] for l in self.data])
        self.n_classes = len(np.unique([d['target'] for d in self.data]))
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data)

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
        entry = self.data[idx]
        img_path = os.path.join(self.visual_frame_dir,
                                entry['model_id'],
                                entry['device_name'], entry['audio_image_frame_name'] + '.png')
        audio_image_frame_indices = entry['audio_image_frame_name'].split('-')[-1]

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

        values = np.transpose(visual_values, (2, 0, 1))

        # values = torch.Tensor(values)
        values = torch.from_numpy(values)

        if self.fft2:
            import matplotlib.pyplot as plt
            # Apply 2D Fourier transform using torch.fft.fft2
            values = torch.fft.fft2(values)

            # Shift the zero frequency to the center using torch.fft.fftshift
            values = torch.fft.fftshift(values)

            # Compute the magnitude (absolute value) of the shifted Fourier transform
            values = torch.abs(values)

            # # Convert to numpy for visualization
            # fft_magnitude_np = fft_magnitude.numpy()
            #
            # # Plot the FFT magnitude for each channel
            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
            #
            # for i in range(3):
            #     axes[i].imshow(np.log1p(fft_magnitude_np[i]), cmap='gray')  # log1p for better visualization
            #     axes[i].set_title(f'{channel_names[i]} FFT Magnitude')
            #     axes[i].axis('off')
            #
            # plt.show()

            # fft2 = torch.fft.fft2(values)

        if self.transforms:
            values = self.transforms(values)
        target = torch.LongTensor([entry["target"]])

        assert 0 <= target <= 33, "Value is out of the expected range (0-33)"

        return values, target, entry['device_name'], audio_image_frame_indices
