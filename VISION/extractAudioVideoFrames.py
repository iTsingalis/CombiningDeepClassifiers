import os
import cv2
import json
import pickle
import librosa
import subprocess
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from itertools import cycle

import soundfile as sf
import torchaudio
import torch
import matplotlib.pyplot as plt
from array import array
from itertools import groupby
from operator import itemgetter
import argparse


def get_video_duration(file):
    """Get the duration of a video using ffprobe."""
    cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(file)
    output = subprocess.check_output(
        cmd,
        shell=True,  # Let this run in the shell
        stderr=subprocess.STDOUT
    )
    # return round(float(output))  # ugly, but rounds your seconds up or down
    return float(output)


def process_brand(model_ids):
    asynchronous_audio_video_content, audio_frame_stats = [], []

    for n_model, model_id in enumerate(model_ids):
        natural_videos_flat = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            flat_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'flat', type)
            if os.path.exists(Path(flat_path).parent):
                audio_files_flat = glob(flat_path)
                for i, x in enumerate(audio_files_flat):
                    natural_videos_flat.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        natural_videos_flatWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            flatWA_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'flatWA', type)
            if os.path.exists(Path(flatWA_path).parent):
                audio_files_flatWA = glob(flatWA_path)
                for i, x in enumerate(audio_files_flatWA):
                    natural_videos_flatWA.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        natural_videos_flatYT = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            flatYT_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'flatYT', type)
            if os.path.exists(Path(flatYT_path).parent):
                audio_files_flatYT = glob(flatYT_path)
                for i, x in enumerate(audio_files_flatYT):
                    natural_videos_flatYT.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        natural_videos_indoor = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoor_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'indoor', type)
            if os.path.exists(Path(indoor_path).parent):
                audio_files_indoor = glob(indoor_path)
                for i, x in enumerate(audio_files_indoor):
                    natural_videos_indoor.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        natural_videos_outdoor = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoor_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'outdoor', type)
            if os.path.exists(Path(outdoor_path).parent):
                audio_files_outdoor = glob(outdoor_path)
                for i, x in enumerate(audio_files_outdoor):
                    natural_videos_outdoor.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        natural_videos_indoorWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoorWA_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'indoorWA', type)
            if os.path.exists(Path(indoorWA_path).parent):
                audio_files_indoorWA = glob(indoorWA_path)
                for i, x in enumerate(audio_files_indoorWA):
                    natural_videos_indoorWA.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        natural_videos_indoorYT = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoorYT_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'indoorYT', type)
            if os.path.exists(Path(indoorYT_path).parent):
                audio_files_indoorYT = glob(indoorYT_path)
                for i, x in enumerate(audio_files_indoorYT):
                    natural_videos_indoorYT.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        natural_videos_outdoorWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoorWA_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'outdoorWA', type)
            if os.path.exists(Path(outdoorWA_path).parent):
                audio_files_outdoorWA = glob(outdoorWA_path)
                for i, x in enumerate(audio_files_outdoorWA):
                    natural_videos_outdoorWA.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        natural_videos_outdoorYT = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoorYT_path = os.path.join(args.vision_dataset_folder, model_id, 'videos', 'outdoorYT', type)
            if os.path.exists(Path(outdoorYT_path).parent):
                audio_files_outdoorYT = glob(outdoorYT_path)
                for i, x in enumerate(audio_files_outdoorYT):
                    natural_videos_outdoorYT.update({Path(x).stem: {"video_path": x, 'model_id': model_id}})

        all_videos = {**natural_videos_flat,
                      **natural_videos_flatWA,
                      **natural_videos_flatYT,
                      **natural_videos_indoor,
                      **natural_videos_outdoor,
                      **natural_videos_indoorWA,
                      **natural_videos_outdoorWA,
                      **natural_videos_indoorYT,
                      **natural_videos_outdoorYT}

        excluded_devs = ['D12']

        for indoor_or_outdoor_or_flat_key, natural_video in all_videos.items():
            video_id = Path(natural_video["video_path"]).stem

            excluded_flag = any([excluded_dev in natural_video['model_id'] for excluded_dev in excluded_devs])

            if excluded_flag:
                continue

            video_duration_sec = get_video_duration(natural_video['video_path'])

            cap = cv2.VideoCapture(natural_video['video_path'])
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            audio_clip, sr = librosa.load(os.path.join(str(args.extracted_wav_folder),
                                                       natural_video['model_id'],
                                                       video_id,
                                                       video_id + '.wav'))
            audio_duration_sec = len(audio_clip) / sr
            if not np.allclose(video_duration_sec, audio_duration_sec, rtol=1.e-2):
                asynchronous_audio_video_content.append({'model_id': model_id,
                                                         'video_id': video_id,
                                                         'sec_dif': abs(video_duration_sec - audio_duration_sec)})
                continue

            save_folder = os.path.join(os.path.join(str(args.output_folder), natural_video['model_id'],
                                                    Path(natural_video['video_path']).stem))

            # extracted_video_frames = glob(os.path.join(args.extracted_visual_frames_folder,
            #                                            natural_video['model_id'],
            #                                            Path(natural_video['video_path']).stem, '*.png'))

            Path(save_folder).mkdir(parents=True, exist_ok=True)

            pbar = tqdm(total=total_frames + 1, position=0, leave=True,
                        desc=f'Device {model_id} ({n_model}/{len(model_ids)}) '
                             f'-- Video: {Path(natural_video["video_path"]).stem} -- Process frames...')

            audio_frame_lengths = []
            audio_frames = []
            count = 0
            while cap.isOpened():
                pbar.update()

                # exit condition
                count += 1
                if count > total_frames and cap.isOpened():
                    cap.release()
                    break

                frame_exists, curr_frame = cap.read()

                save_name = lambda ext_name, str_frame_id: \
                    f"{Path(natural_video['video_path']).stem}-{str_frame_id}.{ext_name}"

                if frame_exists:
                    int_frame_id = int(cap.get(1))
                    str_frame_id = str(int_frame_id).zfill(5)

                    if args.frame_type == 'I':
                        with open(os.path.join(args.extracted_visual_frames_folder,
                                               'I_ids', model_id,
                                               indoor_or_outdoor_or_flat_key,
                                               'IFrames.txt')) as file:
                            key_frame_ids = [line.rstrip() for line in file]

                        if str_frame_id in key_frame_ids:
                            continue

                    running_video_frame_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    running_audio_index = round(running_video_frame_timestamp_sec * sr)
                    if args.extend_duration_sec > 0:
                        num_audio_elements_to_extend = round(args.extend_duration_sec * sr)
                    else:
                        num_audio_elements_to_extend = 735
                    running_audio_index_offset = running_audio_index + num_audio_elements_to_extend

                    audio_frame_stats.append({'model_id': model_id,
                                              'video_id': Path(natural_video['video_path']).stem})

                    audio_frame = {'start': running_audio_index, 'stop': running_audio_index_offset}
                    # print(audio_frame)

                    audio_frames.append(audio_frame)

                    audio_frame_lengths.append(len(audio_frame))

                    save_wav = False
                    if save_wav:
                        sf.write(os.path.join(save_folder, save_name("wav", str_frame_id)),
                                 audio_frame, sr, subtype='PCM_24')

                    save_pickle = False
                    if save_pickle:
                        with open(os.path.join(save_folder, save_name("pickle", str_frame_id)), 'wb') as handle:
                            pickle.dump(audio_frame, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    plt_mel = False
                    if plt_mel:
                        num_channels = 3
                        window_sizes = [25, 50, 100]
                        hop_sizes = [10, 25, 50]

                        for i in range(num_channels):
                            window_length = int(round(window_sizes[i] * sr / 1000))
                            hop_length = int(round(hop_sizes[i] * sr / 1000))
                            # Generate Mel spectrogram

                            clip = torch.Tensor(audio_frame)
                            spec = torchaudio.transforms.MelSpectrogram(sample_rate=int(sr),
                                                                        n_fft=window_length,
                                                                        win_length=window_length,
                                                                        hop_length=hop_length,
                                                                        f_max=int(sr / 3),
                                                                        n_mels=64)(
                                clip)  # Check this otherwise use 2400
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

            # assert len(set(audio_frame_lengths)) == 1
            # if len(audio_frame_lengths) != len(extracted_video_frames):
            #     raise ValueError(f'audio_frame_lengths: {audio_frame_lengths} '
            #                      f'-- extracted_video_frames: {extracted_video_frames}')
            pbar.close()
            cap.release()


def main():
    model_ids = [name for name in os.listdir(args.vision_dataset_folder)]
    process_brand(model_ids=model_ids)

    print('That is all folks...')


if __name__ == '__main__':
    """
--vision_dataset_folder
/media/red/sharedFolder/Datasets/VISION/dataset/
--extracted_wav_folder
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/
--output_folder
/media/red/sharedFolder/Datasets/VISION/AudioFrameIndices
--extracted_visual_frames_folder
/media/red/sharedFolder/Datasets/VISION/keyFrames/
--frame_type
I
--extend_duration_sec
2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_dataset_folder", type=str, required=True,
                        help="The folder where the VISION dataset downloaded.")
    parser.add_argument("--extracted_wav_folder", type=str, required=True,
                        help="The folder where the VISION dataset wav files are extract.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="The folder where the audio frame indices are extracted.")
    parser.add_argument("--extracted_visual_frames_folder", type=str, required=True,
                        help="The folder where the visual frames are extracted.")
    parser.add_argument("--frame_type", type=str, required=True, choices=['all', 'I'])
    parser.add_argument("--extend_duration_sec", type=int, required=True)

    args = parser.parse_args()

    # vision_dataset_folder = '/media/red/sharedFolder/Datasets/VISION/dataset/'
    # extracted_wav_folder = '/media/blue/tsingalis/DevIDFusion/audio/extractedWav/'
    # extracted_visual_frames_folder = '/media/red/sharedFolder/Datasets/VISION/keyFrames/'
    # output_folder = '/media/red/sharedFolder/Datasets/VISION/extractedAudioVideoFrames'

    args.output_folder = os.path.join(args.output_folder, args.frame_type, f'winSize{args.extend_duration_sec}sec')
    # args.extracted_visual_frames_folder = os.path.join(args.extracted_visual_frames_folder, args.frame_type)

    main()
