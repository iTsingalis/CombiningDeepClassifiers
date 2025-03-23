"""
Reference: https://github.com/bgswaroop/scd-videos/blob/main/dataset/vision/frame_extraction.py
"""
import os
import re
import cv2
import argparse
import subprocess
from glob import glob
from tqdm import tqdm
from pathlib import Path


def parse_args():
    """
--input_dir
/media/red/sharedFolder/Datasets/VISION/dataset/
--output_dir
/media/red/sharedFolder/Datasets/VISION/keyFrames/
--frame_type
I
    """
    """Command line arguments for frame extraction

    :return: args namespace
    """
    parser = argparse.ArgumentParser(
        description='Extract and save I-frames from videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=Path, required=True,
                        help='Path to VISION dataset (directory consisting of folders per device)')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory path to save the video frames')
    parser.add_argument('--frame_type', type=str, default='I', choices=['I', 'all'])
    parser.add_argument('--save_IFrame_idx_only', action='store_true')
    parser.add_argument('--I_ids', action='store_true', default=False)

    args = parser.parse_args()

    assert args.input_dir.exists(), 'Input directory does not exists!'
    if args.I_ids:
        args.output_dir = args.output_dir.joinpath('I_ids')
    else:
        args.output_dir = args.output_dir.joinpath(args.frame_type)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.devices = ['D01_Samsung_GalaxyS3Mini',
                    'D02_Apple_iPhone4s',
                    'D03_Huawei_P9',
                    'D04_LG_D290',
                    'D05_Apple_iPhone5c',
                    'D06_Apple_iPhone6',
                    'D07_Lenovo_P70A',
                    'D08_Samsung_GalaxyTab3',
                    'D09_Apple_iPhone4',
                    'D10_Apple_iPhone4s',
                    'D11_Samsung_GalaxyS3',
                    'D12_Sony_XperiaZ1Compact',
                    'D14_Apple_iPhone5c',
                    'D15_Apple_iPhone6',
                    'D16_Huawei_P9Lite',
                    'D18_Apple_iPhone5c',
                    'D19_Apple_iPhone6Plus',
                    'D24_Xiaomi_RedmiNote3',
                    'D25_OnePlus_A3000',
                    'D26_Samsung_GalaxyS3Mini',
                    'D27_Samsung_GalaxyS5',
                    'D28_Huawei_P8',
                    'D29_Apple_iPhone5',
                    'D30_Huawei_Honor5c',
                    'D31_Samsung_GalaxyS4Mini',
                    'D32_OnePlus_A3003',
                    'D33_Huawei_Ascend',
                    'D34_Apple_iPhone5']

    # Additional devices
    args.devices += ['D13_Apple_iPad2',
                     'D17_Microsoft_Lumia640LTE',
                     'D20_Apple_iPadMini',
                     'D21_Wiko_Ridge4G',
                     'D22_Samsung_GalaxyTrendPlus',
                     'D23_Asus_Zenfone2Laser',
                     'D35_Samsung_GalaxyTabA']

    return args


def extract_frames_from_a_video(video, dest_device_dir, frame_type, I_ids):
    """Extract and save all frames from a video to a destination directory

    :param video: Path to the video file
    :param dest_device_dir: Path to the destination directory corresponding to the video's source camera device
    :param frame_type: A string indicating the type of frames to extract ['I', 'all']
    :return: None
    """

    dest_frames_dir = dest_device_dir.joinpath(video.stem)
    dest_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = None
    # load the ffmpeg module on the Peregrine
    # subprocess.run('module load FFmpeg/4.2.2-GCCcore-9.3.0', shell=True, capture_output=True)
    if frame_type == 'I':
        if I_ids:
            if not Path(dest_frames_dir.joinpath('IFrames.txt')).exists():
                # Identify the I-frames
                cmd = f"ffprobe {str(video)} -show_frames | grep -E 'pict_type'"
                frame_types = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode('ascii').split('\n')
                frame_ids = set([str(idx + 1).zfill(5) for idx, x in enumerate(frame_types) if x == 'pict_type=I'])

                with open(dest_frames_dir.joinpath('IFrames.txt'), 'w') as f:
                    for line in list(sorted(frame_ids)):
                        f.write(f"{line}\n")
        else:
            # Identify the I-frames
            cmd = f"ffprobe {str(video)} -show_frames | grep -E 'pict_type'"
            frame_types = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode('ascii').split('\n')
            frame_ids = set([str(idx + 1).zfill(5) for idx, x in enumerate(frame_types) if x == 'pict_type=I'])

    existing_frame_ids = [re.search(r'-(\d+)\.png', Path(i).name).group(1)
                          for i in glob(os.path.join(dest_frames_dir, '*.png'))]

    if len(set(frame_ids) - set(existing_frame_ids)) > 1:
        frame_ids = set(frame_ids) - set(existing_frame_ids)
    else:
        return
    cap = cv2.VideoCapture(str(video))  # Start capturing the feed
    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    while cap.isOpened():
        # exit condition
        count += 1
        if count > total_num_frames and cap.isOpened():
            cap.release()
            break

        ret, frame = cap.read()  # Extract the frame
        if ret:  # Frame is available
            # cap.get(1) means CAP_PROP_POS_FRAMES - 0-based index of the frame to be decoded/captured next.
            frame_id = str(int(cap.get(1))).zfill(5)  # Get current frame id
            if frame_type == 'I' and frame_id not in frame_ids and not I_ids:
                continue
            frame_path = dest_frames_dir.joinpath(f"{video.stem}-{frame_id}.png")
            if not Path(frame_path).exists():
                cv2.imwrite(str(frame_path), frame)


def run_flow():
    args = parse_args()

    for dev_i, device_name in enumerate(args.devices):

        source_device_dir = args.input_dir.joinpath(device_name)
        dest_device_dir = args.output_dir.joinpath(device_name)
        dest_device_dir.mkdir(parents=True, exist_ok=True)
        videos = list(source_device_dir.glob('videos/*/*'))
        nth_video = 1

        with tqdm(total=len(videos), desc=f'Device: ?? Video: ?? ', leave=True, position=0) as pbar:
            for video in videos:
                # print(f'\nStarted processing video - {video}')
                extract_frames_from_a_video(video, dest_device_dir, args.frame_type, args.I_ids)
                pbar.set_description(desc=f'Device: {device_name} ({dev_i + 1} / {len(args.devices)}) '
                                          f'-- Video: {Path(video).stem}')
                pbar.update()
                nth_video += 1
        # print('Finished processing the video')
        # create_frames_dataset(source_device_dir, dest_device_dir, args.frame_type, args.save_IFrame_idx_only)


def measure_time_to_extract_I_frames():
    """
    Sample function to measure the runtime performance of frame extraction.
    The results could vary depending on the hardware.
    """
    path_to_sample_video = 'replace_with_absolute_path_to_the_video_file'
    cmd = f"""time ffmpeg -i {path_to_sample_video} -vf "select='eq(pict_type,I)'" -vsync vfr out-%04d.png"""
    subprocess.run(cmd, shell=True, capture_output=True)


if __name__ == '__main__':
    run_flow()
