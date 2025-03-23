import os
import subprocess
from glob import glob
from pathlib import Path


def extract_audio_from_video(video_path, output_directory):
    try:
        video_name, video_ext = os.path.splitext(os.path.basename(video_path))
        audio_output_filename = video_name + ".wav"
        audio_output_path = os.path.join(output_directory, audio_output_filename)

        # Use subprocess to call ffmpeg for audio extraction (Christos Korgialas)
        subprocess.run(['ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le', '-ar', '44100', audio_output_path],
                       check=True)
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")


def process_brand(dev_ids):
    excluded_devs = ['D12']

    for n_dev, dev_id in enumerate(dev_ids):

        excluded_flag = any([excluded_dev in dev_id for excluded_dev in excluded_devs])

        if excluded_flag:
            continue

        natural_videos_flat = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            flat_path = os.path.join(devices_folder, dev_id, 'videos', 'flat', type)
            if os.path.exists(Path(flat_path).parent):
                for i, x in enumerate(glob(flat_path)):
                    natural_videos_flat.update({f'flat{i}': x})

        natural_videos_flatWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            flatWA_path = os.path.join(devices_folder, dev_id, 'videos', 'flatWA', type)
            if os.path.exists(Path(flatWA_path).parent):
                for i, x in enumerate(glob(flatWA_path)):
                    natural_videos_flatWA.update({f'flatWA{i}': x})

        natural_videos_flatYT = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            flatYT_path = os.path.join(devices_folder, dev_id, 'videos', 'flatYT', type)
            if os.path.exists(Path(flatYT_path).parent):
                for i, x in enumerate(glob(flatYT_path)):
                    natural_videos_flatYT.update({f'flatYT{i}': x})

        natural_videos_indoor = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoor_path = os.path.join(devices_folder, dev_id, 'videos', 'indoor', type)
            if os.path.exists(Path(indoor_path).parent):
                for i, x in enumerate(glob(indoor_path)):
                    natural_videos_indoor.update({f'indoor{i}': x})

        natural_videos_outdoor = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoor_path = os.path.join(devices_folder, dev_id, 'videos', 'outdoor', type)
            if os.path.exists(Path(outdoor_path).parent):
                for i, x in enumerate(glob(outdoor_path)):
                    natural_videos_outdoor.update({f'outdoor{i}': x})

        natural_videos_indoorWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoorWA_path = os.path.join(devices_folder, dev_id, 'videos', 'indoorWA', type)
            if os.path.exists(Path(indoorWA_path).parent):
                for i, x in enumerate(glob(indoorWA_path)):
                    natural_videos_indoorWA.update({f'indoorWA{i}': x})

        natural_videos_indoorYT = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoorYT_path = os.path.join(devices_folder, dev_id, 'videos', 'indoorYT', type)
            if os.path.exists(Path(indoorYT_path).parent):
                for i, x in enumerate(glob(indoorYT_path)):
                    natural_videos_indoorYT.update({f'indoorYT{i}': x})

        natural_videos_outdoorWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoorWA_path = os.path.join(devices_folder, dev_id, 'videos', 'outdoorWA', type)
            if os.path.exists(Path(outdoorWA_path).parent):
                for i, x in enumerate(glob(outdoorWA_path)):
                    natural_videos_outdoorWA.update({f'outdoorWA{i}': x})

        natural_videos_outdoorYT = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoorYT_path = os.path.join(devices_folder, dev_id, 'videos', 'outdoorYT', type)
            if os.path.exists(Path(outdoorYT_path).parent):
                for i, x in enumerate(glob(outdoorYT_path)):
                    natural_videos_outdoorYT.update({f'outdoorYT{i}': x})

        videos = {**natural_videos_flat,
                  **natural_videos_flatWA,
                  **natural_videos_flatYT,
                  **natural_videos_indoor,
                  **natural_videos_outdoor,
                  **natural_videos_indoorWA,
                  **natural_videos_indoorYT,
                  **natural_videos_outdoorWA,
                  **natural_videos_outdoorYT}

        exception_wav = []

        for indoor_or_outdoor_or_flat, video in videos.items():

            save_folder = os.path.join(output_folder, dev_id, Path(video).stem)

            Path(save_folder).mkdir(parents=True, exist_ok=True)

            try:
                audio_output_path = os.path.join(save_folder, Path(video).stem + '.wav')
                # if os.path.isfile(audio_output_path):
                #     continue

                # Use subprocess to call ffmpeg for audio extraction
                subprocess.run(['ffmpeg', '-i', video, '-acodec', 'pcm_s16le',
                                '-ar', '22050', audio_output_path], check=True)

            except Exception as e:
                print(f"Error processing video {Path(video).stem + '.wav'}: {e}")
                exception_wav.append(Path(video).stem + '.wav')
                with open(os.path.join(output_folder, 'exception_wav.txt'), 'w') as file:
                    for line in exception_wav:
                        file.write("%s\n" % line)


def main():
    dev_ids = [name for name in os.listdir(devices_folder)]

    process_brand(dev_ids=dev_ids)

    print('That is all folks...')


if __name__ == '__main__':
    # You replace devices_folder with /IARIADevIDFusion/datasets/VISION/
    devices_folder = '/media/red/sharedFolder/Datasets/VISION/dataset/'
    output_folder = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav'

    main()
