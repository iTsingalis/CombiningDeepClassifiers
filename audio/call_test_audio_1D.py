import subprocess
import os, re


def extract_run_numbers(directory):
    # Regular expression to match 'run' followed by digits and capture the digits
    pattern = re.compile(r"run(\d+)$")
    run_numbers = []

    # Iterate through items in the directory
    for item in os.listdir(directory):
        match = pattern.match(item)
        # If there is a match, extract the number part
        if match and os.path.isdir(os.path.join(directory, item)):
            run_numbers.append(int(match.group(1)))  # Convert the captured number to an integer

    return run_numbers


def call_test_audio():
    root = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/'

    for model in ["MobileNetV3Small"]:
        # ["ResNet50", "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"]:
        for sub_task in ['Native', 'WA', 'YT']:  # ['Native', 'WA', 'YT']:
            for n_fold in range(0, 5):
                directory_with_runs = os.path.join(root, 'results1D', 'winSize2sec', sub_task, f'fold{n_fold}', model)
                run_numbers = extract_run_numbers(directory_with_runs)

                for run_number in run_numbers:
                    # if os.path.isfile(os.path.join(directory_with_runs, f'run{run_number}', 'audio_stats.pkl')):
                    #     continue
                    subprocess.run([
                        f'/media/blue/tsingalis/miniconda3/envs/videoClass/bin/python {root}test_audio_1D.py '
                        f'--n_run {run_number} '
                        '--results_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/ '
                        '--project_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/ '
                        f'--data_content {sub_task} '
                        f'--model {model} '
                        f'--n_fold {n_fold} '
                        '--data_loader_fold_type my_data_I '
                        '--audio_frame_indices_dir /media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/ '
                        '--extracted_wav_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/ '
                        '--extend_duration_sec 2 '
                        '--test_batch 256 '
                        # '--save_stats '
                    ], shell=True, executable='/bin/bash')


def main():
    call_test_audio()


if __name__ == "__main__":
    main()
