import subprocess


def call_train_image():
    root = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/'

    for model in ['ResNet18']:  # ["ResNet50", "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"]:
        for sub_task in ['Native', 'WA', 'YT']:  # ['Native', 'WA', 'YT']:
            for n_fold in range(0, 5):
                subprocess.run([
                    f'/media/blue/tsingalis/miniconda3/envs/videoClass/bin/python {root}train_audio_1D.py '
                    f'--data_content {sub_task} '
                    '--nfft_scale 2 '
                    '--log_mel '
                    f'--n_fold {n_fold} '
                    f'--model {model} '
                    '--project_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers '
                    '--epochs 15 '
                    '--lr 1e-4 '
                    '--train_batch 128 '
                    '--valid_batch 128 '
                    '--audio_frame_indices_dir /media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/ '
                    '--extracted_wav_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/ '
                    '--optimizer Adam --results_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/ '
                    '--data_loader_fold_type my_data_I '
                    '--excluded_devices D12 '
                    '--extend_duration_sec 2 '
                    '--softmax '
                    '--milestones 2 '
                    '--milestones 4 '
                    '--milestones 8 '
                ], shell=True, executable='/bin/bash')


def main():
    call_train_image()


if __name__ == "__main__":
    main()
