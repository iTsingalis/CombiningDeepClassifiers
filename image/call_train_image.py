import subprocess



def call_train_image():
    root = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/image/'

    for model in ['MobileNetV3Large']:  # ["ResNet50", "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"]:
        for sub_task in ['Native', 'WA', 'YT']:  # ['Native', 'WA', 'YT']:
            for n_fold in range(0, 5):
                subprocess.run([
                    f'/media/blue/tsingalis/miniconda3/envs/videoClass/bin/python {root}train_image.py '
                    f'--data_content {sub_task} '
                    f'--n_fold {n_fold} '
                    f'--model {model} '
                    f'--project_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/ '
                    f'--epochs 12 '
                    f'--lr 1e-4 '
                    f'--visual_frames_dir /media/red/sharedFolder/Datasets/VISION/keyFrames/I/ '
                    f'--optimizer Adam --results_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/image/results/ '
                    f'--data_loader_fold_type my_data_I '
                    f'--excluded_devices D12 '
                    f'--extend_duration_sec 2 '
                    f'--train_batch 128 '
                    f'--valid_batch 128 '
                    f'--milestones 2 '
                    f'--milestones 4 '
                    f'--milestones 8 '
                    f'--softmax '
                    # '--save_stats'
                ], shell=True, executable='/bin/bash')


def main():
    call_train_image()


if __name__ == "__main__":
    main()
