import subprocess


def call_test_image():
    root = '/media/blue/tsingalis/DevIDFusion/image/'

    for model in [
        "ResNet50", "ResNet18", "MobileNetV3Small", "MobileNetV3Large"]:  # ["ResNet50", "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"]:
        for sub_task in ['Native', 'WA', 'YT']:  # ['Native', 'WA', 'YT']:
            for n_fold in range(0, 5):
                subprocess.run([
                    f'/media/blue/tsingalis/miniconda3/envs/videoClass/bin/python {root}test_image.py '
                    '--n_run 1 '
                    f'--data_content {sub_task} '
                    f'--model {model} '
                    f'--n_fold {n_fold} '
                    '--visual_frames_dir /media/red/sharedFolder/Datasets/VISION/keyFrames/I/ '
                    '--project_dir /media/blue/tsingalis/DevIDFusion/ '
                    '--results_dir /media/blue/tsingalis/DevIDFusion/image/results/ '
                    '--data_loader_fold_type my_data_I '
                    '--extend_duration_sec 2 '
                    '--test_batch 128'
                ], shell=True, executable='/bin/bash')


def main():
    call_test_image()


if __name__ == "__main__":
    main()
