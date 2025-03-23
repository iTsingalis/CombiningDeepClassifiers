import subprocess


def call_online_fusion_test():
    root = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/fusion/'

    for sub_task in ['YT']:  # ['Native', 'WA', 'YT']:
        for n_fold in range(0, 5):
            subprocess.run([
            f'/media/blue/tsingalis/miniconda3/envs/videoClass/bin/python {root}online_fusion_test.py '
            '--results_dir_fusion /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/fusion/results/ '
            '--results_dir_audio /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/ '
            '--results_dir_visual /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/image/results/ '
            '--audio_frame_indices_dir /media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/ '
            '--project_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/ '
            '--visual_frame_dir /media/red/sharedFolder/Datasets/VISION/keyFrames/I/ '
            '--wav_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/ '
            '--data_loader_fold_type my_data_I '
            '--extend_duration_sec 2 '
            f'--data_content {sub_task} '
            f'--n_fold {n_fold} '
            '--reduction '
            '--audio_model MobileNetV3Large '
            '--visual_model MobileNetV3Small '
            '--loss_fn ProductRuleLoss '
            '--test_batch_size 128'
            f' --n_run 1 '
            # '--save_stats'
            ], shell=True, executable='/bin/bash')


def main():
    call_online_fusion_test()


if __name__ == "__main__":
    main()
