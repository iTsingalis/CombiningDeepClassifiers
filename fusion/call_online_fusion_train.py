import subprocess


def call_online_fusion_train():

    project_root = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/'
    dataset_root = '/media/red/sharedFolder/Datasets/VISION/'

    for model in [
        {'audio': 'MobileNetV3Large', 'visual': 'MobileNetV3Small'}]:
        # ["ResNet50", "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"]:
        for sub_task in ['YT']:  # ['Native', 'WA', 'YT']:
            for n_fold in range(0, 5):
                subprocess.run([
                    f'python {project_root}/fusion/online_fusion_train.py '
                    f'--results_dir_fusion {project_root}/fusion/results/ '
                    f'--results_dir_audio {project_root}/audio/results1D/ '
                    f'--results_dir_visual {project_root}/image/results/ '
                    f'--audio_frame_indices_dir {dataset_root}/AudioFrameIndices/I/ '
                    f'--project_dir {project_root} '
                    f'--visual_frame_dir {dataset_root}/keyFrames/I/ '
                    f'--wav_dir {project_root}/audio/extractedWav/ '
                    '--extend_duration_sec 2 '
                    f'--data_content {sub_task} '
                    f'--n_fold {n_fold} '
                    '--n_run_audio 1 '
                    '--n_run_visual 1 '
                    '--milestones 2 '
                    '--milestones 4 '
                    '--milestones 7 '
                    '--milestones 10 '
                    '--reduction '
                    '--train_batch 128 '
                    '--valid_batch 128 '
                    '--epochs 55 '
                    '--loss_fn ProductRuleLoss '
                    f'--audio_model {model["audio"]} '
                    f'--visual_model {model["visual"]} '
                    '--audio_lr 1e-4 '
                    '--visual_lr 1e-4 '
                    '--lr_scheduler MultiStepLR '
                    '--pre_train '
                    '--label_smoothing 0.2 '
                    '--adam_weight_decay 0 '
                    '--window_size 3 '
                    # '--save_stats'
                ], shell=True, executable='/bin/bash')


def main():
    call_online_fusion_train()


if __name__ == "__main__":
    main()
