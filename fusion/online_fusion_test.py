from fusion.dataset import ImageAudioDataset
from sklearn.metrics import confusion_matrix
from models.audio_image_models import *
from torch.utils.data import DataLoader
from collections import defaultdict
from general_utils import print_args
from general_utils import save_cm_stats
from tqdm import tqdm
import pickle as pkl
import numpy as np
import argparse
import torch
import json
import os


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def most_common(lst):
    return max(set(lst), key=lst.count)


def save_stats(modality_stats, best_epoch, modality_type):
    modality_stats = {k: defaultdict_to_dict(v) for k, v in modality_stats.items()}

    per_frame_target_labels_dict, per_frame_gnd_labels_dict = {}, {}

    for devices_name in test_loader.dataset.device_names:
        per_frame_target_labels_dict.update({devices_name: {}})
        per_frame_gnd_labels_dict.update({devices_name: {}})

    per_vid_predicted_labels_dicts, per_vid_target_labels_dicts = [], []
    for dev_key, frame_values in modality_stats.items():
        pred_frame_dict = dict([(frame_idx, frame_value["pred_frame_label"]) for
                                frame_idx, frame_value in frame_values.items()])

        gnd_frame_dict = dict([(frame_idx, frame_value["gnd_frame_label"]) for
                               frame_idx, frame_value in frame_values.items()])

        # frame_probabilities = [frame_value["frame_probability"] for _, frame_value in frame_values.items()]

        major_vid_label = most_common(list(pred_frame_dict.values()))
        gnd_vid_label = most_common(list(gnd_frame_dict.values()))

        per_vid_predicted_labels_dicts.append({dev_key: int(major_vid_label)})
        per_vid_target_labels_dicts.append({dev_key: int(gnd_vid_label)})

        for k, v in frame_values.items():
            per_frame_target_labels_dict[dev_key].update({k: v["pred_frame_label"]})
            per_frame_gnd_labels_dict[dev_key].update({k: v["gnd_frame_label"]})

        # per_vid_proba.update({dev_key: np.mean(frame_probabilities)})

    # Compare the values for each dictionary
    total_vid_acc = np.mean([
        all(p_val == g_val for p_val, g_val in zip(p.values(), g.values()))
        for p, g in zip(per_vid_predicted_labels_dicts, per_vid_target_labels_dicts)
    ])

    per_vid_target, per_vid_pred = zip(*[(p_val, g_val)
                                         for p, g in zip(per_vid_predicted_labels_dicts, per_vid_target_labels_dicts)
                                         for p_val, g_val in zip(p.values(), g.values())])

    print(f'total_video_acc ({modality_type}) : {total_vid_acc}')

    per_frame_pred_labels, per_frame_target_labels = [], []
    for device_id, frame_id_target_label in per_frame_target_labels_dict.items():
        for frame_id, frame_target_label in frame_id_target_label.items():
            frame_gnd_label = per_frame_gnd_labels_dict[device_id][frame_id]
            per_frame_pred_labels.append(frame_gnd_label)
            per_frame_target_labels.append(frame_target_label)

    # total_frame_acc = np.mean([p == g for p, g in zip(per_frame_pred_labels, per_frame_target_labels_dict)])
    total_frame_acc = np.mean([p == g for p, g in zip(per_frame_pred_labels, per_frame_target_labels)])
    print(f'total_frame_acc ({modality_type}): {total_frame_acc}')

    with open(os.path.join(fusion_stats_path, f'{modality_type}_stats.pkl'), "wb") as handler:
        pkl.dump(modality_stats, handler, protocol=pkl.HIGHEST_PROTOCOL)

    # Reading the pickle file
    with open(os.path.join(fusion_stats_path, f'{modality_type}_stats.pkl'), "rb") as handler:
        loaded_image_stats = pkl.load(handler)

    # frames_cm = confusion_matrix(per_frame_target, per_frame_pred, labels=class_names)
    videos_cm = confusion_matrix(y_true=per_vid_target, y_pred=per_vid_pred)
    save_cm_stats(videos_cm, classes=class_names, normalize=True,
                  title=f"{modality_type}_model_best_epoch_{best_epoch}_{total_vid_acc:.04f}_video",
                  save_dir=fusion_stats_path, figsize=(20, 20))
    save_cm_stats(videos_cm, classes=class_names, normalize=False,
                  title=f"{modality_type}_model_best_epoch_{best_epoch}_{total_vid_acc:.04f}_video",
                  save_dir=fusion_stats_path, figsize=(20, 20))

    frame_cm = confusion_matrix(y_true=per_frame_target_labels, y_pred=per_frame_pred_labels)
    save_cm_stats(frame_cm, classes=class_names, normalize=True,
                  title=f"{modality_type}_model_best_epoch_{best_epoch}_{total_frame_acc:.04f}_frame",
                  save_dir=fusion_stats_path, figsize=(20, 20))

    save_cm_stats(frame_cm, classes=class_names, normalize=False,
                  title=f"{modality_type}_model_best_epoch_{best_epoch}_{total_frame_acc:.04f}_frame",
                  save_dir=fusion_stats_path, figsize=(20, 20))


def testing(model, device, test_loader, log_softmax, softmax, modality_type):
    assert modality_type != 'audio' or modality_type != 'visual'

    video_stats = defaultdict(lambda: defaultdict(lambda: {
        "gnd_frame_label": None,
        "pred_frame_label": None,
        "frame_probabilities": []
    }))

    total_acc, total_cnt = 0, 0
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f'Predicting {modality_type}...'):

            if modality_type == 'visual':
                inputs = data[0].to(device)
            elif modality_type == 'audio':
                inputs = data[1].to(device)
            else:
                raise ValueError(f'modality_type should be audio or visual, {modality_type} is given')

            target_frame_labels = data[2].squeeze(1).to(device)
            device_names = data[3]
            audio_image_frame_indices = data[4]

            model_outputs = model(inputs)

            if softmax:
                pred_probs = model_outputs.data
                pred_proba, pred_frame_labels = torch.max(model_outputs, 1)
            elif log_softmax:
                pred_probs = torch.exp(model_outputs.data)
                pred_proba, pred_frame_labels = torch.max(torch.exp(model_outputs), 1)

            total_acc += torch.sum((pred_frame_labels == target_frame_labels).float()).item()
            total_cnt += target_frame_labels.size(0)

            target_frame_labels_np = target_frame_labels.detach().cpu().numpy()
            pred_frame_labels_np = pred_frame_labels.detach().cpu().numpy()
            pred_probs_np = pred_probs.detach().cpu().numpy()

            for device_name, pred_label, pred_prob, target_label, audio_frame_index in zip(device_names,
                                                                                           pred_frame_labels_np,
                                                                                           pred_probs_np,
                                                                                           target_frame_labels_np,
                                                                                           audio_image_frame_indices):
                # Append to gnd_frame_label, pred_frame_labels, and frame_probabilities lists
                video_stats[device_name][audio_frame_index]["gnd_frame_label"] = target_label.item()
                video_stats[device_name][audio_frame_index]["pred_frame_label"] = pred_label.item()
                video_stats[device_name][audio_frame_index]["frame_probabilities"] = pred_prob

                assert pred_label.item() == np.argmax(pred_prob)

    return video_stats


def main():
    print(f'Loading model audio model {args.audio_model}...')

    audio_model = get_model(args.audio_model, num_classes, log_softmax=args.log_softmax,
                            softmax=args.softmax, freeze_params=False, device=device)

    print(f'Loading model visual model {args.visual_model}...')
    visual_model = get_model(args.visual_model, num_classes, log_softmax=args.log_softmax,
                             softmax=args.softmax, freeze_params=False, device=device)

    # visual_model = get_model(args.visual_model)

    fusion_checkpoint = torch.load(os.path.join(fusion_run_path, "fusion_model_best.ckpt"))
    best_epoch = fusion_checkpoint['epoch']
    audio_state_dict = fusion_checkpoint['state_dict']['audio_model']
    audio_model.load_state_dict(audio_state_dict)
    print(f"Best audio model loaded epoch {best_epoch}")

    visual_state_dict = fusion_checkpoint['state_dict']['visual_model']
    visual_model.load_state_dict(visual_state_dict)
    print(f"Best visual model loaded epoch {best_epoch}")

    # Predicted results
    visual_stats = testing(visual_model, device, test_loader, args.log_softmax, args.softmax, 'visual')
    audio_stats = testing(audio_model, device, test_loader, args.log_softmax, args.softmax, 'audio')

    if args.save_stats:
        save_stats(visual_stats, best_epoch, 'visual')
        save_stats(audio_stats, best_epoch, 'audio')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir_fusion", type=str, required=True, default=None,
                        help='This a temporary variable that is changed to results_dir_audio or results_dir_visual')
    parser.add_argument("--results_dir_audio", type=str, required=True, default=None)
    parser.add_argument("--results_dir_visual", type=str, required=True, default=None)
    parser.add_argument("--audio_frame_indices_dir", type=str, required=True)

    parser.add_argument("--project_dir", type=str, required=True, default=None)
    parser.add_argument("--visual_frame_dir", type=str, required=True, default=None)
    parser.add_argument("--wav_dir", type=str, required=True)

    parser.add_argument("--data_loader_fold_type", type=str, required=True, default=None,
                        choices=['my_data_all', 'my_data_I'])
    parser.add_argument('--excluded_devices', default=['D12'], action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)])

    parser.add_argument("--cuda", default=0, type=int, help='Specify cuda number')
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--extend_duration_sec", default=None, type=int, required=True)

    parser.add_argument('--data_content', default=None,
                        choices=['YT', 'WA', 'Native'], required=True)

    parser.add_argument("--n_fold", type=int, required=True, default=None)

    parser.add_argument('--log_softmax', action='store_true', default=False)
    parser.add_argument('--softmax', action='store_true', default=False)

    parser.add_argument('--fft2', action='store_true', default=False)
    parser.add_argument('--prnu_enabled', action='store_true', default=False)

    parser.add_argument("--audio_model", default=None, type=str, required=True,
                        choices=["Audio1DDevIdentification", "MobileNetV3Small", "MobileNetV3Large",
                                 "SqueezeNet1_1", "ResNet18", "RawNet", "M5", "M11", "M18", "M34"])

    parser.add_argument("--visual_model", default=None, type=str, required=True,
                        choices=["DenseNet201", "ResNet50", "InceptionV3",
                                 "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"])

    parser.add_argument('--nfft_scale', type=int, default=None, help='The FFT scale')
    parser.add_argument('--log_mel', action='store_true', help="Use log Mel-Spectrogram")
    parser.add_argument('--n_mels', type=int, default=128, help='n_mels')

    parser.add_argument('--reduction', action='store_true', default=False)

    parser.add_argument("--n_run", type=int, required=True)

    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--valid_batch_size", default=64, type=int)

    parser.add_argument("--test_batch_size", default=64, type=int)

    parser.add_argument('--loss_fn', default=None,
                        choices=['SumRuleLoss', 'ProductRuleLoss'], required=True)

    parser.add_argument('--gaussian_noise_flag', action='store_true', default=False)

    parser.add_argument('--save_stats', action='store_true', default=False)

    args = parser.parse_args()

    assert not (args.log_softmax and args.softmax), 'log_softmax and softmax can not be enabled simultaneously'

    return args


def load_args_json(json_path):
    """Loads arguments from a JSON file into a namespace."""
    with open(os.path.join(json_path, 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
    return t_args


def update_args(args_dict, keys, source_dict):
    """Updates the main args dictionary with specific keys from the source."""
    for key in keys:
        if key in args_dict and key in source_dict:
            args_dict[key] = source_dict[key]


if __name__ == '__main__':
    """
--results_dir_fusion 
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/fusion/results/ 
--results_dir_audio 
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/ 
--results_dir_visual
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/image/results/ 
--audio_frame_indices_dir 
/media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/ 
--project_dir 
/media/red/tsingalis/gitRepositoriesCombiningDeepClassifiers/ 
--visual_frame_dir 
/media/red/sharedFolder/Datasets/VISION/keyFrames/I/ 
--wav_dir 
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/ 
--data_loader_fold_type 
my_data_I 
--extend_duration_sec 
2 
--data_content 
Native 
--n_fold 
0 
--reduction 
--test_batch_size 
128 
--n_run 
1 
--visual_model 
MobileNetV3Small 
--audio_model 
MobileNetV3Large 
--loss_fn 
ProductRuleLoss
    """

    args = get_args()

    # Paths for audio and visual JSON files
    fusion_run_path = os.path.join(args.results_dir_fusion, f'winSize{args.extend_duration_sec}sec',
                                   args.loss_fn,
                                   f'visual{args.visual_model}_audio{args.audio_model}',
                                   args.data_content, f"fold{args.n_fold}", f'run{args.n_run}')

    # Load audio and visual arguments
    fusion_args = load_args_json(fusion_run_path)

    # Update args with the relevant audio and visual values
    update_args(args.__dict__, list(args.__dict__.keys()), fusion_args.__dict__)

    print('Creating audio visual dataloader...')

    tst_pkl_name = f"test_audio_image_fold{args.n_fold}.pkl"

    data_content = f'{args.data_content}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else args.data_content

    audio_frame_indices_dir = os.path.join(args.audio_frame_indices_dir,
                                           f'winSize{args.extend_duration_sec}sec')

    pkl_dir_tst_fold = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                                    f'winSize{args.extend_duration_sec}sec', data_content, tst_pkl_name)

    test_set = ImageAudioDataset(pkl_fold_dir=pkl_dir_tst_fold,
                                 visual_frame_dir=args.visual_frame_dir,
                                 wav_dir=args.wav_dir,
                                 audio_frame_indices_dir=audio_frame_indices_dir,
                                 fft2_enabled=args.fft2, prnu_enabled=args.prnu_enabled,
                                 nfft_scale=args.nfft_scale,
                                 log_mel=args.log_mel, n_mels=args.n_mels,
                                 gaussian_noise_flag=False,
                                 center_crop=True,
                                 sample_prc=-0.1)
    print(f"Number of test audio visual samples {len(test_set)}")

    test_loader = DataLoader(test_set,
                             shuffle=False,
                             batch_size=args.test_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True)

    num_classes = test_set.n_classes

    assert num_classes == 34

    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from pathlib import Path

    fusion_stats_path = os.path.join(fusion_run_path, 'stats')

    Path(fusion_stats_path).mkdir(parents=True, exist_ok=True)

    class_names = ['D' + str(i + 1).zfill(2) for i in range(35)]
    class_names = [class_name for class_name in class_names if
                   args.excluded_devices is None or class_name not in args.excluded_devices]

    print_args(args)

    main()
