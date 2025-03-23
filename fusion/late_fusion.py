import os
import torch
import argparse
import numpy as np
import pickle as pkl
from general_utils import print_args
from general_utils import save_cm_stats
from sklearn.metrics import confusion_matrix


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--n_run_image_dir", type=str, required=False)
    # parser.add_argument("--n_run_audio_dir", type=str, required=False)
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--data_loader_fold_type", type=str, required=True,
                        choices=['my_data_all', 'my_data_I'])
    parser.add_argument('--data_content', choices=['YT', 'WA', 'Native'], required=True)
    parser.add_argument("--n_fold", type=int, required=True)
    parser.add_argument("--extend_duration_sec", type=int, required=True)
    parser.add_argument("--n_run_audio", type=int, required=True)
    parser.add_argument("--n_run_visual", type=int, required=True)
    parser.add_argument('--excluded_devices', default=None, action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)], required=True)

    parser.add_argument("--audio_model", default=None, type=str,
                        choices=["Audio1DDevIdentification", "MobileNetV3Small", "MobileNetV3Large",
                                 "SqueezeNet1_1", "ResNet18", "RawNet", "M5", "M11", "M18", "M34"])

    parser.add_argument("--visual_model", default=None, type=str, required=True,
                        choices=["DenseNet201", "ResNet50", "InceptionV3",
                                 "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"])

    parser.add_argument("--fusion_rule", default=None, type=str, required=True,
                        choices=["product_rule", "sum_rule", "max_rule"])

    args = parser.parse_args()

    return args


def most_common(lst):
    return max(set(lst), key=lst.count)


def compute_video_frame_acc(audio_or_image_stats, device_sample_names):
    # Compute Frame and Video Accuracy

    per_frame_pred_labels_dict, per_frame_gnd_labels_dict = {}, {}
    for devices_name in device_sample_names:
        per_frame_pred_labels_dict.update({devices_name: {}})
        per_frame_gnd_labels_dict.update({devices_name: {}})

    per_vid_predicted_labels_dicts, per_vid_target_labels_dicts = {}, {}
    for dev_sample_id, frame_values in audio_or_image_stats.items():
        pred_frame_dict = dict([(frame_idx, frame_value["pred_frame_label"]) for
                                frame_idx, frame_value in frame_values.items()])

        gnd_frame_dict = dict([(frame_idx, frame_value["gnd_frame_label"]) for
                               frame_idx, frame_value in frame_values.items()])

        major_vid_label = most_common(list(pred_frame_dict.values()))
        gnd_vid_label = most_common(list(gnd_frame_dict.values()))

        per_vid_predicted_labels_dicts.update({dev_sample_id: int(major_vid_label)})
        per_vid_target_labels_dicts.update({dev_sample_id: int(gnd_vid_label)})

        # Update dictionaries and collect labels in one go
        _per_frame_pred_labels = [
            v["pred_frame_label"]
            for k, v in frame_values.items()
        ]

        _per_frame_gnd_labels = [
            v["gnd_frame_label"]
            for k, v in frame_values.items()
        ]

        # Update dictionaries using dictionary comprehensions
        for k, v in frame_values.items():
            per_frame_pred_labels_dict[dev_sample_id][k] = v["pred_frame_label"]
            per_frame_gnd_labels_dict[dev_sample_id][k] = v["gnd_frame_label"]

    per_vid_visual_gnd_labels = [per_vid_target_labels_dicts[key] for key in per_vid_predicted_labels_dicts]
    per_vid_visual_pred_labels = [per_vid_predicted_labels_dicts[key] for key in per_vid_predicted_labels_dicts]

    per_frame_visual_gnd_labels = [
        per_frame_gnd_labels_dict[device_id][frame_id]
        for device_id, frame_id_target_label in per_frame_pred_labels_dict.items()
        for frame_id in frame_id_target_label
    ]

    per_frame_visual_pred_labels = [
        frame_target_label
        for frame_id_target_label in per_frame_pred_labels_dict.values()
        for frame_target_label in frame_id_target_label.values()
    ]

    per_vid_audio_acc = per_vid_visual_gnd_labels, per_vid_visual_pred_labels
    per_frame_audio_acc = per_frame_visual_gnd_labels, per_frame_visual_pred_labels
    return per_vid_audio_acc, per_frame_audio_acc


def main():
    args = get_args()

    # Number of modalities M (see paper)
    M = 2

    data_content = f'{args.data_content}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else args.data_content

    priors_path = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                               f'winSize{args.extend_duration_sec}sec',
                               data_content)
    print(f'Priors Path: {priors_path}')

    with open(os.path.join(priors_path, f'train_audio_image_priors_fold{args.n_fold}.pkl'), "rb") as handler:
        train_audio_image_priors = dict(sorted(pkl.load(handler).items()))

    train_audio_image_priors = np.array(list(train_audio_image_priors.values()))
    if args.fusion_rule == 'product_rule':
        train_audio_image_priors = train_audio_image_priors ** (1 - M)
    elif args.fusion_rule == 'sum_rule':
        train_audio_image_priors = (1 - M) * train_audio_image_priors
    elif args.fusion_rule == 'sum_rule' or args.fusion_rule == 'max_rule':
        train_audio_image_priors = (1 - M) * train_audio_image_priors
    # Load audio stats
    n_run_audio_dir = os.path.join(args.project_dir, 'audio', 'results1D', f'winSize{args.extend_duration_sec}sec',
                                   args.data_content, f"fold{args.n_fold}", args.audio_model, f'run{args.n_run_audio}')
    print(f'Run audio dir: {n_run_audio_dir}')

    with open(os.path.join(n_run_audio_dir, 'audio_stats.pkl'), "rb") as f:
        audio_stats = pkl.load(f)

    audio_checkpoint = torch.load(os.path.join(n_run_audio_dir, "audio_model_best.ckpt"), map_location='cpu')
    audio_best_val_epoch = audio_checkpoint['epoch']
    device_sample_names = list(audio_stats.keys())

    # Initialize dictionaries with empty dictionaries for each device name
    # per_audio_frame_probs_dict = {device_name: {} for device_name in device_sample_names}
    per_image_frame_probs_dict = {device_name: {} for device_name in device_sample_names}

    # per_audio_frame_pred_dict = {device_name: {} for device_name in device_sample_names}
    per_image_frame_pred_dict = {device_name: {} for device_name in device_sample_names}

    per_image_frame_gnd_dict = {device_name: {} for device_name in device_sample_names}
    # per_audio_frame_gnd_dict = {device_name: {} for device_name in device_sample_names}

    per_frame_prod_rule_pred_dict = {device_name: {} for device_name in device_sample_names}
    per_vid_prod_rule_pred_dict = {device_name: {} for device_name in device_sample_names}

    # Validate that the ground truth labels match the transformed device ID
    # all(v["gnd_frame_label"] == int(le.transform([dev_sample_id[:3]]))
    #     for dev_sample_id, frame_values in audio_stats.items()
    #     for v in frame_values.values())

    assert all(
        v["gnd_frame_label"] == dev_id
        for dev_sample_id, frame_values in audio_stats.items()
        for dev_id in [int(le.transform([dev_sample_id[:3]]))]
        for v in frame_values.values()
    )
    # Populate the dictionaries using comprehensions
    per_audio_frame_probs_dict = {
        dev_sample_id: {k: (v["frame_probabilities"]) for k, v in frame_values.items()}
        for dev_sample_id, frame_values in audio_stats.items()
    }
    assert all([np.allclose(np.sum(list(per_audio_frame_probs_dict[dev_sample_id].values()), axis=1), 1)
                for dev_sample_id, frame_values in audio_stats.items()])

    per_audio_frame_pred_dict = {
        dev_sample_id: {k: v["pred_frame_label"] for k, v in frame_values.items()}
        for dev_sample_id, frame_values in audio_stats.items()
    }

    per_audio_frame_gnd_dict = {
        dev_sample_id: {k: v["gnd_frame_label"] for k, v in frame_values.items()}
        for dev_sample_id, frame_values in audio_stats.items()
    }

    assert all([len(set(per_audio_frame_gnd_dict[d].values())) == 1 for d in device_sample_names])

    n_run_image_dir = os.path.join(args.project_dir, 'image', 'results', f'winSize{args.extend_duration_sec}sec',
                                   args.data_content, f"fold{args.n_fold}",
                                   args.visual_model, f'run{args.n_run_visual}')
    print(f'Run image dir: {n_run_image_dir}')

    with open(os.path.join(n_run_image_dir, 'image_stats.pkl'), "rb") as f:
        image_stats = pkl.load(f)

    image_checkpoint = torch.load(os.path.join(n_run_image_dir, "visual_model_best.ckpt"), map_location='cpu')
    visual_best_val_epoch = image_checkpoint['epoch']

    per_image_frame_probs_dict = {
        dev_sample_id: {k: v["frame_probabilities"] for k, v in frame_values.items()}
        for dev_sample_id, frame_values in image_stats.items()
    }
    assert all([np.allclose(np.sum(list(per_image_frame_probs_dict[dev_sample_id].values()), axis=1), 1)
                for dev_sample_id, frame_values in image_stats.items()])

    per_image_frame_gnd_dict = {
        dev_sample_id: {k: v["gnd_frame_label"] for k, v in frame_values.items()}
        for dev_sample_id, frame_values in image_stats.items()
    }

    per_image_frame_pred_dict = {
        dev_sample_id: {k: v["pred_frame_label"] for k, v in frame_values.items()}
        for dev_sample_id, frame_values in image_stats.items()
    }

    assert all(
        v["gnd_frame_label"] == dev_id
        for dev_sample_id, frame_values in image_stats.items()
        for dev_id in [int(le.transform([dev_sample_id[:3]]))]
        for v in frame_values.values()
    )

    assert all([len(set(per_image_frame_gnd_dict[d].values())) == 1 for d in device_sample_names])

    plt_cm = False

    ## Compute Audio Accuracy results
    per_vid_audio_acc, per_frame_audio_acc = compute_video_frame_acc(audio_stats, device_sample_names)

    per_vid_audio_gnd_labels, per_vid_audio_pred_labels = per_vid_audio_acc
    per_frame_audio_gnd_labels, per_frame_audio_pred_labels = per_frame_audio_acc

    total_audio_vid_acc = np.mean([p == g for p, g in zip(per_vid_audio_pred_labels, per_vid_audio_gnd_labels)])
    total_audio_frame_acc = np.mean([p == g for p, g in zip(per_frame_audio_pred_labels, per_frame_audio_gnd_labels)])
    print(f'total_audio_vid_acc: {total_audio_vid_acc} -- total_audio_frame_acc:{total_audio_frame_acc}')

    if plt_cm:
        videos_audio_cm = confusion_matrix(y_true=per_vid_audio_gnd_labels, y_pred=per_vid_audio_pred_labels,
                                           labels=range(0, 34))
        save_cm_stats(videos_audio_cm, classes=class_names, normalize=False,
                      title=f"model_best_epoch_{audio_best_val_epoch}_{total_audio_vid_acc:.04f}_audio_vid",
                      save_dir=None, figsize=(20, 20))

        videos_audio_cm = confusion_matrix(y_true=per_frame_audio_gnd_labels, y_pred=per_frame_audio_pred_labels,
                                           labels=range(0, 34))
        save_cm_stats(videos_audio_cm, classes=class_names, normalize=False,
                      title=f"model_best_epoch_{audio_best_val_epoch}_{total_audio_frame_acc:.04f}_audio_frame",
                      save_dir=None, figsize=(20, 20))

    ## Compute Visual Accuracy results
    per_vid_visual_acc, per_frame_visual_acc = compute_video_frame_acc(image_stats, device_sample_names)

    per_vid_visual_gnd_labels, per_vid_visual_pred_labels = per_vid_visual_acc
    per_frame_visual_gnd_labels, per_frame_visual_pred_labels = per_frame_visual_acc

    total_visual_vid_acc = np.mean([p == g for p, g in zip(per_vid_visual_pred_labels, per_vid_visual_gnd_labels)])
    total_visual_frame_acc = np.mean(
        [p == g for p, g in zip(per_frame_visual_pred_labels, per_frame_visual_gnd_labels)])
    print(f'total_visual_vid_acc: {total_visual_vid_acc} -- total_visual_frame_acc:{total_visual_frame_acc}')

    if plt_cm:
        videos_visual_cm = confusion_matrix(y_true=per_vid_visual_gnd_labels,
                                            y_pred=per_vid_visual_pred_labels, labels=range(0, 34))
        save_cm_stats(videos_visual_cm, classes=class_names, normalize=False,
                      title=f"model_best_epoch_{visual_best_val_epoch}_{total_visual_vid_acc:.04f}_visual_vid",
                      save_dir=None, figsize=(20, 20))

        videos_visual_cm = confusion_matrix(y_true=per_frame_visual_gnd_labels,
                                            y_pred=per_frame_visual_pred_labels, labels=range(0, 34))
        save_cm_stats(videos_visual_cm, classes=class_names, normalize=False,
                      title=f"model_best_epoch_{visual_best_val_epoch}_{total_visual_frame_acc:.04f}_visual_frame",
                      save_dir=None, figsize=(20, 20))

    # Compute Product rule probabilities
    for dev_sample_id in device_sample_names:
        assert per_audio_frame_probs_dict[dev_sample_id].keys() == per_image_frame_probs_dict[dev_sample_id].keys()
        per_audio_frame_probs_dict[dev_sample_id] = dict(sorted(per_audio_frame_probs_dict[dev_sample_id].items()))
        per_image_frame_probs_dict[dev_sample_id] = dict(sorted(per_image_frame_probs_dict[dev_sample_id].items()))

        per_audio_frame_pred_dict[dev_sample_id] = dict(sorted(per_audio_frame_pred_dict[dev_sample_id].items()))
        per_image_frame_pred_dict[dev_sample_id] = dict(sorted(per_image_frame_pred_dict[dev_sample_id].items()))

        per_audio_frame_probs = np.array(list(per_audio_frame_probs_dict[dev_sample_id].values()))
        per_audio_frame_keys = list(per_audio_frame_probs_dict[dev_sample_id].keys())
        # per_audio_frame_probs /= train_audio_image_priors

        per_image_frame_probs = np.array(list(per_image_frame_probs_dict[dev_sample_id].values()))
        # per_image_frame_probs /= train_audio_image_priors

        if args.fusion_rule == 'product_rule':
            # Element wise multiplication
            prod_rule = per_audio_frame_probs * per_image_frame_probs
            prod_rule *= train_audio_image_priors
        elif args.fusion_rule == 'sum_rule':
            # Element wise multiplication
            prod_rule = per_audio_frame_probs + per_image_frame_probs
            prod_rule += train_audio_image_priors
        elif args.fusion_rule == 'max_rule':
            prod_rule = np.maximum(per_audio_frame_probs, per_image_frame_probs)
            prod_rule += train_audio_image_priors

        y_pred_prod_rule = np.argmax(prod_rule, axis=1)
        per_frame_prod_rule_pred_dict[dev_sample_id] = dict(zip(per_audio_frame_keys, y_pred_prod_rule))
        per_vid_prod_rule_pred_dict[dev_sample_id] = most_common(list(y_pred_prod_rule))

    per_frame_visual_gnd_labels = [
        per_image_frame_gnd_dict[device_id][frame_id]
        for device_id in device_sample_names
        for frame_id in per_image_frame_gnd_dict[device_id]
    ]
    per_frame_audio_gnd_labels = [
        per_audio_frame_gnd_dict[device_id][frame_id]
        for device_id in device_sample_names
        for frame_id in per_audio_frame_gnd_dict[device_id]
    ]
    # Just checking
    assert all([g == p for g, p in zip(per_frame_visual_gnd_labels, per_frame_audio_gnd_labels)])

    per_frame_prod_pred_labels = [
        per_frame_prod_rule_pred_dict[device_id][frame_id]
        for device_id, frame_id_target_label in per_frame_prod_rule_pred_dict.items()
        for frame_id in frame_id_target_label
    ]

    per_vid_prod_pred_labels, per_vid_gnd_labels = [], []
    for dev_sample_id in device_sample_names:
        per_vid_prod_pred_labels.append(per_vid_prod_rule_pred_dict[dev_sample_id])
        per_vid_gnd_labels.append(set(np.array(list(per_image_frame_gnd_dict[dev_sample_id].values()))).pop())

    total_prod_vid_acc = np.mean([p == g for p, g in zip(per_vid_prod_pred_labels, per_vid_gnd_labels)])
    total_prod_frame_acc = np.mean([p == g for p, g in zip(per_frame_prod_pred_labels, per_frame_audio_gnd_labels)])
    print(f'total_prod_vid_acc: {total_prod_vid_acc} -- total_prod_frame_acc:{total_prod_frame_acc}')

    if plt_cm:
        videos_prod_cm = confusion_matrix(y_true=per_vid_gnd_labels,
                                          y_pred=per_vid_prod_pred_labels, labels=range(0, 34))
        save_cm_stats(videos_prod_cm, classes=class_names, normalize=False,
                      title=f"model_best_epoch_{visual_best_val_epoch}_{total_prod_vid_acc:.04f}_prod_vid",
                      save_dir=None, figsize=(20, 20))

        frames_prod_cm = confusion_matrix(y_true=per_frame_audio_gnd_labels,
                                          y_pred=per_frame_prod_pred_labels, labels=range(0, 34))
        save_cm_stats(frames_prod_cm, classes=class_names, normalize=False,
                      title=f"model_best_epoch_{visual_best_val_epoch}_{total_prod_frame_acc:.04f}_prod_frame",
                      save_dir=None, figsize=(20, 20))


if __name__ == '__main__':
    """
--project_dir 
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/ 
--data_loader_fold_type 
my_data_I 
--extend_duration_sec 
2 
--n_run_audio 
1 
--n_run_visual 
1 
--excluded_devices 
D12 
--data_content 
YT 
--n_fold 
4 
--visual_model 
MobileNetV3Small 
--audio_model 
MobileNetV3Large 
--fusion_rule 
product_rule
    """
    args = get_args()

    excluded_devices = ['D12']
    class_names = ['D' + str(i + 1).zfill(2) for i in range(35)]
    class_names = [class_name for class_name in class_names if
                   excluded_devices is None or class_name not in excluded_devices]

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(class_names)

    print(class_names)

    print_args(args)

    main()
