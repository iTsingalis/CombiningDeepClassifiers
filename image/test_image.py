import os
import json
import argparse
import numpy as np
import pickle as pkl
from tqdm.auto import tqdm
from general_utils import save_cm_stats
from models.audio_image_models import *
from image.dataset import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from collections import defaultdict
from general_utils import print_args


def most_common(lst):
    return max(set(lst), key=lst.count)


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=False)
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--visual_frames_dir", type=str, required=True)
    parser.add_argument("--data_loader_fold_type", type=str, required=True,
                        choices=['my_data_I', 'my_data_all'])
    # parameter #
    parser.add_argument("--cuda", default=0, type=int, help='Specify cuda number')
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--train_batch", default=8, type=int)
    parser.add_argument("--test_batch", default=8, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=int)

    parser.add_argument("--extend_duration_sec", type=int, required=True)

    parser.add_argument('--excluded_devices', default=None, action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)])

    parser.add_argument("--model", default=None, type=str,
                        choices=["DenseNet201", "ResNet50", "InceptionV3", "ResNet101",
                                 "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"])

    parser.add_argument('--data_content',
                        choices=['YT', 'WA', 'Native'], required=False)

    parser.add_argument("--n_fold", type=int, required=False)

    parser.add_argument("--n_run", type=int, required=True)

    parser.add_argument('--fft2', action='store_true', default=False)
    parser.add_argument('--prnu_enabled', action='store_true', default=False)

    parser.add_argument('--log_softmax', action='store_true', default=False)
    parser.add_argument('--softmax', action='store_true', default=False)

    args = parser.parse_args()

    with open(os.path.join(args.results_dir,
                           f'winSize{args.extend_duration_sec}sec',
                           args.data_content,
                           f"fold{args.n_fold}",
                           args.model,
                           f'run{args.n_run}', 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    assert not (args.log_softmax and args.softmax), 'log_softmax and softmax can not be enabled simultaneously'

    return args


def testing(model, device, test_loader, log_softmax, softmax):
    video_stats = defaultdict(lambda: defaultdict(lambda: {
        "gnd_frame_label": None,
        "pred_frame_label": None,
        "frame_probabilities": []
    }))

    train_loss, total_acc, total_cnt = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Predicting...'):
            inputs = data[0].to(device)
            target_frame_labels = data[1].squeeze(1).to(device)
            device_names = data[2]
            audio_image_frame_indices = data[3]

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

    if args.model == "DenseNet201":
        model = DenseNet201(num_classes=num_classes, log_softmax=args.log_softmax).to(device)
    elif args.model == "ResNet50":
        model = ResNet50(num_classes=num_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "ResNet101":
        model = ResNet101(weights=models.ResNet101_Weights.DEFAULT,
                          num_classes=num_classes,
                          log_softmax=args.log_softmax,
                          softmax=args.softmax).to(device)
    elif args.model == "InceptionV3":
        model = InceptionV3(weights=models.Inception_V3_Weights.DEFAULT,
                            num_classes=num_classes).to(device)
    elif args.model == "ResNet18":
        model = ResNet18(weights=models.ResNet18_Weights.DEFAULT,
                         num_classes=num_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "MobileNetV3Small":
        model = MobileNetV3Small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,
                                 num_classes=num_classes,
                                 log_softmax=args.log_softmax,
                                 softmax=args.softmax).to(device)
    elif args.model == "MobileNetV3Large":
        model = MobileNetV3Large(weights=models.MobileNet_V3_Large_Weights.DEFAULT,
                                 num_classes=num_classes,
                                 log_softmax=args.log_softmax,
                                 softmax=args.softmax).to(device)
    elif args.model == "SqueezeNet1_1":
        model = SqueezeNet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT,
                              num_classes=num_classes,
                              log_softmax=args.log_softmax,
                              softmax=args.softmax).to(device)
    print(f'Image Model: {model.__class__.__name__}')

    checkpoint = torch.load(os.path.join(str(args.results_dir),
                                         f'winSize{args.extend_duration_sec}sec',
                                         args.data_content,
                                         f"fold{args.n_fold}",
                                         args.model,
                                         f"run{args.n_run}",
                                         "visual_model_best.ckpt"),
                            map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])  # already is ['state_dict']
    model = model.to(device)
    print(f"Retrieved epoch (image): {checkpoint['epoch']}")
    # Predicted result
    image_stats = testing(model, device, test_loader, args.log_softmax, args.softmax)

    image_stats = {k: defaultdict_to_dict(v) for k, v in image_stats.items()}

    per_frame_target_labels_dict, per_frame_gnd_labels_dict = {}, {}

    for devices_name in test_loader.dataset.device_names:
        per_frame_target_labels_dict.update({devices_name: {}})
        per_frame_gnd_labels_dict.update({devices_name: {}})

    per_vid_predicted_labels_dicts, per_vid_target_labels_dicts = [], []
    for dev_key, frame_values in image_stats.items():
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

    print(f'total_video_acc (image): {total_vid_acc}')

    per_frame_pred_labels, per_frame_target_labels = [], []
    for device_id, frame_id_target_label in per_frame_target_labels_dict.items():
        for frame_id, frame_target_label in frame_id_target_label.items():
            frame_gnd_label = per_frame_gnd_labels_dict[device_id][frame_id]
            per_frame_pred_labels.append(frame_gnd_label)
            per_frame_target_labels.append(frame_target_label)

    # total_frame_acc = np.mean([p == g for p, g in zip(per_frame_pred_labels, per_frame_target_labels_dict)])
    total_frame_acc = np.mean([p == g for p, g in zip(per_frame_pred_labels, per_frame_target_labels)])
    print(f'total_frame_acc (image): {total_frame_acc}')

    with open(os.path.join(run_folder, 'image_stats.pkl'), "wb") as handler:
        pkl.dump(image_stats, handler, protocol=pkl.HIGHEST_PROTOCOL)

    # Reading the pickle file
    with open(os.path.join(run_folder, 'image_stats.pkl'), "rb") as handler:
        loaded_image_stats = pkl.load(handler)

    # frames_cm = confusion_matrix(per_frame_target, per_frame_pred, labels=class_names)
    videos_cm = confusion_matrix(y_true=per_vid_target, y_pred=per_vid_pred)
    save_cm_stats(videos_cm, classes=class_names, normalize=True,
                  # title=f"model_epoch{args.n_epoch}_{total_vid_acc:.04f}"
                  title=f"model_best_epoch_{checkpoint['epoch']}_{total_vid_acc:.04f}_video",
                  # if args.n_epoch > 0 else f"best_model_epoch_{checkpoint['epoch']}_{total_vid_acc:.04f}",
                  #       f"best_model_epoch_{checkpoint['epoch']}_{total_vid_acc:.04f}",
                  save_dir=run_folder, figsize=(20, 20))
    save_cm_stats(videos_cm, classes=class_names, normalize=False,
                  title=f"model_best_epoch_{checkpoint['epoch']}_{total_vid_acc:.04f}_video",
                  save_dir=run_folder, figsize=(20, 20))

    frame_cm = confusion_matrix(y_true=per_frame_target_labels, y_pred=per_frame_pred_labels)
    save_cm_stats(frame_cm, classes=class_names, normalize=True,
                  # title=f"model_epoch{args.n_epoch}_{total_frame_acc:.04f}"
                  title=f"model_best_epoch_{checkpoint['epoch']}_{total_frame_acc:.04f}_frame",
                  # if args.n_epoch > 0 else f"best_model_epoch_{checkpoint['epoch']}_{total_frame_acc:.04f}_frame",
                  #       f"best_model_epoch_{checkpoint['epoch']}_{total_frame_acc:.04f}_frame",
                  save_dir=run_folder, figsize=(20, 20))

    save_cm_stats(frame_cm, classes=class_names, normalize=False,
                  title=f"model_best_epoch_{checkpoint['epoch']}_{total_frame_acc:.04f}_frame",
                  save_dir=run_folder, figsize=(20, 20))


if __name__ == '__main__':

    """
--n_run
1
--data_content
WA
--model
ResNet50
--n_fold
1
--visual_frames_dir
/media/red/sharedFolder/Datasets/VISION/keyFrames/I/
--project_dir
/media/blue/tsingalis/DevIDFusion/
--results_dir
/media/blue/tsingalis/DevIDFusion/image/results/
--data_loader_fold_type
my_data_I
--extend_duration_sec
2
--test_batch
128
    """
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    class_names = ['D' + str(i + 1).zfill(2) for i in range(35)]
    class_names = [class_name for class_name in class_names if
                   args.excluded_devices is None or class_name not in args.excluded_devices]

    num_classes = len(class_names)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(class_names)

    print('Creating testing dataloader...')
    if args.data_loader_fold_type == 'SNComputerScience':
        pkl_name = os.path.join(f"{args.data_content}_{'_'.join(args.excluded_devices)}_Excluded", "train_images.pkl")
    else:
        pkl_name = f"test_audio_image_fold{args.n_fold}.pkl"

    # pkl_dir_tr = f"{args.project_dir}/preprocessed_images/test_128images_{args.data_content}_fold{args.n_fold}.pkl"
    data_content = f'{args.data_content}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else args.data_content

    pkl_dir_tst = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                               f'winSize{args.extend_duration_sec}sec', data_content, pkl_name)

    test_set = ImageDataset(pkl_dir_tst, args.visual_frames_dir,
                            fft2=args.fft2,
                            prnu_enabled=args.prnu_enabled,
                            gaussian_noise_flag=False,
                            center_crop=True,
                            sample_prc=-0.001)
    print(f"Number of Test samples {len(test_set)}")

    test_loader = DataLoader(test_set, batch_size=args.test_batch, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers)

    run_folder = os.path.join(args.results_dir,
                              f'winSize{args.extend_duration_sec}sec',
                              args.data_content, f"fold{args.n_fold}", args.model, f'run{args.n_run}')

    print_args(args)

    main()
