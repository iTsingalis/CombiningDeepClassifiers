import os
import json
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from general_utils import print_args
from general_utils import save_cm_stats
from collections import defaultdict
from models.audio_image_models import *
from audio.dataset import AudioDataset1D
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


def most_common(lst):
    return max(set(lst), key=lst.count)


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, required=False)
    parser.add_argument("--results_dir", type=str, required=False)

    # parameter #
    parser.add_argument("--test_batch", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--n_epoch", default=-1, type=int)

    parser.add_argument("--model", default=None, type=str,
                        choices=["Audio1DDevIdentification", "MobileNetV3Small", "MobileNetV3Large",
                                 "SqueezeNet1_1", "ResNet18", "RawNet", "M5", "M11", "M18", "M34"])

    parser.add_argument('--data_content',
                        choices=['YT', 'WA', 'Native'],
                        required=True)

    parser.add_argument("--data_loader_fold_type", type=str, required=True,
                        choices=['my_data_all', 'my_data_I'])

    parser.add_argument('--excluded_devices', default=None, action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)])

    parser.add_argument("--audio_frame_indices_dir", type=str, required=True)

    parser.add_argument("--extracted_wav_dir", type=str, required=True)

    parser.add_argument('--nfft_scale', type=int, default=1, help='The FFT scale')

    parser.add_argument("--n_fold", type=int, required=True)

    parser.add_argument("--n_run", type=int, required=False)

    parser.add_argument("--extend_duration_sec", type=int, required=True)

    parser.add_argument('--log_mel', action='store_true', help="Use log Mel-Spectrogram")

    parser.add_argument('--save_stats', action='store_true', default=False)

    args = parser.parse_args()

    json_path = os.path.join(args.results_dir, f'winSize{args.extend_duration_sec}sec',
                             args.data_content,
                             f"fold{args.n_fold}",
                             args.model,
                             f'run{args.n_run}')
    with open(os.path.join(json_path, 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    return args


def testing(model, device, data_loader, loss_fn, reduction, log_softmax, softmax):
    video_stats = defaultdict(lambda: defaultdict(lambda: {
        "gnd_frame_label": None,
        "pred_frame_label": None,
        "frame_probabilities": []
    }))

    test_loss, total_acc, total_cnt = 0, 0, 0
    pbar = tqdm(data_loader, disable=False)
    model.eval()
    with torch.no_grad():
        for data in pbar:
            pbar.set_description("Testing batch")
            batch_size = data[0].shape[0]

            if model.__class__.__name__ in ["Audio1DDevIdentification", "RawNet", "M5", "M11", "M18", "M34"]:
                inputs = data[0].reshape(batch_size, 1, -1).to(device)  # [batch, 3, 128, 1500]
            else:
                inputs = data[0].to(device)

            # inputs = data[0].reshape(batch_size, 1, -1).to(device)  # [batch, 3, 128, 1500]
            target_frame_labels = data[1].squeeze(1).to(device)
            device_names = data[2]
            audio_image_frame_indices = data[3]

            model_outputs = model(inputs)  # .squeeze()
            if loss_fn is not None:
                if softmax:
                    loss = loss_fn(torch.log(model_outputs), target_frame_labels)
                elif log_softmax:
                    loss = loss_fn(model_outputs, target_frame_labels)
                else:
                    raise ValueError('We work with log softmax or softmax')
                    loss = loss_fn(model_outputs, target_frame_labels)

                if reduction:
                    test_loss += loss.item() * batch_size  # Unscale the batch-averaged loss
                else:
                    test_loss += loss.item()

                # loss = loss_fn(model_outputs, target_frame_labels)
                # test_loss += loss.item()
            
            if softmax:
                pred_probs = model_outputs.data
                pred_proba, pred_frame_label = torch.max(model_outputs, 1)
            elif log_softmax:
                pred_probs = torch.exp(model_outputs.data)
                pred_proba, pred_frame_label = torch.max(torch.exp(model_outputs), 1)
            
            # pred_probs = torch.exp(model_outputs.data)
            # _, pred_frame_label = torch.max(torch.exp(model_outputs.data), 1)

            acc = torch.sum((pred_frame_label == target_frame_labels).float()).item()
            total_acc += acc
            total_cnt += batch_size

            target_labels_np = target_frame_labels.detach().cpu().numpy()
            pred_frame_labels_np = pred_frame_label.detach().cpu().numpy()
            pred_probs_np = pred_probs.detach().cpu().numpy()

            for device_name, pred_label, pred_prob, target_label, audio_frame_index in zip(device_names,
                                                                                           pred_frame_labels_np,
                                                                                           pred_probs_np,
                                                                                           target_labels_np,
                                                                                           audio_image_frame_indices):
                dev_id = int(le.transform([device_name[:3]]))
                assert target_label == dev_id
                video_stats[device_name][audio_frame_index]["gnd_frame_label"] = target_label.item()
                video_stats[device_name][audio_frame_index]["pred_frame_label"] = pred_label.item()
                video_stats[device_name][audio_frame_index]["frame_probabilities"] = pred_prob

                assert pred_label.item() == np.argmax(pred_prob)

    print(f"Test Loss: {test_loss / total_cnt:.3f}"
          f" Test Frame Acc: {100 * total_acc / total_cnt:.3f}%")

    return video_stats


def main():
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.model == "DenseNet201":
        model = DenseNet201(weights=models.DenseNet201_Weights.DEFAULT, num_classes=test_set.n_classes).to(device)
    elif args.model == "ResNet50":
        model = ResNet50(weights=models.ResNet50_Weights.DEFAULT, num_classes=test_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "InceptionV3":
        model = InceptionV3(weights=models.Inception_V3_Weights.DEFAULT, num_classes=test_set.n_classes).to(device)
    elif args.model == "ResNet18":
        model = ResNet18(weights=models.ResNet18_Weights.DEFAULT,
                         num_classes=test_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "MobileNetV3Small":
        model = MobileNetV3Small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,
                                 num_classes=test_set.n_classes,
                                 log_softmax=args.log_softmax,
                                 softmax=args.softmax).to(device)
    elif args.model == "MobileNetV3Large":
        model = MobileNetV3Large(weights=models.MobileNet_V3_Large_Weights.DEFAULT,
                                 num_classes=test_set.n_classes,
                                 log_softmax=args.log_softmax,
                                 softmax=args.softmax).to(device)
    elif args.model == "SqueezeNet1_1":
        model = SqueezeNet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT,
                              num_classes=test_set.n_classes,
                              log_softmax=args.log_softmax,
                              softmax=args.softmax).to(device)

    print(f'Model: {model.__class__.__name__}')

    if args.n_epoch > 0:
        checkpoint = torch.load(os.path.join(run_folder, f"audio_model_epoch{args.n_epoch}.ckpt"), map_location='cpu')
        print(f'Selected epoch {args.n_epoch}')
    else:
        checkpoint = torch.load(os.path.join(run_folder, "audio_model_best.ckpt"), map_location='cpu')
        print(f'Selected epoch {checkpoint["epoch"]}')

    model.load_state_dict(checkpoint['state_dict'])  # already is ['state_dict']
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model : %.3f M' % (pytorch_total_params / 1e6))

    if args.log_softmax or args.softmax:
        loss_fn = nn.NLLLoss(reduction='mean' if args.reduction else 'sum')
    else:
        raise ValueError('!!!!!!!!')
        # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        loss_fn = nn.CrossEntropyLoss(reduction='mean' if args.reduction else 'sum')

    audio_stats = testing(model, device, test_loader, loss_fn=loss_fn,
                          reduction=args.reduction,
                          log_softmax=args.log_softmax,
                          softmax=args.softmax)

    ###################################################################
    ## Compute Frame and Video Accuracy
    ###################################################################

    audio_stats = {k: defaultdict_to_dict(v) for k, v in audio_stats.items()}

    per_frame_target_labels_dict, per_frame_gnd_labels_dict = {}, {}

    for devices_name in test_loader.dataset.device_names:
        per_frame_target_labels_dict.update({devices_name: {}})
        per_frame_gnd_labels_dict.update({devices_name: {}})

    per_vid_predicted_labels_dicts, per_vid_target_labels_dicts = [], []
    for dev_key, frame_values in audio_stats.items():
        pred_frame_dict = dict([(frame_idx, frame_value["pred_frame_label"]) for
                                frame_idx, frame_value in frame_values.items()])

        gnd_frame_dict = dict([(frame_idx, frame_value["gnd_frame_label"]) for
                               frame_idx, frame_value in frame_values.items()])

        major_vid_label = most_common(list(pred_frame_dict.values()))
        gnd_vid_label = most_common(list(gnd_frame_dict.values()))

        per_vid_predicted_labels_dicts.append({dev_key: int(major_vid_label)})
        per_vid_target_labels_dicts.append({dev_key: int(gnd_vid_label)})

        for k, v in frame_values.items():
            per_frame_target_labels_dict[dev_key].update({k: v["pred_frame_label"]})
            per_frame_gnd_labels_dict[dev_key].update({k: v["gnd_frame_label"]})

    # Compare the values for each dictionary
    total_vid_acc = np.mean([
        all(p_val == g_val for p_val, g_val in zip(p.values(), g.values()))
        for p, g in zip(per_vid_predicted_labels_dicts, per_vid_target_labels_dicts)
    ])

    per_vid_target, per_vid_pred = zip(*[(p_val, g_val)
                                         for p, g in zip(per_vid_predicted_labels_dicts, per_vid_target_labels_dicts)
                                         for p_val, g_val in zip(p.values(), g.values())])

    print(f'total_video_acc (audio): {total_vid_acc}')

    per_frame_pred_labels, per_frame_target_labels = [], []
    for device_id, frame_id_target_label in per_frame_target_labels_dict.items():
        for frame_id, frame_target_label in frame_id_target_label.items():
            frame_gnd_label = per_frame_gnd_labels_dict[device_id][frame_id]
            per_frame_pred_labels.append(frame_gnd_label)
            per_frame_target_labels.append(frame_target_label)

    # total_frame_acc = np.mean([p == g for p, g in zip(per_frame_pred_labels, per_frame_target_labels_dict)])
    total_frame_acc = np.mean([p == g for p, g in zip(per_frame_pred_labels, per_frame_target_labels)])
    print(f'total_frame_acc (audio): {total_frame_acc}')

    if args.save_stats:
        with open(os.path.join(run_folder, 'audio_stats.pkl'), "wb") as handler:
            pkl.dump(audio_stats, handler, protocol=pkl.HIGHEST_PROTOCOL)

        # Reading the pickle file
        with open(os.path.join(run_folder, 'audio_stats.pkl'), "rb") as handler:
            loaded_audio_stats = pkl.load(handler)

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
--results_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/
--project_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers
--data_content
Native
--n_fold
0
--model
ResNet18
--data_loader_fold_type
my_data_I
--audio_frame_indices_dir
/media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/
--extracted_wav_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/
--extend_duration_sec
2
--test_batch
256
    """
    args = get_args()

    # results_dir = os.path.join(args.results_dir, f'winSize{args.extend_duration_sec}sec',
    #                            args.data_content, f"fold{args.n_fold}", args.model)

    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_content = f'{args.data_content}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else args.data_content

    print('Creating testing dataloader...')
    pkl_name = f"test_audio_image_fold{args.n_fold}.pkl"

    pkl_dir_set = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                               f'winSize{args.extend_duration_sec}sec', data_content, pkl_name)

    test_set = AudioDataset1D(pkl_dir_set,
                              audio_frame_indices_dir=os.path.join(args.audio_frame_indices_dir,
                                                                   f'winSize{args.extend_duration_sec}sec'),
                              extracted_wav_dir=args.extracted_wav_dir,
                              sample_prc=-.001,
                              nfft_scale=args.nfft_scale,
                              log_mel=args.log_mel,
                              n_mels=args.n_mels)
    print(f"Number of Test samples {len(test_set)}")

    # print('Number of Test samples : %.3f M' % (len(test_set) / 1e6))

    test_loader = DataLoader(test_set,
                             shuffle=True,
                             batch_size=args.test_batch,
                             num_workers=args.num_workers,
                             pin_memory=True)

    class_names = ['D' + str(i + 1).zfill(2) for i in range(35)]
    class_names = [class_name for class_name in class_names if
                   args.excluded_devices is None or class_name not in args.excluded_devices]

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(class_names)

    run_folder = os.path.join(args.results_dir, f'winSize{args.extend_duration_sec}sec',
                              args.data_content, f"fold{args.n_fold}", args.model, f'run{args.n_run}')
    print_args(args)

    main()
