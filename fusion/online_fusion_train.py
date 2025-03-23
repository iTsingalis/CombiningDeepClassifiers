import os

import json
import argparse

import numpy as np
import pickle as pkl

from tqdm import tqdm
from general_utils import print_args
from tensorboardX import SummaryWriter
from models.audio_image_models import *
from torch.utils.data import DataLoader
from fusion.dataset import ImageAudioDataset
from general_utils import check_run_folder, EarlyStoppingWithMovingAverage, save_checkpoint, CustomReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def pad_image(image, target_size):
    """
    Pads an image tensor to the target size with zeros.
    Args:
        image (Tensor): Image tensor of shape (3, H, W).
        target_size (tuple): Target size (H_max, W_max).
    Returns:
        Padded image tensor of shape (3, H_max, W_max).
    """
    _, h, w = image.shape
    pad_h = target_size[0] - h
    pad_w = target_size[1] - w

    # Padding format: (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    return torch.nn.functional.pad(image, padding)


def custom_collate(batch):
    """
    Custom collate function for handling batches with variable-sized images (visual_values)
    and other fields returned by the dataset.
    """
    visual_values, audio_values, targets, device_names, frame_indices = zip(*batch)

    # Determine the max height and width in the batch
    max_h = max(img.shape[1] for img in visual_values)
    max_w = max(img.shape[2] for img in visual_values)

    # Pad images to the maximum size
    padded_visual_values = torch.stack([
        pad_image(img, (max_h, max_w))
        for img in visual_values
    ])

    # Convert `audio_values` to tensor
    audio_values = torch.stack(audio_values)

    # Convert `targets` to tensor
    targets = torch.tensor(targets, dtype=torch.long).view(-1, 1)

    return (
        padded_visual_values,
        audio_values,
        targets,
        list(device_names),
        list(frame_indices)
    )


# https://github.com/elinorwahl/pytorch-image-classifier/blob/master/predict.py
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

    parser.add_argument("--data_loader_fold_type", type=str, required=False, default=None,
                        choices=['my_data_all', 'my_data_I'])
    parser.add_argument('--excluded_devices', default=None, action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)])

    parser.add_argument("--cuda", default=0, type=int, help='Specify cuda number')
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--valid_batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument("--audio_lr", default=1e-5, type=float)
    parser.add_argument("--visual_lr", default=1e-4, type=float)

    parser.add_argument("--adam_weight_decay", default=0, type=float)
    parser.add_argument("--extend_duration_sec", default=None, type=int, required=True)
    parser.add_argument("--window_size", default=2, type=int, required=False)

    parser.add_argument('--optimizer', default=None,
                        choices=['SGD', 'Adam', 'AdaCubic'],
                        required=False)

    parser.add_argument('--lr_scheduler', default='MultiStepLR',
                        choices=['CosineAnnealingLR', 'MultiStepLR'],
                        required=False)

    parser.add_argument('--data_content', default=None,
                        choices=['YT', 'WA', 'Native'], required=True)

    parser.add_argument('--loss_fn', default=None,
                        choices=['SumRuleLoss', 'ProductRuleLoss'], required=True)

    parser.add_argument("--label_smoothing", type=float, required=True, default=0.0)

    parser.add_argument("--n_fold", type=int, required=True, default=None)

    parser.add_argument("--n_run_audio", type=int, required=True, default=None)
    parser.add_argument("--n_run_visual", type=int, required=True, default=None)

    parser.add_argument('--log_softmax', action='store_true', default=False)
    parser.add_argument('--softmax', action='store_true', default=False)
    parser.add_argument('--pre_train', action='store_true', default=False)

    parser.add_argument('--fft2', action='store_true', default=False)
    parser.add_argument('--prnu_enabled', action='store_true', default=False)

    parser.add_argument("--audio_model", default=None, type=str,
                        choices=["Audio1DDevIdentification", "MobileNetV3Small", "MobileNetV3Large",
                                 "SqueezeNet1_1", "ResNet18", "RawNet", "M5", "M11", "M18", "M34"])

    parser.add_argument("--visual_model", default=None, type=str, required=True,
                        choices=["DenseNet201", "ResNet50", "InceptionV3",
                                 "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"])

    parser.add_argument('--nfft_scale', type=int, default=None, help='The FFT scale')
    parser.add_argument('--log_mel', action='store_true', help="Use log Mel-Spectrogram")
    parser.add_argument('--n_mels', type=int, default=128, help='n_mels')

    parser.add_argument('--milestones', default=None, action='append')

    parser.add_argument('--reduction', action='store_true', default=False)
    parser.add_argument('--disable_priors', action='store_true', default=False)

    parser.add_argument('--freeze_params', action='store_true', default=False)

    parser.add_argument('--gaussian_noise_flag', action='store_true', default=False)

    parser.add_argument('--save_stats', action='store_true', default=False)

    args = parser.parse_args()

    # if args.loss_fn == 'SumRuleLoss':
    #     args.log_softmax = False
    return args


def get_optimizer(model, optimizer_name, adam_weight_decay, lr):
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=adam_weight_decay)
    elif optimizer_name == 'AdaCubic':
        raise ValueError('AdaCubic is not published yet.')

    return optimizer


class ProductRuleLoss(nn.Module):
    def __init__(self, num_classes, class_priors,
                 n_modalities=2, reduction=False, label_smoothing=0.0,
                 disable_priors=False):
        self.num_classes = num_classes
        self.n_modalities = n_modalities
        self.reduction = reduction
        self.class_priors = class_priors
        self.disable_priors = disable_priors
        self.label_smoothing = label_smoothing

        assert 0 <= label_smoothing <= 1, 'Label smoothing should be between 0 and 1'

        super(ProductRuleLoss, self).__init__()

    def forward(self, audio_probs, visual_probs, targets):
        if self.disable_priors:
            norm_factors_audio = torch.sum(audio_probs, dim=1, keepdim=True)  # shape: [batch_size, 1]
            norm_factors_visual = torch.sum(visual_probs, dim=1, keepdim=True)  # shape: [batch_size, 1]

            normalized_audio_probs = audio_probs / norm_factors_audio
            normalized_visual_probs = visual_probs / norm_factors_visual
        else:
            prior_scaled_audio_probs = audio_probs * self.class_priors ** (1 - self.n_modalities)
            prior_scaled_visual_probs = visual_probs * self.class_priors ** (1 - self.n_modalities)

            norm_factors_audio = torch.sum(prior_scaled_audio_probs, dim=1, keepdim=True)  # shape: [batch_size, 1]
            norm_factors_visual = torch.sum(prior_scaled_visual_probs, dim=1, keepdim=True)  # shape: [batch_size, 1]

            normalized_audio_probs = prior_scaled_audio_probs / norm_factors_audio
            normalized_visual_probs = prior_scaled_visual_probs / norm_factors_visual

        prod_loss = - torch.sum(torch.log(normalized_audio_probs) * targets
                                + torch.log(normalized_visual_probs) * targets)

        if self.label_smoothing > 0.0:
            smooth_part = torch.sum(torch.log(normalized_audio_probs) + torch.log(normalized_visual_probs))
            prod_loss = (1 - self.label_smoothing) * prod_loss - self.label_smoothing * smooth_part / self.num_classes

        prod_loss = prod_loss / self.n_modalities / self.num_classes if self.reduction else prod_loss

        return prod_loss


def training_fusion(audio_model, visual_model, device,
                    data_loader, optimizer, loss_fn,
                    log_softmax, softmax, reduction, verbose=False):
    audio_model.train()
    visual_model.train()

    total_loss, total_audio_acc, total_visual_acc, total_cnt = 0, 0, 0, 0

    pbar = tqdm(data_loader, disable=not verbose)
    for i, data in enumerate(pbar):
        pbar.set_description("Training batch")

        visual_inputs = data[0].to(device)
        audio_inputs = data[1].to(device)
        targets = data[2].squeeze(1).to(device)

        batch_size = targets.size(0)

        # Convert class labels to one-hot encoding
        one_hot_targets = F.one_hot(targets, num_classes=num_classes)

        def closure(backward=True):
            if backward:
                optimizer.zero_grad()

            # Forward pass through both models
            audio_model_outputs = audio_model(audio_inputs)  # log softmax probs from audio model
            visual_model_outputs = visual_model(visual_inputs)  # log softmax probs  from visual model

            if softmax:
                if loss_fn.__class__.__name__ == 'SumRuleLoss':
                    cri_loss = loss_fn(audio_model_outputs, visual_model_outputs, one_hot_targets)
                elif loss_fn.__class__.__name__ == 'ProductRuleLoss':
                    cri_loss = loss_fn(audio_model_outputs, visual_model_outputs, one_hot_targets)

            elif log_softmax:
                if loss_fn.__class__.__name__ == 'SumRuleLoss':
                    cri_loss = loss_fn(torch.exp(audio_model_outputs), torch.exp(visual_model_outputs), one_hot_targets)
                elif loss_fn.__class__.__name__ == 'ProductRuleLoss':
                    cri_loss = loss_fn(audio_model_outputs, visual_model_outputs, one_hot_targets)

            create_graph = type(optimizer).__name__ == "AdaCubic" or type(optimizer).__name__ == "AdaHessian"

            if backward:
                cri_loss.backward(create_graph=create_graph)

            _, audio_pred_label = torch.max(audio_model_outputs, dim=1)
            _, visual_pred_label = torch.max(visual_model_outputs, dim=1)

            return cri_loss, audio_pred_label, visual_pred_label

        loss, audio_pred_label, visual_pred_label = optimizer.step(closure=closure)

        if reduction:
            total_loss += loss.item()
        else:
            total_loss += loss.item() * batch_size

        audio_acc = torch.sum((audio_pred_label == targets).float()).item()
        total_audio_acc += audio_acc

        visual_acc = torch.sum((visual_pred_label == targets).float()).item()
        total_visual_acc += visual_acc

        total_cnt += batch_size

    return total_loss / total_cnt, total_audio_acc / total_cnt, total_visual_acc / total_cnt


def validating_fusion(audio_model, visual_model, device,
                      data_loader, loss_fn,
                      log_softmax, softmax, reduction, verbose=False):
    audio_model.eval()
    visual_model.eval()

    total_loss, total_audio_acc, total_visual_acc, total_cnt = 0, 0, 0, 0

    with torch.no_grad():
        pbar = tqdm(data_loader, disable=not verbose)
        for data in pbar:
            pbar.set_description("Validation batch")

            visual_inputs = data[0].to(device)
            audio_inputs = data[1].to(device)
            targets = data[2].squeeze(1).to(device)

            batch_size = targets.size(0)

            # Convert class labels to one-hot encoding
            one_hot_targets = F.one_hot(targets, num_classes=num_classes)

            audio_model_outputs = audio_model(audio_inputs)
            visual_model_outputs = visual_model(visual_inputs)

            if softmax:
                if loss_fn.__class__.__name__ == 'SumRuleLoss':
                    cri_loss = loss_fn(audio_model_outputs, visual_model_outputs, one_hot_targets)
                elif loss_fn.__class__.__name__ == 'ProductRuleLoss':
                    cri_loss = loss_fn(audio_model_outputs, visual_model_outputs, one_hot_targets)

            elif log_softmax:
                if loss_fn.__class__.__name__ == 'SumRuleLoss':
                    cri_loss = loss_fn(audio_model_outputs, visual_model_outputs, one_hot_targets)
                elif loss_fn.__class__.__name__ == 'ProductRuleLoss':
                    cri_loss = loss_fn(torch.log(audio_model_outputs), torch.log(visual_model_outputs), one_hot_targets)

            total_loss += cri_loss.item()
            if reduction:
                total_loss += cri_loss.item()
            else:
                total_loss += cri_loss.item() * batch_size

            _, audio_pred_label = torch.max(audio_model_outputs, dim=1)
            _, visual_pred_label = torch.max(visual_model_outputs, dim=1)

            audio_acc = torch.sum((audio_pred_label == targets).float()).item()
            total_audio_acc += audio_acc

            visual_acc = torch.sum((visual_pred_label == targets).float()).item()
            total_visual_acc += visual_acc

            total_cnt += batch_size

    return total_loss / total_cnt, total_audio_acc / total_cnt, total_visual_acc / total_cnt


def main():
    audio_model = get_model(args.audio_model, num_classes, log_softmax=args.log_softmax,
                            softmax=args.softmax, freeze_params=False, device=device)

    visual_model = get_model(args.visual_model, num_classes, log_softmax=args.log_softmax,
                             softmax=args.softmax, freeze_params=False, device=device)

    # audio_model = get_model(args.audio_model)
    # visual_model = get_model(args.visual_model)

    if args.pre_train:
        print('Use pre_train models...')
        audio_checkpoint = torch.load(os.path.join(str(audio_run_path), "audio_model_best.ckpt"))
        audio_model.load_state_dict(audio_checkpoint['state_dict'])
        print(f"Best audio model loaded epoch {audio_checkpoint['epoch']}")
        visual_checkpoint = torch.load(os.path.join(str(visual_run_path), "visual_model_best.ckpt"))
        visual_model.load_state_dict(visual_checkpoint['state_dict'])
        print(f"Best visual model loaded epoch {visual_checkpoint['epoch']}")

    with open(os.path.join(results_dir_fusion, 'audio_network_arch.txt'), "a") as f:
        print(audio_model, file=f)

    pytorch_total_params_audio_model = sum(p.numel() for p in audio_model.parameters() if p.requires_grad)
    print('Number of parameters in the audio model: %.3f M' % (pytorch_total_params_audio_model / 1e6))

    with open(os.path.join(results_dir_fusion, 'visual_network_arch.txt'), "a") as f:
        print(visual_model, file=f)

    pytorch_total_params_visual_model = sum(p.numel() for p in visual_model.parameters() if p.requires_grad)
    print('Number of parameters in the visual model: %.3f M' % (pytorch_total_params_visual_model / 1e6))

    # audio_model_weights = audio_model.state_dict()
    # visual_model_weights = visual_model.state_dict()

    # Define optimizer (using Adam here) that optimizes parameters of both models
    optimizer = torch.optim.Adam([
        # {'params': audio_model.model.fc.parameters(), 'lr': 1e-4},  # Parameter group for audio model
        # {'params': visual_model.model.fc.parameters(), 'lr': 1e-5}  # Parameter group for visual model
        {'params': audio_model.parameters(), 'lr': args.audio_lr, 'weight_decay': args.adam_weight_decay},
        # Parameter group for audio model
        {'params': visual_model.parameters(), 'lr': args.visual_lr, 'weight_decay': args.adam_weight_decay},
        # Parameter group for visual model
    ])
    # Number of modalities M (see paper)
    n_modalities = 2

    priors_path = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                               f'winSize{args.extend_duration_sec}sec',
                               data_content)
    print(f'Priors Path: {priors_path}')

    with open(os.path.join(str(priors_path), f'train_audio_image_priors_fold{args.n_fold}.pkl'), "rb") as handler:
        train_audio_image_priors = dict(sorted(pkl.load(handler).items()))
    train_audio_image_priors = np.array(list(train_audio_image_priors.values()), dtype=np.float32)
    train_audio_image_priors = torch.from_numpy(train_audio_image_priors).to(device)

    loss_fn = ProductRuleLoss(num_classes=num_classes, n_modalities=n_modalities,
                              class_priors=train_audio_image_priors,
                              label_smoothing=args.label_smoothing,
                              reduction=args.reduction, disable_priors=args.disable_priors)

    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2,
                                     factor=0.1, threshold=1e-4, threshold_mode='rel',
                                     cooldown=0, min_lr=0, eps=1e-08)

    # Initialize Early Stopping for validation loss
    validation_loss_early_stopper = EarlyStoppingWithMovingAverage(
        patience=2 * args.window_size, delta=1e-4, window_size=args.window_size, wait_for_full_window=True)

    verbose = False
    # Training loop
    for epoch in range(1, args.epochs + 1):

        # Training step
        train_loss, train_audio_frame_acc, train_visual_frame_acc = training_fusion(
            audio_model=audio_model,
            visual_model=visual_model,
            device=device,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            log_softmax=args.log_softmax,
            softmax=args.softmax,
            reduction=args.reduction,
            verbose=verbose
        )

        # Validation step
        val_loss, val_audio_frame_acc, val_visual_frame_acc = validating_fusion(
            audio_model=audio_model,
            visual_model=visual_model,
            device=device,
            data_loader=val_loader,
            loss_fn=loss_fn,
            log_softmax=args.log_softmax,
            softmax=args.softmax,
            reduction=args.reduction,
            verbose=verbose
        )

        # Early stopping based on validation loss (after validation step)
        validation_loss_early_stopper(val_loss)

        # Save the average validation loss
        avg_loss = validation_loss_early_stopper.get_average_loss()

        # Adjust learning rate based on validation loss
        lr_scheduler.step(val_loss)
        if args.save_stats:
            # Log metrics to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Loss/AvgValidation", avg_loss, epoch)
            writer.add_scalar("Accuracy/Train_Audio", train_audio_frame_acc, epoch)
            writer.add_scalar("Accuracy/Train_Visual", train_visual_frame_acc, epoch)
            writer.add_scalar("Accuracy/Validation_Audio", val_audio_frame_acc, epoch)
            writer.add_scalar("Accuracy/Validation_Visual", val_visual_frame_acc, epoch)

        # Save metrics to log files
        metrics_logs = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'avg_validation_loss': avg_loss,
            'train_audio_acc': train_audio_frame_acc,
            'train_visual_acc': train_visual_frame_acc,
            'val_audio_acc': val_audio_frame_acc,
            'val_visual_acc': val_visual_frame_acc,
        }
        if args.save_stats:
            # Save metrics to individual log files
            for metric, value in metrics_logs.items():
                log_path = os.path.join(results_dir_fusion, f"{metric}.log")
                with open(log_path, 'a') as log_file:
                    log_file.write(f"{epoch}\t{value:.4f}\n")

        # Save the best model if validation loss improves
        if validation_loss_early_stopper.is_best_model() and args.save_stats:
            save_checkpoint({
                "epoch": epoch,
                "state_dict": {
                    "audio_model": audio_model.state_dict(),
                    "visual_model": visual_model.state_dict(),
                },
                "optimizer": optimizer.state_dict(),
            }, is_best=True, checkpoint_dir=results_dir_fusion, model_name="fusion_model")

        # Construct the base print message
        epoch_summary = (f"Epoch: {epoch} | "
                         f"Train Loss: {train_loss:.4f} | "
                         f"Validation Loss: {val_loss:.4f} | "
                         f"AvgValidation Loss: {avg_loss:.4f} | "
                         f"Train Audio Frame Acc: {train_audio_frame_acc:.4f} | "
                         f"Train Visual Frame Acc: {train_visual_frame_acc:.4f} | "
                         f"Validation Audio Frame Acc: {val_audio_frame_acc:.4f} | "
                         f"Validation Visual Frame Acc: {val_visual_frame_acc:.4f}")

        # Get the learning rate for the audio model
        audio_lr, visual_lr = optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']
        epoch_summary += f" | Audio optimizer lr {audio_lr} | Visual optimizer lr {visual_lr}"

        # Check if the ReduceLROnPlateau scheduler is triggered
        if lr_scheduler.num_bad_epochs > lr_scheduler.patience:
            epoch_summary += f" | ReduceLROnPlateau scheduler is triggered (lr: {lr_scheduler.get_last_lr()})!"
        # if validation_loss_early_stopper.is_lr_reduced():
        #     epoch_summary += f" | lr reduced by early stopper (lr: {lr_scheduler.get_last_lr()})!"

        # Append the best model message if applicable
        if validation_loss_early_stopper.is_best_model():
            best_moving_average = validation_loss_early_stopper.get_best_moving_average()
            epoch_summary += f" | Best model saved with lowest AvgValidation Loss: {best_moving_average:.4f}"

        # Print the concatenated message
        print(epoch_summary)
        if args.save_stats:
            # Save to a text file
            with open(os.path.join(results_dir_fusion, "print_logs.txt"),
                      "a") as log_file:  # Use "a" mode to append to the file
                log_file.write(epoch_summary + "\n")  # Add a newline at the end

        # Stop training if early stopping condition is met
        if validation_loss_early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Close TensorBoard writer
    writer.close()

    # Final best validation loss and moving average
    print(f"Final Best Validation Loss: {validation_loss_early_stopper.get_best_moving_average():.4f}")


def load_args_json(json_path):
    """Loads arguments from a JSON file into a namespace."""
    with open(os.path.join(json_path, 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
    return t_args


def update_args(args_dict, keys, source_dict):
    """Updates the main args dictionary with specific keys from the source."""
    for key in keys:
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
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/
--visual_frame_dir
/media/red/sharedFolder/Datasets/VISION/keyFrames/I/
--wav_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/
--extend_duration_sec
2
--data_content
Native
--n_fold
0
--n_run_audio
2
--n_run_visual
1
--milestones
2
--milestones
4
--milestones
8
--reduction
--train_batch
128
--valid_batch
128
--epochs
10
--loss_fn
ProductRuleLoss
--pre_train
--audio_model
ResNet18
--visual_model
ResNet18
--label_smoothing
0.2
    """

    args = get_args()

    results_dir_fusion = check_run_folder(os.path.join(args.results_dir_fusion,
                                                       f'winSize{args.extend_duration_sec}sec',
                                                       args.loss_fn,
                                                       f'visual{args.visual_model}_audio{args.audio_model}',
                                                       args.data_content, f"fold{args.n_fold}"))

    # Paths for audio and visual JSON files
    audio_run_path = os.path.join(args.results_dir_audio, f'winSize{args.extend_duration_sec}sec',
                                  args.data_content, f"fold{args.n_fold}",
                                  args.audio_model, f'run{args.n_run_audio}')
    visual_run_path = os.path.join(args.results_dir_visual, f'winSize{args.extend_duration_sec}sec',
                                   args.data_content, f"fold{args.n_fold}",
                                   args.visual_model, f'run{args.n_run_visual}')

    # Load audio and visual arguments
    t_args_audio = load_args_json(audio_run_path)
    t_args_visual = load_args_json(visual_run_path)

    # if 'model' in t_args_audio.__dict__:
    t_args_audio.__dict__['audio_model'] = t_args_audio.__dict__.pop('model')
    # if 'model' in t_args_visual.__dict__:
    t_args_visual.__dict__['visual_model'] = t_args_visual.__dict__.pop('model')

    # Extract specific values
    audio_keys = ['audio_model', 'n_mels', 'log_mel', 'nfft_scale', 'log_softmax', 'softmax',
                  'log_softmax', 'excluded_devices', 'data_loader_fold_type']
    visual_keys = ['visual_model', 'prnu_enabled', 'fft2', 'log_softmax', 'softmax',
                   'excluded_devices', 'data_loader_fold_type']

    audio_values = {key: t_args_audio.__dict__[key] for key in audio_keys}
    visual_values = {key: t_args_visual.__dict__[key] for key in visual_keys}

    # Validate consistency between audio and visual models
    assert audio_values['log_softmax'] == visual_values['log_softmax'], \
        'Both audio and visual model should have log_softmax enabled'
    assert audio_values['excluded_devices'] == visual_values['excluded_devices'], \
        'Both audio and visual model should have the same excluded devices'
    assert audio_values['data_loader_fold_type'] == visual_values['data_loader_fold_type'], \
        'Audio and visual split type should be the same.'

    # Update args with the relevant audio and visual values
    update_args(args.__dict__, ['audio_model', 'n_mels',
                                'nfft_scale', 'log_mel', 'log_softmax', 'softmax'], audio_values)
    update_args(args.__dict__, ['visual_model', 'prnu_enabled', 'fft2', 'log_softmax', 'softmax',
                                'excluded_devices', 'data_loader_fold_type'], visual_values)

    assert audio_values['log_softmax'] == visual_values['log_softmax']
    assert audio_values['softmax'] == visual_values['softmax']

    # assert not (args.log_softmax and args.softmax), 'log_softmax and softmax can not be enabled simultaneously'

    assert args.softmax, 'Softmax is only allowed in oy experimental result'

    assert args.softmax and (args.loss_fn == 'ProductRuleLoss' or args.loss_fn == 'SumRuleLoss'), \
        'ProductRuleLoss/SumRuleLoss needs softmax to be enabled.'

    print('Creating audio visual dataloader...')

    tr_pkl_name = f"train_audio_image_fold{args.n_fold}.pkl"

    data_content = f'{args.data_content}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else args.data_content

    audio_frame_indices_dir = os.path.join(args.audio_frame_indices_dir,
                                           f'winSize{args.extend_duration_sec}sec')

    pkl_dir_tr_fold = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                                   f'winSize{args.extend_duration_sec}sec', data_content, tr_pkl_name)
    train_set = ImageAudioDataset(pkl_fold_dir=pkl_dir_tr_fold,
                                  visual_frame_dir=args.visual_frame_dir,
                                  wav_dir=args.wav_dir,
                                  audio_frame_indices_dir=audio_frame_indices_dir,
                                  fft2_enabled=args.fft2, prnu_enabled=args.prnu_enabled,
                                  nfft_scale=args.nfft_scale, log_mel=args.log_mel,
                                  n_mels=args.n_mels,
                                  gaussian_noise_flag=args.gaussian_noise_flag,
                                  center_crop=False,  # perform random crop
                                  sample_prc=-0.02)
    print(f"Number of train audio-visual samples {len(train_set)}")

    train_loader = DataLoader(train_set,
                              shuffle=True,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)

    valid_pkl_name = f"valid_audio_image_fold{args.n_fold}.pkl"
    pkl_dir_valid_fold = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                                      f'winSize{args.extend_duration_sec}sec', data_content, valid_pkl_name)
    valid_set = ImageAudioDataset(pkl_fold_dir=pkl_dir_valid_fold,
                                  visual_frame_dir=args.visual_frame_dir,
                                  wav_dir=args.wav_dir,
                                  audio_frame_indices_dir=audio_frame_indices_dir,
                                  fft2_enabled=args.fft2, prnu_enabled=args.prnu_enabled,
                                  nfft_scale=args.nfft_scale, log_mel=args.log_mel,
                                  n_mels=args.n_mels,
                                  gaussian_noise_flag=False,
                                  center_crop=True,
                                  sample_prc=-0.02)
    print(f"Number of valid audio-visual samples {len(valid_set)}")

    val_loader = DataLoader(valid_set,
                            shuffle=False,
                            batch_size=args.valid_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)

    num_classes = train_set.n_classes

    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    writer = SummaryWriter(comment=f"VISION-{args.data_content}", log_dir=os.path.join(results_dir_fusion, 'logs'))

    with open(os.path.join(results_dir_fusion, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    print_args(args)

    main()
