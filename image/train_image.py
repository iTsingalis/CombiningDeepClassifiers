import os
import json
import argparse
from models.audio_image_models import *
from image.dataset import ImageDataset
from image.train import training, validating
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from general_utils import check_run_folder
from general_utils import print_args, EarlyStopperLoss, manage_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    # path #
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--visual_frames_dir", type=str, required=True)
    parser.add_argument("--data_loader_fold_type", type=str, required=True,
                        choices=['my_data_all', 'my_data_I'])
    parser.add_argument('--milestones', default=None, action='append')

    parser.add_argument('--excluded_devices', required=True, default=None, action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)])

    # parameter #
    parser.add_argument("--cuda", default=0, type=int, help='Specify cuda number')
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--train_batch", default=32, type=int)
    parser.add_argument("--valid_batch", default=32, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=int)
    parser.add_argument("--extend_duration_sec", type=int, required=True)

    parser.add_argument('--optimizer',
                        choices=['SGD', 'Adam', 'AdaCubic'],
                        required=True)

    # model #
    parser.add_argument("--model", default=None, type=str,
                        choices=["DenseNet201", "ResNet50", "ResNet101", "ResNet152",
                                 "InceptionV3", "vit_l_16", "vit_b_16", "vit_b_32",
                                 "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"])

    parser.add_argument('--data_content',
                        choices=['YT', 'WA', 'Native'], required=True)

    parser.add_argument("--n_fold", type=int, required=True)

    parser.add_argument('--log_softmax', action='store_true', default=False)
    parser.add_argument('--softmax', action='store_true', default=False)

    parser.add_argument('--fft2', action='store_true', default=False)
    parser.add_argument('--prnu_enabled', action='store_true', default=False)

    parser.add_argument('--reduction', action='store_true', default=False)

    parser.add_argument('--gaussian_noise_flag', action='store_true', default=False)

    parser.add_argument('--save_stats', action='store_true', default=False)

    args = parser.parse_args()

    assert not (args.log_softmax and args.softmax), 'log_softmax and softmax can not be enabled simultaneously'

    return args


def get_model():
    if args.model == "DenseNet201":
        model = DenseNet201(weights=models.DenseNet201_Weights.DEFAULT,
                            num_classes=train_set.n_classes,
                            log_softmax=args.log_softmax,
                            softmax=args.softmax).to(device)
    elif args.model == "ResNet50":
        model = ResNet50(weights=models.ResNet50_Weights.DEFAULT,
                         num_classes=train_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "ResNet101":
        model = ResNet101(weights=models.ResNet101_Weights.DEFAULT,
                         num_classes=train_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "InceptionV3":
        model = InceptionV3(weights=models.Inception_V3_Weights.DEFAULT, num_classes=train_set.n_classes).to(device)
    elif args.model == "ResNet18":
        model = ResNet18(weights=models.ResNet18_Weights.DEFAULT,
                         num_classes=train_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "MobileNetV3Small":
        model = MobileNetV3Small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,
                                 num_classes=train_set.n_classes,
                                 log_softmax=args.log_softmax,
                                 softmax=args.softmax).to(device)
    elif args.model == "MobileNetV3Large":
        model = MobileNetV3Large(weights=models.MobileNet_V3_Large_Weights.DEFAULT,
                                 num_classes=train_set.n_classes,
                                 log_softmax=args.log_softmax,
                                 softmax=args.softmax).to(device)
    elif args.model == "SqueezeNet1_1":
        model = SqueezeNet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT,
                              num_classes=train_set.n_classes,
                              log_softmax=args.log_softmax,
                              softmax=args.softmax).to(device)
    elif args.model == "vit_l_16":
        model = vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT,
                         num_classes=train_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "vit_b_16":
        model = vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT,
                         num_classes=train_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "vit_b_32":
        model = vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT,
                         num_classes=train_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    return model


def get_optimizer(model, optimizer_name):
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif optimizer_name == 'AdaCubic':
        raise ValueError('AdaCubic is not published yet.')

    return optimizer


def main():
    print('Loading model...')

    model = get_model()

    with open(os.path.join(results_dir, 'network_arch.txt'), "a") as f:
        print(model, file=f)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model : %.3f M' % (pytorch_total_params / 1e6))

    if args.log_softmax or args.softmax:
        loss_fn = nn.NLLLoss(reduction='mean' if args.reduction else 'sum')
    else:
        raise ValueError('!!!!!!!!')
        # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        loss_fn = nn.CrossEntropyLoss(reduction='mean' if args.reduction else 'sum')

    optimizer = get_optimizer(model, args.optimizer)

    best_valid_frame_acc = 0.0
    best_valid_loss = float('inf')
    patience, min_delta = 5, 1e-3
    early_stopper = EarlyStopperLoss(patience=patience, min_delta=min_delta)

    if not type(optimizer).__name__ == "AdaCubic":
        # scheduler = MultiStepLR(optimizer, milestones=[1, 2], gamma=0.1)
        milestones = [int(m) for m in args.milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    for epoch in range(args.epochs):
        train_loss, train_frame_acc = training(model=model,
                                               device=device,
                                               data_loader=train_loader,
                                               optimizer=optimizer,
                                               loss_fn=loss_fn,
                                               reduction=args.reduction,
                                               log_softmax=args.log_softmax, softmax=args.softmax)

        valid_loss, valid_frame_acc = validating(model=model,
                                                 device=device,
                                                 data_loader=val_loader,
                                                 loss_fn=loss_fn,
                                                 reduction=args.reduction,
                                                 log_softmax=args.log_softmax, softmax=args.softmax)

        if not type(optimizer).__name__ == "AdaCubic":
            scheduler.step()

        print(f"Epoch {epoch} "
              f" Train Loss: {train_loss:.3f} "
              f" Train Frame Acc: {100 * train_frame_acc:.3f} "
              f" Valid Loss: {valid_loss:.3f}"
              f" Valid Frame Acc: {100 * valid_frame_acc:.3f}")

        # Early stopping check: if no improvement count reaches patience, stop training
        if early_stopper.early_stop(valid_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs. "
                  f"No improvement for {patience} consecutive epochs.")
            break

        if args.save_stats:
            # Save a model only if validation accuracy is increased and loss is less that the minimum recorded loss
            best_valid_frame_acc, best_valid_loss = manage_checkpoint(
                epoch, model, optimizer, valid_frame_acc, valid_loss,
                best_valid_frame_acc, best_valid_loss, "{}".format(results_dir), model_name='visual_model')

            writer.add_scalar("train frame loss", train_loss, epoch + 1)
            writer.add_scalar("validation frame loss", valid_loss, epoch + 1)
            writer.add_scalar("training frame accuracy", train_frame_acc, epoch + 1)
            writer.add_scalar("validation frame accuracy", valid_frame_acc, epoch + 1)

            with open(os.path.join(results_dir, 'tr_frame_acc.log'), 'a') as outfile:
                outfile.write('{}\t{}\n'.format(epoch + 1, train_frame_acc))

            with open(os.path.join(results_dir, 'val_frame_acc.log'), 'a') as outfile:
                outfile.write('{}\t{}\n'.format(epoch + 1, valid_frame_acc))

            with open(os.path.join(results_dir, 'tr_frame_loss.log'), 'a') as outfile:
                outfile.write('{}\t{}\n'.format(epoch + 1, train_loss))

            with open(os.path.join(results_dir, 'val_frame_loss.log'), 'a') as outfile:
                outfile.write('{}\t{}\n'.format(epoch + 1, valid_loss))

    writer.close()


if __name__ == '__main__':
    """
--data_content
Native
--n_fold
0
--model
ResNet50
--project_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/
--epochs
25
--lr
1e-4
--visual_frames_dir
/media/red/sharedFolder/Datasets/VISION/keyFrames/I/
--optimizer
Adam
--results_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/image/results/
--data_loader_fold_type
my_data_I
--excluded_devices
D12
--extend_duration_sec
2
--train_batch
128
--valid_batch
128
--milestones
2
--milestones
4
--milestones
14
--softmax
    """
    args = get_args()

    results_dir = check_run_folder(os.path.join(args.results_dir, f'winSize{args.extend_duration_sec}sec',
                                                args.data_content, f"fold{args.n_fold}", args.model))

    with open(os.path.join(results_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    writer = SummaryWriter(comment=f"VISION-{args.data_content}", log_dir=os.path.join(results_dir, 'logs'))

    print('Creating dataloader...')
    pkl_name = f"train_audio_image_fold{args.n_fold}.pkl"

    data_content = f'{args.data_content}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else args.data_content

    pkl_dir_tr_fold = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                                   f'winSize{args.extend_duration_sec}sec', data_content, pkl_name)

    # pkl_dir_tr_fold = os.path.join(args.project_dir, 'preprocessed_images', args.data_loader_fold_type, args.data_content, pkl_name)
    x_crop_size, y_crop_size = 224, 224

    train_set = ImageDataset(pkl_dir_tr_fold,
                             args.visual_frames_dir,
                             crop_size=(x_crop_size, y_crop_size, 3),
                             fft2=args.fft2, prnu_enabled=args.prnu_enabled,
                             gaussian_noise_flag=args.gaussian_noise_flag,
                             center_crop=False,
                             sample_prc=-0.01)
    print(f"Number of train samples {len(train_set)}")

    train_loader = DataLoader(train_set,
                              shuffle=True,
                              batch_size=args.train_batch,
                              num_workers=args.num_workers,
                              pin_memory=True)

    pkl_name = f"valid_audio_image_fold{args.n_fold}.pkl"

    pkl_dir_valid = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                                 f'winSize{args.extend_duration_sec}sec', data_content, pkl_name)

    valid_set = ImageDataset(pkl_dir_valid, args.visual_frames_dir,
                             crop_size=(x_crop_size, y_crop_size, 3),
                             fft2=args.fft2, prnu_enabled=args.prnu_enabled,
                             gaussian_noise_flag=False,
                             center_crop=True,
                             sample_prc=-0.01)

    print(f"Number of valid samples {len(valid_set)}")
    val_loader = DataLoader(valid_set,
                            shuffle=True,
                            batch_size=args.valid_batch,
                            num_workers=args.num_workers,
                            pin_memory=True)

    print_args(args)

    main()
