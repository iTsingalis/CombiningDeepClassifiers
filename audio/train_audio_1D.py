import os
import json
import argparse
from general_utils import print_args, EarlyStopperLoss, manage_checkpoint
from models.audio_image_models import *
from audio.dataset import AudioDataset1D
from train1D import training, validation
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from general_utils import check_run_folder


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def get_args():
    parser = argparse.ArgumentParser()
    # path #
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--audio_frame_indices_dir", type=str, required=True)
    parser.add_argument("--extracted_wav_dir", type=str, required=True)
    parser.add_argument("--data_loader_fold_type", type=str, required=True,
                        choices=['my_data_all', 'my_data_I'])
    parser.add_argument('--excluded_devices', default=None, action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)])
    # parameter #
    parser.add_argument("--cuda", default=0, type=int, help='Specify cuda number')
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--train_batch", default=65, type=int)
    parser.add_argument("--valid_batch", default=65, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=int)
    parser.add_argument("--extend_duration_sec", type=int, required=True)
    #
    parser.add_argument('--augment', action='store_true', help="Augment data")
    parser.add_argument('--log_mel', required=True, action='store_true', help="Use log Mel-Spectrogram")
    parser.add_argument('--n_mels', type=int, default=128, help='Min SRN in db')
    parser.add_argument('--min_snr_in_db', type=int, default=-2, help='Min SRN in db')
    parser.add_argument('--max_snr_in_db', type=int, default=2, help='Max SRN in db')

    parser.add_argument('--nfft_scale', required=True, type=int, default=2, help='The FFT scale')

    parser.add_argument('--log_softmax', action='store_true', default=False)
    parser.add_argument('--softmax', action='store_true', default=False)

    parser.add_argument('--milestones', default=None, action='append')

    parser.add_argument('--optimizer',
                        choices=['SGD', 'Adam', 'AdaCubic'],
                        required=True)
    # model #
    parser.add_argument("--model", default=None, type=str,
                        choices=["Audio1DDevIdentification", "MobileNetV3Small", "MobileNetV3Large",
                                 "SqueezeNet1_1", "ResNet18", "RawNet", "M5", "M11", "M18", "M34"])

    parser.add_argument('--data_content',
                        choices=['YT', 'WA', 'Native'], required=True)

    parser.add_argument("--n_fold", type=int, required=True)

    parser.add_argument('--reduction', action='store_true', default=False)

    parser.add_argument('--save_stats', action='store_true', default=False)

    args = parser.parse_args()

    assert not (args.log_softmax and args.softmax), 'log_softmax and softmax can not be enabled simultaneously'

    return args


def get_model():
    print('Loading model...')
    if args.model == "DenseNet201":
        model = DenseNet201(weights=models.DenseNet201_Weights.DEFAULT, num_classes=train_set.n_classes).to(device)
    elif args.model == "ResNet50":
        model = ResNet50(weights=models.ResNet50_Weights.DEFAULT,
                         num_classes=train_set.n_classes,
                         log_softmax=args.log_softmax,
                         softmax=args.softmax).to(device)
    elif args.model == "InceptionV3":
        model = InceptionV3(weights=models.Inception_V3_Weights.DEFAULT,
                            num_classes=train_set.n_classes).to(device)
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
    elif args.model == "Audio1DDevIdentification":
        model_args = {'n_filters': 4,
                      'input_linear_dim': 500,
                      'n_linear_out_layer': 1,
                      'n_cnn_layers': 3,
                      'kernel_sizes': [3],
                      'conv_bias': False,
                      'linear_bias': False,
                      'n_classes': train_set.n_classes}
        model = Audio1DDevIdentification(**model_args).to(device)
        with open(os.path.join(results_dir, 'model_args.json'), 'w') as fp:
            json.dump(model_args, fp, indent=4)

    return model


def main():
    model = get_model()

    # print(model)
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

    # Optimizers
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdaCubic':
        # raise ValueError('AdaCubic is published yet.')

        eta1 = 0.05
        eta2 = 0.75
        alpha1 = 2.5  # very successful
        alpha2 = 0.25  # unsuccessful

        optimizer = AdaCubic(model.parameters(), eta1=eta1, eta2=eta2, alpha1=alpha1, alpha2=alpha2,
                             xi0=0.05, tol=1e-4, n_samples=1, average_conv_kernel=False, solver='exact',
                             kappa_easy=0.01, gamma1=0.25)

    best_valid_frame_acc = 0.0
    best_valid_loss = float('inf')
    patience, min_delta = 5, 1e-3
    early_stopper = EarlyStopperLoss(patience=patience, min_delta=min_delta)

    if not type(optimizer).__name__ == "AdaCubic":
        milestones = [int(m) for m in args.milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    for epoch in range(args.epochs):

        train_loss, train_frame_acc = training(model=model,
                                               device=device,
                                               data_loader=train_loader,
                                               optimizer=optimizer, loss_fn=loss_fn,
                                               augmenter=augmenter,
                                               reduction=args.reduction,
                                               log_softmax=args.log_softmax, softmax=args.softmax)

        valid_loss, valid_frame_acc = validation(model=model, device=device,
                                                 data_loader=val_loader, loss_fn=loss_fn,
                                                 reduction=args.reduction,
                                                 log_softmax=args.log_softmax, softmax=args.softmax)

        if not type(optimizer).__name__ == "AdaCubic":
            scheduler.step()

        print(f"Epoch {epoch + 1} Train Loss: {train_loss:.3f}"
              f" Train Acc: {100 * train_frame_acc:.3f} "
              f"Valid Loss: {valid_loss:.3f}"
              f" Valid Frame Acc: {100 * valid_frame_acc:.3f}%")

        # Early stopping check: if no improvement count reaches patience, stop training
        if early_stopper.early_stop(valid_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs. "
                  f"No improvement for {patience} consecutive epochs.")
            break

        if args.save_stats:
            best_valid_frame_acc, best_valid_loss = manage_checkpoint(
                epoch, model, optimizer, valid_frame_acc, valid_loss,
                best_valid_frame_acc, best_valid_loss, "{}".format(results_dir), model_name='audio_model')

            writer.add_scalar("train frame loss", train_loss, epoch + 1)
            writer.add_scalar("validation frame loss", valid_loss, epoch + 1)
            writer.add_scalar("training frame accuracy", train_frame_acc, epoch + 1)
            writer.add_scalar("valid frame accuracy", valid_frame_acc, epoch + 1)

            with open(os.path.join(results_dir, 'tr_acc.log'), 'a') as outfile:
                outfile.write('{}\t{}\n'.format(epoch + 1, train_frame_acc))

            with open(os.path.join(results_dir, 'val_frame_acc.log'), 'a') as outfile:
                outfile.write('{}\t{}\n'.format(epoch + 1, valid_frame_acc))

            with open(os.path.join(results_dir, 'tr_loss.log'), 'a') as outfile:
                outfile.write('{}\t{}\n'.format(epoch + 1, train_loss))

            with open(os.path.join(results_dir, 'val_loss.log'), 'a') as outfile:
                outfile.write('{}\t{}\n'.format(epoch + 1, valid_loss))

    writer.close()


if __name__ == '__main__':
    """
--data_content
Native
--nfft_scale
2
--log_mel
--n_fold
0
--model
ResNet18
--project_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/
--epochs
100
--lr
1e-4
--audio_frame_indices_dir
/media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/
--extracted_wav_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/
--optimizer
Adam
--results_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/
--data_loader_fold_type
my_data_I
--excluded_devices
D12
--extend_duration_sec
2
--softmax
--milestones 2
--milestones 4
--milestones 8
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

    data_content = f'{args.data_content}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else args.data_content

    print('Creating train dataloader...')
    pkl_name = f"train_audio_image_fold{args.n_fold}.pkl"

    pkl_dir_tr_fold = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                                   f'winSize{args.extend_duration_sec}sec', data_content, pkl_name)

    train_set = AudioDataset1D(pkl_dir_tr_fold,
                               audio_frame_indices_dir=os.path.join(args.audio_frame_indices_dir,
                                                                    f'winSize{args.extend_duration_sec}sec'),
                               extracted_wav_dir=args.extracted_wav_dir,
                               sample_prc=-.01,
                               nfft_scale=args.nfft_scale,
                               log_mel=args.log_mel,
                               n_mels=args.n_mels)

    print('Number of train samples : %.3f M -- Input dim : %d' %
          (len(train_set) / 1e6, 22050 * args.extend_duration_sec))

    print('Creating valid dataloader...')
    train_loader = DataLoader(train_set,
                              shuffle=True,
                              batch_size=args.train_batch,
                              num_workers=args.num_workers,
                              pin_memory=True)

    pkl_name = f"valid_audio_image_fold{args.n_fold}.pkl"

    pkl_dir_valid_fold = os.path.join(args.project_dir, 'dataLoaderFolds', args.data_loader_fold_type,
                                      f'winSize{args.extend_duration_sec}sec', data_content, pkl_name)

    valid_set = AudioDataset1D(pkl_dir_valid_fold,
                               audio_frame_indices_dir=os.path.join(args.audio_frame_indices_dir,
                                                                    f'winSize{args.extend_duration_sec}sec'),
                               extracted_wav_dir=args.extracted_wav_dir,
                               sample_prc=-.01,
                               nfft_scale=args.nfft_scale,
                               log_mel=args.log_mel,
                               n_mels=args.n_mels)

    print('Number of valid samples : %.3f M' % (len(valid_set) / 1e6))

    val_loader = DataLoader(valid_set, shuffle=True,
                            batch_size=args.valid_batch,
                            num_workers=args.num_workers,
                            pin_memory=True)

    if args.augment:
        p = .1
        mode = p_mode = "per_example"
        from torch_audiomentations import Compose, AddColoredNoise

        augmenter = Compose([
            # PolarityInversion(p=p, mode=mode, p_mode=p_mode),
            # Shift(min_shift=-0.1, max_shift=0.1, p=p, mode=mode, p_mode=p_mode),
            AddColoredNoise(p=p, min_snr_in_db=args.min_snr_in_db,
                            max_snr_in_db=args.max_snr_in_db, mode=mode, p_mode=p_mode)
        ])

    else:
        augmenter = None

    print_args(args)

    main()
