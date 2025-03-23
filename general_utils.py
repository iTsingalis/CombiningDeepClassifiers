import os
import csv
import shutil
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class EarlyStopperAccuracy:
    """
    Early stopping based on validation accuracy.
    Stops training if the validation accuracy does not improve beyond a threshold
    (min_delta) for a given number of consecutive checks (patience).
    """

    def __init__(self, patience=1, min_delta=0, model_name=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_accuracy = 0  # Start with the worst possible accuracy (0)
        self.model_name = model_name

    def early_stop(self, validation_accuracy):
        """
        Checks if training should be stopped based on validation accuracy.

        Args:
            validation_accuracy (float): The current validation accuracy.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if validation_accuracy > self.max_validation_accuracy + self.min_delta:
            # Improvement detected
            self.max_validation_accuracy = validation_accuracy
            self.counter = 0
            print(
                f"\nValidation accuracy {self.model_name} improved to {validation_accuracy:.6f}. Counter reset to {self.counter}.\n")
        else:
            # No improvement
            self.counter += 1
            print(
                f"\nValidation accuracy {self.model_name} did not improve. Current accuracy: {validation_accuracy:.6f}, "
                f"Best accuracy: {self.max_validation_accuracy:.6f}. Counter: {self.counter}/{self.patience}\n")
            if self.counter >= self.patience:
                print(f"\nEarly stopping triggered due to no improvement in validation accuracy {self.model_name}.\n")
                return True
        return False


class EarlyStopperLoss:
    """
    Early stopping based on validation loss.
    Stops training if the validation loss does not improve beyond a threshold
    (min_delta) for a given number of consecutive checks (patience).

    Reference:
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Checks if training should be stopped based on validation loss.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print(f"\nValidation loss improved to {validation_loss:.6f}. Counter reset to {self.counter}.\n")
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"\nValidation loss did not improve. Current loss: {validation_loss:.6f}, "
                  f"Best loss: {self.min_validation_loss:.6f}. Counter: {self.counter}/{self.patience}\n")
            if self.counter >= self.patience:
                print("\nEarly stopping triggered due to no improvement in validation loss.\n")
                return True
        return False


import collections
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau


class CustomReduceLROnPlateau(ReduceLROnPlateau):
    def reduce_lr_manually(self, epoch=None):
        """Manually reduce the learning rate."""
        if epoch is None:
            epoch = self.last_epoch
        self._reduce_lr(epoch)
        # print(f"Manually reduced LR to: {self.optimizer.param_groups[0]['lr']}")


class EarlyStoppingWithMovingAverage:
    def __init__(self, patience=5, delta=0.01, window_size=5, wait_for_full_window=False, lr_scheduler=None):
        """
        Initializes EarlyStopping with Moving Average.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            window_size (int): Size of the moving window to calculate the average validation loss.
            wait_for_full_window (bool): Whether to wait until the deque is filled before computing the moving average.
        """
        self.counter = 0  # Counts epochs since the last improvement in validation loss
        self.delta = delta  # Minimum change in the monitored metric to qualify as an improvement
        self.is_best = False  # Tracks if the current epoch has the best model
        self.patience = patience  # Number of epochs to wait before stopping if no improvement is observed
        self.early_stop = False  # Flag indicating whether early stopping has been triggered
        self.moving_average = None  # Moving average of validation loss over a specified window
        self.best_loss = float("inf")  # The best (lowest) validation loss encountered so far
        self.window_size = window_size  # Size of the window for calculating the moving average
        self.wait_for_full_window = wait_for_full_window  # Wait until the moving average window is full
        self.losses = collections.deque(maxlen=window_size)  # Stores the last `window_size` validation losses
        self.lr_scheduler = lr_scheduler
        self.lr_reduced = False

    def __call__(self, val_loss):
        """
        Update the early stopping condition based on the validation loss.

        Args:
            val_loss (float): The validation loss for the current epoch.
        """
        # Add current validation loss to the deque
        self.losses.append(val_loss)

        # Calculate the moving average
        self.moving_average = sum(self.losses) / len(self.losses)

        # Ensure calculations proceed only if enough data is available
        if self.wait_for_full_window and len(self.losses) < self.window_size:
            # print('Wait until the deque is filled...')
            return  # Wait until the deque is filled

        # Check if the moving average shows improvement
        if self.moving_average is not None and self.moving_average < self.best_loss - self.delta:
            self.best_loss = self.moving_average
            self.counter = 0  # Reset counter
            self.is_best = True  # Mark this epoch as the best
            self.lr_reduced = False  # Reset the lr_reduced flag since improvement occurred
        else:
            self.counter += 1
            self.is_best = False  # Not the best model

        # Apply lr_reduced if counter reaches half the patience threshold. Do that only once.
        if self.lr_reduced and self.counter == self.patience:
            self.lr_reduced.reduce_lr_manually()
            self.lr_reduced = True

        # Check if patience has been exceeded
        if self.counter >= self.patience:
            self.lr_reduced = False
            self.early_stop = True

    def get_best_moving_average(self):
        """
        Get the best (lowest) moving average loss.

        Returns:
            float: The moving average loss, or the fallback value if not enough data is available.
        """
        return self.best_loss

    def get_average_loss(self):
        """
        Get the current moving average loss.

        Returns:
            float: The moving average loss, or the fallback value if not enough data is available.
        """
        return self.moving_average

    def is_best_model(self):
        """
        Check if the current epoch has the best model.

        Returns:
            bool: True if the current model is the best, False otherwise.
        """
        return self.is_best

    def is_lr_reduced(self):
        """
        Check if the learning rate has been reduced.

        This method checks the `lr_reduced` flag to determine if the learning rate has been
        reduced by the lr_reduced at any point during the training process.

        Returns:
            bool: True if the learning rate has been reduced, False otherwise.
        """
        return self.lr_reduced


def manage_checkpoint_with_early_stopping(epoch, model, optimizer, valid_loss, best_valid_loss,
                                          checkpoint_dir, model_name, early_stopping):
    """
    Function to manage model saving with EarlyStoppingWithMovingAverage using the average loss from the early stopper.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model being trained.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        valid_loss (float): Validation loss for the current epoch.
        best_valid_loss (float): Best validation loss observed so far.
        checkpoint_dir (str): Directory to save checkpoints.
        model_name (str): Name of the model.
        early_stopping (EarlyStoppingWithMovingAverage): Early stopping instance.

    Returns:
        best_valid_loss, should_stop
    """
    # Update early stopping with the current validation loss
    early_stopping(valid_loss)

    # Early stopping triggered
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch + 1}.\n")
        return best_valid_loss, True

    # Use the average loss from the early stopping mechanism
    avg_loss = early_stopping.moving_average

    # Criterion: Save the model if the average validation loss improves
    is_avg_loss_improved = avg_loss < best_valid_loss

    if is_avg_loss_improved:
        print(f'\nSave {model_name}: Previous Best Avg Loss {best_valid_loss:.4f} '
              f'-- Current Avg Loss {avg_loss:.4f}\n')

        # Update the best validation loss
        best_valid_loss = avg_loss

        # Save the checkpoint with the current model and optimizer state
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": model.state_dict(),
                         'best_valid_loss': best_valid_loss,
                         "optimizer": optimizer.state_dict()},
                        is_best=True,
                        checkpoint_dir=checkpoint_dir,
                        model_name=model_name)
    else:
        print(f"\n{model_name} not saved: No improvement in average validation loss. "
              f"Current Avg Loss: {avg_loss:.4f} (Best: {best_valid_loss:.4f})\n")

    return best_valid_loss, False


# Function to manage saving and avoid overfitting
def manage_checkpoint(epoch, model, optimizer, valid_frame_acc, valid_loss,
                      best_valid_frame_acc, best_valid_loss, checkpoint_dir, model_name):
    # Criterion: Model is saved if validation accuracy improves and validation loss does not increase.
    is_best_valid_frame_acc = valid_frame_acc > best_valid_frame_acc
    is_loss_improved_or_stable = valid_loss <= best_valid_loss

    if is_best_valid_frame_acc and is_loss_improved_or_stable:
        print(f'\nSave {model_name}: Previous Best Frame Acc {100 * best_valid_frame_acc:.3f}% '
              f'-- Current Frame Acc {100 * valid_frame_acc:.3f}%\n')

        # Update the best metrics
        best_valid_frame_acc = valid_frame_acc
        best_valid_loss = valid_loss

        # Save the checkpoint with the current model and optimizer state
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": model.state_dict(),
                         'best_valid_frame_acc': best_valid_frame_acc,
                         'best_valid_loss': best_valid_loss,
                         "optimizer": optimizer.state_dict()},
                        is_best=True,
                        checkpoint_dir=checkpoint_dir,
                        model_name=model_name)
    else:
        print(f"\n{model_name} not saved: No improvement in validation metrics. "
              f"Current {model_name} frame acc: {100 * valid_frame_acc:.3f}% (Best: {100 * best_valid_frame_acc:.3f}%) "
              f"Current {model_name} valid loss: {valid_loss:.4f} (Best: {best_valid_loss:.4f})\n")

    return best_valid_frame_acc, best_valid_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def check_run_folder(exp_folder):
    run = 1
    run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    if not os.path.exists(run_folder):
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        # os.makedirs(run_folder)
        print("Path {} created".format(run_folder))
        return run_folder

    while os.path.exists(run_folder):
        run += 1
        run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    # os.makedirs(run_folder)
    print("Path {} created".format(run_folder))
    return run_folder


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def save_checkpoint(state, is_best, checkpoint_dir, model_name):
    filename = os.path.join(checkpoint_dir, f'{model_name}.ckpt')
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist")
        os.mkdir(checkpoint_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, f"{model_name}_best.ckpt"))


def save_cm_stats(cm, classes, normalize, title, save_dir, figsize):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    else:
        cm = cm.astype('int')
    import seaborn as sns

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, cmap='Blues',
                fmt='.1%' if normalize else "d",
                annot_kws={"fontsize": 8},
                xticklabels=classes,
                yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if normalize:
        title = title + '_normalized'
    if save_dir is not None:
        plt.savefig(f'{save_dir}/cm_{title}.jpg', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{save_dir}/cm_{title}.eps', bbox_inches='tight', pad_inches=0, format='eps')
        plt.close()
    else:
        plt.show()


def print_args(args, logger=None):
    message = ''
    for k, v in sorted(vars(args).items()):
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
    if logger is not None:
        logger.info(message)
    print(message)

