import os
import mpltex
import matplotlib.pyplot as plt


def main():
    # sets = ['tr', 'val']
    sets = ['val']
    for set in sets:
        for model in ['MobileNetV3Large']: #['ResNet18', 'MobileNetV3Large', 'MobileNetV3Small', 'SqueezeNet1_1']:  # Models to iterate through
            for sub_task in ['Native', 'WA', 'YT']:  # Sub-tasks to iterate through
                val_losses = []
                for n_fold in range(0, 5):  # Folds to iterate through
                    n_run = 'run2' if n_fold == 0 else 'run1'
                    run_path = os.path.join(exp_path, sub_task, f'fold{n_fold}', model, n_run)
                    try:
                        with open(os.path.join(run_path, f'{set}_loss.log'), 'r') as infile:
                            _losses = []
                            for line in infile:
                                # Split each line into epoch and loss, and append loss to the list
                                _, loss = line.split()
                                _losses.append(float(loss))
                            val_losses.append(_losses)
                    except FileNotFoundError:
                        print(f"File not found: {os.path.join(run_path, 'val_loss.log')}")
                    except ValueError:
                        print(f"Invalid line in file: {os.path.join(run_path, 'val_loss.log')}")

                # Plotting the validation losses
                linestyles = mpltex.linestyle_generator()

                fig, ax = plt.subplots()
                for i, val_loss in enumerate(val_losses):
                    plt.plot(val_loss, **next(linestyles), label=f'fold {i}')

                ax.yaxis.offsetText.set_fontsize(24)

                plt.xlabel('Epoch', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                plt.title('Loss Over Epochs')
                plt.legend(prop={'size': 15})
                # plt.axis('equal')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'plots/audio_{model}_{sub_task}.eps', bbox_inches='tight',  format='eps')
                # plt.show()


if __name__ == "__main__":
    exp_path = '/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/winSize2sec/'
    main()
