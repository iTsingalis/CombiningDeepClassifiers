import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def main():
    # save_path = os.path.join(args.output_dir, args.visual_content)

    save_path = Path(__file__).resolve().parent.joinpath('my_folds', args.visual_content)
    save_path = f'{save_path}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else save_path

    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_names_dic = {}
    for model_names in sorted(os.listdir(args.input_dir)):
        if os.path.isdir(os.path.join(args.input_dir, model_names)):
            sub_folder = glob(os.path.join(args.input_dir, model_names, '*'), recursive=False)
            device_names = set([os.path.basename(os.path.normpath(sf)) for sf in sub_folder])
            model_names_dic.update({model_names: device_names})

    class_names = ['D' + str(i + 1).zfill(2) for i in range(35)]
    class_names = [class_name for class_name in class_names if
                   args.excluded_devices is None or class_name not in args.excluded_devices]

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(class_names)

    # show how many classes there are
    n_classes = len(le.classes_)
    print(f'n_classes: {len(list(le.classes_))}')

    dev_numerical_id_list = []
    dev_alphabetical_id_list = []
    model_id_list = []
    for model_name, dev_names in model_names_dic.items():
        for dev_name in dev_names:

            if args.flat and 'flat' in dev_name:
                continue

            if (('YT' in dev_name and 'YT' in args.visual_content)
                    or ('WA' in dev_name and 'WA' in args.visual_content)
                    or ('YT' not in dev_name and 'WA' not in dev_name and 'Native' in args.visual_content)):

                str_dev_id = dev_name[:3]
                if str_dev_id not in class_names:
                    continue

                dev_id = int(le.transform([str_dev_id]))
                dev_numerical_id_list.append(dev_id)
                dev_alphabetical_id_list.append(dev_name)
                model_id_list.append(model_name)

    print(f'Number of labels: {len(np.unique(dev_numerical_id_list))}')

    kf_pairs = list(kf.split(dev_alphabetical_id_list, dev_numerical_id_list))
    for i, (train_val_index, test_index) in enumerate(kf_pairs):
        train_index, val_index = train_test_split(train_val_index,
                                                  stratify=[dev_numerical_id_list[t] for t in train_val_index],
                                                  train_size=0.8, random_state=i)
        tr_data_pair = [(model_id_list[j], dev_alphabetical_id_list[j], dev_numerical_id_list[j]) for j in train_index]
        val_data_pair = [(model_id_list[j], dev_alphabetical_id_list[j], dev_numerical_id_list[j]) for j in val_index]
        tst_data_pair = [(model_id_list[j], dev_alphabetical_id_list[j], dev_numerical_id_list[j]) for j in test_index]

        save = True
        if save:
            tr_df = pd.DataFrame(tr_data_pair, columns=['model_id_list', 'dev_alphabetical_id', 'dev_numerical_id'])
            tr_df.to_csv(f'{save_path}/train_fold{i}.csv', index=False)

            tst_df = pd.DataFrame(tst_data_pair, columns=['model_id_list', 'dev_alphabetical_id', 'dev_numerical_id'])
            tst_df.to_csv(f'{save_path}/test_fold{i}.csv', index=False)

            val_df = pd.DataFrame(val_data_pair, columns=['model_id_list', 'dev_alphabetical_id', 'dev_numerical_id'])
            val_df.to_csv(f'{save_path}/val_fold{i}.csv', index=False)

            assert len(np.unique(tr_df['dev_numerical_id'])) == n_classes
            assert len(np.unique(tst_df['dev_numerical_id'])) == n_classes
            assert len(np.unique(val_df['dev_numerical_id'])) == n_classes

    all_train, all_test, all_valid = [], [], []

    for cnt_fold in range(n_splits):
        all_train.extend(pd.read_csv(f'{save_path}/train_fold{cnt_fold}.csv')['dev_alphabetical_id'])
        all_test.extend(pd.read_csv(f'{save_path}/test_fold{cnt_fold}.csv')['dev_alphabetical_id'])
        all_valid.extend(pd.read_csv(f'{save_path}/val_fold{cnt_fold}.csv')['dev_alphabetical_id'])

    assert set(all_test) == set(dev_alphabetical_id_list)
    print()


if __name__ == '__main__':
    """
    --output_dir /media/blue/tsingalis/DevIDFusion/folds/my_folds 
    --input_dir /media/blue/tsingalis/DevIDFusion/audio/extractedWav/
    --visual_content Native
     --excluded_devices D12
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sr", default=22050, type=int)
    parser.add_argument("--n_splits", default=4, type=int)
    parser.add_argument('--visual_content',
                        choices=['YT', 'WA', 'Native'], required=True)
    parser.add_argument('--excluded_devices', default=None, action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)])

    parser.add_argument('--flat', action='store_true', default=False)

    args = parser.parse_args()

    main()
