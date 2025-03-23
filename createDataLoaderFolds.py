import os
import argparse
import warnings
import numpy as np
from glob import glob
import pickle as pkl
import pandas as pd
from pathlib import Path

from pandas.core.common import random_state
from tqdm.auto import tqdm
from collections import Counter

warnings.filterwarnings('ignore')

image_types = {"Flat": "flat", "Native": "nat", "NativeFBH": "natFBH", "NativeFBL": "natFBL", "NativeWA": "natWA"}


def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state=1))
    df_.reset_index(drop=True, inplace=True)
    # df_.index = df_.index.droplevel(0)
    return df_


def extract_features(devices_names_df, n_sample_per_class=-1):
    n_classes = len(np.unique(devices_names_df['dev_numerical_id'].values))

    if n_sample_per_class != -1:
        devices_names_df = stratified_sample_df(devices_names_df,
                                                'dev_numerical_id', 1)

    audio_frame_list = []
    for index, row in devices_names_df.iterrows():
        model_id = row.model_id_list
        devices_name = row.dev_alphabetical_id
        target = row['dev_numerical_id']

        audio_frames = sorted(glob(os.path.join(args.vision_audio_frames_dir, f'winSize{args.extend_duration_sec}sec',
                                                model_id, devices_name, '*.pickle')))

        if len(audio_frames) == 0:
            print(f'File where the audio duration was not the same as the video duration: {devices_name}' )
            continue
        n_frames = len(audio_frames)

        audio_frames_pbar = tqdm(audio_frames, position=0, leave=True, total=n_frames)
        for audio_frame_path in audio_frames_pbar:

            audio_frames_pbar.set_description("Processing (%i / %i): Model name %s Device name %s -- frame: %s"
                                              % (index + 1, len(devices_names_df),
                                                 model_id, devices_name, Path(audio_frame_path).stem))
            audio_frame_list.append({"audio_image_frame_name": Path(audio_frame_path).stem,
                                     "device_name": devices_name,
                                     "model_id": model_id,
                                     "target": target,
                                     "n_classes": n_classes})
            assert target >= 0 or target < n_classes

    assert len(set([a["audio_image_frame_name"] for a in audio_frame_list])) == len(audio_frame_list)

    assert len(np.unique([i['target'] for i in audio_frame_list])) == n_classes
    return audio_frame_list


def calculate_posterior_prob(values):
    histogram = dict(Counter([audio_frame["target"] for audio_frame in values]))
    assert sum(histogram.values()) == len(values)
    posterior_prob = {k: v / sum(histogram.values()) for k, v in histogram.items()}
    return posterior_prob


def main():
    visual_content = f'{args.visual_content}_{"_".join(args.excluded_devices)}_Excluded' \
        if args.excluded_devices is not None else args.visual_content
    tr_df = pd.read_csv(f'{args.fold_dir}/{visual_content}/train_fold{args.n_fold}.csv')
    tst_df = pd.read_csv(f'{args.fold_dir}/{visual_content}/test_fold{args.n_fold}.csv')
    val_df = pd.read_csv(f'{args.fold_dir}/{visual_content}/val_fold{args.n_fold}.csv')

    save_path = os.path.join(args.output_dir, f'winSize{args.extend_duration_sec}sec', visual_content)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Training data
    training_values = extract_features(tr_df, n_sample_per_class=-1)
    training_priors_prob = calculate_posterior_prob(training_values)
    print(f'training_priors_prob: {training_priors_prob}')
    with open(os.path.join(save_path, f'train_audio_image_fold{args.n_fold}.pkl'), "wb") as handler:
        pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(save_path, f'train_audio_image_priors_fold{args.n_fold}.pkl'), "wb") as handler:
        pkl.dump(training_priors_prob, handler, protocol=pkl.HIGHEST_PROTOCOL)

    # Validation data
    validation_values = extract_features(val_df)
    validation_priors_prob = calculate_posterior_prob(validation_values)
    print(f'validation_priors_prob: {validation_priors_prob}')
    with open(os.path.join(save_path, f'valid_audio_image_fold{args.n_fold}.pkl'), "wb") as handler:
        pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(save_path, f'valid_audio_image_priors_fold{args.n_fold}.pkl'), "wb") as handler:
        pkl.dump(validation_priors_prob, handler, protocol=pkl.HIGHEST_PROTOCOL)

    # Test data
    test_values = extract_features(tst_df)
    test_priors_prob = calculate_posterior_prob(test_values)
    print(f'test_priors_prob: {test_priors_prob}')
    with open(os.path.join(save_path, f'test_audio_image_fold{args.n_fold}.pkl'), "wb") as handler:
        pkl.dump(test_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(save_path, f'test_audio_image_priors_fold{args.n_fold}.pkl'), "wb") as handler:
        pkl.dump(test_priors_prob, handler, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    """
--output_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/dataLoaderFolds/my_data_I
--vision_audio_frames_dir
/media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I
--visual_content
Native
--fold_dir
/media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/folds/my_folds
--n_fold
1
--excluded_devices
D12
--extend_duration_sec
2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_audio_frames_dir", type=str, required=True)
    parser.add_argument("--fold_dir", type=str, required=True)
    parser.add_argument("--n_fold", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument('--visual_content',
                        choices=['YT', 'Native', 'WA'], required=True)
    parser.add_argument('--excluded_devices', default=None, action='append',
                        choices=[f'D{i:02d}' for i in range(1, 36)])
    parser.add_argument("--extend_duration_sec", type=int, required=True)

    args = parser.parse_args()

    main()
