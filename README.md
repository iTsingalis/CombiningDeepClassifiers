# On Combining Deep Neural Network Classifiers for Source Device Identification
This repository contains the code for the paper On Combining Deep Neural Network Classifiers for Source Device Identification introduced by Ioannis Tsingalis and Constantine Kotropoulos

## Abstract
This paper proposes combining deep neural network classifiers while simultaneously optimizing the networks. 
The proposed combination scheme enhances the accuracy of each classifier, which, in turn, boosts the overall combined 
accuracy during a post-processing step. The proposed classification scheme is thoroughly evaluated on a dataset 
specifically designed for multimedia forensics research. The combined classifiers include shallow and deep neural 
networks, with input data comprising original and manipulated content processed through online social networks such 
as YouTube, WhatsApp, and Facebook. The experimental results demonstrate promising performance, proving the usability 
of the proposed classifier combination scheme. Specifically, it is observed that the accuracy of shallow neural networks 
improves significantly when combined with deep neural networks. This performance enhancement is particularly notable when 
the combined classifiers are trained on data manipulated by online social network platforms. 

----------------------------------------------------------
## Repository Preparation
### 1. Clone the repository
### 2. Download datasets
You can download the VISION dataset from [here](https://lesc.dinfo.unifi.it/VISION/) using the script 
```angular2
'/CombiningDeepClassifiers/VISION/downloadVISION.py'
```
### 3. Extract audio (.wav) files from videos
Run the script 
```angular2
'/CombiningDeepClassifiers/audio/extractWav.py'
```
This script creates the folder 
```angular2
'/CombiningDeepClassifiers/audio/extractedWav/' 
```

with structure

```angular2html
├── D01_V_flat_move_0001
│   └── D01_V_flat_move_0001.wav
├── D01_V_flat_move_0002
│   └── D01_V_flat_move_0002.wav
├── D01_V_flat_panrot_0001
│   └── D01_V_flat_panrot_0001.wav
├── D01_V_flat_still_0001
│   └── D01_V_flat_still_0001.wav
├── D01_V_flatWA_move_0001
│   └── D01_V_flatWA_move_0001.wav
├── D01_V_flatWA_move_0002
│   └── D01_V_flatWA_move_0002.wav
├── D01_V_flatWA_panrot_0001
│   └── D01_V_flatWA_panrot_0001.wav
├── D01_V_flatWA_still_0001
...
...
```

#### Alternatively, extract the audio modality related files from [here](), [here](), and [here]() into the folders 
```angular2
'/CombiningDeepClassifiers/audio/extractedMel/', 
'/CombiningDeepClassifiers/audio/extractedWav/', and 
'/CombiningDeepClassifiers/audio/results1D/',
```
respectively.
- 
### 4. Extract log mel spectrograms from the audio (.wav) files

Run the script 
```angular2
'/CombiningDeepClassifiers/audio/extractMel.py'
```

This script creates the folder 
```angular2
'/CombiningDeepClassifiers/audio/extractedMel/' 
```
with structure

```angular2html
├── D01_V_flat_move_0001
│   ├── D01_V_flat_move_0001_chanel0.png
│   ├── D01_V_flat_move_0001_chanel1.png
│   ├── D01_V_flat_move_0001_chanel2.png
│   └── D01_V_flat_move_0001.pkl
├── D01_V_flat_move_0002
│   ├── D01_V_flat_move_0002_chanel0.png
│   ├── D01_V_flat_move_0002_chanel1.png
│   ├── D01_V_flat_move_0002_chanel2.png
│   └── D01_V_flat_move_0002.pkl
├── D01_V_flat_panrot_0001
│   ├── D01_V_flat_panrot_0001_chanel0.png
│   ├── D01_V_flat_panrot_0001_chanel1.png
│   ├── D01_V_flat_panrot_0001_chanel2.png
│   └── D01_V_flat_panrot_0001.pkl
├── D01_V_flat_still_0001
│   ├── D01_V_flat_still_0001_chanel0.png
│   ├── D01_V_flat_still_0001_chanel1.png
│   ├── D01_V_flat_still_0001_chanel2.png
│   └── D01_V_flat_still_0001.pkl
├── D01_V_flatWA_move_0001
│   ├── D01_V_flatWA_move_0001_chanel0.png
│   ├── D01_V_flatWA_move_0001_chanel1.png
│   ├── D01_V_flatWA_move_0001_chanel2.png
│   └── D01_V_flatWA_move_0001.pkl
```

#### Alternatively, extract the image modality related file from [here]() into the folder
```angular2
'/CombiningDeepClassifiers/audio/results/'.
```

### 5. Create training, evaluation and test Splits (a.k.a. folds)

Run the script 
```angular2
'/CombiningDeepClassifiers/folds/create_splits.py'
```

This script creates the folder 
```angular2
'/CombiningDeepClassifiers/folds/my_folds/'
 ```
with subfolders `Native`, `WA`, and `YT`.

### 6. Extract Video I-Frame (images) 

Run the script 
```angular2
'/CombiningDeepClassifiers/VISION/key_frame_extraction.py' --input_dir /media/red/sharedFolder/Datasets/VISION/dataset/ --output_dir /media/red/sharedFolder/Datasets/VISION/keyFrames/ --frame_type I ```
 ```

This script creates the folder 
```angular2
'/media/red/sharedFolder/Datasets/VISION/keyFrames/I'
```
with structure
```angular2
├── D01_Samsung_GalaxyS3Mini
│   ├── D01_V_flat_move_0001
│   │   ├── D01_V_flat_move_0001-00001.png
│   │   ├── D01_V_flat_move_0001-00031.png
│   │   ├── D01_V_flat_move_0001-00061.png
......................................................
│   ├── D01_V_flat_move_0002
│   │   ├── D01_V_flat_move_0002-00001.png
│   │   ├── D01_V_flat_move_0002-00031.png
│   │   ├── D01_V_flat_move_0002-00061.png
......................................................
├── D02_Apple_iPhone4s
│   ├── D02_V_flat_move_0001
│   │   ├── D02_V_flat_move_0001-00001.png
│   │   ├── D02_V_flat_move_0001-00031.png
│   │   ├── D02_V_flat_move_0001-00061.png
......................................................
```

that contains the extracted keyframes used to train the image models.

### 7. Extract Audio and Video Frame (images) Indices

Run the script 
```angular2
'/CombiningDeepClassifiers/VISION/extractAudioVideoFrames.py'
```

This script creates the folder 
```angular2
'/media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/'
 ```
with structure
```angular2
├── D01_Samsung_GalaxyS3Mini
│   ├── D01_V_flat_move_0001
│   │   ├── D01_V_flat_move_0001-00001.pickle
│   │   ├── D01_V_flat_move_0001-00031.pickle
│   │   ├── D01_V_flat_move_0001-00061.pickle
| .......................................................
|   ├── D01_V_flat_move_0002
|   |   ├── D01_V_flat_move_0002-00001.pickle
│   │   ├── D01_V_flat_move_0002-00031.pickle
|   │   │   ├── D01_V_flat_move_0002-00061.pickle
├── D02_Apple_iPhone4s
│   ├── D02_V_flat_move_0001
│   │   ├── D02_V_flat_move_0001-00001.pickle
│   │   ├── D02_V_flat_move_0001-00031.pickle
│   │   ├── D02_V_flat_move_0001-00061.pickle
| .......................................................
│   ├── D02_V_flat_move_0002
│   │   ├── D02_V_flat_move_0002-00001.pickle
│   │   ├── D02_V_flat_move_0002-00031.pickle
│   │   ├── D02_V_flat_move_0002-00061.pickle
| .......................................................
 ```
Each `.pickle` files contains a one-to-one correspondence between the audio and video frame. 
This correspondence will be used in the online fusion training.  


--------------------------------------

## Train Models
### 1. Train Audio Network
Run the script 

```angular2
/CombiningDeepClassifiers/audio/train_audio_1D.py --data_content Native --nfft_scale 2 --log_mel --n_fold 0 --model ResNet18 --project_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/ --epochs 100 --lr 1e-4 --audio_frame_indices_dir /media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/ --extracted_wav_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/ --optimizer Adam --results_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/ --data_loader_fold_type my_data_I --excluded_devices D12 --extend_duration_sec 2 --softmax --milestones 2 --milestones 4 --milestones 8
```

This script creates the folder 
```angular2
CombiningDeepClassifiers/audio/results1D/winSize2Sec/Native/fold0/ResNet18/run{some incremental id}
```

with subfolders `Native`, `WA`, and `YT`, where the best model `model_best.ckpt` is saved for each `fold{i}`, with `i=0, 1, 2, 3, 4`. Each fold folder has the following structure

```angular2html
MobileNetV3Large
└── run1
    ├── args.json
    ├── logs
    ├── audio_model_best.ckpt
    ├── audio_model.ckpt
    ├── proba.pkl
    ├── tr_frame_acc.log
    ├── tr_frame_loss.log
    ├── val_frame_acc.log
    └── val_frame_loss.log
```
Alternatively, you can run the script  

```angular2
/CombiningDeepClassifiers/audio/call_train_audio_1D.py 
```

which trains models with different neural networks (e.g., MobileNetV3Large, ResNet18, etc.) and sub-tasks (e.g., Native, YT, ...).

### 2. Train Image Network

Run the script 

```angular2
/CombiningDeepClassifiers/image/train_image.py --data_content Native --n_fold 0 --model ResNet50 --project_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/ --epochs 25 --lr 1e-4 --visual_frames_dir /media/red/sharedFolder/Datasets/VISION/keyFrames/I/ --optimizer Adam --results_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/image/results/ --data_loader_fold_type my_data_I --excluded_devices D12 --extend_duration_sec 2 --train_batch 128 --valid_batch 128 --milestones 2 --milestones 4 --milestones 14 --softmax
```

This script creates the folder 
```angular2
CombiningDeepClassifiers/image/results/winSize2Sec/Native/fold0/ResNet50/run{some incremental id}
```

with subfolders `Native`, `WA`, and `YT`, where the best model `model_best.ckpt` is saved for each `fold{i}`, with `i=0, 1, 2, 3, 4`. Each fold folder has the following structure

```angular2html
MobileNetV3Large
└── run1
    ├── args.json
    ├── logs
    ├── visual_model_best.ckpt
    ├── visual_model.ckpt
    ├── proba.pkl
    ├── tr_frame_acc.log
    ├── tr_frame_loss.log
    ├── val_frame_acc.log
    └── val_frame_loss.log
```
Alternatively, you can run the script  

```angular2
/CombiningDeepClassifiers/image/call_train_image.py 
```

which trains models with different neural networks (e.g., MobileNetV3Large, ResNet18, etc.)  and sub-tasks (e.g., Native, YT, ...).

### 3. Train Fusion Network

Run the script 

```angular2
/CombiningDeepClassifiers/fusion/online_fusion_train.py --results_dir_fusion /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/fusion/results/ --results_dir_audio /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/ --results_dir_visual /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/image/results/ --audio_frame_indices_dir /media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/ --project_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/ --visual_frame_dir /media/red/sharedFolder/Datasets/VISION/keyFrames/I/ --wav_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/ --extend_duration_sec 2 --data_content Native --n_fold 0 --n_run_audio 2 --n_run_visual 1 --milestones 2 --milestones 4 --milestones 8 --reduction --train_batch 128 --valid_batch 128 --epochs 10 --loss_fn ProductRuleLoss --pre_train --audio_model MobileNetV3Large --visual_model MobileNetV3Large --label_smoothing 0.2 
```

This script creates the folder 
```angular2
CombiningDeepClassifiers/fusion/results/winSize2Sec/ProductRuleLoss/visualMobileNetV3Large_audioMobileNetV3Large/Native/fold0/
```

with subfolders `Native`, `WA`, and `YT`, where the best model `model_best.ckpt` is saved for each `fold{i}`, with `i=0, 1, 2, 3, 4`. Each fold folder has the following structure

```angular2html
visualMobileNetV3Small_audioMobileNetV3Large
├── args.json
├── audio_network_arch.txt
├── avg_validation_loss.log
├── fusion_model_best.ckpt
├── fusion_model.ckpt
├── logs
├── print_logs.txt
├── stats
│   ├── audio_stats.pkl
│   └── visual_stats.pkl
├── train_audio_acc.log
├── train_loss.log
├── train_visual_acc.log
├── treeStructure.txt
├── val_audio_acc.log
├── val_loss.log
├── val_visual_acc.log
└── visual_network_arch.txt
```
Alternatively, you can run the script  

```angular2
/CombiningDeepClassifiers/audio/call_online_fusion_train.py 
```

which trains models with different neural networks (e.g., MobileNetV3Large, ResNet18, etc.) and sub-tasks (e.g., Native, YT, ...).


## Test Models
### 1. Test Audio Network
Run the script 

```angular2
/CombiningDeepClassifiers/audio/test_audio_1D.py --data_content Native --nfft_scale 2 --log_mel --n_fold 0 --model ResNet18 --project_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/ --epochs 100 --lr 1e-4 --audio_frame_indices_dir /media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/ --extracted_wav_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/ --optimizer Adam --results_dir /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/ --data_loader_fold_type my_data_I --excluded_devices D12 --extend_duration_sec 2 --softmax --milestones 2 --milestones 4 --milestones 8
```

This script saves the tests results in the folder 
```angular2
CombiningDeepClassifiers/audio/results1D/winSize2Sec/Native/fold0/ResNet18/run1
```

Alternatively, you can run the script  

```angular2
/CombiningDeepClassifiers/audio/call_test_audio_1D.py 
```

which trains models with different neural networks (e.g., MobileNetV3Large, ResNet18, etc.) and sub-tasks (e.g., Native, YT, ...).

### 2. Test Image Network

Run the script 

```angular2
/CombiningDeepClassifiers/image/test_image.py --n_run 1 --data_content WA --model ResNet50 --n_fold 1 --visual_frames_dir /media/red/sharedFolder/Datasets/VISION/keyFrames/I/ --project_dir /media/blue/tsingalis/DevIDFusion/ --results_dir /media/blue/tsingalis/DevIDFusion/image/results/ --data_loader_fold_type my_data_I --extend_duration_sec 2 --test_batch 128
```

This script saves the tests results in the folder 
```angular2
CombiningDeepClassifiers/image/results/winSize2Sec/WA/fold1/ResNet50/run1
```

Alternatively, you can run the script  

```angular2
/CombiningDeepClassifiers/image/call_test_image.py 
```

which tests models with different neural networks (e.g., MobileNetV3Large, ResNet18, etc.)  and sub-tasks (e.g., Native, YT, ...).


### 3. Test Fusion Network

Run the script 

```angular2
/CombiningDeepClassifiers/fusion/online_fusion_test.py --results_dir_fusion  /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/fusion/results/  --results_dir_audio  /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/results1D/  --results_dir_visual /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/image/results/  --audio_frame_indices_dir  /media/red/sharedFolder/Datasets/VISION/AudioFrameIndices/I/  --project_dir  /media/red/tsingalis/gitRepositoriesCombiningDeepClassifiers/  --visual_frame_dir  /media/red/sharedFolder/Datasets/VISION/keyFrames/I/  --wav_dir  /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/audio/extractedWav/  --data_loader_fold_type  my_data_I  --extend_duration_sec  2  --data_content  Native  --n_fold  0  --reduction  --test_batch_size  128  --n_run  1  --visual_model  MobileNetV3Large  --audio_model  MobileNetV3Large  --loss_fn  ProductRuleLoss
```

This script creates accuracy results to the folder 
```angular2
CombiningDeepClassifiers/fusion/results/winSize2Sec/ProductRuleLoss/visualMobileNetV3Large_audioMobileNetV3Large/Native/fold0/run1/stats
```

with subfolders `Native`, `WA`, and `YT`, where the best model `model_best.ckpt` is saved for each `fold{i}`, with `i=0, 1, 2, 3, 4`. Each fold folder has the following structure

```angular2html
stats
├── audio_stats.pkl
├── cm_audio_model_best_epoch_26_0.9181_frame.eps
├── cm_audio_model_best_epoch_26_0.9181_frame.jpg
├── cm_audio_model_best_epoch_26_0.9181_frame_normalized.eps
├── cm_audio_model_best_epoch_26_0.9181_frame_normalized.jpg
├── cm_audio_model_best_epoch_26_0.9675_video.eps
├── cm_audio_model_best_epoch_26_0.9675_video.jpg
├── cm_audio_model_best_epoch_26_0.9675_video_normalized.eps
├── cm_audio_model_best_epoch_26_0.9675_video_normalized.jpg
├── cm_visual_model_best_epoch_26_0.7095_frame.eps
├── cm_visual_model_best_epoch_26_0.7095_frame.jpg
├── cm_visual_model_best_epoch_26_0.7095_frame_normalized.eps
├── cm_visual_model_best_epoch_26_0.7095_frame_normalized.jpg
├── cm_visual_model_best_epoch_26_0.7967_video.eps
├── cm_visual_model_best_epoch_26_0.7967_video.jpg
├── cm_visual_model_best_epoch_26_0.7967_video_normalized.eps
├── cm_visual_model_best_epoch_26_0.7967_video_normalized.jpg
├── treeStructure.txt
└── visual_stats.pkl
```
Alternatively, you can run the script  

```angular2
/CombiningDeepClassifiers/audio/call_online_fusion_test.py 
```

which tests models with different neural networks (e.g., MobileNetV3Large, ResNet18, etc.) and sub-tasks (e.g., Native, YT, ...).

## Late Fusion Models
### 1. Late Fusion without online training
To perform late fusion using the trained audio and image models, you can run the script

```angular2
/CombiningDeepClassifiers/audio/late_fusion.py --project_dir  /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/  --data_loader_fold_type  my_data_I  --extend_duration_sec  2  --n_run_audio  1  --n_run_visual  1  --excluded_devices  D12  --data_content  YT  --n_fold  4  --visual_model  MobileNetV3Small  --audio_model  MobileNetV3Large  --fusion_rule  product_rule
```

To run this script you need to test the fine-tuned models in the previous steps in order to get the test labels.

### 2. Late Fusion after online training (online fusion)

To perform late fusion using the fine-tuned (after fusion) audio and image models, you can run the script

```angular2
late_fusion_on_online_results.py --project_dir  /media/red/tsingalis/gitRepositories/CombiningDeepClassifiers/  --data_loader_fold_type  my_data_I  --extend_duration_sec  2  --n_run_audio  1  --n_run_visual  1  --excluded_devices  D12  --data_content  YT  --n_fold  4  --visual_model  MobileNetV3Small  --audio_model  MobileNetV3Large  --fusion_rule  product_rule
```
To run this script you need to test the fine-tuned models in the previous steps in order to get the test labels.

## Reference
If you use this code in your experiments please cite this work by using the following bibtex entry:

```
@article{referenceID2025,
  author    = {Ioannis Tsingalis and Constantine Kotropoulos},
  title     = {On Combining Deep Neural Network Classifiers for Source Device Identification},
  journal   = {IEEE Access},
  year      = {2025},
  publisher = {IEEE}
}
```