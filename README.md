# Video Classification with UCF50 using Dual-Stream LRCN

This project implements an end-to-end video classification pipeline on the UCF50 dataset. It leverages a Slow Fusion Long-term Recurrent Convolutional Network (LRCN) architecture that extracts spatial features (RGB) and temporal dynamics (Optical Flow) via fine-tuned **ResNet-50** backbones and models sequence progression using a recurrent network (LSTM).

---

## Table of Contents

- [Model Architecture & Specifications](#model-architecture--specifications)
- [Performance Summary](#performance-summary)
- [Model Improvements & Evolution](#model-improvements--evolution)
- [Data Augmentation Strategy](#data-augmentation-strategy)
- [Key Bug Fixes & Technical Lessons](#key-bug-fixes--technical-lessons)
- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Preprocessing and Frame Extraction](#preprocessing-and-frame-extraction)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [Customization and Hyperparameters](#customization-and-hyperparameters)
- [Summary of Steps](#summary-of-steps)

---

## Model Architecture & Specifications

* **Backbone Architecture:** Dual-stream ResNet-50 (`ResNet50_Weights.DEFAULT` / ImageNet-1K V2) for RGB and Optical Flow.
* **Fine-Tuning Strategy:** Fine-tuned `layer4` on both backbones while freezing early layers to preserve pre-trained spatial/motion feature extractors.
* **Feature Dimension:** 2048 features per stream (4096-dimensional concatenated slow fusion feature vector per frame).
* **Frame Sampling:** 8 frames per video clip (`--fr_per_vid 8`) for efficient temporal coverage.
* **Input Channels:** 
  * RGB Stream: 3 channels
  * Optical Flow Stream: 5 channels (`--in_channels 5`)

---

## Performance Summary

| Metric | Result |
| :--- | :--- |
| **Final Test Accuracy (Held-out Set)** | **84.8039%** |
| **Peak Validation Accuracy** | **85.92%** |
| **Final Epoch Val Accuracy** | **85.34%** |
| **Final Training Loss** | **~0.024** |
| **Final Validation Loss** | **~0.568** |

---

## Model Improvements & Evolution

1. **Backbone Upgrade (ResNet-34 traded for ResNet-50):**  
   Transitioning from a ResNet-34 backbone to a dual-stream ResNet-50 significantly expanded the model's feature extraction capability. Moving to `ResNet50_Weights.DEFAULT` (ImageNet-1K V2) provided richer representations for spatial features and complex motion dynamics.
2. **Selective Fine-Tuning (`layer4` Unfreezing):**  
   Rather than fine-tuning the entire backbone (which risks overfitting on limited data) or keeping it entirely frozen, we unfroze `layer4` across both streams. This allowed high-level domain adaptation while preserving general low-level vision weights.
3. **Temporal Sampling Optimization:**  
   Reduced frame sampling from 16 frames to 8 uniform frames per clip (`--fr_per_vid 8`). This halved computational overhead per epoch (~13.8s/it down to ~7.5s/it) with zero degradation in validation accuracy, keeping training runs within strict time limits.

---

## Data Augmentation Strategy

To prevent overfitting and handle spatial variance across video clips, spatial transformations are applied consistently across all frames of a given sequence:

* **Random Horizontal Flips:** Applied uniformly across spatial RGB frames and optical flow maps to maintain directional alignment.
* **Spatial Cropping & Resizing:** Clips are resized and randomly cropped to standardized spatial dimensions (224 by 224), ensuring input compatibility with ResNet-50.
* **Color & Intensity Jittering:** Light jitter applied to the RGB stream to improve robustness against lighting variations across different video source clips.
* **Optical Flow Normalization:** Flow vectors (5 channels) are normalized to stabilize magnitude distribution across movement speeds.

---

## Key Bug Fixes & Technical Lessons
* **Group Split Data Leakage Protection:**  
  Enforced group-aware partitioning via `splits.npy` (`GroupShuffleSplit`). This ensured that different clips originating from the same video source group were never split across both training and evaluation subsets.

---

## Dataset Preparation

### Step 0: Download and Unzip Dataset

1. **Download Dataset:**  
   Download the UCF50 dataset from the official source: `https://www.crcv.ucf.edu/data/UCF50.php`. The dataset contains videos across 50 action categories, grouped into 25 groups.

2. **Unzip and Organize:**  
   Unzip the dataset into the expected directory structure:

```text
UCF50/
├── BaseballPitch/
├── Basketball/
├── Biking/
│   ...
└── YoYo/
```

---

## Environment Setup

1. **Python Version:**  
   Python 3.10+ recommended (compatible with 3.7+).

2. **Dependencies:**  
   Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### Key Libraries
- **PyTorch** & **torchvision**
- **OpenCV (`opencv-python`)**
- **scikit-learn**
- **tqdm**
- **numpy**
- **Pillow**
- **Weights & Biases (`wandb`)**

---

## Preprocessing and Frame Extraction

Before training, raw video files are converted into frame sequences and precomputed optical flow maps.

To preprocess the raw dataset, run:

```bash
python preprocess.py
```

---

## Training the Model

### Step 1: Run Training

Training execution is managed via `train.sh`:

```bash
#!/bin/bash
python run.py \
    --frame_dir UCF50_frames \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --in_channels 5 \
    --n_classes 50 \
    --fr_per_vid 8 \
    --batch_size 4 \
    --dropout 0.1 \
    --learning_rate 3e-5 \
    --n_epochs 15 \
    --mode 'train' \
    --use_wandb \
    --wandb_project "Assignment_4"
```

Execute training from your terminal:

```bash
bash train.sh
```

---

## Testing and Evaluation

### Step 2: Run Evaluation

Update `test.sh` to load the best saved model checkpoint:

```bash
#!/bin/bash
python run.py \
    --frame_dir UCF50_frames \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --in_channels 5 \
    --n_classes 50 \
    --fr_per_vid 8 \
    --batch_size 4 \
    --mode 'eval' \
    --ckpt ./models/best_model_wts.pt
```

Launch evaluation:

```bash
bash test.sh
```

---

## Customization and Hyperparameters

| Flag | Description | Default / Recommended |
| :--- | :--- | :--- |
| `--frame_dir` | Directory containing processed frames | `UCF50_frames` |
| `--fr_per_vid` | Number of frames sampled per video | `8` |
| `--cnn_backbone` | Feature extraction backbone | `resnet50` |
| `--in_channels` | Input channels (3 for RGB, 5 for Flow stream) | `5` |
| `--batch_size` | Training batch size | `4` |
| `--learning_rate` | Initial learning rate | `3e-5` |
| `--mode` | Pipeline mode (`train` or `eval`) | `train` |
| `--ckpt` | Checkpoint file path for evaluation | `./models/best_model_wts.pt` |

---

## Summary of Steps

1. **Dataset Setup & Preprocessing:** Download UCF50 and extract frames/flow maps into `UCF50_frames/`.
2. **Train Model:** Run `bash train.sh` to train the ResNet-50 dual-stream LRCN model and save `models/best_model_wts.pt`.
3. **Evaluate:** Run `bash test.sh` to measure final accuracy on the held-out test dataset.

Happy Training!