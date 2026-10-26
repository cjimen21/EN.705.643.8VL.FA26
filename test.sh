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