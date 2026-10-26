"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the required tasks and logs outputs.
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

'''Import wandb for experiment logging per requirements.'''

import wandb

from video_datasets import VideoDataset, load_dataset, dataset_split
from utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from models import LRCN
from train import train
from test import test, get_test_report, get_confusion_matrix


def args_parser():
    """
    Parse command-line arguments for configuring video classification training or evaluation.
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.15, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('-p', '--pretrained', action='store_true', default=True, help='Use pretrained 2D CNN backbone')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')
    parser.add_argument('-ic', '--in_channels', type=int, default=5, help='Number of input channels (3 for RGB, 5 for RGB+Flow)')

    parser.add_argument('-m', '--mode', type=str, default='train', help="Either 'train' or 'eval'", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')

    # FIX ADDED: Command-line flags for Weights & Biases experiment tracking
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='UCF50-Action-Recognition', help='W&B project name')

    args = parser.parse_args()
    return args


def main(args):
    """
    Main function to execute training or evaluation based on parsed arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    in_channels = args.in_channels

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type, in_channels=in_channels)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Initialize the model (LRCN)
    model = LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
                 n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone,
                 in_channels=in_channels)

    if mode == 'train':
        # FIX ADDED: Initialize Weights & Biases run if requested
        if args.use_wandb:
            wandb.init(project=args.wandb_project, config=vars(args))

        # Load dataset and split into train/validation/test (using group-aware split)
        vid_dataset, label_dict = load_dataset(frame_dir)
        tr_split, val_split, ts_split = dataset_split(vid_dataset, tr_size, ts_size)

        # Save the splits for reproducibility and later evaluation
        splits = {'train': np.array(tr_split, dtype=object),
                  'val': np.array(val_split, dtype=object),
                  'test': np.array(ts_split, dtype=object)}
        np.save('./splits.npy', splits)

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms)
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, model_type)

        '''Changed loss function reduction from 'sum' to 'mean' as 'sum' reduction scales loss with batch size rather 
        than averaging, which distorts effective gradient magnitudes when batch sizes vary.'''
        loss_func = nn.CrossEntropyLoss(reduction='mean')

        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        os.makedirs("./models", exist_ok=True)
        optim_model_dir = './models'

        # Main training procedure
        model.to(device)
        model, loss_hist, acc_hist = train(
            dataloaders, model, loss_func, opt, lr_scheduler,
            device, optim_model_dir, n_epochs=n_epochs, use_wandb=args.use_wandb
        )

        if args.use_wandb:
            wandb.finish()

    elif mode == 'eval':
        # Load saved dataset splits
        splits = np.load('./splits.npy', allow_pickle=True)
        ts_split = splits.item()['test']
        ts_split = [(sample[0], int(sample[1])) for sample in ts_split]

        # Create PyTorch Dataset and DataLoader for the test set
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        dataloaders = test_dloaders(ts_dataset, batch_size, model_type)

        # Load the trained model checkpoint
        if not args.ckpt:
            raise ValueError("Must provide `--ckpt` path when running in 'eval' mode.")

        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.to(device)
        targets, outputs, accuracy = test(model, dataloaders['test'], device)

        print('The overall test accuracy is {:.4f}%.'.format(100 * accuracy))

    else:
        raise ValueError('The mode argument must be either "train" or "eval".')


if __name__ == "__main__":
    args = args_parser()
    main(args)
