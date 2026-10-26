"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset using group-based splitting, and custom
collate functions for handling variable-length video sequences.
"""

import os
import re
import glob
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import GroupShuffleSplit


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.

    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame image (e.g., resizing, normalization).
        use_optical_flow (bool, optional): If True, computes Farneback optical flow and concatenates it to create a 5-channel tensor (RGB + Flow). Default is True.
    """

    def __init__(self, vid_dataset, fr_per_vid, transforms=None, use_optical_flow=True):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.use_optical_flow = use_optical_flow

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def _compute_optical_flow(self, pil_imgs):
        """
        Compute Farneback Optical Flow between consecutive frames.
        Returns a tensor of shape (T, 2, H, W).
        """
        flows = []
        cv_imgs = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_imgs]
        grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in cv_imgs]

        for i in range(len(grays)):
            if i == 0:
                # For first frame, compute flow relative to next frame if available, else zero flow
                if len(grays) > 1:
                    flow = cv2.calcOpticalFlowFarneback(grays[0], grays[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                else:
                    flow = np.zeros((grays[0].shape[0], grays[0].shape[1], 2), dtype=np.float32)
            else:
                flow = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Normalize flow values to range roughly [-1, 1]
            flow = np.clip(flow / 15.0, -1.0, 1.0)
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # Shape: (2, H, W)
            flows.append(flow_tensor)

        return torch.stack(flows)  # Shape: (T, 2, H, W)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        optionally compute optical flow, and return stacked tensor along with its label.

        Returns:
            tuple: (frames_tensor, label) where frames_tensor is shape (T, 5, H, W) if optical flow is enabled.
        """
        # Get all JPEG frame paths from the video directory, sort chronologically, and select up to fr_per_vid frames
        fr_paths = sorted(glob.glob(os.path.join(self.dataset[idx][0], '*.jpg')))
        fr_paths = fr_paths[:self.fpv]

        if len(fr_paths) == 0:
            return torch.empty(0), self.dataset[idx][1]

        # Open images using PIL
        fr_imgs = [Image.open(fr_path).convert('RGB') for fr_path in fr_paths]
        fr_label = self.dataset[idx][1]

        # Compute optical flow if enabled before applying image transforms
        flow_tensor = None
        if self.use_optical_flow:
            flow_tensor = self._compute_optical_flow(fr_imgs)

        # Apply spatial data transforms (resizing, augmentation, normalization) to RGB images
        if self.transforms:
            rgb_tensors = [self.transforms(fr_img) for fr_img in fr_imgs]
        else:
            to_tensor = transforms.ToTensor()
            rgb_tensors = [to_tensor(fr_img) for fr_img in fr_imgs]

        rgb_stack = torch.stack(rgb_tensors)  # Shape: (T, 3, H, W)

        if self.use_optical_flow and flow_tensor is not None:
            # Resize flow maps to match transformed image dimensions if necessary
            if flow_tensor.shape[-2:] != rgb_stack.shape[-2:]:
                resize_flow = transforms.Resize(rgb_stack.shape[-2:])
                flow_tensor = resize_flow(flow_tensor)

            # Concatenate 3 RGB channels with 2 Optical Flow channels along channel dimension (dim=1)
            frames_tensor = torch.cat([rgb_stack, flow_tensor], dim=1)  # Shape: (T, 5, H, W)
        else:
            frames_tensor = rgb_stack

        return frames_tensor, fr_label


'''
This was the origin of the data leak; the function get_group_id is a new one that overcomes the issue where
videos were not split by group id. This was problematic because it allowed data leakage of identical videos 
between training and testing datasets. Separating them by group id corrects this issue.
'''


def get_group_id(vid_path):
    """
    Extract a unique group ID from a video folder path.
    UCF50 paths typically look like: .../Biking/v_Biking_g01_c01

    Combining class name + group tag (e.g. 'Biking_g01') guarantees unique
    group identification across all 50 categories.
    """
    folder_name = os.path.basename(vid_path)
    # Extract pattern like 'g01', 'g02', etc.
    match = re.search(r'g\d+', folder_name)
    if match:
        group_tag = match.group(0)
        # Combine parent class folder with group tag for uniqueness across classes
        class_name = os.path.basename(os.path.dirname(vid_path))
        return f"{class_name}_{group_tag}"

    # Fallback if no 'gXX' pattern is found
    return folder_name


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.

    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.

    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.

    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))
                  if os.path.isdir(os.path.join(frame_dir, vid_cat))}
    vid_dataset = {}
    print('Loading video dataset....')

    for vid_cat in tqdm(sorted(label_dict.keys())):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            if os.path.isdir(vid_path):
                vid_dataset[vid_path] = label_dict[vid_cat]

    return vid_dataset, label_dict


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    """
    Split the dataset into training, validation, and test sets using GroupShuffleSplit
    to prevent data leakage across UCF50 video groups.

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the data to use for training (e.g., 0.70).
        ts_ratio (float): Proportion of the data to use for testing (e.g., 0.15).
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """
    vid_paths = np.array(list(vid_dataset.keys()))
    vid_labels = np.array(list(vid_dataset.values()))

    # Extract group IDs for each video sample
    groups = np.array([get_group_id(p) for p in vid_paths])

    print('Splitting train/validation/test datasets with GroupShuffleSplit....')

    # Step 1: Separate Test set using GroupShuffleSplit
    ts_splitter = GroupShuffleSplit(n_splits=1, test_size=ts_ratio, random_state=seed)
    tr_val_idx, ts_idx = next(ts_splitter.split(vid_paths, vid_labels, groups=groups))

    ts_paths, ts_labels = vid_paths[ts_idx], vid_labels[ts_idx]
    tr_val_paths, tr_val_labels = vid_paths[tr_val_idx], vid_labels[tr_val_idx]
    tr_val_groups = groups[tr_val_idx]

    ts_dataset = [(ts_path, ts_label) for ts_path, ts_label in zip(ts_paths, ts_labels)]

    # Step 2: Separate Train and Validation sets from the remaining data
    val_ratio = 1.0 - tr_ratio - ts_ratio

    val_wt = val_ratio / (tr_ratio + val_ratio)

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_wt, random_state=seed)
    tr_idx, val_idx = next(val_splitter.split(tr_val_paths, tr_val_labels, groups=tr_val_groups))

    tr_paths, tr_labels = tr_val_paths[tr_idx], tr_val_labels[tr_idx]
    val_paths, val_labels = tr_val_paths[val_idx], tr_val_labels[val_idx]

    tr_dataset = [(tr_path, tr_label) for tr_path, tr_label in zip(tr_paths, tr_labels)]
    val_dataset = [(val_path, val_label) for val_path, val_label in zip(val_paths, val_labels)]

    # Verification log
    tr_groups = set(groups[tr_val_idx[tr_idx]])
    val_groups = set(groups[tr_val_idx[val_idx]])
    ts_groups = set(groups[ts_idx])

    assert tr_groups.isdisjoint(val_groups), "Leak detected between Train and Val!"
    assert tr_groups.isdisjoint(ts_groups), "Leak detected between Train and Test!"
    assert val_groups.isdisjoint(ts_groups), "Leak detected between Val and Test!"
    print(
        f"Data leak resolved: {len(tr_groups)} Train, {len(val_groups)} Val, and {len(ts_groups)} Test distinct groups.")

    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.
    """
    imgs_batch, label_batch = list(zip(*batch))

    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)

    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    labels_tensor = torch.tensor(label_batch)

    return padded_imgs, labels_tensor