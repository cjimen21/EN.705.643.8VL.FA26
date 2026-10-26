"""
Module: extract_flow.py

Pre-computes dense Farneback optical flow between consecutive RGB frames
for each video directory and saves flow maps as .npy files.
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


def compute_dense_flow(frame_dir):
    """
    Computes Farneback Optical Flow for all frame folders in frame_dir.
    Flows are saved as 2-channel float16 numpy arrays in each video directory.
    """
    for vid_cat in sorted(os.listdir(frame_dir)):
        cat_path = os.path.join(frame_dir, vid_cat)
        if not os.path.isdir(cat_path):
            continue

        print(f"Extracting optical flow for category: {vid_cat}")
        for vid_folder in tqdm(os.listdir(cat_path)):
            vid_path = os.path.join(cat_path, vid_folder)
            if not os.path.isdir(vid_path):
                continue

            frame_paths = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
            if len(frame_paths) < 2:
                continue

            prev_gray = None
            for idx, fpath in enumerate(frame_paths):
                img = cv2.imread(fpath)

                # Guard check against corrupt or unreadable image files
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    # Calculate Farneback dense optical flow (2 channels: u, v)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                    # Save as float16 .npy file
                    flow_save_path = os.path.join(vid_path, f"flow_{idx - 1:04d}.npy")
                    np.save(flow_save_path, flow.astype(np.float16))

                prev_gray = gray


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "UCF50_frames"
    compute_dense_flow(data_dir)