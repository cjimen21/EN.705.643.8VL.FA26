"""
Module: preprocess.py

Extracts raw .avi videos from UCF50 into folders of JPEG frames using utils.get_frames
and utils.store_frames.
"""

import os
from tqdm import tqdm
from utils import get_frames, store_frames


def extract_all_frames(src_dir, dst_dir, n_frames=16):
    """Extracts uniformly sampled JPEG frames from raw .avi files."""
    os.makedirs(dst_dir, exist_ok=True)

    for cat in sorted(os.listdir(src_dir)):
        cat_path = os.path.join(src_dir, cat)
        if not os.path.isdir(cat_path):
            continue

        dst_cat_path = os.path.join(dst_dir, cat)
        os.makedirs(dst_cat_path, exist_ok=True)

        print(f"Extracting frames for category: {cat}")
        avi_files = [f for f in os.listdir(cat_path) if f.endswith(".avi")]

        for avi_file in tqdm(avi_files):
            vid_name = os.path.splitext(avi_file)[0]
            vid_path = os.path.join(cat_path, avi_file)
            store_path = os.path.join(dst_cat_path, vid_name)

            if os.path.exists(store_path) and len(os.listdir(store_path)) > 0:
                continue

            os.makedirs(store_path, exist_ok=True)
            frames, _ = get_frames(vid_path, n_frames=n_frames)
            if len(frames) > 0:
                store_frames(frames, store_path)


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "UCF50"
    dst = sys.argv[2] if len(sys.argv) > 2 else "UCF50_frames"
    extract_all_frames(src, dst, n_frames=16)