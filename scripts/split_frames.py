#!/usr/bin/env python3
import os
import argparse
import random
import shutil
from tqdm import tqdm

EXCLUDE_DIRS = {"train", "test", "Train", "Test", "logs"}

def split_dataset(base_dir: str, train_ratio: float = 0.8, move: bool = True):
    classes = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d)) and d not in EXCLUDE_DIRS]
    if not classes:
        print(f"No class folders under {base_dir}")
        return False

    train_root = os.path.join(base_dir, "train")
    test_root = os.path.join(base_dir, "test")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    random.seed(42)
    total_train = total_test = 0

    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        # video-level subdirectories (each contains frames)
        video_dirs = [d for d in os.listdir(cls_dir)
                      if os.path.isdir(os.path.join(cls_dir, d))]
        if not video_dirs:
            # If frames are flat in the class dir, treat as one pseudo-video
            video_dirs = [""]

        random.shuffle(video_dirs)
        split_idx = int(len(video_dirs) * train_ratio)
        train_videos = video_dirs[:split_idx]
        test_videos = video_dirs[split_idx:]

        train_cls_dir = os.path.join(train_root, cls)
        test_cls_dir = os.path.join(test_root, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        def iter_frames(video_subdir):
            if video_subdir:
                src_dir = os.path.join(cls_dir, video_subdir)
            else:
                src_dir = cls_dir
            for fname in os.listdir(src_dir):
                if fname.lower().endswith(".png"):
                    yield os.path.join(src_dir, fname), video_subdir

        # Move/copy frames with prefix to avoid name collisions
        for v in tqdm(train_videos, desc=f"Train {cls}"):
            prefix = (v + "_") if v else ""
            for src_path, subdir in iter_frames(v):
                dst_name = prefix + os.path.basename(src_path)
                dst_path = os.path.join(train_cls_dir, dst_name)
                if move:
                    os.replace(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                total_train += 1

        for v in tqdm(test_videos, desc=f"Test {cls}"):
            prefix = (v + "_") if v else ""
            for src_path, subdir in iter_frames(v):
                dst_name = prefix + os.path.basename(src_path)
                dst_path = os.path.join(test_cls_dir, dst_name)
                if move:
                    os.replace(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                total_test += 1

    print(f"Done. Moved/copied {total_train} train frames and {total_test} test frames.")
    return True


def main():
    p = argparse.ArgumentParser(description="Split existing frame dataset into train/test")
    p.add_argument("--base_dir", type=str, default="data",
                  help="Base directory containing class folders with frames (and video subfolders)")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--copy", action="store_true", help="Copy instead of move (uses more disk space)")
    args = p.parse_args()

    split_dataset(args.base_dir, args.train_ratio, move=(not args.copy))


if __name__ == "__main__":
    main()
