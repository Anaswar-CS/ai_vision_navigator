"""
Stratified train/val/test split for the custom object dataset.

Assumes you've already labeled images in YOLO format and have matching
pairs like:
    some_folder/pen_0001.jpg
    some_folder/pen_0001.txt

Point --labeled-dir at the folder containing ALL labeled image+txt pairs
(across all six classes mixed together is fine — this script only moves
files, it doesn't need to know which class is in which image).

Usage:
    python dataset/scripts/split_dataset.py --labeled-dir path/to/labeled \
        --train 0.7 --val 0.2 --test 0.1
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split labeled data into train/val/test.")
    parser.add_argument("--labeled-dir", required=True, help="Folder with image+label pairs.")
    parser.add_argument("--dataset-root", default="dataset", help="Target dataset/ root.")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "Splits must sum to 1.0"

    labeled_dir = Path(args.labeled_dir)
    dataset_root = Path(args.dataset_root)

    image_paths = sorted(
        p for p in labeled_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    pairs = []
    for img_path in image_paths:
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"WARNING: no label file for {img_path.name}, skipping.")
            continue
        pairs.append((img_path, label_path))

    if not pairs:
        raise RuntimeError(f"No labeled image/txt pairs found in {labeled_dir}")

    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        img_dest = dataset_root / "images" / split_name
        label_dest = dataset_root / "labels" / split_name
        img_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_pairs:
            shutil.copy(img_path, img_dest / img_path.name)
            shutil.copy(label_path, label_dest / label_path.name)

        print(f"{split_name}: {len(split_pairs)} images")

    print(f"\nTotal: {n} labeled pairs split {args.train}/{args.val}/{args.test}")


if __name__ == "__main__":
    main()
