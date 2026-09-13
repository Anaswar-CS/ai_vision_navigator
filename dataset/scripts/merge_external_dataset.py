"""
Merge one or more downloaded Roboflow YOLO-format exports into this
project's unified dataset (see dataset/data.yaml for the target class list).

WHY THIS SCRIPT EXISTS:
Every Roboflow dataset defines its own class list and class indices in its
own data.yaml (e.g. source dataset might have class 0 = "pen", but another
source dataset might have class 0 = "box"). Blindly copying label files
across sources would silently corrupt them — a "0" in one dataset does not
mean the same thing as a "0" in another. This script re-writes each label
file's class indices to match THIS project's fixed 6-class scheme before
copying anything in.

HOW TO USE:
1. On Roboflow Universe, open a dataset relevant to one of your target
   classes (pen, pencil, spectacles, medicine_box, instrumentation_box).
2. Click "Download Dataset" -> format "YOLOv8" -> download zip -> unzip it
   locally. You'll get a folder with train/valid/test subfolders and its
   own data.yaml.
3. Decide which class(es) in that source dataset map to which of YOUR
   target classes. Not every class in the source dataset is relevant —
   e.g. a "medicine box" dataset might also have "aspirin", "ibuprofen"
   brand-specific classes; you likely want to map ALL of those to your
   single "medicine_box" class, and IGNORE anything irrelevant.
4. Run this script once per source dataset:

   python dataset/scripts/merge_external_dataset.py \
       --source /path/to/unzipped_roboflow_export \
       --map "pen=pen" "biro=pen" \
       --split train

   --map takes SOURCE_CLASS_NAME=TARGET_CLASS_NAME pairs. Any source class
   NOT listed in --map is skipped (its labels are dropped, not copied).

5. Repeat for each source dataset / split (train, valid, test) you download.
6. Once merged, run split_dataset.py if you want to re-shuffle everything,
   or just proceed straight to train_custom_model.py if you're comfortable
   with each source's own train/valid/test split.

TARGET CLASSES (fixed, see dataset/data.yaml):
    0: pen
    1: pencil
    2: spectacles
    3: medicine_box
    4: instrumentation_box
    5: computer
"""

import argparse
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"

# Roboflow's default split folder names -> this project's split folder names
SPLIT_DIR_ALIASES = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge an external Roboflow YOLO export into the unified dataset."
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to the unzipped Roboflow export (contains its own data.yaml).",
    )
    parser.add_argument(
        "--map", nargs="+", required=True,
        help=(
            "SOURCE_CLASS=TARGET_CLASS pairs, e.g. --map pen=pen biro=pen. "
            "TARGET_CLASS must be one of the 6 names in dataset/data.yaml. "
            "Source classes omitted here are skipped entirely."
        ),
    )
    parser.add_argument(
        "--split", default="all", choices=["train", "valid", "val", "test", "all"],
        help="Which split(s) to pull from the source export. Default: all available.",
    )
    parser.add_argument(
        "--prefix", default=None,
        help=(
            "Filename prefix to avoid collisions between sources, "
            "e.g. 'roboflow_penset1_'. Defaults to the source folder's name."
        ),
    )
    return parser.parse_args()


def load_class_names(data_yaml_path: Path) -> dict:
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        return {v: k for k, v in names.items()}  # name -> index
    return {name: idx for idx, name in enumerate(names)}


def main():
    args = parse_args()
    source_root = Path(args.source)
    source_yaml = source_root / "data.yaml"

    if not source_yaml.exists():
        raise FileNotFoundError(
            f"No data.yaml found at {source_yaml}. Point --source at the "
            "unzipped Roboflow export root (the folder containing data.yaml)."
        )

    source_name_to_idx = load_class_names(source_yaml)
    target_name_to_idx = load_class_names(TARGET_DATA_YAML)

    class_map = {}
    for pair in args.map:
        src_name, tgt_name = pair.split("=", 1)
        if src_name not in source_name_to_idx:
            print(f"WARNING: '{src_name}' not found in source classes {list(source_name_to_idx)}, skipping.")
            continue
        if tgt_name not in target_name_to_idx:
            raise ValueError(
                f"'{tgt_name}' is not one of this project's target classes: "
                f"{list(target_name_to_idx)}"
            )
        class_map[source_name_to_idx[src_name]] = target_name_to_idx[tgt_name]

    if not class_map:
        raise ValueError("No valid class mappings resolved — check --map against the source data.yaml.")

    print(f"Class ID remapping for this source: {class_map}")

    splits_to_process = (
        list(SPLIT_DIR_ALIASES.keys()) if args.split == "all" else [args.split]
    )

    prefix = args.prefix or (source_root.name + "_")
    total_copied = 0

    for split_dir_name in splits_to_process:
        src_images_dir = source_root / split_dir_name / "images"
        src_labels_dir = source_root / split_dir_name / "labels"
        if not src_images_dir.exists():
            continue  # this split doesn't exist in the source export

        target_split = SPLIT_DIR_ALIASES[split_dir_name]
        dest_images_dir = PROJECT_ROOT / "dataset" / "images" / target_split
        dest_labels_dir = PROJECT_ROOT / "dataset" / "labels" / target_split
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        dest_labels_dir.mkdir(parents=True, exist_ok=True)

        for label_path in src_labels_dir.glob("*.txt"):
            image_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = src_images_dir / (label_path.stem + ext)
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                print(f"WARNING: no matching image for {label_path.name}, skipping.")
                continue

            remapped_lines = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    src_class_id = int(parts[0])
                    if src_class_id not in class_map:
                        continue  # this class wasn't in --map, drop the box
                    parts[0] = str(class_map[src_class_id])
                    remapped_lines.append(" ".join(parts))

            if not remapped_lines:
                continue  # every box in this image was for an unmapped class

            new_stem = prefix + label_path.stem
            shutil.copy(image_path, dest_images_dir / (new_stem + image_path.suffix))
            with open(dest_labels_dir / (new_stem + ".txt"), "w") as f:
                f.write("\n".join(remapped_lines) + "\n")
            total_copied += 1

    print(f"\nMerged {total_copied} image/label pairs into dataset/images and dataset/labels.")
    print("Review a few merged images + labels visually before training — ")
    print("auto-remapping doesn't guarantee visual quality or relevance.")


if __name__ == "__main__":
    main()
