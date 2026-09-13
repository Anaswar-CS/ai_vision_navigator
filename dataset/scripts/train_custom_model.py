"""
Fine-tune a lightweight YOLO model on the six custom classes:
pen, pencil, spectacles, medicine_box, instrumentation_box, computer.

Run from the project root's venv, after the dataset/ folder has been
populated and labeled (see dataset/README.md).

Usage:
    python dataset/scripts/train_custom_model.py
    python dataset/scripts/train_custom_model.py --epochs 100 --imgsz 640

Output:
    Trained weights land in runs/detect/train*/weights/best.pt
    Copy that file to models/custom/best.pt to activate it in the app
    (see vision/detector.py's MODEL_PATH fallback logic).
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
CUSTOM_MODEL_DEST = PROJECT_ROOT / "models" / "custom" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on custom objects.")
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Starting checkpoint. Use the nano model — this is a CPU-target project.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="'cpu' or a CUDA index if available.")
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of layers to freeze (e.g. 10 for backbone). Default 10 for frozen backbone training.",
    )
    parser.add_argument(
        "--auto-copy",
        action="store_true",
        help="Copy the resulting best.pt straight into models/custom/ after training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_YAML}. Populate dataset/images and dataset/labels "
            "and confirm dataset/data.yaml is in place before training."
        )

    model = YOLO(args.base_model)

    train_kwargs = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": 42,
        "deterministic": True,
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": True,
        "patience": 15,          # early stop if val metrics plateau
        "workers": 2,            # keep worker count low — target machine has 8GB RAM
        "project": str(PROJECT_ROOT / "runs" / "detect"),
        "name": "custom_objects",
        "exist_ok": True,
    }
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    results = model.train(**train_kwargs)

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")

    if args.auto_copy:
        CUSTOM_MODEL_DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, CUSTOM_MODEL_DEST)
        print(f"Copied to {CUSTOM_MODEL_DEST} — restart the Django server to pick it up.")
    else:
        print(f"To activate: copy {best_weights} -> {CUSTOM_MODEL_DEST}")


if __name__ == "__main__":
    main()
