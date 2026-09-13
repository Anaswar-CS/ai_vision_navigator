"""
Extract still frames from a panning video for dataset collection.

Recommended workflow per class (pen, pencil, spectacles, medicine_box,
instrumentation_box, computer):
  1. Record a ~30-60 second video panning/rotating around the object,
     varying background, lighting, and distance as you go.
  2. Run this script to pull frames out at a fixed interval.
  3. Manually cull blurry/duplicate frames before labeling.

Usage:
    python dataset/scripts/extract_frames.py --video pen_session1.mp4 \
        --out dataset/raw_captures/pen --every-n 8
"""

import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video for labeling.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--out", required=True, help="Output directory for extracted frames.")
    parser.add_argument(
        "--every-n", type=int, default=8,
        help="Save 1 out of every N frames (default 8). Lower = more frames, more redundancy.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=400,
        help="Safety cap so one long video doesn't flood the dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved = 0
    stem = video_path.stem

    while saved < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.every_n == 0:
            out_path = out_dir / f"{stem}_frame{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path.name} -> {out_dir}")
    print("Next: cull blurry/duplicate frames, then label with Roboflow/CVAT/Label Studio.")


if __name__ == "__main__":
    main()
