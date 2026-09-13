# Custom Object Dataset — AI Vision Navigator

This dataset covers the six classes the pretrained YOLO model can't reliably
detect: **pen, pencil, spectacles, medicine_box, instrumentation_box, computer**.

## 0. Decide these definitions BEFORE shooting anything

- **"computer"**: recommended definition = a desktop tower (CPU case), optionally
  with monitor attached, as distinct from a laptop's clamshell form. Lock this in
  with whoever specced the project — inconsistent definitions ruin labeling.
- **"instrumentation_box"**: this term is ambiguous (geometry box? electronics kit?
  lab instrument case?). Get the exact intended object before shooting.

Once decided, write the definition at the top of this file so every annotator
follows the same rule.

## 1. Folder layout

```
dataset/
├── data.yaml                  # YOLO class config — do not reorder class list after labeling starts
├── raw_captures/              # unlabeled photos/frames, staged per class before labeling
│   ├── pen/
│   ├── pencil/
│   ├── spectacles/
│   ├── medicine_box/
│   ├── instrumentation_box/
│   └── computer/
├── images/{train,val,test}    # final labeled images, populated by split_dataset.py
├── labels/{train,val,test}    # matching YOLO-format .txt label files
└── scripts/
    ├── extract_frames.py      # pull frames out of panning videos
    ├── split_dataset.py       # stratified train/val/test split
    └── train_custom_model.py  # fine-tune YOLOv8n on the six classes
```

## 2. Per-class targets and public data reality (checked Sept 2026)

| Class | Target images | Public data available? |
|---|---|---|
| pen | 200–300 | Yes — several Roboflow Universe datasets, one with ~700 images |
| pencil | 200–300 | Yes — a few small-to-medium Roboflow datasets |
| spectacles | 200–300 | Thin — one usable dataset, small; expect to supplement with self-captured photos |
| medicine_box | 200-300 | Yes - decent options ("medicine_kor" ~384 images, "MedicineBoxes", "medicine_bottle" ~308 images), mostly generic pill bottles/boxes |
| instrumentation_box | 80 | **Synthetic Placeholder Only.** instrumentation_box is trained on synthetic placeholder shapes only. It detects plain-background rectangles, not real instrumentation boxes. This is permanent, not provisional — there is no planned upgrade path to real photos for this class. |
| computer | 200-300 | Partial - COCO/Open Images have generic desktop/monitor imagery, but no clean "desktop tower distinct from laptop" class; expect to supplement |

Search Roboflow Universe (universe.roboflow.com) yourself for each class name
before starting — new datasets get added constantly and these findings will
age. Check each dataset's license (CC BY 4.0 requires attribution; a couple
found were Public Domain) before using it.

## 3. Merging public datasets into this project (public-data-first workflow)

Since every Roboflow dataset defines its own class list and index order, you
can't just copy files across — a "class 0" in one dataset isn't the same
object as "class 0" in another. Use the merge script to remap safely:

1. On Roboflow Universe, open a dataset for one of your target classes.
2. Click **Download Dataset → format YOLOv8** → download and unzip locally.
3. Look at that dataset's own `data.yaml` to see its class names.
4. Run the merge script, mapping its class name(s) to your target class:
   ```
   python dataset/scripts/merge_external_dataset.py \
       --source /path/to/unzipped_export \
       --map "pen=pen" "biro=pen"
   ```
   Any source class you don't list in `--map` is dropped, not copied — so
   irrelevant classes in a multi-class source dataset won't pollute yours.
5. Repeat per source dataset, once per class you're sourcing publicly.
6. For **instrumentation_box** (and to shore up **spectacles** and
   **computer**), still follow the self-capture workflow in section 4 below
   — there's no public substitute.
7. Once merged, spot-check a sample of the merged images + labels visually
   (open a few in an annotation tool or plot the boxes) before training —
   auto-remapping guarantees correct indices, not correct or relevant boxes.
8. Train:
   ```
   python dataset/scripts/train_custom_model.py --auto-copy
   ```



## 4. Self-capture workflow (recommended to supplement spectacles and computer)

1. **Capture with the actual deployed camera**, not a phone, so training data
   matches inference-time image quality and field of view.
2. **Record short panning videos** per class instead of individual stills:
   ```
   python dataset/scripts/extract_frames.py --video pen_session1.mp4 \
       --out dataset/raw_captures/pen --every-n 8
   ```
3. **Manually cull** blurry or near-duplicate frames from `raw_captures/<class>/`.
4. **Label** using Roboflow, CVAT, or Label Studio. Auto-label a first pass with
   pretrained YOLO or a zero-shot model (e.g. Grounding DINO) to draft boxes,
   then correct manually — much faster than labeling from scratch. Export in
   YOLO format (image + matching `.txt` file with `class_id x_center y_center
   width height`, all normalized 0-1).
5. **Split** into train/val/test:
   ```
   python dataset/scripts/split_dataset.py --labeled-dir path/to/labeled_output
   ```
6. **Train**:
   ```
   python dataset/scripts/train_custom_model.py --auto-copy
   ```
   This fine-tunes `yolov8n.pt` (CPU-friendly, matches the target hardware) and,
   with `--auto-copy`, drops the result straight into `models/custom/best.pt`,
   which the Django app already checks for (see `vision/detector.py`).
7. **Validate** against a held-out set of real-world images taken separately
   from the training data — not just the val split from the same sessions.

Across every self-captured class, vary background, lighting, distance, angle,
multiple physical instances of the object, partial occlusion, and a few
hard-negative backgrounds (similar-looking objects that aren't the target).

## 5. Class index stability

`data.yaml` fixes the class order:

```
0: pen
1: pencil
2: spectacles
3: medicine_box
4: instrumentation_box
5: computer
```

Label files store the index, not the name — if this order changes after
labeling starts, existing labels silently point to the wrong class. If you
add a new class later, append it at the end; don't reorder.

## 6. This work is parallel, not blocking

Everything else in the app (UI, camera pipeline, voice, navigation) can be
built and tested against the pretrained YOLO model in the meantime. Swapping
in `models/custom/best.pt` once it's trained doesn't require touching the
rest of the pipeline — the detector already has a fallback/active-model
reporting mechanism built in for exactly this handoff.
