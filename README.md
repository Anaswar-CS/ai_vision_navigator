# AI Vision Navigator

Real-time object detection, approximate distance estimation, voice search,
object tracking, and voice-guided navigation — built to run entirely on a
CPU-only laptop (Intel i5, 8 GB RAM, no dedicated GPU).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Requirements](#software-requirements)
5. [Installation](#installation)
6. [Model Installation](#model-installation)
7. [Camera Permissions](#camera-permissions)
8. [Running the Server](#running-the-server)
9. [Calibration](#calibration)
10. [Voice Commands](#voice-commands)
11. [Supported Objects](#supported-objects)
12. [Architecture](#architecture)
13. [Performance](#performance)
14. [Limitations](#limitations)
15. [Troubleshooting](#troubleshooting)
16. [Future Improvements](#future-improvements)

---

## Project Overview

AI Vision Navigator turns a laptop's built-in webcam into a lightweight
vision assistant. Open the site, click **Capture Video**, and ask "Where is
my mobile?" — the app detects the object, estimates its approximate
distance and left/right/center position, locks onto it, and gives spoken
guidance as you move ("You're getting closer", "Turn slightly left", etc).

The entire pipeline — detection, depth, distance, tracking, navigation — is
designed to run on **CPU only**, with no GPU, no cloud AI calls, and no
video ever stored to disk.

## Features

- Real-time object detection (YOLOv8-nano, CPU inference)
- Approximate monocular distance estimation (MiDaS-small + calibration,
  with a known-object-size fallback)
- Left / Center / Right positioning
- Voice search ("Where is my mobile?") via the browser's native
  SpeechRecognition API — no external LLM required
- Spoken guidance via the browser's native SpeechSynthesis API
- Lightweight IoU-based object tracking with stable IDs
- Movement analysis: getting closer / moving away / turn left / turn right
- Target-lost grace period with automatic reacquisition
- Voice cooldown so guidance doesn't repeat every frame
- Calibration page to map raw depth values to real-world meters
- Test Image mode for debugging without a live camera
- Automatic performance backoff if the CPU can't keep up
- Developer Mode performance panel
- Accessible UI: large buttons, keyboard shortcuts, high contrast

## Hardware Requirements

**Minimum target (fully supported):**

```
Intel Core i5
8 GB RAM
SSD
Integrated graphics (no dedicated GPU required)
720p or 480p webcam
Windows 10/11 (also runs on macOS/Linux)
```

The application explicitly avoids assuming a dedicated GPU, 16+ GB of RAM,
or cloud compute.

## Software Requirements

- Python 3.10+ (3.11 recommended)
- pip
- A modern browser with camera + microphone support (Chrome or Edge
  recommended for the best SpeechRecognition support)

## Installation

```bash
# 1. Clone/extract the project, then enter the folder
cd ai_vision_navigator

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template and adjust if needed
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux

# 5. Apply database migrations
python manage.py migrate
```

> **Note on `torch`/`ultralytics` install size:** these packages are a few
> hundred MB. On a slow connection, installation may take several minutes.
> The app still boots and serves pages even before they finish installing
> (Django itself has zero dependency on them) — but detection and depth
> will report "unavailable" until they're present.

## Model Installation

The app uses a **pretrained lightweight YOLO model** by default
(`yolov8n.pt`), which `ultralytics` downloads automatically on first use.

For objects a standard COCO-pretrained model cannot recognize —
**spectacles, pencil, pen, medicine box, instrumentation box** — train and
place a custom YOLO model at:

```
models/custom/best.pt
```

If this file exists, it is used automatically and takes priority over the
pretrained model. If it doesn't exist, the app clearly reports (in the
Developer Panel and Test Image page) that it has fallen back to the
pretrained model.

The MiDaS-small depth model is downloaded automatically via `torch.hub` on
first use (requires internet access once). If it cannot be downloaded, the
app automatically falls back to known-object-size distance estimation
instead of crashing.

## Camera Permissions

The browser will prompt for camera access when you click **Capture Video**.
No audio is requested. If you deny access, the app shows:
_"Camera access was denied. Please allow camera access in your browser
settings."_ — re-enable it from your browser's site settings and reload.

## Running the Server

For the live camera page to work, WebSockets must be served over ASGI.
This project uses `daphne` (already listed first in `INSTALLED_APPS`), so
the standard command works out of the box:

```bash
python manage.py runserver
```

Open your browser to:

```
http://127.0.0.1:8000/
```

## Calibration

A plain RGB webcam cannot measure physical distance directly. Visit
`/calibrate/` to improve accuracy:

1. Place an object at a known distance (e.g. exactly 1.0 m).
2. Note the raw depth value shown on the Test Image page / Developer Panel.
3. Enter both values and click **Capture Reference**.
4. Repeat for a few distances (suggested: 0.5 m, 1 m, 1.5 m, 2 m, 3 m).

The app fits a simple inverse model (`distance ≈ k / depth_value`) from
your calibration points and immediately starts using it for new distance
estimates.

## Voice Commands

```
Where is my mobile?          Find my phone.
Where is my laptop?          Where is my computer?
Where is my notebook?        Find my book.
Where is my bag?             Find my water bottle.
Where are my spectacles?     Find my pen.
Find my pencil.              Where is the medicine box?
Where is the instrumentation box?

Start tracking.   Stop tracking.   What objects can you see?
```

Typed commands work identically via the text box on the camera page, for
browsers without SpeechRecognition support (e.g. Firefox).

## Supported Objects

| Canonical name         | Common aliases recognized              | Detection source |
|-------------------------|-----------------------------------------|-------------------|
| `laptop`                | laptop, notebook computer               | pretrained |
| `computer`               | computer, pc, desktop                   | pretrained (approx.) |
| `mobile_phone`           | phone, mobile, smartphone, cell phone   | pretrained |
| `table`                  | table, desk                             | pretrained |
| `notebook`               | book, textbook, notebook, diary         | pretrained |
| `bag`                    | school bag, college bag, backpack       | pretrained |
| `water_bottle`           | bottle, water bottle                    | pretrained |
| `pen`                    | pen                                      | **custom model required** |
| `pencil`                 | pencil                                   | **custom model required** |
| `spectacles`             | glasses, specs, eyeglasses              | **custom model required** |
| `medicine_box`           | medicine, medkit, pill box              | **custom model required** |
| `instrumentation_box`    | instrument box, toolbox                 | **custom model required** |

## Architecture

```
Browser (getUserMedia, 24-30 FPS display)
   |
   |  frame sampled at 4-8 FPS, JPEG-encoded
   v
WebSocket  (dashboard/consumers.py)
   |
   v
YOLO detection  (vision/detector.py)
   |
   v
Depth estimation, every 2nd-3rd AI frame  (vision/depth_estimator.py)
   |
   v
Distance estimation  (vision/distance_estimator.py + vision/calibration.py)
   |
   v
IoU Tracking, stable IDs  (vision/tracker.py)
   |
   v
Navigation engine  (navigation/navigator.py, direction.py, tracking.py)
   |
   v
JSON  -->  Browser  (overlay.js draws boxes, navigation.js speaks guidance)
```

Voice input/output never touches the server for TTS/ASR — those run
entirely in-browser (SpeechRecognition / speechSynthesis). Only the parsed
*text* of a voice command is sent to the server
(`voice/command_parser.py`), which resolves it locally with no external
LLM call.

### Module responsibilities

| Module | Responsibility |
|---|---|
| `vision/detector.py` | Loads YOLO once, runs detection, custom/pretrained fallback |
| `vision/depth_estimator.py` | Loads MiDaS-small once, produces relative depth maps |
| `vision/distance_estimator.py` | Combines depth + calibration + known-size into meters |
| `vision/calibration.py` | Fits depth-to-meters conversion from user calibration points |
| `vision/tracker.py` | Lightweight IoU tracker assigning stable track IDs |
| `voice/command_parser.py` | Local, LLM-free parsing of recognized speech |
| `voice/responses.py` | Template-based natural language response generation |
| `navigation/direction.py` | Left/center/right classification + distance smoothing |
| `navigation/tracking.py` | Per-session target lock, history, grace period |
| `navigation/navigator.py` | Ties it all together: movement analysis + voice cooldown |
| `dashboard/consumers.py` | Real-time WebSocket pipeline orchestration |
| `api/views.py` | REST endpoints (detect, voice-command, tracking, calibration, status) |

## Performance

Configurable via `.env` / `config/settings.py`:

```
AI_PROCESSING_FPS=6        # 4, 6, or 8
CONFIDENCE_THRESHOLD=0.45
PERFORMANCE_MODE=balanced  # low | balanced | high
```

| Mode | AI FPS | Depth frequency | YOLO image size |
|---|---|---|---|
| low | 4 | every 3rd AI frame | 320px |
| balanced (default) | 6 | every 2nd AI frame | 416px |
| high | 8 | every 2nd AI frame | 480px |

The app also **auto-reduces** AI FPS and depth frequency at runtime if
inference consistently exceeds ~300 ms (Section 31 of the original spec),
so it degrades gracefully rather than freezing the browser.

To measure your own machine's performance, use Developer Mode on the
camera page (shows live inference time in ms) or `/api/status/`.

## Limitations

> This system provides experimental AI-based object guidance. Distance
> estimates are approximate and may be inaccurate. A normal RGB webcam
> cannot provide guaranteed physical distance. **The system should not be
> used as a safety-critical navigation system**, and should not replace a
> white cane, guide dog, or other certified mobility aid.

Additional known limitations:

- Pretrained YOLO cannot recognize spectacles, pens, pencils, medicine
  boxes, or instrumentation boxes without a custom-trained model.
- Distance accuracy depends heavily on calibration quality and lighting.
- Only one target object is actively tracked at a time (multiple
  detections of the same class are reported, but only the
  nearest/highest-confidence one is locked).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "AI detection model is unavailable" | `pip install -r requirements.txt` didn't finish, or first YOLO download needs internet |
| "Depth estimation is temporarily unavailable" | MiDaS couldn't download (no internet on first run) — known-size fallback is used automatically |
| Camera page has no live boxes | Check the browser console for WebSocket errors; ensure `daphne` is installed and listed first in `INSTALLED_APPS` |
| Voice button does nothing | Your browser may not support SpeechRecognition (try Chrome/Edge) — use the text command box instead |
| Everything feels slow | Switch `PERFORMANCE_MODE` to `low` in `.env`, or lower AI FPS in the camera page dropdown |

## Future Improvements

- Swap `vision/distance_estimator.py`'s depth backend for a stereo camera,
  Intel RealSense, or LiDAR sensor when available — the module boundary
  is designed to make this a drop-in replacement.
- Multi-target simultaneous tracking and announcement.
- On-device fine-tuning workflow for the custom model.
- Optional persistent detection history/analytics dashboard.

---

### Safety statement

*This system provides experimental AI-based object guidance. Distance
estimates are approximate and may be inaccurate. A normal RGB webcam
cannot provide guaranteed physical distance. The system should not be
used as a safety-critical navigation system.*
