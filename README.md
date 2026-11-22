# Pose AR Mirror

Real-time Augmented Reality Mirror powered by Computer Vision. Detects human poses and gestures from a live webcam feed, triggers interactive AR overlays, and provides synchronized audio feedback.

## Overview

Pose AR Mirror transforms a standard webcam into an interactive augmented reality display. The application identifies body postures in real time, validates intentional gestures using a dwell-time state machine, and renders a split-screen canvas featuring live tracking alongside an augmented stage with character overlays and background segmentation.

## Features

- Real-Time Pose Tracking: Fast 3D body landmark extraction using MediaPipe Pose.
- Geometric Gesture Engine: Normalized metric evaluation invariant to user scale, body type, and camera distance.
- Robust State Machine: Dwell timer (hold-to-activate) and debounce grace period to prevent accidental triggers.
- Dual-Pane Canvas: Split-screen interface showing the live feed with HUD alongside the augmented reality output.
- Background Segmentation: Clean-plate room calibration and soft-edge person inpainting.
- Audio & Snapshot Capture: Sound effects on gesture confirmation and manual or automated screenshot capture.

## Pipeline Architecture

```
[ Webcam Feed ] -> [ Pose Landmark Extraction ] -> [ Geometric Pose Engine ]
                                                             |
                                                             v
[ Dual-Pane Display ] <- [ Overlay & Inpainting ] <- [ State Machine (Dwell / Debounce) ]
```

## Project Structure

```
├── assets/
│   ├── audio/              # Sound effects
│   ├── models/             # MediaPipe task models
│   └── players/            # Character cutouts (RGBA)
├── screenshots/            # Output captures
├── src/
│   ├── camera.py           # Video capture wrapper
│   ├── celebrations.py     # Pose detection rules
│   ├── config.py           # System settings and constants
│   ├── player_overlay.py   # AR overlay compositing
│   ├── pose_detector.py    # MediaPipe pose wrapper
│   ├── renderer.py         # Dual-canvas and HUD rendering
│   ├── segmenter.py        # Background inpainting
│   └── state_machine.py    # Dwell timer and debounce logic
├── main.py                 # Application entry point
├── test_pipeline.py        # Verification test suite
└── requirements.txt        # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- Webcam

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/youssef19-ll/pose-ar-mirror.git
   cd pose-ar-mirror
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run verification tests:
```bash
python3 test_pipeline.py
```

Launch the application:
```bash
python3 main.py
```

### Controls

- C: Calibrate room background
- S: Take screenshot
- M: Toggle sound
- Q / ESC: Exit application

## Roadmap

- Temporal gesture recognition for dynamic movements
- Multi-user simultaneous tracking
- Expandable theme packs (action, sports, fitness, and sign language)
- Desktop GUI gallery for browsing captured snapshots
