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
├── test_logic.py           # Pure-logic tests (rules, state machine)
├── test_pipeline.py        # End-to-end pipeline check
└── requirements.txt        # Python dependencies
```

## Getting Started

The project runs on **Linux** (including Ubuntu on WSL2) and **Windows**. Commands below use
`python3` on Linux and `python` on Windows; once the virtual environment is active, `python`
works on both.

### Prerequisites

- Python 3.10+
- A webcam (without one, a simulated feed is shown)
- Linux only: `sudo apt install python3-venv python3-tk`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/youssef19-ll/pose-ar-mirror.git
   cd pose-ar-mirror
   ```

2. Create and activate a virtual environment:

   Linux / WSL2:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Windows (PowerShell):
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. Optional: copy `.env.example` to `.env` to choose the camera (`CAMERA_SOURCE`) or set Supabase keys.

A virtual environment is tied to the OS that created it: create one per OS, never share the
`.venv` folder between Windows and Linux.

### Usage

Run the logic tests (no webcam or model needed):
```bash
python test_logic.py
```

Run the full pipeline check (loads the MediaPipe model):
```bash
python test_pipeline.py
```

Launch the application:
```bash
python main.py
```

### Webcam on WSL2

WSL2 does not see USB webcams by default. The window and sound work out of the box (WSLg), but
the camera has to be forwarded with [usbipd-win](https://github.com/dorssel/usbipd-win):

1. In an **administrator** PowerShell on Windows:
   ```powershell
   winget install usbipd
   usbipd list                      # note the BUSID of the camera, e.g. 2-4
   usbipd bind --busid 2-4
   usbipd attach --wsl --busid 2-4
   ```
2. In Ubuntu:
   ```bash
   sudo modprobe uvcvideo
   ls /dev/video*                   # should list /dev/video0
   sudo usermod -aG video $USER     # once, then restart the WSL session
   ```

`usbipd attach` has to be run again after each reboot or unplug. While attached, the camera is not
available to Windows apps. Alternatively, set `CAMERA_SOURCE` to a network stream URL (for
example a phone camera app) instead of forwarding the USB device.

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
