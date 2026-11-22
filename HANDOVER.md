# 📋 Project Handover & Technical Blueprint: Pose AR Mirror

> **Target Audience:** AI Developer Agents & Software Engineers taking over or contributing to the codebase.  
> **Last Updated:** November 2026  
> **Repository Name:** `pose-ar-mirror` (Folder: `handsign_recognizing`)  
> **Primary Language & Stack:** Python 3.10+, OpenCV 4.8+, Google MediaPipe Pose, Pygame, Supabase, CustomTkinter.

---

## 🧭 1. Executive Summary & Vision

### 1.1 The Concept
**Pose AR Mirror** is an interactive, real-time Computer Vision and Augmented Reality (AR) mirror designed to run on consumer hardware (CPU, 720p/1080p webcams) without dedicated GPU or depth sensors.

The user performs iconic body poses in front of their webcam. When the system recognizes a target posture and confirms the user holds it intentionally (via a dwell-time state machine), it summons virtual characters, overlays thematic AR assets, plays audio feedback, and captures high-resolution souvenir snapshots.

### 1.2 Evolution & Extensibility: Beyond Football Celebrations
While **Version 1 (V1)** implements 6 iconic football star celebrations (Bellingham, Messi, Ronaldo, Mbappé, Rashford, Haaland), **the project is architected to be multi-thematic**. Future versions will support interchangeable "Theme Packs", including:
- **Action & Gaming Poses** (e.g. martial arts katas, victory gestures, power-up stances).
- **Cinema & Memes** (iconic movie stances).
- **Sign Language & Hand Signs** (integrating MediaPipe Hands alongside Pose).
- **Fitness & Posture Coaching** (squats, yoga stances with repetition counting).

---

## 🏛️ 2. System Architecture & Data Flow

The application executes a synchronous, 30 FPS pipeline rendered onto a split-screen broadcast canvas (`1280 x 480` total; two `640 x 480` panes):

```
                        [ Webcam Frame (640 x 480 RGB) ]
                                       │
                                       ▼
                     [ MediaPipe Pose Landmarker Lite ]
                         ├── 33 3D Landmarks (x, y, z)
                         └── Person Segmentation Mask
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
  [ CelebrationDetector (Engine) ]               [ BackgroundInpainter ]
  - Computes normalized body metrics             - Subtracts user from background
  - Matches 6 geometric pose rules               - Soft edge feathering (7x7 blur)
                │                                             │
                ▼                                             │
  [ CelebrationStateMachine ]                                 │
  - Accumulates Dwell Time (0.8s)                             │
  - Handles Debounce Grace (0.4s)                             │
  - Generates Alpha Fade (0.0 -> 1.0)                         │
  - FIFO buffer (anti-repeat last 2 assets)                   │
                │                                             │
                ▼                                             ▼
  [ Audio Engine (Pygame) ]                      [ PlayerOverlayManager ]
  - Plays cheer / celebration sound             - Composites player cut-out / card
                │                                - Dynamic positioning on Right Pane
                └──────────────────────┬──────────────────────┘
                                       ▼
                              [ CanvasRenderer ]
                - Left Pane : Live camera feed + Skeleton HUD + Dwell circle
                - Right Pane: AR Stage + Segmented User + Summoned Character
                - Broadcast HUD (FPS, status banner, calibration alerts)
                                       │
                                       ▼
                       [ Output & Interactive Hooks ]
                - OpenCV HighGUI Window ('Champions Celebrations AR Mirror')
                - Hotkeys: [C] Calibrate, [S] Screenshot, [M] Mute, [Q] Quit
                - Local save (`screenshots/`) + Cloud Sync (`Supabase`)
```

---

## 📂 3. Directory Layout & Module Responsibilities

```text
handsign_recognizing/
├── assets/
│   ├── audio/              # Sound files (.wav, .mp3) triggered on active celebration
│   ├── models/             # Google MediaPipe TFLite model: pose_landmarker_lite.task (5.7 MB)
│   └── players/            # PNG cutouts with alpha channel: {player_id}_{celebration_id}.png
├── screenshots/            # Output folder for user snapshots (PNG)
├── src/
│   ├── __init__.py         # Package root
│   ├── config.py           # Single Source of Truth: resolutions, thresholds, colors, timings
│   ├── camera.py           # Thread-safe OpenCV VideoCapture wrapper with mirror mode
│   ├── pose_detector.py    # MediaPipe Pose Landmarker Lite wrapper (extracts 33 points & mask)
│   ├── celebrations.py     # Heuristic geometric engine evaluating the 6 celebrations
│   ├── state_machine.py    # Dwell charge, debounce grace, alpha interpolation, asset history
│   ├── segmenter.py        # Clean-plate room background capture & human inpainting
│   ├── player_overlay.py   # Alpha compositing & fallback dynamic card generator
│   └── renderer.py         # Dual-pane canvas compositor & broadcast HUD overlay
├── main.py                 # Primary entry point: application runtime loop
├── test_pipeline.py        # Headless unit & verification tests
├── requirements.txt        # Pinned runtime dependencies
├── README.md               # User-facing project presentation
└── HANDOVER.md             # This document (AI & Engineering blueprint)
```

---

## 🔍 4. Current Implementation Deep-Dive

### 4.1 Implemented Celebrations (`src/celebrations.py`)
All detections use normalized coordinates ($x, y \in [0, 1]$) relative to `shoulder_width` ($W_s$) and `torso_height` ($H_t$):

| Player | Celebration ID | Geometric Rule & Logic |
| :--- | :--- | :--- |
| **Jude Bellingham** | `open_arms` | $\text{arm\_span} > 1.8 \times W_s$ and both wrists vertically within $0.35 \times H_t$ of shoulder level. |
| **Lionel Messi** | `to_the_heavens` | Both wrists above head ($y_{\text{wrist}} < y_{\text{head}} - 0.05 \times H_t$), arms substantially extended vertically. |
| **Marcus Rashford** | `temple_tap` | One index within $0.35 \times H_t$ of the nearest ear, elbow bent ($< 0.5 \times H_t$ below shoulder), other arm relaxed below shoulder level. |
| **Kylian Mbappé** | `folded_arms` | Forearms crossed over chest: each wrist within $0.4 \times H_t$ of opposite shoulder, both wrists between shoulder and hip lines. |
| **Erling Haaland** | `zen_lotus` | Both wrists at chest level ($0.1 \times H_t$ to $0.7 \times H_t$ below shoulders), distance between wrists $< 0.45 \times W_s$, elbows flared outwards. |
| **Cristiano Ronaldo** | `siuuu_landing` | Wrists driven down and behind hips ($y_{\text{wrist}} > y_{\text{hip}} + 0.1 \times H_t$), arms angled outward ($> 1.1 \times W_s$). |

### 4.2 State Machine Dynamics (`src/state_machine.py`)
To ensure zero accidental triggers during normal body movement:
- **`DWELL_TIME_SEC` (0.8s):** The user must maintain the detected pose continuously. `progress` charges linearly from $0.0$ to $1.0$. Only when `progress == 1.0` does `is_active` become `True`.
- **`DEBOUNCE_GRACE_SEC` (0.4s):** If a pose is temporarily dropped (e.g. tracking glitch), the state does not reset instantly. It decays smoothly over 400ms.
- **`alpha` Blending:** A smooth float ($0.0 \rightarrow 1.0$) easing player opacity in and out.
- **Asset History Buffer:** Uses a `deque(maxlen=2)` per celebration so the same variant asset is never shown consecutively.

### 4.3 Fallback Card Generation (`src/player_overlay.py`)
If physical image assets are missing from `assets/players/{key}.png`, `PlayerOverlayManager.generate_card_cutout()` automatically builds a procedural digital player card featuring the club theme, jersey number, and player name. **The application never crashes due to missing image assets.**

---

## 🛠️ 5. Next Tasks & Implementation Checklist

When picking up this project, tackle tasks in this exact order:

### Task 1: Complete Pygame Sound Integration
- **Current state:** `pygame>=2.5.0` is installed, and `assets/audio/` exists, but sound trigger calls in `main.py` are commented out or minimal.
- **Work needed:**
  1. Add a sound manager (`src/audio_engine.py` or directly in `main.py`).
  2. Map each celebration ID to a sound effect (e.g., stadium roar, referee whistle, or iconic commentator clip).
  3. Fire the sound **once** on the rising edge of `is_active` (when `state_machine.is_active` transitions from `False` to `True`).
  4. Honor the `sound_muted` flag toggled by key `[M]`.

### Task 2: Populate Visual Assets (`assets/players/`)
- **Current state:** Procedural fallback cards render nicely, but real cutouts provide the true AR magic.
- **Work needed:**
  - Add transparent PNG images (resolution $\approx 400 \times 600$, RGBA) under `assets/players/`:
    - `bellingham_open_arms.png`
    - `messi_to_the_heavens.png`
    - `ronaldo_siuuu_landing.png`
    - `mbappe_folded_arms.png`
    - `haaland_zen_lotus.png`
    - `rashford_temple_tap.png`

### Task 3: Interactive Screenshot Gallery (CustomTkinter)
- **Current state:** Key `[S]` saves snapshots into `screenshots/moment_YYYYMMDD_HHMMSS.png`.
- **Work needed:**
  - Build a lightweight desktop gallery window launched via a hotkey `[G]` or separate script:
    - Display a grid of recent screenshots.
    - Buttons: "Open in File Explorer", "Share / Upload", "Delete".
    - Uses `customtkinter` (already in `requirements.txt`).

### Task 4: Supabase Cloud Sync
- **Current state:** Config keys exist in `src/config.py` (`SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_BUCKET_SCREENSHOTS`).
- **Work needed:**
  - Implement asynchronous background upload (using `threading.Thread`) so frame rate never drops:
    ```python
    def upload_screenshot_async(filepath: Path):
        # Authenticate Supabase client
        # Upload binary stream to SUPABASE_BUCKET_SCREENSHOTS
        # Insert row into 'captures' table (timestamp, player_id, image_url)
    ```

---

## 🔮 6. Multi-Theme Refactoring Guide (Future Architecture)

To expand beyond football without rewriting core code, refactor `src/celebrations.py` into a **Theme Provider Pattern**:

```
src/themes/
├── __init__.py
├── base_theme.py         # Abstract Base Class: BaseTheme (evaluate, get_assets, get_audio)
├── football_theme.py     # Migrated logic of current celebrations.py
├── action_theme.py       # Martial arts stances, energy charges, victory poses
└── sign_language_theme.py# Hand signs / Dactylology recognition (linking MediaPipe Hands)
```

### Proposed Interface (`base_theme.py`):
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class PoseMatch:
    theme_id: str
    action_id: str
    display_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

class BaseTheme(ABC):
    @abstractmethod
    def evaluate(self, landmarks) -> Optional[PoseMatch]:
        """Evaluates 3D landmarks and returns a match if pose matches criteria."""
        pass

    @abstractmethod
    def get_overlay_asset(self, match: PoseMatch):
        pass
```

---

## 💻 7. Developer & AI Instructions: How to Run and Test

### 7.1 Environment Setup (Ubuntu / WSL2)
```bash
cd /home/youssef19/handsign_recognizing

# Activate existing virtualenv
source .venv/bin/activate

# Install any missing packages
pip install -r requirements.txt
```

### 7.2 Running Headless Pipeline Tests
Before making commits, always run:
```bash
python3 test_pipeline.py
```
This tests state machine dwell progression, debounce grace decay, history buffer anti-repeat, and dummy frame rendering without needing a live webcam.

### 7.3 Running the Live Application
```bash
python3 main.py
```

### 7.4 Runtime Hotkeys
| Key | Action |
| :---: | :--- |
| `[C]` | **Calibrate Room** (step out of camera view for 2 seconds to snapshot clean background). |
| `[S]` | **Take Screenshot** (saves dual-screen composite to `screenshots/`). |
| `[M]` | **Mute / Unmute** sound effects. |
| `[Q]` or `[ESC]` | **Quit** application cleanly. |

---

## 📝 8. Coding Conventions & Invariants

1. **Coordinate Systems:** MediaPipe landmarks are normalized in range $[0.0, 1.0]$. Always multiply by pixel dimensions (`PANE_WIDTH`, `PANE_HEIGHT`) only at the final rendering stage in `renderer.py` or `player_overlay.py`.
2. **Never block the main loop:** Any I/O (cloud upload, file writing, heavy processing) must happen asynchronously or within $<2\text{ms}$ to maintain the target 30 FPS.
3. **Graceful Asset Fallback:** Never assume local assets (images/audio) exist on disk. Always catch `FileNotFoundError` or check `.exists()` and fall back to procedural graphics.
