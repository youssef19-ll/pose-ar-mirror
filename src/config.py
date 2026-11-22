"""Configuration and constants for Champions Celebration AR Mirror."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
PLAYERS_DIR = ASSETS_DIR / "players"
AUDIO_DIR = ASSETS_DIR / "audio"
CACHE_DIR = BASE_DIR / ".cache"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# Ensure runtime directories exist
for path in [ASSETS_DIR, MODELS_DIR, PLAYERS_DIR, AUDIO_DIR, CACHE_DIR, SCREENSHOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

# Video & Display Settings
# Single window split down the middle (Left: Live Feed, Right: Virtual Stadium)
PANE_WIDTH = 640
PANE_HEIGHT = 480
CANVAS_WIDTH = PANE_WIDTH * 2   # 1280
CANVAS_HEIGHT = PANE_HEIGHT      # 480
TARGET_FPS = 30

# Background Calibration Settings
CALIBRATION_COUNTDOWN_SEC = 2.0  # Duration user steps aside to capture room
SEGMENTATION_THRESHOLD = 0.55     # Confidence threshold for human mask (0.0 to 1.0)
MASK_BLUR_KERNEL = (7, 7)         # Soft edge feathering for seamless blending

# Dwell & Gesture Timing Settings
DWELL_TIME_SEC = 0.8             # Seconds holding gesture before triggering celebration
DEBOUNCE_GRACE_SEC = 0.4         # Grace period when gesture is dropped before effect vanishes
ACCURACY_EXPONENT = 1.2          # Scoring curve

# Supabase Settings (Placeholder / .env defaults)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-anon-or-service-key")
SUPABASE_BUCKET_ASSETS = "assets"
SUPABASE_BUCKET_SCREENSHOTS = "screenshots"

# Visual Theme Colors (BGR format for OpenCV)
COLOR_BG_DARK = (18, 18, 24)        # Deep stadium navy
COLOR_ACCENT = (0, 215, 255)        # Gold / Yellow
COLOR_CYAN = (255, 200, 0)          # Champions League cyan
COLOR_TEXT = (245, 245, 245)        # White
COLOR_ALERT = (40, 40, 230)         # Red
COLOR_SUCCESS = (60, 220, 90)       # Pitch green
