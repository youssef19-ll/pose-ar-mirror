"""Background subtraction & person inpainting engine using MediaPipe."""

import time
import cv2
import numpy as np
from src.config import (
    CALIBRATION_COUNTDOWN_SEC,
    SEGMENTATION_THRESHOLD,
    MASK_BLUR_KERNEL,
    COLOR_CYAN,
    COLOR_ALERT
)


class BackgroundInpainter:
    """Manages clean plate calibration and removes the user from the right pane."""

    def __init__(self, threshold: float = SEGMENTATION_THRESHOLD):
        self.threshold = threshold
        self.clean_bg = None
        self.is_calibrated = False

        # Calibration state
        self.is_calibrating = False
        self.calibration_start_time = 0.0
        self.calibration_frames = []

    def start_calibration(self):
        """Initiates the clean plate calibration countdown."""
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_frames = []
        print("[Calibration] Step out of frame. Capturing room background...")

    def update_calibration(self, current_frame: np.ndarray) -> tuple[bool, float]:
        """
        Gathers frames during countdown. Returns (is_done, remaining_seconds).
        Averages frames over time to eliminate sensor noise.
        """
        elapsed = time.time() - self.calibration_start_time
        remaining = max(0.0, CALIBRATION_COUNTDOWN_SEC - elapsed)

        self.calibration_frames.append(current_frame.astype(np.float32))

        if elapsed >= CALIBRATION_COUNTDOWN_SEC:
            # Compute temporal median/mean to get crystal clean plate
            avg_bg = np.median(self.calibration_frames, axis=0).astype(np.uint8)
            self.clean_bg = avg_bg
            self.is_calibrated = True
            self.is_calibrating = False
            self.calibration_frames = []
            print("[Calibration] Background captured successfully!")
            return True, 0.0

        return False, remaining

    def remove_person(self, live_frame: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
        """
        Replaces the segmented person with the clean background plate.
        seg_mask: Float32 array (0.0 to 1.0) of shape (H, W).
        """
        if not self.is_calibrated or self.clean_bg is None:
            # Fallback if not calibrated: show darkened frame with prompt
            uncalibrated = (live_frame * 0.35).astype(np.uint8)
            h, w = uncalibrated.shape[:2]
            cv2.putText(uncalibrated, "ROOM NOT CALIBRATED", (w // 2 - 190, h // 2 - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_ALERT, 2)
            cv2.putText(uncalibrated, "Step aside and Press 'C' to calibrate", (w // 2 - 230, h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
            return uncalibrated

        # Binary thresholding
        binary_mask = (seg_mask > self.threshold).astype(np.float32)

        # Soften edges (feathering) for natural blending
        feathered_mask = cv2.GaussianBlur(binary_mask, MASK_BLUR_KERNEL, 0)
        # 3-channel mask for broadcasting
        alpha_3d = np.repeat(feathered_mask[:, :, np.newaxis], 3, axis=2)

        # Inpainting formula: Person area becomes clean_bg, surroundings stay live_frame
        inpainted = (live_frame * (1.0 - alpha_3d) + self.clean_bg * alpha_3d).astype(np.uint8)
        return inpainted
