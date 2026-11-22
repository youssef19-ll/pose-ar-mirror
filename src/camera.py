"""Webcam video capture manager with fallback test-card generator."""

import time
import cv2
import numpy as np
from src.config import PANE_WIDTH, PANE_HEIGHT


class CameraStream:
    """Handles real-time webcam frame acquisition with mirroring and fallback."""

    def __init__(self, camera_index: int = 0, width: int = PANE_WIDTH, height: int = PANE_HEIGHT, mirror: bool = True):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.cap = None
        self.is_virtual = False
        self._open_camera()

    def _open_camera(self):
        """Attempts to open physical webcam, falls back to simulated feed if unavailable."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Retry with default backend
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.is_virtual = False
            print(f"[Camera] Physical webcam #{self.camera_index} connected successfully.")
        else:
            self.is_virtual = True
            print("[Camera] Warning: No physical camera found or camera in use. Using virtual test feed.")

    def read(self):
        """Returns (success: bool, frame: np.ndarray in BGR)."""
        if not self.is_virtual and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Resize if camera did not match desired resolution
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                return True, frame

        # Fallback simulated frame (if camera unavailable)
        return True, self._generate_simulated_frame()

    def _generate_simulated_frame(self) -> np.ndarray:
        """Generates a moving placeholder frame for headless or camera-busy environments."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background pitch gradient
        frame[:, :] = (30, 45, 30)
        t = time.time()
        # Animated pulse circle
        cx = int(self.width / 2 + 100 * np.sin(t * 2))
        cy = int(self.height / 2 + 50 * np.cos(t * 2))
        cv2.circle(frame, (cx, cy), 60, (0, 215, 255), -1)
        cv2.putText(frame, "SIMULATED WEBCAM FEED", (50, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Connect webcam to test live pose", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
