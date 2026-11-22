"""Cross-platform webcam capture manager with fallback test-card generator."""

import platform
import sys
import time
from typing import Union

import cv2
import numpy as np
from src.config import PANE_WIDTH, PANE_HEIGHT, CAMERA_SOURCE


def _is_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.release().lower()


def _backends_for(source: Union[int, str]) -> list[int]:
    """Ordered list of OpenCV capture backends to try for this OS / source type."""
    if isinstance(source, str):
        # URL (http/rtsp), video file or device path: let OpenCV pick (FFmpeg / GStreamer)
        return [cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


class CameraStream:
    """Handles real-time frame acquisition with mirroring and a simulated fallback.

    ``source`` is a camera index (0, 1, ...) or a string: a stream URL
    (``http://host:port/video``, ``rtsp://...``), a video file, or a device path
    (``/dev/video0``). Defaults to the ``CAMERA_SOURCE`` setting (env var / .env).
    """

    def __init__(
        self,
        source: Union[int, str] = CAMERA_SOURCE,
        width: int = PANE_WIDTH,
        height: int = PANE_HEIGHT,
        mirror: bool = True,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.mirror = mirror
        self.cap = None
        self.is_virtual = False
        self._open_camera()

    def _open_camera(self):
        """Tries each backend in order; falls back to a simulated feed if none delivers frames."""
        # Probing backends that don't fit makes OpenCV log noisy warnings: silence them here
        set_log_level = getattr(getattr(cv2, "utils", None), "logging", None)
        prev_level = None
        if set_log_level is not None:
            prev_level = set_log_level.getLogLevel()
            set_log_level.setLogLevel(set_log_level.LOG_LEVEL_SILENT)
        try:
            self._try_backends()
        finally:
            if prev_level is not None:
                set_log_level.setLogLevel(prev_level)

        if self.is_virtual:
            print(f"[Camera] Warning: camera source {self.source!r} unavailable. Using virtual test feed.")
            if _is_wsl() and isinstance(self.source, int):
                print("[Camera] WSL2 detected: attach the webcam with usbipd-win (see README, "
                      "'Webcam on WSL2') or set CAMERA_SOURCE to a stream URL.")

    def _try_backends(self):
        for backend in _backends_for(self.source):
            cap = cv2.VideoCapture(self.source, backend)
            if not cap.isOpened():
                cap.release()
                continue

            if isinstance(self.source, int):
                if backend == cv2.CAP_V4L2:
                    # MJPEG keeps 30 FPS at 640x480 on USB 2.0 and over usbip (WSL2)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Lower latency where supported

            ok, frame = cap.read()
            if ok and frame is not None:
                self.cap = cap
                self.is_virtual = False
                print(f"[Camera] Source {self.source!r} opened with backend {cap.getBackendName()}.")
                return
            cap.release()

        self.is_virtual = True

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
