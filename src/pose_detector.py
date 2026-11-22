"""MediaPipe Pose & Segmentation extractor using modern Tasks API."""

import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import MODEL_PATH, MODEL_URL


class PoseSegmenter:
    """Extracts 3D body pose landmarks and real-time segmentation mask in a single forward pass."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 2
    ):
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
        self.drawing_utils = vision.drawing_utils

    def _ensure_model_exists(self):
        """Auto-downloads the task model weights if not locally present."""
        if not MODEL_PATH.exists():
            print(f"[Model] Downloading Pose Landmarker model from Google...")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print(f"[Model] Download complete: {MODEL_PATH}")

    def process(self, bgr_frame: np.ndarray):
        """
        Processes BGR frame.
        Returns:
            landmarks_list: List of NormalizedLandmark lists (one per detected person)
            combined_seg_mask: np.ndarray (H, W) float32 in [0.0, 1.0]
        """
        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect(mp_image)

        landmarks_list = result.pose_landmarks if result.pose_landmarks else []

        # Combine segmentation masks from all detected individuals
        if result.segmentation_masks and len(result.segmentation_masks) > 0:
            mask_arrays = []
            for m in result.segmentation_masks:
                arr = m.numpy_view()
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr.squeeze(-1)
                mask_arrays.append(arr)

            if len(mask_arrays) == 1:
                combined_mask = mask_arrays[0].astype(np.float32)
            else:
                combined_mask = np.maximum.reduce(mask_arrays).astype(np.float32)
        else:
            combined_mask = np.zeros((h, w), dtype=np.float32)

        return landmarks_list, combined_mask

    def draw_skeletons(self, frame: np.ndarray, landmarks_list: list) -> np.ndarray:
        """Draws glowing sports-tracking skeleton over all detected users."""
        if not landmarks_list:
            return frame

        annotated = frame.copy()
        for landmarks in landmarks_list:
            self.drawing_utils.draw_landmarks(
                annotated,
                landmarks,
                self.connections,
                landmark_drawing_spec=self.drawing_utils.DrawingSpec(color=(0, 215, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=self.drawing_utils.DrawingSpec(color=(255, 200, 0), thickness=2)
            )
        return annotated

    def close(self):
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
