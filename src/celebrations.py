"""Celebration detector for the 6 iconic football stars."""

import math
from dataclasses import dataclass
from typing import Optional

from src.config import PANE_WIDTH, PANE_HEIGHT, VISIBILITY_THRESHOLD


# MediaPipe Landmark Index Map
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_HIP = 23
RIGHT_HIP = 24

# Typical torso height (shoulder -> hip) relative to shoulder width, used when the hips
# are out of frame (user standing close to the camera).
TORSO_TO_SHOULDER_RATIO = 1.3


@dataclass
class CelebrationMatch:
    player_id: str
    player_name: str
    celebration_id: str
    celebration_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y (normalized)


class CelebrationDetector:
    """Evaluates body pose landmarks to detect football stars' celebrations.

    All rules are expressed relative to the user's own body (shoulder width ``sw`` and
    torso height ``th``) so they hold at any distance from the camera. Coordinates are
    aspect-corrected (x is scaled by width / height) so that horizontal and vertical
    distances share the same unit.
    """

    def __init__(
        self,
        frame_width: int = PANE_WIDTH,
        frame_height: int = PANE_HEIGHT,
        visibility_threshold: float = VISIBILITY_THRESHOLD,
    ):
        self.aspect = frame_width / float(frame_height)
        self.visibility_threshold = visibility_threshold

    def evaluate(self, landmarks) -> Optional[CelebrationMatch]:
        """
        Evaluates a single person's landmarks list (33 NormalizedLandmarks).
        Returns CelebrationMatch if a celebration pose is matched, else None.
        Rules are evaluated in priority order; the first match wins.
        """
        if not landmarks or len(landmarks) < 25:
            return None

        lm = landmarks
        aspect = self.aspect

        def X(i):
            return lm[i].x * aspect

        def Y(i):
            return lm[i].y

        def dist(i1, i2):
            return math.hypot(X(i1) - X(i2), Y(i1) - Y(i2))

        def visible(*idx):
            for i in idx:
                v = getattr(lm[i], "visibility", None)
                if v is not None and v < self.visibility_threshold:
                    return False
            return True

        # Shoulders are the anchor of every rule
        if not visible(LEFT_SHOULDER, RIGHT_SHOULDER):
            return None

        sw = max(0.01, dist(LEFT_SHOULDER, RIGHT_SHOULDER))
        shoulder_y = (Y(LEFT_SHOULDER) + Y(RIGHT_SHOULDER)) / 2.0

        hips_visible = visible(LEFT_HIP, RIGHT_HIP)
        if hips_visible:
            hip_y = (Y(LEFT_HIP) + Y(RIGHT_HIP)) / 2.0
            th = max(0.01, (dist(LEFT_SHOULDER, LEFT_HIP) + dist(RIGHT_SHOULDER, RIGHT_HIP)) / 2.0)
        else:
            th = TORSO_TO_SHOULDER_RATIO * sw
            hip_y = shoulder_y + th  # estimate only; hip-dependent rules are skipped

        head_pts = [i for i in (NOSE, LEFT_EAR, RIGHT_EAR) if visible(i)]
        head_y = min(Y(i) for i in head_pts) if head_pts else None

        wrists_visible = visible(LEFT_WRIST, RIGHT_WRIST)
        arms_visible = wrists_visible and visible(LEFT_ELBOW, RIGHT_ELBOW)

        # Bounding box (normalized, un-corrected) for overlay placement
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        bbox = (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))

        # -------------------------------------------------------------
        # 1. JUDE BELLINGHAM — OPEN ARMS (wrists wide apart, near shoulder level)
        # -------------------------------------------------------------
        if wrists_visible:
            arm_span = abs(X(LEFT_WRIST) - X(RIGHT_WRIST))
            wrists_near_shoulder_level = (
                abs(Y(LEFT_WRIST) - shoulder_y) < 0.35 * th and
                abs(Y(RIGHT_WRIST) - shoulder_y) < 0.35 * th
            )
            if arm_span > 2.0 * sw and wrists_near_shoulder_level:
                conf = min(1.0, (arm_span / (2.0 * sw)) * 0.9)
                return CelebrationMatch("bellingham", "Jude Bellingham", "open_arms", "Open Arms", conf, bbox)

        # -------------------------------------------------------------
        # 2. LIONEL MESSI
        # -------------------------------------------------------------
        # A: TO THE HEAVENS (both wrists high above head)
        if wrists_visible and head_y is not None:
            top_wrist_gap = head_y - max(Y(LEFT_WRIST), Y(RIGHT_WRIST))
            if top_wrist_gap > 0.15 * th:
                conf = min(1.0, top_wrist_gap / th + 0.5)
                return CelebrationMatch("messi", "Lionel Messi", "to_the_heavens", "To The Heavens", conf, bbox)

        # B: TOPO GIGIO (hands cupped behind ears)
        if wrists_visible and visible(LEFT_EAR, RIGHT_EAR):
            if dist(LEFT_WRIST, LEFT_EAR) < 0.45 * sw and dist(RIGHT_WRIST, RIGHT_EAR) < 0.45 * sw:
                return CelebrationMatch("messi", "Lionel Messi", "topo_gigio", "Topo Gigio", 0.95, bbox)

        # -------------------------------------------------------------
        # 3. MARCUS RASHFORD — TEMPLE TAP (one index on temple, other arm relaxed)
        # -------------------------------------------------------------
        relaxed_below = shoulder_y + 0.3 * th
        left_tap = (visible(LEFT_INDEX, LEFT_EAR, RIGHT_WRIST) and
                    dist(LEFT_INDEX, LEFT_EAR) < 0.35 * sw and Y(RIGHT_WRIST) > relaxed_below)
        right_tap = (visible(RIGHT_INDEX, RIGHT_EAR, LEFT_WRIST) and
                     dist(RIGHT_INDEX, RIGHT_EAR) < 0.35 * sw and Y(LEFT_WRIST) > relaxed_below)
        if left_tap or right_tap:
            return CelebrationMatch("rashford", "Marcus Rashford", "temple_tap", "Mind / Temple Tap", 0.95, bbox)

        # Rules below need the hips as a reference line
        if not hips_visible:
            return None

        # -------------------------------------------------------------
        # 4. KYLIAN MBAPPÉ — FOLDED ARMS (wrists crossed near opposite shoulders)
        # -------------------------------------------------------------
        if wrists_visible:
            wrists_crossed = (
                dist(LEFT_WRIST, RIGHT_SHOULDER) < 0.85 * sw and
                dist(RIGHT_WRIST, LEFT_SHOULDER) < 0.85 * sw and
                Y(LEFT_WRIST) < hip_y and Y(RIGHT_WRIST) < hip_y
            )
            if wrists_crossed:
                return CelebrationMatch("mbappe", "Kylian Mbappe", "folded_arms", "Folded Arms", 0.92, bbox)

        # -------------------------------------------------------------
        # 5. ERLING HAALAND
        # -------------------------------------------------------------
        # A: LOTUS / ZEN (hands together at chest height, elbows outward)
        if arms_visible:
            hands_distance = dist(LEFT_WRIST, RIGHT_WRIST)
            wrists_at_chest = (
                hands_distance < 0.6 * sw and
                shoulder_y < Y(LEFT_WRIST) < hip_y and
                shoulder_y < Y(RIGHT_WRIST) < hip_y and
                X(LEFT_ELBOW) > X(LEFT_SHOULDER) and  # Elbows flaring out
                X(RIGHT_ELBOW) < X(RIGHT_SHOULDER)
            )
            if wrists_at_chest:
                return CelebrationMatch("haaland", "Erling Haaland", "zen_lotus", "Zen Meditation", 0.90, bbox)

        # B: FINGER TO EAR (one hand at ear, other arm down)
        arm_down = hip_y - 0.6 * th
        left_ear_touch = (visible(LEFT_WRIST, LEFT_EAR, RIGHT_WRIST) and
                          dist(LEFT_WRIST, LEFT_EAR) < 0.4 * sw and Y(RIGHT_WRIST) > arm_down)
        right_ear_touch = (visible(RIGHT_WRIST, RIGHT_EAR, LEFT_WRIST) and
                           dist(RIGHT_WRIST, RIGHT_EAR) < 0.4 * sw and Y(LEFT_WRIST) > arm_down)
        if left_ear_touch or right_ear_touch:
            return CelebrationMatch("haaland", "Erling Haaland", "finger_to_ear", "Finger To Ear", 0.88, bbox)

        # -------------------------------------------------------------
        # 6. CRISTIANO RONALDO
        # -------------------------------------------------------------
        if wrists_visible:
            # A: SIUUU LANDING (wrists swept down and outward, at/below hips)
            wrists_behind_and_down = (
                Y(LEFT_WRIST) > hip_y - 0.15 * th and
                Y(RIGHT_WRIST) > hip_y - 0.15 * th and
                abs(X(LEFT_WRIST) - X(RIGHT_WRIST)) > 1.3 * sw
            )
            if wrists_behind_and_down:
                return CelebrationMatch("ronaldo", "Cristiano Ronaldo", "siuuu_landing", "Siuuu Landing", 0.90, bbox)

            # B: PEACE OF MIND (hands clasped flat over center chest)
            hands_clasped = dist(LEFT_WRIST, RIGHT_WRIST) < 0.35 * sw
            near_center_chest = (
                abs((Y(LEFT_WRIST) + Y(RIGHT_WRIST)) / 2.0 - (shoulder_y + 0.35 * th)) < 0.2 * th
            )
            if hands_clasped and near_center_chest:
                return CelebrationMatch("ronaldo", "Cristiano Ronaldo", "peace_of_mind", "Peace of Mind", 0.92, bbox)

        return None
