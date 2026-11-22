"""Celebration detector for the 6 iconic football stars."""

import math
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class CelebrationMatch:
    player_id: str
    player_name: str
    celebration_id: str
    celebration_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y (normalized)


class CelebrationDetector:
    """Evaluates 3D body pose landmarks to detect 6 football stars' celebrations."""

    def __init__(self):
        pass

    def evaluate(self, landmarks) -> Optional[CelebrationMatch]:
        """
        Evaluates a single person's landmarks list (33 NormalizedLandmarks).
        Returns CelebrationMatch if a celebration pose is matched, else None.
        """
        if not landmarks or len(landmarks) < 25:
            return None

        # Extract key points
        lm = landmarks

        # Helper Euclidean distance
        def dist(i1, i2):
            return math.hypot(lm[i1].x - lm[i2].x, lm[i1].y - lm[i2].y)

        # Baseline body metrics for normalization
        shoulder_width = max(0.01, dist(LEFT_SHOULDER, RIGHT_SHOULDER))
        torso_height = max(0.01, dist(LEFT_SHOULDER, LEFT_HIP))
        head_y = min(lm[LEFT_EAR].y, lm[RIGHT_EAR].y, lm[NOSE].y)
        shoulder_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        hip_y = (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2.0

        # Compute bounding box
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        bbox = (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))

        # -------------------------------------------------------------
        # 1. JUDE BELLINGHAM
        # -------------------------------------------------------------
        # Celebration A: OPEN ARMS (Wrists wide apart, near shoulder level)
        arm_span = abs(lm[LEFT_WRIST].x - lm[RIGHT_WRIST].x)
        wrists_near_shoulder_level = (
            abs(lm[LEFT_WRIST].y - shoulder_y) < 0.35 * torso_height and
            abs(lm[RIGHT_WRIST].y - shoulder_y) < 0.35 * torso_height
        )
        if arm_span > 2.0 * shoulder_width and wrists_near_shoulder_level:
            conf = min(1.0, (arm_span / (2.0 * shoulder_width)) * 0.9)
            return CelebrationMatch("bellingham", "Jude Bellingham", "open_arms", "Open Arms", conf, bbox)

        # -------------------------------------------------------------
        # 2. LIONEL MESSI
        # -------------------------------------------------------------
        # Celebration A: TO THE HEAVENS (Both wrists high above head)
        both_wrists_above_head = (
            lm[LEFT_WRIST].y < head_y - 0.15 * torso_height and
            lm[RIGHT_WRIST].y < head_y - 0.15 * torso_height
        )
        if both_wrists_above_head:
            height_factor = min(1.0, (head_y - max(lm[LEFT_WRIST].y, lm[RIGHT_WRIST].y)) / torso_height + 0.5)
            return CelebrationMatch("messi", "Lionel Messi", "to_the_heavens", "To The Heavens", height_factor, bbox)

        # Celebration B: TOPO GIGIO (Hands cupped behind ears)
        left_hand_ear = dist(LEFT_WRIST, LEFT_EAR)
        right_hand_ear = dist(RIGHT_WRIST, RIGHT_EAR)
        if left_hand_ear < 0.45 * shoulder_width and right_hand_ear < 0.45 * shoulder_width:
            return CelebrationMatch("messi", "Lionel Messi", "topo_gigio", "Topo Gigio", 0.95, bbox)

        # -------------------------------------------------------------
        # 3. MARCUS RASHFORD
        # -------------------------------------------------------------
        # Celebration A: TEMPLE TAP (One index finger touching temple, other arm relaxed)
        left_index_temple = dist(LEFT_INDEX, LEFT_EAR)
        right_index_temple = dist(RIGHT_INDEX, RIGHT_EAR)
        left_tap = left_index_temple < 0.35 * shoulder_width and lm[RIGHT_WRIST].y > shoulder_y + 0.1
        right_tap = right_index_temple < 0.35 * shoulder_width and lm[LEFT_WRIST].y > shoulder_y + 0.1
        if left_tap or right_tap:
            return CelebrationMatch("rashford", "Marcus Rashford", "temple_tap", "Mind / Temple Tap", 0.95, bbox)

        # -------------------------------------------------------------
        # 4. KYLIAN MBAPPÉ
        # -------------------------------------------------------------
        # Celebration A: FOLDED ARMS (Wrists crossed near opposite armpits/chest)
        wrist_to_opposite_shoulder_left = dist(LEFT_WRIST, RIGHT_SHOULDER)
        wrist_to_opposite_shoulder_right = dist(RIGHT_WRIST, LEFT_SHOULDER)
        wrists_crossed = (
            wrist_to_opposite_shoulder_left < 0.85 * shoulder_width and
            wrist_to_opposite_shoulder_right < 0.85 * shoulder_width and
            lm[LEFT_WRIST].y < hip_y and lm[RIGHT_WRIST].y < hip_y
        )
        if wrists_crossed:
            return CelebrationMatch("mbappe", "Kylian Mbappe", "folded_arms", "Folded Arms", 0.92, bbox)

        # -------------------------------------------------------------
        # 5. ERLING HAALAND
        # -------------------------------------------------------------
        # Celebration A: LOTUS / ZEN MEDITATION (Hands cupped together at chest height, elbows outward)
        hands_distance = dist(LEFT_WRIST, RIGHT_WRIST)
        wrists_at_chest = (
            hands_distance < 0.6 * shoulder_width and
            shoulder_y < lm[LEFT_WRIST].y < hip_y and
            shoulder_y < lm[RIGHT_WRIST].y < hip_y and
            lm[LEFT_ELBOW].x > lm[LEFT_SHOULDER].x and  # Elbows flaring out
            lm[RIGHT_ELBOW].x < lm[RIGHT_SHOULDER].x
        )
        if wrists_at_chest:
            return CelebrationMatch("haaland", "Erling Haaland", "zen_lotus", "Zen Meditation", 0.90, bbox)

        # Celebration B: FINGER TO EAR (One finger pointing to ear, other down)
        left_ear_touch = dist(LEFT_WRIST, LEFT_EAR) < 0.4 * shoulder_width and lm[RIGHT_WRIST].y > hip_y - 0.2
        right_ear_touch = dist(RIGHT_WRIST, RIGHT_EAR) < 0.4 * shoulder_width and lm[LEFT_WRIST].y > hip_y - 0.2
        if left_ear_touch or right_ear_touch:
            return CelebrationMatch("haaland", "Erling Haaland", "finger_to_ear", "Finger To Ear", 0.88, bbox)

        # -------------------------------------------------------------
        # 6. CRISTIANO RONALDO
        # -------------------------------------------------------------
        # Celebration A: SIUUU LANDING (Wrists swept backward and downward, below hips)
        wrists_behind_and_down = (
            lm[LEFT_WRIST].y > hip_y - 0.05 and
            lm[RIGHT_WRIST].y > hip_y - 0.05 and
            abs(lm[LEFT_WRIST].x - lm[RIGHT_WRIST].x) > 1.3 * shoulder_width
        )
        if wrists_behind_and_down:
            return CelebrationMatch("ronaldo", "Cristiano Ronaldo", "siuuu", "Siuuu Landing", 0.90, bbox)

        # Celebration B: PEACE OF MIND / NAP (Hands clasped flat over center chest)
        hands_clasped = hands_distance < 0.35 * shoulder_width
        near_center_chest = (
            abs((lm[LEFT_WRIST].y + lm[RIGHT_WRIST].y) / 2.0 - (shoulder_y + 0.35 * torso_height)) < 0.2 * torso_height
        )
        if hands_clasped and near_center_chest:
            return CelebrationMatch("ronaldo", "Cristiano Ronaldo", "peace_of_mind", "Peace of Mind", 0.92, bbox)

        return None
