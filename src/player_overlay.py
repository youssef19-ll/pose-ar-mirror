"""Player Cutout Overlay Engine with dynamic positioning and card generation."""

import cv2
import numpy as np
from pathlib import Path
from src.config import PLAYERS_DIR, PANE_WIDTH, PANE_HEIGHT, COLOR_ACCENT, COLOR_CYAN
from src.celebrations import CelebrationMatch


# Player branding themes (Jersey Color, Accent Color, Number)
PLAYER_THEMES = {
    "bellingham": {"primary": (245, 245, 245), "accent": (0, 200, 255), "number": "5", "club": "REAL MADRID"},
    "messi": {"primary": (230, 200, 100), "accent": (255, 120, 200), "number": "10", "club": "INTER MIAMI"},
    "ronaldo": {"primary": (40, 180, 240), "accent": (20, 40, 160), "number": "7", "club": "AL-NASSR"},
    "mbappe": {"primary": (160, 60, 20), "accent": (0, 215, 255), "number": "9", "club": "REAL MADRID"},
    "haaland": {"primary": (230, 210, 130), "accent": (80, 20, 80), "number": "9", "club": "MANCHESTER CITY"},
    "rashford": {"primary": (30, 30, 210), "accent": (255, 255, 255), "number": "10", "club": "MANCHESTER UNITED"}
}


class PlayerOverlayManager:
    """Loads or generates transparent player cutouts and composites them onto the right pane."""

    def __init__(self, assets_dir: Path = PLAYERS_DIR):
        self.assets_dir = assets_dir
        self.loaded_images: dict[str, np.ndarray] = {}

    def get_player_image(self, player_id: str, celebration_id: str) -> np.ndarray:
        """Returns BGRA image (with alpha channel) for the player celebration."""
        key = f"{player_id}_{celebration_id}"
        if key in self.loaded_images:
            return self.loaded_images[key]

        # Check if local image file exists (e.g. assets/players/bellingham_open_arms.png)
        img_path = self.assets_dir / f"{key}.png"
        if img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                self.loaded_images[key] = img
                return img

        # Procedurally generate a stylish football card cutout
        generated = self._generate_player_card(player_id, celebration_id)
        self.loaded_images[key] = generated
        return generated

    def _generate_player_card(self, player_id: str, celebration_id: str) -> np.ndarray:
        """Generates a high-res transparent broadcast card for the player."""
        card_w, card_h = 360, 520
        # BGRA canvas (4 channels, transparent background)
        card = np.zeros((card_h, card_w, 4), dtype=np.uint8)

        theme = PLAYER_THEMES.get(player_id, {"primary": (220, 220, 220), "accent": COLOR_ACCENT, "number": "00", "club": "CHAMPIONS"})

        # Card container with rounded styling
        margin = 12
        cv2.rectangle(card, (margin, margin), (card_w - margin, card_h - margin), (*theme["primary"], 235), -1)
        cv2.rectangle(card, (margin, margin), (card_w - margin, card_h - margin), (*theme["accent"], 255), 3)

        # Header bar
        cv2.rectangle(card, (margin, margin), (card_w - margin, margin + 45), (15, 18, 26, 255), -1)
        club_text = theme["club"]
        (cw, _), _ = cv2.getTextSize(club_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.putText(card, club_text, (card_w // 2 - cw // 2, margin + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, theme["accent"], 1)

        # Jersey Number Badge
        num_str = theme["number"]
        cv2.circle(card, (margin + 36, margin + 90), 24, (15, 18, 26, 255), -1)
        cv2.putText(card, num_str, (margin + 26, margin + 98),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, theme["accent"], 2)

        # Player Silhouette / Pose Art placeholder
        cx, cy = card_w // 2, card_h // 2
        cv2.circle(card, (cx, cy - 35), 45, (15, 18, 26, 255), -1)  # Head
        cv2.rectangle(card, (cx - 65, cy + 20), (cx + 65, cy + 120), (15, 18, 26, 255), -1)  # Body
        cv2.putText(card, "★ ★ ★", (cx - 36, cy + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme["accent"], 1)

        # Player Name Banner (Bottom)
        banner_y = card_h - 110
        cv2.rectangle(card, (margin, banner_y), (card_w - margin, card_h - margin), (15, 18, 26, 255), -1)
        cv2.line(card, (margin, banner_y), (card_w - margin, banner_y), theme["accent"], 2)

        name_str = player_id.upper()
        (nw, _), _ = cv2.getTextSize(name_str, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(card, name_str, (card_w // 2 - nw // 2, banner_y + 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        cel_str = celebration_id.replace("_", " ").upper()
        (clw, _), _ = cv2.getTextSize(cel_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(card, cel_str, (card_w // 2 - clw // 2, banner_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, theme["accent"], 1)

        return card

    def overlay_player(
        self,
        target_frame: np.ndarray,
        match: CelebrationMatch,
        alpha: float
    ) -> np.ndarray:
        """
        Overlays the player cutout on the target frame (right pane).
        Scales and positions the cutout to match the user's bounding box.
        """
        if alpha <= 0.01:
            return target_frame

        cutout = self.get_player_image(match.player_id, match.celebration_id)
        frame_h, frame_w = target_frame.shape[:2]

        # Use match.bbox (min_x, min_y, max_x, max_y)
        min_x, min_y, max_x, max_y = match.bbox
        user_w = (max_x - min_x) * frame_w
        user_h = (max_y - min_y) * frame_h
        center_x = int((min_x + max_x) * 0.5 * frame_w)
        center_y = int((min_y + max_y) * 0.5 * frame_h)

        # Scale cutout: height roughly 1.1x of user torso/body height
        target_h = int(max(180, min(frame_h * 0.9, user_h * 1.15)))
        aspect = cutout.shape[1] / cutout.shape[0]
        target_w = int(target_h * aspect)

        resized = cv2.resize(cutout, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Coordinates on target frame
        x1 = center_x - target_w // 2
        y1 = center_y - target_h // 2
        x2 = x1 + target_w
        y2 = y1 + target_h

        # Clip to target frame boundaries
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = target_w - max(0, x2 - frame_w)
        src_y2 = target_h - max(0, y2 - frame_h)

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(frame_w, x2)
        dst_y2 = min(frame_h, y2)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return target_frame

        # Alpha blend
        crop_cutout = resized[src_y1:src_y2, src_x1:src_x2]
        crop_target = target_frame[dst_y1:dst_y2, dst_x1:dst_x2]

        cutout_bgr = crop_cutout[:, :, :3]
        cutout_a = (crop_cutout[:, :, 3] / 255.0) * alpha

        # Broadcast alpha to 3 channels
        alpha_3d = np.repeat(cutout_a[:, :, np.newaxis], 3, axis=2)

        blended = (crop_target * (1.0 - alpha_3d) + cutout_bgr * alpha_3d).astype(np.uint8)
        target_frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended

        return target_frame
