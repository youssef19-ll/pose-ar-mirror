"""Canvas compositor, broadcast HUD, and visual effects rendering."""

import cv2
import numpy as np
from src.config import (
    PANE_WIDTH,
    PANE_HEIGHT,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    COLOR_BG_DARK,
    COLOR_ACCENT,
    COLOR_CYAN,
    COLOR_TEXT,
    COLOR_ALERT,
    COLOR_SUCCESS
)


class CanvasRenderer:
    """Combines dual panes, draws broadcast HUD, divider, and interactive alerts."""

    def __init__(self, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT):
        self.width = width
        self.height = height
        self.pane_width = PANE_WIDTH
        self.pane_height = PANE_HEIGHT

    def compose_dual_canvas(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        fps: float = 30.0,
        is_calibrated: bool = False,
        is_calibrating: bool = False,
        calib_remaining: float = 0.0,
        dwell_progress: float = 0.0,
        detected_celebration: str = None,
        sound_muted: bool = False
    ) -> np.ndarray:
        """Assembles the final split-screen broadcast canvas."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Place Left & Right frames
        canvas[0:self.pane_height, 0:self.pane_width] = left_frame
        canvas[0:self.pane_height, self.pane_width:self.width] = right_frame

        # 1. Draw Sleek Center Broadcast Divider
        self._draw_divider(canvas)

        # 2. Draw Top Broadcast Scorebug / Bar
        self._draw_top_bar(canvas, fps, is_calibrated, sound_muted)

        # 3. Draw Bottom Action Keys Drawer
        self._draw_bottom_bar(canvas)

        # 4. Draw Dwell / Hold Progress Bar (Left Pane)
        if dwell_progress > 0.0:
            self._draw_dwell_gauge(canvas, dwell_progress, detected_celebration)

        # 5. Draw Calibration Overlay (if currently calibrating)
        if is_calibrating:
            self._draw_calibration_modal(canvas, calib_remaining)

        return canvas

    def _draw_divider(self, canvas: np.ndarray):
        """Draws a vertical broadcast divider with glow."""
        x = self.pane_width
        # Dark border
        cv2.line(canvas, (x - 2, 0), (x - 2, self.height), (20, 20, 20), 1)
        cv2.line(canvas, (x + 2, 0), (x + 2, self.height), (20, 20, 20), 1)
        # Gold neon center line
        cv2.line(canvas, (x, 0), (x, self.height), COLOR_ACCENT, 2)

        # Small diamond badge at center
        cy = self.height // 2
        pts = np.array([[x, cy - 12], [x + 10, cy], [x, cy + 12], [x - 10, cy]], np.int32)
        cv2.fillPoly(canvas, [pts], COLOR_ACCENT)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1)

    def _draw_top_bar(self, canvas: np.ndarray, fps: float, is_calibrated: bool, sound_muted: bool):
        """Draws top broadcast status bar with Champions League styling."""
        # Top bar background with alpha blending
        bar_height = 42
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, bar_height), (12, 14, 22), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, bar_height), (self.width, bar_height), (60, 70, 90), 1)

        # Left label: User / Live Mirror
        cv2.putText(canvas, "LIVE CAMERA (YOU)", (20, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_TEXT, 1)

        # Center Title Badge
        title = "CHAMPIONS CELEBRATIONS AR"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        cv2.putText(canvas, title, (self.width // 2 - tw // 2, 27),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_ACCENT, 2)

        # Right label: Virtual Pitch
        cv2.putText(canvas, "VIRTUAL PITCH (SUMMONS)", (self.pane_width + 20, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_CYAN, 1)

        # Calibration Pill Badge
        calib_text = "ROOM CALIBRATED" if is_calibrated else "PRESS 'C' TO CALIBRATE"
        calib_color = COLOR_SUCCESS if is_calibrated else COLOR_ALERT
        cv2.putText(canvas, calib_text, (self.width - 250, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, calib_color, 1)

        # Audio status & FPS
        audio_str = "MUTE" if sound_muted else "AUDIO ON"
        cv2.putText(canvas, f"{int(fps)} FPS | {audio_str}", (self.width - 95, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    def _draw_bottom_bar(self, canvas: np.ndarray):
        """Draws bottom control drawer hints."""
        bar_y = self.height - 32
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, bar_y), (self.width, self.height), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, bar_y), (self.width, bar_y), (50, 60, 80), 1)

        controls = (
            "[C] Calibrate Room    "
            "[S] Take Snapshot    "
            "[TAB] Roster / Help    "
            "[G] Photo Gallery    "
            "[M] Toggle Sound    "
            "[Q] Quit"
        )
        cv2.putText(canvas, controls, (24, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 215, 220), 1)

    def _draw_dwell_gauge(self, canvas: np.ndarray, progress: float, celebration_name: str):
        """Draws a futuristic charging meter on the left pane."""
        # Meter dimensions
        bar_w = 280
        bar_h = 16
        bar_x = (self.pane_width - bar_w) // 2
        bar_y = self.height - 70

        # Background container
        cv2.rectangle(canvas, (bar_x - 4, bar_y - 4), (bar_x + bar_w + 4, bar_y + bar_h + 4), (20, 24, 30), -1)
        cv2.rectangle(canvas, (bar_x - 4, bar_y - 4), (bar_x + bar_w + 4, bar_y + bar_h + 4), COLOR_ACCENT, 1)

        # Fill progress
        fill_w = int(bar_w * min(1.0, max(0.0, progress)))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), COLOR_ACCENT, -1)

        # Label above bar
        label = f"HOLDING: {celebration_name.upper()} ({int(progress * 100)}%)" if celebration_name else "HOLDING..."
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.putText(canvas, label, (self.pane_width // 2 - lw // 2, bar_y - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    def _draw_calibration_modal(self, canvas: np.ndarray, remaining_sec: float):
        """Renders countdown modal over entire canvas when calibrating room."""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        box_w, box_h = 560, 220
        bx = (self.width - box_w) // 2
        by = (self.height - box_h) // 2
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), (25, 30, 42), -1)
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), COLOR_CYAN, 2)

        # Modal text
        cv2.putText(canvas, "CALIBRATING ROOM BACKGROUND", (bx + 40, by + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_ACCENT, 2)
        cv2.putText(canvas, "Please STEP OUT of the camera frame!", (bx + 80, by + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1)

        count_str = f"Capturing in: {remaining_sec:.1f}s"
        (cw, _), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.putText(canvas, count_str, (self.width // 2 - cw // 2, by + 165),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, COLOR_CYAN, 2)
