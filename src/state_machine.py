"""Celebration State Machine with hold-to-activate dwell timer and debounce."""

import time
from collections import deque
from typing import Optional
from src.config import DWELL_TIME_SEC, DEBOUNCE_GRACE_SEC
from src.celebrations import CelebrationMatch


class CelebrationStateMachine:
    """Manages dwell charging, debounce decay, and active state transitions per user."""

    def __init__(self, dwell_time: float = DWELL_TIME_SEC, debounce_time: float = DEBOUNCE_GRACE_SEC):
        self.dwell_time = dwell_time
        self.debounce_time = debounce_time

        # Current state
        self.current_match: Optional[CelebrationMatch] = None
        self.is_active = False
        self.dwell_start_time = 0.0
        self.last_detected_time = 0.0
        self.dwell_progress = 0.0
        self.alpha = 0.0  # Smooth visual opacity (0.0 to 1.0)

        # Asset history manager: ensures no repetition from last 2 triggers per celebration
        # Dict: celebration_key -> deque(maxlen=2)
        self.asset_history: dict[str, deque] = {}

    def update(self, detected_match: Optional[CelebrationMatch], current_time: float = None):
        """
        Updates the state machine for the current frame.
        Returns:
            is_active (bool): Whether the celebration is currently triggered.
            progress (float): Dwell charging progress (0.0 to 1.0).
            active_match (Optional[CelebrationMatch]): Currently held/active celebration.
            alpha (float): Visual opacity for overlay blending (0.0 to 1.0).
        """
        if current_time is None:
            current_time = time.time()

        if detected_match is not None:
            # If switching to a completely different celebration, restart dwell
            if (self.current_match is None or
                    self.current_match.celebration_id != detected_match.celebration_id or
                    self.current_match.player_id != detected_match.player_id):
                self.current_match = detected_match
                self.dwell_start_time = current_time
                self.dwell_progress = 0.0

            self.last_detected_time = current_time

            # Compute dwell progress
            elapsed = current_time - self.dwell_start_time
            self.dwell_progress = min(1.0, elapsed / self.dwell_time)

            if self.dwell_progress >= 1.0:
                self.is_active = True

        else:
            # Pose was not detected in this frame - check debounce grace period
            time_since_last_seen = current_time - self.last_detected_time

            if time_since_last_seen > self.debounce_time:
                # Grace period expired: dismiss celebration
                self.is_active = False
                self.dwell_progress = 0.0
                self.current_match = None
            else:
                # Within grace period: maintain active state or progress
                pass

        # Smooth alpha interpolation
        target_alpha = 1.0 if self.is_active else 0.0
        self.alpha += (target_alpha - self.alpha) * 0.25
        if self.alpha < 0.02:
            self.alpha = 0.0

        return self.is_active, self.dwell_progress, self.current_match, self.alpha

    def pick_next_asset(self, celebration_key: str, available_assets: list[str]) -> Optional[str]:
        """
        Picks an asset from available_assets such that it is different from at least
        the last 2 shown before (Point 1 requirement).
        """
        if not available_assets:
            return None

        if celebration_key not in self.asset_history:
            self.asset_history[celebration_key] = deque(maxlen=2)

        history = self.asset_history[celebration_key]

        # Candidates that are not in the last 2 history buffer
        candidates = [a for a in available_assets if a not in history]
        if not candidates:
            # If all are exhausted or pool < 3, pick least recently used
            candidates = available_assets

        import random
        chosen = random.choice(candidates)
        history.append(chosen)
        return chosen
