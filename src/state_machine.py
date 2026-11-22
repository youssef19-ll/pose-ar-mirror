"""Celebration State Machine with hold-to-activate dwell timer and debounce."""

import math
import random
import time
from collections import deque
from typing import Optional
from src.config import DWELL_TIME_SEC, DEBOUNCE_GRACE_SEC, ALPHA_FADE_TAU_SEC
from src.celebrations import CelebrationMatch


def _same_celebration(a: Optional[CelebrationMatch], b: Optional[CelebrationMatch]) -> bool:
    return (a is not None and b is not None and
            a.player_id == b.player_id and a.celebration_id == b.celebration_id)


class CelebrationStateMachine:
    """Manages dwell charging, debounce grace, and active state transitions per user.

    Two matches are tracked separately:
      - ``current_match``: the pose currently being held / charged (drives the dwell gauge).
      - ``display_match``: the celebration shown on screen. It survives after the pose is
        dropped so the overlay can fade out, and is only replaced once a new pose completes
        its dwell.
    """

    def __init__(
        self,
        dwell_time: float = DWELL_TIME_SEC,
        debounce_time: float = DEBOUNCE_GRACE_SEC,
        fade_tau: float = ALPHA_FADE_TAU_SEC,
    ):
        self.dwell_time = dwell_time
        self.debounce_time = debounce_time
        self.fade_tau = fade_tau

        # Current state
        self.current_match: Optional[CelebrationMatch] = None
        self.display_match: Optional[CelebrationMatch] = None
        self.is_active = False
        self.just_activated = False  # True only on the frame where is_active rises
        self.dwell_start_time = 0.0
        self.last_detected_time = 0.0
        self.last_update_time: Optional[float] = None
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
            current_match (Optional[CelebrationMatch]): Pose currently held / charging.
            alpha (float): Visual opacity of ``display_match`` (0.0 to 1.0).
        """
        if current_time is None:
            current_time = time.time()
        dt = 0.0 if self.last_update_time is None else max(0.0, current_time - self.last_update_time)
        self.last_update_time = current_time

        was_active = self.is_active

        if detected_match is not None:
            if not _same_celebration(self.current_match, detected_match):
                # Switching to a different celebration: it must earn its own dwell
                self.is_active = False
                self.dwell_start_time = current_time
                self.dwell_progress = 0.0
            # Always keep the freshest match (bbox / confidence follow the user)
            self.current_match = detected_match
            self.last_detected_time = current_time

            elapsed = current_time - self.dwell_start_time
            self.dwell_progress = min(1.0, elapsed / self.dwell_time) if self.dwell_time > 0 else 1.0

            if self.dwell_progress >= 1.0:
                self.is_active = True

        elif self.current_match is not None:
            # Pose not detected this frame: hold state during the grace period
            if current_time - self.last_detected_time > self.debounce_time:
                self.is_active = False
                self.dwell_progress = 0.0
                self.current_match = None

        self.just_activated = self.is_active and not was_active
        if self.is_active:
            self.display_match = self.current_match

        # Frame-rate independent exponential easing
        target_alpha = 1.0 if self.is_active else 0.0
        if self.fade_tau > 0:
            k = 1.0 - math.exp(-dt / self.fade_tau)
        else:
            k = 1.0
        self.alpha += (target_alpha - self.alpha) * k
        if self.alpha < 0.02 and not self.is_active:
            self.alpha = 0.0
            self.display_match = None
        elif self.alpha > 0.98 and self.is_active:
            self.alpha = 1.0

        return self.is_active, self.dwell_progress, self.current_match, self.alpha

    def pick_next_asset(self, celebration_key: str, available_assets: list[str]) -> Optional[str]:
        """
        Picks an asset from available_assets such that it is different from at least
        the last 2 shown before.
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

        chosen = random.choice(candidates)
        history.append(chosen)
        return chosen
