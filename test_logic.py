"""Pure-logic tests (no webcam, no MediaPipe): celebration rules, state machine, overlay card.

Run with:  python3 test_logic.py
"""

from types import SimpleNamespace

import numpy as np

from src import celebrations as C
from src.celebrations import CelebrationDetector, CelebrationMatch
from src.state_machine import CelebrationStateMachine
from src.player_overlay import PlayerOverlayManager


# ---------------------------------------------------------------------------
# Synthetic landmarks
# ---------------------------------------------------------------------------
# Normalized coordinates as MediaPipe sees a (mirrored) 640x480 frame of a user facing
# the camera: LEFT_* landmarks sit at larger x.
NEUTRAL = {
    C.NOSE: (0.50, 0.22),
    C.LEFT_EAR: (0.535, 0.22), C.RIGHT_EAR: (0.465, 0.22),
    C.LEFT_SHOULDER: (0.58, 0.35), C.RIGHT_SHOULDER: (0.42, 0.35),
    C.LEFT_ELBOW: (0.60, 0.50), C.RIGHT_ELBOW: (0.40, 0.50),
    C.LEFT_WRIST: (0.60, 0.63), C.RIGHT_WRIST: (0.40, 0.63),
    C.LEFT_INDEX: (0.60, 0.66), C.RIGHT_INDEX: (0.40, 0.66),
    C.LEFT_HIP: (0.555, 0.65), C.RIGHT_HIP: (0.445, 0.65),
}


def make_pose(overrides=None, scale=1.0, hidden=()):
    pts = dict(NEUTRAL)
    pts.update(overrides or {})
    lm = []
    for i in range(33):
        x, y = pts.get(i, (0.5, 0.8))
        # Scale around the frame center to simulate moving closer / farther
        x = 0.5 + (x - 0.5) * scale
        y = 0.5 + (y - 0.5) * scale
        lm.append(SimpleNamespace(x=x, y=y, z=0.0, visibility=0.1 if i in hidden else 0.99))
    return lm


OPEN_ARMS = {C.LEFT_ELBOW: (0.70, 0.35), C.RIGHT_ELBOW: (0.30, 0.35),
             C.LEFT_WRIST: (0.80, 0.35), C.RIGHT_WRIST: (0.20, 0.35),
             C.LEFT_INDEX: (0.83, 0.35), C.RIGHT_INDEX: (0.17, 0.35)}
SIUUU = {C.LEFT_WRIST: (0.72, 0.68), C.RIGHT_WRIST: (0.28, 0.68)}
HIPS = (C.LEFT_HIP, C.RIGHT_HIP)


def celebration_of(lm):
    m = CelebrationDetector().evaluate(lm)
    return None if m is None else m.celebration_id


def test_celebration_rules():
    assert celebration_of(make_pose()) is None, "Neutral pose must not trigger"
    assert celebration_of(make_pose(OPEN_ARMS)) == "open_arms"
    assert celebration_of(make_pose(SIUUU)) == "siuuu_landing"
    print("[OK] Basic rules (neutral / open arms / siuuu_landing id)")


def test_scale_invariance():
    for scale in (0.6, 0.8, 1.2):
        assert celebration_of(make_pose(scale=scale)) is None, f"neutral @ {scale}"
        assert celebration_of(make_pose(OPEN_ARMS, scale=scale)) == "open_arms", f"open arms @ {scale}"
        assert celebration_of(make_pose(SIUUU, scale=scale)) == "siuuu_landing", f"siuuu @ {scale}"
    print("[OK] Rules are invariant to camera distance")


def test_visibility_gating():
    # Hips out of frame: hip-based rules must not fire on hallucinated hip positions
    assert celebration_of(make_pose(SIUUU, hidden=HIPS)) is None
    # ...but rules that don't need the hips still work
    assert celebration_of(make_pose(OPEN_ARMS, hidden=HIPS)) == "open_arms"
    # No shoulders: nothing can be evaluated
    assert celebration_of(make_pose(OPEN_ARMS, hidden=(C.LEFT_SHOULDER,))) is None
    # Hidden wrists cannot produce an arm-based match
    assert celebration_of(make_pose(OPEN_ARMS, hidden=(C.LEFT_WRIST,))) is None
    print("[OK] Low-visibility landmarks are ignored")


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
A = CelebrationMatch("bellingham", "Jude Bellingham", "open_arms", "Open Arms", 0.95, (0.2, 0.1, 0.8, 0.9))
B = CelebrationMatch("messi", "Lionel Messi", "to_the_heavens", "To The Heavens", 0.9, (0.2, 0.1, 0.8, 0.9))
FPS = 30.0


def run(sm, match, t, duration, fps=FPS):
    """Feeds `match` for `duration` seconds starting at t; returns end time."""
    n = int(round(duration * fps))
    for k in range(1, n + 1):
        sm.update(match, current_time=t + k / fps)
    return t + n / fps


def test_dwell_and_rising_edge():
    sm = CelebrationStateMachine(dwell_time=0.5, debounce_time=0.2)
    t = 1000.0
    sm.update(A, current_time=t)
    assert not sm.is_active and sm.dwell_progress == 0.0
    edges = 0
    for k in range(1, 31):
        sm.update(A, current_time=t + k / FPS)
        edges += sm.just_activated
    assert sm.is_active and sm.dwell_progress == 1.0
    assert edges == 1, f"just_activated must fire exactly once, got {edges}"
    print("[OK] Dwell charging + single rising edge")


def test_fade_out_keeps_display_match():
    sm = CelebrationStateMachine(dwell_time=0.5, debounce_time=0.2)
    t = run(sm, A, 1000.0, 1.5)
    assert sm.is_active and sm.alpha == 1.0 and sm.display_match is A

    # Drop the pose: grace period keeps it fully active
    t = run(sm, None, t, 0.1)
    assert sm.is_active and sm.display_match is A

    # Just after the grace period: inactive, but still visible and fading
    t = run(sm, None, t, 0.15)
    assert not sm.is_active and sm.current_match is None
    assert sm.display_match is A and 0.0 < sm.alpha < 1.0, f"alpha={sm.alpha}"

    # Eventually fully faded and cleared
    run(sm, None, t, 1.0)
    assert sm.alpha == 0.0 and sm.display_match is None
    print("[OK] Overlay fades out instead of vanishing")


def test_switch_requires_new_dwell():
    sm = CelebrationStateMachine(dwell_time=0.5, debounce_time=0.2)
    t = run(sm, A, 1000.0, 1.0)
    assert sm.is_active

    t = run(sm, B, t, 0.1)
    assert not sm.is_active, "A different pose must not inherit the active state"
    assert sm.current_match is B and sm.dwell_progress < 1.0
    assert sm.display_match is A, "Previous celebration keeps fading out"

    run(sm, B, t, 0.6)
    assert sm.is_active and sm.display_match is B
    print("[OK] Switching pose requires a fresh dwell")


def test_fade_is_frame_rate_independent():
    alphas = []
    for fps in (15.0, 30.0, 120.0):
        sm = CelebrationStateMachine(dwell_time=0.0, debounce_time=0.2)
        sm.update(A, current_time=0.0)
        run(sm, A, 0.0, 0.2, fps=fps)  # 0.2 s = whole number of frames at every fps
        alphas.append(sm.alpha)
    assert max(alphas) - min(alphas) < 0.05, f"alpha depends on fps: {alphas}"
    print("[OK] Fade speed independent of frame rate", [round(a, 3) for a in alphas])


# ---------------------------------------------------------------------------
# Overlay card
# ---------------------------------------------------------------------------
def test_card_drawings_are_opaque():
    card = PlayerOverlayManager()._generate_player_card("bellingham", "open_arms")
    accent = np.array((0, 200, 255))
    accent_px = np.all(card[:, :, :3] == accent, axis=2)
    assert accent_px.any(), "Accent drawings missing"
    assert (card[accent_px][:, 3] == 255).all(), "Accent text/lines must be opaque"
    print("[OK] Card text and lines are opaque")


if __name__ == "__main__":
    test_celebration_rules()
    test_scale_invariance()
    test_visibility_gating()
    test_dwell_and_rising_edge()
    test_fade_out_keeps_display_match()
    test_switch_requires_new_dwell()
    test_fade_is_frame_rate_independent()
    test_card_drawings_are_opaque()
    print("ALL LOGIC TESTS PASSED")
