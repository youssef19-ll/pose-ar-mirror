"""Comprehensive verification script for Step 1, Step 2, and Step 3."""

import time
import cv2
import numpy as np

from src.config import PANE_WIDTH, PANE_HEIGHT, SCREENSHOTS_DIR
from src.camera import CameraStream
from src.pose_detector import PoseSegmenter
from src.segmenter import BackgroundInpainter
from src.renderer import CanvasRenderer
from src.celebrations import CelebrationDetector, CelebrationMatch
from src.state_machine import CelebrationStateMachine
from src.player_overlay import PlayerOverlayManager


def test_celebrations_and_state_machine():
    print("Testing Step 3: Celebrations, State Machine, and Player Overlay...")

    # 1. State machine & history buffer test
    sm = CelebrationStateMachine(dwell_time=0.5, debounce_time=0.2)
    dummy_match = CelebrationMatch("bellingham", "Jude Bellingham", "open_arms", "Open Arms", 0.95, (0.2, 0.1, 0.8, 0.9))

    # Test dwell accumulation
    t0 = 1000.0
    active, prog, match, _ = sm.update(dummy_match, current_time=t0)
    assert not active, "Should not be active at t=0"
    assert prog == 0.0

    active, prog, match, _ = sm.update(dummy_match, current_time=t0 + 0.3)
    assert not active
    assert 0.5 < prog < 0.7

    active, prog, match, _ = sm.update(dummy_match, current_time=t0 + 0.6)
    assert active, "Should be active after dwell time exceeds 0.5s"
    assert prog == 1.0
    print("[OK] State machine dwell charging verified.")

    # Test history buffer (no repeat in last 2)
    assets = ["bellingham_1.png", "bellingham_2.png", "bellingham_3.png", "bellingham_4.png"]
    a1 = sm.pick_next_asset("bellingham_open_arms", assets)
    a2 = sm.pick_next_asset("bellingham_open_arms", assets)
    a3 = sm.pick_next_asset("bellingham_open_arms", assets)
    assert a3 != a2 and a3 != a1, f"Repeated asset in buffer: {a1}, {a2}, {a3}"
    print("[OK] History buffer (no repeat from last 2) verified:", a1, "->", a2, "->", a3)

    # 2. Player Overlay Manager
    overlay_mgr = PlayerOverlayManager()
    bg_frame = np.full((PANE_HEIGHT, PANE_WIDTH, 3), 45, dtype=np.uint8)
    overlaid = overlay_mgr.overlay_player(bg_frame.copy(), dummy_match, alpha=1.0)
    assert overlaid.shape == bg_frame.shape
    assert not np.array_equal(overlaid, bg_frame), "Overlay should modify frame"
    print("[OK] Player overlay rendering & procedural card generation verified.")

    # 3. Full Integrated Dual-Canvas Verification
    camera = CameraStream(camera_index=0, width=PANE_WIDTH, height=PANE_HEIGHT, mirror=True)
    success, raw_frame = camera.read()
    assert success

    pose_segmenter = PoseSegmenter()
    landmarks_list, seg_mask = pose_segmenter.process(raw_frame)

    inpainter = BackgroundInpainter()
    inpainter.is_calibrated = True
    inpainter.clean_bg = raw_frame.copy()

    # Inpaint right pane and overlay player
    right_pane = inpainter.remove_person(raw_frame, seg_mask)
    right_pane = overlay_mgr.overlay_player(right_pane, dummy_match, alpha=1.0)

    left_pane = pose_segmenter.draw_skeletons(raw_frame, landmarks_list)

    renderer = CanvasRenderer()
    canvas = renderer.compose_dual_canvas(
        left_frame=left_pane,
        right_frame=right_pane,
        fps=30.0,
        is_calibrated=True,
        is_calibrating=False,
        calib_remaining=0.0,
        dwell_progress=1.0,
        detected_celebration="Jude Bellingham (Open Arms)",
        sound_muted=False
    )

    test_out = SCREENSHOTS_DIR / "step3_verification.png"
    cv2.imwrite(str(test_out), canvas)
    print(f"[OK] Saved Step 3 verification snapshot to {test_out}")

    camera.release()
    pose_segmenter.close()
    print("ALL STEP 3 TESTS PASSED WITH 100% SUCCESS!")


if __name__ == "__main__":
    test_celebrations_and_state_machine()
