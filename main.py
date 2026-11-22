"""Champions Celebration AR Mirror - Main Application Entry Point."""

import time
import cv2
import numpy as np

from src.config import (
    PANE_WIDTH,
    PANE_HEIGHT,
    SCREENSHOTS_DIR
)
from src.camera import CameraStream
from src.pose_detector import PoseSegmenter
from src.segmenter import BackgroundInpainter
from src.renderer import CanvasRenderer
from src.celebrations import CelebrationDetector
from src.state_machine import CelebrationStateMachine
from src.player_overlay import PlayerOverlayManager


def main():
    print("=" * 60)
    print("  CHAMPIONS CELEBRATIONS AR MIRROR - INITIALIZING")
    print("=" * 60)

    # Initialize vision and render pipeline
    camera = CameraStream(width=PANE_WIDTH, height=PANE_HEIGHT, mirror=True)
    pose_segmenter = PoseSegmenter(num_poses=2)
    inpainter = BackgroundInpainter()
    renderer = CanvasRenderer()

    # Step 3: Celebration recognition, dwell state machine, and overlay manager
    detector = CelebrationDetector()
    state_machine = CelebrationStateMachine()
    overlay_manager = PlayerOverlayManager()

    window_name = "Champions Celebrations AR Mirror"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps = 30.0
    sound_muted = False

    print("\n[Controls]:")
    print("  [C]   - Calibrate Room (Step out of camera to snapshot background)")
    print("  [S]   - Take Screenshot")
    print("  [M]   - Toggle Sound")
    print("  [Q]   - Quit Application\n")

    try:
        while True:
            # 1. Grab camera frame
            success, raw_frame = camera.read()
            if not success or raw_frame is None:
                print("[Main] Failed to read camera frame.")
                break

            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # 2. Process Calibration if triggered
            calib_remaining = 0.0
            if inpainter.is_calibrating:
                done, calib_remaining = inpainter.update_calibration(raw_frame)
                if done:
                    print("[Main] Calibration complete! Background saved.")

            # 3. Vision Pipeline: Multi-person Pose & Segmentation
            landmarks_list, seg_mask = pose_segmenter.process(raw_frame)

            # 4. Step 3: Evaluate Celebration Pose & Dwell State
            detected_match = None
            if landmarks_list and len(landmarks_list) > 0:
                detected_match = detector.evaluate(landmarks_list[0])

            is_active, progress, active_match, alpha = state_machine.update(
                detected_match, current_time=current_time
            )

            # 5. Prepare Left Pane (Live User + Tracking Skeleton HUD)
            left_pane = pose_segmenter.draw_skeletons(raw_frame, landmarks_list)

            # 6. Prepare Right Pane (Inpainted Room: Person Deleted)
            right_pane = inpainter.remove_person(raw_frame, seg_mask)

            # 7. Step 3: Overlay Player Cutout on Right Pane when active/summoned
            # display_match outlives the pose so the cutout can fade out smoothly.
            display_match = state_machine.display_match
            if display_match is not None and alpha > 0.0:
                right_pane = overlay_manager.overlay_player(right_pane, display_match, alpha)

            # 8. Compose Final Dual-Screen Broadcast Canvas
            celebration_title = None
            if active_match:
                celebration_title = f"{active_match.player_name} ({active_match.celebration_name})"

            canvas = renderer.compose_dual_canvas(
                left_frame=left_pane,
                right_frame=right_pane,
                fps=fps,
                is_calibrated=inpainter.is_calibrated,
                is_calibrating=inpainter.is_calibrating,
                calib_remaining=calib_remaining,
                dwell_progress=progress,
                detected_celebration=celebration_title,
                sound_muted=sound_muted
            )

            # 9. Display Canvas
            cv2.imshow(window_name, canvas)

            # 10. Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("[Main] Exiting...")
                break
            elif key == ord('c'):
                inpainter.start_calibration()
            elif key == ord('m'):
                sound_muted = not sound_muted
                print(f"[Main] Sound {'MUTED' if sound_muted else 'UNMUTED'}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = SCREENSHOTS_DIR / f"moment_{timestamp}.png"
                cv2.imwrite(str(save_path), canvas)
                print(f"[Main] Saved screenshot: {save_path}")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted (Ctrl+C).")
    finally:
        camera.release()
        pose_segmenter.close()
        cv2.destroyAllWindows()
        print("[Main] Clean shutdown complete.")


if __name__ == "__main__":
    main()
