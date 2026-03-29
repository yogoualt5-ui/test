"""Main entry point for real-time face and hand detection."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from startup_checker import run_startup_checks


def main() -> None:
    if not run_startup_checks():
        return

    import cv2

    from detectors.face_detector import close_face_detector, detect_faces
    from detectors.hand_detector import close_hand_detector, detect_hands
    from utils.drawing import draw_face_bbox, draw_hand_bbox, draw_hand_landmarks, draw_text
    from utils.fps import FPSCounter

    cap = cv2.VideoCapture(settings.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Could not open camera with CAMERA_ID={settings.CAMERA_ID}")
        return

    fps_counter = FPSCounter(window_size=10)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera.")
                break

            if frame.shape[1] != settings.FRAME_WIDTH or frame.shape[0] != settings.FRAME_HEIGHT:
                frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_boxes = detect_faces(rgb_frame) if settings.USE_FACE_DETECTION else []
            hands = detect_hands(rgb_frame) if settings.USE_HAND_DETECTION else []

            for face_bbox in face_boxes:
                draw_face_bbox(frame, face_bbox, settings.DRAW_BBOX_COLOR)

            for hand in hands:
                draw_hand_bbox(frame, hand["bbox"], settings.DRAW_BBOX_COLOR)
                draw_hand_landmarks(frame, hand["landmarks"], settings.DRAW_LANDMARK_COLOR)

            fps = fps_counter.update()
            draw_text(frame, f"FPS: {fps:.1f}", (10, 30), 0.8, (0, 255, 255))
            draw_text(frame, f"Faces: {len(face_boxes)}  Hands: {len(hands)}", (10, 60), 0.6, (255, 255, 255))

            cv2.imshow("Face & Hand Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == settings.ESC_KEY:
                break
    finally:
        cap.release()
        close_face_detector()
        close_hand_detector()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
