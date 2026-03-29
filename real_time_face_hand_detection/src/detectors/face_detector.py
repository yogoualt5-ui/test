"""Face detection utilities backed by MediaPipe."""

from __future__ import annotations

from typing import List, Tuple

import mediapipe as mp

from config import settings

BoundingBox = Tuple[int, int, int, int]



def _load_face_solution():
    """Return MediaPipe face detection module with compatibility fallback.

    Some environments expose `mediapipe.python.solutions` but not
    `mediapipe.solutions` at the package root.
    """
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_detection"):
        return solutions.face_detection

    try:
        from mediapipe.python import solutions as mp_solutions
    except Exception as exc:  # pragma: no cover - compatibility guard
        raise RuntimeError(
            "MediaPipe face detection API is unavailable. "
            "Reinstall the official `mediapipe` package and remove conflicting "
            "packages/modules named `mediapipe`."
        ) from exc

    if not hasattr(mp_solutions, "face_detection"):
        raise RuntimeError(
            "This MediaPipe installation does not provide `face_detection`. "
            "Install a version that includes solutions APIs (e.g., 0.10.x)."
        )

    return mp_solutions.face_detection


_mp_face_detection = _load_face_solution()
_face_detector = _mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=settings.FACE_DETECTION_CONFIDENCE,
)


def detect_faces(rgb_frame) -> List[BoundingBox]:
    """Detect faces in an RGB frame and return pixel bounding boxes.

    Args:
        rgb_frame: RGB image as a numpy array of shape (H, W, C).

    Returns:
        A list of (xmin, ymin, xmax, ymax) pixel coordinates.
    """
    results = _face_detector.process(rgb_frame)
    if not results.detections:
        return []

    frame_height, frame_width = rgb_frame.shape[:2]
    boxes: List[BoundingBox] = []

    for detection in results.detections:
        rel_box = detection.location_data.relative_bounding_box
        xmin = max(0, int(rel_box.xmin * frame_width))
        ymin = max(0, int(rel_box.ymin * frame_height))
        width = int(rel_box.width * frame_width)
        height = int(rel_box.height * frame_height)

        xmax = min(frame_width - 1, xmin + width)
        ymax = min(frame_height - 1, ymin + height)

        boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def close_face_detector() -> None:
    """Release MediaPipe resources."""
    _face_detector.close()
