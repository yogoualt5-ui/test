"""Hand detection utilities backed by MediaPipe."""

from __future__ import annotations

from typing import Dict, List, Tuple

import mediapipe as mp

from config import settings

BoundingBox = Tuple[int, int, int, int]
Landmark = Tuple[int, int, float]
HandDetection = Dict[str, object]



def _load_hands_solution():
    """Return MediaPipe hands module with compatibility fallback."""
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "hands"):
        return solutions.hands

    try:
        from mediapipe.python import solutions as mp_solutions
    except Exception as exc:  # pragma: no cover - compatibility guard
        raise RuntimeError(
            "MediaPipe hands API is unavailable. "
            "Reinstall the official `mediapipe` package and remove conflicting "
            "packages/modules named `mediapipe`."
        ) from exc

    if not hasattr(mp_solutions, "hands"):
        raise RuntimeError(
            "This MediaPipe installation does not provide `hands`. "
            "Install a version that includes solutions APIs (e.g., 0.10.x)."
        )

    return mp_solutions.hands


_mp_hands = _load_hands_solution()
_hand_detector = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=settings.MAX_HANDS,
    min_detection_confidence=settings.HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=settings.HAND_TRACKING_CONFIDENCE,
)


def detect_hands(rgb_frame) -> List[HandDetection]:
    """Detect hands in an RGB frame.

    Returns one dictionary per hand with:
    - bbox: (xmin, ymin, xmax, ymax) in pixels
    - landmarks: list of 21 tuples (x_px, y_px, z_norm)
    """
    results = _hand_detector.process(rgb_frame)
    if not results.multi_hand_landmarks:
        return []

    frame_height, frame_width = rgb_frame.shape[:2]
    hands: List[HandDetection] = []

    for hand_landmarks in results.multi_hand_landmarks:
        points: List[Landmark] = []
        xs: List[int] = []
        ys: List[int] = []

        for landmark in hand_landmarks.landmark:
            x_px = int(landmark.x * frame_width)
            y_px = int(landmark.y * frame_height)
            x_px = max(0, min(frame_width - 1, x_px))
            y_px = max(0, min(frame_height - 1, y_px))

            points.append((x_px, y_px, float(landmark.z)))
            xs.append(x_px)
            ys.append(y_px)

        bbox: BoundingBox = (min(xs), min(ys), max(xs), max(ys))
        hands.append({"bbox": bbox, "landmarks": points})

    return hands


def close_hand_detector() -> None:
    """Release MediaPipe resources."""
    _hand_detector.close()
