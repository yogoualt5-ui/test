"""Drawing helper functions for OpenCV overlays."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2

BoundingBox = Tuple[int, int, int, int]


def draw_face_bbox(frame, bbox: BoundingBox, color, thickness: int = 2) -> None:
    """Draw a face bounding box on the frame in-place."""
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)


def draw_hand_bbox(frame, bbox: BoundingBox, color, thickness: int = 2) -> None:
    """Draw a hand bounding box on the frame in-place."""
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)


def draw_hand_landmarks(
    frame,
    landmarks: Iterable[Tuple[int, int, float]],
    color,
    radius: int = 3,
) -> None:
    """Draw hand landmarks as circles in-place."""
    for x, y, _z in landmarks:
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_text(
    frame,
    text: str,
    position: Tuple[int, int],
    font_scale: float,
    color,
    thickness: int = 2,
) -> None:
    """Draw text on the frame in-place."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
