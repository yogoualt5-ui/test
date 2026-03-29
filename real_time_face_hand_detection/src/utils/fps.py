"""FPS counter utility."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """Computes FPS using frame-to-frame timing with optional smoothing."""

    def __init__(self, window_size: int = 10) -> None:
        self.prev_time = time.time()
        self.samples: Deque[float] = deque(maxlen=max(1, window_size))

    def update(self) -> float:
        """Return smoothed frames-per-second value."""
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time

        if delta <= 0:
            return 0.0

        fps = 1.0 / delta
        self.samples.append(fps)
        return sum(self.samples) / len(self.samples)
