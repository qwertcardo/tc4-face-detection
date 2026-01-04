"""Helpers to keep MediaPipe video detections strictly monotonic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class _VideoDetector(Protocol):
    """Subset of the MediaPipe detector API used by the wrapper."""

    def detect_for_video(self, mp_roi: Any, timestamp_ms: int) -> Any:  # pragma: no cover - Protocol signature only
        ...


@dataclass
class MonotonicVideoDetector:
    """
    Wraps a MediaPipe video detector ensuring monotonically increasing timestamps.

    MediaPipe expects every call to ``detect_for_video`` to receive a timestamp
    that is strictly greater than the previous one. Instead of relying on
    upstream timestamps, this wrapper uses its own counter that starts at zero
    and increments on every call, guaranteeing strictly increasing values even
    when multiple ROIs share a frame or timestamps repeat.
    """

    detector: _VideoDetector
    start_at: int = 0
    _next_timestamp: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._next_timestamp = self.start_at

    def detect_for_video(self, mp_roi: Any, timestamp_ms: int, face_id: int = 0) -> Any:
        """
        Run detection while keeping the timestamp strictly increasing.

        Args:
            mp_roi: Preprocessed ROI to pass to the MediaPipe detector.
            timestamp_ms: Ignored. Kept for API compatibility.
            face_id: Ignored. Kept for API compatibility.

        Returns:
            The result from ``detector.detect_for_video``.
        """

        forwarded_ts = self._next_timestamp
        self._next_timestamp += 1
        return self.detector.detect_for_video(mp_roi, forwarded_ts)

    def reset(self, start_at: int = 0) -> None:
        """
        Reset the internal counter, e.g., when starting a new video stream.

        Args:
            start_at: Timestamp value to use for the next call.
        """

        self._next_timestamp = start_at
