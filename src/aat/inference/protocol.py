"""Protocol and data types for detection engines.

Any object satisfying DetectionEngine can be used by generate_detect_dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from aat.io.labels import DetectionBox


@dataclass(frozen=True)
class EngineConfig:
    confidence: float = 0.25
    iou: float = 0.45
    imgsz: int | None = None
    device: str | None = None  # "cpu", "0", "cuda:0", etc.


@dataclass
class Prediction:
    """Per-image prediction result (used internally by generators)."""
    image_path: Path
    boxes: list[DetectionBox]
    width: int
    height: int
    error: str | None = None


@runtime_checkable
class DetectionEngine(Protocol):
    """Protocol for anything that can produce detections from images."""

    @property
    def names(self) -> list[str]:
        """Class names in order (index == class_id)."""
        ...

    def predict(
        self,
        images: Path | list[Path],
        *,
        conf: float | None = None,
        iou: float | None = None,
        **kwargs: object,
    ) -> list[Prediction]:
        """Run inference. Must return one Prediction per input image (in order)."""
        ...
