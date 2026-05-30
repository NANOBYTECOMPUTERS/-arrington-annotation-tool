"""Protocol and data types for detection engines in AAT."""

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
    device: str | None = None


@dataclass
class Prediction:
    image_path: Path
    boxes: list[DetectionBox]
    width: int
    height: int
    error: str | None = None


@runtime_checkable
class DetectionEngine(Protocol):
    @property
    def names(self) -> list[str]: ...

    def predict(self, images, *, conf=None, iou=None, **kwargs) -> list[Prediction]: ...