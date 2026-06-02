"""Pluggable inference engines for AAT.

Primary goal: turn a pretrained model (engine) into a YOLO detect dataset.
"""

from pathlib import Path

from aat.inference.protocol import DetectionEngine, EngineConfig, Prediction
from aat.inference.ultralytics_engine import UltralyticsDetectionEngine
from aat.inference.worker_engine import WorkerDetectionEngine


def get_engine(model_path: str | Path, backend: str = "ultralytics", **kw: object) -> DetectionEngine:
    """Factory for available AAT inference backends."""
    if backend == "ultralytics":
        return UltralyticsDetectionEngine(model_path, **kw)  # type: ignore[arg-type]
    if backend in {"worker", "cpp", "tensorrt", "trt"}:
        return WorkerDetectionEngine(model_path, **kw)  # type: ignore[arg-type]
    raise ValueError(f"Unknown inference backend: {backend}")


__all__ = [
    "DetectionEngine",
    "EngineConfig",
    "Prediction",
    "UltralyticsDetectionEngine",
    "WorkerDetectionEngine",
    "get_engine",
]
