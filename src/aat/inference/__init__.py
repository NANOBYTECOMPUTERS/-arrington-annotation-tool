"""Pluggable inference engines for AAT.

The heart of "use an engine to make a detect dataset".
"""

from aat.inference.protocol import DetectionEngine, EngineConfig, Prediction
from aat.inference.ultralytics_engine import UltralyticsDetectionEngine


def get_engine(model_path: str | Path, backend: str = "ultralytics", **kw):
    if backend == "ultralytics":
        return UltralyticsDetectionEngine(model_path, **kw)
    raise ValueError(f"Unknown inference backend: {backend}")


__all__ = [
    "DetectionEngine",
    "EngineConfig",
    "Prediction",
    "UltralyticsDetectionEngine",
    "get_engine",
]