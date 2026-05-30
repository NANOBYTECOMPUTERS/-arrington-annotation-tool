"""Arrington Annotation Tool (AAT)

A modular, library-first toolkit for AI-assisted YOLO annotation.

Key capabilities:
- Use any pretrained model (.pt / .engine / .onnx) to generate YOLO detection datasets
- High-quality manual review & editing with the AAT Viewer
- Built-in "Suggest" integration inside the viewer using the same engines
- Clean, testable, reusable Python package

Example:
    from aat import generate_detect_dataset, get_engine, AATViewerApp
"""

__version__ = "0.2.0"

from aat.config import AATConfig, get_config
from aat.generate import GenerateConfig, GenerateResult, generate_detect_dataset
from aat.inference import DetectionEngine, EngineConfig, get_engine
from aat.viewer import AATViewerApp

__all__ = [
    "__version__",
    "AATConfig",
    "get_config",
    "GenerateConfig",
    "GenerateResult",
    "generate_detect_dataset",
    "DetectionEngine",
    "EngineConfig",
    "get_engine",
    "AATViewerApp",
]