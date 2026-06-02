"""Arrington Annotation Tool (AAT)

Modular library for AI-assisted YOLO annotation.
"Use an engine to make a detect dataset, then review & edit with the viewer."

Core public API:
    from aat import generate_detect_dataset, get_engine, AATViewerApp
"""

__version__ = "0.2.0-dev"

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

