"""Headless annotation helpers (auto-trace, shape ops, etc.).

These are the reusable pieces behind the viewer canvas tools.
For full SAM-based refinement, see optional/ or the refine extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

# Current implementation re-uses the mature logic from the original viewer
# (imported here so the old code continues to work while we migrate).
# In later cleanup we can move the bodies fully into aat.
try:
    # Prefer the new io version when it grows
    from aat.io.shapes import AnnotationShape  # noqa: F401
except Exception:
    pass

# Re-export the auto-trace that the viewer depends on (keeps behavior identical for now)
from yolo_annotator.detect_to_segment import mask_to_segment_points  # type: ignore  # noqa: F401
from yolo_annotator.annotation_viewer import (  # type: ignore
    auto_trace_segment_points,
    canvas_rect_to_image_rect,
    remove_shapes_intersecting_rect,
)

# New modular helpers (preferred)
from aat.dataset import (
    guess_label_for_image,
    guess_label_root,
    scan_viewer_images,
)

__all__ = [
    "auto_trace_segment_points",
    "canvas_rect_to_image_rect",
    "remove_shapes_intersecting_rect",
    "mask_to_segment_points",
    "scan_viewer_images",
    "guess_label_root",
    "guess_label_for_image",
]
