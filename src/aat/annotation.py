"""Headless annotation helpers for AAT (modular core)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aat.dataset import guess_label_for_image, guess_label_root, scan_viewer_images
from aat.io.shapes import AnnotationShape, parse_annotation_row, read_annotation_shapes, write_annotation_shapes

# Optional heavy bridges (kept for advanced features; can be made fully optional later)
try:
    from yolo_annotator.detect_to_segment import mask_to_segment_points  # type: ignore
except Exception:
    mask_to_segment_points = None  # type: ignore

try:
    from yolo_annotator.annotation_viewer import auto_trace_segment_points  # type: ignore
except Exception:
    auto_trace_segment_points = None  # type: ignore


__all__ = [
    "AnnotationShape",
    "parse_annotation_row",
    "read_annotation_shapes",
    "write_annotation_shapes",
    "scan_viewer_images",
    "guess_label_root",
    "guess_label_for_image",
    "canvas_rect_to_image_rect",
    "remove_shapes_intersecting_rect",
    "auto_trace_segment_points",
    "mask_to_segment_points",
]


# ==================== Pure modular helpers (extracted for independence) ====================

def canvas_rect_to_image_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    transform: dict[str, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert canvas selection rectangle to image pixel coordinates."""
    scale = float(transform["scale"])
    if scale <= 0:
        raise ValueError("Canvas transform scale must be positive")
    canvas_x1, canvas_x2 = sorted((float(start[0]), float(end[0])))
    canvas_y1, canvas_y2 = sorted((float(start[1]), float(end[1])))
    x1 = int(max(0, min(image_width, (canvas_x1 - float(transform["x"])) / scale)))
    y1 = int(max(0, min(image_height, (canvas_y1 - float(transform["y"])) / scale)))
    x2 = int(max(0, min(image_width, np.ceil((canvas_x2 - float(transform["x"])) / scale))))
    y2 = int(max(0, min(image_height, np.ceil((canvas_y2 - float(transform["y"])) / scale))))
    return x1, y1, x2, y2


def remove_shapes_intersecting_rect(
    shapes: list[AnnotationShape],
    rect: tuple[int, int, int, int],
) -> tuple[list[AnnotationShape], int]:
    """Remove shapes whose center lies inside the given rect. Returns (kept_shapes, removed_count)."""
    kept: list[AnnotationShape] = []
    removed = 0
    for shape in shapes:
        if _shape_center_inside_rect(shape, rect):
            removed += 1
        else:
            kept.append(shape)
    return kept, removed


def _shape_center_inside_rect(shape: AnnotationShape, rect: tuple[int, int, int, int]) -> bool:
    if not shape.points:
        return False
    x1, y1, x2, y2 = rect
    bounds = _shape_bounds(shape)
    center_x = (bounds[0] + bounds[2]) / 2.0
    center_y = (bounds[1] + bounds[3]) / 2.0
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def _shape_bounds(shape: AnnotationShape) -> tuple[int, int, int, int]:
    if not shape.points:
        return 0, 0, 0, 0
    xs = [point[0] for point in shape.points]
    ys = [point[1] for point in shape.points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


# Simple auto-trace fallback if the heavy one is unavailable
def auto_trace_segment_points(
    image: Image.Image,
    rect: tuple[int, int, int, int],
    *,
    epsilon_ratio: float = 0.01,
    max_points: int = 128,
    threshold: float = 18.0,
) -> list[tuple[float, float]] | None:
    """Lightweight auto-trace. Falls back to None if heavy deps unavailable."""
    if auto_trace_segment_points is not None:  # type: ignore
        try:
            return auto_trace_segment_points(image, rect, epsilon_ratio=epsilon_ratio, max_points=max_points, threshold=threshold)  # type: ignore
        except Exception:
            pass
    return None