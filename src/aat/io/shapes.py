"""Generalized annotation shapes for detect + OBB + segmentation polygons.

Unifies the classic 5-field YOLO rows with variable-length polygon rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
# Re-export the core box type for convenience in shape code
from aat.io.labels import DetectionBox  # noqa: F401


@dataclass(frozen=True)
class AnnotationShape:
    class_id: int
    kind: str  # "detect", "obb", or "segment"
    points: list[tuple[float, float]]


# Supported image extensions (single source of truth)
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_annotation_row(
    line: str,
    *,
    image_width: int,
    image_height: int,
    task: str | None = None,
) -> AnnotationShape | None:
    parts = line.split()
    if not parts:
        return None
    row_task = (task or "").strip().lower()
    if len(parts) != 5 and (len(parts) < 7 or len(parts) % 2 == 0):
        raise ValueError("Expected YOLO detect row, OBB row, or segment polygon row")
    class_id = int(parts[0])
    values = [float(part) for part in parts[1:]]
    if len(parts) == 5:
        x_center, y_center, width, height = values
        x1 = (x_center - width / 2.0) * image_width
        y1 = (y_center - height / 2.0) * image_height
        x2 = (x_center + width / 2.0) * image_width
        y2 = (y_center + height / 2.0) * image_height
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return AnnotationShape(class_id=class_id, kind="detect", points=points)
    points = [
        (values[index] * image_width, values[index + 1] * image_height)
        for index in range(0, len(values), 2)
    ]
    kind = "segment" if row_task == "segment" or len(parts) != 9 else "obb"
    return AnnotationShape(class_id=class_id, kind=kind, points=points)


def read_annotation_shapes(
    label_path: str | Path,
    *,
    image_width: int,
    image_height: int,
    task: str | None = None,
    warnings: list[str] | None = None,
) -> list[AnnotationShape]:
    path = Path(label_path)
    if not path.exists():
        return []
    shapes: list[AnnotationShape] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            shape = parse_annotation_row(line, image_width=image_width, image_height=image_height, task=task)
        except ValueError as exc:
            if warnings is not None:
                warnings.append(f"{path}:{line_number}: {exc}")
                continue
            raise
        if shape is not None:
            shapes.append(shape)
    return shapes


def write_annotation_shapes(
    label_path: str | Path,
    shapes: list[AnnotationShape],
    *,
    image_width: int,
    image_height: int,
    task: str | None = None,
) -> None:
    path = Path(label_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    row_task = (task or "").strip().lower()
    for shape in shapes:
        if shape.kind == "detect" and row_task != "segment":
            rows.append(_detect_row_from_points(shape.class_id, shape.points, image_width, image_height))
        elif len(shape.points) >= 3:
            rows.append(_polygon_row_from_points(shape.class_id, shape.points, image_width, image_height))
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


# --- Internal helpers (ported/adapted from original viewer) ---

def _polygon_row_from_points(
    class_id: int,
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> str:
    values: list[str] = [str(int(class_id))]
    for x, y in points:
        values.append(f"{_clamp(float(x) / float(image_width)):.6f}")
        values.append(f"{_clamp(float(y) / float(image_height)):.6f}")
    return " ".join(values)


def _detect_row_from_points(
    class_id: int,
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> str:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(0.0, min(float(image_width), min(xs)))
    y1 = max(0.0, min(float(image_height), min(ys)))
    x2 = max(0.0, min(float(image_width), max(xs)))
    y2 = max(0.0, min(float(image_height), max(ys)))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return (
        f"{int(class_id)} "
        f"{_clamp((x1 + width / 2.0) / float(image_width)):.6f} "
        f"{_clamp((y1 + height / 2.0) / float(image_height)):.6f} "
        f"{_clamp(width / float(image_width)):.6f} "
        f"{_clamp(height / float(image_height)):.6f}"
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def shapes_to_detection_boxes(shapes: list[AnnotationShape]) -> list[DetectionBox]:
    """Convert any shapes that can be treated as axis-aligned boxes into DetectionBox list (best-effort)."""
    boxes: list[DetectionBox] = []
    for s in shapes:
        if not s.points:
            continue
        xs = [p[0] for p in s.points]
        ys = [p[1] for p in s.points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append(
            DetectionBox(
                class_id=s.class_id,
                confidence=1.0,
                x=x1,
                y=y1,
                width=max(0.0, x2 - x1),
                height=max(0.0, y2 - y1),
            )
        )
    return boxes
