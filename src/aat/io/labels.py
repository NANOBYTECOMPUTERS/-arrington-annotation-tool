"""Core YOLO detection label I/O.

DetectionBox + read/write with confidence metadata support.
This is the single source of truth for axis-aligned box labels in AAT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectionBox:
    class_id: int
    confidence: float = 1.0
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def with_confidence(self, confidence: float) -> "DetectionBox":
        return DetectionBox(
            class_id=self.class_id,
            confidence=confidence,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
        )


def _clamped_box_size(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    x1 = max(0.0, min(float(image_width), float(x)))
    y1 = max(0.0, min(float(image_height), float(y)))
    x2 = max(0.0, min(float(image_width), float(x) + float(width)))
    y2 = max(0.0, min(float(image_height), float(y) + float(height)))
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def pixel_box_to_yolo(
    *,
    class_id: int,
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> str:
    """Convert pixel-space box to single normalized YOLO row (no trailing newline)."""
    x1 = max(0.0, min(float(image_width), float(x)))
    y1 = max(0.0, min(float(image_height), float(y)))
    box_width, box_height = _clamped_box_size(
        x=x, y=y, width=width, height=height,
        image_width=image_width, image_height=image_height,
    )
    x_center = (x1 + box_width / 2.0) / float(image_width)
    y_center = (y1 + box_height / 2.0) / float(image_height)
    norm_width = box_width / float(image_width)
    norm_height = box_height / float(image_height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def read_yolo_label(
    label_path: Path,
    image_width: int,
    image_height: int,
    *,
    warnings: list[str] | None = None,
) -> list[DetectionBox]:
    """Read classic 5-field YOLO detect labels into DetectionBox list.

    Supports optional per-box confidence in extended 6-field rows (class cx cy w h conf).
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if not label_path.exists():
        return []

    boxes: list[DetectionBox] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in (5, 6):
            msg = f"Invalid YOLO label row {line_number} in {label_path}: {line}"
            if warnings is not None:
                warnings.append(msg)
                continue
            raise ValueError(msg)
        try:
            class_id = int(parts[0])
            x_center = float(parts[1]) * float(image_width)
            y_center = float(parts[2]) * float(image_height)
            width = float(parts[3]) * float(image_width)
            height = float(parts[4]) * float(image_height)
            conf = float(parts[5]) if len(parts) == 6 else 1.0
        except ValueError as error:
            msg = f"Invalid YOLO label row {line_number} in {label_path}: {line}"
            if warnings is not None:
                warnings.append(msg)
                continue
            raise ValueError(msg) from error

        x = x_center - width / 2.0
        y = y_center - height / 2.0
        boxes.append(DetectionBox(class_id=class_id, confidence=conf, x=x, y=y, width=width, height=height))
    return boxes


def write_yolo_label(
    label_path: Path,
    boxes: list[DetectionBox],
    *,
    image_width: int,
    image_height: int,
    include_confidence: bool = False,
) -> None:
    """Write DetectionBox list as YOLO .txt.

    Sorted by class_id then descending confidence.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(boxes, key=lambda b: (b.class_id, -b.confidence))
    rows: list[str] = []
    for box in ordered:
        cw, ch = _clamped_box_size(
            x=box.x, y=box.y, width=box.width, height=box.height,
            image_width=image_width, image_height=image_height,
        )
        if cw <= 0 or ch <= 0:
            continue
        row = pixel_box_to_yolo(
            class_id=box.class_id,
            x=box.x, y=box.y, width=box.width, height=box.height,
            image_width=image_width, image_height=image_height,
        )
        if include_confidence and box.confidence < 1.0:
            row = f"{row} {box.confidence:.6f}"
        rows.append(row)
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
