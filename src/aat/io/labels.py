"""Core YOLO detection label I/O for AAT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
            class_id=self.class_id, confidence=confidence,
            x=self.x, y=self.y, width=self.width, height=self.height
        )


def _clamped_box_size(*, x, y, width, height, image_width, image_height):
    x1 = max(0.0, min(float(image_width), float(x)))
    y1 = max(0.0, min(float(image_height), float(y)))
    x2 = max(0.0, min(float(image_width), float(x) + float(width)))
    y2 = max(0.0, min(float(image_height), float(y) + float(height)))
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def pixel_box_to_yolo(*, class_id, x, y, width, height, image_width, image_height):
    x1 = max(0.0, min(float(image_width), float(x)))
    y1 = max(0.0, min(float(image_height), float(y)))
    box_width, box_height = _clamped_box_size(
        x=x, y=y, width=width, height=height,
        image_width=image_width, image_height=image_height
    )
    x_center = (x1 + box_width / 2.0) / float(image_width)
    y_center = (y1 + box_height / 2.0) / float(image_height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width / image_width:.6f} {box_height / image_height:.6f}"


def read_yolo_label(label_path: Path, image_width: int, image_height: int, *, warnings=None):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if not label_path.exists():
        return []

    boxes = []
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
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height
            conf = float(parts[5]) if len(parts) == 6 else 1.0
        except ValueError as error:
            msg = f"Invalid YOLO label row {line_number} in {label_path}: {line}"
            if warnings is not None:
                warnings.append(msg)
                continue
            raise ValueError(msg) from error
        boxes.append(DetectionBox(class_id, conf, x_center - width/2, y_center - height/2, width, height))
    return boxes


def write_yolo_label(label_path: Path, boxes, *, image_width, image_height, include_confidence=False):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(boxes, key=lambda b: (b.class_id, -b.confidence))
    rows = []
    for box in ordered:
        cw, ch = _clamped_box_size(
            x=box.x, y=box.y, width=box.width, height=box.height,
            image_width=image_width, image_height=image_height
        )
        if cw <= 0 or ch <= 0:
            continue
        row = pixel_box_to_yolo(
            class_id=box.class_id, x=box.x, y=box.y,
            width=box.width, height=box.height,
            image_width=image_width, image_height=image_height
        )
        if include_confidence and box.confidence < 1.0:
            row += f" {box.confidence:.6f}"
        rows.append(row)
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")