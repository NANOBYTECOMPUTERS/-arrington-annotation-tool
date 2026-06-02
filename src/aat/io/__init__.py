"""AAT I/O layer — labels, shapes, and dataset primitives."""

from aat.io.labels import DetectionBox, read_yolo_label, write_yolo_label, pixel_box_to_yolo
from aat.io.shapes import AnnotationShape, parse_annotation_row, read_annotation_shapes, write_annotation_shapes

__all__ = [
    "DetectionBox",
    "read_yolo_label",
    "write_yolo_label",
    "pixel_box_to_yolo",
    "AnnotationShape",
    "parse_annotation_row",
    "read_annotation_shapes",
    "write_annotation_shapes",
]
