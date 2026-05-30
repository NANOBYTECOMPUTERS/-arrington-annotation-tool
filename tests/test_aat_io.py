from pathlib import Path
from aat.io.labels import DetectionBox, pixel_box_to_yolo, read_yolo_label, write_yolo_label

def test_pixel_box_to_yolo():
    row = pixel_box_to_yolo(class_id=2, x=-5, y=10, width=30, height=20, image_width=100, image_height=50)
    assert row == "2 0.125000 0.400000 0.250000 0.400000"

def test_read_write_roundtrip(tmp_path):
    label = tmp_path / "test.txt"
    boxes = [
        DetectionBox(0, 0.92, 10, 5, 40, 30),
        DetectionBox(1, 0.7, 120, 40, 60, 50),
    ]
    write_yolo_label(label, boxes, image_width=200, image_height=100, include_confidence=True)
    loaded = read_yolo_label(label, 200, 100)
    assert len(loaded) == 2
    assert abs(loaded[0].confidence - 0.92) < 0.01