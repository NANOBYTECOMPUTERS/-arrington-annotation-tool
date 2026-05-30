from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
from aat.generate import GenerateConfig, generate_detect_dataset
from aat.inference.protocol import Prediction
from aat.io.labels import DetectionBox

def test_generate_writes_yolo_dataset(tmp_path):
    imgs = tmp_path / "raw"
    imgs.mkdir()
    Image.new("RGB", (64, 48)).save(imgs / "a.jpg")
    Image.new("RGB", (64, 48)).save(imgs / "b.jpg")

    out = tmp_path / "out"
    model = tmp_path / "fake.pt"
    model.write_text("stub")

    fake_eng = MagicMock()
    fake_eng.names = ["thing"]
    fake_eng.predict.return_value = [
        Prediction(imgs / "a.jpg", [DetectionBox(0, 0.9, 5, 5, 20, 15)], 64, 48),
        Prediction(imgs / "b.jpg", [], 64, 48),
    ]

    cfg = GenerateConfig(images_root=imgs, model_path=model, output_root=out, confidence=0.1, force=True)
    res = generate_detect_dataset(cfg, engine=fake_eng)

    assert res.processed == 2
    assert (out / "data.yaml").exists()
    assert (out / "labels" / "a.txt").exists()
    assert "thing" in (out / "data.yaml").read_text()