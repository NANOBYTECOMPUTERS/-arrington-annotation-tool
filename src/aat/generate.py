"""High-level API: turn a pretrained model into a ready-to-use YOLO detect dataset.

This is the core "use an engine to make a detect dataset" capability of AAT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2
from typing import Any, Callable

from aat.dataset import scan_dataset
from aat.inference import get_engine
from aat.inference.protocol import DetectionEngine, Prediction
from aat.io.labels import write_yolo_label


@dataclass
class GenerateConfig:
    images_root: Path
    model_path: Path
    output_root: Path
    confidence: float = 0.25
    iou: float = 0.45
    copy_images: bool = True
    batch_size: int = 8
    force: bool = False


@dataclass
class GenerateResult:
    processed: int = 0
    failed: int = 0
    total_detections: int = 0
    output_root: Path | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def generate_detect_dataset(config: GenerateConfig, *, progress=None, engine=None) -> GenerateResult:
    out = GenerateResult()
    out.output_root = Path(config.output_root).expanduser().resolve()

    images_root = Path(config.images_root).expanduser().resolve()
    model_p = Path(config.model_path).expanduser().resolve()

    if out.output_root.exists() and any(out.output_root.iterdir()) and not config.force:
        raise RuntimeError(f"Output directory is not empty: {out.output_root}. Use force=True.")

    scanned = scan_dataset(images_root)
    eng: DetectionEngine = engine or get_engine(model_p)

    names = eng.names or [f"class_{i}" for i in range(80)]
    yaml_lines = ["path: .", "train: images", "names:"]
    for i, n in enumerate(names):
        yaml_lines.append(f"  {i}: {n}")
    (out.output_root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (out.output_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    images_out = out.output_root / "images"
    labels_out = out.output_root / "labels"
    meta_out = out.output_root / "metadata"

    for item in scanned.images:
        try:
            preds = eng.predict([item.path], conf=config.confidence, iou=config.iou)
            pred: Prediction = preds[0] if preds else Prediction(item.path, [], 640, 480)

            split = item.split or ""
            img_target = images_out / split / item.path.name if split else images_out / item.path.name
            lbl_target = labels_out / split / f"{item.path.stem}.txt" if split else labels_out / f"{item.path.stem}.txt"

            if config.copy_images:
                img_target.parent.mkdir(parents=True, exist_ok=True)
                copy2(item.path, img_target)

            lbl_target.parent.mkdir(parents=True, exist_ok=True)
            write_yolo_label(lbl_target, pred.boxes, image_width=pred.width or 640,
                             image_height=pred.height or 480, include_confidence=True)

            meta_target = meta_out / split / f"{item.path.stem}.json" if split else meta_out / f"{item.path.stem}.json"
            meta_target.parent.mkdir(parents=True, exist_ok=True)
            import json
            meta_target.write_text(json.dumps({
                "model": str(model_p),
                "confidence_threshold": config.confidence,
                "detections": [{"class_id": b.class_id, "confidence": b.confidence} for b in pred.boxes],
            }, indent=2), encoding="utf-8")

            out.processed += 1
            out.total_detections += len(pred.boxes)
            if progress:
                progress({"processed": out.processed, "image": str(item.path), "dets": len(pred.boxes)})
        except Exception as exc:
            out.failed += 1
            out.errors.append(f"{item.path}: {exc}")

    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description="AAT: Generate YOLO detect dataset from a pretrained model")
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = GenerateConfig(
        images_root=Path(args.images),
        model_path=Path(args.model),
        output_root=Path(args.out),
        confidence=args.conf,
        force=args.force,
    )
    res = generate_detect_dataset(cfg)
    print(f"AAT generate complete: {res.processed} images, {res.total_detections} boxes, {res.failed} failed")
    if res.errors:
        print("Errors:", res.errors[:3])