"""High-level API: turn a pretrained model (engine) into a ready-to-use YOLO detect dataset.

This is the flagship "automated annotation" entry point for AAT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2
from typing import Callable, Any

from aat.config import get_config
from aat.dataset import scan_dataset
from aat.inference import get_engine
from aat.inference.protocol import DetectionEngine, Prediction
from aat.io.labels import write_yolo_label
from aat.io.shapes import IMAGE_EXTENSIONS


@dataclass
class GenerateConfig:
    images_root: Path
    model_path: Path
    output_root: Path
    backend: str = "ultralytics"
    worker_command: list[str] | None = None
    class_names: list[str] | None = None
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


def generate_detect_dataset(
    config: GenerateConfig,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    engine: DetectionEngine | None = None,
) -> GenerateResult:
    """Run inference over images and write a standard YOLO detection dataset.

    Produces:
      output_root/
        images/...
        labels/...
        data.yaml
        metadata/ (per-image JSON with confidences for later filtering)
    """
    out = GenerateResult()
    out.output_root = config.output_root

    images_root = Path(config.images_root).expanduser().resolve()
    out_root = Path(config.output_root).expanduser().resolve()
    model_p = Path(config.model_path).expanduser().resolve()

    if out_root.exists() and any(out_root.iterdir()) and not config.force:
        raise RuntimeError(f"Output directory is not empty: {out_root}. Use force=True to overwrite.")

    scanned = scan_dataset(images_root)

    engine_kwargs: dict[str, Any] = {"confidence": config.confidence, "iou": config.iou}
    if config.worker_command is not None:
        engine_kwargs["worker_command"] = config.worker_command
    if config.class_names is not None:
        engine_kwargs["names"] = config.class_names
    engine_kwargs["batch_size"] = config.batch_size
    eng = engine or get_engine(model_p, backend=config.backend, **engine_kwargs)

    try:
        # Write a basic data.yaml
        names = config.class_names or eng.names or [f"class_{i}" for i in range(80)]
        yaml_lines = [
            "path: .",
            "train: images/train" if any(i.split == "train" for i in scanned.images) else "train: images",
            "val: images/val" if any(i.split == "val" for i in scanned.images) else "",
            "names:",
        ]
        for i, n in enumerate(names):
            yaml_lines.append(f"  {i}: {n}")
        (out_root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
        (out_root / "data.yaml").write_text("\n".join(l for l in yaml_lines if l) + "\n", encoding="utf-8")

        images_out = out_root / "images"
        labels_out = out_root / "labels"
        meta_out = out_root / "metadata"

        for start in range(0, len(scanned.images), config.batch_size):
            batch = scanned.images[start : start + config.batch_size]
            try:
                preds = eng.predict(
                    [item.path for item in batch],
                    conf=config.confidence,
                    iou=config.iou,
                    batch_size=config.batch_size,
                )
                if len(preds) != len(batch):
                    raise RuntimeError(f"prediction count mismatch: expected {len(batch)}, got {len(preds)}")
            except Exception as exc:
                for item in batch:
                    out.failed += 1
                    out.errors.append(f"{item.path}: {exc}")
                    if progress:
                        progress({"error": str(exc), "image": str(item.path)})
                continue

            for item, pred in zip(batch, preds, strict=True):
                try:
                    # Determine output subdir (preserve split if present)
                    split = item.split or ""
                    img_target = images_out / split / item.path.name if split else images_out / item.path.name
                    lbl_target = labels_out / split / (item.path.stem + ".txt") if split else labels_out / (item.path.stem + ".txt")

                    if config.copy_images:
                        img_target.parent.mkdir(parents=True, exist_ok=True)
                        copy2(item.path, img_target)

                    lbl_target.parent.mkdir(parents=True, exist_ok=True)
                    write_yolo_label(
                        lbl_target,
                        pred.boxes,
                        image_width=pred.width or 640,
                        image_height=pred.height or 480,
                        include_confidence=False,
                    )

                    # lightweight metadata
                    meta_target = meta_out / split / (item.path.stem + ".json") if split else meta_out / (item.path.stem + ".json")
                    meta_target.parent.mkdir(parents=True, exist_ok=True)
                    import json
                    meta_target.write_text(
                        json.dumps({
                            "model": str(model_p),
                            "confidence_threshold": config.confidence,
                            "detections": [{"class_id": b.class_id, "confidence": b.confidence} for b in pred.boxes],
                        }, indent=2),
                        encoding="utf-8",
                    )

                    out.processed += 1
                    out.total_detections += len(pred.boxes)
                    if progress:
                        progress({"processed": out.processed, "image": str(item.path), "dets": len(pred.boxes)})
                except Exception as exc:
                    out.failed += 1
                    out.errors.append(f"{item.path}: {exc}")
                    if progress:
                        progress({"error": str(exc), "image": str(item.path)})
    finally:
        if engine is None:
            close = getattr(eng, "close", None)
            if close:
                close()
    return out


def main() -> None:  # entry point for aat-generate
    import argparse
    p = argparse.ArgumentParser(description="AAT: Generate YOLO detect dataset from a pretrained model/engine")
    p.add_argument("--model", required=True, help="Path to .pt / .engine / .onnx")
    p.add_argument("--images", required=True, help="Folder of images (or YOLO images/ layout)")
    p.add_argument("--out", required=True, help="Output dataset root")
    p.add_argument("--backend", default="ultralytics", choices=["ultralytics", "worker", "cpp", "tensorrt", "trt"])
    p.add_argument("--worker", help="Path to C++ worker executable for worker/tensorrt backend")
    p.add_argument("--class-name", action="append", default=None, help="Class name. Repeat in class-id order for engine worker metadata.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = GenerateConfig(
        images_root=Path(args.images),
        model_path=Path(args.model),
        output_root=Path(args.out),
        backend=args.backend,
        worker_command=[args.worker] if args.worker else None,
        class_names=args.class_name,
        confidence=args.conf,
        force=args.force,
    )
    res = generate_detect_dataset(cfg)
    print(f"AAT generate complete: {res.processed} images, {res.total_detections} boxes, {res.failed} failed")
    if res.errors:
        print("Errors:", res.errors[:3])
