"""Ultralytics-backed DetectionEngine (the recommended default for AAT).

Supports .pt natively. .engine and .onnx work when the user has the
corresponding Ultralytics + runtime environment (TensorRT, ONNX Runtime, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aat.inference.protocol import DetectionEngine, EngineConfig, Prediction
from aat.io.labels import DetectionBox


class UltralyticsDetectionEngine:
    """Wraps `from ultralytics import YOLO` for AAT's protocol."""

    def __init__(self, model_path: str | Path, *, config: EngineConfig | None = None):
        self.model_path = Path(model_path).expanduser().resolve()
        self.config = config or EngineConfig()
        self._model: Any | None = None
        self._names: list[str] = []

    @property
    def names(self) -> list[str]:
        if not self._names:
            model = self._ensure_model()
            # Ultralytics stores names as dict or list
            raw = getattr(model, "names", {}) or {}
            if isinstance(raw, dict):
                self._names = [raw[i] for i in sorted(raw)]
            else:
                self._names = list(raw)
        return self._names

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics is required for UltralyticsDetectionEngine. "
                "Install with: pip install 'arrington-annotation-tool[ultralytics]'"
            ) from e
        self._model = YOLO(str(self.model_path))
        return self._model

    def predict(
        self,
        images: Path | list[Path],
        *,
        conf: float | None = None,
        iou: float | None = None,
        **kwargs: object,
    ) -> list[Prediction]:
        model = self._ensure_model()
        paths = [images] if isinstance(images, Path) else list(images)
        if not paths:
            return []

        c = conf if conf is not None else self.config.confidence
        i = iou if iou is not None else self.config.iou

        results = model.predict(
            source=[str(p) for p in paths],
            conf=c,
            iou=i,
            imgsz=kwargs.get("imgsz", self.config.imgsz),
            device=kwargs.get("device", self.config.device),
            verbose=False,
        )

        out: list[Prediction] = []
        for res, p in zip(results, paths):
            w, h = getattr(res, "orig_shape", (0, 0))[1], getattr(res, "orig_shape", (0, 0))[0]
            if hasattr(res, "orig_img") and res.orig_img is not None:
                h, w = res.orig_img.shape[:2]  # type: ignore

            boxes: list[DetectionBox] = []
            if hasattr(res, "boxes") and res.boxes is not None:
                for b in res.boxes:
                    # xyxy, conf, cls
                    xyxy = b.xyxy[0].tolist() if hasattr(b.xyxy, "tolist") else list(b.xyxy[0])
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    cls = int(b.cls[0]) if hasattr(b, "cls") else 0
                    cf = float(b.conf[0]) if hasattr(b, "conf") else 1.0
                    boxes.append(
                        DetectionBox(
                            class_id=cls,
                            confidence=cf,
                            x=x1,
                            y=y1,
                            width=max(0.0, x2 - x1),
                            height=max(0.0, y2 - y1),
                        )
                    )
            out.append(Prediction(image_path=p, boxes=boxes, width=int(w), height=int(h)))
        return out
