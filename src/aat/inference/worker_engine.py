"""C++ worker-backed detection engine for AAT.

This adapts the original YOLO Auto Annotator NDJSON worker protocol to the
modular AAT DetectionEngine interface. It is intended for TensorRT `.engine`
models through the existing CUDA C++ worker.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aat.inference.protocol import EngineConfig, Prediction
from aat.io.labels import DetectionBox
from yolo_annotator.worker_protocol import WorkerClient


class WorkerDetectionEngine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        worker_command: Sequence[str] | None = None,
        backend: str = "TRT",
        names: Sequence[str] | None = None,
        class_count: int | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        batch_size: int = 16,
        request_timeout: float = 60.0,
        shutdown_timeout: float = 5.0,
        config: EngineConfig | None = None,
        client_factory: Callable[..., Any] = WorkerClient,
    ) -> None:
        if not worker_command:
            raise ValueError("worker_command is required for the AAT worker backend")
        self.model_path = Path(model_path).expanduser().resolve()
        self.worker_command = [str(item) for item in worker_command]
        self.backend = backend
        self._names = list(names or [])
        self.class_count = int(class_count or len(self._names) or 80)
        self.config = config or EngineConfig(
            confidence=confidence if confidence is not None else 0.25,
            iou=iou if iou is not None else 0.45,
        )
        self.batch_size = max(1, int(batch_size))
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self.client_factory = client_factory
        self._worker: Any | None = None
        self._loaded_key: tuple[str, str, int, float, float] | None = None

    @property
    def names(self) -> list[str]:
        if self._names:
            return self._names
        return [f"class_{index}" for index in range(self.class_count)]

    def predict(
        self,
        images: Path | list[Path],
        *,
        conf: float | None = None,
        iou: float | None = None,
        **kwargs: object,
    ) -> list[Prediction]:
        paths = [images] if isinstance(images, Path) else list(images)
        if not paths:
            return []
        confidence = float(conf if conf is not None else self.config.confidence)
        nms = float(iou if iou is not None else self.config.iou)
        batch_size = max(1, int(kwargs.get("batch_size", self.batch_size) or self.batch_size))
        worker = self._ensure_worker()
        self._load_model_if_needed(worker, confidence=confidence, iou=nms)

        predictions: list[Prediction] = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            response = worker.request(
                {
                    "cmd": "annotate_batch",
                    "images": [str(path) for path in batch],
                    "class_count": self.class_count,
                    "confidence_threshold": confidence,
                    "nms_threshold": nms,
                },
                timeout=self.request_timeout,
            )
            results = response.get("results", [])
            if not isinstance(results, list) or len(results) != len(batch):
                raise RuntimeError(f"Worker result count mismatch: expected {len(batch)}, got {len(results) if isinstance(results, list) else 'non-list'}")
            for path, result in zip(batch, results, strict=True):
                predictions.append(self._prediction_from_worker_result(path, result))
        return predictions

    def close(self) -> None:
        worker = self._worker
        self._worker = None
        self._loaded_key = None
        if worker is not None:
            worker.__exit__(None, None, None)

    def __enter__(self) -> "WorkerDetectionEngine":
        self._ensure_worker()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_worker(self) -> Any:
        if self._worker is not None:
            return self._worker
        self._worker = self.client_factory(
            self.worker_command,
            request_timeout=self.request_timeout,
            shutdown_timeout=self.shutdown_timeout,
        ).__enter__()
        self._worker.request({"cmd": "hello"}, timeout=self.request_timeout)
        return self._worker

    def _load_model_if_needed(self, worker: Any, *, confidence: float, iou: float) -> None:
        key = (str(self.model_path), self.backend, self.class_count, confidence, iou)
        if key == self._loaded_key:
            return
        worker.request(
            {
                "cmd": "load_model",
                "model_path": str(self.model_path),
                "backend": self.backend,
                "class_count": self.class_count,
                "confidence_threshold": confidence,
                "nms_threshold": iou,
            },
            timeout=self.request_timeout,
        )
        self._loaded_key = key

    def _prediction_from_worker_result(self, image_path: Path, result: Any) -> Prediction:
        if not isinstance(result, dict):
            raise RuntimeError(f"Worker returned malformed result for {image_path}: expected object")
        if result.get("error"):
            return Prediction(image_path=image_path, boxes=[], width=0, height=0, error=str(result["error"]))
        width = int(result["width"])
        height = int(result["height"])
        detections = result.get("detections", [])
        if not isinstance(detections, list):
            raise RuntimeError(f"Worker returned malformed detections for {image_path}: expected list")
        boxes = [
            DetectionBox(
                class_id=int(det["class_id"]),
                confidence=float(det["confidence"]),
                x=float(det["x"]),
                y=float(det["y"]),
                width=float(det["width"]),
                height=float(det["height"]),
            )
            for det in detections
        ]
        return Prediction(image_path=image_path, boxes=boxes, width=width, height=height)
