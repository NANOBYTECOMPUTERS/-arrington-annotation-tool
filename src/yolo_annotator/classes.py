from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from yolo_annotator.dataset import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ClassMetadata:
    names: list[str]
    source: str


def _clean_names(names: list[Any]) -> list[str]:
    cleaned = [str(name).strip() for name in names]
    cleaned = [name for name in cleaned if name]
    if not cleaned:
        raise ValueError("Class metadata did not contain any class names")
    return cleaned


def _names_from_mapping(names: dict[Any, Any]) -> list[str]:
    try:
        ids = sorted(int(key) for key in names)
    except (TypeError, ValueError) as exc:
        raise ValueError("Class metadata ids must be integers") from exc
    expected = list(range(len(ids)))
    if ids != expected:
        raise ValueError("Class metadata ids must be contiguous from 0")
    ordered = []
    for class_id in ids:
        value = names[class_id] if class_id in names else names[str(class_id)]
        name = str(value).strip()
        if not name:
            raise ValueError(f"Class metadata id {class_id} has an empty class name")
        ordered.append(name)
    return ordered


def _names_from_indexed_list(names: list[Any]) -> list[str]:
    ordered = []
    for class_id, value in enumerate(names):
        name = str(value).strip()
        if not name:
            raise ValueError(f"Class metadata id {class_id} has an empty class name")
        ordered.append(name)
    if not ordered:
        raise ValueError("Class metadata did not contain any class names")
    return ordered


def _names_from_yaml(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return _names_from_mapping(names)
    if isinstance(names, list):
        return _names_from_indexed_list(names)
    raise ValueError(f"data.yaml does not contain a supported names field: {path}")


def _names_from_classes_txt(path: Path) -> list[str]:
    return _clean_names(path.read_text(encoding="utf-8").splitlines())


def _names_from_manual(text: str) -> list[str]:
    if "\n" in text:
        raw = text.splitlines()
    else:
        raw = text.split(",")
    return _clean_names(raw)


def _names_from_onnx_metadata(path: Path) -> list[str] | None:
    try:
        import onnx
    except ImportError:
        return None

    model = onnx.load(str(path), load_external_data=False)
    props = {prop.key: prop.value for prop in model.metadata_props}
    for key in ("names", "classes", "class_names"):
        value = props.get(key)
        if not value:
            continue
        parsed = yaml.safe_load(value)
        if isinstance(parsed, dict):
            return _names_from_mapping(parsed)
        if isinstance(parsed, list):
            return _names_from_indexed_list(parsed)
    return None


def load_class_metadata(
    *,
    model_path: str | Path | None = None,
    data_yaml: str | Path | None = None,
    classes_txt: str | Path | None = None,
    manual_classes: str | None = None,
) -> ClassMetadata:
    model = Path(model_path) if model_path else None
    if manual_classes and manual_classes.strip():
        return ClassMetadata(names=_names_from_manual(manual_classes), source="manual")
    if data_yaml:
        return ClassMetadata(names=_names_from_yaml(Path(data_yaml)), source="data.yaml")
    if classes_txt:
        return ClassMetadata(names=_names_from_classes_txt(Path(classes_txt)), source="classes.txt")
    if model and model.suffix.lower() == ".onnx":
        names = _names_from_onnx_metadata(model)
        if names:
            return ClassMetadata(names=names, source="onnx")
    if model and model.suffix.lower() == ".engine":
        raise ValueError("Engine models require class metadata from data.yaml, classes.txt, or manual input")
    raise ValueError("Class metadata is required")


def write_data_yaml(path: Path, metadata: ClassMetadata, *, dataset_root: Path) -> None:
    images_root = dataset_root / "images"
    has_direct_images = images_root.is_dir() and any(
        image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
        for image in images_root.iterdir()
    )
    train_path = "images" if has_direct_images else "images/train"
    val_path = "images" if has_direct_images else "images/val"
    data = {
        "path": ".",
        "train": train_path,
        "val": val_path,
        "names": dict(enumerate(metadata.names)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
