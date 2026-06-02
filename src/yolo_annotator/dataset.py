from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from yolo_annotator.yolo import DetectionBox, read_yolo_label, write_yolo_label


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_SPLITS = {"train", "val", "test"}
SUPPORTED_SPLIT_ORDER = ("train", "val", "test")


class DatasetKind(str, Enum):
    PLAIN_FOLDER = "plain_folder"
    YOLO_DATASET = "yolo_dataset"


@dataclass(frozen=True)
class ImageItem:
    path: Path
    split: str | None


@dataclass(frozen=True)
class ScannedDataset:
    root: Path
    kind: DatasetKind
    images: list[ImageItem]
    data_yaml: Path | None


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _split_from_yolo_path(root: Path, path: Path) -> str | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "images" and parts[1] in SUPPORTED_SPLITS:
        return parts[1]
    if len(parts) >= 3 and parts[0] in SUPPORTED_SPLITS and parts[1] == "images":
        return parts[0]
    return None


def _with_suffix_txt(path: Path) -> Path:
    return path.with_suffix(".txt")


def label_path_for_image(dataset_root: str | Path, image_path: str | Path, label_root: str | Path | None = None) -> Path:
    root_path = Path(dataset_root).expanduser().resolve()
    resolved_image = Path(image_path).expanduser().resolve()

    if label_root:
        relative_image = _image_relative_path(root_path, resolved_image)
        return Path(label_root).expanduser().resolve() / _with_suffix_txt(relative_image)

    try:
        relative_to_root = resolved_image.relative_to(root_path)
    except ValueError:
        relative_to_root = None

    if relative_to_root and relative_to_root.parts and relative_to_root.parts[0] == "images":
        return root_path / "labels" / _with_suffix_txt(Path(*relative_to_root.parts[1:]))

    if (
        relative_to_root
        and len(relative_to_root.parts) >= 3
        and relative_to_root.parts[0] in SUPPORTED_SPLITS
        and relative_to_root.parts[1] == "images"
    ):
        return (
            root_path
            / relative_to_root.parts[0]
            / "labels"
            / _with_suffix_txt(Path(*relative_to_root.parts[2:]))
        )

    if relative_to_root and root_path.name.lower() == "images":
        return root_path.parent / "labels" / _with_suffix_txt(relative_to_root)

    if relative_to_root:
        return root_path / _with_suffix_txt(relative_to_root)

    return _with_suffix_txt(resolved_image)


def _image_search_root(root_path: Path) -> Path:
    yolo_images_root = root_path / "images"
    return yolo_images_root if yolo_images_root.is_dir() else root_path


def _image_relative_path(root_path: Path, resolved_image: Path) -> Path:
    yolo_images_root = root_path / "images"
    search_root = yolo_images_root if yolo_images_root.is_dir() else root_path
    try:
        relative_image = resolved_image.relative_to(search_root)
    except ValueError as error:
        raise ValueError(f"Image is not inside dataset image root: {resolved_image}") from error
    if search_root == root_path:
        parts = relative_image.parts
        if len(parts) >= 3 and parts[0] in SUPPORTED_SPLITS and parts[1] == "images":
            return Path(parts[0]) / Path(*parts[2:])
    return relative_image


def _is_root_split_dataset(root_path: Path) -> bool:
    return bool(_root_split_image_roots(root_path))


def _root_split_image_roots(root_path: Path) -> list[tuple[str, Path]]:
    return [
        (split, root_path / split / "images")
        for split in SUPPORTED_SPLIT_ORDER
        if (root_path / split / "images").is_dir()
    ]


def _validate_image_in_dataset(dataset_root: str | Path, image_path: str | Path) -> tuple[Path, Path]:
    root_path = Path(dataset_root).expanduser().resolve()
    resolved_image = Path(image_path).expanduser().resolve()
    _image_relative_path(root_path, resolved_image)
    if not _is_image(resolved_image):
        raise FileNotFoundError(f"Dataset image not found: {resolved_image}")
    return root_path, resolved_image


def _box_from_payload(item: dict[str, Any]) -> DetectionBox:
    return DetectionBox(
        class_id=int(item["class_id"]),
        confidence=float(item.get("confidence", 1.0)),
        x=float(item["x"]),
        y=float(item["y"]),
        width=float(item["width"]),
        height=float(item["height"]),
    )


def dataset_image_payload(dataset_root: str | Path, item: ImageItem, label_root: str | Path | None = None) -> dict[str, Any]:
    label_path = label_path_for_image(dataset_root, item.path, label_root=label_root)
    warnings: list[str] = []
    with Image.open(item.path) as image:
        width, height = image.size
    boxes = read_yolo_label(label_path, width, height, warnings=warnings)
    payload: dict[str, Any] = {
        "path": str(item.path),
        "split": item.split,
        "width": width,
        "height": height,
        "label": str(label_path),
        "label_exists": label_path.exists(),
        "boxes": [box.__dict__ for box in boxes],
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def update_dataset_label(
    dataset_root: str | Path,
    image_path: str | Path,
    boxes_payload: list[dict[str, Any]],
    label_root: str | Path | None = None,
) -> dict[str, Any]:
    root_path, resolved_image = _validate_image_in_dataset(dataset_root, image_path)
    label_path = label_path_for_image(root_path, resolved_image, label_root=label_root)
    with Image.open(resolved_image) as image:
        width, height = image.size
    boxes = [_box_from_payload(item) for item in boxes_payload]
    write_yolo_label(label_path, boxes, image_width=width, image_height=height)
    return {
        "ok": True,
        "path": str(resolved_image),
        "label": str(label_path.resolve()),
        "width": width,
        "height": height,
        "boxes": [box.__dict__ for box in boxes],
    }


def scan_dataset(root: str | Path) -> ScannedDataset:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root_path}")

    data_yaml = root_path / "data.yaml"
    yolo_images_root = root_path / "images"
    root_split_image_roots = _root_split_image_roots(root_path)
    is_yolo = yolo_images_root.is_dir() or bool(root_split_image_roots)

    if yolo_images_root.is_dir():
        images = [
            ImageItem(path=p, split=_split_from_yolo_path(root_path, p))
            for p in sorted(yolo_images_root.rglob("*"))
            if _is_image(p)
        ]
    elif root_split_image_roots:
        images = [
            ImageItem(path=p, split=split)
            for split, image_root in root_split_image_roots
            for p in sorted(image_root.rglob("*"))
            if _is_image(p)
        ]
    else:
        images = [
            ImageItem(path=p, split=None)
            for p in sorted(root_path.rglob("*"))
            if _is_image(p)
        ]
    if not images:
        raise ValueError(f"No supported images found under {root_path}")

    return ScannedDataset(
        root=root_path,
        kind=DatasetKind.YOLO_DATASET if is_yolo else DatasetKind.PLAIN_FOLDER,
        images=images,
        data_yaml=data_yaml if data_yaml.exists() else None,
    )
