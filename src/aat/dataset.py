"""Dataset scanning and YOLO layout helpers for AAT (single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from aat.io.shapes import IMAGE_EXTENSIONS

# Re-export for convenience
__all__ = [
    "DatasetKind",
    "ImageItem",
    "ScannedDataset",
    "scan_dataset",
    "resolve_label_path",
    "scan_viewer_images",
    "guess_label_root",
    "guess_label_for_image",
]


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
    if len(parts) >= 3 and parts[0].lower() == "images":
        return parts[1]
    return None


def scan_dataset(root: str | Path) -> ScannedDataset:
    """Scan a folder for images. Supports flat or standard YOLO images/train + labels/train layout."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root_path}")

    data_yaml = root_path / "data.yaml"
    yolo_images_root = root_path / "images"
    is_yolo = yolo_images_root.is_dir()
    search_root = yolo_images_root if is_yolo else root_path

    images = [
        ImageItem(path=p, split=_split_from_yolo_path(root_path, p) if is_yolo else None)
        for p in sorted(search_root.rglob("*"))
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


def resolve_label_path(
    image_path: Path,
    *,
    images_root: Path | None = None,
    labels_root: Path | None = None,
) -> Path:
    """Best-effort resolution of the corresponding YOLO .txt label path for an image.

    Handles common layouts (flat, images/..., images/train/...).
    """
    img = Path(image_path).expanduser().resolve()
    if labels_root is not None:
        labels_root = Path(labels_root).expanduser().resolve()
        # Try direct relative
        if images_root is not None:
            try:
                rel = img.relative_to(Path(images_root).expanduser().resolve())
                return labels_root / rel.with_suffix(".txt")
            except ValueError:
                pass
        # Fallback: same name next to image or in labels/
        cand = labels_root / img.name
        if cand.with_suffix(".txt").exists():
            return cand.with_suffix(".txt")
        return labels_root / img.with_suffix(".txt").name

    # No explicit labels_root — guess beside image or images <-> labels swap
    direct = img.with_suffix(".txt")
    if direct.exists():
        return direct
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            cand = Path(*parts).with_suffix(".txt")
            return cand
    return direct


# --- Viewer / annotation tool helpers (extracted for modularity) ---

def scan_viewer_images(image_root: str | Path) -> list[Path]:
    """Scan a folder recursively for supported image files."""
    root = Path(image_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image folder not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def guess_label_root(image_root: Path) -> Path | None:
    """Best-effort guess for the sibling 'labels' folder given an images folder."""
    root = image_root.expanduser().resolve()
    if root.name.lower() == "images" and (root.parent / "labels").is_dir():
        return root.parent / "labels"
    parts = list(root.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            candidate = Path(*parts)
            return candidate if candidate.is_dir() else None
    return None


def guess_label_for_image(image_path: Path) -> Path | None:
    """Best-effort guess for the corresponding .txt label file for a given image."""
    image = image_path.expanduser().resolve()
    direct = image.with_suffix(".txt")
    if direct.exists():
        return direct
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            return candidate if candidate.exists() else None
    return None
