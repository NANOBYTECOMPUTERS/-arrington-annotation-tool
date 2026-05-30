from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def find_workspace_root() -> Path:
    """Find the workspace root for AAT (looks for pyproject.toml or .git)."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return here.parents[2]


@dataclass(frozen=True)
class AATConfig:
    """Portable configuration for Arrington Annotation Tool.

    No machine-specific hard-coded paths.
    """

    workspace_root: Path = field(default_factory=find_workspace_root)
    default_confidence: float = 0.25
    default_iou: float = 0.45
    default_batch_size: int = 8

    jobs_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "jobs_dir", self.workspace_root / ".aat" / "jobs")
        object.__setattr__(self, "cache_dir", self.workspace_root / ".aat" / "cache")

    @property
    def version(self) -> str:
        try:
            from aat import __version__
            return __version__
        except Exception:
            return "0.2.0"


_CONFIG: AATConfig | None = None


def get_config(**overrides: Any) -> AATConfig:
    global _CONFIG
    if _CONFIG is None or overrides:
        base = _CONFIG or AATConfig()
        if overrides:
            data = {**base.__dict__, **overrides}
            return AATConfig(
                workspace_root=data.get("workspace_root", base.workspace_root),
                default_confidence=data.get("default_confidence", base.default_confidence),
                default_iou=data.get("default_iou", base.default_iou),
                default_batch_size=data.get("default_batch_size", base.default_batch_size),
            )
        _CONFIG = base
    return _CONFIG