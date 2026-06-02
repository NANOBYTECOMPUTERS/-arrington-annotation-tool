from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def find_workspace_root() -> Path:
    """Find the project/workspace root for AAT.

    Starts from this file and walks up until it finds a marker
    (pyproject.toml or .git). Falls back to three parents.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return here.parents[2]  # fallback for src/aat/config.py


@dataclass(frozen=True)
class AATConfig:
    """Central, portable configuration for Arrington Annotation Tool.

    All paths are explicit or derived from workspace_root / CWD.
    No machine-specific hardcodes allowed in this module.
    """

    workspace_root: Path = field(default_factory=find_workspace_root)
    default_confidence: float = 0.25
    default_iou: float = 0.45
    default_batch_size: int = 8
    # Storage for jobs, drafts, etc. (local to workspace by default)
    jobs_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        # Use object.__setattr__ because frozen dataclass
        object.__setattr__(self, "jobs_dir", self.workspace_root / ".aat" / "jobs")
        object.__setattr__(self, "cache_dir", self.workspace_root / ".aat" / "cache")

    @property
    def version(self) -> str:
        try:
            from aat import __version__  # type: ignore

            return __version__
        except Exception:
            return "0.2.0-dev"


# Simple module-level cache for get_config (can be expanded later)
_CONFIG: AATConfig | None = None


def get_config(**overrides: Any) -> AATConfig:
    """Return the process-wide AATConfig (or a fresh one with overrides).

    Later versions can load ~/.aat/config.yaml or .aat.yaml from CWD.
    """
    global _CONFIG
    if _CONFIG is None or overrides:
        base = _CONFIG or AATConfig()
        if overrides:
            data = {**base.__dict__, **overrides}
            # Rebuild cleanly
            return AATConfig(
                workspace_root=data.get("workspace_root", base.workspace_root),
                default_confidence=data.get("default_confidence", base.default_confidence),
                default_iou=data.get("default_iou", base.default_iou),
                default_batch_size=data.get("default_batch_size", base.default_batch_size),
            )
        _CONFIG = base
    return _CONFIG
