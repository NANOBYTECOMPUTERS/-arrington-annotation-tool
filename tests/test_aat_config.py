from pathlib import Path
from aat.config import AATConfig, get_config

def test_default_config_has_sensible_paths():
    cfg = AATConfig()
    assert cfg.version is not None
    assert cfg.workspace_root.exists()
    assert "0bs" not in str(cfg.workspace_root).lower()
    assert cfg.jobs_dir.name == "jobs"
    assert cfg.default_confidence == 0.25

def test_get_config_returns_consistent():
    c1 = get_config()
    c2 = get_config()
    assert c1.workspace_root == c2.workspace_root