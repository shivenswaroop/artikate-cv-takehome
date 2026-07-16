"""Make Ultralytics data.yaml paths absolute relative to the repo root."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from src.paths import REPO_ROOT


def materialize_data_yaml(data_yaml: str | Path) -> Path:
    """Return a temp yaml whose `path` is absolute so Ultralytics finds images."""
    src = Path(data_yaml)
    if not src.is_absolute():
        src = REPO_ROOT / src
    cfg = yaml.safe_load(src.read_text())
    path = Path(cfg["path"])
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    cfg["path"] = str(path)
    tmp = Path(tempfile.mkdtemp(prefix="artikate_data_")) / "data.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return tmp
