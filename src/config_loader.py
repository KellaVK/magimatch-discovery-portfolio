"""config_loader.py — loads config/config.yaml, exposes cfg and paths."""

import yaml
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).parent.parent


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _ns(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _ns(v) if isinstance(v, dict) else v)
    return ns


def _resolve_paths(d, root):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, root / v)
    return ns


_raw = _load(_ROOT / "config" / "config.yaml")
cfg = _ns(_raw)
paths = _resolve_paths(_raw["paths"], _ROOT)


def get_project_root():
    return _ROOT
