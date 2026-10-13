"""checkpoint.py — tracks completed work so steps can safely resume."""

import json
from pathlib import Path
from typing import Any


class Checkpoint:
    def __init__(self, step_name: str, checkpoint_dir: Path):
        self.path = Path(checkpoint_dir) / f"{step_name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"done": [], "meta": {}}
        )

    def _save(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def is_done(self, key: Any) -> bool:
        return str(key) in self._data["done"]

    def mark_done(self, key: Any):
        k = str(key)
        if k not in self._data["done"]:
            self._data["done"].append(k)
            self._save()

    def set_meta(self, key: str, value: Any):
        self._data["meta"][key] = value
        self._save()

    def get_meta(self, key: str, default: Any = None) -> Any:
        return self._data["meta"].get(key, default)

    def reset(self):
        self._data = {"done": [], "meta": {}}
        self._save()

    def count_done(self) -> int:
        return len(self._data["done"])
