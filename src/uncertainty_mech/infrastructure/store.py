"""A run directory on disk: JSON, NPZ, CSV, markdown and an append-only audit log."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class FileRunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> None:
        self._path(name).write_text(json.dumps(payload, indent=2, default=_encode) + "\n", encoding="utf-8")

    def read_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def write_arrays(self, name: str, **arrays: np.ndarray) -> None:
        np.savez(self._path(name), **arrays)

    def read_arrays(self, name: str, keys: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        with np.load(self.root / name) as data:
            return {key: data[key] for key in (keys or data.files)}

    def write_table(self, name: str, rows: Sequence[dict[str, Any]]) -> None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with self._path(name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def read_table(self, name: str) -> list[dict[str, str]]:
        with (self.root / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def write_text(self, name: str, text: str) -> None:
        self._path(name).write_text(text, encoding="utf-8")

    def append_jsonl(self, name: str, payload: Any) -> None:
        with self._path(name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_encode) + "\n")

    def _path(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")
