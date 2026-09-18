"""Boundaries the use cases depend on. Adapters live in `infrastructure`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from uncertainty_mech.domain.prompts import ChatPrompt
from uncertainty_mech.domain.questions import HealthQuestion


@dataclass(frozen=True)
class Readings:
    """One forward pass per prompt, read at the last prompt token."""

    letter_logprobs: np.ndarray  # [n, k] log-softmax over the offered letters
    letter_mass: np.ndarray  # [n] full-vocabulary probability on those letters
    residuals: np.ndarray | None = None  # [n, n_layers, d_model] decoder-layer outputs


@dataclass(frozen=True)
class Steering:
    """Add `scale * direction` to the output of one decoder layer at every position."""

    direction: np.ndarray
    layer: int
    scale: float


class LanguageModel(Protocol):
    name: str
    num_layers: int

    def read(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        capture_residuals: bool = False,
        steering: Steering | None = None,
    ) -> Readings: ...


class QuestionSource(Protocol):
    def load(self) -> list[HealthQuestion]: ...


class ReferenceCorpus(Protocol):
    def corpus_words(self) -> set[str]: ...


class RunStore(Protocol):
    root: Path

    def write_json(self, name: str, payload: Any) -> None: ...
    def read_json(self, name: str) -> Any: ...
    def write_arrays(self, name: str, **arrays: np.ndarray) -> None: ...
    def read_arrays(self, name: str) -> dict[str, np.ndarray]: ...
    def write_table(self, name: str, rows: Sequence[dict[str, Any]]) -> None: ...
    def write_text(self, name: str, text: str) -> None: ...
    def append_jsonl(self, name: str, payload: Any) -> None: ...


class FigureWriter(Protocol):
    def layer_sweep(self, rows: Sequence[dict[str, Any]], chosen_layer: int, path: Path) -> None: ...

    def risk_coverage(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray]],
        prompt_only_point: tuple[float, float | None],
        target_risk: float,
        path: Path,
    ) -> None: ...
