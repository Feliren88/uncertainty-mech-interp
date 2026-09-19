"""Boundaries the use cases depend on. Adapters live in `infrastructure`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
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


class SiteKind(StrEnum):
    RESID = "resid"  # output of a decoder layer (the residual stream after it)
    ATTN_Z = "attn_z"  # concatenated attention-head outputs, the input of o_proj
    MLP = "mlp"  # output of a layer's MLP


@dataclass(frozen=True)
class Site:
    """A place inside the model. `head` selects one head's slice of ATTN_Z."""

    kind: SiteKind
    layer: int
    head: int | None = None


# Positions are counted from the end of each prompt: 1 is the last token. Two
# prompts that share a suffix line up position by position that way. A tuple
# gives the same positions for every prompt; an int array [n, k] gives k
# positions per prompt.
Offsets = tuple[int, ...] | np.ndarray


@dataclass(frozen=True, eq=False)
class Capture:
    site: Site
    offsets: Offsets


@dataclass(frozen=True, eq=False)
class Patch:
    """Replace the activation at `site` and `offsets` with `values` [n, k, width]."""

    site: Site
    offsets: Offsets
    values: np.ndarray


@dataclass(frozen=True, eq=False)
class Addition:
    """Add `scale[i] * vector` at `site` for prompt i. `offsets=None` means every position."""

    site: Site
    offsets: Offsets | None
    vector: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class Trace:
    letter_logprobs: np.ndarray  # [n, k]
    letter_mass: np.ndarray  # [n]
    activations: list[np.ndarray]  # one [n, k, width] array per capture, in request order


class CircuitModel(LanguageModel, Protocol):
    """A language model that can be read and edited at named sites (activation patching)."""

    num_heads: int
    head_dim: int

    def trace(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        captures: Sequence[Capture] = (),
        patches: Sequence[Patch] = (),
        additions: Sequence[Addition] = (),
    ) -> Trace: ...

    def common_suffix_length(self, prompts: Sequence[ChatPrompt]) -> int:
        """Number of final tokens shared by every prompt, after the chat template is applied."""
        ...


class QuestionSource(Protocol):
    def load(self) -> list[HealthQuestion]: ...


class ReferenceCorpus(Protocol):
    def corpus_words(self) -> set[str]: ...


class RunStore(Protocol):
    root: Path

    def write_json(self, name: str, payload: Any) -> None: ...
    def read_json(self, name: str) -> Any: ...
    def write_arrays(self, name: str, **arrays: np.ndarray) -> None: ...
    def read_arrays(self, name: str, keys: Sequence[str] | None = None) -> dict[str, np.ndarray]: ...
    def write_table(self, name: str, rows: Sequence[dict[str, Any]]) -> None: ...
    def read_table(self, name: str) -> list[dict[str, str]]: ...
    def write_text(self, name: str, text: str) -> None: ...
    def append_jsonl(self, name: str, payload: Any) -> None: ...


class CircuitFigureWriter(Protocol):
    def heatmap(
        self,
        values: np.ndarray,
        column_labels: Sequence[str] | None,
        xlabel: str,
        ylabel: str,
        value_label: str,
        path: Path,
    ) -> None: ...

    def steering_tradeoff(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray]],
        points: dict[str, tuple[float, float]],
        path: Path,
    ) -> None: ...


class FigureWriter(Protocol):
    def layer_sweep(self, rows: Sequence[dict[str, Any]], chosen_layer: int, path: Path) -> None: ...

    def risk_coverage(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray]],
        prompt_only_point: tuple[float, float | None],
        target_risk: float,
        path: Path,
    ) -> None: ...
