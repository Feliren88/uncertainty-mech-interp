"""Stages 1 and 2: localize the abstention signal with activation patching, then pick and test a circuit.

Every run uses the prompt that offers option E. A patch copies activations from
the invented-entity prompt into the matched real-entity prompt ("sufficiency":
does this piece alone make a known question look unknown?) or the reverse
("necessity"). Behavior is read as the abstention gap and as semantic entropy
over A to D, both from the same pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from uncertainty_mech.application.circuit_config import CircuitConfig
from uncertainty_mech.application.entity_pairs import EntityPair
from uncertainty_mech.application.ports import Capture, CircuitModel, Offsets, Patch, Site, SiteKind, Trace
from uncertainty_mech.domain.grading import abstention_gap, chosen_indices, semantic_entropy
from uncertainty_mech.domain.patching import restoration, smallest_sufficient_k
from uncertainty_mech.domain.prompts import ChatPrompt, answer_letters, build_prompt
from uncertainty_mech.domain.questions import OPTION_LETTERS

SEGMENTS = ("entity", "after_entity", "tail", "final")
E_LETTERS = answer_letters(allow_abstain=True)
Head = tuple[int, int]  # (layer, head)


@dataclass(frozen=True)
class Behavior:
    gap: np.ndarray  # abstention gap per prompt
    se: np.ndarray  # semantic entropy over A to D per prompt
    abstained: np.ndarray  # the model picked E

    @classmethod
    def of(cls, trace: Trace) -> Behavior:
        return cls(
            gap=abstention_gap(trace.letter_logprobs),
            se=semantic_entropy(trace.letter_logprobs),
            abstained=chosen_indices(trace.letter_logprobs) == len(OPTION_LETTERS),
        )


@dataclass(frozen=True)
class PairSet:
    """Matched prompts with option E, aligned by counting positions from the end."""

    pairs: tuple[EntityPair, ...]
    real: list[ChatPrompt]
    invented: list[ChatPrompt]
    entity_offsets: np.ndarray  # [n] last token where the two prompts differ
    tail_length: int  # final tokens shared by every prompt in the set

    def __len__(self) -> int:
        return len(self.pairs)

    def offsets(self, segment: str) -> Offsets:
        if segment == "entity":
            return self.entity_offsets[:, None]
        if segment == "after_entity":
            return (self.entity_offsets - 1)[:, None]
        if segment == "tail":
            return tuple(range(2, self.tail_length + 1))
        if segment == "final":
            return (1,)
        raise ValueError(f"unknown segment {segment!r}")


@dataclass(frozen=True)
class PairBaseline:
    real: Behavior
    invented: Behavior


def keep_known_pairs(model: CircuitModel, pairs: Sequence[EntityPair]) -> list[EntityPair]:
    """Pairs whose real question the model answers correctly when it must choose A to D."""
    prompts = [build_prompt(p.real.question, p.real.options, allow_abstain=False) for p in pairs]
    chosen = chosen_indices(model.trace(prompts, answer_letters(allow_abstain=False)).letter_logprobs)
    return [pair for pair, choice in zip(pairs, chosen, strict=True) if choice == pair.real.answer_index]


def align_pairs(model: CircuitModel, pairs: Sequence[EntityPair]) -> PairSet:
    real = [build_prompt(p.real.question, p.real.options, allow_abstain=True) for p in pairs]
    invented = [build_prompt(p.invented.question, p.invented.options, allow_abstain=True) for p in pairs]
    entity_offsets = np.array([model.common_suffix_length([r, i]) + 1 for r, i in zip(real, invented, strict=True)])
    tail_length = model.common_suffix_length(real + invented)
    if (entity_offsets - 1 <= tail_length).any():
        raise ValueError("an entity sits inside the shared tail; the pair prompts are not aligned")
    return PairSet(tuple(pairs), real, invented, entity_offsets, tail_length)


def pair_baseline(model: CircuitModel, pairs: PairSet) -> PairBaseline:
    return PairBaseline(
        real=Behavior.of(model.trace(pairs.real, E_LETTERS)),
        invented=Behavior.of(model.trace(pairs.invented, E_LETTERS)),
    )


def _effect(patched: Behavior, target: Behavior, source: Behavior, **labels: Any) -> dict[str, Any]:
    return {
        **labels,
        "restoration_gap": restoration(patched.gap, target.gap, source.gap),
        "restoration_se": restoration(patched.se, target.se, source.se),
        "delta_gap": float(patched.gap.mean() - target.gap.mean()),
        "delta_se": float(patched.se.mean() - target.se.mean()),
        "abstain_rate": float(patched.abstained.mean()),
    }


def sweep_residual(model: CircuitModel, pairs: PairSet, baseline: PairBaseline) -> list[dict[str, Any]]:
    """Invented-into-real patch of the residual stream, per layer and segment."""
    cells = [(layer, segment) for layer in range(model.num_layers) for segment in SEGMENTS]
    captures = [Capture(Site(SiteKind.RESID, layer), pairs.offsets(segment)) for layer, segment in cells]
    source = model.trace(pairs.invented, E_LETTERS, captures=captures)
    rows = []
    for (layer, segment), capture, values in zip(cells, captures, source.activations, strict=True):
        patched = model.trace(pairs.real, E_LETTERS, patches=[Patch(capture.site, capture.offsets, values)])
        rows.append(_effect(Behavior.of(patched), baseline.real, baseline.invented, layer=layer, segment=segment))
    return rows


def capture_heads(model: CircuitModel, prompts: Sequence[ChatPrompt]) -> np.ndarray:
    """Every head's output at the final token: [n, layers, heads, head_dim]."""
    captures = [Capture(Site(SiteKind.ATTN_Z, layer), (1,)) for layer in range(model.num_layers)]
    trace = model.trace(prompts, E_LETTERS, captures=captures)
    stacked = np.stack([values[:, 0] for values in trace.activations], axis=1)
    return stacked.reshape(len(prompts), model.num_layers, model.num_heads, model.head_dim)


def _head_patches(heads: Sequence[Head], source_z: np.ndarray) -> list[Patch]:
    return [
        Patch(Site(SiteKind.ATTN_Z, layer, head), (1,), source_z[:, layer, head][:, None, :]) for layer, head in heads
    ]


def sweep_heads(
    model: CircuitModel, pairs: PairSet, baseline: PairBaseline, invented_z: np.ndarray
) -> list[dict[str, Any]]:
    """Invented-into-real patch of one head at the final token, for every head."""
    rows = []
    for layer in range(model.num_layers):
        for head in range(model.num_heads):
            patched = model.trace(pairs.real, E_LETTERS, patches=_head_patches([(layer, head)], invented_z))
            rows.append(_effect(Behavior.of(patched), baseline.real, baseline.invented, layer=layer, head=head))
    return rows


def sweep_mlps(model: CircuitModel, pairs: PairSet, baseline: PairBaseline) -> list[dict[str, Any]]:
    """Invented-into-real patch of each MLP output at the final token."""
    captures = [Capture(Site(SiteKind.MLP, layer), (1,)) for layer in range(model.num_layers)]
    source = model.trace(pairs.invented, E_LETTERS, captures=captures)
    rows = []
    for layer, (capture, values) in enumerate(zip(captures, source.activations, strict=True)):
        patched = model.trace(pairs.real, E_LETTERS, patches=[Patch(capture.site, capture.offsets, values)])
        rows.append(_effect(Behavior.of(patched), baseline.real, baseline.invented, layer=layer))
    return rows


def rank_heads(head_rows: Sequence[dict[str, Any]]) -> list[Head]:
    """Heads ordered by how far a single-head patch moves the abstention gap toward the invented prompt."""
    scored = [row for row in head_rows if row["restoration_gap"] is not None]
    scored.sort(key=lambda row: row["restoration_gap"], reverse=True)
    return [(row["layer"], row["head"]) for row in scored]


@dataclass(frozen=True)
class CircuitChoice:
    heads: tuple[Head, ...]
    restoration_by_k: dict[int, float | None]


def choose_circuit(
    model: CircuitModel,
    pairs: PairSet,
    baseline: PairBaseline,
    invented_z: np.ndarray,
    ranking: Sequence[Head],
    config: CircuitConfig,
) -> CircuitChoice:
    """Smallest top-k whose joint patch reaches the sufficiency threshold on discovery pairs."""
    by_k: dict[int, float | None] = {}
    for k in config.k_grid:
        patched = Behavior.of(model.trace(pairs.real, E_LETTERS, patches=_head_patches(ranking[:k], invented_z)))
        by_k[k] = restoration(patched.gap, baseline.real.gap, baseline.invented.gap)
    k = smallest_sufficient_k(by_k, config.sufficiency_threshold)
    return CircuitChoice(tuple(ranking[:k]), by_k)


def validate_circuit(
    model: CircuitModel,
    pairs: PairSet,
    baseline: PairBaseline,
    circuit: Sequence[Head],
    n_random: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """On held-out pairs: the circuit versus random head sets of the same size, both directions."""
    real_z = capture_heads(model, pairs.real)
    invented_z = capture_heads(model, pairs.invented)
    all_heads = [(layer, head) for layer in range(model.num_layers) for head in range(model.num_heads)]
    others = [head for head in all_heads if head not in set(circuit)]
    head_sets = [("circuit", tuple(circuit))]
    for draw in range(n_random):
        picks = rng.choice(len(others), size=len(circuit), replace=False)
        head_sets.append((f"random_{draw + 1}", tuple(others[i] for i in picks)))
    rows = []
    for name, heads in head_sets:
        sufficient = Behavior.of(model.trace(pairs.real, E_LETTERS, patches=_head_patches(heads, invented_z)))
        necessary = Behavior.of(model.trace(pairs.invented, E_LETTERS, patches=_head_patches(heads, real_z)))
        rows.append(
            _effect(sufficient, baseline.real, baseline.invented, head_set=name, direction="invented_into_real")
        )
        rows.append(_effect(necessary, baseline.invented, baseline.real, head_set=name, direction="real_into_invented"))
    return rows


def steering_vectors(invented_z: np.ndarray, real_z: np.ndarray, heads: Sequence[Head]) -> np.ndarray:
    """Per head, the mean output on invented prompts minus real prompts: [k, head_dim]."""
    difference = (invented_z - real_z).mean(axis=0)
    return np.stack([difference[layer, head] for layer, head in heads]).astype(np.float32)
