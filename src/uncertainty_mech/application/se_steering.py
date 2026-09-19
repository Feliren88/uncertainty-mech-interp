"""Stage 3: use semantic entropy to switch the abstention circuit on, so the model itself says "I don't know".

Every condition runs on the prompt that offers option E; the model's own letter
is the answer. Gating uses the forced-choice semantic entropy saved by the
source run. Steering adds `dose * v_h` to each circuit head's output at the
final token. Rows are independent, so a gated condition takes the steered
pass where SE > tau and the plain pass elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from uncertainty_mech.application.circuit_config import SteeringGridConfig
from uncertainty_mech.application.dataset import Dataset, dataset_from_rows
from uncertainty_mech.application.patching import E_LETTERS, Head
from uncertainty_mech.application.ports import Addition, CircuitModel, RunStore, Site, SiteKind
from uncertainty_mech.domain.grading import chosen_indices, semantic_entropy
from uncertainty_mech.domain.prompts import build_prompt
from uncertainty_mech.domain.questions import OPTION_LETTERS, Role, Stratum

ABSTAIN = len(OPTION_LETTERS)
MAX_ENTROPY = math.log(len(OPTION_LETTERS))


@dataclass(frozen=True)
class SteeringVectors:
    heads: tuple[Head, ...]
    vectors: np.ndarray  # [k, head_dim]

    def additions(self, scale: np.ndarray) -> list[Addition]:
        return [
            Addition(Site(SiteKind.ATTN_Z, layer, head), (1,), self.vectors[i], scale)
            for i, (layer, head) in enumerate(self.heads)
        ]


@dataclass(frozen=True)
class EvalSet:
    """Questions from one role of the source run, with its forced-choice semantic entropy."""

    prompts: list
    se: np.ndarray
    answers: np.ndarray  # correct index, or -1 for invented entities
    forced_correct: np.ndarray  # the forced-choice answer (no option E) was right
    groups: np.ndarray  # related questions share a group; bootstraps resample whole groups

    @property
    def real(self) -> np.ndarray:
        return self.answers >= 0

    @classmethod
    def from_role(cls, dataset: Dataset, role: Role, forced_logprobs: np.ndarray, se: np.ndarray) -> EvalSet:
        rows = np.flatnonzero(dataset.mask(role))
        questions = [dataset.questions[i] for i in rows]
        answers = np.array([-1 if q.stratum is Stratum.FICTIONAL else q.answer_index for q in questions])
        return cls(
            prompts=[build_prompt(q.question, q.options, allow_abstain=True) for q in questions],
            se=se[rows],
            answers=answers,
            forced_correct=chosen_indices(forced_logprobs[rows]) == answers,
            groups=dataset.groups[rows],
        )


def ensure_same_model(source: RunStore, model: CircuitModel) -> None:
    """Semantic entropy and roles come from the source run, so the model must be the same revision."""
    source_model = source.read_json("bundle.json")["model_name"]
    if source_model != model.name:
        raise ValueError(f"the source run used {source_model}, not {model.name}")


def load_eval_sets(source: RunStore) -> tuple[EvalSet, EvalSet]:
    """Calibration and test questions of a finished `run`, with its forced-choice semantic entropy."""
    dataset = dataset_from_rows(source.read_table("items.csv"))
    forced = source.read_arrays("readings.npz", keys=("letter_logprobs",))["letter_logprobs"]
    se = semantic_entropy(forced)
    return EvalSet.from_role(dataset, Role.CAL_PROB, forced, se), EvalSet.from_role(dataset, Role.TEST, forced, se)


def outcome(choice: np.ndarray, items: EvalSet) -> dict[str, float]:
    """What a user would see: answers given, how many are wrong, and what got refused."""
    answered = choice != ABSTAIN
    correct = answered & (choice == items.answers)
    wrong = answered & ~correct
    invented = ~items.real
    return {
        "n": int(len(choice)),
        "answered_share": float(answered.mean()),
        "wrong_among_answered": float(wrong.sum() / answered.sum()) if answered.any() else float("nan"),
        "right_kept": float(correct.sum() / items.forced_correct.sum()) if items.forced_correct.any() else float("nan"),
        "real_refused": float((~answered & items.real).sum() / items.real.sum()) if items.real.any() else float("nan"),
        "invented_refused": float((~answered & invented).sum() / invented.sum()) if invented.any() else float("nan"),
        "net_correct": float((correct.sum() - wrong.sum()) / len(choice)),
    }


def _choices(model: CircuitModel, items: EvalSet, vectors: SteeringVectors | None, scale: np.ndarray) -> np.ndarray:
    additions = [] if vectors is None else vectors.additions(scale.astype(np.float32))
    return chosen_indices(model.trace(items.prompts, E_LETTERS, additions=additions).letter_logprobs)


def _uniform(items: EvalSet, dose: float) -> np.ndarray:
    return np.full(len(items.prompts), dose, dtype=np.float32)


def _gated(se: np.ndarray, tau: float, steered: np.ndarray, plain: np.ndarray) -> np.ndarray:
    return np.where(se > tau, steered, plain)


@dataclass(frozen=True)
class OperatingPoints:
    gated_dose: float
    gated_tau: float
    scaled_dose: float
    wrapper_tau: float


def calibrate(
    model: CircuitModel, items: EvalSet, vectors: SteeringVectors, grid: SteeringGridConfig
) -> tuple[OperatingPoints, list[dict[str, Any]]]:
    """Pick dose and threshold that maximize net correct answers on calibration items."""
    plain = _choices(model, items, None, _uniform(items, 0.0))
    rows: list[dict[str, Any]] = []
    for tau in grid.se_thresholds:
        wrapper = np.where(items.se > tau, ABSTAIN, plain)
        rows.append({"condition": "se_wrapper", "dose": 0.0, "tau": tau, **outcome(wrapper, items)})
    for dose in grid.doses:
        steered = _choices(model, items, vectors, _uniform(items, dose))
        for tau in grid.se_thresholds:
            gated = _gated(items.se, tau, steered, plain)
            rows.append({"condition": "se_gated_circuit", "dose": dose, "tau": tau, **outcome(gated, items)})
        scaled = _choices(model, items, vectors, dose * items.se / MAX_ENTROPY)
        rows.append({"condition": "se_scaled_circuit", "dose": dose, "tau": None, **outcome(scaled, items)})

    def best(condition: str) -> dict[str, Any]:
        candidates = [row for row in rows if row["condition"] == condition]
        return max(candidates, key=lambda row: (row["net_correct"], -row["dose"], row["tau"] or 0.0))

    points = OperatingPoints(
        gated_dose=best("se_gated_circuit")["dose"],
        gated_tau=best("se_gated_circuit")["tau"],
        scaled_dose=best("se_scaled_circuit")["dose"],
        wrapper_tau=best("se_wrapper")["tau"],
    )
    return points, rows


@dataclass(frozen=True)
class SteeringEvaluation:
    rows: list[dict[str, Any]]  # one per condition
    flips: list[dict[str, Any]]  # among gated real questions: how often steering flips right vs wrong answers to E
    curves: dict[str, tuple[np.ndarray, np.ndarray]]  # answered share and wrong-among-answered over tau
    curve_rows: list[dict[str, Any]]  # the same curves as a table
    predictions: list[dict[str, Any]]  # one per test item: SE, answer and the letter chosen under each pass


def evaluate(
    model: CircuitModel,
    items: EvalSet,
    circuit: SteeringVectors,
    random_heads: SteeringVectors,
    random_vectors: SteeringVectors,
    points: OperatingPoints,
    thresholds: Sequence[float],
) -> SteeringEvaluation:
    plain = _choices(model, items, None, _uniform(items, 0.0))
    steered = _choices(model, items, circuit, _uniform(items, points.gated_dose))
    scaled = _choices(model, items, circuit, points.scaled_dose * items.se / MAX_ENTROPY)
    steered_random_heads = _choices(model, items, random_heads, _uniform(items, points.gated_dose))
    steered_random_vectors = _choices(model, items, random_vectors, _uniform(items, points.gated_dose))
    tau = points.gated_tau
    conditions = {
        "prompt_only": plain,
        "se_wrapper": np.where(items.se > points.wrapper_tau, ABSTAIN, plain),
        "se_gated_circuit": _gated(items.se, tau, steered, plain),
        "se_scaled_circuit": scaled,
        "always_on_circuit": steered,
        "control_random_heads": _gated(items.se, tau, steered_random_heads, plain),
        "control_random_vectors": _gated(items.se, tau, steered_random_vectors, plain),
    }
    rows = [{"condition": name, **outcome(choice, items)} for name, choice in conditions.items()]

    gated_real = (items.se > tau) & items.real & (plain != ABSTAIN)
    flips = []
    for name, choice in (
        ("circuit", steered),
        ("random_heads", steered_random_heads),
        ("random_vectors", steered_random_vectors),
    ):
        for label, group in (("right before", items.forced_correct), ("wrong before", ~items.forced_correct)):
            rows_in_group = gated_real & group
            flips.append(
                {
                    "steering": name,
                    "group": label,
                    "n": int(rows_in_group.sum()),
                    "flipped_to_e": float((choice[rows_in_group] == ABSTAIN).mean()) if rows_in_group.any() else None,
                }
            )

    curves = {
        "se_wrapper": _curve(items, [np.where(items.se > t, ABSTAIN, plain) for t in thresholds]),
        "se_gated_circuit": _curve(items, [_gated(items.se, t, steered, plain) for t in thresholds]),
        "control_random_heads": _curve(items, [_gated(items.se, t, steered_random_heads, plain) for t in thresholds]),
    }
    curve_rows = [
        {"condition": name, "tau": tau, "answered_share": float(x), "wrong_among_answered": float(y)}
        for name, (answered, wrong) in curves.items()
        for tau, x, y in zip(thresholds, answered, wrong, strict=True)
    ]
    passes = {
        "plain": plain,
        "circuit": steered,
        "circuit_scaled": scaled,
        "random_heads": steered_random_heads,
        "random_vectors": steered_random_vectors,
    }
    predictions = [
        {
            "se": float(items.se[i]),
            "answer": int(items.answers[i]),
            "forced_correct": bool(items.forced_correct[i]),
            **{f"choice_{name}": int(choice[i]) for name, choice in passes.items()},
        }
        for i in range(len(items.prompts))
    ]
    return SteeringEvaluation(rows, flips, curves, curve_rows, predictions)


def _curve(items: EvalSet, choices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    results = [outcome(choice, items) for choice in choices]
    return (
        np.array([r["answered_share"] for r in results]),
        np.array([r["wrong_among_answered"] for r in results]),
    )


def random_head_vectors(
    invented_z: np.ndarray, real_z: np.ndarray, circuit: Sequence[Head], rng: np.random.Generator
) -> SteeringVectors:
    """Control: mean-difference vectors at randomly chosen heads outside the circuit."""
    layers, heads = invented_z.shape[1:3]
    others = [(layer, head) for layer in range(layers) for head in range(heads) if (layer, head) not in set(circuit)]
    picks = tuple(others[i] for i in rng.choice(len(others), size=len(circuit), replace=False))
    difference = (invented_z - real_z).mean(axis=0)
    return SteeringVectors(picks, np.stack([difference[layer, head] for layer, head in picks]).astype(np.float32))


def random_direction_vectors(circuit: SteeringVectors, rng: np.random.Generator) -> SteeringVectors:
    """Control: random vectors at the circuit heads, each with its circuit vector's length."""
    noise = rng.standard_normal(circuit.vectors.shape).astype(np.float32)
    lengths = np.linalg.norm(circuit.vectors, axis=1, keepdims=True) / np.linalg.norm(noise, axis=1, keepdims=True)
    return SteeringVectors(circuit.heads, noise * lengths)
