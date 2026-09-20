"""Tables, figures and the markdown report for the circuit study, built only from its own numbers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

from uncertainty_mech.application.circuit_config import CircuitStudyConfig
from uncertainty_mech.application.markdown_format import num
from uncertainty_mech.application.markdown_format import pct as _pct
from uncertainty_mech.application.markdown_format import table as _table
from uncertainty_mech.application.patching import SEGMENTS, CircuitChoice, Head, PairBaseline, PairSet
from uncertainty_mech.application.ports import CircuitFigureWriter, RunStore
from uncertainty_mech.application.se_steering import OperatingPoints, SteeringEvaluation
from uncertainty_mech.application.tiered_steering import TieredResults
from uncertainty_mech.application.tiers_report import choice_row, draw_tier_figure, tier_markdown, write_tier_tables

SEGMENT_LABELS = ("Last entity token", "Token after entity", "Instruction tail", "Final token")
SEGMENT_TEXT = dict(zip(SEGMENTS, (label.lower() for label in SEGMENT_LABELS), strict=True))
CONDITION_LABELS = {
    "prompt_only": "Prompt only (option E offered)",
    "se_wrapper": 'SE wrapper (reply "I don\'t know" above tau)',
    "se_gated_circuit": "SE-gated circuit steering",
    "se_scaled_circuit": "SE-scaled circuit steering",
    "always_on_circuit": "Circuit steering on every question",
    "control_random_heads": "Control: SE-gated, random heads",
    "control_random_vectors": "Control: SE-gated, random vectors at circuit heads",
}


@dataclass(frozen=True)
class PairSummary:
    rows: list[dict[str, Any]]

    @classmethod
    def of(
        cls,
        discovery: PairSet,
        discovery_baseline: PairBaseline,
        test_baseline: PairBaseline,
        n_discovery: int,
        n_test: int,
    ) -> PairSummary:
        rows = []
        for split, baseline, total in (("discovery", discovery_baseline, n_discovery), ("test", test_baseline, n_test)):
            for side, behavior in (("real", baseline.real), ("invented", baseline.invented)):
                rows.append(
                    {
                        "split": split,
                        "side": side,
                        "pairs_kept": len(behavior.gap),
                        "pairs_built": total,
                        "abstain_rate": float(behavior.abstained.mean()),
                        "mean_gap": float(behavior.gap.mean()),
                        "mean_se": float(behavior.se.mean()),
                    }
                )
        return cls(rows)

    def get(self, split: str, side: str) -> dict[str, Any]:
        return next(row for row in self.rows if row["split"] == split and row["side"] == side)


@dataclass(frozen=True)
class CircuitResults:
    config: CircuitStudyConfig
    model_name: str
    pairs: PairSummary
    tail_length: int
    residual_rows: list[dict[str, Any]]
    head_rows: list[dict[str, Any]]
    mlp_rows: list[dict[str, Any]]
    circuit: CircuitChoice
    validation_rows: list[dict[str, Any]]
    vector_norms: np.ndarray
    random_heads: tuple[Head, ...]
    points: OperatingPoints
    calibration_rows: list[dict[str, Any]]
    evaluation: SteeringEvaluation
    tiered: TieredResults
    started: datetime
    finished: datetime


def write_circuit_report(store: RunStore, figures: CircuitFigureWriter, results: CircuitResults) -> dict[str, Any]:
    _write_tables(store, results)
    layers = 1 + max(row["layer"] for row in results.head_rows)
    heads = 1 + max(row["head"] for row in results.head_rows)
    figures.heatmap(
        _grid(results.residual_rows, layers, SEGMENTS, "segment"),
        SEGMENT_LABELS,
        "Position",
        "Decoder layer",
        "Share of the abstention gap moved",
        store.root / "figures" / "residual_patching.png",
    )
    figures.heatmap(
        _grid(results.head_rows, layers, range(heads), "head"),
        None,
        "Attention head",
        "Decoder layer",
        "Share of the abstention gap moved",
        store.root / "figures" / "head_patching.png",
    )
    figures.steering_tradeoff(
        results.evaluation.curves, _points(results.evaluation.rows), store.root / "figures" / "se_steering.png"
    )
    draw_tier_figure(figures, results.tiered, store.root / "figures" / "tiers_tradeoff.png")
    summary = _summary(results)
    store.write_json("summary.json", summary)
    store.write_text("report.md", _markdown(results, summary))
    return summary


def _write_tables(store: RunStore, results: CircuitResults) -> None:
    store.write_table("pairs.csv", results.pairs.rows)
    store.write_table("residual_patching.csv", results.residual_rows)
    store.write_table("head_patching.csv", results.head_rows)
    store.write_table("mlp_patching.csv", results.mlp_rows)
    store.write_table("circuit_validation.csv", results.validation_rows)
    store.write_table("steering_calibration.csv", results.calibration_rows)
    store.write_table("steering_test.csv", results.evaluation.rows)
    store.write_table("steering_flips.csv", results.evaluation.flips)
    store.write_table("steering_curves.csv", results.evaluation.curve_rows)
    store.write_table("steering_predictions.csv", results.evaluation.predictions)
    write_tier_tables(store, results.tiered)
    store.write_json(
        "circuit.json",
        {
            "heads": [list(head) for head in results.circuit.heads],
            "restoration_by_k": {str(k): v for k, v in results.circuit.restoration_by_k.items()},
            "vector_norms": results.vector_norms.tolist(),
            "random_control_heads": [list(head) for head in results.random_heads],
            "operating_points": asdict(results.points),
        },
    )


def _grid(rows: Sequence[dict[str, Any]], layers: int, columns: Sequence[Any], key: str) -> np.ndarray:
    grid = np.full((layers, len(columns)), np.nan)
    index = {column: i for i, column in enumerate(columns)}
    for row in rows:
        value = row["restoration_gap"]
        grid[row["layer"], index[row[key]]] = np.nan if value is None else value
    return grid


def _points(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    wanted = ("prompt_only", "always_on_circuit", "se_scaled_circuit", "control_random_vectors")
    return {
        row["condition"]: (row["answered_share"], row["wrong_among_answered"])
        for row in rows
        if row["condition"] in wanted
    }


def _validation(results: CircuitResults, direction: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [row for row in results.validation_rows if row["direction"] == direction]
    return next(row for row in rows if row["head_set"] == "circuit"), [
        row for row in rows if row["head_set"] != "circuit"
    ]


def _condition(results: CircuitResults, name: str) -> dict[str, Any]:
    return next(row for row in results.evaluation.rows if row["condition"] == name)


def _flip(results: CircuitResults, steering: str, group: str) -> float | None:
    return next(
        r["flipped_to_e"] for r in results.evaluation.flips if r["steering"] == steering and r["group"] == group
    )


def _summary(results: CircuitResults) -> dict[str, Any]:
    sufficiency, random_sufficiency = _validation(results, "invented_into_real")
    necessity, random_necessity = _validation(results, "real_into_invented")
    return {
        "run_id": results.config.run_id,
        "model": results.model_name,
        "circuit_heads": [list(head) for head in results.circuit.heads],
        "test_sufficiency": sufficiency["restoration_gap"],
        "test_sufficiency_random_mean": _mean([row["restoration_gap"] for row in random_sufficiency]),
        "test_necessity": necessity["restoration_gap"],
        "test_necessity_random_mean": _mean([row["restoration_gap"] for row in random_necessity]),
        "operating_points": asdict(results.points),
        "steering_test": {row["condition"]: row for row in results.evaluation.rows},
        "readout_auroc_invented": results.tiered.readout_auroc_invented,
        "readout_auroc_unknown_real": results.tiered.readout_auroc_unknown_real,
        "tiered_test": list(results.tiered.test_rows),
        "tiered_choices": [choice_row(choice) for choice in results.tiered.choices],
    }


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def _heads_text(k: int) -> str:
    return "1 attention head" if k == 1 else f"{k} attention heads"


def _num(value: float | None, digits: int = 2) -> str:
    return num(value, digits)


def _markdown(results: CircuitResults, summary: dict[str, Any]) -> str:
    sections = [
        f"# Abstention circuit and semantic-entropy steering: {results.config.run_id}",
        _headline(results, summary),
        "## Matched pairs",
        _pairs(results),
        "## Location of the signal in the residual stream",
        _residual(results),
        "## Attention heads at the final token",
        _heads(results),
        "## The circuit on held-out pairs",
        _circuit(results),
        "## Semantic-entropy steering on MedQA",
        _steering(results),
        "## Tiered semantic-entropy steering",
        tier_markdown(results.tiered),
        "## Limits",
        "\n".join(
            [
                "- Patching at the final token finds where the model reads the signal. The path from the entity "
                "to the final token is only partly mapped.",
                "- Invented names differ from real ones in familiarity and in form (real drug names contain class "
                "suffixes such as -pril or -statin). The pairs cannot separate the two.",
                "- The steering vectors come from templated pairs and are applied to MedQA prompts; the transfer is "
                "an empirical result for this prompt format only.",
                "- Dose and threshold were chosen on the source run's cal_prob role and scored once on its test role.",
                "- One model, one prompt format. Nothing here is medical advice.",
            ]
        ),
    ]
    return "\n\n".join(sections) + "\n"


def _headline(results: CircuitResults, summary: dict[str, Any]) -> str:
    k = len(results.circuit.heads)
    gated = _condition(results, "se_gated_circuit")
    plain = _condition(results, "prompt_only")
    wrapper = _condition(results, "se_wrapper")
    return (
        f"On held-out pairs, copying the output of {_heads_text(k)} at the final token from an invented-entity "
        f"prompt into its matched real-entity prompt moved the abstention gap {_pct(summary['test_sufficiency'])} of "
        f"the way toward the invented prompt; {_heads_text(k).replace('attention', 'random')} moved it "
        f"{_pct(summary['test_sufficiency_random_mean'])} "
        f"on average. Steering the circuit only when semantic entropy was above {results.points.gated_tau:.1f} nats "
        f"(dose {results.points.gated_dose:g}), the model answered {_pct(gated['answered_share'])} of MedQA test "
        f"questions and was wrong on {_pct(gated['wrong_among_answered'])} of its answers. With option E alone it "
        f"answered {_pct(plain['answered_share'])} and was wrong on {_pct(plain['wrong_among_answered'])}. An SE "
        f'wrapper that replies "I don\'t know" without changing any activation answered '
        f"{_pct(wrapper['answered_share'])} with {_pct(wrapper['wrong_among_answered'])} wrong."
    )


def _pairs(results: CircuitResults) -> str:
    rows = [
        [
            row["split"],
            row["side"],
            f"{row['pairs_kept']} of {row['pairs_built']}",
            _pct(row["abstain_rate"]),
            _num(row["mean_gap"]),
            _num(row["mean_se"]),
        ]
        for row in results.pairs.rows
    ]
    return "\n\n".join(
        [
            "Each pair asks the same question with the same options about a real entity and an invented one. Only "
            "pairs whose real question the model answers correctly are kept. The abstention gap is log P(E) minus "
            'log P(A to D); above 0 the model prefers "I don\'t know". SE is semantic entropy over A to D in nats.',
            _table(["Split", "Prompt", "Pairs kept", "Picks E", "Mean abstention gap", "Mean SE"], rows),
        ]
    )


def _residual(results: CircuitResults) -> str:
    best = sorted(
        (row for row in results.residual_rows if row["restoration_gap"] is not None),
        key=lambda row: row["restoration_gap"],
        reverse=True,
    )[:6]
    rows = [
        [str(row["layer"]), SEGMENT_TEXT[row["segment"]], _num(row["restoration_gap"]), _num(row["restoration_se"])]
        for row in best
    ]
    return "\n\n".join(
        [
            "Each cell copies the residual stream at one layer and position from the invented prompt into the real "
            "prompt and reports the share of the abstention gap it moves (1 means the real prompt now behaves like "
            f"the invented one). The instruction tail is the {results.tail_length - 1} shared tokens before the "
            "final token.",
            "![Residual patching](figures/residual_patching.png)",
            f"*The strongest single cell is layer {best[0]['layer']} at the {SEGMENT_TEXT[best[0]['segment']]}, which "
            f"moves {_pct(best[0]['restoration_gap'])} of the abstention gap.*",
            _table(["Layer", "Segment", "Gap moved", "SE moved"], rows),
        ]
    )


def _heads(results: CircuitResults) -> str:
    ranked = sorted(
        (row for row in results.head_rows if row["restoration_gap"] is not None),
        key=lambda row: row["restoration_gap"],
        reverse=True,
    )
    top = ranked[:10]
    rows = [
        [
            f"L{row['layer']}.H{row['head']}",
            _num(row["restoration_gap"], 3),
            _num(row["delta_gap"]),
            _num(row["delta_se"], 3),
        ]
        for row in top
    ]
    mlps = sorted(
        (row for row in results.mlp_rows if row["restoration_gap"] is not None),
        key=lambda row: row["restoration_gap"],
        reverse=True,
    )[:3]
    mlp_text = ", ".join(f"layer {row['layer']} ({_pct(row['restoration_gap'])})" for row in mlps)
    return "\n\n".join(
        [
            "Each cell copies one head's output at the final token from the invented prompt into the real prompt.",
            "![Head patching](figures/head_patching.png)",
            f"*The strongest single head, L{top[0]['layer']}.H{top[0]['head']}, moves "
            f"{_pct(top[0]['restoration_gap'])} of the abstention gap on its own.*",
            _table(["Head", "Gap moved", "Change in gap (nats)", "Change in SE (nats)"], rows),
            f"The strongest head in the other direction is L{ranked[-1]['layer']}.H{ranked[-1]['head']}, at "
            f"{_pct(ranked[-1]['restoration_gap'])}: copying its invented-prompt output makes the real prompt even "
            'less likely to abstain, so its output on invented prompts acts against "I don\'t know".',
            f"The MLPs that move the gap most at the final token are {mlp_text}.",
        ]
    )


def _circuit(results: CircuitResults) -> str:
    by_k = [[str(k), _pct(value)] for k, value in results.circuit.restoration_by_k.items()]
    k = len(results.circuit.heads)
    heads = ", ".join(f"L{layer}.H{head}" for layer, head in results.circuit.heads)
    rows = []
    for direction, label in (
        ("invented_into_real", "Invented into real"),
        ("real_into_invented", "Real into invented"),
    ):
        circuit, randoms = _validation(results, direction)
        random_values = [row["restoration_gap"] for row in randoms if row["restoration_gap"] is not None]
        rows.append(
            [
                label,
                _pct(circuit["restoration_gap"]),
                f"{_pct(_mean(random_values))} ({_pct(min(random_values))} to {_pct(max(random_values))})"
                if random_values
                else "n/a",
                _num(circuit["delta_se"], 3),
                _pct(circuit["abstain_rate"]),
            ]
        )
    return "\n\n".join(
        [
            f"The circuit is the smallest set of top-ranked heads whose joint patch moves at least "
            f"{_pct(results.config.circuit.sufficiency_threshold)} of the gap on discovery pairs. It has "
            f"{_heads_text(k)}: {heads}.",
            _table(["Top k heads", "Gap moved on discovery pairs"], by_k),
            f"On held-out test pairs, compared with {results.config.circuit.n_random} random sets of "
            f"{_heads_text(k).replace('attention ', '')}:",
            _table(
                [
                    "Patch",
                    "Gap moved by circuit",
                    "Gap moved by random heads",
                    "Change in SE (nats)",
                    "Picks E after patch",
                ],
                rows,
            ),
        ]
    )


def _steering(results: CircuitResults) -> str:
    points = results.points
    rows = [
        [
            CONDITION_LABELS[row["condition"]],
            _pct(row["answered_share"]),
            _pct(row["wrong_among_answered"]),
            _pct(row["right_kept"]),
            _pct(row["invented_refused"]),
            _num(row["net_correct"], 3),
        ]
        for row in results.evaluation.rows
    ]
    flips = [
        [steering.replace("_", " "), group, str(n), _pct(value)]
        for steering, group, n, value in (
            (row["steering"], row["group"], row["n"], row["flipped_to_e"]) for row in results.evaluation.flips
        )
    ]
    right = _flip(results, "circuit", "right before")
    wrong = _flip(results, "circuit", "wrong before")
    return "\n\n".join(
        [
            f"All conditions use the prompt with option E, so the model's own letter is the answer. Gating uses the "
            f"forced-choice semantic entropy from the source run. Chosen on cal_prob by net correct answers "
            f"(right answers minus wrong answers, per question): gated dose {points.gated_dose:g} at tau "
            f"{points.gated_tau:.1f} nats, scaled dose {points.scaled_dose:g}, wrapper tau {points.wrapper_tau:.1f}.",
            _table(
                [
                    "Condition",
                    "Answered",
                    "Wrong among answered",
                    "Right answers kept",
                    "Invented refused",
                    "Net correct",
                ],
                rows,
            ),
            "![SE steering](figures/se_steering.png)",
            f"*Among real questions above the gate, circuit steering turned {_pct(right)} of previously right answers "
            f'and {_pct(wrong)} of previously wrong answers into "I don\'t know".*',
            "The circuit adds information beyond semantic entropy only if wrong answers are changed to E more often "
            "than right ones, and more often than under the controls.",
            _table(["Steering", "Real answers before steering", "n", "Turned into E"], flips),
        ]
    )
