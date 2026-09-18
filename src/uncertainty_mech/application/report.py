"""Stage 9: CSV tables, two figures and a markdown report built only from this run's numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

from uncertainty_mech.application.config import RunConfig
from uncertainty_mech.application.dataset import Dataset
from uncertainty_mech.application.evaluation import NO_GATE, PROMPT_ONLY, Evaluation
from uncertainty_mech.application.gates import PRIMARY_SIGNAL, FittedGate
from uncertainty_mech.application.ports import FigureWriter, RunStore
from uncertainty_mech.application.risk_model import LayerScore, SignalSet
from uncertainty_mech.application.steering import ERROR_DIRECTION, SteeringResult
from uncertainty_mech.domain.questions import Role, Stratum

METHOD_LABELS = {
    NO_GATE: "No gate (always answer)",
    PROMPT_ONLY: "Prompt-only option E",
    SignalSet.OUTPUT.value: "Output statistics gate",
    SignalSet.PROBE.value: "Activation probe gate",
    SignalSet.COMBINED.value: "Combined gate (primary)",
}

LIMITS = """- One model (Llama 3.1 8B Instruct), one prompt format and one sample of MedQA. Free-text answers would need new calibration.
- The certificate assumes calibration and test questions are independent draws from the same mix of real and invented items. A new mix, model revision or prompt voids it.
- Invented entities are easier to spot than wrong answers to real questions, so read the real-only table before the pooled one.
- MedQA is public and may be in the model's pretraining data.
- The steering check uses one layer, one direction and a small sample. It is exploratory and does not identify a circuit.
- Nothing here is medical advice or a validated clinical tool."""


@dataclass(frozen=True)
class RunResults:
    config: RunConfig
    model_name: str
    dataset: Dataset
    errors: np.ndarray
    sweep: list[LayerScore]
    layer: int
    gates: dict[SignalSet, FittedGate]
    evaluation: Evaluation
    steering: SteeringResult
    started: datetime
    finished: datetime


def write_report(store: RunStore, figures: FigureWriter, results: RunResults) -> dict[str, Any]:
    _write_tables(store, results)
    figures.layer_sweep([asdict(score) for score in results.sweep], results.layer, store.root / "figures" / "layer_sweep.png")
    figures.risk_coverage(
        results.evaluation.curves,
        results.evaluation.prompt_only_point,
        results.config.gate.target_risk,
        store.root / "figures" / "risk_coverage.png",
    )
    summary = _summary(results)
    store.write_json("summary.json", summary)
    store.write_text("report.md", _markdown(results))
    return summary


def _write_tables(store: RunStore, results: RunResults) -> None:
    store.write_table("layer_sweep.csv", [asdict(score) for score in results.sweep])
    store.write_table(
        "c_selection.csv",
        [{"signal": signal.value, **asdict(score)} for signal, gate in results.gates.items() for score in gate.c_scores],
    )
    store.write_table(
        "thresholds.csv",
        [
            {"signal": signal.value, **asdict(check), "selected": check.threshold == gate.selection.threshold}
            for signal, gate in results.gates.items()
            for check in gate.selection.checks
        ],
    )
    evaluation = results.evaluation
    store.write_table("metrics.csv", evaluation.metrics_rows)
    store.write_table("risk_quality.csv", evaluation.quality_rows)
    store.write_table("matched_coverage.csv", evaluation.matched_rows)
    store.write_table("test_predictions.csv", evaluation.prediction_rows)
    store.write_table("steering.csv", results.steering.rows)


def _counts(dataset: Dataset) -> dict[str, dict[str, int]]:
    return {
        role.value: {
            stratum.value: int((dataset.mask(role) & (dataset.strata == stratum.value)).sum()) for stratum in Stratum
        }
        for role in Role
    }


def _summary(results: RunResults) -> dict[str, Any]:
    real = results.dataset.strata == Stratum.REAL.value
    best = next(score for score in results.sweep if score.layer == results.layer)
    test: dict[str, dict[str, Any]] = {}
    for row in results.evaluation.metrics_rows:
        test.setdefault(row["method"], {})[row["stratum"]] = {
            key: row[key] for key in ("n", "coverage", "selective_risk", "risk_upper_95", "correct_retention")
        }
    return {
        "run_id": results.config.run_id,
        "model": results.model_name,
        "started_utc": results.started.isoformat(),
        "finished_utc": results.finished.isoformat(),
        "counts": _counts(results.dataset),
        "real_accuracy_no_gate": float(1 - results.errors[real].mean()) if real.any() else None,
        "probe_layer": results.layer,
        "probe_layer_auroc": best.auroc_mean,
        "primary_signal": PRIMARY_SIGNAL.value,
        "target_risk": results.config.gate.target_risk,
        "thresholds": {signal.value: gate.selection.threshold for signal, gate in results.gates.items()},
        "test": test,
        "steering": {
            "layer": results.steering.layer,
            "identity_ok": results.steering.identity_ok,
            "identity_max_abs_diff": results.steering.identity_max_abs_diff,
        },
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _markdown(results: RunResults) -> str:
    config = results.config
    sections = [
        f'# Health "I don\'t know" gate: {config.run_id}',
        _headline(results),
        _setup(results),
        "## Test results, all questions",
        _metrics_table(results.evaluation, "all"),
        "## Test results, real MedQA questions only",
        _metrics_table(results.evaluation, "real"),
        "## Test results, invented drugs and diseases only",
        _metrics_table(results.evaluation, "fictional"),
        "## Same coverage as the prompt-only baseline",
        _matched(results),
        "## Risk ranking on the test role",
        _quality_table(results.evaluation),
        "## Where the error signal lives",
        _sweep(results),
        "## Threshold certification on cal_gate",
        _thresholds(results),
        "## Steering check (exploratory)",
        _steering(results.steering),
        "## Examples from the test role",
        _examples(results),
        "## Data",
        _data(results),
        "## Limits",
        LIMITS,
    ]
    return "\n\n".join(sections) + "\n"


def _headline(results: RunResults) -> str:
    evaluation = results.evaluation
    primary = evaluation.metric(PRIMARY_SIGNAL.value, "all")
    no_gate = evaluation.metric(NO_GATE, "all")
    target = _pct(results.config.gate.target_risk)
    if results.gates[PRIMARY_SIGNAL].selection.threshold is None:
        return (
            f"Calibration found no threshold whose 95% upper bound on the error rate of released answers stays at "
            f"or below {target}, so the primary gate says \"I don't know.\" to every question. Without a gate the "
            f"model was wrong on {_pct(no_gate['selective_risk'])} of {no_gate['n']} test questions."
        )
    text = (
        f"On {primary['n']} held-out test questions, the combined gate answered {_pct(primary['coverage'])} and was "
        f"wrong on {_pct(primary['selective_risk'])} of those answers (one-sided 95% upper bound "
        f"{_pct(primary['risk_upper_95'])}, target {target}). Without the gate the model answered every question and "
        f"was wrong on {_pct(no_gate['selective_risk'])}."
    )
    gate_fictional = evaluation.metric(PRIMARY_SIGNAL.value, "fictional")
    prompt_fictional = evaluation.metric(PROMPT_ONLY, "fictional")
    if gate_fictional and prompt_fictional:
        text += (
            f" For questions about invented drugs and diseases, the gate said \"I don't know.\" "
            f"{_pct(1 - gate_fictional['coverage'])} of the time. Offering \"I don't know\" as option E in the prompt "
            f"got {_pct(1 - prompt_fictional['coverage'])}."
        )
    return text


def _setup(results: RunResults) -> str:
    config = results.config
    counts = _counts(results.dataset)
    n_real = sum(role[Stratum.REAL.value] for role in counts.values())
    n_fictional = sum(role[Stratum.FICTIONAL.value] for role in counts.values())
    best = next(score for score in results.sweep if score.layer == results.layer)
    rows = [
        ["Model", results.model_name],
        ["Questions", f"{n_real + n_fictional} ({n_real} MedQA, {n_fictional} invented)"],
        ["Probe layer", f"{results.layer} (discovery AUROC {best.auroc_mean:.3f})"],
        [
            "Target",
            f"error rate of released answers at most {_pct(config.gate.target_risk)} with 95% confidence "
            f"(delta {config.gate.delta} split over 20 thresholds)",
        ],
        ["Run time", f"{results.started:%Y-%m-%d %H:%M} to {results.finished:%H:%M} UTC"],
    ]
    return _table(["Item", "Value"], rows)


def _metrics_table(evaluation: Evaluation, stratum: str) -> str:
    rows = []
    for method, label in METHOD_LABELS.items():
        row = evaluation.metric(method, stratum)
        if row is None:
            continue
        rows.append(
            [
                label,
                f"{row['released']}/{row['n']}",
                _pct(row["selective_risk"]),
                _pct(row["risk_upper_95"]),
                _pct(row["false_answer_rate"]),
                _pct(row["correct_retention"]),
            ]
        )
    if not rows:
        return "No test items in this stratum."
    return _table(
        ["Method", "Answered", "Wrong among answered", "95% upper bound", "Wrong per question", "Right answers kept"],
        rows,
    )


def _matched(results: RunResults) -> str:
    coverage, risk = results.evaluation.prompt_only_point
    rows = [
        [METHOD_LABELS[row["method"]], f"{row['answered']}", _pct(row["coverage"]), _pct(row["selective_risk"])]
        for row in results.evaluation.matched_rows
    ]
    combined = next(row for row in results.evaluation.matched_rows if row["method"] == PRIMARY_SIGNAL.value)
    caption = (
        f"*At the prompt-only baseline's coverage of {_pct(coverage)}, the combined gate is wrong on "
        f"{_pct(combined['selective_risk'])} of its answers and the prompt-only baseline on {_pct(risk)}.*"
    )
    return "\n\n".join(
        [
            "Each gate answers the same share of test questions as the prompt-only baseline, taking its "
            "lowest-risk questions first. This comparison is descriptive and uses no certified threshold.",
            _table(["Method", "Answered", "Coverage", "Wrong among answered"], rows),
            "![Risk-coverage curves](figures/risk_coverage.png)",
            caption,
        ]
    )


def _quality_table(evaluation: Evaluation) -> str:
    rows = [
        [
            METHOD_LABELS[row["signal"]],
            row["stratum"],
            str(row["n"]),
            _pct(row["error_rate"]),
            _num(row["auroc"]),
            _num(row["brier"]),
            _num(row["ece"]),
            _num(row["aurc"]),
        ]
        for row in evaluation.quality_rows
    ]
    return "\n\n".join(
        [
            "AUROC is the chance that a wrong answer gets a higher risk than a right one. Brier and ECE "
            "(expected calibration error, 10 bins) measure how close the risk is to the observed error rate. "
            "AURC is the mean error rate over all coverage levels; lower is better.",
            _table(["Signal", "Stratum", "n", "Error rate", "AUROC", "Brier", "ECE", "AURC"], rows),
        ]
    )


def _sweep(results: RunResults) -> str:
    best = next(score for score in results.sweep if score.layer == results.layer)
    return "\n\n".join(
        [
            "A logistic probe on the last prompt token's residual stream (the running sum of layer outputs "
            "the model passes between layers) predicts whether the chosen answer is wrong. Grouped 5-fold "
            "cross-validation on discovery data scores every layer.",
            "![Layer sweep](figures/layer_sweep.png)",
            f"*The error probe reads best at layer {results.layer}, with grouped 5-fold AUROC "
            f"{best.auroc_mean:.3f} on discovery data (chance is 0.5).*",
        ]
    )


def _thresholds(results: RunResults) -> str:
    rows = []
    for signal, gate in results.gates.items():
        selection = gate.selection
        if selection.threshold is None:
            closest = min(selection.checks, key=lambda check: check.upper_bound)
            rows.append(
                [METHOD_LABELS[signal.value], "none", str(closest.accepted), str(closest.errors), _pct(closest.upper_bound)]
            )
            continue
        check = next(c for c in selection.checks if c.threshold == selection.threshold)
        rows.append(
            [METHOD_LABELS[signal.value], f"{check.threshold:.2f}", str(check.accepted), str(check.errors), _pct(check.upper_bound)]
        )
    n_units = next(iter(results.gates.values())).selection.n_units
    return "\n\n".join(
        [
            f"Each gate answers when its calibrated risk is at or below the chosen threshold. The certificate "
            f"uses {n_units} independent cal_gate questions (one per group). A threshold passes when the "
            f"one-sided Clopper-Pearson upper bound on its error rate, at level 1 - 0.05/20, is at most "
            f"{_pct(results.config.gate.target_risk)}. Among passing thresholds the one that answers the most wins. "
            f"When none passes, the row shows the tightest bound reached.",
            _table(["Gate", "Threshold", "Answered", "Wrong", "Upper bound"], rows),
        ]
    )


def _steering(steering: SteeringResult) -> str:
    by_dose: dict[float, dict[str, list[dict[str, Any]]]] = {}
    for row in steering.rows:
        kind = ERROR_DIRECTION if row["direction"] == ERROR_DIRECTION else "random"
        by_dose.setdefault(row["dose"], {}).setdefault(kind, []).append(row)

    def mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    table_rows = [
        [
            f"{dose:+.1f}",
            _num(mean_of(kinds[ERROR_DIRECTION], "mean_p_abstain")),
            _num(mean_of(kinds.get("random", []), "mean_p_abstain")),
            _pct(mean_of(kinds[ERROR_DIRECTION], "real_accuracy")),
            _pct(mean_of(kinds.get("random", []), "real_accuracy")),
        ]
        for dose, kinds in sorted(by_dose.items())
    ]
    verdict = "pass" if steering.identity_ok else "FAIL"
    return "\n\n".join(
        [
            f"The error direction is the mean residual at layer {steering.layer} for wrong answers minus the mean "
            f"for right answers, on discovery data (norm {steering.direction_norm:.1f}). It is added at every token "
            f"position of that layer on {steering.n_items} test prompts that offer option E. Dose 1 adds the full "
            f"difference of means. Random directions have the same norm.",
            f"Identity check ({verdict}): a zero-dose hook changed the answer log-probabilities by at most "
            f"{steering.identity_max_abs_diff:.2e}.",
            _table(
                ["Dose", "P(E), error direction", "P(E), random mean", "Real accuracy, error direction", "Real accuracy, random mean"],
                table_rows,
            ),
        ]
    )


def _examples(results: RunResults) -> str:
    questions = {q.item_id: q for q in results.dataset.questions}
    released_key = f"released_{PRIMARY_SIGNAL.value}"
    risk_key = f"risk_{PRIMARY_SIGNAL.value}"
    picks: list[tuple[str, dict[str, Any]]] = []
    buckets = [
        ("Invented entity, gate abstained", lambda r: r["stratum"] == "fictional" and not r[released_key]),
        ("Real question, answered correctly", lambda r: r["stratum"] == "real" and r[released_key] and not r["error"]),
        ("Real question, wrong candidate, gate abstained", lambda r: r["stratum"] == "real" and not r[released_key] and r["error"]),
        ("Real question, wrong answer released", lambda r: r["stratum"] == "real" and r[released_key] and r["error"]),
        ("Invented entity, answer released", lambda r: r["stratum"] == "fictional" and r[released_key]),
    ]
    for label, matches in buckets:
        found = [row for row in results.evaluation.prediction_rows if matches(row)][:2]
        picks += [(label, row) for row in found]
    if not picks:
        return "No examples."
    rows = []
    for label, row in picks:
        question = questions[row["item_id"]]
        text = question.question if len(question.question) <= 160 else question.question[:157] + "..."
        candidate = f"{row['candidate']}. {question.options['ABCD'.index(row['candidate'])]}"
        rows.append([label, text.replace("|", "/"), candidate, row["correct"] or "none", f"{row[risk_key]:.3f}"])
    return _table(["Case", "Question", "Model's candidate", "Correct", "Risk"], rows)


def _data(results: RunResults) -> str:
    counts = _counts(results.dataset)
    rows = []
    for role in Role:
        mask = results.dataset.mask(role)
        real = mask & (results.dataset.strata == Stratum.REAL.value)
        accuracy = float(1 - results.errors[real].mean()) if real.any() else None
        rows.append(
            [
                role.value,
                str(counts[role.value][Stratum.REAL.value]),
                str(counts[role.value][Stratum.FICTIONAL.value]),
                str(int((mask & results.dataset.canonical).sum())),
                _pct(accuracy),
            ]
        )
    return "\n\n".join(
        [
            "Roles are assigned by group, so a question and its variants never sit in two roles. An invented "
            "entity's three questions form one group.",
            _table(["Role", "MedQA", "Invented", "Independent units", "MedQA accuracy (no gate)"], rows),
        ]
    )
