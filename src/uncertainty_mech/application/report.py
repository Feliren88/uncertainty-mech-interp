"""Stage 9: CSV tables, the JSON summary, two figures and the markdown report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from uncertainty_mech.application.gates import PRIMARY_SIGNAL
from uncertainty_mech.application.ports import FigureWriter, RunStore
from uncertainty_mech.application.report_markdown import render_markdown
from uncertainty_mech.application.results import RunResults
from uncertainty_mech.domain.questions import Stratum


def write_report(store: RunStore, figures: FigureWriter, results: RunResults) -> dict[str, Any]:
    _write_tables(store, results)
    figures.layer_sweep(
        [asdict(score) for score in results.sweep], results.layer, store.root / "figures" / "layer_sweep.png"
    )
    figures.risk_coverage(
        results.evaluation.curves,
        results.evaluation.prompt_only_point,
        results.config.gate.target_risk,
        store.root / "figures" / "risk_coverage.png",
    )
    summary = _summary(results)
    store.write_json("summary.json", summary)
    store.write_text("report.md", render_markdown(results))
    return summary


def _write_tables(store: RunStore, results: RunResults) -> None:
    store.write_table("layer_sweep.csv", [asdict(score) for score in results.sweep])
    store.write_table(
        "c_selection.csv",
        [
            {"signal": signal.value, **asdict(score)}
            for signal, gate in results.gates.items()
            for score in gate.c_scores
        ],
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


def _summary(results: RunResults) -> dict[str, Any]:
    real = results.dataset.strata == Stratum.REAL.value
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
        "counts": results.counts(),
        "real_accuracy_no_gate": float(1 - results.errors[real].mean()) if real.any() else None,
        "probe_layer": results.layer,
        "probe_layer_auroc": results.best_layer_score.auroc_mean,
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
