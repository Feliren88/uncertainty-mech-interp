"""Orchestrates the design's stages: data, reads, layer sweep, gates, test, steering, report."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from uncertainty_mech.application.answer import GateBundle, save_bundle
from uncertainty_mech.application.config import RunConfig
from uncertainty_mech.application.dataset import prepare_dataset, read_questions
from uncertainty_mech.application.evaluation import evaluate_on_test
from uncertainty_mech.application.gates import PRIMARY_SIGNAL, fit_gate
from uncertainty_mech.application.ports import FigureWriter, LanguageModel, RunStore
from uncertainty_mech.application.report import write_report
from uncertainty_mech.application.results import RunResults
from uncertainty_mech.application.risk_model import SignalSet, build_features, sweep_layers
from uncertainty_mech.application.steering import error_direction, run_steering_check
from uncertainty_mech.domain.grading import chosen_indices, error_labels, output_features
from uncertainty_mech.domain.questions import HealthQuestion, Role

log = logging.getLogger(__name__)

_PACKAGES = ("numpy", "scipy", "scikit-learn", "matplotlib", "torch", "transformers")


def run_pipeline(
    config: RunConfig,
    questions: Sequence[HealthQuestion],
    model: LanguageModel,
    store: RunStore,
    figures: FigureWriter,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    store.write_json("config.json", config.to_dict())
    store.write_json("manifest.json", _manifest(config, model.name, started))

    dataset = prepare_dataset(questions, config)
    store.write_table("items.csv", dataset.item_rows())
    log.info("Dataset: %d items in %d groups", len(dataset.questions), len(set(dataset.groups)))

    log.info("Reading the model without option E, capturing %d layers", model.num_layers)
    readings = read_questions(model, dataset.questions, allow_abstain=False, capture_residuals=True)
    log.info("Reading the model with option E")
    with_abstain = read_questions(model, dataset.questions, allow_abstain=True)
    store.write_arrays(
        "readings.npz",
        letter_logprobs=readings.letter_logprobs,
        letter_mass=readings.letter_mass,
        residuals=readings.residuals,
        abstain_letter_logprobs=with_abstain.letter_logprobs,
    )

    chosen = chosen_indices(readings.letter_logprobs)
    errors = error_labels(dataset.questions, chosen)
    discovery = dataset.mask(Role.DISCOVERY)
    log.info("Layer sweep on %d discovery items", int(discovery.sum()))
    sweep = sweep_layers(
        readings.residuals[discovery],
        errors[discovery],
        dataset.groups[discovery],
        config.probe.sweep_c,
        config.probe.cv_folds,
    )
    layer = max(sweep, key=lambda score: score.auroc_mean).layer
    layer_residuals = readings.residuals[:, layer]
    log.info("Probe layer %d", layer)

    out_feats = output_features(readings.letter_logprobs, readings.letter_mass)
    features = {signal: build_features(signal, out_feats, layer_residuals) for signal in SignalSet}
    gates = {signal: fit_gate(signal, features[signal], errors, dataset, config) for signal in SignalSet}
    risks = {signal: gates[signal].risk_model.risk(features[signal]) for signal in SignalSet}
    primary = gates[PRIMARY_SIGNAL]
    save_bundle(
        store,
        GateBundle(
            run_id=config.run_id,
            model_name=model.name,
            signal=PRIMARY_SIGNAL,
            layer=layer,
            risk_model=primary.risk_model,
            threshold=primary.selection.threshold,
            target_risk=config.gate.target_risk,
        ),
    )
    log.info("Thresholds: %s", {signal.value: gate.selection.threshold for signal, gate in gates.items()})

    evaluation = evaluate_on_test(dataset, chosen, errors, risks, gates, with_abstain.letter_logprobs)
    log.info("Steering check at layer %d", layer)
    steering = run_steering_check(
        model, dataset, error_direction(layer_residuals[discovery], errors[discovery]), layer, config
    )

    results = RunResults(
        config=config,
        model_name=model.name,
        dataset=dataset,
        errors=errors,
        sweep=sweep,
        layer=layer,
        gates=gates,
        evaluation=evaluation,
        steering=steering,
        started=started,
        finished=datetime.now(UTC),
    )
    return write_report(store, figures, results)


def _manifest(config: RunConfig, model_name: str, started: datetime) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "model": model_name,
        "started_utc": started.isoformat(),
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in _PACKAGES},
    }


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
