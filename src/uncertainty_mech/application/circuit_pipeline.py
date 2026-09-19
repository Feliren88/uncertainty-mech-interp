"""Orchestrates the circuit study: pairs, patching sweeps, circuit choice and test, SE steering, report."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from uncertainty_mech.application.circuit_config import CircuitStudyConfig
from uncertainty_mech.application.circuit_report import CircuitResults, PairSummary, write_circuit_report
from uncertainty_mech.application.dataset import dataset_from_rows
from uncertainty_mech.application.entity_pairs import EntityPair, split_pairs
from uncertainty_mech.application.patching import (
    align_pairs,
    capture_heads,
    choose_circuit,
    keep_known_pairs,
    pair_baseline,
    rank_heads,
    steering_vectors,
    sweep_heads,
    sweep_mlps,
    sweep_residual,
    validate_circuit,
)
from uncertainty_mech.application.ports import CircuitFigureWriter, CircuitModel, RunStore
from uncertainty_mech.application.se_steering import (
    EvalSet,
    SteeringVectors,
    calibrate,
    evaluate,
    random_direction_vectors,
    random_head_vectors,
)
from uncertainty_mech.application.tiered_steering import run_tiered_study
from uncertainty_mech.domain.grading import semantic_entropy
from uncertainty_mech.domain.questions import Role

log = logging.getLogger(__name__)


def run_circuit_study(
    config: CircuitStudyConfig,
    pairs: Sequence[EntityPair],
    model: CircuitModel,
    source: RunStore,
    store: RunStore,
    figures: CircuitFigureWriter,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    source_model = source.read_json("bundle.json")["model_name"]
    if source_model != model.name:
        raise ValueError(f"the source run used {source_model}, not {model.name}")
    store.write_json("config.json", config.to_dict())
    rng = np.random.default_rng(config.seed)

    discovery_raw, test_raw = split_pairs(pairs, config.pairs.discovery_share, config.seed)
    discovery = align_pairs(model, keep_known_pairs(model, discovery_raw))
    test = align_pairs(model, keep_known_pairs(model, test_raw))
    log.info(
        "Known pairs: %d of %d discovery, %d of %d test", len(discovery), len(discovery_raw), len(test), len(test_raw)
    )

    baseline = pair_baseline(model, discovery)
    log.info("Residual sweep")
    residual_rows = sweep_residual(model, discovery, baseline)
    invented_z = capture_heads(model, discovery.invented)
    real_z = capture_heads(model, discovery.real)
    log.info("Head sweep over %d heads", model.num_layers * model.num_heads)
    head_rows = sweep_heads(model, discovery, baseline, invented_z)
    mlp_rows = sweep_mlps(model, discovery, baseline)
    choice = choose_circuit(model, discovery, baseline, invented_z, rank_heads(head_rows), config.circuit)
    log.info("Circuit: %d heads %s", len(choice.heads), choice.heads)
    test_baseline = pair_baseline(model, test)
    validation_rows = validate_circuit(model, test, test_baseline, choice.heads, config.circuit.n_random, rng)

    circuit = SteeringVectors(choice.heads, steering_vectors(invented_z, real_z, choice.heads))
    random_heads = random_head_vectors(invented_z, real_z, choice.heads, rng)
    random_vectors = random_direction_vectors(circuit, rng)
    store.write_arrays(
        "steering_vectors.npz",
        circuit_heads=np.array(circuit.heads),
        circuit_vectors=circuit.vectors,
        random_heads=np.array(random_heads.heads),
        random_head_vectors=random_heads.vectors,
        random_vectors=random_vectors.vectors,
    )
    dataset = dataset_from_rows(source.read_table("items.csv"))
    forced = source.read_arrays("readings.npz", keys=("letter_logprobs",))["letter_logprobs"]
    se = semantic_entropy(forced)
    calibration_items = EvalSet.from_role(dataset, Role.CAL_PROB, forced, se)
    test_items = EvalSet.from_role(dataset, Role.TEST, forced, se)
    log.info("Calibrating SE steering on %d items", len(calibration_items.prompts))
    points, calibration_rows = calibrate(model, calibration_items, circuit, config.steering)
    log.info("Operating points: %s", points)
    evaluation = evaluate(
        model, test_items, circuit, random_heads, random_vectors, points, config.steering.se_thresholds
    )
    log.info("Tiered steering over wrong-answer costs %s", config.tiers.wrong_costs)
    tiered = run_tiered_study(
        model, calibration_items, test_items, circuit, random_heads, config.tiers, config.steering.se_thresholds
    )

    results = CircuitResults(
        config=config,
        model_name=model.name,
        pairs=PairSummary.of(discovery, baseline, test_baseline, len(discovery_raw), len(test_raw)),
        tail_length=discovery.tail_length,
        residual_rows=residual_rows,
        head_rows=head_rows,
        mlp_rows=mlp_rows,
        circuit=choice,
        validation_rows=validation_rows,
        vector_norms=np.linalg.norm(circuit.vectors, axis=1),
        random_heads=random_heads.heads,
        points=points,
        calibration_rows=calibration_rows,
        evaluation=evaluation,
        tiered=tiered,
        started=started,
        finished=datetime.now(UTC),
    )
    return write_circuit_report(store, figures, results)
