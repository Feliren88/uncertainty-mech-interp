"""Stage 4 on its own: tiered semantic-entropy steering from a finished circuit study's vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from uncertainty_mech.application.circuit_config import TiersStudyConfig
from uncertainty_mech.application.patching import Head
from uncertainty_mech.application.ports import CircuitFigureWriter, CircuitModel, RunStore
from uncertainty_mech.application.se_steering import SteeringVectors, ensure_same_model, load_eval_sets
from uncertainty_mech.application.tiered_steering import run_tiered_study
from uncertainty_mech.application.tiers_report import choice_row, draw_tier_figure, tier_markdown, write_tier_tables


def run_tiers_study(
    config: TiersStudyConfig,
    model: CircuitModel,
    circuit_store: RunStore,
    source: RunStore,
    store: RunStore,
    figures: CircuitFigureWriter,
) -> dict[str, Any]:
    ensure_same_model(source, model)
    store.write_json("config.json", config.to_dict())
    saved = circuit_store.read_arrays("steering_vectors.npz")
    circuit = SteeringVectors(_heads(saved["circuit_heads"]), saved["circuit_vectors"])
    random_heads = SteeringVectors(_heads(saved["random_heads"]), saved["random_head_vectors"])
    calibration, test = load_eval_sets(source)
    tiered = run_tiered_study(
        model,
        calibration,
        test,
        circuit,
        random_heads,
        config.tiers,
        config.wrapper_thresholds,
        np.random.default_rng(config.seed),
    )
    write_tier_tables(store, tiered)
    draw_tier_figure(figures, tiered, store.root / "figures" / "tiers_tradeoff.png")
    heads = ", ".join(f"L{layer}.H{head}" for layer, head in circuit.heads)
    summary = {
        "run_id": config.run_id,
        "model": model.name,
        "circuit_heads": [list(head) for head in circuit.heads],
        "readout_auroc_invented": tiered.readout_auroc_invented,
        "readout_auroc_unknown_real": tiered.readout_auroc_unknown_real,
        "choices": [choice_row(choice) for choice in tiered.choices],
        "test": tiered.test_rows,
        "comparisons": tiered.comparison_rows,
    }
    store.write_json("summary.json", summary)
    store.write_text(
        "report.md",
        f"# Tiered semantic-entropy steering: {config.run_id}\n\n"
        f"Circuit heads {heads}, from `{config.circuit_run.name}`; questions and semantic entropy from "
        f"`{source.root.name}`.\n\n{tier_markdown(tiered)}\n",
    )
    return summary


def _heads(array: np.ndarray) -> tuple[Head, ...]:
    return tuple((int(layer), int(head)) for layer, head in array)
