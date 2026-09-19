"""End to end: `run`, then `circuits` on its output, on fakes with a planted abstention head."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.fakes import FakeCircuitModel, FakeLanguageModel, InMemoryRealSource
from tests.test_end_to_end import CONFIG as RUN_CONFIG
from uncertainty_mech.cli import main
from uncertainty_mech.infrastructure.known_facts import KNOWN_FACTS

CIRCUIT_CONFIG = """
run_id = "circuits"
seed = 23
output_root = "runs"
source_run = "runs/e2e"

[model]
name = "fake-model"
revision = "test"

[pairs]
names_per_fact = 1
discovery_share = 0.6

[circuit]
k_grid = [1, 2, 4]
sufficiency_threshold = 0.5
n_random = 3

[steering]
doses = [1.0, 2.0]
se_thresholds = [0.3, 0.6, 0.9, 1.2]

[tiers]
dose_grid = [1.0, 2.0]
tier_thresholds = [0.3, 0.6, 0.9]
max_tiers = 2
readout_quantiles = [0.5, 0.9]
wrong_costs = [1.0, 2.0]
"""

TIERS_CONFIG = """
run_id = "tiers"
seed = 29
output_root = "runs"
circuit_run = "runs/circuits"

[model]
name = "fake-model"
revision = "test"

[tiers]
dose_grid = [1.0, 2.0]
tier_thresholds = [0.3, 0.6, 0.9]
max_tiers = 2
readout_quantiles = [0.5, 0.9]
wrong_costs = [1.0, 2.0]
bootstrap_samples = 200
"""

EXPECTED_FILES = [
    "config.json",
    "pairs.csv",
    "residual_patching.csv",
    "head_patching.csv",
    "mlp_patching.csv",
    "circuit.json",
    "circuit_validation.csv",
    "steering_calibration.csv",
    "steering_test.csv",
    "steering_flips.csv",
    "steering_curves.csv",
    "steering_predictions.csv",
    "steering_vectors.npz",
    "tiers_choices.csv",
    "tiers_test.csv",
    "tiers_bands.csv",
    "tiers_top_schedules.csv",
    "tiers_comparisons.csv",
    "tiers_predictions.csv",
    "summary.json",
    "report.md",
    "figures/residual_patching.png",
    "figures/head_patching.png",
    "figures/se_steering.png",
    "figures/tiers_tradeoff.png",
]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("circuits")
    (root / "run.toml").write_text(RUN_CONFIG, encoding="utf-8")
    (root / "circuits.toml").write_text(CIRCUIT_CONFIG, encoding="utf-8")
    source = InMemoryRealSource(400, seed=17)
    gate_model = FakeLanguageModel(source.load())
    known = {q.question: q.answer_index for q in source.load() if gate_model.knows(q.question)}
    circuit_model = FakeCircuitModel(known, KNOWN_FACTS)
    assert (
        main(
            ["run", "--config", str(root / "run.toml")],
            model_factory=lambda _config: gate_model,
            real_source_factory=lambda _data, _seed: source,
        )
        == 0
    )
    assert (
        main(
            ["circuits", "--config", str(root / "circuits.toml")],
            circuit_model_factory=lambda _config: circuit_model,
            real_source_factory=lambda _data, _seed: source,
        )
        == 0
    )
    (root / "tiers.toml").write_text(TIERS_CONFIG, encoding="utf-8")
    assert (
        main(["tiers", "--config", str(root / "tiers.toml")], circuit_model_factory=lambda _config: circuit_model) == 0
    )
    return root / "runs" / "circuits"


def test_study_writes_every_artifact(study):
    assert [name for name in EXPECTED_FILES if not (study / name).is_file()] == []


def test_head_sweep_ranks_the_planted_head_first(study):
    circuit = json.loads((study / "circuit.json").read_text())
    assert circuit["heads"][0] == list(FakeCircuitModel.planted_head)


def test_circuit_beats_random_heads_on_held_out_pairs(study):
    rows = [row for row in _rows(study / "circuit_validation.csv") if row["direction"] == "invented_into_real"]
    circuit = float(next(row for row in rows if row["head_set"] == "circuit")["restoration_gap"])
    randoms = [float(row["restoration_gap"]) for row in rows if row["head_set"] != "circuit"]
    assert circuit > 0.5
    assert circuit > max(randoms)


def test_se_gated_steering_turns_uncertain_answers_into_i_dont_know(study):
    rows = {row["condition"]: row for row in _rows(study / "steering_test.csv")}
    gated, plain = rows["se_gated_circuit"], rows["prompt_only"]
    assert float(gated["answered_share"]) < float(plain["answered_share"])
    assert float(gated["net_correct"]) > float(plain["net_correct"])
    flips = {(row["steering"], row["group"]): row for row in _rows(study / "steering_flips.csv")}
    circuit_flips = float(flips[("circuit", "wrong before")]["flipped_to_e"])
    random_flips = float(flips[("random_heads", "wrong before")]["flipped_to_e"])
    assert circuit_flips > random_flips


def test_tiered_controllers_nest_on_calibration_and_beat_random_heads_on_test(study):
    choices = _rows(study / "tiers_choices.csv")
    for cost in {row["wrong_cost"] for row in choices}:
        utility = {row["family"]: float(row["calibration_utility"]) for row in choices if row["wrong_cost"] == cost}
        assert utility["circuit_one_threshold"] <= utility["circuit_tiered"] <= utility["circuit_tiered_readout"]
    for cost in {row["wrong_cost"] for row in choices}:
        test = {row["condition"]: row for row in _rows(study / "tiers_test.csv") if row["wrong_cost"] == cost}
        tiered = float(test["circuit_tiered_readout"]["abstain_on_unknown"])
        assert tiered > float(test["control_random_heads"]["abstain_on_unknown"])
        assert tiered >= float(test["option_e_only"]["abstain_on_unknown"])


def test_circuit_readout_separates_invented_from_real(study):
    summary = json.loads((study / "summary.json").read_text())
    assert summary["readout_auroc_invented"] > 0.9


def test_tiers_command_reuses_the_circuit_and_reports_bootstrap_intervals(study):
    tiers = study.parent / "tiers"
    for name in ("report.md", "summary.json", "tiers_test.csv", "tiers_comparisons.csv", "figures/tiers_tradeoff.png"):
        assert (tiers / name).is_file(), name
    comparisons = _rows(tiers / "tiers_comparisons.csv")
    assert comparisons
    for row in comparisons:
        assert float(row["low_95"]) <= float(row["difference"]) <= float(row["high_95"])
    predictions = _rows(tiers / "tiers_predictions.csv")
    assert len(predictions) == len(_rows(study.parent / "e2e" / "items.csv")) - sum(
        row["role"] != "test" for row in _rows(study.parent / "e2e" / "items.csv")
    )
