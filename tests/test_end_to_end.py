"""End to end: `run` then `ask` through the CLI composition root, on a fake model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.fakes import FakeLanguageModel, InMemoryRealSource
from uncertainty_mech.cli import main

CONFIG = """
run_id = "e2e"
seed = 17
output_root = "runs"

[data]
real_repo = "in-memory"
real_revision = "none"
real_file = "none"
n_real = 400
n_fictional_entities = 40

[data.proportions]
discovery = 0.5
cal_prob = 0.15
cal_gate = 0.2
test = 0.15

[model]
name = "fake-model"
revision = "test"
batch_size = 8

[gate]
target_risk = 0.2

[steering]
n_items = 30
n_random = 2
"""

EXPECTED_FILES = [
    "config.json",
    "manifest.json",
    "items.csv",
    "readings.npz",
    "layer_sweep.csv",
    "c_selection.csv",
    "thresholds.csv",
    "metrics.csv",
    "risk_quality.csv",
    "matched_coverage.csv",
    "test_predictions.csv",
    "steering.csv",
    "bundle.json",
    "bundle_risk_model.npz",
    "summary.json",
    "report.md",
    "figures/layer_sweep.png",
    "figures/risk_coverage.png",
]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    config_path = root / "config.toml"
    config_path.write_text(CONFIG, encoding="utf-8")
    source = InMemoryRealSource(400)
    model = FakeLanguageModel(source.load())
    exit_code = main(
        ["run", "--config", str(config_path)],
        model_factory=lambda _config: model,
        real_source_factory=lambda _data, _seed: source,
    )
    assert exit_code == 0
    return root / "runs" / "e2e", model


def test_run_writes_every_artifact(completed_run):
    run_dir, _ = completed_run
    assert [name for name in EXPECTED_FILES if not (run_dir / name).is_file()] == []


def test_groups_never_cross_roles(completed_run):
    run_dir, _ = completed_run
    roles_by_group: dict[str, set[str]] = {}
    for row in _rows(run_dir / "items.csv"):
        roles_by_group.setdefault(row["group_id"], set()).add(row["role"])
    assert all(len(roles) == 1 for roles in roles_by_group.values())
    assert set().union(*roles_by_group.values()) == {"discovery", "cal_prob", "cal_gate", "test"}


def test_selected_threshold_obeys_the_certificate(completed_run):
    run_dir, _ = completed_run
    bundle = json.loads((run_dir / "bundle.json").read_text())
    assert bundle["threshold"] is not None
    checks = [row for row in _rows(run_dir / "thresholds.csv") if row["signal"] == bundle["signal"]]
    passing = [row for row in checks if float(row["upper_bound"]) <= bundle["target_risk"]]
    best = max(passing, key=lambda row: (float(row["coverage"]), -float(row["threshold"])))
    assert float(best["threshold"]) == bundle["threshold"]


def test_zero_dose_steering_hook_matches_the_plain_pass(completed_run):
    run_dir, _ = completed_run
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["steering"]["identity_ok"] is True


def test_ask_answers_known_questions_and_abstains_on_invented_ones(completed_run, capsys):
    run_dir, model = completed_run
    test_rows = [row for row in _rows(run_dir / "items.csv") if row["role"] == "test"]
    known = next(row for row in test_rows if row["stratum"] == "real" and model.knows(row["question"]))
    invented = next(row for row in test_rows if row["stratum"] == "fictional")
    questions_path = run_dir.parent / "ask.jsonl"
    questions_path.write_text(
        "\n".join(
            json.dumps({"question": row["question"], "options": json.loads(row["options_json"])})
            for row in (known, invented)
        ),
        encoding="utf-8",
    )
    capsys.readouterr()

    exit_code = main(
        ["ask", "--run-dir", str(run_dir), "--questions", str(questions_path)],
        model_factory=lambda _config: model,
    )

    assert exit_code == 0
    answers = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    answer_index = int(known["answer_index"])
    expected = f"{'ABCD'[answer_index]}. {json.loads(known['options_json'])[answer_index]}"
    assert (answers[0]["text"], answers[0]["reason_code"]) == (expected, "ANSWERED")
    assert (answers[1]["text"], answers[1]["reason_code"]) == ("I don't know.", "HIGH_RISK")
    assert len((run_dir / "audit.jsonl").read_text().splitlines()) == 2
