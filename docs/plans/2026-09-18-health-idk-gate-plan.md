# Health "I don't know" gate implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test-run an activation-probe gate that answers health questions with Llama 3.1 8B Instruct or returns exactly `I don't know.`

**Architecture:** Clean architecture in `src/uncertainty_mech`. `domain` holds pure policy and math, `application` holds use cases behind ports, `infrastructure` holds the Hugging Face, MedQA, file and figure adapters, and `cli.py` wires them. One end-to-end pytest drives the CLI with a fake model; the GPU test run drives the same CLI with Llama.

**Tech Stack:** Python 3.13, NumPy, SciPy, scikit-learn, matplotlib, PyTorch 2.12, transformers 4.57, huggingface_hub, pytest.

**Spec:** `docs/specs/2026-09-18-health-idk-gate-design.md`

## Global constraints

- Tests are end to end only (user instruction). No unit-test files.
- `domain` imports only the standard library, NumPy and SciPy. `torch` and `transformers` appear only in `infrastructure/hf_model.py`.
- Model: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Data: `GBaker/MedQA-USMLE-4-options`, revision `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`, file `phrases_no_exclude_train.jsonl`.
- Seed 17. Roles 60/10/10/20 (discovery, cal_prob, cal_gate, test) for the test run.
- Target selective risk 0.10, delta 0.05, the protocol's fixed 20-threshold grid.
- Abstention text is exactly `I don't know.`
- Prose (README, report): no em dashes, sentence-case headings, define jargon, no AI-tell vocabulary.
- Commits carry no AI-assistant attribution and no co-author trailer.
- Run everything from `code/` with `HF_HOME=/home/vfeliren1/lf93_scratch2/vfvic1/hf_cache/huggingface`.

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, pytest paths |
| `src/uncertainty_mech/domain/questions.py` | `HealthQuestion`, `Stratum`, `Role`, letters |
| `src/uncertainty_mech/domain/splits.py` | Seeded group-to-role assignment, canonical units |
| `src/uncertainty_mech/domain/prompts.py` | `ChatPrompt`, prompt text with and without option E |
| `src/uncertainty_mech/domain/grading.py` | Chosen letter, error labels, output features |
| `src/uncertainty_mech/domain/risk_control.py` | Clopper-Pearson bound, threshold selection |
| `src/uncertainty_mech/domain/metrics.py` | Selective metrics, AUROC, ECE, AURC |
| `src/uncertainty_mech/domain/gate.py` | `decide`, `render`, reason codes |
| `src/uncertainty_mech/application/ports.py` | `LanguageModel`, `RunStore`, `FigureWriter`, sources, `Readings`, `Steering` |
| `src/uncertainty_mech/application/config.py` | TOML run config |
| `src/uncertainty_mech/application/risk_model.py` | Signal sets, logistic probe, layer sweep, temperature |
| `src/uncertainty_mech/application/dataset.py` | `Dataset`, role assignment, model reads |
| `src/uncertainty_mech/application/gates.py` | `FittedGate`, fit, calibrate, certify |
| `src/uncertainty_mech/application/evaluation.py` | Final test tables |
| `src/uncertainty_mech/application/steering.py` | Exploratory steering check |
| `src/uncertainty_mech/application/answer.py` | Bundle persistence, `AbstainingAnswerer` |
| `src/uncertainty_mech/application/report.py` | CSV tables, figures, `report.md` |
| `src/uncertainty_mech/application/pipeline.py` | Orchestrates stages 1 to 9 |
| `src/uncertainty_mech/infrastructure/store.py` | `FileRunStore` |
| `src/uncertainty_mech/infrastructure/fictional.py` | Invented drug and disease items |
| `src/uncertainty_mech/infrastructure/figures.py` | House-style matplotlib figures |
| `src/uncertainty_mech/infrastructure/medqa.py` | MedQA source |
| `src/uncertainty_mech/infrastructure/hf_model.py` | transformers adapter with hooks |
| `src/uncertainty_mech/cli.py` | Composition root, `run` and `ask` |
| `tests/fakes.py`, `tests/test_end_to_end.py` | Fake model, fake MedQA, end-to-end test |
| `configs/*.toml`, `examples/health_questions.jsonl`, `scripts/test_run.sh` | Test-run inputs |

Each code step below shows a file as `**File:**` followed by its complete content.

---

### Task 1: Scaffold and the failing end-to-end test

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/uncertainty_mech/__init__.py`, `src/uncertainty_mech/__main__.py`, `tests/__init__.py`, `tests/fakes.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Produces: the contract every later task must satisfy. `uncertainty_mech.cli.main(argv, *, model_factory, real_source_factory) -> int`; run directory files listed in `EXPECTED_FILES`; `FakeLanguageModel.read(prompts, letters, *, capture_residuals, steering) -> Readings`.

- [ ] **Step 1: Write the scaffold**

**File:** `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "uncertainty-mech"
version = "0.1.0"
description = "Activation-probe gate that answers health questions or says I don't know."
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "scipy>=1.11",
  "scikit-learn>=1.4",
  "matplotlib>=3.8",
]

[project.optional-dependencies]
model = ["torch>=2.2", "transformers>=4.56", "accelerate>=0.30", "huggingface_hub>=0.24"]
test = ["pytest>=8"]

[project.scripts]
uncertainty-mech = "uncertainty_mech.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
```

**File:** `.gitignore`
```text
__pycache__/
*.pyc
.pytest_cache/
runs/
*.egg-info/
```

**File:** `src/uncertainty_mech/__init__.py`
```python
"""Activation-probe gate that answers health questions or says "I don't know."."""
```

**File:** `src/uncertainty_mech/__main__.py`
```python
from uncertainty_mech.cli import main

raise SystemExit(main())
```

**File:** `tests/__init__.py`
```python
```

- [ ] **Step 2: Write the test doubles**

**File:** `tests/fakes.py`
```python
"""Stand-ins for the Hugging Face adapter and MedQA, so the pipeline runs on CPU in seconds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import log_softmax

from uncertainty_mech.application.ports import Readings, Steering
from uncertainty_mech.domain.prompts import ChatPrompt
from uncertainty_mech.domain.questions import OPTION_LETTERS, HealthQuestion, Stratum
from uncertainty_mech.domain.splits import stable_unit


class InMemoryRealSource:
    """Synthetic "real" questions with known answers."""

    def __init__(self, n_items: int) -> None:
        self._questions = [
            HealthQuestion(
                item_id=f"real-{i:04d}",
                group_id=f"real-{i:04d}",
                stratum=Stratum.REAL,
                question=f"Case {i}: a patient presents with finding {i}. What is the best next step?",
                options=tuple(f"Step {i}-{letter}" for letter in OPTION_LETTERS),
                answer_index=i % len(OPTION_LETTERS),
            )
            for i in range(n_items)
        ]

    def load(self) -> list[HealthQuestion]:
        return list(self._questions)

    def corpus_words(self) -> set[str]:
        return {"case", "patient", "finding", "step"}


class FakeLanguageModel:
    """Knows a fixed share of the real questions and nothing about invented entities.

    A known question gets a confident correct letter and a residual stream pushed
    along dimension 0 from layer 2 onward, so a probe has a real signal to find.
    """

    name = "fake-model@test"
    num_layers = 4
    d_model = 16

    def __init__(self, real_questions: Sequence[HealthQuestion], known_share: float = 0.7) -> None:
        self._answers = {
            q.question: q.answer_index for q in real_questions if stable_unit(q.item_id, 99) < known_share
        }

    def knows(self, question: str) -> bool:
        return question in self._answers

    def read(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        capture_residuals: bool = False,
        steering: Steering | None = None,
    ) -> Readings:
        rows = [self._read_one(prompt, len(letters), steering) for prompt in prompts]
        logits = np.stack([logit for logit, _, _ in rows])
        return Readings(
            letter_logprobs=log_softmax(logits, axis=1).astype(np.float32),
            letter_mass=np.array([mass for _, mass, _ in rows], dtype=np.float32),
            residuals=np.stack([residual for _, _, residual in rows]) if capture_residuals else None,
        )

    def _read_one(self, prompt: ChatPrompt, n_letters: int, steering: Steering | None):
        question = prompt.user.split("\n\n", 1)[0]
        known = question in self._answers
        rng = np.random.default_rng(int(stable_unit(question, 7) * 2**32))
        logits = rng.normal(0.0, 1.0, n_letters)
        residual = rng.normal(0.0, 1.0, (self.num_layers, self.d_model)).astype(np.float32)
        if known:
            logits[self._answers[question]] += 4.0
        residual[2:, 0] += 3.0 if known else -3.0
        if n_letters > len(OPTION_LETTERS):
            abstain = len(OPTION_LETTERS)
            logits[abstain] = -2.0 if known else 2.0
            if steering is not None:
                logits[abstain] += steering.scale * float(steering.direction[0])
        return logits, (0.95 if known else 0.6), residual
```

- [ ] **Step 3: Write the end-to-end test**

**File:** `tests/test_end_to_end.py`
```python
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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `python -m pytest -q`
Expected: collection error, `ModuleNotFoundError: No module named 'uncertainty_mech.application'`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "Add scaffold and failing end-to-end test"
```

---

### Task 2: Domain layer

**Files:**
- Create: `src/uncertainty_mech/domain/{__init__,questions,splits,prompts,grading,risk_control,metrics,gate}.py`

**Interfaces:**
- Produces: `HealthQuestion(item_id, group_id, stratum, question, options, answer_index)`, `Stratum`, `Role`, `OPTION_LETTERS`, `ABSTAIN_LETTER`; `stable_unit(key, seed) -> float`, `assign_roles(questions, proportions, seed) -> list[Role]`, `canonical_mask(questions, seed) -> list[bool]`; `ChatPrompt(system, user)`, `build_prompt(question, options, *, allow_abstain) -> ChatPrompt`, `answer_letters(allow_abstain) -> tuple[str, ...]`; `chosen_indices`, `error_labels`, `output_features`, `OUTPUT_FEATURE_NAMES`; `THRESHOLDS`, `clopper_pearson_upper`, `ThresholdCheck`, `ThresholdSelection`, `select_threshold(risk, errors, thresholds, target_risk, delta)`; `SelectiveMetrics`, `selective_metrics(errors, released, n_originally_correct)`, `RiskQuality`, `risk_quality`, `auroc`, `risk_coverage_curve`; `Decision`, `ReasonCode`, `GateResult`, `decide(risk, threshold)`, `render(result, letter, option_text)`, `ABSTAIN_TEXT`.

- [ ] **Step 1: Write the domain modules**

**File:** `src/uncertainty_mech/domain/__init__.py`
```python
"""Pure policy and math. No I/O, no model code."""
```

**File:** `src/uncertainty_mech/domain/questions.py`
```python
"""Health questions, their strata and their data roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

OPTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
ABSTAIN_LETTER = "E"


class Stratum(StrEnum):
    REAL = "real"
    FICTIONAL = "fictional"


class Role(StrEnum):
    DISCOVERY = "discovery"
    CAL_PROB = "cal_prob"
    CAL_GATE = "cal_gate"
    TEST = "test"


@dataclass(frozen=True)
class HealthQuestion:
    """One multiple-choice item. `answer_index` is None when no option is supported."""

    item_id: str
    group_id: str
    stratum: Stratum
    question: str
    options: tuple[str, ...]
    answer_index: int | None

    def __post_init__(self) -> None:
        if len(self.options) != len(OPTION_LETTERS):
            raise ValueError(f"{self.item_id}: expected {len(OPTION_LETTERS)} options, got {len(self.options)}")
        if (self.answer_index is not None) != (self.stratum is Stratum.REAL):
            raise ValueError(f"{self.item_id}: real items need an answer and fictional items must have none")
```

**File:** `src/uncertainty_mech/domain/splits.py`
```python
"""Seeded assignment of question groups to data roles."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from uncertainty_mech.domain.questions import HealthQuestion, Role


def stable_unit(key: str, seed: int) -> float:
    """Map a key to [0, 1) by hashing, so assignments never depend on row order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_roles(questions: Sequence[HealthQuestion], proportions: Mapping[Role, float], seed: int) -> list[Role]:
    """Give every item the role of its group, so related items never straddle roles."""
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError(f"role proportions must sum to 1, got {dict(proportions)}")
    cut_points: list[tuple[float, Role]] = []
    total = 0.0
    for role in Role:
        total += proportions.get(role, 0.0)
        cut_points.append((total, role))

    def role_of(group_id: str) -> Role:
        position = stable_unit(group_id, seed)
        return next((role for cut, role in cut_points if position < cut), cut_points[-1][1])

    return [role_of(question.group_id) for question in questions]


def canonical_mask(questions: Sequence[HealthQuestion], seed: int) -> list[bool]:
    """Pick one item per group for the binomial certificate, before any model output exists."""
    best: dict[str, tuple[float, str]] = {}
    for question in questions:
        key = (stable_unit(question.item_id, seed + 1), question.item_id)
        if question.group_id not in best or key < best[question.group_id]:
            best[question.group_id] = key
    chosen = {item_id for _, item_id in best.values()}
    return [question.item_id in chosen for question in questions]
```

**File:** `src/uncertainty_mech/domain/prompts.py`
```python
"""Prompt text for the answer readout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uncertainty_mech.domain.questions import ABSTAIN_LETTER, OPTION_LETTERS

SYSTEM_PROMPT = "You are a medical expert. Answer the multiple-choice question."
ABSTAIN_OPTION = "I don't know"


@dataclass(frozen=True)
class ChatPrompt:
    system: str
    user: str


def answer_letters(allow_abstain: bool) -> tuple[str, ...]:
    return OPTION_LETTERS + (ABSTAIN_LETTER,) if allow_abstain else OPTION_LETTERS


def build_prompt(question: str, options: Sequence[str], *, allow_abstain: bool) -> ChatPrompt:
    """Question, lettered options, and an instruction to reply with one letter.

    With `allow_abstain`, option E is "I don't know". That variant is the
    prompt-only baseline and the steering read-out.
    """
    if len(options) != len(OPTION_LETTERS):
        raise ValueError(f"expected {len(OPTION_LETTERS)} options, got {len(options)}")
    letters = answer_letters(allow_abstain)
    choices = [*options, ABSTAIN_OPTION] if allow_abstain else list(options)
    lines = [question.strip(), ""]
    lines += [f"{letter}. {text}" for letter, text in zip(letters, choices, strict=True)]
    lines.append("")
    if allow_abstain:
        lines.append(f"If you do not know the answer, choose {ABSTAIN_LETTER}.")
    lines.append(f"Reply with the letter only ({', '.join(letters)}).")
    return ChatPrompt(system=SYSTEM_PROMPT, user="\n".join(lines))
```

**File:** `src/uncertainty_mech/domain/grading.py`
```python
"""Error labels and output-only features from the answer-letter distribution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from uncertainty_mech.domain.questions import HealthQuestion

OUTPUT_FEATURE_NAMES = ("top_prob", "top_margin", "letter_entropy", "log_letter_mass")


def chosen_indices(letter_logprobs: np.ndarray) -> np.ndarray:
    """Index of the most likely offered letter in each row."""
    return np.argmax(letter_logprobs, axis=1)


def error_labels(questions: Sequence[HealthQuestion], chosen: np.ndarray) -> np.ndarray:
    """1 when the chosen option is wrong, or when no option is supported at all."""
    return np.array(
        [1 if q.answer_index is None else int(c != q.answer_index) for q, c in zip(questions, chosen, strict=True)],
        dtype=np.int64,
    )


def output_features(letter_logprobs: np.ndarray, letter_mass: np.ndarray) -> np.ndarray:
    """Top probability, top-two margin, entropy over the letters, and log letter mass.

    For multiple choice the answer meanings are the letters, so this entropy is
    the semantic entropy of protocol section 4 without sampling.
    """
    probs = np.exp(letter_logprobs)
    ranked = -np.sort(-probs, axis=1)
    entropy = -np.sum(np.where(probs > 0, probs * letter_logprobs, 0.0), axis=1)
    log_mass = np.log(np.clip(letter_mass, 1e-12, None))
    return np.column_stack([ranked[:, 0], ranked[:, 0] - ranked[:, 1], entropy, log_mass]).astype(np.float32)
```

**File:** `src/uncertainty_mech/domain/risk_control.py`
```python
"""Selective-risk certificate from protocol section 5: Learn then Test with Clopper-Pearson bounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

# Fixed before calibration, copied from the protocol.
THRESHOLDS: tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00,
)


def clopper_pearson_upper(errors: int, n: int, alpha: float) -> float:
    """One-sided upper confidence bound on an error rate, at level 1 - alpha."""
    if not 0 <= errors <= n or not 0 < alpha < 1:
        raise ValueError(f"invalid counts or level: errors={errors}, n={n}, alpha={alpha}")
    if n == 0 or errors == n:
        return 1.0
    return float(beta.ppf(1 - alpha, errors + 1, n - errors))


@dataclass(frozen=True)
class ThresholdCheck:
    threshold: float
    accepted: int
    errors: int
    coverage: float
    upper_bound: float


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float | None
    target_risk: float
    delta: float
    n_units: int
    checks: tuple[ThresholdCheck, ...]


def select_threshold(
    risk: np.ndarray, errors: np.ndarray, thresholds: Sequence[float], target_risk: float, delta: float
) -> ThresholdSelection:
    """Highest-coverage threshold whose simultaneous upper bound meets the target.

    Bonferroni splits delta across the fixed grid, and ties go to the smaller
    threshold. A threshold of None means nothing passed: abstain on everything.
    """
    risk = np.asarray(risk, dtype=float)
    errors = np.asarray(errors, dtype=int)
    if len(risk) == 0:
        raise ValueError("no calibration units")
    alpha = delta / len(thresholds)
    checks = []
    for tau in thresholds:
        accepted = risk <= tau
        n = int(accepted.sum())
        k = int(errors[accepted].sum())
        checks.append(ThresholdCheck(tau, n, k, n / len(risk), clopper_pearson_upper(k, n, alpha)))
    passing = [check for check in checks if check.upper_bound <= target_risk]
    best = max(passing, key=lambda check: (check.coverage, -check.threshold), default=None)
    return ThresholdSelection(best.threshold if best else None, target_risk, delta, len(risk), tuple(checks))
```

**File:** `src/uncertainty_mech/domain/metrics.py`
```python
"""Selective-answer and risk-quality metrics from protocol section 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from uncertainty_mech.domain.risk_control import clopper_pearson_upper


@dataclass(frozen=True)
class SelectiveMetrics:
    n: int
    released: int
    coverage: float
    selective_risk: float | None
    risk_upper_95: float
    false_answer_rate: float
    correct_retention: float | None


def selective_metrics(errors: np.ndarray, released: np.ndarray, n_originally_correct: int) -> SelectiveMetrics:
    """Coverage and error rate of the answers a user actually receives."""
    errors = np.asarray(errors, dtype=bool)
    released = np.asarray(released, dtype=bool)
    n = len(errors)
    if n == 0:
        raise ValueError("no items")
    n_released = int(released.sum())
    wrong = int((errors & released).sum())
    return SelectiveMetrics(
        n=n,
        released=n_released,
        coverage=n_released / n,
        selective_risk=wrong / n_released if n_released else None,
        risk_upper_95=clopper_pearson_upper(wrong, n_released, 0.05),
        false_answer_rate=wrong / n,
        correct_retention=(n_released - wrong) / n_originally_correct if n_originally_correct else None,
    )


@dataclass(frozen=True)
class RiskQuality:
    n: int
    error_rate: float
    auroc: float | None
    brier: float
    ece: float
    aurc: float


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Chance that a random error outscores a random non-error. None when one class is missing."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(risk: np.ndarray, errors: np.ndarray, bins: int = 10) -> float:
    """Count-weighted gap between mean predicted and observed error in equal-width bins."""
    which = np.minimum((risk * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        in_bin = which == b
        if in_bin.any():
            total += in_bin.mean() * abs(risk[in_bin].mean() - errors[in_bin].mean())
    return float(total)


def risk_coverage_curve(risk: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk when answering the k lowest-risk items, for k = 1..n."""
    order = np.argsort(np.asarray(risk, dtype=float), kind="stable")
    cumulative = np.cumsum(np.asarray(errors, dtype=float)[order])
    k = np.arange(1, len(order) + 1)
    return k / len(order), cumulative / k


def risk_quality(risk: np.ndarray, errors: np.ndarray, bins: int = 10) -> RiskQuality:
    risk = np.asarray(risk, dtype=float)
    errors = np.asarray(errors, dtype=int)
    _, curve = risk_coverage_curve(risk, errors)
    return RiskQuality(
        n=len(risk),
        error_rate=float(errors.mean()),
        auroc=auroc(risk, errors),
        brier=float(np.mean((risk - errors) ** 2)),
        ece=expected_calibration_error(risk, errors, bins),
        aurc=float(curve.mean()),
    )
```

**File:** `src/uncertainty_mech/domain/gate.py`
```python
"""Release decision and the only text a user sees (protocol section 5)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

ABSTAIN_TEXT = "I don't know."


class Decision(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


class ReasonCode(StrEnum):
    ANSWERED = "ANSWERED"
    HIGH_RISK = "HIGH_RISK"
    NO_CERTIFIED_THRESHOLD = "NO_CERTIFIED_THRESHOLD"
    INVALID_SCORE = "INVALID_SCORE"


@dataclass(frozen=True)
class GateResult:
    decision: Decision
    reason_code: ReasonCode


def decide(risk: float | None, threshold: float | None) -> GateResult:
    """Answer only with a certified threshold and a finite risk at or below it."""
    if threshold is None:
        return GateResult(Decision.ABSTAIN, ReasonCode.NO_CERTIFIED_THRESHOLD)
    if risk is None or not math.isfinite(risk):
        return GateResult(Decision.ABSTAIN, ReasonCode.INVALID_SCORE)
    if risk <= threshold:
        return GateResult(Decision.ANSWER, ReasonCode.ANSWERED)
    return GateResult(Decision.ABSTAIN, ReasonCode.HIGH_RISK)


def render(result: GateResult, letter: str, option_text: str) -> str:
    return f"{letter}. {option_text}" if result.decision is Decision.ANSWER else ABSTAIN_TEXT
```

- [ ] **Step 2: Verify the domain imports cleanly**

Run: `PYTHONPATH=src python -c "import uncertainty_mech.domain.metrics, uncertainty_mech.domain.gate, uncertainty_mech.domain.splits, uncertainty_mech.domain.grading, uncertainty_mech.domain.prompts"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/uncertainty_mech/domain
git commit -m "Add domain layer: questions, splits, prompts, grading, certificate, metrics, gate"
```

---

### Task 3: Application ports, config and risk model

**Files:**
- Create: `src/uncertainty_mech/application/{__init__,ports,config,risk_model}.py`

**Interfaces:**
- Consumes: Task 2 names.
- Produces: `Readings(letter_logprobs, letter_mass, residuals)`, `Steering(direction, layer, scale)`, protocols `LanguageModel`, `RunStore`, `FigureWriter`, `QuestionSource`, `ReferenceCorpus`; `RunConfig`, `DataConfig`, `ModelConfig`, `ProbeConfig`, `GateConfig`, `SteeringConfig`, `load_config(path)`; `SignalSet`, `build_features`, `RiskModel`, `fit_logistic`, `grouped_cv_auroc`, `LayerScore`, `sweep_layers`, `CScore`, `fit_risk_model`, `fit_temperature`.

- [ ] **Step 1: Write the modules**

**File:** `src/uncertainty_mech/application/__init__.py`
```python
"""Use cases. They reach the model, files and figures only through `ports`."""
```

**File:** `src/uncertainty_mech/application/ports.py`
```python
"""Boundaries the use cases depend on. Adapters live in `infrastructure`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from uncertainty_mech.domain.prompts import ChatPrompt
from uncertainty_mech.domain.questions import HealthQuestion


@dataclass(frozen=True)
class Readings:
    """One forward pass per prompt, read at the last prompt token."""

    letter_logprobs: np.ndarray  # [n, k] log-softmax over the offered letters
    letter_mass: np.ndarray  # [n] full-vocabulary probability on those letters
    residuals: np.ndarray | None = None  # [n, n_layers, d_model] decoder-layer outputs


@dataclass(frozen=True)
class Steering:
    """Add `scale * direction` to the output of one decoder layer at every position."""

    direction: np.ndarray
    layer: int
    scale: float


class LanguageModel(Protocol):
    name: str
    num_layers: int

    def read(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        capture_residuals: bool = False,
        steering: Steering | None = None,
    ) -> Readings: ...


class QuestionSource(Protocol):
    def load(self) -> list[HealthQuestion]: ...


class ReferenceCorpus(Protocol):
    def corpus_words(self) -> set[str]: ...


class RunStore(Protocol):
    root: Path

    def write_json(self, name: str, payload: Any) -> None: ...
    def read_json(self, name: str) -> Any: ...
    def write_arrays(self, name: str, **arrays: np.ndarray) -> None: ...
    def read_arrays(self, name: str) -> dict[str, np.ndarray]: ...
    def write_table(self, name: str, rows: Sequence[dict[str, Any]]) -> None: ...
    def write_text(self, name: str, text: str) -> None: ...
    def append_jsonl(self, name: str, payload: Any) -> None: ...


class FigureWriter(Protocol):
    def layer_sweep(self, rows: Sequence[dict[str, Any]], chosen_layer: int, path: Path) -> None: ...

    def risk_coverage(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray]],
        prompt_only_point: tuple[float, float | None],
        target_risk: float,
        path: Path,
    ) -> None: ...
```

**File:** `src/uncertainty_mech/application/config.py`
```python
"""Run configuration read from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from uncertainty_mech.domain.questions import Role


@dataclass(frozen=True)
class DataConfig:
    real_repo: str
    real_revision: str
    real_file: str
    n_real: int
    n_fictional_entities: int
    proportions: dict[Role, float]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str
    dtype: str = "bfloat16"
    batch_size: int = 16


@dataclass(frozen=True)
class ProbeConfig:
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    sweep_c: float = 0.1
    cv_folds: int = 5


@dataclass(frozen=True)
class GateConfig:
    target_risk: float = 0.10
    delta: float = 0.05


@dataclass(frozen=True)
class SteeringConfig:
    doses: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    n_items: int = 200
    n_random: int = 3
    identity_tolerance: float = 1e-4


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    output_root: Path
    data: DataConfig
    model: ModelConfig
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["data"]["proportions"] = {role.value: share for role, share in self.data.proportions.items()}
        return payload


def load_config(path: Path) -> RunConfig:
    """Read a run config. `output_root` is resolved relative to the config file."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    data = dict(raw["data"])
    data["proportions"] = {Role(name): float(share) for name, share in data["proportions"].items()}
    return RunConfig(
        run_id=raw["run_id"],
        seed=int(raw["seed"]),
        output_root=(path.parent / raw.get("output_root", "runs")).resolve(),
        data=DataConfig(**data),
        model=ModelConfig(**raw["model"]),
        probe=ProbeConfig(**_as_tuples(raw.get("probe", {}))),
        gate=GateConfig(**raw.get("gate", {})),
        steering=SteeringConfig(**_as_tuples(raw.get("steering", {}))),
    )


def _as_tuples(section: dict[str, Any]) -> dict[str, Any]:
    return {key: tuple(value) if isinstance(value, list) else value for key, value in section.items()}
```

**File:** `src/uncertainty_mech/application/risk_model.py`
```python
"""Logistic risk probes, the layer sweep and temperature calibration (protocol sections 3 and 4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from uncertainty_mech.domain.metrics import auroc


class SignalSet(StrEnum):
    OUTPUT = "output"
    PROBE = "probe"
    COMBINED = "combined"

    @property
    def needs_residuals(self) -> bool:
        return self is not SignalSet.OUTPUT


def build_features(signal: SignalSet, output_feats: np.ndarray, layer_residuals: np.ndarray | None) -> np.ndarray:
    """Feature matrix for one signal set. `layer_residuals` is [n, d_model] at the probe layer."""
    if signal is SignalSet.OUTPUT:
        return output_feats
    if layer_residuals is None:
        raise ValueError(f"signal {signal} needs residual activations")
    residuals = np.asarray(layer_residuals, dtype=np.float32)
    if signal is SignalSet.PROBE:
        return residuals
    return np.hstack([output_feats.astype(np.float32), residuals])


@dataclass(frozen=True)
class RiskModel:
    """Standardize, apply a logistic probe, then divide the logit by a temperature."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    c: float
    temperature: float = 1.0

    def logit(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept

    def risk(self, features: np.ndarray) -> np.ndarray:
        return expit(self.logit(features) / self.temperature)

    def with_temperature(self, temperature: float) -> RiskModel:
        return replace(self, temperature=temperature)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": np.array(self.intercept),
            "c": np.array(self.c),
            "temperature": np.array(self.temperature),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> RiskModel:
        return cls(
            mean=arrays["mean"],
            scale=arrays["scale"],
            coef=arrays["coef"],
            intercept=float(arrays["intercept"]),
            c=float(arrays["c"]),
            temperature=float(arrays["temperature"]),
        )


def fit_logistic(features: np.ndarray, errors: np.ndarray, c: float) -> RiskModel:
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(features), errors)
    return RiskModel(
        mean=scaler.mean_, scale=scaler.scale_, coef=classifier.coef_[0], intercept=float(classifier.intercept_[0]), c=c
    )


def grouped_cv_auroc(features: np.ndarray, errors: np.ndarray, groups: np.ndarray, c: float, folds: int) -> list[float]:
    """Held-out AUROC for each grouped fold. Related items never sit on both sides of a fold."""
    scores = []
    for train, valid in GroupKFold(n_splits=folds).split(features, errors, groups):
        if len(np.unique(errors[train])) < 2:
            continue
        score = auroc(fit_logistic(features[train], errors[train], c).logit(features[valid]), errors[valid])
        if score is not None:
            scores.append(score)
    if not scores:
        raise ValueError("no grouped fold contained both correct and wrong answers")
    return scores


@dataclass(frozen=True)
class LayerScore:
    layer: int
    auroc_mean: float
    auroc_sd: float


def sweep_layers(residuals: np.ndarray, errors: np.ndarray, groups: np.ndarray, c: float, folds: int) -> list[LayerScore]:
    """Grouped-CV AUROC of a logistic error probe at every layer: the activation map."""
    scores = []
    for layer in range(residuals.shape[1]):
        fold_scores = grouped_cv_auroc(np.asarray(residuals[:, layer], dtype=np.float32), errors, groups, c, folds)
        scores.append(LayerScore(layer, float(np.mean(fold_scores)), float(np.std(fold_scores))))
    return scores


@dataclass(frozen=True)
class CScore:
    c: float
    auroc_mean: float


def fit_risk_model(
    features: np.ndarray, errors: np.ndarray, groups: np.ndarray, c_grid: Sequence[float], folds: int
) -> tuple[RiskModel, list[CScore]]:
    """Pick C by grouped-CV AUROC, ties to stronger regularization, then refit on every row."""
    c_scores = [CScore(c, float(np.mean(grouped_cv_auroc(features, errors, groups, c, folds)))) for c in c_grid]
    best = max(c_scores, key=lambda score: (score.auroc_mean, -score.c))
    return fit_logistic(features, errors, best.c), c_scores


def fit_temperature(logits: np.ndarray, errors: np.ndarray) -> float:
    """Scalar temperature T minimizing the Bernoulli NLL of sigmoid(logit / T)."""
    signs = np.where(np.asarray(errors) == 1, 1.0, -1.0)

    def nll(log_t: float) -> float:
        return -float(np.mean(log_expit(signs * logits / np.exp(log_t))))

    return float(np.exp(minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded").x))
```

- [ ] **Step 2: Verify imports**

Run: `PYTHONPATH=src python -c "import uncertainty_mech.application.risk_model, uncertainty_mech.application.config, uncertainty_mech.application.ports"`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/uncertainty_mech/application
git commit -m "Add application ports, run config and logistic risk model"
```

---

### Task 4: Use cases

**Files:**
- Create: `src/uncertainty_mech/application/{dataset,gates,evaluation,steering,answer,report,pipeline}.py`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: `Dataset(questions, roles, canonical)` with `groups`, `strata`, `mask(role)`, `item_rows()`; `prepare_dataset(questions, config)`; `read_questions(model, questions, *, allow_abstain, capture_residuals=False) -> Readings`; `PRIMARY_SIGNAL`, `FittedGate`, `fit_gate(signal, features, errors, dataset, config)`; `Evaluation`, `evaluate_on_test(dataset, chosen, errors, risks, gates, abstain_logprobs)`, `NO_GATE`, `PROMPT_ONLY`; `error_direction`, `SteeringResult`, `run_steering_check(model, dataset, direction, layer, config)`; `GateBundle`, `save_bundle`, `load_bundle`, `Answer`, `AbstainingAnswerer(model, bundle, audit_store=None).answer(question, options)`; `RunResults`, `write_report(store, figures, results) -> dict`; `run_pipeline(config, questions, model, store, figures) -> dict`.

- [ ] **Step 1: Write the dataset and gate modules**

**File:** `src/uncertainty_mech/application/dataset.py`
```python
"""Stage 1 and 2: questions with their roles, and one model read per question."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from uncertainty_mech.application.config import RunConfig
from uncertainty_mech.application.ports import LanguageModel, Readings
from uncertainty_mech.domain.prompts import answer_letters, build_prompt
from uncertainty_mech.domain.questions import HealthQuestion, Role
from uncertainty_mech.domain.splits import assign_roles, canonical_mask


@dataclass(frozen=True)
class Dataset:
    questions: tuple[HealthQuestion, ...]
    roles: np.ndarray  # role value per item
    canonical: np.ndarray  # True for the one certification unit per group

    @cached_property
    def groups(self) -> np.ndarray:
        return np.array([q.group_id for q in self.questions])

    @cached_property
    def strata(self) -> np.ndarray:
        return np.array([q.stratum.value for q in self.questions])

    def mask(self, role: Role) -> np.ndarray:
        return self.roles == role.value

    def item_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "item_id": q.item_id,
                "group_id": q.group_id,
                "stratum": q.stratum.value,
                "role": role,
                "canonical": bool(canonical),
                "answer_index": q.answer_index,
                "question": q.question,
                "options_json": json.dumps(list(q.options)),
            }
            for q, role, canonical in zip(self.questions, self.roles, self.canonical, strict=True)
        ]


def prepare_dataset(questions: Sequence[HealthQuestion], config: RunConfig) -> Dataset:
    ids = [q.item_id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique")
    roles = assign_roles(questions, config.data.proportions, config.seed)
    return Dataset(
        questions=tuple(questions),
        roles=np.array([role.value for role in roles]),
        canonical=np.array(canonical_mask(questions, config.seed)),
    )


def read_questions(
    model: LanguageModel, questions: Sequence[HealthQuestion], *, allow_abstain: bool, capture_residuals: bool = False
) -> Readings:
    prompts = [build_prompt(q.question, q.options, allow_abstain=allow_abstain) for q in questions]
    return model.read(prompts, answer_letters(allow_abstain), capture_residuals=capture_residuals)
```

**File:** `src/uncertainty_mech/application/gates.py`
```python
"""Stages 4 to 6: fit on discovery, calibrate on cal_prob, certify on cal_gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncertainty_mech.application.config import RunConfig
from uncertainty_mech.application.dataset import Dataset
from uncertainty_mech.application.risk_model import CScore, RiskModel, SignalSet, fit_risk_model, fit_temperature
from uncertainty_mech.domain.questions import Role
from uncertainty_mech.domain.risk_control import THRESHOLDS, ThresholdSelection, select_threshold

# Fixed before the run: the gate that `ask` deploys.
PRIMARY_SIGNAL = SignalSet.COMBINED


@dataclass(frozen=True)
class FittedGate:
    signal: SignalSet
    risk_model: RiskModel
    c_scores: list[CScore]
    selection: ThresholdSelection

    def released(self, risk: np.ndarray) -> np.ndarray:
        if self.selection.threshold is None:
            return np.zeros(len(risk), dtype=bool)
        return risk <= self.selection.threshold


def fit_gate(signal: SignalSet, features: np.ndarray, errors: np.ndarray, dataset: Dataset, config: RunConfig) -> FittedGate:
    discovery = dataset.mask(Role.DISCOVERY)
    cal_prob = dataset.mask(Role.CAL_PROB)
    gate_units = dataset.mask(Role.CAL_GATE) & dataset.canonical
    model, c_scores = fit_risk_model(
        features[discovery], errors[discovery], dataset.groups[discovery], config.probe.c_grid, config.probe.cv_folds
    )
    model = model.with_temperature(fit_temperature(model.logit(features[cal_prob]), errors[cal_prob]))
    selection = select_threshold(
        model.risk(features[gate_units]), errors[gate_units], THRESHOLDS, config.gate.target_risk, config.gate.delta
    )
    return FittedGate(signal, model, c_scores, selection)
```

- [ ] **Step 2: Write the evaluation and steering modules**

**File:** `src/uncertainty_mech/application/evaluation.py`
```python
"""Stage 7: one pass over the test role with frozen gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from uncertainty_mech.application.dataset import Dataset
from uncertainty_mech.application.gates import FittedGate
from uncertainty_mech.application.risk_model import SignalSet
from uncertainty_mech.domain.grading import chosen_indices, error_labels
from uncertainty_mech.domain.metrics import risk_coverage_curve, risk_quality, selective_metrics
from uncertainty_mech.domain.prompts import answer_letters
from uncertainty_mech.domain.questions import OPTION_LETTERS, Role, Stratum

NO_GATE = "no_gate"
PROMPT_ONLY = "prompt_only"


@dataclass(frozen=True)
class Evaluation:
    metrics_rows: list[dict[str, Any]]
    quality_rows: list[dict[str, Any]]
    matched_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    curves: dict[str, tuple[np.ndarray, np.ndarray]]
    prompt_only_point: tuple[float, float | None]

    def metric(self, method: str, stratum: str) -> dict[str, Any] | None:
        return next((r for r in self.metrics_rows if r["method"] == method and r["stratum"] == stratum), None)


def evaluate_on_test(
    dataset: Dataset,
    chosen: np.ndarray,
    errors: np.ndarray,
    risks: dict[SignalSet, np.ndarray],
    gates: dict[SignalSet, FittedGate],
    abstain_logprobs: np.ndarray,
) -> Evaluation:
    test = dataset.mask(Role.TEST)
    subsets = {
        "all": test,
        "real": test & (dataset.strata == Stratum.REAL.value),
        "fictional": test & (dataset.strata == Stratum.FICTIONAL.value),
    }
    prompt_choice = chosen_indices(abstain_logprobs)
    prompt_abstains = prompt_choice == len(OPTION_LETTERS)
    # Rows that chose E are never released, so their error label is never used.
    prompt_errors = error_labels(dataset.questions, prompt_choice)

    released = {NO_GATE: np.ones(len(errors), dtype=bool), PROMPT_ONLY: ~prompt_abstains}
    candidate_errors = {NO_GATE: errors, PROMPT_ONLY: prompt_errors}
    for signal, gate in gates.items():
        released[signal.value] = gate.released(risks[signal])
        candidate_errors[signal.value] = errors
    originally_correct = errors == 0

    metrics_rows = [
        {
            "method": method,
            "stratum": stratum,
            **asdict(selective_metrics(candidate_errors[method][mask], kept[mask], int(originally_correct[mask].sum()))),
        }
        for method, kept in released.items()
        for stratum, mask in subsets.items()
        if mask.any()
    ]
    quality_rows = [
        {"signal": signal.value, "stratum": stratum, **asdict(risk_quality(risks[signal][subsets[stratum]], errors[subsets[stratum]]))}
        for signal in gates
        for stratum in ("all", "real")
        if subsets[stratum].any()
    ]

    n_test = int(test.sum())
    prompt_kept = released[PROMPT_ONLY][test]
    prompt_coverage = float(prompt_kept.mean())
    prompt_risk = float(prompt_errors[test][prompt_kept].mean()) if prompt_kept.any() else None
    k = int(round(prompt_coverage * n_test))
    matched_rows = [{"method": PROMPT_ONLY, "answered": int(prompt_kept.sum()), "coverage": prompt_coverage, "selective_risk": prompt_risk}]
    for signal in gates:
        lowest = np.argsort(risks[signal][test], kind="stable")[:k]
        matched_rows.append(
            {
                "method": signal.value,
                "answered": k,
                "coverage": k / n_test,
                "selective_risk": float(errors[test][lowest].mean()) if k else None,
            }
        )

    letters = answer_letters(allow_abstain=True)
    prediction_rows = []
    for i in np.flatnonzero(test):
        question = dataset.questions[i]
        row: dict[str, Any] = {
            "item_id": question.item_id,
            "stratum": question.stratum.value,
            "candidate": OPTION_LETTERS[chosen[i]],
            "correct": "" if question.answer_index is None else OPTION_LETTERS[question.answer_index],
            "error": int(errors[i]),
            "prompt_only_choice": letters[prompt_choice[i]],
        }
        for signal in gates:
            row[f"risk_{signal.value}"] = float(risks[signal][i])
            row[f"released_{signal.value}"] = bool(released[signal.value][i])
        prediction_rows.append(row)

    return Evaluation(
        metrics_rows=metrics_rows,
        quality_rows=quality_rows,
        matched_rows=matched_rows,
        prediction_rows=prediction_rows,
        curves={signal.value: risk_coverage_curve(risks[signal][test], errors[test]) for signal in gates},
        prompt_only_point=(prompt_coverage, prompt_risk),
    )
```

**File:** `src/uncertainty_mech/application/steering.py`
```python
"""Stage 8, exploratory: does adding the error direction change the model's own abstention?

The direction is the difference of class means (the mass-mean probe of ARENA
1.3.1). It is added to one decoder layer's output on prompts that offer option
E. Norm-matched random directions are the control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from uncertainty_mech.application.config import RunConfig
from uncertainty_mech.application.dataset import Dataset
from uncertainty_mech.application.ports import LanguageModel, Readings, Steering
from uncertainty_mech.domain.grading import chosen_indices
from uncertainty_mech.domain.prompts import answer_letters, build_prompt
from uncertainty_mech.domain.questions import OPTION_LETTERS, Role

ERROR_DIRECTION = "error_direction"


def error_direction(residuals: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Mean residual of wrong answers minus mean residual of right answers."""
    values = np.asarray(residuals, dtype=np.float64)
    return (values[errors == 1].mean(axis=0) - values[errors == 0].mean(axis=0)).astype(np.float32)


@dataclass(frozen=True)
class SteeringResult:
    layer: int
    n_items: int
    direction_norm: float
    identity_max_abs_diff: float
    identity_ok: bool
    rows: list[dict[str, Any]]


def run_steering_check(
    model: LanguageModel, dataset: Dataset, direction: np.ndarray, layer: int, config: RunConfig
) -> SteeringResult:
    rng = np.random.default_rng(config.seed)
    test_indices = np.flatnonzero(dataset.mask(Role.TEST))
    picks = np.sort(rng.choice(test_indices, size=min(config.steering.n_items, len(test_indices)), replace=False))
    questions = [dataset.questions[i] for i in picks]
    prompts = [build_prompt(q.question, q.options, allow_abstain=True) for q in questions]
    letters = answer_letters(allow_abstain=True)

    plain = model.read(prompts, letters)
    zero_dose = model.read(prompts, letters, steering=Steering(direction, layer, 0.0))
    identity_diff = float(np.max(np.abs(plain.letter_logprobs - zero_dose.letter_logprobs)))

    norm = float(np.linalg.norm(direction))
    directions = {ERROR_DIRECTION: direction}
    for i in range(config.steering.n_random):
        random = rng.standard_normal(direction.shape).astype(np.float32)
        directions[f"random_{i + 1}"] = random * (norm / float(np.linalg.norm(random)))

    answers = np.array([-1 if q.answer_index is None else q.answer_index for q in questions])
    rows = []
    for name, vector in directions.items():
        for dose in config.steering.doses:
            readings = plain if dose == 0 else model.read(prompts, letters, steering=Steering(vector, layer, dose))
            rows.append({"direction": name, "dose": dose, **_behavior(readings, answers)})
    return SteeringResult(
        layer=layer,
        n_items=len(questions),
        direction_norm=norm,
        identity_max_abs_diff=identity_diff,
        identity_ok=identity_diff <= config.steering.identity_tolerance,
        rows=rows,
    )


def _behavior(readings: Readings, answers: np.ndarray) -> dict[str, float | None]:
    abstain = len(OPTION_LETTERS)
    choice = chosen_indices(readings.letter_logprobs)
    real = answers >= 0
    return {
        "mean_p_abstain": float(np.exp(readings.letter_logprobs[:, abstain]).mean()),
        "abstain_rate": float((choice == abstain).mean()),
        "real_accuracy": float((choice[real] == answers[real]).mean()) if real.any() else None,
    }
```

- [ ] **Step 3: Write the inference service**

**File:** `src/uncertainty_mech/application/answer.py`
```python
"""The inference service: answer a health question, or say exactly "I don't know."."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from uncertainty_mech.application.ports import LanguageModel, RunStore
from uncertainty_mech.application.risk_model import RiskModel, SignalSet, build_features
from uncertainty_mech.domain.gate import Decision, ReasonCode, decide, render
from uncertainty_mech.domain.grading import chosen_indices, output_features
from uncertainty_mech.domain.prompts import answer_letters, build_prompt
from uncertainty_mech.domain.questions import OPTION_LETTERS

BUNDLE_FILE = "bundle.json"
BUNDLE_ARRAYS = "bundle_risk_model.npz"
AUDIT_FILE = "audit.jsonl"


@dataclass(frozen=True)
class GateBundle:
    """Everything `ask` needs, frozen at the end of calibration."""

    run_id: str
    model_name: str
    signal: SignalSet
    layer: int
    risk_model: RiskModel
    threshold: float | None
    target_risk: float


def save_bundle(store: RunStore, bundle: GateBundle) -> None:
    store.write_json(
        BUNDLE_FILE,
        {
            "run_id": bundle.run_id,
            "model_name": bundle.model_name,
            "signal": bundle.signal.value,
            "layer": bundle.layer,
            "threshold": bundle.threshold,
            "target_risk": bundle.target_risk,
        },
    )
    store.write_arrays(BUNDLE_ARRAYS, **bundle.risk_model.to_arrays())


def load_bundle(store: RunStore) -> GateBundle:
    meta = store.read_json(BUNDLE_FILE)
    return GateBundle(
        run_id=meta["run_id"],
        model_name=meta["model_name"],
        signal=SignalSet(meta["signal"]),
        layer=int(meta["layer"]),
        risk_model=RiskModel.from_arrays(store.read_arrays(BUNDLE_ARRAYS)),
        threshold=meta["threshold"],
        target_risk=float(meta["target_risk"]),
    )


@dataclass(frozen=True)
class Answer:
    text: str
    decision: Decision
    reason_code: ReasonCode
    risk: float | None
    candidate_letter: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "decision": self.decision.value,
            "reason_code": self.reason_code.value,
            "risk": self.risk,
            "candidate_letter": self.candidate_letter,
        }


class AbstainingAnswerer:
    """Reads the model once, scores the risk, and releases the answer only below the frozen threshold."""

    def __init__(self, model: LanguageModel, bundle: GateBundle, audit_store: RunStore | None = None) -> None:
        if model.name != bundle.model_name:
            raise ValueError(f"the gate was calibrated on {bundle.model_name}, not {model.name}")
        self._model = model
        self._bundle = bundle
        self._audit_store = audit_store

    def answer(self, question: str, options: Sequence[str]) -> Answer:
        started = time.perf_counter()
        readings = self._model.read(
            [build_prompt(question, options, allow_abstain=False)],
            answer_letters(allow_abstain=False),
            capture_residuals=self._bundle.signal.needs_residuals,
        )
        index = int(chosen_indices(readings.letter_logprobs)[0])
        risk = self._risk(readings)
        result = decide(risk, self._bundle.threshold)
        answer = Answer(render(result, OPTION_LETTERS[index], options[index]), result.decision, result.reason_code, risk, OPTION_LETTERS[index])
        if self._audit_store is not None:
            self._audit_store.append_jsonl(AUDIT_FILE, self._audit_record(question, answer, started))
        return answer

    def _risk(self, readings) -> float | None:
        residuals = readings.residuals
        if self._bundle.signal.needs_residuals and residuals is None:
            return None
        layer_residuals = None if residuals is None else residuals[:, self._bundle.layer]
        features = build_features(
            self._bundle.signal, output_features(readings.letter_logprobs, readings.letter_mass), layer_residuals
        )
        return float(self._bundle.risk_model.risk(features)[0])

    def _audit_record(self, question: str, answer: Answer, started: float) -> dict[str, Any]:
        return {
            "time_utc": datetime.now(UTC).isoformat(),
            "run_id": self._bundle.run_id,
            "model": self._model.name,
            "signal": self._bundle.signal.value,
            "layer": self._bundle.layer,
            "threshold": self._bundle.threshold,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            **answer.to_dict(),
        }
```

- [ ] **Step 4: Write the report and the pipeline**

**File:** `src/uncertainty_mech/application/report.py`
```python
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
```

**File:** `src/uncertainty_mech/application/pipeline.py`
```python
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
from uncertainty_mech.application.report import RunResults, write_report
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
        readings.residuals[discovery], errors[discovery], dataset.groups[discovery], config.probe.sweep_c, config.probe.cv_folds
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
    steering = run_steering_check(model, dataset, error_direction(layer_residuals[discovery], errors[discovery]), layer, config)

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
```

- [ ] **Step 5: Verify imports**

Run: `PYTHONPATH=src python -c "import uncertainty_mech.application.pipeline, uncertainty_mech.application.answer"`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/uncertainty_mech/application
git commit -m "Add use cases: dataset, gates, evaluation, steering, answer service, report, pipeline"
```

---

### Task 5: Local adapters and the composition root (end-to-end test passes)

**Files:**
- Create: `src/uncertainty_mech/infrastructure/{__init__,store,fictional,figures}.py`, `src/uncertainty_mech/cli.py`

**Interfaces:**
- Consumes: ports from Task 3, `run_pipeline`, `AbstainingAnswerer`, `load_bundle` from Task 4.
- Produces: `FileRunStore(root)`, `FictionalHealthSource(n_entities, seed, exclude_words=())`, `MatplotlibFigures()`, `cli.main`.

- [ ] **Step 1: Write the adapters**

**File:** `src/uncertainty_mech/infrastructure/__init__.py`
```python
"""Adapters for the ports: files, figures, question sources and the Hugging Face model."""
```

**File:** `src/uncertainty_mech/infrastructure/store.py`
```python
"""A run directory on disk: JSON, NPZ, CSV, markdown and an append-only audit log."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class FileRunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> None:
        self._path(name).write_text(json.dumps(payload, indent=2, default=_encode) + "\n", encoding="utf-8")

    def read_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def write_arrays(self, name: str, **arrays: np.ndarray) -> None:
        np.savez(self._path(name), **arrays)

    def read_arrays(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.root / name) as data:
            return {key: data[key] for key in data.files}

    def write_table(self, name: str, rows: Sequence[dict[str, Any]]) -> None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with self._path(name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_text(self, name: str, text: str) -> None:
        self._path(name).write_text(text, encoding="utf-8")

    def append_jsonl(self, name: str, payload: Any) -> None:
        with self._path(name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_encode) + "\n")

    def _path(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")
```

**File:** `src/uncertainty_mech/infrastructure/fictional.py`
```python
"""Invented drugs and diseases: questions whose only right answer is "I don't know.".

Each entity gets three clinical questions in MedQA style. The options are real
drugs, genes, organisms and so on, so a fluent guess looks plausible.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from uncertainty_mech.domain.questions import HealthQuestion, Stratum

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kl", "tr", "gr", "st")
_VOWELS = ("a", "e", "i", "o", "u")
_CODAS = ("l", "n", "r", "s", "v", "x", "m", "d", "k", "th")
_DRUG_ENDINGS = ("axil", "ovane", "urex", "endra", "ivon", "omyr", "alith", "eprax")
_EPONYM_ENDINGS = ("nick", "dorf", "holm", "ley", "sen", "ward", "berg", "ton")
_DISEASE_FORMS = ("{} syndrome", "{} disease")
# Real drug-class stems. An invented name must not hint at a real class.
_BANNED_FRAGMENTS = (
    "olol", "pril", "sartan", "statin", "dipine", "mab", "nib", "azole", "cillin", "mycin", "cycline", "terol",
    "prazole", "tidine", "vir", "parin", "gliptin", "gliflozin", "afil", "triptan", "setron", "oxetine", "caine",
    "floxacin", "sone", "lone", "pam", "lam",
)
_TEMPLATES_PER_ENTITY = 3

_POOLS: dict[str, tuple[str, ...]] = {
    "drugs": (
        "Metformin", "Lisinopril", "Amoxicillin", "Prednisone", "Levothyroxine", "Methotrexate",
        "Hydroxychloroquine", "Allopurinol", "Warfarin", "Omeprazole", "Azithromycin", "Furosemide",
    ),
    "genes": ("CFTR", "FBN1", "BRCA1", "NF1", "HBB", "DMD", "COL1A1", "PKD1", "HTT", "FMR1", "APC", "RET"),
    "organisms": (
        "Staphylococcus aureus", "Streptococcus pneumoniae", "Escherichia coli", "Borrelia burgdorferi",
        "Mycobacterium tuberculosis", "Plasmodium falciparum", "Treponema pallidum", "Clostridioides difficile",
        "Listeria monocytogenes", "Coxiella burnetii",
    ),
    "enzymes": (
        "Glucose-6-phosphate dehydrogenase", "Hexosaminidase A", "Glucocerebrosidase", "Phenylalanine hydroxylase",
        "Alpha-galactosidase A", "21-hydroxylase", "Ornithine transcarbamylase", "Adenosine deaminase",
        "Homogentisate oxidase", "Acid alpha-glucosidase",
    ),
    "mechanisms": (
        "Inhibition of angiotensin-converting enzyme", "Blockade of beta-1 adrenergic receptors",
        "Inhibition of HMG-CoA reductase", "Irreversible inhibition of the H+/K+ ATPase",
        "Blockade of L-type calcium channels", "Inhibition of cyclooxygenase-1 and -2",
        "Inhibition of dihydrofolate reductase", "Inhibition of bacterial cell wall synthesis",
        "Agonism at mu-opioid receptors", "Inhibition of vitamin K epoxide reductase",
    ),
    "adverse_effects": (
        "Hepatotoxicity", "Agranulocytosis", "QT prolongation", "Lactic acidosis", "Angioedema", "Ototoxicity",
        "Pulmonary fibrosis", "Stevens-Johnson syndrome", "Hyperkalemia", "Tendon rupture",
    ),
    "conditions": (
        "Type 2 diabetes mellitus", "Essential hypertension", "Rheumatoid arthritis", "Major depressive disorder",
        "Migraine", "Gout", "Asthma", "Hypothyroidism", "Atrial fibrillation", "Parkinson disease",
    ),
    "doses": (
        "5 mg once daily", "10 mg twice daily", "25 mg once daily", "50 mg three times daily", "100 mg once daily",
        "250 mg twice daily", "500 mg twice daily", "1 g once daily", "2.5 mg once weekly", "20 mg at bedtime",
    ),
}
_DISEASE_TEMPLATES = (
    ("treatment", "Which drug is the first-line treatment for this condition?", "drugs"),
    ("gene", "Mutation of which gene causes this condition?", "genes"),
    ("organism", "Which organism causes this condition?", "organisms"),
    ("enzyme", "Deficiency of which enzyme causes this condition?", "enzymes"),
)
_DRUG_TEMPLATES = (
    ("mechanism", "What is the mechanism of action of this drug?", "mechanisms"),
    ("adverse", "Which serious adverse effect is most characteristic of this drug?", "adverse_effects"),
    ("indication", "For which condition is this drug first-line therapy?", "conditions"),
    ("dose", "What is the usual adult starting dose of this drug?", "doses"),
)


@dataclass(frozen=True)
class _Entity:
    kind: str  # "drug" or "disease"
    slug: str
    display: str


class FictionalHealthSource:
    def __init__(self, n_entities: int, seed: int, exclude_words: Iterable[str] = ()) -> None:
        self._n_entities = n_entities
        self._seed = seed
        self._exclude = {word.lower() for word in exclude_words}

    def load(self) -> list[HealthQuestion]:
        rng = random.Random(self._seed)
        questions = []
        for entity in self._entities(rng):
            is_drug = entity.kind == "drug"
            templates = _DRUG_TEMPLATES if is_drug else _DISEASE_TEMPLATES
            for key, question, pool in rng.sample(templates, _TEMPLATES_PER_ENTITY):
                patient = f"A {rng.randint(19, 82)}-year-old {rng.choice(('man', 'woman'))}"
                vignette = f"{patient} is started on {entity.display}." if is_drug else f"{patient} is diagnosed with {entity.display}."
                questions.append(
                    HealthQuestion(
                        item_id=f"fict-{entity.slug}-{key}",
                        group_id=f"fict-{entity.slug}",
                        stratum=Stratum.FICTIONAL,
                        question=f"{vignette} {question}",
                        options=tuple(rng.sample(_POOLS[pool], 4)),
                        answer_index=None,
                    )
                )
        return questions

    def _entities(self, rng: random.Random) -> list[_Entity]:
        entities: list[_Entity] = []
        seen: set[str] = set()
        for _ in range(1000 * self._n_entities):
            if len(entities) == self._n_entities:
                return entities
            kind = "drug" if len(entities) % 2 == 0 else "disease"
            stem = rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
            if kind == "drug":
                slug = stem + rng.choice(_DRUG_ENDINGS)
                display = slug.capitalize()
            else:
                slug = stem + rng.choice(_VOWELS) + rng.choice(_EPONYM_ENDINGS)
                display = rng.choice(_DISEASE_FORMS).format(slug.capitalize())
            if slug in seen or slug in self._exclude or any(fragment in slug for fragment in _BANNED_FRAGMENTS):
                continue
            seen.add(slug)
            entities.append(_Entity(kind, slug, display))
        raise RuntimeError(f"could only invent {len(entities)} of {self._n_entities} unused names")
```

**File:** `src/uncertainty_mech/infrastructure/figures.py`
```python
"""Figures in the house style: serif type, no titles, sentence case, legends outside the axes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 11.5,
    "legend.fontsize": 10,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
}
# One colour per method, held across every figure.
_COLORS = {
    "output": "#7f7f7f",
    "probe": "#1b6ca8",
    "combined": "#c0392b",
    "prompt_only": "#2e8b57",
    "reference": "#444444",
    "chance": "#b0b0b0",
}
_LABELS = {
    "output": "Output statistics",
    "probe": "Activation probe",
    "combined": "Combined gate",
    "prompt_only": "Prompt-only option E",
}


class MatplotlibFigures:
    def layer_sweep(self, rows: Sequence[dict[str, Any]], chosen_layer: int, path: Path) -> None:
        layers = np.array([row["layer"] for row in rows])
        mean = np.array([row["auroc_mean"] for row in rows])
        spread = np.array([row["auroc_sd"] for row in rows])
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            ax.fill_between(layers, mean - spread, mean + spread, color=_COLORS["probe"], alpha=0.18, linewidth=0, label="One fold SD")
            ax.plot(layers, mean, color=_COLORS["probe"], marker="o", markersize=3, label="Mean over 5 grouped folds")
            ax.axvline(chosen_layer, color=_COLORS["reference"], linestyle="--", linewidth=1, label=f"Chosen layer ({chosen_layer})")
            ax.axhline(0.5, color=_COLORS["chance"], linewidth=0.8, label="Chance")
            _finish(fig, ax, "Decoder layer", "Error-probe AUROC on discovery folds", path)

    def risk_coverage(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray]],
        prompt_only_point: tuple[float, float | None],
        target_risk: float,
        path: Path,
    ) -> None:
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            for name, (coverage, risk) in curves.items():
                ax.plot(coverage, risk, color=_COLORS[name], linewidth=1.4, label=_LABELS[name])
            coverage, risk = prompt_only_point
            if risk is not None:
                ax.plot([coverage], [risk], marker="D", linestyle="none", color=_COLORS["prompt_only"], label=_LABELS["prompt_only"])
            ax.axhline(target_risk, color=_COLORS["reference"], linestyle=":", linewidth=1, label="Target error rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            _finish(fig, ax, "Share of test questions answered", "Error rate among answered questions", path)


def _finish(fig: plt.Figure, ax: plt.Axes, xlabel: str, ylabel: str, path: Path) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
```

- [ ] **Step 2: Write the composition root**

**File:** `src/uncertainty_mech/cli.py`
```python
"""Composition root: the only module that knows every adapter.

    python -m uncertainty_mech run --config configs/health_test_run.toml
    python -m uncertainty_mech ask --run-dir runs/<run_id> --questions examples/health_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from uncertainty_mech.application.answer import AbstainingAnswerer, load_bundle
from uncertainty_mech.application.config import DataConfig, ModelConfig, load_config
from uncertainty_mech.application.pipeline import run_pipeline
from uncertainty_mech.application.ports import LanguageModel, QuestionSource, ReferenceCorpus
from uncertainty_mech.infrastructure.fictional import FictionalHealthSource
from uncertainty_mech.infrastructure.figures import MatplotlibFigures
from uncertainty_mech.infrastructure.store import FileRunStore

log = logging.getLogger(__name__)


class RealSource(QuestionSource, ReferenceCorpus, Protocol):
    pass


ModelFactory = Callable[[ModelConfig], LanguageModel]
RealSourceFactory = Callable[[DataConfig, int], RealSource]


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: ModelFactory | None = None,
    real_source_factory: RealSourceFactory | None = None,
) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    model_factory = model_factory or _huggingface_model
    if args.command == "run":
        return _run(args.config, model_factory, real_source_factory or _medqa_source)
    return _ask(args.run_dir, args.questions, model_factory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uncertainty-mech", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Build data, read the model, fit and certify the gates, evaluate, report.")
    run.add_argument("--config", type=Path, required=True)
    ask = commands.add_parser("ask", help="Answer questions with a frozen gate, or say I don't know.")
    ask.add_argument("--run-dir", type=Path, required=True)
    ask.add_argument("--questions", type=Path, required=True, help="JSONL lines with 'question' and 4 'options'.")
    return parser


def _run(config_path: Path, model_factory: ModelFactory, real_source_factory: RealSourceFactory) -> int:
    config = load_config(config_path)
    store = FileRunStore(config.run_dir)
    handler = logging.FileHandler(store.root / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        real_source = real_source_factory(config.data, config.seed)
        real = real_source.load()
        fictional = FictionalHealthSource(
            config.data.n_fictional_entities, config.seed, exclude_words=real_source.corpus_words()
        ).load()
        summary = run_pipeline(config, real + fictional, model_factory(config.model), store, MatplotlibFigures())
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    print(json.dumps(summary, indent=2))
    return 0


def _ask(run_dir: Path, questions_path: Path, model_factory: ModelFactory) -> int:
    store = FileRunStore(run_dir)
    model_config = ModelConfig(**store.read_json("config.json")["model"])
    answerer = AbstainingAnswerer(model_factory(model_config), load_bundle(store), audit_store=store)
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        answer = answerer.answer(item["question"], item["options"])
        print(json.dumps({"question": item["question"], **answer.to_dict()}))
    return 0


def _huggingface_model(config: ModelConfig) -> LanguageModel:
    from uncertainty_mech.infrastructure.hf_model import HuggingFaceModel  # torch loads only when a real model is needed

    return HuggingFaceModel.from_config(config)


def _medqa_source(config: DataConfig, seed: int) -> RealSource:
    from uncertainty_mech.infrastructure.medqa import MedQASource

    return MedQASource(config.real_repo, config.real_revision, config.real_file, config.n_real, seed)
```

- [ ] **Step 3: Run the end-to-end test**

Run: `python -m pytest -q`
Expected: `5 passed`.

- [ ] **Step 4: Commit**

```bash
git add src/uncertainty_mech/infrastructure src/uncertainty_mech/cli.py
git commit -m "Add file store, fictional source, figures and CLI; end-to-end test passes"
```

---

### Task 6: MedQA and Hugging Face adapters, GPU smoke run

**Files:**
- Create: `src/uncertainty_mech/infrastructure/medqa.py`, `src/uncertainty_mech/infrastructure/hf_model.py`, `configs/health_smoke.toml`

**Interfaces:**
- Consumes: `Readings`, `Steering`, `ChatPrompt`, `ModelConfig`, `HealthQuestion`, `stable_unit`.
- Produces: `MedQASource(repo_id, revision, filename, n_items, seed)` with `load()` and `corpus_words()`; `HuggingFaceModel.from_config(config)` satisfying `LanguageModel`.

- [ ] **Step 1: Write the adapters and smoke config**

**File:** `src/uncertainty_mech/infrastructure/medqa.py`
```python
"""MedQA-USMLE four-option questions from the Hugging Face Hub, pinned to one revision."""

from __future__ import annotations

import hashlib
import json
import re
from functools import cached_property
from typing import Any

from huggingface_hub import hf_hub_download

from uncertainty_mech.domain.questions import OPTION_LETTERS, HealthQuestion, Stratum
from uncertainty_mech.domain.splits import stable_unit

_WORD = re.compile(r"[a-z][a-z'-]+")


class MedQASource:
    def __init__(self, repo_id: str, revision: str, filename: str, n_items: int, seed: int) -> None:
        self._repo_id = repo_id
        self._revision = revision
        self._filename = filename
        self._n_items = n_items
        self._seed = seed

    def load(self) -> list[HealthQuestion]:
        """Deduplicate by normalized question text, then take a seeded sample."""
        unique: dict[str, dict[str, Any]] = {}
        for record in self._records:
            unique.setdefault(" ".join(record["question"].lower().split()), record)
        questions = [self._to_question(key, record) for key, record in unique.items()]
        questions.sort(key=lambda question: stable_unit(question.group_id, self._seed))
        if len(questions) < self._n_items:
            raise ValueError(f"asked for {self._n_items} questions, only {len(questions)} are unique")
        return questions[: self._n_items]

    def corpus_words(self) -> set[str]:
        """Every word in the whole file, so invented names can avoid real ones."""
        words: set[str] = set()
        for record in self._records:
            words.update(_WORD.findall(" ".join([record["question"], *record["options"].values()]).lower()))
        return words

    @cached_property
    def _records(self) -> list[dict[str, Any]]:
        path = hf_hub_download(self._repo_id, self._filename, repo_type="dataset", revision=self._revision)
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _to_question(key: str, record: dict[str, Any]) -> HealthQuestion:
        item_id = "medqa-" + hashlib.sha256(key.encode()).hexdigest()[:16]
        return HealthQuestion(
            item_id=item_id,
            group_id=item_id,
            stratum=Stratum.REAL,
            question=record["question"].strip(),
            options=tuple(record["options"][letter] for letter in OPTION_LETTERS),
            answer_index=OPTION_LETTERS.index(record["answer_idx"]),
        )
```

**File:** `src/uncertainty_mech/infrastructure/hf_model.py`
```python
"""Hugging Face transformers adapter for the `LanguageModel` port.

Reads follow the ARENA 1.3.1 pattern: forward hooks on `model.model.layers[i]`
capture the residual stream at the last prompt token, and one more hook adds a
steering vector to a layer's output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uncertainty_mech.application.config import ModelConfig
from uncertainty_mech.application.ports import Readings, Steering
from uncertainty_mech.domain.prompts import ChatPrompt

log = logging.getLogger(__name__)


class HuggingFaceModel:
    def __init__(self, name: str, revision: str, dtype: str = "bfloat16", batch_size: int = 16, device: str = "cuda") -> None:
        self.name = f"{name}@{revision}"
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self._tokenizer.padding_side = "left"  # the last position is then the last prompt token for every row
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=getattr(torch, dtype), device_map=device
        ).eval()
        self._layers = self._model.model.layers
        self.num_layers = len(self._layers)

    @classmethod
    def from_config(cls, config: ModelConfig) -> HuggingFaceModel:
        return cls(config.name, config.revision, config.dtype, config.batch_size)

    def read(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        capture_residuals: bool = False,
        steering: Steering | None = None,
    ) -> Readings:
        letter_ids = [self._single_token(letter) for letter in letters]
        texts = [self._render(prompt) for prompt in prompts]
        lengths = [len(ids) for ids in self._tokenizer(texts, add_special_tokens=False).input_ids]
        order = np.argsort(lengths)[::-1]  # longest first, so the memory peak comes early
        logprobs = np.zeros((len(texts), len(letters)), dtype=np.float32)
        mass = np.zeros(len(texts), dtype=np.float32)
        hidden_size = self._model.config.hidden_size
        residuals = np.zeros((len(texts), self.num_layers, hidden_size), dtype=np.float32) if capture_residuals else None
        n_batches = (len(order) + self._batch_size - 1) // self._batch_size
        for number, start in enumerate(range(0, len(order), self._batch_size), start=1):
            batch = order[start : start + self._batch_size]
            batch_logprobs, batch_mass, batch_residuals = self._read_batch(
                [texts[i] for i in batch], letter_ids, capture_residuals, steering
            )
            logprobs[batch] = batch_logprobs
            mass[batch] = batch_mass
            if residuals is not None:
                residuals[batch] = batch_residuals
            if number % 50 == 0 or number == n_batches:
                log.info("Read batch %d/%d", number, n_batches)
        return Readings(logprobs, mass, residuals)

    def _single_token(self, letter: str) -> int:
        ids = self._tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"letter {letter!r} is not a single token: {ids}")
        return ids[0]

    def _render(self, prompt: ChatPrompt) -> str:
        messages = [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def _read_batch(
        self, texts: list[str], letter_ids: list[int], capture: bool, steering: Steering | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self._model.device)
        # Left padding shifts the real tokens, so positions must count real tokens only.
        position_ids = (encoded.attention_mask.cumsum(-1) - 1).clamp(min=0)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        if steering is not None:  # registered first, so any capture at that layer sees the steered state
            handles.append(self._layers[steering.layer].register_forward_hook(self._steering_hook(steering)))
        if capture:
            handles += [layer.register_forward_hook(_capture_hook(index, captured)) for index, layer in enumerate(self._layers)]
        try:
            logits = self._model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :].float()
        finally:
            for handle in handles:
                handle.remove()
        letter_logprobs_full = torch.log_softmax(logits, dim=-1)[:, letter_ids]
        restricted = torch.log_softmax(logits[:, letter_ids], dim=-1)
        residuals = torch.stack([captured[i] for i in range(self.num_layers)], dim=1).numpy() if capture else None
        return restricted.cpu().numpy(), letter_logprobs_full.exp().sum(-1).cpu().numpy(), residuals

    def _steering_hook(self, steering: Steering) -> Callable:
        vector = torch.as_tensor(steering.direction * steering.scale, dtype=self._model.dtype, device=self._model.device)

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + vector, *output[1:])
            return output + vector

        return hook


def _capture_hook(index: int, sink: dict[int, torch.Tensor]) -> Callable:
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        sink[index] = hidden[:, -1, :].float().cpu()

    return hook
```

**File:** `configs/health_smoke.toml`
```toml
# Tiny GPU run that checks the real adapters before the full test run.
run_id = "health-llama31-8b-smoke"
seed = 17
output_root = "../runs"

[data]
real_repo = "GBaker/MedQA-USMLE-4-options"
real_revision = "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"
real_file = "phrases_no_exclude_train.jsonl"
n_real = 60
n_fictional_entities = 20

[data.proportions]
discovery = 0.6
cal_prob = 0.1
cal_gate = 0.1
test = 0.2

[model]
name = "meta-llama/Llama-3.1-8B-Instruct"
revision = "0e9e39f249a16976918f6564b8830bc894c89659"
dtype = "bfloat16"
batch_size = 16

[gate]
target_risk = 0.10
delta = 0.05

[steering]
n_items = 12
n_random = 1
```

- [ ] **Step 2: Run the smoke test on the GPU**

Run: `HF_HOME=/home/vfeliren1/lf93_scratch2/vfvic1/hf_cache/huggingface PYTHONPATH=src python -m uncertainty_mech run --config configs/health_smoke.toml`
Expected: exit 0, `runs/health-llama31-8b-smoke/report.md` exists, summary `steering.identity_ok` is `true`, MedQA accuracy above 25%. The gate may certify nothing at this size; the report must say so.

- [ ] **Step 3: Re-run the end-to-end test**

Run: `python -m pytest -q`
Expected: `5 passed`.

- [ ] **Step 4: Commit**

```bash
git add src/uncertainty_mech/infrastructure configs/health_smoke.toml
git commit -m "Add MedQA and Hugging Face adapters with a GPU smoke config"
```

---

### Task 7: Health test run, demo questions and README

**Files:**
- Create: `configs/health_test_run.toml`, `examples/health_questions.jsonl`, `scripts/test_run.sh`, `README.md`

- [ ] **Step 1: Write the test-run inputs**

**File:** `configs/health_test_run.toml`
```toml
# Health test run: 4,000 MedQA questions plus 200 invented entities (600 questions).
run_id = "health-llama31-8b-test-run"
seed = 17
output_root = "../runs"

[data]
real_repo = "GBaker/MedQA-USMLE-4-options"
real_revision = "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"
real_file = "phrases_no_exclude_train.jsonl"
n_real = 4000
n_fictional_entities = 200

[data.proportions]
discovery = 0.6
cal_prob = 0.1
cal_gate = 0.1
test = 0.2

[model]
name = "meta-llama/Llama-3.1-8B-Instruct"
revision = "0e9e39f249a16976918f6564b8830bc894c89659"
dtype = "bfloat16"
batch_size = 16

[probe]
c_grid = [0.01, 0.1, 1.0, 10.0]
sweep_c = 0.1
cv_folds = 5

[gate]
target_risk = 0.10
delta = 0.05

[steering]
doses = [-1.0, -0.5, 0.0, 0.5, 1.0]
n_items = 200
n_random = 3
identity_tolerance = 1e-4
```

**File:** `examples/health_questions.jsonl`
```json
{"question": "A 52-year-old man with newly diagnosed type 2 diabetes has an HbA1c of 7.9% after three months of diet and exercise. His kidney function is normal. Which drug is the usual first-line treatment?", "options": ["Metformin", "Insulin glargine", "Glipizide", "Pioglitazone"]}
{"question": "A 64-year-old woman is started on Velquarin after an episode of atrial fibrillation. What is the usual adult starting dose of this drug?", "options": ["5 mg once daily", "20 mg twice daily", "150 mg twice daily", "2.5 mg once weekly"]}
{"question": "A 7-year-old boy is diagnosed with Marrowick-Tessel syndrome. Mutation of which gene causes this condition?", "options": ["FBN1", "DMD", "NF1", "COL1A1"]}
{"question": "A 30-year-old woman has fatigue, cold intolerance, weight gain and a raised TSH with low free T4. Which drug is the standard treatment?", "options": ["Propylthiouracil", "Levothyroxine", "Methimazole", "Hydrocortisone"]}
```

**File:** `scripts/test_run.sh`
```bash
#!/usr/bin/env bash
# Health test run on one GPU: end-to-end test, GPU smoke run, full test run, then demo questions.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/home/vfeliren1/lf93_scratch2/vfvic1/hf_cache/huggingface}"
export PYTHONPATH=src

python -m pytest -q
python -m uncertainty_mech run --config configs/health_smoke.toml
python -m uncertainty_mech run --config configs/health_test_run.toml
python -m uncertainty_mech ask --run-dir runs/health-llama31-8b-test-run --questions examples/health_questions.jsonl
```

- [ ] **Step 2: Run the full test run**

Run: `bash scripts/test_run.sh 2>&1 | tee runs/test_run.log` (in the background; about 20 to 40 minutes)
Expected: pytest `5 passed`, both runs exit 0, four JSON answers printed.

- [ ] **Step 3: Read the report and check it against the raw tables**

Run: `cat runs/health-llama31-8b-test-run/report.md` and `cat runs/health-llama31-8b-test-run/metrics.csv`
Expected: headline numbers match `metrics.csv`; identity check passes; figures render without clipped text (open both PNGs).

- [ ] **Step 4: Write the README from the actual results**

`README.md` covers purpose, how to run, architecture, the test-run results table copied from `report.md`, and limits. Write it after the run so every number is real, then run the no-ai-slop check on it.

- [ ] **Step 5: Commit**

```bash
git add configs examples scripts README.md
git commit -m "Add health test run config, demo questions, run script and README"
```

---

## Self-review

- Spec coverage: data roles (Task 2 splits, Task 4 dataset), signals and prompt-only baseline (Tasks 2, 4), layer sweep, fit, calibration, certificate (Tasks 3, 4), final test metrics by stratum (Task 4 evaluation), steering with identity check and random control (Task 4), report and two figures (Tasks 4, 5), inference contract and reason codes (Tasks 2, 4, 5), end-to-end test (Task 1), GPU test run (Tasks 6, 7). No gaps.
- Placeholders: README content is written in Task 7 from real numbers by design; everything else is complete code.
- Type consistency: `evaluate_on_test(dataset, chosen, errors, risks, gates, abstain_logprobs)` matches its call in `pipeline.py`; `FigureWriter.layer_sweep(rows, chosen_layer, path)` matches `MatplotlibFigures` and `write_report`; `ThresholdSelection.n_units` is set in `select_threshold` and read in `report._thresholds`.
