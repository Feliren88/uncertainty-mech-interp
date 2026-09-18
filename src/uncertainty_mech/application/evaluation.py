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
            **asdict(
                selective_metrics(candidate_errors[method][mask], kept[mask], int(originally_correct[mask].sum()))
            ),
        }
        for method, kept in released.items()
        for stratum, mask in subsets.items()
        if mask.any()
    ]
    quality_rows = [
        {
            "signal": signal.value,
            "stratum": stratum,
            **asdict(risk_quality(risks[signal][subsets[stratum]], errors[subsets[stratum]])),
        }
        for signal in gates
        for stratum in ("all", "real")
        if subsets[stratum].any()
    ]

    n_test = int(test.sum())
    prompt_kept = released[PROMPT_ONLY][test]
    prompt_coverage = float(prompt_kept.mean())
    prompt_risk = float(prompt_errors[test][prompt_kept].mean()) if prompt_kept.any() else None
    k = int(round(prompt_coverage * n_test))
    matched_rows = [
        {
            "method": PROMPT_ONLY,
            "answered": int(prompt_kept.sum()),
            "coverage": prompt_coverage,
            "selective_risk": prompt_risk,
        }
    ]
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
