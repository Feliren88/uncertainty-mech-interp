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
