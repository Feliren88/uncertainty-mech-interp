"""Tiered steering: several semantic-entropy thresholds, each with a stronger push on the circuit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncertainty_mech.domain.questions import OPTION_LETTERS

ABSTAIN = len(OPTION_LETTERS)  # index of option E


@dataclass(frozen=True)
class DoseSchedule:
    """Dose `doses[k]` applies when semantic entropy exceeds `thresholds[k]` (the highest such k).

    Optionally, when the circuit's own readout exceeds `readout_threshold`, the
    dose rises to at least `readout_dose`. Below every threshold the dose is 0.
    """

    thresholds: tuple[float, ...]
    doses: tuple[float, ...]
    readout_threshold: float | None = None
    readout_dose: float = 0.0

    def __post_init__(self) -> None:
        if len(self.thresholds) != len(self.doses) or not self.thresholds:
            raise ValueError("a schedule needs one dose per threshold, and at least one threshold")
        if any(b <= a for a, b in zip(self.thresholds, self.thresholds[1:], strict=False)):
            raise ValueError(f"thresholds must increase: {self.thresholds}")
        if any(b < a for a, b in zip(self.doses, self.doses[1:], strict=False)) or self.doses[0] <= 0:
            raise ValueError(f"doses must be positive and non-decreasing: {self.doses}")

    @property
    def tiers(self) -> int:
        return len(self.thresholds)

    def doses_for(self, entropy: np.ndarray, readout: np.ndarray | None = None) -> np.ndarray:
        tier = np.searchsorted(np.asarray(self.thresholds), entropy, side="left")  # thresholds strictly below SE
        dose = np.concatenate([[0.0], self.doses])[tier]
        if self.readout_threshold is not None:
            if readout is None:
                raise ValueError("this schedule reads the circuit readout")
            dose = np.where(readout > self.readout_threshold, np.maximum(dose, self.readout_dose), dose)
        return dose

    def describe(self) -> str:
        steps = ", ".join(f"SE > {t:.1f}: dose {d:g}" for t, d in zip(self.thresholds, self.doses, strict=True))
        if self.readout_threshold is not None:
            steps += f"; readout > {self.readout_threshold:.2f}: dose at least {self.readout_dose:g}"
        return steps


def utility(choice: np.ndarray, answers: np.ndarray, wrong_cost: float) -> float:
    """(right answers - wrong_cost * wrong answers) per question. "I don't know" scores 0."""
    answered = choice != ABSTAIN
    right = answered & (choice == answers)
    wrong = answered & ~right
    return float((right.sum() - wrong_cost * wrong.sum()) / len(choice))


def abstention_profile(choice: np.ndarray, answers: np.ndarray, forced_correct: np.ndarray) -> dict[str, float]:
    """Abstention where the model does not know versus where it does.

    Unknown: invented entities (answer -1) and questions the model gets wrong
    when forced to choose. Known: questions it gets right when forced to choose.
    """
    abstained = choice == ABSTAIN
    unknown = (answers < 0) | ~forced_correct
    known = ~unknown
    answered = ~abstained
    wrong = answered & (choice != answers)
    return {
        "abstain_on_unknown": float(abstained[unknown].mean()) if unknown.any() else float("nan"),
        "abstain_on_known": float(abstained[known].mean()) if known.any() else float("nan"),
        "abstain_on_invented": float(abstained[answers < 0].mean()) if (answers < 0).any() else float("nan"),
        "answered_share": float(answered.mean()),
        "wrong_among_answered": float(wrong.sum() / answered.sum()) if answered.any() else float("nan"),
    }
