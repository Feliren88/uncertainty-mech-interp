"""Everything one run produced, handed from the pipeline to the report writers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from uncertainty_mech.application.config import RunConfig
from uncertainty_mech.application.dataset import Dataset
from uncertainty_mech.application.evaluation import Evaluation
from uncertainty_mech.application.gates import FittedGate
from uncertainty_mech.application.risk_model import LayerScore, SignalSet
from uncertainty_mech.application.steering import SteeringResult
from uncertainty_mech.domain.questions import Role, Stratum


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

    @property
    def best_layer_score(self) -> LayerScore:
        return next(score for score in self.sweep if score.layer == self.layer)

    def counts(self) -> dict[str, dict[str, int]]:
        """Items per role and stratum."""
        return {
            role.value: {
                stratum.value: int((self.dataset.mask(role) & (self.dataset.strata == stratum.value)).sum())
                for stratum in Stratum
            }
            for role in Role
        }
