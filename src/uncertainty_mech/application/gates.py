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
