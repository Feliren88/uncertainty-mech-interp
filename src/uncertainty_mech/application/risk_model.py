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


def sweep_layers(
    residuals: np.ndarray, errors: np.ndarray, groups: np.ndarray, c: float, folds: int
) -> list[LayerScore]:
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
