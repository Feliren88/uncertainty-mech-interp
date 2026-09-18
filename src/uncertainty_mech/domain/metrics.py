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
