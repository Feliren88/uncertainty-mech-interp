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
