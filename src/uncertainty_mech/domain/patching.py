"""Arithmetic for activation-patching results."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def restoration(patched: np.ndarray, target: np.ndarray, source: np.ndarray) -> float | None:
    """Share of the source-minus-target gap that a patch moves, averaged over pairs before dividing.

    0 means the patch changed nothing; 1 means the patched target now behaves
    like the source. None when the gap is too small to divide by (RFC section 3:
    report raw effects instead).
    """
    gap = float(np.mean(source) - np.mean(target))
    if abs(gap) < 1e-6:
        return None
    return float((np.mean(patched) - np.mean(target)) / gap)


def smallest_sufficient_k(restoration_by_k: Mapping[int, float | None], threshold: float) -> int:
    """Smallest k whose restoration reaches the threshold, else the largest k tried."""
    passing = [k for k, value in restoration_by_k.items() if value is not None and value >= threshold]
    return min(passing) if passing else max(restoration_by_k)
