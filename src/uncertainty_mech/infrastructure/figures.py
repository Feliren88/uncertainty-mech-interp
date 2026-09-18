"""Figures in the house style: serif type, no titles, sentence case, legends outside the axes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

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
            ax.fill_between(
                layers,
                mean - spread,
                mean + spread,
                color=_COLORS["probe"],
                alpha=0.18,
                linewidth=0,
                label="One fold SD",
            )
            ax.plot(layers, mean, color=_COLORS["probe"], marker="o", markersize=3, label="Mean over 5 grouped folds")
            ax.axvline(
                chosen_layer,
                color=_COLORS["reference"],
                linestyle="--",
                linewidth=1,
                label=f"Chosen layer ({chosen_layer})",
            )
            ax.axhline(0.5, color=_COLORS["chance"], linewidth=0.8, label="Chance")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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
                ax.plot(
                    [coverage],
                    [risk],
                    marker="D",
                    linestyle="none",
                    color=_COLORS["prompt_only"],
                    label=_LABELS["prompt_only"],
                )
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
