"""House style for matplotlib figures.

Drop this next to your analysis code and import it. Importing applies the
rcParams, so a figure made without going through `finish()` still gets the type
and the spines right.

    import chartstyle as cs

    fig, ax = cs.new_fig(7.2, 4.2)
    ax.bar(xs, vals, color=cs.PALETTE["a"])
    cs.label_above(ax, xs, vals, hi)
    cs.legend_outside(ax, title="condition")
    cs.finish(fig, ax,
              message="Refusal collapses and abstention does not",
              xlabel="", ylabel="rate on held-out items (Wilson 95%)",
              path="figures/F3_ablation.png")

The one thing to remember: `finish()` does the sentence-casing centrally, at
save time, by walking every text artist on the axes. Doing it at each call site
means the figure you add next month quietly misses it.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times",
         "DejaVu Serif"]

# One colour, one meaning, held across every figure in a document.
# Colourblind-safe. Rename the keys for your domain, keep the mapping fixed.
PALETTE: Dict[str, str] = {
    "a": "#1b6ca8",       # blue
    "b": "#c0392b",       # red
    "c": "#2e8b57",       # green
    "d": "#7f7f7f",       # grey, for the neutral or baseline series
    "e": "#9467bd",       # purple
    "f": "#d4820a",       # amber
    "null": "#b0b0b0",    # noise bands
    "ref": "#444444",     # reference lines, ceilings
}

CAPTIONS: Dict[str, Dict[str, str]] = {}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


apply_style()


def new_fig(w: float = 7.2, h: float = 4.2) -> Tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(w, h))


# ------------------------------------------------------------------ rule 6

def cap(s: str) -> str:
    """Uppercase the first letter, leave everything else alone.

    Leaves a label starting with a digit untouched, so "95% interval" and
    "2024 cohort" survive. If a label starts with a maths token, rewrite the
    label so a word comes first rather than reaching for an exception here.
    """
    if not s:
        return s
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
        if ch.isdigit():
            return s
    return s


def _capitalise_all(ax: plt.Axes, cbar=None) -> None:
    ax.set_xlabel(cap(ax.get_xlabel()))
    ax.set_ylabel(cap(ax.get_ylabel()))
    # Only touch tick labels when the locator is fixed. Calling
    # set_ticklabels on an automatic locator warns and can silently
    # misalign labels against ticks.
    for axis, setter in ((ax.xaxis, ax.set_xticklabels),
                         (ax.yaxis, ax.set_yticklabels)):
        if isinstance(axis.get_major_locator(), mticker.FixedLocator):
            setter([cap(t.get_text()) for t in axis.get_ticklabels()])
    for t in ax.texts:
        t.set_text(cap(t.get_text()))
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_text(cap(t.get_text()))
        if leg.get_title() is not None:
            leg.set_title(cap(leg.get_title().get_text()))
    if cbar is not None:
        cbar.set_label(cap(cbar.ax.get_ylabel()))


# ------------------------------------------------------------------ rule 2

def legend_outside(ax: plt.Axes, handles=None, labels=None,
                   title: Optional[str] = None, ncol: int = 1) -> None:
    """Park the legend to the right of the axes so it can never cover data."""
    kw = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              borderaxespad=0.0, ncol=ncol, title=title,
              handlelength=1.6, labelspacing=0.6)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


# ------------------------------------------------------------------ rule 9

def label_above(ax: plt.Axes, xs: Sequence[float], values: Sequence[float],
                tops: Optional[Sequence[float]] = None, fmt: str = "{:.2f}",
                pad: float = 0.03, fontsize: float = 10.0) -> None:
    """Value labels that clear the error bars.

    Pass `tops` as the upper end of each interval. Passing the bar heights
    instead puts the number on the whisker.
    """
    tops = list(values) if tops is None else list(tops)
    for x, v, t in zip(xs, values, tops):
        ax.text(x, t + pad, fmt.format(v), ha="center", fontsize=fontsize)


# ------------------------------------------------------------------ rule 8

def mark_unreliable(ax: plt.Axes, x: Sequence[float], mask: Sequence[bool],
                    note: str, colour: str = "#f0e3c8", y: float = 0.93) -> None:
    """Shade and label a region where the quantity is not estimable.

    Use this instead of leaving NaNs to break a line. A reader cannot tell a
    gap caused by missing data from a gap caused by a plotting bug, so say
    which it is.
    """
    x = np.asarray(x)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return
    lo, hi = float(x[m].min()), float(x[m].max())
    ax.axvspan(lo - 0.5, hi + 0.5, color=colour, alpha=0.55, zorder=0)
    ax.text((lo + hi) / 2, y, note, ha="center", va="top", fontsize=9,
            color="#6b5a2e", transform=ax.get_xaxis_transform())


# ------------------------------------------------------------------ rules 4, 11

def finish(fig: plt.Figure, ax: plt.Axes, message: str, xlabel: str,
           ylabel: str, path: str, cbar=None, context: str = "",
           captions_path: Optional[str] = None) -> str:
    """Save with no title on the plot, and record the message for the caption.

    `message` is the single claim this chart makes. It is deliberately not
    drawn on the axes: it belongs in the caption next to the figure, where
    there is room for the interval, the n and the caveat. It is written to
    captions.json so the prose and the picture cannot drift apart.

    `context` is the dataset, model or condition, and also namespaces nothing
    by itself: give `path` a distinguishing suffix when more than one data
    source writes figures, or the second run overwrites the first.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _capitalise_all(ax, cbar)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    # bbox_inches="tight" is what stops the outside legend being cropped off.
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    name = os.path.splitext(os.path.basename(path))[0]
    CAPTIONS[name] = {"message": message, "context": context, "path": path}
    cp = captions_path or os.path.join(d or ".", "captions.json")
    with open(cp, "w") as fh:
        json.dump(CAPTIONS, fh, indent=2)
    return path


def caption_markdown(name: str) -> str:
    """The figure and its caption, ready to paste into a document."""
    c = CAPTIONS[name]
    line = c["message"].rstrip()
    if not line.endswith((".", "!", "?")):
        line += "."
    ctx = c["context"].rstrip()
    if ctx:
        line += " " + ctx + ("" if ctx.endswith((".", "!", "?")) else ".")
    return f'![{name}]({c["path"]})\n\n*{line}*'
