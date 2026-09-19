"""Number and table formatting shared by the markdown reports."""

from __future__ import annotations

import math


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def pct(value: float | None) -> str:
    return "n/a" if _missing(value) else f"{100 * value:.1f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if _missing(value) else f"{value:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
