"""Health questions, their strata and their data roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

OPTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
ABSTAIN_LETTER = "E"


class Stratum(StrEnum):
    REAL = "real"
    FICTIONAL = "fictional"


class Role(StrEnum):
    DISCOVERY = "discovery"
    CAL_PROB = "cal_prob"
    CAL_GATE = "cal_gate"
    TEST = "test"


@dataclass(frozen=True)
class HealthQuestion:
    """One multiple-choice item. `answer_index` is None when no option is supported."""

    item_id: str
    group_id: str
    stratum: Stratum
    question: str
    options: tuple[str, ...]
    answer_index: int | None

    def __post_init__(self) -> None:
        if len(self.options) != len(OPTION_LETTERS):
            raise ValueError(f"{self.item_id}: expected {len(OPTION_LETTERS)} options, got {len(self.options)}")
        if (self.answer_index is not None) != (self.stratum is Stratum.REAL):
            raise ValueError(f"{self.item_id}: real items need an answer and fictional items must have none")
