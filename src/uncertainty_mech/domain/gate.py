"""Release decision and the only text a user sees (protocol section 5)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

ABSTAIN_TEXT = "I don't know."


class Decision(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


class ReasonCode(StrEnum):
    ANSWERED = "ANSWERED"
    HIGH_RISK = "HIGH_RISK"
    NO_CERTIFIED_THRESHOLD = "NO_CERTIFIED_THRESHOLD"
    INVALID_SCORE = "INVALID_SCORE"


@dataclass(frozen=True)
class GateResult:
    decision: Decision
    reason_code: ReasonCode


def decide(risk: float | None, threshold: float | None) -> GateResult:
    """Answer only with a certified threshold and a finite risk at or below it."""
    if threshold is None:
        return GateResult(Decision.ABSTAIN, ReasonCode.NO_CERTIFIED_THRESHOLD)
    if risk is None or not math.isfinite(risk):
        return GateResult(Decision.ABSTAIN, ReasonCode.INVALID_SCORE)
    if risk <= threshold:
        return GateResult(Decision.ANSWER, ReasonCode.ANSWERED)
    return GateResult(Decision.ABSTAIN, ReasonCode.HIGH_RISK)


def render(result: GateResult, letter: str, option_text: str) -> str:
    return f"{letter}. {option_text}" if result.decision is Decision.ANSWER else ABSTAIN_TEXT
