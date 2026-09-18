"""Prompt text for the answer readout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uncertainty_mech.domain.questions import ABSTAIN_LETTER, OPTION_LETTERS

SYSTEM_PROMPT = "You are a medical expert. Answer the multiple-choice question."
ABSTAIN_OPTION = "I don't know"


@dataclass(frozen=True)
class ChatPrompt:
    system: str
    user: str


def answer_letters(allow_abstain: bool) -> tuple[str, ...]:
    return OPTION_LETTERS + (ABSTAIN_LETTER,) if allow_abstain else OPTION_LETTERS


def build_prompt(question: str, options: Sequence[str], *, allow_abstain: bool) -> ChatPrompt:
    """Question, lettered options, and an instruction to reply with one letter.

    With `allow_abstain`, option E is "I don't know". That variant is the
    prompt-only baseline and the steering read-out.
    """
    if len(options) != len(OPTION_LETTERS):
        raise ValueError(f"expected {len(OPTION_LETTERS)} options, got {len(options)}")
    letters = answer_letters(allow_abstain)
    choices = [*options, ABSTAIN_OPTION] if allow_abstain else list(options)
    lines = [question.strip(), ""]
    lines += [f"{letter}. {text}" for letter, text in zip(letters, choices, strict=True)]
    lines.append("")
    if allow_abstain:
        lines.append(f"If you do not know the answer, choose {ABSTAIN_LETTER}.")
    lines.append(f"Reply with the letter only ({', '.join(letters)}).")
    return ChatPrompt(system=SYSTEM_PROMPT, user="\n".join(lines))
