"""Stand-ins for the Hugging Face adapter and MedQA, so the pipeline runs on CPU in seconds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import log_softmax

from uncertainty_mech.application.ports import Readings, Steering
from uncertainty_mech.domain.prompts import ChatPrompt
from uncertainty_mech.domain.questions import OPTION_LETTERS, HealthQuestion, Stratum
from uncertainty_mech.domain.splits import sample_questions, stable_unit


class InMemoryRealSource:
    """Synthetic "real" questions with known answers, sampled from a larger pool like MedQA."""

    def __init__(self, n_items: int, seed: int) -> None:
        pool = [
            HealthQuestion(
                item_id=f"real-{i:04d}",
                group_id=f"real-{i:04d}",
                stratum=Stratum.REAL,
                question=f"Case {i}: a patient presents with finding {i}. What is the best next step?",
                options=tuple(f"Step {i}-{letter}" for letter in OPTION_LETTERS),
                answer_index=i % len(OPTION_LETTERS),
            )
            for i in range(3 * n_items)
        ]
        self._questions = sample_questions(pool, n_items, seed)

    def load(self) -> list[HealthQuestion]:
        return list(self._questions)

    def corpus_words(self) -> set[str]:
        return {"case", "patient", "finding", "step"}


class FakeLanguageModel:
    """Knows a fixed share of the real questions and nothing about invented entities.

    A known question gets a confident correct letter and a residual stream pushed
    along dimension 0 from layer 2 onward, so a probe has a real signal to find.
    """

    name = "fake-model@test"
    num_layers = 4
    d_model = 16

    def __init__(self, real_questions: Sequence[HealthQuestion], known_share: float = 0.7) -> None:
        self._answers = {q.question: q.answer_index for q in real_questions if stable_unit(q.item_id, 99) < known_share}

    def knows(self, question: str) -> bool:
        return question in self._answers

    def read(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        capture_residuals: bool = False,
        steering: Steering | None = None,
    ) -> Readings:
        rows = [self._read_one(prompt, len(letters), steering) for prompt in prompts]
        logits = np.stack([logit for logit, _, _ in rows])
        return Readings(
            letter_logprobs=log_softmax(logits, axis=1).astype(np.float32),
            letter_mass=np.array([mass for _, mass, _ in rows], dtype=np.float32),
            residuals=np.stack([residual for _, _, residual in rows]) if capture_residuals else None,
        )

    def _read_one(self, prompt: ChatPrompt, n_letters: int, steering: Steering | None):
        question = prompt.user.split("\n\n", 1)[0]
        known = question in self._answers
        rng = np.random.default_rng(int(stable_unit(question, 7) * 2**32))
        logits = rng.normal(0.0, 1.0, n_letters)
        residual = rng.normal(0.0, 1.0, (self.num_layers, self.d_model)).astype(np.float32)
        if known:
            logits[self._answers[question]] += 4.0
        residual[2:, 0] += 3.0 if known else -3.0
        if n_letters > len(OPTION_LETTERS):
            abstain = len(OPTION_LETTERS)
            logits[abstain] = -2.0 if known else 2.0
            if steering is not None:
                logits[abstain] += steering.scale * float(steering.direction[0])
        return logits, (0.95 if known else 0.6), residual
