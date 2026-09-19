"""Stand-ins for the Hugging Face adapter and MedQA, so the pipeline runs on CPU in seconds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import log_softmax

from uncertainty_mech.application.ports import Addition, Capture, Readings, Site, SiteKind, Steering, Trace
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


class FakeCircuitModel:
    """A hand-wired four-layer "transformer" with one planted abstention head.

    At the final token, layer 1's MLP writes entity familiarity on dimension 0:
    +1 for any real entity (known answer or not), -1 for an invented one. Head 1
    of layer 2 copies residual dimensions 0 to 3 into its output, which the
    identity o_proj places on dimensions 4 to 7. The E logit reads dimension 4,
    and the model is less sure among A to D when dimension 4 is low. Everything
    else is small fixed noise, so patching it changes nothing. Like Llama, it
    abstains on invented entities but answers real questions it does not know.
    """

    name = "fake-model@test"
    num_layers = 4
    num_heads = 4
    head_dim = 4
    d_model = 16
    planted_head = (2, 1)

    def __init__(self, known_answers: dict[str, int], facts: Sequence[tuple[str, str, str]]) -> None:
        self._known_answers = known_answers  # question text -> correct index, for real questions it knows
        self._facts = {entity: correct for _, entity, correct in facts}

    def read(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        capture_residuals: bool = False,
        steering: Steering | None = None,
    ) -> Readings:
        captures = [Capture(Site(SiteKind.RESID, layer), (1,)) for layer in range(self.num_layers)]
        additions = (
            []
            if steering is None
            else [
                Addition(
                    Site(SiteKind.RESID, steering.layer),
                    None,
                    steering.direction,
                    np.full(len(prompts), steering.scale),
                )
            ]
        )
        trace = self.trace(prompts, letters, captures=captures if capture_residuals else (), additions=additions)
        residuals = np.stack([values[:, 0] for values in trace.activations], axis=1) if capture_residuals else None
        return Readings(trace.letter_logprobs, trace.letter_mass, residuals)

    def common_suffix_length(self, prompts: Sequence[ChatPrompt]) -> int:
        token_lists = [self._tokens(prompt) for prompt in prompts]
        length = 0
        while length < min(map(len, token_lists)) and len({tokens[-1 - length] for tokens in token_lists}) == 1:
            length += 1
        return length

    def trace(self, prompts, letters, *, captures=(), patches=(), additions=()) -> Trace:
        activations = [
            np.zeros((len(prompts), _count(c.offsets), self._width(c.site)), dtype=np.float32) for c in captures
        ]
        rows = [
            self._forward(row, prompt, len(letters), captures, patches, additions, activations)
            for row, prompt in enumerate(prompts)
        ]
        logits = np.stack([logit for logit, _ in rows])
        return Trace(log_softmax(logits, axis=1), np.array([mass for _, mass in rows]), activations)

    def _tokens(self, prompt: ChatPrompt) -> list[str]:
        return prompt.system.split() + prompt.user.split() + ["<eot>", "<assistant>"]

    def _width(self, site: Site) -> int:
        if site.kind is SiteKind.ATTN_Z:
            return self.head_dim if site.head is not None else self.num_heads * self.head_dim
        return self.d_model

    def _recall(self, prompt: ChatPrompt) -> tuple[bool, int | None]:
        """(the entity is real, index of the right option if the model knows it)."""
        question = prompt.user.split("\n\n", 1)[0]
        if question in self._known_answers:
            return True, self._known_answers[question]
        if question.startswith("Case "):
            return True, None
        options = [line[3:] for line in prompt.user.splitlines() if line[:3] in {"A. ", "B. ", "C. ", "D. "}]
        for entity, correct in self._facts.items():
            if f" {entity}." in question and correct in options:
                return True, options.index(correct)
        return False, None

    def _forward(self, row, prompt, n_letters, captures, patches, additions, activations):
        tokens = self._tokens(prompt)
        length = len(tokens)
        real, answer = self._recall(prompt)
        rng = np.random.default_rng(int(stable_unit(prompt.user.split("\n\n", 1)[0], 11) * 2**32))

        def edit(kind: SiteKind, layer: int, values: np.ndarray) -> np.ndarray:
            for patch in patches:
                if (patch.site.kind, patch.site.layer) == (kind, layer):
                    values[self._positions(patch.offsets, row, length), self._columns(patch.site)] = patch.values[row]
            for addition in additions:
                if (addition.site.kind, addition.site.layer) == (kind, layer):
                    positions = (
                        slice(None) if addition.offsets is None else self._positions(addition.offsets, row, length)
                    )
                    values[positions, self._columns(addition.site)] += addition.scale[row] * addition.vector
            for index, capture in enumerate(captures):
                if (capture.site.kind, capture.site.layer) == (kind, layer):
                    activations[index][row] = values[
                        self._positions(capture.offsets, row, length), self._columns(capture.site)
                    ]
            return values

        residual = rng.normal(0.0, 0.1, (length, self.d_model))
        for layer in range(self.num_layers):
            z = rng.normal(0.0, 0.1, (length, self.num_heads * self.head_dim))
            if (layer, 1) == self.planted_head:
                z[-1, 4:8] = residual[-1, 0:4]
            z = edit(SiteKind.ATTN_Z, layer, z)
            mlp = rng.normal(0.0, 0.1, (length, self.d_model))
            if layer == 1:
                mlp[-1, 0] += 1.0 if real else -1.0
            mlp = edit(SiteKind.MLP, layer, mlp)
            residual = edit(SiteKind.RESID, layer, residual + z + mlp)

        familiarity = residual[-1, 4]
        answers = rng.normal(0.0, 1.0, len(OPTION_LETTERS))
        if answer is not None:
            answers[answer] += 4.0
        answers *= 0.3 + 1.0 / (1.0 + np.exp(-4.0 * familiarity))
        logits = answers if n_letters == len(OPTION_LETTERS) else np.append(answers, -3.0 * familiarity)
        return logits, (0.95 if real else 0.6)

    @staticmethod
    def _positions(offsets, row: int, length: int) -> np.ndarray:
        per_row = offsets[row] if isinstance(offsets, np.ndarray) else np.asarray(offsets)
        return length - per_row

    def _columns(self, site: Site) -> slice:
        if site.head is None:
            return slice(None)
        return slice(site.head * self.head_dim, (site.head + 1) * self.head_dim)


def _count(offsets) -> int:
    return offsets.shape[1] if isinstance(offsets, np.ndarray) else len(offsets)
