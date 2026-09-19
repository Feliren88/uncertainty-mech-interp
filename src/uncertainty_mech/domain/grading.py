"""Error labels and output-only features from the answer-letter distribution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

from uncertainty_mech.domain.questions import OPTION_LETTERS, HealthQuestion

OUTPUT_FEATURE_NAMES = ("top_prob", "top_margin", "letter_entropy", "log_letter_mass")


def chosen_indices(letter_logprobs: np.ndarray) -> np.ndarray:
    """Index of the most likely offered letter in each row."""
    return np.argmax(letter_logprobs, axis=1)


def error_labels(questions: Sequence[HealthQuestion], chosen: np.ndarray) -> np.ndarray:
    """1 when the chosen option is wrong, or when no option is supported at all."""
    return np.array(
        [1 if q.answer_index is None else int(c != q.answer_index) for q, c in zip(questions, chosen, strict=True)],
        dtype=np.int64,
    )


def output_features(letter_logprobs: np.ndarray, letter_mass: np.ndarray) -> np.ndarray:
    """Top probability, top-two margin, entropy over the letters, and log letter mass.

    For multiple choice the answer meanings are the letters, so this entropy is
    the semantic entropy of protocol section 4 without sampling.
    """
    probs = np.exp(letter_logprobs)
    ranked = -np.sort(-probs, axis=1)
    entropy = semantic_entropy(letter_logprobs)
    log_mass = np.log(np.clip(letter_mass, 1e-12, None))
    return np.column_stack([ranked[:, 0], ranked[:, 0] - ranked[:, 1], entropy, log_mass]).astype(np.float32)


def semantic_entropy(letter_logprobs: np.ndarray, n_answers: int = len(OPTION_LETTERS)) -> np.ndarray:
    """Entropy (nats) over the answer options A to D, renormalized if option E is also offered.

    In multiple choice each letter is one meaning, so this is the semantic
    entropy of protocol section 4, computed exactly instead of by sampling.
    """
    answers = np.asarray(letter_logprobs, dtype=np.float64)[:, :n_answers]
    logp = answers - logsumexp(answers, axis=1, keepdims=True)
    return -np.sum(np.where(np.isfinite(logp), np.exp(logp) * logp, 0.0), axis=1)


def abstention_gap(letter_logprobs: np.ndarray, n_answers: int = len(OPTION_LETTERS)) -> np.ndarray:
    """log P(E) - log P(any of A to D) on the prompt that offers E. Above 0: the model prefers "I don't know"."""
    logp = np.asarray(letter_logprobs, dtype=np.float64)
    return logp[:, n_answers] - logsumexp(logp[:, :n_answers], axis=1)
