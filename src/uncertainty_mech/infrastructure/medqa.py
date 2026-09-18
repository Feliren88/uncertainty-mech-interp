"""MedQA-USMLE four-option questions from the Hugging Face Hub, pinned to one revision."""

from __future__ import annotations

import hashlib
import json
import re
from functools import cached_property
from typing import Any

from huggingface_hub import hf_hub_download

from uncertainty_mech.domain.questions import OPTION_LETTERS, HealthQuestion, Stratum
from uncertainty_mech.domain.splits import sample_questions

_WORD = re.compile(r"[a-z][a-z'-]+")


class MedQASource:
    def __init__(self, repo_id: str, revision: str, filename: str, n_items: int, seed: int) -> None:
        self._repo_id = repo_id
        self._revision = revision
        self._filename = filename
        self._n_items = n_items
        self._seed = seed

    def load(self) -> list[HealthQuestion]:
        """Deduplicate by normalized question text, then take a seeded sample."""
        unique: dict[str, dict[str, Any]] = {}
        for record in self._records:
            unique.setdefault(" ".join(record["question"].lower().split()), record)
        questions = [self._to_question(key, record) for key, record in unique.items()]
        return sample_questions(questions, self._n_items, self._seed)

    def corpus_words(self) -> set[str]:
        """Every word in the whole file, so invented names can avoid real ones."""
        words: set[str] = set()
        for record in self._records:
            words.update(_WORD.findall(" ".join([record["question"], *record["options"].values()]).lower()))
        return words

    @cached_property
    def _records(self) -> list[dict[str, Any]]:
        path = hf_hub_download(self._repo_id, self._filename, repo_type="dataset", revision=self._revision)
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _to_question(key: str, record: dict[str, Any]) -> HealthQuestion:
        item_id = "medqa-" + hashlib.sha256(key.encode()).hexdigest()[:16]
        return HealthQuestion(
            item_id=item_id,
            group_id=item_id,
            stratum=Stratum.REAL,
            question=record["question"].strip(),
            options=tuple(record["options"][letter] for letter in OPTION_LETTERS),
            answer_index=OPTION_LETTERS.index(record["answer_idx"]),
        )
