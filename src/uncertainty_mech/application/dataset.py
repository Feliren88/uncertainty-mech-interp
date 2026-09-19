"""Stage 1 and 2: questions with their roles, and one model read per question."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from uncertainty_mech.application.config import RunConfig
from uncertainty_mech.application.ports import LanguageModel, Readings
from uncertainty_mech.domain.prompts import answer_letters, build_prompt
from uncertainty_mech.domain.questions import HealthQuestion, Role, Stratum
from uncertainty_mech.domain.splits import assign_roles, canonical_mask


@dataclass(frozen=True)
class Dataset:
    questions: tuple[HealthQuestion, ...]
    roles: np.ndarray  # role value per item
    canonical: np.ndarray  # True for the one certification unit per group

    @cached_property
    def groups(self) -> np.ndarray:
        return np.array([q.group_id for q in self.questions])

    @cached_property
    def strata(self) -> np.ndarray:
        return np.array([q.stratum.value for q in self.questions])

    def mask(self, role: Role) -> np.ndarray:
        return self.roles == role.value

    def item_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "item_id": q.item_id,
                "group_id": q.group_id,
                "stratum": q.stratum.value,
                "role": role,
                "canonical": bool(canonical),
                "answer_index": q.answer_index,
                "question": q.question,
                "options_json": json.dumps(list(q.options)),
            }
            for q, role, canonical in zip(self.questions, self.roles, self.canonical, strict=True)
        ]


def prepare_dataset(questions: Sequence[HealthQuestion], config: RunConfig) -> Dataset:
    ids = [q.item_id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique")
    roles = assign_roles(questions, config.data.proportions, config.seed)
    empty = [role.value for role, share in config.data.proportions.items() if share > 0 and role not in roles]
    if empty:
        raise ValueError(f"no items were assigned to roles {empty}; increase the sample or check the sampler")
    return Dataset(
        questions=tuple(questions),
        roles=np.array([role.value for role in roles]),
        canonical=np.array(canonical_mask(questions, config.seed)),
    )


def read_questions(
    model: LanguageModel, questions: Sequence[HealthQuestion], *, allow_abstain: bool, capture_residuals: bool = False
) -> Readings:
    prompts = [build_prompt(q.question, q.options, allow_abstain=allow_abstain) for q in questions]
    return model.read(prompts, answer_letters(allow_abstain), capture_residuals=capture_residuals)


def dataset_from_rows(rows: Sequence[dict[str, str]]) -> Dataset:
    """Rebuild a finished run's dataset from its `items.csv`."""
    questions = tuple(
        HealthQuestion(
            item_id=row["item_id"],
            group_id=row["group_id"],
            stratum=Stratum(row["stratum"]),
            question=row["question"],
            options=tuple(json.loads(row["options_json"])),
            answer_index=None if row["answer_index"] == "" else int(row["answer_index"]),
        )
        for row in rows
    )
    return Dataset(
        questions=questions,
        roles=np.array([row["role"] for row in rows]),
        canonical=np.array([row["canonical"] == "True" for row in rows]),
    )
