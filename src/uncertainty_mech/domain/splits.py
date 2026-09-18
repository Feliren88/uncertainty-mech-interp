"""Seeded assignment of question groups to data roles."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from uncertainty_mech.domain.questions import HealthQuestion, Role


def stable_unit(key: str, seed: int) -> float:
    """Map a key to [0, 1) by hashing, so assignments never depend on row order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_roles(questions: Sequence[HealthQuestion], proportions: Mapping[Role, float], seed: int) -> list[Role]:
    """Give every item the role of its group, so related items never straddle roles."""
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError(f"role proportions must sum to 1, got {dict(proportions)}")
    cut_points: list[tuple[float, Role]] = []
    total = 0.0
    for role in Role:
        total += proportions.get(role, 0.0)
        cut_points.append((total, role))

    def role_of(group_id: str) -> Role:
        position = stable_unit(group_id, seed)
        return next((role for cut, role in cut_points if position < cut), cut_points[-1][1])

    return [role_of(question.group_id) for question in questions]


def canonical_mask(questions: Sequence[HealthQuestion], seed: int) -> list[bool]:
    """Pick one item per group for the binomial certificate, before any model output exists."""
    best: dict[str, tuple[float, str]] = {}
    for question in questions:
        key = (stable_unit(question.item_id, seed + 1), question.item_id)
        if question.group_id not in best or key < best[question.group_id]:
            best[question.group_id] = key
    chosen = {item_id for _, item_id in best.values()}
    return [question.item_id in chosen for question in questions]
