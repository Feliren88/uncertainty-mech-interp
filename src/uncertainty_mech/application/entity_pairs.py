"""Matched pairs: the same question about a real entity and about an invented one."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uncertainty_mech.domain.questions import HealthQuestion
from uncertainty_mech.domain.splits import stable_unit


@dataclass(frozen=True)
class EntityPair:
    """Two prompts that differ only in the entity name. Same vignette, same options, same order."""

    pair_id: str
    entity_group: str  # the real entity; pairs sharing it stay on one side of the split
    real: HealthQuestion
    invented: HealthQuestion


def split_pairs(
    pairs: Sequence[EntityPair], discovery_share: float, seed: int
) -> tuple[list[EntityPair], list[EntityPair]]:
    """Discovery and test pairs, split by real entity so no test entity helped find the circuit."""
    discovery = [pair for pair in pairs if stable_unit(f"pairs:{pair.entity_group}", seed) < discovery_share]
    test = [pair for pair in pairs if stable_unit(f"pairs:{pair.entity_group}", seed) >= discovery_share]
    return discovery, test
