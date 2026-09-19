"""Builds matched real/invented pairs from the curated facts and the shared templates."""

from __future__ import annotations

import random
from collections.abc import Iterable

from uncertainty_mech.application.entity_pairs import EntityPair
from uncertainty_mech.domain.questions import HealthQuestion, Stratum
from uncertainty_mech.infrastructure.health_templates import POOLS, TEMPLATES, invent_names, patient, vignette
from uncertainty_mech.infrastructure.known_facts import KNOWN_FACTS


class EntityPairSource:
    def __init__(
        self,
        names_per_fact: int,
        seed: int,
        exclude_words: Iterable[str] = (),
        facts: tuple[tuple[str, str, str], ...] = KNOWN_FACTS,
    ) -> None:
        self._names_per_fact = names_per_fact
        self._seed = seed
        self._exclude = tuple(exclude_words)
        self._facts = facts

    def load(self) -> list[EntityPair]:
        rng = random.Random(self._seed)
        kinds = [TEMPLATES[key].kind for key, _, _ in self._facts for _ in range(self._names_per_fact)]
        names = iter(invent_names(rng, kinds, self._exclude))
        pairs = []
        for fact_index, (key, entity, correct) in enumerate(self._facts):
            template = TEMPLATES[key]
            if correct not in POOLS[template.pool]:
                raise ValueError(f"{entity}: {correct!r} is not in the {template.pool} pool")
            entity_slug = entity.lower().replace(" ", "-")
            for copy in range(self._names_per_fact):
                name = next(names)
                options = [correct, *rng.sample([o for o in POOLS[template.pool] if o != correct], 3)]
                rng.shuffle(options)
                person = patient(rng)
                pair_id = f"pair-{fact_index:03d}-{copy}"
                pairs.append(
                    EntityPair(
                        pair_id=pair_id,
                        entity_group=entity_slug,
                        real=HealthQuestion(
                            item_id=f"{pair_id}-real",
                            group_id=entity_slug,
                            stratum=Stratum.REAL,
                            question=f"{vignette(person, entity, template.kind)} {template.question}",
                            options=tuple(options),
                            answer_index=options.index(correct),
                        ),
                        invented=HealthQuestion(
                            item_id=f"{pair_id}-invented",
                            group_id=f"fict-{name.slug}",
                            stratum=Stratum.FICTIONAL,
                            question=f"{vignette(person, name.display, template.kind)} {template.question}",
                            options=tuple(options),
                            answer_index=None,
                        ),
                    )
                )
        return pairs
