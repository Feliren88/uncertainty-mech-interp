"""Invented drugs and diseases: questions whose only right answer is "I don't know.".

Each entity gets three clinical questions in MedQA style. The options are real
drugs, genes, organisms and so on, so a fluent guess looks plausible.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from uncertainty_mech.domain.questions import HealthQuestion, Stratum
from uncertainty_mech.infrastructure.health_templates import (
    DISEASE,
    DRUG,
    POOLS,
    TEMPLATES,
    invent_names,
    patient,
    vignette,
)

_TEMPLATES_PER_ENTITY = 3


class FictionalHealthSource:
    def __init__(self, n_entities: int, seed: int, exclude_words: Iterable[str] = ()) -> None:
        self._n_entities = n_entities
        self._seed = seed
        self._exclude = tuple(exclude_words)

    def load(self) -> list[HealthQuestion]:
        rng = random.Random(self._seed)
        kinds = [DRUG if i % 2 == 0 else DISEASE for i in range(self._n_entities)]
        questions = []
        for name in invent_names(rng, kinds, self._exclude):
            templates = [template for template in TEMPLATES.values() if template.kind == name.kind]
            for template in rng.sample(templates, _TEMPLATES_PER_ENTITY):
                opening = vignette(patient(rng), name.display, name.kind)
                questions.append(
                    HealthQuestion(
                        item_id=f"fict-{name.slug}-{template.key}",
                        group_id=f"fict-{name.slug}",
                        stratum=Stratum.FICTIONAL,
                        question=f"{opening} {template.question}",
                        options=tuple(rng.sample(POOLS[template.pool], 4)),
                        answer_index=None,
                    )
                )
        return questions
