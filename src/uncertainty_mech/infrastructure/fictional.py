"""Invented drugs and diseases: questions whose only right answer is "I don't know.".

Each entity gets three clinical questions in MedQA style. The options are real
drugs, genes, organisms and so on, so a fluent guess looks plausible.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from uncertainty_mech.domain.questions import HealthQuestion, Stratum

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kl", "tr", "gr", "st")
_VOWELS = ("a", "e", "i", "o", "u")
_CODAS = ("l", "n", "r", "s", "v", "x", "m", "d", "k", "th")
_DRUG_ENDINGS = ("axil", "ovane", "urex", "endra", "ivon", "omyr", "alith", "eprax")
_EPONYM_ENDINGS = ("nick", "dorf", "holm", "ley", "sen", "ward", "berg", "ton")
_DISEASE_FORMS = ("{} syndrome", "{} disease")
# Real drug-class stems. An invented name must not hint at a real class.
_BANNED_FRAGMENTS = (
    "olol", "pril", "sartan", "statin", "dipine", "mab", "nib", "azole", "cillin", "mycin", "cycline", "terol",
    "prazole", "tidine", "vir", "parin", "gliptin", "gliflozin", "afil", "triptan", "setron", "oxetine", "caine",
    "floxacin", "sone", "lone", "pam", "lam",
)
_TEMPLATES_PER_ENTITY = 3

_POOLS: dict[str, tuple[str, ...]] = {
    "drugs": (
        "Metformin", "Lisinopril", "Amoxicillin", "Prednisone", "Levothyroxine", "Methotrexate",
        "Hydroxychloroquine", "Allopurinol", "Warfarin", "Omeprazole", "Azithromycin", "Furosemide",
    ),
    "genes": ("CFTR", "FBN1", "BRCA1", "NF1", "HBB", "DMD", "COL1A1", "PKD1", "HTT", "FMR1", "APC", "RET"),
    "organisms": (
        "Staphylococcus aureus", "Streptococcus pneumoniae", "Escherichia coli", "Borrelia burgdorferi",
        "Mycobacterium tuberculosis", "Plasmodium falciparum", "Treponema pallidum", "Clostridioides difficile",
        "Listeria monocytogenes", "Coxiella burnetii",
    ),
    "enzymes": (
        "Glucose-6-phosphate dehydrogenase", "Hexosaminidase A", "Glucocerebrosidase", "Phenylalanine hydroxylase",
        "Alpha-galactosidase A", "21-hydroxylase", "Ornithine transcarbamylase", "Adenosine deaminase",
        "Homogentisate oxidase", "Acid alpha-glucosidase",
    ),
    "mechanisms": (
        "Inhibition of angiotensin-converting enzyme", "Blockade of beta-1 adrenergic receptors",
        "Inhibition of HMG-CoA reductase", "Irreversible inhibition of the H+/K+ ATPase",
        "Blockade of L-type calcium channels", "Inhibition of cyclooxygenase-1 and -2",
        "Inhibition of dihydrofolate reductase", "Inhibition of bacterial cell wall synthesis",
        "Agonism at mu-opioid receptors", "Inhibition of vitamin K epoxide reductase",
    ),
    "adverse_effects": (
        "Hepatotoxicity", "Agranulocytosis", "QT prolongation", "Lactic acidosis", "Angioedema", "Ototoxicity",
        "Pulmonary fibrosis", "Stevens-Johnson syndrome", "Hyperkalemia", "Tendon rupture",
    ),
    "conditions": (
        "Type 2 diabetes mellitus", "Essential hypertension", "Rheumatoid arthritis", "Major depressive disorder",
        "Migraine", "Gout", "Asthma", "Hypothyroidism", "Atrial fibrillation", "Parkinson disease",
    ),
    "doses": (
        "5 mg once daily", "10 mg twice daily", "25 mg once daily", "50 mg three times daily", "100 mg once daily",
        "250 mg twice daily", "500 mg twice daily", "1 g once daily", "2.5 mg once weekly", "20 mg at bedtime",
    ),
}
_DISEASE_TEMPLATES = (
    ("treatment", "Which drug is the first-line treatment for this condition?", "drugs"),
    ("gene", "Mutation of which gene causes this condition?", "genes"),
    ("organism", "Which organism causes this condition?", "organisms"),
    ("enzyme", "Deficiency of which enzyme causes this condition?", "enzymes"),
)
_DRUG_TEMPLATES = (
    ("mechanism", "What is the mechanism of action of this drug?", "mechanisms"),
    ("adverse", "Which serious adverse effect is most characteristic of this drug?", "adverse_effects"),
    ("indication", "For which condition is this drug first-line therapy?", "conditions"),
    ("dose", "What is the usual adult starting dose of this drug?", "doses"),
)


@dataclass(frozen=True)
class _Entity:
    kind: str  # "drug" or "disease"
    slug: str
    display: str


class FictionalHealthSource:
    def __init__(self, n_entities: int, seed: int, exclude_words: Iterable[str] = ()) -> None:
        self._n_entities = n_entities
        self._seed = seed
        self._exclude = {word.lower() for word in exclude_words}

    def load(self) -> list[HealthQuestion]:
        rng = random.Random(self._seed)
        questions = []
        for entity in self._entities(rng):
            is_drug = entity.kind == "drug"
            templates = _DRUG_TEMPLATES if is_drug else _DISEASE_TEMPLATES
            for key, question, pool in rng.sample(templates, _TEMPLATES_PER_ENTITY):
                age = rng.randint(19, 82)
                article = "An" if str(age).startswith("8") else "A"
                patient = f"{article} {age}-year-old {rng.choice(('man', 'woman'))}"
                vignette = f"{patient} is started on {entity.display}." if is_drug else f"{patient} is diagnosed with {entity.display}."
                questions.append(
                    HealthQuestion(
                        item_id=f"fict-{entity.slug}-{key}",
                        group_id=f"fict-{entity.slug}",
                        stratum=Stratum.FICTIONAL,
                        question=f"{vignette} {question}",
                        options=tuple(rng.sample(_POOLS[pool], 4)),
                        answer_index=None,
                    )
                )
        return questions

    def _entities(self, rng: random.Random) -> list[_Entity]:
        entities: list[_Entity] = []
        seen: set[str] = set()
        for _ in range(1000 * self._n_entities):
            if len(entities) == self._n_entities:
                return entities
            kind = "drug" if len(entities) % 2 == 0 else "disease"
            stem = rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
            if kind == "drug":
                slug = stem + rng.choice(_DRUG_ENDINGS)
                display = slug.capitalize()
            else:
                slug = stem + rng.choice(_VOWELS) + rng.choice(_EPONYM_ENDINGS)
                display = rng.choice(_DISEASE_FORMS).format(slug.capitalize())
            if slug in seen or slug in self._exclude or any(fragment in slug for fragment in _BANNED_FRAGMENTS):
                continue
            seen.add(slug)
            entities.append(_Entity(kind, slug, display))
        raise RuntimeError(f"could only invent {len(entities)} of {self._n_entities} unused names")
