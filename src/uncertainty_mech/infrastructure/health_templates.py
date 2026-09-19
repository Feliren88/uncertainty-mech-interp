"""Question templates, option pools and invented names shared by invented items and matched pairs."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

DRUG = "drug"
DISEASE = "disease"

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
)  # fmt: skip

POOLS: dict[str, tuple[str, ...]] = {
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
}  # fmt: skip


@dataclass(frozen=True)
class Template:
    key: str
    kind: str  # the entity kind it asks about: DRUG or DISEASE
    question: str
    pool: str


TEMPLATES: dict[str, Template] = {
    t.key: t
    for t in (
        Template("treatment", DISEASE, "Which drug is the first-line treatment for this condition?", "drugs"),
        Template("gene", DISEASE, "Mutation of which gene causes this condition?", "genes"),
        Template("organism", DISEASE, "Which organism causes this condition?", "organisms"),
        Template("enzyme", DISEASE, "Deficiency of which enzyme causes this condition?", "enzymes"),
        Template("mechanism", DRUG, "What is the mechanism of action of this drug?", "mechanisms"),
        Template(
            "adverse", DRUG, "Which serious adverse effect is most characteristic of this drug?", "adverse_effects"
        ),
        Template("indication", DRUG, "For which condition is this drug first-line therapy?", "conditions"),
        Template("dose", DRUG, "What is the usual adult starting dose of this drug?", "doses"),
    )
}


@dataclass(frozen=True)
class InventedName:
    kind: str
    slug: str
    display: str


def invent_names(rng: random.Random, kinds: Iterable[str], exclude_words: Iterable[str] = ()) -> list[InventedName]:
    """One unused invented name per requested kind, in order."""
    exclude = {word.lower() for word in exclude_words}
    names: list[InventedName] = []
    seen: set[str] = set()
    for kind in kinds:
        for _ in range(1000):
            stem = rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
            if kind == DRUG:
                slug = stem + rng.choice(_DRUG_ENDINGS)
                display = slug.capitalize()
            else:
                slug = stem + rng.choice(_VOWELS) + rng.choice(_EPONYM_ENDINGS)
                display = rng.choice(_DISEASE_FORMS).format(slug.capitalize())
            if slug not in seen and slug not in exclude and not any(part in slug for part in _BANNED_FRAGMENTS):
                seen.add(slug)
                names.append(InventedName(kind, slug, display))
                break
        else:
            raise RuntimeError(f"could not invent an unused {kind} name after {len(names)} names")
    return names


def patient(rng: random.Random) -> str:
    """For example "An 81-year-old man"."""
    age = rng.randint(19, 82)
    article = "An" if str(age).startswith("8") else "A"
    return f"{article} {age}-year-old {rng.choice(('man', 'woman'))}"


def vignette(patient_text: str, entity: str, kind: str) -> str:
    """A MedQA-style opening sentence naming the entity."""
    return f"{patient_text} is started on {entity}." if kind == DRUG else f"{patient_text} is diagnosed with {entity}."
