"""Composition root: the only module that knows every adapter.

python -m uncertainty_mech run --config configs/health_full_run.toml
python -m uncertainty_mech ask --run-dir runs/<run_id> --questions examples/health_questions.jsonl
python -m uncertainty_mech circuits --config configs/circuit_study.toml
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from uncertainty_mech.application.answer import AbstainingAnswerer, load_bundle
from uncertainty_mech.application.circuit_config import load_circuit_config
from uncertainty_mech.application.circuit_pipeline import run_circuit_study
from uncertainty_mech.application.config import DataConfig, ModelConfig, load_config
from uncertainty_mech.application.pipeline import run_pipeline
from uncertainty_mech.application.ports import CircuitModel, LanguageModel, QuestionSource, ReferenceCorpus
from uncertainty_mech.domain.questions import Role, Stratum
from uncertainty_mech.infrastructure.entity_pairs import EntityPairSource
from uncertainty_mech.infrastructure.fictional import FictionalHealthSource
from uncertainty_mech.infrastructure.figures import MatplotlibFigures
from uncertainty_mech.infrastructure.store import FileRunStore

log = logging.getLogger(__name__)


class RealSource(QuestionSource, ReferenceCorpus, Protocol):
    pass


ModelFactory = Callable[[ModelConfig], LanguageModel]
CircuitModelFactory = Callable[[ModelConfig], CircuitModel]
RealSourceFactory = Callable[[DataConfig, int], RealSource]


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: ModelFactory | None = None,
    circuit_model_factory: CircuitModelFactory | None = None,
    real_source_factory: RealSourceFactory | None = None,
) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    model_factory = model_factory or _huggingface_model
    real_source_factory = real_source_factory or _medqa_source
    if args.command == "run":
        return _run(args.config, model_factory, real_source_factory)
    if args.command == "circuits":
        return _circuits(args.config, circuit_model_factory or _huggingface_model, real_source_factory)
    return _ask(args.run_dir, args.questions, model_factory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uncertainty-mech", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Build data, read the model, fit and certify the gates, evaluate, report.")
    run.add_argument("--config", type=Path, required=True)
    ask = commands.add_parser("ask", help="Answer questions with a frozen gate, or say I don't know.")
    ask.add_argument("--run-dir", type=Path, required=True)
    ask.add_argument("--questions", type=Path, required=True, help="JSONL lines with 'question' and 4 'options'.")
    circuits = commands.add_parser(
        "circuits", help="Find the abstention circuit and test semantic-entropy steering on a finished run."
    )
    circuits.add_argument("--config", type=Path, required=True)
    return parser


def _run(config_path: Path, model_factory: ModelFactory, real_source_factory: RealSourceFactory) -> int:
    config = load_config(config_path)
    store = FileRunStore(config.run_dir)
    with _log_to(store):
        real_source = real_source_factory(config.data, config.seed)
        real = real_source.load()
        fictional = FictionalHealthSource(
            config.data.n_fictional_entities, config.seed, exclude_words=real_source.corpus_words()
        ).load()
        summary = run_pipeline(config, real + fictional, model_factory(config.model), store, MatplotlibFigures())
    print(json.dumps(summary, indent=2))
    return 0


def _circuits(config_path: Path, model_factory: CircuitModelFactory, real_source_factory: RealSourceFactory) -> int:
    config = load_circuit_config(config_path)
    store = FileRunStore(config.run_dir)
    source = FileRunStore(config.source_run)
    with _log_to(store):
        source_config = source.read_json("config.json")
        data = source_config["data"]
        data_config = DataConfig(**{**data, "proportions": {Role(k): v for k, v in data["proportions"].items()}})
        used_names = {
            row["group_id"].removeprefix("fict-")
            for row in source.read_table("items.csv")
            if row["stratum"] == Stratum.FICTIONAL.value
        }
        corpus = real_source_factory(data_config, int(source_config["seed"])).corpus_words()
        pairs = EntityPairSource(config.pairs.names_per_fact, config.seed, exclude_words=corpus | used_names).load()
        summary = run_circuit_study(config, pairs, model_factory(config.model), source, store, MatplotlibFigures())
    print(json.dumps(summary, indent=2, default=str))
    return 0


@contextmanager
def _log_to(store: FileRunStore) -> Iterator[None]:
    handler = logging.FileHandler(store.root / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        yield
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _ask(run_dir: Path, questions_path: Path, model_factory: ModelFactory) -> int:
    store = FileRunStore(run_dir)
    model_config = ModelConfig(**store.read_json("config.json")["model"])
    answerer = AbstainingAnswerer(model_factory(model_config), load_bundle(store), audit_store=store)
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        answer = answerer.answer(item["question"], item["options"])
        print(json.dumps({"question": item["question"], **answer.to_dict()}))
    return 0


def _huggingface_model(config: ModelConfig) -> CircuitModel:
    # Imported here so torch loads only when a real model is needed.
    from uncertainty_mech.infrastructure.hf_model import HuggingFaceModel

    return HuggingFaceModel.from_config(config)


def _medqa_source(config: DataConfig, seed: int) -> RealSource:
    from uncertainty_mech.infrastructure.medqa import MedQASource

    return MedQASource(config.real_repo, config.real_revision, config.real_file, config.n_real, seed)
