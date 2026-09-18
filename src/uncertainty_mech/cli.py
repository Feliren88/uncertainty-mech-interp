"""Composition root: the only module that knows every adapter.

    python -m uncertainty_mech run --config configs/health_test_run.toml
    python -m uncertainty_mech ask --run-dir runs/<run_id> --questions examples/health_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from uncertainty_mech.application.answer import AbstainingAnswerer, load_bundle
from uncertainty_mech.application.config import DataConfig, ModelConfig, load_config
from uncertainty_mech.application.pipeline import run_pipeline
from uncertainty_mech.application.ports import LanguageModel, QuestionSource, ReferenceCorpus
from uncertainty_mech.infrastructure.fictional import FictionalHealthSource
from uncertainty_mech.infrastructure.figures import MatplotlibFigures
from uncertainty_mech.infrastructure.store import FileRunStore

log = logging.getLogger(__name__)


class RealSource(QuestionSource, ReferenceCorpus, Protocol):
    pass


ModelFactory = Callable[[ModelConfig], LanguageModel]
RealSourceFactory = Callable[[DataConfig, int], RealSource]


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: ModelFactory | None = None,
    real_source_factory: RealSourceFactory | None = None,
) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    model_factory = model_factory or _huggingface_model
    if args.command == "run":
        return _run(args.config, model_factory, real_source_factory or _medqa_source)
    return _ask(args.run_dir, args.questions, model_factory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uncertainty-mech", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Build data, read the model, fit and certify the gates, evaluate, report.")
    run.add_argument("--config", type=Path, required=True)
    ask = commands.add_parser("ask", help="Answer questions with a frozen gate, or say I don't know.")
    ask.add_argument("--run-dir", type=Path, required=True)
    ask.add_argument("--questions", type=Path, required=True, help="JSONL lines with 'question' and 4 'options'.")
    return parser


def _run(config_path: Path, model_factory: ModelFactory, real_source_factory: RealSourceFactory) -> int:
    config = load_config(config_path)
    store = FileRunStore(config.run_dir)
    handler = logging.FileHandler(store.root / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        real_source = real_source_factory(config.data, config.seed)
        real = real_source.load()
        fictional = FictionalHealthSource(
            config.data.n_fictional_entities, config.seed, exclude_words=real_source.corpus_words()
        ).load()
        summary = run_pipeline(config, real + fictional, model_factory(config.model), store, MatplotlibFigures())
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    print(json.dumps(summary, indent=2))
    return 0


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


def _huggingface_model(config: ModelConfig) -> LanguageModel:
    from uncertainty_mech.infrastructure.hf_model import HuggingFaceModel  # torch loads only when a real model is needed

    return HuggingFaceModel.from_config(config)


def _medqa_source(config: DataConfig, seed: int) -> RealSource:
    from uncertainty_mech.infrastructure.medqa import MedQASource

    return MedQASource(config.real_repo, config.real_revision, config.real_file, config.n_real, seed)
