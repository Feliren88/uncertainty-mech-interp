"""The inference service: answer a health question, or say exactly "I don't know."."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from uncertainty_mech.application.ports import LanguageModel, RunStore
from uncertainty_mech.application.risk_model import RiskModel, SignalSet, build_features
from uncertainty_mech.domain.gate import Decision, ReasonCode, decide, render
from uncertainty_mech.domain.grading import chosen_indices, output_features
from uncertainty_mech.domain.prompts import answer_letters, build_prompt
from uncertainty_mech.domain.questions import OPTION_LETTERS

BUNDLE_FILE = "bundle.json"
BUNDLE_ARRAYS = "bundle_risk_model.npz"
AUDIT_FILE = "audit.jsonl"


@dataclass(frozen=True)
class GateBundle:
    """Everything `ask` needs, frozen at the end of calibration."""

    run_id: str
    model_name: str
    signal: SignalSet
    layer: int
    risk_model: RiskModel
    threshold: float | None
    target_risk: float


def save_bundle(store: RunStore, bundle: GateBundle) -> None:
    store.write_json(
        BUNDLE_FILE,
        {
            "run_id": bundle.run_id,
            "model_name": bundle.model_name,
            "signal": bundle.signal.value,
            "layer": bundle.layer,
            "threshold": bundle.threshold,
            "target_risk": bundle.target_risk,
        },
    )
    store.write_arrays(BUNDLE_ARRAYS, **bundle.risk_model.to_arrays())


def load_bundle(store: RunStore) -> GateBundle:
    meta = store.read_json(BUNDLE_FILE)
    return GateBundle(
        run_id=meta["run_id"],
        model_name=meta["model_name"],
        signal=SignalSet(meta["signal"]),
        layer=int(meta["layer"]),
        risk_model=RiskModel.from_arrays(store.read_arrays(BUNDLE_ARRAYS)),
        threshold=meta["threshold"],
        target_risk=float(meta["target_risk"]),
    )


@dataclass(frozen=True)
class Answer:
    text: str
    decision: Decision
    reason_code: ReasonCode
    risk: float | None
    candidate_letter: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "decision": self.decision.value,
            "reason_code": self.reason_code.value,
            "risk": self.risk,
            "candidate_letter": self.candidate_letter,
        }


class AbstainingAnswerer:
    """Reads the model once, scores the risk, and releases the answer only below the frozen threshold."""

    def __init__(self, model: LanguageModel, bundle: GateBundle, audit_store: RunStore | None = None) -> None:
        if model.name != bundle.model_name:
            raise ValueError(f"the gate was calibrated on {bundle.model_name}, not {model.name}")
        self._model = model
        self._bundle = bundle
        self._audit_store = audit_store

    def answer(self, question: str, options: Sequence[str]) -> Answer:
        started = time.perf_counter()
        readings = self._model.read(
            [build_prompt(question, options, allow_abstain=False)],
            answer_letters(allow_abstain=False),
            capture_residuals=self._bundle.signal.needs_residuals,
        )
        index = int(chosen_indices(readings.letter_logprobs)[0])
        risk = self._risk(readings)
        result = decide(risk, self._bundle.threshold)
        answer = Answer(
            render(result, OPTION_LETTERS[index], options[index]),
            result.decision,
            result.reason_code,
            risk,
            OPTION_LETTERS[index],
        )
        if self._audit_store is not None:
            self._audit_store.append_jsonl(AUDIT_FILE, self._audit_record(question, answer, started))
        return answer

    def _risk(self, readings) -> float | None:
        residuals = readings.residuals
        if self._bundle.signal.needs_residuals and residuals is None:
            return None
        layer_residuals = None if residuals is None else residuals[:, self._bundle.layer]
        features = build_features(
            self._bundle.signal, output_features(readings.letter_logprobs, readings.letter_mass), layer_residuals
        )
        return float(self._bundle.risk_model.risk(features)[0])

    def _audit_record(self, question: str, answer: Answer, started: float) -> dict[str, Any]:
        return {
            "time_utc": datetime.now(UTC).isoformat(),
            "run_id": self._bundle.run_id,
            "model": self._model.name,
            "signal": self._bundle.signal.value,
            "layer": self._bundle.layer,
            "threshold": self._bundle.threshold,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            **answer.to_dict(),
        }
