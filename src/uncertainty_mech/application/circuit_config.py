"""Configuration for the abstention-circuit and SE-steering study, read from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from uncertainty_mech.application.config import ModelConfig


@dataclass(frozen=True)
class PairConfig:
    names_per_fact: int = 3
    discovery_share: float = 0.6


@dataclass(frozen=True)
class CircuitConfig:
    k_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    sufficiency_threshold: float = 0.5
    n_random: int = 5


@dataclass(frozen=True)
class SteeringGridConfig:
    doses: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    se_thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)


@dataclass(frozen=True)
class CircuitStudyConfig:
    run_id: str
    seed: int
    output_root: Path
    source_run: Path  # a finished `run` directory: items.csv, readings.npz, config.json
    model: ModelConfig
    pairs: PairConfig = field(default_factory=PairConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    steering: SteeringGridConfig = field(default_factory=SteeringGridConfig)

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["source_run"] = str(self.source_run)
        return payload


def load_circuit_config(path: Path) -> CircuitStudyConfig:
    """Paths in the file are resolved relative to the config file."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return CircuitStudyConfig(
        run_id=raw["run_id"],
        seed=int(raw["seed"]),
        output_root=(path.parent / raw.get("output_root", "runs")).resolve(),
        source_run=(path.parent / raw["source_run"]).resolve(),
        model=ModelConfig(**raw["model"]),
        pairs=PairConfig(**raw.get("pairs", {})),
        circuit=CircuitConfig(**_as_tuples(raw.get("circuit", {}))),
        steering=SteeringGridConfig(**_as_tuples(raw.get("steering", {}))),
    )


def _as_tuples(section: dict[str, Any]) -> dict[str, Any]:
    return {key: tuple(value) if isinstance(value, list) else value for key, value in section.items()}
