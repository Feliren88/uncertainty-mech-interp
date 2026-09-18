"""Run configuration read from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from uncertainty_mech.domain.questions import Role


@dataclass(frozen=True)
class DataConfig:
    real_repo: str
    real_revision: str
    real_file: str
    n_real: int
    n_fictional_entities: int
    proportions: dict[Role, float]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str
    dtype: str = "bfloat16"
    batch_size: int = 16


@dataclass(frozen=True)
class ProbeConfig:
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    sweep_c: float = 0.1
    cv_folds: int = 5


@dataclass(frozen=True)
class GateConfig:
    target_risk: float = 0.10
    delta: float = 0.05


@dataclass(frozen=True)
class SteeringConfig:
    doses: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    n_items: int = 200
    n_random: int = 3
    identity_tolerance: float = 1e-4


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    output_root: Path
    data: DataConfig
    model: ModelConfig
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["data"]["proportions"] = {role.value: share for role, share in self.data.proportions.items()}
        return payload


def load_config(path: Path) -> RunConfig:
    """Read a run config. `output_root` is resolved relative to the config file."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    data = dict(raw["data"])
    data["proportions"] = {Role(name): float(share) for name, share in data["proportions"].items()}
    return RunConfig(
        run_id=raw["run_id"],
        seed=int(raw["seed"]),
        output_root=(path.parent / raw.get("output_root", "runs")).resolve(),
        data=DataConfig(**data),
        model=ModelConfig(**raw["model"]),
        probe=ProbeConfig(**_as_tuples(raw.get("probe", {}))),
        gate=GateConfig(**raw.get("gate", {})),
        steering=SteeringConfig(**_as_tuples(raw.get("steering", {}))),
    )


def _as_tuples(section: dict[str, Any]) -> dict[str, Any]:
    return {key: tuple(value) if isinstance(value, list) else value for key, value in section.items()}
