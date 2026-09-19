"""Stage 4: tiered semantic-entropy steering, with the circuit's own readout as a second trigger.

The model is run once per dose in a fixed grid. Any schedule is then scored
exactly by taking each question's letter from the pass at that question's dose,
because prompts in a batch do not interact. Schedules are chosen on calibration
items for each wrong-answer cost, then scored once on test items.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from typing import Any

import numpy as np

from uncertainty_mech.application.circuit_config import TierConfig
from uncertainty_mech.application.patching import E_LETTERS
from uncertainty_mech.application.ports import Capture, CircuitModel, Site, SiteKind
from uncertainty_mech.application.se_steering import EvalSet, SteeringVectors
from uncertainty_mech.domain.dose_schedule import ABSTAIN, DoseSchedule, abstention_profile, item_utility, utility
from uncertainty_mech.domain.grading import chosen_indices
from uncertainty_mech.domain.metrics import auroc, group_bootstrap_difference

SE_BANDS = ((0.0, 0.2), (0.2, 0.6), (0.6, 1.0), (1.0, np.inf))
# (better, baseline): what the tiers and the readout add over simpler controllers.
COMPARISONS = (
    ("circuit_tiered_readout", "se_wrapper"),
    ("circuit_tiered_readout", "circuit_one_threshold"),
    ("circuit_tiered", "circuit_one_threshold"),
)


@dataclass(frozen=True)
class DosePasses:
    """The model's letter for every item at each dose (dose 0 is the plain pass)."""

    doses: np.ndarray  # [m], sorted, starts at 0
    choices: np.ndarray  # [m, n]

    def letters(self, dose_per_item: np.ndarray) -> np.ndarray:
        index = np.searchsorted(self.doses, dose_per_item)
        if not np.allclose(self.doses[np.minimum(index, len(self.doses) - 1)], dose_per_item):
            raise ValueError("a schedule asked for a dose outside the grid")
        return self.choices[index, np.arange(self.choices.shape[1])]


def circuit_readout(model: CircuitModel, items: EvalSet, circuit: SteeringVectors) -> np.ndarray:
    """Sum over circuit heads of the head output's projection on its unit steering direction (plain pass)."""
    captures = [Capture(Site(SiteKind.ATTN_Z, layer, head), (1,)) for layer, head in circuit.heads]
    trace = model.trace(items.prompts, E_LETTERS, captures=captures)
    units = circuit.vectors / np.linalg.norm(circuit.vectors, axis=1, keepdims=True)
    return sum(values[:, 0] @ unit for values, unit in zip(trace.activations, units, strict=True))


def dose_passes(
    model: CircuitModel, items: EvalSet, vectors: SteeringVectors, dose_grid: Sequence[float]
) -> DosePasses:
    doses = np.array([0.0, *sorted(dose_grid)])
    choices = []
    for dose in doses:
        additions = [] if dose == 0 else vectors.additions(np.full(len(items.prompts), dose, dtype=np.float32))
        choices.append(chosen_indices(model.trace(items.prompts, E_LETTERS, additions=additions).letter_logprobs))
    return DosePasses(doses, np.stack(choices))


def candidate_schedules(config: TierConfig, readout_thresholds: Sequence[float]) -> list[DoseSchedule]:
    """Every monotone schedule with up to `max_tiers` tiers, with and without a readout trigger."""
    triggers: list[tuple[float | None, float]] = [(None, 0.0)]
    triggers += [(threshold, dose) for threshold in readout_thresholds for dose in config.dose_grid]
    schedules = []
    for tiers in range(1, config.max_tiers + 1):
        for thresholds in combinations(sorted(config.tier_thresholds), tiers):
            for doses in combinations_with_replacement(sorted(config.dose_grid), tiers):
                for readout_threshold, readout_dose in triggers:
                    schedules.append(DoseSchedule(thresholds, doses, readout_threshold, readout_dose))
    return schedules


@dataclass(frozen=True)
class Choice:
    family: str
    wrong_cost: float
    schedule: DoseSchedule | None  # None for the SE wrapper
    wrapper_threshold: float | None
    calibration_utility: float


FAMILIES = ("circuit_one_threshold", "circuit_tiered", "circuit_tiered_readout")


def _family(schedule: DoseSchedule) -> str:
    if schedule.readout_threshold is not None:
        return "circuit_tiered_readout"
    return "circuit_one_threshold" if schedule.tiers == 1 else "circuit_tiered"


def _members(family: str, schedule: DoseSchedule) -> bool:
    """Families are nested: tiered includes one threshold, tiered_readout includes both."""
    order = FAMILIES.index
    return order(_family(schedule)) <= order(family)


def choose(
    passes: DosePasses,
    items: EvalSet,
    readout: np.ndarray,
    schedules: Sequence[DoseSchedule],
    wrapper_thresholds: Sequence[float],
    wrong_cost: float,
) -> tuple[list[Choice], list[dict[str, Any]]]:
    """Best schedule per family on calibration items; ties go to fewer tiers and smaller doses."""
    scored = [
        (utility(passes.letters(s.doses_for(items.se, readout)), items.answers, wrong_cost), s) for s in schedules
    ]
    choices = []
    for family in FAMILIES:
        pool = [(u, s) for u, s in scored if _members(family, s)]
        best_utility, best = max(
            pool, key=lambda pair: (pair[0], -pair[1].tiers, -sum(pair[1].doses) - pair[1].readout_dose)
        )
        choices.append(Choice(family, wrong_cost, best, None, best_utility))
    plain = passes.choices[0]
    wrapper = [
        (utility(np.where(items.se > t, ABSTAIN, plain), items.answers, wrong_cost), t) for t in wrapper_thresholds
    ]
    best_utility, threshold = max(wrapper, key=lambda pair: (pair[0], pair[1]))
    choices.append(Choice("se_wrapper", wrong_cost, None, threshold, best_utility))
    top = sorted(scored, key=lambda pair: pair[0], reverse=True)[:20]
    top_rows = [{"wrong_cost": wrong_cost, "schedule": s.describe(), "calibration_utility": u} for u, s in top]
    return choices, top_rows


@dataclass(frozen=True)
class TieredResults:
    choices: list[Choice]
    readout_thresholds: list[float]
    readout_auroc_invented: float | None
    readout_auroc_unknown_real: float | None
    test_rows: list[dict[str, Any]]
    band_rows: list[dict[str, Any]]
    top_rows: list[dict[str, Any]]
    comparison_rows: list[dict[str, Any]]  # paired group-bootstrap differences on test
    prediction_rows: list[dict[str, Any]]  # one per test question: signals, doses and letters


def run_tiered_study(
    model: CircuitModel,
    calibration: EvalSet,
    test: EvalSet,
    circuit: SteeringVectors,
    random_heads: SteeringVectors,
    config: TierConfig,
    wrapper_thresholds: Sequence[float],
    rng: np.random.Generator,
) -> TieredResults:
    calibration_readout = circuit_readout(model, calibration, circuit)
    test_readout = circuit_readout(model, test, circuit)
    readout_thresholds = [float(np.quantile(calibration_readout, q)) for q in config.readout_quantiles]
    schedules = candidate_schedules(config, readout_thresholds)
    calibration_passes = dose_passes(model, calibration, circuit, config.dose_grid)
    test_passes = dose_passes(model, test, circuit, config.dose_grid)
    control_passes = dose_passes(model, test, random_heads, config.dose_grid)

    choices: list[Choice] = []
    top_rows: list[dict[str, Any]] = []
    for cost in config.wrong_costs:
        chosen, top = choose(calibration_passes, calibration, calibration_readout, schedules, wrapper_thresholds, cost)
        choices += chosen
        top_rows += top

    plain = test_passes.choices[0]
    test_rows, band_rows, comparison_rows = [], [], []
    predictions = {"group": test.groups, "answer": test.answers, "forced_correct": test.forced_correct}
    predictions |= {"se": test.se, "readout": test_readout}
    for cost in config.wrong_costs:
        conditions = {"option_e_only": plain}
        for choice in (c for c in choices if c.wrong_cost == cost):
            if choice.schedule is None:
                conditions["se_wrapper"] = np.where(test.se > choice.wrapper_threshold, ABSTAIN, plain)
                continue
            doses = choice.schedule.doses_for(test.se, test_readout)
            conditions[choice.family] = test_passes.letters(doses)
            if choice.family == "circuit_tiered_readout":
                conditions["control_random_heads"] = control_passes.letters(doses)
                predictions[f"c{cost:g}_dose_tiered_readout"] = doses
        predictions |= {f"c{cost:g}_{name}": letters for name, letters in conditions.items()}
        comparison_rows += _comparisons(cost, conditions, test, rng, config.bootstrap_samples)
        for name, letters in conditions.items():
            test_rows.append(
                {
                    "wrong_cost": cost,
                    "condition": name,
                    "utility": utility(letters, test.answers, cost),
                    **abstention_profile(letters, test.answers, test.forced_correct),
                }
            )
        band_rows += _band_rows(cost, conditions, test)

    invented = test.answers < 0
    real = ~invented
    return TieredResults(
        choices=choices,
        readout_thresholds=readout_thresholds,
        readout_auroc_invented=auroc(test_readout, invented.astype(int)),
        readout_auroc_unknown_real=auroc(test_readout[real], (~test.forced_correct[real]).astype(int)),
        test_rows=test_rows,
        band_rows=band_rows,
        top_rows=top_rows,
        comparison_rows=comparison_rows,
        prediction_rows=[
            {key: _plain(values[i]) for key, values in predictions.items()} for i in range(len(test.prompts))
        ],
    )


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _comparisons(
    cost: float, conditions: dict[str, np.ndarray], items: EvalSet, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    """Paired differences on the same test questions, with 95% intervals over resampled groups."""
    unknown = (items.answers < 0) | ~items.forced_correct
    rows = []
    for better, baseline in COMPARISONS:
        a, b = conditions[better], conditions[baseline]
        measures = {
            "utility": (item_utility(a, items.answers, cost), item_utility(b, items.answers, cost), None),
            "abstain_on_unknown": (a == ABSTAIN, b == ABSTAIN, unknown),
            "abstain_on_known": (a == ABSTAIN, b == ABSTAIN, ~unknown),
        }
        for measure, (values_a, values_b, mask) in measures.items():
            difference, low, high = group_bootstrap_difference(values_a, values_b, items.groups, rng, samples, mask)
            rows.append(
                {
                    "wrong_cost": cost,
                    "controller": better,
                    "baseline": baseline,
                    "measure": measure,
                    "difference": difference,
                    "low_95": low,
                    "high_95": high,
                }
            )
    return rows


def _band_rows(cost: float, conditions: dict[str, np.ndarray], items: EvalSet) -> list[dict[str, Any]]:
    """Abstention by semantic-entropy band, for the plain prompt, the wrapper and the full controller."""
    rows = []
    for name in ("option_e_only", "se_wrapper", "circuit_tiered_readout"):
        for low, high in SE_BANDS:
            band = (items.se > low) & (items.se <= high) if low > 0 else items.se <= high
            if not band.any():
                continue
            profile = abstention_profile(conditions[name][band], items.answers[band], items.forced_correct[band])
            rows.append(
                {
                    "wrong_cost": cost,
                    "condition": name,
                    "se_band": f"{low:.1f} to {high:.1f}" if np.isfinite(high) else f"above {low:.1f}",
                    "n": int(band.sum()),
                    "abstained": float((conditions[name][band] == ABSTAIN).mean()),
                    "abstain_on_unknown": profile["abstain_on_unknown"],
                    "abstain_on_known": profile["abstain_on_known"],
                }
            )
    return rows
