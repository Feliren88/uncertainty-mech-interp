"""Tables, figure and markdown for tiered semantic-entropy steering (stage 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from uncertainty_mech.application.markdown_format import num as _num
from uncertainty_mech.application.markdown_format import pct as _pct
from uncertainty_mech.application.markdown_format import table as _table
from uncertainty_mech.application.ports import CircuitFigureWriter, RunStore
from uncertainty_mech.application.tiered_steering import Choice, TieredResults

TIER_LABELS = {
    "option_e_only": "Option E only",
    "se_wrapper": "SE wrapper, one threshold",
    "circuit_one_threshold": "Circuit, one SE threshold",
    "circuit_tiered": "Circuit, tiered SE thresholds",
    "circuit_tiered_readout": "Circuit, tiered SE + circuit readout",
    "control_random_heads": "Control: tiered + readout schedule on random heads",
}


def write_tier_tables(store: RunStore, tiered: TieredResults) -> None:
    store.write_table("tiers_choices.csv", [choice_row(choice) for choice in tiered.choices])
    store.write_table("tiers_test.csv", tiered.test_rows)
    store.write_table("tiers_bands.csv", tiered.band_rows)
    store.write_table("tiers_top_schedules.csv", tiered.top_rows)
    store.write_table("tiers_comparisons.csv", tiered.comparison_rows)
    store.write_table("tiers_predictions.csv", tiered.prediction_rows)


def draw_tier_figure(figures: CircuitFigureWriter, tiered: TieredResults, path: Path) -> None:
    figures.abstention_tradeoff(*_tier_paths(tiered), path)


def choice_row(choice: Choice) -> dict[str, Any]:
    return {
        "wrong_cost": choice.wrong_cost,
        "family": choice.family,
        "schedule": _schedule_text(choice),
        "calibration_utility": choice.calibration_utility,
    }


def _tier_paths(
    tiered: TieredResults,
) -> tuple[dict[str, list[tuple[float, float, float]]], dict[str, tuple[float, float]]]:
    paths: dict[str, list[tuple[float, float, float]]] = {}
    points: dict[str, tuple[float, float]] = {}
    for row in tiered.test_rows:
        point = (row["abstain_on_known"], row["abstain_on_unknown"])
        if row["condition"] == "option_e_only":
            points["option_e_only"] = point
        else:
            paths.setdefault(row["condition"], []).append((*point, row["wrong_cost"]))
    return paths, points


def _tier_row(tiered: TieredResults, cost: float, condition: str) -> dict[str, Any]:
    return next(r for r in tiered.test_rows if r["wrong_cost"] == cost and r["condition"] == condition)


def tier_markdown(tiered: TieredResults) -> str:
    costs = sorted({row["wrong_cost"] for row in tiered.test_rows})
    middle = costs[len(costs) // 2]
    full = _tier_row(tiered, middle, "circuit_tiered_readout")
    wrapper = _tier_row(tiered, middle, "se_wrapper")
    plain = _tier_row(tiered, middle, "option_e_only")
    choice_rows = [
        [
            f"{choice.wrong_cost:g}",
            TIER_LABELS[choice.family],
            _schedule_text(choice),
            _num(choice.calibration_utility, 3),
        ]
        for choice in tiered.choices
    ]
    test_rows = [
        [
            f"{row['wrong_cost']:g}",
            TIER_LABELS[row["condition"]],
            _num(row["utility"], 3),
            _pct(row["abstain_on_unknown"]),
            _pct(row["abstain_on_known"]),
            _pct(row["abstain_on_invented"]),
            _pct(row["wrong_among_answered"]),
        ]
        for row in tiered.test_rows
    ]
    band_rows = [
        [
            TIER_LABELS[row["condition"]],
            row["se_band"],
            str(row["n"]),
            _pct(row["abstain_on_unknown"]),
            _pct(row["abstain_on_known"]),
        ]
        for row in tiered.band_rows
        if row["wrong_cost"] == middle
    ]
    return "\n\n".join(
        [
            "A schedule maps semantic entropy to a dose: above each threshold the circuit gets a stronger push, and "
            "the reply is the model's own letter under that push. The circuit readout is how far the circuit heads' "
            "outputs point along their steering directions in the plain pass. Above its threshold the dose rises to at "
            "least the readout dose, which can catch confident answers about unfamiliar entities. Schedules are chosen "
            "on cal_prob to maximize utility, right answers minus c times wrong answers per question, for each "
            "wrong-answer cost c, and scored once on test. Unknown questions are invented entities plus real questions "
            "the model gets wrong when forced to choose; known questions are those it gets right.",
            f"The circuit readout separates invented from real questions with AUROC "
            f"{_num(tiered.readout_auroc_invented, 3)}, and wrong from right answers on real questions with AUROC "
            f"{_num(tiered.readout_auroc_unknown_real, 3)}.",
            _table(["Cost c", "Controller", "Chosen on cal_prob", "Calibration utility"], choice_rows),
            _table(
                [
                    "Cost c",
                    "Condition",
                    "Utility",
                    "Abstains on unknown",
                    "Abstains on known",
                    "Invented refused",
                    "Wrong among answered",
                ],
                test_rows,
            ),
            "![Tiered steering](figures/tiers_tradeoff.png)",
            f"*At cost {middle:g}, tiered steering with the circuit readout abstains on "
            f"{_pct(full['abstain_on_unknown'])} of unknown questions and {_pct(full['abstain_on_known'])} of known "
            f"ones; the one-threshold SE wrapper abstains on {_pct(wrapper['abstain_on_unknown'])} and "
            f"{_pct(wrapper['abstain_on_known'])}, and option E alone on {_pct(plain['abstain_on_unknown'])} and "
            f"{_pct(plain['abstain_on_known'])}.*",
            "Paired differences on the same test questions, with 95% intervals from resampling question groups "
            "(an invented entity's three questions form one group). Abstention differences are in percentage points.",
            _table(
                ["Cost c", "Controller", "Compared with", "Utility", "Abstains on unknown", "Abstains on known"],
                _comparison_rows(tiered),
            ),
            f"Abstention by semantic-entropy band at cost {middle:g}:",
            _table(["Condition", "SE band (nats)", "n", "Abstains on unknown", "Abstains on known"], band_rows),
        ]
    )


def _comparison_rows(tiered: TieredResults) -> list[list[str]]:
    cells: dict[tuple[float, str, str], dict[str, str]] = {}
    for row in tiered.comparison_rows:
        scale, digits = (1.0, 3) if row["measure"] == "utility" else (100.0, 1)
        text = (
            f"{scale * row['difference']:+.{digits}f} "
            f"[{scale * row['low_95']:+.{digits}f}, {scale * row['high_95']:+.{digits}f}]"
        )
        cells.setdefault((row["wrong_cost"], row["controller"], row["baseline"]), {})[row["measure"]] = text
    return [
        [
            f"{cost:g}",
            TIER_LABELS[controller],
            TIER_LABELS[baseline],
            m["utility"],
            m["abstain_on_unknown"],
            m["abstain_on_known"],
        ]
        for (cost, controller, baseline), m in cells.items()
    ]


def _schedule_text(choice: Choice) -> str:
    return choice.schedule.describe() if choice.schedule else f"SE > {choice.wrapper_threshold:.1f}: reply E"
