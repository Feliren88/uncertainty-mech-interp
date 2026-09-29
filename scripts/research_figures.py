"""Render the figures for docs/research/README.md from saved run tables.

Run from the repository root after the runs exist:

    python scripts/research_figures.py

Every figure reads a CSV under runs/ and writes a PNG to docs/research/figures/.
The one-sentence message of each figure is written to captions.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import chartstyle as cs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "runs/health-llama31-8b-full-run"
PILOT = ROOT / "runs/health-llama31-8b-test-run"
CIRC = ROOT / "runs/circuit-se-steering"
OUT = ROOT / "docs/research/figures"

# One colour per meaning across every figure.
KNOWN = cs.PALETTE["c"]  # real question answered correctly
WRONG = cs.PALETTE["b"]  # real question answered incorrectly
INVENTED = cs.PALETTE["e"]  # invented-entity question
BASE = cs.PALETTE["d"]  # model alone (no gate, option E only)
EXTERNAL = cs.PALETTE["f"]  # output statistics or SE wrapper outside the model
ACT = cs.PALETTE["a"]  # activation-based method (probe, circuit steering)
ACT_DARK = "#0b3d63"  # strongest activation-based variant
CONTROL = cs.PALETTE["null"]  # random controls
REF = cs.PALETTE["ref"]

COST_MARK = {1.0: "o", 2.0: "s", 4.0: "^"}


def path(name: str) -> str:
    return str(OUT / f"{name}.png")


def tiers() -> pd.DataFrame:
    t = pd.read_csv(CIRC / "tiers_predictions.csv")
    t["invented"] = ~t["group"].str.startswith("medqa")
    t["kind"] = np.where(t["invented"], "invented", np.where(t["forced_correct"], "known", "wrong"))
    return t


def fig_se_by_type(t: pd.DataFrame) -> None:
    fig, ax = cs.new_fig(7.2, 4.0)
    bins = np.linspace(0, np.log(4), 29)
    for kind, colour, label in [
        ("known", KNOWN, "Real, answered correctly"),
        ("wrong", WRONG, "Real, answered incorrectly"),
        ("invented", INVENTED, "Invented entity"),
    ]:
        se = t.loc[t["kind"] == kind, "se"]
        ax.hist(se, bins=bins, density=True, histtype="step", linewidth=1.8, color=colour,
                label=f"{label} (n = {len(se):,}, median {se.median():.2f})")
    ax.axvline(np.log(4), color=REF, linestyle=":", linewidth=1)
    ax.text(np.log(4) - 0.02, 0.97, "Maximum ln 4", ha="right", va="top", fontsize=9,
            transform=ax.get_xaxis_transform())
    cs.legend_outside(ax, title="Question type")
    cs.finish(fig, ax, "Semantic entropy separates correct from incorrect real answers, and invented-entity "
              "questions overlap the incorrect ones", "Semantic entropy under forced choice (nats)",
              "Density", path("f01_se_by_question_type"), context="Test role, 2,388 questions")


def fig_option_e(t: pd.DataFrame) -> None:
    fig, ax = cs.new_fig(6.0, 3.8)
    kinds = [("known", KNOWN, "Real,\nanswered correctly"), ("wrong", WRONG, "Real,\nanswered incorrectly"),
             ("invented", INVENTED, "Invented\nentity")]
    xs = np.arange(len(kinds))
    vals, ns = [], []
    for kind, _, _ in kinds:
        sub = t[t["kind"] == kind]
        vals.append(float((sub["c1_option_e_only"] == 4).mean()))
        ns.append(len(sub))
    ax.bar(xs, vals, color=[k[1] for k in kinds], width=0.6)
    for x, v, n in zip(xs, vals, ns):
        ax.text(x, v + 0.02, f"{v:.1%}\n(n = {n:,})", ha="center", fontsize=9.5)
    ax.set_xticks(xs, [k[2] for k in kinds])
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.finish(fig, ax, "The model chooses option E almost only on invented-entity questions",
              "", "Share choosing \"E. I don't know\"", path("f02_option_e_by_question_type"),
              context="Test role")


def fig_layer_sweep() -> None:
    d = pd.read_csv(FULL / "layer_sweep.csv")
    fig, ax = cs.new_fig(7.0, 3.8)
    ax.plot(d["layer"], d["auroc_mean"], color=ACT, linewidth=1.8, marker="o", markersize=3.5,
            label="Logistic probe, mean of 5 folds")
    ax.fill_between(d["layer"], d["auroc_mean"] - d["auroc_sd"], d["auroc_mean"] + d["auroc_sd"],
                    color=ACT, alpha=0.2, linewidth=0, label="Plus or minus one standard deviation")
    ax.axvline(19, color=REF, linestyle="--", linewidth=1)
    ax.text(19.3, 0.72, "Layer 19\nused by the gate", fontsize=9)
    cs.legend_outside(ax)
    cs.finish(fig, ax, "Wrong answers become more linearly decodable between layers 12 and 18 and the "
              "curve is flat after that", "Decoder layer", "Cross-validated AUROC (wrong vs right)",
              path("f03_probe_layer_sweep"), context="Discovery role, grouped 5-fold cross-validation")


def fig_certification() -> None:
    fig, ax = cs.new_fig(7.2, 4.0)
    for run, colour, style, label in [
        (PILOT, CONTROL, "--", "Pilot (427 certification units)"),
        (FULL, ACT, "-", "Second run (3,148 certification units)"),
    ]:
        d = pd.read_csv(run / "thresholds.csv")
        d = d[(d["signal"] == "combined") & (d["accepted"] > 0)]
        ax.plot(d["threshold"], d["upper_bound"], linestyle=style, color=colour, marker="o",
                markersize=3.5, linewidth=1.8, label=label)
    ax.axhline(0.10, color=WRONG, linewidth=1.2, label="Target, 10%")
    ax.axvline(0.15, color=REF, linestyle=":", linewidth=1)
    ax.text(0.155, 0.5, "Selected\nthreshold 0.15", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.legend_outside(ax, title="Combined gate")
    cs.finish(fig, ax, "With 427 certification units no threshold met the 10% target; with 3,148 units "
              "thresholds from 0.02 to 0.15 did", "Risk threshold (released if predicted risk is below it)",
              "Bonferroni-corrected 95% upper bound", path("f04_certification_bounds"),
              context="Clopper-Pearson bound at level 0.05/20 per threshold")


def fig_risk_coverage() -> None:
    d = pd.read_csv(FULL / "test_predictions.csv")
    fig, ax = cs.new_fig(7.2, 4.0)
    n = len(d)
    for sig, colour, label in [("output", EXTERNAL, "Output statistics"), ("probe", ACT, "Activation probe"),
                               ("combined", ACT_DARK, "Combined (primary)")]:
        order = d.sort_values(f"risk_{sig}")["error"].to_numpy()
        k = np.arange(1, n + 1)
        ax.plot(k / n, np.cumsum(order) / k, color=colour, linewidth=1.6, label=label)
    ax.axhline(0.10, color=WRONG, linewidth=1, linestyle="--", label="Target, 10%")
    ax.scatter([0.913], [0.331], color=BASE, zorder=3, s=40, label="Option E offered")
    ax.scatter([1.0], [0.377], color=BASE, marker="s", zorder=3, s=40, label="No gate")
    ax.scatter([0.312], [0.059], color=ACT_DARK, marker="*", s=140, zorder=4,
               label="Frozen gate at threshold 0.15")
    ax.set_ylim(0, 0.45)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.legend_outside(ax)
    cs.finish(fig, ax, "All three risk scores keep the error rate under 10% up to about 40% coverage, and the "
              "frozen gate sits at 31% coverage and 5.9% error", "Share of questions answered (coverage)",
              "Wrong among answered", path("f05_risk_coverage"), context="Test role, 2,388 questions")


def fig_auroc() -> None:
    d = pd.read_csv(FULL / "risk_quality.csv")
    fig, ax = cs.new_fig(6.4, 3.8)
    sigs = [("output", EXTERNAL, "Output statistics"), ("probe", ACT, "Activation probe"),
            ("combined", ACT_DARK, "Combined")]
    xs = np.arange(2)
    w = 0.26
    for i, (sig, colour, label) in enumerate(sigs):
        vals = [float(d[(d.signal == sig) & (d.stratum == s)]["auroc"].iloc[0]) for s in ("all", "real")]
        ax.bar(xs + (i - 1) * w, vals, width=w, color=colour, label=label)
        for x, v in zip(xs + (i - 1) * w, vals):
            ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs, ["All test questions\n(real and invented)", "Real MedQA\nquestions only"])
    ax.set_ylim(0.7, 0.88)
    cs.legend_outside(ax, title="Risk score")
    cs.finish(fig, ax, "The activation advantage comes from invented questions; on real questions output "
              "statistics rank errors slightly better", "", "Test AUROC for predicting a wrong answer",
              path("f06_auroc_by_signal"), context="Test role; y-axis starts at 0.7")


def fig_error_direction() -> None:
    d = pd.read_csv(FULL / "steering.csv")
    fig, ax = cs.new_fig(7.0, 3.8)
    for name, g in d.groupby("direction"):
        is_err = name == "error_direction"
        ax.plot(g["dose"], g["mean_p_abstain"], color=ACT if is_err else CONTROL,
                linewidth=2.2 if is_err else 1.2, marker="o", markersize=3.5,
                label="Wrong-minus-right direction" if is_err else None)
    ax.plot([], [], color=CONTROL, marker="o", markersize=3.5, label="Random directions, same norm (3)")
    ax.set_xticks([-4, -2, -1, 0, 1, 2, 4])
    cs.legend_outside(ax)
    cs.finish(fig, ax, "Adding the layer-19 error direction does not raise the probability of E above random "
              "directions of the same norm", "Dose (multiples of the class-mean difference)",
              "Mean probability of \"I don't know\"", path("f07_error_direction_steering"),
              context="200 test questions, all positions steered")


def fig_residual_patching() -> None:
    d = pd.read_csv(CIRC / "residual_patching.csv")
    names = {"entity": "Last entity token", "after_entity": "Token after entity",
             "tail": "Shared instruction tail", "final": "Final token"}
    colours = {"entity": INVENTED, "after_entity": CONTROL, "tail": EXTERNAL, "final": ACT}
    fig, ax = cs.new_fig(7.2, 3.9)
    for seg in ["entity", "after_entity", "tail", "final"]:
        g = d[d["segment"] == seg]
        ax.plot(g["layer"], g["restoration_gap"], color=colours[seg], linewidth=1.8, marker="o",
                markersize=3, label=names[seg])
    ax.axhline(0, color=REF, linewidth=0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.legend_outside(ax, title="Patched position")
    cs.finish(fig, ax, "The abstention information sits on the entity in early layers, passes through the "
              "instruction tail around layer 14 and reaches the final token from layer 15",
              "Decoder layer", "Share of abstention gap restored", path("f08_residual_patching"),
              context="126 discovery pairs, invented into real")


def fig_head_heatmap() -> None:
    d = pd.read_csv(CIRC / "head_patching.csv")
    grid = d.pivot(index="head", columns="layer", values="restoration_gap").to_numpy()
    fig, ax = cs.new_fig(7.6, 5.4)
    lim = 0.55
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cbar.set_label("Share of abstention gap restored")
    for (layer, head, text, dx) in [(15, 4, "L15.H4", -1), (17, 25, "L17.H25", -1), (30, 25, "L30.H25", -1),
                                    (30, 27, "L30.H27", -1)]:
        ax.annotate(text, (layer, head), xytext=(layer + dx * 5, head + 2.5), fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=REF, linewidth=0.8))
    ax.set_xticks(range(0, 32, 4))
    ax.set_yticks(range(0, 32, 4))
    cs.finish(fig, ax, "Single-head effects are confined to layers 13 to 31; L15.H4 and L17.H25 are the largest "
              "positive heads and L30.H25 is strongly negative", "Layer", "Head",
              path("f09_head_patching_heatmap"), cbar=cbar,
              context="126 discovery pairs, final token, one head at a time")


def fig_circuit_size() -> None:
    # Joint top-k patches on discovery pairs, from runs/circuit-se-steering/report.md.
    k = np.array([1, 2, 4, 8, 16, 32])
    r = np.array([0.320, 0.521, 0.733, 0.916, 0.968, 0.992])
    fig, ax = cs.new_fig(6.4, 3.7)
    ax.plot(k, r, color=ACT, marker="o", linewidth=1.8)
    for x, y in zip(k, r):
        ax.text(x, y + 0.035, f"{y:.0%}", ha="center", fontsize=9)
    ax.axhline(0.5, color=REF, linestyle="--", linewidth=1)
    ax.text(20, 0.45, "Criterion, 50%", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.finish(fig, ax, "Two heads are the smallest set that passes 50% restoration, so the circuit is "
              "L15.H4 and L17.H25", "Number of top-ranked heads patched jointly",
              "Share of abstention gap restored", path("f10_circuit_size"), context="126 discovery pairs")


def fig_circuit_validation() -> None:
    d = pd.read_csv(CIRC / "circuit_validation.csv")
    fig, ax = cs.new_fig(6.8, 3.8)
    dirs = [("invented_into_real", "Invented into real\n(sufficiency)"),
            ("real_into_invented", "Real into invented\n(necessity)")]
    for i, (key, _) in enumerate(dirs):
        c = d[(d.direction == key) & (d.head_set == "circuit")]["restoration_gap"].iloc[0]
        rnd = d[(d.direction == key) & (d.head_set != "circuit")]["restoration_gap"].to_numpy()
        ax.bar(i - 0.18, c, width=0.34, color=ACT, label="Circuit (2 heads)" if i == 0 else None)
        ax.bar(i + 0.18, rnd.mean(), width=0.34, color=CONTROL,
               label="Random 2-head sets, mean" if i == 0 else None)
        ax.scatter(np.full(len(rnd), i + 0.18), rnd, color=REF, s=12, zorder=3,
                   label="Individual random sets (5)" if i == 0 else None)
        ax.text(i - 0.18, c + 0.02, f"{c:.1%}", ha="center", fontsize=9.5)
        ax.text(i + 0.18, max(rnd.mean(), 0) + 0.02, f"{rnd.mean():.1%}", ha="center", fontsize=9.5)
    ax.axhline(0, color=REF, linewidth=0.8)
    ax.set_xticks([0, 1], [x[1] for x in dirs])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.legend_outside(ax)
    cs.finish(fig, ax, "On held-out pairs the two-head circuit restores 51% and 21% of the gap while random "
              "head pairs restore about zero", "", "Share of abstention gap restored",
              path("f11_circuit_validation"), context="99 test pairs")


def fig_steering_curves() -> None:
    d = pd.read_csv(CIRC / "steering_curves.csv")
    fig, ax = cs.new_fig(7.2, 4.0)
    for cond, colour, label in [("se_wrapper", EXTERNAL, "SE wrapper (outside the model)"),
                                ("se_gated_circuit", ACT, "SE-gated circuit steering, dose 1"),
                                ("control_random_heads", CONTROL, "SE-gated steering at random heads")]:
        g = d[d.condition == cond]
        ax.plot(g["answered_share"], g["wrong_among_answered"], color=colour, marker="o", markersize=3.5,
                linewidth=1.8, label=label)
    ax.scatter([0.913], [0.330], color=BASE, s=40, zorder=3, label="Option E only")
    ax.annotate("Random heads: all 13 thresholds\nfall on the option E point", (0.913, 0.330),
                xytext=(0.62, 0.315), fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="-", color=REF, linewidth=0.8))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.legend_outside(ax, title="Each point is one SE threshold")
    cs.finish(fig, ax, "Circuit steering at dose 1 traces the same trade-off as the external SE wrapper over "
              "the range it covers, and random heads do nothing", "Share of questions answered",
              "Wrong among answered", path("f12_se_steering_curves"), context="Test role, 2,388 questions")


def fig_flips() -> None:
    d = pd.read_csv(CIRC / "steering_flips.csv")
    fig, ax = cs.new_fig(6.6, 3.8)
    groups = [("right before", "Right before steering\n(n = 646)"), ("wrong before", "Wrong before steering\n(n = 531)")]
    conds = [("circuit", ACT, "Circuit heads"), ("random_heads", CONTROL, "Random heads"),
             ("random_vectors", BASE, "Random vectors at circuit heads")]
    w = 0.26
    for j, (cond, colour, label) in enumerate(conds):
        vals = [float(d[(d.steering == cond) & (d.group == g)]["flipped_to_e"].iloc[0]) for g, _ in groups]
        xs = np.arange(2) + (j - 1) * w
        ax.bar(xs, vals, width=w, color=colour, label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_ylim(0, 0.72)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.legend_outside(ax, title="Steered at")
    cs.finish(fig, ax, "Steering changes wrong answers to E more often than right answers (60% vs 46%), so it "
              "is partly selective", "", "Share changed to \"I don't know\"", path("f13_steering_flips"),
              context="Real test questions above the SE gate, dose 1")


def fig_readout(t: pd.DataFrame) -> None:
    fig, ax = cs.new_fig(7.2, 3.9)
    bins = np.linspace(t["readout"].min(), t["readout"].max(), 40)
    for kind, colour, label in [("known", KNOWN, "Real, answered correctly"),
                                ("wrong", WRONG, "Real, answered incorrectly"),
                                ("invented", INVENTED, "Invented entity")]:
        ax.hist(t.loc[t.kind == kind, "readout"], bins=bins, density=True, histtype="step", linewidth=1.8,
                color=colour, label=label)
    auc_inv = roc_auc_score(t["invented"], t["readout"])
    real = t[~t["invented"]]
    auc_real = roc_auc_score(~real["forced_correct"], real["readout"])
    ax.text(0.98, 0.95, f"AUROC invented vs real: {auc_inv:.3f}\nAUROC wrong vs right (real): {auc_real:.3f}",
            ha="right", va="top", fontsize=9, transform=ax.transAxes)
    cs.legend_outside(ax, title="Question type")
    cs.finish(fig, ax, "The circuit readout separates invented from real entities almost perfectly but only "
              "weakly separates wrong from right real answers", "Circuit readout (projection on steering "
              "direction, summed over 2 heads)", "Density", path("f14_circuit_readout"),
              context="Test role, unsteered pass")


def fig_tradeoff() -> None:
    d = pd.read_csv(CIRC / "tiers_test.csv")
    styles = {"se_wrapper": (EXTERNAL, "SE wrapper"), "circuit_one_threshold": (CONTROL, "Circuit, one threshold"),
              "circuit_tiered": (ACT, "Circuit, tiered"), "circuit_tiered_readout": (ACT_DARK, "Circuit, tiered + readout")}
    fig, ax = cs.new_fig(7.2, 4.4)
    for cond, (colour, _) in styles.items():
        g = d[d.condition == cond].sort_values("wrong_cost")
        ax.plot(g["abstain_on_known"], g["abstain_on_unknown"], color=colour, linewidth=0.8, alpha=0.6)
        for _, r in g.iterrows():
            ax.scatter(r["abstain_on_known"], r["abstain_on_unknown"], color=colour,
                       marker=COST_MARK[r["wrong_cost"]], s=55, zorder=3)
    o = d[d.condition == "option_e_only"].iloc[0]
    ax.scatter(o["abstain_on_known"], o["abstain_on_unknown"], color=BASE, marker="D", s=40, zorder=3)
    handles = [Patch(color=c, label=l) for c, l in styles.values()]
    handles.append(Patch(color=BASE, label="Option E only"))
    handles += [Line2D([], [], color=REF, marker=m, linestyle="", label=f"Wrong-answer cost {int(c)}")
                for c, m in COST_MARK.items()]
    cs.legend_outside(ax, handles=handles, labels=[h.get_label() for h in handles])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.finish(fig, ax, "At every cost the tiered controller with the readout sits above or left of the SE "
              "wrapper, abstaining as often on unknown questions with fewer false abstentions",
              "Abstains on known questions (false abstention)", "Abstains on unknown questions",
              path("f15_tiered_tradeoff"), context="Test role; upper left is better")


def fig_paired_utility() -> None:
    d = pd.read_csv(CIRC / "tiers_comparisons.csv")
    d = d[d.measure == "utility"].copy()
    names = {("circuit_tiered_readout", "se_wrapper"): "Tiered + readout vs SE wrapper",
             ("circuit_tiered_readout", "circuit_one_threshold"): "Tiered + readout vs one threshold",
             ("circuit_tiered", "circuit_one_threshold"): "Tiered vs one threshold"}
    d["label"] = [names[(a, b)] for a, b in zip(d.controller, d.baseline)]
    fig, ax = cs.new_fig(7.2, 4.2)
    rows = []
    for cost in (1.0, 2.0, 4.0):
        for lab in names.values():
            rows.append((cost, lab))
    y = np.arange(len(rows))[::-1]
    colours = {1.0: CONTROL, 2.0: ACT, 4.0: ACT_DARK}
    for yy, (cost, lab) in zip(y, rows):
        r = d[(d.wrong_cost == cost) & (d.label == lab)].iloc[0]
        ax.errorbar(r["difference"], yy, xerr=[[r["difference"] - r["low_95"]], [r["high_95"] - r["difference"]]],
                    fmt=COST_MARK[cost], color=colours[cost], capsize=3, markersize=6)
    ax.axvline(0, color=REF, linewidth=0.9)
    ax.annotate("Same schedule chosen", (0.0, y[-1]), xytext=(0.004, y[-1]), va="center", fontsize=9)
    ax.set_yticks(y, [f"Cost {int(c)}: {lab[0].lower() + lab[1:]}" for c, lab in rows])
    cs.finish(fig, ax, "Tiered steering with the readout improves utility at wrong-answer costs 2 and 4; at "
              "cost 1 every interval includes zero", "Difference in utility (95% group-bootstrap interval)",
              "", path("f16_paired_utility_differences"),
              context="Test role, 2,000 bootstrap resamples of question groups")


def fig_bands() -> None:
    d = pd.read_csv(CIRC / "tiers_bands.csv")
    d = d[d.wrong_cost == 2.0]
    bands = ["0.0 to 0.2", "0.2 to 0.6", "0.6 to 1.0", "above 1.0"]
    conds = [("option_e_only", BASE, "Option E only"), ("se_wrapper", EXTERNAL, "SE wrapper"),
             ("circuit_tiered_readout", ACT_DARK, "Circuit, tiered + readout")]
    fig, ax = cs.new_fig(7.2, 3.9)
    w = 0.26
    for j, (cond, colour, label) in enumerate(conds):
        vals = [float(d[(d.condition == cond) & (d.se_band == b)]["abstain_on_unknown"].iloc[0]) for b in bands]
        xs = np.arange(len(bands)) + (j - 1) * w
        ax.bar(xs, vals, width=w, color=colour, label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.5)
    ns = [int(d[(d.condition == "se_wrapper") & (d.se_band == b)]["n"].iloc[0]) for b in bands]
    ax.set_xticks(range(len(bands)), [f"{b} nats\n(n = {n:,})" for b, n in zip(bands, ns)])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cs.legend_outside(ax, title="Controller")
    cs.finish(fig, ax, "Below 0.2 nats the readout trigger catches 41% of unknown questions, where SE-based "
              "methods catch 7%", "Semantic entropy band", "Abstains on unknown questions",
              path("f17_abstention_by_se_band"), context="Test role, wrong-answer cost 2")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = tiers()
    fig_se_by_type(t)
    fig_option_e(t)
    fig_layer_sweep()
    fig_certification()
    fig_risk_coverage()
    fig_auroc()
    fig_error_direction()
    fig_residual_patching()
    fig_head_heatmap()
    fig_circuit_size()
    fig_circuit_validation()
    fig_steering_curves()
    fig_flips()
    fig_readout(t)
    fig_tradeoff()
    fig_paired_utility()
    fig_bands()
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
