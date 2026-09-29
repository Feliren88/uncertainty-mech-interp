# Illustrated methodology overview

The figure follows the completed study from acquiring questions and a model checkpoint to evaluating an external filter and an intervention inside the model. It replaces the overview chart in the [research README](../README.md#methodology).

The README uses a [TikZ document](f00_methodology_overview_tikz.tex) that places the original six-panel artwork at its native aspect ratio. Its [PDF](f00_methodology_overview_tikz.pdf) embeds the original image pixels unchanged. The [README PNG](f00_methodology_overview_tikz.png) is rendered from that PDF at 300 dpi; this resampling does not add detail to the original artwork. To copy and compile the TikZ document, keep [`f00_methodology_overview.png`](f00_methodology_overview.png) beside the `.tex` file and run `pdflatex f00_methodology_overview_tikz.tex` twice.

The [fully editable TikZ redraw](f00_methodology_overview_editable_redraw.tex) and its [vector PDF](f00_methodology_overview_editable_redraw.pdf) remain available for changing individual labels and shapes. That redraw simplifies the original illustrations and is **not visually identical**; it is not the image displayed in the README.

The [original PNG](f00_methodology_overview.png) is 1,024 × 1,536 pixels. Its [original PDF](f00_methodology_overview.pdf) is 6.9 × 10.35 inches, so the effective image resolution is about 148 pixels per inch.

## Scientific content

The panels show two experimental branches. The external filter in (c) estimates error risk and decides whether to release an answer. The circuit study in (d) identifies heads through activation patching; (e) steers those heads so the model chooses its own answer. The branches meet at held-out evaluation in (f).

The main pool contains 10,000 real MedQA questions and 1,500 invented-entity questions. Its test split contains 2,388 questions: 2,091 real and 297 invented. The matched-pair set is separate: 75 facts with three invented counterparts each give 225 pairs, or 450 prompts. Circuit selection uses 126 discovery pairs and validation uses 99 test pairs.

Semantic entropy uses the normalized probabilities of A to D from the forced-choice prompt. The circuit readout comes from a separate, unsteered prompt that offers option E. It sums each selected head's projection onto its unit steering direction. The steering vectors are mean invented-prompt activations minus mean real-prompt activations, fitted on discovery pairs. A selected dose multiplies each vector before addition to L15.H4 and L17.H25 at the final prompt token. Llama's weights stay fixed.

Question cards, model stacks, head grids, probability bars, calibration sketches and interval glyphs illustrate operations. They are not additional measured distributions or empirical heatmaps. The question-card snippets illustrate the prompt format rather than identify stored dataset rows. The Atorvastatin/Beleprax cards abbreviate the four inhibition mechanisms in the README's prompt example. The 51.0% restoration result is a fraction of the abstention gap recovered by patching, not a percentage of answers changed to E. The +0.001-nat SE change applies to the invented-into-real patch direction.

## Sources

| Content | Source |
|---|---|
| Model and data revisions, sample sizes and main split settings | [`health_full_run.toml`](../../../configs/health_full_run.toml) |
| Matched-pair construction and discovery/test split | [`entity_pairs.py`](../../../src/uncertainty_mech/infrastructure/entity_pairs.py), [`known_facts.py`](../../../src/uncertainty_mech/infrastructure/known_facts.py), [`circuit_study.toml`](../../../configs/circuit_study.toml) |
| Forward hooks and activation intervention | [`hf_model.py`](../../../src/uncertainty_mech/infrastructure/hf_model.py) |
| Patching and steering-vector construction | [`patching.py`](../../../src/uncertainty_mech/application/patching.py) |
| SE-triggered steering and the external SE comparison | [`se_steering.py`](../../../src/uncertainty_mech/application/se_steering.py) |
| Separate readout pass, tier selection and bootstrap comparisons | [`tiered_steering.py`](../../../src/uncertainty_mech/application/tiered_steering.py), [`tiers.toml`](../../../configs/tiers.toml) |
| Filter coverage, selective risk and upper bound | `runs/health-llama31-8b-full-run/metrics.csv` |
| Test question counts | `runs/health-llama31-8b-full-run/test_predictions.csv` |
| Head selection, restoration and SE change | `runs/circuit-se-steering/circuit.json`, `circuit_validation.csv` in the same directory |
| Utility differences and confidence intervals | `runs/circuit-se-steering/tiers_comparisons.csv` |
| Contribution of real and invented questions to utility gains | `runs/circuit-se-steering/tiers_predictions.csv` |

The saved `runs/` files are local experiment outputs and are excluded from version control. Their hashes and the values used in the illustration are recorded in [`methodology-overview.provenance.json`](methodology-overview.provenance.json).

## Original raster generation

The original raster rendering used the built-in imagegen tool. The local `paperbanana` launcher failed with `ModuleNotFoundError: No module named 'paperbanana'`, so the PaperBanana CLI did not run. Planning, styling and visual review were carried out inline using the reference–plan–style–render–critique sequence described in [PaperBanana](https://github.com/dwzhu-pku/PaperBanana). The style reference was its [diagram style guide](https://github.com/dwzhu-pku/PaperBanana/blob/main/style_guides/neurips2025_diagram_style_guide.md).

The stored prompts record the render and its revisions:

1. [Initial scientific and visual specification](methodology-overview.prompt.txt).
2. [Larger portrait layout and scientific corrections](methodology-overview.edit-prompt.txt).
3. [Branch connections, activation-copy arrows and readout glyph](methodology-overview.final-edit-prompt.txt).
4. [Final connector cleanup](methodology-overview.connector-edit-prompt.txt).

The prompts can be used again with imagegen. Generated images can vary between runs. No new experiment, model fine-tuning or statistical estimation was performed to produce this illustration.

## Checks

The visual review checked the two experimental branches, the matched-pair comparison, the direction of activation transfer, the invented-minus-real subtraction, the separate entropy and readout passes, the option-E prompt during steering, the two head names and the fixed-weight model. It also checked the distinction between probability spread and vector projection.

The numerical review compared the figure's labels with saved run tables, including the full-pool and test counts, filter metrics, patch restoration, direction-specific SE change, utility differences and confidence intervals. The image extracted from the faithful TikZ PDF was checked pixel for pixel against the original PNG. The README PNG was rendered from that PDF and visually checked against the original. The README's asset links were checked on disk.

## LaTeX inclusion

Use the faithful PDF as a full-page methods figure. The [TikZ source](f00_methodology_overview_tikz.tex) can be copied into a LaTeX project along with the [original PNG](f00_methodology_overview.png). Its image node preserves the artwork's appearance; the separate [redraw source](f00_methodology_overview_editable_redraw.tex) is available when individual objects need editing.

```latex
\begin{figure*}[p]
  \centering
  \includegraphics[width=\textwidth,height=0.92\textheight,keepaspectratio]{f00_methodology_overview_tikz.pdf}
  \caption{Overview of the medical-question abstention study. Data acquisition and frozen-model inference feed two branches: an external error filter and a circuit intervention controlled by semantic entropy and a separate readout. Both are evaluated on held-out questions. Illustrations are schematic; numerical labels summarize saved experiments.}
  \label{fig:abstention-methodology}
\end{figure*}
```
