# Health "I don't know" gate: design

Status: approved design for a first end-to-end test run, 18 September 2026.
Parent documents: `../uncertainty-mech-interp/protocol.md` and `rfc.pdf` (Version 1.3).

## Goal

Build the RFC's inference gate for health questions. The model answers a medical
question only when a probe on its internal activations says the answer is likely
correct. Otherwise it returns exactly `I don't know.` The run must go end to end:
data, activations, probes, calibration, certified threshold, final test, a causal
steering check, and a command that answers new questions.

This is a test run of the whole pipeline at modest scale. It is not the full RFC
study. It skips SAEs, path patching and TruthfulQA transfer.

## Decisions that differ from the RFC

| RFC | This run | Reason |
|---|---|---|
| `google/gemma-2-2b-it` | `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f2` | Gemma 2 is gated for this account (HTTP 403). Llama 3.1 8B is cached locally. |
| SQuAD 2.0, TriviaQA, synthetic worlds | MedQA-USMLE 4-option (real) plus fictional health items | The user asked for a health application. |
| Free-text answers, 64 tokens | One forward pass, answer letter read from restricted next-token logits | Grading becomes exact; no LLM judge needed. |
| Target selective risk 5% | 10% | About 450 calibration items cannot certify 5% at useful coverage (RFC section 5 needs at least 117 zero-error accepted cases). Fixed before any model output. |
| Ten-sample semantic entropy | Entropy of the answer-letter distribution | For multiple choice the answer clusters are the letters, so the entropy is exact without sampling. |

## Task and labels

Each item is a multiple-choice health question with four options, A to D.

- **Real items** come from `GBaker/MedQA-USMLE-4-options` (train file). The
  candidate is wrong when the chosen letter differs from `answer_idx`.
- **Fictional items** ask about invented drugs and diseases, for example the
  mechanism of "Zorvaxil" or the gene behind "Tarvenick syndrome". Options are
  real drugs, genes or organisms. No option is supported, so every substantive
  answer is an error (`e = 1`). The only right behavior is to abstain.

Fictional names are built from syllables, avoid known drug-class suffixes
(-olol, -pril, -statin and so on), and are rejected if they appear in the MedQA text.

## Data roles

Groups, not rows, go to roles. A real group is one deduplicated question. A
fictional group is one invented entity with its three question templates.
A seeded hash of the group id assigns roles: 60% discovery, 10% `cal_prob`,
10% `cal_gate`, 20% test.

For the binomial certificate, each `cal_gate` group contributes one canonical
item, chosen by a seeded hash before any model output exists.

## Signals

One forward pass per prompt with the Llama chat template returns:

- log-probabilities over the option letters (restricted softmax),
- the total full-vocabulary probability on those letters ("letter mass"),
- the residual stream at the last prompt token for every decoder layer
  (`output_hidden_states`, the pattern used in ARENA 1.3.1).

Three risk signal sets are compared, each with its own calibrator and threshold:

1. **Output**: top probability, top-two margin, letter entropy, letter mass.
2. **Probe**: the residual stream at one layer chosen on discovery data.
3. **Combined**: output plus probe features.

A fourth comparator needs no fitting. The **prompt-only baseline** adds
option `E. I don't know` and abstains when the model picks E.

## Pipeline

1. **Build data.** Load, deduplicate and sample real items (seed 17),
   generate fictional items, assign roles.
2. **Read the model.** Forward pass for every item, with and without option E.
   Save readings to disk.
3. **Layer sweep.** On discovery only: grouped 5-fold AUROC of an L2 logistic
   probe for each layer. The best mean AUROC picks the probe layer.
4. **Fit risk models.** On discovery: standardize, L2 logistic regression,
   `C` chosen from `{0.01, 0.1, 1, 10}` by grouped CV.
5. **Calibrate.** On `cal_prob`: one temperature `T` minimizing Bernoulli NLL
   of `sigmoid(logit / T)`.
6. **Select threshold.** On `cal_gate` canonical items: the RFC's 20-value grid,
   one-sided Clopper-Pearson bound at `delta / M = 0.05 / 20`, keep the threshold
   with the highest coverage whose bound is at most 10%. If none passes, the gate
   abstains on everything and the report says the target was not met.
7. **Final test.** Apply frozen artifacts once. Report coverage, selective risk,
   false-answer rate, correct-answer retention, fictional abstention recall,
   unnecessary abstention, Brier, AUROC, ECE (10 bins), AURC, and the held-out
   one-sided bound. Report all items, real only, and fictional only.
8. **Steering check (exploratory).** Difference-of-means "error direction" at
   the probe layer (ARENA 1.3.1 style). Add it at doses -1, -0.5, 0, 0.5, 1 to
   the probe layer on test prompts that include option E. Measure the change in
   P(E) and in real-item accuracy. Control: random directions of equal norm.
   The dose-0 hook must reproduce unhooked log-probabilities (identity check).
9. **Report.** Markdown report, CSV tables, two figures (layer sweep,
   risk-coverage curves).

## Architecture

```
src/uncertainty_mech/
  domain/          pure Python and NumPy; no torch, no I/O
    questions.py   HealthQuestion, Stratum, Role
    splits.py      seeded group-to-role assignment, canonical items
    prompts.py     prompt text with and without option E
    grading.py     error labels, output features
    risk_control.py  Clopper-Pearson bound, threshold selection
    metrics.py     selective and calibration metrics
    gate.py        Decision, ReasonCode, decide(), render()
  application/     use cases; depends on domain and ports only
    ports.py       QuestionSource, LanguageModel, Steering, Readings
    config.py      RunConfig loaded from TOML
    risk_model.py  layer sweep, logistic probe, temperature calibrator
    pipeline.py    stages 1 to 7
    steering.py    stage 8
    answer.py      AbstainingAnswerer (the inference service)
    report.py      report tables and markdown
  infrastructure/  adapters
    hf_model.py    transformers adapter, hidden states, forward hooks
    medqa.py       MedQA source
    fictional.py   fictional health item generator
    store.py       run directory (JSON, NPZ, CSV)
    figures.py     matplotlib figures
  cli.py           composition root: `run` and `ask`
```

Dependencies point inward. The domain imports nothing from the other layers.
Application code talks to the model only through the `LanguageModel` port, so
the end-to-end test swaps in a fake model without touching the pipeline.

## Inference contract

`ask` takes a question and options, returns
`{text, decision, reason_code, risk}` and appends an audit line. The text is the
chosen option or exactly `I don't know.` Reason codes:

- `ANSWERED`: calibrated risk at or below the frozen threshold.
- `HIGH_RISK`: risk above the threshold.
- `NO_CERTIFIED_THRESHOLD`: calibration found no threshold meeting the target.
- `INVALID_SCORE`: risk missing or not finite.

## Testing

End-to-end only, as requested.

- `tests/test_end_to_end.py` runs `run` and `ask` through the CLI composition
  on a fake language model and a small in-memory real-question source. It checks
  that roles do not share groups, the chosen threshold obeys the certificate
  rule, abstentions render exactly `I don't know.`, and every report file exists.
- The GPU test run executes the same pipeline with Llama 3.1 8B on the health
  data, then asks a few held-out and hand-written questions.

## Honest scope

Results describe one model, one prompt format and one sample. A good gate here
does not show a general uncertainty circuit. The steering check is exploratory.
Fictional items are easier to detect than wrong real answers, so real-only
metrics are reported next to the pooled ones.

## Addendum after the pilot, 19 September 2026

The 4,600-question test run (`configs/health_test_run.toml`) became the pilot.
It ran end to end but certified no threshold. Its 427 independent `cal_gate`
units gave a tightest upper bound of 15.9%, above the 10% target. Discovery
cross-validation also picked C = 0.01, the smallest value in the grid.

The second run (`configs/health_full_run.toml`) changes only the sample and
the C grid. It uses all 10,000 unique MedQA training questions and 500
invented entities, gives `cal_gate` 30% of groups, and extends the C grid to
0.0001. The 10% target, the threshold grid, delta and the primary signal stay
fixed. The pilot's test role informed no choice except the decision to rerun.
Because the two runs share data, the second run is a larger test run, not a
confirmatory study.
