# Health "I don't know" gate

This code lets Llama 3.1 8B Instruct answer a medical multiple-choice question only when a probe on its own activations says the answer is likely right. Otherwise it replies exactly `I don't know.` It implements the inference gate from the research design in `../uncertainty-mech-interp/` (RFC Version 1.3 and `protocol.md`) and test-runs it on health questions.

## Result

On 2,388 held-out test questions, the gate answered 745 (31.2%) and got 44 of them wrong (5.9%). Without the gate, the model answered everything and was wrong on 37.7%. The target was an error rate of at most 10% among answered questions, with 95% confidence. On the test data the one-sided 95% upper bound is 7.5%, so the held-out result agrees with the calibration certificate.

| Method | Answered | Wrong among answered | Invented questions refused | Right answers kept |
|---|---|---|---|---|
| No gate | 2,388 | 37.7% | 0% | 100% |
| Prompt offers "E. I don't know" | 2,180 | 33.1% | 60.9% | 98.0% |
| Gate on output statistics | 706 | 6.4% | 94.9% | 44.5% |
| Gate on activations | 744 | 6.5% | 100% | 46.8% |
| Gate on both (primary) | 745 | 5.9% | 100% | 47.1% |

Source: `runs/health-llama31-8b-full-run/report.md`, test role. "Invented questions" ask about drugs and diseases that do not exist, so every answer to them is a hallucination.

1. **Asking the model to say "I don't know" barely helps on real questions.** With option E offered, it still answered 98.7% of MedQA questions and was wrong on 29.4% of them.
2. **The gate refused every invented question.** All 297 got `I don't know.` The output-statistics gate let 15 through, and the prompt let 116 through.
3. **On real MedQA questions, activations did not beat output statistics.** Test AUROC (the chance that a wrong answer gets a higher risk than a right one) was 0.78 for the combined gate and 0.80 for output statistics alone. The activation features help with invented entities and add nothing on hard real questions.
4. **The price is coverage.** To keep errors under 10%, the gate answers about a third of questions and drops about half of the answers the model would have gotten right.

A stricter gate looks possible. After the run, the calibration table showed that threshold 0.07 would have certified the RFC's original 5% target (588 of 3,148 calibration questions answered, 14 wrong, bound 4.7%). This reading came after the run, from calibration data, and the test role never checked it.

Demo, `examples/health_questions.jsonl`:

| Question | Gate reply | Risk |
|---|---|---|
| First-line drug for new type 2 diabetes | A. Metformin | 0.014 |
| Starting dose of "Velquarin" (invented drug) | I don't know. | 0.941 |
| Gene behind "Marrowick-Tessel syndrome" (invented) | I don't know. | 0.997 |
| Treatment for low T4 with raised TSH | B. Levothyroxine | 0.096 |

### Where the signal lives

A probe on the residual stream (the vector that each transformer layer reads from and adds to) at the last prompt token predicts a wrong answer with AUROC 0.70 at layer 0. It rises between layers 12 and 18 and levels off at 0.83 from layer 18. The gate uses layer 19. See `runs/health-llama31-8b-full-run/figures/layer_sweep.png`.

### The steering check found no causal handle on "I don't know"

The "error direction" is the mean activation for wrong answers minus the mean for right answers. Adding it to layer 19 did not raise the probability that the model picks "I don't know" beyond what random directions of the same size did. At four times the class difference it cut MedQA accuracy from 67.6% to 55.3%. Subtracting it moved the probability of "I don't know" from 0.087 to 0.099, a small shift in the opposite direction to the hypothesis. The direction reads errors, but pushing along it at this layer does not make the model abstain. The check is exploratory, with one layer, one direction and 200 prompts.

### The first run failed, and why

The first test run (4,600 questions, `configs/health_test_run.toml`) certified no threshold. Its 427 independent calibration questions could not rule out an error rate above 10%. The tightest bound was 15.9%. Following the protocol, that gate answered nothing. The second run used all 10,000 unique MedQA training questions and gave the calibration role 30% of them (3,148 independent units). It also widened the probe regularization grid, because the first run picked the grid's edge. The 10% target stayed fixed. `docs/specs/2026-09-18-health-idk-gate-design.md` records both changes. Because the two runs share questions, the second run counts as a larger test run. A confirmatory study would need fresh questions.

## How it works

One forward pass per question reads the probabilities of answer letters A to D, the total probability on answer letters, and the residual stream at all 32 layers, all at the last prompt token. Forward hooks on `model.model.layers[i]` capture it, following ARENA 1.3.1.

1. **Data.** MedQA-USMLE 4-option questions (real exam questions with known answers) and three questions for each invented entity. Related questions share a group. Whole groups go to one of four roles: `discovery`, `cal_prob`, `cal_gate` and `test`.
2. **Layer sweep.** On discovery data, a logistic probe per layer predicts whether the chosen answer is wrong. Grouped 5-fold cross-validation picks the layer.
3. **Risk model.** Logistic regression on that layer plus the output statistics gives a risk score. One temperature (a scale on the score) is fitted on `cal_prob` so risk matches observed error rates.
4. **Certified threshold.** On `cal_gate`, each of 20 fixed thresholds gets a Clopper-Pearson upper bound (an exact binomial confidence bound) on the error rate of the answers it would release, with the 5% error budget split across the 20. The threshold that answers the most while keeping its bound at or below 10% wins. If none passes, the gate answers nothing.
5. **Test.** The frozen gate runs once on the test role. The report compares it with no gate, the option-E prompt, and gates built from output statistics or activations alone.
6. **Steering check.** Described above.
7. **Answering.** `ask` runs the frozen gate on new questions and appends an audit line per answer.

## Run it

```bash
cd uncertainty-mech-interp/code
export HF_HOME=/home/vfeliren1/lf93_scratch2/vfvic1/hf_cache/huggingface
export PYTHONPATH=src

python -m pytest -q                                                   # end-to-end test on a fake model, about 10 s
python -m uncertainty_mech run --config configs/health_smoke.toml     # 120 questions on the GPU, about 2 min
python -m uncertainty_mech run --config configs/health_full_run.toml  # 11,500 questions, about 30 min on one A100
python -m uncertainty_mech ask --run-dir runs/health-llama31-8b-full-run \
    --questions examples/health_questions.jsonl
```

`scripts/test_run.sh` runs the test, the smoke run, the 4,600-question run and the demo in one go.

`ask` reads JSON lines with a `question` and four `options`, and prints one JSON line per question:

```json
{"text": "A. Metformin", "decision": "answer", "reason_code": "ANSWERED", "risk": 0.014, "candidate_letter": "A"}
```

`reason_code` is `ANSWERED`, `HIGH_RISK`, `NO_CERTIFIED_THRESHOLD` or `INVALID_SCORE`. Every abstention returns exactly `I don't know.`

Each run writes `runs/<run_id>/`: `report.md`, `figures/`, CSV tables, `summary.json`, the frozen gate (`bundle.json`, `bundle_risk_model.npz`), the raw readings (`readings.npz`, 5.8 GB for the full run) and `audit.jsonl`.

## Code layout

Dependencies point inward. The domain knows nothing about models or files, and only one module imports `torch`.

| Layer | Folder | Holds | Imports |
|---|---|---|---|
| Domain | `src/uncertainty_mech/domain/` | Questions and roles, prompts, grading, the risk certificate, metrics, the release decision | NumPy, SciPy |
| Application | `src/uncertainty_mech/application/` | Use cases (dataset, gates, evaluation, steering, answering, report) and the ports they call | Domain |
| Infrastructure | `src/uncertainty_mech/infrastructure/` | Hugging Face model with hooks, MedQA source, invented-question generator, file store, figures | Application ports, domain |
| Composition root | `src/uncertainty_mech/cli.py` | Wires adapters to use cases for `run` and `ask` | Everything |

The end-to-end test swaps the model and MedQA for fakes through the same ports, so it runs the real pipeline code on a CPU.

## Tests

As requested, the tests are end to end only. `tests/test_end_to_end.py` runs `run` and then `ask` through the CLI, on a fake model that knows 70% of a pool of synthetic questions. It checks five things:

- every artifact is written;
- groups never cross roles, and every role holds both real and invented questions;
- the chosen threshold obeys the certificate rule;
- a zero-dose steering hook leaves the output unchanged;
- `ask` answers a known question and says `I don't know.` to an invented one.

The GPU runs are the real end-to-end tests. The smoke run caught one bug the fake could not. MedQA sampling reused the hash that assigns roles, so every sampled question landed in `discovery`. Sampling now has its own hash stream, and the test checks role balance.

## Changes from the RFC

| RFC | Here | Why |
|---|---|---|
| Gemma 2 2B | Llama 3.1 8B Instruct, revision `0e9e39f2` | Gemma 2 is gated for this account (HTTP 403) |
| SQuAD 2.0, TriviaQA, synthetic worlds | MedQA-USMLE plus invented drugs and diseases | Health application |
| Free-text answers | Answer letter from one forward pass | Exact grading, no judge model |
| 5% risk target | 10% | Fixed before any model output, when the plan had about 450 calibration questions |
| SAEs, path patching, TruthfulQA | Not done | Out of scope for this test run |

## Limits

- One model, one prompt format, one benchmark. Free-text answers would need new calibration.
- The certificate holds only for questions drawn the way the calibration set was, with the same mix of real and invented items, the same model revision and the same prompt.
- MedQA is public and may be in the model's pretraining data.
- Invented entities are easier to catch than wrong answers to real questions. Read the real-only table in the report before the pooled one.
- Nothing here is medical advice or a validated clinical tool.
