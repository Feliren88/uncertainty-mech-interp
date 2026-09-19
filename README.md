# Health "I don't know" gate

This code lets Llama 3.1 8B Instruct answer a medical multiple-choice question only when a probe on its own activations says the answer is likely right. Otherwise it replies exactly `I don't know.` It implements the inference gate from the research design in `../uncertainty-mech-interp/` (RFC Version 1.3 and `protocol.md`) and test-runs it on health questions. A follow-on study finds the attention heads that make the model say "I don't know" and switches them on with semantic entropy (see the circuit section below).

The full narrative write-up of all four experiments, with methodology, results, discussion and references, is in [docs/report/2026-09-19-health-idk-report.md](docs/report/2026-09-19-health-idk-report.md).

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

### A single residual direction did not steer abstention

The gate run also tried the simplest intervention. It added the "error direction" (mean activation for wrong answers minus right answers) to layer 19 at every position. That did not raise the probability of "I don't know" beyond random directions of the same size, and at four times the class difference it cut MedQA accuracy from 67.6% to 55.3%. The direction reads errors but does not drive abstention. The circuit study below finds components that do.

### The first run failed, and why

The first test run (4,600 questions, `configs/health_test_run.toml`) certified no threshold. Its 427 independent calibration questions could not rule out an error rate above 10%. The tightest bound was 15.9%. Following the protocol, that gate answered nothing. The second run used all 10,000 unique MedQA training questions and gave the calibration role 30% of them (3,148 independent units). It also widened the probe regularization grid, because the first run picked the grid's edge. The 10% target stayed fixed. `docs/specs/2026-09-18-health-idk-gate-design.md` records both changes. Because the two runs share questions, the second run counts as a larger test run. A confirmatory study would need fresh questions.

## The "I don't know" circuit and semantic-entropy steering

Llama 3.1 8B already has an "I don't know" pathway, and we can find it. It fires when the entity in a question is unfamiliar, and it runs through a few attention heads in layers 15 to 17. Copying the outputs of two of them (L15.H4 and L17.H25) moves half of the way from "answer" to "I don't know". Switching those heads on whenever semantic entropy is high makes the model say "I don't know" itself. On MedQA test questions, its error rate among answers falls from 33.0% to 21.3%, and it refuses 86.9% of invented-entity questions instead of 60.6%. Random heads do nothing. On real questions it matches, but does not beat, simply thresholding semantic entropy outside the model. Full report: `runs/circuit-se-steering/report.md`.

### The pathway follows entity familiarity

| Question type | Median semantic entropy | Picks "I don't know" when offered |
|---|---|---|
| Real, answered correctly | 0.13 nats | 0% |
| Real, answered wrong | 0.75 nats | 2% |
| Invented entity | 0.74 nats | 63% |

Semantic entropy (the spread of the model's answer distribution over the four options) is as high on real questions it gets wrong as on invented ones, yet it only abstains on the invented ones.

### Where the signal travels

Activation patching copies one internal activation from a prompt about an invented entity into the matched prompt about a real one, and measures how far the abstention gap (log P("I don't know") minus log P(any answer)) moves toward the invented prompt. The study uses 225 matched pairs built from 75 well-known health facts.

1. **Entity token, layers 0 to 6.** Patching the last entity token moves 21% to 27% of the gap. The effect fades by layer 10.
2. **Instruction tail, layers 12 to 16.** The signal passes through the shared instruction tokens, peaking at 47% at layer 14.
3. **Final token, from layer 14.** The final token carries 45% at layer 14, 91% at layer 15 and 98% at layer 18.

At the final token, single heads L15.H4 and L17.H25 each move 32% of the gap, and L30.H27 moves 31%. One head, L30.H25, moves -51%, so on invented prompts it pushes against "I don't know". On discovery pairs, the top 8 heads together move 92%.

Patching these heads changes whether the model abstains but not its semantic entropy (change 0.001 nats). In this model, abstaining and being unsure about the answer look like separate mechanisms. That fits the earlier finding that offering "I don't know" barely helps on hard real questions.

### The circuit on held-out pairs

| Patch on 99 test pairs | Circuit (2 heads) | 5 random sets of 2 heads |
|---|---|---|
| Invented into real (sufficiency) | 51.0% of the gap | -1.9% (-9.4% to 0.2%) |
| Real into invented (necessity) | 21.0%; abstention drops from 76.8% to 25.3% | -0.2% |

### Switching the circuit on with semantic entropy

Steering adds the circuit's "invented minus real" output to its two heads at the final token. All conditions use the prompt that offers "E. I don't know", so the reply is the model's own. Dose and threshold were chosen on the second run's calibration role and scored once on its test role (2,388 questions).

| Condition | Answered | Wrong among answered | Invented refused | Net correct |
|---|---|---|---|---|
| Option E only | 91.3% | 33.0% | 60.6% | 0.311 |
| SE wrapper (reply "I don't know" when SE > 0.7) | 64.5% | 21.9% | 73.4% | 0.362 |
| SE-gated circuit steering (SE > 0.2, dose 1) | 62.2% | 21.3% | 86.9% | 0.357 |
| SE-scaled circuit steering (dose 2 x SE / ln 4) | 63.4% | 21.1% | 83.8% | 0.366 |
| Control: same gate, random heads | 91.3% | 33.0% | 60.9% | 0.310 |
| Control: same gate, random vectors at the circuit heads | 90.2% | 32.7% | 64.3% | 0.312 |

Net correct is right answers minus wrong answers, per question. Among real questions above the gate, circuit steering turned 60.1% of wrong answers and 46.1% of right answers into "I don't know"; random heads turned none. The push is somewhat selective, and the model's own evidence sets which answers flip. At dose 4 and above every gated answer flips, and the result equals the SE wrapper at the same threshold.

### Several SE thresholds plus the circuit's own readout

One threshold either leaves a question alone or pushes the circuit at a single dose. A tiered schedule uses up to three SE thresholds, each with a stronger push, so the model's own evidence can still resist a mild push on borderline questions. A second trigger reads the circuit itself, measuring how far heads L15.H4 and L17.H25 point along their "invented" direction in the plain pass. That readout separates invented from real questions with AUROC 0.994, so it catches confident hallucinations about unfamiliar entities that SE misses. Schedules are chosen on calibration questions to maximize right answers minus c times wrong answers, where c is the cost of a wrong answer, and scored once on the 2,388 test questions. Full report: `runs/tiered-se-steering/report.md`.

| Wrong-answer cost c | Controller (chosen on calibration) | Abstains on unknown | Abstains on known | Invented refused | Utility |
|---|---|---|---|---|---|
| any | Option E only | 21.6% | 0.8% | 60.6% | |
| 2 | SE wrapper: reply E when SE > 0.2 | 90.1% | 44.3% | 89.6% | 0.273 |
| 2 | Tiered: SE > 0.2 dose 0.5, > 0.4 dose 2, > 0.6 dose 4; readout high: dose 4 | 87.5% | 34.7% | 98.7% | 0.307 |
| 4 | SE wrapper: reply E when SE > 0.1 | 93.7% | 54.1% | 91.9% | 0.190 |
| 4 | Tiered: SE > 0.2 dose 4; readout high: dose 4 | 93.7% | 46.4% | 99.0% | 0.236 |

"Unknown" means invented entities plus real questions the model gets wrong when forced to choose; "known" means real questions it gets right. Paired differences against the SE wrapper, with 95% intervals from resampling question groups:

- **c = 4:** utility +0.046 [+0.020, +0.071]. Abstention on unknown questions is unchanged (+0.0 points [-1.5, +1.5]), and false abstention falls by 7.7 points [6.1, 9.3].
- **c = 2:** utility +0.034 [+0.014, +0.053]. The tiered controller gives up 2.7 points [0.6, 4.7] of abstention on unknown questions and saves 9.5 points [7.8, 11.5] of false abstention. Tiers without the readout already help here (+0.019 [+0.005, +0.033] over one threshold).
- **c = 1:** no clear utility difference (+0.008 [-0.008, +0.025]). The tiered controller abstains more on both unknown and known questions.

The readout does most of its work where SE is blind. Among questions with SE below 0.2 nats, the tiered controller at c = 2 abstains on 40.6% of the unknown ones and 4.0% of the known ones; every SE-only method abstains on 7.3% of the unknown ones there. Random heads under the same schedule change nothing (utility 0.005 against 0.010 for option E alone at c = 2).

The price is false abstention. Between 0.2 and 0.6 nats the controller still refuses 44% of questions the model knows, because SE does not separate right from wrong answers well in that band, and the readout separates them only weakly on real questions (AUROC 0.646).

### Limits of the circuit study

- Patching at the final token shows where the model reads the signal. The route from the entity to the tail is only partly mapped.
- Invented names differ from real ones in form as well as familiarity. Real drug names carry class suffixes such as -pril or -statin, and the pairs cannot separate the two.
- The steering vectors come from templated pairs and transfer to MedQA prompts in this one format; other formats need their own test.

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
python -m uncertainty_mech circuits --config configs/circuit_study.toml  # patching and steering, about 1 hour
python -m uncertainty_mech tiers --config configs/tiers.toml             # tiered steering only, about 15 min
```

`circuits` reads the full run's directory (its model, items, roles and saved answer distributions) and writes `runs/circuit-se-steering/`: `report.md`, four figures, the steering vectors, and a CSV for every sweep, including per-question predictions under each steering condition. `tiers` reruns only the tiered stage from those saved vectors, so schedules can be revised without repeating the patching sweeps.

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
| Application | `src/uncertainty_mech/application/` | Use cases (dataset, gates, evaluation, steering, answering, report; entity pairs, patching sweeps, SE steering) and the ports they call, including a `CircuitModel` port that captures, patches and adds to activations | Domain |
| Infrastructure | `src/uncertainty_mech/infrastructure/` | Hugging Face model with hooks on the residual stream, attention-head outputs and MLPs; MedQA source; invented-question generator; curated health facts and matched pairs; file store; figures | Application ports, domain |
| Composition root | `src/uncertainty_mech/cli.py` | Wires adapters to use cases for `run` and `ask` | Everything |

The end-to-end test swaps the model and MedQA for fakes through the same ports, so it runs the real pipeline code on a CPU.

## Tests

As requested, the tests are end to end only. `tests/test_end_to_end.py` runs `run` and then `ask` through the CLI, on a fake model that knows 70% of a pool of synthetic questions. It checks five things:

- every artifact is written;
- groups never cross roles, and every role holds both real and invented questions;
- the chosen threshold obeys the certificate rule;
- a zero-dose steering hook leaves the output unchanged;
- `ask` answers a known question and says `I don't know.` to an invented one.

`tests/test_circuit_study.py` runs `run` and then `circuits` on a fake four-layer model with one planted abstention head (layer 2, head 1). It checks that the head sweep ranks the planted head first, that the circuit beats random heads on held-out pairs, that SE-gated steering turns uncertain answers into "I don't know" while random heads do not, that tiered controllers score at least as well as simpler ones on calibration data and beat random heads on test, that the `tiers` command reproduces the stage from saved vectors with valid bootstrap intervals, and that every artifact is written.

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
