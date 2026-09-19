# Abstention circuit and semantic-entropy steering: design

Status: design for a follow-on study, 19 September 2026. Builds on the health
gate (`2026-09-18-health-idk-gate-design.md`) and its second run
(`runs/health-llama31-8b-full-run`).

## Question

Can we find the components inside Llama 3.1 8B Instruct that make it answer
"I don't know", and use semantic entropy to switch them on so the model itself
abstains when it is unsure?

## What the second run already shows

With option E ("I don't know") offered, the model picks E on 63% of invented
entities but on only 2% of real MedQA questions it gets wrong. Semantic entropy
over the answer letters is about as high on those wrong real answers (median
0.75 nats) as on invented entities (0.74), and low on right answers (0.13). So
the model has an abstention mechanism, but it is driven by entity
familiarity, not by answer uncertainty. This matches the familiarity
hypothesis in RFC reference [1].

## Definitions

- **Semantic entropy (SE).** Entropy of the answer distribution over meanings.
  In multiple choice the meanings are the letters, so SE over A to D is exact
  without sampling (RFC section 4). The forced-choice SE comes from the prompt
  without option E.
- **Abstention gap (LD_E).** On the prompt with option E:
  `logit(E) - logsumexp(logit(A..D))`. Positive means the model prefers
  "I don't know".
- **Restoration.** For a patch from run X into run Y, `R = (m_patched - m_Y) /
  (m_X - m_Y)`, averaged over pairs before dividing. R = 1 means the patch moves
  the metric all the way.

## Data

**Matched pairs.** 75 curated real health facts in the health templates, for
example "A 45-year-old man is started on Atorvastatin. What is the mechanism of
action of this drug?" with the right option among four. Each fact is paired
with three invented names in the same sentence with the same options, for
example "started on Zoraxil". Pairs keep only real questions the model answers
correctly. Real entities split 50/50 into discovery and test, so a test entity
was never used to find the circuit. Invented names avoid real drug-class
suffixes, so pairs differ in both familiarity and morphology. That confound is
reported.

**MedQA.** Items, roles and forced-choice readings come from the second run.
Its `cal_prob` role picks steering settings; its `test` role evaluates them.

## Stage 1: localize with activation patching

All patches run on the prompt with option E and read both LD_E and SE over
A to D (renormalized) from the same pass. Positions count from the end of
the prompt, so a pair's two prompts align even when names differ in length.

1. **Residual stream, layer by segment.** Segments: last entity token, the
   rest of the question and options, the fixed instruction tail, the final
   token. Direction "invented into real": does this state make a known
   question abstain?
2. **Attention heads at the final token.** Patch one head's output (the input
   slice of `o_proj`) invented into real, for all 32 x 32 heads.
3. **MLP outputs at the final token**, all 32 layers.

## Stage 2: define and validate the circuit

Rank heads by their single-head effect on LD_E. The circuit is the smallest
top-k (k in 1, 2, 4, 8, 16, 32) whose joint patch reaches R >= 0.5 on
discovery pairs, or k = 32 if none does. On test pairs, report:

- sufficiency: patch the circuit invented into real;
- necessity: patch the circuit real into invented;
- the same for 5 draws of k random heads;
- the change in SE caused by each patch (RFC's Delta-H).

## Stage 3: semantic-entropy steering

The steering vector for circuit head h is the mean, over discovery pairs, of
its output on the invented prompt minus the real prompt, at the final token.
Steering adds `dose * v_h` to each circuit head at the final token.

Conditions on MedQA plus invented items, all on the prompt with option E, so
the model's own letter is the answer:

1. Prompt only (no steering).
2. SE wrapper: reply "I don't know" when forced-choice SE > tau, otherwise
   use the model's letter. It shows what SE alone buys.
3. **SE-gated circuit steering**: steer only when SE > tau.
4. **SE-scaled circuit steering**: dose `alpha * SE / log 4` on every item.
5. Always-on circuit steering.
6. Control: SE-gated steering with mean-difference vectors at k random heads.
7. Control: SE-gated steering with norm-matched random vectors at the circuit
   heads.

Dose alpha comes from {1, 2, 4, 8}, and tau from a fixed grid of SE values.
Both are chosen on `cal_prob` to maximize net correct answers (right answered
minus wrong answered). All conditions are then scored on `test`: answered,
wrong among answered, right answers kept, invented refused. Among gated rows,
the report also compares how often steering flips right versus wrong answers
to E. The circuit adds something beyond SE only if it flips wrong answers more
often than right ones at the same SE.

## Architecture

Additions follow the existing layers.

- `domain/grading.py`: `semantic_entropy`, `abstention_gap`.
- `domain/patching.py`: `restoration`, `smallest_sufficient_k`.
- `application/ports.py`: `SiteKind`, `Site`, `Capture`, `Patch`, `Addition`,
  `Trace`, and a `CircuitModel` protocol with `trace(...)` and
  `common_suffix_length(...)`.
- `application/circuit_config.py`: TOML config for the study.
- `application/entity_pairs.py`: `EntityPair`, discovery and test split.
- `application/patching.py`: stages 1 and 2.
- `application/se_steering.py`: stage 3.
- `application/circuit_report.py`, `application/circuit_pipeline.py`.
- `infrastructure/health_templates.py`: templates and option pools shared by
  invented items and pairs.
- `infrastructure/known_facts.py`: the curated real facts.
- `infrastructure/entity_pairs.py`: builds pairs from facts and invented names.
- `infrastructure/hf_model.py`: `trace` with hooks on decoder layers,
  `self_attn.o_proj` inputs and `mlp` outputs. `read` is rebuilt on `trace`.
- `infrastructure/figures.py`: two heatmaps and a steering trade-off plot.
- `cli.py`: `circuits --config ...`.

## Testing

End to end only. A fake circuit model with a planted head (layer 2, head 1)
copies an "unfamiliar" signal to the final token and raises the E logit. The
test runs `run`, then `circuits`, and checks that the study ranks the planted
head first, that the circuit beats random heads, that gated steering raises
abstention only on gated rows, and that every artifact is written. The GPU run
on Llama is the real test.

## Honest scope

Patching at the final token finds where the signal is read, not the whole
path from the entity. Head effects can be distributed, so the circuit may be
large. Invented names differ in morphology as well as familiarity. Steering
results hold for this prompt format only. Negative results are reported as
results.
