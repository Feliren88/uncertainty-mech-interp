# Abstention circuit and SE steering implementation plan

**Goal:** find the components that make Llama 3.1 8B Instruct pick "I don't know", and use semantic entropy to switch them on.
**Spec:** `docs/specs/2026-09-19-circuit-se-steering-design.md`
**Constraints:** same as the first plan (end-to-end tests only, torch only in `infrastructure/hf_model.py`, no AI-assistant attribution in commits, house-style figures, no em dashes in prose).

Code is written directly into the source files; each task names its files and the interfaces later tasks rely on.

1. **Shared templates, facts, pairs.** `infrastructure/health_templates.py` (templates, pools, name inventor), `infrastructure/fictional.py` refactored onto it, `infrastructure/known_facts.py` (75 facts), `infrastructure/entity_pairs.py` (`build_entity_pairs(names_per_fact, seed, exclude_words) -> list[EntityPair]`), `application/entity_pairs.py` (`EntityPair(pair_id, entity_group, real, invented)`, `split_pairs(pairs, discovery_share, seed)`).
2. **Trace port and adapter.** `application/ports.py`: `SiteKind{RESID, ATTN_Z, MLP}`, `Site(kind, layer, head=None)`, `Capture(site, offsets)`, `Patch(site, offsets, values)`, `Addition(site, offsets, vector, scale)`, `Trace(letter_logprobs, letter_mass, activations)`, `CircuitModel` with `num_heads`, `head_dim`, `trace(prompts, letters, *, captures, patches, additions)`, `common_suffix_length(prompts)`. `infrastructure/hf_model.py`: `trace`, `read` rebuilt on `trace`. Check: e2e test passes; `read` via `trace` reproduces the second run's saved log-probabilities.
3. **Domain math.** `domain/grading.py`: `semantic_entropy(letter_logprobs, n_answers=4)`, `abstention_gap(letter_logprobs)`. `domain/patching.py`: `restoration(patched, target, source)`, `smallest_sufficient_k(effects_by_k, threshold)`.
4. **Use cases.** `application/circuit_config.py` (`CircuitStudyConfig`, `load_circuit_config`), `application/patching.py` (segments, residual/head/MLP sweeps, circuit choice, validation, steering vectors), `application/se_steering.py` (calibrate dose and tau on cal_prob, score conditions on test), `application/circuit_report.py`, `application/circuit_pipeline.py`, figures, `cli.py circuits`.
5. **Fake circuit model and e2e test.** `tests/fakes.py::FakeCircuitModel` with a planted head (2, 1); `tests/test_circuit_study.py` runs `run` then `circuits`.
6. **GPU study.** `configs/circuit_study.toml`, run on the second run's artifacts, check figures, update README.
