#!/usr/bin/env bash
# Health test run on one GPU: end-to-end test, GPU smoke run, full test run, then demo questions.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/home/vfeliren1/lf93_scratch2/vfvic1/hf_cache/huggingface}"
export PYTHONPATH=src

python -m pytest -q
python -m uncertainty_mech run --config configs/health_smoke.toml
python -m uncertainty_mech run --config configs/health_test_run.toml
python -m uncertainty_mech ask --run-dir runs/health-llama31-8b-test-run --questions examples/health_questions.jsonl
