#!/usr/bin/env bash
# Reproduce every number, table and figure in the SENTRY manuscript.
#
# The real agent trajectories are committed under real_data/agentdojo/ and
# real_data/tau_bench/, the model-judge scores are committed as caches, and the
# evaluation is deterministic (seed-stable hashing, fixed seeds 0-9). So this
# script reproduces the paper's numbers WITHOUT any API key or network access.
#
# Usage:
#   ./reproduce.sh            # tests + every table + every figure + the check
#   ./reproduce.sh --collect  # ALSO re-collect trajectories from OpenRouter first
#                             #   (needs OPENROUTER_API_KEY in .env; costs money; LLM
#                             #    sampling is not bit-reproducible, so numbers may
#                             #    shift slightly from the committed corpus)
set -euo pipefail
cd "$(dirname "$0")"

# The instruction-likeness scorer and the action-justification audit both read
# committed caches; these two variables select the configuration the paper
# reports. Without them the lexicon scorer is used and the audit is disabled.
export SENTRY_IL_MODE=llm
export SENTRY_JUSTIFY=1
export PYTHONPATH=.

if [[ "${1:-}" == "--collect" ]]; then
  echo "### [0/8] Re-collecting real trajectories from OpenRouter ..."
  python real_data/agentdojo/run.py --max
  SENTRY_ATTACK=injecagent SENTRY_LOGDIR=logs_deepseek_injecagent \
    python real_data/agentdojo/run.py --attack-expansion
  SENTRY_MODEL=openai/gpt-4o-mini SENTRY_MODEL_NAME=gpt-4o-mini-local \
    SENTRY_LOGDIR=logs_gpt4omini python real_data/agentdojo/run.py --max
  python real_data/tau_bench/run.py
fi

echo "### [1/11] Unit + end-to-end synthetic tests ..."
pytest -q

echo "### [2/11] Synthetic ARL/FAR + detection-delay validation ..."
python examples/run_synthetic_validation.py

echo "### [3/11] Compromise detection and the signal ablation"
echo "          (writes real_data/results_fused.json) ..."
python -m real_data.evaluate_fused

echo "### [4/11] Evidence channels / detectability"
echo "          (writes real_data/results_ceiling_per_corpus.json) ..."
python -m real_data.ceiling

echo "### [5/11] Normalisation ablation"
echo "          (writes real_data/results_signal_ablation.json) ..."
python -m real_data.normalisation_ablation

echo "### [6/11] Corpus composition, label polarity, and quantisation loss"
echo "          (writes real_data/results_{corpus,quantisation}.json) ..."
python -m real_data.corpus_table
python -m real_data.quantisation_demo
python -m real_data.distributions

echo "### [7/11] Task Shield baseline, re-implemented on our corpus"
echo "          (writes real_data/results_taskshield.json; uses the committed"
echo "           judgment cache, so no API access is needed) ..."
python -m real_data.baseline_taskshield

echo "### [8/11] Attempt detection and generalisation"
echo "          (writes real_data/results{,_generalization}.json) ..."
python -m real_data.evaluate
python -m real_data.evaluate_generalization

echo "### [9/11] Long-horizon e-detector comparison (negative result)"
echo "          (writes real_data/results_longhorizon.json) ..."
python -m real_data.longhorizon

echo "### [10/11] Regenerating figures ..."
python manuscript/make_numbers.py   # regenerates the manuscript's number macros
python manuscript/make_figures.py
python manuscript/make_paper_figures.py

echo "### [11/11] Checking every reproduced value against the manuscript ..."
python -m real_data.report
echo
echo "Figures are in manuscript/figures/."
