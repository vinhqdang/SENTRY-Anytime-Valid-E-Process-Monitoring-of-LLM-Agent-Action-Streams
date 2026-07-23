#!/usr/bin/env bash
# Reproduce every number and figure in the SENTRY manuscript.
#
# The real agent trajectories are committed under real_data/agentdojo/ and
# real_data/tau_bench/, and the evaluation is deterministic (seed-stable CRC
# hashing, fixed seeds 0-9), so this script reproduces the paper's numbers
# WITHOUT any API key or network access.
#
# Usage:
#   ./reproduce.sh            # tests + real-data eval + generalisation + figures + report
#   ./reproduce.sh --collect  # ALSO re-collect trajectories from OpenRouter first
#                             #   (needs OPENROUTER_API_KEY in .env; costs money; the
#                             #    LLM sampling is not bit-reproducible, so numbers may
#                             #    shift slightly from the committed corpus)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--collect" ]]; then
  echo "### [0/5] Re-collecting real trajectories from OpenRouter ..."
  python real_data/agentdojo/run.py --max
  SENTRY_ATTACK=injecagent SENTRY_LOGDIR=logs_deepseek_injecagent \
    python real_data/agentdojo/run.py --attack-expansion
  SENTRY_MODEL=openai/gpt-4o-mini SENTRY_MODEL_NAME=gpt-4o-mini-local \
    SENTRY_LOGDIR=logs_gpt4omini python real_data/agentdojo/run.py --max
  python real_data/tau_bench/run.py
fi

echo "### [1/5] Unit + end-to-end synthetic tests ..."
pytest -q

echo "### [2/5] Synthetic ARL/FAR + detection-delay validation ..."
python examples/run_synthetic_validation.py

echo "### [3/5] Real-data evaluation (writes real_data/results.json) ..."
python -m real_data.evaluate

echo "### [4/5] Generalisation study (writes real_data/results_generalization.json) ..."
python -m real_data.evaluate_generalization

echo "### [5/5] Regenerating manuscript figures ..."
PYTHONPATH=. python manuscript/make_figures.py

echo
python -m real_data.report
echo "Done. Numbers above should match the manuscript; figures are in manuscript/figures/."
