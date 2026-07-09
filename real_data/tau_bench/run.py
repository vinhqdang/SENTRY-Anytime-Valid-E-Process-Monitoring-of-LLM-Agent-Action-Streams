"""Run a slice of tau-bench (retail domain) through OpenRouter's free-model
router -- agent and simulated user both use it -- to collect real nominal
LLM-agent trajectories for SENTRY's evaluation (algorithm.md §7 items 1-2).

tau-bench has no built-in attack suite (unlike AgentDojo); it supplies the
"realistic benign multi-turn tool-use" half of the real-data validation,
complementing AgentDojo's benign+attack pairs.

Requires OPENROUTER_API_KEY in the repo-root .env. Run from the repo root
with the project venv active:

    source .venv/bin/activate
    python real_data/tau_bench/run.py [--pilot]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tau_bench.run import run  # noqa: E402
from tau_bench.types import RunConfig  # noqa: E402

LOGDIR = Path(__file__).parent / "logs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="tiny smoke-test run")
    args = parser.parse_args()

    end_index = 2 if args.pilot else 15

    config = RunConfig(
        model_provider="openrouter",
        user_model_provider="openrouter",
        model="free",
        user_model="free",
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        start_index=0,
        end_index=end_index,
        log_dir=str(LOGDIR),
        max_concurrency=1,
        user_strategy="llm",
    )
    run(config)


if __name__ == "__main__":
    main()
