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

import os  # noqa: E402

MODEL = os.environ.get("SENTRY_MODEL", "deepseek/deepseek-v4-flash")
LOGDIR = Path(__file__).parent / os.environ.get("SENTRY_LOGDIR", "logs_deepseek")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="tiny smoke-test run")
    args = parser.parse_args()

    end_index = 2 if args.pilot else 50

    config = RunConfig(
        model_provider="openrouter",
        user_model_provider="openrouter",
        model=MODEL,
        user_model=MODEL,
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
    # tau-bench builds its checkpoint path by interpolating user_model raw;
    # a provider-qualified id like "deepseek/deepseek-v4-flash" contains a
    # slash, so the path gains a subdirectory (".../user-deepseek/...") that
    # tau-bench never creates, crashing the checkpoint write. Pre-create every
    # such implied directory (the slashy segment is deterministic -- the
    # timestamp lives only in the filename).
    if "/" in MODEL:
        LOGDIR.mkdir(parents=True, exist_ok=True)
        # the implied parent dir ends with the user_model org segment
        implied = LOGDIR / (
            f"{config.agent_strategy}-{MODEL.split('/')[-1]}-{config.temperature}"
            f"_range_{config.start_index}-{config.end_index}"
            f"_user-{MODEL.split('/')[0]}"
        )
        implied.mkdir(parents=True, exist_ok=True)
    run(config)


if __name__ == "__main__":
    main()
