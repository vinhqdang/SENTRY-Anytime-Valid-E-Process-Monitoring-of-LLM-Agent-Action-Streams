"""Run a slice of AgentDojo suites (benign trajectories + the
"important_instructions" prompt-injection attack) through OpenRouter's
free-model router, saving per-task trace logs for SENTRY's real-data
evaluation (algorithm.md §7 items 1-2).

Requires OPENROUTER_API_KEY in the repo-root .env. Run from the repo root
with the project venv active:

    source .venv/bin/activate
    python real_data/agentdojo/run.py [--pilot]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig  # noqa: E402
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks  # noqa: E402
from agentdojo.logging import OutputLogger  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

# Must contain a substring from agentdojo.attacks.base_attacks.MODEL_NAMES so
# the important_instructions attack can phrase its injection ("local" ->
# "Local model" -- an honest label for a router over open-weight free models).
MODEL_NAME = "openrouter-free-router-local"
LOGDIR = Path(__file__).parent / "logs"

FULL_CONFIG = {
    "banking": {"n_benign": 10, "n_attack_user_tasks": 4, "n_injection_tasks": 2},
    "workspace": {"n_benign": 8, "n_attack_user_tasks": 3, "n_injection_tasks": 2},
}
PILOT_CONFIG = {
    "banking": {"n_benign": 2, "n_attack_user_tasks": 1, "n_injection_tasks": 1},
}


def build_pipeline() -> AgentPipeline:
    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    llm = OpenAILLM(client, "openrouter/free")
    llm.name = MODEL_NAME
    config = PipelineConfig(
        llm=llm,
        model_id=None,
        defense=None,
        tool_delimiter="tool",
        system_message_name=None,
        system_message=None,
    )
    return AgentPipeline.from_config(config)


def run_suite(suite_name: str, cfg: dict, pipeline: AgentPipeline) -> None:
    suite = get_suite("v1.2.2", suite_name)

    benign_task_ids = list(suite.user_tasks.keys())[: cfg["n_benign"]]
    print(f"[{suite_name}] benign: {len(benign_task_ids)} tasks")
    for tid in benign_task_ids:
        task = suite.get_user_task_by_id(tid)
        t0 = time.time()
        try:
            utility, security = run_task_without_injection_tasks(
                suite, pipeline, task, logdir=LOGDIR, force_rerun=False
            )
            print(f"  benign {tid}: utility={utility} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001 - real free-tier models fail unpredictably
            print(f"  benign {tid}: FAILED ({type(e).__name__}: {e})")

    attack = load_attack("important_instructions", suite, pipeline)
    attack_user_task_ids = list(suite.user_tasks.keys())[: cfg["n_attack_user_tasks"]]
    injection_task_ids = list(suite.injection_tasks.keys())[: cfg["n_injection_tasks"]]
    print(f"[{suite_name}] attack: {len(attack_user_task_ids)} user tasks x {len(injection_task_ids)} injections")
    for tid in attack_user_task_ids:
        task = suite.get_user_task_by_id(tid)
        t0 = time.time()
        try:
            run_task_with_injection_tasks(
                suite,
                pipeline,
                task,
                attack,
                logdir=LOGDIR,
                force_rerun=False,
                injection_tasks=injection_task_ids,
            )
            print(f"  attack {tid}: done ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"  attack {tid}: FAILED ({type(e).__name__}: {e})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="tiny smoke-test run")
    args = parser.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline()
    config = PILOT_CONFIG if args.pilot else FULL_CONFIG
    with OutputLogger(str(LOGDIR)):
        for suite_name, cfg in config.items():
            run_suite(suite_name, cfg, pipeline)


if __name__ == "__main__":
    main()
