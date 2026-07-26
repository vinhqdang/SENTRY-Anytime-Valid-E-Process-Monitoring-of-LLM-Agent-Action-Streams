"""Distributional statistics the manuscript quotes, with a generator.

Three quantities motivate Proposition 1 and the tie discussion after
Proposition 2, and none of them previously had a script behind it:

  * the benign instruction-likeness distribution (median, mean, maximum), which is
    what makes excess-over-maximum clip an entire attack family;
  * the maximum InjecAgent instruction-likeness, which must lie below that benign
    maximum for Proposition 1's hypothesis to hold;
  * the fraction of benign trajectories on which the action audit S4 saturates at
    1.0, which is why the conformal p-value's tie handling matters.

    SENTRY_IL_MODE=llm SENTRY_JUSTIFY=1 python -m real_data.distributions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from real_data.evaluate_fused import MIN_ACTIONS

ROOT = Path(__file__).parent


def main() -> None:
    import os

    # Instruction-likeness is attached to each Action when the log is parsed, so
    # the corpus must be loaded once per scorer mode.
    os.environ["SENTRY_IL_MODE"] = "lexicon"
    import importlib
    from real_data import adapters, evaluate_fused
    importlib.reload(adapters); importlib.reload(evaluate_fused)
    benign = evaluate_fused._corpus("logs_gpt4omini")[0] + \
        evaluate_fused._corpus("logs_deepseek")[0]
    ben_il = np.asarray([a.obs_instruction_likeness for t, _ in benign for _, a in t],
                        dtype=float)
    inj_rows = [(t, m) for t, m in
                adapters.load_agentdojo_logs(ROOT / "agentdojo" / "logs_deepseek_injecagent")
                if m["is_attack"] and len(t) >= MIN_ACTIONS]
    inj_il = np.asarray([a.obs_instruction_likeness for t, _ in inj_rows for _, a in t],
                        dtype=float)

    # The audit score is independent of the instruction scorer.
    sat = [1.0 if max((a.unjustified for _, a in t), default=0.0) >= 1.0 else 0.0
           for t, _ in benign]

    out = {
        "scorer": "lexicon",
        "benign_instruction_likeness": {
            "n_actions": int(ben_il.size),
            "median": round(float(np.median(ben_il)), 4),
            "mean": round(float(ben_il.mean()), 4),
            "max": round(float(ben_il.max()), 4)},
        "injecagent_instruction_likeness": {
            "n_actions": int(inj_il.size),
            "median": round(float(np.median(inj_il)), 4),
            "max": round(float(inj_il.max()), 4)},
        "s4_saturates_at_one": {
            "n_benign": len(sat), "n": int(sum(sat)),
            "pct": round(100.0 * sum(sat) / len(sat), 1)},
    }
    (ROOT / "results_distributions.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'results_distributions.json'}")


if __name__ == "__main__":
    main()
