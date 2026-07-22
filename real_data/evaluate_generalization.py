"""Generalisation study for SENTRY: does the monitor hold up under (i) a
DIFFERENT attack family and (ii) a DIFFERENT agent model than the ones it was
first validated on?

Three scenarios, each evaluated with the shipped SequentialWorldModel over
N_SEEDS random splits (same protocol as real_data/evaluate.py):

  A. baseline   -- DeepSeek-V4-Flash nominal + important_instructions attacks
                   (the headline configuration; reproduced here for contrast)
  B. new attack -- DeepSeek-V4-Flash nominal + InjecAgent attacks
                   (reference unchanged; tests cross-ATTACK generalisation)
  C. new agent  -- GPT-4o-mini nominal + GPT-4o-mini important_instructions
                   attacks (reference re-fit on a different agent; tests
                   cross-AGENT generalisation)

Writes results_generalization.json and prints a comparison table.

    source .venv/bin/activate
    python -m real_data.evaluate_generalization
"""

from __future__ import annotations

import json
from pathlib import Path

from functools import partial

from real_data.adapters import load_agentdojo_logs, load_taubench_logs
from real_data.evaluate import aggregate, evaluate_once
from sentry.scores import SequentialWorldModel

ROOT = Path(__file__).parent
MIN_ACTIONS = 2
N_SEEDS = 10
METRICS = [
    "pac_observable_rate",
    "pac_attempt_rate",
    "pac_detection_rate",
    "pac_nominal_far",
    "auroc_attempt_max",
]


def _agentdojo(logdir: str):
    rows = [(t, m) for t, m in load_agentdojo_logs(ROOT / "agentdojo" / logdir) if len(t) >= MIN_ACTIONS]
    nominal = [(t, m) for t, m in rows if not m["is_attack"] and m.get("utility") is not None]
    attacks = [(t, m) for t, m in rows if m["is_attack"] and m.get("security") is not None]
    succ = [x for x in attacks if x[1]["security"] is False]
    resisted = [x for x in attacks if x[1]["security"] is True]
    return nominal, succ, resisted


def _taubench(logdir: str):
    return [(t, m) for t, m in load_taubench_logs(ROOT / "tau_bench" / logdir) if len(t) >= MIN_ACTIONS]


def scenario(nominal, succ, resisted):
    if not succ and not resisted:
        return None
    rows = [evaluate_once(SequentialWorldModel, nominal, succ, resisted, s) for s in range(N_SEEDS)]
    out = {"n_nominal": len(nominal), "n_successful": len(succ), "n_resisted": len(resisted),
           "n_observable": rows[0].get("n_observable"), "n_attempts": rows[0].get("n_attempts")}
    for k in METRICS:
        mean, std = aggregate(rows, k)
        out[k] = {"mean": mean, "std": std}
    return out


def main() -> None:
    # Shared DeepSeek nominal (AgentDojo benign + tau-bench retail).
    ds_nom, ds_succ, ds_res = _agentdojo("logs_deepseek")
    ds_nom = ds_nom + _taubench("logs_deepseek")

    # B: same nominal, InjecAgent attacks.
    _, inj_succ, inj_res = _agentdojo("logs_deepseek_injecagent")

    # C: GPT-4o-mini nominal + its important_instructions attacks.
    gpt_nom, gpt_succ, gpt_res = _agentdojo("logs_gpt4omini")

    results = {
        "A_baseline_deepseek_important": scenario(ds_nom, ds_succ, ds_res),
        "B_newattack_deepseek_injecagent": scenario(ds_nom, inj_succ, inj_res),
        "C_newagent_gpt4omini_important": scenario(gpt_nom, gpt_succ, gpt_res),
    }
    (ROOT / "results_generalization.json").write_text(json.dumps(results, indent=2))

    hdr = f"{'scenario':<38}{'nom':>5}{'succ':>5}{'res':>5}{'obs%':>7}{'att%':>7}{'FAR':>7}{'AUROC':>7}"
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        if r is None:
            print(f"{name:<38}  (no data yet)")
            continue
        print(f"{name:<38}{r['n_nominal']:>5}{r['n_successful']:>5}{r['n_resisted']:>5}"
              f"{r['pac_observable_rate']['mean']*100:>6.0f} "
              f"{r['pac_attempt_rate']['mean']*100:>6.0f} "
              f"{r['pac_nominal_far']['mean']:>6.3f} "
              f"{r['auroc_attempt_max']['mean']:>6.2f}")


if __name__ == "__main__":
    main()
