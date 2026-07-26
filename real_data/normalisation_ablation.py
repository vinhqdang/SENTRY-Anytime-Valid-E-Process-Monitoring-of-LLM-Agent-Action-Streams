"""Normalisation ablation: which scorer, and how it is normalised.

Two axes, crossed:

  scorer        `lexicon`  -- keyword/pattern instruction-likeness
                `llm`      -- the cached model judge
  normalisation `excess_max` -- max{0, x - max(benign training)}
                `conformal`  -- the tail surprisal of Definition 3

evaluated on two attack families (important_instructions, InjecAgent). The
point of the cross is the interaction: `excess_max` subtracts an outlying benign
maximum, which clips an entire attack family to zero and drives AUROC to chance
or below, and it does so *worst* for the scorer whose output is most
concentrated (the judge). This is the ablation behind Proposition 1.

Writes results_signal_ablation.json, keyed `scorer|normalisation|family`.

    SENTRY_JUSTIFY=1 python -m real_data.normalisation_ablation
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from real_data.adapters import load_agentdojo_logs, load_taubench_logs
from real_data.evaluate import aggregate, evaluate_once
from sentry.scores import SequentialWorldModel

ROOT = Path(__file__).parent
MIN_ACTIONS = 2
N_SEEDS = 10
SCORERS = ("lexicon", "llm")
TRANSFORMS = ("excess_max", "conformal")
METRICS = ("pac_observable_rate", "pac_attempt_rate", "pac_nominal_far",
           "auroc_attempt_max")


def _agentdojo(logdir: str):
    rows = [(t, m) for t, m in load_agentdojo_logs(ROOT / "agentdojo" / logdir)
            if len(t) >= MIN_ACTIONS]
    nominal = [(t, m) for t, m in rows
               if not m["is_attack"] and m.get("utility") is not None]
    attacks = [(t, m) for t, m in rows
               if m["is_attack"] and m.get("security") is not None]
    # security=True means the injection's goal was achieved (agent compromised).
    succ = [x for x in attacks if x[1]["security"] is True]
    resisted = [x for x in attacks if x[1]["security"] is False]
    return nominal, succ, resisted


def main() -> None:
    import os

    out = {}
    for scorer in SCORERS:
        # The instruction-likeness score is attached to each Action when the log
        # is parsed, so the corpus must be RELOADED after switching scorers --
        # setting the mode alone would silently reuse the previous scores.
        os.environ["SENTRY_IL_MODE"] = scorer
        nominal, imp_succ, imp_res = _agentdojo("logs_deepseek")
        nominal = nominal + [(t, m) for t, m in
                             load_taubench_logs(ROOT / "tau_bench" / "logs_deepseek")
                             if len(t) >= MIN_ACTIONS]
        _, inj_succ, inj_res = _agentdojo("logs_deepseek_injecagent")
        families = {"important_instr": (imp_succ, imp_res),
                    "injecagent": (inj_succ, inj_res)}
        for transform in TRANSFORMS:
            factory = partial(SequentialWorldModel,
                              instruction_transform=transform)
            for fam, (succ, res) in families.items():
                rows = [evaluate_once(factory, nominal, succ, res, s)
                        for s in range(N_SEEDS)]
                rec = {}
                for k in METRICS:
                    mean, _ = aggregate(rows, k)
                    rec[k] = mean
                out[f"{scorer}|{transform}|{fam}"] = rec
                print(f"{scorer:<8}{transform:<12}{fam:<18}"
                      f"obs={rec['pac_observable_rate']:.3f} "
                      f"att={rec['pac_attempt_rate']:.3f} "
                      f"FAR={rec['pac_nominal_far']:.4f} "
                      f"AUROC={rec['auroc_attempt_max']:.3f}")

    (ROOT / "results_signal_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'results_signal_ablation.json'}")


if __name__ == "__main__":
    main()
