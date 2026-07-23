"""Print the headline numbers reported in the manuscript from the committed
results JSON, next to the values printed in the paper, so a reproducer can
check at a glance that a fresh run matches.

    python -m real_data.report
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent


def _load(name):
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    main_r = _load("results.json")
    gen_r = _load("results_generalization.json")

    print("=" * 68)
    print("SENTRY -- manuscript headline numbers (reproduced vs. paper)")
    print("=" * 68)

    if main_r:
        m = main_r["models"]["bigram_novelty_summed"]["aggregate"]
        mix = main_r["models"]["bigram_novelty_evalue_mixture"]["aggregate"]
        print(f"\nCorpus: {main_r['n_nominal']} nominal, "
              f"{main_r['n_successful_attacks']} successful + "
              f"{main_r['n_resisted_attacks']} resisted attacks  (paper: 121 / 180 / 18)")
        rows = [
            ("Observable-injection detection (PAC)", "pac_observable_rate", m, "0.99"),
            ("All-attempts detection (PAC)",         "pac_attempt_rate",    m, "0.66"),
            ("Hijack-success detection (PAC)",       "pac_detection_rate",  m, "0.64"),
            ("Per-step false-alarm rate",            "pac_nominal_far",     m, "0.02"),
            ("Attempt-vs-benign AUROC",              "auroc_attempt_max",   m, "0.77"),
            ("Ville attempt detection",              "ville_attempt_rate",  m, "0.71"),
            ("e-value-mixture attempt (PAC)",        "pac_attempt_rate",    mix, "0.65"),
            ("e-value-mixture FAR (Ville)",          "ville_nominal_far",   mix, "0.008"),
        ]
        print(f"\n{'metric':<40}{'reproduced':>12}{'paper':>8}")
        print("-" * 60)
        for label, key, src, paper in rows:
            v = src.get(key, {}).get("mean", float("nan"))
            print(f"{label:<40}{v:>12.3f}{paper:>8}")
    else:
        print("\n(real_data/results.json missing -- run `python -m real_data.evaluate`)")

    if gen_r:
        print("\nGeneralisation (attempt-AUROC / observable detection):")
        spec = [
            ("A_baseline_deepseek_important", "baseline (DeepSeek, imp.instr.)", "0.77 / 0.99"),
            ("C_newagent_gpt4omini_important", "new agent (GPT-4o-mini)", "0.82 / 1.00"),
            ("B_newattack_deepseek_injecagent", "new attack (InjecAgent)", "0.47 / 0.10"),
        ]
        print(f"{'scenario':<34}{'AUROC':>8}{'obs':>7}{'':>4}{'paper':>12}")
        print("-" * 66)
        for key, label, paper in spec:
            r = gen_r.get(key)
            if not r:
                continue
            a = r["auroc_attempt_max"]["mean"]
            o = r["pac_observable_rate"]["mean"]
            print(f"{label:<34}{a:>8.2f}{o:>7.2f}{'':>4}{paper:>12}")
    else:
        print("\n(results_generalization.json missing -- run "
              "`python -m real_data.evaluate_generalization`)")
    print()


if __name__ == "__main__":
    main()
