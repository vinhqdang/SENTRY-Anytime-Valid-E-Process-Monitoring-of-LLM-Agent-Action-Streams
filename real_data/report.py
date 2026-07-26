"""Check the reproduced numbers against the values reported in the manuscript.

Every expected value below is the number printed in the paper, so a mismatch
means either the code changed or the paper is stale. Run after reproduce.sh.

    python -m real_data.report
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent

# (label, path-into-json, expected-as-printed-in-the-paper, tolerance)
FUSED_CHECKS = [
    ("GPT-4o-mini  SENTRY-Fuse AUROC",
     ("corpora", "GPT-4o-mini", "detectors", "SENTRY-Fuse (S3+S4+S5)", "auroc_mean"),
     0.985, 0.002),
    ("GPT-4o-mini  SENTRY-Fuse TPR@5%",
     ("corpora", "GPT-4o-mini", "detectors", "SENTRY-Fuse (S3+S4+S5)", "tpr_at_0.05_mean"),
     0.956, 0.003),
    ("GPT-4o-mini  realised FPR",
     ("corpora", "GPT-4o-mini", "detectors", "SENTRY-Fuse (S3+S4+S5)", "realised_fpr_at_0.05"),
     0.029, 0.003),
    ("GPT-4o-mini  S3 alone AUROC",
     ("corpora", "GPT-4o-mini", "detectors", "S3 instruction only", "auroc_mean"),
     0.951, 0.003),
    ("DeepSeek     SENTRY-Fuse AUROC",
     ("corpora", "DeepSeek", "detectors", "SENTRY-Fuse (S3+S4+S5)", "auroc_mean"),
     0.843, 0.003),
    ("DeepSeek     SENTRY-Fuse TPR@5%",
     ("corpora", "DeepSeek", "detectors", "SENTRY-Fuse (S3+S4+S5)", "tpr_at_0.05_mean"),
     0.778, 0.003),
    ("Pooled       SENTRY-Fuse AUROC",
     ("pooled", "detectors", "SENTRY-Fuse (S3+S4+S5)", "auroc_mean"),
     0.957, 0.003),
    ("Pooled       SENTRY-Fuse TPR@5%",
     ("pooled", "detectors", "SENTRY-Fuse (S3+S4+S5)", "tpr_at_0.05_mean"),
     0.919, 0.003),
    ("Pooled       S1+S2 behavioural AUROC (near chance)",
     ("pooled", "detectors", "S1 + S2 behavioural", "auroc_mean"),
     0.561, 0.005),
]

CEILING_CHECKS = [
    ("Compromised trajectories (pooled)", ("pooled", "n_compromised"), 90, 0),
    ("Payload visible, permissive criterion",
     ("pooled", "payload_visible_in_trace", "frac"), 1.000, 0.001),
    ("Payload visible, strict criterion",
     ("pooled", "payload_visible_strict", "frac"), 0.867, 0.002),
    ("Calls an effectful sink", ("pooled", "has_effectful_sink_call", "frac"),
     0.856, 0.002),
    ("nu, permissive criterion", ("pooled", "primary", "nu"), 0.000, 0.001),
    ("nu, strict criterion", ("pooled", "strict", "nu"), 0.044, 0.002),
    ("Distinctive directive visible for >= 5 words",
     ("pooled", "distinctive_core_visible", "5"), 1.000, 0.001),
    ("Distinctive directive visible in full (k=30)",
     ("pooled", "distinctive_core_visible", "30"), 0.144, 0.002),
]

CORPUS_CHECKS = [
    ("Benign trajectories", ("totals", "benign"), 191, 0),
    ("Attacked trajectories", ("totals", "attacks"), 393, 0),
    ("Compromised", ("totals", "compromised"), 90, 0),
    ("Resisted", ("totals", "resisted"), 303, 0),
    ("Polarity: security=True and paid attacker",
     ("polarity_check", "security_true_pays"), 31, 0),
    ("Polarity: security=True and did not pay",
     ("polarity_check", "security_true_no_pay"), 0, 0),
    ("Polarity: security=False but paid (undercount)",
     ("polarity_check", "security_false_pays"), 3, 0),
]

QUANT_CHECKS = [
    ("Conformal AUROC at n_cal=35 (raw 1.000)",
     ("by_calibration_size", "35", "psi_auroc"), 0.986, 0.002),
    ("Conformal AUROC at n_cal=5 (raw 1.000)",
     ("by_calibration_size", "5", "psi_auroc"), 0.925, 0.002),
    ("Counterexample: saturation to one half",
     ("counterexamples", "saturation_to_half", "psi"), 0.500, 0.001),
    ("Counterexample: above chance to below chance",
     ("counterexamples", "below_half", "psi"), 0.333, 0.001),
]

ABLATION_CHECKS = [
    ("judge + excess-over-max, important_instr  (destroyed)",
     "llm|excess_max|important_instr", 0.515, 0.01),
    ("judge + conformal,       important_instr  (works)",
     "llm|conformal|important_instr", 0.761, 0.01),
    ("judge + excess-over-max, InjecAgent       (destroyed)",
     "llm|excess_max|injecagent", 0.531, 0.01),
    ("judge + conformal,       InjecAgent       (recovered)",
     "llm|conformal|injecagent", 0.706, 0.01),
]


def _dig(d, path):
    for k in path:
        d = d[k]
    return d


def _check(rows, data, getter):
    bad = 0
    print(f"\n{'quantity':<52}{'reproduced':>12}{'paper':>9}{'':>4}")
    print("-" * 77)
    for label, key, expected, tol in rows:
        try:
            got = getter(data, key)
        except (KeyError, TypeError):
            print(f"{label:<52}{'MISSING':>12}{expected:>9}   x")
            bad += 1
            continue
        ok = abs(got - expected) <= tol
        bad += not ok
        fmt = f"{got:>12.3f}" if isinstance(got, float) else f"{got:>12d}"
        exp = f"{expected:>9.3f}" if isinstance(expected, float) else f"{expected:>9d}"
        print(f"{label:<52}{fmt}{exp}{'   ok' if ok else '   MISMATCH'}")
    return bad


def main() -> None:
    print("=" * 77)
    print("SENTRY -- reproduced vs. manuscript")
    print("=" * 77)
    bad, missing = 0, []

    p = ROOT / "results_fused.json"
    if p.exists():
        print("\n## Compromise detection (Table: fusion ablation)")
        bad += _check(FUSED_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_fused.json -- run: python -m real_data.evaluate_fused")

    p = ROOT / "results_ceiling_per_corpus.json"
    if p.exists():
        print("\n## Evidence channels (Table: detectability)")
        bad += _check(CEILING_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_ceiling_per_corpus.json -- run: python -m real_data.ceiling")

    p = ROOT / "results_corpus.json"
    if p.exists():
        print("\n## Corpus composition and label polarity (Tables: corpus, polarity)")
        bad += _check(CORPUS_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_corpus.json -- run: python -m real_data.corpus_table")

    p = ROOT / "results_quantisation.json"
    if p.exists():
        print("\n## Conformal quantisation loss (Proposition 3(iii))")
        bad += _check(QUANT_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_quantisation.json -- run: "
                       "python -m real_data.quantisation_demo")

    p = ROOT / "results_signal_ablation.json"
    if p.exists():
        print("\n## Normalisation ablation, attempt-detection AUROC")
        data = json.loads(p.read_text())
        bad += _check(ABLATION_CHECKS, data,
                      lambda d, k: d[k]["auroc_attempt_max"])
    else:
        missing.append("results_signal_ablation.json -- run: "
                       "python -m real_data.normalisation_ablation")

    for m in missing:
        print(f"\n  skipped: {m}")
    print()
    if bad:
        print(f"{bad} value(s) disagree with the manuscript.")
        raise SystemExit(1)
    print("All checked values match the manuscript.")


if __name__ == "__main__":
    main()
