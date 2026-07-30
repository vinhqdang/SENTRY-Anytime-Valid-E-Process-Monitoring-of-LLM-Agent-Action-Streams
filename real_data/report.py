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
     ("corpora", "GPT-4o-mini", "detectors", "SENTRY-Fuse (S3+S4)", "auroc_mean"),
     0.981, 0.002),
    ("GPT-4o-mini  SENTRY-Fuse TPR@5%",
     ("corpora", "GPT-4o-mini", "detectors", "SENTRY-Fuse (S3+S4)", "tpr_at_0.05_mean"),
     0.914, 0.003),
    ("GPT-4o-mini  realised FPR",
     ("corpora", "GPT-4o-mini", "detectors", "SENTRY-Fuse (S3+S4)", "realised_fpr_at_0.05"),
     0.025, 0.003),
    ("GPT-4o-mini  S3 alone AUROC",
     ("corpora", "GPT-4o-mini", "detectors", "S3 instruction only", "auroc_mean"),
     0.951, 0.003),
    ("DeepSeek     SENTRY-Fuse AUROC",
     ("corpora", "DeepSeek", "detectors", "SENTRY-Fuse (S3+S4)", "auroc_mean"),
     0.842, 0.003),
    ("DeepSeek     SENTRY-Fuse TPR@5%",
     ("corpora", "DeepSeek", "detectors", "SENTRY-Fuse (S3+S4)", "tpr_at_0.05_mean"),
     0.778, 0.003),
    ("Pooled       SENTRY-Fuse AUROC",
     ("pooled", "detectors", "SENTRY-Fuse (S3+S4)", "auroc_mean"),
     0.953, 0.003),
    ("Pooled       SENTRY-Fuse TPR@5%",
     ("pooled", "detectors", "SENTRY-Fuse (S3+S4)", "tpr_at_0.05_mean"),
     0.898, 0.003),
    ("Pooled       S1+S2 behavioural AUROC (near chance)",
     ("pooled", "detectors", "S1 + S2 behavioural", "auroc_mean"),
     0.557, 0.008),
]

CEILING_CHECKS = [
    ("Compromised trajectories (pooled)", ("pooled", "n_compromised"), 90, 0),
    ("Payload visible, permissive criterion",
     ("pooled", "payload_visible_in_trace", "frac"), 1.000, 0.001),
    ("Payload visible, strict criterion",
     ("pooled", "payload_visible_strict", "frac"), 0.867, 0.002),
    ("Full directive visible verbatim",
     ("pooled", "core_fully_visible", "frac"), 0.400, 0.002),
    ("Calls an effectful sink", ("pooled", "has_effectful_sink_call", "frac"),
     0.856, 0.002),
    ("nu, permissive criterion", ("pooled", "primary", "nu"), 0.000, 0.001),
    ("nu, strict criterion", ("pooled", "strict", "nu"), 0.044, 0.002),
    ("Distinctive directive visible for >= 5 words",
     ("pooled", "distinctive_core_visible", "5"), 1.000, 0.001),
    ("Distinctive directive visible for >= 30 words",
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
     ("by_calibration_size", "5", "psi_auroc"), 0.924, 0.001),
    ("Counterexample: saturation to one half",
     ("counterexamples", "saturation_to_half", "psi"), 0.500, 0.001),
    ("Counterexample: above chance to below chance",
     ("counterexamples", "below_half", "psi"), 0.333, 0.001),
]

COMBINATION_CHECKS = [
    ("e-average test inert below n_cal (alpha=0.05)",
     ("n_required_for_e_test",), 312, 0),
    ("max attainable e-value at n_cal=47",
     ("attainability", "47"), 4.56, 0.01),
    ("positives with BOTH signals at the p-value floor",
     ("floor_diagnostic", "both_at_floor"), 0.000, 1e-9),
    ("positives with exactly ONE signal at the floor",
     ("floor_diagnostic", "one_at_floor"), 0.416, 0.002),
    ("min_p a-priori TPR equals the one-at-floor rate",
     ("pooled", "rules", "min_p (Bonferroni)", "tpr_apriori_threshold"), 0.416, 0.002),
    ("sum_log union-bound a-priori TPR (inert)",
     ("pooled", "rules", "sum_log (Fisher), union bound", "tpr_apriori_threshold"), 0.000, 1e-9),
    ("e_avg a-priori TPR (inert)",
     ("pooled", "rules", "e_avg (calibrated)", "tpr_apriori_threshold"), 0.000, 1e-9),
    ("Fisher chi2 a-priori TPR (assumes independence)",
     ("pooled", "rules", "sum_log (Fisher), chi2", "tpr_apriori_threshold"), 0.850, 0.005),
    ("Fisher chi2 realised FPR under that threshold",
     ("pooled", "rules", "sum_log (Fisher), chi2", "fpr_apriori_threshold"), 0.026, 0.005),
    ("Stouffer a-priori TPR", 
     ("pooled", "rules", "stouffer", "tpr_apriori_threshold"), 0.867, 0.005),
    ("sum_log AUROC (ranking, pooled)",
     ("pooled", "rules", "sum_log (Fisher), union bound", "auroc_mean"), 0.953, 0.002),
    ("min_p AUROC (ranking, pooled)",
     ("pooled", "rules", "min_p (Bonferroni)", "auroc_mean"), 0.943, 0.002),
]

TASKSHIELD_CHECKS = [
    ("Task Shield pooled AUROC (vs SENTRY-Fuse 0.953)",
     ("pooled", "auroc"), 0.764, 0.002),
    ("Task Shield pooled published-rule TPR",
     ("pooled", "rule_tpr"), 0.933, 0.002),
    ("Task Shield pooled published-rule FPR",
     ("pooled", "rule_fpr"), 0.390, 0.002),
    ("Task Shield pooled TPR at our 5% target (quantisation)",
     ("pooled", "tpr_at_0.05_mean"), 0.000, 1e-9),
    ("Task Shield pooled distinct score values",
     ("pooled", "n_distinct_scores"), 7, 0),
    ("Task Shield pooled benign fraction at score floor",
     ("pooled", "frac_benign_at_floor"), 0.390, 0.002),
    ("Task Shield GPT-4o-mini AUROC",
     ("corpora", "GPT-4o-mini", "auroc"), 0.818, 0.002),
    ("Task Shield DeepSeek AUROC",
     ("corpora", "DeepSeek", "auroc"), 0.688, 0.002),
]

LONGHORIZON_CHECKS = [
    ("e-detector a=0.2   false alarms/1k",
     ("methods", "SENTRY (alpha=0.2)", "false_alarms_per_1k"), 45.6, 0.1),
    ("e-detector a=0.2   attack detection",
     ("methods", "SENTRY (alpha=0.2)", "attack_detection"), 0.612, 0.002),
    ("e-detector a=0.05  false alarms/1k (saturated)",
     ("methods", "SENTRY (alpha=0.05)", "false_alarms_per_1k"), 23.0, 0.1),
    ("e-detector a=0.002 identical to a=0.05 (saturation)",
     ("methods", "SENTRY (alpha=0.002)", "attack_detection"), 0.538, 0.002),
    ("fixed q=0.95       attack detection",
     ("methods", "fixed threshold (q=0.95)", "attack_detection"), 0.658, 0.002),
    ("fixed q=0.999      false alarms/1k",
     ("methods", "fixed threshold (q=0.999)", "false_alarms_per_1k"), 10.2, 0.1),
    ("fixed q=0.999      attack detection",
     ("methods", "fixed threshold (q=0.999)", "attack_detection"), 0.524, 0.002),
]

GENERALISATION_CHECKS = [
    ("DeepSeek/imp.instr.  attempt AUROC",
     ("A_baseline_deepseek_important", "auroc_attempt_max", "mean"), 0.76, 0.01),
    ("DeepSeek/InjecAgent  attempt AUROC",
     ("B_newattack_deepseek_injecagent", "auroc_attempt_max", "mean"), 0.71, 0.01),
    ("GPT-4o-mini/imp.instr. attempt AUROC",
     ("C_newagent_gpt4omini_important", "auroc_attempt_max", "mean"), 0.82, 0.01),
]

BOOTSTRAP_CHECKS = [
    ("AUROC gain over S3 (bootstrap mean)",
     ("bootstrap", "s3s4_minus_s3_auroc", "mean"), 0.0163, 0.0010),
    ("AUROC gain 95% CI lower bound",
     ("bootstrap", "s3s4_minus_s3_auroc", "ci_lo"), -0.0006, 0.0025),
    ("AUROC gain P(<=0)",
     ("bootstrap", "s3s4_minus_s3_auroc", "p_le_zero"), 0.030, 0.015),
    ("S5 contribution to AUROC (not significant)",
     ("bootstrap", "s5_contribution_auroc", "mean"), 0.0005, 0.0008),
    ("S5 contribution P(<=0) -- must be large",
     ("bootstrap", "s5_contribution_auroc", "p_le_zero"), 0.432, 0.06),
]

DISTRIBUTION_CHECKS = [
    ("Benign instruction-likeness median",
     ("benign_instruction_likeness", "median"), 0.000, 0.001),
    ("Benign instruction-likeness maximum",
     ("benign_instruction_likeness", "max"), 0.613, 0.001),
    ("InjecAgent maximum (must be below the benign max)",
     ("injecagent_instruction_likeness", "max"), 0.352, 0.001),
    ("S4 saturates at 1.0 on this %% of benign",
     ("s4_saturates_at_one", "pct"), 9.9, 0.1),
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


def _check_macros() -> int:
    """Assert the manuscript's generated macros agree with the JSON.

    report.py's own expectations are transcribed literals, so they can drift from
    the paper just as the prose once did. numbers.tex is generated from the JSON and
    is what the PDF actually renders, so re-generating it and diffing against the
    committed copy is the check that catches a stale manuscript.
    """
    import subprocess, sys
    gen = ROOT.parent / "manuscript" / "make_numbers.py"
    tex = ROOT.parent / "manuscript" / "numbers.tex"
    if not gen.exists() or not tex.exists():
        print("\n  MISSING: manuscript/numbers.tex -- run make_numbers.py")
        return 1
    before = tex.read_text()
    subprocess.run([sys.executable, str(gen)], check=True, capture_output=True,
                   cwd=str(ROOT.parent))
    after = tex.read_text()
    print("\n## Manuscript macros (numbers.tex regenerated from JSON)")
    if before == after:
        print("  numbers.tex is current: the PDF's numbers match the results files.")
        return 0
    n = sum(1 for a, b in zip(before.splitlines(), after.splitlines()) if a != b)
    print(f"  MISMATCH: numbers.tex was stale in {n} macro(s). The manuscript PDF")
    print("  was built from superseded values; rebuild it after this run.")
    return 1


def main() -> None:
    print("=" * 77)
    print("SENTRY -- reproduced vs. manuscript")
    print("=" * 77)
    bad, missing = 0, []
    bad += _check_macros()

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

    p = ROOT / "results_fused.json"
    if p.exists():
        print("\n## Trajectory bootstrap (claims about the method)")
        bad += _check(BOOTSTRAP_CHECKS, json.loads(p.read_text()), _dig)

    p = ROOT / "results_distributions.json"
    if p.exists():
        print("\n## Distributional statistics (Propositions 1 and 2)")
        bad += _check(DISTRIBUTION_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_distributions.json -- run: "
                       "python -m real_data.distributions")

    p = ROOT / "results_combination.json"
    if p.exists():
        print("\n## Combination rules: validity, attainability, and the mechanism")
        bad += _check(COMBINATION_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_combination.json -- run: "
                       "python -m real_data.combination_rules")

    p = ROOT / "results_taskshield.json"
    if p.exists():
        print("\n## Task Shield baseline, re-implemented on our corpus")
        bad += _check(TASKSHIELD_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_taskshield.json -- run: "
                       "python -m real_data.baseline_taskshield")

    p = ROOT / "results_longhorizon.json"
    if p.exists():
        print("\n## Long-horizon e-detector comparison (negative result)")
        bad += _check(LONGHORIZON_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_longhorizon.json -- run: python -m real_data.longhorizon")

    p = ROOT / "results_generalization.json"
    if p.exists():
        print("\n## Attempt detection and generalisation")
        bad += _check(GENERALISATION_CHECKS, json.loads(p.read_text()), _dig)
    else:
        missing.append("results_generalization.json -- run: "
                       "python -m real_data.evaluate_generalization")

    for m in missing:
        print(f"\n  MISSING: {m}")
    print()
    if bad or missing:
        if bad:
            print(f"{bad} value(s) disagree with the manuscript.")
        if missing:
            print(f"{len(missing)} results file(s) absent -- nothing was verified for them.")
        raise SystemExit(1)
    print("All checked values match the manuscript.")


if __name__ == "__main__":
    main()
