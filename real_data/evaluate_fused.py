"""SENTRY-Fuse: conformal fusion of heterogeneous trace signals.

The signals live on incompatible scales (a bigram surprise in nats, a [0,1] judge
score, a binary provenance flag), so summing them raw lets whichever happens to
have the widest range dominate. Instead each signal is converted to its conformal
tail probability against a BENIGN calibration split,

    p_k(x) = (1 + #{benign_k >= x}) / (n_k + 1),

and the fused statistic is the sum of surprisals, S = sum_k -log p_k(x_k). This is
scale-free and needs no tuned weights. It is NOT injective -- psi takes at most
n+1 values and saturates above the calibration maximum, so it can lose
discrimination; what it guarantees is a far finer quantisation than the
excess-over-maximum transform, which collapses a whole class to a point.

Evaluation protocol. Benign trajectories are split THREE ways per seed: one third
fits the reference model, one third supplies the conformal calibration values, one
third supplies the held-out negatives. Fitting and calibrating on the same split
would make the calibration values in-sample and the p-values anti-conservative.
Compromised trajectories are always held out. Reported over 10 seeds as
mean +/- std.

The reported operating point is the largest ATTAINABLE false-positive rate at or
below the target: with n held-out negatives only multiples of 1/n are reachable,
and ties in the discrete surprisal make the realised rate lower still, so
_tpr_at returns it explicitly rather than assuming the nominal value.

    SENTRY_IL_MODE=llm SENTRY_JUSTIFY=1 python -m real_data.evaluate_fused
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from real_data.adapters import load_agentdojo_logs
from real_data.sink_provenance import score_trajectory_from_log
from sentry.scores import SequentialWorldModel

ROOT = Path(__file__).parent
MIN_ACTIONS = 2
N_SEEDS = 10
FPR_POINTS = (0.0366, 0.05)
SOTA = {"name": "AgentArmor (arXiv:2508.01249v1)", "tpr": 0.9575, "fpr": 0.0366,
        "note": "v1 only; removed in v2/v3. GPT-4 agent, different corpus."}


def _auroc(neg, pos):
    """AUROC with ties counted as one half. Vectorised: the scalar double loop is
    too slow for the trajectory bootstrap."""
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (neg.size * pos.size))


def _tpr_at(neg, pos, fpr):
    """TPR at an attainable false-positive rate <= `fpr`, with that rate returned.

    Two things make the nominal target unreachable. First, with n held-out
    negatives only multiples of 1/n are attainable at all, so e.g. a 3.66% target
    on 35 negatives cannot be hit; interpolating an empirical quantile silently
    lands somewhere else (there, 2/35 = 5.71%). Second, the conformal surprisal is
    discrete -- at most n_cal+1 levels -- so negatives tie at the saturated value
    and FEWER than k of them may strictly exceed the k-th largest.

    We therefore take tau as the k-th largest negative for k = floor(fpr * n) and
    then MEASURE the realised rate as mean(neg > tau), which is <= k/n and often
    strictly less. Both numbers are returned so the caller can report the realised
    one rather than the nominal one. k = 0 means the target is finer than the
    sample can express; tau is then the maximum negative and the realised rate 0.
    """
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    neg = np.sort(np.asarray(neg, dtype=float))
    n = neg.size
    k = int(np.floor(fpr * n))
    # threshold strictly above the k-th largest negative => exactly k negatives
    # exceed it (ties permitting, which we account for by recomputing)
    tau = float(neg[n - 1] if k == 0 else neg[n - k - 1])
    realised = float(np.mean(neg > tau))
    return float(np.mean(np.asarray(pos, dtype=float) > tau)), realised


def _conformal_surprisal(cal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """-log of the conformal tail probability of each x against calibration."""
    cal = np.sort(np.asarray(cal, dtype=float))
    n = cal.size
    n_ge = n - np.searchsorted(cal, np.asarray(x, dtype=float), side="left")
    return -np.log((1.0 + n_ge) / (n + 1.0))


def _load(ld):
    pairs = [(t, m) for t, m in load_agentdojo_logs(ROOT / "agentdojo" / ld) if len(t) >= MIN_ACTIONS]
    benign, comp = [], []
    for t, m in pairs:
        log = json.loads(Path(m["path"]).read_text())
        if not m["is_attack"]:
            if m.get("utility") is not None:
                benign.append((t, log))
        elif m.get("security") is True:
            comp.append((t, log))
    return benign, comp


# Column order of the per-trajectory signal matrix. Each column is a SEPARATE
# conformal component: they are never combined on their raw scales, because the
# raw scales are incommensurable (nats, a [0,1] fraction, a [0,1] judge score, a
# binary flag). An earlier version summed transition + 4.0 * novelty before
# conformalising, which is exactly the tuned-weight failure this construction
# exists to avoid; the 4.0 is gone and novelty stands on its own.
COLUMNS = ("transition", "novelty", "instruction", "audit", "provenance")


def _signals(items, model):
    """Per-trajectory signal matrix, one column per signal in COLUMNS."""
    rows = []
    for t, log in items:
        best_tr, best_nv, hist = 0.0, 0.0, []
        for ctx, a in t:
            sv = model.signal_vector(ctx, hist, a)
            best_tr = max(best_tr, sv["transition"])
            best_nv = max(best_nv, sv["novelty"])
            hist = list(hist) + [(ctx, a)]
        instr = max((a.obs_instruction_likeness for _, a in t), default=0.0)
        audit = max((a.unjustified for _, a in t), default=0.0)
        prov = float(score_trajectory_from_log(log)[0] > 0) if log.get("messages") else 0.0
        rows.append([best_tr, best_nv, instr, audit, prov])
    return np.asarray(rows, dtype=float)


SIGNAL_SETS = {
    "S3 instruction only": [2],
    "S4 audit only": [3],
    "S5 provenance only": [4],
    "S1 transition only": [0],
    "S2 novelty only": [1],
    "S1 + S2 behavioural": [0, 1],
    "SENTRY-Fuse (S3+S4)": [2, 3],
    "S3+S4+S5 (adds provenance)": [2, 3, 4],
    "all five signals": [0, 1, 2, 3, 4],
}


def _corpus(ld):
    """Benign and compromised trajectories for one agent's corpus.

    AgentDojo only. An earlier version added the 50 tau-bench retail trajectories
    to the DeepSeek benign set, which was wrong twice over: it made a row labelled
    per-corpus actually a mixture of two domains with different tool vocabularies
    (straining the exchangeability Proposition 2 needs), and because those
    trajectories were inserted without their message list, the sink-provenance
    signal S5 was silently forced to 0 for all 50 rather than measured -- 26% of
    the pooled benign reference. That concentrated S5's benign distribution at
    zero and inflated the surprisal it assigned to positives. tau-bench is
    retained in the repository and used for attempt detection, where S5 plays no
    part.
    """
    return _load(ld)


def _split(benign, seed, four_way):
    """Disjoint benign splits for one seed.

    Three-way: fit / calibrate / evaluate. The threshold is then read off the
    evaluation negatives, so the reported false-positive rate is in-sample on
    them -- a point on their empirical ROC curve, not an out-of-sample guarantee.

    Four-way: fit / calibrate / threshold / evaluate, which makes the operating
    point genuinely out-of-sample at the cost of halving the negatives (and hence
    coarsening the attainable FPR grid to 1/n_eval).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(benign))
    if four_way:
        q = len(benign) // 4
        return ([benign[i] for i in idx[:q]], [benign[i] for i in idx[q:2 * q]],
                [benign[i] for i in idx[2 * q:3 * q]], [benign[i] for i in idx[3 * q:]])
    third = len(benign) // 3
    return ([benign[i] for i in idx[:third]], [benign[i] for i in idx[third:2 * third]],
            None, [benign[i] for i in idx[2 * third:]])


def _bootstrap_gain(benign, comp, set_a, set_b, metric, n_boot=2000, seed=0):
    """Cluster bootstrap CI for (metric of set_a) - (metric of set_b).

    The per-seed spread reported in the tables measures how much the answer moves
    when the same fixed sample is re-split. It says nothing about sampling error in
    the trajectories themselves, which is the uncertainty a claim about the method
    needs, so we resample trajectories with replacement.

    The subtlety that matters: the ten splits are re-splits of ONE corpus, and the
    positives are identical across them. A replicate must therefore draw a SINGLE
    trajectory resample and reuse it across all ten splits. Drawing an independent
    resample inside each split and averaging would estimate the variance of the mean
    of ten independent draws, shrinking the corpus-sampling variance by a factor of
    about ten -- which is exactly the error an earlier version of this function made,
    and it turned a borderline effect into an apparently decisive one.
    """
    rng = np.random.default_rng(seed)
    per_seed = []
    for sd in range(N_SEEDS):
        fit, cal, _, test = _split(benign, sd, False)
        model = SequentialWorldModel().fit([t for t, _ in fit])
        S_cal, S_test, S_pos = (_signals(x, model) for x in (cal, test, comp))
        row = {}
        for nm, cols in (("a", set_a), ("b", set_b)):
            row[nm] = (sum(_conformal_surprisal(S_cal[:, c], S_test[:, c]) for c in cols),
                       sum(_conformal_surprisal(S_cal[:, c], S_pos[:, c]) for c in cols))
        per_seed.append(row)

    n_pos = per_seed[0]["a"][1].size
    diffs = []
    for _ in range(n_boot):
        # one resample of the positives per replicate, shared across the splits
        i_pos = rng.integers(0, n_pos, n_pos)
        vals = []
        for row in per_seed:
            n_neg = row["a"][0].size
            i_neg = rng.integers(0, n_neg, n_neg)
            out = {}
            for nm in ("a", "b"):
                neg, pos = row[nm][0][i_neg], row[nm][1][i_pos]
                out[nm] = _auroc(neg, pos) if metric == "auroc" else _tpr_at(neg, pos, 0.05)[0]
            vals.append(out["a"] - out["b"])
        diffs.append(float(np.mean(vals)))
    d = np.sort(np.asarray(diffs))
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "ci_lo": float(d[int(0.025 * len(d))]),
            "ci_hi": float(d[int(0.975 * len(d)) - 1]),
            "p_le_zero": float(np.mean(d <= 0.0))}


def _evaluate(benign, comp, label, four_way=False):
    acc = {k: {"auroc": [], **{f: {"tpr": [], "realised": []} for f in FPR_POINTS}}
           for k in SIGNAL_SETS}
    for seed in range(N_SEEDS):
        fit, cal, thr, test = _split(benign, seed, four_way)
        model = SequentialWorldModel().fit([t for t, _ in fit])
        parts = (cal, test, comp) if thr is None else (cal, test, comp, thr)
        mats = [_signals(x, model) for x in parts]
        S_cal, S_test, S_pos = mats[0], mats[1], mats[2]
        S_thr = mats[3] if thr is not None else None
        for name, cols in SIGNAL_SETS.items():
            neg = sum(_conformal_surprisal(S_cal[:, c], S_test[:, c]) for c in cols)
            pos = sum(_conformal_surprisal(S_cal[:, c], S_pos[:, c]) for c in cols)
            acc[name]["auroc"].append(_auroc(neg, pos))
            for f in FPR_POINTS:
                if S_thr is None:
                    tpr, realised = _tpr_at(neg, pos, f)
                else:
                    # threshold from the disjoint threshold split; FPR and TPR
                    # then measured out-of-sample on `test`.
                    thr_scores = sum(_conformal_surprisal(S_cal[:, c], S_thr[:, c])
                                     for c in cols)
                    ts = np.sort(thr_scores)
                    k = int(np.floor(f * ts.size))
                    tau = float(ts[-1] if k == 0 else ts[ts.size - k - 1])
                    realised = float(np.mean(neg > tau))
                    tpr = float(np.mean(pos > tau))
                acc[name][f]["tpr"].append(tpr)
                acc[name][f]["realised"].append(realised)

    n_test = len(_split(benign, 0, four_way)[3])
    tag = "four-way (threshold out-of-sample)" if four_way else "three-way"
    print(f"\n=== {label}, {tag}: benign={len(benign)} eval-negatives={n_test} "
          f"compromised={len(comp)}, {N_SEEDS} seeds ===")
    print(f"    attainable FPR grid: multiples of {1/n_test:.4f}")
    hdr = f"{'detector':<24}{'AUROC':>14}" + "".join(
        f"{'TPR@'+f'{f:.2%}':>18}" for f in FPR_POINTS)
    print(hdr); print("-" * len(hdr))
    rec_all = {"n_benign": len(benign), "n_compromised": len(comp),
               "n_test": n_test, "attainable_fpr_step": 1.0 / n_test,
               "protocol": tag, "detectors": {}}
    for name in SIGNAL_SETS:
        a = np.array(acc[name]["auroc"])
        line = f"{name:<24}{a.mean():>8.3f}±{a.std():<5.3f}"
        rec = {"auroc_mean": float(a.mean()), "auroc_std": float(a.std())}
        for f in FPR_POINTS:
            v = np.array(acc[name][f]["tpr"]); r = np.array(acc[name][f]["realised"])
            line += f"{v.mean():>10.3f}±{v.std():<5.3f} [{r.mean():.3f}]"
            rec[f"tpr_at_{f}_mean"] = float(v.mean())
            rec[f"tpr_at_{f}_std"] = float(v.std())
            rec[f"realised_fpr_at_{f}"] = float(r.mean())
            rec[f"tpr_at_{f}_per_seed"] = [float(x) for x in acc[name][f]["tpr"]]
        rec_all["detectors"][name] = rec
        print(line)
    print("  [bracketed] = realised FPR")
    return rec_all


def main() -> None:
    corpora = {"GPT-4o-mini": "logs_gpt4omini", "DeepSeek": "logs_deepseek"}
    out = {"sota": SOTA, "n_seeds": N_SEEDS, "columns": list(COLUMNS), "corpora": {}}
    pooled_benign, pooled_comp = [], []
    for label, ld in corpora.items():
        benign, comp = _corpus(ld)
        pooled_benign += benign
        pooled_comp += comp
        if not comp:
            continue
        out["corpora"][label] = _evaluate(benign, comp, label)

    # Pooled across both agents. Reported so that no corpus can be quietly
    # dropped for being unfavourable.
    out["pooled"] = _evaluate(pooled_benign, pooled_comp, "Pooled")

    # Robustness: the same evaluation with the operating point set on a disjoint
    # fourth split, so the reported false-positive rate is out-of-sample.
    out["pooled_four_way"] = _evaluate(pooled_benign, pooled_comp, "Pooled",
                                       four_way=True)

    # Trajectory-level bootstrap for the two differences the paper claims.
    print("\n=== trajectory bootstrap (2000 replicates), pooled ===")
    out["bootstrap"] = {}
    for tag, a, b, metric in (
            ("s3s4_minus_s3_auroc", SIGNAL_SETS["SENTRY-Fuse (S3+S4)"],
             SIGNAL_SETS["S3 instruction only"], "auroc"),
            ("s3s4_minus_s3_tpr", SIGNAL_SETS["SENTRY-Fuse (S3+S4)"],
             SIGNAL_SETS["S3 instruction only"], "tpr"),
            ("fuse_minus_s3_auroc", SIGNAL_SETS["S3+S4+S5 (adds provenance)"],
             SIGNAL_SETS["S3 instruction only"], "auroc"),
            ("fuse_minus_s3_tpr", SIGNAL_SETS["S3+S4+S5 (adds provenance)"],
             SIGNAL_SETS["S3 instruction only"], "tpr"),
            ("s5_contribution_auroc", SIGNAL_SETS["S3+S4+S5 (adds provenance)"],
             SIGNAL_SETS["SENTRY-Fuse (S3+S4)"], "auroc"),
            ("s5_contribution_tpr", SIGNAL_SETS["S3+S4+S5 (adds provenance)"],
             SIGNAL_SETS["SENTRY-Fuse (S3+S4)"], "tpr")):
        r = _bootstrap_gain(pooled_benign, pooled_comp, a, b, metric)
        out["bootstrap"][tag] = r
        print(f"  {tag:<26} {r['mean']:+.4f}  95% CI [{r['ci_lo']:+.4f}, "
              f"{r['ci_hi']:+.4f}]  P(<=0)={r['p_le_zero']:.3f}")

    (ROOT / "results_fused.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'results_fused.json'}")
    print(f"\nreference point: {SOTA['name']} TPR={SOTA['tpr']} at FPR={SOTA['fpr']} "
          f"(arXiv:2508.01249v1, GPT-4 agent, different corpus -- NOT matched)")


if __name__ == "__main__":
    main()
