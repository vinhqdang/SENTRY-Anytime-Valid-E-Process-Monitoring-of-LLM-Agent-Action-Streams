"""SENTRY-Fuse: conformal fusion of heterogeneous trace signals.

The four signals live on incompatible scales (a bigram surprise in nats, a
[0,1] judge score, a binary provenance flag), so summing them raw lets whichever
happens to have the widest range dominate. Instead each signal is converted to
its conformal tail probability against a BENIGN calibration split,

    p_k(x) = (1 + #{benign_k >= x}) / (n_k + 1),

and the fused statistic is the sum of surprisals, S = sum_k -log p_k(x_k).
This is scale-free, monotone in every component (so no signal can be clipped
away -- the failure mode that destroyed the InjecAgent signal earlier), and
needs no tuned weights.

Evaluation protocol, to avoid the optimism of scoring a fused detector on the
data used to build it: benign trajectories are split in half per seed; one half
calibrates the conformal transforms and the operating threshold, the other half
provides the held-out false positives. Compromised trajectories are always
held out. Reported over 10 seeds as mean +/- std.

    SENTRY_IL_MODE=llm SENTRY_JUSTIFY=1 python -m real_data.evaluate_fused
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from real_data.adapters import load_agentdojo_logs, load_taubench_logs
from real_data.sink_provenance import score_trajectory_from_log
from sentry.scores import SequentialWorldModel

ROOT = Path(__file__).parent
MIN_ACTIONS = 2
N_SEEDS = 10
FPR_POINTS = (0.0366, 0.05)
SOTA = {"name": "AgentArmor (published)", "tpr": 0.9575, "fpr": 0.0366}


def _auroc(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(neg) * len(pos))


def _tpr_at(neg, pos, fpr):
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    tau = float(np.quantile(neg, 1.0 - fpr))
    return float(np.mean([x > tau for x in pos]))


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


def _signals(items, model):
    """Per-trajectory signal matrix: [behavioural, instruction, audit, provenance]."""
    rows = []
    for t, log in items:
        best_b, hist = 0.0, []
        for ctx, a in t:
            sv = model.signal_vector(ctx, hist, a)
            best_b = max(best_b, sv["transition"] + 4.0 * sv["novelty"])
            hist = list(hist) + [(ctx, a)]
        instr = max((a.obs_instruction_likeness for _, a in t), default=0.0)
        audit = max((a.unjustified for _, a in t), default=0.0)
        prov = float(score_trajectory_from_log(log)[0] > 0) if log.get("messages") else 0.0
        rows.append([best_b, instr, audit, prov])
    return np.asarray(rows, dtype=float)


SIGNAL_SETS = {
    "instruction only": [1],
    "behavioural only": [0],
    "instr + audit": [1, 2],
    "behav + instr": [0, 1],
    "SENTRY-Fuse (all 4)": [0, 1, 2, 3],
}


def main() -> None:
    corpora = {"GPT-4o-mini": "logs_gpt4omini", "DeepSeek": "logs_deepseek"}
    out = {"sota": SOTA, "n_seeds": N_SEEDS, "corpora": {}}
    for label, ld in corpora.items():
        benign, comp = _load(ld)
        if ld == "logs_deepseek":
            benign += [(t, {"messages": []}) for t, m in
                       load_taubench_logs(ROOT / "tau_bench" / "logs_deepseek")
                       if len(t) >= MIN_ACTIONS]
        if not comp:
            continue
        print(f"\n=== {label}: benign={len(benign)} compromised={len(comp)} "
              f"({N_SEEDS} seeds) ===")
        acc = {k: {"auroc": [], **{f: [] for f in FPR_POINTS}} for k in SIGNAL_SETS}
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(benign))
            half = len(benign) // 2
            cal = [benign[i] for i in idx[:half]]
            test = [benign[i] for i in idx[half:]]
            model = SequentialWorldModel().fit([t for t, _ in cal])
            S_cal, S_test, S_pos = (_signals(x, model) for x in (cal, test, comp))
            for name, cols in SIGNAL_SETS.items():
                neg = sum(_conformal_surprisal(S_cal[:, c], S_test[:, c]) for c in cols)
                pos = sum(_conformal_surprisal(S_cal[:, c], S_pos[:, c]) for c in cols)
                acc[name]["auroc"].append(_auroc(neg, pos))
                for f in FPR_POINTS:
                    acc[name][f].append(_tpr_at(neg, pos, f))
        hdr = f"{'detector':<24}{'AUROC':>14}" + "".join(f"{'TPR@'+f'{f:.2%}':>16}" for f in FPR_POINTS)
        print(hdr); print("-" * len(hdr))
        out["corpora"][label] = {"n_benign": len(benign), "n_compromised": len(comp),
                                 "detectors": {}}
        for name in SIGNAL_SETS:
            a = np.array(acc[name]["auroc"]); line = f"{name:<24}{a.mean():>8.3f}±{a.std():<5.3f}"
            rec = {"auroc_mean": float(a.mean()), "auroc_std": float(a.std())}
            for f in FPR_POINTS:
                v = np.array(acc[name][f])
                line += f"{v.mean():>10.3f}±{v.std():<5.3f}"
                rec[f"tpr_at_{f}_mean"] = float(v.mean()); rec[f"tpr_at_{f}_std"] = float(v.std())
            out["corpora"][label]["detectors"][name] = rec
            print(line)
        print(f"{SOTA['name']:<24}{'--':>14}{SOTA['tpr']:>16.3f}")

    (ROOT / "results_fused.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'results_fused.json'}")


if __name__ == "__main__":
    main()
