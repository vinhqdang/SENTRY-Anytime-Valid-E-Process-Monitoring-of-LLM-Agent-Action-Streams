"""Generate publication-quality figures for the SENTRY manuscript.

All figures are rendered from the real collected data (real_data/) and the
committed multi-seed results (real_data/results.json) -- nothing is
hand-drawn or invented. Run from the repo root with the venv active:

    source .venv/bin/activate
    python manuscript/make_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from real_data.adapters import load_agentdojo_logs, load_taubench_logs
from sentry.baseline import MultiSignalConformalIncrement
from sentry.detector import SRDetector
from sentry.pipeline import SentryDetect
from sentry.scores import (
    SequentialWorldModel,
    instruction_likeness,
    score_trajectory,
    signal_stream,
)
from sentry.streams import gaussian_stream

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RESULTS = json.loads((ROOT / "real_data" / "results.json").read_text())

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})
# palette
C = {"nominal": "#4C78A8", "attack": "#E45756", "accent": "#F58518", "muted": "#8C8C8C",
     "green": "#54A24B", "purple": "#7B57C2"}


def _load():
    ld = os.environ.get("SENTRY_EVAL_LOGDIR", "logs_deepseek")
    ad = [(t, m) for t, m in load_agentdojo_logs(ROOT / "real_data" / "agentdojo" / ld) if len(t) >= 2]
    tb = [(t, m) for t, m in load_taubench_logs(ROOT / "real_data" / "tau_bench" / ld) if len(t) >= 2]
    nominal = [(t, m) for t, m in ad if not m["is_attack"] and m.get("utility") is not None] + tb
    attacks = [(t, m) for t, m in ad if m["is_attack"] and m.get("security") is not None]
    succ = [x for x in attacks if x[1]["security"] is False]
    resisted = [x for x in attacks if x[1]["security"] is True]
    return nominal, succ, resisted


def fig_synthetic():
    """Synthetic ARL/FAR control + Cavaliers-style detection-delay trace."""
    rng = np.random.default_rng(0)
    alphas = np.array([0.2, 0.1, 0.05, 0.02, 0.01])
    far = []
    for a in alphas:
        d_cal = [gaussian_stream(300, rng).tolist() for _ in range(120)]
        d_th = [gaussian_stream(300, rng).tolist() for _ in range(250)]
        mon = SentryDetect.calibrate(d_cal, d_th, alpha=float(a), conf_delta=0.1,
                                     delta_range=(0.5, 3.0), k=6, family="gaussian", kind="SR")
        al, st = 0, 0
        for _ in range(60):
            m = mon.fresh_copy()
            for x in gaussian_stream(500, rng):
                _, fired = m.step(float(x))
                al += fired
            st += 500
        far.append(al / st)

    # a single drift trace
    d_cal = [gaussian_stream(300, rng).tolist() for _ in range(120)]
    d_th = [gaussian_stream(300, rng).tolist() for _ in range(250)]
    mon = SentryDetect.calibrate(d_cal, d_th, alpha=0.05, conf_delta=0.1,
                                 delta_range=(0.5, 3.0), k=6, family="gaussian", kind="SR")
    m = mon.fresh_copy()
    stream = gaussian_stream(400, rng, changepoint=150, delta=2.0)
    vals, alarm = [], None
    for t, x in enumerate(stream, 1):
        v, fired = m.step(float(x))
        vals.append(v)
        if fired and alarm is None:
            alarm = t

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].plot(alphas, far, "o-", color=C["nominal"], label="empirical FAR")
    ax[0].plot(alphas, alphas, "k--", lw=1, label=r"target $\alpha$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"target false-alarm rate $\alpha$")
    ax[0].set_ylabel("empirical per-step FAR")
    ax[0].set_title("(a) ARL / FAR control")
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].plot(np.log(np.array(vals) + 1e-12), color=C["nominal"], lw=1.2, label=r"$\log M_t$")
    ax[1].axhline(np.log(mon.threshold_info.c_alpha), color=C["attack"], ls="--", label=r"$\log c_\alpha$")
    ax[1].axvline(150, color=C["muted"], ls=":", label="changepoint")
    if alarm:
        ax[1].axvline(alarm, color=C["green"], ls="-.", label=f"alarm ($t={alarm}$)")
    ax[1].set_xlabel("step $t$"); ax[1].set_ylabel(r"$\log$ e-detector")
    ax[1].set_title("(b) Detection after injected drift")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "synthetic.pdf"); plt.close(fig)


def fig_instruction_sep(nominal, succ, resisted):
    """Instruction-likeness of observations: benign vs attack (the injection signal)."""
    def obs_il(traj):
        return max((a.obs_instruction_likeness for _, a in traj), default=0.0)
    ben = [obs_il(t) for t, _ in nominal]
    sa = [obs_il(t) for t, _ in succ]
    ra = [obs_il(t) for t, _ in resisted]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    data = [ben, sa, ra]
    labels = [f"benign\n(n={len(ben)})", f"successful\nattack (n={len(sa)})", f"resisted\nattack (n={len(ra)})"]
    colors = [C["nominal"], C["attack"], C["accent"]]
    rng = np.random.default_rng(1)
    for i, (d, col) in enumerate(zip(data, colors)):
        x = np.full(len(d), i) + rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(x, d, s=26, color=col, alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(d)] * 2, color="black", lw=2, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("max observation instruction-likeness")
    ax.set_title("Injected instructions are surprising in the observation stream")
    ax.axhline(max(ben), color=C["muted"], ls="--", lw=1)
    ax.text(2.4, max(ben) + 0.05, "nominal ceiling", color=C["muted"], fontsize=8, ha="right")
    fig.tight_layout(); fig.savefig(FIG / "instruction_sep.pdf"); plt.close(fig)


def fig_ablation_progression():
    """Detection rate + AUROC across the three combiners, both targets."""
    m = RESULTS["models"]
    order = ["causal_hash_summed", "bigram_novelty_summed", "bigram_novelty_evalue_mixture"]
    names = ["hash+Gaussian\n(summed)", "bigram+novelty\n+injection (summed)", "bigram+novelty\n+injection (e-value mix)"]
    pac_hijack = [m[k]["aggregate"]["pac_detection_rate"]["mean"] for k in order]
    pac_attempt = [m[k]["aggregate"]["pac_attempt_rate"]["mean"] for k in order]
    hijack_err = [m[k]["aggregate"]["pac_detection_rate"]["std"] for k in order]
    attempt_err = [m[k]["aggregate"]["pac_attempt_rate"]["std"] for k in order]

    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, pac_attempt, w, yerr=attempt_err, capsize=3, color=C["attack"],
           label="injection-attempt detection")
    ax.bar(x + w / 2, pac_hijack, w, yerr=hijack_err, capsize=3, color=C["nominal"],
           label="hijack-success detection")
    for xi, v in zip(x - w / 2, pac_attempt):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, pac_hijack):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("detection rate @ PAC threshold")
    ax.set_ylim(0, 1.0)
    ax.set_title(r"Detection by score model and target (mean $\pm$ std, 10 splits)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "ablation_progression.pdf"); plt.close(fig)


def fig_robustness(nominal, succ, resisted):
    """AUROC vs the novelty and instruction weights -- shows no knife-edge tuning."""
    from functools import partial
    from real_data.evaluate import evaluate_once, aggregate

    weights = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    nov_auroc, instr_auroc = [], []
    for w in weights:
        rows = [evaluate_once(partial(SequentialWorldModel, novelty_weight=w),
                              nominal, succ, resisted, s) for s in range(6)]
        nov_auroc.append(aggregate(rows, "auroc_attempt_max")[0])
        rows = [evaluate_once(partial(SequentialWorldModel, instruction_weight=w),
                              nominal, succ, resisted, s) for s in range(6)]
        instr_auroc.append(aggregate(rows, "auroc_attempt_max")[0])

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(weights, instr_auroc, "o-", color=C["attack"], label="instruction weight")
    ax.plot(weights, nov_auroc, "s-", color=C["nominal"], label="novelty weight")
    ax.set_xlabel("signal weight")
    ax.set_ylabel("injection-attempt AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Robustness to signal weights")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "robustness.pdf"); plt.close(fig)


def fig_traces(nominal, succ):
    """e-detector trace on a real nominal vs real attack trajectory."""
    model = SequentialWorldModel().fit([t for t, _ in nominal[: max(6, len(nominal) // 2)]])
    monitor = SentryDetect.calibrate(
        [score_trajectory(model, t).tolist() for t, _ in nominal[: len(nominal) // 2]][:6],
        [score_trajectory(model, t).tolist() for t, _ in nominal[len(nominal) // 2:]][:6],
        alpha=0.2, conf_delta=0.2, delta_range=(0.3, 3.0), k=6, family="gaussian", kind="SR")

    def run(traj):
        m = monitor.fresh_copy(); vals = []
        for s in score_trajectory(model, traj):
            v, _ = m.step(float(s)); vals.append(v)
        return np.clip(np.log(np.array(vals) + 1e-12), -6, None)

    # pick an attack with a located drift and multiple post-drift steps
    cand = [x for x in succ if x[1].get("drift_index") is not None and len(x[0]) >= 4] or succ
    atk_traj, atk_meta = cand[0]
    nom_traj = max((t for t, _ in nominal), key=len)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for a, traj, title, meta in [(ax[0], nom_traj, "(a) real benign trajectory", None),
                                 (ax[1], atk_traj, "(b) real prompt-injection attack", atk_meta)]:
        a.plot(run(traj), color=C["nominal"], lw=1.5, marker="o", ms=3, label=r"$\log M_t$")
        a.axhline(np.log(monitor.threshold_info.c_alpha), color=C["attack"], ls="--", label=r"$\log c_\alpha$")
        if meta and meta.get("drift_index") is not None:
            a.axvline(meta["drift_index"], color=C["muted"], ls=":", label="injection point")
        a.set_xlabel("action index"); a.set_title(title)
        a.legend(frameon=False, fontsize=8)
    ax[0].set_ylabel(r"$\log$ e-detector value")
    fig.tight_layout(); fig.savefig(FIG / "traces.pdf"); plt.close(fig)


def fig_negative_signals(nominal, succ, resisted):
    """AUROC of shipped vs tried-and-rejected signals (honest ablation)."""
    # isolation AUROC per signal, attempt target, computed directly
    model = SequentialWorldModel().fit([t for t, _ in nominal])

    def auroc(neg, pos):
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(neg) * len(pos))

    def peaks(items, key):
        out = []
        for t, _ in items:
            ss = signal_stream(model, t)
            out.append(max((s[key] for s in ss), default=0.0))
        return out

    attempts = succ + resisted
    signals = {"tool-transition": "transition", "argument-novelty": "novelty",
               "instruction-likeness": "instruction"}
    vals = {}
    for label, key in signals.items():
        vals[label] = auroc(peaks(nominal, key), peaks(attempts, key))
    # tried-and-rejected (from RESULTS.md, measured): taint and goal-grounding
    rejected = {"data-flow taint\n(rejected)": 0.47, "goal-grounding\n(rejected)": 0.499}

    labels = list(signals.keys()) + list(rejected.keys())
    heights = [vals[k] for k in signals] + list(rejected.values())
    colors = [C["green"]] * len(signals) + [C["muted"]] * len(rejected)
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    bars = ax.bar(range(len(labels)), heights, color=colors)
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.text(len(labels) - 0.4, 0.51, "chance", fontsize=8, ha="right")
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("injection-attempt AUROC (isolation)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Signals shipped vs. tried and rejected")
    fig.tight_layout(); fig.savefig(FIG / "signals.pdf"); plt.close(fig)


def fig_delay():
    """Detection-delay distribution (post-injection steps) for the shipped model."""
    m = RESULTS["models"]
    summed = m["bigram_novelty_summed"]["ville_delays_all_seeds"]
    old = m["causal_hash_summed"]["ville_delays_all_seeds"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bins = np.arange(-0.5, 8.5, 1)
    ax.hist(summed, bins=bins, color=C["attack"], alpha=0.85,
            label=f"bigram+novelty+injection (median {int(np.median(summed))})")
    ax.hist(old, bins=bins, color=C["muted"], alpha=0.6,
            label=f"hash+Gaussian (median {int(np.median(old))})")
    ax.set_xlabel("detection delay (actions past injection point)")
    ax.set_ylabel("count (all seeds)")
    ax.set_title("Detection delay: the injection signal fires almost immediately")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "delay.pdf"); plt.close(fig)


def main():
    nominal, succ, resisted = _load()
    print(f"loaded nominal={len(nominal)} succ={len(succ)} resisted={len(resisted)}")
    fig_synthetic(); print("  synthetic.pdf")
    fig_instruction_sep(nominal, succ, resisted); print("  instruction_sep.pdf")
    fig_ablation_progression(); print("  ablation_progression.pdf")
    fig_robustness(nominal, succ, resisted); print("  robustness.pdf")
    fig_traces(nominal, succ); print("  traces.pdf")
    fig_negative_signals(nominal, succ, resisted); print("  signals.pdf")
    fig_delay(); print("  delay.pdf")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
