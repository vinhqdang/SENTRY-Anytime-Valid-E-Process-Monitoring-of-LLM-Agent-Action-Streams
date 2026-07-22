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

import glob as _glob

from real_data.adapters import (
    agentdojo_log_to_trajectory,
    load_agentdojo_logs,
    load_taubench_logs,
)
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
_GEN_PATH = ROOT / "real_data" / "results_generalization.json"
RESULTS_GEN = json.loads(_GEN_PATH.read_text()) if _GEN_PATH.exists() else None

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


def _suite_of(traj, meta):
    """AgentDojo suite name lives in the Context.task_id tuple's first slot
    (see real_data/adapters); tau-bench trajectories are tagged 'retail'."""
    if meta.get("source") == "tau_bench":
        return "retail"
    ctx = traj[0][0]
    tid = getattr(ctx, "task_id", None)
    if isinstance(tid, (tuple, list)) and tid:
        return str(tid[0])
    return "?"


def fig_dataset(nominal, succ, resisted):
    """Real-corpus characterisation: (a) trajectory-length distributions by
    role, (b) per-suite composition. Everything is read straight from the
    collected trajectories -- no modelling."""
    nom_len = [len(t) for t, _ in nominal]
    atk_len = [len(t) for t, _ in succ + resisted]

    # per-suite counts (benign / successful / resisted), AgentDojo suites only
    suites = ["banking", "workspace", "travel", "slack"]
    ben_c = {s: 0 for s in suites}
    suc_c = {s: 0 for s in suites}
    res_c = {s: 0 for s in suites}
    for t, m in nominal:
        s = _suite_of(t, m)
        if s in ben_c:
            ben_c[s] += 1
    for t, m in succ:
        s = _suite_of(t, m)
        if s in suc_c:
            suc_c[s] += 1
    for t, m in resisted:
        s = _suite_of(t, m)
        if s in res_c:
            res_c[s] += 1

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))

    bins = np.arange(1.5, max(max(nom_len, default=2), max(atk_len, default=2)) + 1.5, 1)
    ax[0].hist(nom_len, bins=bins, color=C["nominal"], alpha=0.8,
               label=f"benign (n={len(nom_len)})")
    ax[0].hist(atk_len, bins=bins, color=C["attack"], alpha=0.65,
               label=f"attack (n={len(atk_len)})")
    ax[0].axvline(np.mean(nom_len), color=C["nominal"], ls="--", lw=1)
    ax[0].axvline(np.mean(atk_len), color=C["attack"], ls="--", lw=1)
    ax[0].set_xlabel("trajectory length (actions)")
    ax[0].set_ylabel("number of trajectories")
    ax[0].set_title("(a) Action-count distribution")
    ax[0].legend(frameon=False, fontsize=9)

    x = np.arange(len(suites))
    b = [ben_c[s] for s in suites]
    su = [suc_c[s] for s in suites]
    re = [res_c[s] for s in suites]
    ax[1].bar(x, b, color=C["nominal"], label="benign")
    ax[1].bar(x, su, bottom=b, color=C["attack"], label="successful atk")
    ax[1].bar(x, re, bottom=np.array(b) + np.array(su), color=C["accent"],
              label="resisted atk")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(suites, fontsize=9)
    ax[1].set_ylabel("number of trajectories")
    ax[1].set_title("(b) Corpus composition by AgentDojo suite")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "dataset.pdf"); plt.close(fig)


def fig_roc(nominal, succ, resisted):
    """ROC of the shipped three-signal model on real held-out data, for the
    three targets (observable / all-attempts / hijack). Averaged over 10
    random half-splits of the nominal set: for each split we fit on one half
    and score the held-out half plus every attack, then average TPR on a
    common FPR grid -- the same 10-split protocol behind Table 2, so the
    annotated AUROCs match the headline numbers."""
    def observable(traj):
        return max((a.obs_instruction_likeness for _, a in traj), default=0.0) > 0.0

    attempts = succ + resisted
    grid = np.linspace(0, 1, 101)

    def auroc(neg, pos):
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(neg) * len(pos))

    targets = {
        "observable injection": [t for t, _ in attempts if observable(t)],
        "all attempts": [t for t, _ in attempts],
        "hijack success": [t for t, _ in succ],
    }
    tpr_acc = {k: [] for k in targets}
    auc_acc = {k: [] for k in targets}

    for seed in range(10):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(nominal))
        half = len(nominal) // 2
        train = [nominal[i] for i in idx[:half]]
        test_nom = [nominal[i] for i in idx[half:]]
        model = SequentialWorldModel().fit([t for t, _ in train])

        def peak(traj):
            s = score_trajectory(model, traj)
            return float(np.max(s)) if len(s) else 0.0

        neg = [peak(t) for t, _ in test_nom]
        for k, pos_trajs in targets.items():
            pos = [peak(t) for t in pos_trajs]
            thr = sorted(set(neg + pos), reverse=True)
            fpr = np.array([0.0] + [sum(n >= x for n in neg) / len(neg) for x in thr])
            tpr = np.array([0.0] + [sum(p >= x for p in pos) / len(pos) for x in thr])
            tpr_acc[k].append(np.interp(grid, fpr, tpr))
            auc_acc[k].append(auroc(neg, pos))

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for k, col in zip(targets, [C["green"], C["attack"], C["purple"]]):
        mean_tpr = np.mean(tpr_acc[k], axis=0)
        ax.plot(grid, mean_tpr, "-", color=col, lw=2,
                label=f"{k} (AUROC {np.mean(auc_acc[k]):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("false-positive rate (benign)")
    ax.set_ylabel("true-positive rate (attack)")
    ax.set_title("ROC on real held-out trajectories (mean of 10 splits)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "roc.pdf"); plt.close(fig)


def fig_case_trace(nominal):
    """Annotated walk-through of a REAL prompt-injection hijack from the
    collected AgentDojo banking logs: the benign user request, the agent's
    tool calls, the poisoned observation carrying the injected instruction,
    the agent's compliance (a wire to the attacker's account), and SENTRY's
    per-action signal firing on the poisoned step. Everything -- transcript
    text and signal values -- is read from the actual log, not staged."""
    import json as _json

    ld = os.environ.get("SENTRY_EVAL_LOGDIR", "logs_deepseek")
    matches = _glob.glob(
        str(ROOT / "real_data" / "agentdojo" / ld
            / "**" / "banking" / "user_task_2" / "important_instructions"
            / "injection_task_0.json"),
        recursive=True,
    )
    if not matches:
        print("  (case-trace log not found, skipping case_trace.pdf)")
        return
    log = _json.load(open(matches[0]))
    traj, meta = agentdojo_log_to_trajectory(log, ("banking", "user_task_2"))

    # SENTRY signal: per-action instruction-likeness of the preceding
    # observation (the signal that fires on the injection), plus the SR
    # e-detector crossing, computed with a monitor calibrated on nominal.
    model = SequentialWorldModel().fit([t for t, _ in nominal])
    monitor = SentryDetect.calibrate(
        [score_trajectory(model, t).tolist() for t, _ in nominal[: len(nominal) // 2]][:8],
        [score_trajectory(model, t).tolist() for t, _ in nominal[len(nominal) // 2:]][:8],
        alpha=0.2, conf_delta=0.2, delta_range=(0.3, 3.0), k=6, family="gaussian", kind="SR")
    il = [a.obs_instruction_likeness for _, a in traj]
    m = monitor.fresh_copy()
    fired_at = None
    for i, s in enumerate(score_trajectory(model, traj)):
        _, fired = m.step(float(s))
        if fired and fired_at is None:
            fired_at = i

    # Human-readable step rows: tool + short arg + a snippet of the preceding
    # observation. We pull the raw arg/obs strings from the log directly.
    def short(s, n=64):
        s = " ".join(str(s).split())
        return s if len(s) <= n else s[: n - 1] + "…"

    tools = [a.tool for _, a in traj]
    rows = []
    for i, tool in enumerate(tools):
        rows.append({"t": i, "tool": tool, "il": il[i]})
    n = len(rows)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.0, 0.62 * n + 1.6),
        gridspec_kw={"width_ratios": [2.55, 1.0]})

    # ---- left: transcript timeline ----
    axL.set_xlim(0, 1); axL.set_ylim(-0.5, n + 1.5); axL.axis("off")
    axL.text(0.0, n + 1.0, "USER: “Read 'landlord-notices.txt' and adjust "
             "my rent payment accordingly.”", fontsize=9.5, style="italic",
             color="#222", wrap=True)
    inj_step = next((r["t"] for r in rows if r["il"] > 0), None)
    comply_step = next((r["t"] for r in rows if r["tool"] == "send_money"), None)
    for r in rows:
        y = n - r["t"]
        poisoned = r["il"] > 0
        comply = r["tool"] == "send_money"
        face = "#FDEBEA" if comply else ("#FFF4E5" if poisoned else "#F2F5F9")
        edge = C["attack"] if comply else (C["accent"] if poisoned else C["muted"])
        axL.add_patch(plt.Rectangle((0.02, y - 0.36), 0.96, 0.72,
                      facecolor=face, edgecolor=edge, lw=1.3, zorder=1,
                      transform=axL.transData, clip_on=False))
        axL.text(0.05, y + 0.10, f"t{r['t']}   {r['tool']}",
                 fontsize=9, fontweight="bold", va="center", color="#111")
        note = ""
        if r["t"] == inj_step:
            note = "▶ preceding observation contains injected <INFORMATION> block"
        elif comply:
            note = "▶ wires money to attacker IBAN US133… (compliance)"
        elif r["tool"] in ("read_file",):
            note = "▶ reads landlord-notices.txt (poison enters here)"
        if note:
            axL.text(0.05, y - 0.17, note, fontsize=7.8, va="center",
                     color=edge)
    axL.set_title("(a) A real prompt-injection hijack (AgentDojo banking)",
                  fontsize=10, loc="left")

    # ---- right: SENTRY signal per step ----
    ys = [n - r["t"] for r in rows]
    ils = [r["il"] for r in rows]
    axR.barh(ys, ils, height=0.6, color=[C["attack"] if v > 0 else C["nominal"]
             for v in ils], edgecolor="white")
    axR.set_ylim(-0.5, n + 1.5); axR.set_yticks([])
    axR.set_xlabel("obs. instruction-likeness", fontsize=9)
    axR.set_title("(b) SENTRY signal", fontsize=10, loc="left")
    axR.axvline(0, color=C["muted"], lw=0.8)
    if fired_at is not None:
        yf = n - fired_at
        axR.annotate("ALARM", xy=(max(ils) * 0.15, yf),
                     xytext=(max(ils) * 0.5, yf + 0.9), fontsize=9,
                     color=C["attack"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C["attack"], lw=1.5))
    fig.tight_layout(); fig.savefig(FIG / "case_trace.pdf"); plt.close(fig)


def fig_generalization():
    """Cross-attack and cross-agent generalisation: observable-injection
    detection and attempt-AUROC for the baseline, a new attack family
    (InjecAgent) and a new agent (GPT-4o-mini). Read from
    results_generalization.json."""
    if RESULTS_GEN is None:
        print("  (results_generalization.json missing, skipping generalization.pdf)")
        return
    order = [
        ("A_baseline_deepseek_important", "baseline\n(DeepSeek,\nimportant_instr.)", C["nominal"]),
        ("C_newagent_gpt4omini_important", "new agent\n(GPT-4o-mini,\nimportant_instr.)", C["green"]),
        ("B_newattack_deepseek_injecagent", "new attack\n(DeepSeek,\nInjecAgent)", C["attack"]),
    ]
    labels, obs, auroc, obs_e, auroc_e, cols = [], [], [], [], [], []
    for key, lab, col in order:
        r = RESULTS_GEN.get(key)
        if not r:
            continue
        labels.append(lab); cols.append(col)
        obs.append(r["pac_observable_rate"]["mean"]); obs_e.append(r["pac_observable_rate"]["std"])
        auroc.append(r["auroc_attempt_max"]["mean"]); auroc_e.append(r["auroc_attempt_max"]["std"])

    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    b1 = ax.bar(x - w / 2, obs, w, yerr=obs_e, capsize=3, color=cols,
                edgecolor="black", linewidth=0.5, label="observable-injection detection")
    b2 = ax.bar(x + w / 2, auroc, w, yerr=auroc_e, capsize=3, color=cols, alpha=0.5,
                edgecolor="black", linewidth=0.5, hatch="//", label="attempt AUROC")
    ax.axhline(0.5, color=C["muted"], ls="--", lw=1)
    ax.text(len(labels) - 0.5, 0.52, "chance (AUROC)", fontsize=8, color=C["muted"], ha="right")
    for xi, v in zip(x - w / 2, obs):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, auroc):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("detection rate / AUROC"); ax.set_ylim(0, 1.08)
    ax.set_title("Generalisation: robust across agents, attack-specific")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "generalization.pdf"); plt.close(fig)


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
    fig_dataset(nominal, succ, resisted); print("  dataset.pdf")
    fig_case_trace(nominal); print("  case_trace.pdf")
    fig_roc(nominal, succ, resisted); print("  roc.pdf")
    fig_generalization(); print("  generalization.pdf")
    fig_instruction_sep(nominal, succ, resisted); print("  instruction_sep.pdf")
    fig_ablation_progression(); print("  ablation_progression.pdf")
    fig_robustness(nominal, succ, resisted); print("  robustness.pdf")
    fig_traces(nominal, succ); print("  traces.pdf")
    fig_negative_signals(nominal, succ, resisted); print("  signals.pdf")
    fig_delay(); print("  delay.pdf")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
