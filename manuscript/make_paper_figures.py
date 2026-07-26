"""Generate the figures the manuscript includes, from committed results only.

`make_figures.py` renders the corpus-level exploratory figures. This script
renders the three that previously had no generator at all (they were drawn by
hand in a scratch directory, so the paper's claim that every figure is
reproducible was false):

    overview.pdf   the SENTRY-Fuse pipeline diagram
    fusion.pdf     the signal ablation, from results_fused.json
    evidence.pdf   evidence-channel coverage, from results_ceiling_per_corpus.json

Run from the repo root:

    python manuscript/make_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})
C = {"obs": "#4C78A8", "act": "#E45756", "accent": "#F58518", "muted": "#8C8C8C",
     "green": "#54A24B", "purple": "#7B57C2", "box": "#F2F2F2"}

FUSED = json.loads((ROOT / "real_data" / "results_fused.json").read_text())
CEIL = json.loads((ROOT / "real_data" / "results_ceiling_per_corpus.json").read_text())


# --------------------------------------------------------------------------- #
# overview.pdf
# --------------------------------------------------------------------------- #

def _box(ax, x, y, w, h, text, fc, fontsize=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.0, edgecolor="#444444", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, zorder=5)


def _arrow(ax, xy, xytext, style="-|>"):
    ax.add_patch(FancyArrowPatch(xytext, xy, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.0, color="#444444",
                                 shrinkA=1, shrinkB=1))


def fig_overview():
    """Pipeline: trace -> three signals -> conformal surprisal -> sum -> alarm."""
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.axis("off")

    _box(ax, 0.0, 0.375, 0.152, 0.245,
         "agent trace\n$(o_t, a_t)_{t \\leq T}$\nand request $u$", C["box"],
         fontsize=8, weight="bold")

    sig = [
        (0.655, "S3  instruction-likeness\nof observation $o_{t-1}$", C["obs"]),
        (0.395, "S4  action-justification\naudit of $a_t$ given $u$", C["act"]),
        (0.135, "S5  sensitive-sink\nprovenance of $a_t$", C["act"]),
    ]
    for y, label, col in sig:
        _box(ax, 0.185, y, 0.295, 0.175, label, "#FFFFFF")
        ax.add_patch(FancyBboxPatch((0.185, y), 0.010, 0.175,
                                    boxstyle="square,pad=0", linewidth=0,
                                    facecolor=col))
        _arrow(ax, (0.183, y + 0.0875), (0.154, 0.4975))
        _box(ax, 0.505, y, 0.16, 0.175,
             "conformal\nsurprisal $\\psi_k$", C["box"])
        _arrow(ax, (0.503, y + 0.0875), (0.482, y + 0.0875))
        _arrow(ax, (0.700, 0.4975), (0.667, y + 0.0875))

    _box(ax, 0.700, 0.38, 0.115, 0.235, "$S=\\sum_k \\psi_k$", C["box"],
         fontsize=10, weight="bold")
    _box(ax, 0.855, 0.38, 0.145, 0.235, "alarm if\n$S > \\tau$", "#FFF1E6",
         weight="bold")
    _arrow(ax, (0.853, 0.4975), (0.817, 0.4975))

    ax.text(0.3325, 0.955, "one observation-side,\ntwo action-side signals",
            ha="center", va="top", fontsize=7.5, color="#333333",
            style="italic", linespacing=1.35)
    ax.text(0.585, 0.955, "scale-free,\nno tuned weights",
            ha="center", va="top", fontsize=7.5, color="#333333",
            style="italic", linespacing=1.35)
    ax.plot([0.171, 0.171], [0.10, 0.80], color=C["muted"], lw=0.6, ls=":")
    ax.plot([0.684, 0.684], [0.10, 0.80], color=C["muted"], lw=0.6, ls=":")

    fig.tight_layout()
    fig.savefig(FIG / "overview.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fusion.pdf
# --------------------------------------------------------------------------- #

ROWS = ["S3 instruction only", "S4 audit only", "S5 provenance only",
        "S1 + S2 behavioural", "S3 + S4", "SENTRY-Fuse (S3+S4+S5)"]
SHORT = ["S3\nonly", "S4\nonly", "S5\nonly", "S1+S2\nbehav.", "S3+S4",
         "SENTRY-\nFuse"]


def fig_fusion():
    """AUROC and TPR by signal subset, per corpus."""
    panels = [("GPT-4o-mini", FUSED["corpora"]["GPT-4o-mini"]),
              ("DeepSeek-V4-Flash", FUSED["corpora"]["DeepSeek"]),
              ("Pooled", FUSED["pooled"])]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), sharey=True)
    x = np.arange(len(ROWS))
    w = 0.38
    for ax, (name, blk) in zip(axes, panels):
        det = blk["detectors"]
        auroc = [det[r]["auroc_mean"] for r in ROWS]
        auroc_e = [det[r]["auroc_std"] for r in ROWS]
        tpr = [det[r]["tpr_at_0.05_mean"] for r in ROWS]
        tpr_e = [det[r]["tpr_at_0.05_std"] for r in ROWS]
        # The realised FPR that matters is the one at the proposed operating
        # point; it varies by detector, so averaging it across rows would be
        # misleading.
        realised = det["SENTRY-Fuse (S3+S4+S5)"]["realised_fpr_at_0.05"]
        b1 = ax.bar(x - w / 2, auroc, w, yerr=auroc_e, capsize=2.5,
                    color=C["obs"], label="AUROC")
        b2 = ax.bar(x + w / 2, tpr, w, yerr=tpr_e, capsize=2.5,
                    color=C["act"], label="TPR")
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT, fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{name}\n{blk['n_compromised']} compromised, "
                     f"{blk['n_test']} held-out benign\n"
                     f"SENTRY-Fuse realised FPR {realised:.1%}", fontsize=8.5)
        ax.axhline(1.0, color=C["muted"], lw=0.5, ls=":")
    axes[0].set_ylabel("score (mean $\\pm$ std, 10 seeds)")
    fig.legend([b1, b2], ["AUROC", "TPR at a 5% target"],
               frameon=False, fontsize=8.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("Signal ablation: the behavioural signals are near-chance, and "
                 "adding them costs accuracy", fontsize=9.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fusion.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# evidence.pdf
# --------------------------------------------------------------------------- #

def fig_evidence():
    """Evidence-channel coverage of the compromised class, per corpus."""
    labels, obs, obs_s, act, ns = [], [], [], [], []
    blocks = [("GPT-4o-mini", CEIL["corpora"]["GPT-4o-mini"]),
              ("DeepSeek", CEIL["corpora"]["DeepSeek"]),
              ("Pooled", CEIL["pooled"])]
    for name, r in blocks:
        labels.append(name)
        obs.append(r["payload_visible_in_trace"]["frac"])
        obs_s.append(r["payload_visible_strict"]["frac"])
        act.append(r["has_effectful_sink_call"]["frac"])
        ns.append(r["n_compromised"])

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w, obs, w, color=C["obs"], label="payload visible (permissive)")
    ax.bar(x, obs_s, w, color=C["purple"], label="payload visible (strict)")
    ax.bar(x + w, act, w, color=C["act"], label="calls an effectful sink")
    for xi, (o, o2, a) in zip(x, zip(obs, obs_s, act)):
        for dx, v in ((-w, o), (0, o2), (w, a)):
            ax.text(xi + dx, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=9)
    ax.set_ylim(0, 1.30)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("fraction of compromised trajectories")
    ax.axhline(1.0, color=C["muted"], lw=0.8, ls="--")
    ax.text((len(labels) - 1) / 2, 1.115,
            "observation channel near-saturated; action channel is not",
            ha="center", fontsize=8.5, color="#333333")
    ax.set_title("Where evidence of compromise lives",
                 fontsize=10, pad=22)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(FIG / "evidence.pdf")
    plt.close(fig)


def main() -> None:
    fig_overview(); print("  overview.pdf")
    fig_fusion(); print("  fusion.pdf")
    fig_evidence(); print("  evidence.pdf")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
