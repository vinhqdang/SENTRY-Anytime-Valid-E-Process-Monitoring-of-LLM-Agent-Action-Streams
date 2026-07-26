"""How much AUROC does the conformal transform cost? (Proposition 3(iii).)

psi takes at most n_cal+1 values and saturates above the calibration maximum, so
it is many-to-one and can lose discrimination. This quantifies the loss on
synthetic Gaussians as a function of calibration size, and reproduces the two
counterexamples quoted in the paper.

    python -m real_data.quantisation_demo
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
N_REPLICATES = 400
SEPARATION = 6.0          # sigma; enough that the raw signal separates perfectly
CAL_SIZES = (1, 5, 10, 23, 35, 63, 200)
N_EVAL = 200


def surprisal(cal: np.ndarray, x: np.ndarray) -> np.ndarray:
    cal = np.sort(np.asarray(cal, dtype=float))
    n = cal.size
    n_ge = n - np.searchsorted(cal, np.asarray(x, dtype=float), side="left")
    return -np.log((1.0 + n_ge) / (n + 1.0))


def auroc(neg, pos) -> float:
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def main() -> None:
    out = {"n_replicates": N_REPLICATES, "separation_sigma": SEPARATION,
           "n_eval_per_class": N_EVAL, "by_calibration_size": {}}

    print(f"Gaussian benign N(0,1) vs attack N({SEPARATION:g},1), "
          f"{N_EVAL} per class, {N_REPLICATES} replicates\n")
    print(f"{'n_cal':>7}{'raw AUROC':>12}{'psi AUROC':>12}{'loss':>10}")
    print("-" * 41)
    for n_cal in CAL_SIZES:
        raws, psis = [], []
        for r in range(N_REPLICATES):
            rng = np.random.default_rng(r)
            cal = rng.normal(0.0, 1.0, n_cal)
            ben = rng.normal(0.0, 1.0, N_EVAL)
            atk = rng.normal(SEPARATION, 1.0, N_EVAL)
            raws.append(auroc(ben, atk))
            psis.append(auroc(surprisal(cal, ben), surprisal(cal, atk)))
        raw, psi = float(np.mean(raws)), float(np.mean(psis))
        out["by_calibration_size"][str(n_cal)] = {
            "raw_auroc": round(raw, 4), "psi_auroc": round(psi, 4),
            "loss": round(raw - psi, 4)}
        print(f"{n_cal:>7}{raw:>12.4f}{psi:>12.4f}{raw - psi:>10.4f}")

    # The two exact counterexamples quoted in the paper.
    ce = {}
    cal, ben, atk = [0.0], [0.1, 0.2], [0.3, 0.4]
    ce["saturation_to_half"] = {
        "calibration": cal, "benign": ben, "attack": atk,
        "raw": auroc(ben, atk),
        "psi": auroc(surprisal(np.array(cal), ben), surprisal(np.array(cal), atk))}
    cal, ben, atk = [10.0], [0.0, 0.0, 11.0], [1.0, 2.0]
    ce["below_half"] = {
        "calibration": cal, "benign": ben, "attack": atk,
        "raw": auroc(ben, atk),
        "psi": auroc(surprisal(np.array(cal), ben), surprisal(np.array(cal), atk))}
    out["counterexamples"] = ce
    print("\ncounterexamples quoted in the paper:")
    for k, v in ce.items():
        print(f"  {k:<20} raw {v['raw']:.3f} -> psi {v['psi']:.3f}")

    (ROOT / "results_quantisation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'results_quantisation.json'}")


if __name__ == "__main__":
    main()
