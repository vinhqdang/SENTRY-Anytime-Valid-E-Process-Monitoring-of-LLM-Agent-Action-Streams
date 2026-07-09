"""Calibrate and evaluate SENTRY-Detect on real AgentDojo + tau-bench agent
trajectories (algorithm.md §7 items 1-2, on real rather than synthetic data).

Pipeline: fit CausalWorldModel on a training split of nominal trajectories
-> score every trajectory into a surprise stream -> calibrate SentryDetect
(Phases 1-2) on held-out nominal splits -> evaluate empirical FAR on a
further held-out nominal split and detection delay on real AgentDojo attack
trajectories where the injection actually hijacked the agent
(security == False).

Run from the repo root with the project venv active:
    source .venv/bin/activate
    python -m real_data.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from real_data.adapters import load_agentdojo_logs, load_taubench_logs
from sentry.pipeline import SentryDetect
from sentry.scores import CausalWorldModel, score_trajectory

ROOT = Path(__file__).parent
MIN_ACTIONS = 2


def _complete(meta: dict) -> bool:
    return meta.get("utility") is not None


def load_all():
    agentdojo = [
        (t, m) for t, m in load_agentdojo_logs(ROOT / "agentdojo" / "logs") if len(t) >= MIN_ACTIONS
    ]
    taubench = [(t, m) for t, m in load_taubench_logs(ROOT / "tau_bench" / "logs") if len(t) >= MIN_ACTIONS]

    nominal_agentdojo = [(t, m) for t, m in agentdojo if not m["is_attack"] and _complete(m)]
    attacks = [(t, m) for t, m in agentdojo if m["is_attack"] and m.get("security") is not None]
    successful_attacks = [(t, m) for t, m in attacks if m["security"] is False]
    resisted_attacks = [(t, m) for t, m in attacks if m["security"] is True]

    nominal = nominal_agentdojo + taubench
    return nominal, successful_attacks, resisted_attacks


def split(items: list, fracs: tuple[float, ...], rng: np.random.Generator) -> list[list]:
    idx = rng.permutation(len(items))
    n = len(items)
    bounds = np.cumsum([int(n * f) for f in fracs])
    parts, start = [], 0
    for b in bounds:
        parts.append([items[i] for i in idx[start:b]])
        start = b
    parts.append([items[i] for i in idx[start:]])  # remainder -> last bucket
    return parts[:-1] if len(parts) > len(fracs) else parts


def main() -> None:
    rng = np.random.default_rng(0)
    nominal, successful_attacks, resisted_attacks = load_all()

    print(f"nominal trajectories: {len(nominal)}")
    print(f"successful-attack trajectories: {len(successful_attacks)}")
    print(f"resisted-attack trajectories: {len(resisted_attacks)}")

    if len(nominal) < 8:
        raise SystemExit(
            f"only {len(nominal)} usable nominal trajectories -- need more data collected "
            "before calibration is meaningful (see real_data/agentdojo/run.py, real_data/tau_bench/run.py)"
        )

    n = len(nominal)
    n_train = max(4, n // 3)
    n_cal = max(2, n // 4)
    train, cal, rest = split(nominal, (n_train / n, n_cal / n), rng)
    half = max(1, len(rest) // 2)
    thresh, test_nominal = rest[:half], rest[half:]
    print(f"split: train={len(train)} cal={len(cal)} thresh={len(thresh)} test_nominal={len(test_nominal)}")

    model = CausalWorldModel().fit([t for t, _ in train])

    def scores(items):
        return [score_trajectory(model, t).tolist() for t, _ in items if len(t) >= MIN_ACTIONS]

    cal_scores = scores(cal)
    thresh_scores = scores(thresh)
    test_nominal_scores = scores(test_nominal)
    attack_scores = scores(successful_attacks)
    resisted_scores = scores(resisted_attacks)

    pooled = np.concatenate([np.asarray(s) for s in cal_scores])
    delta_lo = max(0.05, float(pooled.std()) * 0.5)
    delta_hi = max(delta_lo * 4, float(pooled.std()) * 6 + 1e-3)

    # Small real-data sample sizes cannot support an aggressive (alpha, delta)
    # target -- report what's actually achievable rather than overclaiming.
    alpha, conf_delta = 0.2, 0.2

    monitor = SentryDetect.calibrate(
        d_cal_streams=cal_scores,
        d_thresh_streams=thresh_scores,
        alpha=alpha,
        conf_delta=conf_delta,
        delta_range=(delta_lo, delta_hi),
        k=6,
        family="gaussian",
        kind="SR",
    )
    info = monitor.threshold_info
    print(f"c_alpha={info.c_alpha:.3f} (Ville 1/alpha={info.ville_c_alpha:.3f}), "
          f"order stat k*={info.order_statistic_index}/{len(thresh_scores)}")
    if info.order_statistic_index == len(thresh_scores):
        print("WARNING: order statistic saturated at m -- D_thresh too small to "
              "certify the requested (alpha, delta); c_alpha is just the max "
              "observed nominal statistic, not a real PAC bound.")

    def run_far(stream_list):
        alarms, total_steps = 0, 0
        for s in stream_list:
            m = monitor.fresh_copy()
            for x in s:
                _, alarmed = m.step(float(x))
                if alarmed:
                    alarms += 1
            total_steps += len(s)
        return alarms, total_steps

    far_alarms, far_steps = run_far(test_nominal_scores)
    empirical_far = far_alarms / far_steps if far_steps else float("nan")
    print(f"empirical FAR on held-out nominal: {far_alarms}/{far_steps} = {empirical_far:.4f}")

    delays = []
    detected = 0
    for (traj, meta), s in zip(successful_attacks, attack_scores):
        m = monitor.fresh_copy()
        alarm_at = None
        for t, x in enumerate(s, start=1):
            _, alarmed = m.step(float(x))
            if alarmed:
                alarm_at = t
                break
        if alarm_at is not None:
            detected += 1
            drift_idx = meta.get("drift_index")
            delays.append(alarm_at - drift_idx if drift_idx is not None else alarm_at)
    detection_rate = detected / len(successful_attacks) if successful_attacks else float("nan")
    print(f"detected {detected}/{len(successful_attacks)} successful attacks "
          f"(rate={detection_rate:.2f}); delays={delays}")

    resisted_alarms, resisted_steps = run_far(resisted_scores) if resisted_scores else (0, 0)

    def auroc(nominal_scores, attack_scores_):
        nom_max = [max(s) if s else 0.0 for s in nominal_scores]
        atk_max = [max(s) if s else 0.0 for s in attack_scores_]
        if not nom_max or not atk_max:
            return float("nan")
        wins = sum((a > n_) + 0.5 * (a == n_) for a in atk_max for n_ in nom_max)
        return wins / (len(nom_max) * len(atk_max))

    power_auroc = auroc(test_nominal_scores, attack_scores)
    print(f"surprise-score AUROC (held-out nominal vs successful attacks): {power_auroc:.4f}")

    results = {
        "n_nominal": len(nominal),
        "n_successful_attacks": len(successful_attacks),
        "n_resisted_attacks": len(resisted_attacks),
        "split_sizes": {"train": len(train), "cal": len(cal), "thresh": len(thresh), "test_nominal": len(test_nominal)},
        "alpha": alpha,
        "conf_delta": conf_delta,
        "c_alpha": info.c_alpha,
        "ville_c_alpha": info.ville_c_alpha,
        "order_statistic_saturated": info.order_statistic_index == len(thresh_scores),
        "empirical_far": empirical_far,
        "far_alarms": far_alarms,
        "far_steps": far_steps,
        "detection_rate": detection_rate,
        "detected": detected,
        "n_successful_attacks_eval": len(successful_attacks),
        "detection_delays": delays,
        "resisted_attack_alarms": resisted_alarms,
        "resisted_attack_steps": resisted_steps,
        "surprise_score_auroc": power_auroc,
    }
    out_path = ROOT / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
