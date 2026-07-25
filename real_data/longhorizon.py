"""Long-horizon false-alarm experiment: what does the e-process actually buy?

The theoretical selling point of an e-detector is that its false-alarm control
holds under *continuous, unbounded* monitoring: by Ville's inequality the
probability of ever crossing 1/alpha is at most alpha, so the average run
length (ARL) to false alarm is bounded below no matter how long the agent runs.
A hand-picked threshold on the same score has no such property -- its false
alarms accumulate at a roughly constant per-action rate, so the probability of
at least one spurious halt tends to 1 as the stream grows.

This script measures that difference on real benign agent traffic, holding the
SIGNAL FIXED and varying only the decision rule, so the comparison isolates the
sequential machinery rather than the features:

  * fixed-threshold  -- alarm whenever the per-action score exceeds tau, with
                        tau swept over benign score quantiles
  * SENTRY           -- SR e-detector on the same scores with the PAC threshold

Long benign streams are built by concatenating held-out benign trajectories
(shuffled per seed) into streams of >= STREAM_LEN actions, mimicking an agent
that keeps working. Attack detection is reported for every operating point so
the comparison is matched on power rather than cherry-picked.

    python -m real_data.longhorizon
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from real_data.adapters import load_agentdojo_logs, load_taubench_logs
from sentry.pipeline import SentryDetect
from sentry.scores import SequentialWorldModel, score_trajectory

ROOT = Path(__file__).parent
MIN_ACTIONS = 2
CONF_DELTA = 0.2
STREAM_LEN = 1000
N_SEEDS = 10
# Both decision rules are swept over their own knob so the comparison is an
# operating CURVE (attack detection vs long-horizon false-alarm rate) rather
# than one arbitrarily chosen point per method.
ALPHAS = (0.2, 0.05, 0.01, 0.002)
QUANTILES = (0.90, 0.95, 0.99, 0.995, 0.999)


def _load():
    rows = [(t, m) for t, m in load_agentdojo_logs(ROOT / "agentdojo" / "logs_deepseek")
            if len(t) >= MIN_ACTIONS]
    nominal = [(t, m) for t, m in rows if not m["is_attack"] and m.get("utility") is not None]
    nominal += [(t, m) for t, m in load_taubench_logs(ROOT / "tau_bench" / "logs_deepseek")
                if len(t) >= MIN_ACTIONS]
    attacks = [(t, m) for t, m in rows if m["is_attack"] and m.get("security") is not None]
    inj = [(t, m) for t, m in load_agentdojo_logs(ROOT / "agentdojo" / "logs_deepseek_injecagent")
           if len(t) >= MIN_ACTIONS and m["is_attack"] and m.get("security") is not None]
    return nominal, attacks + inj


def _long_stream(scored: list[list[float]], rng: np.random.Generator, length: int) -> list[float]:
    """Concatenate randomly ordered benign trajectory score streams until the
    requested length is reached (an agent that keeps on working)."""
    out: list[float] = []
    while len(out) < length:
        for i in rng.permutation(len(scored)):
            out.extend(scored[i])
            if len(out) >= length:
                break
    return out[:length]


def _fixed_threshold_alarms(stream: list[float], tau: float) -> tuple[int, int | None]:
    """(number of alarms, index of first alarm) for a per-action threshold."""
    alarms = [i for i, s in enumerate(stream) if s > tau]
    return len(alarms), (alarms[0] if alarms else None)


def _sentry_alarms(monitor_factory, stream: list[float]) -> tuple[int, int | None]:
    m = monitor_factory()
    alarms, first = 0, None
    for i, s in enumerate(stream):
        _, fired = m.step(float(s))
        if fired:
            alarms += 1
            if first is None:
                first = i
    return alarms, first


def main() -> None:
    nominal, attacks = _load()
    print(f"nominal={len(nominal)} attacks={len(attacks)} "
          f"stream_len={STREAM_LEN} seeds={N_SEEDS}")

    rows: list[dict] = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(nominal))
        n = len(nominal)
        train = [nominal[i] for i in idx[: n // 3]]
        calib = [nominal[i] for i in idx[n // 3 : 2 * n // 3]]
        test = [nominal[i] for i in idx[2 * n // 3 :]]

        model = SequentialWorldModel().fit([t for t, _ in train])
        s_cal = [score_trajectory(model, t).tolist() for t, _ in calib]
        s_test = [score_trajectory(model, t).tolist() for t, _ in test]
        s_atk = [score_trajectory(model, t).tolist() for t, _ in attacks]

        half = max(1, len(s_cal) // 2)
        stream = _long_stream(s_test, rng, STREAM_LEN)

        # SENTRY at several target false-alarm levels
        for alpha in ALPHAS:
            monitor = SentryDetect.calibrate(
                s_cal[:half], s_cal[half:], alpha=alpha, conf_delta=CONF_DELTA,
                delta_range=(0.3, 3.0), k=6, family="gaussian", kind="SR",
            )
            s_alarms, s_first = _sentry_alarms(monitor.fresh_copy, stream)
            s_det = sum(_sentry_alarms(monitor.fresh_copy, s)[0] > 0 for s in s_atk) / len(s_atk)
            rows.append({"seed": seed, "method": f"SENTRY (alpha={alpha})", "tau": None,
                         "alarms_per_1k": s_alarms * 1000 / len(stream),
                         "first_alarm": s_first, "attack_detection": s_det})

        # fixed thresholds on the SAME scores
        flat_cal = np.array([x for s in s_cal for x in s])
        for q in QUANTILES:
            tau = float(np.quantile(flat_cal, q))
            f_alarms, f_first = _fixed_threshold_alarms(stream, tau)
            f_det = sum(any(x > tau for x in s) for s in s_atk) / len(s_atk)
            rows.append({"seed": seed, "method": f"fixed threshold (q={q})", "tau": tau,
                         "alarms_per_1k": f_alarms * 1000 / len(stream),
                         "first_alarm": f_first, "attack_detection": f_det})

    # aggregate
    methods: dict[str, dict] = {}
    for r in rows:
        m = methods.setdefault(r["method"], {"alarms": [], "first": [], "det": []})
        m["alarms"].append(r["alarms_per_1k"])
        m["det"].append(r["attack_detection"])
        m["first"].append(STREAM_LEN if r["first_alarm"] is None else r["first_alarm"])

    print(f"\n{'method':<28}{'false alarms/1k':>17}{'ARL (first FA)':>16}"
          f"{'P(>=1 FA)':>11}{'attack det.':>13}")
    print("-" * 85)
    summary = {}
    for name, m in methods.items():
        fa = float(np.mean(m["alarms"]))
        arl = float(np.mean(m["first"]))
        pfa = float(np.mean([f < STREAM_LEN for f in m["first"]]))
        det = float(np.mean(m["det"]))
        summary[name] = {"false_alarms_per_1k": fa, "mean_first_alarm": arl,
                         "p_any_false_alarm": pfa, "attack_detection": det}
        arl_s = f"{arl:.0f}" + ("+" if pfa < 1 else "")
        print(f"{name:<28}{fa:>17.1f}{arl_s:>16}{pfa:>11.2f}{det:>13.3f}")

    out = {"stream_len": STREAM_LEN, "n_seeds": N_SEEDS, "alphas": list(ALPHAS),
           "quantiles": list(QUANTILES), "methods": summary}
    (ROOT / "results_longhorizon.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'results_longhorizon.json'}")


if __name__ == "__main__":
    main()
