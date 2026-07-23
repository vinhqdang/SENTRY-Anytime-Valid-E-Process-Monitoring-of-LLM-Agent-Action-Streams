# SENTRY-Anytime-Valid-E-Process-Monitoring-of-LLM-Agent-Action-Streams

SENTRY continuously monitors an LLM agent's action stream for behavioral
drift (prompt injection, goal drift, reward hacking, tool misuse), with a
nonasymptotic ARL/false-alarm guarantee that holds without any i.i.d.
assumption on the stream. The design is in [`algorithm.md`](algorithm.md);
[`survey.md`](survey.md) is the background literature survey.

## Implementation

`sentry/` implements the algorithm described in `algorithm.md`:

| module | algorithm.md section |
|---|---|
| `sentry/baseline.py` | §3 — exponential + conformal baseline increments, mixture over unknown drift magnitude |
| `sentry/detector.py` | §4 Phase 3 — SR/CUSUM e-detector recursions, alarm + restart |
| `sentry/scores.py` | §2 — causal-world-model surprise score and intervention-consistency score (a lightweight, learnable stand-in for a full CAIRN-style causal world model; see the module docstring) |
| `sentry/calibration.py` | §4 Phases 1-2 — mixture fitting and PAC-threshold order-statistic calibration |
| `sentry/pipeline.py` | Algorithm SENTRY-Detect end-to-end |
| `sentry/streams.py` | §8 next steps — synthetic Bernoulli/Gaussian toy streams for validating the ARL guarantee before wiring in a real agent harness |

```bash
pip install -e .
pytest                                   # unit + end-to-end synthetic validation
python examples/run_synthetic_validation.py   # ARL/FAR + detection-delay plots
```

### Reproducing the paper

The real agent trajectories are committed under `real_data/`, and the
evaluation is deterministic (seed-stable hashing, fixed seeds 0–9), so every
number and figure in the manuscript reproduces **without any API key or
network access**:

```bash
./reproduce.sh          # tests + real-data eval + generalisation + figures + report
```

This writes `real_data/results.json` and `real_data/results_generalization.json`,
regenerates `manuscript/figures/*.pdf`, and prints the manuscript's headline
numbers next to the paper's values (`python -m real_data.report`). To also
re-collect the trajectories from OpenRouter first (needs `OPENROUTER_API_KEY`
in `.env`; costs money; LLM sampling is not bit-reproducible), run
`./reproduce.sh --collect`.

Minimal usage:

```python
from sentry.pipeline import SentryDetect
from sentry.streams import gaussian_stream
import numpy as np

rng = np.random.default_rng(0)
d_cal = [gaussian_stream(300, rng).tolist() for _ in range(100)]
d_thresh = [gaussian_stream(300, rng).tolist() for _ in range(150)]

monitor = SentryDetect.calibrate(
    d_cal_streams=d_cal, d_thresh_streams=d_thresh,
    alpha=0.05, conf_delta=0.1, delta_range=(0.5, 3.0),
)

for x in gaussian_stream(1000, rng, changepoint=400, delta=2.0):
    value, alarmed = monitor.step(float(x))
    if alarmed:
        print("ALARM")
```

This synthetic-stream path is pure CPU/NumPy and needs no GPU. Wiring in a
real causal world model over LLM agent action embeddings (§2) is the
natural next step and would be where GPU training applies.