# SENTRY: Anytime-Valid E-Process Monitoring of LLM Agent Action Streams

SENTRY is a runtime monitor for LLM-agent action streams. It treats agent
safety as **sequential change detection**: a nonnegative e-process is maintained
over per-action surprise scores and an alarm is raised when the process crosses
a threshold, controlling the average run length (ARL) to false alarm **with no
i.i.d. assumption** on the highly dependent action stream (Ville's inequality +
the e-detector of Shin–Ramdas–Rinaldo). Because the reference is learned from a
small corpus of trusted trajectories, the threshold is calibrated with a
distribution-free **PAC** procedure.

The method design is in [`algorithm.md`](algorithm.md); the background survey is
in [`survey.md`](survey.md); the full paper is in
[`manuscript/`](manuscript/) (targeting *IEEE Transactions on Dependable and
Secure Computing*, Special Issue on Safety, Alignment, and Responsibility of
LLMs).

## Headline results

On real AgentDojo + τ-bench trajectories from a capable agent
(DeepSeek-V4-Flash), monitoring only the black-box trace (no model internals,
no extra LLM call):

| Metric | Value |
|---|---|
| Prompt-injection attempts detected (payload observable in trace) | **99%** at **2%** per-step false-alarm rate |
| All injection attempts detected | 66% |
| Median detection delay | 1 action |
| Attempt-vs-benign AUROC | 0.77 |
| PAC threshold | certified (non-saturated) |

**Generalisation.** The method transfers to a second, unrelated agent
(GPT-4o-mini, AUROC 0.82) but its trace-only signal is *attack-specific*: it
detects imperative-style injections near-perfectly yet fails on InjecAgent's
data-integration payloads (AUROC 0.47) — a boundary we report and explain
rather than hide.

## Repository layout

| path | contents |
|---|---|
| `sentry/` | the algorithm (see table below) |
| `real_data/` | AgentDojo + τ-bench collection harnesses, log→trajectory adapter, evaluation, and committed trajectory corpora (`agentdojo/logs_*`, `tau_bench/logs_*`) |
| `manuscript/` | LaTeX sources + compiled PDFs (Springer and IEEE TDSC versions), figures, bibliography, cover letter, appendix, submission checklist |
| `examples/` | synthetic ARL/FAR + detection-delay validation |
| `tests/` | unit + end-to-end tests |
| `reproduce.sh` | one-shot reproduction of every paper number and figure |

`sentry/` implements [`algorithm.md`](algorithm.md):

| module | algorithm.md section |
|---|---|
| `sentry/baseline.py` | §3 — exponential + conformal baseline increments, mixture over unknown drift magnitude |
| `sentry/detector.py` | §4 Phase 3 — SR/CUSUM e-detector recursions, alarm + restart |
| `sentry/scores.py` | §2 — surprise scores: tool-transition bigram, argument-token novelty, observation instruction-likeness |
| `sentry/calibration.py` | §4 Phases 1-2 — mixture fitting and PAC-threshold order-statistic calibration |
| `sentry/pipeline.py` | Algorithm SENTRY-Detect end-to-end |
| `sentry/streams.py` | synthetic Bernoulli/Gaussian toy streams for validating the ARL guarantee |

## Install and test

```bash
pip install -e .
pytest                                        # unit + end-to-end synthetic validation
python examples/run_synthetic_validation.py   # ARL/FAR + detection-delay plots
```

## Reproducing the paper

The real agent trajectories are committed under `real_data/`, and the
evaluation is deterministic (seed-stable hashing, fixed seeds 0–9), so every
number and figure in the manuscript reproduces **without any API key or
network access**:

```bash
./reproduce.sh          # tests + real-data eval + generalisation + figures + report
```

This writes `real_data/results.json` and `real_data/results_generalization.json`,
regenerates `manuscript/figures/*.pdf`, and prints the manuscript's headline
numbers next to the paper's values (`python -m real_data.report`).

To also re-collect the trajectories from OpenRouter first (needs
`OPENROUTER_API_KEY` in `.env`; costs money; LLM sampling is not
bit-reproducible, so numbers may shift slightly), run `./reproduce.sh --collect`.

### Datasets

All collected from OpenRouter-served agents and committed to the repo:

- **AgentDojo** (banking, workspace, travel, slack) — benign + two attack
  families: `important_instructions` and `injecagent` — on DeepSeek-V4-Flash,
  and `important_instructions` on GPT-4o-mini (cross-agent study).
- **τ-bench** (retail) — benign multi-turn tool use on DeepSeek-V4-Flash.

## Minimal usage

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

The whole pipeline is pure CPU/NumPy and needs no GPU. Replacing the
lightweight reference with a learned causal world model over agent-action
embeddings (§2) is the natural next step and where GPU training would apply.

## Citation

```bibtex
@misc{dang2026sentry,
  title  = {SENTRY: Anytime-Valid E-Process Monitoring of LLM Agent Action Streams},
  author = {Quang-Vinh Dang},
  year   = {2026},
  note   = {Under review, IEEE Transactions on Dependable and Secure Computing},
  url    = {https://github.com/vinhqdang/SENTRY-Anytime-Valid-E-Process-Monitoring-of-LLM-Agent-Action-Streams}
}
```

## License

AGPL-3.0 (see [`LICENSE`](LICENSE)).
