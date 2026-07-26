# SENTRY-Fuse: Conformal Surprisal Fusion for Detecting Prompt-Injection Compromise in LLM Agent Traces

SENTRY-Fuse is a runtime monitor that detects when an LLM agent has been
compromised by an **indirect prompt injection**, using only the agent's black-box
execution trace: no model weights, no activations, no retraining, no agent
re-execution.

Three heterogeneous signals are computed per action, each converted to its
**conformal tail surprisal** against a benign calibration split and then summed:

| signal | channel | what it asks |
|---|---|---|
| **S3** instruction-likeness | observation | does an untrusted tool output read as a directive to the agent? |
| **S4** action-justification audit | action | does this tool call fail to serve the user's actual request? |
| **S5** sensitive-sink provenance | action | did an identifier from untrusted prose reach an externally-effectful call? |

The conformal step is what makes signals on incompatible scales (a $[0,1]$ judge
score, a binary flag) commensurable, so the sum needs **no tuned weights**.

> **Note on history.** This repository began as an application of anytime-valid
> e-processes to agent monitoring, which is where its name comes from. That
> machinery is *no longer the method*: a fixed threshold on the same score
> outperformed it, and we report this as a negative result. The e-detector code
> remains in `sentry/` because the paper's negative-results section is measured
> with it.

## Headline results

Compromise detection over **90 genuinely compromised trajectories** (AgentDojo +
τ-bench, two agents), ten seeds, mean ± std. Benign trajectories are split three
ways per seed — one third fits the reference, one third supplies conformal
calibration values, one third the held-out negatives — so nothing the detector is
scored on was used to build it.

| corpus | detector | AUROC | TPR | realised FPR |
|---|---|---|---|---|
| GPT-4o-mini (72 comp.) | S3 alone | 0.951 ± 0.026 | 0.661 ± 0.433 | 1.2% |
| GPT-4o-mini | **SENTRY-Fuse** | **0.985 ± 0.003** | **0.956 ± 0.006** | 2.9% |
| DeepSeek-V4-Flash (18 comp.) | S3 alone | 0.871 ± 0.014 | 0.622 ± 0.311 | 2.2% |
| DeepSeek-V4-Flash | **SENTRY-Fuse** | **0.843 ± 0.008** | **0.778 ± 0.000** | 3.9% |
| Pooled (90 comp.) | S3 alone | 0.944 ± 0.012 | 0.729 ± 0.364 | 2.5% |
| Pooled | **SENTRY-Fuse** | **0.957 ± 0.004** | **0.919 ± 0.007** | 3.4% |

The gain over the single-signal baseline is largest in **variance**: pooled recall
goes from 0.729 ± 0.364 to 0.919 ± 0.007. A ±0.36 standard deviation means the
detector's recall at a fixed false-alarm budget is close to unpredictable across
calibration draws; ±0.007 is deployable.

We report the *realised* false-positive rate because with `n` held-out negatives
only multiples of `1/n` are attainable, so a nominal target such as 3.66% cannot
be hit exactly and quoting it would misstate the operating point.

**We do not claim to beat the state of the art.** The strongest reported
trace-only figure (AgentArmor, 95.75% TPR at 3.66% FPR) appears only in v1 of that
preprint — later versions withdraw it — was measured on a different corpus with a
different agent, and we re-implemented no baseline here. Our numbers are in the
same regime; that is all the evidence supports.

## Negative results

These are in the repository and the paper because they are the part most likely to
save someone else time.

- **Two behavioural signals we dropped.** Tool-transition surprise and argument
  novelty were the original method. Under an honest three-way split they are
  near-chance (pooled AUROC 0.558 and 0.546; *below* chance on DeepSeek), and
  adding them lowers pooled AUROC from 0.957 to 0.951. Statistical unusualness is
  not a proxy for misalignment.
- **Subtracting the benign maximum destroys signals.** Normalising a score as
  `max(0, x − max(benign))` collapses any class lying under that maximum to a
  single point, provably capping AUROC at ½. It cost us 0.25 AUROC and produced a
  confident, wrong explanation ("the attack is too subtly phrased") for a failure
  that was entirely our own normalisation. The conformal transform replaces it.
- **AgentDojo's `security` flag is easy to invert.** `security=True` means the
  *injection* succeeded, i.e. the agent **was** compromised. We had it backwards,
  which silently graded every effect-based signal against trajectories where the
  agent had done nothing harmful. `real_data/evaluate.py` documents the
  in-trace ground-truth check that settles it.
- **The trace-only detectability ceiling is vacuous on AgentDojo.** We prove that
  the fraction of compromises leaving no trace evidence bounds *every* trace-only
  monitor, then measure that fraction: it is **0**, because AgentDojo delivers
  payloads through tool outputs a successful injection must have read. So the
  ceiling is an artefact of benchmark construction, and the ~8% we still miss is
  scoring error, not absent evidence.
- **The anytime-valid machinery did not earn its place.** A fixed benign quantile
  outperformed the PAC-calibrated e-detector this project started from.

## Repository layout

| path | contents |
|---|---|
| `sentry/` | the algorithm (see table below) |
| `real_data/` | AgentDojo + τ-bench collection harnesses, log→trajectory adapter, evaluation, and committed trajectory corpora (`agentdojo/logs_*`, `tau_bench/logs_*`) |
| `manuscript/` | LaTeX source, compiled PDF, figures, bibliography, and submission documents |
| `examples/` | synthetic ARL/FAR + detection-delay validation |
| `tests/` | unit + end-to-end tests |
| `reproduce.sh` | one-shot reproduction of every paper number and figure |

`sentry/` contains the scoring and (now-superseded) e-detector machinery; the
fusion monitor evaluated in the paper lives in `real_data/evaluate_fused.py`.

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

The agent trajectories and the cached model-judge scores are committed under
`real_data/`, and the evaluation is deterministic (seed-stable hashing, fixed
seeds 0–9), so every number, table and figure in the manuscript reproduces
**without any API key or network access**:

```bash
./reproduce.sh
```

This runs the tests, then every table's generator — compromise detection and the
signal ablation, the evidence-channel measurement, the normalisation ablation,
attempt detection and generalisation, and the long-horizon e-detector comparison —
regenerates `manuscript/figures/*.pdf`, and finally runs
`python -m real_data.report`, which checks each reproduced value against the number
printed in the paper and exits non-zero on any mismatch.

To also re-collect the trajectories from OpenRouter first (needs
`OPENROUTER_API_KEY` in `.env`; costs money; LLM sampling is not
bit-reproducible, so numbers may shift slightly), run `./reproduce.sh --collect`.

### Datasets

All collected from OpenRouter-served agents and committed to the repo:

- **AgentDojo** (banking, workspace, travel, slack) — benign + two attack
  families: `important_instructions` and `injecagent` — on DeepSeek-V4-Flash,
  and `important_instructions` on GPT-4o-mini (cross-agent study).
- **τ-bench** (retail) — benign multi-turn tool use on DeepSeek-V4-Flash.

A partial τ-bench *airline* collection (11 of 50 tasks) is also committed but is
**not used in any evaluation** — it is too small to form a meaningful benign
stratum. It is left in place so the released data matches what was collected.

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

The whole pipeline is pure CPU/NumPy and needs no GPU. The two semantic signals
(S3, S4) each need one small-model call per distinct observation and per distinct
action; these are cached by content hash and committed, so re-evaluation makes no
API calls at all.

## Citation

```bibtex
@misc{dang2026sentryfuse,
  title  = {Conformal Surprisal Fusion for Detecting Prompt-Injection Compromise
            in LLM Agent Traces},
  author = {Quang-Vinh Dang},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/vinhqdang/SENTRY-Anytime-Valid-E-Process-Monitoring-of-LLM-Agent-Action-Streams}
}
```

## License

AGPL-3.0 (see [`LICENSE`](LICENSE)).
