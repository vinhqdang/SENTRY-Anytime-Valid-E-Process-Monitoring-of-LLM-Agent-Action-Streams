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

A third signal, **S5** sensitive-sink provenance (did an identifier from untrusted
prose reach an externally-effectful call?), is implemented and evaluated but **not
part of the method**: its contribution does not survive a trajectory bootstrap
(see Negative results).

The conformal step is what makes signals on incompatible scales (a $[0,1]$ judge
score, a binary flag) commensurable, so the sum needs **no tuned weights**.

> **Note on history.** This repository began as an application of anytime-valid
> e-processes to agent monitoring, which is where its name comes from. That
> machinery is *no longer the method*: a fixed threshold on the same score
> outperformed it, and we report this as a negative result. The e-detector code
> remains in `sentry/` because the paper's negative-results section is measured
> with it.

## Headline results

Compromise detection over **90 genuinely compromised trajectories** (AgentDojo,
two agents), ten seeds, mean ± std. Benign trajectories are split three
ways per seed — one third fits the reference, one third supplies conformal
calibration values, one third the held-out negatives — The operating point is read off the
evaluation negatives, so the reported false-positive rate is descriptive of them
rather than an out-of-sample guarantee; `results_fused.json` also carries a
four-way split in which the threshold is fitted on a disjoint fourth part.

| corpus | detector | AUROC | TPR@5% | realised FPR |
|---|---|---|---|---|
| GPT-4o-mini (72 comp.) | S3 alone | 0.951 ± 0.026 | 0.661 ± 0.433 | 1.2% |
| GPT-4o-mini | **SENTRY-Fuse** | **0.981 ± 0.009** | **0.914 ± 0.040** | 2.5% |
| DeepSeek-V4-Flash (18 comp.) | S3 alone | 0.850 ± 0.021 | 0.467 ± 0.381 | 2.4% |
| DeepSeek-V4-Flash | **SENTRY-Fuse** | **0.842 ± 0.006** | **0.778 ± 0.000** | 4.0% |
| Pooled (90 comp.) | S3 alone | 0.937 ± 0.012 | 0.709 ± 0.357 | 2.8% |
| Pooled | **SENTRY-Fuse** | **0.953 ± 0.005** | **0.898 ± 0.020** | 3.8% |

The gain over the single-signal baseline is largest in **variance**: pooled recall
goes from 0.709 ± 0.357 to 0.898 ± 0.020, and the baseline collapses to *exactly zero*
recall in 2 of 10 splits. A ±0.36 standard deviation means the detector's recall at
a fixed false-alarm budget is close to unpredictable across calibration draws;
±0.020 is deployable.

Resampling trajectories (2000 replicates), the recall gain over the single-signal
baseline is **+0.1706, 95% CI [+0.0033, +0.3533]** and the AUROC gain **+0.0163, 95%
CI [-0.0006, +0.0351]**. The AUROC interval only marginally excludes no effect, and
90 positives cannot resolve a difference of that size, so we do not present the
AUROC gain as established — the stability result above is the claim we rest on.

We report bootstrap intervals rather than a t-statistic over re-splits, because the
splits permute one fixed sample and so describe calibration-draw variance rather
than sampling error. One resample is drawn per replicate and reused across the ten
splits; resampling independently inside each split and averaging estimates the
variance of a mean of ten draws and shrinks the corpus-sampling variance by roughly
a factor of ten, which turns this borderline effect into an apparently decisive one.

We report the *realised* false-positive rate because with `n` held-out negatives
only multiples of `1/n` are attainable, so a nominal target such as 3.66% cannot
be hit exactly and quoting it would misstate the operating point.

### Against a re-implemented baseline

Comparing our own signal subsets is a self-ablation, so we re-implemented **Task
Shield** — the closest prior work to our action-side signal — and ran it on the same
trajectories, with the same splits and the same judge model, so the comparison
isolates the method rather than the defender (`real_data/baseline_taskshield.py`).

| detector | AUROC | TPR | FPR |
|---|---|---|---|
| Task Shield, published rule (flag if any call misaligned) | — | 0.933 | 39.0% |
| Task Shield, ranked by minimum alignment score | 0.764 | 0.000 | 0.0% |
| **SENTRY-Fuse** | **0.953** | **0.898** | 3.8% |

Task Shield's own rule *catches more compromises than we do*. It buys that with a
39% false-alarm rate, which is the real difference between the two: it was designed
as an inline defence, where a false alarm costs a retry, not as a monitor, where it
costs an analyst.

The 0.000 in its second row is **an artefact, not a defeat**. Its aggregate score is
a sum of {0, 0.5, 1} judgments, so the trajectory score takes only 7 distinct values
and 39% of benign traces tie at the floor. A 5% budget puts the threshold below that
tied block and nothing is flagged. No threshold on that score yields a false-alarm
rate strictly between 0 and 39% — a real limitation for monitoring use, but one of
resolution rather than of ranking.

This is **not** a verdict on Task Shield as a defence: it blocks the calls it flags
and we only raise a flag, and our traces contain no assistant prose, so one of its
two checking layers has no counterpart here.

**We do not claim to beat the state of the art.** The strongest reported
trace-only figure (AgentArmor, 95.75% TPR at 3.66% FPR) appears only in v1 of that
preprint — later versions withdraw it — and was measured on a different corpus with
a different agent. Our numbers are in the same regime; that is all the evidence
supports.

## Negative results

These are in the repository and the paper because they are the part most likely to
save someone else time.

- **Two behavioural signals we dropped.** Tool-transition surprise and argument
  novelty were the original method. Under an honest three-way split they are
  near-chance (pooled AUROC 0.526 and 0.602; *below* chance on DeepSeek), and
  adding them lowers pooled AUROC from 0.953 to 0.948. Statistical unusualness is
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
- **We withdrew an information-theoretic ceiling rather than defend it.** An
  earlier version proved that the fraction of compromises leaving no trace evidence
  bounds every trace-only monitor, and reported that fraction as 0. It is not
  identified: three defensible text-matching criteria give 0.000, 0.044 and 0.089,
  and at the strictest the implied bound (0.916) falls *below* the true-positive
  rate we measure. The paper now reports the underlying channel coverage
  descriptively and draws only the benchmark-design conclusion — AgentDojo delivers
  payloads through tool outputs a successful injection must have read, so corpora of
  this shape cannot contain a compromise invisible to a trace monitor, and cannot
  test the hard case.
- **A third signal did not earn inclusion.** Sink provenance is well motivated and
  precise, but adding it moves pooled AUROC by +0.0004, 95% CI [-0.0043, +0.0056] — an
  interval containing zero — and recovers fewer than two extra positives out of 90.
  It is reported as an ablation, not as part of the method.
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

The monitor evaluated in the paper is `real_data/evaluate_fused.py`. The `sentry/`
package holds the per-action scoring code it calls, plus the e-detector and
PAC-calibration machinery from the project's original design — retained because the
paper's negative-results section is measured with it, not because it is the method.

| module | contents |
|---|---|
| `sentry/scores.py` | per-action signals: tool-transition bigram, argument-token novelty, observation instruction-likeness, and the conformal transform |
| `sentry/baseline.py` | exponential + conformal baseline increments, mixture over unknown drift magnitude *(superseded)* |
| `sentry/detector.py` | SR/CUSUM e-detector recursions, alarm + restart *(superseded)* |
| `sentry/calibration.py` | mixture fitting and PAC-threshold order-statistic calibration *(superseded)* |
| `sentry/pipeline.py` | the original end-to-end e-process monitor *(superseded)* |
| `sentry/streams.py` | synthetic Bernoulli/Gaussian streams for validating the ARL guarantee |
| `sentry/llm_judge.py` | the two cached model judges behind S3 and S4 |

`algorithm.md` and `survey.md` are the original design and literature documents;
both carry a banner noting they describe the superseded method.

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
signal ablation, the evidence-channel measurement, the normalisation ablation, the
re-implemented Task Shield baseline, attempt detection and generalisation, and the
long-horizon e-detector comparison — regenerates `manuscript/figures/*.pdf`, and
finally runs
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
