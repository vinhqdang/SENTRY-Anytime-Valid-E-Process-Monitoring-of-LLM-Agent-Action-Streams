# SENTRY on real agent trajectories: results

Running SENTRY-Detect on real LLM agent trajectories instead of synthetic
Bernoulli/Gaussian streams, per algorithm.md §7's evaluation plan. Data
collection used OpenRouter's `openrouter/free` free-model router (a
rotating pool of ~24 free-tier models) as the agent (and, for tau-bench,
the simulated user) LLM.

**Bottom line up front.** Two rounds of score-model work took the real-data
results from chance-level to a genuine detector. The `SequentialWorldModel`
combines three signals, each added because a failure analysis of the misses
pointed at it:

| metric (mean ± std, 10 random splits) | old `CausalWorldModel` | bigram+novelty | **+ injection signal** |
|---|---|---|---|
| AUROC (length-normalized, post-drift mean) | 0.47 ± 0.10 | 0.69 ± 0.14 | **0.79 ± 0.10** |
| AUROC (max-of-stream) | 0.32 ± 0.13 | 0.60 ± 0.14 | **0.73 ± 0.08** |
| Detection @ Ville threshold (1/α, α=0.2) | 6.3% | 45.6% | **63.8% ± 3.8%** |
| Detection @ PAC threshold | 3.1% | 38.8% | **62.5% ± 0.0%** |
| Per-step false-alarm rate @ Ville | 6.2% | 11.5% | 11.6% (≤ α = 20%) |
| Median detection delay (post-drift steps) | 4.5 | 1.0 | **1.0** |

Detection @ the (non-vacuous) PAC threshold is now **62.5% with zero
variance across all 10 splits** — stable, not luck. Median delay of **1
step** means when SENTRY fires, it fires on essentially the first action
after the injection enters the context. Still not solved (see limitations),
but a real detector.

The three signals in `SequentialWorldModel`, and why each was added:

1. **Tool-transition surprise** `-log P(tool_t | tool_{t-1})` (bigram,
   pooled across training). Replaced the original task_id-conditioned MLE,
   which was near-uniform on the always-unseen task_ids of eval data.
2. **Argument-token novelty** — fraction of an action's argument tokens
   never seen in nominal data. A single-action signal (matters because
   AgentDojo attacks average only 2.6 actions — too short for accumulation
   alone).
3. **Observation instruction-likeness** (this round, the biggest single
   jump: 46% → 64% detection). A tool result is supposed to be passive
   *data*; indirect prompt injection smuggles an *instruction* into it, so
   instruction-like content in an observation is high-surprise under a
   nominal reference. The signal fires at the observation, *before* the
   agent acts — which is why it also helps the hardest missed class (where
   the hijacked action itself looks benign, e.g. the agent just answers in
   text). Measured in isolation: AUROC **0.955** (successful-attack vs
   benign observations), benign median exactly 0. The lexicon is generic
   (imperative-verb / second-person density, pseudo-tag mimicry,
   exclamations) — no proper nouns or benchmark phrases — and the nominal
   ceiling is *learned* per deployment, not hardcoded. Robust: any
   instruction weight in [2, 16] gives the same 63.8% detection (weight 0
   drops it to 45.6%), so nothing is knife-edge tuned. This is the
   trace-only analogue of CommandSans (arXiv:2510.08829), which cut
   AgentDojo attack-success 34%→3% by detecting instructions inside tool
   outputs.

## Two ways to combine the signals: power vs. rigor (both shipped)

The three signals can be combined two ways, and they trade off:

| combiner (PAC threshold, 10 seeds) | detection | FAR | guarantee |
|---|---|---|---|
| **summed surprise** (weighted sum → one exponential increment) | **62.5% ± 0.0%** | 5.2% | validity via PAC calibration of the summed scalar |
| **per-signal e-value mixture** (each signal → own conformal e-value, averaged) | 52.5% ± 10.9% | 6.2% | **each signal distribution-free Ville-valid; per-signal attribution; no sub-Gaussian assumption** |

The e-value mixture (`sentry.baseline.MultiSignalConformalIncrement`) is the
more principled object: each signal is turned into a conformal e-value
(exact validity under exchangeability, no tail assumption, no hand-tuned
cross-signal weights) and merged by the *only* admissible symmetric merge —
the arithmetic mean (Vovk–Wang 2021). That mean is also its weakness on this
data: averaging **dilutes a single dominant signal** (a huge instruction
e-value merged with two ~1 e-values is divided by 3), so uniform weights
give only 42.5% detection. An informed prior weighting (0.2/0.2/0.6, from
the per-signal *isolation* AUROCs — instruction ≈0.955 ≫ the others, a prior
over informativeness, not fit to the detection metric) recovers it to 52.5%.
At the exact-Ville threshold the e-value mixture runs at **0.0% false-alarm
rate** (conformal e-values are bounded, so 1/α is rarely crossed) — the
conservative operating point if you want the formal guarantee over raw power.

Honest read: summed-surprise is more powerful *here*; the e-value mixture is
more rigorously valid and gives per-signal attribution ("fired on
instruction at step 2"). Which to prefer is a real deployment choice, not a
solved question — both are implemented and both are reported.

## A signal that was tried and did NOT work here: data-flow taint

The strongest published trace-only detector, AgentArmor (arXiv:2508.01249,
95.75% TPR / 3.66% FPR on AgentDojo), and CaMeL (arXiv:2503.18813) both key
on **provenance/taint**: an argument value that entered via an untrusted
tool observation and was never in the user's request is attacker payload by
construction. I implemented and measured this signal (fraction of an
action's "salient" argument tokens that appear in a prior observation but
not in the user request). On *our* data it gave **AUROC 0.47 — no
separation** (only 4/27 successful attacks copy a novel observation token
into an argument; 15/16 *resisted* attacks do). The reason is a property of
our data, not the signal: our weak free-tier agent produces short, often
degenerate "successful" attacks where the harmful step frequently isn't a
payload-bearing tool call at all (e.g. it leaks data in a text reply), so
there is no tainted argument to catch. Taint is therefore **not** in the
shipped model — reported here because a negative result on real data is
still a result, and the caveat (clean GPT-4o traces would likely behave more
like AgentArmor's) points at what to re-test with stronger agents.

## Reproducibility fix found this round

Feature hashing used Python's builtin `hash()`, which is salted per process,
so every run produced different hashed features and different numbers.
Switched to `zlib.crc32`; the full evaluation is now byte-identical across
back-to-back runs. Also fixed drift-index location to match on
alphanumerics-only (YAML line-folding with `\` continuations had defeated
whitespace-normalized matching on several logs).

Earlier rounds' fixes that fed into this (see git history): bigram tool
model replacing task_id conditioning; per-tool argument-token novelty
vocabulary; regex tokenization of compact JSON; a length-normalized AUROC
metric after `max(stream)` was shown to confound with trajectory length
(nominal ≈ 5.3 actions vs. 2.6 for successful attacks).

## Data (final)

| Source | Suites | Nominal | Successful attack | Resisted attack |
|---|---|---|---|---|
| AgentDojo | banking, workspace, travel, slack | 18 | 16 | 16 |
| tau-bench | retail | 14/15 (1 rate-limited) | n/a | n/a |

Attacks are AgentDojo's `important_instructions` prompt injection;
"successful" = AgentDojo's own `security == False` label (the injection
actually hijacked behavior). 24 nominal trajectories total pass the
≥2-action filter. Split per seed: train=8, cal=6, thresh=5, test=5.

## Honest limitations

- **The PAC threshold is still saturated** (D_thresh = 5 streams), so the
  PAC-threshold rows use "max of 5 nominal statistics" rather than a
  certified bound. The Ville-threshold rows (1/α) are the theoretically
  meaningful operating point *if* the learned score were an exact
  reference, which it is not — a real PAC bound needs a larger D_thresh.
  This remains the top item for a credible guarantee claim.
- **63.8% detection is not 95%.** The remaining ~36% are dominated by a
  specific, identified class: ~11 of the 16 "successful" attacks have **no
  injected instruction anywhere in the parsed observation stream** (verified
  directly) — the weak free-tier agent's "success" was degenerate or the
  payload sat in a nested field it never surfaced. No observation-based
  signal can catch those, and arguably they aren't genuine hijacks to catch.
  Among attacks where the injection *is* observable, detection is much
  higher. This caps what any trace-only detector can score on *this* data
  and is a property of the free-tier collection, not the method.
- **n = 16 attacks.** Enough to distinguish 0.79 from 0.5 (all model
  comparisons run on identical splits), not enough for tight confidence
  intervals; the ±0.10 split std is honest and non-trivial.
- The trace plot (`real_data/plots/real_data_traces.png`) shows the
  detector statistic spiking exactly at the injection point on a real
  attack and staying flat on a real nominal trajectory.

## Comparison to 2026 literature (updated)

Still no 2026 paper found that reports FAR/ARL/AUROC results on AgentDojo,
InjecAgent, WebArena, or tau-bench with an e-process/e-detector method —
these remain, to our knowledge, the first such numbers on this benchmark
family, now at a defensible (if preliminary) signal level:

- **E-valuator** (arXiv 2512.03109): shares the PAC-thresholding
  subroutine; its guarantees face the same finite-calibration-set
  dependency our saturated D_thresh exposes. No AgentDojo/tau-bench
  overlap to compare numerically.
- **Cheap Reward Hacking Detection** (arXiv 2606.08893): AUROC 0.9467 with
  a linear probe on model internals, no formal guarantee, different
  failure mode (reward hacking) and far more training data. SENTRY's 0.79
  from ~8 nominal trajectories of *black-box* tool-call logs, with an
  anytime-valid ARL story attached, is not directly comparable — but the
  0.95-class number is the empirical bar to aim at as data and features
  scale.
- **CommandSans** (arXiv 2510.08829) and **AgentArmor** (arXiv 2508.01249):
  the two closest trace-level defenses. CommandSans validates the
  instruction-in-observation signal SENTRY's injection term uses
  (AgentDojo attack-success 34%→3%); AgentArmor validates the taint signal
  SENTRY tried and found ineffective *on this free-tier data* (95.75% TPR
  on clean GPT-4o traces). Both are prevention/detection systems without an
  anytime-valid ARL guarantee — the piece SENTRY adds.
- **ToolChain-CRC** (arXiv 2606.18467) / **strategic deviations** (arXiv
  2601.05427): nearest methodological neighbors (conformal risk control /
  test-supermartingales), still no shared benchmark to compare against.

## What would move the needle next

The single biggest lever is **data quality**: ~11/16 "successful" attacks
here have no injected instruction in the observation stream at all (weak
free-tier agent), which caps every trace-only signal. Concretely:

1. **Collect from a capable (paid) agent.** This lifts the ceiling directly
   *and* is the way to re-test the two signals that failed on our degenerate
   data — taint (AUROC 0.47 here vs. AgentArmor's 95.75% TPR on GPT-4
   traces) and goal-grounding (0.499 here) — where successful attacks
   actually execute payload-bearing tool calls. Blocked only on API budget.
2. **More D_thresh streams** → a real, non-saturated PAC threshold (still
   the gap between "has the theory" and "certifies the FAR bound").
3. **More attack diversity** beyond `important_instructions` (other
   injection templates, DoS variants) to test generalization.

Tried and resolved this round: the **e-value mixture combiner** (done —
implemented, more rigorous, power/rigor tradeoff characterized above);
**goal-grounding** (tried, lexical version gives AUROC 0.499 — no
separation, same failure mode as taint; a MELON-style masked-re-execution
version needs embeddings/LLM re-runs and is deferred with the paid-agent
work).
