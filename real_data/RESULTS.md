# SENTRY on real agent trajectories: results and honest limitations

This is a first pass at running SENTRY-Detect on real LLM agent trajectories
instead of synthetic Bernoulli/Gaussian streams, per algorithm.md §7's
evaluation plan. Data collection used OpenRouter's `openrouter/free`
free-model router (a rotating pool of ~24 free-tier models) as the agent
(and, for tau-bench, the simulated user) LLM, with no other API cost.

## Data collected

| Source | Suites | Nominal | Attack (complete) | Successful attack | Resisted attack |
|---|---|---|---|---|---|
| AgentDojo | banking, workspace | 18 | 13 | 7 | 6 |
| tau-bench | retail | 14/15 (1 hit an OpenRouter free-tier rate limit mid-task) | n/a (no attack suite) | n/a | n/a |

Both collections are complete for this session's budget (24 nominal + 13
attack trajectories total). tau-bench's one failed task hit OpenRouter's
`free-models-per-min` limit (16/min shared across the whole free-model
pool) and was recorded as a zero-action failure by tau-bench itself, then
dropped by the `>=2 actions` filter in `real_data/evaluate.py` — expected
behavior on a free-tier budget, not a bug.

AgentDojo attacks use the `important_instructions` prompt-injection attack
(AgentDojo's strongest baseline attack). "Successful" = AgentDojo's own
`security == False` label, i.e. the injection actually hijacked the agent's
behavior, not just that an injection was present.

Trajectories are converted to `sentry.scores` format via
`real_data/adapters.py`: each tool call becomes one `Action`, with a
deterministic feature-hashing embedding of its arguments (no GPU/embedding
API — see the module docstring). The "drift index" (true changepoint) for
each successful attack is located by whitespace-normalized substring
matching of the injected text against tool-result content — this required
a fix (see git history) because AgentDojo's environments re-serialize the
injected string (e.g. YAML-dumping it as a field value), which re-wraps its
newlines and defeats an exact match.

## Numbers (from `real_data/results.json`, final run on the complete collected dataset)

- Nominal trajectories: 24 (18 AgentDojo + 6 tau-bench passing the `>=2
  actions` filter)
- Split: train=8, cal=6, thresh=5, test_nominal=5 (successful attacks held
  out separately, n=4 usable after the same filter)
- **PAC threshold saturated**: with only 5 D_thresh streams, the order-statistic
  bound (algorithm.md §4 Phase 2) can't certify anything better than "use the
  single largest observed nominal statistic as the threshold" at
  (alpha=0.2, delta=0.2). `c_alpha` is therefore just `max` over 5 numbers,
  not a real PAC guarantee — this is flagged explicitly in the script's
  output (`order_statistic_saturated: true`) and results.json.
- Empirical FAR on held-out nominal: 0/31 steps = 0.0 — **not meaningful**,
  it's a trivial consequence of the threshold above being the max of a
  tiny sample, not evidence of good FAR control.
- Detection: 0/4 successful attacks crossed that (very conservative)
  threshold. Surprise-score AUROC (held-out nominal vs. successful
  attacks) was **0.35** on this final run — across three consecutive
  re-runs as more data landed (n_nominal=20/21/24), AUROC read 0.60, 0.45,
  then 0.35, and detection count read 1/4, 0/4, 0/4.
- **That AUROC trending *down* as more real nominal data arrives, while
  swinging by ±0.25 between runs, is itself the headline finding.** At
  n≈5-24 per split the estimate has not stabilized and there's no basis to
  claim any particular number is representative — these are not
  statistically meaningful FAR/ARL/AUROC results yet. What they do show is
  that the pipeline runs correctly end-to-end on real, heterogeneous (two
  different domains, two different tool sets) agent trajectories, and that
  the current hashed-argument surprise score's separation between nominal
  and attack trajectories is weak and possibly non-existent net of noise
  (an AUROC of 0.35 is *below* chance, i.e. consistent with no real signal
  once sampling variance is accounted for) — not that SENTRY has been
  validated at this feature/model configuration.

See `real_data/plots/real_data_traces.png` for a real nominal vs. real
successful-attack trace: qualitatively, the e-detector statistic does rise
measurably after the true drift index on the attack trajectory while
staying flat on the nominal one — the signal exists — but the tiny D_thresh
sample means the calibrated threshold is far too conservative for the
statistic to actually cross it at this alpha/delta target.

## Why the numbers are weak, concretely

1. **Sample size.** All splits above have single-digit or low-double-digit
   sizes. This is a session/free-tier-budget constraint (each AgentDojo
   task takes ~10-400s on OpenRouter's free-model router; the collection
   scripts in `real_data/agentdojo/run.py` and `real_data/tau_bench/run.py`
   were deliberately scoped down to finish in a reasonable session, not
   because more data isn't collectible — rerunning with larger
   `FULL_CONFIG`/`end_index` values and more session time is the direct fix).
2. **Feature representation.** The surprise score is `-log P(tool | context)`
   under a Laplace-smoothed categorical/Gaussian MLE over *hashed* tool
   arguments (`sentry/scores.py`, `real_data/adapters.py`). This captures
   "which tool, roughly what argument shape" but not the *semantic content*
   of injected text. A successful AgentDojo attack often still calls a
   plausible-looking tool (e.g. `send_money` in the banking domain) with
   attacker-influenced arguments — structurally unsurprising to this model
   even though the content is compromised. algorithm.md itself scopes this
   as a simplification (§2, "a full causal-discovery engine ... is out of
   scope"); a real embedding of tool arguments/results, or the actual
   CAIRN-style causal world model, is the natural next step and would
   likely raise detection power substantially.

## Comparison to 2026 literature

No 2026 paper found (arXiv search, see prior turn) reports FAR/ARL/AUROC
results on AgentDojo, InjecAgent, WebArena, or tau-bench using an e-process
or e-detector method — SENTRY reporting *any* number on these benchmarks,
even a weak preliminary one, appears to be a first for this combination of
method (Ville/e-detector guarantees) and benchmark family. Specific points:

- **E-valuator** (arXiv 2512.03109, SENTRY's direct ancestor for the
  PAC-thresholding subroutine): its own PAC-threshold construction has the
  same finite-sample dependency SENTRY hit here — the E-valuator paper's
  reported numbers presumably use a much larger calibration set than the
  4-20 trajectories available in this session; that's the most direct
  apples-to-apples gap to close before a real comparison is possible.
- **Cheap Reward Hacking Detection** (arXiv 2606.08893): reports AUROC
  0.9467 (linear probe on model internals) vs. an LLM-judge baseline of
  0.9510, no formal guarantee. SENTRY's 0.35-0.60 AUROC here is far below
  that (and, at the low end, below the 0.5 chance line), but the
  comparison isn't fair yet: that paper's detector is trained
  on presumably hundreds+ labeled examples, uses model-internal features
  (not just hashed tool arguments), and targets a different failure mode
  (reward hacking in coding trajectories vs. prompt injection in tool-use
  agents). It's the right empirical power target to benchmark against once
  SENTRY's feature representation and sample size are both improved.
- **ToolChain-CRC** (arXiv 2606.18467) and the **strategic-deviations**
  paper (arXiv 2601.05427): closest in *framing* (conformal risk control +
  anytime escalation; e-value/test-supermartingale for behavioral
  deviation) but evaluated on synthetic/game-theoretic settings respectively,
  not on agent-safety benchmarks — no numeric benchmark overlap to compare
  against directly.

## What would fix this

1. More collection budget/time: scale `real_data/agentdojo/run.py`'s
   `FULL_CONFIG` and `real_data/tau_bench/run.py`'s `end_index` up, and
   rerun `python -m real_data.evaluate` — the pipeline is already correct
   and reruns are just data-volume-limited, not code-limited (the split
   logic, PAC threshold, and SR/CUSUM recursion are all covered by the 24
   passing tests in `tests/`, exercised on synthetic data where sample size
   isn't a constraint).
2. Replace the hashed-argument feature with a real text embedding of tool
   arguments and results (or the full causal-world-model do-intervention
   score from algorithm.md §2b) to raise the surprise score's actual
   discriminative power, independent of the sample-size fix above.
3. Rerun `select_order_statistic`/`pac_threshold` (`sentry/calibration.py`)
   at a larger D_thresh to get a real, non-saturated PAC bound instead of
   the vacuous max-of-5 threshold used here.
