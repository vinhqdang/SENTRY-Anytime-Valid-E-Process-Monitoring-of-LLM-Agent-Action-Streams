# SENTRY on real agent trajectories: results

Running SENTRY-Detect on real LLM agent trajectories instead of synthetic
Bernoulli/Gaussian streams, per algorithm.md §7's evaluation plan. Data
collection used OpenRouter's `openrouter/free` free-model router (a
rotating pool of ~24 free-tier models) as the agent (and, for tau-bench,
the simulated user) LLM.

**Bottom line up front.** Replacing the original surprise score with a
task-transferable tool-transition model plus an explicit argument-novelty
term (`sentry.scores.SequentialWorldModel`) moved the real-data results
from chance-level to a real, reproducible signal:

| metric (mean ± std over 10 random splits) | old score (`CausalWorldModel`) | **new score (`SequentialWorldModel`)** |
|---|---|---|
| AUROC (length-normalized, post-drift mean) | 0.47 ± 0.10 | **0.69 ± 0.14** |
| AUROC (max-of-stream) | 0.32 ± 0.13 | **0.60 ± 0.14** |
| Detection rate @ Ville threshold (1/α, α=0.2) | 6.3% | **45.6% ± 8.9%** |
| Per-step false-alarm rate @ Ville threshold | 6.2% | 11.5% (≤ α = 20%, consistent with the ARL bound) |
| Detection rate @ PAC threshold | 3.1% | **38.8% ± 9.2%** (per-step FAR 5.6%) |
| Median detection delay (post-drift steps) | 4.5 | **1.0** |

Median delay of **1 step past the injection point** matters most: when the
new score fires, it fires on essentially the first action the attacker
influences. This is still far from a solved problem (see limitations), but
it is a genuine detector now, not noise.

## What changed (the algorithmic improvements)

Diagnosed from the previous round's failure analysis and implemented in
`sentry/scores.py` (`SequentialWorldModel`) + `real_data/adapters.py`:

1. **Tool-transition (bigram) model instead of per-task conditioning.**
   The original `CausalWorldModel` conditions tool probabilities on
   `task_id` — but held-out and attack trajectories always have *unseen*
   task_ids, so its tool term degenerated to near-uniform exactly where it
   was evaluated. The new model learns `P(tool_t | tool_{t-1})` pooled
   across all training trajectories: "read_file is followed by get_balance,
   not send_money" is task-independent structure that transfers.
2. **Explicit argument-token novelty.** Hashed features can only surface
   novel values through accidental collision statistics. The new model
   keeps a per-tool vocabulary of argument tokens seen in nominal training
   data and scores the *fraction of unseen tokens* directly. An injected
   recipient IBAN (`US133000000121212121212`) or attacker URL is by
   construction absent from nominal data — this is precisely the signature
   of a hijacked action, and it is a **single-action** signal, which is why
   the median delay dropped to 1 step (AgentDojo attacks average only 2.6
   actions, too short for multi-step evidence accumulation alone).
3. **Robustness checks.** The improvement is stable across the novelty-
   weight hyperparameter (AUROC 0.68 ± 0.01 for weights 1–12, so nothing
   is tuned to a knife-edge) and across 10 random data splits (all numbers
   above are mean ± std over splits, after single-split results were shown
   to swing ±0.15 from split luck alone).
4. **Reproducibility fix (found during this round).** Feature hashing used
   Python's builtin `hash()`, which is salted per process — every run of
   the evaluation produced different hashed features and therefore
   different `CausalWorldModel` numbers. Switched to `zlib.crc32`; the
   full evaluation is now byte-identical across runs (verified by running
   it twice back-to-back).

Earlier rounds' fixes that fed into this (see git history): regex
tokenization of compact JSON (whitespace-splitting was nearly blind to
argument content), hashing the preceding tool-result text into each
action's features, whitespace-normalized injection matching for drift-index
location, and a length-normalized AUROC metric after `max(stream)` was
shown to confound with trajectory length (nominal ≈ 5.3 actions vs. 2.6
for successful attacks).

## Data (final)

| Source | Suites | Nominal | Successful attack | Resisted attack |
|---|---|---|---|---|
| AgentDojo | banking, workspace, travel, slack | 18 | 16 | 16 |
| tau-bench | retail | 14/15 (1 rate-limited) | n/a | n/a |

Attacks are AgentDojo's `important_instructions` prompt injection;
"successful" = AgentDojo's own `security == False` label (the injection
actually hijacked behavior). 24 nominal trajectories total pass the
≥2-action filter. Split per seed: train=8, cal=6, thresh=5, test=5.

## Honest limitations (unchanged in kind, updated in degree)

- **The PAC threshold is still saturated** (D_thresh = 5 streams), so the
  PAC-threshold rows above use "max of 5 nominal statistics" rather than a
  certified bound. The Ville-threshold rows (1/α) are the theoretically
  meaningful operating point *if* the learned score were an exact
  reference, which it is not — a real PAC bound needs a larger D_thresh.
  This remains the top item for a credible guarantee claim.
- **45.6% detection is not 95%.** Slightly over half of successful attacks
  still evade at this threshold, typically those whose injected action
  uses only tokens that also appear in nominal data.
- **n = 16 attacks.** Enough to distinguish 0.69 from 0.5 (the old-model
  comparison is run on identical splits), not enough for tight confidence
  intervals; ±0.14 std over splits is honest and large.
- The trace plot (`real_data/plots/real_data_traces.png`) shows the
  detector statistic spiking exactly at the drift index on a real attack
  and staying flat on a real nominal trajectory.

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
  failure mode (reward hacking) and far more training data. SENTRY's 0.69
  from ~8 nominal trajectories of *black-box* tool-call logs, with an
  anytime-valid ARL story attached, is not directly comparable — but the
  0.95-class number is the empirical bar to aim at as data and features
  scale.
- **ToolChain-CRC** (arXiv 2606.18467) / **strategic deviations** (arXiv
  2601.05427): nearest methodological neighbors (conformal risk control /
  test-supermartingales), still no shared benchmark to compare against.

## What would move the needle next

1. **More D_thresh streams** → a real, non-saturated PAC threshold and a
   defensible FAR guarantee (the single biggest gap between "has the
   theory" and "demonstrates the theory").
2. **Argument-slot-aware novelty** (per argument *key*, not pooled tokens)
   and value-shape features (IBAN-like, URL-like, email-like) to catch
   attacks whose injected values are individually familiar tokens.
3. **More attacks and more diverse attack types** (AgentDojo has 16
   further injection tasks per suite plus DoS variants untouched) to
   tighten the ±0.14 split-variance and test generalization beyond
   `important_instructions`.
4. **Mixture of both scores**: the bigram+novelty score and the
   hash-Gaussian score can be combined as parallel e-detector mixtures
   (algorithm.md §3 supports this natively) rather than choosing one.
