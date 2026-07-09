# SENTRY on real agent trajectories: results and honest limitations

This is a first pass at running SENTRY-Detect on real LLM agent trajectories
instead of synthetic Bernoulli/Gaussian streams, per algorithm.md §7's
evaluation plan. Data collection used OpenRouter's `openrouter/free`
free-model router (a rotating pool of ~24 free-tier models) as the agent
(and, for tau-bench, the simulated user) LLM, with no other API cost.

**Bottom line up front**: after two rounds of real bug fixes and a 4x
increase in attack-trajectory sample size, SENTRY does not yet reliably
detect real AgentDojo prompt-injection attacks at this feature/model
configuration (length-normalized AUROC 0.44, i.e. no better than chance
within this sample's noise). This section explains exactly what was fixed,
what changed as a result, what's still broken, and why -- no number here
should be read as "SENTRY works" or "SENTRY is the best."

## Data collected (final)

| Source | Suites | Nominal | Attack (complete) | Successful attack | Resisted attack |
|---|---|---|---|---|---|
| AgentDojo | banking, workspace, travel, slack | 18 | 32 | **16** | **16** |
| tau-bench | retail | 14/15 (1 hit an OpenRouter free-tier rate limit) | n/a (no attack suite) | n/a | n/a |

Successful-attack count went from 4 (2 suites only) to 16 (all 4 suites) via
`real_data/agentdojo/run.py --attack-expansion`, specifically to get past a
sample size too small for any AUROC/detection-rate number to mean anything.

AgentDojo attacks use the `important_instructions` prompt-injection attack
(AgentDojo's strongest baseline attack). "Successful" = AgentDojo's own
`security == False` label, i.e. the injection actually hijacked the agent's
behavior, not just that an injection was present.

## Two real bugs found and fixed (not just sample-size noise)

1. **Feature hashing tokenized on whitespace.** `json.dumps`'s default
   compact separators leave almost no whitespace --
   `{"file_path":"bill.txt"}` splits into 2 whitespace-tokens, not 3
   meaningful ones (`file_path`, `bill.txt`, plus structure) -- so the hash
   was nearly blind to argument content. Fixed with a word/number regex
   tokenizer (`real_data/adapters.py`, commit `3b6db76`).
2. **The surprise score never saw tool results, only actions.** Injected
   content lives in what a tool *returns* (e.g. a file/email body), not in
   the next action's own arguments -- so a score computed from the action
   alone structurally cannot see where the injection is. Fixed by hashing
   the immediately preceding tool-result text into each action's feature
   vector, which is the `H_{t-1}` conditioning algorithm.md §2a's
   `s_t = -log P(a_t | do(c), H_{t-1})` actually calls for.

Both are real, verifiable defects (reproduced with a raw JSON string and
`str.split()` vs. a regex tokenizer) independent of anything statistical.

## A third finding: max(stream) confounds with trajectory length

After fixing the above and quadrupling the attack sample, AUROC (nominal
vs. successful attack, using `max` of the raw surprise-score stream per
trajectory) came out at **0.27 -- decisively below chance**, and this time
with enough data (16 attacks x 5 held-out nominal = 80 pairs) that it
wasn't just noise. Investigating: nominal trajectories average **5.3**
actions, successful attacks average **2.6** -- about half as long. `max`
over a longer stream has more chances to hit a high value by chance alone,
regardless of whether anything is actually anomalous, so this metric was
measuring trajectory length more than surprise.

Switching to a length-normalized, drift-aware metric -- mean surprise over
the *post-drift* region for attacks (the quantity algorithm.md §5's
detection-delay bound actually depends on, `D(Q‖P)`) vs. mean surprise over
the whole trajectory for nominal streams -- gives **AUROC 0.44**. Still
below 0.5, but no longer a dramatic, confidently-wrong signal; more
accurately described as "no reliable separation found yet" rather than
"actively anti-correlated." (`real_data/evaluate.py` now reports both
numbers explicitly, labeled, so this confound can't silently reappear.)

One individual trace (`real_data/plots/real_data_traces.png`) shows a
clean, steady rise in the e-detector statistic immediately after the true
drift index on a real successful attack, while a real nominal trajectory
stays flat -- so the signal *can* show up clearly case-by-case. It just
doesn't show up reliably enough across the full 16-attack set to move the
aggregate AUROC away from chance.

## Why detection is still structurally hard here

- **PAC threshold still saturated.** D_thresh has only 5 streams, so
  `c_alpha` (131.7) is literally the max observed nominal statistic in
  that tiny sample, not a real bound -- `empirical_far=0.0` remains a
  vacuous consequence of that, not evidence of good FAR control.
  `detected 0/16` follows directly: the threshold is enormous relative to
  any individual trajectory's statistic at this scale.
- **Attacks are short by construction.** AgentDojo's `important_instructions`
  attack is designed to make the agent perform one extra malicious action
  and then continue -- median attack length here is 2 actions, meaning as
  few as 1 post-drift observation exists to accumulate evidence on. This
  is a structural limit for *any* sequential change-point detector, not
  specific to SENTRY's feature choices: e-detectors need multiple
  post-change steps to build up a statistic past a threshold (algorithm.md
  §5's delay bound assumes this). A single-shot "is this one action
  causally consistent" score (algorithm.md §2b's intervention-consistency
  score, implemented in `sentry/scores.py` but **not yet wired into this
  evaluation**) is the more natural tool for a 1-2 step attack than
  accumulation-based SR/CUSUM, and is the most promising concrete next step.
- **Argument features are still generic hashing, not novelty-aware.** The
  fix above lets the model see argument/observation *content*, but it's
  still a frequency-based hash, not "is this recipient IBAN one I've seen
  before." A per-argument-slot novelty/OOV feature (has this exact value,
  or this value's type/shape, appeared in training data before?) is a
  concrete, implementable improvement that directly targets what actually
  makes an injected `send_money` call anomalous -- an unfamiliar
  recipient -- rather than relying on generic hash collisions to surface it.

## Comparison to 2026 literature

No 2026 paper found (arXiv search, prior turn) reports FAR/ARL/AUROC
results on AgentDojo, InjecAgent, WebArena, or tau-bench using an e-process
or e-detector method. That's still true after this round -- SENTRY
reporting *any* number here, even a weak one, remains a first for this
combination of method and benchmark family, not a reproduction of anyone
else's numbers. Specific points, updated:

- **E-valuator** (arXiv 2512.03109, SENTRY's direct ancestor for the
  PAC-thresholding subroutine): same finite-sample dependency SENTRY hit
  here (D_thresh=5). Its own reported numbers presumably use a
  substantially larger calibration set; that gap, not the feature
  engineering done in this round, is the most direct lever for a real
  apples-to-apples comparison.
- **Cheap Reward Hacking Detection** (arXiv 2606.08893): AUROC 0.9467
  (linear probe on model internals), no formal guarantee. SENTRY's
  length-normalized AUROC of 0.44 here is far below that. The gap didn't
  close this round -- it's now measured more honestly (0.27 vs. 0.44
  depending on metric) rather than closed.
- **ToolChain-CRC** (arXiv 2606.18467) and the **strategic-deviations**
  paper (arXiv 2601.05427): still no numeric benchmark overlap -- different
  evaluation settings (synthetic/RAG drift; game-theoretic), so still no
  direct comparison possible.

## What would actually move the needle (in priority order)

1. **Wire in the intervention-consistency score** (`sentry/scores.py`'s
   `CausalWorldModel.intervention_consistency_score`, unused by
   `real_data/evaluate.py` so far) as a second signal alongside the
   predictive-residual score, specifically because it's a single-action
   score and attacks here average 2.6 actions -- it doesn't need
   accumulation time the way the SR/CUSUM recursion does.
2. **Novelty-aware argument features**: track, per tool and argument key,
   the set of values seen during training, and score unseen values as
   surprising directly -- rather than relying on generic hash collisions
   to indirectly surface novelty.
3. **More D_thresh samples** to get a real (non-saturated) PAC threshold --
   orthogonal to the two items above, and the fix for the vacuous
   `empirical_far=0.0` specifically.
4. **More attack trajectories still** -- 16 is enough to make AUROC
   estimates meaningful (as this round demonstrated, going from 4 to 16
   changed the finding qualitatively), but a future round with 40-60 would
   tighten the confidence interval on whatever the next feature change
   produces.
