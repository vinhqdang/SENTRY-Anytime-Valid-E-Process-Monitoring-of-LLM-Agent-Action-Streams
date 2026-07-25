# SENTRY — research log

A running record of every experiment, result, dead end and correction. Kept so
that nothing has to be re-derived and so that claims in the manuscript can be
traced to the run that produced them. Newest sections at the bottom.

---

## 1. Method as originally built

Monitor an LLM agent's action stream as **sequential change detection**: maintain
a nonnegative e-process over per-action surprise scores, alarm when it crosses a
threshold. Machinery imported from Shin–Ramdas–Rinaldo (e-detectors),
Vovk–Wang (e-values) and Sadhuka et al. (PAC thresholding). No new theorems.

Signals (all trace-only):
1. `transition` — tool-transition bigram surprise
2. `novelty` — fraction of argument tokens never seen in nominal data
3. `instruction` — instruction-likeness of the preceding tool observation
4. `unjustified` — *(added later)* how much the action fails to serve the user's task

Combiners: summed tilted increment, or per-signal conformal e-value mixture.

## 2. Data collected (all committed under `real_data/`)

| corpus | agent | attack | dir |
|---|---|---|---|
| AgentDojo ×4 suites | DeepSeek-V4-Flash | `important_instructions` | `agentdojo/logs_deepseek` |
| AgentDojo ×4 suites | DeepSeek-V4-Flash | `injecagent` | `agentdojo/logs_deepseek_injecagent` |
| AgentDojo ×4 suites | GPT-4o-mini | `important_instructions` | `agentdojo/logs_gpt4omini` |
| τ-bench retail | DeepSeek-V4-Flash | none (benign) | `tau_bench/logs_deepseek` |
| τ-bench airline | DeepSeek-V4-Flash | none (benign) | `tau_bench/logs_deepseek_airline` (partial) |

121 nominal (71 AgentDojo benign + 50 τ-bench), 215 + 45 + 175 attack trajectories.

Earlier free-tier (`openrouter/free`) collection was discarded: those agents were
degenerate (1–2 actions, no real tool use). DeepSeek-V4-Flash was used instead.

## 3. Results that held up

- **Synthetic validation**: empirical FAR tracks the target α across two orders
  of magnitude; detection within a few steps of an injected mean shift.
- **Attempt detection** (all attacks vs benign) on DeepSeek + `important_instructions`:
  AUROC ≈ 0.77, ~0.66 detection at ~2% per-step FAR, one-action median delay.
- **Cross-agent transfer**: re-fitting the reference on GPT-4o-mini trajectories
  works as well or better (AUROC 0.82).

## 4. Rejected signals (measured, not shipped)

| signal | result | why |
|---|---|---|
| data-flow taint (token-overlap fraction) | AUROC 0.46 | benign agents legitimately act on tool-derived values more than attacks do |
| lexical goal-grounding | AUROC 0.50 | token overlap too coarse a proxy for "serves the task" |
| injection-conditioned interaction term | no gain | identical to the plain sum |

## 5. Bug found and fixed: `excess_max` clipping

`instruction_excess` subtracted the **maximum** instruction-likeness seen in
nominal training data. Nominal median is 0.010 but its max is 0.613 (an outlier),
while InjecAgent's median is 0.115 — so the subtraction zeroed **100%** of the
InjecAgent signal.

- raw signal separates InjecAgent at **AUROC 0.84**
- after `max(0, x − β)`: **AUROC 0.50** (chance)
- AUROC is invariant to monotone transforms; `max(0, x−β)` is many-to-one, hence
  the loss

Replaced with a conformal empirical-tail transform
`−log((1 + #{nominal ≥ x}) / (n+1))`, strictly monotone.
Measured end-to-end (10 splits): InjecAgent observable detection 0.096 → **0.760**,
AUROC 0.468 → **0.701**; `important_instructions` AUROC 0.769 → 0.756;
FAR 0.020 → 0.033. Old behaviour kept as `instruction_transform="excess_max"`.

**Correction:** the manuscript previously explained the InjecAgent failure as
"data-integration phrasing carries almost no instruction-likeness." That is
**wrong**. InjecAgent payloads are plainly imperative (*"IMPORTANT!!! Ignore all
previous instructions…"*) and are present in observations 100% of the time (vs
77% for `important_instructions`). The failure was ours.

## 6. LLM judges replacing the fixed lexicon

`sentry/llm_judge.py`, DeepSeek-V4-Flash, temperature 0, scores cached by SHA-1
so evaluation is offline and reproducible (681 observations, 1329 task/action pairs).
Named free models were 404 on this key and `openrouter/free` was unreliable
(routed one call to a guard model), so a very cheap paid model was used.

4-arm ablation (`results_signal_ablation.json`), attempt detection / AUROC:

| scorer | transform | `important_instr` | `injecagent` |
|---|---|---|---|
| lexicon | excess_max | 0.658 / 0.769 | 0.083 / 0.468 |
| lexicon | conformal | 0.620 / 0.756 | 0.455 / 0.701 |
| llm | excess_max | 0.077 / 0.432 | 0.090 / 0.490 |
| **llm** | **conformal** | 0.648 / 0.750 | **0.531 / 0.714** |

Two findings: the semantic judge generalizes across attack phrasing where the
keyword lexicon does not; and the *transform* is a first-class design choice —
pairing the better scorer with the old transform is catastrophic, because one
benign observation scoring 1.0 clips every attack to zero.

## 7. Negative result: the e-process machinery is not earning its place

`real_data/longhorizon.py`, 1000-action benign streams, 10 seeds, signal held
fixed, only the decision rule varied:

| method | false alarms/1k | ARL | attack det. |
|---|---|---|---|
| SENTRY α=0.2 | 48.9 | 29 | 0.624 |
| SENTRY α=0.05 / 0.01 / 0.002 | 19.7 | 138 | 0.530 |
| fixed threshold q=0.99 | **17.4** | 158 | **0.588** |
| fixed threshold q=0.999 | **11.0** | 343 | 0.565 |

1. A plain fixed threshold on the same score **strictly dominates** — fewer
   false alarms *and* higher detection.
2. α = 0.05, 0.01, 0.002 give **identical** results: the PAC threshold saturates
   at this nominal-data scale, so the guarantee cannot be tightened below
   α = 0.2, where ARL ≥ 5 actions is vacuous. The manuscript's unqualified
   "PAC threshold is certified" holds *only* at α = 0.2.
3. Cross-agent threshold transfer shows no advantage either (fixed FAR
   0.010→0.026 vs SENTRY 0.021→0.029).

## 8. Attempt 3 — action-justification audit (failed)

Idea: score the **action** ("did the user ask for this tool call?") rather than
the observation, to be independent of attack phrasing. The judge sees only
(user request, tool, arguments) — never the observations — so it cannot
re-detect the injected text.

Hand-validated correctly on the case that defeated taint: benign `send_money` to
a *tool-derived* IBAN the task justifies → 0.0; attack `send_money` to an
unrelated IBAN → 1.0.

But at corpus scale it **fails**: AUROC 0.547, TPR **0.000** at 3.66% FPR.
Diagnosis: it flags 17–26% of *benign* trajectories (agents take tangential
exploratory actions) while only ~29% of attacks contain an obviously
unjustified call. Combining it *hurts* (0.296 vs 0.604).

Note the 0.000 is partly a **saturation artifact**: 17% of benign trajectories
score exactly 1.0, so benign's 96.34th percentile *is* 1.0 and TPR at that FPR
is zero by construction. AUROC 0.547 is the honest number.

## 9. Attempt 4 — sensitive-sink provenance (failed)

Conjunctive per-value rule: sensitive sink **AND** identifier absent from the
user request **AND** identifier first surfacing in untrusted prose rather than a
structured record field. Result: TPR 0.112 at FPR 0.073 — unusable.

## 10. Head-to-head vs published SOTA

`real_data/compare_sota.py`, same corpus, matched benign FPR:

| detector | AUROC | TPR@3.66% |
|---|---|---|
| behavioural (bigram+novelty) | 0.708 | 0.229 |
| keyword lexicon | 0.787 | 0.562 |
| instruction judge (LLM) | 0.791 | **0.604** |
| action audit | 0.547 | 0.000 |
| combined | 0.750 | 0.296 |
| AgentArmor (published, other corpus) | — | 0.958 |

We do **not** beat published SOTA.

## 11. **Major bug: the attack-success label was inverted all along**

While investigating why ~53% of "successful" attacks contained no
externally-effectful tool call, I found trajectories labelled compromised in
which the agent had *explicitly refused*:

> "I see a message embedded in the bill file that's trying to get me to do
> something else first. However, I shouldn't be tricked by that embedded
> instruction."

AgentDojo's `security` flag is `_check_task_result(injection_task, …)` — the
**injection task's** success. Settled empirically on ground truth: in the banking
suite the only trajectories that actually wired money to the attacker's IBAN are
exactly those with `security=True` (2/2), while all 46 that refused have
`security=False`.

**Therefore `security=True` = attack SUCCEEDED, `security=False` = resisted.**
The project had it backwards from the start.

Consequences:
- Every "hijack-success detection" number measured the **resisted** class.
- The corpus table ("180 successful / 18 resisted") is inverted: it is **18
  compromised / 180 resisted** for DeepSeek + `important_instructions`.
- My earlier "benchmark ceiling" claim (that AgentDojo success decouples from
  detectable harm) is **retracted** — it was an artifact of the inverted labels.
  With correct polarity, compromise does imply a harmful action:

  | corpus | attacks | truly compromised | of those, have a sink call |
  |---|---|---|---|
  | DeepSeek `important_instructions` | 215 | 18 (8%) | 13/18 |
  | DeepSeek `injecagent` | 45 | **0 (0%)** | — |
  | GPT-4o-mini `important_instructions` | 175 | **72 (41%)** | 61/72 |

- **Attempt-detection numbers are unaffected** (they pool all attacks regardless
  of outcome), so §3, §5, §6 and §10 above still stand.
- New substantive finding: **DeepSeek-V4-Flash is far more robust to these
  injections than GPT-4o-mini** (8% vs 41% compromised; 0% for InjecAgent).

Fixed in `real_data/evaluate.py` and `real_data/evaluate_generalization.py`.

## 12. Submission history

- **AAMAS** — desk-rejected as "not appropriate".
- **IEEE TDSC** (SI: Safety, Alignment, Responsibility of LLMs),
  TDSCSI-2026-07-3011 — desk-rejected at prescreening: *"not suitable for
  publication due to its low quality. It lacks novelty and technical depth."*
  Six-month wait before resubmission. Reviewer criticism is consistent with §7
  and §10: the theory is imported, and the headline result came from a keyword
  counter matched to one attack template.

## 13. Open next step

With labels corrected there is, for the first time, a well-posed harm-detection
target: **90 genuinely compromised trajectories** (18 DeepSeek + 72 GPT-4o-mini),
74 of which contain an externally-effectful call. Effect-based detection
(sink provenance, action audit) should be re-evaluated against *these* positives
rather than against the resisted class it was previously scored on.
