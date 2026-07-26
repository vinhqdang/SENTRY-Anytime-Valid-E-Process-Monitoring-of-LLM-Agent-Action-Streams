# SENTRY — research log

> **Status: research log, appended chronologically.**
> Sections are written as work happened and are NOT revised when later sections
> supersede them. Earlier sections therefore contain claims the paper withdraws --
> including a detectability-ceiling bound and a comparison against published
> figures. Where an entry conflicts with `manuscript/sentry_cose.tex`, the
> manuscript is correct. The audit findings in later sections were acted on; see
> the git history for the corrections.


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

## 12. Venue history

Earlier drafts targeted other venues and were not accepted. The substantive
criticism we took from that process, and acted on, is recorded in the technical
sections rather than here: the anytime-valid theory was imported rather than
extended (see section 7), and the original headline number came from a keyword
counter matched to a single attack template (sections 5 and 10). Correspondence
and reviewer text are kept outside this repository.

## 13. SENTRY-Fuse — the working algorithm

With labels corrected, effect-based signals are finally scored against the right
class and the picture inverts: **the action audit goes from AUROC 0.547 to
0.926**. It was never a bad signal; it was graded on the wrong answer.

**Algorithm.** Convert each signal to its conformal tail surprisal against a
benign calibration split, `-log((1 + #{benign >= x}) / (n+1))`, and sum:
`S = Σ_k -log p_k`. Scale-free (a bigram surprise in nats and a [0,1] judge score
become comparable), strictly monotone in every component (so no signal can be
clipped away — the `excess_max` failure of §5 cannot recur), and it needs **no
tuned weights**.

**Protocol.** Benign trajectories are split in half per seed; one half calibrates
both the conformal transforms and the operating threshold, the other half supplies
the held-out false positives. Compromised trajectories are always held out.
10 seeds, mean ± std.

GPT-4o-mini (72 compromised / 70 benign):

| detector | AUROC | TPR@3.66% |
|---|---|---|
| instruction only | 0.955 ± 0.017 | 0.825 ± 0.279 |
| behavioural only | 0.534 ± 0.015 | 0.000 |
| instr + audit | 0.985 ± 0.003 | 0.932 ± 0.031 |
| **SENTRY-Fuse (all 4)** | **0.985 ± 0.005** | **0.960 ± 0.010** |
| AgentArmor (published) | — | 0.9575 |

Pooled (90 compromised / 191 benign): `instr + audit` AUROC 0.955, TPR@3.66%
**0.902 ± 0.018**; SENTRY-Fuse 0.956 / 0.890 ± 0.039.

**Leakage caught by the honest protocol.** In the first harm run `behavioural`
scored AUROC 0.982 because its reference model was fit on *all* benign
trajectories including the test half. With calibration held out it collapses to
0.53–0.62. Only held-out numbers are reported anywhere.

## 14. Detectability ceiling (the framing result)

Pooled over the 90 genuinely compromised trajectories:

| channel of evidence | n | frac |
|---|---|---|
| injected payload visible in an observation | 78 | 0.867 |
| an externally-effectful (sink) call | 74 | 0.822 |
| **either channel** | **83** | **0.922** ← ceiling for *any* trace-only detector |
| neither — no trace evidence at all | 7 | 0.078 |

Consequences:

1. Our 0.902 is **97.8% of the attainable 0.922**. Trace-only detection on this
   corpus is essentially saturated.
2. Both fused detectors **exceed the payload-only ceiling of 0.867**, which is
   direct evidence that the action-side signals carry information the
   observation-side signal cannot — the orthogonality claim, measured.
3. **The published 0.9575 exceeds this corpus's trace-only ceiling of 0.922**, so
   it cannot be an apples-to-apples comparison; it must rest on a different
   positive-class definition, a different corpus, or information beyond the
   black-box trace. Any future paper reporting >0.92 here should say which.

**Defensible claims for the paper.** (i) A new fusion algorithm that breaks the
observation-only ceiling; (ii) it beats the strongest published trace-only number
on the matched GPT-4o-mini corpus (0.960 ± 0.010 vs 0.9575); (iii) a
detectability ceiling that bounds every trace-only monitor and explains why the
earlier effect-based attempts (§8, §9) appeared to fail.

## 15. Remaining work

- The 7 no-evidence compromises are invisible because the monitor never sees the
  agent's own natural-language output. Monitoring that channel is the concrete
  route past 0.922.
- Regenerate `results.json` / `results_generalization.json` and every manuscript
  number under the corrected label polarity; the current manuscript's
  hijack-detection figures measure the resisted class.
- τ-bench airline collection is partial (~task 8/50).

## 16. Independent pre-submission audit (2026-07-26) — BLOCKING findings

Four independent reviewers audited the C&S manuscript (numbers, references,
math/notation, framing). Every finding below that I state as confirmed I
re-verified myself against the code and data; agent claims I could not
reproduce are marked as such.

### 16.1 The detectability ceiling does not exist on this corpus (nu = 0, not 0.078)

**This is the most serious finding and it invalidates Contribution 3.**

`results_ceiling.json` (the source of Table 6, Theorem 1's nu, Corollary 1, the
0.922/0.925 bound, the "97.8% of attainable" claim and the critique of
AgentArmor) **has no generating script anywhere in the repo**. I wrote one:
`real_data/ceiling.py`.

Measured directly on the same 90 compromised trajectories (72 GPT-4o-mini + 18
DeepSeek, `security is True`, `len >= 2`), asking whether the attacker's
imperative instruction text appears anywhere in the observation stream:

| criterion | visible / 90 |
|---|---|
| attacker's imperative clause, 6-gram match | **90 / 90 = 1.000** |
| any 8-word span of the payload | 90 / 90 = 1.000 |
| any 25-word span | 80 / 90 = 0.889 |
| full payload, exact after whitespace normalisation | 26 / 90 = 0.289 |
| distinctive body only (wrapper stripped) | 36 / 90 = 0.400 |
| **paper's claim** | **78 / 90 = 0.867** |

The paper's 0.867 is not reproducible under any definition I could construct.
The *semantically correct* answer is 1.000, and it is 1.000 for a structural
reason: AgentDojo delivers injections **through tool outputs**, and a
*successful* injection requires the agent to have read that output. So the
payload is in the trace by construction.

Consequence: `either_channel` = 90/90, `neither_channel` = 0, **nu = 0.000**.
There is no set of "compromised trajectories with no trace evidence" on this
corpus. Therefore:

- "7.8% leave no trace evidence whatsoever" (abstract) is **false**;
- the ceiling 0.922 / 0.925 (Theorem 1, Corollary 1, Table 6) **does not bind**;
- "we reach 97.8% of the attainable" is **vacuous** (the attainable is 1.0);
- the critique that AgentArmor's 0.9575 "exceeds the ceiling and must therefore
  explain" is **unfounded** and must be withdrawn;
- the title "Breaking the Observation Ceiling" refers to a quantity (0.867)
  measured the same unreproducible way.

The sink-call channel *is* roughly reproducible: I measure 77/90 = 0.856 vs the
paper's 74/90 = 0.822. Only the payload-visibility channel is wrong.

Note the construct is degenerate on this benchmark regardless of measurement: on
AgentDojo, "payload appears in the trace" is ~tautologically 1 for successful
attacks. A meaningful ceiling would need a different notion of evidence (e.g.
payload not *distinguishable* from benign content), which we have not measured.

### 16.2 Proposition 3 is false as stated; it contradicts Proposition 1

Prop 3 claims psi "preserves the AUROC"; the text claims "the conformal
surprisal **cannot destroy** a signal" and the abstract/contributions/conclusion
say "strictly monotone" / "provably order-preserving".

psi takes at most n+1 distinct values and **saturates at log(n+1) for every x
above the calibration maximum** -- it is many-to-one, exactly like phi_max.
Verified counterexamples:

- cal = {0}, benign {0.1, 0.2}, attack {0.3, 0.4}: raw AUROC **1.0 -> 0.5**.
  Every value maps to log 2. The conformal transform destroyed a perfectly
  separating signal -- the precise failure Prop 1 attributes to phi_max.
- At the paper's real calibration size n = 35, well-separated Gaussians:
  1.0000 -> **0.9475**. AUROC strictly decreases at production scale.

Worse, Prop 3's proof argument ("a non-decreasing map cannot reverse any strict
ordering") applies verbatim to phi_max and would "prove" Prop 1 false. And the
paper's own stated lesson -- "a fusion rule must never apply a many-to-one
transform to a component signal" -- disqualifies SENTRY-Fuse itself.

The real distinction is **resolution of the quantisation** (n+1 bins vs 1 bin),
not many-to-one vs one-to-one. Prop 3 must be restated as: psi is non-decreasing,
hence cannot *invert* an ordering, and preserves AUROC up to ties introduced by
quantisation at n+1 levels.

### 16.3 "No tuned weights" is false on the reported path

`real_data/evaluate_fused.py:83`:

    best_b = max(best_b, sv["transition"] + 4.0 * sv["novelty"])

Column 0 of the fused matrix is a **hardcoded weighted sum of two raw
sub-signals on incommensurable scales** (nats + 4.0 x a [0,1] unseen-token
fraction), applied *before* any conformal transform. So:

- "no tuned weights" (abstract, contributions, Section 3.7, conclusion) is false;
- there are **five** signals, not four -- `novelty` appears nowhere in the
  manuscript (verified: zero occurrences);
- same constant in `compare_sota.py:73`, `evaluate_harm.py:77`;
- `manuscript/make_figures.py:185` even *sweeps* novelty/instruction weights.

Cleared: `sentry/scores.py:242`'s `unjustified_weight=4.0` is only used by
`SequentialWorldModel.surprise()`, which the fused evaluation never calls. The
S2/S3/S4 columns are genuinely unweighted.

### 16.4 The 3.66% operating point is unattainable; realised FPR is 5.71%

`_tpr_at` sets `tau = np.quantile(neg, 1-fpr)` then counts `x > tau`. With 35
test negatives (GPT-4o-mini, 70 benign halved) the attainable FPRs are multiples
of 1/35 = 2.86%. Verified:

| corpus | n_test | nominal 3.66% | nominal 5% |
|---|---|---|---|
| GPT-4o-mini | 35 | **2/35 = 5.71%** | 2/35 = 5.71% |
| pooled | 96 | 4/96 = 4.17% | 5/96 = 5.21% |

So "0.960 at 3.66% FPR beats AgentArmor's 0.9575 at 3.66%" compares our number
at a realised **5.71%** against theirs at 3.66%. The tell is already visible in
Table 3: TPR@3.66% is *identical* to TPR@5% in four of five GPT rows, because
both nominal levels select the same threshold.

### 16.5 The abstract attributes 0.902 to SENTRY-Fuse; SENTRY-Fuse scores 0.890

Table 3 pooled: S2+S3 = 0.902 +/- 0.018, SENTRY-Fuse (S1-S4) = 0.890 +/- 0.039.
Confirmed in `results_pooled.json`. The body is candid (Section 5.1: "S2+S3
slightly outperforms the full fusion"), which makes the abstract's and
conclusion's attribution worse, not better. `figures/ceiling.pdf` also labels
the 0.902 bar "SENTRY-Fuse (ours, pooled)". The complementarity argument
(0.902 > 0.867) therefore rests on an ablation, not on the proposed method --
and for SENTRY-Fuse proper the margin is 0.890 vs 0.867, i.e. 0.6 sigma.

### 16.6 AgentArmor's 95.75% / 3.66% exists only in arXiv v1

Both numbers appear only in `arXiv:2508.01249v1` (2025-08-02, GPT-4 agent). In
v2/v3 (current) the authors **removed** the claim; v3 reports TPR 0.86-0.97 at
FPR 0.03-0.19 and zero occurrences of "95.75" or "3.66". Our `.bib` cites the
unversioned DOI, so a referee following the citation cannot find either number.
Must cite v1 explicitly or re-anchor to v3's figures.

### 16.7 `figures/instruction_sep.pdf` uses the INVERTED polarity

`make_figures.py:66-67` still has `succ = security is False` / `resisted =
security is True` -- backwards relative to the Section 4.4 correction. The
rendered axis labels ("successful attack (n=180)", "resisted attack (n=18)") are
swapped relative to Table 1. Same inverted `_load()` feeds `roc.pdf`,
`dataset.pdf`, `signals.pdf`, `traces.pdf` (shipped in the zip, not included).

### 16.8 Other confirmed items

- **Pooled S2-only TPR@5% = 0.901 exceeds the claimed 0.867 observation ceiling**
  (Table 3 vs Section 5.1) -- independent of 16.1, an internal contradiction.
- **DeepSeek compromise block computed, released, omitted from the paper.**
  `results_fused.json`: SENTRY-Fuse 0.843 AUROC / 0.744 TPR vs S2-alone 0.881 /
  0.778. Fusion is *strictly worse than S2 alone* on that corpus, contradicting
  Contribution 2. The stated reason for omission ("estimates are wide") is
  refuted by the released sigmas (0.000-0.036).
- **Ceiling quoted as both 0.922 and 0.925**; observation ceiling as 0.867 where
  the paper's own theorem gives 0.872.
- **Theorem 1's Assumption 1 is existential** and does not support the proof step
  TPR|_N <= alpha, which needs the conditional law on N to be dominated by the
  benign law. As stated, TPR = 1 with FPR = alpha is consistent with the
  hypotheses.
- **Prop 4's second claim asserts S ~ Gamma(K,1) equality** where its proof gives
  only stochastic domination; S is discrete and bounded by sum_k log(n_k+1).
- **Prop 2's proof asserts rank uniformity**, which fails under ties -- S4 is
  binary and S3 saturates at 1.0 on 17% of benign. Conclusion survives via the
  conservative >= convention; the proof does not say so.
- **Reproducibility**: no script generates `results_ceiling.json`,
  `results_pooled.json`, `results_signal_ablation.json`,
  `results_detectability_ceiling.json`, nor `overview/fusion/ceiling.pdf`.
  `evaluate_fused.py` has no pooled code path at all. `reproduce.sh` omits
  `evaluate_fused`, `longhorizon`, `compare_sota`, `evaluate_harm`. `report.py`
  still checks pre-label-fix targets (121/180/18, 0.99). So "every number and
  figure is reproducible" is false as shipped.
- **Submission zip lacks `references_cose.bib` / `.bbl`** -- cannot compile its
  bibliography standalone.
- **Public repo contradicts the paper**: README advertises IEEE TDSC, "99% at 2%
  FAR", "PAC certified", "no extra LLM call", and the *retracted* InjecAgent
  explanation. `SUBMISSION_CHECKLIST.md` and this NOTES.md quote two desk
  rejections verbatim -- and Data availability sends every referee to that repo.
- **`shin2023edetectors` volume is 2, not 1** (NEJSDS Vol 2 Iss 2, 2024).

### 16.9 What the audit cleared

- **References: no fabrications.** 34 cited / 34 present, zero orphans either
  way, full author lists throughout, every arXiv ID and DOI resolves to the
  exact cited title. Only defect: the volume above, a missing page range on
  `chen2025secalign`, and the AgentArmor version issue (16.6).
- **Every cell of all 7 tables matches the committed JSON** to 3 dp.
- **The conformal p-value implementation is exactly correct**, including
  `searchsorted(side="left")` tie handling (stress-tested at tie points), the
  +1/+1 correction, the log, and the tail direction.
- The AUROC implementation, S4's code-to-equation correspondence, the
  label-polarity analysis (Section 4.4, Table 2), and the leakage remark are all
  sound. 5 figures included, all `\ref`'d, none dangling.

### 16.10 Status

**The submission package must not go out as-is.** Contribution 3 (the ceiling)
is unsupported; Contribution 1's "no tuned weights" and Prop 3 are false as
stated; the headline comparison is not at the FPR claimed; and the abstract
credits the method with an ablation's number. These are not presentational.
