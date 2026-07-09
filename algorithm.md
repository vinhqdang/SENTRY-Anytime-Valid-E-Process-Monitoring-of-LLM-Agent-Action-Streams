# SENTRY: Algorithm Design
## Sequential E-process monitoriNg for Trustworthy autonomous AgencY

## 0. What changed after reading the two PDFs

**E-valuator (Sadhuka et al. 2512.03109v2)** gives SENTRY two directly reusable tools:
- The **log-optimal density-ratio test statistic** $M_t = p_0(S_{[1:t]})/p_1(S_{[1:t]})$ (their Eq. 2), proven log-optimal via Ramdas–Wang Thm 7.11 — fastest expected-log-growth e-process against a fixed alternative.
- **PAC thresholding** (their Algorithm 1 / Proposition 1): when $p_0, p_1$ must be *estimated*, exact Ville thresholds ($c_\alpha = 1/\alpha$) no longer control FAR, because the estimated process is not an exact e-process. Their fix — take the max of the estimated process over each calibration trajectory, then set $c_\alpha$ to a distribution-free binomial-tail upper confidence bound on the $(1-\alpha)$-quantile of that max — is a **necessary component for SENTRY**, since SENTRY's causal-world-model reference will also be *learned*, not known exactly.

Their scope, however, is narrowly per-task: null = "trajectory will succeed" (label $Y=1$) vs. alternative = "trajectory will fail" ($Y=0$), calibrated from a labeled dataset of complete trajectories, monitored until the *task* ends. This is fundamentally a two-class classification-calibration problem dressed as sequential testing. SENTRY's null is instead "behavior is consistent with the safe/nominal reference" with **no natural alternative class to label** — safety violations are open-ended and often unseen at calibration time — and the monitor runs **continuously across many tasks in a deployment**, not once per task. So E-valuator's PAC thresholding is adopted as a subroutine, but its density-ratio/two-class-label machinery is not: SENTRY needs a **one-class / distributional-deviation** construction instead.

**E-detectors (Shin, Ramdas & Rinaldo, NEJSDS 2024)** gives SENTRY its actual backbone:
- The **e-detector** primitive: $M$ is an e-detector for pre-change class $\mathcal{P}$ if $\mathbb{E}_{P,\infty}[M_\tau] \le \mathbb{E}_{P,\infty}[\tau]$ for every stopping time $\tau$ and every $P \in \mathcal{P}$ (their Def. 2.2) — thresholding at $1/\alpha$ controls **Average Run Length** (ARL $\ge 1/\alpha$, equivalently FAR $\le \alpha$) by their Theorem 2.4, *with no i.i.d. assumption on the data stream*. This is exactly the guarantee SENTRY wants: agent action streams are highly dependent (each action conditions on the whole history), and e-detectors are proven valid regardless.
- **SR and CUSUM e-detectors built from sequences of $e_j$-processes** (Def 2.5–2.6): $M^{SR}_n = \sum_{j=1}^n \Lambda^{(j)}_n$, $M^{CU}_n = \max_{j\in[n]} \Lambda^{(j)}_n$ — each $\Lambda^{(j)}$ is an e-process "restarted" at time $j$, so the detector continuously re-tests "has a change occurred *since* some recent time $j$" rather than testing against a single fixed alternative. This solves SENTRY's core requirement: **detect a change in agent behavior at an unknown point in a long, open-ended deployment**, not just certify one bounded trajectory.
- **Baseline increments and exponential baseline e-detectors** (Def 2.8–2.9, Section 4): reduce the update to $O(1)$ recursions ($M^{SR}_n = L_n \cdot (M^{SR}_{n-1}+1)$, $M^{CU}_n = L_n \cdot \max(M^{CU}_{n-1}, 1)$), which is essential for a monitor that must run indefinitely online.
- **Method-of-mixtures betting over unknown post-change parameters** (Section 3–4): exactly the tool needed since SENTRY does not know in advance what kind or magnitude of behavioral drift (prompt injection, goal drift, reward hacking, tool misuse) it is looking for.
- Nonparametric examples directly reusable: **Example 2 (change from exchangeability)** ties to conformal test martingales — the bridge to PRISM's conformal machinery; **Example 7 (change in mean of a bounded/sub-Gaussian variable)** gives a ready-made construction for scalar risk/anomaly scores derived from agent actions.

The design below is SENTRY as **an e-detector (à la Shin–Ramdas–Rinaldo) whose $e_j$-processes are built from a causal-world-model likelihood/conformal nonconformity score, with PAC-style thresholding (à la Sadhuka et al.) applied when that score is learned rather than known.**

---

## 1. Problem setup

An LLM agent operating in deployment produces a single unbounded stream of actions across possibly many tasks: $a_1, a_2, \ldots$ with associated observations $o_1, o_2, \ldots$ and contextual metadata (tool name, arguments, execution result, latency, task/session boundary marker). Let $H_t = (a_1, o_1, \ldots, a_t, o_t)$ denote the history and $X_t$ a feature representation of the $t$-th action extracted from $H_t$ (embedding of the action+arguments, a causal-world-model prediction residual, a conformal nonconformity score — defined in §2).

Unlike E-valuator, there is **no changepoint-free "task end"** at which a success/failure label arrives — SENTRY must be valid *at every point in an indefinitely long stream*, and unlike a single hypothesis test, the question is not "is this whole stream nominal or not" but **"has the agent's behavior departed from nominal at some unknown recent time $\nu$."** This is precisely the e-detector setting of Shin–Ramdas–Rinaldo, §1: pre-change class $\mathcal{P}$ = distributions consistent with safe/nominal agent behavior, post-change class $\mathcal{Q}$ = distributions consistent with drifted/unsafe behavior, changepoint $\nu$ unknown, no i.i.d. assumption.

**Goal.** A stopping rule $N^*$ (raise alarm / halt agent / escalate to human) such that:
$$\text{ARL} := \mathbb{E}_{P,\infty}[N^*] \ge 1/\alpha \quad \forall P \in \mathcal{P} \tag{ARL control}$$
equivalently $\text{FAR} \le \alpha$, holding **nonasymptotically, without i.i.d. assumptions, and under continuous monitoring** (Ville-type guarantee) — and with detection delay after a true drift at $\nu$ as small as possible.

---

## 2. Reference distribution: causal-world-model $e_j$-process construction

This is SENTRY's principal novelty relative to both source papers. Neither E-valuator's labeled-two-class density ratio nor E-detectors' generic pre/post-change classes specify *what* the reference distribution should be for agent behavior. SENTRY instantiates $\mathcal{P}$ via a causal world model (CAIRN-style: learned sparse causal graph over task-state variables + do-interventions), used two ways:

**(a) Predictive-residual score.** Given task context $c$ (the stated goal, available tools, policy constraints) and history $H_{t-1}$, the causal world model predicts a distribution $\hat{P}_\theta(a_t \mid \text{do}(c), H_{t-1})$ over the *next expected action* consistent with pursuing the stated task under the inferred causal structure. Define the per-step nonconformity/surprise score
$$ s_t := -\log \hat{P}_\theta(a_t \mid \text{do}(c), H_{t-1}) $$
Large $s_t$ = the observed action is poorly explained by "pursuing the stated task under normal causal dynamics" — the natural signature of an unstated goal, injected instruction, or reward-hacking deviation, distinct from ordinary task variability which the causal model, having seen do-interventions across many tasks, already absorbs.

**(b) Intervention-consistency score.** Where the world model supports explicit do-surgery (e.g., "if tool X had not been available"), a secondary check tests whether the agent's tool-selection pattern is consistent with the *causal* effect of available interventions on task completion, rather than a spurious correlate (this directly reuses CAIRN's do-calculus layer and gives SENTRY a defense against agents that learn to game a purely correlational monitor).

Both scores are estimated, not known exactly — exactly the situation E-valuator's PAC thresholding was built for. So SENTRY treats $\hat P_\theta$ as a **learned reference**, calibrated on a corpus of trusted/safe trajectories $\mathcal{D}_\text{cal}$, and folds the resulting estimation error into the threshold rather than the martingale (§4).

---

## 3. From scores to $e_j$-processes (baseline increments)

Follow E-detectors §2.4 exactly: define a **baseline increment** $L_n$ satisfying $\sup_{P\in\mathcal P}\mathbb E_{P,\infty}[L_n\mid \mathcal F_{n-1}]\le 1$. Two constructions, matched to the two SENTRY score types:

**Exponential baseline increment (parametric family, fast/efficient).** Following E-detectors Def. 4.1, with $s(\cdot)$ the surprise score and $v(\cdot)$ a variance proxy (e.g. squared score, or a fixed bound if scores are bounded post-clipping):
$$ L_n^{(\lambda)} = \exp\{\lambda\, s(X_n) - \psi(\lambda)\, v(X_n)\}, \qquad \lambda \in \Pi $$
where $\psi$ is chosen so $L^{(\lambda)}$ satisfies the baseline-increment inequality under the pre-change class (e.g. sub-exponential/sub-Gaussian tail assumption on $s_t$ under nominal behavior — realistic since $s_t = -\log \hat P_\theta(a_t\mid\cdot)$ is typically light-tailed when the world model is well-calibrated on nominal data). This directly reuses E-detectors §5.2's bounded/sub-exponential mean-shift construction, treating "nominal mean surprise $\le m$" as $\mathcal P$ and "elevated mean surprise $\ge m+\delta$" as $\mathcal Q$ — i.e. SENTRY inherits a **ready-made, proof-complete instantiation** (E-detectors Prop 4.2, Eq 5.5–5.11) rather than needing new theory for this piece.

**Conformal baseline increment (nonparametric, handles mixed discrete+continuous action features — ties to PRISM).** When action features $X_t$ are heterogeneous (tool-type categorical + continuous arguments + text embedding), avoid needing a tractable joint density for $\hat P_\theta$ by using a **conformal e-value** instead of a raw likelihood: maintain a calibration set of nonconformity scores from nominal trajectories, and at each step compute
$$ L_n = \frac{1}{|\mathcal D_\text{cal}|+1}\sum_{i} \mathbb 1\{\text{nonconf}(X_n) \ge \text{nonconf}(X^{(i)})\}^{-1} \quad\text{(conformal p-value inverted to an e-value via Vovk–Wang calibrator)} $$
This is the E-detectors **Example 2 / "change from exchangeability"** instantiation — nominal behavior corresponds to (a relaxed, causally-adjusted notion of) exchangeability of action-nonconformity scores; drift breaks it. This gives SENTRY a second, assumption-light detector to run in parallel/mixture with the exponential one, and is exactly the intended fusion point with PRISM's conformal-under-topology-uncertainty machinery: the "topology uncertainty" there becomes "action-space/tool-graph uncertainty" here.

**Mixture over both constructions and over unknown drift magnitude.** Since neither the *type* of drift (sudden prompt injection vs. slow goal drift) nor its *magnitude* is known a priori, take
$$ L_n = \frac{1}{K}\sum_{k=1}^K L_n^{(\lambda_k)} \quad\text{(finite mixture, E-detectors §3)} \quad\text{or}\quad L_n^{a} \text{(adaptive/growing mixture, E-detectors §3.1, } K(n)=O(\log n)) $$
using E-detectors' **adaptive reweighting** construction (their Eqs. 3.10–3.13) so the number of active mixture components grows only logarithmically, keeping each update $O(\log n)$ time/space — required for an indefinitely-running production monitor.

---

## 4. PAC-thresholded SR/CUSUM e-detector (the SENTRY statistic)

Because $\hat P_\theta$ (or the conformal calibrator) is **learned** from finite $\mathcal D_\text{cal}$, the resulting $L_n$ is only an *approximate* baseline increment — same issue E-valuator confronts. SENTRY therefore combines E-detectors' SR/CUSUM recursion with E-valuator's PAC-thresholding correction rather than using the exact Ville threshold $1/\alpha$.

### Algorithm SENTRY-Detect

```
Inputs:
  α        target false-alarm rate (ARL ≥ 1/α)
  δ        PAC confidence parameter
  θ̂        fitted causal world model (from D_train)
  D_cal    held-out nominal/safe trajectories, disjoint from D_train
  K(n)     mixture-component schedule (e.g. K(n) = ceil(log n))

Phase 1 — Calibrate scores and baseline increments:
  for each trajectory in D_cal:
      compute the stream of surprise/conformal scores s_1, s_2, ...
      compute exponential + conformal baseline increments L_n(λ_k) per §3
  Fit mixture weights {ω_k} via E-detectors Algorithm 1/2 (computeBaseline),
  using a guessed range (Δ_L, Δ_U) for drift magnitude (elicited from
  red-team/injected-attack severity, e.g. AgentDojo attack strengths).

Phase 2 — PAC threshold (E-valuator Algorithm 1, adapted to e-detector max-over-time):
  for each nominal trajectory i in a second calibration split D_thresh:
      compute the running e-detector M_n^(i) = SR or CUSUM combination of
          the mixture baseline increments (E-detectors Eq. 3.5/3.7)
      record M^(i) = max_n M_n^(i)
  sort {M^(i)}, pick order statistic index k* via the Binomial-tail bound
  (E-valuator Prop. 1 / Algorithm 1) so that
      Pr[ max_n M_n > c_α ] ≤ α  holds with probability ≥ 1-δ over D_thresh
  set c_α ← M^(k*)

Phase 3 — Online deployment monitor:
  M_0^SR ← 0   (or M_0^CU ← 0)
  for each incoming agent action a_t:
      compute X_t, s_t via θ̂ (causal-model surprise) and/or conformal score
      update baseline increments L_t(λ_k) for active k ≤ K(t)
      M_t^SR ← Σ_k ω_k · L_t(k) · (M_{t-1}^SR(k) + 1)      [SR recursion]
      or
      M_t^CU ← Σ_k ω_k · L_t(k) · max(M_{t-1}^CU(k), 1)    [CUSUM recursion]
      if M_t ≥ c_α:
          RAISE ALARM at time t   (flag / halt / escalate to human review)
          reset M ← 0, j ← t+1    (restart detector; ARL guarantee re-applies
                                    from the restart point, cf. E-detectors §6.2
                                    generalized Lorden argument)
      else:
          continue monitoring
```

### Guarantee

By E-detectors Theorem 2.4 applied to the mixture-of-baseline SR/CUSUM e-detector (their Prop. 2.3 + §3), thresholding the *exact* mixture e-detector at $1/\alpha$ controls ARL $\ge 1/\alpha$ under any $P \in \mathcal P$, with **no i.i.d. assumption** on the agent's action stream — the key property that lets SENTRY run over a single agent's entire dependent, multi-turn, multi-tool trajectory. Because the mixture is built from an *estimated* $\hat P_\theta$, exact Ville validity is not guaranteed at $c_\alpha = 1/\alpha$; substituting the PAC threshold from Phase 2 restores a **probabilistic guarantee** analogous to E-valuator's Proposition 1:
$$ \Pr_{\mathcal D_\text{cal}}\Big(\Pr_{H_N}\big[\exists\, t: M_t > c_\alpha\big] \le \alpha\Big) \ge 1-\delta. $$
SENTRY should report *both* numbers explicitly in any writeup (this is a known reviewer pressure point on E-valuator itself): the "PAC-approximate" guarantee under estimation, and the exact-Ville guarantee that would hold if $\hat P_\theta$ were exact — with an ablation (as E-valuator does in their §8.2.3–8.2.4) showing how threshold conservativeness shrinks as $|\mathcal D_\text{cal}|$ grows.

---

## 5. Detection-delay behavior

By E-detectors Theorem 4.3 / Corollary 4.6 (exponential-family baseline case) and Proposition 2.14 (general case), the worst-case average delay of the mixture e-SR/e-CUSUM procedure is bounded by
$$ \max\big(J_L(N^*_{mSR}), J_L(N^*_{mCU})\big) \;\le\; \frac{g_\alpha}{D(Q\Vert P)} + \frac{\mathbb V_{0,Q}[\log L_1^{(\lambda_{op})}]}{[D(Q\Vert P)]^2} + 1 $$
where $D(Q\Vert P)$ is the KL-type separation between nominal and drifted score distributions achievable by the mixture, and $g_\alpha = O(\log(1/\alpha))$. Practically: **the tighter the causal world model's surprise score separates nominal from drifted behavior (larger $D(Q\Vert P)$), the faster SENTRY detects** — this is the direct, provable payoff of using a causal reference rather than a generic anomaly score, and gives SENTRY a natural ablation axis (causal-world-model score vs. a plain embedding-distance baseline) to report empirically, exactly mirroring how E-detectors validate detection-delay theory against the Cavaliers case study and E-valuator validates power/ARL against baselines.

---

## 6. Explicit differentiation from E-valuator (must appear in the paper's Related Work)

| | E-valuator | SENTRY |
|---|---|---|
| Null hypothesis | trajectory will **succeed** ($Y=1$) | behavior is **safe/nominal** (causal-consistent) |
| Alternative | trajectory will **fail** ($Y=0$), two labeled classes | drift/attack/unstated-goal, **open-ended, unlabeled** |
| Reference distribution | empirical density ratio from labeled $(S,Y)$ pairs | causal world-model prediction + conformal nonconformity |
| Test statistic | single density-ratio e-process (SPRT-style) | **e-detector** (SR/CUSUM sum/max of restarted $e_j$-processes) |
| Horizon | one task, stops at task end | **continuous deployment**, restarts after each alarm |
| Thresholding | PAC thresholding (Algorithm 1) | PAC thresholding **adapted to e-detector max-over-restarts**, plus exact-Ville comparison |
| Change-point framing | none (fixed alternative) | explicit ARL / detection-delay framing (Shin–Ramdas–Rinaldo) |
| Data dependence | assumes calibration trajectories i.i.d. from $P_1$ | **no i.i.d. assumption** on the deployed stream (only on $\mathcal D_\text{cal}$) |

SENTRY's Related Work should cite E-valuator as the closest prior instantiation of "e-processes for agent trajectories," explicitly note the shared PAC-thresholding tool, and state that E-valuator's own Discussion flags "early detection of unsafe or harmful trajectories" as unrealized future work — which SENTRY undertakes with a materially different (causal, continuous, change-point) formulation rather than a direct extension of their two-class construction.

---

## 7. Evaluation plan (unchanged from prior design, now concretized)

1. **ARL/FAR validation** on pure nominal streams (τ-bench, WebArena, AgentDojo benign trajectories) — reproduce the E-valuator-style plot (empirical FAR vs. nominal $\alpha$) but as a function of *stream length*, not per-task, to demonstrate ARL control holds under continuous monitoring (something E-valuator's per-task design cannot show).
2. **Detection delay** on AgentDojo/InjecAgent attacks and TrajAD anomalous trajectories injected mid-stream at varying $\nu$ — report worst-case and average delay, replicate E-detectors' Cavaliers-style visualization (log e-detector value vs. step, with the $\log(1/\alpha)$ threshold line) as the primary figure.
3. **Ablation**: causal-world-model surprise score vs. (a) plain LLM-judge score, (b) embedding-distance score, (c) E-valuator's own two-class density ratio repurposed as a drift detector — isolates the value of the causal reference specifically.
4. **Baseline comparison**: NeMo Guardrails / LLM-as-judge threshold alarms (no ARL guarantee) vs. SENTRY, holding empirical power comparable, to show guarantee-for-power tradeoff is favorable.
5. **Compute cost**: report per-step update time to confirm the $O(\log n)$ adaptive-mixture cost (E-detectors §3.1) is practical for production latency budgets, echoing E-valuator's "runs in under a minute on a standard laptop" framing.

---

## 8. Immediate next steps

1. Decide whether the exponential or conformal baseline increment is primary for the first paper draft (recommend: **exponential as the main theoretical result** — full delay bounds available via E-detectors §4 directly — with **conformal as the mixed-feature extension** in a later section/companion paper tying to PRISM).
2. Pin down the causal world model's training data source: either synthetic do-intervention trajectories from a controlled agent harness, or SmartOSC deployment logs (anonymized) if accessible for the "real production trace" advantage flagged in the SOTA survey.
3. Implement Phase 1–2 calibration first against a synthetic Bernoulli/Gaussian toy stream (replicate E-detectors' Cavaliers example structurally) to validate the ARL guarantee end-to-end before wiring in the LLM agent harness — this gives a fast correctness check independent of agent/LLM variance.
