# SENTRY State-of-the-Art Survey: Anytime-Valid E-Process Monitoring of LLM Agent Action Streams

## TL;DR
- **The core idea is defensibly novel but the neighborhood is closing fast.** No published work applies e-processes/test martingales to LLM agent action streams *for safety/drift detection against a safe-behavior (or causal-world-model) reference distribution*. The single closest competitor, E-valuator (arXiv:2512.03109, revised v2 dated 28 May 2026), applies e-processes/density-ratio martingales to agent trajectories but for **task-success** verification, and it explicitly lists early detection of unsafe or harmful trajectories as *unrealized future work* — SENTRY should treat it as the primary baseline.
- **The causal-world-model-as-reference-distribution angle (tying to CAIRN) is the least-contested, most defensible contribution.** Causal world models of agents exist (DeepMind's "Causal Analysis of Agent Behavior") and e-process monitors exist, but no paper combines a do-calculus/SCM-predicted expected trajectory as the test statistic inside an anytime-valid monitor.
- **Foundations, tooling, and benchmarks are mature and reusable:** the SAVI/e-process theory (Ramdas, Grünwald, Vovk, Waudby-Smith), conformal test martingales (Vovk), e-detectors (Shin–Ramdas–Rinaldo), conformal e-values (Vovk; Gauthier–Bach–Jordan), the `safestats` R package and betting/WSR code, and agent-safety benchmarks (AgentDojo, AgentHarm, τ-bench, TrajAD) all provide immediate building blocks.

## Key Findings

**1. The exact gap is real but narrow.** "E-process + agent monitoring" as a generic primitive is now actively occupied (E-valuator; Gauthier–Bach–Jordan on multi-agent equilibrium; LLM-eval-drift and AI-audit papers). SENTRY must differentiate sharply on: (a) null = *safe/nominal behavior* vs. task-correct; (b) *drift/adversarial* threat model; (c) *reference distribution* (empirical safe trajectories or causal world model) vs. labeled two-class density ratio; (d) *continuous long-horizon* deployment monitoring vs. per-task early-termination wrapper.

**2. The anytime-valid statistical foundations are solid and well-cited.** Ville's inequality, test (super)martingales, e-values/e-processes, and testing-by-betting give exactly the Type-I-error-under-optional-stopping guarantee SENTRY promises.

**3. Change-point detection has a game-theoretic reformulation (e-detectors) that is the natural technical backbone** for detecting *when* an agent's behavior changes, with nonasymptotic average-run-length (false-alarm) and detection-delay bounds.

**4. Conformal test martingales and conformal e-values solve the mixed discrete/continuous, non-exchangeable data problem** (tying to PRISM), and weighted CTMs (WATCH) already handle benign vs. harmful shift adaptation.

**5. Causal/world-model approaches to agent behavior and reward-hacking/goal-misgeneralization detection exist but lack anytime-valid guarantees** — the fusion is open.

## Details

### Area 1 — Anytime-valid inference / testing-by-betting foundations
The canonical framework: an **e-value** is a nonnegative random variable E with E[E] ≤ 1 under the null; an **e-process** is a sequence (E_1, E_2, …) such that stopping at any (even data-dependent) time yields a valid e-value. Test martingales are nonnegative supermartingales starting at 1. The master guarantee is **Ville's inequality (1939)**: P_H0(∃ n : E_n ≥ 1/α) ≤ α — the analog of alpha-spending but valid for *every* stopping rule simultaneously.

Foundational and survey references SENTRY must cite:
- Ramdas, Grünwald, Vovk & Shafer, "Game-theoretic statistics and safe anytime-valid inference," *Statistical Science* 38(4):576–601, 2023, DOI 10.1214/23-STS894 (arXiv:2210.01948) — the central SAVI survey.
- Ramdas & Wang, "Hypothesis testing with e-values," *Foundations and Trends in Statistics* 1(1-2):1–390, 2025 (book-length treatment).
- Grünwald, de Heide & Koolen, "Safe testing," *JRSS-B*, 2024 — safe tests / GROW e-variables.
- Vovk & Wang, "E-values: calibration, combination and applications," *Annals of Statistics* 49(3):1736–1754, 2021.
- Shafer, "Testing by betting," *JRSS-A* 184(2):407–431, 2021; Shafer, Shen, Vereshchagin & Vovk, "Test martingales, Bayes factors and p-values," *Statistical Science* 26(1):84–101, 2011.
- Waudby-Smith & Ramdas, "Estimating means of bounded random variables by betting," *JRSS-B* 86(1):1–27, 2024 (arXiv:2010.09686) — the hedged-capital/betting confidence sequences, variance-adaptive, state-of-the-art for bounded means; provides the "WSR bounds."
- Ramdas, Ruf, Larsson & Koolen, "Admissible anytime-valid sequential inference must rely on nonnegative martingales" (arXiv:2009.03167) — universality result.
- Howard, Ramdas, McAuliffe & Sekhon on time-uniform Chernoff bounds / confidence sequences.

For **unknown drift**, the relevant machinery is *mixture / method-of-mixtures* betting (Robbins; Kaufmann & Koolen, "Mixture martingales revisited," *JMLR* 22(246), 2021) and *universal portfolio* betting strategies — you bet across a family of alternatives so the wealth process grows under any drift in the family, avoiding needing to know the post-change distribution.

### Area 2 — Sequential/streaming anomaly & change-point detection
Classical: CUSUM (Page 1954), Shiryaev–Roberts (Shiryaev 1961/1963), and GLR (Siegmund & Venkatraman 1995) — parametric, mostly exponential-family, optimal detection-delay under known pre/post distributions. Modern deep learning-based streaming anomaly detection (autoencoders, density estimation) lacks Type-I-error/false-alarm guarantees.

The bridge is **e-detectors**: Shin, Ramdas & Rinaldo, "E-detectors: a nonparametric framework for sequential change detection," *New England Journal of Statistics in Data Science* 2(2):229–260, 2024, DOI 10.51387/23-NEJSDS51 (arXiv:2203.03532). An e-detector is a sum of e-processes started at consecutive times (Shiryaev-Roberts- and CUSUM-style), and the paper's procedures "come with clean, nonasymptotic bounds on the average run length (frequency of false alarms)" and "near-optimal bounds on the detection delay" in sub-Gaussian/sub-exponential cases — for *nonparametric, composite* pre/post-change families. Related: Shekhar & Ramdas, "Sequential change detection via backward confidence sequences," ICML 2023. This is the statistically-guaranteed alternative to CUSUM/GLR/deep anomaly scores and the natural backbone for SENTRY's "when did behavior change" question.

### Area 3 — LLM agent safety monitoring (state of the art + novelty gap)
Current practice falls in tiers, none of which give anytime-valid statistical guarantees:
- **Guardrail systems / policy enforcement:** NVIDIA NeMo Guardrails (Colang DSL), Guardrails AI (RAIL specs), Meta Llama Guard / Prompt-Guard, LlamaFirewall, Rebuff, LLM Guard, Lakera. These are rule/classifier-based input-output filters — a systematic review is "Safeguarding large language models: a survey," *Artificial Intelligence Review*, 2025.
- **LLM-as-judge / process reward models / observability:** MLflow, Langfuse, Galileo, Braintrust production monitoring; threshold-based drift alerts.
- **Agent trajectory anomaly detection:** TraceAegis (provenance graphs), TrajAD ("Trajectory Anomaly Detection for Trustworthy LLM Agents," arXiv:2602.06443), MindGuard (attention/decision-dependence graphs), SentinelAgent, Agentproof (static verification of agent workflow graphs), and content-aware tool-call attack detectors.
- **Runtime enforcement:** customizable runtime enforcement frameworks intercepting unsafe code/tool actions.
- **Reasoning-trace monitoring:** OpenAI's "Monitoring Reasoning Models for Misbehavior…" (arXiv:2503.11926) uses CoT monitors.

**Statistical/anytime-valid approaches applied to AI monitoring (adjacent, not agent-safety):** Vovk et al. conformal test martingales for retraining; Podkopaev & Ramdas "Tracking the risk of a deployed model"; Prinster, Han, Liu & Saria, "WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales," ICML 2025 (arXiv:2505.04608); "Adaptive auditing of AI systems with anytime-valid guarantees" (arXiv:2605.07002, which itself states "there are currently no e-value frameworks for highly-targeted and adaptive audits of AI systems"); "Who Drifted: the System or the Judge?" (arXiv:2606.15474); "Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained LLMs" (arXiv:2605.20270).

**The single closest competitor — E-valuator** (Sadhuka, Prinster, Fannjiang, Scalia, Berger, Regev, Wang; Genentech/MIT/Johns Hopkins/Stanford; arXiv:2512.03109, revised v2 dated 28 May 2026; code at github.com/shuvom-s/e-valuator, PyPI `e-valuator`). It converts any black-box verifier score into a decision rule with false-alarm-rate control, framing "successful vs. unsuccessful trajectory" as sequential hypothesis testing. Its null and guarantee are stated verbatim as "H_N : S ∼ P_1 (the final output is correct); H_A : S ∼ P_0 (the final output is incorrect)" with false-alarm control "Pr_{H_N}[∃ t ∈ [T] : M_t > c_α] ≤ α." Method: an e-process instantiated with a **log-optimal density-ratio (SPRT-style) test statistic** M_t = p_0(S[1:t])/p_1(S[1:t]), estimated via classifier-based density-ratio estimation, with a novel **PAC thresholding** procedure (because estimated densities break exact Ville thresholds). It does **not** address safety/adversarial drift — its Discussion states verbatim, "Finally, e-valuator can be used for other applications, such as early detection of unsafe or harmful trajectories" (i.e., unrealized future work) — requires per-task success/failure labels, and is a per-task early-termination wrapper. It is evaluated on 6 datasets × 3 agent/verifier combos: GSM8k→Aviary+Claude Haiku 3.5; HotpotQA→Aviary+Claude Haiku 3.5; MedQA→OctoTools+Claude Haiku 3.5; MATH/AIME/MMLU-Pro→Claude Sonnet 4+Pretrained PRM. **SENTRY differs on null (safe vs. correct), threat model (drift/adversarial), reference distribution (safe trajectories or causal world model vs. labeled two-class ratio), and horizon (continuous deployment vs. per-task).**

Second-closest: Gauthier, Bach & Jordan, "Anytime Detection of Strategic Deviations in Multi-Agent Systems" (arXiv:2601.05427) — testing-by-betting supermartingales + e-BH for detecting equilibrium deviation in repeated/stochastic games. Not LLM agents; reference is a game-theoretic equilibrium/target policy.

### Area 4 — Conformal prediction for sequential/non-exchangeable data & LLMs
- **Adaptive/online CP under shift:** Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift," NeurIPS 2021 (arXiv:2106.00170); Gibbs & Candès, "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts," 2022 (arXiv:2208.08401); Barber, Candès, Ramdas & Tibshirani, "Conformal prediction beyond exchangeability," 2023; conformal PID control (Angelopoulos et al.); Bhatnagar et al. SAOCP.
- **Conformal test martingales & conformal e-values:** Vovk et al. "Testing exchangeability online"; Vovk et al. "Retrain or not retrain," COPA 2021 (arXiv:2102.10439); Ramdas, Ruf, Larsson & Koolen, "Testing exchangeability: fork-convexity, supermartingales and e-processes" (arXiv:2102.00630); Vovk, "Conformal e-prediction," *Pattern Recognition*, 2025 (arXiv:2001.05989); Gauthier, Bach & Jordan, "E-Values Expand the Scope of Conformal Prediction" (arXiv:2503.13050, giving *batch anytime-valid* conformal prediction). Wang & Ramdas, "False discovery rate control with e-values," *JRSS-B* 84(3):822–852, 2022.
- **CP for LLMs:** Angelopoulos & Bates tutorial + Angelopoulos, Barber & Bates "Theoretical Foundations of Conformal Prediction" (arXiv:2411.11824); Quach et al. "Conformal Language Modeling"; Mohri & Hashimoto "Language models with conformal factuality guarantees"; Cherian, Gibbs & Candès "LLM validity via enhanced conformal prediction" (NeurIPS 2024); Yadkori et al. "Mitigating LLM hallucinations via conformal abstention" (arXiv:2405.01563); Ulmer et al. "Non-exchangeable conformal language generation with nearest neighbors"; Kumar et al. MCQA. Survey: "Conformal Prediction for NLP: A Survey," *TACL*.

### Area 5 — Causal world models for agent behavior / goal inference
- Deletang, Grau-Moya, Martic, Genewein, McGrath, Mikulik, Kunesch, Legg & Ortega, "Causal Analysis of Agent Behavior for AI Safety" (arXiv:2103.03938, DeepMind) — SCMs + do-interventions ("Agent Debugger") to infer agent causal models. No sequential testing.
- "The Limits of Predicting Agents from Behaviour" (arXiv:2506.02923) — bounds on predicting agent actions OOD from a well-specified SCM/world model.
- "Language Agents Meet Causality — Bridging LLMs and Causal World Models" (Gkountouras et al., 2024).
- Reward-hacking / goal-misgeneralization detection: Amodei et al. "Concrete Problems in AI Safety"; Di Langosco et al. "Goal misgeneralization in deep RL" (2022); Skalse et al. reward-hacking; Everitt et al. reward tampering; OpenAI CoT-monitoring (arXiv:2503.11926); "Natural Emergent Misalignment from Reward Hacking in Production RL" (arXiv:2511.18397). None use anytime-valid e-processes — the causal-reference + e-process fusion is unoccupied.

### Area 6 — Benchmarks & datasets
- **Adversarial / prompt-injection agent benchmarks:** AgentDojo (Debenedetti et al., arXiv:2406.13352, NeurIPS 2024 Datasets & Benchmarks) — "97 realistic tasks (e.g., managing an email client, navigating an e-banking website, or making travel bookings), 629 security test cases" across four environments (Workspace, Slack, Travel, Banking), with benign-utility, utility-under-attack, and attack-success-rate metrics; available in Inspect Evals. InjecAgent; AgentHarm (Andriushchenko et al., ICLR 2025); BIPIA; OS-Harm (computer-use agents); AgentDyn (dynamic AgentDojo extension).
- **Trajectory-anomaly datasets:** TrajAD (contrasts normal vs. anomalous trajectories — infinite loops, redundant/invalid tool calls); ATBench-Claw / ATBench-Codex (trajectory-level safety diagnosis). Code-safety: RedCode-Exec; SafeAgentBench (embodied).
- **Tool-use / commerce agent benchmarks (safe baselines, injectable anomalies):** τ-bench / τ²-bench (Sierra; retail/airline/telecom, policy-compliant tool use, pass^k reliability metric); WebShop, Mind2Web/Online-Mind2Web, WebArena, WebMall, ShoppingBench, EComAgentBench, Amazon-Bench (safety focus); AgentBench; SWE-bench (coding); GAIA; OSWorld.
SENTRY can build safe/unsafe evaluation streams by injecting AgentDojo/InjecAgent attacks into τ-bench/WebArena trajectories or using TrajAD's paired normal/anomalous trajectories.

### Area 7 — Target venues & competitive landscape
- **Venues:** NeurIPS, ICML, ICLR main tracks; ICLR 2026 "Agents in the Wild: Safety, Security, and Beyond" (AIWILD) workshop; ICML 2026 AIWILD; SAVI/game-theoretic statistics venues; COPA (Conformal and Probabilistic Prediction); AAAI (safety/guardrails). AISTATS/JRSS-B for the theory.
- **Most relevant recent competitors to differentiate from:** E-valuator (2512.03109); WATCH (2505.04608); Gauthier–Bach–Jordan multi-agent (2601.05427); "Adaptive auditing of AI systems with anytime-valid guarantees" (2605.07002); "Who Drifted…" (2606.15474); "Conformal Selective Acting" (2605.20270); Podkopaev & Ramdas risk tracking.

## Synthesis

**(a) Is the core idea a novel, unoccupied gap?** *Yes, with an important qualifier.* The generic primitive "anytime-valid/e-process sequential monitoring of AI/agent behavior" is now actively occupied (E-valuator, Gauthier–Bach–Jordan, WATCH, the audit/drift papers), so a reviewer will see a crowded neighborhood and demand sharp differentiation. But the *specific* SENTRY thesis — anytime-valid e-process monitoring of LLM **agent action streams for safety/drift** (null = safe/nominal behavior, adversarial/drift threat model, reference distribution of safe trajectories or a **causal-world-model do-calculus prediction**), for continuous long-horizon deployment — is unoccupied. The causal-world-model-as-reference-distribution angle (tying to CAIRN) is the single strongest, least-contested novelty claim; the conformal-e-value angle for mixed discrete/continuous action features (tying to PRISM) is a strong secondary contribution.

**(b) The 5–10 most important papers to cite and differentiate from:**
1. Sadhuka et al., **E-valuator** (arXiv:2512.03109, v2 2026) — primary baseline; differentiate on safety-vs-success null.
2. Prinster et al., **WATCH** (ICML 2025, arXiv:2505.04608) — closest CTM-monitoring method.
3. Shin, Ramdas & Rinaldo, **E-detectors** (NEJSDS 2(2):229–260, 2024, arXiv:2203.03532) — change-detection backbone.
4. Ramdas, Grünwald, Vovk & Shafer, **Game-theoretic statistics & SAVI** (Statistical Science 38(4):576–601, 2023) — foundational framing.
5. Waudby-Smith & Ramdas, **Estimating means by betting** (JRSS-B 86(1), 2024) — betting confidence sequences / test statistic.
6. Vovk et al., **Retrain or not retrain: conformal test martingales** (COPA 2021, arXiv:2102.10439) — CTM origin for monitoring.
7. Gauthier, Bach & Jordan, **E-Values Expand the Scope of Conformal Prediction** (arXiv:2503.13050) — conformal e-values for mixed data.
8. Gibbs & Candès, **Adaptive Conformal Inference** (NeurIPS 2021, arXiv:2106.00170) — non-exchangeable/streaming CP.
9. Deletang et al., **Causal Analysis of Agent Behavior for AI Safety** (arXiv:2103.03938) — causal-reference foundation.
10. Debenedetti et al., **AgentDojo** (arXiv:2406.13352, NeurIPS 2024 D&B) — primary evaluation benchmark.

**(c) Technical prerequisites & reusable tooling:**
- **Theory prerequisites:** nonnegative supermartingales & Ville's inequality; e-value calibration/combination (arithmetic-mean merging is always valid); mixture/universal-portfolio betting for composite alternatives; conformal p-values → e-values → exchangeability martingales; e-detector construction (sum of e-processes) for change-point/ARL control; for the causal reference, structural causal models + do-calculus and a learned world model.
- **Open-source tooling:** `safestats` (R; safe tests incl. safe logrank); Waudby-Smith/Ramdas betting & WSR code (Python, e.g. `confseq`); `e-valuator` (PyPI + github.com/shuvom-s/e-valuator); conformal libraries (MAPIE, TorchCP, `crepes`, Angelopoulos' conformal code); AgentDojo (via Inspect Evals), AgentHarm, τ-bench, TrajAD, WebArena for data; NeMo Guardrails / Guardrails AI as integration targets and non-guaranteed baselines; DeepMind `agent_debugger` for the causal-analysis component.

## Recommendations
1. **Position against E-valuator as the explicit primary baseline** and lead with the safety/drift null and the causal-world-model reference distribution; this is the cleanest differentiation and directly extends CAIRN and VOUCH.
2. **Build the change-detection core on e-detectors** (Shin–Ramdas–Rinaldo) for ARL/detection-delay guarantees, with mixture betting for unknown drift.
3. **Use conformal e-values / weighted CTMs** to handle mixed discrete+continuous action features and relax exchangeability (ties to PRISM); benchmark against WATCH.
4. **Evaluate on injected-attack streams**: AgentDojo/InjecAgent attacks embedded in τ-bench/WebArena/agentic-commerce trajectories, plus TrajAD paired normal/anomalous data; report false-alarm rate (ARL₀), detection delay, and power, comparing to guardrail/LLM-judge baselines that lack guarantees.
5. **Target ICLR/ICML AIWILD workshops for early visibility, then a main-track submission.** Move quickly — several directly adjacent papers are dated mid-2026, so the gap is closing.
- *Benchmarks/thresholds that change the plan:* if a mid-2026 paper appears combining causal/world-model references with anytime-valid agent monitoring, pivot emphasis to the conformal-e-value mixed-feature contribution and empirical scale; if E-valuator's authors release a safety-focused extension, differentiate on the causal reference and continuous (non-per-task) monitoring.

## Caveats
- **Fast-moving area:** many key references are dated 2025–2026 (E-valuator revised May 2026; Gauthier–Bach–Jordan and several audit/drift papers mid-2026); the novelty window is narrowing and some 2026 arXiv IDs were verified via abstract/HTML rather than full PDF.
- **PDFs to download manually:** the full E-valuator PDF (arXiv:2512.03109) and the E-detectors journal version (NEJSDS, DOI 10.51387/23-NEJSDS51) are worth pulling in full for exact method/threshold details; the JRSS-B Waudby-Smith–Ramdas paper is paywalled (arXiv:2010.09686 is the open version).
- **Guarantee nuance:** E-valuator relies on PAC thresholding rather than exact Ville thresholds because its densities are estimated — SENTRY should be explicit about whether its guarantees are exact (Ville) or estimation-dependent, since this is a likely reviewer pressure point.
- Commercial LLM-monitoring/guardrail claims (Galileo, NeMo, etc.) are vendor marketing and are cited only as evidence of the non-guaranteed status quo, not as validated performance.
