"""Causal-world-model surprise score, algorithm.md §2.

This module implements a *lightweight, learnable* stand-in for the
CAIRN-style sparse causal graph + do-calculus world model that algorithm.md
§2 specifies as SENTRY's reference distribution. A full causal-discovery
engine (sparse structure learning over task-state variables, do-surgery
consistency across arbitrary interventions) is a separate research system
(CAIRN) and out of scope here; what's implemented is the *interface and the
two score types* §2 calls for, backed by a simple, honestly-scoped
predictive model:

  * tool choice given context: categorical, Laplace-smoothed MLE
  * continuous action features given the chosen tool: diagonal Gaussian MLE

Any object exposing ``.surprise(context, history, action) -> float`` can be
substituted as the world model without touching the rest of the pipeline
(sentry.pipeline only depends on that interface), so a real CAIRN model can
be dropped in later per algorithm.md §8 "next steps".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np

_EPS = 1e-9
_MAX_SURPRISE = 50.0  # clip -log(0) to a large finite value

# --- Instruction-likeness of a tool observation (algorithm.md §2, extended) --
#
# A tool result is supposed to be passive *data*. Indirect prompt injection
# works by smuggling an *instruction* into that data. So instruction-like
# content appearing in a tool observation is, under a nominal reference where
# tool outputs are data, a high-surprise event -- and it fires at the
# observation, before the agent acts on it (near-zero detection delay).
#
# The lexicon below is deliberately GENERIC (imperative verbs, second-person
# address, delimiter/tag mimicry, exclamations) -- no proper nouns, no
# benchmark-specific phrases -- so it is not overfit to AgentDojo's particular
# attack template. Measured AUROC (successful-attack vs benign observations)
# on real collected data was 0.955 with these a-priori weights; the CommandSans
# result (arXiv:2510.08829, AgentDojo attack-success 34%->3%) validates the
# general approach of detecting instructions inside tool outputs.
_IMPERATIVE_VERBS = re.compile(
    r"\b(please|send|ignore|forward|delete|do|make|ensure|note|remember|must|should|"
    r"now|first|before|after|reply|share|provide|include|update|create|add|remove|"
    r"transfer|pay|buy|book|post|invite|execute|run|call|fetch|download|upload)\b"
)
_SECOND_PERSON = re.compile(r"\b(you|your|yours)\b")
_PSEUDO_TAG = re.compile(r"<[a-zA-Z/][^>]{0,40}>")
_WORD = re.compile(r"[a-z']+")
# a-priori weights on [imperative density, second-person density, tag count,
# exclamation density]; not fit to labels.
_INSTR_WEIGHTS = np.array([3.0, 2.0, 2.0, 1.0])


def instruction_likeness(text: str) -> float:
    """Generic instruction-likeness of a text (higher = more instruction-like,
    ~0 for passive data). Model-free, domain-independent, cheap."""
    if not text:
        return 0.0
    tl = text.lower()
    words = _WORD.findall(tl)
    n = max(len(words), 1)
    feats = np.array(
        [
            len(_IMPERATIVE_VERBS.findall(tl)) / n,
            len(_SECOND_PERSON.findall(tl)) / n,
            float(len(_PSEUDO_TAG.findall(text))),
            text.count("!") / n,
        ]
    )
    return float(_INSTR_WEIGHTS @ feats)


@dataclass(frozen=True)
class Context:
    """Task context c: the stated goal and the tools available to pursue it."""

    task_id: Hashable
    available_tools: frozenset[str]


@dataclass(frozen=True)
class Action:
    """A single agent action a_t: which tool was invoked, a continuous
    feature vector summarizing its arguments (e.g. an embedding), and the
    raw argument tokens (for vocabulary/novelty-based scoring, which needs
    exact values -- a hash of "US133000000121212121212" can collide with
    anything; the token itself can be checked against training data)."""

    tool: str
    features: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tokens: tuple[str, ...] = ()
    obs_instruction_likeness: float = 0.0
    """instruction_likeness() of the tool observation immediately preceding
    this action -- the injection signal, computed by the log adapter which
    owns the raw observation text (algorithm.md §2's H_{t-1} conditioning)."""


Trajectory = Sequence[tuple[Context, Action]]


class CausalWorldModel:
    """Predictive-residual score (§2a) and intervention-consistency score
    (§2b), P-hat_theta(a_t | do(c), H_{t-1}), fit by MLE on trusted/safe
    trajectories D_train.
    """

    def __init__(self, laplace_alpha: float = 1.0, min_std: float = 1e-3) -> None:
        self.laplace_alpha = laplace_alpha
        self.min_std = min_std
        self._tool_counts: dict[Hashable, dict[str, float]] = {}
        self._tool_totals: dict[Hashable, float] = {}
        self._all_tools: set[str] = set()
        self._feature_mean: dict[str, np.ndarray] = {}
        self._feature_std: dict[str, np.ndarray] = {}

    def fit(self, trajectories: Sequence[Trajectory]) -> "CausalWorldModel":
        per_tool_features: dict[str, list[np.ndarray]] = {}
        for traj in trajectories:
            for context, action in traj:
                self._all_tools.add(action.tool)
                counts = self._tool_counts.setdefault(context.task_id, {})
                counts[action.tool] = counts.get(action.tool, 0.0) + 1.0
                self._tool_totals[context.task_id] = self._tool_totals.get(context.task_id, 0.0) + 1.0
                per_tool_features.setdefault(action.tool, []).append(action.features)

        for tool, feats in per_tool_features.items():
            arr = np.stack(feats, axis=0) if feats else np.zeros((0, 0))
            if arr.size == 0:
                self._feature_mean[tool] = np.zeros(0)
                self._feature_std[tool] = np.zeros(0)
            else:
                self._feature_mean[tool] = arr.mean(axis=0)
                self._feature_std[tool] = np.maximum(arr.std(axis=0), self.min_std)
        return self

    def tool_log_prob(self, context: Context, tool: str, exclude: frozenset[str] = frozenset()) -> float:
        """log P-hat(tool | do(c)); ``exclude`` implements a do-intervention
        that removes tools from availability -- P is exactly 0 (clipped) for
        any excluded tool, and renormalized over the remaining support."""
        if tool in exclude:
            return -_MAX_SURPRISE
        counts = self._tool_counts.get(context.task_id, {})
        candidate_tools = (self._all_tools & context.available_tools) - exclude
        if not candidate_tools:
            candidate_tools = self._all_tools - exclude or {tool}
        alpha = self.laplace_alpha
        denom = sum(counts.get(t, 0.0) for t in candidate_tools) + alpha * len(candidate_tools)
        numer = counts.get(tool, 0.0) + alpha
        return math.log(numer / denom) if denom > 0 else -_MAX_SURPRISE

    def feature_log_prob(self, tool: str, features: np.ndarray) -> float:
        mean = self._feature_mean.get(tool)
        std = self._feature_std.get(tool)
        if mean is None or mean.size == 0 or features.size == 0:
            return 0.0
        var = std ** 2
        log_prob = -0.5 * np.sum(
            np.log(2 * math.pi * var) + (features - mean) ** 2 / var
        )
        return float(log_prob)

    def surprise(self, context: Context, history: Trajectory, action: Action) -> float:
        """s_t = -log P-hat_theta(a_t | do(c), H_{t-1}), algorithm.md §2(a).
        The simplified model conditions on context+tool rather than the full
        causal graph over H_{t-1}; substitute a richer world model via the
        same interface for full history-dependence."""
        log_p = self.tool_log_prob(context, action.tool) + self.feature_log_prob(action.tool, action.features)
        return float(min(-log_p, _MAX_SURPRISE))

    def intervention_consistency_score(
        self, context: Context, action: Action, do_remove: frozenset[str]
    ) -> float:
        """§2(b): tests whether the agent's tool choice is consistent with
        the *causal* effect of tool availability, via the KL-type gap
        between the factual and the do(remove tools)-intervened tool
        distribution restricted to the tools left after intervention. Large
        gap = the agent's tool preference among the *remaining* tools shifts
        when an unrelated tool is removed -- a signature of a spurious
        (non-causal) correlation the do-surgery layer is meant to catch."""
        remaining = (self._all_tools & context.available_tools) - do_remove
        if action.tool in do_remove or not remaining:
            return _MAX_SURPRISE
        gap = 0.0
        for t in remaining:
            p_factual = math.exp(self.tool_log_prob(context, t))
            p_intervened = math.exp(self.tool_log_prob(context, t, exclude=do_remove))
            if p_factual > _EPS:
                gap += p_factual * math.log((p_factual + _EPS) / (p_intervened + _EPS))
        return float(max(gap, 0.0))


_START_TOOL = "<START>"


class SequentialWorldModel:
    """Tool-transition + argument-novelty surprise score.

    Fixes two failure modes of CausalWorldModel measured on real AgentDojo
    data (see real_data/RESULTS.md):

      1. CausalWorldModel conditions tool probabilities on ``task_id``,
         which is *always unseen* for held-out and attack trajectories, so
         its tool term degenerates to near-uniform exactly where it is
         evaluated. This model conditions on the previous tool instead
         (Laplace-smoothed bigram over tool transitions, pooled across all
         training trajectories), which transfers across tasks: "read_file
         is usually followed by get_balance, not send_money" is task-
         independent structure.
      2. Hashed argument features can only surface novelty via accidental
         collision statistics. This model keeps a per-tool vocabulary of
         argument tokens seen during training and scores the *fraction of
         unseen tokens* directly -- an injected recipient IBAN or attacker
         URL is, by construction, absent from nominal training data, which
         is precisely the signature of a hijacked action.

    surprise(a_t) = -log P(tool_t | tool_{t-1}) + novelty_weight * unseen_fraction(tokens_t)

    This remains a learned reference in the sense of algorithm.md §2/§4 --
    the PAC thresholding downstream is what absorbs its estimation error.
    """

    def __init__(
        self,
        laplace_alpha: float = 1.0,
        novelty_weight: float = 4.0,
        instruction_weight: float = 4.0,
    ) -> None:
        self.laplace_alpha = laplace_alpha
        self.novelty_weight = novelty_weight
        self.instruction_weight = instruction_weight
        self._bigram: dict[str, dict[str, float]] = {}
        self._tools: set[str] = set()
        self._tool_vocab: dict[str, set[str]] = {}
        self._global_vocab: set[str] = set()
        self._instr_baseline: float = 0.0

    def fit(self, trajectories: Sequence[Trajectory]) -> "SequentialWorldModel":
        instr_vals: list[float] = []
        for traj in trajectories:
            prev = _START_TOOL
            for _context, action in traj:
                self._tools.add(action.tool)
                row = self._bigram.setdefault(prev, {})
                row[action.tool] = row.get(action.tool, 0.0) + 1.0
                vocab = self._tool_vocab.setdefault(action.tool, set())
                vocab.update(action.tokens)
                self._global_vocab.update(action.tokens)
                instr_vals.append(action.obs_instruction_likeness)
                prev = action.tool
        # Learn the nominal instruction-likeness ceiling: any observation more
        # instruction-like than anything seen in nominal data is the signal.
        # Learned (not hardcoded) so the term self-calibrates to a deployment
        # whose tools legitimately return more directive-sounding text.
        self._instr_baseline = max(instr_vals) if instr_vals else 0.0
        return self

    def tool_log_prob(self, prev_tool: str, tool: str) -> float:
        counts = self._bigram.get(prev_tool, {})
        support = self._tools | {tool}
        alpha = self.laplace_alpha
        denom = sum(counts.values()) + alpha * len(support)
        numer = counts.get(tool, 0.0) + alpha
        return math.log(numer / denom) if denom > 0 else -_MAX_SURPRISE

    def novelty_fraction(self, action: Action) -> float:
        """Fraction of the action's argument tokens never seen during
        training, for this tool or any other. 1.0 for an entirely novel
        argument set (or an unseen tool with tokens), 0.0 when every token
        was observed in nominal data."""
        if not action.tokens:
            return 0.0
        vocab = self._tool_vocab.get(action.tool, set())
        unseen = sum(1 for t in action.tokens if t not in vocab and t not in self._global_vocab)
        return unseen / len(action.tokens)

    def instruction_excess(self, action: Action) -> float:
        """How much the preceding observation's instruction-likeness exceeds
        anything seen in nominal training data (0 if within nominal range).
        This is the indirect-prompt-injection signal."""
        return max(0.0, action.obs_instruction_likeness - self._instr_baseline)

    def surprise(self, context: Context, history: Trajectory, action: Action) -> float:
        prev = history[-1][1].tool if history else _START_TOOL
        nll = -self.tool_log_prob(prev, action.tool)
        s = (
            nll
            + self.novelty_weight * self.novelty_fraction(action)
            + self.instruction_weight * self.instruction_excess(action)
        )
        return float(min(s, _MAX_SURPRISE))


def score_trajectory(model, trajectory: Trajectory) -> np.ndarray:
    """Map a trajectory of (context, action) pairs to its surprise-score
    stream s_1, ..., s_T (algorithm.md §2(a)), the input to the baseline
    increments in sentry.baseline. ``model`` is anything exposing
    ``surprise(context, history, action)`` (CausalWorldModel,
    SequentialWorldModel, or a future CAIRN-style world model)."""
    scores = np.empty(len(trajectory))
    history: list[tuple[Context, Action]] = []
    for i, (context, action) in enumerate(trajectory):
        scores[i] = model.surprise(context, history, action)
        history.append((context, action))
    return scores
