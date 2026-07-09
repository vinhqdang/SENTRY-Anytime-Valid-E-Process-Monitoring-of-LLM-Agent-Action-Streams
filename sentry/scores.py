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
from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np

_EPS = 1e-9
_MAX_SURPRISE = 50.0  # clip -log(0) to a large finite value


@dataclass(frozen=True)
class Context:
    """Task context c: the stated goal and the tools available to pursue it."""

    task_id: Hashable
    available_tools: frozenset[str]


@dataclass(frozen=True)
class Action:
    """A single agent action a_t: which tool was invoked and a continuous
    feature vector summarizing its arguments (e.g. an embedding)."""

    tool: str
    features: np.ndarray = field(default_factory=lambda: np.zeros(0))


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


def score_trajectory(model: CausalWorldModel, trajectory: Trajectory) -> np.ndarray:
    """Map a trajectory of (context, action) pairs to its surprise-score
    stream s_1, ..., s_T (algorithm.md §2(a)), the input to the baseline
    increments in sentry.baseline."""
    scores = np.empty(len(trajectory))
    history: list[tuple[Context, Action]] = []
    for i, (context, action) in enumerate(trajectory):
        scores[i] = model.surprise(context, history, action)
        history.append((context, action))
    return scores
