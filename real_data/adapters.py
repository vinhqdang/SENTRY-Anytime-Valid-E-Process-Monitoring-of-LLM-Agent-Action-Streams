"""Convert AgentDojo / tau-bench trace logs into sentry.scores Trajectory
objects (algorithm.md §1's X_t = "feature representation of the t-th action
... embedding of the action+arguments"), plus per-trajectory metadata needed
for evaluation: whether it's an attack trajectory, whether the attack
succeeded (security==False), and a "drift index" -- the position in the
action stream from which the trajectory's actions could have been
influenced by injected content, used as the changepoint for measuring
detection delay (algorithm.md §7 item 2).

Feature extraction uses the hashing trick (a fixed, deterministic
projection of tool arguments and the immediately preceding tool-result
text into a small numeric vector) rather than a learned text embedding --
this keeps the whole pipeline GPU-free and reproducible; swapping in a
real embedding model later only touches `_hash_tokens`. Two things matter
for this to carry any signal at all:

  1. Tokenizing on a word/number regex, not whitespace. `json.dumps`'s
     default compact separators leave almost no whitespace
     (`{"file_path":"x.txt"}` splits into ~2 whitespace-tokens but 3
     meaningful ones), so whitespace-splitting was nearly blind to content.
  2. Hashing the preceding tool *result* text alongside the action's own
     arguments. The action alone doesn't show where injected content
     lives -- it lives in what the previous tool call returned -- so a
     surprise score computed from the action alone can't see it; this is
     the do_C(t-1) part of algorithm.md §2a's s_t = -log P(a_t | do(c),
     H_{t-1}) that a bare argument hash was dropping.
"""

from __future__ import annotations

import json
import os
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from sentry.scores import Action, Context, Trajectory, instruction_likeness

# Which instruction-likeness scorer feeds Action.obs_instruction_likeness:
#   "lexicon" (default) -- the fixed keyword/regex statistic
#   "llm"               -- the cached semantic LLM judge (sentry.llm_judge);
#                          falls back to the lexicon for any observation not in
#                          the cache, so a partial cache degrades gracefully.
IL_MODE = os.environ.get("SENTRY_IL_MODE", "lexicon")
_LLM_CACHE: dict[str, float] | None = None
# Action-justification audit (sentry.llm_judge.score_actions). Enabled with
# SENTRY_JUSTIFY=1; reads the committed cache, so it needs no API at eval time.
JUSTIFY_ON = os.environ.get("SENTRY_JUSTIFY", "0") == "1"
_JUSTIFY_CACHE: dict[str, float] | None = None


def _action_unjustified(user_task: str, tool: str, args) -> float:
    if not JUSTIFY_ON:
        return 0.0
    global _JUSTIFY_CACHE
    if _JUSTIFY_CACHE is None:
        from sentry.llm_judge import JUSTIFY_CACHE

        _JUSTIFY_CACHE = json.loads(JUSTIFY_CACHE.read_text()) if JUSTIFY_CACHE.exists() else {}
    from sentry.llm_judge import justify_key

    args_json = args if isinstance(args, str) else json.dumps(args, sort_keys=True, default=str)
    return _JUSTIFY_CACHE.get(justify_key(user_task, tool, args_json[:400]), 0.0)


def _first_user_text(messages) -> str:
    for m in messages:
        if m.get("role") == "user":
            return " ".join(str(_content_text(m.get("content"))).split())[:600]
    return ""


def _obs_instruction_score(text: str) -> float:
    if IL_MODE != "llm":
        return instruction_likeness(text)
    global _LLM_CACHE
    if _LLM_CACHE is None:
        from sentry.llm_judge import load_cache

        _LLM_CACHE = load_cache()
    from sentry.llm_judge import text_key

    if not text or not text.strip():
        return 0.0
    hit = _LLM_CACHE.get(text_key(text))
    return hit if hit is not None else instruction_likeness(text)


ARGS_HASH_DIM = 24
OBS_HASH_DIM = 16
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./@-]{2,}")


def _hash_tokens(tokens: list[str], dim: int) -> np.ndarray:
    vec = np.zeros(dim)
    for token in tokens:
        # crc32, not builtin hash(): str hashing is salted per process
        # (PYTHONHASHSEED), which made hashed features -- and every score
        # derived from them -- non-reproducible across runs.
        h = zlib.crc32(token.encode())
        vec[h % dim] += 1.0 if (h // dim) % 2 == 0 else -1.0
    return vec


def _hash_args(args: Any, dim: int = ARGS_HASH_DIM) -> np.ndarray:
    s = json.dumps(args, sort_keys=True, default=str)
    tokens = _TOKEN_RE.findall(s)
    vec = _hash_tokens(tokens, dim)
    extra = np.array([len(tokens), len(args) if isinstance(args, (dict, list)) else 0])
    return np.concatenate([vec, extra])


def _hash_observation(text: str, dim: int = OBS_HASH_DIM) -> np.ndarray:
    return _hash_tokens(_TOKEN_RE.findall(text or ""), dim)


def _hash_features(args: Any, preceding_observation: str = "") -> np.ndarray:
    return np.concatenate([_hash_args(args), _hash_observation(preceding_observation)])


def _arg_tokens(args: Any) -> tuple[str, ...]:
    """Raw argument tokens for vocabulary/novelty scoring
    (sentry.scores.SequentialWorldModel) -- unlike the hash, these preserve
    exact values so 'never seen this recipient before' is checkable."""
    return tuple(_TOKEN_RE.findall(json.dumps(args, sort_keys=True, default=str)))


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("content", "") or "" for c in content if isinstance(c, dict))
    return str(content)


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def _alnum(s: str) -> str:
    """Collapse to lowercase alphanumerics only -- robust to any
    re-serialization the environment applies to an injected string (YAML
    line-folding with '\\' continuations, re-indentation, quoting), which
    defeated whitespace-only normalization on several real logs."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def agentdojo_log_to_trajectory(log: dict, task_key: Any) -> tuple[Trajectory, dict]:
    """One AgentDojo TaskResults JSON dict -> (Trajectory, metadata)."""
    injections = log.get("injections") or {}
    # Environments re-serialize injected strings (e.g. YAML-dump a dict that
    # embeds one as a field value), which re-wraps/re-indents its newlines.
    # Whitespace-normalize both sides so the substring match survives that.
    injected_strings = [_alnum(v) for v in injections.values() if isinstance(v, str) and v.strip()]
    user_task = _first_user_text(log["messages"])

    tools_seen: set[str] = set()
    actions: list[tuple[str, Any, str]] = []
    drift_index: int | None = None
    injected_content_seen = False
    last_observation = ""

    for m in log["messages"]:
        if m["role"] == "tool":
            raw_text = _content_text(m.get("content"))
            last_observation = raw_text
            text = _alnum(raw_text)
            if injected_strings and any(s in text for s in injected_strings):
                injected_content_seen = True
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if injected_content_seen and drift_index is None:
                    drift_index = len(actions)
                tools_seen.add(tc["function"])
                actions.append((tc["function"], tc["args"], last_observation))

    ctx = Context(task_id=task_key, available_tools=frozenset(tools_seen))
    traj: Trajectory = [
        (
            ctx,
            Action(
                tool=n,
                features=_hash_features(a, obs),
                tokens=_arg_tokens(a),
                obs_instruction_likeness=_obs_instruction_score(obs),
                unjustified=_action_unjustified(user_task, n, a),
            ),
        )
        for n, a, obs in actions
    ]
    meta = {
        "source": "agentdojo",
        "is_attack": log.get("attack_type") not in (None, "none"),
        "attack_type": log.get("attack_type"),
        "security": log.get("security"),
        "utility": log.get("utility"),
        "drift_index": drift_index,
        "n_actions": len(traj),
    }
    return traj, meta


def load_agentdojo_logs(logdir: Path) -> list[tuple[Trajectory, dict]]:
    """Walk <logdir>/<pipeline>/<suite>/<user_task>/<attack>/<injection>.json
    and convert every log file found."""
    results = []
    for path in sorted(Path(logdir).rglob("*.json")):
        with path.open() as f:
            log = json.load(f)
        suite_name = path.parents[2].name
        user_task_id = path.parents[1].name
        task_key = (suite_name, user_task_id, path.parent.name, path.stem)
        traj, meta = agentdojo_log_to_trajectory(log, task_key)
        meta["path"] = str(path)
        if traj:
            results.append((traj, meta))
    return results


def taubench_entry_to_trajectory(entry: dict, task_key: Any) -> tuple[Trajectory, dict]:
    """One tau-bench EnvRunResult dict -> (Trajectory, metadata). tau-bench
    has no built-in attack suite, so every trajectory here is nominal."""
    tools_seen: set[str] = set()
    actions: list[tuple[str, Any, str]] = []
    last_observation = ""
    user_task = _first_user_text(entry.get("traj", []))
    for m in entry.get("traj", []):
        if m.get("role") == "tool":
            last_observation = str(m.get("content") or "")
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                name = fn.get("name")
                if name is None:
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tools_seen.add(name)
                actions.append((name, args, last_observation))

    ctx = Context(task_id=task_key, available_tools=frozenset(tools_seen))
    traj: Trajectory = [
        (
            ctx,
            Action(
                tool=n,
                features=_hash_features(a, obs),
                tokens=_arg_tokens(a),
                obs_instruction_likeness=_obs_instruction_score(obs),
                unjustified=_action_unjustified(user_task, n, a),
            ),
        )
        for n, a, obs in actions
    ]
    meta = {
        "source": "tau_bench",
        "is_attack": False,
        "reward": entry.get("reward"),
        "n_actions": len(traj),
    }
    return traj, meta


def load_taubench_logs(logdir: Path) -> list[tuple[Trajectory, dict]]:
    """Load every tau-bench checkpoint JSON (a list of EnvRunResult dicts)
    under logdir."""
    results = []
    # rglob (not glob): tau-bench may write the checkpoint into a subdirectory
    # when the model id contains a slash, so search recursively.
    for path in sorted(Path(logdir).rglob("*.json")):
        with path.open() as f:
            entries = json.load(f)
        for entry in entries:
            task_key = ("tau_bench_retail", entry.get("task_id"), entry.get("trial"))
            traj, meta = taubench_entry_to_trajectory(entry, task_key)
            meta["path"] = str(path)
            if traj:
                results.append((traj, meta))
    return results
