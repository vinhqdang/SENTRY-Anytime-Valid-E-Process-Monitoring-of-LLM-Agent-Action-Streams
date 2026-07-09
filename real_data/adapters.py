"""Convert AgentDojo / tau-bench trace logs into sentry.scores Trajectory
objects (algorithm.md §1's X_t = "feature representation of the t-th action
... embedding of the action+arguments"), plus per-trajectory metadata needed
for evaluation: whether it's an attack trajectory, whether the attack
succeeded (security==False), and a "drift index" -- the position in the
action stream from which the trajectory's actions could have been
influenced by injected content, used as the changepoint for measuring
detection delay (algorithm.md §7 item 2).

Feature extraction uses the hashing trick (a fixed, deterministic
projection of the JSON-serialized tool arguments into a small numeric
vector) rather than a learned text embedding -- this keeps the whole
pipeline GPU-free and reproducible; swapping in a real embedding model
later only touches `_hash_features`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sentry.scores import Action, Context, Trajectory

HASH_DIM = 16


def _hash_features(args: Any, dim: int = HASH_DIM) -> np.ndarray:
    s = json.dumps(args, sort_keys=True, default=str)
    vec = np.zeros(dim)
    for token in s.split():
        h = hash(token)
        vec[h % dim] += 1.0 if (h // dim) % 2 == 0 else -1.0
    extra = np.array([len(s), len(args) if isinstance(args, (dict, list)) else 0])
    return np.concatenate([vec, extra])


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("content", "") or "" for c in content if isinstance(c, dict))
    return str(content)


def agentdojo_log_to_trajectory(log: dict, task_key: Any) -> tuple[Trajectory, dict]:
    """One AgentDojo TaskResults JSON dict -> (Trajectory, metadata)."""
    injections = log.get("injections") or {}
    injected_strings = [v for v in injections.values() if isinstance(v, str) and v.strip()]

    tools_seen: set[str] = set()
    actions: list[tuple[str, Any]] = []
    drift_index: int | None = None
    injected_content_seen = False

    for m in log["messages"]:
        if m["role"] == "tool":
            text = _content_text(m.get("content"))
            if injected_strings and any(s in text for s in injected_strings):
                injected_content_seen = True
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if injected_content_seen and drift_index is None:
                    drift_index = len(actions)
                tools_seen.add(tc["function"])
                actions.append((tc["function"], tc["args"]))

    ctx = Context(task_id=task_key, available_tools=frozenset(tools_seen))
    traj: Trajectory = [(ctx, Action(tool=n, features=_hash_features(a))) for n, a in actions]
    meta = {
        "source": "agentdojo",
        "is_attack": log.get("attack_type") not in (None, "none"),
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
    actions: list[tuple[str, Any]] = []
    for m in entry.get("traj", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
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
                actions.append((name, args))

    ctx = Context(task_id=task_key, available_tools=frozenset(tools_seen))
    traj: Trajectory = [(ctx, Action(tool=n, features=_hash_features(a))) for n, a in actions]
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
    for path in sorted(Path(logdir).glob("*.json")):
        with path.open() as f:
            entries = json.load(f)
        for entry in entries:
            task_key = ("tau_bench_retail", entry.get("task_id"), entry.get("trial"))
            traj, meta = taubench_entry_to_trajectory(entry, task_key)
            meta["path"] = str(path)
            if traj:
                results.append((traj, meta))
    return results
