import numpy as np

from sentry.scores import Action, CausalWorldModel, Context, score_trajectory


def _toy_trajectory(rng, task_id="t1", n=20, tools=("search", "write_file"), drift=False):
    ctx = Context(task_id=task_id, available_tools=frozenset(tools))
    traj = []
    for i in range(n):
        tool = tools[0] if rng.random() < (0.2 if drift and i > n // 2 else 0.8) else tools[1]
        feats = rng.normal(0, 1, 3)
        traj.append((ctx, Action(tool=tool, features=feats)))
    return traj


def test_world_model_fits_and_scores_nominal_lower_than_drifted():
    rng = np.random.default_rng(0)
    train = [_toy_trajectory(rng) for _ in range(50)]
    model = CausalWorldModel().fit(train)

    nominal = _toy_trajectory(rng, drift=False, n=200)
    drifted = _toy_trajectory(rng, drift=True, n=200)

    s_nominal = score_trajectory(model, nominal)
    s_drifted = score_trajectory(model, drifted)

    assert s_nominal[100:].mean() < s_drifted[100:].mean()


def test_do_intervention_removed_tool_is_maximally_surprising():
    rng = np.random.default_rng(1)
    train = [_toy_trajectory(rng) for _ in range(30)]
    model = CausalWorldModel().fit(train)
    ctx = Context(task_id="t1", available_tools=frozenset({"search", "write_file"}))
    logp_free = model.tool_log_prob(ctx, "search")
    logp_removed = model.tool_log_prob(ctx, "search", exclude=frozenset({"search"}))
    assert logp_removed < logp_free


def test_intervention_consistency_score_nonnegative():
    rng = np.random.default_rng(2)
    train = [_toy_trajectory(rng) for _ in range(30)]
    model = CausalWorldModel().fit(train)
    ctx = Context(task_id="t1", available_tools=frozenset({"search", "write_file"}))
    action = Action(tool="write_file", features=np.zeros(3))
    score = model.intervention_consistency_score(ctx, action, do_remove=frozenset({"search"}))
    assert score >= 0.0
