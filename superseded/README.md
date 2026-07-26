# Superseded scripts

These three scripts are kept for provenance only. **Do not read their output as
results.** `reproduce.sh` does not invoke them and the paper does not cite them.

| script | why it is superseded |
|---|---|
| `evaluate_harm.py` | Pools the 50 τ-bench retail trajectories into the compromise-detection benign set. Those trajectories are loaded without their message list, so the sink-provenance signal is forced to 0 rather than measured — the contamination the paper's limitations describe as removed. Its numbers are computed on a two-way split under the old protocol. |
| `compare_sota.py` | Prints AgentArmor's published operating point as a table row directly beneath our own, which reads as a head-to-head comparison. The paper explicitly declines that comparison: the corpora, agents and positive-class definitions differ and no baseline was re-implemented. |
| `evaluate_audit.py` | Superseded two-way-split protocol; its combined-vs-observation-only figures do not match the current tables. |

The current evaluation is `real_data/evaluate_fused.py` (compromise detection and
the signal ablation), with `real_data/report.py` checking every reported value.
