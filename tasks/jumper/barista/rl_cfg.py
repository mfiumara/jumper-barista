"""PPO configuration inherited from the five-foot manipulation task."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..five_foot.rl_cfg import agent_cfg as five_foot_agent_cfg
from ..five_foot.rl_cfg import runner_cls


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Reuse the five-foot baseline under this task's own log namespace."""
    cfg = five_foot_agent_cfg()
    # Registry log-layout invariant: experiment_name="jumper.barista".
    cfg.experiment_name = "jumper.barista"
    return cfg


__all__ = ["agent_cfg", "runner_cls"]