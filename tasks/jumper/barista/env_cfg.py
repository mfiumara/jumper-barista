"""Environment configuration for the simulation-only barista reference task."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..five_foot.env_cfg import env_cfg as five_foot_env_cfg
from ..five_foot.objects import PROP_CONE, PROP_IMPRATIO
from .audited_servo import AuditedServoCurveActuatorCfg


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the five-foot world with the task-local audited servo runtime."""
    cfg = five_foot_env_cfg(asset=asset, play=play)
    robot = cfg.scene.entities["robot"]
    articulation = robot.articulation
    if articulation is None or len(articulation.actuators) != 1:
        raise RuntimeError("jumper.barista requires the common single 22-joint actuator")
    common = articulation.actuators[0]
    values = {field.name: getattr(common, field.name) for field in fields(common) if field.init}
    audited = AuditedServoCurveActuatorCfg(**values)
    robot.articulation = replace(articulation, actuators=(audited,))
    # A friction grasp depends on the solver settings as much as the geom
    # coefficients. Configure them before compilation so native per-world models
    # and the app consumer receive the same settings as the reference replay.
    cfg.sim.mujoco = replace(cfg.sim.mujoco, cone=PROP_CONE, impratio=PROP_IMPRATIO)
    return cfg


__all__ = ["env_cfg"]