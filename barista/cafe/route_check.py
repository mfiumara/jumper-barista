#!/usr/bin/env python3
"""Route and dance-area clearance on the cafe bar, plus the overhead route plot.

Simulation only. Criterion A1-2 and the ``dance_area`` amendment. Reads the
policy footprints measured in the cafe by ``footprints.py`` and checks, against
every obstacle on the bar top (brew station, pitcher, kettle and cup dressing)
and against the bar's edges:

- ``dance_area``: the swept footprint of every dance and of the flat jump stays
  inside the declared disc, and the disc keeps >= 30 mm from edges and dressing;
- ``turn``: the measured turn in place at the start slot (walking policy);
- ``walk``: the measured walk along +X from the start slot to ``pre_align``;
- ``align``: the remaining straight segment from ``pre_align`` to
  ``pour_start_pose``, swept kinematically with both the walking stance and the
  accepted M4 start configuration (LF claw raised to keyframe 0, open), and
  checked in 3D with MuJoCo's own geom distance at the pour-start pose.

Footprint clearance is the 2D distance between XY convex hulls of all collision
geometry, so it is conservative: it ignores height (a leg above the V60 plate
counts as touching it).

    python barista/cafe/route_check.py --footprints out/cafe/footprints --out out/cafe/route
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cafe_common as cc
import numpy as np

REQUIRED_CLEARANCE_M = 0.03
OBSTACLE_GROUPS = {
    "server": ("server_",),
    "v60_and_cues": ("v60_", "dripper_"),
    "pitcher": None,  # every geom on the pitcher body
    "kettle": ("cafe_kettle_",),
    "cup_stack": ("cafe_cup_",),
}
DANCES = ("dance_crab", "dance_brazilian", "dance_dream_wings", "dance_maze", "dance_waist",
          "jump")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--footprints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def obstacles() -> dict[str, list[np.ndarray]]:
    """XY convex hull per obstacle geom, grouped, from the compiled cafe world."""
    import mujoco

    from scenes import cafe_bar
    from scenes.package import read_package

    package = read_package(cafe_bar.package_dir())
    model = package.spec("scene.xml").compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    pitcher = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pitcher")
    groups: dict[str, list[np.ndarray]] = {k: [] for k in OBSTACLE_GROUPS}
    for g in range(model.ngeom):
        name = cc.geom_name(model, g)
        for group, tokens in OBSTACLE_GROUPS.items():
            hit = (model.geom_bodyid[g] == pitcher) if tokens is None else \
                any(name.startswith(t) for t in tokens)
            if hit:
                groups[group].append(cc.hull(cc.geom_points(model, data, g)))
    return groups


def edge_clearance(poly: np.ndarray, bar: dict) -> float:
    x0, x1 = bar["top_x_range_m"]
    y0, y1 = bar["top_y_range_m"]
    return float(np.min(np.stack([poly[:, 0] - x0, x1 - poly[:, 0],
                                  poly[:, 1] - y0, y1 - poly[:, 1]])))


def clearances(polys, groups, bar) -> dict[str, float]:
    result = {"bar_edges": min(edge_clearance(p, bar) for p in polys)}
    for group, hulls in groups.items():
        result[group] = min(cc.polygon_distance(p, h) for p in polys for h in hulls)
    return result


def load_hulls(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    data = np.load(path)
    points, sizes = data["hull_points"], data["hull_sizes"]
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [points[edges[i]:edges[i + 1]] for i in range(len(sizes))], data["base_xy"]


def pour_start_configuration(d: dict) -> dict:
    """The accepted M4 start state at the pour-start pose, measured in the cafe."""
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    import scenes
    import tasks
    from tasks.jumper.common.constants import HOME
    from tasks.jumper.five_foot.claw import GRIPPER_OPEN

    config = __import__("json").loads(cc.M4_TRAJECTORY.read_text("utf-8"))
    pose = d["route"]["pour_start_pose"]
    resolution = cc.backend()
    env_cfg = tasks.load_env_cfg("jumper.barista", play=True)
    scenes.apply(env_cfg, cc.SCENE)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    try:
        env.reset()
        robot = env.scene["robot"]
        yaw = math.radians(pose["yaw_deg"])
        root = torch.tensor([[pose["x_m"], pose["y_m"], pose["base_z_m"],
                              math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                            dtype=torch.float, device=resolution.device)
        cols = {n: i for i, n in enumerate(robot.joint_names)}
        initial = np.asarray([HOME[n] for n in robot.joint_names], dtype=np.float64)
        for name, value in zip(config["arm_joint_order"], config["keyframes"][0]["arm_rad"],
                               strict=True):
            initial[cols[name]] = value
        initial[cols["LF_J4_joint"]] = GRIPPER_OPEN
        joints = torch.tensor(initial[None, :], dtype=torch.float, device=resolution.device)
        robot.write_root_link_pose_to_sim(root)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.sim.forward()
        model = env.sim.mj_model
        data = env.sim._datas[0]
        mujoco.mj_forward(model, data)
        geoms = cc.robot_collision_geoms(model)
        poly = cc.hull(np.concatenate([cc.geom_points(model, data, g) for g in geoms]))
        local = cc.transform(poly - [pose["x_m"], pose["y_m"]], 0.0, 0.0, -yaw)
        pitcher = next(b for b in range(model.nbody)
                       if cc.name(model, mujoco.mjtObj.mjOBJ_BODY, b).endswith("_pitcher"))
        groups3d: dict[str, list[int]] = {k: [] for k in OBSTACLE_GROUPS}
        for g in range(model.ngeom):
            name = cc.geom_name(model, g).removeprefix("bar_")
            for group, tokens in OBSTACLE_GROUPS.items():
                hit = (model.geom_bodyid[g] == pitcher) if tokens is None else \
                    any(name.startswith(t) for t in tokens)
                if hit:
                    groups3d[group].append(g)
        fromto = np.zeros(6)
        distance3d = {}
        for group, members in groups3d.items():
            best = (math.inf, "", "")
            for g in members:
                for r in geoms:
                    value = float(mujoco.mj_geomDistance(model, data, r, g, 1.0, fromto))
                    if value < best[0]:
                        best = (value, cc.geom_name(model, r), cc.geom_name(model, g))
            distance3d[group] = {"distance_m": best[0], "robot_geom": best[1],
                                 "obstacle_geom": best[2]}
        return {"hull_world": poly, "hull_local": local, "clearance_3d": distance3d}
    finally:
        env.close()


def plot(path: Path, d: dict, groups, segments: dict, dance_hulls, walk_base) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    bar = d["bar"]
    x0, x1 = bar["top_x_range_m"]
    y0, y1 = bar["top_y_range_m"]
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=120)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#8a5a33", alpha=0.35,
                           edgecolor="#5a3418", lw=2, label="bar top (z = 0)"))
    colours = {"server": "#6b8fa3", "v60_and_cues": "#e8e2d4", "pitcher": "#555555",
               "kettle": "#222222", "cup_stack": "#d9cbb0"}
    for group, hulls in groups.items():
        for k, h in enumerate(hulls):
            ax.add_patch(Polygon(h, closed=True, facecolor=colours[group], edgecolor="k",
                                 lw=0.4, alpha=0.9, label=group if k == 0 else None))
    dance = d["dance_area"]
    ax.add_patch(Circle(dance["centre_m"], dance["radius_m"], fill=False, ls="--",
                        color="#2a5bd7", lw=1.5, label=f"dance_area r={dance['radius_m']:.2f} m"))
    for k, h in enumerate(dance_hulls):
        ax.add_patch(Polygon(h, closed=True, fill=False, edgecolor="#2a5bd7", lw=0.3,
                             alpha=0.25, label="dance/jump swept footprint" if k == 0 else None))
    for name, colour in (("turn", "#c78a00"), ("walk", "#2b9a3f"), ("align", "#d04a2a")):
        for k, h in enumerate(segments[name]["sample_hulls"]):
            ax.add_patch(Polygon(h, closed=True, fill=False, edgecolor=colour, lw=0.5,
                                 alpha=0.5, label=f"{name} footprint" if k == 0 else None))
    ax.plot(walk_base[:, 0], walk_base[:, 1], color="#2b9a3f", lw=2, label="measured walk base path")
    route = d["route"]
    xs = [w["x_m"] for w in route["walk"]]
    ys = [w["y_m"] for w in route["walk"]]
    ax.plot(xs, ys, "o--", color="k", ms=4, lw=0.8, label="route waypoints")
    for w in route["walk"]:
        ax.annotate(w["name"], (w["x_m"], w["y_m"]), textcoords="offset points", xytext=(4, -12),
                    fontsize=8)
    start = route["start_slot"]
    ax.arrow(start["x_m"], start["y_m"], 0.0, -0.12, width=0.006, color="#7a1fa2")
    ax.annotate("start_slot (faces -Y, customer)", (0.0, -0.13), fontsize=8, color="#7a1fa2",
                xytext=(6, -4), textcoords="offset points")
    pour = route["pour_start_pose"]
    ax.arrow(pour["x_m"], pour["y_m"], 0.12, 0.0, width=0.006, color="#d04a2a")
    cam = d["customer_camera"]
    ax.plot(cam["position_m"][0], cam["position_m"][1], "s", color="#7a1fa2")
    ax.annotate("customer_pov camera (z = 0.35 m)", cam["position_m"][:2], fontsize=8,
                xytext=(6, 4), textcoords="offset points")
    ax.annotate("door + door_chime at x = -2.2 m, y = -1.95 m (off-plot)", (x0, -0.95),
                fontsize=8, color="#555")
    table = [f"{seg}: " + ", ".join(f"{k} {v * 1000:.0f} mm" for k, v in
                                    segments[seg]["clearance_m"].items())
             for seg in ("dance_area", "turn", "walk", "align")]
    ax.text(x0, y1 + 0.03, "min 2D clearance (footprint hull):\n" + "\n".join(table),
            fontsize=7, va="bottom", family="monospace")
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 0.1, x1 + 0.1)
    ax.set_ylim(-1.05, y1 + 0.32)
    ax.set_xlabel("x (m) along the bar")
    ax.set_ylabel("y (m), customer at -Y")
    ax.set_title("Jumper Barista cafe_bar: overhead route and clearance (simulation only)")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = _parser().parse_args()
    cc.headless()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fp = args.footprints.resolve()
    d = cc.dims()
    bar = d["bar"]
    groups = obstacles()

    segments: dict[str, dict] = {}
    # dance area: disc vs everything, and every dance/jump hull inside the disc
    dance = d["dance_area"]
    centre = np.asarray(dance["centre_m"], dtype=float)
    radius = float(dance["radius_m"])
    dance_hulls = []
    per_policy = {}
    windows = dance.get("policy_windows_s", {})
    full_run = {}
    for label in DANCES:
        hulls, _ = load_hulls(fp / f"{label}_hulls.npz")
        full_run[label] = float(np.linalg.norm(cc.hull(np.concatenate(hulls)) - centre,
                                               axis=1).max())
        if label in windows:
            dt = __import__("json").loads((fp / f"{label}.json").read_text("utf-8"))["control_dt_s"]
            hulls = hulls[:round(float(windows[label]) / dt)]
        swept = cc.hull(np.concatenate(hulls))
        dance_hulls.append(swept)
        per_policy[label] = float(np.linalg.norm(swept - centre, axis=1).max())
    angles = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    disc = centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    segments["dance_area"] = {
        "clearance_m": clearances([disc], groups, bar),
        "policy_footprint_radius_m": per_policy,
        "policy_windows_s": windows,
        "policy_full_run_footprint_radius_m": full_run,
        "max_policy_footprint_radius_m": max(per_policy.values()),
        "footprints_inside_disc": max(per_policy.values()) <= radius,
        "sample_hulls": [],
    }

    turn_hulls, _ = load_hulls(fp / "walk_turn_hulls.npz")
    walk_hulls, walk_base = load_hulls(fp / "walk_along_bar_hulls.npz")
    for name, hulls in (("turn", turn_hulls), ("walk", walk_hulls)):
        step = max(1, len(hulls) // 12)
        segments[name] = {"clearance_m": clearances(hulls, groups, bar),
                          "frames": len(hulls), "sample_hulls": hulls[::step] + [hulls[-1]]}
    ys = np.concatenate([h[:, 1] for h in walk_hulls])
    y0, y1 = bar["top_y_range_m"]
    segments["walk"]["gait_width"] = {
        "swept_y_range_m": [float(ys.min()), float(ys.max())],
        "front_edge_margin_m": float(ys.min() - y0),
        "back_edge_margin_m": float(y1 - ys.max()),
        "bar_depth_m": y1 - y0,
        "swept_width_m": float(ys.max() - ys.min()),
    }

    # align: pre_align -> pour_start, both stances, 1 cm steps along the segment
    route = d["route"]
    pre = next(w for w in route["walk"] if w["name"] == "pre_align")
    pour = route["pour_start_pose"]
    m4 = pour_start_configuration(d)
    end_local = walk_hulls[-1] - walk_base[-1]
    samples = []
    count = max(2, round(math.hypot(pour["x_m"] - pre["x_m"],
                                        pour["y_m"] - pre["y_m"]) / 0.01) + 1)
    for s in np.linspace(0.0, 1.0, count):
        x = pre["x_m"] + s * (pour["x_m"] - pre["x_m"])
        y = pre["y_m"] + s * (pour["y_m"] - pre["y_m"])
        samples.append(end_local + [x, y])
        samples.append(cc.transform(m4["hull_local"], x, y, math.radians(pour["yaw_deg"])))
    segments["align"] = {
        "clearance_m": clearances(samples, groups, bar),
        "clearance_at_pour_start_m4_configuration_m": clearances([m4["hull_world"]], groups, bar),
        "clearance_3d_at_pour_start_m": m4["clearance_3d"],
        "sample_hulls": samples[::4] + [samples[-1]],
        "note": "pre_align -> pour_start is the grasp approach of the accepted M4 start: "
                "the pitcher is the object the claw is about to take, so its clearance here "
                "is reported, and the transfer replay (run_m4_transfer.py) is the gate for it",
    }

    checks = {
        "dance_footprints_inside_dance_area": segments["dance_area"]["footprints_inside_disc"],
        "dance_area_clear_30mm": min(segments["dance_area"]["clearance_m"].values())
        >= REQUIRED_CLEARANCE_M,
        "turn_clear_30mm": min(segments["turn"]["clearance_m"].values()) >= REQUIRED_CLEARANCE_M,
        "walk_clear_30mm": min(segments["walk"]["clearance_m"].values()) >= REQUIRED_CLEARANCE_M,
        "align_clear_30mm_of_fixed_props_dressing_and_edges": min(
            v for k, v in segments["align"]["clearance_m"].items() if k != "pitcher"
        ) >= REQUIRED_CLEARANCE_M,
        "pour_start_3d_clear_30mm_of_fixed_props_and_dressing": min(
            v["distance_m"] for k, v in m4["clearance_3d"].items() if k != "pitcher"
        ) >= REQUIRED_CLEARANCE_M,
    }
    plot_path = out / "cafe_route_overhead.png"
    plot(plot_path, d, groups, segments, dance_hulls, walk_base)
    report = {
        "simulation_only": True,
        "required_clearance_m": REQUIRED_CLEARANCE_M,
        "method": "2D distance between XY convex hulls of all robot collision geometry and of "
                  "each obstacle geom (conservative: ignores height); 3D mj_geomDistance at the "
                  "pour-start pose",
        "segments": {k: {kk: vv for kk, vv in v.items() if kk != "sample_hulls"}
                     for k, v in segments.items()},
        "checks": checks,
        "passes": all(checks.values()),
        "plot": str(plot_path),
    }
    cc.write_json(out / "route_clearance.json", report)
    print("[cafe-route] minimum 2D footprint clearance (mm):")
    header = ["segment", "bar_edges", *OBSTACLE_GROUPS]
    print("  " + " ".join(f"{h:>13s}" for h in header))
    for seg in ("dance_area", "turn", "walk", "align"):
        c = segments[seg]["clearance_m"]
        print("  " + " ".join([f"{seg:>13s}", *(f"{c[h] * 1000:13.1f}" for h in header[1:])]))
    print("[cafe-route] 3D clearance at pour_start (M4 start configuration, mm): " + ", ".join(
        f"{k} {v['distance_m'] * 1000:.1f}" for k, v in m4["clearance_3d"].items()))
    print(f"[cafe-route] dance/jump footprint radius max "
          f"{segments['dance_area']['max_policy_footprint_radius_m']:.4f} m vs dance_area "
          f"{radius:.3f} m; gait width {segments['walk']['gait_width']}")
    for k, v in checks.items():
        print(f"[cafe-route] {k}: {v}")
    print(f"[cafe-route] plot: {plot_path}")
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
