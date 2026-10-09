#!/usr/bin/env python3
"""Load and render the registered Jumper coffee-bar scene offscreen.

This is simulation evidence only.  It runs the native CPU backend with a zero
agent (the task's standing/home action), records the complete MuJoCo state, checks
the dimensioned payload against the compiled model, logs actuator torque, and
renders through macOS CGL without creating a window.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DIMENSIONS = REPO / "barista" / "coffee_scene_dimensions.json"
WIDTH, HEIGHT = 960, 540
DEFAULT_STEPS = 150
DEFAULT_SETTLE_STEPS = 50


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    return parser


def _id_ending(model, mujoco, object_type, suffix: str) -> int:
    count_name = {
        mujoco.mjtObj.mjOBJ_BODY: "nbody",
        mujoco.mjtObj.mjOBJ_CAMERA: "ncam",
        mujoco.mjtObj.mjOBJ_GEOM: "ngeom",
    }[object_type]
    matches = []
    for index in range(getattr(model, count_name)):
        name = mujoco.mj_id2name(model, object_type, index)
        if name and (name == suffix or name.endswith((f"/{suffix}", suffix))):
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix!r}, found ids {matches}")
    return matches[0]


def _payload_check(model, mujoco) -> dict[str, float | bool]:
    source = json.loads(DIMENSIONS.read_text("utf-8"))
    objects = source["objects"]
    required_dimensions = {
        "counter": "dimensions_m",
        "server": "outer_diameter_m",
        "v60": "maximum_diameter_m",
        "pitcher": "body_outer_diameter_m",
        "nominal_water": "volume_mL",
    }
    missing = [f"{name}.{field}" for name, field in required_dimensions.items()
               if field not in objects.get(name, {})]
    if missing:
        raise RuntimeError(f"objects lack explicit real dimensions: {missing}")

    dry = float(objects["pitcher"]["dry_mass_kg"])
    water = float(objects["nominal_water"]["mass_kg"])
    design = dry + water
    limit = float(source["payload_check"]["design_limit_kg"])
    pitcher_body = _id_ending(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "pitcher")
    compiled = float(model.body_subtreemass[pitcher_body])
    passes = design <= limit and abs(compiled - design) <= 1e-9
    if not passes:
        raise RuntimeError(
            f"payload check failed: design={design:.6f} kg, compiled={compiled:.6f} kg, "
            f"limit={limit:.6f} kg"
        )
    return {
        "pitcher_dry_kg": dry,
        "nominal_water_kg": water,
        "design_total_kg": design,
        "compiled_total_kg": compiled,
        "design_limit_kg": limit,
        "margin_kg": limit - design,
        "passes": passes,
    }


def _render(model, qpos: np.ndarray, camera_id: int, fps: float, video: Path) -> None:
    import imageio_ffmpeg
    import mujoco

    writer = imageio_ffmpeg.write_frames(
        str(video), size=(WIDTH, HEIGHT), fps=fps, macro_block_size=1,
        codec="libx264", pix_fmt_out="yuv420p", quality=7,
    )
    writer.send(None)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    data = mujoco.MjData(model)
    try:
        for state in qpos:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_id)
            writer.send(renderer.render().tobytes())
    finally:
        renderer.close()
        writer.close()


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _penetration(data) -> float:
    if not data.ncon:
        return 0.0
    return max(0.0, -min(float(data.contact[index].dist) for index in range(data.ncon)))


def main() -> None:
    args = _parser().parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2")

    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

    import imageio_ffmpeg
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import scenes
    import tasks

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    resolution = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(resolution)
    print(resolution.banner())

    env_cfg = tasks.load_env_cfg("jumper.flat", play=True)
    env_cfg.seed = 0
    # The locomotion task normally randomises reset position for training.  A
    # presentation scene has one authored standing slot, so pin planar pose and
    # initial velocity while preserving the task's own small vertical settle.
    reset = env_cfg.events["reset_base"]
    pose_range = dict(reset.params.get("pose_range", {}))
    for axis in ("x", "y", "roll", "pitch", "yaw"):
        pose_range[axis] = (0.0, 0.0)
    reset.params["pose_range"] = pose_range
    velocity_range = dict(reset.params.get("velocity_range", {}))
    for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
        velocity_range[axis] = (0.0, 0.0)
    reset.params["velocity_range"] = velocity_range
    scenes.apply(env_cfg, "coffee_bar")
    env_cfg.scene.num_envs = 1
    fps = 1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    torque_rows: list[np.ndarray] = []
    qpos = np.empty((args.steps, env.sim.mj_model.nq), dtype=np.float64)
    maximum_penetration = 0.0
    maximum_render_penetration = 0.0
    try:
        env.reset()
        robot = env.scene["robot"]
        model = env.sim.mj_model
        payload = _payload_check(model, mujoco)

        # The importer deliberately renames the declared counter ground to the
        # terrain geom that mjlab's foot sensors count.
        required = {
            "counter": "terrain",
            "server": "server_body",
            "v60": "v60_1",
            "pitcher": "pitcher_body",
        }
        for name in required.values():
            _id_ending(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, name)
        camera_id = _id_ending(model, mujoco, mujoco.mjtObj.mjOBJ_CAMERA, "hero")
        pitcher_body_id = _id_ending(
            model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "pitcher"
        )

        actions = torch.zeros(env.action_space.shape, device=env.device)
        pose_data = mujoco.MjData(model)
        pose_data.qpos[:] = env.sim.data.qpos[0].detach().cpu().numpy()
        mujoco.mj_forward(model, pose_data)
        pitcher_spawn = pose_data.xpos[pitcher_body_id].copy()

        for _ in range(args.settle_steps):
            env.step(actions)
            torque_rows.append(robot.data.actuator_force[0].detach().cpu().numpy().copy())
            maximum_penetration = max(maximum_penetration, _penetration(env.sim._datas[0]))

        robot_start = robot.data.root_link_pose_w[0, :3].detach().cpu().numpy().copy()
        qpos[0] = env.sim.data.qpos[0].detach().cpu().numpy()
        pose_data.qpos[:] = qpos[0]
        mujoco.mj_forward(model, pose_data)
        pitcher_start = pose_data.xpos[pitcher_body_id].copy()
        maximum_render_penetration = _penetration(env.sim._datas[0])

        for step in range(1, args.steps):
            env.step(actions)
            qpos[step] = env.sim.data.qpos[0].detach().cpu().numpy()
            torque_rows.append(robot.data.actuator_force[0].detach().cpu().numpy().copy())
            current_penetration = _penetration(env.sim._datas[0])
            maximum_penetration = max(maximum_penetration, current_penetration)
            maximum_render_penetration = max(
                maximum_render_penetration, current_penetration
            )

        robot_end = robot.data.root_link_pose_w[0, :3].detach().cpu().numpy().copy()
        pose_data.qpos[:] = qpos[-1]
        mujoco.mj_forward(model, pose_data)
        pitcher_end = pose_data.xpos[pitcher_body_id].copy()
        actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            or f"actuator_{index}"
            for index in range(model.nu)
        ]

        video = out / "coffee_bar.mp4"
        frame = out / "coffee_bar.png"
        torque_csv = out / "torque.csv"
        summary_path = out / "summary.json"
        _render(model, qpos, camera_id, fps, video)

        timestamp = args.steps / fps / 2.0
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-ss", f"{timestamp:.6f}", "-i", str(video), "-frames:v", "1",
                str(frame),
            ],
            check=True,
        )

        torque = np.asarray(torque_rows)
        with torque_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", *actuator_names])
            for step, values in enumerate(torque):
                writer.writerow([step, *(f"{value:.9f}" for value in values)])

        peak_index = np.unravel_index(np.abs(torque).argmax(), torque.shape)
        peak = float(abs(torque[peak_index]))
        continuous_limit = 1.2
        peak_limit = 1.7464
        longest_over_steps = max(
            _longest_true_run(np.abs(torque[:, column]) > continuous_limit)
            for column in range(torque.shape[1])
        )
        longest_over_s = longest_over_steps / fps
        summary = {
            "simulation_only": True,
            "backend": "native:cpu",
            "seed": 0,
            "task": "jumper.flat",
            "scene": "coffee_bar",
            "settle_steps": args.settle_steps,
            "steps": args.steps,
            "fps": fps,
            "duration_s": args.steps / fps,
            "dimensions_px": [WIDTH, HEIGHT],
            "payload": payload,
            "required_geoms_present": required,
            "robot_xy_displacement_m": float(np.linalg.norm(robot_end[:2] - robot_start[:2])),
            "robot_start_xyz_m": robot_start.tolist(),
            "robot_end_xyz_m": robot_end.tolist(),
            "pitcher_spawn_xyz_m": pitcher_spawn.tolist(),
            "pitcher_start_xyz_m": pitcher_start.tolist(),
            "pitcher_end_xyz_m": pitcher_end.tolist(),
            "pitcher_drop_m": float(pitcher_start[2] - pitcher_end[2]),
            "maximum_contact_penetration_all_steps_m": maximum_penetration,
            "maximum_contact_penetration_rendered_steps_m": maximum_render_penetration,
            "final_contact_penetration_m": _penetration(env.sim._datas[0]),
            "torque": {
                "peak_abs_nm": peak,
                "peak_actuator": actuator_names[peak_index[1]],
                "peak_step": int(peak_index[0]),
                "peak_curve_limit_nm": peak_limit,
                "peak_within_curve": peak <= peak_limit + 1e-6,
                "samples_over_continuous_1_2_nm": int(
                    np.count_nonzero(np.abs(torque) > continuous_limit)
                ),
                "longest_contiguous_over_continuous_steps": longest_over_steps,
                "longest_contiguous_over_continuous_s": longest_over_s,
                "peak_duration_limit_s": 0.3,
                "peak_duration_passes": longest_over_s <= 0.3,
                "continuous_limit_nm": continuous_limit,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    finally:
        env.close()

    print(f"[barista] rendered {args.steps} frames at {fps:g} fps ({args.steps / fps:.2f} s)")
    print(f"[barista] payload {payload['compiled_total_kg']:.3f} kg <= "
          f"{payload['design_limit_kg']:.3f} kg (margin {payload['margin_kg']:.3f} kg)")
    print(f"[barista] peak actuator torque {peak:.6f} N m; "
          f"samples over 1.2 N m: {summary['torque']['samples_over_continuous_1_2_nm']}")
    for path in (video, frame, torque_csv, summary_path):
        print(f"[barista] artifact: {path}")


if __name__ == "__main__":
    main()
