#!/usr/bin/env python3
"""Cut the Jumper Barista competition film from a logged M4 replay (simulation only).

The film is rendered from ``render_states.npz``, written by
``barista/verification/run_m4.py --save-render-states``: the render-rate MuJoCo
``qpos`` of the verified replay and the ballistic stream path logged at each frame.
Nothing is re-simulated here and nothing is staged: every frame is the replayed
state, every drawn stream segment is the logged path launched from the measured
spout pose and velocity, and the overhead trail marks the logged landing points.
The water is a massless particle/visual approximation; the film says so on every
frame, and it never implies a real-robot run.

Before rendering, the qpos hash is checked against the npz header and against the
replay's ``summary.json`` (which must report an overall pass), so the film can only
be cut from the accepted, verified replay.

    python barista/film/make_film.py \\
        --replay <run_m4 --out dir> --out <dir>/jumper_barista_sim.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 1280, 720
INSET_W, INSET_H = 448, 252
FPS = 30
STREAM_RGBA = (0.18, 0.58, 0.82, 0.85)
TRAIL_RGBA = (0.36, 0.20, 0.10, 0.85)
DISCLAIMER = "MuJoCo simulation  |  water is a particle/visual approximation  |  not a real-robot run"


@dataclass(frozen=True)
class Shot:
    """One continuous piece of the replay: sim time [start, end) at a playback speed."""

    label: str
    start_s: float
    end_s: float
    speed: float
    view: str  # "hero" | "overhead"
    inset: bool = False


# The cut. Sim-time boundaries are the M4 trajectory's phase boundaries
# (barista/trajectory/m4_spiral_pour.json): approach/close/lift end at 6.0 s,
# transit at 21.5 s, the spiral runs 23.5-41.5 s, pour_out ends 43.5 s,
# set-down/release/retreat end at 54 s and final_settle at 57 s.
SHOTS = (
    Shot("1  pickup: approach, grasp, lift", 0.0, 6.5, 1.0, "hero"),
    Shot("1  pickup: lift-hold and carry to the dripper", 6.5, 22.6, 4.0, "hero"),
    Shot("2  spiral pour, overhead", 22.6, 42.5, 1.5, "overhead", inset=True),
    Shot("3  set-down and release", 42.5, 57.0, 2.0, "hero"),
)
TITLE_S = 3.0
END_S = 4.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", type=Path, required=True,
                        help="run_m4.py --out directory holding render_states.npz and summary.json")
    parser.add_argument("--out", type=Path, required=True, help="output .mp4")
    parser.add_argument("--frames", type=Path, default=None,
                        help="where to write representative stills (default: beside --out)")
    parser.add_argument("--no-inset", action="store_true",
                        help="alternative cut: overhead pour without the hero inset")
    return parser


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).astype("<f8").tobytes()).hexdigest()


def _load(replay: Path) -> tuple[np.ndarray, np.ndarray, float, dict]:
    states = np.load(replay / "render_states.npz")
    summary = json.loads((replay / "summary.json").read_text("utf-8"))
    qpos = states["qpos"]
    digest = _sha256(qpos)
    if digest != str(states["qpos_sha256"]):
        raise SystemExit(f"[film] qpos hash {digest} does not match the npz header")
    if digest != summary["determinism"]["qpos_sha256"]:
        raise SystemExit("[film] render_states.npz is not the replay summary.json describes")
    if not summary.get("passes") or not summary.get("release_evidence"):
        raise SystemExit("[film] refusing to cut a film from a replay that did not pass M4")
    if not summary.get("simulation_only"):
        raise SystemExit("[film] summary does not declare simulation_only")
    return qpos, states["stream_paths"], float(states["render_fps"]), summary


def _model():
    """The replay's compiled model, built exactly as run_m4.py builds it."""
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    resolution = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(resolution)
    print(resolution.banner())
    from mjlab.envs import ManagerBasedRlEnv

    import scenes
    import tasks

    env_cfg = tasks.load_env_cfg("jumper.barista", play=True)
    scenes.apply(env_cfg, "coffee_bar")
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    model = env.sim.mj_model
    env.close()
    return model


def _font(size: int):
    from PIL import ImageFont

    # Pillow's bundled Aileron Regular (CC0; see barista/SOURCES.md).
    return ImageFont.load_default(size=size)


def _overlay(frame: np.ndarray, label: str | None, speed: float | None) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, HEIGHT - 44, WIDTH, HEIGHT), fill=(0, 0, 0, 165))
    draw.text((WIDTH // 2, HEIGHT - 22), DISCLAIMER, font=_font(22),
              fill=(255, 255, 255, 235), anchor="mm")
    if label:
        text = label + (f"   ({speed:g}x)" if speed and speed != 1.0 else "   (1x)")
        box = draw.textbbox((24, 22), text, font=_font(26))
        draw.rectangle((box[0] - 12, box[1] - 8, box[2] + 12, box[3] + 8), fill=(0, 0, 0, 150))
        draw.text((24, 22), text, font=_font(26), fill=(255, 255, 255, 240))
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def _card(background: np.ndarray, lines: list[tuple[str, int]]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFilter

    image = Image.fromarray(background).filter(ImageFilter.GaussianBlur(6)).convert("RGBA")
    layer = Image.new("RGBA", image.size, (8, 6, 4, 190))
    draw = ImageDraw.Draw(layer)
    total = sum(size + 18 for _, size in lines)
    y = (HEIGHT - total) // 2
    for text, size in lines:
        draw.text((WIDTH // 2, y + size // 2), text, font=_font(size),
                  fill=(255, 255, 255, 245), anchor="mm")
        y += size + 18
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


class Camera:
    def __init__(self, mujoco, lookat, distance, azimuth, elevation):
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = lookat
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation


def _add_capsule(mujoco, scene, first, second, width, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    geom = scene.geoms[scene.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_initGeom(geom=geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
                        size=np.zeros(3), pos=np.zeros(3), mat=np.zeros(9),
                        rgba=np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom=geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
                         width=width, from_=first, to=second)


def _add_disc(mujoco, scene, centre, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    geom = scene.geoms[scene.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_initGeom(geom=geom, type=mujoco.mjtGeom.mjGEOM_CYLINDER.value,
                        size=np.asarray((radius, 0.0004, 0.0)), pos=np.asarray(centre),
                        mat=np.eye(3).reshape(-1), rgba=np.asarray(rgba, dtype=np.float32))


def _draw(mujoco, renderer, data, camera, path, trail, stream_width) -> np.ndarray:
    renderer.update_scene(data, camera=camera.cam)
    scene = renderer.scene
    for point in trail:
        _add_disc(mujoco, scene, point + np.asarray((0.0, 0.0, 0.0006)), 0.0016, TRAIL_RGBA)
    if path is not None:
        for first, second in itertools.pairwise(path):
            _add_capsule(mujoco, scene, first, second, stream_width, STREAM_RGBA)
    return renderer.render()


def main() -> int:
    args = _parser().parse_args()
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    import imageio_ffmpeg
    import mujoco
    from PIL import Image

    replay = args.replay.resolve()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = (args.frames or out.parent / (out.stem + "_frames")).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    qpos, stream_paths, render_fps, summary = _load(replay)

    model = _model()
    if model.nq != qpos.shape[1]:
        raise SystemExit(f"[film] model nq={model.nq} but the replay logged {qpos.shape[1]}")
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), HEIGHT)
    model.vis.quality.shadowsize = 4096
    data = mujoco.MjData(model)
    main_renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH, max_geom=20000)
    inset_renderer = mujoco.Renderer(model, height=INSET_H, width=INSET_W, max_geom=20000)

    bed = np.asarray(json.loads((REPO / "barista" / "trajectory" / "m4_spiral_pour.json")
                                .read_text("utf-8"))["stream"]["bed_centre_m"])
    views = {
        # Low three-quarter view from in front of the robot: robot, pitcher and
        # dripper all in frame, the pitcher on the near side of the claw.
        "hero": Camera(mujoco, (0.05, 0.20, 0.09), 0.66, 205.0, -26.0),
        # Straight down onto the coffee bed, robot at the bottom of the frame.
        "overhead": Camera(mujoco, (bed[0], bed[1] - 0.01, bed[2]), 0.25, 90.0, -89.0),
        # Inset during the pour: closer hero angle on the tilted pitcher.
        "close": Camera(mujoco, (0.08, 0.27, 0.19), 0.38, 225.0, -16.0),
    }

    landing = stream_paths[:, -1, :]
    has_stream = ~np.isnan(landing[:, 0])

    def state_at(time_s: float) -> int:
        return int(min(round(time_s * render_fps), len(qpos) - 1))

    def frame(index: int, view: str, inset: bool, trail: bool) -> np.ndarray:
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        path = None if not has_stream[index] else stream_paths[index]
        points = landing[: index + 1][has_stream[: index + 1]][::2] if trail else ()
        width = 0.0012 if view == "overhead" else 0.0017
        image = _draw(mujoco, main_renderer, data, views[view], path, points, width)
        if inset:
            small = _draw(mujoco, inset_renderer, data, views["close"], path, (), 0.0017)
            image = image.copy()
            x0, y0 = WIDTH - INSET_W - 24, 24
            image[y0 - 3:y0 + INSET_H + 3, x0 - 3:x0 + INSET_W + 3] = 235
            image[y0:y0 + INSET_H, x0:x0 + INSET_W] = small
        return image

    writer = imageio_ffmpeg.write_frames(
        str(out), size=(WIDTH, HEIGHT), fps=FPS, macro_block_size=1,
        codec="libx264", pix_fmt_out="yuv420p", quality=8,
    )
    writer.send(None)
    timeline: list[dict] = []
    stills: dict[str, np.ndarray] = {}
    out_frame = 0

    def emit(image: np.ndarray, tag: str | None = None) -> None:
        nonlocal out_frame
        writer.send(np.ascontiguousarray(image).tobytes())
        if tag:
            stills[tag] = image
        out_frame += 1

    stream = summary["stream"]
    torque = summary["torque"]
    payload = summary["payload"]
    try:
        title_bg = frame(0, "hero", False, False)
        title = _card(title_bg, [
            ("Jumper Barista", 64),
            ("a slow, centred spiral pour over a V60", 30),
            ("MuJoCo simulation of the 22-DoF Jumper crab robot", 26),
            ("Water is a particle/visual approximation. This is not a real-robot run.", 24),
        ])
        start = out_frame
        for _ in range(round(TITLE_S * FPS)):
            emit(title, "title")
        timeline.append({"shot": "title card", "out_start_s": start / FPS,
                         "out_end_s": out_frame / FPS})

        for shot in SHOTS:
            if args.no_inset and shot.inset:
                shot = Shot(shot.label, shot.start_s, shot.end_s, shot.speed, shot.view, False)
            start = out_frame
            count = round((shot.end_s - shot.start_s) / shot.speed * FPS)
            for k in range(count):
                sim_t = shot.start_s + k * shot.speed / FPS
                index = state_at(sim_t)
                image = frame(index, shot.view, shot.inset, shot.view == "overhead")
                image = _overlay(image, shot.label, shot.speed)
                tag = None
                if k == count // 2:
                    tag = f"{shot.view}_{shot.start_s:g}-{shot.end_s:g}s_mid"
                if shot.view == "overhead" and k == count - 1:
                    tag = "overhead_spiral_complete"
                emit(image, tag)
            timeline.append({
                "shot": shot.label, "view": shot.view, "inset_close_hero": shot.inset,
                "sim_start_s": shot.start_s, "sim_end_s": shot.end_s, "speed": shot.speed,
                "out_start_s": start / FPS, "out_end_s": out_frame / FPS,
            })

        end_bg = frame(len(qpos) - 1, "hero", False, False)
        if not torque["passes"] or not payload["passes"]:
            raise SystemExit("[film] the end card's torque/payload lines need passing audits")
        turns_line = (
            f"{stream['actual_spout_turn_count']:.2f} outward spiral turns at the spout,"
            f" {stream['landing_success_count']}/{stream['sample_count']}"
            " simulated stream samples landing in the bed"
        )
        torque_line = (
            f"{torque['joint_count']} joints within the servo curve, 300 ms peak and"
            " 1.2 N m budget (simulated torque log)"
        )
        payload_line = (
            f"pitcher + grip + water proxy {payload['design_total_kg'] * 1000:.0f} g"
            f" (limit {payload['design_limit_kg'] * 1000:.0f} g)"
        )
        end = _card(end_bg, [
            ("What the simulation measured", 40),
            (turns_line, 24),
            (torque_line, 24),
            (payload_line, 24),
            ("MuJoCo simulation, native:cpu. Particle/visual water. Not a real-robot run.", 24),
        ])
        start = out_frame
        for _ in range(round(END_S * FPS)):
            emit(end, "end_card")
        timeline.append({"shot": "end card", "out_start_s": start / FPS,
                         "out_end_s": out_frame / FPS})
    finally:
        main_renderer.close()
        inset_renderer.close()
        writer.close()

    still_paths = {}
    for tag, image in stills.items():
        path = frames_dir / f"{tag}.png"
        Image.fromarray(image).save(path)
        still_paths[tag] = str(path)

    manifest = {
        "simulation_only": True,
        "water_model": summary["water_model"],
        "film": str(out), "fps": FPS, "frames": out_frame, "duration_s": out_frame / FPS,
        "size": [WIDTH, HEIGHT], "audio": None,
        "variant": "no_inset" if args.no_inset else "inset",
        "source_replay": str(replay),
        "source_commit_sha": summary["commit_sha"],
        "source_qpos_sha256": summary["determinism"]["qpos_sha256"],
        "source_trajectory_sha256": summary["trajectory_sha256"],
        "timeline": timeline, "stills": still_paths,
    }
    (out.parent / (out.stem + ".json")).write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    print(f"[film] {out} {out_frame} frames, {out_frame / FPS:.2f} s at {FPS} fps")
    for row in timeline:
        print(f"[film]   {row['out_start_s']:6.2f}-{row['out_end_s']:6.2f} s  {row['shot']}")
    for tag, path in still_paths.items():
        print(f"[film] still {tag}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
