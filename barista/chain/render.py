#!/usr/bin/env python3
"""Verifier video and per-beat stills of a run_chain output (offscreen, simulation only).

Re-renders the saved render-rate states (no re-simulation) through the cafe's wide camera,
drawing the logged ballistic stream exactly as run_m4 does. One still per beat, taken at the
beat's midpoint.

    python barista/chain/render.py out/chain/run1
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def _by_path(stem: str):
    name = f"barista_chain_{stem}"
    if name not in sys.modules:
        spec = _ilu.spec_from_file_location(name, Path(__file__).with_name(f"{stem}.py"))
        module = _ilu.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


common = _by_path("common")
footprint = _by_path("footprint")

WIDTH, HEIGHT = 960, 540


def _camera(model, mujoco, suffix: str) -> int:
    hits = [i for i in range(model.ncam)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or "").endswith(suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"expected one camera ending {suffix!r}, found {hits}")
    return hits[0]


def _draw_stream(renderer, mujoco, points) -> None:
    if points is None or np.isnan(points).any():
        return
    for first, second in itertools.pairwise(points):
        renderer.scene.ngeom += 1
        geom = renderer.scene.geoms[renderer.scene.ngeom - 1]
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_initGeom(geom=geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE.value, size=np.zeros(3),
                            pos=np.zeros(3), mat=np.zeros(9),
                            rgba=np.asarray((0.18, 0.58, 0.82, 0.82), dtype=np.float32))
        mujoco.mjv_connector(geom=geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE.value, width=0.0017,
                             from_=first, to=second)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--camera", default="cafe_wide")
    parser.add_argument("--stride", type=int, default=1, help="render every Nth state")
    args = parser.parse_args()
    common.headless()
    import imageio_ffmpeg
    import mujoco
    from PIL import Image

    run = args.run.resolve()
    rs = np.load(run / "render_states.npz")
    beats = json.loads((run / "beats.json").read_text())
    qpos, times, streams = rs["qpos"], rs["time_s"], rs["stream_paths"]
    env = footprint.build_model(beats.get("scene", common.SCENE))
    try:
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        cam = _camera(model, mujoco, args.camera)
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        fps = float(rs["render_fps"]) / args.stride
        video = run / "chain_wide.mp4"
        writer = imageio_ffmpeg.write_frames(str(video), size=(WIDTH, HEIGHT), fps=fps,
                                             macro_block_size=1, codec="libx264",
                                             pix_fmt_out="yuv420p", quality=7)
        writer.send(None)

        def frame(i):
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            _draw_stream(renderer, mujoco, streams[i])
            return renderer.render()

        try:
            for i in range(0, len(qpos), args.stride):
                writer.send(np.ascontiguousarray(frame(i)).tobytes())
        finally:
            writer.close()
        stills = {}
        stills_dir = run / "stills"
        stills_dir.mkdir(exist_ok=True)
        for k, b in enumerate(beats["beats"]):
            t = 0.5 * (b["t_start_s"] + b["t_end_s"])
            i = int(np.argmin(np.abs(times - t)))
            path = stills_dir / f"{k:02d}_{b['beat']}.png"
            Image.fromarray(frame(i)).save(path)
            stills[f"{k:02d}_{b['beat']}"] = {"path": str(path), "t_s": float(times[i])}
        renderer.close()
    finally:
        env.close()
    record = {"video": str(video), "frames": len(range(0, len(qpos), args.stride)), "fps": fps,
              "camera": args.camera, "size": [WIDTH, HEIGHT], "stills": stills,
              "qpos_sha256": str(rs["qpos_sha256"]), "simulation_only": True}
    (run / "render.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("video", "frames", "fps")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
