#!/usr/bin/env python3
"""Render an existing Jumper policy replay without opening a window.

This is a small M1 evidence tool, not another replay implementation. It selects
``native:cpu`` and reuses the policy rollout and backend-independent qpos renderer
already exercised by the dance exporter. Generated media goes to an explicit
output directory so it never becomes repository debris.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_TASK = "jumper.gesture_hello"
DEFAULT_CHECKPOINT = Path("tasks/jumper/gesture_hello/out/example/model_999.pt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")

    # MuJoCo chooses its GL implementation when its first renderer is created.
    # CGL is macOS's offscreen path and does not create or foreground a window.
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

    import imageio_ffmpeg
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks
    from tasks.jumper.dance.export_media import _render, _rollout

    resolution = resolve(
        backend="native", device="cpu", num_envs=1, training=False
    )
    use_backend(resolution)
    print(resolution.banner())

    spec = tasks.get(args.task)
    asset = spec.resolve_asset(None)
    qpos, model, fps, frames_total, stopped_early = _rollout(
        args.task, asset, checkpoint, resolution
    )
    if stopped_early is not None:
        print(
            f"[barista] replay terminated at frame {stopped_early} of "
            f"{frames_total}",
            file=sys.stderr,
        )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    video = out / "replay.mp4"
    frame = out / "replay.png"
    _render(model, qpos, fps, video)

    # Extract the middle encoded frame: this proves the MP4 both contains a
    # decodable picture and frames the robot away from lead-in/lead-out edges.
    timestamp = len(qpos) / fps / 2.0
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(frame),
        ],
        check=True,
    )
    print(
        f"[barista] rendered {len(qpos)} frames at {fps:g} fps "
        f"({len(qpos) / fps:.2f} s)"
    )
    print(f"[barista] video: {video}")
    print(f"[barista] frame: {frame}")


if __name__ == "__main__":
    main()
