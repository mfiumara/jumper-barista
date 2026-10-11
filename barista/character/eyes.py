#!/usr/bin/env python3
"""Display-only eye panel for Jumper: five expressions on the display module's face.

Jumper's model has a ``display_module_link`` (the black window on the front of the shell)
but nothing drawn on it. This module adds one thin, textured panel just in front of that
window, with one material per expression, and a timeline that switches the material while
a replay is rendered. Simulation-only and visual-only:

* the panel is a visual geom: ``contype=0``, ``conaffinity=0``, ``density=0``, ``mass=0``,
  in the robot's visual group (2), attached to ``display_module_link``, whose inertia is
  given explicitly in ``assets/jumper/jumper.xml`` and so is not recomputed from geoms;
* switching an expression writes ``model.geom_matid`` for the panel only, which the
  physics never reads.

``python barista/character/eyes.py render --out DIR`` renders one PNG per expression.
``python barista/character/eyes.py m4-hash-check --out DIR`` replays the accepted M4 pour
(``barista/verification/run_m4.py --no-render``) with the panel installed and compares
the motion/torque/stream CSVs and the qpos hash with the accepted M4 evidence.

Library use::

    eyes.install_on_env_cfg(env_cfg)        # after scenes.apply(...), before the env
    panel = eyes.EyePanel(model)            # after the model is compiled
    timeline = eyes.EyeTimeline([(0.0, "neutral"), (2.4, "surprised"), (6.0, "happy")])
    panel.set(timeline.expression_at(t))    # before each rendered frame
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]

EXPRESSIONS: tuple[str, ...] = ("neutral", "surprised", "listening", "happy", "wink")
DEFAULT_EXPRESSION = "neutral"

DISPLAY_BODY = "display_module_link"
PANEL_GEOM = "barista_eye_panel"
MATERIAL_PREFIX = "barista_eye_"
TEXTURE_PREFIX = "barista_eye_tex_"

# The display window's face, measured by ray casting the visual mesh of
# display_module_link from the front (249 hits): centroid at (0.07979, 0, 0.04928) m in the
# base frame, outward normal (0.822, 0, 0.569), about 35 degrees above the robot's +x.
# The body sits at (0.0802718, 1.01591e-05, 0.0495238) with identity orientation, so the
# centroid is (-0.00048, 0, -0.00024) in the body frame. The panel floats 1.5 mm out from
# that centroid so that it is not z-fighting with the curved window.
FACE_NORMAL = np.array([0.822205961, 0.0, 0.569190088])
FACE_CENTROID_BODY = np.array([-0.00048, -0.00001, -0.00024])
PANEL_STANDOFF_M = 0.0015
# Half sizes: across the face (robot y), up the face, thickness.
PANEL_HALF_SIZE = (0.036, 0.012, 0.0002)
# Texture resolution, 6 px per mm of panel.
TEXTURE_SIZE = (432, 144)  # width, height

BACKGROUND = (6, 8, 12)
GLOW = (120, 232, 255)
CORE = (225, 252, 255)


def _panel_frame() -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, quat) of the panel in the display body's frame.

    The panel's local x runs across the face from the viewer's left to right (robot -y),
    local y runs up the face, and local z points out of the face.
    """
    normal = FACE_NORMAL / np.linalg.norm(FACE_NORMAL)
    across = np.array([0.0, -1.0, 0.0])
    up = np.cross(normal, across)
    rotation = np.column_stack([across, up, normal])
    if np.linalg.det(rotation) < 0:
        raise RuntimeError("eye panel frame is not right-handed")
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
    pos = FACE_CENTROID_BODY + PANEL_STANDOFF_M * normal
    return pos, quat


# --------------------------------------------------------------------------- textures


def _draw(expression: str, size: tuple[int, int] = TEXTURE_SIZE) -> np.ndarray:
    """Draw one expression as an RGB uint8 image (height, width, 3), viewer's orientation."""
    from PIL import Image, ImageDraw, ImageFilter

    if expression not in EXPRESSIONS:
        raise ValueError(f"unknown eye expression {expression!r}; expected one of {EXPRESSIONS}")
    width, height = size
    scale = 4  # supersample, then downscale for smooth edges
    big = (width * scale, height * scale)
    mask = Image.new("L", big, 0)
    draw = ImageDraw.Draw(mask)
    w, h = big
    centres = (w * 0.30, w * 0.70)
    eye_w, eye_h = w * 0.13, h * 0.62
    stroke = int(h * 0.11)

    def pill(cx: float, cy: float, ew: float, eh: float) -> None:
        draw.rounded_rectangle(
            (cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2), radius=ew / 2, fill=255
        )

    def arc_up(cx: float, cy: float, ew: float) -> None:
        # "^"-shaped closed-happy eye: upper half of an ellipse, drawn as a thick arc.
        box = (cx - ew * 0.75, cy - ew * 0.55, cx + ew * 0.75, cy + ew * 0.95)
        draw.arc(box, start=200, end=340, fill=255, width=stroke)

    cy = h * 0.5
    if expression == "neutral":
        for cx in centres:
            pill(cx, cy, eye_w, eye_h)
    elif expression == "surprised":
        radius = h * 0.40
        for cx in centres:
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
            inner = radius * 0.42
            draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=0)
    elif expression == "listening":
        # Attentive: slightly smaller eyes looking up and to one side, one brow raised.
        for index, cx in enumerate(centres):
            pill(cx + w * 0.02, h * 0.42, eye_w * 0.95, eye_h * 0.78)
            brow_y = h * (0.07 if index == 1 else 0.12)
            draw.line(
                (cx - eye_w * 0.6, brow_y + h * 0.03, cx + eye_w * 0.7, brow_y),
                fill=255, width=int(stroke * 0.6),
            )
    elif expression == "happy":
        for cx in centres:
            arc_up(cx, h * 0.55, eye_w * 1.6)
    elif expression == "wink":
        pill(centres[0], cy, eye_w, eye_h)
        arc_up(centres[1], h * 0.55, eye_w * 1.6)
    glow = mask.filter(ImageFilter.GaussianBlur(radius=h * 0.05))
    mask_a = np.asarray(mask, dtype=np.float32) / 255.0
    glow_a = np.asarray(glow, dtype=np.float32) / 255.0
    image = np.empty((h, w, 3), dtype=np.float32)
    for channel in range(3):
        value = BACKGROUND[channel] + (GLOW[channel] - BACKGROUND[channel]) * glow_a
        value = value + (CORE[channel] - value) * mask_a
        image[..., channel] = value
    small = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    return np.asarray(small, dtype=np.uint8)


def expression_texture(expression: str, size: tuple[int, int] = TEXTURE_SIZE) -> np.ndarray:
    """Public: the RGB image (height, width, 3) for an expression, as seen by the viewer."""
    return _draw(expression, size)


# --------------------------------------------------------------------------- model edit


def _find_body(spec: Any, suffix: str) -> Any:
    import mujoco

    matches = [
        body for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)
        if body.name == suffix or body.name.endswith(("/" + suffix, "_" + suffix))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one body ending {suffix!r}; found {len(matches)}")
    return matches[0]


def install(spec: Any, initial: str = DEFAULT_EXPRESSION) -> None:
    """Add the eye textures, materials and the panel geom to an ``MjSpec`` in place.

    The image is rotated 180 degrees before upload: on this box face MuJoCo's 2D texture
    coordinates run against the panel frame on both axes. With that, every expression is
    upright and unmirrored for a viewer in front of the robot (the wink's open eye is on
    the viewer's left; checked visually in the render command's contact sheet).
    """
    import mujoco

    if initial not in EXPRESSIONS:
        raise ValueError(f"unknown eye expression {initial!r}")
    width, height = TEXTURE_SIZE
    for expression in EXPRESSIONS:
        texture = spec.add_texture(
            name=TEXTURE_PREFIX + expression,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            width=width,
            height=height,
            nchannel=3,
        )
        texture.data = np.ascontiguousarray(_draw(expression)[::-1, ::-1]).tobytes()
        material = spec.add_material(name=MATERIAL_PREFIX + expression)
        textures = list(material.textures)
        textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = TEXTURE_PREFIX + expression
        material.textures = textures
        material.emission = 0.9
        material.specular = 0.1
        material.shininess = 0.2
        material.rgba = [1.0, 1.0, 1.0, 1.0]
    pos, quat = _panel_frame()
    body = _find_body(spec, DISPLAY_BODY)
    body.add_geom(
        name=PANEL_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(PANEL_HALF_SIZE),
        pos=pos.tolist(),
        quat=quat.tolist(),
        contype=0,
        conaffinity=0,
        density=0.0,
        mass=0.0,
        group=2,
        material=MATERIAL_PREFIX + initial,
        rgba=[1.0, 1.0, 1.0, 1.0],
    )


def install_on_env_cfg(env_cfg: Any, initial: str = DEFAULT_EXPRESSION) -> None:
    """Chain :func:`install` after any ``SceneCfg.spec_fn`` already installed (scene, task)."""
    previous: Callable[[Any], None] | None = getattr(env_cfg.scene, "spec_fn", None)

    def edit(spec: Any) -> None:
        if previous is not None:
            previous(spec)
        install(spec, initial)

    env_cfg.scene.spec_fn = edit


class EyePanel:
    """Switch the compiled panel's expression. Writes ``model.geom_matid`` only."""

    def __init__(self, model: Any) -> None:
        import mujoco

        geoms = [
            index for index in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").endswith(
                PANEL_GEOM
            )
        ]
        if len(geoms) != 1:
            raise RuntimeError(f"expected one {PANEL_GEOM!r} geom; found {len(geoms)}")
        self.model = model
        self.geom = geoms[0]
        self.materials: dict[str, int] = {}
        for expression in EXPRESSIONS:
            names = [
                index for index in range(model.nmat)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, index) or "")
                .endswith(MATERIAL_PREFIX + expression)
            ]
            if len(names) != 1:
                raise RuntimeError(f"expected one material for {expression!r}; found {names}")
            self.materials[expression] = names[0]

    @property
    def expression(self) -> str:
        current = int(self.model.geom_matid[self.geom])
        return next(name for name, index in self.materials.items() if index == current)

    def set(self, expression: str) -> None:
        if expression not in self.materials:
            raise ValueError(f"unknown eye expression {expression!r}; expected {EXPRESSIONS}")
        self.model.geom_matid[self.geom] = self.materials[expression]


class EyeTimeline:
    """Piecewise-constant expression schedule: a list of ``(start_s, expression)`` cues."""

    def __init__(
        self, cues: Iterable[tuple[float, str]], default: str = DEFAULT_EXPRESSION
    ) -> None:
        ordered = sorted((float(start), str(name)) for start, name in cues)
        for _, name in ordered:
            if name not in EXPRESSIONS:
                raise ValueError(f"unknown eye expression {name!r}; expected {EXPRESSIONS}")
        if default not in EXPRESSIONS:
            raise ValueError(f"unknown default expression {default!r}")
        self.cues = ordered
        self.default = default
        self._starts = [start for start, _ in ordered]

    def expression_at(self, time_s: float) -> str:
        index = bisect.bisect_right(self._starts, float(time_s)) - 1
        return self.cues[index][1] if index >= 0 else self.default

    @classmethod
    def from_manifest(
        cls,
        manifest: dict,
        starts_s: dict[str, float],
        hold_s: float = 0.6,
        rest: str = DEFAULT_EXPRESSION,
    ) -> EyeTimeline:
        """Cue each manifest line's ``eye_expression`` at its start time (seconds into the
        film, chosen by the editor) and return to ``rest`` ``hold_s`` after the line ends.
        """
        cues: list[tuple[float, str]] = []
        for line in manifest["lines"]:
            if line["id"] not in starts_s:
                continue
            start = float(starts_s[line["id"]])
            cues.append((start, line["eye_expression"]))
            cues.append((start + float(line["duration_s"]) + hold_s, rest))
        return cls(cues, default=rest)


# --------------------------------------------------------------------------- commands


def _render(out: Path) -> list[Path]:
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    import mujoco
    from PIL import Image

    spec = mujoco.MjSpec.from_file(str(REPO / "assets" / "jumper" / "jumper.xml"))
    install(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    panel = EyePanel(model)
    renderer = mujoco.Renderer(model, 540, 720)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.07, 0.0, 0.045]
    camera.distance = 0.17
    camera.azimuth = 180.0  # looking back along -x, i.e. facing the robot
    camera.elevation = -28.0
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    tiles = []
    for expression in EXPRESSIONS:
        panel.set(expression)
        renderer.update_scene(data, camera)
        pixels = renderer.render()
        path = out / f"eyes_{expression}.png"
        Image.fromarray(pixels).save(path)
        Image.fromarray(expression_texture(expression)).save(out / f"eyes_{expression}_tex.png")
        paths.append(path)
        tiles.append(pixels)
    sheet = np.concatenate(
        [np.concatenate(tiles[:3], axis=1),
         np.concatenate([*tiles[3:], np.zeros_like(tiles[0])], axis=1)],
        axis=0,
    )
    sheet_path = out / "eyes_contact_sheet.png"
    Image.fromarray(sheet).save(sheet_path)
    renderer.close()
    return [*paths, sheet_path]


ACCEPTED_M4 = {
    # Accepted M4 replay (candidate cdae494, re-verified for the M5 release at 287861a):
    # run_m4.py on the coffee_bar scene, native:cpu.
    "qpos_sha256": "7adf1e8e12e2e37bfe8c0a794b6ec1b47d7cc7d21fd83589332c8dac4f233f81",
    "motion_substeps.csv": "65a1e6231acf2dbb9a4c4ae9310c5f7da7932f10f8d061611e912b90f168bf72",
    "torque_substeps.csv": "7c6a39e7601d93b88ef4798d29d8c49b4aa946a8a7b56a7f7fcd44c24a86a8bb",
    "stream_samples.csv": "561a87aaa138a3b9cdb25928d7c92453f94643f0ce220dfae5fa187079f9024b",
}


def _m4_hash_check(out: Path, extra: Sequence[str]) -> int:
    """Run run_m4.py in-process with the eye panel installed on its env config."""
    import importlib.util

    import scenes

    installed: list[int] = []
    original_apply = scenes.apply

    def apply_with_eyes(env_cfg: Any, scene_id: str, **kwargs: Any) -> Any:
        result = original_apply(env_cfg, scene_id, **kwargs)
        previous = getattr(env_cfg.scene, "spec_fn", None)

        def edit(spec: Any) -> None:
            if previous is not None:
                previous(spec)
            install(spec, "happy")
            installed.append(1)

        env_cfg.scene.spec_fn = edit
        return result

    scenes.apply = apply_with_eyes
    path = REPO / "barista" / "verification" / "run_m4.py"
    spec = importlib.util.spec_from_file_location("barista_run_m4_with_eyes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = sys.argv
    sys.argv = [str(path), "--out", str(out), *extra]
    try:
        code = module.main()
    finally:
        sys.argv = argv
        scenes.apply = original_apply
    if not installed:
        raise RuntimeError("the eye panel was never installed into the M4 model")
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    observed = {"qpos_sha256": summary["determinism"]["qpos_sha256"]}
    for name in ("motion_substeps.csv", "torque_substeps.csv", "stream_samples.csv"):
        observed[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    comparison = {
        key: {"accepted": ACCEPTED_M4[key], "with_eyes": observed[key],
              "identical": ACCEPTED_M4[key] == observed[key]}
        for key in ACCEPTED_M4
    }
    report = {
        "eye_panel_installed": bool(installed),
        "run_m4_exit": code,
        "m4_passes": summary.get("passes"),
        "comparison": comparison,
        "all_identical": all(item["identical"] for item in comparison.values()),
    }
    (out / "eyes_hash_check.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    for key, item in comparison.items():
        print(f"[eyes] {key}: accepted={item['accepted']} with_eyes={item['with_eyes']} "
              f"identical={item['identical']}")
    print(f"[eyes] all identical={report['all_identical']}")
    return 0 if (code == 0 and report["all_identical"]) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="render one PNG per expression")
    render.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("m4-hash-check", help="replay M4 with eyes; compare hashes")
    check.add_argument("--out", type=Path, required=True)
    args, extra = parser.parse_known_args(argv)
    if args.command == "render":
        if extra:
            parser.error(f"unrecognised arguments: {extra}")
        for path in _render(args.out.resolve()):
            print(f"[eyes] artifact: {path}")
        return 0
    return _m4_hash_check(args.out.resolve(), extra or ["--no-render"])


if __name__ == "__main__":
    raise SystemExit(main())
