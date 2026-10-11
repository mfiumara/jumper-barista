#!/usr/bin/env python3
"""Exercise the Dapper Tricolore outfit (A2b) in simulation: physics, motion, clearance, renders.

Subcommands (all headless; MuJoCo simulation only, nothing here ran on the real robot):

    m4        run the unmodified barista/verification/run_m4.py with the outfit applied
    compare   compare a dressed M4 replay against a stock one, file by file and hash by hash
    rollout   roll out the stock dance (brazilian) and jump policies with the outfit applied
    clearance outfit vs claws / legs / pitcher over M4 + dance/jump rollouts + references
    render    front 3/4, side, pour pose, a dance frame and a jump-landing frame

    python barista/outfit/outfit_check.py m4 --out <dir>
    python barista/outfit/outfit_check.py compare --stock <dir> --dressed <dir> --out <json>
    python barista/outfit/outfit_check.py rollout --out <dir>
    python barista/outfit/outfit_check.py clearance --replay <dressed m4> --rollouts <dir> --out <json>
    python barista/outfit/outfit_check.py render --replay <dressed m4> --rollouts <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE), str(REPO / "barista" / "film")):
    if p not in sys.path:
        sys.path.insert(0, p)

import outfit

DANCE_TASK = "jumper.dance_brazilian"
DANCE_CKPT = REPO / "tasks/jumper/dance_brazilian/out/example/model_1999.pt"
JUMP_TASK = "jumper.jump"
JUMP_CKPT = REPO / "tasks/jumper/jump/out/example/model_3100.pt"
M4_FILES = ("motion_substeps.csv", "torque_substeps.csv", "stream_samples.csv")
WIDTH, HEIGHT = 960, 720
POUR_TIME_S = 30.0
CLEARANCE_MIN_M = 0.010
SEARCH_M = 0.030
#: Sampling rate for the five kinematic dance references (the policy rollouts and the
#: jump reference are swept at their native rates).
REF_HZ = 50.0
#: Target sample spacing: one sample per 1 mm voxel of each visual mesh.
TARGET_VOXEL_M = 0.001
FOOTER = ("MuJoCo simulation  |  display-only outfit on the Jumper model  |  "
          "not the real robot")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _headless():
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


# ── m4 / compare ──────────────────────────────────────────────────────────────


def cmd_m4(args) -> int:
    _headless()
    built = []
    original = outfit.add_to_spec

    def counting(spec):
        original(spec)
        built.append(1)

    outfit.add_to_spec = counting
    argv = sys.argv
    sys.argv = [str(REPO / "barista/verification/run_m4.py"), "--out", str(args.out),
                "--save-render-states"]
    try:
        with outfit.installed():
            try:
                runpy.run_path(sys.argv[0], run_name="__main__")
            except SystemExit as exc:
                code = exc.code or 0
            else:
                code = 0
    finally:
        sys.argv = argv
        outfit.add_to_spec = original
    if not built:
        raise SystemExit("[outfit] run_m4 never built a robot spec: the outfit was not applied")
    (Path(args.out) / "outfit_applied.json").write_text(json.dumps(
        {"outfit": outfit.NAME, "robot_specs_dressed": len(built),
         "run_m4_exit": code}, indent=2) + "\n")
    print(f"[outfit] run_m4 exit={code}, robot specs dressed={len(built)}")
    return int(code)


def _npz_qpos_sha(path: Path) -> str:
    with np.load(path) as z:
        return hashlib.sha256(np.asarray(z["qpos"]).astype("<f8").tobytes()).hexdigest()


def cmd_compare(args) -> int:
    stock, dressed = Path(args.stock), Path(args.dressed)
    s_sum = json.loads((stock / "summary.json").read_text())
    d_sum = json.loads((dressed / "summary.json").read_text())
    rows = {}
    for name in M4_FILES:
        rows[name] = (_sha(stock / name), _sha(dressed / name))
    rows["summary.determinism.qpos_sha256"] = (s_sum["determinism"]["qpos_sha256"],
                                               d_sum["determinism"]["qpos_sha256"])
    rows["render_states.npz qpos (<f8)"] = (_npz_qpos_sha(stock / "render_states.npz"),
                                            _npz_qpos_sha(dressed / "render_states.npz"))

    def strip(d):  # the summary minus fields that name the checkout or output path
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items()
                    if k not in ("commit_sha", "trajectory_source", "artifacts")}
        if isinstance(d, list):
            return [strip(v) for v in d]
        return d

    s_c = json.dumps(strip(s_sum), sort_keys=True).encode()
    d_c = json.dumps(strip(d_sum), sort_keys=True).encode()
    rows["summary.json minus commit/paths"] = (hashlib.sha256(s_c).hexdigest(),
                                               hashlib.sha256(d_c).hexdigest())
    report = {"stock": str(stock), "dressed": str(dressed),
              "stock_commit": s_sum.get("commit_sha"), "dressed_commit": d_sum.get("commit_sha"),
              "dressed_passed": d_sum.get("passed", d_sum.get("pass")),
              "hashes": {k: {"stock": a, "dressed": b, "identical": a == b}
                         for k, (a, b) in rows.items()}}
    if args.accepted_qpos:
        report["accepted_qpos_sha256"] = args.accepted_qpos
        report["dressed_matches_accepted"] = (
            d_sum["determinism"]["qpos_sha256"] == args.accepted_qpos)
    report["all_identical"] = all(v["identical"] for v in report["hashes"].values()) and \
        report.get("dressed_matches_accepted", True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for k, v in report["hashes"].items():
        print(f"[compare] {k}: {'IDENTICAL' if v['identical'] else 'DIFFERENT'} "
              f"{v['dressed'][:16]}")
    print(f"[compare] all_identical={report['all_identical']}")
    return 0 if report["all_identical"] else 1


# ── rollout ───────────────────────────────────────────────────────────────────


def _jump_rollout(res, seconds: float):
    from dataclasses import asdict

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import tasks
    from tasks.jumper.dance.export_media import _qpos_of

    env_cfg = tasks.load_env_cfg(JUMP_TASK, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = tasks.load_agent_cfg(JUMP_TASK)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = tasks.load_runner_cls(JUMP_TASK) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=res.device)
        runner.load(str(JUMP_CKPT), load_cfg={"actor": True}, strict=True,
                    map_location=res.device)
        policy = runner.get_inference_policy(device=res.device)
        robot = env.scene["robot"]
        dt = float(env.step_dt)
        n = round(seconds / dt)
        obs, _ = wrapped.reset()
        qpos = np.zeros((n, env.sim.mj_model.nq))
        with torch.inference_mode():
            for t in range(n):
                qpos[t] = _qpos_of(env, robot)
                obs, _, dones, _ = wrapped.step(policy(obs))
                if bool(dones[0]):
                    qpos = qpos[: t + 1]
                    break
        return qpos, 1.0 / dt
    finally:
        env.close()


def cmd_rollout(args) -> int:
    _headless()
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    from tasks.jumper.dance.export_media import _rollout

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(res)
    print(res.banner())
    meta = {}
    with outfit.installed():
        qpos, model, fps, total, stopped = _rollout(DANCE_TASK, None, DANCE_CKPT, res)
        n_out = len(outfit.geom_ids(model))
        np.savez(out / "dance.npz", qpos=qpos, fps=fps)
        meta["dance"] = {"task": DANCE_TASK, "checkpoint": str(DANCE_CKPT.relative_to(REPO)),
                         "frames": len(qpos), "fps": fps, "frames_total": total,
                         "stopped_early": stopped, "outfit_geoms_in_model": n_out}
        print(f"[rollout] dance: {len(qpos)} frames @ {fps:g} Hz, stopped_early={stopped}")
        qpos, fps = _jump_rollout(res, args.jump_seconds)
        np.savez(out / "jump.npz", qpos=qpos, fps=fps)
        meta["jump"] = {"task": JUMP_TASK, "checkpoint": str(JUMP_CKPT.relative_to(REPO)),
                        "frames": len(qpos), "fps": fps}
        print(f"[rollout] jump: {len(qpos)} frames @ {fps:g} Hz")
    for k in ("dance", "jump"):
        meta[k]["qpos_sha256"] = _npz_qpos_sha(out / f"{k}.npz")
    (out / "rollouts.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


# ── shared model for clearance / render ───────────────────────────────────────


def _dressed_m4_model():
    """The M4 replay model (coffee_bar scene + pitcher), dressed; as make_film builds it."""
    from make_film import _model

    with outfit.installed():
        model = _model()
    if not outfit.geom_ids(model):
        raise RuntimeError("outfit geoms missing from the replay model")
    return model


def _robot_model(stage: bool = False):
    """The bare dressed robot for the dance/jump rollouts' qpos; ``stage`` adds a
    render-only floor and light (no robot geometry changes)."""
    import mujoco

    from tasks.jumper.common.constants import get_spec

    spec = get_spec()
    outfit.add_to_spec(spec)
    if stage:
        spec.worldbody.add_geom(name="stage_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[3, 3, 0.1], rgba=[0.32, 0.24, 0.18, 1],
                                contype=0, conaffinity=0)
        spec.worldbody.add_light(pos=[0.4, -0.6, 1.8], dir=[-0.2, 0.3, -1.0],
                                 diffuse=[0.6, 0.58, 0.55], castshadow=True)
    return spec.compile()


def _load_replay(replay: Path):
    from make_film import _load

    return _load(replay.resolve())


# ── clearance ─────────────────────────────────────────────────────────────────


def _targets(model, mujoco, shell_rigid, pitcher_body):
    """Moving geoms by group, as (geom id, local points): visual meshes vertex+face-centroid
    sampled; primitives surface-sampled."""
    from concepts import _surface

    groups = {"claws": [], "legs": [], "pitcher": []}
    out_ids = set(outfit.geom_ids(model))
    for g in range(model.ngeom):
        if g in out_ids:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        short = body.split("/", 1)[-1]
        if body == pitcher_body:
            group = "pitcher"
        elif short in shell_rigid or short in ("floating_base", "world") or \
                not (body.startswith("robot/") or "/" not in body and short.startswith(
                    ("LF_", "RF_", "LM_", "RM_", "LR_", "RR_", "LH_", "RH_"))):
            continue
        else:
            group = "claws" if short.startswith(("LF_", "RF_")) else "legs"
        if group != "pitcher" and model.geom_group[g] != outfit.VISUAL_GROUP:
            continue  # robot collision meshes duplicate the visual ones; check what is seen
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            v0, nv = model.mesh_vertadr[m], model.mesh_vertnum[m]
            f0, nf = model.mesh_faceadr[m], model.mesh_facenum[m]
            verts = np.asarray(model.mesh_vert[v0:v0 + nv], float)
            faces = np.asarray(model.mesh_face[f0:f0 + nf])
            tri = verts[faces]
            meshdata = (verts, faces)
            # vertices, face centroids and edge midpoints: denser than vertices alone
            pts = np.concatenate((verts, tri.mean(1), (tri[:, 0] + tri[:, 1]) / 2,
                                  (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 2] + tri[:, 0]) / 2))
        else:
            kind = mujoco.mjtGeom(model.geom_type[g]).name.removeprefix("mjGEOM_").lower()
            if kind not in ("sphere", "ellipsoid", "capsule", "cylinder", "box"):
                continue
            pts = _surface(kind, np.asarray(model.geom_size[g], float), 0.0015)
            meshdata = None
        keys = np.floor(pts / TARGET_VOXEL_M).astype(np.int64)
        pts = pts[np.sort(np.unique(keys, axis=0, return_index=True)[1])]
        groups[group].append((g, pts, float(np.linalg.norm(pts, axis=1).max()), meshdata))
    return groups


def _outfit_points(model, mujoco, data, shell):
    """Outfit surface samples (1.5 mm) in the shell frame, with owner part, per layer."""
    from concepts import _surface

    layers = {"proud": ([], []), "decal": ([], [])}
    rot, org = data.xmat[shell].reshape(3, 3), data.xpos[shell]
    for g in outfit.geom_ids(model):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g).rsplit("/", 1)[-1]
        kind = mujoco.mjtGeom(model.geom_type[g]).name.removeprefix("mjGEOM_").lower()
        local = _surface(kind, np.asarray(model.geom_size[g], float), 0.0015)
        world = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        pts = (world - org) @ rot
        key = "decal" if name.startswith(outfit.PREFIX + "tache") else "proud"
        layers[key][0].append(pts)
        layers[key][1].extend([name.removeprefix(outfit.PREFIX)] * len(pts))
    return {k: (np.concatenate(v[0]), v[1]) for k, v in layers.items()}


def _part_frames():
    import mujoco

    out = []
    for g in outfit.geoms():
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.asarray(g.quat, float))
        out.append((g.part, "decal" if g.decal else "proud", g.kind, np.asarray(g.size),
                    np.asarray(g.pos), mat.reshape(3, 3)))
    return out


_PARTS = None


def _inside_parts(points: np.ndarray, layer: str) -> set[str]:
    """Names of the outfit parts of ``layer`` containing any of ``points`` (shell frame)."""
    global _PARTS
    if _PARTS is None:
        _PARTS = _part_frames()
    found = set()
    for name, lay, kind, size, pos, mat in _PARTS:
        if lay != layer:
            continue
        p = (points - pos) @ mat
        if kind == "box":
            ins = np.all(np.abs(p) < size, axis=1)
        elif kind == "cylinder":
            ins = (p[:, 0] ** 2 + p[:, 1] ** 2 < size[0] ** 2) & (np.abs(p[:, 2]) < size[1])
        elif kind == "ellipsoid":
            ins = ((p / size) ** 2).sum(1) < 1.0
        else:
            raise ValueError(kind)
        if ins.any():
            found.add(name)
    return found


def _behind_decals(points: np.ndarray) -> set[str]:
    """Decals with a target point inside the robot's own face behind them (the target
    passes through the stock face, independent of the outfit)."""
    global _PARTS
    if _PARTS is None:
        _PARTS = _part_frames()
    found = set()
    for name, lay, _kind, size, pos, mat in _PARTS:
        if lay != "decal":
            continue
        p = (points - pos) @ mat
        r2 = (p[:, 1] / size[1]) ** 2 + (p[:, 2] / size[2]) ** 2
        if np.any((r2 < 1.0) & (p[:, 0] < -size[0]) & (p[:, 0] > -0.012)):
            found.add(name)
    return found


_BARY = None


def _dense_triangles(meshdata, xmat, xpos, org, rot):
    """Target mesh triangles near the face, sampled on a 40x40 barycentric grid."""
    global _BARY
    if _BARY is None:
        u, v = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
        m = (u + v) <= 1
        _BARY = (u[m], v[m])
    verts, faces = meshdata
    loc = (verts @ xmat.reshape(3, 3).T + xpos - org) @ rot
    face_box = np.array(((0.090, 0.115), (-0.045, 0.045), (-0.032, -0.016)))
    near = np.all((loc > face_box[:, 0]) & (loc < face_box[:, 1]), axis=1)
    tri = faces[np.any(near[faces], axis=1)]
    if not len(tri):
        return np.zeros((0, 3))
    a, b, c = loc[tri[:, 0]], loc[tri[:, 1]], loc[tri[:, 2]]
    u, v = _BARY
    return (a[:, None] + u[None, :, None] * (b - a)[:, None]
            + v[None, :, None] * (c - a)[:, None]).reshape(-1, 3)


def _sweep(model, mujoco, rows, groups, layers, trees, shell, best, label, fps):
    data = mujoco.MjData(model)
    clip_min = {}
    inter: dict = {}
    boxes = {k: (pts.min(0) - SEARCH_M, pts.max(0) + SEARCH_M) for k, (pts, _) in layers.items()}
    for i, row in enumerate(rows):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        rot, org = data.xmat[shell].reshape(3, 3), data.xpos[shell]
        for group, items in groups.items():
            if not items:
                continue
            ids = [it[0] for it in items]
            centres = (data.geom_xpos[ids] - org) @ rot
            for key, (pts, owner) in layers.items():
                lo, hi = boxes[key]
                gap = np.linalg.norm(np.maximum(0.0, np.maximum(lo - centres, centres - hi)),
                                     axis=1)
                keep = [k for k, it in enumerate(items) if gap[k] <= it[2]]
                if not keep:
                    continue
                local = np.concatenate([
                    (items[k][1] @ data.geom_xmat[items[k][0]].reshape(3, 3).T
                     + data.geom_xpos[items[k][0]] - org) @ rot for k in keep])
                near = local[np.all((local > lo) & (local < hi), axis=1)]
                if not len(near):
                    continue
                d, idx = trees[key].query(near, k=1, distance_upper_bound=SEARCH_M)
                j = int(np.argmin(d))
                if math.isinf(d[j]):
                    continue
                k = (key, group)
                # Penetration: nearest-neighbour distance cannot see a surface passing
                # through a part, so test target samples for being inside each part, and
                # for the 0.3 mm decals test densely sampled target triangles.
                hit = _inside_parts(near, key)
                if key == "decal" and float(d[j]) < 0.005:
                    tris = [_dense_triangles(items[q][3], data.geom_xmat[items[q][0]],
                                             data.geom_xpos[items[q][0]], org, rot)
                            for q in keep if items[q][3] is not None]
                    if tris:
                        dense = np.concatenate(tris)
                        hit |= _inside_parts(dense, key)
                        for part in _behind_decals(dense):
                            inter.setdefault(("stock_face_penetrated_under", group, part),
                                             []).append(i)
                for part in hit:
                    inter.setdefault((key, group, part), []).append(i)
                if d[j] < clip_min.get(k, math.inf):
                    clip_min[k] = float(d[j])
                if d[j] < best.get(k, {}).get("min_m", math.inf):
                    best[k] = {"min_m": float(d[j]), "clip": label, "frame": i,
                               "time_s": round(i / fps, 4), "part": owner[int(idx[j])]}
    intersections = {f"{k}/{g}/{part}": {"frames": len(fr), "first_s": round(fr[0] / fps, 3),
                                         "last_s": round(fr[-1] / fps, 3)}
                     for (k, g, part), fr in sorted(inter.items())}
    return {f"{k}/{g}": round(v, 5) for (k, g), v in clip_min.items()}, intersections


def cmd_clearance(args) -> int:
    _headless()
    import mujoco
    from concepts import DANCE_CLIPS, JUMP_CLIP, SHELL_RIGID
    from scipy.spatial import cKDTree

    from tasks.jumper.common.constants import HOME
    from tasks.jumper.common.dance.motion import load_source

    qpos_m4, _streams, render_fps, summary = _load_replay(Path(args.replay))
    m4 = _dressed_m4_model()
    rob = _robot_model()
    report = {"method": (
        "outfit geoms surface-sampled at 1.5 mm in the upper_shell_link frame; target "
        "visual geoms (robot group 2 meshes, all pitcher geoms) sampled at vertices + face "
        "centroids + edge midpoints, thinned to one per 1 mm voxel; nearest-neighbour "
        "distance, so each figure overstates the true gap by about 1 mm at most; M4 every "
        f"render frame, policy rollouts and the jump reference at native rate, dance "
        f"references at {REF_HZ:g} Hz; null = clear by more than {SEARCH_M * 1000:.0f} mm"),
        "threshold_m": CLEARANCE_MIN_M, "search_m": SEARCH_M, "clips": []}
    best: dict = {}
    for name, model, pitcher in (("m4", m4, "bar_pitcher"), ("robot", rob, None)):
        shell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  ("robot/" if name == "m4" else "") + outfit.SHELL_BODY)
        data = mujoco.MjData(model)
        if name == "m4":
            data.qpos[:] = qpos_m4[0]
        mujoco.mj_kinematics(model, data)
        layers = _outfit_points(model, mujoco, data, shell)
        trees = {k: cKDTree(v[0]) for k, v in layers.items()}
        groups = _targets(model, mujoco, SHELL_RIGID, pitcher)
        report.setdefault("target_geoms", {})[name] = {g: len(v) for g, v in groups.items()}
        if name == "m4":
            clips = [("m4 replay (every render frame)", qpos_m4, render_fps)]
        else:
            rolls = Path(args.rollouts)
            clips = []
            for k in ("dance", "jump"):
                with np.load(rolls / f"{k}.npz") as z:
                    clips.append((f"{k} policy rollout (outfit applied)", z["qpos"],
                                  float(z["fps"])))
            home = list(HOME)
            adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in home])
            base = np.zeros(model.nq)
            base[3] = 1.0
            base[2] = 0.2

            def rows(jp, base=base, adr=adr):
                out = np.repeat(base[None], len(jp), 0)
                out[:, adr] = jp
                return out

            for rel in DANCE_CLIPS:
                c = load_source(REPO / rel)
                stride = max(1, round(1.0 / (REF_HZ * c.dt)))
                clips.append(((f"dance reference {Path(rel).stem} "
                               f"({1.0 / (c.dt * stride):g} Hz)"),
                              rows(c.joint_pos[::stride]), 1.0 / (c.dt * stride)))
            with np.load(REPO / JUMP_CLIP) as z:
                order = json.loads(str(z["meta"]))["joint_order"]
                q = np.asarray(z["q"], float)[:, [order.index(n) for n in home]]
                dt = float(np.median(np.diff(z["t"])))
            clips.append(("jump reference high_jump_flat (native rate)", rows(q), 1.0 / dt))
        for label, rows_, fps in clips:
            mins, inter = _sweep(model, mujoco, rows_, groups, layers, trees, shell, best,
                                 label, fps)
            report["clips"].append({"clip": label, "frames": len(rows_), "fps": fps,
                                    "min_m": mins, "intersections": inter})
            if inter:
                print(f"[clearance] INTERSECTION in {label}: {inter}", flush=True)
            print(f"[clearance] {label}: {len(rows_)} frames; "
                  + ", ".join(f"{k} {v * 1000:.1f} mm" for k, v in sorted(mins.items())),
                  flush=True)
        if name == "m4":
            # eye-panel box: no proud outfit point in front of the display
            from concepts import FACE_BOX
            pts, owner = layers["proud"]
            inside = np.all([(pts[:, i] > FACE_BOX[i][0]) & (pts[:, i] < FACE_BOX[i][1])
                             for i in range(3)], axis=0)
            report["face_box_intrusions"] = sorted({owner[i] for i in np.flatnonzero(inside)})
    report["minimum"] = {f"{k}/{g}": v for (k, g), v in sorted(best.items())}
    proud = [v["min_m"] for (k, _g), v in best.items() if k == "proud"]
    report["proud_min_m"] = min(proud) if proud else None
    report["proud_intersections"] = any(k.startswith("proud/") for c in report["clips"]
                                        for k in c["intersections"])
    report["decal_intersections"] = {c["clip"]: {k: v for k, v in c["intersections"].items()
                                                 if k.startswith("decal/")}
                                     for c in report["clips"]
                                     if any(k.startswith("decal/") for k in c["intersections"])}
    report["passes"] = (not report["face_box_intrusions"]) and all(
        v >= CLEARANCE_MIN_M for v in proud) and not report["proud_intersections"]
    report["source_qpos_sha256"] = summary["determinism"]["qpos_sha256"]
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for k, v in report["minimum"].items():
        print(f"[clearance] MIN {k}: {v['min_m'] * 1000:.1f} mm ({v['part']}, {v['clip']}, "
              f"t={v['time_s']} s)")
    print(f"[clearance] face_box_intrusions={report['face_box_intrusions']} "
          f"proud_intersections={report['proud_intersections']} "
          f"decal_intersections={report['decal_intersections']} passes={report['passes']}")
    return 0 if report["passes"] else 1


# ── render ────────────────────────────────────────────────────────────────────


def _label(frame, title, caption):
    from make_film import _font
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, WIDTH, 58), fill=(18, 14, 12))
    d.text((18, 8), title, font=_font(28), fill=(255, 236, 200))
    d.text((WIDTH - 18, 18), caption, font=_font(18), fill=(220, 220, 220), anchor="ra")
    d.rectangle((0, HEIGHT - 30, WIDTH, HEIGHT), fill=(18, 14, 12))
    d.text((WIDTH // 2, HEIGHT - 15), FOOTER, font=_font(15), fill=(200, 200, 200), anchor="mm")
    return img


def _setup_vis(model):
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), HEIGHT)
    model.vis.quality.shadowsize = 4096
    model.vis.headlight.ambient[:] = (0.35, 0.33, 0.30)
    model.vis.headlight.diffuse[:] = (0.55, 0.52, 0.48)


def _landing_index(model, mujoco, qpos, fps):
    """First frame after the apex where the base is back near its start height."""
    z = qpos[:, 2]
    apex = int(np.argmax(z))
    start = float(np.median(z[: max(3, int(0.15 * fps))]))
    after = np.flatnonzero(z[apex:] <= start + 0.01)
    return apex + int(after[0]) if len(after) else len(z) - 1, apex


def cmd_render(args) -> int:
    _headless()
    import mujoco
    from make_film import Camera

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qpos, _streams, render_fps, summary = _load_replay(Path(args.replay))
    files = {}
    m4 = _dressed_m4_model()
    _setup_vis(m4)
    data = mujoco.MjData(m4)
    shell = mujoco.mj_name2id(m4, mujoco.mjtObj.mjOBJ_BODY, "robot/" + outfit.SHELL_BODY)

    def centre(d, sh):
        return d.xpos[sh] + d.xmat[sh].reshape(3, 3) @ np.array((0.02, 0.0, 0.02))

    title = f"Jumper Barista outfit: {outfit.NAME}"
    renderer = mujoco.Renderer(m4, height=HEIGHT, width=WIDTH)
    try:
        for key, t, dist, az, el, cap in (
                ("front_threequarter", 4.0, 0.46, 150.0, -14.0,
                 "front three-quarter (M4 grasp, t=4 s)"),
                ("side", 4.0, 0.50, 268.0, -6.0, "left side (M4 grasp, t=4 s)"),
                ("rear_threequarter", 4.0, 0.50, 330.0, -20.0,
                 "rear three-quarter, bar towel (M4 grasp, t=4 s)"),
                ("pour", POUR_TIME_S, 0.55, 160.0, -18.0,
                 "pour pose with pitcher (M4, t=30 s)")):
            i = round(t * render_fps)
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(m4, data)
            cam = Camera(mujoco, centre(data, shell), dist, az, el)
            renderer.update_scene(data, camera=cam.cam)
            path = out / f"outfit_{key}.png"
            _label(renderer.render(), title, cap).save(path)
            files[key] = str(path)
    finally:
        renderer.close()

    rob = _robot_model(stage=True)
    _setup_vis(rob)
    rob.vis.global_.offwidth, rob.vis.global_.offheight = WIDTH, HEIGHT
    rdata = mujoco.MjData(rob)
    rshell = mujoco.mj_name2id(rob, mujoco.mjtObj.mjOBJ_BODY, outfit.SHELL_BODY)
    picks = {}
    with np.load(Path(args.rollouts) / "dance.npz") as z:
        dq, dfps = z["qpos"], float(z["fps"])
    with np.load(Path(args.rollouts) / "jump.npz") as z:
        jq, jfps = z["qpos"], float(z["fps"])
    di = min(len(dq) - 1, round(args.dance_time * dfps))
    li, apex = _landing_index(rob, mujoco, jq, jfps)
    picks["dance"] = (dq[di], f"brazilian dance policy, t={di / dfps:.2f} s", 140.0)
    picks["jump_landing"] = (jq[li], (f"jump policy landing, t={li / jfps:.2f} s "
                             f"(apex {apex / jfps:.2f} s)"), 150.0)
    renderer = mujoco.Renderer(rob, height=HEIGHT, width=WIDTH)
    try:
        for key, (row, cap, az) in picks.items():
            rdata.qpos[:] = row
            mujoco.mj_forward(rob, rdata)
            cam = Camera(mujoco, centre(rdata, rshell), 0.55, az, -12.0)
            renderer.update_scene(rdata, camera=cam.cam)
            path = out / f"outfit_{key}.png"
            _label(renderer.render(), title, cap).save(path)
            files[key] = str(path)
    finally:
        renderer.close()
    manifest = {"outfit": outfit.NAME, "simulation_only": True,
                "source_replay_qpos_sha256": summary["determinism"]["qpos_sha256"],
                "dance_frame": di, "jump_landing_frame": li, "jump_apex_frame": apex,
                "files": files, "png_sha256": {k: _sha(Path(v)) for k, v in files.items()}}
    (out / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for k, v in files.items():
        print(f"[render] {k}: {v}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("m4")
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("compare")
    s.add_argument("--stock", type=Path, required=True)
    s.add_argument("--dressed", type=Path, required=True)
    s.add_argument("--accepted-qpos", default=None)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("rollout")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--jump-seconds", type=float, default=3.5)
    s = sub.add_parser("clearance")
    s.add_argument("--replay", type=Path, required=True)
    s.add_argument("--rollouts", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("render")
    s.add_argument("--replay", type=Path, required=True)
    s.add_argument("--rollouts", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--dance-time", type=float, default=12.0)
    args = p.parse_args()
    return {"m4": cmd_m4, "compare": cmd_compare, "rollout": cmd_rollout,
            "clearance": cmd_clearance, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
