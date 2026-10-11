"""The barista outfit (A2b, Concept C "Dapper Tricolore") is display-only.

Simulation only: these check the compiled MuJoCo model, not the real robot.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tasks.jumper.common.constants import get_spec

REPO = Path(__file__).resolve().parents[1]


def _load_outfit():
    # outfit.py puts its own directory on the import path for its sibling concepts.py
    spec = importlib.util.spec_from_file_location("outfit", REPO / "barista/outfit/outfit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["outfit"] = module
    spec.loader.exec_module(module)
    return module


outfit = _load_outfit()

PHYSICS_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_subtreemass",
                  "body_pos", "body_quat", "jnt_range", "dof_damping", "dof_armature",
                  "geom_contype", "geom_conaffinity")


@pytest.fixture(scope="module")
def models():
    stock = get_spec().compile()
    spec = get_spec()
    outfit.add_to_spec(spec)
    return stock, spec.compile()


def test_is_the_picked_concept():
    assert outfit.KEY == "C" and outfit.NAME == "Dapper Tricolore"
    parts = {g.part for g in outfit.geoms()}
    assert {"hat_brim", "hat_crown", "hat_crown_top", "band_verde", "band_bianco",
            "band_rosso", "tache_l", "tache_l_r", "towel_top", "towel_back",
            "towel_stripe"} == parts
    assert outfit.RECOLOUR == {"base_link_visual": (0.96, 0.96, 0.94, 1.0)}


def test_masses_and_inertias_unchanged(models):
    stock, dressed = models
    assert stock.nbody == dressed.nbody and stock.nq == dressed.nq and stock.nv == dressed.nv
    for field in PHYSICS_FIELDS[:-2]:
        assert np.array_equal(getattr(stock, field), getattr(dressed, field)), field


def test_outfit_geoms_are_display_only(models):
    stock, dressed = models
    ids = outfit.geom_ids(dressed)
    assert len(ids) == len(outfit.PARTS) == dressed.ngeom - stock.ngeom
    shell = mujoco.mj_name2id(dressed, mujoco.mjtObj.mjOBJ_BODY, outfit.SHELL_BODY)
    for g in ids:
        assert dressed.geom_contype[g] == 0 and dressed.geom_conaffinity[g] == 0
        assert dressed.geom_group[g] == outfit.VISUAL_GROUP
        assert dressed.geom_bodyid[g] == shell
    # every pre-existing geom keeps its collision bits
    names = [mujoco.mj_id2name(stock, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(stock.ngeom)]
    for g, name in enumerate(names):
        d = mujoco.mj_name2id(dressed, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert dressed.geom_contype[d] == stock.geom_contype[g]
        assert dressed.geom_conaffinity[d] == stock.geom_conaffinity[g]


def test_free_fall_and_contact_are_bit_identical(models):
    trace = []
    for model in models:
        data = mujoco.MjData(model)
        data.qpos[2] += 0.05
        data.ctrl[:] = 0.0
        out = []
        for _ in range(400):
            mujoco.mj_step(model, data)
            out.append(np.concatenate((data.qpos, data.qvel, data.qfrc_constraint)).copy())
        trace.append(np.stack(out))
    assert np.array_equal(trace[0], trace[1])


def test_head_front_stays_free_and_decal_is_flush():
    from concepts import FACE_BOX, _face_x

    for g in outfit.geoms():
        if g.decal:
            # thickness 2 * half-t, centre at proud - half-t: at most DECAL_PROUD_M proud
            assert g.size[0] * 2 <= outfit.DECAL_PROUD_M + 1e-12
            assert g.pos[0] + g.size[0] <= _face_x(g.pos[1]) + outfit.DECAL_PROUD_M + 1e-9
            continue
        # proud parts: centre and bounding radius stay behind the eye-panel box
        reach = float(np.linalg.norm(g.size))
        assert g.pos[0] + reach < FACE_BOX[0][0], g.name


def test_apply_wraps_spec_fn_once():
    import tasks

    cfg = tasks.load_env_cfg("jumper.barista", play=True)
    outfit.apply(cfg)
    first = cfg.scene.entities["robot"].spec_fn
    outfit.apply(cfg)
    assert cfg.scene.entities["robot"].spec_fn is first
    model = first().compile()
    assert len(outfit.geom_ids(model)) == len(outfit.PARTS)


def test_visual_overrides_round_trip(tmp_path):
    path = outfit.write_visual_overrides(tmp_path / "overrides.json")
    data = json.loads(path.read_text())
    assert data["display_only"] and data["collision"] == {"contype": 0, "conaffinity": 0}
    assert len(data["geoms"]) == len(outfit.PARTS)
    committed = REPO / "barista/outfit/dapper_tricolore.visual_overrides.json"
    assert committed.read_text() == path.read_text(), "regenerate the committed overrides"
