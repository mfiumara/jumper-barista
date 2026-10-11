"""The Jumper Barista cafe scene (Act 2, milestone A1).

Fast, simulation-free checks of what the cafe promises the later lanes: the brew
station is the accepted one moved rigidly, the pour-start pose is the accepted M4
start moved by the same transform, the route and named sites exist, the added
cafe geometry cannot collide unless it is declared physical, and the build is
deterministic. The physical evidence (settle, footprints, route clearance, the
M4 transfer replay) comes from the scripts in ``barista/cafe/``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import scenes
from scenes import cafe_bar
from scenes.package import read_package

REPO = Path(__file__).resolve().parents[1]
CAFE = REPO / "barista" / "cafe"


def _load(name: str):
    """Load a barista/cafe script by path (barista/ is not an installed package)."""
    spec = importlib.util.spec_from_file_location(name, CAFE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # the scripts import cafe_common by name
    spec.loader.exec_module(module)
    return module


cc = _load("cafe_common")

COFFEE = REPO / "scenes" / "coffee_bar.scene"
M4 = REPO / "barista" / "trajectory" / "m4_spiral_pour.json"
#: Static geometry the cafe deliberately makes physical. Everything else it adds
#: must be unable to collide.
PHYSICAL_ADDITIONS = {"bar_top_ext_left", "bar_top_ext_right", "bar_top_ext_back",
                      "cafe_kettle_body", "cafe_cup_stack", "v60_coffee_bed"}
BREWER_TOKENS = ("server_", "v60_")


@pytest.fixture(scope="module")
def dims() -> dict:
    return cafe_bar.dimensions()


@pytest.fixture(scope="module")
def models():
    import mujoco

    coffee = read_package(COFFEE)
    cafe = read_package(cafe_bar.package_dir())
    out = {}
    for key, package in (("coffee", coffee), ("cafe", cafe)):
        model = package.spec("scene.xml").compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        out[key] = (model, data)
    return out


def _geoms(model) -> dict[str, int]:
    return {cc.geom_name(model, g): g for g in range(model.ngeom)}


def test_cafe_bar_is_registered_and_loads() -> None:
    spec = scenes.get("cafe_bar")
    assert spec.use == "watching"
    scene = scenes.load("cafe_bar")
    assert scene.lights and scene.headlight is not None
    assert scene.terrain_type == "plane", "the bar top must replace the task's ground"


def test_dimensions_schema_and_route(dims) -> None:
    assert dims["schema"] == "jumper-barista-cafe/1"
    assert dims["coordinate_frame"]["bar_top_z_m"] == 0.0
    d_walk = dims["brew_station"]["d_walk_m"]
    assert 0.6 <= d_walk <= 1.0
    route = dims["route"]
    assert route["start_slot"]["yaw_deg"] == -90.0, "start slot faces -Y, the customer"
    assert route["turn"]["delta_deg"] == 90.0 and route["turn"]["to_yaw_deg"] == 0.0
    walk = route["walk"]
    assert walk[0]["x_m"] == route["start_slot"]["x_m"] == 0.0
    xs = [w["x_m"] for w in walk]
    assert xs == sorted(xs), "walk waypoints run along +X"
    assert walk[-1]["name"] == "pour_start"
    pour = route["pour_start_pose"]
    assert (walk[-1]["x_m"], walk[-1]["y_m"]) == (pour["x_m"], pour["y_m"])
    assert math.isclose(pour["x_m"], d_walk)
    assert "back_counter_slot" not in route and "h_step" not in json.dumps(dims)
    camera = dims["customer_camera"]
    assert math.isclose(camera["position_m"][2], camera["eye_height_above_bar_m"])
    assert camera["position_m"][1] < dims["bar"]["top_y_range_m"][0], "customer is on -Y"
    assert dims["door"]["door_chime"]["site"] == "door_chime"


def test_pour_start_is_the_accepted_m4_start_moved_rigidly(dims) -> None:
    config = json.loads(M4.read_text("utf-8"))
    coffee = json.loads((COFFEE / "scene-package.json").read_text("utf-8"))
    station = dims["brew_station"]
    authored = np.asarray(station["authored_offset_m"])
    compiled = np.asarray(station["compiled_offset_m"])
    # The accepted package is attached with its standing-slot site at the origin.
    assert coffee["spawn"]["site"] == "jumper_standing_slot"
    assert np.allclose(compiled - authored, [0.0, 0.0, 0.002])
    pour = dims["route"]["pour_start_pose"]
    start = np.asarray(config["base_start"]["position_m"])
    assert np.allclose([pour["x_m"], pour["y_m"], pour["base_z_m"]], start + compiled, atol=1e-9)
    assert math.isclose(math.radians(pour["yaw_deg"]), config["base_start"]["yaw_rad"])


def test_the_m4_transfer_moves_only_world_frame_positions(dims) -> None:
    transferred_trajectory = _load("run_m4_transfer").transferred_trajectory

    original = json.loads(M4.read_text("utf-8"))
    moved = transferred_trajectory(dims["brew_station"]["compiled_offset_m"])
    offset = np.asarray(dims["brew_station"]["compiled_offset_m"])
    for path in (("base_start", "position_m"), ("stream", "bed_centre_m")):
        a, b = original[path[0]][path[1]], moved[path[0]][path[1]]
        assert np.allclose(np.asarray(b) - a, offset)
    assert np.allclose(np.asarray(moved["set_down_target_m"]) - original["set_down_target_m"],
                       offset)
    for key in ("keyframes", "support_pour_rad", "support_base_rad", "support_pour_profile_s",
                "thresholds", "arm_joint_order", "physics_dt_s", "control_dt_s", "seed"):
        assert moved[key] == original[key]
    assert moved["base_start"]["yaw_rad"] == original["base_start"]["yaw_rad"]


def test_brew_station_is_a_rigid_copy_of_the_accepted_one(dims, models) -> None:
    import mujoco

    (cm, cd), (fm, fd) = models["coffee"], models["cafe"]
    authored = np.asarray(dims["brew_station"]["authored_offset_m"])
    # The station's XML is copied verbatim into a translated <frame>, so the only
    # difference left is floating-point summation of the offset.
    tol = 1e-12
    cafe_geoms = _geoms(fm)
    for body in cafe_bar.STATION_BODIES:
        cb = mujoco.mj_name2id(cm, mujoco.mjtObj.mjOBJ_BODY, body)
        fb = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, body)
        assert np.allclose(fd.xpos[fb] - cd.xpos[cb], authored, rtol=0, atol=tol)
        assert np.allclose(fd.xmat[fb], cd.xmat[cb])
        assert fm.body_mass[fb] == cm.body_mass[cb]
        assert np.allclose(fm.body_inertia[fb], cm.body_inertia[cb], rtol=0, atol=tol)
        assert np.allclose(fm.body_ipos[fb], cm.body_ipos[cb], rtol=0, atol=tol)
        for g in range(cm.ngeom):
            if cm.geom_bodyid[g] != cb:
                continue
            name = cc.geom_name(cm, g)
            f = cafe_geoms[name]
            for field in ("geom_type", "geom_size", "geom_contype", "geom_conaffinity",
                          "geom_condim", "geom_friction", "geom_solref", "geom_solimp",
                          "geom_priority", "geom_pos", "geom_quat"):
                assert np.allclose(getattr(fm, field)[f], getattr(cm, field)[g],
                                   rtol=0, atol=tol), (name, field)
            assert np.allclose(fd.geom_xpos[f] - cd.geom_xpos[g], authored,
                               rtol=0, atol=tol), name
    for name in cafe_bar.STATION_GEOMS:  # the accepted counter slab: same collision, moved
        g, f = _geoms(cm)[name], cafe_geoms[name]
        for field in ("geom_type", "geom_size", "geom_contype", "geom_conaffinity", "geom_condim",
                      "geom_friction", "geom_solref", "geom_solimp", "geom_priority",
                      "geom_margin", "geom_gap"):
            assert np.allclose(getattr(fm, field)[f], getattr(cm, field)[g], rtol=0, atol=tol)
        assert np.allclose(fd.geom_xpos[f] - cd.geom_xpos[g], authored, rtol=0, atol=tol)
        assert fm.geom_rgba[f][3] == 0.0, "the wooden visual top replaces its look"
    pitcher = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "pitcher")
    assert math.isclose(float(fm.body_subtreemass[pitcher]), 0.3452, abs_tol=1e-9)
    for camera in cafe_bar.STATION_CAMERAS:
        cc_ = mujoco.mj_name2id(cm, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        fc = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        assert np.allclose(fm.cam_pos[fc] - cm.cam_pos[cc_], authored, rtol=0, atol=tol)


def test_only_declared_cafe_geometry_is_physical(models) -> None:
    (cm, _), (fm, _) = models["coffee"], models["cafe"]
    coffee_names = set(_geoms(cm))
    added = {name: g for name, g in _geoms(fm).items() if name not in coffee_names}
    physical = {n for n, g in added.items() if fm.geom_contype[g] or fm.geom_conaffinity[g]}
    assert physical == PHYSICAL_ADDITIONS
    # The M4 verifier treats any geom whose name holds these tokens as brewer
    # geometry and requires it to collide; a visual cue must never be named so.
    brewer_added = {n for n in added if any(t in n for t in BREWER_TOKENS)}
    assert brewer_added == {"v60_coffee_bed"}
    # The station's own XML is the accepted text, element for element.
    import xml.etree.ElementTree as ET

    accepted = ET.fromstring((COFFEE / "scene.xml").read_text("utf-8")).find("worldbody")
    cafe = ET.fromstring((cafe_bar.package_dir() / "scene.xml").read_text("utf-8"))
    frame = cafe.find("worldbody/frame[@name='brew_station']")
    for element in accepted:
        if element.get("name") in {*cafe_bar.STATION_BODIES, *cafe_bar.STATION_CAMERAS}:
            copy = frame.find(f"{element.tag}[@name='{element.get('name')}']")
            assert ET.tostring(copy).split() == ET.tostring(element).split()
    for name in cafe_bar.DROPPED_GEOMS:
        assert name not in _geoms(fm)


def test_the_bar_top_is_one_flat_surface(dims, models) -> None:
    fm, fd = models["cafe"]
    geoms = _geoms(fm)
    x0, x1 = dims["bar"]["top_x_range_m"]
    y0, y1 = dims["bar"]["top_y_range_m"]
    area = 0.0
    for name in ("counter_surface", *cafe_bar.EXTENSIONS):
        g = geoms[name]
        assert math.isclose(fd.geom_xpos[g][2] + fm.geom_size[g][2], 0.0, abs_tol=1e-12), name
        area += 4 * fm.geom_size[g][0] * fm.geom_size[g][1]
    assert math.isclose(area, (x1 - x0) * (y1 - y0), rel_tol=1e-9), "gaps or overlaps"


def test_the_coffee_bed_stays_below_the_stream_plane_and_inside_the_ribs(models) -> None:
    import mujoco

    fm, fd = models["cafe"]
    bed = _geoms(fm)["v60_coffee_bed"]
    target = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_SITE, "coffee_bed_target")
    top = fd.geom_xpos[bed][2] + fm.geom_size[bed][1]
    assert fd.site_xpos[target][2] - top >= 0.001, "stream rays end on the bed plane"
    radius, centre_z = fm.geom_size[bed][0], fd.geom_xpos[bed][2] - fd.xpos[fm.geom_bodyid[bed]][2]
    rib_inner = 0.012 + (0.037 - 0.012) / 0.080 * (centre_z - fm.geom_size[bed][1] - 0.005) - 0.002
    assert radius <= rib_inner


def test_named_sites_and_cameras(dims, models) -> None:
    import mujoco

    fm, fd = models["cafe"]
    chime = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_SITE, "door_chime")
    assert chime >= 0
    assert np.allclose(fd.site_xpos[chime], dims["door"]["door_chime"]["position_m"])
    for site in ("start_slot", "pour_start", "dance_area", "coffee_bed_target"):
        assert mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_SITE, site) >= 0, site
    for key in ("customer_camera", "wide_camera"):
        cam = dims[key]
        index = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_CAMERA, cam["camera"])
        assert index >= 0
        assert np.allclose(fm.cam_pos[index], cam["position_m"])
        assert math.isclose(fm.cam_fovy[index], cam["fovy_deg"])
        forward = -fd.cam_xmat[index].reshape(3, 3)[:, 2]
        want = np.asarray(cam["look_at_m"]) - cam["position_m"]
        assert np.allclose(forward, want / np.linalg.norm(want), atol=1e-6)
    door_geoms = [n for n in _geoms(fm) if n.startswith("cafe_door")]
    assert door_geoms and all(fm.geom_contype[_geoms(fm)[n]] == 0 for n in door_geoms)


def test_dance_area_is_clear_of_the_edges_and_everything_on_the_bar(dims, models) -> None:
    fm, fd = models["cafe"]
    dance = dims["dance_area"]
    centre, radius = np.asarray(dance["centre_m"]), dance["radius_m"]
    x0, x1 = dims["bar"]["top_x_range_m"]
    y0, y1 = dims["bar"]["top_y_range_m"]
    assert min(centre[0] - x0, x1 - centre[0], centre[1] - y0, y1 - centre[1]) - radius >= 0.03
    for g in range(fm.ngeom):
        name = cc.geom_name(fm, g)
        if not (fm.geom_contype[g] or fm.geom_conaffinity[g]) or name == "counter_surface" \
                or name.startswith("bar_top_ext_"):
            continue
        poly = cc.hull(cc.geom_points(fm, fd, g))
        angles = np.linspace(0, 2 * math.pi, 180, endpoint=False)
        disc = centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        assert cc.polygon_distance(disc, poly) >= 0.03, cc.geom_name(fm, g)


def test_the_generated_package_is_deterministic_and_hash_checked(tmp_path) -> None:
    first = cafe_bar.write_package(tmp_path / "a")
    second = cafe_bar.write_package(tmp_path / "b")
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    read_package(first)  # SHA-256 and size of every declared file
    manifest = json.loads((first / "scene-package.json").read_text("utf-8"))
    assert manifest["world"]["ground"]["geom"] == "counter_surface"
    assert manifest["rules"]["simulationOnly"] is True


def test_polygon_distance_helper() -> None:
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert cc.polygon_distance(square, square + [2.0, 0.0]) == pytest.approx(1.0)
    assert cc.polygon_distance(square, square + [0.5, 0.5]) == 0.0
    assert cc.polygon_distance(square, square + [2.0, 2.0]) == pytest.approx(math.sqrt(2))


def test_grasp_approach_check_classifies_support_and_measures_distance() -> None:
    """The exemption sweep lets only the stance contacts touch the bar, and its geom
    distance agrees with the true gap between two separated geoms."""
    import mujoco

    route_check = _load("route_check")
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="robot/LM_foot_tip_link_meshcol" type="box" size=".01 .01 .01" pos="0 0 .01"/>
      <geom name="robot/RF_palm_pad_b_link_meshcol" type="box" size=".01 .01 .01" pos=".1 0 .01"/>
      <geom name="robot/LF_finger_tip_link_meshcol" type="box" size=".01 .01 .01" pos="0 .1 .01"/>
      <geom name="terrain" type="box" size=".5 .5 .02" pos="0 0 -.05"/>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    support = {n for g in range(model.ngeom)
               if route_check._support_geom(model, g) for n in [cc.geom_name(model, g)]}
    assert support == {"robot/LM_foot_tip_link_meshcol", "robot/RF_palm_pad_b_link_meshcol"}
    value, consistent = route_check._distance(model, data, 0, 3)
    assert consistent and value == pytest.approx(0.03, abs=1e-6)
    assert "pitcher" in route_check.GRASP_APPROACH_EXEMPTION
