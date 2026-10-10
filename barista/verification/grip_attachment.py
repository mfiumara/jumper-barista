#!/usr/bin/env python3
"""Static attachment check for the pitcher's 3D-printed rear grip add-on.

Simulation only. In MuJoCo every geom of the free ``pitcher`` body is welded to
it, so a floating add-on would still "carry" the pitcher in the replay. This
check makes the claimed physical load path explicit instead, at the scene's
initial pose:

* a *seat* is a collidable (structural) pitcher geom that is not part of the
  add-on: the vessel, and the handle once it is collidable. Visual-only
  geometry (``contype 0`` and ``conaffinity 0``) can never be a seat;
* an add-on geom (``top_grip_*``) is *seated* when its signed distance to a
  seat lies within [-0.5 mm, +1.0 mm];
* every add-on geom must be seated, or lie within 1.0 mm of an add-on geom
  that is (transitively) seated -- the printed pieces form one part, so the
  load path is the chain of touching pieces ending on a seat. The per-geom
  distance to the nearest pitcher geometry is reported for every piece;
* at least one add-on geom must be seated;
* no add-on geom may penetrate any non-add-on pitcher geometry (seats or the
  visual spout) by more than 0.5 mm.

``mujoco.mj_geomDistance`` is independent of contact filtering. Its convex
solver can return spurious low values (0 or a bogus negative) for some
``distmax`` values; a reading below ``distmax`` is therefore repeated at
several ``distmax`` values and the largest reading kept. A genuine contact or
penetration reads the same at every ``distmax``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "scenes" / "coffee_bar.scene" / "scene.xml"
ADDON_PREFIX = "top_grip_"
NON_GEOMETRY = ("nominal_water_mass", "water_surface")
MAX_GAP_M = 0.0010
MAX_PENETRATION_M = 0.0005
DISTMAX_M = (0.003, 0.01, 0.05, 0.5)


def check(scene: Path = SCENE) -> dict:
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pitcher = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pitcher")
    names = {
        index: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) == pitcher
    }
    addon = sorted(index for index, name in names.items() if name.startswith(ADDON_PREFIX))
    body = sorted(
        index for index, name in names.items()
        if not name.startswith(ADDON_PREFIX) and name not in NON_GEOMETRY
    )
    seats = [
        index for index in body
        if int(model.geom_contype[index]) or int(model.geom_conaffinity[index])
    ]
    fromto = np.zeros(6, dtype=np.float64)

    def distance(a: int, b: int) -> float:
        first = float(mujoco.mj_geomDistance(model, data, a, b, DISTMAX_M[-1], fromto))
        if first >= DISTMAX_M[-1]:
            return first
        return max(
            float(mujoco.mj_geomDistance(model, data, a, b, distmax, fromto))
            for distmax in DISTMAX_M
        )

    pieces = {}
    for index in addon:
        to_body = {names[other]: distance(index, other) for other in body}
        nearest = min(to_body, key=lambda key: to_body[key])
        to_seat = {names[other]: to_body[names[other]] for other in seats}
        seat = min(to_seat, key=lambda key: to_seat[key]) if to_seat else None
        seat_distance = to_seat[seat] if seat else float("inf")
        pieces[names[index]] = {
            "nearest_pitcher_geom": nearest,
            "nearest_pitcher_distance_m": to_body[nearest],
            "nearest_seat_geom": seat,
            "nearest_seat_distance_m": seat_distance,
            "max_penetration_into_pitcher_m": max(0.0, -min(to_body.values())),
            "seated": -MAX_PENETRATION_M <= seat_distance <= MAX_GAP_M,
        }

    attached = {name for name, piece in pieces.items() if piece["seated"]}
    changed = True
    while changed:
        changed = False
        for index in addon:
            name = names[index]
            if name in attached:
                continue
            for other in addon:
                if names[other] in attached and distance(index, other) <= MAX_GAP_M:
                    attached.add(name)
                    pieces[name]["attached_via"] = names[other]
                    changed = True
                    break
    for name, piece in pieces.items():
        piece["attached"] = name in attached
        if piece["seated"]:
            piece["attached_via"] = piece["nearest_seat_geom"]
        piece["penetration_ok"] = piece["max_penetration_into_pitcher_m"] <= MAX_PENETRATION_M

    failures = []
    if not addon:
        failures.append(f"no {ADDON_PREFIX}* geoms on the pitcher body")
    if not attached:
        failures.append(
            "no add-on geom is seated on collidable pitcher geometry "
            f"({', '.join(names[index] for index in seats)})"
        )
    for name, piece in pieces.items():
        if not piece["attached"]:
            failures.append(
                f"{name} floats: {piece['nearest_pitcher_distance_m'] * 1000:.2f} mm from "
                f"{piece['nearest_pitcher_geom']} and not connected to a seated piece"
            )
        if not piece["penetration_ok"]:
            failures.append(
                f"{name} penetrates pitcher geometry by "
                f"{piece['max_penetration_into_pitcher_m'] * 1000:.2f} mm"
            )
    return {
        "simulation_only": True,
        "scene": str(scene),
        "max_gap_m": MAX_GAP_M,
        "max_penetration_m": MAX_PENETRATION_M,
        "seat_geoms": [names[index] for index in seats],
        "pieces": pieces,
        "compiled_pitcher_mass_kg": float(model.body_subtreemass[pitcher]),
        "failures": failures,
        "passes": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = check(args.scene.resolve())
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", "utf-8")
    print(f"[attach] scene {result['scene']}; seats: {', '.join(result['seat_geoms'])}")
    for name, piece in result["pieces"].items():
        print(
            f"[attach] {name}: nearest {piece['nearest_pitcher_geom']} "
            f"{piece['nearest_pitcher_distance_m'] * 1000:+.2f} mm; "
            f"attached={piece['attached']} via {piece.get('attached_via', '-')}"
        )
    for failure in result["failures"]:
        print(f"[attach] FAIL {failure}")
    print(
        f"[attach] compiled pitcher payload {result['compiled_pitcher_mass_kg']:.4f} kg; "
        f"passes={result['passes']}"
    )
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
