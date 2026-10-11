#!/usr/bin/env python3
"""Arrival sweeps of the closed-loop pour (supervisor decisions on A4).

Restarts the pour from the chain's real arrival state (``arrival_state.npz``: qpos, qvel and
servo heat at the end of the approach), rigidly moved about its own base, and runs the
unmodified run_m4 gates on each case. One subprocess per case.

Two sweeps:

* the +-2 mm / +-2 deg perturbation sweep (default ``--cases``), relative to the arrival;
* ``--yaw-grid START STOP STEP``: absolute arrival yaw (deg) relative to the M4
  ``pour_start`` yaw, x/y left as walked (dx = dy = 0). The rotation applied to the arrival
  state is ``target - measured arrival yaw error``. Used to find the passing yaw band.

    python barista/chain/pour_sweep.py --run RUN --reference REF.npz --jobs 4
    python barista/chain/pour_sweep.py --run RUN --reference REF.npz --yaw-grid -1 3 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CASES = {
    "dx+2mm": (0.002, 0.0, 0.0), "dx-2mm": (-0.002, 0.0, 0.0),
    "dy+2mm": (0.0, 0.002, 0.0), "dy-2mm": (0.0, -0.002, 0.0),
    "yaw+2deg": (0.0, 0.0, 2.0), "yaw-2deg": (0.0, 0.0, -2.0),
}
#: Gates the supervisor asked to see per point, besides the full gate list.
NAMED_GATES = ("motion.no_brewer_or_carried_forbidden_contact", "stream.inward_excursion",
               "motion.carried_grasp_translation")


def yaw_grid(start: float, stop: float, step: float) -> list[float]:
    count = round((stop - start) / step)
    if count < 0 or abs(start + count * step - stop) > 1e-9:
        raise ValueError(f"grid {start}..{stop} is not a whole number of {step} steps")
    return [round(start + i * step, 6) for i in range(count + 1)]


def arrival_yaw_error_deg(arrival_qpos, goal_yaw_rad: float) -> float:
    """Arrival base yaw minus the M4 pour_start yaw, wrapped, in degrees."""
    w, x, y, z = (float(v) for v in arrival_qpos[3:7])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    err = (yaw - goal_yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(err)


def yaw_cases(grid: list[float], measured_err_deg: float) -> dict[str, tuple[float, float, float]]:
    return {f"yaw{v:+.1f}deg_abs": (0.0, 0.0, v - measured_err_deg) for v in grid}


def bands(points: list[tuple[float, bool]]) -> list[dict]:
    """Contiguous passing runs of a sorted (value, passes) list, with their width."""
    out, cur = [], []
    for value, ok in sorted(points):
        if ok:
            cur.append(value)
            continue
        if cur:
            out.append(cur)
        cur = []
    if cur:
        out.append(cur)
    return [{"from_deg": b[0], "to_deg": b[-1], "width_deg": b[-1] - b[0],
             "centre_deg": 0.5 * (b[0] + b[-1]), "points": len(b)} for b in out]


def _goal_yaw_rad() -> float:
    import importlib.util

    spec = importlib.util.spec_from_file_location("barista_chain_common",
                                                  Path(__file__).with_name("common.py"))
    common = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("barista_chain_common", common)
    spec.loader.exec_module(common)
    cfg = common.cafe_m4_trajectory(1.0 / common.PHYSICS_HZ, common.M4_DECIMATION_1K)
    return float(cfg["base_start"]["yaw_rad"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="run_chain output dir")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--yaw-grid", nargs=3, type=float, default=None,
                        metavar=("START", "STOP", "STEP"),
                        help="absolute arrival yaw vs pour_start, deg (dx = dy = 0)")
    parser.add_argument("--name", default=None, help="sweep subdir (default sweep / yaw_sweep)")
    args = parser.parse_args()
    import numpy as np

    run = args.run.resolve()
    arrival = np.load(run / "arrival_state.npz")
    measured = arrival_yaw_error_deg(arrival["qpos"], _goal_yaw_rad())
    if args.yaw_grid is not None:
        cases = yaw_cases(yaw_grid(*args.yaw_grid), measured)
        name = args.name or "yaw_sweep"
    else:
        cases = {k: CASES[k] for k in args.cases}
        name = args.name or "sweep"
    pour = Path(__file__).with_name("pour.py")
    sweep_dir = run / name
    sweep_dir.mkdir(exist_ok=True)

    def one(case: str) -> dict:
        dx, dy, dyaw = cases[case]
        out = sweep_dir / case
        cmd = [sys.executable, str(pour), "--mode", "retarget", "--reference", str(args.reference),
               "--start-state", str(run / "arrival_state.npz"), "--dx", repr(dx), "--dy", repr(dy),
               "--dyaw-deg", repr(dyaw), "--out", str(out)]
        with open(sweep_dir / f"{case}.log", "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        result = {"case": case, "exit_code": code, "dx_m": dx, "dy_m": dy, "dyaw_deg": dyaw,
                  "arrival_yaw_vs_pour_start_deg": measured + dyaw}
        record = out / "pour_result.json"
        if record.exists():
            data = json.loads(record.read_text())
            summary = json.loads((out / "summary.json").read_text())
            motion, stream = summary["motion"], summary["stream"]
            gates = {f"motion.{k}": bool(v) for k, v in motion["checks"].items()}
            gates.update({f"stream.{k}": bool(v) for k, v in stream["checks"].items()})
            gates["torque"] = bool(summary["torque"]["passes"])
            gates["payload"] = bool(summary["payload"]["passes"])
            display = [p for p in motion.get("contact_pairs_seen", [])
                       if any("display_module" in str(g) for g in p)]
            result.update(
                passes=data["passes"], failing_gates=data["failing_gates"], gates=gates,
                named_gates={g: gates.get(g) for g in NAMED_GATES},
                display_module_contact_pairs=display,
                turns=stream["turn_count"],
                inward_excursion_m=stream["maximum_inward_excursion_m"],
                carried_grasp_translation_m=motion["maximum_carried_grasp_translation_m"],
                set_down_error_m=motion["maximum_final_position_error_m"],
                spout_height_m=[motion["spout_height_min_m"], motion["spout_height_max_m"]],
                landing_success_rate=stream["landing_success_rate"],
                torque_passes=summary["torque"]["passes"])
            (out / "motion_substeps.csv").unlink(missing_ok=True)  # 66 MB each; summary kept
        else:
            result.update(passes=False, failing_gates=["no result (crash)"])
        print(json.dumps({k: v for k, v in result.items() if k != "gates"}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(one, list(cases)))
    doc = {"simulation_only": True, "arrival_state": str(run / "arrival_state.npz"),
           "arrival_yaw_vs_pour_start_deg": measured,
           "note": "thermal state carried from the chain; the pour restarts at its own t=0",
           "pass_count": sum(r["passes"] for r in results), "case_count": len(results),
           "cases": results}
    if args.yaw_grid is not None:
        doc["passing_bands"] = bands([(r["arrival_yaw_vs_pour_start_deg"], r["passes"])
                                      for r in results])
    (sweep_dir / "sweep.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[sweep] {doc['pass_count']}/{doc['case_count']} pass")
    if "passing_bands" in doc:
        print("[sweep] passing bands:", json.dumps(doc["passing_bands"]))
    return 0 if doc["pass_count"] == doc["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
