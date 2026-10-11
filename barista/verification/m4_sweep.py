#!/usr/bin/env python3
"""Robustness sweep of the M4 pour against a rigid stance offset (simulation only).

Each point runs the unmodified verifier ``barista/verification/run_m4.py`` in its own
process with ``--spawn-offset DX DY DYAW``: the robot is spawned rigidly offset from the
trajectory's ``base_start`` (the M4 ``pour_start``) and every run_m4 gate is evaluated as is.
The default grid is the M4R acceptance grid: yaw -2..+2 deg in 0.5 deg steps (dx = dy = 0)
plus dx +-2 mm and dy +-2 mm (yaw 0).

    python barista/verification/m4_sweep.py --out OUT [--jobs 7] [--keep-csv]

Writes ``OUT/sweep.json`` and ``OUT/sweep_table.csv`` (one row per point, every gate and the
margin metrics) and exits 0 only when every point passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN_M4 = Path(__file__).with_name("run_m4.py")

#: Metrics reported per point, as (column, path into summary.json, scale).
METRICS = (
    ("carried_grasp_translation_mm", ("motion", "maximum_carried_grasp_translation_m"), 1000.0),
    ("carried_grasp_rotation_deg", ("motion", "maximum_carried_grasp_rotation_deg"), 1.0),
    ("set_down_error_mm", ("motion", "maximum_final_position_error_m"), 1000.0),
    ("inward_excursion_mm", ("stream", "maximum_inward_excursion_m"), 1000.0),
    ("turns", ("stream", "turn_count"), 1.0),
    ("first_radius_mm", ("stream", "first_radius_m"), 1000.0),
    ("edge_margin_mm", ("stream", "minimum_edge_margin_m"), 1000.0),
    ("landing_success_rate", ("stream", "landing_success_rate"), 1.0),
    ("spout_height_min_mm", ("motion", "spout_height_min_m"), 1000.0),
    ("spout_height_max_mm", ("motion", "spout_height_max_m"), 1000.0),
    ("pour_duration_s", ("motion", "pour_duration_s"), 1.0),
    ("min_brewer_clearance_mm", ("motion", "minimum_pitcher_brewer_clearance_all_phases_m"),
     1000.0),
    ("min_robot_brewer_clearance_mm", ("motion", "minimum_robot_brewer_clearance_all_phases_m"),
     1000.0),
)


def default_grid() -> dict[str, tuple[float, float, float]]:
    """The M4R acceptance grid: name -> (dx m, dy m, dyaw deg)."""
    grid = {}
    for tenth in range(-20, 21, 5):
        yaw = tenth / 10.0
        grid[f"yaw{yaw:+.1f}deg"] = (0.0, 0.0, yaw)
    grid.update({
        "dx+2mm": (0.002, 0.0, 0.0), "dx-2mm": (-0.002, 0.0, 0.0),
        "dy+2mm": (0.0, 0.002, 0.0), "dy-2mm": (0.0, -0.002, 0.0),
    })
    return grid


def _dig(summary: dict, path: tuple[str, ...]):
    value = summary
    for key in path:
        value = value[key]
    return value


def point_row(name: str, offset: tuple[float, float, float], exit_code: int,
              summary: dict | None) -> dict:
    """One table row: the offset, the exit code, every gate and the margin metrics."""
    dx, dy, dyaw = offset
    row: dict = {"point": name, "dx_mm": dx * 1000.0, "dy_mm": dy * 1000.0, "dyaw_deg": dyaw,
                 "exit_code": exit_code}
    if summary is None:
        row.update(passes=False, failing_gates="no summary (crash)")
        return row
    gates = {f"motion.{key}": bool(value) for key, value in summary["motion"]["checks"].items()}
    gates.update(
        {f"stream.{key}": bool(value) for key, value in summary["stream"]["checks"].items()}
    )
    gates["torque"] = bool(summary["torque"]["passes"])
    gates["payload"] = bool(summary["payload"]["passes"])
    clearance = summary["motion"]["pitcher_robot_clearance"]["by_scope"]
    row["passes"] = bool(summary["passes"])
    row["failing_gates"] = ";".join(sorted(key for key, value in gates.items() if not value))
    for column, path, scale in METRICS:
        row[column] = float(_dig(summary, path)) * scale
    row["min_pitcher_robot_clearance_carried_mm"] = clearance["carried"]["minimum_m"] * 1000.0
    row["min_pitcher_robot_clearance_spiral_mm"] = (
        clearance["spiral"]["minimum_m"] * 1000.0 if "spiral" in clearance else float("nan")
    )
    row["pitcher_robot_nearest_pair"] = clearance["carried"]["nearest_pair"]
    torque = [v for v in summary["torque"]["per_joint_and_phase"].values()
              if v["phase"] == "whole_run"]
    row["torque_peak_nm"] = max(v["peak_applied_abs_nm"] for v in torque)
    row["torque_max_theta"] = max(v["maximum_thermal_theta"] for v in torque)
    row["torque_longest_above_1p2_s"] = max(
        v["above_continuous_max_contiguous_s"] for v in torque
    )
    row["qpos_sha256"] = summary["determinism"]["qpos_sha256"]
    row.update(gates)
    return row


def write_table(rows: list[dict], path: Path) -> None:
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--points", nargs="*", default=None, help="subset of the grid names")
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--keep-csv", action="store_true",
                        help="keep each point's per-substep CSVs (about 0.4 GB a point)")
    args = parser.parse_args()
    grid = default_grid()
    names = args.points or list(grid)
    unknown = sorted(set(names) - set(grid))
    if unknown:
        raise SystemExit(f"unknown points {unknown}; grid is {list(grid)}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> dict:
        dx, dy, dyaw = grid[name]
        case = out / name
        command = [sys.executable, str(RUN_M4), "--no-render", "--out", str(case),
                   "--spawn-offset", repr(dx), repr(dy), repr(dyaw)]
        if args.trajectory is not None:
            command += ["--trajectory", str(args.trajectory)]
        with (out / f"{name}.log").open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        summary_path = case / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if not args.keep_csv:
            for stem in ("motion_substeps.csv", "torque_substeps.csv"):
                (case / stem).unlink(missing_ok=True)
        row = point_row(name, grid[name], code, summary)
        print(json.dumps({k: row.get(k) for k in (
            "point", "exit_code", "passes", "failing_gates", "carried_grasp_translation_mm",
            "set_down_error_mm", "inward_excursion_mm", "min_pitcher_robot_clearance_carried_mm",
        )}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(one, names))
    write_table(rows, out / "sweep_table.csv")
    doc = {
        "simulation_only": True,
        "note": "rigid spawn offset from base_start (pour_start); unmodified run_m4 gates",
        "pass_count": sum(bool(row["passes"]) for row in rows), "point_count": len(rows),
        "points": rows,
    }
    (out / "sweep.json").write_text(json.dumps(doc, indent=1) + "\n", "utf-8")
    print(f"[m4-sweep] {doc['pass_count']}/{doc['point_count']} pass; "
          f"table {out / 'sweep_table.csv'}")
    return 0 if doc["pass_count"] == doc["point_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
