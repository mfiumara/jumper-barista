#!/usr/bin/env python3
"""Arrival-perturbation sweep of the closed-loop pour (supervisor decision on A4).

Restarts the pour from the chain's real arrival state (``arrival_state.npz``: qpos, qvel and
servo heat at the end of the approach), rigidly moved by +-2 mm in x, +-2 mm in y and +-2 deg
in yaw about the base, and runs the unmodified run_m4 gates on each. One subprocess per case.

    python barista/chain/pour_sweep.py --run out/chain/run1 --reference REF.npz --jobs 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CASES = {
    "dx+2mm": (0.002, 0.0, 0.0), "dx-2mm": (-0.002, 0.0, 0.0),
    "dy+2mm": (0.0, 0.002, 0.0), "dy-2mm": (0.0, -0.002, 0.0),
    "yaw+2deg": (0.0, 0.0, 2.0), "yaw-2deg": (0.0, 0.0, -2.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="run_chain output dir")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    args = parser.parse_args()
    run = args.run.resolve()
    pour = Path(__file__).with_name("pour.py")
    sweep_dir = run / "sweep"
    sweep_dir.mkdir(exist_ok=True)

    def one(name: str) -> dict:
        dx, dy, dyaw = CASES[name]
        out = sweep_dir / name
        cmd = [sys.executable, str(pour), "--mode", "retarget", "--reference", str(args.reference),
               "--start-state", str(run / "arrival_state.npz"), "--dx", str(dx), "--dy", str(dy),
               "--dyaw-deg", str(dyaw), "--out", str(out)]
        with open(sweep_dir / f"{name}.log", "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        result = {"case": name, "exit_code": code, "dx_m": dx, "dy_m": dy, "dyaw_deg": dyaw}
        record = out / "pour_result.json"
        if record.exists():
            data = json.loads(record.read_text())
            summary = json.loads((out / "summary.json").read_text())
            result.update(passes=data["passes"], failing_gates=data["failing_gates"],
                          turns=summary["stream"]["turn_count"],
                          set_down_error_m=summary["motion"]["maximum_final_position_error_m"],
                          spout_height_m=[summary["motion"]["spout_height_min_m"],
                                          summary["motion"]["spout_height_max_m"]],
                          landing_success_rate=summary["stream"]["landing_success_rate"],
                          torque_passes=summary["torque"]["passes"])
            (out / "motion_substeps.csv").unlink(missing_ok=True)  # 66 MB each; summary kept
        else:
            result.update(passes=False, failing_gates=["no result (crash)"])
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(one, args.cases))
    doc = {"simulation_only": True, "arrival_state": str(run / "arrival_state.npz"),
           "note": "thermal state carried from the chain; the pour restarts at its own t=0",
           "pass_count": sum(r["passes"] for r in results), "case_count": len(results),
           "cases": results}
    (sweep_dir / "sweep.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[sweep] {doc['pass_count']}/{doc['case_count']} pass")
    return 0 if doc["pass_count"] == doc["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
