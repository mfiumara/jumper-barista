"""Augment pour_reference.npz with the pitcher's measured/commanded origin, from run_m4's CSV."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
ref = dict(np.load(run / "pour_reference.npz"))
with open(run / "motion_substeps.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
cfg = json.loads((run / "m4_spiral_pour_cafe_1k.json").read_text())
dt = float(cfg["physics_dt_s"])
# motion row i is logged after substep i (time (i+1)*dt); the hook at control tick k sees the
# state after substep k*dec - 1, i.e. row k*dec - 1 (tick 0: the spawn, row -> use row 0).
dec = round(float(cfg["control_dt_s"]) / dt)
pin, pmeas, pcmd = [], [], []
for k, tick in enumerate(ref["tick"]):
    i = max(0, int(tick) * dec - 1)
    r = rows[i]
    p = np.array([float(r[f"pitcher_{a}_m"]) for a in "xyz"])
    palm = np.array([float(r[f"palm_{a}_m"]) for a in "xyz"])
    rot = np.array([[float(r[f"palm_{a}{b}"]) for b in "xyz"] for a in "xyz"])
    off = rot.T @ (p - palm)
    pin.append(off)
    pmeas.append(p)
    cr = ref["palm_cmd_rot"][k].reshape(3, 3)
    pcmd.append(ref["palm_cmd"][k] + cr @ off)
ref["pitcher_in_palm"] = np.asarray(pin)
ref["pitcher_meas"] = np.asarray(pmeas)
ref["pitcher_cmd"] = np.asarray(pcmd)
final = [r for r in rows if r["phase"] == "final_settle"]
pf = np.array([[float(r[f"pitcher_{a}_m"]) for a in "xyz"] for r in final]).mean(axis=0)
target = np.asarray(cfg["set_down_target_m"])
ref["setdown_bias_xy"] = (pf - target)[:2]
touch = next(float(r["time_s"]) for r in rows if r["phase"] in ("lower", "set_down") and r["counter_contact"] == "True")
ref["touchdown_time_s"] = np.float64(touch)
np.savez_compressed(run / "pour_reference.npz", **ref)
print("bias_mm", 1000 * ref["setdown_bias_xy"], "touchdown", touch, "n", len(ref["tick"]))
