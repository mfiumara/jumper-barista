"""Closed-loop turn / walk / align on the locomotion policy, gated at the gait threshold.

From the accepted A4 spike (MOTION_CHAIN.md Q3): commands go in through the app's own
``set_command_axes`` in physical units; below |v| 0.05 m/s the policy stands, so motion
commands are kept at or above 0.06 m/s and 0.3 rad/s. The heading comes from simulation
ground truth: this is a film harness, not a robot estimator.
"""

from __future__ import annotations

import math

GATE_V = 0.06
GATE_W = 0.30


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def nav_errors(c, target):
    tx, ty, tyaw = target
    r = c.log[-1]
    ex, ey = tx - r["x"], ty - r["y"]
    along = ex * math.cos(tyaw) + ey * math.sin(tyaw)
    lat = -ex * math.sin(tyaw) + ey * math.cos(tyaw)
    return along, lat, _wrap(tyaw - r["yaw"]), math.hypot(ex, ey)


def _to_body(c, target, vx_t, vy_t):
    d = _wrap(c.log[-1]["yaw"] - target[2])
    return vx_t * math.cos(d) + vy_t * math.sin(d), -vx_t * math.sin(d) + vy_t * math.cos(d)


def turn_to(c, target, wmax=0.6, tol_deg=1.5, timeout=8.0, beat="turn", settle_s=1.0):
    c.beat = beat
    t_lim = c.t_substep() + timeout
    while c.t_substep() < t_lim:
        e = nav_errors(c, target)[2]
        if abs(e) < math.radians(tol_deg):
            break
        c.cmd = {"ang_vel_z": math.copysign(max(GATE_W, min(wmax, 2.0 * abs(e))), e)}
        c.step()
    c.cmd = {}
    c.run(settle_s, beat=beat)


def walk_to(c, target, vmax=0.15, stop_at=-0.005, beat="walk", settle_s=1.5):
    """Gated walk: forward speed tapers to the gait gate; heading and lateral servo."""
    c.beat = beat
    along0 = nav_errors(c, target)[0]
    t_lim = c.t_substep() + abs(along0) / GATE_V + 5.0
    while c.t_substep() < t_lim:
        along, lat, eyaw, _ = nav_errors(c, target)
        if along < stop_at:
            break
        vx = max(GATE_V, min(vmax, 1.0 * along))
        vy = max(-0.04, min(0.04, 1.5 * lat))
        bx, by = _to_body(c, target, vx, vy)
        c.cmd = {"lin_vel_x": bx, "lin_vel_y": by, "ang_vel_z": max(-0.5, min(0.5, 2.0 * eyaw))}
        c.step()
    c.cmd = {}
    c.run(settle_s, beat=beat)


def align_to(c, target, passes=3, tol_m=0.005, tol_deg=1.5, beat="align", gain=2.0, settle_s=1.5):
    """Gate-speed pulses along the residual, re-measured after each settle."""
    hist = []
    for k in range(passes):
        along, lat, eyaw, d = nav_errors(c, target)
        hist.append({"pass": k, "along_mm": 1e3 * along, "lat_mm": 1e3 * lat,
                     "yaw_deg": math.degrees(eyaw), "t": c.t_substep()})
        if d < tol_m and abs(eyaw) < math.radians(tol_deg):
            break
        if abs(eyaw) >= math.radians(tol_deg):
            turn_to(c, target, wmax=GATE_W, tol_deg=tol_deg * 0.5, timeout=3.0, beat=beat)
            continue
        dur = gain * d / GATE_V
        vx_t, vy_t = GATE_V * along / d, GATE_V * lat / d
        c.beat = beat
        for _ in range(max(1, round(dur / c.step_dt))):
            bx, by = _to_body(c, target, vx_t, vy_t)
            c.cmd = {"lin_vel_x": bx, "lin_vel_y": by}
            c.step()
        c.cmd = {}
        c.run(settle_s, beat=beat)
    along, lat, eyaw, d = nav_errors(c, target)
    hist.append({"pass": "final", "along_mm": 1e3 * along, "lat_mm": 1e3 * lat,
                 "yaw_deg": math.degrees(eyaw), "t": c.t_substep()})
    return hist
