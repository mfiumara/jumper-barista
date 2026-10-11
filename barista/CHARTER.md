# Jumper Barista — charter

Fork of [KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper) (Apache-2.0).
Goal: Jumper, the 22-DoF crab robot, becomes Mattia's pour-over barista — in simulation.

## The one skill

**Spiral pour.** Jumper holds a small gooseneck pitcher in one claw and pours a slow,
centred spiral over a V60 dripper sitting on a cup. One skill done well beats a full routine
done badly.

A good pour looks like:

- The pitcher is grasped and lifted from the counter, not spawned in the claw.
- The spout stays 2–5 cm above the coffee bed and the stream traces an outward spiral from the
  centre (≈ 3–4 turns, ≈ 15–25 s), stopping before the paper edge.
- The body stays stable: no tipping or sliding, and the free legs do not touch the brewer.
- The pitcher is set back down at the end.
- Optional garnish if time allows: a bloom pause (short pour, ~30 s wait, main pour), a small
  "proud crab" gesture at the end, and a barista skin.

## Why (competition)

KingKong's *Crab Robot Open-Source Challenge*
(beunlimited.me/zh/events/crab-robot-challenge-2026):

- Batch 1 closes **2026-10-30**, with judging in 3–4 rounds. Earlier entries get better odds.
  **Our target submission is 2026-10-24.**
- Entry: an open-source project plus a video longer than 15 s on X and YouTube, tagged
  `#Jumper #CrabRobot #OpenSourceRobot @KingKong Robotics`.
- Scoring is 60 % popularity and 40 % expert panel, judged on looks, fun and usefulness.
  This entry aims at *useful* and *fun*.
- KingKong may reproduce winning entries on the real robot. **Everything must stay within
  real hardware limits** so the motion would be plausible on the real Jumper.

## Hard constraints

| Constraint | Source |
|---|---|
| Grasp payload ≤ 1 kg. Design target: pitcher + water ≤ 450 g | `docs/HARDWARE.md` |
| Joint torque within the servo curve and the 300 ms peak / 1.2 N·m continuous budget | `assets/jumper/motor/` |
| No `base_lin_vel` in actor observations (export refuses it) | `docs/jumper_jump.md` |
| The `.app` bundle must build and replay via the repo's own tooling | `deploy/BUNDLE.md` |
| Compute: Apple M5 Pro, macOS → `native:cpu` backend only (no CUDA) | `docs/AGENT_SETUP.md` |

Because training runs on CPU, prefer a reference trajectory or keyframes (the claw and gesture
tasks show the pattern), with RL only as a small residual where balance needs it. Renting a GPU
costs money and is a human gate.

**Water is approximate.** MuJoCo has no fluid, so the stream is modelled as particles or a
visual stream. The video and README must say "simulation" and must not imply the real robot
has done this.

## Deliverables

1. Coffee-bar scene: counter, V60 + cup/server, gooseneck pitcher, kettle as set dressing.
   Registered in `scenes/` and exported as a `.map` if the design tooling allows.
2. The spiral-pour motion: a task or trajectory, a replay script and a packaged `.app`.
3. A rendered video of at least 15 s (target 20–40 s) with a hero angle and an overhead
   spiral shot.
4. A README section ("Jumper Barista") with reproduction commands.
5. Optional: a barista `.skin`.

## Milestones

| # | Milestone | Target | Exit evidence |
|---|---|---|---|
| M1 | Toolkit runs on this Mac | 2026-10-11 | `detect.py` gates pass; an existing claw/gesture policy replays headless and renders a frame |
| M2 | Coffee-bar scene | 2026-10-14 | Scene loads with Jumper standing; render of the scene; objects at real masses and dimensions |
| M3 | Grasp + lift pitcher | 2026-10-17 | Replay: pitcher lifted ≥ 5 cm and held 3 s with no tip; torque log within budget |
| M4 | Spiral pour | 2026-10-21 | Replay: spiral of ≥ 3 turns with the stream landing inside the dripper; pitcher set down; torque log within budget |
| M5 | Video + release | 2026-10-24 | Final video file, README section, `.app` built; reviewed. **Posting is Mattia's.** |

## Gates (human: Mattia)

- Posting to X, YouTube or any other platform, and submitting the entry.
- Spending money (GPU rental, assets).
- **No pushes at all** while `origin` (mfiumara/jumper-barista) is public. Everything stays
  local until Mattia explicitly decides how to publish. Never open PRs or issues against
  upstream KingKongRobotics.
- Adding third-party assets (meshes, music, textures) needs licence evidence recorded in
  `barista/SOURCES.md`.

## Working rules

- Work only in background or headless mode. Never foreground apps, switch windows/spaces, or
  take keyboard or mouse focus. Use headless rendering (`MUJOCO_GL=egl/osmesa`, or offscreen
  CGL on macOS).
- One writer per worktree. Implementation and review are done by different profiles. Each
  review is bound to an exact commit SHA.
- Keep upstream code changes small and in their own files under `barista/`, `scenes/`, and
  `tasks/jumper/barista/`, so rebasing on upstream stays easy.
