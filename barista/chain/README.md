# A4 motion chain (Act 2)

Simulation only (MuJoCo, `native:cpu`, offscreen). The water is a visual ballistic stream
drawn from the measured spout; there is no fluid, and nothing here implies that the real
robot has done any of it.

`run_chain.py` runs **one continuous, deterministic world** in the A1 scene `cafe_bar`:

```
dance (idle, existing policy) -> chime -> stop -> excited flat jump in place -> exclaim
-> look / listen -> confirm -> turn 90 deg -> walk along the bar to pre_align -> align
-> scripted approach to pour_start -> accepted M4 spiral pour (closed-loop) -> set-down
```

The world is built once. After the boot nothing resets it or writes its state: the run
counts every root/joint state write (`continuity.state_writes_after_boot`, must be 0) and
logs a qpos hash at each beat boundary. Physics runs at 1 kHz, the app and its policies tick
at 200 Hz, and the M4 keyframe targets update at 50 Hz (MOTION_CHAIN.md Q2, R3).

## Modules

| file | role |
|---|---|
| `common.py` | paths, the cafe M4 trajectory (A1's rigid transfer, re-rated to 1 kHz), pose helpers |
| `world.py` | the chain world: jumper.barista + the jumper app's FSM (`deploy/fsm`) with a scripted pad; every physics substep goes through the M4 audited servo and is reduced as `run_m4._summarize_torque` does |
| `navigation.py` | gated closed-loop turn / walk / align on the locomotion policy (spike Q3) |
| `approach.py` | `pre_align` -> `pour_start`: scripted quasi-static six-foot approach (leg IK, kp 10 / kd 0.5) |
| `pour.py` | the accepted `run_m4.py`, unmodified, with a re-targeting hook (below) |
| `beats.py` | the `jumper-barista-beats/1` log and its validator |
| `footprint.py` | dance-area envelope (the spike's p8 tool with its 36-vs-29 qpos crash fixed) |
| `pour_sweep.py` | +-2 mm / +-2 deg arrival-perturbation sweep from the chain's real arrival state |
| `render.py` | verifier MP4 (wide camera) and one still per beat from the saved render states |
| `augment_reference.py` | adds the pitcher's commanded/measured path to a recorded pour reference |

## Supervisor decisions implemented

* **Scene.** The accepted A1 scene is `cafe_bar` (`scenes/cafe_bar.py`, A1 SHA 1da1c85). The
  supervisor confirmed this at 14:22: `coffee_bar` in the 13:15 decision was a naming slip.
  `coffee_bar` is the M2 package that `cafe_bar` embeds rigidly.
* **No open-loop blend into the M4 start.** The walk ends at A1's `pre_align`
  (x 0.72 m, 80 mm short of `pour_start`). No policy can cover those 80 mm: the six-leg
  stance's LF palm sits in the pitcher at `pour_start`, and the five-foot claw walk stalls or
  overheats with the arm raised (spike Q4). `approach.py` covers them as a reference
  trajectory instead:
  1. blend to HOME in place;
  2. slide the body 40 mm over all six planted feet;
  3. step RR, LR, RM, LM and RF, one at a time with five feet down, to their HOME footholds
     *relative to `pour_start`*;
  4. lift LF straight up from where it walked to (it stays >= 36 mm from the pitcher), raise
     the shoulder, then fold to a high claw-open pose above the grip bar;
  5. slide the body the last 40 mm on five feet;
  6. lower LF onto the pour's own first palm target, by IK on the measured base.

  The walk's x/y/yaw error is corrected here, because the footholds and the body target are
  `pour_start`'s.
* **Closed-loop spout-space re-targeting** (`pour.py`, no RL). The verifier runs inside the
  chain's world: its env construction, reset, close and its two initial teleports are
  skipped. Every 20 ms, the four LF targets that `run_m4._target` returns are corrected by
  damped least-squares IK against one nominal reference replay (`--mode reference`):
  * before the grasp: the palm frame tracks the nominal commanded palm pose;
  * while carried: the *measured* spout outlet (its offset in the measured palm frame) tracks
    the nominal commanded spout position;
  * while lowering: the measured pitcher origin is servoed onto its pickup spot. The accepted
    keyframes set it down about 20 mm short at 1 kHz (20.06 mm, which fails the 20 mm gate;
    this is risk R3);
  * from touchdown on: the accepted release and retreat motion, offset by the touchdown
    correction.

  A base-yaw error is split: the spout stays on the station's path, and the pitcher heading
  turns by half the error. With the robot exactly where the reference had it, the IK returns
  the accepted keyframes (residual < 1e-6 m). Every run_m4 gate is evaluated unchanged.
* **Arrival-perturbation sweep:** `pour_sweep.py`.
* **The dance area is re-measured on `cafe_bar`** from the chain's own states
  (`summary.json` `jump`), and with `footprint.py` on any run directory.

## Reproduce

```bash
source .venv/bin/activate
B=<scratch>/bundle
python scripts/deploy.py --manifest jumper --allow-incomplete --bundle $B    # builds the FSM dylib too
python barista/chain/pour.py --mode reference --out <scratch>/ref_1k         # nominal 1 kHz replay
python barista/chain/augment_reference.py <scratch>/ref_1k
python barista/chain/run_chain.py --bundle $B/jumper --reference <scratch>/ref_1k/pour_reference.npz --out <run>
python barista/chain/pour_sweep.py --run <run> --reference <scratch>/ref_1k/pour_reference.npz --jobs 3
python barista/chain/render.py <run>
python barista/chain/footprint.py <run>
python -m pytest tests/test_barista_chain.py
```

Outputs in `<run>`:

* `beats.json` (`jumper-barista-beats/1`, with the exclaim/listen/confirm holds as parameters
  `--exclaim-s/--listen-s/--confirm-s`);
* `summary.json` (per-beat roll/pitch/slip, jump and dance-area figures, walk/align/approach
  errors, pour gates, torque audit, determinism hashes);
* `trace.csv` (200 Hz);
* `torque_applied.npz` (every substep, all 22 joints);
* `render_states.npz` (50 fps qpos, the stream paths, the qpos SHA-256; A5 cuts from these
  without re-simulating);
* `pour/` (run_m4's own summary, CSVs, render states, re-targeting log).

## Known limitations

* The approach (step 3 above) is scripted joint-space motion at run_m4's soft gains, not a
  trained gait. It is plausible on the hardware (quasi-static, five or six feet down,
  inside the torque audit), but it is slow.
* The physics contact model switches to M4's elliptic cone (`impratio`) when the pour starts.
  `run_m4` sets this itself, and the grasp needs it. Before that point the policies run in
  their trained contact model. This is a solver setting, not a state write.
* run_m4's 1 kHz torque CSV export keeps one substep in 100 (`pour.py::_SlimCsv`; about 390 MB
  in full). Every substep is still reduced by run_m4's own torque gate, and the chain keeps
  all of them in `torque_applied.npz`.
* Heading and position feedback are simulation ground truth. This is a film harness, not a
  robot estimator.
