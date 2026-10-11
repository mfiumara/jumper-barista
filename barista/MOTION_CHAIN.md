# A4 spike: one continuous motion chain (dance -> chime -> jump -> look -> turn -> walk -> align -> pour)

Research note for milestone A4 (card t_a37e3ed8, base `e6765ea`). Simulation only: every number
below comes from a MuJoCo probe on this Mac (`native:cpu`), not from the robot. The water is a
visual approximation. Follows the supervisor amendment: the ledge/step jump is dropped, and the
jump is the existing flat jump, done in place on the bar.

## Verdict in five lines

1. **Mode switching is solved by the repository itself.** Drive the jumper app's own controller
   (`deploy/fsm`, `mjrl_fsm`, through `rl/mjrl/app_play.py::AppPlayer`) inside one
   `jumper.barista` world, with a scripted pad and physical-unit commands
   (`Fsm.set_command_axes`). Every transition probed stays inside the torque budget.
2. **The chain must run at 1 kHz physics, not M4's 200 Hz.** At 200 Hz the trained policies
   degrade: the jump lands 78 mm / 10.6 deg off, against 8 mm / 1.5 deg at 1 kHz. M4 replayed at
   1 kHz with its 50 Hz keyframe targets passes every gate, but its set-down margin is 0.09 mm.
3. **Turn and walk are accurate enough:** 90 deg plus 0.6-1.0 m lands within 1.8-2.9 mm and
   0.9 deg using closed-loop gated commands.
4. **The pour hand-off is the blocker.** The open-loop M4 replay fails its own gates when the
   base starts 2 mm or 1 deg away from its recorded start. It also fails when started from a
   real walked-and-blended stance, even after an exact rigid re-anchoring of the station. Two
   more problems: the walking stance's LF palm overlaps the pitcher at the M4 slot, and the
   five-foot approach with the claw raised breaks the thermal budget. **A4 needs a closed-loop,
   spout-space re-targeting of the pour (and an A1 station layout that keeps the pitcher clear
   of the walking feet). This is scope growth and needs a supervisor decision.**
5. **Excited jump in place: feasible.** Rise 0.120-0.122 m, flight 0.275 s, landing drift
   8.2-9.2 mm and -1.06 to -1.51 deg, about 1.66 s from touchdown to settled, torque audit pass
   on all 4 runs. Size the dance area to a **0.40 m radius** around the dance spot (the measured
   envelope is 0.36 m; see Q5).

## Harness (how every number was produced)

Probe code is out of tree, as the card requires. It is archived at
`~/.hermes/profiles/jumper-barista-engineer/a4_spike/` (`*.py`, `runs/*/summary.json`, logs, videos)
and as `a4_spike_probes.tgz`, which is attached to the card. To rerun it:

```bash
cd <worktree> && uv venv -p python3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision && uv pip install -e ".[dev]"
S=~/.hermes/profiles/jumper-barista-engineer/cache/scratch/a4   # chainlib.py's BUNDLE/SCRATCH
mkdir -p $S && cp ~/.hermes/profiles/jumper-barista-engineer/a4_spike/*.py $S/
python scripts/deploy.py --manifest jumper --allow-incomplete --bundle $S/bundle   # exit 0, ~4 min
export A4_REPO=$PWD
```

`chainlib.py::Chain` works as follows:

- It builds `jumper.barista` (the five-foot world with the M4 `AuditedServoCurveActuator`).
- It sets the physics rate and calls `step_the_world_at(200 Hz)` and `leave_falls_to_the_app`.
- It removes the random spawn, the five-foot carried-arm reset and the claw step events, which
  would overwrite the app's LF targets. It keeps the seeded startup randomisation, as M4 does.
- It boots from the locomotion export's `default_joint_pos`.
- It loads the bundle's controller (`deploy/fsm/target/release/libmjrl_fsm.dylib`, built by
  `deploy.py`) with `operators=False`. Commands then go in through `set_command_axes`, and
  switches go in as pad frames (`A`, `menu`, `LB`, and the d-pad).
- The torque audit snapshots **every physics substep** through the M4 actuator's own audit API.
  It reduces the snapshots exactly as `barista/verification/run_m4.py::_summarize_torque` does:
  curve or effective-limit violations, the longest contiguous run above 1.2 N·m (≤0.300 s
  passes), thermal θ < 1, and a thermal recompute error ≤ 1e-5.
- `step_direct` leaves the app and writes joint targets with kp 10 / kd 0.5, as run_m4 does.

| probe | command (from repo root, venv active) | exit |
|---|---|---|
| p2 dance→stop→jump→walk | `python $S/p2_dance_jump.py --hz 1000 --dance <dance_brazilian\|dance_maze\|dance_crab\|dance_dream_wings> --dance-s 15 --out $S/runs/p2_<dance>_1k --render` | 0 ×4 |
| p2 at M4's rate | `python $S/p2_dance_jump.py --hz 200 --dance dance_brazilian --dance-s 12 --out $S/runs/p2_brazilian_200` | 0 |
| p2b settle / trigger | `python $S/p2b_settle.py $S/runs/p2_dance_*_1k` | 0 |
| p8 footprint / dance area | `python $S/p8_footprint.py $S/runs/p2_dance_*_1k` | 0 |
| p3 turn+walk (4 legs) | `python $S/p3_turn_walk.py --hz 1000 --stop-at -0.005 --align-gain 2.0 --passes 5 --out $S/runs/p3_legs_tuned_1k --render` (and the untuned default `--out $S/runs/p3_legs_1k`) | 0 |
| p4 hand-off (bar, 6 legs) | `python $S/p4_handoff.py --hz 1000 --walk 0.30 --ramp-s 2.0 --bow --out $S/runs/p4_handoff_1k --render`, then `python $S/p4b_pitcher_contacts.py $S/runs/p4_handoff_1k` | 0, 0 |
| p4 five-foot approach | `... --claw-approach --pre-dist 0.10 [--arm-kf0] --out $S/runs/p4_claw[kf0]_1k` | 0 |
| p4 plain arrival | `python $S/p4_handoff.py --hz 1000 --scene plain --walk 0.60 --out $S/runs/p4_plain_1k` | 0 |
| M4 baseline | `python barista/verification/run_m4.py --out $S/runs/m4_baseline --no-render` | 0 (pass) |
| p6 M4 at 1 kHz | `python $S/p6_m4_rate.py --hz 1000 [--decimation 20] --out $S/runs/p6_m4_1k[_dec20]` | 1 / **0** |
| p6 base offsets | `python $S/p6_m4_rate.py --hz 200 --base-dx/--base-dy/--base-dyaw-deg <v> --out $S/runs/p6_off_<n>` | see Q4 |
| p9 M4 from chain state | `python $S/p9_m4_from_handoff.py --state $S/runs/p4_plain_1k/handoff_state.npz --hz 1000 --decimation 20 --out $S/runs/p9_from_plain_1k` | **1** |

`p6`/`p9` monkeypatch only the inputs: the timestep, the decimation, and the initial robot state.
The verifier (`run_m4.main()`) and its gates are used unmodified, and the exit code is
run_m4's. The wall time of one 1 kHz probe is 4-8 min. One 1 kHz M4 replay takes 14 min.

## Q1 Inventory (local, all real files, not LFS pointers; `git lfs` is not installed here)

The `tasks/jumper/*/out/example/` exports are committed as plain files (each `actor.onnx` is
1.0-1.55 MB). The jumper bundle builds from them, and the `.app` replays on CPU through
onnxruntime.

| need | mode in `deploy/manifests.json` | export | rate | notes |
|---|---|---|---|---|
| walk/turn/stand | `locomotion` (default) | `posture/out/example` (model_74800) | 200 Hz control, trained at **1 kHz physics** | gait gate: \|cmd\| < 0.05 m/s means stand (`gait_phase.command_threshold`) |
| dances | `dance_crab/brazilian/maze/dream_wings` (Menu + d-pad) | `dance*/out/example` | 50 Hz | `dance_waist` exported but not in the manifest; maze clip ends by itself after ~10 s |
| excited jump | `jump` (A, from locomotion) | `jump/out/example` (model_3100, residual on `high_jump_flat`) | 200 Hz, trained at 400 Hz | `starts_on_entry`; hands back to locomotion when the recording ends |
| bow | `gesture_bow` (d-pad down, from locomotion) | `gesture_bow/out/example` | 50 Hz | exercised in p4 after the pour segment |
| claw walk | `claw_left` (LB) | `five_foot/out/example` (model_86600) | 50 Hz | carried-arm hook not active with `operators=False` |
| pour | none (reference replay) | `barista/trajectory/m4_spiral_pour.json` | 200 Hz physics, 50 Hz targets | open loop |

`ref_free_jump/out/example` also exists. It is not in the jumper manifest and is not needed.
Nothing the chain needs is missing.

## Q2 Mode switching and transients

The FSM switches as follows (from `deploy/jumper.controller.toml` and the probes):

- A switch arms `mode_switch_ramp_s = 1.0` with ramp gains kp 15 / kd 0.8.
- The new policy runs only after the measured pose is within `pose_reach_tol = 0.10` rad.
- Timed motions (the jump and the clips) hand back to `locomotion` when they end.
- Menu released on its own (`leave_motion`, on `fall`) leaves a dance.
- Mode timings from p2: press to jump policy in 0.17-0.66 s, which depends on the pose. Dance
  stop to locomotion running in 1.07-1.11 s.

**The Python loop pattern:** call `AppPlayer.__call__` once per 5 ms tick. To leave the app for
a scripted segment, write targets directly and stop ticking the controller. To come back,
advance `player._tick` by the gap. After a direct segment the controller re-entered
`locomotion` with tilt ≤ 2.2 deg and the bow then ran (p4). It went straight to `running`, with
no warm-start ramp. A4 should force `safe`→warm start explicitly rather than rely on that.

Transients at 1 kHz. The p2 rows are means of 4 runs; the torque columns are the worst beat.

| transition | max tilt | foot slip, max per 5 ms | base xy | peak τ / max θ / longest >1.2 N·m | audit |
|---|---|---|---|---|---|
| stand → dance entry | **15.8 deg spike** (brazilian), 7-17 deg | 1.5-2.7 mm | dance start shifts the base 60-90 mm | 1.746 / 0.88 / 0.219 s | pass |
| dance → stand (Menu) | 4.6-9.6 deg | 0.9-1.6 mm | 72-80 mm (returns toward start) | 1.746 / 0.60 / 0.009 s | pass |
| stand → walk 0.15 m/s | 1.3-1.4 deg | 1.4-1.7 mm | — | 1.746 / 0.54 / 0.025 s | pass |
| walk → stand | 1.3-1.5 deg | 0.9-1.4 mm | 3-13 mm coast | 1.39 / 0.54 / 0.003 s | pass |
| locomotion → claw_left (LB) | ≤ 2 deg | — | — | in the `align` beat | pass (default arm) |

## Q3 Turn 90 deg and walk 0.6-1.0 m (p3, flat floor, 1 kHz)

- Open loop (0.6 rad/s for 90 deg, then 0.15 m/s for the time the distance needs): the 0.8 m
  leg ends **205 mm off** (196 mm short, 61 mm lateral, 2.1 deg).
- Closed loop on base pose: the turn takes 4.1 s at a minimum of 0.3 rad/s. Slower turns sit
  inside the gait gate, because turn_radius 0.2 × ω must be ≥ 0.06.
- Walk: 0.06-0.15 m/s with heading and lateral servo.
- Gate-speed align pulses below 0.06 m/s do nothing (the policy stands).

| leg | yaw err after turn | after walk, stop at +20 mm | after walk, stop at −5 mm (tuned) | walk time (tuned) |
|---|---|---|---|---|
| +90° / 0.6 m | 0.74° | 26.8 mm / 0.9° → align 8.6 mm | **2.4 mm / 0.91°** | 8.0 s |
| −90° / 1.0 m | 0.16-1.60° | 27.2 mm → align 11.4 mm | **2.9 mm / 0.22°** | 11.5 s |
| +90° / 0.8 m | 0.86-1.84° | 26.8 mm → align 11.5 mm | **1.8 mm / 0.74°** | 9.7 s |
| −90° / 0.8 m | 0.09-1.47° | 26.9 mm → align 9.6 mm | **2.2 mm / 0.92°** | 9.8 s |

The torque audit passes on the whole run: peak 1.746 N·m, θ ≤ 0.60, longest stretch above
1.2 N·m 0.115 s (during the turn).

Alignment strategy: walk the closed loop with the stop threshold biased by −5 mm, then settle
for 1.5 s. Use 0.06 m/s gate pulses only for a residual above 5 mm. The yaw correction is a
≥ 0.3 rad/s turn pulse. The heading comes from simulation ground truth; this is a film harness,
not a robot estimator.

## Q4 Pour hand-off: reachable, or does M4 need re-anchoring?

The joint-space blend itself is benign. p4_plain: a 2 s smoothstep from the walking stance to
M4's start (HOME + keyframe-0 arm + claw open) at kp 10 / kd 0.5 gives:

- base shift 1.0 mm and 0.58 deg;
- tilt ≤ 1.36 deg, 5 feet in contact;
- peak 0.63 N·m, θ 0.26.

**But M4 does not survive a non-identical start.** Base-offset matrix (p6, run_m4 at 200 Hz,
gates unmodified):

| offset | result | failing gates |
|---|---|---|
| none (baseline) | **pass** | — |
| dx −2 mm | **pass** | — |
| yaw +2° | **pass** | — |
| dx +2 mm | fail | turn_count (2.74 turns), outward_progress, inward_excursion |
| dy +2 mm | fail | carried_grasp_translation |
| dy −2 mm | fail | stable_final_set_down (20.1 mm) |
| yaw −1° | fail | turn_count (2.77), grasp translation |
| ±5 mm, dy +10, dx +10, yaw ±2/5° | fail | spout_height, set-down, or turn count |

- **p9:** M4 restarted from the real p4_plain hand-off state, with the robot rigidly placed
  exactly on `base_start` (shift 11.3 mm, 1.56 deg) to stand in for a station re-anchored to
  the arrival pose. It **fails 12 gates**: spout height 9.6-15.8 mm (needs 20-50), 2.34 turns,
  the pitcher touches the brewer (−0.1 mm), and set-down 26.8 mm. The cause is that the walked
  stance is not M4's settled stance: base z is 0.1011 m after the blend, against M4's spawn at
  0.10647. The open-loop arm keyframes then miss the bed.
- **On the bar:** walking into the M4 slot on 6 legs puts the LF palm on the pitcher. p4 had
  0.3-0.8 mm nudges from t = 11 s, at 20 mm short, and a **19 mm shove during the blend**.
- **Five-foot approach** (claw_left from 0.10 m short): the default carried arm hits the pitcher
  (34 mm). With the arm held at keyframe 0 the claw clears it, but the policy did not close the
  gap (stuck at 104 mm). The audit also fails: θ 1.001 and **29.6 s above 1.2 N·m** in `align`.
  This is not usable.
- Stopping 0.10 m short in the 6-leg stance and blending in place causes **zero pitcher contact**
  and passes the audit (p4_clawkf0 hold, p4b events = []).

**Answer: no.** The ≤ 5 mm / 2° hand-off is achievable for the base (Q3). It is not sufficient,
because the M4 replay tolerates about 0 mm. A rigid re-anchoring is not sufficient either (p9).
A4 needs:

1. **Closed-loop pour re-targeting:** M4's spout/handle path expressed in the bed and pitcher
   frame, with per-tick arm IK from the measured base pose. Re-verify with the run_m4 gates
   unchanged.
2. **An A1 layout or grasp change** that keeps the 6-leg arrival stance ≥ 20 mm clear of the
   pitcher. Either stop 0.10 m short and reach further, or place the pitcher handle outside the
   LF palm's stance footprint.

Both re-open accepted M4 work, so they are a supervisor decision.

## Q5 Dance → stop → excited flat jump in place → settle (p2, 1 kHz, 4 runs)

| metric | brazilian | maze | crab | dream_wings |
|---|---|---|---|---|
| jump rise (apex − stance) | 0.122 m | 0.121 m | 0.121 m | 0.120 m |
| longest flight | 0.275 s | 0.275 s | 0.275 s | 0.275 s |
| landing drift xy / yaw | 8.2 mm / −1.51° | 8.7 mm / −1.13° | 9.0 mm / −1.51° | 9.2 mm / −1.06° |
| max tilt during jump | 8.3° | 8.6° | 8.3° | 8.5° |
| touchdown → settled (tilt < 2°, \|Δz\| < 3 mm for 0.5 s) | 1.66 s | 1.66 s | 1.66 s | 1.66 s |
| A press → settled | 3.03 s | 2.54 s | 2.94 s | 2.93 s |
| in-place shuffle after settle | 3.3 mm | 3.3 mm | 3.4 mm | 3.3 mm |
| `excited_jump` audit: peak / θ / >1.2 N·m | 1.746 / 0.48 / 0.06 s | 1.746 / 0.47 / 0.06 s | 1.746 / 0.47 / 0.06 s | 1.746 / 0.47 / 0.06 s |
| whole run audit | pass | pass | pass | pass |

Dance excursion and dance area (p8; radius measured from the pre-dance base position to any
robot collision geom, centre plus bounding radius):

| | brazilian | maze | crab | dream_wings |
|---|---|---|---|---|
| base radius during the dance (11-15 s) | 0.090 m | 0.056 m | 0.061 m | 0.110 m |
| base yaw range | 17.3° | 3.3° | 19.9° | 19.2° |
| standing footprint radius | 0.274 m | 0.274 m | 0.274 m | 0.274 m |
| envelope: dance + stop + jump | 0.338 m | 0.310 m | 0.321 m | **0.360 m** |

**Lane A1 `dance_area`: radius 0.40 m** around the dance spot, with ≥ 0.40 m to every bar edge.
That covers the worst measured envelope, 0.36 m, plus 40 mm. Any dance window longer than 15 s
needs re-measuring: the brazilian excursion grew from 0.080 m at 11.5 s to 0.090 m at 14.6 s.
The bar must also be wider than the robot. p4 spawned 0.47 m from the 0.60 m counter edge, and
a turn put a leg over the edge and the robot fell, so keep the walking route ≥ 0.30 m from the
edges.

**Robust trigger point.**

1. Fire `chime` at any time during the dance.
2. Press Menu (stop) no earlier than 0.5 s later.
3. Press A when locomotion has been `running` for ≥ 1.0 s and the base has been quiet for
   0.25 s: tilt < 2°, |v| < 0.05 m/s, yaw rate < 10°/s. Measured pre-jump values were
   0.9-1.5° / 0.014-0.041 m/s / 4-10°/s. This is about 2.9-3.2 s after the stop press.
4. Start `exclaim` at touchdown (press + 0.7 s).
5. Allow `look` once settled (press + 3.0 s).

**Jump verdict: the flat jump in place is feasible as is, with no residual RL.**

## Q6 Torque budget

Each probe audits every physics substep with the M4 actuator. Whole-run result per probe:

- **pass:** the 4 × p2 dance/jump/walk runs; p3 (both); p4_plain; p4_handoff_1k; M4 at 1 kHz
  with dec 20.
- **fail:**
  - p2_brazilian_1k (first run, old boot): warm start lowered the five-foot carried LF arm
    under kp 15. Fixed by booting from the locomotion stance; boot is now 1.244 N·m and θ 0.11.
  - p2_brazilian_200: LF_J0 derates (θ 1.012) at 200 Hz physics.
  - p4_clawkf0: claw walk with the arm held high overheats.

Heat carries across beats, which matters for the chain: θ reached 0.88 after 15 s of dancing.
The chain should measure θ per beat and either insert stand pauses or shorten the dance. The
real-time budget on the robot is the 300 ms rule plus the thermal recurrence, the same as M4's.

## Recommended chain design (A4)

**Module layout** (new files only; no edits to shared code):

```
barista/chain/
  __init__.py
  world.py        # build jumper.barista at 1 kHz physics, 200 Hz app tick; drop reset_base /
                  # hold_carried_arm / gripper_teleop / claw_hold events; boot from locomotion stance
  app_driver.py   # AppPlayer + mjrl_fsm (operators=False): press(button|chord), command(**axes),
                  # direct(targets, kp, kd), reenter(); per-substep audit via AuditedServoCurveActuator
  navigation.py   # turn_to / walk_to / align_to with the gait gate (0.06 m/s, 0.3 rad/s), stop bias −5 mm
  pour.py         # closed-loop spout-space re-targeting of M4 (supervisor decision, see Q4)
  beats.py        # the beat script and the jumper-barista-beats/1 log
  run_chain.py    # one continuous run -> beats.json, torque_substeps.csv, summary.json, render_states.npz
barista/verification/run_chain_audit.py  # reuses run_m4 gates for the pour window and the torque reducers
```

**Beat log `jumper-barista-beats/1`** (JSON):

```json
{"schema": "jumper-barista-beats/1", "commit": "<sha>", "physics_dt_s": 0.001, "control_dt_s": 0.005,
 "beats": [{"beat": "dance", "t_start_s": 0.0, "t_end_s": 10.0}, ...]}
```

`beat` is one of `dance, chime, stop, excited_jump, exclaim, look, listen, confirm, turn, walk,
align, pour, set_down, bow`. Times are simulation time. Beats may overlap only for
audio/expression beats: `chime` overlaps `dance`, `exclaim` overlaps `excited_jump`, and
`listen`/`confirm` overlap `look`. Motion beats are contiguous and do not overlap. Suggested
timeline from the measured durations:

dance 8-12 s → chime (0.5 s before the stop) → stop 3.0 s → excited_jump 3.0 s (exclaim from
+0.7 s) → look/listen/confirm (voice length) → turn 4.1 s → walk 8-12 s for 0.6-1.0 m →
align ≤ 3 s → pour/set_down (M4 50.5 s, re-targeted) → bow ~9 s.

The motion beats total about 85-95 s of simulation before trimming for the film.

## Risks (honest)

- **R1, high:** the M4 pour is an exact-replay artifact. It is sensitive to 2 mm / 1° and to
  the start stance (p6, p9). A4 cannot "blend into the accepted M4" without the closed-loop
  re-targeting, which re-opens accepted work. **Needs a supervisor decision.**
- **R2, high:** at the M4 slot the walking stance's LF palm overlaps the pitcher. This needs an
  A1 layout or grasp change. The five-foot approach with the arm raised overheats (θ 1.001).
- **R3, medium:** the physics rate. The chain needs 1 kHz. M4 at 1 kHz passes with a 0.09 mm
  set-down margin (19.91 mm against 20 mm), and with 4 ms targets it failed at 20.24 mm.
- **R4, medium:** re-entry after a direct segment skipped the warm start. Force it.
- **R5, low:** feedback is ground-truth pose, so sim only. The dance entry tilts the body 16° and
  shifts it ~70 mm, which is on camera. Heat accumulates across beats.
- **R6, low:** with `operators=False` no task hook runs, so the five-foot claw hook is inactive.
  The pour keeps run_m4's own finger squeeze rule.

## Artifacts

Under `~/.hermes/profiles/jumper-barista-engineer/a4_spike/`, also attached to the card:

- `runs/plot_p2_dance_jump.png`, `runs/plot_p3_legs_xy.png`
- videos: `runs/p2_dance_*_1k/probe.mp4` (dance→jump→walk), `runs/p3_legs_tuned_1k/probe.mp4`,
  `runs/p4_handoff_1k/probe.mp4` (pitcher shove), `runs/p4_claw_1k/probe.mp4`,
  `runs/p4_clawkf0_1k/probe.mp4`
- per-probe `summary.json` (with `torque.<beat>`), `p2b_settle.json`, `p8_footprint.json`,
  `p4b_pitcher_contacts.json`, the run_m4 summaries for p6/p9, and every log
