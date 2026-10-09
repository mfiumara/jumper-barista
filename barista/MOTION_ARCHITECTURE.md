# Barista motion architecture and verification design

This note is a design input for M3 and M4. It records repository facts separately from
recommendations. It does not report a completed grasp, pour, hardware run, or measured result.
All water behaviour discussed here is a MuJoCo visual/particle approximation.

## Observed repository facts

### Claw and payload path

- `tasks/jumper/five_foot/claw.py` is the geometry and limit source for the carried
  left-front claw. `ARM_JOINTS`, `FINGER_JOINT`, `CARRIED_JOINTS`, `GRIPPER_OPEN`,
  `GRIPPER_CLOSED`, `SQUEEZE_LEAD`, `LEAD_PER_SPEED`, and `aperture_mm()` define its
  ownership, working aperture, and bounded squeeze. `FIVE_FOOT_JOINTS` excludes the carried
  arm from the locomotion policy, while `OBSERVED_JOINTS` keeps the arm and finger observable.
- `tasks/jumper/five_foot/mdp/gripper.py::GripperTeleop` is replay-only claw actuation.
  `squeeze_limited()` caps a closing target at measured finger angle plus `SQUEEZE_LEAD`
  plus a speed lead. `gripper_teleop_event()` installs it as a step event.
- `tasks/jumper/five_foot/deploy/lib.rs` implements the deployed equivalent as a
  `ModeHook`: it mirrors a left-trained policy when the claw is carried on the right,
  reads the task controls, and rewrites the decoded motor command. The hook points are
  declared by `deploy/fsm/src/hook.rs::ModeHook`; notably `before_observation()` can read
  state and `after_decode()` can rewrite the command that is published.
- `tasks/jumper/five_foot/claw.py::add_payload_body` and
  `tasks/jumper/five_foot/mdp/events.py::claw_payload` model training payload mass and
  inertia at a held object's centre. They do not model a free pitcher being contacted,
  grasped, lifted, and released.
- `tasks/jumper/five_foot/tools/pick_place.py` is an independent plain-MuJoCo claw probe.
  Its `Servo`, `Observer`, and `main()` exercise contact grasping and CPU ONNX inference,
  but its module note explicitly says its closed walking loop is not yet equivalent to
  `scripts/play.py`; it is useful as a geometry probe, not M3/M4 acceptance evidence.
- `docs/HARDWARE.md` states a 1 kg maximum grasping weight. The charter tightens this
  project to pitcher plus water at or below 450 g.

### Reference motion, replay, export, and app path

- `tasks/jumper/common/dance/env.py::dance_env_cfg` is the direct motion-tracking pattern.
  Its `_reference_contract()` attaches `q`, `qd`, and `root_quat`; the action is not a
  residual, and `starts_on_entry` is true. Gesture tasks such as
  `tasks/jumper/gesture_hello/env_cfg.py::env_cfg` reuse that skeleton.
- `tasks/jumper/jump/mdp/actions.py::ReferenceResidualJointPositionAction` is the existing
  residual pattern. `_get_ref_q()` samples `q_cmd` one control interval ahead, then
  `apply_actions()` adds the bounded policy output to that baseline.
- `tasks/jumper/jump/mdp/reference.py::JumpReference.deploy_contract` describes a
  reference-guided export. It distinguishes the reached state `q` from controller targets
  `q_cmd`, records their separate rates, and emits `baseline="q_cmd"` for residual action.
- `scripts/export.py::_reference` writes `<name>.trajectory.json`, permutes every per-joint
  table into wire order, and returns the contract's `reference` block. It refuses missing
  joints. `_UNMEASURABLE_TERMS` rejects `base_lin_vel` for an actor because Jumper has no
  state estimator that supplies it.
- `deploy/fsm/src/trajectory.rs::Trajectory::parse` validates joint order, table shapes,
  rates, start semantics, and the `q_cmd` baseline. `Trajectory::baseline_into`,
  `q_at_into`, and `finished` perform the runtime sampling and end detection.
- `scripts/play.py::_run` and `_loop` are the canonical replay path. `--app` opens a bundle
  through `rl/mjrl/app_play.py::open_app`, sets the world rate with `step_the_world_at`, and
  drives the simulated servos from the bundled controller's final targets and gains.
- `scripts/play.py --headless --steps N --measure` is already a CPU-safe telemetry path.
  `rl/mjrl/viewer/monitor.py::JointMonitor` records control-step joint position, velocity,
  and `data.actuator_force`, then writes `measure.csv` and `measure.png`. That sampling is
  useful for overview plots but is too coarse to prove a limit applied every physics
  substep.
- `scripts/deploy.py` builds the app. `deploy/fsm/src/bundle.rs::Bundle::open` opens it;
  `scripts/deploy.py::validate_app` checks its schema; and
  `deploy/fsm/src/reference.rs::Reference::replay` checks recorded observations, targets,
  and mode transitions. Current `reference.json` generation reaches only the default mode,
  so it is not by itself evidence for a barista mode.
- `deploy/BUNDLE.md` requires a model and contract per mode, carries a named trajectory
  beside a reference-guided contract, and uses CPU ONNX for the `mjlab` host. A residual
  mode cannot run on the board with a stub model because zero residual would play the
  recording open loop.

### Servo limits

- `assets/jumper/motor/motor_config.yaml` is the source of truth for the 50:1 servo used by
  all 22 joints: 1.7464 N m plateau, 30.7352 rad/s corner, 25.1851 rad/s exponential decay,
  63.9838 rad/s cutoff, 300 ms peak duration, and 1.2 N m continuous torque.
- `tasks/jumper/common/actuator.py::servo_torque_limit` evaluates the speed curve.
  `ServoCurveActuator.control_law` clamps the PD result to it.
  `ServoCurveActuator.compute` additionally derates a hot joint, and `_integrate_heat`
  advances the per-joint normalized thermal state from applied torque at every physics
  substep. A thermal state of 1 is the derating trip point.
- `tasks/jumper/common/dance/rewards.py::torque_headroom_cost` penalizes applied torque above
  1.2 N m; `peak_actuator_force` reports the episode peak. These are useful policy signals,
  but neither replaces a time-resolved limit audit.

## Recommended architecture

### Default: deterministic reference with a zero residual

Use one barista task under `tasks/jumper/barista/` and keep scene construction under
`scenes/` and analysis/replay utilities under `barista/`. Generate the nominal motion as
Cartesian keyframes followed by offline IK and a plain-MuJoCo feasibility rollout. Save the
result as a task-local source recording, then expose a `reference_contract` with:

- `q`: reached joint state, for tracking and diagnostics;
- `q_cmd`: the position targets actually sent, for the actuator baseline;
- `qd` and `root_quat`: reference diagnostics;
- `residual_action: true`, `baseline: "q_cmd"`, and one-control-step
  `baseline_lead_s`;
- `starts_on_entry: true`, so video replay does not depend on live input timing.

The nominal actor should be a fixed zero-output ONNX produced by a small barista-owned export
helper, not by training. This preserves the existing app contract: the controller samples
`q_cmd`, adds zero, applies joint limits, and publishes through the normal app path. It also
makes failure of a missing model loud instead of relying on a forbidden residual stub.

Do not use the dance pattern as an open-loop joint driver: there the trajectory is an
observation target and the network generates all joint commands. Do reuse its recording
fields and end-of-motion hand-back. Do not add `base_lin_vel` to actor observations. If a
later residual is admitted, its actor inputs should be limited to deployable measurements:
projected gravity, base angular velocity, joint position/velocity/applied torque, previous
residual action, and reference phase/lookahead.

The claw needs one barista task hook, analogous to the five-foot hook, because the reference
alone cannot safely command through an unknown handle contact. Cache measured finger angle
and velocity in `before_observation`; in `after_decode`, apply the same squeeze rule as
`gripper.py::squeeze_limited` while the grasp phases request closure. Keep the finger target
open during approach and release. The hook must not weld the pitcher to the claw.

### Nominal motion phases

All durations are motion-design parameters to be finalized by measured replay, not results.
Use continuous interpolation with zero velocity at phase joins unless a join deliberately
continues the spiral.

1. **Stand and settle**: five support feet loaded, carried arm stowed, claw open, pitcher a
   free body on the counter.
2. **Pre-grasp**: move the claw to a collision-free pose around the handle without touching
   the pitcher or brewer.
3. **Close**: advance slowly until handle contact; hold with squeeze limiting rather than
   driving the finger to its hard stop.
4. **Lift**: raise the pitcher at least 50 mm, with a vertical first segment that avoids
   dragging it over the counter.
5. **M3 hold**: remain lifted for at least 3 s while the body and grasp settle.
6. **Transit**: move the spout above the coffee-bed centre, keeping the pitcher upright and
   the brewer collision-free.
7. **Pour-in**: rotate the pitcher about a handle/spout-aware pivot until the visual stream
   begins; do not use visual particles as force-bearing fluid.
8. **Spiral**: trace a slow outward 3-4 turn spiral with the *spout outlet*, not the wrist,
   20-50 mm above the bed. Keep the stream target inside the paper edge.
9. **Pour-out**: return upright, stop emission, and pause long enough to make cessation
   visible.
10. **Return and set down**: reverse through a collision-free transit and lower until the
    pitcher is supported by the counter.
11. **Release and retreat**: open the claw, verify the pitcher remains settled, then stow the
    arm. A short final hold gives the video an unambiguous end.

Treat pitcher plus represented water as one physical payload budget. Their combined designed
mass must be at most 0.450 kg even when the water is rendered only as massless particles: put
an equivalent water mass and inertia in the pitcher body so the robot still carries the load.
Never describe the particles as MuJoCo fluid or the result as hardware execution.

### Residual RL go/no-go gate and CPU budget

Do not train while the nominal trajectory fails any unperturbed M3/M4 check. First repair
keyframes, IK, collision geometry, timing, grasp pose, or gains. Admit a residual only if all
of the following are true:

1. the deterministic reference passes the nominal seed and torque audit;
2. a fixed perturbation matrix shows repeated failure from small, hardware-plausible initial
   pose, handle pose, payload, and friction variations;
3. the failure is balance/tracking error rather than geometry, contact planning, stream
   visualization, or set-down sequencing; and
4. the exact matrix, seeds, and before/after metrics are retained.

Bound the first residual experiment to one wall-clock shift (8 hours), `native:cpu`, at most
64 environments and 8 CPU threads, with an actor no larger than two 128-unit hidden layers.
Bound residual targets to at most 0.05 rad on the carried arm and 0.03 rad on support joints,
with the finger still owned by the squeeze hook. Stop if the residual does not improve the
predeclared perturbation pass rate without worsening nominal torque, collision, or landing
metrics. Increasing those limits, runtime, or compute is a separate decision; paid GPU use is
a human gate.

## Measurement and exit checks

The thresholds below are proposed acceptance definitions. They must be evaluated from saved
logs; this note does not claim they have passed.

### Common run record

Every acceptance run should record the commit, task and scene IDs, native backend/device,
seed, control and physics rates, exact pitcher dry mass, represented water mass, combined
mass, initial pitcher pose, and friction/contact settings. Record named phase intervals so a
failure can be assigned to grasp, lift, hold, spiral, or set-down rather than only to a run.

Use scene sites with stable names for the pitcher handle centre, spout outlet, coffee-bed
centre, paper-edge radius, counter support plane, and claw mouth. Derive metrics from those
sites and body poses in simulation, not from rendered pixels.

### M3: grasp and lift

A run passes M3 only when all checks hold in the same replay:

- combined pitcher and represented-water mass is at most 0.450 kg;
- the pitcher begins supported by the counter as a free body, not spawned in or welded to
  the claw;
- opposing jaw/handle contacts occur before lift and no non-claw robot geom contacts the
  pitcher;
- the pitcher rises at least 0.050 m above its settled pre-grasp height and remains at or
  above that height continuously for at least 3.0 s;
- during that hold, the grasp does not slip more than 0.010 m relative to the claw and the
  pitcher does not contact the counter, brewer, cup, or free legs;
- base roll and pitch each remain within 5 degrees of the settled start, base XY drift is at
  most 0.010 m, and every designated support foot remains in contact;
- no joint-limit or self-collision event occurs; and
- the torque audit below passes.

Report the minimum lift, continuous hold duration, maximum grasp-relative translation and
rotation, maximum base tilt/drift, support-contact minimum, and the torque summary. A render
alone is not exit evidence.

### M4: spiral, stream landing, and set-down

M4 includes every M3 constraint through the carried portion and additionally requires:

- a pour interval of 15-25 s with spout height continuously between 0.020 and 0.050 m above
  the coffee-bed plane;
- unwrapped polar angle of valid stream-plane intersections advances by at least 6 pi radians
  (3 turns); the target is 3-4 turns, not more than 4;
- spiral radius starts within 0.010 m of bed centre, grows outward overall, and every valid
  landing remains inside the configured paper-edge radius with at least 0.005 m margin;
- at least 95% of emitted particle/visual-stream samples intersect the bed acceptance disk;
  missed/no-intersection samples count as failures, not as absent data;
- no pitcher, claw, leg, or particle emitter geom contacts the brewer or cup;
- after pour-out there are no emitted samples, the pitcher returns to the counter, and it is
  released only after support contact;
- for a final 2.0 s settle window, pitcher linear speed is at most 0.010 m/s, angular speed
  at most 0.10 rad/s, and its position is within 0.020 m of the declared set-down target;
- the claw retreats without moving the settled pitcher more than 0.005 m; and
- the torque audit passes for the whole run, including return and set-down.

The stream metric is explicitly a particle/visual approximation. Save its samples as data;
do not infer landing quality from a beauty render.

### Torque log schema and windowing

Add a barista-owned substep logger; do not modify `JointMonitor` for M3/M4-specific fields.
Write one CSV row per physics substep and actuator with these columns:

| column | unit or type |
|---|---|
| `run_id`, `phase`, `joint` | string |
| `substep`, `control_step` | integer |
| `time_s`, `dt_s` | s |
| `q_rad`, `qd_rad_s` | rad, rad/s |
| `tau_raw_nm`, `tau_applied_nm` | N m |
| `curve_limit_nm`, `continuous_limit_nm`, `effective_limit_nm` | N m |
| `thermal_theta` | normalized; 1 is the derating trip |
| `curve_margin_nm`, `effective_margin_nm` | limit minus absolute applied torque, N m |
| `above_continuous`, `derating_active`, `curve_violation` | boolean |
| `above_continuous_window_id`, `above_continuous_elapsed_s` | integer, s |

Also write a JSON summary per joint and phase: sample count; peak speed; peak raw/applied
absolute torque; minimum curve/effective margin; maximum thermal state; count and total time
above 1.2 N m; maximum contiguous time above 1.2 N m; derated-substep count; and first
violation time.

Sample at the physics substep because `ServoCurveActuator.compute` clamps and integrates heat
there. Compute the curve limit from the logged speed with the exact YAML parameters. Recompute
thermal state from applied torque using the actuator recurrence and compare it with the logged
state; use an absolute numerical tolerance of 1e-4 N m for torque margins and 1e-5 for thermal
state. Pass only when, for every joint:

1. no applied sample exceeds the speed-dependent curve or effective derated limit by more
   than 1e-4 N m;
2. no contiguous interval above 1.2 N m exceeds 0.300 s;
3. `thermal_theta` remains below 1.0, so the motion did not consume the modelled peak budget
   to the derating point; and
4. the recomputed and actuator thermal states agree within 1e-5.

Keep phase boundaries out of the window reset: an over-continuous interval that starts during
lift and continues into hold is one interval. Report both whole-run and per-phase summaries.
The conservative 300 ms contiguous check is intentional; the thermal recurrence remains the
more complete partial-duty-cycle check.

## Headless native-CPU verification plan

Run from the repository environment, without a viewer or foreground application:

1. Resolve the task with `python scripts/train.py --task jumper.barista --backend native
   --device cpu --dry-run` and verify the printed backend is `native:cpu`.
2. Run the deterministic actor for the full routine with `python scripts/play.py --task
   jumper.barista --backend native --device cpu --headless --no-obs-noise --steps <full-run>
   --measure`. Run the barista substep and object/stream logger in the same replay.
3. Repeat the declared nominal seeds and perturbation matrix. Aggregate pass/fail by script,
   never by manual transcription.
4. Render selected frames and the final video offscreen from saved state. On macOS use the
   offscreen MuJoCo/CGL path; do not launch `mjpython` or a live viewer. Inspect at least
   pre-grasp, secured lift, centre pour, outer spiral, set-down, and release frames.
5. Export the constant-zero nominal actor and reference with `scripts/export.py`-compatible
   layout, then build a local app with the repository bundler. No push or upload.
6. Run `python scripts/deploy.py --check-reference <bundle>` and then replay the app headless
   with `python scripts/play.py --app <bundle-or-app> --task jumper.barista --backend native
   --device cpu --headless --steps <full-run> --measure` so the bundled controller, hook,
   reference baseline, limits, and ONNX are the path under test.
7. Compare task replay and app replay phase metrics and final object poses. Archive commands,
   stdout, torque CSV/JSON, object/stream metrics, frames/video, and exact commit SHA.

`reference.json` parity is necessary but not sufficient: its current synthetic frames do not
enter non-default modes, and it does not exercise pitcher contact or the whole trajectory.

## File and API ownership

To reduce conflicts with upstream and parallel scene work:

- `scenes/<barista scene package>/`: scene geometry, stable measurement sites, masses, and
  initial object poses. It owns no motion timing.
- `tasks/jumper/barista/`: task registration/config, task-local reference source, residual
  action wiring, and `deploy/lib.rs` for claw squeeze limiting. Reuse common actuator and
  trajectory contracts; do not add barista tuning to `tasks/jumper/common/`.
- `barista/trajectory/`: keyframe/IK generation and its declarative phase file.
- `barista/verification/`: substep torque logger, object/contact/stream metrics, aggregation,
  and headless offscreen renderer. These consume stable scene site names.
- `barista/artifacts/` or an ignored run directory: generated CSV, JSON, frames, and video;
  do not commit measured outputs unless a later card explicitly selects them as evidence.
- `scripts/`, `rl/mjlab/`, `rl/rsl_rl/`, and shared controller files: no barista-specific
  edits. If a genuinely generic missing API is discovered, raise a separate card rather
  than expanding M3/M4 into an upstream refactor.

The narrow shared interfaces are therefore: scene site/body names; a task-local phase schema;
the existing `reference_contract`; and the existing `ModeHook` state-in/command-out boundary.
Third-party assets remain subject to licence evidence in `barista/SOURCES.md`.