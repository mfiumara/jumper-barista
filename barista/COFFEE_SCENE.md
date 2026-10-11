# Coffee-bar scene design

`scenes/coffee_bar.scene/` is an original, project-owned MuJoCo scene package. It uses only
MJCF primitive geometry and built-in textures; there are no downloaded meshes, textures, or
other third-party scene assets.

The counter surface is the local `z = 0` datum. The package attachment slot remains centred at
the origin and faces `+X`; the reachability replay starts Jumper at `(-0.018725, 0.095488)` m.
The free pitcher starts at
`(0.168255, 0.190982, 0.001)` m so the carried claw can approach the rear-handle grip. The V60 is
fixed into the server at `(0.0934, 0.32)` m. The nominal 0.50 m standing marker overlaps that
fixed-prop envelope, so the collidable native:cpu replay—not the marker—is the clearance authority.
Named sites mark the standing slot, forward direction, pitcher grip, spout tip, and coffee-bed target.

## Dimensions and design masses

| Object | Principal dimensions | Design mass | Simulation treatment |
|---|---:|---:|---|
| Counter | 1.20 × 0.80 × 0.04 m | 18.00 kg | Static slab; top at `z = 0` |
| Cup/server | 0.090 m diameter × 0.080 m high | 0.250 kg | Static |
| V60 | 0.076 m maximum diameter × 0.090 m high; 0.024 m outlet | 0.120 kg | Static open 12-rib cone, segmented rim, and support plate on server |
| Pitcher vessel | 0.056 m body diameter × 0.080 m high; 0.129 m spout reach | 0.240 kg | Free rigid body; collidable vessel and steel handle (handle mass lumped into the vessel), visual-only spout |
| Rear-handle grip add-on | Closed collar around the 0.010 m upper handle rod (0.012 m long) and 0.008 m square post to a lower shelf; two 0.008 m posts carry a 0.064 m bridge with a 0.036 m square TPU sleeve and 0.048 m end stops | 0.0252 kg | 3D-printed PETG piece clamped to the collidable handle; ordinary contact geometry |
| Nominal water | 80 mL at 1000 kg/m³ | 0.080 kg | Rigid interior mass proxy; not fluid |
| Pitcher + add-on + nominal water | — | **0.3452 kg** | 0.1048 kg below the 0.450 kg design limit |

The machine-readable source is `barista/coffee_scene_dimensions.json`. In the MJCF, the vessel
carries 0.240 kg, the rear-handle grip add-on carries 0.0252 kg, and an invisible interior cylinder
carries 0.080 kg, so the compiled free body has the complete 0.3452 kg payload mass. The add-on's
load path is the steel handle: its closed collar's jaws and side walls touch the collidable
`handle_upper` rod with zero gap and no penetration, a post rises from the collar into the shelf,
and the bridge stands on two posts on that shelf. `barista/verification/grip_attachment.py` (also
`tests/test_barista_grip_attachment.py`) fails if any piece is neither seated on collidable pitcher
geometry nor joined to a seated piece within 1.0 mm, or penetrates pitcher geometry by more than
0.5 mm. Because MuJoCo welds every geom to the rigid pitcher body, clamp preload and retention are
not simulated. The grip-bar
centre is 0.075 m behind the vessel axis, 0.0699 m from the compiled loaded centre of mass in the
pitcher's XY plane; the replay
therefore audits the resulting offset load against the servo curve and thermal budget. The visible
water surface and any later stream are visual approximations only. The V60 collision shell is physically present: twelve
sloping ribs and twelve rim segments leave its centre open rather than filling the cone with
stacked cylinders. This scene does not represent a physical-robot trial.

## Loading and package boundary

The package is registered as `coffee_bar`; list or replay it with the repository's normal scene
interface:

```bash
python scripts/train.py --list
python scripts/play.py --task jumper.gesture_hello --scene coffee_bar \
  --checkpoint tasks/jumper/gesture_hello/out/example/model_999.pt \
  --backend native --device cpu --headless --steps 100 --speed 0 --no-obs-noise
python barista/render_coffee_scene.py --out <output-directory>
```

The evidence renderer uses `jumper.flat`'s home action and pins its normally randomised planar
reset to the authored standing slot before a one-second settle; it does not alter the task or the
scene package.

The current Design workflow was consulted at commit
`918d192857fcea1678c2a40b339501004a94d838`. New Design exports use
`kk-scene-package/2` and require a verified complete default `.skin/3`, a reviewed robot-free
preview, a trusted Jumper profile, and Design's `export-map` plus `verify-package --mujoco` flow.
This accepted scene is a Train-owned `kk-scene-package/1` made from original MJCF primitives; it
does not have those Design inputs and must not be renamed or presented as a verified `.map/2`.
It therefore remains a hash-checked, replay-only MJCF package. A future `.map` export belongs in
the independent Design repository and must record the exact Design and default-skin inputs used.

## Spiral envelope and set-down path

The M4 keyframes are corrected against the measured actual spout, not a prescribed path. In this
standing layout the tilted pitcher's rear meets the robot's upper shell/display, and the arm nears
full extension, when the spout moves toward the far (+y, about 70°) side of the bed. A centred
spiral there is limited to roughly 16 mm at a 24 mm spout height. The pour is therefore a centred,
counter-clockwise, 3.5-turn spiral from 3.5 mm to 24 mm. Its radius is compressed by up to 36%
toward 70°, the spout is held 24-27 mm above the bed at a 50° pour tilt, and the spiral ends
laterally at 320°. `pour_in` passes over the V60 rim through a via point. `pour_out` backs
radially away and rights the pitcher to about 5°.

After pouring, the pitcher returns to its pickup spot (`set_down_target_m`). The arm goes back to
the grasp pose while the five-support body lowers (44.5-47.5 s), so the contact-triggered release
happens in `lower`/`set_down`. Once the claw is open, `clear_retreat` blends from the arm pose
held at contact (not the nominal release keyframe, which used to step the LF_J2/J3 targets and
shove the released pitcher). It lifts the palm straight up with J1 only, with no yaw, so the palm
separates without dragging the pitcher. The claw rises clear and closes only after that.
`barista/verification/m4_spout_metrics.py` recomputes the spiral, rotation, stream, per-phase
contact/body/clearance, set-down, payload and per-joint torque tables from a replay's raw logs.

## Stream and collision gates (simulation only)

The water stream is a massless visual approximation, not fluid. Every 20 ms of the spiral it is
launched from the measured `pitcher_spout_tip` pose and its measured linear velocity, then
follows a gravity-only ballistic path to the bed plane. The rendered stream is the same logged
path. `run_m4.py` gates:

- horizontal span of at most 6 mm and horizontal launch speed of at most 0.2 m/s;
- a valid bed intersection, with the path ray-cast against all scene and robot geometry. A
  control ray straight down must hit the brewer, which proves the check can see it;
- at least 95% of landings within 33 mm (the 38 mm acceptance radius less a 5 mm edge margin).

Collision gates run over the whole replay. They cover robot-brewer contact, free-leg contact
(the LF claw limb may touch only the pitcher), any robot contact with scene geometry other than
the counter, and pitcher-brewer, non-claw, self and joint-limit contacts. Measured signed
clearance is taken from every pitcher geom, including the visual-only spout, and from all 48
robot collision geoms to the brewer. An integrity gate fails if contact is disabled, if any
brewer geom is non-collidable, if the filters stop robot or pitcher geoms colliding with the
brewer, if a brewer contact exclusion exists, or if the pitcher's non-collidable set grows beyond
its visual/mass-only spout, water surface and water-mass proxy.

The fast check (`run_m4.py --fast-check`) stops at the end of `pour_out`. Set-down, release and
retreat evidence needs the full verifier.

## M4R margins: 1 kHz physics and a walked stance (simulation only)

A4 chains M4 after a walk, so M4 now runs at **1 kHz physics with 50 Hz keyframe targets**
(`physics_dt_s` 0.001, decimation derived from the trajectory) and must survive an arrival a
few millimetres or degrees off `base_start`. At 1 kHz the accepted pour passed with almost no
margin: carried grasp translation 9.71 mm (limit 10), set-down 19.9 mm (limit 20), inward
excursion 7.98 mm (limit 10), and the pitcher grip came within a few millimetres of the
display module in the last turn. No gate threshold or definition changed; what changed:

- **Stance re-targeting** (`barista/verification/m4_retarget.py`). The planar offset `D` of
  `base_link` from `base_start` is measured once at the first tick. Every 20 ms a damped
  least-squares IK on the four LF arm joints puts the measured spout outlet (palm when nothing
  is held) at `D^-1` of the keyframe's pose, so the spiral stays centred on the bed in the
  station's frame. At `base_start` the stance correction is inactive; the spiral-radius and
  set-down corrections still apply.
- **Shorter carry.** The grip creeps slowly and steadily under the carried load (soft-contact
  friction), so carried time is grasp translation: transit 4.0 -> 2.5 s and the same spiral
  keyframes time-scaled 17.5 -> 16.0 s (still >= 15 s of continuous spiral for the video).
- **Grip preload.** The claw's squeeze lead in the replay is 0.11 rad instead of 0.06
  (`grip_squeeze_lead_rad`): 1.1 N·m of finger preload instead of 0.6, inside the servo's
  1.2 N·m continuous rating. The whole-run torque audit still gates it (LF_J4 thermal state is
  reported). Friction coefficients are unchanged.
- **Spiral radius 0.85x** about the bed centre (`spiral_radius_scale`), applied to the measured
  spout target by the same IK and eased back over the first 0.5 s of `pour_out`. It moves the
  outer far-side turn away from the head and lowers the inward excursion of the compressed
  sector; radial progress stays above its 15 mm gate.
- **Set-down servo** (`set_down_servo`): during `lower` the measured pitcher origin is servoed
  in x/y onto the pickup spot (`set_down_target_m`) with a small clamped integral, so the grip
  creep no longer becomes set-down error. The release and retreat keep that x/y shift, so the
  open claw leaves along the accepted path relative to where the pitcher actually stands.
- **Reported clearance.** `summary.json` `motion.pitcher_robot_clearance` gives the minimum
  distance between the carried pitcher (including its visual spout) and every robot collision
  geom except the LF claw, per carried phase, confirmed on a 0.5 mm distmax ladder so a
  spurious solver zero reads as a lower bound rather than a capped 3 or 10 mm.
- **Memory.** The 1 kHz torque audit (about 1.2 M rows) is streamed to CSV with a running
  summary identical to the in-memory one, so a 13-point sweep fits in RAM.

Robustness sweep (each point an independent `run_m4.py --spawn-offset DX DY DYAW`, unmodified
gates): yaw -2..+2 deg in 0.5 deg steps plus dx/dy +-2 mm.

```bash
python barista/verification/run_m4.py --out out/barista/m4 --save-render-states
python barista/verification/m4_sweep.py --out out/barista/m4_sweep --jobs 2
```

The measured margins and the per-point table for the exact commit are in the M4R card's
evidence comment; they are simulation results with approximate (visual) water only.

### Release/retreat continuation and stricter clearance reporting

The 2 s clear-retreat and 2 s retreat disturbed the released pitcher through contact with
`LF_palm_link_meshcol1/2`. At yaw +0.5 degrees and dx +2 mm its speed in `final_settle`
reached 0.082 and 0.085 m/s respectively. Doubling both retreat segments to 4 s, without
changing their endpoints or any gate, reduced those final-settle maxima to 0.00000370 and
0.00000275 m/s in the targeted replays. The final hold remains 3 s, now at 54.5–57.5 s.
This is a slower retreat, not a collision-free release: palm contact and pitcher disturbance
still occur before the final hold. The 16 s spiral is unchanged.

The earlier 6.67 mm nominal clearance excluded the whole claw and is **not** evidence for
the stricter all-non-grasp-surface requirement. The independent
`motion.pitcher_robot_pair_clearance` report samples every spiral physics substep and gives
both the unrestricted minimum and the minimum excluding only these opposing intentional
grasp pairs (robot geom, pitcher geom):

| Robot geom | Pitcher geom |
|---|---|
| `robot/LF_finger_grip_insert_link_meshcol` | `bar_top_grip_collar_a` |
| `robot/LF_palm_grip_insert_link_meshcol` | `bar_top_grip_collar_b` |

Every other claw surface and pitcher part remains in that measurement, as does the display.
Actual contact distances contribute to the unrestricted and non-exempt minima, preserving
penetration rather than treating a distance-solver confirmation as positive clearance.
The old non-claw metric remains for comparison; neither metric changes the run_m4 gates.

The full-suite baseline at `e6765ea` failed deploy-source provenance, posture mirroring and
one tensorboard port test. The continuation before the pair-metric addition had 1036 passed,
25 skipped and the same first two failures; whole-repository ruff had 348 errors both before
and after. These are unmet full-green acceptance criteria, not a waiver. No third-party
assets were added, and no hardware run is implied.
