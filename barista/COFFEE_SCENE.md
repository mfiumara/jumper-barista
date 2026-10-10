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
