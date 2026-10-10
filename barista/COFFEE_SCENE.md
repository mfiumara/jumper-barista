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

## Open risks after `pour_out` (for t_27a0d8ff / t_41ccaaa7)

The M4 reachability fast check (`run_m4.py --fast-check --reachability-check`) stops at the end of
`pour_out`; its contact, clearance, support and base-drift gates cover `transit`, `pour_in`,
`spiral` and `pour_out` only, never the whole run. In the full replay at 74c66cf (reviewed in
t_5d3954d9), after `pour_out` the non-colliding spout reached -6.75 mm clearance into `v60_rib_10`
(negative from `lower` through `final_settle`, deepest in `clear_retreat`), support contacts fell to
1-2, and base drift reached 21.8 mm. Set-down and the post-pour retreat still use the old layout;
these remain OPEN and are not accepted by the fast check.
