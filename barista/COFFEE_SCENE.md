# Coffee-bar scene design

`scenes/coffee_bar.scene/` is an original, project-owned MuJoCo scene package. It uses only
MJCF primitive geometry and built-in textures; there are no downloaded meshes, textures, or
other third-party scene assets.

The counter surface is the local `z = 0` datum. Jumper's standing slot is centred at the origin
and faces `+X`. The nearest fixed prop surface is 0.362 m from the slot centre, leaving 0.112 m
beyond the marked 0.25 m radius. The free pitcher is deliberately inside that boundary at
`(0.18698, 0.02047, 0.001)` m so the carried claw can approach it; it is not part of the fixed
clearance claim. The V60 is fixed directly above the server at `(0.38, 0.16)` m. Named sites mark
the standing slot, forward direction, centred pitcher grip, spout tip, and coffee-bed target.

## Dimensions and design masses

| Object | Principal dimensions | Design mass | Simulation treatment |
|---|---:|---:|---|
| Counter | 1.20 × 0.80 × 0.04 m | 18.00 kg | Static slab; top at `z = 0` |
| Cup/server | 0.090 m diameter × 0.080 m high | 0.250 kg | Static |
| V60 | 0.100 m maximum diameter × 0.090 m high | 0.120 kg | Static stepped-cone proxy on server |
| Pitcher vessel | 0.056 m body diameter × 0.080 m high; 0.129 m spout reach | 0.240 kg | Free rigid body |
| Centred top-grip add-on | 0.064 m overall bar × 0.024 m diameter; two 0.008 m struts | 0.024 kg | Hollow 3D-printed PETG bridge; ordinary contact geometry |
| Nominal water | 180 mL at 1000 kg/m³ | 0.180 kg | Rigid interior mass proxy; not fluid |
| Pitcher + add-on + nominal water | — | **0.444 kg** | 0.006 kg below the 0.450 kg design limit |

The machine-readable source is `barista/coffee_scene_dimensions.json`. In the MJCF, the vessel
carries 0.240 kg, the symmetric grip bridge carries 0.024 kg, and an invisible interior cylinder
carries 0.180 kg, so the compiled free body has the complete 0.444 kg payload mass. The grip-bar
centre is directly over the loaded body's centre of mass in the pitcher's XY plane, allowing the
pitcher to hang level while the claw pinches the bar ends. The visible water surface and any later
stream are visual approximations only. This scene does not represent a physical-robot trial.

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
