# Coffee-bar scene design

`scenes/coffee_bar.scene/` is an original, project-owned MuJoCo scene package. It uses only
MJCF primitive geometry and built-in textures; there are no downloaded meshes, textures, or
other third-party scene assets.

The counter surface is the local `z = 0` datum. Jumper's standing slot is centred at the origin,
faces `+X`, and has a clear diameter of 0.50 m. The nearest prop surface is 0.299 m from the
slot centre, leaving 0.049 m beyond the marked clear boundary. The pitcher begins upright on the
counter at `(0.30, -0.18, 0.001)` m, separate from the robot. The V60 is fixed directly above the
server at `(0.32, 0.14)` m. Named sites mark the standing slot, forward direction, pitcher grip,
spout tip, and coffee-bed target for later motion work.

## Dimensions and design masses

| Object | Principal dimensions | Design mass | Simulation treatment |
|---|---:|---:|---|
| Counter | 1.20 × 0.80 × 0.04 m | 18.00 kg | Static slab; top at `z = 0` |
| Cup/server | 0.090 m diameter × 0.080 m high | 0.250 kg | Static |
| V60 | 0.100 m maximum diameter × 0.090 m high | 0.120 kg | Static stepped-cone proxy on server |
| Pitcher, dry | 0.056 m body diameter × 0.080 m high; 0.129 m spout reach | 0.240 kg | Free rigid body |
| Nominal water | 180 mL at 1000 kg/m³ | 0.180 kg | Rigid interior mass proxy; not fluid |
| Pitcher + nominal water | — | **0.420 kg** | 0.030 kg below the 0.450 kg design limit |

The machine-readable source is `barista/coffee_scene_dimensions.json`. In the MJCF, the pitcher
body carries 0.240 kg and an invisible interior cylinder carries 0.180 kg, so the compiled free
body has the complete 0.420 kg payload mass. The visible water surface and any later stream are
visual approximations only. This scene does not represent a physical-robot trial.
