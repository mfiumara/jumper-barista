# Jumper Barista outfit: concepts (A2a, design gate)

These are three outfit concepts for Jumper as an affectionate Italian hipster barista
(`barista/STORY.md`, "Character" and "Outfit"). **They are concept mock-ups, not a model.**
Each image shows the real Jumper model at the accepted M4 pour pose, with rough primitive
accessories drawn into the renderer's scene and garments shown as render-time recolours of
existing visual geoms. Nothing is added to the compiled model: no outfit geometry, `.skin`,
mesh or `visual_overrides` file exists yet. That is A2b, and it starts only after
**Mattia picks one** (the jumper-design appearance-selection gate). Silence or this note's
ordering is not a pick.

Everything here is a MuJoCo simulation. The pose comes from the accepted M4 replay
(`cdae494`), which has not been run on the real robot.

## Reproduce

```sh
# 1. The accepted M4 replay with render states (skip if you already have it):
python barista/verification/run_m4.py --out <replay_dir> --save-render-states
# 2. Concept renders, contact sheet, clearance sweep (11 min on an idle M5 Pro, 23 min under load):
python barista/outfit/concepts.py --replay <replay_dir> --out <out_dir>
#    add --skip-clearance for renders only (about 30 s)
```

`concepts.py` refuses a replay whose `qpos` hash does not match its `summary.json` or that
did not pass M4, using the same check as `barista/film/make_film.py`. It writes
`concept_{A,B,C}_{threequarter,side}.png`, `concepts_contact_sheet.png` and
`concepts_manifest.json` (PNG SHA-256s, palettes, parts, clearance numbers). The images are
deliverables and are not committed.

Design-workflow reference: jumper-design `918d192857fcea1678c2a40b339501004a94d838`
(`AGENTS.md`, "Required design selection before modeling"; `docs/workflow.md`,
"Appearance selection gate"). It was read only; no jumper-design tool or asset was used.

## The three concepts

Coordinates are in the `upper_shell_link` frame (+x towards the face, +y robot-left, +z up).

### Concept A: Il Classico Milanese

This is the neighbourhood barista who has pulled shots since 1987. A low tweed coppola (flat
cap) sits on the back of the dome, and its short brim stops well behind the eye panel. A black
handlebar moustache with curled tips is a flush decal under the sensor "nose". The red
upper shell becomes a charcoal waistcoat with two rows of brass buttons along its rim, and
the lower body reads as a cream shirt. The legs and claws stay stock red, so it is still
clearly Jumper.

- Palette: tweed cap `#5C4533`, waistcoat charcoal `#2B2B30`, shirt cream `#EDE3CC`, brass
  buttons `#CCA34C`, moustache espresso-black `#120D0A`. The legs and claws stay stock red
  and white.
- Parts: cap crown (ellipsoid) + brim (flat ellipsoid) + top button; moustache decal (7
  strokes); 6 brass buttons on the shell rim (3 per side); recolours: `upper_shell_link`
  visual → charcoal, `base_link` visual → cream.

### Concept B: Boheme del Caffe

This is the third-wave hipster who talks about the farm while he pours. A soft bordeaux beret
is pulled to one side of the dome. Round gold-wire glasses are pushed up onto its front edge,
so they sit above the eye panel and never cover it. A full chestnut chevron moustache is a
flush decal under the nose. The lower body becomes a forest-green waist apron tied at the
back with a cream bow. The red shell, legs and claws stay stock.

- Palette: beret bordeaux `#6B121F`, glasses gold wire `#C79E40`, moustache chestnut
  `#4C2B17`, apron forest green `#1F5438`, apron ties cream `#EDE3CC`. The shell stays stock
  red `#D9140D`.
- Parts: beret (tilted ellipsoid) + stalk; glasses: two 14 mm wire rings, a bridge and two
  temples resting on the beret; moustache decal (2 strokes); apron bow at the rear: knot,
  2 loops, 2 tails; recolour: `base_link` visual → forest green.

### Concept C: Dapper Tricolore

This is the dapper espresso-bar host from a 1960s Roman film. A narrow-brim straw trilby with a
thin green-white-red band gives the tallest silhouette, and a neat pencil-moustache decal
keeps the face light. A cream bar towel with a red stripe is folded over the back of the
shell, and the lower body becomes a crisp white barista jacket. The red shell, legs and claws
stay stock, so the tricolore comes from the robot itself.

- Palette: straw `#DBBD80`, band verde `#008C45`, bianco `#F5F5F0`, rosso `#CC1C2B`, towel
  cream `#EDE3CC`, jacket white `#F5F5F0`, moustache espresso-black `#120D0A`. The shell stays
  stock red.
- Parts: hat brim (thin cylinder), crown (cylinder + flattened top), three band rings;
  pencil-moustache decal (2 strokes); towel (top fold, back drop, red stripe) on the
  rear-right of the shell; recolour: `base_link` visual → white.

## Clearance

The claw/leg envelope is checked as follows. All accessories ride on `upper_shell_link`, so
the only things that can reach them are parts that move relative to the shell: the two front
claws (LF/RF chains), the four legs and the pitcher. `concepts.py` measures the distance from
every accessory to those parts:

- at the **grasp pose** (M4 t = 4.0 s, claw closed on the handle) and the **pour pose**
  (M4 t = 30.0 s, mid-spiral);
- over the **whole M4 replay** (approach, grasp, lift, carry, spiral, pour-out, set-down,
  release, retreat) at 25 Hz;
- over the reference motions of **all five dance clips** (`dance` demo, `brazilian`,
  `dream_wings`, `maze`, `waist`) at 25 Hz;
- over the **flat jump** reference `high_jump_flat` (crouch, take-off, tuck, landing).

The pitcher is checked only in M4. The dance and jump sweeps are kinematic replays of the
reference joint angles, not policy rollouts. Distances are vertex-sampled: meshes are
voxelised at 2.5 mm and primitives surface-sampled at 2.5 mm, so each number is an
estimate at about that resolution. In the sweeps, distances beyond 20 mm are reported only as
"> 20 mm".

The rule has two parts:

- **Proud parts** (hats, glasses, buttons, bow, towel) must keep ≥ 10 mm from the claws,
  legs and pitcher on every swept sample, and stay out of the eye-panel box in front of
  `display_module_link`. The A3 eye panel stays free in every concept.
- **Moustache decals** stand at most 0.8 mm proud of the face. Their sweep minimum is
  reported, not gated, because the robot's own claw already grazes the bare face there.

### Results (from `concepts_manifest.json`)

The sweep shows that the left claw already passes within about 0.1 mm of the robot's own
face under the sensors during M4 set-down and retreat (t ≈ 47–53 s). In the `maze` dance it
comes within 6.5 mm of the face, and in the other dances 12.7–18.8 mm. **Nothing proud can
sit on the lower face.** That is why every moustache here is a flush decal, not a 3D part.

| | A Il Classico | B Boheme | C Tricolore |
|---|---|---|---|
| Proud parts vs claws, all sweeps | 16.6 mm (cap brim, `brazilian` dance) | 12.0 mm (glasses rim, `brazilian` dance) | > 20 mm |
| Proud parts vs legs, all sweeps | > 20 mm | > 20 mm | 19.7 mm (towel, jump tuck) |
| Proud parts vs pitcher, M4 | > 20 mm | > 20 mm | > 20 mm |
| Proud parts at grasp pose: claws / legs / pitcher | 27.0 / 25.2 / > 100 mm | 45.3 / 29.9 / > 100 mm | 54.8 / 23.8 / > 100 mm |
| Proud parts at pour pose: claws / legs / pitcher | 27.0 / 25.2 / 51.6 mm | 26.3 / 29.9 / 38.7 mm | 54.8 / 23.8 / 76.4 mm |
| Moustache decal vs claws, all sweeps | 0.1 mm (M4 retreat) | 0.1 mm (M4 retreat) | 0.1 mm (M4 retreat) |
| Moustache decal at grasp / pour (claws) | 34.0 / 34.9 mm | 35.4 / 39.5 mm | 36.4 / 41.7 mm |
| Eye-panel box intrusions | none | none | none |
| Proud-part rule (≥ 10 mm) | pass | pass | pass |

### Per-concept clearance note

- **A:** The cap sits behind the shell's highest point. Its brim ends at x ≈ 56 mm, behind
  the eye panel (x ≥ 73 mm). The closest any claw gets is 16.6 mm, to the brim, when an arm
  is raised in the `brazilian` dance. The pitcher and the pouring claw pass ahead of and to
  the left of the cap, never within 20 mm. The brass buttons sit on the upper shell rim,
  above the hips (z = +18 mm vs the hip joints at −21 to −24 mm), and no leg comes within 20 mm in
  any dance or the jump. Everything is rigid to the shell, so nothing swings: on the jump
  landing the whole outfit moves with the shell.
- **B:** The beret is low and set back. The glasses are the most exposed proud part: they
  come within 12.0 mm of a raised claw in the `brazilian` dance, the smallest proud margin
  of all three concepts. In M4, B's proud parts stay at least 17.1 mm from the claws. If B is
  picked, A2b should keep the glasses no further forward or higher than shown, or shrink
  the rims. The apron is a recolour, and its bow sits on the rear face of the body, more than
  20 mm from every leg in all dances and the jump. There are no loose apron strings: the
  short tails are rigid and stay on the rear face, within 10 mm of the centre line, between
  the rear legs.
- **C:** The hat is the tallest part (crown top at z ≈ 81 mm), but it is set back on the dome
  and no claw comes within 20 mm, even at the top of the pour. The towel drapes over the
  rear-right of the shell. Its lower edge comes within 19.7 mm of a leg at the jump tuck
  (t = 0.8 s, airborne, legs folded), which is the tightest leg margin of the three
  concepts. It must stay rigid, with no cloth simulation and no hanging corner, and must
  not be lengthened.

### What this means for A2b (after the pick)

- Build the chosen accessories as display-only visual geoms/meshes on `upper_shell_link`
  (`contype=0`, `conaffinity=0`, no mass or inertia), so the accepted M4 replay's motion and
  torque hashes stay byte-identical. Then re-run the sweep above on the real meshes.
- Make the moustache a zero-thickness texture or colour on the face (a `.skin/3`
  `visual_overrides`-style recolour region), or ≤ 0.3 mm proud. At 0.8 mm the decal would
  visibly clip the left claw for a moment during the M4 retreat (t ≈ 47–53 s). A3 could
  also draw it on the eye display, but that is A3's call.
- Every part stays rigid. Nothing hangs, swings or uses cloth, so the dance and the
  in-place jump cannot fling a loose part into a leg.

## Licences

No third-party asset is used. All geometry is project-authored primitives. The label font
is Pillow's bundled Aileron Regular (CC0), which is already recorded in `barista/SOURCES.md`.
