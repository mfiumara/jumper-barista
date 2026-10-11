![Jumper](docs/media/jumper-hero-en.png)

# Jumper

**English** | [简体中文](README.zh.md)

**Jumper is a 22-DoF crab robot.** [View hardware →](docs/HARDWARE.md)

This repo is Jumper’s AI toolkit for appearance, motion, and scene creation. Open this repo in an AI coding assistant and try the prompts below.

> 🦀 **Get a free Jumper!** [Find out how →](https://beunlimited.me/zh/events/crab-robot-challenge-2026)

## One sentence to design an appearance

> Design a warm sand ranger appearance for Jumper with coordinated body and limb colors, then export a `.skin`.

| | | |
|:-:|:-:|:-:|
| ![Warm sand ranger appearance](docs/media/design-warm-sand.png) | ![Silver armor appearance](docs/media/design-silver-armor.png) | ![Raphael Turtle appearance](docs/media/design-raphael.png) |
| [**Warm sand ranger**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/warm-sand-ranger-integrated-v2.skin) | [**Silver armor**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/mecha-tripo-v3.skin) | [**Raphael Turtle**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/raphael-turtle-v1.skin) |

[Browse all skins](https://github.com/KingKongRobotics/jumper-design/tree/main/library/skins)

## One sentence to train a motion

> Train a stable tripod gait for Jumper, replay and evaluate the result, then package it as an `.app`.

| | | |
|:-:|:-:|:-:|
| ![Jumper walking](docs/media/walk.gif) | ![Jumper changing posture](docs/media/posture.gif) | ![Jumper waving](docs/media/gesture.gif) |
| **Walk** | **Posture** | **Gesture** |
| ![Dance simulation](docs/media/dance.gif)<br>![Dance website showcase](docs/media/official-dance.gif) | ![Jump simulation](docs/media/jump.gif)<br>![Jump website showcase](docs/media/official-jump.gif) | ![Grasp simulation](docs/media/claw.gif)<br>![Grasp website showcase](docs/media/official-grasp.gif) |
| **Dance** | **Jump** | **Grasp** |

## One sentence to create a scene

> Create a park pump-track scene for Jumper with rolling terrain, trees and benches, then export a `.map`.

| | | |
|:-:|:-:|:-:|
| ![Park pump track scene](docs/media/design-park.png) | ![Bedroom scene](docs/media/design-bedroom.png) | ![Soccer scene](docs/media/design-soccer.png) |
| [**Park pump track**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/park-pump-track.map) | [**Bedroom**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/bedroom.map) | [**Soccer**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/soccer.map) |

[Browse all maps](https://github.com/KingKongRobotics/jumper-design/tree/main/library/maps)

## Where to find things

| Resource | Purpose |
|---|---|
| [jumper-design](https://github.com/KingKongRobotics/jumper-design) | Appearance and scene creation; your assistant reads its [instructions](https://github.com/KingKongRobotics/jumper-design/blob/main/AGENTS.md) and uses its tools as needed. |
| [Training tutorial](docs/TUTORIAL.md) | Motion training, replay and policy export in this repository. |
| [Motion bundles](deploy/BUNDLE.md) | Package trained motions and their controller as an `.app`; see the [build guide](deploy/README.md) for prerequisites. |
| [Project guide](docs/PROJECT_GUIDE.md) | Setup, current capabilities and further documentation. |

Training builds on [mjlab](https://github.com/mujocolab/mjlab),
[rsl_rl](https://github.com/leggedrobotics/rsl_rl), [MuJoCo](https://github.com/google-deepmind/mujoco)
and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
Example appearance and scene images come from jumper-design; [image sources](docs/media/DESIGN_SOURCES.md)
and [third-party notices](NOTICE) record attribution.

## Jumper Barista

A fork project: in a **MuJoCo simulation**, Jumper picks up a small gooseneck pitcher, pours a
slow, centred 3.5-turn spiral over a V60 and sets the pitcher back down. **The water is a
particle/visual approximation** (a massless ballistic stream drawn from the measured spout pose,
with an 80 g rigid mass proxy in the pitcher); MuJoCo has no fluid. Nothing here has run on a
real robot. Goal, limits and milestones are in [`barista/CHARTER.md`](barista/CHARTER.md); the
scene and the replay gates are described in [`barista/COFFEE_SCENE.md`](barista/COFFEE_SCENE.md).

Everything runs on the `native:cpu` backend (tested on an Apple M5 Pro, macOS). Generated
outputs go under `out/`, which is gitignored.

Setup, from a clean checkout (Python 3.10–3.13; Rust/cargo is needed only for the bundle):

```bash
git clone https://github.com/mfiumara/jumper-barista.git && cd jumper-barista
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -e .
```

Replay and verify the pour (about 3 minutes; exit code 0 only if every gate passes: spout
spiral, stream landings, contacts, set-down, payload and the per-substep torque audit against
the servo curve, 300 ms peak and 1.2 N·m budget):

```bash
python barista/verification/run_m4.py --out out/barista/m4 --save-render-states
```

It writes `summary.json`, the per-substep `torque_substeps.csv`, `motion_substeps.csv`,
`stream_samples.csv`, a 960×540 verifier video, and `render_states.npz` (render-rate `qpos` and
logged stream paths, with the qpos SHA-256 that `summary.json` reports).

Render the film from that replay (headless; it refuses a replay that did not pass):

```bash
python barista/film/make_film.py --replay out/barista/m4 --out out/barista/film/jumper_barista_sim.mp4
# alternative cut without the close-up inset during the overhead pour:
python barista/film/make_film.py --replay out/barista/m4 --out out/barista/film/jumper_barista_sim_no_inset.mp4 --no-inset
```

Build and check the `.app` with the repository's own tooling. The bundle carries the
`jumper.barista` contract driven by the five-foot stance policy with the left-front claw
carried; the spiral pour itself is the reference replay above, not a mode of the bundle.
`--allow-incomplete` leaves out the robot's cross-built controller and `.rknn`, because this
bundle is for simulation only.

```bash
python scripts/export.py --task jumper.barista --backend native --device cpu \
  --checkpoint tasks/jumper/five_foot/out/example/model_86600.pt \
  --out tasks/jumper/barista/out/m5 --no-video
python scripts/deploy.py --manifests barista/deploy/manifests.json --allow-incomplete --bundle out/barista/bundle
python scripts/deploy.py --check-reference out/barista/bundle/barista
python scripts/play.py --task jumper.barista --backend native --device cpu --num_envs 1 \
  --scene coffee_bar --headless --steps 250 --app out/barista/bundle/barista.app
```

Third-party assets and their licences are listed in [`barista/SOURCES.md`](barista/SOURCES.md).

## License

Copyright 2026 KingKong Robotics.

Maintainer-owned project materials are licensed under Apache-2.0. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [licensing details](docs/PROJECT_GUIDE.md#license).
Third-party materials remain under their respective terms, and generated outputs do not
automatically inherit this repository's license.
