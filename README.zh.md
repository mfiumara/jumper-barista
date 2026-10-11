<!-- tracks: README.md @ sha256:73b9122b8745f39b -->

![跳跳](docs/media/jumper-hero-zh.png)

# 跳跳

[English](README.md) | **简体中文**

**跳跳是一台 22 自由度螃蟹机器人。** [查看硬件 →](docs/HARDWARE.zh.md)

本仓库是跳跳的外观、动作与场景创作 AI 工具集。在 AI 编程助手中打开本仓库，试试下面的提示词。

> 🦀 **免费获得跳跳！** [了解如何领取 →](https://beunlimited.me/zh/events/crab-robot-challenge-2026)

## 一句话，设计外观

> 为跳跳设计一个暖沙色游侠外观，统一身体和四肢配色，并导出 `.skin`。

| | | |
|:-:|:-:|:-:|
| ![暖沙色游侠外观](docs/media/design-warm-sand.png) | ![银色装甲外观](docs/media/design-silver-armor.png) | ![Raphael Turtle 外观](docs/media/design-raphael.png) |
| [**暖沙色游侠**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/warm-sand-ranger-integrated-v2.skin) | [**银色装甲**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/mecha-tripo-v3.skin) | [**Raphael Turtle**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/raphael-turtle-v1.skin) |

[浏览全部外观](https://github.com/KingKongRobotics/jumper-design/tree/main/library/skins)

## 一句话，训练动作

> 为跳跳训练稳定的三足步态，回放并评估效果，然后打包生成 `.app` 动作包。

| | | |
|:-:|:-:|:-:|
| ![跳跳行走](docs/media/walk.gif) | ![跳跳改变姿态](docs/media/posture.gif) | ![跳跳挥手](docs/media/gesture.gif) |
| **行走** | **姿态** | **手势** |
| ![舞蹈仿真](docs/media/dance.gif)<br>![舞蹈官网展示](docs/media/official-dance.gif) | ![跳跃仿真](docs/media/jump.gif)<br>![跳跃官网展示](docs/media/official-jump.gif) | ![抓取仿真](docs/media/claw.gif)<br>![抓取官网展示](docs/media/official-grasp.gif) |
| **舞蹈** | **跳跃** | **抓取** |

## 一句话，生成场景

> 为跳跳生成一个有起伏地形、树木和长椅的公园泵道场景，并导出 `.map`。

| | | |
|:-:|:-:|:-:|
| ![公园泵道场景](docs/media/design-park.png) | ![卧室场景](docs/media/design-bedroom.png) | ![足球场景](docs/media/design-soccer.png) |
| [**公园泵道**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/park-pump-track.map) | [**卧室**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/bedroom.map) | [**足球**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/soccer.map) |

[浏览全部场景](https://github.com/KingKongRobotics/jumper-design/tree/main/library/maps)

## 相关项目与指南

| 资源 | 用途 |
|---|---|
| [jumper-design](https://github.com/KingKongRobotics/jumper-design) | 外观与场景生成；AI 读取其[工作说明](https://github.com/KingKongRobotics/jumper-design/blob/main/AGENTS.md)，按需使用工具。 |
| [训练教程](docs/TUTORIAL.zh.md) | 本仓库中的动作训练、回放与策略导出。 |
| [动作包格式](deploy/BUNDLE.md) | 将训练好的动作及控制器打包为 `.app`；环境要求见[构建指南](deploy/README.md)。 |
| [项目指南](docs/PROJECT_GUIDE.zh.md) | 环境安装、当前能力和更多文档。 |

训练基于 [mjlab](https://github.com/mujocolab/mjlab)、
[rsl_rl](https://github.com/leggedrobotics/rsl_rl)、[MuJoCo](https://github.com/google-deepmind/mujoco)
和 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)。
外观与场景示例图片来自 jumper-design，出处见[图片来源](docs/media/DESIGN_SOURCES.md)，
第三方材料归属见 [NOTICE](NOTICE)。

## Jumper Barista（跳跳咖啡师）

这是一个 fork 项目：在 **MuJoCo 仿真**中，跳跳拿起一只小手冲壶，在 V60 上方缓慢、居中地注出
3.5 圈螺旋，然后把手冲壶放回台面。**水流只是粒子/视觉近似**（从实测壶嘴位姿发出的无质量弹道水流，
壶内另有 80 g 刚性质量代替水）；MuJoCo 没有流体。这里的一切都没有在真实机器人上运行过。目标、
限制与里程碑见 [`barista/CHARTER.md`](barista/CHARTER.md)；场景与回放检验见
[`barista/COFFEE_SCENE.md`](barista/COFFEE_SCENE.md)。

全部在 `native:cpu` 后端运行（在 Apple M5 Pro、macOS 上测试）。生成的输出放在 `out/` 下，
该目录已被 git 忽略。

从干净的检出开始安装（Python 3.10–3.13；只有构建 bundle 才需要 Rust/cargo）：

```bash
git clone https://github.com/mfiumara/jumper-barista.git && cd jumper-barista
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -e .
```

回放并验证注水（约 3 分钟；只有全部检验通过才以退出码 0 结束：壶嘴螺旋、水流落点、接触、放回、
载荷，以及按每个物理子步对照舵机曲线、300 ms 峰值和 1.2 N·m 预算的力矩审计）：

```bash
python barista/verification/run_m4.py --out out/barista/m4 --save-render-states
```

它会写出 `summary.json`、逐子步的 `torque_substeps.csv`、`motion_substeps.csv`、
`stream_samples.csv`、一段 960×540 的验证视频，以及 `render_states.npz`（渲染帧率的 `qpos`
与记录的水流路径，附带 `summary.json` 中报告的 qpos SHA-256）。

从这次回放渲染影片（无界面；回放未通过时会拒绝）：

```bash
python barista/film/make_film.py --replay out/barista/m4 --out out/barista/film/jumper_barista_sim.mp4
# 另一个剪辑版本，俯视注水段不带近景小窗：
python barista/film/make_film.py --replay out/barista/m4 --out out/barista/film/jumper_barista_sim_no_inset.mp4 --no-inset
```

用仓库自带的工具构建并检查 `.app`。该 bundle 携带 `jumper.barista` 的契约，由五足站立策略驱动，
左前爪作为夹爪；螺旋注水本身是上面的参考回放，并不是 bundle 中的一个模式。`--allow-incomplete`
不包含机器人的交叉编译控制器和 `.rknn`，因为这个 bundle 只用于仿真。

```bash
python scripts/export.py --task jumper.barista --backend native --device cpu \
  --checkpoint tasks/jumper/five_foot/out/example/model_86600.pt \
  --out tasks/jumper/barista/out/m5 --no-video
python scripts/deploy.py --manifests barista/deploy/manifests.json --allow-incomplete --bundle out/barista/bundle
python scripts/deploy.py --check-reference out/barista/bundle/barista
python scripts/play.py --task jumper.barista --backend native --device cpu --num_envs 1 \
  --scene coffee_bar --headless --steps 250 --app out/barista/bundle/barista.app
```

第三方素材及其许可记录在 [`barista/SOURCES.md`](barista/SOURCES.md)。

## 许可证

Copyright 2026 KingKong Robotics.

维护者拥有权利的项目内容采用 Apache-2.0。详见 [LICENSE](LICENSE)、[NOTICE](NOTICE)
和[许可说明](docs/PROJECT_GUIDE.zh.md#许可证)。第三方内容保留其各自的许可条款；使用本工具生成的文件不会自动继承本仓库许可证。
