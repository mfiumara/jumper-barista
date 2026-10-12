# Third-party asset sources

| Asset | Source URL | Licence | Added by |
|---|---|---|---|
| Coffee-bar MJCF primitive geometry and built-in materials | Project-authored; no external source | Apache-2.0 | Jumper Barista project |
| Film caption font: Aileron Regular, as embedded in Pillow and loaded by `PIL.ImageFont.load_default(size=...)` in `barista/film/make_film.py` (no font file is added to this repository) | https://dotcolon.net/fonts/aileron (Pillow's `ImageFont.load_default` docstring names this source) | CC0 1.0 Universal (public domain dedication), per the page's schema.org `license` field `https://creativecommons.org/publicdomain/zero/1.0/`, checked 2026-10-11 | Jumper Barista project (M5) |
| Film shots, title and end cards | Rendered by `barista/film/make_film.py` from this repository's MuJoCo replay (Jumper model from `assets/jumper/`, upstream Apache-2.0) | Apache-2.0 project materials; generated output | Jumper Barista project (M5) |

## Audit notes

- The M5 film has **no audio track** (silence). No music, sound effects, stock footage,
  third-party textures or meshes are used.
- The `.app` bundle built from `barista/deploy/manifests.json` contains only this repository's
  controller, the upstream `jumper.five_foot` example checkpoint exported under
  `jumper.barista`, and generated metadata.

## Native prerequisite licence checkpoint (M4R-H2b)

- Rockchip v2.3.2 licence inspected at
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/LICENSE
  (SHA256 `d846f57d942c7dfdca7b8b54f9e8bb39e1e226790dc4f5ee205d6fd678961720`).
  It contains acceptance-by-opening-or-use wording. Mattia accepted this licence
  on 2026-10-12 for local use of the pinned files via
  `deploy/fsm/vendor/rknpu2/fetch.sh` only (recorded on task `t_88ae5e12`).
  No committing, redistribution, or inclusion in pushed/packaged artifacts beyond
  upstream's build is authorized. No Apache-2.0 grant is asserted.
  The manager's final answer on `t_88ae5e12` explicitly authorizes direct download
  of only the pinned header from tag v2.3.2 into the script's ignored path.
  Header URL: https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h
  Required SHA256: `c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a`.
  Never run the runtime-fetching script, download the runtime, or redistribute SDK files.
- CycloneDDS 11.0.1 remote tag resolves to
  `e54e991f75a3e67f8e628da3171122e36ea5b872` at
  https://github.com/eclipse-cyclonedds/cyclonedds.git . Continuation `t_dfaf7b4a`
  fetched and verified this exact HEAD before building, with a clean source tree.
  `git archive --format=tar HEAD` SHA256:
  `025c5cd000babe569c8eb3b1692bc3a2a09966b31a91b45babc55332425671d9`.
  Source `LICENSE` and `NOTICE.md` declare EPL-2.0 OR BSD-3-Clause
  (Eclipse Distribution License 1.0); local use follows the permissive EDL option.
  Notice also lists libmd5-rfc under Zlib and getopt 1.5 as public domain.
  Exact licence: https://github.com/eclipse-cyclonedds/cyclonedds/blob/e54e991f75a3e67f8e628da3171122e36ea5b872/LICENSE
  SHA256 `d4486603568510bb9a1e6c5ee239a46ec513de55593e86176ccd81a5fdd3cfa3`.
  `NOTICE.md` SHA256 `dfe2fcb64896f2fb49a749496cf858524c4972331070a3c0efbf4e8f932b6ace`.
  Source and installed binaries remain ignored under `out/toolchain`, not redistributed.
- See `NATIVE_PREREQUISITES.md` for the bounded checkpoint and resume conditions.
