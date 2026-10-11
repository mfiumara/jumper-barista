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
  It contains acceptance-by-opening-or-use wording. Setup stopped for Mattia's
  decision under this card's explicit acceptance gate. No header or runtime was
  fetched, incorporated, or redistributed; no Apache-2.0 grant is asserted.
- CycloneDDS 11.0.1 remote tag resolves to
  `e54e991f75a3e67f8e628da3171122e36ea5b872` at
  https://github.com/eclipse-cyclonedds/cyclonedds.git . No source was downloaded
  and its licence audit/build remain pending. This is not an added asset.
- See `NATIVE_PREREQUISITES.md` for the bounded checkpoint and resume conditions.
