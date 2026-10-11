# Third-party asset sources

| Asset | Source URL | Licence | Added by |
|---|---|---|---|
| Coffee-bar MJCF primitive geometry and built-in materials | Project-authored; no external source | Apache-2.0 | Jumper Barista project |
| Film caption font: Aileron Regular, as embedded in Pillow and loaded by `PIL.ImageFont.load_default(size=...)` in `barista/film/make_film.py` (no font file is added to this repository) | https://dotcolon.net/fonts/aileron (Pillow's `ImageFont.load_default` docstring names this source) | CC0 1.0 Universal (public domain dedication), per the page's schema.org `license` field `https://creativecommons.org/publicdomain/zero/1.0/`, checked 2026-10-11 | Jumper Barista project (M5) |
| Outfit concept mock-ups (A2a): primitive accessories, recolours, CONCEPT labels and contact sheet rendered by `barista/outfit/concepts.py` (labels use the same Pillow-bundled Aileron Regular as above; no other font, texture, mesh or image is used) | Project-authored; rendered from this repository's MuJoCo model (Jumper from `assets/jumper/`, upstream Apache-2.0) at the accepted M4 replay pose | Apache-2.0 project materials; generated output; font CC0 1.0 as above | Jumper Barista project (A2a) |
| Barista outfit, Concept C "Dapper Tricolore" (A2b): display-only primitive geoms (straw trilby, tricolore band, pencil-moustache decal, bar towel) and the `base_link_visual` white recolour, built by `barista/outfit/outfit.py`; `dapper_tricolore.visual_overrides.json`; outfit renders from `barista/outfit/outfit_check.py render` (labels use the Pillow-bundled Aileron Regular above) | Project-authored from the A2a concept geometry (`barista/outfit/concepts.py` @711e027); no mesh, texture, image or generated asset from any third party or paid service | Apache-2.0 project materials; generated output; font CC0 1.0 as above | Jumper Barista project (A2b) |
| Film shots, title and end cards | Rendered by `barista/film/make_film.py` from this repository's MuJoCo replay (Jumper model from `assets/jumper/`, upstream Apache-2.0) | Apache-2.0 project materials; generated output | Jumper Barista project (M5) |

## Audit notes

- The M5 film has **no audio track** (silence). No music, sound effects, stock footage,
  third-party textures or meshes are used.
- The `.app` bundle built from `barista/deploy/manifests.json` contains only this repository's
  controller, the upstream `jumper.five_foot` example checkpoint exported under
  `jumper.barista`, and generated metadata.
