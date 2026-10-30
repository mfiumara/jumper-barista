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

## Native prerequisite licence preflight (t_67782df4)

Checked 2026-10-12 before downloading any source archive, header or runtime:

- CycloneDDS 11.0.1 licence:
  https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/11.0.1/LICENSE
  SHA256 `d4486603568510bb9a1e6c5ee239a46ec513de55593e86176ccd81a5fdd3cfa3`.
  The public licence offers EPL-2.0 or EDL-1.0; EDL permits source/binary use
  and redistribution subject to notice, disclaimer and non-endorsement conditions.
  No click-through, registration or NDA was encountered retrieving this text.
  No source archive was fetched and no toolchain was built.
- Rockchip v2.3.2 licence:
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/LICENSE
  SHA256 `d846f57d942c7dfdca7b8b54f9e8bb39e1e226790dc4f5ee205d6fd678961720`.
  Its copyright statement says: "BY OPENING OR USING THIS FILE, RECEIVER
  HEREBY ACKNOWLEDGES AND AGREES" and reserves all rights. The retrieved
  text does not contain an affirmative local-development licence grant.
  This does not substantiate the use permission described in the vendor README.
  **Stopped for human licence review; no acceptance on the user's behalf.**
  Only licence text was retrieved; no header, runtime or SDK payload was fetched,
  installed or redistributed. The expected header hash in `fetch.sh` was not
  verified against a payload because the licence gate prevented its download.
