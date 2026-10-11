# Third-party asset sources

| Asset | Source URL | Licence | Added by |
|---|---|---|---|
| Coffee-bar MJCF primitive geometry and built-in materials | Project-authored; no external source | Apache-2.0 | Jumper Barista project |
| Film caption font: Aileron Regular, as embedded in Pillow and loaded by `PIL.ImageFont.load_default(size=...)` in `barista/film/make_film.py` (no font file is added to this repository) | https://dotcolon.net/fonts/aileron (Pillow's `ImageFont.load_default` docstring names this source) | CC0 1.0 Universal (public domain dedication), per the page's schema.org `license` field `https://creativecommons.org/publicdomain/zero/1.0/`, checked 2026-10-11 | Jumper Barista project (M5) |
| Film shots, title and end cards | Rendered by `barista/film/make_film.py` from this repository's MuJoCo replay (Jumper model from `assets/jumper/`, upstream Apache-2.0) | Apache-2.0 project materials; generated output | Jumper Barista project (M5) |
| Speech-to-text model `ggml-base.en.bin` (whisper.cpp conversion of OpenAI Whisper base.en), used locally by `barista/voice/stt.py`; downloaded to `~/.cache/jumper-barista/whisper/`, not committed (sha256 `a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002`) | https://huggingface.co/ggerganov/whisper.cpp (model card front matter `license: mit`, checked 2026-10-11); engine whisper.cpp 1.9.1 via Homebrew `whisper-cpp`, https://github.com/ggerganov/whisper.cpp (MIT) | MIT | Jumper Barista project (A3a/A3b) |
| Placeholder voices: macOS `say` voices `Alice` (it_IT, Jumper) and `Karen` (en_AU, customer), rendered to WAV by `barista/voice/providers.py` `placeholder-local` | Apple macOS system voices (bundled with the OS) | Apple macOS software licence; **draft/internal only, not for publication**. These renders must not appear in the competition video or any posted material | Jumper Barista project (A3a); A3b scene audio `exclaim.wav`, `order.wav`, `confirm.wav` are these **PLACEHOLDER** renders (manifest `placeholder: true`, `publishable: false`) for the A5 draft cut only |
| Final scene voices (A3c, deferred): Fish Audio Voice Design `voice-design-1`, designed voices `jumper-designed-b` and `customer-designed-neutral-en` (prompts and seeds below); no Voice Library/community voice, no cloning, no reference audio uploaded | https://fish.audio/terms (Terms of Use, effective date August 18, 2024, checked 2026-10-11); pricing https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits (checked 2026-10-11) | Fish Audio Terms of Use: Paid Services users "are licensed to use the Services for commercial uses and otherwise to the fullest extent possible under applicable law"; no attribution clause found in the terms (checked 2026-10-11) | Jumper Barista project - **deferred to A3c: no audio generated, no Fish Audio call in A3b, $0 spent** |

## Audit notes

- The M5 film has **no audio track** (silence). No music, sound effects, stock footage,
  third-party textures or meshes are used.
- The `.app` bundle built from `barista/deploy/manifests.json` contains only this repository's
  controller, the upstream `jumper.five_foot` example checkpoint exported under
  `jumper.barista`, and generated metadata.

## Fish Audio voice generation (prepared in A3b, run in A3c)

Approved by Mattia on the voice gate (t_0bfa02ac, 2026-10-11): Fish Audio pay-as-you-go,
**usage cap $3**, at most one top-up of $5 or less, no auto-recharge, designed voices only.

- Models and prices (docs pricing page, checked 2026-10-11): `voice-design-1` $0.01 per
  successful request (once per request, however many candidates it returns); `s2.1-pro`
  $15.00 per million UTF-8 bytes; `s2.1-pro-free` $0 (dry runs only, never shipped). A3b uses
  only `voice-design-1`: one request per line, the line as `reference_text`.
- Jumper voice `jumper-designed-b`: instruction "small robot, bright and friendly, mild
  Italian accent, slightly synthetic" (prompt b; Mattia's taste note "cute"), seed 6060,
  2 candidates per request, candidate 0 kept.
- Customer voice `customer-designed-neutral-en`: instruction "neutral, clear adult English
  speaker, relaxed café tone", seed 4242, language `en`, 4 candidates per request; the first
  candidate whose whisper.cpp transcript parses to `v60_pour_over` is kept.
- Spend ledger (every billed request, numbers only): `barista/voice/fish_audio_spend.json`.
  The cap is enforced in code before each request (`barista/voice/fish_audio.py`).

| date (UTC) | event | wallet credit (USD) | cumulative top-up (USD) | usage this run (USD) | cumulative usage (USD) |
|---|---|---|---|---|---|
| 2026-10-11T08:53Z | wallet read before any generation (`GET /wallet/self/api-credit`, free) | 0 | 0 | 0 | 0 |
| 2026-10-11 | `pipeline --provider fish_audio --allow-spend` preflight: refused, credit $0 < planned $0.03; no generation request sent | 0 | 0 | 0 | 0 |

Mattia (2026-10-11) deferred Fish Audio generation to the last step (card A3c, after A5
acceptance and a confirmed credit > 0). A3b therefore ships **placeholder `say` voices only**,
labelled `placeholder: true`, `publishable: false` in the manifest. A3c adds the
per-generation rows here from the ledger.
