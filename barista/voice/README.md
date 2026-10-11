# Order-scene voice pipeline (Act 2, A3)

Simulation film only: synthetic voices for the café story in `barista/STORY.md`. The real
robot has no speaker pipeline and did not do this.

```
script.json --TTS provider--> audio/<id>.wav --whisper.cpp (local)--> transcript
            --intent.parse_intent--> intent --> manifest.json (jumper-barista-voice/1)
```

## Run

From the repository root, with the Train environment's Python (`mujoco` is not needed for
the voice part; `say`, `ffmpeg` and `whisper-cli` are):

```bash
python -m barista.voice.pipeline --out /path/outside/the/repo/voice
python -m barista.voice.pipeline --out DIR --use-alternative exclaim   # "Oh, a customer!"
python -m barista.voice.pipeline --out DIR --include-optional greet    # adds the greeting
```

Exit status 0 means the customer transcript equals the script line after `normalise()` and
the parsed intent is `v60_pour_over`; 2 means a check failed (see `checks` in the
manifest); 3 means a paid provider refused to spend. Generated audio is an artefact: write
it outside the repository and do not commit it.

Eye panel renders and the M4 physics-isolation check:

```bash
python barista/character/eyes.py render --out DIR
python barista/character/eyes.py m4-hash-check --out DIR   # ~2-3 min, native:cpu
```

## Providers

| name | what | network | cost |
|---|---|---|---|
| `placeholder-local` | macOS `say` to a 22.05 kHz 16-bit mono WAV (Jumper: `Alice`, it_IT; customer: `Karen`) | none | free |
| `elevenlabs` | stub until A3b | refused | credits |
| `fish_audio` | stub until A3b | refused | credits |

`providers.get_provider(name, allow_spend=...)` is the only switch; the pipeline, manifest
and film do not change with the provider. A paid provider refuses unless **both**
`--allow-spend` is passed **and** its key is present (`ELEVENLABS_API_KEY`;
`FISH_API_KEY` or `FISH_AUDIO_API_KEY`) in the environment or in the engineer profile's
`.env`. With both present it still stops with `NotImplementedError`: the request itself is
A3b, after Mattia's provider/budget gate (`PROVIDER_PREFLIGHT.md`). Keys are never printed.

The customer placeholder voice was chosen by measurement, not taste: whisper.cpp
`base.en` heard Samantha and Alex as "Child jumper", Karen as "Ciao Jumper" at 165, 175
and 185 wpm. "Ciao" is the fragile word; re-check it with the real voices in A3b.

## STT

whisper.cpp `whisper-cli` (Homebrew `whisper-cpp`, 1.9.1 at the time of writing) with
`ggml-base.en.bin` from <https://huggingface.co/ggerganov/whisper.cpp> (MIT, like
whisper.cpp and OpenAI Whisper; downloaded free, not committed; sha256
`a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002`). Default location
`~/.cache/jumper-barista/whisper/ggml-base.en.bin`; override with `JUMPER_WHISPER_MODEL`
or `--stt-model`. The audio is resampled to 16 kHz mono with `ffmpeg` first. No prompt or
vocabulary hint is given to the recogniser; the transcript is exactly what it heard.

## Manifest schema `jumper-barista-voice/1`

A5 (the film) consumes this; keep it stable. Additive top-level fields are allowed.

```jsonc
{
  "schema": "jumper-barista-voice/1",
  "simulation_note": "...",
  "script_sha256": "<sha256 of script.json used>",
  "provider": "placeholder-local",
  "stt": {"engine": "whisper.cpp", "engine_version": "1.9.1",
          "model": "ggml-base.en.bin", "model_sha256": "..."},
  "lines": [                       // in scene order: exclaim, order, confirm
    {
      "id": "order",               // stable line id from script.json
      "speaker": "customer",       // "customer" | "jumper"
      "text": "Ciao Jumper! Could I get a V60 pour-over, please?",  // what was synthesised
      "audio_path": "audio/order.wav",   // relative to the manifest's directory
      "duration_s": 4.4162,        // of the audio file, including lead-in/tail silence
      "sample_rate": 22050,
      "provider": "placeholder-local",
      "transcript": "Ciao Jumper. Could I get a V60 pour over, please?",
                                   // customer lines: whisper.cpp output on audio_path,
                                   // never hand-written; jumper lines: null
      "intent": {"intent": "order_drink", "drink": "v60_pour_over",
                 "matched": "v60", "order_cue": true},   // customer lines; jumper: null
      "eye_expression": "listening"  // neutral | surprised | listening | happy | wink
    }
  ],
  "checks": [ {"line": "order", "check": "...", "pass": true, ...} ],
  "passes": true
}
```

Subtitles must show `transcript` for the customer line (STORY beat 5), not `text`.

## Eye expressions

`barista/character/eyes.py` adds one display-only panel on the display module's face
(visual group 2, `contype=0`, `conaffinity=0`, `density=0`, `mass=0`) and five materials:
`neutral`, `surprised`, `listening`, `happy`, `wink`. `EyePanel(model).set(name)` writes
only that geom's `geom_matid`; `EyeTimeline([(start_s, name), ...])` or
`EyeTimeline.from_manifest(manifest, {line_id: start_s})` gives the expression at a time.
Install it on an env config with `eyes.install_on_env_cfg(env_cfg)` after
`scenes.apply(...)`. With the panel installed the accepted M4 replay's qpos hash and
motion/torque/stream CSVs are byte-identical (`m4-hash-check`).

Textures are drawn procedurally by `eyes.py` (no third-party asset).
