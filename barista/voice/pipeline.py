#!/usr/bin/env python3
"""Order-scene voice pipeline: TTS -> audio files -> local STT -> intent -> manifest.

Run from the repository root::

    python -m barista.voice.pipeline --out OUT_DIR                 # placeholder-local voices
    python -m barista.voice.pipeline --out OUT_DIR --provider elevenlabs --allow-spend  # A3b

Writes ``OUT_DIR/audio/<line id>.wav`` and ``OUT_DIR/manifest.json`` (schema
``jumper-barista-voice/1``, documented in ``barista/voice/README.md``). The customer line's
``transcript`` is whisper.cpp's output on that line's audio file, never the script text;
``intent`` is parsed from that transcript. Exit status: 0 when the customer transcript
matches the script up to normalisation and every expected intent is met, 2 when a check
fails, 3 when a paid provider refuses to spend.

Generated audio is an artefact: keep OUT_DIR outside the repository (or untracked).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import intent as intent_mod
from . import providers, stt

SCRIPT = Path(__file__).resolve().parent / "script.json"
SCHEMA = "jumper-barista-voice/1"
EYE_EXPRESSIONS = ("neutral", "surprised", "listening", "happy", "wink")
SIMULATION_NOTE = (
    "Synthetic voices for a MuJoCo simulation film. Placeholder audio is macOS `say`; the "
    "real robot has no speaker pipeline and did not do this."
)


def select_lines(
    script: dict, include_optional: list[str], alternatives: list[str]
) -> list[dict]:
    lines = [dict(line) for line in script["lines"]]
    for extra_id in include_optional:
        extra = next((x for x in script.get("optional_lines", []) if x["id"] == extra_id), None)
        if extra is None:
            raise ValueError(f"no optional line {extra_id!r}")
        after = next(i for i, line in enumerate(lines) if line["id"] == extra["after"])
        lines.insert(after + 1, dict(extra))
    for line_id in alternatives:
        line = next((x for x in lines if x["id"] == line_id), None)
        if line is None or not line.get("alternatives"):
            raise ValueError(f"line {line_id!r} has no alternative text")
        line["text"] = line["alternatives"][0]
    for line in lines:
        if line["speaker"] not in ("customer", "jumper"):
            raise ValueError(f"line {line['id']!r}: unknown speaker {line['speaker']!r}")
        if line["eye_expression"] not in EYE_EXPRESSIONS:
            raise ValueError(f"line {line['id']!r}: unknown eye {line['eye_expression']!r}")
    return lines


def run(
    out: Path,
    provider_name: str = "placeholder-local",
    allow_spend: bool = False,
    stt_model: Path | None = None,
    script_path: Path = SCRIPT,
    include_optional: list[str] | None = None,
    alternatives: list[str] | None = None,
) -> dict:
    script_bytes = script_path.read_bytes()
    script = json.loads(script_bytes)
    lines = select_lines(script, include_optional or [], alternatives or [])
    provider = providers.get_provider(provider_name, allow_spend=allow_spend)
    provider.ensure_allowed()  # a paid provider refuses here, before any file or STT work
    voices = script["voices"][provider_name]
    recogniser = stt.WhisperCpp(stt_model)
    out.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    checks = []
    for line in lines:
        rel = Path("audio") / f"{line['id']}.wav"
        audio = provider.synthesize(line["text"], voices[line["speaker"]], out / rel)
        entry = {
            "id": line["id"],
            "speaker": line["speaker"],
            "text": line["text"],
            "audio_path": rel.as_posix(),
            "duration_s": audio.duration_s,
            "sample_rate": audio.sample_rate,
            "provider": provider.name,
            "transcript": None,
            "intent": None,
            "eye_expression": line["eye_expression"],
        }
        if line["speaker"] == "customer":
            transcript = recogniser.transcribe(out / rel)
            entry["transcript"] = transcript.text
            entry["intent"] = intent_mod.parse_intent(transcript.text)
            matches = intent_mod.normalise(transcript.text) == intent_mod.normalise(line["text"])
            checks.append({
                "line": line["id"], "check": "transcript_matches_script_normalised",
                "pass": matches,
                "normalised_transcript": intent_mod.normalise(transcript.text),
                "normalised_script": intent_mod.normalise(line["text"]),
            })
            expected = line.get("expected_intent")
            if expected:
                ok = all(entry["intent"].get(key) == value for key, value in expected.items())
                checks.append({"line": line["id"], "check": "intent", "pass": ok,
                               "expected": expected, "observed": entry["intent"]})
        manifest_lines.append(entry)
    manifest = {
        "schema": SCHEMA,
        "simulation_note": SIMULATION_NOTE,
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "provider": provider.name,
        "stt": {
            "engine": recogniser.engine,
            "engine_version": recogniser.version,
            "model": recogniser.model.name,
            "model_sha256": recogniser.model_sha256,
        },
        "lines": manifest_lines,
        "checks": checks,
        "passes": all(check["pass"] for check in checks) and bool(checks),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", default="placeholder-local",
                        choices=sorted(providers.PROVIDERS))
    parser.add_argument("--allow-spend", action="store_true",
                        help="permit a paid provider to generate (Mattia's gate; needs a key)")
    parser.add_argument("--stt-model", type=Path, default=None)
    parser.add_argument("--script", type=Path, default=SCRIPT)
    parser.add_argument("--include-optional", action="append", default=[], metavar="ID")
    parser.add_argument("--use-alternative", action="append", default=[], metavar="ID")
    args = parser.parse_args(argv)
    try:
        manifest = run(
            args.out.resolve(), args.provider, args.allow_spend, args.stt_model, args.script,
            args.include_optional, args.use_alternative,
        )
    except providers.SpendRefused as refused:
        print(f"[voice] refused: {refused}", file=sys.stderr)
        return 3
    for line in manifest["lines"]:
        heard = f" | heard: {line['transcript']!r} -> {line['intent']['drink']}" if (
            line["transcript"] is not None) else ""
        print(f"[voice] {line['id']:8s} {line['speaker']:8s} {line['duration_s']:6.2f}s "
              f"eyes={line['eye_expression']:9s} {line['text']!r}{heard}")
    for check in manifest["checks"]:
        print(f"[voice] check {check['line']}/{check['check']}: pass={check['pass']}")
    print(f"[voice] stt: {manifest['stt']['engine']} {manifest['stt']['engine_version']} "
          f"{manifest['stt']['model']}")
    print(f"[voice] manifest: {args.out.resolve() / 'manifest.json'} passes={manifest['passes']}")
    return 0 if manifest["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
