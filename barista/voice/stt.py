"""Local speech-to-text with whisper.cpp (``whisper-cli``), run on the actual audio file.

Nothing leaves the machine: the audio is resampled to 16 kHz mono with ``ffmpeg`` into a
temporary file and transcribed by ``whisper-cli`` with a local ggml model.

Default model: ``ggml-base.en.bin`` from https://huggingface.co/ggerganov/whisper.cpp
(MIT, same licence as whisper.cpp and OpenAI Whisper), expected under
``~/.cache/jumper-barista/whisper/``; override with ``JUMPER_WHISPER_MODEL`` or ``--stt-model``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = Path.home() / ".cache" / "jumper-barista" / "whisper" / "ggml-base.en.bin"


@dataclass(frozen=True)
class Transcript:
    text: str
    engine: str
    engine_version: str
    model: str
    model_sha256: str


def default_model() -> Path:
    return Path(os.environ.get("JUMPER_WHISPER_MODEL", str(DEFAULT_MODEL)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def whisper_version(binary: str = "whisper-cli") -> str:
    """``whisper-cli --version`` prints ``whisper.cpp version: X.Y.Z``."""
    result = subprocess.run([binary, "--version"], check=False, capture_output=True, text=True)
    match = re.search(r"whisper\.cpp version:\s*v?(\d+\.\d+\.\d+)", result.stdout + result.stderr)
    return match.group(1) if match else "unknown"


class WhisperCpp:
    engine = "whisper.cpp"

    def __init__(self, model: Path | None = None, binary: str = "whisper-cli") -> None:
        self.model = Path(model) if model else default_model()
        self.binary = binary
        if not self.model.is_file():
            raise FileNotFoundError(
                f"whisper model {self.model} missing; download ggml-base.en.bin from "
                "huggingface.co/ggerganov/whisper.cpp (free) or set JUMPER_WHISPER_MODEL"
            )
        for tool in (binary, "ffmpeg"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"{tool!r} not found on PATH")
        self.version = whisper_version(binary)
        self.model_sha256 = _sha256(self.model)

    def transcribe(self, audio: Path, language: str = "en") -> Transcript:
        with tempfile.TemporaryDirectory(prefix="barista-stt-") as tmp:
            wav16 = Path(tmp) / "input16k.wav"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(audio),
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav16)],
                check=True, capture_output=True,
            )
            stem = Path(tmp) / "out"
            subprocess.run(
                [self.binary, "-m", str(self.model), "-f", str(wav16), "-l", language,
                 "-t", "4", "-nt", "-np", "-oj", "-of", str(stem)],
                check=True, capture_output=True, text=True,
            )
            payload = json.loads(stem.with_suffix(".json").read_text("utf-8"))
        text = " ".join(
            segment["text"].strip() for segment in payload.get("transcription", [])
        ).strip()
        return Transcript(
            text=text,
            engine=self.engine,
            engine_version=self.version,
            model=self.model.name,
            model_sha256=self.model_sha256,
        )
