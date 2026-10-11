"""Text-to-speech providers behind one interface, so the pipeline swaps provider by name.

* ``placeholder-local``: macOS ``say`` rendering straight to a WAV file. Offline, free, never
  plays through the speakers. Used until Mattia's provider gate (A3b).
* ``elevenlabs`` and ``fish_audio``: **stubs**. They refuse to run unless the caller passes
  ``allow_spend=True`` (the pipeline's ``--allow-spend``) **and** the provider's key is
  present. Even then they stop with ``NotImplementedError``: the real request is A3b's job,
  after the gate. This module imports no network library at all.

Keys are read from the environment, then from the engineer profile's ``.env``. Their values
are held in memory only, never printed, logged or written; :func:`key_present` is the only
thing that reports on them, and it reports a boolean.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

PROFILE_ENV = Path.home() / ".hermes" / "profiles" / "jumper-barista-engineer" / ".env"


class SpendRefused(RuntimeError):
    """A paid provider was asked to generate without an explicit spend permission and key."""


@dataclass(frozen=True)
class Audio:
    path: Path
    duration_s: float
    sample_rate: int


def wav_info(path: Path) -> Audio:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
    return Audio(path=path, duration_s=round(frames / float(rate), 4), sample_rate=rate)


def _read_env_file(path: Path, name: str) -> str | None:
    """Return the value of ``name`` in a dotenv file, without ever echoing it."""
    if not path.is_file():
        return None
    for raw in path.read_text("utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip().strip("'\"")
        return value or None
    return None


def load_key(name: str, env_file: Path | None = PROFILE_ENV) -> str | None:
    value = os.environ.get(name) or None
    if value is None and env_file is not None:
        value = _read_env_file(env_file, name)
    return value


def key_present(name: str, env_file: Path | None = PROFILE_ENV) -> bool:
    return load_key(name, env_file) is not None


class Provider:
    """Interface: ``synthesize(text, voice, out_path) -> Audio``."""

    name = "base"
    paid = False

    def ensure_allowed(self) -> None:
        """Raise :class:`SpendRefused` if this provider may not generate. Free: no-op."""

    def synthesize(self, text: str, voice: dict, out_path: Path) -> Audio:
        raise NotImplementedError


class PlaceholderLocal(Provider):
    """macOS ``say`` to a 16-bit mono WAV file. No playback, no network."""

    name = "placeholder-local"
    sample_rate = 22050

    def __init__(self, binary: str = "say") -> None:
        self.binary = binary

    def synthesize(self, text: str, voice: dict, out_path: Path) -> Audio:
        executable = shutil.which(self.binary)
        if executable is None:
            raise RuntimeError(f"{self.binary!r} not found; placeholder TTS needs macOS `say`")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lead = int(voice.get("lead_in_ms", 0))
        tail = int(voice.get("tail_ms", 0))
        spoken = f"[[slnc {lead}]] {text} [[slnc {tail}]]"
        command = [
            executable, "-v", str(voice["voice"]), "-r", str(int(voice.get("rate_wpm", 175))),
            "-o", str(out_path), "--file-format=WAVE",
            f"--data-format=LEI16@{self.sample_rate}", spoken,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return wav_info(out_path)


class _PaidStub(Provider):
    paid = True
    key_names: tuple[str, ...] = ()

    def __init__(self, allow_spend: bool = False, env_file: Path | None = PROFILE_ENV) -> None:
        self.allow_spend = bool(allow_spend)
        self.env_file = env_file

    def _key(self) -> str | None:
        for name in self.key_names:
            value = load_key(name, self.env_file)
            if value is not None:
                return value
        return None

    def ensure_allowed(self) -> None:
        if not self.allow_spend:
            raise SpendRefused(
                f"{self.name}: refusing to generate; costs credits. Pass --allow-spend only "
                "after Mattia's provider/budget gate."
            )
        if self._key() is None:
            raise SpendRefused(
                f"{self.name}: refusing to generate; none of {list(self.key_names)} is set. "
                "Mattia adds the key to the engineer profile .env himself."
            )

    def synthesize(self, text: str, voice: dict, out_path: Path) -> Audio:
        self.ensure_allowed()
        raise NotImplementedError(
            f"{self.name}: generation is implemented in A3b, after the provider gate"
        )


class ElevenLabs(_PaidStub):
    name = "elevenlabs"
    key_names = ("ELEVENLABS_API_KEY",)


class FishAudio(_PaidStub):
    name = "fish_audio"
    key_names = ("FISH_AUDIO_API_KEY", "FISH_API_KEY")


PROVIDERS: dict[str, type[Provider]] = {
    PlaceholderLocal.name: PlaceholderLocal,
    ElevenLabs.name: ElevenLabs,
    FishAudio.name: FishAudio,
}


def get_provider(name: str, allow_spend: bool = False, **kwargs) -> Provider:
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown TTS provider {name!r}; expected one of {sorted(PROVIDERS)}")
    if issubclass(cls, _PaidStub):
        return cls(allow_spend=allow_spend, **kwargs)
    return cls(**kwargs)
