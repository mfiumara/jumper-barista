"""Text-to-speech providers behind one interface, so the pipeline swaps provider by name.

* ``placeholder-local``: macOS ``say`` rendering straight to a WAV file. Offline, free, never
  plays through the speakers. Used until Mattia's provider gate (A3b).
* ``fish_audio`` (A3b, Mattia's approved provider): Fish Audio Voice Design. Refuses unless
  the caller passes ``allow_spend=True`` (the pipeline's ``--allow-spend``) **and** the key
  is present; every request is also checked against the $3 cap in ``fish_audio.SpendLedger``
  before it is sent. The HTTP client lives in ``fish_audio.py`` and is imported lazily, so
  this module itself imports no network library.
* ``elevenlabs``: a **stub**. Same refusals; with both gates open it still stops with
  ``NotImplementedError`` (not the approved provider).

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
    """Interface: ``synthesize(text, voice, out_path, line_id=None) -> Audio``.

    After a call, ``last_generation`` holds what the manifest records about it (``None`` for
    free providers) and ``candidates`` lists every audio file the call produced, chosen one
    first.
    """

    name = "base"
    paid = False
    last_generation: dict | None = None
    candidates: list[Audio] | tuple[()] = ()

    def ensure_allowed(self) -> None:
        """Raise :class:`SpendRefused` if this provider may not generate. Free: no-op."""

    def preflight(self, planned_requests: int) -> dict | None:
        """Paid providers check credit and cap for a batch; returns the wallet. Free: None."""
        return None

    def wallet(self) -> dict | None:
        return None

    def synthesize(self, text: str, voice: dict, out_path: Path,
                   line_id: str | None = None) -> Audio:
        raise NotImplementedError


class PlaceholderLocal(Provider):
    """macOS ``say`` to a 16-bit mono WAV file. No playback, no network."""

    name = "placeholder-local"
    sample_rate = 22050

    def __init__(self, binary: str = "say") -> None:
        self.binary = binary

    def synthesize(self, text: str, voice: dict, out_path: Path,
                   line_id: str | None = None) -> Audio:
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
        audio = wav_info(out_path)
        self.candidates = [audio]
        return audio


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

    def synthesize(self, text: str, voice: dict, out_path: Path,
                   line_id: str | None = None) -> Audio:
        self.ensure_allowed()
        raise NotImplementedError(
            f"{self.name}: not the approved provider (Mattia's gate chose fish_audio)"
        )


class ElevenLabs(_PaidStub):
    name = "elevenlabs"
    key_names = ("ELEVENLABS_API_KEY",)


def to_pcm16(raw: Path, out_path: Path) -> Audio:
    """Keep a WAV the stdlib can read as-is; transcode anything else to 16-bit mono PCM."""
    try:
        with wave.open(str(raw), "rb") as handle:
            ok = handle.getsampwidth() == 2 and handle.getnchannels() == 1
    except (wave.Error, EOFError):
        ok = False
    if ok:
        shutil.copyfile(raw, out_path)
    else:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(raw), "-ac", "1",
             "-c:a", "pcm_s16le", str(out_path)],
            check=True, capture_output=True,
        )
    return wav_info(out_path)


class FishAudio(_PaidStub):
    """Voice Design per line: approved instruction, fixed seed, the line as reference text.

    ``voice`` keys: ``instruction`` (required), ``seed`` (required), ``n`` (candidates per
    request, billed once), ``language``, ``speed``. Candidate 0 is written to ``out_path``;
    all candidates are kept beside it as ``<id>.candN.wav`` for the pipeline's selection and
    for the record. The caller may re-point ``out_path`` at another candidate.
    """

    name = "fish_audio"
    key_names = ("FISH_AUDIO_API_KEY", "FISH_API_KEY")

    def __init__(self, allow_spend: bool = False, env_file: Path | None = PROFILE_ENV,
                 ledger_path: Path | None = None, opener=None) -> None:
        super().__init__(allow_spend=allow_spend, env_file=env_file)
        from . import fish_audio  # the only network code; imported only when used

        self._api = fish_audio
        self.ledger = fish_audio.SpendLedger(ledger_path or fish_audio.LEDGER)
        self._opener = opener

    def _kw(self) -> dict:
        return {} if self._opener is None else {"opener": self._opener}

    def wallet(self) -> dict | None:
        self.ensure_allowed()
        return self._api.read_wallet(self._key(), **self._kw())

    def preflight(self, planned_requests: int) -> dict | None:
        self.ensure_allowed()
        planned = planned_requests * self._api.request_price(self._api.VOICE_DESIGN_MODEL)
        try:
            return self._api.preflight(self._key(), planned, self.ledger, **self._kw())
        except self._api.SpendCapExceeded as refused:
            raise SpendRefused(f"{self.name}: {refused}") from None

    def synthesize(self, text: str, voice: dict, out_path: Path,
                   line_id: str | None = None) -> Audio:
        self.ensure_allowed()
        api = self._api
        try:
            candidates = api.voice_design(
                self._key(), voice["instruction"], text, seed=int(voice["seed"]),
                n=int(voice.get("n", 1)), language=voice.get("language"),
                speed=float(voice.get("speed", 1.0)), ledger=self.ledger,
                meta={"line_id": line_id, "voice": voice.get("name")}, **self._kw(),
            )
        except api.SpendCapExceeded as refused:
            raise SpendRefused(f"{self.name}: {refused}") from None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.candidates = []
        for cand in candidates:
            raw = out_path.with_name(f"{out_path.stem}.cand{cand.index}.raw.wav")
            raw.write_bytes(cand.wav_bytes)
            self.candidates.append(
                to_pcm16(raw, out_path.with_name(f"{out_path.stem}.cand{cand.index}.wav"))
            )
            raw.unlink()
        shutil.copyfile(self.candidates[0].path, out_path)
        self.last_generation = {
            "endpoint": "POST /v1/voice-design",
            "model": api.VOICE_DESIGN_MODEL,
            "voice": voice.get("name"),
            "instruction": voice["instruction"],
            "seed": int(voice["seed"]),
            "n": len(candidates),
            "language": voice.get("language"),
            "speed": float(voice.get("speed", 1.0)),
            "usd": api.request_price(api.VOICE_DESIGN_MODEL),
            "reference_text_utf8_bytes": len(text.encode("utf-8")),
            "api_sample_rate": [c.sample_rate for c in candidates],
            "api_duration_ms": [c.duration_ms for c in candidates],
            "candidate_index": 0,
        }
        return wav_info(out_path)


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
