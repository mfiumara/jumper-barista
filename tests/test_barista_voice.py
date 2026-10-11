"""Act 2 / A3a: the order-scene voice pipeline and Jumper's display-only eye panel.

Fast tests cover the intent parser, the paid-provider spend guard (with the network
blocked), the script and manifest contract and the eye panel's physics isolation. The
end-to-end test (placeholder `say` -> whisper.cpp -> intent -> manifest) runs only where
`say`, `whisper-cli`, `ffmpeg` and the local ggml model exist, and is skipped elsewhere.
The full M4 hash comparison is `python barista/character/eyes.py m4-hash-check`; it takes
minutes, so it is release evidence rather than a unit test.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VOICE = REPO / "barista" / "voice"
EYES = REPO / "barista" / "character" / "eyes.py"


def _voice(name: str):
    if "barista_voice" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "barista_voice", VOICE / "__init__.py", submodule_search_locations=[str(VOICE)]
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        sys.modules["barista_voice"] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"barista_voice.{name}")


def _eyes():
    spec = importlib.util.spec_from_file_location("barista_eyes", EYES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_network(monkeypatch):
    """Any socket creation fails the test, so 'refused' cannot hide a request."""
    opened = []

    def guard(*args, **kwargs):
        opened.append(args)
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    return opened


# --------------------------------------------------------------------------- intent


@pytest.mark.parametrize("transcript", [
    "Ciao Jumper. Could I get a V60 pour over, please?",
    "Ciao, jumper, could I get a v60 pour-over please",
    "Could I have a V-60, please?",
    "I'd like a pourover.",
    "One vee sixty please",
    "Can I get a V 60?",
    "Could I get a filter coffee?",
    "A hand drip, please.",
])
def test_intent_maps_pour_over_phrasings(transcript):
    result = _voice("intent").parse_intent(transcript)
    assert result["intent"] == "order_drink"
    assert result["drink"] == "v60_pour_over"


@pytest.mark.parametrize(("transcript", "drink"), [
    ("Could I get an espresso?", "espresso"),
    ("One cappuccino, please", "cappuccino"),
    ("A latte please", "latte"),
])
def test_intent_maps_other_drinks(transcript, drink):
    assert _voice("intent").parse_intent(transcript)["drink"] == drink


@pytest.mark.parametrize("transcript", [
    "", "Child jumper, could I get a v64 over, please?", "What a nice day", "I'll be 60 soon",
])
def test_intent_falls_back_to_unknown(transcript):
    result = _voice("intent").parse_intent(transcript)
    assert result == {"intent": "unknown", "drink": "unknown", "matched": None,
                      "order_cue": False}


def test_normalise_is_insensitive_to_punctuation_case_and_hyphens():
    normalise = _voice("intent").normalise
    script = "Ciao Jumper! Could I get a V60 pour-over, please?"
    assert normalise(script) == "ciao jumper could i get a v60 pour over please"
    assert normalise("Ciao Jumper. Could I get a V60 pour over, please?") == normalise(script)
    assert normalise("ciao jumper, could I get a V-60 pourover please") == normalise(script)
    assert normalise("Order noted \u2014 I prepare it now!") == "order noted i prepare it now"


# --------------------------------------------------------------------------- providers


def test_paid_providers_refuse_by_default_without_touching_the_network(
    no_network, tmp_path, monkeypatch
):
    providers = _voice("providers")
    for name in ("ELEVENLABS_API_KEY", "FISH_API_KEY", "FISH_AUDIO_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("elevenlabs", "fish_audio"):
        provider = providers.get_provider(name, env_file=None)
        assert provider.paid
        with pytest.raises(providers.SpendRefused, match="costs credits"):
            provider.synthesize("hello", {}, tmp_path / "x.wav")
        # --allow-spend alone is not enough: the key must be present too.
        provider = providers.get_provider(name, allow_spend=True, env_file=None)
        with pytest.raises(providers.SpendRefused, match="is set"):
            provider.synthesize("hello", {}, tmp_path / "x.wav")
    assert not no_network
    assert not list(tmp_path.iterdir())


def test_a_key_without_allow_spend_is_still_refused(no_network, tmp_path, monkeypatch):
    providers = _voice("providers")
    env = tmp_path / ".env"
    env.write_text("ELEVENLABS_API_KEY=dummy-not-a-key\nexport FISH_API_KEY='dummy'\n")
    for name in ("ELEVENLABS_API_KEY", "FISH_API_KEY", "FISH_AUDIO_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert providers.key_present("ELEVENLABS_API_KEY", env)
    assert providers.key_present("FISH_API_KEY", env)
    assert not providers.key_present("FISH_AUDIO_API_KEY", env)
    for name in ("elevenlabs", "fish_audio"):
        with pytest.raises(providers.SpendRefused, match="costs credits"):
            providers.get_provider(name, env_file=env).synthesize("hi", {}, tmp_path / "x.wav")
    # ElevenLabs is not the approved provider: both gates open still sends nothing.
    with pytest.raises(NotImplementedError, match="not the approved provider"):
        providers.get_provider("elevenlabs", allow_spend=True, env_file=env).synthesize(
            "hi", {}, tmp_path / "x.wav"
        )
    assert not no_network


def test_providers_module_imports_no_network_library():
    """Network code lives only in fish_audio.py, which providers imports lazily."""
    tree = ast.parse((VOICE / "providers.py").read_text("utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"urllib", "http", "requests", "httpx", "aiohttp", "socket",
                           "elevenlabs", "fish_audio_sdk", "fishaudio"}


def test_pipeline_cli_refuses_a_paid_provider_before_doing_anything(no_network, tmp_path):
    pipeline = _voice("pipeline")
    out = tmp_path / "out"
    assert pipeline.main(["--out", str(out), "--provider", "elevenlabs"]) == 3
    assert pipeline.main(["--out", str(out), "--provider", "fish_audio"]) == 3
    assert not out.exists()
    assert not no_network


# --------------------------------------------------------------------------- script


def test_script_lines_and_eye_vocabulary():
    pipeline = _voice("pipeline")
    script = json.loads((VOICE / "script.json").read_text("utf-8"))
    lines = pipeline.select_lines(script, [], [])
    assert [(x["id"], x["speaker"], x["eye_expression"]) for x in lines] == [
        ("exclaim", "jumper", "surprised"),
        ("order", "customer", "listening"),
        ("confirm", "jumper", "happy"),
    ]
    assert lines[0]["text"] == "Ah! Un cliente!"
    assert lines[1]["text"] == "Ciao Jumper! Could I get a V60 pour-over, please?"
    assert lines[2]["text"] == "Order noted \u2014 I prepare it now!"
    assert pipeline.select_lines(script, [], ["exclaim"])[0]["text"] == "Oh, a customer!"
    with_greet = pipeline.select_lines(script, ["greet"], [])
    assert [x["id"] for x in with_greet] == ["exclaim", "greet", "order", "confirm"]
    assert set(pipeline.EYE_EXPRESSIONS) == set(_eyes().EXPRESSIONS)
    for provider in ("placeholder-local", "elevenlabs", "fish_audio"):
        assert set(script["voices"][provider]) == {"jumper", "customer"}
    assert lines[2]["eye_expression_end"] == "wink"
    # Mattia's gate (t_0bfa02ac): the approved Voice Design prompts, verbatim.
    fish = script["voices"]["fish_audio"]
    assert fish["jumper"]["instruction"] == (
        "small robot, bright and friendly, mild Italian accent, slightly synthetic")
    assert fish["customer"]["instruction"] == (
        "neutral, clear adult English speaker, relaxed caf\u00e9 tone")
    assert all(isinstance(v["seed"], int) and 1 <= v["n"] <= 4 for v in fish.values())


def test_eye_keyframes_end_the_confirm_line_in_a_wink():
    pipeline = _voice("pipeline")
    script = json.loads((VOICE / "script.json").read_text("utf-8"))
    exclaim, order, confirm = pipeline.select_lines(script, [], [])
    assert pipeline.eye_keyframes(exclaim, 1.2) == [[0.0, "surprised"]]
    assert pipeline.eye_keyframes(order, 3.0) == [[0.0, "listening"]]
    assert pipeline.eye_keyframes(confirm, 2.5) == [[0.0, "happy"], [1.9, "wink"]]
    assert pipeline.eye_keyframes(confirm, 0.3) == [[0.0, "happy"], [0.0, "wink"]]


def test_validate_manifest_catches_contract_breaks():
    pipeline = _voice("pipeline")
    good = {
        "schema": "jumper-barista-voice/1",
        "lines": [
            {"id": "order", "speaker": "customer", "text": "t", "audio_path": "a.wav",
             "duration_s": 2.0, "sample_rate": 44100, "provider": "fish_audio",
             "transcript": "heard", "intent": {"drink": "v60_pour_over"},
             "eye_expression": "listening", "eye_keyframes": [[0.0, "listening"]]},
            {"id": "confirm", "speaker": "jumper", "text": "t", "audio_path": "b.wav",
             "duration_s": 2.0, "sample_rate": 44100, "provider": "fish_audio",
             "transcript": None, "intent": None, "eye_expression": "happy",
             "eye_keyframes": [[0.0, "happy"], [1.4, "wink"]]},
        ],
        "checks": [{"pass": True}],
        "passes": True,
    }
    assert pipeline.validate_manifest(good) == []
    bad = json.loads(json.dumps(good))
    bad["lines"][1]["eye_keyframes"][1] = [3.0, "wink"]
    bad["lines"][0]["transcript"] = None
    bad["passes"] = False
    errors = pipeline.validate_manifest(bad)
    assert len(errors) == 3, errors


# --------------------------------------------------------------------------- fish audio

_DUMMY_KEY = "dummy-fish-key-must-never-be-written"
_HAVE_LOCAL_SPEECH = (
    shutil.which("say") and shutil.which("whisper-cli") and shutil.which("ffmpeg")
)


class _FakeFish:
    """Stands in for urllib.request.urlopen: a wallet and a voice-design endpoint."""

    def __init__(self, credit="1.00", wav_for=None):
        self.credit = float(credit)
        self.wav_for = wav_for or (lambda text, index: _tone_wav(0.5 + 0.1 * index))
        self.requests = []

    def __call__(self, request, timeout=None):
        import base64
        import io

        assert request.get_header("Authorization") == "Bearer " + _DUMMY_KEY
        self.requests.append((request.get_method(), request.full_url))
        if request.full_url.endswith("/wallet/self/api-credit"):
            body = {"credit": f"{self.credit:.4f}", "cumulative_top_up": "5", "_id": "x"}
        else:
            assert request.get_header("Model") == "voice-design-1"
            payload = json.loads(request.data)
            assert set(payload) <= {"instruction", "reference_text", "language", "n", "speed",
                                    "num_step", "guidance_scale", "seed"}
            self.credit -= 0.01
            body = {"candidates": [
                {"id": f"c{i}", "index": i, "sample_rate": 44100, "duration_ms": 500,
                 "audio_base64": base64.b64encode(
                     self.wav_for(payload["reference_text"], i)).decode()}
                for i in range(payload["n"])]}

        class _Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Response(json.dumps(body).encode())


def _tone_wav(seconds: float, rate: int = 44100) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        t = np.arange(int(seconds * rate)) / rate
        handle.writeframes((3000 * np.sin(2 * np.pi * 220 * t)).astype("<i2").tobytes())
    return buffer.getvalue()


def _fish(tmp_path, monkeypatch, fake, cap=None):
    providers = _voice("providers")
    fish_audio = _voice("fish_audio")
    monkeypatch.setenv("FISH_AUDIO_API_KEY", _DUMMY_KEY)
    provider = providers.get_provider("fish_audio", allow_spend=True, env_file=None,
                                      ledger_path=tmp_path / "ledger.json", opener=fake)
    if cap is not None:
        provider.ledger.cap_usd = cap
    return providers, fish_audio, provider


def test_fish_audio_zero_credit_refuses_before_any_generation(no_network, tmp_path,
                                                               monkeypatch):
    fake = _FakeFish(credit="0")
    providers, _, provider = _fish(tmp_path, monkeypatch, fake)
    with pytest.raises(providers.SpendRefused, match="credit is \\$0.0000"):
        provider.preflight(3)
    assert fake.requests == [("GET", "https://api.fish.audio/wallet/self/api-credit")]
    pipeline = _voice("pipeline")
    fake.requests.clear()
    with pytest.raises(providers.SpendRefused):
        pipeline.run(tmp_path / "out", "fish_audio", True,
                     provider_kwargs={"env_file": None, "ledger_path": tmp_path / "l.json",
                                      "opener": fake})
    assert all(url.endswith("api-credit") for _, url in fake.requests)
    assert not (tmp_path / "out").exists() and not (tmp_path / "l.json").exists()
    assert not no_network


def test_fish_audio_cap_is_enforced_before_the_request(no_network, tmp_path, monkeypatch):
    fake = _FakeFish(credit="5")
    providers, fish_audio, provider = _fish(tmp_path, monkeypatch, fake, cap=0.025)
    voice = {"instruction": "x", "seed": 1, "n": 2, "name": "v"}
    provider.synthesize("Ah! Un cliente!", voice, tmp_path / "a" / "exclaim.wav", "exclaim")
    provider.synthesize("Ah! Un cliente!", voice, tmp_path / "a" / "exclaim.wav", "exclaim")
    sent = len(fake.requests)
    with pytest.raises(providers.SpendRefused, match="exceed"):
        provider.synthesize("third", voice, tmp_path / "a" / "x.wav", "x")
    assert len(fake.requests) == sent  # refused without sending
    with pytest.raises(providers.SpendRefused, match="exceed"):
        provider.preflight(1)
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["total_usd"] == pytest.approx(0.02)
    assert [r["line_id"] for r in ledger["requests"]] == ["exclaim", "exclaim"]
    assert fish_audio.request_price("s2.1-pro", "Ah! Un cliente!") == pytest.approx(15 * 15e-6)
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == [
        "exclaim.cand0.wav", "exclaim.cand1.wav", "exclaim.wav"]
    assert not no_network


def test_fish_audio_http_errors_never_echo_the_key(no_network, tmp_path, monkeypatch):
    import io
    import urllib.error

    def failing(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 402, "Payment Required", {},
                                     io.BytesIO(b'{"status":402,"message":"no credit"}'))

    _, fish_audio, provider = _fish(tmp_path, monkeypatch, failing)
    with pytest.raises(fish_audio.FishAudioError) as caught:
        provider.synthesize("hi", {"instruction": "x", "seed": 1}, tmp_path / "x.wav", "x")
    assert "402" in str(caught.value) and _DUMMY_KEY not in str(caught.value)
    assert not (tmp_path / "ledger.json").exists()  # failed requests are not billed
    assert not no_network


@pytest.mark.skipif(not _HAVE_LOCAL_SPEECH, reason="needs say, whisper-cli, ffmpeg")
def test_fish_audio_pipeline_offline_with_a_fake_server(no_network, tmp_path, monkeypatch):
    """Full pipeline against a fake Fish Audio whose 'candidates' are `say` renders.

    Customer candidate 0 says the wrong drink, candidate 1 the right one: the pipeline must
    keep candidate 1 and record both transcripts. No key value may reach any output.
    """
    stt = _voice("stt")
    if not stt.default_model().is_file():
        pytest.skip("whisper model not downloaded")
    providers = _voice("providers")
    say = providers.PlaceholderLocal()

    def wav_for(text, index):
        if text.startswith("Ciao") and index == 0:
            text = "Ciao Jumper! Could I get an espresso, please?"
        path = tmp_path / "say" / f"{abs(hash((text, index)))}.wav"
        say.synthesize(text, {"voice": "Karen", "rate_wpm": 175}, path)
        return path.read_bytes()

    fake = _FakeFish(credit="5", wav_for=wav_for)
    monkeypatch.setenv("FISH_AUDIO_API_KEY", _DUMMY_KEY)
    pipeline = _voice("pipeline")
    out = tmp_path / "voice"
    manifest = pipeline.run(out, "fish_audio", True, provider_kwargs={
        "env_file": None, "ledger_path": tmp_path / "ledger.json", "opener": fake})
    assert pipeline.validate_manifest(manifest, out) == []
    order = next(x for x in manifest["lines"] if x["id"] == "order")
    assert order["generation"]["candidate_index"] == 1
    assert [c["intent"]["drink"] for c in order["stt_candidates"]][:2] == [
        "espresso", "v60_pour_over"]
    assert order["intent"]["drink"] == "v60_pour_over"
    confirm = manifest["lines"][-1]
    assert confirm["eye_keyframes"][-1][1] == "wink"
    assert manifest["placeholder"] is False and manifest["placeholder_label"] is None
    # Drop-in swap: same filenames and line keys as the placeholder run.
    assert [x["audio_path"] for x in manifest["lines"]] == [
        "audio/exclaim.wav", "audio/order.wav", "audio/confirm.wav"]
    spend = manifest["spend"]
    assert spend["usd_this_run_ledger"] == pytest.approx(0.03)
    assert spend["usd_this_run_wallet_delta"] == pytest.approx(0.03)
    assert spend["wallet_before"]["credit_usd"] == pytest.approx(5.0)
    for path in [*out.rglob("*"), tmp_path / "ledger.json"]:
        if path.is_file():
            assert _DUMMY_KEY.encode() not in path.read_bytes()
    assert not no_network


@pytest.mark.skipif(not _HAVE_LOCAL_SPEECH, reason="needs macOS say, whisper-cli and ffmpeg")
def test_end_to_end_placeholder_pipeline_writes_the_manifest(tmp_path):
    stt = _voice("stt")
    if not stt.default_model().is_file():
        pytest.skip(f"whisper model {stt.default_model()} not downloaded")
    pipeline = _voice("pipeline")
    out = tmp_path / "voice"
    assert pipeline.main(["--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["schema"] == "jumper-barista-voice/1"
    assert manifest["passes"] is True
    keys = {"id", "speaker", "text", "audio_path", "duration_s", "sample_rate", "provider",
            "transcript", "intent", "eye_expression"}
    for line in manifest["lines"]:
        assert keys <= set(line)
        assert line["provider"] == "placeholder-local"
        assert (out / line["audio_path"]).is_file()
        assert line["duration_s"] > 0.5 and line["sample_rate"] == 22050
    order = next(x for x in manifest["lines"] if x["id"] == "order")
    intent = _voice("intent")
    assert intent.normalise(order["transcript"]) == intent.normalise(order["text"])
    assert order["intent"]["drink"] == "v60_pour_over"
    assert all(x["transcript"] is None for x in manifest["lines"] if x["speaker"] == "jumper")
    assert manifest["placeholder"] is True and manifest["publishable"] is False
    assert "PLACEHOLDER" in manifest["placeholder_label"]
    assert all(x["placeholder"] is True for x in manifest["lines"])
    assert pipeline.validate_manifest(manifest, out) == []
    assert manifest["lines"][-1]["eye_keyframes"][-1][1] == "wink"


# --------------------------------------------------------------------------- eyes


def test_eye_textures_are_distinct_for_all_five_expressions():
    eyes = _eyes()
    assert eyes.EXPRESSIONS == ("neutral", "surprised", "listening", "happy", "wink")
    images = {name: eyes.expression_texture(name) for name in eyes.EXPRESSIONS}
    for image in images.values():
        assert image.shape == (eyes.TEXTURE_SIZE[1], eyes.TEXTURE_SIZE[0], 3)
        assert image.max() > 200  # something glows
    names = list(images)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.abs(images[a].astype(int) - images[b].astype(int)).mean() > 1.0, (a, b)
    # Wink: open eye on the viewer's left, closed on the right -> left half is brighter.
    wink = images["wink"].astype(float)
    half = wink.shape[1] // 2
    assert wink[:, :half].sum() > wink[:, half:].sum()
    with pytest.raises(ValueError):
        eyes.expression_texture("angry")


def test_eye_panel_is_visual_only_and_leaves_mass_and_contacts_unchanged():
    mujoco = pytest.importorskip("mujoco")
    eyes = _eyes()
    xml = str(REPO / "assets" / "jumper" / "jumper.xml")
    plain = mujoco.MjSpec.from_file(xml).compile()
    spec = mujoco.MjSpec.from_file(xml)
    eyes.install(spec)
    model = spec.compile()
    assert model.ngeom == plain.ngeom + 1
    panel = eyes.EyePanel(model)
    geom = panel.geom
    assert model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0
    assert model.geom_group[geom] == 2
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) == (
        "display_module_link"
    )
    np.testing.assert_array_equal(model.body_mass, plain.body_mass)
    np.testing.assert_array_equal(model.body_inertia, plain.body_inertia)
    np.testing.assert_array_equal(model.body_ipos, plain.body_ipos)
    # Switching expressions writes only the panel's material id.
    before = model.geom_matid.copy()
    for name in eyes.EXPRESSIONS:
        panel.set(name)
        assert panel.expression == name
    changed = np.flatnonzero(model.geom_matid != before)
    assert set(changed.tolist()) <= {geom}
    with pytest.raises(ValueError):
        panel.set("angry")
    # Same stepping with and without the panel, expression switched mid-run.
    datas = []
    for index, m in enumerate((plain, model)):
        data = mujoco.MjData(m)
        data.qpos[2] = 0.2
        for step in range(400):
            if index == 1 and step % 50 == 0:
                panel.set(eyes.EXPRESSIONS[(step // 50) % len(eyes.EXPRESSIONS)])
            mujoco.mj_step(m, data)
        datas.append(data)
    assert datas[0].qpos.tobytes() == datas[1].qpos.tobytes()
    assert datas[0].qvel.tobytes() == datas[1].qvel.tobytes()
    assert datas[0].ncon == datas[1].ncon


def test_eye_timeline_is_piecewise_constant():
    eyes = _eyes()
    timeline = eyes.EyeTimeline([(2.0, "surprised"), (0.0, "neutral"), (5.0, "happy")])
    assert timeline.expression_at(-1.0) == "neutral"
    assert timeline.expression_at(0.0) == "neutral"
    assert timeline.expression_at(1.99) == "neutral"
    assert timeline.expression_at(2.0) == "surprised"
    assert timeline.expression_at(9.0) == "happy"
    with pytest.raises(ValueError):
        eyes.EyeTimeline([(0.0, "angry")])
    manifest = {"lines": [
        {"id": "exclaim", "eye_expression": "surprised", "duration_s": 1.5},
        {"id": "order", "eye_expression": "listening", "duration_s": 4.0},
    ]}
    timeline = eyes.EyeTimeline.from_manifest(manifest, {"exclaim": 3.0, "order": 6.0})
    assert timeline.expression_at(3.1) == "surprised"
    assert timeline.expression_at(5.2) == "neutral"
    assert timeline.expression_at(7.0) == "listening"
    assert timeline.expression_at(10.7) == "neutral"


def test_install_on_env_cfg_chains_an_existing_spec_fn():
    eyes = _eyes()
    calls = []

    class Scene:
        spec_fn = staticmethod(lambda spec: calls.append("previous"))

    class Cfg:
        scene = Scene()

    cfg = Cfg()
    original_install = eyes.install
    eyes.install = lambda spec, initial: calls.append(("eyes", initial))
    try:
        eyes.install_on_env_cfg(cfg, "wink")
        cfg.scene.spec_fn(object())
    finally:
        eyes.install = original_install
    assert calls == ["previous", ("eyes", "wink")]
