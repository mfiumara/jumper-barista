"""M5 release wiring for Jumper Barista (simulation only).

The film may only be cut from the verified M4 replay, must say on every frame that it
is a MuJoCo simulation with visual water, and the bundle must reuse the five-foot claw
hook through the barista task's own manifest rather than editing upstream's.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tomllib

REPO = Path(__file__).resolve().parents[1]
FILM = REPO / "barista" / "film" / "make_film.py"
MANIFESTS = REPO / "barista" / "deploy" / "manifests.json"
HOOK = REPO / "tasks" / "jumper" / "barista" / "deploy" / "lib.rs"


def _film():
    spec = importlib.util.spec_from_file_location("barista_make_film", FILM)
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses resolve their module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replay(tmp_path: Path, *, passes: bool = True, header: str | None = None,
            summary_hash: str | None = None) -> Path:
    qpos = np.arange(12, dtype=np.float64).reshape(3, 4)
    digest = hashlib.sha256(qpos.astype("<f8").tobytes()).hexdigest()
    np.savez_compressed(
        tmp_path / "render_states.npz", qpos=qpos,
        stream_paths=np.full((3, 13, 3), np.nan), render_fps=np.float64(50.0),
        qpos_sha256=np.str_(header or digest), commit_sha=np.str_("0" * 40),
        trajectory_sha256=np.str_("1" * 64),
    )
    summary = {
        "passes": passes, "release_evidence": True, "simulation_only": True,
        "determinism": {"qpos_sha256": summary_hash or digest},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), "utf-8")
    return tmp_path


def test_film_loads_only_the_verified_replay(tmp_path):
    film = _film()
    qpos, _, fps, _ = film._load(_replay(tmp_path))
    assert qpos.shape == (3, 4) and fps == 50.0


@pytest.mark.parametrize("kwargs", [
    {"passes": False},
    {"header": "f" * 64},
    {"summary_hash": "e" * 64},
])
def test_film_refuses_an_unverified_or_mismatched_replay(tmp_path, kwargs):
    with pytest.raises(SystemExit):
        _film()._load(_replay(tmp_path, **kwargs))


def test_film_cut_covers_pickup_pour_and_set_down_within_target_length():
    film = _film()
    shots = film.SHOTS
    # Contiguous, starting at the replay's start and ending at its final settle (57 s).
    assert shots[0].start_s == 0.0 and shots[-1].end_s == 57.0
    for before, after in itertools.pairwise(shots):
        assert before.end_s == after.start_s
    overhead = [shot for shot in shots if shot.view == "overhead"]
    # The whole spiral phase (23.5-41.5 s in m4_spiral_pour.json) is on the overhead view.
    assert overhead and overhead[0].start_s <= 23.5 and overhead[-1].end_s >= 41.5
    assert any(shot.view == "hero" for shot in shots)
    duration = film.TITLE_S + film.END_S + sum(
        round((shot.end_s - shot.start_s) / shot.speed * film.FPS) / film.FPS for shot in shots
    )
    assert 20.0 <= duration <= 40.0


def test_film_disclaimer_says_simulation_and_visual_water():
    text = _film().DISCLAIMER.lower()
    assert "mujoco simulation" in text
    assert "particle/visual" in text
    assert "not a real-robot run" in text


def test_barista_bundle_manifest_reuses_the_five_foot_hook():
    spec = json.loads(MANIFESTS.read_text("utf-8"))
    bundle = spec["bundles"]["barista"]
    mode = bundle["modes"]["barista"]
    assert mode["task"] == "jumper.barista"
    assert mode["hook"] == {"side": "left"}
    assert mode["policy"].startswith("tasks/jumper/barista/out/")
    fsm = tomllib.loads((MANIFESTS.parent / bundle["fsm"]).read_text("utf-8"))["fsm"]
    # The manifest owns the modes: the FSM file may not name a model.
    assert not any("model" in state for state in fsm["state"])
    assert fsm["initial_state"] == "barista"
    assert "pub use super::task_jumper__five_foot::build;" in HOOK.read_text("utf-8")
