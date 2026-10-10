"""The coffee-bar pitcher's grip add-on must be physically attached (simulation only).

A rigid MuJoCo body welds every geom to the pitcher, so a floating add-on would
still carry it in the replay; these tests keep the claimed load path honest.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "scenes" / "coffee_bar.scene" / "scene.xml"


def _module():
    path = REPO / "barista" / "verification" / "grip_attachment.py"
    spec = importlib.util.spec_from_file_location("grip_attachment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "scene.xml"
    path.write_text(text, "utf-8")
    return path


def test_the_grip_addon_is_attached_to_structural_pitcher_geometry():
    result = _module().check(SCENE)
    assert result["passes"], result["failures"]
    seated = [name for name, piece in result["pieces"].items() if piece["seated"]]
    assert seated, "no add-on piece touches collidable pitcher geometry"
    assert result["compiled_pitcher_mass_kg"] <= 0.450


def test_a_floating_addon_fails(tmp_path):
    text = SCENE.read_text("utf-8")
    floating = re.sub(
        r'\s*<geom name="top_grip_handle_(clamp_\w+|post)"[^>]*/>', "", text
    )
    assert floating != text
    result = _module().check(_variant(tmp_path, floating))
    assert not result["passes"]
    assert any("floats" in failure for failure in result["failures"])


def test_a_visual_only_handle_is_not_a_seat(tmp_path):
    text = SCENE.read_text("utf-8").replace(
        'class="pitcher_handle" type="capsule"', 'class="pitcher_visual" type="capsule"'
    )
    result = _module().check(_variant(tmp_path, text))
    assert not result["passes"]
    assert result["seat_geoms"] == ["pitcher_body"]


def test_a_penetrating_addon_fails(tmp_path):
    text = SCENE.read_text("utf-8").replace(
        'pos="0 -0.052 0.071" size="0.007 0.006 0.002"',
        'pos="0 -0.052 0.0705" size="0.007 0.006 0.002"',
    )
    result = _module().check(_variant(tmp_path, text))
    assert not result["passes"]
    assert any("penetrates" in failure for failure in result["failures"])
