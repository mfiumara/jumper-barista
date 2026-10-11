"""The ``jumper-barista-beats/1`` beat log: build, validate, serialise.

Times are simulation seconds of the one continuous run. Motion beats are contiguous and never
overlap; ``chime`` may overlap ``dance``, ``exclaim`` overlaps ``excited_jump``, and ``listen``
and ``confirm`` overlap ``look`` (MOTION_CHAIN.md, "Beat log").
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

common = sys.modules.get("barista_chain_common")
if common is None:  # loaded by path
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("barista_chain_common", Path(__file__).with_name("common.py"))
    common = _ilu.module_from_spec(_spec)
    sys.modules["barista_chain_common"] = common
    _spec.loader.exec_module(common)

MOTION_BEATS = ("dance", "stop", "excited_jump", "look", "turn", "walk", "align", "pour",
                "set_down", "bow")
OVERLAY_HOST = {"chime": "dance", "exclaim": "excited_jump", "listen": "look", "confirm": "look"}
DEFAULT_HOLDS_S = {"exclaim": 1.5, "listen": 4.0, "confirm": 2.5}
TOLERANCE_S = 1e-6


def validate(beats: list[dict]) -> list[str]:
    """Return a list of problems (empty when the log is valid)."""
    problems = []
    names = set(common.BEAT_NAMES)
    for b in beats:
        if b["beat"] not in names:
            problems.append(f"unknown beat {b['beat']!r}")
        if not b["t_end_s"] > b["t_start_s"]:
            problems.append(f"{b['beat']}: empty or reversed interval")
    motion = [b for b in beats if b["beat"] in MOTION_BEATS]
    for a, b in itertools.pairwise(motion):
        if abs(a["t_end_s"] - b["t_start_s"]) > TOLERANCE_S:
            problems.append(f"motion beats {a['beat']} -> {b['beat']} not contiguous "
                            f"({a['t_end_s']} vs {b['t_start_s']})")
    for b in beats:
        host = OVERLAY_HOST.get(b["beat"])
        if host is None:
            continue
        hosts = [h for h in motion if h["beat"] == host]
        if not any(h["t_start_s"] - TOLERANCE_S <= b["t_start_s"] and
                   b["t_end_s"] <= h["t_end_s"] + TOLERANCE_S for h in hosts):
            problems.append(f"{b['beat']} [{b['t_start_s']}, {b['t_end_s']}] is not inside {host}")
    return problems


def document(beats: list[dict], *, commit: str, physics_dt_s: float, control_dt_s: float,
             extra: dict | None = None) -> dict:
    doc = {"schema": common.BEATS_SCHEMA, "commit": commit, "physics_dt_s": physics_dt_s,
           "control_dt_s": control_dt_s,
           "beats": [{"beat": b["beat"], "t_start_s": round(float(b["t_start_s"]), 6),
                      "t_end_s": round(float(b["t_end_s"]), 6)} for b in beats]}
    if extra:
        doc.update(extra)
    return doc


def write(path: Path, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", "utf-8")
