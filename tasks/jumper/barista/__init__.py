"""Registered simulation task for the Jumper Barista reference motion."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.barista",
    assets=JUMPER_ASSETS,
    description="simulation-only five-foot pitcher grasp and pour reference task",
    tags=("jumper", "barista", "manipulation", "reference"),
)
