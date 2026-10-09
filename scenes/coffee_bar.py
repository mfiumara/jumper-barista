"""Dimensioned coffee counter for the Jumper Barista simulation.

The authored world remains a self-contained ``kk-scene-package/1`` under
``scenes/coffee_bar.scene``.  This small adapter makes that replay package a named
scene without copying its geometry or bypassing the package integrity checks.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .registry import Headlight, Scene

PACKAGE = Path(__file__).with_name("coffee_bar.scene")


def scene() -> Scene:
    """Load the hash-checked coffee-bar package as the registered scene."""
    from mjlab.utils import spec_config as sc

    from .package import load_package

    loaded = load_package(PACKAGE)
    # The package supplies two attached lights.  A registered Scene also carries
    # an ordinary light so the registry's explicit non-black-world invariant
    # remains true without special-casing package-backed scenes.
    return replace(
        loaded,
        headlight=Headlight(
            ambient=(0.12, 0.11, 0.10),
            diffuse=(0.24, 0.23, 0.22),
        ),
        lights=(
            sc.LightCfg(
                name="coffee_bar_ambient_fill",
                type="directional",
                pos=(-0.8, -0.6, 1.4),
                dir=(0.4, 0.3, -1.0),
                diffuse=(0.18, 0.17, 0.16),
                castshadow=False,
            ),
        ),
    )
