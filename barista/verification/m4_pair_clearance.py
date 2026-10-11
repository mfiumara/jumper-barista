"""Pair-specific pour clearance reporting, without modifying any acceptance gate."""
from __future__ import annotations

import math

import numpy as np

# The opposing grip inserts bear on the two printed end stops. No other claw surface
# or pitcher part is exempt, even if it happens to contact during a replay.
INTENTIONAL_GRASP_PAIRS = (
    ("robot/LF_finger_grip_insert_link_meshcol", "bar_top_grip_collar_a"),
    ("robot/LF_palm_grip_insert_link_meshcol", "bar_top_grip_collar_b"),
)


class PairClearance:
    """Track independent unrestricted and intentional-pair-excluded minima during pour."""

    def __init__(self, robot_geoms, pitcher_geoms, names):
        self.robot = np.asarray(robot_geoms, dtype=np.int64)
        self.pitcher = np.asarray(pitcher_geoms, dtype=np.int64)
        self.names = names
        by_name = {name: index for index, name in names.items()}
        self.excluded = {tuple(by_name[name] for name in pair) for pair in INTENTIONAL_GRASP_PAIRS}
        self.minima = {scope: {"minimum_m": math.inf, "nearest_pair": None, "time_s": None}
                       for scope in ("unrestricted", "excluding_intentional_grasp_pairs")}
        self.samples = 0

    def observe(self, model, data, time_s, distance, prefilter_m):
        """Near pairs use the confirmed distance; far pairs retain a lower bound.

        Actual contact distances also contribute so penetration cannot become a positive
        clearance through the convex-distance solver's distmax confirmation ladder.
        """
        gaps = (np.linalg.norm(data.geom_xpos[self.robot][:, None, :]
                               - data.geom_xpos[self.pitcher][None], axis=2)
                - model.geom_rbound[self.robot][:, None] - model.geom_rbound[self.pitcher][None])
        contacts = {}
        for contact in data.contact[:data.ncon]:
            key = frozenset((int(contact.geom1), int(contact.geom2)))
            contacts[key] = min(contacts.get(key, math.inf), float(contact.dist))
        for i, robot in enumerate(self.robot):
            for j, pitcher in enumerate(self.pitcher):
                pair = (int(robot), int(pitcher))
                value = float(gaps[i, j])
                if value <= prefilter_m:
                    value = distance(*pair)
                value = min(value, contacts.get(frozenset(pair), math.inf))
                for scope, minimum in self.minima.items():
                    if scope != "unrestricted" and pair in self.excluded:
                        continue
                    if value < minimum["minimum_m"]:
                        minimum.update(minimum_m=value, time_s=time_s,
                                       nearest_pair=[self.names[index] for index in pair])
        self.samples += 1

    def report(self):
        return {"scope": "spiral, every physics substep",
                "robot_scope": "all robot collision geoms, including every claw surface",
                "pitcher_scope": "all external pitcher geoms including the visual spout",
                "excluded_pairs": [list(pair) for pair in INTENTIONAL_GRASP_PAIRS],
                "samples": self.samples, **self.minima}
