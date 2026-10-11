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
        self.non_exempt_contact = {"minimum_m": math.inf, "nearest_pair": None, "time_s": None}

    def observe(self, model, data, time_s, distance, prefilter_m):
        """Contact pairs use measured contact depth; other pairs use confirmed distance.

        The convex distance solver sometimes returns spurious deep penetration for
        separated meshes. Confirm non-contact pairs, but never promote a real contact
        depth to positive clearance. Far pairs retain their bounding-sphere lower bound.
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
                contact_depth = contacts.get(frozenset(pair))
                if contact_depth is not None:
                    value = contact_depth
                    if pair not in self.excluded and value < self.non_exempt_contact["minimum_m"]:
                        self.non_exempt_contact.update(
                            minimum_m=value, time_s=time_s,
                            nearest_pair=[self.names[index] for index in pair],
                        )
                elif value <= prefilter_m:
                    value = distance(*pair)
                for scope, minimum in self.minima.items():
                    if scope != "unrestricted" and pair in self.excluded:
                        continue
                    if value < minimum["minimum_m"]:
                        minimum.update(minimum_m=value, time_s=time_s,
                                       nearest_pair=[self.names[index] for index in pair])
        self.samples += 1

    def report(self):
        return {"scope": "spiral, every physics substep",
                "robot_scope": "all robot geoms, including visual geoms and every claw surface",
                "pitcher_scope": "all external pitcher geoms including the visual spout",
                "distance_method": "actual contact depths, otherwise confirmed geom distance; "
                                   "far pairs use bounding-sphere lower bounds",
                "excluded_pairs": [list(pair) for pair in INTENTIONAL_GRASP_PAIRS],
                "non_exempt_actual_contact": self.non_exempt_contact,
                "samples": self.samples, **self.minima}
