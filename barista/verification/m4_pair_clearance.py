"""Pair-specific pour clearance reporting, without modifying any acceptance gate."""
from __future__ import annotations

import math

import numpy as np

# Collision identities stay separate: visual aliases never establish physical grip.
INTENTIONAL_GRASP_PAIRS = (
    ("robot/LF_finger_grip_insert_link_meshcol", "bar_top_grip_collar_a"),
    ("robot/LF_palm_grip_insert_link_meshcol", "bar_top_grip_collar_b"),
)
INTERFACE_CONTRACT_VERSION = "m4r-two-interface-v1"
INTENTIONAL_INTERFACE_PAIRS = (
    ("F", INTENTIONAL_GRASP_PAIRS[0],
     ("robot/LF_finger_grip_insert_link_visual", "bar_top_grip_collar_a")),
    ("P", INTENTIONAL_GRASP_PAIRS[1],
     ("robot/LF_palm_grip_insert_link_visual", "bar_top_grip_collar_b")),
)
EXEMPT_GEOM_PAIRS = tuple(pair for _, *pairs in INTENTIONAL_INTERFACE_PAIRS for pair in pairs)


class PairClearance:
    """Track independent unrestricted and intentional-pair-excluded minima during pour."""

    def __init__(self, robot_geoms, pitcher_geoms, names):
        self.robot = np.asarray(robot_geoms, dtype=np.int64)
        self.pitcher = np.asarray(pitcher_geoms, dtype=np.int64)
        self.names = names
        if len(set(names.values())) != len(names):
            raise ValueError("geom names must be unique for exact interface accounting")
        self.excluded = {(int(robot), int(pitcher))
                         for robot in self.robot for pitcher in self.pitcher
                         if (names[robot], names[pitcher]) in EXEMPT_GEOM_PAIRS}
        self.physical_contacts = {
            pair: {"interface": interface, "pair": list(pair), "contact_count": 0,
                   "minimum_m": None, "time_s": None, "maximum_normal_force_n": None,
                   "force_samples": 0}
            for interface, pair, _ in INTENTIONAL_INTERFACE_PAIRS
        }
        self.minima = {scope: {"minimum_m": math.inf, "nearest_pair": None, "time_s": None}
                       for scope in ("unrestricted", "excluding_intentional_grasp_pairs")}
        self.samples = 0
        self.non_exempt_contact = {"minimum_m": math.inf, "nearest_pair": None, "time_s": None}

    def observe(self, model, data, time_s, distance, prefilter_m, contact_force=None):
        """Contact pairs use measured contact depth; other pairs use confirmed distance.

        The convex distance solver sometimes returns spurious deep penetration for
        separated meshes. Confirm non-contact pairs, but never promote a real contact
        depth to positive clearance. Far pairs retain their bounding-sphere lower bound.
        """
        gaps = (np.linalg.norm(data.geom_xpos[self.robot][:, None, :]
                               - data.geom_xpos[self.pitcher][None], axis=2)
                - model.geom_rbound[self.robot][:, None] - model.geom_rbound[self.pitcher][None])
        contacts = {}
        for index, contact in enumerate(data.contact[:data.ncon]):
            key = frozenset((int(contact.geom1), int(contact.geom2)))
            contacts[key] = min(contacts.get(key, math.inf), float(contact.dist))
            for pair, physical in self.physical_contacts.items():
                if {self.names[int(contact.geom1)], self.names[int(contact.geom2)]} != set(pair):
                    continue
                physical["contact_count"] += 1
                if physical["minimum_m"] is None or contact.dist < physical["minimum_m"]:
                    physical.update(minimum_m=float(contact.dist), time_s=time_s)
                if contact_force is not None:
                    force = float(contact_force(index))
                    previous = physical["maximum_normal_force_n"]
                    physical["maximum_normal_force_n"] = (
                        force if previous is None else max(previous, force)
                    )
                    physical["force_samples"] += 1
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
                "interface_contract_version": INTERFACE_CONTRACT_VERSION,
                "intentional_interfaces": [
                    {"interface": interface, "collision_pair": list(collision),
                     "visual_alias_pair": list(visual)}
                    for interface, collision, visual in INTENTIONAL_INTERFACE_PAIRS
                ],
                "excluded_pairs": [list(pair) for pair in EXEMPT_GEOM_PAIRS],
                "physical_collision_contacts": list(self.physical_contacts.values()),
                "non_exempt_actual_contact": self.non_exempt_contact,
                "samples": self.samples, **self.minima}
