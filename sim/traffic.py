"""Traffic: cars that follow their lane and keep a safe gap (IDM), plus parked cars.

Traffic never changes lane, so parked (broken-down) cars only go in the
rightmost lane and moving cars only in the others. That way traffic can never
queue forever behind a parked car, and the road is always passable. On a
one-lane road there are no parked cars.
"""
import numpy as np

from .car import DT, LENGTH, WIDTH
from .world import LANE_WIDTH

# Intelligent Driver Model parameters
IDM_ACCEL, IDM_BRAKE, IDM_HEADWAY, IDM_MIN_GAP = 1.5, 3.0, 1.5, 4.0


class Traffic:
    def __init__(self, road, rng, density=0.1, parked=0.2, start_s=5.0):
        self.road = road
        count = int(round(density * road.length * road.lanes / 20))
        s, lane, v0 = [], [], []
        for _ in range(count * 5):  # rejection sampling; gives up quietly if the road is full
            if len(s) == count:
                break
            is_parked = road.lanes > 1 and rng.random() < parked
            cs = rng.uniform(start_s + 25, road.length - 20)
            cl = 0 if is_parked else int(rng.integers(1 if road.lanes > 1 else 0, road.lanes))
            if all(abs(cs - os) > 15 for os, ol in zip(s, lane) if ol == cl):
                s.append(cs)
                lane.append(cl)
                v0.append(0.0 if is_parked else rng.uniform(8, 16))  # desired speed, 0 = parked
        self.s, self.lane = np.array(s, float), np.array(lane, int)
        self.v0 = np.array(v0, float)
        self.v = self.v0.copy()

    def __len__(self):
        return len(self.s)

    def step(self, ego_s, ego_lat, ego_v, blockers=None):
        """`blockers`: optional (s, lat, reach) arrays of other things to stop for, e.g. pedestrians on the road."""
        if len(self) == 0:
            return
        # A car's leader is the closest car ahead in its lane, ego included when it overlaps that lane.
        all_s, all_v = np.append(self.s, ego_s), np.append(self.v, ego_v)
        centre = self.road.lane_center(self.lane)
        ego_in_lane = np.abs(centre - ego_lat) < (LANE_WIDTH + WIDTH) / 2
        same_lane = np.hstack([self.lane[:, None] == self.lane[None, :], ego_in_lane[:, None]])
        if blockers is not None and len(blockers[0]):
            b_s, b_lat, b_reach = blockers
            all_s, all_v = np.append(all_s, b_s), np.append(all_v, np.zeros(len(b_s)))
            same_lane = np.hstack([same_lane, np.abs(centre[:, None] - b_lat[None, :]) < b_reach[None, :]])
        ahead = np.where(same_lane & (all_s[None, :] > self.s[:, None]), all_s[None, :], np.inf)
        lead = ahead.argmin(1)
        gap = ahead.min(1) - self.s - LENGTH
        dv = np.where(np.isfinite(gap), self.v - all_v[lead], 0.0)
        want = IDM_MIN_GAP + self.v * IDM_HEADWAY + self.v * dv / (2 * np.sqrt(IDM_ACCEL * IDM_BRAKE))
        free = (self.v / np.maximum(self.v0, 1e-3)) ** 4
        acc = IDM_ACCEL * (1 - free - (np.maximum(want, 0) / np.maximum(gap, 0.1)) ** 2)
        self.v = np.where(self.v0 > 0, np.clip(self.v + acc * DT, 0, None), 0.0)
        self.s += self.v * DT
        keep = self.s < self.road.length - 5  # cars leave at the end of the road
        self.s, self.lane, self.v, self.v0 = self.s[keep], self.lane[keep], self.v[keep], self.v0[keep]

    def poses(self):
        """x, y, yaw, vx, vy arrays."""
        x, y, yaw = self.road.pose_at(self.s, self.road.lane_center(self.lane))
        return x, y, yaw, self.v * np.cos(yaw), self.v * np.sin(yaw)
