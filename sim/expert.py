"""Hand-written driver with perfect knowledge of the world.

It isn't a learned policy and cheats by reading the simulator state directly.
It exists to prove the roads, physics and traffic are drivable, and gives a
reference score for trained policies to beat.
"""
import numpy as np

from .car import DT, LENGTH, MAX_BRAKE, MAX_SPEED, MAX_STEER, WHEELBASE
from .world import LANE_WIDTH, wrap

LAT_ACCEL = 3.0  # m/s^2 allowed in bends
SAFE_GAP = 8.0  # metres kept to the car ahead when stopped
CRUISE = 0.8 * MAX_SPEED


def _gap_ahead(env, lane, behind=0.0):
    """Free road in a lane: distance to the nearest car ahead (or within `behind` metres back)."""
    t = env.traffic
    rel = t.s[t.lane == lane] - env.s
    rel = rel[rel > -behind]
    return (rel.min() - LENGTH) if len(rel) else np.inf


def act(env):
    road, car = env.road, env.car
    current = int(np.clip(round((env.lat + road.width / 2) / LANE_WIDTH - 0.5), 0, road.lanes - 1))
    gap = _gap_ahead(env, current, behind=LENGTH / 2)
    lane = current
    if gap < 60:  # look for a clearer lane next door, only if it's free alongside us
        for other in (current - 1, current + 1):
            if 0 <= other < road.lanes:
                other_gap = _gap_ahead(env, other, behind=10.0)
                if other_gap > max(gap + 20, 25):
                    lane, gap = other, other_gap

    # Pure pursuit towards the centre of the chosen lane.
    look = max(6.0, car.v * 0.8)
    px, py, _ = road.pose_at(env.s + look, road.lane_center(lane))
    alpha = wrap(np.arctan2(py - car.y, px - car.x) - car.yaw)
    steer = np.arctan(2 * WHEELBASE * np.sin(alpha) / look) / MAX_STEER

    # Speed: slow for upcoming bends and for the car ahead.
    _, _, h = road.pose_at(env.s + np.array([0.0, 15.0, 30.0, 45.0]), 0.0)
    bend = max(np.abs(wrap(h[1:] - h[0])).max() / 45.0, 1e-4)  # rough curvature, 1/m
    target = min(CRUISE, np.sqrt(LAT_ACCEL / bend))
    if np.isfinite(gap):
        target = min(target, np.sqrt(max(0.0, 2 * 0.5 * MAX_BRAKE * (gap - SAFE_GAP))))
    pedal = np.clip((target - car.v) / (MAX_BRAKE * DT * 3), -1, 1)
    return np.array([np.clip(steer, -1, 1), pedal])
