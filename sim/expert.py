"""Hand-written driver with perfect knowledge of the world.

It isn't a learned policy and cheats by reading the simulator state directly.
It exists to prove the roads, physics, traffic and hazards are drivable, and
gives a reference score for trained policies to beat.
"""
import numpy as np

from .car import DT, LENGTH, MAX_BRAKE, MAX_SPEED, MAX_STEER, WHEELBASE, WIDTH
from .hazards import PED_CLEAR, POTHOLE_DAMAGE_SPEED, WHEEL_REACH
from .world import LANE_WIDTH, wrap

LAT_ACCEL = 3.0  # m/s^2 allowed in bends
SAFE_GAP = 8.0  # metres kept to the car ahead when stopped
CRUISE = 0.8 * MAX_SPEED
PED_LOOKAHEAD = 3.0  # seconds: yield to pedestrians who could be in our path within this time
POTHOLE_SPEED = 30 / 3.6  # crawl over potholes we can't avoid, safely under the damage speed
BRAKE_PLAN = 0.5 * MAX_BRAKE  # deceleration assumed when planning to slow down


def _gap_ahead(env, lane, behind=0.0):
    """Free road in a lane: distance to the nearest car, obstacle or pedestrian in the way."""
    t = env.traffic
    rel = t.s[t.lane == lane] - env.s
    rel = rel[rel > -behind]
    gap = (rel.min() - LENGTH) if len(rel) else np.inf
    hz = env.hazards
    if hz is None:
        return gap
    centre = env.road.lane_center(lane)
    reach = WIDTH / 2 + 0.3
    # debris in this lane
    rel = hz.obs_s - env.s
    blocking = (np.abs(hz.obs_lat - centre) < reach + hz.obs_size / 2) & (rel > -behind)
    if blocking.any():
        gap = min(gap, (rel[blocking] - LENGTH / 2 - hz.obs_size[blocking] / 2).min())
    # Crosswalks: yield the whole crosswalk while someone is crossing or about to. A waiting pedestrian won't
    # step out once a car is within PED_CLEAR of it, so only those still far enough away can surprise us.
    rel = hz.ped_s - env.s
    about_to = (hz.ped_wait < PED_LOOKAHEAD) & (rel > PED_CLEAR + 3)
    yield_to = (hz.crossing() | about_to) & (rel > -LENGTH / 2)
    if yield_to.any():
        gap = min(gap, (rel[yield_to] - LENGTH / 2 - 2.0).min())
    return gap


def _pothole_ahead(env, lat_lo, lat_hi, distance):
    """Distance to the first unhit pothole within `distance` metres whose hit zone touches [lat_lo, lat_hi]."""
    hz = env.hazards
    if hz is None or not len(hz.hole_s):
        return np.inf
    rel = hz.hole_s - env.s
    zone = hz.hole_r + WHEEL_REACH + 0.2
    hit = ~hz.hole_hit & (rel > -1) & (rel < distance) & (hz.hole_lat + zone > lat_lo) & (hz.hole_lat - zone < lat_hi)
    return rel[hit].min() if hit.any() else np.inf


def act(env):
    road, car = env.road, env.car
    current = int(np.clip(round((env.lat + road.width / 2) / LANE_WIDTH - 0.5), 0, road.lanes - 1))
    gap = _gap_ahead(env, current, behind=LENGTH / 2)
    lane = current
    centre = road.lane_center(current)
    holes_here = _pothole_ahead(env, centre, centre, 40)
    if gap < 60 or np.isfinite(holes_here):  # look for a clearer lane next door, only if it's free alongside us
        for other in (current - 1, current + 1):
            if 0 <= other < road.lanes:
                other_gap = _gap_ahead(env, other, behind=10.0)
                oc = road.lane_center(other)
                other_holes = _pothole_ahead(env, oc, oc, 40)
                # a pothole in the free lane is fine if ours is blocked: we just slow down for it
                better_gap = other_gap > max(gap + 20, 25) and (other_holes >= holes_here or gap < 20)
                dodges_hole = np.isfinite(holes_here) and not np.isfinite(other_holes) and other_gap > max(min(gap, 60), 25)
                if better_gap or dodges_hole:
                    lane, gap, holes_here = other, other_gap, other_holes

    # Pure pursuit towards the centre of the chosen lane.
    target_lat = road.lane_center(lane)
    look = max(6.0, car.v * 0.8)
    px, py, _ = road.pose_at(env.s + look, target_lat)
    alpha = wrap(np.arctan2(py - car.y, px - car.x) - car.yaw)
    steer = np.arctan(2 * WHEELBASE * np.sin(alpha) / look) / MAX_STEER

    # Speed: slow for upcoming bends, the car or pedestrian ahead, and potholes we can't avoid.
    _, _, h = road.pose_at(env.s + np.array([0.0, 15.0, 30.0, 45.0]), 0.0)
    bend = max(np.abs(wrap(h[1:] - h[0])).max() / 45.0, 1e-4)  # rough curvature, 1/m
    target = min(CRUISE, np.sqrt(LAT_ACCEL / bend))
    if np.isfinite(gap):
        target = min(target, np.sqrt(max(0.0, 2 * 0.5 * MAX_BRAKE * (gap - SAFE_GAP))))
    stopping = (car.v ** 2 - POTHOLE_SPEED ** 2) / (2 * BRAKE_PLAN) + 10
    hole = _pothole_ahead(env, min(env.lat, target_lat), max(env.lat, target_lat), max(stopping, 15))
    if np.isfinite(hole):
        target = min(target, POTHOLE_SPEED)
    pedal = np.clip((target - car.v) / (MAX_BRAKE * DT * 3), -1, 1)
    return np.array([np.clip(steer, -1, 1), pedal])


assert POTHOLE_SPEED < POTHOLE_DAMAGE_SPEED
