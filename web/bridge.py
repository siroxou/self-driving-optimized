"""Glue between the simulator and the page. Runs inside Pyodide in the Web Worker.

Every function returns a JSON string so the worker can pass it straight to the
main thread.
"""
import json

import numpy as np

from sim import expert
from sim.env import HORIZON, SCENARIO, SCENARIO_RANGES, DriveEnv
from sim.levels import LEVELS
from sim.policy import Policy
from sim.sensors import DEFAULTS, RANGES
from sim.world import LANE_WIDTH

env = DriveEnv()
policy = None  # None = reference lane follower
obs = None


def load_policy(path=None):
    """Drive with the exported model at `path`, or the reference lane follower if None."""
    global policy
    policy = Policy(path) if path else None


def settings():
    """Defaults and allowed ranges, so the page builds its controls from the simulator's own limits."""
    return json.dumps({"defaults": DEFAULTS, "ranges": RANGES, "scenario": SCENARIO, "scenarioRanges": SCENARIO_RANGES,
                       "levels": LEVELS})


def _r(a):
    return np.round(np.asarray(a, float), 2).tolist()


def reset(seed, scenario_json="{}", horizon=HORIZON):
    global obs
    env.configure(scenario=json.loads(scenario_json))
    env.horizon = int(horizon)
    obs, _ = env.reset(seed=int(seed))
    r = env.road
    every = slice(None, None, 2)
    lane_lines = [(r.pts + r.normal * (-r.width / 2 + LANE_WIDTH * k))[every] for k in range(1, r.lanes)]
    dest = r.pose_at(env.dest, 0.0)
    hz = env.hazards
    static = {"crosswalks": [], "obstacles": [], "potholes": []}
    if hz:
        x, y, h = r.pose_at(hz.crosswalks, 0.0)
        static = {
            "crosswalks": _r(np.stack([x, y, h], -1)),
            "obstacles": _r(np.stack([hz.obs_x, hz.obs_y, hz.obs_yaw, hz.obs_size], -1)),
            "potholes": _r(np.stack([hz.hole_x, hz.hole_y, hz.hole_r], -1)),
        }
    return json.dumps({
        **static,
        "left": _r((r.pts + r.normal * r.width / 2)[every]),
        "right": _r((r.pts - r.normal * r.width / 2)[every]),
        "lanes": [_r(line) for line in lane_lines],
        "width": r.width,
        "dest": [float(dest[0]), float(dest[1]), float(dest[2])],
        "length": float(r.length),
        "frame": json.loads(frame()),
    })


def configure(sensors_json):
    env.configure(sensors=json.loads(sensors_json))


def step():
    global obs
    if env.status == "driving":
        obs, *_ = env.step(policy.act(obs) if policy else expert.act(env))
    return frame()


def frame():
    car, d, cfg = env.car, env.debug, env.sensor_cfg
    x, y, yaw, _, _ = env.traffic_pose
    out = {
        "ego": [car.x, car.y, car.yaw, car.v],
        "traffic": _r(np.stack([x, y, yaw, env.traffic.v0 == 0], -1)) if len(x) else [],
        "status": env.status if env.status != "driving" or env.steps < env.horizon else "timeout",
        "potholes": env.potholes_hit,
        "progress": float(np.clip(env.s / env.dest, 0, 1)),
        "steps": env.steps,
    }
    for name in ("lidar", "sonar"):
        if name in d:
            out[name] = [_r(d[name][0]), _r(d[name][1])]
    if "radar" in d:
        out["radar"] = _r(d["radar"])
    hz = env.hazards
    if hz is not None:
        out["pedestrians"] = _r(np.stack([hz.ped_x, hz.ped_y, hz.ped_yaw, hz.ped_wait <= 0], -1))
        out["holesHit"] = np.flatnonzero(hz.hole_hit).tolist()
    if cfg["gps"]["on"] and not cfg["nav"]["on"]:
        out["gps"] = _r(d["gps_estimate"])
    return json.dumps(out)
