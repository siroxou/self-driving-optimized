"""Glue between the simulator and the page. Runs inside Pyodide in the Web Worker.

Every function returns a JSON string so the worker can pass it straight to the
main thread.
"""
import json

import numpy as np

from sim import expert
from sim.env import SCENARIO, SCENARIO_RANGES, DriveEnv
from sim.sensors import DEFAULTS, RANGES
from sim.world import LANE_WIDTH

env = DriveEnv()


def settings():
    """Defaults and allowed ranges, so the page builds its controls from the simulator's own limits."""
    return json.dumps({"defaults": DEFAULTS, "ranges": RANGES, "scenario": SCENARIO, "scenarioRanges": SCENARIO_RANGES})


def _r(a):
    return np.round(np.asarray(a, float), 2).tolist()


def reset(seed, scenario_json="{}"):
    env.configure(scenario=json.loads(scenario_json))
    env.reset(seed=int(seed))
    r = env.road
    every = slice(None, None, 2)
    lane_lines = [(r.pts + r.normal * (-r.width / 2 + LANE_WIDTH * k))[every] for k in range(1, r.lanes)]
    dest = r.pose_at(env.dest, 0.0)
    return json.dumps({
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
    if env.status == "driving":
        env.step(expert.act(env))
    return frame()


def frame():
    car, d, cfg = env.car, env.debug, env.sensor_cfg
    x, y, yaw, _, _ = env.traffic_pose
    out = {
        "ego": [car.x, car.y, car.yaw, car.v],
        "traffic": _r(np.stack([x, y, yaw, env.traffic.v0 == 0], -1)) if len(x) else [],
        "status": env.status if env.steps < 1500 else "timeout",
        "progress": float(np.clip(env.s / env.dest, 0, 1)),
        "steps": env.steps,
    }
    for name in ("lidar", "sonar"):
        if name in d:
            out[name] = [_r(d[name][0]), _r(d[name][1])]
    if "radar" in d:
        out["radar"] = [int(i) for i in d["radar"]]
    if cfg["gps"]["on"] and not cfg["nav"]["on"]:
        out["gps"] = _r(d["gps_estimate"])
    return json.dumps(out)
