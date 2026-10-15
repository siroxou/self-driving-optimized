"""The seven switchable sensors and the fixed observation layout.

Every sensor owns a slot of fixed size in the observation, whatever its
settings, so one policy works for any combination the user picks:

    [ readings ... | present flag | settings normalised to 0..1 ]

A switched-off sensor's slot is all zeros. `nav` and `gps` share the route
slot: `nav` is exact, `gps` is the same guidance from a noisy position.

Sensor definitions match the research track; see docs/sensors.md.
"""
from copy import deepcopy

import numpy as np

from .car import DT, MAX_SPEED, MAX_STEER
from .world import raycast, wrap

LIDAR_BINS, SONAR_BINS, RADAR_SLOTS = 72, 16, 8
SONAR_CONE = np.radians([-10, 0, 10])  # each ultrasonic beam is three rays spread over a cone
GPS_MAX_BIAS = 10.0  # metres
LOOKAHEAD = np.array([5.0, 15.0, 30.0, 50.0])  # route points the navigation reports, metres ahead

DEFAULTS = {
    "odom": {"on": True, "noise": 0.0},
    "lane": {"on": True, "noise": 0.0},
    "nav": {"on": True},
    "gps": {"on": False, "jitter": 1.5, "drift": 0.1},
    "lidar": {"on": True, "beams": 240, "range": 50.0, "noise": 0.0, "dropout": 0.0},
    "sonar": {"on": False, "beams": 8, "range": 5.0},
    "radar": {"on": False, "tracks": 4, "range": 80.0},
}

# Allowed range of every setting. Drives normalisation, UI sliders and training randomisation.
RANGES = {
    "odom": {"noise": (0.0, 0.3)},  # std dev in normalised units
    "lane": {"noise": (0.0, 2.0)},  # metres
    "nav": {},
    "gps": {"jitter": (0.0, 10.0), "drift": (0.0, 1.0)},  # metres, metres per step
    "lidar": {"beams": (4, 240), "range": (5.0, 100.0), "noise": (0.0, 0.2), "dropout": (0.0, 0.9)},
    "sonar": {"beams": (2, 16), "range": (1.0, 10.0)},
    "radar": {"tracks": (1, RADAR_SLOTS), "range": (20.0, 200.0)},
}
INTEGER = {"beams", "tracks"}


def merge(base, update=None):
    """Return a copy of a sensor config with `update` applied. Values are clamped to RANGES."""
    cfg = deepcopy(base)
    for name, values in (update or {}).items():
        if name not in cfg:
            raise ValueError(f"unknown sensor {name!r}; choose from {list(cfg)}")
        for key, v in values.items():
            if key not in cfg[name]:
                raise ValueError(f"unknown setting {name}.{key}")
            if key == "on":
                cfg[name][key] = bool(v)
            else:
                lo, hi = RANGES[name][key]
                v = float(np.clip(v, lo, hi))
                cfg[name][key] = int(round(v)) if key in INTEGER else v
    return cfg


def from_names(spec):
    """'odom,lidar,nav' -> config with exactly those sensors on, default settings."""
    names = {s.strip() for s in spec.split(",") if s.strip()}
    if not names or names - DEFAULTS.keys():
        raise ValueError(f"bad sensor set {spec!r}; choose from {list(DEFAULTS)}")
    return merge(DEFAULTS, {n: {"on": n in names} for n in DEFAULTS})


def _norm(name, key, value):
    lo, hi = RANGES[name][key]
    return (value - lo) / (hi - lo)


def _to_car_frame(vec, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]], -1)


def _bin(rel_angles, dist, keep, bins):
    """Shortest reading per angular sector, plus a mask of sectors that got any reading."""
    k = (rel_angles % (2 * np.pi) / (2 * np.pi) * bins).astype(int) % bins
    values, mask = np.ones(bins), np.zeros(bins)
    np.minimum.at(values, k[keep], dist[keep])
    mask[k[keep]] = 1.0
    return values, mask


def _scan(env, angles, max_range):
    near = env.seg_dist < max_range + 3.0
    return raycast(env.car.pos, angles, max_range, env.seg_p[near], env.seg_q[near])[0]


def _odom(env, c):
    car = env.car
    yaw_rate = wrap(car.yaw - car.last_yaw) / DT
    r = np.array([
        car.v / MAX_SPEED,
        (car.steer / MAX_STEER + 1) / 2,
        (car.action[1] + 1) / 2,
        (car.action[0] + 1) / 2,
        (yaw_rate + 1) / 2,  # +-1 rad/s covers normal driving
    ])
    r = r + env.rng.normal(0, c["noise"], r.shape)
    return [r, [_norm("odom", "noise", c["noise"])]], {}


def _lane(env, c):
    w = env.road.width
    lat = env.lat + env.rng.normal(0, c["noise"])
    err = wrap(env.car.yaw - env.road.heading[env.idx]) + env.rng.normal(0, 0.05 * c["noise"])
    return [[(w / 2 - lat) / w, (w / 2 + lat) / w, (err / np.pi + 1) / 2], [_norm("lane", "noise", c["noise"])]], {}


def _route(env, nav, gps):
    """Route guidance: where the road goes next, as seen from where the car thinks it is."""
    car, road = env.car, env.road
    est = car.pos
    if not nav["on"]:  # gps only: jitter plus a slowly wandering bias
        env.gps_bias = env.gps_bias + env.rng.normal(0, gps["drift"], 2)
        size = np.linalg.norm(env.gps_bias)
        if size > GPS_MAX_BIAS:
            env.gps_bias *= GPS_MAX_BIAS / size
        est = est + env.gps_bias + env.rng.normal(0, gps["jitter"], 2)
    _, s_est, _ = road.project(est, env.idx)
    x, y, _ = road.pose_at(s_est + LOOKAHEAD, 0.0)
    rel = _to_car_frame(np.stack([x, y], -1) - est, car.yaw)
    readings = [(rel.ravel() / 50 + 1) / 2, [s_est / env.dest, road.lanes / 3]]
    settings = [float(nav["on"]), 0.0, 0.0] if nav["on"] else \
        [0.0, _norm("gps", "jitter", gps["jitter"]), _norm("gps", "drift", gps["drift"])]
    return readings + [settings], {"gps_estimate": est}


def _lidar(env, c):
    beams = c["beams"]
    rel = np.arange(beams) * 2 * np.pi / beams
    dist = _scan(env, env.car.yaw + rel, c["range"])
    norm = dist / c["range"] + env.rng.normal(0, c["noise"], beams)
    keep = env.rng.random(beams) >= c["dropout"]
    values, mask = _bin(rel, np.clip(norm, 0, 1), keep, LIDAR_BINS)
    settings = [_norm("lidar", k, c[k]) for k in ("beams", "range", "noise", "dropout")]
    return [values, mask, settings], {"lidar": (env.car.yaw + rel[keep], dist[keep])}


def _sonar(env, c):
    beams = c["beams"]
    rel = np.arange(beams) * 2 * np.pi / beams
    cone = (rel[:, None] + SONAR_CONE[None, :]).ravel()
    dist = _scan(env, env.car.yaw + cone, c["range"]).reshape(beams, -1).min(1)
    values, mask = _bin(rel, dist / c["range"], np.ones(beams, bool), SONAR_BINS)
    settings = [_norm("sonar", k, c[k]) for k in ("beams", "range")]
    return [values, mask, settings], {"sonar": (env.car.yaw + rel, dist)}


def _radar(env, c):
    tracks = np.zeros((RADAR_SLOTS, 5))
    settings = [_norm("radar", k, c[k]) for k in ("tracks", "range")]
    x, y, vx, vy = env.radar_targets  # traffic, then pedestrians
    rel = np.stack([x, y], -1) - env.car.pos
    dist = np.linalg.norm(rel, axis=-1)
    cand = np.flatnonzero(dist < c["range"])
    cand = cand[np.argsort(dist[cand])]
    if len(cand):
        # A target is visible if a ray towards its centre hits it before any other car.
        angles = np.arctan2(rel[cand, 1], rel[cand, 0])
        _, seg = raycast(env.car.pos, angles, c["range"], env.seg_p[env.n_edges:], env.seg_q[env.n_edges:])
        cand = cand[seg // 4 == cand][:c["tracks"]]
    ego_v = env.car.v * np.array([np.cos(env.car.yaw), np.sin(env.car.yaw)])
    pos = _to_car_frame(rel[cand], env.car.yaw) / c["range"]
    vel = _to_car_frame(np.stack([vx, vy], -1)[cand] - ego_v, env.car.yaw) / (2 * MAX_SPEED)
    tracks[:len(cand), :4] = (np.hstack([pos, vel]) + 1) / 2
    tracks[:len(cand), 4] = 1.0
    return [tracks.ravel(), settings], {"radar": np.stack([x, y], -1)[cand]}


# slot name -> (readings + settings size, reader). The present flag adds one more value.
SLOTS = {
    "odom": (5 + 1, _odom),
    "lane": (3 + 1, _lane),
    "route": (10 + 3, None),
    "lidar": (2 * LIDAR_BINS + 4, _lidar),
    "sonar": (2 * SONAR_BINS + 2, _sonar),
    "radar": (5 * RADAR_SLOTS + 2, _radar),
}
OBS_SIZE = sum(size + 1 for size, _ in SLOTS.values())


def observe(env, cfg):
    """Build the observation for the current state. Returns (obs, debug geometry for rendering)."""
    parts, debug = [], {}
    for name, (size, read) in SLOTS.items():
        if name == "route":
            on = cfg["nav"]["on"] or cfg["gps"]["on"]
            if on:
                chunks, dbg = _route(env, cfg["nav"], cfg["gps"])
        else:
            on = cfg[name]["on"]
            if on:
                chunks, dbg = read(env, cfg[name])
        if on:
            slot = np.concatenate([np.ravel(c) for c in chunks] + [[1.0]])
            debug.update(dbg)
        else:
            slot = np.zeros(size + 1)
        parts.append(slot)
    return np.clip(np.concatenate(parts), 0, 1).astype(np.float32), debug
