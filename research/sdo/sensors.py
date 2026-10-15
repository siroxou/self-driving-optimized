"""Switchable sensor suite for MetaDrive.

Each sensor is a small function that reads the ego vehicle and returns a list of
floats in [0, 1]. An observation is the concatenation of the selected sensors in
a fixed order, so the same policy code works for any combination.

See docs/sensors.md (repo root) for how each simulated sensor relates to the real hardware.
"""
import logging

import gymnasium as gym
import numpy as np
from metadrive.envs import MetaDriveEnv
from metadrive.obs.observation_base import BaseObservation
from metadrive.utils.math import clip, norm

# Sensor parameters. These are calibration knobs, not ground truth: change them
# to model cheaper or better hardware and re-run the ablation.
LIDAR_BEAMS, LIDAR_RANGE = 240, 50.0  # roughly a 360-degree spinning unit
SONAR_BEAMS, SONAR_RANGE = 8, 5.0  # ring of ultrasonic parking sensors
RADAR_SWEEP_BEAMS, RADAR_RANGE, RADAR_TRACKS = 72, 80.0, 4
GPS_NOISE = 0.03  # per-step jitter, in normalised navigation units (~1.5 m)
GPS_DRIFT = 0.005  # per-step random-walk step of the slow bias


def _odom(o, v):
    """Wheel odometry + IMU: speed, steering angle, last action, yaw rate."""
    yaw_rate = 0.0
    last, now = v.last_heading_dir, v.heading
    denom = norm(*now) * norm(*last)
    if denom > 0:
        yaw_rate = np.arccos(clip(now.dot(last) / denom, 0.0, 1.0)) / 0.1
    return [
        clip((v.speed_km_h + 1) / (v.max_speed_km_h + 1), 0.0, 1.0),
        clip((v.steering / v.MAX_STEERING + 1) / 2, 0.0, 1.0),
        clip((v.last_current_action[1][0] + 1) / 2, 0.0, 1.0),
        clip((v.last_current_action[1][1] + 1) / 2, 0.0, 1.0),
        clip(yaw_rate, 0.0, 1.0),
    ]


def _lane(o, v):
    """Lane-keeping camera output: distance to both road edges and heading error."""
    road = o.engine.current_map
    width = float((road.MAX_LANE_NUM + 1) * road.MAX_LANE_WIDTH) if road else 100.0
    lanes = v.navigation.current_ref_lanes if v.navigation else None
    heading = v.heading_diff(lanes[-1]) if lanes else 0.5
    return [clip(v.dist_to_left_side / width, 0.0, 1.0), clip(v.dist_to_right_side / width, 0.0, 1.0), heading]


def _nav(o, v):
    """Route guidance with perfect localisation (HD map + exact position)."""
    return list(v.navigation.get_navi_info())


def _gps(o, v):
    """Route guidance from consumer-grade GNSS: jitter plus a slowly drifting bias."""
    o.gps_bias = np.clip(o.gps_bias + o.rng.normal(0, GPS_DRIFT, o.gps_bias.shape), -0.1, 0.1)
    navi = np.asarray(v.navigation.get_navi_info())
    return list(np.clip(navi + o.gps_bias + o.rng.normal(0, GPS_NOISE, navi.shape), 0.0, 1.0))


def _ray(o, v, beams, dist):
    return o.engine.get_sensor("lidar").perceive(
        v, physics_world=o.engine.physics_world.dynamic_world, num_lasers=beams, distance=dist
    )


def _lidar(o, v):
    """Long-range 2D lidar: normalised distance per beam, clockwise from the nose."""
    return list(_ray(o, v, LIDAR_BEAMS, LIDAR_RANGE)[0])


def _sonar(o, v):
    """Ultrasonic ring: a handful of very short beams."""
    return list(_ray(o, v, SONAR_BEAMS, SONAR_RANGE)[0])


def _radar(o, v):
    """Radar tracks: relative position and velocity of the nearest vehicles."""
    _, hits = _ray(o, v, RADAR_SWEEP_BEAMS, RADAR_RANGE)
    lidar = o.engine.get_sensor("lidar")
    return list(lidar.get_surrounding_vehicles_info(v, hits, RADAR_RANGE, RADAR_TRACKS, False))


# name -> (observation size, reader). Order here is the order in the observation.
SENSORS = {
    "odom": (5, _odom),
    "lane": (3, _lane),
    "nav": (10, _nav),
    "gps": (10, _gps),
    "lidar": (LIDAR_BEAMS, _lidar),
    "sonar": (SONAR_BEAMS, _sonar),
    "radar": (RADAR_TRACKS * 4, _radar),
}


def parse(spec):
    """'lidar,nav' -> ['nav', 'lidar'] (canonical order, validated)."""
    names = {s.strip() for s in spec.split(",") if s.strip()}
    unknown = names - SENSORS.keys()
    if unknown or not names:
        raise ValueError(f"unknown or empty sensor set {sorted(unknown)}; choose from {list(SENSORS)}")
    return [n for n in SENSORS if n in names]


def observation_for(sensors):
    """Build a MetaDrive observation class that reads exactly these sensors."""
    sensors = parse(",".join(sensors))

    class SensorObservation(BaseObservation):
        def __init__(self, config):
            super().__init__(config)
            self.rng = np.random.default_rng()
            self.gps_bias = np.zeros(10)

        @property
        def observation_space(self):
            size = sum(SENSORS[n][0] for n in sensors)
            return gym.spaces.Box(0.0, 1.0, shape=(size, ), dtype=np.float32)

        def reset(self, env, vehicle=None):
            self.gps_bias = np.zeros(10)

        def observe(self, vehicle):
            return np.concatenate([SENSORS[n][1](self, vehicle) for n in sensors]).astype(np.float32)

    return SensorObservation


# Training and test maps never overlap, so evaluation measures generalisation.
TRAIN_SEEDS = (0, 500)
TEST_SEEDS = (1000, 100)


def make_env(sensors, seeds=TRAIN_SEEDS, traffic=0.1, render=False):
    start, count = seeds
    return MetaDriveEnv(
        dict(
            agent_observation=observation_for(sensors),
            start_seed=start,
            num_scenarios=count,
            traffic_density=traffic,
            horizon=1500,
            use_render=render,
            log_level=logging.WARNING,
        )
    )
