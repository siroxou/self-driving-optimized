"""Driving environment with the Gymnasium interface: drive a procedurally generated road to the end without crashing.

    env = DriveEnv(sensors={"lidar": {"beams": 16}}, scenario={"traffic": 0.2})
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step([steer, pedal])
    env.configure(sensors={"gps": {"on": True}, "nav": {"on": False}})  # takes effect next step

Run `python -m sim.env` for a speed check and a lane-follower sanity run.

Gymnasium is only needed for training. The browser (Pyodide) can't import it
because it has no multiprocessing, so without it the env is a plain class with
the same methods.
"""
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None

from .sensors import DEFAULTS, OBS_SIZE, merge, observe
from .car import WIDTH, Car, MAX_SPEED, box_segments, collides
from .hazards import POTHOLE_DAMAGE_SPEED, POTHOLE_SLOWDOWN, Hazards
from .traffic import Traffic
from .world import Road

SCENARIO = {"lanes": 2, "complexity": 0.5, "traffic": 0.1, "parked": 0.2, "pedestrians": 0.0, "obstacles": 0.0, "potholes": 0.0}
SCENARIO_RANGES = {"lanes": (1, 3), "complexity": (0.0, 1.0), "traffic": (0.0, 0.5), "parked": (0.0, 1.0),
                   "pedestrians": (0.0, 20.0), "obstacles": (0.0, 15.0), "potholes": (0.0, 40.0)}  # hazards: per km
HAZARDS = ("pedestrians", "obstacles", "potholes")
HORIZON = 1500  # steps (150 s)
START_S, END_MARGIN = 5.0, 10.0

# Same structure as MetaDrive's default reward, so the two tracks stay comparable.
PROGRESS_REWARD, SPEED_REWARD = 1.0, 0.1
CRASH_PENALTY, OFF_ROAD_PENALTY, SUCCESS_REWARD = 5.0, 5.0, 10.0
POTHOLE_PENALTY = 1.0
OFF_ROAD_TOLERANCE = 0.3  # metres a wheel may stray past the road edge


class DriveEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}
    rng = None

    def __init__(self, sensors=None, scenario=None, horizon=HORIZON):
        self.horizon = horizon  # steps before a run is cut off as a timeout
        self.sensor_cfg = merge(DEFAULTS, sensors)
        self.scenario = dict(SCENARIO)
        self.configure(scenario=scenario)
        if gym:
            self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_SIZE, ), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2, ), np.float32)
        self.debug = {}

    def configure(self, sensors=None, scenario=None):
        """Change sensors (applies from the next step) or the scenario (applies from the next reset)."""
        self.sensor_cfg = merge(self.sensor_cfg, sensors)
        for key, v in (scenario or {}).items():
            if key not in SCENARIO:
                raise ValueError(f"unknown scenario setting {key!r}; choose from {list(SCENARIO)}")
            lo, hi = SCENARIO_RANGES[key]
            self.scenario[key] = int(np.clip(v, lo, hi)) if key == "lanes" else float(np.clip(v, lo, hi))

    def reset(self, seed=None, options=None):
        # Same seeding rule as gymnasium: a seed restarts the generator, no seed carries on.
        if seed is not None or self.rng is None:
            self.rng = np.random.default_rng(seed)
        sc = self.scenario
        self.road = Road(self.rng, sc["lanes"], sc["complexity"])
        x, y, yaw = self.road.pose_at(START_S, self.road.lane_center(self.rng.integers(sc["lanes"])))
        self.car = Car(x, y, yaw)
        self.traffic = Traffic(self.road, self.rng, sc["traffic"], sc["parked"], START_S)
        self.dest = self.road.length - END_MARGIN
        self.hazards = None
        if any(sc[k] > 0 for k in HAZARDS):  # own generator, so hazards never disturb anything else
            parked = self.traffic.s[self.traffic.v0 == 0]
            self.hazards = Hazards(self.road, np.random.default_rng(self.rng.integers(2**63)),
                                   sc["pedestrians"], sc["obstacles"], sc["potholes"], START_S, parked)
        self.potholes_hit = 0
        self.gps_bias = np.zeros(2)
        self.steps = 0
        self.idx, self.s, self.lat = self.road.project(self.car.pos)
        self.status = "driving"
        return self._observe(), self._info()

    def step(self, action):
        self.car.step(action)
        hz = self.hazards
        if hz:
            hz.step(np.append(self.traffic.s[self.traffic.v0 > 0], self.s))  # parked cars don't make pedestrians wait
        self.traffic.step(self.s, self.lat, self.car.v, hz.blockers() if hz else None)
        last_s = self.s
        self.idx, self.s, self.lat = self.road.project(self.car.pos, self.idx)
        self.steps += 1

        x, y, yaw, _, _ = self.traffic.poses()
        reward = PROGRESS_REWARD * (self.s - last_s) + SPEED_REWARD * self.car.v / MAX_SPEED
        hit, holes = hz.contact(self.car) if hz else (None, 0)
        if holes:
            fast = self.car.v > POTHOLE_DAMAGE_SPEED
            self.car.v *= POTHOLE_SLOWDOWN ** holes
            self.potholes_hit += holes
            reward -= POTHOLE_PENALTY * holes
        if collides(self.car, x, y, yaw) or hit == "obstacle":
            self.status, reward = "crash", reward - CRASH_PENALTY
        elif hit == "pedestrian":
            self.status, reward = "hit_pedestrian", reward - CRASH_PENALTY
        elif holes and fast:
            self.status, reward = "damaged", reward - CRASH_PENALTY
        elif abs(self.lat) + WIDTH / 2 > self.road.width / 2 + OFF_ROAD_TOLERANCE:
            self.status, reward = "out_of_road", reward - OFF_ROAD_PENALTY
        elif self.s >= self.dest:
            self.status, reward = "success", reward + SUCCESS_REWARD
        terminated = self.status != "driving"
        truncated = not terminated and self.steps >= self.horizon
        return self._observe(), float(reward), terminated, truncated, self._info()

    def _observe(self):
        self.traffic_pose = self.traffic.poses()
        x, y, yaw, vx, vy = self.traffic_pose
        car_p, car_q = box_segments(x, y, yaw)
        self.n_edges = len(self.road.edge_p)
        self.radar_targets = (x, y, vx, vy)
        if self.hazards:  # pedestrian and obstacle outlines come after the cars, so segment j // 4 still names the target
            hz_p, hz_q = self.hazards.segments()
            car_p, car_q = np.vstack([car_p, hz_p]), np.vstack([car_q, hz_q])
            self.radar_targets = tuple(np.append(a, b) for a, b in zip(self.radar_targets, self.hazards.radar_targets()))
        self.seg_p = np.vstack([self.road.edge_p, car_p])
        self.seg_q = np.vstack([self.road.edge_q, car_q])
        self.seg_dist = np.linalg.norm((self.seg_p + self.seg_q) / 2 - self.car.pos, axis=1)
        obs, self.debug = observe(self, self.sensor_cfg)
        return obs

    def _info(self):
        return {
            "success": self.status == "success",
            "crash": self.status == "crash",
            "out_of_road": self.status == "out_of_road",
            "hit_pedestrian": self.status == "hit_pedestrian",
            "damaged": self.status == "damaged",
            "potholes_hit": self.potholes_hit,
            "route_completion": float(np.clip(self.s / self.dest, 0, 1)),
            "speed": self.car.v,
            "steps": self.steps,
        }



if __name__ == "__main__":
    import time

    from . import expert

    env = DriveEnv(sensors={n: {"on": True} for n in DEFAULTS if n != "gps"})
    env.reset(seed=0)
    start, n = time.perf_counter(), 3000
    for _ in range(n):
        *_, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    print(f"speed, every sensor on: {n / (time.perf_counter() - start):.0f} steps/s")

    for traffic in (0.0, SCENARIO["traffic"], 0.3):
        env.configure(scenario={"traffic": traffic})
        results = []
        for seed in range(100, 130):
            env.reset(seed=seed)
            term = trunc = False
            while not (term or trunc):
                *_, term, trunc, info = env.step(expert.act(env))
            results.append(env.status)
        rate = results.count("success") / len(results)
        print(f"lane follower, traffic {traffic}: {rate:.0%} success, outcomes {sorted(set(results))}")
