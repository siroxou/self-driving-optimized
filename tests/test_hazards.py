import numpy as np
import pytest

from sim import expert
from sim.car import LENGTH
from sim.env import DriveEnv
from sim.hazards import POTHOLE_DAMAGE_SPEED
from sim.levels import LEVELS
from sim.sensors import LIDAR_BINS, SLOTS, from_names


def slot(obs, name):
    start = 0
    for n, (size, _) in SLOTS.items():
        if n == name:
            return obs[start:start + size + 1]
        start += size + 1


def hazard_env(sensors="odom,lane,nav,lidar,radar", **hazards):
    env = DriveEnv(sensors=from_names(sensors), scenario={"traffic": 0.0, "lanes": 2, **hazards})
    env.reset(seed=1)
    return env


def put_pedestrian(env, ahead, lat, crossing=True):
    hz = env.hazards
    hz.ped_s[:] = env.s + 400  # park the others far away
    hz.ped_s[0], hz.ped_lat[0], hz.ped_speed[0] = env.s + ahead, lat, 0.0
    hz.ped_wait[0] = -1.0 if crossing else 100.0
    hz.refresh()


def put_pothole(env, ahead, lat):
    hz = env.hazards
    hz.hole_s, hz.hole_lat, hz.hole_r = np.array([env.s + ahead]), np.array([lat]), np.array([0.6])
    hz.hole_hit = np.zeros(1, bool)
    hz.refresh()


def test_no_hazards_means_no_hazard_object():
    env = DriveEnv()
    env.reset(seed=0)
    assert env.hazards is None


def test_hazards_are_deterministic():
    def run():
        env = DriveEnv(scenario=LEVELS["gauntlet"]["scenario"])
        obs, _ = env.reset(seed=5)
        out = [obs]
        for _ in range(150):
            out.append(env.step(expert.act(env))[0])
        return np.array(out), env.hazards.ped_lat.copy()

    (a, pa), (b, pb) = run(), run()
    assert np.array_equal(a, b) and np.array_equal(pa, pb)


def test_driving_into_a_pedestrian_ends_the_run():
    env = hazard_env(pedestrians=5)
    put_pedestrian(env, 12, env.lat)
    for _ in range(60):
        *_, term, _, info = env.step([0.0, 1.0])
        if term:
            break
    assert env.status == "hit_pedestrian" and info["hit_pedestrian"]


def test_fast_pothole_damages_slow_one_only_costs_speed():
    env = hazard_env(potholes=5)
    put_pothole(env, 6, env.lat)
    env.car.v = POTHOLE_DAMAGE_SPEED + 3
    for _ in range(10):
        *_, term, _, info = env.step([0.0, 0.0])
        if term:
            break
    assert env.status == "damaged"

    env = hazard_env(potholes=5)
    put_pothole(env, 6, env.lat)
    env.car.v = 5.0
    for _ in range(25):
        *_, info = env.step([0.0, 0.0])
    assert env.status == "driving" and info["potholes_hit"] == 1 and env.car.v < 5.0


def test_lidar_sees_debris_and_pedestrians_radar_tracks_pedestrians():
    env = hazard_env(obstacles=10)
    hz = env.hazards
    hz.obs_s, hz.obs_lat, hz.obs_size = np.array([env.s + 15]), np.array([env.lat]), np.array([1.4])
    hz.refresh()
    lidar = slot(env._observe(), "lidar")
    assert lidar[0] == pytest.approx((15 - 0.7) / 50, abs=0.03)

    env = hazard_env(pedestrians=5)
    put_pedestrian(env, 20, env.lat)
    obs = env._observe()
    assert slot(obs, "lidar")[0] < 0.45
    assert slot(obs, "radar")[:40].reshape(8, 5)[0, 4] == 1.0
    assert len(env.debug["radar"]) == 1


def test_traffic_waits_for_a_crossing_pedestrian_then_goes():
    env = DriveEnv(scenario={"traffic": 0.0, "lanes": 2, "pedestrians": 5})
    env.reset(seed=2)
    hz, t = env.hazards, env.traffic
    crosswalk = hz.crosswalks[-1]
    t.s, t.lane, t.v, t.v0 = np.array([crosswalk - 60]), np.array([1]), np.array([10.0]), np.array([10.0])
    hz.ped_s[:] = crosswalk
    hz.ped_lat[:], hz.ped_speed[:], hz.ped_wait[:] = 0.0, 0.0, -1.0  # standing in the middle of the road
    hz.refresh()
    for _ in range(150):
        env.step([0.0, -1.0])
    assert t.s[0] < crosswalk - LENGTH and t.v[0] < 0.5

    hz.ped_lat[:], hz.ped_wait[:] = hz.kerb, 100.0  # back on the kerb
    for _ in range(100):
        env.step([0.0, -1.0])
    assert t.s[0] > crosswalk


@pytest.mark.parametrize("level, seeds, need", [("commute", 5, 5), ("gauntlet", 8, 7)])
def test_reference_driver_gets_through_levels(level, seeds, need):
    spec = LEVELS[level]
    env = DriveEnv(scenario=spec["scenario"], horizon=spec["horizon"])
    wins = 0
    for seed in spec["seeds"][:seeds]:
        env.reset(seed=seed)
        done = False
        while not done:
            *_, term, trunc, info = env.step(expert.act(env))
            done = term or trunc
        wins += info["success"]
    assert wins >= need


def test_level_roads_are_held_out():
    for spec in LEVELS.values():
        assert min(spec["seeds"]) >= 20_000_000
    assert not set(LEVELS["commute"]["seeds"]) & set(LEVELS["gauntlet"]["seeds"])
