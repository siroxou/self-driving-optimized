import numpy as np
import pytest

from sim import expert
from sim.car import LENGTH
from sim.env import DriveEnv
from sim.sensors import DEFAULTS, LIDAR_BINS, OBS_SIZE, RANGES, SLOTS, from_names, merge
from sim.world import Road, raycast


def slot(obs, name):
    start = 0
    for n, (size, _) in SLOTS.items():
        if n == name:
            return obs[start:start + size + 1]
        start += size + 1


def empty_road_env(sensors=None, seed=0):
    env = DriveEnv(sensors=sensors, scenario={"traffic": 0.0, "lanes": 2})
    env.reset(seed=seed)
    return env


def place_cars(env, ahead, lane):
    """Put parked cars `ahead` metres in front of the ego car, then refresh the observation."""
    t = env.traffic
    t.s, t.lane = env.s + np.asarray(ahead, float), np.full(len(ahead), lane)
    t.v = t.v0 = np.zeros(len(ahead))
    return env._observe()


def ego_lane(env):
    return int(round((env.lat + env.road.width / 2) / 3.5 - 0.5))


def test_raycast_hits_wall_at_right_distance():
    p, q = np.array([[10.0, -5.0]]), np.array([[10.0, 5.0]])
    dist, idx = raycast(np.zeros(2), np.array([0.0, np.pi, np.pi / 4, np.arctan2(4, 10)]), 50, p, q)
    assert dist[0] == pytest.approx(10)
    assert dist[1] == 50 and idx[1] == -1  # facing away
    assert dist[2] == 50  # passes above the wall's end
    assert dist[3] == pytest.approx(np.hypot(10, 4))


def test_roads_never_double_back_and_coordinates_round_trip():
    for seed in range(20):
        road = Road(np.random.default_rng(seed), lanes=3, complexity=1.0)
        assert np.all(np.diff(road.pts[:, 0]) > 0)  # heading stays within +-80 deg, so x always grows
        s = np.array([10.0, road.length / 2, road.length - 10])
        x, y, _ = road.pose_at(s, 2.0)
        for i in range(3):
            _, s2, lat2 = road.project(np.array([x[i], y[i]]))
            assert s2 == pytest.approx(s[i], abs=0.1) and lat2 == pytest.approx(2.0, abs=0.1)


def test_same_seed_same_run():
    def run():
        env = DriveEnv(sensors={"gps": {"on": True}, "nav": {"on": False}, "lidar": {"noise": 0.1}})
        obs, _ = env.reset(seed=7)
        out = [obs]
        for t in range(100):
            out.append(env.step([np.sin(t / 10) * 0.3, 0.6])[0])
        return np.array(out)

    assert np.array_equal(run(), run())


def test_observation_shape_and_range_under_any_settings():
    rng = np.random.default_rng(0)
    env = DriveEnv(scenario={"traffic": 0.3})
    env.reset(seed=1)
    for step in range(300):
        if step % 10 == 0:  # change sensors mid-drive, like the browser does
            update = {}
            for name in DEFAULTS:
                update[name] = {"on": bool(rng.random() < 0.5)}
                for key, (lo, hi) in RANGES[name].items():
                    update[name][key] = rng.uniform(lo, hi)
            env.configure(sensors=update)
        obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
        assert obs.shape == (OBS_SIZE, ) and obs.dtype == np.float32
        assert np.all((obs >= 0) & (obs <= 1))
        if term or trunc:
            env.reset()


def test_switched_off_sensor_slot_is_zero_and_present_flag_set():
    env = empty_road_env(from_names("odom,lidar"))
    obs = env._observe()
    for name in ("lane", "route", "sonar", "radar"):
        assert not slot(obs, name).any()
    for name in ("odom", "lidar"):
        assert slot(obs, name)[-1] == 1.0


def test_lidar_sees_car_ahead_and_binning_keeps_shape():
    env = empty_road_env(from_names("lidar"))
    obs = place_cars(env, [20.0], ego_lane(env))
    lidar = slot(obs, "lidar")
    expected = (20.0 - LENGTH / 2) / DEFAULTS["lidar"]["range"]
    assert lidar[0] == pytest.approx(expected, abs=0.02)  # sector 0 points straight ahead

    env.configure(sensors={"lidar": {"beams": 4}})
    lidar = slot(env._observe(), "lidar")
    assert lidar[LIDAR_BINS:2 * LIDAR_BINS].sum() == 4  # only 4 sectors have data


def test_radar_reports_nearest_and_misses_hidden_car():
    env = empty_road_env(from_names("radar"))
    radar = slot(place_cars(env, [20.0, 40.0], ego_lane(env)), "radar")
    tracks = radar[:40].reshape(8, 5)
    assert tracks[0, 4] == 1.0  # car at 20 m tracked
    assert tracks[1, 4] == 0.0  # car at 40 m hidden behind it


def test_gps_estimate_wanders_but_nav_is_exact():
    env = empty_road_env(merge(from_names("gps"), {"gps": {"jitter": 5.0}}))
    errors = []
    for _ in range(20):
        env.step([0.0, 0.5])
        errors.append(np.linalg.norm(env.debug["gps_estimate"] - env.car.pos))
    assert np.mean(errors) > 1.0

    env.configure(sensors={"nav": {"on": True}})
    env.step([0.0, 0.5])
    assert np.allclose(env.debug["gps_estimate"], env.car.pos)


def test_config_is_validated_and_clamped():
    cfg = merge(DEFAULTS, {"lidar": {"beams": 1000, "range": -5}})
    assert cfg["lidar"]["beams"] == 240 and cfg["lidar"]["range"] == 5.0
    with pytest.raises(ValueError):
        merge(DEFAULTS, {"xray": {"on": True}})
    with pytest.raises(ValueError):
        merge(DEFAULTS, {"lidar": {"colour": 1}})
    with pytest.raises(ValueError):
        DriveEnv(scenario={"snow": 1})


@pytest.mark.parametrize("traffic", [0.0, 0.2])
def test_lane_follower_completes_routes(traffic):
    env = DriveEnv(scenario={"traffic": traffic})
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            *_, term, trunc, info = env.step(expert.act(env))
            done = term or trunc
        assert info["success"], f"seed {seed}: {env.status}"
