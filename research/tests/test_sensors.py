import numpy as np
import pytest

from sdo.sensors import SENSORS, make_env, parse


def test_parse_orders_and_validates():
    assert parse("lidar, odom") == ["odom", "lidar"]
    with pytest.raises(ValueError):
        parse("lidar,xray")
    with pytest.raises(ValueError):
        parse("")


def test_every_sensor_fits_its_space():
    # MetaDrive allows one engine per process, so check all sensors in a single env.
    env = make_env(list(SENSORS))
    try:
        obs, _ = env.reset()
        for _ in range(10):
            obs, *_ = env.step([0.0, 0.5])
        assert obs.shape == env.observation_space.shape == (sum(d for d, _ in SENSORS.values()), )
        assert np.all((obs >= 0) & (obs <= 1))
    finally:
        env.close()
