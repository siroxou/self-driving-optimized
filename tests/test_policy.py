import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402

from sim.env import DriveEnv  # noqa: E402
from sim.policy import Policy  # noqa: E402
from train.export import export  # noqa: E402


def test_numpy_policy_matches_torch(tmp_path):
    env = DriveEnv()
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": {"pi": [64, 32], "vf": [16]}}, device="cpu", seed=0)
    export(model, tmp_path / "p.npz")
    policy = Policy(tmp_path / "p.npz")
    obs = np.random.default_rng(0).random((50, env.observation_space.shape[0]), dtype=np.float32)
    expected, _ = model.predict(obs, deterministic=True)
    got = np.array([policy.act(o) for o in obs])
    assert np.allclose(got, expected, atol=1e-5)
