"""Train a driving policy with PPO.

    # specialist: one fixed sensor set
    python -m train.train --name specialist --sensors odom,lidar,nav --steps 3000000

    # generalist: sensors and their settings randomised every episode
    python -m train.train --name generalist --random-sensors --steps 20000000

Writes runs/<name>/model.zip (plus checkpoints) and web/models/<name>.npz.
See docs/training.md.
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from sim.env import SCENARIO_RANGES, DriveEnv
from sim.sensors import DEFAULTS, INTEGER, RANGES, from_names
from train.export import export

TRAIN_SEEDS = 10_000_000  # roads 0 .. 9,999,999 are for training; benchmarks use roads above this
FULL_SET_START, FULL_SET_END = 1.0, 0.2  # curriculum: chance of the default full sensor set, start -> end
# Start with less steering noise (std 0.6): full-scale random steering at speed just crashes.
POLICY_KWARGS = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "log_std_init": -0.5}


def sample_sensors(rng, progress):
    """A random sensor setup. Early in training it's usually the full default set; later, mostly random."""
    if rng.random() < FULL_SET_START + (FULL_SET_END - FULL_SET_START) * progress:
        return {name: dict(values) for name, values in DEFAULTS.items()}
    cfg = {}
    for name in DEFAULTS:
        cfg[name] = {"on": bool(rng.random() < 0.5)}
        for key, (lo, hi) in RANGES[name].items():
            v = rng.uniform(lo, hi)
            cfg[name][key] = int(round(v)) if key in INTEGER else v
    return cfg


def sample_scenario(rng):
    lanes = SCENARIO_RANGES["lanes"]
    return {
        "lanes": int(rng.integers(lanes[0], lanes[1] + 1)),
        "complexity": rng.uniform(*SCENARIO_RANGES["complexity"]),
        "traffic": rng.uniform(0.0, 0.3),
        "parked": rng.uniform(0.0, 0.5),
    }


class TrainEnv(gym.Wrapper):
    """Random training roads and scenarios; optionally random sensors."""

    def __init__(self, sensors, randomise, seed):
        super().__init__(DriveEnv(sensors=sensors))
        self.randomise = randomise
        self.rng = np.random.default_rng(seed)
        self.progress = 0.0

    def set_progress(self, progress):
        self.progress = progress

    def reset(self, **kwargs):
        if self.randomise:
            self.env.configure(sensors=sample_sensors(self.rng, self.progress))
        self.env.configure(scenario=sample_scenario(self.rng))
        return self.env.reset(seed=int(self.rng.integers(TRAIN_SEEDS)))

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if term or trunc:
            info["is_success"] = info["success"]  # SB3 logs this as rollout/success_rate
        return obs, reward, term, trunc, info


class Curriculum(BaseCallback):
    def __init__(self, total):
        super().__init__()
        self.total = total

    def _on_rollout_start(self):
        self.training_env.env_method("set_progress", min(self.num_timesteps / self.total, 1.0))

    def _on_step(self):
        return True


def make(sensors, randomise, seed):
    return lambda: TrainEnv(sensors, randomise, seed)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--sensors", default="odom,lane,nav,lidar,sonar,radar", help="fixed sensor set (ignored with --random-sensors)")
    p.add_argument("--random-sensors", action="store_true")
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="continue training from this model.zip")
    p.add_argument("--lr", type=float, default=3e-4, help="starting learning rate, decays linearly to 0")
    a = p.parse_args()

    torch.set_num_threads(4)
    out = Path("runs") / a.name
    sensors = from_names(a.sensors)
    env = VecMonitor(SubprocVecEnv([make(sensors, a.random_sensors, a.seed * 1000 + i) for i in range(a.envs)]))
    # Returns reach several hundred; scaling rewards keeps the value loss sane. Observations are already 0..1.
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)
    if a.resume:
        model = PPO.load(a.resume, env=env, device="cpu", custom_objects={
            "learning_rate": lambda left: a.lr * left, "target_kl": 0.02, "n_steps": 512, "batch_size": 1024})
    else:
        model = PPO(
            "MlpPolicy", env, n_steps=512, batch_size=1024, n_epochs=10, learning_rate=lambda left: a.lr * left,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, target_kl=0.02, seed=a.seed, device="cpu",
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
        )
    model.set_logger(configure(str(out / "logs"), ["stdout", "csv"]))  # logs/progress.csv = learning curve
    callbacks = [Curriculum(a.steps), CheckpointCallback(max(1_000_000 // a.envs, 1), str(out / "checkpoints"), "model")]
    try:
        model.learn(a.steps, callback=callbacks, reset_num_timesteps=not a.resume, progress_bar=False)
    finally:
        model.save(out / "model")
        Path("web/models").mkdir(parents=True, exist_ok=True)
        export(model, f"web/models/{a.name}.npz")
        env.close()
    print(f"saved runs/{a.name}/model.zip and web/models/{a.name}.npz")


if __name__ == "__main__":
    main()
