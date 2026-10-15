"""Evaluate a trained run on held-out maps it never saw during training.

    python evaluate.py runs/lidar+nav+odom-s0 --episodes 50
    python evaluate.py runs/lidar+nav+odom-s0 --episodes 3 --render
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from sdo.sensors import TEST_SEEDS, make_env


def evaluate(run, episodes=50, render=False):
    run = Path(run)
    sensors = json.loads((run / "run.json").read_text())["sensors"]
    model = PPO.load(run / "model")
    env = make_env(sensors, seeds=TEST_SEEDS, render=render)
    rows = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=TEST_SEEDS[0] + ep % TEST_SEEDS[1])
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            rows.append(info)
    finally:
        env.close()
    return {
        "sensors": "+".join(sensors),
        "obs_dim": int(model.observation_space.shape[0]),
        "success": np.mean([r["arrive_dest"] for r in rows]),
        "crash": np.mean([r["crash"] for r in rows]),
        "out_of_road": np.mean([r["out_of_road"] for r in rows]),
        "route_completion": np.mean([r["route_completion"] for r in rows]),
        "episodes": episodes,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--render", action="store_true", help="open a window and watch it drive")
    a = p.parse_args()
    for k, v in evaluate(a.run, a.episodes, a.render).items():
        print(f"{k:>17}: {v:.3f}" if isinstance(v, float) else f"{k:>17}: {v}")
