"""Train a PPO driving policy on a chosen sensor set.

    python train.py --sensors lidar,nav,odom --steps 300000
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from sdo.sensors import make_env, parse


def train(sensors, steps, out, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(sensors)
    try:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, seed=seed, verbose=1)
        model.learn(total_timesteps=steps)
        model.save(out / "model")
    finally:
        env.close()
    (out / "run.json").write_text(json.dumps({"sensors": sensors, "steps": steps, "seed": seed}, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensors", required=True, help="comma-separated, e.g. lidar,nav,odom")
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="output folder (default: runs/<sensors>-s<seed>)")
    a = p.parse_args()
    sensors = parse(a.sensors)
    print("saved to", train(sensors, a.steps, a.out or f"runs/{'+'.join(sensors)}-s{a.seed}", a.seed))
