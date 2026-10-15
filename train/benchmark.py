"""Score a driver on held-out roads, once per sensor set.

    python -m train.benchmark web/models/generalist.npz
    python -m train.benchmark expert --episodes 20
    python -m train.benchmark web/models/specialist.npz --sets "odom,lidar,nav" --out results.md

Uses sim/policy.py (numpy), i.e. exactly what the browser runs.
"""
import argparse
from multiprocessing import Pool

import numpy as np

from sim import expert
from sim.env import DriveEnv
from sim.policy import Policy
from sim.sensors import from_names

TEST_SEEDS = 10_000_000  # first held-out road; training only uses roads below this
SETS = [
    "odom,lane,nav,lidar,sonar,radar",
    "odom,lane,nav,lidar",  # the default set the viewer starts with
    "odom,lidar,nav",
    "odom,lane,nav",
    "odom,lidar,gps",
    "odom,lane",
    "odom,lidar",
    "lidar",
    "odom,sonar,radar,nav",
    "odom,nav",
    "odom",
]


def run(args):
    driver, spec, episodes, scenario = args
    policy = None if driver == "expert" else Policy(driver)
    env = DriveEnv(sensors=from_names(spec), scenario=scenario)
    results = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=TEST_SEEDS + ep)
        done = False
        while not done:
            action = expert.act(env) if policy is None else policy.act(obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        results.append(info)
    mean = lambda k: float(np.mean([r[k] for r in results]))
    return {"sensors": spec, "success": mean("success"), "crash": mean("crash"),
            "out_of_road": mean("out_of_road"), "route_completion": mean("route_completion")}


def benchmark(driver, sets=SETS, episodes=50, scenario=None):
    with Pool(min(len(sets), 12)) as pool:
        return pool.map(run, [(driver, s, episodes, scenario or {}) for s in sets])


def table(rows):
    lines = ["| Sensors | Success | Crash | Off road | Route completion |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['sensors']}` | {r['success']:.0%} | {r['crash']:.0%} | {r['out_of_road']:.0%} | {r['route_completion']:.0%} |")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("driver", help='path to a .npz policy, or "expert"')
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--sets", help="semicolon-separated sensor sets (default: the standard list)")
    p.add_argument("--out", help="also write the table to this Markdown file")
    a = p.parse_args()
    rows = benchmark(a.driver, a.sets.split(";") if a.sets else SETS, a.episodes)
    md = table(rows)
    print(md)
    if a.out:
        open(a.out, "w").write(md + "\n")
