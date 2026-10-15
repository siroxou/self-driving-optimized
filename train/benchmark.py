"""Score a driver on held-out roads, once per sensor set.

    python -m train.benchmark web/models/generalist.npz
    python -m train.benchmark expert --episodes 20
    python -m train.benchmark web/models/specialist-odom-nav-lidar.npz --sets "odom,lidar,nav" --out results.md
    python -m train.benchmark auto --level gauntlet     # the viewer's Auto choice per sensor set, on a level

Uses sim/policy.py (numpy), i.e. exactly what the browser runs.
"""
import argparse
from multiprocessing import Pool

import numpy as np

import json

from sim import expert
from sim.env import DriveEnv
from sim.levels import LEVELS
from sim.policy import Policy
from sim.sensors import DEFAULTS, from_names

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


def auto_model(spec):
    """The model the viewer's Auto driver uses for this sensor set."""
    cfg = from_names(spec)
    key = "-".join(n for n in DEFAULTS if cfg[n]["on"])
    manifest = {"-".join(e["sensors"]): e["model"] for e in json.load(open("web/models/specialists.json"))}
    return f"web/models/{manifest.get(key, 'generalist')}.npz"


def run(args):
    """Score one driver on one sensor set. `episodes` is a count (standard roads) or a level name."""
    driver, spec, episodes, scenario = args
    if driver == "auto":
        driver = auto_model(spec)
    policy = None if driver == "expert" else Policy(driver)
    if isinstance(episodes, str):
        level = LEVELS[episodes]
        env = DriveEnv(sensors=from_names(spec), scenario=level["scenario"], horizon=level["horizon"])
        seeds = level["seeds"]
    else:
        env = DriveEnv(sensors=from_names(spec), scenario=scenario)
        seeds = [TEST_SEEDS + ep for ep in range(episodes)]
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = expert.act(env) if policy is None else policy.act(obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        results.append(info)
    mean = lambda k: float(np.mean([r.get(k, 0) for r in results]))
    return {"sensors": spec, "success": mean("success"), "crash": mean("crash"),
            "out_of_road": mean("out_of_road"), "route_completion": mean("route_completion"),
            "hit_pedestrian": mean("hit_pedestrian"), "damaged": mean("damaged"), "potholes_hit": mean("potholes_hit"),
            "timeout": float(np.mean([not (r["success"] or r["crash"] or r["out_of_road"] or r.get("hit_pedestrian")
                                           or r.get("damaged")) for r in results]))}


def benchmark(driver, sets=SETS, episodes=50, scenario=None):
    with Pool(min(len(sets), 12)) as pool:
        return pool.map(run, [(driver, s, episodes, scenario or {}) for s in sets])


def table(rows, hazards=False):
    head = "| Sensors | Success | Crash | Off road | Route completion |"
    if hazards:
        head = "| Sensors | Success | Crash | Off road | Pedestrian hit | Pothole damage | Out of time | Route completion | Potholes per run |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        cells = [f"`{r['sensors']}`", f"{r['success']:.0%}", f"{r['crash']:.0%}", f"{r['out_of_road']:.0%}"]
        if hazards:
            cells += [f"{r['hit_pedestrian']:.0%}", f"{r['damaged']:.0%}", f"{r['timeout']:.0%}"]
        cells.append(f"{r['route_completion']:.0%}")
        if hazards:
            cells.append(f"{r['potholes_hit']:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("driver", help='path to a .npz policy, "expert", or "auto" (the viewer\'s choice per set)')
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--level", choices=list(LEVELS), help="score on an evaluation level's fixed roads instead")
    p.add_argument("--sets", help="semicolon-separated sensor sets (default: the standard list)")
    p.add_argument("--out", help="also write the table to this Markdown file")
    a = p.parse_args()
    rows = benchmark(a.driver, a.sets.split(";") if a.sets else SETS, a.level or a.episodes)
    md = table(rows, hazards=bool(a.level))
    print(md)
    if a.out:
        open(a.out, "w").write(md + "\n")
