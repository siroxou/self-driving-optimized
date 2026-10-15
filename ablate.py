"""Train and evaluate every sensor set, then write a results table.

    python ablate.py                                   # default grid, 300k steps, 3 seeds
    python ablate.py --steps 2000 --episodes 2 --seeds 1 --sets "odom,nav;lidar,nav,odom"

Writes results/results.csv (one row per seed) and results/results.md (mean over seeds).
"""
import argparse
from pathlib import Path

import pandas as pd

from evaluate import evaluate
from sdo.sensors import parse
from train import train

# From "almost blind" to "everything on". Edit freely.
DEFAULT_SETS = [
    "odom,nav",
    "odom,gps",
    "odom,lane,nav",
    "odom,sonar,nav",
    "odom,radar,nav",
    "odom,lidar,nav",
    "odom,lidar,gps",
    "odom,lidar",
    "lidar,nav",
    "odom,lane,radar,sonar,gps",
    "odom,lane,lidar,radar,sonar,nav",
]


def main(sets, steps, episodes, seeds, out):
    out = Path(out)
    out.mkdir(exist_ok=True)
    rows = []
    for spec in sets:
        sensors = parse(spec)
        for seed in range(seeds):
            run = train(sensors, steps, f"runs/{'+'.join(sensors)}-s{seed}", seed)
            rows.append({**evaluate(run, episodes), "seed": seed, "steps": steps})
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)  # save progress after every run
    df = pd.DataFrame(rows)
    table = df.groupby(["sensors", "obs_dim"], as_index=False)[["success", "crash", "out_of_road", "route_completion"]]
    table = table.mean().sort_values("success", ascending=False)
    (out / "results.md").write_text(table.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", help='semicolon-separated sensor sets, e.g. "odom,nav;lidar,nav"')
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    main(a.sets.split(";") if a.sets else DEFAULT_SETS, a.steps, a.episodes, a.seeds, a.out)
