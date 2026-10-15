"""Train one specialist per sensor set, then write the manifest the viewer uses to pick a model.

For each set the manifest names whichever model scored higher on it, the specialist or the
generalist: some specialist runs land on a worse policy than the generalist already has.

    python -m train.specialists              # train any missing ones, benchmark all, write manifest
    python -m train.specialists --manifest   # only re-benchmark and rewrite the manifest

Each specialist is a normal `train.train` run on a fixed sensor set (about 5 minutes).
Models go to web/models/specialist-<sensors>.npz; the manifest is web/models/specialists.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from sim.sensors import DEFAULTS, from_names
from train.benchmark import run

# A handful of sets that tell a story: lidar-based, camera-based, radar-based, and everything.
SETS = [
    "odom,lane,nav,lidar",  # the viewer's default
    "odom,lidar,nav",
    "odom,lidar,gps",
    "odom,lane,nav,lidar,sonar,radar",
    "odom,lane,nav,radar",  # camera + radar, the usual production pairing
    "odom,lane,nav",  # camera only
    "odom,sonar,radar,nav",  # no lidar, no camera
    "lidar",
]
STEPS = 3_000_000
MODELS = Path("web/models")


def key(spec):
    """Canonical name for a sensor set, in DEFAULTS order: 'lidar,odom,nav' -> 'odom-nav-lidar'."""
    cfg = from_names(spec)
    return "-".join(n for n in DEFAULTS if cfg[n]["on"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", action="store_true", help="skip training")
    p.add_argument("--episodes", type=int, default=50)
    a = p.parse_args()

    for spec in SETS:
        name = f"specialist-{key(spec)}"
        if a.manifest or (MODELS / f"{name}.npz").exists():
            continue
        print(f"training {name}", flush=True)
        subprocess.run([sys.executable, "-m", "train.train", "--name", name, "--sensors", spec, "--steps", str(STEPS)],
                       check=True, stdout=subprocess.DEVNULL)

    entries = []
    for spec in SETS:
        name = f"specialist-{key(spec)}"
        own = run((str(MODELS / f"{name}.npz"), spec, a.episodes, {}))["success"]
        general = run((str(MODELS / "generalist.npz"), spec, a.episodes, {}))["success"]
        best = name if own >= general else "generalist"
        entries.append({"sensors": key(spec).split("-"), "model": best, "success": round(max(own, general), 2),
                        "specialist": round(own, 2), "generalist": round(general, 2)})
        print(f"{key(spec)}: specialist {own:.0%}, generalist {general:.0%} -> {best}", flush=True)
    (MODELS / "specialists.json").write_text(json.dumps(entries, indent=2) + "\n")
    print("wrote", MODELS / "specialists.json")


if __name__ == "__main__":
    main()
