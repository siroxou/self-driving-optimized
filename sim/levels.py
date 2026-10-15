"""Evaluation levels: fixed roads with pedestrians, debris and potholes on top of traffic.

Each level is a scenario plus a fixed list of road numbers, so every driver faces
exactly the same roads. Road numbers start at 20,000,000, clear of training
(below 10,000,000) and the standard benchmark (from 10,000,000). See docs/levels.md.
"""

LEVELS = {
    "commute": {
        "title": "Commute",
        "description": "Light traffic, a few crossings, the odd cone and pothole.",
        "scenario": {"lanes": 2, "complexity": 0.4, "traffic": 0.1, "parked": 0.2,
                     "pedestrians": 4, "obstacles": 2, "potholes": 5},
        "seeds": list(range(20_000_000, 20_000_050)),
        "horizon": 1500,  # steps: 150 s
    },
    "gauntlet": {
        "title": "Gauntlet",
        "description": "Three busy lanes, sharp bends, crossings every 80 m, debris and a pothole every 50 m.",
        "scenario": {"lanes": 3, "complexity": 0.8, "traffic": 0.25, "parked": 0.4,
                     "pedestrians": 12, "obstacles": 8, "potholes": 20},
        "seeds": list(range(20_100_000, 20_100_050)),
        "horizon": 2400,  # steps: 4 minutes, it stops a lot
    },
}
