# Roadmap

The end goal is a public web page where anyone can switch off a self-driving car's senses, tune what's left, and watch what happens. That page is the answer to the project's question: how little does a self-driving car need to sense?

There are two tracks:

- **Browser playground** (main): our own lightweight driving sim written in Python + numpy. It trains offline and runs in the browser through Pyodide, with 3D rendering in three.js. It's a static site, so there's no server and anyone can open it.
- **Research track** (`research/`): the same sensor study on MetaDrive, a higher-fidelity simulator. It's used to check that the browser findings hold up.

Each phase ends with an exit check that has to pass before the next one starts.

## Phase 0: Restructure ✓

- [x] Move the MetaDrive study into `research/`
- [x] Add the `sim/` package skeleton
- [x] Write the browser architecture doc ([docs/web.md](docs/web.md))

**Exit:** research tests pass from `research/`, and the research scripts still run.

## Phase 1: Sim core ← exit check passed, in review

- [x] Procedural roads: straights and arcs chained into a route, 1–3 lanes, road-edge segments (`sim/world.py`)
- [x] Car: kinematic bicycle model, two-disc collision shape (`sim/car.py`)
- [x] Traffic: lane-following cars with an IDM speed model, plus parked cars (`sim/traffic.py`)
- [x] Sensors (`odom`, `lane`, `nav`, `gps`, `lidar`, `sonar`, `radar`), each with live-tunable parameters and debug geometry for drawing (`sim/sensors.py`)
- [x] Fixed observation layout: sensor slot, present flag, normalised settings; lidar binned into 72 sectors, sonar into 16
- [x] Gymnasium env with the same metrics as the research track: `success`, `crash`, `out_of_road`, `route_completion` (`sim/env.py`)
- [x] Hand-written lane follower as a reference driver (`sim/expert.py`)
- [x] Tests: same seed gives the same run, raycasts hit hand-built geometry where expected, every sensor stays in `[0, 1]` under random settings changed mid-drive, observation shape never changes, lidar and radar see placed cars, radar misses hidden cars, the lane follower finishes routes

**Exit:** at least 2,000 steps/s in one process on a laptop, and a hand-written lane follower completes routes.

Measured with `python -m sim.env` on an Apple Silicon laptop: **2,825 steps/s** with every sensor on, and the lane follower finished **30/30 routes** at each traffic level tested (0, 0.1 and 0.3).

## Phase 2: Training ← exit check not met, in review

- [x] PPO with 12 parallel envs, reward scaling, KL limit (`train/train.py`)
- [x] Specialist baseline on `odom,lidar,nav`: **90% success** on held-out roads
- [x] Generalist policy: sensors and their settings randomised every episode, with a curriculum. 92% on the default set, 36% with sonar and radar added too, 56% on the specialist's set
- [x] Export to `.npz` and a numpy forward pass (`sim/policy.py`), with a test that torch and numpy outputs match to 1e-5
- [x] `train/benchmark.py`: 50 held-out roads per sensor set; tables in `results/`, write-up in [docs/training.md](docs/training.md#results)
- [x] Trained models drive in the 3D viewer (driver switch: Trained / Specialist / Reference)

**Exit:** the generalist reaches at least 80% success with all sensors on, on held-out maps, and stays within 10 points of the specialist on key sensor sets. If it doesn't, ship a small set of specialist models and switch between them in the UI.

**Status:** partly met. 92% on the default set passes the 80% bar, but adding sonar and radar drops it to 36%, and it's 34 points behind the specialist on its set. Tried and not adopted (details in [docs/training.md](docs/training.md#experiments-that-werent-adopted)):

- [x] Train longer (30M steps): worse almost everywhere
- [x] Crash penalty −20 and the full set in 50% of late episodes: 88% on the specialist's set, but 8% with everything on and 0% with sonar added. A second specialist, not a better generalist

Still to try, roughly cheapest first:

- [ ] Fallback from the exit rule: train specialists for a handful of sensor sets (~5 min each) and switch model when the sensor set changes
- [ ] Curriculum over *which* sensors are on: add sensors one at a time instead of jumping to fully random setups
- [ ] Only randomise settings, not presence, in part of the episodes
- [ ] Larger network (3 × 512) for the generalist
- [ ] Warm-start the generalist from the specialist instead of from scratch

## Phase 3: Browser MVP ← mostly done, waiting on a better generalist and deployment

The 3D viewer (`web/`) runs the real `sim/` in the page, driven by a trained network by default.

- [x] Web Worker that loads Pyodide, numpy and `sim/`, then steps the env at 10 Hz (the sim's step is 0.1 s; rendering interpolates between steps)
- [x] three.js scene: road mesh, cars, finish gate, chase / top-down / free cameras
- [x] Live sensor overlays: lidar rays, sonar rays, radar track boxes, GPS ghost car
- [x] Sensor panel: toggles and sliders that apply mid-drive, built from the simulator's own ranges
- [x] Road panel: lanes, bendiness, traffic, parked share, road number
- [x] HUD: speed, route progress, crash/off-road status; next road starts automatically
- [x] Loading screen (Pyodide is about 10 MB on first visit, cached after that)
- [x] Mobile layout: controls in a bottom drawer
- [x] Drive with trained policies from Phase 2 (driver switch), loaded in the browser and run with numpy
- [ ] GitHub Pages deploy through Actions

**Exit:** the public URL loads in under 10 s on broadband, runs at 30 fps on a mid-range laptop, and switching any sensor visibly changes how the car drives.

## Phase 4: Tinkering depth

- [ ] Failure buttons: kill GPS, blind lidar for 3 s, radar ghost targets
- [ ] Scenario panel: map seed, road complexity, traffic density, speed limit, weather presets (weather = sensor noise)
- [ ] Top-down camera; sensor overlay with sonar cones, radar track boxes, and a "ghost" car where GPS thinks you are
- [ ] "AI's eyes" panel: live observation bars and a polar lidar plot
- [ ] Drive yourself (keyboard and touch), with the same sensor overlay
- [ ] Race mode: 2–4 cars on one map with different sensor setups
- [ ] Benchmark button: N fixed-seed episodes at full speed → success, crash, off-road, route completion, and a sensor-cost score
- [ ] Share links: the full setup is encoded in the URL
- [ ] Personal leaderboard in local storage
- [ ] Mobile layout

**Exit:** every feature above works on desktop and phone, and a shared link reproduces the exact run.

## Phase 5: Camera and explainability

- [ ] Low-res semantic camera rendered by raycasting in numpy, shown as a small viewport
- [ ] Camera-only and camera+odom policies
- [ ] Saliency overlay: which inputs moved the steering this step (finite differences)

**Exit:** camera policies are selectable in the playground, and saliency updates live.

## Phase 6: Bring your own policy

- [ ] Colab notebook to train a custom policy on any sensor setup
- [ ] Drag a `.npz` onto the page to drive with it
- [ ] Challenge page: fixed seeds and sensor-cost targets
- [ ] Write-up of the findings

## Phase 7: Research cross-check

- [ ] Run the best browser sensor sets through the MetaDrive study in `research/`
- [ ] Report whether the ranking holds in the higher-fidelity sim

## Research track backlog

These carry over from the original plan and run alongside the phases above:

- [ ] Tune PPO until `odom,lidar,nav` finishes more than 80% of routes on unseen MetaDrive maps
- [ ] Full ablation grid, 3 seeds each, with learning curves
- [ ] Noise, dropout and mid-episode failure sweeps
- [ ] Greedy backward elimination of sensors and beam counts, plotted against hardware cost
- [ ] Port the best sets to CARLA on a cloud GPU

## Out of scope

- Real hardware
- A server-side simulator: it would cost money, limit concurrent users, and add lag
- Language or vision-language models in the control loop: too slow for 10 Hz control, and they work on objects that have already been detected, which hides the sensor question
