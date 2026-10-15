# Browser playground: architecture

Status: the simulator (`sim/`) is built (Phase 1). The browser side starts in Phase 3 of the [roadmap](../ROADMAP.md).

## Goals

- Opens from a link, with nothing to install
- Anyone can switch sensors off, change their settings mid-drive, and see the effect straight away
- Free to host, with no limit on visitors
- The browser runs the same simulation the policy was trained on, so the numbers on screen mean something

## Why not MetaDrive in the browser

MetaDrive is built on Panda3D and Bullet, and neither runs in a browser. Streaming it from a server would work, but it costs money, handles only a few users at a time (one simulator per process), and adds input lag. So the browser gets its own simulator, and MetaDrive stays as the research track.

## How it fits together

```
sim/ (Python + numpy)
  ├── offline: train/ uses it with Stable-Baselines3 → model weights (.npz)
  └── browser: Pyodide runs it inside a Web Worker
                    │  state snapshot every sim step
                    ▼
               main thread: three.js renders the scene, ui.js handles controls
```

- **One simulator, two places.** `sim/` is plain Python with only numpy as a dependency. Pyodide (CPython compiled to WebAssembly) runs it unchanged in the browser. No logic is rewritten in JavaScript, so training and the demo can't drift apart.
- **Physics is 2D, rendering is 3D.** Cars on a road only need position, heading and speed on a plane. three.js adds the third dimension for looks: road mesh, car models, chase camera.
- **The policy runs in numpy.** Training uses PyTorch through Stable-Baselines3. `train/export.py` writes the network weights to `.npz`, and `sim/policy.py` runs the same small network with numpy matrix multiplies. A test checks both give the same outputs.
- **A worker keeps the page smooth.** The simulator and policy run in a Web Worker so the page never freezes. The worker sends a small snapshot each step and the main thread draws it.

## Worker messages

Main thread → worker:

| Message | Payload |
|---|---|
| `init` | model URL, initial config |
| `reset` | map seed, scenario settings |
| `config` | changed sensor toggles and parameters (applied on the next step) |
| `fail` | sensor name and duration |
| `mode` | `ai`, `manual`, `race` |
| `input` | steering and throttle (manual mode) |
| `benchmark` | config, seeds, episode count |

Worker → main thread:

| Message | Payload |
|---|---|
| `ready` | load time |
| `frame` | car poses, ray endpoints, radar tracks, GPS ghost pose, observation vector, metrics |
| `done` | end reason and episode metrics |
| `benchmark` | progress, then the final table |

Snapshots are plain arrays (Float32Array where it matters) so passing them is cheap.

## One policy for every sensor setup

The sliders change the size of sensor readings (4 lidar beams vs. 240), and the toggles remove sensors entirely. A normal network needs a fixed input, so:

- every sensor always has a slot in the observation, plus a "present" flag and its current settings
- lidar is binned into 72 fixed sectors and sonar into 16, with "no data" where no beam passes
- training randomises sensors and settings every episode

Details are in [sensors.md](sensors.md#browser-sim-differences). If one policy can't cover every setup well enough, the fallback is a few specialist models that the page switches between.

## Frontend

- Plain HTML, CSS and JavaScript modules, with no build step
- three.js loaded from a CDN at a pinned version
- The current setup lives in the URL hash, so any link reproduces a run
- The personal leaderboard is stored in `localStorage`, wrapped so it fails quietly in private browsing

## Running locally

Once `web/` exists (Phase 3):

```bash
python -m http.server -d web 8000
```

Then open http://localhost:8000. The page fetches `sim/` as a zip, which the deploy workflow builds. Locally, `python tools/pack_sim.py` does the same.

## Deploying

A GitHub Actions workflow packs `sim/` into `web/sim.zip`, copies the exported models into `web/models/`, and publishes `web/` to GitHub Pages on every push to `main`.

## Performance targets

| | Target |
|---|---|
| First load | under 10 s on broadband (Pyodide + numpy ≈ 10 MB, cached afterwards) |
| Simulation | 10 Hz in real time (0.1 s steps), rendering interpolates between them; benchmark mode runs as fast as the worker allows. Native Python runs about 2,800 steps/s with every sensor on, so the browser has plenty of headroom even if Pyodide is several times slower |
| Rendering | 30 fps on a mid-range laptop |
| Phone | usable at 375 px wide |
