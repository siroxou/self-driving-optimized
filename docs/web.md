# Browser playground: architecture

Status: the simulator (`sim/`) and a 3D viewer (`web/`) run in the browser, driven by trained networks (or the reference lane follower). Deployment and the tinkering features follow the [roadmap](../ROADMAP.md).

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

- **One simulator, two places.** `sim/` is plain Python with only numpy as a dependency. Gymnasium is optional: training uses it, but Pyodide can't import it (it needs `multiprocessing`), so without it `DriveEnv` is a plain class with the same methods and the same seeding. Pyodide (CPython compiled to WebAssembly) runs it unchanged in the browser. No logic is rewritten in JavaScript, so training and the demo can't drift apart.
- **Physics is 2D, rendering is 3D.** Cars on a road only need position, heading and speed on a plane. three.js adds the third dimension for looks: road mesh, car models, chase camera.
- **The policy runs in numpy.** Training uses PyTorch through Stable-Baselines3. `train/export.py` writes the network weights to `.npz`, and `sim/policy.py` runs the same small network with numpy matrix multiplies. A test checks both give the same outputs.
- **A worker keeps the page smooth.** The simulator and policy run in a Web Worker so the page never freezes. The worker sends a small snapshot each step and the main thread draws it.

## Worker messages

Implemented now (`web/worker.js`, with the Python side in `web/bridge.py`):

| Direction | Message | Payload |
|---|---|---|
| page → worker | `reset` | road number, scenario settings, time limit (levels use their own) |
| page → worker | `config` | changed sensor toggles and settings (applied on the next step) |
| page → worker | `driver` | a model name (loads `models/<name>.npz`) or `reference` |
| page → worker | `speed` | 1, 2 or 4× |
| page → worker | `pause` | true / false |
| worker → page | `status` | loading progress text |
| worker → page | `settings` | sensor and scenario defaults and ranges, used to build the controls |
| worker → page | `road` | road edges, lane lines, finish position, crosswalks, debris, potholes, first frame |
| worker → page | `driver` | which driver is active, and whether its model loaded |
| worker → page | `frame` | ego pose and speed, traffic and pedestrian poses, lidar and sonar rays, radar target positions, GPS estimate, status, progress, potholes hit |
| worker → page | `error` | why loading failed |

Planned for later phases:

| Direction | Message | Payload |
|---|---|---|
| page → worker | `fail` | sensor name and duration |
| page → worker | `mode` | `ai`, `manual`, `race` |
| page → worker | `input` | steering and throttle (manual mode) |
| page → worker | `benchmark` | config, seeds, episode count |
| worker → page | `benchmark` | progress, then the final table |

Frames are sent as JSON strings. At 10 per second and a few kilobytes each, that's cheap enough; switch to typed arrays if it ever shows up in profiles.

## Choosing a model

The page fetches `models/specialists.json`, written by `train/specialists.py`, which lists every specialist with its sensor set and benchmark score. In Auto mode, every sensor toggle recomputes the set of sensors that are on (in the simulator's fixed order, e.g. `odom-nav-lidar`). The page loads `specialist-<set>.npz` if it's listed, and `generalist.npz` if not. A model is only fetched when the choice changes. Settings sliders never switch models; they change what the current model sees.

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

Serve the repo root (not `web/`), because the worker fetches the simulator from `../sim/`:

```bash
python -m http.server 8000
```

Then open http://localhost:8000/web/. No build step and no Python packages are needed on your machine; the page downloads Pyodide and numpy itself. The worker fetches the `sim/` files with `cache: "no-cache"`, so edits show up on a normal reload.

## Deploying

`.github/workflows/pages.yml` publishes to GitHub Pages on every push to `main` (or by hand from the Actions tab). It copies `web/` (page, worker and models) and `sim/` into one site, keeping them side by side so the worker's `../sim/` paths still work. It also adds a root page that redirects to `web/`.

Live: https://siroxou.github.io/self-driving-optimized/web/

## Performance targets

| | Target |
|---|---|
| First load | under 10 s on broadband (Pyodide + numpy ≈ 10 MB, cached afterwards) |
| Simulation | 10 Hz in real time (0.1 s steps), rendering interpolates between them; benchmark mode runs as fast as the worker allows. Native Python runs about 2,800 steps/s with every sensor on, so the browser has plenty of headroom even if Pyodide is several times slower |
| Rendering | 30 fps on a mid-range laptop |
| Phone | usable at 375 px wide |
