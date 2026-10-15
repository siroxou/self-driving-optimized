# Training

How the driving policies for the browser simulator are trained, exported and scored. Code is in `train/`.

## The models

| Model | Sensors during training | Purpose |
|---|---|---|
| `specialist` | fixed: odometry, lidar, exact route | baseline, and a check that the setup can be learned at all |
| `generalist` | random setup every episode | the playground's default driver: one network that copes with whatever you switch on |

Both are small networks: 253 inputs (the fixed observation layout in [sensors.md](sensors.md#browser-sim-differences)), two hidden layers of 256 with tanh, and 2 outputs (steering, pedal). An exported model is about 0.5 MB.

## Pipeline

```
train/train.py (PPO, 12 sims in parallel) ──► runs/<name>/model.zip
        │
train/export.py ──► web/models/<name>.npz ──► sim/policy.py (numpy)
                                                 ├── train/benchmark.py
                                                 └── the browser
```

### PPO (`train/train.py`)

Stable-Baselines3 PPO with 12 simulators in parallel, about 11,000 steps/s on an Apple Silicon laptop.

| Setting | Value | Why |
|---|---|---|
| rollout | 512 steps × 12 envs | |
| batch, epochs | 1024, 10 | |
| learning rate | 3e-4, decaying linearly to 0 | |
| `target_kl` | 0.02 | stops an update early if the policy moves too far |
| reward scaling | `VecNormalize(norm_reward=True)` | episode returns reach ~700; unscaled, the value loss sat in the hundreds |
| initial action noise | std 0.6 | full-scale random steering at 90 km/h just crashes |
| γ, λ | 0.99, 0.95 | |
| network | 2 × 256, tanh, separate actor and critic | |

The reward is the env's own: progress along the route, a small speed bonus, −5 for a crash or leaving the road, +10 for arriving. The browser, training and benchmark all agree on what "good" means.

**What didn't work, and why the settings above exist.** The first specialist run used plain PPO with no KL limit, no reward scaling and full initial noise. It climbed to 45% success during training, then collapsed to 4% at 1.8M steps (KL spikes up to 0.07, a clip fraction of 35%). Scored deterministically it reached only 10% success, crashing in 90% of runs. With the three changes above, the same 3M-step budget gave 90–97%.

An imitation warm start (DAgger, copying the reference driver) was also tried for the generalist. The copy never got its loss low enough to keep the car in lane without lidar, and longer training made it worse. The stabilised PPO didn't need it, so it was dropped.

**Randomisation.** Every episode gets a random road from the training range of road numbers, with lanes 1–3, any bendiness, traffic 0–0.3 and a parked share of 0–0.5. With `--random-sensors` it also gets a random sensor setup: each sensor on with 50% chance, and every setting drawn uniformly from its allowed range. A curriculum controls how often the full default set is used instead: 100% at the start, falling to 20% by the end, so the network learns to drive before it learns to drive half-blind.

### Export (`train/export.py`)

Only the actor network is needed to drive. Its weights are saved as `.npz`, and `sim/policy.py` runs them with numpy, so the browser needs no PyTorch. `tests/test_policy.py` checks the numpy and PyTorch outputs match to 1e-5.

### Benchmark (`train/benchmark.py`)

Runs the exported `.npz`, exactly what the browser runs, on 50 held-out roads per sensor set. Held-out roads are numbers 10,000,000 and up; training only uses lower ones. Traffic is at the default level and actions are deterministic (no exploration noise).

## Reproduce

```bash
pip install -r requirements.txt -r requirements-train.txt

python -m train.train --name specialist --sensors odom,lidar,nav --steps 3000000      # ~5 min
python -m train.train --name generalist --random-sensors --steps 15000000            # ~25 min

python -m train.benchmark web/models/generalist.npz --out results/generalist.md
python -m train.benchmark expert --out results/expert.md
```

Learning curves are written to `runs/<name>/logs/progress.csv`.

## Results

Scored with `train/benchmark.py`: 50 held-out roads per sensor set, default traffic, deterministic actions. Raw tables are in `results/`. "Default set" is `odom,lane,nav,lidar`, what the viewer starts with and what the curriculum calls the full set. "Everything on" also adds sonar and radar.

**Reference driver** (reads the simulator, ignores sensors): 100% success on every set. It's the ceiling.

### Specialist (3M steps, trained on `odom,lidar,nav`)

| Sensors | Success | Crash | Off road | Route completion |
|---|---|---|---|---|
| `odom,lane,nav,lidar,sonar,radar` | 0% | 100% | 0% | 35% |
| `odom,lane,nav,lidar` | 86% | 14% | 0% | 91% |
| `odom,lidar,nav` | 90% | 10% | 0% | 94% |
| `odom,lane,nav` | 0% | 0% | 100% | 2% |
| `odom,lidar,gps` | 88% | 12% | 0% | 93% |
| `odom,lane` | 0% | 0% | 100% | 2% |
| `odom,lidar` | 78% | 22% | 0% | 88% |
| `lidar` | 42% | 58% | 0% | 68% |
| `odom,sonar,radar,nav` | 0% | 0% | 98% | 1% |
| `odom,nav` | 0% | 0% | 100% | 2% |
| `odom` | 0% | 0% | 100% | 4% |

- **90% on its own sensor set**, and 86% on the default set (the lane camera is extra, unseen input).
- **Lidar does almost all the work.** Swap exact route for GPS and it barely notices (88%). Remove route information entirely and it still manages 78%; lidar alone, 42%. These roads have no junctions, so "where does the road go" can be read off the road edges.
- **Without lidar it can't drive at all**, even with the lane camera and route.
- **Unseen sensors break it.** Adding sonar drops it to 0%; adding radar, to 42%.

### Generalist (shipped: 15M steps, random sensors every episode)

| Sensors | Success | Crash | Off road | Route completion |
|---|---|---|---|---|
| `odom,lane,nav,lidar,sonar,radar` | 36% | 64% | 0% | 62% |
| `odom,lane,nav,lidar` | 92% | 8% | 0% | 95% |
| `odom,lidar,nav` | 56% | 44% | 0% | 79% |
| `odom,lane,nav` | 0% | 100% | 0% | 34% |
| `odom,lidar,gps` | 86% | 14% | 0% | 92% |
| `odom,lane` | 2% | 98% | 0% | 35% |
| `odom,lidar` | 0% | 100% | 0% | 36% |
| `lidar` | 36% | 64% | 0% | 62% |
| `odom,sonar,radar,nav` | 0% | 98% | 2% | 35% |
| `odom,nav` | 0% | 0% | 100% | 5% |
| `odom` | 0% | 0% | 100% | 4% |

- **92% on the default set.**
- **56% on the specialist's set**, 34 points behind the specialist.
- **36% with everything on.** Each extra sensor costs some success (radar alone: 70%; sonar alone: 74%), but nothing collapses to zero.
- **It learned the right failure modes.** Without lidar it stays on the road (0% off road with `odom,lane,nav`) and crashes into traffic instead, which is what you'd expect from a car with no obstacle sensing. The specialist just drives off the road.
- **Results are uneven between similar sets** (86% with GPS vs 56% with exact route; 36% with lidar only vs 0% with lidar + odometry). That points to an under-trained policy rather than a real effect of those sensors.

### Experiments that weren't adopted

**Longer training (30M steps):** lowered most scores (20% with everything on, 44% on the specialist's set).

<details><summary>Generalist after 30M steps</summary>

| Sensors | Success | Crash | Off road | Route completion |
|---|---|---|---|---|
| `odom,lane,nav,lidar,sonar,radar` | 20% | 80% | 0% | 55% |
| `odom,lidar,nav` | 44% | 56% | 0% | 67% |
| `odom,lane,nav` | 0% | 100% | 0% | 27% |
| `odom,lidar,gps` | 64% | 36% | 0% | 77% |
| `odom,lane` | 0% | 98% | 2% | 34% |
| `odom,lidar` | 2% | 98% | 0% | 38% |
| `lidar` | 6% | 94% | 0% | 46% |
| `odom,sonar,radar,nav` | 0% | 100% | 0% | 32% |
| `odom,nav` | 0% | 84% | 16% | 30% |
| `odom` | 0% | 12% | 88% | 12% |

</details>

**Crash penalty −20 and the full set in 50% of late episodes (v2, 15M steps):** the idea was to punish speeding into traffic harder and keep more full-sensor practice.

| Sensors | Success | Crash | Off road | Route completion |
|---|---|---|---|---|
| `odom,lane,nav,lidar,sonar,radar` | 8% | 92% | 0% | 42% |
| `odom,lane,nav,lidar` | 90% | 10% | 0% | 95% |
| `odom,lidar,nav` | 88% | 12% | 0% | 93% |
| `odom,lane,nav` | 0% | 0% | 100% | 2% |
| `odom,lidar,gps` | 80% | 20% | 0% | 88% |
| `odom,lane` | 0% | 2% | 98% | 5% |
| `odom,lidar` | 4% | 96% | 0% | 42% |
| `lidar` | 4% | 96% | 0% | 42% |
| `odom,sonar,radar,nav` | 0% | 4% | 96% | 9% |
| `odom,nav` | 0% | 0% | 100% | 2% |
| `odom` | 0% | 0% | 100% | 3% |

It made a second specialist, not a better generalist:

| Sensors | v1 (shipped) | v2 |
|---|---|---|
| default set | **92%** | 90% |
| `odom,lidar,nav` | 56% | **88%** |
| default + radar | **70%** | 44% |
| default + sonar | **74%** | 0% |
| everything on | **36%** | 8% |
| `lidar` only | **36%** | 4% |
| no lidar | stays on road, hits traffic | drives off road |

v2 is excellent on lidar-based sets and brittle everywhere else. Coping with whatever you switch on is the generalist's whole job, so v1 stays as the shipped model. The code is back to the v1 settings (crash −5, full set falling to 20%), so it reproduces the shipped model.

### Phase 2 exit check

| Criterion | Target | Shipped generalist |
|---|---|---|
| Success with all sensors on | ≥ 80% | **92%** on the default set ✓, 36% with sonar and radar added ✗ |
| Gap to the specialist on its set | ≤ 10 points | 34 points ✗ (v2 managed 2 points, but at too high a cost elsewhere) |

### Takeaways so far

1. A fixed sensor set is easy to learn here: 3M steps, about 5 minutes.
2. One network for every sensor setup handles the default set well but degrades as sensors are added or removed. Both training changes tried so far traded robustness for peak performance on lidar sets.
3. Sonar is the most disruptive sensor to add to a network that hasn't seen it much, probably because at 5 m range it reads road edges constantly, which a lidar-trained network misreads.
