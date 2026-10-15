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

Scored with `train/benchmark.py`: 50 held-out roads per sensor set, default traffic, deterministic actions. Raw tables are in `results/`.

**Reference driver** (reads the simulator, ignores sensors): 100% success on every set. It's the ceiling.

### Specialist (3M steps, trained on `odom,lidar,nav`)

| Sensors | Success | Crash | Off road | Route completion |
|---|---|---|---|---|
| `odom,lane,nav,lidar,sonar,radar` | 0% | 100% | 0% | 35% |
| `odom,lidar,nav` | 90% | 10% | 0% | 94% |
| `odom,lane,nav` | 0% | 0% | 100% | 2% |
| `odom,lidar,gps` | 88% | 12% | 0% | 93% |
| `odom,lane` | 0% | 0% | 100% | 2% |
| `odom,lidar` | 78% | 22% | 0% | 88% |
| `lidar` | 42% | 58% | 0% | 68% |
| `odom,sonar,radar,nav` | 0% | 0% | 98% | 1% |
| `odom,nav` | 0% | 0% | 100% | 2% |
| `odom` | 0% | 0% | 100% | 4% |

- **90% on its own sensor set** (97% on an earlier 30-road check). That's the main proof that the simulator and training setup work.
- **Lidar does almost all the work.** Swap exact route for GPS and it barely notices (88%). Remove route information entirely and it still manages 78%; lidar alone, 42%. These roads have no junctions, so "where does the road go" can be read off the road edges.
- **Without lidar it can't drive at all**, even with the lane camera and route.
- **Extra sensors break it** (0% with everything on). It never saw those inputs during training, so they look like noise. That's the problem the generalist is meant to solve.

### Generalist (15M steps, random sensors every episode)

| Sensors | Success | Crash | Off road | Route completion |
|---|---|---|---|---|
| `odom,lane,nav,lidar,sonar,radar` | 36% | 64% | 0% | 62% |
| `odom,lidar,nav` | 56% | 44% | 0% | 79% |
| `odom,lane,nav` | 0% | 100% | 0% | 34% |
| `odom,lidar,gps` | 86% | 14% | 0% | 92% |
| `odom,lane` | 2% | 98% | 0% | 35% |
| `odom,lidar` | 0% | 100% | 0% | 36% |
| `lidar` | 36% | 64% | 0% | 62% |
| `odom,sonar,radar,nav` | 0% | 98% | 2% | 35% |
| `odom,nav` | 0% | 0% | 100% | 5% |
| `odom` | 0% | 0% | 100% | 4% |

- **Phase 2 exit check: not met.** The target was ≥80% with all sensors on and within 10 points of the specialist on its set; it gets 36% and 56%.
- **It did learn the right failure modes.** Without lidar, it now stays on the road (0% off road with `odom,lane,nav`) and crashes into traffic instead, which is what you'd expect from a car with no obstacle sensing. The specialist just drives off the road.
- **Results are uneven between similar sets** (86% with GPS, 56% with exact route; 36% with lidar only but 0% with lidar + odometry). That points to an under-trained policy rather than a real effect of those sensors.
- **More training made it worse.** Continuing to 30M steps lowered most scores (20% with all sensors):

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

### Takeaways so far

1. A fixed sensor set is easy to learn here; 3M steps (about 5 minutes) is enough.
2. One network for every sensor setup is much harder, and the current recipe plateaus. Likely causes: 80% of late-training episodes have setups that can't drive well at all, which drowns out the learning signal; and the −5 crash penalty is small next to the progress reward, so speeding into traffic isn't punished much. See the roadmap for what to try next.

