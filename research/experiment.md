# Experiment method

## Question

For each sensor set: can a policy trained only on that input drive a route to the end on roads it has never seen, without crashing or leaving the road?

## Environment

- MetaDrive 0.4.3 `MetaDriveEnv`, procedurally generated road networks (straights, curves, roundabouts, intersections, ramps)
- Traffic density 0.1 (light traffic)
- Episode limit 1500 steps; an episode also ends on arrival, crash or leaving the road
- Action: continuous `[steering, throttle/brake]`, both in `[-1, 1]`
- Reward: MetaDrive's default (progress along the route, speed bonus, penalties for crashing and leaving the road). It's kept identical for every sensor set so only the input changes.

## Train/test split

| | Map seeds |
|---|---|
| Training | 0–499 |
| Evaluation | 1000–1099 |

They never overlap, so the numbers measure generalisation rather than memorised maps.

## Training

PPO (`MlpPolicy`, two hidden layers of 64), `n_steps=2048`, `batch_size=256`, everything else at Stable-Baselines3 defaults. The default budget is 300k steps per run. That's a starting point: Phase 0 of the roadmap is finding the budget where the baseline plateaus.

Each sensor set is trained with 3 seeds (`--seeds`), because RL results vary a lot between seeds and one lucky run proves nothing.

## Metrics

Measured over `--episodes` evaluation episodes with deterministic actions:

| Metric | Meaning |
|---|---|
| `success` | fraction of episodes that reached the destination |
| `crash` | fraction that ended touching a vehicle, object, building or person |
| `out_of_road` | fraction that ended off the drivable area |
| `route_completion` | mean fraction of the route covered before the episode ended |
| `obs_dim` | input size, a rough proxy for sensor bandwidth |

`route_completion` is the most informative number for weak sensor sets, since their success rate is often zero.

## Outputs

- `runs/<sensors>-s<seed>/model.zip`: trained policy
- `runs/<sensors>-s<seed>/run.json`: sensors, steps and seed used
- `results/results.csv`: one row per (sensor set, seed), written after every run so a crash doesn't lose progress
- `results/results.md`: mean over seeds, sorted by success

## Running time

On an Apple Silicon laptop CPU it runs at around 600 steps per second with vector sensors. One 300k-step run takes roughly 10–30 minutes once PPO updates are included, so the default 11-set × 3-seed grid is an overnight job. For a quick sanity pass:

```bash
python ablate.py --steps 50000 --episodes 10 --seeds 1
```

## Reading the results

- If a set fails at **routing** (low route completion, few crashes), it's missing `nav`/`gps`.
- If it fails at **staying on the road** (high `out_of_road`), it's missing `lane` or `lidar`.
- If it fails in **traffic** (high `crash`), it's missing `lidar`/`radar`/`sonar`; raise `traffic` in `make_env` to make this clearer.
- A sensor is worth keeping only if removing it costs more than the seed-to-seed spread.

## Known limitations

- Sensor models are idealised (see [docs/sensors.md](../docs/sensors.md)).
- PPO from scratch is a weak learner. A set that "fails" might succeed with more steps or imitation learning, so treat results as "under this training budget".
- MetaDrive allows one simulator per process, so runs are sequential. Parallelising means running separate processes.
