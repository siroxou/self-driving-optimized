# self-driving-optimized

How little does a self-driving car actually need to sense?

This project trains a driving policy in simulation and then takes sensors away, one at a time, to see where it starts to fail. Every sensor (lidar, ultrasonic, radar, GPS, lane camera, odometry) is a switch. The same training recipe runs on each combination, and the results go into one table so the trade-offs are easy to compare.

Simulator: [MetaDrive](https://github.com/metadriverse/metadrive) (procedurally generated roads, runs on a laptop CPU, works on Apple Silicon).
Learning: PPO from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Quickstart

Tested on Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # checks every sensor builds and stays in range
```

The first run downloads MetaDrive's map and vehicle assets.

Train one agent:

```bash
python train.py --sensors odom,lidar,nav --steps 300000
```

Evaluate it on maps it has never seen, or watch it drive:

```bash
python evaluate.py runs/odom+nav+lidar-s0 --episodes 50
python evaluate.py runs/odom+nav+lidar-s0 --episodes 3 --render
```

Run the full sensor study (this takes a while, see [docs/experiment.md](docs/experiment.md)):

```bash
python ablate.py
```

## Sensors

| Name | Real-world equivalent | Size | What the policy gets |
|---|---|---|---|
| `odom` | Wheel encoders + IMU | 5 | speed, steering angle, last throttle/steer, yaw rate |
| `lane` | Lane-keeping camera | 3 | distance to left/right road edge, heading error to the lane |
| `nav` | HD map + exact localisation | 10 | offsets to the next two route checkpoints, lane curvature |
| `gps` | Consumer GNSS + map | 10 | same as `nav`, with jitter and a slowly drifting bias |
| `lidar` | 360° 2D lidar | 240 | distance per beam, 50 m range |
| `sonar` | Ultrasonic parking sensors | 8 | distance per beam, 5 m range |
| `radar` | Automotive radar | 16 | relative position and velocity of the 4 nearest vehicles, 80 m |

`nav` and `gps` describe the same thing at two quality levels, so use one or the other. Camera input is planned (see the [roadmap](ROADMAP.md)).

All ranges, beam counts and noise levels are constants at the top of [`sdo/sensors.py`](sdo/sensors.py). Details and limitations are in [docs/sensors.md](docs/sensors.md).

## Results

Pending the first full run. `ablate.py` writes `results/results.md`, and that table will be copied here.

| Sensors | Obs size | Success | Crash | Off road | Route completion |
|---|---|---|---|---|---|
| … | | | | | |

## Layout

```
sdo/sensors.py      sensor definitions, observation builder, environment factory
train.py            train PPO on one sensor set
evaluate.py         score a trained run on held-out maps
ablate.py           train + evaluate a whole grid of sensor sets
tests/              smoke tests
docs/               sensor models and experiment method
ROADMAP.md          what comes next
```

## Adding a sensor

Write a function `reader(obs, vehicle) -> list[float]` that returns values in `[0, 1]`, then add it to `SENSORS` in `sdo/sensors.py` with its output size. It's immediately usable from every script.

## License

MIT
