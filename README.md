# self-driving-optimized

How little does a self-driving car actually need to sense?

This project trains a car to drive in simulation and then takes its senses away (lidar, ultrasonic, radar, GPS, lane camera, odometry) to see where it starts to fail.

It has two parts:

- **Browser playground** (in development): open a link, switch sensors off, drag sliders for beam count, range and noise, and watch the car cope or crash in 3D. It runs entirely in your browser. The plan is in [ROADMAP.md](ROADMAP.md) and the design is in [docs/web.md](docs/web.md).
- **Research track** (`research/`): a sensor ablation study on the [MetaDrive](https://github.com/metadriverse/metadrive) simulator using PPO. It's higher fidelity and slower, and it's used to check the playground's findings.

## Sensors

Both parts use the same sensors:

| Name | Real-world equivalent | What the car gets |
|---|---|---|
| `odom` | Wheel encoders + IMU | speed, steering angle, last throttle/steer, yaw rate |
| `lane` | Lane-keeping camera | distance to each road edge, heading error to the lane |
| `nav` | HD map + exact localisation | offsets to the next two route checkpoints, lane curvature |
| `gps` | Consumer GNSS + map | same as `nav`, with jitter and a slowly drifting bias |
| `lidar` | 360° 2D lidar | distance per beam, 240 beams, 50 m |
| `sonar` | Ultrasonic parking sensors | distance per beam, 8 beams, 5 m |
| `radar` | Automotive radar | position and velocity of the 4 nearest vehicles, 80 m |

How each one is modelled, and what it gets wrong compared with real hardware, is in [docs/sensors.md](docs/sensors.md).

## Research track quickstart

Tested on Python 3.11.

```bash
cd research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

The first run downloads MetaDrive's map and vehicle assets.

```bash
python train.py --sensors odom,lidar,nav --steps 300000      # train one sensor set
python evaluate.py runs/odom+nav+lidar-s0 --episodes 50        # score it on unseen maps
python evaluate.py runs/odom+nav+lidar-s0 --episodes 3 --render  # watch it drive
python ablate.py                                               # the full study
```

The method, metrics and expected running times are in [research/experiment.md](research/experiment.md).

## Results

Pending the first full run.

## Layout

```
sim/                 browser simulator (Python + numpy, runs in Pyodide). In progress
research/            MetaDrive study: sensors, training, evaluation, ablation
docs/sensors.md      sensor definitions shared by both parts
docs/web.md          browser playground architecture
ROADMAP.md           phases and exit criteria
```

## License

MIT
