# Sensor models

This page defines the sensors for both parts of the project. The browser sim and the MetaDrive research track use the same names, meanings and default values, so results from one can be compared with the other.

- **Research track:** `research/sdo/sensors.py`. Every sensor is a function that reads the ego vehicle and returns floats in `[0, 1]`. The observation is the concatenation of the selected sensors in a fixed order (the order of the `SENSORS` dict), so `lidar,nav` and `nav,lidar` produce identical inputs.
- **Browser sim:** `sim/sensors.py`. It uses the same sensors, plus live-tunable parameters and a fixed observation layout. See [Browser sim differences](#browser-sim-differences) below.

These are simulator stand-ins, not physically accurate models. The point is to give each sensor roughly the information content and failure modes of the real thing, so the ablation says something useful.

## odom: wheel odometry + IMU

5 values: speed (normalised to the vehicle's top speed), current steering angle, last throttle/brake command, last steering command, and yaw rate.

Every real car already has these. Treat it as the "free" baseline that almost every set includes.

Limitation: it's noise-free. Real wheel odometry drifts with tyre slip.

## lane: lane-keeping camera

3 values: distance to the left and right road edge (normalised by the maximum road width), and the heading difference between the car and the current lane.

This is the output of a typical lane-detection camera module, not raw pixels. It's taken straight from the simulator, so it never misses a faded line.

## nav: route guidance with perfect localisation

10 values from MetaDrive's navigation module: for each of the next two route checkpoints, the forward and lateral offset in the car's frame, the radius of the lane leading to it, its turn direction, and its angle.

This assumes an HD map and centimetre-level localisation (RTK GPS or map matching). It tells the car where the road goes next, which is very powerful. Expect most sets without it to struggle on routing even when they drive smoothly.

## gps: route guidance from consumer GNSS

Same 10 values as `nav`, corrupted by:

- **jitter**: Gaussian noise every step, `GPS_NOISE = 0.03` (about 1.5 m)
- **drift**: a bias that random-walks by `GPS_DRIFT = 0.005` per step, clipped to ±0.1, and reset each episode

This imitates the slow wander of phone-grade GPS. Use `gps` or `nav`, not both.

## lidar: 360° 2D lidar

`LIDAR_BEAMS = 240` beams, `LIDAR_RANGE = 50` m, clockwise from the nose. Each value is the hit distance divided by the range (1.0 means nothing hit). It sees vehicles and obstacles, but not lane paint.

Lower `LIDAR_BEAMS` to find out how sparse a scan still works (Phase 4).

## sonar: ultrasonic parking sensors

`SONAR_BEAMS = 8` beams, `SONAR_RANGE = 5` m. This is the closest available stand-in for echolocation: a ring of short-range distance readings. Real ultrasonics have wide cones rather than thin rays and pick up ghost echoes; neither is modelled.

## radar: automotive radar tracks

The `RADAR_TRACKS = 4` nearest vehicles within `RADAR_RANGE = 80` m. For each one: relative position (longitudinal, lateral) and relative velocity (longitudinal, lateral). Empty slots are zero. Detection uses a 72-beam sweep, so a vehicle fully hidden behind another isn't seen.

Real radar measures range and Doppler velocity directly but has poor angular resolution and sees through some occlusion. The model is more precise in angle than real radar and less capable around occlusion.

## Camera (planned)

A camera is in Phase 5 of the roadmap. It will be a low-resolution semantic camera rendered by raycasting in numpy, so training and the browser see identical images. Learning from pixels takes far more samples than learning from vector sensors, so it comes after the vector-sensor demo works.

## Changing a sensor

In the research track, all parameters are constants at the top of `research/sdo/sensors.py`. Change one, re-run `research/ablate.py`, and the observation sizes update automatically. Existing trained models won't load against a different size, so retrain after changing beam or track counts.

## Browser sim differences

In the browser, people change sensors while the car is driving, so one trained policy has to cope with any setup. That changes three things compared with the research track.

### Fixed observation layout

Every sensor always has its own slot in the observation, in this order:

```
[ readings ... | settings ... | present ]
```

A switched-off sensor's whole slot is zeros, including the `present` flag. Settings are normalised to `[0, 1]` using the ranges below, so the policy can tell a 4-beam lidar from a 240-beam one, or clean GPS from noisy GPS.

| Slot | Readings | Settings | Total |
|---|---|---|---|
| `odom` | 5 (as above) | noise | 7 |
| `lane` | 3 (as above) | noise | 5 |
| `route` | 4 route points 5/15/30/50 m ahead in the car's frame (8), route progress, lane count | exact?, GPS jitter, GPS drift | 14 |
| `lidar` | 72 sector distances + 72 "has data" flags | beams, range, noise, dropout | 149 |
| `sonar` | 16 sector distances + 16 "has data" flags | beams, range | 35 |
| `radar` | 8 tracks × (position x/y, velocity x/y, valid) | tracks, range | 43 |

Total: 253 values.

`nav` and `gps` share the `route` slot. If `nav` is on, the route is computed from the true position. If only `gps` is on, it's computed from a noisy estimate, so the car "thinks" it's somewhere slightly different and the route points shift with it.

### Sector binning

Lidar beams are binned into 72 sectors of 5°, keeping the shortest distance per sector. A sector no beam passes through has its "has data" flag at 0. Sonar works the same way with 16 sectors of 22.5°. The input shape therefore stays the same whether you pick 4 lidar beams or 240.

Each sonar beam is three rays spread over ±10°, keeping the shortest, which is closer to how a real ultrasonic cone behaves than a single thin ray.

### Live-tunable settings

| Sensor | Setting | Range | Default | Unit |
|---|---|---|---|---|
| `odom` | noise | 0–0.3 | 0 | std dev, normalised units |
| `lane` | noise | 0–2 | 0 | std dev, metres |
| `gps` | jitter | 0–10 | 1.5 | std dev per step, metres |
| `gps` | drift | 0–1 | 0.1 | bias random-walk step, metres (bias capped at 10 m) |
| `lidar` | beams | 4–240 | 240 | |
| `lidar` | range | 5–100 | 50 | metres |
| `lidar` | noise | 0–0.2 | 0 | std dev, fraction of range |
| `lidar` | dropout | 0–0.9 | 0 | chance each beam returns nothing |
| `sonar` | beams | 2–16 | 8 | |
| `sonar` | range | 1–10 | 5 | metres |
| `radar` | tracks | 1–8 | 4 | |
| `radar` | range | 20–200 | 80 | metres |

Values outside these ranges are clamped. These ranges also define what the training randomises over (Phase 2), so a policy is only expected to work inside them.

### Radar occlusion

A vehicle counts as visible to radar if a ray from the ego car to its centre hits it before any other car. Road edges don't block radar.

### Scenario

The road and traffic are set per episode, separately from the sensors:

| Setting | Range | Default | Meaning |
|---|---|---|---|
| `lanes` | 1–3 | 2 | lanes, all one direction |
| `complexity` | 0–1 | 0.5 | more road pieces and tighter bends |
| `traffic` | 0–0.5 | 0.1 | cars per 20 m of lane |
| `parked` | 0–1 | 0.2 | share of cars that are parked (broken down) |

Parked cars only stand in the rightmost lane and moving cars use the other lanes, because traffic doesn't change lanes and would otherwise queue forever behind a parked car.
