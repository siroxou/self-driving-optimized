# Sensor models

Every sensor lives in `sdo/sensors.py` as a function that reads the ego vehicle and returns floats in `[0, 1]`. The observation is the concatenation of the selected sensors in a fixed order (the order of the `SENSORS` dict), so `lidar,nav` and `nav,lidar` produce identical inputs.

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

RGB and depth cameras are in Phase 2 of the roadmap. They need a CNN policy and offscreen rendering, and learning from them takes far more samples, so they're kept separate from the vector sensors.

## Changing a sensor

All parameters are constants at the top of `sdo/sensors.py`. Change one, re-run `ablate.py`, and the observation sizes update automatically. Existing trained models won't load against a different size, so retrain after changing beam or track counts.
