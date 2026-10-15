# Roadmap

The goal is a defensible answer to one question: what is the cheapest sensor set that still drives safely, and how much does each sensor buy you?

## Phase 0: Baseline ← current

- [x] Switchable sensor suite on MetaDrive (odometry, lane, nav, GPS, lidar, sonar, radar)
- [x] PPO training, held-out evaluation, ablation runner
- [x] Smoke tests
- [ ] Tune PPO until `odom,lidar,nav` reliably finishes routes on unseen maps (target: >80% success)
- [ ] Pick the training budget (steps) where the baseline stops improving

## Phase 1: Ablation study

- [ ] Run the default grid in `ablate.py` with 3 seeds each
- [ ] Learning-curve plot per sensor set (steps to reach 50% success)
- [ ] Publish the results table in the README
- [ ] Short write-up: which sensors matter, which are redundant, and which only help in traffic

## Phase 2: Vision

- [ ] RGB and depth camera observations (MetaDrive offscreen rendering)
- [ ] CNN policy (`MultiInputPolicy`), with the image combined with the vector sensors
- [ ] Compare camera-only, camera+odom and camera+gps against the lidar results
- [ ] Note the extra compute and sample cost of learning from pixels

## Phase 3: Robustness

- [ ] Sweep noise levels and beam dropout for every sensor
- [ ] Sensor failure mid-episode (e.g. GPS dropout in a "tunnel")
- [ ] Heavier traffic and harder maps
- [ ] Check whether redundant sensors start to matter once failures are possible

## Phase 4: Minimal-set search

- [ ] Greedy backward elimination: drop the sensor whose removal hurts least, and repeat
- [ ] Shrink resolution too: how few lidar beams and how short a range still work
- [ ] Rough hardware cost per sensor, then plot a cost vs. safety frontier

## Phase 5: Realism

- [ ] Port the best two or three sensor sets to CARLA on a cloud GPU (native radar, GNSS and IMU models, real camera rendering)
- [ ] Check whether the ranking from MetaDrive holds up
- [ ] Optional: imitation learning from CARLA's autopilot as a stronger baseline than PPO from scratch

## Out of scope for now

- Real hardware
- Multi-agent scenarios
- Language or vision-language models in the control loop: too slow for 10–20 Hz control, and they work on already-perceived objects, which hides the sensor question this project is trying to answer
