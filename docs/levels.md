# Evaluation levels

Two fixed courses for stress-testing drivers: ordinary traffic plus pedestrians, road debris and potholes. Every driver faces exactly the same roads, so scores are directly comparable. Levels are defined in `sim/levels.py`, hazards in `sim/hazards.py`.

In the viewer, pick **Commute** or **Gauntlet** under *Road*. The car drives the level's roads in order, the HUD keeps score, and switching a sensor on or off (or changing driver) restarts the level, so a score always belongs to one setup.

From the command line:

```bash
python -m train.benchmark expert --level gauntlet --sets odom    # reference driver
python -m train.benchmark auto --level gauntlet                  # the viewer's Auto driver, per sensor set
```

## The levels

| | Commute | Gauntlet |
|---|---|---|
| Lanes | 2 | 3 |
| Bendiness | 0.4 | 0.8 |
| Traffic | 0.1 | 0.25 |
| Parked share | 0.2 | 0.4 |
| Crosswalks per km | 4 | 12 |
| Debris per km | 2 | 8 |
| Potholes per km | 5 | 20 |
| Roads | 50 (from 20,000,000) | 50 (from 20,100,000) |
| Time limit | 150 s | 240 s |

Road numbers are well clear of training (below 10,000,000) and of the standard benchmark (from 10,000,000).

## Hazards

| Hazard | What it does | Who can see it | If you hit it |
|---|---|---|---|
| **Pedestrians** | One per crosswalk. Waits at the kerb for 2–10 s, crosses the whole road at 1.0–1.6 m/s, then waits and crosses back. Won't step out while a moving car is within 12 m of the crosswalk, but anyone already crossing must be waited for | lidar, sonar, radar | run ends: *pedestrian hit* |
| **Debris** | A staggered row of three cones or a fallen load, in the rightmost lane only (like parked cars, so traffic can't get stuck) | lidar, sonar | run ends: *crash* |
| **Potholes** | Holes 0.8–1.8 m across, anywhere in the lanes | **no current sensor** | −30% speed and counted; above 40 km/h the run ends: *damaged* |

Traffic yields the whole crosswalk while someone is crossing and drives over potholes. All three hazard settings are also sliders in Free drive (per km, default 0). With all three at 0 the simulator behaves exactly as before; a test checks this bit for bit.

Potholes are invisible to every sensor on purpose, for now. The 2D lidar scans at bumper height, radar tracks moving objects, and the lane camera reports lane geometry only. Seeing them would need a new reading in the camera, and that changes every model's input. That's the next step on the [roadmap](../ROADMAP.md).

## Results

### Reference driver: 100% on both

The reference driver reads the simulator directly. It yields whole crosswalks, treats debris like parked cars, changes lane away from potholes and slows to 30 km/h for ones it can't avoid. It finishes all 50 roads of both levels, so both are drivable. (Building it turned up two standoffs worth knowing about: a pedestrian waiting for the car while the car waits for the pedestrian, and the same with a parked car next to a crosswalk. Pedestrians now only wait for *moving* cars, and the driver only treats a waiting pedestrian as a threat while it's far enough away that the pedestrian could still step out.)

### Trained models: nearly zero

Same models as the rest of the project, never trained with any of these hazards. Scored with the viewer's Auto choice per sensor set (`results/level-commute.md`, `results/level-gauntlet.md`):

| Sensors | Commute | Gauntlet | Main failure |
|---|---|---|---|
| Odometry + Camera + Route + Lidar | 6% | 0% | pothole damage (66% / 68%) |
| Odometry + Route + Lidar | 4% | 0% | pothole damage |
| Odometry + GPS + Lidar | 2% | 0% | pothole damage |
| everything on | 6% | 0% | pothole damage (76% / 68%) |
| Odometry + Camera + Route + Radar | 0% | 0% | crashes (66% / 36%) |
| Odometry + Route + Sonar + Radar | 2% | 0% | pothole damage |
| Lidar only | 6% | 0% | pothole damage |
| Odometry + Camera + Route | 0% | 0% | crashes |

### What breaks them

Commute roads with one hazard type at a time (50 standard test roads, default-set and everything-on drivers):

| Commute roads with | Default set | Everything on |
|---|---|---|
| no hazards | 88% | 88% |
| debris only | 68% | 64% |
| pedestrians only | 52% | 52% |
| potholes only | 18% | 16% |
| all hazards | 6% | 2% |

- **Potholes are the biggest problem, and no sensor sees them.** At 90 km/h the first pothole ends the run. Only slowing down everywhere, or a sensor that can see the road surface, would help.
- **Pedestrians and debris hurt even though lidar and radar can see them.** A pedestrian or a cone looks like nothing the networks met in training: too small to be a car and in the wrong place. 28–30% of runs with pedestrians end in a hit.
- **More sensors don't help with hazards they weren't trained on.** The everything-on specialist is no better than the default set here.

This is the expected result for an evaluation-first step. The levels measure what the current models can't do; the next step is to train with hazards switched on (see the roadmap).
