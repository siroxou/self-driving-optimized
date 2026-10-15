"""Pedestrians, road debris and potholes, used by the evaluation levels (sim/levels.py).

All three are per-km densities in the scenario and default to 0. With all of them
at 0 the env never creates this object, so ordinary runs are unchanged.

- Pedestrians: one per crosswalk. Each waits at the kerb, crosses the whole road,
  waits on the other side and crosses back. They don't step out while a car is
  close to the crosswalk, but anyone already crossing has to be waited for, and
  traffic yields the whole crosswalk while they do.
  Lidar, sonar and radar can see them. Hitting one ends the run.
- Obstacles: debris in the rightmost lane (the same rule as parked cars, so
  traffic, which never changes lane, can't get stuck). A row of cones or a fallen
  load. Lidar and sonar see them; hitting one is a crash.
- Potholes: holes in the road surface. No current sensor can see them. Each hit
  costs speed; a hit above POTHOLE_DAMAGE_SPEED ends the run as "damaged".
"""
import numpy as np

from .car import DISC_RADIUS, DT, box_segments, discs
from .world import LANE_WIDTH

PED_SIZE = 0.6  # metres, square footprint
PED_SPEED = (1.0, 1.6)  # m/s
PED_WAIT = (2.0, 10.0)  # seconds at the kerb between crossings
PED_KERB = 1.5  # metres beyond the road edge where pedestrians wait
PED_CLEAR = 12.0  # don't step out if a car is this close to the crosswalk
CONE, LOAD = 0.4, 1.4  # obstacle piece sizes, metres
POTHOLE_RADIUS = (0.4, 0.9)
POTHOLE_SLOWDOWN = 0.7  # speed kept after a hit
POTHOLE_DAMAGE_SPEED = 40 / 3.6  # m/s; faster hits end the run
WHEEL_REACH = 0.8  # a pothole counts as hit if it's this close to the centre line of the car's discs


class Hazards:
    def __init__(self, road, rng, pedestrians=0.0, obstacles=0.0, potholes=0.0, start_s=5.0, parked_s=()):
        self.road = road
        km = road.length / 1000
        kerb = road.width / 2 + PED_KERB

        # crosswalks, one pedestrian each
        self.crosswalks = self._spread(rng, round(pedestrians * km), start_s + 40, road.length - 30, 25)
        n = len(self.crosswalks)
        side = rng.choice([-1.0, 1.0], n)
        self.ped_s = self.crosswalks.copy()
        self.ped_lat = side * kerb
        self.ped_dir = -side  # walk towards the other kerb
        self.ped_speed = rng.uniform(*PED_SPEED, n)
        self.ped_wait = rng.uniform(0, PED_WAIT[1], n)
        self.rng = rng
        self.kerb = kerb

        # obstacles: rightmost lane only, away from parked cars
        pieces = []
        if road.lanes > 1:
            lane0 = road.lane_center(0)
            for s in self._spread(rng, round(obstacles * km), start_s + 30, road.length - 20, 20, avoid=parked_s):
                if rng.random() < 0.5:  # three cones across the lane, staggered
                    for k in range(3):
                        pieces.append((s + 1.5 * k, lane0 - 1.0 + 1.0 * k, CONE))
                else:  # a fallen load
                    pieces.append((s, lane0 + rng.uniform(-0.6, 0.6), LOAD))
        self.obs_s, self.obs_lat, self.obs_size = (np.array(v, float) for v in zip(*pieces)) if pieces else (np.zeros(0),) * 3

        # potholes: anywhere in the lanes
        m = round(potholes * km)
        self.hole_s = np.sort(rng.uniform(start_s + 20, road.length - 15, m))
        self.hole_lat = rng.uniform(-road.width / 2 + 0.6, road.width / 2 - 0.6, m)
        self.hole_r = rng.uniform(*POTHOLE_RADIUS, m)
        self.hole_hit = np.zeros(m, bool)
        self.refresh()

    def refresh(self):
        """Recompute cached world geometry; call after moving hazards by hand (tests do)."""
        self.obs_x, self.obs_y, self.obs_yaw = self.road.pose_at(self.obs_s, self.obs_lat)
        self.obs_p, self.obs_q = box_segments(self.obs_x, self.obs_y, self.obs_yaw, self.obs_size, self.obs_size)
        self.hole_x, self.hole_y, _ = self.road.pose_at(self.hole_s, self.hole_lat)
        self._update_poses()

    @staticmethod
    def _spread(rng, count, lo, hi, gap, avoid=()):
        """Up to `count` positions in [lo, hi], at least `gap` apart and from anything in `avoid`."""
        out = []
        for _ in range(count * 10):
            if len(out) == count or hi <= lo:
                break
            s = rng.uniform(lo, hi)
            if all(abs(s - o) > gap for o in out) and all(abs(s - a) > gap / 2 for a in avoid):
                out.append(s)
        return np.sort(np.array(out, float))

    # ---------- pedestrians ----------
    def _update_poses(self):
        """World pose and velocity of every pedestrian, computed once per step."""
        x, y, h = self.road.pose_at(self.ped_s, self.ped_lat)
        self.ped_x, self.ped_y, self.ped_yaw = x, y, h + self.ped_dir * np.pi / 2
        speed = np.where(self.ped_wait <= 0, self.ped_speed, 0.0) * self.ped_dir
        self.ped_vx, self.ped_vy = -np.sin(h) * speed, np.cos(h) * speed
        self.ped_p, self.ped_q = box_segments(x, y, self.ped_yaw, PED_SIZE, PED_SIZE)

    def step(self, vehicle_s):
        """Move pedestrians. `vehicle_s`: positions along the road of every moving car, ego included."""
        if not len(self.ped_s):
            return
        crossing = self.ped_wait <= 0
        self.ped_lat = np.where(crossing, self.ped_lat + self.ped_dir * self.ped_speed * DT, self.ped_lat)
        arrived = crossing & (self.ped_dir * self.ped_lat >= self.kerb)
        self.ped_lat = np.where(arrived, self.ped_dir * self.kerb, self.ped_lat)
        self.ped_dir = np.where(arrived, -self.ped_dir, self.ped_dir)
        self.ped_wait = np.where(arrived, self.rng.uniform(*PED_WAIT, len(self.ped_s)), self.ped_wait)

        waiting = ~crossing & ~arrived
        near = np.abs(np.asarray(vehicle_s)[None, :] - (self.ped_s[:, None] - PED_CLEAR / 2)) < PED_CLEAR / 2 + 3
        hold = waiting & (self.ped_wait <= DT) & near.any(1)  # ready, but a car is close: keep waiting
        self.ped_wait = np.where(waiting & ~hold, self.ped_wait - DT, self.ped_wait)
        self._update_poses()

    def crossing(self):
        return self.ped_wait <= 0

    def blockers(self):
        """Crosswalks with someone crossing, for traffic to stop at: (s, lat, lane reach).
        Like a real driver, traffic yields the whole crosswalk, not just the lane the pedestrian is in."""
        c = self.crossing()
        return self.ped_s[c], np.zeros(c.sum()), np.full(c.sum(), np.inf)

    # ---------- geometry for sensors and collisions ----------
    def segments(self):
        """Pedestrian outlines then obstacle outlines, 4 segments each."""
        return np.vstack([self.ped_p, self.obs_p]), np.vstack([self.ped_q, self.obs_q])

    def radar_targets(self):
        """Pedestrian positions and velocities (x, y, vx, vy)."""
        return self.ped_x, self.ped_y, self.ped_vx, self.ped_vy

    def contact(self, car):
        """What the car touched this step: ('pedestrian' | 'obstacle' | None, number of new pothole hits)."""
        ego = discs(np.array([car.x]), np.array([car.y]), np.array([car.yaw]))[0]
        near = lambda x, y: np.linalg.norm(np.stack([x, y], -1)[:, None] - ego[None], axis=-1)
        hit = None
        if len(self.ped_s) and (near(self.ped_x, self.ped_y) < DISC_RADIUS + PED_SIZE / 2).any():
            hit = "pedestrian"
        elif len(self.obs_s) and (near(self.obs_x, self.obs_y) < DISC_RADIUS + self.obs_size[:, None] * 0.6).any():
            hit = "obstacle"
        holes = 0
        if len(self.hole_s):
            new = ~self.hole_hit & (near(self.hole_x, self.hole_y).min(1) < self.hole_r + WHEEL_REACH)
            self.hole_hit |= new
            holes = int(new.sum())
        return hit, holes
