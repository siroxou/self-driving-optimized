"""Vehicle model and collision geometry, shared by the ego car and traffic."""
import numpy as np

DT = 0.1  # seconds per sim step
WHEELBASE, LENGTH, WIDTH = 2.7, 4.5, 1.9
MAX_STEER = np.radians(35)
MAX_SPEED = 25.0  # m/s (90 km/h)
MAX_ACCEL, MAX_BRAKE = 4.0, 8.0  # m/s^2
DISC_OFFSET, DISC_RADIUS = 1.2, 1.05  # two discs cover the body for collision checks


class Car:
    """Kinematic bicycle model. Action is [steering, pedal], both in [-1, 1]."""

    def __init__(self, x, y, yaw, v=0.0):
        self.x, self.y, self.yaw, self.v = float(x), float(y), float(yaw), float(v)
        self.last_yaw = self.yaw
        self.steer = 0.0
        self.action = np.zeros(2)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def step(self, action):
        self.action = np.clip(np.asarray(action, float), -1, 1)
        steer, pedal = self.action
        self.steer = steer * MAX_STEER
        self.v = float(np.clip(self.v + pedal * (MAX_ACCEL if pedal >= 0 else MAX_BRAKE) * DT, 0, MAX_SPEED))
        self.last_yaw = self.yaw
        self.yaw += self.v / WHEELBASE * np.tan(self.steer) * DT
        self.x += self.v * np.cos(self.yaw) * DT
        self.y += self.v * np.sin(self.yaw) * DT


def box_segments(x, y, yaw):
    """Four outline segments per car, for raycasting. Segment j belongs to car j // 4."""
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    dx = np.array([1, 1, -1, -1]) * LENGTH / 2
    dy = np.array([1, -1, -1, 1]) * WIDTH / 2
    corners = np.stack([x[:, None] + dx * c - dy * s, y[:, None] + dx * s + dy * c], -1)
    return corners.reshape(-1, 2), np.roll(corners, -1, axis=1).reshape(-1, 2)


def discs(x, y, yaw):
    """Front and rear disc centres, shape (n, 2, 2)."""
    off = DISC_OFFSET * np.stack([np.cos(yaw), np.sin(yaw)], -1)
    centre = np.stack([x, y], -1)
    return np.stack([centre + off, centre - off], -2)


def collides(car, x, y, yaw):
    if len(x) == 0:
        return False
    ego = discs(np.array([car.x]), np.array([car.y]), np.array([car.yaw]))[0]  # (2, 2)
    gaps = np.linalg.norm(discs(x, y, yaw)[:, :, None] - ego[None, None], axis=-1)
    return bool((gaps < 2 * DISC_RADIUS).any())
