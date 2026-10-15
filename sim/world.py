"""Procedural roads and the raycaster every range sensor uses."""
import numpy as np

LANE_WIDTH = 3.5
STEP = 1.0  # metres between centreline samples
EDGE_STEP = 4  # road edges for raycasting use every 4th sample: 4x faster, <7 cm off on the tightest bend
MAX_TURN = np.radians(80)  # total heading stays within +-80 deg, so roads never cross themselves


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Road:
    """A one-way multi-lane road built from straights and arcs.

    Positions along the road are (s, lat): distance along the centreline and
    lateral offset from it, positive to the left.
    """

    def __init__(self, rng, lanes=2, complexity=0.5):
        self.lanes = lanes
        self.width = lanes * LANE_WIDTH
        r_min = 120 - 90 * complexity  # tighter bends on complex roads
        curvature, heading = [], 0.0
        for i in range(int(round(4 + 8 * complexity))):
            if i == 0 or rng.random() < 0.4:
                curvature += [0.0] * int(rng.uniform(40, 120) / STEP)
                continue
            radius = rng.uniform(r_min, 2.5 * r_min)
            angle = np.radians(rng.uniform(20, 80))
            sign = rng.choice([-1, 1])
            if abs(heading + sign * angle) > MAX_TURN:
                sign = -sign
            heading += sign * angle
            curvature += [sign / radius] * int(angle * radius / STEP)

        seg_heading = np.cumsum(curvature) * STEP
        steps = STEP * np.stack([np.cos(seg_heading), np.sin(seg_heading)], 1)
        self.pts = np.vstack([[0.0, 0.0], np.cumsum(steps, 0)])
        self.heading = np.append(seg_heading, seg_heading[-1])  # heading[i] = direction from point i to i+1
        self.s = np.arange(len(self.pts)) * STEP
        self.length = self.s[-1]
        self.normal = np.stack([-np.sin(self.heading), np.cos(self.heading)], 1)

        coarse = np.append(np.arange(0, len(self.pts) - 1, EDGE_STEP), len(self.pts) - 1)
        left = (self.pts + self.normal * self.width / 2)[coarse]
        right = (self.pts - self.normal * self.width / 2)[coarse]
        self.edge_p = np.vstack([left[:-1], right[:-1]])
        self.edge_q = np.vstack([left[1:], right[1:]])

    def lane_center(self, lane):
        return -self.width / 2 + LANE_WIDTH * (np.asarray(lane) + 0.5)

    def pose_at(self, s, lat):
        """(x, y, heading) arrays for road coordinates (s, lat)."""
        s = np.clip(np.asarray(s, float), 0, self.length)
        i = np.minimum((s / STEP).astype(int), len(self.pts) - 2)
        h = self.heading[i]
        f = (s - self.s[i])[..., None]
        p = self.pts[i] + f * np.stack([np.cos(h), np.sin(h)], -1) + np.asarray(lat)[..., None] * self.normal[i]
        return p[..., 0], p[..., 1], h

    def project(self, pos, hint=None):
        """Nearest centreline index, distance along the road, lateral offset."""
        lo, hi = (0, len(self.pts)) if hint is None else (max(hint - 20, 0), min(hint + 60, len(self.pts)))
        d = self.pts[lo:hi] - pos
        i = lo + int(np.argmin(np.einsum("ij,ij->i", d, d)))
        off = pos - self.pts[i]
        h = self.heading[i]
        return i, self.s[i] + off[0] * np.cos(h) + off[1] * np.sin(h), float(off @ self.normal[i])


def raycast(origin, angles, max_dist, p, q):
    """Cast rays from origin at the given world angles against segments p->q.

    Returns (distance, index of the segment hit). Misses get max_dist and -1.
    """
    n = len(angles)
    if len(p) == 0:
        return np.full(n, float(max_dist)), np.full(n, -1)
    d = np.stack([np.cos(angles), np.sin(angles)], 1)[:, None, :]  # (rays, 1, 2)
    e = (q - p)[None]  # (1, segs, 2)
    w = (p - origin)[None]
    den = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / den
        u = (w[..., 0] * d[..., 1] - w[..., 1] * d[..., 0]) / den
    t = np.where((den != 0) & (t >= 0) & (t <= max_dist) & (u >= 0) & (u <= 1), t, np.inf)
    j = t.argmin(1)
    dist = t[np.arange(n), j]
    hit = np.isfinite(dist)
    return np.where(hit, dist, float(max_dist)), np.where(hit, j, -1)
