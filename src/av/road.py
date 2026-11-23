"""Road geometry: arc-length parameterised paths, multi-lane roads and Frenet frames.

A `Path` is a dense polyline (0.5 m spacing) with heading and curvature. A `Road`
is a centre line plus a number of lanes; lane i (0 = rightmost) is centred at a
signed lateral offset d (left of the centre line positive). Every object in the
simulator is located by Frenet coordinates (s along the road, d across it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Path:
    def __init__(self, xy, ds: float = 0.5, closed: bool = False):
        xy = np.asarray(xy, float)
        seg = np.hypot(*np.diff(xy, axis=0).T)
        keep = np.r_[True, seg > 1e-9]
        xy = xy[keep]
        s_raw = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        n = max(int(s_raw[-1] / ds) + 1, 2)
        self.s = np.linspace(0.0, s_raw[-1], n)
        self.x = np.interp(self.s, s_raw, xy[:, 0])
        self.y = np.interp(self.s, s_raw, xy[:, 1])
        dx, dy = np.gradient(self.x, self.s), np.gradient(self.y, self.s)
        self.heading = np.unwrap(np.arctan2(dy, dx))
        self.kappa = np.gradient(self.heading, self.s)
        self.length = float(self.s[-1])
        self.closed = closed

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_segments(cls, segments, x0=0.0, y0=0.0, heading0=0.0, step=0.25, closed=False, ds=0.5):
        """segments: ("straight", length) or ("arc", radius, angle_deg); +angle turns left.
        ds is the spacing of the final path points (use ~0.02 m for model-car tracks)."""
        step = min(step, ds / 2)
        pts = [(x0, y0)]
        x, y, h = x0, y0, heading0
        for seg in segments:
            if seg[0] == "straight":
                n = max(int(seg[1] / step), 1)
                for _ in range(n):
                    x += seg[1] / n * np.cos(h); y += seg[1] / n * np.sin(h)
                    pts.append((x, y))
            else:
                r, ang = seg[1], np.radians(seg[2])
                n = max(int(abs(ang) * r / step), 2)
                for _ in range(n):
                    dh = ang / n
                    chord = 2 * r * np.sin(abs(dh) / 2)
                    x += chord * np.cos(h + dh / 2); y += chord * np.sin(h + dh / 2)
                    h += dh
                    pts.append((x, y))
        return cls(pts, ds=ds, closed=closed)

    # ---------------------------------------------------------------- queries
    def _s(self, s):
        return np.mod(s, self.length) if self.closed else np.clip(s, 0.0, self.length)

    def at(self, s):
        """(x, y, heading, curvature) at arc length s (vectorised)."""
        s = self._s(np.asarray(s, float))
        return (np.interp(s, self.s, self.x), np.interp(s, self.s, self.y),
                np.interp(s, self.s, self.heading), np.interp(s, self.s, self.kappa))

    def frenet_to_xy(self, s, d):
        x, y, h, _ = self.at(s)
        return x - d * np.sin(h), y + d * np.cos(h), h

    def project(self, x, y, hint: int | None = None, window: int = 200):
        """Nearest point: returns (s, d, index). d > 0 means left of the path."""
        if hint is None:
            idx = np.arange(len(self.s))
        else:
            lo, hi = hint - window, hint + window
            idx = np.arange(lo, hi) % len(self.s) if self.closed else np.arange(max(lo, 0), min(hi, len(self.s)))
        d2 = (self.x[idx] - x) ** 2 + (self.y[idx] - y) ** 2
        i = int(idx[np.argmin(d2)])
        # refine on the neighbouring segment
        j = (i + 1) % len(self.s) if self.closed else min(i + 1, len(self.s) - 1)
        if j == i:
            j = i - 1
        ax, ay = self.x[i], self.y[i]
        bx, by = self.x[j], self.y[j]
        seg = np.array([bx - ax, by - ay]); L2 = float(seg @ seg) or 1e-9
        u = float(np.clip(((x - ax) * seg[0] + (y - ay) * seg[1]) / L2, -1.0, 1.0))
        s = self.s[i] + u * (self.s[j] - self.s[i])
        h = self.heading[i]
        d = -(x - ax) * np.sin(h) + (y - ay) * np.cos(h)
        return float(self._s(s)), float(d), i

    def max_abs_kappa(self, s0, s1):
        if s1 <= s0:
            return 0.0
        ss = np.arange(s0, s1, 1.0)
        return float(np.max(np.abs(self.at(ss)[3]))) if len(ss) else 0.0


@dataclass
class TrafficLight:
    s: float                    # stop-line position along the road
    green: float = 12.0
    yellow: float = 3.0
    red: float = 10.0
    offset: float = 0.0         # phase offset [s]

    def state(self, t):
        cyc = self.green + self.yellow + self.red
        p = (t + self.offset) % cyc
        return "green" if p < self.green else "yellow" if p < self.green + self.yellow else "red"


@dataclass
class Road:
    center: Path
    n_lanes: int = 3
    lane_width: float = 3.5
    speed_limit: float = 30.0
    lights: list = field(default_factory=list)
    crosswalks: list = field(default_factory=list)      # s positions
    name: str = "road"

    def lane_d(self, lane: int) -> float:
        return (lane - (self.n_lanes - 1) / 2) * self.lane_width

    def lane_of(self, d: float) -> int:
        return int(np.clip(np.round(d / self.lane_width + (self.n_lanes - 1) / 2), 0, self.n_lanes - 1))

    @property
    def half_width(self):
        return self.n_lanes * self.lane_width / 2

    def boundary_xy(self, d, s_step=1.0):
        ss = np.arange(0.0, self.center.length + 1e-6, s_step)
        x, y, _ = self.center.frenet_to_xy(ss, d)
        return x, y
