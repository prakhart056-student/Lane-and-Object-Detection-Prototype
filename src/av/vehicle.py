"""Kinematic bicycle model with realistic actuator behaviour.

State is referenced to the rear axle (x, y), heading `yaw` (rad, CCW from +x) and
speed `v`. Steering follows the command through a rate limit and a first-order
lag; acceleration commands pass through a pure delay (perception + actuation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config as C


@dataclass
class VehicleParams:
    wheelbase: float = 2.8
    length: float = 4.6
    width: float = 1.85
    rear_overhang: float = 0.9           # rear axle to rear bumper
    max_steer: float = np.radians(35)
    max_steer_rate: float = np.radians(90)   # rad/s at the road wheels
    steer_tau: float = 0.12              # s, first-order steering lag
    a_min: float = C.VEHICLE_A_MIN
    a_max: float = C.VEHICLE_A_MAX
    accel_delay: float = C.REACTION_DELAY_S

    @classmethod
    def rc_car(cls):
        """1:10 scale RC car (typical 260 mm wheelbase chassis)."""
        return cls(wheelbase=0.26, length=0.42, width=0.20, rear_overhang=0.08,
                   max_steer=np.radians(25), max_steer_rate=np.radians(300), steer_tau=0.08,
                   a_min=-3.0, a_max=1.5, accel_delay=0.1)


@dataclass
class KinematicBicycle:
    x: float
    y: float
    yaw: float
    v: float
    params: VehicleParams = field(default_factory=VehicleParams)
    dt: float = C.DT

    def __post_init__(self):
        self.delta = 0.0
        self.a = 0.0
        n = int(round(self.params.accel_delay / self.dt))
        self._queue = [0.0] * n

    # ------------------------------------------------------------------ geometry
    @property
    def center(self):
        off = self.params.length / 2 - self.params.rear_overhang
        return self.x + off * np.cos(self.yaw), self.y + off * np.sin(self.yaw)

    @property
    def front_axle(self):
        L = self.params.wheelbase
        return self.x + L * np.cos(self.yaw), self.y + L * np.sin(self.yaw)

    def corners(self):
        cx, cy = self.center
        return box_corners(cx, cy, self.yaw, self.params.length, self.params.width)

    @property
    def lat_accel(self):
        return self.v ** 2 * np.tan(self.delta) / self.params.wheelbase

    # ------------------------------------------------------------------ dynamics
    def step(self, a_cmd: float, delta_cmd: float):
        p = self.params
        a_cmd = float(np.clip(a_cmd, p.a_min, p.a_max))
        self._queue.append(a_cmd)
        a = self._queue.pop(0) if len(self._queue) > 0 else a_cmd
        if self.v <= 1e-3 and a < 0:
            a = 0.0
        self.a = a

        target = float(np.clip(delta_cmd, -p.max_steer, p.max_steer))
        lagged = self.delta + (target - self.delta) * min(self.dt / p.steer_tau, 1.0)
        max_step = p.max_steer_rate * self.dt
        self.delta = float(np.clip(lagged, self.delta - max_step, self.delta + max_step))

        dt = self.dt
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw = wrap(self.yaw + self.v / p.wheelbase * np.tan(self.delta) * dt)
        self.v = max(self.v + a * dt, 0.0)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def box_corners(cx, cy, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2, width / 2
    pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([cx, cy])


def boxes_overlap(a, b):
    """Separating-axis test for two convex quadrilaterals (4x2 corner arrays)."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True
