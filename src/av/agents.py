"""Traffic participants: IDM-driven vehicles with scripted events, and pedestrians.

Traffic vehicles live in Frenet coordinates (s, d) on the road. They follow the
nearest vehicle ahead in their lane (the ego car included) with the Intelligent
Driver Model, and can be scripted to brake, change speed or change lanes at a
given time, which is how cut-ins and emergency stops are staged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config as C
from .tracking import smoothstep5


@dataclass
class TrafficCar:
    id: int
    s: float
    d: float
    v: float
    v_des: float
    length: float = 4.6
    width: float = 1.85
    kind: str = "car"                       # "car" or "truck"
    events: list = field(default_factory=list)   # dicts with "t" and "type"
    a: float = 0.0
    T: float = 1.4
    brake_until: float = -1.0
    brake_decel: float = 0.0
    lc: dict | None = None                  # active lane change {d0, d1, t0, dur}

    def heading_offset(self):
        """Yaw relative to the road while changing lanes (for drawing)."""
        if not self.lc:
            return 0.0
        dd = (self.lc["d1"] - self.lc["d0"]) / self.lc["dur"]
        return float(np.arctan2(dd * 1.875, max(self.v, 1.0)))   # 1.875 = peak slope of the quintic


@dataclass
class Pedestrian:
    id: int
    s: float                 # crosswalk position along the road
    d_start: float           # lateral start (beyond the road edge)
    d_end: float
    t_start: float | None        # None: start when the ego car is within trigger_ds
    speed: float = 1.4
    trigger_ds: float = 0.0
    d: float = 0.0
    active: bool = False
    done: bool = False

    def __post_init__(self):
        self.d = self.d_start

    def update(self, t, dt):
        if self.done or self.t_start is None or t < self.t_start:
            return
        self.active = True
        step = np.sign(self.d_end - self.d) * self.speed * dt
        if abs(self.d_end - self.d) <= abs(step):
            self.d, self.done, self.active = self.d_end, True, False
        else:
            self.d += step


def idm(v, gap, dv, v0, T=1.4, s0=2.0, a_max=1.5, b=2.0):
    s_star = s0 + max(0.0, v * T + v * dv / (2 * np.sqrt(a_max * b)))
    inter = (s_star / max(gap, 0.1)) ** 2 if gap is not None else 0.0
    return a_max * (1 - (v / max(v0, 0.1)) ** 4 - inter)


def step_traffic(cars, ego_frenet, road, t, dt=C.DT):
    """Advance all traffic cars one step. ego_frenet = (s, d, v, length)."""
    lw = road.lane_width
    objs = [(c.s, c.d, c.v, c.length, c.id) for c in cars] + [(ego_frenet[0], ego_frenet[1], ego_frenet[2],
                                                               ego_frenet[3], -1)]
    L = road.center.length
    for c in cars:
        for ev in c.events:
            if not ev.get("done") and t >= ev["t"]:
                ev["done"] = True
                if ev["type"] == "brake":
                    c.brake_until, c.brake_decel = t + ev.get("duration", 99), ev["decel"]
                elif ev["type"] == "speed":
                    c.v_des = ev["v"]
                elif ev["type"] == "lane_change":
                    c.lc = dict(d0=c.d, d1=road.lane_d(ev["lane"]), t0=t, dur=ev.get("duration", 3.0))
        # leader in own lane (or in the target lane while changing)
        lanes_d = [c.d] + ([c.lc["d1"]] if c.lc else [])
        gap, dv = None, 0.0
        for (s2, d2, v2, l2, i2) in objs:
            if i2 == c.id:
                continue
            if min(abs(d2 - dd) for dd in lanes_d) > lw * 0.6:
                continue
            ds = s2 - c.s
            if road.center.closed:
                ds = (ds + L / 2) % L - L / 2
            if ds <= 0:
                continue
            g = ds - (c.length + l2) / 2
            if gap is None or g < gap:
                gap, dv = g, c.v - v2
        for tl in road.lights:                      # traffic obeys red / yellow lights
            dist = tl.s - c.s - c.length / 2 - 1.0
            if road.center.closed:
                dist %= L
            if tl.state(t) != "green" and 0 < dist < 80 and c.v ** 2 / (2 * dist) < 4.0:
                if gap is None or dist < gap:
                    gap, dv = dist, c.v
        a = idm(c.v, gap, dv, c.v_des, T=c.T)
        if t < c.brake_until:
            a = min(a, -c.brake_decel)
        c.a = float(np.clip(a, -9.0, 2.0))
        c.v = max(c.v + c.a * dt, 0.0)
        c.s += c.v * dt
        if road.center.closed:
            c.s %= L
        if c.lc:
            u = (t - c.lc["t0"]) / c.lc["dur"]
            c.d = c.lc["d0"] + (c.lc["d1"] - c.lc["d0"]) * smoothstep5(u)
            if u >= 1:
                c.d, c.lc = c.lc["d1"], None
