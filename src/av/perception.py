"""Sensor model: what the ego vehicle 'sees'.

Produces a noisy object list in Frenet coordinates relative to the ego car
(radar/camera fusion abstraction): longitudinal gap, lateral position, speed and
acceleration of every vehicle within range, pedestrians, and the next traffic
light with its state and the distance to its stop line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRONT_RANGE, REAR_RANGE = 150.0, 80.0
NOISE = dict(gap=0.3, v=0.2, a=0.3, d=0.1)


@dataclass
class Track:
    id: int
    ds: float        # centre-to-centre distance along the road (+ ahead)
    gap: float       # bumper-to-bumper gap (>0), sign follows ds
    d: float
    v: float
    a: float
    length: float
    lane: int


def perceive(world, rng, noise_scale=1.0):
    ego = world.ego_frenet
    road = world.road
    L = road.center.length
    tracks = []
    for c in world.cars:
        ds = c.s - ego["s"]
        if road.center.closed:
            ds = (ds + L / 2) % L - L / 2
        if not (-REAR_RANGE < ds < FRONT_RANGE):
            continue
        gap = abs(ds) - (c.length + world.ego.params.length) / 2
        n = lambda k: rng.normal(0, NOISE[k] * noise_scale)   # noqa: E731
        d_meas = c.d + n("d")
        tracks.append(Track(c.id, ds + n("gap"), np.sign(ds) * max(gap + n("gap"), 0.0), d_meas,
                            max(c.v + n("v"), 0.0), c.a + n("a"), c.length, road.lane_of(d_meas)))
    peds = []
    for p in world.peds:
        ds = p.s - ego["s"]
        if 0 < ds < FRONT_RANGE and (p.active or abs(p.d) < road.half_width + 2.0) and not p.done:
            peds.append(dict(id=p.id, ds=ds, d=p.d, v_d=np.sign(p.d_end - p.d) * p.speed if p.active else 0.0))
    light = None
    for tl in road.lights:
        ds = tl.s - ego["s"]
        if road.center.closed:
            ds = ds % L
        if 0 < ds < FRONT_RANGE and (light is None or ds < light["ds"]):
            light = dict(ds=ds, state=tl.state(world.t))
    return tracks, peds, light


def leader_in(tracks, d_lo, d_hi, margin=1.9):
    """Closest vehicle ahead whose centre lies within [d_lo - margin, d_hi + margin].
    margin 1.9 m catches a car whose body already overlaps a 3.5 m lane (cut-ins)."""
    best = None
    for tr in tracks:
        if tr.ds > 0 and d_lo - margin <= tr.d <= d_hi + margin:
            if best is None or tr.ds < best.ds:
                best = tr
    return best


def follower_in(tracks, d_lo, d_hi, margin=1.9):
    best = None
    for tr in tracks:
        if tr.ds < 0 and d_lo - margin <= tr.d <= d_hi + margin:
            if best is None or tr.ds > best.ds:
                best = tr
    return best
