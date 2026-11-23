"""Scenario library for the 2-D simulator.

Every builder takes an rng and a `randomize` flag: with randomize=False it returns
the canonical scenario (used in the demo and the report figures); with
randomize=True speeds, spacings, event times and decelerations are perturbed so
each call is a new, unseen variant (used for the batch evaluation).
"""
from __future__ import annotations

import numpy as np

from .agents import Pedestrian, TrafficCar
from .planner import PlannerConfig
from .road import Path, Road, TrafficLight
from .vehicle import VehicleParams
from .world import World


def _u(rng, lo, hi, randomize, default):
    return float(rng.uniform(lo, hi)) if randomize else default


def _highway(n_lanes=3):
    center = Path.from_segments([("straight", 300), ("arc", 600, 20), ("straight", 400), ("arc", 500, -25),
                                 ("straight", 500), ("arc", 700, 15), ("straight", 700)])
    return Road(center, n_lanes=n_lanes, lane_width=3.5, speed_limit=33.3, name="3-lane highway")


def highway_overtake(rng, randomize=False, **kw):
    kw, wk = _split(kw)
    road = _highway()
    v_truck = _u(rng, 17, 22, randomize, 20.0)
    v_fast = _u(rng, 31, 36, randomize, 33.0)
    cars = [
        TrafficCar(1, s=_u(rng, 150, 220, randomize, 180), d=road.lane_d(0), v=v_truck, v_des=v_truck,
                   length=12.0, width=2.5, kind="truck"),
        TrafficCar(2, s=_u(rng, 100, 160, randomize, 125), d=road.lane_d(1), v=26, v_des=_u(rng, 25, 28, randomize, 26)),
        TrafficCar(3, s=_u(rng, 0, 35, randomize, 20), d=road.lane_d(1), v=v_fast, v_des=v_fast),
        TrafficCar(4, s=_u(rng, 260, 360, randomize, 300), d=road.lane_d(2), v=31, v_des=31),
    ]
    return World(road, ego_s=50, ego_lane=0, ego_v=25, cars=cars, cfg=PlannerConfig(v_des=30.0, **kw),
                 duration=75, name="Highway: overtake slow truck", seed=int(rng.integers(1e9)), **wk)


def highway_cut_in(rng, randomize=False, **kw):
    kw, wk = _split(kw)
    road = _highway()
    # the cutting car starts ahead in the left lane, slower than the ego car, and swerves in
    # when it is roughly 8-14 m ahead (gap at cut-in = start gap - closing speed x cut-in time)
    v2 = _u(rng, 22, 25, randomize, 23.5)
    t_cut = _u(rng, 2.0, 3.5, randomize, 2.5)
    gap_at_cut = _u(rng, 8, 14, randomize, 10)
    gap0 = gap_at_cut + (28 - v2) * t_cut
    cars = [
        TrafficCar(1, s=80 + 60, d=road.lane_d(1), v=27, v_des=27),
        TrafficCar(2, s=80 + gap0 + 4.6, d=road.lane_d(2), v=v2, v_des=v2,
                   events=[dict(t=t_cut, type="lane_change", lane=1,
                                duration=_u(rng, 1.8, 3.0, randomize, 2.2))]),
        TrafficCar(3, s=80 - 3, d=road.lane_d(0), v=28, v_des=28),
        TrafficCar(4, s=80 - 25, d=road.lane_d(2), v=27, v_des=27),
    ]
    return World(road, ego_s=80, ego_lane=1, ego_v=28, cars=cars, cfg=PlannerConfig(v_des=28.0, **kw),
                 duration=30, name="Highway: aggressive cut-in", seed=int(rng.integers(1e9)), **wk)


def highway_emergency(rng, randomize=False, **kw):
    kw, wk = _split(kw)
    road = _highway()
    gap = _u(rng, 28, 45, randomize, 35)
    decel = _u(rng, 6.0, 8.5, randomize, 7.5)
    cars = [
        TrafficCar(1, s=100 + gap + 4.6, d=road.lane_d(1), v=28, v_des=28,
                   events=[dict(t=_u(rng, 6, 10, randomize, 8.0), type="brake", decel=decel)]),
        TrafficCar(2, s=100 + 2, d=road.lane_d(0), v=28, v_des=28,
                   events=[dict(t=8.5, type="brake", decel=4.0)]),
        TrafficCar(3, s=100 + 8, d=road.lane_d(2), v=28, v_des=28,
                   events=[dict(t=8.5, type="brake", decel=4.0)]),
        TrafficCar(4, s=100 - 40, d=road.lane_d(1), v=28, v_des=28),
    ]
    return World(road, ego_s=100, ego_lane=1, ego_v=28, cars=cars, cfg=PlannerConfig(v_des=28.0, **kw),
                 duration=25, name="Highway: sudden stop ahead",
                 seed=int(rng.integers(1e9)), **wk)


def urban_route(rng, randomize=False, **kw):
    kw, wk = _split(kw)
    center = Path.from_segments([("straight", 120), ("arc", 20, 90), ("straight", 150), ("arc", 25, -90),
                                 ("straight", 110), ("arc", 30, 60), ("straight", 140)])
    road = Road(center, n_lanes=2, lane_width=3.25, speed_limit=13.9, name="urban route (50 km/h)",
                lights=[TrafficLight(s=110.0, green=20, yellow=3, red=12, offset=_u(rng, 14, 18, randomize, 17))],
                crosswalks=[290.0])
    peds = [Pedestrian(1, s=290.0, d_start=-(road.half_width + 2.0), d_end=road.half_width + 2.0,
                       t_start=None, speed=_u(rng, 1.1, 1.8, randomize, 1.4),
                       trigger_ds=_u(rng, 30, 45, randomize, 38))]
    cars = [TrafficCar(1, s=_u(rng, 40, 60, randomize, 45), d=road.lane_d(0), v=11, v_des=12.5),
            TrafficCar(2, s=0, d=road.lane_d(1), v=12, v_des=13.5)]
    return World(road, ego_s=12, ego_lane=0, ego_v=10, cars=cars, peds=peds,
                 cfg=PlannerConfig(v_des=13.9, a_lat_max=2.0, **kw), duration=80,
                 name="Urban: turns, traffic light, pedestrian", seed=int(rng.integers(1e9)), **wk)


def rc_track(rng, randomize=False, **kw):
    """1:10-scale oval track: the same software driving a model car."""
    kw, wk = _split(kw)
    center = Path.from_segments([("straight", 3.0), ("arc", 1.0, 180), ("straight", 3.0), ("arc", 1.0, 180)],
                                closed=True, ds=0.02)
    road = Road(center, n_lanes=1, lane_width=0.45, speed_limit=1.2, name="RC oval track (1:10)",
                lights=[TrafficLight(s=1.5, green=18, yellow=1.5, red=6, offset=_u(rng, -3, 3, randomize, 0))])
    lead = TrafficCar(1, s=_u(rng, 1.8, 2.6, randomize, 2.2), d=0.0, v=0.6, v_des=_u(rng, 0.45, 0.7, randomize, 0.6),
                      length=0.42, width=0.2, T=0.8,
                      events=[dict(t=_u(rng, 10, 14, randomize, 12), type="brake", decel=2.5, duration=3.0)])
    cfg = PlannerConfig(v_des=1.0, allow_lane_change=False, a_lat_max=0.8, light_stop_margin=0.1, ped_margin=0.3,
                        **kw)
    w = World(road, ego_s=0.2, ego_lane=0, ego_v=0.5, cars=[lead], cfg=cfg, params=VehicleParams.rc_car(),
              duration=45, name="RC track: follow, stop light, emergency stop", seed=int(rng.integers(1e9)),
              **wk)
    # scale the speed controller to the model car
    for ctrl in (w.planner.long, w.planner.long_ped, w.planner.long_light):
        ctrl.idm.update(s0=0.25, T=0.8, a_max=0.6, b_comf=0.8, v0=1.0)
        ctrl.acc_min = -1.2
        ctrl.aeb.update(a_partial=-1.5, a_full=-3.0, drac_min=0.3)
    w.noise = 0.05            # sensor noise scaled to centimetre-level ranges
    return w


def _split(kw):
    """Separate World arguments (risk model, sensor noise) from planner settings."""
    kw = dict(kw)
    wk = {k: kw.pop(k) for k in ("risk_model", "sensor_noise") if k in kw}
    return kw, wk


SCENARIOS = {
    "highway_overtake": highway_overtake,
    "highway_cut_in": highway_cut_in,
    "highway_emergency": highway_emergency,
    "urban_route": urban_route,
    "rc_track": rc_track,
}
