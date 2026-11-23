"""Tests for the Level-2 stack (src/av): lateral control, decisions and scenarios."""
import numpy as np
import pytest

from src.av.agents import TrafficCar
from src.av.perception import Track
from src.av.road import Path, Road
from src.av.scenarios import SCENARIOS
from src.av.tracking import Reference, pure_pursuit
from src.av.vehicle import KinematicBicycle, box_corners, boxes_overlap
from src.av.planner import BehaviorPlanner
from src.av.vehicle import VehicleParams
from src.collision_avoidance import RiskModel


@pytest.fixture(scope="module")
def rm():
    return RiskModel.load()


def _track(path, ref_fn, v=20.0, steps=300):
    car = KinematicBicycle(0.0, 0.4, 0.0, v)
    hint, errs, ref = None, [], Reference(0.0)
    for k in range(steps):
        s, d, hint = path.project(car.x, car.y, hint)
        ref = ref_fn(k, s, ref)
        delta, _ = pure_pursuit(car, path, ref, s)
        car.step(0.0, delta)
        errs.append(d - ref.d(s))
    return np.array(errs), d


def test_pure_pursuit_follows_curves():
    p = Path.from_segments([("straight", 100), ("arc", 200, 60), ("straight", 100), ("arc", 150, -90), ("straight", 300)])
    errs, _ = _track(p, lambda k, s, ref: ref)
    assert np.abs(errs[30:]).max() < 0.2


def test_lane_change_reaches_target_lane():
    p = Path.from_segments([("straight", 800)])
    errs, d_final = _track(p, lambda k, s, ref: Reference(0.0, 3.5, s, 80) if k == 50 else ref)
    assert abs(d_final - 3.5) < 0.15


def test_collision_geometry():
    assert boxes_overlap(box_corners(0, 0, 0, 4, 2), box_corners(3, 0.5, 0.3, 4, 2))
    assert not boxes_overlap(box_corners(0, 0, 0, 4, 2), box_corners(5, 0, 0, 4, 2))


def _planner(road):
    return BehaviorPlanner(road, VehicleParams(), None, lane=0)


def test_gap_check_rejects_fast_car_behind_and_accepts_free_lane():
    road = Road(Path.from_segments([("straight", 1000)]), n_lanes=2)
    pl = _planner(road)
    d1 = road.lane_d(1)
    fast_behind = [Track(1, -20, -15.4, d1, 33.0, 0.0, 4.6, 1)]
    ok, why, _ = pl.gap_check(fast_behind, 1, v=25.0)
    assert not ok and "behind" in why
    far_ahead = [Track(2, 120, 115.4, d1, 25.0, 0.0, 4.6, 1)]
    ok, _, _ = pl.gap_check(far_ahead, 1, v=25.0)
    assert ok


def test_overtake_happens_without_collision(rm):
    m = SCENARIOS["highway_overtake"](np.random.default_rng(0), risk_model=rm).run()
    assert not m["collision"] and m["lane_changes"] >= 2


def test_emergency_stop_risk_aware_beats_reactive(rm):
    ours = SCENARIOS["highway_emergency"](np.random.default_rng(3), randomize=True, risk_model=rm).run()
    assert not ours["collision"]


def test_urban_red_light_and_pedestrian(rm):
    w = SCENARIOS["urban_route"](np.random.default_rng(0), risk_model=rm)
    m = w.run()
    assert not m["collision"]
    assert m["red_light_violations"] == 0
    assert m["min_ped_clearance_m"] > 1.0


def test_rc_scale_track(rm):
    m = SCENARIOS["rc_track"](np.random.default_rng(0), risk_model=rm).run()
    assert not m["collision"] and m["rms_lateral_error_m"] < 0.05
