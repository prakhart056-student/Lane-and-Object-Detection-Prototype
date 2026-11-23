"""Tests for the RC-car stack: camera perception, autopilot decisions, simulated drives."""
import numpy as np
import pytest

from rc_car.autopilot import Autopilot
from rc_car.config import RCConfig
from rc_car.hal import SimCar, speed_to_us, steer_to_us
from rc_car.lane_detection import LaneDetector


@pytest.fixture
def cfg():
    return RCConfig()


def _drive(cfg, scenario, seconds):
    car, det, ap = SimCar(cfg, scenario), LaneDetector(cfg), Autopilot(cfg)
    ap.start()
    for _ in range(int(seconds * cfg.rate_hz)):
        s = car.read()
        lane, dbg = det.detect(s.frame)
        cmd = ap.update(s.t, lane, s.range_m, det.detect_stop_line(dbg["bev"]), det.detect_light(s.frame), s.range_t)
        car.command(cmd.steer, cmd.speed, cmd.brake)
        car.advance()
    return car, ap


def test_lane_detector_measures_offset(cfg):
    car = SimCar(cfg, "empty")
    lane, _ = LaneDetector(cfg).detect(car.read().frame)
    assert lane is not None and lane.n_lines == 2
    assert abs(lane.offset) < 0.03                     # car starts centred


def test_stop_line_and_light_detection(cfg):
    car = SimCar(cfg, "light")
    x, y, h = car.track.frenet_to_xy(1.7, 0.0)            # put the car ~1 m before the line at s = 3.0
    car.car.x, car.car.y, car.car.yaw = float(x), float(y), float(h)
    s_center, _ = car._car_s()
    true_dist = 3.0 - (s_center + cfg.car_length / 2)
    det = LaneDetector(cfg)
    frame = car.read().frame
    stop = det.detect_stop_line(det.birdseye(frame))
    assert stop is not None and abs(stop - true_dist) < 0.06
    assert det.detect_light(frame) == car.light_state()


def test_drives_laps_without_collision(cfg):
    car, ap = _drive(cfg, "full", 40)
    e = np.array(car.lat_err)
    assert car.collisions == 0
    assert np.sqrt(np.mean(e ** 2)) < 0.06
    states = {m for _, m in ap.events}
    assert {"STOP_RED", "FOLLOW"} <= states


def test_obstacle_stops_before_contact(cfg):
    car, _ = _drive(cfg, "obstacle", 20)
    assert car.collisions == 0 and car.min_gap > 0.10


def test_watchdog_and_estop(cfg):
    ap = Autopilot(cfg)
    ap.start()
    cmd = ap.update(0.0, None, None)
    assert cmd.state.startswith("FAULT") and cmd.speed == 0.0
    ap.estop()
    cmd = ap.update(0.1, None, 1.0)
    assert cmd.state == "ESTOP" and cmd.speed == 0.0 and cmd.brake


def test_lane_lost_brings_car_to_stop(cfg):
    ap = Autopilot(cfg)
    ap.start()
    ap.v_set = ap.v_est = 0.8
    for k in range(60):
        cmd = ap.update(k * 0.05, None, 2.5)
    assert cmd.state == "LANE_LOST" and cmd.speed < 0.05


def test_pulse_mapping_is_safe(cfg):
    assert steer_to_us(cfg, 0.0) == cfg.steer_center_us
    assert speed_to_us(cfg, 0.0, False) == cfg.esc_neutral_us
    assert speed_to_us(cfg, 99.0, False) == cfg.esc_max_us
    assert speed_to_us(cfg, 0.5, True) == cfg.esc_brake_us
