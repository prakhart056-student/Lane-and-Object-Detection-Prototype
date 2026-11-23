"""Run every simulated RC-track scenario and save the results for the report.

    python -m rc_car.evaluate        ->  reports/rc_results.json, reports/figures/fig15_rc_view.png
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from src import config as C

from .autopilot import Autopilot
from .config import RCConfig
from .hal import SimCar
from .lane_detection import LaneDetector
from .main import overlay

SCEN = {"empty": "Lane following only", "follow": "Following a car that brakes", "obstacle": "Stationary box ahead",
        "stopline": "Stop line (stop, wait, go)", "light": "Traffic light (red / green)", "full": "All combined"}


def run_one(scenario, seconds=45, snapshots=None):
    cfg = RCConfig()
    car, det, ap = SimCar(cfg, scenario), LaneDetector(cfg), Autopilot(cfg)
    ap.start()
    states = {}
    for _ in range(int(seconds * cfg.rate_hz)):
        s = car.read()
        lane, dbg = det.detect(s.frame)
        stop, light = det.detect_stop_line(dbg["bev"]), det.detect_light(s.frame)
        cmd = ap.update(s.t, lane, s.range_m, stop, light, s.range_t)
        states[cmd.state] = states.get(cmd.state, 0) + 1
        if snapshots is not None and any(abs(s.t - t) < 1e-6 for t in snapshots):
            snapshots[round(s.t, 2)] = overlay(cfg, det, dbg, lane, cmd, s, stop, light, s.frame)
        car.command(cmd.steer, cmd.speed, cmd.brake)
        car.advance()
    e = np.array(car.lat_err)
    return dict(scenario=SCEN[scenario], distance_m=car.distance, rms_lateral_cm=float(np.sqrt(np.mean(e ** 2)) * 100),
                max_lateral_cm=float(np.abs(e).max() * 100), collisions=car.collisions,
                closest_cm=None if not np.isfinite(car.min_gap) else float(car.min_gap * 100),
                states={k: round(v / cfg.rate_hz, 1) for k, v in states.items()})


def main():
    res = {}
    snaps = {2.6: None, 6.0: None}
    for sc in SCEN:
        res[sc] = run_one(sc, snapshots=snaps if sc == "full" else None)
        r = res[sc]
        print(f"  {r['scenario']:30s} collisions {r['collisions']}  lane RMS {r['rms_lateral_cm']:.1f} cm  "
              f"closest {r['closest_cm'] if r['closest_cm'] is not None else '-'} cm")
    with open(C.REPORTS / "rc_results.json", "w") as f:
        json.dump(res, f, indent=2)
    imgs = [v for v in snaps.values() if v is not None]
    if imgs:
        cv2.imwrite(str(C.FIGURES / "fig15_rc_view.png"), np.vstack(imgs))
    return res


if __name__ == "__main__":
    main()
