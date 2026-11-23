"""Closed-loop 2-D world: ego vehicle + traffic + pedestrians + traffic lights.

Each 0.1 s step:  perceive -> plan (behaviour + speed) -> track path (steering)
-> vehicle dynamics -> move traffic -> check collisions / rule violations -> log.
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from .agents import step_traffic
from .perception import perceive
from .planner import BehaviorPlanner, PlannerConfig
from .tracking import pure_pursuit
from .vehicle import KinematicBicycle, VehicleParams, box_corners, boxes_overlap


class World:
    def __init__(self, road, ego_s, ego_lane, ego_v, cars=(), peds=(), cfg: PlannerConfig | None = None,
                 risk_model=None, params: VehicleParams | None = None, duration=60.0, seed=0, name="",
                 sensor_noise=1.0, stop_on_collision=True):
        self.road, self.name, self.duration = road, name, duration
        self.cars, self.peds = list(cars), list(peds)
        self.params = params or VehicleParams()
        off = self.params.length / 2 - self.params.rear_overhang
        cx, cy, h = road.center.frenet_to_xy(ego_s, road.lane_d(ego_lane))
        self.ego = KinematicBicycle(float(cx - off * np.cos(h)), float(cy - off * np.sin(h)), float(h), ego_v,
                                    self.params, C.DT)
        self.cfg = cfg or PlannerConfig()
        self.planner = BehaviorPlanner(road, self.params, risk_model, self.cfg, lane=ego_lane)
        self.t, self.rng, self.noise = 0.0, np.random.default_rng(seed), sensor_noise
        self._hint = None
        self.collision = None
        self.min_ped_clear = np.inf
        self.violations = []
        self.stop_on_collision = stop_on_collision
        self.log = []
        self.target_pt = (cx, cy)
        self.info = {}
        self._update_frenet()

    # ------------------------------------------------------------------ state
    def _update_frenet(self):
        cx, cy = self.ego.center
        s, d, self._hint = self.road.center.project(cx, cy, self._hint)
        self.ego_frenet = dict(s=s, d=d)

    def car_pose(self, c):
        x, y, h = self.road.center.frenet_to_xy(c.s, c.d)
        return float(x), float(y), float(h + c.heading_offset())

    def car_corners(self, c):
        x, y, h = self.car_pose(c)
        return box_corners(x, y, h, c.length, c.width)

    def ped_xy(self, p):
        x, y, _ = self.road.center.frenet_to_xy(p.s, p.d)
        return float(x), float(y)

    @property
    def done(self):
        return self.t >= self.duration - 1e-9 or (self.collision is not None and self.stop_on_collision) or \
            (not self.road.center.closed and self.ego_frenet["s"] > self.road.center.length - 30)

    # ------------------------------------------------------------------ step
    def step(self):
        road, ego = self.road, self.ego
        tracks, peds, light = perceive(self, self.rng, self.noise)
        a_cmd, info = self.planner.plan(self, tracks, peds, light)
        off = self.params.length / 2 - self.params.rear_overhang
        delta, self.target_pt = pure_pursuit(ego, road.center, self.planner.ref, self.ego_frenet["s"] - off,
                                             min_lookahead=max(6.0 * self.params.wheelbase / 2.8, 0.4))
        s_front_before = self.ego_frenet["s"] + self.params.length / 2
        ego.step(a_cmd, delta)
        step_traffic(self.cars, (self.ego_frenet["s"], self.ego_frenet["d"], ego.v, self.params.length),
                     road, self.t)
        for p in self.peds:
            if p.t_start is None and 0 < p.s - self.ego_frenet["s"] < p.trigger_ds:
                p.t_start = self.t
            p.update(self.t, C.DT)
        self.t = round(self.t + C.DT, 6)
        self._update_frenet()

        # collisions
        ec = ego.corners()
        for c in self.cars:
            if self.collision is None and boxes_overlap(ec, self.car_corners(c)):
                self.collision = dict(t=self.t, with_=f"{c.kind} {c.id}", speed=ego.v)
                self.planner.log(self.t, f"COLLISION with {c.kind} {c.id} at {ego.v * 3.6:.0f} km/h")
        for p in self.peds:
            if p.done or not p.active:
                continue
            px, py = self.ped_xy(p)
            cx, cy = ego.center
            local = np.array([px - cx, py - cy]) @ np.array([[np.cos(ego.yaw), -np.sin(ego.yaw)],
                                                             [np.sin(ego.yaw), np.cos(ego.yaw)]])
            clear = max(abs(local[0]) - self.params.length / 2, abs(local[1]) - self.params.width / 2, 0.0)
            self.min_ped_clear = min(self.min_ped_clear, clear)
            if abs(local[0]) < self.params.length / 2 + 0.3 and abs(local[1]) < self.params.width / 2 + 0.3:
                if self.collision is None:
                    self.collision = dict(t=self.t, with_=f"pedestrian {p.id}", speed=ego.v)
                    self.planner.log(self.t, f"COLLISION with pedestrian at {ego.v * 3.6:.0f} km/h")
        # red-light violations: front bumper crosses a stop line while red
        s_front = self.ego_frenet["s"] + self.params.length / 2
        for tl in road.lights:
            if s_front_before < tl.s <= s_front and tl.state(self.t) == "red":
                self.violations.append(dict(t=self.t, type="red light", s=tl.s))
                self.planner.log(self.t, "RED LIGHT VIOLATION")

        lead = info["leader"]
        self.info = info
        self.log.append(dict(t=self.t, x=ego.x, y=ego.y, yaw=ego.yaw, v=ego.v, a=ego.a, delta=ego.delta,
                             s=self.ego_frenet["s"], d=self.ego_frenet["d"],
                             d_ref=float(self.planner.ref.d(self.ego_frenet["s"], road.center.length,
                                                            road.center.closed)),
                             lane_change=self.planner.lc is not None, mode=info["mode"], risk=info["risk"],
                             ttc=info["ttc"], gap=lead.gap if lead is not None else np.nan,
                             lat_acc=ego.lat_accel, stage=info["stage"]))

    def run(self):
        while not self.done:
            self.step()
        return self.metrics()

    # ------------------------------------------------------------------ metrics
    def metrics(self):
        import pandas as pd
        df = pd.DataFrame(self.log)
        keep = ~df.lane_change
        lat_err = (df.d - df.d_ref)[keep]
        return dict(
            scenario=self.name, duration_s=float(self.t), distance_m=float(df.v.sum() * C.DT),
            collision=self.collision is not None, collision_with=(self.collision or {}).get("with_"),
            impact_speed_kmh=(self.collision or {}).get("speed", 0.0) * 3.6,
            red_light_violations=len(self.violations),
            lane_changes=self.planner.stats["lane_changes"], lc_aborts=self.planner.stats["aborts"],
            lc_rejected=self.planner.stats["rejected"],
            rms_lateral_error_m=float(np.sqrt(np.mean(lat_err ** 2))) if len(lat_err) else 0.0,
            max_lateral_error_m=float(np.abs(lat_err).max()) if len(lat_err) else 0.0,
            max_lat_accel=float(np.abs(df.lat_acc).max()), max_decel=float(-df.a.min()),
            min_ttc_s=float(np.nanmin(df.ttc)) if len(df) else np.nan,
            mean_speed_kmh=float(df.v.mean() * 3.6),
            min_ped_clearance_m=float(self.min_ped_clear) if self.peds else np.nan,
        )
