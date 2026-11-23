"""Behaviour planner: the vehicle's decision-making layer.

Every 0.1 s it decides, on its own:
  * which lane to be in       - lane keeping, or a risk-aware lane change
                                (MOBIL incentive + TTC / gap safety gate, with abort)
  * how fast to go            - risk-aware speed control towards the in-path leader,
                                curve speed limits, the speed limit
  * whether to stop           - red / yellow traffic lights (dilemma-zone logic),
                                pedestrians crossing into the path, emergency braking

It outputs an acceleration command and a lateral reference d_ref(s) that the path
tracker (Pure Pursuit) turns into a steering angle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config as C
from ..risk_metrics import composite_risk_index, time_headway, time_to_collision
from .agents import idm
from .longitudinal import RiskAwareLongitudinal
from .perception import follower_in, leader_in
from .tracking import Reference, curve_speed_limit


@dataclass
class PlannerConfig:
    v_des: float = 30.0
    allow_lane_change: bool = True
    risk_aware: bool = True
    a_lat_max: float = 2.5          # comfortable lateral acceleration [m/s²]
    lc_time: float = 4.0            # lane-change duration [s]
    lc_min_len: float = 25.0
    ttc_gap_accept: float = 4.0     # TTC needed in the target lane to start a lane change
    ttc_gap_abort: float = 2.0      # TTC that aborts a lane change in progress
    b_safe: float = 3.0             # max braking imposed on the new follower (MOBIL)
    politeness: float = 0.2
    lc_threshold: float = 0.3       # m/s² of advantage needed to change lanes
    keep_right_bias: float = 0.25
    yellow_stop_decel: float = 3.0  # stop for yellow if this deceleration is enough
    light_stop_margin: float = 1.0  # stop this far before the line [m]
    ped_margin: float = 2.0         # stop this far before a pedestrian's path [m]


def _ttc(gap, v_back, v_front):
    return float(time_to_collision(np.array([max(gap, 0.01)]), np.array([v_back]), np.array([v_front]))[0])


class BehaviorPlanner:
    def __init__(self, road, vehicle_params, risk_model=None, cfg: PlannerConfig | None = None, lane: int = 0):
        self.road, self.vp = road, vehicle_params
        self.cfg = cfg or PlannerConfig()
        self.long = RiskAwareLongitudinal(risk_model, self.cfg.risk_aware)
        self.long_ped = RiskAwareLongitudinal(risk_model, self.cfg.risk_aware)
        self.long_light = RiskAwareLongitudinal(None, False)
        self.lane = lane
        self.ref = Reference(road.lane_d(lane))
        self.lc = None
        self.cooldown_until = 0.0
        self.mode = "CRUISE"
        self.events: list[tuple[float, str]] = []
        self.light_commit = None          # stop-line s we decided to drive through on yellow
        self.stats = dict(lane_changes=0, aborts=0, rejected=0)
        self._last_reject = -99.0
        self._logged: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    def log(self, t, msg, key=None):
        """Event log; repeated messages of the same kind are shown at most every 4 s."""
        key = key or msg
        if t - self._logged.get(key, -99.0) < 4.0:
            return
        self._logged[key] = t
        self.events.append((round(t, 1), msg))
        self.events = self.events[-30:]

    def _lead(self, tracks, d):
        return leader_in(tracks, d, d, margin=self.road.lane_width / 2)

    def _follow(self, tracks, d):
        return follower_in(tracks, d, d, margin=self.road.lane_width / 2)

    def gap_check(self, tracks, lane, v, strict=True):
        """Is the target lane safe? Returns (safe, reason, worst composite risk)."""
        d = self.road.lane_d(lane)
        lf, lr = self._lead(tracks, d), self._follow(tracks, d)
        need = self.cfg.ttc_gap_accept if strict else self.cfg.ttc_gap_abort
        reasons, risks = [], [0.0]
        if lf is not None:
            g = lf.gap
            ttc = _ttc(g, v, lf.v)
            risks.append(float(composite_risk_index(np.array([ttc]), time_headway(np.array([g]), np.array([max(v, 0.1)])),
                                                    np.array([0.0]))[0]))
            if g < max(5.0, 0.4 * v) or ttc < need:
                reasons.append(f"car ahead {g:.0f} m, TTC {min(ttc, 99):.1f} s")
        if lr is not None:
            g = -lr.gap
            ttc = _ttc(g, lr.v, v)
            imposed = idm(lr.v, g, lr.v - v, lr.v + 1.0)
            risks.append(float(composite_risk_index(np.array([ttc]), time_headway(np.array([g]), np.array([max(lr.v, 0.1)])),
                                                    np.array([0.0]))[0]))
            if g < max(5.0, 0.4 * lr.v) or ttc < need or (strict and imposed < -self.cfg.b_safe):
                reasons.append(f"car behind {g:.0f} m closing at {max(lr.v - v, 0):.1f} m/s (TTC {min(ttc, 99):.1f} s)")
        return not reasons, "; ".join(reasons), max(risks)

    def _consider_lane_change(self, tracks, s, v, v0, t):
        road, cfg = self.road, self.cfg
        cur = self.lane
        lead = self._lead(tracks, road.lane_d(cur))
        a_cur = idm(v, lead.gap if lead else None, v - lead.v if lead else 0.0, v0)
        best = None
        for tgt in (cur + 1, cur - 1):                 # left first (overtaking), then right
            if not (0 <= tgt < road.n_lanes):
                continue
            d_t = road.lane_d(tgt)
            lf, lr = self._lead(tracks, d_t), self._follow(tracks, d_t)
            a_new = idm(v, lf.gap if lf else None, v - lf.v if lf else 0.0, v0)
            follower_cost = 0.0
            if lr is not None:
                follower_cost = min(idm(lr.v, -lr.gap, lr.v - v, lr.v + 1.0), 0.0)
            gain = a_new - a_cur + cfg.politeness * follower_cost
            if tgt < cur:      # moving right: keep-right rule makes it attractive when nothing is lost
                incentive = gain + cfg.keep_right_bias
                wanted = incentive > 0 and (lf is None or lf.gap > 60 or lf.v > v0 - 1.0)
            else:              # moving left: only to pass a slower vehicle
                incentive = gain - cfg.lc_threshold
                wanted = incentive > 0 and lead is not None and lead.v < v0 - 2.0 and lead.gap < 90
            if not wanted:
                continue
            safe, why, risk = self.gap_check(tracks, tgt, v, strict=True)
            if safe:
                if best is None or incentive > best[1]:
                    best = (tgt, incentive, risk)
            elif t - self._last_reject > 3.0:
                self._last_reject = t
                self.stats["rejected"] += 1
                self.log(t, f"{'Left' if tgt > cur else 'Right'} lane change rejected: {why}")
        return best

    # ------------------------------------------------------------------ main
    def plan(self, world, tracks, peds, light):
        road, cfg = self.road, self.cfg
        t, v, a = world.t, world.ego.v, world.ego.a
        s, d = world.ego_frenet["s"], world.ego_frenet["d"]
        L, closed = road.center.length, road.center.closed
        v0 = min(cfg.v_des, road.speed_limit,
                 curve_speed_limit(road.center, s, v, a_lat_max=cfg.a_lat_max, v_cap=road.speed_limit))
        front = self.vp.length / 2

        # ---------------- lane management
        near_conflict = (light is not None and light["ds"] < 80) or any(p["ds"] < 80 for p in peds)
        if self.lc is not None:
            prog = self.ref.progress(s, L, closed)
            if not self.lc.get("abort"):
                safe, why, _ = self.gap_check(tracks, self.lc["target"], v, strict=False)
                if not safe and prog < 0.5:
                    origin = self.lc["origin"]
                    self.ref = Reference(d, road.lane_d(origin), s, max(cfg.lc_min_len * 0.8, v * 2.5))
                    self.lc = dict(target=origin, origin=self.lc["target"], abort=True)
                    self.stats["aborts"] += 1
                    self.log(t, f"Lane change ABORTED: {why}")
                    prog = 0.0
            if self.lc is not None and prog >= 1.0:
                self.lane = self.lc["target"]
                self.ref = Reference(road.lane_d(self.lane))
                if not self.lc.get("abort"):
                    self.stats["lane_changes"] += 1
                    self.log(t, f"Lane change complete: now in lane {self.lane + 1}")
                self.lc, self.cooldown_until = None, t + 3.0
        elif cfg.allow_lane_change and road.n_lanes > 1 and t >= self.cooldown_until and v > 8 and not near_conflict:
            choice = self._consider_lane_change(tracks, s, v, v0, t)
            if choice is not None:
                tgt, _, risk = choice
                self.ref = Reference(road.lane_d(self.lane), road.lane_d(tgt), s,
                                     max(cfg.lc_min_len, v * cfg.lc_time))
                self.lc = dict(target=tgt, origin=self.lane)
                self.log(t, f"{'Overtaking: left' if tgt > self.lane else 'Returning: right'} lane change "
                            f"(gap safe, risk {risk:.2f})")

        # ---------------- longitudinal: leader in every lane the car occupies
        if self.lc is not None:
            lo, hi = sorted((self.road.lane_d(self.lc["origin"]), self.road.lane_d(self.lc["target"])))
        else:
            lo = hi = self.ref.d_to
        lead = leader_in(tracks, lo, hi, margin=1.9)
        lead_d = dict(gap=lead.gap, v=lead.v, a=lead.a, id=lead.id) if lead is not None else None
        a_lead = self.long.command(v, a, v0, lead_d)
        info_lead = dict(self.long.last)

        a_light, light_mode = np.inf, None
        if light is not None and light["state"] in ("red", "yellow"):
            dist = light["ds"] - front - cfg.light_stop_margin
            need = v ** 2 / (2 * max(dist, 0.1))
            line_s = (s + light["ds"]) % L if closed else s + light["ds"]
            if light["state"] == "yellow" and self.light_commit != round(line_s) and need > cfg.yellow_stop_decel:
                self.light_commit = round(line_s)
                self.log(t, f"Yellow light {light['ds']:.0f} m ahead: too close to stop comfortably, proceeding")
            if self.light_commit != round(line_s) and dist > -0.5:
                a_light = self.long_light.command(v, a, v0, dict(gap=max(dist, 0.05), v=0.0, a=0.0, id="light"))
                light_mode = "STOP: RED LIGHT" if light["state"] == "red" else "STOP: YELLOW LIGHT"
        elif light is not None and light["state"] == "green":
            self.light_commit = None

        a_ped, ped_mode = np.inf, None
        corridor = (lo - self.road.lane_width / 2 - 0.5, hi + self.road.lane_width / 2 + 0.5)
        for p in peds:
            horizon = min(p["ds"] / max(v, 1.0), 4.0)
            d_future = p["d"] + p["v_d"] * horizon
            crossing = min(p["d"], d_future) <= corridor[1] and max(p["d"], d_future) >= corridor[0]
            if crossing and (p["v_d"] != 0 or corridor[0] <= p["d"] <= corridor[1]):
                gap = p["ds"] - front - cfg.ped_margin
                a_p = self.long_ped.command(v, a, v0, dict(gap=max(gap, 0.05), v=0.0, a=0.0, id=f"ped{p['id']}"))
                if a_p < a_ped:
                    a_ped, ped_mode = a_p, "YIELD: PEDESTRIAN"

        a_cmd = min(a_lead, a_light, a_ped)
        stage = max(info_lead["stage"], self.long_ped.last["stage"] if ped_mode else 0)

        # ---------------- mode label (highest priority first)
        prev = self.mode
        if stage >= 2:
            mode = "EMERGENCY BRAKE" if stage == 3 else "AEB: PARTIAL BRAKE"
        elif ped_mode and a_ped <= a_cmd + 1e-6:
            mode = ped_mode
        elif light_mode and a_light <= a_cmd + 1e-6:
            mode = light_mode
        elif self.lc is not None:
            mode = "LANE CHANGE: ABORTING" if self.lc.get("abort") else \
                   ("LANE CHANGE: LEFT" if self.lc["target"] > self.lc["origin"] else "LANE CHANGE: RIGHT")
        elif lead is not None and lead.gap < max(60.0, 3.0 * v) and a_lead < 0.5:
            mode = "FOLLOW"
        elif v < 0.3 and a_cmd <= 0:
            mode = "STOPPED"
        else:
            mode = "CRUISE"
        if stage == 1 and mode in ("FOLLOW", "CRUISE"):
            mode = "COLLISION WARNING"
        if mode != prev and mode in ("EMERGENCY BRAKE", "AEB: PARTIAL BRAKE", "YIELD: PEDESTRIAN",
                                     "STOP: RED LIGHT", "COLLISION WARNING"):
            self.log(t, f"{mode.title()}")
        self.mode = mode
        return a_cmd, dict(mode=mode, risk=max(info_lead["risk"], self.long_ped.last["risk"] if ped_mode else 0.0),
                           ttc=info_lead["ttc"], mttc=info_lead["mttc"], stage=stage, v0=v0,
                           leader=lead, lane=self.lane, lc=self.lc)
