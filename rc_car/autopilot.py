"""Hardware-independent decision making and control for the RC car.

Inputs each tick: lane measurement (camera), range to the obstacle ahead (sensor),
stop-line distance and traffic-light state (camera). Output: steering angle,
target speed and a brake flag. Everything here is plain Python, so the exact
same code runs in the simulator, in the unit tests and on the real car.

States (priority high -> low)
  ESTOP, FAULT (stale sensors), AEB_FULL, AEB_PARTIAL, STOP_RED, STOP_SIGN,
  LANE_LOST, WARNING, FOLLOW, CRUISE, IDLE
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.estimation import Debounce, RangeEstimator
from src.risk_metrics import modified_ttc, time_to_collision


@dataclass
class Command:
    steer: float          # rad, + = left
    speed: float          # m/s target
    brake: bool
    state: str
    info: dict = field(default_factory=dict)


class Autopilot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.range_est = RangeEstimator(window_s=0.5)
        self.aeb = Debounce(up=2, down=6)
        self.enabled = False
        self.estopped = False
        self.v_set = 0.0
        self.v_est = 0.0
        self.steer = 0.0
        self.last_t = None
        self.last_lane_t = -1e9
        self.last_range_t = None
        self.stop_since = None
        self.ignore_stop_until = -1e9
        self.state = "IDLE"
        self.light_mem = None
        self.events: list[tuple[float, str]] = []

    # ------------------------------------------------------------------ control
    def start(self):
        self.enabled, self.estopped = True, False

    def pause(self):
        self.enabled = False

    def estop(self):
        self.enabled, self.estopped = False, True
        self.v_set = 0.0

    def _log(self, t, msg):
        if not self.events or self.events[-1][1] != msg:
            self.events.append((round(t, 2), msg))
            self.events = self.events[-50:]

    def _idm(self, v, gap, dv, v0, s0=None):
        c = self.cfg
        free = 1 - (v / max(v0, 0.05)) ** 4
        if gap is None:
            return c.a_accel * free
        s_star = (c.s0 if s0 is None else s0) + max(0.0, v * c.T + v * dv / (2 * np.sqrt(c.a_accel * c.a_brake)))
        return c.a_accel * (free - (s_star / max(gap, 0.01)) ** 2)

    def _stop(self, t, state, brake=False, info=None):
        self.v_set = 0.0
        if state != self.state:
            self._log(t, state)
        self.state = state
        return Command(self.steer * 0.5, 0.0, brake, state, info or {})

    # ------------------------------------------------------------------ main step
    def update(self, t, lane, range_m, stop_line_m=None, light=None, range_t=None, v_meas=None):
        c = self.cfg
        dt = 1.0 / c.rate_hz if self.last_t is None else float(np.clip(t - self.last_t, 0.0, 0.2))
        self.last_t = t
        # speed estimate: encoder if available, else first-order model of the ESC
        self.v_est = v_meas if v_meas is not None else self.v_est + (self.v_set - self.v_est) * min(dt / 0.35, 1.0)
        v = self.v_est
        if range_m is not None:
            self.last_range_t = t if range_t is None else range_t
        if lane is not None and lane.confidence > 0.15:
            self.last_lane_t = t
        if light is not None:                               # remember the light through short occlusions
            self.light_mem = (light, t)
        elif self.light_mem is not None and t - self.light_mem[1] <= 1.5:
            light = self.light_mem[0]

        if self.estopped:
            return self._stop(t, "ESTOP", brake=True)
        if not self.enabled:
            return self._stop(t, "IDLE")
        if self.last_range_t is None or t - self.last_range_t > c.watchdog_s:
            return self._stop(t, "FAULT: range sensor", brake=True)

        # ---------------- lateral: Stanley on the camera lane estimate
        lane_ok = t - self.last_lane_t <= c.lane_lost_s
        if lane is not None and lane.confidence > 0.15:
            if c.steer_law == "stanley":
                delta = (lane.heading + np.arctan2(c.k_stanley * lane.center_at(0.0), v + c.k_soft)
                         + np.arctan(c.wheelbase * lane.curvature))
            else:                                   # pure pursuit to a point on the detected lane centre
                f_t = c.lookahead_m
                x_t, y_t = f_t + c.rear_axle_to_bumper, lane.center_at(f_t)
                alpha = np.arctan2(y_t, x_t)
                delta = np.arctan2(2 * c.wheelbase * np.sin(alpha), np.hypot(x_t, y_t))
            delta = float(np.clip(delta, -np.radians(c.max_steer_deg), np.radians(c.max_steer_deg)))
            self.steer = c.steer_smooth * self.steer + (1 - c.steer_smooth) * delta
        else:
            self.steer *= 0.9
        # corner speed from the steering angle: lateral acceleration = v² tan(delta) / L
        k_path = abs(np.tan(self.steer)) / c.wheelbase
        v0 = min(c.v_cruise, np.sqrt(c.a_lat_max / max(k_path, 1e-3)))
        if not lane_ok:
            a_lane = -c.a_brake
        else:
            a_lane = np.inf

        # ---------------- obstacle ahead: following + risk-staged AEB
        info = dict(v=v, v0=v0, ttc=np.inf, mttc=np.inf, gap=None, closing=0.0, stage=0)
        a_follow, a_aeb, stage = self._idm(v, None, 0.0, v0), np.inf, 0
        if range_m is not None and c.range_min <= range_m < c.range_max:
            d, rr, racc = self.range_est.update(t, range_m)
            gap, closing = max(d, 0.01), max(-rr, 0.0)
            v_l = max(v - closing, 0.0)
            ttc = float(time_to_collision(np.array([gap]), np.array([v]), np.array([v_l]))[0])
            mttc = float(modified_ttc(np.array([gap]), np.array([v]), np.array([v_l]),
                                      np.array([0.0]), np.array([-racc]))[0])
            t_min = min(ttc, mttc)
            raw = 3 if t_min < c.ttc_full else 2 if t_min < c.ttc_partial else 1 if t_min < c.ttc_warn else 0
            drac = closing ** 2 / (2 * gap)
            if raw >= 2 and drac < c.drac_min:
                raw = 1
            if gap < c.s0 * 0.6 and v > 0.05:          # inside the standstill margin: always stop
                raw = 3
            stage = self.aeb.update(raw)
            a_follow = self._idm(v, gap, v - v_l, v0)
            a_aeb = -c.a_emergency if stage == 3 else -c.a_brake if stage == 2 else np.inf
            info.update(ttc=ttc, mttc=mttc, gap=gap, closing=closing, stage=stage, drac=drac)
        else:
            self.range_est.reset()
            self.aeb.reset()

        # ---------------- stop line (stop, wait, then continue) and red light
        a_stop, stop_state = np.inf, None
        if stop_line_m is not None and t >= self.ignore_stop_until:
            dist = stop_line_m - c.stop_margin
            if light == "green":
                pass                                       # green light: drive over the line
            else:
                a_stop = self._idm(v, max(dist, 0.01), v, v0, s0=0.03)   # stop right at the line
                stop_state = "STOP_RED" if light == "red" else "STOP_SIGN"
                if v < 0.05 and dist < 0.15:
                    if light != "red":
                        self.stop_since = t if self.stop_since is None else self.stop_since
                        if t - self.stop_since >= c.stop_wait_s:
                            self.ignore_stop_until, self.stop_since = t + 2.5, None
                            self._log(t, "stop complete: continuing")
                else:
                    self.stop_since = None
        elif light == "red":
            a_stop, stop_state = -c.a_brake * 0.5, "STOP_RED"

        a_cmd = min(a_follow, a_aeb, a_stop, a_lane)
        brake = stage == 3
        if brake:
            self.v_set = 0.0
        else:
            self.v_set = float(np.clip(self.v_set + a_cmd * dt, 0.0, c.v_max))
            if a_cmd < -c.a_brake * 0.9 and v > 0.1:
                brake = True

        if stage == 3:
            state = "AEB_FULL"
        elif stage == 2:
            state = "AEB_PARTIAL"
        elif stop_state and a_stop <= a_cmd + 1e-9:
            state = stop_state
        elif not lane_ok:
            state = "LANE_LOST"
        elif stage == 1:
            state = "WARNING"
        elif info["gap"] is not None and a_follow < 0.3 * c.a_accel and info["gap"] < 1.5:
            state = "FOLLOW"
        else:
            state = "CRUISE"
        if state != self.state:
            self._log(t, state)
        self.state = state
        info.update(a_cmd=float(a_cmd) if np.isfinite(a_cmd) else 0.0)
        return Command(float(self.steer), float(self.v_set), bool(brake), state, info)
