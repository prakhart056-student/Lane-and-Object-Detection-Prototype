"""Path tracking (lateral control) and curvature-aware speed limits.

Pure Pursuit (Coulter, 1992) steers towards a look-ahead point on the reference
path; Stanley (Thrun et al., 2006) combines the heading error with a cross-track
term at the front axle. Both follow a reference defined in Frenet coordinates
as the road centre line shifted by a lateral offset d_ref(s), which is how lane
changes are expressed.
"""
from __future__ import annotations

import numpy as np

from .vehicle import wrap


def smoothstep5(u):
    """Quintic blend 0 -> 1 with zero slope and curvature at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class Reference:
    """Lateral reference d_ref(s): a lane centre, or a quintic lane-change blend."""

    def __init__(self, d_from, d_to=None, s0=0.0, length=1.0):
        self.d_from = d_from
        self.d_to = d_from if d_to is None else d_to
        self.s0, self.length = s0, max(length, 1e-3)

    def progress(self, s, path_length=None, closed=False):
        ds = s - self.s0
        if closed and path_length:
            ds = (ds + path_length / 2) % path_length - path_length / 2
        return float(np.clip(ds / self.length, 0.0, 1.0))

    def d(self, s, path_length=None, closed=False):
        ds = np.asarray(s, float) - self.s0
        if closed and path_length:
            ds = (ds + path_length / 2) % path_length - path_length / 2
        return self.d_from + (self.d_to - self.d_from) * smoothstep5(ds / self.length)

    @property
    def is_change(self):
        return abs(self.d_to - self.d_from) > 1e-6


def pure_pursuit(vehicle, path, ref: Reference, s_now, lookahead_gain=0.9, min_lookahead=6.0):
    """Steering angle and the look-ahead point (for display)."""
    Ld = max(min_lookahead, lookahead_gain * vehicle.v)
    st = s_now + Ld
    tx, ty, _ = path.frenet_to_xy(st, ref.d(st, path.length, path.closed))
    alpha = wrap(np.arctan2(ty - vehicle.y, tx - vehicle.x) - vehicle.yaw)
    dist = max(np.hypot(tx - vehicle.x, ty - vehicle.y), 1e-3)
    delta = np.arctan2(2.0 * vehicle.params.wheelbase * np.sin(alpha), dist)
    return float(delta), (float(tx), float(ty))


def stanley(heading_error, cross_track, v, k=1.2, k_soft=1.0, kappa=0.0, wheelbase=2.8):
    """Stanley law. heading_error: path heading minus vehicle heading (rad);
    cross_track: lateral offset of the path from the front axle (left positive)."""
    return float(heading_error + np.arctan2(k * cross_track, v + k_soft) + np.arctan(wheelbase * kappa))


def curve_speed_limit(path, s_now, v, a_lat_max=2.5, b_comf=2.0, horizon_s=6.0, v_cap=40.0):
    """Highest speed from which every curve in the preview can be taken at a_lat_max,
    braking comfortably at b_comf before it."""
    preview = max(30.0, v * horizon_s)
    ss = np.arange(0.0, preview, 2.0)
    k = np.abs(path.at(s_now + ss)[3])
    v_curve = np.sqrt(a_lat_max / np.maximum(k, 1e-4))
    v_allow = np.sqrt(np.minimum(v_curve, v_cap) ** 2 + 2 * b_comf * ss)
    return float(min(v_cap, v_allow.min()))
