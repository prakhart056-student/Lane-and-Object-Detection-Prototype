"""Surrogate Safety Measures (SSMs) and a composite risk index.

All functions are vectorised over NumPy arrays and use SI units:
gap [m], speeds [m/s], accelerations [m/s^2].

Conventions (follower F behind leader L in the same lane):
    gap  = bumper-to-bumper distance (leader rear - follower front)
    dv   = v_F - v_L   (closing speed, positive when F is approaching L)
"""
from __future__ import annotations

import numpy as np

from . import config as C


def time_to_collision(gap, v_f, v_l, cap: float = C.TTC_CAP):
    """Constant-velocity TTC (Hayward, 1972). Returns `cap` when not closing."""
    gap, dv = np.asarray(gap, float), np.asarray(v_f, float) - np.asarray(v_l, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(dv > 1e-3, gap / dv, np.inf)
    return np.clip(ttc, 0.0, cap)


def inverse_ttc(gap, v_f, v_l):
    """1/TTC [1/s]; continuous and bounded, preferred as an ML feature."""
    gap = np.maximum(np.asarray(gap, float), 0.1)
    return np.maximum(np.asarray(v_f, float) - np.asarray(v_l, float), 0.0) / gap


def time_headway(gap, v_f, cap: float = 10.0):
    """Time headway THW = gap / v_F [s]."""
    v_f = np.asarray(v_f, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        thw = np.where(v_f > 0.1, np.asarray(gap, float) / v_f, cap)
    return np.clip(thw, 0.0, cap)


def drac(gap, v_f, v_l):
    """Deceleration Rate to Avoid Crash (Cooper & Ferguson, 1976) [m/s^2]."""
    gap = np.maximum(np.asarray(gap, float), 0.1)
    dv = np.asarray(v_f, float) - np.asarray(v_l, float)
    return np.where(dv > 0, dv ** 2 / (2.0 * gap), 0.0)


def modified_ttc(gap, v_f, v_l, a_f, a_l, cap: float = C.TTC_CAP):
    """Modified TTC (Ozbay et al., 2008) accounting for relative acceleration.

    Solves  gap - dv*t - 0.5*da*t^2 = 0  for the smallest positive t, where
    dv = v_F - v_L and da = a_F - a_L.  Falls back to TTC when da ~ 0.
    """
    gap = np.asarray(gap, float)
    dv = np.asarray(v_f, float) - np.asarray(v_l, float)
    da = np.asarray(a_f, float) - np.asarray(a_l, float)

    out = np.full(gap.shape, np.inf)
    lin = np.abs(da) < 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        out[lin] = np.where(dv[lin] > 1e-3, gap[lin] / dv[lin], np.inf)

        q = ~lin
        disc = dv[q] ** 2 + 2.0 * da[q] * gap[q]
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        r1 = (-dv[q] + sq) / da[q]
        r2 = (-dv[q] - sq) / da[q]
        roots = np.stack([r1, r2])
        roots = np.where(roots > 0, roots, np.inf)
        out[q] = np.nanmin(np.where(np.isnan(roots), np.inf, roots), axis=0)
    return np.clip(out, 0.0, cap)


def crash_potential_index(drac_values, madr_mean: float = 8.45, madr_std: float = 1.4):
    """Per-sample Crash Potential Index (Cunto & Saccomanno, 2008).

    Probability that the deceleration required (DRAC) exceeds the vehicle's
    maximum available deceleration (MADR ~ N(8.45, 1.4) m/s^2).
    """
    from scipy.stats import norm
    return norm.sf(madr_mean, loc=np.asarray(drac_values, float), scale=madr_std)


def composite_risk_index(ttc, thw, drac_values):
    """Composite Risk Index (CRI) in [0, 1].

    Each SSM is mapped to [0,1] with a smooth saturating transform that equals
    0.5 at its critical threshold, then combined with weights reflecting
    their predictive strength for rear-end conflicts.
    """
    r_ttc = 1.0 / (1.0 + (np.asarray(ttc) / C.TTC_CRITICAL) ** 3)
    r_thw = 1.0 / (1.0 + (np.asarray(thw) / C.THW_CRITICAL) ** 3)
    r_drac = 1.0 - np.exp(-np.log(2) * np.asarray(drac_values) / C.DRAC_CRITICAL)
    return np.clip(0.5 * r_ttc + 0.2 * r_thw + 0.3 * r_drac, 0.0, 1.0)


RISK_LEVELS = ["Safe", "Caution", "Danger", "Critical"]


def risk_level(cri):
    """Discretise CRI into four driver-facing levels."""
    return np.digitize(np.asarray(cri), [0.25, 0.5, 0.75])


def time_exposed_ttc(ttc_series, dt: float = C.DT, thr: float = C.TTC_CRITICAL):
    """Time Exposed TTC (TET, Minderhoud & Bovy 2001): seconds spent below thr."""
    return float(np.sum(np.asarray(ttc_series) < thr) * dt)


def time_integrated_ttc(ttc_series, dt: float = C.DT, thr: float = C.TTC_CRITICAL):
    """Time Integrated TTC (TIT): integral of (thr - TTC) while TTC < thr [s^2]."""
    t = np.asarray(ttc_series)
    return float(np.sum(np.where(t < thr, thr - t, 0.0)) * dt)
