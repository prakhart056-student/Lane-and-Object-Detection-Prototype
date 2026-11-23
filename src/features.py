"""Feature engineering and future-risk labelling.

Task definition (risk anticipation)
-----------------------------------
At time t the ego (follower) vehicle is *not yet* in a critical state.
Predict whether a critical rear-end conflict will occur within the next
HORIZON_S seconds, where "critical" means

    TTC < TTC_CRITICAL (3 s)   or   DRAC > DRAC_CRITICAL (3.35 m/s^2).

Only information observable at time t (and up to HISTORY_S in the past) is
used, so the model can run online on a vehicle.

Two feature sets are compared:
  * ONBOARD  - quantities measurable by the ego vehicle's own sensors
               (radar/lidar/camera + odometry).
  * CONTEXT  - ONBOARD + traffic context that could arrive through V2X /
               roadside infrastructure (lane speed, density, vehicle classes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

W = int(round(C.HISTORY_S / C.DT)) + 1          # history window incl. current (11)
H = int(round(C.HORIZON_S / C.DT))              # prediction horizon in frames (30)

ONBOARD = ["v", "a", "v_l", "a_l", "dv", "gap", "thw", "inv_ttc", "drac",
           "inv_mttc", "d_dv_1s", "d_gap_1s", "a_l_min_1s", "a_l_mean_1s",
           "v_l_std_1s", "a_f_mean_1s", "jerk_f"]
CONTEXT = ONBOARD + ["lane_speed", "lane_density", "lane_id", "v_class",
                     "class_l", "len_l"]

FEATURE_LABELS = {
    "v": "Ego speed", "a": "Ego acceleration", "v_l": "Leader speed",
    "a_l": "Leader acceleration", "dv": "Closing speed", "gap": "Gap",
    "thw": "Time headway", "inv_ttc": "Inverse TTC", "drac": "DRAC",
    "inv_mttc": "Inverse MTTC", "d_dv_1s": "Closing-speed change (1 s)",
    "d_gap_1s": "Gap change (1 s)", "a_l_min_1s": "Leader min accel (1 s)",
    "a_l_mean_1s": "Leader mean accel (1 s)", "v_l_std_1s": "Leader speed std (1 s)",
    "a_f_mean_1s": "Ego mean accel (1 s)", "jerk_f": "Ego jerk",
    "lane_speed": "Lane mean speed", "lane_density": "Lane density",
    "lane_id": "Lane index", "v_class": "Ego vehicle class",
    "class_l": "Leader vehicle class", "len_l": "Leader length",
}


def onboard_from_window(win: dict[str, np.ndarray]) -> np.ndarray:
    """Compute ONBOARD features from history windows.

    `win` maps v, a, v_l, a_l, gap -> arrays of shape (N, W) ordered oldest ->
    newest. Used by the online controller; mirrors `add_history_features`.
    """
    v, a, vl, al, gap = (win[k] for k in ("v", "a", "v_l", "a_l", "gap"))
    dv = v - vl
    g = np.maximum(gap[:, -1], 0.1)
    vc, vlc, ac, alc, dvc = v[:, -1], vl[:, -1], a[:, -1], al[:, -1], dv[:, -1]

    from .risk_metrics import modified_ttc
    feats = {
        "v": vc, "a": ac, "v_l": vlc, "a_l": alc, "dv": dvc, "gap": g,
        "thw": np.clip(np.where(vc > 0.1, g / np.maximum(vc, 0.1), 10.0), 0, 10.0),
        "inv_ttc": np.maximum(dvc, 0) / g,
        "drac": np.where(dvc > 0, dvc ** 2 / (2 * g), 0.0),
        "inv_mttc": 1.0 / np.maximum(modified_ttc(g, vc, vlc, ac, alc), 0.05),
        "d_dv_1s": dvc - dv[:, 0],
        "d_gap_1s": gap[:, -1] - gap[:, 0],
        "a_l_min_1s": al.min(axis=1),
        "a_l_mean_1s": al.mean(axis=1),
        "v_l_std_1s": vl.std(axis=1),
        "a_f_mean_1s": a.mean(axis=1),
        "jerk_f": (a[:, -1] - a[:, -2]) / C.DT,
    }
    return np.column_stack([feats[k] for k in ONBOARD])


def add_history_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["episode", "frame_id"]).reset_index(drop=True)
    g = p.groupby("episode", sort=False)

    p["inv_mttc"] = 1.0 / np.maximum(p["mttc"], 0.05)
    p["d_dv_1s"] = p["dv"] - g["dv"].shift(W - 1)
    p["d_gap_1s"] = p["gap"] - g["gap"].shift(W - 1)
    def roll(col: str):
        return g[col].rolling(W, min_periods=W)

    def flat(s: pd.Series) -> pd.Series:
        return s.reset_index(level=0, drop=True)

    p["a_l_min_1s"] = flat(roll("a_l").min())
    p["a_l_mean_1s"] = flat(roll("a_l").mean())
    p["v_l_std_1s"] = flat(roll("v_l").std(ddof=0))
    p["a_f_mean_1s"] = flat(roll("a").mean())
    p["jerk_f"] = (p["a"] - g["a"].shift(1)) / C.DT
    return p


def add_future_labels(p: pd.DataFrame) -> pd.DataFrame:
    """Label = critical conflict in (t, t+HORIZON]; also time-to-event."""
    crit = ((p["ttc"] < C.TTC_CRITICAL) | (p["drac"] > C.DRAC_CRITICAL)).astype(float)
    p["critical_now"] = crit.astype(bool)
    g = crit.groupby(p["episode"], sort=False)

    # reversed rolling max over next H frames (excluding current)
    fut = np.full(len(p), np.nan)
    tte = np.full(len(p), np.nan)
    for _, idx in g.indices.items():
        c = crit.to_numpy()[idx]
        n = len(c)
        if n <= H:
            continue
        # next critical index for each position
        nxt = np.full(n, np.inf)
        last = np.inf
        for i in range(n - 1, -1, -1):
            nxt[i] = last
            if c[i] > 0:
                last = i
        steps = nxt - np.arange(n)
        valid = np.arange(n) + H < n            # full horizon observed
        f = np.where(valid, (steps <= H).astype(float), np.nan)
        fut[idx] = f
        tte[idx] = np.where(steps <= H, steps * C.DT, np.nan)
    p["label"] = fut
    p["time_to_event"] = tte
    return p


def build_dataset(pairs: pd.DataFrame, subsample: int = 5) -> pd.DataFrame:
    p = add_history_features(pairs)
    p = add_future_labels(p)
    p = p.dropna(subset=ONBOARD + ["label"])
    p = p[~p["critical_now"]]                     # anticipation task only
    p = p[p["v"] > 1.0]                           # exclude standstill queues
    p = p[p["frame_id"] % subsample == 0]         # reduce autocorrelation
    p["label"] = p["label"].astype(int)
    return p.reset_index(drop=True)
