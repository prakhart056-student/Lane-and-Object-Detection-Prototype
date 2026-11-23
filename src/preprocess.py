"""Clean NGSIM trajectories and build leader-follower interaction pairs.

Steps
-----
1. Remove duplicate (vehicle, frame) records and convert feet -> metres.
2. Re-derive speed and acceleration from Savitzky-Golay smoothed positions
   (raw NGSIM speed/acceleration are known to contain measurement noise,
   see Punzo et al., 2011; Montanino & Punzo, 2015).
3. Join every follower with its leader at the same frame (NGSIM `preceding`)
   and compute bumper-to-bumper gap.
4. Compute the surrogate safety measures for every pair-frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from . import config as C
from . import risk_metrics as R


def clean_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(["vehicle_id", "frame_id"]).copy()
    df = df.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)

    for col in ["local_x", "local_y", "v_length", "v_width"]:
        df[col] = df[col] * C.FT_TO_M
    df["v_vel_raw"] = df["v_vel"] * C.FT_TO_M
    df["v_acc_raw"] = df["v_acc"] * C.FT_TO_M

    # Split each vehicle into contiguous segments (no missing frames)
    seg_break = df.groupby("vehicle_id")["frame_id"].diff().fillna(1).ne(1)
    df["segment"] = seg_break.groupby(df["vehicle_id"]).cumsum()

    y_s = np.empty(len(df))
    v_s = np.empty(len(df))
    a_s = np.empty(len(df))
    for _, idx in df.groupby(["vehicle_id", "segment"]).indices.items():
        y = df["local_y"].to_numpy()[idx]
        if len(idx) >= C.SMOOTH_WINDOW:
            y_s[idx] = savgol_filter(y, C.SMOOTH_WINDOW, C.SMOOTH_POLY)
            v_s[idx] = savgol_filter(y, C.SMOOTH_WINDOW, C.SMOOTH_POLY, deriv=1, delta=C.DT)
            a_s[idx] = savgol_filter(y, C.SMOOTH_WINDOW, C.SMOOTH_POLY, deriv=2, delta=C.DT)
        else:
            y_s[idx] = y
            v_s[idx] = df["v_vel_raw"].to_numpy()[idx]
            a_s[idx] = df["v_acc_raw"].to_numpy()[idx]

    # Drop 0.5 s at both ends of every segment: polynomial-filter edge effects
    # produce unreliable derivatives there.
    pos = df.groupby(["vehicle_id", "segment"]).cumcount()
    rev = df.groupby(["vehicle_id", "segment"]).cumcount(ascending=False)
    edge = ((pos < C.EDGE_TRIM) | (rev < C.EDGE_TRIM)).to_numpy()

    df["y"] = y_s
    df["v"] = np.clip(v_s, 0.0, 45.0)
    df["a"] = np.clip(a_s, -9.0, 5.0)
    df["t"] = (df["global_time"] - df["global_time"].min()) / 1000.0
    keep = ["vehicle_id", "frame_id", "t", "lane_id", "local_x", "y", "v", "a",
            "v_vel_raw", "v_acc_raw", "v_length", "v_width", "v_class",
            "preceding", "following", "location"]
    return df.loc[~edge, keep].reset_index(drop=True)


def build_pairs(traj: pd.DataFrame) -> pd.DataFrame:
    """Attach leader state to each follower record and compute SSMs."""
    lead = traj[["vehicle_id", "frame_id", "y", "v", "a", "v_length", "v_class",
                 "lane_id"]].rename(columns={
        "vehicle_id": "preceding", "y": "y_l", "v": "v_l", "a": "a_l",
        "v_length": "len_l", "v_class": "class_l", "lane_id": "lane_l"})

    p = traj[traj["preceding"] > 0].merge(lead, on=["preceding", "frame_id"], how="inner")
    p = p[p["lane_id"] == p["lane_l"]]            # same lane only
    p = p[p["lane_id"].between(1, 5)]             # main-line lanes (drop ramps/aux)

    p["gap"] = p["y_l"] - p["len_l"] - p["y"]
    p = p[(p["gap"] > 0.5) & (p["gap"] < 150)].copy()

    p["dv"] = p["v"] - p["v_l"]
    p["ttc"] = R.time_to_collision(p["gap"], p["v"], p["v_l"])
    p["inv_ttc"] = R.inverse_ttc(p["gap"], p["v"], p["v_l"])
    p["thw"] = R.time_headway(p["gap"], p["v"])
    p["drac"] = R.drac(p["gap"], p["v"], p["v_l"])
    p["mttc"] = R.modified_ttc(p["gap"], p["v"], p["v_l"], p["a"], p["a_l"])
    p["cpi"] = R.crash_potential_index(p["drac"])
    p["cri"] = R.composite_risk_index(p["ttc"], p["thw"], p["drac"])
    p["risk_level"] = R.risk_level(p["cri"])

    # Traffic context (what a V2X / infrastructure feed could provide)
    lane_stats = traj.groupby(["frame_id", "lane_id"]).agg(
        lane_speed=("v", "mean"), lane_count=("v", "size")).reset_index()
    section_len_km = (traj["y"].max() - traj["y"].min()) / 1000.0
    lane_stats["lane_density"] = lane_stats["lane_count"] / section_len_km
    p = p.merge(lane_stats[["frame_id", "lane_id", "lane_speed", "lane_density"]],
                on=["frame_id", "lane_id"], how="left")

    # Identify pair episodes: contiguous frames with the same leader
    p = p.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)
    new_ep = (p.groupby("vehicle_id")["frame_id"].diff().ne(1)
              | p["preceding"].ne(p.groupby("vehicle_id")["preceding"].shift()))
    p["episode"] = new_ep.cumsum()
    return p


def run(site: str) -> pd.DataFrame:
    raw = pd.read_csv(C.DATA_RAW / f"ngsim_{site}.csv")
    traj = clean_trajectories(raw)
    pairs = build_pairs(traj)
    _save(pairs, f"pairs_{site}")
    _save(traj, f"traj_{site}")
    print(f"[{site}] vehicles={traj.vehicle_id.nunique():,} records={len(traj):,} "
          f"pairs={len(pairs):,} episodes={pairs.episode.nunique():,}")
    return pairs


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _save(df: pd.DataFrame, name: str) -> None:
    if _has_parquet():
        df.to_parquet(C.DATA_PROCESSED / f"{name}.parquet", index=False)
    else:
        df.to_pickle(C.DATA_PROCESSED / f"{name}.pkl")


def load(name: str) -> pd.DataFrame:
    pq = C.DATA_PROCESSED / f"{name}.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.read_pickle(C.DATA_PROCESSED / f"{name}.pkl")


if __name__ == "__main__":
    for s in C.SITES:
        run(s)
