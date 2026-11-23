"""Batch evaluation of the Level-2 stack on randomised (unseen) scenario variants.

    python -m src.av.evaluate --runs 30

Compares the risk-aware planner (ML risk model + MTTC-staged AEB) with a reactive
baseline (same planner, plain TTC-triggered AEB) on every scenario family, and
writes reports/av_results.json and reports/figures/fig13_av_evaluation.png.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from .. import config as C
from ..collision_avoidance import RiskModel
from .scenarios import SCENARIOS

CONFIGS = {"Risk-aware (ours)": dict(risk_aware=True), "Reactive baseline": dict(risk_aware=False)}


def run(n_runs=30, seed=7, families=None, verbose=True):
    warnings.filterwarnings("ignore")
    rm = RiskModel.load()
    rows = []
    t0 = time.time()
    for fam in families or SCENARIOS:
        for i in range(n_runs):
            for cname, ckw in CONFIGS.items():
                rng = np.random.default_rng(seed * 100003 + i)        # same variant for both configs
                w = SCENARIOS[fam](rng, randomize=True, risk_model=rm, **ckw)
                m = w.run()
                m.update(family=fam, config=cname, run=i)
                rows.append(m)
        if verbose:
            d = pd.DataFrame([r for r in rows if r["family"] == fam])
            print(f"  {fam:20s} " + "  |  ".join(
                f"{c}: coll {d[d.config == c].collision.mean() * 100:4.0f}%" for c in CONFIGS)
                + f"   ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    agg = {}
    for (fam, cname), g in df.groupby(["family", "config"]):
        agg.setdefault(fam, {})[cname] = dict(
            runs=int(len(g)), collision_rate=float(g.collision.mean()),
            red_light_violations=int(g.red_light_violations.sum()),
            lane_changes_per_run=float(g.lane_changes.mean()), aborts=int(g.lc_aborts.sum()),
            rms_lateral_error_m=float(g.rms_lateral_error_m.mean()),
            max_lateral_error_m=float(g.max_lateral_error_m.max()),
            max_lat_accel=float(g.max_lat_accel.max()), median_min_ttc_s=float(g.min_ttc_s.median()),
            mean_speed_kmh=float(g.mean_speed_kmh.mean()),
            min_ped_clearance_m=float(g.min_ped_clearance_m.min()) if g.min_ped_clearance_m.notna().any() else None)
    out = dict(n_runs=n_runs, seed=seed, results=agg, runtime_s=time.time() - t0)
    with open(C.REPORTS / "av_results.json", "w") as f:
        json.dump(out, f, indent=2)
    df.to_csv(C.REPORTS / "av_runs.csv", index=False)
    figure(agg)
    return out


def figure(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(agg)
    labels = {"highway_overtake": "Highway\novertake", "highway_cut_in": "Highway\ncut-in",
              "highway_emergency": "Highway\nsudden stop", "urban_route": "Urban\nlight + pedestrian",
              "rc_track": "RC track\n(1:10)"}
    cols = {"Risk-aware (ours)": "#2a78d6", "Reactive baseline": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
    x = np.arange(len(fams)); w = 0.36
    for k, cname in enumerate(cols):
        off = (k - 0.5) * (w + 0.03)
        coll = [agg[f][cname]["collision_rate"] * 100 for f in fams]
        b = axes[0].bar(x + off, coll, w, color=cols[cname], label=cname)
        for bb, v in zip(b, coll):
            axes[0].text(bb.get_x() + bb.get_width() / 2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8)
        ttc = [agg[f][cname]["median_min_ttc_s"] for f in fams]
        axes[1].bar(x + off, np.minimum(ttc, 10), w, color=cols[cname])
        lat = [agg[f][cname]["rms_lateral_error_m"] * 100 for f in fams]
        axes[2].bar(x + off, lat, w, color=cols[cname])
    titles = ["Collision rate [%]", "Median minimum TTC [s] (capped at 10)", "Lane-keeping RMS error [cm]"]
    for ax, ttl in zip(axes, titles):
        ax.set_xticks(x); ax.set_xticklabels([labels.get(f, f) for f in fams], fontsize=8.5)
        ax.set_title(ttl, loc="left", fontsize=11, weight="bold")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e4e3de"); ax.set_axisbelow(True)
    axes[1].axhline(C.TTC_CRITICAL, color="#52514e", ls="--", lw=1)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig13_av_evaluation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run(a.runs, a.seed)
