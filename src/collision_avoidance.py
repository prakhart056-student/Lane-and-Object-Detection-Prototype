"""Risk-aware longitudinal collision avoidance: controllers + closed-loop simulation.

Controllers compared
--------------------
1. ACC (IDM)          - Intelligent Driver Model adaptive cruise control only.
2. Reactive AEB       - IDM + staged Autonomous Emergency Braking triggered by
                        constant-velocity TTC (Euro NCAP-style warning /
                        partial / full braking cascade).
3. Risk-Aware (ours)  - IDM whose desired time-gap grows with the ML-predicted
                        conflict probability p(t), a pre-emptive graded
                        deceleration when p(t) exceeds the model threshold, and
                        a staged AEB driven by min(TTC, MTTC) so that leader
                        deceleration is anticipated.
4. Human (replay)     - the real NGSIM driver behind the same leader
                        (real-world scenarios only).

The simulator is fully vectorised: N scenarios advance in lock-step, so the
risk model is queried once per time step for all scenarios.

Realism: 0.2 s actuation latency, physical acceleration limits, Gaussian
sensor noise on gap / leader speed / leader acceleration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import ONBOARD, W, onboard_from_window
from .preprocess import load
from .risk_metrics import modified_ttc, time_exposed_ttc, time_integrated_ttc, time_to_collision

CONTROLLERS = ["ACC (IDM)", "Reactive AEB", "Risk-Aware (ours)"]
SENSOR_NOISE = dict(gap=0.3, v_l=0.2, a_l=0.3)


# ---------------------------------------------------------------------------
# Control laws
# ---------------------------------------------------------------------------
def idm_accel(v, gap, dv, T=None, v0=None):
    p = C.IDM
    T = p["T"] if T is None else T
    v0 = p["v0"] if v0 is None else v0
    s_star = p["s0"] + np.maximum(0.0, v * T + v * dv / (2 * np.sqrt(p["a_max"] * p["b_comf"])))
    a = p["a_max"] * (1 - (v / v0) ** p["delta"] - (s_star / np.maximum(gap, 0.1)) ** 2)
    return np.maximum(a, C.ACC_A_MIN)          # ACC has limited braking authority


def staged_aeb(ttc):
    a = C.AEB
    out = np.full(ttc.shape, np.inf)
    out = np.where(ttc < a["ttc_partial"], a["a_partial"], out)
    out = np.where(ttc < a["ttc_full"], a["a_full"], out)
    return out


@dataclass
class RiskModel:
    model: object
    threshold: float

    @classmethod
    def load(cls):
        b = joblib.load(C.MODELS / "risk_model.joblib")
        return cls(b["model"], b["threshold"])

    def __call__(self, win):
        X = pd.DataFrame(onboard_from_window(win), columns=ONBOARD)
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Vectorised closed-loop simulator
# ---------------------------------------------------------------------------
def simulate(lead_x, lead_v, lead_a, ego_v0, gap0, controller, risk_model=None,
             valid=None, seed=0):
    """Simulate N ego vehicles following N scripted leaders.

    lead_x : (N, T) leader rear-bumper position [m]  (ego front starts at 0 - gap0)
    lead_v, lead_a : (N, T)
    Returns dict of (N, T) arrays plus per-scenario collision info.
    """
    rng = np.random.default_rng(seed)
    N, T = lead_v.shape
    valid = np.ones((N, T), bool) if valid is None else valid
    delay = int(round(C.REACTION_DELAY_S / C.DT))

    x = lead_x[:, 0] - gap0
    v = ego_v0.astype(float).copy()
    a = np.zeros(N)
    cmd_queue = [np.zeros(N) for _ in range(delay)]

    rec = {k: np.full((N, T), np.nan) for k in ("gap", "v", "a", "ttc", "risk", "a_cmd")}
    risk_f = np.zeros(N)                   # filtered risk estimate
    collided = np.zeros(N, bool)
    impact_speed = np.zeros(N)
    t_collision = np.full(N, np.nan)

    # history windows of *perceived* quantities (warm-started)
    def meas(t):
        g = lead_x[:, t] - x + rng.normal(0, SENSOR_NOISE["gap"], N)
        vl = lead_v[:, t] + rng.normal(0, SENSOR_NOISE["v_l"], N)
        al = lead_a[:, t] + rng.normal(0, SENSOR_NOISE["a_l"], N)
        return np.maximum(g, 0.1), np.maximum(vl, 0), al

    g0, vl0, al0 = meas(0)
    hist = {"v": np.repeat(v[:, None], W, 1), "a": np.zeros((N, W)),
            "v_l": np.repeat(vl0[:, None], W, 1), "a_l": np.repeat(al0[:, None], W, 1),
            "gap": np.repeat(g0[:, None], W, 1)}

    for t in range(T):
        active = valid[:, t] & ~collided
        true_gap = lead_x[:, t] - x
        g_m, vl_m, al_m = meas(t)
        for k, val in (("v", v), ("a", a), ("v_l", vl_m), ("a_l", al_m), ("gap", g_m)):
            hist[k] = np.roll(hist[k], -1, axis=1)
            hist[k][:, -1] = val

        dv_m = v - vl_m
        ttc_m = time_to_collision(g_m, v, vl_m)
        risk = np.zeros(N)

        if controller == "ACC (IDM)":
            a_cmd = idm_accel(v, g_m, dv_m)
        elif controller == "Reactive AEB":
            a_cmd = np.minimum(idm_accel(v, g_m, dv_m), staged_aeb(ttc_m))
        elif controller == "Risk-Aware (ours)":
            r_now = risk_model(hist)
            # fast-attack / slow-release filter: react to rising risk instantly,
            # decay smoothly to avoid on/off brake chatter from sensor noise
            r_now = np.where(v > 1.0, r_now, 0.0)   # model is trained for moving traffic only
            risk_f = np.where(r_now > risk_f, r_now, 0.8 * risk_f + 0.2 * r_now)
            risk = risk_f
            thr = risk_model.threshold
            T_eff = C.IDM["T"] * (1.0 + 0.8 * risk)
            a_idm = idm_accel(v, g_m, dv_m, T=T_eff)
            excess = np.clip((risk - thr) / (1 - thr), 0, 1)
            closing = ((dv_m > 0) | (al_m < -1.0)) & (v > 0.5)
            a_pre = np.where((risk >= thr) & closing,
                             -1.5 + (C.RISK_A_MIN + 1.5) * excess, np.inf)
            mttc_m = modified_ttc(g_m, v, vl_m, a, al_m)
            a_cmd = np.minimum.reduce([a_idm, a_pre, staged_aeb(np.minimum(ttc_m, mttc_m))])
        else:
            raise ValueError(controller)

        a_cmd = np.clip(a_cmd, C.VEHICLE_A_MIN, C.VEHICLE_A_MAX)
        cmd_queue.append(a_cmd)
        a_new = cmd_queue.pop(0) if delay else a_cmd
        a = np.where(active, a_new, a)
        a = np.where((v <= 1e-3) & (a < 0), 0.0, a)      # a stopped car holds, it cannot brake further

        # record
        rec["gap"][active, t] = true_gap[active]
        rec["v"][active, t] = v[active]
        rec["a"][active, t] = a[active]
        rec["a_cmd"][active, t] = a_cmd[active]
        rec["ttc"][active, t] = time_to_collision(true_gap, v, lead_v[:, t])[active]
        rec["risk"][active, t] = risk[active]

        # integrate (exact for constant accel, no reversing)
        v_next = np.maximum(v + a * C.DT, 0.0)
        dt_eff = np.where((v + a * C.DT) < 0, v / np.maximum(-a, 1e-6), C.DT)
        x = np.where(active, x + v * dt_eff + 0.5 * a * dt_eff ** 2, x)
        v = np.where(active, v_next, v)

        if t + 1 < T:
            hit = active & (lead_x[:, t + 1] - x <= 0)
            impact_speed[hit] = np.maximum(v[hit] - lead_v[hit, t + 1], 0)
            t_collision[hit] = (t + 1) * C.DT
            collided |= hit

    rec.update(collided=collided, impact_speed=impact_speed, t_collision=t_collision)
    return rec


def summarise(rec, valid=None):
    N = rec["v"].shape[0]
    rows = []
    for i in range(N):
        m = ~np.isnan(rec["v"][i])
        ttc, a = rec["ttc"][i, m], rec["a"][i, m]
        jerk = np.diff(a) / C.DT if len(a) > 1 else np.array([0.0])
        rows.append({
            "collision": bool(rec["collided"][i]),
            "impact_speed": float(rec["impact_speed"][i]),
            "min_ttc": float(np.min(ttc)) if len(ttc) else np.nan,
            "min_gap": float(np.nanmin(rec["gap"][i])),
            "TET": time_exposed_ttc(ttc), "TIT": time_integrated_ttc(ttc),
            "max_decel": float(-np.min(a)), "rms_jerk": float(np.sqrt(np.mean(jerk ** 2))),
            "mean_speed": float(np.mean(rec["v"][i, m])),
            "harsh_brake_s": float(np.sum(a < -4.0) * C.DT),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------
def _from_speed(v, dt=C.DT, x0=None):
    x0 = np.zeros(v.shape[0]) if x0 is None else x0
    x = x0[:, None] + np.concatenate([np.zeros((v.shape[0], 1)),
                                      np.cumsum(v[:, :-1] * dt, axis=1)], axis=1)
    a = np.gradient(v, dt, axis=1)
    return x, a


def _required_decel(v0, stop_dist):
    """Constant deceleration needed to stop within `stop_dist` after actuation latency."""
    room = np.maximum(stop_dist - v0 * C.REACTION_DELAY_S, 0.1)
    return v0 ** 2 / (2 * room)


def scenario_emergency_brake(n, rng, T=250):
    v0 = rng.uniform(15, 30, n)
    thw = rng.uniform(0.8, 2.0, n)
    decel = rng.uniform(4.0, 8.0, n)
    t_brake = rng.uniform(2.0, 4.0, n)
    t = np.arange(T) * C.DT
    vl = np.maximum(v0[:, None] - decel[:, None] * np.maximum(t[None] - t_brake[:, None], 0), 0)
    xl, al = _from_speed(vl, x0=thw * v0)
    return dict(lead_x=xl, lead_v=vl, lead_a=al, ego_v0=v0, gap0=thw * v0,
                params=pd.DataFrame({"v0": v0, "thw0": thw, "lead_decel": decel,
                                     "required_decel": _required_decel(v0, thw * v0 + v0 ** 2 / (2 * decel))}))


def scenario_cut_in(n, rng, T=250):
    v0 = rng.uniform(15, 30, n)
    gap0 = rng.uniform(5.0, 20.0, n)
    vl0 = v0 - rng.uniform(0, 8, n)
    dec = rng.uniform(0, 3.0, n)
    t = np.arange(T) * C.DT
    vl = np.maximum(vl0[:, None] - dec[:, None] * np.clip(t[None] - 1.0, 0, 2.0), 0)
    xl, al = _from_speed(vl, x0=gap0)
    return dict(lead_x=xl, lead_v=vl, lead_a=al, ego_v0=v0, gap0=gap0,
                params=pd.DataFrame({"v0": v0, "gap0": gap0, "dv0": v0 - vl0}))


def scenario_stop_and_go(n, rng, T=300):
    vbar = rng.uniform(10, 20, n)
    amp = rng.uniform(3, 9, n)
    period = rng.uniform(8, 18, n)
    t = np.arange(T) * C.DT
    vl = np.maximum(vbar[:, None] + amp[:, None] * np.sin(2 * np.pi * t[None] / period[:, None]), 0)
    gap0 = rng.uniform(1.0, 1.8, n) * vbar
    xl, al = _from_speed(vl, x0=gap0)
    return dict(lead_x=xl, lead_v=vl, lead_a=al, ego_v0=vbar.copy(), gap0=gap0,
                params=pd.DataFrame({"vbar": vbar, "amp": amp, "period": period}))


def scenario_stopped_traffic(n, rng, T=250):
    v0 = rng.uniform(15, 30, n)
    d0 = rng.uniform(50, 150, n)        # radar detection range
    vl = np.zeros((n, T))
    xl, al = _from_speed(vl, x0=d0)
    return dict(lead_x=xl, lead_v=vl, lead_a=al, ego_v0=v0, gap0=d0,
                params=pd.DataFrame({"v0": v0, "d0": d0, "required_decel": _required_decel(v0, d0)}))


def scenario_real_world(n=200, T=250):
    """Leaders replayed from NGSIM I-80 (unseen site); pick the most severe braking."""
    p = load("pairs_i-80").sort_values(["episode", "frame_id"])
    lens = p.groupby("episode").size()
    skip = 20                                    # start 2 s into each episode
    eps = lens[lens >= T + skip].index
    q = p[p.episode.isin(eps)]
    first = q[q.groupby("episode").cumcount().between(skip, skip + T - 1)]
    sev = first.groupby("episode")["a_l"].min().sort_values().index[:n]
    first = first[first.episode.isin(sev)]
    grp = [g for _, g in first.groupby("episode")]
    lead_x = np.stack([g["y_l"].to_numpy() - g["len_l"].to_numpy() for g in grp])
    lead_v = np.stack([g["v_l"].to_numpy() for g in grp])
    lead_a = np.stack([g["a_l"].to_numpy() for g in grp])
    ego_v0 = np.array([g["v"].iloc[0] for g in grp])
    gap0 = np.array([g["gap"].iloc[0] for g in grp])
    human = pd.DataFrame([{
        "collision": False, "impact_speed": 0.0, "min_ttc": g["ttc"].min(),
        "min_gap": g["gap"].min(),
        "TET": time_exposed_ttc(g["ttc"].to_numpy()), "TIT": time_integrated_ttc(g["ttc"].to_numpy()),
        "max_decel": -g["a"].min(),
        "rms_jerk": float(np.sqrt(np.mean((np.diff(g["a"].to_numpy()) / C.DT) ** 2))),
        "mean_speed": g["v"].mean(), "harsh_brake_s": float((g["a"] < -4).sum() * C.DT),
    } for g in grp])
    human_trace = {"gap": np.stack([g["gap"].to_numpy() for g in grp]),
                   "v": np.stack([g["v"].to_numpy() for g in grp]),
                   "a": np.stack([g["a"].to_numpy() for g in grp]),
                   "ttc": np.stack([g["ttc"].to_numpy() for g in grp])}
    return dict(lead_x=lead_x, lead_v=lead_v, lead_a=lead_a, ego_v0=ego_v0, gap0=gap0,
                human=human, human_trace=human_trace)


SCENARIOS = {
    "Lead emergency braking": scenario_emergency_brake,
    "Aggressive cut-in": scenario_cut_in,
    "Stop-and-go wave": scenario_stop_and_go,
    "Stopped traffic ahead": scenario_stopped_traffic,
}


def run_all(n_mc: int = 500):
    rng = np.random.default_rng(C.RANDOM_STATE)
    risk_model = RiskModel.load()
    summary, traces = {}, {}

    all_scen = {name: gen(n_mc, rng) for name, gen in SCENARIOS.items()}
    all_scen["NGSIM I-80 replay"] = scenario_real_world()

    for sname, sc in all_scen.items():
        summary[sname] = {}
        traces[sname] = {"lead_v": sc["lead_v"], "params": sc.get("params")}
        for ctrl in CONTROLLERS:
            rec = simulate(sc["lead_x"], sc["lead_v"], sc["lead_a"], sc["ego_v0"],
                           sc["gap0"], ctrl, risk_model, seed=1)
            df = summarise(rec)
            summary[sname][ctrl] = df
            traces[sname][ctrl] = rec
            print(f"{sname:24s} {ctrl:18s} collisions={df.collision.mean()*100:5.1f}% "
                  f"minTTC(med)={df.min_ttc.median():5.2f}s TET={df.TET.mean():5.2f}s "
                  f"maxDecel={df.max_decel.mean():4.2f}")
        if "human" in sc:
            summary[sname]["Human (replay)"] = sc["human"]
            traces[sname]["Human (replay)"] = sc["human_trace"]
            print(f"{sname:24s} {'Human (replay)':18s} minTTC(med)={sc['human'].min_ttc.median():5.2f}s "
                  f"TET={sc['human'].TET.mean():5.2f}s")

    avoidable = {}
    for sname, sc in all_scen.items():
        prm = sc.get("params")
        if prm is not None and "required_decel" in prm:
            avoidable[sname] = (prm["required_decel"] <= -C.VEHICLE_A_MIN).to_numpy()
        else:
            avoidable[sname] = np.ones(len(sc["ego_v0"]), bool)

    # comfort / efficiency are compared on the runs where *no* controller collided,
    # otherwise early termination of crashed runs biases the averages
    common = {s: ~np.any([d[c].collision.to_numpy() for c in CONTROLLERS], axis=0)
              for s, d in summary.items()}

    def cm(s, df, col):
        return float(df[col].to_numpy()[common[s]].mean())

    agg = {s: {c: {"collision_rate": float(df.collision.mean()),
                   "collision_rate_avoidable": float(df.collision[avoidable[s]].mean()),
                   "share_avoidable": float(avoidable[s].mean()),
                   "mean_impact_speed": float(df.loc[df.collision, "impact_speed"].mean()) if df.collision.any() else 0.0,
                   "median_min_ttc": float(df.min_ttc.median()),
                   "p5_min_ttc": float(df.min_ttc.quantile(0.05)),
                   "mean_TET": cm(s, df, "TET"), "mean_TIT": cm(s, df, "TIT"),
                   "mean_max_decel": cm(s, df, "max_decel"),
                   "mean_rms_jerk": cm(s, df, "rms_jerk"),
                   "mean_speed": cm(s, df, "mean_speed"),
                   "harsh_brake_s": cm(s, df, "harsh_brake_s"),
                   "n_runs": int(len(df)), "n_common": int(common[s].sum())}
               for c, df in d.items()} for s, d in summary.items()}
    with open(C.REPORTS / "simulation_results.json", "w") as f:
        json.dump(agg, f, indent=2)
    joblib.dump({"summary": summary, "traces": traces}, C.DATA_PROCESSED / "sim_traces.joblib")
    return agg


if __name__ == "__main__":
    run_all()
