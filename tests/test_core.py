"""Unit tests for risk metrics, feature consistency and the simulator.

Run:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import risk_metrics as R
from src.collision_avoidance import idm_accel, simulate, staged_aeb
from src.features import ONBOARD, W, add_history_features, onboard_from_window


# ----------------------------------------------------------------- metrics
def test_ttc_basic_and_non_closing():
    assert R.time_to_collision(30.0, 20.0, 10.0) == pytest.approx(3.0)
    assert R.time_to_collision(30.0, 10.0, 20.0) == C.TTC_CAP     # opening gap


def test_thw_and_drac():
    assert R.time_headway(25.0, 25.0) == pytest.approx(1.0)
    # closing 10 m/s over 10 m needs 5 m/s^2
    assert R.drac(10.0, 20.0, 10.0) == pytest.approx(5.0)
    assert R.drac(10.0, 10.0, 20.0) == 0.0


def test_mttc_reduces_to_ttc_without_relative_accel():
    assert R.modified_ttc(30.0, 20.0, 10.0, 0.0, 0.0) == pytest.approx(3.0)


def test_mttc_detects_braking_leader():
    # equal speeds -> TTC infinite, but a braking leader makes a collision possible
    assert R.time_to_collision(20.0, 20.0, 20.0) == C.TTC_CAP
    mttc = R.modified_ttc(20.0, 20.0, 20.0, 0.0, -5.0)
    assert mttc == pytest.approx(np.sqrt(2 * 20 / 5.0))


def test_cri_monotone_and_bounded():
    ttc = np.array([20, 6, 3, 1.5, 0.5])
    cri = R.composite_risk_index(ttc, np.full(5, 1.5), np.array([0, 0.5, 1.5, 4, 8]))
    assert np.all(np.diff(cri) > 0)
    assert cri.min() >= 0 and cri.max() <= 1


def test_tet_tit():
    ttc = np.array([5, 2.5, 2.0, 4.0])
    assert R.time_exposed_ttc(ttc) == pytest.approx(0.2)
    assert R.time_integrated_ttc(ttc) == pytest.approx((0.5 + 1.0) * C.DT)


# ----------------------------------------------------------------- features
def test_online_features_match_offline_features():
    """The controller must compute exactly the features the model was trained on."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "episode": 1, "frame_id": np.arange(n),
        "v": 20 + rng.normal(0, 1, n), "a": rng.normal(0, 1, n),
        "v_l": 18 + rng.normal(0, 1, n), "a_l": rng.normal(0, 1, n),
        "gap": 25 + rng.normal(0, 2, n),
    })
    df["dv"] = df.v - df.v_l
    df["thw"] = R.time_headway(df.gap, df.v)
    df["inv_ttc"] = R.inverse_ttc(df.gap, df.v, df.v_l)
    df["drac"] = R.drac(df.gap, df.v, df.v_l)
    df["mttc"] = R.modified_ttc(df.gap, df.v, df.v_l, df.a, df.a_l)
    off = add_history_features(df).iloc[-1][ONBOARD].to_numpy(float)

    win = {k: df[k].to_numpy()[-W:][None, :] for k in ("v", "a", "v_l", "a_l", "gap")}
    on = onboard_from_window(win)[0]
    np.testing.assert_allclose(on, off, rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------- control
def test_idm_equilibrium_is_near_zero_accel():
    v = 20.0
    s_eq = (C.IDM["s0"] + v * C.IDM["T"]) / np.sqrt(1 - (v / C.IDM["v0"]) ** 4)
    assert abs(idm_accel(np.array([v]), np.array([s_eq]), np.array([0.0]))[0]) < 1e-6


def test_aeb_stages():
    out = staged_aeb(np.array([5.0, 1.2, 0.5]))
    assert out[0] == np.inf and out[1] == C.AEB["a_partial"] and out[2] == C.AEB["a_full"]


def test_simulator_no_collision_with_steady_leader():
    T, N = 200, 3
    v = np.full((N, T), 20.0)
    x = 40.0 + np.cumsum(np.c_[np.zeros(N), v[:, :-1] * C.DT], axis=1)
    rec = simulate(x, v, np.zeros((N, T)), np.full(N, 20.0), np.full(N, 40.0), "Reactive AEB")
    assert not rec["collided"].any()
    assert np.nanmin(rec["gap"]) > 20


def test_simulator_detects_unavoidable_collision():
    T, N = 100, 1
    v = np.zeros((N, T)); x = np.full((N, T), 5.0)      # stopped car 5 m ahead
    rec = simulate(x, v, np.zeros((N, T)), np.array([30.0]), np.array([5.0]), "Reactive AEB")
    assert rec["collided"].all()
