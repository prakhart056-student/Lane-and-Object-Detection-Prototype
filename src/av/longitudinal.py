"""Per-vehicle risk-aware speed control.

The same control law that was validated in `src/collision_avoidance.py`, written as
a stateful object so it can run one step at a time inside the 2-D simulator:

    a = min( IDM(time gap T * (1 + 0.8 p)),  pre-emptive braking(p),  AEB(min(TTC, MTTC)) )

where p is the ML-predicted probability of a conflict within 3 s, computed from a
1-s history of the ego and leader kinematics.
"""
from __future__ import annotations

import collections

import numpy as np

from .. import config as C
from ..collision_avoidance import idm_accel, staged_aeb
from ..features import W, onboard_from_window
from ..risk_metrics import modified_ttc, time_to_collision


class RiskAwareLongitudinal:
    def __init__(self, risk_model=None, risk_aware: bool = True, idm: dict | None = None,
                 acc_min: float = C.ACC_A_MIN, aeb: dict | None = None):
        self.model = risk_model
        self.risk_aware = risk_aware and risk_model is not None
        self.idm = dict(C.IDM, **(idm or {}))
        self.acc_min = acc_min
        self.aeb = dict(C.AEB, **(aeb or {}))
        self.hist = None
        self.leader_id = None
        self.risk_f = 0.0
        self.last = dict(risk=0.0, ttc=C.TTC_CAP, mttc=C.TTC_CAP, stage=0)

    # -------------------------------------------------------------- helpers
    def _predict(self):
        """Risk probability from the 1-s history (numpy fast path, same features as training)."""
        import warnings
        win = {k: np.asarray(dq, float)[None, :] for k, dq in self.hist.items()}
        X = onboard_from_window(win)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(self.model.model.predict_proba(X)[0, 1])

    def _idm(self, v, gap, dv, T, v0):
        p = self.idm
        s_star = p["s0"] + max(0.0, v * T + v * dv / (2 * np.sqrt(p["a_max"] * p["b_comf"])))
        free = 1 - (v / max(v0, 0.1)) ** p["delta"]
        inter = (s_star / max(gap, 0.1)) ** 2 if gap is not None else 0.0
        return max(p["a_max"] * (free - inter), self.acc_min)

    def _stage(self, t_min):
        a = self.aeb
        return 3 if t_min < a["ttc_full"] else 2 if t_min < a["ttc_partial"] else 1 if t_min < a["ttc_warn"] else 0

    def _aeb_accel(self, stage):
        return self.aeb["a_full"] if stage == 3 else self.aeb["a_partial"] if stage == 2 else np.inf

    # -------------------------------------------------------------- main entry
    def command(self, v, a, v0, leader=None, use_risk=True):
        """leader: dict(gap, v, a, id) in the ego lane, or None.  Returns acceleration."""
        if leader is None:
            self.hist, self.leader_id, self.risk_f = None, None, 0.0
            self.last = dict(risk=0.0, ttc=C.TTC_CAP, mttc=C.TTC_CAP, stage=0)
            return self._idm(v, None, 0.0, self.idm["T"], v0)

        gap, vl, al = max(leader["gap"], 0.05), max(leader["v"], 0.0), leader.get("a", 0.0)
        if self.hist is None or leader.get("id") != self.leader_id:   # new leader: warm-start history
            self.hist = {k: collections.deque([val] * W, maxlen=W)
                         for k, val in (("v", v), ("a", a), ("v_l", vl), ("a_l", al), ("gap", gap))}
            self.leader_id = leader.get("id")
        else:
            for k, val in (("v", v), ("a", a), ("v_l", vl), ("a_l", al), ("gap", gap)):
                self.hist[k].append(val)

        ttc = float(time_to_collision(np.array([gap]), np.array([v]), np.array([vl]))[0])
        # the ego's own acceleration is the controller's choice, so only its braking counts
        mttc = float(modified_ttc(np.array([gap]), np.array([v]), np.array([vl]),
                                  np.array([min(a, 0.0)]), np.array([al]))[0])
        risk = 0.0
        T = self.idm["T"]
        a_pre = np.inf
        if self.risk_aware and use_risk:
            r_now = self._predict() if v > 1.0 else 0.0
            self.risk_f = r_now if r_now > self.risk_f else 0.8 * self.risk_f + 0.2 * r_now
            risk = self.risk_f
            T = T * (1.0 + 0.8 * risk)
            thr = self.model.threshold
            if risk >= thr and (v - vl > 0 or al < -1.0) and v > 0.5:
                excess = np.clip((risk - thr) / (1 - thr), 0, 1)
                a_pre = -1.5 + (C.RISK_A_MIN + 1.5) * excess
            t_min = min(ttc, mttc)
        else:
            t_min = ttc                       # reactive AEB looks at plain TTC only
        stage = self._stage(t_min)
        # DRAC gate (as production AEB): a short TTC at walking pace with a tiny closing speed
        # needs almost no braking, so it is not an emergency; and no warning is raised when
        # the car is already braking at least as hard as the situation requires.
        drac = max(v - vl, 0.0) ** 2 / (2 * gap)
        if stage >= 2 and drac < self.aeb.get("drac_min", 1.0):
            stage = 1
        if stage == 1 and -a >= 0.9 * drac:
            stage = 0
        a_cmd = min(self._idm(v, gap, v - vl, T, v0), a_pre, self._aeb_accel(stage))
        self.last = dict(risk=risk, ttc=ttc, mttc=mttc, stage=stage, drac=drac)
        return float(a_cmd)
