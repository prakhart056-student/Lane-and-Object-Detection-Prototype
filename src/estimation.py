"""State estimation shared by the webcam demo and the RC car.

RangeEstimator turns a noisy stream of distance measurements (camera size or a
range sensor) into distance, range rate and range acceleration; Debounce is the
confirmation / hold logic used before raising or lowering a warning level.
"""
from __future__ import annotations

import collections

import numpy as np


class RangeEstimator:
    """Range, range-rate and range-acceleration from noisy monocular distance.

    A quadratic is fitted to the raw distances of the last `window_s` seconds.
    Monocular range noise grows with distance (one pixel of width is about 3.5 %
    of the range at 2 m), so the speed deadband grows with distance too, and the
    acceleration is trusted only after a real approach has lasted 0.3 s.
    """

    def __init__(self, window_s=1.0):
        self.window = window_s
        self.reset()

    def reset(self):
        self.buf = collections.deque()
        self.v_s, self.a_s, self.approach_since = 0.0, 0.0, None

    def update(self, t, d):
        if self.buf:                                   # clamp single-frame outliers
            med = float(np.median([b[1] for b in list(self.buf)[-5:]]))
            d = float(np.clip(d, 0.7 * med, 1.3 * med))
        self.buf.append((t, d))
        while self.buf and t - self.buf[0][0] > self.window:
            self.buf.popleft()
        if len(self.buf) < 6:
            return d, 0.0, 0.0
        ts = np.array([b[0] for b in self.buf]) - t
        ds = np.array([b[1] for b in self.buf])
        acc, vel, dist = np.polyfit(ts, ds, 2) * np.array([2.0, 1.0, 1.0])
        self.v_s = 0.6 * self.v_s + 0.4 * float(vel)
        self.a_s = 0.8 * self.a_s + 0.2 * float(np.clip(acc, -1.5, 1.5))
        dead = 0.04 + 0.06 * max(float(dist), 0.0)    # m/s; pixel jitter must not look like motion
        v_out = self.v_s if abs(self.v_s) > dead else 0.0
        if v_out < 0:                                  # range shrinking: an approach is under way
            self.approach_since = t if self.approach_since is None else self.approach_since
        else:
            self.approach_since = None
        steady = self.approach_since is not None and t - self.approach_since >= 0.3
        a_out = self.a_s if steady and abs(self.a_s) > 0.3 else 0.0
        return float(dist), float(v_out), float(a_out)


class Debounce:
    """Confirmation logic: raise a level only after `up` frames, lower it after `down` frames."""

    def __init__(self, up=3, down=10):
        self.up, self.down = up, down
        self.reset()

    def reset(self):
        self.state, self.pending, self.count = 0, 0, 0

    def update(self, cand):
        if cand == self.state:
            self.count = 0
            return self.state
        if cand == self.pending:
            self.count += 1
        else:
            self.pending, self.count = cand, 1
        if self.count >= (self.up if cand > self.state else self.down):
            self.state, self.count = cand, 0
        return self.state
