"""Camera perception for the RC car: lane lines, stop line and traffic-light card.

Pipeline
  1. bird's-eye view (BEV): warp the camera image with the calibrated homography so
     every pixel maps to a point on the floor (200 px per metre by default)
  2. lane lines: threshold the tape colour, find the two lines with a column
     histogram + sliding windows, and fit l(f) = a f² + b f + c in metres
     (f = forward from the bumper, l = lateral, left positive)
  3. lane centre: average of both lines, or one line shifted by half a lane width
  4. stop line: a red band across the lane in the BEV
  5. traffic light: a compact red or green blob (card) in the camera image
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class LaneMeasurement:
    offset: float          # lateral position of the lane centre at the bumper [m], + = left
    heading: float         # lane direction relative to the car [rad], + = turning left
    curvature: float       # [1/m], + = curving left
    confidence: float      # 0..1
    n_lines: int
    fits: dict = field(default_factory=dict)

    def center_at(self, f):
        a, b, c = self.fits["center"]
        return a * f * f + b * f + c


class LaneDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.H = np.array(cfg.homography, float) if cfg.homography else None
        self.prev = None

    # ------------------------------------------------------------------ geometry
    def px_to_car(self, cols, rows):
        c = self.cfg
        f = c.bev_near + (c.bev_h - np.asarray(rows, float)) / c.bev_px_per_m
        lat = (c.bev_w / 2 - np.asarray(cols, float)) / c.bev_px_per_m
        return f, lat

    def car_to_px(self, f, lat):
        c = self.cfg
        cols = c.bev_w / 2 - np.asarray(lat) * c.bev_px_per_m
        rows = c.bev_h - (np.asarray(f) - c.bev_near) * c.bev_px_per_m
        return cols, rows

    def birdseye(self, frame):
        c = self.cfg
        if self.H is None:        # simulator frames are already a bird's-eye view
            return frame if frame.shape[:2] == (c.bev_h, c.bev_w) else cv2.resize(frame, (c.bev_w, c.bev_h))
        return cv2.warpPerspective(frame, self.H, (c.bev_w, c.bev_h))

    def _mask(self, img, lo, hi):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _red(self, img):
        c = self.cfg
        m = self._mask(img, c.red_hsv_lo, c.red_hsv_hi)
        lo2 = (170, c.red_hsv_lo[1], c.red_hsv_lo[2])
        return m | self._mask(img, lo2, (179, 255, 255))

    # ------------------------------------------------------------------ lanes
    def detect(self, frame):
        c = self.cfg
        bev = self.birdseye(frame)
        mask = self._mask(bev, c.lane_hsv_lo, c.lane_hsv_hi)
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        debug = dict(bev=bev, mask=mask, lines={})
        if len(xs) < 50:
            return None, debug
        hist = mask[int(h * 0.55):, :].sum(axis=0).astype(float)
        mid = w // 2
        bases = {}
        if self.prev is not None:          # start the search where the lines were last frame
            for side in ("left", "right"):
                if side in self.prev.fits:
                    col, _ = self.car_to_px(c.bev_near + 0.05, np.polyval(self.prev.fits[side], c.bev_near + 0.05))
                    if 0 <= col < w:
                        bases[side] = int(col)
        if "left" not in bases and hist[:mid].max() > 255 * 8:
            bases["left"] = int(np.argmax(hist[:mid]))
        if "right" not in bases and hist[mid:].max() > 255 * 8:
            bases["right"] = int(mid + np.argmax(hist[mid:]))

        nwin, margin, minpix = 10, 26, 12
        win_h = h // nwin
        fits = {}
        for side, x0 in bases.items():
            x = x0
            idx = []
            for k in range(nwin):
                y_lo, y_hi = h - (k + 1) * win_h, h - k * win_h
                sel = (ys >= y_lo) & (ys < y_hi) & (xs >= x - margin) & (xs < x + margin)
                good = np.nonzero(sel)[0]
                idx.append(good)
                if len(good) > minpix:
                    x = int(xs[good].mean())
            idx = np.concatenate(idx)
            f, lat = self.px_to_car(xs[idx], ys[idx])
            near = f <= c.bev_near + c.fit_range_m          # a quadratic only fits the near part of a tight bend
            idx, f, lat = idx[near], f[near], lat[near]
            if len(idx) < 60:
                continue
            fits[side] = np.polyfit(f, lat, 2)
            debug["lines"][side] = (xs[idx], ys[idx])

        # sanity: two lines must be roughly a lane apart and not swapped
        if "left" in fits and "right" in fits:
            sep = np.polyval(fits["left"], 0.4) - np.polyval(fits["right"], 0.4)
            if not (0.5 * c.lane_width < sep < 1.6 * c.lane_width):
                weaker = min(("left", "right"), key=lambda s_: len(debug["lines"][s_][0]))
                fits.pop(weaker)
        if not fits:
            self.prev = None
            return None, debug
        if "left" in fits and "right" in fits:
            center = (fits["left"] + fits["right"]) / 2
        elif "left" in fits:
            center = fits["left"] - np.array([0, 0, c.lane_width / 2])
        else:
            center = fits["right"] + np.array([0, 0, c.lane_width / 2])
        a, b, cc = center
        n_px = sum(len(v[0]) for v in debug["lines"].values())
        meas = LaneMeasurement(offset=float(cc), heading=float(np.arctan(b)), curvature=float(2 * a),
                               confidence=float(min(1.0, n_px / 900.0)), n_lines=len(fits),
                               fits=dict(fits, center=center))
        self.prev = meas
        return meas, debug

    # ------------------------------------------------------------------ markers
    def detect_stop_line(self, bev):
        """Distance [m] from the bumper to a red band crossing the lane, or None."""
        c = self.cfg
        red = self._red(bev)
        half = int(c.lane_width / 2 * c.bev_px_per_m)
        band = red[:, max(c.bev_w // 2 - half, 0): c.bev_w // 2 + half]
        frac = band.mean(axis=1) / 255.0
        rows = np.nonzero(frac > 0.45)[0]
        if len(rows) < 3:
            return None
        f, _ = self.px_to_car(c.bev_w / 2, rows.max())
        return float(f)

    def detect_light(self, frame, min_area=150):
        """'red' / 'green' if a compact coloured card is visible (wide red bands are stop lines)."""
        c = self.cfg
        found = {}
        for name, m in (("red", self._red(frame)), ("green", self._mask(frame, c.green_hsv_lo, c.green_hsv_hi))):
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                if cv2.contourArea(cnt) >= min_area and max(w, h) / max(min(w, h), 1) < 2.5:
                    found[name] = max(found.get(name, 0), cv2.contourArea(cnt))
        if not found:
            return None
        return max(found, key=found.get)

    # ------------------------------------------------------------------ drawing
    def draw(self, debug, meas, stop_m=None):
        c = self.cfg
        img = debug["bev"].copy()
        for side, (xs, ys) in debug["lines"].items():
            img[ys, xs] = (0, 200, 255) if side == "left" else (255, 160, 0)
        if meas is not None:
            ff = np.linspace(c.bev_near, c.bev_near + c.bev_h / c.bev_px_per_m, 30)
            for key, col in (("left", (0, 200, 255)), ("right", (255, 160, 0)), ("center", (80, 220, 80))):
                if key in meas.fits:
                    cols, rows = self.car_to_px(ff, np.polyval(meas.fits[key], ff))
                    pts = np.stack([cols, rows], 1).astype(np.int32)
                    cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
        cv2.line(img, (c.bev_w // 2, c.bev_h), (c.bev_w // 2, c.bev_h - 40), (255, 255, 255), 2)
        if stop_m is not None:
            _, r = self.car_to_px(stop_m, 0)
            cv2.line(img, (0, int(r)), (c.bev_w, int(r)), (0, 0, 255), 2)
        return img
