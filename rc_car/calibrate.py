"""Calibration tools for the real RC car. Results are saved to rc_car/calibration.json.

    python -m rc_car.calibrate camera   # bird's-eye homography from an A4 sheet on the floor
    python -m rc_car.calibrate colors   # click the tape / stop line / light cards to set colours
    python -m rc_car.calibrate servo  --backend serial --port COM5   # steering centre and range
    python -m rc_car.calibrate esc    --backend serial --port COM5   # ESC neutral / slowest / fastest
    python -m rc_car.calibrate sensor --backend serial --port COM5   # live range readings

Put the car on a stand (wheels off the ground) for servo and esc calibration.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from .config import RCConfig
from .hal import make_car

A4_LONG, A4_SHORT = 0.297, 0.210


def open_camera(cfg, index=None):
    cap = cv2.VideoCapture(cfg.camera_index if index is None else index)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera not found (try --camera 1)")
    return cap


def camera(args, cfg):
    """Place an A4 sheet flat on the floor, centred in front of the car, its near (short)
    edge exactly --near metres in front of the bumper. Click its 4 corners in the order
    near-left, near-right, far-right, far-left, then press S to save."""
    cap = open_camera(cfg, args.camera)
    pts = []
    win = "click: near-left, near-right, far-right, far-left  |  S save  R reset  Q quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda e, x, y, *_: pts.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4 else None)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for p in pts:
            cv2.circle(frame, p, 6, (0, 0, 255), -1)
        if len(pts) == 4:
            cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        if k == ord("q"):
            break
        if k == ord("s") and len(pts) == 4:
            ppm, W, H, near = cfg.bev_px_per_m, cfg.bev_w, cfg.bev_h, cfg.bev_near
            def px(f, lat):
                return [W / 2 - lat * ppm, H - (f - near) * ppm]
            n = args.near
            dst = np.float32([px(n, A4_SHORT / 2), px(n, -A4_SHORT / 2),
                              px(n + A4_LONG, -A4_SHORT / 2), px(n + A4_LONG, A4_SHORT / 2)])
            Hm = cv2.getPerspectiveTransform(np.float32(pts), dst)
            cfg.save(homography=Hm.tolist())
            bev = cv2.warpPerspective(frame, Hm, (W, H))
            cv2.imshow("bird's-eye check (the sheet should look like an upright rectangle)", bev)
            print("saved homography to rc_car/calibration.json")
            cv2.waitKey(0)
            break
    cap.release(); cv2.destroyAllWindows()


def colors(args, cfg):
    """Click on: 1) the lane tape, 2) the red stop line, 3) the green light card. S saves."""
    cap = open_camera(cfg, args.camera)
    picks = []
    names = ["lane tape", "red stop line / red card", "green card"]
    win = "click " + " -> ".join(names) + "  |  S save  Q quit"
    cv2.namedWindow(win)
    frame_box = {}
    def on(e, x, y, *_):
        if e == cv2.EVENT_LBUTTONDOWN and len(picks) < 3:
            hsv = cv2.cvtColor(frame_box["f"], cv2.COLOR_BGR2HSV)
            patch = hsv[max(y - 5, 0):y + 6, max(x - 5, 0):x + 6].reshape(-1, 3)
            picks.append(np.median(patch, axis=0).astype(int))
            print(f"  {names[len(picks) - 1]}: HSV {picks[-1].tolist()}")
    cv2.setMouseCallback(win, on)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_box["f"] = frame
        cv2.imshow(win, frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and picks:
            upd = {}
            if len(picks) >= 1:
                h, s_, v = (int(x) for x in picks[0])
                if s_ < 60:     # white / grey tape: low saturation, bright
                    upd.update(lane_hsv_lo=[0, 0, max(v - 60, 100)], lane_hsv_hi=[179, min(s_ + 50, 255), 255])
                else:           # coloured tape (e.g. yellow)
                    upd.update(lane_hsv_lo=[max(h - 12, 0), max(s_ - 70, 40), max(v - 80, 60)],
                               lane_hsv_hi=[min(h + 12, 179), 255, 255])
            if len(picks) >= 2:
                h, s, v = picks[1]
                upd.update(red_hsv_lo=[0, max(int(s) - 80, 60), max(int(v) - 90, 50)], red_hsv_hi=[10, 255, 255])
            if len(picks) >= 3:
                h, s, v = picks[2]
                upd.update(green_hsv_lo=[max(int(h) - 15, 0), max(int(s) - 80, 60), max(int(v) - 90, 50)],
                           green_hsv_hi=[min(int(h) + 15, 179), 255, 255])
            cfg.save(**upd)
            print("saved colours to rc_car/calibration.json:", upd)
            break
    cap.release(); cv2.destroyAllWindows()


def _keyboard_loop(title, lines, on_key):
    img = np.full((220, 620, 3), 245, np.uint8)
    while True:
        canvas = img.copy()
        for k, ln in enumerate(lines()):
            cv2.putText(canvas, ln, (14, 34 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 30, 20), 1, cv2.LINE_AA)
        cv2.imshow(title, canvas)
        k = cv2.waitKey(50) & 0xFF
        if k == ord("q"):
            break
        on_key(k)
    cv2.destroyAllWindows()


def servo(args, cfg):
    car = make_car(args.backend, cfg, port=args.port)
    st = dict(center=cfg.steer_center_us, rng=cfg.steer_range_us, pos=0)
    def send():
        us = st["center"] + st["pos"] * st["rng"]
        car.raw(us, cfg.esc_neutral_us)
    def key(k):
        if k == ord("a"): st["center"] -= 5
        if k == ord("d"): st["center"] += 5
        if k == ord("w"): st["rng"] += 10
        if k == ord("s"): st["rng"] -= 10
        if k == ord("1"): st["pos"] = -1
        if k == ord("2"): st["pos"] = 0
        if k == ord("3"): st["pos"] = 1
        if k == ord("x"):
            cfg.save(steer_center_us=st["center"], steer_range_us=st["rng"]); print("saved", st)
        send()
    _keyboard_loop("servo calibration", lambda: [
        f"centre {st['center']} us   range {st['rng']} us   position {['full left', 'centre', 'full right'][st['pos'] + 1]}",
        "A/D: move centre until the wheels point straight", "1/2/3: full left / centre / full right",
        "W/S: range until full lock does not strain the servo", "X: save    Q: quit"], key)
    car.close()


def esc(args, cfg):
    car = make_car(args.backend, cfg, port=args.port)
    st = dict(us=cfg.esc_neutral_us)
    def key(k):
        if k == ord("w"): st["us"] += 5
        if k == ord("s"): st["us"] -= 5
        if k == ord("n"): st["us"] = cfg.esc_neutral_us
        if k == ord("1"): cfg.save(esc_neutral_us=st["us"]); print("neutral", st["us"])
        if k == ord("2"): cfg.save(esc_min_fwd_us=st["us"]); print("slowest forward", st["us"])
        if k == ord("3"): cfg.save(esc_max_us=st["us"]); print("v_max pulse", st["us"])
        if k == ord("4"): cfg.save(esc_brake_us=st["us"]); print("brake", st["us"])
        car.raw(cfg.steer_center_us, st["us"])
    _keyboard_loop("ESC calibration (WHEELS OFF THE GROUND)", lambda: [
        f"ESC pulse {st['us']} us", "W/S: +/- 5 us    N: neutral",
        "1: save as neutral   2: save as slowest forward (wheels just turn)",
        f"3: save as top speed (= v_max {cfg.v_max} m/s)   4: save as brake",
        "Q: quit (returns to neutral)"], key)
    car.close()


def sensor(args, cfg):
    car = make_car(args.backend, cfg, port=args.port)
    print("range readings (Ctrl+C to stop):")
    try:
        while True:
            s = car.read()
            print(f"\r  {s.range_m if s.range_m is not None else float('nan'):5.3f} m   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        car.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", choices=["camera", "colors", "servo", "esc", "sensor"])
    ap.add_argument("--backend", choices=["serial", "pi"], default="serial")
    ap.add_argument("--port", default=None)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--near", type=float, default=0.30, help="camera tool: bumper to the near edge of the A4 sheet [m]")
    args = ap.parse_args()
    cfg = RCConfig.load()
    dict(camera=camera, colors=colors, servo=servo, esc=esc, sensor=sensor)[args.tool](args, cfg)


if __name__ == "__main__":
    main()
