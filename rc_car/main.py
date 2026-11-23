"""Run the RC-car autopilot.

    python -m rc_car.main --backend sim                       # simulated car, no hardware
    python -m rc_car.main --backend sim --view --realtime     # watch the simulation
    python -m rc_car.main --backend serial --port COM5 --view # laptop + Arduino car
    python -m rc_car.main --backend pi --stream 8080          # Raspberry Pi car, video in a browser

Keys in the view window:  SPACE start / pause   E emergency stop   Q quit
On a headless Pi use --autostart (with --stream to watch) and Ctrl+C to stop.
Every run writes rc_car/logs/run_<time>.csv (all sensor values, decisions, commands).
"""
from __future__ import annotations

import argparse
import csv
import http.server
import signal
import socketserver
import threading
import time

import cv2
import numpy as np

from .autopilot import Autopilot
from .config import LOG_DIR, RCConfig
from .hal import SimCar, make_car
from .lane_detection import LaneDetector

STATE_BGR = {"CRUISE": (122, 175, 27), "FOLLOW": (214, 120, 42), "WARNING": (0, 161, 237),
             "AEB_PARTIAL": (52, 104, 235), "AEB_FULL": (47, 49, 200), "STOP_RED": (47, 49, 200),
             "STOP_SIGN": (0, 140, 230), "LANE_LOST": (0, 161, 237), "IDLE": (120, 113, 107),
             "ESTOP": (47, 49, 200), "FAULT: range sensor": (47, 49, 200)}


# ---------------------------------------------------------------------------
# optional MJPEG stream so the view can be watched from a laptop browser
# ---------------------------------------------------------------------------
class _Stream:
    frame = None

    @classmethod
    def serve(cls, port):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        if cls.frame is not None:
                            ok, jpg = cv2.imencode(".jpg", cls.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *a):
                pass

        srv = socketserver.ThreadingTCPServer(("0.0.0.0", port), Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"  live view: http://<this-computer-ip>:{port}/")


def overlay(cfg, det, debug, lane, cmd, sens, stop_m, light, cam_frame):
    bev = det.draw(debug, lane, stop_m)
    h = max(cfg.bev_h, 320)
    panel = np.full((h, 330, 3), (247, 246, 242), np.uint8)
    col = STATE_BGR.get(cmd.state, (61, 33, 20))
    cv2.rectangle(panel, (10, 10), (320, 56), col, -1)
    cv2.putText(panel, cmd.state, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    i = cmd.info
    ttc = i.get("ttc", np.inf)
    rows = [f"speed cmd  {cmd.speed:4.2f} m/s", f"steer      {np.degrees(cmd.steer):+5.1f} deg",
            f"range      {sens.range_m:4.2f} m" if sens.range_m is not None else "range      --",
            f"closing    {i.get('closing', 0):4.2f} m/s",
            f"TTC        {'inf' if not np.isfinite(ttc) or ttc > 19 else f'{ttc:4.1f} s'}",
            f"AEB stage  {i.get('stage', 0)}",
            f"lane       " + (f"off {lane.offset * 100:+4.0f} cm  conf {lane.confidence:.2f}" if lane else "NOT FOUND"),
            f"stop line  " + (f"{stop_m:4.2f} m" if stop_m is not None else "--"),
            f"light      {light or '--'}", f"brake      {'ON' if cmd.brake else 'off'}"]
    for k, r in enumerate(rows):
        cv2.putText(panel, r, (14, 86 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (61, 33, 20), 1, cv2.LINE_AA)
    parts = [cv2.resize(bev, (int(cfg.bev_w * h / cfg.bev_h), h)), panel]
    if cam_frame is not None and det.H is not None:
        cam = cv2.resize(cam_frame, (int(cam_frame.shape[1] * h / cam_frame.shape[0]), h))
        parts.insert(0, cam)
    return np.hstack(parts)


def run(args):
    cfg = RCConfig.load()
    if args.speed:
        cfg.v_cruise = args.speed
    car = make_car(args.backend, cfg, port=args.port, camera=args.camera, scenario=args.scenario, seed=args.seed)
    sim = isinstance(car, SimCar)
    det, ap = LaneDetector(cfg), Autopilot(cfg)
    if args.autostart or sim:
        ap.start()
    if args.stream:
        _Stream.serve(args.stream)
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{args.backend}.csv"
    fields = ["t", "state", "steer_deg", "speed_cmd", "brake", "range_m", "gap", "closing", "ttc", "stage",
              "lane_offset", "lane_heading_deg", "lane_conf", "stop_line_m", "light"]
    f = open(log_path, "w", newline="")
    wr = csv.writer(f); wr.writerow(fields)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    period = 1.0 / cfg.rate_hz
    print(f"RC autopilot running on '{args.backend}'  (log: {log_path})")
    if not sim:
        print("  SPACE = start/pause, E = emergency stop, Q = quit" if args.view else "  Ctrl+C to stop")
    n = 0
    try:
        while not stop["flag"]:
            t_loop = time.time()
            sens = car.read()
            if sens.frame is None:
                cmd = ap.update(sens.t, None, sens.range_m, None, None, sens.range_t)
                stop_m = light = None; lane = None; debug = None
            else:
                lane, debug = det.detect(sens.frame)
                stop_m = det.detect_stop_line(debug["bev"])
                light = det.detect_light(sens.frame)
                cmd = ap.update(sens.t, lane, sens.range_m, stop_m, light, sens.range_t, sens.v_meas)
            car.command(cmd.steer, cmd.speed, cmd.brake)
            i = cmd.info
            wr.writerow([f"{sens.t:.3f}", cmd.state, f"{np.degrees(cmd.steer):.2f}", f"{cmd.speed:.3f}", int(cmd.brake),
                         f"{sens.range_m:.3f}" if sens.range_m is not None else "",
                         f"{i['gap']:.3f}" if i.get("gap") is not None else "", f"{i.get('closing', 0):.3f}",
                         f"{min(i.get('ttc', 99), 99):.2f}", i.get("stage", 0),
                         f"{lane.offset:.3f}" if lane else "", f"{np.degrees(lane.heading):.1f}" if lane else "",
                         f"{lane.confidence:.2f}" if lane else "", f"{stop_m:.2f}" if stop_m is not None else "",
                         light or ""])
            if (args.view or args.stream) and debug is not None:
                img = overlay(cfg, det, debug, lane, cmd, sens, stop_m, light, sens.frame)
                _Stream.frame = img
                if args.view:
                    cv2.imshow("RC autopilot  (SPACE start/pause, E e-stop, Q quit)", img)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord(" "):
                        ap.pause() if ap.enabled else ap.start()
                    if k == ord("e"):
                        ap.estop(); print("  EMERGENCY STOP")
            if sim:
                car.advance()
                if sens.t >= args.duration:
                    break
                if args.realtime:
                    time.sleep(max(0.0, period - (time.time() - t_loop)))
            else:
                time.sleep(max(0.0, period - (time.time() - t_loop)))
                if args.duration and sens.t >= args.duration:
                    break
            n += 1
    finally:
        car.close()
        f.close()
        cv2.destroyAllWindows()
    print(f"  {n} control cycles, log saved to {log_path}")
    for t, msg in ap.events:
        print(f"    t={t:6.2f}s  {msg}")
    if sim:
        e = np.array(car.lat_err)
        print(f"  simulation: distance {car.distance:.1f} m, lane-keeping RMS {np.sqrt(np.mean(e ** 2)) * 100:.1f} cm "
              f"(max {np.abs(e).max() * 100:.1f} cm), closest approach {car.min_gap * 100:.0f} cm, "
              f"collisions {car.collisions}")
        return dict(distance=car.distance, rms=float(np.sqrt(np.mean(e ** 2))), max_err=float(np.abs(e).max()),
                    min_gap=float(car.min_gap), collisions=car.collisions, events=list(ap.events), log=str(log_path))
    return dict(log=str(log_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["sim", "serial", "pi"], default="sim")
    ap.add_argument("--port", default=None, help="serial port of the Arduino (e.g. COM5 or /dev/ttyACM0)")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--speed", type=float, default=None, help="cruise speed [m/s] (start low: 0.4)")
    ap.add_argument("--view", action="store_true", help="show the bird's-eye view and decisions")
    ap.add_argument("--stream", type=int, default=0, help="serve the view as MJPEG on this port")
    ap.add_argument("--autostart", action="store_true", help="start driving immediately (real car)")
    ap.add_argument("--duration", type=float, default=45.0, help="seconds to run (simulator; 0 = forever on a car)")
    ap.add_argument("--realtime", action="store_true", help="simulator: run at wall-clock speed")
    ap.add_argument("--scenario", default="full", choices=["full", "follow", "obstacle", "stopline", "light", "empty"],
                    help="simulator track contents")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
