"""Live webcam collision-warning demo.

The laptop camera plays the role of the vehicle's forward sensor. An object in
front of it (a face, or any coloured object such as a toy car) is detected in
every frame; its distance and approach speed are estimated from its apparent
size, and the project's own risk pipeline runs live:

    distance  ->  TTC, MTTC, DRAC, Composite Risk Index (src/risk_metrics.py)
              ->  staged AEB decision on min(TTC, MTTC)  (thresholds from src/config.py)

    python webcam_demo.py                     # face mode (walk towards the camera)
    python webcam_demo.py --color             # colour mode: click the object to track
    python webcam_demo.py --width 0.10        # real width of your object in metres
    python webcam_demo.py --video clip.mp4    # run on a recorded video instead

Keys:  click = track the clicked object's colour   M = face/colour mode
       K = calibrate (hold the object at --calib-dist, default 0.50 m)
       R = reset    S = screenshot    Q / Esc = quit (saves a session log + plot)

At walking speeds the warning levels use the same time thresholds as the car:
TTC below 3 s is critical, and the AEB stages fire at 2.6 / 1.6 / 0.9 s.
"""
from __future__ import annotations

import argparse
import collections
import os
import time
import warnings

warnings.filterwarnings("ignore")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import config as C  # noqa: E402
from src import risk_metrics as R  # noqa: E402
from src.estimation import Debounce, RangeEstimator  # noqa: E402

LEVEL_NAMES = ["SAFE", "CAUTION", "DANGER", "CRITICAL"]
# BGR colours
LEVEL_BGR = [(122, 175, 27), (0, 161, 237), (52, 104, 235), (47, 49, 200)]
INK, PANEL, MUTED = (61, 33, 20), (242, 246, 247), (120, 113, 107)
FACE_WIDTH_M = 0.16          # typical adult face width, used in face mode
PANEL_W = 420


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def load_face_detector():
    if not hasattr(cv2, "CascadeClassifier"):      # OpenCV 5 dropped the bundled Haar cascades
        return None
    names = ["haarcascade_frontalface_default.xml"]
    dirs = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        dirs.append(cv2.data.haarcascades)
    dirs.append(os.path.join(os.path.dirname(cv2.__file__), "data"))
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    return c
    return None


def detect_face(gray, cascade, prev):
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
    if len(faces) == 0:
        return None
    if prev is not None:   # keep following the same face: closest to the previous box
        px, py = prev[0] + prev[2] / 2, prev[1] + prev[3] / 2
        return min(faces, key=lambda f: (f[0] + f[2] / 2 - px) ** 2 + (f[1] + f[3] / 2 - py) ** 2)
    return max(faces, key=lambda f: f[2] * f[3])


def detect_color(hsv, target, tol):
    h, s, v = target
    lo = np.array([max(h - tol, 0), max(s - 70, 40), max(v - 90, 40)], np.uint8)
    hi = np.array([min(h + tol, 179), 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    if h - tol < 0:      # red wraps around the hue circle
        mask |= cv2.inRange(hsv, np.array([180 + h - tol, lo[1], lo[2]], np.uint8), np.array([179, 255, 255], np.uint8))
    if h + tol > 179:
        mask |= cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), np.array([h + tol - 180, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 150:
        return None
    return cv2.boundingRect(c)


def assess(gap, closing, approach_acc):
    """Run the project's surrogate safety measures on one measurement."""
    g, vf, vl = np.array([max(gap, 0.01)]), np.array([max(closing, 0.0)]), np.array([0.0])
    ttc = float(R.time_to_collision(g, vf, vl)[0])
    mttc = float(R.modified_ttc(g, vf, vl, np.array([approach_acc]), np.array([0.0]))[0])
    drac = float(R.drac(g, vf, vl)[0])
    thw = float(R.time_headway(g, vf)[0])
    cri = float(R.composite_risk_index(np.array([ttc]), np.array([thw]), np.array([drac]))[0])
    level = int(R.risk_level(np.array([cri]))[0])
    t_min = min(ttc, mttc)
    a = C.AEB
    stage = 3 if t_min < a["ttc_full"] else 2 if t_min < a["ttc_partial"] else 1 if t_min < a["ttc_warn"] else 0
    return dict(ttc=ttc, mttc=mttc, drac=drac, cri=cri, level=level, stage=stage, t_min=t_min)


AEB_STAGES = [("NO ACTION", ""), ("COLLISION WARNING", "driver alert"),
              ("PARTIAL BRAKE", f"{C.AEB['a_partial']:.0f} m/s2"),
              ("EMERGENCY BRAKE", f"{C.AEB['a_full']:.0f} m/s2")]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def put(img, text, org, scale=0.6, color=INK, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def fmt_t(x):
    return "inf" if x >= C.TTC_CAP - 1e-6 else f"{x:4.1f} s"


def draw_panel(h, st, hist, mode, calibrated):
    p = np.full((h, PANEL_W, 3), PANEL, np.uint8)
    put(p, "RISK-AWARE COLLISION WARNING", (16, 30), 0.58, INK, 2)
    put(p, f"sensor: webcam  |  mode: {mode}" + ("  |  calibrated" if calibrated else ""), (16, 52), 0.42, MUTED)
    if st is None:
        put(p, "No object detected", (16, 100), 0.7, MUTED, 2)
        put(p, "face mode: look at the camera", (16, 130), 0.5, MUTED)
        put(p, "colour mode: click on the object", (16, 152), 0.5, MUTED)
        return p
    lvl = st["level"]
    rows = [("Distance", f"{st['gap']:.2f} m"), ("Closing speed", f"{st['closing']:+.2f} m/s"),
            ("TTC", fmt_t(st["ttc"])), ("Modified TTC", fmt_t(st["mttc"])),
            ("DRAC", f"{st['drac']:.2f} m/s2"), ("Risk index (CRI)", f"{st['cri']:.2f}")]
    y = 88
    for k, v in rows:
        put(p, k, (16, y), 0.55, MUTED); put(p, v, (230, y), 0.62, INK, 2); y += 30
    # CRI bar
    y += 2
    cv2.rectangle(p, (16, y), (PANEL_W - 16, y + 16), (220, 222, 226), -1)
    cv2.rectangle(p, (16, y), (16 + int((PANEL_W - 32) * min(st["cri"], 1)), y + 16), LEVEL_BGR[lvl], -1)
    for f in (0.25, 0.5, 0.75):
        x = 16 + int((PANEL_W - 32) * f)
        cv2.line(p, (x, y - 2), (x, y + 18), INK, 1)
    y += 46
    # level + AEB banner
    cv2.rectangle(p, (16, y - 26), (PANEL_W - 16, y + 12), LEVEL_BGR[lvl], -1)
    put(p, f"RISK: {LEVEL_NAMES[lvl]}", (28, y), 0.8, (255, 255, 255), 2)
    y += 50
    name, detail = st["aeb"]
    col = LEVEL_BGR[3] if "BRAKE" in name else LEVEL_BGR[1] if "WARNING" in name else LEVEL_BGR[0]
    cv2.rectangle(p, (16, y - 26), (PANEL_W - 16, y + 30), col, -1)
    put(p, name, (28, y), 0.8, (255, 255, 255), 2)
    put(p, "AEB: " + (detail or "monitoring"), (28, y + 22), 0.5, (255, 255, 255), 1)
    # live TTC chart (last 10 s)
    y0, y1 = y + 60, h - 30
    if y1 - y0 > 60:
        put(p, "min(TTC, MTTC), last 10 s", (16, y0 - 8), 0.45, MUTED)
        cv2.rectangle(p, (16, y0), (PANEL_W - 16, y1), (255, 255, 255), -1)
        top = 8.0
        for thr, c in ((C.AEB["ttc_warn"], LEVEL_BGR[1]), (C.AEB["ttc_partial"], LEVEL_BGR[2]),
                       (C.AEB["ttc_full"], LEVEL_BGR[3])):
            yy = int(y1 - (y1 - y0) * thr / top)
            cv2.line(p, (16, yy), (PANEL_W - 16, yy), c, 1)
            put(p, f"{thr:.1f}s", (PANEL_W - 60, yy - 3), 0.38, c)
        if len(hist) > 1:
            tn = hist[-1][0]
            pts = [(int(PANEL_W - 16 - (tn - t) / 10 * (PANEL_W - 32)),
                    int(y1 - (y1 - y0) * min(v, top) / top)) for t, v in hist if tn - t <= 10]
            if len(pts) > 1:
                cv2.polylines(p, [np.array(pts, np.int32)], False, (214, 120, 42), 2, cv2.LINE_AA)
    return p


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None, help="run on a video file instead of the camera")
    ap.add_argument("--color", action="store_true", help="start in colour-tracking mode")
    ap.add_argument("--hsv", default=None, help="colour to track as H,S,V (OpenCV ranges)")
    ap.add_argument("--width", type=float, default=0.10, help="real width of the coloured object [m]")
    ap.add_argument("--fov", type=float, default=60.0, help="camera horizontal field of view [deg]")
    ap.add_argument("--calib-dist", type=float, default=0.50, help="distance used by the K key [m]")
    ap.add_argument("--out", default=None, help="also write the annotated video to this file")
    ap.add_argument("--headless", action="store_true", help="no window (for testing with --video)")
    args = ap.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera/video. Try --camera 1, or close other apps using the camera.")
    H, W = frame.shape[:2]
    fps_file = cap.get(cv2.CAP_PROP_FPS) if args.video else 0

    cascade = load_face_detector()
    mode = "colour" if (args.color or args.hsv or cascade is None) else "face"
    target = tuple(int(x) for x in args.hsv.split(",")) if args.hsv else None
    focal = (W / 2) / np.tan(np.radians(args.fov) / 2)     # pixels
    calibrated = False
    est = RangeEstimator()
    deb_level, deb_aeb = Debounce(), Debounce()
    hist = collections.deque(maxlen=600)
    log = []
    prev_box = None
    click = {"pt": None}
    writer = None
    win = "Risk-Aware Collision Warning  (Q to quit)"

    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, W + PANEL_W, max(H, 620))
        cv2.setMouseCallback(win, lambda ev, x, y, *_: click.update(pt=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)

    print("Webcam collision-warning demo running.  Q = quit, M = mode, K = calibrate, click = track colour")
    t_start, n = time.time(), 0
    last_level, last_aeb = None, None
    while ok:
        t = (n / fps_file) if fps_file else (time.time() - t_start)
        n += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if click["pt"] is not None and click["pt"][0] < W:
            x, y = click["pt"]
            patch = hsv[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5].reshape(-1, 3)
            target = tuple(int(v) for v in np.median(patch, axis=0))
            mode = "colour"; est.reset(); prev_box = None
            print(f"  tracking colour HSV={target}")
        click["pt"] = None

        box = None
        if mode == "face" and cascade is not None:
            box = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cascade, prev_box)
            real_w = FACE_WIDTH_M
        elif target is not None:
            box = detect_color(hsv, target, 10)
            real_w = args.width
        st = None
        if box is not None:
            x, y, w, h = (int(v) for v in box)
            prev_box = (x, y, w, h)
            d_raw = focal * real_w / max(w, 1)
            d, vel, acc = est.update(t, d_raw)
            closing, approach_acc = -vel, -acc
            st = assess(d, closing, approach_acc)
            st["level"] = deb_level.update(st["level"])
            st["aeb"] = AEB_STAGES[deb_aeb.update(st["stage"])]
            st.update(gap=d, closing=closing)
            hist.append((t, min(st["t_min"], 99)))
            log.append((round(t, 3), round(d, 4), round(closing, 4), st["ttc"], st["mttc"], st["drac"], st["cri"],
                        LEVEL_NAMES[st["level"]], st["aeb"][0]))
            col = LEVEL_BGR[st["level"]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 3)
            put(frame, f"{d:.2f} m  TTC {fmt_t(st['ttc'])}", (x, max(y - 10, 20)), 0.7, col, 2)
            if "BRAKE" in st["aeb"][0]:
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), LEVEL_BGR[3], 14)
                put(frame, "BRAKE!", (W // 2 - 90, 70), 2.0, LEVEL_BGR[3], 5)
            if st["level"] != last_level or st["aeb"][0] != last_aeb:
                print(f"  t={t:6.2f}s  d={d:5.2f} m  closing={closing:+5.2f} m/s  TTC={fmt_t(st['ttc'])}  "
                      f"risk={LEVEL_NAMES[st['level']]:8s}  -> {st['aeb'][0]}")
                last_level, last_aeb = st["level"], st["aeb"][0]
        else:
            if len(est.buf) and t - est.buf[-1][0] > 0.5:     # object lost for 0.5 s: start afresh
                est.reset(); deb_level.reset(); deb_aeb.reset()
            if mode == "colour" and target is None:
                put(frame, "Click on the object you want to track", (20, 40), 0.8, (255, 255, 255), 2)

        panel = draw_panel(max(H, 620), st, hist, mode, calibrated)
        canvas = np.full((max(H, 620), W + PANEL_W, 3), PANEL, np.uint8)
        canvas[:H, :W] = frame
        canvas[:, W:] = panel
        if args.out:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_file or 20,
                                         (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)

        if not args.headless:
            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("m") and cascade is not None:
                mode = "face" if mode == "colour" else "colour"; est.reset(); prev_box = None
            if k == ord("r"):
                est.reset(); hist.clear(); prev_box = None
            if k == ord("k") and box is not None:
                focal = box[2] * args.calib_dist / real_w; calibrated = True; est.reset()
                print(f"  calibrated: object at {args.calib_dist} m -> focal length {focal:.0f} px")
            if k == ord("s"):
                fn = str(C.REPORTS / f"webcam_screenshot_{int(time.time())}.png")
                cv2.imwrite(fn, canvas); print(f"  screenshot saved: {fn}")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        ok, frame = cap.read()

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    save_session(log)


def save_session(log):
    if len(log) < 5:
        return
    import csv
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = C.REPORTS / f"webcam_session_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "distance_m", "closing_mps", "ttc_s", "mttc_s", "drac_mps2", "cri", "risk_level", "aeb"])
        w.writerows(log)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r[0] for r in log]); d = np.array([r[1] for r in log])
    tm = np.minimum(np.array([r[3] for r in log]), np.array([r[4] for r in log]))
    cri = np.array([r[6] for r in log])
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, d, color="#2a78d6"); ax[0].set_ylabel("distance [m]")
    ax[1].plot(t, np.minimum(tm, 10), color="#2a78d6")
    for thr, c in ((C.AEB["ttc_warn"], "#eda100"), (C.AEB["ttc_partial"], "#eb6834"), (C.AEB["ttc_full"], "#c8312f")):
        ax[1].axhline(thr, color=c, ls="--", lw=1)
    ax[1].set_ylabel("min(TTC, MTTC) [s]"); ax[1].set_ylim(0, 10)
    ax[2].plot(t, cri, color="#eb6834"); ax[2].set_ylabel("risk index"); ax[2].set_ylim(0, 1); ax[2].set_xlabel("time [s]")
    for a in ax:
        a.grid(color="#e4e3de"); a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title("Webcam session: live risk assessment", loc="left", weight="bold")
    fig.tight_layout()
    png = csv_path.with_suffix(".png")
    fig.savefig(png, dpi=130)
    print(f"Session log saved: {csv_path}\nSession plot saved: {png}")


if __name__ == "__main__":
    main()
