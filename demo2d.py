"""Top-down visual demo of the Level-2 autonomous driving stack.

    python demo2d.py                         # interactive window
    python demo2d.py --scenario urban_route  # start with a given scenario
    python demo2d.py --save drive.gif --scenario highway_overtake
    python demo2d.py --evaluate 30           # batch test on unseen random variants

The car steers, keeps its lane, follows the road through curves and turns, decides
on its own to overtake (risk-checked lane changes), stops at red lights, yields to
pedestrians and emergency-brakes. The right-hand panel shows what it is doing and why.

Window controls:  scenario list | Run [Space] | Random unseen [R] |
                  Controller [C] (risk-aware <-> reactive baseline) | playback speed slider
"""
from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from src import config as C  # noqa: E402
from src.av.scenarios import SCENARIOS  # noqa: E402
from src.av.vehicle import box_corners  # noqa: E402

NAMES = {"highway_overtake": "Highway: overtake", "highway_cut_in": "Highway: cut-in",
         "highway_emergency": "Highway: sudden stop", "urban_route": "Urban: light + pedestrian",
         "rc_track": "RC track (1:10 scale)"}
VIEW = {"highway_overtake": 42, "highway_cut_in": 42, "highway_emergency": 42, "urban_route": 40, "rc_track": 2.6}
MODE_COL = {"CRUISE": "#1baf7a", "FOLLOW": "#2a78d6", "LANE CHANGE": "#6250d6", "STOP": "#c98500",
            "YIELD": "#c98500", "COLLISION WARNING": "#c98500", "AEB": "#eb6834", "EMERGENCY": "#c8312f",
            "STOPPED": "#646b78"}
LIGHT_COL = {"green": "#1baf7a", "yellow": "#eda100", "red": "#e34948"}
BG, ROAD, INK = "#e6e9e1", "#4a4d52", "#14213d"


def mode_color(mode):
    for k, c in MODE_COL.items():
        if mode.startswith(k):
            return c
    return INK


# ---------------------------------------------------------------------------
# Simulation -> recorded frames
# ---------------------------------------------------------------------------
_RM = None


def risk_model():
    global _RM
    if _RM is None:
        from src.collision_avoidance import RiskModel
        _RM = RiskModel.load()
    return _RM


def record(world):
    road = world.road
    frames = []
    scale = world.params.wheelbase / 2.8
    while not world.done:
        world.step()
        e, pl = world.ego, world.planner
        s = world.ego_frenet["s"]
        ss = s + np.linspace(0, max(40 * scale, e.v * 4), 40)
        rx, ry, _ = road.center.frenet_to_xy(ss, pl.ref.d(ss, road.center.length, road.center.closed))
        cx, cy = e.center
        frames.append(dict(
            t=world.t, ego=(cx, cy, e.yaw), v=e.v, a=e.a, delta=e.delta, info=dict(world.info),
            cars=[(world.car_pose(c), c.length, c.width, c.a < -1.0, c.kind) for c in world.cars],
            peds=[world.ped_xy(p) for p in world.peds if not p.done and (p.active or p.t_start is None)],
            lights=[tl.state(world.t) for tl in road.lights], ref=(rx, ry), target=world.target_pt,
            events=list(pl.events[-7:]), collision=world.collision, lane=pl.lane))
    return frames, world.metrics()


def build_world(name, rng, randomize, risk_aware):
    return SCENARIOS[name](rng, randomize=randomize, risk_model=risk_model(), risk_aware=risk_aware)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
class Demo2D:
    def __init__(self, scenario="highway_overtake", interactive=True):
        import matplotlib
        if not interactive:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider
        self.plt = plt
        self.name, self.risk_aware = scenario, True
        self.rng = np.random.default_rng(int(time.time()))
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
        self.fig = fig = plt.figure(figsize=(16, 9), facecolor="#f7f6f2")
        try:
            fig.canvas.manager.set_window_title("Level-2 autonomous driving: live simulator")
        except Exception:
            pass
        self.ax = fig.add_axes([0.01, 0.02, 0.64, 0.92])
        self.title = fig.text(0.012, 0.955, "", fontsize=15, weight="bold", color=INK)

        # ---- right panel
        fig.text(0.665, 0.955, "Scenario", fontsize=12, weight="bold", color=INK)
        rax = fig.add_axes([0.66, 0.76, 0.16, 0.19], facecolor="#f7f6f2")
        keys = list(SCENARIOS)
        self.radio = RadioButtons(rax, [NAMES[k] for k in keys], active=keys.index(scenario), activecolor="#2a78d6")
        for lbl in self.radio.labels:
            lbl.set_fontsize(9.5)
        self.radio.on_clicked(lambda lbl: self._pick({v: k for k, v in NAMES.items()}[lbl]))
        self.b_run = Button(fig.add_axes([0.83, 0.895, 0.155, 0.045]), "Run  [Space]", color="#2a78d6", hovercolor="#1c5cab")
        self.b_rand = Button(fig.add_axes([0.83, 0.843, 0.155, 0.045]), "Random unseen  [R]", color="#eb6834", hovercolor="#c2491b")
        self.b_ctrl = Button(fig.add_axes([0.83, 0.791, 0.155, 0.045]), "", color="#646b78", hovercolor="#4a4d52")
        for b in (self.b_run, self.b_rand, self.b_ctrl):
            b.label.set_color("white"); b.label.set_weight("bold"); b.label.set_fontsize(9.5)
        self.b_run.on_clicked(lambda _e: self.start(False))
        self.b_rand.on_clicked(lambda _e: self.start(True))
        self.b_ctrl.on_clicked(lambda _e: self.toggle_ctrl())
        sax = fig.add_axes([0.73, 0.745, 0.2, 0.02], facecolor="#e4e3de")
        self.speed = Slider(sax, "Playback ×", 1, 8, valinit=2, valstep=1, color="#2a78d6")
        fig.canvas.mpl_connect("key_press_event", self._key)

        self.mode_txt = fig.text(0.665, 0.695, "", fontsize=15, weight="bold", color="white",
                                 bbox=dict(fc=INK, ec="none", pad=6))
        self.hud = fig.text(0.665, 0.505, "", fontsize=10.5, family="monospace", color=INK, va="bottom")
        fig.text(0.665, 0.475, "Decision log", fontsize=11, weight="bold", color=INK)
        self.log_txt = fig.text(0.665, 0.455, "", fontsize=9, color=INK, va="top")
        self.ax_v = fig.add_axes([0.69, 0.045, 0.13, 0.12])
        self.ax_r = fig.add_axes([0.855, 0.045, 0.13, 0.12])
        for a_, ttl in ((self.ax_v, "Speed [km/h]"), (self.ax_r, "Predicted risk")):
            a_.set_title(ttl, fontsize=9.5, loc="left"); a_.tick_params(labelsize=8)
            a_.spines[["top", "right"]].set_visible(False); a_.grid(color="#e4e3de")
        self.l_v, = self.ax_v.plot([], [], color="#2a78d6", lw=1.8)
        self.l_r, = self.ax_r.plot([], [], color="#eb6834", lw=1.8)
        self.ax_r.set_ylim(0, 1.02)
        self.summary = fig.text(0.665, 0.275, "", fontsize=9.5, color=INK, va="top", weight="bold")
        self.frames, self.k, self.static = [], 0, []
        self.view_override = None
        self._update_ctrl_label()

    # ------------------------------------------------------------------ events
    def _key(self, e):
        if e.key == " ":
            self.start(False)
        elif e.key in ("r", "R"):
            self.start(True)
        elif e.key in ("c", "C"):
            self.toggle_ctrl()

    def _pick(self, name):
        self.name = name
        self.start(False)

    def toggle_ctrl(self):
        self.risk_aware = not self.risk_aware
        self._update_ctrl_label()
        self.start(False)

    def _update_ctrl_label(self):
        self.b_ctrl.label.set_text("Controller: RISK-AWARE  [C]" if self.risk_aware else "Controller: REACTIVE  [C]")
        self.b_ctrl.color = "#2a78d6" if self.risk_aware else "#646b78"

    # ------------------------------------------------------------------ run
    def start(self, randomize):
        seed = int(self.rng.integers(1_000_000)) if randomize else 0
        world = build_world(self.name, np.random.default_rng(seed), randomize, self.risk_aware)
        t0 = time.time()
        self.frames, metrics = record(world)
        self.world, self.metrics = world, metrics
        tag = f"random unseen variant, seed {seed}" if randomize else "canonical scenario"
        ctrl = "risk-aware" if self.risk_aware else "reactive baseline"
        self.title.set_text(f"{world.name}   ·   {ctrl}   ·   {tag}")
        print(f"\n> {world.name} [{ctrl}, {tag}]  simulated {metrics['duration_s']:.0f} s in {time.time() - t0:.1f} s")
        print("   " + ("COLLISION with " + metrics["collision_with"] if metrics["collision"] else "no collision") +
              f" | lane changes {metrics['lane_changes']} | red-light violations {metrics['red_light_violations']}"
              f" | lane-keeping RMS {metrics['rms_lateral_error_m'] * 100:.0f} cm | mean {metrics['mean_speed_kmh']:.0f} km/h")
        for t, msg in world.planner.events:
            print(f"     t={t:5.1f}s  {msg}")
        self._draw_static()
        self.k = 0
        self.ax_v.set_xlim(0, self.frames[-1]["t"]); self.ax_r.set_xlim(0, self.frames[-1]["t"])
        vmax = max(f["v"] for f in self.frames) * 3.6
        self.ax_v.set_ylim(0, vmax * 1.15 + 1)
        m = metrics
        self.summary.set_text(
            ("RESULT: COLLISION (" + m["collision_with"] + ")" if m["collision"] else "RESULT: no collision") +
            f"\nlane changes {m['lane_changes']} (aborted {m['lc_aborts']}, rejected {m['lc_rejected']})"
            f"   red-light violations {m['red_light_violations']}"
            f"\nlane-keeping RMS error {m['rms_lateral_error_m'] * 100:.1f} cm   max lateral accel {m['max_lat_accel']:.1f} m/s²")
        self.summary.set_color("#c8312f" if m["collision"] else INK)
        self.draw(0)

    # ------------------------------------------------------------------ drawing
    def _draw_static(self):
        from matplotlib.patches import Circle, Polygon
        ax, road = self.ax, self.world.road
        ax.clear(); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        step = 0.02 if road.lane_width < 1 else 1.0
        hw = road.half_width
        xl, yl = road.boundary_xy(hw, step)
        xr, yr = road.boundary_xy(-hw, step)
        if road.center.closed:
            area = lambda x, y: 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # noqa: E731
            outer, inner = ((xl, yl), (xr, yr)) if area(xl, yl) > area(xr, yr) else ((xr, yr), (xl, yl))
            ax.fill(*outer, color=ROAD, zorder=1); ax.fill(*inner, color=BG, zorder=1)
        else:
            ax.fill(np.r_[xl, xr[::-1]], np.r_[yl, yr[::-1]], color=ROAD, zorder=1)
        lw_line = 1.6 if road.lane_width > 1 else 2.5
        for dd in (-hw, hw):
            ax.plot(*road.boundary_xy(dd, step), color="#f2f2f2", lw=lw_line, zorder=2)
        for k in range(1, road.n_lanes):
            ax.plot(*road.boundary_xy(-hw + k * road.lane_width, step), color="#f2f2f2", lw=lw_line * 0.8,
                    ls=(0, (6, 9)), zorder=2)
        self.light_patches = []
        for tl in road.lights:
            x0, y0, _ = road.center.frenet_to_xy(tl.s, -hw); x1, y1, _ = road.center.frenet_to_xy(tl.s, hw)
            ax.plot([x0, x1], [y0, y1], color="white", lw=4 if road.lane_width > 1 else 3, zorder=2)
            lx, ly, _ = road.center.frenet_to_xy(tl.s, -hw - road.lane_width * 0.45)
            c = Circle((lx, ly), road.lane_width * 0.35, fc="#1baf7a", ec=INK, lw=1.2, zorder=6)
            ax.add_patch(c); self.light_patches.append(c)
        for cw in road.crosswalks:
            for dd in np.arange(-hw + 0.3, hw, 0.9):
                pts = np.array([road.center.frenet_to_xy(s_, d_)[:2] for s_, d_ in
                                ((cw - 2, dd), (cw + 2, dd), (cw + 2, dd + 0.5), (cw - 2, dd + 0.5))])
                ax.add_patch(Polygon(pts, closed=True, fc="#f2f2f2", ec="none", zorder=2))
        self.ref_line, = ax.plot([], [], color="#7CFC9A", lw=2.2, zorder=4, alpha=0.9)
        self.tgt_pt, = ax.plot([], [], "o", color="#7CFC9A", ms=5, zorder=4)
        self.car_patches = [Polygon(np.zeros((4, 2)), closed=True, fc="#9aa0a8", ec=INK, lw=0.8, zorder=5)
                            for _ in self.frames[0]["cars"]]
        for p in self.car_patches:
            ax.add_patch(p)
        self.ego_patch = Polygon(np.zeros((4, 2)), closed=True, fc="#2a78d6", ec="white", lw=1.5, zorder=7)
        ax.add_patch(self.ego_patch)
        self.ped_dots = ax.scatter([], [], s=70, c="#eb6834", edgecolors="white", zorder=8)
        self.crash_txt = ax.text(0.5, 0.9, "", transform=ax.transAxes, ha="center", fontsize=20, weight="bold",
                                 color="white", bbox=dict(fc="#c8312f", ec="none", pad=6), zorder=10)
        self.crash_txt.set_visible(False)
        self.clock = ax.text(0.015, 0.975, "", transform=ax.transAxes, fontsize=12, family="monospace",
                             color=INK, va="top", bbox=dict(fc="#f7f6f2", ec="none", pad=3), zorder=10)

    def draw(self, k):
        if not self.frames:
            return
        f = self.frames[k]
        p = self.world.params
        cx, cy, yaw = f["ego"]
        self.ego_patch.set_xy(box_corners(cx, cy, yaw, p.length, p.width))
        for patch, ((x, y, h), L, Wd, braking, kind) in zip(self.car_patches, f["cars"]):
            patch.set_xy(box_corners(x, y, h, L, Wd))
            patch.set_facecolor("#6b7078" if kind == "truck" else "#9aa0a8")
            patch.set_edgecolor("#ff3b30" if braking else INK); patch.set_linewidth(2.5 if braking else 0.8)
        self.ped_dots.set_offsets(np.array(f["peds"]) if f["peds"] else np.empty((0, 2)))
        for c, st in zip(self.light_patches, f["lights"]):
            c.set_facecolor(LIGHT_COL[st])
        self.ref_line.set_data(*f["ref"]); self.tgt_pt.set_data([f["target"][0]], [f["target"][1]])
        view = self.view_override or VIEW.get(self.name, 50)
        bb = self.ax.get_position(original=True); W_, H_ = self.fig.get_size_inches()
        hv = view * (bb.height * H_) / (bb.width * W_)
        ahead = 0.3 * min(view, hv)                    # look ahead of the car: what it is reacting to
        mx, my = cx + ahead * np.cos(yaw), cy + ahead * np.sin(yaw)
        self.ax.set_xlim(mx - view, mx + view); self.ax.set_ylim(my - hv, my + hv)
        crashed = f["collision"] is not None
        self.crash_txt.set_visible(crashed)
        if crashed:
            self.crash_txt.set_text(f"COLLISION with {f['collision']['with_']}")
        self.clock.set_text(f"t = {f['t']:5.1f} s")

        info = f["info"]
        mode = info["mode"]
        self.mode_txt.set_text(f" {mode} ")
        self.mode_txt.get_bbox_patch().set_facecolor(mode_color(mode))
        lead = info.get("leader")
        ttc = "—" if info["ttc"] >= C.TTC_CAP - 1e-6 else f"{info['ttc']:.1f} s"
        unit = 1.0 if p.wheelbase > 1 else 100.0
        self.hud.set_text(
            f"speed      {f['v'] * 3.6:6.1f} km/h   (target {info['v0'] * 3.6:5.1f})\n"
            f"accel      {f['a']:+6.2f} m/s²\n"
            f"steering   {np.degrees(f['delta']):+6.1f}°\n"
            f"lane       {f['lane'] + 1} of {self.world.road.n_lanes}\n"
            f"leader     " + (f"{lead.gap * unit:6.1f} {'m' if unit == 1 else 'cm'} ahead" if lead is not None else "none") + "\n"
            f"TTC        {ttc}\n"
            f"risk p     {info['risk']:.2f}   AEB stage {info['stage']}")
        self.log_txt.set_text("\n".join(f"{t:5.1f}s  {m}" for t, m in f["events"]) or "—")
        ts = [g["t"] for g in self.frames[:k + 1]]
        self.l_v.set_data(ts, [g["v"] * 3.6 for g in self.frames[:k + 1]])
        self.l_r.set_data(ts, [g["info"]["risk"] for g in self.frames[:k + 1]])

    def _tick(self, _):
        if self.frames and self.k < len(self.frames) - 1:
            self.k = min(self.k + int(self.speed.val), len(self.frames) - 1)
            self.draw(self.k)
        return []

    def show(self):
        from matplotlib.animation import FuncAnimation
        self.start(False)
        self.anim = FuncAnimation(self.fig, self._tick, interval=60, cache_frame_data=False)
        self.plt.show()

    def save_gif(self, path, step=3, fps=10, dpi=70):
        from matplotlib.animation import FuncAnimation, PillowWriter
        self.start(False)
        idx = list(range(0, len(self.frames), step))
        anim = FuncAnimation(self.fig, lambda i: (self.draw(i), [])[1], frames=idx)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="highway_overtake")
    ap.add_argument("--save", default=None, help="render the scenario to a GIF instead of opening a window")
    ap.add_argument("--evaluate", type=int, default=0, help="batch-test N random variants of every scenario")
    args = ap.parse_args()
    if args.evaluate:
        from src.av.evaluate import run
        out = run(args.evaluate)
        print(f"\nresults: reports/av_results.json   chart: reports/figures/fig13_av_evaluation.png  "
              f"({out['runtime_s']:.0f} s)")
    elif args.save:
        Demo2D(args.scenario, interactive=False).save_gif(args.save)
    else:
        print("Opening the simulator window ...  (Space = run, R = random unseen, C = controller, close to exit)")
        Demo2D(args.scenario).show()


if __name__ == "__main__":
    main()


def save_report_figure(path=None):
    """Three snapshots for the report: overtake, pedestrian yield, emergency brake."""
    import io

    import matplotlib.pyplot as plt
    from PIL import Image
    path = path or (C.FIGURES / "fig14_av_scenarios.png")
    d = Demo2D("highway_overtake", interactive=False)
    d.ax.set_position([0.01, 0.055, 0.5, 0.89])          # square map for the snapshots
    shots = []
    for name, want, pick, view, title in (
            ("highway_overtake", "LANE CHANGE", 8, 28, "Overtaking: risk-checked lane change"),
            ("urban_route", "YIELD", 22, 24, "Urban: yielding to a pedestrian"),
            ("highway_emergency", "EMERGENCY", 4, 24, "Sudden stop ahead: emergency braking")):
        d.name, d.view_override = name, view
        d.start(False)
        ks = [i for i, f in enumerate(d.frames) if f["info"]["mode"].startswith(want)]
        k = ks[min(len(ks) - 1, pick)] if ks else len(d.frames) // 2
        d.draw(k)
        ext = d.ax.get_window_extent().transformed(d.fig.dpi_scale_trans.inverted())
        buf = io.BytesIO()
        d.fig.savefig(buf, format="png", bbox_inches=ext, dpi=90)
        f = d.frames[k]
        shots.append((Image.open(buf).copy(), title,
                      f"t = {f['t']:.1f} s   state: {f['info']['mode']}   speed {f['v'] * 3.6:.0f} km/h"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.9), facecolor="#fcfcfb")
    for ax, (img, title, sub) in zip(axes, shots):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title + "\n" + sub, loc="left", fontsize=10.5, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close("all")
    print(f"saved {path}")
