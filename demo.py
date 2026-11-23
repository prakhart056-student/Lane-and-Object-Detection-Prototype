"""Interactive visual demo of the risk-aware collision avoidance system.

    python demo.py                      # open the interactive simulator window
    python demo.py --batch 300          # stress-test on 300 brand-new random scenarios per family
    python demo.py --scenario brake --speed 110 --gap 30 --intensity 7.5
    python demo.py --save demo.gif      # render one run to an animated GIF (no window)

In the window:
  * pick a scenario on the right, move the sliders, press  Run  (or Space)
  * Random unseen (or R) generates a scenario the system has never seen
  * Real I-80 scene replays a random clip of real traffic from the unseen
    I-80 freeway, with the real human driver shown in the bottom lane
Three identical leaders drive in three lanes; each lane's ego car is driven by
a different controller, so you can watch which one brakes in time.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from src import collision_avoidance as CA  # noqa: E402
from src import config as C  # noqa: E402

CTRLS = ["ACC (IDM)", "Reactive AEB", "Risk-Aware (ours)"]
HUMAN = "Human driver (real)"
COLORS = {"ACC (IDM)": "#1baf7a", "Reactive AEB": "#eb6834", "Risk-Aware (ours)": "#2a78d6", HUMAN: "#eda100"}
LANE_Y = {"Risk-Aware (ours)": 3.5, "Reactive AEB": 2.5, "ACC (IDM)": 1.5, HUMAN: 0.5}

EB, CI, SG, ST, RW = ("Leader emergency brake", "Aggressive cut-in", "Stop-and-go wave",
                      "Stopped traffic ahead", "Real I-80 scene")
SCENARIOS = [EB, CI, SG, ST, RW]
CLI_NAMES = {"brake": EB, "cutin": CI, "stopgo": SG, "stopped": ST, "real": RW}
# starting (speed km/h, gap m, leader braking m/s²) loaded when a scenario is picked
DEFAULTS = {EB: (100, 32, 7.5), CI: (90, 10, 3.0), SG: (80, 35, 6.0), ST: (90, 90, 0.5)}
T = 250                                   # 25 s at 10 Hz
LEAD_LEN = 6.0                            # drawn car length [m] (schematic, a bit longer than real)
BASE_NOISE = dict(CA.SENSOR_NOISE)
VIEW = 125.0                              # metres of road shown behind the leader


# ---------------------------------------------------------------------------
# Scenario construction (vectorised: every parameter is an array of length n)
# ---------------------------------------------------------------------------
def make_leader(kind, v0, gap, inten):
    n = len(v0)
    t = np.arange(T) * C.DT
    if kind == EB:      # leader cruises, then brakes to a stop at `inten` m/s² after 3 s
        vl = np.maximum(v0[:, None] - inten[:, None] * np.maximum(t - 3.0, 0), 0)
        ego = v0
    elif kind == CI:    # slower car cuts in `gap` m ahead, then brakes at `inten` for 2 s
        vl0 = np.maximum(v0 - 6.0, 0)
        vl = np.maximum(vl0[:, None] - inten[:, None] * np.clip(t - 1.0, 0, 2.0), 0)
        ego = v0
    elif kind == SG:    # traffic wave: leader speed oscillates with amplitude from `inten`
        base = 0.8 * v0
        amp = np.minimum(1.2 * inten, 0.9 * base)
        vl = np.maximum(base[:, None] + amp[:, None] * np.sin(2 * np.pi * t / 12.0), 0)
        ego = base
    elif kind == ST:    # stationary queue `gap` m ahead
        vl = np.zeros((n, T))
        ego = v0
    else:
        raise ValueError(kind)
    xl = gap[:, None] + np.concatenate([np.zeros((n, 1)), np.cumsum(vl[:, :-1] * C.DT, axis=1)], axis=1)
    return dict(lead_x=xl, lead_v=vl, lead_a=np.gradient(vl, C.DT, axis=1),
                ego_v0=np.asarray(ego, float), gap0=np.asarray(gap, float))


def required_decel(kind, v0, gap, inten):
    """Braking an ideal car would need (after latency) to avoid the crash."""
    if kind == EB:
        return CA._required_decel(v0, gap + v0 ** 2 / (2 * np.maximum(inten, 0.1)))
    if kind == ST:
        return CA._required_decel(v0, gap)
    return np.zeros_like(v0)


def random_params(kind, n, rng):
    """Wider ranges than the evaluation in the report, drawn from a fresh seed."""
    if kind == EB:
        v0 = rng.uniform(40, 140, n) / 3.6
        return v0, rng.uniform(0.6, 2.5, n) * v0, rng.uniform(3, 9, n)
    if kind == CI:
        v0 = rng.uniform(40, 140, n) / 3.6
        return v0, rng.uniform(4, 30, n), rng.uniform(0, 4, n)
    if kind == SG:
        v0 = rng.uniform(30, 110, n) / 3.6
        return v0, rng.uniform(0.8, 2.0, n) * 0.8 * v0, rng.uniform(2, 9, n)
    v0 = rng.uniform(40, 130, n) / 3.6
    return v0, rng.uniform(40, 150, n), np.zeros(n)


_I80 = None


def real_scenes():
    """Cache of real I-80 leader–follower clips (built once from the processed data)."""
    global _I80
    if _I80 is not None:
        return _I80
    path = C.DATA_PROCESSED / "i80_scenes.npz"
    if not path.exists():
        print("  (first use) extracting real I-80 clips ...", flush=True)
        from src.preprocess import load
        p = load("pairs_i-80").sort_values(["episode", "frame_id"])
        skip = 20
        sizes = p.groupby("episode").size()
        q = p[p.episode.isin(sizes[sizes >= T + skip].index)]
        q = q[q.groupby("episode").cumcount().between(skip, skip + T - 1)]
        cols = {k: [] for k in ("lead_x", "lead_v", "lead_a", "v", "a", "gap", "ttc")}
        for _, g in q.groupby("episode"):
            cols["lead_x"].append(g.y_l.to_numpy() - g.len_l.to_numpy())
            cols["lead_v"].append(g.v_l.to_numpy()); cols["lead_a"].append(g.a_l.to_numpy())
            cols["v"].append(g.v.to_numpy()); cols["a"].append(g.a.to_numpy())
            cols["gap"].append(g.gap.to_numpy()); cols["ttc"].append(g.ttc.to_numpy())
        np.savez_compressed(path, **{k: np.stack(v) for k, v in cols.items()})
    _I80 = dict(np.load(path))
    return _I80


def real_scene(rng):
    d = real_scenes()
    hard = np.where(d["lead_a"].min(axis=1) < -2.0)[0]
    i = int(rng.choice(hard)) if len(hard) and rng.random() < 0.8 else int(rng.integers(len(d["v"])))
    sc = dict(lead_x=d["lead_x"][i:i + 1], lead_v=d["lead_v"][i:i + 1], lead_a=d["lead_a"][i:i + 1],
              ego_v0=d["v"][i:i + 1, 0].copy(), gap0=d["gap"][i:i + 1, 0].copy())
    sc["human"] = {k: d[k][i] for k in ("gap", "v", "a", "ttc")}
    sc["clip"] = i
    return sc


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
_RISK = None


def risk_model():
    global _RISK
    if _RISK is None:
        _RISK = CA.RiskModel.load()
    return _RISK


def simulate_all(sc, noise=1.0, seed=0):
    for k, v in BASE_NOISE.items():
        CA.SENSOR_NOISE[k] = v * noise
    return {c: CA.simulate(sc["lead_x"], sc["lead_v"], sc["lead_a"], sc["ego_v0"], sc["gap0"],
                           c, risk_model(), seed=seed) for c in CTRLS}


def outcome_lines(res, human=None):
    lines = []
    for c in CTRLS:
        r = res[c]
        if r["collided"][0]:
            lines.append((c, f"CRASH at {r['t_collision'][0]:.1f} s, impact {r['impact_speed'][0] * 3.6:.0f} km/h"))
        else:
            lines.append((c, f"safe: min gap {np.nanmin(r['gap'][0]):.1f} m, "
                             f"min TTC {np.nanmin(r['ttc'][0]):.1f} s"))
    if human is not None:
        lines.append((HUMAN, f"real driver: min gap {np.nanmin(human['gap']):.1f} m, "
                             f"min TTC {np.nanmin(human['ttc']):.1f} s"))
    return lines


# ---------------------------------------------------------------------------
# Interactive window
# ---------------------------------------------------------------------------
class Demo:
    def __init__(self, args, interactive=True):
        import matplotlib
        if not interactive:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button, RadioButtons, Slider
        self.plt, self.Rectangle = plt, Rectangle
        self.rng = np.random.default_rng(int(time.time()))
        self.kind = CLI_NAMES.get(args.scenario, EB)
        self.thr = risk_model().threshold

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        self.fig = fig = plt.figure(figsize=(16, 9), facecolor="#f7f6f2")
        try:
            fig.canvas.manager.set_window_title("Risk-Aware Collision Avoidance: live simulator")
        except Exception:
            pass
        self.title = fig.text(0.03, 0.955, "", fontsize=15, weight="bold", color="#14213d")
        self.subtitle = fig.text(0.03, 0.925, "", fontsize=11, color="#52514e")

        # ---- road
        ax = self.ax_road = fig.add_axes([0.03, 0.56, 0.72, 0.34])
        ax.set_xlim(-VIEW - 8, 10); ax.set_ylim(0, 4)
        ax.set_facecolor("#3b3e44"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        for y in (1, 2, 3):
            ax.plot([-VIEW - 8, 10], [y, y], color="#e8e8e8", lw=1.2, ls=(0, (6, 6)))
        self.lane_label = {}
        for name, y in LANE_Y.items():
            self.lane_label[name] = ax.text(-VIEW - 6, y + 0.32, name, color=COLORS[name], fontsize=10,
                                            weight="bold", va="center")
        self.cars, self.halo, self.brake, self.info = {}, None, {}, {}
        self.lead, self.lead_brake = {}, {}
        for name, y in LANE_Y.items():
            self.lead[name] = ax.add_patch(Rectangle((0, y - 0.22), LEAD_LEN, 0.44, fc="#d9dde3", ec="#14213d", lw=1))
            self.lead_brake[name] = ax.add_patch(Rectangle((-0.35, y - 0.22), 0.35, 0.44, fc="#ff3b30", ec="none"))
            if name == "Risk-Aware (ours)":
                self.halo = ax.add_patch(Rectangle((0, y - 0.36), LEAD_LEN + 2, 0.72, fc="none", ec="none", lw=3))
            self.cars[name] = ax.add_patch(Rectangle((-30, y - 0.22), LEAD_LEN, 0.44, fc=COLORS[name], ec="#14213d", lw=1))
            self.brake[name] = ax.add_patch(Rectangle((-30, y - 0.22), 0.35, 0.44, fc="#ff3b30", ec="none"))
            self.info[name] = ax.text(-VIEW - 6, y - 0.3, "", color="#f0f0f0", fontsize=9, va="center", family="monospace")
        self.crash = {n: ax.text(-60, LANE_Y[n], "", color="#ffffff", fontsize=13, weight="bold", ha="center",
                                 va="center", bbox=dict(fc="#c8312f", ec="none", pad=3)) for n in LANE_Y}
        self.clock = fig.text(0.03, 0.515, "", color="#14213d", fontsize=13, weight="bold", family="monospace")
        self.no_human = ax.text(-60, 0.5, "real human driver appears in  Real I-80 scene", color="#9aa0a8",
                                fontsize=10, ha="center", va="center", style="italic")
        self.risk_txt = fig.text(0.75, 0.515, "", color="#14213d", fontsize=12, ha="right", weight="bold",
                                 bbox=dict(fc="#f7f6f2", ec="none", pad=4))

        # ---- time-series plots
        self.ax_gap = fig.add_axes([0.04, 0.08, 0.21, 0.38])
        self.ax_v = fig.add_axes([0.29, 0.08, 0.21, 0.38])
        self.ax_r = fig.add_axes([0.54, 0.08, 0.21, 0.38])
        for a, ttl, yl in ((self.ax_gap, "Gap to leader", "m"), (self.ax_v, "Speed", "km/h"),
                           (self.ax_r, "Predicted conflict risk (ours)", "P(conflict within 3 s)")):
            a.set_title(ttl, loc="left", fontsize=11, weight="bold"); a.set_ylabel(yl)
            a.set_xlabel("time [s]"); a.grid(color="#e4e3de"); a.set_xlim(0, T * C.DT); a.set_facecolor("#fcfcfb")
        self.lines = {}
        for name in list(CTRLS) + [HUMAN]:
            self.lines[(name, "gap")], = self.ax_gap.plot([], [], color=COLORS[name], lw=2, label=name)
            self.lines[(name, "v")], = self.ax_v.plot([], [], color=COLORS[name], lw=2)
        self.lines[("lead", "v")], = self.ax_v.plot([], [], color="#52514e", lw=1.5, ls="--", label="Leader")
        self.lines[("risk", "p")], = self.ax_r.plot([], [], color=COLORS["Risk-Aware (ours)"], lw=2)
        self.ax_r.axhline(self.thr, color="#52514e", ls="--", lw=1)
        self.ax_r.text(0.3, self.thr + 0.02, "warning threshold", fontsize=8, color="#52514e")
        self.ax_r.set_ylim(0, 1.02)
        self.ax_gap.legend(loc="upper right", fontsize=8, frameon=False)
        self.ax_v.legend(loc="upper right", fontsize=8, frameon=False)
        self.crash_marks = []

        # ---- control panel
        fig.text(0.785, 0.905, "Scenario", fontsize=12, weight="bold", color="#14213d")
        rax = fig.add_axes([0.78, 0.66, 0.20, 0.24], facecolor="#f7f6f2")
        self.radio = RadioButtons(rax, SCENARIOS, active=SCENARIOS.index(self.kind), activecolor="#2a78d6")
        for lbl in self.radio.labels:
            lbl.set_fontsize(10.5)
        self.radio.on_clicked(self._on_radio)

        def slider(y, label, lo, hi, val, fmt):
            sax = fig.add_axes([0.85, y, 0.12, 0.025], facecolor="#e4e3de")
            s = Slider(sax, label, lo, hi, valinit=val, valfmt=fmt, color="#2a78d6")
            s.label.set_fontsize(9.5)
            return s

        self.s_speed = slider(0.60, "Speed (km/h)", 30, 140, args.speed, "%.0f")
        self.s_gap = slider(0.555, "Gap (m)", 3, 150, args.gap, "%.0f")
        self.s_int = slider(0.51, "Leader braking\n(m/s²)", 0.5, 9, args.intensity, "%.1f")
        self.s_noise = slider(0.465, "Sensor noise ×", 0, 3, args.noise, "%.1f")

        self.b_run = Button(fig.add_axes([0.78, 0.375, 0.095, 0.055]), "Run  [Space]", color="#2a78d6", hovercolor="#1c5cab")
        self.b_rand = Button(fig.add_axes([0.885, 0.375, 0.095, 0.055]), "Random unseen [R]", color="#eb6834", hovercolor="#c2491b")
        for b in (self.b_run, self.b_rand):
            b.label.set_color("white"); b.label.set_fontsize(10); b.label.set_weight("bold")
        self.b_run.on_clicked(lambda _e: self.run())
        self.b_rand.on_clicked(lambda _e: self.random_unseen())
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        fig.text(0.785, 0.335, "Outcome", fontsize=12, weight="bold", color="#14213d")
        self.out_txt = {}
        for i, name in enumerate(list(CTRLS) + [HUMAN]):
            fig.text(0.785, 0.30 - i * 0.065, name, fontsize=10, weight="bold", color=COLORS[name])
            self.out_txt[name] = fig.text(0.785, 0.275 - i * 0.065, "", fontsize=9.5, color="#14213d")
        self.k = 0
        self.res = None

    # ---------------- events
    def _on_radio(self, label):
        self.kind = label
        if not getattr(self, "_quiet", False):
            if label in DEFAULTS:                 # sensible starting values for this scenario
                sp, gp, it = DEFAULTS[label]
                self.s_speed.set_val(sp); self.s_gap.set_val(gp); self.s_int.set_val(it)
            self.run()

    def _on_key(self, e):
        if e.key == " ":
            self.run()
        elif e.key in ("r", "R"):
            self.random_unseen()

    def random_unseen(self):
        kind = SCENARIOS[int(self.rng.integers(len(SCENARIOS)))]
        if kind != RW:
            v0, gap, inten = (x[0] for x in random_params(kind, 1, self.rng))
            self.s_speed.set_val(round(v0 * 3.6)); self.s_gap.set_val(round(float(gap)))
            self.s_int.set_val(round(float(inten), 1))
        self.s_noise.set_val(round(float(self.rng.uniform(0.5, 2.0)), 1))
        self._quiet = True
        self.radio.set_active(SCENARIOS.index(kind))
        self._quiet = False
        self.kind = kind
        self.run(tag="random unseen")

    # ---------------- run one scenario
    def run(self, tag=""):
        seed = int(self.rng.integers(1_000_000))
        noise = float(self.s_noise.val)
        if self.kind == RW:
            sc = real_scene(self.rng)
            desc = (f"real I-80 clip #{sc['clip']}  ·  start speed {sc['ego_v0'][0] * 3.6:.0f} km/h  ·  "
                    f"gap {sc['gap0'][0]:.0f} m  ·  noise ×{noise:.1f}")
        else:
            v0 = np.array([self.s_speed.val / 3.6]); gap = np.array([float(self.s_gap.val)])
            inten = np.array([float(self.s_int.val)])
            sc = make_leader(self.kind, v0, gap, inten)
            need = float(required_decel(self.kind, v0, gap, inten)[0])
            desc = (f"{self.s_speed.val:.0f} km/h  ·  gap {gap[0]:.0f} m  ·  leader braking {inten[0]:.1f} m/s²  ·  "
                    f"noise ×{noise:.1f}" + (f"  ·  braking needed {need:.1f} m/s²"
                                             + ("  (beyond the 8 m/s² limit: unavoidable)" if need > 8 else "")
                                             if self.kind in (EB, ST) else ""))
        self.sc, self.res = sc, simulate_all(sc, noise, seed)
        self.human = sc.get("human")
        self.title.set_text(f"{self.kind}" + (f"   [{tag}]" if tag else ""))
        self.subtitle.set_text(desc + f"  ·  seed {seed}")
        lines = outcome_lines(self.res, self.human)
        print(f"\n> {self.kind}{' [' + tag + ']' if tag else ''}: {desc}")
        for name, txt in lines:
            print(f"    {name:20s} {txt}")
        for name in self.out_txt:
            self.out_txt[name].set_text("")
        for name, txt in lines:
            self.out_txt[name].set_text(txt)
            self.out_txt[name].set_color("#c8312f" if txt.startswith("CRASH") else "#14213d")
        # rescale plots
        gmax = np.nanmax([np.nanmax(self.res[c]["gap"]) for c in CTRLS] +
                         ([np.nanmax(self.human["gap"])] if self.human is not None else []))
        vmax = max(np.nanmax(sc["lead_v"]), np.nanmax([np.nanmax(self.res[c]["v"]) for c in CTRLS])) * 3.6
        self.ax_gap.set_ylim(0, gmax * 1.08 + 1); self.ax_v.set_ylim(0, vmax * 1.1 + 5)
        for m in self.crash_marks:
            m.remove()
        self.crash_marks = []
        self.k = 0
        self.draw_frame(0)

    # ---------------- one animation frame
    def draw_frame(self, k):
        if self.res is None:
            return
        t = np.arange(k + 1) * C.DT
        sc = self.sc
        al = sc["lead_a"][0, k]
        for name, y in LANE_Y.items():
            is_h = name == HUMAN
            show = (not is_h) or self.human is not None
            self.no_human.set_visible(self.human is None)
            for p in (self.lead[name], self.cars[name]):
                p.set_visible(show)
            self.lead_brake[name].set_visible(show and al < -1.0)
            self.lane_label[name].set_visible(True)
            if not show:
                self.brake[name].set_visible(False); self.info[name].set_text(""); self.crash[name].set_text("")
                self.lines[(name, "gap")].set_data([], []); self.lines[(name, "v")].set_data([], [])
                continue
            if is_h:
                gap_s, v_s, a_s, ttc_s = (self.human[q] for q in ("gap", "v", "a", "ttc"))
                crashed, tc = False, np.nan
            else:
                r = self.res[name]
                gap_s, v_s, a_s, ttc_s = r["gap"][0], r["v"][0], r["a"][0], r["ttc"][0]
                tc = r["t_collision"][0]
                crashed = (not np.isnan(tc)) and k * C.DT >= tc - 1e-9
            valid = ~np.isnan(gap_s[: k + 1])
            j = int(np.where(valid)[0][-1]) if valid.any() else 0
            g = 0.0 if crashed else gap_s[j]
            x = -min(max(g, -0.3), VIEW) - LEAD_LEN
            self.cars[name].set_x(x)
            self.brake[name].set_x(x - 0.35 + 0.0)
            self.brake[name].set_visible(a_s[j] < -1.5 and not crashed)
            far = f"  ◀ {g:.0f} m" if g > VIEW else ""
            ttc_txt = "∞" if ttc_s[j] >= C.TTC_CAP - 1e-6 else f"{ttc_s[j]:.1f} s"
            self.info[name].set_text(f"{v_s[j] * 3.6:5.0f} km/h   gap {g:5.1f} m   TTC {ttc_txt}{far}")
            self.crash[name].set_text("COLLISION" if crashed else "")
            self.crash[name].set_position((-14, y))
            if crashed and not any(getattr(m, "_who", None) == name for m in self.crash_marks):
                m, = self.ax_gap.plot([tc], [0], "X", color=COLORS[name], ms=12, mec="white")
                m._who = name
                self.crash_marks.append(m)
            self.lines[(name, "gap")].set_data(t, gap_s[: k + 1])
            self.lines[(name, "v")].set_data(t, v_s[: k + 1] * 3.6)
        self.lines[("lead", "v")].set_data(t, sc["lead_v"][0, : k + 1] * 3.6)
        p = self.res["Risk-Aware (ours)"]["risk"][0]
        self.lines[("risk", "p")].set_data(t, p[: k + 1])
        valid = ~np.isnan(p[: k + 1])
        pk = p[: k + 1][valid][-1] if valid.any() else 0.0
        level, col = (("LOW", "none") if pk < self.thr else ("ELEVATED", "#eda100") if pk < 0.5
                      else ("HIGH", "#eb6834") if pk < 0.8 else ("CRITICAL", "#ff3b30"))
        car = self.cars["Risk-Aware (ours)"]
        self.halo.set_x(car.get_x() - 1.0); self.halo.set_edgecolor(col)
        self.risk_txt.set_text(f"Risk-aware car: predicted risk {pk:.2f}  ({level})")
        self.risk_txt.set_color("#14213d" if col == "none" else ("#c2491b" if level == "HIGH" else
                                                                 "#c8312f" if level == "CRITICAL" else "#9a6a00"))
        self.clock.set_text(f"t = {k * C.DT:4.1f} s")

    def _tick(self, _frame):
        if self.res is not None and self.k < T - 1:
            self.k += 1
            self.draw_frame(self.k)
        return []

    def show(self):
        from matplotlib.animation import FuncAnimation
        self.run()
        self.anim = FuncAnimation(self.fig, self._tick, interval=50, cache_frame_data=False)
        self.plt.show()

    def save_gif(self, path, step=2, fps=12, dpi=72):
        from matplotlib.animation import FuncAnimation, PillowWriter
        self.run()
        frames = list(range(0, T, step))
        anim = FuncAnimation(self.fig, lambda i: (self.draw_frame(i), [])[1], frames=frames)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        print(f"saved {path}")


# ---------------------------------------------------------------------------
# Batch stress test on unseen scenarios
# ---------------------------------------------------------------------------
def batch(n, seed, noise, show=True):
    rng = np.random.default_rng(seed)
    print("=" * 78)
    print(f" UNSEEN-SCENARIO STRESS TEST   {n} new random scenarios per family   seed {seed}")
    print(f" parameter ranges are wider than in the report; sensor noise x{noise}")
    print("=" * 78)
    rows = []
    for kind in (EB, CI, SG, ST):
        v0, gap, inten = random_params(kind, n, rng)
        sc = make_leader(kind, v0, gap, inten)
        need = required_decel(kind, v0, gap, inten)
        avoid = need <= -C.VEHICLE_A_MIN
        res = simulate_all(sc, noise, seed)
        row = {"scenario": kind, "avoidable": avoid.mean()}
        for c in CTRLS:
            df = CA.summarise(res[c])
            row[c] = df.collision.mean()
            row[c + " (avoidable)"] = df.collision[avoid].mean() if avoid.any() else np.nan
            row[c + " minTTC"] = df.min_ttc.median()
        rows.append(row)
        print(f"\n {kind}   ({avoid.mean() * 100:.0f}% of runs physically avoidable)")
        for c in CTRLS:
            print(f"   {c:20s} collisions {row[c] * 100:5.1f}%   on avoidable runs {row[c + ' (avoidable)'] * 100:5.1f}%"
                  f"   median min-TTC {row[c + ' minTTC']:5.2f} s")
    _batch_chart(rows, n, seed, show)
    return rows


def _batch_chart(rows, n, seed, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor="#fcfcfb")
    x = np.arange(len(rows)); w = 0.26
    for i, c in enumerate(CTRLS):
        vals = [r[c + " (avoidable)"] * 100 for r in rows]
        bars = ax.bar(x + (i - 1) * (w + 0.02), vals, w, color=COLORS[c], label=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([r["scenario"] for r in rows])
    ax.set_ylabel("Collision rate on avoidable runs [%]")
    ax.set_title(f"Unseen-scenario stress test: {n} new random scenarios per family (seed {seed})",
                 loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e4e3de"); ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = C.FIGURES / "unseen_stress_test.png"
    fig.savefig(out, dpi=150)
    print(f"\n chart saved to {out}")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(CLI_NAMES), default="brake")
    ap.add_argument("--speed", type=float, default=100, help="ego speed [km/h]")
    ap.add_argument("--gap", type=float, default=32, help="initial gap [m]")
    ap.add_argument("--intensity", type=float, default=7.5, help="leader braking [m/s²]")
    ap.add_argument("--noise", type=float, default=1.0, help="sensor-noise multiplier")
    ap.add_argument("--batch", type=int, default=0, help="run N unseen random scenarios per family")
    ap.add_argument("--seed", type=int, default=None, help="seed for --batch (default: current time)")
    ap.add_argument("--save", default=None, help="render one run to a GIF instead of opening a window")
    ap.add_argument("--no-show", action="store_true", help="with --batch: save the chart without opening it")
    args = ap.parse_args()

    if not (C.MODELS / "risk_model.joblib").exists():
        sys.exit("No trained model found. Run  python run_pipeline.py  first.")
    if args.batch:
        batch(args.batch, args.seed if args.seed is not None else int(time.time()) % 100000, args.noise,
              show=not args.no_show)
    elif args.save:
        Demo(args, interactive=False).save_gif(args.save)
    else:
        print("Opening the simulator window ...  (Space = run, R = random unseen scenario, close window to exit)")
        Demo(args).show()


if __name__ == "__main__":
    main()
