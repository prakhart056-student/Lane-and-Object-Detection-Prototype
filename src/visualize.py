"""Publication-quality figures for the report (saved to reports/figures/)."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from . import config as C  # noqa: E402
from .features import FEATURE_LABELS  # noqa: E402
from .preprocess import load  # noqa: E402

# ---------------------------------------------------------------------------
# Palette (validated categorical order; colour follows the entity everywhere)
# ---------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
STATUS = {"Caution": "#eda100", "Danger": "#eb6834", "Critical": "#c8312f"}

CTRL_COLOR = {"Risk-Aware (ours)": S1, "Reactive AEB": S2, "ACC (IDM)": S3, "Human (replay)": S4}
CTRL_ORDER = ["ACC (IDM)", "Reactive AEB", "Risk-Aware (ours)", "Human (replay)"]
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "text.color": INK, "axes.axisbelow": True,
})


def _save(fig, name):
    path = C.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# 1. System architecture
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 110); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc="white", ec=color, lw=1.8))
        ax.add_patch(FancyBboxPatch((x, y + h - 4.2), w, 4.2, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=color, ec=color, lw=0))
        ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="center", color="white",
                fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + (h - 4.2) / 2, body, ha="center", va="center", color=INK,
                fontsize=8.2, linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     color=INK2, lw=1.4))

    box(1, 25, 19, 18, "1  Perception", "NGSIM trajectories\n(10 Hz video tracking)\nLeader gap, speed,\nacceleration\n+ sensor noise", "#52514e")
    box(24, 25, 19, 18, "2  Preprocessing", "ft → m conversion\nSavitzky–Golay\nsmoothing\nLeader–follower\npairing", "#52514e")
    box(47, 25, 19, 18, "3  Risk Assessment", "TTC · THW · DRAC\nMTTC · CPI\nComposite Risk\nIndex (CRI)\n4 risk levels", S2)
    box(70, 25, 19, 18, "4  Risk Prediction", "ML classifier\n(17 onboard features)\nP(conflict in 3 s)\nTrained US-101\nTested I-80", S1)
    box(70, 2, 19, 18, "5  Decision Layer", "Risk-adaptive\ntime gap\nPre-emptive braking\nMTTC-staged AEB", S1)
    box(47, 2, 19, 18, "6  Actuation", "0.2 s latency\nACC ≤ 3.5 m/s²\nAEB ≤ 8 m/s²\nVehicle dynamics", "#52514e")
    box(24, 2, 19, 18, "7  Evaluation", "Collision rate\nmin-TTC, TET, TIT\nComfort (jerk)\nEfficiency (speed)", S3)
    box(92, 13, 16.5, 18, "Scenarios", "Emergency brake\nCut-in\nStop-and-go\nStopped traffic\nReal I-80 replay", "#8a8984")

    arrow(20.8, 34, 23.4, 34); arrow(43.8, 34, 46.4, 34); arrow(66.8, 34, 69.4, 34)
    arrow(79.5, 24.4, 79.5, 20.8); arrow(69.4, 11, 66.8, 11); arrow(46.4, 11, 43.8, 11)
    arrow(91.6, 22, 89.6, 22)
    ax.add_patch(FancyArrowPatch((56.5, 20.8), (56.5, 24.4), arrowstyle="-|>", mutation_scale=14,
                                 color=MUTED, lw=1.2, linestyle="--"))
    ax.text(57.5, 22.6, "closed loop", fontsize=7.5, color=MUTED, va="center")
    ax.set_title("System architecture: risk-aware collision avoidance pipeline", loc="left")
    _save(fig, "fig01_architecture.png")


# ---------------------------------------------------------------------------
# 2-4. Dataset exploration
# ---------------------------------------------------------------------------
def fig_time_space(site="us-101", lane=2):
    traj = load(f"traj_{site}")
    pairs = load(f"pairs_{site}")
    d = traj[traj.lane_id == lane]
    tmax = min(d.t.max(), 420)
    d = d[d.t <= tmax]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    ax = axes[0]
    sc = ax.scatter(d.t, d.y, c=d.v * 3.6, s=0.35, cmap=BLUES, vmin=0, vmax=80, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label("Speed [km/h]"); cb.outline.set_visible(False)
    ax.set_title(f"Time–space diagram, {site.upper()} lane {lane}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Longitudinal position [m]")

    ax = axes[1]
    ax.scatter(d.t, d.y, c="#d6d5cf", s=0.25, rasterized=True)
    p = pairs[(pairs.lane_id == lane) & (pairs.t <= tmax)]
    for lvl, name in [(1, "Caution"), (2, "Danger"), (3, "Critical")]:
        q = p[p.risk_level == lvl]
        ax.scatter(q.t, q.y, s=5 if lvl < 3 else 9, color=STATUS[name], label=f"{name} (n={len(q):,})",
                   rasterized=True, edgecolors="white" if lvl == 3 else "none", linewidths=0.3)
    ax.legend(loc="upper left", markerscale=2.5, fontsize=8)
    ax.set_title("Detected risk states (Composite Risk Index)")
    ax.set_xlabel("Time [s]")
    _save(fig, "fig02_time_space_risk.png")


def fig_ssm_distributions():
    p = pd.concat([load(f"pairs_{s}").assign(site=s.upper()) for s in C.SITES])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    spec = [("ttc", "Time-to-Collision [s]", np.linspace(0, 19.9, 60), C.TTC_CRITICAL, "TTC < 3 s"),
            ("thw", "Time headway [s]", np.linspace(0, 6, 60), C.THW_CRITICAL, "THW < 1 s"),
            ("drac", "DRAC [m/s²] (log)", np.logspace(-3, 1.3, 60), C.DRAC_CRITICAL, "DRAC > 3.35"),
            ("cri", "Composite Risk Index", np.linspace(0, 1, 50), 0.5, "Danger ≥ 0.5")]
    for ax, (col, lab, bins, thr, tl) in zip(axes, spec):
        for site, colr in [("US-101", S1), ("I-80", S2)]:
            x = p.loc[p.site == site, col]
            if col == "ttc":
                x = x[x < C.TTC_CAP]
            if col == "drac":
                x = x[x > 1e-3]
            ax.hist(x, bins=bins, density=True, histtype="step", lw=1.8, color=colr, label=site)
        ax.axvline(thr, color=INK2, lw=1, ls="--")
        ax.text(thr, 1.01, f" {tl}", fontsize=7.5, color=INK2, transform=ax.get_xaxis_transform(),
                va="bottom", ha="center")
        if col == "drac":
            ax.set_xscale("log")
        if col == "cri":
            ax.set_yscale("log")
        ax.set_xlabel(lab); ax.set_yticklabels([]) if col != "ttc" else ax.set_ylabel("Density")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Surrogate safety measure distributions (dashed: critical threshold)", x=0.01, ha="left",
                 fontsize=11, weight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "fig03_ssm_distributions.png")


# ---------------------------------------------------------------------------
# 5-9. Prediction results
# ---------------------------------------------------------------------------
def _series_colors(best):
    return {best: S1, "TTC rule": S2, "MTTC rule": S3, "CRI rule": S4}


def fig_roc_pr():
    m = json.load(open(C.REPORTS / "metrics.json"))
    cv = json.load(open(C.REPORTS / "curves.json"))
    best = m["best_model"]
    cols = _series_colors(best)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    for j, sname in enumerate(["US-101 test", "I-80 (OOD)"]):
        for key in [best, "TTC rule", "MTTC rule", "CRI rule"]:
            c = cv[key][sname]; r = m["results"][key][sname]
            lab = f"{key.replace(' [onboard]', '')}"
            axes[0, j].plot(c["fpr"], c["tpr"], color=cols[key], label=f"{lab}  (AUC {r['roc_auc']:.3f})")
            axes[1, j].plot(c["recall"], c["precision"], color=cols[key], label=f"{lab}  (AP {r['pr_auc']:.3f})")
        axes[0, j].plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
        prev = m["results"][best][sname]
        base = (prev["tp"] + prev["fn"]) / (prev["tp"] + prev["fn"] + prev["fp"] + prev["tn"])
        axes[1, j].axhline(base, color=MUTED, lw=1, ls=":")
        axes[1, j].text(0.5, base + 0.015, f"base rate (prevalence) {base:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color=MUTED, bbox=dict(fc=SURFACE, ec="none", pad=1))
        axes[0, j].set_title(f"ROC — {sname}"); axes[1, j].set_title(f"Precision–Recall — {sname}")
        axes[0, j].set_xlabel("False positive rate"); axes[0, j].set_ylabel("True positive rate")
        axes[1, j].set_xlabel("Recall"); axes[1, j].set_ylabel("Precision")
        axes[0, j].legend(loc="lower right", fontsize=8); axes[1, j].legend(loc="upper right", fontsize=8)
        axes[1, j].set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, "fig04_roc_pr_curves.png")


def fig_model_comparison():
    m = json.load(open(C.REPORTS / "metrics.json"))
    models = ["Logistic Regression", "Random Forest", "MLP Neural Network", "Gradient Boosting"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, sname in zip(axes, ["US-101 test", "I-80 (OOD)"]):
        y = np.arange(len(models) + 3)
        names = models + ["TTC rule", "MTTC rule", "CRI rule"]
        on = [m["results"][f"{k} [onboard]"][sname]["pr_auc"] for k in models] + \
             [m["results"][k][sname]["pr_auc"] for k in ["TTC rule", "MTTC rule", "CRI rule"]]
        cx = [m["results"][f"{k} [context]"][sname]["pr_auc"] for k in models] + [np.nan] * 3
        h = 0.36
        ax.barh(y - h / 2 - 0.02, on, height=h, color=S1, label="Onboard sensors only")
        ax.barh(y + h / 2 + 0.02, cx, height=h, color=S2, label="+ V2X traffic context")
        for yi, v in zip(y, on):
            ax.text(v + 0.005, yi - h / 2, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
        for yi, v in zip(y, cx):
            if not np.isnan(v):
                ax.text(v + 0.005, yi + h / 2, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
        ax.axhline(len(models) - 0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("PR-AUC (average precision)"); ax.set_title(sname)
        ax.grid(axis="y", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _save(fig, "fig05_model_comparison.png")


def fig_importance():
    m = json.load(open(C.REPORTS / "metrics.json"))
    imp = pd.Series(m["importance"]).sort_values()
    imp = imp[imp.abs() > 1e-4]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([FEATURE_LABELS.get(k, k) for k in imp.index], imp.values,
            color=[S1 if v > 0 else MUTED for v in imp.values], height=0.62)
    ax.set_xlabel("Drop in average precision when feature is permuted")
    ax.set_title(f"Permutation feature importance — {m['best_model']}")
    ax.grid(axis="y", visible=False)
    _save(fig, "fig06_feature_importance.png")


def fig_anticipation():
    m = json.load(open(C.REPORTS / "metrics.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, sname in zip(axes, ["US-101 test", "I-80 (OOD)"]):
        b = pd.DataFrame(m["anticipation"][sname]["bins"])
        x = np.arange(len(b)); w = 0.38
        ax.bar(x - w / 2 - 0.02, b.model_recall, w, color=S1, label="ML risk model")
        ax.bar(x + w / 2 + 0.02, b.ttc_recall, w, color=S2, label="TTC rule")
        ax.set_xticks(x); ax.set_xticklabels([f"{s} s\n(n={n})" for s, n in zip(b.bin, b.n)], fontsize=8)
        ax.set_xlabel("Time until conflict onset"); ax.set_ylim(0, 1.05)
        ax.set_title(f"{sname} — equal false-alarm rate {m['anticipation'][sname]['false_alarm_rate']*100:.1f}%")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Recall (conflicts warned)"); axes[0].legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "fig07_anticipation_recall.png")


def fig_calibration():
    cal = json.load(open(C.REPORTS / "calibration.json"))
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Perfect calibration")
    for sname, colr in [("US-101 test", S1), ("I-80 (OOD)", S2)]:
        d = pd.DataFrame(cal[sname])
        ax.plot(d.pred, d.obs, "o-", color=colr, ms=6, mec="white", mew=1.5, label=sname)
    ax.set_xlabel("Predicted conflict probability"); ax.set_ylabel("Observed conflict frequency")
    ax.set_title("Reliability diagram"); ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, "fig08_calibration.png")


# ---------------------------------------------------------------------------
# 10-13. Collision avoidance results
# ---------------------------------------------------------------------------
def fig_scenario_example():
    d = joblib.load(C.DATA_PROCESSED / "sim_traces.joblib")
    tr = d["traces"]["Lead emergency braking"]
    ctrls = ["ACC (IDM)", "Reactive AEB", "Risk-Aware (ours)"]
    col = {c: tr[c]["collided"] for c in ctrls}
    cand = np.where(col["ACC (IDM)"] & col["Reactive AEB"] & ~col["Risk-Aware (ours)"])[0]
    i = int(cand[0]) if len(cand) else 0
    T = tr["lead_v"].shape[1]
    t = np.arange(T) * C.DT
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    for c in ctrls:
        r = tr[c]
        axes[0, 0].plot(t, r["gap"][i], color=CTRL_COLOR[c], label=c)
        axes[0, 1].plot(t, r["v"][i] * 3.6, color=CTRL_COLOR[c])
        axes[1, 0].plot(t, r["a"][i], color=CTRL_COLOR[c])
        if not np.isnan(r["t_collision"][i]):
            for ax in axes.flat:
                ax.axvline(r["t_collision"][i], color=CTRL_COLOR[c], lw=1, ls=":")
            axes[0, 0].plot(r["t_collision"][i], 0, "X", color=CTRL_COLOR[c], ms=11, mec="white")
    axes[0, 1].plot(t, tr["lead_v"][i] * 3.6, color=INK2, lw=1.4, ls="--", label="Leader")
    axes[1, 1].plot(t, tr["Risk-Aware (ours)"]["risk"][i], color=S1)
    bundle = joblib.load(C.MODELS / "risk_model.joblib")
    axes[1, 1].axhline(bundle["threshold"], color=INK2, lw=1, ls="--")
    axes[1, 1].text(t[-1], bundle["threshold"] + 0.02, "warning threshold ", ha="right", va="bottom",
                    fontsize=8, color=INK2, bbox=dict(fc=SURFACE, ec="none", pad=1))
    axes[0, 0].set_ylabel("Gap [m]"); axes[0, 0].set_title("Bumper-to-bumper gap  (✕ = collision)")
    axes[0, 1].set_ylabel("Speed [km/h]"); axes[0, 1].set_title("Speed")
    axes[1, 0].set_ylabel("Acceleration [m/s²]"); axes[1, 0].set_title("Applied acceleration")
    axes[1, 1].set_ylabel("P(conflict within 3 s)"); axes[1, 1].set_title("Predicted risk (risk-aware controller)")
    axes[1, 1].set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Time [s]")
    axes[0, 0].legend(loc="upper right", fontsize=8); axes[0, 1].legend(loc="upper right", fontsize=8)
    p = tr["params"].iloc[i]
    fig.suptitle(f"Example — leader emergency braking at {p.lead_decel:.1f} m/s², initial speed "
                 f"{p.v0*3.6:.0f} km/h, headway {p.thw0:.2f} s", x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout()
    _save(fig, "fig09_scenario_example.png")


def fig_controller_safety():
    res = json.load(open(C.REPORTS / "simulation_results.json"))
    scen = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ctrls = ["ACC (IDM)", "Reactive AEB", "Risk-Aware (ours)"]
    x = np.arange(len(scen)); w = 0.26
    for k, c in enumerate(ctrls):
        v = [res[s][c]["collision_rate"] * 100 for s in scen]
        bars = axes[0].bar(x + (k - 1) * (w + 0.015), v, w, color=CTRL_COLOR[c], label=c)
        for b, val in zip(bars, v):
            axes[0].text(b.get_x() + b.get_width() / 2, val + 0.8, f"{val:.1f}", ha="center", fontsize=7, color=INK2)
    axes[0].set_xticks(x); axes[0].set_xticklabels([s.replace(" ", "\n", 1) for s in scen], fontsize=8)
    axes[0].set_ylabel("Collision rate [%]"); axes[0].set_title("Collisions per scenario family")
    axes[0].legend(fontsize=8); axes[0].grid(axis="x", visible=False)

    d = joblib.load(C.DATA_PROCESSED / "sim_traces.joblib")["summary"]
    data, cols, labels = [], [], []
    for s in scen:
        for c in ctrls:
            data.append(np.clip(d[s][c]["min_ttc"].to_numpy(), 0, 10)); cols.append(CTRL_COLOR[c])
    pos = np.concatenate([i + (np.arange(3) - 1) * (w + 0.015) for i in range(len(scen))])
    bp = axes[1].boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False,
                         medianprops=dict(color="white", lw=1.6), whiskerprops=dict(color=MUTED),
                         capprops=dict(color=MUTED))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_edgecolor(c)
    axes[1].axhline(C.TTC_CRITICAL, color=INK2, lw=1, ls="--")
    axes[1].text(-0.5, C.TTC_CRITICAL + 0.1, "critical TTC = 3 s", ha="left", va="bottom", fontsize=8,
                 color=INK2, bbox=dict(fc=SURFACE, ec="none", pad=1))
    axes[1].set_xticks(x); axes[1].set_xticklabels([s.replace(" ", "\n", 1) for s in scen], fontsize=8)
    axes[1].set_ylabel("Minimum TTC per run [s] (clipped at 10)"); axes[1].set_title("Safety margin distribution")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig10_controller_safety.png")


def fig_comfort_efficiency():
    res = json.load(open(C.REPORTS / "simulation_results.json"))
    scen = list(res)
    metrics = [("mean_TET", "Time exposed to TTC < 3 s [s]"), ("mean_max_decel", "Mean peak deceleration [m/s²]"),
               ("mean_rms_jerk", "Mean RMS jerk [m/s³]"), ("mean_speed", "Mean speed [m/s]")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    y = np.arange(len(scen))
    for ax, (k, lab) in zip(axes, metrics):
        for j, c in enumerate(CTRL_ORDER):
            vals = [res[s][c][k] if c in res[s] else np.nan for s in scen]
            ax.scatter(vals, y + (j - 1.5) * 0.17, color=CTRL_COLOR[c], s=42, edgecolors="white",
                       linewidths=1.2, label=c, zorder=3)
        ax.set_xlabel(lab, fontsize=8.5)
        ax.set_yticks(y); ax.set_yticklabels(scen)
        ax.grid(axis="y", visible=False)
        for yi in y[:-1]:
            ax.axhline(yi + 0.5, color=GRID, lw=0.7)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.3, 1.2), ncol=4, fontsize=8.5)
    fig.text(0.01, -0.04, "Averages over runs in which no controller collided (common support).",
             fontsize=8, color=MUTED)
    _save(fig, "fig11_comfort_efficiency.png")


def fig_replay_vs_human():
    d = joblib.load(C.DATA_PROCESSED / "sim_traces.joblib")["summary"]["NGSIM I-80 replay"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for c in CTRL_ORDER:
        x = np.sort(np.clip(d[c]["min_ttc"].to_numpy(), 0, C.TTC_CAP))
        axes[0].step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=CTRL_COLOR[c], label=c)
        x2 = np.sort(d[c]["TET"].to_numpy())
        axes[1].step(x2, np.arange(1, len(x2) + 1) / len(x2), where="post", color=CTRL_COLOR[c], label=c)
    axes[0].axvline(C.TTC_CRITICAL, color=INK2, lw=1, ls="--")
    axes[0].set_xlabel("Minimum TTC in scenario [s]"); axes[0].set_ylabel("Cumulative share of scenarios")
    axes[0].set_title("Real-world replay (200 hardest I-80 leaders)"); axes[0].set_xlim(0, 20)
    axes[1].set_xlabel("Time exposed to TTC < 3 s [s]"); axes[1].set_title("Exposure to critical TTC")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig12_replay_vs_human.png")


def make_all():
    print("Generating figures ...")
    for f in [fig_architecture, fig_time_space, fig_ssm_distributions, fig_roc_pr,
              fig_model_comparison, fig_importance, fig_anticipation, fig_calibration,
              fig_scenario_example, fig_controller_safety, fig_comfort_efficiency,
              fig_replay_vs_human]:
        f()


if __name__ == "__main__":
    make_all()
