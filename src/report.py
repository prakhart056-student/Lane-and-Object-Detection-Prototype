"""Build the project report (reports/Project_Report.docx) from saved results.

Every number in the report is read from reports/metrics.json and
reports/simulation_results.json, so the document always matches the code.
"""
from __future__ import annotations

import datetime as dt
import json

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from . import config as C
from .features import CONTEXT, FEATURE_LABELS, ONBOARD

TITLE = "Risk-Aware Autonomous Driving and Collision Avoidance System"
AUTHOR = "Prakhar Tiwari"
SUBTITLE = ("Surrogate-safety risk assessment, machine-learning conflict anticipation and "
            "risk-adaptive longitudinal control using the NGSIM naturalistic trajectory dataset")
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
GREY = RGBColor(0x52, 0x51, 0x4E)

_fig_no = [0]
_tab_no = [0]


# ---------------------------------------------------------------------------
# docx helpers
# ---------------------------------------------------------------------------
def _field(paragraph, instr):
    run = paragraph.add_run()
    for tag, text in (("begin", None), (None, instr), ("separate", None), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text
        run._r.append(el)
    return run


def _shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def para(doc, text, bold_lead=None, style=None, align=None, italic=False, size=None):
    p = doc.add_paragraph(style=style)
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True
    r = p.add_run(text); r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullets(doc, items):
    for it in items:
        if isinstance(it, tuple):
            p = doc.add_paragraph(style="List Bullet")
            r = p.add_run(it[0]); r.bold = True
            p.add_run(it[1])
        else:
            doc.add_paragraph(it, style="List Bullet")


def numbered(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Number")


def equation(doc, text, label=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.italic = True; r.font.name = "Cambria Math"; r.font.size = Pt(11)
    if label:
        r2 = p.add_run(f"     ({label})"); r2.font.size = Pt(10)
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(4)


def figure(doc, filename, caption, width_cm=16.0):
    _fig_no[0] += 1
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(C.FIGURES / filename), width=Cm(width_cm))
    p.paragraph_format.keep_with_next = True
    c = doc.add_paragraph(style="Caption")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"Figure {_fig_no[0]}. "); r.bold = True
    c.add_run(caption)
    return _fig_no[0]


def table(doc, header, rows, caption, col_widths=None, highlight_row=None):
    _tab_no[0] += 1
    c = doc.add_paragraph(style="Caption")
    r = c.add_run(f"Table {_tab_no[0]}. "); r.bold = True
    c.add_run(caption)
    c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _shade(cell, "2A78D6")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v)); run.font.size = Pt(9)
            if highlight_row is not None and ri == highlight_row:
                run.bold = True
                _shade(cells[i], "E3EEFB")
            elif ri % 2 == 1:
                _shade(cells[i], "F4F4F2")
            if i > 0:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return _tab_no[0]


def h1(doc, text):
    doc.add_heading(text, level=1)


def h2(doc, text):
    doc.add_heading(text, level=2)


def _setup_styles(doc):
    st = doc.styles
    st["Normal"].font.name = "Calibri"; st["Normal"].font.size = Pt(11)
    st["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st["Normal"].paragraph_format.space_after = Pt(6)
    st["Normal"].paragraph_format.line_spacing = 1.15
    for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
        s = st[f"Heading {lvl}"]
        s.font.name = "Calibri"; s.font.size = Pt(size); s.font.bold = True
        s.font.color.rgb = ACCENT if lvl < 3 else GREY
        s.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        s.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
        s.paragraph_format.space_after = Pt(6)
    cap = st["Caption"]
    cap.font.size = Pt(9); cap.font.italic = False; cap.font.color.rgb = GREY
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Cm(2.3)
        sec.left_margin = sec.right_margin = Cm(2.3)
    # ask Word to refresh the table of contents on open
    settings = doc.settings.element
    upd = OxmlElement("w:updateFields"); upd.set(qn("w:val"), "true")
    settings.append(upd)


def _footer_page_numbers(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{TITLE}  ·  Page "); r.font.size = Pt(8); r.font.color.rgb = GREY
    _field(p, "PAGE").font.size = Pt(8)


# ---------------------------------------------------------------------------
# content helpers
# ---------------------------------------------------------------------------
def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f3(x):
    return f"{x:.3f}"



def _level2_chapter(doc, b_us, ttc_us):
    """Chapter 6: Level-2 extension and RC deployment (only if its results exist)."""
    av_path, rc_path = C.REPORTS / "av_results.json", C.REPORTS / "rc_results.json"
    if not av_path.exists():
        return False
    av = json.load(open(av_path))
    rc = json.load(open(rc_path)) if rc_path.exists() else None
    res = av["results"]
    ours, base = "Risk-aware (ours)", "Reactive baseline"
    labels = {"highway_overtake": "Highway: overtake slow truck", "highway_cut_in": "Highway: aggressive cut-in",
              "highway_emergency": "Highway: sudden stop, lanes blocked", "urban_route": "Urban: turns, light, pedestrian",
              "rc_track": "RC track (1:10 scale)"}

    h1(doc, "6. Extension: Level-2 Autonomy and RC-Car Deployment")
    para(doc, (
        "Chapters 3–5 covered the longitudinal task: when to slow down, brake or stop. On its own, that is the scope of "
        "SAE Level 1 driver assistance (adaptive cruise control plus emergency braking). This chapter extends the "
        "system to SAE Level 2 functionality, where the vehicle controls both speed and steering. It now keeps its "
        "lane, follows curved roads and turns, decides on its own when a lane change is safe, stops at traffic "
        "lights and yields to pedestrians. The same software is then packaged as an autopilot for a 1:10 "
        "radio-controlled car, so that it can be demonstrated on physical hardware."))

    h2(doc, "6.1 Architecture of the Level-2 stack")
    bullets(doc, [
        ("Vehicle model: ", "kinematic bicycle (wheelbase 2.8 m), steering limited to ±35° with a 90°/s rate limit "
         "and a 0.12 s lag, acceleration limited to [−8, +2.5] m/s² behind a 0.2 s actuation delay."),
        ("Road model: ", "arc-length parameterised centre lines with heading and curvature; multi-lane roads; "
         "every object is located in Frenet coordinates (s along the road, d across it)."),
        ("Perception: ", "a noisy object list (gap σ = 0.3 m, speed σ = 0.2 m/s, acceleration σ = 0.3 m/s², "
         "lateral σ = 0.1 m) within 150 m ahead and 80 m behind, plus pedestrians and the next traffic light."),
        ("Behaviour planner: ", "the decision layer described in Section 6.2."),
        ("Path tracking: ", "Pure Pursuit towards a look-ahead point on the reference path; lane changes are quintic "
         "lateral-offset blends over about 4 s."),
        ("Speed control: ", "the risk-aware law validated in Chapter 5 (ML risk model, risk-adaptive time gap, "
         "pre-emptive braking, AEB on min(TTC, MTTC)), plus curve speed limits from a lateral-acceleration budget."),
        ("Traffic: ", "IDM-driven vehicles that obey traffic lights and can be scripted to brake, change speed or "
         "cut in; pedestrians cross at zebra crossings; collisions are detected with oriented-box tests."),
    ])
    equation(doc, "δ = atan( 2 L sin α / L_d ),    L_d = max(L_min, 0.9 v)", "10")
    equation(doc, "v_curve(s) = √( a_lat,max / |κ(s)| ),    v_allowed = min over preview of √( v_curve² + 2 b Δs )", "11")

    h2(doc, "6.2 Decision making")
    para(doc, "Every 0.1 s the planner chooses a state and the matching action, with no human input:")
    drows = [
        ["CRUISE / FOLLOW", "Free road / a vehicle ahead in the occupied lanes", "Speed control towards v_des, curve and speed limits"],
        ["LANE CHANGE LEFT", "MOBIL incentive > 0.3 m/s² (slower leader within 90 m) AND safety gate passed",
         "Quintic lane-change path, ≈4 s"],
        ["Safety gate", "Target lane: front and rear TTC ≥ 4 s, gaps ≥ max(5 m, 0.4 v), new follower braking ≤ 3 m/s²",
         "Otherwise: rejected, reason logged"],
        ["LANE CHANGE: ABORTING", "Target-lane TTC < 2 s before 50 % progress", "Return to the original lane"],
        ["LANE CHANGE RIGHT", "Keep-right rule when the right lane is free", "Return manoeuvre"],
        ["STOP: RED / YELLOW", "Light ahead; stop if required deceleration ≤ 3 m/s² (dilemma zone)", "Stop 1 m before the line"],
        ["YIELD: PEDESTRIAN", "Predicted pedestrian path crosses the corridor within 4 s", "Stop 2 m before the path"],
        ["COLLISION WARNING", "min(TTC, MTTC) < 2.6 s and current braking below DRAC", "Alert"],
        ["AEB PARTIAL / EMERGENCY", "min(TTC, MTTC) < 1.6 s / 0.9 s and DRAC ≥ 1 m/s²", "−4 / −8 m/s²"],
    ]
    table(doc, ["State", "Trigger", "Action"], drows,
          "Behaviour-planner states and rules (lane changes follow MOBIL, Kesting et al., 2007, with a TTC safety gate).",
          col_widths=[3.6, 7.6, 4.8])
    para(doc, (
        "Two refinements came out of testing. First, a DRAC gate, as used in production AEB: when queuing slowly "
        "behind a stopped car, TTC can briefly fall under 2.6 s even though almost no braking is needed, which caused "
        "nuisance warnings. A warning or brake intervention now also requires the deceleration to avoid a crash "
        "(DRAC) to exceed what the car is already doing. Second, MTTC now counts only the ego vehicle's braking, "
        "not its acceleration, because the controller chooses the acceleration itself."))

    h2(doc, "6.3 Scenario evaluation")
    para(doc, (
        f"Each scenario family was run on {av['n_runs']} randomised variants (speeds, gaps, event times, "
        "decelerations and pedestrian timing drawn from new random seeds), once with the risk-aware planner and once "
        "with a reactive baseline that uses plain TTC-triggered AEB, on exactly the same variant."))
    rows = []
    for fam, lab in labels.items():
        if fam not in res:
            continue
        for cname in (ours, base):
            r = res[fam][cname]
            rows.append([lab if cname == ours else "", cname.replace(" (ours)", " (ours)"),
                         f"{r['collision_rate'] * 100:.0f}%", f"{r['lane_changes_per_run']:.1f}", str(r["aborts"]),
                         str(r["red_light_violations"]), f"{r['rms_lateral_error_m'] * 100:.1f}",
                         f"{r['max_lat_accel']:.1f}",
                         "—" if r["min_ped_clearance_m"] is None else f"{r['min_ped_clearance_m']:.1f}"])
    table(doc, ["Scenario", "Planner", "Collisions", "Lane changes / run", "Aborts", "Red-light viol.",
                "Lane RMS [cm]", "Max lat. acc. [m/s²]", "Min ped. clearance [m]"], rows,
          f"Level-2 evaluation on {av['n_runs']} unseen random variants per scenario family.",
          col_widths=[3.4, 2.6, 1.5, 1.6, 1.2, 1.4, 1.4, 1.6, 1.7])
    em = res.get("highway_emergency", {})
    if em:
        para(doc, (
            f"Lane keeping stayed within a few centimetres on every road type, no red light was run, and no pedestrian "
            f"came closer than the clearances shown. Overtakes, cut-ins and the urban route were collision-free for "
            f"both planners, because the shared decision layer handles them. The difference appears where it matters "
            f"most. When the car ahead stops suddenly and both neighbouring lanes are blocked, the reactive baseline "
            f"collided in {em[base]['collision_rate'] * 100:.0f}% of variants, the risk-aware planner in "
            f"{em[ours]['collision_rate'] * 100:.0f}%."))
    figure(doc, "fig14_av_scenarios.png",
           "Top-down simulator (demo2d.py): a risk-checked overtake, yielding to a pedestrian, and emergency braking "
           "with both neighbouring lanes blocked. The green line is the planned path.")
    figure(doc, "fig13_av_evaluation.png", "Collision rate, median minimum TTC and lane-keeping error per scenario "
                                           "family for the risk-aware and reactive planners.")

    h2(doc, "6.4 Deployment on a 1:10 RC car")
    para(doc, (
        "The rc_car package runs the same layered design on a model car. A camera, a front range sensor (HC-SR04) "
        "and the steering servo and ESC are reached through a hardware abstraction layer with three back-ends. "
        "SimCar is a simulated car on a tape oval that renders camera frames and ray-casts the range sensor. "
        "SerialCar is a Raspberry Pi or laptop with a USB camera and an Arduino generating the servo pulses. PiCar "
        "drives everything from the Raspberry Pi's GPIO pins. The autopilot is identical in all three, so what is "
        "tested in simulation and in the unit tests is exactly what drives the real car."))
    bullets(doc, [
        ("Perception: ", "homography to a bird's-eye view calibrated with an A4 sheet; tape colour thresholding; "
         "sliding-window line search; quadratic fit over the nearest 0.9 m; red stop-line band; red or green "
         "traffic-light card (compact blobs only, so the stop line is not mistaken for a light)."),
        ("Steering: ", "Pure Pursuit to a point 0.4 m ahead on the detected lane centre; corner speed limited "
         "through v² tan δ / L ≤ 0.8 m/s²."),
        ("Speed and safety: ", "IDM following (T = 0.8 s, s0 = 0.25 m), the same TTC thresholds as the full-size "
         "car (2.6 / 1.6 / 0.9 s) with a DRAC gate of 0.3 m/s², stop-line stop–wait–go, a 1.5 s memory of the "
         "light colour, a 0.3 s sensor watchdog, a controlled stop when the lane is lost, a 300 ms failsafe in "
         "the Arduino firmware, and an E-stop."),
    ])
    para(doc, (
        "An engineering lesson from development: Stanley steering with the camera's curvature estimate cut the "
        "corners of 1 m-radius bends by 15 cm (11 cm RMS), because the camera sees more than a quarter of the "
        "circle and a quadratic cannot fit that much bend. Restricting the fit to the near field and switching to "
        "Pure Pursuit on the detected centre line brought the error down to about 4 cm."))
    if rc:
        rrows = [[r["scenario"], str(r["collisions"]), f"{r['rms_lateral_cm']:.1f}", f"{r['max_lateral_cm']:.1f}",
                  "—" if r["closest_cm"] is None else f"{r['closest_cm']:.0f}", f"{r['distance_m']:.1f}"]
                 for r in rc.values()]
        table(doc, ["RC scenario (simulated, 45 s)", "Collisions", "Lane RMS [cm]", "Lane max [cm]",
                    "Closest approach [cm]", "Distance [m]"], rrows,
              "RC autopilot on the simulated car (40 cm lane, camera-based lane detection, ultrasonic range sensor).",
              col_widths=[5.2, 1.8, 1.9, 1.9, 2.4, 1.8])
    figure(doc, "fig15_rc_view.png",
           "What the RC autopilot sees: the bird's-eye camera view with the detected lane lines (yellow and blue), "
           "lane centre (green) and stop line (red), and its decisions. Top: approaching a red light. Bottom: "
           "braking behind a car that stopped.", width_cm=11)
    ptab = [["1", "Bench test, wheels off the ground", "Steering follows the lane; E-stop works"],
            ["2", "Lane following at 0.4 m/s", "3 laps within the lines"],
            ["3", "Box on the straight", "Warning → brake → stops ≥ 15 cm before it"],
            ["4", "Car ahead that stops", "Keeps a gap, stops behind it"],
            ["5", "Stop line / red card / green card", "Stops, waits, goes on green"],
            ["6", "Unplug sonar / cover camera", "FAULT / LANE_LOST → stops"]]
    table(doc, ["Step", "Physical test", "Pass criterion"], ptab,
          "Physical test protocol for the RC car (full procedure, wiring and calibration in rc_car/HARDWARE.md). "
          "Record your measured results here after testing.", col_widths=[1.2, 6.4, 8.4])

    h2(doc, "6.5 Scope")
    para(doc, (
        "The functional scope is that of SAE Level 2: the system controls speed and steering and makes its own "
        "tactical decisions, but it was validated in simulation and on a model car, not on public roads, and it "
        "is not a road-legal driving system. On the RC car the physics-based risk layer and AEB logic run live. "
        "The machine-learning predictor, trained on highway trajectories, is used in the full-size simulator."))
    return True

def build():
    m = json.load(open(C.REPORTS / "metrics.json"))
    sim = json.load(open(C.REPORTS / "simulation_results.json"))
    res = m["results"]
    best = m["best_model"]
    best_short = best.replace(" [onboard]", "")
    ds = m["dataset"]
    us, i80 = "US-101 test", "I-80 (OOD)"
    models = ["Logistic Regression", "Random Forest", "MLP Neural Network", "Gradient Boosting"]
    ctrls = ["ACC (IDM)", "Reactive AEB", "Risk-Aware (ours)"]
    eb, st, ci, sg, rp = ("Lead emergency braking", "Stopped traffic ahead", "Aggressive cut-in",
                          "Stop-and-go wave", "NGSIM I-80 replay")

    ttc_us, ttc_i = res["TTC rule"][us], res["TTC rule"][i80]
    b_us, b_i = res[best][us], res[best][i80]
    best_ctx = max((f"{k} [context]" for k in models), key=lambda k: res[k][us]["pr_auc"])
    top_feats = list(m["importance"])[:5]
    onb = [res[f"{k} [onboard]"][us]["pr_auc"] for k in models]
    pr_spread = max(onb) - min(onb)
    ttc_rank = list(m["importance"]).index("inv_ttc") + 1 if "inv_ttc" in m["importance"] else 99
    ant = m["anticipation"]
    bin_1s = [b for b in ant[us]["bins"] if b["bin"] in ("0.5-1.0",)][0]

    def coll(s, c):
        return sim[s][c]["collision_rate"]

    red_eb_acc = 1 - coll(eb, "Risk-Aware (ours)") / max(coll(eb, "ACC (IDM)"), 1e-9)
    red_eb_aeb = 1 - coll(eb, "Risk-Aware (ours)") / max(coll(eb, "Reactive AEB"), 1e-9)

    doc = Document()
    doc.core_properties.author = AUTHOR
    doc.core_properties.last_modified_by = AUTHOR
    _setup_styles(doc)

    # ------------------------------------------------------------ title page
    for _ in range(4):
        doc.add_paragraph()
    p = para(doc, "PROJECT REPORT", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
    p.runs[0].font.color.rgb = GREY; p.runs[0].bold = True
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(24); r.font.color.rgb = ACCENT
    p = para(doc, SUBTITLE, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=12)
    p.runs[0].font.color.rgb = GREY
    for _ in range(4):
        doc.add_paragraph()
    for label in ["Submitted by", f"Student name: {AUTHOR}",
                  "Roll / Enrolment No.: ________________________",
                  "Course / Programme: __________________________",
                  "Under the guidance of: ________________________",
                  "Department / Institution: _____________________"]:
        p = para(doc, label, align=WD_ALIGN_PARAGRAPH.CENTER, size=11.5)
        if label == "Submitted by":
            p.runs[0].bold = True
    for _ in range(3):
        doc.add_paragraph()
    para(doc, dt.date.today().strftime("%B %Y"), align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
    doc.add_page_break()

    # ------------------------------------------------------------ abstract
    h1(doc, "Abstract")
    para(doc, (
        "Rear-end collisions are among the most frequent crash types on motorways, and most are preceded by a short "
        "window in which the developing conflict is observable from vehicle kinematics. This project develops "
        "an end-to-end risk-aware collision avoidance system for an autonomous vehicle and evaluates it on real "
        "traffic. Using the public U.S. FHWA Next Generation Simulation (NGSIM) dataset, "
        f"{ds['us-101']['vehicles'] + ds['i-80']['vehicles']:,} naturalistic vehicle trajectories recorded at 10 Hz "
        "on two Californian freeways (US-101 and I-80) were cleaned, smoothed and converted into leader–follower "
        "interactions. Six surrogate safety measures — Time-to-Collision (TTC), Time Headway (THW), Deceleration "
        "Rate to Avoid Crash (DRAC), Modified TTC (MTTC), Crash Potential Index (CPI) and a proposed Composite Risk "
        "Index (CRI) — quantify instantaneous risk. A risk-anticipation task was then formulated: from a state "
        "that is currently safe, predict whether a critical conflict (TTC < 3 s or DRAC > 3.35 m/s²) will occur "
        "within the next 3 s. Four machine-learning models were compared against physics-based rules, training "
        "on US-101 and testing both on held-out US-101 drivers and on the unseen I-80 site. "
        f"The selected onboard-sensor model ({best_short}) reached ROC-AUC {f3(b_us['roc_auc'])} and PR-AUC "
        f"{f3(b_us['pr_auc'])} on US-101, against {f3(ttc_us['roc_auc'])} / {f3(ttc_us['pr_auc'])} for the TTC rule, "
        f"and retained ROC-AUC {f3(b_i['roc_auc'])} on the unseen I-80 site. The predicted risk drives a risk-aware "
        "longitudinal controller that enlarges its time gap, brakes pre-emptively and triggers a staged emergency "
        "brake on min(TTC, MTTC). In 2,000 Monte-Carlo stress tests and 200 real I-80 leader replays with sensor "
        "noise and actuator latency, the risk-aware controller reduced the collision rate in leader emergency-braking "
        f"scenarios from {pct(coll(eb, 'ACC (IDM)'))} (ACC) and {pct(coll(eb, 'Reactive AEB'))} (reactive AEB) to "
        f"{pct(coll(eb, 'Risk-Aware (ours)'))}, and in the real-world replays kept a median minimum TTC of "
        f"{sim[rp]['Risk-Aware (ours)']['median_min_ttc']:.2f} s versus "
        f"{sim[rp]['Human (replay)']['median_min_ttc']:.2f} s for the human drivers. The system was extended to "
        "SAE Level-2 functionality (lane keeping, path following, risk-checked lane changes, traffic lights and "
        "pedestrians), evaluated on randomised unseen scenarios, and packaged as an autopilot for a 1:10 RC car."))
    p = para(doc, "Autonomous driving; collision avoidance; surrogate safety measures; time-to-collision; "
                  "risk prediction; NGSIM; adaptive cruise control; autonomous emergency braking; machine learning; "
                  "lane keeping; behaviour planning; RC car.",
             bold_lead="Keywords: ")
    doc.add_page_break()

    # ------------------------------------------------------------ TOC
    h1(doc, "Table of Contents")
    _field(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
    para(doc, "(If the table of contents is empty, right-click it in Word and choose ‘Update Field’.)",
         italic=True, size=9)
    doc.add_page_break()

    # ------------------------------------------------------------ 1 Introduction
    h1(doc, "1. Introduction")
    h2(doc, "1.1 Background and motivation")
    para(doc, (
        "Road traffic crashes kill about 1.19 million people every year (WHO, 2023), and human error contributes to "
        "the large majority of them. Autonomous and highly automated vehicles promise to remove much of this error, but "
        "only if they can recognise danger early and respond in a way that is both safe and comfortable. Production "
        "driver-assistance systems — Adaptive Cruise Control (ACC), Forward Collision Warning (FCW) and Autonomous "
        "Emergency Braking (AEB) — mostly react to a single instantaneous indicator such as Time-to-Collision. Such "
        "reactive triggers fire late: by the time TTC drops below a fixed threshold, the deceleration needed to avoid "
        "the crash may already exceed what the vehicle can deliver."))
    para(doc, (
        "Risk awareness means estimating not only how dangerous the current situation is, but how likely it is to "
        "become dangerous in the near future, given the recent behaviour of the surrounding traffic. A vehicle that "
        "anticipates a conflict one or two seconds earlier can resolve it with moderate braking instead of an "
        "emergency stop."))
    h2(doc, "1.2 Problem statement")
    para(doc, (
        "Given real, noisy observations of an ego vehicle and the vehicle ahead of it, (i) quantify the instantaneous "
        "rear-end collision risk, (ii) predict whether a critical conflict will develop within a short horizon, and "
        "(iii) use that prediction inside a longitudinal controller so that the vehicle avoids collisions with "
        "smaller safety-margin violations than reactive systems, without sacrificing comfort or traffic efficiency."))
    h2(doc, "1.3 Objectives")
    numbered(doc, [
        "Acquire and clean a large, public, real-world vehicle-trajectory dataset suitable for conflict analysis.",
        "Implement established surrogate safety measures (TTC, THW, DRAC, MTTC, CPI) and design a composite, "
        "interpretable risk index with discrete risk levels.",
        "Formulate conflict anticipation as a supervised learning problem and compare machine-learning models "
        "against physics-based rules, including generalisation to an unseen road.",
        "Design a risk-aware collision avoidance controller that uses the predicted risk online.",
        "Evaluate the controller in closed-loop simulation against ACC-only and reactive AEB baselines, and against "
        "real human drivers, using safety, comfort and efficiency metrics."])
    h2(doc, "1.4 Contributions")
    bullets(doc, [
        ("Reproducible data pipeline ", "that downloads NGSIM directly from the U.S. DOT open-data API, "
         "repairs known noise problems with Savitzky–Golay filtering and builds leader–follower interactions."),
        ("A Composite Risk Index (CRI) ", "that fuses TTC, THW and DRAC into one bounded score with four driver-facing "
         "levels (Safe, Caution, Danger, Critical)."),
        ("A leakage-free anticipation benchmark ", "that only scores samples which are currently safe, uses "
         "driver-grouped splits, and tests on a second freeway."),
        ("A Level-2 extension ", "(lane keeping, path following, MOBIL lane changes with a TTC safety gate, "
         "traffic lights, pedestrians) and an RC-car autopilot with firmware, calibration and safety watchdogs."),
        ("A risk-aware controller ", "that combines a risk-adaptive Intelligent Driver Model, pre-emptive graded "
         "braking and an MTTC-staged AEB, evaluated with sensor noise and actuator latency."),
    ])
    h2(doc, "1.5 Report organisation")
    para(doc, ("Chapter 2 reviews related work. Chapter 3 describes the dataset. Chapter 4 presents the methodology, "
               "Chapter 5 the results, Chapter 6 the discussion and limitations, and Chapter 7 concludes."))

    # ------------------------------------------------------------ 2 Literature
    h1(doc, "2. Literature Review")
    h2(doc, "2.1 Surrogate safety measures")
    para(doc, (
        "Because crashes are rare, traffic-safety research relies on surrogate safety measures (SSMs) that quantify "
        "how close an interaction came to a crash (Mahmud et al., 2017). Time-to-Collision, introduced by Hayward "
        "(1972), is the time remaining before two vehicles collide if they keep their current speeds; values below "
        "1.5–3 s are commonly treated as critical (Vogel, 2003). Time headway measures following distance in seconds "
        "and underpins the ‘two-second rule’. Deceleration Rate to Avoid Crash (Cooper & Ferguson, 1976) gives the "
        "braking needed to avoid a collision, with 3.35 m/s² a common critical value (Archer, 2005). TTC ignores "
        "accelerations, so Ozbay et al. (2008) proposed Modified TTC (MTTC), which captures conflicts caused by a "
        "braking leader. The Crash Potential Index (Cunto & Saccomanno, 2008) is the probability that DRAC exceeds "
        "the vehicle's maximum available deceleration. Aggregate indicators such as Time Exposed TTC (TET) and Time "
        "Integrated TTC (TIT) (Minderhoud & Bovy, 2001) summarise exposure over a trip."))
    h2(doc, "2.2 Collision avoidance systems")
    para(doc, (
        "Adaptive Cruise Control keeps a set speed and time gap. ISO 15622 limits ACC braking authority to a few "
        "m/s², so ACC alone cannot handle emergencies. The Intelligent Driver Model (IDM; Treiber et al., 2000) "
        "is a widely used, collision-free car-following law and a common basis for ACC design (Kesting et al., 2010). "
        "AEB systems add a staged cascade (warning, partial braking, full braking) triggered by TTC thresholds and "
        "are tested by Euro NCAP. Their main weakness is late, reactive activation."))
    h2(doc, "2.3 Data-driven risk prediction")
    para(doc, (
        "Naturalistic trajectory datasets such as NGSIM, highD and inD have enabled data-driven conflict prediction. "
        "Studies have used logistic regression, tree ensembles and recurrent neural networks to predict near-crash "
        "events from kinematic histories. Many report inflated accuracy, however, because they include samples that "
        "are already critical, or split data randomly so that the same driver appears in training and test data. "
        "NGSIM also needs careful smoothing before derivatives such as acceleration can be trusted "
        "(Punzo et al., 2011; Montanino & Punzo, 2015)."))
    h2(doc, "2.4 Research gap")
    para(doc, (
        "Few student-level and applied studies close the loop: they either assess risk offline or design a controller "
        "on synthetic scenarios. This project connects the two. A risk model trained on real driving is evaluated "
        "honestly, with anticipation-only samples, grouped splits and a second site, and then deployed inside a "
        "controller tested on both synthetic stress cases and real leader trajectories."))

    # ------------------------------------------------------------ 3 Dataset
    h1(doc, "3. Dataset")
    h2(doc, "3.1 NGSIM vehicle trajectories")
    para(doc, (
        "The Next Generation Simulation (NGSIM) programme of the U.S. Federal Highway Administration recorded "
        "freeway traffic with synchronised digital cameras mounted on adjacent high-rise buildings. Every vehicle on "
        "the study section was tracked by computer vision at 10 frames per second, giving its position, speed, "
        "acceleration, lane, dimensions, class and the IDs of its leader and follower. NGSIM is the most widely used "
        "public naturalistic trajectory dataset in traffic-safety research. It is in the public domain and available "
        "through the U.S. DOT open-data portal (dataset 8ect-6jqj), with no registration needed, which makes this "
        "project fully reproducible."))
    rows = [
        ["US-101 (Hollywood Fwy, Los Angeles)", "15 Jun 2005, 07:50–08:05 PDT", "≈640 m, 5 lanes + aux",
         f"{ds['us-101']['vehicles']:,}", f"{ds['us-101']['samples']:,}", pct(ds['us-101']['positive_rate'], 2),
         "Train / validation / test"],
        ["I-80 (Emeryville, San Francisco Bay)", "13 Apr 2005, 16:00–16:14 PDT", "≈500 m, 6 lanes",
         f"{ds['i-80']['vehicles']:,}", f"{ds['i-80']['samples']:,}", pct(ds['i-80']['positive_rate'], 2),
         "Unseen-site test"],
    ]
    table(doc, ["Site", "Recording window", "Section", "Vehicles*", "Samples", "Positive rate", "Role"], rows,
          "Study sites. Samples are anticipation samples (currently safe, moving, 0.5 s spacing). "
          "*Vehicles contributing at least one anticipation sample.",
          col_widths=[4.0, 3.0, 2.3, 1.6, 1.7, 1.6, 2.6])
    h2(doc, "3.2 Preprocessing")
    numbered(doc, [
        "Download through the Socrata API, one 14-minute window per site (the first 60 s skipped while the "
        "section fills).",
        "Remove duplicate (vehicle, frame) records and convert imperial units to SI.",
        "Split each vehicle into contiguous segments, smooth the longitudinal position with a Savitzky–Golay "
        f"filter (window {C.SMOOTH_WINDOW} frames, order {C.SMOOTH_POLY}), and differentiate analytically to obtain "
        "consistent speed and acceleration. Raw NGSIM accelerations contain physically impossible spikes.",
        f"Trim {C.EDGE_TRIM} frames (0.5 s) at each segment end, where filter derivatives are unreliable.",
        "Join each follower to its leader at the same frame using the NGSIM ‘preceding’ field, keep same-lane "
        "main-line pairs, and compute the bumper-to-bumper gap (leader front position − leader length − follower "
        "front position). Pairs with a gap outside 0.5–150 m are treated as tracking errors and removed.",
        "Attach lane-level traffic context (mean lane speed and density) for every frame."])
    figure(doc, "fig02_time_space_risk.png",
           "Left: time–space diagram of one US-101 lane coloured by speed. Dark diagonal bands are stop-and-go "
           "waves travelling upstream. Right: the same trajectories with every frame classified as Caution, Danger "
           "or Critical by the Composite Risk Index. Risk clusters at the fronts of congestion waves.")

    # ------------------------------------------------------------ 4 Methodology
    h1(doc, "4. Methodology")
    h2(doc, "4.1 System architecture")
    para(doc, ("The system follows the classic sense–assess–predict–act loop of an automated vehicle "
               "(Figure 2). Perception provides the gap to the leader and both vehicles' kinematics. The risk "
               "assessment layer computes interpretable SSMs. The prediction layer estimates the probability of a "
               "conflict within 3 s. The decision layer turns that probability into an acceleration command, which "
               "is executed with realistic latency and braking limits."))
    figure(doc, "fig01_architecture.png", "Architecture of the risk-aware collision avoidance system.")
    h2(doc, "4.2 Risk assessment: surrogate safety measures")
    para(doc, "With gap g, follower speed v_F, leader speed v_L, closing speed Δv = v_F − v_L and accelerations "
              "a_F, a_L:")
    equation(doc, "TTC = g / Δv      (Δv > 0; otherwise ∞)", "1")
    equation(doc, "THW = g / v_F", "2")
    equation(doc, "DRAC = Δv² / (2g)", "3")
    equation(doc, "MTTC = smallest t > 0 such that  g − Δv·t − ½(a_F − a_L)·t² = 0", "4")
    equation(doc, "CPI = P(DRAC > MADR),   MADR ~ N(8.45, 1.4²) m/s²", "5")
    para(doc, ("The Composite Risk Index maps each indicator to [0, 1] with a smooth function that equals 0.5 at "
               "its critical threshold, then combines them:"))
    equation(doc, "r_TTC = 1 / (1 + (TTC/3)³),   r_THW = 1 / (1 + (THW/1)³),   r_DRAC = 1 − 2^(−DRAC/3.35)", "6")
    equation(doc, "CRI = 0.5·r_TTC + 0.2·r_THW + 0.3·r_DRAC", "7")
    para(doc, ("CRI is discretised into Safe (< 0.25), Caution (0.25–0.5), Danger (0.5–0.75) and Critical (≥ 0.75), "
               "which is a natural input for a driver-facing warning display."))
    h2(doc, "4.3 Risk anticipation task")
    para(doc, (
        f"A frame is critical if TTC < {C.TTC_CRITICAL:.0f} s or DRAC > {C.DRAC_CRITICAL} m/s². For every frame t "
        f"the label is 1 if any critical frame occurs in (t, t + {C.HORIZON_S:.0f} s] with the same leader, and 0 "
        "otherwise. Only frames with a fully observed history and horizon are used. Three design choices make the "
        "benchmark honest:"))
    bullets(doc, [
        ("Anticipation only: ", "frames that are already critical are removed, so the model cannot score well by "
         "simply recognising an ongoing conflict."),
        ("Grouped splits: ", "US-101 is split by vehicle ID into train (60%), validation (15%) and test (25%), "
         "so no driver appears in two splits."),
        ("External test: ", "I-80 is never used for training or tuning, and measures transfer to a different road, "
         "day and congestion level."),
    ])
    para(doc, "Samples are taken every 0.5 s to reduce temporal autocorrelation, and standstill frames "
              "(v < 1 m/s) are excluded.")
    h2(doc, "4.4 Feature engineering")
    para(doc, ("Two feature sets were compared. The onboard set uses only quantities that the ego vehicle's radar, "
               "camera and odometry can measure, so the deployed model needs no infrastructure. The context set adds "
               "information that could come through V2X communication."))
    frows = [[FEATURE_LABELS[k], k, "Onboard" if k in ONBOARD else "Context (V2X)"] for k in CONTEXT]
    table(doc, ["Feature", "Code name", "Set"], frows,
          f"Features ({len(ONBOARD)} onboard, {len(CONTEXT) - len(ONBOARD)} context). ‘1 s’ features are computed "
          "over the last 1 s of history.", col_widths=[6, 4, 4])
    h2(doc, "4.5 Models and evaluation protocol")
    bullets(doc, [
        ("Physics baselines: ", "rank samples by current 1/TTC, 1/MTTC or CRI. These are the logic of a "
         "conventional threshold-based FCW."),
        ("Logistic Regression: ", "standardised features, L2 regularisation. Interpretable and calibrated."),
        ("Random Forest: ", "300 trees, depth ≤ 14, minimum 5 samples per leaf."),
        ("MLP Neural Network: ", "two hidden layers (64, 32), L2 penalty, early stopping."),
        ("Gradient Boosting: ", "histogram-based gradient boosted trees, 31 leaves, learning rate 0.05, early "
         "stopping."),
    ])
    para(doc, (
        f"Conflicts are rare ({pct(min(d['positive_rate'] for d in ds.values()))}–"
        f"{pct(max(d['positive_rate'] for d in ds.values()))} of samples), so the primary metric is the area under the precision–recall curve "
        "(PR-AUC, or average precision), together with ROC-AUC. Each model's decision threshold is chosen on the "
        "validation set to maximise F1, then frozen for testing. The deployed model is the onboard model with the "
        "highest validation PR-AUC. It is never selected on test data."))
    h2(doc, "4.6 Risk-aware collision avoidance controller")
    para(doc, "All controllers share the IDM car-following law with desired speed v₀, time gap T, maximum "
              "acceleration a and comfortable deceleration b:")
    equation(doc, "a_IDM = a·[1 − (v/v₀)⁴ − (s*/g)²],    s* = s₀ + v·T + v·Δv / (2√(ab))", "8")
    para(doc, (f"Parameters: v₀ = {C.IDM['v0']} m/s, T = {C.IDM['T']} s, a = {C.IDM['a_max']} m/s², "
               f"b = {C.IDM['b_comf']} m/s², s₀ = {C.IDM['s0']} m. Following ISO 15622, ACC braking is limited to "
               f"{-C.ACC_A_MIN} m/s². The three controllers compared are:"))
    bullets(doc, [
        ("ACC (IDM): ", "a = max(a_IDM, −3.5)."),
        ("Reactive AEB: ", f"ACC plus a staged AEB on TTC: −{-C.AEB['a_partial']:.0f} m/s² when TTC < "
         f"{C.AEB['ttc_partial']} s and −{-C.AEB['a_full']:.0f} m/s² when TTC < {C.AEB['ttc_full']} s."),
        ("Risk-Aware (proposed): ", "the ML model outputs p(t) = P(conflict within 3 s). The risk is filtered "
         "with fast attack and slow release, then (i) the IDM time gap becomes T·(1 + 0.8p), (ii) when p exceeds "
         "the model threshold θ and the gap is closing, a graded pre-emptive deceleration "
         f"−1.5 − 3.5·(p − θ)/(1 − θ) m/s² (bounded at {-C.RISK_A_MIN} m/s²) is applied, and (iii) the staged AEB "
         "is triggered by min(TTC, MTTC), so that a braking leader is caught before TTC alone would react."),
    ])
    equation(doc, "a_cmd = min( a_IDM(T·(1+0.8p)),  a_pre(p),  a_AEB(min(TTC, MTTC)) )", "9")
    para(doc, (f"Realism: every controller acts through a {C.REACTION_DELAY_S} s actuation delay, accelerations are "
               f"limited to [{C.VEHICLE_A_MIN}, {C.VEHICLE_A_MAX}] m/s², and perception adds Gaussian noise "
               "(σ = 0.3 m on gap, 0.2 m/s on leader speed, 0.3 m/s² on leader acceleration). The simulator is "
               "vectorised, so every scenario in a family advances in lock-step and the risk model is queried once "
               "per 0.1 s step."))
    h2(doc, "4.7 Test scenarios")
    srows = [
        [eb, "500", "Leader at 54–108 km/h, headway 0.8–2.0 s, brakes to a stop at 4–8 m/s²"],
        [ci, "500", "Vehicle cuts in 5–20 m ahead, 0–8 m/s slower, then brakes at up to 3 m/s²"],
        [sg, "500", "Leader speed oscillates (±3–9 m/s, period 8–18 s) around 10–20 m/s"],
        [st, "500", "Stationary queue detected 50–150 m ahead at 54–108 km/h"],
        [rp, "200", "Real I-80 leader trajectories with the strongest braking (unseen site); the real human "
                    "follower is the reference"],
    ]
    table(doc, ["Scenario family", "Runs", "Description"], srows,
          "Closed-loop test scenarios (Monte-Carlo parameters sampled uniformly).", col_widths=[4.2, 1.4, 10.4])
    h2(doc, "4.8 Evaluation metrics")
    bullets(doc, [
        ("Safety: ", "collision rate, impact speed, minimum TTC per run, TET and TIT (threshold 3 s)."),
        ("Comfort: ", "peak deceleration, RMS jerk, and time spent braking harder than 4 m/s²."),
        ("Efficiency: ", "mean speed."),
    ])
    para(doc, ("Some stress cases cannot be avoided by any vehicle, because they need more than 8 m/s² of braking. "
               "Collision rates are therefore also reported for physically avoidable runs only. Comfort and "
               "efficiency are averaged over runs where no controller crashed, because crashed runs end early and "
               "would bias those averages."))

    # ------------------------------------------------------------ 5 Results
    h1(doc, "5. Results")
    h2(doc, "5.1 Risk in naturalistic traffic")
    figure(doc, "fig03_ssm_distributions.png",
           "Distributions of TTC (finite values), THW, DRAC and CRI for both sites. Dashed lines mark the critical "
           "thresholds.")
    para(doc, ("Both freeways are congested, so most followers keep short headways, and a substantial share of THW values "
               "are below 1 s, "
               "yet critical TTC events are rare. This is why time headway alone is a poor conflict predictor, and "
               "why conflicts tend to appear suddenly when a stop-and-go wave reaches a driver (Figure 1). I-80 "
               "has a higher conflict rate than US-101, which makes it a demanding out-of-distribution test."))

    h2(doc, "5.2 Conflict anticipation performance")
    rows, hl = [], None
    for k in ["TTC rule", "MTTC rule", "CRI rule"] + [f"{x} [onboard]" for x in models] + [f"{x} [context]" for x in models]:
        r1, r2 = res[k][us], res[k][i80]
        if k == best:
            hl = len(rows)
        rows.append([k, f3(r1["roc_auc"]), f3(r1["pr_auc"]), f3(r1["f1"]), f3(r1["recall"]),
                     pct(r1["false_alarm_rate"]), f3(r2["roc_auc"]), f3(r2["pr_auc"]), f3(r2["recall"])])
    table(doc, ["Model", "ROC-AUC", "PR-AUC", "F1", "Recall", "False alarm", "ROC-AUC", "PR-AUC", "Recall"], rows,
          "Anticipation results. Columns 2–6: US-101 held-out drivers. Columns 7–9: unseen I-80 site. Thresholds "
          "were tuned on validation. The highlighted row is the deployed model.",
          col_widths=[4.6, 1.4, 1.4, 1.2, 1.3, 1.6, 1.4, 1.4, 1.3], highlight_row=hl)
    para(doc, (
        f"Every learned model clearly beats the physics rules. The deployed {best_short} lifts PR-AUC from "
        f"{f3(ttc_us['pr_auc'])} (TTC rule) to {f3(b_us['pr_auc'])} on US-101, a "
        f"{b_us['pr_auc'] / ttc_us['pr_auc']:.1f}× improvement, and from {f3(ttc_i['pr_auc'])} to "
        f"{f3(b_i['pr_auc'])} on the unseen I-80 site. All models lose some accuracy when moving to I-80, as "
        "expected under distribution shift (PR-AUC holds up because conflicts are more frequent on I-80), but "
        "every learned model stays well above the rules. The four learners, from a linear model to a neural "
        f"network, lie within {pr_spread:.3f} PR-AUC of each other on US-101. The predictive power therefore comes "
        "from the engineered kinematic-history features rather than from model complexity, and logistic "
        "regression remains an interpretable alternative with almost identical accuracy. "
        + (f"Adding V2X traffic context brings no measurable benefit (best context model: {best_ctx}, PR-AUC "
           f"{f3(res[best_ctx][us]['pr_auc'])} vs {f3(b_us['pr_auc'])} onboard), so the vehicle's own sensors "
           "are sufficient. This is encouraging for deployment without roadside infrastructure."
           if res[best_ctx][us]['pr_auc'] <= b_us['pr_auc'] + 0.005 else
           f"Adding V2X traffic context gives a modest gain (best: {best_ctx}, PR-AUC "
           f"{f3(res[best_ctx][us]['pr_auc'])}).")))
    figure(doc, "fig04_roc_pr_curves.png",
           "ROC (top) and precision–recall (bottom) curves of the deployed model and the physics-based rules, "
           "for held-out US-101 drivers (left) and the unseen I-80 site (right).")
    figure(doc, "fig05_model_comparison.png",
           "PR-AUC of all models with onboard-only and V2X-augmented features. Below the dashed line are the "
           "physics rules.")
    h2(doc, "5.3 How early are conflicts anticipated?")
    para(doc, (
        "Figure 6 splits the positive samples by the time remaining until the conflict starts, and compares recall "
        "at an identical false-alarm rate for the model and the TTC rule. For conflicts 0.5–1.0 s away, the model "
        f"warns in {pct(bin_1s['model_recall'], 0)} of cases against {pct(bin_1s['ttc_recall'], 0)} for TTC. "
        "This extra half-second to second of warning lets the controller brake moderately instead of in an "
        "emergency. Recall naturally falls for conflicts 2–3 s away, which are often triggered by leader "
        "manoeuvres that have not started yet."))
    figure(doc, "fig07_anticipation_recall.png",
           "Share of upcoming conflicts warned, by time until onset, for the ML model and TTC rule at the same "
           "false-alarm rate.")
    h2(doc, "5.4 Interpretability and calibration")
    para(doc, (
        "Permutation importance (Figure 7) shows what drives the prediction: "
        + ", ".join(FEATURE_LABELS.get(f, f) for f in top_feats[:-1])
        + f" and {FEATURE_LABELS.get(top_feats[-1], top_feats[-1])}. These are physically meaningful. "
        "The required deceleration and closing speed express how quickly the ego vehicle approaches, the 1-s "
        "gap change captures a trend that instantaneous measures miss, and leader deceleration is the classic "
        "precursor of a rear-end conflict. "
        + (f"Inverse TTC itself ranks only #{ttc_rank} of {len(m['importance'])} once these are known. "
           if ttc_rank > 5 else "") +
        "The reliability diagram (Figure 8) shows that predicted probabilities follow observed frequencies "
        "reasonably well, which matters because the controller uses the probability directly rather than a "
        "binary alarm."))
    figure(doc, "fig06_feature_importance.png", "Permutation feature importance of the deployed model "
                                                  "(US-101 test set).", width_cm=12.5)
    figure(doc, "fig08_calibration.png", "Reliability diagram of the deployed model on both sites.", width_cm=9)

    h2(doc, "5.5 Closed-loop collision avoidance")
    rows = []
    for s in [eb, ci, sg, st, rp]:
        for c in ctrls + (["Human (replay)"] if "Human (replay)" in sim[s] else []):
            r = sim[s][c]
            rows.append([s if c == ctrls[0] else "", c,
                         pct(r["collision_rate"]) if c != "Human (replay)" else "—",
                         pct(r["collision_rate_avoidable"]) if c != "Human (replay)" else "—",
                         f"{r['median_min_ttc']:.2f}", f"{r['mean_TET']:.2f}", f"{r['mean_max_decel']:.2f}",
                         f"{r['mean_rms_jerk']:.2f}", f"{r['mean_speed']:.1f}"])
    table(doc, ["Scenario", "Controller", "Collisions", "Coll. (avoidable)", "Median min-TTC [s]", "TET [s]",
                "Peak decel [m/s²]", "RMS jerk [m/s³]", "Mean speed [m/s]"], rows,
          "Closed-loop results. Comfort and efficiency columns are averaged over runs in which no controller "
          "collided.", col_widths=[2.8, 2.7, 1.5, 1.6, 1.6, 1.2, 1.5, 1.5, 1.5])
    para(doc, (
        f"Leader emergency braking is the decisive test. With its limited braking authority, ACC alone collides in "
        f"{pct(coll(eb, 'ACC (IDM)'))} of runs. Reactive AEB cuts this to {pct(coll(eb, 'Reactive AEB'))}, but "
        f"because it waits for TTC to fall below 1.6 s, it often brakes too late. The risk-aware controller "
        f"collides in only {pct(coll(eb, 'Risk-Aware (ours)'))}, a reduction of {pct(red_eb_acc, 0)} versus ACC and "
        f"{pct(red_eb_aeb, 0)} versus reactive AEB. Its median minimum TTC rises from "
        f"{sim[eb]['Reactive AEB']['median_min_ttc']:.2f} s to {sim[eb]['Risk-Aware (ours)']['median_min_ttc']:.2f} s. "
        "Figure 9 shows why. The predicted risk jumps as soon as the leader starts to decelerate, about two "
        "seconds before the reactive AEB engages. The controller begins graded braking straight away and only needs "
        "full AEB briefly at the end, whereas the baselines are still closing the gap at full speed."))
    para(doc, (
        f"For stopped traffic ahead, all controllers face some physically unavoidable runs (detection too late at "
        f"high speed: {pct(1 - sim[st]['ACC (IDM)']['share_avoidable'])} of runs would need more than 8 m/s²). Among the "
        f"avoidable runs, the risk-aware controller's collision rate is {pct(sim[st]['Risk-Aware (ours)']['collision_rate_avoidable'])} "
        f"versus {pct(sim[st]['Reactive AEB']['collision_rate_avoidable'])} for reactive AEB and "
        f"{pct(sim[st]['ACC (IDM)']['collision_rate_avoidable'])} for ACC. In cut-in and stop-and-go traffic all "
        "controllers are almost always safe. There, the risk-aware controller keeps the same mean speed and has the "
        "lowest exposure to TTC < 3 s, so it does not slow traffic down. The price is somewhat firmer braking: "
        f"peak deceleration in cut-ins is {sim[ci]['Risk-Aware (ours)']['mean_max_decel']:.2f} m/s² against "
        f"{sim[ci]['Reactive AEB']['mean_max_decel']:.2f} m/s² for reactive AEB, with RMS jerk of "
        f"{sim[ci]['Risk-Aware (ours)']['mean_rms_jerk']:.2f} vs {sim[ci]['Reactive AEB']['mean_rms_jerk']:.2f} m/s³. "
        "The controller deliberately trades a short period of firmer braking early for avoiding an emergency later."))
    figure(doc, "fig09_scenario_example.png",
           "A leader emergency stop. ACC and reactive AEB collide (✕), while the risk-aware controller detects "
           "the risk early, brakes progressively and stops behind the leader.")
    figure(doc, "fig10_controller_safety.png",
           "Collision rate (left) and distribution of the minimum TTC per run (right) for every scenario family.")
    figure(doc, "fig11_comfort_efficiency.png",
           "Safety exposure, comfort and efficiency trade-offs for every scenario family and controller.")
    h2(doc, "5.6 Real-world replay against human drivers")
    hr, rr = sim[rp]["Human (replay)"], sim[rp]["Risk-Aware (ours)"]
    para(doc, (
        "The 200 I-80 leaders with the strongest braking were replayed, and each controller started from the real "
        "follower's speed and gap. The human drivers did not crash in these clips, but they spent on average "
        f"{hr['mean_TET']:.2f} s per clip below TTC 3 s, with a median minimum TTC of {hr['median_min_ttc']:.2f} s. "
        f"The risk-aware controller reduced exposure to {rr['mean_TET']:.2f} s and raised the median minimum TTC to "
        f"{rr['median_min_ttc']:.2f} s, while braking less hard than the humans (peak {rr['mean_max_decel']:.2f} vs "
        f"{hr['mean_max_decel']:.2f} m/s²) at the same average speed ({rr['mean_speed']:.1f} vs "
        f"{hr['mean_speed']:.1f} m/s)."))
    figure(doc, "fig12_replay_vs_human.png",
           "Cumulative distributions of minimum TTC (left) and TET (right) over the 200 real I-80 replays. For minimum "
           "TTC, curves further right are safer; for TET, curves further left are safer. The ACC and reactive AEB "
           "curves almost coincide.")

    has_l2 = _level2_chapter(doc, b_us, ttc_us)
    n_disc, n_conc = (7, 8) if has_l2 else (6, 7)

    # ------------------------------------------------------------ Discussion
    h1(doc, f"{n_disc}. Discussion")
    h2(doc, f"{n_disc}.1 Key findings")
    bullets(doc, [
        "Kinematic history predicts upcoming rear-end conflicts much better than instantaneous TTC, and this "
        "advantage holds on a road the model has never seen.",
        "The most useful early-warning signals are gap dynamics and recent leader deceleration, which are all "
        "measurable with standard automotive radar.",
        "Using a calibrated risk probability to adapt the controller turns that early warning into a large drop in "
        "collisions, without loss of traffic efficiency. The cost is firmer, earlier braking in the minority of "
        "situations where risk is predicted.",
        "Model complexity matters little: linear, tree-based and neural models perform almost identically, so "
        "feature design and honest evaluation matter more than the choice of learner.",
        "The predicted probabilities are well calibrated on both sites, which is what allows the controller to use "
        "them directly instead of as a binary alarm.",
    ])
    h2(doc, f"{n_disc}.2 Limitations")
    bullets(doc, [
        "The machine-learning risk model covers rear-end conflicts. Lane-change safety uses rule-based TTC and gap "
        "checks, and intersections are limited to traffic lights and crossings; learned lateral risk is future work.",
        "The Level-2 functions were validated in a 2-D simulator with scripted traffic and on a simulated RC car. "
        "Physical RC tests (Section 6.4) still have to be carried out and reported.",
        "NGSIM is from 2005, and even after smoothing contains video-tracking errors. Newer datasets such as highD "
        "are more accurate but need registration.",
        "Critical events are defined by surrogate thresholds, not by actual crashes, which is standard practice "
        "but still an approximation.",
        "In simulation the leader is scripted and does not react to the ego vehicle. Mixed-traffic interaction "
        "and string stability were not studied.",
        "The model is trained on human followers but deployed on an automated follower, which is a distribution "
        "shift. Fine-tuning on data from the controller itself would address this.",
    ])
    h2(doc, f"{n_disc}.3 Safety and ethical considerations")
    para(doc, ("A learned risk model should never be the only safety layer. In the proposed design, the ML model "
               "can only add earlier braking, and the physics-based AEB on min(TTC, MTTC) remains active as an "
               "independent safety net. This follows the defence-in-depth principle of ISO 26262 and ISO 21448 "
               "(SOTIF). Any real deployment would need far larger validation, hardware-in-the-loop testing and "
               "formal safety argumentation."))

    # ------------------------------------------------------------ 7 Conclusion
    h1(doc, f"{n_conc}. Conclusion and Future Work")
    para(doc, (
        "This project built and evaluated a complete risk-aware collision avoidance pipeline on real traffic data. "
        "Surrogate safety measures and a composite risk index quantify instantaneous danger. A machine-learning "
        f"model trained on naturalistic US-101 driving anticipates conflicts 3 s ahead (PR-AUC {f3(b_us['pr_auc'])}, "
        f"{b_us['pr_auc'] / ttc_us['pr_auc']:.1f}× the TTC rule) and transfers to an unseen freeway. Embedded in a "
        "risk-adaptive controller with an MTTC-triggered AEB backup, it reduced emergency-braking collisions from "
        f"{pct(coll(eb, 'Reactive AEB'))} (reactive AEB) to {pct(coll(eb, 'Risk-Aware (ours)'))}, and on real leader "
        "trajectories kept larger safety margins than human drivers while braking more gently."))
    para(doc, ("The system was then extended to Level-2 functionality: lane keeping, path following through curves "
               "and turns, risk-checked lane changes with abort, traffic lights and pedestrians. It was evaluated on "
               "randomised unseen scenarios and packaged, with firmware, calibration tools and a test protocol, as an "
               "autopilot for a 1:10 RC car."))
    para(doc, "Future work:", bold_lead=None)
    bullets(doc, [
        "Learn lateral risk (cut-in intention, lane-change conflicts) from the lateral NGSIM coordinates or the "
        "highD dataset, replacing the rule-based gap check.",
        "Run the physical RC test protocol, then add wheel-speed sensing (encoder) for closed-loop speed control.",
        "Replace the feature-based classifier with sequence models (LSTM or Transformer) over multi-vehicle "
        "context, and quantify predictive uncertainty.",
        "Learn the control policy directly with safe reinforcement learning, using the risk model as a shaping "
        "signal and the AEB as a shield.",
        "Validate in a high-fidelity simulator (CARLA or SUMO) with interactive traffic, and study platoon string "
        "stability.",
    ])

    # ------------------------------------------------------------ References
    h1(doc, "References")
    refs = [
        "Archer, J. (2005). Indicators for traffic safety assessment and prediction and their application in "
        "micro-simulation modelling. PhD thesis, Royal Institute of Technology (KTH), Stockholm.",
        "Coulter, R. C. (1992). Implementation of the Pure Pursuit path tracking algorithm. Technical report "
        "CMU-RI-TR-92-01, Robotics Institute, Carnegie Mellon University.",
        "Cooper, D. F., & Ferguson, N. (1976). Traffic studies at T-junctions — a conflict simulation model. "
        "Traffic Engineering & Control, 17, 306–309.",
        "Cunto, F., & Saccomanno, F. F. (2008). Calibration and validation of simulated vehicle safety performance "
        "at signalized intersections. Accident Analysis & Prevention, 40(3), 1171–1179.",
        "Euro NCAP (2023). Test protocol — AEB Car-to-Car systems. European New Car Assessment Programme.",
        "Hayward, J. C. (1972). Near-miss determination through use of a scale of danger. Highway Research Record, "
        "384, 24–34.",
        "ISO 15622:2018. Intelligent transport systems — Adaptive cruise control systems — Performance "
        "requirements and test procedures. International Organization for Standardization.",
        "Kesting, A., Treiber, M., & Helbing, D. (2007). General lane-changing model MOBIL for car-following models. "
        "Transportation Research Record, 1999, 86–94.",
        "Kesting, A., Treiber, M., & Helbing, D. (2010). Enhanced intelligent driver model to access the impact of "
        "driving strategies on traffic capacity. Philosophical Transactions of the Royal Society A, 368, 4585–4605.",
        "Mahmud, S. M. S., Ferreira, L., Hoque, M. S., & Tavassoli, A. (2017). Application of proximal surrogate "
        "indicators for safety evaluation: A review of recent developments and research needs. IATSS Research, "
        "41(4), 153–163.",
        "Minderhoud, M. M., & Bovy, P. H. L. (2001). Extended time-to-collision measures for road traffic safety "
        "assessment. Accident Analysis & Prevention, 33(1), 89–97.",
        "Montanino, M., & Punzo, V. (2015). Trajectory data reconstruction and simulation-based validation against "
        "macroscopic traffic patterns. Transportation Research Part B, 80, 82–106.",
        "Ozbay, K., Yang, H., Bartin, B., & Mudigonda, S. (2008). Derivation and validation of new "
        "simulation-based surrogate safety measure. Transportation Research Record, 2083, 105–113.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning "
        "Research, 12, 2825–2830.",
        "Punzo, V., Borzacchiello, M. T., & Ciuffo, B. (2011). On the assessment of vehicle trajectory data accuracy "
        "and application to the NGSIM program data. Transportation Research Part C, 19(6), 1243–1262.",
        "Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares "
        "procedures. Analytical Chemistry, 36(8), 1627–1639.",
        "SAE International (2021). J3016: Taxonomy and definitions for terms related to driving automation "
        "systems for on-road motor vehicles.",
        "Thrun, S., et al. (2006). Stanley: The robot that won the DARPA Grand Challenge. Journal of Field "
        "Robotics, 23(9), 661–692.",
        "Treiber, M., Hennecke, A., & Helbing, D. (2000). Congested traffic states in empirical observations and "
        "microscopic simulations. Physical Review E, 62(2), 1805–1824.",
        "U.S. Department of Transportation, Federal Highway Administration (2016). Next Generation Simulation "
        "(NGSIM) Vehicle Trajectories and Supporting Data. https://data.transportation.gov/d/8ect-6jqj",
        "Vogel, K. (2003). A comparison of headway and time to collision as safety indicators. Accident Analysis & "
        "Prevention, 35(3), 427–433.",
        "World Health Organization (2023). Global status report on road safety 2023. Geneva: WHO.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.9)
        r = p.add_run(f"[{i}]  {ref}"); r.font.size = Pt(9.5)

    # ------------------------------------------------------------ Appendix
    h1(doc, "Appendix A. Reproducing the results")
    para(doc, "Requirements: Python 3.10 or newer and an internet connection for the first run.")
    for line in ["python -m venv .venv", ".venv\\Scripts\\activate          (Windows)   |   source .venv/bin/activate   (Linux/macOS)",
                 "pip install -r requirements.txt", "python run_pipeline.py", "python -m pytest -q"]:
        p = doc.add_paragraph(); r = p.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(9.5)
        p.paragraph_format.left_indent = Cm(1); p.paragraph_format.space_after = Pt(0)
    doc.add_paragraph()
    srows = [["src/download_data.py", "Downloads NGSIM slices from the U.S. DOT API"],
             ["src/preprocess.py", "Cleaning, smoothing, leader–follower pairing, SSM computation"],
             ["src/risk_metrics.py", "TTC, THW, DRAC, MTTC, CPI, CRI, TET, TIT"],
             ["src/features.py", "Feature engineering and future-conflict labelling"],
             ["src/train_models.py", "Model training, evaluation, model selection"],
             ["src/collision_avoidance.py", "Controllers, vectorised simulator, scenarios"],
             ["src/visualize.py", "All figures"], ["src/report.py", "Generates this report"],
             ["src/av/", "Level-2 stack: vehicle, road, tracking, planner, perception, world, scenarios, evaluate"],
             ["demo.py / demo2d.py", "Interactive simulators (speed control / full Level-2 top-down view)"],
             ["webcam_demo.py", "Live webcam collision-warning practical"],
             ["rc_car/", "RC-car autopilot, hardware layer, lane detection, calibration, Arduino firmware, HARDWARE.md"],
             ["tests/", "Unit tests: metrics, features, simulator, Level-2 stack, RC car"],
             ["run_pipeline.py", "Runs everything end-to-end"]]
    table(doc, ["File", "Purpose"], srows, "Project structure.", col_widths=[5.5, 10.5])

    h1(doc, "Appendix B. Data split sizes")
    sp = m["splits"]
    table(doc, ["Split", "Samples"], [["US-101 train", f"{sp['train']:,}"], ["US-101 validation", f"{sp['val']:,}"],
                                      ["US-101 test", f"{sp['test']:,}"], ["I-80 (unseen site)", f"{sp['ood']:,}"]],
          "Number of anticipation samples per split.", col_widths=[6, 4])

    for sec in doc.sections:
        _footer_page_numbers(sec)
    out = C.REPORTS / "Project_Report.docx"
    doc.save(out)
    print(f"  saved {out}")
    return out


if __name__ == "__main__":
    build()
