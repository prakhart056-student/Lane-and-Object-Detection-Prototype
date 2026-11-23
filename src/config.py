"""Central configuration for the Risk-Aware Collision Avoidance project.

All paths, dataset windows, physical constants and model hyper-parameters live
here so that every experiment in the report is reproducible from one place.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset: FHWA NGSIM Vehicle Trajectories (U.S. DOT open data portal)
# https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj
# --------------------------------------------------------------------------
NGSIM_API = "https://data.transportation.gov/resource/8ect-6jqj.csv"
PAGE_SIZE = 50_000

# Each study site is sampled at 10 Hz. We take a 14-minute window per site (one NGSIM recording period),
# skipping the first 60 s of recording (vehicles still entering the section).
SITES = {
    # Training / validation site: Hollywood Freeway, Los Angeles (June 15 2005)
    "us-101": {"t_start_ms": 1118846979700 + 60_000, "duration_ms": 840_000},
    # Out-of-distribution test site: Interstate 80, Emeryville CA (April 13 2005)
    "i-80": {"t_start_ms": 1113433135300 + 60_000, "duration_ms": 840_000},
}

# --------------------------------------------------------------------------
# Physics / unit conversion
# --------------------------------------------------------------------------
FT_TO_M = 0.3048
DT = 0.1                    # NGSIM sampling period [s]
SMOOTH_WINDOW = 11          # Savitzky-Golay window (1.1 s) for position smoothing
SMOOTH_POLY = 3
EDGE_TRIM = 5               # frames trimmed at each end of a trajectory segment

# --------------------------------------------------------------------------
# Surrogate-safety thresholds (literature defaults)
# --------------------------------------------------------------------------
TTC_CRITICAL = 3.0          # s   - Time-to-Collision (Hayward 1972; Vogel 2003)
THW_CRITICAL = 1.0          # s   - Time headway
DRAC_CRITICAL = 3.35        # m/s^2 - Deceleration Rate to Avoid Crash (Archer, 2005)
MADR = 8.0                  # m/s^2 - Max Available Deceleration Rate (dry asphalt)
TTC_CAP = 20.0              # s   - TTC values above this are "no conflict"

# Risk prediction horizon: predict whether a critical conflict occurs within
# the next HORIZON seconds, using only information available *now*.
HORIZON_S = 3.0
HISTORY_S = 1.0             # history length used for temporal features

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Collision avoidance controller parameters
# --------------------------------------------------------------------------
IDM = dict(v0=30.0, T=1.5, a_max=1.5, b_comf=2.0, s0=2.0, delta=4)
AEB = dict(ttc_warn=2.6, ttc_partial=1.6, ttc_full=0.9,
           a_partial=-4.0, a_full=-8.0)
VEHICLE_A_MIN = -8.0        # physical braking limit [m/s^2]
ACC_A_MIN = -3.5            # ACC comfort-braking authority (ISO 15622)
RISK_A_MIN = -5.0           # max pre-emptive braking of the risk-aware layer
VEHICLE_A_MAX = 2.5
REACTION_DELAY_S = 0.2      # actuator / perception latency
