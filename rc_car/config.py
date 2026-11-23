"""All RC-car settings in one place. Values measured on your car (servo centre,
ESC range, camera homography, colours) are saved to rc_car/calibration.json by
`python -m rc_car.calibrate` and override the defaults below."""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
CALIB_FILE = HERE / "calibration.json"
LOG_DIR = HERE / "logs"


@dataclass
class RCConfig:
    # ---- vehicle geometry (typical 1:10 chassis) [m]
    wheelbase: float = 0.26
    car_length: float = 0.42
    car_width: float = 0.20
    max_steer_deg: float = 25.0

    # ---- track: two lines of tape on the floor [m]
    lane_width: float = 0.40          # distance between the tape centre lines
    tape_width: float = 0.05

    # ---- speed limits [m/s, m/s²]
    v_cruise: float = 0.8
    v_max: float = 1.2
    a_accel: float = 0.8
    a_brake: float = 1.5              # AEB partial braking
    a_emergency: float = 3.0          # AEB full braking (ESC brake)
    a_lat_max: float = 0.8            # cornering limit -> curve speed

    # ---- following (Intelligent Driver Model at model scale)
    T: float = 0.8                    # time gap [s]
    s0: float = 0.25                  # standstill gap [m]

    # ---- AEB: the SAME time thresholds as the full-size car
    ttc_warn: float = 2.6
    ttc_partial: float = 1.6
    ttc_full: float = 0.9
    drac_min: float = 0.3             # below this required braking it is not an emergency [m/s²]

    # ---- range sensor (front bumper)
    range_max: float = 2.0
    range_min: float = 0.03

    # ---- stop line and traffic light
    stop_wait_s: float = 2.0
    stop_margin: float = 0.12         # stop this far before the line [m]

    # ---- camera / bird's-eye view
    bev_w: int = 240                  # px  (lateral)
    bev_h: int = 320                  # px  (forward)
    bev_px_per_m: float = 200.0       # -> 1.2 m wide, 1.6 m deep
    bev_near: float = 0.10            # bumper to the bottom edge of the bird's-eye view [m]
    homography: list | None = None    # camera image -> bird's-eye view (from calibrate camera)
    lookahead_m: float = 0.40          # pure-pursuit look-ahead (from the bumper)
    fit_range_m: float = 0.9           # fit lane lines only over the nearest part of the view
    lane_hsv_lo: tuple = (0, 0, 170)  # white tape
    lane_hsv_hi: tuple = (179, 70, 255)
    red_hsv_lo: tuple = (0, 120, 90)  # stop line / red light (wraps around hue 180)
    red_hsv_hi: tuple = (10, 255, 255)
    green_hsv_lo: tuple = (45, 100, 70)
    green_hsv_hi: tuple = (85, 255, 255)

    # ---- steering law: "pure_pursuit" (default, robust on tight curves) or "stanley"
    steer_law: str = "pure_pursuit"
    rear_axle_to_bumper: float = 0.34
    k_stanley: float = 1.6
    k_soft: float = 0.3
    steer_smooth: float = 0.5         # low-pass on the steering command (0 = none)

    # ---- loop and safety
    rate_hz: float = 20.0
    watchdog_s: float = 0.3           # stale sensor data -> stop
    lane_lost_s: float = 0.6          # no lane for this long -> controlled stop

    # ---- actuators (servo pulses in microseconds)
    steer_center_us: int = 1500
    steer_range_us: int = 400         # pulse change at full steering lock
    steer_invert: bool = False
    esc_neutral_us: int = 1500
    esc_min_fwd_us: int = 1560        # smallest pulse that makes the car roll
    esc_max_us: int = 1680            # pulse at v_max
    esc_brake_us: int = 1300          # full brake / reverse pulse
    esc_invert: bool = False

    # ---- connections
    serial_port: str = "COM5"
    baud: int = 115200
    camera_index: int = 0
    pi_steer_pin: int = 18
    pi_esc_pin: int = 13
    pi_trig_pin: int = 23
    pi_echo_pin: int = 24

    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls):
        cfg = cls()
        if CALIB_FILE.exists():
            for k, v in json.loads(CALIB_FILE.read_text()).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
        return cfg

    def save(self, **updates):
        data = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}
        data.update(updates)
        CALIB_FILE.write_text(json.dumps(data, indent=2))
        for k, v in updates.items():
            setattr(self, k, v)

    def as_dict(self):
        return asdict(self)
