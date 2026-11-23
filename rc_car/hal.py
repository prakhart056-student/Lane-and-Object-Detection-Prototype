"""Hardware abstraction layer: one interface, three cars.

    SimCar     simulated 1:10 car on a tape oval: renders bird's-eye camera frames,
               ray-casts a range sensor, models the ESC and the steering servo
    SerialCar  any computer (laptop or Raspberry Pi) + USB webcam + an Arduino running
               rc_car/arduino/rc_actuator (servo, ESC, ultrasonic sensor)
    PiCar      Raspberry Pi driving the servo/ESC/HC-SR04 directly from its GPIO pins

All cars expose  read() -> Sensors  and  command(steer_rad, speed_mps, brake).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Sensors:
    t: float
    frame: np.ndarray | None
    range_m: float | None
    range_t: float | None = None
    v_meas: float | None = None


# ---------------------------------------------------------------------------
# pulse mapping shared by the real cars
# ---------------------------------------------------------------------------
def steer_to_us(cfg, steer_rad):
    frac = float(np.clip(np.degrees(steer_rad) / cfg.max_steer_deg, -1, 1))
    if cfg.steer_invert:
        frac = -frac
    return int(cfg.steer_center_us - frac * cfg.steer_range_us)     # left = shorter pulse on most servos


def speed_to_us(cfg, speed, brake):
    if brake:
        us = cfg.esc_brake_us
    elif speed <= 0.02:
        us = cfg.esc_neutral_us
    else:
        frac = float(np.clip(speed / cfg.v_max, 0, 1))
        us = cfg.esc_min_fwd_us + frac * (cfg.esc_max_us - cfg.esc_min_fwd_us)
    if cfg.esc_invert:
        us = 2 * cfg.esc_neutral_us - us
    return int(us)


class BaseCar:
    def read(self) -> Sensors: ...

    def command(self, steer, speed, brake=False): ...

    def stop(self):
        self.command(0.0, 0.0, brake=True)

    def close(self):
        self.stop()


# ---------------------------------------------------------------------------
# Simulated car
# ---------------------------------------------------------------------------
class SimCar(BaseCar):
    def __init__(self, cfg, scenario="full", seed=0, dt=None):
        from src.av.road import Path
        from src.av.vehicle import KinematicBicycle, VehicleParams
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.dt = dt or 1.0 / cfg.rate_hz
        self.track = Path.from_segments([("straight", 3.0), ("arc", 1.0, 180), ("straight", 3.0), ("arc", 1.0, 180)],
                                        closed=True, ds=0.01)
        p = VehicleParams.rc_car()
        p.wheelbase, p.length, p.width = cfg.wheelbase, cfg.car_length, cfg.car_width
        p.max_steer = np.radians(cfg.max_steer_deg)
        p.accel_delay = 0.0
        x, y, h = self.track.frenet_to_xy(0.3, 0.0)
        self.car = KinematicBicycle(float(x), float(y), float(h), 0.0, p, self.dt)
        self.t = 0.0
        self.v_cmd, self.brake, self._steer_cmd = 0.0, False, 0.0
        self.hint = None
        self.scenario = scenario
        # obstacles: static box and/or a slow leading car (Frenet s, length, speed function)
        self.obstacles = []
        if scenario in ("full", "obstacle"):
            self.obstacles.append(dict(name="box", s=9.5, length=0.15, width=0.25, v=0.0, t_place=0.0, t_remove=24.0))
        if scenario in ("full", "follow"):
            self.obstacles.append(dict(name="lead car", s=1.6, length=0.42, width=0.2, v=0.45,
                                       brake_at=(6.0, 9.0) if scenario == "follow" else (5.0, 7.0), t_remove=10.0
                                       if scenario == "full" else 1e9))
        self.stop_lines = [3.0] if scenario in ("full", "stopline", "light") else []
        # the light card stands 0.4 m beyond the stop line (visible while stopped at the line)
        self.light = dict(s=3.4, period=(6.0, 5.0), offset=0.0) if scenario in ("full", "light") else None
        self.collisions = 0
        self.min_gap = np.inf
        self.lat_err = []
        self.distance = 0.0

    # ---------------- world state
    def light_state(self):
        if self.light is None:
            return None
        g, r = self.light["period"]
        return "green" if (self.t + self.light["offset"]) % (g + r) >= r else "red"

    def _car_s(self):
        cx, cy = self.car.center
        s, d, self.hint = self.track.project(cx, cy, self.hint, window=300)
        return s, d

    def _obstacle_s(self, ob):
        if ob.get("v", 0) == 0:
            return ob["s"]
        return ob["s"] % self.track.length

    def _step_obstacles(self):
        for ob in self.obstacles:
            if ob.get("v", 0) > 0:
                b = ob.get("brake_at")
                v = 0.0 if (b and b[0] <= self.t < b[1]) else ob["v"]
                ob["s"] = (ob["s"] + v * self.dt) % self.track.length
                ob["cur_v"] = v

    def _active(self, ob):
        return ob.get("t_place", 0) <= self.t < ob.get("t_remove", 1e9)

    # ---------------- sensors
    def _render(self):
        c = self.cfg
        img = np.full((c.bev_h, c.bev_w, 3), (58, 60, 64), np.uint8)
        img = cv2.add(img, self.rng.integers(0, 12, img.shape, dtype=np.uint8))
        car = self.car
        bump = car.params.wheelbase + (car.params.length - car.params.wheelbase - car.params.rear_overhang)
        bx, by = car.x + bump * np.cos(car.yaw), car.y + bump * np.sin(car.yaw)
        s_now, _ = self._car_s()
        ss = s_now + np.linspace(-0.5, 2.5, 120)

        def to_px(x, y):
            dx, dy = x - bx, y - by
            f = dx * np.cos(car.yaw) + dy * np.sin(car.yaw)
            lat = -dx * np.sin(car.yaw) + dy * np.cos(car.yaw)
            cols = c.bev_w / 2 - lat * c.bev_px_per_m
            rows = c.bev_h - (f - c.bev_near) * c.bev_px_per_m
            return np.stack([cols, rows], 1).astype(np.int32)

        tape = max(int(c.tape_width * c.bev_px_per_m), 2)
        for d in (c.lane_width / 2, -c.lane_width / 2):
            x, y, _ = self.track.frenet_to_xy(ss, d)
            cv2.polylines(img, [to_px(x, y)], False, (235, 235, 235), tape, cv2.LINE_AA)
        state = self.light_state()
        for s_line in self.stop_lines:
            x, y, _ = self.track.frenet_to_xy(np.array([s_line, s_line]), np.array([-c.lane_width / 2, c.lane_width / 2]))
            cv2.polylines(img, [to_px(x, y)], False, (40, 40, 220), tape, cv2.LINE_AA)
        if self.light is not None:
            sx, sy, _ = self.track.frenet_to_xy(np.array([self.light["s"] - 0.05, self.light["s"] + 0.05,
                                                          self.light["s"] + 0.05, self.light["s"] - 0.05]),
                                                np.array([0.34, 0.34, 0.44, 0.44]))
            col = (40, 200, 60) if state == "green" else (40, 40, 220)
            cv2.fillPoly(img, [to_px(sx, sy)], col)
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _range(self):
        c = self.cfg
        car = self.car
        bump = car.params.wheelbase + (car.params.length - car.params.wheelbase - car.params.rear_overhang)
        bx, by = car.x + bump * np.cos(car.yaw), car.y + bump * np.sin(car.yaw)
        best = None
        for ob in self.obstacles:
            if not self._active(ob):
                continue
            s = self._obstacle_s(ob)
            ss = s + np.linspace(-ob["length"] / 2, ob["length"] / 2, 5)
            for w in np.linspace(-ob["width"] / 2, ob["width"] / 2, 5):
                x, y, _ = self.track.frenet_to_xy(ss, w)
                dx, dy = x - bx, y - by
                f = dx * np.cos(car.yaw) + dy * np.sin(car.yaw)
                lat = -dx * np.sin(car.yaw) + dy * np.cos(car.yaw)
                ok = (f > 0) & (np.abs(lat) < 0.06 + 0.27 * f)       # HC-SR04 beam: about ±15°
                if ok.any():
                    dmin = float(f[ok].min())
                    best = dmin if best is None else min(best, dmin)
        if best is None or best > c.range_max:
            return c.range_max + 0.5
        return max(best + self.rng.normal(0, 0.008), 0.0)

    # ---------------- interface
    def read(self):
        return Sensors(self.t, self._render(), self._range(), self.t, None)

    def command(self, steer, speed, brake=False):
        self.v_cmd, self.brake, self._steer_cmd = speed, brake, steer

    def advance(self):
        """Advance the simulated world by one control period (ESC lag + dynamics)."""
        c = self.cfg
        car = self.car
        target = 0.0 if self.brake else self.v_cmd
        a = (target - car.v) / 0.25
        a = float(np.clip(a, -(c.a_emergency if self.brake else 2.0), 1.5))
        car.step(a, self._steer_cmd)
        self._step_obstacles()
        self.t = round(self.t + self.dt, 6)
        self.distance += car.v * self.dt
        s, d = self._car_s()
        self.lat_err.append(d)
        car_front = s + car.params.length / 2
        for ob in self.obstacles:
            if not self._active(ob):
                continue
            gap = (self._obstacle_s(ob) - ob["length"] / 2 - car_front) % self.track.length
            if gap < 3.0:
                self.min_gap = min(self.min_gap, gap)
            if gap < 0.005 or gap > self.track.length - car.params.length:
                self.collisions += 1


# ---------------------------------------------------------------------------
# Laptop / Pi + Arduino over USB serial
# ---------------------------------------------------------------------------
class SerialCar(BaseCar):
    """Protocol (115200 baud, newline-terminated ASCII):
         host -> Arduino :  C <steer_us> <esc_us>      every control tick
         Arduino -> host :  R <distance_mm>            ~25 Hz from the ultrasonic sensor
       The Arduino returns the ESC to neutral if no command arrives for 300 ms."""

    def __init__(self, cfg, port=None, camera=None):
        import serial  # pyserial
        self.cfg = cfg
        self.ser = serial.Serial(port or cfg.serial_port, cfg.baud, timeout=0.05)
        time.sleep(2.0)                               # Arduino resets when the port opens
        self.cap = cv2.VideoCapture(cfg.camera_index if camera is None else camera)
        self.range_m, self.range_t = None, None
        self._stop = False
        self._lock = threading.Lock()
        self.t0 = time.time()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                buf += self.ser.read(64)
            except Exception:
                break
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                parts = line.decode(errors="ignore").strip().split()
                if len(parts) == 2 and parts[0] == "R":
                    try:
                        mm = float(parts[1])
                    except ValueError:
                        continue
                    with self._lock:
                        self.range_m = mm / 1000.0 if mm > 0 else self.cfg.range_max + 0.5
                        self.range_t = time.time() - self.t0

    def read(self):
        ok, frame = self.cap.read()
        with self._lock:
            r, rt = self.range_m, self.range_t
        return Sensors(time.time() - self.t0, frame if ok else None, r, rt, None)

    def command(self, steer, speed, brake=False):
        self.raw(steer_to_us(self.cfg, steer), speed_to_us(self.cfg, speed, brake))

    def raw(self, steer_us, esc_us):
        """Send pulse widths directly (also used by the calibration tools)."""
        self.ser.write(f"C {int(steer_us)} {int(esc_us)}\n".encode())

    def close(self):
        try:
            for _ in range(3):
                self.stop(); time.sleep(0.05)
        finally:
            self._stop = True
            self.cap.release()
            self.ser.close()


# ---------------------------------------------------------------------------
# Raspberry Pi GPIO
# ---------------------------------------------------------------------------
class PiCar(BaseCar):
    """Servo + ESC on hardware-PWM pins and an HC-SR04 ultrasonic sensor (echo through a
    voltage divider!). Uses gpiozero, with the pigpio daemon for jitter-free pulses if running."""

    def __init__(self, cfg, camera=None):
        from gpiozero import DistanceSensor, Servo
        try:
            from gpiozero.pins.pigpio import PiGPIOFactory
            factory = PiGPIOFactory()
        except Exception:
            factory = None
        self.cfg = cfg
        kw = dict(min_pulse_width=0.001, max_pulse_width=0.002, frame_width=0.02, pin_factory=factory)
        self.steer_servo = Servo(cfg.pi_steer_pin, **kw)
        self.esc = Servo(cfg.pi_esc_pin, **kw)
        self.sonar = DistanceSensor(echo=cfg.pi_echo_pin, trigger=cfg.pi_trig_pin, max_distance=cfg.range_max + 0.5,
                                    pin_factory=factory)
        self.cam = self._open_camera(camera)
        self.t0 = time.time()
        self.esc.value = 0.0                          # neutral: arms most ESCs
        time.sleep(1.0)

    def _open_camera(self, camera):
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
            cam.start()
            return ("picam", cam)
        except Exception:
            return ("cv", cv2.VideoCapture(self.cfg.camera_index if camera is None else camera))

    @staticmethod
    def _us_to_value(us):
        return float(np.clip((us - 1500) / 500.0, -1, 1))

    def read(self):
        kind, cam = self.cam
        if kind == "picam":
            frame = cam.capture_array()
        else:
            ok, frame = cam.read()
            frame = frame if ok else None
        t = time.time() - self.t0
        return Sensors(t, frame, float(self.sonar.distance), t, None)

    def command(self, steer, speed, brake=False):
        self.raw(steer_to_us(self.cfg, steer), speed_to_us(self.cfg, speed, brake))

    def raw(self, steer_us, esc_us):
        self.steer_servo.value = self._us_to_value(steer_us)
        self.esc.value = self._us_to_value(esc_us)

    def close(self):
        self.stop(); time.sleep(0.2)
        self.esc.value = 0.0
        kind, cam = self.cam
        cam.stop() if kind == "picam" else cam.release()


def make_car(backend, cfg, **kw):
    if backend == "sim":
        return SimCar(cfg, scenario=kw.get("scenario", "full"), seed=kw.get("seed", 0))
    if backend == "serial":
        return SerialCar(cfg, port=kw.get("port"), camera=kw.get("camera"))
    if backend == "pi":
        return PiCar(cfg, camera=kw.get("camera"))
    raise ValueError(backend)
