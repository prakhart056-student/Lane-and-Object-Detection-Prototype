# Building and testing the real RC car

The RC car runs **the same decision-making code** that is tested in simulation:
camera lane following, risk-staged emergency braking on a range sensor, following a
car ahead, stop lines and traffic-light cards. Everything below is about getting
that code onto a physical 1:10 car **safely**.

> **Safety first.** Test with the wheels off the ground before every first drive,
> keep the speed low (`--speed 0.4`), keep the E-stop within reach (key **E**, the
> Arduino button, or simply lifting the car), and never test near people's feet,
> stairs or roads.

---

## 1. Two supported setups

| | **A. Pi or laptop + Arduino (recommended)** | **B. Raspberry Pi only** |
|---|---|---|
| Brain | Raspberry Pi 4/5 on the car (or a laptop on a long USB cable for first tests) | Raspberry Pi 4/5 |
| Pulses to servo/ESC | Arduino Nano/Uno (`rc_actuator.ino`) | Pi GPIO (gpiozero + pigpio) |
| Range sensor | HC-SR04 on the Arduino | HC-SR04 on the Pi (echo through a voltage divider) |
| Why choose it | Rock-steady pulses, built-in 300 ms failsafe, E-stop button | Fewer parts |
| Run | `python -m rc_car.main --backend serial --port COM5 --view` | `python -m rc_car.main --backend pi --stream 8080 --autostart` |

## 2. Parts list (approximate prices, they vary by seller)

| Part | Notes | ≈ ₹ |
|---|---|---|
| 1:10 RC car with **steering servo + ESC** (3-wire leads) | Any brushed or brushless "RTR" car; a slow (≤ 20 km/h) one is easier | 3,000–8,000 |
| Raspberry Pi 4 (2 GB+) or Pi 5 + 32 GB microSD | Brain on the car | 4,500–7,000 |
| USB webcam (wide angle) or Pi Camera Module 3 Wide | Lane detection | 800–3,000 |
| Arduino Nano or Uno (setup A) | Servo/ESC pulses + sonar | 300–700 |
| HC-SR04 ultrasonic sensor (or VL53L1X ToF) | Range to the obstacle ahead | 80–600 |
| 5 V 3 A power bank or 5 V 3 A buck converter | Separate supply for the Pi | 600–1,200 |
| Push button, jumper wires, 1 kΩ + 2 kΩ resistors, double-sided tape | E-stop, voltage divider | 150 |
| White insulation/masking tape (≈ 5 cm wide), red tape, red + green card | Track, stop line, traffic light | 300 |

## 3. Wiring

### Setup A: Arduino (`rc_car/arduino/rc_actuator/rc_actuator.ino`)

| From | To Arduino pin |
|---|---|
| Steering servo signal (white/yellow) | **D9** |
| ESC signal (white/yellow) | **D10** |
| ESC / servo ground (black/brown) | **GND** (common ground is essential) |
| HC-SR04 VCC / GND | 5V / GND |
| HC-SR04 TRIG / ECHO | **D2** / **D3** |
| E-stop button | **D4** ↔ GND (latches neutral until reset) |
| USB | to the Pi or laptop (power + data) |

* Do **not** connect the ESC's red (BEC 5 V) wire to the Arduino 5 V if the Arduino
  is USB-powered; leave it for the servo only.
* Brushed motor without an ESC? Set `#define DRIVE_MODE DRIVE_L298N` and wire
  ENA→D5, IN1→D7, IN2→D8 on an L298N/L9110 driver.
* Upload the sketch with the Arduino IDE (board: Nano / Uno, 115200 baud).

### Setup B: Raspberry Pi GPIO

| From | To Pi (BCM numbering) |
|---|---|
| Steering servo signal | **GPIO18** (pin 12) |
| ESC signal | **GPIO13** (pin 33) |
| Servo / ESC ground | any GND pin |
| HC-SR04 VCC / GND | 5V (pin 2) / GND |
| HC-SR04 TRIG | **GPIO23** (pin 16) |
| HC-SR04 ECHO | **GPIO24** (pin 18) **through a divider**: ECHO–1 kΩ–GPIO24–2 kΩ–GND (5 V → 3.3 V) |

On the Pi: `sudo apt install pigpio python3-picamera2 && sudo systemctl enable --now pigpiod`,
then `pip install -r requirements.txt gpiozero pigpio`.

### Power
* Drive battery → ESC (→ motor; the ESC's BEC powers the servo).
* Pi → its **own** 5 V 3 A supply (power bank or buck converter). A motor sharing the
  Pi's supply causes brown-outs and random reboots.
* All grounds (Pi/Arduino, ESC, sensor) connected together.

### Mounting
* Camera at the front, 15–20 cm high, tilted down ~30–40°, so it sees the floor from
  ~10 cm to ~1.5 m ahead. Fix it rigidly: if it moves, redo the camera calibration.
* Sonar at the front bumper, horizontal, 5–10 cm above the floor.

## 4. The track

* Two parallel lines of **white tape**, **40 cm apart** (centre to centre), on a darker
  floor. Change `lane_width` in `rc_car/config.py` if you use another width.
* Curves: radius **≥ 1 m** (measured to the lane centre).
* **Stop line:** red tape straight across the lane.
* **Traffic light:** a red card and a green card (≈ 10 × 10 cm) held upright beside the
  lane, about **40 cm beyond** the stop line, so the camera still sees it while the car
  waits at the line.
* **Obstacle / car ahead:** a box at least 10 cm tall (sonar needs a flat face), or a
  second car driven slowly by hand.

## 5. Calibration (once per car, saved to `rc_car/calibration.json`)

Wheels **off the ground** for steps 1–2.

1. **Steering:** `python -m rc_car.calibrate servo --port COM5`
   A/D until the wheels point straight, 1/3 for full lock, W/S until full lock doesn't
   strain the servo, X to save.
2. **ESC:** `python -m rc_car.calibrate esc --port COM5`
   Find neutral (1), the slowest pulse that turns the wheels (2), a pulse for about
   1.2 m/s (3), and brake (4). If forward is backwards, set `esc_invert: true`.
3. **Camera:** put an A4 sheet flat on the floor, centred, its near edge 30 cm in front
   of the bumper. Run `python -m rc_car.calibrate camera --near 0.30` and click its corners
   in the order near-left, near-right, far-right, far-left; press S. The check window
   should show the sheet as an upright rectangle.
4. **Colours:** `python -m rc_car.calibrate colors`. Click the tape, then the red line,
   then the green card; press S.
5. **Sensor:** `python -m rc_car.calibrate sensor --port COM5`. Readings should match a
   tape measure within ±2 cm.

## 6. Test procedure (do these in order)

| Step | Command | Pass when |
|---|---|---|
| 0. Simulation | `python -m rc_car.main --backend sim --view --realtime` | Laps, stops, no collisions |
| 1. Bench (wheels off ground) | `... --backend serial --port COM5 --view` then SPACE | Wheels turn; steering follows the lane you hold in front of the camera; E stops |
| 2. Lane following, slow | `... --speed 0.4` | Stays between the lines for 3 laps |
| 3. Obstacle braking | place a box on the straight | Warning → brake → stops ≥ 15 cm before the box |
| 4. Following | push a second car slowly ahead, then stop it | Keeps a gap, stops behind it |
| 5. Stop line | red tape, no card | Stops at the line, waits 2 s, continues |
| 6. Traffic light | red card, then swap to green | Waits on red, goes on green |
| 7. Faults | unplug the sonar / cover the camera | `FAULT` / `LANE_LOST` → stops |
| 8. Speed up | `--speed 0.6`, then `0.8` | Repeat 2–4 |

Every run writes `rc_car/logs/run_<time>.csv`: time, state, steering, speed command,
range, gap, closing speed, TTC, AEB stage, lane offset/heading/confidence, stop line,
light. Plot it (or open it in Excel) to show your professor exactly what the car
measured and decided.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `LANE_LOST` all the time | Redo `calibrate colors`; better lighting; matte floor; check the camera tilt |
| Car weaves left-right | Increase `lookahead_m` (0.45–0.55) or `steer_smooth` (0.6); lower `--speed` |
| Cuts corners | Decrease `lookahead_m` (0.3) |
| Steers the wrong way | `steer_invert: true` in calibration.json |
| Brakes for no reason | Sonar sees the floor: mount it higher / angle it up slightly |
| Stops too late | Lower the speed; check `calibrate esc` brake pulse; increase `T` |
| Pi reboots when the motor starts | Separate 5 V supply for the Pi (see Power) |
| ESC beeps / doesn't arm | It needs ~2 s of neutral at power-on: power the ESC after the Arduino/Pi is running |
