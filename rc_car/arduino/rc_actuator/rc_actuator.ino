/*
  rc_actuator.ino  -  low-level board for the risk-aware RC car
  ----------------------------------------------------------------
  Runs on an Arduino Uno / Nano. The laptop or Raspberry Pi runs the autopilot
  (python -m rc_car.main --backend serial) and talks to this sketch over USB.

  Host -> Arduino :  "C <steer_us> <esc_us>\n"   (servo pulse widths, 1000-2000 us)
  Arduino -> host :  "R <distance_mm>\n"          (~25 Hz, 0 = no echo / out of range)

  Safety:
    * FAILSAFE: if no valid command arrives for 300 ms -> ESC neutral, steering centred
    * E-STOP button (to GND on pin 4): latches neutral until the board is reset
    * pulses are clamped to safe limits below

  Drive options (pick one with DRIVE_MODE):
    DRIVE_ESC   hobby ESC with a 3-wire servo lead (most RC cars)
    DRIVE_L298N brushed DC motor through an L298N / L9110 H-bridge
*/
#include <Servo.h>

#define DRIVE_ESC   1
#define DRIVE_L298N 2
#define DRIVE_MODE  DRIVE_ESC

// ---------------- pins
const int PIN_STEER = 9;       // steering servo signal
const int PIN_ESC   = 10;      // ESC signal (DRIVE_ESC)
const int PIN_ENA   = 5;       // L298N ENA (PWM)  (DRIVE_L298N)
const int PIN_IN1   = 7;       // L298N IN1
const int PIN_IN2   = 8;       // L298N IN2
const int PIN_TRIG  = 2;       // HC-SR04 trigger
const int PIN_ECHO  = 3;       // HC-SR04 echo
const int PIN_ESTOP = 4;       // push button to GND (optional)
const int PIN_LED   = 13;

// ---------------- limits
const int STEER_MIN_US = 1100, STEER_MAX_US = 1900;
const int ESC_MIN_US = 1250, ESC_MAX_US = 1750, ESC_NEUTRAL_US = 1500;
const unsigned long FAILSAFE_MS = 300;
const unsigned long SONAR_PERIOD_MS = 40;
const unsigned long SONAR_TIMEOUT_US = 15000;     // ~2.5 m

Servo steer;
#if DRIVE_MODE == DRIVE_ESC
Servo esc;
#endif

char buf[40];
uint8_t blen = 0;
unsigned long lastCmd = 0, lastSonar = 0;
bool estop = false;

void drive(int us) {
  us = constrain(us, ESC_MIN_US, ESC_MAX_US);
#if DRIVE_MODE == DRIVE_ESC
  esc.writeMicroseconds(us);
#else
  int v = us - ESC_NEUTRAL_US;                  // -250 .. +250
  if (abs(v) < 20) { analogWrite(PIN_ENA, 0); digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, LOW); return; }
  bool fwd = v > 0;
  digitalWrite(PIN_IN1, fwd ? HIGH : LOW);
  digitalWrite(PIN_IN2, fwd ? LOW : HIGH);
  analogWrite(PIN_ENA, map(abs(v), 20, 250, 70, 255));
#endif
}

void neutral() {
  drive(ESC_NEUTRAL_US);
  steer.writeMicroseconds((STEER_MIN_US + STEER_MAX_US) / 2);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_ESTOP, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  steer.attach(PIN_STEER);
#if DRIVE_MODE == DRIVE_ESC
  esc.attach(PIN_ESC);
#else
  pinMode(PIN_ENA, OUTPUT); pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT);
#endif
  neutral();
  delay(2000);                                   // let the ESC arm at neutral
}

long readSonarMm() {
  digitalWrite(PIN_TRIG, LOW);  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long us = pulseIn(PIN_ECHO, HIGH, SONAR_TIMEOUT_US);
  if (us == 0) return 0;
  return (long)(us * 0.1715);                    // speed of sound / 2, in mm per us
}

void handleLine(char *line) {
  if (line[0] != 'C') return;
  int s, e;
  if (sscanf(line + 1, "%d %d", &s, &e) != 2) return;
  lastCmd = millis();
  if (estop) return;
  steer.writeMicroseconds(constrain(s, STEER_MIN_US, STEER_MAX_US));
  drive(e);
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (blen) { buf[blen] = 0; handleLine(buf); blen = 0; }
    } else if (blen < sizeof(buf) - 1) {
      buf[blen++] = ch;
    }
  }
  if (digitalRead(PIN_ESTOP) == LOW) estop = true;
  if (estop || millis() - lastCmd > FAILSAFE_MS) {
    neutral();
    digitalWrite(PIN_LED, (millis() / 150) % 2);  // fast blink = not driving
  } else {
    digitalWrite(PIN_LED, HIGH);
  }
  if (millis() - lastSonar >= SONAR_PERIOD_MS) {
    lastSonar = millis();
    Serial.print("R ");
    Serial.println(readSonarMm());
  }
}
