# Risk-Aware Autonomous Driving: Lane Detection, Collision Prediction & Avoidance

An end-to-end autonomous driving research prototype that combines vehicle trajectory analysis, collision-risk estimation, machine learning, lane keeping, and risk-aware control. The system is developed using real-world traffic trajectories from the U.S. Federal Highway Administration's NGSIM dataset and evaluated through closed-loop driving simulations and a 1:10 scale RC-car platform.

The central objective is not simply to detect vehicles or maintain a lane. It is to anticipate potentially dangerous traffic situations, estimate how much time is available to react, and use that information to make safer driving decisions.

**Project scope:** SAE Level 2 driving functionality, evaluated in simulation and through an RC-car software/hardware architecture.

---

## Contents

1. Project overview
2. TL;DR — Key findings
3. Research problem and objectives
4. Research questions
5. System architecture
6. Dataset and preprocessing
7. Collision-risk assessment
8. Machine learning models
9. Risk-aware collision avoidance
10. Autonomous driving and lane management
11. Experimental results
12. Understanding the results
13. RC-car implementation
14. Reproducibility and installation
15. Repository structure
16. Limitations and research gaps
17. Scientific conclusions
18. Future work
19. References

---

## 1. Project overview

Road safety remains a major challenge for autonomous vehicles, particularly in situations involving sudden braking, aggressive lane changes, stopped traffic, and rapidly changing vehicle gaps.

Conventional adaptive cruise control systems generally react to the immediate distance and relative speed of the vehicle ahead. However, a purely reactive strategy may not provide sufficient time to respond to a developing collision scenario.

This project investigates a more predictive approach.

The proposed system combines conventional safety metrics with machine learning-based conflict anticipation. Instead of waiting until a vehicle reaches a critical distance, it estimates the probability of a dangerous interaction occurring within the next three seconds.

The predicted risk is then incorporated into a driving controller that can adjust the following distance, reduce speed, apply graded braking, or trigger emergency braking.

The broader driving stack also includes:

- Lane keeping and curved-road tracking.
- Safe lane-change decisions.
- Obstacle and vehicle interaction.
- Traffic-light and pedestrian handling.
- Risk-aware acceleration and braking.
- A 2-D autonomous driving simulator.
- A camera-based RC-car autopilot architecture.

The same underlying risk-management principles are used across the simulation and RC-car implementations.

### Core pipeline

```text
Real-world traffic trajectories
             |
             v
       Data preprocessing
             |
             v
     Vehicle interaction analysis
             |
             v
       Collision-risk metrics
     (TTC, THW, DRAC, MTTC,
          CPI and CRI)
             |
             v
      Machine learning model
             |
             v
    Future conflict prediction
             |
             v
      Risk-aware controller
             |
             v
    Driving and braking decisions
             |
             v
       Vehicle simulation
             |
             v
     Performance evaluation
```

---

## 2. TL;DR — Key findings

| Research component | Main finding |
|---|---|
| Dataset | FHWA NGSIM vehicle trajectories |
| Training location | US-101, Los Angeles |
| Unseen test location | I-80, Emeryville |
| Sampling frequency | 10 Hz |
| Prediction objective | Anticipate a critical conflict within 3 seconds |
| Deployed prediction model | Multi-Layer Perceptron (MLP) |
| US-101 ROC-AUC | 0.904 |
| US-101 PR-AUC | 0.406 |
| Unseen I-80 ROC-AUC | 0.804 |
| Unseen I-80 PR-AUC | 0.393 |
| Main control strategy | Risk-adaptive following distance and anticipatory braking |
| Simulation evaluation | Emergency braking, cut-ins, stop-and-go traffic and stopped vehicles |
| RC-car evaluation | Camera-based lane detection and ultrasonic distance sensing |

### Main experimental observation

The MLP outperforms the evaluated TTC, MTTC and CRI threshold-based baselines in conflict anticipation.

In closed-loop simulation, the risk-aware controller substantially reduces collision rates in challenging scenarios, particularly emergency braking and stopped-traffic situations.

An important result is the generalisation test: the prediction model is trained using US-101 trajectories and evaluated on the unseen I-80 location.

This provides evidence of transfer across two different traffic environments, although it does not establish generalisation to every road, weather condition or driving environment.

**Honest summary:** Combining physical collision-risk indicators with machine learning-based anticipation and risk-aware control provides a promising approach to safer autonomous driving. The results are encouraging, but the prototype is not a production-ready or road-certified autonomous driving system.

---

## 3. Research problem and objectives

### 3.1 Problem statement

A vehicle travelling in traffic must continuously evaluate the motion of surrounding vehicles and determine whether its current speed, following distance and planned manoeuvres are safe.

Distance alone is not sufficient to determine collision risk. Two vehicles can be separated by the same distance but have very different risk levels depending on their relative velocities and accelerations.

Similarly, a vehicle that appears safe at the current instant may enter a critical situation within a few seconds.

The research problem is therefore:

**Can a machine learning model, supported by established traffic-safety metrics, anticipate developing rear-end conflicts early enough to improve autonomous vehicle control?**

### 3.2 Research objectives

1. Analyse real-world vehicle trajectories using the NGSIM dataset.
2. Calculate established surrogate safety measures for vehicle interactions.
3. Develop machine learning models for short-horizon collision-risk prediction.
4. Compare learned predictions against conventional threshold-based methods.
5. Develop a risk-aware longitudinal controller.
6. Evaluate collision avoidance under different simulated traffic scenarios.
7. Extend the driving architecture to lane keeping, lane changes and urban driving.
8. Implement the same safety principles in a small-scale RC-car platform.

### 3.3 Why this research matters

A conventional controller generally reacts to an observed unsafe condition.

A predictive controller can begin responding before the condition becomes critical.

This distinction is particularly important when:

- A leading vehicle suddenly brakes.
- Another vehicle cuts into the ego vehicle's lane.
- Traffic ahead becomes stationary.
- A lane-change manoeuvre creates a closing-speed conflict.
- The available braking distance becomes insufficient.

The project investigates how prediction and physical safety constraints can work together rather than treating machine learning as a replacement for vehicle dynamics.

---

## 4. Research questions

| ID | Research question | Experimental approach |
|---|---|---|
| RQ1 | Can machine learning anticipate critical traffic conflicts more accurately than conventional safety thresholds? | Compare Logistic Regression, Random Forest, MLP and Gradient Boosting against TTC, MTTC and CRI rules. |
| RQ2 | Does the trained model generalise to an unseen traffic location? | Train on US-101 and evaluate on I-80. |
| RQ3 | Does predictive risk information improve collision avoidance? | Compare ACC, reactive AEB and risk-aware control. |
| RQ4 | Can risk-aware control handle different traffic scenarios? | Test emergency braking, cut-ins, stop-and-go waves and stopped traffic. |
| RQ5 | Can the driving architecture support higher-level autonomous behaviour? | Evaluate lane keeping, overtaking, traffic lights and pedestrian handling. |
| RQ6 | Can the risk-management logic be transferred to a small physical vehicle architecture? | Develop a camera and ultrasonic-sensor-based RC-car autopilot. |

---

## 5. System architecture

The architecture consists of interconnected perception, prediction, planning and control components.

### 5.1 Perception and data processing

The system receives vehicle trajectory information and identifies the interaction between a following vehicle and its leader.

For the NGSIM-based research, vehicle positions, velocities, accelerations and lane information are used.

For the RC-car architecture, the inputs include camera-derived lane information and ultrasonic distance measurements.

### 5.2 Risk assessment

The system calculates physical indicators of collision risk, including:

- Time-to-Collision (TTC)
- Time Headway (THW)
- Deceleration Rate to Avoid a Crash (DRAC)
- Modified Time-to-Collision (MTTC)
- Crash Potential Index (CPI)
- Composite Risk Index (CRI)

These indicators provide interpretable information about the current vehicle interaction.

### 5.3 Risk prediction

The trained MLP estimates the likelihood of a critical conflict occurring within the next three seconds.

The prediction is used as an additional input to the vehicle controller.

### 5.4 Decision and control

The risk-aware controller combines the predicted risk with physical safety conditions.

Depending on the situation, it can:

- Maintain the desired speed.
- Increase the following distance.
- Reduce acceleration.
- Apply partial braking.
- Trigger emergency braking.
- Reject or abort an unsafe lane change.

### 5.5 Vehicle actuation

The final commands are passed to the simulated vehicle or RC-car control interface.

The complete architecture is designed around the relationship between perception, prediction and action.

---

## 6. Dataset and preprocessing

### 6.1 NGSIM dataset

The project uses the Next Generation Simulation (NGSIM) vehicle trajectory dataset, published by the U.S. Department of Transportation and Federal Highway Administration.

NGSIM provides vehicle-level trajectory information extracted from overhead traffic video.

| Property | Description |
|---|---|
| Dataset | NGSIM |
| Source | U.S. DOT / FHWA |
| Data collection | Overhead video tracking |
| Sampling frequency | 10 Hz |
| Training location | US-101, Los Angeles |
| Training date | 15 June 2005 |
| Training window | 07:50–08:05 |
| Training records | Approximately 1.1 million |
| Test location | I-80, Emeryville, California |
| Test date | 13 April 2005 |
| Test window | 16:00–16:14 |
| Test records | Approximately 1.15 million |

Dataset: https://data.transportation.gov/d/8ect-6jqj

### 6.2 Why two locations are used

A model evaluated only on data from its training location may perform well because it has learned location-specific traffic patterns.

To investigate whether the model can transfer to a different traffic environment, US-101 is used for training and I-80 is held out for evaluation.

The I-80 evaluation is therefore an important part of the research design.

It provides a more meaningful generalisation test than randomly mixing observations from both locations.

### 6.3 Preprocessing steps

The preprocessing pipeline performs the following operations:

**1. Unit conversion**

NGSIM positions are converted from feet to metres.

\[
1\text{ ft}=0.3048\text{ m}
\]

**2. Trajectory smoothing**

A Savitzky–Golay filter is applied to vehicle positions.

This helps reduce measurement noise while preserving important changes in the trajectory.

**3. Velocity estimation**

Velocity is derived from the smoothed position:

\[
v(t)=\frac{dx(t)}{dt}
\]

**4. Acceleration estimation**

Acceleration is calculated from the velocity:

\[
a(t)=\frac{dv(t)}{dt}
\]

**5. Boundary handling**

Filter-edge samples are removed because the smoothing operation is less reliable near the boundaries of a trajectory.

**6. Leader-follower pairing**

Each vehicle is paired with its corresponding leader in the same lane.

This makes it possible to calculate relative distance, closing speed and collision-risk indicators.

---

## 7. Collision-risk assessment

This section contains the physical foundation of the research.

### 7.1 Time-to-Collision (TTC)

TTC estimates how long it would take for two vehicles to collide if their current relative velocity remained constant.

For a following vehicle:

\[
TTC=\frac{d}{v_f-v_l}
\]

Where:

- \(d\) = distance between the vehicles.
- \(v_f\) = follower velocity.
- \(v_l\) = leader velocity.

The equation applies when the follower is closing in on the leader.

If the follower is not closing the gap, TTC is not a positive finite collision-time estimate.

**Interpretation:** A smaller TTC indicates a more urgent collision threat.

### 7.2 Time Headway (THW)

Time Headway measures the time gap between two vehicles based on their current separation and the follower's velocity.

\[
THW=\frac{d}{v_f}
\]

A small THW means the follower is travelling close behind the leader relative to its speed.

Unlike TTC, THW does not require the follower to be closing in.

### 7.3 Deceleration Rate to Avoid a Crash (DRAC)

DRAC estimates the deceleration required by the following vehicle to avoid a collision under a simplified constant-deceleration assumption.

For a closing vehicle pair:

\[
DRAC=\frac{(v_f-v_l)^2}{2d}
\]

Where the closing speed is positive.

A higher DRAC indicates that stronger braking is required to avoid a collision.

### 7.4 Modified Time-to-Collision (MTTC)

TTC assumes constant relative velocity.

However, vehicles also accelerate and decelerate.

MTTC incorporates relative acceleration into the collision-time calculation.

The relative motion model is:

\[
d(t)=d_0-v_rt-\frac{1}{2}a_rt^2
\]

Where:

- \(d_0\) = initial separation.
- \(v_r\) = initial closing speed.
- \(a_r\) = relative acceleration under the chosen sign convention.

MTTC is obtained from the physically relevant positive collision-time root.

This provides a more detailed estimate when relative acceleration is important.

### 7.5 Crash Potential Index (CPI)

CPI estimates the probability or proportion of situations in which the required braking exceeds the available braking capability, according to the selected model.

It provides a probabilistic interpretation of collision risk rather than relying only on an instantaneous time measurement.

### 7.6 Composite Risk Index (CRI)

CRI combines multiple safety indicators into a single risk representation.

The project uses four risk levels:

| Level | Interpretation |
|---|---|
| Safe | No currently identified critical condition |
| Caution | Increased risk requiring attention |
| Danger | Significant risk requiring intervention |
| Critical | Severe condition requiring urgent action |

CRI provides an interpretable way to combine several physical risk indicators.

Its thresholds and weights are configuration-dependent.

---

## 8. Machine learning models

### 8.1 Prediction objective

The machine learning task is a binary classification problem.

The model predicts whether a vehicle interaction that is currently considered safe will develop into a critical conflict within the next three seconds.

A critical conflict is defined using the project's surrogate safety thresholds:

\[
TTC<3\text{ s}
\]

or

\[
DRAC>3.35\text{ m/s}^2
\]

The model therefore performs short-horizon risk anticipation rather than simply classifying the current TTC value.

### 8.2 Models evaluated

| Model | Description |
|---|---|
| Logistic Regression | Linear probabilistic classification baseline |
| Random Forest | Ensemble of decision trees |
| MLP Neural Network | Multilayer neural network capable of learning nonlinear relationships |
| Gradient Boosting | Sequential ensemble learning method |
| TTC rule | Conventional TTC threshold baseline |
| MTTC rule | Acceleration-aware collision-time baseline |
| CRI rule | Composite safety-indicator baseline |

### 8.3 Why an MLP?

An MLP can learn nonlinear relationships between traffic variables and future conflict labels.

For example, the same following distance may correspond to different risk levels depending on relative velocity, acceleration and other traffic conditions.

The MLP learns these relationships from the training data.

The selected MLP is the deployed risk-prediction model used by the collision-avoidance controller.

### 8.4 Training and validation strategy

The evaluation uses:

- Vehicle-grouped data splits to reduce leakage between observations from the same vehicle.
- US-101 for model training and development.
- I-80 as a completely held-out test location.
- Conventional threshold rules as interpretable baselines.

The separation of the I-80 location is especially important when interpreting the model's generalisation performance.

---

## 9. Risk-aware collision avoidance

The prediction model alone does not prevent collisions.

Its output must be converted into appropriate control actions.

The project compares three longitudinal control strategies.

### 9.1 Adaptive Cruise Control (ACC)

The baseline controller uses the Intelligent Driver Model (IDM).

IDM calculates acceleration using the desired speed, actual speed, following distance and relative velocity.

A simplified form is:

\[
a=a_{\max}
\left[
1-\left(\frac{v}{v_0}\right)^\delta
-\left(\frac{s^*}{s}\right)^2
\right]
\]

Where:

- \(a\) = commanded acceleration.
- \(a_{\max}\) = maximum acceleration.
- \(v\) = current velocity.
- \(v_0\) = desired velocity.
- \(\delta\) = acceleration exponent.
- \(s\) = actual following distance.
- \(s^*\) = desired dynamic following distance.

The project's ACC braking is limited to 3.5 m/s².

### 9.2 Reactive Automatic Emergency Braking (AEB)

This controller extends ACC with staged emergency braking based on TTC.

It reacts when the collision-time estimate reaches predefined thresholds.

The objective is to improve safety without continuously applying strong braking.

### 9.3 Proposed risk-aware controller

The risk-aware controller combines:

1. Machine learning-based future conflict probability.
2. Risk-adaptive following distance.
3. Pre-emptive graded braking.
4. Emergency braking based on the minimum of TTC and MTTC.
5. Physical braking and actuation constraints.

Instead of waiting for an immediate collision threat, it can begin reducing speed when the predicted risk becomes elevated.

### 9.4 Simulation conditions

The closed-loop simulator includes:

- Sensor noise.
- 0.2-second actuation delay.
- Realistic braking limits.
- Multiple traffic scenarios.
- Monte Carlo variations.

The evaluation includes four families of stress tests, with 500 Monte Carlo runs per family, and 200 real I-80 leader trajectories.

---

## 10. Autonomous driving and lane management

Beyond longitudinal collision avoidance, the project includes a higher-level driving architecture.

### 10.1 Vehicle dynamics

The vehicle is represented using a kinematic bicycle model.

This model approximates the motion of a car through its position, orientation, velocity and steering angle.

It is useful for planning and tracking vehicle movement in a two-dimensional environment.

### 10.2 Lane keeping

The system uses Pure Pursuit and Stanley tracking approaches.

Pure Pursuit calculates a steering command by following a target point ahead of the vehicle.

Stanley control uses lateral tracking error and heading error to correct the vehicle's path.

### 10.3 Lane changes

Lane changes are governed by safety checks.

The system evaluates the available gap and TTC before initiating a manoeuvre.

If the gap becomes unsafe, the lane change can be aborted.

### 10.4 Traffic rules

The driving stack also handles:

- Red and yellow traffic lights.
- Pedestrian yielding.
- Urban turns.
- Emergency braking.
- Curve-dependent speed limits.

These functions are evaluated within the project's simulated Level-2 driving scenarios.

---

## 11. Experimental results

The following tables report the experimental values documented in the repository.

### 11.1 Conflict anticipation performance

The objective is to predict a critical conflict within three seconds from a currently safe state.

| Model | US-101 ROC-AUC | US-101 PR-AUC | I-80 ROC-AUC | I-80 PR-AUC |
|---|---:|---:|---:|---:|
| TTC rule | 0.741 | 0.217 | 0.677 | 0.253 |
| MTTC rule | 0.678 | 0.198 | 0.642 | 0.244 |
| CRI rule | 0.731 | 0.165 | 0.730 | 0.244 |
| **MLP Neural Network** | **0.904** | **0.406** | **0.804** | **0.393** |

**Interpretation**

The MLP achieves the highest ROC-AUC and PR-AUC among the evaluated approaches on both locations.

Its ROC-AUC of 0.904 on US-101 indicates strong discrimination between positive and negative conflict examples.

On the unseen I-80 location, the ROC-AUC decreases to 0.804.

This reduction is expected to be important when studying location transfer, but the model still performs better than the listed threshold-based baselines.

PR-AUC is particularly useful because critical traffic conflicts are less common than safe traffic observations.

The MLP also achieves the highest PR-AUC on the unseen test location.

### 11.2 Closed-loop collision avoidance

Values are reported as collision rate / median minimum TTC.

| Scenario | ACC (IDM) | Reactive AEB | Risk-Aware |
|---|---|---|---|
| Lead emergency braking | 67.8% / 0.07 s | 18.2% / 0.74 s | **0.8% / 1.74 s** |
| Aggressive cut-in | 9.4% / 2.97 s | 0.2% / 2.97 s | 0.2% / 2.97 s |
| Stop-and-go wave | 0.0% / 6.23 s | 0.0% / 6.23 s | 0.0% / 6.41 s |
| Stopped traffic ahead | 28.6% / 2.12 s | 19.4% / 2.12 s | **9.2% / 3.23 s** |
| NGSIM I-80 replay | 0.0% / 2.31 s | 0.0% / 2.31 s | 0.0% / 3.35 s |

The human-driver reference for I-80 replay recorded 0% collision rate and 1.93 s median minimum TTC.

**Interpretation**

The largest improvement occurs in the lead emergency-braking scenario.

The baseline ACC experiences a collision rate of 67.8%, while risk-aware control reduces it to 0.8%.

The stopped-traffic scenario also demonstrates a substantial improvement, reducing collisions from 28.6% with ACC to 9.2%.

In the aggressive cut-in and stop-and-go scenarios, the difference is smaller because the comparison controllers already achieve low or zero collision rates.

This is an important distinction: the risk-aware controller does not provide a measurable advantage in every scenario.

### 11.3 Level-2 autonomous driving

The following results are from 30 unseen random variants per scenario.

| Scenario | Collisions (ours / reactive) | Lane changes per run | Red-light violations | Lane-keeping RMS |
|---|---:|---:|---:|---:|
| Highway: overtake slow truck | 0% / 0% | 2.9 | – | 4.5 cm |
| Highway: aggressive cut-in | 0% / 0% | 1.2 | – | 3.8 cm |
| Highway: sudden stop, lanes blocked | 0% / 47% | 0.2 | – | 1.9 cm |
| Urban: turns, traffic light, pedestrian | 0% / 0% | 0 | 0 | 6.4 cm |
| RC track (1:10 scale) | 0% / 0% | 0 | 0 | 2.4 cm |

**Interpretation**

The most significant difference occurs in the sudden-stop scenario.

The risk-aware planner records zero collisions, whereas the reactive planner records 47%.

The lane-keeping RMS values show that the vehicle can follow the planned path with relatively small lateral tracking errors in the evaluated scenarios.

However, these results are specific to the simulator's tested scenarios and should not be interpreted as a guarantee of real-world driving safety.

### 11.4 RC autopilot results

The RC autopilot was evaluated across six scenarios.

| Metric | Result |
|---|---|
| Scenarios | 6 |
| Collisions | 0 |
| Lane-keeping RMS | 3.6–4.5 cm |
| Lane width | 40 cm |
| Perception | Camera-based lane detection |
| Distance sensing | Ultrasonic sensor |

These results describe the documented RC-autopilot evaluation. They should not be confused with completed physical road testing.

---

## 12. Understanding the results

Several important conclusions can be drawn from the experiments.

### Finding 1: Learned risk prediction outperforms simple threshold rules

TTC, MTTC and CRI are useful indicators, but each relies on a particular mathematical representation of risk.

The MLP can learn combinations of traffic features that are difficult to represent using a single fixed threshold.

The improvement in both ROC-AUC and PR-AUC suggests that this additional information is useful for identifying developing conflicts.

### Finding 2: Location generalisation is possible, but performance decreases

The US-101 results are stronger than the I-80 results.

This demonstrates why evaluation on unseen traffic locations matters.

A model that performs well on its training environment should not automatically be assumed to work equally well elsewhere.

### Finding 3: Predictive control matters most in demanding scenarios

The largest collision-rate improvements occur when the leading vehicle brakes suddenly or when traffic ahead becomes stationary.

These are precisely the situations where additional reaction time can be valuable.

### Finding 4: A good controller should combine learning with physics

The project does not rely exclusively on the MLP.

The controller also uses TTC, MTTC, DRAC-based safety conditions and braking constraints.

This hybrid approach helps retain interpretable physical safety checks while benefiting from learned prediction.

### Finding 5: Not every scenario produces an improvement

In some scenarios, the reactive controller already achieves zero collisions.

The purpose of the proposed approach is therefore not to claim universal superiority, but to improve safety where conventional reactive control is insufficient.

---

## 13. RC-car implementation

The project also contains an RC-car autopilot architecture under `rc_car/`.

It is designed to transfer the driving and safety logic into a small-scale vehicle.

### Components

- Camera for lane detection.
- Ultrasonic sensor for distance measurement.
- Arduino-based actuator interface.
- Steering servo.
- Electronic speed controller (ESC).
- Raspberry Pi support.
- Software simulation backend.

### Operating principle

1. The camera detects the lane boundaries.
2. The system estimates the vehicle's lateral position.
3. The ultrasonic sensor measures the distance to the vehicle ahead.
4. The autopilot calculates steering and speed commands.
5. TTC-based safety checks trigger braking when required.
6. Traffic-light indicators are interpreted using stop-line and light-card detection.
7. Sensor failures and lane-loss conditions trigger safe stopping behaviour.

The software supports simulation, serial hardware control and Raspberry Pi operation.

**Hardware safety:** Read `rc_car/HARDWARE.md` before connecting or operating a physical vehicle. The prototype is not intended for public-road use.

---

## 14. Reproducibility and installation

### Requirements

- Python
- pip
- Git
- Required Python packages from `requirements.txt`

### Clone the repository

```bash
git clone https://github.com/prakhart056-student/Lane-and-Object-Detection-Prototype.git

cd Lane-and-Object-Detection-Prototype
```

### Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the complete research pipeline

```bash
python run_pipeline.py
```

The pipeline performs data downloading, preprocessing, model training, simulation, figure generation and report creation.

### Run tests

```bash
python -m pytest -q
```

The repository documents 26 unit tests.

### Skip data downloading

```bash
python run_pipeline.py --skip-download
```

### Skip autonomous-driving and RC evaluation

```bash
python run_pipeline.py --skip-av
```

---

## 15. Demonstrations

### Longitudinal collision-avoidance simulation

```bash
python demo.py
```

Run a batch evaluation:

```bash
python demo.py --batch 300
```

Save a simulation:

```bash
python demo.py --save demo.gif
```

### Full 2-D autonomous driving simulation

```bash
python demo2d.py
```

Run an urban route:

```bash
python demo2d.py --scenario urban_route
```

Evaluate unseen scenarios:

```bash
python demo2d.py --evaluate 30
```

Save a simulation:

```bash
python demo2d.py --save drive.gif
```

### Webcam collision-warning demonstration

```bash
python webcam_demo.py
```

Colour-based object tracking:

```bash
python webcam_demo.py --color
```

Specify the real object width:

```bash
python webcam_demo.py --width 0.10
```

The webcam demonstration estimates object distance from apparent size and derives approach motion from the observed image sequence.

The physical risk metrics and staged braking logic are demonstrated using the live camera input.

The highway-trained ML model is evaluated in the driving simulator rather than being treated as validated for walking-speed webcam observations.

### RC-car simulation

```bash
python -m rc_car.main --backend sim --view --realtime
```

Hardware operation requires the appropriate wiring, calibration and configuration described in `rc_car/HARDWARE.md`.

---

## 16. Repository structure

```text
Lane-and-Object-Detection-Prototype/
│
├── models/
│   └── risk_model.joblib
│
├── rc_car/
│   ├── main.py
│   ├── autopilot.py
│   ├── lane_detection.py
│   ├── hal.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── arduino/
│   │   └── rc_actuator/
│   └── HARDWARE.md
│
├── reports/
│   ├── Project_Report.docx
│   ├── figures/
│   ├── metrics.json
│   └── simulation_results.json
│
├── src/
│   ├── config.py
│   ├── download_data.py
│   ├── preprocess.py
│   ├── risk_metrics.py
│   ├── features.py
│   ├── train_models.py
│   ├── collision_avoidance.py
│   ├── visualize.py
│   ├── estimation.py
│   ├── report.py
│   └── av/
│       ├── vehicle.py
│       ├── road.py
│       ├── tracking.py
│       ├── longitudinal.py
│       ├── planner.py
│       ├── perception.py
│       ├── agents.py
│       ├── world.py
│       ├── scenarios.py
│       └── evaluate.py
│
├── tests/
│
├── demo.py
├── demo2d.py
├── webcam_demo.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## 17. Limitations and research gaps

Although the results demonstrate the potential of risk-aware driving, several limitations remain.

### 17.1 Rear-end conflict prediction

The ML risk model is designed around rear-end vehicle conflicts.

It does not provide a complete prediction framework for every possible collision type.

### 17.2 Lane-change safety

Lane-change safety is primarily handled through rule-based TTC and gap checks rather than the same trained conflict-prediction model.

### 17.3 Dataset limitations

NGSIM was collected in 2005 and contains trajectory measurement noise.

Its traffic conditions may not represent all modern driving environments.

### 17.4 Surrogate safety measures

Critical conflicts are defined using TTC and DRAC thresholds.

These are surrogate safety indicators, not direct measurements of actual crashes.

Consequently, predicting a critical conflict is not equivalent to predicting that a real collision will occur.

### 17.5 Simulation-to-reality gap

The Level-2 driving functions have been evaluated in a 2-D simulation with scripted traffic.

The RC architecture also has a simulated evaluation.

Full physical RC-car validation still requires completion of the documented hardware test procedure.

### 17.6 Environmental robustness

The current architecture does not establish comprehensive performance under rain, fog, nighttime conditions, poor road markings or complex sensor failures.

### 17.7 Safety certification

The project is a research prototype and is not a road-legal or safety-certified autonomous driving system.

---

## 18. Scientific conclusions

The experiments support three main conclusions.

**First, machine learning can improve short-horizon traffic-conflict anticipation.**

The MLP achieves better ROC-AUC and PR-AUC than the evaluated conventional safety-threshold baselines on both the training-site evaluation and the unseen I-80 evaluation.

**Second, predictive risk information can improve collision-avoidance behaviour.**

The closed-loop experiments show particularly strong improvements in lead emergency braking and stopped-traffic scenarios.

The results suggest that anticipating risk and adjusting vehicle behaviour before a critical condition develops can be more effective than relying exclusively on reactive braking.

**Third, a hybrid architecture is a practical direction for autonomous driving research.**

Physical safety indicators, machine learning predictions, vehicle dynamics and rule-based decision-making each provide different forms of information.

Combining them creates a more comprehensive driving architecture than relying on any single component.

The overall contribution is an integrated research prototype connecting trajectory-based safety analysis, conflict prediction, autonomous driving decisions and a small-scale vehicle implementation.

The results are promising within the evaluated experimental conditions, but broader real-world validation remains necessary.

---

## 19. Future work

Potential research directions include:

1. Extend collision prediction to side-impact and intersection conflicts.
2. Integrate camera-based deep learning object detection with the existing risk-prediction pipeline.
3. Investigate sensor fusion using camera, LiDAR, radar and ultrasonic measurements.
4. Improve generalisation through additional traffic datasets and locations.
5. Evaluate uncertainty-aware collision prediction.
6. Investigate reinforcement learning for high-level driving decisions.
7. Expand physical RC-car experiments under controlled conditions.
8. Study robust lane detection under variable lighting and road conditions.
9. Evaluate performance using hardware-in-the-loop experiments.
10. Investigate real-time deployment constraints and computational efficiency.

These are proposed research directions rather than completed experimental results.

---

## 20. References

1. Federal Highway Administration, Next Generation Simulation (NGSIM) Vehicle Trajectory Dataset. U.S. Department of Transportation. https://data.transportation.gov/d/8ect-6jqj

2. Hayward, J. C. (1972). Near-miss determination through use of a scale of danger.

3. Minderhoud, M. M., and Bovy, P. H. L. (2001). Extended time-to-collision measures for road traffic safety assessment.

4. Ozbay, K., et al. (2008). Modified time-to-collision and traffic conflict analysis.

5. Cunto, F., and Saccomanno, F. F. (2008). Calibration and validation of simulated vehicle safety performance measures.

6. Treiber, M., Hennecke, A., and Helbing, D. (2000). Congested traffic states in empirical observations and microscopic simulations.

7. Punzo, V., Borzacchiello, M. T., and Ciuffo, B. (2011). On the assessment of vehicle trajectory data accuracy and application to the Next Generation Simulation program data.

---

## Project information

**Repository:** https://github.com/prakhart056-student/Lane-and-Object-Detection-Prototype

**Research area:** Autonomous Vehicles, Intelligent Transportation Systems, Machine Learning and Traffic Safety.

**Primary focus:** Predictive collision-risk assessment and risk-aware autonomous driving.

**Project status:** Research prototype with documented simulation results and an RC-car implementation architecture.

