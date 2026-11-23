"""Level-2 autonomous driving stack: lateral + longitudinal control, behaviour planning
and a 2-D multi-agent simulator.

Modules
-------
vehicle       kinematic bicycle model with actuator limits and latency
road          arc-length parameterised paths, multi-lane roads, Frenet projection
tracking      Pure Pursuit / Stanley path tracking and curvature speed limits
longitudinal  risk-aware speed control (ML risk model + MTTC-staged AEB) for one vehicle
agents        traffic vehicles, pedestrians, traffic lights
perception    noisy object list in Frenet coordinates (what the ego's sensors report)
planner       behaviour planner: lane keeping, risk-aware lane changes, lights, pedestrians
world         the closed-loop simulator and its metrics
scenarios     highway / urban / RC-track scenarios, with randomised variants
"""
