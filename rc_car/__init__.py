"""Risk-aware autopilot for a 1:10 RC car.

    python -m rc_car.main --backend sim      # full stack on a simulated car (no hardware)
    python -m rc_car.main --backend serial --port COM5 --view
    python -m rc_car.main --backend pi --view

Layers (same design as the full-size stack in src/av):
    camera -> lane_detection (bird's-eye view, sliding windows, polynomial fit)
    range sensor -> src.estimation.RangeEstimator (distance, closing speed)
    autopilot: Stanley steering + IDM following + MTTC-staged AEB with DRAC gate,
               stop lines, traffic-light cards, lane-lost and sensor watchdogs
    hal: SimCar | SerialCar (laptop/Pi + Arduino) | PiCar (Raspberry Pi GPIO)
"""
