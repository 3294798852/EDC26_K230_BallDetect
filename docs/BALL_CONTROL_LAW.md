# K230 visual steel-ball state feedback

## Responsibility boundary

K230 owns only the camera-derived ball state and feedback component. It does
not receive the reference trajectory, car acceleration or measured pipe angle,
and it does not calculate a pipe-angle, PWM or motor command.

```text
K230: image -> tracked ball -> x, v, a -> a_fb
MCU:  trajectory/feedforward + a_fb -> plant inversion -> angle inner loop
```

The MCU remains responsible for reference generation, car-acceleration
feedforward, the rolling-ball/pipe model, desired-angle calculation, measured
pipe-angle feedback, actuator limits, fault handling and safe state.

## Coordinate calibration

The configured pipe axis points from the negative image endpoint to the
positive endpoint. Filtered image state is converted to physical state by

```text
x_cm   = dot(pixel - pipe_center, pipe_axis) / pixels_per_cm
v_cm_s = dot(pixel_velocity, pipe_axis) / pixels_per_cm
pixels_per_cm = endpoint_distance_px / pipe_length_cm
```

The endpoint constants in `main.py` are provisional and must be measured after
the camera is fixed. A positive `x`, `v`, `a` and `a_fb` all use this configured
axis direction.

## Visual state feedback

K230 calculates a feedback-equivalent ball acceleration:

```text
a_fb = -(Kp*Db(x, 0.05) + Kd*Db(v, 1.0) + Ka*a)
Db(z, width) = sign(z) * max(abs(z) - width, 0)
```

Default values are:

```text
Kp = 16.0 1/s^2
Kd = 7.2  1/s
Ka = 0.0  dimensionless
|a_fb| <= 250 cm/s^2
```

Physical `x` and `v` are converted directly from the single alpha-beta tracker;
there is no second low-pass stage, avoiding extra phase lag. A continuous soft
deadband removes the first `0.05 cm` of position and `1.0 cm/s` of velocity only
in the feedback calculation. UART continues to publish the tracked physical
state before this deadband. `a` remains a bounded low-pass difference of tracked
velocity and is still published for diagnostics, but its default feedback
coefficient is zero because camera differentiation is noisy at approximately
30 FPS.

K230 exposes three tracking/control states:

- `DETECT`: the current non-edge measurement was accepted; feedback uses the
  updated alpha-beta state.
- `HOLD`: no measurement was accepted this frame, but the last valid state is
  no more than 8 missed frames and 300 ms old; feedback uses the bounded
  constant-velocity prediction and remains valid.
- `LOST`: either hold limit is exceeded, calibration is invalid, or no valid
  state exists; `FEEDBACK_VALID=0` and `a_fb=0`.

Only a valid accepted measurement can arm `HOLD`. Acceleration state is reset
while holding so reacquisition does not create a derivative spike; `Ka` remains
zero by default. `FEEDBACK_SATURATED` reports the 250 cm/s2 clamp.

## MCU composition

For a reference `X`, `X_dot`, `X_ddot`, the MCU can compose the relative-ball
acceleration command as

```text
a_cmd = X_ddot + Kp*X + Kd*X_dot + Ka*X_ddot + a_fb + a_other
```

where `a_other` denotes any MCU-side integral or deliberately designed term.
The MCU then combines `a_cmd` with its own car-acceleration measurement and the
ball/pipe model before calculating the desired pipe angle. Do not add the
K230 feedback twice: `a_fb` already contains the negative measured-state terms.

The contest motion model may be handled entirely on the MCU:

```text
x_ddot = (5/7) * (g*sin(theta) - a_car*cos(theta)) - 0.04*x_dot
```

K230 does not need `a_car` or `theta` to calculate its camera-state feedback.

## Tuning and verification

1. Measure the two pipe endpoints and confirm zero, scale and positive direction.
2. Log K230 `x/v/a/a_fb` while moving the ball manually with actuators disabled.
3. Verify `a_fb` opposes displacement and motion; a sign error must be fixed
   before enabling any MCU actuator output.
4. Tune `Kp/Kd` with `Ka=0`, then introduce only a small `Ka` if acceleration
   feedback improves measured response without amplifying frame jitter.
5. On the MCU, distinguish `DETECT`, bounded `HOLD`, and `LOST` from the flag
   combinations. Reject stale, CRC-invalid or `FEEDBACK_VALID=0` frames.

Host tests validate equations, bounds and serialization only. Camera scale,
actual differentiation noise, KPU timing, UART electrical behavior, MCU model
composition, pipe-angle polarity and actuator safety require target hardware.