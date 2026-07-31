# K230 visual steel-ball state feedback

## Responsibility boundary

K230 owns the camera-derived ball state, the currently applied constant
equilibrium preset `x_ref`, and the resulting feedback component. It does not
receive a moving reference trajectory, car acceleration or measured pipe
angle, and it does not calculate a pipe-angle, PWM or motor command.

```text
K230: preset x_ref + image -> tracked/calibrated x, v, a
      -> e=x-x_ref -> state feedback a_fb
MCU:  car feedforward + a_fb -> plant inversion -> angle inner loop
```

The MCU remains responsible for car-acceleration feedforward, the
rolling-ball/pipe model, desired-angle calculation, measured pipe-angle
feedback, actuator limits, fault handling and safe state. If a moving
trajectory `X(t)/X_dot(t)/X_ddot(t)` is introduced later, its generation and
the validated command path are also MCU responsibilities; the current firmware
implements only the K230-side stationary `x_ref` preset and does not receive
that trajectory.

## Coordinate calibration

Detection and tracking remain entirely in the 320x320 pixel domain. Between
the tracker and this control layer, `ball_calibration.py` converts the filtered
state into physical coordinates with image `(160,160)` as `(0,0)` cm.

`X_CALIBRATION_MODE` in `main.py` selects the physical conversion.
`"measured_distortion"` is the default and uses this observed correspondence
between the old displayed coordinate and a physical ruler:

```text
old:    0, 2.5, 5.0, 7.5, 10.0, 12.5 cm
actual: 0, 3.0, 5.8, 8.5, 10.7, 12.5 cm
```

The negative side is mirrored. The calibration anchors were converted to pixel
positions with the old table and fitted offline using monotone PCHIP. This
shape-preserving piecewise cubic passes every anchor without a global high-order
polynomial's overshoot risk. Its 320 integer-pixel results are stored as signed
integers scaled by 10000. Runtime conversion does not evaluate the cubic: a
subpixel X uses linear interpolation between adjacent table entries, and X
velocity uses the same local cm/pixel slope.

`"pinhole"` preserves the original model as a selectable fallback. Given
camera distance `D=15 cm`, half-view distance `L=12.5 cm`, and
`theta=atan(L/D)=39.8056 deg`, that table was generated from

```text
x_cm(u) = D * tan((u - 160) / 160 * theta)
```

Y is used only as a diagnostic transverse coordinate and is linearized with
the selected X calibration's center slope:

```text
measured distortion: 0.0789 cm/pixel
original pinhole:     0.0651317134 cm/pixel
```

Positive physical X follows increasing image X; positive Y follows increasing
image Y. Position feedback, velocity feedback, acceleration differentiation
and `a_fb` all use the calibrated X values. Pipe endpoint constants in
`main.py` remain image-only detector/tracker geometry and no longer set the
physical scale.

## Visual state feedback

`FEEDBACK_REFERENCE_POSITION_CM` in `main.py` defines the constant equilibrium
point as an absolute calibrated X coordinate. It currently defaults to
`0.0 cm`. Positive and negative values move the balance point right or left in
the same sign convention as transmitted `x`.

`BallStateFeedback.set_reference_position_cm(value)` is a bounded runtime
interface for future reference-selection logic. It accepts only finite values
within `+/-12.5 cm`. The current application does not parse an MCU command and
does not silently change the reference at runtime. A later command interface
should rate-limit or ramp large reference changes, because an instantaneous
step `Delta x_ref` produces an immediate feedback step `Kp*Delta x_ref`.

K230 calculates a feedback-equivalent ball acceleration:

```text
e = x - x_ref
a_fb = -(Kp*Db(e, 0.12) + Kd*Db(v, 0.40) + Ka*a)
Db(z, width) = sign(z) * max(abs(z) - width, 0)
```

Default values are:

```text
Kp = 8.0  1/s^2
Kd = 7.5  1/s
Ka = 0.0  dimensionless
|a_fb| <= 140 cm/s^2
```

Physical `x` and `v` are converted directly from the single alpha-beta tracker;
there is no unconditional second low-pass stage, avoiding extra motion lag. A
continuous soft deadband removes the first `0.12 cm` of position and `0.40 cm/s`
of velocity only in the feedback calculation. UART continues to publish the
tracked physical state before this deadband. `a` remains a bounded low-pass
difference of tracked velocity and is still published for diagnostics, but its
default feedback coefficient is zero because camera differentiation remains
noise-sensitive at the measured camera frame rate.

The tracker also reports an internal `QUIET/PENDING/MOVING` state. Both
`PENDING` and confirmed `MOVING` velocity enter the `Kd` term, preserving
braking while a stop or reversal is still being confirmed. Only `QUIET` removes
the velocity term. A transition from `QUIET` requires at least two reliable
same-direction center measurements; low-quality, single-support, spread or
closely competing Hough measurements require three. The first measurement never
creates a new velocity, while the confirming frame estimates motion from the
pre-transition anchor. A moving track must then observe four consecutive axial
center steps no larger than `0.75 px` before it can return to `QUIET`. Confirmed motion uses
the raw feedback target immediately; sustained non-moving output approaches its
target with gain `0.35`. The configured hysteresis is zero, so the filtered
feedback continues converging to zero rather than sticking at a residual value.
Transition away from `MOVING`, filter reset and invalid feedback reset the
stabilization state. This changes only `a_fb`; UART still publishes the tracked
physical `x/v/a`, and schema 6 remains exactly 64 bytes.

The reference changes only the position-error term. UART still publishes the
absolute calibrated `x` and continues to publish absolute `v/a`. Schema 6 also
publishes the active `x_ref`, `e`, reserved `v_ref/a_ref` fields and reference
source/status flags in a fixed 64-byte frame. `a_fb` already incorporates
`x_ref`; transmitting the reference lets the MCU log and cross-check the K230
calculation but does not transfer ownership of the same feedback term.

The displayed `e` is a control error and is forced to zero when feedback is
invalid; do not interpret it as a valid measurement in `LOST` state.

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
zero by default. `FEEDBACK_SATURATED` reports the 140 cm/s2 clamp.

## Complete MCU control signal flow

The UART frame carries absolute `x/v/a` and the already-computed `a_fb`. The
current constant-reference balance task uses

```text
a_ball_cmd = a_fb + a_extra
```

where `a_extra` is zero unless the MCU deliberately owns an additional term.
Do not calculate `-Kp*(x-x_ref)-Kd*v` again on the MCU: it is already inside
`a_fb`. For a later moving trajectory, either move the full reference state
`X/X_dot/X_ddot` to one controller or define a richer command explicitly;
updating only `x_ref` is a stationary-equilibrium interface.

The solid-ball pure-rolling model is

```text
x_ddot = k_roll*(g*sin(theta) - a_car*cos(theta)) - b*x_dot
k_roll = 5/7
b = 0.04 1/s
```

Use one unit system throughout the MCU calculation. With UART values in
centimeters, use `g = 980.665 cm/s2` and express `a_car` and `a_ball_cmd` in
`cm/s2`. The nonlinear pipe-angle inversion is

```text
c   = (a_ball_cmd + b*v) / k_roll
r   = sqrt(g*g + a_car*a_car)
phi = atan2(a_car, g)

require |c| <= r
theta_ff = phi + asin(clamp(c/r, -1, +1))
```

The MCU then applies its known mechanical angle/servo limits and closes the
measured pipe-angle inner loop around `theta_ff`. The sign of image X, vehicle
acceleration, measured pipe angle and actuator command must be verified on the
disabled or mechanically constrained plant before enabling motion.

Recommended MCU processing order for every frame:

1. Find sync, require 64 bytes, envelope version 1, message `0x02`, schema 6,
   kind 5, payload length 48 and valid CRC16 over frame offsets `2..61`.
2. Reject duplicate/out-of-order data and enforce an MCU arrival timeout in
   addition to K230 `measurement_age_ms`.
3. Decode `DETECT`, `HOLD` and `LOST`. Use `a_fb` only when
   `FEEDBACK_VALID=1`; treat `FEEDBACK_SATURATED=1` as a control-limit warning.
4. Require `REFERENCE_VALID`, check exactly one supported reference-source bit,
   require reserved bytes to be zero, and when feedback is valid verify
   `e ~= x-x_ref` within fixed-point quantization tolerance.
5. Form `a_ball_cmd`. For the current balance mode this is normally `a_fb`.
6. Add the damping compensation `b*v`, combine the latest valid `a_car`, and
   perform the nonlinear inversion above.
7. Check inverse feasibility and apply angle/rate/actuator limits. Never pass
   a stale, invalid, NaN or infeasible command to the actuator.
8. Run the faster measured-angle inner loop independently of UART arrival. If
   vision becomes stale or `LOST`, transition through the defined safe policy
   instead of holding an unlimited active command.

K230 does not need `a_car` or `theta` to calculate its camera-state feedback;
the MCU has the information required for plant inversion and safety arbitration.

## Tuning and verification

1. Confirm the selected calibration, reference `x_ref`, image center and positive
   directions on the actual fixed camera.
2. Log K230 `x/v/a/a_fb`, internal motion state and raw/stabilized feedback while
   moving the ball manually with actuators disabled.
3. Verify `a_fb` opposes displacement and motion; a sign error must be fixed
   before enabling any MCU actuator output.
4. Tune `Kp/Kd` with `Ka=0`, then introduce only a small `Ka` if acceleration
   feedback improves measured response without amplifying frame jitter.
5. With a static ball, verify the screen remains mostly `MQ`; an occasional
   `MP1W` or `MP1C` must return without creating a velocity/feedback spike.
6. On the MCU, distinguish `DETECT`, bounded `HOLD`, and `LOST` from the flag
   combinations. Reject stale, CRC-invalid or `FEEDBACK_VALID=0` frames.

Host tests validate equations, bounds and serialization only. Camera scale,
actual differentiation noise, selected-detector timing, UART electrical behavior, MCU model
composition, pipe-angle polarity and actuator safety require target hardware.
