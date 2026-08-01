# K230 visual steel-ball state feedback

## Responsibility boundary

K230 owns the camera-derived ball state, the currently applied constant
equilibrium reference `x_ref`, and the resulting feedback component. It does not
receive a moving reference trajectory, car acceleration or measured pipe
angle, and it does not calculate a pipe-angle, PWM or motor command.

```text
K230: preset/MCU x_ref + image -> tracked/calibrated x, v, a
      -> e=x-x_ref -> selected feedback backend -> a_fb
MCU:  car feedforward + a_fb -> plant inversion -> angle inner loop
```

The MCU remains responsible for car-acceleration feedforward, the
rolling-ball/pipe model, desired-angle calculation, measured pipe-angle
feedback, actuator limits, fault handling and safe state. If a moving
trajectory `X(t)/X_dot(t)/X_ddot(t)` is introduced later, its generation and
the validated command path are also MCU responsibilities; the current firmware
implements only a stationary `x_ref`; the MCU may update that constant with the
bounded command in `docs/MCU_UART_PROTOCOL.md`, but K230 does not receive a
moving reference trajectory.

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

`BallStateFeedback.set_reference_position_cm(value)` is the bounded runtime
interface. It accepts only finite values within `+/-12.5 cm`. A valid fixed
10-byte MCU command applies an absolute reference in `0.01 cm` units and changes
the state-frame source to `SOURCE_MCU`. The existing state-frame `x_ref` echo is
the confirmation; there is no separate ACK. An instantaneous step
`Delta x_ref` changes the proportional target by `Kp*Delta x_ref`, so the MCU
must command only references appropriate for the active plant and actuator
limits.
Changing the reference preserves the current integral correction. A full
runtime/filter reset clears that correction together with the filtered state.

## Selectable control backends

`CONTROL_BACKEND` in `main.py` selects one of two implementations behind the
same `update(state, candidate, now_ms)` interface:

- `pid` is the established filtered physical-unit PI-D algorithm and remains
  the default.
- `reference_profile` is a K230/MicroPython port of the supplied upper-computer
  balance-ball feedback algorithm.

Both backends receive the same calibrated state, validity state and constant
`x_ref`; both pass through the common output low-pass filter, dither option,
`+/-140 cm/s2` clamp, snapshot and schema-6 encoder. Selecting a backend does
not alter detector/tracker behavior or the UART frame. In reference mode,
`FEEDBACK_SATURATED` also reports its internal P/dynamic-angle limit, even when
the resulting acceleration remains below the common clamp.

### Backend 1: physical-unit PI-D

The `pid` backend calculates a continuous feedback-equivalent ball acceleration:

```text
e = x - x_ref
e_f = LPF(e, 25 ms), v_f = LPF(v, 50 ms)
r = clamp(1 - abs(e_f)/5, 0, 1) when scheduling is enabled, else 0
Kp_eff = Kp * (1 + 2*r)
Kd_eff = Kd * (1 + 4*r)
I = clamp(I + (-Ki*clamp(Db(e_f, 0.10), +/-1.0) - 0.15*I)*dt, +/-80)
a_raw = -(Kp_eff*Db(e_f, 0.10) + Kd_eff*Db(v_f, 0.35) + Ka*a) + I
n = alternating_sign * pair_uniform(0, 5] when enabled, else 0
a_fb = Sat140(LPF(a_raw, 25 ms) + n)
Db(z, width) = sign(z) * max(abs(z) - width, 0)
```

Default values are:

```text
Kp = 8.0  1/s^2
Kd = 12.0  1/s
Ki = 12.0  1/s^3
Ka = 0.0  dimensionless
|I| <= 80 cm/s^2
|n| <= 5 cm/s^2
|a_fb| <= 140 cm/s^2
```

`Kp`, `Kd` and `Ki` can be adjusted from the LCD `-`/`+` buttons in `0.1`
steps. Each press triggers once; values take effect immediately and are not
saved across reboot. A touch fault is reported on LCD/PC debug and does not stop
camera, control or UART processing.

`FEEDBACK_PD_SCHEDULE_ENABLED` enables a PID-only linear near-target schedule.
At `|e_f| >= FEEDBACK_PD_SCHEDULE_RADIUS_CM` (default 5 cm), effective gains
equal the touch-adjusted base gains. Toward zero error, P rises continuously to
`FEEDBACK_PD_SCHEDULE_KP_MAX_SCALE=3` and D to
`FEEDBACK_PD_SCHEDULE_KD_MAX_SCALE=5`. The schedule uses filtered physical
centimeters, so it is independent of image distortion and detector backend.
Soft deadbands remain unchanged and prevent nonzero feedback at a fully settled
state. Disabling the switch restores constant P/D gains. The reference backend
forces this common switch off because its own approach-dependent D profile is
already part of that algorithm.

UART publishes the tracked physical `x/v/a` before the controller's additional
25/50 ms position/velocity filters and soft deadbands. The controller applies
the first `0.10 cm` of position error and `0.35 cm/s` of velocity as continuous
soft deadbands. Only when both filtered values are inside their deadbands is the
dynamic feedback considered settled.

The integral correction `I` estimates a constant disturbance such as fixed
vehicle/pipe tilt. It integrates at most `+/-1 cm` of soft-deadband position
error, leaks toward zero at `0.15/s`, and is bounded to `+/-80 cm/s2`. It holds
during bounded prediction and when both position and velocity are settled. An
update that would push the total feedback farther into the `140 cm/s2` output
limit is rejected (conditional anti-windup). Invalid feedback or a full
loss/filter reset clears the integral and filtered control state. The LCD and
PC debug output show the current integral correction and state.

The optional `n` term is zero-mean uniform feedback dither for overcoming
static friction. `FEEDBACK_DITHER_ENABLED` enables it and
`FEEDBACK_DITHER_AMPLITUDE_CM_S2` sets its symmetric bound, defaulting to
`+/-5 cm/s2`. Each random magnitude is held for two valid frames, while its
sign flips every valid `DETECT/HOLD` frame. Adjacent valid samples are therefore
equal and opposite within each pair, and direction still alternates across pair
boundaries. The term is applied after the 25 ms output filter and before the
final `+/-140 cm/s2` clamp, so the filter does not attenuate it and saturation
remains explicit. `LOST` output remains zero and does not consume a sign step;
the next valid frame continues the alternation. The generator is a fixed-memory
32-bit integer xorshift implementation;
it adds constant-time work and no allocation or external module dependency.
Disabling the option exactly restores the deterministic PI-D output path.

### Backend 2: supplied reference profile

The source controller works in pixels and produces a pipe-angle correction.
The port keeps source-domain parameters in `REFERENCE_CONTROL_SETTINGS` and
converts them once at construction:

```text
s_px = 383 px / 25 cm = 15.32 px/cm
e_src = -(x - x_ref) * s_px
v_src = v * s_px
gain_cm = gain_src * s_px * (5/7)*g*pi/180
a(theta) = (5/7)*g*sin(theta), g=980.665 cm/s2
```

The leading minus sign converts this project's `x-x_ref` convention to the
source controller's `target-measurement` convention. A positive physical
displacement therefore still produces negative restoring acceleration. The
final `sin` conversion is used instead of its small-angle approximation when
forming the output.

The rolling profile computes bounded P/I/D angle terms. When the ball moves
toward the reference, D scale is `0.2` outside 30 source pixels, interpolates
linearly from `0.2` to `1.0` over 30 to 8 pixels, and is full inside 8 pixels.
Motion away from the reference always uses full D damping. Integral action is
allowed only inside the configured error and velocity gates.

After 0.35 s of direct low-speed observations, the backend selects a stronger
stationary P profile. If error also exceeds the configured threshold, a bounded
`3..5 deg` breakaway bias ramps toward the target. The separate slow trim learns
a persistent low-speed offset up to `+/-4 deg`. These conservative defaults come
from the reference implementation's reusable class defaults; its production
`18..26 deg` motor-specific boost is deliberately not copied into the K230
acceleration contract. During `HOLD`, profile state, integral, trim and
breakaway bias freeze rather than learning from extrapolated data.

The source project's car feedforward, motor-neutral angle, command-rate limiter,
endpoint handling, hard pulse, motor feedback and actuator limits are not part
of this backend. Those functions remain on the MCU. `reference_profile` still
publishes only `a_fb`, never a servo or pipe-angle command.

LCD and PC debug identify the selected backend. Reference diagnostics include
`ROLLING/STATIONARY/BREAKAWAY`, angle correction, D scale, breakaway angle, trim
angle and whether adaptation is frozen. Before enabling an actuator, verify
feedback sign with the plant constrained and retune the source-domain settings
for this camera, pipe friction and MCU angle loop.

The `80 cm/s2` integral limit corresponds to roughly `6.6 deg` of static
pipe-angle authority under the documented `5/7*g` rolling model. Larger
installation errors should be fixed mechanically rather than hidden by raising
this limit. `a` remains a bounded low-pass velocity difference for diagnostics;
`Ka=0` by default because camera-derived acceleration is noise-sensitive.

The tracker still reports `QUIET/PENDING/MOVING` for diagnostics. Those labels
do not switch the default PI-D equation; the reference backend instead owns its
profile state using physical position/velocity and measurement freshness.
Schema 6 remains exactly 64 bytes.

The reference changes only the position-error term. UART still publishes the
absolute calibrated `x` and continues to publish absolute `v/a`. Schema 6 also
publishes the active `x_ref`, `e`, reserved `v_ref/a_ref` fields and reference
source/status flags in a fixed 64-byte frame. `a_fb` already incorporates
`x_ref` and the integral correction; transmitting the reference lets the
MCU log and cross-check the K230 calculation but does not transfer ownership of
the same feedback term. Schema 6 has no separate `Ki` or integral-state field:
the existing
`a_fb` field is the complete value the MCU should compose into its plant model.
The LCD and PC debug output expose `Ki`, the integral correction and its state.

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
Do not calculate `-Kp*(x-x_ref)-Kd*v` or integrate the same position error again
on the MCU: both the fast state feedback and stationary-error correction are
already inside `a_fb`. For a later moving trajectory, either move the full reference state
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
4. Start with `Ki=0` and tune `Kp/Kd` for fast, damped return. Then raise `Ki`
   gradually until a constant installation tilt is rejected without slow
   oscillation. The touch panel changes all three gains live in `0.1` steps.
5. If integral compensation overshoots, reduce `Ki` before changing the
   `+/-80 cm/s2` limit or leak rate. If a removed tilt leaves a long opposite
   offset, increase leak only after checking feedback sign and actuator dead
   zone. Keep `Ka=0` unless acceleration feedback proves useful without noise.
6. With a static ball, verify the screen remains mostly `MQ`; an occasional
   `MP1W` or `MP1C` must return without creating a velocity/feedback spike.
7. On the MCU, distinguish `DETECT`, bounded `HOLD`, and `LOST` from the flag
   combinations. Reject stale, CRC-invalid or `FEEDBACK_VALID=0` frames.

Host tests validate equations, bounds and serialization only. Camera scale,
actual differentiation noise, selected-detector timing, UART electrical behavior, MCU model
composition, pipe-angle polarity and actuator safety require target hardware.
