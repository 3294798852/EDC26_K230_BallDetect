# K230 steel-ball visual feedback UART protocol

## Scope and physical link

This standalone application publishes camera-derived image state, physical
ball state and a bounded feedback component. It does not use the legacy task
handshake and does not require MCU-to-K230 control data.

Physical fields use the compile-time `X_CALIBRATION_MODE` selected in
`main.py`. The default `measured_distortion` table corrects the installed
camera from physical-ruler anchors; `pinhole` restores the original ideal
camera table. Both modes keep the same schema, field scales and units, so the
MCU decoder does not branch on calibration mode. Record the selected mode
during system calibration; the LCD physical-state row and PC debug identify it.

| Item | Value |
| --- | --- |
| K230 peripheral | UART1 |
| K230 pins | IO9 TX, IO10 RX (RX currently unused by this application) |
| MCU connection | K230 IO9 TX -> MCU RX; common ground |
| Format | 512000 baud, 8N1, no flow control |
| Levels | 3.3 V TTL |
| Direction | K230 to MCU |
| Frame size | fixed 64 bytes |

K230 never waits for an MCU response. UART writes use a bounded two-frame,
latest-wins queue, so a disconnected receiver cannot block camera processing or
cause unbounded memory growth.

## Envelope

All multibyte fields are little-endian. CRC16-CCITT-FALSE uses polynomial
`0x1021`, initial value `0xFFFF`, no reflection/final xor. It covers offsets
`2..61`; CRC is stored at `62..63`.

| Offset | Size | Field | Value |
| ---: | ---: | --- | --- |
| 0 | 1 | sync 0 | `0xA5` |
| 1 | 1 | sync 1 | `0x5A` |
| 2 | 1 | envelope version | `1` |
| 3 | 1 | message type | `0x02` (`VISION_EVENT`) |
| 4 | 1 | payload schema | `6` |
| 5 | 1 | transport flags | `0x01` (`REALTIME`) |
| 6..7 | 2 | payload length | `48` |
| 8..9 | 2 | sequence | `u16`, wraps modulo 65536 |
| 10..13 | 4 | timestamp | K230 milliseconds, `u32` |
| 14..61 | 48 | payload | schema-6 kind-5 visual feedback |
| 62..63 | 2 | CRC16 | over offsets `2..61` |

## Payload schema 6, kind 5

| Payload offset | Frame offset | Size | Type / scale | Field |
| ---: | ---: | ---: | --- | --- |
| 0 | 14 | 1 | `u8` | kind `5` |
| 1 | 15 | 1 | `u8` | flags |
| 2 | 16 | 1 | `u8` | selected confidence x255 |
| 3 | 17 | 1 | `u8` | consecutive missed/rejected frames |
| 4..5 | 18..19 | 2 | `u16` px | raw selected center X |
| 6..7 | 20..21 | 2 | `u16` px | raw selected center Y |
| 8..9 | 22..23 | 2 | `i16`, /100 cm | calibrated physical X |
| 10..11 | 24..25 | 2 | `i16`, /100 cm | calibrated physical Y |
| 12..13 | 26..27 | 2 | `i16`, /10 cm/s | calibrated physical X velocity |
| 14..15 | 28..29 | 2 | `i16`, /10 cm/s | calibrated physical Y velocity |
| 16..17 | 30..31 | 2 | `i16`, /100 | axial physical position `x`, cm; repeats offset 8 |
| 18..19 | 32..33 | 2 | `i16`, /10 | axial physical velocity `v`, cm/s; repeats offset 12 |
| 20..21 | 34..35 | 2 | `i16`, /10 | visual acceleration `a`, cm/s2 |
| 22..23 | 36..37 | 2 | `i16`, /10 | feedback component `a_fb`, cm/s2 |
| 24..25 | 38..39 | 2 | `u16` ms | total selected-detector run time; YOLO, ROI and Hough share this field |
| 26..27 | 40..41 | 2 | `u16` ms | accepted-measurement age; `0xFFFF` unavailable |
| 28..29 | 42..43 | 2 | `u16`, /100 | `Kp` |
| 30 | 44 | 1 | `u8`, /10 | `Kd` |
| 31 | 45 | 1 | `u8`, /100 | `Ka` |
| 32..33 | 46..47 | 2 | `i16`, /100 cm | active absolute reference position `x_ref` |
| 34..35 | 48..49 | 2 | `i16`, /100 cm | position error `e=x-x_ref`; zero when feedback invalid |
| 36..37 | 50..51 | 2 | `i16`, /10 cm/s | reference velocity `v_ref`; currently zero |
| 38..39 | 52..53 | 2 | `i16`, /10 cm/s2 | reference acceleration `a_ref`; currently zero |
| 40..41 | 54..55 | 2 | `u16` | reference state/source flags |
| 42..47 | 56..61 | 6 | bytes | reserved extension; schema 6 requires all zero |

The raw feedback target is

```text
e = x - x_ref
a_fb = -(Kp*Db(e, 0.12) + Kd*Db(v, 0.40) + Ka*a)
Db(z, width) = sign(z) * max(abs(z) - width, 0)
```

Default gains are `Kp=8.0`, `Kd=7.5`, `Ka=0`, and `a_fb` is limited to
`+/-140 cm/s2`. Tracker-internal `PENDING` and `MOVING` states both retain the
velocity used by `Kd`; only `QUIET` sets the control velocity to zero. Motion
starts after at least two reliable same-direction measurements; uncertain Hough
measurements require three, and no single quiet-state jump creates velocity.
Motion must then show four consecutive axial center steps no larger than
`0.75 px` before returning to `QUIET`. Confirmed `MOVING` uses the raw target immediately. Sustained
non-moving output follows it with gain `0.35` and zero hysteresis, allowing the
feedback to converge fully to zero. This internal motion state and the raw
pre-stabilization target are screen/PC diagnostics and are not additional
schema-6 fields.

`x_ref` is the currently applied K230 reference, expressed as an absolute
calibrated X coordinate in centimeters. Offsets 8/16 continue to carry absolute
ball position. The LCD and PC debug output also show `x`, `ref` and `e`. The MCU
can cross-check position/reference consistency and feedback bounds, but cannot
reconstruct every quiet-state `a_fb` sample from the UART fields alone. It must
not apply the same `Kp/Kd` feedback a second time. A
future MCU command that changes the reference must define validation, sequence,
acknowledgement, applied-value reporting and a bounded slew/ramp before it is
safety-usable. A reference step creates an immediate `Kp*Delta x_ref` change in
`a_fb`.

When `FEEDBACK_VALID=1`, fixed-point values should satisfy
`abs((x-x_ref)-e) <= 0.03 cm`; this tolerance covers independent `/100`
rounding. When feedback is invalid, K230 transmits `e=0` deliberately, so this
consistency equation must not be enforced. `v_ref/a_ref` are reserved for a
later moving-reference design and are zero in the current constant-reference
controller.

Offsets 16 and 18 intentionally repeat the physical X and X-velocity used by
the control law, preserving the established one-dimensional control offsets
from schema 4.
The deadbands and quiet-state stabilization affect only `a_fb`; transmitted
physical state remains the filtered physical state before these operations.
`Ka` defaults to zero, while `a`
remains available for diagnostics. `a_fb` is an equivalent ball-acceleration
feedback component, not a desired pipe angle or actuator/PWM command.

### Flags

| Bit | Name | Meaning |
| ---: | --- | --- |
| 0 | `RAW_DETECTED` | one candidate was selected this frame |
| 1 | `MEASUREMENT_ACCEPTED` | candidate updated the motion filter |
| 2 | `TRACK_VALID` | tracked or bounded-hold state is usable |
| 3 | `PREDICTED` | no current detection; bounded hold prediction is published |
| 4 | `FEEDBACK_VALID` | physical state and `a_fb` may be used by MCU composition |
| 5 | `OUTLIER_REJECTED` | association or measurement gate rejected detections |
| 6 | `EDGE_CLIPPED` | selected box touches/crosses the image edge |
| 7 | `FEEDBACK_SATURATED` | `a_fb` reached its configured magnitude limit |

### Reference state flags

| Bit | Name | Meaning |
| ---: | --- | --- |
| 0 | `REFERENCE_VALID` | `x_ref` and the reference-state fields are valid |
| 1 | `SOURCE_PRESET` | reference comes from `FEEDBACK_REFERENCE_POSITION_CM` |
| 2 | `SOURCE_ALGORITHM` | reserved for a future K230 algorithm-selected reference |
| 3 | `SOURCE_MCU` | reserved for a future validated MCU command reference |
| 4 | `SLEW_ACTIVE` | reference is moving through a bounded ramp |
| 5..15 | reserved | schema 6 requires zero |

Current firmware sends `REFERENCE_VALID|SOURCE_PRESET` (`0x0003`). Exactly one
source bit should be set when `REFERENCE_VALID=1`. Unsupported or contradictory
source flags must be logged and rejected from safety-critical use.

## MCU use

The MCU should validate exact size, version, schema, kind, transport flags,
reserved fields and CRC before decoding. Do not map the payload directly to a C
structure: decode little-endian integers explicitly so alignment and packing do
not depend on the compiler.

After validating freshness and ranges, decode the three states as follows:

| State | Required flag combination | MCU action |
| --- | --- | --- |
| `DETECT` | `MEASUREMENT_ACCEPTED=1`, `TRACK_VALID=1`, `FEEDBACK_VALID=1`, `PREDICTED=0`, `EDGE_CLIPPED=0` | use measured-state feedback |
| `HOLD` | `MEASUREMENT_ACCEPTED=0`, `TRACK_VALID=1`, `FEEDBACK_VALID=1`, `PREDICTED=1` | use bounded predicted feedback; monitor `missed_frames` and `measurement_age_ms` |
| `LOST` | `TRACK_VALID=0` or `FEEDBACK_VALID=0` | ignore `a_fb`, warn and enter the defined safe/fallback state |

K230 limits `HOLD` to at most 8 consecutive missed frames and 300 ms, with
either limit ending hold. The MCU must still enforce its own frame-arrival
freshness timeout, because receiving no UART frames cannot be reported by K230
flags.

The MCU owns:

- any future moving-trajectory `X(t)`, `X_dot(t)`, `X_ddot(t)` generation and
  its validated command transport; the current active stationary `x_ref` is a
  K230 preset and is reported in every frame;
- car-acceleration sensing/feedforward;
- any integral or additional state terms;
- rolling-ball model inversion and desired pipe angle;
- measured pipe-angle inner loop;
- actuator bounds, stale-data handling and safety arbitration.

For the current K230 preset constant equilibrium reference, `a_fb` already
contains both the position-error and ball-velocity feedback. The MCU must not
apply the same `Kp/Kd` terms to `x` a second time. It may add separately
designed trajectory, integral or disturbance terms only when their ownership
and anti-windup rules are explicit. See `docs/BALL_CONTROL_LAW.md` for the
complete signal flow and rolling-ball inversion.

Schema 6 preserves schema 5 payload offsets `0..31` and appends reference fields
at `32..47`; it changes the total frame length from 48 to 64 bytes. An MCU must
require schema 6, payload length 48 and total length 64 before using the
extension. Do not pad an old frame or attempt mixed-length decoding on the
control path. Schema 4 used offsets 8..15 for filtered pixel state, and schema 3
carried a desired angle at payload 28..29; neither may be decoded as schema 6.

## Verification boundary

K230 host tests cover fixed length, scales, signed values, CRC, split/coalesced
frames, noise recovery, motion hysteresis and feedback stabilization/bounds.
The MCU decoder has not been modified in this task and must be updated
separately for schema 6. End-to-end
UART, sign convention, physical calibration, controller composition and
actuator safety still require hardware verification.
