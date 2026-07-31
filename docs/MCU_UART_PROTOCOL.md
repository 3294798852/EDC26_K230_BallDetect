# K230 steel-ball visual feedback UART protocol

## Scope and physical link

This standalone application publishes camera-derived image state, physical
ball state and a bounded feedback component. The MCU may send one compact
command to update the stationary equilibrium reference. There is no legacy
task handshake and K230 never waits for a command or acknowledgement.

Physical fields use the compile-time `X_CALIBRATION_MODE` selected in
`main.py`. The default `measured_distortion` table corrects the installed
camera from physical-ruler anchors; `pinhole` restores the original ideal
camera table. Both modes keep the same schema, field scales and units, so the
MCU decoder does not branch on calibration mode. Record the selected mode
during system calibration; the LCD physical-state row and PC debug identify it.

| Item | Value |
| --- | --- |
| K230 peripheral | UART1 |
| K230 pins | IO9 TX, IO10 RX |
| MCU connection | K230 IO9 TX -> MCU RX; K230 IO10 RX <- MCU TX; common ground |
| Format | 512000 baud, 8N1, no flow control |
| Levels | 3.3 V TTL |
| Direction | bidirectional |
| Frames | K230 state 64 bytes; MCU command 10 bytes |

K230 never waits for an MCU response. UART writes use a bounded two-frame,
latest-wins queue, so a disconnected receiver cannot block camera processing or
cause unbounded memory growth.

## MCU set-reference command

To set the stationary balance point, send exactly 10 bytes. Multibyte fields
are little-endian. `x_ref_i16 = round(x_ref_cm * 100)` and the accepted range is
`-1250..+1250` (`-12.50..+12.50 cm`). This is an absolute calibrated physical X
coordinate using the same sign convention as state-frame `x`, not a pixel
coordinate or a relative movement.

| Offset | Size | Field | Value |
| ---: | ---: | --- | --- |
| 0 | 1 | sync 0 | `0xA5` |
| 1 | 1 | sync 1 | `0x5A` |
| 2 | 1 | protocol version | `1` |
| 3 | 1 | message type | `0x10` (`SET_REFERENCE`) |
| 4 | 1 | command schema | `1` |
| 5 | 1 | flags | `0` |
| 6..7 | 2 | reference | signed `i16`, `/100 cm` |
| 8..9 | 2 | CRC16 | over offsets `2..7` |

Known-good complete frame examples:

```text
x_ref =  0.00 cm: A5 5A 01 10 01 00 00 00 5E 39
x_ref = +2.50 cm: A5 5A 01 10 01 00 FA 00 54 C5
x_ref = -2.50 cm: A5 5A 01 10 01 00 06 FF 08 8D
```

CRC16-CCITT-FALSE uses polynomial `0x1021`, initial `0xFFFF`, no reflection and
no final xor. Invalid sync/version/type/schema/flags/range/CRC commands are
discarded without changing the current reference. The parser accepts split and
coalesced frames, resynchronizes after noise, uses fixed memory, and reads at
most 32 UART bytes per camera loop.

The operation is idempotent. No separate ACK frame is sent. After applying the
command, K230 continuously reports the quantized applied value in state-frame
payload `x_ref` and reports reference flags `REFERENCE_VALID|SOURCE_MCU`
(`0x0009`). The MCU should regard that state-frame echo as confirmation. If it
is not observed within a bounded timeout, the MCU may resend the same 10-byte
command a limited number of times; duplicate commands are harmless. A command
does not reset tracking or the learned integral correction.

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
e_f = LPF(e, 25 ms), v_f = LPF(v, 50 ms)
e_db = Db(e_f, 0.10)
Ufric = -20 * clamp(e_db / 0.20, -1, +1)
if e_f*v_db < 0 and I != 0:
    I = move_toward_zero(I, 20*abs(v_db)*dt)
else:
    I = clamp(I - Ki*clamp(e_db, +/-1.0)*dt, +/-80)
a_raw = -(Kp*e_db + Kd*Db(v_f, 0.35) + Ka*a) + Ufric + I
a_fb = clamp(LPF(a_raw, 25 ms), +/-140)
Db(z, width) = sign(z) * max(abs(z) - width, 0)
```

Default gains are `Kp=8.0`, `Kd=12.0`, `Ki=12.0`, `Ka=0`. `I` is bounded to
`+/-80 cm/s2`, has no default leakage, uses conditional anti-windup, holds on
predicted frames and resets on full loss/filter reset. During motion toward the
reference it moves toward zero by `20*|v_db|*dt`, without crossing zero.
`Ufric` is continuous,
ramps over `0.20 cm` beyond the position deadband and is bounded to
`+/-20 cm/s2`. Final `a_fb` is bounded to `+/-140 cm/s2`.
The LCD can change `Kp/Kd/Ki` live in `0.1` steps. Schema 6 continues to carry
the applied `Kp/Kd/Ka` and complete final `a_fb`; it has no `Ki` field and the
reserved bytes remain zero. The MCU must not calculate or add the PI-D term a
second time.

`x_ref` is the currently applied K230 reference, expressed as an absolute
calibrated X coordinate in centimeters. Offsets 8/16 continue to carry absolute
ball position. The LCD and PC debug output also show `x`, `ref` and `e`. The MCU
can cross-check position/reference consistency and feedback bounds, but cannot
reconstruct every quiet-state `a_fb` sample from the UART fields alone. It must
not apply the same `Kp/Kd` feedback or integrate the same position error a
second time. A
future MCU command that changes the reference must define validation, sequence,
acknowledgement, applied-value reporting and a bounded slew/ramp before it is
safety-usable. A reference step changes the proportional target by
`Kp*Delta x_ref`; the existing 25 ms output filter smooths its published
`a_fb` transition.

When `FEEDBACK_VALID=1`, fixed-point values should satisfy
`abs((x-x_ref)-e) <= 0.03 cm`; this tolerance covers independent `/100`
rounding. When feedback is invalid, K230 transmits `e=0` deliberately, so this
consistency equation must not be enforced. `v_ref/a_ref` are reserved for a
later moving-reference design and are zero in the current constant-reference
controller.

Offsets 16 and 18 intentionally repeat the physical X and X-velocity used by
the control law, preserving the established one-dimensional control offsets
from schema 4.
The controller's additional position/velocity filters, deadbands, integral and
output stabilization affect only `a_fb`; transmitted physical state remains
the tracker output before these operations.
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
| 3 | `SOURCE_MCU` | reference was set by a valid MCU command |
| 4 | `SLEW_ACTIVE` | reference is moving through a bounded ramp |
| 5..15 | reserved | schema 6 requires zero |

At boot firmware sends `REFERENCE_VALID|SOURCE_PRESET` (`0x0003`). After a
valid command it sends `REFERENCE_VALID|SOURCE_MCU` (`0x0009`). Exactly one
source bit must be set when `REFERENCE_VALID=1`. Unsupported or contradictory
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
  its validated command transport; the current stationary `x_ref` is the K230
  boot preset or latest valid MCU command and is reported in every frame;
- car-acceleration sensing/feedforward;
- any independently designed moving-trajectory term; K230 already owns the
  stationary ball-position PI-D correction inside `a_fb`;
- rolling-ball model inversion and desired pipe angle;
- measured pipe-angle inner loop, including any integral state local to that
  separate angle loop;
- actuator bounds, stale-data handling and safety arbitration.

For the current K230 preset constant equilibrium reference, `a_fb` already
contains position, velocity and bounded integral correction. The MCU must not
apply the same K230 PI-D behavior a second time. It may add a
separately designed trajectory or inner-angle-loop term only when ownership
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
state and command frames, noise recovery, command range rejection, reference
application, motion hysteresis, PI-D anti-windup and feedback stabilization.
The MCU decoder has not been modified in this task and must be updated
separately for schema 6. End-to-end
UART, sign convention, physical calibration, controller composition and
actuator safety still require hardware verification.
