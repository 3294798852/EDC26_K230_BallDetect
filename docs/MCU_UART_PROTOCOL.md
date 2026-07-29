# K230 steel-ball visual feedback UART protocol

## Scope and physical link

This standalone application publishes camera-derived image state, physical
ball state and a bounded feedback component. It does not use the legacy task
handshake and does not require MCU-to-K230 control data.

| Item | Value |
| --- | --- |
| K230 peripheral | UART1 |
| K230 pins | IO9 TX, IO10 RX (RX currently unused by this application) |
| MCU connection | K230 IO9 TX -> MCU RX; common ground |
| Format | 512000 baud, 8N1, no flow control |
| Levels | 3.3 V TTL |
| Direction | K230 to MCU |
| Frame size | fixed 48 bytes |

K230 never waits for an MCU response. UART writes use a bounded two-frame,
latest-wins queue, so a disconnected receiver cannot block camera processing or
cause unbounded memory growth.

## Envelope

All multibyte fields are little-endian. CRC16-CCITT-FALSE uses polynomial
`0x1021`, initial value `0xFFFF`, no reflection/final xor. It covers offsets
`2..45`; CRC is stored at `46..47`.

| Offset | Size | Field | Value |
| ---: | ---: | --- | --- |
| 0 | 1 | sync 0 | `0xA5` |
| 1 | 1 | sync 1 | `0x5A` |
| 2 | 1 | envelope version | `1` |
| 3 | 1 | message type | `0x02` (`VISION_EVENT`) |
| 4 | 1 | payload schema | `4` |
| 5 | 1 | transport flags | `0x01` (`REALTIME`) |
| 6..7 | 2 | payload length | `32` |
| 8..9 | 2 | sequence | `u16`, wraps modulo 65536 |
| 10..13 | 4 | timestamp | K230 milliseconds, `u32` |
| 14..45 | 32 | payload | schema-4 kind-5 visual feedback |
| 46..47 | 2 | CRC16 | over offsets `2..45` |

## Payload schema 4, kind 5

| Payload offset | Frame offset | Size | Type / scale | Field |
| ---: | ---: | ---: | --- | --- |
| 0 | 14 | 1 | `u8` | kind `5` |
| 1 | 15 | 1 | `u8` | flags |
| 2 | 16 | 1 | `u8` | selected confidence x255 |
| 3 | 17 | 1 | `u8` | consecutive missed/rejected frames |
| 4..5 | 18..19 | 2 | `u16` px | raw selected center X |
| 6..7 | 20..21 | 2 | `u16` px | raw selected center Y |
| 8..9 | 22..23 | 2 | `u16`, /100 | filtered image X |
| 10..11 | 24..25 | 2 | `u16`, /100 | filtered image Y |
| 12..13 | 26..27 | 2 | `i16`, /10 | filtered image X velocity, px/s |
| 14..15 | 28..29 | 2 | `i16`, /10 | filtered image Y velocity, px/s |
| 16..17 | 30..31 | 2 | `i16`, /100 | physical position `x`, cm |
| 18..19 | 32..33 | 2 | `i16`, /10 | physical velocity `v`, cm/s |
| 20..21 | 34..35 | 2 | `i16`, /10 | visual acceleration `a`, cm/s2 |
| 22..23 | 36..37 | 2 | `i16`, /10 | feedback component `a_fb`, cm/s2 |
| 24..25 | 38..39 | 2 | `u16` ms | AI2D/KPU/postprocess time |
| 26..27 | 40..41 | 2 | `u16` ms | accepted-measurement age; `0xFFFF` unavailable |
| 28..29 | 42..43 | 2 | `u16`, /100 | `Kp` |
| 30 | 44 | 1 | `u8`, /10 | `Kd` |
| 31 | 45 | 1 | `u8`, /100 | `Ka` |

The feedback definition is

```text
a_fb = -(Kp*Db(x, 0.05) + Kd*Db(v, 1.0) + Ka*a)
Db(z, width) = sign(z) * max(abs(z) - width, 0)
```

The deadbands above affect only `a_fb`; the transmitted `x` and `v` fields remain
the filtered physical state before deadband. `Ka` defaults to zero, while `a`
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

## MCU use

The MCU should validate exact size, version, schema, kind, transport flags and
CRC before decoding. Do not map the payload directly to a C structure.

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

- `X`, `X_dot`, `X_ddot` generation;
- car-acceleration sensing/feedforward;
- any integral or additional state terms;
- rolling-ball model inversion and desired pipe angle;
- measured pipe-angle inner loop;
- actuator bounds, stale-data handling and safety arbitration.

Schema 3 frames from the previous build carried a desired angle at payload
28..29 and are semantically incompatible with schema 4. The MCU must not decode
schema 4 using the schema-3 structure or treat `a_fb` as an angle.

## Verification boundary

K230 host tests cover fixed length, scales, signed values, CRC, split/coalesced
frames, noise recovery and feedback bounds. The MCU decoder has not been
modified in this task and must be updated separately for schema 4. End-to-end
UART, sign convention, physical calibration, controller composition and
actuator safety still require hardware verification.