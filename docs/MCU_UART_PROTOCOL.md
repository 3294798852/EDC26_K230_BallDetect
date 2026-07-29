# Standalone steel-ball UART protocol (MCU integration)

This document specifies the stream emitted by `K230_SDcard_proj2/main.py` and
the corresponding 3519 implementation. It deliberately does not reuse the old
multi-task lifecycle handshake: a camera frame is useful even when the MCU has
not selected a task.

## Physical link and framing

| Item | Value |
| --- | --- |
| K230 peripheral | UART1 |
| K230 pins | IO9 TX, IO10 RX |
| 3519 peripheral | UART6 (`UART_K230`), PC10 RX / PC11 TX |
| Format | 512000 baud, 8 data bits, no parity, 1 stop bit |
| Flow control | none |
| Levels | 3.3 V TTL; common ground required |
| Direction | K230 sends vision frames; MCU may still send legacy commands, which this standalone app ignores |
| Frame size | fixed 48 bytes |

The generic envelope is little-endian. CRC16-CCITT-FALSE (polynomial `0x1021`,
initial `0xFFFF`, no reflection/final xor) covers byte offsets `2` through `45`,
excluding the two sync bytes and the two CRC bytes. The CRC is stored
little-endian at offsets `46..47`.

| Offset | Size | Field | Value |
| ---: | ---: | --- | --- |
| 0 | 1 | sync 0 | `0xA5` |
| 1 | 1 | sync 1 | `0x5A` |
| 2 | 1 | envelope version | `1` |
| 3 | 1 | message type | `0x02` (`VISION_EVENT`) |
| 4 | 1 | payload schema | `2` |
| 5 | 1 | transport flags | bit 0 `REALTIME` |
| 6..7 | 2 | payload length | `32` |
| 8..9 | 2 | sequence | `u16`, wraps naturally |
| 10..13 | 4 | sender timestamp | K230 milliseconds, `u32` wrap |
| 14..45 | 32 | payload | schema-2 kind-3 state |
| 46..47 | 2 | CRC16 | over offsets `2..45` |

The MCU generic parser validates sync, version, length and CRC before calling
the payload decoder. Do not map the payload to a C `struct`: padding and
alignment are implementation-defined. Read each field from its documented
little-endian offset.

## Payload schema 2, kind 3

Payload offset `0` is always `kind=3`.

| Offset | Size | Type / scale | Field |
| ---: | ---: | --- | --- |
| 0 | 1 | `u8` | kind (`3`) |
| 1 | 1 | `u8` bit mask | state flags (below) |
| 2 | 1 | `u8` | confidence x 255 |
| 3 | 1 | `u8` | consecutive missed/rejected frames, saturating |
| 4..5 | 2 | `u16` pixels | raw selected-box center X |
| 6..7 | 2 | `u16` pixels | raw selected-box center Y |
| 8..9 | 2 | `u16`, /100 | filtered X in the 320x320 inference image |
| 10..11 | 2 | `u16`, /100 | filtered Y in the 320x320 inference image |
| 12..13 | 2 | `i16`, /10 | filtered X velocity, pixels/s |
| 14..15 | 2 | `i16`, /10 | filtered Y velocity, pixels/s |
| 16..17 | 2 | `u16` pixels | selected box X |
| 18..19 | 2 | `u16` pixels | selected box Y |
| 20..21 | 2 | `u16` pixels | selected box width |
| 22..23 | 2 | `u16` pixels | selected box height |
| 24..25 | 2 | `u16` ms | KPU/AI2D/postprocess inference time |
| 26..27 | 2 | `u16` ms | age of last accepted measurement; `0xFFFF` unavailable |
| 28..29 | 2 | `i16`, /16384 | pipe-axis X unit component |
| 30..31 | 2 | `i16`, /16384 | pipe-axis Y unit component |

Coordinates are bounded to `[0,319]` for pixel positions and `[0,319.99]` for
filtered fixed-point positions. Velocities are signed. A no-ball frame has
`RAW_DETECTED=0`; its box and raw-coordinate fields are zero, while the filter
may still report a short predicted coast. `TRACK_VALID=0` is the authoritative
indicator that the snapshot is not usable for control.

### Flags

| Bit | Name | Meaning |
| ---: | --- | --- |
| 0 | `RAW_DETECTED` | a valid class-0 candidate was selected this frame |
| 1 | `MEASUREMENT_ACCEPTED` | the candidate passed the motion gate and updated the filter |
| 2 | `TRACK_VALID` | last accepted measurement is no older than 250 ms |
| 3 | `PREDICTED` | current position/velocity includes coast prediction |
| 4 | `AXIS_LOCKED` | pipe motion axis has been learned and locked |
| 5 | `OUTLIER_REJECTED` | the highest-confidence candidate failed the motion gate |
| 6 | `EDGE_CLIPPED` | selected box touched or crossed the image edge |
| 7 | `FILTER_RESET` | this frame initialized/reinitialized the filter |

`RAW_DETECTED` and `MEASUREMENT_ACCEPTED` are intentionally separate. A
high-confidence false detection can be selected but rejected by the
one-dimensional motion gate; the program does not silently fall back to a
lower-confidence box.

## 3519 decoder and usage

The MCU changes are intentionally limited to the existing transport/payload
layers:

```c
#include "Comm/k230_link.h"

k230_ball_state_t ball;
k230_data_state_t state;

k230_link_service(now_ms);
state = k230_link_read_ball_state(&ball, now_ms);
if (state == K230_DATA_FRESH &&
    (ball.flags & K230_BALL_STATE_FLAG_TRACK_VALID) != 0U) {
    /* Use ball.filtered_*_x100 and ball.velocity_*_x10 after scaling. */
}
```

`k230_decode_ball_state()` validates the exact 32-byte payload, kind, fixed
coordinate ranges, Q14 axis range, and raw box ranges whenever `RAW_DETECTED`
is set. It decodes into a temporary object and publishes only on success, so a
malformed frame cannot partially overwrite the previous snapshot.
`k230_link_read_ball_state()` returns:

- `K230_DATA_NONE` before the first valid state frame;
- `K230_DATA_FRESH` when the local receive time is within 250 ms;
- `K230_DATA_STALE` after that timeout (the last values remain available for
  diagnostics, but are not control-valid).

The MCU UART6 RX/TX rings are 256 bytes (255 usable), and the foreground parser
is bounded per service pass. This is larger than one 48-byte frame and absorbs
several frames if the foreground is briefly occupied. Keep any additional
consumer ring at least 128 bytes and preserve the existing SPSC ownership
rules.

The standalone stream does not establish `session_established`, send ACKs, or
produce STATUS. The existing legacy link state machine can therefore remain in
its HELLO/retry state without preventing vision publication; applications that
use this project should read the ball-state API directly and must not gate it on
`k230_link_task_confirmed()`.

## Safety and freshness

This frame is a measurement, not an actuator command. Before any actuator
decision the MCU must independently check freshness, `TRACK_VALID`, acceptable
confidence/edge flags, local limits, and the current safety state. A stale,
predicted-only, CRC-invalid, disconnected, or otherwise rejected stream must
lead to the application's defined safe behavior. Never drive an actuator
directly from an unchecked raw coordinate or from a snapshot older than the
250 ms timeout.

## Verification status

The 3519 host C suite includes a fixed schema-2 frame with signed velocities,
Q14 axis values, CRC/framing, and invalid-coordinate rejection. The K230
modules pass AST and protocol/filter model checks. K230 KPU loading, display,
UART electrical levels, wire-level CRC, and end-to-end board reception remain
target-board verification items rather than claims made by these host checks.
