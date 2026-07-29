# K230 standalone steel-ball tracker (proj2)

This directory is the small, single-task deployment for the steel-ball camera
application.  It does not load the old task supervisor, OCR, line-following
modules, or a command handshake.  `main.py` starts the camera, YOLOv8n KModel,
one constrained motion filter, the on-screen overlay, and a one-way UART stream.

## SD-card layout

The application-specific files use this relative layout:

```text
main.py
ball_protocol.py
ball_tracking.py
pc_uart_test.py                         # optional dual-mode diagnostic
task_assets/ball_det/ball_det_yolov8n_320.kmodel
```

Burn `main.py` as the CanMV `main` entry.  The model path used by the program is
`/sdcard/task_assets/ball_det/ball_det_yolov8n_320.kmodel`.  No file from the old
`K230_SDcard` supervisor is required.

This directory also deliberately retains the board's common SD-card support
content restored byte-for-byte from the original backup:

```text
boot.py
libs/
ybUtils/
audio/
configs/
res/
resources/
utils/
ybMain/
revision.txt, Version.txt, yahboom.jpg, YbRequests.py
```

These are shared board/runtime resources, not obsolete task modules, and must
not be removed by project cleanup. When updating an existing SD card, preserve
its corresponding system directories. When preparing a card from this project
directory, copy them with the application files. Keep this project's `main.py`
as the startup entry; do not replace it with the vendor launcher's old
top-level `main.py`.

`configs/sys_config.json` is intentionally excluded from Git because a live
board copy may contain Wi-Fi and API credentials. For a fresh checkout, copy
`configs/sys_config.example.json` to `configs/sys_config.json` and fill only the
local device values. `.vscode/` is also local-only because its CanMV stub paths
are workstation-specific.

## Vision and filter

- Camera channel 0 supplies the 640x480 live image; channel 2 supplies a
  320x320 RGB888-planar frame to AI2D/KPU.
- The model is checked at startup for one input and one output.  The first
  output is expected to be `[1, 5, 2100]` and is passed to
  `aidemo.yolov8_det_postprocess`.
- Only class `0` (the steel ball) is considered.  Of all valid postprocessed
  candidates, exactly the highest-confidence one is selected; a rejected
  candidate is not replaced by the second-best candidate in that frame.
- `PipeMotionFilter` is a bounded constant-velocity alpha-beta filter.  It
  learns and locks the image-space axis of the one-dimensional pipe, gates
  implausible jumps, projects velocity onto that axis, and coasts through a
  short miss.  After 250 ms without an accepted measurement the track is no
  longer marked valid.

The overlay is deliberately lightweight (about 10 Hz): it draws the selected
box, filtered center/pipe axis, filtered `x/y`, `Vx/Vy`, confidence, inference
time, measurement age, and UART counters. UART exception writes, invalid write
results, short writes, zero writes and error-dropped frames have separate
counters; the latest UART error text remains visible. A recoverable frame error
is also latched on the normal overlay, while three consecutive frame errors or
an initialization error enter a full-screen FATAL page showing stage, exception
type and message until the program is stopped manually.

`main.py` captures failures while importing CanMV/KPU/protocol dependencies and
uses a minimal emergency OSD path, so a failure before normal camera setup is
not intentionally left as a black screen. The emergency page itself still
depends on the firmware's `image`, `Display` and `MediaManager` stack; if that
display stack is the component that cannot load or initialize, the USB console
is the only remaining error channel.

CanMV may deliver its IDE Stop/Run takeover request as a plain
`Exception("IDE interrupt")` from camera capture, KPU inference or UART write
instead of as `KeyboardInterrupt`. Version `proj2-1.2` recognizes both forms as
a normal stop request: it does not count or latch them as camera/model/UART
errors and releases the resources so the IDE can run another script.

## UART connection

The K230 uses UART1 at `512000 8N1`:

```text
K230 IO9 (UART1 TX)  -> 3519 UART6 RX (PC10)
K230 IO10 (UART1 RX) <- 3519 UART6 TX (PC11)
K230 GND             -- 3519 GND
```

Use 3.3 V TTL levels and no hardware flow control.  The K230 sends one fixed
48-byte state frame after every inference, including a no-ball state.  Frames
are latest-wins in a two-frame (96-byte) software queue; short writes resume at
the next loop and a stalled write cannot grow without bound.

The stream is intentionally one-way from the application point of view.  It
does not emit `HELLO`, `STATUS`, or `ACK`, and it does not wait for an MCU
response before capturing the next frame.  The existing 3519 transport may
continue its legacy HELLO retries, but it can consume this vision stream
independently through `k230_link_read_ball_state()`.

## Direct PC UART test and K230-side debug

`pc_uart_test.py` chooses its mode before importing optional libraries. It does
not import `argparse` or `pyserial` under CanMV/MicroPython.

### Desktop receiver mode

Run the file from Windows PowerShell/cmd with desktop CPython, not with CanMV's
Run button. This receiver can run on the PC while K230 `main.py` continues
running. It receives UART1 through a USB-to-TTL adapter that supports 3.3 V
logic and 512000 baud:

```text
K230 IO9 (UART1 TX) -> USB-TTL RX
K230 GND            -- USB-TTL GND
```

Adapter TX and VCC are not required. Never connect a 5 V UART signal or use the
adapter VCC to power the K230. Install pyserial, find the port, then receive:

```text
python -m pip install pyserial
python pc_uart_test.py --self-test
python pc_uart_test.py --list
python pc_uart_test.py --port COM7
python pc_uart_test.py --port COM7 --every-frame --raw
```

Replace `COM7` with the actual port. The program resynchronizes after noise or a
bad CRC and reports valid-frame rate, CRC/header errors, discarded bytes,
sequence gaps/reboots, flags, raw and filtered coordinates, X/Y velocities,
box, confidence, inference time and measurement age. Some inexpensive USB-TTL
adapters or drivers do not support 512000 baud; an open/configuration failure is
therefore an adapter/driver compatibility result rather than proof that the K230
program failed.

### CanMV/K230 debug mode

K230 can execute only one foreground Python application. To use the optional
board-side debug wrapper, first click **Stop** in CanMV and wait for
`steel-ball tracker stopped by IDE/user`; then open `pc_uart_test.py` and run it
as a temporary script while the real proj2 `main.py` remains at
`/sdcard/main.py`. It imports that tracker entry and calls
`main.run(pc_debug=True)`: camera inference, the normal screen overlay and UART1
transmission continue, and one `PCDBG` status line is mirrored to the USB/IDE
console about every 500 ms. Each line includes sequence, flags, raw and filtered
coordinates, X/Y velocity, confidence, inference time, TX frame/byte/queue
counters, exception/short/zero/drop counters and the last UART/runtime error
text.

Do not burn `pc_uart_test.py` *as* `/sdcard/main.py`; it is a wrapper and expects
the production tracker entry to stay there. The K230 debug mode verifies the
application path and reports software TX results, but USB console output does
not prove that the electrical UART1 IO9 waveform reaches a PC. Use desktop
receiver mode plus the 3.3 V USB-TTL wiring above for that end-to-end check.

## MCU side

The matching decoder is in the 3519 project:

```text
Comm/k230_payload.h/.c  - schema-2 payload validation and decoding
Comm/k230_link.h/.c     - publication and 250 ms freshness state
```

The foreground must call `k230_link_service(now_ms)` as usual, then read the
latest snapshot with `k230_link_read_ball_state(&state, now_ms)`.  A return of
`K230_DATA_FRESH` is required before using coordinates or velocities.  A stale
or never-seen snapshot must not directly command an actuator.  See
`docs/MCU_UART_PROTOCOL.md` for byte offsets, scales, flags, and the safety
rules.

## Verification boundary

The source has passed host-side protocol/filter regression and Python AST
checks.  Loading the KModel, actual frame rate, display alignment, UART waveform
and end-to-end 3519 reception still require a target-board test with the stated
wiring and power conditions.
