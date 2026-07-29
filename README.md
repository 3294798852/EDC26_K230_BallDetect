# K230 standalone steel-ball tracker (proj2)

This directory is the small, single-task deployment for the steel-ball camera
application.  It does not load the old task supervisor, OCR, line-following
modules, or a command handshake. `main.py` starts the camera, one selected
YOLOv8 or native ROI detector, one constrained motion filter, camera-derived
physical-state feedback, the
on-screen overlay, and a bounded one-way UART stream.

## SD-card layout

The application-specific files use this relative layout:

```text
main.py
ball_roi_detector.py
ball_control.py
ball_protocol.py
ball_tracking.py
pc_uart_test.py                         # optional dual-mode diagnostic
task_assets/ball_det/ball_det_yolov8n_320.kmodel
task_assets/ball_det/ball_det_yolov8_320.kmodel
```

Burn `main.py` as the CanMV `main` entry. Select the detector by changing only
`MODEL_BACKEND` near the top of `main.py`:

| Value | Input/asset | Identification |
| --- | --- | --- |
| `"yolov8n"` | `ball_det_yolov8n_320.kmodel` | existing YOLOv8n export |
| `"yolov8"` | `ball_det_yolov8_320.kmodel` | larger custom YOLOv8 export; current default |
| `"roi_blob"` | native 320x320 RGB565 image | calibrated pipe ROI, adaptive dark-blob detection; no KModel/KPU inference |

Both files are nncase KModel v7 with `1x3x320x320` input. The second file is
10,273,352 bytes versus 6,440,328 bytes for the known `yolov8n` file, so it is
a different and larger graph, but size alone does not establish a specific
Ultralytics subtype. No ONNX, training YAML or export log is present; the code
therefore labels it accurately as `YOLOv8 custom`, not speculatively as `s` or
`m`. No file from the old `K230_SDcard` supervisor is required.

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

- Camera channel 0 supplies the 640x480 live image. Channel 2 supplies either a
  320x320 RGB888-planar frame for YOLO or a native RGB565 image for `roi_blob`.
- The ROI backend scans only the calibrated pipe band with native
  `get_statistics`, `binary`, morphology and `find_blobs` operations. It adapts
  its dark threshold from a narrower bright pipe sample band, then scores blobs
  by diameter, fill, aspect ratio and pipe-axis distance. It has no Python pixel
  loop and reuses the existing continuity tracker.
- Each YOLO backend declares its path and expected output independently. Both current
  one-class exports are configured for `[1, 5, 2100]`; the first inference
  checks the actual rank and shape before postprocessing. A mismatch enters the
  visible FATAL path with backend, actual shape and expected shape instead of
  interpreting incompatible data. Native `aidemo.yolov8_det_postprocess` then
  performs bounded confidence/NMS and returns at most four class-0 candidates.
- Startup console, normal overlay and PCDBG identify the selected backend so
  timing and recognition quality can be compared without ambiguity.
- Only one target is published. While a track is valid, candidates are
  associated by predicted-position continuity first, with confidence used only
  as a near-tie breaker. A clipped edge candidate therefore cannot replace a
  well-predicted interior target merely because its confidence is higher.
- If no candidate matches the active track, an interior challenger must remain
  spatially consistent for three consecutive frames before it can replace that
  track. Edge-clipped candidates cannot initiate a normal target switch.
- When no track exists, a new target requires three consecutive, spatially
  consistent, non-edge candidates with confidence at least 0.50. The
  MCU can set `REACQUIRE` in a validated control-input frame to invoke
  `PipeMotionFilter.request_reacquire()` once; normal tracking remains strict.
- `PipeMotionFilter` is a bounded constant-velocity alpha-beta filter. It
  learns and locks the image-space axis of the one-dimensional pipe, gates
  implausible jumps, projects velocity onto that axis, and coasts through a
  short miss. Its axial gains are tuned for lower acceleration/reversal lag
  (`alpha=0.82`, `beta=0.28`) and ease down only to `alpha=0.72`, `beta=0.18`
  for small innovations. The fast gains return by roughly 4 px innovation or
  120 px/s, preserving response while retaining modest jitter suppression.
  Box centers retain subpixel precision
  for the filter; integer
  geometry is used only for drawing and the wire format. A missed detection
  enters bounded prediction hold for at most 8 frames and 300 ms; either limit
  ending first changes the state to fully lost.

## Visual state feedback

`ball_control.py` projects the filtered image state onto the calibrated pipe
axis and produces physical position, velocity and acceleration. K230 then
calculates only the camera-state feedback component

```text
a_fb = -(Kp*Db(x, 0.05) + Kd*Db(v, 1.0) + Ka*a)
Db(z, width) = sign(z) * max(abs(z) - width, 0)
```

Defaults are `Kp=16`, `Kd=7.2`, `Ka=0`, with `a_fb` bounded to
`+/-250 cm/s2`. No second physical-state low-pass is applied: `x/v` are converted
directly from the single alpha-beta tracker to avoid additional phase lag.
Continuous soft deadbands of `0.05 cm` for position and `1.0 cm/s` for velocity
apply only when calculating feedback; UART still reports the tracked physical
state itself. Acceleration remains published for diagnostics but is disabled in
the default feedback law. These values are configurable near the top of
`main.py` and are sent with every state frame.

K230 does not receive `X(t)`, car acceleration or pipe angle and does not
calculate desired pipe angle, PWM or motor commands. The MCU combines `a_fb`
with its own trajectory and car feedforward, performs the ball/pipe model
inversion and closes the measured-angle inner loop.

The current `(10,160)` to `(310,160)` image endpoints are provisional for the
25 cm pipe. Measure both endpoints and verify zero, scale and positive direction
before allowing the MCU to use `FEEDBACK_VALID` data.

The overlay is deliberately lightweight (about 10 Hz). It explicitly shows
`DETECT`, `HOLD n/8`, or `LOST`, plus selected box, tracked image position/
velocity, physical `x/v/a`, `a_fb`, gains, performance and UART diagnostics.
`HOLD` continues bounded prediction feedback while remaining distinguishable
from a real detection. Recoverable errors remain latched; three consecutive
frame errors or an initialization failure enter the visible FATAL page.

See `docs/ROI_BALL_DETECTOR.md` for ROI geometry, diagnostics and tuning, and
`docs/BALL_CONTROL_LAW.md` for the control responsibility boundary.
`main.py` captures failures while importing CanMV/KPU/protocol dependencies and
uses a minimal emergency OSD path, so a failure before normal camera setup is
not intentionally left as a black screen. The emergency page itself still
depends on the firmware's `image`, `Display` and `MediaManager` stack; if that
display stack is the component that cannot load or initialize, the USB console
is the only remaining error channel.

CanMV may deliver its IDE Stop/Run takeover request as a plain
`Exception("IDE interrupt")` from camera capture, KPU inference or UART write
instead of as `KeyboardInterrupt`. Version `proj2-1.14` recognizes both forms as
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

K230 emits one fixed 48-byte schema-4 frame per inference. It carries image
coordinates/velocities, physical `x/v/a`, `a_fb`, `Kp/Kd/Ka`, inference time and
freshness flags. This application does not read MCU control data, does not use
the legacy `HELLO`/`STATUS`/`ACK` handshake and never waits for a response. See
`docs/MCU_UART_PROTOCOL.md` for exact offsets, scales, CRC and safety rules.

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
sequence gaps/reboots, flags, raw and filtered coordinates, image velocity,
physical position/velocity/acceleration, `a_fb`, gains, confidence, inference
time and measurement age. Some inexpensive USB-TTL
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
coordinates, image and physical one-dimensional state, `a_fb`, gain/validity
status, confidence, split performance timings, candidate/association/
acquisition diagnostics, TX frame/byte/queue counters,
exception/short/zero/drop counters and the last UART/runtime error text.

Do not burn `pc_uart_test.py` *as* `/sdcard/main.py`; it is a wrapper and expects
the production tracker entry to stay there. The K230 debug mode verifies the
application path and reports software TX results, but USB console output does
not prove that the electrical UART1 IO9 waveform reaches a PC. Use desktop
receiver mode plus the 3.3 V USB-TTL wiring above for that end-to-end check.

## MCU side

The MCU decoder must be updated for schema 4/kind 5. Three states use existing
flags: `MEASUREMENT_ACCEPTED+TRACK_VALID+FEEDBACK_VALID` is `DETECT`;
`PREDICTED+TRACK_VALID+FEEDBACK_VALID` is bounded `HOLD`; absence of
`TRACK_VALID` or `FEEDBACK_VALID` is `LOST`. A stale, CRC-invalid,
invalid-feedback or fully lost frame must not directly affect an actuator. The
MCU owns
reference generation, car feedforward, pipe-angle solution, measured-angle
inner loop and all actuator safety.

## Verification boundary

Host checks validate detector selection, ROI geometry/scoring with simulated
CanMV objects, YOLO candidate normalization, association rules, feedback
equations, fixed-point protocol and parser recovery. They cannot establish
RGB565 threshold behavior, K230 execution time or camera differentiation noise. The
camera is configured for 30 FPS, so a frame interval near 33 ms can be normal.
Use on-screen `cap/pre/kpu/post/ai/fb` and log `x/v/a/a_fb` on hardware.
Loading each KModel and validating ROI threshold/geometry under the fixed camera
and contest lighting require K230 target-board tests. KModel output shape, score
range, recognition quality, KPU time and memory headroom also remain target
checks. Display alignment, UART waveform and end-to-end MCU
reception likewise remain hardware verification items.
