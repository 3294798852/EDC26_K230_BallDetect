# K230 standalone steel-ball tracker (proj2)

This directory is the small, single-task deployment for the steel-ball camera
application.  It does not load the old task supervisor, OCR, line-following
modules, or a command handshake. `main.py` starts the camera, one selected
detector backend, one constrained motion filter, camera-derived
physical-state feedback, the
on-screen overlay, and a bounded one-way UART stream.

## SD-card layout

The application-specific files use this relative layout:

```text
main.py
ball_calibration.py
ball_edge_hough_detector.py
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
| `"yolov8"` | `ball_det_yolov8_320.kmodel` | larger custom YOLOv8 export |
| `"roi_blob"` | native 320x320 RGB565 image | calibrated pipe ROI with tracking-aware detection; no KModel/KPU inference |
| `"edge_hough"` | fixed pipe-band crop from native 320x320 RGB565 image | current default; Canny edge and Hough-circle voting in `cv_lite`, followed by bounded pipe/tracking gates |

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

Camera exposure is configurable through `CAMERA_EXPOSURE_US` near
`CAMERA_FPS` in `main.py`. Its default is `None`, which deliberately does not
call `Sensor.auto_exposure()` or `Sensor.exposure()` and therefore preserves
the sensor firmware's original exposure behavior. To reduce fast-motion blur,
set a positive exposure time in microseconds, for example:

```python
CAMERA_EXPOSURE_US = 5000
```

The application then disables automatic exposure and requests the fixed value.
Use `5000 us` as an initial fast-motion test, then try `4000`, `3000` or
`2000 us` while increasing illumination as needed. Do not use
`cv_lite.rgb888_adjust_exposure()` for this purpose: post-capture brightness
adjustment cannot remove motion blur already present in the frame. Startup,
the normal overlay and PCDBG show `EXP=SYS` for the untouched default or
`EXP=<value>us` for fixed exposure. An invalid value or unsupported sensor call
enters the existing visible FATAL path rather than being silently ignored.

- Camera channel 0 supplies the 640x480 live image. Channel 2 supplies either a
  320x320 RGB888-planar frame for YOLO or a native RGB565 image for
  `roi_blob`/`edge_hough`.
- The ROI backend scans only the calibrated pipe band with native
  `get_statistics`, `binary`, morphology and `find_blobs` operations. It adapts
  its dark threshold from a narrower bright pipe sample band. During an active
  track it also uses the filter's read-only predicted position, speed and
  observed ball diameter to recover very small components, merge nearby dark
  arcs split by a specular highlight, and accept bounded motion blur only along
  the pipe axis. Global acquisition remains strict. It has no Python pixel loop
  and still performs only one native connected-component scan per frame.
- The edge-Hough backend first uses the target-verified
  `image.copy(roi=...)` API to crop a fixed pipe band. The horizontal extent is
  aligned outward for the verified fast path, while the vertical extent keeps
  the configured 80-pixel height. The
  current calibration produces a `320x80` crop, which is converted to RGB888
  before calling `cv_lite.rgb888_find_circles` once. Returned local circle
  coordinates are translated back to the unchanged 320x320 image coordinate
  system. Its internal Canny and circle voting can
  retain a small ball from partial outer arcs even when a specular highlight
  breaks the edge. Python examines at most 24 returned circles and rejects
  wrong radii, centers outside the pipe band/endpoints and implausible tracked
  jumps. Straight pipe walls do not by themselves satisfy the bounded-radius
  circle vote; the existing multi-frame acquisition gate further suppresses
  isolated false circles.
- Each YOLO backend declares its path and expected output independently. Both current
  one-class exports are configured for `[1, 5, 2100]`; the first inference
  checks the actual rank and shape before postprocessing. A mismatch enters the
  visible FATAL path with backend, actual shape and expected shape instead of
  interpreting incompatible data. Native `aidemo.yolov8_det_postprocess` then
  performs bounded confidence/NMS and returns at most four class-0 candidates.
- Startup console, normal overlay and PCDBG identify the selected backend so
  timing and recognition quality can be compared without ambiguity. Timing
  labels describe the selected backend: YOLO uses `ai2d/kpu/post/det`, ROI uses
  `mask/blob/select/det`, and edge-Hough uses `crop/hough/select/det`. Here
  `det` is the total detector-run time; native backends do not run a KPU model.
- The timing row reports `FPS=<measured>/<requested>` and `dt=<ms>` from
  consecutive successful camera-frame completion timestamps. A `60` request
  that reaches `59-60 FPS` and `16-17 ms` is effective end to end. If it remains
  near `29-30 FPS` and `33-34 ms`, use `cap`, `det` and `loop` together: a long
  blocking `cap` with a short detector points to the sensor/driver mode, while a
  long detector or loop points to application-side backpressure. The measured
  period uses a constant-cost exponential average and allocates no history list.
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
- `PipeMotionFilter` uses the calibrated pipe endpoints as a fixed axis from
  the first accepted frame. Axial and lateral states are independent: axial
  position has a small innovation noise band and rises from quiet gain `0.18`
  to fast gain `0.96` by `1.5 px`, while lateral position uses gain `0.08`
  and lateral velocity is forced to zero. Axial velocity follows consecutive
  accepted measurements with fast reversal response and decays strongly when
  displacement remains inside a cumulative `0.90 px` velocity deadband for
  `120 ms`; the bounded accumulation preserves slow real motion instead of
  damping every subpixel frame. A residual opposite to an established motion
  activates a short braking latch, so stopping does not coast until the quiet
  timeout. Apparent
  diameter uses a slow, one-pixel-bounded update because the camera geometry is
  fixed. Correction uses the capture-complete timestamp, then predicts only
  across the measured detector latency to the publish instant; this removes the
  roughly 20 ms capture-to-detection age without predicting an extra control
  period. Key `TRACK_*` constants are exposed near the top of `main.py`.
  Box centers retain subpixel precision. A missed detection enters bounded
  prediction hold for at most 8 frames and 300 ms; either limit ending first
  changes the state to fully lost.

## Visual state feedback

`X_CALIBRATION_MODE` in `main.py` selects one of two precomputed 320-entry X
tables. The default `"measured_distortion"` mode corrects the installed camera
from this physical-ruler comparison (old displayed coordinate -> actual cm):

```text
0 -> 0, 2.5 -> 3.0, 5.0 -> 5.8, 7.5 -> 8.5,
10.0 -> 10.7, 12.5 -> 12.5
```

Negative anchors are mirrored about zero. A monotone shape-preserving cubic
fit was generated offline in pixel space, then sampled into the fixed table;
it passes through the measured anchors without the between-point oscillation
possible with a high-order global polynomial. Runtime conversion still uses
only adjacent table interpolation. Set the mode to `"pinhole"` to restore the
original `15 cm` distance/`25 cm` field-of-view tangent table unchanged.

`ball_control.py` consumes the selected table for X position and its local
cm/pixel slope for X velocity, derives acceleration and then
calculates only the camera-state feedback component

```text
e = x - x_ref
a_fb = -(Kp*Db(e, 0.05) + Kd*Db(v, 1.0) + Ka*a)
Db(z, width) = sign(z) * max(abs(z) - width, 0)
```

`FEEDBACK_REFERENCE_POSITION_CM` in `main.py` is the absolute calibrated X
coordinate of the current equilibrium origin and defaults to `0.0 cm`.
`BallStateFeedback.set_reference_position_cm()` provides a bounded runtime
interface for a future validated algorithm or MCU command; no MCU-to-K230
reference command is parsed in this version. UART continues to publish absolute
`x/v/a` plus the feedback already computed from `e`. Schema 6 uses a fixed
64-byte frame and also carries the active `x_ref`, `e`, zero-valued reserved
`v_ref/a_ref`, reference source/status flags and six reserved zero bytes.

Defaults are `Kp=16`, `Kd=7.2`, `Ka=0`, with `a_fb` bounded to
`+/-250 cm/s2`. No second physical-state low-pass is applied: `x/v` are converted
directly from the single alpha-beta tracker to avoid additional phase lag.
Continuous soft deadbands of `0.05 cm` for position and `1.0 cm/s` for velocity
apply only when calculating feedback; UART still reports the tracked physical
state itself. Acceleration remains published for diagnostics but is disabled in
the default feedback law. These values are configurable near the top of
`main.py` and are sent with every state frame.

K230 does not receive a moving `X(t)` trajectory, car acceleration or pipe
angle and does not calculate desired pipe angle, PWM or motor commands. The
active stationary `x_ref` is currently the K230 preset transmitted in schema
6. The MCU combines `a_fb` with car feedforward, performs the ball/pipe model
inversion and closes the measured-angle inner loop; it owns any future moving
trajectory generator and validated command path.

The current `(10,160)` to `(310,160)` image endpoints define detector/tracker
geometry only. Physical zero is image `(160,160)`. Y remains a local linear
diagnostic coordinate and uses the selected X table's center scale: about
`0.0789 cm/px` in measured mode or the unchanged `0.0651317134 cm/px` in
pinhole mode. Position, velocity, acceleration, feedback and UART values all
use the selected mode. Recheck the optical center and positive direction before
using `FEEDBACK_VALID` data.

For calibration/debugging, `SHOW_PHYSICAL_X_RULER` in `main.py` optionally
draws a physical X ruler once on a dedicated static `OSD0` layer, below the
image centerline. The default `PHYSICAL_X_RULER_TICK_CM = 0.5` gives minor
ticks every 0.5 cm and numeric labels every 2.5 cm; the supported lower-density
setting is `1.0`. `PHYSICAL_X_RULER_Y` moves the ruler vertically. Tick
positions use the inverse of the selected calibration LUT used by the
control path, so the frame loop performs no ruler lookup or redraw. Disable the
option to restore the original one-OSD memory footprint; the enabled mode still
needs K230 hardware verification for layer persistence and memory margin.

The overlay refreshes near the camera frame cadence so visual inspection does not
add the former roughly 100 ms display-only lag. The raw detector box remains
orange; a stabilized square derived from filtered center and slowly tracked
diameter is cyan, making detector lag distinguishable from observer lag. It
explicitly shows `DETECT`, `HOLD n/8`, or `LOST`, plus observer fast-mode/
innovation diagnostics, image position/velocity, physical `x/v/a`, `a_fb`,
performance and UART diagnostics.
`HOLD` continues bounded prediction feedback while remaining distinguishable
from a real detection. Recoverable errors remain latched; three consecutive
frame errors or an initialization failure enter the visible FATAL page.

See `docs/ROI_BALL_DETECTOR.md` for ROI geometry and tuning,
`docs/EDGE_HOUGH_DETECTOR.md` for the circle backend,
`docs/HOUGH_ROI_PROBE.md` for the independent fixed-ROI compatibility and
timing probe, and
`docs/BALL_CONTROL_LAW.md` for the control responsibility boundary.
`main.py` captures failures while importing CanMV/KPU/protocol dependencies and
uses a minimal emergency OSD path, so a failure before normal camera setup is
not intentionally left as a black screen. The emergency page itself still
depends on the firmware's `image`, `Display` and `MediaManager` stack; if that
display stack is the component that cannot load or initialize, the USB console
is the only remaining error channel.

CanMV may deliver its IDE Stop/Run takeover request as a plain
`Exception("IDE interrupt")` from camera capture, KPU inference or UART write
instead of as `KeyboardInterrupt`. Version `proj2-1.21` recognizes both forms as
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
64-byte state frame after every detector run, including a no-ball state. Frames
are latest-wins in a two-frame (128-byte) software queue; short writes resume at
the next loop and a stalled write cannot grow without bound.

K230 emits one fixed 64-byte schema-6 frame per detector run. It carries raw
pixel coordinates, calibrated physical X/Y position and velocity, physical
X acceleration, `a_fb`, `Kp/Kd/Ka`, detector time, freshness flags, active
`x_ref`, `e=x-x_ref`, reserved reference velocity/acceleration and reference
source/status flags. The final six payload bytes are reserved and zero. This
application does not read MCU control data, does not use
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
physical position/velocity/acceleration, `a_fb`, gains, confidence, detector
time, measurement age, reference state and `x-ref-e` consistency. Some inexpensive USB-TTL
adapters or drivers do not support 512000 baud; an open/configuration failure is
therefore an adapter/driver compatibility result rather than proof that the K230
program failed.

### CanMV/K230 debug mode

K230 can execute only one foreground Python application. To use the optional
board-side debug wrapper, first click **Stop** in CanMV and wait for
`steel-ball tracker stopped by IDE/user`; then open `pc_uart_test.py` and run it
as a temporary script while the real proj2 `main.py` remains at
`/sdcard/main.py`. It imports that tracker entry and calls
`main.run(pc_debug=True)`: camera detection, the normal screen overlay and UART1
transmission continue, and one `PCDBG` status line is mirrored to the USB/IDE
console about every 500 ms. Each line includes sequence, flags, raw pixels,
filtered pixels, calibrated physical X/Y state, `a_fb`, gain/validity
status, confidence, split performance timings, candidate/association/
acquisition diagnostics, TX frame/byte/queue counters,
exception/short/zero/drop counters and the last UART/runtime error text.

Do not burn `pc_uart_test.py` *as* `/sdcard/main.py`; it is a wrapper and expects
the production tracker entry to stay there. The K230 debug mode verifies the
application path and reports software TX results, but USB console output does
not prove that the electrical UART1 IO9 waveform reaches a PC. Use desktop
receiver mode plus the 3.3 V USB-TTL wiring above for that end-to-end check.

## MCU side

The MCU decoder must be updated for schema 6/kind 5 and a fixed 64-byte frame.
The first 32 payload bytes preserve the schema-5 field offsets. Raw detector coordinates
remain pixels, while the former filtered pixel X/Y and velocity fields now carry
signed physical X/Y centimeters and centimeters per second. The established
axial X/V/A/feedback offsets remain unchanged; the 16-byte extension carries
the current reference and reserves space for later trajectory work. Three states use existing
flags: `MEASUREMENT_ACCEPTED+TRACK_VALID+FEEDBACK_VALID` is `DETECT`;
`PREDICTED+TRACK_VALID+FEEDBACK_VALID` is bounded `HOLD`; absence of
`TRACK_VALID` or `FEEDBACK_VALID` is `LOST`. A stale, CRC-invalid,
invalid-feedback or fully lost frame must not directly affect an actuator. The
current stationary `x_ref` is owned by the K230 preset and transmitted in every
frame. The MCU owns any future moving-trajectory generation and command path,
car feedforward, pipe-angle solution, measured-angle inner loop and all
actuator safety.

## Verification boundary

Host checks validate detector selection, ROI/Hough geometry and scoring with
simulated CanMV objects, YOLO candidate normalization, association rules, feedback
equations, fixed-point protocol and parser recovery. They cannot establish
RGB565 threshold behavior, K230 execution time or camera differentiation noise. The
camera currently requests 60 FPS; use the measured on-screen FPS and frame
period to determine whether the active sensor/driver mode delivers it.
Use the backend-specific on-screen timing labels and log `x/v/a/a_fb` on
hardware. YOLO shows `cap/ai2d/kpu/post/det/fb`; ROI shows
`cap/mask/blob/select/det/fb`; Hough shows
`cap/crop/hough/select/det/fb`.
Loading each KModel and validating ROI threshold/geometry under the fixed camera
and contest lighting require K230 target-board tests. KModel output shape, score
range, recognition quality, KPU time and memory headroom also remain target
checks. Display alignment, UART waveform and end-to-end MCU
reception likewise remain hardware verification items.
