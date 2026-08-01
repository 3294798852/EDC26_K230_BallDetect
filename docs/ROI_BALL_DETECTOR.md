# K230 ROI steel-ball detector

## When to use it

Set `MODEL_BACKEND = "roi_blob"` in `main.py` when the fixed camera sees a small
steel ball inside a much longer bright pipe and YOLO misses the small target.
The UART schema, tracker, `DETECT/HOLD/LOST` states and feedback law do not
change when the detector backend changes.

The supplied reference photo supports this approach: the pipe interior is a
long bright constrained band, while the steel ball contributes a small dark,
roughly compact component. At 320-pixel input width its dark core is only about
6-12 pixels, so full-frame learned detection can lose detail.

## Native processing path

```text
RGB565 frame
  -> brightness statistics in narrow pipe sample band
  -> adaptive dark LAB threshold
  -> native close (dilate then erode)
  -> one find_blobs call in wider pipe ROI
  -> bounded merge of adjacent dark fragments
  -> strict global gates or prediction-aware tracking gates
  -> cross-pipe diameter and along-pipe motion-blur scoring
  -> at most four scored candidates
  -> existing continuity tracker
```

All image operations are existing CanMV/K230 native methods already used by the
board backup: `get_statistics`, `binary`, `dilate`, `erode` and `find_blobs`.
There is no Python pixel loop and ROI mode does not instantiate KPU/AI2D.
Fragment merging compares each component with at most the six preceding
axis-sorted groups, so its Python work remains bounded even in a noisy frame.

The tracking hint is read-only. It can improve detector recall, but cannot move
the filter or bypass the normal association gate. A three-pixel component or a
long blurred component is accepted with relaxed geometry only inside the
predicted search radius. The same component elsewhere still has to pass the
strict global acquisition rules and the tracker's multi-frame confirmation.

## Configuration

The ROI follows the same two pipe endpoints used for physical calibration, so
measure these first in the 320x320 detector image:

```text
PIPE_NEGATIVE_END_X_PX/Y_PX
PIPE_POSITIVE_END_X_PX/Y_PX
```

Then tune only the exposed `ROI_*` constants in `main.py`:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `ROI_HALF_WIDTH_PX` | 22 px | accepted distance from pipe axis and blob scan band |
| `ROI_THRESHOLD_SAMPLE_HALF_WIDTH_PX` | 10 px | narrow bright band used for illumination statistics |
| `ROI_MIN_DIAMETER_PX` | 3 px | strict global minimum; acquisition still needs three consistent frames |
| `ROI_MAX_DIAMETER_PX` | 26 px | reject pipe edges and large shadows |
| `ROI_TARGET_DIAMETER_PX` | 11 px | preferred ball scale in geometric score |
| `ROI_MIN_PIXELS` | 6 | strict global candidate dark-pixel minimum |
| `ROI_FRAGMENT_MIN_PIXELS` | 3 | native fragment entrance threshold; relaxed candidates require a tracking hint |
| `ROI_FRAGMENT_MERGE_GAP_PX` | 3 px | maximum gap used to reconnect highlight-split dark arcs |
| `ROI_TRACKED_MIN_ASPECT_RATIO` | 0.16 | tracked-candidate limit for along-pipe motion blur |
| `ROI_TRACKED_MIN_FILL_RATIO` | 0.08 | tracked-candidate limit for a bright-centered reflective ball |
| `ROI_MOTION_BLUR_SECONDS` | 0.025 s | converts tracked pixel speed to permitted axial blur length |
| `ROI_MOTION_BLUR_MIN_EXTRA_PX` | 8 px | bounded allowance for the first blurred frame after sudden acceleration |
| `ROI_MOTION_BLUR_MAX_EXTRA_PX` | 24 px | hard upper bound on the axial blur allowance |
| `ROI_THRESHOLD_RATIO` | 0.68 | dark threshold relative to pipe-band L median |
| `ROI_THRESHOLD_L_MIN/MAX` | 20/62 | bounded adaptive LAB-L threshold |

## On-screen diagnostics

ROI mode draws its scan rectangle and reports:

```text
ROI L=<threshold>/<sample median> b=<native blobs> h=<hint> m=<merged>/<blur>
```

`h=1` means a non-expired tracking prediction is available. The first `m`
value counts fragments absorbed into neighboring groups; the second counts
accepted along-pipe motion-blurred candidates.

Use this order when tuning:

1. Fix the camera and make the drawn ROI cover only the pipe with a small margin.
2. Check that the sample median follows illumination changes and remains much
   brighter than the selected threshold.
3. If a reflective ball appears as two dark arcs, confirm `m` merge activity;
   then increase the merge gap by at most one pixel at a time if needed.
4. If the small ball is absent from the mask, increase `ROI_THRESHOLD_RATIO` or
   `ROI_THRESHOLD_L_MAX` gradually. If noise grows, reverse that change before
   lowering pixel gates.
5. For fast motion, confirm that blur is along the calibrated pipe axis and
   `m` blur activity increases. Adjust the blur time before its hard maximum.
6. If scratches/background create many blobs, narrow the ROI or tighten the
   diameter/fill/aspect limits. Do not broadly enlarge the prediction radius.
7. Verify `DETECT` while moving the ball end-to-end, then verify bounded `HOLD`
   and `LOST` behavior before enabling actuators.

A morphology API incompatibility is reported once to the console and latched in
the diagnostic text; detection continues without close. Failure of threshold or
blob APIs is not swallowed and enters the existing visible runtime/FATAL path.

## Verification boundary

Host simulations cover strict acquisition, a 3x3 predicted small target,
prediction-external noise rejection, highlight-split fragment merging,
along-pipe motion blur and wrong-direction streak rejection. RGB565 LAB values,
real specular reflections, blur length, camera exposure, frame time and final
thresholds still require the actual K230 camera at its fixed mounting and
contest lighting.
## Low-lag one-dimensional tracking

Detector candidates from this backend feed the shared fixed-axis observer. The
calibrated `PIPE_*_END_*` line is active immediately; the observer no longer
waits for enough motion to learn an axis. Axial innovation outside the small
noise band receives fast position and reversal correction, while cross-pipe
position and apparent diameter remain strongly smoothed.

The candidate is timestamped at camera capture completion, not inference
completion. After correction the state is advanced only through the measured
detector latency before UART publication. The screen updates near camera rate:
orange is the raw ROI candidate and cyan is the filtered fixed-diameter track.
If both boxes trail the ball, improve exposure or ROI candidate geometry. If
only cyan trails orange, tune the `TRACK_AXIAL_*` observer constants.

Axial position gain rises with innovation and reaches its fast value by
`TRACK_AXIAL_RESPONSE_FULL_PX`. The current low-lag profile uses `0.40` as its
base axial gain, reaches `0.96` at `0.75 px`, and uses `0.50/0.90` as the normal
and fast velocity gains. Before motion confirmation, reliable/ambiguous
position gains are `0.24/0.18`; the two/three-frame confirmation rule remains
unchanged to reject one-frame false circles. Axial velocity accumulates subpixel movement
until either `TRACK_VELOCITY_NOISE_PX` is crossed or
`TRACK_VELOCITY_QUIET_TIMEOUT_MS` confirms a stationary interval; lowering the
distance threshold alone converts detector jitter into speed. A sufficiently
large innovation opposite to the current velocity activates a short braking
latch for fast stops and reversals. Lateral position and diameter use separate
strong smoothing and do not weaken axial response.
