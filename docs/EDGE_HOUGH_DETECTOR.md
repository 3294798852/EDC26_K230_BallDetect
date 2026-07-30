# K230 edge-Hough steel-ball detector

## Selection and processing path

Set `MODEL_BACKEND = "edge_hough"` in `main.py`. The UART schema, motion
filter, three tracking states and feedback law remain unchanged.

```text
320x320 RGB565 frame
  -> verified image.copy(roi=...) fixed pipe-band crop
  -> native RGB888 conversion of the cropped image
  -> cv_lite internal Canny edge extraction and Hough-circle voting
  -> at most 24 raw circle triples
  -> radius, pipe-axis and endpoint gates
  -> prediction/observed-radius scoring when a track exists
  -> near-duplicate circle suppression
  -> at most four candidates
  -> existing continuity tracker and three-frame acquisition gate
```

`cv_lite.rgb888_find_circles` is a verified K230 firmware API and returns a
flat `[x0, y0, r0, ...]` sequence. Its `param1` argument is the internal Canny
high threshold and `param2` is the Hough accumulator threshold. Do not add a
separate edge-extraction pass: that would repeat work and allocate another
image buffer.

The pipe endpoints and crop half-width form a bounding rectangle. Its horizontal
extent is expanded outward to 16-pixel boundaries while its vertical extent is
kept exact. With the current horizontal calibration this is `(0,120,320,80)`.
The expensive per-pixel edge and vote operations therefore run only on 25% of
the original pixels. Python does not sample pixels or circle
circumferences and processes no more than 24 raw circles per frame. Local ROI
circle coordinates are translated back to the original 320x320 coordinates
before any gate, tracker, display or UART code sees them.

## Reflection and pipe-edge robustness

A reflective ball may lose the edge around its bright center, but its remaining
outer arcs still vote for a common center and radius. The default accumulator
threshold is deliberately low but bounded so a partial circumference can pass.
Unlike the dark-blob backend, this path does not require a closed filled region.

Long pipe walls mainly produce straight edges rather than coherent votes for a
small radius. Returned circles must additionally have an allowed radius, lie
near the calibrated pipe axis and remain within its endpoints. During tracking,
predicted center and measured diameter increase the score of the continuous
candidate. During acquisition, the existing tracker still requires three
spatially consistent frames, so a one-frame false circle is not published.

This method cannot reconstruct a circle when motion blur removes almost all
curvature. Keep `roi_blob` and both YOLO backends available for comparison; no
single backend is selected automatically.

## Configuration

The detector uses the shared `PIPE_*_END_*` calibration. Tune these first, then
adjust the `HOUGH_*` constants in `main.py`:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `HOUGH_CROP_HALF_WIDTH_PX` | 40 px | perpendicular crop padding; gives the current 80 px ROI height |
| `HOUGH_AXIS_CENTER_LIMIT_PX` | 28 px | maximum accepted circle-center distance from the pipe axis |
| `HOUGH_ENDPOINT_MARGIN_PX` | 4 px | small accepted extension beyond each pipe endpoint |
| `HOUGH_DP` | 1 | Hough accumulator resolution ratio |
| `HOUGH_MIN_CENTER_DISTANCE_PX` | 6 px | native suppression of nearby circle centers |
| `HOUGH_CANNY_HIGH_THRESHOLD` | 80 | internal Canny high threshold (`param1`) |
| `HOUGH_ACCUMULATOR_THRESHOLD` | 9 | minimum circle votes (`param2`) |
| `HOUGH_MIN/MAX_RADIUS_PX` | 2/14 px | reject noise and large pipe/shadow structures |
| `HOUGH_TARGET_RADIUS_PX` | 5.5 px | preferred radius before a tracked diameter exists |
| `HOUGH_MAX_RAW_CIRCLES` | 24 | hard Python postprocessing bound |

Lowering the Canny threshold admits weaker edges but also pipe texture. Lowering
the accumulator threshold tolerates a shorter visible arc but increases false
circles. Change one threshold at a time and watch both candidate counts and
frame time.

The crop and center gate are deliberately separate. Keep the crop wide enough
to preserve the full circumference and the native detector's border context;
use the center gate to reject background instead of trimming the crop to the
visible pipe wall. A returned circle that touches the cropped-image boundary is
marked `edge_clipped`: it cannot establish a new track, but an existing
prediction may follow it through the tracker's stricter edge association gate.

## Diagnostics and tuning

The screen and PC debug line identify `edge_hough`, draw the pipe search band
and report:

```text
Hough c=<accepted>/<raw> h=<hint> p=<Canny>/<accumulator>
```

`h=1` means a non-expired motion prediction is available. A `circles clipped`
warning means native Hough returned more than the bounded raw-candidate limit.

Tune on the actual fixed camera in this order:

1. Calibrate the two pipe endpoints and make the drawn band cover the ball
   centers with little background.
2. Measure the ball radius at both ends and tighten `MIN/MAX_RADIUS` around that
   range.
3. Raise the Canny threshold until wall texture mostly disappears, then reduce
   it only enough to recover weak ball arcs.
4. Raise the accumulator threshold until isolated false circles disappear;
   lower it only if highlights consistently split the real circumference.
5. If a static ball is detected but a fast ball becomes a long axial streak,
   set `CAMERA_EXPOSURE_US = 5000` in `main.py`, then reduce it toward
   `3000-2000 us` as lighting permits. Exposure must be shortened before
   lowering Hough thresholds further because a severe streak is no longer a
   circle. `None` retains the firmware's original exposure behavior.
6. Move the ball end-to-end and verify `DETECT`, bounded `HOLD`, `LOST`, raw
   circle count and `crop/hough/select/det` timing before enabling control
   output. `hough` is the native Canny/Hough call and does not indicate KPU use.

Host simulation can verify parsing, gates, scoring, duplicate suppression and
bounds. Only K230 hardware can establish actual `cv_lite` execution time,
camera-edge quality and final thresholds under contest lighting and motion.
## Low-lag one-dimensional tracking

Hough circle centers use the same fixed-axis observer as the other backends.
Correction is timestamped at camera capture completion and the state is then
advanced only through the measured Hough processing latency before UART
publication. Axial response becomes fast when prediction innovation grows;
cross-pipe motion and apparent radius changes are strongly suppressed.

The screen updates near camera rate. Orange shows the selected raw circle box;
cyan shows the filtered center with the slowly tracked diameter. If orange and
cyan both trail the real ball, the cause is capture exposure or detector
geometry rather than the state filter. If only cyan trails orange, adjust the
`TRACK_AXIAL_*` constants in `main.py`.

Axial position gain rises with innovation and reaches its fast value by
`TRACK_AXIAL_RESPONSE_FULL_PX`. Axial velocity accumulates subpixel movement
until either `TRACK_VELOCITY_NOISE_PX` is crossed or
`TRACK_VELOCITY_QUIET_TIMEOUT_MS` confirms a stationary interval; lowering the
distance threshold alone converts detector jitter into speed. A sufficiently
large innovation opposite to the current velocity activates a short braking
latch for fast stops and reversals. Lateral position and diameter use separate
strong smoothing and do not weaken axial response.
