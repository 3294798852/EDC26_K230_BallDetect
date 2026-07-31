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

Native Hough can return several nearby circles for one physical ball. The
bounded Python postprocess clusters at most 24 returned circles by compatible
center and radius, then publishes the median center and radius of each cluster.
Cluster support, maximum center/radius spread and geometric consistency form a
measurement-quality value and an acquisition score. Median fusion prevents the
selected orange box from alternating between different arc hypotheses; a
supported cluster normally outranks an isolated false circle during acquisition.
An excessively spread cluster is retained but marked ambiguous for stricter
motion confirmation instead of being silently discarded.

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
| `HOUGH_MIN_CENTER_DISTANCE_PX` | 8 px | native suppression of nearby circle centers |
| `HOUGH_CANNY_HIGH_THRESHOLD` | 80 | internal Canny high threshold (`param1`) |
| `HOUGH_ACCUMULATOR_THRESHOLD` | 9 | minimum circle votes (`param2`) |
| `HOUGH_MIN/MAX_RADIUS_PX` | 3/9 px | reject noise and large pipe/shadow structures |
| `HOUGH_TARGET_RADIUS_PX` | 5.5 px | preferred radius before a tracked diameter exists |
| `HOUGH_MAX_RAW_CIRCLES` | 24 | hard Python postprocessing bound |
| `HOUGH_CLUSTER_CENTER_FACTOR` | 1.0 | maximum same-ball center spread relative to diameter |
| `HOUGH_CLUSTER_RADIUS_TOLERANCE_PX` | 2 px | maximum radius difference inside one cluster |
| `HOUGH_TRACKED_RADIUS_TOLERANCE_PX` | 3 px | reject implausible radius changes near an active track |
| `HOUGH_AMBIGUOUS_SPREAD_RADIUS_FACTOR` | 0.65 | flag a wide fused cluster for stricter motion confirmation |

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
H<accepted>/<clusters>/<raw> S<support> D<center>/<radius> Q<quality>[A]
h<hint> P<Canny>/<accumulator>
```

`S` is the selected cluster's circle count, `D` gives maximum spread in pixels,
`Q` is quality in percent, and `A` marks an ambiguous spread. `h=1` means a
non-expired motion prediction is available. A `circles clipped` warning means
native Hough returned more than the bounded raw-candidate limit. The following
motion suffix adds `W` when the selected center is outside the strict quiet gate
and `C` when another similarly close candidate competes with it. Both are
diagnostics only; UART schema 6 is unchanged.

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

The current geometry was tightened before changing vote sensitivity: radius is
`3..9 px`, native center separation is `8 px`, and cluster center/radius
tolerances are `1.0`/`2 px`. The accumulator threshold deliberately remains
`9` to preserve partial-arc and motion-blur recall. If a static scene still
reports raw counts near `20` after this change, and the real ball remains
reliably detected, test `HOUGH_ACCUMULATOR_THRESHOLD = 10` as a separate final
step. `HOUGH_MAX_RAW_CIRCLES` is only a Python work bound and does not suppress
native false circles or reduce native Hough time.

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

The observer uses an internal `QUIET/PENDING/MOVING` state. Reliable motion
requires two same-direction raw-center steps; an ambiguous fused cluster
normally requires three. Low `Q`, `S<2`, spread `A`, or a close competing
candidate also makes that measurement ambiguous. Direction-alternating Hough
hypotheses therefore receive only slow position correction and cannot update
velocity. There is no quiet-state one-frame fast path: even a reliable `14 px`
step first enters `PENDING`. A second reliable same-direction step confirms
`MOVING`, retains the velocity anchor from before the first step, and restores
the fast correction. Its association gate expands only in the pending direction
and is bounded by the normal maximum gate, allowing very fast travel without
relaxing ordinary quiet tracking. If the real ball is still visible near the
quiet prediction, it has priority over every distant continuation. Once already
`MOVING`, a reliable large reversal remains immediate so braking is not delayed.
Confirmed motion restores the innovation-dependent fast axial gain. Lateral
position and diameter use separate strong smoothing and do not weaken axial
response. The screen suffix `M<Q/P/M><count>[F][W][C]` shows this state,
confirmation count, optional fast response, wide-gate selection and close
competition.
