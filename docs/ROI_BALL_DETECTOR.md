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
  -> size/fill/aspect/axis geometric gates
  -> at most four scored candidates
  -> existing continuity tracker
```

All image operations are existing CanMV/K230 native methods already used by the
board backup: `get_statistics`, `binary`, `dilate`, `erode` and `find_blobs`.
There is no Python pixel loop and ROI mode does not instantiate KPU/AI2D.

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
| `ROI_MIN_DIAMETER_PX` | 4 px | reject tiny sensor noise |
| `ROI_MAX_DIAMETER_PX` | 26 px | reject pipe edges and large shadows |
| `ROI_TARGET_DIAMETER_PX` | 11 px | preferred ball scale in geometric score |
| `ROI_MIN_PIXELS` | 12 | minimum dark connected-component pixels |
| `ROI_THRESHOLD_RATIO` | 0.60 | dark threshold relative to pipe-band L median |
| `ROI_THRESHOLD_L_MIN/MAX` | 20/62 | bounded adaptive LAB-L threshold |

## On-screen diagnostics

ROI mode draws its scan rectangle and reports:

```text
ROI L=<threshold>/<sample median> b=<native blobs>
```

Use this order when tuning:

1. Fix the camera and make the drawn ROI cover only the pipe with a small margin.
2. Check that the sample median follows illumination changes and remains much
   brighter than the selected threshold.
3. If the ball is absent from the mask, increase `ROI_THRESHOLD_RATIO` or
   `ROI_THRESHOLD_L_MAX` gradually.
4. If scratches/background create many blobs, reduce the ratio, narrow the ROI,
   or tighten diameter/fill/aspect limits.
5. Verify `DETECT` while moving the ball end-to-end, then verify bounded `HOLD`
   and `LOST` behavior before enabling actuators.

A morphology API incompatibility is reported once to the console and latched in
the diagnostic text; detection continues without close. Failure of threshold or
blob APIs is not swallowed and enters the existing visible runtime/FATAL path.

## Verification boundary

The reference photograph and host tests validate initial scale, ROI geometry,
candidate scoring and bounded error paths only. RGB565 LAB values, reflections,
pipe scratches, camera exposure, frame time and final thresholds require the
actual K230 camera at its fixed mounting and contest lighting.