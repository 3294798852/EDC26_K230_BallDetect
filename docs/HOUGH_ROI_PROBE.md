# Hough fixed-ROI compatibility probe

`hough_roi_probe.py` is an independent K230/CanMV diagnostic. It does not
replace or import `main.py`, initialize UART, load a KModel, or run the control
law. Use it before changing the production Hough backend because the local K230
examples verify only full-frame `to_rgb888()` plus
`cv_lite.rgb888_find_circles()`; they do not document a circle-detector ROI
argument or an image-crop API.

## Running it

1. Keep the production application burned as `/sdcard/main.py`.
2. In CanMV, stop the running application and wait for its normal stop message.
3. Open `hough_roi_probe.py` from the project and run it as a temporary IDE
   script. Do not burn it as `main`.
4. Put the ball inside the cyan `320x64` pipe band and record the LCD or IDE
   results for at least several seconds.

The probe tests, in order:

- `image.copy(roi=(x,y,w,h))`;
- `image.copy((x,y,w,h))`;
- an RGB888 ROI view followed by `ndarray.copy()`;
- an RGB888 ROI view copied through `ulab.numpy.array()`.

Every call is guarded. A failed candidate is reported as `ERR <type>:<message>`
on the LCD and IDE; it does not silently select that path. A candidate is
accepted only when it produces an HWC `64x320x3` array and that array completes
one real `cv_lite.rgb888_find_circles()` call. The first passing method is then
run continuously. Full-frame Hough is sampled every ten frames for comparison.

Key output fields are:

```text
FULL color=<ms> hough=<ms> circles=<n>
ROI  crop=<ms>  hough=<ms> circles=<n>
Hough speedup x<ratio>
```

Success requires more than an `OK` API result: ROI runtime errors must remain
zero, the real ball must appear among the ROI circles, and ROI Hough time must
be consistently lower than the full-frame baseline. Circle coordinates drawn
by the probe include the ROI offset.

The probe does not attempt a grayscale input. The verified firmware entry is
named `rgb888_find_circles` and accepts an RGB888 HWC array; its internal Canny
stage already performs the intensity-domain work needed for edge extraction.
Adding a separate Python-visible grayscale conversion would add work unless a
different native circle API is first verified.

AI2D crop is also excluded from this first probe. The repository verifies AI2D
crop for NCHW model preprocessing, but not a contiguous HWC RGB888 output that
can safely be passed to `cv_lite`. If all four direct methods fail, write and run
a separate AI2D layout probe rather than assuming compatibility in `main.py`.

## Verified target result

The target probe selected `image.copy(roi=...)` with a `64x320x3` RGB888 array:

```text
Hough speedup x5.03
image.copy(roi=) OK, crop 0 ms, ROI Hough 3 ms, 22 circles
image.copy(tuple) rejected: it returned the full 320x320 image
ndarray.copy OK, but required full-frame RGB conversion and took 3/4 ms
np.array(view) rejected for production: 33 ms Hough and 159 circles
```

The production edge-Hough backend therefore uses only the keyword
`image.copy(roi=...)` path. The alternatives remain probe evidence and are not
used as automatic runtime fallbacks.
