"""Native K230 ROI dark-blob detector for the steel ball."""

import math
import time


def _clamp(value, low, high):
    return max(low, min(high, value))


def _ticks_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.time() * 1000)


def _ticks_diff(new_value, old_value):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(int(new_value), int(old_value))
    return int(new_value) - int(old_value)


def _blob_rect(blob):
    try:
        return blob.rect()
    except Exception:
        return int(blob[0]), int(blob[1]), int(blob[2]), int(blob[3])


def _blob_pixels(blob, width, height):
    try:
        return int(blob.pixels())
    except Exception:
        try:
            return int(blob[4])
        except Exception:
            return int(width) * int(height)


class RoiDarkBlobDetector:
    """Detect one small dark near-round object inside a calibrated pipe ROI.

    The input is a native RGB565 CanMV Image. Processing remains in native image
    methods; no Python pixel loop or KPU allocation is used.
    """

    def __init__(
        self,
        frame_width=320,
        frame_height=320,
        negative_end_x_px=10.0,
        negative_end_y_px=160.0,
        positive_end_x_px=310.0,
        positive_end_y_px=160.0,
        roi_half_width_px=22.0,
        threshold_sample_half_width_px=10.0,
        endpoint_margin_px=4.0,
        min_diameter_px=4.0,
        max_diameter_px=26.0,
        target_diameter_px=11.0,
        min_pixels=12,
        min_aspect_ratio=0.38,
        min_fill_ratio=0.22,
        threshold_ratio=0.60,
        threshold_l_min=20,
        threshold_l_max=62,
        threshold_update_frames=6,
        max_native_blobs=32,
        max_candidates=4,
    ):
        self.model_backend = "roi_blob"
        self.model_display_name = "ROI dark blob"
        self.model_path = "native-image"
        self.expected_output = "dark blobs"
        self.pipeline_name = "roi_dark_blob"
        self.actual_output = None
        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))
        self.negative_end_x_px = float(negative_end_x_px)
        self.negative_end_y_px = float(negative_end_y_px)
        self.positive_end_x_px = float(positive_end_x_px)
        self.positive_end_y_px = float(positive_end_y_px)
        axis_dx = self.positive_end_x_px - self.negative_end_x_px
        axis_dy = self.positive_end_y_px - self.negative_end_y_px
        self.axis_length = math.sqrt(axis_dx * axis_dx + axis_dy * axis_dy)
        if self.axis_length < 20.0:
            raise ValueError("ROI pipe endpoints are too close")
        self.axis_x = axis_dx / self.axis_length
        self.axis_y = axis_dy / self.axis_length
        self.roi_half_width_px = max(4.0, float(roi_half_width_px))
        self.threshold_sample_half_width_px = _clamp(
            float(threshold_sample_half_width_px),
            3.0,
            self.roi_half_width_px,
        )
        self.endpoint_margin_px = max(0.0, float(endpoint_margin_px))
        self.min_diameter_px = max(2.0, float(min_diameter_px))
        self.max_diameter_px = max(
            self.min_diameter_px + 1.0,
            float(max_diameter_px),
        )
        self.target_diameter_px = _clamp(
            float(target_diameter_px),
            self.min_diameter_px,
            self.max_diameter_px,
        )
        self.min_pixels = max(1, int(min_pixels))
        self.min_aspect_ratio = _clamp(float(min_aspect_ratio), 0.05, 1.0)
        self.min_fill_ratio = _clamp(float(min_fill_ratio), 0.05, 1.0)
        self.threshold_ratio = _clamp(float(threshold_ratio), 0.10, 0.95)
        self.threshold_l_min = int(_clamp(int(threshold_l_min), 0, 100))
        self.threshold_l_max = int(_clamp(int(threshold_l_max), 0, 100))
        if self.threshold_l_max < self.threshold_l_min:
            self.threshold_l_max = self.threshold_l_min
        self.threshold_update_frames = max(1, int(threshold_update_frames))
        self.max_native_blobs = max(1, int(max_native_blobs))
        self.max_candidates = max(1, int(max_candidates))
        self.roi = self._build_roi(self.roi_half_width_px)
        self.threshold_sample_roi = self._build_roi(
            self.threshold_sample_half_width_px
        )
        self.threshold_l = self.threshold_l_max
        self.roi_l_median = 0
        self.frame_count = 0
        self.blob_count = 0
        self.native_close_ok = True
        self.mask_thresholds = [(80, 100, -20, 20, -20, 20)]
        self.preprocess_ms = 0
        self.kpu_ms = 0
        self.postprocess_ms = 0
        self.last_detection_count = 0
        self.last_warning = None

    def _build_roi(self, half_width_px):
        margin = float(half_width_px) + self.endpoint_margin_px
        left = int(max(0, math.floor(
            min(self.negative_end_x_px, self.positive_end_x_px) - margin
        )))
        top = int(max(0, math.floor(
            min(self.negative_end_y_px, self.positive_end_y_px) - margin
        )))
        right = int(min(self.frame_width, math.ceil(
            max(self.negative_end_x_px, self.positive_end_x_px) + margin
        )))
        bottom = int(min(self.frame_height, math.ceil(
            max(self.negative_end_y_px, self.positive_end_y_px) + margin
        )))
        width = max(1, right - left)
        height = max(1, bottom - top)
        return left, top, width, height

    def _update_threshold(self, frame):
        if self.frame_count > 1 and self.frame_count % self.threshold_update_frames:
            return
        stats = frame.get_statistics(roi=self.threshold_sample_roi)
        try:
            median_l = int(stats.l_median())
        except Exception:
            median_l = int(stats.l_mean())
        target = int(_clamp(
            round(median_l * self.threshold_ratio),
            self.threshold_l_min,
            self.threshold_l_max,
        ))
        if self.frame_count <= 1:
            self.threshold_l = target
        else:
            self.threshold_l = (self.threshold_l * 3 + target + 2) // 4
        self.roi_l_median = median_l

    def _make_mask(self, frame):
        frame.binary([
            (
                0,
                int(self.threshold_l),
                -128,
                127,
                -128,
                127,
            )
        ])
        if self.native_close_ok:
            try:
                frame.dilate(1)
                frame.erode(1)
            except Exception as exc:
                self.native_close_ok = False
                self.last_warning = "close {}: {}".format(
                    type(exc).__name__,
                    exc,
                )
                print("ROI native close disabled:", self.last_warning)
        return frame

    def _find_blobs(self, mask):
        try:
            return mask.find_blobs(
                self.mask_thresholds,
                roi=self.roi,
                pixels_threshold=self.min_pixels,
                area_threshold=self.min_pixels,
                merge=False,
            )
        except Exception:
            if self.mask_thresholds[0] == (80, 100, -20, 20, -20, 20):
                self.mask_thresholds = [(200, 255)]
                return mask.find_blobs(
                    self.mask_thresholds,
                    roi=self.roi,
                    pixels_threshold=self.min_pixels,
                    area_threshold=self.min_pixels,
                    merge=False,
                )
            raise

    def _axis_coordinates(self, center_x, center_y):
        relative_x = float(center_x) - self.negative_end_x_px
        relative_y = float(center_y) - self.negative_end_y_px
        along = relative_x * self.axis_x + relative_y * self.axis_y
        lateral = -relative_x * self.axis_y + relative_y * self.axis_x
        return along, lateral

    def _candidate_from_blob(self, blob, source_index):
        x, y, width, height = _blob_rect(blob)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return None
        roi_x, roi_y, roi_width, roi_height = self.roi
        if (
            x <= roi_x
            or y <= roi_y
            or x + width >= roi_x + roi_width
            or y + height >= roi_y + roi_height
        ):
            return None
        diameter = (width + height) * 0.5
        if diameter < self.min_diameter_px or diameter > self.max_diameter_px:
            return None
        aspect = min(width, height) / float(max(width, height))
        if aspect < self.min_aspect_ratio:
            return None
        pixels = _blob_pixels(blob, width, height)
        if pixels < self.min_pixels:
            return None
        fill_ratio = pixels / float(max(1, width * height))
        if fill_ratio < self.min_fill_ratio:
            return None
        center_x = x + width * 0.5
        center_y = y + height * 0.5
        along, lateral = self._axis_coordinates(center_x, center_y)
        if (
            along < -self.endpoint_margin_px
            or along > self.axis_length + self.endpoint_margin_px
            or abs(lateral) > self.roi_half_width_px
        ):
            return None

        aspect_score = _clamp(
            (aspect - self.min_aspect_ratio)
            / max(0.001, 1.0 - self.min_aspect_ratio),
            0.0,
            1.0,
        )
        fill_score = _clamp(
            (fill_ratio - self.min_fill_ratio)
            / max(0.001, 0.75 - self.min_fill_ratio),
            0.0,
            1.0,
        )
        size_span = max(
            self.target_diameter_px - self.min_diameter_px,
            self.max_diameter_px - self.target_diameter_px,
            1.0,
        )
        size_score = 1.0 - _clamp(
            abs(diameter - self.target_diameter_px) / size_span,
            0.0,
            1.0,
        )
        axis_score = 1.0 - _clamp(
            abs(lateral) / self.roi_half_width_px,
            0.0,
            1.0,
        )
        confidence = _clamp(
            0.40
            + 0.20 * aspect_score
            + 0.18 * fill_score
            + 0.12 * size_score
            + 0.10 * axis_score,
            0.0,
            1.0,
        )
        return {
            "confidence": confidence,
            "center_x": center_x,
            "center_y": center_y,
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "diameter": diameter,
            "edge_clipped": bool(
                x <= 0
                or y <= 0
                or x + width >= self.frame_width
                or y + height >= self.frame_height
            ),
            "source_index": int(source_index),
        }

    def run(self, frame):
        self.frame_count = (self.frame_count + 1) & 0xFFFFFFFF
        self.preprocess_ms = 0
        self.kpu_ms = 0
        self.postprocess_ms = 0
        self.last_detection_count = 0
        self.blob_count = 0
        preprocess_start = _ticks_ms()
        self._update_threshold(frame)
        mask = self._make_mask(frame)
        preprocess_finished = _ticks_ms()
        self.preprocess_ms = max(
            0,
            _ticks_diff(preprocess_finished, preprocess_start),
        )
        blobs = self._find_blobs(mask)
        try:
            self.blob_count = len(blobs)
        except Exception:
            self.blob_count = 0
        if self.blob_count > self.max_native_blobs:
            self.last_warning = "too many blobs: {}".format(self.blob_count)
            self.postprocess_ms = max(
                0,
                _ticks_diff(_ticks_ms(), preprocess_finished),
            )
            return []

        candidates = []
        for source_index in range(self.blob_count):
            candidate = self._candidate_from_blob(
                blobs[source_index],
                source_index,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )
        if len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        self.last_detection_count = len(candidates)
        self.postprocess_ms = max(
            0,
            _ticks_diff(_ticks_ms(), preprocess_finished),
        )
        return candidates

    def diagnostic_text(self):
        text = "ROI L={}/{} b={}".format(
            int(self.threshold_l),
            int(self.roi_l_median),
            int(self.blob_count),
        )
        if self.last_warning is not None:
            text += " !" + str(self.last_warning)[:20]
        return text

    def deinit(self):
        return None