"""K230 cv_lite Canny-edge Hough-circle steel-ball detector."""

import math
import time


try:
    import cv_lite
    _CV_LITE_IMPORT_ERROR = None
except Exception as exc:
    cv_lite = None
    _CV_LITE_IMPORT_ERROR = "{}: {}".format(type(exc).__name__, exc)


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


class EdgeHoughCircleDetector:
    """Detect small circles near a calibrated straight-pipe axis.

    cv_lite performs Canny extraction and Hough voting in native code. Python
    only filters a bounded number of returned (x, y, radius) triples.
    """

    def __init__(
        self,
        frame_width=320,
        frame_height=320,
        negative_end_x_px=10.0,
        negative_end_y_px=160.0,
        positive_end_x_px=310.0,
        positive_end_y_px=160.0,
        crop_half_width_px=40.0,
        axis_center_limit_px=28.0,
        endpoint_margin_px=4.0,
        dp=1,
        min_center_distance_px=6,
        canny_high_threshold=80,
        accumulator_threshold=8,
        min_radius_px=2,
        max_radius_px=14,
        target_radius_px=5.5,
        max_raw_circles=24,
        max_candidates=4,
    ):
        if cv_lite is None:
            raise RuntimeError(
                "cv_lite unavailable: " + str(_CV_LITE_IMPORT_ERROR)
            )
        self.model_backend = "edge_hough"
        self.model_display_name = "Canny Hough circle"
        self.model_path = "native-cv_lite"
        self.expected_output = "flat x,y,r triples"
        self.pipeline_name = "cv_lite_canny_hough"
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
            raise ValueError("Hough pipe endpoints are too close")
        self.axis_x = axis_dx / self.axis_length
        self.axis_y = axis_dy / self.axis_length
        self.crop_half_width_px = max(3.0, float(crop_half_width_px))
        self.axis_center_limit_px = _clamp(
            float(axis_center_limit_px),
            2.0,
            self.crop_half_width_px - 1.0,
        )
        self.endpoint_margin_px = max(0.0, float(endpoint_margin_px))
        self.dp = max(1, int(dp))
        self.min_center_distance_px = max(1, int(min_center_distance_px))
        self.canny_high_threshold = max(1, int(canny_high_threshold))
        self.accumulator_threshold = max(1, int(accumulator_threshold))
        self.min_radius_px = max(1, int(min_radius_px))
        self.max_radius_px = max(
            self.min_radius_px + 1,
            int(max_radius_px),
        )
        self.target_radius_px = _clamp(
            float(target_radius_px),
            float(self.min_radius_px),
            float(self.max_radius_px),
        )
        self.max_raw_circles = max(1, int(max_raw_circles))
        self.max_candidates = max(1, int(max_candidates))
        self.roi = self._build_roi()
        self.roi_x = int(self.roi[0])
        self.roi_y = int(self.roi[1])
        self.roi_width = int(self.roi[2])
        self.roi_height = int(self.roi[3])
        self.image_shape = [self.roi_height, self.roi_width]
        self.preprocess_ms = 0
        self.kpu_ms = 0
        self.postprocess_ms = 0
        self.timing_labels = ("crop", "hough", "select")
        self.raw_circle_count = 0
        self.processed_circle_count = 0
        self.last_detection_count = 0
        self.hint_active = False
        self.last_warning = None
        self.roi_shape_checked = False

    def _build_roi(self):
        # Build the bounding rectangle from the two axis endpoints, independent
        # along-axis endpoint padding and perpendicular crop padding. This keeps
        # the crop wider than the later logical center gate without coupling
        # either value to the detected circle radius.
        normal_x = -self.axis_y
        normal_y = self.axis_x
        start_x = self.negative_end_x_px - self.axis_x * self.endpoint_margin_px
        start_y = self.negative_end_y_px - self.axis_y * self.endpoint_margin_px
        end_x = self.positive_end_x_px + self.axis_x * self.endpoint_margin_px
        end_y = self.positive_end_y_px + self.axis_y * self.endpoint_margin_px
        corners = (
            (
                start_x + normal_x * self.crop_half_width_px,
                start_y + normal_y * self.crop_half_width_px,
            ),
            (
                start_x - normal_x * self.crop_half_width_px,
                start_y - normal_y * self.crop_half_width_px,
            ),
            (
                end_x + normal_x * self.crop_half_width_px,
                end_y + normal_y * self.crop_half_width_px,
            ),
            (
                end_x - normal_x * self.crop_half_width_px,
                end_y - normal_y * self.crop_half_width_px,
            ),
        )
        left = int(max(0, math.floor(min(point[0] for point in corners))))
        top = int(max(0, math.floor(min(point[1] for point in corners))))
        right = int(min(
            self.frame_width,
            math.ceil(max(point[0] for point in corners)),
        ))
        bottom = int(min(
            self.frame_height,
            math.ceil(max(point[1] for point in corners)),
        ))
        # Preserve the verified fast full-width path by aligning only the
        # horizontal extent. The current horizontal calibration then maps to
        # (0, 120, 320, 80), as requested, instead of expanding vertically to
        # the next 16-pixel boundary.
        alignment = 16
        left = max(0, (left // alignment) * alignment)
        right = min(
            self.frame_width,
            ((right + alignment - 1) // alignment) * alignment,
        )
        return left, top, max(1, right - left), max(1, bottom - top)

    def _axis_coordinates(self, center_x, center_y):
        relative_x = float(center_x) - self.negative_end_x_px
        relative_y = float(center_y) - self.negative_end_y_px
        along = relative_x * self.axis_x + relative_y * self.axis_y
        lateral = -relative_x * self.axis_y + relative_y * self.axis_x
        return along, lateral

    def _hint_values(self, tracking_hint):
        if tracking_hint is None:
            return None
        try:
            expected_radius = float(tracking_hint.get("diameter", 0.0)) * 0.5
            if expected_radius <= 0.0:
                expected_radius = self.target_radius_px
            return {
                "center_x": float(tracking_hint.get("center_x")),
                "center_y": float(tracking_hint.get("center_y")),
                "search_radius": max(
                    1.0,
                    float(tracking_hint.get("search_radius_px", 0.0)),
                ),
                "expected_radius": _clamp(
                    expected_radius,
                    float(self.min_radius_px),
                    float(self.max_radius_px),
                ),
            }
        except Exception:
            return None

    def _candidate(self, center_x, center_y, radius, hint_values, source_index):
        center_x = float(center_x)
        center_y = float(center_y)
        radius = float(radius)
        if radius < self.min_radius_px or radius > self.max_radius_px:
            return None
        along, lateral = self._axis_coordinates(center_x, center_y)
        center_lateral_limit = self.axis_center_limit_px
        if (
            along < -self.endpoint_margin_px
            or along > self.axis_length + self.endpoint_margin_px
            or abs(lateral) > center_lateral_limit
        ):
            return None

        expected_radius = self.target_radius_px
        tracked_candidate = False
        prediction_score = 0.0
        if hint_values is not None:
            dx = center_x - hint_values["center_x"]
            dy = center_y - hint_values["center_y"]
            predicted_distance = math.sqrt(dx * dx + dy * dy)
            tracked_candidate = predicted_distance <= hint_values["search_radius"]
            if tracked_candidate:
                expected_radius = hint_values["expected_radius"]
                prediction_score = 1.0 - _clamp(
                    predicted_distance / hint_values["search_radius"],
                    0.0,
                    1.0,
                )

        radius_span = max(
            expected_radius - self.min_radius_px,
            self.max_radius_px - expected_radius,
            1.0,
        )
        radius_score = 1.0 - _clamp(
            abs(radius - expected_radius) / radius_span,
            0.0,
            1.0,
        )
        axis_score = 1.0 - _clamp(
            abs(lateral) / center_lateral_limit,
            0.0,
            1.0,
        )
        # Judge clipping against the actual cropped image, not only the source
        # frame. A circle touching a crop border has incomplete evidence and
        # must not start a new track; an existing track may still follow it
        # through the tracker's tighter edge gate.
        roi_border_distance = min(
            center_x - self.roi_x,
            center_y - self.roi_y,
            self.roi_x + self.roi_width - 1 - center_x,
            self.roi_y + self.roi_height - 1 - center_y,
        )
        border_score = _clamp(
            roi_border_distance / max(1.0, radius + 1.0),
            0.0,
            1.0,
        )
        if tracked_candidate:
            confidence = _clamp(
                0.46
                + 0.24 * prediction_score
                + 0.16 * radius_score
                + 0.14 * axis_score,
                0.0,
                1.0,
            )
        else:
            confidence = _clamp(
                0.30
                + 0.22 * radius_score
                + 0.20 * axis_score
                + 0.10 * border_score,
                0.0,
                0.82,
            )
        left = int(round(center_x - radius))
        top = int(round(center_y - radius))
        right = int(round(center_x + radius))
        bottom = int(round(center_y + radius))
        clipped_left = max(0, left)
        clipped_top = max(0, top)
        clipped_right = min(self.frame_width, right)
        clipped_bottom = min(self.frame_height, bottom)
        edge_clipped = roi_border_distance <= radius + 1.0
        return {
            "confidence": confidence,
            "center_x": center_x,
            "center_y": center_y,
            "x": clipped_left,
            "y": clipped_top,
            "w": max(1, clipped_right - clipped_left),
            "h": max(1, clipped_bottom - clipped_top),
            "diameter": radius * 2.0,
            "radius": radius,
            "edge_clipped": edge_clipped,
            "source_index": int(source_index),
            "tracked_candidate": tracked_candidate,
        }

    @staticmethod
    def _same_circle(first, second):
        dx = first["center_x"] - second["center_x"]
        dy = first["center_y"] - second["center_y"]
        center_distance = math.sqrt(dx * dx + dy * dy)
        center_limit = max(2.0, min(first["radius"], second["radius"]) * 0.75)
        return (
            center_distance <= center_limit
            and abs(first["radius"] - second["radius"]) <= 2.0
        )

    def run(self, frame, tracking_hint=None):
        self.preprocess_ms = 0
        self.kpu_ms = 0
        self.postprocess_ms = 0
        self.raw_circle_count = 0
        self.processed_circle_count = 0
        self.last_detection_count = 0
        self.last_warning = None
        hint_values = self._hint_values(tracking_hint)
        self.hint_active = hint_values is not None

        preprocess_start = _ticks_ms()
        roi_image = frame.copy(roi=self.roi)
        if roi_image is None:
            raise RuntimeError("Hough ROI copy returned None")
        rgb_image = roi_image.to_rgb888()
        rgb_array = rgb_image.to_numpy_ref()
        if not self.roi_shape_checked:
            actual_shape = tuple(int(value) for value in rgb_array.shape)
            expected_shape = (self.roi_height, self.roi_width, 3)
            if actual_shape != expected_shape:
                raise RuntimeError(
                    "Hough ROI RGB shape {} != {}".format(
                        actual_shape,
                        expected_shape,
                    )
                )
            self.actual_output = actual_shape
            self.roi_shape_checked = True
        preprocess_finished = _ticks_ms()
        self.preprocess_ms = max(
            0,
            _ticks_diff(preprocess_finished, preprocess_start),
        )
        try:
            circles = cv_lite.rgb888_find_circles(
                self.image_shape,
                rgb_array,
                self.dp,
                self.min_center_distance_px,
                self.canny_high_threshold,
                self.accumulator_threshold,
                self.min_radius_px,
                self.max_radius_px,
            )
        finally:
            del rgb_array
            del rgb_image
            del roi_image
        hough_finished = _ticks_ms()
        self.kpu_ms = max(0, _ticks_diff(hough_finished, preprocess_finished))

        if circles is None:
            circles = ()
        value_count = len(circles)
        self.raw_circle_count = value_count // 3
        if value_count % 3:
            self.last_warning = "circle output tail={}".format(value_count % 3)
        process_count = min(self.raw_circle_count, self.max_raw_circles)
        self.processed_circle_count = process_count
        if self.raw_circle_count > self.max_raw_circles:
            self.last_warning = "circles clipped {}/{}".format(
                self.max_raw_circles,
                self.raw_circle_count,
            )

        candidates = []
        for circle_index in range(process_count):
            offset = circle_index * 3
            candidate = self._candidate(
                float(circles[offset]) + self.roi_x,
                float(circles[offset + 1]) + self.roi_y,
                circles[offset + 2],
                hint_values,
                circle_index,
            )
            if candidate is not None:
                candidates.append(candidate)
        del circles
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        selected = []
        for candidate in candidates:
            duplicate = False
            for existing in selected:
                if self._same_circle(candidate, existing):
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(candidate)
            if len(selected) >= self.max_candidates:
                break
        self.last_detection_count = len(selected)
        self.postprocess_ms = max(0, _ticks_diff(_ticks_ms(), hough_finished))
        return selected

    def diagnostic_text(self):
        text = "Hough c={}/{} h={} r={}x{} p={}/{}".format(
            int(self.last_detection_count),
            int(self.raw_circle_count),
            1 if self.hint_active else 0,
            self.roi_width,
            self.roi_height,
            int(self.canny_high_threshold),
            int(self.accumulator_threshold),
        )
        if self.last_warning is not None:
            text += " !" + str(self.last_warning)[:20]
        return text

    def deinit(self):
        return None
