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
        min_pixels=6,
        fragment_min_pixels=3,
        fragment_merge_gap_px=3.0,
        min_aspect_ratio=0.38,
        min_fill_ratio=0.18,
        tracked_min_aspect_ratio=0.16,
        tracked_min_fill_ratio=0.08,
        motion_blur_seconds=0.025,
        motion_blur_min_extra_px=8.0,
        motion_blur_max_extra_px=24.0,
        threshold_ratio=0.68,
        threshold_l_min=20,
        threshold_l_max=62,
        threshold_update_frames=6,
        max_native_blobs=48,
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
        self.fragment_min_pixels = max(
            1,
            min(self.min_pixels, int(fragment_min_pixels)),
        )
        self.fragment_merge_gap_px = _clamp(
            float(fragment_merge_gap_px),
            0.0,
            8.0,
        )
        self.max_fragment_neighbors = 6
        self.min_aspect_ratio = _clamp(float(min_aspect_ratio), 0.05, 1.0)
        self.min_fill_ratio = _clamp(float(min_fill_ratio), 0.05, 1.0)
        self.tracked_min_aspect_ratio = _clamp(
            float(tracked_min_aspect_ratio),
            0.05,
            self.min_aspect_ratio,
        )
        self.tracked_min_fill_ratio = _clamp(
            float(tracked_min_fill_ratio),
            0.02,
            self.min_fill_ratio,
        )
        self.motion_blur_seconds = _clamp(
            float(motion_blur_seconds),
            0.0,
            0.050,
        )
        self.motion_blur_max_extra_px = _clamp(
            float(motion_blur_max_extra_px),
            0.0,
            64.0,
        )
        self.motion_blur_min_extra_px = _clamp(
            float(motion_blur_min_extra_px),
            0.0,
            self.motion_blur_max_extra_px,
        )
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
        self.timing_labels = ("mask", "blob", "select")
        self.last_detection_count = 0
        self.hint_active = False
        self.merged_fragment_count = 0
        self.motion_candidate_count = 0
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
                pixels_threshold=self.fragment_min_pixels,
                area_threshold=self.fragment_min_pixels,
                merge=False,
            )
        except Exception:
            if self.mask_thresholds[0] == (80, 100, -20, 20, -20, 20):
                self.mask_thresholds = [(200, 255)]
                return mask.find_blobs(
                    self.mask_thresholds,
                    roi=self.roi,
                    pixels_threshold=self.fragment_min_pixels,
                    area_threshold=self.fragment_min_pixels,
                    merge=False,
                )
            raise

    def _axis_coordinates(self, center_x, center_y):
        relative_x = float(center_x) - self.negative_end_x_px
        relative_y = float(center_y) - self.negative_end_y_px
        along = relative_x * self.axis_x + relative_y * self.axis_y
        lateral = -relative_x * self.axis_y + relative_y * self.axis_x
        return along, lateral

    def _projected_extents(self, width, height):
        along_extent = (
            abs(self.axis_x) * float(width)
            + abs(self.axis_y) * float(height)
        )
        lateral_extent = (
            abs(self.axis_y) * float(width)
            + abs(self.axis_x) * float(height)
        )
        return along_extent, lateral_extent

    def _hint_parameters(self, hint):
        if hint is None:
            return None
        try:
            center_x = float(hint.get("center_x"))
            center_y = float(hint.get("center_y"))
            search_radius = _clamp(
                float(hint.get("search_radius_px", 0.0)),
                self.min_diameter_px * 2.0,
                max(self.frame_width, self.frame_height),
            )
            speed = max(0.0, float(hint.get("speed_px_s", 0.0)))
            diameter = float(hint.get("diameter", 0.0))
        except Exception:
            return None
        if diameter <= 0.0:
            diameter = self.target_diameter_px
        diameter = _clamp(
            diameter,
            self.min_diameter_px,
            self.max_diameter_px,
        )
        blur_extra = _clamp(
            speed * self.motion_blur_seconds,
            self.motion_blur_min_extra_px,
            self.motion_blur_max_extra_px,
        )
        return {
            "center_x": center_x,
            "center_y": center_y,
            "search_radius": search_radius,
            "speed": speed,
            "diameter": diameter,
            "blur_extra": blur_extra,
        }

    def _record_from_blob(self, blob, source_index):
        x, y, width, height = _blob_rect(blob)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return None
        pixels = _blob_pixels(blob, width, height)
        if pixels < self.fragment_min_pixels:
            return None
        return {
            "x0": x,
            "y0": y,
            "x1": x + width,
            "y1": y + height,
            "pixels": pixels,
            "source_index": int(source_index),
            "fragments": 1,
        }

    def _records_close(self, first, second):
        first_cx = (first["x0"] + first["x1"]) * 0.5
        first_cy = (first["y0"] + first["y1"]) * 0.5
        second_cx = (second["x0"] + second["x1"]) * 0.5
        second_cy = (second["y0"] + second["y1"]) * 0.5
        first_along, first_lateral = self._axis_coordinates(
            first_cx,
            first_cy,
        )
        second_along, second_lateral = self._axis_coordinates(
            second_cx,
            second_cy,
        )
        first_width = first["x1"] - first["x0"]
        first_height = first["y1"] - first["y0"]
        second_width = second["x1"] - second["x0"]
        second_height = second["y1"] - second["y0"]
        first_along_extent, first_lateral_extent = self._projected_extents(
            first_width,
            first_height,
        )
        second_along_extent, second_lateral_extent = self._projected_extents(
            second_width,
            second_height,
        )
        along_gap = (
            abs(second_along - first_along)
            - (first_along_extent + second_along_extent) * 0.5
        )
        lateral_gap = (
            abs(second_lateral - first_lateral)
            - (first_lateral_extent + second_lateral_extent) * 0.5
        )
        return (
            along_gap <= self.fragment_merge_gap_px
            and lateral_gap <= self.fragment_merge_gap_px
        )

    def _merge_fragments(self, records, hint_values):
        if len(records) < 2 or self.fragment_merge_gap_px <= 0.0:
            return records
        expected_diameter = self.target_diameter_px
        blur_extra = 0.0
        if hint_values is not None:
            expected_diameter = hint_values["diameter"]
            blur_extra = hint_values["blur_extra"]
        merge_diameter = max(
            expected_diameter,
            self.target_diameter_px * 0.75,
        )
        max_merged_along = min(
            self.max_diameter_px + self.motion_blur_max_extra_px,
            max(
                self.min_diameter_px + self.fragment_merge_gap_px,
                merge_diameter * 1.8 + blur_extra,
            ),
        )
        max_merged_lateral = min(
            self.max_diameter_px,
            max(
                self.min_diameter_px + self.fragment_merge_gap_px,
                merge_diameter * 1.6,
            ),
        )
        records.sort(
            key=lambda item: self._axis_coordinates(
                (item["x0"] + item["x1"]) * 0.5,
                (item["y0"] + item["y1"]) * 0.5,
            )[0]
        )
        merged = []
        for record in records:
            best_index = -1
            best_distance = 0.0
            record_cx = (record["x0"] + record["x1"]) * 0.5
            record_cy = (record["y0"] + record["y1"]) * 0.5
            record_along, record_lateral = self._axis_coordinates(
                record_cx,
                record_cy,
            )
            first_index = max(0, len(merged) - self.max_fragment_neighbors)
            for index in range(len(merged) - 1, first_index - 1, -1):
                existing = merged[index]
                if not self._records_close(existing, record):
                    continue
                x0 = min(existing["x0"], record["x0"])
                y0 = min(existing["y0"], record["y0"])
                x1 = max(existing["x1"], record["x1"])
                y1 = max(existing["y1"], record["y1"])
                along_extent, lateral_extent = self._projected_extents(
                    x1 - x0,
                    y1 - y0,
                )
                if (
                    along_extent > max_merged_along
                    or lateral_extent > max_merged_lateral
                ):
                    continue
                existing_cx = (existing["x0"] + existing["x1"]) * 0.5
                existing_cy = (existing["y0"] + existing["y1"]) * 0.5
                existing_along, existing_lateral = self._axis_coordinates(
                    existing_cx,
                    existing_cy,
                )
                distance = abs(record_along - existing_along) + abs(
                    record_lateral - existing_lateral
                )
                if best_index < 0 or distance < best_distance:
                    best_index = index
                    best_distance = distance
            if best_index < 0:
                merged.append(record)
                continue
            existing = merged[best_index]
            existing["x0"] = min(existing["x0"], record["x0"])
            existing["y0"] = min(existing["y0"], record["y0"])
            existing["x1"] = max(existing["x1"], record["x1"])
            existing["y1"] = max(existing["y1"], record["y1"])
            existing["pixels"] += record["pixels"]
            existing["fragments"] += record["fragments"]
            self.merged_fragment_count += record["fragments"]
        return merged

    def _candidate_from_record(self, record, hint_values):
        x = int(record["x0"])
        y = int(record["y0"])
        width = int(record["x1"] - x)
        height = int(record["y1"] - y)
        if width <= 0 or height <= 0:
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

        predicted_distance = 0.0
        tracked_candidate = False
        expected_diameter = self.target_diameter_px
        blur_extra = 0.0
        if hint_values is not None:
            dx = center_x - hint_values["center_x"]
            dy = center_y - hint_values["center_y"]
            predicted_distance = math.sqrt(dx * dx + dy * dy)
            tracked_candidate = (
                predicted_distance <= hint_values["search_radius"]
            )
            expected_diameter = hint_values["diameter"]
            if tracked_candidate:
                blur_extra = hint_values["blur_extra"]

        along_extent, lateral_extent = self._projected_extents(
            width,
            height,
        )
        min_lateral = self.min_diameter_px
        min_along = self.min_diameter_px
        max_along = self.max_diameter_px
        min_aspect = self.min_aspect_ratio
        min_fill = self.min_fill_ratio
        min_pixels = self.min_pixels
        if tracked_candidate:
            min_lateral *= 0.60
            min_along *= 0.50
            max_along += blur_extra
            min_aspect = self.tracked_min_aspect_ratio
            min_fill = self.tracked_min_fill_ratio
            min_pixels = self.fragment_min_pixels
        if (
            lateral_extent < min_lateral
            or lateral_extent > self.max_diameter_px
            or along_extent < min_along
            or along_extent > max_along
        ):
            return None

        aspect = min(along_extent, lateral_extent) / max(
            along_extent,
            lateral_extent,
        )
        if aspect < min_aspect:
            return None
        pixels = int(record["pixels"])
        if pixels < min_pixels:
            return None
        fill_ratio = pixels / float(max(1, width * height))
        if fill_ratio < min_fill:
            return None

        diameter = lateral_extent
        aspect_score = _clamp(
            (aspect - min_aspect) / max(0.001, 1.0 - min_aspect),
            0.0,
            1.0,
        )
        fill_score = _clamp(
            (fill_ratio - min_fill) / max(0.001, 0.75 - min_fill),
            0.0,
            1.0,
        )
        size_span = max(
            expected_diameter - self.min_diameter_px,
            self.max_diameter_px - expected_diameter,
            1.0,
        )
        size_score = 1.0 - _clamp(
            abs(diameter - expected_diameter) / size_span,
            0.0,
            1.0,
        )
        axis_score = 1.0 - _clamp(
            abs(lateral) / self.roi_half_width_px,
            0.0,
            1.0,
        )
        prediction_score = 0.0
        if tracked_candidate:
            prediction_score = 1.0 - _clamp(
                predicted_distance / hint_values["search_radius"],
                0.0,
                1.0,
            )
            confidence = _clamp(
                0.38
                + 0.24 * prediction_score
                + 0.14 * size_score
                + 0.10 * axis_score
                + 0.08 * fill_score
                + 0.06 * aspect_score,
                0.0,
                1.0,
            )
        else:
            confidence = _clamp(
                0.35
                + 0.22 * aspect_score
                + 0.18 * fill_score
                + 0.14 * size_score
                + 0.11 * axis_score,
                0.0,
                1.0,
            )
        motion_blurred = bool(
            tracked_candidate
            and along_extent > lateral_extent * 1.25
        )
        if motion_blurred:
            self.motion_candidate_count += 1
        roi_x, roi_y, roi_width, roi_height = self.roi
        roi_clipped = bool(
            x <= roi_x
            or y <= roi_y
            or x + width >= roi_x + roi_width
            or y + height >= roi_y + roi_height
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
                roi_clipped
                or x <= 0
                or y <= 0
                or x + width >= self.frame_width
                or y + height >= self.frame_height
            ),
            "source_index": int(record["source_index"]),
            "fragments": int(record["fragments"]),
            "tracked_candidate": tracked_candidate,
            "motion_blurred": motion_blurred,
        }
    def run(self, frame, tracking_hint=None):
        self.frame_count = (self.frame_count + 1) & 0xFFFFFFFF
        self.preprocess_ms = 0
        self.kpu_ms = 0
        self.postprocess_ms = 0
        self.last_detection_count = 0
        self.blob_count = 0
        self.merged_fragment_count = 0
        self.motion_candidate_count = 0
        hint_values = self._hint_parameters(tracking_hint)
        self.hint_active = hint_values is not None

        preprocess_start = _ticks_ms()
        self._update_threshold(frame)
        mask = self._make_mask(frame)
        preprocess_finished = _ticks_ms()
        self.preprocess_ms = max(
            0,
            _ticks_diff(preprocess_finished, preprocess_start),
        )
        blobs = self._find_blobs(mask)
        blob_finished = _ticks_ms()
        self.kpu_ms = max(
            0,
            _ticks_diff(blob_finished, preprocess_finished),
        )
        try:
            self.blob_count = len(blobs)
        except Exception:
            self.blob_count = 0
        if self.blob_count > self.max_native_blobs:
            self.last_warning = "too many blobs: {}".format(self.blob_count)
            self.postprocess_ms = max(
                0,
                _ticks_diff(_ticks_ms(), blob_finished),
            )
            return []

        records = []
        for source_index in range(self.blob_count):
            record = self._record_from_blob(
                blobs[source_index],
                source_index,
            )
            if record is not None:
                records.append(record)
        records = self._merge_fragments(records, hint_values)

        candidates = []
        for record in records:
            candidate = self._candidate_from_record(record, hint_values)
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
            _ticks_diff(_ticks_ms(), blob_finished),
        )
        return candidates

    def diagnostic_text(self):
        text = "ROI L={}/{} b={} h={} m={}/{}".format(
            int(self.threshold_l),
            int(self.roi_l_median),
            int(self.blob_count),
            1 if self.hint_active else 0,
            int(self.merged_fragment_count),
            int(self.motion_candidate_count),
        )
        if self.last_warning is not None:
            text += " !" + str(self.last_warning)[:20]
        return text
    def deinit(self):
        return None
