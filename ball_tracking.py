"""Bounded candidate association and constrained steel-ball motion filter."""

import math
import time


def _ticks_diff(new_value, old_value):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(new_value, old_value)
    return int(new_value) - int(old_value)


def _elapsed_ms(new_value, old_value):
    value = _ticks_diff(int(new_value), int(old_value))
    return max(0, int(value))


def _clamp(value, low, high):
    return max(low, min(high, value))


def build_ball_candidate_from_xywh(
    center_x,
    center_y,
    raw_width,
    raw_height,
    confidence,
    source_index,
    confidence_threshold=0.35,
    frame_width=320,
    frame_height=320,
):
    """Validate and clip one already-selected YOLOv8 center-xywh box."""
    center_x = float(center_x)
    center_y = float(center_y)
    raw_width = float(raw_width)
    raw_height = float(raw_height)
    confidence = float(confidence)
    frame_width = max(1, int(frame_width))
    frame_height = max(1, int(frame_height))
    if (
        center_x != center_x
        or center_y != center_y
        or raw_width != raw_width
        or raw_height != raw_height
        or confidence != confidence
        or confidence < float(confidence_threshold)
        or raw_width <= 0.0
        or raw_height <= 0.0
        or raw_width > frame_width * 4.0
        or raw_height > frame_height * 4.0
    ):
        return None

    raw_x = center_x - raw_width * 0.5
    raw_y = center_y - raw_height * 0.5
    raw_x1 = center_x + raw_width * 0.5
    raw_y1 = center_y + raw_height * 0.5
    if (
        raw_x1 <= 0.0
        or raw_y1 <= 0.0
        or raw_x >= frame_width
        or raw_y >= frame_height
    ):
        return None

    x0 = int(_clamp(round(raw_x), 0, max(0, frame_width - 1)))
    y0 = int(_clamp(round(raw_y), 0, max(0, frame_height - 1)))
    x1 = int(_clamp(round(raw_x1), x0 + 1, frame_width))
    y1 = int(_clamp(round(raw_y1), y0 + 1, frame_height))
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    clipped_x0 = max(0.0, raw_x)
    clipped_y0 = max(0.0, raw_y)
    clipped_x1 = min(float(frame_width), raw_x1)
    clipped_y1 = min(float(frame_height), raw_y1)
    return {
        "confidence": _clamp(confidence, 0.0, 1.0),
        # Keep the model's subpixel center for filtering; integer geometry is
        # retained separately for drawing and the wire format.
        "center_x": (clipped_x0 + clipped_x1) * 0.5,
        "center_y": (clipped_y0 + clipped_y1) * 0.5,
        "x": x0,
        "y": y0,
        "w": width,
        "h": height,
        "diameter": (width + height) * 0.5,
        "edge_clipped": (
            raw_x < 0.0
            or raw_y < 0.0
            or raw_x1 > frame_width
            or raw_y1 > frame_height
            or x0 == 0
            or y0 == 0
            or x1 == frame_width
            or y1 == frame_height
        ),
        "source_index": int(source_index),
    }


def build_ball_candidate_from_tlwh(
    x,
    y,
    width,
    height,
    confidence,
    source_index,
    confidence_threshold=0.35,
    frame_width=320,
    frame_height=320,
):
    """Convert one native postprocessed top-left-XYWH box to a candidate."""
    width = float(width)
    height = float(height)
    return build_ball_candidate_from_xywh(
        float(x) + width * 0.5,
        float(y) + height * 0.5,
        width,
        height,
        confidence,
        source_index,
        confidence_threshold,
        frame_width,
        frame_height,
    )


def normalize_postprocessed_ball_candidates(
    raw_result,
    confidence_threshold=0.35,
    frame_width=320,
    frame_height=320,
    max_candidates=4,
):
    """Normalize a bounded native YOLO result list for target association."""
    if raw_result is None or len(raw_result) < 3:
        return []
    boxes = raw_result[0]
    class_ids = raw_result[1]
    scores = raw_result[2]
    count = min(
        len(boxes),
        len(class_ids),
        len(scores),
        max(0, int(max_candidates)),
    )
    candidates = []
    for source_index in range(count):
        if int(class_ids[source_index]) != 0:
            continue
        box = boxes[source_index]
        candidate = build_ball_candidate_from_tlwh(
            box[0],
            box[1],
            box[2],
            box[3],
            scores[source_index],
            source_index,
            confidence_threshold,
            frame_width,
            frame_height,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def select_highest_confidence_ball(
    raw_result,
    frame_width=320,
    frame_height=320,
):
    """Compatibility helper for callers of the former top-1 API."""
    try:
        max_candidates = 0 if raw_result is None else len(raw_result[0])
        candidates = normalize_postprocessed_ball_candidates(
            raw_result,
            0.0,
            frame_width,
            frame_height,
            max_candidates,
        )
    except Exception:
        return None
    best = None
    for candidate in candidates:
        if (
            best is None
            or candidate["confidence"] > best["confidence"]
        ):
            best = candidate
    return best


class PipeMotionFilter:
    """Asymmetric low-lag observer for one-dimensional pipe motion.

    A configured pipe axis is used immediately when available. Axial position
    has a small noise band and a fast measurement response outside it. Axial
    velocity follows consecutive accepted measurements, including reversals,
    while lateral position is strongly smoothed and lateral velocity is zero.
    Apparent diameter changes slowly with a per-frame bound. The former learned
    axis remains as a compatibility fallback when no fixed axis is supplied.
    """

    def __init__(
        self,
        alpha=0.96,
        beta=0.55,
        lateral_alpha=0.08,
        lateral_beta=0.0,
        base_gate_px=32.0,
        max_gate_px=96.0,
        lateral_gate_px=14.0,
        axis_lock_distance_px=12.0,
        axis_adapt=0.08,
        max_coast_ms=300,
        max_coast_frames=8,
        coast_velocity_decay=0.98,
        frame_width=320,
        frame_height=320,
        max_speed_px_s=2500.0,
        response_alpha=0.18,
        response_beta=0.30,
        response_start_px=0.25,
        response_full_px=1.50,
        response_speed_px_s=60.0,
        acquire_confirm_frames=3,
        acquire_timeout_ms=180,
        acquire_radius_px=36.0,
        acquire_min_confidence=0.50,
        fixed_axis_x=None,
        fixed_axis_y=None,
        velocity_fast_beta=0.85,
        velocity_quiet_decay=0.35,
        velocity_noise_px=0.90,
        velocity_quiet_timeout_ms=120,
        diameter_alpha=0.08,
        diameter_max_step_px=1.0,
        motion_start_px=2.5,
        motion_immediate_px=14.0,
        motion_confirm_frames=2,
        motion_stop_frames=4,
        motion_direction_epsilon_px=0.75,
        ambiguous_position_alpha=0.10,
        quiet_position_alpha=0.12,
        quiet_velocity_decay=0.20,
        quiet_gate_diameter_factor=1.25,
        quiet_gate_min_px=10.0,
        motion_min_quality=0.62,
        motion_min_support=2,
        association_ambiguity_diameter_factor=0.75,
    ):
        self.alpha = _clamp(float(alpha), 0.0, 1.0)
        self.beta = _clamp(float(beta), 0.0, 1.0)
        self.response_alpha = min(
            _clamp(float(response_alpha), 0.0, 1.0),
            self.alpha,
        )
        self.response_beta = min(
            _clamp(float(response_beta), 0.0, 1.0),
            self.beta,
        )
        self.response_start_px = max(0.0, float(response_start_px))
        self.response_full_px = max(
            self.response_start_px + 0.001,
            float(response_full_px),
        )
        self.response_speed_px_s = max(1.0, float(response_speed_px_s))
        self.response_start_squared = self.response_start_px * self.response_start_px
        self.response_full_squared = self.response_full_px * self.response_full_px
        self.velocity_fast_beta = _clamp(
            float(velocity_fast_beta),
            self.beta,
            1.0,
        )
        self.velocity_quiet_decay = _clamp(
            float(velocity_quiet_decay),
            0.0,
            1.0,
        )
        self.velocity_noise_px = max(0.0, float(velocity_noise_px))
        self.velocity_quiet_timeout_ms = max(
            1,
            int(velocity_quiet_timeout_ms),
        )
        self.diameter_alpha = _clamp(float(diameter_alpha), 0.0, 1.0)
        self.diameter_max_step_px = max(
            0.0,
            float(diameter_max_step_px),
        )
        self.motion_start_px = max(0.1, float(motion_start_px))
        self.motion_immediate_px = max(
            self.motion_start_px,
            float(motion_immediate_px),
        )
        self.motion_confirm_frames = max(1, int(motion_confirm_frames))
        self.motion_stop_frames = max(1, int(motion_stop_frames))
        self.motion_direction_epsilon_px = max(
            0.0,
            float(motion_direction_epsilon_px),
        )
        self.ambiguous_position_alpha = _clamp(
            float(ambiguous_position_alpha),
            0.0,
            self.alpha,
        )
        self.quiet_position_alpha = _clamp(
            float(quiet_position_alpha),
            0.0,
            self.alpha,
        )
        self.quiet_velocity_decay = _clamp(
            float(quiet_velocity_decay),
            0.0,
            1.0,
        )
        self.quiet_gate_diameter_factor = max(
            0.50,
            float(quiet_gate_diameter_factor),
        )
        self.quiet_gate_min_px = max(2.0, float(quiet_gate_min_px))
        self.motion_min_quality = _clamp(
            float(motion_min_quality),
            0.0,
            1.0,
        )
        self.motion_min_support = max(1, int(motion_min_support))
        self.association_ambiguity_diameter_factor = max(
            0.10,
            float(association_ambiguity_diameter_factor),
        )
        axis_x = 0.0 if fixed_axis_x is None else float(fixed_axis_x)
        axis_y = 0.0 if fixed_axis_y is None else float(fixed_axis_y)
        axis_magnitude = math.sqrt(axis_x * axis_x + axis_y * axis_y)
        self.fixed_axis = axis_magnitude > 0.0001
        if self.fixed_axis:
            self.config_axis_x = axis_x / axis_magnitude
            self.config_axis_y = axis_y / axis_magnitude
        else:
            self.config_axis_x = 1.0
            self.config_axis_y = 0.0
        self.acquire_confirm_frames = max(1, int(acquire_confirm_frames))
        self.acquire_timeout_ms = max(1, int(acquire_timeout_ms))
        self.acquire_radius_px = max(1.0, float(acquire_radius_px))
        self.acquire_min_confidence = _clamp(
            float(acquire_min_confidence),
            0.0,
            1.0,
        )
        self.lateral_alpha = float(lateral_alpha)
        self.lateral_beta = float(lateral_beta)
        self.base_gate_px = float(base_gate_px)
        self.max_gate_px = float(max_gate_px)
        self.lateral_gate_px = float(lateral_gate_px)
        self.axis_lock_distance_px = float(axis_lock_distance_px)
        self.axis_adapt = float(axis_adapt)
        self.max_coast_ms = max(1, int(max_coast_ms))
        self.max_coast_frames = max(1, int(max_coast_frames))
        self.coast_velocity_decay = _clamp(
            float(coast_velocity_decay),
            0.0,
            1.0,
        )
        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))
        self.max_speed_px_s = max(1.0, float(max_speed_px_s))
        self.clear()

    def _restore_axis(self):
        self.axis_locked = bool(self.fixed_axis)
        self.axis_x = self.config_axis_x
        self.axis_y = self.config_axis_y

    def clear(self):
        self.initialized = False
        self.track_valid = False
        self._restore_axis()
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.axis_seed_x = 0.0
        self.axis_seed_y = 0.0
        self.last_measure_x = 0.0
        self.last_measure_y = 0.0
        self.velocity_anchor_x = 0.0
        self.velocity_anchor_y = 0.0
        self.velocity_anchor_ms = None
        self.tracked_diameter = 0.0
        self.last_update_ms = None
        self.last_measurement_ms = None
        self.missed_frames = 0
        self.pending_x = 0.0
        self.pending_y = 0.0
        self.pending_last_ms = None
        self.pending_count = 0
        self.reacquire_requested = False
        self.last_detection_count = 0
        self.last_association_rejected = False
        self.association_reject_count = 0
        self.last_axial_innovation = 0.0
        self.last_measured_velocity = 0.0
        self.fast_response_active = False
        self.velocity_braking = False
        self.motion_state = "QUIET"
        self.motion_confirm_count = 0
        self.motion_pending_sign = 0
        self.motion_direction = 0
        self.motion_quiet_count = 0
        self.last_motion_step = 0.0
        self.measurement_ambiguous = False
        self.measurement_quality = 1.0
        self.measurement_support = 0
        self.association_wide = False
        self.association_competing = False

    def _reset_from_measurement(self, candidate, now_ms):
        self.initialized = True
        self.track_valid = True
        self._restore_axis()
        self.x = float(candidate["center_x"])
        self.y = float(candidate["center_y"])
        self.vx = 0.0
        self.vy = 0.0
        self.axis_seed_x = self.x
        self.axis_seed_y = self.y
        self.last_measure_x = self.x
        self.last_measure_y = self.y
        self.velocity_anchor_x = self.x
        self.velocity_anchor_y = self.y
        self.velocity_anchor_ms = int(now_ms)
        self.tracked_diameter = max(
            0.0,
            float(candidate.get("diameter", 0.0)),
        )
        self.last_update_ms = int(now_ms)
        self.last_measurement_ms = int(now_ms)
        self.missed_frames = 0
        self.last_axial_innovation = 0.0
        self.last_measured_velocity = 0.0
        self.fast_response_active = False
        self.velocity_braking = False
        self.motion_state = "QUIET"
        self.motion_confirm_count = 0
        self.motion_pending_sign = 0
        self.motion_direction = 0
        self.motion_quiet_count = 0
        self.last_motion_step = 0.0
        self.measurement_ambiguous = False
        self.measurement_quality = float(
            candidate.get("measurement_quality", 1.0)
        )
        self.measurement_support = int(candidate.get("support_count", 0))
        self.association_wide = False
        self.association_competing = False
        self._clear_pending_acquisition()
        self.reacquire_requested = False

    def _clear_pending_acquisition(self):
        self.pending_x = 0.0
        self.pending_y = 0.0
        self.pending_last_ms = None
        self.pending_count = 0

    def request_reacquire(self):
        """Allow the next valid detector result to replace the current target."""
        self.reacquire_requested = True
        self._clear_pending_acquisition()

    def _is_expired(self, now_ms):
        return (
            self.last_measurement_ms is None
            or self._measurement_age(now_ms) > self.max_coast_ms
            or self.missed_frames > self.max_coast_frames
        )

    @staticmethod
    def _highest_confidence(candidates):
        best = None
        for candidate in candidates:
            if candidate is None:
                continue
            if (
                best is None
                or float(candidate.get("confidence", 0.0))
                > float(best.get("confidence", 0.0))
            ):
                best = candidate
        return best

    @staticmethod
    def _acquisition_score(candidate):
        return float(candidate.get(
            "acquisition_score",
            candidate.get("confidence", 0.0),
        ))

    def _highest_acquisition_candidate(self, candidates):
        best = None
        for candidate in candidates:
            if (
                candidate is None
                or candidate.get("edge_clipped")
                or float(candidate.get("confidence", 0.0))
                < self.acquire_min_confidence
            ):
                continue
            if (
                best is None
                or self._acquisition_score(candidate)
                > self._acquisition_score(best)
            ):
                best = candidate
        return best

    def _predicted_position(self, now_ms):
        if self.last_update_ms is None:
            return self.x, self.y, 0.010
        elapsed_ms = _elapsed_ms(now_ms, self.last_update_ms)
        dt = _clamp(elapsed_ms / 1000.0, 0.010, 0.200)
        velocity_x = self.vx
        velocity_y = self.vy
        if self.axis_locked:
            along_velocity = (
                velocity_x * self.axis_x + velocity_y * self.axis_y
            )
            velocity_x = along_velocity * self.axis_x
            velocity_y = along_velocity * self.axis_y
        return self.x + velocity_x * dt, self.y + velocity_y * dt, dt

    def _measurement_age(self, now_ms):
        if self.last_measurement_ms is None:
            return 0xFFFF
        return min(
            0xFFFF,
            _elapsed_ms(now_ms, self.last_measurement_ms),
        )

    def detection_hint(self, now_ms):
        """Return a read-only prediction for a tracking-aware detector.

        The hint never changes filter state and expires with the normal HOLD
        interval. Detectors may narrow their search with it, while the normal
        association gate remains authoritative.
        """
        now_ms = int(now_ms)
        if not self.initialized or self._is_expired(now_ms):
            return None
        predicted_x, predicted_y, dt = self._predicted_position(now_ms)
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        search_radius = _clamp(
            self.base_gate_px + speed * dt * 1.5,
            self.base_gate_px,
            self.max_gate_px,
        )
        return {
            "center_x": _clamp(
                predicted_x,
                0.0,
                float(self.frame_width - 1),
            ),
            "center_y": _clamp(
                predicted_y,
                0.0,
                float(self.frame_height - 1),
            ),
            "speed_px_s": speed,
            "diameter": self.tracked_diameter,
            "search_radius_px": search_radius,
            "measurement_age_ms": self._measurement_age(now_ms),
        }

    def _bound_state(self):
        self.x = _clamp(self.x, 0.0, float(self.frame_width - 1))
        self.y = _clamp(self.y, 0.0, float(self.frame_height - 1))
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > self.max_speed_px_s:
            scale = self.max_speed_px_s / speed
            self.vx *= scale
            self.vy *= scale

    def _predict(self, now_ms):
        elapsed_ms = _elapsed_ms(now_ms, self.last_update_ms)
        dt = _clamp(elapsed_ms / 1000.0, 0.010, 0.200)
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.axis_locked:
            along_velocity = (
                self.vx * self.axis_x + self.vy * self.axis_y
            )
            self.vx = along_velocity * self.axis_x
            self.vy = along_velocity * self.axis_y
        self._bound_state()
        self.last_update_ms = int(now_ms)
        return dt

    def _learn_axis(self, candidate):
        if self.fixed_axis:
            return
        measurement_x = float(candidate["center_x"])
        measurement_y = float(candidate["center_y"])
        if not self.axis_locked:
            dx = measurement_x - self.axis_seed_x
            dy = measurement_y - self.axis_seed_y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance >= self.axis_lock_distance_px:
                self.axis_x = dx / distance
                self.axis_y = dy / distance
                self.axis_locked = True
            return

        dx = measurement_x - self.last_measure_x
        dy = measurement_y - self.last_measure_y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 3.0:
            return
        unit_x = dx / distance
        unit_y = dy / distance
        if unit_x * self.axis_x + unit_y * self.axis_y < 0.0:
            unit_x = -unit_x
            unit_y = -unit_y
        mixed_x = (
            (1.0 - self.axis_adapt) * self.axis_x
            + self.axis_adapt * unit_x
        )
        mixed_y = (
            (1.0 - self.axis_adapt) * self.axis_y
            + self.axis_adapt * unit_y
        )
        magnitude = math.sqrt(mixed_x * mixed_x + mixed_y * mixed_y)
        if magnitude > 0.0001:
            self.axis_x = mixed_x / magnitude
            self.axis_y = mixed_y / magnitude

    def _measurement_gate(self, candidate, dt, speed=None):
        if speed is None:
            speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        total_gate = _clamp(
            self.base_gate_px + speed * dt * 1.5,
            self.base_gate_px,
            self.max_gate_px,
        )
        if candidate.get("edge_clipped"):
            edge_gate = max(
                16.0,
                float(candidate.get("diameter", 0.0)) * 1.25
                + speed * dt * 0.90,
            )
            total_gate = min(total_gate, edge_gate)
        if (
            self.axis_locked
            and self.motion_state == "PENDING"
            and self.motion_pending_sign != 0
        ):
            measurement_dx = (
                float(candidate.get("center_x", 0.0))
                - self.last_measure_x
            )
            measurement_dy = (
                float(candidate.get("center_y", 0.0))
                - self.last_measure_y
            )
            measurement_step = (
                measurement_dx * self.axis_x
                + measurement_dy * self.axis_y
            )
            step_direction = 0
            if abs(measurement_step) >= self.motion_immediate_px:
                step_direction = 1 if measurement_step > 0.0 else -1
            if step_direction == self.motion_pending_sign:
                total_gate = max(
                    total_gate,
                    min(
                        self.max_gate_px,
                        self.base_gate_px
                        + max(
                            self.motion_immediate_px,
                            abs(measurement_step),
                        ),
                    ),
                )
        return total_gate

    def _quiet_association_gate(self, candidate):
        diameter = max(
            1.0,
            self.tracked_diameter,
            float(candidate.get("diameter", 0.0)),
        )
        return min(
            self.base_gate_px,
            max(
                self.quiet_gate_min_px,
                diameter * self.quiet_gate_diameter_factor,
            ),
        )

    def _candidate_tracking_ambiguous(self, candidate):
        if (
            candidate.get("measurement_ambiguous")
            or candidate.get("association_competing")
        ):
            return True
        if (
            float(candidate.get("measurement_quality", 1.0))
            < self.motion_min_quality
        ):
            return True
        if (
            "support_count" in candidate
            and int(candidate.get("support_count", 0))
            < self.motion_min_support
        ):
            return True
        return False

    def _pending_direction_priority(self, candidate):
        if (
            not self.axis_locked
            or self.motion_state != "PENDING"
            or self.motion_pending_sign == 0
        ):
            return 0
        measurement_dx = (
            float(candidate.get("center_x", 0.0)) - self.last_measure_x
        )
        measurement_dy = (
            float(candidate.get("center_y", 0.0)) - self.last_measure_y
        )
        measurement_step = (
            measurement_dx * self.axis_x
            + measurement_dy * self.axis_y
        )
        if abs(measurement_step) <= self.motion_direction_epsilon_px:
            return 1
        direction = 1 if measurement_step > 0.0 else -1
        return 0 if direction == self.motion_pending_sign else 1

    def select_candidate(self, candidates, now_ms):
        """Select one detection while preserving the currently tracked target.

        The detector may return a small bounded list. During a valid track,
        distance from the predicted position is the primary association key;
        confidence only breaks a near tie. When no track exists, the highest
        confidence candidate is passed to the strict acquisition confirmer.
        """
        now_ms = int(now_ms)
        if candidates is None:
            candidates = []
        self.last_detection_count = len(candidates)
        self.last_association_rejected = False
        if len(candidates) == 0:
            if not self.reacquire_requested:
                self._clear_pending_acquisition()
            return None

        if self.reacquire_requested:
            return self._highest_confidence(candidates)
        if not self.initialized or self._is_expired(now_ms):
            best = self._highest_acquisition_candidate(candidates)
            if best is None:
                self.last_association_rejected = True
                self.association_reject_count = min(
                    0xFFFFFFFF,
                    self.association_reject_count + 1,
                )
            return best

        predicted_x, predicted_y, dt = self._predicted_position(now_ms)
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        best = None
        best_distance = 0.0
        best_priority = 0
        best_wide = False
        second = None
        second_distance = 0.0
        second_priority = 0
        challenger = None
        challenger_distance = 0.0
        for candidate in candidates:
            if candidate is None:
                continue
            candidate["association_wide"] = False
            candidate["association_competing"] = False
            candidate["tracking_ambiguous"] = False
            dx = float(candidate.get("center_x", 0.0)) - predicted_x
            dy = float(candidate.get("center_y", 0.0)) - predicted_y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= self._measurement_gate(candidate, dt, speed):
                candidate_ambiguous = (
                    self._candidate_tracking_ambiguous(candidate)
                )
                quiet_wide = bool(
                    self.motion_state != "MOVING"
                    and distance > self._quiet_association_gate(candidate)
                )
                priority = 2 if candidate_ambiguous else 0
                if quiet_wide:
                    priority += 1
                    if self.motion_state == "PENDING":
                        priority += self._pending_direction_priority(
                            candidate
                        )
                better = bool(
                    best is None
                    or priority < best_priority
                    or (
                        priority == best_priority
                        and (
                            distance < best_distance
                            or (
                                abs(distance - best_distance) <= 0.5
                                and float(candidate.get("confidence", 0.0))
                                > float(best.get("confidence", 0.0))
                            )
                        )
                    )
                )
                if better:
                    second = best
                    second_distance = best_distance
                    second_priority = best_priority
                    best = candidate
                    best_distance = distance
                    best_priority = priority
                    best_wide = bool(
                        self.motion_state != "MOVING"
                        and distance > self._quiet_association_gate(candidate)
                    )
                elif (
                    second is None
                    or priority < second_priority
                    or (
                        priority == second_priority
                        and distance < second_distance
                    )
                ):
                    second = candidate
                    second_distance = distance
                    second_priority = priority
                continue
            pending_distance = distance
            if self.pending_count > 0:
                pending_dx = (
                    float(candidate.get("center_x", 0.0)) - self.pending_x
                )
                pending_dy = (
                    float(candidate.get("center_y", 0.0)) - self.pending_y
                )
                pending_distance = math.sqrt(
                    pending_dx * pending_dx + pending_dy * pending_dy
                )
            if (
                not candidate.get("edge_clipped")
                and (
                    distance <= self.max_gate_px
                    or (
                        self.pending_count > 0
                        and pending_distance <= self.acquire_radius_px
                    )
                )
                and (
                    challenger is None
                    or pending_distance < challenger_distance
                )
            ):
                challenger = candidate
                challenger_distance = pending_distance

        if best is not None:
            diameter = max(
                1.0,
                self.tracked_diameter,
                float(best.get("diameter", 0.0)),
            )
            competing = bool(
                second is not None
                and second_priority == best_priority
                and second_distance
                <= best_distance
                + diameter * self.association_ambiguity_diameter_factor
            )
            best["association_wide"] = best_wide
            best["association_competing"] = competing
            best["tracking_ambiguous"] = (
                self._candidate_tracking_ambiguous(best)
            )
            self._clear_pending_acquisition()
            return best
        if challenger is not None and self._confirm_new_target(
            challenger,
            now_ms,
        ):
            self.reacquire_requested = True
            return challenger
        if challenger is None:
            self._clear_pending_acquisition()
        self.last_association_rejected = True
        self.association_reject_count = min(
            0xFFFFFFFF,
            self.association_reject_count + 1,
        )
        return None

    def _confirm_new_target(self, candidate, now_ms):
        """Require consecutive, non-edge, sufficiently confident detections."""
        if candidate is None:
            self._clear_pending_acquisition()
            return False
        if (
            float(candidate.get("confidence", 0.0))
            < self.acquire_min_confidence
            or candidate.get("edge_clipped")
        ):
            self._clear_pending_acquisition()
            return False

        center_x = float(candidate["center_x"])
        center_y = float(candidate["center_y"])
        if (
            self.pending_count == 0
            or self.pending_last_ms is None
            or _elapsed_ms(now_ms, self.pending_last_ms)
            > self.acquire_timeout_ms
        ):
            self.pending_count = 1
        else:
            dx = center_x - self.pending_x
            dy = center_y - self.pending_y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > self.acquire_radius_px:
                self.pending_count = 1
            else:
                self.pending_count += 1
        self.pending_x = center_x
        self.pending_y = center_y
        self.pending_last_ms = int(now_ms)
        return self.pending_count >= self.acquire_confirm_frames

    def _accept_measurement(self, candidate, dt):
        residual_x = float(candidate["center_x"]) - self.x
        residual_y = float(candidate["center_y"]) - self.y
        residual_distance = math.sqrt(
            residual_x * residual_x + residual_y * residual_y
        )
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        total_gate = self._measurement_gate(candidate, dt, speed)
        if residual_distance > total_gate:
            return False
        if self.axis_locked:
            normal_x = -self.axis_y
            normal_y = self.axis_x
            lateral = residual_x * normal_x + residual_y * normal_y
            diameter_gate = _clamp(
                float(candidate.get("diameter", 0.0)) * 0.75,
                self.lateral_gate_px,
                32.0,
            )
            if abs(lateral) > diameter_gate:
                return False
        return True

    def _motion_response(self, along, candidate):
        self.measurement_ambiguous = bool(
            candidate.get("measurement_ambiguous")
            or candidate.get("tracking_ambiguous")
            or self._candidate_tracking_ambiguous(candidate)
        )
        self.measurement_quality = _clamp(
            float(candidate.get("measurement_quality", 1.0)),
            0.0,
            1.0,
        )
        self.measurement_support = int(candidate.get("support_count", 0))
        self.association_wide = bool(candidate.get("association_wide"))
        self.association_competing = bool(
            candidate.get("association_competing")
        )
        measurement_dx = (
            float(candidate["center_x"]) - self.last_measure_x
        )
        measurement_dy = (
            float(candidate["center_y"]) - self.last_measure_y
        )
        measurement_step = (
            measurement_dx * self.axis_x + measurement_dy * self.axis_y
        )
        self.last_motion_step = measurement_step
        magnitude = abs(along)
        step_magnitude = abs(measurement_step)
        required_frames = self.motion_confirm_frames
        start_threshold = self.motion_start_px
        if self.measurement_ambiguous:
            required_frames += 1
            start_threshold *= 1.5
        direction = 0
        if step_magnitude > self.motion_direction_epsilon_px:
            direction = 1 if measurement_step > 0.0 else -1
        immediate_confirmed = False
        if step_magnitude >= self.motion_immediate_px and direction != 0:
            if self.motion_state == "MOVING":
                immediate_confirmed = bool(
                    not self.measurement_ambiguous
                    or (
                        self.motion_pending_sign == direction
                        and self.motion_confirm_count > 0
                    )
                )
            else:
                immediate_confirmed = bool(
                    not self.measurement_ambiguous
                    and self.motion_pending_sign == direction
                    and self.motion_confirm_count > 0
                )

        if self.motion_state == "MOVING":
            if direction == self.motion_direction and direction != 0:
                self.motion_quiet_count = 0
                self.motion_confirm_count = required_frames
                self.motion_pending_sign = direction
                return True
            if direction == 0:
                self.motion_quiet_count = min(
                    self.motion_stop_frames,
                    self.motion_quiet_count + 1,
                )
                if self.motion_quiet_count < self.motion_stop_frames:
                    return True
                self.motion_state = "QUIET"
                self.motion_confirm_count = 0
                self.motion_pending_sign = 0
                self.motion_direction = 0
                return False

            self.motion_quiet_count = 0
            if self.motion_pending_sign == direction:
                self.motion_confirm_count = min(
                    required_frames,
                    self.motion_confirm_count + 1,
                )
            else:
                self.motion_pending_sign = direction
                self.motion_confirm_count = 1
            if (
                immediate_confirmed
                or self.motion_confirm_count >= required_frames
            ):
                self.motion_state = "MOVING"
                self.motion_direction = direction
                self.motion_confirm_count = required_frames
                return True
            self.motion_state = "PENDING"
            return False

        if direction == 0:
            self.motion_state = "QUIET"
            self.motion_confirm_count = 0
            self.motion_pending_sign = 0
            self.motion_direction = 0
            self.motion_quiet_count = 0
            return False

        if (
            self.motion_pending_sign == direction
        ):
            self.motion_confirm_count = min(
                required_frames,
                self.motion_confirm_count + 1,
            )
        else:
            self.motion_pending_sign = direction
            self.motion_confirm_count = 1
        if immediate_confirmed:
            self.motion_state = "MOVING"
            self.motion_direction = direction
            self.motion_quiet_count = 0
            return True
        if self.motion_confirm_count >= required_frames:
            if magnitude >= start_threshold:
                self.motion_state = "MOVING"
                self.motion_direction = direction
                self.motion_quiet_count = 0
                return True
        self.motion_state = "PENDING"
        return False

    def _correct(self, candidate, dt, now_ms):
        measurement_x = float(candidate["center_x"])
        measurement_y = float(candidate["center_y"])
        residual_x = measurement_x - self.x
        residual_y = measurement_y - self.y
        residual_squared = residual_x * residual_x + residual_y * residual_y
        normal_x = -self.axis_y
        normal_y = self.axis_x
        along = residual_x * self.axis_x + residual_y * self.axis_y
        lateral = residual_x * normal_x + residual_y * normal_y
        response_squared = (
            along * along if self.axis_locked else residual_squared
        )
        response = _clamp(
            (response_squared - self.response_start_squared)
            / (self.response_full_squared - self.response_start_squared),
            0.0,
            1.0,
        )
        motion_confirmed = self._motion_response(along, candidate)
        alpha = self.response_alpha + (
            self.alpha - self.response_alpha
        ) * response
        if not motion_confirmed:
            alpha = min(
                alpha,
                self.ambiguous_position_alpha
                if self.measurement_ambiguous
                else self.quiet_position_alpha,
            )
        if not self.axis_locked:
            beta = self.response_beta + (
                self.beta - self.response_beta
            ) * response
            self.x += alpha * residual_x
            self.y += alpha * residual_y
            if motion_confirmed:
                self.vx += beta * residual_x / dt
                self.vy += beta * residual_y / dt
            else:
                self.vx *= self.quiet_velocity_decay
                self.vy *= self.quiet_velocity_decay
                if abs(self.vx) < 1.0:
                    self.vx = 0.0
                if abs(self.vy) < 1.0:
                    self.vy = 0.0
            self.last_axial_innovation = math.sqrt(residual_squared)
            self.last_measured_velocity = (
                math.sqrt(self.vx * self.vx + self.vy * self.vy)
                if motion_confirmed
                else 0.0
            )
            self.fast_response_active = bool(
                motion_confirmed and response >= 0.5
            )
            self._bound_state()
            return

        self.last_axial_innovation = along
        self.fast_response_active = bool(
            motion_confirmed and response >= 0.5
        )
        self.x += (
            alpha * along * self.axis_x
            + self.lateral_alpha * lateral * normal_x
        )
        self.y += (
            alpha * along * self.axis_y
            + self.lateral_alpha * lateral * normal_y
        )

        measured_dx = measurement_x - self.velocity_anchor_x
        measured_dy = measurement_y - self.velocity_anchor_y
        measured_along = (
            measured_dx * self.axis_x + measured_dy * self.axis_y
        )
        along_velocity = self.vx * self.axis_x + self.vy * self.axis_y
        velocity_anchor_age_ms = 0
        if self.velocity_anchor_ms is not None:
            velocity_anchor_age_ms = _elapsed_ms(
                now_ms,
                self.velocity_anchor_ms,
            )
        if not motion_confirmed:
            self.last_measured_velocity = 0.0
            along_velocity *= self.quiet_velocity_decay
            if abs(along_velocity) < 1.0:
                along_velocity = 0.0
            if self.motion_state != "PENDING":
                self.velocity_anchor_x = measurement_x
                self.velocity_anchor_y = measurement_y
                self.velocity_anchor_ms = int(now_ms)
            self.velocity_braking = False
        elif abs(measured_along) <= self.velocity_noise_px:
            self.last_measured_velocity = 0.0
            braking_innovation = (
                along_velocity * along < 0.0
                and abs(along) >= self.velocity_noise_px
            )
            if braking_innovation:
                self.velocity_braking = True
            quiet_confirmed = (
                velocity_anchor_age_ms >= self.velocity_quiet_timeout_ms
            )
            if self.velocity_braking or quiet_confirmed:
                along_velocity *= self.velocity_quiet_decay
                if abs(along_velocity) < 1.0:
                    along_velocity = 0.0
                    self.velocity_braking = False
                self.velocity_anchor_x = measurement_x
                self.velocity_anchor_y = measurement_y
                self.velocity_anchor_ms = int(now_ms)
        else:
            self.velocity_braking = False
            velocity_dt = 0.010
            if self.velocity_anchor_ms is not None:
                velocity_dt = _clamp(
                    velocity_anchor_age_ms / 1000.0,
                    0.010,
                    0.200,
                )
            measured_velocity = measured_along / velocity_dt
            self.last_measured_velocity = measured_velocity
            direction_changed = (
                along_velocity * measured_velocity < 0.0
                and abs(measured_velocity) > self.response_speed_px_s
            )
            velocity_beta = self.response_beta + (
                self.velocity_fast_beta - self.response_beta
            ) * response
            if direction_changed:
                velocity_beta = self.velocity_fast_beta
            along_velocity += velocity_beta * (
                measured_velocity - along_velocity
            )
            self.velocity_anchor_x = measurement_x
            self.velocity_anchor_y = measurement_y
            self.velocity_anchor_ms = int(now_ms)
        self.vx = along_velocity * self.axis_x
        self.vy = along_velocity * self.axis_y
        self._bound_state()

    def _update_diameter(self, measured_diameter):
        measured_diameter = max(0.0, float(measured_diameter))
        if measured_diameter <= 0.0:
            return
        if self.tracked_diameter <= 0.0:
            self.tracked_diameter = measured_diameter
            return
        difference = measured_diameter - self.tracked_diameter
        difference = _clamp(
            difference,
            -self.diameter_max_step_px,
            self.diameter_max_step_px,
        )
        self.tracked_diameter += self.diameter_alpha * difference

    def _snapshot(
        self,
        now_ms,
        measurement_accepted=False,
        outlier_rejected=False,
        filter_reset=False,
    ):
        age_ms = self._measurement_age(now_ms)
        self.track_valid = bool(
            self.initialized
            and age_ms <= self.max_coast_ms
            and self.missed_frames <= self.max_coast_frames
        )
        return {
            "measurement_accepted": bool(measurement_accepted),
            "track_valid": self.track_valid,
            "predicted": bool(self.track_valid and not measurement_accepted),
            "axis_locked": self.axis_locked,
            "outlier_rejected": bool(outlier_rejected),
            "association_rejected": bool(self.last_association_rejected),
            "filter_reset": bool(filter_reset),
            "missed_frames": min(255, int(self.missed_frames)),
            "measurement_age_ms": age_ms,
            "detection_count": min(255, int(self.last_detection_count)),
            "association_reject_count": min(
                0xFFFFFFFF,
                int(self.association_reject_count),
            ),
            "acquisition_pending": self.pending_count > 0,
            "acquisition_count": min(255, int(self.pending_count)),
            "acquisition_required": self.acquire_confirm_frames,
            "reacquire_requested": bool(self.reacquire_requested),
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "axis_x": self.axis_x,
            "axis_y": self.axis_y,
            "tracked_diameter": self.tracked_diameter,
            "axial_innovation": self.last_axial_innovation,
            "measured_axial_velocity": self.last_measured_velocity,
            "fast_response": bool(self.fast_response_active),
            "velocity_braking": bool(self.velocity_braking),
            "motion_state": self.motion_state,
            "motion_confirm_count": min(
                255,
                int(self.motion_confirm_count),
            ),
            "motion_direction": int(self.motion_direction),
            "motion_step_px": float(self.last_motion_step),
            "measurement_ambiguous": bool(self.measurement_ambiguous),
            "measurement_quality": float(self.measurement_quality),
            "measurement_support": min(
                255,
                max(0, int(self.measurement_support)),
            ),
            "association_wide": bool(self.association_wide),
            "association_competing": bool(self.association_competing),
        }

    def _advance_to_output(self, output_ms):
        if output_ms is None:
            return self.last_update_ms
        output_ms = int(output_ms)
        if (
            self.last_update_ms is not None
            and _elapsed_ms(output_ms, self.last_update_ms) > 0
        ):
            self._predict(output_ms)
        return output_ms

    def update(self, candidate, now_ms, output_ms=None):
        """Update at measurement time, then predict to the bounded output time."""
        now_ms = int(now_ms)
        expired = self._is_expired(now_ms)
        if self.reacquire_requested and candidate is not None:
            self._reset_from_measurement(candidate, now_ms)
            state_time_ms = self._advance_to_output(output_ms)
            return self._snapshot(
                now_ms if state_time_ms is None else state_time_ms,
                measurement_accepted=True,
                filter_reset=True,
            )

        if not self.initialized or expired:
            if candidate is not None and self._confirm_new_target(
                candidate,
                now_ms,
            ):
                self._reset_from_measurement(candidate, now_ms)
                state_time_ms = self._advance_to_output(output_ms)
                return self._snapshot(
                    now_ms if state_time_ms is None else state_time_ms,
                    measurement_accepted=True,
                    filter_reset=True,
                )
            if candidate is None:
                self._clear_pending_acquisition()
            if not self.initialized:
                return self._snapshot(
                    now_ms if output_ms is None else int(output_ms)
                )
            # An expired track may coast for diagnostics, but no candidate is
            # allowed to update it until the new-target confirmation succeeds.
            candidate = None

        dt = self._predict(now_ms)
        accepted = False
        outlier_rejected = False
        if candidate is not None:
            if self._accept_measurement(candidate, dt):
                self._learn_axis(candidate)
                self._correct(candidate, dt, now_ms)
                self.last_measure_x = float(candidate["center_x"])
                self.last_measure_y = float(candidate["center_y"])
                self._update_diameter(candidate.get("diameter", 0.0))
                self.last_measurement_ms = now_ms
                self.missed_frames = 0
                accepted = True
            else:
                outlier_rejected = True

        if not accepted:
            self.missed_frames = min(255, self.missed_frames + 1)
            self.vx *= self.coast_velocity_decay
            self.vy *= self.coast_velocity_decay
            self.motion_confirm_count = 0
            self.motion_pending_sign = 0
        if (
            self._measurement_age(now_ms) > self.max_coast_ms
            or self.missed_frames > self.max_coast_frames
        ):
            self.vx = 0.0
            self.vy = 0.0
        state_time_ms = self._advance_to_output(output_ms)
        return self._snapshot(
            now_ms if state_time_ms is None else state_time_ms,
            measurement_accepted=accepted,
            outlier_rejected=outlier_rejected,
        )
