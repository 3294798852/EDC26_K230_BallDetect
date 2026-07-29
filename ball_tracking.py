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
    """Constant-velocity filter with a learned straight-pipe constraint.

    Before enough motion is observed, the state is an ordinary 2-D alpha-beta
    filter.  Once the displacement is large enough, an image-space pipe axis
    is locked.  Velocity is then projected onto that axis and lateral position
    corrections are deliberately weak.  The axial gains ease down for a small
    innovation and return to the fast gains for acceleration or reversal.
    Bounded candidate association preserves target identity, and an out-of-gate
    challenger must remain spatially consistent before it may replace a track.
    """

    def __init__(
        self,
        alpha=0.82,
        beta=0.28,
        lateral_alpha=0.12,
        lateral_beta=0.02,
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
        response_alpha=0.72,
        response_beta=0.18,
        response_start_px=0.8,
        response_full_px=4.0,
        response_speed_px_s=120.0,
        acquire_confirm_frames=3,
        acquire_timeout_ms=180,
        acquire_radius_px=36.0,
        acquire_min_confidence=0.50,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.response_alpha = min(float(response_alpha), self.alpha)
        self.response_beta = min(float(response_beta), self.beta)
        self.response_start_px = max(0.0, float(response_start_px))
        self.response_full_px = max(
            self.response_start_px + 0.001,
            float(response_full_px),
        )
        self.response_speed_px_s = max(1.0, float(response_speed_px_s))
        self.response_start_squared = self.response_start_px * self.response_start_px
        self.response_full_squared = self.response_full_px * self.response_full_px
        self.response_speed_squared = (
            self.response_speed_px_s * self.response_speed_px_s
        )
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

    def clear(self):
        self.initialized = False
        self.track_valid = False
        self.axis_locked = False
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.axis_x = 1.0
        self.axis_y = 0.0
        self.axis_seed_x = 0.0
        self.axis_seed_y = 0.0
        self.last_measure_x = 0.0
        self.last_measure_y = 0.0
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

    def _reset_from_measurement(self, candidate, now_ms):
        self.initialized = True
        self.track_valid = True
        self.axis_locked = False
        self.x = float(candidate["center_x"])
        self.y = float(candidate["center_y"])
        self.vx = 0.0
        self.vy = 0.0
        self.axis_x = 1.0
        self.axis_y = 0.0
        self.axis_seed_x = self.x
        self.axis_seed_y = self.y
        self.last_measure_x = self.x
        self.last_measure_y = self.y
        self.last_update_ms = int(now_ms)
        self.last_measurement_ms = int(now_ms)
        self.missed_frames = 0
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
                or float(candidate.get("confidence", 0.0))
                > float(best.get("confidence", 0.0))
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
        return total_gate

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
        challenger = None
        challenger_distance = 0.0
        for candidate in candidates:
            if candidate is None:
                continue
            dx = float(candidate.get("center_x", 0.0)) - predicted_x
            dy = float(candidate.get("center_y", 0.0)) - predicted_y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= self._measurement_gate(candidate, dt, speed):
                if (
                    best is None
                    or distance < best_distance
                    or (
                        abs(distance - best_distance) <= 0.5
                        and float(candidate.get("confidence", 0.0))
                        > float(best.get("confidence", 0.0))
                    )
                ):
                    best = candidate
                    best_distance = distance
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

    def _correct(self, candidate, dt):
        residual_x = float(candidate["center_x"]) - self.x
        residual_y = float(candidate["center_y"]) - self.y
        residual_squared = residual_x * residual_x + residual_y * residual_y
        response = _clamp(
            (residual_squared - self.response_start_squared)
            / (self.response_full_squared - self.response_start_squared),
            0.0,
            1.0,
        )
        speed_squared = self.vx * self.vx + self.vy * self.vy
        speed_response = _clamp(
            speed_squared / self.response_speed_squared,
            0.0,
            1.0,
        )
        response = max(response, speed_response)
        alpha = self.response_alpha + (
            self.alpha - self.response_alpha
        ) * response
        beta = self.response_beta + (
            self.beta - self.response_beta
        ) * response
        if not self.axis_locked:
            self.x += alpha * residual_x
            self.y += alpha * residual_y
            self.vx += beta * residual_x / dt
            self.vy += beta * residual_y / dt
            self._bound_state()
            return

        normal_x = -self.axis_y
        normal_y = self.axis_x
        along = residual_x * self.axis_x + residual_y * self.axis_y
        lateral = residual_x * normal_x + residual_y * normal_y
        self.x += (
            alpha * along * self.axis_x
            + self.lateral_alpha * lateral * normal_x
        )
        self.y += (
            alpha * along * self.axis_y
            + self.lateral_alpha * lateral * normal_y
        )
        along_velocity = self.vx * self.axis_x + self.vy * self.axis_y
        along_velocity += beta * along / dt
        lateral_velocity = self.lateral_beta * lateral / dt
        self.vx = (
            along_velocity * self.axis_x
            + lateral_velocity * normal_x
        )
        self.vy = (
            along_velocity * self.axis_y
            + lateral_velocity * normal_y
        )
        # The pipe is one-dimensional: retain only axial velocity.
        along_velocity = self.vx * self.axis_x + self.vy * self.axis_y
        self.vx = along_velocity * self.axis_x
        self.vy = along_velocity * self.axis_y
        self._bound_state()

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
        }

    def update(self, candidate, now_ms):
        """Update from one associated candidate or from a miss."""
        now_ms = int(now_ms)
        expired = self._is_expired(now_ms)
        if self.reacquire_requested and candidate is not None:
            self._reset_from_measurement(candidate, now_ms)
            return self._snapshot(
                now_ms,
                measurement_accepted=True,
                filter_reset=True,
            )

        if not self.initialized or expired:
            if candidate is not None and self._confirm_new_target(
                candidate,
                now_ms,
            ):
                self._reset_from_measurement(candidate, now_ms)
                return self._snapshot(
                    now_ms,
                    measurement_accepted=True,
                    filter_reset=True,
                )
            if candidate is None:
                self._clear_pending_acquisition()
            if not self.initialized:
                return self._snapshot(now_ms)
            # An expired track may coast for diagnostics, but no candidate is
            # allowed to update it until the new-target confirmation succeeds.
            candidate = None

        dt = self._predict(now_ms)
        accepted = False
        outlier_rejected = False
        if candidate is not None:
            if self._accept_measurement(candidate, dt):
                self._learn_axis(candidate)
                self._correct(candidate, dt)
                self.last_measure_x = float(candidate["center_x"])
                self.last_measure_y = float(candidate["center_y"])
                self.last_measurement_ms = now_ms
                self.missed_frames = 0
                accepted = True
            else:
                outlier_rejected = True

        if not accepted:
            self.missed_frames = min(255, self.missed_frames + 1)
            self.vx *= self.coast_velocity_decay
            self.vy *= self.coast_velocity_decay
        if (
            self._measurement_age(now_ms) > self.max_coast_ms
            or self.missed_frames > self.max_coast_frames
        ):
            self.vx = 0.0
            self.vy = 0.0
        return self._snapshot(
            now_ms,
            measurement_accepted=accepted,
            outlier_rejected=outlier_rejected,
        )
