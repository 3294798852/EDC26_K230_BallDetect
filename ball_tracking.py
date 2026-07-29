"""Highest-confidence selection and constrained steel-ball motion filter."""

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


def select_highest_confidence_ball(raw_result, frame_width=320, frame_height=320):
    """Return only class-0's highest-confidence valid detection."""
    if raw_result is None:
        return None
    try:
        if len(raw_result) < 3:
            return None
        boxes = raw_result[0]
        class_ids = raw_result[1]
        scores = raw_result[2]
        count = min(len(boxes), len(class_ids), len(scores))
    except Exception:
        return None

    best = None
    for index in range(count):
        try:
            if int(class_ids[index]) != 0:
                continue
            box = boxes[index]
            raw_x = float(box[0])
            raw_y = float(box[1])
            raw_w = float(box[2])
            raw_h = float(box[3])
            confidence = float(scores[index])
        except Exception:
            continue
        if confidence != confidence or raw_w <= 0.0 or raw_h <= 0.0:
            continue
        if best is not None and confidence <= best["confidence"]:
            continue

        raw_x1 = raw_x + raw_w
        raw_y1 = raw_y + raw_h
        if (
            raw_x1 <= 0.0
            or raw_y1 <= 0.0
            or raw_x >= frame_width
            or raw_y >= frame_height
        ):
            continue

        x0 = int(_clamp(round(raw_x), 0, max(0, frame_width - 1)))
        y0 = int(_clamp(round(raw_y), 0, max(0, frame_height - 1)))
        x1 = int(_clamp(round(raw_x1), x0 + 1, frame_width))
        y1 = int(_clamp(round(raw_y1), y0 + 1, frame_height))
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        edge_clipped = (
            raw_x < 0.0
            or raw_y < 0.0
            or raw_x1 > frame_width
            or raw_y1 > frame_height
            or x0 == 0
            or y0 == 0
            or x1 == frame_width
            or y1 == frame_height
        )
        best = {
            "confidence": _clamp(confidence, 0.0, 1.0),
            "center_x": x0 + width * 0.5,
            "center_y": y0 + height * 0.5,
            "x": x0,
            "y": y0,
            "w": width,
            "h": height,
            "diameter": (width + height) * 0.5,
            "edge_clipped": edge_clipped,
            "source_index": index,
        }
    return best


class PipeMotionFilter:
    """Constant-velocity filter with a learned straight-pipe constraint.

    Before enough motion is observed, the state is an ordinary 2-D alpha-beta
    filter.  Once the displacement is large enough, an image-space pipe axis
    is locked.  Velocity is then projected onto that axis and lateral position
    corrections are deliberately weak.  A bounded innovation gate rejects a
    high-confidence false target without falling back to a second detection.
    """

    def __init__(
        self,
        alpha=0.68,
        beta=0.16,
        lateral_alpha=0.12,
        lateral_beta=0.02,
        base_gate_px=64.0,
        max_gate_px=110.0,
        lateral_gate_px=14.0,
        axis_lock_distance_px=12.0,
        axis_adapt=0.08,
        max_coast_ms=250,
        coast_velocity_decay=0.98,
        frame_width=320,
        frame_height=320,
        max_speed_px_s=2500.0,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lateral_alpha = float(lateral_alpha)
        self.lateral_beta = float(lateral_beta)
        self.base_gate_px = float(base_gate_px)
        self.max_gate_px = float(max_gate_px)
        self.lateral_gate_px = float(lateral_gate_px)
        self.axis_lock_distance_px = float(axis_lock_distance_px)
        self.axis_adapt = float(axis_adapt)
        self.max_coast_ms = max(1, int(max_coast_ms))
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

    def _accept_measurement(self, candidate, dt):
        residual_x = float(candidate["center_x"]) - self.x
        residual_y = float(candidate["center_y"]) - self.y
        residual_distance = math.sqrt(
            residual_x * residual_x + residual_y * residual_y
        )
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        total_gate = _clamp(
            self.base_gate_px + speed * dt * 1.5,
            self.base_gate_px,
            self.max_gate_px,
        )
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
        if not self.axis_locked:
            self.x += self.alpha * residual_x
            self.y += self.alpha * residual_y
            self.vx += self.beta * residual_x / dt
            self.vy += self.beta * residual_y / dt
            self._bound_state()
            return

        normal_x = -self.axis_y
        normal_y = self.axis_x
        along = residual_x * self.axis_x + residual_y * self.axis_y
        lateral = residual_x * normal_x + residual_y * normal_y
        self.x += (
            self.alpha * along * self.axis_x
            + self.lateral_alpha * lateral * normal_x
        )
        self.y += (
            self.alpha * along * self.axis_y
            + self.lateral_alpha * lateral * normal_y
        )
        along_velocity = self.vx * self.axis_x + self.vy * self.axis_y
        along_velocity += self.beta * along / dt
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
            self.initialized and age_ms <= self.max_coast_ms
        )
        return {
            "measurement_accepted": bool(measurement_accepted),
            "track_valid": self.track_valid,
            "predicted": bool(self.initialized and not measurement_accepted),
            "axis_locked": self.axis_locked,
            "outlier_rejected": bool(outlier_rejected),
            "filter_reset": bool(filter_reset),
            "missed_frames": min(255, int(self.missed_frames)),
            "measurement_age_ms": age_ms,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "axis_x": self.axis_x,
            "axis_y": self.axis_y,
        }

    def update(self, candidate, now_ms):
        """Update from one highest-confidence candidate or from a miss."""
        now_ms = int(now_ms)
        expired = (
            self.last_measurement_ms is None
            or self._measurement_age(now_ms) > self.max_coast_ms
        )
        if candidate is not None and (not self.initialized or expired):
            self._reset_from_measurement(candidate, now_ms)
            return self._snapshot(
                now_ms,
                measurement_accepted=True,
                filter_reset=True,
            )
        if not self.initialized:
            return self._snapshot(now_ms)

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
        if self._measurement_age(now_ms) > self.max_coast_ms:
            self.vx = 0.0
            self.vy = 0.0
        return self._snapshot(
            now_ms,
            measurement_accepted=accepted,
            outlier_rejected=outlier_rejected,
        )
