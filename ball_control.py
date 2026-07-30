"""Visual steel-ball state feedback for the K230 camera task."""

import time

from ball_calibration import (
    X_CALIBRATION_PINHOLE,
    get_x_calibration_lut,
    x_center_scale,
    x_position_and_scale,
    y_position_cm,
)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _ticks_diff(new_value, old_value):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(int(new_value), int(old_value))
    return int(new_value) - int(old_value)


class BallStateFeedback:
    """Convert filtered pixels into physical state and feedback acceleration.

    K230 owns the camera-derived state-feedback term and current stationary
    reference preset. Position and velocity use continuous soft deadbands before
    their gains are applied. The MCU owns any future moving trajectory,
    car-acceleration feedforward, beam model inversion, measured-angle inner
    loop and all actuator safety decisions.
    """

    def __init__(
        self,
        kp=16.0,
        kd=7.2,
        ka=0.0,
        max_feedback_accel_cm_s2=250.0,
        max_measurement_age_ms=80,
        max_prediction_age_ms=300,
        max_prediction_frames=8,
        position_deadband_cm=0.05,
        velocity_deadband_cm_s=1.0,
        acceleration_alpha=0.30,
        max_measured_accel_cm_s2=2000.0,
        x_calibration_mode=X_CALIBRATION_PINHOLE,
        reference_position_cm=0.0,
        max_abs_reference_position_cm=12.5,
        reference_state_flags=0,
    ):
        self.x_calibration_mode = str(x_calibration_mode)
        self.x_calibration_lut = get_x_calibration_lut(
            self.x_calibration_mode
        )
        self.calibration_valid = True
        self.center_cm_per_pixel = x_center_scale(self.x_calibration_lut)
        self.max_abs_reference_position_cm = max(
            0.1,
            float(max_abs_reference_position_cm),
        )
        self.reference_position_cm = 0.0
        self.reference_state_flags = 0
        self.set_reference_position_cm(
            reference_position_cm,
            reference_state_flags,
        )
        self.kp = max(0.0, float(kp))
        self.kd = max(0.0, float(kd))
        self.ka = max(0.0, float(ka))
        self.max_feedback_accel_cm_s2 = max(
            1.0,
            float(max_feedback_accel_cm_s2),
        )
        self.max_measurement_age_ms = max(1, int(max_measurement_age_ms))
        self.max_prediction_age_ms = max(1, int(max_prediction_age_ms))
        self.max_prediction_frames = max(1, int(max_prediction_frames))
        self.position_deadband_cm = max(0.0, float(position_deadband_cm))
        self.velocity_deadband_cm_s = max(
            0.0,
            float(velocity_deadband_cm_s),
        )
        self.acceleration_alpha = _clamp(
            float(acceleration_alpha),
            0.0,
            1.0,
        )
        self.max_measured_accel_cm_s2 = max(
            1.0,
            float(max_measured_accel_cm_s2),
        )
        self.last_update_ms = None
        self.last_velocity_cm_s = None
        self.prediction_armed = False
        self.acceleration_cm_s2 = 0.0

    def set_reference_position_cm(
        self,
        reference_position_cm,
        reference_state_flags=None,
    ):
        """Set the absolute calibrated X coordinate used as equilibrium."""
        value = float(reference_position_cm)
        if value != value or abs(value) > self.max_abs_reference_position_cm:
            raise ValueError(
                "reference position must be within +/-{:.2f} cm".format(
                    self.max_abs_reference_position_cm
                )
            )
        flags = self.reference_state_flags
        if reference_state_flags is not None:
            flags = int(reference_state_flags)
            if flags < 0 or flags > 0xFFFF:
                raise ValueError("reference state flags must fit u16")
        self.reference_position_cm = value
        self.reference_state_flags = flags
        return self.reference_position_cm

    def reset_runtime(self):
        self.last_update_ms = None
        self.last_velocity_cm_s = None
        self.prediction_armed = False
        self.acceleration_cm_s2 = 0.0

    def _physical_state(self, state):
        position_x_cm, x_cm_per_pixel = x_position_and_scale(
            state.get("x", 0.0),
            self.x_calibration_lut,
        )
        position_y_cm = y_position_cm(
            state.get("y", 0.0),
            self.center_cm_per_pixel,
        )
        velocity_x_cm_s = (
            float(state.get("vx", 0.0)) * x_cm_per_pixel
        )
        velocity_y_cm_s = (
            float(state.get("vy", 0.0)) * self.center_cm_per_pixel
        )
        return (
            position_x_cm,
            position_y_cm,
            velocity_x_cm_s,
            velocity_y_cm_s,
        )

    def _apply_deadband(self, value, width):
        value = float(value)
        magnitude = abs(value)
        if magnitude <= width:
            return 0.0
        if value < 0.0:
            return -(magnitude - width)
        return magnitude - width

    def _update_acceleration(self, velocity_cm_s, dt):
        if self.last_velocity_cm_s is None:
            self.acceleration_cm_s2 = 0.0
        else:
            raw_acceleration = _clamp(
                (float(velocity_cm_s) - self.last_velocity_cm_s) / dt,
                -self.max_measured_accel_cm_s2,
                self.max_measured_accel_cm_s2,
            )
            self.acceleration_cm_s2 += self.acceleration_alpha * (
                raw_acceleration - self.acceleration_cm_s2
            )
        self.last_velocity_cm_s = float(velocity_cm_s)

    def update(self, state, candidate, now_ms):
        now_ms = int(now_ms)
        if self.last_update_ms is None:
            dt = 0.035
        else:
            dt = _clamp(
                _ticks_diff(now_ms, self.last_update_ms) / 1000.0,
                0.010,
                0.200,
            )
        self.last_update_ms = now_ms

        if not state.get("track_valid"):
            self.reset_runtime()
            return self._snapshot(0.0, 0.0, 0.0, 0.0, False, False)

        (
            position_cm,
            position_y_cm,
            velocity_cm_s,
            velocity_y_cm_s,
        ) = self._physical_state(state)
        measurement_age_ms = int(state.get("measurement_age_ms", 0xFFFF))
        missed_frames = int(state.get("missed_frames", 0xFFFF))
        measurement_current = bool(
            state.get("measurement_accepted")
            and candidate is not None
            and not candidate.get("edge_clipped")
            and measurement_age_ms <= self.max_measurement_age_ms
        )
        if state.get("measurement_accepted"):
            self.prediction_armed = measurement_current
        prediction_current = bool(
            self.prediction_armed
            and state.get("predicted")
            and 0 < missed_frames <= self.max_prediction_frames
            and measurement_age_ms <= self.max_prediction_age_ms
        )
        feedback_current = measurement_current or prediction_current
        if state.get("filter_reset"):
            self.last_velocity_cm_s = None
            self.acceleration_cm_s2 = 0.0
        if not feedback_current:
            self.last_velocity_cm_s = None
            self.acceleration_cm_s2 = 0.0
            return self._snapshot(
                position_cm,
                position_y_cm,
                velocity_cm_s,
                velocity_y_cm_s,
                False,
                False,
            )

        if prediction_current:
            self.last_velocity_cm_s = None
            self.acceleration_cm_s2 = 0.0
        else:
            self._update_acceleration(velocity_cm_s, dt)
        feedback_position_cm = self._apply_deadband(
            position_cm - self.reference_position_cm,
            self.position_deadband_cm,
        )
        feedback_velocity_cm_s = self._apply_deadband(
            velocity_cm_s,
            self.velocity_deadband_cm_s,
        )
        unconstrained_feedback = -(
            self.kp * feedback_position_cm
            + self.kd * feedback_velocity_cm_s
            + self.ka * self.acceleration_cm_s2
        )
        feedback_accel = _clamp(
            unconstrained_feedback,
            -self.max_feedback_accel_cm_s2,
            self.max_feedback_accel_cm_s2,
        )
        return self._snapshot(
            position_cm,
            position_y_cm,
            velocity_cm_s,
            velocity_y_cm_s,
            self.calibration_valid,
            feedback_accel != unconstrained_feedback,
            feedback_accel,
        )

    def _snapshot(
        self,
        position_cm,
        position_y_cm,
        velocity_cm_s,
        velocity_y_cm_s,
        feedback_valid,
        feedback_saturated,
        feedback_acceleration_cm_s2=0.0,
    ):
        position_error_cm = 0.0
        if feedback_valid:
            position_error_cm = position_cm - self.reference_position_cm
        return {
            "position_cm": float(position_cm),
            "reference_position_cm": float(self.reference_position_cm),
            "position_error_cm": float(position_error_cm),
            "reference_velocity_cm_s": 0.0,
            "reference_acceleration_cm_s2": 0.0,
            "reference_state_flags": int(self.reference_state_flags),
            "position_y_cm": float(position_y_cm),
            "velocity_cm_s": float(velocity_cm_s),
            "velocity_y_cm_s": float(velocity_y_cm_s),
            "acceleration_cm_s2": float(self.acceleration_cm_s2),
            "feedback_acceleration_cm_s2": float(
                feedback_acceleration_cm_s2
            ),
            "feedback_valid": bool(feedback_valid),
            "feedback_saturated": bool(feedback_saturated),
            "calibration_valid": bool(self.calibration_valid),
            "center_cm_per_pixel": float(self.center_cm_per_pixel),
            "kp": float(self.kp),
            "kd": float(self.kd),
            "ka": float(self.ka),
            "compute_ms": 0,
        }
