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


def _low_pass_alpha(dt, time_constant):
    if time_constant <= 0.0:
        return 1.0
    return _clamp(float(dt) / (float(time_constant) + float(dt)), 0.0, 1.0)


def _soft_deadband(value, width):
    value = float(value)
    magnitude = abs(value)
    if magnitude <= width:
        return 0.0
    if value < 0.0:
        return -(magnitude - width)
    return magnitude - width


class BallStateFeedback:
    """Convert tracked pixels into physical state and continuous PI-D feedback.

    The control path deliberately has no motion-state switching. A single
    fixed-filter PI-D equation runs every valid frame. Only a small joint
    position-and-velocity settling band suppresses dynamic feedback; its
    retained integral value continues to cancel a constant pipe inclination.
    """

    def __init__(
        self,
        kp=8.0,
        kd=12.0,
        ki=12.0,
        ka=0.0,
        max_feedback_accel_cm_s2=140.0,
        max_measurement_age_ms=80,
        max_prediction_age_ms=300,
        max_prediction_frames=8,
        position_deadband_cm=0.10,
        velocity_deadband_cm_s=0.35,
        position_filter_tau_s=0.025,
        velocity_filter_tau_s=0.050,
        feedback_filter_tau_s=0.025,
        integral_limit_cm_s2=80.0,
        integral_error_limit_cm=1.0,
        integral_leak_per_s=0.0,
        integral_unwind_gain_s2=20.0,
        breakaway_accel_cm_s2=20.0,
        breakaway_transition_cm=0.20,
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
        self.ki = max(0.0, float(ki))
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
        self.position_filter_tau_s = max(
            0.0,
            float(position_filter_tau_s),
        )
        self.velocity_filter_tau_s = max(
            0.0,
            float(velocity_filter_tau_s),
        )
        self.feedback_filter_tau_s = max(
            0.0,
            float(feedback_filter_tau_s),
        )
        self.integral_limit_cm_s2 = max(
            0.0,
            float(integral_limit_cm_s2),
        )
        self.integral_error_limit_cm = max(
            0.0,
            float(integral_error_limit_cm),
        )
        self.integral_leak_per_s = max(
            0.0,
            float(integral_leak_per_s),
        )
        self.integral_unwind_gain_s2 = max(
            0.0,
            float(integral_unwind_gain_s2),
        )
        self.breakaway_accel_cm_s2 = max(
            0.0,
            float(breakaway_accel_cm_s2),
        )
        self.breakaway_transition_cm = max(
            0.01,
            float(breakaway_transition_cm),
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
        self.prediction_armed = False
        self.filtered_position_cm = None
        self.filtered_velocity_cm_s = None
        self.last_velocity_cm_s = None
        self.acceleration_cm_s2 = 0.0
        self.integral_correction_cm_s2 = 0.0
        self.breakaway_feedback_cm_s2 = 0.0
        self.feedback_acceleration_cm_s2 = 0.0
        self.feedback_initialized = False
        self.integral_state = "RESET"
        self.integral_learning = False
        self.integral_limited = False
        self.integral_unwind_active = False
        self.joint_deadband_active = False

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

    @staticmethod
    def _validated_gain(name, value):
        value = float(value)
        if value != value or value < 0.0 or value > 1000.0:
            raise ValueError("{} gain must be finite and within 0..1000".format(name))
        return value

    def set_gains(self, kp=None, kd=None, ki=None, ka=None):
        """Update controller gains without resetting the live control state."""
        if kp is not None:
            self.kp = self._validated_gain("Kp", kp)
        if kd is not None:
            self.kd = self._validated_gain("Kd", kd)
        if ki is not None:
            self.ki = self._validated_gain("Ki", ki)
        if ka is not None:
            self.ka = self._validated_gain("Ka", ka)
        return self.kp, self.kd, self.ki, self.ka

    def reset_runtime(self):
        self.last_update_ms = None
        self.prediction_armed = False
        self.filtered_position_cm = None
        self.filtered_velocity_cm_s = None
        self.last_velocity_cm_s = None
        self.acceleration_cm_s2 = 0.0
        self.integral_correction_cm_s2 = 0.0
        self.breakaway_feedback_cm_s2 = 0.0
        self.feedback_acceleration_cm_s2 = 0.0
        self.feedback_initialized = False
        self.integral_state = "RESET"
        self.integral_learning = False
        self.integral_limited = False
        self.integral_unwind_active = False
        self.joint_deadband_active = False

    def _physical_state(self, state):
        position_x_cm, x_cm_per_pixel = x_position_and_scale(
            state.get("x", 0.0),
            self.x_calibration_lut,
        )
        position_y_cm = y_position_cm(
            state.get("y", 0.0),
            self.center_cm_per_pixel,
        )
        velocity_x_cm_s = float(state.get("vx", 0.0)) * x_cm_per_pixel
        velocity_y_cm_s = (
            float(state.get("vy", 0.0)) * self.center_cm_per_pixel
        )
        return (
            position_x_cm,
            position_y_cm,
            velocity_x_cm_s,
            velocity_y_cm_s,
        )

    def _update_control_state(self, position_cm, velocity_cm_s, dt):
        if self.filtered_position_cm is None:
            self.filtered_position_cm = float(position_cm)
        else:
            position_alpha = _low_pass_alpha(
                dt,
                self.position_filter_tau_s,
            )
            self.filtered_position_cm += position_alpha * (
                float(position_cm) - self.filtered_position_cm
            )
        if self.filtered_velocity_cm_s is None:
            self.filtered_velocity_cm_s = float(velocity_cm_s)
        else:
            velocity_alpha = _low_pass_alpha(
                dt,
                self.velocity_filter_tau_s,
            )
            self.filtered_velocity_cm_s += velocity_alpha * (
                float(velocity_cm_s) - self.filtered_velocity_cm_s
            )

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

    def _update_integral(
        self,
        position_error_cm,
        center_error_cm,
        center_velocity_cm_s,
        dynamic_feedback_cm_s2,
        measurement_current,
        settled,
        dt,
    ):
        self.integral_learning = False
        self.integral_limited = False
        self.integral_unwind_active = False
        if self.ki <= 0.0 or self.integral_limit_cm_s2 <= 0.0:
            self.integral_correction_cm_s2 = 0.0
            self.integral_state = "OFF"
            return
        if not measurement_current:
            self.integral_state = "HOLD"
            return
        if settled:
            self.integral_state = "SETTLE"
            return

        previous_correction = self.integral_correction_cm_s2
        moving_toward_reference = bool(
            float(center_error_cm) * float(center_velocity_cm_s) < 0.0
        )
        unwind_active = bool(
            previous_correction != 0.0
            and moving_toward_reference
            and self.integral_unwind_gain_s2 > 0.0
        )
        if unwind_active:
            unwind_amount = min(
                abs(previous_correction),
                self.integral_unwind_gain_s2
                * abs(float(center_velocity_cm_s))
                * float(dt),
            )
            if previous_correction < 0.0:
                proposed_correction = previous_correction + unwind_amount
            else:
                proposed_correction = previous_correction - unwind_amount
        else:
            bounded_error_cm = _clamp(
                float(position_error_cm),
                -self.integral_error_limit_cm,
                self.integral_error_limit_cm,
            )
            integral_rate = -(
                self.ki * bounded_error_cm
                + self.integral_leak_per_s
                * previous_correction
            )
            proposed_correction = _clamp(
                previous_correction + integral_rate * float(dt),
                -self.integral_limit_cm_s2,
                self.integral_limit_cm_s2,
            )
        correction_delta = proposed_correction - previous_correction
        proposed_feedback = dynamic_feedback_cm_s2 + proposed_correction
        drives_positive_saturation = bool(
            proposed_feedback > self.max_feedback_accel_cm_s2
            and correction_delta > 0.0
        )
        drives_negative_saturation = bool(
            proposed_feedback < -self.max_feedback_accel_cm_s2
            and correction_delta < 0.0
        )
        if (
            not unwind_active
            and (drives_positive_saturation or drives_negative_saturation)
        ):
            self.integral_limited = True
            self.integral_state = "LIMIT"
            return
        if proposed_correction != previous_correction:
            self.integral_correction_cm_s2 = proposed_correction
            if unwind_active:
                self.integral_unwind_active = True
                self.integral_state = "UNWIND"
            elif abs(proposed_correction) >= self.integral_limit_cm_s2:
                self.integral_learning = True
                self.integral_limited = True
                self.integral_state = "LIMIT"
            else:
                self.integral_learning = True
                self.integral_state = "TRACK"
            return
        self.integral_state = "HOLD"

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
            self.reset_runtime()
            self.last_update_ms = now_ms
        if not feedback_current:
            self.reset_runtime()
            return self._snapshot(
                position_cm,
                position_y_cm,
                velocity_cm_s,
                velocity_y_cm_s,
                False,
                False,
            )

        self._update_control_state(position_cm, velocity_cm_s, dt)
        if prediction_current:
            self.last_velocity_cm_s = None
            self.acceleration_cm_s2 = 0.0
        else:
            self._update_acceleration(self.filtered_velocity_cm_s, dt)

        control_error_cm = (
            self.filtered_position_cm - self.reference_position_cm
        )
        control_velocity_cm_s = self.filtered_velocity_cm_s
        feedback_position_cm = _soft_deadband(
            control_error_cm,
            self.position_deadband_cm,
        )
        feedback_velocity_cm_s = _soft_deadband(
            control_velocity_cm_s,
            self.velocity_deadband_cm_s,
        )
        self.joint_deadband_active = bool(
            feedback_position_cm == 0.0
            and feedback_velocity_cm_s == 0.0
        )
        dynamic_feedback = -(
            self.kp * feedback_position_cm
            + self.kd * feedback_velocity_cm_s
            + self.ka * self.acceleration_cm_s2
        )
        self.breakaway_feedback_cm_s2 = -(
            self.breakaway_accel_cm_s2
            * _clamp(
                feedback_position_cm / self.breakaway_transition_cm,
                -1.0,
                1.0,
            )
        )
        non_integral_feedback = (
            dynamic_feedback + self.breakaway_feedback_cm_s2
        )
        self._update_integral(
            feedback_position_cm,
            control_error_cm,
            feedback_velocity_cm_s,
            non_integral_feedback,
            measurement_current,
            self.joint_deadband_active,
            dt,
        )
        unconstrained_feedback = (
            non_integral_feedback + self.integral_correction_cm_s2
        )
        if not self.feedback_initialized:
            self.feedback_acceleration_cm_s2 = unconstrained_feedback
            self.feedback_initialized = True
        else:
            output_alpha = _low_pass_alpha(dt, self.feedback_filter_tau_s)
            self.feedback_acceleration_cm_s2 += output_alpha * (
                unconstrained_feedback - self.feedback_acceleration_cm_s2
            )
        stabilized_feedback = self.feedback_acceleration_cm_s2
        feedback_accel = _clamp(
            stabilized_feedback,
            -self.max_feedback_accel_cm_s2,
            self.max_feedback_accel_cm_s2,
        )
        return self._snapshot(
            position_cm,
            position_y_cm,
            velocity_cm_s,
            velocity_y_cm_s,
            self.calibration_valid,
            feedback_accel != stabilized_feedback,
            feedback_accel,
            unconstrained_feedback,
            str(state.get("motion_state", "?")),
            control_error_cm,
            feedback_velocity_cm_s,
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
        unfiltered_feedback_acceleration_cm_s2=0.0,
        motion_state="?",
        control_error_cm=0.0,
        control_velocity_cm_s=0.0,
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
            "unfiltered_feedback_acceleration_cm_s2": float(
                unfiltered_feedback_acceleration_cm_s2
            ),
            "motion_state": str(motion_state),
            "velocity_feedback_weight": 1.0,
            "feedback_velocity_cm_s": float(control_velocity_cm_s),
            "control_position_error_cm": float(control_error_cm),
            "joint_deadband_active": bool(self.joint_deadband_active),
            "feedback_valid": bool(feedback_valid),
            "feedback_saturated": bool(feedback_saturated),
            "calibration_valid": bool(self.calibration_valid),
            "center_cm_per_pixel": float(self.center_cm_per_pixel),
            "kp": float(self.kp),
            "kd": float(self.kd),
            "ka": float(self.ka),
            "ki": float(self.ki),
            "breakaway_feedback_cm_s2": float(
                self.breakaway_feedback_cm_s2
            ),
            "breakaway_accel_cm_s2": float(self.breakaway_accel_cm_s2),
            "breakaway_transition_cm": float(
                self.breakaway_transition_cm
            ),
            "steady_correction_cm_s2": float(
                self.integral_correction_cm_s2
            ),
            "steady_correction_target_cm_s2": float(
                self.integral_correction_cm_s2
            ),
            "steady_correction_capture_count": 0,
            "steady_correction_capture_armed": False,
            "steady_correction_unwind_active": bool(
                self.integral_unwind_active
            ),
            "steady_correction_enabled": bool(
                self.ki > 0.0 and self.integral_limit_cm_s2 > 0.0
            ),
            "steady_correction_eligible": bool(
                self.integral_state not in ("OFF", "HOLD", "RESET")
            ),
            "steady_correction_learning": bool(self.integral_learning),
            "steady_correction_limited": bool(self.integral_limited),
            "steady_correction_dwell_ms": 0,
            "steady_correction_state": str(self.integral_state),
            "integral_unwind_gain_s2": float(
                self.integral_unwind_gain_s2
            ),
            "compute_ms": 0,
        }
