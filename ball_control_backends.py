"""Selectable K230 ball-feedback control backends."""

import math

from ball_control import BallStateFeedback, _clamp


REFERENCE_BACKEND = "reference_profile"
PID_BACKEND = "pid"
CASCADE_PID_BACKEND = "cascade_pid"
SUPPORTED_CONTROL_BACKENDS = (
    PID_BACKEND,
    REFERENCE_BACKEND,
    CASCADE_PID_BACKEND,
)


def _approach(value, target, maximum_step):
    if value < target:
        return min(target, value + maximum_step)
    if value > target:
        return max(target, value - maximum_step)
    return target


class ReferenceProfileFeedback(BallStateFeedback):
    """Physical-unit port of the supplied upper-computer ball controller.

    The source controller produces pipe-angle corrections from a pipe-axis
    pixel coordinate. This port preserves its profile switching, approach
    damping, integral gate, slow trim and bounded breakaway bias, but converts
    the final pipe angle to rolling-ball acceleration for the MCU-owned plant
    inversion and angle inner loop.
    """

    backend_name = REFERENCE_BACKEND

    def __init__(self, reference_settings=None, **common_kwargs):
        settings = reference_settings or {}
        common_kwargs["pd_schedule_enabled"] = False
        self.source_axis_length_px = max(
            1.0,
            float(settings.get("source_axis_length_px", 383.0)),
        )
        self.pipe_length_cm = max(
            0.1,
            float(settings.get("pipe_length_cm", 25.0)),
        )
        self.source_pixels_per_cm = (
            self.source_axis_length_px / self.pipe_length_cm
        )
        self.gravity_cm_s2 = max(
            1.0,
            float(settings.get("gravity_cm_s2", 980.665)),
        )
        self.rolling_factor = _clamp(
            float(settings.get("rolling_factor", 5.0 / 7.0)),
            0.01,
            1.0,
        )
        self.accel_per_degree_small = (
            self.rolling_factor
            * self.gravity_cm_s2
            * math.pi
            / 180.0
        )
        gain_scale = self.accel_per_degree_small * self.source_pixels_per_cm
        rolling_kp_deg_per_px = max(
            0.0,
            float(settings.get("rolling_kp_deg_per_px", 0.035)),
        )
        rolling_ki_deg_per_px_s = max(
            0.0,
            float(settings.get("rolling_ki_deg_per_px_s", 0.0)),
        )
        rolling_kd_deg_s_per_px = max(
            0.0,
            float(settings.get("rolling_kd_deg_s_per_px", 0.012)),
        )
        common_kwargs["kp"] = rolling_kp_deg_per_px * gain_scale
        common_kwargs["ki"] = rolling_ki_deg_per_px_s * gain_scale
        common_kwargs["kd"] = rolling_kd_deg_s_per_px * gain_scale
        common_kwargs["ka"] = 0.0
        super().__init__(**common_kwargs)

        self.stationary_kp_ratio = max(
            1.0,
            float(settings.get("stationary_kp_deg_per_px", 0.055))
            / max(rolling_kp_deg_per_px, 0.000001),
        )
        deadband_cm = settings.get("deadband_cm")
        if deadband_cm is None and "deadband_px" in settings:
            deadband_cm = (
                float(settings.get("deadband_px", 0.0))
                / self.source_pixels_per_cm
            )
        if deadband_cm is None:
            deadband_cm = common_kwargs.get("position_deadband_cm", 0.30)
        self.position_deadband_cm = max(0.0, float(deadband_cm))
        self.velocity_deadband_cm_s = 0.0
        self.proportional_limit_deg = max(
            0.0,
            float(settings.get("proportional_limit_deg", 8.0)),
        )
        self.derivative_limit_deg = max(
            0.0,
            float(settings.get("derivative_limit_deg", 5.0)),
        )
        self.dynamic_tilt_limit_deg = max(
            0.1,
            float(settings.get("dynamic_tilt_limit_deg", 10.0)),
        )
        self.target_brake_start_cm = (
            float(settings.get("target_brake_start_px", 30.0))
            / self.source_pixels_per_cm
        )
        self.target_brake_full_cm = (
            float(settings.get("target_brake_full_px", 8.0))
            / self.source_pixels_per_cm
        )
        if self.target_brake_start_cm <= self.target_brake_full_cm:
            raise ValueError("reference brake start must exceed full distance")
        self.target_brake_far_scale = _clamp(
            float(settings.get("target_brake_far_scale", 0.20)),
            0.0,
            1.0,
        )
        self.integral_active_error_cm = (
            float(settings.get("integral_active_error_px", 15.0))
            / self.source_pixels_per_cm
        )
        self.integral_active_velocity_cm_s = (
            float(settings.get("integral_active_velocity_px_s", 10.0))
            / self.source_pixels_per_cm
        )
        self.integral_limit_px_s = max(
            0.0,
            float(settings.get("integral_limit_px_s", 500.0)),
        )
        self.stationary_confirm_s = max(
            0.0,
            float(settings.get("stationary_confirm_s", 0.35)),
        )
        self.stationary_velocity_cm_s = (
            float(settings.get("stationary_velocity_px_s", 2.0))
            / self.source_pixels_per_cm
        )
        self.rolling_enter_velocity_cm_s = (
            float(settings.get("rolling_enter_velocity_px_s", 6.0))
            / self.source_pixels_per_cm
        )
        self.rolling_enter_displacement_cm = (
            float(settings.get("rolling_enter_displacement_px", 1.5))
            / self.source_pixels_per_cm
        )
        self.breakaway_error_min_cm = (
            float(settings.get("breakaway_error_min_px", 5.0))
            / self.source_pixels_per_cm
        )
        self.breakaway_initial_deg = max(
            0.0,
            float(settings.get("breakaway_initial_deg", 3.0)),
        )
        self.breakaway_max_deg = max(
            self.breakaway_initial_deg,
            float(settings.get("breakaway_max_deg", 5.0)),
        )
        self.breakaway_ramp_deg_s = max(
            0.0,
            float(settings.get("breakaway_ramp_deg_s", 12.0)),
        )
        self.breakaway_escalation_deg_s = max(
            0.0,
            float(settings.get("breakaway_escalation_deg_s", 0.8)),
        )
        self.breakaway_error_gain_deg_per_cm = (
            max(0.0, float(settings.get("breakaway_error_gain_deg_per_px", 1.8)))
            * self.source_pixels_per_cm
        )
        self.breakaway_release_deg_s = max(
            0.0,
            float(settings.get("breakaway_release_deg_s", 30.0)),
        )
        self.trim_enabled = bool(settings.get("trim_enabled", True))
        self.trim_deadband_cm = (
            float(settings.get("trim_deadband_px", 5.0))
            / self.source_pixels_per_cm
        )
        self.trim_gain_deg_per_cm_s = (
            float(settings.get("trim_gain_deg_per_px_s", 0.40))
            * self.source_pixels_per_cm
        )
        self.trim_max_deg = max(
            0.0,
            float(settings.get("trim_max_deg", 4.0)),
        )
        self.trim_max_rate_deg_s = max(
            0.0,
            float(settings.get("trim_max_rate_deg_s", 1.2)),
        )
        self.trim_activation_ms = max(
            0,
            int(float(settings.get("trim_activation_s", 0.20)) * 1000.0),
        )
        self.trim_velocity_gate_cm_s = (
            float(settings.get("trim_velocity_gate_px_s", 12.0))
            / self.source_pixels_per_cm
        )
        self.reference_integral_px_s = 0.0
        self.reference_profile = "ROLLING"
        self.reference_brake_scale = 1.0
        self.reference_correction_deg = 0.0
        self.reference_breakaway_deg = 0.0
        self.reference_trim_deg = 0.0
        self.reference_stationary_since_ms = None
        self.reference_motion_anchor_cm = None
        self.reference_breakaway_started_ms = None
        self.reference_breakaway_elapsed_s = 0.0
        self.reference_trim_direction = 0
        self.reference_trim_since_ms = None
        self.reference_last_saturated = False

    def reset_runtime(self):
        super().reset_runtime()
        self.reference_integral_px_s = 0.0
        self.reference_profile = "ROLLING"
        self.reference_brake_scale = 1.0
        self.reference_correction_deg = 0.0
        self.reference_breakaway_deg = 0.0
        self.reference_trim_deg = 0.0
        self.reference_stationary_since_ms = None
        self.reference_motion_anchor_cm = None
        self.reference_breakaway_started_ms = None
        self.reference_breakaway_elapsed_s = 0.0
        self.reference_trim_direction = 0
        self.reference_trim_since_ms = None
        self.reference_last_saturated = False

    def _angle_to_acceleration(self, angle_deg):
        angle_deg = _clamp(
            float(angle_deg),
            -self.dynamic_tilt_limit_deg,
            self.dynamic_tilt_limit_deg,
        )
        return (
            self.rolling_factor
            * self.gravity_cm_s2
            * math.sin(angle_deg * math.pi / 180.0)
        )

    def _update_profile(self, error_cm, velocity_cm_s, measurement_current, now_ms):
        if not measurement_current:
            return
        speed = abs(float(velocity_cm_s))
        position = float(self.filtered_position_cm)
        if speed >= self.rolling_enter_velocity_cm_s:
            self.reference_profile = "ROLLING"
            self.reference_stationary_since_ms = None
            self.reference_motion_anchor_cm = position
            self.reference_breakaway_started_ms = None
            self.reference_breakaway_elapsed_s = 0.0
            return
        if self.reference_motion_anchor_cm is None:
            self.reference_motion_anchor_cm = position
        if (
            abs(position - self.reference_motion_anchor_cm)
            >= self.rolling_enter_displacement_cm
        ):
            self.reference_profile = "ROLLING"
            self.reference_stationary_since_ms = None
            self.reference_motion_anchor_cm = position
            self.reference_breakaway_started_ms = None
            self.reference_breakaway_elapsed_s = 0.0
            return
        if speed > self.stationary_velocity_cm_s:
            self.reference_stationary_since_ms = None
            return
        if self.reference_stationary_since_ms is None:
            self.reference_stationary_since_ms = int(now_ms)
            return
        if (
            int(now_ms) - self.reference_stationary_since_ms
            < int(self.stationary_confirm_s * 1000.0)
        ):
            return
        if abs(error_cm) > self.breakaway_error_min_cm:
            if self.reference_profile != "BREAKAWAY":
                self.reference_breakaway_started_ms = int(now_ms)
                self.reference_breakaway_elapsed_s = 0.0
            self.reference_profile = "BREAKAWAY"
        else:
            self.reference_profile = "STATIONARY"
            self.reference_breakaway_started_ms = None
            self.reference_breakaway_elapsed_s = 0.0

    def _derivative_brake_scale(self, error_cm, velocity_cm_s):
        moving_toward_reference = float(error_cm) * float(velocity_cm_s) < 0.0
        if not moving_toward_reference:
            return 1.0
        distance = abs(float(error_cm))
        if distance >= self.target_brake_start_cm:
            return self.target_brake_far_scale
        if distance <= self.target_brake_full_cm:
            return 1.0
        fraction = (
            (self.target_brake_start_cm - distance)
            / (self.target_brake_start_cm - self.target_brake_full_cm)
        )
        return self.target_brake_far_scale + (
            1.0 - self.target_brake_far_scale
        ) * fraction

    def _update_breakaway(
        self,
        source_error_px,
        error_cm,
        measurement_current,
        dt,
        now_ms,
    ):
        if not measurement_current:
            return self.reference_breakaway_deg
        active = (
            self.reference_profile == "BREAKAWAY"
            and abs(error_cm) > self.breakaway_error_min_cm
        )
        if active:
            if self.reference_breakaway_started_ms is None:
                self.reference_breakaway_started_ms = int(now_ms)
                self.reference_breakaway_elapsed_s = 0.0
            self.reference_breakaway_elapsed_s += max(0.0, float(dt))
            magnitude = min(
                self.breakaway_max_deg,
                self.breakaway_initial_deg
                + self.breakaway_escalation_deg_s
                * self.reference_breakaway_elapsed_s,
                abs(error_cm) * self.breakaway_error_gain_deg_per_cm,
            )
            desired = (1.0 if source_error_px > 0.0 else -1.0) * magnitude
            rate = self.breakaway_ramp_deg_s
        else:
            desired = 0.0
            rate = self.breakaway_release_deg_s
            self.reference_breakaway_started_ms = None
            self.reference_breakaway_elapsed_s = 0.0
        self.reference_breakaway_deg = _approach(
            self.reference_breakaway_deg,
            desired,
            rate * float(dt),
        )
        return self.reference_breakaway_deg

    def _update_trim(
        self,
        source_error_px,
        velocity_cm_s,
        measurement_current,
        dt,
        now_ms,
    ):
        if not self.trim_enabled or not measurement_current:
            return self.reference_trim_deg
        error_cm = -float(source_error_px) / self.source_pixels_per_cm
        magnitude = abs(error_cm)
        if magnitude <= self.trim_deadband_cm or self.reference_last_saturated:
            self.reference_trim_direction = 0
            self.reference_trim_since_ms = None
            return self.reference_trim_deg
        direction = 1 if source_error_px > 0.0 else -1
        if direction != self.reference_trim_direction:
            self.reference_trim_direction = direction
            self.reference_trim_since_ms = int(now_ms)
            return self.reference_trim_deg
        if self.reference_trim_since_ms is None:
            self.reference_trim_since_ms = int(now_ms)
            return self.reference_trim_deg
        if int(now_ms) - self.reference_trim_since_ms < self.trim_activation_ms:
            return self.reference_trim_deg
        speed = abs(float(velocity_cm_s))
        velocity_weight = _clamp(
            1.0 - speed / max(self.trim_velocity_gate_cm_s, 0.000001),
            0.0,
            1.0,
        )
        rate = min(
            self.trim_max_rate_deg_s,
            self.trim_gain_deg_per_cm_s
            * (magnitude - self.trim_deadband_cm),
        ) * velocity_weight
        self.reference_trim_deg = _clamp(
            self.reference_trim_deg + direction * rate * float(dt),
            -self.trim_max_deg,
            self.trim_max_deg,
        )
        return self.reference_trim_deg

    def _compute_unconstrained_feedback(
        self,
        control_error_cm,
        control_velocity_cm_s,
        feedback_position_cm,
        feedback_velocity_cm_s,
        measurement_current,
        settled,
        dt,
        now_ms,
    ):
        del feedback_velocity_cm_s, settled
        error_cm = float(control_error_cm)
        velocity_cm_s = float(control_velocity_cm_s)
        source_error_px = -error_cm * self.source_pixels_per_cm
        source_velocity_px_s = velocity_cm_s * self.source_pixels_per_cm
        control_error_px = (
            -float(feedback_position_cm) * self.source_pixels_per_cm
        )
        self.joint_deadband_active = bool(
            control_error_px == 0.0
            and abs(velocity_cm_s) <= self.stationary_velocity_cm_s
        )
        self._update_profile(error_cm, velocity_cm_s, measurement_current, now_ms)
        self.reference_brake_scale = self._derivative_brake_scale(
            error_cm,
            velocity_cm_s,
        )
        gain_scale = self.accel_per_degree_small * self.source_pixels_per_cm
        rolling_kp_deg_per_px = self.kp / gain_scale
        rolling_ki_deg_per_px_s = self.ki / gain_scale
        rolling_kd_deg_s_per_px = self.kd / gain_scale
        kp_deg_per_px = rolling_kp_deg_per_px
        allow_integral = bool(
            self.reference_profile == "ROLLING"
            and measurement_current
            and abs(error_cm) <= self.integral_active_error_cm
            and abs(velocity_cm_s) <= self.integral_active_velocity_cm_s
            and not self.reference_last_saturated
        )
        if self.reference_profile in ("STATIONARY", "BREAKAWAY"):
            kp_deg_per_px *= self.stationary_kp_ratio
            allow_integral = False
        if allow_integral:
            self.reference_integral_px_s = _clamp(
                self.reference_integral_px_s + control_error_px * float(dt),
                -self.integral_limit_px_s,
                self.integral_limit_px_s,
            )
            self.integral_state = "TRACK"
        elif measurement_current:
            self.integral_state = "GATE"
        else:
            self.integral_state = "HOLD"
        raw_p_deg = kp_deg_per_px * control_error_px
        p_deg = _clamp(
            raw_p_deg,
            -self.proportional_limit_deg,
            self.proportional_limit_deg,
        )
        i_deg = rolling_ki_deg_per_px_s * self.reference_integral_px_s
        d_deg = _clamp(
            rolling_kd_deg_s_per_px
            * (-source_velocity_px_s)
            * self.reference_brake_scale,
            -self.derivative_limit_deg,
            self.derivative_limit_deg,
        )
        trim_deg = self._update_trim(
            source_error_px,
            velocity_cm_s,
            measurement_current,
            dt,
            now_ms,
        )
        breakaway_deg = self._update_breakaway(
            source_error_px,
            error_cm,
            measurement_current,
            dt,
            now_ms,
        )
        raw_angle_deg = p_deg + i_deg + d_deg + trim_deg + breakaway_deg
        angle_deg = _clamp(
            raw_angle_deg,
            -self.dynamic_tilt_limit_deg,
            self.dynamic_tilt_limit_deg,
        )
        self.reference_last_saturated = bool(
            angle_deg != raw_angle_deg or p_deg != raw_p_deg
        )
        self.reference_correction_deg = angle_deg
        self.integral_correction_cm_s2 = self._angle_to_acceleration(
            i_deg + trim_deg
        )
        self.integral_learning = allow_integral
        self.integral_limited = self.reference_last_saturated
        return self._angle_to_acceleration(angle_deg)

    def _snapshot(self, *args, **kwargs):
        snapshot = super()._snapshot(*args, **kwargs)
        snapshot["feedback_saturated"] = bool(
            snapshot.get("feedback_saturated")
            or self.reference_last_saturated
        )
        snapshot["reference_profile"] = str(self.reference_profile)
        snapshot["reference_brake_scale"] = float(self.reference_brake_scale)
        snapshot["reference_correction_deg"] = float(
            self.reference_correction_deg
        )
        snapshot["reference_breakaway_deg"] = float(
            self.reference_breakaway_deg
        )
        snapshot["reference_trim_deg"] = float(self.reference_trim_deg)
        snapshot["reference_adaptation_frozen"] = bool(
            snapshot.get("feedback_valid")
            and snapshot.get("steady_correction_state") == "HOLD"
        )
        snapshot["source_pixels_per_cm"] = float(self.source_pixels_per_cm)
        return snapshot


class CascadePidFeedback(BallStateFeedback):
    """Position-P outer loop and velocity-PID inner loop in physical units."""

    backend_name = CASCADE_PID_BACKEND
    touch_gain_labels = ("Kx", "Kv", "Ki")

    def __init__(self, cascade_settings=None, **common_kwargs):
        settings = cascade_settings or {}
        for filter_name, default_value in (
            ("position_filter_tau_s", 0.015),
            ("velocity_filter_tau_s", 0.035),
            ("feedback_filter_tau_s", 0.010),
        ):
            common_kwargs[filter_name] = max(
                0.0,
                float(settings.get(filter_name, default_value)),
            )
        common_kwargs["kp"] = max(
            0.0,
            float(settings.get("position_kp_s_inv", 0.55)),
        )
        common_kwargs["kd"] = max(
            0.0,
            float(settings.get("velocity_kp_s_inv", 12.0)),
        )
        common_kwargs["ki"] = max(
            0.0,
            float(settings.get("velocity_ki_s2_inv", 18.0)),
        )
        common_kwargs["ka"] = max(
            0.0,
            float(settings.get("velocity_kd", 0.0)),
        )
        common_kwargs["pd_schedule_enabled"] = False
        common_kwargs["integral_limit_cm_s2"] = max(
            0.0,
            float(settings.get("velocity_integral_limit_cm_s2", 80.0)),
        )
        common_kwargs["integral_leak_per_s"] = max(
            0.0,
            float(settings.get("velocity_integral_leak_per_s", 0.05)),
        )
        super().__init__(**common_kwargs)

        self.max_reference_velocity_cm_s = max(
            0.1,
            float(settings.get("max_reference_velocity_cm_s", 6.0)),
        )
        self.velocity_integral_error_limit_cm_s = max(
            0.0,
            float(settings.get("velocity_integral_error_limit_cm_s", 4.0)),
        )
        self.velocity_integral_leak_per_s = self.integral_leak_per_s
        self.velocity_integral_unwind_scale = max(
            1.0,
            float(settings.get("velocity_integral_unwind_scale", 4.0)),
        )
        self.cascade_reference_velocity_cm_s = 0.0
        self.cascade_velocity_error_cm_s = 0.0
        self.cascade_position_limited = False
        self.cascade_inner_p_cm_s2 = 0.0
        self.cascade_inner_i_cm_s2 = 0.0
        self.cascade_inner_d_cm_s2 = 0.0
        self.cascade_integral_unwind_active = False

    def reset_runtime(self):
        super().reset_runtime()
        self.cascade_reference_velocity_cm_s = 0.0
        self.cascade_velocity_error_cm_s = 0.0
        self.cascade_position_limited = False
        self.cascade_inner_p_cm_s2 = 0.0
        self.cascade_inner_i_cm_s2 = 0.0
        self.cascade_inner_d_cm_s2 = 0.0
        self.cascade_integral_unwind_active = False

    def _update_velocity_integral(
        self,
        velocity_error_cm_s,
        position_error_active,
        dynamic_feedback_cm_s2,
        measurement_current,
        settled,
        dt,
    ):
        self.integral_learning = False
        self.integral_limited = False
        self.cascade_integral_unwind_active = False
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

        bounded_error = _clamp(
            float(velocity_error_cm_s),
            -self.velocity_integral_error_limit_cm_s,
            self.velocity_integral_error_limit_cm_s,
        )
        previous = self.integral_correction_cm_s2
        error_drive = self.ki * bounded_error
        unwinding = bool(previous * error_drive < 0.0)
        self.cascade_integral_unwind_active = unwinding
        if not position_error_active and not unwinding:
            self.integral_state = "CENTER"
            return
        if unwinding:
            error_drive *= self.velocity_integral_unwind_scale
        rate = error_drive - self.velocity_integral_leak_per_s * previous
        proposed = _clamp(
            previous + rate * float(dt),
            -self.integral_limit_cm_s2,
            self.integral_limit_cm_s2,
        )
        if unwinding and previous * proposed < 0.0:
            proposed = 0.0
        delta = proposed - previous
        proposed_feedback = dynamic_feedback_cm_s2 + proposed
        if (
            proposed_feedback > self.max_feedback_accel_cm_s2
            and delta > 0.0
        ) or (
            proposed_feedback < -self.max_feedback_accel_cm_s2
            and delta < 0.0
        ):
            self.integral_limited = True
            self.integral_state = "LIMIT"
            return
        self.integral_correction_cm_s2 = proposed
        self.integral_learning = proposed != previous
        self.integral_limited = bool(
            abs(proposed) >= self.integral_limit_cm_s2
        )
        if self.integral_limited:
            self.integral_state = "LIMIT"
        elif unwinding:
            self.integral_state = "UNWIND"
        else:
            self.integral_state = "TRACK"

    def _compute_unconstrained_feedback(
        self,
        control_error_cm,
        control_velocity_cm_s,
        feedback_position_cm,
        feedback_velocity_cm_s,
        measurement_current,
        settled,
        dt,
        now_ms,
    ):
        del control_error_cm, control_velocity_cm_s, now_ms
        requested_velocity = -self.kp * float(feedback_position_cm)
        self.cascade_reference_velocity_cm_s = _clamp(
            requested_velocity,
            -self.max_reference_velocity_cm_s,
            self.max_reference_velocity_cm_s,
        )
        self.cascade_position_limited = bool(
            self.cascade_reference_velocity_cm_s != requested_velocity
        )
        self.cascade_velocity_error_cm_s = (
            self.cascade_reference_velocity_cm_s
            - float(feedback_velocity_cm_s)
        )
        self.cascade_inner_p_cm_s2 = (
            self.kd * self.cascade_velocity_error_cm_s
        )
        self.cascade_inner_d_cm_s2 = -self.ka * self.acceleration_cm_s2
        dynamic_feedback = (
            self.cascade_inner_p_cm_s2 + self.cascade_inner_d_cm_s2
        )
        self._update_velocity_integral(
            self.cascade_velocity_error_cm_s,
            bool(feedback_position_cm != 0.0),
            dynamic_feedback,
            measurement_current,
            settled,
            dt,
        )
        self.cascade_inner_i_cm_s2 = self.integral_correction_cm_s2
        self.effective_kp = self.kp
        self.effective_kd = self.kd
        return dynamic_feedback + self.integral_correction_cm_s2

    def _snapshot(self, *args, **kwargs):
        snapshot = super()._snapshot(*args, **kwargs)
        snapshot["reference_velocity_cm_s"] = float(
            self.cascade_reference_velocity_cm_s
        )
        snapshot["cascade_velocity_error_cm_s"] = float(
            self.cascade_velocity_error_cm_s
        )
        snapshot["cascade_position_limited"] = bool(
            self.cascade_position_limited
        )
        snapshot["cascade_inner_p_cm_s2"] = float(
            self.cascade_inner_p_cm_s2
        )
        snapshot["cascade_inner_i_cm_s2"] = float(
            self.cascade_inner_i_cm_s2
        )
        snapshot["cascade_inner_d_cm_s2"] = float(
            self.cascade_inner_d_cm_s2
        )
        snapshot["cascade_max_reference_velocity_cm_s"] = float(
            self.max_reference_velocity_cm_s
        )
        snapshot["cascade_integral_unwind_active"] = bool(
            self.cascade_integral_unwind_active
        )
        return snapshot


def create_control_backend(
    name,
    common_settings,
    reference_settings=None,
    cascade_settings=None,
):
    backend = str(name).strip().lower()
    if backend == PID_BACKEND:
        return BallStateFeedback(**common_settings)
    if backend == REFERENCE_BACKEND:
        return ReferenceProfileFeedback(
            reference_settings=reference_settings,
            **common_settings
        )
    if backend == CASCADE_PID_BACKEND:
        return CascadePidFeedback(
            cascade_settings=cascade_settings,
            **common_settings
        )
    raise ValueError(
        "unsupported control backend {!r}; expected {}".format(
            name,
            ", ".join(SUPPORTED_CONTROL_BACKENDS),
        )
    )
