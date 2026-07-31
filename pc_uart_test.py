#!/usr/bin/env python3
"""Dual-mode PC/K230 diagnostic for the steel-ball UART stream.

Connect K230 IO9 (UART1 TX) to a 3.3 V USB-TTL adapter RX and connect GND.
On desktop CPython this is a receive-only parser.  Under CanMV/MicroPython it
launches the board application with a 2 Hz USB-console diagnostic mirror.
"""

import sys
import time


def _is_k230_runtime():
    implementation = getattr(sys, "implementation", None)
    if getattr(implementation, "name", "") == "micropython":
        return True
    try:
        from machine import FPIOA

        return FPIOA is not None
    except (ImportError, AttributeError):
        return False


def _is_ide_interrupt(exc):
    if isinstance(exc, KeyboardInterrupt):
        return True
    try:
        return "ide interrupt" in str(exc).strip().lower()
    except Exception:
        return False


K230_RUNTIME = _is_k230_runtime()

if not K230_RUNTIME:
    import argparse

    from ball_protocol import (
        BALL_FLAG_FEEDBACK_SATURATED,
        BALL_FLAG_FEEDBACK_VALID,
        BALL_FLAG_EDGE_CLIPPED,
        BALL_FLAG_MEASUREMENT_ACCEPTED,
        BALL_FLAG_OUTLIER_REJECTED,
        BALL_FLAG_PREDICTED,
        BALL_FLAG_RAW_DETECTED,
        BALL_FLAG_TRACK_VALID,
        BALL_REFERENCE_FLAG_SOURCE_ALGORITHM,
        BALL_REFERENCE_FLAG_SOURCE_MCU,
        BALL_REFERENCE_FLAG_SOURCE_PRESET,
        BALL_REFERENCE_FLAG_SLEW_ACTIVE,
        BALL_REFERENCE_FLAG_VALID,
        BALL_STATE_FRAME_SIZE,
        BALL_STATE_PAYLOAD_SIZE,
        REFERENCE_COMMAND_FRAME_SIZE,
        ReferenceCommandParser,
        UART_HEADER_SIZE,
        UART_SYNC_0,
        UART_SYNC_1,
        crc16_ccitt_false,
        decode_reference_command_frame,
        decode_ball_state_frame,
        encode_reference_command_frame,
        encode_ball_state_frame,
        validate_reference_command_frame,
        validate_ball_state_frame,
    )
else:
    # Desktop parser definitions are not executed on K230, but keeping their
    # constants defined avoids importing the protocol before the fatal guard.
    argparse = None
    BALL_FLAG_RAW_DETECTED = 0x01
    BALL_FLAG_MEASUREMENT_ACCEPTED = 0x02
    BALL_FLAG_TRACK_VALID = 0x04
    BALL_FLAG_PREDICTED = 0x08
    BALL_FLAG_FEEDBACK_VALID = 0x10
    BALL_FLAG_OUTLIER_REJECTED = 0x20
    BALL_FLAG_EDGE_CLIPPED = 0x40
    BALL_FLAG_FEEDBACK_SATURATED = 0x80
    BALL_REFERENCE_FLAG_VALID = 0x0001
    BALL_REFERENCE_FLAG_SOURCE_PRESET = 0x0002
    BALL_REFERENCE_FLAG_SOURCE_ALGORITHM = 0x0004
    BALL_REFERENCE_FLAG_SOURCE_MCU = 0x0008
    BALL_REFERENCE_FLAG_SLEW_ACTIVE = 0x0010
    BALL_STATE_FRAME_SIZE = 64
    BALL_STATE_PAYLOAD_SIZE = 48
    UART_HEADER_SIZE = 14
    UART_SYNC_0 = 0xA5
    UART_SYNC_1 = 0x5A


DEFAULT_BAUDRATE = 512000
SERIAL_READ_LIMIT = 4096
SYNC = bytes((UART_SYNC_0, UART_SYNC_1))

FLAG_NAMES = (
    (BALL_FLAG_RAW_DETECTED, "RAW"),
    (BALL_FLAG_MEASUREMENT_ACCEPTED, "MEAS"),
    (BALL_FLAG_TRACK_VALID, "TRACK"),
    (BALL_FLAG_PREDICTED, "PRED"),
    (BALL_FLAG_FEEDBACK_VALID, "FB"),
    (BALL_FLAG_OUTLIER_REJECTED, "OUTLIER"),
    (BALL_FLAG_EDGE_CLIPPED, "EDGE"),
    (BALL_FLAG_FEEDBACK_SATURATED, "FBSAT"),
)

REFERENCE_FLAG_NAMES = (
    (BALL_REFERENCE_FLAG_VALID, "VALID"),
    (BALL_REFERENCE_FLAG_SOURCE_PRESET, "PRESET"),
    (BALL_REFERENCE_FLAG_SOURCE_ALGORITHM, "ALGO"),
    (BALL_REFERENCE_FLAG_SOURCE_MCU, "MCU"),
    (BALL_REFERENCE_FLAG_SLEW_ACTIVE, "SLEW"),
)


def _read_u16(buffer, offset):
    return int(buffer[offset]) | (int(buffer[offset + 1]) << 8)


class BallStreamParser:
    """Bounded fixed-frame parser that tolerates noise and split reads."""

    def __init__(self):
        self.buffer = bytearray()
        self.bytes_received = 0
        self.valid_frames = 0
        self.invalid_candidates = 0
        self.crc_errors = 0
        self.header_errors = 0
        self.discarded_bytes = 0
        self.sequence_gaps = 0
        self.duplicate_frames = 0
        self.sequence_resets = 0
        self.last_sequence = None

    def _discard_prefix(self, count):
        count = max(0, min(int(count), len(self.buffer)))
        if count == 0:
            return
        self.buffer = self.buffer[count:]
        self.discarded_bytes += count

    def _seek_sync(self):
        index = self.buffer.find(SYNC)
        if index >= 0:
            self._discard_prefix(index)
            return True

        keep = 1 if self.buffer and self.buffer[-1] == UART_SYNC_0 else 0
        self._discard_prefix(len(self.buffer) - keep)
        return False

    def _track_sequence(self, sequence):
        sequence = int(sequence) & 0xFFFF
        if self.last_sequence is not None:
            delta = (sequence - self.last_sequence) & 0xFFFF
            if delta == 0:
                self.duplicate_frames += 1
            elif delta < 0x8000:
                self.sequence_gaps += max(0, delta - 1)
            else:
                # A backwards jump normally means the K230 application rebooted.
                self.sequence_resets += 1
        self.last_sequence = sequence

    def _classify_invalid(self, candidate):
        crc_offset = UART_HEADER_SIZE + BALL_STATE_PAYLOAD_SIZE
        received_crc = _read_u16(candidate, crc_offset)
        calculated_crc = crc16_ccitt_false(candidate, 2, crc_offset)
        if received_crc != calculated_crc:
            self.crc_errors += 1
        else:
            self.header_errors += 1

    def feed(self, data):
        if data:
            self.buffer.extend(data)
            self.bytes_received += len(data)

        frames = []
        while True:
            if not self._seek_sync():
                break
            if len(self.buffer) < BALL_STATE_FRAME_SIZE:
                break

            candidate = bytes(self.buffer[:BALL_STATE_FRAME_SIZE])
            if validate_ball_state_frame(candidate):
                self.buffer = self.buffer[BALL_STATE_FRAME_SIZE:]
                self.valid_frames += 1
                self._track_sequence(_read_u16(candidate, 8))
                frames.append(candidate)
                continue

            self.invalid_candidates += 1
            self._classify_invalid(candidate)
            # The candidate length may itself be corrupt; advance only one byte.
            self._discard_prefix(1)

        return frames


def _format_flags(flags):
    names = [name for mask, name in FLAG_NAMES if int(flags) & mask]
    return "|".join(names) if names else "NONE"


def _format_reference_flags(flags):
    names = [
        name for mask, name in REFERENCE_FLAG_NAMES
        if int(flags) & mask
    ]
    return "|".join(names) if names else "NONE"


def _format_age(age_ms):
    return "N/A" if int(age_ms) == 0xFFFF else "{}ms".format(int(age_ms))


def _format_tracking_mode(flags):
    flags = int(flags)
    if (
        flags & BALL_FLAG_MEASUREMENT_ACCEPTED
        and flags & BALL_FLAG_TRACK_VALID
        and flags & BALL_FLAG_FEEDBACK_VALID
        and not flags & BALL_FLAG_PREDICTED
    ):
        return "DETECT"
    if (
        flags & BALL_FLAG_PREDICTED
        and flags & BALL_FLAG_TRACK_VALID
        and flags & BALL_FLAG_FEEDBACK_VALID
    ):
        return "HOLD"
    return "LOST"


def print_state(state, raw_frame=None):
    print(
        "seq={:5d} k230_ms={:10d} mode={:<6s} flags={:<34s} conf={:.3f} "
        "miss={} det={}ms meas_age={}".format(
            state["sequence"],
            state["timestamp_ms"],
            _format_tracking_mode(state["flags"]),
            _format_flags(state["flags"]),
            state["confidence_x255"] / 255.0,
            state["missed_frames"],
            state["detector_ms"],
            _format_age(state["measurement_age_ms"]),
        )
    )
    print(
        "  raw_px=({:3d},{:3d}) phys=({:+7.2f},{:+7.2f})cm "
        "vel=({:+8.1f},{:+8.1f})cm/s".format(
            state["raw_x"],
            state["raw_y"],
            state["physical_x_cm"],
            state["physical_y_cm"],
            state["physical_vx_cm_s"],
            state["physical_vy_cm_s"],
        )
    )
    print(
        "  axial p={:+.2f}cm v={:+.1f}cm/s a={:+.1f}cm/s2 "
        "feedback={:+.1f}cm/s2 kp={:.2f} kd={:.2f} ka={:.2f}".format(
            state["position_cm"],
            state["velocity_cm_s"],
            state["acceleration_cm_s2"],
            state["feedback_acceleration_cm_s2"],
            state["kp"],
            state["kd"],
            state["ka"],
        )
    )
    consistency_text = "N/A"
    if int(state["flags"]) & BALL_FLAG_FEEDBACK_VALID:
        consistency_error = (
            state["position_cm"]
            - state["reference_position_cm"]
            - state["position_error_cm"]
        )
        consistency_text = "{:+.3f}cm".format(consistency_error)
    print(
        "  ref p={:+.2f}cm e={:+.2f}cm v={:+.1f}cm/s a={:+.1f}cm/s2 "
        "state={} check={}".format(
            state["reference_position_cm"],
            state["position_error_cm"],
            state["reference_velocity_cm_s"],
            state["reference_acceleration_cm_s2"],
            _format_reference_flags(state["reference_state_flags"]),
            consistency_text,
        )
    )
    if raw_frame is not None:
        print("  HEX " + " ".join("{:02X}".format(byte) for byte in raw_frame))


def print_statistics(parser, interval_frames, interval_seconds):
    frame_rate = interval_frames / max(0.001, interval_seconds)
    print(
        "[UART] bytes={} valid={} rate={:.1f}fps invalid={} crc={} header={} "
        "discard={} gaps={} dup={} reboot={}".format(
            parser.bytes_received,
            parser.valid_frames,
            frame_rate,
            parser.invalid_candidates,
            parser.crc_errors,
            parser.header_errors,
            parser.discarded_bytes,
            parser.sequence_gaps,
            parser.duplicate_frames,
            parser.sequence_resets,
        )
    )


def load_pyserial():
    try:
        import serial
    except ImportError:
        raise RuntimeError(
            "pyserial is not installed; run: python -m pip install pyserial"
        )
    return serial


def list_serial_ports():
    try:
        from serial.tools import list_ports
    except ImportError:
        raise RuntimeError(
            "pyserial is not installed; run: python -m pip install pyserial"
        )

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    for port in ports:
        print("{:<12s} {}".format(port.device, port.description))


def _tracking_candidate(
    center_x,
    center_y=160.0,
    ambiguous=False,
    quality=0.90,
    support=None,
    confidence=0.85,
):
    candidate = {
        "confidence": float(confidence),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "x": float(center_x) - 5.0,
        "y": float(center_y) - 5.0,
        "w": 10.0,
        "h": 10.0,
        "diameter": 10.0,
        "edge_clipped": False,
        "measurement_ambiguous": bool(ambiguous),
        "measurement_quality": float(quality),
    }
    if support is not None:
        candidate["support_count"] = int(support)
    return candidate


def _new_motion_filter(fixed_axis=True):
    from ball_tracking import PipeMotionFilter

    tracker = PipeMotionFilter(
        acquire_confirm_frames=1,
        fixed_axis_x=1.0 if fixed_axis else None,
        fixed_axis_y=0.0 if fixed_axis else None,
    )
    state = tracker.update(_tracking_candidate(160.0), 1000)
    assert state["track_valid"]
    assert state["filter_reset"]
    return tracker


def _associated_tracking_step(tracker, candidate, now_ms):
    selected = tracker.select_candidate([candidate], now_ms)
    assert selected is candidate
    return tracker.update(selected, now_ms)


def _run_motion_filter_self_test():
    tracker = _new_motion_filter()
    positions = []
    max_velocity = 0.0
    for index, center_x in enumerate((155.0, 165.0) * 6):
        state = tracker.update(
            _tracking_candidate(center_x),
            1033 + index * 33,
        )
        positions.append(state["x"])
        max_velocity = max(max_velocity, abs(state["vx"]))
    assert max_velocity < 1.0
    assert max(positions) - min(positions) < 2.0
    assert state["motion_state"] != "MOVING"

    tracker = _new_motion_filter()
    reliable_states = []
    for index, center_x in enumerate((162.0, 164.0)):
        reliable_states.append(tracker.update(
            _tracking_candidate(center_x),
            1033 + index * 33,
        ))
    assert reliable_states[0]["motion_state"] == "PENDING"
    assert reliable_states[1]["motion_state"] == "MOVING"
    assert reliable_states[1]["vx"] > 20.0

    tracker = _new_motion_filter()
    ambiguous_states = []
    for index, center_x in enumerate((162.0, 164.0, 166.0)):
        ambiguous_states.append(tracker.update(
            _tracking_candidate(center_x, ambiguous=True),
            1033 + index * 33,
        ))
    assert ambiguous_states[0]["motion_state"] == "PENDING"
    assert ambiguous_states[1]["motion_state"] == "PENDING"
    assert ambiguous_states[2]["motion_state"] == "MOVING"
    assert ambiguous_states[2]["vx"] > 20.0

    tracker = _new_motion_filter()
    ambiguous_jump = _associated_tracking_step(
        tracker,
        _tracking_candidate(180.0, ambiguous=True),
        1033,
    )
    assert ambiguous_jump["motion_state"] == "PENDING"
    assert abs(ambiguous_jump["vx"]) < 1.0
    ambiguous_fast = _associated_tracking_step(
        tracker,
        _tracking_candidate(195.0, ambiguous=True),
        1066,
    )
    assert ambiguous_fast["motion_state"] == "PENDING"
    assert abs(ambiguous_fast["vx"]) < 1.0
    ambiguous_confirmed = _associated_tracking_step(
        tracker,
        _tracking_candidate(210.0, ambiguous=True),
        1099,
    )
    assert ambiguous_confirmed["motion_state"] == "MOVING"
    assert not ambiguous_confirmed["filter_reset"]
    assert ambiguous_confirmed["vx"] > 100.0

    tracker = _new_motion_filter()
    _associated_tracking_step(
        tracker,
        _tracking_candidate(180.0, ambiguous=True),
        1033,
    )
    opposite_jump = _associated_tracking_step(
        tracker,
        _tracking_candidate(145.0, ambiguous=True),
        1066,
    )
    assert opposite_jump["motion_state"] == "PENDING"
    assert abs(opposite_jump["vx"]) < 1.0

    tracker = _new_motion_filter()
    fast_pending = tracker.update(_tracking_candidate(175.0), 1033)
    assert fast_pending["motion_state"] == "PENDING"
    assert abs(fast_pending["vx"]) < 1.0
    fast_state = tracker.update(_tracking_candidate(190.0), 1066)
    assert fast_state["motion_state"] == "MOVING"
    assert fast_state["vx"] > 100.0
    reverse_state = tracker.update(_tracking_candidate(160.0), 1099)
    assert reverse_state["motion_state"] == "MOVING"
    assert reverse_state["motion_direction"] == -1
    assert reverse_state["vx"] < -100.0

    tracker = _new_motion_filter()
    very_fast_pending = _associated_tracking_step(
        tracker,
        _tracking_candidate(190.0, support=3),
        1033,
    )
    assert very_fast_pending["motion_state"] == "PENDING"
    very_fast_state = _associated_tracking_step(
        tracker,
        _tracking_candidate(220.0, support=3),
        1066,
    )
    assert very_fast_state["motion_state"] == "MOVING"
    assert very_fast_state["vx"] > 300.0

    tracker = _new_motion_filter()
    false_jump = _associated_tracking_step(
        tracker,
        _tracking_candidate(180.0, support=3),
        1033,
    )
    assert false_jump["motion_state"] == "PENDING"
    assert false_jump["association_wide"]
    assert abs(false_jump["vx"]) < 1.0
    return_state = _associated_tracking_step(
        tracker,
        _tracking_candidate(160.0, support=3),
        1066,
    )
    assert return_state["motion_state"] == "PENDING"
    assert abs(return_state["vx"]) < 1.0
    quiet_state = _associated_tracking_step(
        tracker,
        _tracking_candidate(160.0, support=3),
        1099,
    )
    assert quiet_state["motion_state"] == "QUIET"
    assert abs(quiet_state["x"] - 160.0) < 2.0

    tracker = _new_motion_filter()
    low_support_states = []
    for index, center_x in enumerate((175.0, 190.0, 205.0)):
        low_support_states.append(_associated_tracking_step(
            tracker,
            _tracking_candidate(center_x, support=1),
            1033 + index * 33,
        ))
    assert low_support_states[0]["motion_state"] == "PENDING"
    assert low_support_states[1]["motion_state"] == "PENDING"
    assert low_support_states[2]["motion_state"] == "MOVING"
    assert low_support_states[2]["measurement_ambiguous"]

    tracker = _new_motion_filter()
    near_candidate = _tracking_candidate(162.0, support=3)
    wide_candidate = _tracking_candidate(
        180.0,
        support=3,
        confidence=0.99,
    )
    selected = tracker.select_candidate(
        [wide_candidate, near_candidate],
        1033,
    )
    assert selected is near_candidate

    tracker = _new_motion_filter()
    weak_near_candidate = _tracking_candidate(
        162.0,
        support=1,
        quality=0.50,
    )
    strong_wide_candidate = _tracking_candidate(
        180.0,
        support=3,
        quality=0.90,
    )
    selected = tracker.select_candidate(
        [weak_near_candidate, strong_wide_candidate],
        1033,
    )
    assert selected is strong_wide_candidate

    tracker = _new_motion_filter()
    _associated_tracking_step(
        tracker,
        _tracking_candidate(180.0, support=3),
        1033,
    )
    stationary_candidate = _tracking_candidate(160.0, support=3)
    continuing_false_candidate = _tracking_candidate(195.0, support=3)
    selected = tracker.select_candidate(
        [continuing_false_candidate, stationary_candidate],
        1066,
    )
    assert selected is stationary_candidate

    tracker = _new_motion_filter()
    closest_candidate = _tracking_candidate(164.0, support=3)
    nearby_competitor = _tracking_candidate(170.0, support=3)
    selected = tracker.select_candidate(
        [nearby_competitor, closest_candidate],
        1033,
    )
    assert selected is closest_candidate
    assert selected["association_competing"]
    competing_state = tracker.update(selected, 1033)
    assert competing_state["measurement_ambiguous"]

    tracker = _new_motion_filter()
    low_quality_states = []
    for index, center_x in enumerate((162.0, 164.0, 166.0)):
        low_quality_states.append(tracker.update(
            _tracking_candidate(
                center_x,
                quality=0.50,
                support=3,
            ),
            1033 + index * 33,
        ))
    assert low_quality_states[0]["motion_state"] == "PENDING"
    assert low_quality_states[1]["motion_state"] == "PENDING"
    assert low_quality_states[2]["motion_state"] == "MOVING"

    tracker = _new_motion_filter()
    pending_state = tracker.update(_tracking_candidate(175.0), 1033)
    assert pending_state["motion_state"] == "PENDING"
    moving_state = tracker.update(_tracking_candidate(190.0), 1066)
    assert moving_state["motion_state"] == "MOVING"
    stop_states = []
    for index in range(4):
        stop_states.append(tracker.update(
            _tracking_candidate(190.0),
            1099 + index * 33,
        ))
    assert all(
        state["motion_state"] == "MOVING"
        for state in stop_states[:3]
    )
    assert stop_states[3]["motion_state"] == "QUIET"

    tracker = _new_motion_filter(fixed_axis=False)
    tracker.update(_tracking_candidate(166.0, 164.0), 1033)
    learned_state = tracker.update(
        _tracking_candidate(172.0, 168.0),
        1066,
    )
    assert learned_state["axis_locked"]
    assert learned_state["motion_state"] == "MOVING"
    assert learned_state["vx"] > 0.0
    assert learned_state["vy"] > 0.0


def _run_hough_fusion_self_test():
    import ball_edge_hough_detector as hough_module

    original_cv_lite = hough_module.cv_lite
    hough_module.cv_lite = object()
    try:
        detector = hough_module.EdgeHoughCircleDetector()
        assert detector.min_center_distance_px == 8
        assert detector.min_radius_px == 3
        assert detector.max_radius_px == 9
        assert detector.cluster_center_factor == 1.0
        assert detector.cluster_radius_tolerance_px == 2.0
        circles = (
            (155.0, 160.0, 5.0),
            (165.0, 160.5, 5.5),
            (159.0, 159.5, 5.2),
            (162.0, 160.0, 5.4),
            (205.0, 160.0, 5.5),
        )
        candidates = []
        for source_index, circle in enumerate(circles):
            candidate = detector._candidate(
                circle[0],
                circle[1],
                circle[2],
                None,
                source_index,
            )
            assert candidate is not None
            candidates.append(candidate)
        clusters = detector._cluster_candidates(candidates)
        assert len(clusters) == 2
        fused = []
        for cluster_index, cluster in enumerate(clusters):
            fused.append(detector._fuse_cluster(
                cluster,
                None,
                cluster_index,
            ))
        fused.sort(
            key=lambda item: item["acquisition_score"],
            reverse=True,
        )
        ball = fused[0]
        assert ball["support_count"] == 4
        assert abs(ball["center_x"] - 160.5) < 0.0001
        assert abs(ball["radius"] - 5.3) < 0.0001
        assert ball["measurement_ambiguous"]
        assert ball["acquisition_score"] > fused[1]["acquisition_score"]
    finally:
        hough_module.cv_lite = original_cv_lite


def _run_pc_debug_format_self_test():
    import contextlib
    import io
    import main as tracker_main

    class Stub:
        pass

    detector = Stub()
    detector.model_backend = "edge_hough"
    detector.timing_labels = ("crop", "hough", "select")
    detector.preprocess_ms = 1
    detector.kpu_ms = 2
    detector.postprocess_ms = 0
    detector.diagnostic_text = lambda candidate=None: "H2/3/8 S2 Q80"

    camera = Stub()
    camera.exposure_text = lambda: "auto"

    tx = Stub()
    tx.sent_frames = 1
    tx.bytes_sent = 64
    tx.queued_bytes = lambda: 0
    tx.write_errors = 0
    tx.short_writes = 0
    tx.zero_writes = 0
    tx.dropped_error_frames = 0
    tx.last_error = None

    rx = Stub()
    rx.parser = Stub()
    rx.parser.bytes_received = 10
    rx.parser.invalid_frames = 0
    rx.applied_commands = 1
    rx.read_errors = 0
    rx.last_applied_cm = 1.25
    rx.last_error = None

    wifi_video = Stub()
    wifi_video.diagnostic_text = lambda: "off"

    candidate = _tracking_candidate(170.0, support=1, quality=0.50)
    state = {
        "x": 161.0,
        "y": 160.0,
        "vx": 0.0,
        "vy": 0.0,
        "track_valid": True,
        "measurement_accepted": True,
        "predicted": False,
        "motion_state": "PENDING",
        "motion_confirm_count": 1,
        "motion_step_px": 10.0,
        "measurement_ambiguous": True,
        "measurement_quality": 0.50,
        "measurement_support": 1,
        "association_wide": True,
        "association_competing": True,
    }
    feedback = {
        "feedback_valid": True,
        "feedback_saturated": False,
    }
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        tracker_main._print_pc_debug_status(
            1,
            candidate,
            state,
            feedback,
            3,
            10,
            detector,
            camera,
            tx,
            rx,
            wifi_video,
            None,
            None,
            30.0,
            33.3,
        )
    text = output.getvalue()
    assert "obs[m=PENDING1AWC" in text
    assert "q=0.50,s=1" in text


def run_self_test():
    from ball_calibration import (
        X_CALIBRATION_MEASURED,
        X_CALIBRATION_PINHOLE,
        X_POSITION_CM_MEASURED_X10000,
        X_POSITION_CM_X10000,
        Y_CM_PER_PIXEL,
        get_x_calibration_lut,
        x_center_scale,
        x_pixel_for_position_cm,
        x_position_and_scale,
        y_position_cm,
    )

    pinhole_lut = get_x_calibration_lut(X_CALIBRATION_PINHOLE)
    measured_lut = get_x_calibration_lut(X_CALIBRATION_MEASURED)
    assert pinhole_lut is X_POSITION_CM_X10000
    assert measured_lut is X_POSITION_CM_MEASURED_X10000
    for calibration_lut in (pinhole_lut, measured_lut):
        assert len(calibration_lut) == 320
        assert all(
            calibration_lut[index] < calibration_lut[index + 1]
            for index in range(319)
        )
        for pixel_x in (
            0.0, 1.0, 80.25, 160.0, 240.75, 318.0, 319.0
        ):
            position_cm = x_position_and_scale(
                pixel_x,
                calibration_lut,
            )[0]
            assert abs(
                x_pixel_for_position_cm(position_cm, calibration_lut)
                - pixel_x
            ) < 0.0001
    assert x_position_and_scale(160.0)[0] == 0.0
    assert abs(x_position_and_scale(160.5)[0] - 0.03255) < 0.0001
    assert x_pixel_for_position_cm(-100.0) == 0.0
    assert x_pixel_for_position_cm(100.0) == 319.0
    nominal_marks_cm = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5)
    measured_marks_cm = (0.0, 3.0, 5.8, 8.5, 10.7, 12.5)
    for nominal_cm, measured_cm in zip(
        nominal_marks_cm,
        measured_marks_cm,
    ):
        old_mark_pixel = x_pixel_for_position_cm(
            nominal_cm,
            pinhole_lut,
        )
        corrected_cm = x_position_and_scale(
            old_mark_pixel,
            measured_lut,
        )[0]
        assert abs(corrected_cm - measured_cm) < 0.0002
        if nominal_cm:
            negative_pixel = x_pixel_for_position_cm(
                -nominal_cm,
                pinhole_lut,
            )
            corrected_negative_cm = x_position_and_scale(
                negative_pixel,
                measured_lut,
            )[0]
            assert abs(corrected_negative_cm + measured_cm) < 0.0002
    assert x_center_scale(pinhole_lut) == Y_CM_PER_PIXEL
    assert abs(x_center_scale(measured_lut) - 0.0789) < 0.000000001
    assert y_position_cm(160.0) == 0.0
    assert abs(y_position_cm(161.0) - Y_CM_PER_PIXEL) < 0.000000001

    reference_positive = encode_reference_command_frame(2.50)
    reference_negative = encode_reference_command_frame(-12.50)
    reference_zero = encode_reference_command_frame(0.0)
    assert len(reference_positive) == REFERENCE_COMMAND_FRAME_SIZE == 10
    assert validate_reference_command_frame(reference_positive)
    assert validate_reference_command_frame(reference_negative)
    assert validate_reference_command_frame(reference_zero)
    assert decode_reference_command_frame(reference_positive) == 2.50
    assert decode_reference_command_frame(reference_negative) == -12.50
    assert decode_reference_command_frame(reference_zero) == 0.0
    try:
        encode_reference_command_frame(12.51)
        raise AssertionError("out-of-range reference command must fail")
    except ValueError:
        pass
    corrupted_reference = bytearray(reference_positive)
    corrupted_reference[6] ^= 0x01
    assert not validate_reference_command_frame(corrupted_reference)
    overflow_reference = bytearray(reference_positive)
    overflow_reference[6] = 0xE3
    overflow_reference[7] = 0x04
    overflow_crc = crc16_ccitt_false(overflow_reference, 2, 8)
    overflow_reference[8] = overflow_crc & 0xFF
    overflow_reference[9] = (overflow_crc >> 8) & 0xFF
    assert not validate_reference_command_frame(overflow_reference)
    reference_parser = ReferenceCommandParser()
    assert reference_parser.feed(b"\x00\xA5" + reference_positive[:3]) is None
    assert reference_parser.feed(reference_positive[3:]) == 2.50
    latest_reference = reference_parser.feed(
        corrupted_reference + reference_negative + reference_zero
    )
    assert latest_reference == 0.0
    assert reference_parser.valid_frames == 3
    assert reference_parser.invalid_frames >= 1

    import main as tracker_main

    class FakeReferenceController:
        def __init__(self):
            self.reference_position_cm = 0.0
            self.reference_state_flags = 0

        def set_reference_position_cm(self, value, flags):
            self.reference_position_cm = float(value)
            self.reference_state_flags = int(flags)
            return self.reference_position_cm

    class FakeReferenceUart:
        def __init__(self, data):
            self.data = bytes(data)

        def any(self):
            return len(self.data)

        def read(self, size):
            chunk = self.data[:int(size)]
            self.data = self.data[int(size):]
            return chunk

    tracker_main.ReferenceCommandParser = ReferenceCommandParser
    fake_controller = FakeReferenceController()
    command_rx = tracker_main.UartReferenceCommandRx(
        FakeReferenceUart(reference_positive + reference_negative),
        fake_controller,
    )
    assert command_rx.poll() == -12.50
    assert fake_controller.reference_position_cm == -12.50
    assert fake_controller.reference_state_flags == (
        BALL_REFERENCE_FLAG_VALID | BALL_REFERENCE_FLAG_SOURCE_MCU
    )
    assert command_rx.applied_commands == 1

    _run_hough_fusion_self_test()
    _run_motion_filter_self_test()
    _run_pc_debug_format_self_test()

    from ball_control import BallStateFeedback

    controller = BallStateFeedback(x_calibration_mode=X_CALIBRATION_MEASURED)
    assert (controller.kp, controller.kd, controller.ki, controller.ka) == (
        8.0, 12.0, 12.0, 0.0
    )
    assert controller.position_deadband_cm == 0.10
    assert controller.velocity_deadband_cm_s == 0.35
    assert controller.integral_limit_cm_s2 == 80.0
    assert controller.integral_leak_per_s == 0.0
    assert controller.integral_unwind_gain_s2 == 20.0
    assert controller.breakaway_accel_cm_s2 == 20.0
    assert controller.breakaway_transition_cm == 0.20
    assert controller.set_gains(kp=8.1, kd=12.1, ki=12.1)[:3] == (
        8.1, 12.1, 12.1
    )
    try:
        controller.set_gains(kp=-0.1)
        raise AssertionError("negative gain must fail")
    except ValueError:
        pass

    reference_candidate = {"edge_clipped": False}
    settled_state = {
        "x": x_pixel_for_position_cm(0.05, measured_lut),
        "y": 160.0,
        "vx": 0.1 / x_center_scale(measured_lut),
        "vy": 0.0,
        "track_valid": True,
        "measurement_accepted": True,
        "predicted": False,
        "missed_frames": 0,
        "measurement_age_ms": 0,
        "motion_state": "QUIET",
    }
    settled_feedback = controller.update(
        settled_state, reference_candidate, 1000
    )
    assert settled_feedback["joint_deadband_active"]
    assert settled_feedback["feedback_acceleration_cm_s2"] == 0.0

    active_state = dict(settled_state)
    active_state["x"] = x_pixel_for_position_cm(1.0, measured_lut)
    active_feedback = controller.update(active_state, reference_candidate, 1033)
    assert not active_feedback["joint_deadband_active"]
    assert active_feedback["feedback_acceleration_cm_s2"] < 0.0

    breakaway_controller = BallStateFeedback(
        kp=0.0,
        kd=0.0,
        ki=0.0,
        ka=0.0,
        position_deadband_cm=0.10,
        velocity_deadband_cm_s=0.0,
        position_filter_tau_s=0.0,
        velocity_filter_tau_s=0.0,
        feedback_filter_tau_s=0.0,
        breakaway_accel_cm_s2=20.0,
        breakaway_transition_cm=0.20,
        x_calibration_mode=X_CALIBRATION_MEASURED,
    )
    breakaway_state = dict(settled_state)
    breakaway_state["vx"] = 0.0
    breakaway_state["x"] = x_pixel_for_position_cm(0.20, measured_lut)
    breakaway_feedback = breakaway_controller.update(
        breakaway_state,
        reference_candidate,
        2000,
    )
    assert abs(breakaway_feedback["breakaway_feedback_cm_s2"] + 10.0) < 0.01
    assert abs(breakaway_feedback["feedback_acceleration_cm_s2"] + 10.0) < 0.01
    breakaway_state["x"] = x_pixel_for_position_cm(-0.50, measured_lut)
    breakaway_feedback = breakaway_controller.update(
        breakaway_state,
        reference_candidate,
        2033,
    )
    assert abs(breakaway_feedback["breakaway_feedback_cm_s2"] - 20.0) < 0.01

    unwind_controller = BallStateFeedback(
        kp=0.0,
        kd=0.0,
        ki=12.0,
        ka=0.0,
        position_deadband_cm=0.0,
        velocity_deadband_cm_s=0.0,
        position_filter_tau_s=0.0,
        velocity_filter_tau_s=0.0,
        feedback_filter_tau_s=0.0,
        integral_unwind_gain_s2=20.0,
        breakaway_accel_cm_s2=0.0,
        x_calibration_mode=X_CALIBRATION_MEASURED,
    )
    unwind_controller.integral_correction_cm_s2 = -40.0
    unwind_state = dict(settled_state)
    unwind_state["x"] = x_pixel_for_position_cm(2.0, measured_lut)
    unwind_scale = x_position_and_scale(
        unwind_state["x"], measured_lut
    )[1]
    unwind_state["vx"] = -5.0 / unwind_scale
    unwind_feedback = unwind_controller.update(
        unwind_state,
        reference_candidate,
        3000,
    )
    assert abs(unwind_feedback["steady_correction_cm_s2"] + 36.5) < 0.01
    assert unwind_feedback["steady_correction_unwind_active"]
    assert unwind_feedback["steady_correction_state"] == "UNWIND"
    unwind_controller.integral_correction_cm_s2 = -0.5
    unwind_feedback = unwind_controller.update(
        unwind_state,
        reference_candidate,
        3033,
    )
    assert unwind_feedback["steady_correction_cm_s2"] == 0.0

    candidate = {
        "confidence": 0.92,
        "center_x": 123,
        "center_y": 68,
        "x": 110,
        "y": 55,
        "w": 27,
        "h": 26,
    }
    state = {
        "x": 123.45,
        "y": 67.89,
        "vx": -45.6,
        "vy": 7.8,
        "missed_frames": 0,
        "measurement_age_ms": 20,
    }
    feedback = {
        "position_cm": -3.25,
        "reference_position_cm": 1.25,
        "position_error_cm": -4.50,
        "reference_velocity_cm_s": 0.0,
        "reference_acceleration_cm_s2": 0.0,
        "reference_state_flags": (
            BALL_REFERENCE_FLAG_VALID
            | BALL_REFERENCE_FLAG_SOURCE_PRESET
        ),
        "position_y_cm": 0.75,
        "velocity_cm_s": 12.3,
        "velocity_y_cm_s": -1.4,
        "acceleration_cm_s2": 18.4,
        "feedback_acceleration_cm_s2": -36.6,
        "kp": 16.0,
        "kd": 7.2,
        "ka": 0.05,
    }
    flags = (
        BALL_FLAG_RAW_DETECTED
        | BALL_FLAG_MEASUREMENT_ACCEPTED
        | BALL_FLAG_TRACK_VALID
        | BALL_FLAG_FEEDBACK_VALID
    )
    first = encode_ball_state_frame(
        7, 1000, flags, state, candidate, 17, feedback
    )
    second = encode_ball_state_frame(
        8, 1033, flags, state, candidate, 18, feedback
    )
    assert len(first) == 64
    assert first[4] == 6
    assert _read_u16(first, 6) == 48
    assert first[56:62] == b"\x00\x00\x00\x00\x00\x00"
    schema5_candidate = bytearray(first)
    schema5_candidate[4] = 5
    crc_offset = UART_HEADER_SIZE + BALL_STATE_PAYLOAD_SIZE
    schema5_crc = crc16_ccitt_false(schema5_candidate, 2, crc_offset)
    schema5_candidate[crc_offset] = schema5_crc & 0xFF
    schema5_candidate[crc_offset + 1] = (schema5_crc >> 8) & 0xFF
    assert not validate_ball_state_frame(schema5_candidate)
    assert not validate_ball_state_frame(first[:48])
    reserved_candidate = bytearray(first)
    reserved_candidate[56] = 1
    reserved_crc = crc16_ccitt_false(reserved_candidate, 2, crc_offset)
    reserved_candidate[crc_offset] = reserved_crc & 0xFF
    reserved_candidate[crc_offset + 1] = (reserved_crc >> 8) & 0xFF
    assert not validate_ball_state_frame(reserved_candidate)
    multiple_sources_candidate = bytearray(first)
    multiple_sources_candidate[UART_HEADER_SIZE + 40] = (
        BALL_REFERENCE_FLAG_VALID
        | BALL_REFERENCE_FLAG_SOURCE_PRESET
        | BALL_REFERENCE_FLAG_SOURCE_MCU
    )
    multiple_sources_candidate[UART_HEADER_SIZE + 41] = 0
    multiple_sources_crc = crc16_ccitt_false(
        multiple_sources_candidate,
        2,
        crc_offset,
    )
    multiple_sources_candidate[crc_offset] = multiple_sources_crc & 0xFF
    multiple_sources_candidate[crc_offset + 1] = (
        multiple_sources_crc >> 8
    ) & 0xFF
    assert not validate_ball_state_frame(multiple_sources_candidate)
    unknown_reference_flag_candidate = bytearray(first)
    unknown_reference_flag_candidate[UART_HEADER_SIZE + 40] |= 0x20
    unknown_reference_flag_crc = crc16_ccitt_false(
        unknown_reference_flag_candidate,
        2,
        crc_offset,
    )
    unknown_reference_flag_candidate[crc_offset] = (
        unknown_reference_flag_crc & 0xFF
    )
    unknown_reference_flag_candidate[crc_offset + 1] = (
        unknown_reference_flag_crc >> 8
    ) & 0xFF
    assert not validate_ball_state_frame(unknown_reference_flag_candidate)

    parser = BallStreamParser()
    decoded_frames = []
    for chunk in (
        b"\x00\x7E" + first[:9],
        first[9:31],
        first[31:] + second[:5],
        second[5:],
    ):
        decoded_frames.extend(parser.feed(chunk))
    assert [decode_ball_state_frame(frame)["sequence"] for frame in decoded_frames] == [
        7,
        8,
    ]
    assert parser.sequence_gaps == 0
    assert parser.discarded_bytes == 2
    decoded = decode_ball_state_frame(decoded_frames[0])
    assert decoded["position_cm"] == -3.25
    assert decoded["physical_x_cm"] == -3.25
    assert decoded["physical_y_cm"] == 0.75
    assert decoded["velocity_cm_s"] == 12.3
    assert decoded["physical_vx_cm_s"] == 12.3
    assert decoded["physical_vy_cm_s"] == -1.4
    assert decoded["acceleration_cm_s2"] == 18.4
    assert decoded["feedback_acceleration_cm_s2"] == -36.6
    assert decoded["reference_position_cm"] == 1.25
    assert decoded["position_error_cm"] == -4.5
    assert decoded["reference_velocity_cm_s"] == 0.0
    assert decoded["reference_acceleration_cm_s2"] == 0.0
    assert decoded["reference_state_flags"] == (
        BALL_REFERENCE_FLAG_VALID | BALL_REFERENCE_FLAG_SOURCE_PRESET
    )
    assert decoded["reserved_extension"] == b"\x00" * 6
    assert decoded["kp"] == 16.0
    assert decoded["kd"] == 7.2
    assert decoded["ka"] == 0.05

    corrupted = bytearray(first)
    corrupted[20] ^= 0x01
    recovery = BallStreamParser()
    recovered = recovery.feed(corrupted + second)
    assert len(recovered) == 1
    assert decode_ball_state_frame(recovered[0])["sequence"] == 8
    assert recovery.crc_errors >= 1
    print("PC UART parser self-test: PASS")


def receive(args):
    serial = load_pyserial()
    try:
        port = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
    except Exception as exc:
        print("Failed to open {} at {} baud: {}".format(args.port, args.baud, exc))
        return 2

    parser = BallStreamParser()
    latest_state = None
    latest_frame = None
    started = time.monotonic()
    last_report = started
    last_report_count = 0
    interrupted = False

    try:
        try:
            port.dtr = False
            port.rts = False
            port.reset_input_buffer()
        except Exception:
            pass

        print(
            "Listening on {} at {}-8N1. Press Ctrl+C to stop.".format(
                args.port,
                args.baud,
            )
        )
        while True:
            now = time.monotonic()
            if args.duration > 0.0 and now - started >= args.duration:
                break

            try:
                waiting = int(getattr(port, "in_waiting", 0))
                read_size = min(SERIAL_READ_LIMIT, max(1, waiting))
                data = port.read(read_size)
            except Exception as exc:
                print("Serial read failed: {}".format(exc))
                return 3

            for frame in parser.feed(data):
                latest_frame = frame
                latest_state = decode_ball_state_frame(frame)
                if args.every_frame:
                    print_state(
                        latest_state,
                        raw_frame=frame if args.raw else None,
                    )

            now = time.monotonic()
            if now - last_report >= args.interval:
                interval_frames = parser.valid_frames - last_report_count
                print_statistics(parser, interval_frames, now - last_report)
                if latest_state is None:
                    print(
                        "  WAITING: no valid frame; check IO9->adapter RX, GND, "
                        "3.3 V TTL and 512000 baud."
                    )
                elif not args.every_frame:
                    print_state(
                        latest_state,
                        raw_frame=latest_frame if args.raw else None,
                    )
                last_report = now
                last_report_count = parser.valid_frames
    except KeyboardInterrupt:
        interrupted = True
    finally:
        port.close()

    if interrupted:
        print("Stopped by user.")
    print_statistics(parser, parser.valid_frames, max(0.001, time.monotonic() - started))
    return 0 if parser.valid_frames > 0 else 4


def _k230_hold_fatal(stage, exc):
    """Last-resort LCD page if importing or dispatching main.py itself fails."""
    print(
        "K230 PC diagnostic fatal at {}: {}: {}".format(
            stage,
            type(exc).__name__,
            exc,
        )
    )
    printer = getattr(sys, "print_exception", None)
    if printer is not None:
        try:
            printer(exc)
        except Exception as print_exc:
            print("exception printer warning:", print_exc)

    emergency_display = None
    emergency_media = None
    display_initialized = False
    media_initialized = False
    stop_requested = False
    try:
        import image as emergency_image
        from media.display import Display as emergency_display
        from media.media import MediaManager as emergency_media

        width = 640
        height = 480
        canvas = emergency_image.Image(width, height, emergency_image.ARGB8888)
        emergency_display.init(
            emergency_display.ST7701,
            width=width,
            height=height,
            osd_num=1,
            to_ide=True,
        )
        display_initialized = True
        try:
            emergency_media.init()
            media_initialized = True
        except Exception as media_exc:
            if _is_ide_interrupt(media_exc):
                raise
            print("emergency MediaManager init warning:", media_exc)

        canvas.draw_rectangle(
            0,
            0,
            width,
            height,
            color=(255, 0, 0, 0),
            fill=True,
        )

        def draw_line(row, text, color=(255, 0, 255, 255), size=16):
            canvas.draw_string_advanced(
                5,
                5 + int(row) * 28,
                size,
                str(text),
                color=color,
            )

        message = str(exc)
        stage_text = str(stage)
        draw_line(0, "K230 PC DIAGNOSTIC FATAL", (255, 255, 40, 40), 22)
        draw_line(2, "stage: " + stage_text[:58], (255, 255, 220, 0))
        draw_line(3, "       " + stage_text[58:116], (255, 255, 220, 0))
        draw_line(4, "type: " + type(exc).__name__, (255, 255, 40, 40))
        draw_line(5, "error: " + message[:58], (255, 255, 40, 40))
        draw_line(6, "       " + message[58:116], (255, 255, 40, 40))
        draw_line(7, "       " + message[116:174], (255, 255, 40, 40))
        draw_line(11, "Keep /sdcard/main.py as the tracker entry.")
        draw_line(12, "This page is latched; stop manually.")
        emergency_display.show_image(
            canvas,
            0,
            0,
            emergency_display.LAYER_OSD3,
        )
    except Exception as screen_exc:
        if _is_ide_interrupt(screen_exc):
            stop_requested = True
        else:
            print(
                "fatal screen unavailable: {}: {}".format(
                    type(screen_exc).__name__,
                    screen_exc,
                )
            )

    try:
        import os

        while not stop_requested:
            try:
                exitpoint = getattr(os, "exitpoint", None)
                if exitpoint is not None:
                    exitpoint()
            except KeyboardInterrupt:
                break
            except Exception as exit_exc:
                if _is_ide_interrupt(exit_exc):
                    break
                raise
            try:
                if hasattr(time, "sleep_ms"):
                    time.sleep_ms(100)
                else:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                break
            except Exception as sleep_exc:
                if _is_ide_interrupt(sleep_exc):
                    break
                raise
    finally:
        if display_initialized and emergency_display is not None:
            try:
                emergency_display.deinit()
            except Exception as cleanup_exc:
                print("emergency display deinit warning:", cleanup_exc)
        if media_initialized and emergency_media is not None:
            try:
                emergency_media.deinit()
            except Exception as cleanup_exc:
                print("emergency media deinit warning:", cleanup_exc)


def run_k230_debug():
    try:
        import main as tracker_main

        runner = getattr(tracker_main, "run", None)
        if runner is None:
            raise RuntimeError(
                "/sdcard/main.py is not the proj2 tracker (run() missing)"
            )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        if _is_ide_interrupt(exc):
            return 0
        _k230_hold_fatal("import_tracker_main", exc)
        return 1

    try:
        runner(pc_debug=True)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        if _is_ide_interrupt(exc):
            return 0
        emergency = getattr(tracker_main, "_emergency_hold_fatal", None)
        if emergency is not None:
            emergency("pc_uart_test/run_guard", exc)
        else:
            _k230_hold_fatal("tracker_run_guard", exc)
        return 1
    return 0


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Receive and decode K230 proj2 64-byte steel-ball frames."
    )
    parser.add_argument("--port", help="serial port, for example COM7 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--list", action="store_true", help="list serial ports and exit")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run parser/CRC recovery checks without serial hardware",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="summary interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after N seconds; zero means run until Ctrl+C",
    )
    parser.add_argument(
        "--every-frame",
        action="store_true",
        help="print every valid frame instead of only the latest summary",
    )
    parser.add_argument("--raw", action="store_true", help="also print frame hex")
    args = parser.parse_args(argv)
    args.interval = max(0.1, float(args.interval))
    args.duration = max(0.0, float(args.duration))
    if not args.list and not args.self_test and not args.port:
        parser.error("--port is required unless --list or --self-test is used")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    try:
        if args.self_test:
            run_self_test()
            return 0
        if args.list:
            list_serial_ports()
            return 0
        return receive(args)
    except RuntimeError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    if K230_RUNTIME:
        run_k230_debug()
    else:
        raise SystemExit(main())
