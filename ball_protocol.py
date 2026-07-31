"""Fixed 64-byte UART protocol for visual ball state and feedback.

Schema 6 preserves the schema-5 base field offsets and appends the active
reference state. Car feedforward, beam-angle inversion and the measured-angle
inner loop belong to the MCU.
"""


UART_SYNC_0 = 0xA5
UART_SYNC_1 = 0x5A
UART_PROTOCOL_VERSION = 1
UART_MSG_VISION_EVENT = 0x02
UART_MSG_SET_REFERENCE = 0x10
UART_FRAME_FLAG_REALTIME = 1 << 0

BALL_STATE_SCHEMA_VERSION = 6
BALL_STATE_KIND = 5
BALL_STATE_PAYLOAD_SIZE = 48
UART_HEADER_SIZE = 14
UART_CRC_SIZE = 2
BALL_STATE_FRAME_SIZE = (
    UART_HEADER_SIZE + BALL_STATE_PAYLOAD_SIZE + UART_CRC_SIZE
)

REFERENCE_COMMAND_SCHEMA_VERSION = 1
REFERENCE_COMMAND_FLAGS = 0
REFERENCE_COMMAND_POSITION_SCALE = 100
REFERENCE_COMMAND_MAX_ABS_CM = 12.5
REFERENCE_COMMAND_FRAME_SIZE = 10

BALL_FLAG_RAW_DETECTED = 1 << 0
BALL_FLAG_MEASUREMENT_ACCEPTED = 1 << 1
BALL_FLAG_TRACK_VALID = 1 << 2
BALL_FLAG_PREDICTED = 1 << 3
BALL_FLAG_FEEDBACK_VALID = 1 << 4
BALL_FLAG_OUTLIER_REJECTED = 1 << 5
BALL_FLAG_EDGE_CLIPPED = 1 << 6
BALL_FLAG_FEEDBACK_SATURATED = 1 << 7

BALL_REFERENCE_FLAG_VALID = 1 << 0
BALL_REFERENCE_FLAG_SOURCE_PRESET = 1 << 1
BALL_REFERENCE_FLAG_SOURCE_ALGORITHM = 1 << 2
BALL_REFERENCE_FLAG_SOURCE_MCU = 1 << 3
BALL_REFERENCE_FLAG_SLEW_ACTIVE = 1 << 4
BALL_REFERENCE_SOURCE_MASK = (
    BALL_REFERENCE_FLAG_SOURCE_PRESET
    | BALL_REFERENCE_FLAG_SOURCE_ALGORITHM
    | BALL_REFERENCE_FLAG_SOURCE_MCU
)
BALL_REFERENCE_KNOWN_MASK = (
    BALL_REFERENCE_FLAG_VALID
    | BALL_REFERENCE_SOURCE_MASK
    | BALL_REFERENCE_FLAG_SLEW_ACTIVE
)

BALL_COORD_SCALE = 100
BALL_PHYSICAL_XY_VELOCITY_SCALE = 10
BALL_POSITION_CM_SCALE = 100
BALL_PHYSICAL_VELOCITY_SCALE = 10
BALL_ACCELERATION_SCALE = 10
BALL_FEEDBACK_SCALE = 10
BALL_GAIN_SCALE = 100
BALL_DERIVATIVE_GAIN_SCALE = 10
BALL_ACCEL_GAIN_SCALE = 100
BALL_REFERENCE_POSITION_SCALE = 100
BALL_REFERENCE_VELOCITY_SCALE = 10
BALL_REFERENCE_ACCELERATION_SCALE = 10


def _clamp(value, low, high):
    return max(low, min(high, value))


def _round_int(value):
    return int(round(float(value)))


def _put_u16(buffer, offset, value):
    value = int(value) & 0xFFFF
    buffer[offset] = value & 0xFF
    buffer[offset + 1] = (value >> 8) & 0xFF


def _put_i16(buffer, offset, value):
    _put_u16(buffer, offset, int(value) & 0xFFFF)


def _put_u32(buffer, offset, value):
    value = int(value) & 0xFFFFFFFF
    buffer[offset] = value & 0xFF
    buffer[offset + 1] = (value >> 8) & 0xFF
    buffer[offset + 2] = (value >> 16) & 0xFF
    buffer[offset + 3] = (value >> 24) & 0xFF


def _get_u16(buffer, offset):
    return int(buffer[offset]) | (int(buffer[offset + 1]) << 8)


def _get_i16(buffer, offset):
    value = _get_u16(buffer, offset)
    return value - 0x10000 if value & 0x8000 else value


def _get_u32(buffer, offset):
    return (
        int(buffer[offset])
        | (int(buffer[offset + 1]) << 8)
        | (int(buffer[offset + 2]) << 16)
        | (int(buffer[offset + 3]) << 24)
    )


def crc16_ccitt_false(buffer, start=0, end=None):
    """CRC16-CCITT-FALSE, polynomial 0x1021, initial value 0xFFFF."""
    if end is None:
        end = len(buffer)
    crc = 0xFFFF
    for index in range(int(start), int(end)):
        crc ^= int(buffer[index]) << 8
        for unused in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_reference_command_frame(reference_position_cm):
    """Encode one fixed command that sets the K230 equilibrium reference."""
    reference_position_cm = float(reference_position_cm)
    if (
        reference_position_cm != reference_position_cm
        or abs(reference_position_cm) > REFERENCE_COMMAND_MAX_ABS_CM
    ):
        raise ValueError("reference command must be within +/-12.50 cm")
    frame = bytearray(REFERENCE_COMMAND_FRAME_SIZE)
    frame[0] = UART_SYNC_0
    frame[1] = UART_SYNC_1
    frame[2] = UART_PROTOCOL_VERSION
    frame[3] = UART_MSG_SET_REFERENCE
    frame[4] = REFERENCE_COMMAND_SCHEMA_VERSION
    frame[5] = REFERENCE_COMMAND_FLAGS
    _put_i16(
        frame,
        6,
        _round_int(
            reference_position_cm * REFERENCE_COMMAND_POSITION_SCALE
        ),
    )
    _put_u16(frame, 8, crc16_ccitt_false(frame, 2, 8))
    return frame


def validate_reference_command_frame(frame):
    if frame is None or len(frame) != REFERENCE_COMMAND_FRAME_SIZE:
        return False
    if int(frame[0]) != UART_SYNC_0 or int(frame[1]) != UART_SYNC_1:
        return False
    if int(frame[2]) != UART_PROTOCOL_VERSION:
        return False
    if int(frame[3]) != UART_MSG_SET_REFERENCE:
        return False
    if int(frame[4]) != REFERENCE_COMMAND_SCHEMA_VERSION:
        return False
    if int(frame[5]) != REFERENCE_COMMAND_FLAGS:
        return False
    if abs(_get_i16(frame, 6)) > int(
        REFERENCE_COMMAND_MAX_ABS_CM * REFERENCE_COMMAND_POSITION_SCALE
    ):
        return False
    return _get_u16(frame, 8) == crc16_ccitt_false(frame, 2, 8)


def decode_reference_command_frame(frame):
    if not validate_reference_command_frame(frame):
        raise ValueError("invalid set-reference UART command")
    return _get_i16(frame, 6) / float(REFERENCE_COMMAND_POSITION_SCALE)


class ReferenceCommandParser:
    """Fixed-memory parser for split, coalesced and noisy command streams."""

    def __init__(self):
        self.frame = bytearray(REFERENCE_COMMAND_FRAME_SIZE)
        self.index = 0
        self.bytes_received = 0
        self.valid_frames = 0
        self.invalid_frames = 0
        self.discarded_bytes = 0

    def _restart_after_invalid_frame(self):
        restart = -1
        for offset in range(1, REFERENCE_COMMAND_FRAME_SIZE - 1):
            if (
                int(self.frame[offset]) == UART_SYNC_0
                and int(self.frame[offset + 1]) == UART_SYNC_1
            ):
                restart = offset
        if restart < 0:
            if int(self.frame[REFERENCE_COMMAND_FRAME_SIZE - 1]) == UART_SYNC_0:
                self.frame[0] = UART_SYNC_0
                self.index = 1
                self.discarded_bytes += REFERENCE_COMMAND_FRAME_SIZE - 1
            else:
                self.index = 0
                self.discarded_bytes += REFERENCE_COMMAND_FRAME_SIZE
            return
        remaining = REFERENCE_COMMAND_FRAME_SIZE - restart
        for target in range(remaining):
            self.frame[target] = self.frame[restart + target]
        self.index = remaining
        self.discarded_bytes += restart

    def _consume_byte(self, value):
        value = int(value) & 0xFF
        if self.index == 0:
            if value == UART_SYNC_0:
                self.frame[0] = value
                self.index = 1
            else:
                self.discarded_bytes += 1
            return None
        if self.index == 1:
            if value == UART_SYNC_1:
                self.frame[1] = value
                self.index = 2
            elif value == UART_SYNC_0:
                self.frame[0] = value
                self.discarded_bytes += 1
            else:
                self.index = 0
                self.discarded_bytes += 2
            return None

        self.frame[self.index] = value
        self.index += 1
        if self.index < REFERENCE_COMMAND_FRAME_SIZE:
            return None
        if validate_reference_command_frame(self.frame):
            reference_position_cm = decode_reference_command_frame(self.frame)
            self.valid_frames += 1
            self.index = 0
            return reference_position_cm
        self.invalid_frames += 1
        self._restart_after_invalid_frame()
        return None

    def feed(self, data):
        """Return the newest valid reference in this chunk, or None."""
        latest_reference = None
        if data is None:
            return None
        self.bytes_received += len(data)
        for value in data:
            decoded = self._consume_byte(value)
            if decoded is not None:
                latest_reference = decoded
        return latest_reference


def _candidate_value(candidate, name, default=0):
    if candidate is None:
        return default
    return candidate.get(name, default)


def _state_value(state, name, default=0):
    if state is None:
        return default
    return state.get(name, default)


def encode_ball_state_frame(
    sequence,
    timestamp_ms,
    flags,
    state,
    candidate=None,
    inference_ms=0,
    feedback=None,
):
    """Encode one fixed 64-byte schema-6 visual feedback frame."""
    frame = bytearray(BALL_STATE_FRAME_SIZE)
    frame[0] = UART_SYNC_0
    frame[1] = UART_SYNC_1
    frame[2] = UART_PROTOCOL_VERSION
    frame[3] = UART_MSG_VISION_EVENT
    frame[4] = BALL_STATE_SCHEMA_VERSION
    frame[5] = UART_FRAME_FLAG_REALTIME
    _put_u16(frame, 6, BALL_STATE_PAYLOAD_SIZE)
    _put_u16(frame, 8, sequence)
    _put_u32(frame, 10, timestamp_ms)

    payload = UART_HEADER_SIZE
    frame[payload] = BALL_STATE_KIND
    frame[payload + 1] = int(flags) & 0xFF
    frame[payload + 2] = _clamp(
        _round_int(_candidate_value(candidate, "confidence", 0.0) * 255.0),
        0,
        255,
    )
    frame[payload + 3] = _clamp(
        int(_state_value(state, "missed_frames", 0)),
        0,
        255,
    )
    _put_u16(
        frame,
        payload + 4,
        _clamp(_round_int(_candidate_value(candidate, "center_x", 0)), 0, 0xFFFF),
    )
    _put_u16(
        frame,
        payload + 6,
        _clamp(_round_int(_candidate_value(candidate, "center_y", 0)), 0, 0xFFFF),
    )
    _put_i16(
        frame,
        payload + 8,
        _clamp(
            _round_int(
                _state_value(feedback, "position_cm", 0.0)
                * BALL_COORD_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 10,
        _clamp(
            _round_int(
                _state_value(feedback, "position_y_cm", 0.0)
                * BALL_COORD_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 12,
        _clamp(
            _round_int(
                _state_value(feedback, "velocity_cm_s", 0.0)
                * BALL_PHYSICAL_XY_VELOCITY_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 14,
        _clamp(
            _round_int(
                _state_value(feedback, "velocity_y_cm_s", 0.0)
                * BALL_PHYSICAL_XY_VELOCITY_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 16,
        _clamp(
            _round_int(
                _state_value(feedback, "position_cm", 0.0)
                * BALL_POSITION_CM_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 18,
        _clamp(
            _round_int(
                _state_value(feedback, "velocity_cm_s", 0.0)
                * BALL_PHYSICAL_VELOCITY_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 20,
        _clamp(
            _round_int(
                _state_value(feedback, "acceleration_cm_s2", 0.0)
                * BALL_ACCELERATION_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 22,
        _clamp(
            _round_int(
                _state_value(feedback, "feedback_acceleration_cm_s2", 0.0)
                * BALL_FEEDBACK_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_u16(frame, payload + 24, _clamp(int(inference_ms), 0, 0xFFFF))
    _put_u16(
        frame,
        payload + 26,
        _clamp(
            int(_state_value(state, "measurement_age_ms", 0xFFFF)),
            0,
            0xFFFF,
        ),
    )
    _put_u16(
        frame,
        payload + 28,
        _clamp(
            _round_int(_state_value(feedback, "kp", 0.0) * BALL_GAIN_SCALE),
            0,
            0xFFFF,
        ),
    )
    frame[payload + 30] = _clamp(
        _round_int(
            _state_value(feedback, "kd", 0.0)
            * BALL_DERIVATIVE_GAIN_SCALE
        ),
        0,
        255,
    )
    frame[payload + 31] = _clamp(
        _round_int(
            _state_value(feedback, "ka", 0.0) * BALL_ACCEL_GAIN_SCALE
        ),
        0,
        255,
    )
    _put_i16(
        frame,
        payload + 32,
        _clamp(
            _round_int(
                _state_value(feedback, "reference_position_cm", 0.0)
                * BALL_REFERENCE_POSITION_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 34,
        _clamp(
            _round_int(
                _state_value(feedback, "position_error_cm", 0.0)
                * BALL_REFERENCE_POSITION_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 36,
        _clamp(
            _round_int(
                _state_value(feedback, "reference_velocity_cm_s", 0.0)
                * BALL_REFERENCE_VELOCITY_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 38,
        _clamp(
            _round_int(
                _state_value(
                    feedback,
                    "reference_acceleration_cm_s2",
                    0.0,
                ) * BALL_REFERENCE_ACCELERATION_SCALE
            ),
            -32768,
            32767,
        ),
    )
    _put_u16(
        frame,
        payload + 40,
        _clamp(
            int(_state_value(feedback, "reference_state_flags", 0)),
            0,
            0xFFFF,
        ),
    )
    # Payload 42..47 is reserved and remains zero from bytearray allocation.

    crc_offset = UART_HEADER_SIZE + BALL_STATE_PAYLOAD_SIZE
    _put_u16(frame, crc_offset, crc16_ccitt_false(frame, 2, crc_offset))
    return frame


def validate_ball_state_frame(frame):
    if frame is None or len(frame) != BALL_STATE_FRAME_SIZE:
        return False
    if int(frame[0]) != UART_SYNC_0 or int(frame[1]) != UART_SYNC_1:
        return False
    if int(frame[2]) != UART_PROTOCOL_VERSION:
        return False
    if int(frame[3]) != UART_MSG_VISION_EVENT:
        return False
    if int(frame[4]) != BALL_STATE_SCHEMA_VERSION:
        return False
    if int(frame[5]) != UART_FRAME_FLAG_REALTIME:
        return False
    if _get_u16(frame, 6) != BALL_STATE_PAYLOAD_SIZE:
        return False
    if int(frame[UART_HEADER_SIZE]) != BALL_STATE_KIND:
        return False
    reference_flags = _get_u16(frame, UART_HEADER_SIZE + 40)
    if reference_flags & ~BALL_REFERENCE_KNOWN_MASK:
        return False
    source_flags = reference_flags & BALL_REFERENCE_SOURCE_MASK
    if reference_flags & BALL_REFERENCE_FLAG_VALID:
        if source_flags not in (
            BALL_REFERENCE_FLAG_SOURCE_PRESET,
            BALL_REFERENCE_FLAG_SOURCE_ALGORITHM,
            BALL_REFERENCE_FLAG_SOURCE_MCU,
        ):
            return False
    elif reference_flags != 0:
        return False
    for offset in range(UART_HEADER_SIZE + 42, UART_HEADER_SIZE + 48):
        if int(frame[offset]) != 0:
            return False
    crc_offset = UART_HEADER_SIZE + BALL_STATE_PAYLOAD_SIZE
    return _get_u16(frame, crc_offset) == crc16_ccitt_false(
        frame,
        2,
        crc_offset,
    )


def decode_ball_state_frame(frame):
    """Validate and decode one schema-6 frame for host diagnostics."""
    if not validate_ball_state_frame(frame):
        raise ValueError("invalid visual-ball UART frame")
    payload = UART_HEADER_SIZE
    return {
        "schema_version": int(frame[4]),
        "kind": int(frame[payload]),
        "sequence": _get_u16(frame, 8),
        "timestamp_ms": _get_u32(frame, 10),
        "flags": int(frame[payload + 1]),
        "confidence_x255": int(frame[payload + 2]),
        "missed_frames": int(frame[payload + 3]),
        "raw_x": _get_u16(frame, payload + 4),
        "raw_y": _get_u16(frame, payload + 6),
        "physical_x_cm": _get_i16(frame, payload + 8)
        / float(BALL_COORD_SCALE),
        "physical_y_cm": _get_i16(frame, payload + 10)
        / float(BALL_COORD_SCALE),
        "physical_vx_cm_s": _get_i16(frame, payload + 12)
        / float(BALL_PHYSICAL_XY_VELOCITY_SCALE),
        "physical_vy_cm_s": _get_i16(frame, payload + 14)
        / float(BALL_PHYSICAL_XY_VELOCITY_SCALE),
        "position_cm": _get_i16(frame, payload + 16)
        / float(BALL_POSITION_CM_SCALE),
        "velocity_cm_s": _get_i16(frame, payload + 18)
        / float(BALL_PHYSICAL_VELOCITY_SCALE),
        "acceleration_cm_s2": _get_i16(frame, payload + 20)
        / float(BALL_ACCELERATION_SCALE),
        "feedback_acceleration_cm_s2": _get_i16(frame, payload + 22)
        / float(BALL_FEEDBACK_SCALE),
        "detector_ms": _get_u16(frame, payload + 24),
        # Compatibility alias for existing PC/MCU-side Python tooling.
        "inference_ms": _get_u16(frame, payload + 24),
        "measurement_age_ms": _get_u16(frame, payload + 26),
        "kp": _get_u16(frame, payload + 28) / float(BALL_GAIN_SCALE),
        "kd": int(frame[payload + 30])
        / float(BALL_DERIVATIVE_GAIN_SCALE),
        "ka": int(frame[payload + 31]) / float(BALL_ACCEL_GAIN_SCALE),
        "reference_position_cm": _get_i16(frame, payload + 32)
        / float(BALL_REFERENCE_POSITION_SCALE),
        "position_error_cm": _get_i16(frame, payload + 34)
        / float(BALL_REFERENCE_POSITION_SCALE),
        "reference_velocity_cm_s": _get_i16(frame, payload + 36)
        / float(BALL_REFERENCE_VELOCITY_SCALE),
        "reference_acceleration_cm_s2": _get_i16(frame, payload + 38)
        / float(BALL_REFERENCE_ACCELERATION_SCALE),
        "reference_state_flags": _get_u16(frame, payload + 40),
        "reserved_extension": bytes(frame[payload + 42:payload + 48]),
        "crc16": _get_u16(frame, BALL_STATE_FRAME_SIZE - UART_CRC_SIZE),
    }
