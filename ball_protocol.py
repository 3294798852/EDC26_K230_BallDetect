"""Fixed-size UART protocol for the standalone steel-ball tracker.

The envelope is compatible with the previous K230 project, while schema 2
contains one filtered target state instead of a variable list of detections.
All multi-byte values are little-endian and the complete frame is 48 bytes.
"""


UART_SYNC_0 = 0xA5
UART_SYNC_1 = 0x5A
UART_PROTOCOL_VERSION = 1
UART_MSG_VISION_EVENT = 0x02
UART_FRAME_FLAG_REALTIME = 1 << 0

BALL_STATE_SCHEMA_VERSION = 2
BALL_STATE_KIND = 3
BALL_STATE_PAYLOAD_SIZE = 32
UART_HEADER_SIZE = 14
UART_CRC_SIZE = 2
BALL_STATE_FRAME_SIZE = (
    UART_HEADER_SIZE + BALL_STATE_PAYLOAD_SIZE + UART_CRC_SIZE
)

# Payload flags.
BALL_FLAG_RAW_DETECTED = 1 << 0
BALL_FLAG_MEASUREMENT_ACCEPTED = 1 << 1
BALL_FLAG_TRACK_VALID = 1 << 2
BALL_FLAG_PREDICTED = 1 << 3
BALL_FLAG_AXIS_LOCKED = 1 << 4
BALL_FLAG_OUTLIER_REJECTED = 1 << 5
BALL_FLAG_EDGE_CLIPPED = 1 << 6
BALL_FLAG_FILTER_RESET = 1 << 7

BALL_COORD_SCALE = 100
BALL_VELOCITY_SCALE = 10
BALL_AXIS_Q = 14
BALL_AXIS_SCALE = 1 << BALL_AXIS_Q


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
):
    """Encode one fixed 48-byte filtered-ball state frame."""
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
    confidence = _clamp(
        _round_int(_candidate_value(candidate, "confidence", 0.0) * 255.0),
        0,
        255,
    )
    frame[payload + 2] = confidence
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
    _put_u16(
        frame,
        payload + 8,
        _clamp(
            _round_int(_state_value(state, "x", 0.0) * BALL_COORD_SCALE),
            0,
            0xFFFF,
        ),
    )
    _put_u16(
        frame,
        payload + 10,
        _clamp(
            _round_int(_state_value(state, "y", 0.0) * BALL_COORD_SCALE),
            0,
            0xFFFF,
        ),
    )
    _put_i16(
        frame,
        payload + 12,
        _clamp(
            _round_int(_state_value(state, "vx", 0.0) * BALL_VELOCITY_SCALE),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 14,
        _clamp(
            _round_int(_state_value(state, "vy", 0.0) * BALL_VELOCITY_SCALE),
            -32768,
            32767,
        ),
    )
    _put_u16(
        frame,
        payload + 16,
        _clamp(int(_candidate_value(candidate, "x", 0)), 0, 0xFFFF),
    )
    _put_u16(
        frame,
        payload + 18,
        _clamp(int(_candidate_value(candidate, "y", 0)), 0, 0xFFFF),
    )
    _put_u16(
        frame,
        payload + 20,
        _clamp(int(_candidate_value(candidate, "w", 0)), 0, 0xFFFF),
    )
    _put_u16(
        frame,
        payload + 22,
        _clamp(int(_candidate_value(candidate, "h", 0)), 0, 0xFFFF),
    )
    _put_u16(
        frame,
        payload + 24,
        _clamp(int(inference_ms), 0, 0xFFFF),
    )
    _put_u16(
        frame,
        payload + 26,
        _clamp(
            int(_state_value(state, "measurement_age_ms", 0xFFFF)),
            0,
            0xFFFF,
        ),
    )
    _put_i16(
        frame,
        payload + 28,
        _clamp(
            _round_int(_state_value(state, "axis_x", 1.0) * BALL_AXIS_SCALE),
            -32768,
            32767,
        ),
    )
    _put_i16(
        frame,
        payload + 30,
        _clamp(
            _round_int(_state_value(state, "axis_y", 0.0) * BALL_AXIS_SCALE),
            -32768,
            32767,
        ),
    )

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
    if _get_u16(frame, 6) != BALL_STATE_PAYLOAD_SIZE:
        return False
    if int(frame[UART_HEADER_SIZE]) != BALL_STATE_KIND:
        return False
    crc_offset = UART_HEADER_SIZE + BALL_STATE_PAYLOAD_SIZE
    return _get_u16(frame, crc_offset) == crc16_ccitt_false(
        frame,
        2,
        crc_offset,
    )


def decode_ball_state_frame(frame):
    """Validate and decode a frame; intended for tests and MCU-side models."""
    if not validate_ball_state_frame(frame):
        raise ValueError("invalid filtered-ball UART frame")
    payload = UART_HEADER_SIZE
    return {
        "sequence": _get_u16(frame, 8),
        "timestamp_ms": _get_u32(frame, 10),
        "flags": int(frame[payload + 1]),
        "confidence_x255": int(frame[payload + 2]),
        "missed_frames": int(frame[payload + 3]),
        "raw_x": _get_u16(frame, payload + 4),
        "raw_y": _get_u16(frame, payload + 6),
        "filtered_x": _get_u16(frame, payload + 8) / float(BALL_COORD_SCALE),
        "filtered_y": _get_u16(frame, payload + 10) / float(BALL_COORD_SCALE),
        "vx": _get_i16(frame, payload + 12) / float(BALL_VELOCITY_SCALE),
        "vy": _get_i16(frame, payload + 14) / float(BALL_VELOCITY_SCALE),
        "box_x": _get_u16(frame, payload + 16),
        "box_y": _get_u16(frame, payload + 18),
        "box_w": _get_u16(frame, payload + 20),
        "box_h": _get_u16(frame, payload + 22),
        "inference_ms": _get_u16(frame, payload + 24),
        "measurement_age_ms": _get_u16(frame, payload + 26),
        "axis_x": _get_i16(frame, payload + 28) / float(BALL_AXIS_SCALE),
        "axis_y": _get_i16(frame, payload + 30) / float(BALL_AXIS_SCALE),
        "crc16": _get_u16(frame, BALL_STATE_FRAME_SIZE - UART_CRC_SIZE),
    }
