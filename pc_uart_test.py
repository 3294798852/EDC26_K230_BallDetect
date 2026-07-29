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
        BALL_STATE_FRAME_SIZE,
        BALL_STATE_PAYLOAD_SIZE,
        UART_HEADER_SIZE,
        UART_SYNC_0,
        UART_SYNC_1,
        crc16_ccitt_false,
        decode_ball_state_frame,
        encode_ball_state_frame,
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
    BALL_STATE_FRAME_SIZE = 48
    BALL_STATE_PAYLOAD_SIZE = 32
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
        "miss={} infer={}ms meas_age={}".format(
            state["sequence"],
            state["timestamp_ms"],
            _format_tracking_mode(state["flags"]),
            _format_flags(state["flags"]),
            state["confidence_x255"] / 255.0,
            state["missed_frames"],
            state["inference_ms"],
            _format_age(state["measurement_age_ms"]),
        )
    )
    print(
        "  raw=({:3d},{:3d}) filt=({:7.2f},{:7.2f}) "
        "vel=({:8.1f},{:8.1f})px/s".format(
            state["raw_x"],
            state["raw_y"],
            state["filtered_x"],
            state["filtered_y"],
            state["vx"],
            state["vy"],
        )
    )
    print(
        "  phys p={:+.2f}cm v={:+.1f}cm/s a={:+.1f}cm/s2 "
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


def run_self_test():
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
        "velocity_cm_s": 12.3,
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
    assert decoded["velocity_cm_s"] == 12.3
    assert decoded["acceleration_cm_s2"] == 18.4
    assert decoded["feedback_acceleration_cm_s2"] == -36.6
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
        description="Receive and decode K230 proj2 48-byte steel-ball frames."
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
