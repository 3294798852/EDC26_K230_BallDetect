"""Standalone K230 steel-ball tracker with filtered UART output.

This entry intentionally has no command handshake or task supervisor.  It
starts the camera, detector, constrained motion filter, UART and lightweight
overlay once, then continuously publishes the newest 48-byte state frame.
"""

import gc
import os
import sys
import time

# Keep module loading alive long enough to put dependency failures on the LCD.
# These defaults are used only while reporting a failed import.
aidemo = None
image = None
nn = None
np = None
FPIOA = None
UART = None
Display = None
MediaManager = None
Sensor = None
CAM_CHN_ID_0 = 0
CAM_CHN_ID_2 = 2
PIXEL_FORMAT_RGB_888_PLANAR = 0
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 0
PipeMotionFilter = None
select_highest_confidence_ball = None
encode_ball_state_frame = None

BALL_FLAG_RAW_DETECTED = 0x01
BALL_FLAG_MEASUREMENT_ACCEPTED = 0x02
BALL_FLAG_TRACK_VALID = 0x04
BALL_FLAG_PREDICTED = 0x08
BALL_FLAG_AXIS_LOCKED = 0x10
BALL_FLAG_OUTLIER_REJECTED = 0x20
BALL_FLAG_EDGE_CLIPPED = 0x40
BALL_FLAG_FILTER_RESET = 0x80
BALL_STATE_FRAME_SIZE = 48

_BOOT_IMPORT_ERROR = None
_BOOT_IMPORT_STAGE = None
try:
    _BOOT_IMPORT_STAGE = "import_aidemo"
    import aidemo
    _BOOT_IMPORT_STAGE = "import_image"
    import image
    _BOOT_IMPORT_STAGE = "import_nncase_runtime"
    import nncase_runtime as nn
    _BOOT_IMPORT_STAGE = "import_ulab_numpy"
    import ulab.numpy as np
    _BOOT_IMPORT_STAGE = "import_machine_uart"
    from machine import FPIOA, UART
    _BOOT_IMPORT_STAGE = "import_media_display"
    from media.display import Display
    _BOOT_IMPORT_STAGE = "import_media_manager"
    from media.media import MediaManager
    _BOOT_IMPORT_STAGE = "import_media_sensor"
    from media.sensor import (
        CAM_CHN_ID_0,
        CAM_CHN_ID_2,
        PIXEL_FORMAT_RGB_888_PLANAR,
        PIXEL_FORMAT_YUV_SEMIPLANAR_420,
        Sensor,
    )

    _BOOT_IMPORT_STAGE = "import_ball_protocol"
    from ball_protocol import (
        BALL_FLAG_AXIS_LOCKED,
        BALL_FLAG_EDGE_CLIPPED,
        BALL_FLAG_FILTER_RESET,
        BALL_FLAG_MEASUREMENT_ACCEPTED,
        BALL_FLAG_OUTLIER_REJECTED,
        BALL_FLAG_PREDICTED,
        BALL_FLAG_RAW_DETECTED,
        BALL_FLAG_TRACK_VALID,
        BALL_STATE_FRAME_SIZE,
        encode_ball_state_frame,
    )
    _BOOT_IMPORT_STAGE = "import_ball_tracking"
    from ball_tracking import PipeMotionFilter, select_highest_confidence_ball
    _BOOT_IMPORT_STAGE = None
except Exception as _boot_exc:
    _BOOT_IMPORT_ERROR = _boot_exc


APP_VERSION = "proj2-1.2"
MODEL_PATH = "/sdcard/task_assets/ball_det/ball_det_yolov8n_320.kmodel"
MODEL_INPUT_WIDTH = 320
MODEL_INPUT_HEIGHT = 320
MODEL_EXPECTED_OUTPUT = (1, 5, 2100)
CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45
POSTPROCESS_MAX_BOXES = 8

CAMERA_FPS = 30
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
OVERLAY_INTERVAL_MS = 100
PC_DEBUG_INTERVAL_MS = 500
GC_INTERVAL_FRAMES = 60
MAX_CONSECUTIVE_FRAME_ERRORS = 3

UART_BAUDRATE = 512000
UART_MAX_WRITE_BYTES = 64
UART_SOFTWARE_FRAME_CAPACITY = 2
UART_SOFTWARE_BUFFER_BYTES = (
    BALL_STATE_FRAME_SIZE * UART_SOFTWARE_FRAME_CAPACITY
)

COLOR_PANEL = (180, 0, 0, 0)
COLOR_TITLE = (255, 0, 255, 255)
COLOR_GOOD = (255, 0, 255, 0)
COLOR_WARN = (255, 255, 220, 0)
COLOR_BAD = (255, 255, 40, 40)
COLOR_RAW = (255, 255, 180, 0)
COLOR_FILTERED = (255, 0, 255, 255)
COLOR_AXIS = (255, 80, 160, 255)
STATUS_PANEL_HEIGHT = 182


def _ticks_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.time() * 1000)


def _ticks_diff(new_value, old_value):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(new_value, old_value)
    return int(new_value) - int(old_value)


def _sleep_ms(milliseconds):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(int(milliseconds))
    else:
        time.sleep(float(milliseconds) / 1000.0)


def _exitpoint():
    if hasattr(os, "exitpoint"):
        os.exitpoint()


def _print_exception(exc):
    printer = getattr(sys, "print_exception", None)
    if printer is not None:
        try:
            printer(exc)
            return
        except Exception:
            pass
    print(type(exc).__name__, exc)


def _is_ide_interrupt(exc):
    """CanMV reports an IDE stop request as a plain Exception in C calls."""
    if isinstance(exc, KeyboardInterrupt):
        return True
    try:
        return "ide interrupt" in str(exc).strip().lower()
    except Exception:
        return False


def _fatal_hold_loop(tx=None):
    """Keep the current diagnostic page visible until the IDE stops it."""
    poll_warning_printed = False
    exitpoint_warning_printed = False
    sleep_warning_printed = False
    while True:
        try:
            _exitpoint()
        except KeyboardInterrupt:
            return
        except Exception as exit_exc:
            if _is_ide_interrupt(exit_exc):
                return
            if not exitpoint_warning_printed:
                print("fatal hold exitpoint warning:", exit_exc)
                exitpoint_warning_printed = True
        if tx is not None:
            try:
                tx.poll()
            except KeyboardInterrupt:
                return
            except Exception as poll_exc:
                if _is_ide_interrupt(poll_exc):
                    return
                if not poll_warning_printed:
                    print("fatal hold UART poll warning:", poll_exc)
                    poll_warning_printed = True
        try:
            _sleep_ms(100)
        except KeyboardInterrupt:
            return
        except Exception as sleep_exc:
            if _is_ide_interrupt(sleep_exc):
                return
            if not sleep_warning_printed:
                print("fatal hold sleep warning:", sleep_exc)
                sleep_warning_printed = True


def _emergency_hold_fatal(stage, exc, tx=None, already_reported=False):
    """Render a fatal page even when normal application imports/init failed."""
    if not already_reported:
        print(
            "steel-ball tracker fatal at {}: {}: {}".format(
                stage,
                type(exc).__name__,
                exc,
            )
        )
        _print_exception(exc)

    emergency_display = None
    emergency_media = None
    display_initialized = False
    media_initialized = False
    stop_requested = False
    try:
        import image as emergency_image
        from media.display import Display as emergency_display
        from media.media import MediaManager as emergency_media

        canvas = emergency_image.Image(
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            emergency_image.ARGB8888,
        )
        emergency_display.init(
            emergency_display.ST7701,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
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
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            color=(255, 0, 0, 0),
            fill=True,
        )

        def draw_line(row, text, color=COLOR_TITLE, size=16):
            canvas.draw_string_advanced(
                5,
                4 + int(row) * 27,
                size,
                str(text),
                color=color,
            )

        message = str(exc)
        stage_text = str(stage)
        draw_line(0, "STEEL BALL TRACKER FATAL", COLOR_BAD, 22)
        draw_line(1, "version: " + APP_VERSION, COLOR_TITLE)
        draw_line(2, "stage: " + stage_text[:58], COLOR_WARN)
        draw_line(3, "       " + stage_text[58:116], COLOR_WARN)
        draw_line(4, "type: " + type(exc).__name__, COLOR_BAD)
        draw_line(5, "error: " + message[:58], COLOR_BAD)
        draw_line(6, "       " + message[58:116], COLOR_BAD)
        draw_line(7, "       " + message[116:174], COLOR_BAD)
        if tx is not None:
            draw_line(
                9,
                "UART txF={} txB={} err={} short={} zero={} drop={}".format(
                    getattr(tx, "sent_frames", 0),
                    getattr(tx, "bytes_sent", 0),
                    getattr(tx, "write_errors", 0),
                    getattr(tx, "short_writes", 0),
                    getattr(tx, "zero_writes", 0),
                    getattr(tx, "dropped_error_frames", 0),
                ),
                COLOR_WARN,
            )
            draw_line(
                10,
                "UART last: " + str(getattr(tx, "last_error", None))[:58],
                COLOR_WARN,
            )
        draw_line(13, "Error is latched. Record this page.", COLOR_TITLE)
        draw_line(14, "Stop the program manually to exit.", COLOR_TITLE)
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
            _print_exception(screen_exc)

    try:
        if not stop_requested:
            _fatal_hold_loop(tx)
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


def _verify_model():
    try:
        stat = os.stat(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "model stat failed {}: {}: {}".format(
                MODEL_PATH,
                type(exc).__name__,
                exc,
            )
        )
    if int(stat[6]) <= 0:
        raise RuntimeError("empty model: " + MODEL_PATH)


class BallDetector:
    """Minimal single-output YOLOv8 KPU path using verified CanMV APIs."""

    def __init__(self):
        self.kpu = None
        self.ai2d = None
        self.ai2d_builder = None
        self.ai2d_output_tensor = None
        self.ai2d_input_tensor = None
        self.output_shape_checked = False
        try:
            self.kpu = nn.kpu()
            self.kpu.load_kmodel(MODEL_PATH)
            if int(self.kpu.inputs_size()) != 1:
                raise RuntimeError("ball model must have exactly one input")
            if int(self.kpu.outputs_size()) != 1:
                raise RuntimeError("ball model must have exactly one output")

            self.ai2d = nn.ai2d()
            self.ai2d.set_dtype(
                nn.ai2d_format.NCHW_FMT,
                nn.ai2d_format.NCHW_FMT,
                np.uint8,
                np.uint8,
            )
            self.ai2d.set_resize_param(
                True,
                nn.interp_method.tf_bilinear,
                nn.interp_mode.half_pixel,
            )
            input_shape = [
                1,
                3,
                MODEL_INPUT_HEIGHT,
                MODEL_INPUT_WIDTH,
            ]
            output_shape = [
                1,
                3,
                MODEL_INPUT_HEIGHT,
                MODEL_INPUT_WIDTH,
            ]
            self.ai2d_builder = self.ai2d.build(
                input_shape,
                output_shape,
            )
            output_array = np.ones(
                (
                    output_shape[0],
                    output_shape[1],
                    output_shape[2],
                    output_shape[3],
                ),
                dtype=np.uint8,
            )
            self.ai2d_output_tensor = nn.from_numpy(output_array)
        except Exception:
            self.deinit()
            raise

    def _check_output_shape(self, output_array):
        shape = getattr(output_array, "shape", None)
        if shape is None or len(shape) != 3:
            raise RuntimeError("ball model returned an invalid output rank")
        actual = (int(shape[0]), int(shape[1]), int(shape[2]))
        if actual != MODEL_EXPECTED_OUTPUT:
            raise RuntimeError(
                "ball model output {} != {}".format(
                    actual,
                    MODEL_EXPECTED_OUTPUT,
                )
            )
        self.output_shape_checked = True

    def run(self, input_array):
        self.ai2d_input_tensor = nn.from_numpy(input_array)
        self.ai2d_builder.run(
            self.ai2d_input_tensor,
            self.ai2d_output_tensor,
        )
        self.kpu.set_input_tensor(0, self.ai2d_output_tensor)
        self.kpu.run()
        output_tensor = self.kpu.get_output_tensor(0)
        output_array = output_tensor.to_numpy()
        del output_tensor
        self.ai2d_input_tensor = None
        if not self.output_shape_checked:
            self._check_output_shape(output_array)
        candidates = output_array[0].transpose()
        return aidemo.yolov8_det_postprocess(
            candidates.copy(),
            [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
            [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
            [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
            1,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
            POSTPROCESS_MAX_BOXES,
        )

    def deinit(self):
        self.ai2d_input_tensor = None
        self.ai2d_output_tensor = None
        self.ai2d_builder = None
        self.ai2d = None
        self.kpu = None
        gc.collect()
        try:
            nn.shrink_memory_pool()
        except Exception as exc:
            print("KPU memory-pool shrink warning:", exc)


class CameraDisplay:
    """Own camera channel 0 video, channel 2 AI input and one OSD layer."""

    def __init__(self):
        self.sensor = None
        self.osd = None
        self.display_initialized = False
        self.media_initialized = False
        self.sensor_running = False

    def start(self):
        sensor = Sensor(fps=CAMERA_FPS)
        self.sensor = sensor
        sensor.reset()
        sensor.set_framesize(
            w=DISPLAY_WIDTH,
            h=DISPLAY_HEIGHT,
            chn=CAM_CHN_ID_0,
        )
        sensor.set_pixformat(
            PIXEL_FORMAT_YUV_SEMIPLANAR_420,
            chn=CAM_CHN_ID_0,
        )
        sensor.set_framesize(
            w=MODEL_INPUT_WIDTH,
            h=MODEL_INPUT_HEIGHT,
            chn=CAM_CHN_ID_2,
        )
        sensor.set_pixformat(
            PIXEL_FORMAT_RGB_888_PLANAR,
            chn=CAM_CHN_ID_2,
        )
        self.osd = image.Image(
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            image.ARGB8888,
        )
        bind_info = sensor.bind_info(
            x=0,
            y=0,
            chn=CAM_CHN_ID_0,
        )
        Display.bind_layer(**bind_info, layer=Display.LAYER_VIDEO1)
        Display.init(
            Display.ST7701,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            osd_num=1,
            to_ide=True,
        )
        self.display_initialized = True
        MediaManager.init()
        self.media_initialized = True
        sensor.run()
        self.sensor_running = True

    def get_frame(self):
        frame = self.sensor.snapshot(chn=CAM_CHN_ID_2)
        if frame is None:
            raise RuntimeError("camera returned no AI frame")
        return frame.to_numpy_ref()

    @staticmethod
    def _screen_x(value):
        return int(float(value) * DISPLAY_WIDTH / MODEL_INPUT_WIDTH)

    @staticmethod
    def _screen_y(value):
        return int(float(value) * DISPLAY_HEIGHT / MODEL_INPUT_HEIGHT)

    def _draw_text(self, row, text, color):
        self.osd.draw_string_advanced(
            5,
            4 + int(row) * 24,
            16,
            str(text),
            color=color,
        )

    def draw_status(
        self,
        candidate,
        state,
        inference_ms,
        frame_interval_ms,
        tx,
        runtime_error=None,
    ):
        osd = self.osd
        osd.clear()
        osd.draw_rectangle(
            0,
            0,
            DISPLAY_WIDTH,
            STATUS_PANEL_HEIGHT,
            color=COLOR_PANEL,
            fill=True,
        )

        if candidate is not None:
            box_color = (
                COLOR_BAD if state.get("outlier_rejected") else COLOR_RAW
            )
            osd.draw_rectangle(
                self._screen_x(candidate["x"]),
                self._screen_y(candidate["y"]),
                max(1, self._screen_x(candidate["w"])),
                max(1, self._screen_y(candidate["h"])),
                color=box_color,
                thickness=3,
            )

        if state.get("track_valid"):
            center_x = self._screen_x(state["x"])
            center_y = self._screen_y(state["y"])
            osd.draw_cross(
                center_x,
                center_y,
                color=COLOR_FILTERED,
                size=12,
                thickness=2,
            )
            if state.get("axis_locked"):
                axis_x = state["axis_x"]
                axis_y = state["axis_y"]
                span = 55.0
                osd.draw_line(
                    self._screen_x(state["x"] - span * axis_x),
                    self._screen_y(state["y"] - span * axis_y),
                    self._screen_x(state["x"] + span * axis_x),
                    self._screen_y(state["y"] + span * axis_y),
                    color=COLOR_AXIS,
                    thickness=2,
                )

        if state.get("measurement_accepted"):
            mode = "MEAS"
            mode_color = COLOR_GOOD
        elif state.get("track_valid"):
            mode = "PRED"
            mode_color = COLOR_WARN
        else:
            mode = "NO TRACK"
            mode_color = COLOR_BAD
        self._draw_text(
            0,
            "STEEL BALL {} {}".format(APP_VERSION, mode),
            mode_color,
        )
        self._draw_text(
            1,
            "F x={:.2f} y={:.2f}  Vx={:.1f} Vy={:.1f} px/s".format(
                state.get("x", 0.0),
                state.get("y", 0.0),
                state.get("vx", 0.0),
                state.get("vy", 0.0),
            ),
            COLOR_GOOD if state.get("track_valid") else COLOR_WARN,
        )
        confidence = 0.0 if candidate is None else candidate["confidence"]
        self._draw_text(
            2,
            "conf={:.3f} infer={}ms age={}ms axis={} frame={}ms".format(
                confidence,
                int(inference_ms),
                int(state.get("measurement_age_ms", 0xFFFF)),
                "LOCK" if state.get("axis_locked") else "LEARN",
                int(frame_interval_ms),
            ),
            COLOR_TITLE,
        )
        self._draw_text(
            3,
            "UART txF={} txB={} repl={} queued={}/{}B".format(
                tx.sent_frames,
                tx.bytes_sent,
                tx.replaced_frames,
                tx.queued_bytes(),
                UART_SOFTWARE_BUFFER_BYTES,
            ),
            COLOR_GOOD,
        )
        self._draw_text(
            4,
            "UART err={} short={} zero={} drop={}".format(
                tx.write_errors,
                tx.short_writes,
                tx.zero_writes,
                tx.dropped_error_frames,
            ),
            COLOR_BAD
            if (
                tx.write_errors
                or tx.short_writes
                or tx.zero_writes
                or tx.dropped_error_frames
            )
            else COLOR_GOOD,
        )
        self._draw_text(
            5,
            "UART last: " + str(tx.last_error)[:65],
            COLOR_BAD if tx.last_error is not None else COLOR_GOOD,
        )
        if runtime_error is not None:
            self._draw_text(
                6,
                "latched error: " + str(runtime_error)[:65],
                COLOR_BAD,
            )
        Display.show_image(osd, 0, 0, Display.LAYER_OSD3)

    def draw_fatal(self, stage, exc, tx):
        if self.osd is None or not self.display_initialized:
            return False
        self.osd.clear()
        self.osd.draw_rectangle(
            0,
            0,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            color=(255, 0, 0, 0),
            fill=True,
        )
        self._draw_text(0, "STEEL BALL TRACKER FATAL", COLOR_BAD)
        self._draw_text(1, "version: " + APP_VERSION, COLOR_TITLE)
        stage_text = str(stage)
        self._draw_text(2, "stage: " + stage_text[:65], COLOR_WARN)
        self._draw_text(3, "       " + stage_text[65:130], COLOR_WARN)
        self._draw_text(4, "type: " + type(exc).__name__, COLOR_BAD)
        message = str(exc)
        self._draw_text(5, "error: " + message[:65], COLOR_BAD)
        self._draw_text(6, "       " + message[65:130], COLOR_BAD)
        if tx is not None:
            self._draw_text(
                8,
                "UART frame={} byte={} err={} short={} zero={} drop={}".format(
                    tx.sent_frames,
                    tx.bytes_sent,
                    tx.write_errors,
                    tx.short_writes,
                    tx.zero_writes,
                    tx.dropped_error_frames,
                ),
                COLOR_WARN,
            )
            self._draw_text(
                9,
                "UART last: " + str(tx.last_error)[:58],
                COLOR_WARN,
            )
        self._draw_text(12, "Stop manually after recording this page.", COLOR_TITLE)
        Display.show_image(self.osd, 0, 0, Display.LAYER_OSD3)
        return True

    def stop(self):
        if self.sensor_running and self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception as exc:
                print("sensor stop warning:", exc)
        self.sensor_running = False
        if self.display_initialized:
            try:
                Display.deinit()
            except Exception as exc:
                print("display deinit warning:", exc)
        self.display_initialized = False
        _sleep_ms(50)
        if self.media_initialized:
            try:
                MediaManager.deinit()
            except Exception as exc:
                print("media deinit warning:", exc)
        self.media_initialized = False
        self.sensor = None
        self.osd = None


class LatestFrameUartTx:
    """Two-frame, latest-wins software TX buffer with bounded writes."""

    def __init__(self, writer):
        if writer is None or not hasattr(writer, "write"):
            raise ValueError("UART writer must provide write()")
        self.writer = writer
        self.active_frame = None
        self.active_offset = 0
        self.pending_frame = None
        self.submitted_frames = 0
        self.sent_frames = 0
        self.replaced_frames = 0
        self.dropped_error_frames = 0
        self.bytes_sent = 0
        self.short_writes = 0
        self.zero_writes = 0
        self.write_errors = 0
        self.consecutive_errors = 0
        self.last_error = None

    def submit_latest(self, frame):
        if frame is None or len(frame) != BALL_STATE_FRAME_SIZE:
            raise ValueError("UART state frame must be exactly 48 bytes")
        self.submitted_frames += 1
        if self.active_frame is None:
            self.active_frame = frame
            self.active_offset = 0
        elif self.active_offset == 0:
            self.active_frame = frame
            self.replaced_frames += 1
        else:
            if self.pending_frame is not None:
                self.replaced_frames += 1
            self.pending_frame = frame

    def _promote_pending(self):
        if self.active_frame is None and self.pending_frame is not None:
            self.active_frame = self.pending_frame
            self.pending_frame = None
            self.active_offset = 0

    def _drop_active(self):
        self.active_frame = None
        self.active_offset = 0
        self.dropped_error_frames += 1
        self.consecutive_errors = 0

    def poll(self):
        self._promote_pending()
        if self.active_frame is None:
            return 0
        frame_size = len(self.active_frame)
        end = min(
            frame_size,
            self.active_offset + UART_MAX_WRITE_BYTES,
        )
        if self.active_offset == 0 and end == frame_size:
            chunk = self.active_frame
        else:
            chunk = self.active_frame[self.active_offset:end]
        try:
            written = self.writer.write(chunk)
        except Exception as exc:
            if _is_ide_interrupt(exc):
                raise KeyboardInterrupt()
            self.write_errors += 1
            self.consecutive_errors += 1
            self.last_error = "{}: {}".format(type(exc).__name__, exc)
            if self.consecutive_errors >= 3:
                self._drop_active()
            return 0
        if written is None:
            written = len(chunk)
        try:
            written = int(written)
        except Exception as exc:
            self.write_errors += 1
            self.consecutive_errors += 1
            self.last_error = "bad write result {}: {}".format(
                type(exc).__name__,
                exc,
            )
            if self.consecutive_errors >= 3:
                self._drop_active()
            return 0
        if written < 0 or written > len(chunk):
            self.write_errors += 1
            self.consecutive_errors += 1
            self.last_error = "write result {}/{} out of range".format(
                written,
                len(chunk),
            )
            if self.consecutive_errors >= 3:
                self._drop_active()
            return 0
        if written == 0:
            self.zero_writes += 1
            self.consecutive_errors += 1
            self.last_error = "UART.write returned 0/{}".format(len(chunk))
            if self.consecutive_errors >= 3:
                self._drop_active()
            return 0
        if written < len(chunk):
            self.short_writes += 1
            self.last_error = "short write {}/{}".format(
                written,
                len(chunk),
            )
        self.consecutive_errors = 0
        self.active_offset += written
        self.bytes_sent += written
        if self.active_offset >= frame_size:
            self.active_frame = None
            self.active_offset = 0
            self.sent_frames += 1
        return written

    def queued_bytes(self):
        total = 0
        if self.active_frame is not None:
            total += len(self.active_frame) - self.active_offset
        if self.pending_frame is not None:
            total += len(self.pending_frame)
        return total


def _open_uart():
    fpioa = FPIOA()
    fpioa.set_function(9, fpioa.UART1_TXD, ie=0, oe=1, pu=1)
    fpioa.set_function(10, fpioa.UART1_RXD, ie=1, oe=0, pu=1)
    return UART(UART.UART1, UART_BAUDRATE)


def _state_flags(candidate, state):
    flags = 0
    if candidate is not None:
        flags |= BALL_FLAG_RAW_DETECTED
        if candidate.get("edge_clipped"):
            flags |= BALL_FLAG_EDGE_CLIPPED
    if state.get("measurement_accepted"):
        flags |= BALL_FLAG_MEASUREMENT_ACCEPTED
    if state.get("track_valid"):
        flags |= BALL_FLAG_TRACK_VALID
    if state.get("predicted"):
        flags |= BALL_FLAG_PREDICTED
    if state.get("axis_locked"):
        flags |= BALL_FLAG_AXIS_LOCKED
    if state.get("outlier_rejected"):
        flags |= BALL_FLAG_OUTLIER_REJECTED
    if state.get("filter_reset"):
        flags |= BALL_FLAG_FILTER_RESET
    return flags


def _publish_state(tx, sequence, now_ms, candidate, state, inference_ms):
    frame = encode_ball_state_frame(
        sequence=sequence,
        timestamp_ms=now_ms,
        flags=_state_flags(candidate, state),
        state=state,
        candidate=candidate,
        inference_ms=inference_ms,
    )
    tx.submit_latest(frame)
    tx.poll()


def _print_pc_debug_status(
    sequence,
    candidate,
    state,
    inference_ms,
    tx,
    runtime_error,
):
    if candidate is None:
        raw_x = -1.0
        raw_y = -1.0
        confidence = 0.0
    else:
        raw_x = float(candidate.get("center_x", -1.0))
        raw_y = float(candidate.get("center_y", -1.0))
        confidence = float(candidate.get("confidence", 0.0))
    print(
        "PCDBG seq={} flags=0x{:02X} raw=({:.1f},{:.1f}) "
        "filt=({:.2f},{:.2f}) vel=({:.1f},{:.1f}) conf={:.3f} "
        "infer={}ms UART[f={},b={},q={},e={},s={},z={},d={}] "
        "uart_last={} runtime_last={}".format(
            int(sequence),
            _state_flags(candidate, state),
            raw_x,
            raw_y,
            float(state.get("x", 0.0)),
            float(state.get("y", 0.0)),
            float(state.get("vx", 0.0)),
            float(state.get("vy", 0.0)),
            confidence,
            int(inference_ms),
            tx.sent_frames,
            tx.bytes_sent,
            tx.queued_bytes(),
            tx.write_errors,
            tx.short_writes,
            tx.zero_writes,
            tx.dropped_error_frames,
            str(tx.last_error)[:48],
            str(runtime_error)[:48],
        )
    )


def _hold_fatal(camera, stage, exc, tx):
    print(
        "steel-ball tracker fatal at {}: {}: {}".format(
            stage,
            type(exc).__name__,
            exc,
        )
    )
    _print_exception(exc)
    drawn = False
    if camera is not None:
        try:
            drawn = bool(camera.draw_fatal(stage, exc, tx))
        except Exception as draw_exc:
            if _is_ide_interrupt(draw_exc):
                return
            print("fatal overlay warning:", draw_exc)
    if drawn:
        _fatal_hold_loop(tx)
        return

    if camera is not None:
        try:
            camera.stop()
        except Exception as stop_exc:
            if _is_ide_interrupt(stop_exc):
                return
            print("fatal fallback camera stop warning:", stop_exc)
    _emergency_hold_fatal(
        stage,
        exc,
        tx=tx,
        already_reported=True,
    )


def main(pc_debug=False):
    uart = None
    tx = None
    camera = None
    detector = None
    tracker = None
    stage = "camera_object_init"
    sequence = 0
    try:
        camera = CameraDisplay()
        stage = "camera_display_init"
        camera.start()
        stage = "model_check"
        _verify_model()
        stage = "uart_init"
        uart = _open_uart()
        tx = LatestFrameUartTx(uart)
        stage = "tracker_init"
        tracker = PipeMotionFilter()
        stage = "detector_init"
        detector = BallDetector()
        print(
            "steel-ball tracker started, version={}, model={}, UART={} 8N1, "
            "frame={}B, software_tx={}B, pc_debug={}".format(
                APP_VERSION,
                MODEL_PATH,
                UART_BAUDRATE,
                BALL_STATE_FRAME_SIZE,
                UART_SOFTWARE_BUFFER_BYTES,
                bool(pc_debug),
            )
        )

        last_overlay_ms = _ticks_ms() - OVERLAY_INTERVAL_MS
        last_pc_debug_ms = _ticks_ms() - PC_DEBUG_INTERVAL_MS
        last_frame_ms = _ticks_ms()
        frame_count = 0
        consecutive_errors = 0
        runtime_error_count = 0
        latched_runtime_error = None
        while True:
            _exitpoint()
            tx.poll()
            candidate = None
            inference_ms = 0
            now_ms = _ticks_ms()
            frame_interval_ms = max(
                0,
                _ticks_diff(now_ms, last_frame_ms),
            )
            last_frame_ms = now_ms
            try:
                stage = "camera_capture"
                input_array = camera.get_frame()
                stage = "model_inference"
                inference_start_ms = _ticks_ms()
                raw_result = detector.run(input_array)
                finished_ms = _ticks_ms()
                inference_ms = max(
                    0,
                    _ticks_diff(finished_ms, inference_start_ms),
                )
                stage = "primary_selection"
                candidate = select_highest_confidence_ball(
                    raw_result,
                    MODEL_INPUT_WIDTH,
                    MODEL_INPUT_HEIGHT,
                )
                stage = "motion_filter"
                state = tracker.update(candidate, finished_ms)
                sequence = (sequence + 1) & 0xFFFF
                stage = "uart_publish"
                _publish_state(
                    tx,
                    sequence,
                    finished_ms,
                    candidate,
                    state,
                    inference_ms,
                )
                consecutive_errors = 0
                now_ms = finished_ms
            except Exception as frame_exc:
                if _is_ide_interrupt(frame_exc):
                    raise KeyboardInterrupt()
                failed_stage = stage
                consecutive_errors += 1
                runtime_error_count += 1
                latched_runtime_error = "#{} {} {}: {}".format(
                    runtime_error_count,
                    failed_stage,
                    type(frame_exc).__name__,
                    frame_exc,
                )
                print(
                    "steel-ball frame error {}/{} at {}: {}".format(
                        consecutive_errors,
                        MAX_CONSECUTIVE_FRAME_ERRORS,
                        failed_stage,
                        latched_runtime_error,
                    )
                )
                now_ms = _ticks_ms()
                candidate = None
                state = tracker.update(None, now_ms)
                sequence = (sequence + 1) & 0xFFFF
                stage = "uart_error_publish"
                _publish_state(
                    tx,
                    sequence,
                    now_ms,
                    None,
                    state,
                    0,
                )
                if consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                    stage = "frame_errors/" + str(failed_stage)
                    raise RuntimeError(
                        "{} consecutive frame errors; last {}".format(
                            consecutive_errors,
                            latched_runtime_error,
                        )
                    )

            if _ticks_diff(now_ms, last_overlay_ms) >= OVERLAY_INTERVAL_MS:
                stage = "overlay"
                camera.draw_status(
                    candidate,
                    state,
                    inference_ms,
                    frame_interval_ms,
                    tx,
                    runtime_error=latched_runtime_error,
                )
                last_overlay_ms = now_ms
            if (
                pc_debug
                and _ticks_diff(now_ms, last_pc_debug_ms)
                >= PC_DEBUG_INTERVAL_MS
            ):
                stage = "pc_debug_report"
                _print_pc_debug_status(
                    sequence,
                    candidate,
                    state,
                    inference_ms,
                    tx,
                    latched_runtime_error,
                )
                last_pc_debug_ms = now_ms
            frame_count = (frame_count + 1) & 0xFFFFFFFF
            if frame_count % GC_INTERVAL_FRAMES == 0:
                stage = "gc_collect"
                gc.collect()
            stage = "runtime"
    except KeyboardInterrupt:
        print("steel-ball tracker stopped by IDE/user")
    except Exception as exc:
        if _is_ide_interrupt(exc):
            print("steel-ball tracker stopped by IDE/user")
        else:
            _hold_fatal(camera, stage, exc, tx)
    finally:
        if detector is not None:
            try:
                detector.deinit()
            except Exception as exc:
                print("detector deinit warning:", exc)
        if camera is not None:
            camera.stop()
        if uart is not None:
            try:
                uart.deinit()
            except Exception as exc:
                print("UART deinit warning:", exc)
        gc.collect()


def run(pc_debug=False):
    if _BOOT_IMPORT_ERROR is not None:
        _emergency_hold_fatal(
            _BOOT_IMPORT_STAGE or "boot_import",
            _BOOT_IMPORT_ERROR,
        )
        return
    main(pc_debug=pc_debug)


if __name__ == "__main__":
    run()
