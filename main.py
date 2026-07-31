"""Standalone K230 steel-ball tracker with filtered UART control/state.

This entry has no command handshake or task supervisor. It starts the camera,
detector, constrained motion filter, UART and lightweight overlay once, then
accepts bounded set-reference commands and publishes the newest state frame.
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
TOUCH = None
Display = None
MediaManager = None
Sensor = None
CAM_CHN_ID_0 = 0
CAM_CHN_ID_2 = 2
PIXEL_FORMAT_RGB_565 = 0
PIXEL_FORMAT_RGB_888_PLANAR = 0
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 0
PipeMotionFilter = None
normalize_postprocessed_ball_candidates = None
BallStateFeedback = None
RoiDarkBlobDetector = None
EdgeHoughCircleDetector = None
WifiVideoService = None
DisabledWifiVideoService = None
encode_ball_state_frame = None
ReferenceCommandParser = None
x_pixel_for_position_cm = None
get_x_calibration_lut = None
X_CALIBRATION_MEASURED = "measured_distortion"
X_CALIBRATION_PINHOLE = "pinhole"

BALL_FLAG_RAW_DETECTED = 0x01
BALL_FLAG_MEASUREMENT_ACCEPTED = 0x02
BALL_FLAG_TRACK_VALID = 0x04
BALL_FLAG_PREDICTED = 0x08
BALL_FLAG_FEEDBACK_VALID = 0x10
BALL_FLAG_OUTLIER_REJECTED = 0x20
BALL_FLAG_EDGE_CLIPPED = 0x40
BALL_FLAG_FEEDBACK_SATURATED = 0x80
BALL_STATE_FRAME_SIZE = 64
BALL_REFERENCE_FLAG_VALID = 0x0001
BALL_REFERENCE_FLAG_SOURCE_PRESET = 0x0002
BALL_REFERENCE_FLAG_SOURCE_MCU = 0x0008

_BOOT_IMPORT_ERROR = None
_BOOT_IMPORT_STAGE = None
_TOUCH_IMPORT_ERROR = None
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
    try:
        from machine import TOUCH
    except Exception as _touch_import_exc:
        _TOUCH_IMPORT_ERROR = _touch_import_exc
    _BOOT_IMPORT_STAGE = "import_media_display"
    from media.display import Display
    _BOOT_IMPORT_STAGE = "import_media_manager"
    from media.media import MediaManager
    _BOOT_IMPORT_STAGE = "import_media_sensor"
    from media.sensor import (
        CAM_CHN_ID_0,
        CAM_CHN_ID_2,
        PIXEL_FORMAT_RGB_565,
        PIXEL_FORMAT_RGB_888_PLANAR,
        PIXEL_FORMAT_YUV_SEMIPLANAR_420,
        Sensor,
    )

    _BOOT_IMPORT_STAGE = "import_ball_protocol"
    from ball_protocol import (
        BALL_FLAG_FEEDBACK_SATURATED,
        BALL_FLAG_FEEDBACK_VALID,
        BALL_FLAG_EDGE_CLIPPED,
        BALL_FLAG_MEASUREMENT_ACCEPTED,
        BALL_FLAG_OUTLIER_REJECTED,
        BALL_FLAG_PREDICTED,
        BALL_FLAG_RAW_DETECTED,
        BALL_FLAG_TRACK_VALID,
        BALL_REFERENCE_FLAG_SOURCE_PRESET,
        BALL_REFERENCE_FLAG_SOURCE_MCU,
        BALL_REFERENCE_FLAG_VALID,
        BALL_STATE_FRAME_SIZE,
        ReferenceCommandParser,
        encode_ball_state_frame,
    )
    _BOOT_IMPORT_STAGE = "import_ball_tracking"
    from ball_tracking import (
        PipeMotionFilter,
        normalize_postprocessed_ball_candidates,
    )
    _BOOT_IMPORT_STAGE = "import_ball_control"
    from ball_control import BallStateFeedback
    _BOOT_IMPORT_STAGE = "import_ball_calibration"
    from ball_calibration import (
        X_CALIBRATION_MEASURED,
        X_CALIBRATION_PINHOLE,
        get_x_calibration_lut,
        x_pixel_for_position_cm,
    )
    _BOOT_IMPORT_STAGE = "import_ball_roi_detector"
    from ball_roi_detector import RoiDarkBlobDetector
    _BOOT_IMPORT_STAGE = "import_ball_edge_hough_detector"
    from ball_edge_hough_detector import EdgeHoughCircleDetector
    _BOOT_IMPORT_STAGE = "import_wifi_video_service"
    from wifi_video_service import DisabledWifiVideoService, WifiVideoService
    _BOOT_IMPORT_STAGE = None
except Exception as _boot_exc:
    _BOOT_IMPORT_ERROR = _boot_exc


APP_VERSION = "proj2-1.60"

# Select one detector backend: yolov8n, yolov8, roi_blob, or edge_hough.
# The generic "yolov8" artifact has no trustworthy subtype metadata.
MODEL_BACKEND = "edge_hough"
MODEL_BACKENDS = {
    "yolov8n": {
        "display_name": "YOLOv8n",
        "path": "/sdcard/task_assets/ball_det/ball_det_yolov8n_320.kmodel",
        "expected_output": (1, 5, 2100),
    },
    "yolov8": {
        "display_name": "YOLOv8 custom",
        "path": "/sdcard/task_assets/ball_det/ball_det_yolov8_320.kmodel",
        "expected_output": (1, 5, 2100),
    },
    "roi_blob": {
        "display_name": "ROI dark blob",
        "native_roi": True,
    },
    "edge_hough": {
        "display_name": "Canny Hough circle",
        "native_hough": True,
    },
}
MODEL_INPUT_WIDTH = 320
MODEL_INPUT_HEIGHT = 320
CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45
POSTPROCESS_MAX_BOXES = 4

# Native ROI dark-blob backend. The ROI follows the calibrated pipe axis.
ROI_HALF_WIDTH_PX = 22.0
ROI_THRESHOLD_SAMPLE_HALF_WIDTH_PX = 10.0
ROI_ENDPOINT_MARGIN_PX = 4.0
ROI_MIN_DIAMETER_PX = 3.0
ROI_MAX_DIAMETER_PX = 26.0
ROI_TARGET_DIAMETER_PX = 11.0
ROI_MIN_PIXELS = 6
ROI_FRAGMENT_MIN_PIXELS = 3
ROI_FRAGMENT_MERGE_GAP_PX = 3.0
ROI_TRACKED_MIN_ASPECT_RATIO = 0.16
ROI_TRACKED_MIN_FILL_RATIO = 0.08
ROI_MOTION_BLUR_SECONDS = 0.025
ROI_MOTION_BLUR_MIN_EXTRA_PX = 8.0
ROI_MOTION_BLUR_MAX_EXTRA_PX = 24.0
ROI_THRESHOLD_RATIO = 0.68
ROI_THRESHOLD_L_MIN = 20
ROI_THRESHOLD_L_MAX = 62
ROI_THRESHOLD_UPDATE_FRAMES = 6

# Native Canny-edge Hough-circle backend. cv_lite performs both expensive
# stages in native code; Python only filters a bounded candidate list.
# Keep the expensive Hough image crop wider than the logical pipe-center gate.
# With the current horizontal axis this produces an exact 320x80 ROI, while
# the separate 28 px gate rejects most pipe-wall/background circles.
HOUGH_CROP_HALF_WIDTH_PX = 40.0
HOUGH_AXIS_CENTER_LIMIT_PX = 28.0
HOUGH_ENDPOINT_MARGIN_PX = 4.0
HOUGH_DP = 1
HOUGH_MIN_CENTER_DISTANCE_PX = 8
HOUGH_CANNY_HIGH_THRESHOLD = 80
HOUGH_ACCUMULATOR_THRESHOLD = 9
HOUGH_MIN_RADIUS_PX = 3
HOUGH_MAX_RADIUS_PX = 9
HOUGH_TARGET_RADIUS_PX = 5.5
HOUGH_MAX_RAW_CIRCLES = 24
HOUGH_CLUSTER_CENTER_FACTOR = 1.0
HOUGH_CLUSTER_RADIUS_TOLERANCE_PX = 2.0
HOUGH_TRACKED_RADIUS_TOLERANCE_PX = 3.0
HOUGH_AMBIGUOUS_SPREAD_RADIUS_FACTOR = 0.65

# Image-space pipe geometry used only by detection and pixel-domain tracking.
# Physical conversion is isolated in ball_calibration.py after tracking.
PIPE_NEGATIVE_END_X_PX = 10.0
PIPE_NEGATIVE_END_Y_PX = 160.0
PIPE_POSITIVE_END_X_PX = 310.0
PIPE_POSITIVE_END_Y_PX = 160.0

# Physical-coordinate calibration used by position, velocity, acceleration,
# feedback, Y center scale and the optional screen ruler. "measured_distortion"
# is the current physical-ruler fit; "pinhole" preserves the original model.
X_CALIBRATION_MODE = X_CALIBRATION_MEASURED

# K230 publishes a continuous camera-derived PI-D term. The control equation
# never switches on tracker motion labels. Small soft deadbands are applied to
# position and velocity independently, so dynamic feedback is zero only when
# both values are small.
FEEDBACK_KP = 8.0
FEEDBACK_KD = 12.0
FEEDBACK_KI = 12.0
FEEDBACK_KA = 0.0
FEEDBACK_ACCEL_LIMIT_CM_S2 = 140.0
FEEDBACK_POSITION_DEADBAND_CM = 0.10
FEEDBACK_VELOCITY_DEADBAND_CM_S = 0.35
FEEDBACK_POSITION_FILTER_TAU_S = 0.025
FEEDBACK_VELOCITY_FILTER_TAU_S = 0.050
FEEDBACK_OUTPUT_FILTER_TAU_S = 0.025
FEEDBACK_INTEGRAL_LIMIT_CM_S2 = 80.0
FEEDBACK_INTEGRAL_ERROR_LIMIT_CM = 1.0
# Continuous static-friction compensation. It ramps linearly from zero over
# the first 0.20 cm beyond the position soft deadband, then stays at 20 cm/s2.
# Measure the real breakaway threshold and keep this at roughly 70--85% of it.
FEEDBACK_INTEGRAL_LEAK_PER_S = 0.0
FEEDBACK_INTEGRAL_UNWIND_GAIN_S2 = 20.0
FEEDBACK_BREAKAWAY_ACCEL_CM_S2 = 20.0
FEEDBACK_BREAKAWAY_TRANSITION_CM = 0.20
# Absolute calibrated X coordinate used as the ball equilibrium origin.
# A future validated algorithm/MCU command may call
# BallStateFeedback.set_reference_position_cm() to update it at runtime.
FEEDBACK_REFERENCE_POSITION_CM = 0.0
TRACK_HOLD_MAX_FRAMES = 8
TRACK_HOLD_MAX_MS = 300

# Fixed-pipe asymmetric tracker. Axial motion gets a fast response outside a
# small noise band; lateral position and apparent diameter remain strongly
# smoothed because the camera and pipe geometry are fixed.
TRACK_AXIAL_FAST_ALPHA = 0.96
TRACK_AXIAL_QUIET_ALPHA = 0.18
TRACK_AXIAL_RESPONSE_START_PX = 0.25
TRACK_AXIAL_RESPONSE_FULL_PX = 1.50
TRACK_VELOCITY_BETA = 0.30
TRACK_VELOCITY_FAST_BETA = 0.85
TRACK_VELOCITY_NOISE_PX = 0.90
TRACK_VELOCITY_QUIET_DECAY = 0.35
TRACK_VELOCITY_QUIET_TIMEOUT_MS = 120
TRACK_LATERAL_ALPHA = 0.08
TRACK_DIAMETER_ALPHA = 0.08
TRACK_DIAMETER_MAX_STEP_PX = 1.0
TRACK_MOTION_START_PX = 2.5
TRACK_MOTION_IMMEDIATE_PX = 14.0
TRACK_MOTION_CONFIRM_FRAMES = 2
TRACK_MOTION_STOP_FRAMES = 4
TRACK_MOTION_DIRECTION_EPSILON_PX = 0.75
TRACK_AMBIGUOUS_POSITION_ALPHA = 0.10
TRACK_QUIET_POSITION_ALPHA = 0.12
TRACK_QUIET_VELOCITY_DECAY = 0.20
TRACK_QUIET_GATE_DIAMETER_FACTOR = 1.25
TRACK_QUIET_GATE_MIN_PX = 10.0
TRACK_MOTION_MIN_QUALITY = 0.62
TRACK_MOTION_MIN_SUPPORT = 2
TRACK_ASSOCIATION_AMBIGUITY_DIAMETER_FACTOR = 0.75

CAMERA_FPS = 60
# Leave as None to preserve the camera firmware's original exposure behavior.
# Set a positive value in microseconds (for example 5000) to disable automatic
# exposure and use a fixed exposure time. Shorter exposure reduces motion blur
# but needs more illumination. This is sensor exposure, not image brightening.
CAMERA_EXPOSURE_US = None

# Optional CH1 -> hardware H.264 -> RTSP service. Real credentials belong in
# ignored /sdcard/wifi_video_secrets.py as WIFI_VIDEO_SSID/WIFI_VIDEO_PASSWORD.
# With no secrets file the service stays off and the vision path is unchanged.
WIFI_VIDEO_ENABLED = False
# Conservative debug-video profile. It reduces IDR/RTP fragmentation peaks and
# Python/RTSP backpressure without changing CH2 detection, control or UART.
# After a long stable test, 512x288/30/1200-1600 may be tried again as an
# explicit quality profile rather than the reliability default.
WIFI_VIDEO_WIDTH = 320
WIFI_VIDEO_HEIGHT = 240
WIFI_VIDEO_FPS = 15
WIFI_VIDEO_BIT_RATE_KBPS = 600
WIFI_VIDEO_GOP_FRAMES = 15
WIFI_VIDEO_RTSP_PORT = 8554
WIFI_VIDEO_SESSION_NAME = "k230"
WIFI_VIDEO_DISCOVERY_PORT = 19124
# Discovery is best-effort and deliberately sparse. A non-blocking beacon
# avoids a periodic pause in the same worker that drains encoded VENC packs.
WIFI_VIDEO_BEACON_INTERVAL_MS = 10000
WIFI_VIDEO_CONNECT_TIMEOUT_MS = 30000
WIFI_VIDEO_STREAM_STALL_TIMEOUT_MS = 3000
WIFI_VIDEO_OUTPUT_BUFFERS = 8
# Reliability default: copy each encoded pack before ReleaseStream. The
# physical-address API remains opt-in because its asynchronous buffer ownership
# is not documented and long-run testing showed H.264 corruption with it.
WIFI_VIDEO_PREFER_ZERO_COPY = False
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
# Optional one-shot physical X ruler on a dedicated OSD layer. It does not
# redraw in the frame loop. Disable it to retain the original single-OSD memory
# footprint. Tick spacing supports 0.5 cm or 1.0 cm; labels stay at 2.5 cm.
SHOW_PHYSICAL_X_RULER = True
PHYSICAL_X_RULER_TICK_CM = 0.5
PHYSICAL_X_RULER_Y = 320
# Draw close to the camera frame rate. The former 100 ms interval made a
# correctly updated 30 FPS state look about three frames late on screen.
OVERLAY_INTERVAL_MS = 30
PC_DEBUG_INTERVAL_MS = 500
GC_INTERVAL_FRAMES = 60
MAX_CONSECUTIVE_FRAME_ERRORS = 3

UART_BAUDRATE = 512000
UART_MAX_WRITE_BYTES = 64
UART_MAX_READ_BYTES = 32

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
COLOR_RULER = (255, 255, 255, 255)
COLOR_RULER_MAJOR = (255, 0, 255, 255)
COLOR_RULER_SHADOW = (180, 0, 0, 0)
STATUS_PANEL_HEIGHT = 230

# Optional live PID tuning panel. TOUCH(0).read(1) is polled once per frame;
# each press edge changes one gain once, so holding a button cannot run away.
# Changes are intentionally RAM-only and revert to the defaults above at boot.
TOUCH_PID_ENABLED = True
TOUCH_PID_STEP_KP = 0.1
TOUCH_PID_STEP_KD = 0.1
TOUCH_PID_STEP_KI = 0.1
TOUCH_PID_KP_RANGE = (0.0, 30.0)
# Schema-6 encodes Kd as u8/10, so 25.5 is the largest exactly reportable
# value. Keeping the runtime bound aligned avoids MCU diagnostic mismatch.
TOUCH_PID_KD_RANGE = (0.0, 25.5)
TOUCH_PID_KI_RANGE = (0.0, 30.0)
TOUCH_PID_DEBOUNCE_MS = 120
TOUCH_PID_RETRY_MS = 2000
TOUCH_PID_PANEL_TOP = 350
TOUCH_PID_ROW_HEIGHT = 42
TOUCH_PID_MINUS_X = 430
TOUCH_PID_PLUS_X = 540
TOUCH_PID_BUTTON_WIDTH = 82
TOUCH_PID_BUTTON_HEIGHT = 34
COLOR_TOUCH_BUTTON = (210, 24, 24, 24)


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


class TouchPidTuner:
    """Non-blocking TOUCH(0) press-edge gain adjustment."""

    def __init__(self, controller):
        self.controller = controller
        self.touch = None
        self.enabled = bool(TOUCH_PID_ENABLED)
        self.state = "OFF" if not self.enabled else "INIT"
        self.last_press_ms = None
        self.retry_after_ms = 0
        self.error_count = 0
        self.change_count = 0
        self.last_error = None
        self.last_action = ""
        if self.enabled:
            self._try_init(_ticks_ms())

    def _set_error(self, stage, exc, now_ms):
        self.touch = None
        self.state = "ERR"
        self.error_count += 1
        self.retry_after_ms = int(now_ms) + int(TOUCH_PID_RETRY_MS)
        self.last_error = "touch #{} {} {}: {}".format(
            self.error_count,
            stage,
            type(exc).__name__,
            exc,
        )
        print(self.last_error)

    def _try_init(self, now_ms):
        if TOUCH is None:
            exc = _TOUCH_IMPORT_ERROR
            if exc is None:
                exc = RuntimeError("machine.TOUCH unavailable")
            self._set_error("init", exc, now_ms)
            return
        try:
            self.touch = TOUCH(0)
            self.state = "RUN"
            self.last_error = None
        except Exception as exc:
            self._set_error("init", exc, now_ms)

    @staticmethod
    def _inside(value, start, length):
        return int(start) <= int(value) < int(start) + int(length)

    def _adjust(self, name, delta):
        if name == "kp":
            low, high = TOUCH_PID_KP_RANGE
            value = max(low, min(high, self.controller.kp + delta))
            self.controller.set_gains(kp=value)
        elif name == "kd":
            low, high = TOUCH_PID_KD_RANGE
            value = max(low, min(high, self.controller.kd + delta))
            self.controller.set_gains(kd=value)
        else:
            low, high = TOUCH_PID_KI_RANGE
            value = max(low, min(high, self.controller.ki + delta))
            self.controller.set_gains(ki=value)
        self.change_count += 1
        self.last_action = "{}={:.1f}".format(name.upper(), value)
        print("touch PID {} changes={}".format(
            self.last_action,
            self.change_count,
        ))

    def _handle_press(self, x, y):
        if not self._inside(y, TOUCH_PID_PANEL_TOP, 3 * TOUCH_PID_ROW_HEIGHT):
            return
        row = (int(y) - TOUCH_PID_PANEL_TOP) // TOUCH_PID_ROW_HEIGHT
        if row == 0:
            name = "kp"
            step = TOUCH_PID_STEP_KP
        elif row == 1:
            name = "kd"
            step = TOUCH_PID_STEP_KD
        else:
            name = "ki"
            step = TOUCH_PID_STEP_KI
        if self._inside(x, TOUCH_PID_MINUS_X, TOUCH_PID_BUTTON_WIDTH):
            self._adjust(name, -float(step))
        elif self._inside(x, TOUCH_PID_PLUS_X, TOUCH_PID_BUTTON_WIDTH):
            self._adjust(name, float(step))

    def poll(self, now_ms):
        """Poll once; all failures are retained for LCD/PC diagnostics."""
        if not self.enabled:
            return
        now_ms = int(now_ms)
        if self.touch is None:
            if _ticks_diff(now_ms, self.retry_after_ms) >= 0:
                self._try_init(now_ms)
            return
        try:
            points = self.touch.read(1)
            if points is not None and len(points):
                point = points[0]
                event = int(point.event)
                event_down = getattr(TOUCH, "EVENT_DOWN", 2)
                debounce_ready = bool(
                    self.last_press_ms is None
                    or _ticks_diff(now_ms, self.last_press_ms)
                    >= TOUCH_PID_DEBOUNCE_MS
                )
                if event == event_down and debounce_ready:
                    self._handle_press(int(point.x), int(point.y))
                    self.last_press_ms = now_ms
        except Exception as exc:
            self._set_error("read", exc, now_ms)

    def diagnostic_text(self):
        text = "{} n={}".format(self.state, self.change_count)
        if self.last_action:
            text += " " + self.last_action
        if self.last_error is not None:
            text += " " + str(self.last_error)[:44]
        return text


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


def _selected_model_config():
    config = MODEL_BACKENDS.get(MODEL_BACKEND)
    if config is None:
        raise RuntimeError(
            "unknown MODEL_BACKEND {}; use yolov8n, yolov8, roi_blob or "
            "edge_hough".format(
                MODEL_BACKEND
            )
        )
    if config.get("native_roi") or config.get("native_hough"):
        return config
    if not config.get("path") or not config.get("expected_output"):
        raise RuntimeError("incomplete model backend: " + MODEL_BACKEND)
    return config


def _detector_diagnostic_text(detector, candidate):
    if detector.model_backend == "edge_hough":
        return detector.diagnostic_text(candidate)
    return detector.diagnostic_text()


def _create_wifi_video_service():
    if not WIFI_VIDEO_ENABLED:
        return DisabledWifiVideoService("config disabled")
    try:
        import wifi_video_secrets as secrets
    except ImportError:
        return DisabledWifiVideoService("no secrets file")
    except Exception as exc:
        print(
            "wifi video secrets warning: {}: {}".format(
                type(exc).__name__,
                exc,
            )
        )
        return DisabledWifiVideoService("secrets import error")

    ssid = getattr(secrets, "WIFI_VIDEO_SSID", "")
    password = getattr(secrets, "WIFI_VIDEO_PASSWORD", "")
    if not ssid:
        return DisabledWifiVideoService("SSID empty")
    return WifiVideoService(
        ssid=ssid,
        password=password,
        width=WIFI_VIDEO_WIDTH,
        height=WIFI_VIDEO_HEIGHT,
        source_fps=CAMERA_FPS,
        stream_fps=WIFI_VIDEO_FPS,
        bit_rate_kbps=WIFI_VIDEO_BIT_RATE_KBPS,
        gop_frames=WIFI_VIDEO_GOP_FRAMES,
        rtsp_port=WIFI_VIDEO_RTSP_PORT,
        session_name=WIFI_VIDEO_SESSION_NAME,
        discovery_port=WIFI_VIDEO_DISCOVERY_PORT,
        beacon_interval_ms=WIFI_VIDEO_BEACON_INTERVAL_MS,
        connect_timeout_ms=WIFI_VIDEO_CONNECT_TIMEOUT_MS,
        stream_stall_timeout_ms=WIFI_VIDEO_STREAM_STALL_TIMEOUT_MS,
        output_buffers=WIFI_VIDEO_OUTPUT_BUFFERS,
        prefer_zero_copy=WIFI_VIDEO_PREFER_ZERO_COPY,
    )


def _verify_model():
    config = _selected_model_config()
    if config.get("native_roi") or config.get("native_hough"):
        return
    model_path = config["path"]
    try:
        stat = os.stat(model_path)
    except Exception as exc:
        raise RuntimeError(
            "model stat failed {}: {}: {}".format(
                model_path,
                type(exc).__name__,
                exc,
            )
        )
    if int(stat[6]) <= 0:
        raise RuntimeError("empty model: " + model_path)


class BallDetector:
    """Minimal single-output YOLOv8 KPU path using verified CanMV APIs."""

    def __init__(self):
        config = _selected_model_config()
        self.model_backend = MODEL_BACKEND
        self.model_display_name = config["display_name"]
        self.model_path = config["path"]
        self.expected_output = config["expected_output"]
        self.pipeline_name = "native_nms_top4"
        self.actual_output = None
        self.kpu = None
        self.ai2d = None
        self.ai2d_builder = None
        self.ai2d_output_tensor = None
        self.ai2d_input_tensor = None
        self.output_shape_checked = False
        self.preprocess_ms = 0
        self.kpu_ms = 0
        self.postprocess_ms = 0
        self.timing_labels = ("ai2d", "kpu", "post")
        self.last_detection_count = 0
        try:
            self.kpu = nn.kpu()
            self.kpu.load_kmodel(self.model_path)
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
        self.actual_output = actual
        if actual != self.expected_output:
            raise RuntimeError(
                "model {} output {} != {}".format(
                    self.model_backend,
                    actual,
                    self.expected_output,
                )
            )
        self.output_shape_checked = True

    def run(self, input_array):
        self.preprocess_ms = 0
        self.kpu_ms = 0
        self.postprocess_ms = 0
        self.last_detection_count = 0
        preprocess_start_ms = _ticks_ms()
        self.ai2d_input_tensor = nn.from_numpy(input_array)
        self.ai2d_builder.run(
            self.ai2d_input_tensor,
            self.ai2d_output_tensor,
        )
        preprocess_finished_ms = _ticks_ms()
        self.preprocess_ms = max(
            0,
            _ticks_diff(preprocess_finished_ms, preprocess_start_ms),
        )
        self.kpu.set_input_tensor(0, self.ai2d_output_tensor)
        self.kpu.run()
        kpu_finished_ms = _ticks_ms()
        self.kpu_ms = max(
            0,
            _ticks_diff(kpu_finished_ms, preprocess_finished_ms),
        )
        output_tensor = self.kpu.get_output_tensor(0)
        try:
            output_array = output_tensor.to_numpy()
        finally:
            del output_tensor
        self.ai2d_input_tensor = None
        if not self.output_shape_checked:
            self._check_output_shape(output_array)
        candidates = output_array[0].transpose()
        raw_result = aidemo.yolov8_det_postprocess(
            candidates.copy(),
            [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
            [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
            [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
            1,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
            POSTPROCESS_MAX_BOXES,
        )
        candidates = normalize_postprocessed_ball_candidates(
            raw_result,
            CONFIDENCE_THRESHOLD,
            MODEL_INPUT_WIDTH,
            MODEL_INPUT_HEIGHT,
            POSTPROCESS_MAX_BOXES,
        )
        self.last_detection_count = len(candidates)
        self.postprocess_ms = max(
            0,
            _ticks_diff(_ticks_ms(), kpu_finished_ms),
        )
        return candidates

    def diagnostic_text(self):
        return "YOLO det={}".format(int(self.last_detection_count))

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


def _create_detector():
    config = _selected_model_config()
    if config.get("native_hough"):
        return EdgeHoughCircleDetector(
            frame_width=MODEL_INPUT_WIDTH,
            frame_height=MODEL_INPUT_HEIGHT,
            negative_end_x_px=PIPE_NEGATIVE_END_X_PX,
            negative_end_y_px=PIPE_NEGATIVE_END_Y_PX,
            positive_end_x_px=PIPE_POSITIVE_END_X_PX,
            positive_end_y_px=PIPE_POSITIVE_END_Y_PX,
            crop_half_width_px=HOUGH_CROP_HALF_WIDTH_PX,
            axis_center_limit_px=HOUGH_AXIS_CENTER_LIMIT_PX,
            endpoint_margin_px=HOUGH_ENDPOINT_MARGIN_PX,
            dp=HOUGH_DP,
            min_center_distance_px=HOUGH_MIN_CENTER_DISTANCE_PX,
            canny_high_threshold=HOUGH_CANNY_HIGH_THRESHOLD,
            accumulator_threshold=HOUGH_ACCUMULATOR_THRESHOLD,
            min_radius_px=HOUGH_MIN_RADIUS_PX,
            max_radius_px=HOUGH_MAX_RADIUS_PX,
            target_radius_px=HOUGH_TARGET_RADIUS_PX,
            max_raw_circles=HOUGH_MAX_RAW_CIRCLES,
            max_candidates=POSTPROCESS_MAX_BOXES,
            cluster_center_factor=HOUGH_CLUSTER_CENTER_FACTOR,
            cluster_radius_tolerance_px=(
                HOUGH_CLUSTER_RADIUS_TOLERANCE_PX
            ),
            tracked_radius_tolerance_px=(
                HOUGH_TRACKED_RADIUS_TOLERANCE_PX
            ),
            ambiguous_spread_radius_factor=(
                HOUGH_AMBIGUOUS_SPREAD_RADIUS_FACTOR
            ),
        )
    if config.get("native_roi"):
        return RoiDarkBlobDetector(
            frame_width=MODEL_INPUT_WIDTH,
            frame_height=MODEL_INPUT_HEIGHT,
            negative_end_x_px=PIPE_NEGATIVE_END_X_PX,
            negative_end_y_px=PIPE_NEGATIVE_END_Y_PX,
            positive_end_x_px=PIPE_POSITIVE_END_X_PX,
            positive_end_y_px=PIPE_POSITIVE_END_Y_PX,
            roi_half_width_px=ROI_HALF_WIDTH_PX,
            threshold_sample_half_width_px=ROI_THRESHOLD_SAMPLE_HALF_WIDTH_PX,
            endpoint_margin_px=ROI_ENDPOINT_MARGIN_PX,
            min_diameter_px=ROI_MIN_DIAMETER_PX,
            max_diameter_px=ROI_MAX_DIAMETER_PX,
            target_diameter_px=ROI_TARGET_DIAMETER_PX,
            min_pixels=ROI_MIN_PIXELS,
            fragment_min_pixels=ROI_FRAGMENT_MIN_PIXELS,
            fragment_merge_gap_px=ROI_FRAGMENT_MERGE_GAP_PX,
            tracked_min_aspect_ratio=ROI_TRACKED_MIN_ASPECT_RATIO,
            tracked_min_fill_ratio=ROI_TRACKED_MIN_FILL_RATIO,
            motion_blur_seconds=ROI_MOTION_BLUR_SECONDS,
            motion_blur_min_extra_px=ROI_MOTION_BLUR_MIN_EXTRA_PX,
            motion_blur_max_extra_px=ROI_MOTION_BLUR_MAX_EXTRA_PX,
            threshold_ratio=ROI_THRESHOLD_RATIO,
            threshold_l_min=ROI_THRESHOLD_L_MIN,
            threshold_l_max=ROI_THRESHOLD_L_MAX,
            threshold_update_frames=ROI_THRESHOLD_UPDATE_FRAMES,
            max_candidates=POSTPROCESS_MAX_BOXES,
        )
    return BallDetector()


class CameraDisplay:
    """Own camera video, detector input and dynamic/static OSD layers."""

    def __init__(self):
        self.sensor = None
        self.osd = None
        self.display_initialized = False
        self.media_initialized = False
        self.sensor_running = False
        self.ruler_drawn = False
        self.exposure_us = None
        self.wifi_video = None
        self.x_calibration_lut = get_x_calibration_lut(X_CALIBRATION_MODE)
        config = _selected_model_config()
        self.native_image_input = bool(
            config.get("native_roi") or config.get("native_hough")
        )

    def start(self, wifi_video=None):
        self.wifi_video = wifi_video
        sensor = Sensor(fps=CAMERA_FPS)
        self.sensor = sensor
        sensor.reset()
        if CAMERA_EXPOSURE_US is not None:
            try:
                exposure_us = int(CAMERA_EXPOSURE_US)
            except Exception:
                raise ValueError("CAMERA_EXPOSURE_US must be None or positive")
            if exposure_us <= 0:
                raise ValueError("CAMERA_EXPOSURE_US must be positive")
            sensor.auto_exposure(False)
            sensor.exposure(exposure_us)
            self.exposure_us = exposure_us
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
            PIXEL_FORMAT_RGB_565
            if self.native_image_input
            else PIXEL_FORMAT_RGB_888_PLANAR,
            chn=CAM_CHN_ID_2,
        )
        if wifi_video is not None:
            wifi_video.prepare(sensor)
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
            osd_num=2 if SHOW_PHYSICAL_X_RULER else 1,
            to_ide=True,
        )
        self.display_initialized = True
        MediaManager.init()
        self.media_initialized = True
        if wifi_video is not None:
            wifi_video.bind()
        sensor.run()
        self.sensor_running = True
        if SHOW_PHYSICAL_X_RULER:
            self._draw_physical_x_ruler()

    def get_frame(self):
        frame = self.sensor.snapshot(chn=CAM_CHN_ID_2)
        if frame is None:
            raise RuntimeError("camera returned no detector frame")
        if self.native_image_input:
            return frame
        return frame.to_numpy_ref()

    def exposure_text(self):
        if self.exposure_us is None:
            return "EXP=SYS"
        return "EXP={}us".format(int(self.exposure_us))

    @staticmethod
    def _screen_x(value):
        return int(float(value) * DISPLAY_WIDTH / MODEL_INPUT_WIDTH)

    @staticmethod
    def _screen_y(value):
        return int(float(value) * DISPLAY_HEIGHT / MODEL_INPUT_HEIGHT)

    def _physical_x_screen(self, position_cm):
        pixel_x = x_pixel_for_position_cm(
            position_cm,
            self.x_calibration_lut,
        )
        screen_x = int(
            pixel_x * DISPLAY_WIDTH / MODEL_INPUT_WIDTH + 0.5
        )
        if float(position_cm) <= -12.5:
            return 0
        if float(position_cm) >= 12.5:
            return DISPLAY_WIDTH - 1
        if screen_x < 0:
            return 0
        if screen_x >= DISPLAY_WIDTH:
            return DISPLAY_WIDTH - 1
        return screen_x

    @staticmethod
    def _ruler_label(value_x10):
        if value_x10 == 0:
            return "0"
        sign = "-" if value_x10 < 0 else ""
        magnitude = -value_x10 if value_x10 < 0 else value_x10
        whole = magnitude // 10
        tenth = magnitude % 10
        if tenth:
            return "{}{}.{}".format(sign, whole, tenth)
        return "{}{}".format(sign, whole)

    def _draw_physical_x_ruler(self):
        tick_x10 = int(float(PHYSICAL_X_RULER_TICK_CM) * 10.0 + 0.5)
        if tick_x10 not in (5, 10):
            raise ValueError(
                "PHYSICAL_X_RULER_TICK_CM must be 0.5 or 1.0"
            )
        ruler_y = int(PHYSICAL_X_RULER_Y)
        if ruler_y < 16 or ruler_y > DISPLAY_HEIGHT - 40:
            raise ValueError("PHYSICAL_X_RULER_Y outside drawable range")

        ruler = image.Image(
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            image.ARGB8888,
        )
        ruler.clear()
        left_x = self._physical_x_screen(-12.5)
        right_x = self._physical_x_screen(12.5)
        ruler.draw_line(
            left_x,
            ruler_y,
            right_x,
            ruler_y,
            color=COLOR_RULER_SHADOW,
            thickness=3,
        )
        ruler.draw_line(
            left_x,
            ruler_y,
            right_x,
            ruler_y,
            color=COLOR_RULER,
            thickness=1,
        )

        # A 1 cm grid is zero-centered; endpoint labels still mark +/-12.5 cm.
        minor_value_x10 = -125 if tick_x10 == 5 else -120
        while minor_value_x10 <= 125:
            screen_x = self._physical_x_screen(minor_value_x10 / 10.0)
            ruler.draw_line(
                screen_x,
                ruler_y - 4,
                screen_x,
                ruler_y + 5,
                color=COLOR_RULER_SHADOW,
                thickness=3,
            )
            ruler.draw_line(
                screen_x,
                ruler_y - 4,
                screen_x,
                ruler_y + 5,
                color=COLOR_RULER,
                thickness=1,
            )
            minor_value_x10 += tick_x10

        label_value_x10 = -125
        while label_value_x10 <= 125:
            screen_x = self._physical_x_screen(label_value_x10 / 10.0)
            ruler.draw_line(
                screen_x,
                ruler_y - 8,
                screen_x,
                ruler_y + 9,
                color=COLOR_RULER_SHADOW,
                thickness=3,
            )
            ruler.draw_line(
                screen_x,
                ruler_y - 8,
                screen_x,
                ruler_y + 9,
                color=COLOR_RULER_MAJOR,
                thickness=2,
            )
            label = self._ruler_label(label_value_x10)
            estimated_width = len(label) * 9
            label_x = screen_x - estimated_width // 2
            if label_x < 1:
                label_x = 1
            max_label_x = DISPLAY_WIDTH - estimated_width - 1
            if label_x > max_label_x:
                label_x = max_label_x
            ruler.draw_string_advanced(
                label_x + 1,
                ruler_y + 12,
                16,
                label,
                color=COLOR_RULER_SHADOW,
            )
            ruler.draw_string_advanced(
                label_x,
                ruler_y + 11,
                16,
                label,
                color=COLOR_RULER_MAJOR,
            )
            label_value_x10 += 25

        Display.show_image(ruler, 0, 0, Display.LAYER_OSD0)
        self.ruler_drawn = True
        ruler = None
        gc.collect()

    def _draw_text(self, row, text, color):
        self.osd.draw_string_advanced(
            5,
            4 + int(row) * 24,
            16,
            str(text),
            color=color,
        )

    def _draw_pid_panel(self, pid_tuner, feedback):
        if pid_tuner is None or not pid_tuner.enabled:
            return
        osd = self.osd
        panel_top = TOUCH_PID_PANEL_TOP - 20
        osd.draw_rectangle(
            0,
            panel_top,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT - panel_top,
            color=COLOR_PANEL,
            fill=True,
        )
        osd.draw_string_advanced(
            8,
            panel_top + 1,
            16,
            "PID TOUCH step=0.1 {}".format(pid_tuner.diagnostic_text()),
            color=(COLOR_BAD if pid_tuner.last_error is not None else COLOR_TITLE),
        )
        values = (
            ("Kp", float(feedback.get("kp", pid_tuner.controller.kp))),
            ("Kd", float(feedback.get("kd", pid_tuner.controller.kd))),
            ("Ki", float(feedback.get("ki", pid_tuner.controller.ki))),
        )
        for row, item in enumerate(values):
            label, value = item
            top = TOUCH_PID_PANEL_TOP + row * TOUCH_PID_ROW_HEIGHT
            button_y = top + 3
            osd.draw_string_advanced(
                16,
                top + 8,
                20,
                "{}={:4.1f}".format(label, value),
                color=COLOR_TITLE,
            )
            osd.draw_rectangle(
                TOUCH_PID_MINUS_X,
                button_y,
                TOUCH_PID_BUTTON_WIDTH,
                TOUCH_PID_BUTTON_HEIGHT,
                color=COLOR_TOUCH_BUTTON,
                fill=True,
            )
            osd.draw_rectangle(
                TOUCH_PID_MINUS_X,
                button_y,
                TOUCH_PID_BUTTON_WIDTH,
                TOUCH_PID_BUTTON_HEIGHT,
                color=COLOR_TITLE,
                thickness=2,
            )
            osd.draw_string_advanced(
                TOUCH_PID_MINUS_X + 33,
                top + 5,
                24,
                "-",
                color=COLOR_TITLE,
            )
            osd.draw_rectangle(
                TOUCH_PID_PLUS_X,
                button_y,
                TOUCH_PID_BUTTON_WIDTH,
                TOUCH_PID_BUTTON_HEIGHT,
                color=COLOR_TOUCH_BUTTON,
                fill=True,
            )
            osd.draw_rectangle(
                TOUCH_PID_PLUS_X,
                button_y,
                TOUCH_PID_BUTTON_WIDTH,
                TOUCH_PID_BUTTON_HEIGHT,
                color=COLOR_TITLE,
                thickness=2,
            )
            osd.draw_string_advanced(
                TOUCH_PID_PLUS_X + 31,
                top + 5,
                24,
                "+",
                color=COLOR_TITLE,
            )

    def draw_status(
        self,
        candidate,
        state,
        feedback,
        inference_ms,
        capture_ms,
        frame_interval_ms,
        measured_fps,
        capture_interval_ms,
        detector,
        tx,
        rx,
        wifi_video,
        pid_tuner=None,
        runtime_error=None,
    ):
        osd = self.osd
        if feedback is None:
            feedback = {}
        osd.clear()
        osd.draw_rectangle(
            0, 0, DISPLAY_WIDTH, STATUS_PANEL_HEIGHT,
            color=COLOR_PANEL, fill=True,
        )

        if detector.model_backend in ("roi_blob", "edge_hough"):
            roi_x, roi_y, roi_width, roi_height = detector.roi
            osd.draw_rectangle(
                self._screen_x(roi_x),
                self._screen_y(roi_y),
                max(1, self._screen_x(roi_width)),
                max(1, self._screen_y(roi_height)),
                color=COLOR_AXIS,
                thickness=1,
            )

        if candidate is not None:
            box_color = COLOR_BAD if state.get("outlier_rejected") else COLOR_RAW
            osd.draw_rectangle(
                self._screen_x(candidate["x"]),
                self._screen_y(candidate["y"]),
                max(1, self._screen_x(candidate["w"])),
                max(1, self._screen_y(candidate["h"])),
                color=box_color,
                thickness=3,
            )

        if state.get("track_valid"):
            tracked_diameter = max(
                2.0,
                float(state.get("tracked_diameter", 0.0)),
            )
            tracked_left = state["x"] - tracked_diameter * 0.5
            tracked_top = state["y"] - tracked_diameter * 0.5
            osd.draw_rectangle(
                self._screen_x(tracked_left),
                self._screen_y(tracked_top),
                max(1, self._screen_x(tracked_diameter)),
                max(1, self._screen_y(tracked_diameter)),
                color=COLOR_FILTERED,
                thickness=2,
            )
            osd.draw_cross(
                self._screen_x(state["x"]),
                self._screen_y(state["y"]),
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

        mode = _tracking_mode(state, feedback)
        if mode == "DETECT":
            mode_color = COLOR_GOOD
        elif mode.startswith("HOLD"):
            mode_color = COLOR_WARN
        else:
            mode_color = COLOR_BAD
        self._draw_text(
            0,
            "STEEL BALL {} {} {} NET:{}".format(
                APP_VERSION, detector.model_backend, mode, wifi_video.state
            ),
            mode_color,
        )
        self._draw_text(
            1,
            "px=({:.2f},{:.2f}) v=({:.1f},{:.1f}) c={:.2f} age={}".format(
                state.get("x", 0.0),
                state.get("y", 0.0),
                state.get("vx", 0.0),
                state.get("vy", 0.0),
                0.0 if candidate is None else candidate["confidence"],
                int(state.get("measurement_age_ms", 0xFFFF)),
            ),
            COLOR_GOOD if state.get("track_valid") else COLOR_WARN,
        )
        self._draw_text(
            2,
            "cm[{}] x={:+.2f} ref={:+.2f} e={:+.2f} v={:+.2f} a={:+.1f}".format(
                "MEAS" if X_CALIBRATION_MODE == X_CALIBRATION_MEASURED
                else "PIN",
                float(feedback.get("position_cm", 0.0)),
                float(feedback.get("reference_position_cm", 0.0)),
                float(feedback.get("position_error_cm", 0.0)),
                float(feedback.get("velocity_cm_s", 0.0)),
                float(feedback.get("acceleration_cm_s2", 0.0)),
            ),
            COLOR_TITLE,
        )
        self._draw_text(
            3,
            "FB{:+.1f}/{:+.1f} I{:+.1f} F{:+.1f} s={} db={}{}".format(
                float(feedback.get("feedback_acceleration_cm_s2", 0.0)),
                float(feedback.get(
                    "unfiltered_feedback_acceleration_cm_s2",
                    0.0,
                )),
                float(feedback.get("steady_correction_cm_s2", 0.0)),
                float(feedback.get("breakaway_feedback_cm_s2", 0.0)),
                str(feedback.get("steady_correction_state", "?"))[:6],
                "Y" if feedback.get("joint_deadband_active") else "N",
                " SAT" if feedback.get("feedback_saturated") else "",
            ),
            COLOR_GOOD if feedback.get("feedback_valid") else COLOR_WARN,
        )
        self._draw_text(
            4,
            "T FPS={:.1f}/{} dt={:.1f} loop={} cap={} {}={} {}={} "
            "{}={} det={} fb={} {}".format(
                float(measured_fps),
                int(CAMERA_FPS),
                float(capture_interval_ms),
                int(frame_interval_ms),
                int(capture_ms),
                detector.timing_labels[0],
                int(detector.preprocess_ms),
                detector.timing_labels[1],
                int(detector.kpu_ms),
                detector.timing_labels[2],
                int(detector.postprocess_ms),
                int(inference_ms),
                int(feedback.get("compute_ms", 0)),
                self.exposure_text(),
            ),
            COLOR_TITLE,
        )
        self._draw_text(
            5,
            "{} M{}{}{}{}{} d{:+.1f} vm{:+.1f} R{} A{}/{}".format(
                _detector_diagnostic_text(detector, candidate),
                str(state.get("motion_state", "?"))[:1],
                int(state.get("motion_confirm_count", 0)),
                "F" if state.get("fast_response") else "",
                "W" if state.get("association_wide") else "",
                "C" if state.get("association_competing") else "",
                float(state.get("motion_step_px", 0.0)),
                float(state.get("measured_axial_velocity", 0.0)),
                int(state.get("association_reject_count", 0)),
                int(state.get("acquisition_count", 0)),
                int(state.get("acquisition_required", 3)),
            ),
            COLOR_BAD if state.get("association_rejected") else COLOR_TITLE,
        )
        self._draw_text(
            6,
            "UART TX f={} b={} q={} RX ok={} bad={} ref={:+.2f}".format(
                tx.sent_frames,
                tx.bytes_sent,
                tx.queued_bytes(),
                rx.applied_commands,
                rx.parser.invalid_frames,
                rx.last_applied_cm,
            ),
            COLOR_GOOD,
        )
        self._draw_text(
            7,
            "UART err tx={} short={} zero={} drop={} rx={}".format(
                tx.write_errors,
                tx.short_writes,
                tx.zero_writes,
                tx.dropped_error_frames,
                rx.read_errors,
            ),
            COLOR_BAD if (
                tx.write_errors or tx.short_writes or tx.zero_writes
                or tx.dropped_error_frames or rx.read_errors
            ) else COLOR_GOOD,
        )
        visible_error = runtime_error
        if visible_error is None:
            visible_error = tx.last_error
        if visible_error is None:
            visible_error = rx.last_error
        if (
            visible_error is None
            and pid_tuner is not None
            and pid_tuner.last_error is not None
        ):
            visible_error = pid_tuner.last_error
        if visible_error is not None:
            status_text = "last: " + str(visible_error)[:70]
            status_color = COLOR_BAD
        else:
            status_text = wifi_video.diagnostic_text()
            status_color = (
                COLOR_GOOD if wifi_video.state in ("RUN", "OFF")
                else COLOR_WARN
            )
        self._draw_text(8, status_text, status_color)
        self._draw_pid_panel(pid_tuner, feedback)
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
        if self.wifi_video is not None:
            try:
                if not self.wifi_video.stop():
                    print(
                        "camera cleanup skipped: wifi video worker active"
                    )
                    return False
            except Exception as exc:
                print("wifi video stop warning:", exc)
                return False
        if self.sensor_running and self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception as exc:
                print("sensor stop warning:", exc)
        self.sensor_running = False
        if self.wifi_video is not None:
            try:
                self.wifi_video.release()
            except Exception as exc:
                print("wifi video release warning:", exc)
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
        self.ruler_drawn = False
        self.wifi_video = None
        return True


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
            raise ValueError(
                "UART state frame must be exactly {} bytes".format(
                    BALL_STATE_FRAME_SIZE
                )
            )
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


class UartReferenceCommandRx:
    """Bounded non-blocking receiver for MCU set-reference commands."""

    def __init__(self, reader, controller):
        if (
            reader is None
            or not hasattr(reader, "any")
            or not hasattr(reader, "read")
        ):
            raise ValueError("UART reader must provide any() and read()")
        self.reader = reader
        self.controller = controller
        self.parser = ReferenceCommandParser()
        self.read_calls = 0
        self.read_errors = 0
        self.applied_commands = 0
        self.last_applied_cm = float(controller.reference_position_cm)
        self.last_error = None

    def poll(self):
        """Read at most one bounded chunk and apply only the newest command."""
        try:
            waiting = self.reader.any()
            if waiting is None:
                return None
            waiting = int(waiting)
            if waiting <= 0:
                return None
            data = self.reader.read(min(waiting, UART_MAX_READ_BYTES))
        except Exception as exc:
            if _is_ide_interrupt(exc):
                raise KeyboardInterrupt()
            self.read_errors += 1
            self.last_error = "UART RX {}: {}".format(
                type(exc).__name__,
                exc,
            )
            if self.read_errors == 1 or self.read_errors % 20 == 0:
                print(self.last_error)
            return None
        self.read_calls += 1
        if data is None or len(data) == 0:
            return None
        reference_position_cm = self.parser.feed(data)
        if reference_position_cm is None:
            return None
        try:
            applied_cm = self.controller.set_reference_position_cm(
                reference_position_cm,
                BALL_REFERENCE_FLAG_VALID
                | BALL_REFERENCE_FLAG_SOURCE_MCU,
            )
        except Exception as exc:
            self.read_errors += 1
            self.last_error = "UART REF {}: {}".format(
                type(exc).__name__,
                exc,
            )
            print(self.last_error)
            return None
        self.applied_commands += 1
        self.last_applied_cm = float(applied_cm)
        self.last_error = None
        print(
            "UART reference applied x_ref={:+.2f}cm count={}".format(
                self.last_applied_cm,
                self.applied_commands,
            )
        )
        return self.last_applied_cm



def _open_uart():
    fpioa = FPIOA()
    fpioa.set_function(9, fpioa.UART1_TXD, ie=0, oe=1, pu=1)
    fpioa.set_function(10, fpioa.UART1_RXD, ie=1, oe=0, pu=1)
    return UART(UART.UART1, UART_BAUDRATE)


def _tracking_mode(state, feedback):
    if feedback.get("feedback_valid") and state.get("measurement_accepted"):
        return "DETECT"
    if feedback.get("feedback_valid") and state.get("predicted"):
        return "HOLD {}/{}".format(
            int(state.get("missed_frames", 0)),
            TRACK_HOLD_MAX_FRAMES,
        )
    if state.get("acquisition_pending"):
        return "LOST ACQ {}/{}".format(
            int(state.get("acquisition_count", 0)),
            int(state.get("acquisition_required", 3)),
        )
    return "LOST"


def _state_flags(candidate, state, feedback=None):
    if feedback is None:
        feedback = {}
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
    if feedback.get("feedback_valid"):
        flags |= BALL_FLAG_FEEDBACK_VALID
    if state.get("outlier_rejected") or state.get("association_rejected"):
        flags |= BALL_FLAG_OUTLIER_REJECTED
    if feedback.get("feedback_saturated"):
        flags |= BALL_FLAG_FEEDBACK_SATURATED
    return flags


def _publish_state(
    tx,
    sequence,
    now_ms,
    candidate,
    state,
    feedback,
    inference_ms,
):
    frame = encode_ball_state_frame(
        sequence=sequence,
        timestamp_ms=now_ms,
        flags=_state_flags(candidate, state, feedback),
        state=state,
        candidate=candidate,
        inference_ms=inference_ms,
        feedback=feedback,
    )
    tx.submit_latest(frame)
    tx.poll()


def _print_pc_debug_status(
    sequence,
    candidate,
    state,
    feedback,
    inference_ms,
    capture_ms,
    detector,
    camera,
    tx,
    rx,
    wifi_video,
    pid_tuner,
    runtime_error,
    measured_fps,
    capture_interval_ms,
):
    if feedback is None:
        feedback = {}
    if candidate is None:
        raw_x = -1.0
        raw_y = -1.0
        confidence = 0.0
    else:
        raw_x = float(candidate.get("center_x", -1.0))
        raw_y = float(candidate.get("center_y", -1.0))
        confidence = float(candidate.get("confidence", 0.0))
    print(
        "PCDBG seq={} mode={} flags=0x{:02X} raw=({:.1f},{:.1f}) "
        "filt=({:.2f},{:.2f}) vel=({:.1f},{:.1f}) conf={:.3f} "
        "obs[m={}{}{}{}{},step={:+.1f},fast={},in={:+.2f},vm={:+.1f},"
        "q={:.2f},s={}] "
        "phys[p=({:+.2f},{:+.2f}),v=({:+.2f},{:+.2f}),aX={:+.1f},"
        "ref={:+.2f},e={:+.2f}] "
        "ctrl[e={:+.2f},v={:+.2f},db={}] "
        "fb[a={:+.1f},raw={:+.1f},kp={:.2f},kd={:.2f},ki={:.2f},"
        "ka={:.3f},I={:+.2f},F={:+.2f},is={},valid={},sat={}] "
        "model={} diag={} cal={} exposure={} fps={:.2f}/{} cap_dt={:.2f}ms "
        "time[cap={},{}={},{}={},{}={},det={}]ms "
        "UART[txf={},txb={},q={},e={},s={},z={},d={},rxb={},rxok={},"
        "rxbad={},rxerr={},ref={:+.2f}] NET[{}] TOUCH[{}] "
        "uart_last={} runtime_last={}".format(
            int(sequence),
            _tracking_mode(state, feedback),
            _state_flags(candidate, state, feedback),
            raw_x,
            raw_y,
            float(state.get("x", 0.0)),
            float(state.get("y", 0.0)),
            float(state.get("vx", 0.0)),
            float(state.get("vy", 0.0)),
            confidence,
            str(state.get("motion_state", "?")),
            int(state.get("motion_confirm_count", 0)),
            "A" if state.get("measurement_ambiguous") else "",
            "W" if state.get("association_wide") else "",
            "C" if state.get("association_competing") else "",
            float(state.get("motion_step_px", 0.0)),
            "Y" if state.get("fast_response") else "N",
            float(state.get("axial_innovation", 0.0)),
            float(state.get("measured_axial_velocity", 0.0)),
            float(state.get("measurement_quality", 0.0)),
            int(state.get("measurement_support", 0)),
            float(feedback.get("position_cm", 0.0)),
            float(feedback.get("position_y_cm", 0.0)),
            float(feedback.get("velocity_cm_s", 0.0)),
            float(feedback.get("velocity_y_cm_s", 0.0)),
            float(feedback.get("acceleration_cm_s2", 0.0)),
            float(feedback.get("reference_position_cm", 0.0)),
            float(feedback.get("position_error_cm", 0.0)),
            float(feedback.get("control_position_error_cm", 0.0)),
            float(feedback.get("feedback_velocity_cm_s", 0.0)),
            "Y" if feedback.get("joint_deadband_active") else "N",
            float(feedback.get("feedback_acceleration_cm_s2", 0.0)),
            float(feedback.get(
                "unfiltered_feedback_acceleration_cm_s2",
                0.0,
            )),
            float(feedback.get("kp", 0.0)),
            float(feedback.get("kd", 0.0)),
            float(feedback.get("ki", 0.0)),
            float(feedback.get("ka", 0.0)),
            float(feedback.get("steady_correction_cm_s2", 0.0)),
            float(feedback.get("breakaway_feedback_cm_s2", 0.0)),
            str(feedback.get("steady_correction_state", "?")),
            "Y" if feedback.get("feedback_valid") else "N",
            "Y" if feedback.get("feedback_saturated") else "N",
            detector.model_backend,
            _detector_diagnostic_text(detector, candidate),
            X_CALIBRATION_MODE,
            camera.exposure_text(),
            float(measured_fps),
            int(CAMERA_FPS),
            float(capture_interval_ms),
            int(capture_ms),
            detector.timing_labels[0],
            int(detector.preprocess_ms),
            detector.timing_labels[1],
            int(detector.kpu_ms),
            detector.timing_labels[2],
            int(detector.postprocess_ms),
            int(inference_ms),
            tx.sent_frames,
            tx.bytes_sent,
            tx.queued_bytes(),
            tx.write_errors,
            tx.short_writes,
            tx.zero_writes,
            tx.dropped_error_frames,
            rx.parser.bytes_received,
            rx.applied_commands,
            rx.parser.invalid_frames,
            rx.read_errors,
            rx.last_applied_cm,
            wifi_video.diagnostic_text(),
            "OFF" if pid_tuner is None else pid_tuner.diagnostic_text(),
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
    rx = None
    camera = None
    detector = None
    tracker = None
    feedback_filter = None
    pid_tuner = None
    wifi_video = None
    stage = "camera_object_init"
    sequence = 0
    try:
        wifi_video = _create_wifi_video_service()
        camera = CameraDisplay()
        stage = "camera_display_init"
        camera.start(wifi_video=wifi_video)
        stage = "model_check"
        _verify_model()
        stage = "uart_init"
        uart = _open_uart()
        tx = LatestFrameUartTx(uart)
        stage = "tracker_init"
        tracker = PipeMotionFilter(
            alpha=TRACK_AXIAL_FAST_ALPHA,
            response_alpha=TRACK_AXIAL_QUIET_ALPHA,
            response_start_px=TRACK_AXIAL_RESPONSE_START_PX,
            response_full_px=TRACK_AXIAL_RESPONSE_FULL_PX,
            response_beta=TRACK_VELOCITY_BETA,
            velocity_fast_beta=TRACK_VELOCITY_FAST_BETA,
            velocity_noise_px=TRACK_VELOCITY_NOISE_PX,
            velocity_quiet_decay=TRACK_VELOCITY_QUIET_DECAY,
            velocity_quiet_timeout_ms=TRACK_VELOCITY_QUIET_TIMEOUT_MS,
            lateral_alpha=TRACK_LATERAL_ALPHA,
            diameter_alpha=TRACK_DIAMETER_ALPHA,
            diameter_max_step_px=TRACK_DIAMETER_MAX_STEP_PX,
            motion_start_px=TRACK_MOTION_START_PX,
            motion_immediate_px=TRACK_MOTION_IMMEDIATE_PX,
            motion_confirm_frames=TRACK_MOTION_CONFIRM_FRAMES,
            motion_stop_frames=TRACK_MOTION_STOP_FRAMES,
            motion_direction_epsilon_px=(
                TRACK_MOTION_DIRECTION_EPSILON_PX
            ),
            ambiguous_position_alpha=TRACK_AMBIGUOUS_POSITION_ALPHA,
            quiet_position_alpha=TRACK_QUIET_POSITION_ALPHA,
            quiet_velocity_decay=TRACK_QUIET_VELOCITY_DECAY,
            quiet_gate_diameter_factor=(
                TRACK_QUIET_GATE_DIAMETER_FACTOR
            ),
            quiet_gate_min_px=TRACK_QUIET_GATE_MIN_PX,
            motion_min_quality=TRACK_MOTION_MIN_QUALITY,
            motion_min_support=TRACK_MOTION_MIN_SUPPORT,
            association_ambiguity_diameter_factor=(
                TRACK_ASSOCIATION_AMBIGUITY_DIAMETER_FACTOR
            ),
            max_coast_ms=TRACK_HOLD_MAX_MS,
            max_coast_frames=TRACK_HOLD_MAX_FRAMES,
            fixed_axis_x=(
                PIPE_POSITIVE_END_X_PX - PIPE_NEGATIVE_END_X_PX
            ),
            fixed_axis_y=(
                PIPE_POSITIVE_END_Y_PX - PIPE_NEGATIVE_END_Y_PX
            ),
        )
        stage = "feedback_init"
        feedback_filter = BallStateFeedback(
            kp=FEEDBACK_KP,
            kd=FEEDBACK_KD,
            ki=FEEDBACK_KI,
            ka=FEEDBACK_KA,
            max_feedback_accel_cm_s2=FEEDBACK_ACCEL_LIMIT_CM_S2,
            position_deadband_cm=FEEDBACK_POSITION_DEADBAND_CM,
            velocity_deadband_cm_s=(
                FEEDBACK_VELOCITY_DEADBAND_CM_S
            ),
            max_prediction_age_ms=TRACK_HOLD_MAX_MS,
            max_prediction_frames=TRACK_HOLD_MAX_FRAMES,
            x_calibration_mode=X_CALIBRATION_MODE,
            reference_position_cm=FEEDBACK_REFERENCE_POSITION_CM,
            reference_state_flags=(
                BALL_REFERENCE_FLAG_VALID
                | BALL_REFERENCE_FLAG_SOURCE_PRESET
            ),
            position_filter_tau_s=FEEDBACK_POSITION_FILTER_TAU_S,
            velocity_filter_tau_s=FEEDBACK_VELOCITY_FILTER_TAU_S,
            feedback_filter_tau_s=FEEDBACK_OUTPUT_FILTER_TAU_S,
            integral_limit_cm_s2=FEEDBACK_INTEGRAL_LIMIT_CM_S2,
            integral_error_limit_cm=FEEDBACK_INTEGRAL_ERROR_LIMIT_CM,
            integral_leak_per_s=FEEDBACK_INTEGRAL_LEAK_PER_S,
            integral_unwind_gain_s2=FEEDBACK_INTEGRAL_UNWIND_GAIN_S2,
            breakaway_accel_cm_s2=FEEDBACK_BREAKAWAY_ACCEL_CM_S2,
            breakaway_transition_cm=FEEDBACK_BREAKAWAY_TRANSITION_CM,
        )
        stage = "uart_rx_init"
        rx = UartReferenceCommandRx(uart, feedback_filter)
        stage = "touch_pid_init"
        pid_tuner = TouchPidTuner(feedback_filter)
        stage = "detector_init"
        detector = _create_detector()
        stage = "wifi_video_start"
        if wifi_video.state == "RDY":
            wifi_video.start()
        print(
            "steel-ball tracker started, version={}, backend={}, model={}, "
            "expected={}, UART={} 8N1, frame={}B, software_tx={}B, "
            "pipeline={}, fb_kp={:.2f}, fb_kd={:.2f}, fb_ki={:.2f}, "
            "fb_ka={:.3f}, fb_i_limit={:.1f}, fb_unwind={:.1f}, "
            "fb_break={:.1f}/{:.2f}, "
            "fb_db={:.2f}/{:.2f}, "
            "calibration={}, ref={:+.2f}cm, center_cm_per_px={:.5f}, exposure={}, "
            "net={}, pc_debug={}".format(
                APP_VERSION,
                detector.model_backend,
                detector.model_path,
                detector.expected_output,
                UART_BAUDRATE,
                BALL_STATE_FRAME_SIZE,
                UART_SOFTWARE_BUFFER_BYTES,
                detector.pipeline_name,
                float(feedback_filter.kp),
                float(feedback_filter.kd),
                float(feedback_filter.ki),
                float(feedback_filter.ka),
                float(feedback_filter.integral_limit_cm_s2),
                float(feedback_filter.integral_unwind_gain_s2),
                float(feedback_filter.breakaway_accel_cm_s2),
                float(feedback_filter.breakaway_transition_cm),
                float(feedback_filter.position_deadband_cm),
                float(feedback_filter.velocity_deadband_cm_s),
                feedback_filter.x_calibration_mode,
                feedback_filter.reference_position_cm,
                float(feedback_filter.center_cm_per_pixel),
                camera.exposure_text(),
                wifi_video.diagnostic_text(),
                bool(pc_debug),
            )
        )

        last_overlay_ms = _ticks_ms() - OVERLAY_INTERVAL_MS
        last_pc_debug_ms = _ticks_ms() - PC_DEBUG_INTERVAL_MS
        last_frame_ms = _ticks_ms()
        last_capture_finished_ms = None
        capture_interval_ms = 0.0
        measured_fps = 0.0
        frame_count = 0
        consecutive_errors = 0
        runtime_error_count = 0
        latched_runtime_error = None
        while True:
            _exitpoint()
            tx.poll()
            stage = "uart_command_poll"
            rx.poll()
            stage = "touch_pid_poll"
            pid_tuner.poll(_ticks_ms())
            candidate = None
            detections = []
            feedback = None
            inference_ms = 0
            capture_ms = 0
            now_ms = _ticks_ms()
            frame_interval_ms = max(
                0,
                _ticks_diff(now_ms, last_frame_ms),
            )
            last_frame_ms = now_ms
            try:
                stage = "camera_capture"
                capture_start_ms = _ticks_ms()
                detector_input = camera.get_frame()
                capture_finished_ms = _ticks_ms()
                if last_capture_finished_ms is not None:
                    capture_sample_ms = max(
                        1,
                        _ticks_diff(
                            capture_finished_ms,
                            last_capture_finished_ms,
                        ),
                    )
                    if capture_interval_ms <= 0.0:
                        capture_interval_ms = float(capture_sample_ms)
                    else:
                        capture_interval_ms += 0.12 * (
                            float(capture_sample_ms) - capture_interval_ms
                        )
                    measured_fps = 1000.0 / capture_interval_ms
                last_capture_finished_ms = capture_finished_ms
                capture_ms = max(
                    0,
                    _ticks_diff(capture_finished_ms, capture_start_ms),
                )
                stage = "detector_run"
                inference_start_ms = capture_finished_ms
                if detector.model_backend in ("roi_blob", "edge_hough"):
                    detections = detector.run(
                        detector_input,
                        tracker.detection_hint(capture_finished_ms),
                    )
                else:
                    detections = detector.run(detector_input)
                finished_ms = _ticks_ms()
                inference_ms = max(
                    0,
                    _ticks_diff(finished_ms, inference_start_ms),
                )
                # The frame represents the capture-finished instant. Associate
                # and correct at that measurement time, then predict only
                # through the measured detector latency to the publish instant.
                measurement_ms = capture_finished_ms
                stage = "target_association"
                candidate = tracker.select_candidate(detections, measurement_ms)
                stage = "motion_filter"
                state = tracker.update(
                    candidate,
                    measurement_ms,
                    output_ms=finished_ms,
                )
                stage = "visual_feedback"
                feedback_start_ms = _ticks_ms()
                feedback = feedback_filter.update(
                    state,
                    candidate,
                    finished_ms,
                )
                feedback_finished_ms = _ticks_ms()
                feedback["compute_ms"] = max(
                    0,
                    _ticks_diff(feedback_finished_ms, feedback_start_ms),
                )
                sequence = (sequence + 1) & 0xFFFF
                stage = "uart_publish"
                _publish_state(
                    tx,
                    sequence,
                    finished_ms,
                    candidate,
                    state,
                    feedback,
                    inference_ms,
                )
                consecutive_errors = 0
                now_ms = feedback_finished_ms
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
                detections = []
                tracker.select_candidate(detections, now_ms)
                state = tracker.update(None, now_ms)
                feedback = feedback_filter.update(state, None, now_ms)
                sequence = (sequence + 1) & 0xFFFF
                stage = "uart_error_publish"
                _publish_state(
                    tx,
                    sequence,
                    now_ms,
                    None,
                    state,
                    feedback,
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
                    feedback,
                    inference_ms,
                    capture_ms,
                    frame_interval_ms,
                    measured_fps,
                    capture_interval_ms,
                    detector,
                    tx,
                    rx,
                    wifi_video,
                    pid_tuner=pid_tuner,
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
                    feedback,
                    inference_ms,
                    capture_ms,
                    detector,
                    camera,
                    tx,
                    rx,
                    wifi_video,
                    pid_tuner,
                    latched_runtime_error,
                    measured_fps,
                    capture_interval_ms,
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
