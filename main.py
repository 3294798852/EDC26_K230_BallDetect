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
PIXEL_FORMAT_RGB_565 = 0
PIXEL_FORMAT_RGB_888_PLANAR = 0
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 0
PipeMotionFilter = None
normalize_postprocessed_ball_candidates = None
BallStateFeedback = None
RoiDarkBlobDetector = None
encode_ball_state_frame = None

BALL_FLAG_RAW_DETECTED = 0x01
BALL_FLAG_MEASUREMENT_ACCEPTED = 0x02
BALL_FLAG_TRACK_VALID = 0x04
BALL_FLAG_PREDICTED = 0x08
BALL_FLAG_FEEDBACK_VALID = 0x10
BALL_FLAG_OUTLIER_REJECTED = 0x20
BALL_FLAG_EDGE_CLIPPED = 0x40
BALL_FLAG_FEEDBACK_SATURATED = 0x80
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
        BALL_STATE_FRAME_SIZE,
        encode_ball_state_frame,
    )
    _BOOT_IMPORT_STAGE = "import_ball_tracking"
    from ball_tracking import (
        PipeMotionFilter,
        normalize_postprocessed_ball_candidates,
    )
    _BOOT_IMPORT_STAGE = "import_ball_control"
    from ball_control import BallStateFeedback
    _BOOT_IMPORT_STAGE = "import_ball_roi_detector"
    from ball_roi_detector import RoiDarkBlobDetector
    _BOOT_IMPORT_STAGE = None
except Exception as _boot_exc:
    _BOOT_IMPORT_ERROR = _boot_exc


APP_VERSION = "proj2-1.14"

# Select one detector backend here: "yolov8n", "yolov8", or "roi_blob".
# The generic "yolov8" artifact has no trustworthy subtype metadata.
MODEL_BACKEND = "roi_blob"
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
ROI_MIN_DIAMETER_PX = 4.0
ROI_MAX_DIAMETER_PX = 26.0
ROI_TARGET_DIAMETER_PX = 11.0
ROI_MIN_PIXELS = 12
ROI_THRESHOLD_RATIO = 0.60
ROI_THRESHOLD_L_MIN = 20
ROI_THRESHOLD_L_MAX = 62
ROI_THRESHOLD_UPDATE_FRAMES = 6

# Visual-to-physical calibration. Positive position points from the negative
# endpoint to the positive endpoint. Values remain provisional until measured.
PIPE_NEGATIVE_END_X_PX = 10.0
PIPE_NEGATIVE_END_Y_PX = 160.0
PIPE_POSITIVE_END_X_PX = 310.0
PIPE_POSITIVE_END_Y_PX = 160.0
PIPE_LENGTH_CM = 25.0

# K230 publishes only its camera-derived state-feedback term. Physical position
# and velocity pass through continuous soft deadbands before their gains.
FEEDBACK_KP = 16.0
FEEDBACK_KD = 7.2
FEEDBACK_KA = 0.0
FEEDBACK_ACCEL_LIMIT_CM_S2 = 250.0
TRACK_HOLD_MAX_FRAMES = 8
TRACK_HOLD_MAX_MS = 300

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
STATUS_PANEL_HEIGHT = 230


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


def _selected_model_config():
    config = MODEL_BACKENDS.get(MODEL_BACKEND)
    if config is None:
        raise RuntimeError(
            "unknown MODEL_BACKEND {}; use yolov8n, yolov8 or roi_blob".format(
                MODEL_BACKEND
            )
        )
    if config.get("native_roi"):
        return config
    if not config.get("path") or not config.get("expected_output"):
        raise RuntimeError("incomplete model backend: " + MODEL_BACKEND)
    return config


def _verify_model():
    config = _selected_model_config()
    if config.get("native_roi"):
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
            threshold_ratio=ROI_THRESHOLD_RATIO,
            threshold_l_min=ROI_THRESHOLD_L_MIN,
            threshold_l_max=ROI_THRESHOLD_L_MAX,
            threshold_update_frames=ROI_THRESHOLD_UPDATE_FRAMES,
            max_candidates=POSTPROCESS_MAX_BOXES,
        )
    return BallDetector()


class CameraDisplay:
    """Own camera channel 0 video, channel 2 AI input and one OSD layer."""

    def __init__(self):
        self.sensor = None
        self.osd = None
        self.display_initialized = False
        self.media_initialized = False
        self.sensor_running = False
        self.native_roi_input = bool(
            _selected_model_config().get("native_roi")
        )

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
            PIXEL_FORMAT_RGB_565
            if self.native_roi_input
            else PIXEL_FORMAT_RGB_888_PLANAR,
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
            raise RuntimeError("camera returned no detector frame")
        if self.native_roi_input:
            return frame
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
        feedback,
        inference_ms,
        capture_ms,
        frame_interval_ms,
        detector,
        tx,
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

        if detector.model_backend == "roi_blob":
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
            "STEEL BALL {} {} {}".format(
                APP_VERSION, detector.model_backend, mode
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
            "cm p={:+.2f} v={:+.2f} a={:+.1f}".format(
                float(feedback.get("position_cm", 0.0)),
                float(feedback.get("velocity_cm_s", 0.0)),
                float(feedback.get("acceleration_cm_s2", 0.0)),
            ),
            COLOR_TITLE,
        )
        self._draw_text(
            3,
            "FB={:+.1f} kp={:.1f} kd={:.1f} ka={:.2f} {} sat={}".format(
                float(feedback.get("feedback_acceleration_cm_s2", 0.0)),
                float(feedback.get("kp", 0.0)),
                float(feedback.get("kd", 0.0)),
                float(feedback.get("ka", 0.0)),
                "VALID" if feedback.get("feedback_valid") else "INVALID",
                "Y" if feedback.get("feedback_saturated") else "N",
            ),
            COLOR_GOOD if feedback.get("feedback_valid") else COLOR_WARN,
        )
        self._draw_text(
            4,
            "T cap={} pre={} kpu={} post={} ai={} fb={} frm={}".format(
                int(capture_ms),
                int(detector.preprocess_ms),
                int(detector.kpu_ms),
                int(detector.postprocess_ms),
                int(inference_ms),
                int(feedback.get("compute_ms", 0)),
                int(frame_interval_ms),
            ),
            COLOR_TITLE,
        )
        self._draw_text(
            5,
            "{} reject={} acq={}/{}".format(
                detector.diagnostic_text(),
                int(state.get("association_reject_count", 0)),
                int(state.get("acquisition_count", 0)),
                int(state.get("acquisition_required", 3)),
            ),
            COLOR_BAD if state.get("association_rejected") else COLOR_TITLE,
        )
        self._draw_text(
            6,
            "UART TX f={} b={} repl={} q={}/{}B".format(
                tx.sent_frames,
                tx.bytes_sent,
                tx.replaced_frames,
                tx.queued_bytes(),
                UART_SOFTWARE_BUFFER_BYTES,
            ),
            COLOR_GOOD,
        )
        self._draw_text(
            7,
            "UART err={} short={} zero={} drop={}".format(
                tx.write_errors,
                tx.short_writes,
                tx.zero_writes,
                tx.dropped_error_frames,
            ),
            COLOR_BAD if (
                tx.write_errors or tx.short_writes or tx.zero_writes
                or tx.dropped_error_frames
            ) else COLOR_GOOD,
        )
        visible_error = runtime_error
        if visible_error is None:
            visible_error = tx.last_error
        self._draw_text(
            8,
            "last: " + str(visible_error)[:70],
            COLOR_BAD if visible_error is not None else COLOR_GOOD,
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
    tx,
    runtime_error,
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
        "phys[p={:+.2f},v={:+.2f},a={:+.1f}] "
        "fb[a={:+.1f},kp={:.2f},kd={:.2f},ka={:.3f},valid={},sat={}] "
        "model={} diag={} time[cap={},pre={},kpu={},post={},ai={}]ms "
        "UART[f={},b={},q={},e={},s={},z={},d={}] "
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
            float(feedback.get("position_cm", 0.0)),
            float(feedback.get("velocity_cm_s", 0.0)),
            float(feedback.get("acceleration_cm_s2", 0.0)),
            float(feedback.get("feedback_acceleration_cm_s2", 0.0)),
            float(feedback.get("kp", 0.0)),
            float(feedback.get("kd", 0.0)),
            float(feedback.get("ka", 0.0)),
            "Y" if feedback.get("feedback_valid") else "N",
            "Y" if feedback.get("feedback_saturated") else "N",
            detector.model_backend,
            detector.diagnostic_text(),
            int(capture_ms),
            int(detector.preprocess_ms),
            int(detector.kpu_ms),
            int(detector.postprocess_ms),
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
    feedback_filter = None
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
        tracker = PipeMotionFilter(
            max_coast_ms=TRACK_HOLD_MAX_MS,
            max_coast_frames=TRACK_HOLD_MAX_FRAMES,
        )
        stage = "feedback_init"
        feedback_filter = BallStateFeedback(
            negative_end_x_px=PIPE_NEGATIVE_END_X_PX,
            negative_end_y_px=PIPE_NEGATIVE_END_Y_PX,
            positive_end_x_px=PIPE_POSITIVE_END_X_PX,
            positive_end_y_px=PIPE_POSITIVE_END_Y_PX,
            pipe_length_cm=PIPE_LENGTH_CM,
            kp=FEEDBACK_KP,
            kd=FEEDBACK_KD,
            ka=FEEDBACK_KA,
            max_feedback_accel_cm_s2=FEEDBACK_ACCEL_LIMIT_CM_S2,
            max_prediction_age_ms=TRACK_HOLD_MAX_MS,
            max_prediction_frames=TRACK_HOLD_MAX_FRAMES,
        )
        stage = "detector_init"
        detector = _create_detector()
        print(
            "steel-ball tracker started, version={}, backend={}, model={}, "
            "expected={}, UART={} 8N1, frame={}B, software_tx={}B, "
            "pipeline={}, fb_kp={:.2f}, fb_kd={:.2f}, fb_ka={:.3f}, "
            "px_per_cm={:.3f}, pc_debug={}".format(
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
                float(feedback_filter.ka),
                float(feedback_filter.pixels_per_cm),
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
                capture_ms = max(
                    0,
                    _ticks_diff(capture_finished_ms, capture_start_ms),
                )
                stage = "model_inference"
                inference_start_ms = capture_finished_ms
                detections = detector.run(detector_input)
                finished_ms = _ticks_ms()
                inference_ms = max(
                    0,
                    _ticks_diff(finished_ms, inference_start_ms),
                )
                stage = "target_association"
                candidate = tracker.select_candidate(detections, finished_ms)
                stage = "motion_filter"
                state = tracker.update(candidate, finished_ms)
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
                    detector,
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
                    feedback,
                    inference_ms,
                    capture_ms,
                    detector,
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
