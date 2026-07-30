"""Minimal K230 probe for fixed-ROI input to cv_lite Hough circles.

Run temporarily from CanMV after stopping /sdcard/main.py. This script does not
import the application, open UART, load a model, or run the control law.
"""

import gc
import os
import sys
import time

import cv_lite
import image
import ulab.numpy as np
from media.display import Display
from media.media import MediaManager
from media.sensor import *


FRAME_W = 320
FRAME_H = 320
DISPLAY_W = 640
DISPLAY_H = 480
ROI_X = 0
ROI_Y = 128
ROI_W = 320
ROI_H = 64

DP = 1
MIN_DIST = 6
CANNY_HIGH = 80
ACCUMULATOR = 8
MIN_RADIUS = 2
MAX_RADIUS = 14

WHITE = (255, 255, 255, 255)
GREEN = (255, 0, 255, 120)
YELLOW = (255, 255, 220, 0)
RED = (255, 255, 60, 60)
CYAN = (255, 0, 255, 255)


def ticks_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.time() * 1000)


def ticks_diff(new_value, old_value):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(int(new_value), int(old_value))
    return int(new_value) - int(old_value)


def sleep_ms(value):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(int(value))
    else:
        time.sleep(float(value) / 1000.0)


def is_ide_interrupt(exc):
    if isinstance(exc, KeyboardInterrupt):
        return True
    try:
        return "ide interrupt" in str(exc).strip().lower()
    except Exception:
        return False


def shape_text(array):
    try:
        return "x".join(str(int(value)) for value in array.shape)
    except Exception:
        return "unknown"


def validate_hwc(array, width, height):
    shape = array.shape
    if (
        len(shape) != 3
        or int(shape[0]) != int(height)
        or int(shape[1]) != int(width)
        or int(shape[2]) != 3
    ):
        raise RuntimeError(
            "shape {} != {}x{}x3".format(shape_text(array), height, width)
        )


def run_hough(array, width, height):
    validate_hwc(array, width, height)
    started = ticks_ms()
    circles = cv_lite.rgb888_find_circles(
        [int(height), int(width)],
        array,
        DP,
        MIN_DIST,
        CANNY_HIGH,
        ACCUMULATOR,
        MIN_RADIUS,
        MAX_RADIUS,
    )
    elapsed = max(0, ticks_diff(ticks_ms(), started))
    if circles is None:
        circles = ()
    if len(circles) % 3:
        raise RuntimeError("circle output tail {}".format(len(circles) % 3))
    return circles, len(circles) // 3, elapsed


METHODS = (
    ("image.copy(roi=)", "image_keyword"),
    ("image.copy(tuple)", "image_positional"),
    ("ndarray.copy", "array_copy"),
    ("np.array(view)", "array_constructor"),
)


def extract_roi(method, frame, full_array):
    roi_image = None
    roi_rgb = None
    started = ticks_ms()
    if method == "image_keyword":
        copy_method = getattr(frame, "copy", None)
        if copy_method is None:
            raise RuntimeError("image.copy missing")
        roi_image = copy_method(roi=(ROI_X, ROI_Y, ROI_W, ROI_H))
        roi_rgb = roi_image.to_rgb888()
        roi_array = roi_rgb.to_numpy_ref()
    elif method == "image_positional":
        copy_method = getattr(frame, "copy", None)
        if copy_method is None:
            raise RuntimeError("image.copy missing")
        roi_image = copy_method((ROI_X, ROI_Y, ROI_W, ROI_H))
        roi_rgb = roi_image.to_rgb888()
        roi_array = roi_rgb.to_numpy_ref()
    else:
        if full_array is None:
            raise RuntimeError("full RGB array unavailable")
        roi_view = full_array[
            ROI_Y:ROI_Y + ROI_H,
            ROI_X:ROI_X + ROI_W,
            :,
        ]
        if method == "array_copy":
            copy_method = getattr(roi_view, "copy", None)
            if copy_method is None:
                raise RuntimeError("ndarray.copy missing")
            roi_array = copy_method()
        elif method == "array_constructor":
            roi_array = np.array(roi_view)
        else:
            raise RuntimeError("unknown method " + str(method))
    validate_hwc(roi_array, ROI_W, ROI_H)
    extract_ms = max(0, ticks_diff(ticks_ms(), started))
    return roi_image, roi_rgb, roi_array, extract_ms


def short_error(exc):
    return "{}: {}".format(type(exc).__name__, exc)[:54]


def ema(old_value, sample):
    if old_value is None:
        return float(sample)
    return old_value + 0.15 * (float(sample) - old_value)


def draw_text(osd, row, text, color=WHITE):
    osd.draw_string_advanced(5, 4 + int(row) * 25, 16, str(text), color=color)


def show_status(osd, status, reports, circles):
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_W, 330, color=(210, 0, 0, 0), fill=True)
    scale_x = DISPLAY_W / float(FRAME_W)
    scale_y = DISPLAY_H / float(FRAME_H)
    osd.draw_rectangle(
        int(ROI_X * scale_x),
        int(ROI_Y * scale_y),
        int(ROI_W * scale_x),
        int(ROI_H * scale_y),
        color=CYAN,
        thickness=2,
    )
    if circles:
        osd.draw_circle(
            int((float(circles[0]) + ROI_X) * scale_x),
            int((float(circles[1]) + ROI_Y) * scale_y),
            max(2, int(float(circles[2]) * scale_x)),
            color=YELLOW,
            thickness=2,
        )
    draw_text(osd, 0, "K230 HOUGH FIXED-ROI API PROBE")
    draw_text(osd, 1, "selected: " + status["method"], GREEN)
    draw_text(
        osd,
        2,
        "FULL color={:.1f} hough={:.1f} c={}".format(
            status["full_color"] or 0.0,
            status["full_hough"] or 0.0,
            status["full_count"],
        ),
    )
    draw_text(
        osd,
        3,
        "ROI crop={:.1f} hough={:.1f} c={} err={}".format(
            status["roi_extract"] or 0.0,
            status["roi_hough"] or 0.0,
            status["roi_count"],
            status["errors"],
        ),
        GREEN if status["errors"] == 0 else YELLOW,
    )
    speedup = 0.0
    if status["full_hough"] and status["roi_hough"]:
        speedup = status["full_hough"] / status["roi_hough"]
    draw_text(osd, 4, "Hough speedup x{:.2f}".format(speedup), GREEN)
    for index in range(min(4, len(reports))):
        name, ok, text = reports[index]
        draw_text(osd, 6 + index, name + " " + text, GREEN if ok else YELLOW)
    draw_text(osd, 11, "last: " + status["last_error"], YELLOW)
    draw_text(osd, 12, "Gray not tested: cv_lite API requires RGB888")
    Display.show_image(osd, 0, 0, Display.LAYER_OSD3)


def show_fatal(osd, stage, exc):
    print("HOUGH ROI PROBE FATAL", stage, type(exc).__name__, exc)
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_W, DISPLAY_H, color=(255, 0, 0, 0), fill=True)
    draw_text(osd, 0, "K230 HOUGH ROI PROBE FATAL", RED)
    draw_text(osd, 2, "stage: " + str(stage)[:58], YELLOW)
    draw_text(osd, 3, "type: " + type(exc).__name__, RED)
    message = str(exc)
    draw_text(osd, 4, "error: " + message[:58], RED)
    draw_text(osd, 5, "       " + message[58:116], RED)
    draw_text(osd, 7, "See IDE console; stop manually.")
    Display.show_image(osd, 0, 0, Display.LAYER_OSD3)


def probe_methods(frame, full_array):
    reports = []
    selected = None
    for name, method in METHODS:
        try:
            roi_image, roi_rgb, roi_array, extract_ms = extract_roi(
                method, frame, full_array
            )
            circles, count, hough_ms = run_hough(roi_array, ROI_W, ROI_H)
            reports.append((
                name,
                True,
                "OK {} c={} t={}/{}".format(
                    shape_text(roi_array), count, extract_ms, hough_ms
                ),
            ))
            if selected is None:
                selected = method
            circles = None
            roi_array = None
            roi_rgb = None
            roi_image = None
        except Exception as exc:
            if is_ide_interrupt(exc):
                raise KeyboardInterrupt()
            reports.append((name, False, "ERR " + short_error(exc)))
        gc.collect()
    return reports, selected


def run():
    sensor = None
    osd = None
    display_started = False
    media_started = False
    sensor_started = False
    stage = "startup"
    status = {
        "method": "NONE",
        "full_color": None,
        "full_hough": None,
        "full_count": 0,
        "roi_extract": None,
        "roi_hough": None,
        "roi_count": 0,
        "errors": 0,
        "last_error": "none",
    }
    reports = []
    selected = None
    last_circles = ()
    frame_count = 0
    last_display = 0
    last_print = 0
    try:
        stage = "camera_setup"
        sensor = Sensor(fps=30)
        sensor.reset()
        sensor.set_framesize(w=DISPLAY_W, h=DISPLAY_H, chn=CAM_CHN_ID_0)
        sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420, chn=CAM_CHN_ID_0)
        sensor.set_framesize(w=FRAME_W, h=FRAME_H, chn=CAM_CHN_ID_2)
        sensor.set_pixformat(PIXEL_FORMAT_RGB_565, chn=CAM_CHN_ID_2)
        osd = image.Image(DISPLAY_W, DISPLAY_H, image.ARGB8888)
        Display.bind_layer(
            **sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0),
            layer=Display.LAYER_VIDEO1
        )
        Display.init(
            Display.ST7701,
            width=DISPLAY_W,
            height=DISPLAY_H,
            osd_num=1,
            to_ide=True,
        )
        display_started = True
        MediaManager.init()
        media_started = True
        sensor.run()
        sensor_started = True

        while True:
            exitpoint = getattr(os, "exitpoint", None)
            if exitpoint is not None:
                exitpoint()
            stage = "capture"
            frame = sensor.snapshot(chn=CAM_CHN_ID_2)
            if frame is None:
                raise RuntimeError("camera returned no frame")
            full_rgb = None
            full_array = None
            need_full = (
                selected is None
                or frame_count % 10 == 0
                or selected in ("array_copy", "array_constructor")
            )
            if need_full:
                stage = "full_color"
                started = ticks_ms()
                full_rgb = frame.to_rgb888()
                full_array = full_rgb.to_numpy_ref()
                validate_hwc(full_array, FRAME_W, FRAME_H)
                color_ms = max(0, ticks_diff(ticks_ms(), started))
                if frame_count % 10 == 0:
                    stage = "full_hough"
                    full_circles, count, hough_ms = run_hough(
                        full_array, FRAME_W, FRAME_H
                    )
                    status["full_color"] = ema(status["full_color"], color_ms)
                    status["full_hough"] = ema(status["full_hough"], hough_ms)
                    status["full_count"] = count
                    full_circles = None
            if selected is None:
                stage = "api_probe"
                reports, selected = probe_methods(frame, full_array)
                status["method"] = selected or "NONE"
                print("HOUGH ROI API PROBE RESULTS")
                for report in reports:
                    print(report[0], report[2])
                if selected is None:
                    raise RuntimeError("no fixed-ROI method passed real Hough")

            try:
                stage = "roi_extract"
                roi_image, roi_rgb, roi_array, extract_ms = extract_roi(
                    selected, frame, full_array
                )
                stage = "roi_hough"
                circles, count, hough_ms = run_hough(roi_array, ROI_W, ROI_H)
                status["roi_extract"] = ema(status["roi_extract"], extract_ms)
                status["roi_hough"] = ema(status["roi_hough"], hough_ms)
                status["roi_count"] = count
                status["last_error"] = "none"
                last_circles = circles
                roi_array = None
                roi_rgb = None
                roi_image = None
            except Exception as exc:
                if is_ide_interrupt(exc):
                    raise KeyboardInterrupt()
                status["errors"] += 1
                status["last_error"] = short_error(exc)
                last_circles = ()
                print("ROI runtime error at", stage, status["last_error"])
            full_array = None
            full_rgb = None

            now = ticks_ms()
            if ticks_diff(now, last_display) >= 250:
                show_status(osd, status, reports, last_circles)
                last_display = now
            if ticks_diff(now, last_print) >= 500:
                print(
                    "ROI_PROBE method={} full={:.1f} roi={:.1f} "
                    "extract={:.1f} circles={} errors={}".format(
                        status["method"],
                        status["full_hough"] or 0.0,
                        status["roi_hough"] or 0.0,
                        status["roi_extract"] or 0.0,
                        status["roi_count"],
                        status["errors"],
                    )
                )
                last_print = now
            frame_count = (frame_count + 1) & 0xFFFFFFFF
            if frame_count % 30 == 0:
                gc.collect()
            stage = "runtime"
    except KeyboardInterrupt:
        print("Hough ROI probe stopped by IDE/user")
    except Exception as exc:
        if is_ide_interrupt(exc):
            print("Hough ROI probe stopped by IDE/user")
        else:
            if osd is not None and display_started:
                show_fatal(osd, stage, exc)
            printer = getattr(sys, "print_exception", None)
            if printer is not None:
                printer(exc)
            try:
                while True:
                    exitpoint = getattr(os, "exitpoint", None)
                    if exitpoint is not None:
                        exitpoint()
                    sleep_ms(100)
            except Exception as hold_exc:
                if not is_ide_interrupt(hold_exc):
                    print("fatal hold warning:", hold_exc)
    finally:
        if sensor_started and sensor is not None:
            try:
                sensor.stop()
            except Exception as exc:
                print("sensor stop warning:", exc)
        if display_started:
            try:
                Display.deinit()
            except Exception as exc:
                print("display deinit warning:", exc)
        sleep_ms(50)
        if media_started:
            try:
                MediaManager.deinit()
            except Exception as exc:
                print("media deinit warning:", exc)
        gc.collect()


if __name__ == "__main__":
    run()
