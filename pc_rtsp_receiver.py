"""PC-only K230 RTSP receiver with UDP discovery and bounded reconnect.

Run this file with desktop CPython. It is not imported by K230 main.py.
"""

import os
import socket
import time


# FFmpeg may otherwise choose UDP, where a lost RTP packet corrupts an H.264
# reference frame. Set this before opening VideoCapture.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2


UDP_PORT = 19124
SESSION_NAME = "k230"
DISCOVERY_TIMEOUT_S = 12.0
RECONNECT_DELAY_S = 1.0
REDISCOVER_AFTER_FAILURES = 3
OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000


def opencv_package_summary():
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return "package metadata unavailable"
    packages = []
    for name in (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ):
        try:
            packages.append("{}={}".format(name, version(name)))
        except PackageNotFoundError:
            pass
    return ", ".join(packages) if packages else "no OpenCV wheel metadata"


def initialize_window():
    try:
        cv2.namedWindow("K230", cv2.WINDOW_NORMAL)
        return True
    except Exception as exc:
        print("[GUI] 当前 cv2 不包含可用的窗口后端：", exc)
        print("[GUI] installed:", opencv_package_summary())
        print("[GUI] 请在运行本脚本的同一个 Python 环境执行：")
        print(
            "  python -m pip uninstall -y opencv-python-headless "
            "opencv-contrib-python-headless"
        )
        print(
            "  python -m pip install --upgrade --force-reinstall "
            "opencv-python"
        )
        return False


def discover(timeout_s=DISCOVERY_TIMEOUT_S):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind(("0.0.0.0", UDP_PORT))
    receiver.settimeout(0.5)
    deadline = time.monotonic() + float(timeout_s)
    try:
        print("[发现] 等待 K230...")
        while time.monotonic() < deadline:
            try:
                data, address = receiver.recvfrom(256)
            except socket.timeout:
                continue
            port = 8554
            try:
                fields = data.decode("utf-8", "replace").split(":")
                for field in fields:
                    if field.isdigit():
                        port = int(field)
                        break
            except Exception:
                pass
            print("[发现] {}:{}".format(address[0], port))
            return address[0], port
    finally:
        receiver.close()
    return None, None


def open_capture(url):
    parameters = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        parameters.extend((cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS))
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        parameters.extend((cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS))
    try:
        if parameters:
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, parameters)
            if capture.isOpened():
                return capture
            capture.release()
            print("[RTSP] parameterized open failed; trying compatibility mode")
    except Exception as exc:
        print("[RTSP] timeout parameters unavailable:", exc)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def main():
    if not initialize_window():
        return 2
    ip_address = None
    port = None
    consecutive_failures = 0
    try:
        while True:
            if ip_address is None:
                ip_address, port = discover()
                if ip_address is None:
                    time.sleep(RECONNECT_DELAY_S)
                    continue

            url = "rtsp://{}:{}/{}".format(
                ip_address,
                port,
                SESSION_NAME,
            )
            print("[RTSP/TCP]", url)
            capture = open_capture(url)
            stream_opened = capture.isOpened()
            received_frame = False
            if stream_opened:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        print("[RTSP] read failed; reconnecting")
                        break
                    received_frame = True
                    consecutive_failures = 0
                    cv2.imshow("K230", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        return 0
            else:
                print("[RTSP] open failed; reconnecting")
            capture.release()

            if not received_frame:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            if consecutive_failures >= REDISCOVER_AFTER_FAILURES:
                ip_address = None
                port = None
                consecutive_failures = 0
            time.sleep(RECONNECT_DELAY_S)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
