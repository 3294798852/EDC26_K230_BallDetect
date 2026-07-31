"""Optional non-blocking Wi-Fi H.264/RTSP service for the K230 camera.

The application owns Sensor and MediaManager. This service only configures an
unused sensor channel before MediaManager.init(), then owns one VENC channel,
the RTSP server, Wi-Fi reconnect and UDP discovery in a background thread.
"""

import time


STATE_OFF = "OFF"
STATE_PREPARED = "PREP"
STATE_READY = "RDY"
STATE_CONNECTING = "WIFI"
STATE_STREAMING = "RUN"
STATE_RETRY = "RETRY"
STATE_ERROR = "ERR"
STATE_STOPPED = "STOP"


def _ticks_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.time() * 1000)


def _ticks_diff(new_value, old_value):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(int(new_value), int(old_value))
    return int(new_value) - int(old_value)


def _sleep_ms(milliseconds):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(int(milliseconds))
    else:
        time.sleep(float(milliseconds) / 1000.0)


def _sat_add(value, increment=1):
    value = int(value) + int(increment)
    return 0x7FFFFFFF if value > 0x7FFFFFFF else value


class WifiVideoService:
    """Bounded background Sensor-to-VENC-to-RTSP service.

    Lifecycle order is prepare(sensor), MediaManager.init(), bind(),
    sensor.run(), start(), then stop() before the application stops
    Sensor/MediaManager.
    """

    def __init__(
        self,
        ssid,
        password,
        width=512,
        height=288,
        source_fps=60,
        stream_fps=30,
        bit_rate_kbps=1600,
        gop_frames=30,
        rtsp_port=8554,
        session_name="k230",
        discovery_port=19124,
        beacon_interval_ms=10000,
        connect_timeout_ms=30000,
        stream_stall_timeout_ms=3000,
        output_buffers=6,
        prefer_zero_copy=False,
    ):
        self.ssid = "" if ssid is None else str(ssid)
        self.password = "" if password is None else str(password)
        self.width = int(width)
        self.height = int(height)
        self.source_fps = int(source_fps)
        self.stream_fps = int(stream_fps)
        self.bit_rate_kbps = int(bit_rate_kbps)
        self.gop_frames = int(gop_frames)
        self.rtsp_port = int(rtsp_port)
        self.session_name = str(session_name)
        self.discovery_port = int(discovery_port)
        self.beacon_interval_ms = int(beacon_interval_ms)
        self.connect_timeout_ms = int(connect_timeout_ms)
        self.stream_stall_timeout_ms = int(stream_stall_timeout_ms)
        self.output_buffers = int(output_buffers)
        self.prefer_zero_copy = bool(prefer_zero_copy)

        self.state = STATE_OFF
        self.ip_address = "0.0.0.0"
        self.last_error = None
        self.last_warning = None
        self.error_count = 0
        self.reconnect_count = 0
        self.restart_count = 0
        self.encoded_frames = 0
        self.sent_frames = 0
        self.sent_bytes = 0
        self.send_errors = 0
        self.release_errors = 0
        self.beacon_errors = 0
        self.zero_copy_fallbacks = 0
        self.stream_stalls = 0
        self.slow_sends = 0
        self.last_send_ms = 0
        self.max_send_ms = 0
        self.max_pack_count = 0
        self.send_fps = 0.0
        self.send_kbps = 0.0
        self.wifi_link_state = "?"
        self.wifi_connect_elapsed_ms = 0
        self.wifi_timeout_reason = "WAIT"
        self.wifi_ap_seen = "?"
        self.wifi_ap_rssi = None
        self.wifi_ap_security = "?"
        self.wifi_scan_network_count = 0
        self.wifi_scan_count = 0
        self.wifi_scan_error = None
        self.wifi_link_probe_error = None
        self.wifi_ip_probe_error = None

        self._running = False
        self._thread_started = False
        self._thread_exited = True
        self._prepared = False
        self._stream_active = False
        self._encoder_created = False
        self._encoder_started = False
        self._server_initialized = False
        self._server_started = False
        self._zero_copy_active = False
        self._last_beacon_ms = None
        self._rate_started_ms = 0
        self._rate_frames = 0
        self._rate_bytes = 0
        self._last_stream_ms = 0

        self._thread = None
        self._network = None
        self._socket = None
        self._uctypes = None
        self._mm = None
        self._Encoder = None
        self._ChnAttrStr = None
        self._StreamData = None
        self._venc_chn = None
        self._video_encode_mod_id = None
        self._venc_dev_id = None
        self._camera_channel = None
        self._pixel_format = None
        self._MediaManager = None
        self._encoder = None
        self._server = None
        self._link = None
        self._station = None
        self._beacon_socket = None
        self._sensor = None
        self._wifi_connect_attempts = 0
        self._wifi_attempt_started_ms = None

    def _validate(self):
        if not self.ssid:
            raise ValueError("Wi-Fi SSID is empty")
        if self.width <= 0 or self.width % 16:
            raise ValueError("video width must be positive and 16-aligned")
        if self.height <= 0 or self.height % 2:
            raise ValueError("video height must be positive and even")
        if self.source_fps <= 0 or self.stream_fps <= 0:
            raise ValueError("video frame rates must be positive")
        if self.stream_fps > self.source_fps:
            raise ValueError("stream FPS cannot exceed source FPS")
        if self.bit_rate_kbps <= 0 or self.gop_frames <= 0:
            raise ValueError("bit rate and GOP must be positive")
        if self.rtsp_port <= 0 or self.rtsp_port > 65535:
            raise ValueError("invalid RTSP port")
        if self.discovery_port <= 0 or self.discovery_port > 65535:
            raise ValueError("invalid discovery port")
        if self.beacon_interval_ms < 1000:
            raise ValueError("beacon interval must be at least 1000 ms")
        if not self.session_name or "/" in self.session_name:
            raise ValueError("invalid RTSP session name")
        if self.output_buffers < 3 or self.output_buffers > 8:
            raise ValueError("VENC output buffers must be within 3..8")
        if self.connect_timeout_ms <= 0:
            raise ValueError("Wi-Fi connect timeout must be positive")
        if self.stream_stall_timeout_ms <= 0:
            raise ValueError("stream stall timeout must be positive")

    def _record_error(self, stage, exc):
        self.error_count = _sat_add(self.error_count)
        self.last_error = "{} {}: {}".format(
            str(stage),
            type(exc).__name__,
            str(exc),
        )
        print("wifi video error #{}: {}".format(
            self.error_count,
            self.last_error,
        ))

    def prepare(self, sensor):
        """Configure CH1 and reserve VENC buffers before MediaManager.init()."""
        if self._prepared:
            return True
        try:
            self._validate()
            if sensor is None or not hasattr(sensor, "set_framesize"):
                raise ValueError("sensor does not provide camera configuration")

            import _thread
            import network
            import socket
            import uctypes
            import multimedia as mm
            import media.media as media_module
            import media.vencoder as vencoder_module
            from media.media import MediaManager
            from media.sensor import (
                CAM_CHN_ID_1,
                PIXEL_FORMAT_YUV_SEMIPLANAR_420,
            )
            from media.vencoder import (
                ChnAttrStr,
                Encoder,
                StreamData,
                VENC_CHN_ID_0,
            )

            video_encode_mod_id = getattr(
                vencoder_module,
                "VIDEO_ENCODE_MOD_ID",
                getattr(media_module, "VIDEO_ENCODE_MOD_ID", None),
            )
            venc_dev_id = getattr(
                vencoder_module,
                "VENC_DEV_ID",
                getattr(media_module, "VENC_DEV_ID", None),
            )
            if video_encode_mod_id is None or venc_dev_id is None:
                raise ImportError("VENC media-link constants unavailable")

            self._thread = _thread
            self._network = network
            self._socket = socket
            self._uctypes = uctypes
            self._mm = mm
            self._MediaManager = MediaManager
            self._Encoder = Encoder
            self._ChnAttrStr = ChnAttrStr
            self._StreamData = StreamData
            self._venc_chn = VENC_CHN_ID_0
            self._venc_dev_id = venc_dev_id
            self._video_encode_mod_id = video_encode_mod_id
            self._camera_channel = CAM_CHN_ID_1
            self._pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420

            sensor.set_framesize(
                w=self.width,
                h=self.height,
                chn=self._camera_channel,
            )
            sensor.set_pixformat(
                self._pixel_format,
                chn=self._camera_channel,
            )
            self._encoder = self._Encoder()
            self._encoder.SetOutBufs(
                self._venc_chn,
                self.output_buffers,
                self.width,
                self.height,
            )
            try:
                self._start_wifi_connect_attempt()
            except Exception as exc:
                self.last_warning = "wifi preconnect {}: {}".format(
                    type(exc).__name__,
                    str(exc),
                )
                print("wifi video warning: " + self.last_warning)
            self._sensor = sensor
            self._prepared = True
            self.state = STATE_PREPARED
            return True
        except Exception as exc:
            self.state = STATE_ERROR
            self._record_error("prepare", exc)
            self._release_link()
            self._encoder = None
            return False

    def bind(self):
        """Create the media link and encoder after MediaManager.init()."""
        if self._link is not None:
            return True
        if not self._prepared or self._sensor is None:
            return False
        try:
            source = self._sensor.bind_info(
                chn=self._camera_channel
            )["src"]
            target = (
                self._video_encode_mod_id,
                self._venc_dev_id,
                self._venc_chn,
            )
            self._link = self._MediaManager.link(source, target)
            attribute = self._ChnAttrStr(
                self._encoder.PAYLOAD_TYPE_H264,
                self._encoder.H264_PROFILE_MAIN,
                self.width,
                self.height,
                bit_rate=self.bit_rate_kbps,
                gopLen=self.gop_frames,
                src_frame_rate=self.source_fps,
                dst_frame_rate=self.stream_fps,
            )
            self._encoder.Create(self._venc_chn, attribute)
            self._encoder_created = True
            self.state = STATE_READY
            return True
        except Exception as exc:
            self.state = STATE_ERROR
            self._record_error("bind", exc)
            self._destroy_encoder()
            self._release_link()
            self._encoder = None
            return False

    def start(self):
        """Start the worker and return immediately; never wait for Wi-Fi."""
        if self._running:
            return True
        if not self._prepared or self._link is None:
            return False
        self._running = True
        self._thread_exited = False
        self._thread_started = True
        self.state = STATE_CONNECTING
        try:
            self._thread.start_new_thread(self._worker, ())
            return True
        except Exception as exc:
            self._running = False
            self._thread_started = False
            self._thread_exited = True
            self.state = STATE_ERROR
            self._record_error("thread_start", exc)
            return False

    def _sleep_running(self, milliseconds):
        remaining = max(0, int(milliseconds))
        while self._running and remaining > 0:
            interval = min(remaining, 100)
            _sleep_ms(interval)
            remaining -= interval
        return self._running

    def _wifi_snapshot(self):
        if self._station is None:
            return False, "0.0.0.0"
        link_state = False
        try:
            link_state = bool(self._station.isconnected())
            self.wifi_link_state = "Y" if link_state else "N"
            self.wifi_link_probe_error = None
        except Exception as exc:
            self.wifi_link_state = "?"
            self.wifi_link_probe_error = "{}: {}".format(
                type(exc).__name__,
                str(exc),
            )
        try:
            ip_address = self._station.ifconfig()[0]
            ip_address = str(ip_address) if ip_address else "0.0.0.0"
            self.wifi_ip_probe_error = None
        except Exception as exc:
            ip_address = "0.0.0.0"
            self.wifi_ip_probe_error = "{}: {}".format(
                type(exc).__name__,
                str(exc),
            )
        self.ip_address = ip_address
        return link_state, ip_address

    def _wifi_connected(self):
        link_state, ip_address = self._wifi_snapshot()
        if ip_address != "0.0.0.0":
            self.ip_address = ip_address
            self.wifi_timeout_reason = "OK"
            return True
        return False

    def _scan_target_ap(self):
        """Classify an association timeout without retaining scan results."""
        self.wifi_scan_count = _sat_add(self.wifi_scan_count)
        self.wifi_ap_seen = "?"
        self.wifi_ap_rssi = None
        self.wifi_ap_security = "?"
        self.wifi_scan_network_count = 0
        self.wifi_scan_error = None
        try:
            if hasattr(self._station, "disconnect"):
                self._station.disconnect()
                _sleep_ms(1000)
            networks = self._station.scan()
            target_found = False
            if networks is not None:
                for entry in networks:
                    self.wifi_scan_network_count = _sat_add(
                        self.wifi_scan_network_count
                    )
                    try:
                        raw_ssid = getattr(entry, "ssid", None)
                        if raw_ssid is None:
                            continue
                        if isinstance(raw_ssid, (bytes, bytearray)):
                            entry_ssid = raw_ssid.decode("utf-8")
                        else:
                            entry_ssid = str(raw_ssid)
                        if entry_ssid != self.ssid:
                            continue
                        target_found = True
                        raw_rssi = getattr(entry, "rssi", None)
                        if raw_rssi is not None:
                            try:
                                self.wifi_ap_rssi = int(raw_rssi)
                            except Exception:
                                self.wifi_ap_rssi = None
                        raw_security = getattr(entry, "security", None)
                        if raw_security is not None:
                            self.wifi_ap_security = (
                                "Y" if bool(raw_security) else "N"
                            )
                    except Exception:
                        continue
            self.wifi_ap_seen = "Y" if target_found else "N"
            if self.wifi_scan_network_count <= 0:
                self.wifi_timeout_reason = "NO_AP_VISIBLE"
            elif not target_found:
                self.wifi_timeout_reason = "AP_NOT_FOUND"
            elif (
                self.wifi_ap_rssi is not None
                and self.wifi_ap_rssi <= -80
            ):
                self.wifi_timeout_reason = "WEAK_SIGNAL"
            elif self.wifi_ap_security == "N" and self.password:
                self.wifi_timeout_reason = "OPEN_AP_PASSWORD"
            elif self.wifi_ap_security == "Y" and not self.password:
                self.wifi_timeout_reason = "PASSWORD_REQUIRED"
            else:
                self.wifi_timeout_reason = "AUTH_OR_MODE"
        except Exception as exc:
            self.wifi_timeout_reason = "SCAN_ERROR"
            self.wifi_scan_error = "{}: {}".format(
                type(exc).__name__,
                str(exc),
            )

    def _diagnose_wifi_timeout(self):
        if self.wifi_ip_probe_error is not None:
            self.wifi_timeout_reason = "IP_API_ERROR"
            return
        if self.wifi_link_state == "Y":
            self.wifi_timeout_reason = "DHCP_TIMEOUT"
            return
        self._scan_target_ap()

    def _start_wifi_connect_attempt(self):
        if self._station is None:
            station_mode = getattr(self._network, "STA_IF", 0)
            self._station = self._network.WLAN(station_mode)
        if self._wifi_connect_attempts > 0 and hasattr(
            self._station,
            "disconnect",
        ):
            try:
                self._station.disconnect()
                _sleep_ms(200)
            except Exception as exc:
                self.last_warning = "wifi disconnect {}: {}".format(
                    type(exc).__name__,
                    str(exc),
                )
        self._station.connect(self.ssid, self.password)
        self._wifi_connect_attempts = _sat_add(self._wifi_connect_attempts)
        self._wifi_attempt_started_ms = _ticks_ms()
        self.wifi_connect_elapsed_ms = 0

    def _connect_wifi(self):
        self.state = STATE_CONNECTING
        if self._wifi_connected():
            return True
        if self._wifi_attempt_started_ms is None:
            self._start_wifi_connect_attempt()
        while self._running:
            if self._wifi_connected():
                return True
            self.wifi_connect_elapsed_ms = max(
                0,
                _ticks_diff(_ticks_ms(), self._wifi_attempt_started_ms),
            )
            if self.wifi_connect_elapsed_ms >= self.connect_timeout_ms:
                self._wifi_attempt_started_ms = None
                self._diagnose_wifi_timeout()
                return False
            _sleep_ms(100)
        return False

    def _open_stream(self):
        self._server = self._mm.rtsp_server()
        self._server.rtspserver_init(self.rtsp_port)
        self._server_initialized = True
        self._server.rtspserver_createsession(
            self.session_name,
            self._mm.multi_media_type.media_h264,
            False,
        )
        self._server.rtspserver_start()
        self._server_started = True
        self._encoder.Start(self._venc_chn)
        self._encoder_started = True
        self._zero_copy_active = bool(
            self.prefer_zero_copy
            and hasattr(
                self._server,
                "rtspserver_sendvideodata_byphyaddr",
            )
        )
        self._open_beacon()
        self._stream_active = True
        self.state = STATE_STREAMING
        now_ms = _ticks_ms()
        self._last_beacon_ms = None
        self._rate_started_ms = now_ms
        self._rate_frames = 0
        self._rate_bytes = 0
        self._last_stream_ms = now_ms
        print("wifi video RTSP: {}".format(self.rtsp_url()))

    def _open_beacon(self):
        try:
            beacon_socket = self._socket.socket(
                self._socket.AF_INET,
                self._socket.SOCK_DGRAM,
            )
            beacon_socket.setsockopt(
                self._socket.SOL_SOCKET,
                self._socket.SO_BROADCAST,
                1,
            )
            # K230 examples verify setblocking(False). Discovery is best-effort
            # and must never pause the VENC drain path on a congested radio.
            beacon_socket.setblocking(False)
            self._beacon_socket = beacon_socket
        except Exception as exc:
            self.beacon_errors = _sat_add(self.beacon_errors)
            self.last_warning = "beacon_open {}: {}".format(
                type(exc).__name__,
                str(exc),
            )
            self._close_beacon()

    def _send_beacon_if_due(self):
        if self._beacon_socket is None:
            return
        now_ms = _ticks_ms()
        if (
            self._last_beacon_ms is not None
            and _ticks_diff(now_ms, self._last_beacon_ms)
            < self.beacon_interval_ms
        ):
            return
        self._last_beacon_ms = now_ms
        message = "K230VIDEO/1:{}:{}".format(
            self.rtsp_port,
            self.session_name,
        )
        try:
            self._beacon_socket.sendto(
                message,
                ("255.255.255.255", self.discovery_port),
            )
        except Exception as exc:
            self.beacon_errors = _sat_add(self.beacon_errors)
            self.last_warning = "beacon_send {}: {}".format(
                type(exc).__name__,
                str(exc),
            )

    def _send_pack(self, stream_data, pack_index, data_size):
        if self._zero_copy_active:
            try:
                physical_address = stream_data.phy_addr[pack_index]
            except Exception as exc:
                self._zero_copy_active = False
                self.zero_copy_fallbacks = _sat_add(
                    self.zero_copy_fallbacks
                )
                self.last_warning = "zero_copy_fallback {}: {}".format(
                    type(exc).__name__,
                    str(exc),
                )
            else:
                result = self._server.rtspserver_sendvideodata_byphyaddr(
                    self.session_name,
                    physical_address,
                    data_size,
                    1000,
                )
                if result not in (None, 0):
                    raise RuntimeError(
                        "RTSP physical-address send returned {}".format(
                            result
                        )
                    )
                return

        payload = bytes(
            self._uctypes.bytearray_at(
                stream_data.data[pack_index],
                data_size,
            )
        )
        result = self._server.rtspserver_sendvideodata(
            self.session_name,
            payload,
            data_size,
            1000,
        )
        if result not in (None, 0):
            raise RuntimeError("RTSP copied send returned {}".format(result))

    def _send_one_stream(self, stream_data):
        result = self._encoder.GetStream(
            self._venc_chn,
            stream_data,
            timeout=0,
        )
        if result != 0:
            _sleep_ms(2)
            return False

        send_started_ms = _ticks_ms()
        send_exception = None
        sent_bytes = 0
        try:
            pack_count = int(stream_data.pack_cnt)
            if pack_count > self.max_pack_count:
                self.max_pack_count = pack_count
            for pack_index in range(pack_count):
                data_size = int(stream_data.data_size[pack_index])
                if data_size <= 0:
                    continue
                self._send_pack(stream_data, pack_index, data_size)
                sent_bytes += data_size
        except Exception as exc:
            send_exception = exc
        finally:
            try:
                self._encoder.ReleaseStream(self._venc_chn, stream_data)
            except Exception as release_exc:
                self.release_errors = _sat_add(self.release_errors)
                self._record_error("release_stream", release_exc)
                if send_exception is None:
                    send_exception = release_exc

        self.encoded_frames = _sat_add(self.encoded_frames)
        if send_exception is not None:
            self.send_errors = _sat_add(self.send_errors)
            raise send_exception
        self.sent_frames = _sat_add(self.sent_frames)
        self.sent_bytes = _sat_add(self.sent_bytes, sent_bytes)
        self._last_stream_ms = _ticks_ms()
        self.last_send_ms = max(
            0,
            _ticks_diff(self._last_stream_ms, send_started_ms),
        )
        if self.last_send_ms > self.max_send_ms:
            self.max_send_ms = self.last_send_ms
        if self.last_send_ms > max(20, int(2000 / self.stream_fps)):
            self.slow_sends = _sat_add(self.slow_sends)
        self._rate_frames += 1
        self._rate_bytes += sent_bytes
        now_ms = _ticks_ms()
        rate_elapsed_ms = _ticks_diff(now_ms, self._rate_started_ms)
        if rate_elapsed_ms >= 1000:
            self.send_fps = (
                float(self._rate_frames) * 1000.0 / rate_elapsed_ms
            )
            self.send_kbps = (
                float(self._rate_bytes) * 8.0 / rate_elapsed_ms
            )
            self._rate_started_ms = now_ms
            self._rate_frames = 0
            self._rate_bytes = 0
        return True

    def _worker(self):
        retry_ms = 1000
        try:
            while self._running:
                try:
                    if not self._connect_wifi():
                        raise RuntimeError(
                            "wifi timeout cause={} ap={} total={} rssi={} sec={} "
                            "link={} ip={} scanerr={}".format(
                                self.wifi_timeout_reason,
                                self.wifi_ap_seen,
                                self.wifi_scan_network_count,
                                self.wifi_ap_rssi,
                                self.wifi_ap_security,
                                self.wifi_link_state,
                                self.ip_address,
                                self.wifi_scan_error,
                            )
                        )
                    self._open_stream()
                    stable_started_ms = _ticks_ms()
                    stream_data = self._StreamData()
                    while self._running and self._wifi_connected():
                        self._send_beacon_if_due()
                        stream_sent = self._send_one_stream(stream_data)
                        if (
                            not stream_sent
                            and _ticks_diff(
                                _ticks_ms(),
                                self._last_stream_ms,
                            ) >= self.stream_stall_timeout_ms
                        ):
                            self.stream_stalls = _sat_add(self.stream_stalls)
                            raise RuntimeError(
                                "VENC stream stall {}ms".format(
                                    self.stream_stall_timeout_ms
                                )
                            )
                        if (
                            _ticks_diff(_ticks_ms(), stable_started_ms)
                            >= 10000
                        ):
                            retry_ms = 1000
                    if self._running:
                        self.reconnect_count = _sat_add(self.reconnect_count)
                        raise RuntimeError("Wi-Fi disconnected")
                except Exception as exc:
                    if not self._running:
                        break
                    self.state = STATE_RETRY
                    self._record_error("worker", exc)
                    self.restart_count = _sat_add(self.restart_count)
                finally:
                    self._close_stream()

                if not self._sleep_running(retry_ms):
                    break
                retry_ms = min(16000, retry_ms * 2)
        finally:
            self._close_stream()
            self._thread_exited = True
            self._thread_started = False
            if self.state != STATE_ERROR:
                self.state = STATE_STOPPED

    def _close_beacon(self):
        beacon_socket = self._beacon_socket
        self._beacon_socket = None
        if beacon_socket is not None:
            try:
                beacon_socket.close()
            except Exception:
                pass

    def _close_stream(self):
        self._stream_active = False
        self._close_beacon()
        if self._server_started and self._server is not None:
            try:
                self._server.rtspserver_stop()
            except Exception as exc:
                self._record_error("rtsp_stop", exc)
        self._server_started = False
        if self._encoder_started and self._encoder is not None:
            try:
                self._encoder.Stop(self._venc_chn)
            except Exception as exc:
                self._record_error("encoder_stop", exc)
        self._encoder_started = False
        if self._server_initialized and self._server is not None:
            try:
                self._server.rtspserver_deinit()
            except Exception as exc:
                self._record_error("rtsp_deinit", exc)
        self._server_initialized = False
        self._server = None
        self._zero_copy_active = False
        self.send_fps = 0.0
        self.send_kbps = 0.0
        self._last_stream_ms = 0

    def _destroy_encoder(self):
        if self._encoder_created and self._encoder is not None:
            try:
                self._encoder.Destroy(self._venc_chn)
            except Exception as exc:
                self._record_error("encoder_destroy", exc)
        self._encoder_created = False

    def _release_link(self):
        link = self._link
        self._link = None
        if link is not None:
            del link
        self._prepared = False
        self._sensor = None

    def stop(self, timeout_ms=2500):
        """Request worker exit before Sensor and MediaManager are destroyed."""
        self._running = False
        wait_started_ms = _ticks_ms()
        timeout_ms = max(0, int(timeout_ms))
        while self._thread_started and not self._thread_exited:
            if _ticks_diff(_ticks_ms(), wait_started_ms) >= timeout_ms:
                self.state = STATE_ERROR
                self._record_error(
                    "shutdown",
                    RuntimeError("worker exit timeout"),
                )
                return False
            _sleep_ms(10)
        self._close_stream()
        if self.state != STATE_ERROR:
            self.state = STATE_STOPPED
        return True

    def release(self):
        """Destroy VENC/link after Sensor has stopped, before media deinit."""
        if self._thread_started and not self._thread_exited:
            self._record_error(
                "release",
                RuntimeError("worker still active"),
            )
            return False
        self._close_stream()
        self._destroy_encoder()
        self._release_link()
        self._encoder = None
        return True

    def rtsp_url(self):
        return "rtsp://{}:{}/{}".format(
            self.ip_address,
            self.rtsp_port,
            self.session_name,
        )

    def diagnostic_text(self):
        if self.state == STATE_STREAMING:
            return "NET {} {} f={:.1f} kb={:.0f} z={} e={} r={} s={} tx={}/{} p={}".format(
                self.state,
                self.ip_address,
                self.send_fps,
                self.send_kbps,
                "Y" if self._zero_copy_active else "N",
                self.error_count,
                self.restart_count,
                self.stream_stalls,
                self.last_send_ms,
                self.max_send_ms,
                self.max_pack_count,
            )
        error_text = ""
        if self.last_error is not None:
            error_text = str(self.last_error)
            if "wifi timeout" in error_text:
                error_text = ""
            else:
                error_text = " " + error_text[:18]
        if self.wifi_timeout_reason == "SCAN_ERROR" and self.wifi_scan_error:
            error_text = " " + self.wifi_scan_error[:18]
        elif (
            self.wifi_timeout_reason == "IP_API_ERROR"
            and self.wifi_ip_probe_error
        ):
            error_text = " " + self.wifi_ip_probe_error[:18]
        rssi_text = "-" if self.wifi_ap_rssi is None else str(
            self.wifi_ap_rssi
        )
        ip_text = "0" if self.ip_address == "0.0.0.0" else self.ip_address
        return "NET {} r={} e={} why={} n={} ap={}/{}/{} l={} ip={} t={:.0f}s{}".format(
            self.state,
            self.restart_count,
            self.error_count,
            self.wifi_timeout_reason,
            self.wifi_scan_network_count,
            self.wifi_ap_seen,
            rssi_text,
            self.wifi_ap_security,
            self.wifi_link_state,
            ip_text,
            float(self.wifi_connect_elapsed_ms) / 1000.0,
            error_text,
        )


class DisabledWifiVideoService:
    """No-op service used when optional Wi-Fi video is not configured."""

    def __init__(self, reason="disabled"):
        self.state = STATE_OFF
        self.last_error = None
        self.reason = str(reason)

    def prepare(self, sensor):
        return False

    def bind(self):
        return False

    def start(self):
        return False

    def stop(self, timeout_ms=0):
        return True

    def release(self):
        return True

    def rtsp_url(self):
        return ""

    def diagnostic_text(self):
        return "NET OFF " + self.reason[:52]
