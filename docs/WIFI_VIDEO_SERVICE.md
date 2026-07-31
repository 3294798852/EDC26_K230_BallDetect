# K230 optional Wi-Fi video service

> **Development status:** unfinished experimental implementation. Wi-Fi and
> RTSP have worked in limited tests, but the current K230 integration is not
> stable: some builds fail to start video, while others develop H.264 decoder
> corruption and frequent stalls. It is retained for later diagnosis and must
> not be treated as a production-ready monitoring path.

## Purpose and boundary

`wifi_video_service.py` is an optional service used by `main.py`. It does not
read detector frames, change tracking/control/UART data, initialize a second
Sensor, or call `MediaManager.init/deinit`. The application keeps ownership of
the camera, display and global media manager.

The media path is:

```text
Sensor CH0 640x480 YUV -> local LCD
Sensor CH2 320x320     -> detector and control
Sensor CH1 320x240 YUV -> hardware VENC0 -> H.264 -> RTSP/Wi-Fi worker
```

CH1 was selected because the current application already owns CH0 and CH2.
The worker performs Wi-Fi connection, non-blocking VENC polling, RTSP sending,
UDP discovery and bounded retry. The vision loop only starts/stops the service
and reads its diagnostic counters for the LCD/PCDBG line.
The RTSP view is intentionally raw camera video: local OSD, detection boxes and
control text are not composited into CH1. Adding them would require another
per-frame image path and would undermine the low-overhead isolation goal.

## Enable and configure

`WIFI_VIDEO_ENABLED=True` in `main.py` permits the feature, but credentials are
not stored in Git. Create `/sdcard/wifi_video_secrets.py` on the K230:

```python
WIFI_VIDEO_SSID = "your access point"
WIFI_VIDEO_PASSWORD = "your password"
```

The file is ignored by Git. Do not print or display its values. If it is
absent, has an empty SSID, or the feature switch is false, the screen shows
`NET OFF` and the original vision/UART path runs without configuring CH1 or
VENC.

Default stream parameters are configured near the camera constants in
`main.py`:

| Parameter | Default | Reason |
| --- | ---: | --- |
| Resolution | `320x240` | conservative debug view with much smaller NAL/RTP fragmentation peaks |
| Stream FPS | `15` | sufficient for monitoring while reducing Python/RTSP scheduling pressure |
| H.264 rate | `600 kb/s` | conservative local-Wi-Fi load; confirm the SDK rate unit on target |
| GOP | `15` frames | approximately one IDR per second at 15 FPS |
| VENC buffers | `8` | SDK/example-proven maximum to absorb bounded scheduling jitter |
| VENC stall watchdog | `3000 ms` | restart RTSP/VENC if no encoded pack is produced |
| RTSP | port `8554`, session `k230` | URL `rtsp://<board-ip>:8554/k230` |
| Discovery | UDP broadcast port `19124` | non-blocking payload `K230VIDEO/1:8554:k230` every ten seconds |

The source frame rate passed to VENC matches `CAMERA_FPS`; VENC requests the
configured 15 FPS output. This uses `ChnAttrStr` arguments already exercised by
the reference K230 program. Whether every sensor mode performs hardware frame
rate conversion exactly as requested still requires target measurement.

## Interface

The service lifecycle is deliberately small:

```python
service = WifiVideoService(...)
service.prepare(sensor)  # before MediaManager.init() and sensor.run()
service.bind()           # after MediaManager.init(), before sensor.run()
service.start()          # after sensor.run(); returns immediately
status = service.diagnostic_text()
service.stop()           # stop worker and VENC before sensor.stop()
service.release()        # after sensor.stop(), before MediaManager.deinit()
```

`prepare()` configures CH1, reserves six encoder buffers and issues one
non-blocking `WLAN.connect()` request from the main thread before media
initialization, matching the supplied working K230 Wi-Fi call order. `bind()`
creates the media link and encoder channel only after the application's global
media manager is ready. `start()` launches one background worker, which waits
for the already-started connection and handles all later retries; the main
thread never waits for Wi-Fi.
`stop()` requests worker exit and waits at most 2.5 seconds. The camera owner
then stops Sensor and calls `release()` to destroy VENC and unlink media before
global media deinit. Repeated calls are safe. `DisabledWifiVideoService`
implements the same interface and permits the main program to avoid
feature-specific branches.

## Reliability behavior

- Wi-Fi connection waits only inside the worker and times out after 30 seconds.
  A nonzero IPv4 address is authoritative, matching the supplied working K230
  reference. This also tolerates firmware variants whose `isconnected()` state
  lags DHCP or is inconsistent while `ifconfig()` already contains a valid IP.
- A failed attempt is disconnected once before the next bounded retry, so a
  stale association/DHCP attempt is not repeatedly reused. The LCD error keeps
  the last link flag, IP and timeout duration without revealing credentials.
- A disconnect or RTSP/VENC error closes the current stream and retries with
  exponential backoff bounded at 16 seconds.
- `GetStream(timeout=0)` is polled with a 2 ms yield when no encoded frame is
  ready; the vision main loop is never involved.
- Every successful `GetStream` is paired with `ReleaseStream` in `finally`,
  including send failures.
- If `GetStream(timeout=0)` produces no encoded pack for three seconds, the
  worker records a VENC stall, restarts RTSP/VENC and continues independently
  of visual tracking and UART.
- Counters saturate instead of wrapping into negative or growing containers.
- UDP discovery failures are warnings and do not restart video.
- The UDP discovery socket is non-blocking and broadcasts only every ten
  seconds. It cannot wait inside the worker that drains VENC output. The first
  beacon is still sent immediately when streaming starts.
- The service holds no software frame queue. Hardware VENC buffers are the
  only bounded backlog, so Python cannot accumulate stale frames or memory.
- Initialization/runtime errors remain visible through `NET ... err=...` on
  the LCD and the `NET[...]` PCDBG field. They do not become visual-control
  frame errors and do not stop UART publication.
- If a lower-level send never returns, shutdown times out and intentionally
  does not destroy Sensor/MediaManager underneath the live worker. This may
  require an IDE/device reset, but avoids a use-after-free crash.

The reliability default uses the supplied working reference's verified
`uctypes.bytearray_at` -> independent `bytes` ->
`rtspserver_sendvideodata` path. The copy remains alive until the RTSP send
call returns, so `ReleaseStream` cannot immediately recycle the VENC memory
being submitted. This costs one bounded encoded-pack allocation/copy and
removes a plausible corruption path. Subsequent TCP tests still showed decoder
errors, so physical-buffer lifetime is not treated as the only cause.

`rtspserver_sendvideodata_byphyaddr` remains available through
`WIFI_VIDEO_PREFER_ZERO_COPY=True` for controlled performance experiments. It
is not the default because the installed API does not document whether RTSP
finishes consuming the physical buffer before the call returns. Do not enable
it for control operation without a multi-hour corruption and reconnect test.

The fourth RTSP send argument remains `1000`, exactly as in both existing K230
RTSP implementations. The available repository and reference material do not
define whether it is a timeout or timestamp. Changing it to a fabricated PTS
would be unsafe without the exact installed-firmware API documentation.

## Wi-Fi timeout diagnosis

After an association attempt reaches 30 seconds, the background worker performs
one `WLAN.scan()` and releases the returned list immediately. Scanning is never
performed in the visual frame loop or while streaming. The non-streaming LCD
line includes `why`, the scan result count `n`,
`ap=<seen>/<rssi>/<security>`, `link`, `ip` and `t`:

- `why=NO_AP_VISIBLE n=0`: the K230 scan returned no networks at all. Check the
  Wi-Fi module/antenna, firmware support and whether any nearby compatible AP
  is broadcasting.
- `why=AP_NOT_FOUND n>0`: the radio can scan other networks, but the configured
  SSID was absent. Check spelling, hidden-SSID configuration, hotspot range and
  especially whether only an unsupported band/channel is enabled.
- `why=WEAK_SIGNAL`: the target was visible at `-80 dBm` or weaker. Move the
  devices closer or improve the antenna/orientation before judging passwords.
- `why=OPEN_AP_PASSWORD` or `PASSWORD_REQUIRED`: the configured password and
  the scanned open/secured property disagree.
- `why=AUTH_OR_MODE`: the target was visible with usable RSSI, but association
  did not complete. The available K230 binding exposes no verified WPA failure
  code, so this deliberately groups a wrong password, WPA3-only/incompatible
  security mode and AP-side rejection. Recheck credentials and start with a
  2.4 GHz WPA2 compatibility mode.
- `why=DHCP_TIMEOUT`: association succeeded but DHCP did not assign an address.
  Check the router DHCP pool and client limits.
- `why=IP_API_ERROR` or `SCAN_ERROR`: the corresponding K230 API raised an
  exception; its type and truncated message are appended on screen and the
  complete message is printed to the IDE console.
- `link=?`: this firmware did not return a usable `isconnected()` state. A
  nonzero `ip` is still accepted as connected because it is the more useful
  network-ready condition and matches the supplied working reference.
- `NET RUN <ip>`: RTSP/VENC is running; open the URL shown in this document.

For a running stream the diagnostic format is:

```text
NET RUN <ip> f=<fps> kb=<kbps> z=<Y/N> e=<errors> r=<restarts> s=<stalls> tx=<last/max-ms> p=<max-packs>
```

`f` and `kb` are the measured encoded-send rate, `z=N` confirms the reliable
copied-pack path, `e` counts service errors, `r` counts stream restarts, `s`
counts three-second VENC stalls, `tx` exposes current/maximum RTSP send-call
latency and `p` is the maximum number of packs returned for one encoded frame.
These fields distinguish a slow/blocked RTSP queue from a decoder error that
occurs despite prompt, continuous sends. The SSID and password are never
included in LCD or PC debug output.

## Performance choices and verification

Hardware Sensor-to-VENC binding avoids acquiring or converting video frames in
Python. Python still retrieves encoded packs and invokes the RTSP server, so
`_thread` alone is not proof of zero CPU impact or strict non-blocking behavior.
The copied default adds allocation and memory bandwidth proportional to the
encoded bitrate (nominally about 75 kB/s at 600 kb/s), not to the raw image
size. It is expected to be modest relative to Python-side raw-frame copying,
but its real detector/GC impact still requires target measurement.

## PC RTSP/TCP receiver

Install OpenCV and run the receiver from this project with desktop CPython:

```text
python -m pip install opencv-python
python pc_rtsp_receiver.py
```

The viewer requires OpenCV HighGUI. If `namedWindow` is reported as not
implemented, remove `opencv-python-headless` and
`opencv-contrib-python-headless` from that same Python environment, then
force-reinstall `opencv-python`. The script catches this condition, reports the
installed OpenCV wheel names/versions and prints the exact repair commands
instead of terminating with a traceback.

The receiver uses the UDP beacon only to discover `<IP>:8554`, then forces the
RTSP media transport to TCP. This prevents ordinary Wi-Fi RTP/UDP loss from
turning into invalid H.264 reference blocks. When supported by the installed
OpenCV build, open and read waits are bounded to five seconds; failed streams
are released and reopened, and repeated open failures trigger rediscovery in
case DHCP changed the K230 address. VLC can also be used, but select RTSP over
TCP in its input/network options when diagnosing corruption.

Measure the same scene with `WIFI_VIDEO_ENABLED=False` and then enabled:

1. Warm up for at least 30 seconds.
2. Compare the displayed detector `FPS`, `dt`, `cap`, `det` and loop time for at
   least 10 minutes. Target average and tail slowdown below 10%.
3. Confirm `NET RUN`, increasing sent frame/byte counters, and stable memory.
4. Open `rtsp://<board-ip>:8554/k230` in VLC/FFmpeg/OpenCV and verify latency
   does not continuously grow.
5. Disconnect/reconnect the AP and confirm visual tracking/UART continue while
   `NET RETRY` is shown, then video returns without reboot.
6. Stop from the IDE repeatedly and confirm no shutdown timeout or media-buffer
   error.
7. Run a 2--4 hour weak-network test before relying on the feature in control
   operation.

The current `320x240 / 15 FPS / 600 kb/s` profile is the reliability baseline.
Only after a long stable run should a quality profile such as
`512x288 / 30 FPS / 1200--1600 kb/s` be tried. If a quality profile slows the
detector by more than 10% or increases `tx`/`s`, return to the baseline rather
than lowering the detector channel resolution or frame rate.

## API evidence and remaining hardware boundary

The implementation only uses calls present in this repository or the supplied
working reference: `network.WLAN`, `_thread.start_new_thread`, Sensor CH1
configuration and `bind_info`, `MediaManager.link`, `Encoder.SetOutBufs`,
`ChnAttrStr`, `Create/Start/GetStream/ReleaseStream/Stop/Destroy`,
`multimedia.rtsp_server`, both RTSP send variants, and UDP broadcast sockets.

Host checks can validate syntax, state logic and main-program call boundaries.
They cannot prove Wi-Fi firmware availability, the exact VENC rate unit,
320x240 sensor mode, physical-address pack layout, RTSP transport behavior,
thread scheduling/core placement, or performance. Those claims require the
installed K230 firmware and actual AP/receiver test.
