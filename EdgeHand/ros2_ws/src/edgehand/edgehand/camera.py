"""The camera on the wrist (eye-in-hand). config.CAMERA_SOURCE picks it:
    "usb"  - a camera cabled to this computer, e.g. our Intel RealSense (its colour camera shows up as a UVC webcam)
    "rtsp" - the Kinova Vision module, rtsp://<ROBOT_IP>/color
    "url"  - any stream OpenCV/FFmpeg can open: cam_relay.py on the Jetson serving the RealSense as MJPEG over HTTP
A background thread keeps reading so the newest frame is always ready, and reconnects if the stream drops.

    python -m edgehand.camera --list      which OpenCV index is which camera (saves a snapshot of each)
    python -m edgehand.camera [index|url] live view + whether the 4 table tags are visible (no robot motion)
"""
import sys, threading, time
import cv2
from edgehand import config


def open_capture(source):
    """source: int (USB index) or str (RTSP url)."""
    if isinstance(source, int):
        api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else (cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2)
        cap = cv2.VideoCapture(source, api)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    else:
        # without explicit timeouts FFmpeg blocks ~30 s on a dead stream
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                                                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def usb_cameras():
    """[(OpenCV index, name)] on macOS; [] if it can't tell. OpenCV's AVFoundation backend numbers cameras sorted by
    uniqueID (checked on the cell: RealSense 0x1400..., MacBook 6C70...), not in the order macOS lists them."""
    if sys.platform != "darwin":
        return []
    try:
        import AVFoundation as AV
    except ImportError:
        return []
    types = [getattr(AV, t) for t in ("AVCaptureDeviceTypeBuiltInWideAngleCamera", "AVCaptureDeviceTypeExternal",
                                      "AVCaptureDeviceTypeExternalUnknown", "AVCaptureDeviceTypeContinuityCamera",
                                      "AVCaptureDeviceTypeDeskViewCamera") if hasattr(AV, t)]
    devs = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types, AV.AVMediaTypeVideo, 0).devices()
    return [(i, str(d.localizedName())) for i, d in enumerate(sorted(devs, key=lambda d: str(d.uniqueID())))]


def configured_source():
    if config.CAMERA_SOURCE == "url":
        return config.CAMERA_URL
    if config.CAMERA_SOURCE != "usb":
        return config.CAMERA_RTSP
    cams = usb_cameras()                             # by name: indexes shift when an iPhone etc. connects
    hits = [(i, n) for i, n in cams if config.CAMERA_USB_NAME.lower() in n.lower()]
    # a RealSense D4xx is two UVC devices on macOS: "... Depth" (infrared, greyscale with the projector's dots) and
    # "... RGB". We want the colour one; a depth-only match means the RGB function didn't enumerate (re-plug).
    colour = [(i, n) for i, n in hits if not n.strip().lower().endswith("depth")]
    if colour:
        return colour[0][0]
    if hits:
        # macOS often lists just the "Depth" device. What it streams depends on the USB path: on the dock's USB 2
        # port it gave colour; on a USB 3 port it gives IR (is_infrared below tells them apart on the first frame).
        return hits[0][0]
    if cams:                                         # we can see the cameras and the right one isn't there: say so
        raise RuntimeError(f"no camera named like {config.CAMERA_USB_NAME!r}; connected: "
                           + ", ".join(f"{i}: {n}" for i, n in cams) + ". Plug the RealSense into this computer.")
    return config.CAMERA_USB_INDEX                   # can't list names (no AVFoundation binding): the configured index


def is_infrared(frame):
    """True for the D435's IR stream: a greyscale image (all channels equal), unlike any colour scene."""
    import numpy as np
    f = frame[::8, ::8].astype(int)
    return bool(np.abs(f[:, :, 0] - f[:, :, 2]).mean() < 1.0)


IR_HINT = ("the RealSense is streaming INFRARED (greyscale with the projector's dot pattern), not colour. On this Mac "
           "the colour stream came through the dock's USB 2 port; a direct USB 3 port gives IR. Move the camera.")


class StreamCamera:
    def __init__(self, source=None, on_frame=lambda f: None):
        self.source = configured_source() if source is None else source
        self.on_frame = on_frame
        self.frame = None
        self._run = True
        self.cap = open_capture(self.source)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
                self.on_frame(f)
            else:                                    # cable bump, arm reboot: reconnect instead of dying
                self.cap.release()
                time.sleep(1.0)
                if self._run:
                    self.cap = open_capture(self.source)

    def wait_first(self, timeout_s=10.0):
        t0 = time.time()
        while self.frame is None and time.time() - t0 < timeout_s:
            time.sleep(0.05)
        return self.frame is not None

    def close(self):
        self._run = False
        self.cap.release()


if __name__ == "__main__":
    if "--list" in sys.argv:
        names = dict(usb_cameras())
        for i in range(5):
            cap = open_capture(i)
            f, t0 = None, time.time()
            while cap.isOpened() and time.time() - t0 < 3:          # cameras take a moment to start
                ok, fr = cap.read()
                if ok:
                    f = fr
            if f is not None:
                path = f"/tmp/edgehand_camera_{i}.jpg"
                cv2.imwrite(path, f)
                print(f"index {i}: {names.get(i, '?')}  {f.shape[1]}x{f.shape[0]}  snapshot {path}")
            cap.release()
        print("configured source ->", configured_source())
        sys.exit(0)
    from edgehand.vision import TableMap, draw, find_tools
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    cam = StreamCamera(None if arg is None else (int(arg) if arg.isdigit() else arg))
    if not cam.wait_first():
        sys.exit(f"no frames from {cam.source}")
    print(f"streaming {cam.frame.shape[1]}x{cam.frame.shape[0]} from {cam.source}; Esc to quit")
    while True:
        f = cam.frame.copy()
        tm = TableMap()
        tools = find_tools(f, tm) if tm.update(f) else []
        cv2.imshow("wrist camera", draw(f, tm, [dict(t, tool=t["colour"]) for t in tools]))
        if cv2.waitKey(30) == 27:
            break
    cam.close()
