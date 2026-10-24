"""Camera backends, picked by config.CAMERA: USB webcam, Kinova Gen3 Vision module (RTSP), or fake.

The colour stream is H.264 RTSP at rtsp://<ROBOT_IP>/color (default 1280x720 @ 30 fps).
Resolution / fps are set on the arm (Web App -> Camera, or the Kortex VisionConfig API), not here.
Callers move the arm with arm.look() before a perception frame: to the viewing pose when the
camera is on the wrist, or out of the picture when it is on a stand.
"""
import threading, time
import cv2
import numpy as np
import config


class KinovaVisionCamera:
    """Background-thread RTSP reader so the newest frame is always ready."""
    def __init__(self, url=None, timeout=config.VISION_CONNECT_TIMEOUT):
        self.url = url or f"rtsp://{config.ROBOT_IP}/color"
        self.timeout_ms = int(timeout * 1000)
        self.frame = None
        self._run = True
        self.cap = self._open()
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while self.frame is None:
            if time.time() - t0 > timeout:
                self.close()
                raise RuntimeError(f"No frames from Kinova Vision at {self.url}. "
                                   "Check the arm is on, the laptop is on 192.168.1.x, and the stream "
                                   "plays in VLC. Set CAMERA = \"mock\" to run without it.")
            time.sleep(0.05)

    def _open(self):
        # Without explicit timeouts FFmpeg blocks ~30 s on a dead stream.
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
            else:
                # Stream dropped (arm reboot, cable bump): reconnect rather than die mid-demo.
                self.cap.release()
                time.sleep(1.0)
                if self._run:
                    self.cap = self._open()

    def read(self):
        return self.frame.copy()

    def close(self):
        self._run = False
        self.cap.release()


class Webcam:
    """Background-thread USB/laptop camera so the latest frame is always ready."""
    def __init__(self, index=config.WEBCAM_INDEX, timeout=config.VISION_CONNECT_TIMEOUT):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # DirectShow opens fast on Windows
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        self.frame = None
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while self.frame is None:
            if time.time() - t0 > timeout:
                self.close()
                raise RuntimeError(f"No frames from webcam {index}. Close other apps using it "
                                   "(Teams, Zoom, Camera) or try WEBCAM_INDEX = 1.")
            time.sleep(0.05)

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
            else:
                time.sleep(0.05)

    def read(self):
        return self.frame.copy()

    def close(self):
        self._run = False
        self.cap.release()


class FakeCamera:
    def read(self):
        return np.full((config.FRAME_H, config.FRAME_W, 3), 127, np.uint8)

    def close(self):
        pass


def make_camera():
    return {"webcam": Webcam, "kinova": KinovaVisionCamera, "mock": FakeCamera}[config.CAMERA]()
