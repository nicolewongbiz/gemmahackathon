"""Serve a USB camera as an MJPEG stream over HTTP, for a computer that can't open it itself.

Runs on the Jetson with the RealSense on its USB 3 port (needs only python3 + opencv + numpy, which JetPack has):
    python3 cam_relay.py                       picks the first V4L2 device that gives a COLOUR 1280x720 frame
    python3 cam_relay.py --device 4            a specific /dev/videoN
    python3 cam_relay.py --port 8089 --list    show every device with what it delivers, then exit
Then on the laptop: config.CAMERA_SOURCE = "url", CAMERA_URL = "http://192.168.55.1:8089/stream.mjpg".
Also GET /frame.jpg for one frame and / for a plain status line. Stdlib HTTP server, one thread per client.

Why: on macOS the D435 shows up as either its infrared or its colour camera, by a race at plug-in; Linux/V4L2 lists
the colour camera as its own device every time.
"""
import argparse, glob, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np


def is_infrared(frame):
    f = frame[::8, ::8].astype(int)
    return bool(np.abs(f[:, :, 0] - f[:, :, 2]).mean() < 1.0)


def open_device(index, w, h, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))      # the D435 colour stream's native format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def probe(index, w, h, fps):
    """(frame, cap) for a device that delivers colour at the asked size within 3 s, else (None, None)."""
    cap = open_device(index, w, h, fps)
    if cap is None:
        return None, None
    t0 = time.time()
    while time.time() - t0 < 3.0:
        ok, f = cap.read()
        if ok and f is not None and f.shape[1] == w and not is_infrared(f):
            return f, cap
    cap.release()
    return None, None


def devices():
    return sorted(int(p[len("/dev/video"):]) for p in glob.glob("/dev/video*"))


class Relay:
    def __init__(self, cap, quality):
        self.cap, self.quality = cap, quality
        self.jpeg, self.n, self.t = None, 0, time.time()
        self.lock = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, f = self.cap.read()
            if not ok or f is None:
                time.sleep(0.02)
                continue
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                with self.lock:
                    self.jpeg, self.n, self.t = buf.tobytes(), self.n + 1, time.time()
                    self.lock.notify_all()

    def wait_frame(self, after):
        with self.lock:
            self.lock.wait_for(lambda: self.n > after, timeout=2.0)
            return self.jpeg, self.n


def make_handler(relay, info):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                n = 0
                try:
                    while True:
                        jpeg, n = relay.wait_frame(n)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif self.path.startswith("/frame"):
                jpeg, _ = relay.wait_frame(relay.n)
                self.send_response(200 if jpeg else 503)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                if jpeg:
                    self.wfile.write(jpeg)
            else:
                body = f"cam_relay {info}: {relay.n} frames, last {time.time() - relay.t:.1f} s ago\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, help="/dev/videoN; default: the first that gives colour at the size")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i in devices():
            f, cap = probe(i, a.width, a.height, a.fps)
            print(f"/dev/video{i}: " + ("colour %dx%d" % (f.shape[1], f.shape[0]) if f is not None else "no colour frame"))
            if cap:
                cap.release()
        return
    picks = [a.device] if a.device is not None else devices()
    frame = cap = None
    for i in picks:
        frame, cap = probe(i, a.width, a.height, a.fps)
        if frame is not None:
            print(f"/dev/video{i}: colour {frame.shape[1]}x{frame.shape[0]}", flush=True)
            break
        print(f"/dev/video{i}: no colour frame", flush=True)
    if cap is None:
        sys.exit("no device gives a colour frame; is the camera on a USB 3 port? (--list)")
    relay = Relay(cap, a.quality)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(relay, f"/dev/video{i}"))
    print(f"serving http://0.0.0.0:{a.port}/stream.mjpg  (/frame.jpg, /)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
