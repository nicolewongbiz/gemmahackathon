"""The surgical video node (runs on the Jetson): a recorded clip stands in for the operating microscope's camera.

It plays the clip in real time, serves the frames as MJPEG (http://<jetson>:8090/stream.mjpg, /frame.jpg, /status),
and tells the laptop's surgical assistant the current phase over edgehand/link.py on PHASE_UDP_PORT:
    {"type": "phase", "phase", "prev", "clip_t", "annotated": true, "seq"}   on every phase change
    {"type": "phase_status", ...same...}                                    every 2 s, so a late assistant syncs
The phase comes from a time_s,phase CSV. For the demo that's a HAND-ANNOTATED timeline standing in for the phase
model, whose weights aren't available, so every message says annotated=true and the assistant shows it.

    python -m edgehand.surgical_feed clip.mp4 phases.csv                   on the Jetson (laptop = 192.168.55.100)
    python -m edgehand.surgical_feed clip.mp4 phases.csv --laptop 127.0.0.1 --loop     everything on one machine
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from edgehand import config, link
from edgehand.phase_player import load_segments, phase_at


class Feed:
    """Plays the clip on a thread; calls on_frame(jpeg_bytes) and on_phase(msg)."""
    def __init__(self, video, phases, on_frame=lambda b: None, on_phase=lambda m: None, speed=1.0, loop=False,
                 width=960, quality=70):
        self.video, self.speed, self.loop, self.width, self.quality = video, speed, loop, width, quality
        self.on_frame, self.on_phase = on_frame, on_phase
        cap = cv2.VideoCapture(video)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        self.duration = n / self.fps if n > 0 else None
        self.segs = load_segments(phases, self.duration)
        self.duration = self.duration or self.segs[-1][1]
        self.state = {"phase": None, "prev": None, "clip_t": 0.0, "seq": 0, "annotated": True}
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._status, daemon=True).start()

    def _loop(self):
        while self._run:
            cap = cv2.VideoCapture(self.video)
            t0 = time.time()
            while self._run:
                t = (time.time() - t0) * self.speed
                if t >= self.duration:
                    break
                if abs(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - t) > 0.5:     # fell behind: jump to now
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    break
                ph = phase_at(self.segs, t)
                if ph != self.state["phase"]:
                    self.state.update(prev=self.state["phase"], phase=ph, clip_t=round(t, 1),
                                      seq=self.state["seq"] + 1)
                    self.on_phase(dict(self.state, type="phase"))
                self.state["clip_t"] = round(t, 1)
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (self.width, int(h * self.width / w)))
                self.on_frame(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes())
                time.sleep(max(0.0, 1 / self.fps / self.speed - 0.002))
            cap.release()
            if not self.loop:
                self._run = False
            else:
                self.state.update(prev=self.state["phase"], phase=None)

    def _status(self):
        while self._run:
            time.sleep(2.0)
            if self.state["phase"] is not None:
                self.on_phase(dict(self.state, type="phase_status"))

    def close(self):
        self._run = False


def serve(port, latest, status):
    """MJPEG of the newest frame, /frame.jpg, /status (JSON)."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while True:
                        b = latest[0]
                        if b is not None and b is not last:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                            last = b
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif self.path.startswith("/frame.jpg") and latest[0] is not None:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(latest[0])
            elif self.path.startswith("/status"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(status()).encode())
            else:
                self.send_response(404)
                self.end_headers()
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("phases")
    ap.add_argument("--laptop", default="192.168.55.100", help="where the surgical assistant runs")
    ap.add_argument("--port", type=int, default=config.PHASE_UDP_PORT)
    ap.add_argument("--http", type=int, default=8090)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    latest = [None]

    def on_phase(msg):
        if msg["type"] == "phase":
            print(f"{msg['clip_t']:6.1f} s  phase: {msg['phase']}  (was {msg['prev']})", flush=True)
        try:
            link.send(dict(msg, source="jetson"), a.laptop, a.port)
        except OSError:
            pass                                        # laptop not plugged in yet

    feed = Feed(a.video, a.phases, lambda b: latest.__setitem__(0, b), on_phase, a.speed, a.loop)
    serve(a.http, latest, lambda: feed.state)
    beat = link.Heartbeat("jetson", a.laptop, a.port)
    print(f"{a.video}: {feed.duration:.1f} s, {len(feed.segs)} phases; MJPEG on :{a.http}/stream.mjpg; "
          f"phase events -> {a.laptop}:{a.port}", flush=True)
    try:
        while feed._run:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    beat.close()


if __name__ == "__main__":
    main()
