"""Eyes: wrist-camera frames -> a scene the planner can use (table x,y per numbered object, bins).

Runs on the Jetson (jetson/eyes_server.py serves LocalEyes over HTTP) or on the laptop when
EYES = "local". main.py talks to either through the same methods:
    snapshot_empty(), scene(), teach_start(), teach_stop(), status()
Frames are only valid at LOOK_POSE, so the caller moves the arm with arm.look() first.
"""
import base64, threading, time
import cv2
import numpy as np
import requests
import config
from perception.camera import make_camera
from perception.calibrate import TableMap
from perception.objects import ObjectFinder, KeyframeRecorder, draw_marks
from planner import gemma_vision

LABEL = """Objects are marked with numbered white circles: {ids}.
Name each numbered object in 1-3 words (what it is, plus its colour).
Return JSON: {{"1": "red apple", "2": "blue sponge"}}"""


def encode(img):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()


def decode(s):
    return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)


class LocalEyes:
    def __init__(self):
        self.cam, self.tmap, self.finder = make_camera(), TableMap(), ObjectFinder()
        self.recorder, self.teaching = None, False

    def snapshot_empty(self):
        self.finder.snapshot_empty(self.cam.read())

    def scene(self, label=False):
        """{"calibrated", "objects": {id: (x, y)}, "bins": {"A": (x, y), "B": ...}, "labels": {id: str}, "marked": img}"""
        f = self.cam.read()
        if not self.tmap.update(f):
            return {"calibrated": False, "objects": {}, "bins": {}, "labels": {}, "marked": f}
        found = [p for p in self.finder.find(f, self.tmap.tag_mask(f)).values() if self.tmap.pickable(*p)]
        px = {i + 1: p for i, p in enumerate(found)}
        marked = draw_marks(f, px, self.tmap.bins_px)
        return {"calibrated": True,
                "objects": {i: self.tmap.px_to_table(*p) for i, p in px.items()},
                "bins": {b: self.tmap.bin_xy(b) for b in ("A", "B")},
                "labels": self._label(marked, list(px)) if label and px else {},
                "marked": marked}

    def _label(self, marked, ids):
        """Gemma on the eyes node names each mark. Only a hint for the planner, so never fatal."""
        try:
            out = gemma_vision.ask_json(LABEL.format(ids=ids), [marked],
                                        backend=config.EYES_GEMMA_BACKEND, model=config.EYES_GEMMA_MODEL)
        except Exception as e:
            print("[eyes] labelling failed:", e)
            return {}
        return {int(k): str(v) for k, v in out.items() if str(k).isdigit() and int(k) in ids}

    def teach_start(self):
        self.recorder, self.teaching = KeyframeRecorder(), True
        threading.Thread(target=self._teach_loop, daemon=True).start()

    def _teach_loop(self):
        while self.teaching:
            self.recorder.push(self.cam.read())
            time.sleep(1 / 15)

    def teach_stop(self):
        """Keyframes with BIN A / BIN B drawn on them: Gemma can't read tag ids, so it can't tell the bins apart otherwise."""
        self.teaching = False
        time.sleep(0.2)
        out = []
        for f in self.recorder.result() if self.recorder else []:
            self.tmap.update(f)
            out.append(draw_marks(f, {}, self.tmap.bins_px))
        return out

    def status(self):
        return {"online": True, "calibrated": self.tmap.H is not None}


class RemoteEyes:
    """Same interface as LocalEyes, over HTTP to jetson/eyes_server.py."""
    def __init__(self, url=None):
        self.url = url or f"http://{config.JETSON_IP}:{config.JETSON_PORT}"

    def _post(self, path, timeout=10, **body):
        r = requests.post(self.url + path, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def snapshot_empty(self):
        self._post("/empty")

    def scene(self):
        s = self._post("/scene", timeout=90, label=config.EYES_LABELS)
        return {"calibrated": s["calibrated"],
                "objects": {int(k): tuple(v) for k, v in s["objects"].items()},
                "bins": {k: tuple(v) for k, v in s["bins"].items()},
                "labels": {int(k): v for k, v in s["labels"].items()},
                "marked": decode(s["marked"])}

    def teach_start(self):
        self._post("/teach/start")

    def teach_stop(self):
        return [decode(j) for j in self._post("/teach/stop", timeout=30)["frames"]]

    def status(self):
        try:
            return requests.get(self.url + "/status", timeout=2).json()
        except requests.RequestException:
            return {"online": False, "calibrated": False}


def make_eyes():
    return RemoteEyes() if config.EYES == "jetson" else LocalEyes()
