"""Find objects by differencing against an empty-table snapshot, then draw numbered marks."""
import cv2
import numpy as np

MIN_AREA = 600   # px^2, tune for camera height


class ObjectFinder:
    def __init__(self):
        self.empty = None

    def snapshot_empty(self, frame):
        self.empty = cv2.GaussianBlur(frame, (7, 7), 0)

    def find(self, frame, ignore_mask=None):
        """Returns {id: (u, v)} pixel centroids, ids start at 1."""
        if self.empty is None:
            raise RuntimeError("Call snapshot_empty() with a clear table first")
        diff = cv2.absdiff(cv2.GaussianBlur(frame, (7, 7), 0), self.empty)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        if ignore_mask is not None:
            m[ignore_mask > 0] = 0
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objs = []
        for c in cnts:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            M = cv2.moments(c)
            objs.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        objs.sort(key=lambda p: (p[1] // 80, p[0]))   # reading order
        return {i + 1: p for i, p in enumerate(objs)}


def draw_marks(frame, objs, bins_px=None):
    out = frame.copy()
    for i, (u, v) in objs.items():
        c = (int(u), int(v))
        cv2.circle(out, c, 16, (255, 255, 255), -1)
        cv2.circle(out, c, 16, (0, 0, 0), 2)
        cv2.putText(out, str(i), (c[0] - 9 if i < 10 else c[0] - 14, c[1] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for name, (u, v) in (bins_px or {}).items():
        cv2.putText(out, f"BIN {name}", (int(u) - 30, int(v) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return out


class KeyframeRecorder:
    """During teaching, keep a frame each time the scene settles after a change."""
    def __init__(self, still_frames=10, change_thresh=4.0):
        self.prev, self.still, self.changed = None, 0, False
        self.frames = []
        self.still_frames, self.change_thresh = still_frames, change_thresh

    def push(self, frame):
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.prev is not None:
            delta = float(np.mean(np.abs(small - self.prev)))
            if delta > self.change_thresh:
                self.changed, self.still = True, 0
            else:
                self.still += 1
                if self.changed and self.still >= self.still_frames:
                    self.frames.append(frame.copy())
                    self.changed = False
        self.prev = small

    def result(self, max_frames=6):
        f = self.frames
        if len(f) > max_frames:
            idx = np.linspace(0, len(f) - 1, max_frames).astype(int)
            f = [f[i] for i in idx]
        return f
