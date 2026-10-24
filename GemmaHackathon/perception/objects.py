"""Find objects by differencing against an empty-table snapshot, then draw numbered marks."""
import cv2
import numpy as np
import config

MIN_AREA = 600   # px^2, tune for camera height


class ObjectFinder:
    def __init__(self):
        self.empty = None

    def snapshot_empty(self, frame):
        self.empty = cv2.GaussianBlur(frame, (7, 7), 0)

    def find(self, frame, ignore_mask=None):
        """Returns {id: (u, v)} pixel centroids, ids start at 1."""
        return {i + 1: c for i, (c, _) in enumerate(self.blobs(frame, ignore_mask))}

    def blobs(self, frame, ignore_mask=None):
        """[((u, v) centroid, contour)] in reading order."""
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
            objs.append(((M["m10"] / M["m00"], M["m01"] / M["m00"]), c))
        objs.sort(key=lambda o: (o[0][1] // 80, o[0][0]))   # reading order
        return objs

    def at(self, frame, u, v, ignore_mask=None, max_px=40):
        """Centroid of the object under (or within max_px of) pixel (u, v), else None."""
        best, best_d = None, max_px
        for c, cnt in self.blobs(frame, ignore_mask):
            d = -cv2.pointPolygonTest(cnt, (float(u), float(v)), True)   # <= 0 inside
            if d <= 0:
                return c
            if d < best_d:
                best, best_d = c, d
        return best


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

    def result(self, max_frames=config.KEYFRAME_MAX):
        f = self.frames
        if len(f) > max_frames:
            idx = np.linspace(0, len(f) - 1, max_frames).astype(int)
            f = [f[i] for i in idx]
        return f
