"""Overhead camera -> where each tool is and where to grasp it.

Calibration: 4 AprilTags (tag36h11) at measured base-frame positions -> pixel->table homography.
Detection: tools on a dark mat. A colourful grip with a grey, unsaturated part sticking out of one end = a tool; the
grey part is the shaft. Naming is separate: by handle colour (config.TOOLS) or by Gemma (namer.py). The grasp point is on the SHAFT, GRASP_FROM_FERRULE_M from the handle, so the robot can
present the handle to the operator.
"""
import cv2
import numpy as np
from edgehand import config

_aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                 cv2.aruco.DetectorParameters())


def detect_tags(frame):
    """{id: (corners 4x2)}"""
    corners, ids, _ = _aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


class TableMap:
    def __init__(self):
        self.H = None
        self.tags = {}

    def update(self, frame):
        self.tags = detect_tags(frame)
        ids = [i for i in config.TABLE_TAGS if i in self.tags]
        if len(ids) == len(config.TABLE_TAGS) >= 4:
            src = np.array([self.tags[i].mean(axis=0) for i in ids], np.float32)
            dst = np.array([config.TABLE_TAGS[i] for i in ids], np.float32)
            self.H, _ = cv2.findHomography(src, dst)
        else:
            self.H = None
        return self.H is not None

    def px_to_table(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return np.array([p[0] / p[2], p[1] / p[2]])

    def table_to_px(self, x, y):
        p = np.linalg.inv(self.H) @ np.array([x, y, 1.0])
        return np.array([p[0] / p[2], p[1] / p[2]])

    def tag_mask(self, shape):
        mask = np.zeros(shape[:2], np.uint8)
        for c in self.tags.values():
            cv2.fillConvexPoly(mask, c.astype(np.int32), 255)
        return cv2.dilate(mask, np.ones((21, 21), np.uint8))


def _mask(hsv, ranges):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, lo, hi)
    return m


def find_tools(frame, tmap):
    """Every tool on the mat, unnamed: a colourful grip with a metal end. Biggest first. Each is
    {"grasp_xy", "handle_dir", "grasp_px", "handle_px", "area", "contour", "colour"}; handle_dir is a unit vector
    (table frame) from the grasp point towards the grip, "colour" is the TOOLS class whose HSV range covers the grip
    (or None). Names come from colour_class or namer.GemmaNamer."""
    if tmap.H is None:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ignore = tmap.tag_mask(frame.shape)
    k = np.ones((5, 5), np.uint8)
    shaft = cv2.inRange(hsv, (0, 0, config.SHAFT_MIN_VAL), (180, config.SHAFT_MAX_SAT, 255))
    shaft[ignore > 0] = 0
    shaft = cv2.morphologyEx(shaft, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = []
    for c in _grip_contours(hsv, ignore):
        area = cv2.contourArea(c)
        pts = c.reshape(-1, 2).astype(float)
        centre = pts.mean(axis=0)
        axis = np.linalg.svd(pts - centre)[2][0]                 # grip's long axis (px)
        proj = (pts - centre) @ axis
        length = proj.max() - proj.min()
        ends = [centre + axis * proj.min(), centre + axis * proj.max()]
        # the metal part sticks out of one end: count shaft pixels just beyond each end
        scores = []
        for e, sgn in zip(ends, (-1, 1)):
            probe = np.zeros_like(shaft)
            a, b = e + axis * sgn * 0.05 * length, e + axis * sgn * 0.6 * length
            cv2.line(probe, tuple(map(int, a)), tuple(map(int, b)), 255, max(3, int(0.15 * length)))
            scores.append(int(np.count_nonzero(probe & shaft)))
        if max(scores) < 0.05 * length * 3 or max(scores) < 2 * min(scores):
            continue                                              # no metal end, or can't tell which end has it
        ferrule_px = ends[int(np.argmax(scores))]
        ferrule, handle = tmap.px_to_table(*ferrule_px), tmap.px_to_table(*centre)
        d = handle - ferrule
        d /= np.linalg.norm(d)
        grasp = ferrule - d * config.GRASP_FROM_FERRULE_M
        out.append({"grasp_xy": grasp, "handle_dir": d, "grasp_px": tmap.table_to_px(*grasp), "handle_px": centre,
                    "area": area, "contour": c, "colour": colour_class(hsv, c)})
    return sorted(out, key=lambda t: -t["area"])


def _grip_contours(hsv, ignore):
    """Colourful grips, split by hue so two touching grips of different colours stay two tools. Hue bands of +-16 degrees
    every 16 degrees; a grip found in two neighbouring bands is kept once (the bigger outline)."""
    k = np.ones((5, 5), np.uint8)
    grips = cv2.inRange(hsv, (0, config.GRIP_MIN_SAT, config.GRIP_MIN_VAL), (180, 255, 255))
    grips[ignore > 0] = 0
    hue = hsv[:, :, 0].astype(np.int16)
    found = []
    for centre in range(0, 180, 8):                                  # OpenCV hue: 0-180, i.e. 2 degrees per unit
        dist = np.abs((hue - centre + 90) % 180 - 90)
        m = np.where(dist <= 8, grips, 0).astype(np.uint8)
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            if cv2.contourArea(c) >= config.TOOL_MIN_AREA_PX:
                found.append(c)
    found.sort(key=lambda c: -cv2.contourArea(c))
    kept = []
    for c in found:
        p = c.reshape(-1, 2).mean(axis=0)
        if all(cv2.pointPolygonTest(o, (float(p[0]), float(p[1])), False) < 0 for o in kept):
            kept.append(c)
    return kept


def colour_class(hsv, contour):
    """The TOOLS class whose HSV ranges cover most of the grip (at least half of it), else None."""
    m = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    n = max(1, np.count_nonzero(m))
    best, frac = None, 0.5
    for name, ranges in config.TOOLS.items():
        f = np.count_nonzero(_mask(hsv, ranges) & m) / n
        if f > frac:
            best, frac = name, f
    return best


def detect_tools(frame, tmap):
    """find_tools named by handle colour (config.TOOLS); tools whose colour matches no class are dropped."""
    return [dict(t, tool=t["colour"]) for t in find_tools(frame, tmap) if t["colour"]]


def draw(frame, tmap, tools):
    out = frame.copy()
    for c in tmap.tags.values():
        cv2.polylines(out, [c.astype(np.int32)], True, (0, 255, 0), 2)
    for t in tools:
        g, h = tuple(map(int, t["grasp_px"])), tuple(map(int, t["handle_px"]))
        cv2.arrowedLine(out, g, h, (255, 255, 0), 2, tipLength=0.2)
        cv2.circle(out, g, 8, (0, 0, 255), -1)
        cv2.putText(out, f"{t.get('tool') or '?'} {t['grasp_xy'][0]:.3f},{t['grasp_xy'][1]:.3f}", (g[0] + 10, g[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, "CALIBRATED" if tmap.H is not None else "NOT CALIBRATED: need tags " + str(list(config.TABLE_TAGS)),
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if tmap.H is not None else (0, 0, 255), 2)
    return out
