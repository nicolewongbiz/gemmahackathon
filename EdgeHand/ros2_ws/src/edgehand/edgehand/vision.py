"""Overhead camera -> where each tool is and where to grasp it.

Calibration: 4 AprilTags (tag36h11) at measured base-frame positions -> pixel->table homography.
Detection: tools on a dark mat. Handle colour = class (config.TOOLS); the grey, unsaturated part touching one end of
the handle is the shaft. The grasp point is on the SHAFT, GRASP_FROM_FERRULE_M from the handle, so the robot can
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


def detect_tools(frame, tmap):
    """List of {"tool", "grasp_xy", "handle_dir", "grasp_px", "handle_px", "area"}, biggest first.
    handle_dir is a unit vector (table frame) pointing from the grasp point towards the handle."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ignore = tmap.tag_mask(frame.shape)
    k = np.ones((5, 5), np.uint8)
    shaft = cv2.inRange(hsv, (0, 0, config.MAT_MAX_VAL), (180, config.SHAFT_MAX_SAT, 255))
    shaft[ignore > 0] = 0
    shaft = cv2.morphologyEx(shaft, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = []
    for name, ranges in config.TOOLS.items():
        m = _mask(hsv, ranges)
        m[ignore > 0] = 0
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            area = cv2.contourArea(c)
            if area < config.TOOL_MIN_AREA_PX:
                continue
            pts = c.reshape(-1, 2).astype(float)
            centre = pts.mean(axis=0)
            axis = np.linalg.svd(pts - centre)[2][0]                 # handle's long axis (px)
            proj = (pts - centre) @ axis
            length = proj.max() - proj.min()
            ends = [centre + axis * proj.min(), centre + axis * proj.max()]
            # the shaft sticks out of one end: count shaft pixels just beyond each end
            scores = []
            for e, sgn in zip(ends, (-1, 1)):
                probe = np.zeros_like(shaft)
                a, b = e + axis * sgn * 0.05 * length, e + axis * sgn * 0.6 * length
                cv2.line(probe, tuple(map(int, a)), tuple(map(int, b)), 255, max(3, int(0.15 * length)))
                scores.append(int(np.count_nonzero(probe & shaft)))
            if max(scores) < 0.05 * length * 3:
                continue                                              # no shaft: not a screwdriver lying flat
            ferrule_px = ends[int(np.argmax(scores))]
            ferrule, handle = tmap.px_to_table(*ferrule_px), tmap.px_to_table(*centre)
            d = handle - ferrule
            d /= np.linalg.norm(d)
            grasp = ferrule - d * config.GRASP_FROM_FERRULE_M
            out.append({"tool": name, "grasp_xy": grasp, "handle_dir": d, "grasp_px": tmap.table_to_px(*grasp),
                        "handle_px": centre, "area": area})
    return sorted(out, key=lambda t: -t["area"])


def draw(frame, tmap, tools):
    out = frame.copy()
    for c in tmap.tags.values():
        cv2.polylines(out, [c.astype(np.int32)], True, (0, 255, 0), 2)
    for t in tools:
        g, h = tuple(map(int, t["grasp_px"])), tuple(map(int, t["handle_px"]))
        cv2.arrowedLine(out, g, h, (255, 255, 0), 2, tipLength=0.2)
        cv2.circle(out, g, 8, (0, 0, 255), -1)
        cv2.putText(out, f"{t['tool']} {t['grasp_xy'][0]:.3f},{t['grasp_xy'][1]:.3f}", (g[0] + 10, g[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, "CALIBRATED" if tmap.H is not None else "NOT CALIBRATED: need tags " + str(list(config.TABLE_TAGS)),
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if tmap.H is not None else (0, 0, 255), 2)
    return out
