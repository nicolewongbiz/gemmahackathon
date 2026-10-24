"""Wrist camera at the look pose -> where each tool is and where to grasp it.

Calibration: AprilTags (tag36h11) at measured base-frame positions -> pixel->table homography (TableMap: any 2 of the
4 tags will do, and the last good fit covers frames where they're hidden).
Detection, config.SEGMENTATION:
  "contrast" (default): anything that differs from the mat, seen top-down through the homography, is an object; its
      width profile along its long axis says which end is the handle (wide) and where the shaft starts (narrow).
      Works for black, clear and coloured handles alike; needs a plain mat that contrasts with the tools (white paper
      under dark precision screwdrivers; the black breadboard hid their dark handles).
  "colour": a colourful grip with a grey, unsaturated part sticking out of one end = a tool; the grey part is the
      shaft. The original method, for coloured handles on a dark mat.
Naming is separate: by handle colour (config.TOOLS) or by Gemma (namer.py). The grasp point is on the SHAFT,
GRASP_FROM_FERRULE_M from the handle, so the robot can present the handle to the operator.
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


def _square():
    """A tag's 4 corners in the order ArUco reports them (clockwise from the printed top-left), for a tag lying face
    up, relative to its centre. The tag's yaw on the table is fitted, so tags can be taped down at any angle."""
    return np.array([(-1, 1), (1, 1), (1, -1), (-1, -1)]) * config.TAG_SIZE_M / 2


def _to_3x3(A):
    return np.vstack([A, [0, 0, 1]]) if A.shape == (2, 3) else A


def _apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def _signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class TableMap:
    """Pixel <-> table (base frame x, y) from the AprilTags in view.

    Every visible TABLE_TAGS tag gives 5 points (4 corners + centre), so CALIB_MIN_TAGS tags are enough when a tool or
    the arm hides the others. A tag's corners on the table are a TAG_SIZE_M square around its measured centre, turned
    by the tag's yaw as seen through a rough fit on the centres alone. The camera always looks from the same look pose,
    so with too few tags the last good fit is reused: calibrated_from says "live", "cached" or None."""
    def __init__(self):
        self.H = None
        self.tags = {}
        self.calibrated_from = None
        self.tags_used = 0
        self.bad_tag = None             # a visible tag left out because it disagrees with the others
        self.fit_err = np.inf           # worst tag-corner mismatch of the last fit (m)
        self._last_good = None

    def update(self, frame):
        self.tags = detect_tags(frame)
        ids = [i for i in config.TABLE_TAGS if i in self.tags]
        H, err = self._fit(ids)
        self.bad_tag = None
        if H is None and len(ids) > config.CALIB_MIN_TAGS:
            # one tag that disagrees with the rest (moved, or mis-measured in TABLE_TAGS): the best fit without it, if
            # that one lines up, and say which tag it was
            fits = [(self._fit([i for i in ids if i != bad]), bad) for bad in ids]
            (H, err), bad = min(fits, key=lambda f: f[0][1])
            if H is not None:
                self.bad_tag, ids = bad, [i for i in ids if i != bad]
        self.fit_err = err
        if H is not None:
            self.H, self._last_good, self.calibrated_from, self.tags_used = H, H, "live", len(ids)
        elif self._last_good is not None and len(ids) < config.CALIB_MIN_TAGS:
            # tags hidden (tools, the arm), not moved: same look pose, so the last fit still holds. Tags that are
            # visible but disagree mean something moved, so they never fall back to the cache.
            self.H, self.calibrated_from, self.tags_used = self._last_good, "cached", 0
        else:
            self.H, self.calibrated_from, self.tags_used = None, None, 0
        return self.H is not None

    def why_not(self):
        """Why the last update() didn't calibrate, in words for the window and vision_node's reply."""
        seen = [i for i in config.TABLE_TAGS if i in self.tags]
        if len(seen) < config.CALIB_MIN_TAGS:
            return f"see tags {seen}, need {config.CALIB_MIN_TAGS} of {list(config.TABLE_TAGS)}"
        return (f"tags {seen} don't match TABLE_TAGS (corners off by {self.fit_err * 1000:.0f} mm): "
                f"re-measure them (python -m edgehand.measure_tags)")

    def _rough(self, ids):
        """Centres-only fit: homography (4+ tags), affine (3) or similarity (2). Two points fit a similarity with
        or without a mirror flip; a face-up tag's corners must come out clockwise, which picks one."""
        src = np.float32([self.tags[i].mean(axis=0) for i in ids])
        dst = np.float32([config.TABLE_TAGS[i] for i in ids])
        if len(ids) >= 4:
            return cv2.findHomography(src, dst)[0]
        if len(ids) == 3:
            return _to_3x3(cv2.getAffineTransform(src, dst))
        for flip in (1, -1):
            s = src * [1, flip]
            a = complex(*(dst[1] - dst[0])) / complex(*(s[1] - s[0]))       # rotation + scale as a complex number
            b = complex(*dst[0]) - a * complex(*s[0])
            M = np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag], [0, 0, 1]]) @ np.diag([1, flip, 1])
            if _signed_area(_apply(M, self.tags[ids[0]])) < 0:
                return M
        return None

    def _fit(self, ids):
        """(H, worst corner error in metres), or (None, error) if it doesn't line up / (None, inf) if it can't fit."""
        if len(ids) < config.CALIB_MIN_TAGS:
            return None, np.inf
        rough = self._rough(ids)
        if rough is None:
            return None, np.inf
        sq, src, dst = _square(), [], []
        for i in ids:
            c = np.array(config.TABLE_TAGS[i], float)
            seen = _apply(rough, self.tags[i]) - c                       # this tag's corners on the table, roughly
            yaw = np.arctan2(np.sum(sq[:, 0] * seen[:, 1] - sq[:, 1] * seen[:, 0]), np.sum(sq * seen))
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            src += [*self.tags[i], self.tags[i].mean(axis=0)]
            dst += [*(c + sq @ R.T), c]
        H = cv2.findHomography(np.float32(src), np.float32(dst))[0]
        if H is None:
            return None, np.inf
        # corners that don't line up = a misread tag or a wrong TABLE_TAGS entry: don't trust the fit
        err = float(np.linalg.norm(_apply(H, np.array(src)) - np.array(dst), axis=1).max())
        return (H if err < config.CALIB_MAX_ERR_M else None), err

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
    """Every tool on the mat, unnamed, biggest first (config.SEGMENTATION picks the method). Each is
    {"grasp_xy", "handle_dir", "grasp_px", "handle_px", "area", "contour", "colour"} (+ "width_m", "length_m" from
    the contrast method); handle_dir is a unit vector (table frame) from the grasp point towards the grip, "colour" is
    the TOOLS class whose HSV range covers the grip (or None). Names come from colour_class or namer.GemmaNamer."""
    if config.SEGMENTATION == "colour":
        return find_tools_colour(frame, tmap)
    return find_tools_contrast(frame, tmap)


def find_tools_colour(frame, tmap):
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
        if not on_tray(grasp):
            continue                                              # desk clutter beyond the tags (cables, etc.)
        out.append({"grasp_xy": grasp, "handle_dir": d, "grasp_px": tmap.table_to_px(*grasp), "handle_px": centre,
                    "area": area, "contour": c, "colour": colour_class(hsv, c)})
    return sorted(out, key=lambda t: -t["area"])


def tray_bounds(margin=None):
    """(lo, hi) table corners of the tag rectangle plus a margin (TRAY_MARGIN_M by default)."""
    xs, ys = zip(*config.TABLE_TAGS.values())
    m = config.TRAY_MARGIN_M if margin is None else margin
    return np.array([min(xs) - m, min(ys) - m]), np.array([max(xs) + m, max(ys) + m])


def top_down(frame, tmap):
    """The tray seen from straight above, WARP_MARGIN_M beyond the tags so overhanging handles stay whole: (image, T)
    with T mapping table (x, y) -> warped pixel (u, v) at WARP_PX_PER_M. Row = x (arm forward = up), column = y
    mirrored so the picture isn't flipped. Parts outside the camera's view come out black."""
    lo, hi = tray_bounds(config.WARP_MARGIN_M)
    k = config.WARP_PX_PER_M
    wpx, hpx = int(np.ceil((hi[1] - lo[1]) * k)), int(np.ceil((hi[0] - lo[0]) * k))
    # table -> warped: u = (hi_y - y) k, v = (hi_x - x) k
    T = np.array([[0, -k, hi[1] * k], [-k, 0, hi[0] * k], [0, 0, 1.0]])
    M = T @ tmap.H                                               # pixel -> table -> warped
    return cv2.warpPerspective(frame, M, (wpx, hpx), flags=cv2.INTER_LINEAR), T


def _width_profile(mask_pts, centre, axis, n=40):
    """(stations along the axis from one end to the other, width at each) in the mask's units."""
    rel = mask_pts - centre
    along, across = rel @ axis, rel @ np.array([-axis[1], axis[0]])
    edges = np.linspace(along.min(), along.max(), n + 1)
    idx = np.clip(np.searchsorted(edges, along, side="right") - 1, 0, n - 1)
    width = np.zeros(n)
    for i in range(n):
        a = across[idx == i]
        if len(a):
            width[i] = a.max() - a.min()
    return (edges[:-1] + edges[1:]) / 2, width


def find_tools_contrast(frame, tmap):
    """Objects that differ from the mat, split into handle and shaft by their width profile. See the module doc."""
    if tmap.H is None:
        return []
    img, T = top_down(frame, tmap)
    k = config.WARP_PX_PER_M
    Hinv = np.linalg.inv(tmap.H)
    # the mat: the median colour inside the tag rectangle, with the tags (and their paper) cut out
    tagm = cv2.warpPerspective(tmap.tag_mask(frame.shape), T @ tmap.H, (img.shape[1], img.shape[0]))
    tagm = cv2.dilate(tagm, np.ones((int(0.03 * k) | 1,) * 2, np.uint8))
    lo, hi = tray_bounds(0.0)
    inner = np.zeros(tagm.shape, np.uint8)
    (u0, v0), (u1, v1) = (T @ [lo[0], hi[1], 1])[:2], (T @ [hi[0], lo[1], 1])[:2]   # table corners -> warped
    inner[int(min(v0, v1)):int(max(v0, v1)), int(min(u0, u1)):int(max(u0, u1))] = 255
    # what the camera actually sees: outside it the warp is black fill. Anything touching that edge (the gripper
    # fingers at the bottom of every frame, a hand reaching in) isn't a tool on the tray.
    seen = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), T @ tmap.H, (img.shape[1], img.shape[0]))
    seen = cv2.erode(seen, np.ones((5, 5), np.uint8))
    cv2.rectangle(seen, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), 0, 2)      # the warp's own border too
    unseen = seen == 0
    bg = np.median(img[(tagm == 0) & (inner > 0) & ~unseen].reshape(-1, 3), axis=0)
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = ((diff > config.CONTRAST_MIN) & (tagm == 0) & ~unseen).astype(np.uint8) * 255
    edge = cv2.dilate(unseen.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    k3 = max(3, int(0.0015 * k)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k3, k3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k3 + 2, k3 + 2), np.uint8))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
        area_m2 = cv2.contourArea(c) / k / k
        if area_m2 < config.TOOL_MIN_AREA_M2:
            continue
        filled = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(filled, [c], -1, 255, -1)
        if np.any(edge[filled > 0]):
            continue                                              # touches the edge of the view: the gripper, a hand
        pts = np.argwhere(filled > 0)[:, ::-1].astype(float)       # (u, v)
        centre = pts.mean(axis=0)
        axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
        stations, width = _width_profile(pts, centre, axis, n=60)
        length = stations[-1] - stations[0] + (stations[1] - stations[0])
        n = len(width)
        ends = width[: n // 3].mean(), width[-(n // 3):].mean()
        if ends[0] < ends[1]:                                     # make the handle the first end
            axis, stations, width = -axis, -stations[::-1], width[::-1]
            ends = ends[::-1]
        handle_w = np.percentile(width[: n // 2][width[: n // 2] > 0], 80) if np.any(width[: n // 2] > 0) else 0
        shaft_w = np.percentile(width[n // 2:][width[n // 2:] > 0], 20) if np.any(width[n // 2:] > 0) else 0
        if handle_w <= 0 or length / handle_w < config.MIN_ELONGATION or shaft_w > 0.6 * handle_w:
            continue                                              # a blob that isn't a handle with a shaft
        thr = (handle_w + shaft_w) / 2
        narrow = width < thr
        ferrule_i = None
        for i in range(n - 2):
            if narrow[i] and narrow[i + 1] and narrow[i + 2]:
                ferrule_i = i
                break
        if ferrule_i is None:
            continue
        # where the width crosses thr, interpolated between the last wide station and the first narrow one
        s_f = stations[ferrule_i]
        if ferrule_i > 0 and width[ferrule_i - 1] > width[ferrule_i]:
            frac = (width[ferrule_i - 1] - thr) / (width[ferrule_i - 1] - width[ferrule_i])
            s_f = stations[ferrule_i - 1] + frac * (stations[ferrule_i] - stations[ferrule_i - 1])
        shaft_len = (stations[-1] - s_f) / k
        if shaft_len < config.GRASP_FROM_FERRULE_M + config.MIN_SHAFT_BEYOND_GRASP_M:
            continue                                              # no room on the shaft for the fingers
        # warped (u, v) -> table (x, y): invert T
        Tinv = np.linalg.inv(T)
        to_table = lambda uv: (Tinv @ np.array([uv[0], uv[1], 1.0]))[:2]
        ferrule_uv = centre + axis * s_f
        handle_uv = centre + axis * stations[: max(ferrule_i, 1)].mean()
        ferrule, handle = to_table(ferrule_uv), to_table(handle_uv)
        d = handle - ferrule
        d /= np.linalg.norm(d)
        grasp = ferrule - d * config.GRASP_FROM_FERRULE_M
        if not on_tray(grasp):
            continue
        # the outline back in camera pixels, for colour_class, the namer's marks and draw()
        cam = cv2.perspectiveTransform(c.reshape(-1, 1, 2).astype(np.float32), Hinv @ Tinv).astype(np.int32)
        out.append({"grasp_xy": grasp, "handle_dir": d, "grasp_px": tmap.table_to_px(*grasp),
                    "handle_px": tmap.table_to_px(*handle), "area": cv2.contourArea(cam), "contour": cam,
                    "colour": colour_class(hsv, cam), "width_m": handle_w / k, "length_m": length / k,
                    "shaft_w_m": shaft_w / k})
    return sorted(out, key=lambda t: -t["area"])


def on_tray(xy):
    """Inside the rectangle through the TABLE_TAGS centres, plus TRAY_MARGIN_M."""
    xs, ys = zip(*config.TABLE_TAGS.values())
    m = config.TRAY_MARGIN_M
    return min(xs) - m <= xy[0] <= max(xs) + m and min(ys) - m <= xy[1] <= max(ys) + m


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
    bad = f", tag {tmap.bad_tag} doesn't match TABLE_TAGS" if tmap.bad_tag is not None else ""
    status = (f"CALIBRATED (live, {tmap.tags_used} tags{bad})" if tmap.calibrated_from == "live" else
              "CALIBRATED (cached: tags hidden)" if tmap.calibrated_from == "cached" else
              "NOT CALIBRATED: " + tmap.why_not())
    cv2.putText(out, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if tmap.calibrated_from == "live" else (0, 200, 255) if tmap.H is not None else (0, 0, 255), 2)
    return out
