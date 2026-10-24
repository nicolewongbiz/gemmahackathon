"""Draw a fake overhead camera frame: dark mat, the 4 table tags, screwdrivers at known table poses.
The camera is a pure homography of the table (like a camera looking straight down)."""
import cv2
import numpy as np
from edgehand import config

W, H = 1280, 720
COLOURS = {"phillips": (30, 30, 210), "flathead": (30, 210, 230), "torx": (40, 170, 40)}   # BGR


def camera_H():
    """table (x, y) metres -> pixel. x forward = image up, y left = image left."""
    s = 1400.0                                    # px per metre
    cx, cy = 0.475, 0.0                           # table point at the image centre
    return np.array([[0, -s, W / 2 + s * cy], [-s, 0, H / 2 + s * cx], [0, 0, 1.0]])


def to_px(Hc, x, y):
    p = Hc @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def frame(tools, tag_xy=None):
    """tools: list of (name, grasp_xy, handle_dir). Returns BGR image."""
    Hc = camera_H()
    img = np.full((H, W, 3), 25, np.uint8)
    half = config.TAG_SIZE_M / 2
    for tid, (x, y) in (tag_xy or config.TABLE_TAGS).items():
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tid, 80)
        marker = cv2.copyMakeBorder(marker, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        q = half * 10 / 8
        src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        # tag image top edge faces +x (away from the arm)
        dst = np.float32([to_px(Hc, x + q, y + q), to_px(Hc, x + q, y - q), to_px(Hc, x - q, y - q), to_px(Hc, x - q, y + q)])
        warped = cv2.warpPerspective(cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR), cv2.getPerspectiveTransform(src, dst), (W, H))
        mask = cv2.warpPerspective(np.full((100, 100), 255, np.uint8), cv2.getPerspectiveTransform(src, dst), (W, H))
        img[mask > 0] = warped[mask > 0]
    for name, g, d in tools:
        g, d = np.array(g, float), np.array(d, float) / np.linalg.norm(d)
        ferrule = g + d * config.GRASP_FROM_FERRULE_M
        tip = ferrule - d * 0.10                                   # 10 cm shaft
        handle_end = ferrule + d * 0.10                            # 10 cm handle
        px = lambda p: tuple(int(round(v)) for v in to_px(Hc, *p))
        cv2.line(img, px(tip), px(ferrule), (185, 185, 190), 8)                     # grey metal shaft
        n = np.array([-d[1], d[0]]) * 0.012                         # 2.4 cm wide handle, flat ends
        poly = np.array([px(ferrule + n), px(handle_end + n), px(handle_end - n), px(ferrule - n)], np.int32)
        cv2.fillConvexPoly(img, poly, COLOURS[name])
    return img
