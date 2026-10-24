"""Colour detectors for the ball -> box task: a white ball on a black table, and a blue box."""
import cv2
import numpy as np
import config


def find_ball(frame, ignore_mask=None):
    """(u, v) pixel centre of the roundest white blob, or None. ignore_mask (255 = skip) hides the tags' white borders."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, config.BALL_MIN_VAL), (180, config.BALL_MAX_SAT, 255))
    if ignore_mask is not None:
        m[ignore_mask > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    best, best_score = None, 0.6           # circularity 1.0 = perfect disc
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if not config.BALL_AREA_PX[0] <= area <= config.BALL_AREA_PX[1]:
            continue
        circ = 4 * np.pi * area / cv2.arcLength(c, True) ** 2
        if circ > best_score:
            (u, v), _ = cv2.minEnclosingCircle(c)
            best, best_score = (float(u), float(v)), circ
    return best


def find_box(frame):
    """(u, v) pixel centre of the largest blue blob, or None if there's none or it runs off the frame
    (a cut-off blob's centre is wrong; the caller falls back to BIN_DROP)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, config.BOX_HSV_LO, config.BOX_HSV_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts = [c for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if cv2.contourArea(c) >= config.BOX_MIN_AREA_PX]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if x <= 2 or y <= 2 or x + w >= frame.shape[1] - 2 or y + h >= frame.shape[0] - 2:
        return None
    M = cv2.moments(c)
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def in_box(frame, px, ring=(35, 60)):
    """True if the ball at pixel px is surrounded by blue (it's sitting in the box). Counts only the part of the
    ring that's inside the frame, so it still works when the box runs off the edge of the image."""
    m = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), config.BOX_HSV_LO, config.BOX_HSV_HI)
    yy, xx = np.ogrid[:m.shape[0], :m.shape[1]]
    d2 = (xx - px[0]) ** 2 + (yy - px[1]) ** 2
    annulus = (d2 >= ring[0] ** 2) & (d2 <= ring[1] ** 2)
    return annulus.any() and (m[annulus] > 0).mean() > 0.4


def draw(frame, ball, box):
    out = frame.copy()
    if ball:
        cv2.circle(out, tuple(map(int, ball)), 20, (0, 255, 0), 3)
        cv2.putText(out, "ball", (int(ball[0]) + 24, int(ball[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if box:
        cv2.drawMarker(out, tuple(map(int, box)), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.putText(out, "box", (int(box[0]) + 24, int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out
