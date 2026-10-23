"""AprilTag pixel->table homography. Run directly to check calibration:
    python -m perception.calibrate      (click anywhere to print table x,y)
"""
import cv2
import numpy as np
from pupil_apriltags import Detector
import config

_det = Detector(families="tag36h11")


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {d.tag_id: d.center for d in _det.detect(gray)}


class TableMap:
    def __init__(self):
        self.H = None
        self.bins_px = {}

    def update(self, frame):
        tags = detect(frame)
        ids = [i for i in config.TABLE_TAGS if i in tags]
        if len(ids) >= 4:
            src = np.array([tags[i] for i in ids], np.float32)
            dst = np.array([config.TABLE_TAGS[i] for i in ids], np.float32)
            self.H, _ = cv2.findHomography(src, dst)
        self.bins_px = {config.BIN_TAGS[i]: tags[i] for i in config.BIN_TAGS if i in tags}
        return self.H is not None

    def px_to_table(self, u, v):
        if self.H is None:
            raise RuntimeError("Not calibrated: need all 4 corner tags visible")
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def bin_xy(self, name):
        if name in self.bins_px:
            return self.px_to_table(*self.bins_px[name])
        return config.BIN_DROP[name][:2]

    def tag_mask(self, frame):
        """Mask (255 = tag) so tags are not proposed as objects."""
        mask = np.zeros(frame.shape[:2], np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for d in _det.detect(gray):
            cv2.fillConvexPoly(mask, d.corners.astype(np.int32), 255)
        return cv2.dilate(mask, np.ones((25, 25), np.uint8))


if __name__ == "__main__":
    from perception.camera import Camera
    cam, tm = Camera(), TableMap()
    clicked = []
    cv2.namedWindow("calib")
    cv2.setMouseCallback("calib", lambda e, x, y, *_: clicked.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        f = cam.read()
        ok = tm.update(f)
        for tid, c in detect(f).items():
            cv2.putText(f, str(tid), tuple(map(int, c)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(f, "CALIBRATED" if ok else "need tags 0-3", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if ok else (0, 0, 255), 2)
        if clicked and ok:
            print("pixel", clicked[-1], "-> table", tm.px_to_table(*clicked.pop()))
        cv2.imshow("calib", f)
        if cv2.waitKey(1) == 27:
            break
