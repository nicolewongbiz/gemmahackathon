"""Milestone 1 (by 2:00): click a pixel -> arm picks there and drops in bin A.
    python -m scripts.test_click_pick
"""
import cv2
from perception.camera import Camera
from perception.calibrate import TableMap
from robot.arm import make_arm, pick_and_place

cam, tm, arm = Camera(), TableMap(), make_arm()
clicks = []
cv2.namedWindow("pick")
cv2.setMouseCallback("pick", lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
while True:
    f = cam.read(); tm.update(f)
    cv2.imshow("pick", f)
    if clicks and tm.H is not None:
        xy = tm.px_to_table(*clicks.pop())
        print("pick at", xy)
        pick_and_place(arm, xy, tm.bin_xy("A"))
    if cv2.waitKey(1) == 27:
        break
