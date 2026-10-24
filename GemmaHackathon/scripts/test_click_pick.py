"""Milestone 1 (by 2:00): click an object -> arm picks it and drops it in bin A.
    python -m scripts.test_click_pick      (or double-click click_pick.bat)
Keys: E = save the empty table (do this first, with the table clear), Esc = quit.
A click snaps to the detected centre of the object under the cursor, so you don't have to hit it exactly.
"""
import cv2
from perception.camera import make_camera
from perception.calibrate import TableMap, detect
from perception.objects import ObjectFinder
from robot.arm import make_arm, pick_and_place

GREEN, RED, YELLOW = (0, 255, 0), (0, 0, 255), (0, 255, 255)

cam, tm, finder, arm = make_camera(), TableMap(), ObjectFinder(), make_arm()
mask = None   # tag areas, computed when the empty table is saved (tags don't move)
print("Moving to LOOK_POSE ...")
arm.look()
clicks = []
cv2.namedWindow("pick")
cv2.setMouseCallback("pick", lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
print("Clear the table and press E in the 'pick' window to save it empty. Then place objects and click one.")


def look_safely():
    try:
        arm.look()
    except Exception as e:  # noqa: BLE001
        print("arm error going back to LOOK_POSE:", e)


while True:
    f = cam.read()
    raw = f.copy()
    tags = detect(f)
    tm.update(f)
    ok = tm.H is not None
    if mask is not None:
        for (u, v), cnt in finder.blobs(raw, mask):
            cv2.drawContours(f, [cnt], -1, YELLOW, 2)
            cv2.drawMarker(f, (int(u), int(v)), YELLOW, cv2.MARKER_CROSS, 24, 2)
    for tid, c in tags.items():
        cv2.putText(f, str(tid), tuple(map(int, c)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    if not ok:
        status, col = "NOT CALIBRATED - need tags 0-3 in view", RED
    elif finder.empty is None:
        status, col = "Clear the table, then press E", RED
    else:
        status, col = "CALIBRATED - click an object", GREEN
    cv2.putText(f, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
    cv2.imshow("pick", f)

    if clicks:
        u, v = clicks.pop()
        clicks.clear()
        if not ok:
            print(f"click ignored: not calibrated, seeing tags {sorted(tags)} (need 0, 1, 2, 3)")
        elif mask is None:
            print("click ignored: press E with the table clear first")
        else:
            c = finder.at(raw, u, v, mask)
            if c is None:
                print(f"click ignored: no object detected at pixel ({u}, {v})")
            else:
                xy = tm.px_to_table(*c)
                print(f"pick at x={xy[0]:.3f} y={xy[1]:.3f}  (object centre {int(c[0])}, {int(c[1])}; you clicked {u}, {v})")
                cv2.circle(f, (int(c[0]), int(c[1])), 14, RED, 3); cv2.imshow("pick", f); cv2.waitKey(1)
                try:
                    pick_and_place(arm, xy, tm.bin_xy("A"))
                except ValueError as e:
                    print("refused:", e)
                except Exception as e:  # noqa: BLE001  keep the window alive on arm errors
                    print("arm error:", e)
                look_safely()
                clicks.clear()   # drop clicks made while the arm was busy

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in (ord("e"), ord("E")):
        finder.snapshot_empty(raw)
        mask = tm.tag_mask(raw)
        print("Saved the empty table. Place objects and click one.")
