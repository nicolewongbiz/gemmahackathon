"""Pick the white ball off the table and drop it in the blue box. No Gemma, just colour + AprilTags.
    python -m scripts.ball_to_box          (real arm / mock, per config.py)
    sim/.venv/bin/python -m sim.run         (same code in MuJoCo)
"""
import cv2
import config
from perception.ball import find_ball, find_box, in_box, draw
from perception.calibrate import TableMap
from robot.arm import pick_and_place


def run(arm, cam, tmap=None, should_stop=lambda: False, log=print):
    """Returns (ok, message). Leaves the arm at LOOK_POSE."""
    tmap = tmap or TableMap()
    arm.look()
    frame = cam.read()
    if not tmap.update(frame):
        return False, "can't see all four table tags"
    ball = find_ball(frame, tmap.tag_mask(frame))
    box = find_box(frame)
    cv2.imwrite(str(config.DATA / "last_ball.jpg"), draw(frame, ball, box))
    if ball is None:
        return False, "no ball found"
    ball_xy = tmap.px_to_table(*ball)
    box_xy = tmap.px_to_table(*box) if box else config.BIN_DROP["A"][:2]
    log(f"ball px {tuple(round(c) for c in ball)} -> {ball_xy[0]:.3f}, {ball_xy[1]:.3f} m | "
        f"box {'px ' + str(tuple(round(c) for c in box)) if box else 'not seen, using BIN_DROP A'} -> "
        f"{box_xy[0]:.3f}, {box_xy[1]:.3f} m")
    if not pick_and_place(arm, ball_xy, box_xy, should_stop):
        return False, "stopped"
    arm.look()
    after = cam.read()
    left = find_ball(after, tmap.tag_mask(after))
    cv2.imwrite(str(config.DATA / "last_ball_after.jpg"), draw(after, left, find_box(after)))
    if left is None:
        # a ball in the box is still visible, unless the box is out of view
        return (False, "lost the ball") if box else (True, "ball gone from the table (box not in view to confirm)")
    if not in_box(after, left):
        return False, "ball still on the table" if tmap.pickable(*left) else "ball missed the box"
    return True, "ball in box"


if __name__ == "__main__":
    from perception.camera import make_camera
    from robot.arm import make_arm
    config.DATA.mkdir(exist_ok=True)
    arm, cam = make_arm(), make_camera()
    try:
        print(run(arm, cam))
    finally:
        cam.close(); arm.close()
