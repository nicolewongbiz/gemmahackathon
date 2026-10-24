"""Save one frame from each camera index to data/cam<i>.jpg so you can see which is the webcam.
Windows also lists virtual cameras (OBS) and the laptop camera; set config.WEBCAM_INDEX to the right one.
Run: python -m scripts.find_camera
"""
import time
import cv2
import config

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    frame = None
    for _ in range(20):          # first frames are often black while exposure settles
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    if frame is None:
        print(f"{i}: opens but gives no frames")
        continue
    path = config.DATA / f"cam{i}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"{i}: {frame.shape[1]}x{frame.shape[0]}  -> {path}")
print(f"WEBCAM_INDEX is currently {config.WEBCAM_INDEX}")
