import threading, time
import cv2
import config


class Camera:
    """Background-thread webcam so the latest frame is always ready."""
    def __init__(self, index=config.CAMERA_INDEX):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        self.frame = None
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
        while self.frame is None:
            time.sleep(0.05)

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                self.frame = f

    def read(self):
        return self.frame.copy()

    def close(self):
        self._run = False
        self.cap.release()
