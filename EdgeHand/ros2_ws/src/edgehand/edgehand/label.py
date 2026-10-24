"""Label real camera frames for the vision regression test (test/test_real_frames.py). No robot motion.

    python -m edgehand.label --capture [dir] [--note "side lamp"]   live RealSense: SPACE saves a frame, then label it
    python -m edgehand.label [dir]                                   label the jpgs in dir that have no .json yet
dir defaults to EdgeHand/data/frames. Frames are saved like the stream reaches vision: 1280x720, jpg quality 70.

Labelling a frame (the 4 table tags must be in view):
    1..9   pick the class of the next tool (config.TOOLS order, shown on screen)
    click  its grasp point on the shaft, then a point towards the handle
    u      undo the last tool        ENTER  save and go on        s  skip        q  quit
A frame with no tools (empty tray, or only a non-tool in it) is labelled by pressing ENTER straight away.
Labels are pixels, so they stay valid when TABLE_TAGS is re-measured: <frame>.json =
    {"note": "...", "tools": [{"tool": "phillips", "grasp_px": [u, v], "handle_px": [u, v]}]}
"""
import argparse, json, time
from pathlib import Path
import cv2
from edgehand import config
from edgehand.vision import TableMap, draw

FRAMES = config.ROOT / "data" / "frames"
W, H, QUALITY = 1280, 720, 70


def save_frame(frame, folder, note=""):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), cv2.resize(frame, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if note:
        path.with_suffix(".note").write_text(note)
    return path


def label(path, note=""):
    """Interactive labelling of one frame. Returns False if the user quit."""
    img = cv2.imread(str(path))
    tm = TableMap()
    calibrated = tm.update(img)
    names = list(config.TOOLS)
    note = note or (path.with_suffix(".note").read_text() if path.with_suffix(".note").exists() else "")
    tools, current, clicks = [], None, []
    win = "label (1-9 class, click grasp then handle, u undo, ENTER save, s skip, q quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        if current and clicks:
            current.setdefault("pts", []).append(clicks.pop(0))
            if len(current["pts"]) == 2:
                tools.append({"tool": current["tool"], "grasp_px": list(current["pts"][0]),
                              "handle_px": list(current["pts"][1])})
                current = None
        clicks.clear() if not current else None
        out = draw(img, tm, [])
        for t in tools:
            cv2.arrowedLine(out, tuple(t["grasp_px"]), tuple(t["handle_px"]), (0, 255, 255), 2, tipLength=0.2)
            cv2.circle(out, tuple(t["grasp_px"]), 6, (0, 0, 255), -1)
            cv2.putText(out, t["tool"], (t["grasp_px"][0] + 8, t["grasp_px"][1] - 8), 0, 0.6, (0, 255, 255), 2)
        help_ = "  ".join(f"{i + 1}={n}" for i, n in enumerate(names))
        step = (f"{current['tool']}: click the {'grasp point' if not current.get('pts') else 'handle direction'}"
                if current else "press a class key, or ENTER to save")
        for i, line in enumerate([help_, step, "" if calibrated else "WARNING: not all 4 tags visible"]):
            cv2.putText(out, line, (15, H - 70 + 25 * i), 0, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, out)
        k = cv2.waitKey(30) & 0xFF
        if ord("1") <= k <= ord("9") and k - ord("1") < len(names):
            current = {"tool": names[k - ord("1")]}
        elif k == ord("u") and tools:
            tools.pop()
        elif k in (13, 10):
            path.with_suffix(".json").write_text(json.dumps({"note": note, "tools": tools}, indent=1))
            print(f"saved {path.with_suffix('.json').name}: {len(tools)} tools")
            return True
        elif k == ord("s"):
            return True
        elif k == ord("q"):
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(FRAMES))
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--note", default="", help="lighting / scene, e.g. 'flashlight glare'")
    a = ap.parse_args()
    folder = Path(a.dir)
    if a.capture:
        from edgehand.camera import StreamCamera
        cam = StreamCamera()
        if not cam.wait_first():
            raise SystemExit(f"no frames from camera {cam.source}")
        print("SPACE = save and label this frame, q = quit")
        while True:
            cv2.imshow("capture (SPACE save, q quit)", cam.frame)
            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                if not label(save_frame(cam.frame, folder, a.note), a.note):
                    break
            elif k == ord("q"):
                break
        cam.close()
    else:
        for p in sorted(folder.glob("*.jpg")):
            if not p.with_suffix(".json").exists() and not label(p, a.note):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
