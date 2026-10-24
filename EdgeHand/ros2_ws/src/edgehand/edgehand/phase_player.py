"""Channel A of the demo: a surgical clip with the phase model's predictions, driving the robot.

Plays the clip from a local file (offline) with the current phase, the instrument it calls for and a phase timeline
drawn over it. When the phase changes to one in config.PHASE_TOOLS, it sends {"type": "request", "tool", "source":
"phase"} over edgehand/link.py to the laptop that runs the arm (run on the laptop itself: --host 127.0.0.1).

    python -m edgehand.phase_player clip.mp4 phases.csv [--host 192.168.55.100] [--map Dissection=flathead]
    python -m edgehand.phase_player clip.mp4 phases.csv --headless       print the requests it would send, no window

phases.csv is the phase model's output on THIS clip: a header, then "time_s,phase" rows (a row starts a phase that
lasts until the next row; per-second or per-frame rows work too). --placeholder stamps PLACEHOLDER on the video, for
timelines that aren't real model output. Keys: SPACE pause, ESC quit.
"""
import argparse, csv, time
import cv2
import numpy as np
from edgehand import config, link

W, H = 1280, 720
COLOURS = [(66, 135, 245), (80, 200, 120), (230, 160, 60), (200, 90, 200), (60, 200, 220), (220, 80, 80),
           (150, 150, 150), (240, 220, 80)]


def load_segments(path, duration_s=None):
    """[(start_s, end_s, phase)], consecutive equal phases merged."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = float(r.get("time_s") or r.get("start_s") or r.get("t"))
            rows.append((t, (r.get("phase") or "").strip()))
    rows.sort()
    segs = []
    for (t, ph), nxt in zip(rows, rows[1:] + [(duration_s if duration_s is not None else rows[-1][0] + 1, None)]):
        if segs and segs[-1][2] == ph:
            segs[-1] = (segs[-1][0], nxt[0], ph)
        else:
            segs.append((t, nxt[0], ph))
    return segs


def phase_at(segs, t):
    for s, e, ph in segs:
        if s <= t < e:
            return ph
    return None


def overlay(frame, segs, t, duration, mapping, placeholder):
    out = cv2.resize(frame, (W, H))
    ph = phase_at(segs, t)
    tool = mapping.get(ph)
    cv2.rectangle(out, (0, 0), (W, 70), (0, 0, 0), -1)
    cv2.putText(out, f"Phase: {ph or '-'}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(out, f"Robot: {'fetch ' + tool if tool else '-'}", (W // 2 + 40, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (120, 255, 120) if tool else (200, 200, 200), 2)
    names = list(dict.fromkeys(p for _, _, p in segs))
    y0 = H - 40
    for s, e, p in segs:                              # the timeline: one colour per phase, cursor at now
        x0, x1 = int(W * s / duration), int(W * min(e, duration) / duration)
        cv2.rectangle(out, (x0, y0), (x1, H), COLOURS[names.index(p) % len(COLOURS)], -1)
    cx = int(W * t / duration)
    cv2.line(out, (cx, y0 - 8), (cx, H), (255, 255, 255), 3)
    label = "PLACEHOLDER TIMELINE (not model output)" if placeholder else "phase model predictions on this clip, replayed"
    cv2.putText(out, label, (20, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if placeholder else (255, 255, 255), 2)
    return out


def changes(segs, mapping):
    """[(time_s, phase, tool)] where a phase that calls for a tool starts."""
    return [(s, ph, mapping[ph]) for s, _, ph in segs if ph in mapping]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("phases")
    ap.add_argument("--host", default="127.0.0.1", help="the laptop running the arm (Jetson -> 192.168.55.100)")
    ap.add_argument("--map", nargs="*", default=[], help="phase=tool, adds to config.PHASE_TOOLS")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-send", action="store_true", help="show only, send nothing to the robot")
    ap.add_argument("--headless", action="store_true", help="no window: print the requests and their times")
    ap.add_argument("--placeholder", action="store_true", help="stamp PLACEHOLDER: the timeline isn't model output")
    a = ap.parse_args()
    mapping = dict(config.PHASE_TOOLS)
    mapping.update(kv.split("=", 1) for kv in a.map)
    unknown = {t for t in mapping.values() if t not in config.TOOLS}
    if unknown:
        raise SystemExit(f"tools not in config.TOOLS: {sorted(unknown)} (classes: {list(config.TOOLS)})")
    cap = cv2.VideoCapture(a.video)
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = n / fps if n > 0 else None
    segs = load_segments(a.phases, duration)
    duration = duration or segs[-1][1]
    print(f"{a.video}: {duration:.1f} s; {len(segs)} phase segments; requests at:")
    for t, ph, tool in changes(segs, mapping):
        print(f"  {t:6.1f} s  {ph} -> {tool}")
    if a.headless:
        return
    beat = None if a.no_send else link.Heartbeat("phase_player", a.host)
    t0, paused_at, last = time.time(), None, None
    while True:
        if paused_at is None:
            t = (time.time() - t0) * a.speed
            if t >= duration:
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000) if abs(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - t) > 0.5 else None
            ok, frame = cap.read()
            if not ok:
                break
            ph = phase_at(segs, t)
            if ph != last:
                tool = mapping.get(ph)
                print(f"{t:6.1f} s  phase: {ph}" + (f"  -> request {tool}" if tool else ""))
                if tool and not a.no_send:
                    link.send({"type": "request", "tool": tool, "source": "phase", "phase": ph}, a.host)
                last = ph
            shown = overlay(frame, segs, t, duration, mapping, a.placeholder)
        cv2.imshow("EdgeHand: surgical phase -> instrument", shown)
        k = cv2.waitKey(max(1, int(1000 / min(fps, 30)))) & 0xFF
        if k == 27:
            break
        if k == ord(" "):
            if paused_at is None:
                paused_at = time.time()
            else:
                t0 += time.time() - paused_at
                paused_at = None
    if beat:
        beat.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
