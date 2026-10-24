"""Channel A of the demo: a surgical clip with its phase timeline over it, driving the robot.

Plays the clip from a local file (offline) at its own frame rate, with the current phase, the instrument it calls for,
the last request it sent and a phase timeline drawn over it. When the phase CHANGES to one in config.PHASE_TOOLS (the
first phase at 0 s counts, repeated rows of the same phase don't), it sends {"type": "request", "tool", "source":
"phase", "phase"} over edgehand/link.py to the laptop that runs the arm (run on the laptop itself: --host 127.0.0.1).

    python -m edgehand.phase_player ~/Desktop/clippingdemo.mp4 ../../../data/phases/clippingdemo.csv
    python -m edgehand.phase_player clip.mp4 phases.csv --headless       print the requests it would send, no window
    python -m edgehand.phase_player clip.mp4 phases.csv --no-window --speed 8 --port 5555    the same run, no window
                                                                        (what test/test_phase.py listens to)

phases.csv: a header, then "time_s,phase" rows (a row starts a phase that lasts until the next row; per-second or
per-frame rows work too). There is no phase model in the demo: data/phases/clippingdemo.csv is a HAND annotation of
the clip, and the overlay says so ("phase (annotated)", "hand-annotated timeline, not model output"). --model-output
is only for a CSV that really came from phase/predict_phases.py. Keys: SPACE pause, ESC quit.
"""
import argparse, csv, time
import cv2
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


def changes(segs, mapping):
    """[(time_s, phase, tool)] where a phase that calls for a tool starts."""
    return [(s, ph, mapping[ph]) for s, _, ph in segs if ph in mapping]


def overlay(frame, segs, t, duration, mapping, sent=None, model_output=False):
    """sent: (time_s, phase, tool) of the last request, shown so the audience sees what went to the arm."""
    out = cv2.resize(frame, (W, H))
    ph = phase_at(segs, t)
    tool = mapping.get(ph)
    cv2.rectangle(out, (0, 0), (W, 110), (0, 0, 0), -1)
    cv2.putText(out, f"Phase ({'model' if model_output else 'annotated'}): {ph or '-'}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(out, f"Robot: {'fetch ' + tool if tool else '-'}", (W // 2 + 40, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (120, 255, 120) if tool else (200, 200, 200), 2)
    cv2.putText(out, f"t = {t:5.1f} / {duration:.1f} s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    last = f"sent: request {sent[2]} ({sent[1]}) at {sent[0]:.1f} s" if sent else "sent: nothing yet"
    cv2.putText(out, last, (W // 2 + 40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120) if sent else (200, 200, 200), 2)
    names = list(dict.fromkeys(p for _, _, p in segs))
    y0 = H - 40
    for s, e, p in segs:                              # the timeline: one colour per phase, cursor at now
        x0, x1 = int(W * s / duration), int(W * min(e, duration) / duration)
        cv2.rectangle(out, (x0, y0), (x1, H), COLOURS[names.index(p) % len(COLOURS)], -1)
    cx = int(W * t / duration)
    cv2.line(out, (cx, y0 - 8), (cx, H), (255, 255, 255), 3)
    label = ("phase model predictions on this clip, replayed" if model_output
             else "HAND-ANNOTATED phase timeline (not model output)")
    cv2.putText(out, label, (20, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if model_output else (0, 200, 255), 2)
    return out


def advance(cap, fps, pos, want, decode=True):
    """Move the clip from frame pos to frame want and return (ok, frame, new_pos). Frames the display couldn't keep
    up with are grabbed and dropped (decoded, not converted); more than a second behind, it seeks instead. With
    decode=False (no window) the frame itself isn't retrieved, so faster-than-real-time runs keep their pacing."""
    if want - pos > fps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, want)
        pos = want
    while pos < want and cap.grab():
        pos += 1
    if decode:
        ok, frame = cap.read()
    else:
        ok, frame = cap.grab(), None
    return ok, frame, pos + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("phases")
    ap.add_argument("--host", default="127.0.0.1", help="the laptop running the arm (Jetson -> 192.168.55.100)")
    ap.add_argument("--port", type=int, default=link.PORT, help="event link port (tests pick a spare one)")
    ap.add_argument("--map", nargs="*", default=[], help="phase=tool, adds to config.PHASE_TOOLS")
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate; 1 = the clip's own frame rate")
    ap.add_argument("--no-send", action="store_true", help="show only, send nothing to the robot")
    ap.add_argument("--headless", action="store_true", help="no window: print the requests and their times, don't play")
    ap.add_argument("--no-window", action="store_true", help="play and send with the same pacing, but draw nothing")
    ap.add_argument("--model-output", action="store_true",
                    help="the CSV really came from phase/predict_phases.py: label it as model predictions")
    a = ap.parse_args()
    mapping = dict(config.PHASE_TOOLS)
    mapping.update(kv.split("=", 1) for kv in a.map)
    unknown = {t for t in mapping.values() if t not in config.TOOLS}
    if unknown:
        raise SystemExit(f"tools not in config.TOOLS: {sorted(unknown)} (classes: {list(config.TOOLS)})")
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"can't open {a.video}")
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = n / fps if n > 0 else None
    segs = load_segments(a.phases, duration)
    duration = duration or segs[-1][1]
    print(f"{a.video}: {duration:.1f} s at {fps:.1f} fps; {len(segs)} phase segments; requests at:")
    for t, ph, tool in changes(segs, mapping):
        print(f"  {t:6.1f} s  {ph} -> {tool}")
    if a.headless:
        return
    beat = None if a.no_send else link.Heartbeat("phase_player", a.host, a.port)
    t0, paused_at, last, sent, pos, shown = time.time(), None, None, None, 0, None
    while True:
        if paused_at is None:
            t = (time.time() - t0) * a.speed
            if t >= duration:
                break
            want = int(t * fps)
            if want >= pos:                                        # the next frame is due
                ok, frame, pos = advance(cap, fps, pos, want, decode=not a.no_window)
                if not ok:
                    break
                ph = phase_at(segs, t)
                if ph != last:
                    tool = mapping.get(ph)
                    print(f"{t:6.1f} s  phase: {ph}" + (f"  -> request {tool}" if tool else ""), flush=True)
                    if tool:
                        sent = (t, ph, tool)
                        if not a.no_send:
                            link.send({"type": "request", "tool": tool, "source": "phase", "phase": ph}, a.host, a.port)
                    last = ph
                if not a.no_window:
                    shown = overlay(frame, segs, t, duration, mapping, sent, a.model_output)
            wait_ms = max(1, int((pos / fps - t) / a.speed * 1000))   # until frame pos is due: the clip's own rate
        else:
            wait_ms = 30
        if a.no_window:
            time.sleep(wait_ms / 1000)
            continue
        cv2.imshow("EdgeHand: surgical phase (annotated) -> instrument", shown)
        k = cv2.waitKey(wait_ms) & 0xFF
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
