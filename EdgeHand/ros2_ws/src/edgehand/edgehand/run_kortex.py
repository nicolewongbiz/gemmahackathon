"""EdgeHand without ROS: wrist camera -> vision -> Gemma names the tools -> handover.py on the Kortex API.
Runs on the computer the camera is cabled to (this Mac). Red E-stop in hand; MoveIt isn't involved.

    python -m edgehand.run_kortex --air                 first runs: every grasp 3 cm above the tools
    python -m edgehand.run_kortex                       type requests ("hand me the green screwdriver"); Ctrl-C stops
    python -m edgehand.run_kortex --tool torx           one request, no intent model
    python -m edgehand.run_kortex --look-only           move to the look pose, show what vision sees, no pick
    python -m edgehand.run_kortex --namer colour        name tools by handle colour instead of Gemma
    python -m edgehand.run_kortex --namer first --tool x  ONE tool on the tray: whatever is found is the requested one
    python -m edgehand.run_kortex --listen              also take requests / release / stop from other nodes over
                                                        link.py (UDP 5006): the phase player, the Jetson voice node

Refuses to pick on config.py's placeholder calibration: run measure_tags first (data/calibration.json).
"""
import argparse, json, queue, sys, tempfile, threading, time
from pathlib import Path
import cv2
import numpy as np
from edgehand import config
from edgehand.camera import IR_HINT, StreamCamera, is_infrared
from edgehand.geometry import table_z
from edgehand.handover import Handover
from edgehand.vision import TableMap, detect_tools, draw, find_tools


def calibration_gate(allow_defaults):
    f = getattr(config, "CALIBRATION_FILE", None)
    if f is None:
        print("NO SITE CALIBRATION: config.py placeholders (TABLE_TAGS, flat table at TABLE_Z).")
        if not allow_defaults:
            print("Run `python -m edgehand.measure_tags` on this cell first, or --allow-defaults to look (no picks).")
            sys.exit(2)
    else:
        meta = json.loads(Path(f).read_text())
        print(f"site calibration {f}: site '{meta.get('site', '?')}', measured {meta.get('measured', '?')}")
    if not config.TABLE_TOUCH_Z:
        print(f"TABLE_TOUCH_Z is empty: table assumed flat at z = {config.TABLE_Z} (README Phase 1.6)")
    else:
        zs = {i: round(table_z(*config.TABLE_TAGS[i]), 4) for i in config.TABLE_TOUCH_Z}   # raises if implausible
        print(f"table height at the tags (m): {zs}")
    print(f"clearance {config.TABLE_CLEARANCE_M * 1000:.0f} mm, fingertips {config.FINGERTIP_CLOSED_M:.3f} m, "
          f"air gap {config.AIR_GAP_M * 1000:.0f} mm, empty-gripper threshold {config.GRIPPER_EMPTY_POS}, "
          f"segmentation {config.SEGMENTATION}")


class Requests:
    """Where tool requests come from: typed text (a thread reading stdin) and, with --listen, link.py events.
    Phases change faster than a handover (20-40 s), so requests never pile up: while a handover runs, only the
    LATEST request is kept as pending (each newer one replaces it) and it runs when the arm is free again; a pending
    request for the tool just handed over is dropped. Stop and release act at once."""
    def __init__(self, handover, log=print):
        self.q, self.h, self.log = queue.Queue(), handover, log
        self.busy = False
        self.current = None                   # the tool being handed over right now
        self.pending = None                   # (tool, source) that arrived during the handover: the latest only
        self.nodes = set()

    def on_event(self, ev):
        kind = ev.get("type")
        src = ev.get("source") or ev.get("node") or "link"
        if kind == "stop":
            self.log(f"STOP from {src}")
            if self.h:
                self.h.stop()
        elif kind == "release":
            self.log(f"RELEASE from {src}")
            if self.h:
                self.h.release()
        elif kind == "request":
            tool = ev.get("tool")
            where = f" (phase {ev['phase']})" if ev.get("phase") else ""
            if tool not in config.TOOLS:
                self.log(f"request for {tool!r} from {src} ignored: not a tool class {list(config.TOOLS)}")
            elif self.busy:
                if tool == self.current:
                    self.log(f"request for {tool} from {src}{where}: already handing it over")
                    self.pending = None
                else:
                    if self.pending and self.pending[0] != tool:
                        self.log(f"pending request for {self.pending[0]} replaced by {tool} from {src}{where}")
                    else:
                        self.log(f"request for {tool} from {src}{where}: pending until the current handover ends")
                    self.pending = (tool, src)
            else:
                self.log(f"request from {src}: {tool}{where}")
                self.q.put(("tool", tool))

    def typed(self, text):
        self.q.put(("text", text))

    def start(self, tool):
        self.busy, self.current, self.pending = True, tool, None

    def finish(self):
        """The handover ended: the latest pending request (if any) becomes the next job, unless it's the same tool."""
        self.busy, done = False, self.current
        self.current = None
        if self.pending:
            tool, src = self.pending
            self.pending = None
            if tool != done:
                self.log(f"now the pending request from {src}: {tool}")
                self.q.put(("tool", tool))

    def next(self):
        return self.q.get()


class ArmState:
    """Tells the assistant what the arm is doing: {"type": "arm_state", "state": busy|holding|idle, "tool"} on the
    phase port, on every change and every ~2 s so a late-started assistant syncs. On the change to idle, "tool" is
    the one just handed over, so the assistant doesn't re-ask for it."""
    def __init__(self, send, log=print):
        self.send, self.log = send, log
        self.state, self.tool, self.last_tool = "idle", None, None
        self._stop = threading.Event()
        threading.Thread(target=self._beat, daemon=True).start()

    def update(self, handover_state, tool):
        s = "holding" if handover_state.startswith("holding") else (
            "idle" if handover_state in ("idle", "error", "stopped") else "busy")
        if s == "idle":
            self.last_tool, tool = tool or self.last_tool, tool or self.last_tool
        else:
            self.last_tool = tool
        if (s, tool) != (self.state, self.tool):
            self.state, self.tool = s, tool
            self._emit()

    def _emit(self):
        try:
            self.send({"type": "arm_state", "state": self.state, "tool": self.tool})
        except OSError as e:
            self.log(f"arm_state not sent: {e}")

    def _beat(self):
        while not self._stop.wait(2.0):
            self._emit()


KEYWORD_ALIASES = {"phillips": {"red", "red cap"}, "clear": {"clear", "transparent", "blue collar"},
                   "flathead": {"yellow collar"}, "torx": {"green one"}}   # what people say about the real tray


class KeywordIntent:
    """Typed requests without Gemma, for when the assistant owns the GPU: a tool class name, its colour word (from
    INSTRUMENTS) or its surgical role in the text picks the tool; 'stop' / 'release' / 'let go' are passed through
    as intents. Returns {"tool", "intent", "reply"} like GemmaIntent.parse."""
    def parse(self, text):
        t = text.lower()
        if any(w in t for w in ("stop", "halt", "freeze")):
            return {"tool": None, "intent": "stop", "reply": "Stopping."}
        if any(w in t for w in ("release", "let go", "open the gripper", "drop it")):
            return {"tool": None, "intent": "release", "reply": "Releasing."}
        words = {}
        for cls in config.TOOLS:
            words[cls] = {cls} | KEYWORD_ALIASES.get(cls, set())
            inst = getattr(config, "INSTRUMENTS", {}).get(cls)
            if inst:
                colour, role = inst[0], inst[1]
                words[cls] |= {colour, role} | set(role.split())
        for cls, ws in words.items():
            if any(w and w in t for w in ws):
                return {"tool": cls, "intent": "request", "reply": f"Fetching the {cls}."}
        return {"tool": None, "intent": "none", "reply": "Name a tool: " + ", ".join(config.TOOLS)}


def _read_stdin(requests):
    while True:
        try:
            text = input("\nrequest> ").strip()
        except EOFError:
            return
        if text:
            requests.typed(text)


class Eyes:
    """Fresh frames from the wrist camera, and what vision makes of them."""
    def __init__(self, namer):
        self.n = 0
        self.cam = StreamCamera(on_frame=self._count)
        self.tm = TableMap()
        self.namer = namer
        self.want = None                       # the requested class, for --namer first
        self.warned_ir = False
        self.last = None

    def _count(self, f):
        self.n += 1

    def frame(self):
        """A frame taken after now (the stream lags a few frames)."""
        target, t0 = self.n + config.LOCATE_SKIP_FRAMES, time.time()
        while self.n < target and time.time() - t0 < 5.0:
            time.sleep(0.02)
        return self.cam.frame

    def see(self, show):
        """[(tool dict with 'tool' name)], or None when not calibrated."""
        f = self.frame()
        if f is not None and is_infrared(f) and not self.warned_ir:
            self.warned_ir = True                  # geometry still works in IR (tags, contrast); colour naming won't
            print("CAMERA:", IR_HINT, "Carrying on in IR: use --namer first or gemma, not colour.")
        if f is None or not self.tm.update(f):
            print("NOT CALIBRATED:", getattr(self.tm, "why_not", lambda: "tags not found")())
            return None
        if self.namer == "first":
            tools = [dict(t, tool=self.want or "?") for t in find_tools(f, self.tm)]
            if len(tools) > 1:
                print(f"--namer first: {len(tools)} tools found, taking the biggest; put ONE tool on the tray")
                tools = tools[:1]
        elif self.namer is None:
            tools = detect_tools(f, self.tm)
        else:
            found = find_tools(f, self.tm)
            names = self.namer.name(f, found)
            tools = [dict(t, tool=n) for t, n in zip(found, names) if n]
        print("vision:", ", ".join(f"{t['tool']} at ({t['grasp_xy'][0]:.3f}, {t['grasp_xy'][1]:.3f})" for t in tools)
              or "no tools")
        from edgehand.vision import LAST_NOTES
        for note in LAST_NOTES:
            print("vision:", note)
        self.last = draw(f, self.tm, tools)
        cv2.imwrite(str(Path(tempfile.gettempdir()) / "edgehand_last_look.jpg"), self.last)   # for debugging
        if show:
            cv2.imshow("edgehand", self.last)
            cv2.waitKey(1)
        return tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--air", action="store_true", help="AIR_GAP_M = 0.03: every grasp 3 cm above the tools")
    ap.add_argument("--tool", help="one request for this tool class, no intent model")
    ap.add_argument("--look-only", action="store_true")
    ap.add_argument("--namer", choices=["gemma", "colour", "first"], default=config.VISION_NAMER)
    ap.add_argument("--show", action="store_true", help="window with the detections")
    ap.add_argument("--allow-defaults", action="store_true")
    ap.add_argument("--robot", choices=["kortex", "none"], default="kortex", help="none = vision only")
    ap.add_argument("--listen", action="store_true", help="take requests/release/stop over link.py too")
    a = ap.parse_args()
    if a.air:
        config.AIR_GAP_M = 0.03
    calibration_gate(a.allow_defaults)

    namer = "first" if a.namer == "first" else None
    if a.namer == "gemma":
        from edgehand.namer import GemmaNamer
        namer = GemmaNamer()
    intent = None
    if not a.tool and not a.look_only:
        if a.listen and a.namer != "gemma":
            intent = KeywordIntent()              # the assistant owns the GPU: no second LiteRT process here
            print("typed requests: keyword matching (tool class, colour word or role); Gemma not loaded")
        else:
            from edgehand.intent import GemmaIntent
            intent = GemmaIntent() if isinstance(namer, (str, type(None))) else _share_engine(namer)
    eyes = Eyes(namer)

    robot = None
    if a.robot == "kortex":
        from edgehand.kortex_robot import KortexRobot
        robot = KortexRobot()
    arm_state = None
    if a.listen:
        from edgehand import link
        arm_state = ArmState(lambda m: link.send(m, "127.0.0.1", config.PHASE_UDP_PORT))
    current = {"tool": None}

    def on_state(s):
        print("state:", s)
        if arm_state:
            arm_state.update(s, current["tool"])
    h = Handover(robot, log=print, state=on_state) if robot else None

    def relocate(tool):
        time.sleep(config.LOOK_SETTLE_S)
        eyes.want = tool
        tools = eyes.see(a.show) or []
        hit = [t for t in tools if t["tool"] == tool]
        return (hit[0]["grasp_xy"], hit[0]["handle_dir"]) if hit else None

    def on_interrupt():
        if h:
            h.stop()

    try:
        if robot:
            print("moving to the look pose")
            if not robot.look():
                print("can't reach the look pose; stopping")
                return
        time.sleep(config.LOOK_SETTLE_S)
        if a.look_only or not robot:
            while True:
                eyes.see(True)
                if cv2.waitKey(500) == 27:
                    break
            return
        requests = Requests(h)
        listener = None
        if a.listen:
            from edgehand import link
            listener = link.Listener(requests.on_event)
            print(f"listening for nodes on UDP {config.EVENT_UDP_PORT}")
        if not a.tool:
            threading.Thread(target=_read_stdin, args=(requests,), daemon=True).start()
        while True:
            if a.tool:
                tool = a.tool
            else:
                kind, val = requests.next()
                if kind == "text":
                    r = intent.parse(val)
                    print("intent:", r["reply"])
                    if r.get("intent") == "stop":
                        h.stop()
                        continue
                    if r.get("intent") == "release":
                        h.release()
                        continue
                    tool = r["tool"]
                    if tool is None:
                        continue
                else:
                    tool = val
            if listener is not None:
                new = set(listener.alive()) - requests.nodes
                if new:
                    print("nodes connected:", ", ".join(sorted(new)))
                    requests.nodes |= new
            if tool not in config.TOOLS:
                print(f"unknown tool class {tool!r}; classes: {list(config.TOOLS)}")
                if a.tool:
                    return
                continue
            found = relocate(tool)
            if found is None:
                print(f"can't see the {tool}")
                if a.tool:
                    return
                continue
            requests.start(tool)
            current["tool"] = tool
            try:
                ok, msg = h.run(found[0], found[1], relocate=lambda: relocate(tool))
            finally:
                requests.finish()
                if arm_state:
                    arm_state.update("idle", tool)
            print("handover:", "OK" if ok else "FAILED", msg)
            if a.tool:
                return
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping the arm")
        on_interrupt()
    finally:
        if robot:
            robot.close()


def _share_engine(namer):
    """One Gemma engine for naming and intent (a second copy would compete for the GPU)."""
    from edgehand import intent
    g = intent.GemmaIntent.__new__(intent.GemmaIntent)
    g.engine, g.system = namer.engine, intent.SYSTEM.format(tools=", ".join(config.TOOLS))
    return g


if __name__ == "__main__":
    main()
