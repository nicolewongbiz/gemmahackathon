"""The surgical assistant (laptop): Gemma 4 on LiteRT-LM decides which instrument the surgeon needs next.

On every phase change from the Jetson's video node (edgehand/surgical_feed.py), two Gemma calls:
  1. scene (image only, NOT told the phase): instruments in view, aneurysm visible?, clip applied?, and Gemma's own
     read of the phase, so its read is independent of the annotation and the agreement shown on screen means something;
  2. decision (text only, ~1.5 s): the scene report + the annotated phase + the tray (config.INSTRUMENTS) -> the
     instrument to hand over and why. The pick goes to run_kortex --listen as
{"type": "request", "tool", "source": "gemma-phase", ...}. With ASSISTANT_CONFIRM (default) it first SAYS the
recommendation out loud (macOS `say`, offline) and listens: "yes" fetches it, "no" skips, "no, the suction" fetches
that instead, so the arm never moves without a spoken yes. Press v to talk any time: request an instrument, stop or
release (a spoken stop is a convenience; the red E-stop is the stop).

    python -m edgehand.assistant                                   Jetson feed (SURGICAL_FEED_URL) + phase events
    python -m edgehand.assistant --local ~/Desktop/clippingdemo.mp4 ../../../data/phases/clippingdemo.csv    no Jetson
    --dry-run: decide and show, send nothing to the arm      --no-window: print only
The phase is the hand-annotated timeline (the phase model's weights aren't available); what Gemma sees in the
frame and what it chooses are live. Keys: v talk, r ask again, q quit.
"""
import argparse, tempfile, threading, time
from pathlib import Path
import cv2
import numpy as np
from edgehand import config, gemma, link

COLOURS = {"green": (60, 170, 60), "orange": (40, 140, 245), "red": (50, 50, 220), "yellow": (40, 210, 230),
           "blue": (220, 150, 60)}

SYSTEM = """You are the scrub-nurse assistant in a brain aneurysm clipping operation. You control a robot arm that
hands the surgeon instruments from a tray. Instruments on the tray (colour: instrument - when it is used):
{tray}
You see the current frame from the operating microscope, and the video model tells you the surgical phase."""

SCENE_SYSTEM = """You analyse frames from an operating microscope during brain aneurysm clipping surgery."""

SCENE = """Look at this frame.
1. Count the surgical instruments visible and name them briefly.
2. Is the aneurysm (the bulging dome on the artery) visible?
3. Is an aneurysm clip already applied?
4. Which phase is this most likely: {phases}?
Reply ONLY with JSON: {{"visible": <number>, "seen": "<short list>", "aneurysm_visible": <true|false>,
"clip_applied": <true|false>, "phase_read": "<one of the phases>"}}"""

DECIDE = """The phase model says the surgical phase is: {phase} (previous phase: {prev}).
What you saw in the current frame: {visible} instrument(s) ({seen}); aneurysm visible: {aneurysm}; clip already
applied: {clip}; your own read of the phase: {phase_read}.
Choose the ONE instrument from the tray the surgeon most likely needs next.
Reply ONLY with JSON: {{"colour": "<{colours}>", "reason": "<at most 12 words>"}}"""

VOICE = """Surgical phase now: {phase}. The surgeon said: "{heard}"
If they asked for an instrument (by colour, name or the job it does), intent is request and colour is that
instrument's colour; otherwise intent is none. Reply ONLY with JSON:
{{"intent": "<request|none>", "colour": "<{colours} or null>", "reason": "<at most 8 words>"}}"""


def by_colour():
    return {c: tool for tool, (c, _, _) in config.INSTRUMENTS.items()}


def tray_text():
    return "\n".join(f"  {c}: {role} - {use}" for c, role, use in config.INSTRUMENTS.values())


def parse_decision(d):
    """Gemma's JSON -> {"tool", "colour", "visible", "seen", "reason"}; tool None if the colour isn't on the tray."""
    d = d or {}
    colour = str(d.get("colour") or "").strip().lower()
    tool = by_colour().get(colour)
    try:
        visible = int(d.get("visible"))
    except (TypeError, ValueError):
        visible = None
    seen = d.get("seen", "")
    seen = ", ".join(map(str, seen)) if isinstance(seen, list) else str(seen)
    return {"tool": tool, "colour": colour if tool else None, "visible": visible,
            "seen": seen or "nothing", "reason": str(d.get("reason", ""))}


def _bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return True if s in ("true", "yes", "y", "1") else False if s in ("false", "no", "n", "0") else None


def parse_scene(d):
    """Gemma's scene JSON -> {"visible", "seen", "aneurysm", "clip", "phase_read"} (None where it didn't say)."""
    d = d or {}
    try:
        visible = int(d.get("visible"))
    except (TypeError, ValueError):
        visible = None
    seen = d.get("seen", "")
    seen = ", ".join(map(str, seen)) if isinstance(seen, list) else str(seen)
    read = str(d.get("phase_read") or "").strip().lower()
    read = next((p for p in config.SURGICAL_PHASES if p.lower() == read), None)
    return {"visible": visible, "seen": seen or "nothing", "aneurysm": _bool(d.get("aneurysm_visible")),
            "clip": _bool(d.get("clip_applied")), "phase_read": read}


def parse_voice(d):
    d = d or {}
    intent = d.get("intent") if d.get("intent") in ("request", "stop", "release", "none") else "none"
    colour = str(d.get("colour") or "").strip().lower()
    tool = by_colour().get(colour)
    if intent == "request" and tool is None:
        intent = "none"
    return {"intent": intent, "tool": tool, "colour": colour if tool else None, "heard": str(d.get("heard", "")),
            "reason": str(d.get("reason", ""))}


class Assistant:
    def __init__(self, send=True, arm_host="127.0.0.1", confirm=config.ASSISTANT_CONFIRM):
        self.send, self.arm_host, self.confirm = send, arm_host, confirm
        self.engine = gemma.make_engine(config.GEMMA_MODEL, vision=True, audio=True, gpu=config.ASSISTANT_GEMMA_GPU)
        self.system = SYSTEM.format(tray=tray_text())
        self.colours = "|".join(c for c, _, _ in config.INSTRUMENTS.values())
        self.frame = None                                   # latest surgical frame (BGR)
        self.phase = {"phase": None, "prev": None, "clip_t": 0.0, "annotated": True, "seq": None}
        self.decision, self.status, self.busy = None, "waiting for the video node", threading.Lock()
        self.decided_seq = None
        self.agree, self.compared = 0, 0                    # Gemma's phase read vs the annotation
        # run_kortex reports {"type": "arm_state", "state": busy|holding|idle, "tool"} here; no reports = idle
        self.arm = {"state": "idle", "tool": None}
        self.pending = None                                 # (decision, phase) waiting for the arm to be free
        self.dir = Path(tempfile.gettempdir()) / "edgehand_assistant"
        self.dir.mkdir(exist_ok=True)

    def on_phase(self, msg):
        if msg.get("type") == "arm_state":
            self.on_arm_state(msg)
            return
        if msg.get("type") not in ("phase", "phase_status") or msg.get("phase") is None:
            return
        self.phase.update({k: msg.get(k) for k in ("phase", "prev", "clip_t", "annotated", "seq")})
        self._maybe_decide()

    def on_arm_state(self, msg):
        was = self.arm["state"]
        self.arm = {"state": msg.get("state") or "idle", "tool": msg.get("tool")}
        if self.arm["state"] == "idle" and was != "idle" and self.pending:
            r, ph = self.pending
            self.pending = None
            stale = ph["phase"] != self.phase["phase"] or r["tool"] == msg.get("tool")
            if not stale and self.busy.acquire(blocking=False):
                def run():
                    try:
                        self._confirm(r, ph)
                    finally:
                        self.busy.release()
                        self._maybe_decide()
                threading.Thread(target=run, daemon=True).start()

    def _maybe_decide(self):
        """Decide for the newest phase if it hasn't been decided yet and Gemma is free (else _decide picks it up)."""
        if self.phase["seq"] != self.decided_seq and self.busy.acquire(blocking=False):
            self.decided_seq = self.phase["seq"]
            threading.Thread(target=self._decide, args=(dict(self.phase),), daemon=True).start()

    def _frame_path(self):
        if self.frame is None:
            return None
        h, w = self.frame.shape[:2]
        s = config.ASSISTANT_IMAGE_PX / max(h, w)
        p = self.dir / "frame.jpg"
        cv2.imwrite(str(p), cv2.resize(self.frame, None, fx=s, fy=s) if s < 1 else self.frame)
        return p

    def _decide(self, ph):
        try:
            self.status = "Gemma is looking at the frame..."
            img = self._frame_path()
            t0 = time.time()
            sc = parse_scene(gemma.ask_json(self.engine, SCENE_SYSTEM,
                                            SCENE.format(phases=", ".join(config.SURGICAL_PHASES)),
                                            images=[img] if img else ()))
            t1 = time.time()
            if sc["phase_read"]:
                self.compared += 1
                self.agree += sc["phase_read"].lower() == str(ph["phase"]).lower()
            yn = lambda v: "unclear" if v is None else "yes" if v else "no"
            self.status = "Gemma is choosing the instrument..."
            r = parse_decision(gemma.ask_json(self.engine, self.system, DECIDE.format(
                phase=ph["phase"], prev=ph["prev"] or "none", visible=sc["visible"] if sc["visible"] is not None
                else "?", seen=sc["seen"], aneurysm=yn(sc["aneurysm"]), clip=yn(sc["clip"]),
                phase_read=sc["phase_read"] or "unclear", colours=self.colours)))
            r.update(visible=sc["visible"], seen=sc["seen"], scene=sc, phase=ph["phase"],
                     seconds=time.time() - t0, scene_seconds=t1 - t0)
            self.decision = r
            print(f"[{ph['clip_t']:5.1f} s] {ph['phase']}: Gemma sees {sc['visible']} ({sc['seen']}), aneurysm "
                  f"{yn(sc['aneurysm'])}, clip {yn(sc['clip'])}, reads '{sc['phase_read']}' -> {r['colour']} "
                  f"{config.INSTRUMENTS[r['tool']][1] if r['tool'] else '-'}: {r['reason']} "
                  f"({r['scene_seconds']:.1f} + {r['seconds'] - r['scene_seconds']:.1f} s; agrees "
                  f"{self.agree}/{self.compared})", flush=True)
            if r["tool"] and self.confirm and self.arm["state"] != "idle":
                busy_with = config.INSTRUMENTS.get(self.arm["tool"], ("", "instrument", ""))
                first = self.pending is None
                self.pending = (r, ph)                      # newest pick replaces an older one
                self.status = (f"arm is handing over the {busy_with[0]} {busy_with[1]}; I'll ask about the "
                               f"{config.INSTRUMENTS[r['tool']][1]} when it's free")
                if first:
                    from edgehand.voice_node import speak
                    speak(f"The arm is still handing over the {busy_with[1]}.")
            elif r["tool"] and self.confirm:
                self._confirm(r, ph)
            elif r["tool"]:
                self._request(r["tool"], "gemma-phase", phase=ph["phase"], reason=r["reason"])
            else:
                self.status = "Gemma didn't pick an instrument on the tray"
        finally:
            self.busy.release()
            self._maybe_decide()                # a phase that changed while Gemma was busy is next

    def _confirm(self, r, ph):
        """Say the recommendation, listen, act on the answer. Runs inside the busy lock (one Gemma job at a time)."""
        from edgehand.voice_node import record, rule_confirm, rule_intent, speak, transcribe
        colour, role, _ = config.INSTRUMENTS[r["tool"]]
        question = f"{ph['phase']}. Shall I bring the {colour} {role}?"
        self.status = f'asking: "{question}"  (yes / no / another instrument)'
        speak(question)
        wav = self.dir / "answer.wav"
        self.status = f"listening for your answer ({config.CONFIRM_SECONDS:.0f} s)..."
        record(wav, config.CONFIRM_SECONDS)
        heard = transcribe(self.engine, wav)
        fixed, answer = rule_intent(heard), rule_confirm(heard)
        print(f"  confirm: heard {heard!r}", flush=True)
        if fixed:                                           # stop / release, as always
            if self.send:
                link.send({"type": fixed, "source": "voice"}, self.arm_host)
            self.status = f"heard {heard!r}: {fixed}"
            return
        named = None
        if not (answer and len(heard.split()) <= 3):        # more than a yes/no: did they name an instrument?
            v = parse_voice(dict(gemma.ask_json(self.engine, self.system, VOICE.format(
                phase=ph["phase"], heard=heard, colours=self.colours)) or {}, heard=heard))
            named = v["tool"] if v["intent"] == "request" else None
        if named and named != r["tool"]:
            self._request(named, "voice", phase=ph["phase"], reason=f"surgeon chose it over the {role}")
            speak(f"Bringing the {config.INSTRUMENTS[named][1]} instead.")
        elif answer == "yes" or named == r["tool"]:
            self._request(r["tool"], "gemma-phase", phase=ph["phase"], reason=r["reason"])
            speak(f"Bringing the {role}.")
        elif answer == "no":
            self.status = f"heard {heard!r}: not bringing the {role}"
            speak("Okay.")
        else:
            self.status = f"heard {heard!r}: no clear yes; press v to ask for an instrument"

    def _request(self, tool, source, **extra):
        role = config.INSTRUMENTS[tool][1]
        if self.send:
            link.send(dict({"type": "request", "tool": tool, "source": source}, **extra), self.arm_host)
            self.status = f"asked the arm for the {config.INSTRUMENTS[tool][0]} {role}"
        else:
            self.status = f"(dry run) would ask the arm for the {config.INSTRUMENTS[tool][0]} {role}"

    def talk(self):
        """Push-to-talk: record, Gemma maps it with the phase as context."""
        if not self.busy.acquire(blocking=False):
            self.status = "busy; try again in a second"
            return
        def run():
            try:
                from edgehand.voice_node import record
                wav = self.dir / "voice.wav"
                self.status = f"listening for {config.VOICE_SECONDS:.0f} s..."
                record(wav)
                from edgehand.voice_node import rule_intent, transcribe
                self.status = "Gemma is listening..."
                heard = transcribe(self.engine, wav)        # neutral prompt: the phase can't bias what it hears
                fixed = rule_intent(heard)                  # stop / release: fixed keywords, not Gemma's judgement
                if fixed:
                    v = {"intent": fixed, "tool": None, "colour": None, "heard": heard, "reason": "keyword"}
                else:
                    v = parse_voice(dict(gemma.ask_json(self.engine, self.system, VOICE.format(
                        phase=self.phase["phase"] or "unknown", heard=heard, colours=self.colours)) or {}, heard=heard))
                    if v["intent"] != "request":            # stop / release only ever come from the keywords
                        v["intent"] = "none"
                print(f"voice: heard {v['heard']!r} -> {v['intent']} {v['colour'] or ''}", flush=True)
                if v["intent"] == "request":
                    self._request(v["tool"], "voice", reason=v["reason"])
                elif v["intent"] in ("stop", "release"):
                    if self.send:
                        link.send({"type": v["intent"], "source": "voice"}, self.arm_host)
                    self.status = f"heard {v['heard']!r}: {v['intent']}"
                else:
                    self.status = f"heard {v['heard']!r}: nothing to do"
            finally:
                self.busy.release()
        threading.Thread(target=run, daemon=True).start()

    def render(self, jetson_alive):
        W, H, VW, VH = 1280, 720, 960, 540
        out = np.full((H, W, 3), 18, np.uint8)
        if self.frame is not None:
            out[:VH, :VW] = cv2.resize(self.frame, (VW, VH))
        else:
            cv2.putText(out, "waiting for the Jetson's video feed...", (200, 270), 0, 0.9, (200, 200, 200), 2)
        put = lambda txt, xy, s=0.6, c=(235, 235, 235), t=1: cv2.putText(out, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, s, c, t)
        put("Instrument tray", (VW + 20, 40), 0.75, t=2)
        chosen = self.decision["tool"] if self.decision else None
        for i, (tool, (colour, role, _)) in enumerate(config.INSTRUMENTS.items()):
            y = 75 + i * 110
            cv2.rectangle(out, (VW + 20, y), (VW + 80, y + 60), COLOURS.get(colour, (128, 128, 128)), -1)
            if tool == chosen:
                cv2.rectangle(out, (VW + 12, y - 8), (W - 12, y + 68), (255, 255, 255), 3)
            put(role, (VW + 95, y + 28), 0.65, t=2)
            put(f"{colour} screwdriver", (VW + 95, y + 52), 0.5, (170, 170, 170))
        ph = self.phase
        put(f"Phase: {ph['phase'] or '-'}", (20, VH + 32), 0.85, t=2)
        put(f"t = {ph['clip_t']:.0f} s", (420, VH + 32), 0.5, (170, 170, 170))
        if self.decision:
            d = self.decision
            sc = d.get("scene") or {}
            yn = lambda v: "?" if v is None else "yes" if v else "no"
            read = sc.get("phase_read")
            agrees = read is not None and read.lower() == str(d.get("phase")).lower()
            put(f"Gemma's read: {read or '?'}" + (" (agrees)" if agrees else " (differs)" if read else "")
                + f"   [{self.agree}/{self.compared} agree]   aneurysm visible: {yn(sc.get('aneurysm'))}"
                f"   clip applied: {yn(sc.get('clip'))}", (20, VH + 58), 0.5,
                (120, 255, 120) if agrees else (120, 200, 255))
            put(f"Gemma sees {d['visible'] if d['visible'] is not None else '?'} instrument(s): {d['seen']}"[:150],
                (20, VH + 82), 0.5)
            role = config.INSTRUMENTS[d["tool"]][1] if d["tool"] else "-"
            put(f"Gemma -> {d['colour'] or '-'} {role}", (20, VH + 110), 0.7, (120, 255, 120), 2)
            put(d["reason"][:120], (20, VH + 134), 0.5, (120, 255, 120))
        put(self.status[:150], (20, VH + 160), 0.48, (200, 200, 120))
        put(f"arm: {self.arm['state']}" + (f" ({config.INSTRUMENTS.get(self.arm['tool'], ('', self.arm['tool'], ''))[1]})"
            if self.arm["tool"] else ""), (W - 300, H - 8), 0.42, (150, 150, 150))
        put(f"Gemma 4 E4B on LiteRT-LM (laptop, {'GPU' if config.ASSISTANT_GEMMA_GPU else 'CPU'})   video: "
            f"{'Jetson' if jetson_alive else 'local / waiting'}   keys: v talk, r ask again, q quit",
            (20, H - 8), 0.42, (150, 150, 150))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs=2, metavar=("VIDEO", "PHASES"), help="run the video node here (no Jetson)")
    ap.add_argument("--feed", default=config.SURGICAL_FEED_URL)
    ap.add_argument("--dry-run", action="store_true", help="decide and show, send nothing to the arm")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-confirm", action="store_true", help="fetch Gemma's pick without asking")
    ap.add_argument("--speed", type=float, default=1.0, help="with --local")
    ap.add_argument("--loop", action="store_true", help="with --local")
    a = ap.parse_args()
    asst = Assistant(send=not a.dry_run, confirm=not a.no_confirm and config.ASSISTANT_CONFIRM)
    listener = None
    if a.local:
        from edgehand.surgical_feed import Feed
        feed = Feed(a.local[0], a.local[1],
                    on_frame=lambda b: setattr(asst, "frame", cv2.imdecode(np.frombuffer(b, np.uint8), 1)),
                    on_phase=asst.on_phase, speed=a.speed, loop=a.loop)
        done = lambda: not feed._run
    else:
        listener = link.Listener(asst.on_phase, port=config.PHASE_UDP_PORT)
        from edgehand.camera import StreamCamera
        cam = StreamCamera(a.feed, on_frame=lambda f: setattr(asst, "frame", f))
        done = lambda: False
    print("assistant ready:", "local feed" if a.local else f"Jetson feed {a.feed}, phase events on :{config.PHASE_UDP_PORT}",
          "(dry run)" if a.dry_run else "-> run_kortex on :%d" % config.EVENT_UDP_PORT, flush=True)
    try:
        while not done() or asst.busy.locked():
            alive = listener is not None and "jetson" in listener.alive()
            if a.no_window:
                time.sleep(0.2)
                continue
            cv2.imshow("EdgeHand surgical assistant", asst.render(alive))
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("v"):
                asst.talk()
            if k == ord("r") and asst.phase["phase"]:          # ask Gemma again for the current phase
                asst.on_phase(dict(asst.phase, type="phase", seq=time.time()))
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
