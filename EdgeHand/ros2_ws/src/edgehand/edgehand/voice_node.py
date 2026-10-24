"""Voice on the laptop: push-to-talk mic -> Gemma 4 (native audio input, LiteRT-LM) -> a request to the arm over
edgehand/link.py. run_kortex --listen picks it up like any other node. Offline.

    python -m edgehand.voice_node                    ENTER, speak ("hand me the clip applier", "stop", "let go")
    python -m edgehand.voice_node --text             type instead of speaking (testing)
    python -m edgehand.voice_node --list-mics

Three steps, so what Gemma expects can't change what it hears:
  1. transcribe (audio, neutral prompt: no instruments, no phase) -> the words;
  2. stop / release by fixed keywords on those words (rule_intent), never by Gemma's judgement;
  3. otherwise Gemma maps the words to {"intent": "request" | "none", "tool", "reply"}, tool whitelisted.
With instruments and the phase in the same prompt as the audio, Gemma heard "stop" as "clip applier".
Gemma never produces coordinates. A spoken "stop" is a convenience: the red E-stop is the stop.
"""
import argparse, re, subprocess, tempfile
from pathlib import Path
from edgehand import config, gemma, link

SYSTEM = """You are the voice of a robot arm that hands instruments to a surgeon.
Instruments on the tray: {tools}.
Map what the surgeon says to ONE intent:
  request  - they want an instrument (also when they describe the job it does); give the tool from the list
  stop     - stop, wait, hold, freeze
  release  - let go, release, I've got it
  none     - anything else
Reply ONLY with JSON: {{"intent": "...", "tool": "<tool or null>", "heard": "<their words>", "reply": "<very short spoken reply>"}}"""


TRANSCRIBE_SYSTEM = "You transcribe short spoken sentences exactly, word for word."
TRANSCRIBE = 'Transcribe exactly what the person in the audio said. Reply ONLY with JSON: {"heard": "<their exact words>"}'
STOP_WORDS = r"\b(stop|wait|hold|freeze|halt|pause|abort)\b"
RELEASE_WORDS = r"\b(let go|release|got it|i'?ve got it|drop it|open (the )?gripper|take it)\b"


def transcribe(engine, wav):
    d = gemma.ask_json(engine, TRANSCRIBE_SYSTEM, TRANSCRIBE, audio=wav) or {}
    return str(d.get("heard", "")).strip()


def rule_intent(text):
    """'stop' / 'release' from fixed keywords (stop wins), else None: these never depend on a model's judgement."""
    t = text.lower()
    if re.search(STOP_WORDS, t):
        return "stop"
    if re.search(RELEASE_WORDS, t):
        return "release"
    return None


def mics():
    """[(ffmpeg avfoundation index, name)] of audio inputs."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True).stderr
    audio = out.split("audio devices:")[-1] if "audio devices:" in out else ""
    return [(int(i), n.strip()) for i, n in re.findall(r"\[(\d+)\] (.+)", audio)]


def mic_index():
    for i, n in mics():
        if config.VOICE_MIC_NAME.lower() in n.lower():
            return i
    raise SystemExit(f"no mic named like {config.VOICE_MIC_NAME!r}: {mics()} (set VOICE_MIC_NAME)")


def record(path, seconds=config.VOICE_SECONDS):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "avfoundation", "-i", f":{mic_index()}", "-t", str(seconds),
                    "-ac", "1", "-ar", "16000", str(path)], check=True)


def parse(d):
    d = d or {}
    intent = d.get("intent") if d.get("intent") in ("request", "stop", "release", "none") else "none"
    tool = d.get("tool").strip().lower() if isinstance(d.get("tool"), str) else None
    if tool not in config.TOOLS:
        tool = None
    if intent == "request" and tool is None:
        intent = "none"
    return {"intent": intent, "tool": tool, "heard": str(d.get("heard", "")), "reply": str(d.get("reply", ""))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="type requests instead of speaking")
    ap.add_argument("--list-mics", action="store_true")
    ap.add_argument("--host", default="127.0.0.1", help="the laptop running the arm")
    a = ap.parse_args()
    if a.list_mics:
        print(mics(), "-> using", config.VOICE_MIC_NAME)
        return
    engine = gemma.make_engine(config.GEMMA_MODEL, audio=not a.text)
    system = SYSTEM.format(tools=", ".join(f"{t} ({config.TOOL_DESCRIPTIONS.get(t, t)})" for t in config.TOOLS))
    beat = link.Heartbeat("voice", a.host)
    wav = Path(tempfile.gettempdir()) / "edgehand_voice.wav"
    print("ready:", "type a request" if a.text else f"ENTER, then speak for {config.VOICE_SECONDS:.0f} s", "(Ctrl-C quits)")
    try:
        while True:
            if a.text:
                said = input("> ").strip()
            else:
                input("ENTER to talk...")
                record(wav)
                said = transcribe(engine, wav)
            fixed = rule_intent(said)
            r = ({"intent": fixed, "tool": None, "heard": said, "reply": "Stopping." if fixed == "stop" else "Releasing."}
                 if fixed else parse(dict(gemma.ask_json(engine, system, f'The surgeon said: "{said}"') or {}, heard=said)))
            if not fixed and r["intent"] != "request":       # stop / release only ever come from the keywords
                r["intent"] = "none"
            print(f"  heard {r['heard']!r} -> {r['intent']}" + (f" {r['tool']}" if r["tool"] else ""))
            if r["intent"] == "request":
                link.send({"type": "request", "tool": r["tool"], "source": "voice"}, a.host)
            elif r["intent"] in ("stop", "release"):
                link.send({"type": r["intent"], "source": "voice"}, a.host)
            if config.VOICE_SPEAK and r["reply"]:
                subprocess.run(["say", r["reply"]], check=False)
    except KeyboardInterrupt:
        pass
    finally:
        beat.close()


if __name__ == "__main__":
    main()
