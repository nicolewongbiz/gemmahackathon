"""Raspberry Pi 5 voice front end. Gemma 4 E2B on LiteRT-LM, fully offline.

Push-to-talk: press ENTER (or wire a button to GPIO17 and set USE_GPIO=True).
Audio goes straight into Gemma E2B (native audio input) -> JSON intent -> laptop.
Laptop replies are polled from /say and spoken with espeak-ng.

    python voice_agent.py --laptop 192.168.1.11
    python voice_agent.py --text            # type commands instead (debug)
    python voice_agent.py --stt vosk        # fallback if E2B audio is too slow
"""
import argparse, json, re, subprocess, threading, time
from pathlib import Path
import numpy as np
import requests

HERE = Path(__file__).parent
MODEL = str(HERE / "models" / "gemma-4-E2B-it.litertlm")
RATE, SECONDS = 16000, 4
USE_GPIO = False

SYSTEM = """You control a robot arm by voice. Map what the user says to ONE intent.
Intents: empty (table is clear / take empty picture), teach (watch me / I'll show you),
done (that's it / finished showing), go (start / do it / sort these), stop (stop / halt / wait),
correct (any change to the sorting rule, e.g. "red things go in B"), chat (anything else).
Reply ONLY with JSON: {"intent": "...", "text": "<the user's words>", "reply": "<very short spoken reply or empty>"}"""

_engine = None


def engine():
    """LiteRT-LM engine. Verify kwargs with: python -c "import litert_lm; help(litert_lm.Engine)" """
    global _engine
    if _engine is None:
        import litert_lm
        kw = {}
        B = getattr(litert_lm, "Backend", None)
        if B is not None and hasattr(B, "CPU"):
            kw = {"backend": B.CPU, "audio_backend": B.CPU}
        try:
            _engine = litert_lm.Engine(MODEL, **kw)
        except TypeError:
            _engine = litert_lm.Engine(MODEL)
    return _engine


def gemma(content):
    with engine().create_conversation(messages=[{"role": "system", "content": SYSTEM}]) as conv:
        r = conv.send_message({"role": "user", "content": content})
    txt = r["content"][0]["text"]
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {"intent": "chat", "text": txt, "reply": ""}


def record():
    import sounddevice as sd, soundfile as sf
    print("... listening")
    a = sd.rec(int(SECONDS * RATE), samplerate=RATE, channels=1, dtype="int16")
    sd.wait()
    p = HERE / "_cmd.wav"
    sf.write(p, a, RATE)
    return str(p)


_vosk = None
def vosk_stt(wav):
    global _vosk
    import vosk, wave
    if _vosk is None:
        _vosk = vosk.Model(str(HERE / "models" / "vosk-model-small-en-us-0.15"))
    wf = wave.open(wav)
    rec = vosk.KaldiRecognizer(_vosk, wf.getframerate())
    while data := wf.readframes(4000):
        rec.AcceptWaveform(data)
    return json.loads(rec.FinalResult())["text"]


def speak(text):
    if text:
        print("[say]", text)
        subprocess.run(["espeak-ng", "-s", "165", text], check=False)


def poll_say(laptop):
    while True:
        try:
            for t in requests.get(f"{laptop}/say", timeout=3).json()["say"]:
                speak(t)
        except requests.RequestException:
            pass
        time.sleep(0.5)


def wait_trigger():
    if USE_GPIO:
        from gpiozero import Button
        Button(17).wait_for_press()
    else:
        input("\n[ENTER] to talk ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop", default="192.168.1.11")
    ap.add_argument("--stt", choices=["gemma", "vosk"], default="gemma")
    ap.add_argument("--text", action="store_true")
    a = ap.parse_args()
    laptop = f"http://{a.laptop}:8000"
    threading.Thread(target=poll_say, args=(laptop,), daemon=True).start()
    print("Loading Gemma E2B (LiteRT-LM)..."); engine(); speak("Ready.")
    while True:
        if a.text:
            words = input("\ncommand> ")
            content = [{"type": "text", "text": words}]
        else:
            wait_trigger()
            wav = record()
            if a.stt == "vosk":
                content = [{"type": "text", "text": vosk_stt(wav)}]
            else:
                content = [{"type": "audio", "path": wav},
                           {"type": "text", "text": "Classify this spoken command."}]
        t0 = time.time()
        out = gemma(content)
        print(f"[gemma {time.time() - t0:.1f}s] {out}")
        speak(out.get("reply", ""))
        if out.get("intent") in {"empty", "teach", "done", "go", "stop", "correct"}:
            try:
                requests.post(f"{laptop}/intent", json={"intent": out["intent"], "text": out.get("text", "")}, timeout=60)
            except requests.RequestException as e:
                speak("I can't reach the laptop."); print(e)


if __name__ == "__main__":
    main()
