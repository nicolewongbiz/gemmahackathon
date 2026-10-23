"""Laptop app: Kinova wrist camera + AprilTags + Gemma planner + Kinova + HTTP for the Pi.

    uvicorn main:app --host 0.0.0.0 --port 8000
Keyboard fallback (no Pi): POST /intent from curl, or use the /docs page.
"""
import queue, threading, time
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import config
from perception.camera import make_camera
from perception.calibrate import TableMap
from perception.objects import ObjectFinder, KeyframeRecorder, draw_marks
from planner.brain import Brain
from robot.arm import make_arm, pick_and_place

config.DATA.mkdir(exist_ok=True)
app = FastAPI(title="Show Me Once")
cam, tmap, finder, brain, arm = make_camera(), TableMap(), ObjectFinder(), Brain(), make_arm()
say_q: "queue.Queue[str]" = queue.Queue()
state = {"mode": "idle", "stop": False}
recorder = None


def say(text):
    print("[say]", text)
    say_q.put(text)


def run_sort():
    state["mode"], state["stop"] = "sorting", False
    for _ in range(12):
        if state["stop"]:
            break
        arm.look()
        f = cam.read()
        if not tmap.update(f):
            say("I can't see all four table tags."); break
        found = [p for p in finder.find(f, tmap.tag_mask(f)).values() if tmap.pickable(*p)]
        objs = {i + 1: p for i, p in enumerate(found)}
        if not objs:
            say("All done."); break
        marked = draw_marks(f, objs, tmap.bins_px)
        cv2.imwrite(str(config.DATA / "last_marked.jpg"), marked)
        t0 = time.time()
        step = brain.plan(marked, list(objs))
        print(f"[plan] {step}  ({time.time() - t0:.1f}s)")
        if step.get("object") is None:
            say(step.get("say", "Nothing left that fits.")); break
        say(f"{step.get('label', 'object')}, bin {step['bin']}")
        u, v = objs[step["object"]]
        try:
            if not pick_and_place(arm, tmap.px_to_table(u, v), tmap.bin_xy(step["bin"]),
                                  should_stop=lambda: state["stop"]):
                break
        except ValueError as e:
            print("[arm]", e); say("That's outside my safe area."); break
    if not state["stop"]:
        arm.home()
    state["mode"] = "idle"


def teach_loop():
    global recorder
    recorder = KeyframeRecorder()
    arm.look()
    while state["mode"] == "teaching":
        recorder.push(cam.read())
        time.sleep(1 / 15)


class Intent(BaseModel):
    intent: str          # empty | teach | done | go | stop | correct
    text: str = ""


@app.post("/intent")
def intent(i: Intent):
    if i.intent == "empty":
        arm.look(); finder.snapshot_empty(cam.read()); say("Saved the empty table.")
    elif i.intent == "teach":
        state["mode"] = "teaching"
        threading.Thread(target=teach_loop, daemon=True).start()
        say("Watching. Show me.")
    elif i.intent == "done":
        state["mode"] = "idle"; time.sleep(0.2)
        frames = recorder.result() if recorder else []
        if not frames:
            say("I didn't catch any moves.")
        else:
            say("Thinking about what you did.")
            say(brain.teach(frames))
    elif i.intent == "go":
        if state["mode"] != "sorting":
            threading.Thread(target=run_sort, daemon=True).start()
    elif i.intent == "stop":
        state["stop"] = True; arm.stop(); say("Stopping.")
    elif i.intent == "correct":
        say(brain.correct(i.text))
    return {"ok": True, "mode": state["mode"], "rule": brain.rule}


@app.get("/say")
def get_say():
    """Pi polls this and speaks whatever is queued."""
    out = []
    while not say_q.empty():
        out.append(say_q.get())
    return {"say": out}


@app.get("/status")
def status():
    return {"mode": state["mode"], "rule": brain.rule, "calibrated": tmap.H is not None}
