"""Laptop app, the swarm's brain + hands: Gemma planner + Kinova + HTTP for the Pi.

Perception (wrist camera, AprilTags, objects) runs on the Jetson (EYES = "jetson") or here (EYES = "local").
    uvicorn main:app --host 0.0.0.0 --port 8000
Keyboard fallback (no Pi): POST /intent from curl, or use the /docs page.
"""
import queue, threading, time
import cv2
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import config
from perception.eyes import make_eyes
from planner.brain import Brain
from robot.arm import make_arm, pick_and_place

config.DATA.mkdir(exist_ok=True)
app = FastAPI(title="Show Me Once")
eyes, brain, arm = make_eyes(), Brain(), make_arm()
say_q: "queue.Queue[str]" = queue.Queue()
state = {"mode": "idle", "stop": False, "pi_seen": 0.0}


def say(text):
    print("[say]", text)
    say_q.put(text)


def run_sort():
    state["mode"], state["stop"] = "sorting", False
    for _ in range(12):
        if state["stop"]:
            break
        arm.look()
        try:
            s = eyes.scene()
        except requests.RequestException as e:
            print("[eyes]", e); say("I can't reach the Jetson."); break
        if not s["calibrated"]:
            say("I can't see all four table tags."); break
        objs = s["objects"]
        if not objs:
            say("All done."); break
        cv2.imwrite(str(config.DATA / "last_marked.jpg"), s["marked"])
        t0 = time.time()
        step = brain.plan(s["marked"], list(objs), s["labels"])
        print(f"[plan] {step}  ({time.time() - t0:.1f}s)")
        if step.get("object") is None:
            say(step.get("say", "Nothing left that fits.")); break
        say(f"{step.get('label', 'object')}, bin {step['bin']}")
        try:
            if not pick_and_place(arm, objs[step["object"]], s["bins"][step["bin"]],
                                  should_stop=lambda: state["stop"]):
                break
        except ValueError as e:
            print("[arm]", e); say("That's outside my safe area."); break
    if not state["stop"]:
        arm.home()
    state["mode"] = "idle"


def start_teach():
    arm.look()
    eyes.teach_start()


class Intent(BaseModel):
    intent: str          # empty | teach | done | go | stop | correct
    text: str = ""


@app.post("/intent")
def intent(i: Intent):
    try:
        if i.intent == "empty":
            arm.look(); eyes.snapshot_empty(); say("Saved the empty table.")
        elif i.intent == "teach":
            state["mode"] = "teaching"
            threading.Thread(target=start_teach, daemon=True).start()
            say("Watching. Show me.")
        elif i.intent == "done":
            state["mode"] = "idle"
            frames = eyes.teach_stop()
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
    except requests.RequestException as e:
        print("[eyes]", e); say("I can't reach the Jetson.")
    return {"ok": True, "mode": state["mode"], "rule": brain.rule}


@app.get("/say")
def get_say():
    """Pi polls this and speaks whatever is queued."""
    state["pi_seen"] = time.time()
    out = []
    while not say_q.empty():
        out.append(say_q.get())
    return {"say": out}


@app.get("/status")
def status():
    e = eyes.status()
    return {"mode": state["mode"], "rule": brain.rule, "calibrated": e["calibrated"],
            "nodes": {"pi": time.time() - state["pi_seen"] < 3,
                      "eyes": f"{config.EYES} ({'online' if e['online'] else 'OFFLINE'})",
                      "arm": "mock" if config.USE_MOCK_ARM else "kinova"}}
