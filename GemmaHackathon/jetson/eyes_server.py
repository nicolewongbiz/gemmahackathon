"""Jetson Orin Nano Super: the swarm's eyes.

Reads the Kinova wrist camera (RTSP), finds tags and objects, and has its own Gemma name each
numbered object. The laptop moves the arm to LOOK_POSE, then asks for a scene here.

    cd GemmaHackathon && .venv/bin/uvicorn jetson.eyes_server:app --host 0.0.0.0 --port 8001
"""
import threading, time
from fastapi import FastAPI
from pydantic import BaseModel
import config
from perception.eyes import LocalEyes, encode
from planner import gemma_vision

app = FastAPI(title="Show Me Once: eyes")
eyes = LocalEyes()


def warm_up():
    """The first Ollama call loads the model (~45 s). Do it now, not mid-demo."""
    if not config.EYES_LABELS:
        return
    t0 = time.time()
    try:
        gemma_vision.ask("Say OK.", backend=config.EYES_GEMMA_BACKEND, model=config.EYES_GEMMA_MODEL)
        print(f"[eyes] Gemma warm ({time.time() - t0:.1f}s)")
    except Exception as e:
        print("[eyes] Gemma warm-up failed:", e)


threading.Thread(target=warm_up, daemon=True).start()


class SceneReq(BaseModel):
    label: bool = True


@app.post("/empty")
def empty():
    eyes.snapshot_empty()
    return {"ok": True}


@app.post("/scene")
def scene(req: SceneReq):
    t0 = time.time()
    s = eyes.scene(req.label)
    print(f"[scene] {len(s['objects'])} objects {s['labels']}  ({time.time() - t0:.1f}s)")
    return {**s, "marked": encode(s["marked"])}


@app.post("/teach/start")
def teach_start():
    eyes.teach_start()
    return {"ok": True}


@app.post("/teach/stop")
def teach_stop():
    return {"frames": [encode(f) for f in eyes.teach_stop()]}


@app.get("/status")
def status():
    return eyes.status()
