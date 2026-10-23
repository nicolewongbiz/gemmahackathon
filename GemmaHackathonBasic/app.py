"""Gemma + Kinova Gen3 chat. Run:  python app.py   then open http://127.0.0.1:7860

Type anything to chat with Gemma. Ask it to move the arm in plain English
("move the hand 5 cm forward", "go up 10 cm then open the gripper").
The red STOP button stops the arm immediately (it does not go through Gemma).
"""
import argparse
import logging
import os
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "app.log")])
log = logging.getLogger("app")

app = FastAPI(title="Gemma Arm Chat")
STATE = {"arm": None, "exec": None, "llm": None, "llm_lock": threading.Lock(),
         "execute": True, "last_latency": None, "error": None}


def startup():
    from arm import make_arm
    from commands import Executor
    from gemma import make_backend
    arm = make_arm()
    STATE["arm"], STATE["exec"] = arm, Executor(arm)
    STATE["llm"] = make_backend()


class ChatIn(BaseModel):
    message: str


def _state_line():
    try:
        p = STATE["arm"].pose()
        return (f"[robot state] hand at x={p['x']:.3f} y={p['y']:.3f} z={p['z']:.3f} m, "
                f"gripper={p.get('gripper')}, busy={STATE['exec'].busy}")
    except Exception as e:  # noqa: BLE001
        return f"[robot state] unavailable ({e})"


@app.post("/api/chat")
def chat(body: ChatIn):
    from commands import SafetyError, describe
    from gemma import parse_reply
    text = body.message.strip()
    if not text:
        return {"reply": "", "actions": [], "status": "empty"}
    prompt = f"{text}\n\n{_state_line()}"
    t0 = time.time()
    with STATE["llm_lock"]:
        try:
            raw = STATE["llm"].send(prompt)
        except Exception as e:  # noqa: BLE001
            log.exception("Gemma failed")
            return {"reply": f"Gemma error: {e}", "actions": [], "status": "error"}
    STATE["last_latency"] = round(time.time() - t0, 2)
    out = parse_reply(raw)
    log.info("user=%r raw=%r latency=%.2fs", text, raw[:400], STATE["last_latency"])
    status = "chat"
    planned = []
    if out["actions"]:
        try:
            if STATE["execute"]:
                planned = STATE["exec"].submit(out["actions"])
                status = "executing"
            else:
                from commands import validate
                planned = validate(out["actions"])
                status = "dry-run"
        except SafetyError as e:
            out["reply"] += f"  (Not executed: {e})"
            status = "blocked"
    return {"reply": out["reply"], "actions": [describe(a) for a in planned], "status": status,
            "latency": STATE["last_latency"]}


@app.post("/api/stop")
def stop():
    STATE["exec"].stop()
    return {"ok": True}


@app.post("/api/home")
def home():
    from commands import SafetyError
    try:
        STATE["exec"].submit([{"type": "home"}])
        return {"ok": True}
    except SafetyError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/reset")
def reset():
    with STATE["llm_lock"]:
        STATE["llm"].reset()
    return {"ok": True}


@app.post("/api/execute/{on}")
def set_execute(on: int):
    STATE["execute"] = bool(on)
    return {"execute": STATE["execute"]}


@app.get("/api/state")
def state():
    try:
        pose = STATE["arm"].pose()
    except Exception as e:  # noqa: BLE001
        pose = {"error": str(e)}
    return {"pose": pose, "busy": STATE["exec"].busy, "log": STATE["exec"].log[-30:],
            "arm": STATE["arm"].name, "backend": STATE["llm"].name, "execute": STATE["execute"],
            "latency": STATE["last_latency"]}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma Arm Chat</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--muted:#6a737d;--line:#e1e4e8;--accent:#1a73e8;--me:#e8f0fe;--red:#d93025}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1c1f24;--ink:#e6e6e6;--muted:#9aa0a6;--line:#30343a;--accent:#8ab4f8;--me:#233150}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{display:grid;grid-template-columns:1fr 320px;gap:16px;max-width:1200px;margin:0 auto;padding:16px;height:100vh}
@media (max-width:860px){.wrap{grid-template-columns:1fr;height:auto}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;display:flex;flex-direction:column;min-height:0}
h1{font-size:17px;margin:0 0 4px}.sub{color:var(--muted);font-size:13px;margin-bottom:10px}
#chat{flex:1;overflow:auto;display:flex;flex-direction:column;gap:10px;padding:4px;min-height:300px}
.msg{max-width:85%;padding:9px 12px;border-radius:12px;white-space:pre-wrap}
.me{align-self:flex-end;background:var(--me)}.bot{align-self:flex-start;border:1px solid var(--line)}
.acts{font-size:12.5px;color:var(--muted);margin-top:6px}
form{display:flex;gap:8px;margin-top:10px}textarea{flex:1;resize:none;height:52px;padding:10px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
button{border:0;border-radius:10px;padding:10px 14px;font:inherit;font-weight:600;cursor:pointer;background:var(--accent);color:#fff}
button.sec{background:transparent;color:var(--ink);border:1px solid var(--line)}
#stop{background:var(--red);font-size:20px;padding:18px;width:100%;margin-bottom:10px}
.row{display:flex;gap:8px;margin-bottom:10px}.row button{flex:1}
pre{margin:0;font:12px/1.4 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow:auto;flex:1;color:var(--muted)}
.kv{font:13px ui-monospace,Consolas,monospace;margin-bottom:10px}
label{font-size:13px;display:flex;gap:6px;align-items:center;margin-bottom:10px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}.chip{font-size:12.5px;padding:4px 10px;border:1px solid var(--line);border-radius:999px;cursor:pointer;background:transparent;color:var(--ink);font-weight:400}
</style></head><body><div class="wrap">
<div class="card"><h1>Gemma ⇄ Kinova Gen3</h1><div class="sub" id="sub">loading…</div>
<div id="chat"></div>
<div class="chips">
<button class="chip">move the hand 5 cm forward</button><button class="chip">go up 10 cm</button>
<button class="chip">move left 3 cm and down 2 cm</button><button class="chip">open the gripper</button>
<button class="chip">close the gripper</button><button class="chip">rotate the wrist 20 degrees</button>
<button class="chip">where is the hand right now?</button></div>
<form id="f"><textarea id="t" placeholder="Chat with Gemma or tell the arm what to do…  (Enter to send, Shift+Enter for newline)"></textarea><button>Send</button></form></div>
<div class="card"><button id="stop">STOP</button>
<div class="row"><button class="sec" id="home">Home</button><button class="sec" id="reset">New chat</button></div>
<label><input type="checkbox" id="exec" checked> Execute arm actions (uncheck = dry run)</label>
<div class="kv" id="pose">pose: –</div><pre id="log"></pre></div></div>
<script>
const $=s=>document.querySelector(s),chat=$('#chat');
function add(cls,text,acts){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;
 if(acts&&acts.length){const a=document.createElement('div');a.className='acts';a.textContent='▶ '+acts.join('  ·  ');d.appendChild(a)}
 chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'content-type':'application/json'},body:b?JSON.stringify(b):null});return r.json()}
async function send(text){if(!text.trim())return;add('me',text);const w=add('bot','…');
 try{const r=await post('/api/chat',{message:text});w.remove();
 add('bot',r.reply||'(no reply)',r.actions&&r.actions.length?r.actions.map(x=>x+(r.status==='dry-run'?' (dry run)':'')):null)}
 catch(e){w.textContent='Error: '+e}}
$('#f').onsubmit=e=>{e.preventDefault();const v=$('#t').value;$('#t').value='';send(v)};
$('#t').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#f').requestSubmit()}};
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>send(c.textContent));
$('#stop').onclick=()=>post('/api/stop');$('#home').onclick=()=>post('/api/home');
$('#reset').onclick=async()=>{await post('/api/reset');chat.innerHTML=''};
$('#exec').onchange=e=>post('/api/execute/'+(e.target.checked?1:0));
document.addEventListener('keydown',e=>{if(e.key==='Escape')post('/api/stop')});
async function poll(){try{const s=await (await fetch('/api/state')).json();const p=s.pose||{};
 $('#sub').textContent=`backend: ${s.backend} · arm: ${s.arm} · ${s.busy?'MOVING':'idle'}`+(s.latency?` · last Gemma reply ${s.latency}s`:'')+' · Esc = STOP';
 $('#pose').textContent=p.error?('pose error: '+p.error):`x ${(+p.x).toFixed(3)}  y ${(+p.y).toFixed(3)}  z ${(+p.z).toFixed(3)} m\\nθ ${(+p.tx).toFixed(0)}, ${(+p.ty).toFixed(0)}, ${(+p.tz).toFixed(0)}°  grip ${p.gripper??'?'}`;
 $('#log').textContent=s.log.slice().reverse().join('\\n');$('#exec').checked=s.execute}catch(e){$('#sub').textContent='server offline'}}
setInterval(poll,700);poll();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chat with Gemma and drive a Kinova Gen3")
    ap.add_argument("--mock", action="store_true", help="no robot (MockArm)")
    ap.add_argument("--real", action="store_true", help="connect to the real Kinova arm")
    ap.add_argument("--backend", choices=["litert", "ollama", "rules"], help="Gemma backend")
    ap.add_argument("--port", type=int, default=config.PORT)
    a = ap.parse_args()
    if a.mock:
        config.MOCK_ARM = True
    if a.real:
        config.MOCK_ARM = False
    if a.backend:
        config.BACKEND = a.backend
    startup()
    print(f"\n  Open http://{config.HOST}:{a.port}  in your browser\n")
    uvicorn.run(app, host=config.HOST, port=a.port, log_level="warning")
