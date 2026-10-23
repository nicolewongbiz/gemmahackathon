"""Gemma chat backends. Each backend has .send(user_text) -> raw str and .reset().
parse_reply() turns the raw text into {"reply": str, "actions": [...]}.
"""
import json
import logging
import re
import time

import requests

import config

log = logging.getLogger("gemma")

SYSTEM_PROMPT = """You are Gemma, a friendly assistant running fully on-device on a laptop that is connected to a Kinova Gen3 robot arm.
You can chat normally AND control the arm.

ALWAYS answer with ONE JSON object and nothing else:
{"reply": "<what you say to the user>", "actions": [ ... ]}

Use "actions": [] when the user is just chatting or asking a question.

Coordinate frame (robot base frame, as seen standing behind the robot):
- forward = +x (away from the base), back = -x
- left = +y, right = -y
- up = +z, down = -z
Distances in actions are in METRES (5 cm = 0.05). Angles in DEGREES.

Allowed actions:
{"type": "move", "dx": 0.0, "dy": 0.0, "dz": 0.0}          relative move of the hand
{"type": "move_to", "x": 0.5, "y": 0.0, "z": 0.3}          absolute position (only if the user gives coordinates)
{"type": "rotate", "droll": 0, "dpitch": 0, "dyaw": 0}      relative wrist rotation in degrees
{"type": "gripper", "value": 0.0}                           0.0 = open, 1.0 = closed, 0.5 = half
{"type": "home"}                                            go to the home position
{"type": "stop"}                                            stop immediately
{"type": "wait", "seconds": 1.0}

Rules:
- One move may be at most 0.20 m and one rotation at most 45 degrees; split bigger requests into several actions.
- If the request is ambiguous or unsafe, ask a short clarifying question with "actions": [].
- If no distance is given, use 0.05 m (5 cm). "a little" = 0.02 m. "a lot" = 0.15 m.
- Keep "reply" short (one or two sentences).

Examples:
User: move the hand 5 cm forward
{"reply": "Moving 5 cm forward.", "actions": [{"type": "move", "dx": 0.05, "dy": 0, "dz": 0}]}
User: go up 10 cm and then left 3 cm, then close the gripper
{"reply": "Up 10 cm, left 3 cm, then closing the gripper.", "actions": [{"type": "move", "dx": 0, "dy": 0, "dz": 0.10}, {"type": "move", "dx": 0, "dy": 0.03, "dz": 0}, {"type": "gripper", "value": 1.0}]}
User: what can you do?
{"reply": "I can chat, and move the arm: forward/back, left/right, up/down, rotate the wrist, open or close the gripper, or go home.", "actions": []}
"""


def parse_reply(raw):
    """Extract {"reply", "actions"} from model text. Never raises."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {"reply": str(obj.get("reply", "")), "actions": obj.get("actions") or []}
        except json.JSONDecodeError:
            pass
    return {"reply": text, "actions": []}


# ---------------------------------------------------------------- LiteRT-LM
class LiteRTBackend:
    """Gemma via Google's LiteRT-LM Python API (pip install litert-lm-api).

    Known API: litert_lm.Engine(path) -> engine.create_conversation(messages=[...])
               conv.send_message(text) -> resp["content"][0]["text"]
    VERIFY ON SITE: the GPU backend argument name. Run scripts/check_gemma.py --probe.
    """
    name = "litert"

    def __init__(self):
        import litert_lm
        self.lm = litert_lm
        self.engine = self._make_engine()
        self.conv = None
        self.reset()

    def _make_engine(self):
        lm, path = self.lm, config.LITERT_MODEL
        attempts = []
        B = getattr(lm, "Backend", None)
        if config.LITERT_USE_GPU and B is not None and hasattr(B, "GPU"):
            attempts.append(("backend=Backend.GPU", lambda: lm.Engine(path, backend=B.GPU)))
        if B is not None and hasattr(B, "CPU"):
            attempts.append(("backend=Backend.CPU", lambda: lm.Engine(path, backend=B.CPU)))
        attempts.append(("defaults", lambda: lm.Engine(path)))
        last = None
        for label, fn in attempts:
            try:
                eng = fn()
                log.info("LiteRT-LM engine created with %s", label)
                self.engine_label = label
                return eng
            except Exception as e:  # noqa: BLE001
                log.warning("LiteRT-LM Engine(%s) failed: %s", label, e)
                last = e
        raise RuntimeError(f"Could not create LiteRT-LM engine for {path}: {last}")

    def reset(self):
        if self.conv is not None and hasattr(self.conv, "__exit__"):
            try:
                self.conv.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        self.conv = self.engine.create_conversation(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}])
        if hasattr(self.conv, "__enter__"):
            self.conv = self.conv.__enter__()

    def send(self, text):
        resp = self.conv.send_message(text)
        if isinstance(resp, dict):
            return resp["content"][0]["text"]
        return str(resp)


# ---------------------------------------------------------------- Ollama
class OllamaBackend:
    name = "ollama"

    def __init__(self):
        self.reset()
        try:
            requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Ollama not reachable at {config.OLLAMA_URL}: {e}") from e

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def send(self, text):
        self.history.append({"role": "user", "content": text})
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
            "model": config.OLLAMA_MODEL, "messages": self.history, "stream": False,
            "format": "json", "options": {"temperature": config.TEMPERATURE}}, timeout=180)
        r.raise_for_status()
        out = r.json()["message"]["content"]
        self.history.append({"role": "assistant", "content": out})
        self.history = self.history[:1] + self.history[-20:]
        return out


# ---------------------------------------------------------------- Rules (no model)
_UNITS = {"mm": 0.001, "millimeter": 0.001, "millimetre": 0.001, "cm": 0.01, "centimeter": 0.01,
          "centimetre": 0.01, "m": 1.0, "meter": 1.0, "metre": 1.0, "in": 0.0254, "inch": 0.0254}
_DIRS = {"forward": ("dx", 1), "forwards": ("dx", 1), "ahead": ("dx", 1), "back": ("dx", -1),
         "backward": ("dx", -1), "backwards": ("dx", -1), "left": ("dy", 1), "right": ("dy", -1),
         "up": ("dz", 1), "down": ("dz", -1)}


class RulesBackend:
    """Regex parser so you can test the arm path with no model at all."""
    name = "rules"

    def reset(self):
        pass

    def send(self, text):
        t = text.lower().split("[robot state]")[0]
        actions = []
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?|in|inch(?:es)?)?\s*(?:to the\s+)?(forwards?|ahead|backwards?|back|left|right|up|down)", t):
            val = float(m.group(1))
            unit = (m.group(2) or "cm").rstrip("s")
            unit = next((k for k in _UNITS if unit.startswith(k) and (len(k) > 1 or unit == k)), "cm")
            axis, sign = _DIRS[m.group(3)]
            a = {"type": "move", "dx": 0.0, "dy": 0.0, "dz": 0.0}
            a[axis] = sign * val * _UNITS[unit]
            actions.append(a)
        if re.search(r"\bopen\b", t):
            actions.append({"type": "gripper", "value": 0.0})
        if re.search(r"\b(close|grab|grip)\b", t):
            actions.append({"type": "gripper", "value": 1.0})
        if "home" in t:
            actions.append({"type": "home"})
        if re.search(r"\bstop\b", t):
            actions = [{"type": "stop"}]
        reply = "OK." if actions else "(rules backend) I only understand things like 'move 5 cm forward', 'open', 'close', 'home'."
        return json.dumps({"reply": reply, "actions": actions})


# ----------------------------------------------------------------
def make_backend(name=None):
    name = name or config.BACKEND
    t = time.time()
    b = {"litert": LiteRTBackend, "ollama": OllamaBackend, "rules": RulesBackend}[name]()
    log.info("Backend %s ready in %.1fs", name, time.time() - t)
    return b
