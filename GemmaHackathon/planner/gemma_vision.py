"""One function the rest of the code calls: ask(prompt, images) -> str.

Backends:
  litert  - LiteRT-LM Python API (pip install litert-lm-api)
  ollama  - local Ollama server, fallback

NOTE: The exact LiteRT-LM Python argument names for GPU + vision were not verified
before the event. If litert init fails, run scripts/test_gemma.py, read the error,
check `python -c "import litert_lm; help(litert_lm.Engine)"` and fix _litert_engine().
"""
import base64, json, re
from pathlib import Path
import cv2
import requests
import config

_engine = None


def _litert_engine():
    global _engine
    if _engine is not None:
        return _engine
    import litert_lm
    kwargs = {}
    # Try GPU backend + vision; adjust names after `help(litert_lm.Engine)`.
    Backend = getattr(litert_lm, "Backend", None)
    if Backend is not None:
        kwargs["backend"] = getattr(Backend, "GPU", None) or Backend.CPU
        if hasattr(Backend, "CPU"):
            kwargs["vision_backend"] = getattr(Backend, "GPU", Backend.CPU)
    try:
        _engine = litert_lm.Engine(config.LITERT_MODEL, **kwargs)
    except TypeError:
        print("[gemma] Engine kwargs rejected, retrying with defaults:", kwargs)
        _engine = litert_lm.Engine(config.LITERT_MODEL)
    return _engine


def _save_tmp(img, i):
    p = config.DATA / f"_prompt_{i}.jpg"
    cv2.imwrite(str(p), _shrink(img))
    return str(p)


def _shrink(img):
    h, w = img.shape[:2]
    s = config.PROMPT_IMAGE_MAX / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def _ask_litert(prompt, images, system):
    engine = _litert_engine()
    content = [{"type": "image", "path": _save_tmp(im, i)} for i, im in enumerate(images)]
    content.append({"type": "text", "text": prompt})
    msgs = [{"role": "system", "content": system}] if system else []
    with engine.create_conversation(messages=msgs) as conv:
        resp = conv.send_message({"role": "user", "content": content})
    return resp["content"][0]["text"]


def _ask_ollama(prompt, images, system):
    imgs = [base64.b64encode(cv2.imencode(".jpg", _shrink(im))[1]).decode() for im in images]
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt, "images": imgs}]
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": config.OLLAMA_MODEL, "messages": msgs, "stream": False,
        "format": "json", "options": {"temperature": 0.2}}, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ask(prompt, images=(), system=None, backend=None):
    backend = backend or config.GEMMA_BACKEND
    fn = _ask_litert if backend == "litert" else _ask_ollama
    return fn(prompt, list(images), system)


def ask_json(prompt, images=(), system=None, backend=None, retries=2):
    """ask() and parse the first {...} block. Retries on bad JSON."""
    last = ""
    for _ in range(retries + 1):
        last = ask(prompt, images, system, backend)
        m = re.search(r"\{.*\}", last, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        prompt += "\nReply with ONLY a valid JSON object."
    raise ValueError(f"Gemma did not return JSON: {last[:300]}")
