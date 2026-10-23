"""One function the rest of the code calls: ask(prompt, images) -> str.

Backends:
  litert  - LiteRT-LM Python API (pip install litert-lm-api)
  ollama  - local Ollama server, fallback

The model must be gemma-4-E4B-it.litertlm. The -gpu / -web variants have no vision encoder
and fail with "Failed to create conversation" when given an image.
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
    B = litert_lm.Backend
    # Measured on RTX 4070 / Windows, litert-lm-api 0.18.1: GPU text + CPU vision ≈ 3.5 s/call,
    # faster than GPU vision (≈ 5.4 s).
    _engine = litert_lm.Engine(config.LITERT_MODEL, backend=B.GPU, vision_backend=B.CPU,
                               max_num_images=config.KEYFRAME_MAX)
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
        "format": "json", "keep_alive": "2h", "options": {"temperature": 0.2}}, timeout=120)
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
