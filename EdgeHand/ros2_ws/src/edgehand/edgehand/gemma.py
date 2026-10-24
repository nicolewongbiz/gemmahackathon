"""Gemma 4 on LiteRT-LM: the only place litert_lm is used. Shared by intent.py (text) and namer.py (image + text)."""
import json, re
from edgehand import config


def make_engine(model, vision=False, max_images=1, audio=False):
    import litert_lm
    B = litert_lm.Backend
    # newer litert-lm wants Backend instances (GPU()), older ones take the enum value
    gpu, cpu = (b() if callable(b) else b for b in (B.GPU, B.CPU))
    # GPU text + CPU vision was the fastest combination on the RTX 4070 (GemmaHackathon/planner/gemma_vision.py)
    kw = {"vision_backend": cpu, "max_num_images": max_images} if vision else {}
    if audio:
        kw["audio_backend"] = cpu                 # as in GemmaHackathon/pi/voice_agent.py
    try:
        return litert_lm.Engine(model, backend=gpu if config.GEMMA_USE_GPU else cpu, **kw)
    except Exception:
        return litert_lm.Engine(model, backend=cpu, **kw)


def ask(engine, system, text, images=(), audio=None):
    """images: paths, sent before the text in order (the prompt can call them image 1, 2, ...); audio: a wav path."""
    content = [{"type": "image", "path": str(p)} for p in images] + [{"type": "text", "text": text}]
    if audio:
        content.insert(0, {"type": "audio", "path": str(audio)})
    with engine.create_conversation(messages=[{"role": "system", "content": system}]) as conv:
        r = conv.send_message({"role": "user", "content": content})
    return r["content"][0]["text"]


def first_json(text):
    """First {...} in Gemma's reply. Raises ValueError if there is none."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in: {text[:200]}")
    return json.loads(m.group(0))


def ask_json(engine, system, text, images=(), retries=config.GEMMA_RETRIES, audio=None):
    """None if every try failed: bad JSON, or LiteRT itself failing (it times out after 2 min, e.g. when another
    process holds the GPU). Callers treat None as "didn't understand"; a node must not crash on it."""
    prompt = text
    for _ in range(retries + 1):
        try:
            return first_json(ask(engine, system, prompt, images, audio))
        except (ValueError, json.JSONDecodeError):
            prompt = text + "\nReply with ONLY a valid JSON object."
        except RuntimeError:
            pass
    return None
