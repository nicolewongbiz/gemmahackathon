"""All settings. Any value can be overridden with an environment variable GB_<NAME>,
e.g.  set GB_MOCK_ARM=0   (Windows)   /   export GB_MOCK_ARM=0   (Linux/macOS)
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent


def _env(name, default):
    v = os.environ.get("GB_" + name)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes", "on")
    return type(default)(v)


# ---------- Web UI ----------
HOST = _env("HOST", "127.0.0.1")
PORT = _env("PORT", 7860)

# ---------- Gemma ----------
# "litert" = LiteRT-LM (on-device, preferred)
# "ollama" = local Ollama server (fallback)
# "rules"  = no model, simple regex parser (for testing the arm path)
BACKEND = _env("BACKEND", "litert")
LITERT_MODEL = _env("LITERT_MODEL", str(ROOT / "models" / "gemma.litertlm"))
LITERT_HF_REPO = _env("LITERT_HF_REPO", "litert-community/gemma-4-E4B-it-litert-lm")
LITERT_USE_GPU = _env("LITERT_USE_GPU", True)
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "gemma4:e4b")   # check exact tag with `ollama list`
TEMPERATURE = _env("TEMPERATURE", 0.2)

# ---------- Robot: Kinova Gen3 (7-DoF) over Ethernet ----------
MOCK_ARM = _env("MOCK_ARM", True)          # start in mock mode; set GB_MOCK_ARM=0 for the real arm
ROBOT_IP = _env("ROBOT_IP", "192.168.1.10")
ROBOT_USER = _env("ROBOT_USER", "admin")
ROBOT_PASS = _env("ROBOT_PASS", "admin")
ACTION_TIMEOUT_S = _env("ACTION_TIMEOUT_S", 20.0)

# Speeds (keep slow near people)
LINEAR_SPEED = _env("LINEAR_SPEED", 0.10)      # m/s
ANGULAR_SPEED = _env("ANGULAR_SPEED", 20.0)    # deg/s

# Safety limits
MAX_STEP_M = _env("MAX_STEP_M", 0.20)          # largest single translation per action
MAX_ROT_DEG = _env("MAX_ROT_DEG", 45.0)        # largest single rotation per action
MAX_ACTIONS = _env("MAX_ACTIONS", 8)           # per chat message
# Tool position must stay inside this box (robot BASE frame, metres). Tune to your table.
WORKSPACE = {
    "x": (_env("WS_XMIN", 0.15), _env("WS_XMAX", 0.85)),
    "y": (_env("WS_YMIN", -0.55), _env("WS_YMAX", 0.55)),
    "z": (_env("WS_ZMIN", 0.03), _env("WS_ZMAX", 0.80)),
}
