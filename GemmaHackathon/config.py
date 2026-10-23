"""Shared settings. Edit these on site."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# ---- Network (Pi <-> laptop, no internet needed) ----
LAPTOP_HOST = "0.0.0.0"
LAPTOP_PORT = 8000

# ---- Camera ----
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
PROMPT_IMAGE_MAX = 768          # px, longest side sent to Gemma (lower if VRAM is tight)

# ---- Gemma (laptop) ----
# "litert" = LiteRT-LM Python API (preferred, counts for LiteRT prize)
# "ollama" = fallback if LiteRT vision misbehaves
GEMMA_BACKEND = "litert"
LITERT_MODEL = str(DATA / "gemma-4-E4B-it.litertlm")   # download via scripts/download_models.sh
LITERT_HF_REPO = "litert-community/gemma-4-E4B-it-litert-lm"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"       # verify tag with `ollama list`

# ---- AprilTags (tag36h11) ----
# Corner tags: id -> (x, y) of the tag CENTRE in the robot BASE frame, metres.
# Measure these with a tape from the arm base once the table is set up.
TABLE_TAGS = {
    0: (0.30, -0.25),
    1: (0.30, 0.25),
    2: (0.65, 0.25),
    3: (0.65, -0.25),
}
BIN_TAGS = {10: "A", 11: "B"}     # tag id -> bin name
TAG_SIZE_M = 0.05

# ---- Robot (Kinova Kortex) ----
ROBOT_IP = "192.168.1.10"
ROBOT_USER = "admin"
ROBOT_PASS = "admin"
USE_MOCK_ARM = True               # flip to False once Kortex is connected
Z_HOVER = 0.15                    # m above table
Z_GRASP = 0.025                   # m, tune per object
Z_TABLE_OFFSET = 0.0              # table height in base frame
TOOL_ORIENT = (180.0, 0.0, 90.0)  # theta_x, theta_y, theta_z (deg) = pointing straight down
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.7
BIN_DROP = {"A": (0.45, -0.40, 0.15), "B": (0.45, 0.40, 0.15)}  # fallback if bin tags not seen
