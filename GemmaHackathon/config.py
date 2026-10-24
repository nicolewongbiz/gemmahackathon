"""Shared settings. Edit these on site."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# ---- Network (Pi <-> laptop, no internet needed) ----
LAPTOP_HOST = "0.0.0.0"
LAPTOP_PORT = 8000

# ---- Camera ----
# "webcam" = USB camera (our arm has no Kinova Vision module)
# "kinova" = Kinova Vision wrist module over RTSP     "mock" = grey frames, no hardware
CAMERA = "webcam"
WEBCAM_INDEX = 0                  # Brio 100 on the dev laptop; run scripts/find_camera.py if it's the wrong view
CAMERA_ON_WRIST = False           # False = fixed on a stand above the table
VISION_CONNECT_TIMEOUT = 10.0     # s to wait for the first frame
FRAME_W, FRAME_H = 1280, 720      # requested resolution (Kinova: set the stream to this on the arm)
# Where the arm waits while a perception frame is taken. All four table tags + both bin tags must be in view.
#   On the wrist: the pose the camera looks down from.  On a stand: a pose that keeps the arm out of the picture.
# VERIFY ON SITE: jog in the Web App until calibrate shows CALIBRATED, then copy x, y, z here.
# Stand camera: Kinova factory Home pose (from the Web App, gripper pointing forward, clear of the table view).
LOOK_POSE = (0.475, 0.0, 0.55) if CAMERA_ON_WRIST else (0.579, -0.003, 0.433)   # x, y, z (m; z minus Z_TABLE_OFFSET)
LOOK_ORIENT = None if CAMERA_ON_WRIST else (90.4, 0.0, 89.7)   # theta_x/y/z deg; None = TOOL_ORIENT (pointing down)
PROMPT_IMAGE_MAX = 768          # px, longest side sent to Gemma (lower if VRAM is tight)
KEYFRAME_MAX = 6                # max teach frames per Gemma call

# ---- Gemma (laptop) ----
# "litert" = LiteRT-LM Python API (preferred, counts for LiteRT prize)
# "ollama" = fallback if LiteRT vision misbehaves
GEMMA_BACKEND = "litert"
LITERT_MODEL = str(DATA / "gemma-4-E4B-it.litertlm")   # download: see README §2
LITERT_HF_REPO = "litert-community/gemma-4-E4B-it-litert-lm"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"       # verify tag with `ollama list`

# ---- AprilTags (tag36h11) ----
# Corner tags: id -> (x, y) of the tag CENTRE in the robot BASE frame, metres.
# Measure with measure_tags.bat (jog the gripper onto each tag; the arm reads x, y). Tape is only ~1-2 cm.
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
USE_MOCK_ARM = False               # flip to False once Kortex is connected
Z_HOVER = 0.25                    # m above table; keep >= Z_GRASP + 0.08
Z_GRASP = 0.135                   # m, measured in the Web App (End Effector Z at a good grasp)
Z_TABLE_OFFSET = 0.0              # table height in base frame
TOOL_ORIENT = (180.0, 0.0, 90.0)  # theta_x, theta_y, theta_z (deg) = pointing straight down
GRIPPER_OPEN = 0.0                # 0 = fully open
GRIPPER_CLOSE_SPEED = 0.3         # 0..1; the gripper closes until it stalls on the object. Lower = gentler
GRIPPER_EMPTY = 0.95              # closing past this means it grabbed nothing
BIN_DROP = {"A": (0.45, -0.40, 0.15), "B": (0.45, 0.40, 0.15)}  # fallback if bin tags not seen
BIN_RADIUS_M = 0.12               # objects this close to a bin tag count as already sorted
MAX_LINEAR_SPEED = 0.10           # m/s
MAX_ANGULAR_SPEED = 20.0          # deg/s
# Every arm target must be inside this box (base frame, m, z above table) or the move is refused.
# Must contain the table tags, both bins and LOOK_POSE. Tune to your table.
WORKSPACE = {"x": (0.20, 0.75), "y": (-0.50, 0.50), "z": (0.0, 0.70)}
