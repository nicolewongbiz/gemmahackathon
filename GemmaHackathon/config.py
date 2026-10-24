"""Shared settings. Edit these on site."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# ---- Swarm: one switch, everyone on 192.168.1.x, no internet needed ----
#   arm .10 (camera + motors) | laptop .11 brain + hands (main.py) | Pi .12 ears + mouth | Jetson .13 eyes
LAPTOP_HOST = "0.0.0.0"
LAPTOP_PORT = 8000
EYES = "jetson"                   # "jetson" = perception on the Jetson; "local" = laptop does it (fallback)
JETSON_IP = "192.168.1.13"
JETSON_PORT = 8001
EYES_LABELS = True                # the Jetson's Gemma names each numbered object for the planner
EYES_GEMMA_BACKEND = "ollama"     # Gemma on the Jetson: "ollama" (CUDA) or "litert"
EYES_GEMMA_MODEL = "gemma4:e2b"   # Ollama tag (or a .litertlm path for litert). Verify with `ollama list`

# ---- Camera: Kinova Vision module on the wrist (RTSP from the arm) ----
USE_MOCK_CAMERA = False           # True = grey frames, no arm needed
VISION_CONNECT_TIMEOUT = 10.0     # s to wait for the first RTSP frame
FRAME_W, FRAME_H = 1280, 720      # must match the stream resolution set on the arm
# Pose the wrist camera looks down from; all four table tags + both bin tags must be in view.
# VERIFY ON SITE: jog in the Web App until calibrate shows CALIBRATED, then copy x, y, z here.
# Sim (sim/): the old guess (0.475, 0, 0.55) is out of reach pointing straight down (wrist fully stretched).
# (0.40, 0, 0.50) reaches and sees all four default TABLE_TAGS past the gripper fingers.
LOOK_POSE = (0.40, 0.0, 0.50)     # x, y, z (m, z above table), orientation = TOOL_ORIENT
PROMPT_IMAGE_MAX = 768          # px, longest side sent to Gemma (lower if VRAM is tight)
KEYFRAME_MAX = 6                # max teach frames per Gemma call

# ---- Gemma (laptop planner) ----
# "litert" = LiteRT-LM Python API (preferred, counts for LiteRT prize)
# "ollama" = fallback if LiteRT vision misbehaves
GEMMA_BACKEND = "litert"
LITERT_MODEL = str(DATA / "gemma-4-E4B-it.litertlm")   # download: see README §2
LITERT_HF_REPO = "litert-community/gemma-4-E4B-it-litert-lm"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"       # verify tag with `ollama list`

# ---- AprilTags (tag36h11) ----
# Corner tags: id -> (x, y) of the tag CENTRE in the robot BASE frame, metres.
# Measure these with a tape from the arm base once the table is set up.
# Sim: from LOOK_POSE the wrist camera's short image axis runs along x, and the near edge is cut by the
# gripper fingers, so keep the near/far tags at x = 0.32 / 0.62 (0.30 / 0.65 drops a tag off the frame).
TABLE_TAGS = {
    0: (0.32, -0.25),
    1: (0.32, 0.25),
    2: (0.62, 0.25),
    3: (0.62, -0.25),
}
BIN_TAGS = {10: "A", 11: "B"}     # tag id -> bin name
TAG_SIZE_M = 0.05

# ---- Robot (Kinova Kortex) ----
ROBOT_IP = "192.168.1.10"
ROBOT_USER = "admin"
ROBOT_PASS = "admin"
USE_MOCK_ARM = True               # flip to False once Kortex is connected
Z_HOVER = 0.15                    # m above table
# m, Kortex tool-frame height at the grasp. With the default 0.12 m tool frame the pads hang ~4 cm below it and
# swing ~1 cm lower as they close: 0.025 drives them into the table. Sim, 4 cm ball: 100% at 0.058-0.065.
Z_GRASP = 0.062
Z_TABLE_OFFSET = 0.0              # table height in base frame
TOOL_ORIENT = (180.0, 0.0, 90.0)  # theta_x, theta_y, theta_z (deg) = pointing straight down
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.7
BIN_DROP = {"A": (0.45, -0.40, 0.15), "B": (0.45, 0.40, 0.15)}  # fallback if bin tags not seen
BIN_RADIUS_M = 0.12               # objects this close to a bin tag count as already sorted
MAX_LINEAR_SPEED = 0.10           # m/s
MAX_ANGULAR_SPEED = 20.0          # deg/s
# Every arm target must be inside this box (base frame, m, z above table) or the move is refused.
# Must contain the table tags, both bins and LOOK_POSE. Tune to your table.
WORKSPACE = {"x": (0.20, 0.75), "y": (-0.50, 0.50), "z": (0.0, 0.70)}

# ---- Ball -> box task (scripts/ball_to_box.py): white ball on a black table, into a blue box ----
BALL_MAX_SAT, BALL_MIN_VAL = 60, 170      # HSV: white = low saturation, bright
BALL_AREA_PX = (150, 40000)               # px^2 at LOOK_POSE
BOX_HSV_LO, BOX_HSV_HI = (95, 110, 40), (130, 255, 255)   # OpenCV hue 0-180; blue ~ 110
BOX_MIN_AREA_PX = 3000

# ---- Simulation (sim/, MuJoCo). Real-arm code ignores these ----
# Kortex tool frame offset from the flange along z. The arm stores its own copy:
# Web App -> Configurations -> Robot -> Tool. Make this match, or sim heights won't mean the same thing.
SIM_TOOL_FRAME_Z = 0.12
SIM_BALL_RADIUS = 0.02                    # m (ping-pong ball = 0.02)
SIM_BALL_MASS = 0.0027                    # kg (ping-pong ball = 2.7 g)
SIM_BOX_SIZE = (0.15, 0.15, 0.08)         # m: outer x, y, wall height
SIM_BOX_XY = (0.45, -0.40)                # box centre in the base frame
SIM_TAG_ERROR_M = 0.005                   # std dev of tape-measure error on TABLE_TAGS in randomised trials
# Sorting sim (sim/sort.py): bins with their tag on the floor; the wrist camera at LOOK_POSE sees y within +-0.42
SIM_BIN_XY = {"A": (0.45, -0.35), "B": (0.45, 0.35)}
SIM_BIN_SIZE = (0.16, 0.16, 0.06)         # m: outer x, y, wall height
SIM_OBJ_SIZE = 0.04                       # m: ball diameter / block side, about what Z_GRASP was tuned for
