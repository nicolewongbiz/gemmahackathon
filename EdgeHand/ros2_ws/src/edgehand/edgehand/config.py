"""Every EdgeHand setting. Edit on site. Distances in metres, robot base frame (x forward, y left, z up)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # EdgeHand/ (build with --symlink-install so this resolves)

# ---- Tools: class -> handle colour (OpenCV HSV, hue 0-180). Red wraps around 0, so it has two ranges ----
TOOLS = {
    "phillips": [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (180, 255, 255))],   # red handle
    "flathead": [((20, 120, 90), (35, 255, 255))],                                      # yellow handle
    "torx":     [((45, 100, 60), (85, 255, 255))],                                      # green handle
}
DEFAULT_TOOL = "phillips"          # what the Pi REQUEST button asks for

# ---- Gemma 4 intent parser (LiteRT-LM, workstation) ----
GEMMA_MODEL = os.environ.get("EDGEHAND_MODEL", str(ROOT / "models" / "gemma-4-E4B-it.litertlm"))
GEMMA_USE_GPU = True               # falls back to CPU if the GPU backend fails
GEMMA_RETRIES = 2

# ---- Camera + calibration (Jetson, fixed overhead USB camera) ----
CAMERA_TOPIC = "/image_raw"        # v4l2_camera default
TABLE_TAGS = {                     # tag36h11 id -> tag centre (x, y). MEASURE ON SITE (README Phase 1.4)
    0: (0.35, -0.20),
    1: (0.35, 0.20),
    2: (0.60, 0.20),
    3: (0.60, -0.20),
}
TAG_SIZE_M = 0.05
TABLE_Z = 0.0                      # table surface height in the base frame
SHAFT_MIN_VAL = 140                # shaft = bright (HSV V) and unsaturated; a lit dark mat stays below this
TOOL_MIN_AREA_PX = 800             # smaller blobs are noise
SHAFT_MAX_SAT = 70                 # shaft = low saturation (metal); handle = its TOOLS colour
TOOL_RADIUS_M = 0.012              # grasp height above the table (shaft lies on the handle's radius)
GRASP_FROM_FERRULE_M = 0.02        # grasp the shaft this far from where the handle ends

# ---- Motion (MoveIt 2 via ros2_kortex) ----
PLANNING_GROUP = "manipulator"
BASE_FRAME = "base_link"
EE_LINK = "end_effector_link"      # tip of the manipulator group = the flange
# Flange -> the point between the pads that goes on the shaft axis, along tool z. The 2F-85 fingertips move ~13 mm
# further out as they close on a thin shaft, so: TCP_OFFSET_M = closed fingertip distance + 0.004 - TOOL_RADIUS_M.
# Sim (Menagerie 2F-85): 0.173 + 0.004 - 0.012 = 0.165. MEASURE the real one (README Phase 1.6).
TCP_OFFSET_M = 0.165
FINGERS_ALONG = "x"                # flange axis the 2F-85 fingers close along (check in RViz, Phase 1.6)
HOVER_M = 0.12                     # approach / lift height above the grasp
# Kinova "Retract" (joint_1..7, rad): folded by the base, out of the overhead camera's view. The arm parks here
# before vision looks at the tray and after each handover (in Home it hangs over the tray and hides the tools).
PARK_JOINTS = [0.0, -0.35, 3.14, -2.55, 0.0, -0.87, 1.57]
VEL_SCALE, ACC_SCALE = 0.2, 0.2
GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.8
GRIPPER_MAX_EFFORT = 50.0
HANDOVER_XYZ = (0.45, 0.30, 0.35)  # where the tool is presented (operator's chest height, not face)
OPERATOR_DIR = (0.0, 1.0, 0.0)     # unit vector, base frame: the handle points this way
WORKSPACE = {"x": (0.15, 0.80), "y": (-0.55, 0.55), "z": (0.0, 0.70)}
# MoveIt only avoids what is in its planning scene. handover_node adds the table and a keep-out box this tall over
# the tag rectangle (the tools), so free moves never sweep the tray; only the vertical descend/lift enter it.
TRAY_KEEPOUT_H = 0.06

# ---- Release: tug detection on joint torques (/joint_states effort, N*m) ----
TUG_JOINTS = ["joint_5", "joint_6", "joint_7"]   # wrist joints feel a pull on the tool most
TUG_TORQUE_NM = 1.5                # |torque - baseline| over these joints (L2) to count as a tug
TUG_SAMPLES = 5                    # consecutive samples over threshold
SETTLE_S = 0.8                     # wait after arriving at the handover pose before taking the baseline

# ---- Pi link (UDP, no ROS on the Pi) ----
PI_UDP_PORT = 5005
WATCHDOG_S = 0.5                   # no heartbeat for this long -> /edgehand/stop
