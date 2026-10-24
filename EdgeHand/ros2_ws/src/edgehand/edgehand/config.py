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
# What each tool looks like from above, in words, for Gemma naming the numbered tools (VISION_NAMER = "gemma").
# Gemma can't see a screwdriver tip from the look pose, so describe what IS visible: the handle.
TOOL_DESCRIPTIONS = {
    "phillips": "screwdriver with a red handle",
    "flathead": "screwdriver with a yellow handle",
    "torx": "screwdriver with a green handle",
}

# ---- Gemma 4 intent parser (LiteRT-LM, workstation) ----
GEMMA_MODEL = os.environ.get("EDGEHAND_MODEL", str(ROOT / "models" / "gemma-4-E4B-it.litertlm"))
GEMMA_USE_GPU = True               # falls back to CPU if the GPU backend fails
GEMMA_RETRIES = 2

# ---- Camera: Kinova Vision module on the wrist (eye-in-hand) + calibration ----
ROBOT_IP = "192.168.1.10"
# "usb"   = a camera on the wrist cabled to this computer (our Intel RealSense: its colour camera is a UVC webcam)
# "rtsp"  = the Kinova Vision module, streamed through the arm (this arm doesn't have one)
# "topic" = a ROS image topic (e.g. realsense2_camera's /camera/camera/color/image_raw)
CAMERA_SOURCE = "usb"
CAMERA_USB_INDEX = 0               # OpenCV index; find it with: python -m edgehand.camera --list
CAMERA_RTSP = f"rtsp://{ROBOT_IP}/color"
FRAME_W, FRAME_H = 1280, 720
CAMERA_TOPIC = "/camera/camera/color/image_raw"   # for CAMERA_SOURCE = "topic"
# The camera rides on the wrist, so frames are only used from the look pose: gripper straight down, high over the
# tray. Flange (end_effector_link) position + Kortex theta_x/y/z. From the GemmaHackathon sim: tool frame at
# (0.40, 0, 0.50) = flange at z 0.62 reaches and sees tags across x 0.32-0.62, y +-0.25 past the fingers;
# 0.55 / 0.67 is out of reach pointing down. VERIFY: jog there in the Web App, check all 4 tags are in view.
LOOK_FLANGE_XYZ = (0.40, 0.0, 0.62)
LOOK_THETA = (180.0, 0.0, 90.0)    # Kortex Euler (deg, extrinsic XYZ): pointing straight down
LOOK_SETTLE_S = 0.5                # wait at the look pose before the frame (no motion blur)
LOCATE_SKIP_FRAMES = 5             # frames to drop after a locate request (the RTSP stream lags a few frames)
TABLE_TAGS = {                     # tag36h11 id -> tag centre (x, y). MEASURE ON SITE (README Phase 1.4)
    0: (0.35, -0.20),
    1: (0.35, 0.20),
    2: (0.60, 0.20),
    3: (0.60, -0.20),
}
TAG_SIZE_M = 0.05
TABLE_Z = 0.0                      # table surface height in the base frame (used only while TABLE_TOUCH_Z is empty)
# How the table height is really set (README Phase 1.6): close the gripper, point it straight down, jog until the
# fingertips just pinch a sheet of paper on each tag, and read the flange z (tf2_echo base_link end_effector_link).
# A plane through these 4 heights replaces TABLE_Z: it follows a tilted table, and since the same arm measured it,
# ruler error in FINGERTIP_CLOSED_M and the arm's own z offset cancel out of the grasp height.
TABLE_TOUCH_Z = {}                 # tag id -> flange z (m) with the closed fingertips touching the mat there
SHAFT_MIN_VAL = 140                # shaft = bright (HSV V) and unsaturated; a lit dark mat stays below this
TOOL_MIN_AREA_PX = 800             # smaller blobs are noise
SHAFT_MAX_SAT = 70                 # shaft = low saturation (metal); handle = its TOOLS colour
GRIP_MIN_SAT, GRIP_MIN_VAL = 90, 60    # find_tools: any colourful grip (black grips vanish on a dark mat)
# Who names each found tool: "gemma" (numbered marks + TOOL_DESCRIPTIONS) or "colour" (TOOLS HSV ranges)
VISION_NAMER = "gemma"
VISION_GEMMA_MODEL = os.environ.get("EDGEHAND_VISION_MODEL", GEMMA_MODEL)   # E4B .litertlm (vision encoder)
TOOL_RADIUS_M = 0.012              # grasp height above the table (shaft lies on the handle's radius)
GRASP_FROM_FERRULE_M = 0.02        # grasp the shaft this far from where the handle ends

# ---- Motion (MoveIt 2 via ros2_kortex) ----
PLANNING_GROUP = "manipulator"
BASE_FRAME = "base_link"
EE_LINK = "end_effector_link"      # tip of the manipulator group = the flange
# ---- Never hit the table ----
# The 2F-85 fingertips move ~13 mm further out as they close, so the lowest the gripper ever gets is its closed
# fingertips. Every pick puts them TABLE_CLEARANCE_M above the table, and handover.plan_targets refuses any target
# that would go lower. Sim: the lowest gripper point at the grasp is 0.7 mm below this (pad shape), and picks work
# the same from 4 to 10 mm. Above ~6 mm the pads close on the shaft with only their bottom edge.
FINGERTIP_CLOSED_M = 0.173         # flange -> closed fingertips (Menagerie 2F-85; ruler on the real one, Phase 1.6)
TABLE_CLEARANCE_M = 0.006          # closed fingertips above the table at the bottom of every pick
# Flange -> the point between the pads that goes on the shaft axis, along tool z (0.167)
TCP_OFFSET_M = FINGERTIP_CLOSED_M + TABLE_CLEARANCE_M - TOOL_RADIUS_M
AIR_GAP_M = 0.0                    # first runs on the real arm: 0.03 = every pick closes 3 cm above the tools
DESCEND_SLOW_M = 0.03              # the last 3 cm down to a grasp run at DESCEND_SLOW_SCALE
DESCEND_SLOW_SCALE = 0.05          # (a wrong table height then touches gently, with time to hit the E-stop)
FINGERS_ALONG = "x"                # flange axis the 2F-85 fingers close along (check in RViz, Phase 1.6)
HOVER_M = 0.12                     # approach / lift height above the grasp
VEL_SCALE, ACC_SCALE = 0.2, 0.2
GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.8
GRIPPER_MAX_EFFORT = 50.0
# Missed grasp = after the lift the fingers closed past this (nothing between them). Sim: 0.745 on a 6 mm shaft,
# 0.782 on nothing. MEASURE (Phase 1.6): close on nothing and on the thinnest shaft, set this between the two.
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"    # in /joint_states (VERIFY the name on the real driver)
GRIPPER_EMPTY_POS = 0.765
PICK_ATTEMPTS = 3                  # a miss -> back to the look pose, look again, pick again
HANDOVER_XYZ = (0.45, 0.30, 0.35)  # where the tool is presented (operator's chest height, not face)
OPERATOR_DIR = (0.0, 1.0, 0.0)     # unit vector, base frame: the handle points this way
WORKSPACE = {"x": (0.15, 0.80), "y": (-0.55, 0.55), "z": (0.0, 0.70)}
# MoveIt only avoids what is in its planning scene. handover_node adds the table and a keep-out box this tall over
# the tag rectangle (the tools), so free moves never sweep the tray; only the vertical descend/lift enter it.
TRAY_KEEPOUT_H = 0.06
BASE_HOLE_M = 0.08                 # the scene table leaves this much (half-width) around the base free (sim base radius 0.046)

# ---- Release: tug detection on joint torques (/joint_states effort, N*m) ----
TUG_JOINTS = ["joint_5", "joint_6", "joint_7"]   # wrist joints feel a pull on the tool most
TUG_TORQUE_NM = 1.5                # |torque - baseline| over these joints (L2) to count as a tug
TUG_SAMPLES = 5                    # consecutive samples over threshold
SETTLE_S = 0.8                     # wait after arriving at the handover pose before taking the baseline

# ---- Pi link (UDP, no ROS on the Pi) ----
PI_UDP_PORT = 5005
WATCHDOG_S = 0.5                   # no heartbeat for this long -> /edgehand/stop
