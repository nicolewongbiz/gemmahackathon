"""Every EdgeHand setting. Edit on site. Distances in metres, robot base frame (x forward, y left, z up)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # EdgeHand/ (build with --symlink-install so this resolves)

# ---- Tools: class -> handle colour (OpenCV HSV, hue 0-180). Red wraps around 0, so it has two ranges ----
# The real tray (Oct 24): four precision screwdrivers. Two have BLACK handles with a small colour accent, and the
# clear one is only identifiable by its blue collar, so a class is the dominant saturated colour on the grip
# (vision.colour_class), not "half the handle". Hue is OpenCV's 0-180. Measured on relay frames of the real tools
# on white paper at the look pose (hue 10/50/90 percentiles): green 72/74/77, red cap 0/1 (wraps to 179),
# yellow collar 18/20/21, blue collar 105/108/112; paper, shafts and black handles have saturation < 20.
TOOLS = {
    "phillips": [((0, 90, 60), (8, 255, 255)), ((170, 90, 60), (180, 255, 255))],    # black handle, RED cap
    "flathead": [((12, 80, 70), (28, 255, 255))],                                      # black handle, YELLOW collar
    "torx":     [((60, 70, 40), (90, 255, 255))],                                      # translucent GREEN handle
    "clear":    [((95, 80, 30), (125, 255, 255))],                                     # clear handle, BLUE collar
}
ACCENT_MIN_FRAC = 0.02             # a class needs at least this share of the grip's pixels in its colour (a cap/collar)
DEFAULT_TOOL = "phillips"          # what the Pi REQUEST button asks for
# What each tool looks like from above, in words, for Gemma naming the numbered tools (VISION_NAMER = "gemma").
# Gemma can't see a screwdriver tip from the look pose, so describe what IS visible: the handle.
TOOL_DESCRIPTIONS = {
    "phillips": "precision screwdriver with a black handle and a red cap on the end",
    "flathead": "precision screwdriver with a black handle and a yellow collar",
    "torx": "precision screwdriver with a dark green translucent handle",
    "clear": "precision screwdriver with a clear translucent handle and a blue collar",
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
# "url"   = an MJPEG/RTSP stream, e.g. edgehand/cam_relay.py on the Jetson serving the RealSense over the USB link
CAMERA_SOURCE = "url"
CAMERA_URL = "http://192.168.55.1:8089/stream.mjpg"
CAMERA_USB_NAME = "RealSense"      # pick the USB camera whose name contains this (macOS; needs pyobjc-framework-AVFoundation)
CAMERA_USB_INDEX = 0               # fallback OpenCV index (it shifts when an iPhone etc. connects): camera --list
CAMERA_RTSP = f"rtsp://{ROBOT_IP}/color"
FRAME_W, FRAME_H = 1280, 720
CAMERA_TOPIC = "/camera/camera/color/image_raw"   # for CAMERA_SOURCE = "topic"
# The camera rides on the wrist, so frames are only used from the look pose: gripper straight down, high over the
# tray. Flange (end_effector_link) position + Kortex theta_x/y/z. From the GemmaHackathon sim: tool frame at
# (0.40, 0, 0.50) = flange at z 0.62 reaches and sees tags across x 0.32-0.62, y +-0.25 past the fingers;
# 0.55 / 0.67 is out of reach pointing down. VERIFY: jog there in the Web App, check all 4 tags are in view.
# Measured on the real cell (Oct 24, RealSense on the wrist): all 4 tags in view and CALIBRATED. Web App tool pose
# (0.266, -0.028, 0.447) theta (174.3, -7.2, 89.8) = this flange pose (tool frame z 0.12). Slightly tilted, which the
# tag homography absorbs. 0.5 m above the table, so no clearance concern. The sim's camera mount differs, so sim trials
# check this pose still sees the tags there too.
LOOK_FLANGE_XYZ = (0.254, -0.043, 0.565)
LOOK_THETA = (174.3, -7.2, 89.8)   # Kortex Euler (deg, extrinsic XYZ): roughly straight down
LOOK_SETTLE_S = 0.5                # wait at the look pose before the frame (no motion blur)
LOCATE_SKIP_FRAMES = 5             # frames to drop after a locate request (the RTSP stream lags a few frames)
TABLE_TAGS = {                     # tag36h11 id -> tag centre (x, y). MEASURE ON SITE (README Phase 1.4)
    0: (0.35, -0.20),
    1: (0.35, 0.20),
    2: (0.60, 0.20),
    3: (0.60, -0.20),
}
TAG_SIZE_M = 0.05
CALIB_MIN_TAGS = 2                 # visible TABLE_TAGS needed to calibrate (fewer: reuse the last good fit)
# Worst tag-corner mismatch a fit may have. Right TABLE_TAGS: ~0.6 mm. One tag 3 cm off: 5 mm. Tags 0.3 m apart
# configured as 0.25 x 0.4 m: 10 mm. Above this the tags don't agree with TABLE_TAGS, so don't calibrate.
CALIB_MAX_ERR_M = 0.004
TRAY_MARGIN_M = 0.02               # tools are only looked for within the tag rectangle plus this (desk clutter beyond)
TABLE_Z = 0.0                     # table surface height in the base frame (used only while TABLE_TOUCH_Z is empty)
# How the table height is really set (README Phase 1.6): close the gripper, point it straight down, jog until the
# fingertips just pinch a sheet of paper on each tag, and read the flange z (tf2_echo base_link end_effector_link).
# A plane through these 4 heights replaces TABLE_Z: it follows a tilted table, and since the same arm measured it,
# ruler error in FINGERTIP_CLOSED_M and the arm's own z offset cancel out of the grasp height.
TABLE_TOUCH_Z = {}                 # tag id -> flange z (m) with the closed fingertips touching the mat there
TOUCH_SPREAD_MAX_M = 0.02          # touch heights further apart than this, or a plane tilted more than this, are
TABLE_TILT_MAX_DEG = 3.0           # refused (a typo would move the table plane under the clearance check)
SHAFT_MIN_VAL = 140                # shaft = bright (HSV V) and unsaturated; a lit dark mat stays below this
TOOL_MIN_AREA_PX = 800             # smaller blobs are noise
SHAFT_MAX_SAT = 70                 # shaft = low saturation (metal); handle = its TOOLS colour
GRIP_MIN_SAT, GRIP_MIN_VAL = 90, 60    # find_tools: any colourful grip (black grips vanish on a dark mat)
# ---- Segmentation (vision.py): "contrast" = anything unlike the mat, handle/shaft from the width profile (any handle
# colour; needs a plain mat the tools stand out from, e.g. white paper under dark screwdrivers). "colour" = the
# original colourful-grip + grey-shaft method for coloured handles on a dark mat.
SEGMENTATION = "contrast"
CONTRAST_MIN = 45                  # max |BGR - mat| (0-255) for a pixel to count as an object; shadows stay below
WARP_PX_PER_M = 2000               # the top-down view's scale: 2 px per mm
WARP_MARGIN_M = 0.08               # the top-down view reaches this far past the tags (handles overhang them)
TOOL_MIN_AREA_M2 = 3e-4            # 3 cm^2; a precision screwdriver is ~10
MIN_ELONGATION = 2.5               # blob length / handle width; rounder blobs aren't tools
MIN_SHAFT_BEYOND_GRASP_M = 0.015   # shaft needed past the grasp point so the fingers have something to close on
# Who names each found tool: "gemma" (numbered marks + TOOL_DESCRIPTIONS) or "colour" (TOOLS HSV ranges)
VISION_NAMER = "gemma"
VISION_GEMMA_MODEL = os.environ.get("EDGEHAND_VISION_MODEL", GEMMA_MODEL)   # E4B .litertlm (vision encoder)
# Where on the tool the fingers close. Precision screwdrivers (the real tray): the shaft is 3-4 mm thick, so lying
# flat its top is below the fingertips' clearance and the pads close above it (seen on the cell: brushed the top).
# "handle": GRASP_FROM_FERRULE_M into the handle from the ferrule; the operator gets the handle's free end.
# "shaft": the original rule, GRASP_FROM_FERRULE_M down the shaft, for fat-shafted tools (sim, synthetic tests).
GRASP_SIDE = "handle"
TOOL_RADIUS_M = 0.005              # grasp height above the table: radius of the part gripped (precision handle ~10 mm)
GRASP_FROM_FERRULE_M = 0.02        # grasp this far from where the handle ends (on the GRASP_SIDE)

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
TABLE_CLEARANCE_M = 0.004          # closed fingertips above the table at the bottom of every pick (plane from touches)
# Flange -> the point between the pads that goes on the shaft axis, along tool z (0.167)
TCP_OFFSET_M = FINGERTIP_CLOSED_M + TABLE_CLEARANCE_M - TOOL_RADIUS_M
AIR_GAP_M = 0.0                    # first runs on the real arm: 0.03 = every pick closes 3 cm above the tools
DESCEND_SLOW_M = 0.03              # the last 3 cm down to a grasp run at DESCEND_SLOW_SCALE
DESCEND_SLOW_SCALE = 0.05          # (a wrong table height then touches gently, with time to hit the E-stop)
FINGERS_ALONG = "x"                # flange axis the 2F-85 fingers close along (check in RViz, Phase 1.6)
HOVER_M = 0.12                     # approach / lift height above the grasp
VEL_SCALE, ACC_SCALE = 0.2, 0.2
# ---- Motion without ROS: the Kortex API straight from this computer (edgehand/kortex_robot.py, run_kortex.py) ----
# The arm's own Cartesian controller, no collision planning: free moves rise first, translate at the higher of the
# two heights, then descend, so lateral motion never happens low over the tray. Same clearance checks as MoveIt.
KORTEX_LINEAR_SPEED_MPS = 0.05     # at VEL_SCALE; move_line's scale= (the slow last 3 cm) scales it down
TRAVEL_MIN_FINGERTIP_M = 0.15      # every sideways leg of a free move keeps the closed fingertips at least this
                                   # high above the table (tools are < 6 cm): the "safe path" floor
KORTEX_ANGULAR_SPEED_DPS = 15.0
KORTEX_ACTION_TIMEOUT_S = 120.0    # a reach is also finished when the measured pose matches, and
KORTEX_STALL_S = 6.0               # stopped when the pose hasn't changed for this long (crawling / stuck)
KORTEX_READY_WAIT_S = 1.0          # after a refused action: how long to wait for a settled state before logging it
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

# ---- Event link (edgehand/link.py): Jetson / Pi / phase player -> laptop, JSON over UDP ----
EVENT_UDP_PORT = 5006
# Surgical phase (a label in the phases CSV) -> the tool class to fetch when that phase STARTS (a change only, not
# every row). Phases not listed fetch nothing. The names are the hand annotation in data/phases/clippingdemo.csv;
# the tools are stand-ins for the demo tray (the clip's real instruments aren't on it): edit freely, or add to it
# per run with phase_player --map "Phase=tool". Every value must be a key of TOOLS.
PHASE_TOOLS = {
    "Parent vessel exposure": "phillips",
    "Clipping": "torx",
    "Dome exposure": "flathead",
}

# Stand-in instruments for the surgical demo (edgehand/assistant.py): tool class -> (colour on the tray, surgical role,
# when it's used). Gemma reads this to choose an instrument from the phase context; edit freely.
INSTRUMENTS = {
    "torx":     ("green",  "clip applier",    "placing and adjusting aneurysm clips"),
    "phillips": ("orange", "suction",         "clearing blood and fluid to keep the field visible, used throughout"),
    "flathead": ("yellow", "bipolar forceps", "coagulating small vessels during exposure"),
    "clear":    ("blue",   "micro-dissector", "separating arachnoid and tissue to expose the parent vessel, dome and neck"),
}
# The surgical assistant: phase events from the Jetson's video node arrive on this port (run_kortex owns
# EVENT_UDP_PORT). During the demo the assistant OWNS the laptop GPU (one GPU LiteRT process at a time): 2-8 s per
# decision on GPU vs 16-26 s on CPU, and run_kortex names tray tools by handle colour (--namer colour) meanwhile.
PHASE_UDP_PORT = 5007
JETSON_HOST = "192.168.55.1"
SURGICAL_FEED_URL = f"http://{JETSON_HOST}:8090/stream.mjpg"
ASSISTANT_GEMMA_GPU = True
# Phase names Gemma may give as its own read of a frame (same words as the annotated timeline, so they compare)
SURGICAL_PHASES = ["Brain exposure", "Parent vessel exposure", "Dome exposure", "Clipping"]
ASSISTANT_CONFIRM = True           # speak each recommendation and fetch only on a spoken "yes" (or a named instrument)
CONFIRM_SECONDS = 3.5              # how long it listens for the answer
ASSISTANT_IMAGE_PX = 512             # longest side of the surgical frame Gemma sees (smaller = faster)

# ---- Voice (edgehand/voice_node.py, on the laptop: its mic -> Gemma -> request over the event link) ----
VOICE_MIC_NAME = "MacBook Pro Microphone"   # ffmpeg/AVFoundation input picked by name (indexes shift with an iPhone)
VOICE_SECONDS = 4.0                # push-to-talk recording length
VOICE_SPEAK = True                 # speak Gemma's short reply with macOS `say` (offline)

# ---- Site calibration: data/calibration.json from `python -m edgehand.measure_tags` overrides TABLE_TAGS,
# TABLE_TOUCH_Z, LOOK_FLANGE_XYZ, LOOK_THETA, GRIPPER_EMPTY_POS (edgehand/calibration.py). Moving the cell = re-measure.
from edgehand.calibration import apply as _apply_calibration    # noqa: E402
_apply_calibration(globals())
