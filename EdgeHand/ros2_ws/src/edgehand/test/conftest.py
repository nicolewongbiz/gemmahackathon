# Tests use config.py's defaults, not the real cell's data/calibration.json (test_real_frames loads that itself).
import os
os.environ["EDGEHAND_CALIBRATION"] = "none"

# The synthetic frames and the MuJoCo tools have fat shafts (the sim's grasp site is on the shaft): keep the original
# rule there. The real tray's precision screwdrivers use GRASP_SIDE = "handle" (config.py).
from edgehand import config as _config  # noqa: E402
_config.GRASP_SIDE, _config.TOOL_RADIUS_M = "shaft", 0.012
_config.TCP_OFFSET_M = _config.FINGERTIP_CLOSED_M + _config.TABLE_CLEARANCE_M - _config.TOOL_RADIUS_M   # re-derive
