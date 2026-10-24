# The MuJoCo scenes have their own geometry: don't let the real cell's data/calibration.json move the sim's tags.
import os
os.environ.setdefault("EDGEHAND_CALIBRATION", "none")

# the sim's screwdrivers have fat shafts and their grasp site is on the shaft: keep the original rule here
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "ros2_ws" / "src" / "edgehand"))
from edgehand import config as _config  # noqa: E402
_config.GRASP_SIDE, _config.TOOL_RADIUS_M = "shaft", 0.012
_config.TCP_OFFSET_M = _config.FINGERTIP_CLOSED_M + _config.TABLE_CLEARANCE_M - _config.TOOL_RADIUS_M   # re-derive
