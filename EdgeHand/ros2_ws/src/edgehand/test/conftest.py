# Tests use config.py's defaults, not the real cell's data/calibration.json (test_real_frames loads that itself).
import os
os.environ["EDGEHAND_CALIBRATION"] = "none"
