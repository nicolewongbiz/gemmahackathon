# The MuJoCo scenes have their own geometry: don't let the real cell's data/calibration.json move the sim's tags.
import os
os.environ.setdefault("EDGEHAND_CALIBRATION", "none")
