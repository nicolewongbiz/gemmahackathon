"""Check the Kinova Gen3 connection. KEEP THE E-STOP IN YOUR HAND.

    python scripts/check_robot.py            # connect + print pose only (no motion)
    python scripts/check_robot.py --move     # also: gripper open/close, up 3 cm and back
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

config.MOCK_ARM = False
ap = argparse.ArgumentParser()
ap.add_argument("--move", action="store_true")
a = ap.parse_args()

from arm import KinovaArm  # noqa: E402

arm = KinovaArm()
try:
    p = arm.pose()
    print("Connected. Tool pose:", {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()})
    if a.move:
        input("Press ENTER to open/close the gripper and move up 3 cm and back (Ctrl+C to cancel)...")
        arm.gripper(0.0); arm.gripper(0.6); arm.gripper(0.0)
        arm.move_to(p["x"], p["y"], p["z"] + 0.03, p["tx"], p["ty"], p["tz"])
        arm.move_to(p["x"], p["y"], p["z"], p["tx"], p["ty"], p["tz"])
        print("Motion test OK.")
finally:
    arm.close()
