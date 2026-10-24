"""Measure TABLE_TAGS with the arm itself (~2 mm) instead of a tape (~1-2 cm), and write them to config.py.
    python -m scripts.measure_tags      (or double-click measure_tags.bat)
For each corner tag: jog the gripper (pointing down) in the Kinova Web App until the fingertip centre
is over the tag centre, then press Enter here. This script only reads the pose; it never moves the arm,
so the Web App can stay open while you use it. Close the Web App before running click_pick.
"""
import re
import config
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2

t = TCPTransport()
r = RouterClient(t, lambda e: print("[kortex]", e))
t.connect(config.ROBOT_IP, 10000)
info = Session_pb2.CreateSessionInfo()
info.username, info.password = config.ROBOT_USER, config.ROBOT_PASS
info.session_inactivity_timeout, info.connection_inactivity_timeout = 600000, 2000
s = SessionManager(r)
s.CreateSession(info)
base = BaseClient(r)

measured = {}
try:
    for tid in config.TABLE_TAGS:
        input(f"Jog the fingertip onto the centre of tag {tid}, then press Enter ...")
        p = base.GetMeasuredCartesianPose()
        measured[tid] = (round(p.x, 3), round(p.y, 3))
        print(f"  tag {tid}: x={p.x:.3f} y={p.y:.3f}  (z={p.z:.3f}: the table is about this high under the fingertip)")
finally:
    s.CloseSession()
    t.disconnect()

block = "TABLE_TAGS = {\n" + "".join(f"    {k}: ({x}, {y}),\n" for k, (x, y) in measured.items()) + "}"
path = config.ROOT / "config.py"
src = path.read_text(encoding="utf-8")
new, n = re.subn(r"TABLE_TAGS = \{.*?\n\}", block, src, count=1, flags=re.S)
if n != 1:
    print("Could not find TABLE_TAGS in config.py; paste this in by hand:\n" + block)
else:
    path.write_text(new, encoding="utf-8")
    print("Saved to config.py:\n" + block)
print("Next: calibrate.bat, click a tag centre and check the printed x, y match.")
