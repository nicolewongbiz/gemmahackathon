"""Measure the tag positions (TABLE_TAGS), the table heights (TABLE_TOUCH_Z) and the look pose with the arm itself.
Read-only Kortex API: it never commands motion. You jog the arm (Web App or joystick); it reads the pose on ENTER.

    python -m edgehand.measure_tags            tags + table heights, one tag at a time (README Phase 1.4 / 1.6)
    python -m edgehand.measure_tags --look     print the current pose as LOOK_FLANGE_XYZ / LOOK_THETA
    python -m edgehand.measure_tags --once     print what it reads right now and exit
    python -m edgehand.measure_tags --gripper  finger position closed on nothing vs on the thinnest shaft
                                               -> GRIPPER_EMPTY_POS (close the gripper yourself; this only reads)

For each tag: close the gripper, point it straight down, and jog until the closed fingertips just pinch a sheet of
paper on the tag's centre. The fingertip x, y is the tag centre; the flange z is that tag's TABLE_TOUCH_Z.
"""
import sys
import numpy as np
from edgehand import config
from edgehand.geometry import kortex_R


class Arm:
    """Read-only Kortex session."""
    def __init__(self, ip=config.ROBOT_IP):
        from kortex_api.TCPTransport import TCPTransport
        from kortex_api.RouterClient import RouterClient
        from kortex_api.SessionManager import SessionManager
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
        from kortex_api.autogen.messages import Session_pb2
        self.transport = TCPTransport()
        router = RouterClient(self.transport, lambda e: print("kortex error:", e))
        self.transport.connect(ip, 10000)
        info = Session_pb2.CreateSessionInfo()
        info.username, info.password = "admin", "admin"
        info.session_inactivity_timeout, info.connection_inactivity_timeout = 600000, 2000
        self.session = SessionManager(router)
        self.session.CreateSession(info)
        self.base = BaseClient(router)
        self.router = router
        # the Kortex tool frame sits this far along tool z from the flange (Web App -> Configurations -> Tool)
        self.tool_z = ControlConfigClient(router).GetToolConfiguration().tool_transform.z

    def read(self):
        """(flange position, R, fingertip position, Kortex tool pose, theta) right now."""
        p = self.base.GetMeasuredCartesianPose()
        theta = (p.theta_x, p.theta_y, p.theta_z)
        R = kortex_R(*theta)
        flange = np.array([p.x, p.y, p.z]) - R[:, 2] * self.tool_z
        tip = flange + R[:, 2] * config.FINGERTIP_CLOSED_M
        return flange, R, tip, (p.x, p.y, p.z), theta

    def gripper_pct(self):
        """Robotiq finger position, 0 (open) .. 100 (closed), from the base's cyclic feedback (UDP session)."""
        from kortex_api.UDPTransport import UDPTransport
        from kortex_api.RouterClient import RouterClient
        from kortex_api.SessionManager import SessionManager
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        from kortex_api.autogen.messages import Session_pb2
        udp = UDPTransport()
        router = RouterClient(udp, lambda e: None)
        udp.connect(config.ROBOT_IP, 10001)
        info = Session_pb2.CreateSessionInfo()
        info.username, info.password = "admin", "admin"
        info.session_inactivity_timeout, info.connection_inactivity_timeout = 60000, 2000
        sm = SessionManager(router)
        sm.CreateSession(info)
        try:
            return BaseCyclicClient(router).RefreshFeedback().interconnect.gripper_feedback.motor[0].position
        finally:
            sm.CloseSession()
            udp.disconnect()

    def close(self):
        self.session.CloseSession()
        self.transport.disconnect()


def tilt_deg(R):
    """How far tool z is from pointing straight down."""
    return float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))


def main():
    arm = Arm()
    try:
        if "--gripper" in sys.argv:
            input("Close the gripper on NOTHING (Web App / joystick), wait until it stops. ENTER to read...")
            empty = arm.gripper_pct()
            input("Now open it, put the THINNEST shaft between the pads, close it on the shaft. ENTER to read...")
            held = arm.gripper_pct()
            mid = (empty + held) / 2
            # Kortex 0..100 % maps to the ROS finger joint 0..0.8 rad
            print(f"closed on nothing: {empty:.1f} %  ({0.8 * empty / 100:.3f} rad)")
            print(f"closed on shaft:   {held:.1f} %  ({0.8 * held / 100:.3f} rad)")
            print(f"GRIPPER_EMPTY_POS = {0.8 * mid / 100:.3f}   (midpoint {mid:.1f} %; past this after the lift = missed grasp)")
            if empty - held < 5:
                print("WARNING: less than 5 % between empty and holding: a miss would be hard to tell from a grasp")
            return
        if "--once" in sys.argv or "--look" in sys.argv:
            flange, R, tip, tool, theta = arm.read()
            print(f"Kortex tool pose {np.round(tool, 4)} theta {np.round(theta, 1)} (tool frame z {arm.tool_z:.3f})")
            print(f"flange {np.round(flange, 4)}, fingertips {np.round(tip, 4)}, {tilt_deg(R):.1f} deg from straight down")
            print(f"gripper {arm.gripper_pct():.1f} % closed")
            if "--look" in sys.argv:
                print(f"\nLOOK_FLANGE_XYZ = ({flange[0]:.3f}, {flange[1]:.3f}, {flange[2]:.3f})")
                print(f"LOOK_THETA = ({theta[0]:.1f}, {theta[1]:.1f}, {theta[2]:.1f})")
            return
        tags, touch = {}, {}
        for tid in config.TABLE_TAGS:
            while True:
                input(f"\nTag {tid}: closed fingertips pinching paper on its centre, gripper straight down. ENTER to read...")
                flange, R, tip, _, _ = arm.read()
                t = tilt_deg(R)
                print(f"  fingertips at x {tip[0]:.4f} y {tip[1]:.4f}, flange z {flange[2]:.4f}, tilt {t:.1f} deg")
                if t > 3.0 and input("  More than 3 deg off vertical: the x, y will be off. Keep it anyway? [y/N] ").lower() != "y":
                    continue
                tags[tid], touch[tid] = (round(tip[0], 4), round(tip[1], 4)), round(flange[2], 4)
                break
        print("\nPaste into edgehand/config.py:\n")
        print("TABLE_TAGS = {")
        for tid, (x, y) in tags.items():
            print(f"    {tid}: ({x}, {y}),")
        print("}")
        print("TABLE_TOUCH_Z = {" + ", ".join(f"{tid}: {z}" for tid, z in touch.items()) + "}")
        zs = list(touch.values())
        print(f"\ntable heights span {1000 * (max(zs) - min(zs)):.1f} mm across the tags (tilt / unevenness)")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
