"""The handover Robot interface on the Kortex API, no ROS: the arm's own Cartesian controller (reach_pose).

    move_free  -> up to three reach_pose moves: rise, translate at the higher height, descend (never low and sideways)
    move_line  -> one reach_pose, speed scaled (the slow last 3 cm of a descent)
    gripper    -> SendGripperCommand, then wait for the fingers to stop
    grip_position, efforts -> the base's cyclic feedback over UDP (finger %, actuator torques)

There is no planning scene here. What keeps the gripper off the table is handover.plan_targets (refuses targets that
put the closed fingertips below TABLE_CLEARANCE_M) and a second, independent check of every target in this file.
All poses are the FLANGE (end_effector_link); Kortex wants the TOOL frame, tool_z further along tool z.
"""
import threading, time
import numpy as np
from edgehand import config
from edgehand.geometry import kortex_R, look_pose, table_clearance


def kortex_theta(R):
    """Rotation matrix -> Kortex theta_x, theta_y, theta_z (deg), the inverse of geometry.kortex_R (R = Rz Ry Rx)."""
    R = np.asarray(R, float)
    tx = np.arctan2(R[2, 1], R[2, 2])
    ty = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    tz = np.arctan2(R[1, 0], R[0, 0])
    return tuple(float(np.degrees(v)) for v in (tx, ty, tz))


def too_low(p, R):
    """A reason string if this flange pose puts the closed fingertips under the clearance, else None."""
    c = table_clearance(p, R)
    if c < config.TABLE_CLEARANCE_M - 1e-4:
        return f"fingertips {c * 1000:.1f} mm above the table (minimum {config.TABLE_CLEARANCE_M * 1000:.1f})"
    return None


def already_there(p0, R0, p1, R1, pos_tol=0.003, ang_tol_deg=1.0):
    """Within 3 mm and 1 degree: Kortex has nothing to do (and may abort a zero-length reach)."""
    if np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)) > pos_tol:
        return False
    cos = (np.trace(np.asarray(R0).T @ np.asarray(R1)) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1))) < ang_tol_deg


def free_waypoints(p0, p1, R1, lift=0.0):
    """Flange waypoints for a free move p0 -> p1: rise to max(z0, z1) + lift, translate there, descend. Orientation
    changes during the first leg. Legs shorter than 2 mm are dropped."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    top = max(p0[2], p1[2]) + lift
    pts = [np.array([p0[0], p0[1], top]), np.array([p1[0], p1[1], top]), p1]
    out, last = [], p0
    for p in pts:
        if np.linalg.norm(p - last) > 0.002:
            out.append(p)
            last = p
    return out or [p1]


class KortexRobot:
    def __init__(self, ip=config.ROBOT_IP, log=print):
        from kortex_api.TCPTransport import TCPTransport
        from kortex_api.UDPTransport import UDPTransport
        from kortex_api.RouterClient import RouterClient
        from kortex_api.SessionManager import SessionManager
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
        from kortex_api.autogen.messages import Base_pb2, Session_pb2
        self.log, self.B = log, Base_pb2
        self._sessions = []
        for transport, port in ((TCPTransport(), 10000), (UDPTransport(), 10001)):
            router = RouterClient(transport, lambda e: log(f"kortex error: {e}"))
            transport.connect(ip, port)
            info = Session_pb2.CreateSessionInfo()
            info.username, info.password = "admin", "admin"
            info.session_inactivity_timeout, info.connection_inactivity_timeout = 600000, 2000
            sm = SessionManager(router)
            sm.CreateSession(info)
            self._sessions.append((sm, transport))
            if port == 10000:
                self.base = BaseClient(router)
                self.tool_z = ControlConfigClient(router).GetToolConfiguration().tool_transform.z
            else:
                self.cyclic = BaseCyclicClient(router)
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)
        self._stopped = False
        self._lock = threading.Lock()
        state = Base_pb2.ArmState.Name(self.base.GetArmState().active_state)
        log(f"kortex: connected to {ip}, tool frame z {self.tool_z:.3f} m, arm state {state}")
        if state != "ARMSTATE_SERVOING_READY":
            log("kortex: WARNING the arm isn't SERVOING_READY (fault, E-stop pressed, or another controller): "
                "clear it in the Web App before moving")
        p, R = self.flange()
        log(f"kortex: flange now {np.round(p, 3).tolist()}, theta {np.round(kortex_theta(R), 1).tolist()}")

    # ---- the Kortex calls, kept small so tests can replace them ----
    def _reach(self, p_flange, R, speed_mps):
        """One reach_pose to a flange pose. True when the action ended normally."""
        if r := too_low(p_flange, R):
            self.log(f"REFUSED move: {r}")
            return False
        B = self.B
        a = B.Action()
        a.name = "edgehand"
        tool = np.asarray(p_flange, float) + np.asarray(R)[:, 2] * self.tool_z
        t = a.reach_pose.target_pose
        t.x, t.y, t.z = map(float, tool)
        t.theta_x, t.theta_y, t.theta_z = kortex_theta(R)
        s = a.reach_pose.constraint.speed
        s.translation = float(min(speed_mps, 0.1))
        s.orientation = float(config.KORTEX_ANGULAR_SPEED_DPS)
        done, result = threading.Event(), {}
        self.log(f"kortex: reach tool ({t.x:.3f}, {t.y:.3f}, {t.z:.3f}) theta ({t.theta_x:.1f}, {t.theta_y:.1f}, "
                 f"{t.theta_z:.1f}) at {s.translation:.3f} m/s")

        def cb(n):
            if n.action_event in (B.ACTION_END, B.ACTION_ABORT):
                result["event"] = n.action_event
                result["why"] = str(getattr(n, "abort_details", "")).strip()
                done.set()
        h = self.base.OnNotificationActionTopic(cb, B.NotificationOptions())
        try:
            self._stopped = False
            self.base.ExecuteAction(a)
            if not done.wait(config.KORTEX_ACTION_TIMEOUT_S):
                self.log("kortex: action timed out; stopping")
                self.base.Stop()
                return False
        finally:
            self.base.Unsubscribe(h)
        if result.get("event") != B.ACTION_END:
            why = "stop" if self._stopped else (result.get("why") or "no details: unreachable pose, fault, or a "
                                                 "zero-length move")
            self.log(f"kortex: move aborted ({why}); arm state "
                     f"{B.ArmState.Name(self.base.GetArmState().active_state)}")
            return False
        return True

    def _feedback(self):
        return self.cyclic.RefreshFeedback()

    def flange(self):
        """Measured flange position and R."""
        p = self.base.GetMeasuredCartesianPose()
        R = kortex_R(p.theta_x, p.theta_y, p.theta_z)
        return np.array([p.x, p.y, p.z]) - R[:, 2] * self.tool_z, R

    # ---- the Robot interface ----
    def move_free(self, p, R):
        p0, R0 = self.flange()
        if already_there(p0, R0, p, R):
            self.log("kortex: already at the target")
            return False if too_low(p, R) else True
        for wp in free_waypoints(p0, p, R):
            if not self._reach(wp, R, config.KORTEX_LINEAR_SPEED_MPS):
                return False
        return True

    def move_line(self, p, R, collisions=True, scale=None):
        v = config.KORTEX_LINEAR_SPEED_MPS * ((scale / config.VEL_SCALE) if scale else 1.0)
        return self._reach(p, R, v)

    def gripper(self, pos):
        B = self.B
        cmd = B.GripperCommand()
        cmd.mode = B.GRIPPER_POSITION
        f = cmd.gripper.finger.add()
        f.finger_identifier, f.value = 1, float(np.clip(pos / config.GRIPPER_CLOSED, 0.0, 1.0))
        self.base.SendGripperCommand(cmd)
        # wait until the fingers stop moving (they stall early on a tool): same position for 0.3 s, 4 s at most
        last, still, t0 = None, 0.0, time.time()
        while time.time() - t0 < 4.0:
            time.sleep(0.05)
            cur = self.grip_position()
            if last is not None and abs(cur - last) < 0.004:
                still += 0.05
                if still >= 0.3:
                    return True
            else:
                still = 0.0
            last = cur
        return True

    def grip_position(self):
        """Finger position in the ROS joint's units (0 open .. GRIPPER_CLOSED), from the gripper's %."""
        pct = self._feedback().interconnect.gripper_feedback.motor[0].position
        return float(pct) / 100.0 * config.GRIPPER_CLOSED

    def efforts(self):
        fb = self._feedback()
        return {f"joint_{i + 1}": float(a.torque) for i, a in enumerate(fb.actuators)}

    def look(self):
        return self.move_free(*look_pose())

    def wait(self, seconds):
        time.sleep(seconds)

    def cancel(self):
        self._stopped = True
        self.base.Stop()

    def close(self):
        for sm, transport in self._sessions:
            try:
                sm.CloseSession()
                transport.disconnect()
            except Exception:
                pass
