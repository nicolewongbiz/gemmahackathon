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


# Gen3 7-DoF joint position limits (deg) for the limited joints; 1, 3, 5, 7 are continuous. From a joint at its
# limit (the Retract pose puts joint 4 at -147.8) the Cartesian planner refuses any reach_pose with INVALID_PARAM.
JOINT_LIMITS_DEG = {2: 128.9, 4: 147.8, 6: 120.3}
NEAR_LIMIT_DEG = 3.0


def near_joint_limit(joints_deg):
    """[(joint number, angle)] of limited joints within NEAR_LIMIT_DEG of a limit. joints_deg: 7 angles, any wrap."""
    out = []
    for j, lim in JOINT_LIMITS_DEG.items():
        a = (float(joints_deg[j - 1]) + 180.0) % 360.0 - 180.0
        if abs(a) >= lim - NEAR_LIMIT_DEG:
            out.append((j, round(a, 1)))
    return out


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


def free_waypoints(p0, p1, R1, R0=None, lift=0.0):
    """Flange (position, R) waypoints for a free move p0 -> p1: rise to max(z0, z1) + lift keeping the current
    orientation R0, turn to R1 up there, translate, descend. So the gripper never swings its fingertips down near the
    table and never moves sideways low. Legs that change nothing are dropped."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    R0 = np.asarray(R1 if R0 is None else R0, float)
    top = max(p0[2], p1[2]) + lift
    legs = [(np.array([p0[0], p0[1], top]), R0), (np.array([p0[0], p0[1], top]), R1),
            (np.array([p1[0], p1[1], top]), R1), (p1, R1)]
    out, last_p, last_R = [], p0, R0
    for p, R in legs:
        if np.linalg.norm(p - last_p) > 0.002 or not already_there(last_p, last_R, p, R):
            out.append((p, R))
            last_p, last_R = p, R
    return out or [(p1, R1)]


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
            # connection_inactivity_timeout is how long the arm tolerates SILENCE on the socket before closing it.
            # 2 s (Kinova's example value) broke the pipe while holding for the tug: all polling then goes over the
            # UDP feedback channel and the TCP channel goes quiet. wait() also keeps it alive.
            info.session_inactivity_timeout, info.connection_inactivity_timeout = 600000, 60000
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
        self.log(f"kortex: reach tool ({t.x:.3f}, {t.y:.3f}, {t.z:.3f}) theta ({t.theta_x:.1f}, {t.theta_y:.1f}, "
                 f"{t.theta_z:.1f}) at {s.translation:.3f} m/s")
        return self._execute(a)

    def _execute(self, a):
        """Run a Kortex action and wait for its end. True when it ended normally.
        Kortex replays the last action notification to a new subscriber (an old ABORT, say), so only an END/ABORT
        that arrives AFTER our own action's START counts; anything before is logged and ignored."""
        B = self.B
        done, result, started = threading.Event(), {}, threading.Event()

        def cb(n):
            ev = n.action_event
            code = getattr(n, "abort_details", 0)
            try:
                why = f"{B.SubErrorCodes.Name(code)} ({code})"
            except ValueError:
                why = str(code)
            if ev == B.ACTION_START:
                started.set()
            elif ev in (B.ACTION_END, B.ACTION_ABORT):
                if not started.is_set():
                    self.log(f"kortex: ignoring a stale {B.ActionEvent.Name(ev)} notification ({why}) from before "
                             "our action started")
                    return
                result["event"], result["why"] = ev, why
                done.set()
        # Only a fault (or an arm still initialising) blocks an action. Gripper commands leave the controller
        # reporting MANUALLY_CONTROLLED / PLAYING_SEQUENCE for a while, and Kortex still takes the next action then;
        # a refused action reports ABORT with a reason, which is handled below.
        state = self.base.GetArmState().active_state
        if state in (B.ARMSTATE_IN_FAULT, B.ARMSTATE_UNSPECIFIED, B.ARMSTATE_BASE_INITIALIZATION,
                     B.ARMSTATE_INITIALIZATION, B.ARMSTATE_MAINTENANCE):
            self.log(f"kortex: not moving, arm state is {B.ArmState.Name(state)}. {self._state_hint(state)}")
            return False
        h = self.base.OnNotificationActionTopic(cb, B.NotificationOptions())
        try:
            self._stopped = False
            self.base.ExecuteAction(a)
            if not started.wait(5.0):
                self.log("kortex: the action never reported START; stopping")
                self.base.Stop()
                return False
            if not done.wait(config.KORTEX_ACTION_TIMEOUT_S):
                self.log("kortex: action timed out; stopping")
                self.base.Stop()
                return False
        finally:
            self.base.Unsubscribe(h)
        if result.get("event") != B.ACTION_END:
            why = "stop" if self._stopped else (result.get("why") or "no details: unreachable pose, fault, or a "
                                                 "zero-length move")
            state = self._wait_ready(1.0)
            self.log(f"kortex: move aborted ({why}); arm state {B.ArmState.Name(state)}. {self._state_hint(state)}")
            return False
        return True

    def _wait_ready(self, timeout_s):
        """Poll the arm state until SERVOING_READY or the timeout; returns the last state seen."""
        B = self.B
        t0 = time.time()
        while True:
            state = self.base.GetArmState().active_state
            if state == B.ARMSTATE_SERVOING_READY or time.time() - t0 > timeout_s:
                return state
            self.wait(0.1)

    def _state_hint(self, state):
        B = self.B
        if state in (B.ARMSTATE_SERVOING_MANUALLY_CONTROLLED, B.ARMSTATE_SERVOING_PLAYING_SEQUENCE):
            return ("The controller reports another command in progress: a gripper command just sent, the Web App's "
                    "jog panel, or a gamepad.")
        if state == B.ARMSTATE_IN_FAULT:
            return "Clear the fault in the Web App (or the E-stop is pressed)."
        return "Wait for the current action to finish, or stop it in the Web App."

    def _feedback(self):
        return self.cyclic.RefreshFeedback()

    def joints(self):
        """Measured joint angles (deg), joint 1 first."""
        return [j.value for j in self.base.GetMeasuredJointAngles().joint_angles]

    def home(self):
        """Kinova's stored 'Home' preset: a joint-space move that unfolds the arm forward and up (tool ~0.58, 0,
        0.43), the way out of a folded pose like Retract, from which Cartesian moves are refused."""
        B = self.B
        req = B.RequestedActionType()
        req.action_type = B.REACH_JOINT_ANGLES
        for a in self.base.ReadAllActions(req).action_list:
            if a.name == "Home":
                self.log("kortex: Home (joint move) first")
                return self._execute(a)
        self.log("kortex: no stored 'Home' action on this arm")
        return False

    def unfold_if_needed(self):
        """True if the arm is free to take Cartesian moves (after a Home if a limited joint sat at its limit)."""
        at = near_joint_limit(self.joints())
        if not at:
            return True
        self.log(f"kortex: joint(s) at a limit {at} (folded, e.g. Retract): Cartesian moves would be refused")
        return self.home()

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
        for wp, wR in free_waypoints(p0, p, R, R0):
            if not self._reach(wp, wR, config.KORTEX_LINEAR_SPEED_MPS):
                return False
        return True

    def move_line(self, p, R, collisions=True, scale=None):
        v = config.KORTEX_LINEAR_SPEED_MPS * ((scale / config.VEL_SCALE) if scale else 1.0)
        return self._reach(p, R, v)

    def gripper(self, pos):
        """Sent as a Kortex ACTION (start/end events) rather than SendGripperCommand, which left the controller in
        MANUALLY_CONTROLLED. Then wait until the fingers stop moving (they stall early on a tool)."""
        B = self.B
        a = B.Action()
        a.name = "edgehand gripper"
        a.send_gripper_command.mode = B.GRIPPER_POSITION
        f = a.send_gripper_command.gripper.finger.add()
        f.finger_identifier, f.value = 1, float(np.clip(pos / config.GRIPPER_CLOSED, 0.0, 1.0))
        if not self._execute(a):
            return False
        # same position for 0.3 s, 4 s at most
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
        return self.unfold_if_needed() and self.move_free(*look_pose())

    def wait(self, seconds):
        """Sleep, and keep the TCP session alive with a cheap state read at most once a second (the hold loop waits
        in 20 ms steps for as long as the operator takes)."""
        time.sleep(seconds)
        now = time.time()
        if now - getattr(self, "_last_tcp", 0.0) > 1.0:
            self._last_tcp = now
            try:
                self.base.GetArmState()
            except Exception as e:                    # don't let a keep-alive kill the hold loop
                self.log(f"kortex: keep-alive failed: {e}")

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
