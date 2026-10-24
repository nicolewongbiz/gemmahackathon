"""Kinova Kortex wrapper with a MockArm for developing without the robot.

Based on the patterns in Kinovarobotics/Kinova-kortex2_Gen3_G3L api_python/examples
(01-move_angular_and_cartesian.py, gripper examples). Check against the version you install.
"""
import threading, time
import config


def check_workspace(x, y, z):
    for name, v in (("x", x), ("y", y), ("z", z)):
        lo, hi = config.WORKSPACE[name]
        if not lo <= v <= hi:
            raise ValueError(f"target {name}={v:.3f} m outside workspace [{lo}, {hi}]")


class MockArm:
    def move_to(self, x, y, z, orient=None):
        check_workspace(x, y, z)
        print(f"[mock arm] move_to x={x:.3f} y={y:.3f} z={z:.3f} t={orient or config.TOOL_ORIENT}"); time.sleep(0.3)
    def gripper(self, v):
        print(f"[mock arm] gripper {v:.2f}"); time.sleep(0.2)
    def grasp(self):
        print("[mock arm] grasp (close until contact)"); time.sleep(0.2); return 0.5
    def move_joints(self, x, y, z, orient=None):
        check_workspace(x, y, z)
        print(f"[mock arm] move_joints x={x:.3f} y={y:.3f} z={z:.3f} t={orient or config.TOOL_ORIENT}"); time.sleep(0.3)
    def look(self):
        self.move_joints(*config.LOOK_POSE, orient=config.LOOK_ORIENT)
    def home(self):
        print("[mock arm] home")
    def stop(self):
        print("[mock arm] STOP")
    def close(self):
        pass


class KinovaArm:
    TIMEOUT = 20

    def __init__(self, ip=config.ROBOT_IP, user=config.ROBOT_USER, pw=config.ROBOT_PASS):
        from kortex_api.TCPTransport import TCPTransport
        from kortex_api.RouterClient import RouterClient
        from kortex_api.SessionManager import SessionManager
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        from kortex_api.autogen.messages import Session_pb2, Base_pb2
        self.Base_pb2 = Base_pb2
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, lambda e: print("[kortex]", e))
        self.transport.connect(ip, 10000)
        info = Session_pb2.CreateSessionInfo()
        info.username, info.password = user, pw
        info.session_inactivity_timeout = 60000
        info.connection_inactivity_timeout = 2000
        self.session = SessionManager(self.router)
        self.session.CreateSession(info)
        self.base = BaseClient(self.router)
        self.cyclic = BaseCyclicClient(self.router)
        try:
            self.base.ClearFaults()   # e.g. left over from a crash; does not move the arm
        except Exception as e:  # noqa: BLE001
            print("[kortex] ClearFaults failed:", e)
        time.sleep(0.5)
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)

    def _run(self, action):
        done = threading.Event()
        B = self.Base_pb2
        result = {}
        def cb(n, e=done):
            if n.action_event in (B.ACTION_END, B.ACTION_ABORT):
                result["event"] = n.action_event
                result["why"] = B.SubErrorCodes.Name(n.abort_details) if n.abort_details else "no details"
                e.set()
        h = self.base.OnNotificationActionTopic(cb, B.NotificationOptions())
        try:
            self.base.ExecuteAction(action)
            if not done.wait(self.TIMEOUT):
                self.base.Stop()
                raise TimeoutError("Kinova action timed out")
        finally:
            self.base.Unsubscribe(h)
        if result.get("event") == B.ACTION_ABORT:
            raise RuntimeError(f"Kinova aborted '{action.name}': {result['why']}. "
                               f"Check and clear faults in the Web App at http://{config.ROBOT_IP}")

    def move_to(self, x, y, z, orient=None):
        check_workspace(x, y, z)
        a = self.Base_pb2.Action()
        a.name, a.application_data = "cart", ""
        p = a.reach_pose.target_pose
        p.x, p.y, p.z = x, y, z + config.Z_TABLE_OFFSET
        p.theta_x, p.theta_y, p.theta_z = orient or config.TOOL_ORIENT
        s = a.reach_pose.constraint.speed
        s.translation, s.orientation = config.MAX_LINEAR_SPEED, config.MAX_ANGULAR_SPEED
        self._run(a)

    def move_joints(self, x, y, z, orient=None):
        """Same target as move_to, but solved with IK and driven joint by joint. Use it for moves that
        change the wrist orientation a lot: a straight-line Cartesian move there can wind the wrist
        through a singularity and fault the arm (it did, look pose <-> gripper-down)."""
        check_workspace(x, y, z)
        B = self.Base_pb2
        d = B.IKData()
        p = d.cartesian_pose
        p.x, p.y, p.z = x, y, z + config.Z_TABLE_OFFSET
        p.theta_x, p.theta_y, p.theta_z = orient or config.TOOL_ORIENT
        cur = [j.value for j in self.base.GetMeasuredJointAngles().joint_angles]
        for v in cur:
            d.guess.joint_angles.add().value = v   # stay close to the current configuration
        sol = [j.value for j in self.base.ComputeInverseKinematics(d).joint_angles]
        # Continuous joints (1, 3, 5, 7): use the equivalent angle nearest the current one, e.g. 0.4 from
        # 359.9 -> 360.4 (tested: the arm accepts it). Limited joints (2, 4, 6) stay in 0..360 like Home's.
        target = [c + (s - c + 180.0) % 360.0 - 180.0 if k % 2 == 0 else s % 360.0
                  for k, (c, s) in enumerate(zip(cur, sol))]
        a = B.Action()
        a.name, a.application_data = "joints", ""
        for k, v in enumerate(target):
            ja = a.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier, ja.value = k, v
        # JOINT_CONSTRAINT_SPEED makes the Gen3 abort (METHOD_FAILED) any move of the big joints,
        # so limit speed with a duration: the joint that moves furthest goes at MAX_ANGULAR_SPEED.
        c = a.reach_joint_angles.constraint
        c.type = B.JOINT_CONSTRAINT_DURATION
        c.value = max(1.5, max(abs((t - v + 180.0) % 360.0 - 180.0) for t, v in zip(target, cur)) / config.MAX_ANGULAR_SPEED)
        print(f"[kortex] joint target {[round(v, 1) for v in target]} over {c.value:.1f} s")
        self._run(a)

    def gripper_position(self):
        """Measured gripper opening, 0.0 open .. 1.0 closed."""
        m = self.cyclic.RefreshFeedback().interconnect.gripper_feedback.motor
        return m[0].position / 100.0 if m else None

    def gripper(self, v):
        B = self.Base_pb2
        cmd = B.GripperCommand()
        cmd.mode = B.GRIPPER_POSITION
        f = cmd.gripper.finger.add()
        f.finger_identifier, f.value = 1, v
        self.base.SendGripperCommand(cmd)
        time.sleep(1.5)
        got = self.gripper_position()
        print(f"[kortex] gripper commanded {v:.2f}, measured {got if got is None else round(got, 2)}")

    def grasp(self):
        """Close at GRIPPER_CLOSE_SPEED until the fingers stop (object in the way, or fully shut).
        Fits any object size, unlike a fixed position. Returns the final opening (0 open .. 1 shut);
        >= GRIPPER_EMPTY means it closed on nothing."""
        B = self.Base_pb2
        cmd = B.GripperCommand()
        cmd.mode = B.GRIPPER_SPEED
        f = cmd.gripper.finger.add()
        f.finger_identifier, f.value = 1, -config.GRIPPER_CLOSE_SPEED   # negative speed = close (Kortex example)
        start = self.gripper_position() or 0.0
        self.base.SendGripperCommand(cmd)
        last, still, t0 = start, 0, time.time()
        while time.time() - t0 < 5.0:
            time.sleep(0.05)
            pos = self.gripper_position() or 0.0
            still = still + 1 if abs(pos - last) < 0.002 else 0
            last = pos
            if still >= 6 and time.time() - t0 > 0.4:   # stopped moving for 0.3 s
                break
        if last < start - 0.05:
            print("[kortex] grasp OPENED the gripper: flip the sign of the speed in KinovaArm.grasp")
        print(f"[kortex] grasp: closed from {start:.2f} to {last:.2f}"
              + ("  (nothing in the gripper?)" if last >= config.GRIPPER_EMPTY else ""))
        return last

    def look(self):
        self.move_joints(*config.LOOK_POSE, orient=config.LOOK_ORIENT)
        time.sleep(0.3)   # let the wrist settle so the frame isn't motion-blurred

    def home(self):
        B = self.Base_pb2
        req = B.RequestedActionType(); req.action_type = B.REACH_JOINT_ANGLES
        for a in self.base.ReadAllActions(req).action_list:
            if a.name == "Home":
                return self._run(a)

    def stop(self):
        self.base.Stop()

    def close(self):
        self.session.CloseSession()
        self.transport.disconnect()


def make_arm():
    return MockArm() if config.USE_MOCK_ARM else KinovaArm()


def pick_and_place(arm, xy, bin_xy, should_stop=lambda: False):
    """Returns False if stopped part-way. stop() only aborts the current motion, so check between steps."""
    x, y = xy
    bx, by = bin_xy
    for target in (bin_xy, xy):
        check_workspace(*target, config.Z_HOVER)
    steps = [
        lambda: arm.gripper(config.GRIPPER_OPEN),
        lambda: arm.move_joints(x, y, config.Z_HOVER),   # from the look pose: big wrist turn, so joint space
        lambda: arm.move_to(x, y, config.Z_GRASP),
        arm.grasp,
        lambda: arm.move_to(x, y, config.Z_HOVER),
        lambda: arm.move_to(bx, by, config.Z_HOVER),
        lambda: arm.gripper(config.GRIPPER_OPEN),
    ]
    for step in steps:
        if should_stop():
            return False
        step()
    return True
