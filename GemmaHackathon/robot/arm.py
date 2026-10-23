"""Kinova Kortex wrapper with a MockArm for developing without the robot.

Based on the patterns in Kinovarobotics/Kinova-kortex2_Gen3_G3L api_python/examples
(01-move_angular_and_cartesian.py, gripper examples). Check against the version you install.
"""
import threading, time
import config


class MockArm:
    def move_to(self, x, y, z):
        print(f"[mock arm] move_to x={x:.3f} y={y:.3f} z={z:.3f}"); time.sleep(0.3)
    def gripper(self, v):
        print(f"[mock arm] gripper {v:.2f}"); time.sleep(0.2)
    def look(self):
        print("[mock arm] look pose")
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
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)

    def _run(self, action):
        done = threading.Event()
        B = self.Base_pb2
        def cb(n, e=done):
            if n.action_event in (B.ACTION_END, B.ACTION_ABORT):
                e.set()
        h = self.base.OnNotificationActionTopic(cb, B.NotificationOptions())
        self.base.ExecuteAction(action)
        ok = done.wait(self.TIMEOUT)
        self.base.Unsubscribe(h)
        if not ok:
            raise TimeoutError("Kinova action timed out")

    def move_to(self, x, y, z):
        a = self.Base_pb2.Action()
        a.name, a.application_data = "cart", ""
        p = a.reach_pose.target_pose
        p.x, p.y, p.z = x, y, z + config.Z_TABLE_OFFSET
        p.theta_x, p.theta_y, p.theta_z = config.TOOL_ORIENT
        self._run(a)

    def gripper(self, v):
        B = self.Base_pb2
        cmd = B.GripperCommand()
        cmd.mode = B.GRIPPER_POSITION
        f = cmd.gripper.finger.add()
        f.finger_identifier, f.value = 1, v
        self.base.SendGripperCommand(cmd)
        time.sleep(1.0)

    def look(self):
        self.move_to(*config.LOOK_POSE)
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


def pick_and_place(arm, xy, bin_xy):
    x, y = xy
    bx, by = bin_xy
    arm.gripper(config.GRIPPER_OPEN)
    arm.move_to(x, y, config.Z_HOVER)
    arm.move_to(x, y, config.Z_GRASP)
    arm.gripper(config.GRIPPER_CLOSED)
    arm.move_to(x, y, config.Z_HOVER)
    arm.move_to(bx, by, config.Z_HOVER)
    arm.gripper(config.GRIPPER_OPEN)
