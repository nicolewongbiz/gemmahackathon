"""Kinova Gen3 (7-DoF) control via the Kortex Python API, plus a MockArm.

Frames: all positions are the TOOL position in the robot BASE frame, metres.
    +x = forward (away from the base, the direction the arm faces at Home)
    +y = left
    +z = up
Orientation is Kinova's theta_x / theta_y / theta_z in degrees.
"""
import logging
import threading
import time

import config

log = logging.getLogger("arm")


class MockArm:
    """Pretends to be the robot so you can test chat -> actions with nothing plugged in."""
    name = "mock"

    def __init__(self):
        self._pose = {"x": 0.45, "y": 0.0, "z": 0.40, "tx": 180.0, "ty": 0.0, "tz": 90.0}
        self._grip = 0.0

    def pose(self):
        return dict(self._pose, gripper=self._grip)

    def move_to(self, x, y, z, tx, ty, tz):
        log.info("[mock] move_to x=%.3f y=%.3f z=%.3f  t=(%.1f, %.1f, %.1f)", x, y, z, tx, ty, tz)
        time.sleep(0.5)
        self._pose.update(x=x, y=y, z=z, tx=tx, ty=ty, tz=tz)

    def gripper(self, value):
        log.info("[mock] gripper %.2f", value)
        time.sleep(0.3)
        self._grip = value

    def home(self):
        log.info("[mock] home")
        time.sleep(0.5)
        self._pose.update(x=0.45, y=0.0, z=0.40, tx=180.0, ty=0.0, tz=90.0)

    def stop(self):
        log.warning("[mock] STOP")

    def close(self):
        pass


class KinovaArm:
    name = "kinova-gen3"

    def __init__(self):
        # Imported here so the app runs in mock mode without the Kortex wheel installed.
        from kortex_api.TCPTransport import TCPTransport
        from kortex_api.RouterClient import RouterClient
        from kortex_api.SessionManager import SessionManager
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.messages import Session_pb2, Base_pb2

        self.B = Base_pb2
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, lambda e: log.error("kortex router error: %s", e))
        log.info("Connecting to Kinova at %s:10000 ...", config.ROBOT_IP)
        self.transport.connect(config.ROBOT_IP, 10000)

        info = Session_pb2.CreateSessionInfo()
        info.username = config.ROBOT_USER
        info.password = config.ROBOT_PASS
        info.session_inactivity_timeout = 60000      # ms
        info.connection_inactivity_timeout = 2000    # ms
        self.session = SessionManager(self.router)
        self.session.CreateSession(info)
        self.base = BaseClient(self.router)

        try:
            self.base.ClearFaults()
        except Exception as e:  # noqa: BLE001
            log.warning("ClearFaults failed (ok if no faults): %s", e)
        mode = self.B.ServoingModeInformation()
        mode.servoing_mode = self.B.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)
        self._grip = None
        log.info("Kinova connected.")

    # ---- helpers ----
    def _execute(self, action):
        done = threading.Event()
        result = {}
        B = self.B

        def on_event(notification, e=done):
            if notification.action_event in (B.ACTION_END, B.ACTION_ABORT):
                result["event"] = notification.action_event
                e.set()

        handle = self.base.OnNotificationActionTopic(on_event, B.NotificationOptions())
        try:
            self.base.ExecuteAction(action)
            if not done.wait(config.ACTION_TIMEOUT_S):
                self.base.Stop()
                raise TimeoutError("Arm action timed out")
        finally:
            self.base.Unsubscribe(handle)
        if result.get("event") == B.ACTION_ABORT:
            raise RuntimeError("Arm aborted the move (unreachable pose, limit, or fault). "
                               "Check the Kinova Web App at http://%s" % config.ROBOT_IP)

    # ---- public API ----
    def pose(self):
        p = self.base.GetMeasuredCartesianPose()
        return {"x": p.x, "y": p.y, "z": p.z, "tx": p.theta_x, "ty": p.theta_y, "tz": p.theta_z,
                "gripper": self._grip}

    def move_to(self, x, y, z, tx, ty, tz):
        a = self.B.Action()
        a.name = "gemma_move"
        a.application_data = ""
        rp = a.reach_pose
        rp.target_pose.x, rp.target_pose.y, rp.target_pose.z = x, y, z
        rp.target_pose.theta_x, rp.target_pose.theta_y, rp.target_pose.theta_z = tx, ty, tz
        rp.constraint.speed.translation = config.LINEAR_SPEED
        rp.constraint.speed.orientation = config.ANGULAR_SPEED
        self._execute(a)

    def gripper(self, value):
        """value: 0.0 = fully open, 1.0 = fully closed (Robotiq 2F-85 on Gen3)."""
        cmd = self.B.GripperCommand()
        cmd.mode = self.B.GRIPPER_POSITION
        f = cmd.gripper.finger.add()
        f.finger_identifier = 1
        f.value = float(value)
        self.base.SendGripperCommand(cmd)
        time.sleep(1.0)
        self._grip = value

    def home(self):
        req = self.B.RequestedActionType()
        req.action_type = self.B.REACH_JOINT_ANGLES
        for a in self.base.ReadAllActions(req).action_list:
            if a.name == "Home":
                return self._execute(a)
        raise RuntimeError("No 'Home' action found on the robot")

    def stop(self):
        self.base.Stop()

    def close(self):
        try:
            self.session.CloseSession()
        finally:
            self.transport.disconnect()


def make_arm():
    if config.MOCK_ARM:
        log.info("Using MockArm (set GB_MOCK_ARM=0 for the real robot)")
        return MockArm()
    return KinovaArm()
