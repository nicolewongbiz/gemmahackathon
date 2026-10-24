"""The handover sequence, independent of ROS so it runs the same against MoveIt and the MuJoCo sim.

A Robot provides (all poses are the FLANGE, EE_LINK, as position + 3x3 rotation in the base frame):
    move_free(p, R) -> bool      planned collision-free move (MoveIt move_group)
    move_line(p, R, collisions=True) -> bool   straight-line move (MoveIt compute_cartesian_path); collisions=False
                                 lets the short vertical descend/lift enter the tray keep-out box
    gripper(pos) -> bool         0.0 open ... 0.8 closed
    park() -> bool               joint move to PARK_JOINTS, out of the camera's view
    efforts() -> {joint: N*m}    latest measured joint torques
    wait(seconds)                sleep (ROS) or step physics (sim)
    cancel()                     stop the current motion now
"""
import threading
import numpy as np
from edgehand import config
from edgehand.geometry import flange_from_tcp, handover_R, in_workspace, tool_down_R
from edgehand.release import TugDetector


class Stopped(Exception):
    pass


class Handover:
    def __init__(self, robot, log=print, state=lambda s: None):
        self.robot, self.log, self.state = robot, log, state
        self.stop_flag = threading.Event()
        self.release_flag = threading.Event()
        self.detector = TugDetector()
        self.holding = False             # True while waiting for the tug (for status displays)

    # called from other threads (ROS callbacks)
    def stop(self):
        self.stop_flag.set()
        self.robot.cancel()

    def release(self):
        self.release_flag.set()

    def _step(self, name, ok):
        if self.stop_flag.is_set():
            raise Stopped(name)
        if not ok:
            raise RuntimeError(f"{name} failed")

    def plan_targets(self, grasp_xy, handle_dir):
        """Flange poses for every waypoint. Raises ValueError if any is outside WORKSPACE."""
        z = config.TABLE_Z + config.TOOL_RADIUS_M
        R_pick, R_give = tool_down_R(handle_dir), handover_R()
        grasp = np.array([grasp_xy[0], grasp_xy[1], z])
        give = np.array(config.HANDOVER_XYZ, float)
        back = give - np.array([*config.OPERATOR_DIR[:2], 0.0]) * 0.10    # retreat away from the operator
        t = {"hover": (grasp + [0, 0, config.HOVER_M], R_pick), "grasp": (grasp, R_pick),
             "give": (give, R_give), "retreat": (back, R_give)}
        for name, (p, _) in t.items():
            if not in_workspace(p):
                raise ValueError(f"{name} target {np.round(p, 3)} outside WORKSPACE")
        return {k: (flange_from_tcp(p, R), R) for k, (p, R) in t.items()}

    def run(self, grasp_xy, handle_dir):
        """Pick the tool at grasp_xy and hand it over. Returns (ok, message)."""
        self.stop_flag.clear()
        self.release_flag.clear()
        r = self.robot
        try:
            t = self.plan_targets(grasp_xy, handle_dir)
            self.state("picking")
            self._step("open", r.gripper(config.GRIPPER_OPEN))
            self._step("approach", r.move_free(*t["hover"]))
            self._step("descend", r.move_line(*t["grasp"], collisions=False))
            self._step("close", r.gripper(config.GRIPPER_CLOSED))
            self._step("lift", r.move_line(*t["hover"], collisions=False))
            self.state("presenting")
            self._step("to handover", r.move_free(*t["give"]))
            self._wait_for_tug()
            self._step("release", r.gripper(config.GRIPPER_OPEN))
            self.state("retreating")
            self._step("retreat", r.move_line(*t["retreat"]))
            self._step("park", r.park())
            self.state("idle")
            return True, "handed over"
        except Stopped as e:
            self.state("stopped")
            return False, f"stopped during {e}"
        except (RuntimeError, ValueError) as e:
            self.state("error")
            return False, str(e)

    def _wait_for_tug(self):
        r = self.robot
        r.wait(config.SETTLE_S)
        samples = []
        for _ in range(10):
            samples.append(r.efforts())
            r.wait(0.02)
        self.detector.set_baseline(samples)
        self.state("holding: pull the handle or press RELEASE")
        self.log("holding; waiting for a tug or RELEASE")
        self.holding = True
        try:
            self._hold()
        finally:
            self.holding = False

    def _hold(self):
        r = self.robot
        while True:
            if self.stop_flag.is_set():
                raise Stopped("holding")
            if self.release_flag.is_set():
                self.log("RELEASE button")
                return
            if self.detector.update(r.efforts()):
                self.log(f"tug detected (delta {self.detector.last_delta:.2f} N*m)")
                return
            r.wait(0.02)
