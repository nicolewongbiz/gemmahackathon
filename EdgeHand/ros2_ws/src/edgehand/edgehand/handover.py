"""The handover sequence, independent of ROS so it runs the same against MoveIt and the MuJoCo sim.

A Robot provides (all poses are the FLANGE, EE_LINK, as position + 3x3 rotation in the base frame):
    move_free(p, R) -> bool      planned collision-free move (MoveIt move_group)
    move_line(p, R, collisions=True, scale=None) -> bool   straight-line move (MoveIt compute_cartesian_path);
                                 collisions=False lets the short vertical descend/lift enter the tray keep-out box;
                                 scale = velocity scaling (None = VEL_SCALE)
    gripper(pos) -> bool         0.0 open ... 0.8 closed
    grip_position() -> float     measured finger position (same units), None if unknown
    look() -> bool               move to the look pose (wrist camera over the tray); also where the arm waits
    efforts() -> {joint: N*m}    latest measured joint torques
    wait(seconds)                sleep (ROS) or step physics (sim)
    cancel()                     stop the current motion now
"""
import threading
import numpy as np
from edgehand import config
from edgehand.geometry import flange_from_tcp, handover_R, in_workspace, table_clearance, table_z, tool_down_R, yaw_of
from edgehand.release import TugDetector


class Stopped(Exception):
    pass


class Handover:
    def __init__(self, robot, log=print, state=lambda s: None, check_grasp=True):
        self.robot, self.log, self.state = robot, log, state
        self.check_grasp = check_grasp   # False with fake hardware: its fingers always close fully
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

    def plan_targets(self, grasp_xy, handle_dir, current_R=None):
        """Flange poses for every waypoint. Raises ValueError if any is outside WORKSPACE, or would bring the closed
        fingertips closer to the table than TABLE_CLEARANCE_M. The fingers close across the tool, so the grasp is
        the same turned by 180 degrees: given the arm's current orientation, the pick yaw needing the smaller wrist
        turn is used, and the handover orientation is flipped to match so the handle still faces the operator."""
        z = table_z(*grasp_xy) + config.TOOL_RADIUS_M + config.AIR_GAP_M
        d = np.asarray(handle_dir, float)
        R_pick, R_give = tool_down_R(d), handover_R()
        if current_R is not None:
            alt = tool_down_R(-d)
            turn = lambda R: abs((np.degrees(yaw_of(R) - yaw_of(current_R)) + 180) % 360 - 180)
            if turn(alt) < turn(R_pick):
                R_pick, R_give = alt, tool_down_R(-np.asarray(config.OPERATOR_DIR[:2], float))
        grasp = np.array([grasp_xy[0], grasp_xy[1], z])
        give = np.array(config.HANDOVER_XYZ, float)
        back = give - np.array([*config.OPERATOR_DIR[:2], 0.0]) * 0.10    # retreat away from the operator
        t = {"hover": (grasp + [0, 0, config.HOVER_M], R_pick), "slow": (grasp + [0, 0, config.DESCEND_SLOW_M], R_pick),
             "grasp": (grasp, R_pick), "give": (give, R_give), "retreat": (back, R_give)}
        for name, (p, _) in t.items():
            if not in_workspace(p):
                raise ValueError(f"{name} target {np.round(p, 3)} outside WORKSPACE")
        t = {k: (flange_from_tcp(p, R), R) for k, (p, R) in t.items()}
        for name, (p, R) in t.items():
            c = table_clearance(p, R)
            if c < config.TABLE_CLEARANCE_M - 1e-4:
                raise ValueError(f"{name} target puts the fingertips {c * 1000:.1f} mm above the table "
                                 f"(minimum {config.TABLE_CLEARANCE_M * 1000:.1f} mm)")
        return t

    def run(self, grasp_xy, handle_dir, relocate=None):
        """Pick the tool at grasp_xy and hand it over. After a missed grasp the arm goes back to the look pose and
        relocate() -> (grasp_xy, handle_dir) or None says where the tool is now (relocate=None: no retries).
        Returns (ok, message)."""
        self.stop_flag.clear()
        self.release_flag.clear()
        r = self.robot
        try:
            for attempt in range(1, config.PICK_ATTEMPTS + 1):
                t = self.plan_targets(grasp_xy, handle_dir, self._current_R())
                if self._pick(t):
                    break
                self.log(f"missed the tool (try {attempt} of {config.PICK_ATTEMPTS})")
                self._step("open", r.gripper(config.GRIPPER_OPEN))     # at hover height, nothing between the fingers
                self.state("looking again")
                self._step("look", r.look())
                found = relocate() if relocate and attempt < config.PICK_ATTEMPTS else None
                self._step("look again", True)                         # a STOP while vision was looking
                if found is None:
                    self.state("idle")
                    return False, f"missed the tool {attempt} time{'s' if attempt > 1 else ''}"
                grasp_xy, handle_dir = found
            self.state("presenting")
            self._step("to handover", r.move_free(*t["give"]))
            self._wait_for_tug()
            self._step("release", r.gripper(config.GRIPPER_OPEN))
            self.state("retreating")
            self._step("retreat", r.move_line(*t["retreat"]))
            self._step("look", r.look())
            self.state("idle")
            return True, "handed over"
        except Stopped as e:
            self.state("stopped")
            return False, f"stopped during {e}"
        except (RuntimeError, ValueError) as e:
            self.state("error")
            return False, str(e)

    def _current_R(self):
        """The arm's orientation now, if the robot can tell us (KortexRobot.flange); None otherwise."""
        f = getattr(self.robot, "flange", None)
        try:
            return f()[1] if f else None
        except Exception:
            return None

    def _pick(self, t):
        """Open, down to the grasp (the last DESCEND_SLOW_M slowly), close, lift. False = nothing between the fingers."""
        r = self.robot
        self.state("picking")
        self._step("open", r.gripper(config.GRIPPER_OPEN))
        self._step("approach", r.move_free(*t["hover"]))
        self._step("descend", r.move_line(*t["slow"], collisions=False))
        self._step("descend slowly", r.move_line(*t["grasp"], collisions=False, scale=config.DESCEND_SLOW_SCALE))
        self._step("close", r.gripper(config.GRIPPER_CLOSED))
        self._step("lift", r.move_line(*t["hover"], collisions=False))
        return self._holding()

    def _holding(self):
        """Checked after the lift, so a tool that slipped out on the way up counts as a miss too."""
        if not self.check_grasp or config.AIR_GAP_M > 0:            # an air run closes on nothing on purpose
            return True
        pos = self.robot.grip_position()
        if pos is None:
            self.log("no gripper position: can't tell whether the grasp worked")
            return True
        return pos < config.GRIPPER_EMPTY_POS

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
