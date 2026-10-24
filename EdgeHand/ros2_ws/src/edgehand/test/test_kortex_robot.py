"""No-hardware tests for kortex_robot.py: the Kortex calls are replaced, the safety logic around them is real."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgehand import config                                        # noqa: E402
from edgehand.geometry import kortex_R, table_clearance, tool_down_R, flange_from_tcp, look_pose   # noqa: E402
from edgehand.handover import Handover                             # noqa: E402
from edgehand.kortex_robot import KortexRobot, free_waypoints, kortex_theta, too_low   # noqa: E402


@pytest.mark.parametrize("theta", [(180, 0, 90), (174.3, -7.2, 89.8), (90, 0, 90), (-170, 20, -45), (180, 0, 99)])
def test_kortex_theta_inverts_kortex_R(theta):
    R = kortex_R(*theta)
    assert np.allclose(kortex_R(*kortex_theta(R)), R, atol=1e-9)


def test_free_move_rises_translates_then_descends():
    p0, p1 = (0.40, 0.00, 0.30), (0.55, 0.20, 0.20)
    wps = free_waypoints(p0, p1, np.eye(3))
    assert len(wps) == 2 and np.allclose(wps[0], (0.55, 0.20, 0.30)) and np.allclose(wps[1], p1)
    wps = free_waypoints(p1, p0, np.eye(3))                                 # going up: rise first
    assert len(wps) == 2 and np.allclose(wps[0], (0.55, 0.20, 0.30)) and np.allclose(wps[1], p0)
    assert len(free_waypoints((0.4, 0, 0.3), (0.4, 0, 0.3), np.eye(3))) == 1   # no-op move still returns a target


class FakeKortex(KortexRobot):
    """KortexRobot with the SDK replaced: records reaches, moves the 'arm' instantly."""
    def __init__(self, start=(0.254, -0.043, 0.565), grip=0.7):
        self.log, self.tool_z = lambda *a: None, 0.12
        self.p, self.R = np.array(start, float), kortex_R(180, 0, 90)
        self.reaches, self.grip_pct, self.cmds = [], grip, []
        self._stopped = False

    def _reach(self, p, R, speed):
        if too_low(p, R):
            return False
        self.reaches.append((np.array(p), speed))
        self.p, self.R = np.array(p, float), np.array(R)
        return True

    def flange(self):
        return self.p, self.R

    def gripper(self, pos):
        self.cmds.append(pos)
        return True

    def grip_position(self):
        return self.grip_pct

    def efforts(self):
        return {f"joint_{i}": 0.0 for i in range(1, 8)}

    def wait(self, s):
        pass


def test_full_handover_on_the_kortex_robot_keeps_lateral_moves_high():
    r = FakeKortex()
    h = Handover(r, log=lambda *a: None)
    h._wait_for_tug = lambda: None                                          # operator takes it at once
    ok, msg = h.run((0.45, 0.0), (0, 1))
    assert ok, msg
    R = tool_down_R((0, 1))
    hover = flange_from_tcp((0.45, 0.0, config.TABLE_Z + config.TOOL_RADIUS_M + config.HOVER_M), R)[2]
    # every reach that changes x or y happens at or above hover height
    prev = np.array(r.reaches[0][0])
    for p, _ in r.reaches[1:]:
        if np.linalg.norm(p[:2] - prev[:2]) > 1e-6:
            assert min(p[2], prev[2]) >= hover - 1e-6, (prev, p)
        prev = p
    # the slow last 3 cm used a lower speed
    speeds = [s for _, s in r.reaches]
    assert min(speeds) == pytest.approx(config.KORTEX_LINEAR_SPEED_MPS * config.DESCEND_SLOW_SCALE / config.VEL_SCALE)
    assert max(speeds) == config.KORTEX_LINEAR_SPEED_MPS
    # the lowest flange ever commanded keeps the closed fingertips at the clearance
    low = min(table_clearance(p, tool_down_R((0, 1))) for p, _ in r.reaches)
    assert low == pytest.approx(config.TABLE_CLEARANCE_M, abs=1e-6)


def test_robot_layer_refuses_a_low_target_even_if_asked():
    r = FakeKortex()
    R = tool_down_R((1, 0))
    p = flange_from_tcp((0.45, 0.0, config.TABLE_Z - 0.002), R)           # 2 mm into the table
    assert too_low(p, R) and not r._reach(p, R, 0.05) and r.reaches == []


def test_grip_position_units_match_ros_joint():
    r = FakeKortex()
    r._feedback = lambda: type("F", (), {"interconnect": type("I", (), {"gripper_feedback": type("G", (), {
        "motor": [type("M", (), {"position": 98.0})()]})()})()})()
    assert KortexRobot.grip_position(r) == pytest.approx(0.98 * config.GRIPPER_CLOSED)
    assert KortexRobot.grip_position(r) > config.GRIPPER_EMPTY_POS         # 98 % closed = nothing in the hand
