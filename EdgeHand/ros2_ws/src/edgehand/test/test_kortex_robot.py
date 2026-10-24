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
    assert len(wps) == 2 and np.allclose(wps[0][0], (0.55, 0.20, 0.30)) and np.allclose(wps[1][0], p1)
    wps = free_waypoints(p1, p0, np.eye(3))                                 # going up: rise first
    assert len(wps) == 2 and np.allclose(wps[0][0], (0.55, 0.20, 0.30)) and np.allclose(wps[1][0], p0)
    assert len(free_waypoints((0.4, 0, 0.3), (0.4, 0, 0.3), np.eye(3))) == 1   # no-op move still returns a target


def test_free_move_turns_only_at_height():
    # the arm points forward, low over the table (where the lab found it); the look pose is straight down, higher
    p0, R0 = (0.396, 0.025, 0.241), kortex_R(87.5, 12.1, 93.2)
    p1, R1 = (0.253, -0.067, 0.496), kortex_R(-180, 0, 99)
    wps = free_waypoints(p0, p1, R1, R0)
    assert np.allclose(wps[0][0], (0.396, 0.025, 0.496)) and np.allclose(wps[0][1], R0)     # rise, same orientation
    assert np.allclose(wps[1][0], (0.396, 0.025, 0.496)) and np.allclose(wps[1][1], R1)     # turn up there
    assert np.allclose(wps[2][0], (0.253, -0.067, 0.496)) and np.allclose(wps[2][1], R1)    # translate
    assert len(wps) == 3                                                                     # already at z: no descent
    for p, R in wps:
        assert table_clearance(p, R) > 0.2


class FakeBase:
    """Just enough of BaseClient for _execute: replays a stale ABORT on subscribe, then START and END on execute."""
    def __init__(self, B, stale=True, end=None):
        self.B, self.stale, self.end, self.cb, self.stopped = B, stale, end if end is not None else B.ACTION_END, None, 0
        self.executed = []

    def GetArmState(self):
        # the controller reports MANUALLY_CONTROLLED for a moment after a move, then READY again
        st = self.states.pop(0) if getattr(self, "states", None) else self.B.ARMSTATE_SERVOING_READY
        return type("S", (), {"active_state": st})()

    def OnNotificationActionTopic(self, cb, opts):
        self.cb = cb
        if self.stale:
            cb(self._n(self.B.ACTION_ABORT, 3))                 # the previous action's INVALID_PARAM, replayed
        return "handle"

    def ExecuteAction(self, a):
        self.executed.append(a)
        self.cb(self._n(self.B.ACTION_START, 0))
        self.cb(self._n(self.end, 0))

    def Unsubscribe(self, h):
        pass

    def Stop(self):
        self.stopped += 1

    def _n(self, ev, code):
        return type("N", (), {"action_event": ev, "abort_details": code})()


def _robot_with_base(base):
    r = KortexRobot.__new__(KortexRobot)
    Base_pb2 = pytest.importorskip("kortex_api.autogen.messages.Base_pb2")
    r.B, r.base, r.log, r._stopped, r.tool_z = Base_pb2, base, lambda *a: None, False, 0.12
    r.wait = lambda s: None
    return r


def test_execute_ignores_the_replayed_abort_and_waits_for_its_own_end():
    B = pytest.importorskip("kortex_api.autogen.messages.Base_pb2")      # the Kortex wheel isn't everywhere
    base = FakeBase(B, stale=True)
    r = _robot_with_base(base)
    logs = []
    r.log = logs.append
    assert r._execute(B.Action()) is True
    assert len(base.executed) == 1 and base.stopped == 0
    assert any("stale ACTION_ABORT" in m for m in logs)


def test_execute_reports_a_real_abort():
    B = pytest.importorskip("kortex_api.autogen.messages.Base_pb2")
    base = FakeBase(B, stale=True, end=B.ACTION_ABORT)
    r = _robot_with_base(base)
    assert r._execute(B.Action()) is False


def test_execute_waits_through_the_state_flicker_after_a_move(monkeypatch):
    B = pytest.importorskip("kortex_api.autogen.messages.Base_pb2")
    base = FakeBase(B, stale=False)
    base.states = [B.ARMSTATE_SERVOING_MANUALLY_CONTROLLED, B.ARMSTATE_SERVOING_PLAYING_SEQUENCE,
                   B.ARMSTATE_SERVOING_READY]                       # what the lab saw right after Home
    r = _robot_with_base(base)
    assert r._execute(B.Action()) is True and len(base.executed) == 1


def test_execute_proceeds_while_the_controller_reports_manual_control(monkeypatch):
    # after a gripper command Kortex reports MANUALLY_CONTROLLED / PLAYING_SEQUENCE yet takes the next action
    B = pytest.importorskip("kortex_api.autogen.messages.Base_pb2")
    base = FakeBase(B, stale=False)
    base.states = [B.ARMSTATE_SERVOING_MANUALLY_CONTROLLED] * 50
    r = _robot_with_base(base)
    assert r._execute(B.Action()) is True and len(base.executed) == 1


def test_execute_refuses_a_faulted_arm():
    B = pytest.importorskip("kortex_api.autogen.messages.Base_pb2")
    base = FakeBase(B, stale=False)
    base.states = [B.ARMSTATE_IN_FAULT] * 50
    r = _robot_with_base(base)
    assert r._execute(B.Action()) is False and base.executed == []


def test_gripper_goes_out_as_an_action():
    B = pytest.importorskip("kortex_api.autogen.messages.Base_pb2")
    base = FakeBase(B, stale=False)
    r = _robot_with_base(base)
    r.grip_position = lambda: 0.7
    assert KortexRobot.gripper(r, config.GRIPPER_CLOSED) is True
    a = base.executed[0]
    assert a.send_gripper_command.gripper.finger[0].value == pytest.approx(1.0)
    assert a.send_gripper_command.mode == B.GRIPPER_POSITION


def test_near_joint_limit_spots_the_folded_arm():
    from edgehand.kortex_robot import near_joint_limit
    lab = [208.0, 295.4, 323.3, 212.2, 14.3, 111.1, 140.3]          # measured in Retract: joint 4 = -147.8
    assert near_joint_limit(lab) == [(4, -147.8)]
    assert near_joint_limit([0, 0, 0, 0, 0, 0, 0]) == []
    assert near_joint_limit([0, 127.0, 0, 0, 0, -119.0, 0]) == [(2, 127.0), (6, -119.0)]


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

    def joints(self):
        return [0.0] * 7

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
