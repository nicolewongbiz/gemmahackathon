"""No-ROS tests: pytest ros2_ws/src/edgehand/test"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edgehand import config                                        # noqa: E402
from edgehand.geometry import R_from_quat, flange_from_tcp, handover_R, quat_from_R, tool_down_R  # noqa: E402
from edgehand.handover import Handover                             # noqa: E402
from edgehand.intent import parse_reply                            # noqa: E402
from edgehand.release import TugDetector                           # noqa: E402
from edgehand.vision import TableMap, detect_tools, find_tools     # noqa: E402
from edgehand.namer import marked_image, parse_names               # noqa: E402
import synthetic                                                   # noqa: E402


# ---- intent ----
@pytest.mark.parametrize("text,tool", [
    ('{"tool": "phillips", "reply": "Here you go"}', "phillips"),
    ('Sure! ```json\n{"tool": "Flathead", "reply": "ok"}\n```', "flathead"),
    ('{"tool": "hammer", "reply": "no hammer"}', None),          # not whitelisted
    ('{"tool": null, "reply": "hi"}', None),
])
def test_parse_reply(text, tool):
    assert parse_reply(text)["tool"] == tool


def test_parse_reply_no_json():
    with pytest.raises(ValueError):
        parse_reply("I think you want a screwdriver")


# ---- geometry ----
def test_quat_roundtrip():
    for yaw in np.linspace(-3, 3, 7):
        R = tool_down_R((np.cos(yaw), np.sin(yaw)))
        assert np.allclose(R_from_quat(quat_from_R(R)), R, atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1)


@pytest.mark.parametrize("fingers", ["x", "y"])
def test_tool_down_points_down_and_handle_to_operator(fingers, monkeypatch):
    monkeypatch.setattr(config, "FINGERS_ALONG", fingers)
    R = handover_R()
    along = R[:, 1] if fingers == "x" else R[:, 0]          # the flange axis lying along the tool
    across = R[:, 0] if fingers == "x" else R[:, 1]          # the axis the fingers close along
    assert np.allclose(R[:, 2], [0, 0, -1]) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(along[:2], np.array(config.OPERATOR_DIR[:2]) / np.linalg.norm(config.OPERATOR_DIR[:2]))
    assert np.isclose(across @ np.array([*config.OPERATOR_DIR[:2], 0]), 0)


def test_flange_is_above_fingertips_when_pointing_down():
    p = flange_from_tcp((0.5, 0, 0.012), tool_down_R((1, 0)))
    assert np.allclose(p, (0.5, 0, 0.012 + config.TCP_OFFSET_M))


# ---- release ----
def test_tug_detector():
    d = TugDetector(threshold=1.0, samples=3)
    base = {"joint_5": 0.5, "joint_6": -2.0, "joint_7": 0.1}
    d.set_baseline([base] * 5)
    assert not d.update(base)
    pulled = {**base, "joint_6": -3.5}
    assert [d.update(pulled) for _ in range(3)] == [False, False, True]
    d.set_baseline([base])
    assert not any(d.update(x) for x in [pulled, base, pulled, base])   # blips reset the count


# ---- vision ----
@pytest.mark.parametrize("name,g,d", [
    ("phillips", (0.45, 0.05), (0.0, 1.0)),
    ("flathead", (0.50, -0.05), (1.0, 0.3)),
    ("torx", (0.42, 0.0), (-0.6, -0.8)),
])
def test_detect_tool_grasps_shaft_and_finds_handle(name, g, d):
    img = synthetic.frame([(name, g, d)])
    tm = TableMap()
    assert tm.update(img), "tags not found"
    tools = detect_tools(img, tm)
    assert [t["tool"] for t in tools] == [name]
    t = tools[0]
    d = np.array(d) / np.linalg.norm(d)
    assert np.linalg.norm(t["grasp_xy"] - g) < 0.006, t["grasp_xy"]     # within 6 mm
    assert t["handle_dir"] @ d > 0.99                                    # handle direction right way round


def test_detect_two_tools():
    img = synthetic.frame([("phillips", (0.45, 0.10), (0, 1)), ("flathead", (0.50, -0.10), (0, -1))])
    tm = TableMap()
    tm.update(img)
    assert sorted(t["tool"] for t in detect_tools(img, tm)) == ["flathead", "phillips"]


def test_touching_tools_of_different_colours_stay_separate():
    # handles side by side, touching (2.4 cm wide each, 2.4 cm apart); stacked tools aren't supported
    img = synthetic.frame([("flathead", (0.45, 0.05), (1, 0)), ("torx", (0.45, 0.074), (1, 0))])
    tm = TableMap()
    tm.update(img)
    assert sorted(t["colour"] for t in find_tools(img, tm)) == ["flathead", "torx"]


def test_find_tools_unnamed_colour():
    synthetic.COLOURS["mystery"] = (200, 80, 20)                         # blue handle, no TOOLS class
    try:
        img = synthetic.frame([("mystery", (0.45, 0.0), (0, 1))])
    finally:
        del synthetic.COLOURS["mystery"]
    tm = TableMap()
    tm.update(img)
    found = find_tools(img, tm)
    assert len(found) == 1 and found[0]["colour"] is None and detect_tools(img, tm) == []


def test_parse_names_whitelists():
    assert parse_names({"1": "Phillips", "2": "hammer", "4": "torx"}, 3) == ["phillips", None, None]
    assert parse_names(None, 2) == [None, None]


def test_marked_image_crops_to_the_tools():
    img = synthetic.frame([("phillips", (0.45, 0.0), (0, 1))])
    tm = TableMap()
    tm.update(img)
    out = marked_image(img, find_tools(img, tm))
    assert 0 < out.shape[0] < img.shape[0] and 0 < out.shape[1] < img.shape[1]


def test_not_calibrated_without_all_tags():
    tags = dict(config.TABLE_TAGS)
    tags.pop(2)
    img = synthetic.frame([("phillips", (0.45, 0.0), (0, 1))], tag_xy=tags)
    tm = TableMap()
    assert not tm.update(img) and find_tools(img, tm) == []


# ---- handover state machine ----
class FakeRobot:
    def __init__(self, tug_after=None, fail=None):
        self.calls, self.t, self.tug_after, self.fail = [], 0.0, tug_after, fail

    def _do(self, name, *a):
        self.calls.append(name)
        return name != self.fail

    def move_free(self, p, R): return self._do("free", p, R)
    def move_line(self, p, R, collisions=True): return self._do("line", p, R)
    def gripper(self, pos): return self._do(f"grip{pos}")
    def look(self): return self._do("look")
    def cancel(self): self.calls.append("cancel")

    def wait(self, s):
        self.t += s

    def efforts(self):
        tug = self.tug_after is not None and self.t > self.tug_after
        return {"joint_5": 0.2, "joint_6": -1.0 - (3.0 if tug else 0.0), "joint_7": 0.0}


def test_handover_sequence_with_tug():
    r = FakeRobot(tug_after=3.0)
    ok, msg = Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1))
    assert ok, msg
    assert r.calls == [f"grip{config.GRIPPER_OPEN}", "free", "line", f"grip{config.GRIPPER_CLOSED}", "line",
                       "free", f"grip{config.GRIPPER_OPEN}", "line", "look"]


def test_handover_release_button():
    r = FakeRobot()
    h = Handover(r, log=lambda *a: None)
    orig = r.wait
    r.wait = lambda s: (orig(s), h.release() if r.t > 2 else None)
    assert h.run((0.45, 0.0), (0, 1))[0]


def test_handover_stop_while_holding():
    r = FakeRobot()
    h = Handover(r, log=lambda *a: None)
    orig = r.wait
    r.wait = lambda s: (orig(s), h.stop() if r.t > 2 else None)
    ok, msg = h.run((0.45, 0.0), (0, 1))
    assert not ok and "stopped" in msg and "cancel" in r.calls
    assert f"grip{config.GRIPPER_OPEN}" not in r.calls[1:]        # never drops the tool on a stop


def test_handover_refuses_outside_workspace():
    r = FakeRobot()
    ok, msg = Handover(r, log=lambda *a: None).run((1.5, 0.0), (0, 1))
    assert not ok and "WORKSPACE" in msg and r.calls == []


def test_handover_motion_failure():
    ok, msg = Handover(FakeRobot(fail="line"), log=lambda *a: None).run((0.45, 0.0), (0, 1))
    assert not ok and "descend failed" in msg
