"""No-ROS tests: pytest ros2_ws/src/edgehand/test"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edgehand import config                                        # noqa: E402
from edgehand.geometry import (R_from_quat, flange_from_tcp, handover_R, quat_from_R, table_clearance,  # noqa: E402
                               table_z, tool_down_R)
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


def test_touching_tools_of_different_colours_stay_separate(monkeypatch):
    monkeypatch.setattr(config, "SEGMENTATION", "colour")       # contrast can't split touching blobs
    # handles side by side, touching (2.4 cm wide each, 2.4 cm apart); stacked tools aren't supported
    img = synthetic.frame([("flathead", (0.45, 0.05), (1, 0)), ("torx", (0.45, 0.074), (1, 0))])
    tm = TableMap()
    tm.update(img)
    assert sorted(t["colour"] for t in find_tools(img, tm)) == ["flathead", "torx"]


def test_find_tools_unnamed_colour():
    synthetic.COLOURS["mystery"] = (200, 40, 220)                        # magenta handle, no TOOLS class
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


def _tags_without(*hidden):
    return {i: xy for i, xy in config.TABLE_TAGS.items() if i not in hidden}


@pytest.mark.parametrize("hidden", [(2,), (0,), (1, 2), (0, 2)])     # one tag; two side by side; two diagonal
@pytest.mark.parametrize("yaw", [0.0, 0.5])                           # tags taped square, or crooked
def test_calibrates_from_two_or_three_tags(hidden, yaw):
    g, d = (0.47, 0.03), np.array([0.6, 0.8])
    img = synthetic.frame([("phillips", g, d)], tag_xy=_tags_without(*hidden), tag_yaw=yaw)
    tm = TableMap()
    assert tm.update(img) and tm.calibrated_from == "live" and tm.tags_used == 4 - len(hidden)
    t = detect_tools(img, tm)[0]
    assert np.linalg.norm(t["grasp_xy"] - g) < 0.006 and t["handle_dir"] @ d > 0.99


def test_not_calibrated_from_one_tag():
    img = synthetic.frame([("phillips", (0.45, 0.0), (0, 1))], tag_xy=_tags_without(1, 2, 3))
    tm = TableMap()
    assert not tm.update(img) and tm.calibrated_from is None and find_tools(img, tm) == []


def test_reuses_the_last_fit_when_tags_are_hidden():
    g, d = (0.45, -0.05), np.array([0.0, 1.0])
    tm = TableMap()
    assert tm.update(synthetic.frame([])) and tm.calibrated_from == "live"
    img = synthetic.frame([("torx", g, d)], tag_xy=_tags_without(0, 1, 2))      # tools and arm hide 3 tags
    assert tm.update(img) and tm.calibrated_from == "cached"
    t = detect_tools(img, tm)[0]
    assert np.linalg.norm(t["grasp_xy"] - g) < 0.006
    assert tm.update(synthetic.frame([])) and tm.calibrated_from == "live"     # back to live once they're visible


def test_leaves_out_a_tag_that_doesnt_match_table_tags(monkeypatch):
    # tag 2 lies 10 cm from where TABLE_TAGS says: its corners can't line up with the others, so it's left out
    g = (0.47, 0.03)
    img = synthetic.frame([("phillips", g, (0, 1))])
    monkeypatch.setitem(config.TABLE_TAGS, 2, (0.70, 0.20))
    tm = TableMap()
    assert tm.update(img) and tm.bad_tag == 2 and tm.tags_used == 3
    assert np.linalg.norm(detect_tools(img, tm)[0]["grasp_xy"] - g) < 0.006


def test_refuses_to_calibrate_when_table_tags_are_all_wrong(monkeypatch):
    # TABLE_TAGS from a different table (tags really 0.3 m apart, configured 0.25 x 0.4): no fit rather than a skewed one
    img = synthetic.frame([], tag_xy={0: (0.35, -0.15), 1: (0.35, 0.15), 2: (0.65, 0.15), 3: (0.65, -0.15)})
    tm = TableMap()
    assert tm.update(synthetic.frame([]))                                   # a good fit first...
    assert not tm.update(img) and tm.calibrated_from is None               # ...doesn't cover tags that moved
    assert "re-measure" in tm.why_not()


def test_ignores_clutter_beyond_the_tags():
    # a "tool" with its grasp point well outside the tag rectangle (cables, a game controller on the desk)
    img = synthetic.frame([("phillips", (0.45, 0.0), (0, 1)), ("flathead", (0.45, -0.34), (0, -1))])
    tm = TableMap()
    tm.update(img)
    assert [t["colour"] for t in find_tools(img, tm)] == ["phillips"]


# ---- handover state machine ----
class FakeRobot:
    """grips: finger position after each close, in order (default: always holding a shaft)."""
    def __init__(self, tug_after=None, fail=None, grips=()):
        self.calls, self.t, self.tug_after, self.fail = [], 0.0, tug_after, fail
        self.grips, self.grip_pos, self.poses, self.slow = list(grips), 0.0, [], []    # slow: indices into poses

    def _do(self, name, *a):
        self.calls.append(name)
        return name != self.fail

    def move_free(self, p, R):
        self.poses.append((p, R))
        return self._do("free", p, R)

    def move_line(self, p, R, collisions=True, scale=None):
        self.poses.append((p, R))
        self.slow.append(len(self.poses) - 1) if scale else None
        return self._do("slow line" if scale else "line", p, R)

    def gripper(self, pos):
        self.grip_pos = (self.grips.pop(0) if self.grips else 0.7) if pos == config.GRIPPER_CLOSED else pos
        return self._do(f"grip{pos}")

    def grip_position(self): return self.grip_pos
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
    assert r.calls == PICK + ["free", f"grip{config.GRIPPER_OPEN}", "line", "look"]


PICK = [f"grip{config.GRIPPER_OPEN}", "free", "line", "slow line", f"grip{config.GRIPPER_CLOSED}", "line"]


# ---- never hit the table ----
def test_every_pose_keeps_the_fingertips_above_the_table():
    r = FakeRobot(tug_after=3.0)
    assert Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1))[0]
    clear = [table_clearance(p, R) for p, R in r.poses]
    assert min(clear) >= config.TABLE_CLEARANCE_M - 1e-9
    assert np.isclose(min(clear), config.TABLE_CLEARANCE_M)                 # the grasp itself, nothing lower


def test_last_bit_of_the_descent_is_slow():
    r = FakeRobot(tug_after=3.0)
    Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1))
    i = r.slow[0]
    (p_fast, R), (p_slow, _) = r.poses[i - 1], r.poses[i]
    assert np.isclose(table_clearance(p_fast, R) - table_clearance(p_slow, R), config.DESCEND_SLOW_M)


def test_refuses_a_grasp_too_close_to_the_table(monkeypatch):
    monkeypatch.setattr(config, "TCP_OFFSET_M", config.TCP_OFFSET_M - 0.01)    # e.g. a mis-measured offset
    r = FakeRobot()
    ok, msg = Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1))
    assert not ok and "above the table" in msg and r.calls == []           # refused before any motion


def test_tilted_table_plane_from_fingertip_touches(monkeypatch):
    d = config.FINGERTIP_CLOSED_M
    # the table rises 6 mm from x 0.35 to x 0.60 (1.4 degrees), and sits 5 mm higher overall
    touch = {i: 0.005 + d + 0.006 * (x - 0.35) / 0.25 for i, (x, y) in config.TABLE_TAGS.items()}
    monkeypatch.setattr(config, "TABLE_TOUCH_Z", touch)
    assert np.isclose(table_z(0.35, 0.1), 0.005) and np.isclose(table_z(0.60, -0.1), 0.011)
    for x in (0.38, 0.57):
        r = FakeRobot(tug_after=3.0)
        assert Handover(r, log=lambda *a: None).run((x, 0.0), (0, 1))[0]
        p, R = r.poses[r.slow[0]]
        assert np.isclose(p[2] - d - table_z(x, 0.0), config.TABLE_CLEARANCE_M)   # follows the tilt


def test_touch_heights_that_disagree_are_refused(monkeypatch):
    d = config.FINGERTIP_CLOSED_M
    good = {i: d + 0.002 * i for i in config.TABLE_TAGS}                  # within 6 mm: fine
    monkeypatch.setattr(config, "TABLE_TOUCH_Z", good)
    assert Handover(FakeRobot(tug_after=3.0), log=lambda *a: None).run((0.45, 0.0), (0, 1))[0]
    typo = {**good, 3: d + 0.4}                                           # 0.573 typed instead of 0.173
    monkeypatch.setattr(config, "TABLE_TOUCH_Z", typo)
    r = FakeRobot()
    ok, msg = Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1))
    assert not ok and "re-measure" in msg and r.calls == []               # refused before any motion
    monkeypatch.setattr(config, "TABLE_TOUCH_Z", {0: d, 1: d, 9: d})      # a tag that isn't in TABLE_TAGS
    assert "aren't in TABLE_TAGS" in Handover(FakeRobot(), log=lambda *a: None).run((0.45, 0.0), (0, 1))[1]


def test_air_gap_raises_every_grasp_and_skips_the_grasp_check(monkeypatch):
    monkeypatch.setattr(config, "AIR_GAP_M", 0.03)
    r = FakeRobot(tug_after=3.0, grips=[0.79])                             # closes on air
    assert Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1))[0]
    p, R = r.poses[r.slow[0]]
    assert np.isclose(table_clearance(p, R), config.TABLE_CLEARANCE_M + 0.03)


# ---- missed grasps ----
def test_missed_grasp_looks_again_and_retries():
    r = FakeRobot(tug_after=3.0, grips=[0.79])                             # first close: nothing between the fingers
    seen = []
    relocate = lambda: seen.append(r.calls[-1]) or ((0.47, 0.02), np.array([1.0, 0.0]))
    ok, msg = Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1), relocate=relocate)
    assert ok, msg
    assert seen == ["look"]                                                # vision looks from the look pose
    assert r.calls[:len(PICK) + 2] == PICK + [f"grip{config.GRIPPER_OPEN}", "look"]
    assert r.calls[len(PICK) + 2:len(PICK) * 2 + 2] == PICK
    p, R = r.poses[r.slow[1]]
    assert np.allclose(p[:2], (0.47, 0.02))                                # second try goes where vision says now


def test_gives_up_after_pick_attempts():
    r = FakeRobot(grips=[0.79] * 10)
    n = 0

    def relocate():
        nonlocal n
        n += 1
        return (0.45, 0.0), np.array([0.0, 1.0])
    ok, msg = Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1), relocate=relocate)
    assert not ok and f"{config.PICK_ATTEMPTS} times" in msg
    assert n == config.PICK_ATTEMPTS - 1 and r.calls.count("slow line") == config.PICK_ATTEMPTS
    assert r.calls[-2:] == [f"grip{config.GRIPPER_OPEN}", "look"] and r.calls.count("free") == config.PICK_ATTEMPTS


def test_lost_tool_after_a_miss():
    r = FakeRobot(grips=[0.79])
    ok, msg = Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1), relocate=lambda: None)
    assert not ok and "missed the tool 1 time" in msg and r.calls[-1] == "look"


def test_fake_hardware_skips_the_grasp_check():
    r = FakeRobot(tug_after=3.0, grips=[0.8])
    assert Handover(r, log=lambda *a: None, check_grasp=False).run((0.45, 0.0), (0, 1))[0]


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


# ---- the smaller wrist turn ----
def test_pick_yaw_takes_the_smaller_turn_and_flips_the_handover():
    from edgehand.geometry import kortex_R, yaw_of
    h = Handover(FakeRobot(), log=lambda *a: None)
    # the cell's case: from the look pose (yaw 99) the plain grasp would need a 140 deg wrist turn; the flange x axis
    # of tool_down_R(d) points 90 deg past the handle direction, so a handle at 149 deg puts it at -121
    d = (np.cos(np.radians(149)), np.sin(np.radians(149)))
    now = kortex_R(180, 0, 99)
    t = h.plan_targets((0.30, -0.05), d, current_R=now)
    turn = abs((np.degrees(yaw_of(t["grasp"][1])) - 99 + 180) % 360 - 180)
    assert turn < 90                                                     # flipped grasp: ~40 deg instead of ~140
    # the handover orientation is flipped the same way: the flange axis along the tool points AWAY from the operator
    along = t["give"][1][:, 1] if config.FINGERS_ALONG == "x" else t["give"][1][:, 0]
    assert along[:2] @ np.array(config.OPERATOR_DIR[:2]) < -0.99
    # without the current orientation, nothing changes
    t0 = h.plan_targets((0.30, -0.05), d)
    along0 = t0["give"][1][:, 1] if config.FINGERS_ALONG == "x" else t0["give"][1][:, 0]
    assert along0[:2] @ np.array(config.OPERATOR_DIR[:2]) > 0.99


# ---- a tool left in the gripper ----
def test_held_tool_is_handed_over_before_the_next_pick():
    r = FakeRobot()
    r.grip_pos = 0.6                                                     # closed on something, not empty
    h = Handover(r, log=lambda *a: None)
    assert h.holding_something()
    orig = r.wait
    r.wait = lambda s: (orig(s), h.release() if h.holding else None)     # the operator takes each tool at once
    ok, msg = h.run((0.45, 0.0), (0, 1))
    assert ok, msg
    # first the leftover handover (to handover, release, retreat, look), then the normal pick sequence
    assert r.calls[:4] == ["free", f"grip{config.GRIPPER_OPEN}", "line", "look"]
    assert r.calls[4:4 + len(PICK)] == PICK


def test_open_or_empty_gripper_is_not_holding():
    r = FakeRobot()
    h = Handover(r, log=lambda *a: None)
    r.grip_pos = 0.0
    assert not h.holding_something()
    r.grip_pos = 0.79
    assert not h.holding_something()


def test_tool_radius_override_raises_the_grasp():
    h = Handover(FakeRobot(), log=lambda *a: None)
    base = h.plan_targets((0.45, 0.0), (0, 1))["grasp"][0][2]
    fat = h.plan_targets((0.45, 0.0), (0, 1), tool_radius=config.TOOL_RADIUS_M + 0.005)["grasp"][0][2]
    assert np.isclose(fat - base, 0.005)


def test_gripper_preopens_just_wider_than_the_tool():
    from edgehand.handover import open_for_width
    assert open_for_width(None) == config.GRIPPER_OPEN                   # unknown width: fully open
    pos = open_for_width(0.014)                                           # a 14 mm precision handle
    opening = config.GRIPPER_STROKE_M * (1 - (pos - config.GRIPPER_OPEN) / (config.GRIPPER_CLOSED - config.GRIPPER_OPEN))
    assert np.isclose(opening, 0.014 + 2 * config.GRIPPER_OPEN_MARGIN_M)  # ~38 mm, not 85
    assert config.GRIPPER_OPEN < pos < config.GRIPPER_EMPTY_POS           # never reads as "closed on nothing"
    assert open_for_width(0.2) == config.GRIPPER_OPEN                     # wider than the stroke: fully open
    r = FakeRobot(tug_after=3.0)
    assert Handover(r, log=lambda *a: None).run((0.45, 0.0), (0, 1), tool_width=0.014)[0]
    assert r.calls[0] == f"grip{pos}"                                     # the pick opens to that, not fully
