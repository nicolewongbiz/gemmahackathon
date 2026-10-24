"""Contrast segmentation (config.SEGMENTATION = "contrast"): any handle colour on a plain mat, handle/shaft from the
width profile. Synthetic frames: a white mat with black, clear-ish and coloured handles."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edgehand import config                                        # noqa: E402
from edgehand.vision import TableMap, find_tools, find_tools_contrast   # noqa: E402
import synthetic                                                   # noqa: E402

WHITE = 235
synthetic.COLOURS.update({"black": (20, 20, 20), "clearblue": (170, 120, 90), "green": (40, 120, 40)})


def tm_for(img):
    tm = TableMap()
    assert tm.update(img), "tags not found"
    return tm


@pytest.mark.parametrize("name,g,d", [
    ("black", (0.45, 0.05), (0.0, 1.0)),
    ("clearblue", (0.50, -0.05), (1.0, 0.3)),
    ("green", (0.42, 0.0), (-0.6, -0.8)),
    ("phillips", (0.48, 0.08), (0.3, -1.0)),
])
def test_any_handle_colour_on_a_white_mat(name, g, d, monkeypatch):
    monkeypatch.setattr(config, "SEGMENTATION", "contrast")
    img = synthetic.frame([(name, g, d)], mat=WHITE)
    tools = find_tools(img, tm_for(img))
    assert len(tools) == 1, [t.get("width_m") for t in tools]
    t = tools[0]
    d = np.array(d) / np.linalg.norm(d)
    assert np.linalg.norm(t["grasp_xy"] - g) < 0.006, t["grasp_xy"]     # within 6 mm
    assert t["handle_dir"] @ d > 0.99
    assert 0.018 < t["width_m"] < 0.030                                  # the synthetic handle is 2.4 cm wide
    assert 0.17 < t["length_m"] < 0.23                                   # 10 cm handle + 10 cm shaft


def test_black_handle_is_invisible_to_the_colour_method_but_not_contrast():
    img = synthetic.frame([("black", (0.45, 0.0), (0, 1))], mat=WHITE)
    tm = tm_for(img)
    from edgehand.vision import find_tools_colour
    assert find_tools_colour(img, tm) == []
    assert len(find_tools_contrast(img, tm)) == 1


def test_two_tools_and_the_gripper_at_the_edge():
    # the gripper fingers: a dark block at the frame's bottom-right that touches the image edge
    gripper = [(0.30, -0.30), (0.30, -0.20), (0.22, -0.20), (0.22, -0.30)]
    img = synthetic.frame([("black", (0.45, 0.08), (0, 1)), ("green", (0.50, -0.08), (1, 0))], mat=WHITE,
                          extra=[(gripper, (15, 15, 15))])
    tools = find_tools_contrast(img, tm_for(img))
    assert len(tools) == 2
    assert sorted(round(t["grasp_xy"][1], 2) for t in tools) == [-0.08, 0.08]


def test_round_blob_and_shaftless_thing_are_not_tools():
    # a coin-like disc and a bare rod (no handle): neither has a wide end with a narrow end sticking out
    disc = [(0.45 + 0.02 * np.cos(a), 0.05 + 0.02 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    rod = [(0.50, -0.10), (0.60, -0.10), (0.60, -0.104), (0.50, -0.104)]
    img = synthetic.frame([], mat=WHITE, extra=[(disc, (30, 30, 30)), (rod, (120, 120, 125))])
    assert find_tools_contrast(img, tm_for(img)) == []


def test_contrast_also_works_on_the_dark_mat_with_coloured_handles():
    img = synthetic.frame([("phillips", (0.45, 0.05), (0, 1)), ("torx", (0.50, -0.05), (1, 0))])
    tools = find_tools_contrast(img, tm_for(img))
    assert sorted(t["colour"] for t in tools) == ["phillips", "torx"]
