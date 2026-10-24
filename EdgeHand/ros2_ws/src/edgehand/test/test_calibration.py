"""data/calibration.json -> config overrides (edgehand/calibration.py)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgehand import calibration                                    # noqa: E402


def test_save_then_apply_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGEHAND_CALIBRATION", str(tmp_path / "data" / "calibration.json"))
    calibration.save(tmp_path, {"TABLE_TAGS": {0: (0.31, -0.15), 1: (0.31, 0.15)}, "TABLE_TOUCH_Z": {0: 0.172, 1: 0.174}})
    calibration.save(tmp_path, {"LOOK_FLANGE_XYZ": [0.25, -0.04, 0.57]}, site="venue")      # merges, keeps the tags
    ns = {"ROOT": tmp_path, "TABLE_TAGS": {9: (0, 0)}, "LOOK_THETA": (180.0, 0.0, 90.0)}
    calibration.apply(ns)
    assert ns["TABLE_TAGS"] == {0: (0.31, -0.15), 1: (0.31, 0.15)}       # int ids, tuples
    assert ns["TABLE_TOUCH_Z"] == {0: 0.172, 1: 0.174}
    assert ns["LOOK_FLANGE_XYZ"] == (0.25, -0.04, 0.57)
    assert ns["LOOK_THETA"] == (180.0, 0.0, 90.0)                       # not in the file: config default stays
    data = json.loads((tmp_path / "data" / "calibration.json").read_text())
    assert data["site"] == "venue" and "measured" in data
    assert len(list((tmp_path / "data").glob("calibration_*.json"))) == 1   # the first file was backed up


def test_none_means_config_defaults(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "calibration.json").write_text(json.dumps({"TABLE_TAGS": {"0": [1, 1]}}))
    monkeypatch.setenv("EDGEHAND_CALIBRATION", "none")
    ns = {"ROOT": tmp_path, "TABLE_TAGS": {0: (0.35, -0.2)}}
    calibration.apply(ns)
    assert ns["TABLE_TAGS"] == {0: (0.35, -0.2)} and ns["CALIBRATION_FILE"] is None
