"""Per-site calibration: what `python -m edgehand.measure_tags` measured on the cell, kept out of config.py.

EdgeHand/data/calibration.json (or $EDGEHAND_CALIBRATION) overrides these config values when it has them:
    TABLE_TAGS, TABLE_TOUCH_Z, LOOK_FLANGE_XYZ, LOOK_THETA, GRIPPER_EMPTY_POS
so moving the cell (e.g. to the venue) means re-running measure_tags, not editing code. EDGEHAND_CALIBRATION=none
turns it off: the tests and the MuJoCo sims do that, because their scenes have their own geometry.
This module doesn't import config (config imports it).
"""
import json, os, shutil, time
from pathlib import Path

KEYS = ("TABLE_TAGS", "TABLE_TOUCH_Z", "LOOK_FLANGE_XYZ", "LOOK_THETA", "GRIPPER_EMPTY_POS")


def path(root):
    p = os.environ.get("EDGEHAND_CALIBRATION")
    if p is not None and p.strip().lower() in ("", "none", "off"):
        return None
    return Path(p) if p else Path(root) / "data" / "calibration.json"


def load(root):
    p = path(root)
    return (json.loads(p.read_text()), p) if p and p.exists() else ({}, None)


def values(data):
    """The KEYS in a calibration file's dict, as config types (int tag ids, tuples)."""
    out = {}
    for key in KEYS:
        if key not in data:
            continue
        v = data[key]
        if key in ("TABLE_TAGS", "TABLE_TOUCH_Z"):
            v = {int(k): tuple(x) if isinstance(x, list) else x for k, x in v.items()}
        elif isinstance(v, list):
            v = tuple(v)
        out[key] = v
    return out


def apply(ns):
    """Override the config module's globals (ns) from the calibration file, if there is one."""
    data, p = load(ns["ROOT"])
    ns.update(values(data))
    ns["CALIBRATION_FILE"] = p                       # None = config.py defaults


def save(root, updates, site=""):
    """Merge updates into the calibration file, keeping a timestamped copy of the previous one. Returns the path."""
    p = path(root) or Path(root) / "data" / "calibration.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    if p.exists():
        shutil.copy(p, p.with_name(f"calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    for key, v in updates.items():
        data[key] = {str(k): list(x) if isinstance(x, tuple) else x for k, x in v.items()} if isinstance(v, dict) else v
    data["measured"] = time.strftime("%Y-%m-%d %H:%M")
    if site:
        data["site"] = site
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    return p
