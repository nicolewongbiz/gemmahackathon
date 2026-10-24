"""Vision regression on real labelled camera frames (EdgeHand/data/frames, made with `python -m edgehand.label`).

Per frame: TableMap + find_tools + colour_class against the labels. Every labelled tool must be found, with the grasp
point within 10 mm and the handle direction right (dot > 0.9). Colour class and stray detections inside the tray are
reported. EDGEHAND_TEST_GEMMA=1 also runs the Gemma namer (needs the model; slow) and checks its names.
    python -m pytest -s test/test_real_frames.py
"""
import json, os, sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgehand import calibration, config                           # noqa: E402
from edgehand.vision import TableMap, find_tools                   # noqa: E402

FRAMES = Path(os.environ.get("EDGEHAND_FRAMES", config.ROOT / "data" / "frames"))
LABELLED = sorted(p for p in FRAMES.glob("*.jpg") if p.with_suffix(".json").exists()) if FRAMES.exists() else []
MATCH_PX = 60          # a detection belongs to a label if its grasp point is this close (pixels)
MAX_GRASP_MM, MIN_DOT = 10.0, 0.9
ROWS = []
_namer = None
# real frames need the real cell's tag positions: data/calibration.json (or $EDGEHAND_REAL_CALIBRATION), which the
# other tests deliberately ignore (conftest.py)
SITE = Path(os.environ.get("EDGEHAND_REAL_CALIBRATION", config.ROOT / "data" / "calibration.json"))


@pytest.fixture(autouse=True)
def site_calibration(monkeypatch):
    if SITE.exists():
        for key, v in calibration.values(json.loads(SITE.read_text())).items():
            monkeypatch.setattr(config, key, v)


def namer():
    global _namer
    if _namer is None:
        from edgehand.namer import GemmaNamer
        _namer = GemmaNamer()
    return _namer


def check_frame(path):
    lab = json.loads(path.with_suffix(".json").read_text())
    img = cv2.imread(str(path))
    tm = TableMap()
    row = {"frame": path.name, "note": lab.get("note", ""), "labelled": len(lab["tools"]), "problems": []}
    if not tm.update(img):
        row["problems"].append("not calibrated (tags)")
        return row
    found = find_tools(img, tm)
    names = namer().name(img, found) if os.environ.get("EDGEHAND_TEST_GEMMA") == "1" else None
    used, errs, colour_ok, gemma_ok = set(), [], 0, 0
    for t in lab["tools"]:
        g = np.array(t["grasp_px"], float)
        cand = [(np.linalg.norm(f["grasp_px"] - g), i) for i, f in enumerate(found) if i not in used]
        if not cand or min(cand)[0] > MATCH_PX:
            row["problems"].append(f"{t['tool']} not found")
            continue
        _, i = min(cand)
        used.add(i)
        f = found[i]
        err = 1000 * float(np.linalg.norm(tm.px_to_table(*g) - f["grasp_xy"]))
        d = tm.px_to_table(*t["handle_px"]) - tm.px_to_table(*g)
        dot = float(d / np.linalg.norm(d) @ f["handle_dir"])
        errs.append(err)
        colour_ok += f["colour"] == t["tool"]
        if names is not None:
            gemma_ok += names[i] == t["tool"]
        if err > MAX_GRASP_MM:
            row["problems"].append(f"{t['tool']} grasp off by {err:.0f} mm")
        if dot < MIN_DOT:
            row["problems"].append(f"{t['tool']} handle direction off (dot {dot:.2f})")
    extra = [i for i in range(len(found)) if i not in used]
    row.update(found=len(found) - len(extra), extra=len(extra), colour_ok=colour_ok,
               worst_mm=max(errs) if errs else 0.0, gemma_ok=gemma_ok if names is not None else None)
    if names is not None:
        if gemma_ok < row["found"]:
            row["problems"].append(f"Gemma named {gemma_ok}/{row['found']}")
        if any(names[i] is not None for i in extra):
            row["problems"].append("Gemma named a stray detection as a tool")
    return row


@pytest.mark.skipif(not LABELLED, reason=f"no labelled frames in {FRAMES} (python -m edgehand.label --capture)")
@pytest.mark.parametrize("path", LABELLED, ids=[p.stem for p in LABELLED])
def test_frame(path):
    row = check_frame(path)
    ROWS.append(row)
    assert not row["problems"], f"{path.name}: {'; '.join(row['problems'])}"


def test_summary():
    if not ROWS:
        pytest.skip("no labelled frames checked")
    print(f"\n{'frame':32} {'note':20} found  extra  colour  worst mm  gemma  problems")
    for r in ROWS:
        print(f"{r['frame']:32} {r['note'][:20]:20} {r.get('found', 0)}/{r['labelled']:<4} {r.get('extra', 0):<6} "
              f"{r.get('colour_ok', 0)}/{r.get('found', 0):<5} {r.get('worst_mm', 0):8.1f}  "
              f"{'-' if r.get('gemma_ok') is None else r['gemma_ok']:<5}  {'; '.join(r['problems'])}")
