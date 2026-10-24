"""edgehand/phase_player.py timeline handling and edgehand/link.py messages (no window, no robot).

The last test replays the real clip (~/Desktop/clippingdemo.mp4, skipped if it isn't there) at 8x without a window and
listens to the requests on a spare UDP port: about 9 s.    python -m pytest test/test_phase.py -q
"""
import os, socket, subprocess, sys, time
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgehand import config, link                                   # noqa: E402
from edgehand.phase_player import changes, load_segments, phase_at  # noqa: E402

PKG = Path(__file__).resolve().parents[1]                           # ros2_ws/src/edgehand
ANNOTATION = PKG.parents[2] / "data" / "phases" / "clippingdemo.csv"  # EdgeHand/data/phases/clippingdemo.csv
CLIP = Path.home() / "Desktop" / "clippingdemo.mp4"
EXPECTED = [(0, "Parent vessel exposure", "phillips"), (6, "Clipping", "torx"), (18, "Dome exposure", "flathead"),
            (54, "Clipping", "torx"), (58, "Dome exposure", "flathead")]


def spare_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write(tmp_path, rows):
    p = tmp_path / "phases.csv"
    p.write_text("time_s,phase\n" + "\n".join(f"{t},{ph}" for t, ph in rows) + "\n")
    return p


def test_segments_merge_repeats_and_end_at_the_clip(tmp_path):
    p = write(tmp_path, [(0, "Dissection"), (1, "Dissection"), (2, "Dissection"), (20, "Clipping"), (21, "Clipping"),
                         (50, "Inspection")])
    segs = load_segments(p, duration_s=67.1)
    assert segs == [(0, 20, "Dissection"), (20, 50, "Clipping"), (50, 67.1, "Inspection")]
    assert phase_at(segs, 0) == "Dissection" and phase_at(segs, 49.9) == "Clipping" and phase_at(segs, 67.2) is None


def test_requests_only_for_mapped_phases(tmp_path):
    segs = load_segments(write(tmp_path, [(0, "Dissection"), (20, "Clipping"), (50, "Inspection")]), 67)
    assert changes(segs, {"Clipping": "flathead", "Dissection": "torx"}) == [(0, "Dissection", "torx"),
                                                                            (20, "Clipping", "flathead")]


def test_clippingdemo_annotation_fires_on_changes_only():
    """The hand annotation of the demo clip: 7 rows, 5 segments (6-18 s is one Clipping), one request per change."""
    segs = load_segments(ANNOTATION, duration_s=67.4)
    assert [(s, ph) for s, _, ph in segs] == [(0, "Parent vessel exposure"), (6, "Clipping"), (18, "Dome exposure"),
                                              (54, "Clipping"), (58, "Dome exposure")]
    assert segs[-1][1] == 67.4
    assert set(config.PHASE_TOOLS.values()) <= set(config.TOOLS)
    assert changes(segs, config.PHASE_TOOLS) == EXPECTED


@pytest.mark.skipif(not CLIP.exists(), reason=f"demo clip not at {CLIP}")
def test_player_sends_the_requests_on_time():
    """End to end without the arm: the player replays the real clip at 8x with no window, a link.Listener stands in
    for run_kortex --listen. Order, tool, phase and spacing (in clip seconds, from the first request) must match."""
    port, got = spare_port(), []
    link.Listener(lambda m: got.append((time.time(), m)), port=port)
    speed = 8
    p = subprocess.run([sys.executable, "-m", "edgehand.phase_player", str(CLIP), str(ANNOTATION), "--no-window",
                        "--speed", str(speed), "--port", str(port)], cwd=PKG, capture_output=True, text=True,
                       timeout=90, env=dict(os.environ, EDGEHAND_CALIBRATION="none"))
    assert p.returncode == 0, p.stderr
    time.sleep(0.2)
    assert [(m["tool"], m["phase"]) for _, m in got] == [(tool, ph) for _, ph, tool in EXPECTED], p.stdout
    assert all(m["type"] == "request" and m["source"] == "phase" for _, m in got)
    t0 = got[0][0]
    measured = [(ta - t0) * speed for ta, _ in got]
    assert all(abs(m - want) <= 0.5 for m, (want, _, _) in zip(measured, EXPECTED)), measured


def test_link_round_trip_with_heartbeats():
    port = spare_port()
    got = []
    listener = link.Listener(got.append, port=port)
    beat = link.Heartbeat("jetson", port=port)
    link.send({"type": "request", "tool": "torx", "source": "phase", "phase": "Clipping"}, port=port)
    t0 = time.time()
    while (not got or "jetson" not in listener.alive()) and time.time() - t0 < 3:
        time.sleep(0.05)
    beat.close()
    assert got and got[0]["type"] == "request" and got[0]["tool"] == "torx"
    assert all(m["type"] != "heartbeat" for m in got)                   # heartbeats only update alive()
    assert "jetson" in listener.alive()


def test_voice_parse_whitelists():
    from edgehand.voice_node import parse
    assert parse({"intent": "request", "tool": "Phillips", "heard": "x"})["tool"] == "phillips"
    assert parse({"intent": "request", "tool": "hammer"})["intent"] == "none"      # not on the tray: no request
    assert parse({"intent": "stop"})["intent"] == "stop"
    assert parse({"intent": "fly"})["intent"] == "none"
    assert parse(None)["intent"] == "none"
