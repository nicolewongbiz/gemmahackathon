"""edgehand/phase_player.py timeline handling and edgehand/link.py messages (no window, no robot)."""
import socket, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgehand import link                                           # noqa: E402
from edgehand.phase_player import changes, load_segments, phase_at  # noqa: E402


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


def test_link_round_trip_with_heartbeats():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
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
