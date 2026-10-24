"""run_kortex's request plumbing without a robot or camera: link.py events -> the request queue."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgehand import config                                        # noqa: E402
from edgehand.run_kortex import Requests                           # noqa: E402


class FakeHandover:
    def __init__(self):
        self.stopped = self.released = 0

    def stop(self):
        self.stopped += 1

    def release(self):
        self.released += 1


def test_requests_queue_and_drop_while_busy():
    h, logs = FakeHandover(), []
    r = Requests(h, log=logs.append)
    tool = next(iter(config.TOOLS))
    r.on_event({"type": "request", "tool": tool, "source": "phase", "phase": "clipping"})
    assert r.next() == ("tool", tool)
    r.busy = True
    r.on_event({"type": "request", "tool": tool, "source": "voice"})
    assert r.q.empty() and any("ignored: a handover is running" in m for m in logs)
    r.busy = False
    r.on_event({"type": "request", "tool": "chainsaw", "source": "voice"})
    assert r.q.empty() and any("not a tool class" in m for m in logs)


def test_stop_and_release_act_at_once():
    h = FakeHandover()
    r = Requests(h, log=lambda *a: None)
    r.on_event({"type": "stop", "node": "jetson"})
    r.on_event({"type": "release"})
    assert (h.stopped, h.released) == (1, 1) and r.q.empty()


def test_typed_and_link_requests_share_one_queue():
    r = Requests(FakeHandover(), log=lambda *a: None)
    r.typed("hand me the green one")
    r.on_event({"type": "request", "tool": next(iter(config.TOOLS))})
    assert r.next()[0] == "text" and r.next()[0] == "tool"
