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


def test_requests_queue_and_keep_only_the_latest_while_busy():
    h, logs = FakeHandover(), []
    r = Requests(h, log=logs.append)
    tools = list(config.TOOLS)
    a, b, c = tools[0], tools[1], tools[2]
    r.on_event({"type": "request", "tool": a, "source": "phase", "phase": "clipping"})
    assert r.next() == ("tool", a)
    r.start(a)                                                       # handing over a; phases keep changing
    r.on_event({"type": "request", "tool": a, "source": "voice"})    # same tool again: nothing to do
    r.on_event({"type": "request", "tool": b, "source": "gemma-phase", "phase": "dome exposure"})
    r.on_event({"type": "request", "tool": c, "source": "gemma-phase", "phase": "clipping"})
    assert r.q.empty() and r.pending == (c, "gemma-phase")           # only the latest is kept
    assert any("replaced by" in m for m in logs)
    r.finish()
    assert r.next() == ("tool", c) and r.pending is None and not r.busy
    r.on_event({"type": "request", "tool": "chainsaw", "source": "voice"})
    assert r.q.empty() and any("not a tool class" in m for m in logs)


def test_pending_request_for_the_tool_just_handed_over_is_dropped():
    r = Requests(FakeHandover(), log=lambda *a: None)
    a, b = list(config.TOOLS)[:2]
    r.start(a)
    r.on_event({"type": "request", "tool": b})
    r.on_event({"type": "request", "tool": a})                       # the latest asks for what is being handed over
    r.finish()
    assert r.q.empty()


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
