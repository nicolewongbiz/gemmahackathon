"""The event link between the nodes: small JSON messages over UDP on the cable, no ROS. Nodes send EVENTS to the
laptop (the only computer that commands the arm), never motions:
    {"type": "request", "tool": "<class>", "source": "phase" | "voice" | "button", ...}   fetch this tool
    {"type": "release"}   open the gripper at the handover pose        {"type": "stop"}   stop the arm
    {"type": "heartbeat", "node": "jetson"}                            "I'm here" (twice a second)
A node that isn't plugged in just isn't heard from: the laptop keeps working on typed requests.
The Jetson's USB-C link gives the laptop 192.168.55.100 itself, so plugging it in needs no setup.
"""
import json, socket, threading, time
from edgehand import config

PORT = config.EVENT_UDP_PORT


def send(msg, host="127.0.0.1", port=PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(json.dumps(dict(msg, t=time.time())).encode(), (host, port))
    s.close()


class Heartbeat:
    """Background 'I'm here' from a node, twice a second."""
    def __init__(self, node, host="127.0.0.1", port=PORT):
        self.node, self.host, self.port, self._run = node, host, port, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            try:
                send({"type": "heartbeat", "node": self.node}, self.host, self.port)
            except OSError:                          # cable not plugged in yet: keep trying
                pass
            time.sleep(0.5)

    def close(self):
        self._run = False


class Listener:
    """On the laptop: calls on_event(msg) for every non-heartbeat message; alive() lists nodes heard recently."""
    def __init__(self, on_event, port=PORT, timeout_s=2.0):
        self.on_event, self.timeout_s = on_event, timeout_s
        self.seen = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            data, addr = self.sock.recvfrom(4096)
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            node = msg.get("node") or msg.get("source") or addr[0]
            self.seen[node] = time.time()
            if msg.get("type") != "heartbeat":
                self.on_event(msg)

    def alive(self):
        now = time.time()
        return sorted(n for n, t in self.seen.items() if now - t < self.timeout_s)
