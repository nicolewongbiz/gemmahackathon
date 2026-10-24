"""Workstation: UDP from the Pi (edgehand_pi.py) -> ROS, plus the heartbeat watchdog.

    request -> /edgehand/tool_request (DEFAULT_TOOL)    release -> /edgehand/release    stop -> /edgehand/stop
No heartbeat for WATCHDOG_S -> /edgehand/stop at 5 Hz until it comes back. require_pi:=false disables the watchdog.
"""
import json, socket, threading, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String
from edgehand import config


class PiBridge(Node):
    def __init__(self):
        super().__init__("pi_bridge")
        self.require_pi = self.declare_parameter("require_pi", True).value
        self.port = self.declare_parameter("port", config.PI_UDP_PORT).value
        self.req = self.create_publisher(String, "/edgehand/tool_request", 10)
        self.rel = self.create_publisher(Empty, "/edgehand/release", 10)
        self.stop = self.create_publisher(Empty, "/edgehand/stop", 10)
        self.last_beat = 0.0
        self.lost = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        threading.Thread(target=self._rx, daemon=True).start()
        self.create_timer(0.2, self._watchdog)
        self.get_logger().info(f"listening on UDP :{self.port}, watchdog {'on' if self.require_pi else 'off'}")

    def _rx(self):
        while rclpy.ok():
            data, _ = self.sock.recvfrom(1024)
            try:
                kind = json.loads(data)["type"]
            except (ValueError, KeyError):
                continue
            if kind == "heartbeat":
                self.last_beat = time.time()
            elif kind == "request":
                self.req.publish(String(data=config.DEFAULT_TOOL))
            elif kind == "release":
                self.rel.publish(Empty())
            elif kind == "stop":
                self.stop.publish(Empty())
            if kind != "heartbeat":
                self.get_logger().info(f"Pi: {kind}")

    def _watchdog(self):
        if not self.require_pi:
            return
        lost = time.time() - self.last_beat > config.WATCHDOG_S
        if lost != self.lost:
            self.get_logger().warn("Pi heartbeat LOST: stopping the arm" if lost else "Pi heartbeat back")
            self.lost = lost
        if lost:
            self.stop.publish(Empty())


def main():
    rclpy.init()
    rclpy.spin(PiBridge())
