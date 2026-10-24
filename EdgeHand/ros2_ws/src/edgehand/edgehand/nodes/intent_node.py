"""/edgehand/command (operator text) -> Gemma 4 -> /edgehand/tool_request + /edgehand/say."""
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from edgehand.intent import GemmaIntent


class IntentNode(Node):
    def __init__(self):
        super().__init__("intent_node")
        self.gemma = GemmaIntent()
        self.busy = threading.Lock()
        self.req = self.create_publisher(String, "/edgehand/tool_request", 10)
        self.say = self.create_publisher(String, "/edgehand/say", 10)
        self.create_subscription(String, "/edgehand/command", self.on_command, 10)
        self.get_logger().info("Gemma ready")

    def on_command(self, msg):
        if not self.busy.acquire(blocking=False):
            self.get_logger().warn("still thinking about the last request")
            return
        threading.Thread(target=self._parse, args=(msg.data,), daemon=True).start()

    def _parse(self, text):
        try:
            r = self.gemma.parse(text)
            self.get_logger().info(f"{text!r} -> {r}")
            if r["reply"]:
                self.say.publish(String(data=r["reply"]))
            if r["tool"]:
                self.req.publish(String(data=r["tool"]))
        finally:
            self.busy.release()


def main():
    rclpy.init()
    rclpy.spin(IntentNode())
