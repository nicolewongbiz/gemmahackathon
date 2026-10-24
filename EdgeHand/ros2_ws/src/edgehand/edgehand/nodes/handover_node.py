"""Workstation: /edgehand/tool_request -> move to the look pose (wrist camera over the tray) -> ask vision where the tool is
(/edgehand/locate -> /edgehand/locate_result, matched by id) -> pick + hand over through MoveIt.

/edgehand/stop cancels the motion (the controller holds position); /edgehand/release opens the gripper at the
handover pose. Requests are refused while busy, or while stops are arriving (Pi watchdog).
"""
import itertools, json, threading, time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty, String
from edgehand import config
from edgehand.handover import Handover
from edgehand.moveit_client import MoveItRobot


class HandoverNode(Node):
    def __init__(self, robot=None):
        super().__init__("handover_node")
        # Gemma names the tools on every locate: ~5 s on a laptop GPU, more on the Jetson
        self.locate_timeout = self.declare_parameter("locate_timeout", 30.0).value
        self.robot = robot or MoveItRobot(self)
        self.state_pub = self.create_publisher(String, "/edgehand/state", 10)
        self.say_pub = self.create_publisher(String, "/edgehand/say", 10)
        self.handover = Handover(self.robot, log=self.get_logger().info, state=self.publish_state)
        self.busy = threading.Lock()
        self.results = {}
        self.ids = itertools.count(1)
        self.last_stop = 0.0
        self.create_subscription(String, "/edgehand/tool_request", self.on_request, 10)
        self.locate_pub = self.create_publisher(String, "/edgehand/locate", 10)
        self.create_subscription(String, "/edgehand/locate_result", self.on_result, 10)
        self.create_subscription(Empty, "/edgehand/stop", self.on_stop, 10)
        self.create_subscription(Empty, "/edgehand/release", lambda _: self.handover.release(), 10)
        self.create_timer(0.5, self.publish_tug)
        self.publish_state("idle")

    def publish_state(self, s):
        self.state_pub.publish(String(data=s))
        self.get_logger().info(f"state: {s}")

    def publish_tug(self):
        d = self.handover.detector
        if self.handover.holding:
            self.state_pub.publish(String(data=f"holding: delta torque {d.last_delta:.2f} / {config.TUG_TORQUE_NM} N*m"))

    def on_result(self, msg):
        r = json.loads(msg.data)
        self.results[r.get("id")] = r

    def on_stop(self, _):
        self.last_stop = time.time()
        if self.busy.locked():
            self.handover.stop()

    def on_request(self, msg):
        if msg.data not in config.TOOLS:
            return
        if time.time() - self.last_stop < 1.0:
            self.say_pub.publish(String(data="Stop is active."))
            return
        if not self.busy.acquire(blocking=False):
            self.say_pub.publish(String(data="I'm busy."))
            return
        threading.Thread(target=self._run, args=(msg.data,), daemon=True).start()

    def _run(self, tool):
        t0 = time.time()
        stopped = lambda: self.last_stop > t0           # Handover.run only sees stops once it has started
        try:
            self.publish_state("moving to the look pose")
            there = self.robot.look()
            if stopped():
                self.publish_state("stopped")
                return
            if not there:
                self.say_pub.publish(String(data="I can't move to the look pose."))
                self.publish_state("error")
                return
            time.sleep(config.LOOK_SETTLE_S)            # let the wrist settle so the frame isn't blurred
            self.publish_state(f"looking for the {tool}")
            r = self._locate(tool)
            if stopped():
                self.publish_state("stopped")
                return
            if r is None or not r["found"]:
                seen = [s for s in (r or {}).get("seen", []) if s != "unknown"]
                why = (r or {}).get("note") or (f"I can see {' and '.join(seen)}." if seen else "")
                self.say_pub.publish(String(data=f"I can't see the {tool}. {why}".strip()))
                self.publish_state("idle")
                return
            grasp, handle_dir = r["grasp_xy"], np.array(r["handle_dir"])
            self.get_logger().info(f"{tool}: grasp ({grasp[0]:.3f}, {grasp[1]:.3f}), handle towards {np.round(handle_dir, 2)}")
            ok, msg = self.handover.run(grasp, handle_dir)
            self.say_pub.publish(String(data=f"Here's the {tool}." if ok else f"Handover failed: {msg}"))
            self.get_logger().info(f"handover: {ok} {msg}")
        finally:
            self.busy.release()

    def _locate(self, tool):
        """Ask vision for the tool; its answer comes back with the same id (no clock comparisons across machines)."""
        rid = next(self.ids)
        self.locate_pub.publish(String(data=json.dumps({"id": rid, "tool": tool})))
        deadline = time.time() + self.locate_timeout
        while time.time() < deadline:
            if rid in self.results:
                return self.results.pop(rid)
            time.sleep(0.05)
        return None


def main():
    rclpy.init()
    node = HandoverNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)

    def setup():   # service calls need the executor spinning, so this runs beside it
        if not node.robot.wait_for_servers(30.0):
            node.get_logger().error("MoveIt / gripper servers not available; is the MoveIt launch running?")
        elif not node.robot.add_scene():
            node.get_logger().error("couldn't add the table / tray to the planning scene: free moves may hit them")
        else:
            node.get_logger().info("ready")
    threading.Thread(target=setup, daemon=True).start()
    ex.spin()
